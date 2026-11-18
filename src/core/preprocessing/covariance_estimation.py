"""Covariance estimation helpers with Ledoit-Wolf shrinkage.

Sample covariance is noisy and ill-conditioned when the number of
observations is not much larger than the number of assets - which is
almost always the case in production optimizations (a 90-day price
window with 20-30 assets sits deep in the small-sample regime). The
resulting :math:`\\hat\\Sigma` inflates the tails of the mean-variance
objective, produces near-singular Q matrices for the quantum
encoders, and pushes solvers toward corner solutions.

Ledoit & Wolf (2003) shrink the sample covariance toward a
well-conditioned target (a scaled identity):

.. math::

   \\Sigma_{\\text{LW}} = (1-\\delta) \\hat\\Sigma + \\delta \\mu I

where :math:`\\delta \\in [0,1]` and :math:`\\mu` is the average
diagonal of :math:`\\hat\\Sigma`. The optimal :math:`\\delta` has a
closed form under the Frobenius norm.

This module provides a single entry point,
:func:`ledoit_wolf_covariance`, that returns an annualized covariance
matrix and telemetry suitable for embedding in
``optimization_logs.execution_steps``.

Design notes
------------

* We prefer the ``sklearn.covariance.LedoitWolf`` estimator because it
  is a hard dependency of ``pypfopt`` (already present in
  requirements) and gives us the shrinkage coefficient for free.
* When ``sklearn`` is missing or the estimator blows up on
  degenerate inputs (all-NaN column, single row), we fall back to the
  raw sample covariance and record the reason in the telemetry.
* The helper is intentionally stateless and returns numpy arrays so
  every optimizer - classical MVO, QAOA (Cirq / IBM), Ising (D-Wave),
  IonQ, CVaR-VQE, PCE - can drop it in without adopting pypfopt.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    annualization_factor: float = 252.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the covariance matrix of asset returns with Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns:
        A ``DataFrame`` of asset returns (rows: observations, columns:
        assets). NaN rows are dropped up-front.
    annualization_factor:
        Multiplier applied to the estimator output; ``252`` for daily
        returns is the convention used everywhere else in this
        codebase, ``52`` for weekly, ``12`` for monthly. Pass ``1.0``
        to keep the raw per-period covariance.

    Returns
    -------
    tuple
        ``(sigma, telemetry)`` where ``sigma`` is a ``(n, n)`` numpy
        array with the (annualized) covariance and ``telemetry`` is a
        dict of shape ``{"estimator", "shrinkage", "n_observations",
        "n_assets", "annualization_factor", "fallback_reason"}``
        suitable for embedding in ``optimization_logs.execution_steps``
        so the AI reviewer can surface whether shrinkage was applied.
    """
    cleaned = returns.dropna(how="any") if returns is not None else pd.DataFrame()
    n_obs, n_assets = cleaned.shape if not cleaned.empty else (0, 0)

    telemetry: dict[str, Any] = {
        "estimator": "ledoit_wolf",
        "shrinkage": None,
        "n_observations": int(n_obs),
        "n_assets": int(n_assets),
        "annualization_factor": float(annualization_factor),
        "fallback_reason": None,
    }

    # Degenerate inputs -> return a tiny diagonal so downstream solvers
    # do not divide by zero. We surface the reason so callers can log it.
    if cleaned.empty or n_assets == 0:
        telemetry["estimator"] = "sample_cov"
        telemetry["fallback_reason"] = "empty_or_all_nan_returns"
        sigma = np.eye(max(n_assets, 1)) * 1e-6
        return sigma * float(annualization_factor), telemetry

    if n_obs < 2:
        telemetry["estimator"] = "sample_cov"
        telemetry["fallback_reason"] = "fewer_than_two_observations"
        sigma = np.eye(n_assets) * 1e-6
        return sigma * float(annualization_factor), telemetry

    try:
        # Local import so callers that never construct a covariance do
        # not pay the sklearn import cost at module load.
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf(assume_centered=False)
        # sklearn expects (n_samples, n_features) - matches our
        # DataFrame layout (rows=obs, cols=assets) directly.
        lw.fit(cleaned.values)
        sigma = np.asarray(lw.covariance_, dtype=float)
        telemetry["shrinkage"] = float(lw.shrinkage_)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Ledoit-Wolf shrinkage failed (%s); falling back to sample covariance",
            exc,
        )
        telemetry["estimator"] = "sample_cov"
        telemetry["fallback_reason"] = f"ledoit_wolf_exception:{type(exc).__name__}"
        sigma = np.asarray(cleaned.cov().values, dtype=float)

    return sigma * float(annualization_factor), telemetry


__all__ = ["ledoit_wolf_covariance"]
