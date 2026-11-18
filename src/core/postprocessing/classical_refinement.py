"""Universal post-quantum classical QP refinement.

The quantum step (QAOA / VQE / D-Wave Ising / CVaR-VQE / IonQ BF-DCQO)
decides *which* k assets to hold; a small classical mean-variance QP
on the selected subset then decides *how much* of each. This is much
better than the 1/k baseline that most quantum executors default to
when the quantum output is a binary selection vector.

Extracted from the ``CVaR-VQE._classical_reweight_selected`` method
into a shared helper so QAOA / Cirq / IonQ / D-Wave executors can
opt in with a single call. Behaviour and telemetry match the
original CVaR-VQE implementation so the AI reviewer and the
downstream `execution_steps` writer keep the same schema.

Objective on the k selected assets:

.. math::
    \\min_w \\; \\lambda\\, w^\\top \\Sigma_k w - \\mu_k^\\top w
    \\quad\\text{s.t.}\\quad \\mathbf{1}^\\top w = 1, \\; w_i \\ge 0

Any failure (empty selection, SLSQP non-convergence, degenerate
solution) leaves the raw quantum weights untouched and reports the
reason in the returned telemetry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

logger = logging.getLogger(__name__)


def refine_selected_weights(
    symbols: List[str],
    portfolio_weights: Dict[str, float],
    mu_annual: np.ndarray,
    sigma_annual: np.ndarray,
    *,
    risk_aversion: float = 0.5,
    max_weight: Optional[float] = None,
    max_hhi: Optional[float] = None,
    tag: str = "POST-QP",
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    """Solve a long-only Markowitz QP on the currently-selected assets.

    Parameters
    ----------
    symbols:
        Ordered list of every candidate asset - same order used to
        construct ``mu_annual`` and ``sigma_annual``.
    portfolio_weights:
        Current allocation ``{symbol: weight}``. Any symbol with
        ``weight > 0`` is treated as *selected* by the quantum step.
    mu_annual:
        Annualised expected-return vector, aligned with ``symbols``.
    sigma_annual:
        Annualised covariance matrix, aligned with ``symbols``.
    risk_aversion:
        The :math:`\\lambda` scalar in the objective. Match the value
        used by the quantum encoder for consistency.
    max_weight, max_hhi:
        Optional concentration limits (per-asset cap, Herfindahl cap).
        Pass the degeneracy guard's thresholds so the refined weights stay
        inside them; otherwise the guard replaces the quantum selection.
        Each is relaxed to the equal-weight value when ``k`` is too small.
    tag:
        Free-form label prefixed onto log lines
        (e.g. ``"QAOA"``, ``"CIRQ-VQE"``, ``"DWAVE-BQM"``).

    Returns
    -------
    tuple
        ``(refined_weights, telemetry)`` where ``refined_weights`` is
        ``None`` when refinement was skipped or failed - callers must
        keep the original ``portfolio_weights`` in that case.
    """
    selected_idx = [
        i for i, s in enumerate(symbols)
        if float(portfolio_weights.get(s, 0.0)) > 0.0
    ]
    k = len(selected_idx)
    info: Dict[str, Any] = {
        "applied": False,
        "selected_count": k,
        "reason": None,
    }
    if k <= 1:
        info["reason"] = "trivial_selection"
        return None, info

    idx = np.asarray(selected_idx, dtype=int)
    mu_k = np.asarray(mu_annual, dtype=float)[idx]
    sigma_k = np.asarray(sigma_annual, dtype=float)[np.ix_(idx, idx)]
    # Symmetrise defensively (LW + RMT + float drift can perturb).
    sigma_k = 0.5 * (sigma_k + sigma_k.T)

    lam = float(risk_aversion)
    w0 = np.full(k, 1.0 / k)

    def obj(w: np.ndarray) -> float:
        return float(lam * (w @ sigma_k @ w) - mu_k @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * lam * (sigma_k @ w) - mu_k

    # Both caps keep a small margin: SLSQP lands on the boundary within its
    # tolerance and the renormalisation below can nudge it past.
    cap = 1.0 if max_weight is None else max(float(max_weight) - 1e-4, 1.0 / k)
    constraints: List[Dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
    ]
    if max_hhi is not None:
        hhi_cap = max(float(max_hhi) - 1e-4, 1.0 / k + 1e-9)
        constraints.append({
            "type": "ineq",
            "fun": lambda w: float(hhi_cap - w @ w),
            "jac": lambda w: -2.0 * w,
        })

    try:
        res = _scipy_minimize(
            obj,
            w0,
            jac=grad,
            method="SLSQP",
            bounds=[(0.0, cap)] * k,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[%s] Classical refinement failed (%s); keeping raw quantum weights",
            tag, exc,
        )
        info["reason"] = f"solver_exception:{type(exc).__name__}"
        return None, info

    if not res.success:
        logger.warning(
            "[%s] Classical refinement did not converge (%s); keeping raw quantum weights",
            tag, res.message,
        )
        info["reason"] = f"not_converged:{res.message}"
        return None, info

    w_opt = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    total = float(w_opt.sum())
    if total <= 1e-12:
        info["reason"] = "degenerate_solution"
        return None, info
    w_opt = w_opt / total

    out = {s: 0.0 for s in symbols}
    for j, i in enumerate(selected_idx):
        out[symbols[i]] = float(w_opt[j])

    baseline_obj = float(lam * (w0 @ sigma_k @ w0) - mu_k @ w0)
    opt_obj = float(res.fun)
    info.update({
        "applied": True,
        "max_weight": float(w_opt.max()),
        "min_weight_nonzero": float(w_opt[w_opt > 0].min()) if np.any(w_opt > 0) else 0.0,
        "herfindahl": float(np.sum(w_opt ** 2)),
        "equal_weight_objective": baseline_obj,
        "reweighted_objective": opt_obj,
        "objective_improvement": baseline_obj - opt_obj,
        "iterations": int(getattr(res, "nit", 0)),
    })
    logger.info(
        "[%s] Classical refinement: k=%d, max=%.3f, HHI=%.3f, dObj=%+.4f",
        tag, k, info["max_weight"], info["herfindahl"], info["objective_improvement"],
    )
    return out, info


__all__ = ["refine_selected_weights"]
