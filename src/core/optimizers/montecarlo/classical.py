"""Classical Monte Carlo pricing of European options under GBM.

This is the baseline every amplitude-estimation speedup claim is measured
against, so it is implemented to be genuinely competitive rather than a
strawman: antithetic variates are enabled by default, and the achieved
variance reduction is reported so the comparison cannot quietly benefit
from a weak classical leg.

The convergence trace is derived from the *same* sample that produced the
price (a cumulative mean), not from a second simulation. An earlier
revision re-ran the whole simulation just to draw the chart, which was
both wasteful and free to disagree with the reported price.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

#: Points retained in the convergence trace sent to the UI.
_TRACE_POINTS = 80


def _discounted_payoff(
    terminal: np.ndarray, K: float, r: float, T: float, option_type: str
) -> np.ndarray:
    intrinsic = (
        np.maximum(K - terminal, 0.0)
        if option_type == "put"
        else np.maximum(terminal - K, 0.0)
    )
    return math.exp(-r * T) * intrinsic


def paths_for_target_accuracy(
    payoff_std: float, target_abs_error: float, confidence_level: float = 0.95
) -> int:
    """Paths needed so the CI half-width reaches ``target_abs_error``.

    Inverts the O(1/sqrt(N)) law: N = (z * sigma / epsilon)^2. This is the
    quantity that makes a like-for-like comparison against amplitude
    estimation possible.
    """
    if target_abs_error <= 0 or payoff_std <= 0:
        return 0
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    return int(math.ceil((z * payoff_std / target_abs_error) ** 2))


def _convergence_trace(cumulative_mean: np.ndarray, unit_scale: int) -> Dict[str, List[Any]]:
    """Down-sample the running mean to a chart-sized, log-spaced trace."""
    n = len(cumulative_mean)
    if n <= _TRACE_POINTS:
        idx = np.arange(n)
    else:
        idx = np.unique(
            np.geomspace(1, n, _TRACE_POINTS).astype(int) - 1
        )
    return {
        "samples": [int((i + 1) * unit_scale) for i in idx],
        "estimate": [round(float(cumulative_mean[i]), 6) for i in idx],
    }


def classical_monte_carlo_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Price a European option by Monte Carlo simulation of GBM."""
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)

    if antithetic:
        n_pairs = max(n_paths // 2, 1)
        z = rng.standard_normal(n_pairs)
        up = _discounted_payoff(S0 * np.exp(drift + diffusion * z), K, r, T, option_type)
        down = _discounted_payoff(S0 * np.exp(drift - diffusion * z), K, r, T, option_type)
        # The antithetic pair is the independent sampling unit, so the
        # standard error must be taken across pair averages - not across
        # the 2*n_pairs correlated draws.
        samples = 0.5 * (up + down)
        effective_paths = n_pairs * 2
        naive_std = float(np.concatenate([up, down]).std(ddof=1))
    else:
        z = rng.standard_normal(n_paths)
        samples = _discounted_payoff(
            S0 * np.exp(drift + diffusion * z), K, r, T, option_type
        )
        effective_paths = n_paths
        naive_std = float(samples.std(ddof=1))

    n_units = len(samples)
    price = float(samples.mean())
    unit_std = float(samples.std(ddof=1)) if n_units > 1 else 0.0
    std_error = unit_std / math.sqrt(n_units) if n_units > 1 else 0.0
    z_score = float(norm.ppf(0.5 + confidence_level / 2.0))
    half_width = z_score * std_error

    cumulative_mean = np.cumsum(samples) / np.arange(1, n_units + 1)
    unit_scale = 2 if antithetic else 1

    # Standard error a plain (non-antithetic) run would have achieved with
    # the same number of simulated paths, to quantify the variance
    # reduction rather than assert it.
    naive_std_error = naive_std / math.sqrt(effective_paths) if effective_paths > 1 else 0.0
    reduction = (naive_std_error / std_error) if std_error > 0 else None

    return {
        "method": "classical_monte_carlo",
        "price": round(price, 6),
        "std_error": round(std_error, 6),
        "ci_low": round(price - half_width, 6),
        "ci_high": round(price + half_width, 6),
        "confidence_level": confidence_level,
        "n_paths": int(effective_paths),
        "antithetic": bool(antithetic),
        "payoff_std": round(unit_std, 6),
        "variance_reduction_factor": round(reduction, 3) if reduction else None,
        "convergence": _convergence_trace(cumulative_mean, unit_scale),
        "duration_seconds": round(time.monotonic() - started, 4),
    }
