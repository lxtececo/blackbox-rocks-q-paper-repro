"""Closed-form Black-Scholes reference for European options.

This module exists so that every Monte Carlo run in this vertical ships
with an independently-computed ground truth. For a European vanilla the
analytic price is exact, which lets the product state - rather than
imply - how far each estimator landed from the true value, and lets the
test suite assert correctness instead of merely asserting "it ran".
"""

from __future__ import annotations

import math
from typing import Any, Dict

from scipy.stats import norm


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def black_scholes_price(
    S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> float:
    """Exact risk-neutral price of a European option under GBM."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    discount = math.exp(-r * T)
    if option_type == "put":
        return float(K * discount * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    return float(S0 * norm.cdf(d1) - K * discount * norm.cdf(d2))


def black_scholes_greeks(
    S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> Dict[str, float]:
    """First- and second-order sensitivities.

    ``vega``/``rho`` are reported per 1 percentage-point move and
    ``theta`` per calendar day, which is how traders read them.
    """
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    discount = math.exp(-r * T)
    pdf_d1 = float(norm.pdf(d1))
    is_put = option_type == "put"

    delta = float(norm.cdf(d1)) - (1.0 if is_put else 0.0)
    gamma = pdf_d1 / (S0 * sigma * math.sqrt(T))
    vega = S0 * pdf_d1 * math.sqrt(T)
    common_theta = -(S0 * pdf_d1 * sigma) / (2 * math.sqrt(T))
    if is_put:
        theta = common_theta + r * K * discount * float(norm.cdf(-d2))
        rho = -K * T * discount * float(norm.cdf(-d2))
    else:
        theta = common_theta - r * K * discount * float(norm.cdf(d2))
        rho = K * T * discount * float(norm.cdf(d2))

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "vega_per_1pct": round(vega / 100.0, 6),
        "theta_per_day": round(theta / 365.0, 6),
        "rho_per_1pct": round(rho / 100.0, 6),
    }


def black_scholes_reference(
    S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> Dict[str, Any]:
    """Reference block attached to every Monte Carlo result."""
    price = black_scholes_price(S0, K, r, sigma, T, option_type)
    moneyness = S0 / K
    return {
        "method": "black_scholes_closed_form",
        "price": round(price, 6),
        "greeks": black_scholes_greeks(S0, K, r, sigma, T, option_type),
        "moneyness": round(moneyness, 4),
        "intrinsic_value": round(
            max(K - S0, 0.0) if option_type == "put" else max(S0 - K, 0.0), 6
        ),
        "is_exact": True,
    }
