"""Discretization of the risk-neutral terminal-price distribution onto an
n-qubit register.

The quantum estimator can only ever be as accurate as the grid it loads.
Keeping that grid in its own module lets the service report the two error
sources separately:

* **Discretization bias** - deterministic, caused by binning a continuous
  log-normal into ``2**n_qubits`` buckets. It does not shrink with more
  shots or deeper Grover powers, only with more qubits. It is computed
  exactly here, so the product can state it rather than let it masquerade
  as estimator noise.
* **Estimation error** - statistical, from finite sampling. This is the
  part amplitude estimation actually accelerates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.stats import norm

#: Grid half-width in standard deviations of the log-return. +/-5 sigma
#: leaves <6e-7 of the probability mass untruncated, which is an order of
#: magnitude below the binning error at any qubit count we support.
GRID_SIGMA_WIDTH = 5.0


@dataclass(frozen=True)
class PriceGrid:
    """A discretized terminal-price distribution and its exact expectation."""

    probabilities: np.ndarray
    terminal_prices: np.ndarray
    discounted_payoffs: np.ndarray
    n_qubits: int

    @property
    def n_bins(self) -> int:
        return int(2**self.n_qubits)

    @property
    def exact_discretized_price(self) -> float:
        """Expectation over the grid - the ceiling on quantum accuracy."""
        return float(np.dot(self.probabilities, self.discounted_payoffs))

    @property
    def payoff_ceiling(self) -> float:
        """Normalisation constant mapping payoffs into [0, 1] for encoding."""
        peak = float(self.discounted_payoffs.max())
        return peak if peak > 0 else 1.0

    @property
    def normalized_payoffs(self) -> np.ndarray:
        return self.discounted_payoffs / self.payoff_ceiling


def build_price_grid(
    S0: float, K: float, r: float, sigma: float, T: float, option_type: str, n_qubits: int
) -> PriceGrid:
    """Bin the log-normal terminal price into ``2**n_qubits`` buckets."""
    mu = (r - 0.5 * sigma**2) * T
    sd = sigma * math.sqrt(T)
    n_bins = 2**n_qubits

    edges = np.linspace(mu - GRID_SIGMA_WIDTH * sd, mu + GRID_SIGMA_WIDTH * sd, n_bins + 1)
    # ``edges`` live in log-return space (mu + sd*z). They must be
    # standardized before norm.cdf, otherwise every bucket probability is
    # evaluated against the wrong distribution.
    z_edges = (edges - mu) / sd
    probabilities = np.diff(norm.cdf(z_edges))
    probabilities = probabilities / probabilities.sum()

    terminal_prices = S0 * np.exp((edges[:-1] + edges[1:]) / 2.0)
    if option_type == "put":
        raw_payoff = np.maximum(K - terminal_prices, 0.0)
    else:
        raw_payoff = np.maximum(terminal_prices - K, 0.0)

    return PriceGrid(
        probabilities=probabilities,
        terminal_prices=terminal_prices,
        discounted_payoffs=math.exp(-r * T) * raw_payoff,
        n_qubits=n_qubits,
    )


def describe_grid(grid: PriceGrid, reference_price: float) -> Dict[str, Any]:
    """Quantify how much accuracy the chosen qubit count costs."""
    bias = grid.exact_discretized_price - reference_price
    return {
        "n_qubits": grid.n_qubits,
        "n_bins": grid.n_bins,
        "grid_sigma_width": GRID_SIGMA_WIDTH,
        "exact_discretized_price": round(grid.exact_discretized_price, 6),
        "discretization_bias": round(bias, 6),
        "discretization_bias_pct": round(
            (bias / reference_price * 100.0) if reference_price else 0.0, 4
        ),
        "note": (
            "Discretization bias is fixed by the qubit count. No number of "
            "shots or Grover iterations can reduce it; only more qubits can."
        ),
    }
