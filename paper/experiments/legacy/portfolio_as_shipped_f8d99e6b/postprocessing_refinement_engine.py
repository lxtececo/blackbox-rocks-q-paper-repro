"""
Generic post-processing refinement engine for portfolio optimisation.

Consolidates two previously-duplicated local-search implementations:

* IonQ BF-DCQO ``_postprocess_allocation`` (gradient cardinality repair +
  first-improvement bit-swap on Ising energy).
* IBM PCE ``_bit_swap_post_processing`` (continuous-weight pairwise swap on
  raw QUBO cost).

The engine is **algorithm- and platform-agnostic** and is designed to be
hooked into ``api/executors.py`` after every optimiser call, before result
normalisation. Every swap is gated by a strict decrease of the energy built
here, so refinement is only as good as that energy: it must be the QUBO the
sampler solved (see ``build_ising_from_portfolio``).

References:
    arXiv:2602.23976 §IV.C "Post-processing"
    Pistoia et al., IonQ "Quantum Computing Meets Wall Street" (2025)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from core.quantum_math import cardinality_penalty_lambda, qubo_pair_to_ising
from core.quantum_math import ising_energy as _shared_ising_energy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QUBO → Ising: the same QUBO as ionq_optimizer._build_qubo, converted by the
# shared authority. Until 2026-11-17 this module carried its own conversion
# with three defects (diagonal assigned over accumulated pair shares, linear
# sign flipped, pair energy halved). Its energy then fell with the asset
# index, and the swap search moved every selection to the first K assets.
# ---------------------------------------------------------------------------

def build_ising_from_portfolio(
    returns: np.ndarray,
    covariance: np.ndarray,
    cardinality_K: int,
    risk_aversion: float = 1.0,
    penalty_lambda: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (h, J) Ising coefficients for a cardinality-constrained portfolio.

    Returns
    -------
    h : (n,) linear Ising coefficients
    J : (n, n) upper-triangular quadratic interaction matrix (zeros on/below diag)
    """
    returns = np.asarray(returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(returns)
    if penalty_lambda is None:
        penalty_lambda = cardinality_penalty_lambda(covariance, returns, n)

    q = np.zeros(n)
    Q = np.zeros((n, n))
    for i in range(n):
        q[i] += -returns[i] * (1.0 - risk_aversion)
        q[i] += risk_aversion * covariance[i, i]
        q[i] += penalty_lambda * (1.0 - 2.0 * cardinality_K)
        for j in range(i + 1, n):
            Q[i, j] += risk_aversion * (covariance[i, j] + covariance[j, i])
            Q[i, j] += 2.0 * penalty_lambda

    h, J, _offset = qubo_pair_to_ising(q, Q)
    return h, J


def ising_energy(bits: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """Evaluate Ising energy E = Σᵢ hᵢ zᵢ + Σ_{i<j} J_ij zᵢ z_j with zᵢ = 1 − 2·bitsᵢ."""
    return _shared_ising_energy(bits, h, J)


# ---------------------------------------------------------------------------
# Phase 1 — cardinality repair (gradient/return-greedy)
# ---------------------------------------------------------------------------

def enforce_cardinality(
    bits: np.ndarray,
    returns: np.ndarray,
    target_k: int,
) -> np.ndarray:
    """Greedy cardinality repair on a binary selection vector.

    * Surplus assets are removed in ascending order of expected return.
    * Missing assets are added in descending order of expected return.

    No-op when the input already satisfies ``|bits| == target_k``.
    """
    out = bits.astype(int).copy()
    selected = list(np.where(out == 1)[0])
    excluded = list(np.where(out == 0)[0])

    while len(selected) > target_k:
        worst = min(selected, key=lambda i: returns[i])
        out[worst] = 0
        excluded.append(worst)
        selected.remove(worst)

    while len(selected) < target_k and excluded:
        best = max(excluded, key=lambda i: returns[i])
        out[best] = 1
        selected.append(best)
        excluded.remove(best)

    return out


# ---------------------------------------------------------------------------
# Phase 2a — first-improvement swap on Ising energy (binary)
# ---------------------------------------------------------------------------

def first_improvement_swap_binary(
    bits: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    *,
    max_rounds: int = 10,
) -> np.ndarray:
    """First-improvement local search on a binary {0,1} vector.

    Swaps one selected for one excluded asset whenever the Ising energy
    strictly decreases. Preserves cardinality by construction.
    """
    out = bits.astype(int).copy()
    current_energy = ising_energy(out, h, J)
    rounds = 0
    improved = True
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in list(np.where(out == 1)[0]):
            for j in list(np.where(out == 0)[0]):
                out[i], out[j] = 0, 1
                new_energy = ising_energy(out, h, J)
                if new_energy < current_energy - 1e-12:
                    current_energy = new_energy
                    improved = True
                    break
                out[i], out[j] = 1, 0
            if improved:
                break
    return out


# ---------------------------------------------------------------------------
# Phase 2b — pairwise swap on continuous weights (QUBO cost)
# ---------------------------------------------------------------------------

def _qubo_cost(weights: np.ndarray, linear: np.ndarray, quadratic: np.ndarray) -> float:
    n = len(weights)
    lin = float(np.dot(linear[: min(n, len(linear))], weights[: min(n, len(linear))]))
    nq = min(n, quadratic.shape[0])
    quad = float(weights[:nq] @ quadratic[:nq, :nq] @ weights[:nq])
    return lin + quad


def first_improvement_swap_continuous(
    weights: List[float] | np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    *,
    max_rounds: int = 10,
) -> List[float]:
    """Pairwise weight-swap local search on a continuous allocation vector.

    Mirrors the original PCE implementation: at each round, for every pair
    (i, j) we try swapping weights[i] ↔ weights[j]; accept the first
    improvement and restart.
    """
    current = list(weights)
    current_cost = _qubo_cost(np.asarray(current), linear, quadratic)
    initial_cost = current_cost
    rounds = 0
    improved = True
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cand_cost = _qubo_cost(np.asarray(cand), linear, quadratic)
                if cand_cost < current_cost - 1e-12:
                    current = cand
                    current_cost = cand_cost
                    improved = True
                    break
            if improved:
                break

    if current_cost < initial_cost:
        logger.debug(
            "[refinement] continuous-swap improved cost %.6f -> %.6f over %d rounds",
            initial_cost, current_cost, rounds,
        )
    return current


# ---------------------------------------------------------------------------
# Public high-level entry points
# ---------------------------------------------------------------------------

def refine_binary_solution(
    bits: np.ndarray,
    returns: np.ndarray,
    covariance: np.ndarray,
    *,
    cardinality_K: Optional[int] = None,
    risk_aversion: float = 1.0,
    max_rounds: int = 10,
    telemetry: Optional[dict] = None,
) -> np.ndarray:
    """Two-phase refinement of a binary asset-selection vector.

    Phase 1: cardinality repair (skipped when ``cardinality_K is None``).
    Phase 2: first-improvement Ising-energy swap.

    When ``telemetry`` is supplied, the dict is populated in-place with
    per-phase diagnostics (flips, target satisfaction, energy delta, rounds).
    Used by Job History step-detail rendering.
    """
    initial = np.asarray(bits, dtype=int).copy()
    out = initial.copy()
    phase1: dict = {"applied": False}
    phase2: dict = {"applied": False}

    if cardinality_K is not None:
        before = out.copy()
        out = enforce_cardinality(out, returns, cardinality_K)
        flips = int(np.sum(before != out))
        phase1 = {
            "applied": True,
            "target_k": int(cardinality_K),
            "selected_before": int(before.sum()),
            "selected_after": int(out.sum()),
            "flips": flips,
            "target_k_satisfied": int(out.sum()) == int(cardinality_K),
        }
        h, J = build_ising_from_portfolio(returns, covariance, cardinality_K, risk_aversion)
        e_before = ising_energy(out, h, J)
        before_swap = out.copy()
        out = first_improvement_swap_binary(out, h, J, max_rounds=max_rounds)
        e_after = ising_energy(out, h, J)
        delta = e_before - e_after  # positive => improved (energy decreased)
        rel = (delta / abs(e_before)) if abs(e_before) > 1e-12 else 0.0
        phase2 = {
            "applied": True,
            "max_rounds": int(max_rounds),
            "initial_ising_energy": round(float(e_before), 6),
            "final_ising_energy": round(float(e_after), 6),
            "energy_delta": round(float(delta), 6),
            "improvement_pct": round(float(rel) * 100.0, 3),
            "swap_count": int(np.sum(before_swap != out) // 2),
            "improved": bool(delta > 1e-12),
        }

    if telemetry is not None:
        telemetry.update({
            "engine": "refine_binary_solution",
            "phase1_cardinality_repair": phase1,
            "phase2_swap_local_search": phase2,
        })
    return out


def refine_continuous_weights(
    weights: List[float] | np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    *,
    max_rounds: int = 10,
    telemetry: Optional[dict] = None,
) -> List[float]:
    """Pairwise-swap refinement of a continuous-weight allocation.

    When ``telemetry`` is supplied, populates initial/final QUBO cost,
    delta, and improvement flag for Job History rendering.
    """
    arr = np.asarray(list(weights), dtype=float)
    initial_cost = _qubo_cost(arr, linear, quadratic)
    refined = first_improvement_swap_continuous(weights, linear, quadratic, max_rounds=max_rounds)
    final_cost = _qubo_cost(np.asarray(refined, dtype=float), linear, quadratic)
    delta = initial_cost - final_cost
    rel = (delta / abs(initial_cost)) if abs(initial_cost) > 1e-12 else 0.0
    if telemetry is not None:
        telemetry.update({
            "engine": "refine_continuous_weights",
            "max_rounds": int(max_rounds),
            "initial_qubo_cost": round(float(initial_cost), 6),
            "final_qubo_cost": round(float(final_cost), 6),
            "cost_delta": round(float(delta), 6),
            "improvement_pct": round(float(rel) * 100.0, 3),
            "improved": bool(delta > 1e-12),
        })
    return refined


# ---------------------------------------------------------------------------
# Skip-list of methods that already handle cardinality natively
# ---------------------------------------------------------------------------

#: Methods whose underlying solver enforces cardinality internally and where
#: the local-search refinement step would be redundant or harmful.
NATIVE_CARDINALITY_METHODS = frozenset({
    "dwave-cqm",   # CQM hard constraints
    "dwave-dqm",   # DQM cardinality variable
})


def should_refine(method: str) -> bool:
    """Return True if the given method should run through the refinement engine."""
    return method not in NATIVE_CARDINALITY_METHODS
