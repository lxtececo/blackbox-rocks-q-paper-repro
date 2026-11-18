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

from core.quantum_math import portfolio_qubo, qubo_pair_to_ising
from core.quantum_math import ising_energy as _shared_ising_energy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The QUBO the sampler solves; every move below is judged by it.
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
    q, Q, _const, _lam = portfolio_qubo(returns, covariance, cardinality_K, risk_aversion, penalty_lambda)
    h, J, _offset = qubo_pair_to_ising(q, Q)
    return h, J


def ising_energy(bits: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """Evaluate Ising energy E = Σᵢ hᵢ zᵢ + Σ_{i<j} J_ij zᵢ z_j with zᵢ = 1 − 2·bitsᵢ."""
    return _shared_ising_energy(bits, h, J)


# ---------------------------------------------------------------------------
# Phase 1 — cardinality repair by the QUBO gradient (arXiv:2602.23976, Alg. 4)
# ---------------------------------------------------------------------------

def repair_cardinality(
    bits: np.ndarray,
    returns: np.ndarray,
    covariance: np.ndarray,
    target_k: int,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """One pass of gradient-based flips that leaves exactly ``target_k`` assets.

    g = 2γCx − (1 − γ)μ is the gradient of the objective at the input (the
    penalty adds the same constant to every entry). Surplus assets are dropped
    in order of decreasing g, missing ones added in order of increasing g.
    No-op when the input already holds ``target_k`` assets.
    """
    out = np.asarray(bits, dtype=int).copy()
    surplus = int(out.sum()) - int(target_k)
    if surplus == 0:
        return out
    cov = np.asarray(covariance, dtype=float)
    gradient = 2.0 * risk_aversion * (0.5 * (cov + cov.T)) @ out - (1.0 - risk_aversion) * np.asarray(returns, dtype=float)
    if surplus > 0:
        held = np.flatnonzero(out == 1)
        out[held[np.argsort(-gradient[held], kind="stable")[:surplus]]] = 0
    else:
        free = np.flatnonzero(out == 0)
        out[free[np.argsort(gradient[free], kind="stable")[:-surplus]]] = 1
    return out


# ---------------------------------------------------------------------------
# Phase 2a — first-improvement swap search at fixed cardinality (binary)
# ---------------------------------------------------------------------------

def swap_local_search(
    bits: np.ndarray,
    q: np.ndarray,
    Q: np.ndarray,
    *,
    max_iterations: int = 100,
    patience: int = 1,
    max_swap_evaluations: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """First-improvement swap search on a pair-form QUBO (arXiv:2602.23976, Alg. 4).

    Each iteration scans the (held, free) pairs in random order and accepts the
    first swap that strictly lowers the cost. The search ends after ``patience``
    consecutive iterations without an accepted swap or after ``max_iterations``.
    With ``max_swap_evaluations=None`` every pair is scanned, so the result is a
    local optimum under swaps; the paper caps the scan at 100 pairs.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    out = np.asarray(bits, dtype=int).copy()
    pair = np.triu(np.asarray(Q, dtype=float), 1)
    pair = pair + pair.T
    field = np.asarray(q, dtype=float) + pair @ out
    accepted = evaluated = idle = iterations = 0
    while iterations < max_iterations and idle < patience:
        iterations += 1
        held = rng.permutation(np.flatnonzero(out == 1))
        free = rng.permutation(np.flatnonzero(out == 0))
        # Cost change of swapping held i for free j, in scan order (i outer, j inner).
        delta = (field[free][None, :] - field[held][:, None] - pair[np.ix_(held, free)]).ravel()
        if max_swap_evaluations is not None:
            delta = delta[:int(max_swap_evaluations)]
        improving = np.flatnonzero(delta < -1e-12)
        if improving.size:
            first = int(improving[0])
            i, j = held[first // free.size], free[first % free.size]
            out[i], out[j] = 0, 1
            field += pair[:, j] - pair[:, i]
            evaluated += first + 1
            accepted += 1
            idle = 0
        else:
            evaluated += int(delta.size)
            idle += 1
    return out, {"iterations": iterations, "swaps_accepted": accepted, "swaps_evaluated": evaluated,
                 "local_optimum": bool(max_swap_evaluations is None and idle >= 1)}


def single_flip_descent(
    bits: np.ndarray,
    h: np.ndarray,
    pair: np.ndarray,
    sweeps: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """Lower a selection by single-bit flips, up to ``sweeps`` passes over the bits in random order.

    ``pair`` is the symmetric Ising coupling matrix (J + Jᵀ). A pass that flips
    nothing ends the search. Returns (Ising energy, bits).
    """
    z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    h = np.asarray(h, dtype=float)
    field = h + pair @ z
    for _ in range(sweeps):
        flipped = False
        for j in rng.permutation(len(z)):
            if -2.0 * z[j] * field[j] < -1e-12:
                field -= 2.0 * z[j] * pair[:, j]
                z[j] = -z[j]
                flipped = True
        if not flipped:
            break
    return float(h @ z + 0.5 * z @ pair @ z), ((1.0 - z) / 2.0).round().astype(int)


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
    max_rounds: int = 100,
    patience: int = 1,
    max_swap_evaluations: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    telemetry: Optional[dict] = None,
) -> np.ndarray:
    """Two-phase refinement of a binary asset-selection vector (arXiv:2602.23976, Alg. 4).

    Phase 1: cardinality repair by the QUBO gradient.
    Phase 2: first-improvement swap search on the QUBO cost.
    Both are skipped when ``cardinality_K is None``.

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
        out = repair_cardinality(out, returns, covariance, cardinality_K, risk_aversion)
        flips = int(np.sum(before != out))
        phase1 = {
            "applied": True,
            "rule": "qubo_gradient",
            "target_k": int(cardinality_K),
            "selected_before": int(before.sum()),
            "selected_after": int(out.sum()),
            "flips": flips,
            "target_k_satisfied": int(out.sum()) == int(cardinality_K),
        }
        q, Q, _const, _lam = portfolio_qubo(returns, covariance, cardinality_K, risk_aversion)
        h, J, _offset = qubo_pair_to_ising(q, Q)
        e_before = ising_energy(out, h, J)
        before_swap = out.copy()
        out, search = swap_local_search(out, q, Q, max_iterations=max_rounds, patience=patience,
                                        max_swap_evaluations=max_swap_evaluations, rng=rng)
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
            **search,
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
