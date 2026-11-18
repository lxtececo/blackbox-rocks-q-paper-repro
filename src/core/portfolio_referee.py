"""Exact referee for asset-selection QUBOs.

Every binary portfolio method on the platform minimises a QUBO it builds
itself. Up to ``MAX_ASSETS`` assets the exact optimum of that QUBO is an
enumeration of all 2^n selections, so a returned selection can be placed
among all of them instead of being compared with a classical baseline that
solves a different problem.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

MAX_ASSETS = 20


def enumerate_costs(q: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Cost of every selection of the pair-form QUBO Σ qᵢxᵢ + Σᵢ<ⱼ Q_ij xᵢxⱼ.

    Entry ``k`` is the cost of the selection whose asset i is bit i of ``k``.
    Only the upper triangle of ``Q`` is read. Time and memory grow as 2^n.
    """
    q = np.asarray(q, dtype=float)
    pair = np.triu(np.asarray(Q, dtype=float), 1)
    cost = np.zeros(1)
    for i in range(len(q)):
        coupling = np.zeros(1)
        for j in range(i):
            coupling = np.concatenate([coupling, coupling + pair[j, i]])
        cost = np.concatenate([cost, cost + q[i] + coupling])
    return cost


def enumerate_sizes(n: int) -> np.ndarray:
    """Number of assets held by every selection, in the order of ``enumerate_costs``."""
    sizes = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        sizes = np.concatenate([sizes, sizes + 1])
    return sizes


def referee(
    q: np.ndarray,
    Q: np.ndarray,
    selection: Sequence[int],
    const: float = 0.0,
    cardinality: Optional[int] = None,
    min_assets: int = 0,
    sampled_shots: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Where ``selection`` stands among the admissible selections of the QUBO.

    Admissible are the selections a method may return: those that hold exactly
    ``cardinality`` assets when a target is given, otherwise those that hold at
    least ``min_assets``. Returns None above ``MAX_ASSETS`` assets or when no
    selection is admissible.

    ``sampled_shots``: for a method that returns the cheapest admissible one of
    S sampled selections, the result also states what S uniformly random
    selections give under the same rule (``uniform_null``). A run that only
    matches those numbers says nothing about its sampler.
    """
    n = len(q)
    if n > MAX_ASSETS:
        return None
    costs = enumerate_costs(q, Q)
    sizes = enumerate_sizes(n)
    admissible = sizes == int(cardinality) if cardinality is not None else sizes >= int(min_assets)
    if not admissible.any():
        return None
    index = int(sum(int(round(float(bit))) << i for i, bit in enumerate(selection)))
    pool = costs[admissible]
    lowest, mean = float(pool.min()), float(pool.mean())
    tolerance = 1e-9 * max(1.0, abs(lowest))
    optimal = admissible & (costs <= lowest + tolerance)
    better = int(np.sum(pool < costs[index] - tolerance))
    best = int(np.flatnonzero(optimal)[0])
    out: Dict[str, Any] = {
        "assets": n,
        "cardinality": None if cardinality is None else int(cardinality),
        "selections": int(len(pool)),
        "optimum_cost": lowest + const,
        "returned_cost": float(costs[index]) + const,
        "returned_is_admissible": bool(admissible[index]),
        "is_optimal": bool(optimal[index]),
        "better_selections": better,
        "better_selections_pct": 100.0 * better / len(pool),
        # 0 at the optimum, 1 for the average admissible selection
        "gap": float((costs[index] - lowest) / (mean - lowest)) if mean > lowest else 0.0,
        "optimal_selections": int(optimal.sum()),
        "optimal_selection": [(best >> i) & 1 for i in range(n)],
    }
    if cardinality is not None:
        # False: the penalty in the QUBO is too weak to make a selection with the target size its minimum.
        out["qubo_minimum_holds_target"] = bool(float(costs.min()) >= lowest - tolerance)
    if sampled_shots:
        out["uniform_null"] = _uniform_null(pool, len(costs), int(optimal.sum()), int(sampled_shots), tolerance)
    return out


def _uniform_null(pool: np.ndarray, total: int, optimal: int, shots: int, tolerance: float) -> Dict[str, Any]:
    """The cheapest admissible one of ``shots`` uniformly random selections, exactly.

    With B the number of admissible selections cheaper than the returned one,
    P(B >= t, an admissible sample exists) = (1 - a_t / N)^S - (1 - F / N)^S,
    where a_t counts the admissible selections with fewer than t cheaper ones,
    F the admissible ones and N all 2^n. The expectation is the sum over t.
    """
    cheaper = np.searchsorted(np.sort(pool), pool - tolerance, side="left")
    with_fewer_than = np.cumsum(np.bincount(cheaper, minlength=len(pool)))[:-1]
    none_admissible = (1.0 - len(pool) / total) ** shots
    expected = float(np.sum((1.0 - with_fewer_than / total) ** shots - none_admissible)) / (1.0 - none_admissible)
    return {
        "shots": shots,
        "p_no_admissible_sample": float(none_admissible),
        "p_returns_an_optimum": float(1.0 - (1.0 - optimal / total) ** shots),
        # given that at least one sample is admissible
        "expected_better_selections_pct": 100.0 * expected / len(pool),
    }
