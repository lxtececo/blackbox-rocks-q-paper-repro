"""Classical baseline for facility location (F32).

Greedy open-by-marginal-cost followed by site-swap / add / drop local search.
This is the number every quantum method is scored against, so it is
deliberately a *good* heuristic rather than a strawman - a baseline that is
easy to beat makes the comparison worthless.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.optimizers.network.evaluate import build_solution, finalise_solution
from core.optimizers.network.models import NetworkProblem, NetworkSolution
from core.optimizers.network.qubo import build_cost_matrix


def _overflow_price(problem: NetworkProblem) -> float:
    """Cost charged per demand unit served beyond a site's capacity.

    Without this, greedy happily opens ONE cheap site and overflows it, because
    skipping a second site's fixed cost always looks like a saving when the
    overflow is free. The price is pinned above the worst real cost of buying a
    unit of capacity (priciest site / smallest site) so opening another site is
    always cheaper than overflowing.
    """
    max_fixed = max((float(s.fixed_cost) for s in problem.sites), default=1.0)
    min_cap = min((float(s.capacity) for s in problem.sites if s.capacity > 0), default=1.0)
    return 10.0 * max_fixed / max(min_cap, 1e-9)


def _assign_to_open(
    problem: NetworkProblem,
    open_sites: Sequence[int],
    cost: np.ndarray,
) -> Tuple[Dict[str, List[str]], float]:
    """Assign every demand point to its cheapest open site, respecting capacity.

    Demand points are placed in descending demand order: the largest loads are
    hardest to fit, so committing them first avoids a tail of big customers
    with nowhere left to go.

    The returned cost includes an overflow charge, so callers comparing two
    open-sets are comparing feasibility-aware numbers.
    """
    if not open_sites:
        return {}, float("inf")

    remaining = {j: float(problem.sites[j].capacity) for j in open_sites}
    assignment: Dict[str, List[str]] = {problem.sites[j].id: [] for j in open_sites}
    transport = 0.0
    overflow = 0.0

    order = sorted(
        range(len(problem.demand_points)),
        key=lambda i: -problem.demand_points[i].demand,
    )

    for i in order:
        dp = problem.demand_points[i]
        feasible = [j for j in open_sites if remaining[j] >= dp.demand]
        pool = feasible or list(open_sites)  # overflow rather than drop
        best = min(pool, key=lambda j: cost[i, j])
        assignment[problem.sites[best].id].append(dp.id)
        if not feasible:
            overflow += float(dp.demand) - max(remaining[best], 0.0)
        remaining[best] -= float(dp.demand)
        transport += float(cost[i, best])

    fixed = sum(float(problem.sites[j].fixed_cost) for j in open_sites)
    return assignment, fixed + transport + overflow * _overflow_price(problem)


def _candidate_open_sets(problem: NetworkProblem) -> List[int]:
    """Sites that must be open regardless of cost."""
    return [j for j, s in enumerate(problem.sites) if s.must_open]


def refine_open_set(
    problem: NetworkProblem,
    open_sites: Sequence[int],
    cost: np.ndarray,
    *,
    time_budget_s: float = 10.0,
    started: Optional[float] = None,
) -> Tuple[List[int], float, int, int, List[float]]:
    """Drop / add / swap / merge local search over an existing open set.

    Shared by the classical baseline (seeded from its greedy construction) and
    by the quantum polish (seeded from the sampler's chosen sites), so both
    paths are refined by the *same* operator and a quantum-vs-classical gap
    reflects the site SELECTION rather than a difference in post-processing.

    Merge (two open sites -> one closed site) is what escapes the trap where
    one large site beats two small ones but neither small site can be dropped
    alone without overflowing capacity, and no 1-for-1 swap lowers cost.

    Returns ``(open_sites, cost, drops, swaps, trace)``; ``swaps`` counts every
    move that brings a closed site in (add, swap, merge).
    """
    if started is None:
        started = time.monotonic()
    forced = set(_candidate_open_sets(problem))
    sites = list(open_sites)
    n_sites = len(problem.sites)
    max_open = problem.max_open_sites or n_sites
    _, best_cost = _assign_to_open(problem, sites, cost)
    trace: List[float] = []
    swaps = 0
    drops = 0

    def _accept(trial_sites: List[int]) -> bool:
        nonlocal sites, best_cost
        _, trial = _assign_to_open(problem, trial_sites, cost)
        if trial < best_cost - 1e-9:
            sites = trial_sites
            best_cost = trial
            trace.append(round(best_cost, 3))
            return True
        return False

    improved = True
    while improved and (time.monotonic() - started) < time_budget_s:
        improved = False

        # Drop: is any open site not earning its fixed cost?
        for j in list(sites):
            if j in forced or len(sites) <= 1:
                continue
            if _accept([s for s in sites if s != j]):
                drops += 1
                improved = True
                break
        if improved:
            continue

        closed = [k for k in range(n_sites) if k not in sites]

        # Add: would one more site pay for itself in transport savings?
        if len(sites) < max_open:
            for k in closed:
                if _accept(sites + [k]):
                    swaps += 1
                    improved = True
                    break
        if improved:
            continue

        # Swap: exchange an open site for a closed one.
        for j in list(sites):
            if j in forced:
                continue
            for k in closed:
                if _accept([s for s in sites if s != j] + [k]):
                    swaps += 1
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Merge: replace two open sites with one closed site.
        movable = [j for j in sites if j not in forced]
        for a in range(len(movable)):
            for b in range(a + 1, len(movable)):
                pair = {movable[a], movable[b]}
                base = [s for s in sites if s not in pair]
                for k in closed:
                    if _accept(base + [k]):
                        swaps += 1
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return sites, best_cost, drops, swaps, trace


def solve_classical(
    problem: NetworkProblem,
    *,
    time_budget_s: float = 10.0,
    local_search: bool = True,
    cost_matrix: Optional[np.ndarray] = None,
) -> Tuple[NetworkSolution, Dict[str, Any]]:
    """Greedy construction + local search. Returns ``(solution, diagnostics)``."""
    started = time.monotonic()
    cost = build_cost_matrix(problem) if cost_matrix is None else cost_matrix
    n_sites = len(problem.sites)

    forced = _candidate_open_sets(problem)
    open_sites: List[int] = list(forced)
    trace: List[float] = []

    # --- Greedy: keep opening the site that most reduces total cost --------
    best_cost = float("inf")
    if open_sites:
        _, best_cost = _assign_to_open(problem, open_sites, cost)

    max_open = problem.max_open_sites or n_sites
    demand_total = problem.total_demand

    def _open_capacity(sites: List[int]) -> float:
        return sum(float(problem.sites[j].capacity) for j in sites)

    while len(open_sites) < max_open:
        best_gain = 0.0
        best_site = None
        for j in range(n_sites):
            if j in open_sites:
                continue
            _, trial = _assign_to_open(problem, open_sites + [j], cost)
            if trial < best_cost - 1e-9:
                gain = best_cost - trial
                if gain > best_gain:
                    best_gain = gain
                    best_site = j
        if best_site is None:
            # No single addition improves cost. If capacity still cannot cover
            # demand we are sitting on an infeasible network, so keep opening
            # the cheapest capacity available rather than declaring victory.
            if _open_capacity(open_sites) >= demand_total:
                break
            remaining_sites = [j for j in range(n_sites) if j not in open_sites]
            if not remaining_sites:
                break
            best_site = min(
                remaining_sites,
                key=lambda j: float(problem.sites[j].fixed_cost)
                / max(float(problem.sites[j].capacity), 1e-9),
            )
        open_sites.append(best_site)
        _, best_cost = _assign_to_open(problem, open_sites, cost)
        trace.append(round(best_cost, 3))

    if not open_sites:
        # Every site made things worse on its own, but demand still has to go
        # somewhere - open the single cheapest-to-serve site.
        totals = [
            float(problem.sites[j].fixed_cost) + float(cost[:, j].sum())
            for j in range(n_sites)
        ]
        open_sites = [int(np.argmin(totals))]
        _, best_cost = _assign_to_open(problem, open_sites, cost)
        trace.append(round(best_cost, 3))

    swaps = 0
    drops = 0
    if local_search:
        open_sites, best_cost, drops, swaps, ls_trace = refine_open_set(
            problem,
            open_sites,
            cost,
            time_budget_s=time_budget_s,
            started=started,
        )
        trace.extend(ls_trace)

    assignment, _ = _assign_to_open(problem, open_sites, cost)
    solution = finalise_solution(
        problem, build_solution(problem, assignment, [], cost_matrix=cost)
    )

    diagnostics = {
        "algorithm": "greedy-open + drop/add/swap/merge local search",
        "open_sites": len(open_sites),
        "swaps": swaps,
        "drops": drops,
        "cost_trace": trace,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
    return solution, diagnostics
