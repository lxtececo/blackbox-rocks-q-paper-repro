"""Cost accounting and feasibility for facility-location solutions (F32).

One authority for "what does this network cost", used by the classical
baseline, the quantum decode path and the comparison block alike. If the
baseline and the QUBO disagreed on cost, the optimality gap would be
meaningless — the same reason F6 shares one distance oracle across solvers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.optimizers.network.models import (
    NetworkProblem,
    NetworkSolution,
    SiteAllocation,
)


def build_solution(
    problem: NetworkProblem,
    assignment: Dict[str, List[str]],
    unassigned: Optional[List[str]] = None,
    *,
    cost_matrix: Optional[np.ndarray] = None,
) -> NetworkSolution:
    """Turn ``{site_id: [demand_id]}`` into a costed :class:`NetworkSolution`."""
    from core.optimizers.network.qubo import build_cost_matrix

    cost = build_cost_matrix(problem) if cost_matrix is None else cost_matrix
    site_by_id = {s.id: (idx, s) for idx, s in enumerate(problem.sites)}
    demand_by_id = {d.id: (idx, d) for idx, d in enumerate(problem.demand_points)}

    allocations: List[SiteAllocation] = []
    total_fixed = 0.0
    total_transport = 0.0

    for site_id, demand_ids in assignment.items():
        if site_id not in site_by_id:
            continue
        s_idx, site = site_by_id[site_id]
        # An opened-but-empty site still costs its fixed price; keeping it in
        # the allocation list makes that visible instead of hiding a wasted
        # facility from the cost breakdown.
        served = 0.0
        transport = 0.0
        for demand_id in demand_ids:
            if demand_id not in demand_by_id:
                continue
            d_idx, dp = demand_by_id[demand_id]
            served += float(dp.demand)
            transport += float(cost[d_idx, s_idx])

        total_fixed += float(site.fixed_cost)
        total_transport += transport
        allocations.append(
            SiteAllocation(
                site_id=site.id,
                site_name=site.name,
                demand_point_ids=list(demand_ids),
                served_demand=served,
                capacity=float(site.capacity),
                fixed_cost=float(site.fixed_cost),
                transport_cost=transport,
            )
        )

    allocations.sort(key=lambda a: a.site_id)
    return NetworkSolution(
        allocations=allocations,
        unassigned=list(unassigned or []),
        fixed_cost=total_fixed,
        transport_cost=total_transport,
    )


def finalise_solution(
    problem: NetworkProblem, solution: NetworkSolution
) -> NetworkSolution:
    """Attach the metrics block Job History and the panel render."""
    served_demand = sum(a.served_demand for a in solution.allocations)
    total_demand = problem.total_demand
    overflow = sum(a.overflow for a in solution.allocations)
    open_capacity = sum(a.capacity for a in solution.allocations)

    utilisations = [a.utilization_pct for a in solution.allocations if a.capacity > 0]

    solution.metrics = {
        "open_site_count": len(solution.allocations),
        "candidate_site_count": len(problem.sites),
        "demand_point_count": len(problem.demand_points),
        "served_demand": round(served_demand, 3),
        "total_demand": round(total_demand, 3),
        "coverage_pct": round(
            100.0 * served_demand / total_demand if total_demand > 0 else 0.0, 2
        ),
        "unassigned_count": len(solution.unassigned),
        "capacity_overflow": round(overflow, 3),
        "open_capacity": round(open_capacity, 3),
        "avg_utilization_pct": round(
            float(np.mean(utilisations)) if utilisations else 0.0, 2
        ),
        "max_utilization_pct": round(
            float(np.max(utilisations)) if utilisations else 0.0, 2
        ),
        "fixed_cost": round(solution.fixed_cost, 3),
        "transport_cost": round(solution.transport_cost, 3),
        "total_cost": solution.total_cost,
        "feasible": len(solution.unassigned) == 0 and overflow <= 1e-6,
    }
    return solution


def compare_solutions(
    candidate: NetworkSolution, baseline: NetworkSolution
) -> Dict[str, Any]:
    """Like-for-like gap of ``candidate`` against the classical ``baseline``.

    Positive ``total_cost_gap_pct`` means the candidate is WORSE. F6 shipped a
    first cut 38-51% worse than its baseline and only caught it because this
    number was in the envelope, so it stays prominent here.
    """
    base_cost = baseline.total_cost
    cand_cost = candidate.total_cost
    gap_pct = (
        100.0 * (cand_cost - base_cost) / base_cost if base_cost > 0 else 0.0
    )

    caveats: List[str] = []
    if candidate.unassigned:
        caveats.append(
            f"{len(candidate.unassigned)} demand point(s) left unassigned - "
            "the one-hot penalty did not bind. Raise assign_penalty or reads."
        )
    cand_overflow = candidate.metrics.get("capacity_overflow", 0.0) or 0.0
    if cand_overflow > 1e-6:
        caveats.append(
            f"Capacity exceeded by {cand_overflow:.1f} demand units. The QUBO has "
            "no capacity term: capacity is handled by a classical repair after "
            "sampling, and this plan still exceeds it. Treat the plan as advisory."
        )
    if gap_pct > 0:
        caveats.append(
            f"Costs {gap_pct:.2f}% more than the classical baseline on the "
            "identical instance."
        )

    return {
        "baseline_total_cost": base_cost,
        "candidate_total_cost": cand_cost,
        "total_cost_gap": round(cand_cost - base_cost, 3),
        "total_cost_gap_pct": round(gap_pct, 3),
        "baseline_open_sites": len(baseline.allocations),
        "candidate_open_sites": len(candidate.allocations),
        "baseline_feasible": bool(baseline.metrics.get("feasible", False)),
        "candidate_feasible": bool(candidate.metrics.get("feasible", False)),
        "caveats": caveats,
    }
