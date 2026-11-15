"""Exact capacitated facility location by mixed-integer programming (F38-X).

The heuristic baseline in ``classical.py`` was measured 2.2-3.4% above the
proven optimum on the two larger seeded scenarios, so "matches the baseline"
said nothing about quality. This module solves the same instance exactly with
HiGHS (via SciPy) under a time limit and reports whether optimality was proven.

Formulation (single sourcing, same cost matrix as every other path):
    min  sum_j f_j y_j + sum_ij c_ij x_ij
    s.t. sum_j x_ij = 1            for every demand point i
         x_ij <= y_j               linking
         sum_i d_i x_ij <= cap_j y_j
         sum_j y_j <= m            when max_open_sites is set
         y_j = 1                   for must_open sites
         x, y binary
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.optimizers.network.evaluate import build_solution, finalise_solution
from core.optimizers.network.models import NetworkProblem, NetworkSolution

#: Above this many binaries the MILP is not attempted inside a request.
MAX_EXACT_BINARIES = 5000
DEFAULT_TIME_LIMIT_S = 20.0


class ExactSolveUnavailable(RuntimeError):
    """The instance is too large or the solver returned no feasible point."""


def n_binaries(problem: NetworkProblem) -> int:
    ns, nd = len(problem.sites), len(problem.demand_points)
    return ns + ns * nd


def solve_exact(
    problem: NetworkProblem,
    *,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    cost_matrix: Optional[np.ndarray] = None,
) -> Tuple[NetworkSolution, Dict[str, Any]]:
    """Solve to proven optimality, or return the best plan found in time."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    from core.optimizers.network.qubo import build_cost_matrix

    nb = n_binaries(problem)
    if nb > MAX_EXACT_BINARIES:
        raise ExactSolveUnavailable(
            f"{nb} binaries exceed the {MAX_EXACT_BINARIES}-variable limit for an in-request exact solve"
        )

    c = build_cost_matrix(problem) if cost_matrix is None else cost_matrix
    nd, ns = c.shape
    f = np.array([s.fixed_cost for s in problem.sites], dtype=float)
    cap = np.array([s.capacity for s in problem.sites], dtype=float)
    d = np.array([p.demand for p in problem.demand_points], dtype=float)

    def xi(i: int, j: int) -> int:
        return ns + i * ns + j

    rows = nd + nd * ns + ns + 1
    A = lil_matrix((rows, nb))
    lb = np.full(rows, -np.inf)
    ub = np.zeros(rows)
    r = 0
    for i in range(nd):
        for j in range(ns):
            A[r, xi(i, j)] = 1.0
        lb[r] = ub[r] = 1.0
        r += 1
    for i in range(nd):
        for j in range(ns):
            A[r, xi(i, j)] = 1.0
            A[r, j] = -1.0
            r += 1
    for j in range(ns):
        for i in range(nd):
            A[r, xi(i, j)] = d[i]
        A[r, j] = -cap[j]
        r += 1
    A[r, :ns] = 1.0
    ub[r] = float(problem.max_open_sites) if problem.max_open_sites else float(ns)

    lower = np.zeros(nb)
    for j, site in enumerate(problem.sites):
        if site.must_open:
            lower[j] = 1.0

    started = time.perf_counter()
    res = milp(
        np.concatenate([f, c.reshape(-1)]),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        integrality=np.ones(nb),
        bounds=Bounds(lower, np.ones(nb)),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 1e-9},
    )
    wall = time.perf_counter() - started
    if res.x is None:
        raise ExactSolveUnavailable(f"no feasible plan found within {time_limit_s:.0f}s: {res.message}")

    x = np.asarray(res.x)
    assignment: Dict[str, List[str]] = {}
    for j, site in enumerate(problem.sites):
        if x[j] > 0.5:
            assignment[site.id] = []
    for i, dp in enumerate(problem.demand_points):
        j = int(np.argmax(x[ns + i * ns: ns + (i + 1) * ns]))
        assignment.setdefault(problem.sites[j].id, []).append(dp.id)

    solution = finalise_solution(problem, build_solution(problem, assignment, [], cost_matrix=c))
    proven = int(res.status) == 0
    gap = getattr(res, "mip_gap", None)
    diagnostics = {
        "algorithm": "HiGHS branch-and-cut (exact MILP)",
        "proven_optimal": proven,
        "mip_gap": float(gap) if gap is not None else (0.0 if proven else None),
        "dual_bound": float(getattr(res, "mip_dual_bound", np.nan)) if getattr(res, "mip_dual_bound", None) is not None else None,
        "objective": float(res.fun),
        "status_message": str(res.message),
        "binaries": nb,
        "time_limit_s": float(time_limit_s),
        "elapsed_ms": int(wall * 1000),
    }
    return solution, diagnostics


def exact_reference(
    problem: NetworkProblem,
    *,
    candidate: NetworkSolution,
    baseline: Optional[NetworkSolution] = None,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
) -> Dict[str, Any]:
    """Gap of the run (and the heuristic baseline) to the exact optimum."""
    try:
        exact, diag = solve_exact(problem, time_limit_s=time_limit_s)
    except ExactSolveUnavailable as exc:
        return {"available": False, "reason": str(exc)}

    best = exact.total_cost

    def gap(sol: Optional[NetworkSolution]) -> Optional[float]:
        if sol is None or best <= 0:
            return None
        return round(100.0 * (sol.total_cost - best) / best, 3)

    label = "proven optimum" if diag["proven_optimal"] else "best plan found in the time limit (not proven optimal)"
    caveats = []
    if not diag["proven_optimal"]:
        caveats.append(
            f"The exact solver hit its {time_limit_s:.0f}s limit; gaps are measured against the best plan "
            f"it found, with a remaining optimality gap of {100 * (diag['mip_gap'] or 0):.2f}%."
        )
    return {
        "available": True,
        "reference": label,
        "exact_total_cost": best,
        "exact_open_sites": exact.open_site_ids,
        "proven_optimal": diag["proven_optimal"],
        "mip_gap": diag["mip_gap"],
        "elapsed_ms": diag["elapsed_ms"],
        "candidate_gap_to_exact_pct": gap(candidate),
        "baseline_gap_to_exact_pct": gap(baseline),
        "caveats": caveats,
    }
