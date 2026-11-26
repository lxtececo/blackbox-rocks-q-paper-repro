"""Exact mission allocation as a MILP solved with HiGHS (F7).

Rows: one asset per task, x[a,t] + x[a,u] <= 1 per conflict, duty hours per asset."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from core.optimizers.mission.feasibility import check_plan
from core.optimizers.mission.models import MissionProblem, Pair, build_structure

DEFAULT_TIME_LIMIT_S = 20.0
#: The duty limit binds when dropping it gains more than this.
BINDING_TOLERANCE = 1e-6


class ExactSolveUnavailable(RuntimeError):
    """The solver returned no usable plan."""


def solve_exact(
    problem: MissionProblem,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    enforce_duty_limit: bool = True,
) -> Tuple[List[Pair], Dict[str, Any]]:
    """Solve to proven optimality, or return the best plan found in time."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    st = build_structure(problem)
    n = len(st.pairs)
    if n == 0:
        return [], {
            "objective": 0.0,
            "status": "optimal",
            "proven_optimal": True,
            "solve_seconds": 0.0,
            "binaries": 0,
            "constraints": 0,
        }

    index = {pair: i for i, pair in enumerate(st.pairs)}
    hours = {t.id: float(t.duration_hours) for t in problem.tasks}
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    upper: List[float] = []

    def add_row(entries: List[Tuple[int, float]], bound: float) -> None:
        for col, val in entries:
            rows.append(len(upper))
            cols.append(col)
            vals.append(val)
        upper.append(bound)

    for task in problem.tasks:
        takers = st.assets_of[task.id]
        if takers:
            add_row([(index[(a, task.id)], 1.0) for a in takers], 1.0)
    for asset in problem.assets:
        for t, u in st.conflict_pairs(asset.id):
            add_row([(index[(asset.id, t)], 1.0), (index[(asset.id, u)], 1.0)], 1.0)
    if enforce_duty_limit:
        for asset in problem.assets:
            mine = st.tasks_of[asset.id]
            if mine:
                add_row([(index[(asset.id, t)], hours[t]) for t in mine], float(asset.duty_limit_hours))

    weights = np.array([st.weight[p] for p in st.pairs], dtype=float)
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(upper), n))
    started = time.perf_counter()
    res = milp(
        -weights,
        constraints=LinearConstraint(matrix, np.full(len(upper), -np.inf), np.array(upper)),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 1e-9},
    )
    wall = time.perf_counter() - started
    if res.x is None:
        raise ExactSolveUnavailable(f"no plan found within {time_limit_s:g} s: {res.message}")

    x = np.asarray(res.x)
    pairs = [st.pairs[i] for i in range(n) if x[i] > 0.5]
    found = check_plan(problem, pairs)["violations"]
    if not enforce_duty_limit:
        found = {k: v for k, v in found.items() if k != "duty_limit_exceeded"}
    if any(found.values()):
        raise ExactSolveUnavailable(f"the solver returned a plan that breaks the model: {found}")

    proven = int(res.status) == 0
    diagnostics = {
        "objective": math.fsum(st.weight[p] for p in pairs),
        "status": "optimal" if proven else "time_limit",
        "proven_optimal": proven,
        "solve_seconds": float(wall),
        "binaries": n,
        "constraints": len(upper),
    }
    return pairs, diagnostics


def exact_reference_with_plan(
    problem: MissionProblem, time_limit_s: float = DEFAULT_TIME_LIMIT_S
) -> Tuple[List[Pair], Dict[str, Any]]:
    """The exact plan and the reference block, with the duty limit verdict."""
    pairs, diag = solve_exact(problem, time_limit_s=time_limit_s)
    _, free = solve_exact(problem, time_limit_s=time_limit_s, enforce_duty_limit=False)
    binding = None
    if diag["proven_optimal"] and free["proven_optimal"]:
        binding = bool(free["objective"] - diag["objective"] > BINDING_TOLERANCE)
    reference = {
        "objective": round(diag["objective"], 3),
        "status": diag["status"],
        "proven_optimal": diag["proven_optimal"],
        "solve_seconds": round(diag["solve_seconds"] + free["solve_seconds"], 4),
        "binaries": diag["binaries"],
        "constraints": diag["constraints"],
        "time_limit_s": float(time_limit_s),
        "duty_limit_binding": binding,
        "objective_without_duty_limit": round(free["objective"], 3),
        "status_without_duty_limit": free["status"],
    }
    return pairs, reference


def exact_reference(problem: MissionProblem, time_limit_s: float = DEFAULT_TIME_LIMIT_S) -> Dict[str, Any]:
    """The block for result["reference"]["exact"]."""
    return exact_reference_with_plan(problem, time_limit_s)[1]
