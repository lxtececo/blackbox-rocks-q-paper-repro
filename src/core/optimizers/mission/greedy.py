"""Greedy baseline and fill for mission allocation (F7).

Tasks by decreasing priority, each to the allowed asset with the largest weight."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from core.optimizers.mission.models import TOL, MissionProblem, Pair, Structure, build_structure, sort_pairs


def fill(problem: MissionProblem, pairs: Iterable[Pair], structure: Optional[Structure] = None) -> List[Pair]:
    """Add unassigned tasks to a feasible plan where they fit; never drops."""
    st = structure or build_structure(problem)
    plan = [(str(a), str(t)) for a, t in pairs]
    hours = {t.id: float(t.duration_hours) for t in problem.tasks}
    duty = {a.id: float(a.duty_limit_hours) for a in problem.assets}
    held: Dict[str, List[str]] = {a.id: [] for a in problem.assets}
    used: Dict[str, float] = {a.id: 0.0 for a in problem.assets}
    for asset_id, task_id in plan:
        held[asset_id].append(task_id)
        used[asset_id] += hours[task_id]
    assigned = {task_id for _, task_id in plan}

    # Sorting is stable: equal priorities keep the task order of the problem.
    for task in sorted(problem.tasks, key=lambda t: -t.priority):
        if task.id in assigned:
            continue
        best: Optional[str] = None
        for asset_id in st.assets_of[task.id]:
            # Strictly larger only: equal weights keep the earlier asset.
            if best is not None and st.weight[(asset_id, task.id)] <= st.weight[(best, task.id)]:
                continue
            if used[asset_id] + hours[task.id] > duty[asset_id] + TOL:
                continue
            near = st.conflicts[asset_id][task.id]
            if any(other in near for other in held[asset_id]):
                continue
            best = asset_id
        if best is not None:
            plan.append((best, task.id))
            held[best].append(task.id)
            used[best] += hours[task.id]
            assigned.add(task.id)
    return sort_pairs(problem, plan)


def solve_greedy(problem: MissionProblem, structure: Optional[Structure] = None) -> List[Pair]:
    """The greedy baseline: fill an empty plan."""
    return fill(problem, [], structure)
