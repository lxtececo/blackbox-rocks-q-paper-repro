"""Independent check of a mission allocation plan (F7).

Imports only the model dataclasses and restates every rule from the raw fields."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from core.optimizers.mission.models import Asset, MissionProblem, Task

_TOL = 1e-9


def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def _weight(asset: Asset, task: Task) -> float:
    cost = task.duration_hours * asset.cost_per_hour + 2.0 * _dist(asset.base, task.position) * asset.cost_per_km
    return 1000.0 * task.priority - cost


def _allowed(asset: Asset, task: Task) -> bool:
    leg = _dist(asset.base, task.position) / asset.speed_kmh
    end = task.start_hour + task.duration_hours
    return (
        task.capability in asset.capabilities
        and asset.available_from + leg <= task.start_hour + _TOL
        and end + leg <= asset.available_until + _TOL
        and task.duration_hours <= asset.duty_limit_hours + _TOL
        and _weight(asset, task) > 0.0
    )


def _conflict(asset: Asset, t: Task, u: Task) -> bool:
    hop = _dist(t.position, u.position) / asset.speed_kmh
    first_then_second = t.start_hour + t.duration_hours + hop <= u.start_hour + _TOL
    second_then_first = u.start_hour + u.duration_hours + hop <= t.start_hour + _TOL
    return not first_then_second and not second_then_first


def check_plan(problem: MissionProblem, pairs: Iterable[Sequence[str]]) -> Dict[str, Any]:
    """Count every kind of violation of a plan; feasible when all are zero."""
    assets = {a.id: a for a in problem.assets}
    tasks = {t.id: t for t in problem.tasks}
    not_allowed = 0
    per_task: Dict[str, int] = {}
    per_asset: Dict[str, List[Task]] = {}
    for asset_id, task_id in pairs:
        per_task[task_id] = per_task.get(task_id, 0) + 1
        asset, task = assets.get(asset_id), tasks.get(task_id)
        if asset is None or task is None:
            not_allowed += 1
            continue
        if not _allowed(asset, task):
            not_allowed += 1
        per_asset.setdefault(asset_id, []).append(task)

    conflicts = 0
    duty = 0
    for asset_id, held in per_asset.items():
        asset = assets[asset_id]
        for i, t in enumerate(held):
            for u in held[i + 1:]:
                if t.id != u.id and _conflict(asset, t, u):
                    conflicts += 1
        if math.fsum(t.duration_hours for t in held) > asset.duty_limit_hours + _TOL:
            duty += 1

    violations = {
        "not_allowed": not_allowed,
        "task_multiply_assigned": sum(1 for n in per_task.values() if n > 1),
        "conflicts": conflicts,
        "duty_limit_exceeded": duty,
    }
    return {"feasible": not any(violations.values()), "violations": violations}


def objective(problem: MissionProblem, pairs: Iterable[Sequence[str]]) -> float:
    """Sum of the pair weights of a plan (exact sum, order does not matter)."""
    assets = {a.id: a for a in problem.assets}
    tasks = {t.id: t for t in problem.tasks}
    return math.fsum(_weight(assets[a], tasks[t]) for a, t in pairs)


def schedule_ok(problem: MissionProblem, asset_id: str, task_ids: Iterable[str]) -> bool:
    """Walk the tasks in start order: base, task to task, base, in the window."""
    asset = next(a for a in problem.assets if a.id == asset_id)
    by_id = {t.id: t for t in problem.tasks}
    route = sorted((by_id[t] for t in task_ids), key=lambda t: (t.start_hour, t.start_hour + t.duration_hours))
    clock = asset.available_from
    here = asset.base
    for task in route:
        clock += _dist(here, task.position) / asset.speed_kmh
        if clock > task.start_hour + _TOL:
            return False
        clock = task.start_hour + task.duration_hours
        here = task.position
    clock += _dist(here, asset.base) / asset.speed_kmh
    return clock <= asset.available_until + _TOL
