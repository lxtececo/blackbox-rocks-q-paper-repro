"""Domain model for mission resource allocation (F7).

Hours from scenario start, km on a synthetic area, straight-line distance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

#: Tolerance of every time and hour comparison.
TOL = 1e-9
VALUE_PER_PRIORITY = 1000.0

Point = Tuple[float, float]
Pair = Tuple[str, str]


def _round(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def _point(p: Point) -> Dict[str, float]:
    return {"x_km": _round(p[0]), "y_km": _round(p[1])}


@dataclass
class Asset:
    """A support asset with capabilities, a base and an availability window."""

    id: str
    name: str
    kind: str
    capabilities: List[str]
    base: Point
    available_from: float
    available_until: float
    duty_limit_hours: float
    speed_kmh: float
    cost_per_hour: float
    cost_per_km: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "capabilities": list(self.capabilities),
            "base": _point(self.base),
            "available_from": _round(self.available_from),
            "available_until": _round(self.available_until),
            "duty_limit_hours": _round(self.duty_limit_hours),
            "speed_kmh": _round(self.speed_kmh),
            "cost_per_hour": _round(self.cost_per_hour),
            "cost_per_km": _round(self.cost_per_km),
        }


@dataclass
class Task:
    """A task that needs one capability at a place for a fixed time slot."""

    id: str
    name: str
    capability: str
    position: Point
    start_hour: float
    duration_hours: float
    priority: int

    @property
    def end_hour(self) -> float:
        return self.start_hour + self.duration_hours

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "capability": self.capability,
            "position": _point(self.position),
            "start_hour": _round(self.start_hour),
            "duration_hours": _round(self.duration_hours),
            "end_hour": _round(self.end_hour),
            "priority": int(self.priority),
        }


@dataclass
class MissionProblem:
    """A complete allocation instance: assets, tasks and the scenario texts."""

    assets: List[Asset]
    tasks: List[Task]
    scenario_id: str = "custom"
    name: str = "Custom scenario"
    description: str = ""
    horizon_hours: float = 24.0
    area_km: Point = (300.0, 200.0)

    def __post_init__(self) -> None:
        for label, items in (("asset", self.assets), ("task", self.tasks)):
            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} id in the problem")
        for asset in self.assets:
            if not asset.speed_kmh > 0:
                raise ValueError(f"asset {asset.id} needs a positive speed")
        for task in self.tasks:
            if task.duration_hours < 0:
                raise ValueError(f"task {task.id} has a negative duration")

    def asset(self, asset_id: str) -> Asset:
        return next(a for a in self.assets if a.id == asset_id)

    def task(self, task_id: str) -> Task:
        return next(t for t in self.tasks if t.id == task_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "horizon_hours": _round(self.horizon_hours),
            "area_km": [_round(self.area_km[0]), _round(self.area_km[1])],
            "assets": [a.to_dict() for a in self.assets],
            "tasks": [t.to_dict() for t in self.tasks],
        }


def distance(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def travel_hours(asset: Asset, p: Point, q: Point) -> float:
    return distance(p, q) / asset.speed_kmh


def pair_value(task: Task) -> float:
    return VALUE_PER_PRIORITY * task.priority


def pair_cost(asset: Asset, task: Task) -> float:
    """Task hours at the hourly rate plus a round trip from the base."""
    round_trip = 2.0 * distance(asset.base, task.position)
    return task.duration_hours * asset.cost_per_hour + round_trip * asset.cost_per_km


def pair_weight(asset: Asset, task: Task) -> float:
    return pair_value(task) - pair_cost(asset, task)


def exclusion_reason(asset: Asset, task: Task) -> Optional[str]:
    """The first rule that forbids the pair, in the order checked; None when it is allowed."""
    if task.capability not in asset.capabilities:
        return "capability"
    out = travel_hours(asset, asset.base, task.position)
    if asset.available_from + out > task.start_hour + TOL:
        return "arrives_late"
    if task.end_hour + out > asset.available_until + TOL:
        return "returns_late"
    if task.duration_hours > asset.duty_limit_hours + TOL:
        return "task_exceeds_duty_limit"
    if not pair_weight(asset, task) > 0.0:
        return "not_worth_cost"
    return None


def is_allowed(asset: Asset, task: Task) -> bool:
    """Capability, reachable inside the window, within duty, positive weight."""
    return exclusion_reason(asset, task) is None


def tasks_conflict(asset: Asset, t: Task, u: Task) -> bool:
    """True when the asset cannot do both tasks, in either order."""
    hop = travel_hours(asset, t.position, u.position)
    t_then_u = t.end_hour + hop <= u.start_hour + TOL
    u_then_t = u.end_hour + hop <= t.start_hour + TOL
    return not (t_then_u or u_then_t)


@dataclass
class Structure:
    """Allowed pairs, weights and conflicts derived from a problem."""

    #: Allowed pairs in asset order, then task order: the variable order.
    pairs: List[Pair] = field(default_factory=list)
    weight: Dict[Pair, float] = field(default_factory=dict)
    #: Task id -> ids of the assets allowed to take it, in asset order.
    assets_of: Dict[str, List[str]] = field(default_factory=dict)
    #: Asset id -> ids of the tasks it may take, in task order.
    tasks_of: Dict[str, List[str]] = field(default_factory=dict)
    #: Asset id -> task id -> ids of the tasks in conflict with it.
    conflicts: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)

    def conflict_pairs(self, asset_id: str) -> List[Tuple[str, str]]:
        """Conflicting task pairs of one asset, each once, in task order."""
        tasks = self.tasks_of[asset_id]
        near = self.conflicts[asset_id]
        return [
            (t, u)
            for i, t in enumerate(tasks)
            for u in tasks[i + 1:]
            if u in near[t]
        ]

    @property
    def conflict_count(self) -> int:
        return sum(len(n) for near in self.conflicts.values() for n in near.values()) // 2


def build_structure(problem: MissionProblem) -> Structure:
    """Derive the allowed pairs and the conflicts of every asset."""
    st = Structure()
    st.assets_of = {t.id: [] for t in problem.tasks}
    for asset in problem.assets:
        mine = [t for t in problem.tasks if is_allowed(asset, t)]
        st.tasks_of[asset.id] = [t.id for t in mine]
        near: Dict[str, Set[str]] = {t.id: set() for t in mine}
        for i, t in enumerate(mine):
            st.pairs.append((asset.id, t.id))
            st.weight[(asset.id, t.id)] = pair_weight(asset, t)
            st.assets_of[t.id].append(asset.id)
            for u in mine[i + 1:]:
                if tasks_conflict(asset, t, u):
                    near[t.id].add(u.id)
                    near[u.id].add(t.id)
        st.conflicts[asset.id] = near
    return st


def sort_pairs(problem: MissionProblem, pairs: Iterable[Pair]) -> List[Pair]:
    """Pairs in asset order, then task order."""
    a_pos = {a.id: i for i, a in enumerate(problem.assets)}
    t_pos = {t.id: i for i, t in enumerate(problem.tasks)}
    return sorted(((str(a), str(t)) for a, t in pairs), key=lambda p: (a_pos[p[0]], t_pos[p[1]]))
