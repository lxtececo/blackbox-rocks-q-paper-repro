"""Seeded mission allocation scenarios on a synthetic 300 x 200 km area (F7).

Four presets and a custom generator; neutral names, no real geography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.optimizers.mission.models import Asset, MissionProblem, Task

AREA_KM = (300.0, 200.0)
HORIZON_HOURS = 24.0
DEFAULT_SCENARIO = "relief-airlift"
CUSTOM_SCENARIO_ID = "custom"
CUSTOM_LIMITS = {"assets": [2, 14], "tasks": [4, 40]}


@dataclass(frozen=True)
class _Kind:
    """An asset kind: ranges the generator draws from."""

    kind: str
    label: str
    capabilities: Tuple[str, ...]
    speed_kmh: Tuple[float, float]
    cost_per_hour: Tuple[float, float]
    cost_per_km: Tuple[float, float]
    duty_hours: Tuple[float, float]


@dataclass(frozen=True)
class _TaskType:
    label: str
    capability: str
    duration_hours: Tuple[float, float]
    priority: Tuple[int, int]


@dataclass(frozen=True)
class _Profile:
    id: str
    name: str
    description: str
    asset_count: int
    task_count: int
    seed: int
    kinds: Tuple[_Kind, ...]
    task_types: Tuple[_TaskType, ...]
    bases: int = 2
    #: Earliest and latest task start.
    start_hours: Tuple[float, float] = (2.0, 20.0)
    available_from: Tuple[float, float] = (0.0, 4.0)
    available_until: Tuple[float, float] = (18.0, 24.0)


_TRANSPORT_HELI = _Kind("transport-helicopter", "Transport helicopter", ("airlift", "sling-load"),
                        (180.0, 230.0), (140.0, 240.0), (0.8, 1.6), (5.0, 8.0))
_UTILITY_HELI = _Kind("utility-helicopter", "Utility helicopter", ("airlift", "medical"),
                      (200.0, 250.0), (120.0, 200.0), (0.7, 1.3), (5.0, 8.0))
_MEDEVAC_HELI = _Kind("medevac-helicopter", "Medevac helicopter", ("medical", "airlift"),
                      (210.0, 260.0), (150.0, 240.0), (0.8, 1.5), (5.0, 7.5))
_AMBULANCE = _Kind("ambulance-team", "Ambulance team", ("medical",),
                   (60.0, 85.0), (40.0, 70.0), (0.4, 0.8), (8.0, 11.0))
_MAINTENANCE = _Kind("maintenance-team", "Maintenance team", ("engineering",),
                     (50.0, 75.0), (50.0, 90.0), (0.4, 0.8), (7.0, 10.0))
_RECOVERY = _Kind("recovery-vehicle", "Recovery vehicle", ("recovery", "transport"),
                  (45.0, 65.0), (60.0, 100.0), (0.6, 1.1), (7.0, 10.0))
_ENGINEERS = _Kind("engineer-section", "Engineer section", ("engineering", "recovery"),
                   (50.0, 70.0), (70.0, 110.0), (0.5, 1.0), (7.0, 10.0))
_CARGO_TRUCK = _Kind("cargo-truck", "Cargo truck", ("transport",),
                     (55.0, 80.0), (40.0, 80.0), (0.5, 0.9), (8.0, 11.0))
_FUEL_TRUCK = _Kind("fuel-truck", "Fuel truck", ("fuel",),
                    (50.0, 70.0), (50.0, 90.0), (0.6, 1.0), (8.0, 11.0))
_UTILITY_VEHICLE = _Kind("utility-vehicle", "Utility vehicle", ("transport", "recovery"),
                         (60.0, 85.0), (45.0, 75.0), (0.4, 0.8), (8.0, 11.0))

_PROFILES: Dict[str, _Profile] = {
    "relief-airlift": _Profile(
        id="relief-airlift",
        name="Disaster relief airlift",
        description=(
            "Helicopters deliver supplies, lift loads and pick up casualties "
            "across a flooded area."
        ),
        asset_count=6,
        task_count=18,
        seed=11,
        kinds=(_TRANSPORT_HELI, _UTILITY_HELI),
        task_types=(
            _TaskType("Supply drop", "airlift", (1.0, 2.5), (3, 8)),
            _TaskType("Sling-load lift", "sling-load", (1.0, 2.0), (4, 9)),
            _TaskType("Casualty pickup", "medical", (0.75, 1.5), (7, 10)),
        ),
        bases=2,
        start_hours=(3.0, 15.0),
    ),
    "medevac-readiness": _Profile(
        id="medevac-readiness",
        name="Medical evacuation readiness",
        description=(
            "Medevac helicopters and ambulance teams cover evacuations, "
            "patient transfers and medical resupply."
        ),
        asset_count=5,
        task_count=14,
        seed=12,
        kinds=(_MEDEVAC_HELI, _AMBULANCE),
        task_types=(
            _TaskType("Casualty evacuation", "medical", (1.0, 2.0), (6, 10)),
            _TaskType("Patient transfer", "medical", (1.5, 3.0), (3, 7)),
            _TaskType("Medical resupply", "airlift", (1.0, 2.0), (2, 6)),
        ),
        bases=2,
        start_hours=(3.0, 15.0),
    ),
    "field-maintenance": _Profile(
        id="field-maintenance",
        name="Field maintenance teams",
        description=(
            "Maintenance teams, engineer sections and recovery vehicles "
            "repair equipment and recover vehicles in the field."
        ),
        asset_count=8,
        task_count=24,
        seed=13,
        kinds=(_MAINTENANCE, _RECOVERY, _ENGINEERS),
        task_types=(
            _TaskType("Equipment repair", "engineering", (2.0, 4.0), (3, 9)),
            _TaskType("Vehicle recovery", "recovery", (1.5, 3.0), (4, 10)),
            _TaskType("Spare parts delivery", "transport", (1.0, 2.0), (2, 7)),
        ),
        bases=3,
        start_hours=(4.0, 17.0),
    ),
    "base-logistics": _Profile(
        id="base-logistics",
        name="Base logistics support",
        description=(
            "Trucks, fuel trucks, utility vehicles and a helicopter flight "
            "run supply, refuelling, recovery and airlift tasks."
        ),
        asset_count=10,
        task_count=30,
        seed=14,
        kinds=(_CARGO_TRUCK, _FUEL_TRUCK, _UTILITY_VEHICLE, _TRANSPORT_HELI),
        task_types=(
            _TaskType("Supply run", "transport", (1.5, 3.0), (2, 8)),
            _TaskType("Refuelling", "fuel", (1.0, 2.5), (4, 9)),
            _TaskType("Recovery call", "recovery", (1.5, 3.0), (3, 8)),
            _TaskType("Airlift shuttle", "airlift", (1.0, 2.0), (5, 10)),
        ),
        bases=3,
        start_hours=(4.0, 17.0),
    ),
}

_CUSTOM_PROFILE = _Profile(
    id=CUSTOM_SCENARIO_ID,
    name="Custom scenario",
    description="A generated mix of air and ground assets and support tasks.",
    asset_count=6,
    task_count=16,
    seed=0,
    # Neighbouring kinds share a capability, so small fleets compete for tasks.
    kinds=(_UTILITY_HELI, _TRANSPORT_HELI, _AMBULANCE, _UTILITY_VEHICLE, _ENGINEERS, _CARGO_TRUCK, _FUEL_TRUCK),
    task_types=(
        _TaskType("Supply drop", "airlift", (1.0, 2.5), (3, 8)),
        _TaskType("Casualty pickup", "medical", (0.75, 1.5), (6, 10)),
        _TaskType("Supply run", "transport", (1.5, 3.0), (2, 8)),
        _TaskType("Equipment repair", "engineering", (2.0, 4.0), (3, 9)),
        _TaskType("Vehicle recovery", "recovery", (1.5, 3.0), (3, 8)),
        _TaskType("Sling-load lift", "sling-load", (1.0, 2.0), (4, 9)),
        _TaskType("Refuelling", "fuel", (1.0, 2.5), (4, 9)),
    ),
    bases=2,
    start_hours=(3.0, 16.0),
)


def _grid(value: float, step: float) -> float:
    return float(round(round(value / step) * step, 3))


def _draw(rng: np.random.Generator, bounds: Tuple[float, float], step: float) -> float:
    return _grid(float(rng.uniform(bounds[0], bounds[1])), step)


def _generate(profile: _Profile, asset_count: int, task_count: int, seed: int) -> MissionProblem:
    """Draw one instance of a profile; the same seed gives the same instance."""
    rng = np.random.default_rng(seed)
    width, height = AREA_KM
    bases = [
        (_draw(rng, (0.2 * width, 0.8 * width), 1.0), _draw(rng, (0.2 * height, 0.8 * height), 1.0))
        for _ in range(profile.bases)
    ]

    assets: List[Asset] = []
    numbering: Dict[str, int] = {}
    for i in range(asset_count):
        kind = profile.kinds[i % len(profile.kinds)]
        numbering[kind.kind] = numbering.get(kind.kind, 0) + 1
        assets.append(
            Asset(
                id=f"A{i + 1}",
                name=f"{kind.label} {numbering[kind.kind]}",
                kind=kind.kind,
                capabilities=list(kind.capabilities),
                base=bases[int(rng.integers(len(bases)))],
                available_from=_draw(rng, profile.available_from, 0.5),
                available_until=_draw(rng, profile.available_until, 0.5),
                duty_limit_hours=_draw(rng, kind.duty_hours, 0.5),
                speed_kmh=_draw(rng, kind.speed_kmh, 5.0),
                cost_per_hour=_draw(rng, kind.cost_per_hour, 10.0),
                cost_per_km=_draw(rng, kind.cost_per_km, 0.1),
            )
        )

    # Only ask for capabilities the fleet has, so small fleets stay meaningful.
    offered = {c for a in assets for c in a.capabilities}
    types = [t for t in profile.task_types if t.capability in offered] or list(profile.task_types)
    tasks: List[Task] = []
    for j in range(task_count):
        kind_of_task = types[int(rng.integers(len(types)))]
        duration = _draw(rng, kind_of_task.duration_hours, 0.25)
        latest = min(profile.start_hours[1], HORIZON_HOURS - duration - 1.0)
        tasks.append(
            Task(
                id=f"T{j + 1}",
                name=f"{kind_of_task.label} {j + 1}",
                capability=kind_of_task.capability,
                position=(_draw(rng, (0.0, width), 1.0), _draw(rng, (0.0, height), 1.0)),
                start_hour=_draw(rng, (profile.start_hours[0], latest), 0.25),
                duration_hours=duration,
                priority=int(rng.integers(kind_of_task.priority[0], kind_of_task.priority[1] + 1)),
            )
        )

    return MissionProblem(
        assets=assets,
        tasks=tasks,
        scenario_id=profile.id,
        name=profile.name,
        description=profile.description,
        horizon_hours=HORIZON_HOURS,
        area_km=AREA_KM,
    )


def list_scenarios() -> List[Dict[str, Any]]:
    """Preset catalogue for the UI picker."""
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "asset_count": p.asset_count,
            "task_count": p.task_count,
        }
        for p in _PROFILES.values()
    ]


def get_scenario(scenario_id: Optional[str] = None) -> MissionProblem:
    """Build a preset by id; the default preset when no id is given."""
    key = (scenario_id or DEFAULT_SCENARIO).strip().lower()
    if key not in _PROFILES:
        raise ValueError(f"unknown mission scenario '{scenario_id}'; expected one of {sorted(_PROFILES)}")
    profile = _PROFILES[key]
    return _generate(profile, profile.asset_count, profile.task_count, profile.seed)


def _checked_int(value: Any, label: str, bounds: List[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    if not bounds[0] <= int(value) <= bounds[1]:
        raise ValueError(f"{label} must be between {bounds[0]} and {bounds[1]}, got {value}")
    return int(value)


def build_custom_scenario(asset_count: int = 6, task_count: int = 16, seed: int = 0) -> MissionProblem:
    """Generate a custom instance inside CUSTOM_LIMITS; deterministic per seed."""
    assets = _checked_int(asset_count, "asset_count", CUSTOM_LIMITS["assets"])
    tasks = _checked_int(task_count, "task_count", CUSTOM_LIMITS["tasks"])
    drawn = _checked_int(seed, "seed", [0, 2**32 - 1])
    problem = _generate(_CUSTOM_PROFILE, assets, tasks, drawn)
    problem.name = f"Custom scenario ({assets} assets, {tasks} tasks, seed {drawn})"
    return problem
