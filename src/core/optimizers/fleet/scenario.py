"""Scenario generator for the fleet vertical.

The flagship preset reproduces the reference logistics benchmark: a
**Regional Food Redistributor** operating 3 distribution centres over a
200-mile service radius with 80 trucks, 80 drivers and 620 delivery
points. Smaller presets exist so the interactive panel stays responsive —
the full instance is meant for batch/benchmark runs, not for a 2-second
click-through.

Generation is fully seeded: the same ``(preset, seed)`` always yields the
same instance, which is what makes the classical-vs-quantum comparison and
the disruption replays reproducible.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from core.optimizers.fleet.geo import miles_to_km
from core.optimizers.fleet.models import FleetProblem, Depot, Stop, Vehicle

#: Time windows snap to a 5-minute grid (F6 design decision) so window
#: arithmetic stays integral and the UI timeline has clean ticks.
TIME_BUCKET_MIN = 5

SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "regional_food_redistributor": {
        "label": "Regional Food Redistributor (full)",
        "description": (
            "3 distribution centres, 200-mile service radius, 80 trucks, "
            "80 drivers, 620 delivery points."
        ),
        "depots": 3,
        "stops": 620,
        "vehicles": 80,
        "service_radius_miles": 200.0,
        "vehicle_capacity": 60.0,
        "interactive": False,
    },
    "regional_food_redistributor_demo": {
        "label": "Regional Food Redistributor (demo)",
        "description": (
            "Same network shape at 1/7 scale so the live panel replans in "
            "a couple of seconds."
        ),
        "depots": 3,
        "stops": 90,
        "vehicles": 12,
        "service_radius_miles": 200.0,
        "vehicle_capacity": 60.0,
        "interactive": True,
    },
    "metro_grocery_last_mile": {
        "label": "Metro Grocery Last Mile",
        "description": (
            "Single urban depot, tight afternoon windows, 120 drops, "
            "18 light vehicles."
        ),
        "depots": 1,
        "stops": 120,
        "vehicles": 18,
        "service_radius_miles": 45.0,
        "vehicle_capacity": 35.0,
        "interactive": True,
    },
}

DEFAULT_SCENARIO = "regional_food_redistributor_demo"

#: Anchor for the synthetic region (rural Ohio). Only the relative geometry
#: matters, but real coordinates keep the map plausible.
CENTRE_LAT = 40.05
CENTRE_LON = -83.05

_DEPOT_NAMES = [
    "Northgate DC",
    "Riverport DC",
    "Southfield DC",
    "Westline DC",
    "Eastvale DC",
]

_TOWN_NAMES = [
    "Ashford", "Brookvale", "Cedar Hollow", "Dunmore", "Eastbrook",
    "Fairhaven", "Glenwood", "Harrowgate", "Ivyton", "Junction Mills",
    "Kingsbury", "Lakeport", "Marchfield", "Northwood", "Oakridge",
    "Pinehurst", "Quarry Bend", "Redstone", "Stillwater", "Thornbury",
]

_OUTLET_KINDS = [
    "Supermarket", "Food Bank", "School Kitchen", "Care Home",
    "Convenience", "Restaurant", "Community Pantry", "Hospital Canteen",
]


def _bucket(minutes: float) -> int:
    """Snap a clock value to the 5-minute grid."""
    return int(round(minutes / TIME_BUCKET_MIN) * TIME_BUCKET_MIN)


def _offset(lat: float, lon: float, km_north: float, km_east: float) -> tuple:
    """Shift a coordinate by a local north/east offset in kilometres."""
    dlat = km_north / 110.574
    dlon = km_east / (111.320 * math.cos(math.radians(lat)))
    return (round(lat + dlat, 6), round(lon + dlon, 6))


def _scatter(
    rng: random.Random,
    lat: float,
    lon: float,
    radius_km: float,
) -> tuple:
    """Uniform point inside a disc of ``radius_km`` around (lat, lon)."""
    angle = rng.uniform(0.0, 2.0 * math.pi)
    dist = radius_km * math.sqrt(rng.random())
    return _offset(lat, lon, dist * math.cos(angle), dist * math.sin(angle))


def build_scenario(
    preset: str = DEFAULT_SCENARIO,
    *,
    seed: int = 42,
    stops: Optional[int] = None,
    vehicles: Optional[int] = None,
    depots: Optional[int] = None,
    service_radius_miles: Optional[float] = None,
    vehicle_capacity: Optional[float] = None,
) -> FleetProblem:
    """Build a seeded :class:`FleetProblem` from a named preset."""
    if preset not in SCENARIO_PRESETS:
        raise ValueError(
            f"unknown scenario '{preset}'; expected one of "
            f"{sorted(SCENARIO_PRESETS)}"
        )
    spec = SCENARIO_PRESETS[preset]
    n_depots = max(1, int(depots if depots is not None else spec["depots"]))
    n_stops = max(1, int(stops if stops is not None else spec["stops"]))
    n_vehicles = max(1, int(vehicles if vehicles is not None else spec["vehicles"]))
    radius_km = miles_to_km(
        float(
            service_radius_miles
            if service_radius_miles is not None
            else spec["service_radius_miles"]
        )
    )
    capacity = float(
        vehicle_capacity
        if vehicle_capacity is not None
        else spec["vehicle_capacity"]
    )

    rng = random.Random(seed)  # nosec B311 - random used for non-cryptographic simulation/jitter only

    # ── depots on a ring around the region centre ──────────────────────
    depot_ring_km = min(radius_km * 0.28, 70.0)
    depot_list: List[Depot] = []
    for i in range(n_depots):
        if n_depots == 1:
            lat, lon = CENTRE_LAT, CENTRE_LON
        else:
            angle = 2.0 * math.pi * i / n_depots + 0.4
            lat, lon = _offset(
                CENTRE_LAT,
                CENTRE_LON,
                depot_ring_km * math.cos(angle),
                depot_ring_km * math.sin(angle),
            )
        depot_list.append(
            Depot(
                id=f"DC{i + 1}",
                name=_DEPOT_NAMES[i % len(_DEPOT_NAMES)],
                lat=lat,
                lon=lon,
                open_min=_bucket(4 * 60),
                close_min=_bucket(22 * 60),
            )
        )

    # ── towns: delivery points cluster around population centres ───────
    n_towns = max(3, min(len(_TOWN_NAMES), n_stops // 12 + n_depots))
    towns: List[tuple] = []
    for i in range(n_towns):
        lat, lon = _scatter(rng, CENTRE_LAT, CENTRE_LON, radius_km * 0.72)
        towns.append((_TOWN_NAMES[i % len(_TOWN_NAMES)], lat, lon))

    # Window archetypes on the 5-minute grid: morning, midday, afternoon,
    # and an all-day window for flexible outlets.
    windows = [
        (_bucket(7 * 60), _bucket(11 * 60), 0.32),
        (_bucket(9 * 60), _bucket(14 * 60), 0.26),
        (_bucket(12 * 60), _bucket(17 * 60), 0.22),
        (_bucket(6 * 60), _bucket(18 * 60), 0.20),
    ]

    stop_list: List[Stop] = []
    for i in range(n_stops):
        town_name, town_lat, town_lon = towns[i % len(towns)]
        lat, lon = _scatter(rng, town_lat, town_lon, radius_km * 0.06)
        roll = rng.random()
        cumulative = 0.0
        tw_start, tw_end = windows[-1][0], windows[-1][1]
        for start, end, weight in windows:
            cumulative += weight
            if roll <= cumulative:
                tw_start, tw_end = start, end
                break
        demand = float(rng.randint(1, 6))
        stop_list.append(
            Stop(
                id=f"S{i + 1:04d}",
                name=f"{town_name} {rng.choice(_OUTLET_KINDS)} {i + 1}",
                lat=lat,
                lon=lon,
                demand=demand,
                service_min=float(_bucket(8 + demand * 2)),
                tw_start=tw_start,
                tw_end=tw_end,
                priority=1 if demand < 5 else 2,
            )
        )

    # ── one driver per truck, evenly spread across the depots ──────────
    vehicle_list: List[Vehicle] = []
    for i in range(n_vehicles):
        depot = depot_list[i % n_depots]
        vehicle_list.append(
            Vehicle(
                id=f"T{i + 1:03d}",
                name=f"Truck {i + 1:03d} / Driver {i + 1:03d}",
                depot_id=depot.id,
                capacity=capacity,
                shift_start_min=_bucket(6 * 60),
                shift_end_min=_bucket(18 * 60),
                cost_per_km=1.15,
                fixed_cost=45.0,
            )
        )

    problem = FleetProblem(
        depots=depot_list,
        stops=stop_list,
        vehicles=vehicle_list,
        average_speed_kmh=68.0,
        service_radius_km=radius_km,
        name=spec["label"],
        seed=seed,
    )
    problem.reindex()
    return problem


def list_scenarios() -> List[Dict[str, Any]]:
    """Scenario catalogue for the UI picker."""
    return [
        {
            "id": key,
            "label": spec["label"],
            "description": spec["description"],
            "depots": spec["depots"],
            "stops": spec["stops"],
            "vehicles": spec["vehicles"],
            "service_radius_miles": spec["service_radius_miles"],
            "vehicle_capacity": spec["vehicle_capacity"],
            "interactive": spec["interactive"],
            "default": key == DEFAULT_SCENARIO,
        }
        for key, spec in SCENARIO_PRESETS.items()
    ]
