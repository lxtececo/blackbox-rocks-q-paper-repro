"""Domain model for the fleet-routing vertical (F6).

All clock values are **minutes since midnight** (local depot time). All
distances are kilometres. Keeping a single unit convention here is the
fleet analogue of the ``core.quantum_math`` single-authority rule: every
solver, the disruption engine and the API serializer read the same units,
so a "feasible" route means the same thing everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Sentinel for "no time window" — a full day.
DAY_START_MIN = 0
DAY_END_MIN = 24 * 60


def _round(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


@dataclass
class Depot:
    """A distribution centre that vehicles start and end their shift at."""

    id: str
    name: str
    lat: float
    lon: float
    open_min: int = DAY_START_MIN
    close_min: int = DAY_END_MIN
    # Set False by a ``dock_closure`` disruption.
    open: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "lat": _round(self.lat, 6),
            "lon": _round(self.lon, 6),
            "open_min": self.open_min,
            "close_min": self.close_min,
            "open": self.open,
        }


@dataclass
class Stop:
    """A delivery point with demand and an optional hard time window."""

    id: str
    name: str
    lat: float
    lon: float
    demand: float = 1.0
    service_min: float = 10.0
    tw_start: int = DAY_START_MIN
    tw_end: int = DAY_END_MIN
    priority: int = 1
    # Set False by a ``load_rejection`` disruption.
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "lat": _round(self.lat, 6),
            "lon": _round(self.lon, 6),
            "demand": _round(self.demand),
            "service_min": _round(self.service_min),
            "tw_start": self.tw_start,
            "tw_end": self.tw_end,
            "priority": self.priority,
            "active": self.active,
        }


@dataclass
class Vehicle:
    """A truck + driver pairing bound to a home depot."""

    id: str
    name: str
    depot_id: str
    capacity: float = 100.0
    shift_start_min: int = 6 * 60
    shift_end_min: int = 18 * 60
    cost_per_km: float = 1.0
    fixed_cost: float = 0.0
    # Cleared by ``vehicle_down`` / ``driver_no_show`` disruptions.
    available: bool = True
    # Trimmed by ``driver_out_of_hours``.
    max_route_minutes: Optional[float] = None

    @property
    def shift_minutes(self) -> float:
        span = float(self.shift_end_min - self.shift_start_min)
        if self.max_route_minutes is not None:
            return min(span, float(self.max_route_minutes))
        return span

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "depot_id": self.depot_id,
            "capacity": _round(self.capacity),
            "shift_start_min": self.shift_start_min,
            "shift_end_min": self.shift_end_min,
            "cost_per_km": _round(self.cost_per_km),
            "fixed_cost": _round(self.fixed_cost),
            "available": self.available,
            "max_route_minutes": self.max_route_minutes,
        }


@dataclass
class FleetProblem:
    """A complete CVRP-TW instance."""

    depots: List[Depot]
    stops: List[Stop]
    vehicles: List[Vehicle]
    average_speed_kmh: float = 60.0
    service_radius_km: float = 322.0  # 200 miles
    # Multiplier applied to travel time; raised by a traffic incident.
    traffic_factor: float = 1.0
    # Per-edge congestion overrides keyed by "fromId>toId".
    edge_delays_min: Dict[str, float] = field(default_factory=dict)
    name: str = "fleet-problem"
    seed: int = 42

    # Id lookup maps. The local-search inner loops resolve ids millions of
    # times on a 620-stop instance, so a linear scan is not an option.
    _index: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ── lookups ────────────────────────────────────────────────────────
    def reindex(self) -> None:
        """Rebuild the id lookup maps. Call after mutating the collections."""
        self._index = {
            "depots": {d.id: d for d in self.depots},
            "stops": {s.id: s for s in self.stops},
            "vehicles": {v.id: v for v in self.vehicles},
        }

    def _lookup(self, bucket: str, key: str, label: str) -> Any:
        table = self._index.get(bucket)
        if table is None or key not in table:
            self.reindex()
            table = self._index[bucket]
        try:
            return table[key]
        except KeyError:
            raise KeyError(f"unknown {label}: {key}") from None

    def depot(self, depot_id: str) -> Depot:
        return self._lookup("depots", depot_id, "depot")

    def stop(self, stop_id: str) -> Stop:
        return self._lookup("stops", stop_id, "stop")

    def vehicle(self, vehicle_id: str) -> Vehicle:
        return self._lookup("vehicles", vehicle_id, "vehicle")

    @property
    def active_stops(self) -> List[Stop]:
        return [s for s in self.stops if s.active]

    @property
    def usable_vehicles(self) -> List[Vehicle]:
        open_depots = {d.id for d in self.depots if d.open}
        return [v for v in self.vehicles if v.available and v.depot_id in open_depots]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "average_speed_kmh": _round(self.average_speed_kmh),
            "service_radius_km": _round(self.service_radius_km),
            "traffic_factor": _round(self.traffic_factor),
            "depots": [d.to_dict() for d in self.depots],
            "stops": [s.to_dict() for s in self.stops],
            "vehicles": [v.to_dict() for v in self.vehicles],
        }


@dataclass
class RouteVisit:
    """One stop on a route with the simulated arrival/departure clock."""

    stop_id: str
    arrival_min: float
    departure_min: float
    wait_min: float
    load_after: float
    leg_km: float
    late_min: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_id": self.stop_id,
            "arrival_min": _round(self.arrival_min, 2),
            "departure_min": _round(self.departure_min, 2),
            "wait_min": _round(self.wait_min, 2),
            "load_after": _round(self.load_after),
            "leg_km": _round(self.leg_km),
            "late_min": _round(self.late_min, 2),
        }


@dataclass
class Route:
    """A vehicle's ordered tour out of and back to its depot."""

    vehicle_id: str
    depot_id: str
    visits: List[RouteVisit] = field(default_factory=list)
    distance_km: float = 0.0
    duration_min: float = 0.0
    load: float = 0.0
    cost: float = 0.0
    feasible: bool = True
    violations: List[str] = field(default_factory=list)

    @property
    def stop_ids(self) -> List[str]:
        return [v.stop_id for v in self.visits]

    @property
    def is_empty(self) -> bool:
        return not self.visits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "depot_id": self.depot_id,
            "stop_ids": self.stop_ids,
            "visits": [v.to_dict() for v in self.visits],
            "distance_km": _round(self.distance_km),
            "duration_min": _round(self.duration_min, 2),
            "load": _round(self.load),
            "cost": _round(self.cost),
            "feasible": self.feasible,
            "violations": list(self.violations),
        }


@dataclass
class FleetSolution:
    """Solver output plus the KPI block the UI and Job History render."""

    routes: List[Route] = field(default_factory=list)
    unassigned_stop_ids: List[str] = field(default_factory=list)
    method: str = "classical"
    platform: str = "cpu"
    execution_mode: str = "classical"
    objective: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def active_routes(self) -> List[Route]:
        return [r for r in self.routes if not r.is_empty]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "platform": self.platform,
            "execution_mode": self.execution_mode,
            "objective": _round(self.objective),
            "routes": [r.to_dict() for r in self.active_routes],
            "unassigned_stop_ids": list(self.unassigned_stop_ids),
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
        }


@dataclass
class DisruptionEvent:
    """A mid-shift event that invalidates part of the running plan."""

    kind: str
    at_min: int = 0
    target_id: Optional[str] = None
    magnitude: float = 1.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "at_min": self.at_min,
            "target_id": self.target_id,
            "magnitude": _round(self.magnitude),
            "note": self.note,
        }
