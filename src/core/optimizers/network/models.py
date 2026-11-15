"""Domain model for the facility-location vertical (F32).

Distances are kilometres and costs are currency-neutral units, matching the
fleet vertical's conventions so the shared geo oracle and the Job History
renderers behave identically across both logistics scenarios.

Where F6 asks "given these depots, how do we serve these stops?", this module
asks the prior question: "which of these candidate sites should exist at all?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _round(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


@dataclass
class CandidateSite:
    """A potential distribution centre that may or may not be opened."""

    id: str
    name: str
    lat: float
    lon: float
    #: Amortised cost of operating this site for the planning horizon.
    fixed_cost: float = 10000.0
    #: Total demand units the site can serve per horizon.
    capacity: float = 1000.0
    #: Pre-existing sites can be pinned open; greenfield sites cannot.
    must_open: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "lat": _round(self.lat, 6),
            "lon": _round(self.lon, 6),
            "fixed_cost": _round(self.fixed_cost),
            "capacity": _round(self.capacity),
            "must_open": self.must_open,
        }


@dataclass
class DemandPoint:
    """A customer cluster that must be served by exactly one open site."""

    id: str
    name: str
    lat: float
    lon: float
    demand: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "lat": _round(self.lat, 6),
            "lon": _round(self.lon, 6),
            "demand": _round(self.demand),
        }


@dataclass
class NetworkProblem:
    """A complete capacitated facility-location instance."""

    sites: List[CandidateSite]
    demand_points: List[DemandPoint]
    #: Transport cost per demand-unit per kilometre.
    cost_per_km_per_unit: float = 0.02
    #: Straight-line to road-distance multiplier, shared with the fleet oracle.
    detour_factor: float = 1.25
    #: Optional cap on how many sites may open. ``None`` lets cost decide.
    max_open_sites: Optional[int] = None

    @property
    def total_demand(self) -> float:
        return float(sum(d.demand for d in self.demand_points))

    @property
    def total_capacity(self) -> float:
        return float(sum(s.capacity for s in self.sites))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sites": [s.to_dict() for s in self.sites],
            "demand_points": [d.to_dict() for d in self.demand_points],
            "cost_per_km_per_unit": _round(self.cost_per_km_per_unit, 6),
            "detour_factor": _round(self.detour_factor),
            "max_open_sites": self.max_open_sites,
            "total_demand": _round(self.total_demand),
            "total_capacity": _round(self.total_capacity),
        }


@dataclass
class SiteAllocation:
    """What one opened site ends up serving."""

    site_id: str
    site_name: str
    demand_point_ids: List[str] = field(default_factory=list)
    served_demand: float = 0.0
    capacity: float = 0.0
    fixed_cost: float = 0.0
    transport_cost: float = 0.0

    @property
    def utilization_pct(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return _round(100.0 * self.served_demand / self.capacity, 2)

    @property
    def overflow(self) -> float:
        return _round(max(0.0, self.served_demand - self.capacity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "demand_point_ids": list(self.demand_point_ids),
            "served_demand": _round(self.served_demand),
            "capacity": _round(self.capacity),
            "fixed_cost": _round(self.fixed_cost),
            "transport_cost": _round(self.transport_cost),
            "utilization_pct": self.utilization_pct,
            "overflow": self.overflow,
        }


@dataclass
class NetworkSolution:
    """A complete network design with its cost breakdown."""

    allocations: List[SiteAllocation] = field(default_factory=list)
    unassigned: List[str] = field(default_factory=list)
    fixed_cost: float = 0.0
    transport_cost: float = 0.0
    #: Populated by :func:`evaluate.finalise_solution`.
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return _round(self.fixed_cost + self.transport_cost)

    @property
    def open_site_ids(self) -> List[str]:
        return [a.site_id for a in self.allocations]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocations": [a.to_dict() for a in self.allocations],
            "open_site_ids": self.open_site_ids,
            "open_site_count": len(self.allocations),
            "unassigned": list(self.unassigned),
            "fixed_cost": _round(self.fixed_cost),
            "transport_cost": _round(self.transport_cost),
            "total_cost": self.total_cost,
            "metrics": dict(self.metrics),
        }
