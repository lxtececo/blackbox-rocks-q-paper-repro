"""Disruption simulation and replanning for the fleet vertical.

Covers the six disruption classes a dispatcher actually faces mid-shift:

============================  =========================================
``traffic_incident``          Congestion on a corridor: travel times
                              inflate for the rest of the shift.
``dock_closure``              A distribution centre stops accepting or
                              releasing loads.
``driver_no_show``            A booked driver never signs on.
``driver_out_of_hours``       A driver hits the legal duty limit early.
``vehicle_down``              Breakdown; the truck is out for the day.
``load_rejection``            A consignee refuses the delivery.
============================  =========================================

Replanning freezes everything already delivered at the disruption time,
then re-solves the *remaining* network with the same solver the original
plan used, so the before/after comparison in the UI is like-for-like.

MVP simplification: replanned vehicles restart from their home depot with
their shift clock advanced to the disruption time, rather than from their
exact mid-route position. That keeps the objective identical to the
initial plan's; mid-route origins are tracked in F6 follow-up work.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.optimizers.fleet.evaluate import build_oracle, finalise_solution
from core.optimizers.fleet.models import (
    DisruptionEvent,
    FleetProblem,
    FleetSolution,
    Route,
    Vehicle,
)

logger = logging.getLogger(__name__)

DISRUPTION_KINDS: Dict[str, Dict[str, Any]] = {
    "traffic_incident": {
        "label": "Traffic incident",
        "target": "none",
        "description": "Corridor congestion inflates travel times fleet-wide.",
        "default_magnitude": 1.45,
    },
    "dock_closure": {
        "label": "Dock closure",
        "target": "depot",
        "description": "A distribution centre stops accepting or releasing loads.",
        "default_magnitude": 1.0,
    },
    "driver_no_show": {
        "label": "Driver no-show",
        "target": "vehicle",
        "description": "A booked driver never signs on; the truck stays parked.",
        "default_magnitude": 1.0,
    },
    "driver_out_of_hours": {
        "label": "Driver out of hours",
        "target": "vehicle",
        "description": "Duty-time limit reached; the route must end early.",
        "default_magnitude": 0.5,
    },
    "vehicle_down": {
        "label": "Vehicle down",
        "target": "vehicle",
        "description": "Mechanical failure removes the truck for the rest of the day.",
        "default_magnitude": 1.0,
    },
    "load_rejection": {
        "label": "Load rejection",
        "target": "stop",
        "description": "The consignee refuses the delivery; the drop is cancelled.",
        "default_magnitude": 1.0,
    },
}


def list_disruption_kinds() -> List[Dict[str, Any]]:
    """Catalogue for the UI's disruption toolbar."""
    return [{"kind": key, **spec} for key, spec in DISRUPTION_KINDS.items()]


def apply_disruption(
    problem: FleetProblem, event: DisruptionEvent
) -> Dict[str, Any]:
    """Mutate ``problem`` in place to reflect ``event``.

    Returns a short human-readable impact summary for the timeline feed.
    """
    kind = event.kind
    if kind not in DISRUPTION_KINDS:
        raise ValueError(
            f"unknown disruption '{kind}'; expected one of {sorted(DISRUPTION_KINDS)}"
        )

    impact: Dict[str, Any] = {"kind": kind, "at_min": event.at_min}

    if kind == "traffic_incident":
        factor = max(1.0, float(event.magnitude or 1.45))
        problem.traffic_factor = round(problem.traffic_factor * factor, 4)
        impact["traffic_factor"] = problem.traffic_factor
        impact["summary"] = (
            f"Travel times up {int((factor - 1) * 100)}% for the rest of the shift"
        )

    elif kind == "dock_closure":
        depot = problem.depot(event.target_id) if event.target_id else problem.depots[0]
        depot.open = False
        impact["depot_id"] = depot.id
        impact["summary"] = f"{depot.name} closed; its trucks must be re-homed"

    elif kind in {"driver_no_show", "vehicle_down"}:
        vehicle = (
            problem.vehicle(event.target_id)
            if event.target_id
            else problem.usable_vehicles[0]
        )
        vehicle.available = False
        impact["vehicle_id"] = vehicle.id
        impact["summary"] = (
            f"{vehicle.name} unavailable"
            + (" (breakdown)" if kind == "vehicle_down" else " (no-show)")
        )

    elif kind == "driver_out_of_hours":
        vehicle = (
            problem.vehicle(event.target_id)
            if event.target_id
            else problem.usable_vehicles[0]
        )
        remaining = max(
            30.0,
            float(vehicle.shift_minutes) * max(0.05, min(float(event.magnitude), 0.95)),
        )
        vehicle.max_route_minutes = remaining
        impact["vehicle_id"] = vehicle.id
        impact["max_route_minutes"] = round(remaining, 1)
        impact["summary"] = (
            f"{vehicle.name} limited to {int(remaining)} min of remaining duty time"
        )

    elif kind == "load_rejection":
        stop = (
            problem.stop(event.target_id)
            if event.target_id
            else problem.active_stops[0]
        )
        stop.active = False
        impact["stop_id"] = stop.id
        impact["summary"] = f"{stop.name} refused the load; drop cancelled"

    problem.reindex()
    return impact


def _served_before(solution: FleetSolution, at_min: float) -> Tuple[List[str], Dict[str, List[str]]]:
    """Stops already delivered at ``at_min``, overall and per vehicle."""
    served: List[str] = []
    by_vehicle: Dict[str, List[str]] = {}
    for route in solution.routes:
        done = [v.stop_id for v in route.visits if v.departure_min <= at_min]
        if done:
            by_vehicle[route.vehicle_id] = done
            served.extend(done)
    return served, by_vehicle


def _resume_vehicles(
    problem: FleetProblem, at_min: float
) -> List[Vehicle]:
    """Clones of the still-usable vehicles with their clock advanced."""
    resumed: List[Vehicle] = []
    for vehicle in problem.usable_vehicles:
        start = int(max(vehicle.shift_start_min, at_min))
        if start >= vehicle.shift_end_min:
            continue
        resumed.append(replace(vehicle, shift_start_min=start))
    return resumed


def replan_after_disruption(
    problem: FleetProblem,
    solution: FleetSolution,
    event: DisruptionEvent,
    solver: Callable[..., Tuple[List[Route], List[str], Dict[str, Any]]],
    *,
    solver_kwargs: Optional[Dict[str, Any]] = None,
    method: Optional[str] = None,
    platform: Optional[str] = None,
    execution_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply ``event`` and re-solve the remaining network.

    ``solver`` must accept ``(problem, oracle=..., stop_ids=..., vehicles=...)``
    and return ``(routes, unassigned, diagnostics)`` — i.e. the signature
    shared by the classical and quantum fleet solvers.
    """
    at_min = float(event.at_min)
    baseline_metrics = dict(solution.metrics)

    impact = apply_disruption(problem, event)
    oracle = build_oracle(problem)

    served, served_by_vehicle = _served_before(solution, at_min)
    served_set = set(served)
    open_stops = [
        s.id for s in problem.active_stops if s.id not in served_set
    ]
    resume_pool = _resume_vehicles(problem, at_min)

    if not open_stops:
        replanned = finalise_solution(
            problem,
            [],
            [],
            method=method or solution.method,
            platform=platform or solution.platform,
            execution_mode=execution_mode or solution.execution_mode,
            diagnostics={"note": "all deliveries already completed"},
        )
    elif not resume_pool:
        logger.warning("[fleet-replan] no vehicles left after %s", event.kind)
        replanned = finalise_solution(
            problem,
            [],
            open_stops,
            method=method or solution.method,
            platform=platform or solution.platform,
            execution_mode=execution_mode or solution.execution_mode,
            diagnostics={"note": "no usable vehicles remain"},
        )
    else:
        routes, unassigned, diagnostics = solver(
            problem,
            oracle=oracle,
            stop_ids=open_stops,
            vehicles=resume_pool,
            **(solver_kwargs or {}),
        )
        replanned = finalise_solution(
            problem,
            routes,
            unassigned,
            method=method or solution.method,
            platform=platform or solution.platform,
            execution_mode=execution_mode or solution.execution_mode,
            diagnostics=diagnostics,
        )

    reassigned = _count_reassigned(solution, replanned, served_set)
    delta = {
        "distance_km": round(
            replanned.metrics.get("total_distance_km", 0.0)
            - baseline_metrics.get("total_distance_km", 0.0),
            2,
        ),
        "cost": round(
            replanned.metrics.get("total_cost", 0.0)
            - baseline_metrics.get("total_cost", 0.0),
            2,
        ),
        "unassigned": replanned.metrics.get("stops_unassigned", 0)
        - baseline_metrics.get("stops_unassigned", 0),
        "routes_used": replanned.metrics.get("routes_used", 0)
        - baseline_metrics.get("routes_used", 0),
        "stops_reassigned": reassigned,
    }

    return {
        "event": event.to_dict(),
        "impact": impact,
        "at_min": at_min,
        "completed_stops": served,
        "completed_by_vehicle": served_by_vehicle,
        "open_stops": len(open_stops),
        "vehicles_available": len(resume_pool),
        "solution": replanned.to_dict(),
        "baseline_metrics": baseline_metrics,
        "delta": delta,
        "problem": problem.to_dict(),
    }


def _count_reassigned(
    before: FleetSolution, after: FleetSolution, served: Sequence[str]
) -> int:
    """Stops that changed truck between the two plans (excluding delivered)."""
    served_set = set(served)
    prior = {
        visit.stop_id: route.vehicle_id
        for route in before.routes
        for visit in route.visits
        if visit.stop_id not in served_set
    }
    moved = 0
    for route in after.routes:
        for visit in route.visits:
            if prior.get(visit.stop_id, route.vehicle_id) != route.vehicle_id:
                moved += 1
    return moved
