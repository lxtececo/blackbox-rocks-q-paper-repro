"""Shared route simulation, feasibility and KPI accounting.

Every fleet solver — classical baseline, QUBO annealing, gate-based
sampling and the disruption replanner — scores candidate tours through
this module. That is what makes the "quantum vs classical" comparison in
the UI honest: only the assignment/sequencing search differs, never the
objective.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.optimizers.fleet.geo import DistanceOracle, travel_minutes
from core.optimizers.fleet.models import (
    FleetProblem,
    FleetSolution,
    Route,
    RouteVisit,
    Vehicle,
)

#: Cost added per stop the plan fails to serve. Deliberately large so any
#: feasible assignment beats dropping a delivery, but finite so the solver
#: can still return a best-effort plan when the fleet is over-subscribed.
UNASSIGNED_PENALTY = 500.0

#: Cost per minute of time-window lateness.
LATENESS_PENALTY_PER_MIN = 2.0


def build_oracle(problem: FleetProblem) -> DistanceOracle:
    """Distance oracle covering every depot and stop in the problem."""
    points = [(d.id, d.lat, d.lon) for d in problem.depots]
    points += [(s.id, s.lat, s.lon) for s in problem.stops]
    return DistanceOracle(points)


def _edge_delay(problem: FleetProblem, a: str, b: str) -> float:
    delays = problem.edge_delays_min
    if not delays:
        return 0.0
    return float(delays.get(f"{a}>{b}", delays.get(f"{b}>{a}", 0.0)))


def evaluate_route(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    stop_ids: Sequence[str],
) -> Route:
    """Simulate a tour and return a fully-costed :class:`Route`.

    The clock starts at the later of the vehicle's shift start and the
    depot's opening time, waits out early arrivals, and accumulates
    lateness against each stop's hard window.
    """
    depot = problem.depot(vehicle.depot_id)
    route = Route(vehicle_id=vehicle.id, depot_id=depot.id)

    if not stop_ids:
        return route

    clock = float(max(vehicle.shift_start_min, depot.open_min))
    start_clock = clock
    here = depot.id
    load = sum(problem.stop(sid).demand for sid in stop_ids)
    remaining = load
    total_km = 0.0
    total_late = 0.0

    for sid in stop_ids:
        stop = problem.stop(sid)
        leg_km = oracle.km(here, sid)
        total_km += leg_km
        clock += travel_minutes(
            leg_km, problem.average_speed_kmh, problem.traffic_factor
        )
        clock += _edge_delay(problem, here, sid)

        wait = 0.0
        if clock < stop.tw_start:
            wait = float(stop.tw_start) - clock
            clock = float(stop.tw_start)
        arrival = clock
        late = max(0.0, clock - float(stop.tw_end))
        total_late += late

        clock += stop.service_min
        remaining -= stop.demand
        route.visits.append(
            RouteVisit(
                stop_id=sid,
                arrival_min=arrival,
                departure_min=clock,
                wait_min=wait,
                load_after=remaining,
                leg_km=leg_km,
                late_min=late,
            )
        )
        here = sid

    return_km = oracle.km(here, depot.id)
    total_km += return_km
    clock += travel_minutes(
        return_km, problem.average_speed_kmh, problem.traffic_factor
    )
    clock += _edge_delay(problem, here, depot.id)

    route.distance_km = total_km
    route.duration_min = clock - start_clock
    route.load = load
    route.cost = (
        total_km * vehicle.cost_per_km
        + vehicle.fixed_cost
        + total_late * LATENESS_PENALTY_PER_MIN
    )

    if load > vehicle.capacity + 1e-9:
        route.violations.append(
            f"capacity_exceeded:{load:.1f}>{vehicle.capacity:.1f}"
        )
    if total_late > 1e-9:
        route.violations.append(f"time_window_late:{total_late:.1f}min")
    if route.duration_min > vehicle.shift_minutes + 1e-9:
        route.violations.append(
            f"shift_exceeded:{route.duration_min:.0f}>{vehicle.shift_minutes:.0f}min"
        )
    if clock > depot.close_min + 1e-9:
        route.violations.append(f"depot_closed_on_return:{clock:.0f}min")
    if not depot.open:
        route.violations.append("depot_closed")
    if not vehicle.available:
        route.violations.append("vehicle_unavailable")

    route.feasible = not route.violations
    return route


def route_is_feasible(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    stop_ids: Sequence[str],
) -> bool:
    """Cheap yes/no wrapper used inside the local-search inner loops."""
    if not stop_ids:
        return True
    if sum(problem.stop(sid).demand for sid in stop_ids) > vehicle.capacity + 1e-9:
        return False
    return evaluate_route(problem, oracle, vehicle, stop_ids).feasible


def solution_objective(
    routes: Iterable[Route],
    unassigned: Sequence[str],
) -> float:
    """Total plan cost: route cost + a penalty per unserved delivery."""
    return sum(r.cost for r in routes) + UNASSIGNED_PENALTY * len(unassigned)


def build_metrics(
    problem: FleetProblem,
    routes: Sequence[Route],
    unassigned: Sequence[str],
) -> Dict[str, Any]:
    """KPI block rendered by the fleet panel and stored on the job."""
    active = [r for r in routes if not r.is_empty]
    total_km = sum(r.distance_km for r in active)
    total_min = sum(r.duration_min for r in active)
    total_cost = sum(r.cost for r in active)
    served = sum(len(r.visits) for r in active)
    total_stops = len(problem.active_stops)
    late_stops = sum(1 for r in active for v in r.visits if v.late_min > 1e-9)
    capacity = sum(problem.vehicle(r.vehicle_id).capacity for r in active) or 1.0
    load = sum(r.load for r in active)

    on_time = served - late_stops
    return {
        "total_distance_km": round(total_km, 2),
        "total_distance_miles": round(total_km * 0.621371, 2),
        "total_duration_min": round(total_min, 2),
        "total_cost": round(total_cost, 2),
        "routes_used": len(active),
        "vehicles_available": len(problem.usable_vehicles),
        "stops_served": served,
        "stops_total": total_stops,
        "stops_unassigned": len(unassigned),
        "service_level_pct": round(100.0 * served / total_stops, 2) if total_stops else 0.0,
        "on_time_pct": round(100.0 * on_time / served, 2) if served else 0.0,
        "late_stops": late_stops,
        "avg_stops_per_route": round(served / len(active), 2) if active else 0.0,
        "avg_km_per_stop": round(total_km / served, 3) if served else 0.0,
        "capacity_utilisation_pct": round(100.0 * load / capacity, 2),
        "infeasible_routes": sum(1 for r in active if not r.feasible),
    }


def finalise_solution(
    problem: FleetProblem,
    routes: Sequence[Route],
    unassigned: Sequence[str],
    *,
    method: str,
    platform: str,
    execution_mode: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> FleetSolution:
    """Wrap raw routes into a scored :class:`FleetSolution`."""
    active: List[Route] = [r for r in routes if not r.is_empty]
    return FleetSolution(
        routes=active,
        unassigned_stop_ids=list(unassigned),
        method=method,
        platform=platform,
        execution_mode=execution_mode,
        objective=solution_objective(active, unassigned),
        metrics=build_metrics(problem, active, unassigned),
        diagnostics=dict(diagnostics or {}),
    )
