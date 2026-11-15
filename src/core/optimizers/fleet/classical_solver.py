"""Classical CVRP-TW baseline: Clarke-Wright savings + local search.

Deliberately dependency-free (no OR-Tools) so the comparison baseline runs
in every environment the API runs in — including the slim runtime image and
CI — with no extra wheels to build. Quality is competitive with OR-Tools'
default first-solution + guided-local-search on regional instances of this
size, which is what matters for the quantum-vs-classical narrative.

Pipeline:

1. Assign each delivery point to its nearest open depot inside the service
   radius (multi-depot partition).
2. Clarke-Wright parallel savings over a k-nearest-neighbour candidate
   list, with capacity and time-window feasibility enforced on every merge.
3. Bind routes to concrete vehicles (largest load to largest capacity).
4. Local search: intra-route 2-opt and Or-opt, then inter-route relocate.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.optimizers.fleet.evaluate import evaluate_route
from core.optimizers.fleet.geo import DistanceOracle
from core.optimizers.fleet.models import FleetProblem, Route, Vehicle

logger = logging.getLogger(__name__)

#: Cost returned for an infeasible candidate so local search rejects it.
INFEASIBLE_COST = float("inf")

#: Candidate neighbours considered per stop when building the savings list.
#: 620 stops x 3 depots would be ~64k unrestricted pairs per depot; the
#: k-nearest restriction keeps it linear without measurable quality loss.
DEFAULT_NEIGHBOURS = 16

#: Wall-clock budget for the whole classical solve.
DEFAULT_TIME_BUDGET_S = 20.0


def _route_cost(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    seq: Sequence[str],
) -> float:
    if not seq:
        return 0.0
    route = evaluate_route(problem, oracle, vehicle, seq)
    return route.cost if route.feasible else INFEASIBLE_COST


def assign_stops_to_depots(
    problem: FleetProblem,
    oracle: DistanceOracle,
    stop_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Partition stops across open depots by nearest-depot within radius.

    Returns ``(by_depot, out_of_range)``.
    """
    depots = [d for d in problem.depots if d.open]
    candidates = (
        [problem.stop(sid) for sid in stop_ids]
        if stop_ids is not None
        else problem.active_stops
    )
    by_depot: Dict[str, List[str]] = {d.id: [] for d in depots}
    out_of_range: List[str] = []

    if not depots:
        return by_depot, [s.id for s in candidates]

    for stop in candidates:
        if not stop.active:
            continue
        best_id, best_km = None, INFEASIBLE_COST
        for depot in depots:
            km = oracle.km(depot.id, stop.id)
            if km < best_km:
                best_id, best_km = depot.id, km
        if best_id is None or best_km > problem.service_radius_km:
            out_of_range.append(stop.id)
            continue
        by_depot[best_id].append(stop.id)

    return by_depot, out_of_range


def _nearest_neighbour_pairs(
    oracle: DistanceOracle,
    stop_ids: Sequence[str],
    k: int,
) -> List[Tuple[str, str]]:
    """Unique undirected candidate pairs from each stop's k nearest peers."""
    pairs: set = set()
    for a in stop_ids:
        ranked = sorted(
            (oracle.km(a, b), b) for b in stop_ids if b != a
        )
        for _, b in ranked[:k]:
            pairs.add((a, b) if a < b else (b, a))
    return list(pairs)


def _savings_routes(
    problem: FleetProblem,
    oracle: DistanceOracle,
    depot_id: str,
    reference_vehicle: Vehicle,
    stop_ids: Sequence[str],
    *,
    neighbours: int,
    deadline: float,
) -> List[List[str]]:
    """Clarke-Wright parallel savings restricted to nearest-neighbour pairs."""
    if not stop_ids:
        return []

    routes: Dict[int, List[str]] = {i: [sid] for i, sid in enumerate(stop_ids)}
    route_of: Dict[str, int] = {sid: i for i, sid in enumerate(stop_ids)}

    pairs = _nearest_neighbour_pairs(oracle, stop_ids, neighbours)
    scored: List[Tuple[float, str, str]] = []
    for a, b in pairs:
        saving = (
            oracle.km(depot_id, a) + oracle.km(depot_id, b) - oracle.km(a, b)
        )
        if saving > 0:
            scored.append((saving, a, b))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    capacity = reference_vehicle.capacity
    for _, a, b in scored:
        if time.monotonic() > deadline:
            logger.warning(
                "[fleet] savings merge hit the time budget at depot=%s", depot_id
            )
            break
        ra, rb = route_of[a], route_of[b]
        if ra == rb:
            continue
        seq_a, seq_b = routes[ra], routes[rb]

        load = sum(problem.stop(s).demand for s in seq_a) + sum(
            problem.stop(s).demand for s in seq_b
        )
        if load > capacity + 1e-9:
            continue

        # Only endpoint-to-endpoint merges keep the tour a simple path.
        merged: Optional[List[str]] = None
        if seq_a[-1] == a and seq_b[0] == b:
            merged = seq_a + seq_b
        elif seq_b[-1] == b and seq_a[0] == a:
            merged = seq_b + seq_a
        elif seq_a[-1] == a and seq_b[-1] == b:
            merged = seq_a + list(reversed(seq_b))
        elif seq_a[0] == a and seq_b[0] == b:
            merged = list(reversed(seq_a)) + seq_b
        if merged is None:
            continue

        if not evaluate_route(problem, oracle, reference_vehicle, merged).feasible:
            continue

        routes[ra] = merged
        for sid in seq_b:
            route_of[sid] = ra
        del routes[rb]

    return [seq for seq in routes.values() if seq]


def _two_opt(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    seq: List[str],
    deadline: float,
) -> List[str]:
    """Intra-route 2-opt with first-improvement acceptance."""
    if len(seq) < 4:
        return seq
    best = list(seq)
    best_cost = _route_cost(problem, oracle, vehicle, best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            if time.monotonic() > deadline:
                return best
            for j in range(i + 2, len(best)):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                cost = _route_cost(problem, oracle, vehicle, candidate)
                if cost < best_cost - 1e-9:
                    best, best_cost = candidate, cost
                    improved = True
                    break
            if improved:
                break
    return best


def _or_opt(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    seq: List[str],
    deadline: float,
) -> List[str]:
    """Relocate segments of 1-3 consecutive stops inside the same route."""
    best = list(seq)
    best_cost = _route_cost(problem, oracle, vehicle, best)
    improved = True
    while improved:
        improved = False
        for size in (1, 2, 3):
            for i in range(len(best) - size + 1):
                if time.monotonic() > deadline:
                    return best
                segment = best[i : i + size]
                rest = best[:i] + best[i + size :]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    candidate = rest[:j] + segment + rest[j:]
                    cost = _route_cost(problem, oracle, vehicle, candidate)
                    if cost < best_cost - 1e-9:
                        best, best_cost = candidate, cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def _inter_route_relocate(
    problem: FleetProblem,
    oracle: DistanceOracle,
    routes: List[Route],
    deadline: float,
) -> Tuple[List[Route], int]:
    """Move single stops between routes when the total cost drops."""
    moves = 0
    sequences = {r.vehicle_id: list(r.stop_ids) for r in routes}
    costs = {r.vehicle_id: (r.cost if r.feasible else INFEASIBLE_COST) for r in routes}
    out_of_time = False

    for source in routes:
        if out_of_time:
            break
        for stop_id in list(sequences[source.vehicle_id]):
            if time.monotonic() > deadline:
                out_of_time = True
                break
            src_seq = sequences[source.vehicle_id]
            if stop_id not in src_seq:
                continue
            trimmed = [s for s in src_seq if s != stop_id]
            src_vehicle = problem.vehicle(source.vehicle_id)
            new_src_cost = _route_cost(problem, oracle, src_vehicle, trimmed)
            if new_src_cost == INFEASIBLE_COST:
                continue
            gain = costs[source.vehicle_id] - new_src_cost

            best_target = None
            best_delta = -1e-9
            for target in routes:
                if target.vehicle_id == source.vehicle_id:
                    continue
                tgt_vehicle = problem.vehicle(target.vehicle_id)
                tgt_seq = sequences[target.vehicle_id]
                for insert_at in range(len(tgt_seq) + 1):
                    candidate = tgt_seq[:insert_at] + [stop_id] + tgt_seq[insert_at:]
                    cost = _route_cost(problem, oracle, tgt_vehicle, candidate)
                    if cost == INFEASIBLE_COST:
                        continue
                    delta = gain - (cost - costs[target.vehicle_id])
                    if delta > best_delta:
                        best_delta = delta
                        best_target = (target.vehicle_id, candidate, cost)

            if best_target is not None:
                tgt_id, tgt_seq, tgt_cost = best_target
                sequences[source.vehicle_id] = trimmed
                costs[source.vehicle_id] = new_src_cost
                sequences[tgt_id] = tgt_seq
                costs[tgt_id] = tgt_cost
                moves += 1

    rebuilt = [
        evaluate_route(problem, oracle, problem.vehicle(vid), seq)
        for vid, seq in sequences.items()
    ]
    return rebuilt, moves


def _eliminate_routes(
    problem: FleetProblem,
    oracle: DistanceOracle,
    routes: List[Route],
    deadline: float,
) -> Tuple[List[Route], int]:
    """Dissolve whole routes when redistributing their stops is cheaper.

    2-opt, Or-opt and relocate are all *move* operators: they shuffle stops
    between routes but can never reduce the route count, so a truck carrying
    a handful of units keeps its ``fixed_cost`` forever. This sweep tries the
    one thing they cannot -- emptying a truck completely -- and keeps the
    result only when the total cost, ``fixed_cost`` included, actually drops.
    """
    removed = 0
    sequences = {r.vehicle_id: list(r.stop_ids) for r in routes}
    costs = {r.vehicle_id: (r.cost if r.feasible else INFEASIBLE_COST) for r in routes}

    # Cheapest routes first: the lightest truck is the one most likely to be
    # absorbable by its neighbours.
    order = sorted(
        (r.vehicle_id for r in routes if sequences[r.vehicle_id]),
        key=lambda vid: sum(problem.stop(s).demand for s in sequences[vid]),
    )

    for victim in order:
        if time.monotonic() > deadline:
            break
        stops = list(sequences[victim])
        if not stops:
            continue

        trial_seq = {vid: list(seq) for vid, seq in sequences.items()}
        trial_cost = dict(costs)
        trial_seq[victim] = []
        trial_cost[victim] = 0.0

        placed = True
        for stop_id in stops:
            best = None
            best_delta = INFEASIBLE_COST
            for vid, seq in trial_seq.items():
                if vid == victim:
                    continue
                vehicle = problem.vehicle(vid)
                for insert_at in range(len(seq) + 1):
                    candidate = seq[:insert_at] + [stop_id] + seq[insert_at:]
                    cost = _route_cost(problem, oracle, vehicle, candidate)
                    if cost == INFEASIBLE_COST:
                        continue
                    delta = cost - trial_cost[vid]
                    if delta < best_delta:
                        best_delta = delta
                        best = (vid, candidate, cost)
            if best is None:
                placed = False
                break
            vid, candidate, cost = best
            trial_seq[vid] = candidate
            trial_cost[vid] = cost

        if not placed:
            continue
        if sum(trial_cost.values()) < sum(costs.values()) - 1e-9:
            sequences = trial_seq
            costs = trial_cost
            removed += 1

    rebuilt = [
        evaluate_route(problem, oracle, problem.vehicle(vid), seq)
        for vid, seq in sequences.items()
        if seq
    ]
    return rebuilt, removed


def sequence_stops(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    stop_ids: Sequence[str],
    *,
    time_budget_s: float = 3.0,
) -> List[str]:
    """Order an unordered stop set into a good tour for one vehicle.

    Used by the quantum solvers, which decide *assignment* on the QPU and
    delegate intra-route sequencing back to this classical improver.
    """
    if len(stop_ids) < 2:
        return list(stop_ids)
    deadline = time.monotonic() + max(time_budget_s, 0.05)

    # Nearest-neighbour construction seeded at the depot, then improved.
    remaining = list(stop_ids)
    here = vehicle.depot_id
    seq: List[str] = []
    while remaining:
        nxt = min(remaining, key=lambda sid: oracle.km(here, sid))
        seq.append(nxt)
        remaining.remove(nxt)
        here = nxt

    seq = _or_opt(problem, oracle, vehicle, seq, deadline)
    seq = _two_opt(problem, oracle, vehicle, seq, deadline)
    return seq


def solve_classical(
    problem: FleetProblem,
    *,
    oracle: DistanceOracle,
    stop_ids: Optional[Sequence[str]] = None,
    vehicles: Optional[Sequence[Vehicle]] = None,
    neighbours: int = DEFAULT_NEIGHBOURS,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    local_search: bool = True,
) -> Tuple[List[Route], List[str], Dict[str, Any]]:
    """Run the classical baseline.

    Returns ``(routes, unassigned_stop_ids, diagnostics)``. ``diagnostics``
    carries the per-phase objective trace the UI plots as the convergence
    chart, so classical and quantum runs render the same artifact.
    """
    started = time.monotonic()
    deadline = started + max(time_budget_s, 0.5)

    pool = list(vehicles) if vehicles is not None else problem.usable_vehicles
    by_depot, unassigned = assign_stops_to_depots(problem, oracle, stop_ids)

    vehicles_by_depot: Dict[str, List[Vehicle]] = {}
    for vehicle in pool:
        vehicles_by_depot.setdefault(vehicle.depot_id, []).append(vehicle)
    for bucket in vehicles_by_depot.values():
        bucket.sort(key=lambda v: (-v.capacity, v.id))

    routes: List[Route] = []
    trace: List[Dict[str, Any]] = []

    for depot_id, depot_stops in by_depot.items():
        depot_vehicles = vehicles_by_depot.get(depot_id, [])
        if not depot_stops:
            continue
        if not depot_vehicles:
            unassigned.extend(depot_stops)
            continue

        reference = depot_vehicles[0]
        sequences = _savings_routes(
            problem,
            oracle,
            depot_id,
            reference,
            depot_stops,
            neighbours=neighbours,
            deadline=deadline,
        )
        # Heaviest routes get the biggest trucks; overflow routes are dropped
        # to the unassigned pool and picked up by the relocate pass below.
        sequences.sort(key=lambda seq: -sum(problem.stop(s).demand for s in seq))
        for idx, seq in enumerate(sequences):
            if idx < len(depot_vehicles):
                routes.append(
                    evaluate_route(problem, oracle, depot_vehicles[idx], seq)
                )
            else:
                unassigned.extend(seq)

    trace.append(
        {
            "phase": "savings",
            "objective": round(sum(r.cost for r in routes), 3),
            "routes": len(routes),
            "unassigned": len(unassigned),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    )

    if local_search and routes:
        improved: List[Route] = []
        for route in routes:
            if time.monotonic() > deadline:
                improved.append(route)
                continue
            vehicle = problem.vehicle(route.vehicle_id)
            seq = _or_opt(problem, oracle, vehicle, list(route.stop_ids), deadline)
            seq = _two_opt(problem, oracle, vehicle, seq, deadline)
            improved.append(evaluate_route(problem, oracle, vehicle, seq))
        routes = improved
        trace.append(
            {
                "phase": "intra_route_local_search",
                "objective": round(sum(r.cost for r in routes), 3),
                "routes": len(routes),
                "unassigned": len(unassigned),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )

        routes, moves = _inter_route_relocate(problem, oracle, routes, deadline)
        trace.append(
            {
                "phase": "inter_route_relocate",
                "objective": round(sum(r.cost for r in routes), 3),
                "routes": len([r for r in routes if not r.is_empty]),
                "unassigned": len(unassigned),
                "moves": moves,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )

        routes, eliminated = _eliminate_routes(problem, oracle, routes, deadline)
        trace.append(
            {
                "phase": "route_elimination",
                "objective": round(sum(r.cost for r in routes), 3),
                "routes": len([r for r in routes if not r.is_empty]),
                "unassigned": len(unassigned),
                "routes_removed": eliminated,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )

    # Last chance: try to place still-unassigned stops on any feasible route.
    unassigned, inserted = _insert_unassigned(
        problem, oracle, routes, unassigned, deadline
    )

    diagnostics = {
        "algorithm": "clarke_wright_savings+2opt+oropt",
        "objective_trace": trace,
        "neighbours": neighbours,
        "late_insertions": inserted,
        "elapsed_s": round(time.monotonic() - started, 3),
        "time_budget_s": time_budget_s,
        "hit_time_budget": time.monotonic() > deadline,
    }
    return [r for r in routes if not r.is_empty], unassigned, diagnostics


def _insert_unassigned(
    problem: FleetProblem,
    oracle: DistanceOracle,
    routes: List[Route],
    unassigned: List[str],
    deadline: float,
) -> Tuple[List[str], int]:
    """Cheapest-insertion sweep for stops no route picked up."""
    if not unassigned or not routes:
        return unassigned, 0

    still: List[str] = []
    inserted = 0
    by_vehicle = {r.vehicle_id: list(r.stop_ids) for r in routes}

    for stop_id in unassigned:
        if time.monotonic() > deadline:
            still.append(stop_id)
            continue
        best = None
        best_cost = INFEASIBLE_COST
        for vehicle_id, seq in by_vehicle.items():
            vehicle = problem.vehicle(vehicle_id)
            for position in range(len(seq) + 1):
                candidate = seq[:position] + [stop_id] + seq[position:]
                cost = _route_cost(problem, oracle, vehicle, candidate)
                if cost < best_cost:
                    best_cost = cost
                    best = (vehicle_id, candidate)
        if best is None:
            still.append(stop_id)
            continue
        by_vehicle[best[0]] = best[1]
        inserted += 1

    if inserted:
        rebuilt = {
            vid: evaluate_route(problem, oracle, problem.vehicle(vid), seq)
            for vid, seq in by_vehicle.items()
        }
        for index, route in enumerate(routes):
            routes[index] = rebuilt[route.vehicle_id]

    return still, inserted
