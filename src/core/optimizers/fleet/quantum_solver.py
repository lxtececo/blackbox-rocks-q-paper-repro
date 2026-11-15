"""Quantum-assisted fleet solver (cluster-first / route-second).

The QUBO decides the globally-coupled part — which truck serves which
micro-cluster — and the classical improver sequences each truck's stops.
Large fleets are handled by **batched decomposition**: the depot partition
and the vehicle/cluster batching keep every individual QUBO inside the
width a real device (or a simulator) can actually take, while the plan as
a whole still covers all 600+ delivery points.

Every batch is a genuine QUBO solve; the diagnostics block reports how
many were issued, the total binaries and the per-batch energies so the UI
can show the quantum workload honestly rather than implying one giant
QPU call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.optimizers.fleet.classical_solver import (
    _insert_unassigned,
    _inter_route_relocate,
    _eliminate_routes,
    assign_stops_to_depots,
    sequence_stops,
)
from core.optimizers.fleet.evaluate import evaluate_route
from core.optimizers.fleet.geo import DistanceOracle
from core.optimizers.fleet.models import FleetProblem, Route, Vehicle
from core.optimizers.fleet.qubo import (
    MAX_QUBO_VARS,
    TARGET_FILL,
    build_clusters,
    build_cluster_qubo,
)
from core.optimizers.fleet.samplers import (
    MAX_GATE_QUBITS,
    MAX_REAL_GATE_QUBITS,
    sample_qubo,
)

logger = logging.getLogger(__name__)

#: Vehicles considered in a single QUBO batch. Small enough that the
#: cluster axis stays wide (assignment freedom is what we want on the
#: device), large enough that batches genuinely compete for load.
DEFAULT_VEHICLES_PER_BATCH = 8

#: Micro-clusters offered per vehicle inside a batch. More clusters means
#: finer assignment granularity on the device at the cost of QUBO width.
DEFAULT_CLUSTERS_PER_VEHICLE = 3

#: Assignment freedom a gate batch must keep. One cluster per vehicle makes
#: the QUBO trivial - there is nothing to choose - so a batch narrow enough
#: to fit the qubit cap only at that ratio is not worth a circuit.
MIN_GATE_CLUSTERS_PER_VEHICLE = 2


def _gate_batch_geometry(
    max_vars: int, vehicles_per_batch: int, execution_mode: str
) -> Tuple[int, int]:
    """Clamp ``(max_vars, vehicles_per_batch)`` to what QAOA can actually run.

    The batching layer exists so each QUBO stays inside the width a device or
    a simulator can take, but it used to size batches purely from capacity and
    then hand a 52-binary QUBO to a sampler capped at 20. In simulator mode
    that fell back to CPU annealing on every batch; in real mode it raised.
    Deriving the width from the cap keeps the scenario unbounded - it just
    costs more batches - which is the whole point of decomposing.
    """
    hardware = (execution_mode or "").strip().lower() == "real"
    cap = MAX_REAL_GATE_QUBITS if hardware else MAX_GATE_QUBITS
    # n_c * n_v + n_v <= cap, so n_v alone costs a binary per vehicle.
    max_vehicles = max(1, cap // (MIN_GATE_CLUSTERS_PER_VEHICLE + 1))
    return min(max_vars, cap), min(vehicles_per_batch, max_vehicles)


def _repair_route(
    problem: FleetProblem,
    oracle: DistanceOracle,
    vehicle: Vehicle,
    stop_ids: List[str],
    *,
    sequence_budget_s: float,
) -> Tuple[Route, List[str]]:
    """Sequence a stop set and shed stops until the route is feasible."""
    dropped: List[str] = []
    remaining = list(stop_ids)

    while remaining:
        seq = sequence_stops(
            problem, oracle, vehicle, remaining, time_budget_s=sequence_budget_s
        )
        route = evaluate_route(problem, oracle, vehicle, seq)
        if route.feasible:
            return route, dropped

        # Distance-optimal ordering can violate time windows. Try the
        # classic earliest-deadline-first ordering before shedding work.
        edf = sorted(
            remaining,
            key=lambda sid: (problem.stop(sid).tw_end, problem.stop(sid).tw_start),
        )
        edf_route = evaluate_route(problem, oracle, vehicle, edf)
        if edf_route.feasible:
            return edf_route, dropped

        # Shed the stop contributing the most detour; the classical
        # insertion sweep gets a chance to re-home it afterwards.
        worst = max(route.visits, key=lambda v: v.leg_km + v.late_min)
        remaining = [s for s in remaining if s != worst.stop_id]
        dropped.append(worst.stop_id)

    return Route(vehicle_id=vehicle.id, depot_id=vehicle.depot_id), dropped


def _serve_with_spare_vehicles(
    problem: FleetProblem,
    oracle: DistanceOracle,
    spare: Sequence[Vehicle],
    stop_ids: Sequence[str],
    *,
    sequence_budget_s: float,
) -> Tuple[List[str], List[Route]]:
    """Open new routes on unused trucks for stops no batch could place."""
    pending = list(stop_ids)
    extra: List[Route] = []
    available = list(spare)

    while pending and available:
        vehicle = available.pop(0)
        depot = problem.depot(vehicle.depot_id)
        if not depot.open:
            continue
        # Fill the truck with the nearest pending stops it can legally carry.
        ranked = sorted(pending, key=lambda sid: oracle.km(depot.id, sid))
        chosen: List[str] = []
        load = 0.0
        for sid in ranked:
            demand = problem.stop(sid).demand
            if load + demand > vehicle.capacity:
                continue
            chosen.append(sid)
            load += demand
        if not chosen:
            continue

        route, dropped = _repair_route(
            problem, oracle, vehicle, chosen, sequence_budget_s=sequence_budget_s
        )
        placed = set(route.stop_ids)
        if not placed:
            continue
        extra.append(route)
        pending = [sid for sid in pending if sid not in placed]
        _ = dropped  # dropped stops stay in ``pending`` for the next truck

    return pending, extra


def solve_quantum(
    problem: FleetProblem,
    *,
    oracle: DistanceOracle,
    backend: str = "numpy",
    platform: str = "cpu",
    execution_mode: str = "simulator",
    stop_ids: Optional[Sequence[str]] = None,
    vehicles: Optional[Sequence[Vehicle]] = None,
    max_vars: int = MAX_QUBO_VARS,
    vehicles_per_batch: int = DEFAULT_VEHICLES_PER_BATCH,
    clusters_per_vehicle: int = DEFAULT_CLUSTERS_PER_VEHICLE,
    sampler_options: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    sequence_budget_s: float = 1.0,
    classical_polish: bool = True,
    polish_budget_s: float = 5.0,
) -> Tuple[List[Route], List[str], Dict[str, Any]]:
    """Solve the fleet problem with a QUBO assignment layer.

    Returns ``(routes, unassigned_stop_ids, diagnostics)`` — the same
    triple as :func:`core.optimizers.fleet.classical_solver.solve_classical`
    so the service layer can treat every method uniformly.
    """
    started = time.monotonic()
    options = dict(sampler_options or {})
    if backend == "gate":
        max_vars, vehicles_per_batch = _gate_batch_geometry(
            max_vars, vehicles_per_batch, execution_mode
        )
    pool = list(vehicles) if vehicles is not None else problem.usable_vehicles
    by_depot, unassigned = assign_stops_to_depots(problem, oracle, stop_ids)

    vehicles_by_depot: Dict[str, List[Vehicle]] = {}
    for vehicle in pool:
        vehicles_by_depot.setdefault(vehicle.depot_id, []).append(vehicle)
    for bucket in vehicles_by_depot.values():
        bucket.sort(key=lambda v: (-v.capacity, v.id))

    routes: List[Route] = []
    batches: List[Dict[str, Any]] = []
    energy_trace: List[float] = []
    total_binaries = 0
    backend_names: List[str] = []
    # One vendor job per batch, so these accumulate rather than overwrite.
    vendor_runs: List[Dict[str, Any]] = []
    fallback_reasons: List[str] = []

    for depot_id, depot_stops in by_depot.items():
        depot_vehicles = list(vehicles_by_depot.get(depot_id, []))
        if not depot_stops:
            continue
        if not depot_vehicles:
            unassigned.extend(depot_stops)
            continue

        pending = list(depot_stops)
        while pending and depot_vehicles:
            batch_vehicles = depot_vehicles[: max(1, vehicles_per_batch)]
            # The encoding spends one binary per vehicle on the "truck is
            # open" indicator, so only `max_vars - n_v` is left for the
            # assignment block. Dividing by n_v alone overshoots by a cluster
            # and the batch is then rejected outright by build_cluster_qubo.
            n_v = len(batch_vehicles)
            width = max(1, (max_vars - n_v) // n_v)
            capacity = sum(v.capacity for v in batch_vehicles)

            # Size the batch so its demand roughly fills the trucks offered.
            demand_budget = capacity * TARGET_FILL
            batch_stops: List[str] = []
            running = 0.0
            for sid in pending:
                demand = problem.stop(sid).demand
                if batch_stops and running + demand > demand_budget:
                    break
                batch_stops.append(sid)
                running += demand
            if not batch_stops:
                batch_stops = pending[:1]

            n_clusters = min(
                width,
                max(1, len(batch_vehicles) * clusters_per_vehicle),
                len(batch_stops),
            )

            try:
                qubo = build_cluster_qubo(
                    problem,
                    oracle,
                    batch_stops,
                    batch_vehicles,
                    n_clusters=n_clusters,
                    max_vars=max_vars,
                    seed=seed + len(batches),
                )
            except ValueError as exc:
                logger.warning(
                    "[fleet-quantum] batch skipped at depot=%s: %s", depot_id, exc
                )
                unassigned.extend(batch_stops)
                pending = pending[len(batch_stops) :]
                continue

            outcome = sample_qubo(
                qubo,
                backend=backend,
                platform=platform,
                execution_mode=execution_mode,
                seed=seed + len(batches),
                **options,
            )
            total_binaries += qubo.n_vars
            backend_names.append(outcome.backend_name)
            vendor_runs.extend(outcome.extra.get("vendor_runs") or [])
            if outcome.fallback_reason:
                fallback_reasons.append(outcome.fallback_reason)
            energy_trace.extend(outcome.energy_trace)

            assignment, cluster_misses = qubo.decode(outcome.bits)
            unassigned.extend(cluster_misses)

            used_vehicle_ids: List[str] = []
            for vehicle in batch_vehicles:
                assigned = assignment.get(vehicle.id, [])
                if not assigned:
                    continue
                route, dropped = _repair_route(
                    problem,
                    oracle,
                    vehicle,
                    assigned,
                    sequence_budget_s=sequence_budget_s,
                )
                unassigned.extend(dropped)
                if not route.is_empty:
                    routes.append(route)
                    used_vehicle_ids.append(vehicle.id)

            batches.append(
                {
                    "depot_id": depot_id,
                    "clusters": len(qubo.clusters),
                    "vehicles": len(batch_vehicles),
                    "binaries": qubo.n_vars,
                    "energy": round(float(outcome.energy), 4),
                    "backend": outcome.backend_name,
                    "reads": outcome.reads,
                    "elapsed_ms": outcome.elapsed_ms,
                }
            )

            pending = pending[len(batch_stops) :]
            depot_vehicles = [
                v for v in depot_vehicles if v.id not in set(used_vehicle_ids)
            ]

        if pending:
            unassigned.extend(pending)

    deadline = time.monotonic() + 5.0
    unassigned, inserted = _insert_unassigned(
        problem, oracle, routes, unassigned, deadline
    )

    # Stops the QUBO batches could not place still have to be delivered.
    # Spin up trucks that no batch used before declaring anything unserved.
    spare_routes = 0
    if unassigned:
        used = {r.vehicle_id for r in routes}
        spare = [v for v in pool if v.id not in used]
        unassigned, extra = _serve_with_spare_vehicles(
            problem,
            oracle,
            spare,
            unassigned,
            sequence_budget_s=sequence_budget_s,
        )
        routes.extend(extra)
        spare_routes = len(extra)

    # Hybrid polish: production annealing pipelines always hand the device
    # output to a classical refiner. Reported in diagnostics so the UI can
    # state plainly that the plan is device-assigned, classically refined.
    polish_moves = 0
    routes_eliminated = 0
    if classical_polish and len(routes) > 1:
        deadline = time.monotonic() + max(polish_budget_s, 0.5)
        routes, polish_moves = _inter_route_relocate(
            problem,
            oracle,
            routes,
            deadline,
        )
        routes = [r for r in routes if not r.is_empty]

        # Relocation only moves stops between trucks, so on its own it can
        # never retire one. Batching hands each batch its full vehicle slice,
        # so without this sweep the plan is pinned at one route per vehicle
        # no matter what the device returned.
        routes, routes_eliminated = _eliminate_routes(
            problem, oracle, routes, deadline
        )
        routes = [r for r in routes if not r.is_empty]

        # The polish pass frees slack; give the leftovers one more shot.
        if unassigned:
            unassigned, late = _insert_unassigned(
                problem, oracle, routes, unassigned, time.monotonic() + 3.0
            )
            inserted += late

    diagnostics = {
        "algorithm": "cluster_qubo_assignment+classical_sequencing",
        "classical_polish": bool(classical_polish),
        "polish_moves": polish_moves,
        "routes_eliminated": routes_eliminated,
        "spare_vehicle_routes": spare_routes,
        "qubo_batches": len(batches),
        "qubo_total_binaries": total_binaries,
        "qubo_max_binaries": max((b["binaries"] for b in batches), default=0),
        "batches": batches,
        "energy_trace": [round(float(e), 6) for e in energy_trace[:400]],
        "backends": sorted(set(backend_names)),
        "vendor_runs": vendor_runs,
        "fallbacks": sorted(set(fallback_reasons)),
        "late_insertions": inserted,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    return [r for r in routes if not r.is_empty], unassigned, diagnostics


def preview_clusters(    problem: FleetProblem,
    oracle: DistanceOracle,
    stop_ids: Sequence[str],
    *,
    n_clusters: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Cluster preview for the UI's 'what the QPU actually sees' panel."""
    return [
        cluster.to_dict()
        for cluster in build_clusters(
            problem, oracle, stop_ids, n_clusters=n_clusters, seed=seed
        )
    ]
