"""Fleet solve orchestration: method catalogue, execution and artifacts.

This is the fleet analogue of the portfolio executor layer. It owns:

* ``FLEET_METHODS`` — the method x paradigm x platform catalogue the UI
  and the API both read, so there is one source of truth.
* :func:`solve_fleet_problem` — runs a method, optionally runs the
  classical baseline alongside it for a like-for-like comparison, and
  emits ``execution_steps`` in the exact shape Job History already
  renders for portfolio runs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.diagnostics.vendor_links import vendor_link_artifacts_from_runs
from core.optimizers.quantum.shared.circuit_runner import runs_on_vendor_cloud
from core.optimizers.fleet.classical_solver import solve_classical
from core.optimizers.fleet.evaluate import build_oracle, finalise_solution
from core.optimizers.fleet.geo import DistanceOracle
from core.optimizers.fleet.models import FleetProblem, FleetSolution, Route
from core.optimizers.fleet.quantum_solver import solve_quantum

logger = logging.getLogger(__name__)

#: Diagnostics keys rendered as artifacts or charts rather than raw detail rows.
_NON_DETAIL_DIAGNOSTICS = frozenset(
    {"energy_trace", "objective_trace", "batches", "vendor_runs"}
)

#: Entitlement keys the fleet endpoints enforce.
VERTICAL_ENTITLEMENT = "fleet.routing"
QUANTUM_ENTITLEMENT = "fleet.quantum"


#: Method catalogue. ``backend`` selects the sampler in
#: :mod:`core.optimizers.fleet.samplers`; ``paradigm`` and ``platforms``
#: line up with the finance vertical's vocabulary so the UI can reuse its
#: provider chips. ``entitlement`` is the key the endpoint enforces.
FLEET_METHODS: Dict[str, Dict[str, Any]] = {
    "classical": {
        "label": "Savings + Local Search",
        "paradigm": "classical",
        "family": "classical",
        "backend": None,
        "platforms": ["cpu"],
        "execution_modes": ["classical"],
        "baseline": True,
        "description": (
            "Clarke-Wright savings construction refined by 2-opt, Or-opt and "
            "inter-route relocation. The comparison baseline."
        ),
        "entitlement": VERTICAL_ENTITLEMENT,
    },
    "simulated-annealing": {
        "label": "Simulated Annealing (quantum-inspired)",
        "paradigm": "classical",
        "family": "quantum_inspired",
        "backend": "numpy",
        "platforms": ["cpu"],
        "execution_modes": ["simulator"],
        "baseline": False,
        "description": (
            "Cluster-to-vehicle assignment QUBO annealed on CPU. Same "
            "encoding the annealer runs, without the device."
        ),
        "entitlement": QUANTUM_ENTITLEMENT,
    },
    "qubo-annealing": {
        "label": "QUBO Annealing",
        "paradigm": "dwave-quantum",
        "family": "annealing",
        "backend": "dwave",
        "platforms": ["dwave"],
        "execution_modes": ["simulator", "real"],
        "baseline": False,
        "description": (
            "Cluster-to-vehicle assignment QUBO solved with D-Wave Ocean: "
            "simulated annealing in simulator mode, Leap hybrid on hardware."
        ),
        "entitlement": QUANTUM_ENTITLEMENT,
    },
    "qaoa": {
        "label": "QAOA (gate-based)",
        "paradigm": "ibm-quantum",
        "family": "gate",
        "backend": "gate",
        "platforms": ["ibm", "ionq"],
        "execution_modes": ["simulator", "real"],
        "baseline": False,
        "description": (
            "The same assignment QUBO mapped to an Ising Hamiltonian and "
            "sampled with QAOA. Parameters train on a simulator; in real "
            "mode the final circuit is measured on an IBM or IonQ device. "
            "Small instances only."
        ),
        "entitlement": QUANTUM_ENTITLEMENT,
    },
}

DEFAULT_METHOD = "classical"

#: Entitlement key required for any real-hardware fleet submit.
HARDWARE_ENTITLEMENT = "solver.quantum_hardware"


def list_methods() -> List[Dict[str, Any]]:
    """Method catalogue for the UI picker."""
    return [
        {
            "method": key,
            "label": spec["label"],
            "paradigm": spec["paradigm"],
            "family": spec["family"],
            "platforms": list(spec["platforms"]),
            "execution_modes": list(spec["execution_modes"]),
            "baseline": spec["baseline"],
            "description": spec["description"],
            "entitlement": spec["entitlement"],
            "default": key == DEFAULT_METHOD,
        }
        for key, spec in FLEET_METHODS.items()
    ]


def resolve_method(method: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Validate a method name and return ``(method, spec)``."""
    key = (method or DEFAULT_METHOD).strip().lower()
    if key not in FLEET_METHODS:
        raise ValueError(
            f"unknown fleet method '{method}'; expected one of {sorted(FLEET_METHODS)}"
        )
    return key, FLEET_METHODS[key]


def build_solver(
    method: str,
    *,
    platform: str,
    execution_mode: str,
    config: Optional[Dict[str, Any]] = None,
) -> Callable[..., Tuple[List[Route], List[str], Dict[str, Any]]]:
    """Return a uniform ``solver(problem, oracle=..., ...)`` callable.

    The disruption replanner reuses this so a replan runs on exactly the
    same solver the original plan used.
    """
    key, spec = resolve_method(method)
    cfg = dict(config or {})

    if spec["backend"] is None:
        def _classical(problem, **kwargs):
            return solve_classical(
                problem,
                neighbours=int(cfg.get("neighbours", 16)),
                time_budget_s=float(cfg.get("time_budget_s", 20.0)),
                local_search=bool(cfg.get("local_search", True)),
                **kwargs,
            )

        return _classical

    sampler_options = {
        k: v
        for k, v in {
            "shots": cfg.get("shots"),
            "reads": cfg.get("reads"),
            "reps": cfg.get("reps"),
            "sweeps": cfg.get("sweeps"),
            "restarts": cfg.get("restarts"),
        }.items()
        if v is not None
    }

    def _quantum(problem, **kwargs):
        return solve_quantum(
            problem,
            backend=spec["backend"],
            platform=platform,
            execution_mode=execution_mode,
            max_vars=int(cfg.get("max_qubo_vars", 512)),
            vehicles_per_batch=int(cfg.get("vehicles_per_batch", 8)),
            clusters_per_vehicle=int(cfg.get("clusters_per_vehicle", 3)),
            classical_polish=bool(cfg.get("classical_polish", True)),
            sampler_options=sampler_options,
            seed=int(cfg.get("seed", 42)),
            **kwargs,
        )

    _quantum.__name__ = f"fleet_{key.replace('-', '_')}"
    return _quantum


def _step(
    name: str,
    description: str,
    stage: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "step": name,
        "name": name,
        "description": description,
        "stage": stage,
        "status": "completed",
        "details": details or {},
        "artifacts": artifacts or [],
        "errors": [],
        "warnings": warnings or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _artifact(
    name: str,
    kind: str,
    title: str,
    spec: Dict[str, Any],
    *,
    stage: str = "post",
    fmt: str = "chartjs",
) -> Dict[str, Any]:
    """Wrap a chart spec in the envelope ``StepArtifact.jsx`` expects.

    The renderer keys off ``format``/``spec``; a bare Chart.js object
    (``{type, data, options}``) falls through to "No renderable content".
    """
    return {
        "name": name,
        "kind": kind,
        "title": title,
        "stage": stage,
        "format": fmt,
        "spec": spec,
    }


def _convergence_artifact(diagnostics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Line chart of solver progress — energy for QUBO, cost for classical."""
    energies = diagnostics.get("energy_trace") or []
    if energies:
        return _artifact(
            "cost_vs_iter",
            "convergence",
            "QUBO energy per sample",
            {
                "type": "line",
                "data": {
                    "labels": [str(i) for i in range(len(energies))],
                    "datasets": [
                        {
                            "label": "QUBO energy (best so far)",
                            "data": [float(e) for e in energies],
                            "borderColor": "#7c3aed",
                            "backgroundColor": "rgba(124, 58, 237, 0.15)",
                            "tension": 0.25,
                            "pointRadius": 0,
                        }
                    ],
                },
                "options": {
                    "scales": {"y": {"title": {"display": True, "text": "Energy"}}}
                },
            },
            stage="process",
        )

    trace = diagnostics.get("objective_trace") or []
    if not trace:
        return None
    return _artifact(
        "cost_vs_iter",
        "convergence",
        "Plan cost per optimisation phase",
        {
            "type": "line",
            "data": {
                "labels": [str(entry.get("phase", i)) for i, entry in enumerate(trace)],
                "datasets": [
                    {
                        "label": "Plan cost",
                        "data": [float(entry.get("objective", 0.0)) for entry in trace],
                        "borderColor": "#10b981",
                        "backgroundColor": "rgba(16, 185, 129, 0.15)",
                        "tension": 0.25,
                    }
                ],
            },
            "options": {"scales": {"y": {"title": {"display": True, "text": "Cost"}}}},
        },
        stage="process",
    )


def _route_map_artifact(problem: FleetProblem, solution: FleetSolution) -> Dict[str, Any]:
    """Geographic payload consumed by the fleet map renderer."""
    stop_ids = {
        visit.stop_id for route in solution.routes for visit in route.visits
    }
    stop_ids.update(solution.unassigned_stop_ids)
    return _artifact(
        "route_map",
        "route-map",
        "Dispatch plan",
        {
            "average_speed_kmh": problem.average_speed_kmh,
            "traffic_factor": problem.traffic_factor,
            "depots": [d.to_dict() for d in problem.depots],
            "stops": [problem.stop(sid).to_dict() for sid in sorted(stop_ids)],
            "routes": [r.to_dict() for r in solution.routes],
            "unassigned_stop_ids": list(solution.unassigned_stop_ids),
        },
        fmt="route-map",
    )


def _route_load_artifact(solution: FleetSolution) -> Dict[str, Any]:
    routes = solution.routes[:40]
    return _artifact(
        "route_load",
        "utilisation",
        "Load and stop count per vehicle",
        {
            "type": "bar",
            "data": {
                "labels": [r.vehicle_id for r in routes],
                "datasets": [
                    {
                        "label": "Load",
                        "data": [round(r.load, 2) for r in routes],
                        "backgroundColor": "rgba(102, 126, 234, 0.7)",
                    },
                    {
                        "label": "Stops",
                        "data": [len(r.visits) for r in routes],
                        "backgroundColor": "rgba(16, 185, 129, 0.7)",
                    },
                ],
            },
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
    )


def _comparison_artifact(
    solution: FleetSolution, baseline: FleetSolution
) -> Dict[str, Any]:
    labels = ["Distance (km)", "Cost", "Routes", "Unassigned"]
    return _artifact(
        "baseline_comparison",
        "comparison",
        "This run vs the classical baseline",
        {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": solution.method,
                        "data": [
                            solution.metrics.get("total_distance_km", 0.0),
                            solution.metrics.get("total_cost", 0.0),
                            solution.metrics.get("routes_used", 0),
                            solution.metrics.get("stops_unassigned", 0),
                        ],
                        "backgroundColor": "rgba(124, 58, 237, 0.7)",
                    },
                    {
                        "label": f"{baseline.method} (baseline)",
                        "data": [
                            baseline.metrics.get("total_distance_km", 0.0),
                            baseline.metrics.get("total_cost", 0.0),
                            baseline.metrics.get("routes_used", 0),
                            baseline.metrics.get("stops_unassigned", 0),
                        ],
                        "backgroundColor": "rgba(107, 114, 128, 0.7)",
                    },
                ],
            },
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
    )


def _replan_delta_artifact(
    before: Dict[str, Any], after: Dict[str, Any]
) -> Dict[str, Any]:
    keys = [
        ("total_distance_km", "Distance (km)"),
        ("total_cost", "Cost"),
        ("routes_used", "Routes"),
        ("stops_unassigned", "Unserved"),
    ]
    return _artifact(
        "replan_delta",
        "comparison",
        "Pre-event plan vs recovered plan",
        {
            "type": "bar",
            "data": {
                "labels": [label for _, label in keys],
                "datasets": [
                    {
                        "label": "Before disruption",
                        "data": [float(before.get(k, 0) or 0) for k, _ in keys],
                        "backgroundColor": "rgba(107, 114, 128, 0.7)",
                    },
                    {
                        "label": "After replan",
                        "data": [float(after.get(k, 0) or 0) for k, _ in keys],
                        "backgroundColor": "rgba(124, 58, 237, 0.7)",
                    },
                ],
            },
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
    )


def _disposition_artifact(replan: Dict[str, Any]) -> Dict[str, Any]:
    """Where every stop ended up once the disruption was absorbed."""
    delivered = len(replan.get("completed_stops") or [])
    reassigned = int((replan.get("delta") or {}).get("stops_reassigned", 0) or 0)
    unserved = int(
        ((replan.get("solution") or {}).get("metrics") or {}).get("stops_unassigned", 0) or 0
    )
    held = max((replan.get("open_stops") or 0) - reassigned - unserved, 0)
    return _artifact(
        "stop_disposition",
        "disposition",
        "Stop disposition after the event",
        {
            "type": "pie",
            "data": {
                "labels": [
                    "Delivered before event",
                    "Moved to another vehicle",
                    "Kept original vehicle",
                    "Could not be served",
                ],
                "datasets": [
                    {
                        "data": [delivered, reassigned, held, unserved],
                        "backgroundColor": [
                            "rgba(16, 185, 129, 0.75)",
                            "rgba(124, 58, 237, 0.75)",
                            "rgba(102, 126, 234, 0.75)",
                            "rgba(239, 68, 68, 0.75)",
                        ],
                    }
                ],
            },
            "options": {},
        },
    )


def _route_map_artifact_from_dicts(
    problem: Dict[str, Any], solution: Dict[str, Any]
) -> Dict[str, Any]:
    """``_route_map_artifact`` for the replan path, which only has dicts."""
    routes = solution.get("routes") or []
    keep = {sid for route in routes for sid in (route.get("stop_ids") or [])}
    keep.update(solution.get("unassigned_stop_ids") or [])
    return _artifact(
        "route_map",
        "route-map",
        "Recovered dispatch plan",
        {
            "average_speed_kmh": problem.get("average_speed_kmh"),
            "traffic_factor": problem.get("traffic_factor"),
            "depots": problem.get("depots") or [],
            "stops": [s for s in (problem.get("stops") or []) if s.get("id") in keep],
            "routes": routes,
            "unassigned_stop_ids": list(solution.get("unassigned_stop_ids") or []),
        },
        fmt="route-map",
    )


def build_replan_steps(
    replan: Dict[str, Any], *, method: str = DEFAULT_METHOD
) -> List[Dict[str, Any]]:
    """Execution steps for a disruption replan.

    A replan is a real solve, so Job History should show the same
    three-stage narrative a fresh optimise gets rather than an empty list.
    """
    event = replan.get("event") or {}
    impact = replan.get("impact") or {}
    delta = replan.get("delta") or {}
    solution = replan.get("solution") or {}
    problem = replan.get("problem") or {}
    metrics = solution.get("metrics") or {}
    baseline_metrics = replan.get("baseline_metrics") or {}
    diagnostics = solution.get("diagnostics") or {}
    label = FLEET_METHODS.get(method, {}).get("label", method)

    solve_artifacts = []
    convergence = _convergence_artifact(diagnostics)
    if convergence:
        solve_artifacts.append(convergence)
    solve_artifacts.extend(
        vendor_link_artifacts_from_runs(diagnostics.get("vendor_runs"))
    )

    unserved = int(metrics.get("stops_unassigned", 0) or 0)
    return [
        _step(
            "disruption_intake",
            f"Register the {str(event.get('kind', 'disruption')).replace('_', ' ')} "
            "and lock in completed deliveries",
            "pre",
            details={
                "kind": event.get("kind"),
                "at_min": replan.get("at_min"),
                "target_id": event.get("target_id"),
                "magnitude": event.get("magnitude"),
                "summary": impact.get("summary"),
                "delivered_before_event": len(replan.get("completed_stops") or []),
                "open_stops": replan.get("open_stops"),
                "vehicles_available": replan.get("vehicles_available"),
            },
        ),
        _step(
            "recovery_solve",
            f"Re-optimise the open network with {label}",
            "process",
            details={
                "method": method,
                "platform": solution.get("platform"),
                "execution_mode": solution.get("execution_mode"),
                **{
                    k: v
                    for k, v in diagnostics.items()
                    if k not in _NON_DETAIL_DIAGNOSTICS
                },
            },
            artifacts=solve_artifacts,
        ),
        _step(
            "replan_delta",
            "Compare the recovered plan against the pre-event plan",
            "post",
            details={
                "stops_reassigned": delta.get("stops_reassigned"),
                "distance_delta_km": delta.get("distance_km"),
                "cost_delta": delta.get("cost"),
                "routes_delta": delta.get("routes_used"),
                "unassigned_delta": delta.get("unassigned"),
                **metrics,
            },
            artifacts=[
                _replan_delta_artifact(baseline_metrics, metrics),
                _disposition_artifact(replan),
                _route_map_artifact_from_dicts(problem, solution),
            ],
            warnings=(
                [f"{unserved} stops could not be served after the event"]
                if unserved
                else []
            ),
        ),
    ]


def compare_to_baseline(
    solution: FleetSolution, baseline: FleetSolution
) -> Dict[str, Any]:
    """Percentage deltas of a run against the classical baseline."""

    def _pct(key: str) -> float:
        base = float(baseline.metrics.get(key, 0.0))
        value = float(solution.metrics.get(key, 0.0))
        if abs(base) < 1e-9:
            return 0.0
        return round(100.0 * (value - base) / base, 2)

    return {
        "baseline_method": baseline.method,
        "baseline_metrics": baseline.metrics,
        "distance_delta_pct": _pct("total_distance_km"),
        "cost_delta_pct": _pct("total_cost"),
        "routes_delta": solution.metrics.get("routes_used", 0)
        - baseline.metrics.get("routes_used", 0),
        "unassigned_delta": solution.metrics.get("stops_unassigned", 0)
        - baseline.metrics.get("stops_unassigned", 0),
        "objective_delta": round(solution.objective - baseline.objective, 3),
        "wins": solution.objective < baseline.objective,
    }


def solve_fleet_problem(
    problem: FleetProblem,
    *,
    method: str = DEFAULT_METHOD,
    platform: Optional[str] = None,
    execution_mode: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    with_baseline: bool = True,
    oracle: Optional[DistanceOracle] = None,
) -> Dict[str, Any]:
    """Solve ``problem`` with ``method`` and return a job-shaped result.

    The return value is a plain dict so it can be persisted straight into
    ``OptimizationLog.optimization_result`` and re-rendered by the SPA:
    ``{solution, baseline, comparison, execution_steps, metrics, ...}``.
    """
    started = time.monotonic()
    key, spec = resolve_method(method)
    resolved_platform = (platform or spec["platforms"][0]).strip().lower()
    if resolved_platform not in spec["platforms"]:
        raise ValueError(
            f"platform '{resolved_platform}' is not supported by method '{key}'; "
            f"expected one of {spec['platforms']}"
        )
    resolved_mode = (execution_mode or spec["execution_modes"][0]).strip().lower()
    if resolved_mode not in spec["execution_modes"]:
        raise ValueError(
            f"execution_mode '{resolved_mode}' is not supported by method '{key}'; "
            f"expected one of {spec['execution_modes']}"
        )

    # Gate methods only dispatch on the vendor in `real` mode - or on a
    # vendor-hosted simulator the tenant can submit to with their own key.
    # Anything else executes locally whichever vendor was asked for, so
    # record where the work actually happened and never attribute a local
    # Qiskit result to IBM or IonQ.
    if spec.get("backend") == "gate" and not runs_on_vendor_cloud(
        resolved_platform, resolved_mode
    ):
        resolved_platform = "cpu"

    oracle = oracle or build_oracle(problem)
    steps: List[Dict[str, Any]] = [
        _step(
            "problem_setup",
            "Load the delivery network and build the distance model",
            "pre",
            details={
                "scenario": problem.name,
                "depots": len(problem.depots),
                "stops": len(problem.active_stops),
                "vehicles": len(problem.usable_vehicles),
                "service_radius_km": round(problem.service_radius_km, 1),
                "traffic_factor": problem.traffic_factor,
            },
        )
    ]

    solver = build_solver(
        key, platform=resolved_platform, execution_mode=resolved_mode, config=config
    )
    routes, unassigned, diagnostics = solver(problem, oracle=oracle)
    solution = finalise_solution(
        problem,
        routes,
        unassigned,
        method=key,
        platform=resolved_platform,
        execution_mode=resolved_mode,
        diagnostics=diagnostics,
    )

    solve_artifacts = []
    convergence = _convergence_artifact(diagnostics)
    if convergence:
        solve_artifacts.append(convergence)
    solve_artifacts.extend(
        vendor_link_artifacts_from_runs(diagnostics.get("vendor_runs"))
    )
    steps.append(
        _step(
            "solve",
            f"Optimise routes with {spec['label']}",
            "process",
            details={
                "method": key,
                "platform": resolved_platform,
                "execution_mode": resolved_mode,
                **{
                    k: v
                    for k, v in diagnostics.items()
                    if k not in _NON_DETAIL_DIAGNOSTICS
                },
            },
            artifacts=solve_artifacts,
            warnings=list(diagnostics.get("fallbacks") or []),
        )
    )

    baseline: Optional[FleetSolution] = None
    comparison: Optional[Dict[str, Any]] = None
    if with_baseline and key != DEFAULT_METHOD:
        base_routes, base_unassigned, base_diag = solve_classical(
            problem, oracle=oracle, time_budget_s=float((config or {}).get("baseline_time_budget_s", 10.0))
        )
        baseline = finalise_solution(
            problem,
            base_routes,
            base_unassigned,
            method=DEFAULT_METHOD,
            platform="cpu",
            execution_mode="classical",
            diagnostics=base_diag,
        )
        comparison = compare_to_baseline(solution, baseline)

    result_artifacts = [
        _route_map_artifact(problem, solution),
        _route_load_artifact(solution),
    ]
    if baseline is not None:
        result_artifacts.append(_comparison_artifact(solution, baseline))

    steps.append(
        _step(
            "route_plan",
            "Assemble the dispatch plan and service KPIs",
            "post",
            details=solution.metrics,
            artifacts=result_artifacts,
            warnings=(
                [f"{len(solution.unassigned_stop_ids)} stops could not be served"]
                if solution.unassigned_stop_ids
                else []
            ),
        )
    )

    return {
        "vertical": "fleet_routing",
        "scenario": problem.name,
        "method": key,
        "platform": resolved_platform,
        "execution_mode": resolved_mode,
        "solution": solution.to_dict(),
        "baseline": baseline.to_dict() if baseline else None,
        "comparison": comparison,
        "metrics": solution.metrics,
        "objective": solution.objective,
        "problem": problem.to_dict(),
        "execution_steps": steps,
        "execution_info": {
            "backend_name": ",".join(diagnostics.get("backends", []))
            or "cpu",
            "qubits_used": diagnostics.get("qubo_max_binaries", 0),
            "qubo_batches": diagnostics.get("qubo_batches", 0),
            "duration_seconds": round(time.monotonic() - started, 3),
        },
    }


def solve_scenario(
    *,
    scenario: str,
    method: str = DEFAULT_METHOD,
    platform: Optional[str] = None,
    execution_mode: Optional[str] = None,
    seed: int = 42,
    overrides: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    with_baseline: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper: build a preset scenario and solve it."""
    from core.optimizers.fleet.scenario import build_scenario  # noqa: PLC0415

    problem = build_scenario(scenario, seed=seed, **(overrides or {}))
    return solve_fleet_problem(
        problem,
        method=method,
        platform=platform,
        execution_mode=execution_mode,
        config=config,
        with_baseline=with_baseline,
    )
