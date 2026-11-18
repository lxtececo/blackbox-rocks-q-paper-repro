"""Network-design solve orchestration: catalogue, execution, artifacts (F32).

The facility-location analogue of ``fleet/service.py``. Owns the method
catalogue both the UI and the API read, runs a method alongside the classical
baseline for a like-for-like comparison, and emits ``execution_steps`` in the
shape Job History already renders - including the server-authored
``explanation`` / ``field_notes`` pair introduced with F31.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.diagnostics.vendor_links import vendor_link_artifacts_from_runs
from core.optimizers.quantum.shared.circuit_runner import runs_on_vendor_cloud
from core.optimizers.network.classical import solve_classical
from core.optimizers.network.evaluate import compare_solutions
from core.optimizers.network.exact import DEFAULT_TIME_LIMIT_S, exact_reference, solve_exact
from core.optimizers.network.models import NetworkProblem, NetworkSolution
from core.optimizers.network.qubo import build_network_qubo, qubo_summary
from core.optimizers.network.quantum_solver import solve_quantum
from core.optimizers.fleet.samplers import MAX_GATE_QUBITS, MAX_REAL_GATE_QUBITS
from core.optimizers.network.scenario import (
    CUSTOM_SCENARIO_ID,
    build_custom_scenario,
    build_scenario,
)

logger = logging.getLogger(__name__)

VERTICAL = "facility-location"

#: Entitlement key required for any real-hardware submit.
HARDWARE_ENTITLEMENT = "solver.quantum_hardware"

#: Entitlement keys the API actually enforces. They are advertised on each
#: method so the UI shows the *same* key the server checks - advertising a
#: generic `solver.*` key while gating on a vertical key makes an upgrade
#: prompt point at the wrong feature.
VERTICAL_ENTITLEMENT = "network.design"
QUANTUM_ENTITLEMENT = "network.quantum"

DEFAULT_METHOD = "classical"

#: Said in the description of every method that samples the QUBO. Without a
#: capacity term the QUBO's own optimum overfills sites (paper, facility
#: location: on all four instances), so feasibility comes from the repair.
_NO_CAPACITY_TERM = (
    "The QUBO has no capacity term; capacity is enforced by a classical "
    "repair after sampling."
)

NETWORK_METHODS: Dict[str, Dict[str, Any]] = {
    "classical": {
        "label": "Greedy + Local Search",
        "paradigm": "classical",
        "family": "classical",
        "backend": None,
        "platforms": ["cpu"],
        "execution_modes": ["classical"],
        "baseline": True,
        "description": (
            "Greedy open-by-marginal-cost followed by drop and swap local "
            "search. The comparison baseline."
        ),
        "entitlement": VERTICAL_ENTITLEMENT,
    },
    "exact-milp": {
        "label": "Exact MILP (HiGHS)",
        "paradigm": "classical",
        "family": "classical",
        "backend": "milp",
        "platforms": ["cpu"],
        "execution_modes": ["classical"],
        "baseline": False,
        "description": (
            "Branch-and-cut on the exact mixed-integer model. Proves the "
            "optimum when it finishes inside the time limit; otherwise "
            "returns the best plan found and the remaining gap."
        ),
        # The endpoint asks for this key for every method other than the baseline.
        "entitlement": QUANTUM_ENTITLEMENT,
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
            "The facility-location QUBO annealed on CPU. Same encoding the "
            "annealer runs, without the device. " + _NO_CAPACITY_TERM
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
            "Site-opening and assignment binaries solved with D-Wave Ocean: "
            "simulated annealing in simulator mode, Leap hybrid on hardware. "
            + _NO_CAPACITY_TERM
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
            "The same QUBO mapped to an Ising Hamiltonian and sampled with "
            "QAOA. Parameters train on a simulator; in real mode the final "
            "circuit is measured on an IBM or IonQ device. Small scenarios "
            f"only (<= {MAX_GATE_QUBITS} binaries on a simulator, "
            f"<= {MAX_REAL_GATE_QUBITS} on hardware). " + _NO_CAPACITY_TERM
        ),
        "entitlement": QUANTUM_ENTITLEMENT,
    },
}


def list_methods() -> List[Dict[str, Any]]:
    """Method catalogue for the UI picker."""
    return [
        {
            "method": key,
            "label": spec["label"],
            "paradigm": spec["paradigm"],
            "family": spec["family"],
            "backend": spec["backend"],
            "platforms": list(spec["platforms"]),
            "execution_modes": list(spec["execution_modes"]),
            "baseline": spec["baseline"],
            "description": spec["description"],
            "entitlement": spec["entitlement"],
            "default": key == DEFAULT_METHOD,
        }
        for key, spec in NETWORK_METHODS.items()
    ]


def resolve_method(method: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Validate a method name and return ``(method, spec)``."""
    key = (method or DEFAULT_METHOD).strip().lower()
    if key not in NETWORK_METHODS:
        raise ValueError(
            f"unknown network method '{method}'; "
            f"expected one of {sorted(NETWORK_METHODS)}"
        )
    return key, NETWORK_METHODS[key]


def _step(
    name: str,
    description: str,
    stage: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
    explanation: Optional[str] = None,
    field_notes: Optional[Dict[str, str]] = None,
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
        "explanation": explanation,
        "field_notes": field_notes or {},
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
    return {
        "name": name,
        "kind": kind,
        "title": title,
        "stage": stage,
        "format": fmt,
        "spec": spec,
    }


def _network_map_artifact(
    problem: NetworkProblem, solution: NetworkSolution
) -> Dict[str, Any]:
    """Scatter of the chosen network.

    Uses a plain Chart.js scatter rather than a bespoke map format so it
    renders through the existing ``StepArtifact`` path with no new frontend
    renderer. Longitude is x, latitude is y.
    """
    open_ids = set(solution.open_site_ids)
    served_by = {
        dp_id: alloc.site_id
        for alloc in solution.allocations
        for dp_id in alloc.demand_point_ids
    }
    site_by_id = {s.id: s for s in problem.sites}

    datasets: List[Dict[str, Any]] = [
        {
            "label": "Open sites",
            "data": [
                {"x": s.lon, "y": s.lat, "label": s.name}
                for s in problem.sites
                if s.id in open_ids
            ],
            "backgroundColor": "rgba(16, 185, 129, 0.9)",
            "pointRadius": 10,
            "pointStyle": "rectRot",
        },
        {
            "label": "Rejected sites",
            "data": [
                {"x": s.lon, "y": s.lat, "label": s.name}
                for s in problem.sites
                if s.id not in open_ids
            ],
            "backgroundColor": "rgba(148, 163, 184, 0.55)",
            "pointRadius": 6,
            "pointStyle": "crossRot",
        },
        {
            "label": "Demand points",
            "data": [
                {
                    "x": d.lon,
                    "y": d.lat,
                    "label": f"{d.name} -> "
                    f"{site_by_id[served_by[d.id]].name if d.id in served_by else 'unassigned'}",
                }
                for d in problem.demand_points
            ],
            "backgroundColor": "rgba(102, 126, 234, 0.8)",
            "pointRadius": 5,
        },
    ]

    return _artifact(
        "network_map",
        "network-map",
        "Chosen network",
        {
            "type": "scatter",
            "data": {"datasets": datasets},
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Longitude"}},
                    "y": {"title": {"display": True, "text": "Latitude"}},
                }
            },
        },
    )


def _utilisation_artifact(solution: NetworkSolution) -> Dict[str, Any]:
    allocs = solution.allocations[:40]
    return _artifact(
        "site_utilisation",
        "utilisation",
        "Served demand vs capacity per open site",
        {
            "type": "bar",
            "data": {
                "labels": [a.site_name for a in allocs],
                "datasets": [
                    {
                        "label": "Served demand",
                        "data": [round(a.served_demand, 2) for a in allocs],
                        "backgroundColor": "rgba(102, 126, 234, 0.75)",
                    },
                    {
                        "label": "Capacity",
                        "data": [round(a.capacity, 2) for a in allocs],
                        "backgroundColor": "rgba(16, 185, 129, 0.45)",
                    },
                ],
            },
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
    )


def _convergence_artifact(diagnostics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Solver progress - energy for the QUBO path, cost for the classical one."""
    series = diagnostics.get("energy_trace") or diagnostics.get("cost_trace") or []
    if not series:
        return None
    label = "QUBO energy" if diagnostics.get("energy_trace") else "Total cost"
    return _artifact(
        "convergence",
        "convergence",
        "Solver convergence",
        {
            "type": "line",
            "data": {
                "labels": list(range(1, len(series) + 1)),
                "datasets": [
                    {
                        "label": label,
                        "data": [round(float(v), 4) for v in series],
                        "borderColor": "rgba(102, 126, 234, 1)",
                        "fill": False,
                        "tension": 0.15,
                    }
                ],
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Improvement step"}},
                    "y": {"title": {"display": True, "text": label}},
                }
            },
        },
        stage="process",
    )


def _comparison_artifact(comparison: Dict[str, Any]) -> Dict[str, Any]:
    return _artifact(
        "baseline_comparison",
        "comparison",
        "Cost vs classical baseline",
        {
            "type": "bar",
            "data": {
                "labels": ["Classical baseline", "This run"],
                "datasets": [
                    {
                        "label": "Total cost",
                        "data": [
                            comparison["baseline_total_cost"],
                            comparison["candidate_total_cost"],
                        ],
                        "backgroundColor": [
                            "rgba(148, 163, 184, 0.7)",
                            "rgba(102, 126, 234, 0.8)",
                        ],
                    }
                ],
            },
            "options": {"scales": {"y": {"beginAtZero": True}}},
        },
    )


def solve_network_problem(
    problem: NetworkProblem,
    *,
    method: Optional[str] = None,
    platform: str = "cpu",
    execution_mode: str = "simulator",
    config: Optional[Dict[str, Any]] = None,
    run_baseline: bool = True,
) -> Dict[str, Any]:
    """Run ``method`` on ``problem`` and emit the Job History envelope."""
    key, spec = resolve_method(method)
    cfg = dict(config or {})
    started = time.monotonic()
    steps: List[Dict[str, Any]] = []

    # Gate methods only dispatch on the vendor in `real` mode - or on a
    # vendor-hosted simulator the tenant can submit to with their own key.
    # Anything else executes locally whichever vendor was asked for, so
    # record where the work actually happened and never attribute a local
    # Qiskit result to IBM or IonQ.
    if spec.get("backend") == "gate" and not runs_on_vendor_cloud(
        platform, execution_mode
    ):
        platform = "cpu"

    qubo_preview = build_network_qubo(problem)
    summary = qubo_summary(qubo_preview)

    steps.append(
        _step(
            "problem_setup",
            "Load candidate sites, demand points and cost model",
            "setup",
            details={
                "candidate_sites": len(problem.sites),
                "demand_points": len(problem.demand_points),
                "total_demand": round(problem.total_demand, 2),
                "total_capacity": round(problem.total_capacity, 2),
                "max_open_sites": problem.max_open_sites,
                "cost_per_km_per_unit": problem.cost_per_km_per_unit,
            },
            explanation=(
                "Capacitated facility location asks which candidate sites to "
                "open and which site serves each demand point. Total cost is "
                "the fixed cost of every opened site plus the transport cost "
                "of serving demand from it - opening more sites cuts transport "
                "but adds fixed cost, and the optimum balances the two."
            ),
            field_notes={
                "candidate_sites": "Sites that MAY be opened. Not all will be.",
                "demand_points": "Customer clusters; each must be served by exactly one open site.",
                "total_demand": "Sum of all demand units that must be served.",
                "total_capacity": "Combined capacity if every candidate site opened.",
                "max_open_sites": "Hard cap on opened sites. None lets cost decide.",
                "cost_per_km_per_unit": "Transport cost per demand-unit per road kilometre.",
            },
        )
    )

    steps.append(
        _step(
            "qubo_encoding",
            "Encode the network decision as a QUBO",
            "encoding",
            details=summary,
            explanation=(
                "One binary per candidate site (open or not) plus one binary "
                "per demand-point/site pair (served from there or not). "
                "Constraints become penalty terms: serving a demand point "
                "exactly once, never serving from a closed site, and the cap "
                "on how many sites may open (carried with binary slack, since "
                "it is an upper bound rather than an equality). Capacity is "
                "deliberately NOT a penalty term - a one-sided limit encodes "
                "badly as a quadratic and the assignment repair enforces it "
                "exactly. Penalty weights are derived from this instance's own "
                "cost magnitudes so that breaking a constraint is always more "
                "expensive than any legal choice."
            ),
            field_notes={
                "n_vars": "Total binaries = sites + sites x demand points + slack.",
                "open_vars": "One per candidate site.",
                "assignment_vars": "One per (demand point, site) pair.",
                "nonzero_pairs": "Non-zero quadratic couplings; drives hardware embedding difficulty.",
                "penalties": "Derived constraint weights (assign = one-hot, link = no closed-site service, site_cap = cap on sites opened).",
            },
        )
    )

    if spec["backend"] is None:
        solution, diagnostics = solve_classical(
            problem,
            time_budget_s=float(cfg.get("time_budget_s", 10.0)),
            local_search=bool(cfg.get("local_search", True)),
        )
    elif spec["backend"] == "milp":
        solution, diagnostics = solve_exact(
            problem, time_limit_s=float(cfg.get("exact_time_limit_s", 60.0))
        )
    else:
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
        solution, diagnostics = solve_quantum(
            problem,
            backend=spec["backend"],
            platform=platform,
            execution_mode=execution_mode,
            sampler_options=sampler_options,
            classical_polish=bool(cfg.get("classical_polish", True)),
            seed=int(cfg.get("seed", 42)),
        )

    solve_artifacts = [a for a in [_convergence_artifact(diagnostics)] if a]
    solve_artifacts.extend(
        vendor_link_artifacts_from_runs(diagnostics.get("vendor_runs"))
    )
    steps.append(
        _step(
            "solve",
            f"Solve with {spec['label']}",
            "solve",
            details={k: v for k, v in diagnostics.items() if k != "vendor_runs"},
            artifacts=solve_artifacts,
            explanation=(
                "The solver searches for the site-opening and assignment "
                "combination with the lowest total cost. For QUBO methods the "
                "reported energy is the penalised objective, so it is only "
                "comparable across runs of the same instance - use total cost "
                "for anything else."
            ),
            field_notes={
                "algorithm": "Which solver actually ran.",
                "energy": "Final QUBO objective including constraint penalties.",
                "fallback_reason": "Set when the requested backend was unavailable and CPU annealing ran instead.",
                "classical_polish_applied": "True when the sampler broke the one-hot and assignment was repaired classically.",
                "raw_unassigned": "Demand points the sampler left unassigned before repair.",
                "raw_total_cost": "Total cost of the sampler's own solution, before the classical drop/swap polish. Only comparable to the baseline when raw_feasible is true.",
                "raw_feasible": "False when the sampler's own solution breaks capacity; a lower raw_total_cost is then an artefact, not a win.",
                "polish_improvement": "How much the classical polish took off the sampler's raw cost.",
                "elapsed_ms": "Solver wall time.",
            },
        )
    )

    baseline: Optional[NetworkSolution] = None
    comparison: Optional[Dict[str, Any]] = None
    if run_baseline and not spec["baseline"]:
        baseline, _ = solve_classical(problem)
        comparison = compare_solutions(solution, baseline)

    exact_ref: Optional[Dict[str, Any]] = None
    if spec["backend"] != "milp" and cfg.get("exact_reference", True):
        exact_ref = exact_reference(
            problem,
            candidate=solution,
            baseline=baseline,
            time_limit_s=float(cfg.get("exact_time_limit_s", DEFAULT_TIME_LIMIT_S)),
        )
        steps.append(
            _step(
                "exact_reference",
                "Score against the exact optimum",
                "post",
                details=exact_ref,
                warnings=exact_ref.get("caveats", []),
                explanation=(
                    "The comparison baseline is a heuristic, so beating or "
                    "matching it says little on its own. This step solves the "
                    "same instance as a mixed-integer program and reports how "
                    "far both this run and the baseline are from the true "
                    "optimum. When the solver cannot prove optimality inside its "
                    "time limit it says so and gives the remaining gap."
                ),
                field_notes={
                    "exact_total_cost": "Total cost of the proven optimum (or best plan found in time).",
                    "proven_optimal": "True when the solver proved no cheaper plan exists.",
                    "mip_gap": "Remaining relative gap between the best plan and the lower bound; 0 when proven.",
                    "candidate_gap_to_exact_pct": "How much more this run costs than the exact optimum.",
                    "baseline_gap_to_exact_pct": "How much more the heuristic baseline costs than the exact optimum.",
                },
            )
        )

    result_artifacts = [
        _network_map_artifact(problem, solution),
        _utilisation_artifact(solution),
    ]
    if comparison:
        result_artifacts.append(_comparison_artifact(comparison))

    steps.append(
        _step(
            "network_plan",
            "Final network design",
            "post",
            details={
                "open_sites": solution.open_site_ids,
                **solution.metrics,
            },
            artifacts=result_artifacts,
            warnings=(comparison or {}).get("caveats", []),
            explanation=(
                "The chosen network. Utilisation shows how hard each opened "
                "site is working - consistently low utilisation means the "
                "network is over-built. Overflow means demand was assigned "
                "beyond a site's capacity: the assignment repair fills sites "
                "in cost order and only overflows once every open site is "
                "full, so a non-zero value means the opened set genuinely "
                "cannot cover demand - raise the site cap rather than "
                "re-running."
            ),
            field_notes={
                "open_site_count": "How many sites the solver chose to open.",
                "coverage_pct": "Share of total demand actually served.",
                "capacity_overflow": "Demand units assigned beyond a site's capacity. Should be 0.",
                "avg_utilization_pct": "Mean served-demand-to-capacity ratio across open sites.",
                "fixed_cost": "Sum of opening costs for the chosen sites.",
                "transport_cost": "Cost of moving demand from sites to demand points.",
                "total_cost": "fixed_cost + transport_cost. The number to optimise.",
                "feasible": "True when nothing is unassigned and no site overflows.",
            },
        )
    )

    duration = time.monotonic() - started
    return {
        "vertical": VERTICAL,
        "method": key,
        "method_label": spec["label"],
        "paradigm": spec["paradigm"],
        "platform": platform,
        "execution_mode": execution_mode,
        "problem": problem.to_dict(),
        "solution": solution.to_dict(),
        "baseline": baseline.to_dict() if baseline else None,
        "comparison": comparison,
        "exact_reference": exact_ref,
        "metrics": solution.metrics,
        "qubo": summary,
        # Lifted out of the solve step so the UI can separate what the sampler
        # found from what the classical polish added. Without this the two
        # numbers are indistinguishable and every run looks like a tie.
        "solver_diagnostics": {
            "classical_polish_applied": diagnostics.get("classical_polish_applied", False),
            "raw_total_cost": diagnostics.get("raw_total_cost"),
            "raw_unassigned": diagnostics.get("raw_unassigned"),
            "raw_feasible": diagnostics.get("raw_feasible"),
            "raw_capacity_overflow": diagnostics.get("raw_capacity_overflow"),
            "polish_improvement": diagnostics.get("polish_improvement"),
        },
        "execution_steps": steps,
        "duration_seconds": round(duration, 4),
    }


def solve_scenario(
    scenario_id: Optional[str] = None,
    *,
    method: Optional[str] = None,
    platform: str = "cpu",
    execution_mode: str = "simulator",
    config: Optional[Dict[str, Any]] = None,
    run_baseline: bool = True,
    custom: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a seeded scenario and solve it."""
    if (scenario_id or "").strip() == CUSTOM_SCENARIO_ID:
        problem = build_custom_scenario(**(custom or {}))
    else:
        problem = build_scenario(scenario_id)
    result = solve_network_problem(
        problem,
        method=method,
        platform=platform,
        execution_mode=execution_mode,
        config=config,
        run_baseline=run_baseline,
    )
    result["scenario_id"] = scenario_id or "regional_3of6"
    return result
