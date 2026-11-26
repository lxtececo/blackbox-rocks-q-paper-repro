"""Mission allocation solve orchestration: catalogue, execution, result (F7).

Every method runs next to the exact referee and the greedy baseline."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.optimizers.mission import charts, sampling
from core.optimizers.mission.exact import DEFAULT_TIME_LIMIT_S, exact_reference_with_plan
from core.optimizers.mission.feasibility import check_plan, objective
from core.optimizers.mission.greedy import solve_greedy
from core.optimizers.mission.models import (
    TOL,
    Asset,
    MissionProblem,
    Pair,
    Structure,
    Task,
    _round,
    build_structure,
    distance,
    exclusion_reason,
    pair_cost,
    pair_value,
    pair_weight,
    travel_hours,
)
from core.optimizers.mission.qubo import DEFAULT_PENALTY_FACTOR, NOT_ENCODED, build_qubo, qubo_summary
from core.optimizers.mission.scenario import CUSTOM_SCENARIO_ID, build_custom_scenario, get_scenario

logger = logging.getLogger(__name__)

VERTICAL = "mission-allocation"

#: Entitlement keys the API enforces, advertised on each method.
VERTICAL_ENTITLEMENT = "mission.allocation"
QUANTUM_ENTITLEMENT = "mission.quantum"

DEFAULT_METHOD = "exact"

SAMPLING_DEFAULTS: Dict[str, Any] = {
    "reads": 64,
    "sweeps": 1000,
    "seed": 0,
    "penalty_factor": DEFAULT_PENALTY_FACTOR,
}
#: Inclusive limits of the sampling configuration.
SAMPLING_LIMITS: Dict[str, List[float]] = {
    "reads": [8, 512],
    "sweeps": [100, 20000],
    "seed": [0, 2**32 - 1],
    "penalty_factor": [1.01, 5.0],
}
EXACT_TIME_LIMIT_BOUNDS = [0.1, 600.0]

_SCOPE = "Support and logistics allocation only."

MISSION_METHODS: Dict[str, Dict[str, Any]] = {
    "exact": {
        "label": "Exact (MILP)",
        "paradigm": "classical",
        "family": "classical",
        "backend": "milp",
        "platforms": ["cpu"],
        "execution_modes": ["classical"],
        "baseline": False,
        "description": (
            "The mixed-integer model solved with HiGHS. Proves the optimum "
            "when it finishes inside the time limit; otherwise returns the "
            "best plan found and says so."
        ),
        "entitlement": VERTICAL_ENTITLEMENT,
    },
    "greedy": {
        "label": "Greedy by priority",
        "paradigm": "classical",
        "family": "classical",
        "backend": None,
        "platforms": ["cpu"],
        "execution_modes": ["classical"],
        "baseline": True,
        "description": (
            "Tasks by decreasing priority, each to the cheapest asset that "
            "can still take it. The comparison baseline."
        ),
        "entitlement": VERTICAL_ENTITLEMENT,
    },
    "qubo-annealing": {
        "label": "QUBO + simulated annealing",
        "paradigm": "dwave-quantum",
        "family": "annealing",
        "backend": "dwave",
        "platforms": ["dwave"],
        "execution_modes": ["simulator"],
        "baseline": False,
        "description": (
            "One binary per allowed pair, sampled with simulated annealing. "
            "Every read is repaired and filled classically; a null of random "
            "bitstrings goes through the same stages. The QUBO does not "
            "encode the duty limit; the repair enforces it. Measured on 60 "
            "generated scenarios: on average 0.10 percent below the optimum, "
            "the null 0.88 percent, the greedy baseline 4.13 percent. "
            "Measured on 60 others: random bitstrings given the same time "
            "end as close to the optimum (closer on 24 scenarios, further "
            "on 16, level on 20), and the exact method is faster than both."
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
        for key, spec in MISSION_METHODS.items()
    ]


def resolve_method(method: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Validate a method name and return ``(method, spec)``."""
    key = (method or DEFAULT_METHOD).strip().lower()
    if key not in MISSION_METHODS:
        raise ValueError(
            f"unknown mission method '{method}'; "
            f"expected one of {sorted(MISSION_METHODS)}"
        )
    return key, MISSION_METHODS[key]


def _step(
    name: str,
    description: str,
    stage: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    explanation: Optional[str] = None,
    field_notes: Optional[Dict[str, str]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
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


def _number(cfg: Dict[str, Any], key: str, default: Any, bounds: List[float], integer: bool) -> Any:
    """Read one numeric setting; ValueError when it is not inside its limits."""
    value = cfg.get(key)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number")
    if integer:
        if float(value) != int(value):
            raise ValueError(f"{key} must be an integer")
        value = int(value)
    else:
        value = float(value)
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{key} must be between {bounds[0]} and {bounds[1]}, got {value}")
    return value


def _validated(method: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config or {})
    out: Dict[str, Any] = {
        "exact_time_limit_s": _number(cfg, "exact_time_limit_s", DEFAULT_TIME_LIMIT_S, EXACT_TIME_LIMIT_BOUNDS, False)
    }
    if method == "qubo-annealing":
        for key in ("reads", "sweeps", "seed"):
            out[key] = _number(cfg, key, SAMPLING_DEFAULTS[key], SAMPLING_LIMITS[key], True)
        out["penalty_factor"] = _number(
            cfg, "penalty_factor", SAMPLING_DEFAULTS["penalty_factor"], SAMPLING_LIMITS["penalty_factor"], False
        )
    return out


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole > 0 else 0.0


def _times_overlap(t: Task, u: Task) -> bool:
    return t.start_hour < u.end_hour - TOL and u.start_hour < t.end_hour - TOL


def _exclusion_figures(asset: Asset, task: Task, reason: str) -> Dict[str, float]:
    """The figures the failing rule compares."""
    out = travel_hours(asset, asset.base, task.position)
    if reason == "arrives_late":
        return {"late_by_hours": _round(asset.available_from + out - task.start_hour)}
    if reason == "returns_late":
        return {"late_by_hours": _round(task.end_hour + out - asset.available_until)}
    if reason == "task_exceeds_duty_limit":
        return {"hours_needed": _round(task.duration_hours), "duty_limit_hours": _round(asset.duty_limit_hours)}
    return {"cost": _round(pair_cost(asset, task)), "value": _round(pair_value(task))}


def _unassigned(problem: MissionProblem, plan: List[Pair], st: Structure) -> List[Dict[str, Any]]:
    """Per unassigned task, what keeps each asset with the capability from taking it in this plan."""
    t_pos = {t.id: i for i, t in enumerate(problem.tasks)}
    tasks = {t.id: t for t in problem.tasks}
    held: Dict[str, List[str]] = {a.id: [] for a in problem.assets}
    for asset_id, task_id in plan:
        held[asset_id].append(task_id)
    assigned = {task_id for _, task_id in plan}
    rows = []
    for task in problem.tasks:
        if task.id in assigned:
            continue
        capable = []
        for asset in problem.assets:
            excluded = exclusion_reason(asset, task)
            if excluded == "capability":
                continue
            row: Dict[str, Any] = {"asset_id": asset.id, "allowed": excluded is None, "obstacles": []}
            if excluded is not None:
                row["obstacles"].append(excluded)
                row.update(_exclusion_figures(asset, task, excluded))
            else:
                near = st.conflicts[asset.id][task.id]
                clashes = sorted((t for t in held[asset.id] if t in near), key=t_pos.__getitem__)
                used = math.fsum(tasks[t].duration_hours for t in held[asset.id])
                if clashes:
                    row["obstacles"].append("conflict")
                    # A conflict without an overlap in time is one of travel between the two places.
                    row["conflicts_with"] = [
                        {"task_id": t, "why": "overlap" if _times_overlap(task, tasks[t]) else "travel"}
                        for t in clashes
                    ]
                if used + task.duration_hours > asset.duty_limit_hours + TOL:
                    row["obstacles"].append("duty_limit")
                    row["hours_used"] = _round(used)
                    row["hours_needed"] = _round(task.duration_hours)
                    row["duty_limit_hours"] = _round(asset.duty_limit_hours)
            capable.append(row)
        allowed = [row for row in capable if row["allowed"]]
        if not capable:
            reason = "no_capable_asset"
        elif not allowed:
            reason = "no_allowed_asset"
        elif all(row["obstacles"] for row in allowed):
            reason = "blocked_in_plan"
        else:
            reason = "could_be_added"
        rows.append({"task_id": task.id, "reason": reason, "assets": capable})
    return rows


def _solution(problem: MissionProblem, plan: List[Pair], st: Structure) -> Dict[str, Any]:
    """Assignments in asset then start order, unassigned tasks, hours per asset."""
    assets = {a.id: a for a in problem.assets}
    tasks = {t.id: t for t in problem.tasks}
    a_pos = {a.id: i for i, a in enumerate(problem.assets)}
    t_pos = {t.id: i for i, t in enumerate(problem.tasks)}
    ordered = sorted(plan, key=lambda p: (a_pos[p[0]], tasks[p[1]].start_hour, t_pos[p[1]]))
    assignments = [
        {
            "asset_id": asset_id,
            "task_id": task_id,
            "start_hour": _round(tasks[task_id].start_hour),
            "end_hour": _round(tasks[task_id].end_hour),
            "round_trip_km": _round(2.0 * distance(assets[asset_id].base, tasks[task_id].position)),
            "cost": _round(pair_cost(assets[asset_id], tasks[task_id])),
            "value": _round(pair_value(tasks[task_id])),
            "weight": _round(pair_weight(assets[asset_id], tasks[task_id])),
        }
        for asset_id, task_id in ordered
    ]
    assigned = {task_id for _, task_id in plan}
    usage = []
    for asset in problem.assets:
        mine = [task_id for asset_id, task_id in ordered if asset_id == asset.id]
        hours = math.fsum(tasks[t].duration_hours for t in mine)
        usage.append(
            {
                "asset_id": asset.id,
                "task_ids": mine,
                "hours_used": _round(hours),
                "duty_limit_hours": _round(asset.duty_limit_hours),
                "utilisation_pct": _pct(hours, asset.duty_limit_hours),
            }
        )
    return {
        "assignments": assignments,
        "unassigned_task_ids": [t.id for t in problem.tasks if t.id not in assigned],
        "unassigned": _unassigned(problem, plan, st),
        "asset_usage": usage,
    }


def _metrics(problem: MissionProblem, plan: List[Pair], feasible: bool) -> Dict[str, Any]:
    assets = {a.id: a for a in problem.assets}
    tasks = {t.id: t for t in problem.tasks}
    assigned = {task_id for _, task_id in plan}
    priority_total = int(sum(t.priority for t in problem.tasks))
    priority_assigned = int(sum(t.priority for t in problem.tasks if t.id in assigned))
    return {
        "task_count": len(problem.tasks),
        "asset_count": len(problem.assets),
        "assigned_count": len(assigned),
        "coverage_pct": _pct(len(assigned), len(problem.tasks)),
        "priority_total": priority_total,
        "priority_assigned": priority_assigned,
        "weighted_coverage_pct": _pct(priority_assigned, priority_total),
        "total_value": _round(math.fsum(pair_value(tasks[t]) for _, t in plan)),
        "total_cost": _round(math.fsum(pair_cost(assets[a], tasks[t]) for a, t in plan)),
        "objective": _round(objective(problem, plan)),
        "feasible": bool(feasible),
    }


def _gap(reference: float, returned: float) -> Optional[float]:
    """Shortfall of the returned plan against a reference, in percent."""
    if not reference > 0:
        return None
    return round(100.0 * (reference - returned) / reference, 4)


def solve_mission_problem(
    problem: MissionProblem,
    method: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run ``method`` on ``problem`` next to the exact referee and the greedy baseline."""
    key, spec = resolve_method(method)
    cfg = _validated(key, config)
    started = time.monotonic()
    steps: List[Dict[str, Any]] = []

    st = build_structure(problem)
    orphans = [t.id for t in problem.tasks if not st.assets_of[t.id]]
    problem_block = problem.to_dict()
    problem_block["allowed_pairs"] = len(st.pairs)
    problem_block["conflict_pairs"] = st.conflict_count
    steps.append(
        _step(
            "problem_setup",
            "Load assets and tasks and derive the allowed pairs",
            "setup",
            details={
                "assets": len(problem.assets),
                "tasks": len(problem.tasks),
                "allowed_pairs": len(st.pairs),
                "conflict_pairs": st.conflict_count,
                "tasks_without_allowed_asset": orphans,
                "horizon_hours": _round(problem.horizon_hours),
            },
            explanation=(
                "An asset may take a task when it has the capability, can reach "
                "the task from its base and be back inside its availability "
                "window, and the task is worth more than it costs. Two tasks "
                "conflict on an asset when it cannot travel from one to the "
                "other in time. " + _SCOPE
            ),
            field_notes={
                "allowed_pairs": "Asset and task combinations that may be chosen; one decision each.",
                "conflict_pairs": "Task pairs that one asset cannot both serve, counted per asset.",
                "tasks_without_allowed_asset": "Tasks no asset can take; they stay unassigned in every plan.",
            },
        )
    )

    exact_plan, exact_block = exact_reference_with_plan(problem, time_limit_s=cfg["exact_time_limit_s"])
    exact_objective = objective(problem, exact_plan)
    steps.append(
        _step(
            "exact_reference",
            "Solve the exact model as the referee",
            "solve" if key == "exact" else "post",
            details=dict(exact_block),
            warnings=(
                []
                if exact_block["proven_optimal"]
                else ["The exact solver stopped at its time limit; no gap to the optimum is shown."]
            ),
            explanation=(
                "The same instance as a mixed-integer program: at most one asset "
                "per task, no conflicting tasks on one asset, task hours inside "
                "the duty limit. A second solve without the duty limit shows "
                "whether that limit binds."
            ),
            field_notes={
                "objective": "Sum over the assignments of priority x 1000 minus cost.",
                "status": "optimal, or time_limit when the solver stopped before a proof.",
                "duty_limit_binding": "True when the optimum would be larger without the duty limit.",
                "objective_without_duty_limit": "Optimum of the model without the duty limit; the QUBO minimum.",
            },
        )
    )

    greedy_plan = solve_greedy(problem, st)
    greedy_objective = objective(problem, greedy_plan)
    steps.append(
        _step(
            "greedy_baseline",
            "Greedy baseline by priority",
            "solve" if key == "greedy" else "post",
            details={"objective": _round(greedy_objective), "assigned_count": len(greedy_plan)},
            explanation=(
                "Tasks in order of decreasing priority, each to the allowed "
                "asset with the largest weight that can still take it."
            ),
            field_notes={"objective": "Objective of the greedy plan, for comparison."},
        )
    )

    qubo_block: Optional[Dict[str, Any]] = None
    sampler_block: Optional[Dict[str, Any]] = None
    if key == "exact":
        plan = exact_plan
    elif key == "greedy":
        plan = greedy_plan
    else:
        model = build_qubo(problem, cfg["penalty_factor"], st)
        qubo_block = qubo_summary(model)
        steps.append(
            _step(
                "qubo_encoding",
                "Encode the allocation as a QUBO",
                "encoding",
                details=dict(qubo_block),
                explanation=(
                    "One binary per allowed pair. The energy is minus the sum of "
                    "the chosen weights plus a penalty for two assets on one task "
                    "and for two conflicting tasks on one asset. The penalty is "
                    "larger than the largest weight, so the minimum has no "
                    "violated pair. Not encoded: " + NOT_ENCODED + "."
                ),
                field_notes={
                    "variables": "Binaries, one per allowed pair.",
                    "interactions": "Penalised pairs of binaries.",
                    "penalty": "penalty_factor x max_weight.",
                    "not_encoded": "Constraint left to the repair after sampling.",
                },
            )
        )
        # A sampler failure propagates; no classical plan stands in for it.
        plan, sampler_block = sampling.solve_sampling(
            problem,
            reads=cfg["reads"],
            sweeps=cfg["sweeps"],
            seed=cfg["seed"],
            penalty_factor=cfg["penalty_factor"],
            qubo=model,
        )
        steps.append(
            _step(
                "sampling",
                "Sample the QUBO with simulated annealing",
                "solve",
                details={
                    "sampler": "simulated annealing (simulator)",
                    "reads": sampler_block["reads"],
                    "sweeps": sampler_block["sweeps"],
                    "seed": sampler_block["seed"],
                    "raw": dict(sampler_block["raw"]),
                },
                explanation=(
                    "Raw reads as the annealer returned them. A raw read counts "
                    "as feasible only when it satisfies every constraint, the "
                    "duty limit included."
                ),
                field_notes={
                    "feasible_share": "Share of the raw reads that satisfy every constraint.",
                    "best_feasible_objective": "Best objective among the feasible raw reads; none when no read is feasible.",
                    "lowest_energy": "Lowest QUBO energy over the reads.",
                },
            )
        )
        steps.append(
            _step(
                "repair_and_fill",
                "Repair every read, then fill unassigned tasks",
                "post",
                details={
                    "repaired": dict(sampler_block["repaired"]),
                    "filled": dict(sampler_block["filled"]),
                    "returned": sampler_block["returned"],
                },
                explanation=(
                    "Repair drops assignments until no constraint is violated. "
                    "Fill then adds unassigned tasks greedily where they fit. "
                    "Both are classical; the three stage objectives are kept apart."
                ),
                field_notes={
                    "repaired": "Best and mean objective over the reads after the repair.",
                    "filled": "Best and mean objective over the reads after the fill.",
                },
            )
        )
        steps.append(
            _step(
                "null_reference",
                "Random bitstrings through the same repair and fill",
                "post",
                details=dict(sampler_block["null"]),
                explanation=(
                    "The same number of uniform random bitstrings through the "
                    "same stages. The difference to the sampler's numbers is "
                    "what the sampler added over chance and the classical fill."
                ),
                field_notes={
                    "draws": "Number of random bitstrings; equal to the reads.",
                    "filled": "Best and mean objective of the random bitstrings after repair and fill.",
                },
            )
        )

    plan = [(str(a), str(t)) for a, t in plan]
    verdict = check_plan(problem, plan)
    if not verdict["feasible"]:
        logger.error("[mission] %s returned an infeasible plan: %s", key, verdict["violations"])
        steps[-1]["warnings"].append(f"The returned plan violates the model: {verdict['violations']}")
    returned_objective = objective(problem, plan)

    reference = {
        "exact": exact_block,
        "greedy": {"objective": _round(greedy_objective)},
        "gap_to_exact_pct": _gap(exact_objective, returned_objective) if exact_block["proven_optimal"] else None,
        "gap_to_greedy_pct": _gap(greedy_objective, returned_objective),
    }
    solution = _solution(problem, plan, st)
    metrics = _metrics(problem, plan, verdict["feasible"])

    diagrams = [
        charts.allocation_map(problem, plan),
        charts.duty_chart(problem, solution["asset_usage"]),
        charts.priority_chart(problem, plan),
    ]
    # A shortfall needs a proven optimum to be measured against.
    if exact_block["proven_optimal"] and exact_objective > 0:
        bars = [("This run", _gap(exact_objective, returned_objective))]
        if key != "greedy":
            bars.append(("Greedy baseline", _gap(exact_objective, greedy_objective)))
        if sampler_block:
            bars.append(("Random bitstrings, best", _gap(exact_objective, sampler_block["null"]["filled"]["best"])))
        diagrams.append(charts.shortfall_chart(bars))
    if sampler_block:
        diagrams.append(charts.distribution_chart(sampler_block["filled_distribution"]))
    steps.append(
        _step(
            "allocation_plan",
            "Final allocation",
            "post",
            details=dict(metrics),
            artifacts=diagrams,
            explanation=(
                "The returned plan. The map draws one line from an asset's base "
                "to each task it takes; the model charges every task a round "
                "trip from the base. Positions are synthetic. " + _SCOPE
            ),
            field_notes={
                "coverage_pct": "Share of the tasks that are assigned.",
                "weighted_coverage_pct": "Share of the summed priorities that is assigned.",
                "objective": "Sum over the assignments of priority x 1000 minus cost.",
                "feasible": "True when the plan satisfies every constraint of the model.",
            },
        )
    )

    return {
        "vertical": VERTICAL,
        "method": key,
        "method_label": spec["label"],
        "paradigm": spec["paradigm"],
        "platform": spec["platforms"][0],
        "execution_mode": spec["execution_modes"][0],
        "problem": problem_block,
        "solution": solution,
        "metrics": metrics,
        "reference": reference,
        "qubo": qubo_block,
        "sampler": sampler_block,
        "execution_steps": steps,
        "duration_seconds": round(time.monotonic() - started, 4),
        "problem_size": len(st.pairs),
    }


def solve_scenario(
    scenario_id: Optional[str] = None,
    method: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    custom: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a preset or a custom scenario and solve it."""
    if (scenario_id or "").strip().lower() == CUSTOM_SCENARIO_ID:
        settings = dict(custom or {})
        unknown = sorted(set(settings) - {"asset_count", "task_count", "seed"})
        if unknown:
            raise ValueError(f"unknown custom scenario setting(s): {unknown}")
        problem = build_custom_scenario(**settings)
    else:
        problem = get_scenario(scenario_id)
    return solve_mission_problem(problem, method=method, config=config)
