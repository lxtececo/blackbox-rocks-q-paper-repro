"""Chart.js diagrams of a mission allocation result (F7).

Plain Chart.js specs, so they render through the existing step-artifact path."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from core.optimizers.mission.models import MissionProblem, Pair

_ACCENT = "rgba(102, 126, 234, 0.8)"
_GREEN = "rgba(16, 185, 129, 0.75)"
_GREY = "rgba(148, 163, 184, 0.7)"
_RED = "rgba(239, 68, 68, 0.9)"
_AMBER = "rgba(245, 158, 11, 0.95)"
_ASSET_COLOURS = [
    "rgba(102, 126, 234, 0.9)",
    "rgba(16, 185, 129, 0.9)",
    "rgba(236, 72, 153, 0.9)",
    "rgba(14, 165, 233, 0.9)",
    "rgba(168, 85, 247, 0.9)",
    "rgba(234, 179, 8, 0.9)",
    "rgba(20, 184, 166, 0.9)",
    "rgba(249, 115, 22, 0.9)",
    "rgba(99, 102, 241, 0.9)",
    "rgba(132, 204, 22, 0.9)",
    "rgba(217, 70, 239, 0.9)",
    "rgba(6, 182, 212, 0.9)",
    "rgba(244, 63, 94, 0.9)",
    "rgba(100, 116, 139, 0.9)",
]


def _artifact(name: str, kind: str, title: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "kind": kind, "title": title, "stage": "post", "format": "chartjs", "spec": spec}


def _xy(point: Tuple[float, float], label: str) -> Dict[str, Any]:
    return {"x": round(float(point[0]), 3), "y": round(float(point[1]), 3), "label": label}


def allocation_map(problem: MissionProblem, plan: Sequence[Pair]) -> Dict[str, Any]:
    """Bases, tasks and one spoke per assignment, in km on the scenario area."""
    tasks = {t.id: t for t in problem.tasks}
    assigned = {task_id for _, task_id in plan}
    datasets: List[Dict[str, Any]] = []
    for index, asset in enumerate(problem.assets):
        mine = sorted((tasks[t] for a, t in plan if a == asset.id), key=lambda t: (t.start_hour, t.id))
        if not mine:
            continue
        base = _xy(asset.base, f"{asset.name} base")
        points = [base]
        for task in mine:
            points.extend([_xy(task.position, task.name), base])
        colour = _ASSET_COLOURS[index % len(_ASSET_COLOURS)]
        datasets.append(
            {
                "label": asset.name,
                "data": points,
                "showLine": True,
                "borderColor": colour,
                "backgroundColor": colour,
                "borderWidth": 1.5,
                "pointRadius": 4,
                "order": 2,
            }
        )
    at_base: Dict[Tuple[float, float], List[str]] = {}
    for asset in problem.assets:
        at_base.setdefault((float(asset.base[0]), float(asset.base[1])), []).append(asset.name)
    datasets.append(
        {
            "label": "Bases",
            "data": [_xy(point, ", ".join(names)) for point, names in at_base.items()],
            "backgroundColor": _AMBER,
            "pointRadius": 9,
            "pointStyle": "rectRot",
            "order": 0,
        }
    )
    datasets.append(
        {
            "label": "Unassigned tasks",
            "data": [_xy(t.position, t.name) for t in problem.tasks if t.id not in assigned],
            "backgroundColor": _RED,
            "borderColor": _RED,
            "borderWidth": 2,
            "pointRadius": 7,
            "pointStyle": "crossRot",
            "order": 1,
        }
    )
    return _artifact(
        "mission_map",
        "map",
        "Bases, tasks and assignments",
        {
            "type": "scatter",
            "data": {"datasets": datasets},
            "options": {
                "scales": {
                    "x": {
                        "title": {"display": True, "text": "East (km)"},
                        "suggestedMin": 0,
                        "suggestedMax": float(problem.area_km[0]),
                    },
                    "y": {
                        "title": {"display": True, "text": "North (km)"},
                        "suggestedMin": 0,
                        "suggestedMax": float(problem.area_km[1]),
                    },
                }
            },
        },
    )


def duty_chart(problem: MissionProblem, asset_usage: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Task hours of each asset next to its duty limit."""
    names = {a.id: a.name for a in problem.assets}
    return _artifact(
        "mission_duty",
        "duty",
        "Task hours and duty limit",
        {
            "type": "bar",
            "data": {
                "labels": [names[u["asset_id"]] for u in asset_usage],
                "datasets": [
                    {
                        "label": "Task hours used",
                        "data": [u["hours_used"] for u in asset_usage],
                        "backgroundColor": _ACCENT,
                    },
                    {
                        "label": "Duty limit (hours)",
                        "data": [u["duty_limit_hours"] for u in asset_usage],
                        "backgroundColor": _GREY,
                    },
                ],
            },
            "options": {"scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Hours"}}}},
        },
    )


def priority_chart(problem: MissionProblem, plan: Sequence[Pair]) -> Dict[str, Any]:
    """Assigned and unassigned tasks per priority, highest priority first."""
    assigned = {task_id for _, task_id in plan}
    priorities = sorted({int(t.priority) for t in problem.tasks}, reverse=True)
    taken = [sum(1 for t in problem.tasks if int(t.priority) == p and t.id in assigned) for p in priorities]
    left = [sum(1 for t in problem.tasks if int(t.priority) == p and t.id not in assigned) for p in priorities]
    stacked = {"stacked": True}
    return _artifact(
        "mission_priority",
        "priority",
        "Tasks by priority",
        {
            "type": "bar",
            "data": {
                "labels": [f"Priority {p}" for p in priorities],
                "datasets": [
                    {"label": "Assigned", "data": taken, "backgroundColor": _GREEN},
                    {"label": "Unassigned", "data": left, "backgroundColor": _RED},
                ],
            },
            "options": {
                "scales": {
                    "x": dict(stacked),
                    "y": {
                        **stacked,
                        "beginAtZero": True,
                        "ticks": {"precision": 0},
                        "title": {"display": True, "text": "Tasks"},
                    },
                }
            },
        },
    )


def shortfall_chart(bars: Sequence[Tuple[str, float]]) -> Dict[str, Any]:
    """Shortfall of each plan against the proven optimum, in percent."""
    return _artifact(
        "mission_shortfall",
        "comparison",
        "Shortfall to the proven optimum",
        {
            "type": "bar",
            "data": {
                "labels": [label for label, _ in bars],
                "datasets": [
                    {
                        "label": "Percent below the exact optimum",
                        "data": [round(float(value), 4) for _, value in bars],
                        "backgroundColor": [_ACCENT] + [_GREY] * (len(bars) - 1),
                    }
                ],
            },
            "options": {
                "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Percent"}}},
                "plugins": {"legend": {"display": False}},
            },
        },
    )


def distribution_chart(distribution: Dict[str, Any]) -> Dict[str, Any]:
    """Filled objectives of the sampler's reads and of the random bitstrings, over shared bins."""
    edges = distribution["bin_edges"]
    # One number per bin: the full range as a label leaves no room for the bars.
    labels = [f"{edges[i]:,.0f}" for i in range(len(edges) - 1)]
    return _artifact(
        "mission_sampler_null",
        "distribution",
        "Sampler against random bitstrings",
        {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {"label": "Sampler reads", "data": list(distribution["sampler"]), "backgroundColor": _ACCENT},
                    {"label": "Random bitstrings", "data": list(distribution["null"]), "backgroundColor": _GREY},
                ],
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Objective after repair and fill (lower edge of the bin)"}},
                    "y": {
                        "beginAtZero": True,
                        "ticks": {"precision": 0},
                        "title": {"display": True, "text": "Count"},
                    },
                }
            },
        },
    )
