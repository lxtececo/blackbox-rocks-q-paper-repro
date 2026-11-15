"""
Diagnostic artifact builders — 7-QAP.23 Slice B.

Produces Chart.js spec dictionaries for the most common quantum-optimisation
diagnostic plots (cost-vs-iteration convergence, top-k bitstring histogram,
weights-vs-asset bar). Each builder returns a dict shaped like::

    { "type": "line"|"bar", "data": {...}, "options": {...} }

Suitable for direct attachment via ``ExecutionStep.add_artifact(spec=...)``
and rendering by the frontend ``StepArtifact`` component.

Builders are intentionally pure / dependency-free (only stdlib + numpy
optional) so they can be called from any adapter or executor without
import cycles.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────
# Convergence: cost (or energy) vs iteration
# ──────────────────────────────────────────────────────────────────────
def convergence_chart(
    values: Sequence[float],
    *,
    label: str = "objective",
    title: str = "Convergence",
    y_axis: str = "objective",
    x_axis: str = "iteration",
    color: str = "#3b82f6",
) -> Dict[str, Any]:
    """Build a Chart.js line spec from per-iteration objective values."""
    points = [float(v) for v in values]
    return {
        "type": "line",
        "data": {
            "labels": [str(i + 1) for i in range(len(points))],
            "datasets": [
                {
                    "label": label,
                    "data": points,
                    "borderColor": color,
                    "backgroundColor": color + "33",
                    "fill": False,
                    "tension": 0.2,
                    "pointRadius": 2,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": x_axis}},
                "y": {"title": {"display": True, "text": y_axis}},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Bitstring frequency histogram (top-k)
# ──────────────────────────────────────────────────────────────────────
def bitstring_histogram_chart(
    counts: Mapping[str, int],
    *,
    top_k: int = 12,
    title: str = "Bitstring distribution (top {k})",
    color: str = "#8b5cf6",
) -> Dict[str, Any]:
    """Build a Chart.js bar spec from a measurement counts dict."""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [bs for bs, _ in items]
    data = [int(c) for _, c in items]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "shots",
                    "data": data,
                    "backgroundColor": color,
                    "borderColor": color,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title.format(k=len(items))},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "bitstring"}},
                "y": {"title": {"display": True, "text": "count"}, "beginAtZero": True},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Allocation weights bar chart (final portfolio)
# ──────────────────────────────────────────────────────────────────────
def weights_bar_chart(
    weights: Mapping[str, float],
    *,
    title: str = "Portfolio allocation",
    color: str = "#10b981",
) -> Dict[str, Any]:
    """Build a Chart.js bar spec from an asset → weight mapping."""
    items = [(s, float(w)) for s, w in weights.items() if abs(float(w)) > 1e-9]
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [s for s, _ in items]
    data = [round(w, 6) for _, w in items]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "weight",
                    "data": data,
                    "backgroundColor": color,
                    "borderColor": color,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "asset"}},
                "y": {"title": {"display": True, "text": "weight"}, "beginAtZero": True},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Helper: attach a "Quantum execution diagnostics" step to the latest job.
# Used by adapters that bypass the legacy executor JobHistoryManager hooks.
# ──────────────────────────────────────────────────────────────────────
def attach_diagnostics_step(
    *,
    name: str,
    description: str,
    artifacts: Iterable[Dict[str, Any]],
    method: Optional[str] = None,
):
    """
    Open a new ExecutionStep on the latest job tracked by the singleton
    JobHistoryManager and call ``add_artifact`` for each entry in
    ``artifacts``. Each artifact dict must contain at least ``name``,
    ``kind``, ``spec`` and may include ``title``, ``stage``, ``meta``.

    If the latest job already has a step with the same ``name`` (e.g. an
    executor and its wrapping platform adapter both emit a
    ``quantum-execution-diagnostics`` block), the existing step is
    re-used and only artifacts whose ``name`` is not already present
    are appended. This prevents duplicate vendor-portal deep-link chips
    in the JobHistoryView (F24 §B follow-up).

    If no job exists yet (e.g. adapter-only path), a minimal job is
    created so the step has a parent.

    Errors are swallowed and logged — diagnostics must never break a run.
    """
    try:
        from core.job_tracker import get_job_history_manager  # local import
    except Exception:  # noqa: BLE001
        return

    try:
        jhm = get_job_history_manager()
        jobs = jhm.list_jobs(limit=1)
        if jobs:
            job = jobs[0]
        else:
            job = jhm.create_job(method=method or "quantum", stocks=[])

        # Idempotency: if the latest job already has a step with this
        # name, re-use it. ExecutionStep.artifacts is a list of dicts
        # (already-attached) — collect their names so we don't add the
        # same vendor_link / chart twice.
        step = None
        for existing in reversed(getattr(job, "steps", []) or []):
            if getattr(existing, "name", None) == name:
                step = existing
                break

        if step is None:
            step = job.add_step(name=name, description=description)

        existing_names = {
            (a or {}).get("name")
            for a in (getattr(step, "artifacts", None) or [])
            if isinstance(a, dict)
        }

        added = 0
        for art in artifacts:
            art_name = art.get("name")
            if art_name in existing_names:
                # Already attached by an earlier emitter (executor → adapter
                # chain); skip silently to avoid duplicate chips/charts.
                continue
            try:
                step.add_artifact(
                    name=art_name,
                    kind=art["kind"],
                    spec=art.get("spec"),
                    title=art.get("title"),
                    stage=art.get("stage", "post"),
                    chart_format=art.get("format", "chartjs"),
                    artifact_uri=art.get("artifact_uri"),
                    meta=art.get("meta"),
                )
                if art_name:
                    existing_names.add(art_name)
                added += 1
            except Exception as art_err:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "[diagnostics] add_artifact failed for %r: %s",
                    art_name, art_err,
                )
        # Only (re-)complete the step if we actually appended something;
        # avoid clobbering the original status/timing for a no-op call.
        if added:
            step.complete(status="completed", artifact_count=len(step.artifacts))
        return step
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "[diagnostics] attach_diagnostics_step failed: %s", exc,
        )
        return None


# ──────────────────────────────────────────────────────────────────────
# 7-QAP.25 Slice 4c — explicit-job optimization-trace helper.
#
# Six executors (CLASSICAL / QAOA / PCE / CVAR-VQE / BF-DCQO / D-Wave
# template / cirq-VQE template) used to inline the same 25-line block
# to create an "optimization-trace-diagnostics" step and attach a list
# of charts. This helper consolidates that boilerplate so a single fix
# (or a new chart) lands in all sites at once. The job is supplied
# explicitly (unlike ``attach_diagnostics_step`` which implicitly looks
# up the latest job and races across concurrent submissions).
#
# Each ``artifacts`` entry is ``(name, title, spec, stage)``. ``spec``
# values of ``None`` are silently dropped so callers can just append
# the output of every builder without guarding each one. The whole
# thing is best-effort: a charting bug must never fail the job.
# ──────────────────────────────────────────────────────────────────────
def attach_optimization_trace_step(
    job_exec: Any,
    *,
    log_tag: str,
    artifacts: Iterable[tuple],
    description: str = "Method-agnostic optimisation diagnostics (7-QAP.25)",
    step_name: str = "optimization-trace-diagnostics",
) -> int:
    """Attach the optimization-trace-diagnostics step to ``job_exec``.

    Returns the number of artifacts actually attached (0 means nothing
    rendered — e.g. all builders returned ``None`` or an exception was
    swallowed). Never raises.
    """
    try:
        kept: List[tuple] = []
        for entry in artifacts:
            if not entry:
                continue
            # Accept 3- or 4-tuples; default stage is "post" because the
            # existing call sites all attach final-run diagnostics.
            if len(entry) == 4:
                name, title, spec, stage = entry
            elif len(entry) == 3:
                name, title, spec = entry  # type: ignore[misc]
                stage = "post"
            else:
                continue
            if spec is None:
                continue
            kept.append((name, title, spec, stage))
        if not kept:
            return 0
        diag_step = job_exec.add_step(step_name, description)
        for name, title, spec, stage in kept:
            diag_step.add_artifact(
                name=name,
                kind="chart",
                spec=spec,
                title=title,
                stage=stage,
                chart_format="chartjs",
            )
        diag_step.complete(status="completed", artifact_count=len(kept))
        return len(kept)
    except Exception as diag_err:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "[%s] optimization-trace-diagnostics skipped: %s", log_tag, diag_err,
        )
        return 0


# ──────────────────────────────────────────────────────────────────────
# 7-QAP.25 Slice 2 — Method-agnostic chart builders from OptimizationTrace
#
# Each builder takes a trace snapshot (``OptimizationTrace.to_dict()``)
# *or* an explicit value list, and returns either a Chart.js spec dict
# or ``None`` when the underlying series is missing/empty. Callers MUST
# treat ``None`` as "skip this artifact" — no fabricated charts.
#
# Palette mirrors the existing helpers so the JobHistoryView stays
# visually consistent across 7-QAP.23 and 7-QAP.25 artifacts.
# ──────────────────────────────────────────────────────────────────────

_P3_MAX_LINES = 12  # P3 hard-clamp from the §7-QAP.25 catalogue.


def _trace_iters(trace_or_iters: Any) -> List[Dict[str, Any]]:
    """Normalise input to a list of iter records.

    Accepts ``OptimizationTrace.to_dict()`` output (a dict with
    ``"iters"``), the raw ``iters`` list, or ``None``. Returns ``[]``
    when the trace is disabled or input is unrecognised.
    """
    if trace_or_iters is None:
        return []
    if isinstance(trace_or_iters, Mapping):
        if trace_or_iters.get("enabled") is False:
            return []
        iters = trace_or_iters.get("iters") or []
    else:
        try:
            iters = list(trace_or_iters)
        except TypeError:
            return []
    return [r for r in iters if isinstance(r, Mapping)]


def cost_vs_iter_chart(
    trace: Any,
    *,
    title: str = "Cost vs iteration",
    label: str = "objective",
    color: str = "#3b82f6",
) -> Optional[Dict[str, Any]]:
    """P1 — cost / objective trajectory (line)."""
    iters = _trace_iters(trace)
    pairs = [
        (int(r.get("i", idx)), float(r["cost"]))
        for idx, r in enumerate(iters)
        if r.get("cost") is not None
    ]
    if not pairs:
        return None
    labels = [str(i) for i, _ in pairs]
    data = [v for _, v in pairs]
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": label,
                    "data": data,
                    "borderColor": color,
                    "backgroundColor": color + "33",
                    "fill": False,
                    "tension": 0.2,
                    "pointRadius": 2,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "iteration"}},
                "y": {"title": {"display": True, "text": "objective"}},
            },
        },
    }


def gradient_norm_chart(
    trace: Any,
    *,
    title: str = "Gradient norm vs iteration",
    color: str = "#f59e0b",
) -> Optional[Dict[str, Any]]:
    """P2 — ``‖∇f‖`` trajectory on a log y-axis (line)."""
    iters = _trace_iters(trace)
    pairs = [
        (int(r.get("i", idx)), float(r["grad_norm"]))
        for idx, r in enumerate(iters)
        if r.get("grad_norm") is not None and float(r["grad_norm"]) > 0.0
    ]
    if not pairs:
        return None
    labels = [str(i) for i, _ in pairs]
    data = [v for _, v in pairs]
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "‖grad‖",
                    "data": data,
                    "borderColor": color,
                    "backgroundColor": color + "33",
                    "fill": False,
                    "tension": 0.2,
                    "pointRadius": 2,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "iteration"}},
                "y": {
                    "type": "logarithmic",
                    "title": {"display": True, "text": "‖∇f‖ (log)"},
                },
            },
        },
    }


def param_trajectory_chart(
    trace: Any,
    *,
    title: str = "Parameter trajectories",
    max_lines: int = _P3_MAX_LINES,
) -> Optional[Dict[str, Any]]:
    """P3 — per-parameter trajectories (multi-line, clamped to ≤12)."""
    iters = _trace_iters(trace)
    rows = [r for r in iters if isinstance(r.get("params"), list) and r["params"]]
    if not rows:
        return None
    n_params = min(max(len(r["params"]) for r in rows), int(max_lines))
    if n_params <= 0:
        return None
    labels = [str(int(r.get("i", idx))) for idx, r in enumerate(rows)]
    # Tableau-ish 12-colour palette so up to 12 parameter lines are distinguishable.
    palette = [
        "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
        "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
        "#6366f1", "#84cc16", "#06b6d4", "#a855f7",
    ]
    datasets: List[Dict[str, Any]] = []
    for j in range(n_params):
        series: List[Optional[float]] = []
        for r in rows:
            params = r["params"]
            if j < len(params):
                try:
                    series.append(float(params[j]))
                except (TypeError, ValueError):
                    series.append(None)
            else:
                series.append(None)
        color = palette[j % len(palette)]
        datasets.append(
            {
                "label": f"θ[{j}]",
                "data": series,
                "borderColor": color,
                "backgroundColor": color + "22",
                "fill": False,
                "tension": 0.2,
                "pointRadius": 1,
            }
        )
    return {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "iteration"}},
                "y": {"title": {"display": True, "text": "θ value"}},
            },
        },
    }


def objective_with_reference_chart(
    trace: Any,
    *,
    reference: Optional[float] = None,
    reference_label: str = "reference",
    title: str = "Objective vs reference",
    color: str = "#3b82f6",
    ref_color: str = "#94a3b8",
) -> Optional[Dict[str, Any]]:
    """P4 — objective trajectory with a horizontal reference line."""
    iters = _trace_iters(trace)
    pairs = [
        (int(r.get("i", idx)), float(r["cost"]))
        for idx, r in enumerate(iters)
        if r.get("cost") is not None
    ]
    if not pairs:
        return None
    labels = [str(i) for i, _ in pairs]
    data = [v for _, v in pairs]
    datasets: List[Dict[str, Any]] = [
        {
            "label": "objective",
            "data": data,
            "borderColor": color,
            "backgroundColor": color + "33",
            "fill": False,
            "tension": 0.2,
            "pointRadius": 2,
        }
    ]
    if reference is not None:
        try:
            ref = float(reference)
        except (TypeError, ValueError):
            ref = None
        if ref is not None and ref == ref and ref not in (float("inf"), float("-inf")):
            datasets.append(
                {
                    "label": reference_label,
                    "data": [ref] * len(data),
                    "borderColor": ref_color,
                    "backgroundColor": ref_color + "00",
                    "borderDash": [6, 4],
                    "fill": False,
                    "pointRadius": 0,
                }
            )
    return {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "iteration"}},
                "y": {"title": {"display": True, "text": "objective"}},
            },
        },
    }


def efficient_frontier_chart(
    candidate: Optional[Mapping[str, float]] = None,
    *,
    frontier: Optional[Sequence[Mapping[str, float]]] = None,
    title: str = "Risk × return — candidate vs frontier",
    candidate_color: str = "#ef4444",
    frontier_color: str = "#3b82f6",
) -> Optional[Dict[str, Any]]:
    """P7 — efficient-frontier scatter (risk × return).

    ``candidate`` and each ``frontier`` entry must expose at least
    ``risk`` and ``return`` (numeric). Returns ``None`` if neither
    point nor frontier carries valid data.
    """
    def _xy(p: Mapping[str, float]) -> Optional[Dict[str, float]]:
        if not isinstance(p, Mapping):
            return None
        try:
            x = float(p.get("risk"))  # type: ignore[arg-type]
            y = float(p.get("return"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x or y != y:
            return None
        if x in (float("inf"), float("-inf")) or y in (float("inf"), float("-inf")):
            return None
        return {"x": x, "y": y}

    datasets: List[Dict[str, Any]] = []
    if frontier:
        pts = [q for q in (_xy(p) for p in frontier) if q is not None]
        if pts:
            pts.sort(key=lambda q: q["x"])
            datasets.append(
                {
                    "label": "frontier",
                    "data": pts,
                    "borderColor": frontier_color,
                    "backgroundColor": frontier_color + "33",
                    "showLine": True,
                    "fill": False,
                    "tension": 0.1,
                    "pointRadius": 2,
                }
            )
    if candidate is not None:
        pt = _xy(candidate)
        if pt is not None:
            datasets.append(
                {
                    "label": "candidate",
                    "data": [pt],
                    "borderColor": candidate_color,
                    "backgroundColor": candidate_color,
                    "showLine": False,
                    "pointRadius": 6,
                    "pointStyle": "rectRot",
                }
            )
    if not datasets:
        return None
    return {
        "type": "scatter",
        "data": {"datasets": datasets},
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "risk (σ)"}},
                "y": {"title": {"display": True, "text": "return (μ)"}},
            },
        },
    }


def sample_energy_histogram_chart(
    energies: Optional[Sequence[float]],
    *,
    num_occurrences: Optional[Sequence[int]] = None,
    bins: int = 20,
    title: str = "Sample energy distribution",
    color: str = "#6366f1",
) -> Optional[Dict[str, Any]]:
    """P6 — D-Wave sampleset energy histogram (bar).

    Buckets ``energies`` into ``bins`` equal-width bins. When
    ``num_occurrences`` is provided (matching length), each sample
    contributes its occurrence count instead of 1. Returns ``None``
    if no finite energies are available.
    """
    if not energies:
        return None
    finite: List[float] = []
    weights: List[float] = []
    occ = num_occurrences if num_occurrences is not None else None
    for idx, e in enumerate(energies):
        try:
            v = float(e)
        except (TypeError, ValueError):
            continue
        if v != v or v in (float("inf"), float("-inf")):
            continue
        finite.append(v)
        if occ is not None and idx < len(occ):
            try:
                weights.append(float(occ[idx]))
            except (TypeError, ValueError):
                weights.append(1.0)
        else:
            weights.append(1.0)
    if not finite:
        return None
    lo = min(finite)
    hi = max(finite)
    if lo == hi:
        # Degenerate: every sample has the same energy → single bar.
        labels = [f"{lo:.4g}"]
        counts = [sum(weights)]
    else:
        nbins = max(1, int(bins))
        width = (hi - lo) / nbins
        counts = [0.0] * nbins
        for v, w in zip(finite, weights):
            j = int((v - lo) / width)
            if j == nbins:
                j = nbins - 1
            counts[j] += w
        labels = [f"{lo + i * width:.3g}" for i in range(nbins)]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "occurrences",
                    "data": [int(round(c)) if c == int(c) else round(c, 4) for c in counts],
                    "backgroundColor": color,
                    "borderColor": color,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "energy bin"}},
                "y": {"title": {"display": True, "text": "count"}, "beginAtZero": True},
            },
        },
    }


def best_energy_trace_chart(
    energies: Optional[Sequence[float]],
    *,
    title: str = "Best energy vs sample index",
    label: str = "running min energy",
    color: str = "#10b981",
) -> Optional[Dict[str, Any]]:
    """Cumulative-min energy line over sample order (D-Wave samplesets).

    Mirrors P1 semantics for non-iterative samplers — the "iteration
    axis" becomes the read/sample index and the "objective" becomes
    the best-so-far energy.
    """
    if not energies:
        return None
    running: List[float] = []
    best: Optional[float] = None
    for e in energies:
        try:
            v = float(e)
        except (TypeError, ValueError):
            continue
        if v != v or v in (float("inf"), float("-inf")):
            continue
        best = v if best is None else min(best, v)
        running.append(best)
    if not running:
        return None
    labels = [str(i) for i in range(len(running))]
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": label,
                    "data": running,
                    "borderColor": color,
                    "backgroundColor": color + "33",
                    "fill": False,
                    "tension": 0.0,
                    "pointRadius": 1,
                    "stepped": True,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "sample index"}},
                "y": {"title": {"display": True, "text": "energy"}},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# P5 — Constraint-violation trajectory (multi-line)
# ──────────────────────────────────────────────────────────────────────
def constraint_violation_chart(
    trace: Any,
    *,
    title: str = "Constraint violations vs iteration",
    use_log_y: bool = True,
    max_lines: int = 8,
) -> Optional[Dict[str, Any]]:
    """P5 — per-constraint violation magnitude over iterations.

    Reads ``iters[*].constraint_violations`` (a dict of
    ``{name: float}``). Each distinct constraint key becomes its own
    line. ``use_log_y`` (default) plots on a log y-axis so very small
    residuals after constraint repair are still visible; the builder
    silently falls back to a linear axis when any series contains
    non-positive values that would break the log scale.
    Returns ``None`` when no iteration carries a violations dict.
    """
    iters = _trace_iters(trace)
    rows = [
        r for r in iters
        if isinstance(r.get("constraint_violations"), Mapping)
        and r["constraint_violations"]
    ]
    if not rows:
        return None
    # Discover constraint names (preserve first-seen order, cap to max_lines).
    names: List[str] = []
    for r in rows:
        for k in r["constraint_violations"].keys():
            if k not in names:
                names.append(str(k))
        if len(names) >= int(max_lines):
            names = names[: int(max_lines)]
            break
    if not names:
        return None
    labels = [str(int(r.get("i", idx))) for idx, r in enumerate(rows)]
    palette = [
        "#ef4444", "#f59e0b", "#3b82f6", "#10b981",
        "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
    ]
    datasets: List[Dict[str, Any]] = []
    has_non_positive = False
    for j, name in enumerate(names):
        series: List[Optional[float]] = []
        for r in rows:
            cv = r["constraint_violations"]
            raw = cv.get(name)
            try:
                v = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                v = None
            if v is not None and (v != v or v in (float("inf"), float("-inf"))):
                v = None
            if v is not None and v <= 0.0:
                has_non_positive = True
            series.append(v)
        color = palette[j % len(palette)]
        datasets.append(
            {
                "label": name,
                "data": series,
                "borderColor": color,
                "backgroundColor": color + "22",
                "fill": False,
                "tension": 0.2,
                "pointRadius": 2,
            }
        )
    y_axis: Dict[str, Any] = {"title": {"display": True, "text": "violation"}}
    if use_log_y and not has_non_positive:
        y_axis["type"] = "logarithmic"
        y_axis["title"]["text"] = "violation (log)"
    return {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": True, "position": "bottom"},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "iteration"}},
                "y": y_axis,
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# P8 — Allocation drift delta-bar (final − baseline)
# ──────────────────────────────────────────────────────────────────────
def allocation_drift_chart(
    final_weights: Mapping[str, float],
    baseline_weights: Mapping[str, float],
    *,
    title: str = "Allocation drift vs baseline",
    over_color: str = "#10b981",
    under_color: str = "#ef4444",
    epsilon: float = 1e-9,
) -> Optional[Dict[str, Any]]:
    """P8 — bar chart of ``Δw = final − baseline`` per asset.

    Bars are coloured green when overweight (Δ > 0) and red when
    underweight (Δ < 0). Assets where both weights round to zero are
    omitted. Returns ``None`` when there is no overlap with non-zero
    drift.
    """
    if not isinstance(final_weights, Mapping) or not isinstance(baseline_weights, Mapping):
        return None
    assets = list(dict.fromkeys([*final_weights.keys(), *baseline_weights.keys()]))
    rows: List[tuple[str, float]] = []
    for a in assets:
        try:
            fw = float(final_weights.get(a, 0.0) or 0.0)
            bw = float(baseline_weights.get(a, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        delta = fw - bw
        if abs(fw) < epsilon and abs(bw) < epsilon:
            continue
        if abs(delta) < epsilon:
            # Keep zero-drift rows so users can see "no change" assets,
            # but only when at least one side has a meaningful weight.
            delta = 0.0
        rows.append((str(a), round(delta, 6)))
    if not rows:
        return None
    rows.sort(key=lambda kv: kv[1], reverse=True)
    labels = [a for a, _ in rows]
    data = [d for _, d in rows]
    bar_colors = [over_color if d >= 0 else under_color for d in data]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Δ weight",
                    "data": data,
                    "backgroundColor": bar_colors,
                    "borderColor": bar_colors,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "asset"}},
                "y": {"title": {"display": True, "text": "Δ weight (final − baseline)"}},
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# P9 — Per-iter wallclock bar
# ──────────────────────────────────────────────────────────────────────
def iter_wallclock_chart(
    trace: Any,
    *,
    title: str = "Wallclock per iteration",
    color: str = "#06b6d4",
) -> Optional[Dict[str, Any]]:
    """P9 — per-iteration elapsed time in seconds (bar).

    Reads ``iters[*].wall_s`` (populated by every ``record`` call) and
    emits one bar per iteration. Useful for spotting cold-start spikes
    in transpilation or warmup phases. Returns ``None`` when no
    iteration carries a positive ``wall_s``.
    """
    iters = _trace_iters(trace)
    pairs: List[tuple[int, float]] = []
    for idx, r in enumerate(iters):
        raw = r.get("wall_s")
        try:
            v = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            v = None
        if v is None or v != v or v in (float("inf"), float("-inf")) or v < 0.0:
            continue
        pairs.append((int(r.get("i", idx)), v))
    if not pairs or all(v == 0.0 for _, v in pairs):
        return None
    labels = [str(i) for i, _ in pairs]
    data = [round(v, 6) for _, v in pairs]
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "seconds",
                    "data": data,
                    "backgroundColor": color,
                    "borderColor": color,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "iteration"}},
                "y": {
                    "title": {"display": True, "text": "wallclock (s)"},
                    "beginAtZero": True,
                },
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────
# P10 — Metric / Fisher / Hessian spectrum
# ──────────────────────────────────────────────────────────────────────
def spectrum_chart(
    eigenvalues: Optional[Sequence[float]],
    *,
    title: str = "Eigenvalue spectrum",
    label: str = "|λ|",
    use_log_y: bool = True,
    color: str = "#a855f7",
    negative_color: str = "#ef4444",
    top_k: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """P10 — sorted spectrum of an information metric / Fisher / Hessian.

    Each bar is ``|λ_i|`` ranked descending. Negative eigenvalues are
    plotted with their absolute value but coloured red so saddle-point
    structure stays visible. Log y-axis by default; falls back to
    linear when any eigenvalue is exactly zero. ``top_k`` clamps to
    the dominant eigenvalues (useful for very high-dim Hessians).
    Returns ``None`` when no finite eigenvalues are supplied.
    """
    if not eigenvalues:
        return None
    finite: List[float] = []
    for e in eigenvalues:
        try:
            v = float(e)
        except (TypeError, ValueError):
            continue
        if v != v or v in (float("inf"), float("-inf")):
            continue
        finite.append(v)
    if not finite:
        return None
    finite.sort(key=lambda v: abs(v), reverse=True)
    if top_k is not None and int(top_k) > 0:
        finite = finite[: int(top_k)]
    labels = [str(i + 1) for i in range(len(finite))]
    data = [round(abs(v), 8) for v in finite]
    bar_colors = [negative_color if v < 0.0 else color for v in finite]
    y_axis: Dict[str, Any] = {
        "title": {"display": True, "text": "|λ|"},
        "beginAtZero": True,
    }
    if use_log_y and all(v > 0.0 for v in data):
        # Log axes can't begin at zero — drop beginAtZero when going log.
        y_axis = {"type": "logarithmic", "title": {"display": True, "text": "|λ| (log)"}}
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": label,
                    "data": data,
                    "backgroundColor": bar_colors,
                    "borderColor": bar_colors,
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "plugins": {
                "title": {"display": True, "text": title},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "rank"}},
                "y": y_axis,
            },
        },
    }


__all__ = [
    "convergence_chart",
    "bitstring_histogram_chart",
    "weights_bar_chart",
    "attach_diagnostics_step",
    "attach_optimization_trace_step",
    # 7-QAP.25 Slice 2
    "cost_vs_iter_chart",
    "gradient_norm_chart",
    "param_trajectory_chart",
    "objective_with_reference_chart",
    "efficient_frontier_chart",
    # 7-QAP.25 Slice 2 (P5/P8/P9/P10)
    "constraint_violation_chart",
    "allocation_drift_chart",
    "iter_wallclock_chart",
    "spectrum_chart",
    # 7-QAP.25 Slice 4b — D-Wave sampleset diagnostics
    "sample_energy_histogram_chart",
    "best_energy_trace_chart",
]
