"""Diagnostic artifact builders (7-QAP.23 Slice B) and the shared
optimisation-trace recorder (7-QAP.25 Slice 1)."""
from core.diagnostics.artifacts import (
    allocation_drift_chart,
    attach_diagnostics_step,
    attach_optimization_trace_step,
    best_energy_trace_chart,
    bitstring_histogram_chart,
    constraint_violation_chart,
    convergence_chart,
    cost_vs_iter_chart,
    efficient_frontier_chart,
    gradient_norm_chart,
    iter_wallclock_chart,
    objective_with_reference_chart,
    param_trajectory_chart,
    sample_energy_histogram_chart,
    spectrum_chart,
    weights_bar_chart,
)
from core.diagnostics.trace import (
    DEFAULT_MAX_ITERS,
    OptimizationTrace,
    is_globally_enabled,
)
from core.diagnostics.vendor_links import (
    VENDOR_LABELS,
    build_portal_url,
    vendor_link_artifact,
    vendor_link_artifacts_from_runs,
)

__all__ = [
    "attach_diagnostics_step",
    "attach_optimization_trace_step",
    "bitstring_histogram_chart",
    "convergence_chart",
    "weights_bar_chart",
    # 7-QAP.25 Slice 2 chart builders
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
    # 7-QAP.25 Slice 1 recorder
    "OptimizationTrace",
    "DEFAULT_MAX_ITERS",
    "is_globally_enabled",
    # F24 — Vendor portal deep-links
    "VENDOR_LABELS",
    "build_portal_url",
    "vendor_link_artifact",
    "vendor_link_artifacts_from_runs",
]
