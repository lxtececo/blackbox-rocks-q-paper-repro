"""Generic post-processing for portfolio optimisation.

See :mod:`core.postprocessing.refinement_engine` for the public API.
"""

from .classical_refinement import refine_selected_weights
from .refinement_engine import (
    NATIVE_CARDINALITY_METHODS,
    build_ising_from_portfolio,
    first_improvement_swap_continuous,
    ising_energy,
    refine_binary_solution,
    refine_continuous_weights,
    repair_cardinality,
    should_refine,
    swap_local_search,
)

__all__ = [
    "NATIVE_CARDINALITY_METHODS",
    "build_ising_from_portfolio",
    "first_improvement_swap_continuous",
    "ising_energy",
    "refine_binary_solution",
    "refine_continuous_weights",
    "refine_selected_weights",
    "repair_cardinality",
    "should_refine",
    "swap_local_search",
]
