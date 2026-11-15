"""F6 — Fleet routing (CVRP with capacity + time windows).

Second industry vertical alongside the finance portfolio stack. The
package mirrors the portfolio substrate: a problem model, a classical
baseline, a QUBO encoding routed through the shared
``core.quantum_math`` authority, and a disruption/replan engine.

Public surface::

    from core.optimizers.fleet import solve_fleet_problem, build_scenario

Solver taxonomy (mirrors the finance paradigms):

* ``classical``       — Clarke-Wright savings + Or-opt/2-opt local search.
* ``qubo_annealing``  — cluster→vehicle assignment QUBO, annealed.
* ``qaoa``            — same QUBO, gate-based sampling.

Everything below the assignment layer (intra-route sequencing, feasibility,
KPI accounting) is shared, so a quantum run and a classical run are
directly comparable on the same objective.
"""

from core.optimizers.fleet.disruption import (
    DISRUPTION_KINDS,
    apply_disruption,
    replan_after_disruption,
)
from core.optimizers.fleet.models import (
    Depot,
    DisruptionEvent,
    FleetProblem,
    FleetSolution,
    Route,
    RouteVisit,
    Stop,
    Vehicle,
)
from core.optimizers.fleet.scenario import build_scenario
from core.optimizers.fleet.service import FLEET_METHODS, solve_fleet_problem

__all__ = [
    "DISRUPTION_KINDS",
    "FLEET_METHODS",
    "Depot",
    "DisruptionEvent",
    "FleetProblem",
    "FleetSolution",
    "Route",
    "RouteVisit",
    "Stop",
    "Vehicle",
    "apply_disruption",
    "build_scenario",
    "replan_after_disruption",
    "solve_fleet_problem",
]
