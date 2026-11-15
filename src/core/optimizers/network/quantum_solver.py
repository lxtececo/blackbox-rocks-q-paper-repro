"""Quantum solve path for facility location (F32).

Thin by design: build the QUBO, hand it to the *fleet* sampler substrate, decode.
``fleet.samplers.sample_qubo`` only touches ``q``/``Q``/``const``/``n_vars``/
``to_ising``, all of which :class:`NetworkQubo` provides, so all four backends
(numpy SA, D-Wave Ocean, Leap hybrid, QAOA via Aer) work here unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from core.optimizers.fleet.samplers import (
    gate_qubit_cap,
    sample_qubo,
)
from core.optimizers.network.classical import solve_classical
from core.optimizers.network.evaluate import build_solution, finalise_solution
from core.optimizers.network.models import NetworkProblem, NetworkSolution
from core.optimizers.network.qubo import build_network_qubo, qubo_summary

logger = logging.getLogger(__name__)

# The pure-numpy annealer costs O(sweeps * n_vars^2 * restarts): every sweep
# attempts n flips and each flip evaluates a length-n dot product. Measured on
# this hardware it clears ~1e8 of those units per second, so the 2440-variable
# national scenario takes ~186s - far past any gateway timeout. Ocean's neal
# solves the same instance in ~12s because its inner loop is compiled. Refuse
# the combination outright with a pointer to the method that can do the job,
# instead of letting the request hang until something upstream kills it.
MAX_NUMPY_ANNEAL_VARS = 400


def solve_quantum(
    problem: NetworkProblem,
    *,
    backend: str = "numpy",
    platform: str = "cpu",
    execution_mode: str = "simulator",
    sampler_options: Optional[Dict[str, Any]] = None,
    classical_polish: bool = True,
    seed: int = 42,
) -> Tuple[NetworkSolution, Dict[str, Any]]:
    """Sample the CFLP QUBO and decode it. Returns ``(solution, diagnostics)``."""
    qubo = build_network_qubo(problem)
    summary = qubo_summary(qubo)

    if backend == "gate":
        hardware = (execution_mode or "").strip().lower() == "real"
        cap = gate_qubit_cap(execution_mode, platform)
        if qubo.n_vars > cap:
            raise ValueError(
                f"gate-based sampling in {'real' if hardware else 'simulator'} "
                f"mode is limited to {cap} qubits; this scenario needs "
                f"{qubo.n_vars} ({len(problem.sites)} sites x "
                f"{len(problem.demand_points)} demand points plus slack). Run "
                "the 'Pilot - 2 of 4 sites' scenario to see QAOA end to end, "
                "or use an annealing method, which solves this instance at "
                "full size."
            )

    if backend == "numpy" and qubo.n_vars > MAX_NUMPY_ANNEAL_VARS:
        raise ValueError(
            f"the reference simulated-annealing sampler is limited to "
            f"{MAX_NUMPY_ANNEAL_VARS} variables; this instance needs "
            f"{qubo.n_vars}. Use the 'qubo-annealing' method, which runs the "
            "same QUBO on a compiled annealer, or pick a smaller scenario."
        )

    opts = dict(sampler_options or {})
    # The shared fleet sampler defaults (500 sweeps / 5 restarts) are tuned for
    # small vehicle-routing QUBOs. A CFLP one-hot landscape is far more rugged
    # - flipping a single x_ij always breaks an assignment constraint, so the
    # chain has to climb a full penalty barrier to move between feasible
    # networks - and needs proportionally more sweeps to mix. Measured on the
    # 255-variable metro scenario: 500 sweeps lands ~45% above the classical
    # baseline, 2000 sweeps lands ~2.5% above it. Extra *restarts* bought
    # nothing over the same budget, so scale sweeps only.
    if backend in ("numpy", "dwave"):
        opts.setdefault("sweeps", int(min(max(8 * qubo.n_vars, 500), 4000)))
    if backend == "numpy" and qubo.n_vars > 150:
        # Extra restarts bought nothing measurable on the metro instance while
        # multiplying runtime, so spend the budget on sweeps instead.
        opts.setdefault("restarts", 3)

    outcome = sample_qubo(
        qubo,
        backend=backend,
        platform=platform,
        execution_mode=execution_mode,
        seed=seed,
        **opts,
    )

    assignment, unassigned = qubo.decode(outcome.bits)
    solution = finalise_solution(
        problem,
        build_solution(
            problem, assignment, unassigned, cost_matrix=qubo.cost_matrix
        ),
    )
    raw_cost = solution.total_cost
    raw_open = list(solution.open_site_ids)
    # The raw cost is NOT comparable to the baseline on its own: the QUBO's
    # capacity penalty is soft, so the sampler routinely buys a lower total by
    # overfilling a site. Carry the feasibility verdict alongside the number so
    # no caller can quote the cost without the caveat.
    raw_metrics = dict(solution.metrics or {})
    raw_feasible = bool(raw_metrics.get("feasible"))
    raw_overflow = float(raw_metrics.get("capacity_overflow") or 0.0)

    polished = False
    if classical_polish:
        # Two things are going on here, and they are reported separately so the
        # quantum contribution is not overstated.
        #
        # 1. REPAIR: a sampler that violated the one-hot leaves demand
        #    stranded. Re-running the cheapest-open-site assignment over the
        #    sites the sampler chose turns a broken network into a valid one.
        # 2. REFINE: the same drop/swap local search the classical baseline
        #    uses, seeded from the sampler's open set. Single-spin-flip
        #    annealing mixes badly on a one-hot CFLP landscape - every single
        #    flip breaks an assignment constraint, so the chain must climb a
        #    full penalty barrier to move between feasible networks, and raw
        #    sample quality swings wildly run to run. Refining the proposed
        #    open set is the standard hybrid pattern; the honest framing is
        #    that the sampler proposes structure and the classical operator
        #    polishes it. `raw_total_cost` below preserves the unpolished
        #    number so the split stays auditable.
        from core.optimizers.network.classical import (
            _assign_to_open,
            refine_open_set,
        )

        chosen = [
            idx
            for idx, site in enumerate(problem.sites)
            if site.id in set(solution.open_site_ids)
        ]
        if chosen:
            refined, _, drops, swaps, _ = refine_open_set(
                problem, chosen, qubo.cost_matrix
            )
            repaired, _ = _assign_to_open(problem, refined, qubo.cost_matrix)
            solution = finalise_solution(
                problem,
                build_solution(
                    problem, repaired, [], cost_matrix=qubo.cost_matrix
                ),
            )
            polished = True

    diagnostics: Dict[str, Any] = {
        "algorithm": f"CFLP QUBO / {outcome.backend_name}",
        "qubo": summary,
        "classical_polish_applied": polished,
        "raw_unassigned": len(unassigned),
        "raw_total_cost": raw_cost,
        "raw_open_site_ids": raw_open,
        "raw_feasible": raw_feasible,
        "raw_capacity_overflow": raw_overflow,
        "polish_improvement": (
            round(raw_cost - solution.total_cost, 3) if polished else 0.0
        ),
        **outcome.to_dict(),
    }
    return solution, diagnostics


def solve_baseline(problem: NetworkProblem) -> Tuple[NetworkSolution, Dict[str, Any]]:
    """The classical comparison baseline on the identical instance."""
    return solve_classical(problem)
