"""Quantum amplitude estimation for option pricing, implemented natively
against core Qiskit primitives.

Algorithm
---------
Maximum Likelihood Amplitude Estimation (MLAE), Suzuki et al., *Amplitude
estimation without phase estimation*, Quantum Inf. Process. 19, 75 (2020).

A state-preparation operator ``A`` loads the discretized terminal-price
distribution and encodes the normalised discounted payoff into the
amplitude of one objective qubit, so that

    A|0> = sqrt(1-a)|psi_0>|0> + sqrt(a)|psi_1>|1>,   a = sum_i p_i f_i

where ``f_i`` is the payoff in bin ``i`` normalised to [0, 1]. Recovering
``a`` recovers the option price via ``price = a * payoff_ceiling``.

MLAE measures the objective qubit after applying the Grover operator
``Q = A S_0 A^dagger S_f`` a varying number of times ``m_k``. Because

    P(objective = 1 | m) = sin^2((2m+1) * theta),   a = sin^2(theta)

the amplified measurements pin down ``theta`` far more sharply than
sampling ``A`` alone. Fitting all rounds jointly by maximum likelihood
yields an estimator whose error falls as O(1/N_queries) rather than the
O(1/sqrt(N_samples)) of classical Monte Carlo - the quadratic speedup.

Why MLAE and not a library QAE
------------------------------
Implementing it here keeps the vertical free of ``qiskit-algorithms``
(which is absent from the deployed base image), and - more importantly -
makes the oracle-query count an exact, auditable quantity computed from
the schedule actually executed, rather than a figure inferred from a
target precision. Nothing in this module estimates or extrapolates the
work performed.

Encoding accuracy
-----------------
The payoff is encoded with exact rotation angles ``2*arcsin(sqrt(f_i))``
via a uniformly-controlled (multiplexed) RY, so ``P(1) = sum_i p_i f_i``
holds exactly. This avoids the small-angle linear approximation used by
the textbook construction, removing a systematic bias that would
otherwise be indistinguishable from estimator error.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.optimizers.montecarlo.grid import PriceGrid

logger = logging.getLogger(__name__)

#: Objective-qubit measurement register name, read back from the sampler.
_CREG = "obj"

#: Deepest Grover amplification worth attempting on hardware. Depth grows
#: linearly in the power, and past ~8 iterations a NISQ device's output is
#: closer to uniform than to sin^2((2m+1)theta).
MAX_HARDWARE_GROVER_POWER = 8


class QaeExecutionError(RuntimeError):
    """Raised when the QAE circuit cannot be built or executed.

    Deliberately fatal. An earlier revision of this vertical fell back to
    the exact discretized expectation and returned it labelled as a
    quantum result, complete with a synthesized oracle-query count. That
    turns a missing dependency into a fabricated speedup claim, so the
    failure is now surfaced instead of absorbed.
    """


def exponential_schedule(max_power: int) -> List[int]:
    """Grover powers [0, 1, 2, 4, 8, ...] up to ``max_power`` (Suzuki EIS)."""
    powers = [0]
    step = 1
    while step <= max_power:
        powers.append(step)
        step *= 2
    return powers


def distribution_ry_layers(probabilities: np.ndarray):
    """Yield ``(target_qubit, angles)`` loading ``sqrt(probabilities)``.

    Qiskit's ``StatePreparation`` routes through the generic complex
    isometry decomposition, which repeatedly divides by sub-tree norms.
    Once the log-normal tail bins drop below ~1e-7 those quotients stop
    being unitary to within the transpiler's tolerance and synthesis
    fails outright - that is n_qubits >= 8 on this grid. Amplitudes here
    are square roots of probabilities, so they are real and non-negative,
    and a Mottonen uniformly-controlled Ry tree loads them exactly at any
    width. Layers come coarsest split first.
    """
    amplitudes = np.sqrt(np.asarray(probabilities, dtype=float))
    n = int(round(math.log2(amplitudes.size)))

    # norms[k][j] is the norm of the amplitude block of width 2**k at j.
    norms = [amplitudes]
    while norms[-1].size > 1:
        block = norms[-1]
        norms.append(np.hypot(block[0::2], block[1::2]))

    for level in range(n):
        target = n - 1 - level
        block = norms[target]
        angles = 2.0 * np.arctan2(block[1::2], block[0::2])
        yield target, [float(angle) for angle in angles]


def build_state_preparation(grid: PriceGrid):
    """Build ``A`` on ``n_qubits`` distribution qubits + 1 objective qubit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate

    n = grid.n_qubits
    circuit = QuantumCircuit(n + 1, name="A")
    for target, angles in distribution_ry_layers(grid.probabilities):
        if target == n - 1:
            circuit.ry(angles[0], target)
        else:
            circuit.append(UCRYGate(angles), [target, *range(target + 1, n)])

    angles = 2.0 * np.arcsin(np.sqrt(np.clip(grid.normalized_payoffs, 0.0, 1.0)))
    # UCRYGate takes the target as qubit 0 and the controls as qubits 1..n,
    # with angle_list indexed by the control basis state.
    circuit.append(UCRYGate(list(map(float, angles))), [n, *range(n)])
    return circuit


def build_grover_operator(state_prep, n_qubits: int):
    """Q = A S_0 A^dagger S_f, with S_f phase-flipping objective qubit = |1>."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import GroverOperator

    oracle = QuantumCircuit(n_qubits + 1, name="S_f")
    oracle.z(n_qubits)
    return GroverOperator(oracle, state_preparation=state_prep, name="Q")


def _measurement_circuits(grid: PriceGrid, powers: Sequence[int]):
    """One circuit per Grover power, measuring only the objective qubit."""
    from qiskit import ClassicalRegister, QuantumCircuit

    n = grid.n_qubits
    state_prep = build_state_preparation(grid)
    grover = build_grover_operator(state_prep, n)

    circuits = []
    for power in powers:
        creg = ClassicalRegister(1, _CREG)
        circuit = QuantumCircuit(n + 1)
        circuit.add_register(creg)
        circuit.compose(state_prep, inplace=True)
        for _ in range(power):
            circuit.compose(grover, inplace=True)
        circuit.measure(n, creg[0])
        circuits.append(circuit)
    return circuits


def _run_sampler(
    circuits,
    shots: int,
    seed: int,
    *,
    platform: Optional[str] = None,
    execution_mode: str = "simulator",
):
    """Execute each circuit and return the |1> count on the objective qubit.

    Returns ``(hits, run)`` where ``run`` is the
    :class:`~core.optimizers.quantum.shared.circuit_runner.CircuitRunResult`
    carrying the backend provenance. The whole Grover schedule is submitted
    as one batch, so a hardware run queues once rather than once per depth.
    """
    from core.optimizers.quantum.shared.circuit_runner import (
        run_measured_circuits,
    )

    run = run_measured_circuits(
        circuits,
        shots=shots,
        platform=platform,
        execution_mode=execution_mode,
        seed=seed,
        job_label="mlae grover schedule",
        min_qubits=circuits[0].num_qubits if circuits else 1,
        simulator="statevector",
    )
    return [int(counts.get("1", 0)) for counts in run.counts], run


def _log_likelihood(
    thetas: np.ndarray, powers: Sequence[int], shots: int, hits: Sequence[int]
) -> np.ndarray:
    """Joint log-likelihood of all MLAE rounds over a grid of theta."""
    total = np.zeros_like(thetas)
    for power, hit in zip(powers, hits):
        angle = (2 * power + 1) * thetas
        sin_sq = np.clip(np.sin(angle) ** 2, 1e-300, 1.0)
        cos_sq = np.clip(np.cos(angle) ** 2, 1e-300, 1.0)
        total += hit * np.log(sin_sq) + (shots - hit) * np.log(cos_sq)
    return total


def _maximize_likelihood(
    powers: Sequence[int], shots: int, hits: Sequence[int], coarse_points: int = 60_000
) -> float:
    """Find the ML theta in [0, pi/2].

    The likelihood oscillates once per Grover power, so a coarse grid
    dense enough to resolve the highest-frequency term is swept first,
    then refined locally. A naive local optimizer would lock onto the
    wrong lobe.
    """
    upper = math.pi / 2
    coarse = np.linspace(1e-12, upper - 1e-12, coarse_points)
    best = float(coarse[int(np.argmax(_log_likelihood(coarse, powers, shots, hits)))])

    span = upper / coarse_points
    fine = np.linspace(max(best - span, 1e-12), min(best + span, upper - 1e-12), 2_000)
    return float(fine[int(np.argmax(_log_likelihood(fine, powers, shots, hits)))])


def _fisher_standard_error(theta: float, powers: Sequence[int], shots: int) -> float:
    """Standard error of the amplitude from the MLAE Fisher information.

    Per shot at Grover power m the Fisher information in theta is
    4*(2m+1)^2, so I(theta) = sum_k 4*N_k*(2m_k+1)^2. Propagating to
    a = sin^2(theta) gives da/dtheta = sin(2*theta).
    """
    information = sum(4.0 * shots * (2 * power + 1) ** 2 for power in powers)
    if information <= 0:
        return float("inf")
    return abs(math.sin(2 * theta)) / math.sqrt(information)


def estimate_price_via_qae(
    grid: PriceGrid,
    *,
    shots_per_power: int = 100,
    max_grover_power: int = 32,
    seed: int = 42,
    confidence_level: float = 0.95,
    platform: Optional[str] = None,
    execution_mode: str = "simulator",
) -> Dict[str, Any]:
    """Price the option encoded in ``grid`` by MLAE.

    Raises ``QaeExecutionError`` if the circuit cannot be built or run;
    no classical substitute is ever returned from this function.
    """
    from scipy.stats import norm

    started = time.monotonic()
    hardware = (execution_mode or "simulator").strip().lower() == "real"
    requested_power = int(max_grover_power)
    if hardware:
        # Each Grover iteration re-applies A and its inverse, so depth grows
        # linearly in the power. Past this point a NISQ device returns a
        # coin flip and the extra amplification costs precision instead of
        # buying it. Clamp rather than refuse, and report what ran.
        max_grover_power = min(requested_power, MAX_HARDWARE_GROVER_POWER)
    powers = exponential_schedule(max_grover_power)

    try:
        circuits = _measurement_circuits(grid, powers)
        hits, run = _run_sampler(
            circuits,
            shots_per_power,
            seed,
            platform=platform,
            execution_mode=execution_mode,
        )
    except QaeExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, fatal error
        raise QaeExecutionError(
            f"quantum amplitude estimation circuit failed to execute: {exc}"
        ) from exc

    theta = _maximize_likelihood(powers, shots_per_power, hits)
    amplitude = math.sin(theta) ** 2
    ceiling = grid.payoff_ceiling

    # Re-fit on successive prefixes of the schedule to expose how the
    # estimate tightened as queries accumulated. This needs no extra
    # circuit execution - it reuses the measurements already taken - and
    # yields a convergence curve measured rather than asserted.
    convergence: Dict[str, List[Any]] = {"oracle_queries": [], "estimate": [], "std_error": []}
    for depth in range(1, len(powers) + 1):
        prefix = powers[:depth]
        prefix_theta = _maximize_likelihood(prefix, shots_per_power, hits[:depth])
        convergence["oracle_queries"].append(
            int(sum(shots_per_power * (2 * p + 1) for p in prefix))
        )
        convergence["estimate"].append(round(math.sin(prefix_theta) ** 2 * ceiling, 6))
        convergence["std_error"].append(
            round(_fisher_standard_error(prefix_theta, prefix, shots_per_power) * ceiling, 6)
        )

    amplitude_stderr = _fisher_standard_error(theta, powers, shots_per_power)
    z_score = float(norm.ppf(0.5 + confidence_level / 2.0))
    price = amplitude * ceiling
    half_width = z_score * amplitude_stderr * ceiling

    # Each shot at power m applies A (or its inverse) 2m+1 times; this is
    # the standard oracle-query metric for amplitude estimation, summed
    # over the schedule that actually ran.
    oracle_queries = int(sum(shots_per_power * (2 * p + 1) for p in powers))
    deepest = circuits[-1]

    return {
        "method": "quantum_amplitude_estimation",
        "algorithm": "maximum_likelihood_amplitude_estimation",
        "price": round(price, 6),
        "std_error": round(amplitude_stderr * ceiling, 6),
        "ci_low": round(price - half_width, 6),
        "ci_high": round(price + half_width, 6),
        "confidence_level": confidence_level,
        "estimated_amplitude": round(amplitude, 8),
        "num_oracle_queries": oracle_queries,
        "num_qubits": int(grid.n_qubits + 1),
        "discretization_bins": grid.n_bins,
        "grover_schedule": list(powers),
        "shots_per_power": int(shots_per_power),
        "hits_per_power": [int(h) for h in hits],
        "convergence": convergence,
        "deepest_circuit_depth": int(deepest.depth()),
        "deepest_circuit_gate_count": int(sum(deepest.count_ops().values())),
        "duration_seconds": round(time.monotonic() - started, 4),
        "max_grover_power": int(max_grover_power),
        "max_grover_power_requested": requested_power,
        **run.provenance(),
    }
