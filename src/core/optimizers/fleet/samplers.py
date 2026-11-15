"""QUBO samplers for the fleet vertical.

Every backend consumes the same pair-form QUBO produced by
:mod:`core.optimizers.fleet.qubo`, so the resulting plans are directly
comparable. Backends degrade gracefully: if an SDK or a credential is
missing the sampler falls back to the built-in NumPy annealer and records
why in ``SampleOutcome.fallback_reason`` instead of failing the job.

Backends
--------
``numpy``    Seeded simulated annealing. No third-party dependency, always
             available, used as the reference and the universal fallback.
``dwave``    ``dimod`` / Ocean. Simulated annealing in ``simulator`` mode,
             Leap hybrid or QPU + embedding in ``real`` mode.
``gate``     QAOA on the Ising form. Aer in ``simulator`` mode; IBM Quantum
             or IonQ hardware in ``real`` mode. Small instances only.

A ``real`` request never degrades to a simulator. Silently answering a
hardware run with simulated counts would bill QPU intent against a free
local solve and make the provenance in Job History a lie, so the job fails
with the provider's reason instead.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.optimizers.fleet.qubo import FleetQubo

logger = logging.getLogger(__name__)

#: Gate-based sampling is state-vector bound, and it also has to finish. Both
#: modes train the QAOA parameters on Aer, so both pay local simulation time.
#: Measured in a 2-CPU/4GiB worker with Aer's thread pool pinned to the cgroup
#: quota, on a QAOA layer at the ~30% coupling density these QUBOs have: 20
#: qubits costs 0.23s per objective evaluation (14s for a 60-step COBYLA
#: loop), 22 costs 0.94s (56s), 24 costs 4.5s (4.5min), 27 costs ~47min.
#: 20 is the widest that answers while a user waits on the response; a real
#: run is dispatched and polled, so it can afford the 24-qubit training cost.
#: Raise either with the matching env var on a bigger worker.
INTERACTIVE_GATE_QUBITS = 20
BATCH_GATE_QUBITS = 24

#: Problem size is NOT capped by any of these - the fleet solver decomposes
#: into batches of the allowed width, so a 620-stop scenario runs as many
#: small circuits rather than one impossible one.


def _budget(name: str, default: int) -> int:
    override = os.getenv(name, "").strip()
    if override.isdigit():
        return max(1, int(override))
    return default


def gate_qubit_cap(execution_mode: str, platform: Optional[str] = None) -> int:
    """Widest QAOA circuit this pod may build for the given request.

    Three limits apply and the tightest wins: the state vector has to fit in
    this container's memory, the local training loop has to finish inside the
    mode's time budget, and a real run must also fit the device it targets.
    """
    from core.optimizers.quantum.shared.circuit_runner import (  # noqa: PLC0415
        device_qubit_width,
        statevector_qubit_cap,
    )

    hardware = (execution_mode or "").strip().lower() == "real"
    cap = min(
        statevector_qubit_cap(),
        _budget("BATCH_GATE_QUBITS", BATCH_GATE_QUBITS)
        if hardware
        else _budget("INTERACTIVE_GATE_QUBITS", INTERACTIVE_GATE_QUBITS),
    )
    if not hardware:
        return cap
    width = device_qubit_width(platform, hardware=True)
    return min(cap, width) if width else cap


#: Advertised ceilings for the pickers. `gate_qubit_cap` is the authority; a
#: request that names a narrow device is held to that device instead.
MAX_GATE_QUBITS = gate_qubit_cap("simulator")
MAX_REAL_GATE_QUBITS = gate_qubit_cap("real")


@dataclass
class SampleOutcome:
    """Normalised sampler result shared by all fleet backends."""

    bits: np.ndarray
    energy: float
    backend_name: str
    platform: str
    execution_mode: str
    reads: int = 0
    energy_trace: List[float] = field(default_factory=list)
    elapsed_ms: int = 0
    fallback_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "platform": self.platform,
            "execution_mode": self.execution_mode,
            "energy": round(float(self.energy), 6),
            "reads": self.reads,
            "elapsed_ms": self.elapsed_ms,
            "fallback_reason": self.fallback_reason,
            "energy_trace": [round(float(e), 6) for e in self.energy_trace],
            **self.extra,
        }


def _auto_beta_range(q: np.ndarray, Qs: np.ndarray) -> Tuple[float, float]:
    """Neal-style inverse-temperature schedule bounds.

    A fixed beta range is useless here because fleet QUBO coefficients scale
    with route kilometres: with beta=0.1 and deltas in the hundreds the
    annealer is already frozen on sweep 1 and degenerates to greedy descent.
    The hot bound accepts a typical worst-case flip ~50% of the time; the
    cold bound makes the smallest meaningful flip effectively frozen.
    """
    max_delta = float(np.max(np.abs(q) + np.abs(Qs).sum(axis=1))) if q.size else 1.0
    magnitudes = np.concatenate(
        [np.abs(q[q != 0]), np.abs(Qs[Qs != 0]).ravel()]
    )
    min_delta = float(np.min(magnitudes)) if magnitudes.size else 1.0
    max_delta = max(max_delta, 1e-9)
    min_delta = max(min_delta, 1e-9)
    return (np.log(2.0) / max_delta, np.log(100.0) / min_delta)


def anneal_numpy(
    qubo: FleetQubo,
    *,
    sweeps: int = 500,
    restarts: int = 5,
    seed: int = 42,
    beta_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """Seeded simulated annealing over the pair-form QUBO.

    Returns ``(best_bits, best_energy, trace)`` where ``trace`` is the
    best-so-far energy per sweep of the best restart — the fleet analogue
    of the portfolio ``cost_vs_iter`` artifact.
    """
    q = np.asarray(qubo.q, dtype=float)
    Q = np.asarray(qubo.Q, dtype=float)
    n = q.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), qubo.const, []

    # Symmetric neighbour matrix makes the single-flip delta a dot product.
    Qs = Q + Q.T
    rng = np.random.default_rng(seed)
    if beta_range is None:
        beta_range = _auto_beta_range(q, Qs)
    betas = np.geomspace(beta_range[0], beta_range[1], max(sweeps, 1))

    best_bits = np.zeros(n, dtype=int)
    best_energy = float("inf")
    best_trace: List[float] = []

    for _ in range(max(restarts, 1)):
        x = rng.integers(0, 2, size=n).astype(np.int64)
        energy = float(q @ x + x @ (Q @ x) + qubo.const)
        run_best_energy = energy
        run_best_bits = x.copy()
        trace: List[float] = []

        for sweep in range(max(sweeps, 1)):
            beta = betas[sweep]
            for i in rng.permutation(n):
                delta = (1 - 2 * x[i]) * (q[i] + Qs[i] @ x)
                if delta <= 0 or rng.random() < np.exp(-beta * delta):
                    x[i] = 1 - x[i]
                    energy += float(delta)
            if energy < run_best_energy:
                run_best_energy = energy
                run_best_bits = x.copy()
            trace.append(run_best_energy)

        if run_best_energy < best_energy:
            best_energy = run_best_energy
            best_bits = run_best_bits
            best_trace = trace

    return best_bits, float(best_energy), best_trace


def _to_dimod_bqm(qubo: FleetQubo):
    """Build a dimod BQM from the pair-form arrays (import kept local)."""
    import dimod  # noqa: PLC0415 — optional heavy SDK

    linear = {i: float(qubo.q[i]) for i in range(qubo.n_vars)}
    quadratic = {}
    rows, cols = np.nonzero(qubo.Q)
    for i, j in zip(rows.tolist(), cols.tolist()):
        quadratic[(i, j)] = float(qubo.Q[i, j])
    return dimod.BinaryQuadraticModel(
        linear, quadratic, float(qubo.const), dimod.BINARY
    )


def _sample_dwave(
    qubo: FleetQubo,
    *,
    execution_mode: str,
    reads: int,
    seed: int,
) -> SampleOutcome:
    """Ocean-backed sampling. Falls back to NumPy on any SDK problem."""
    started = time.monotonic()
    bqm = _to_dimod_bqm(qubo)

    if execution_mode == "real":
        # Leap hybrid handles this problem class best; the pure QPU path
        # needs minor-embedding which blows up past a few hundred binaries.
        from dwave.system import LeapHybridSampler  # noqa: PLC0415

        from core.optimizers.quantum.shared.circuit_runner import (  # noqa: PLC0415
            resolve_vendor_token,
        )

        token = resolve_vendor_token("dwave", "DWAVE_API_TOKEN")
        if not token:
            raise RuntimeError(
                "No D-Wave Leap API token available. Add one under IAM & Admin "
                "-> Quantum Credentials, or upgrade to Enterprise for the "
                "platform credit pool."
            )
        sampler = LeapHybridSampler(token=token)
        sampleset = sampler.sample(bqm, label="quantumfpo-fleet-routing")
        backend_name = str(getattr(sampler, "solver", "leap-hybrid"))
    else:
        try:
            from dwave.samplers import SimulatedAnnealingSampler  # noqa: PLC0415
        except ImportError:  # pragma: no cover — older Ocean layout
            from neal import SimulatedAnnealingSampler  # type: ignore  # noqa: PLC0415

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=reads, seed=seed)
        backend_name = "dwave-simulated-annealing"

    best = sampleset.first
    bits = np.array(
        [int(best.sample[i]) for i in range(qubo.n_vars)], dtype=int
    )
    energies = sorted(float(rec.energy) for rec in sampleset.data(["energy"]))
    return SampleOutcome(
        bits=bits,
        energy=float(best.energy),
        backend_name=backend_name,
        platform="dwave",
        execution_mode=execution_mode,
        reads=len(sampleset),
        energy_trace=energies[: min(len(energies), 200)],
        elapsed_ms=int((time.monotonic() - started) * 1000),
        extra={"num_occurrences": int(getattr(best, "num_occurrences", 1))},
    )


def _sample_gate(
    qubo: FleetQubo,
    *,
    platform: str,
    execution_mode: str,
    shots: int,
    reps: int,
    seed: int,
    maxiter: int = 60,
) -> SampleOutcome:
    """QAOA on the Ising form, on a simulator or on vendor hardware.

    The variational loop always trains on Aer. Training on a device would
    queue one job per objective evaluation - sixty round trips through a
    shared queue for a solve worth seconds of QPU time - and the parameters
    it converges on are the same ones the noiseless landscape gives. In
    ``real`` mode the *final* optimised circuit is then executed on the
    device, so the returned bitstring is a genuine hardware measurement.
    ``parameter_training`` in the outcome extras records the split so the
    quantum contribution is never overstated.
    """
    hardware = (execution_mode or "").strip().lower() == "real"
    cap = gate_qubit_cap(execution_mode, platform)
    if qubo.n_vars > cap:
        raise ValueError(
            f"gate-based sampling needs <= {cap} qubits in "
            f"{'real' if hardware else 'simulator'} mode, "
            f"problem has {qubo.n_vars}"
        )

    started = time.monotonic()
    from qiskit import QuantumCircuit  # noqa: PLC0415
    from scipy.optimize import minimize  # noqa: PLC0415

    from core.optimizers.quantum.shared.circuit_runner import (  # noqa: PLC0415
        run_measured_circuits,
    )

    h, J, offset = qubo.to_ising()
    n = qubo.n_vars
    pairs = [(i, j, float(J[i, j])) for i in range(n) for j in range(i + 1, n) if J[i, j]]
    trace: List[float] = []

    def build(params: np.ndarray) -> QuantumCircuit:
        gammas = params[:reps]
        betas = params[reps:]
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for layer in range(reps):
            g = float(gammas[layer])
            for i in range(n):
                if h[i]:
                    qc.rz(2.0 * g * float(h[i]), i)
            for i, j, w in pairs:
                qc.cx(i, j)
                qc.rz(2.0 * g * w, j)
                qc.cx(i, j)
            qc.rx(2.0 * float(betas[layer]), range(n))
        qc.measure(range(n), range(n))
        return qc

    def energy_from_counts(counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        energy = 0.0
        for bitstring, freq in counts.items():
            bits = np.array([int(b) for b in reversed(bitstring)], dtype=int)
            energy += qubo.energy(bits) * freq
        return energy / max(total, 1)

    def expectation(params: np.ndarray) -> float:
        run = run_measured_circuits(
            [build(params)], shots=shots, execution_mode="simulator", seed=seed
        )
        value = energy_from_counts(run.counts[0])
        trace.append(value)
        return value

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, np.pi, size=2 * reps)
    result = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})

    final = run_measured_circuits(
        [build(result.x)],
        shots=shots,
        # Forward the vendor in both modes. IonQ's hosted simulator is a real
        # cloud job with a portal deep link, and run_measured_circuits is what
        # decides whether the request earns that path; nulling the platform
        # here took the decision away from it. A run that does stay local is
        # stamped "cpu" by the runner, so no local result is filed under a
        # vendor that never saw it.
        platform=platform,
        execution_mode=execution_mode,
        seed=seed,
        job_label="qaoa final sample",
        min_qubits=n,
    )
    best_bits, best_energy = None, float("inf")
    for bitstring in final.counts[0]:
        bits = np.array([int(b) for b in reversed(bitstring)], dtype=int)
        energy = qubo.energy(bits)
        if energy < best_energy:
            best_bits, best_energy = bits, energy

    if best_bits is None:
        raise ValueError("the final QAOA sample returned no measurement outcomes")
    return SampleOutcome(
        bits=best_bits,
        energy=float(best_energy),
        backend_name=f"{final.backend_name}(qaoa)",
        platform=final.platform,
        execution_mode=final.execution_mode,
        reads=shots,
        energy_trace=trace,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        extra={
            "qubits_used": n,
            "reps": reps,
            "shots": shots,
            "ising_offset": round(float(offset), 6),
            "optimizer_iterations": int(getattr(result, "nfev", len(trace))),
            "parameter_training": "simulator",
            "final_sample_on": final.execution_mode,
            "vendor_runs": final.vendor_runs,
            "physical_qubits": final.physical_qubits,
        },
    )


def sample_qubo(
    qubo: FleetQubo,
    *,
    backend: str = "numpy",
    platform: str = "cpu",
    execution_mode: str = "simulator",
    shots: int = 1024,
    reads: int = 200,
    reps: int = 2,
    seed: int = 42,
    sweeps: int = 500,
    restarts: int = 5,
    maxiter: int = 60,
) -> SampleOutcome:
    """Sample the fleet QUBO on ``backend``, falling back to NumPy SA.

    The fallback exists so a missing SDK cannot fail an otherwise valid job.
    It does NOT apply to ``execution_mode='real'``: a hardware request that
    quietly returns simulated counts is worse than an error, because the
    result is indistinguishable from a device run in Job History.
    """
    fallback_reason: Optional[str] = None
    hardware = (execution_mode or "").strip().lower() == "real"

    if backend == "dwave":
        try:
            return _sample_dwave(
                qubo, execution_mode=execution_mode, reads=reads, seed=seed
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never fail the job
            if hardware:
                raise
            fallback_reason = f"dwave backend unavailable: {exc}"
            logger.warning("[fleet-sampler] %s", fallback_reason)
    elif backend == "gate":
        try:
            return _sample_gate(
                qubo,
                platform=platform,
                execution_mode=execution_mode,
                shots=shots,
                reps=reps,
                seed=seed,
                maxiter=maxiter,
            )
        except Exception as exc:  # noqa: BLE001
            if hardware:
                raise
            fallback_reason = f"gate backend unavailable: {exc}"
            logger.warning("[fleet-sampler] %s", fallback_reason)
    elif backend != "numpy":
        # Silently treating a typo'd backend as "numpy" would make the run
        # diagnostics claim a clean CPU anneal for a device that was never
        # attempted. Record it instead.
        fallback_reason = f"unknown backend '{backend}'; used numpy simulated annealing"
        logger.warning("[fleet-sampler] %s", fallback_reason)

    started = time.monotonic()
    bits, energy, trace = anneal_numpy(
        qubo, sweeps=sweeps, restarts=restarts, seed=seed
    )
    return SampleOutcome(
        bits=bits,
        energy=energy,
        backend_name="numpy-simulated-annealing",
        platform=platform if fallback_reason is None else f"{platform}(fallback)",
        execution_mode=execution_mode,
        reads=restarts * sweeps,
        energy_trace=trace,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        fallback_reason=fallback_reason,
        extra={"sweeps": sweeps, "restarts": restarts},
    )
