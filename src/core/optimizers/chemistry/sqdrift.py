"""SqDRIFT: sample-based quantum diagonalization driven by qDRIFT circuits.

Follows the IBM Quantum tutorial "SqDRIFT algorithm for ground state
estimation" (arXiv:2508.02578), Python path:

1. Map the molecular Hamiltonian to fermionic operators (qiskit-fermions),
   order and group symmetry-related terms, drop the diagonal ones.
2. Build an ensemble of short, randomized time-evolution circuits: qDRIFT
   draws ``terms_per_circuit`` Hamiltonian groups with probability
   proportional to their weight, at two evolution times, starting from the
   Hartree-Fock state.
3. Sample bitstrings from every circuit (Aer simulator or IBM hardware).
4. Diagonalize the Hamiltonian classically inside the subspace spanned by the
   sampled electron configurations (qiskit-addon-sqd), with iterative
   configuration recovery repairing shots that broke particle number.

The quantum computer's job is step 3 only: proposing which electron
configurations matter. Step 4 makes the result variational, so the energy can
never fall below the exact ground state.
"""

from __future__ import annotations

import logging
import os
import statistics
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.optimizers.chemistry.problem import MolecularProblem

logger = logging.getLogger(__name__)

#: Ensemble sizes offered to users who should not have to know what a
#: circuit ensemble is. ``circuits_per_time`` is per evolution time.
EFFORTS: Dict[str, Dict[str, Any]] = {
    "quick": {"label": "Quick", "circuits_per_time": 25, "shots": 100},
    "standard": {"label": "Standard", "circuits_per_time": 100, "shots": 100},
    "thorough": {"label": "Thorough (tutorial)", "circuits_per_time": 200, "shots": 100},
}
DEFAULT_EFFORT = "standard"
EVOLUTION_TIMES = (1.0, 10.0)
TERMS_PER_CIRCUIT = 10
#: Total circuits allowed on real hardware. Each one is a separate pub in one
#: Sampler job; 200 x 100 shots is roughly a minute of QPU time on Heron.
MAX_HARDWARE_CIRCUITS = 200


class SqdriftExecutionError(RuntimeError):
    """Circuit construction, sampling or diagonalization failed.

    Fatal on purpose: substituting the classical FCI energy and labelling it
    a quantum result would be a fabricated answer.
    """


def _fermionic_hamiltonian(problem: MolecularProblem):
    from pyscf import tools  # noqa: PLC0415
    from qiskit_fermions.operators import FermionOperator  # noqa: PLC0415
    from qiskit_fermions.operators.library import FCIDump  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="sqdrift-") as tmp:
        path = os.path.join(tmp, "molecule.fcidump")
        tools.fcidump.from_integrals(
            path, problem.h1e, problem.h2e, problem.norb, problem.nelec, nuc=problem.ecore,
        )
        return FermionOperator.from_fcidump(FCIDump.from_file(path))


def build_sqdrift_circuits(
    problem: MolecularProblem,
    *,
    circuits_per_time: int,
    terms_per_circuit: int = TERMS_PER_CIRCUIT,
    evolution_times: Sequence[float] = EVOLUTION_TIMES,
    seed: int = 19,
):
    """Return ``(circuits, info)``. Circuits carry their own measurements."""
    from qiskit_fermions.circuit import FermionicCircuit  # noqa: PLC0415
    from qiskit_fermions.circuit.library import Evolution, InitializeModes  # noqa: PLC0415
    from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms  # noqa: PLC0415
    from qiskit_fermions.operators.terms.grouping import (  # noqa: PLC0415
        group_terms_by_electronic_structure,
    )
    from qiskit_fermions.operators.terms.ordering import canonical_order  # noqa: PLC0415
    from qiskit_fermions.transpiler import FermionicPassManager  # noqa: PLC0415
    from qiskit_fermions.transpiler.passes import QDriftTrotterization  # noqa: PLC0415
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager  # noqa: PLC0415

    num_modes = 2 * problem.norb
    hamiltonian = _fermionic_hamiltonian(problem)
    canon = canonical_order(hamiltonian.normal_ordered().simplify(atol=1e-12))
    group_terms_by_electronic_structure(canon, num_modes, two_body_physicist_order=False)
    filter_diagonal_terms(canon)
    n_groups = len(canon.groups)
    if n_groups == 0:
        raise SqdriftExecutionError("the Hamiltonian has no off-diagonal terms to sample")

    hf_gate = InitializeModes.from_hartree_fock(problem.norb, problem.nelec)
    pm = generate_preset_jw_pass_manager()
    pm.optimization = FermionicPassManager([QDriftTrotterization(terms_per_circuit, rng=seed)])

    circuits = []
    for t in evolution_times:
        base = FermionicCircuit(num_modes)
        base.append(hf_gate, base.modes)
        base.append(Evolution(num_modes, canon, float(t)), base.modes)
        for _ in range(circuits_per_time):
            circuit = pm.run(base)
            circuit.measure_all()
            circuits.append(circuit)

    depths = [c.depth() for c in circuits]
    two_qubit = [
        sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in circuits
    ]
    info = {
        "hamiltonian_groups": n_groups,
        "terms_per_circuit": terms_per_circuit,
        "evolution_times": [float(t) for t in evolution_times],
        "circuits_per_time": circuits_per_time,
        "n_circuits": len(circuits),
        "n_qubits": num_modes,
        "depth_mean": round(statistics.fmean(depths), 1),
        "depth_max": max(depths),
        "two_qubit_gates_mean": round(statistics.fmean(two_qubit), 1),
        "two_qubit_gates_max": max(two_qubit),
    }
    return circuits, info


def _occupation(bitstring: str, norb: int) -> tuple[list[int], list[int]]:
    """Alpha/beta occupation per orbital. Qubit q is character ``-1-q``."""
    alpha = [int(bitstring[-1 - i]) for i in range(norb)]
    beta = [int(bitstring[-1 - (norb + i)]) for i in range(norb)]
    return alpha, beta


def summarize_samples(
    combined: Dict[str, int], problem: MolecularProblem, top: int = 8
) -> Dict[str, Any]:
    """Shot statistics that show what the quantum sampler actually found."""
    norb = problem.norb
    n_alpha, n_beta = problem.nelec
    total = sum(combined.values())
    valid = Counter()
    for bits, count in combined.items():
        alpha, beta = _occupation(bits, norb)
        if sum(alpha) == n_alpha and sum(beta) == n_beta:
            valid[bits] += count
    hf_alpha = [1] * n_alpha + [0] * (norb - n_alpha)
    hf_beta = [1] * n_beta + [0] * (norb - n_beta)
    configs = []
    for bits, count in valid.most_common(top):
        alpha, beta = _occupation(bits, norb)
        excited = sum(1 for i in range(norb) if alpha[i] and not hf_alpha[i]) + sum(
            1 for i in range(norb) if beta[i] and not hf_beta[i]
        )
        configs.append({
            "bitstring": bits,
            "alpha": alpha,
            "beta": beta,
            "count": int(count),
            "share_pct": round(100.0 * count / total, 2) if total else 0.0,
            "is_hartree_fock": alpha == hf_alpha and beta == hf_beta,
            "excitation_level": excited,
        })
    valid_shots = sum(valid.values())
    return {
        "total_shots": int(total),
        "distinct_bitstrings": len(combined),
        "valid_shots": int(valid_shots),
        "valid_fraction": round(valid_shots / total, 4) if total else 0.0,
        "distinct_valid_configurations": len(valid),
        "top_configurations": configs,
    }


def diagonalize(
    problem: MolecularProblem,
    combined: Dict[str, int],
    *,
    samples_per_batch: int = 300,
    num_batches: int = 3,
    max_iterations: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """SQD with configuration recovery over the sampled bitstrings."""
    from qiskit.primitives import BitArray  # noqa: PLC0415
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian  # noqa: PLC0415

    norb = problem.norb
    n_alpha, n_beta = problem.nelec
    history: List[Dict[str, Any]] = []

    def _callback(results) -> None:
        energies = [float(r.energy + problem.ecore) for r in results]
        dims = [int(np.prod(r.sci_state.amplitudes.shape)) for r in results]
        history.append({
            "iteration": len(history) + 1,
            "energies": [round(e, 8) for e in energies],
            "best_energy": round(min(energies), 8),
            "subspace_dimensions": dims,
        })

    result = diagonalize_fermionic_hamiltonian(
        problem.h1e,
        problem.h2e,
        # Without num_bits the width is taken from the largest outcome sampled.
        BitArray.from_counts(combined, num_bits=2 * norb),
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=(n_alpha, n_beta),
        num_batches=num_batches,
        energy_tol=1e-3,
        occupancies_tol=1e-3,
        max_iterations=max_iterations,
        initial_occupancies=(
            np.array([1] * n_alpha + [0] * (norb - n_alpha)),
            np.array([1] * n_beta + [0] * (norb - n_beta)),
        ),
        seed=seed,
        callback=_callback,
    )
    occ_a, occ_b = result.orbital_occupancies
    return {
        "energy": float(result.energy + problem.ecore),
        "subspace_dimension": int(np.prod(result.sci_state.amplitudes.shape)),
        "iterations": history,
        "occupancies": {
            "alpha": [round(float(x), 5) for x in occ_a],
            "beta": [round(float(x), 5) for x in occ_b],
        },
        "samples_per_batch": samples_per_batch,
        "num_batches": num_batches,
    }


def run_sqdrift(
    problem: MolecularProblem,
    *,
    circuits_per_time: int,
    shots: int,
    terms_per_circuit: int = TERMS_PER_CIRCUIT,
    seed: int = 42,
    platform: Optional[str] = None,
    execution_mode: str = "simulator",
) -> Dict[str, Any]:
    """Build, sample and diagonalize. Raises :class:`SqdriftExecutionError`."""
    from core.optimizers.quantum.shared.circuit_runner import (  # noqa: PLC0415
        HardwareUnavailableError,
        run_measured_circuits,
    )

    timings: Dict[str, float] = {}
    try:
        started = time.monotonic()
        circuits, circuit_info = build_sqdrift_circuits(
            problem,
            circuits_per_time=circuits_per_time,
            terms_per_circuit=terms_per_circuit,
            seed=seed,
        )
        timings["build_seconds"] = round(time.monotonic() - started, 3)

        started = time.monotonic()
        run = run_measured_circuits(
            circuits,
            shots=shots,
            platform=platform,
            execution_mode=execution_mode,
            seed=seed,
            job_label="sqdrift ensemble",
            min_qubits=problem.n_qubits,
            simulator="aer",
        )
        timings["sampling_seconds"] = round(time.monotonic() - started, 3)
    except HardwareUnavailableError as exc:
        raise SqdriftExecutionError(str(exc)) from exc
    except SqdriftExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - vendor/library failure modes vary
        logger.warning("[sqdrift] circuit stage failed: %s", exc, exc_info=True)
        raise SqdriftExecutionError(f"SqDRIFT circuits could not be executed: {exc}") from exc

    combined: Counter = Counter()
    for counts in run.counts:
        combined.update(counts)
    sampling = summarize_samples(dict(combined), problem)
    if sampling["total_shots"] == 0:
        raise SqdriftExecutionError("the sampler returned no shots")

    try:
        started = time.monotonic()
        sqd = diagonalize(problem, dict(combined), seed=seed)
        timings["diagonalization_seconds"] = round(time.monotonic() - started, 3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[sqdrift] diagonalization failed: %s", exc, exc_info=True)
        raise SqdriftExecutionError(f"subspace diagonalization failed: {exc}") from exc

    return {
        "algorithm": "SqDRIFT (qDRIFT circuits + sample-based quantum diagonalization)",
        "circuits": circuit_info,
        "shots_per_circuit": shots,
        "sampling": sampling,
        "sqd": sqd,
        "timings": timings,
        **run.provenance(),
    }
