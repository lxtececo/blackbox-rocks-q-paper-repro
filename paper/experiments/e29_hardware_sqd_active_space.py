"""E29 - The product's SqDRIFT pipeline on a larger active space: device samples against their own marginals.

Analysis plan fixed before any device job of this script; only --plan, --simulate, --dry-run and
offline tests were run before.

Question. The product's pipeline (qDRIFT circuits from the Hartree-Fock state, samples,
qiskit_addon_sqd with configuration recovery) ran on ibm_marrakesh for N2 / STO-3G (20 qubits,
14,400 determinants) in hardware round 1. Round 2C (E14 part C) put those device samples, the
same shuffled within their marginals, and uniform bits through the product's diagonalization:
mean errors below 0.2 mHa alike, with subspaces that held 0.62 to 0.78 of the space (means of
five runs). E29 moves to an active space in which the largest mean
subspace of the design phase held 0.35 of the space, and asks what the pipeline's energy says
about its samples there: does it return a smaller error from the device's samples than from
samples that keep only each qubit's marginal?

Design phase (21 November 2026, before this plan was written; no device sample existed). A
simulator pilot, stored in results/e29_pilot.json: N2 / cc-pVDZ in CAS(10 electrons, 12
orbitals), 24 qubits, 627,264 determinants, bonds 1.09 and 1.8 Angstrom, 4 groups per bond (a
group = 50 circuits of the product's builder x 100 shots), samples_per_batch 300 (the product's
default), 100 and 30, arms N (noiseless samples), NS (the same shuffled within each qubit's
marginal) and U (uniform bits). Mean |E - E_FCI| in mHa over the 4 groups at the product's
settings, with the mean subspace as a share of the space:
  bond 1.09: N 20.9, NS 19.1, U 0.99 (subspace 1.5%, 0.8%, 28%);
  bond 1.8:  N 65.6, NS 100.8, U 4.4 (subspace 1.4%, 0.3%, 35%).
So uniform bits gave the smallest error, in every group at these settings, through the largest
subspace. The noiseless samples gave a smaller error than their own marginals in 4 of 4 groups
at bond 1.8 at each of the three settings, and in 2 of 4 groups at bond 1.09 at each of them.
These numbers were seen before this plan was written. Groups 0 to 3 of the arms N and NS of this
script are the pilot's groups of bond 1.8 (same circuits, same seeds), and so is their arm U if
the device returns 5,000 shots for the group.

Problem. N2 at 1.8 Angstrom, cc-pVDZ, RHF, CAS(10, 12): build() below, which follows the
product's build_problem and takes the integrals from PySCF mcscf.CASCI. Assertions: RHF and the
FCI solver converged; the Hartree-Fock determinant has, from the active-space integrals, the RHF
energy to 1e-8 Ha; 24 qubits; 627,264 determinants. Every mode that builds the problem stops
unless the RHF and FCI energies are those of the design phase to 1e-8 Ha (-108.4510432403,
-108.8430220346).

Product guard. Every mode stops unless the source of sqdrift.diagonalize contains
"num_bits=2 * norb" (it gives the width of the outcomes to BitArray.from_counts since commit
d65eea48; the pilot found that it did not) and its defaults are samples_per_batch 300,
num_batches 3, max_iterations 5. E29 calls it unchanged.

Groups. K = 10. Group k = sqdrift.build_sqdrift_circuits(problem, circuits_per_time=25,
terms_per_circuit=10, evolution_times=(1.0, 10.0), seed=point_seed(29, 0, 1, k)): 50 circuits of
24 qubits, the product's builder unchanged. 100 shots per circuit.

Simulator samples, --simulate (offline; at any time before the analysis). Per circuit one call
of the product's simulator call, circuit_runner.run_measured_circuits([circuit], shots=100,
platform=None, execution_mode="simulator", seed=builder seed + 2113 x index of the circuit in
its group, job_label="e29", min_qubits=24, simulator="aer"), with AER_MAX_THREADS = 1 (a call
stops otherwise), in a pool of spawned processes (QFPO_WORKERS, else half the cores). One file
per group,
results/raw/e29_sim_group_<k>.json.gz, written when the 50 circuits of the group are done and
never replaced: the combined counts, the shots per circuit, the builder seed, the size of every
circuit as built (two-qubit gates, depth), a signature of the problem (the two energies, h1e,
four projections of h2e), the seconds of every call and the seconds from the start of the pool
to the file. A group that has a file is skipped: an interrupted run is resumed by --simulate,
and the circuits of its unfinished groups are simulated again.

Device. ibm_marrakesh, the device of rounds 1 and 2C, named here before any job. --submit and
--status stop if QFPO_IBM_BACKEND names another device or the service returns another; the
variable need not be set.

Jobs, --submit: three sampler jobs through hj.submit (XY4 dynamical decoupling, gate twirling,
max_execution_time 300 s), ONE per invocation, in this order: kind "samples0" = groups 0 to 3,
"samples1" = groups 4 to 7, "samples2" = groups 8 and 9.
 - Refused, in this order: QFPO_IBM_BACKEND naming another device; three listed jobs (there is
   no fourth); uncommitted code (e20.require_committed_code); a service that returns another
   device; after the states of the listed jobs are read from the service, a manifest whose
   jobs are not these kinds in order, and a listed job that is not DONE (job j needs job j-1
   finished; a job that ended in ERROR or CANCELLED ends the submissions: this plan has no
   further job for its groups); an hj.qpu_seconds estimate above 120 s.
 - Circuits: the 50 circuits of each group, each transpiled by itself with
   generate_preset_pass_manager(optimization_level=3, backend=device, seed_transpiler=0). The
   product's hardware call (circuit_runner._run_ibm) runs every circuit through
   generate_preset_pass_manager(optimization_level=3, backend=backend), which has no seed.
 - Pubs: one per circuit, 100 shots; then two calibration pubs (e14.calibration_circuits: all 0
   and all 1 on the union of the physical qubits that the circuits of the job measure), 2,000
   shots each. Jobs 0 and 1: 202 pubs, 24,000 shots; job 2: 102 pubs, 14,000 shots. The
   calibration is reported, not used.
 - Kept in the manifest with the job: its groups and builder seeds; the shots; the transpiler
   seed; per circuit the two-qubit gates, the depth, the measured physical qubits and the
   product of (1 - reported error) over its two-qubit gates (e14.gate_errors); for the first
   circuit of each group hj.circuit_facts (the device's reported errors, the calibration
   date); the calibrated qubits; the size of every circuit as built; the signature of the
   problem.

Analysis (--collect archives the counts of finished jobs, then analyses; --analyse only
analyses; the same code for a device and a dry run). It runs once the three jobs are archived
and needs the ten simulator files: it stops and names the missing ones. It also stops if the
code is uncommitted; if a job is not the job of this plan (kind, groups, device, number of
pubs) or a simulator file not that of its group (group, builder seed, 50 x 100 shots); if
either is of another problem (signature, 1e-8); or if the circuits of a job and of the
simulation of a group differ in size as built.
 Arms per group k, each one call of the product's sqdrift.diagonalize(problem, counts, seed=k)
 with its defaults:
   H   the device's samples of the 50 circuits of the group, combined;
   HS  H shuffled within each qubit's marginal, as E14 part C: marginals = bits.mean(axis=0),
       (rng.random((shots, 24)) < marginals).astype(np.int8), rng =
       np.random.default_rng(point_seed(29, 4, 1, k)), shots = the number of shots of H;
   U   uniform bits, rng.integers(0, 2, size=(shots, 24)), rng =
       np.random.default_rng(point_seed(29, 2, 1, k)), the same number of shots;
   N   the noiseless samples of the same circuits (results/raw/e29_sim_group_<k>.json.gz);
   NS  N shuffled within marginals in the same way, rng =
       np.random.default_rng(point_seed(29, 1, 1, k)).
 An outcome is a string in which qubit q is character -1-q, as the product reads it
 (sqdrift._occupation); Hartree-Fock is '000000011111000000011111'. hj.counts_of, hj.run_aer and
 the product's simulator call write outcomes in this orientation (checked offline with a
 circuit that flips one qubit, before this plan was fixed).
 Per group and arm: |E - E_FCI| in mHa, the subspace dimension and its share of 627,264, the
 recovery iterations, the share of the input samples with 5 alpha and 5 beta electrons, the
 distinct valid configurations, the share equal to Hartree-Fock, seconds. The 50 calls and the
 selected-CI runs share a pool of spawned processes; hardware_jobs sets the thread counts of a
 process to 1 unless the environment has set them.
 PRIMARY. One-sided exact Wilcoxon signed-rank test over the ten groups, err(H_k) < err(HS_k)
 (scipy.stats.wilcoxon on err(H_k) - err(HS_k), alternative "less", method "exact"), alpha =
 0.01. "The pipeline returns a smaller error from the device's samples than from samples with
 the same marginals" is confirmed only if p <= alpha. If a difference is zero (within 1e-12
 mHa) or two absolute differences are equal, wilcoxon is called with its default method, as
 e20.paired_test calls it, and the result says so; if every difference is zero there is no
 test and nothing is confirmed.
 SECONDARY, unadjusted, alpha = 0.01 each, the same test: (1) one-sided, err(H_k) < err(N_k):
 the device's samples give a smaller error than noiseless samples of the same circuits; (2)
 two-sided, err(H_k) against err(U_k).
 Simulator control, descriptive (groups 0 to 3 were seen in the design phase): in how many of
 the ten groups err(N_k) < err(NS_k), with the one-sided p of the same test.
 Descriptive, no test: mean and range of the error and of the subspace dimension per arm; a
 selected-CI reference at the mean dimension of each arm (PySCF selected CI on the same problem
 at the cutoffs np.geomspace(3e-2, 3e-4, 9), run as e12._sci_cell runs it, interpolated by
 e12.interp_err: log-log, None outside the range of the curve); orbital occupations per group:
 the mean absolute difference between the marginals of H and of N, and between the marginals of
 H and 1/2; per job the measured readout errors (per calibrated qubit P(read 1 | prepared 0) and
 P(read 0 | prepared 1), each pub divided by the shots it returned), the two-qubit gates and the
 depth of its circuits (min, median, max), the shots returned per circuit (min, max), the QPU
 seconds and hj.job_times.

Limits, stated in every result. One molecule, one active space, one geometry, one device, three
jobs. The settings are the product's defaults (a --quick dry run states its own). An error
below another arm's can come from a larger subspace and says nothing by itself about the quality
of the samples; the subspace dimension is reported next to every error.

No run is excluded. Every job is listed in results/e29_submission.json. Raw counts are archived
before any analysis. The device result is never replaced
(results/e29_hardware_sqd_active_space_<tag>.json, <tag> that of the last job).

Usage (on a device in this order: --simulate at any time before the analysis; --submit,
--collect, --submit, --collect, --submit, --collect)
  e29_hardware_sqd_active_space.py --plan
        offline: the product guard, the problem, circuit sizes of group 0 (logical, and six
        circuits on the packaged model of the device), jobs, pubs, shots, the hj.qpu_seconds
        estimates and the QPU seconds measured for jobs of this shape, the pilot table of bond
        1.8, the facts of the primary test, the simulator files, the expected seconds of the
        analysis
  e29_hardware_sqd_active_space.py --simulate
        offline: the noiseless samples of every group that has no file yet
  e29_hardware_sqd_active_space.py --dry-run [--noise Q] [--quick]
        no device. H_k = the noiseless samples of group k in which every sample is replaced,
        with probability Q (default 0.9; 0 leaves them unchanged), by a uniform bitstring
        (np.random.default_rng(point_seed(29, 5, 1, k))). It needs the simulator files of its
        groups and names the missing ones. --quick: groups 0 and 1 only and samples_per_batch
        100 for every arm; the result says so. The result has no job, readout or QPU-seconds
        part. Its name is the stem, then _noise_<Q> if --noise is given (Q as Python prints the
        number, 0.0 for 0), then _quick if --quick is given, then _dryrun; it carries no
        information about hardware.
  e29_hardware_sqd_active_space.py --status
        device, account usage, the listed jobs, the simulator files
  e29_hardware_sqd_active_space.py --submit     the next of the three jobs
  e29_hardware_sqd_active_space.py --collect    archive finished jobs, then analyse
  e29_hardware_sqd_active_space.py --analyse    analyse the archived jobs if they have no result
  (--again, which hardware_jobs.parser defines, is refused.)
"""

from __future__ import annotations

import inspect
import json
import math
import multiprocessing
import os
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import hardware_jobs as hj
import numpy as np
from common import RESULTS, point_seed, pool_workers, provenance
from scipy.stats import wilcoxon

from core.optimizers.chemistry import sqdrift
from core.optimizers.chemistry.problem import MolecularProblem

e14, e20 = hj.e14, hj.e20
e12 = hj._import_script("e12_sqd_selected_ci")           # interp_err

EXPERIMENT = "e29"
STEM = "e29_hardware_sqd_active_space"
DEVICE = "ibm_marrakesh"                         # named before any job: the device of rounds 1 and 2C
ALPHA = 0.01
BASIS, NCAS, NELECAS = "cc-pvdz", 12, 10
BOND, BOND_INDEX = 1.8, 1                        # BOND_INDEX: the pilot's index of this bond in its seeds
NQ = 2 * NCAS
FULL = math.comb(NCAS, NELECAS // 2) ** 2        # 627,264 determinants
HF_STRING = "000000011111000000011111"           # Hartree-Fock as the product writes an outcome: qubit q is character -1-q
PILOT_RHF, PILOT_FCI = -108.4510432403, -108.8430220346  # energies of the design phase, bond 1.8
K = 10                                           # groups
CPT, TERMS, TIMES = 25, 10, (1.0, 10.0)          # the builder's arguments: 2 x 25 = 50 circuits per group
CIRCUITS = len(TIMES) * CPT
SHOTS = 100                                      # per circuit
AER_SEED_SHIFT = 2113                            # simulator seed of circuit i of a group: builder seed + 2113 i
JOB_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
KINDS = [f"samples{j}" for j in range(len(JOB_GROUPS))]
CAL_SHOTS = 2_000
TRANSPILER_SEED = 0
MAX_JOB_SECONDS = e14.MAX_JOB_SECONDS            # 300 s of execution allowed per job
MAX_ESTIMATE_S = 120.0                           # a job with a larger hj.qpu_seconds estimate is not submitted
SIGNATURE_TOL = 1e-8
ARMS = ["H", "HS", "U", "N", "NS"]
DEFAULTS = {"samples_per_batch": 300, "num_batches": 3, "max_iterations": 5}   # of the product's diagonalize
SCI_CUTOFFS = [float(c) for c in np.geomspace(3e-2, 3e-4, 9)]
DRY_NOISE = 0.9                                  # --dry-run: default share of the samples replaced by uniform bitstrings
QUICK = {"groups": [0, 1], "samples_per_batch": 100}
PLAN_MODEL_CIRCUITS = [0, 1, 2, CPT, CPT + 1, CPT + 2]   # --plan: three circuits per evolution time on the device model
# round-1 jobs on ibm_marrakesh: (job, circuits of 100 shots, QPU seconds), read from the service on 21 November 2026
MEASURED_BASIS = [("dbspcf6jjaos73akfn3g", 50, 5), ("dbspg6habdes73frf920", 50, 5), ("dbspjnhabdes73frfcbg", 200, 12)]
LIMITS = ("One molecule, one active space, one geometry, one device, three jobs. {settings} An error below another arm's "
          "can come from a larger subspace and says nothing by itself about the quality of the samples; the subspace "
          "dimension is reported next to every error.")
DESIGN = __doc__.split("\nUsage (")[0].strip()


# --------------------------------------------------------------------------- the problem


def build(bond: float) -> tuple:
    """(problem, facts): N2 / cc-pVDZ in CAS(10 e, 12 o) as the product's MolecularProblem.

    As core/optimizers/chemistry/problem.py build_problem; only the integrals come from mcscf.CASCI.
    facts: what --plan prints and MolecularProblem does not hold.
    """
    from pyscf import ao2mo, fci, gto, mcscf, scf
    from pyscf.fci import cistring

    started = time.monotonic()
    geometry = f"N 0 0 0; N 0 0 {bond}"
    mol = gto.M(atom=geometry, basis=BASIS, symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        # the product's fallback (build_problem): second-order SCF
        mf = scf.RHF(mol).newton()
        mf.conv_tol = 1e-10
        mf.kernel()

    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), NCAS)
    n_alpha, n_beta = NELECAS // 2, NELECAS // 2

    solver = fci.direct_spin1.FCI()
    solver.max_cycle = 200
    solver.conv_tol = 1e-12
    fci_energy, civec = solver.kernel(h1e, h2e, NCAS, (n_alpha, n_beta), ecore=ecore)
    dm_a, dm_b = solver.make_rdm1s(civec, NCAS, (n_alpha, n_beta))

    problem = MolecularProblem(
        molecule_id="n2",
        bond_length=float(bond),
        geometry=geometry,
        norb=int(NCAS),
        nelec=(int(n_alpha), int(n_beta)),
        h1e=np.asarray(h1e),
        h2e=np.asarray(h2e),
        ecore=float(ecore),
        hf_energy=float(mf.e_tot),
        fci_energy=float(fci_energy),
        fci_occupancies=(
            [float(x) for x in np.diag(dm_a)],
            [float(x) for x in np.diag(dm_b)],
        ),
        setup_seconds=time.monotonic() - started,
    )
    assert mf.converged and bool(solver.converged)
    assert problem.n_qubits == NQ and problem.fci_dimension == FULL
    assert abs(hf_determinant_energy(problem) - problem.hf_energy) < 1e-8
    assert int(cistring.make_strings(range(NCAS), n_alpha)[0]) == 0b11111  # civec[0, 0] is the Hartree-Fock determinant
    weights = np.sort((np.asarray(civec) ** 2).ravel())[::-1]
    facts = {"basis_functions": int(mol.nao), "electrons": int(mol.nelectron), "frozen_orbitals": int(mc.ncore),
             "hf_weight_in_fci": float(civec[0, 0] ** 2),
             "determinants_for_99_percent": int(np.searchsorted(np.cumsum(weights), 0.99) + 1)}
    return problem, facts


def n2_active(bond: float) -> MolecularProblem:
    return build(bond)[0]


def hf_determinant_energy(p: MolecularProblem) -> float:
    """Energy of the determinant with the lowest orbitals doubly occupied, from h1e, h2e and ecore."""
    occ = range(p.nelec[0])
    one = 2.0 * sum(p.h1e[i, i] for i in occ)
    two = sum(2.0 * p.h2e[i, i, j, j] - p.h2e[i, j, j, i] for i in occ for j in occ)
    return float(p.ecore + one + two)


def the_problem() -> tuple:
    """build(1.8), stopped unless it has the energies of the design phase."""
    problem, facts = build(BOND)
    if abs(problem.hf_energy - PILOT_RHF) > 1e-8 or abs(problem.fci_energy - PILOT_FCI) > 1e-8:
        raise SystemExit(f"refused: RHF {problem.hf_energy:.10f} and FCI {problem.fci_energy:.10f} are not the energies of the "
                         f"design phase ({PILOT_RHF}, {PILOT_FCI})")
    return problem, facts


def problem_signature(problem: MolecularProblem) -> dict:
    """Numbers that identify the integrals. Files of different phases are used together only if these agree."""
    rng = np.random.default_rng(29)
    return {"hf_energy": problem.hf_energy, "fci_energy": problem.fci_energy, "h1e": [float(x) for x in problem.h1e.ravel()],
            "h2e_projections": [float((problem.h2e * rng.standard_normal(problem.h2e.shape)).sum()) for _ in range(4)]}


def same_problem(a: dict, b: dict) -> bool:
    """Whether two signatures agree to SIGNATURE_TOL in every number."""
    def flat(s: dict) -> np.ndarray:
        return np.array([s["hf_energy"], s["fci_energy"], *s["h1e"], *s["h2e_projections"]])

    return bool(np.abs(flat(a) - flat(b)).max() <= SIGNATURE_TOL)


# --------------------------------------------------------------------------- groups of circuits


def spawn_pool(workers: int) -> ProcessPoolExecutor:
    """Fresh worker processes: a worker forked from a process that has used qiskit crashed in the design phase."""
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def group_seed(k: int) -> int:
    return point_seed(29, 0, BOND_INDEX, k)


def build_group(args) -> list:
    """The 50 circuits of group k: one call of the product's builder, unchanged. args: (problem, k)."""
    problem, k = args
    circuits, _ = sqdrift.build_sqdrift_circuits(problem, circuits_per_time=CPT, terms_per_circuit=TERMS,
                                                 evolution_times=TIMES, seed=group_seed(k))
    if len(circuits) != CIRCUITS or any(c.num_qubits != NQ or c.num_clbits != NQ or len(c.cregs) != 1 for c in circuits):
        raise RuntimeError(f"group {k}: the builder did not return {CIRCUITS} circuits of {NQ} qubits with one register of "
                           f"{NQ} classical bits")
    return circuits


def as_built(circuits: list) -> list:
    """Per circuit [two-qubit gates, depth] as the builder returns it. Two phases used the same circuits only if these agree."""
    return [[e14.two_qubit_gates(c), c.depth()] for c in circuits]


# --------------------------------------------------------------------------- --simulate


def sim_path(k: int):
    return hj.OUT / "raw" / f"{EXPERIMENT}_sim_group_{k}.json.gz"


def sample_circuit(args) -> tuple:
    """(counts, seconds) of one circuit through the product's simulator call. args: (circuit, seed)."""
    from core.optimizers.quantum.shared.circuit_runner import run_measured_circuits

    circuit, seed = args
    if os.environ.get("AER_MAX_THREADS") != "1":
        raise RuntimeError("AER_MAX_THREADS must be 1, as in the design phase")
    started = time.monotonic()
    run = run_measured_circuits([circuit], shots=SHOTS, platform=None, execution_mode="simulator", seed=seed,
                                job_label="e29", min_qubits=NQ, simulator="aer")
    if run.backend_name != "aer_simulator" or run.platform != "cpu":
        raise RuntimeError(f"the product's simulator call ran on {run.backend_name} ({run.platform})")
    return run.counts[0], time.monotonic() - started


def simulate() -> None:
    todo = [k for k in range(K) if not sim_path(k).exists()]
    print(f"E29 --simulate: groups with a file {[k for k in range(K) if k not in todo]}, to simulate {todo}", flush=True)
    if not todo:
        return
    problem, _ = the_problem()
    signature = problem_signature(problem)
    started = time.monotonic()
    with spawn_pool(min(pool_workers(), CIRCUITS * len(todo))) as pool:
        groups = dict(zip(todo, pool.map(build_group, [(problem, k) for k in todo])))
        futures = {pool.submit(sample_circuit, (circuit, group_seed(k) + AER_SEED_SHIFT * i)): (k, i)
                   for k in todo for i, circuit in enumerate(groups[k])}
        print(f"  {len(futures)} circuits x {SHOTS} shots in {min(pool_workers(), CIRCUITS * len(todo))} worker processes, "
              f"built in {time.monotonic() - started:.0f} s", flush=True)
        done = {k: {} for k in todo}
        for future in as_completed(futures):
            k, i = futures[future]
            done[k][i] = future.result()
            if len(done[k]) < CIRCUITS:
                continue
            per_circuit = [done[k][j] for j in range(CIRCUITS)]
            combined = Counter()
            for counts, _ in per_circuit:
                combined.update(counts)
            hj.write_archive(sim_path(k), {
                "experiment": EXPERIMENT, "group": k, "bond": BOND, "builder_seed": group_seed(k),
                "what": "noiseless samples of the circuits of one group: the product's simulator call, one call per circuit",
                "sampler_seed_rule": f"builder_seed + {AER_SEED_SHIFT} * circuit index", "shots_per_circuit": SHOTS,
                "shots": [sum(counts.values()) for counts, _ in per_circuit], "counts": dict(combined),
                "as_built": as_built(groups[k]), "problem": signature,
                "simulation_seconds": [round(seconds, 2) for _, seconds in per_circuit],
                "wall_s": round(time.monotonic() - started, 1), "provenance": provenance()})
            print(f"  group {k}: wrote {sim_path(k).name} after {time.monotonic() - started:.0f} s; "
                  f"{sum(combined.values())} shots, {len(combined)} distinct outcomes", flush=True)
    print(f"E29 --simulate: done in {time.monotonic() - started:.0f} s", flush=True)


def load_sim(groups: list, signature: dict) -> dict:
    """The simulator files of these groups. Stops unless each is of group k, of this problem and of 50 x 100 shots."""
    out = {}
    for k in groups:
        record = hj.read_archive(sim_path(k))
        if (record["group"] != k or record["builder_seed"] != group_seed(k) or record["shots"] != [SHOTS] * CIRCUITS
                or not same_problem(record["problem"], signature)):
            raise SystemExit(f"refused: {sim_path(k).name} is not the simulation of group {k} of this problem")
        out[k] = record
    return out


def print_sim_files() -> None:
    have = [k for k in range(K) if sim_path(k).exists()]
    print(f"Simulator files (results/raw/{EXPERIMENT}_sim_group_<k>.json.gz): groups {have or 'none'} exist, "
          f"{[k for k in range(K) if k not in have] or 'none'} missing")


# --------------------------------------------------------------------------- the product's diagonalize and the arms


def require_product() -> dict:
    """Stops unless the product's diagonalize gives the width of the outcomes and has the defaults E29 states."""
    source = inspect.getsource(sqdrift.diagonalize)
    defaults = {name: p.default for name, p in inspect.signature(sqdrift.diagonalize).parameters.items() if name in DEFAULTS}
    facts = {"width_given": "num_bits=2 * norb" in source, "defaults": defaults}
    if not facts["width_given"] or defaults != DEFAULTS:
        raise SystemExit("refused: the product's sqdrift.diagonalize is not the one E29 calls: " + json.dumps(facts))
    return facts


def bits_of(counts: dict) -> np.ndarray:
    """One row per shot in the order of the counts, column q = qubit q (as e14 part C)."""
    if not counts or any(len(s) != NQ or set(s) - {"0", "1"} for s in counts):
        raise RuntimeError(f"the counts are not outcomes of {NQ} bits")
    return np.array([[int(ch) for ch in s[::-1]] for s in Counter(counts).elements()], dtype=np.int8)


def counts_from(bits) -> dict:
    """Rows of bits written back as outcomes, in the order of their first appearance."""
    return dict(Counter("".join(map(str, row[::-1])) for row in bits))


def arm_inputs(device: dict, noiseless: dict, k: int) -> dict:
    """The counts of the five arms of group k. device, noiseless: {outcome: count} of the 50 circuits of the group."""
    h, n = bits_of(device), bits_of(noiseless)
    rng = {arm: np.random.default_rng(point_seed(29, index, BOND_INDEX, k)) for arm, index in (("NS", 1), ("U", 2), ("HS", 4))}
    return {"H": dict(device),
            "HS": counts_from((rng["HS"].random(h.shape) < h.mean(axis=0)).astype(np.int8)),
            "U": counts_from(rng["U"].integers(0, 2, size=h.shape)),
            "N": dict(noiseless),
            "NS": counts_from((rng["NS"].random(n.shape) < n.mean(axis=0)).astype(np.int8))}


def input_facts(counts: dict) -> dict:
    """Shots, share with 5 alpha and 5 beta electrons, distinct valid configurations, share equal to Hartree-Fock."""
    electrons = NELECAS // 2
    valid = [s for s in counts if s[-NCAS:].count("1") == electrons and s[:-NCAS].count("1") == electrons]
    total = sum(counts.values())
    return {"shots": int(total), "valid_fraction": sum(counts[s] for s in valid) / total, "distinct_valid": len(valid),
            "hf_fraction": counts.get(HF_STRING, 0) / total}


def diagonalize_cell(args) -> dict:
    """One arm of one group through the product's diagonalize. args: (problem, k, arm, counts, samples_per_batch or None)."""
    problem, k, arm, counts, samples_per_batch = args
    settings = {} if samples_per_batch is None else {"samples_per_batch": samples_per_batch}
    started = time.monotonic()
    out = sqdrift.diagonalize(problem, counts, seed=k, **settings)
    return {"group": k, "arm": arm, "energy": out["energy"], "error_mha": abs(1000.0 * (out["energy"] - problem.fci_energy)),
            "dimension": out["subspace_dimension"], "dimension_share": out["subspace_dimension"] / FULL,
            "iterations": len(out["iterations"]), "samples_per_batch": out["samples_per_batch"],
            "num_batches": out["num_batches"], **input_facts(counts), "seconds": time.monotonic() - started}


def sci_cell(args) -> dict:
    """PySCF selected CI at one cutoff, as e12._sci_cell runs it. args: (problem, cutoff)."""
    from pyscf.fci import selected_ci

    p, cut = args
    sci = selected_ci.SCI()
    sci.select_cutoff = float(cut)
    sci.ci_coeff_cutoff = float(cut)
    e, c = sci.kernel(p.h1e, p.h2e, p.norb, p.nelec, ecore=p.ecore)
    return {"cutoff": float(cut), "dim": int(len(c._strs[0]) * len(c._strs[1])), "err": 1000.0 * (float(e) - p.fci_energy)}


# --------------------------------------------------------------------------- analysis (the same for device and dry run)


def signed_rank(first: list, second: list, alternative: str) -> dict:
    """Wilcoxon signed-rank test of first_k - second_k, exact unless a difference is zero or two absolute ones are equal."""
    d = np.asarray(first, float) - np.asarray(second, float)
    out = {"groups": len(d), "alternative": alternative, "alpha": ALPHA, "first_smaller": int((d < -1e-12).sum()),
           "first_larger": int((d > 1e-12).sum()), "zero": int((np.abs(d) <= 1e-12).sum()),
           "median_difference": float(np.median(d)), "range": [float(d.min()), float(d.max())],
           "differences": [float(x) for x in d]}
    if out["zero"] == len(d):
        return {**out, "p": None, "rejects": False, "method": "no test: every difference is zero"}
    tied = bool(out["zero"] or len(np.unique(np.abs(d))) < len(d))
    p = float((wilcoxon(d, alternative=alternative) if tied else wilcoxon(d, alternative=alternative, method="exact")).pvalue)
    method = ("scipy.stats.wilcoxon with its default method, as e20.paired_test calls it: a difference is zero or two "
              "absolute differences are equal" if tied else "exact Wilcoxon signed-rank test (scipy.stats.wilcoxon, method exact)")
    return {**out, "p": p, "rejects": bool(p <= ALPHA), "method": method}


def spread(values) -> dict:
    return {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}


def readout_errors(zero: dict, one: dict, n_bits: int) -> dict:
    """As e14.readout_errors, with each calibration pub divided by its own number of shots."""
    shots = [sum(zero.values()), sum(one.values())]
    p10 = [sum(c for s, c in zero.items() if s[-1 - bit] == "1") / shots[0] for bit in range(n_bits)]
    p01 = [sum(c for s, c in one.items() if s[-1 - bit] == "0") / shots[1] for bit in range(n_bits)]
    return {"shots_returned": shots, "p_read1_given0": p10, "p_read0_given1": p01,
            "mean_assignment_error": float(np.mean(p10 + p01))}


def analyse_arms(problem: MolecularProblem, device: dict, noiseless: dict, samples_per_batch, h_label: str) -> dict:
    """The analysis of the plan. device, noiseless: {group: {outcome: count}}; samples_per_batch None: the product's default."""
    groups = sorted(device)
    inputs = {k: arm_inputs(device[k], noiseless[k], k) for k in groups}
    tasks = [(problem, k, arm, inputs[k][arm], samples_per_batch) for k in groups for arm in ARMS]
    with spawn_pool(min(pool_workers(), len(tasks) + len(SCI_CUTOFFS))) as pool:
        curve = [pool.submit(sci_cell, (problem, cut)) for cut in SCI_CUTOFFS]
        rows = list(pool.map(diagonalize_cell, tasks))
        curve = [future.result() for future in curve]
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    err = {arm: [r["error_mha"] for r in by_arm[arm]] for arm in ARMS}
    primary = signed_rank(err["H"], err["HS"], "less")
    statement = f"the pipeline returns a smaller error from {h_label} than from samples with the same marginals"
    primary.update(hypothesis="err(H_k) < err(HS_k)", verdict=("confirmed: " if primary["rejects"] else "not confirmed: ") + statement)
    below_n = signed_rank(err["H"], err["N"], "less")
    below_n.update(hypothesis="err(H_k) < err(N_k)", note="secondary, unadjusted")
    against_u = signed_rank(err["H"], err["U"], "two-sided")
    against_u.update(hypothesis="err(H_k) differs from err(U_k)", note="secondary, unadjusted")
    control = signed_rank(err["N"], err["NS"], "less")
    control.update(hypothesis="err(N_k) < err(NS_k)", note="descriptive: groups 0 to 3 were seen in the design phase")
    arms = {}
    for arm in ARMS:
        dimension = spread([r["dimension"] for r in by_arm[arm]])
        arms[arm] = {"error_mha": spread(err[arm]), "dimension": dimension, "dimension_share_mean": dimension["mean"] / FULL,
                     "selected_ci_error_mha_at_mean_dimension": e12.interp_err(curve, dimension["mean"]),
                     **{f"{key}_mean": float(np.mean([r[key] for r in by_arm[arm]]))
                        for key in ("iterations", "valid_fraction", "distinct_valid", "hf_fraction", "seconds")}}
    occupations = []
    for k in groups:
        h, n = bits_of(device[k]).mean(axis=0), bits_of(noiseless[k]).mean(axis=0)
        occupations.append({"group": k, "h_marginals": [float(x) for x in h], "n_marginals": [float(x) for x in n],
                            "mean_abs_h_minus_n": float(np.abs(h - n).mean()), "mean_abs_h_minus_half": float(np.abs(h - 0.5).mean())})
    used = rows[0]["samples_per_batch"]
    settings = (f"The settings are the product's defaults (samples_per_batch {used}, num_batches {rows[0]['num_batches']}, "
                f"max_iterations {DEFAULTS['max_iterations']})." if samples_per_batch is None else
                f"samples_per_batch is {used} for every arm (--quick), not the product's default "
                f"{DEFAULTS['samples_per_batch']}; num_batches and max_iterations are the product's defaults.")
    return {"alpha": ALPHA, "groups": groups, "h": h_label, "samples_per_batch": used, "primary": primary,
            "secondary": {"h_below_n": below_n, "h_against_u": against_u}, "simulator_control": control,
            "descriptive": {"note": "descriptive only: no test is made on these", "arms": arms, "selected_ci_curve": curve,
                            "occupations": occupations,
                            "mean_abs_h_minus_n": float(np.mean([o["mean_abs_h_minus_n"] for o in occupations])),
                            "mean_abs_h_minus_half": float(np.mean([o["mean_abs_h_minus_half"] for o in occupations]))},
            "limits": LIMITS.format(settings=settings), "rows": rows}


def report(label: str, result: dict) -> None:
    a, s, c, d = result["primary"], result["secondary"], result["simulator_control"], result["descriptive"]
    n = a["groups"]
    print(label)
    print(f"  PRIMARY err(H) < err(HS), one-sided, {n} groups: p = {a['p']}, alpha {ALPHA}: {a['verdict']}; H smaller in "
          f"{a['first_smaller']}, larger in {a['first_larger']} groups; median difference {a['median_difference']:+.3f} mHa "
          f"({a['method']})")
    print(f"  SECONDARY, unadjusted: err(H) < err(N), one-sided: p = {s['h_below_n']['p']}: "
          f"{'rejects' if s['h_below_n']['rejects'] else 'does not reject'}; H smaller in {s['h_below_n']['first_smaller']} of {n} | "
          f"err(H) against err(U), two-sided: p = {s['h_against_u']['p']}: "
          f"{'rejects' if s['h_against_u']['rejects'] else 'does not reject'}; H smaller in {s['h_against_u']['first_smaller']} of {n}")
    print(f"  simulator control, descriptive: err(N) < err(NS) in {c['first_smaller']} of {n} groups, one-sided p = {c['p']}")
    print("  mean |E - E_FCI| in mHa [min, max] (mean subspace dimension, share of the space; selected CI at that dimension):")
    for arm, x in d["arms"].items():
        sci = x["selected_ci_error_mha_at_mean_dimension"]
        print(f"    {arm:2s} {x['error_mha']['mean']:8.3f} [{x['error_mha']['min']:.3f}, {x['error_mha']['max']:.3f}] "
              f"({x['dimension']['mean']:.0f}, {100 * x['dimension_share_mean']:.2f}%; selected CI "
              f"{'outside the curve' if sci is None else f'{sci:.3f} mHa'}); valid share {x['valid_fraction_mean']:.4f}, "
              f"distinct valid {x['distinct_valid_mean']:.0f}, Hartree-Fock share {x['hf_fraction_mean']:.4f}, "
              f"{x['seconds_mean']:.0f} s per call")
    print(f"  marginals: mean |H - N| {d['mean_abs_h_minus_n']:.4f}, mean |H - 1/2| {d['mean_abs_h_minus_half']:.4f}; "
          f"samples_per_batch {result['samples_per_batch']}", flush=True)


def problem_facts(problem: MolecularProblem, facts: dict) -> dict:
    return {"molecule": "N2", "bond_angstrom": BOND, "basis": BASIS, "active_space": [NELECAS, NCAS], "qubits": NQ,
            "determinants": FULL, "rhf_energy": problem.hf_energy, "fci_energy": problem.fci_energy,
            "correlation_energy_mha": 1000.0 * problem.correlation_energy, **facts}


# --------------------------------------------------------------------------- --plan


def transpile_once(circuit, backend):
    """One pass at optimisation level 3 for the device, with the transpiler seed fixed for the record."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    return generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=TRANSPILER_SEED).run(circuit)


def job_shots(groups: int) -> list:
    """The pubs of a job of this many groups as hj.qpu_seconds reads them: bindings and shots only."""
    return [(None, None, SHOTS)] * (CIRCUITS * groups) + [(None, None, CAL_SHOTS)] * 2


def three(values) -> list:
    """[min, median, max]."""
    return [min(values), statistics.median(values), max(values)]


def slashes(values) -> str:
    low, median, high = three(values)
    return f"{low} / {median:g} / {high}"


def pilot_table(bond: float) -> list:
    """The legend and the lines of one bond of the tables stored in results/e29_pilot.json."""
    lines = json.loads((RESULTS / "e29_pilot.json").read_text())["tables"].splitlines()
    out, keep = [line.split("). ", 1)[1] for line in lines if line.startswith("TASK 4 pilot tables")], False
    for line in lines:
        if line.strip().startswith("bond "):
            keep = line.strip().startswith(f"bond {bond} ")
        elif not line.startswith("      "):
            keep = False
        if keep:
            out.append(line)
    return out


def plan() -> None:
    product = require_product()
    print("E29, the product's SqDRIFT pipeline on N2 / cc-pVDZ in CAS(10 electrons, 12 orbitals)")
    print(f"Product guard passed: sqdrift.diagonalize gives num_bits=2 * norb; defaults {product['defaults']}")
    problem, facts = the_problem()
    print(f"Problem: N2, bond {BOND} Angstrom, {BASIS}: {facts['basis_functions']} basis functions, {facts['electrons']} electrons, "
          f"{facts['frozen_orbitals']} doubly occupied orbitals frozen; {NELECAS} electrons in {NCAS} orbitals, {NQ} qubits, "
          f"{FULL:,} determinants")
    print(f"  RHF {problem.hf_energy:.10f} Ha, FCI {problem.fci_energy:.10f} Ha, correlation energy "
          f"{1000.0 * problem.correlation_energy:.4f} mHa; weight of the Hartree-Fock determinant in FCI "
          f"{facts['hf_weight_in_fci']:.6f}; {facts['determinants_for_99_percent']} determinants hold 99% of the weight")
    circuits = build_group((problem, 0))
    logical = [hj.logical(c) for c in circuits]
    print(f"Group 0 (builder seed {group_seed(0)}), {len(circuits)} circuits in the gate set cx, rz, sx, x (hj.logical), "
          f"min / median / max: two-qubit gates {slashes([e14.two_qubit_gates(c) for c in logical])}; depth "
          f"{slashes([c.depth() for c in logical])}")
    model = next(m for m in hj.device_models() if m.name == "fake_marrakesh")
    isa = [transpile_once(circuits[i], model) for i in PLAN_MODEL_CIRCUITS]
    print(f"  circuits {PLAN_MODEL_CIRCUITS} for {model.name}, one pass at level 3, seed {TRANSPILER_SEED}: two-qubit gates "
          f"{[e14.two_qubit_gates(c) for c in isa]}, depth {[c.depth() for c in isa]} (packaged device model, not the device's "
          "current data)")
    basis = hj.usage_basis()
    estimates = [hj.qpu_seconds(job_shots(len(groups)), basis) for groups in JOB_GROUPS]
    print(f"Device {DEVICE}. Three sampler jobs, one per --submit; per circuit one pub of {SHOTS} shots; per job two "
          f"calibration pubs of {CAL_SHOTS:,} shots; max_execution_time {MAX_JOB_SECONDS} s; not submitted above "
          f"{MAX_ESTIMATE_S:.0f} s estimated")
    for j, (groups, estimate) in enumerate(zip(JOB_GROUPS, estimates)):
        print(f"  job {j} (kind {KINDS[j]}): groups {groups}, {CIRCUITS * len(groups)} circuits, {estimate['pubs']} pubs, "
              f"{estimate['shots']:,} shots, estimated {estimate['seconds']:.1f} QPU seconds")
    print(f"  all three: {sum(e['pubs'] for e in estimates)} pubs, {sum(e['shots'] for e in estimates):,} shots, estimated "
          f"{sum(e['seconds'] for e in estimates):.1f} QPU seconds")
    hj.print_estimate(estimates[0], basis)
    print("  measured for jobs of this shape (the product's round-1 jobs on ibm_marrakesh, read from the service on "
          "21 November 2026): " + "; ".join(f"{job}: {n} circuits x 100 shots, {s} QPU seconds" for job, n, s in MEASURED_BASIS))
    print("Design phase, bond 1.8 (results/e29_pilot.json; seen before this plan was written):")
    for line in pilot_table(BOND):
        print("  " + line.strip())
    print("  Bond 1.8 was chosen because only there the noiseless samples gave a smaller error than their own marginals in "
          "all four groups, at every setting (at bond 1.09 in two of four); the product's defaults because E29 asks about "
          "the pipeline as the product runs it.")
    print(f"Primary test: one-sided exact Wilcoxon signed-rank over {K} groups, err(H) < err(HS), alpha {ALPHA}. p by the "
          "rank sum of the differences with err(H) > err(HS):")
    rejecting = []
    for total in range(8):
        d = [float(-r) for r in range(1, K + 1)]
        if total:
            d[total - 1] = float(total)
        p = float(wilcoxon(d, alternative="less", method="exact").pvalue)
        rejecting += [total] * (p <= ALPHA)
        print(f"    rank sum {total}: p = {p:.5f} {'<=' if p <= ALPHA else '>'} {ALPHA}")
    print(f"  so p <= {ALPHA} needs a rank sum of at most {max(rejecting)}; the smallest p is 1 / {2 ** K} = {1 / 2 ** K:.5f}")
    print_sim_files()
    rows = [r for r in json.loads((RESULTS / "e29_pilot.json").read_text())["rows"]
            if r["bond"] == BOND and r["S"] == DEFAULTS["samples_per_batch"]]
    seconds = {arm: float(np.mean([r["seconds"] for r in rows if r["arm"] == arm])) for arm in ("N", "NS", "U")}
    low = K * (2 * seconds["N"] + 2 * seconds["NS"] + seconds["U"])
    high = K * (seconds["N"] + seconds["NS"] + 3 * seconds["U"])
    workers = min(pool_workers(), K * len(ARMS) + len(SCI_CUTOFFS))
    print(f"Analysis: {K * len(ARMS)} diagonalize calls and {len(SCI_CUTOFFS)} selected-CI runs in {workers} worker "
          f"processes. Seconds per call in the pilot (bond 1.8, the defaults; its cells ran in parallel): N {seconds['N']:.0f}, NS "
          f"{seconds['NS']:.0f}, U {seconds['U']:.0f}. H and HS were not in the pilot: if they cost as N and NS, the calls take "
          f"{low:.0f} s of computation; if as U, {high:.0f} s. Expected seconds of the analysis: the larger of that divided "
          f"by {workers} and one call of U, {max(low / workers, seconds['U']):.0f} to {max(high / workers, seconds['U']):.0f} s, "
          "plus the problem, the worker processes and the selected-CI runs, which the pilot did not time")


# --------------------------------------------------------------------------- --dry-run


def stand_in_counts(noiseless: dict, q: float, k: int) -> dict:
    """The noiseless samples of group k with every sample replaced, with probability q, by a uniform bitstring."""
    bits = bits_of(noiseless)
    rng = np.random.default_rng(point_seed(29, 5, BOND_INDEX, k))
    replaced = rng.random(len(bits)) < q
    uniform = rng.integers(0, 2, size=bits.shape, dtype=np.int8)
    return counts_from(np.where(replaced[:, None], uniform, bits))


def dry_run(args) -> None:
    started = time.time()
    groups = QUICK["groups"] if args.quick else list(range(K))
    missing = [k for k in groups if not sim_path(k).exists()]
    if missing:
        raise SystemExit(f"--dry-run needs the simulator files of groups {groups}; missing: {missing}. Run --simulate")
    q = DRY_NOISE if args.noise is None else args.noise
    problem, facts = the_problem()
    sim = load_sim(groups, problem_signature(problem))
    noiseless = {k: sim[k]["counts"] for k in groups}
    device = {k: stand_in_counts(noiseless[k], q, k) for k in groups}
    execution = f"no device: H is the noiseless samples with a share {q} replaced by uniform bitstrings"
    result = analyse_arms(problem, device, noiseless, QUICK["samples_per_batch"] if args.quick else None,
                          f"the stand-in samples (noiseless samples, a share {q} replaced by uniform bitstrings)")
    report(f"dry run ({execution}; groups {groups})", result)
    name = STEM + ("" if args.noise is None else f"_noise_{args.noise}") + ("_quick" if args.quick else "") + "_dryrun"
    hj.save_result(name, {
        "design": DESIGN, "execution": execution,
        "note": "dry run: it validates the pipeline and the analysis and carries no information about hardware",
        "quick": (f"groups {groups} only; samples_per_batch {QUICK['samples_per_batch']} for every arm instead of the "
                  f"product's default {DEFAULTS['samples_per_batch']}") if args.quick else None,
        "replaced_share": q, "problem": problem_facts(problem, facts), "product": require_product(),
        "simulator_files": [sim_path(k).name for k in groups], **result, "wall_s": round(time.time() - started, 1)})


# --------------------------------------------------------------------------- the device


def refuse_other_device() -> None:
    named = os.environ.get("QFPO_IBM_BACKEND")
    if named and named != DEVICE:
        raise SystemExit(f"refused: QFPO_IBM_BACKEND is {named}; the device of E29 is {DEVICE}")


def open_device() -> tuple:
    """The service and ibm_marrakesh; stops if another device is named or returned."""
    refuse_other_device()
    opened = hj.service()
    try:
        backend = opened.backend(DEVICE)
    except Exception as exc:  # noqa: BLE001 - unknown device, or the service refuses
        raise SystemExit(f"device {DEVICE} could not be opened ({hj._reason(exc)}); nothing was sent or written") from None
    if backend.name != DEVICE:
        raise SystemExit(f"refused: asked for {DEVICE}, the service returned {backend.name}")
    return opened, backend


def refuse_fourth(manifest: dict) -> None:
    if len(manifest["jobs"]) >= len(KINDS):
        raise SystemExit(f"not submitted: the {len(KINDS)} jobs of E29 are listed; there is no further job")


def next_job(manifest: dict) -> int:
    """The index of the job to submit. Stops unless the listed jobs are the first kinds in order and all DONE."""
    refuse_fourth(manifest)
    jobs = manifest["jobs"]
    for j, entry in enumerate(jobs):
        if entry.get("kind") != KINDS[j]:
            raise SystemExit(f"not submitted: the manifest lists job {entry['job_id']} of kind {entry.get('kind')} where "
                             f"{KINDS[j]} is expected")
        if entry["state"] in ("ERROR", "CANCELLED"):
            raise SystemExit(f"not submitted: job {j} ({entry['job_id']}) ended in {entry['state']}; this plan has no "
                             "further job for its groups")
        if entry["state"] != "DONE":
            raise SystemExit(f"not submitted: job {j} ({entry['job_id']}) is {entry['state']}; job {len(jobs)} needs it "
                             "finished")
    return len(jobs)


def job_pubs(problem: MolecularProblem, j: int, backend) -> tuple:
    """(pubs, facts) of job j: per circuit of its groups one pub of 100 shots, then the two calibration pubs."""
    pubs, per_circuit, first, built, measured = [], [], {}, {}, set()
    for k in JOB_GROUPS[j]:
        circuits = build_group((problem, k))
        built[str(k)] = as_built(circuits)
        for i, circuit in enumerate(circuits):
            isa = transpile_once(circuit, backend)
            physical = e14.measured_physical(isa)
            if len(physical) != NQ:
                raise RuntimeError(f"group {k}, circuit {i}: {len(physical)} measured qubits after transpilation")
            measured.update(physical)
            per_circuit.append({"group": k, "circuit": i, "two_qubit_gates": e14.two_qubit_gates(isa), "depth": isa.depth(),
                                "physical_qubits": physical,
                                "p_no_two_qubit_error": e14.gate_errors(backend, isa)["p_no_two_qubit_error"]})
            if i == 0:
                first[str(k)] = hj.circuit_facts(backend, isa)
            pubs.append((isa, None, SHOTS))
    calibrated = sorted(measured)
    pubs += [(circuit, None, CAL_SHOTS) for circuit in e14.calibration_circuits(backend, calibrated)]
    facts = {"kind": KINDS[j], "job": j, "groups": JOB_GROUPS[j], "builder_seeds": [group_seed(k) for k in JOB_GROUPS[j]],
             "shots": SHOTS, "calibration_shots": CAL_SHOTS, "calibration_physical_qubits": calibrated,
             "transpiler_seed": TRANSPILER_SEED, "circuits": per_circuit, "first_circuits": first, "as_built": built,
             "problem": problem_signature(problem)}
    return pubs, facts


def submit() -> None:
    refuse_other_device()
    refuse_fourth(hj.load_manifest(EXPERIMENT))
    e20.require_committed_code()
    opened, backend = open_device()
    manifest = hj.load_manifest(EXPERIMENT)
    hj.refresh(manifest, opened)                         # the states of the listed jobs, from the service
    j = next_job(manifest)
    estimate = hj.qpu_seconds(job_shots(len(JOB_GROUPS[j])))
    if estimate["seconds"] > MAX_ESTIMATE_S:
        raise SystemExit(f"not submitted: {estimate['seconds']:.0f} QPU seconds estimated, above {MAX_ESTIMATE_S:.0f} s")
    problem, _ = the_problem()
    pubs, facts = job_pubs(problem, j, backend)
    hj.submit(EXPERIMENT, __file__, opened, backend, pubs, MAX_JOB_SECONDS, facts)
    print(f"job {j} of E29 (groups {JOB_GROUPS[j]}) is submitted. Next: --collect", flush=True)


def collect() -> None:
    if not hj.load_manifest(EXPERIMENT)["jobs"]:
        raise SystemExit("E29: no job is listed")
    hj.collect(EXPERIMENT, hj.service())
    analyse()


def analyse() -> None:
    entries = hj.load_manifest(EXPERIMENT)["jobs"]
    archived = [entry for entry in entries if entry.get("archive")]
    if len(archived) < len(KINDS):
        print(f"E29: {len(archived)} of {len(KINDS)} jobs are archived; the analysis runs once all are", flush=True)
        print_sim_files()
        return
    name = f"{STEM}_{entries[-1]['tag']}"
    if (hj.OUT / f"{name}.json").exists():
        print(f"E29: the result {name}.json exists and is not replaced")
        return
    missing = [k for k in range(K) if not sim_path(k).exists()]
    if missing:
        raise SystemExit(f"not analysed: the simulator files of groups {missing} are missing. Run --simulate")
    e20.require_committed_code()
    started = time.time()
    problem, facts = the_problem()
    signature = problem_signature(problem)
    sim = load_sim(list(range(K)), signature)
    records = {entry["job_id"]: record for entry, record in hj.to_analyse(EXPERIMENT, STEM)}
    device, jobs = {}, []
    for j, entry in enumerate(entries):
        record, groups = records[entry["job_id"]], JOB_GROUPS[j]
        if (entry.get("kind") != KINDS[j] or entry["groups"] != groups or entry["device"] != DEVICE
                or not same_problem(entry["problem"], signature) or len(record["counts"]) != CIRCUITS * len(groups) + 2):
            raise SystemExit(f"job {entry['job_id']} not analysed: it is not job {j} of this plan and this problem")
        for at, k in enumerate(groups):
            if entry["as_built"][str(k)] != sim[k]["as_built"]:
                raise SystemExit(f"job {entry['job_id']} not analysed: the circuits of group {k} differ from the simulated ones")
            combined = Counter()
            for pub in record["counts"][CIRCUITS * at:CIRCUITS * (at + 1)]:
                combined.update(pub[0])
            device[k] = dict(combined)
        returned = [pub[0] for pub in record["shots"][:CIRCUITS * len(groups)]]
        zero, one = record["counts"][-2][0], record["counts"][-1][0]
        jobs.append({"kind": entry["kind"], "groups": groups, **hj.job_summary(EXPERIMENT, entry, __file__)["job"],
                     "timestamps": entry.get("timestamps"), "max_execution_s": entry["max_execution_s"],
                     "two_qubit_gates_min_median_max": three([c["two_qubit_gates"] for c in entry["circuits"]]),
                     "depth_min_median_max": three([c["depth"] for c in entry["circuits"]]),
                     "shots_returned_per_circuit": [min(returned), max(returned)],
                     "first_circuits": entry["first_circuits"],
                     "readout": {"physical_qubits": entry["calibration_physical_qubits"],
                                 **readout_errors(zero, one, len(entry["calibration_physical_qubits"]))}})
    result = analyse_arms(problem, device, {k: sim[k]["counts"] for k in range(K)}, None, "the device's samples")
    report(f"{DEVICE}, jobs {[entry['job_id'] for entry in entries]}", result)
    for job in jobs:
        print(f"  job {job['job_id']} (groups {job['groups']}): QPU seconds {job['usage_s'] or 'not reported'}, two-qubit gates "
              f"{slashes(job['two_qubit_gates_min_median_max'])}, depth {slashes(job['depth_min_median_max'])} (min / median / "
              f"max), shots per circuit {job['shots_returned_per_circuit']}, mean measured readout error "
              f"{job['readout']['mean_assignment_error']:.4f}")
    hj.save_result(name, {
        "design": DESIGN, "execution": DEVICE, "shots_per_circuit": SHOTS, "calibration_shots": CAL_SHOTS, "jobs": jobs,
        "submissions": hj.job_summary(EXPERIMENT, entries[-1], __file__)["submissions"],
        "qpu_seconds": {"reported_s": float(sum(job["usage_s"] or 0 for job in jobs)),
                        "jobs_without_report": sum(not job["usage_s"] for job in jobs)},
        "problem": problem_facts(problem, facts), "product": require_product(),
        "simulator_files": [sim_path(k).name for k in range(K)], **result, "wall_s": round(time.time() - started, 1)})


def status() -> None:
    opened, backend = open_device()
    hj.status(EXPERIMENT, opened, backend)
    print_sim_files()


def main(argv=None) -> None:
    p = hj.parser(__doc__, {"simulate": "offline: the noiseless samples of every group that has no file yet"})
    for action in p._actions:                            # hj.parser words these for other experiments
        if action.dest == "noise":
            action.metavar, action.help = "Q", "with --dry-run: share of the samples replaced by uniform bitstrings (default 0.9)"
        if action.dest == "again":
            action.help = "refused: E29 has three jobs, one of each kind"
    args = p.parse_args(argv)
    if (args.noise is not None or args.quick) and not args.dry_run:
        p.error("--noise and --quick belong to --dry-run")
    if args.noise is not None and not 0.0 <= args.noise <= 1.0:
        p.error("--noise Q: a share between 0 and 1")
    if args.again:
        p.error("E29 has no --again: three jobs, one of each kind")
    require_product()
    if args.plan:
        plan()
    elif args.simulate:
        simulate()
    elif args.dry_run:
        dry_run(args)
    elif args.status:
        status()
    elif args.submit:
        submit()
    elif args.collect:
        collect()
    else:
        analyse()


if __name__ == "__main__":
    main()
