"""E14 - IBM hardware experiments with matched controls. Analysis plan fixed before the runs.

Three parts. A and B use new QPU time; C re-analyses the stored E8 jobs.

A. QAOA, p = 1, 18 qubits, normalised Hamiltonian, exact-optimal parameters
   (gamma*, beta*) from E10. One parameterised circuit is transpiled once and
   bound three ways, so all three run the identical physical gate sequence:
     qaoa        (gamma*, beta*)
     null_beta   (gamma*, 0)      ideal output exactly uniform
     null_gamma  (0, beta*)       ideal output exactly uniform
   Hypothesis A1: P(feasible | qaoa) > P(feasible | null), one-sided Fisher
   exact test against the two nulls pooled, alpha = 0.01.
   Reported whatever the outcome: counts of constraint-satisfying samples
   ("feasible": one-hot, linking, site limit - the constraints the QUBO
   encodes), of the QUBO ground level, of the true optimum and of the lowest
   0.1% of states, with exact (Clopper-Pearson) 95% intervals; the
   approximation ratio r from the sample mean energy with a Student-t
   interval; the noiseless values; and the retained fraction
   (P_qaoa - P_null) / (P_ideal - P_uniform) of the feasible-probability
   excess, with a parametric-bootstrap interval.

B. MLAE depth probe. State preparation A followed by k Grover iterates Q for
   price registers of n = 2 and n = 3 qubits (k = 0..6) and n = 5 (k = 0, 1, 2),
   every qubit measured. For each register size all circuits start from the
   same physical qubits.
   B1: the probability of 1 on the objective qubit follows the depolarising
       contrast model  P_k = b + f_A f_Q^k (P_k^ideal - b)  with b = 1/2;
       (f_A, f_Q) by binomial maximum likelihood, 95% intervals by parametric
       bootstrap, goodness of fit by the deviance. Secondary: b free.
   B2: signal fraction w_k of the full register: maximum-likelihood weight of
       the ideal distribution in the mixture  w p_ideal + (1 - w) uniform.
   B3: the fitted contrast f_A f_Q^k against the resource model
       prod (1 - e_i) over the two-qubit gates of the transpiled circuit, with
       the backend's reported per-gate errors on the day.
   B4 (exploratory): amplitude estimates from the hardware counts - the
       platform's noiseless-likelihood MLE on k = 0, 1, 2, 4, and a noise-aware
       MLE with (theta, f_A, f_Q) free over all k (Tanaka et al. 2021).

C. SqDRIFT on hardware, from the raw counts of the three stored E8 jobs:
     hardware   SQD on the device counts, 5 SQD seeds
     shuffled   every bit drawn independently from the device's own
                single-qubit marginals, same shot count (removes correlations,
                keeps biases), 5 draws
     uniform    uniformly random bits, same shot count, 5 draws
   plus direct signal tests on the counts: the number of Hartree-Fock strings
   and the valid-string fraction against their uniform values, and the mean
   occupation of Hartree-Fock-occupied against Hartree-Fock-empty qubits.

Readout calibration (all-0 and all-1 on the measured physical qubits) is taken
in the same job as A and as B. It is reported, not used to correct counts.
No run is excluded. Execution settings are the platform's hardware path:
optimisation level 3, XY4 dynamical decoupling, gate twirling.

Usage: e14_ibm_hardware_controls.py [qaoa] [mlae] [sqd] [--dry-run] [--noise P]
  --dry-run   noiseless Aer on the logical circuits (validates the analysis)
  --noise P   Aer with two-qubit depolarising error P (validates the fits)
Environment: QFPO_IBM_BACKEND pins the device (default: least busy).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import RESULTS, mean_ci, point_seed, pool_workers, save  # noqa: E402
from e8_ibm_hardware import load_token  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from scipy.stats import binomtest, chi2, fisher_exact, mannwhitneyu  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY = "--dry-run" in sys.argv
NOISE = float(sys.argv[sys.argv.index("--noise") + 1]) if "--noise" in sys.argv else None
if NOISE is not None:
    ARGS = [a for a in ARGS if a != sys.argv[sys.argv.index("--noise") + 1]]
SIM = DRY or NOISE is not None
PARTS = ARGS or (["qaoa", "mlae"] if SIM else ["qaoa", "mlae", "sqd"])

QAOA_SHOTS = 40_000
MLAE_SHOTS = 4_000
CAL_SHOTS = 4_000
MLAE_PLAN = {2: [0, 1, 2, 3, 4, 5, 6], 3: [0, 1, 2, 3, 4, 5, 6], 5: [0, 1, 2]}
TRANSPILER_SEEDS = range(8)
MAX_JOB_SECONDS = 300
BOOT = 2000
OPTION = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
E8_JOBS = [("dbspcf6jjaos73akfn3g", None, 25), ("dbspg6habdes73frf920", 1.8, 25), ("dbspjnhabdes73frfcbg", 1.8, 100)]
SQD_SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- execution


class Runner:
    """Hardware (SamplerV2) or Aer, behind one interface: prepare() then run()."""

    def __init__(self) -> None:
        self.backend = None
        self.service = None
        if not SIM:
            self.service = ibm_service()
            name = os.environ.get("QFPO_IBM_BACKEND")
            self.backend = (self.service.backend(name) if name else
                            self.service.least_busy(operational=True, simulator=False, min_num_qubits=127))
            print("backend:", self.backend.name, flush=True)

    @property
    def name(self) -> str:
        if self.backend is not None:
            return self.backend.name
        return "aer-noiseless" if NOISE is None else f"aer-depolarising-{NOISE}"

    def prepare(self, circuit, initial_layout=None):
        """Best of several transpiler seeds by two-qubit gate count."""
        if SIM:
            from qiskit import transpile

            return transpile(circuit, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        best = None
        for seed in TRANSPILER_SEEDS:
            pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend, seed_transpiler=seed,
                                              initial_layout=initial_layout)
            isa = pm.run(circuit)
            if best is None or two_qubit_gates(isa) < two_qubit_gates(best):
                best = isa
        return best

    def run(self, pubs, label: str):
        """pubs: list of (circuit, list of parameter dicts or None, shots). Returns (counts per pub, meta)."""
        if SIM:
            return self._run_aer(pubs), {"backend": self.name, "job_id": None, "usage_s": 0.0}
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=self.backend)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.max_execution_time = MAX_JOB_SECONDS
        native = []
        for circuit, bindings, shots in pubs:
            if bindings is None:
                native.append((circuit, None, shots))
            else:
                native.append((circuit, [[b[p.name] for p in circuit.parameters] for b in bindings], shots))
        job = sampler.run(native)
        submitted = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"submitted {label}: job {job.job_id()} ({len(pubs)} pubs)", flush=True)
        result = job.result()
        out = []
        for (circuit, bindings, _), pub in zip(pubs, result):
            bits = getattr(pub.data, circuit.cregs[0].name)
            if bindings is None:
                out.append([{str(k): int(v) for k, v in bits.get_counts().items()}])
            else:
                out.append([{str(k): int(v) for k, v in bits.get_counts(i).items()} for i in range(len(bindings))])
        usage = None
        try:
            usage = float(job.usage())
        except Exception as exc:  # noqa: BLE001 - usage is bookkeeping, never fatal
            print("usage unavailable:", exc)
        meta = {"backend": self.name, "job_id": job.job_id(), "usage_s": usage, "submitted_utc": submitted}
        raw = RESULTS / "raw"
        raw.mkdir(exist_ok=True)
        with gzip.open(raw / f"{label.split()[0].lower()}_{job.job_id()}.json.gz", "wt") as fh:
            json.dump({**meta, "label": label, "shots": [p[2] for p in pubs], "counts": out}, fh)  # before any analysis
        return out, meta

    def _run_aer(self, pubs):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise = None
        if NOISE:
            noise = NoiseModel()
            noise.add_all_qubit_quantum_error(depolarizing_error(NOISE, 2), ["cx"])
        out = []
        for index, (circuit, bindings, shots) in enumerate(pubs):
            rows = []
            for j, binding in enumerate(bindings or [None]):
                bound = circuit if binding is None else circuit.assign_parameters(
                    {p: binding[p.name] for p in circuit.parameters})
                sim = AerSimulator(noise_model=noise, seed_simulator=point_seed(14, index, j))
                rows.append({str(k): int(v) for k, v in sim.run(bound, shots=shots).result().get_counts().items()})
            out.append(rows)
        return out


def ibm_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_token()
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_QUANTUM_TOKEN"])


def two_qubit_gates(circuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)


def measured_physical(circuit) -> list:
    """Physical qubit read into each classical bit, in classical-bit order."""
    found = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            found[circuit.find_bit(inst.clbits[0]).index] = circuit.find_bit(inst.qubits[0]).index
    return [found[i] for i in sorted(found)]


def calibration_circuits(backend, physical):
    from qiskit import ClassicalRegister, QuantumCircuit

    pair = []
    for flip in (False, True):
        qc = QuantumCircuit(backend.num_qubits)
        creg = ClassicalRegister(len(physical), "cal")
        qc.add_register(creg)
        if flip:
            for q in physical:
                qc.x(q)
        for i, q in enumerate(physical):
            qc.measure(q, creg[i])
        pair.append(qc)
    return pair


def readout_errors(zero_counts, one_counts, n_bits, shots):
    """Per-bit P(read 1 | prepared 0) and P(read 0 | prepared 1)."""
    p10, p01 = [], []
    for bit in range(n_bits):
        ones_when_zero = sum(c for s, c in zero_counts.items() if s[::-1][bit] == "1")
        zeros_when_one = sum(c for s, c in one_counts.items() if s[::-1][bit] == "0")
        p10.append(ones_when_zero / shots)
        p01.append(zeros_when_one / shots)
    return {"p_read1_given0": p10, "p_read0_given1": p01,
            "mean_assignment_error": float(np.mean(p10 + p01))}


def gate_errors(backend, circuit):
    """Backend-reported two-qubit errors: device median, and the product over this circuit's gates."""
    target = backend.target
    names = [n for n in target.operation_names if getattr(target.operation_from_name(n), "num_qubits", 0) == 2]
    every = [p.error for n in names for p in target[n].values() if p is not None and p.error is not None]
    log_ok = 0.0
    for inst in circuit.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(circuit.find_bit(q).index for q in inst.qubits)
            error = target[inst.operation.name][pair].error
            log_ok += math.log1p(-float(error)) if error is not None else 0.0
    readout = [target["measure"][(q,)].error for q in measured_physical(circuit)]
    return {"median_two_qubit_error": float(np.median(every)), "two_qubit_gate_names": names,
            "p_no_two_qubit_error": math.exp(log_ok),
            "reported_readout_error": [None if e is None else float(e) for e in readout]}


def exact_ci(k: int, n: int):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def counts_vector(counts: dict, n_bits: int) -> np.ndarray:
    v = np.zeros(2 ** n_bits)
    for bits, c in counts.items():
        v[int(bits, 2)] += c
    return v


# --------------------------------------------------------------------------- A: QAOA


def qaoa_part(runner: Runner) -> dict:
    import e10_qaoa_exact as e10
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector

    from core.optimizers.fleet.samplers import build_qaoa_circuit

    ctx = e10._context()
    stored = json.loads((RESULTS / "e10_qaoa_exact.json").read_text())
    gamma_star, beta_star = stored["exact_opt"][0]["params"]
    n, energies, feasible = ctx["n"], ctx["energies"], ctx["feasible"]
    e0, eu = float(energies.min()), float(energies.mean())
    order = np.argsort(energies, kind="stable")
    top = np.zeros(len(energies), bool)
    top[order[: max(1, int(round(e10.TOP_FRACTION * len(energies))))]] = True

    g, b = Parameter("gamma"), Parameter("beta")
    circuit = build_qaoa_circuit(ctx["qubo"], [g, b], 1)
    bindings = {"qaoa": {"gamma": gamma_star, "beta": beta_star},
                "null_beta": {"gamma": gamma_star, "beta": 0.0},
                "null_gamma": {"gamma": 0.0, "beta": beta_star}}
    bare = circuit.remove_final_measurements(inplace=False)
    ideal = {}
    for name, bind in bindings.items():
        ideal[name] = np.abs(Statevector(bare.assign_parameters({g: bind["gamma"], b: bind["beta"]})).data) ** 2
    reference = np.abs(e10.qaoa_state(np.array([gamma_star, beta_star]), 1, True)) ** 2
    assert np.abs(ideal["qaoa"] - reference).max() < 1e-10, "parameterised circuit differs from the E10 state"
    for name in ("null_beta", "null_gamma"):
        assert np.abs(ideal[name] - 2.0 ** -n).max() < 1e-12, f"{name} is not uniform"

    isa = runner.prepare(circuit)
    physical = measured_physical(isa) if not SIM else list(range(n))
    pubs = [(isa, list(bindings.values()), QAOA_SHOTS)]
    if not SIM:
        pubs += [(c, None, CAL_SHOTS) for c in calibration_circuits(runner.backend, physical)]
    counts, meta = runner.run(pubs, "E14-A qaoa")

    def stats(count_dict, prob):
        v = counts_vector(count_dict, n)
        shots = int(v.sum())
        mean_e = float(v @ energies / shots)
        sd_e = math.sqrt(float(v @ (energies - mean_e) ** 2 / (shots - 1)))
        half = float(student_t.ppf(0.975, shots - 1)) * sd_e / math.sqrt(shots) / (eu - e0)
        row = {"shots": shots, "approx_ratio": (eu - mean_e) / (eu - e0), "approx_ratio_ci95": half,
               "ideal_approx_ratio": float((eu - prob @ energies) / (eu - e0))}
        for key, mask in (("feasible", feasible), ("ground", ctx["ground"]), ("optimum", ctx["optimum"]), ("top", top)):
            k = int(v[mask].sum())
            row[key] = {"count": k, "rate": k / shots, "ci95": exact_ci(k, shots), "ideal_rate": float(prob[mask].sum())}
        return row

    arms = {name: stats(counts[0][i], ideal[name]) for i, name in enumerate(bindings)}
    null_k = arms["null_beta"]["feasible"]["count"] + arms["null_gamma"]["feasible"]["count"]
    null_n = arms["null_beta"]["shots"] + arms["null_gamma"]["shots"]
    q_k, q_n = arms["qaoa"]["feasible"]["count"], arms["qaoa"]["shots"]
    p_uniform = float(feasible.mean())
    excess_ideal = arms["qaoa"]["feasible"]["ideal_rate"] - p_uniform
    rng = np.random.default_rng(point_seed(14, 1))
    boot = (rng.binomial(q_n, q_k / q_n, BOOT) / q_n - rng.binomial(null_n, null_k / null_n, BOOT) / null_n) / excess_ideal
    tests = {
        "feasible_qaoa_vs_pooled_null_fisher_p": float(fisher_exact([[q_k, q_n - q_k], [null_k, null_n - null_k]],
                                                                    alternative="greater").pvalue),
        "pooled_null_feasible": {"count": null_k, "shots": null_n, "rate": null_k / null_n, "ci95": exact_ci(null_k, null_n),
                                 "p_vs_uniform": float(binomtest(null_k, null_n, p_uniform).pvalue)},
        "nulls_differ_fisher_p": float(fisher_exact(
            [[arms["null_beta"]["feasible"]["count"], arms["null_beta"]["shots"] - arms["null_beta"]["feasible"]["count"]],
             [arms["null_gamma"]["feasible"]["count"], arms["null_gamma"]["shots"] - arms["null_gamma"]["feasible"]["count"]]]).pvalue),
        "retained_fraction_of_feasible_excess": {"estimate": (q_k / q_n - null_k / null_n) / excess_ideal,
                                                 "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]},
        "uniform_feasible_rate": p_uniform,
    }
    out = {"parameters": {"gamma": gamma_star, "beta": beta_star}, "qubits": n, "two_qubit_gates": two_qubit_gates(isa),
           "depth": isa.depth(), "physical_qubits": physical, "arms": arms, "tests": tests, "run": meta}
    if not SIM:
        out["calibration"] = {**gate_errors(runner.backend, isa),
                              "measured": readout_errors(counts[1][0], counts[2][0], n, CAL_SHOTS)}
    a = arms["qaoa"]
    print(f"QAOA: 2q gates {out['two_qubit_gates']}; feasible {q_k}/{q_n} (ideal {a['feasible']['ideal_rate']:.4f}) vs "
          f"null {null_k}/{null_n} (uniform {p_uniform:.5f}); Fisher p={tests['feasible_qaoa_vs_pooled_null_fisher_p']:.2e}; "
          f"r={a['approx_ratio']:.3f}±{a['approx_ratio_ci95']:.3f} (ideal {a['ideal_approx_ratio']:.3f}); "
          f"retained {tests['retained_fraction_of_feasible_excess']['estimate']:.3f}")
    return out


# --------------------------------------------------------------------------- B: MLAE depth probe


def objective_mask(n: int) -> np.ndarray:
    """Outcomes (as integers over n + 1 bits) whose objective qubit, bit n, reads 1."""
    return ((np.arange(2 ** (n + 1)) >> n) & 1) == 1


def binomial_nll(p, hits, shots):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -float(np.sum(hits * np.log(p) + (shots - hits) * np.log(1 - p)))


def fit_contrast(powers, hits, shots, ideal, free_floor=False):
    powers, hits, ideal = np.asarray(powers, float), np.asarray(hits, float), np.asarray(ideal, float)

    def model(x):
        floor = x[2] if free_floor else 0.5
        return floor + x[0] * x[1] ** powers * (ideal - floor)

    bounds = [(0.0, 1.0), (0.0, 1.0)] + ([(0.0, 1.0)] if free_floor else [])
    best = None
    for start in ((0.9, 0.9), (0.5, 0.7), (0.3, 0.5), (0.8, 0.3)):
        x0 = list(start) + ([0.5] if free_floor else [])
        res = minimize(lambda x: binomial_nll(model(x), hits, shots), x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, model(best.x), float(best.fun)


def deviance(hits, shots, p):
    hits, p = np.asarray(hits, float), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(hits > 0, hits * np.log(hits / (shots * p)), 0.0)
        b = np.where(hits < shots, (shots - hits) * np.log((shots - hits) / (shots * (1 - p))), 0.0)
    return 2.0 * float(np.sum(a + b))


def mixture_weight(counts_vec, ideal):
    """ML weight of the ideal distribution in  w ideal + (1 - w) uniform."""
    d = len(ideal)

    def nll(w):
        return -float(counts_vec @ np.log(np.clip(w * ideal + (1 - w) / d, 1e-300, None)))

    return float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}).x)


def noise_aware_amplitude(powers, hits, shots):
    """Joint ML over (theta, f_A, f_Q): grid search, then a local refinement. Returns (amplitude, f_A, f_Q)."""
    powers, hits = np.asarray(powers, float), np.asarray(hits, float)
    thetas = np.linspace(1e-3, math.pi / 2 - 1e-3, 400)
    fs = np.linspace(0.0, 1.0, 41)
    ideal = np.sin((2 * powers[None, :] + 1) * thetas[:, None]) ** 2
    contrast = fs[:, None, None] * fs[None, :, None] ** powers[None, None, :]
    p = np.clip(0.5 + contrast[None] * (ideal[:, None, None, :] - 0.5), 1e-12, 1 - 1e-12)
    ll = (hits * np.log(p) + (shots - hits) * np.log(1 - p)).sum(axis=-1)
    t, a, q = np.unravel_index(int(np.argmax(ll)), ll.shape)

    def nll(x):
        model = 0.5 + x[1] * x[2] ** powers * (np.sin((2 * powers + 1) * x[0]) ** 2 - 0.5)
        return binomial_nll(model, hits, shots)

    start = np.array([thetas[t], fs[a], fs[q]])
    res = minimize(nll, start, method="L-BFGS-B", bounds=[(1e-6, math.pi / 2 - 1e-6), (0.0, 1.0), (0.0, 1.0)])
    x = res.x if res.fun <= nll(start) else start
    return math.sin(x[0]) ** 2, float(x[1]), float(x[2])


def mlae_part(runner: Runner) -> dict:
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit.quantum_info import Statevector

    from core.optimizers.montecarlo.grid import build_price_grid
    from core.optimizers.montecarlo.qae import _maximize_likelihood, build_grover_operator, build_state_preparation

    registers, pubs, index = {}, [], []
    for n, powers in MLAE_PLAN.items():
        grid = build_price_grid(OPTION["S0"], OPTION["K"], OPTION["r"], OPTION["sigma"], OPTION["T"], "call", n)
        amplitude = float(grid.exact_discretized_price / grid.payoff_ceiling)
        theta = math.asin(math.sqrt(amplitude))
        prep = build_state_preparation(grid)
        grover = build_grover_operator(prep, n)
        circuits, ideals = [], []
        for k in powers:
            qc = QuantumCircuit(n + 1)
            creg = ClassicalRegister(n + 1, "m")
            qc.add_register(creg)
            qc.compose(prep, inplace=True)
            for _ in range(k):
                qc.compose(grover, inplace=True)
            prob = np.abs(Statevector(qc).data) ** 2
            p1 = float(prob[objective_mask(n)].sum())
            assert abs(p1 - math.sin((2 * k + 1) * theta) ** 2) < 1e-9, (n, k, p1)
            qc.measure(range(n + 1), creg)
            circuits.append(qc)
            ideals.append(prob)
        layout = None
        if not SIM:  # every power on the physical qubits chosen for k = 1
            first = runner.prepare(circuits[min(1, len(circuits) - 1)])
            layout = first.layout.initial_index_layout(filter_ancillas=True)
        isa = [runner.prepare(c, initial_layout=layout) for c in circuits]
        registers[n] = {"amplitude": amplitude, "theta": theta, "powers": powers, "ideals": ideals, "isa": isa,
                        "layout": layout, "price_ceiling": float(grid.payoff_ceiling)}
        for k, circuit in zip(powers, isa):
            pubs.append((circuit, None, MLAE_SHOTS))
            index.append((n, k))
        if not SIM:
            physical = sorted(set(q for c in isa for q in measured_physical(c)))
            registers[n]["calibrated_qubits"] = physical
            for c in calibration_circuits(runner.backend, physical):
                pubs.append((c, None, CAL_SHOTS))
                index.append((n, "cal"))
    counts, meta = runner.run(pubs, "E14-B mlae depth probe")

    out = {"shots": MLAE_SHOTS, "run": meta, "registers": []}
    rng = np.random.default_rng(point_seed(14, 2))
    for n, reg in registers.items():
        powers = reg["powers"]
        rows, hits = [], []
        for k, ideal, isa in zip(powers, reg["ideals"], reg["isa"]):
            c = counts[index.index((n, k))][0]
            v = counts_vector(c, n + 1)
            one = int(v[objective_mask(n)].sum())
            hits.append(one)
            p_ideal = float(ideal[objective_mask(n)].sum())
            boots = [mixture_weight(rng.multinomial(MLAE_SHOTS, v / v.sum()).astype(float), ideal) for _ in range(300)]
            row = {"k": k, "two_qubit_gates": two_qubit_gates(isa), "depth": isa.depth(),
                   "objective_hits": one, "p_hat": one / MLAE_SHOTS, "p_hat_ci95": exact_ci(one, MLAE_SHOTS),
                   "p_ideal": p_ideal, "p_vs_ideal": float(binomtest(one, MLAE_SHOTS, p_ideal).pvalue),
                   "p_vs_half": float(binomtest(one, MLAE_SHOTS, 0.5).pvalue),
                   "signal_fraction": mixture_weight(v, ideal),
                   "signal_fraction_ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                   "tv_to_ideal": 0.5 * float(np.abs(v / v.sum() - ideal).sum()),
                   "tv_to_uniform": 0.5 * float(np.abs(v / v.sum() - 1.0 / len(v)).sum()),
                   "ideal_tv_to_uniform": 0.5 * float(np.abs(ideal - 1.0 / len(v)).sum())}
            if not SIM:
                row["calibration"] = gate_errors(runner.backend, isa)
                row["measured_physical"] = measured_physical(isa)
            rows.append(row)
        ideal_p = [r["p_ideal"] for r in rows]
        entry = {"n_price_qubits": n, "qubits": n + 1, "true_amplitude": reg["amplitude"], "rows": rows,
                 "initial_layout": reg["layout"]}
        if len(powers) >= 4:
            x, fitted, nll = fit_contrast(powers, hits, MLAE_SHOTS, ideal_p)
            xb = np.array([fit_contrast(powers, rng.binomial(MLAE_SHOTS, fitted), MLAE_SHOTS, ideal_p)[0] for _ in range(BOOT // 4)])
            dev = deviance(hits, MLAE_SHOTS, fitted)
            x3, fitted3, nll3 = fit_contrast(powers, hits, MLAE_SHOTS, ideal_p, free_floor=True)
            entry["contrast_fit"] = {
                "f_A": float(x[0]), "f_Q": float(x[1]),
                "f_A_ci95": [float(v) for v in np.percentile(xb[:, 0], [2.5, 97.5])],
                "f_Q_ci95": [float(v) for v in np.percentile(xb[:, 1], [2.5, 97.5])],
                "fitted_p": [float(v) for v in fitted], "deviance": dev, "dof": len(powers) - 2,
                "goodness_of_fit_p": float(chi2.sf(dev, len(powers) - 2)),
                "free_floor": {"f_A": float(x3[0]), "f_Q": float(x3[1]), "floor": float(x3[2]),
                               "deviance": deviance(hits, MLAE_SHOTS, fitted3),
                               "likelihood_ratio_p": float(chi2.sf(2 * (nll - nll3), 1))},
            }
            schedule = [k for k in (0, 1, 2, 4) if k in powers]
            sched_hits = [hits[powers.index(k)] for k in schedule]
            naive = math.sin(_maximize_likelihood(schedule, MLAE_SHOTS, sched_hits)) ** 2
            aware, fa, fq = noise_aware_amplitude(powers, hits, MLAE_SHOTS)
            aware_boot = [noise_aware_amplitude(powers, rng.binomial(MLAE_SHOTS, fitted), MLAE_SHOTS)[0] for _ in range(60)]
            entry["amplitude_estimates"] = {
                "true": reg["amplitude"], "noiseless_likelihood_mle": naive, "noiseless_schedule": schedule,
                "noise_aware_mle": aware, "noise_aware_f_A": fa, "noise_aware_f_Q": fq,
                "noise_aware_bootstrap_sd": float(np.std(aware_boot, ddof=1)),
                "price_ceiling": reg["price_ceiling"],
            }
            print(f"n={n}: f_A={x[0]:.3f} f_Q={x[1]:.3f} fit p={entry['contrast_fit']['goodness_of_fit_p']:.3f}; "
                  f"amplitude true {reg['amplitude']:.4f} naive {naive:.4f} noise-aware {aware:.4f}")
        if not SIM:
            cal = [i for i, key in enumerate(index) if key == (n, "cal")]
            entry["readout"] = {"physical_qubits": reg["calibrated_qubits"],
                                **readout_errors(counts[cal[0]][0], counts[cal[1]][0], len(reg["calibrated_qubits"]), CAL_SHOTS)}
        for r in rows:
            print(f"  n={n} k={r['k']}: 2q={r['two_qubit_gates']:4d} P(1)={r['p_hat']:.3f} ideal {r['p_ideal']:.3f} "
                  f"signal fraction {r['signal_fraction']:.3f} [{r['signal_fraction_ci95'][0]:.3f}, {r['signal_fraction_ci95'][1]:.3f}]")
        out["registers"].append(entry)
    return out


# --------------------------------------------------------------------------- C: SqDRIFT raw counts


def _sqd_cell(args):
    bond, counts, seed = args
    from core.optimizers.chemistry.problem import build_problem, score_energy
    from core.optimizers.chemistry.sqdrift import diagonalize

    problem = build_problem("n2", bond)
    d = diagonalize(problem, counts, seed=seed)
    return score_energy(d["energy"], problem)["error_mha"], d["subspace_dimension"] / 14400


def sqd_part() -> dict:
    from core.optimizers.chemistry.problem import build_problem
    from core.optimizers.chemistry.sqdrift import _occupation

    service = ibm_service()
    stored = json.loads((RESULTS / "e8_ibm_hardware_sqdrift.json").read_text())["sqdrift"]
    rows, jobs = [], []
    for (job_id, bond, cpt), ref in zip(E8_JOBS, stored):
        assert (ref["bond_length"], ref["circuits_per_time"]) == (bond, cpt)
        problem = build_problem("n2", bond)
        norb, (na, nb) = problem.norb, problem.nelec
        nq = 2 * norb
        result = service.job(job_id).result()
        combined = Counter()
        for pub in result:
            field = next(iter(pub.data.keys()))
            combined.update({str(k): int(v) for k, v in getattr(pub.data, field).get_counts().items()})
        shots = sum(combined.values())
        assert shots == ref["total_shots"], (shots, ref["total_shots"])
        bits = np.array([[int(ch) for ch in s[::-1]] for s in combined.elements()], dtype=np.int8)  # column q = qubit q
        valid = (bits[:, :norb].sum(axis=1) == na) & (bits[:, norb:].sum(axis=1) == nb)
        assert abs(valid.mean() - ref["valid_fraction"]) < 5e-4, (valid.mean(), ref["valid_fraction"])
        hf = np.array([1] * na + [0] * (norb - na) + [1] * nb + [0] * (norb - nb), dtype=np.int8)
        hf_count = int((bits == hf).all(axis=1).sum())
        p_valid = math.comb(norb, na) * math.comb(norb, nb) / 2 ** nq
        marginals = bits.mean(axis=0)
        occ, emp = marginals[hf == 1], marginals[hf == 0]
        excitation = (bits[valid] * (1 - hf)).sum(axis=1)
        a0, b0 = _occupation(next(iter(combined)), norb)  # convention check: same split as the platform
        assert list(a0) + list(b0) == [int(ch) for ch in next(iter(combined))[::-1]]
        row = {
            "job_id": job_id, "bond_length": bond, "circuits_per_time": cpt, "shots": shots,
            "valid": {"count": int(valid.sum()), "rate": float(valid.mean()), "ci95": exact_ci(int(valid.sum()), shots),
                      "uniform_rate": p_valid, "p_vs_uniform": float(binomtest(int(valid.sum()), shots, p_valid).pvalue)},
            "hartree_fock": {"count": hf_count, "expected_uniform": shots / 2 ** nq,
                             "p_vs_uniform": float(binomtest(hf_count, shots, 2.0 ** -nq, alternative="greater").pvalue)},
            "marginals": {"p1_per_qubit": [float(x) for x in marginals],
                          "mean_hf_occupied": float(occ.mean()), "mean_hf_empty": float(emp.mean()),
                          "occupied_minus_empty_mannwhitney_p": float(mannwhitneyu(occ, emp, alternative="greater").pvalue)},
            "excitation_level_of_valid": {"mean": float(excitation.mean()) if len(excitation) else None,
                                          "uniform_mean": (na * (norb - na) + nb * (norb - nb)) / norb},
            "stored_hardware_error_mha": ref["error_mha"],
        }
        rows.append(row)
        base = len(jobs)
        counts = dict(combined)
        for s in SQD_SEEDS:
            jobs.append(("hardware", len(rows) - 1, (bond, counts, s)))
        for s in SQD_SEEDS:  # controls: draw s, diagonalised with SQD seed s
            rng = np.random.default_rng(point_seed(14, 3, len(rows), s))
            shuffled = (rng.random((shots, nq)) < marginals).astype(np.int8)
            jobs.append(("shuffled", len(rows) - 1, (bond, dict(Counter("".join(map(str, r[::-1])) for r in shuffled)), s)))
            uniform = rng.integers(0, 2, size=(shots, nq), dtype=np.int8)
            jobs.append(("uniform", len(rows) - 1, (bond, dict(Counter("".join(map(str, r[::-1])) for r in uniform)), s)))
        print(f"N2 {bond} cpt={cpt}: {shots} shots, valid {row['valid']['count']} (uniform {p_valid * shots:.1f}), "
              f"HF strings {hf_count}, P(1) occupied {occ.mean():.3f} vs empty {emp.mean():.3f}; {len(jobs) - base} SQD runs queued",
              flush=True)

    with ProcessPoolExecutor(max_workers=pool_workers()) as ex:
        results = list(ex.map(_sqd_cell, [j[2] for j in jobs]))
    for row_index, row in enumerate(rows):
        for arm in ("hardware", "shuffled", "uniform"):
            sel = [r for (a, i, _), r in zip(jobs, results) if a == arm and i == row_index]
            row[arm] = {"abs_error_mha": mean_ci([abs(e) for e, _ in sel]), "errors_mha": [e for e, _ in sel],
                        "coverage": mean_ci([c for _, c in sel])}
        hw = [abs(e) for e in row["hardware"]["errors_mha"]]
        for arm in ("shuffled", "uniform"):
            other = [abs(e) for e in row[arm]["errors_mha"]]
            row[f"hardware_vs_{arm}_mannwhitney_p"] = float(mannwhitneyu(hw, other, alternative="two-sided").pvalue)
        print(f"N2 {row['bond_length']} cpt={row['circuits_per_time']}: hardware {row['hardware']['abs_error_mha']['mean']:.4f} "
              f"shuffled {row['shuffled']['abs_error_mha']['mean']:.4f} uniform {row['uniform']['abs_error_mha']['mean']:.4f} mHa "
              f"(stored single run {row['stored_hardware_error_mha']:.4f})")
    return {"rows": rows, "sqd_seeds": SQD_SEEDS}


def main() -> None:
    t0 = time.time()
    out = {"mode": "dry-run" if DRY else (f"noise-{NOISE}" if NOISE is not None else "hardware")}
    runner = Runner() if ("qaoa" in PARTS or "mlae" in PARTS) else None
    if runner is not None:
        out["backend"] = runner.name
    if "qaoa" in PARTS:
        out["qaoa"] = qaoa_part(runner)
    if "mlae" in PARTS:
        out["mlae"] = mlae_part(runner)
    if "sqd" in PARTS:
        out["sqdrift_raw_counts"] = sqd_part()
    out["wall_s"] = round(time.time() - t0, 1)
    suffix = "" if not SIM else ("_dryrun" if DRY else "_noisesim")
    part = "" if len(PARTS) == 3 else "_" + "_".join(PARTS)
    save("e14_ibm_hardware_controls" + suffix + part, out)


if __name__ == "__main__":
    main()
