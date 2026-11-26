"""E26 - Replication of hardware round 2A (QAOA against depth-matched nulls) on a second device and day.

Analysis plan fixed before any run of this script.

Question. Round 2A (E14 part A; ibm_marrakesh; job submitted 17 November 2026 UTC) found 244
constraint-satisfying samples in the 40 000 shots of the exact-optimal binding against 70 in the
80 000 shots of the two nulls. Does the test of Round 2A reject again on another device, on another
day?

What is run. Nothing of Round 2A is changed.
 - Instance, circuit and bindings are e14.qaoa_problem(): the 18-qubit instance, QAOA at depth 1 on
   the normalised Hamiltonian. One parameterised circuit is transpiled once and bound three ways, so
   the three executions use the same physical gates on the same qubits:
     qaoa        (gamma*, beta*)   the exact-optimal angles of E10
     null_beta   (gamma*, 0)       ideal output exactly uniform
     null_gamma  (0, beta*)        ideal output exactly uniform
 - 40 000 shots per binding.
 - Transpilation: the procedure of E14's hardware branch. Optimisation level 3 for the device, with
   the device's error data, once per transpiler seed 0 to 7; the circuit with the fewest two-qubit
   gates is kept, the lowest seed among equals.
 - Readout calibration in the same job: all-0 and all-1 on the 18 measured physical qubits, 4 000
   shots each. It is reported, not used to correct counts.
 - Sampler settings of E14: XY4 dynamical decoupling, gate twirling, at most 300 s of execution.
 - One job of three pubs, 128 000 shots.

Hypothesis and test, exactly as in Round 2A. P(constraint-satisfying | qaoa) is greater than
P(constraint-satisfying | null): one-sided Fisher exact test of the constraint-satisfying samples
of the qaoa binding against those of the two nulls pooled, alpha = 0.01. "Constraint-satisfying" is
E14's "feasible": one-hot, linking and site limit, the constraints the QUBO encodes.

Replication criterion. Round 2A is replicated if this test rejects (p <= 0.01) on the new device.
Nothing else enters the criterion.

Reported descriptively only, whatever the outcome
 - the comparison with the Round 2A figures on ibm_marrakesh (244 of 40 000 against 70 of 80 000):
   both pairs of counts, the ratio of the qaoa rate to the null rate in each run, and the ratio of
   this run's rates to those of Round 2A, with 95% intervals (Katz); no test is made on them;
 - what E14 part A reports for each binding: constraint-satisfying samples, QUBO ground level, true
   optimum and lowest 0.1% of states with exact 95% intervals, the approximation ratio with a
   Student-t interval, the noiseless values, whether the two nulls differ from each other and from
   the uniform rate, and the retained fraction of the excess of constraint-satisfying samples;
 - the circuit: transpiler seed, two-qubit gates, depth, physical qubits, the device's reported
   two-qubit errors and their product over the circuit's gates, the reported readout errors and the
   date of the calibration data, all read at submission; the measured readout errors; QPU seconds.

Device. QFPO_IBM_BACKEND names it; there is no default. --submit refuses ibm_marrakesh, the device
of Round 2A, unless --same-device is given. The result of such a run is labelled "same device,
other day": it is a repeat, not a replication on a second device.

Second device, chosen on 20 November 2026 before any run of this script. The account has three
devices, all of the processor type of ibm_marrakesh (Heron r2, 156 qubits): ibm_marrakesh, ibm_fez
and ibm_kingston. Rule: of the two others, the operational one with fewer pending jobs when the
choice is made. That was ibm_kingston (1 pending job; ibm_fez 6).

No run is excluded. Every submitted job is listed in results/e26_submission.json and in every
result. Raw counts are archived before any analysis. A device result is never replaced. A device
that already has a finished job of this experiment gets a further one only with --again.

Usage
  e26_hardware_qaoa_replication.py --plan
        offline: the job, two-qubit gate counts on packaged device models, the QPU-seconds estimate
        with its basis, and the smallest qaoa rate the test detects with power 0.8
  e26_hardware_qaoa_replication.py --dry-run [--noise P] [--quick]
        Aer on the logical circuit: the whole pipeline with its analysis (P: depolarising error of
        every two-qubit gate; --quick: 4 000 shots per binding). The result name ends in _dryrun.
  e26_hardware_qaoa_replication.py --status     device, account usage, state of the submitted jobs
  e26_hardware_qaoa_replication.py --submit [--same-device] [--again]
  e26_hardware_qaoa_replication.py --collect    archive the counts of finished jobs, then analyse
  e26_hardware_qaoa_replication.py --analyse    analyse archived counts that have no result yet
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import hardware_jobs as hj
from common import point_seed

e14 = hj.e14

EXPERIMENT = "e26"
STEM = "e26_hardware_qaoa_replication"
ALPHA = 0.01
SHOTS = e14.QAOA_SHOTS                           # 40 000 per binding, as in Round 2A
CAL_SHOTS = e14.CAL_SHOTS                        # 4 000 per calibration circuit, as in Round 2A
QUICK_SHOTS, QUICK_CAL_SHOTS = 4_000, 400        # --dry-run --quick
DESIGN = __doc__.split("\nUsage\n")[0].strip()


# --------------------------------------------------------------------------- the job


def job_pubs(problem: dict, circuit, backend, physical: list, shots: int, cal_shots: int) -> list:
    """The pubs of E14 part A: the circuit with its three bindings, then all-0 and all-1 on the measured qubits."""
    return ([(circuit, list(problem["bindings"].values()), shots)]
            + [(c, None, cal_shots) for c in e14.calibration_circuits(backend, physical)])


def prepare(problem: dict, backend) -> tuple:
    """E14's hardware branch for a device or a device model. Returns (pubs, what the manifest keeps of the job)."""
    seed, isa = hj.transpile_best(problem["circuit"], e14.TRANSPILER_SEEDS, backend=backend)
    pubs = job_pubs(problem, isa, backend, e14.measured_physical(isa), SHOTS, CAL_SHOTS)
    facts = {"arms": list(problem["bindings"]), "parameters": dict(problem["bindings"]["qaoa"]), "qubits": problem["n"],
             "shots": SHOTS, "calibration_shots": CAL_SHOTS, "transpiler_seed": seed, **hj.circuit_facts(backend, isa)}
    return pubs, facts


# --------------------------------------------------------------------------- analysis (the same for device and Aer)


def analyse_counts(problem: dict, arm_names: list, counts: list) -> dict:
    """E14 part A's analysis of the counts of one job, and the test that decides the replication."""
    arms = {name: e14.qaoa_arm(problem, counts[0][i], problem["ideal"][name]) for i, name in enumerate(arm_names)}
    tests = e14.qaoa_tests(problem, arms, point_seed(26, 1))
    p = tests["feasible_qaoa_vs_pooled_null_fisher_p"]
    zero, one = counts[1][0], counts[2][0]
    return {"test": {"hypothesis": "P(constraint-satisfying | qaoa) > P(constraint-satisfying | pooled nulls)",
                     "method": "one-sided Fisher exact test", "p": p, "alpha": ALPHA, "rejects": bool(p <= ALPHA)},
            "arms": arms, "tests": tests,
            "measured_readout": e14.readout_errors(zero, one, problem["n"], sum(zero.values()))}


def against_round_2a(result: dict) -> dict:
    """Descriptive only: the counts of this run next to the stored ones of Round 2A, and their rate ratios."""
    old = hj.round_2a()
    q, null = result["arms"]["qaoa"], result["tests"]["pooled_null_feasible"]
    k, n, k0, n0 = q["feasible"]["count"], q["shots"], null["count"], null["shots"]
    return {"round_2a": old,
            "this_run": {"feasible": k, "shots": n, "null_feasible": k0, "null_shots": n0,
                         "approx_ratio": q["approx_ratio"],
                         "retained_fraction": result["tests"]["retained_fraction_of_feasible_excess"]},
            "qaoa_rate_over_null_rate": {"this_run": hj.rate_ratio(k, n, k0, n0),
                                         "round_2a": hj.rate_ratio(old["feasible"], old["shots"], old["null_feasible"],
                                                                   old["null_shots"])},
            "this_run_over_round_2a": {"qaoa_rate": hj.rate_ratio(k, n, old["feasible"], old["shots"]),
                                       "null_rate": hj.rate_ratio(k0, n0, old["null_feasible"], old["null_shots"])},
            "note": "descriptive only: no test is made on these comparisons"}


def relation_to_round_2a(device: str, submitted_utc: str) -> str:
    old = hj.round_2a()
    return (("same device" if device == old["device"] else "other device") + ", "
            + ("same day" if submitted_utc[:10] == old["submitted_utc"][:10] else "other day"))


def verdict(rejects: bool, relation: str) -> str:
    outcome = "the test rejects" if rejects else "the test does not reject"
    if relation.startswith("same device"):
        return f"repeat on the device of Round 2A, not a replication on a second device: {outcome}"
    return ("replicated on a second device: " if rejects else "not replicated on a second device: ") + outcome


def ratio_text(r: dict) -> str:
    return "not defined (a count is 0)" if r["ratio"] is None else f"{r['ratio']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"


def report(label: str, result: dict, comparison: dict) -> None:
    q, null, test, old = result["arms"]["qaoa"], result["tests"]["pooled_null_feasible"], result["test"], comparison["round_2a"]
    print(f"{label}: constraint-satisfying {q['feasible']['count']}/{q['shots']} (noiseless rate "
          f"{q['feasible']['ideal_rate']:.4f}) against {null['count']}/{null['shots']} in the pooled nulls (uniform rate "
          f"{result['tests']['uniform_feasible_rate']:.5f}); one-sided Fisher p = {test['p']:.2e}, alpha {ALPHA}: "
          f"{'rejects' if test['rejects'] else 'does not reject'}")
    print(f"  r = {q['approx_ratio']:.3f} +- {q['approx_ratio_ci95']:.3f} (noiseless {q['ideal_approx_ratio']:.3f}); retained "
          f"fraction of the excess {result['tests']['retained_fraction_of_feasible_excess']['estimate']:.3f}; nulls differ "
          f"p = {result['tests']['nulls_differ_fisher_p']:.2f}; mean measured readout error "
          f"{result['measured_readout']['mean_assignment_error']:.4f}")
    print(f"  descriptive, against Round 2A on {old['device']} ({old['feasible']}/{old['shots']} against "
          f"{old['null_feasible']}/{old['null_shots']}): qaoa rate over null rate "
          f"{ratio_text(comparison['qaoa_rate_over_null_rate']['this_run'])} here, "
          f"{ratio_text(comparison['qaoa_rate_over_null_rate']['round_2a'])} there; qaoa rate here over qaoa rate there "
          f"{ratio_text(comparison['this_run_over_round_2a']['qaoa_rate'])}", flush=True)


# --------------------------------------------------------------------------- --plan


def plan() -> None:
    problem, old, basis = e14.qaoa_problem(), hj.round_2a(), hj.usage_basis()
    n, arms = problem["n"], list(problem["bindings"])
    print("E26, replication of hardware round 2A: one job")
    for model in hj.device_models():
        pubs, facts = prepare(problem, model)
        print(f"  {model.name}: {facts['two_qubit_gates']} two-qubit gates, depth {facts['depth']}, transpiler seed "
              f"{facts['transpiler_seed']}; product of (1 - error) over the two-qubit gates {facts['p_no_two_qubit_error']:.3f} "
              "(packaged error data, not the device's current data)")
    print(f"  Round 2A on {old['device']}: {old['two_qubit_gates']} two-qubit gates, depth {old['depth']}")
    print(f"  pub 0: the QAOA circuit, {n} qubits, bindings {', '.join(arms)}: {len(arms)} x {SHOTS:,} shots")
    print(f"  pubs 1 and 2: readout calibration, all 0 and all 1 on the {n} measured qubits: 2 x {CAL_SHOTS:,} shots")
    estimate = hj.qpu_seconds(pubs, basis)
    print(f"  {estimate['pubs']} pubs, {estimate['shots']:,} shots, at most {e14.MAX_JOB_SECONDS} s of execution allowed")
    hj.print_estimate(estimate, basis)

    feasible = problem["masks"]["feasible"]
    null_shots = SHOTS * (len(arms) - 1)
    print(f"Test: one-sided Fisher exact, {SHOTS:,} qaoa shots against {null_shots:,} pooled null shots, alpha {ALPHA}.")
    print(f"  Smallest rate of constraint-satisfying samples of the qaoa binding detected with power {hj.POWER}:")
    for label, null in ((f"uniform rate {int(feasible.sum())}/{len(feasible):,}", float(feasible.mean())),
                        (f"Round 2A null rate {old['null_feasible']}/{old['null_shots']:,}",
                         old["null_feasible"] / old["null_shots"])):
        excess = hj.detectable(lambda d: hj.fisher_power(SHOTS, null * (1 + d), null_shots, null, ALPHA), 8.0)
        rate = null * (1 + excess)
        print(f"    nulls at the {label} = {null:.6f}: qaoa rate {rate:.6f} = {rate * SHOTS:.1f} per {SHOTS:,} shots, "
              f"{1 + excess:.3f} times the null rate (power {hj.fisher_power(SHOTS, rate, null_shots, null, ALPHA):.4f})")
    ideal = float(problem["ideal"]["qaoa"][feasible].sum())
    print(f"  For scale: Round 2A measured {old['feasible']} per {old['shots']:,} ({old['feasible'] / old['shots']:.6f}); "
          f"the noiseless circuit gives {ideal:.6f} = {ideal * SHOTS:.1f} per {SHOTS:,}.")
    print(f"  Method: {hj.POWER_METHOD}.")


# --------------------------------------------------------------------------- --dry-run


def dry_run(args) -> None:
    started = time.time()
    problem = e14.qaoa_problem()
    n, arms = problem["n"], list(problem["bindings"])
    shots, cal_shots = (QUICK_SHOTS, QUICK_CAL_SHOTS) if args.quick else (SHOTS, CAL_SHOTS)
    circuit = hj.logical(problem["circuit"])
    pubs = job_pubs(problem, circuit, SimpleNamespace(num_qubits=n), list(range(n)), shots, cal_shots)
    counts = hj.run_aer(pubs, 26, args.noise)
    result = analyse_counts(problem, arms, counts)
    comparison = against_round_2a(result)
    execution = f"Aer on the logical circuit, two-qubit depolarising error {args.noise or 0}"
    report(f"dry run ({execution}; {shots:,} shots per binding)", result, comparison)
    hj.save_result(hj.dry_run_name(STEM, args), {
        "design": DESIGN, "execution": execution,
        "note": "dry run: it validates the pipeline and the analysis and carries no information about a device",
        "relation_to_round_2a": "dry run, no device", "verdict": "dry run, no verdict",
        "parameters": dict(problem["bindings"]["qaoa"]), "qubits": n, "arm_names": arms, "shots": shots,
        "calibration_shots": cal_shots,
        "circuit": {"two_qubit_gates": e14.two_qubit_gates(circuit), "depth": circuit.depth()},
        **result, "against_round_2a": comparison, "wall_s": round(time.time() - started, 1)})


# --------------------------------------------------------------------------- the device


def submit(args) -> None:
    name, old = os.environ.get("QFPO_IBM_BACKEND"), hj.round_2a()
    if not name:
        raise SystemExit("not submitted: set QFPO_IBM_BACKEND to the name of the device")
    if name == old["device"] and not args.same_device:
        raise SystemExit(f"not submitted: {name} is the device of Round 2A. A replication needs another device; a repeat "
                         "on the same device needs --same-device and is labelled \"same device, other day\"")
    if name != old["device"] and args.same_device:
        raise SystemExit(f"not submitted: --same-device is for a repeat on {old['device']}, but QFPO_IBM_BACKEND is {name}")
    hj.refuse_repeat(EXPERIMENT, name, args.again)
    hj.e20.require_committed_code()
    opened, backend = hj.device()
    if backend.name != name:
        raise SystemExit(f"not submitted: asked for {name}, the service returned {backend.name}")
    pubs, facts = prepare(e14.qaoa_problem(), backend)
    hj.submit(EXPERIMENT, __file__, opened, backend, pubs, e14.MAX_JOB_SECONDS, facts, again=args.again)


def analyse() -> None:
    todo = hj.to_analyse(EXPERIMENT, STEM)
    if not todo:
        print("E26: no archived job is without a result")
        return
    hj.e20.require_committed_code()
    problem = e14.qaoa_problem()
    for entry, record in todo:
        started = time.time()
        requested = [[entry["shots"]] * len(entry["arms"]), [entry["calibration_shots"]], [entry["calibration_shots"]]]
        if record["shots"] != requested:
            raise SystemExit(f"job {entry['job_id']} not analysed: shots per pub {record['shots']}, requested {requested}")
        if entry["parameters"] != problem["bindings"]["qaoa"] or entry["arms"] != list(problem["bindings"]):
            raise SystemExit(f"job {entry['job_id']} not analysed: the bindings of e14.qaoa_problem() are not those submitted")
        result = analyse_counts(problem, entry["arms"], record["counts"])
        comparison = against_round_2a(result)
        relation = relation_to_round_2a(entry["device"], entry["submitted_utc"])
        report(f"{entry['device']}, job {entry['job_id']} ({relation})", result, comparison)
        circuit = {key: entry[key] for key in ("transpiler_seed", "two_qubit_gates", "depth", "physical_qubits",
                                               "median_two_qubit_error", "two_qubit_gate_names", "p_no_two_qubit_error",
                                               "reported_readout_error", "calibration_date")}
        hj.save_result(f"{STEM}_{entry['tag']}", {
            "design": DESIGN, "execution": entry["device"], "relation_to_round_2a": relation,
            "verdict": verdict(result["test"]["rejects"], relation),
            "parameters": entry["parameters"], "qubits": entry["qubits"], "arm_names": entry["arms"],
            "shots": entry["shots"], "calibration_shots": entry["calibration_shots"], "circuit": circuit,
            **result, "against_round_2a": comparison, **hj.job_summary(EXPERIMENT, entry, __file__),
            "wall_s": round(time.time() - started, 1)})


def main(argv=None) -> None:
    p = hj.parser(__doc__)
    p.add_argument("--same-device", action="store_true",
                   help="with --submit: repeat on ibm_marrakesh, the device of Round 2A")
    args = hj.parse(p, argv)
    if args.same_device and not args.submit:
        p.error("--same-device belongs to --submit")
    if args.plan:
        plan()
    elif args.dry_run:
        dry_run(args)
    elif args.status:
        hj.status(EXPERIMENT, *hj.device())
    elif args.submit:
        submit(args)
    elif args.collect:
        if not hj.load_manifest(EXPERIMENT)["jobs"]:
            raise SystemExit("E26: no job has been submitted")
        hj.collect(EXPERIMENT, hj.service())
        analyse()
    else:
        analyse()


if __name__ == "__main__":
    main()
