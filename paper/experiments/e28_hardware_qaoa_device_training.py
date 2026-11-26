"""E28 - The product's portfolio QAOA trained on the device, against angles transferred from the simulator.

Analysis plan fixed before any device job of this script; only --plan, --dry-run and offline tests
were run before. Amended once, after the probe and before any training job (see the amendment).

Question. Hardware round 3 (E22, ibm_marrakesh) sampled the product's portfolio QAOA circuits with
angles trained on the simulator and transferred. The product's own hardware path
(qaoa_circuit._run_real_backend) trains the angles on the device with an Estimator. That path has
never been run or measured. Do angles trained on the device give another approximation ratio than
transferred angles, at equal evaluation shots?

Instances. The ten instances of E21 and E22 with 8 assets (instances 10 to 19), built by
e22.problem: depth 2, risk aversion 0.5. The sizes 12 and 16 are not tested.

Guard on the product's settings. --plan prints it; --probe, --train and --submit stop unless the
source of qaoa_circuit._run_real_backend and quantum_optimize holds the lines that E28 mirrors (the
list MIRRORED: the estimator with exactly five options, 1 000 default shots, XY4 dynamical
decoupling, gate twirling with "auto" randomisations; the pub; COBYLA with tol=1e-2 and maxiter;
max_iterations: int = 12; the pass manager at optimization_level=3; the ansatz with measure_all)
and unless qaoa._interp_init_params(4, 2) is [pi/8, pi/2, pi/2, pi/8].

Phase 0, --probe: the product's hardware path as shipped, once.
 - qaoa.quantum_optimize(prices of instance 10, risk_aversion=0.5, qc_simulator_mode=False,
   max_iterations=5, reps=2), unchanged. The product opens its own service and takes the least
   busy device. 5 is the gateway's cap for a default hardware request; SciPy 1.17.0, the version
   of this environment, turns it into 6 objective values for four variables.
 - Before the call: the code must be committed, the IBM token is loaded as the other hardware
   scripts load it, and the manifest must hold no probe (a second probe needs --again).
 - The manifest results/e28_submission.json gets a record of kind "probe" under "probes", written
   with the state STARTED before the call. After the call it holds start and end (UTC), the
   device, the returned result (final_parameters, optimization_history, selection, solution and
   the rest), whether the returned selection is an exact optimum (e19.score), and the state DONE.
   If the call raises: the state FAILED with the type, the message and the last six frames of the
   traceback as text. The device is read from the product's log line "Using backend", else from
   the returned vendor_runs, else from the account's jobs.
 - Every job is listed under "jobs" with kind "probe": the identifiers of the returned
   vendor_runs, and, whether or not the call succeeds, those of the account's jobs created since
   30 seconds before the call started (service.jobs through hj.service(), at most the 50 most
   recent). Each entry names the list or lists it came from.
 - --collect archives for every finished probe job the QPU seconds, hj.job_times and the raw
   result (estimator: evs and stds as lists; sampler: counts) under results/raw.
 - The device of the last probe is the device of phases 1 and 2. They stop without a finished
   probe, and if QFPO_IBM_BACKEND is set to another name.
 - Contingency. If the probe fails because the service refuses a circuit that contains
   measurements in an Estimator pub, phase 1 uses the same transpiled ansatz with its final
   measurements removed and nothing else changed, and the result says so. This needs --train
   --without-measurements, which is accepted only when the last probe is FAILED; the probe's
   failure text is stored with the training as the reason. The script does not judge the reason.
   Any other failure of the probe: phases 1 and 2 are not run; the plan is revised in a new commit.

Phase 1, --train: training on the device with the product's settings, the ten instances together.
 - Circuits. Per instance the product's ansatz with measure_all, one transpiler pass as the product
   makes it, generate_preset_pass_manager(optimization_level=3, backend=device), with
   seed_transpiler=0 added for the record (the product's pass has no seed). Made once when phase 1
   starts and written to results/raw as QPY; hj.circuit_facts of each is kept in the manifest. The
   circuits read back from that file are the circuits of phases 1 and 2.
 - Optimiser per instance: scipy minimize(method="COBYLA", tol=1e-2, options={"maxiter": 12})
   from qaoa._interp_init_params(4, 2). The objective is the Estimator value.
 - Lockstep by replay. COBYLA is deterministic given the objective values. Per instance the
   recorded (point, value) pairs are kept; the same minimize call is run on them (it stops if a
   point it asks for differs from the recorded one by more than 1e-12) until it asks for a
   further point or returns. Step t is ONE EstimatorV2 job (job mode) whose pubs are, for every
   instance that has not returned, (transpiled circuit, hamiltonian.apply_layout(circuit.layout),
   [point]). Options: default_shots 1 000, XY4 dynamical decoupling, gate twirling with "auto"
   randomisations, as the product sets them, and max_execution_time 300 s; nothing else is set.
   The script waits with job.wait_for_final_state and stores values (evs), stds, QPU seconds and
   hj.job_times in the manifest (kind "train") and the raw result under results/raw.
 - At most 12 values per instance: the script stops rather than ask for a thirteenth. Under
   outcome (b) of the amendment the training ends after the sixth value of every instance: no
   seventh step is submitted.
 - Resumable. Started again, --train goes on after the last stored step. A submitted step without
   values is waited for; it is never submitted twice. A step whose job ends in ERROR or CANCELLED
   is submitted once more (attempt 2, both listed); a second failure stops the run.
 - Budget rule, before step 2 and every later step. As first written: if 12 x (QPU seconds of
   step 1) + the estimate of the evaluation job exceeds 300 s, or step 1 reports no QPU seconds,
   nothing further is submitted. As amended (see the amendment): the budget is 800 s, and the
   QPU seconds of step 1 decide between the outcomes (a), (b) and (c). The decision is taken
   once, before step 2, and stored in the manifest; before every later step it is read. If
   step 1 reports no QPU seconds, no decision is taken and nothing further is submitted.
 - Angles. D12 = the point minimize returns from the recorded values. D5 = the point the same
   call with options={"maxiter": 5} returns from the recorded values of the instance in order;
   SciPy 1.17.0 raises a maxiter below 6 to 6 for four variables, and the number of values used is
   recorded. S = the initial angles. Under outcome (b) there is no D12, and D5 comes from the six
   recorded values: the script stops with an error if that call asks for a seventh point.
 - The code must be committed when --train starts.

Phase 2, --submit: ONE sampler job through hj.submit (XY4, gate twirling; kind "evaluation"). It
needs phase 1 complete. Its arms are those of the stored decision of the budget rule.
 - Per instance one pub: the transpiled circuit of phase 1 bound ten ways, 2 000 shots each: S,
   D5, D12, T0 to T4 (the transferred angles: the product's simulator path with the five E21
   seeds, regenerated by e22.trained_runs and checked by e22.check_against_e21), null_gamma and
   null_beta (the nulls of E22: every gamma 0 with the betas of T0; every beta 0 with the gammas
   of T0). The exact output of each null is checked to be uniform before the job is built.
   Under outcome (b) the circuit is bound nine ways: the same without D12.
 - Per instance a readout calibration on its measured physical qubits, all 0 and all 1, 1 000
   shots each. It is reported, not used.
 - 30 pubs, 10 x 10 x 2 000 + 10 x 2 x 1 000 = 220 000 shots; under outcome (b) 30 pubs,
   10 x 9 x 2 000 + 10 x 2 x 1 000 = 200 000 shots. Not submitted if hj.qpu_seconds estimates
   more than 150 s.

Analysis (--collect, --analyse; the same code for a device and a dry run).
 - Per instance and binding, with e22.analyse_circuit, so the definitions are those of round 3:
   approximation ratio r = (c_mean - mean sampled cost) / (c_mean - c_min), here also with a
   Student-t 95% interval over the shots; probability of an exact optimum; batches of 200 shots
   (share of batches whose selected sample is an optimum; mean gap); the exact r of the same
   angles from the statevector.
 - r_T of an instance = the mean of r over T0 to T4.
 - PRIMARY. Two-sided exact Wilcoxon signed-rank test over the ten instances of r_D12 - r_T
   (scipy.stats.wilcoxon, method "exact"), alpha = 0.01; the test is on the device-trained arm
   (D12, or D5 under outcome (b)), so under outcome (b) it is on r_D5 - r_T. If a difference is
   zero or two absolute differences are equal, wilcoxon is called with its default method, as
   e20.paired_test calls it, and the result says so. The result states the direction: on how
   many instances each arm has the larger r, the median difference and its range. "Training on
   the device and transfer differ" only if p <= alpha; otherwise "no difference shown".
 - SECONDARY, unadjusted, alpha = 0.01, one test: one-sided, r_D12 > r_S (training improved on
   its start), same test; again on the device-trained arm (D12, or D5 under outcome (b)), so
   under outcome (b) it is r_D5 > r_S.
 - Descriptive, no test: D5 against S (under outcome (b) this coincides with the secondary test
   and is not repeated); T against the pooled nulls and the device-trained arm (D12, or D5 under
   outcome (b)) against the pooled nulls (r and optimum probability); the exact r of S, D5, D12
   and T; per instance the recorded objective values, their minimum and the value at the
   returned point, and for S, D5 and D12 the estimator value recorded at that point next to the
   mean sampled cost of the same angles in the evaluation job (under outcome (b) each of these
   without D12); the measured readout errors; circuit facts (two-qubit gates, depth, physical
   qubits, reported errors, calibration date); QPU seconds of probe, training and evaluation;
   hj.job_times of every job; the minutes from the first training step to the evaluation job;
   the probe's outcome; the decision of the budget rule with its numbers.

Limits, stated in every result. The arms are compared at equal evaluation shots. The
device-trained arm used at most 12 x 1 000 estimator shots per instance (6 x 1 000 under outcome
(b) of the amendment) plus the estimator's own calibration shots; the transferred arm used none
on the device. The sizes n = 12 and 16 are not tested. One device, one session.

No run is excluded. Every job is listed in results/e28_submission.json. Raw results are archived
before any analysis. A device result is never replaced
(results/e28_hardware_qaoa_device_training_<tag>.json).

Amendment of 21 November 2026, after the probe and before any training job.
 - What the probe showed. Phase 0 ran on 21 November 2026 on ibm_fez, the device the product
   chose. The product's path ran as shipped: six Estimator jobs of one pub each, 11 QPU seconds
   each, and the sampler job, 5 QPU seconds.
 - The budget rule of phase 1 had been written before any Estimator cost was known.
 - The new budget is 800 QPU seconds for training and evaluation together; the probe's seconds
   are not part of it. u1 = the QPU seconds of step 1 (ten pubs). E12 = the hj.qpu_seconds
   estimate of the evaluation job with ten bindings per instance (69.8 s). E6 = the same
   estimate with nine bindings per instance (63.5 s).
   (a) If 12 x u1 + E12 <= 800: the plan stands. At most 12 values per instance; arms S, D5,
       D12, T0 to T4, null_gamma, null_beta; primary test on r_D12 - r_T; secondary r_D12 > r_S.
   (b) Otherwise, if 6 x u1 + E6 <= 800: the training ends after the sixth value of every
       instance (the number of values the gateway's cap of 5 yields with SciPy 1.17.0). There
       is then no D12. D5 is computed as before (the replay with maxiter 5 on the six recorded
       values; the script stops with an error if it asks for a seventh point). The evaluation
       job has nine bindings per instance (S, D5, T0 to T4, the two nulls), 10 x 9 x 2 000 +
       10 x 2 x 1 000 = 200 000 shots. The device-trained arm of the primary test is D5
       (two-sided, r_D5 - r_T) and of the secondary test is D5 (one-sided, r_D5 > r_S). The
       descriptive item "D5 against S" then coincides with the secondary test and is not
       repeated.
   (c) Otherwise nothing further is submitted.
   If step 1 reports no QPU seconds, nothing further is submitted (as before); the decision is
   then not taken, and a later --train asks for the seconds again.
 - The decision is taken once, when --train goes on after step 1, and stored in the manifest
   under "training" as "decision" (u1_s, e12_s, e6_s, budget_s, outcome "twelve", "six" or
   "stop", values_per_instance, decided_utc). A resumed --train reads it and never decides
   again. --submit, the analysis (--collect, --analyse) and --status read the decision from the
   manifest: the arms of the evaluation job, the device-trained arm of the tests and the texts
   printed follow it. A device result states the decision with its numbers.
 - The choice depends on QPU seconds only: no objective value enters it.
 - Nothing else is changed.

Usage (on a device in this order: --probe, --collect, --train, --submit, --collect)
  e28_hardware_qaoa_device_training.py --plan
        offline: the guard, the instances, circuits on packaged device models, jobs and shots of
        every phase, the QPU-seconds estimate of the evaluation job in both variants, the
        amended budget rule with its thresholds, the exact r of S and T, and the facts of the
        primary test
  e28_hardware_qaoa_device_training.py --dry-run [--noise P] [--quick] [--six]
        Aer on the logical circuits: the whole pipeline with its analysis. Estimator stand-in:
        1 000 shots and the mean cost of the samples, no mitigation. The probe is replaced by the
        same loop with maxiter 5 on instance 10. P: depolarising error of every two-qubit gate.
        --quick: three instances, at most 6 values per instance (D5 and D12 then coincide), 400
        evaluation shots per binding. --six: outcome (b) of the amendment (six values per
        instance, nine bindings, no D12, also with --quick); without it outcome (a). No QPU
        seconds enter a dry run. The result name ends in _dryrun, with --six in _six_dryrun, with
        --six --quick in _six_quick_dryrun.
  e28_hardware_qaoa_device_training.py --status
        account usage, probe, the stored decision, state of the listed jobs
  e28_hardware_qaoa_device_training.py --probe [--again]
  e28_hardware_qaoa_device_training.py --train [--without-measurements]
  e28_hardware_qaoa_device_training.py --submit
  e28_hardware_qaoa_device_training.py --collect    archive finished jobs, then analyse
  e28_hardware_qaoa_device_training.py --analyse    analyse an archived evaluation job without a result
"""

from __future__ import annotations

import calendar
import inspect
import io
import json
import logging
import math
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import hardware_jobs as hj
import numpy as np
from common import mean_ci, provenance
from scipy.optimize import minimize
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

import core.optimizers.quantum.ibm.qaoa_circuit as qaoa

e14, e20 = hj.e14, hj.e20
e19 = hj._import_script("e19_portfolio")                 # instances, scores, exact states
e22 = hj._import_script("e22_hardware_portfolio_qaoa")   # the problem, the transferred angles, the analysis of round 3

EXPERIMENT = "e28"
STEM = "e28_hardware_qaoa_device_training"
ALPHA = 0.01
N = 8
INSTANCES = list(range(10, 20))
REPS, RISK_AVERSION = 2, 0.5
PROBE_INSTANCE, PROBE_MAXITER = 10, 5            # 5: the gateway's cap for a default hardware request
MAXITER, D5_MAXITER, TOL = 12, 5, 1e-2           # 12: the default of the product's hardware path
SIX = 6                                          # values per instance under outcome (b) of the amendment
ESTIMATOR_SHOTS = 1_000
EVAL_SHOTS, CAL_SHOTS = 2_000, 1_000
TRANSPILER_SEED = 0
BUDGET_S = 800.0                                 # training and evaluation together; 300.0 before the amendment
MAX_EVALUATION_S = 150.0                         # largest estimate of the evaluation job
MAX_JOB_SECONDS = e14.MAX_JOB_SECONDS            # 300 s of execution allowed per job
ACCOUNT_MARGIN_S = 30                            # the account's jobs are listed from this long before the probe
QUICK = {"instances": 3, "maxiter": 6, "shots": 400}
T_ARMS = [f"T{k}" for k in range(e22.SEEDS)]
NULLS = ["null_gamma", "null_beta"]
ARMS = ["S", "D5", "D12", *T_ARMS, *NULLS]
ARMS_SIX = [arm for arm in ARMS if arm != "D12"]  # outcome (b) of the amendment
GROUPS = {"S": ["S"], "D5": ["D5"], "D12": ["D12"], "T": T_ARMS, "null": NULLS}
OUTCOMES = {"twelve": "outcome (a): at most 12 values per instance, device-trained arm D12",
            "six": "outcome (b): the training ends after the sixth value of every instance, no D12, device-trained arm D5",
            "stop": "outcome (c): nothing further is submitted"}
ESTIMATOR_LINES = ["estimator.options.default_shots = 1000", "estimator.options.dynamical_decoupling.enable = True",
                   'estimator.options.dynamical_decoupling.sequence_type = "XY4"',
                   "estimator.options.twirling.enable_gates = True", 'estimator.options.twirling.num_randomizations = "auto"']
MIRRORED = {"_run_real_backend": ESTIMATOR_LINES + [
    "estimator = Estimator(mode=backend)", "isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)",
    "pub = (ansatz, isa_hamiltonian, params)", "job = estimator.run([pub])", 'method="COBYLA",', "tol=1e-2,",
    'options={"maxiter": max_iterations},', "max_iterations: int = 12) -> dict:",
    "pm = generate_preset_pass_manager(optimization_level=3, backend=backend)", "candidate_circuit = pm.run(ansatz)"],
    "quantum_optimize": ["ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)", "ansatz.measure_all()"]}
LIMITS = ("The arms are compared at equal evaluation shots. The device-trained arm used at most 12 x 1 000 estimator "
          "shots per instance (6 x 1 000 under outcome (b) of the amendment) plus the estimator's own calibration shots; "
          "the transferred arm used none on the device. The sizes n = 12 and 16 are not tested. One device, one session.")
DESIGN = __doc__.split("\nUsage (")[0].strip()


def utc(moment=None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))


def epoch(stamp: str) -> int:
    return calendar.timegm(time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S"))


# --------------------------------------------------------------------------- the product's settings


def product_settings() -> dict:
    """What E28 mirrors of the product: the mirrored source lines that are missing, and the initial angles."""
    missing = []
    for name, wanted in MIRRORED.items():
        lines = [line.strip() for line in inspect.getsource(getattr(qaoa, name)).splitlines()]
        missing += [f"{name}: {line}" for line in wanted if line not in lines]
        if name == "_run_real_backend" and sum(line.startswith("estimator.options.") for line in lines) != len(ESTIMATOR_LINES):
            missing.append(f"{name}: exactly {len(ESTIMATOR_LINES)} estimator options")
    initial = [float(x) for x in qaoa._interp_init_params(2 * REPS, REPS)]
    expected = [math.pi / 8, math.pi / 2, math.pi / 2, math.pi / 8]
    same = len(initial) == 4 and max(abs(a - b) for a, b in zip(initial, expected)) <= 1e-15
    defaults = e22.D["reps"] == REPS and e22.D["risk_aversion"] == RISK_AVERSION
    return {"missing": missing, "initial_angles": initial, "initial_angles_as_expected": same,
            "e22_depth_and_risk_aversion_as_here": defaults, "ok": not missing and same and defaults}


def require_product_settings() -> None:
    facts = product_settings()
    if not facts["ok"]:
        raise SystemExit("refused: the product's hardware path is not the one E28 mirrors: "
                         + json.dumps({k: v for k, v in facts.items() if k != "ok"}))


# --------------------------------------------------------------------------- instances, angles, bindings

_PROBLEMS: dict = {}


def problem(instance: int) -> dict:
    """E22's problem of an instance with the product's Hamiltonian, the parameter names and the logical circuit."""
    if instance not in _PROBLEMS:
        prices, linear, quadratic, ansatz, cost = e22.problem(N, instance)
        _PROBLEMS[instance] = {"prices": prices, "ansatz": ansatz, "cost": cost, "logical": hj.logical(ansatz),
                               "hamiltonian": qaoa._build_hamiltonian(linear, quadratic, N),
                               "names": [p.name for p in ansatz.parameters]}
    return _PROBLEMS[instance]


def initial_angles() -> list:
    return [float(x) for x in qaoa._interp_init_params(2 * REPS, REPS)]


def exact(instance: int, values: dict) -> dict:
    """e19.state_metrics of the exact state with these angles, and its largest deviation from the uniform distribution."""
    p = problem(instance)
    prob = e19.exact_probabilities(p["logical"].assign_parameters({q: values[q.name] for q in p["logical"].parameters}))
    return {**e19.state_metrics(prob, p["cost"]), "deviation_from_uniform": float(np.abs(prob - 1 / len(prob)).max())}


def bindings_of(instance: int, angles: dict, runs: dict, arms: list) -> list:
    """The bindings of an instance in the order of arms (ARMS, or ARMS_SIX under outcome (b)).

    angles: S, D5 and, where arms has it, D12 as lists in the order of the ansatz parameters.
    """
    p = problem(instance)
    out = [{"arm": arm, "values": dict(zip(p["names"], [float(v) for v in angles[arm]]))}
           for arm in ("S", "D5", "D12") if arm in arms]
    out += [{"arm": f"T{b['rep']}" if b["arm"] == "qaoa" else b["arm"], "values": b["values"]}
            for b in e22.bindings_for(p["ansatz"], runs, N, instance)]
    if [b["arm"] for b in out] != arms:
        raise RuntimeError(f"unexpected bindings {[b['arm'] for b in out]}")
    for b in out:
        if b["arm"] in NULLS and exact(instance, b["values"])["deviation_from_uniform"] > 1e-9:
            raise RuntimeError(f"{b['arm']} of instance {instance} is not uniform in the exact state")
    return out


def transferred(instances: list) -> tuple:
    """The transferred angles: E22's regeneration of the E21 runs and its check against the E21 records.

    Its worker processes transpile: call it before this process has transpiled anything, or the forked workers hang.
    """
    runs = e22.trained_runs([(N, i) for i in instances])
    return runs, e22.check_against_e21(runs)


def evaluation_pubs(instances: list, circuits: dict, bindings: dict, backend, shots: int, cal_shots: int) -> list:
    """Per instance the circuit with its bindings; then per instance all-0 and all-1 on its measured qubits."""
    pubs = [(circuits[i], [b["values"] for b in bindings[i]], shots) for i in instances]
    for i in instances:
        pubs += [(c, None, cal_shots) for c in e14.calibration_circuits(backend, e14.measured_physical(circuits[i]))]
    return pubs


def evaluation_estimate(arms: list, instances: int = len(INSTANCES), shots: int = EVAL_SHOTS, cal_shots: int = CAL_SHOTS) -> dict:
    """hj.qpu_seconds of the evaluation job with these arms; the estimate reads only the bindings and shots of the pubs."""
    return hj.qpu_seconds([(None, arms, shots)] * instances + [(None, None, cal_shots)] * (2 * instances))


def decide(u1: float) -> dict:
    """The amended budget rule: the outcome that follows from the QPU seconds of step 1. No objective value enters."""
    e12, e6 = evaluation_estimate(ARMS)["seconds"], evaluation_estimate(ARMS_SIX)["seconds"]
    outcome = "twelve" if MAXITER * u1 + e12 <= BUDGET_S else "six" if SIX * u1 + e6 <= BUDGET_S else "stop"
    return {"u1_s": u1, "e12_s": e12, "e6_s": e6, "budget_s": BUDGET_S, "outcome": outcome,
            "values_per_instance": {"twelve": MAXITER, "six": SIX, "stop": None}[outcome], "decided_utc": utc()}


def decision_text(decision: dict) -> str:
    """A stored decision with its numbers."""
    u1, e12, e6 = decision["u1_s"], decision["e12_s"], decision["e6_s"]
    return (f"step 1 used {u1:.1f} QPU seconds; {MAXITER} x {u1:.1f} s + {e12:.1f} s = {MAXITER * u1 + e12:.1f} s, "
            f"{SIX} x {u1:.1f} s + {e6:.1f} s = {SIX * u1 + e6:.1f} s, budget {decision['budget_s']:.0f} s: "
            f"{OUTCOMES[decision['outcome']]} (decided {decision['decided_utc']})")


# --------------------------------------------------------------------------- COBYLA by replay


class _NextPoint(Exception):
    def __init__(self, x):
        super().__init__("next point")
        self.x = [float(v) for v in x]


def replay(history: list, maxiter: int) -> dict:
    """The product's minimize call on recorded values. history: [(point, value)] in the order COBYLA asked.

    Returns {"finished": False, "x": the point it asks for next} or {"finished": True, "x": result.x, ...}.
    """
    used = 0

    def objective(x):
        nonlocal used
        if used == len(history):
            raise _NextPoint(x)
        point, value = history[used]
        if np.abs(np.asarray(x, float) - np.asarray(point, float)).max() > 1e-12:
            raise RuntimeError(f"replay: COBYLA asks for {list(x)} at call {used + 1}, recorded {list(point)}")
        used += 1
        return float(value)

    try:
        result = minimize(objective, np.array(initial_angles()), method="COBYLA", tol=TOL, options={"maxiter": maxiter})
    except _NextPoint as asked:
        return {"finished": False, "x": asked.x}
    x = [float(v) for v in result.x]
    at = [k for k, (point, _) in enumerate(history) if np.abs(np.asarray(point) - np.asarray(x)).max() <= 1e-12]
    return {"finished": True, "x": x, "fun": float(result.fun), "values_used": used, "maxiter": maxiter,
            "returned_point_is_value_number": at[0] + 1 if at else None, "status": int(result.status),
            "message": str(result.message)}


def histories(records: list, instances: list) -> dict:
    """Per instance the recorded (point, value) pairs of the steps that have values, in step order."""
    out = {i: [] for i in instances}
    for record in records:
        if record["values"] is not None:
            for i, point, value in zip(record["instances"], record["points"], record["values"]):
                out[i].append((point, value))
    return out


def new_step(step: int, attempt: int, instances: list, points: list) -> dict:
    return {"kind": "train", "step": step, "attempt": attempt, "instances": list(instances),
            "points": [[float(v) for v in x] for x in points], "state": "SUBMITTED", "values": None, "stds": None,
            "usage_s": None}


def lockstep(steps, instances: list, maxiter: int, cap: int | None) -> None:
    """Train every instance, one job per step. steps: .records, .allow(step), .submit(...), .wait(record).

    cap: the largest number of values per instance; the run stops rather than ask for more.
    allow(step) is False if the training ends before that step (outcome (b) of the amendment).
    """
    while True:
        records = steps.records
        waiting = [r for r in records if r["values"] is None and r["state"] not in ("ERROR", "CANCELLED")]
        if waiting:                                      # submitted, no values: wait, never submit it twice
            steps.wait(waiting[-1])
            continue
        step, attempt = (records[-1]["step"] + 1 if records else 1), 1
        if records and records[-1]["values"] is None:    # the last job ended in ERROR or CANCELLED
            if records[-1]["attempt"] >= 2:
                raise SystemExit(f"stopped: step {records[-1]['step']} failed twice (jobs "
                                 f"{[r['job_id'] for r in records if r['step'] == records[-1]['step']]}); nothing further is submitted")
            step, attempt = records[-1]["step"], 2
        history = histories(records, instances)
        asked = {i: replay(history[i], maxiter) for i in instances}
        todo = [i for i in instances if not asked[i]["finished"]]
        if not todo:
            return
        if cap is not None and any(len(history[i]) >= cap for i in todo):
            raise SystemExit(f"stopped: COBYLA asks for more than {cap} values; nothing further is submitted")
        if not steps.allow(step):
            return
        steps.submit(step, attempt, todo, [asked[i]["x"] for i in todo])


def training_outcome(records: list, instances: list, maxiter: int | None) -> dict:
    """Per instance the recorded points and values and the angles S, D5 and D12. Stops unless every instance is finished.

    maxiter: that of D12's call. None under outcome (b): no D12, and D5's call must return on the six recorded values.
    """
    history, out = histories(records, instances), {}
    for i in instances:
        made = {"D5": replay(history[i], D5_MAXITER)}
        if maxiter is not None:
            made["D12"] = replay(history[i], maxiter)
        elif len(history[i]) >= SIX and not made["D5"]["finished"]:
            raise RuntimeError(f"instance {i}: the replay with maxiter {D5_MAXITER} asks for a seventh point")
        if not all(r["finished"] for r in made.values()):
            raise SystemExit(f"training is not complete: instance {i} has {len(history[i])} values and COBYLA asks for more")
        values = [value for _, value in history[i]]
        out[str(i)] = {"points": [point for point, _ in history[i]], "values": values, "minimum": min(values),
                       "S": initial_angles(), **{arm: r["x"] for arm, r in made.items()},
                       **{f"replay_{arm}": r for arm, r in made.items()},
                       "estimator_value": {"S": values[0], **{arm: r["fun"] for arm, r in made.items()}}}
        if np.abs(np.asarray(history[i][0][0]) - np.asarray(initial_angles())).max() > 1e-12:
            raise RuntimeError(f"instance {i}: the first recorded point is not the initial angles")
    return out


# --------------------------------------------------------------------------- analysis (the same for device and Aer)


def cost_stats(cost: np.ndarray, count: dict) -> tuple:
    """Mean cost of the samples and the standard error of that mean. count: {bitstring: count}, asset i is bit i."""
    keys, c = np.array([int(k, 2) for k in count]), np.array(list(count.values()), float)
    mean = float(c @ cost[keys] / c.sum())
    return mean, float(math.sqrt(c @ (cost[keys] - mean) ** 2 / (c.sum() - 1) / c.sum()))


def signed_rank(diff: list, alternative: str) -> dict:
    """Wilcoxon signed-rank test of the differences, exact unless a difference is zero or two absolute ones are equal."""
    d = np.asarray(diff, float)
    out = {"instances": len(d), "alternative": alternative, "alpha": ALPHA, "positive": int((d > 1e-12).sum()),
           "negative": int((d < -1e-12).sum()), "zero": int((np.abs(d) <= 1e-12).sum()), "median": float(np.median(d)),
           "range": [float(d.min()), float(d.max())], "differences": [float(x) for x in d]}
    if out["zero"] == len(d):
        return {**out, "p": None, "rejects": False, "method": "no test: every difference is zero"}
    tied = bool(out["zero"] or len(np.unique(np.abs(d))) < len(d))
    p = float((wilcoxon(d, alternative=alternative) if tied else wilcoxon(d, alternative=alternative, method="exact")).pvalue)
    method = ("scipy.stats.wilcoxon with its default method, as e20.paired_test calls it: a difference is zero or two "
              "absolute differences are equal" if tied else "exact Wilcoxon signed-rank test (scipy.stats.wilcoxon, method exact)")
    return {**out, "p": p, "rejects": bool(p <= ALPHA), "method": method}


def analyse_counts(instances: list, bindings: dict, counts: list, training: dict) -> dict:
    """bindings: {instance: its bindings, ARMS or ARMS_SIX}; counts: the layout of hj.counts_of; training: training_outcome()."""
    arms = [b["arm"] for b in bindings[instances[0]]]
    trained = [arm for arm in ("S", "D5", "D12") if arm in arms]
    device = trained[-1]                                 # the device-trained arm: D12, or D5 under outcome (b)
    groups = {group: members for group, members in GROUPS.items() if set(members) <= set(arms)}
    entries = [{"n": N, "instance": i, "counts": [{str(int(k, 2)): v for k, v in c.items()} for c in counts[at]],
                "bindings": [{"arm": b["arm"] if b["arm"] in NULLS else "qaoa", "rep": j, "values": b["values"]}
                             for j, b in enumerate(bindings[i])]} for at, i in enumerate(instances)]
    rows = []
    analysed = [row for e in entries for row in e22.analyse_circuit((e["n"], e["instance"], e["bindings"], e["counts"]))]
    for row, (i, j, at) in zip(analysed, [(i, j, at) for at, i in enumerate(instances) for j in range(len(arms))]):
        cost = problem(i)["cost"]
        span, mean_cost = float(cost.mean() - cost.min()), cost_stats(cost, counts[at][j])
        ratio = (float(cost.mean()) - mean_cost[0]) / span
        if abs(ratio - row["ratio"]) > 1e-9 or row["instance"] != i or row["binding"] != j:
            raise RuntimeError(f"instance {i}, binding {j}: the ratio computed here differs from e22.analyse_circuit")
        rows.append({"instance": i, "arm": bindings[i][j]["arm"], "values": bindings[i][j]["values"], "shots": row["shots"],
                     "ratio": row["ratio"], "ratio_ci95": float(student_t.ppf(0.975, row["shots"] - 1) * mean_cost[1] / span),
                     "sampled_mean_cost": mean_cost[0], "p_optimum": row["p_optimum"], "optimum_shots": row["optimum_shots"],
                     "p_optimum_uniform": row["p_optimum_uniform"], "batches": row["batches"],
                     "batch_optimal": row["batch_optimal"], "batch_gap": row["batch_gap"], "ideal_ratio": row["ideal_ratio"],
                     "ideal_p_optimum": row["ideal_p_optimum"], "referee_agrees": row["referee_agrees"]})

    def grouped(field: str) -> dict:
        return {group: {i: float(np.mean([r[field] for r in rows if r["instance"] == i and r["arm"] in members]))
                        for i in instances} for group, members in groups.items()}

    def difference(table: dict, a: str, b: str) -> dict:
        d = [table[a][i] - table[b][i] for i in instances]
        return {**mean_ci(d), "range": [float(min(d)), float(max(d))], "first_larger_on": int(sum(x > 0 for x in d))}

    r, p_opt, ideal = grouped("ratio"), grouped("p_optimum"), grouped("ideal_ratio")
    primary = signed_rank([r[device][i] - r["T"][i] for i in instances], "two-sided")
    primary.update(hypothesis=f"r_{device} differs from r_T", larger_r_on_instances={device: primary["positive"], "T": primary["negative"]},
                   verdict="training on the device and transfer differ" if primary["rejects"] else "no difference shown")
    secondary = signed_rank([r[device][i] - r["S"][i] for i in instances], "greater")
    secondary.update(hypothesis=f"r_{device} > r_S (training improved on its start)", note="secondary, unadjusted")
    per_instance = {}
    for i in instances:
        mine, sampled = training[str(i)], {a: next(x["sampled_mean_cost"] for x in rows if x["instance"] == i and x["arm"] == a)
                                             for a in trained}
        per_instance[str(i)] = {"values": mine["values"], "minimum": mine["minimum"],
                                **{f"values_used_{a}": mine[f"replay_{a}"]["values_used"] for a in trained[1:]},
                                "returned_point_is_value_number": {a: mine[f"replay_{a}"]["returned_point_is_value_number"]
                                                                   for a in trained[1:]},
                                "estimator_value_at_point": mine["estimator_value"], "sampled_mean_cost_at_point": sampled}
    readout = []
    for at, i in enumerate(instances):
        zero, one = counts[len(instances) + 2 * at][0], counts[len(instances) + 2 * at + 1][0]
        readout.append({"instance": i, **e14.readout_errors(zero, one, N, sum(zero.values()))})
    descriptive = {
        "note": "descriptive only: no test is made on these",
        "ratio": {g: mean_ci(list(v.values())) for g, v in r.items()}, "ratio_by_instance": r,
        "p_optimum": {g: mean_ci(list(v.values())) for g, v in p_opt.items()}, "p_optimum_by_instance": p_opt,
        "exact_ratio": {g: mean_ci(list(v.values())) for g, v in ideal.items() if g != "null"}, "exact_ratio_by_instance": ideal,
        **({} if device == "D5" else {"D5_minus_S": {"ratio": difference(r, "D5", "S"), "p_optimum": difference(p_opt, "D5", "S")}}),
        "T_minus_null": {"ratio": difference(r, "T", "null"), "p_optimum": difference(p_opt, "T", "null")},
        f"{device}_minus_null": {"ratio": difference(r, device, "null"), "p_optimum": difference(p_opt, device, "null")},
        "training": per_instance, "measured_readout": readout,
        "mean_measured_readout_error": float(np.mean([x["mean_assignment_error"] for x in readout]))}
    return {"alpha": ALPHA, "device_trained_arm": device, "primary": primary, "secondary": secondary, "descriptive": descriptive,
            "limits": LIMITS, "rows": rows}


def report(label: str, result: dict) -> None:
    a, b, d, arm = result["primary"], result["secondary"], result["descriptive"], result["device_trained_arm"]
    print(f"{label}")
    print(f"  PRIMARY r_{arm} - r_T, two-sided: p = {a['p']}, alpha {ALPHA}: {a['verdict']}; {arm} larger on {a['positive']}, T larger "
          f"on {a['negative']} of {a['instances']} instances; median difference {a['median']:+.4f}, range {a['range'][0]:+.4f} "
          f"to {a['range'][1]:+.4f} ({a['method']})")
    print(f"  SECONDARY r_{arm} > r_S, one-sided, unadjusted: p = {b['p']}: {'rejects' if b['rejects'] else 'does not reject'}; "
          f"median difference {b['median']:+.4f}")
    print("  descriptive, mean r over instances: " + ", ".join(f"{g} {v['mean']:+.4f}" for g, v in d["ratio"].items())
          + " | exact r: " + ", ".join(f"{g} {v['mean']:.4f}" for g, v in d["exact_ratio"].items())
          + f" | mean measured readout error {d['mean_measured_readout_error']:.4f}", flush=True)


# --------------------------------------------------------------------------- --plan


def transpile_once(ansatz, backend):
    """The product's single pass (optimisation level 3 for the device), with the transpiler seed fixed for the record."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    return generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=TRANSPILER_SEED).run(ansatz)


def plan() -> None:
    settings, basis = product_settings(), hj.usage_basis()
    e12, e6 = evaluation_estimate(ARMS), evaluation_estimate(ARMS_SIX)
    runs, check = transferred(INSTANCES)
    print("E28, training on the device against transferred angles")
    print(f"Guard on the product's settings: {'passed' if settings['ok'] else 'FAILED: ' + json.dumps(settings)}; initial "
          f"angles {[round(x, 6) for x in settings['initial_angles']]}")
    print(f"Instances {INSTANCES[0]}..{INSTANCES[-1]}, {N} assets, depth {REPS}, risk aversion {RISK_AVERSION}; generator seeds "
          f"{e22.INSTANCE_BASE + 100 * (N // 4) + INSTANCES[0]}..{e22.INSTANCE_BASE + 100 * (N // 4) + INSTANCES[-1]}")
    for model in hj.device_models():
        sizes = [(e14.two_qubit_gates(isa), isa.depth()) for isa in (transpile_once(problem(i)["ansatz"], model) for i in INSTANCES)]
        print(f"  {model.name}, one pass at level 3, seed {TRANSPILER_SEED}: two-qubit gates {min(s[0] for s in sizes)} to "
              f"{max(s[0] for s in sizes)}, depth {min(s[1] for s in sizes)} to {max(s[1] for s in sizes)} over the ten circuits "
              "(packaged device model, not the device's current data)")
    print(f"Phase 0, --probe: the product's call on instance {PROBE_INSTANCE} with max_iterations {PROBE_MAXITER}; SciPy raises "
          f"a COBYLA maxiter below {2 * REPS + 2} to {2 * REPS + 2} for {2 * REPS} variables, so {2 * REPS + 2} estimator jobs of one pub "
          f"with {ESTIMATOR_SHOTS:,} default shots are expected, then one sampler job of 10,000 shots")
    print(f"Phase 1, --train: at most {MAXITER} estimator jobs; each has one pub per unfinished instance (at most "
          f"{len(INSTANCES)}), one point per pub, {ESTIMATOR_SHOTS:,} default shots: at most {MAXITER * len(INSTANCES) * ESTIMATOR_SHOTS:,} "
          f"shots plus the estimator's own calibration shots; under outcome (b) {SIX} jobs, "
          f"{SIX * len(INSTANCES) * ESTIMATOR_SHOTS:,} shots")
    print(f"Phase 2, --submit: one sampler job, {len(INSTANCES)} pubs x {len(ARMS)} bindings x {EVAL_SHOTS:,} shots + "
          f"{2 * len(INSTANCES)} calibration pubs x {CAL_SHOTS:,} shots = {e12['shots']:,} shots; under outcome (b) "
          f"{len(ARMS_SIX)} bindings (no D12), {e6['shots']:,} shots")
    hj.print_estimate(e12, basis)
    print(f"  under outcome (b): {e6['seconds']:.1f} s estimated = {e6['shots']:,} shots ({e6['pubs']} pubs)")
    print(f"  the evaluation job is not submitted above {MAX_EVALUATION_S:.0f} s estimated")
    print("Budget rule as amended on 21 November 2026, decided once after step 1 from u1 = the QPU seconds of step 1 (ten "
          f"pubs; an estimator job of ten pubs has no measured basis, step 1 gives one). Budget {BUDGET_S:.0f} s for training "
          f"and evaluation; E12 = {e12['seconds']:.1f} s, E6 = {e6['seconds']:.1f} s")
    print(f"  (a) if {MAXITER} x u1 + E12 <= {BUDGET_S:.0f} s, that is u1 <= "
          f"{math.floor(100 * (BUDGET_S - e12['seconds']) / MAXITER) / 100:.2f} s (rounded down): at most {MAXITER} values per "
          f"instance, arms {', '.join(ARMS)}; tests on D12; evaluation {e12['shots']:,} shots")
    print(f"  (b) otherwise, if {SIX} x u1 + E6 <= {BUDGET_S:.0f} s, that is u1 <= "
          f"{math.floor(100 * (BUDGET_S - e6['seconds']) / SIX) / 100:.2f} s (rounded down): the training ends after the sixth "
          f"value of every instance, arms {', '.join(ARMS_SIX)}; tests on D5; evaluation {e6['shots']:,} shots")
    print("  (c) otherwise nothing further is submitted. If step 1 reports no QPU seconds, no decision is taken and nothing "
          "further is submitted")
    print(f"Exact r (noiseless), per instance; {check['equal_to_e21_record']} of {check['runs']} regenerated runs equal their E21 record")
    start, moved = [], []
    for i in INSTANCES:
        s = exact(i, dict(zip(problem(i)["names"], initial_angles())))["approx_ratio"]
        t = [exact(i, b["values"])["approx_ratio"] for b in e22.bindings_for(problem(i)["ansatz"], runs, N, i)[:e22.SEEDS]]
        start.append(s)
        moved.append(float(np.mean(t)))
        print(f"  instance {i}: S {s:+.4f}; T mean {np.mean(t):.4f}, range {min(t):.4f} to {max(t):.4f}")
    print(f"  over the instances: S mean {np.mean(start):+.4f}, range {min(start):+.4f} to {max(start):+.4f}; T mean "
          f"{np.mean(moved):.4f}, range {min(moved):.4f} to {max(moved):.4f}")
    print(f"Primary test: two-sided exact Wilcoxon signed-rank over {len(INSTANCES)} instances, alpha {ALPHA}. p by the ranks "
          "of the differences of the minority sign:")
    for minority in ([], [1], [2], [3], [1, 2], [4], [1, 3]):
        d = [-k if k in minority else k for k in range(1, len(INSTANCES) + 1)]
        p = float(wilcoxon(d, alternative="two-sided", method="exact").pvalue)
        print(f"    ranks {minority or 'none (all ten differences of one sign)'}: p = {p:.5f} {'<=' if p <= ALPHA else '>'} {ALPHA}")
    print("  so p <= 0.01 needs a rank sum of the minority sign of at most 3; the smallest p is 2 / 1024 = 0.00195")


# --------------------------------------------------------------------------- --dry-run


class AerSteps:
    """Stand-in for the estimator jobs: 1 000 Aer shots of the logical circuit per point, the mean cost of the samples.

    last: the training ends after this step (outcome (b) of the amendment); None: no such end.
    """

    def __init__(self, noise, seed: int, records: list | None = None, last: int | None = None):
        self.noise, self.seed, self.records, self.submitted, self.last = noise, seed, list(records or []), [], last

    def allow(self, step: int) -> bool:
        return self.last is None or step <= self.last

    def submit(self, step: int, attempt: int, instances: list, points: list) -> None:
        self.records.append(new_step(step, attempt, instances, points))
        self.submitted.append(step)

    def wait(self, record: dict) -> None:
        pubs = [(problem(i)["logical"], [dict(zip(problem(i)["names"], x))], ESTIMATOR_SHOTS)
                for i, x in zip(record["instances"], record["points"])]
        stats = [cost_stats(problem(i)["cost"], c[0])
                 for i, c in zip(record["instances"], hj.run_aer(pubs, self.seed + record["step"], self.noise))]
        record.update(state="DONE", values=[s[0] for s in stats], stds=[s[1] for s in stats])


def dry_run(args) -> None:
    started = time.time()
    instances = INSTANCES[:QUICK["instances"]] if args.quick else INSTANCES
    maxiter, shots = (QUICK["maxiter"], QUICK["shots"]) if args.quick else (MAXITER, EVAL_SHOTS)
    arms = ARMS_SIX if args.six else ARMS
    runs, check = transferred(instances)
    stand_in = AerSteps(args.noise, 2850)
    lockstep(stand_in, [PROBE_INSTANCE], PROBE_MAXITER, None)
    probe_like = replay(histories(stand_in.records, [PROBE_INSTANCE])[PROBE_INSTANCE], PROBE_MAXITER)
    steps = AerSteps(args.noise, 2800, last=SIX if args.six else None)
    lockstep(steps, instances, maxiter, maxiter)
    training = training_outcome(steps.records, instances, None if args.six else maxiter)
    bindings = {i: bindings_of(i, training[str(i)], runs, arms) for i in instances}
    circuits = {i: problem(i)["logical"] for i in instances}
    pubs = evaluation_pubs(instances, circuits, bindings, SimpleNamespace(num_qubits=N), shots, CAL_SHOTS)
    result = analyse_counts(instances, bindings, hj.run_aer(pubs, 28, args.noise), training)
    execution = f"Aer on the logical circuits, two-qubit depolarising error {args.noise or 0}"
    report(f"dry run ({execution}; {len(instances)} instances, at most {SIX if args.six else maxiter} values, {shots:,} shots per "
           f"binding; outcome {'(b), set by --six' if args.six else '(a)'})", result)
    tail = ("_quick" if args.quick else "") + "_dryrun"
    name = hj.dry_run_name(STEM, args)
    hj.save_result(name[:-len(tail)] + "_six" + tail if args.six else name, {
        "design": DESIGN, "execution": execution,
        "note": "dry run: it validates the pipeline and the analysis and carries no information about hardware",
        "outcome": "six" if args.six else "twelve", "outcome_set_by": "--six" if args.six else "no --six",
        "instances": instances, "maxiter": maxiter, "shots": shots, "calibration_shots": CAL_SHOTS, "arm_names": arms,
        "estimator": f"stand-in: {ESTIMATOR_SHOTS} Aer shots and the mean cost of the samples, no mitigation",
        "probe": {"label": "stand-in, not the product function", "instance": PROBE_INSTANCE, "maxiter": PROBE_MAXITER,
                  "steps": len(stand_in.records), **probe_like},
        "training_steps": steps.records, "regenerated_runs": check, "product_settings": product_settings(),
        "circuits": [{"instance": i, "two_qubit_gates": e14.two_qubit_gates(circuits[i]), "depth": circuits[i].depth()}
                     for i in instances],
        **result, "wall_s": round(time.time() - started, 1)})


# --------------------------------------------------------------------------- the device: probe


class _ProductLog(logging.Handler):
    """Keeps the device that the product's hardware path logs as chosen."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.device = None

    def emit(self, record) -> None:
        if "Using backend" in str(record.msg) and isinstance(record.args, tuple) and record.args:
            self.device = str(getattr(record.args[0], "name", record.args[0]))


def scrub(text: str) -> str:
    return text.replace(os.environ.get("IBM_QUANTUM_TOKEN") or chr(0), "<token>")


def plain(value):
    """A returned object as JSON data."""
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def usage_of(job) -> float | None:
    try:
        return float(job.usage() or 0) or None
    except Exception:  # noqa: BLE001 - bookkeeping only
        return None


def raw_result(result) -> list:
    """Per pub: an estimator's evs and stds as lists, or a sampler's counts in the layout of hj.counts_of."""
    out = []
    for pub in result:
        if hasattr(pub.data, "evs"):
            out.append({"evs": np.atleast_1d(np.asarray(pub.data.evs, float)).ravel().tolist(),
                        "stds": np.atleast_1d(np.asarray(pub.data.stds, float)).ravel().tolist()})
        else:
            out.append({"counts": hj.counts_of([pub])[0]})
    return out


def archive_name(entry: dict) -> str:
    return f"{EXPERIMENT}_{entry['tag']}_{entry['job_id']}.json.gz"


def probe(args) -> None:
    require_product_settings()
    manifest = hj.load_manifest(EXPERIMENT)
    if manifest.get("probes") and not args.again:
        raise SystemExit(f"not run: the manifest already has a probe ({manifest['probes'][-1]['state']}, started "
                         f"{manifest['probes'][-1]['started_utc']}). A second probe needs --again; every probe stays listed")
    e20.require_committed_code()
    e14.load_token()
    if not os.environ.get("IBM_QUANTUM_TOKEN"):
        raise SystemExit("not run: no IBM token in the environment or the repository's .env")
    opened = hj.service()
    p = problem(PROBE_INSTANCE)
    record = {"kind": "probe", "state": "STARTED", "started_utc": utc(), "instance": PROBE_INSTANCE,
              "max_iterations": PROBE_MAXITER, "provenance": provenance(), "code_sha256": hj.code_hash(__file__)}
    manifest.setdefault("probes", []).append(record)
    hj.store_manifest(manifest)
    since = datetime.now(timezone.utc) - timedelta(seconds=ACCOUNT_MARGIN_S)
    seen, level = _ProductLog(), qaoa.logger.level
    logging.disable(logging.NOTSET)                  # e22's analysis disables logging in its process
    qaoa.logger.addHandler(seen)
    qaoa.logger.setLevel(logging.INFO)
    returned, failure = None, None
    try:
        returned = qaoa.quantum_optimize(p["prices"], risk_aversion=RISK_AVERSION, qc_simulator_mode=False,
                                         max_iterations=PROBE_MAXITER, reps=REPS)
    except Exception as exc:  # noqa: BLE001 - the failure of the product's path is the finding
        failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:2000],
                   "traceback": scrub("".join(traceback.format_tb(exc.__traceback__)[-6:]))}
    finally:
        qaoa.logger.removeHandler(seen)
        qaoa.logger.setLevel(level)
    record["ended_utc"] = utc()
    vendor = (returned or {}).get("vendor_runs") or []
    jobs = {run["job_id"]: {"job_id": run["job_id"], "sources": ["vendor_runs"], "stage": run.get("stage"),
                            "backend": run.get("backend")} for run in vendor}
    try:
        for job in opened.jobs(limit=50, created_after=since):
            try:
                name = str(job.backend().name)
            except Exception:  # noqa: BLE001 - bookkeeping only
                name = None
            jobs.setdefault(job.job_id(), {"job_id": job.job_id(), "sources": [], "stage": None, "backend": name})[
                "sources"].append("account")
        record["account_jobs"] = f"read: jobs created since {utc(since.timestamp())}"
    except Exception as exc:  # noqa: BLE001 - the probe's record must not be lost
        record["account_jobs"] = f"not read: {hj._reason(exc)}"
    device = seen.device or next((job["backend"] for job in jobs.values() if job["backend"]), None)
    solution = (returned or {}).get("solution")
    record.update(state="FAILED" if failure else "DONE", device=device, failure=failure, returned=plain(returned),
                  jobs=list(jobs.values()),
                  selection_is_exact_optimum=None if solution is None else bool(e19.score(p["cost"], e19.index_of(solution))["optimal"]))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch(record["started_utc"])))
    for k, job in enumerate(jobs.values()):
        manifest["jobs"].append({"kind": "probe", "job_id": job["job_id"], "device": job["backend"] or device,
                                 "state": "SUBMITTED", "usage_s": None, "estimate": None, "submitted_utc": record["started_utc"],
                                 "tag": f"{job['backend'] or device}_{stamp}_p{k:02d}", "sources": job["sources"],
                                 "stage": job["stage"]})
    hj.store_manifest(manifest)
    print(f"probe {record['state']} on {device}: {len(jobs)} jobs listed ({len(vendor)} in vendor_runs); "
          + (f"{failure['type']}: {failure['message'][:300]}" if failure else
             f"{len(returned.get('optimization_history') or [])} objective values, selection is an exact optimum: "
             f"{record['selection_is_exact_optimum']}") + ". Next: --collect", flush=True)


def probe_device(manifest: dict, without_measurements: bool) -> str:
    """The device of the last probe; stops unless phases 1 and 2 may run on it."""
    probes = manifest.get("probes") or []
    if not probes or probes[-1]["state"] == "STARTED":
        raise SystemExit("refused: the manifest has no finished probe. Run --probe first")
    last = probes[-1]
    if last["state"] == "FAILED" and not without_measurements:
        raise SystemExit(f"refused: the probe FAILED ({last['failure']['type']}: {last['failure']['message'][:300]}). If the "
                         "service refused a circuit with measurements in an Estimator pub, phase 1 runs with --train "
                         "--without-measurements. Any other failure: the plan is revised in a new commit")
    if last["state"] == "DONE" and without_measurements:
        raise SystemExit("refused: --without-measurements is accepted only when the last probe is FAILED")
    if not last.get("device"):
        raise SystemExit("refused: the probe names no device")
    named = os.environ.get("QFPO_IBM_BACKEND")
    if named and named != last["device"]:
        raise SystemExit(f"refused: QFPO_IBM_BACKEND is {named}, the device of the probe is {last['device']}")
    return last["device"]


def open_device(name: str) -> tuple:
    opened = hj.service()
    try:
        backend = opened.backend(name)
    except Exception as exc:  # noqa: BLE001 - unknown device, or the service refuses
        raise SystemExit(f"device {name} could not be opened ({hj._reason(exc)}); nothing was sent or written") from None
    if backend.name != name:
        raise SystemExit(f"refused: asked for {name}, the service returned {backend.name}")
    return opened, backend


# --------------------------------------------------------------------------- the device: training


def device_circuits(manifest: dict, backend) -> dict:
    """The ten transpiled circuits, read back from the QPY file that is written once when phase 1 starts."""
    from qiskit import qpy

    path = hj.OUT / "raw" / f"{EXPERIMENT}_{backend.name}_training_circuits.qpy.gz"
    if "circuits" not in manifest:
        made = [transpile_once(problem(i)["ansatz"], backend) for i in INSTANCES]
        buffer = io.BytesIO()
        qpy.dump(made, buffer)
        try:
            hj.write_once(path, buffer.getvalue())
        except FileExistsError:
            raise SystemExit(f"refused: {path} exists but the manifest does not list it; nothing was sent") from None
        manifest["circuits"] = {"device": backend.name, "file": path.name, "written_utc": utc(), "transpiler_seed": TRANSPILER_SEED,
                                "facts": [{"instance": i, **hj.circuit_facts(backend, isa)} for i, isa in zip(INSTANCES, made)]}
        hj.store_manifest(manifest)
    if manifest["circuits"]["device"] != backend.name:
        raise SystemExit(f"refused: the circuits were made for {manifest['circuits']['device']}, the device is {backend.name}")
    import gzip

    with gzip.open(path, "rb") as fh:
        loaded = qpy.load(fh)
    circuits = dict(zip(INSTANCES, loaded))
    for fact in manifest["circuits"]["facts"]:
        isa = circuits[fact["instance"]]
        if isa.layout is None or e14.measured_physical(isa) != fact["physical_qubits"] or e14.two_qubit_gates(isa) != fact["two_qubit_gates"]:
            raise SystemExit(f"refused: the circuit of instance {fact['instance']} read from {path.name} is not the one recorded")
    return circuits


class DeviceSteps:
    """The estimator jobs of phase 1, recorded in the manifest."""

    def __init__(self, manifest: dict, opened, backend, circuits: dict, without_measurements: bool):
        self.manifest, self.opened, self.backend = manifest, opened, backend
        self.observables = {i: problem(i)["hamiltonian"].apply_layout(circuits[i].layout) for i in INSTANCES}
        self.circuits = {i: c.remove_final_measurements(inplace=False) if without_measurements else c for i, c in circuits.items()}

    @property
    def records(self) -> list:
        return [entry for entry in self.manifest["jobs"] if entry.get("kind") == "train"]

    def allow(self, step: int) -> bool:
        """The amended budget rule. The decision is taken once, from the QPU seconds of step 1, and stored; later it is read.

        False: the training ends before this step (outcome (b): after the sixth value of every instance).
        """
        if step == 1:
            return True
        training = self.manifest["training"]
        if not training.get("decision"):
            first = next(r for r in self.records if r["step"] == 1 and r["values"] is not None)
            if not first["usage_s"]:
                first["usage_s"] = usage_of(self.opened.job(first["job_id"]))
                hj.store_manifest(self.manifest)
            if not first["usage_s"]:
                raise SystemExit(f"stopped: step 1 (job {first['job_id']}) reports no QPU seconds yet, so the budget rule cannot be "
                                 "checked; nothing further is submitted. Start --train again later")
            training["decision"] = decide(first["usage_s"])
            hj.store_manifest(self.manifest)
            print(f"decision of the budget rule: {decision_text(training['decision'])}", flush=True)
        if training["decision"]["outcome"] == "stop":
            raise SystemExit(f"stopped by the budget rule: {decision_text(training['decision'])}")
        return step <= training["decision"]["values_per_instance"]

    def submit(self, step: int, attempt: int, instances: list, points: list) -> None:
        from qiskit_ibm_runtime import EstimatorV2

        pubs = [(self.circuits[i], self.observables[i], [x]) for i, x in zip(instances, points)]
        # everything that can fail is computed before the job exists; afterwards only its identifier is added
        entry = {**new_step(step, attempt, instances, points), "device": self.backend.name, "estimate": None,
                 "default_shots": ESTIMATOR_SHOTS, "max_execution_s": MAX_JOB_SECONDS, "provenance": provenance(),
                 "code_sha256": hj.code_hash(__file__)}
        json.dumps(entry)
        estimator = EstimatorV2(mode=self.backend)
        estimator.options.default_shots = ESTIMATOR_SHOTS
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"
        estimator.options.max_execution_time = MAX_JOB_SECONDS
        job = estimator.run(pubs)
        now = time.gmtime()
        entry = {"job_id": job.job_id(), "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
                 "tag": f"{self.backend.name}_{time.strftime('%Y%m%dT%H%M%SZ', now)}_s{step:02d}a{attempt}", **entry}
        self.manifest["jobs"].append(entry)
        hj.store_manifest(self.manifest)
        print(f"step {step} (attempt {attempt}): job {entry['job_id']}, {len(pubs)} pubs", flush=True)

    def wait(self, record: dict) -> None:
        job = self.opened.job(record["job_id"])
        if hasattr(job, "wait_for_final_state"):
            job.wait_for_final_state()
        record["state"] = hj.state_of(job)
        if record["state"] in ("ERROR", "CANCELLED"):
            hj.store_manifest(self.manifest)
            print(f"step {record['step']} (attempt {record['attempt']}): job {record['job_id']} is {record['state']}", flush=True)
            return
        if record["state"] != "DONE":
            raise SystemExit(f"stopped: job {record['job_id']} is {record['state']} after the wait; start --train again")
        result = job.result()
        raw = raw_result(result)
        if len(raw) != len(record["instances"]) or any(len(pub.get("evs", [])) != 1 for pub in raw):
            raise SystemExit(f"stopped: job {record['job_id']} did not return one value per pub")
        stamps, path = hj.job_times(job, result), hj.OUT / "raw" / archive_name(record)
        usage = usage_of(job)
        if not path.exists():
            hj.write_archive(path, {"experiment": EXPERIMENT, "kind": "train", "step": record["step"], "job_id": record["job_id"],
                                    "device": record["device"], "submitted_utc": record["submitted_utc"], "usage_s": usage,
                                    "timestamps": stamps, "instances": record["instances"], "points": record["points"],
                                    "result": raw})
        record.update(values=[pub["evs"][0] for pub in raw], stds=[pub["stds"][0] for pub in raw], usage_s=usage,
                      timestamps=stamps, archive=path.name)
        hj.store_manifest(self.manifest)
        print(f"step {record['step']}: values {[round(v, 5) for v in record['values']]}; QPU seconds "
              f"{usage or 'not reported yet'}", flush=True)


def train(args) -> None:
    require_product_settings()
    manifest = hj.load_manifest(EXPERIMENT)
    name = probe_device(manifest, args.without_measurements)
    training = manifest.get("training")
    if training and (training["without_measurements"] != bool(args.without_measurements) or training["device"] != name):
        raise SystemExit(f"refused: the training in the manifest was started on {training['device']} with without_measurements "
                         f"{training['without_measurements']}; it goes on only in that form")
    e20.require_committed_code()
    opened, backend = open_device(name)
    circuits = device_circuits(manifest, backend)
    if not training:
        failure = manifest["probes"][-1]["failure"]
        manifest["training"] = {"device": name, "started_utc": utc(), "maxiter": MAXITER,
                                "without_measurements": bool(args.without_measurements),
                                "reason": f"{failure['type']}: {failure['message']}" if args.without_measurements else None}
        hj.store_manifest(manifest)
    steps = DeviceSteps(manifest, opened, backend, circuits, bool(args.without_measurements))
    lockstep(steps, INSTANCES, MAXITER, MAXITER)
    decision = manifest["training"]["decision"]
    outcome = training_outcome(steps.records, INSTANCES, None if decision["outcome"] == "six" else MAXITER)
    used = [r["usage_s"] for r in steps.records if r["values"] is not None]
    print(f"training complete on {name}: {len(used)} steps with values, QPU seconds {sum(u or 0 for u in used):.1f} "
          f"({sum(u is None for u in used)} steps not reported); values per instance "
          f"{[len(outcome[str(i)]['values']) for i in INSTANCES]}; D5 used {[outcome[str(i)]['replay_D5']['values_used'] for i in INSTANCES]} "
          f"values; {decision_text(decision)}. Next: --submit", flush=True)


# --------------------------------------------------------------------------- the device: evaluation, collection, analysis


def train_records(manifest: dict) -> list:
    return [entry for entry in manifest["jobs"] if entry.get("kind") == "train"]


def submit(args) -> None:
    require_product_settings()
    manifest = hj.load_manifest(EXPERIMENT)
    training = manifest.get("training")
    name = probe_device(manifest, bool(training and training["without_measurements"]))
    if not training or training["device"] != name:
        raise SystemExit(f"not submitted: no training on {name} is listed. Run --train first")
    if any(r["values"] is None and r["state"] not in ("ERROR", "CANCELLED") for r in train_records(manifest)):
        raise SystemExit("not submitted: a training step is listed without values. Run --train to finish phase 1")
    decision = training.get("decision")
    if not decision:
        raise SystemExit("not submitted: the training has no decision of the budget rule yet. Run --train")
    if decision["outcome"] == "stop":
        raise SystemExit(f"not submitted: {decision_text(decision)}")
    arms = ARMS_SIX if decision["outcome"] == "six" else ARMS
    try:
        outcome = training_outcome(train_records(manifest), INSTANCES, None if decision["outcome"] == "six" else MAXITER)
    except SystemExit as stop:
        raise SystemExit(f"not submitted: {stop}. Run --train to finish phase 1") from None
    hj.refuse_repeat(EXPERIMENT, name, False, "evaluation")
    estimate = evaluation_estimate(arms)
    if estimate["seconds"] > MAX_EVALUATION_S:
        raise SystemExit(f"not submitted: {estimate['seconds']:.0f} QPU seconds estimated, above {MAX_EVALUATION_S:.0f} s")
    e20.require_committed_code()
    runs, check = transferred(INSTANCES)
    opened, backend = open_device(name)
    circuits = device_circuits(manifest, backend)
    bindings = {i: bindings_of(i, outcome[str(i)], runs, arms) for i in INSTANCES}
    pubs = evaluation_pubs(INSTANCES, circuits, bindings, backend, EVAL_SHOTS, CAL_SHOTS)
    facts = {"kind": "evaluation", "instances": INSTANCES, "arms": arms, "shots": EVAL_SHOTS, "calibration_shots": CAL_SHOTS,
             "bindings": {str(i): bindings[i] for i in INSTANCES}, "regenerated_runs": check,
             "circuits_file": manifest["circuits"]["file"], "without_measurements_in_training": training["without_measurements"]}
    print(f"decision of the budget rule: {decision_text(decision)}", flush=True)
    hj.submit(EXPERIMENT, __file__, opened, backend, pubs, MAX_JOB_SECONDS, facts)


def collect() -> None:
    manifest = hj.load_manifest(EXPERIMENT)
    if not manifest["jobs"]:
        raise SystemExit("E28: no job is listed")
    opened = hj.service()
    for entry in manifest["jobs"]:
        if entry.get("kind") != "probe" or entry.get("archive"):
            continue
        job = opened.job(entry["job_id"])
        entry["state"] = hj.state_of(job)
        if entry["state"] == "DONE":
            result = job.result()
            entry.update(usage_s=usage_of(job), timestamps=hj.job_times(job, result))
            path = hj.write_archive(hj.OUT / "raw" / archive_name(entry), {
                "experiment": EXPERIMENT, "kind": "probe", "job_id": entry["job_id"], "device": entry["device"],
                "sources": entry["sources"], "stage": entry["stage"], "usage_s": entry["usage_s"],
                "timestamps": entry["timestamps"], "result": raw_result(result)})
            entry["archive"] = path.name
            print(f"archived probe job {entry['job_id']} as {path.name}; QPU seconds {entry['usage_s'] or 'not reported yet'}",
                  flush=True)
        hj.store_manifest(manifest)
    if any(r["values"] is None and r["state"] not in ("ERROR", "CANCELLED") for r in train_records(manifest)):
        print("a training step is listed without values: --train waits for it; nothing else is collected", flush=True)
        return
    hj.collect(EXPERIMENT, opened)
    analyse()


def job_table(manifest: dict) -> dict:
    """QPU seconds per kind of job and the times of every job."""
    jobs = [{key: entry.get(key) for key in ("kind", "step", "attempt", "job_id", "device", "state", "submitted_utc", "usage_s",
                                             "timestamps", "archive", "sources", "stage")} for entry in manifest["jobs"]]
    seconds = {kind: {"reported_s": float(sum(j["usage_s"] or 0 for j in jobs if j["kind"] == kind)),
                      "jobs": sum(j["kind"] == kind for j in jobs),
                      "jobs_without_report": sum(j["kind"] == kind and not j["usage_s"] for j in jobs)}
               for kind in ("probe", "train", "evaluation")}
    return {"jobs": jobs, "qpu_seconds": seconds}


def analyse() -> None:
    todo = [(entry, record) for entry, record in hj.to_analyse(EXPERIMENT, STEM) if entry.get("kind") == "evaluation"]
    if not todo:
        print("E28: no archived evaluation job is without a result")
        return
    e20.require_committed_code()
    manifest = hj.load_manifest(EXPERIMENT)
    decision = manifest["training"]["decision"]
    arms = ARMS_SIX if decision["outcome"] == "six" else ARMS
    for entry, record in todo:
        started = time.time()
        instances = entry["instances"]
        if entry["arms"] != arms:
            raise SystemExit(f"job {entry['job_id']} not analysed: its arms {entry['arms']} are not those of the stored decision")
        requested = [[entry["shots"]] * len(entry["arms"])] * len(instances) + [[entry["calibration_shots"]]] * (2 * len(instances))
        if record["shots"] != requested:
            raise SystemExit(f"job {entry['job_id']} not analysed: shots per pub {record['shots']}, requested {requested}")
        training = training_outcome(train_records(manifest), instances, None if decision["outcome"] == "six" else MAXITER)
        bindings = {i: entry["bindings"][str(i)] for i in instances}
        for i in instances:
            for arm, b in zip([a for a in ("S", "D5", "D12") if a in arms], bindings[i]):
                if b["arm"] != arm or list(b["values"].values()) != training[str(i)][arm]:
                    raise SystemExit(f"job {entry['job_id']} not analysed: the angles {arm} of instance {i} are not those of the training")
        result = analyse_counts(instances, bindings, record["counts"], training)
        report(f"{entry['device']}, evaluation job {entry['job_id']}", result)
        print(f"  decision of the budget rule: {decision_text(decision)}", flush=True)
        table, first = job_table(manifest), min(r["submitted_utc"] for r in train_records(manifest))
        probes = [{key: p.get(key) for key in ("state", "started_utc", "ended_utc", "device", "failure", "returned", "jobs",
                                               "selection_is_exact_optimum", "account_jobs")} for p in manifest.get("probes", [])]
        hj.save_result(f"{STEM}_{entry['tag']}", {
            "design": DESIGN, "execution": entry["device"], "instances": instances, "maxiter": MAXITER, "shots": entry["shots"],
            "calibration_shots": entry["calibration_shots"], "arm_names": entry["arms"], "decision": decision,
            "training_record": manifest["training"], "probes": probes, "circuits": manifest["circuits"],
            "minutes_from_first_training_step_to_evaluation": round((epoch(entry["submitted_utc"]) - epoch(first)) / 60, 1),
            **table, "regenerated_runs": entry["regenerated_runs"], "product_settings": product_settings(), **result,
            **hj.job_summary(EXPERIMENT, entry, __file__), "wall_s": round(time.time() - started, 1)})


def status() -> None:
    manifest = hj.load_manifest(EXPERIMENT)
    opened = hj.service()
    try:
        print("account usage:", json.dumps(opened.usage(), default=str)[:600])
    except Exception as exc:  # noqa: BLE001 - bookkeeping only
        print("account usage unavailable:", type(exc).__name__, str(exc)[:120])
    hj.refresh(manifest, opened)
    for p in manifest.get("probes", []):
        print(f"probe started {p['started_utc']}: {p['state']}, device {p.get('device')}, {len(p.get('jobs', []))} jobs")
    if manifest.get("training"):
        decision = manifest["training"].get("decision")
        print("decision of the budget rule: " + (decision_text(decision) if decision else "not taken yet (--train takes it after step 1)"))
    for entry in manifest["jobs"]:
        print(f"{entry['kind']} job {entry['job_id']} on {entry['device']}"
              + (f", step {entry['step']} attempt {entry['attempt']}" if entry["kind"] == "train" else "")
              + f", submitted {entry['submitted_utc']}: {entry['state']}, QPU seconds {entry['usage_s'] or 'not reported'}, "
              f"archive {entry.get('archive', 'none')}")
    if not manifest["jobs"]:
        print("no job of e28 is listed")


def main(argv=None) -> None:
    p = hj.parser(__doc__, {"probe": "phase 0: the product's hardware path as shipped, once",
                            "train": "phase 1: training on the device of the probe; resumable"})
    p.add_argument("--without-measurements", action="store_true",
                   help="with --train: estimator circuits without their final measurements; only after a FAILED probe")
    p.add_argument("--six", action="store_true",
                   help="with --dry-run: outcome (b) of the amendment: six values per instance, nine bindings, no D12")
    for action in p._actions:                            # hj.parser words --again for --submit
        if action.dest == "again":
            action.help = "with --probe: a second probe; every probe stays listed"
    args = p.parse_args(argv)
    if (args.noise is not None or args.quick) and not args.dry_run:
        p.error("--noise and --quick belong to --dry-run")
    if args.six and not args.dry_run:
        p.error("--six belongs to --dry-run")
    if args.again and not args.probe:
        p.error("--again belongs to --probe: a second probe")
    if args.without_measurements and not args.train:
        p.error("--without-measurements belongs to --train")
    if args.plan:
        plan()
    elif args.dry_run:
        dry_run(args)
    elif args.status:
        status()
    elif args.probe:
        probe(args)
    elif args.train:
        train(args)
    elif args.submit:
        submit(args)
    elif args.collect:
        collect()
    else:
        analyse()


if __name__ == "__main__":
    main()
