"""Shared parts of the IBM hardware experiments E26 to E30.

Manifest. results/<experiment>_submission.json lists every submitted job: identifier, device, UTC
time, state, QPU seconds. submit() refuses while a listed job is not finished. collect() archives the
counts of finished jobs. QPU seconds are read only from a finished job, and a value of 0 is taken as
"not reported yet": E14 and E15 read the usage as their jobs returned and stored 0.0.
collect() also stores the job's time stamps (job_times) in the manifest entry and in the archive.

Files. A job gets the tag <device>_<UTC time of submission>. Its circuits go to
results/raw/<experiment>_<tag>_circuits.qpy.gz, its raw counts to
results/raw/<experiment>_<tag>_<job>.json.gz before any analysis, its result to
results/<script>_<tag>.json. None of these is ever replaced; only a dry-run result (name ending in
_dryrun) is.

QPU-seconds estimate (qpu_seconds). seconds = 0.317 ms x shots, where shots is the sum over the pubs
of a job of bindings x shots. The factor is a least-squares fit through the origin to the six jobs of
E22 on ibm_marrakesh, the only jobs with a usage read after they had finished: 140 000 shots used 44,
44, 45 and 45 s, 350 000 shots used 111 and 111 s, with 213, 465 and 827 two-qubit gates per circuit.
Largest residual 0.6 s; the records are whole seconds. What the fit cannot show: each of the six jobs
has 5 pubs of 7 bindings, so a cost per pub is not identified (a straight line through the two shot
totals has an intercept of 0.2 s). Check on jobs outside the fit: the five jobs of hardware round 2
have 2 to 41 pubs and 392 000 shots in their archived counts, and the paper records 124 s for them;
the estimate gives 124.4 s. All records are from one device, whose packaged model has a default delay
between shots of 0.250 ms; on a device with another delay the factor is an assumption. Results state
estimated and recorded seconds side by side.

Detectable difference (fisher_power, detectable). The probability, for two independent binomial
samples, that the one-sided Fisher exact test gives p <= alpha, computed exactly; see fisher_power.

The IBM token is read by e14.ibm_service() only (environment or the repository's .env). Nothing in
this module prints or stores it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import RESULTS, point_seed, pool_workers, provenance  # noqa: E402
from scipy.stats import binom, hypergeom  # noqa: E402


def _import_script(name: str):
    """Import an earlier experiment script. It reads sys.argv when imported, so it gets an empty command line."""
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        return importlib.import_module(name)
    finally:
        sys.argv = argv


e14 = _import_script("e14_ibm_hardware_controls")       # token, gate counts, readout calibration, round 2A analysis
e20 = _import_script("e20_portfolio_corrected")         # require_committed_code
state_of = _import_script("e22_hardware_portfolio_qaoa").state_of

OUT = RESULTS                       # manifests, archives and results are written below this directory
FINAL = ("DONE", "ERROR", "CANCELLED")
AER_CHUNK = 2_000                   # shots per noisy Aer task
POWER = 0.8
ROUND_2_SECONDS = 124.0             # main.tex: "Round 2 (five jobs, 124 s)"
POWER_METHOD = ("exact power of the one-sided Fisher exact test for two independent binomial samples; no normal "
                "approximation; counts further than 8 standard deviations from their mean are left out; the difference "
                "is located by bisection")


# --------------------------------------------------------------------------- command line


def parser(doc: str, modes: dict | None = None) -> argparse.ArgumentParser:
    """The command line of a hardware script. modes: further exclusive actions, {flag name: help}."""
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0],
                                epilog="The module docstring states the design and the analysis plan.")
    standard = {"plan": "offline: jobs, pubs, shots, gate counts, QPU-seconds estimate, detectable differences",
                "dry-run": "Aer: the whole pipeline with its analysis; the result name ends in _dryrun",
                "status": "device, account usage, state and QPU seconds of the submitted jobs",
                "submit": "submit the job and record it in the manifest",
                "collect": "archive the counts of finished jobs, then analyse them",
                "analyse": "analyse archived counts that have no result yet"}
    group = p.add_mutually_exclusive_group(required=True)
    for name, text in {**standard, **(modes or {})}.items():
        group.add_argument(f"--{name}", action="store_true", help=text)
    p.add_argument("--noise", type=float, metavar="P", help="with --dry-run: error level of the simulation")
    p.add_argument("--quick", action="store_true", help="with --dry-run: the faster variant named in the module docstring")
    p.add_argument("--again", action="store_true",
                   help="with --submit: a further job on a device that already has a finished job of this experiment")
    return p


def parse(p: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    args = p.parse_args(argv)
    if (args.noise is not None or args.quick) and not args.dry_run:
        p.error("--noise and --quick belong to --dry-run")
    if args.again and not args.submit:
        p.error("--again belongs to --submit")
    return args


def dry_run_name(stem: str, args) -> str:
    return stem + (f"_noise_{args.noise}" if args.noise else "") + ("_quick" if args.quick else "") + "_dryrun"


# --------------------------------------------------------------------------- device and transpilation


def _reason(exc: Exception) -> str:
    """Type and first 200 characters of an error, with the token removed should it occur in the text."""
    return f"{type(exc).__name__}: {str(exc).replace(os.environ.get('IBM_QUANTUM_TOKEN') or chr(0), '<token>')[:200]}"


def service():
    """The IBM service, opened with the token that e14 loads. The only place that contacts IBM."""
    try:
        return e14.ibm_service()
    except Exception as exc:  # noqa: BLE001 - no token, no network, or the service refuses
        raise SystemExit(f"the IBM service could not be opened ({_reason(exc)}); nothing was sent or written") from None


def device() -> tuple:
    """The service and the device named by QFPO_IBM_BACKEND; there is no default device."""
    name = os.environ.get("QFPO_IBM_BACKEND")
    if not name:
        raise SystemExit("set QFPO_IBM_BACKEND to the name of the device")
    opened = service()
    try:
        return opened, opened.backend(name)
    except Exception as exc:  # noqa: BLE001 - unknown device, or the service refuses
        raise SystemExit(f"device {name} could not be opened ({_reason(exc)}); nothing was sent or written") from None


def device_models() -> list:
    """Packaged models of two 156-qubit Heron devices, for --plan. They need no network."""
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh

    return [FakeFez(), FakeMarrakesh()]


def transpile_best(circuit, seeds, **target) -> tuple:
    """Optimisation level 3 once per seed. Returns (seed, circuit) with the fewest two-qubit gates, the first of equals.

    target: backend=..., or coupling_map=... and basis_gates=... for a transpilation that sees no error data.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    best = None
    for seed in seeds:
        isa = generate_preset_pass_manager(optimization_level=3, seed_transpiler=seed, **target).run(circuit)
        if best is None or e14.two_qubit_gates(isa) < e14.two_qubit_gates(best[1]):
            best = (seed, isa)
    return best


def circuit_facts(backend, isa) -> dict:
    """Size of a transpiled circuit and the device's reported errors on the gates it uses."""
    facts = {"two_qubit_gates": e14.two_qubit_gates(isa), "depth": isa.depth(),
             "physical_qubits": e14.measured_physical(isa), **e14.gate_errors(backend, isa)}
    try:
        facts["calibration_date"] = str(backend.properties().last_update_date)
    except Exception as exc:  # noqa: BLE001 - bookkeeping only
        facts["calibration_date"] = f"unavailable: {type(exc).__name__}"
    return facts


def logical(circuit):
    """The circuit in the gate set of the dry runs (cx, rz, sx, x), prepared as e14 and e22 do."""
    from qiskit import transpile

    return transpile(circuit, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)


# --------------------------------------------------------------------------- manifest, archive, results


def load_manifest(experiment: str) -> dict:
    path = OUT / f"{experiment}_submission.json"
    return json.loads(path.read_text()) if path.exists() else {"experiment": experiment, "jobs": []}


def store_manifest(manifest: dict) -> None:
    (OUT / f"{manifest['experiment']}_submission.json").write_text(json.dumps(manifest, indent=1))


def refresh(manifest: dict, opened) -> list:
    """Read the state of unfinished jobs and the QPU seconds of finished ones. Returns the unfinished jobs."""
    for entry in manifest["jobs"]:
        if entry["state"] in FINAL and (entry["state"] != "DONE" or entry["usage_s"]):
            continue
        job = opened.job(entry["job_id"])
        entry["state"] = state_of(job)
        if entry["state"] == "DONE":
            entry["usage_s"] = float(job.usage() or 0) or None
    if manifest["jobs"]:
        store_manifest(manifest)
    return [entry for entry in manifest["jobs"] if entry["state"] not in FINAL]


def refuse_unfinished(experiment: str, opened) -> None:
    unfinished = refresh(load_manifest(experiment), opened)
    if unfinished:
        raise SystemExit(f"not submitted: job {unfinished[0]['job_id']} of {experiment} is {unfinished[0]['state']}")


def refuse_repeat(experiment: str, device: str, again: bool, kind: str | None = None) -> None:
    """A device that has a finished job of this experiment gets a further one only on request. Reads the manifest only.

    kind: count only finished jobs whose manifest entry has this "kind" (an experiment with several kinds of job).
    """
    done = [entry for entry in load_manifest(experiment)["jobs"] if entry["device"] == device and entry["state"] == "DONE"
            and (kind is None or entry.get("kind") == kind)]
    if done and not again:
        raise SystemExit(f"not submitted: job {done[-1]['job_id']} of {experiment} on {device} is finished. A further job "
                         "on this device needs --again; every job stays listed in every result")


def write_once(path: Path, data: bytes) -> Path:
    """A new gzip file. An existing file is never replaced: the call fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xb") as fh:
        fh.write(data)
    return path


def write_archive(path: Path, record: dict) -> Path:
    return write_once(path, json.dumps(record).encode())


def read_archive(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def code_hash(script: str) -> str:
    """SHA-256 of an experiment script and of this module."""
    return hashlib.sha256(Path(script).read_bytes() + Path(__file__).read_bytes()).hexdigest()


def counts_of(result) -> list:
    """Counts of a sampler result: per pub a list with one {bitstring: count} per binding."""
    out = []
    for pub in result:
        (bits,) = pub.data.values()
        index = range(bits.shape[0]) if bits.shape else [None]
        out.append([{str(k): int(v) for k, v in bits.get_counts(i).items()} for i in index])
    return out


def submit(experiment: str, script: str, opened, backend, pubs: list, max_seconds: int, facts: dict,
           again: bool = False) -> dict:
    """Submit one sampler job with the settings of the platform's hardware path and record it at once.

    pubs: (circuit, bindings, shots), bindings a list of {parameter name: value} or None.
    facts: what the script keeps with the job in the manifest; a "kind" in it limits the repeat guard to that kind.
    again: allow a further job on a device that already has a finished one (see refuse_repeat).
    """
    from qiskit import qpy
    from qiskit_ibm_runtime import SamplerV2

    refuse_unfinished(experiment, opened)
    refuse_repeat(experiment, backend.name, again, facts.get("kind"))
    # everything that can fail is computed before the job exists; afterwards only its identifier is added
    entry = {"device": backend.name, "state": "SUBMITTED", "usage_s": None, "estimate": qpu_seconds(pubs),
             "max_execution_s": max_seconds, "provenance": provenance(), "code_sha256": code_hash(script), **facts}
    json.dumps(entry)
    buffer = io.BytesIO()
    qpy.dump([circuit for circuit, _, _ in pubs], buffer)
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.max_execution_time = max_seconds
    job = sampler.run([(circuit, None if bindings is None else [[b[p.name] for p in circuit.parameters] for b in bindings],
                        shots) for circuit, bindings, shots in pubs])
    now = time.gmtime()
    entry = {"job_id": job.job_id(), "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
             "tag": f"{backend.name}_{time.strftime('%Y%m%dT%H%M%SZ', now)}", **entry}
    manifest = load_manifest(experiment)
    manifest["jobs"].append(entry)
    store_manifest(manifest)
    write_once(OUT / "raw" / f"{experiment}_{entry['tag']}_circuits.qpy.gz", buffer.getvalue())
    print(f"submitted {experiment} to {backend.name}: job {entry['job_id']}, {entry['estimate']['pubs']} pubs, "
          f"{entry['estimate']['shots']:,} shots, estimated {entry['estimate']['seconds']:.1f} QPU seconds", flush=True)
    return entry


def status(experiment: str, opened, backend) -> None:
    state = backend.status()
    print(f"{backend.name}: operational {state.operational}, pending jobs {state.pending_jobs}, {backend.num_qubits} qubits")
    try:
        print("account usage:", json.dumps(opened.usage(), default=str)[:600])
    except Exception as exc:  # noqa: BLE001 - bookkeeping only
        print("account usage unavailable:", type(exc).__name__, str(exc)[:120])
    manifest = load_manifest(experiment)
    refresh(manifest, opened)
    for entry in manifest["jobs"]:
        print(f"job {entry['job_id']} on {entry['device']}, submitted {entry['submitted_utc']}: {entry['state']}, "
              f"QPU seconds {entry['usage_s'] or 'not reported'} (estimate {entry['estimate']['seconds']:.1f}), "
              f"archive {entry.get('archive', 'none')}")
    if not manifest["jobs"]:
        print(f"no job of {experiment} has been submitted")


def job_times(job, result) -> dict | None:
    """When a job was created, started, finished and executed: ISO-8601 UTC strings ending in Z, None where unknown.

    result: the job's result. A time without a zone is taken as local time in the job's metrics (the local mode
    states it so) and as UTC in the execution spans. Returns None if nothing can be read.
    """
    from datetime import datetime, timezone

    def utc(value, zone=None):
        if value is None:
            return None
        if isinstance(value, str):
            value, zone = datetime.fromisoformat(value), timezone.utc
        if value.tzinfo is None:
            value = value.replace(tzinfo=zone) if zone else value.astimezone()
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    times = dict.fromkeys(("created", "running", "finished", "execution_start", "execution_stop"))
    try:
        stamps = job.metrics()["timestamps"]
        times.update({key: utc(stamps.get(key)) for key in ("created", "running", "finished")})
    except Exception:  # noqa: BLE001 - bookkeeping only
        pass
    try:
        spans = list(result.metadata["execution"]["execution_spans"])
        times.update({"execution_start": utc(min(span.start for span in spans), timezone.utc),
                      "execution_stop": utc(max(span.stop for span in spans), timezone.utc)})
    except Exception:  # noqa: BLE001 - bookkeeping only
        pass
    return times if any(times.values()) else None


def collect(experiment: str, opened) -> None:
    """Archive the counts and the times (job_times) of every finished job that has no archive yet. Nothing is analysed here."""
    manifest = load_manifest(experiment)
    refresh(manifest, opened)
    for entry in manifest["jobs"]:
        if entry.get("archive"):
            continue
        if entry["state"] != "DONE":
            print(f"job {entry['job_id']} is {entry['state']}: nothing collected", flush=True)
            continue
        job = opened.job(entry["job_id"])
        result = job.result()
        counts = counts_of(result)
        entry["usage_s"] = float(job.usage() or 0) or entry["usage_s"]
        entry["timestamps"] = job_times(job, result)
        path = write_archive(OUT / "raw" / f"{experiment}_{entry['tag']}_{entry['job_id']}.json.gz",
                             {"experiment": experiment, "job_id": entry["job_id"], "device": entry["device"],
                              "submitted_utc": entry["submitted_utc"], "usage_s": entry["usage_s"],
                              "timestamps": entry["timestamps"],
                              "shots": [[sum(binding.values()) for binding in pub] for pub in counts], "counts": counts})
        entry["archive"] = path.name
        store_manifest(manifest)
        print(f"archived job {entry['job_id']} as {path.name}; QPU seconds {entry['usage_s'] or 'not reported yet'}",
              flush=True)


def to_analyse(experiment: str, stem: str) -> list:
    """(manifest entry, archived record) of every archived job whose result file does not exist."""
    return [(entry, read_archive(OUT / "raw" / entry["archive"])) for entry in load_manifest(experiment)["jobs"]
            if entry.get("archive") and not (OUT / f"{stem}_{entry['tag']}.json").exists()]


def job_summary(experiment: str, entry: dict, script: str) -> dict:
    """What a result says about its job and about every submission of the experiment."""
    job = {key: entry.get(key) for key in ("job_id", "device", "submitted_utc", "usage_s", "estimate", "archive")}
    job["provenance_at_submission"] = entry["provenance"]
    job["analysis_code_changed_since_submission"] = entry["code_sha256"] != code_hash(script)
    return {"job": job, "submissions": [{key: e.get(key) for key in ("job_id", "device", "submitted_utc", "state")}
                                        for e in load_manifest(experiment)["jobs"]]}


def save_result(name: str, payload: dict) -> Path:
    """As common.save, below OUT. The result of a device job is never replaced; a dry-run result is."""
    path = OUT / f"{name}.json"
    if path.exists() and not name.endswith("_dryrun"):
        raise SystemExit(f"{path} exists and is not replaced")
    path.write_text(json.dumps({"provenance": provenance(), **payload}, indent=2, default=float))
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- QPU seconds


def usage_basis() -> dict:
    """The records behind the estimate: the six E22 jobs, the fit, and the check against hardware round 2."""
    stored = json.loads((RESULTS / "e22_hardware_portfolio_qaoa.json").read_text())
    jobs = []
    for job in stored["jobs"]:
        rows = [r for r in stored["rows"] if r["n"] == job["n"] and r["instance"] in job["instances"]]
        jobs.append({"job_id": job["job_id"], "pubs": len(job["instances"]), "bindings": len(rows),
                     "shots": sum(r["shots"] for r in rows), "seconds": job["usage_s"]})
    shots = np.array([job["shots"] for job in jobs], float)
    seconds = np.array([job["seconds"] for job in jobs], float)
    per_shot = float(shots @ seconds / (shots @ shots))
    round_2 = 0
    for path in sorted((RESULTS / "raw").glob("e1[45]-*.json.gz")):
        round_2 += sum(sum(binding.values()) for pub in read_archive(path)["counts"] for binding in pub)
    return {"device": stored["execution"], "jobs": jobs, "seconds_per_shot": per_shot,
            "largest_residual_s": float(np.abs(seconds - per_shot * shots).max()),
            "round_2": {"shots": round_2, "recorded_s": ROUND_2_SECONDS, "estimated_s": per_shot * round_2}}


def qpu_seconds(pubs: list, basis: dict | None = None) -> dict:
    """Estimated QPU seconds of one job: the fitted seconds per shot times its shots. pubs as for submit()."""
    basis = basis or usage_basis()
    shots = sum(len(bindings or [None]) * n for _, bindings, n in pubs)
    return {"pubs": len(pubs), "shots": shots, "seconds": basis["seconds_per_shot"] * shots,
            "seconds_per_shot": basis["seconds_per_shot"]}


def print_estimate(estimate: dict, basis: dict) -> None:
    print(f"QPU seconds: {estimate['seconds']:.1f} s estimated = {estimate['shots']:,} shots x "
          f"{1000 * basis['seconds_per_shot']:.3f} ms ({estimate['pubs']} pubs)")
    print(f"  basis: least squares through the origin over the {len(basis['jobs'])} jobs of E22 on {basis['device']}, "
          "usage read after completion:")
    for job in basis["jobs"]:
        print(f"    {job['job_id']}: {job['pubs']} pubs, {job['bindings']} bindings, {job['shots']:,} shots, "
              f"{job['seconds']:.0f} s")
    print(f"    largest residual {basis['largest_residual_s']:.1f} s; every job has 5 pubs, so a cost per pub is not identified")
    check = basis["round_2"]
    print(f"  check outside the fit: hardware round 2, five jobs, {check['shots']:,} shots: {check['recorded_s']:.0f} s "
          f"recorded for the round, {check['estimated_s']:.1f} s estimated")
    print(f"  all records are from {basis['device']}; on another device the seconds per shot are an assumption")


# --------------------------------------------------------------------------- statistics


def round_2a() -> dict:
    """Round 2A as stored (ibm_marrakesh): constraint-satisfying samples of the QAOA binding and of the pooled nulls."""
    stored = json.loads((RESULTS / "e14_ibm_hardware_controls_qaoa.json").read_text())
    q, null = stored["qaoa"]["arms"]["qaoa"], stored["qaoa"]["tests"]["pooled_null_feasible"]
    return {"device": stored["backend"], "job_id": stored["qaoa"]["run"]["job_id"],
            "submitted_utc": stored["qaoa"]["run"]["submitted_utc"],
            "feasible": q["feasible"]["count"], "shots": q["shots"], "null_feasible": null["count"],
            "null_shots": null["shots"], "approx_ratio": q["approx_ratio"],
            "two_qubit_gates": stored["qaoa"]["two_qubit_gates"], "depth": stored["qaoa"]["depth"],
            "retained_fraction": stored["qaoa"]["tests"]["retained_fraction_of_feasible_excess"]}


def rate_ratio(k1: int, n1: int, k2: int, n2: int) -> dict:
    """(k1 / n1) / (k2 / n2) with the 95% interval of its logarithm (Katz); nothing if a count is 0."""
    if min(k1, k2) == 0:
        return {"ratio": None, "ci95": None}
    ratio = (k1 / n1) / (k2 / n2)
    half = 1.959964 * math.sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2)
    return {"ratio": ratio, "ci95": [ratio * math.exp(-half), ratio * math.exp(half)]}


_BOUNDARY: dict = {}


def _boundary(a: np.ndarray, n1: int, n2: int, alpha: float) -> np.ndarray:
    """Per count a of the first sample: the largest count b of the second with one-sided p <= alpha (-1: none)."""
    known = _BOUNDARY.setdefault((n1, n2, alpha), {})
    new = np.array([x for x in a.tolist() if x not in known], dtype=int)
    low, high = np.full(len(new), -1), np.full(len(new), n2 + 1)
    while (high - low > 1).any():
        mid = (low + high) // 2
        rejects = hypergeom.sf(new - 1, n1 + n2, new + mid, n1) <= alpha
        low, high = np.where(rejects, mid, low), np.where(rejects, high, mid)
    known.update(zip(new.tolist(), low.tolist()))
    return np.array([known[x] for x in a.tolist()])


def fisher_power(n1: int, p1: float, n2: int, p2: float, alpha: float) -> float:
    """P(one-sided Fisher exact p <= alpha) when the first sample has rate p1, the second p2, and p1 > p2 is tested.

    Exact for independent binomial samples: for each count of the first sample within 8 standard
    deviations of its mean the largest rejecting count of the second is found (the p-value grows with
    it), and the binomial probabilities are summed.
    """
    sd = math.sqrt(n1 * p1 * (1 - p1))
    a = np.arange(max(0, math.floor(n1 * p1 - 8 * sd - 1)), min(n1, math.ceil(n1 * p1 + 8 * sd + 1)) + 1)
    return float(binom.pmf(a, n1, p1) @ binom.cdf(_boundary(a, n1, n2, alpha), n2, p2))


def detectable(power_at, high: float):
    """Smallest d in (0, high] with power_at(d) >= 0.8, for a power that grows with d; None if high is too small."""
    if power_at(high) < POWER:
        return None
    low = 0.0
    for _ in range(14):
        mid = (low + high) / 2
        low, high = (low, mid) if power_at(mid) >= POWER else (mid, high)
    return high


# --------------------------------------------------------------------------- Aer


def _aer_task(task) -> dict:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    circuit, shots, seed, noise = task
    model = None
    if noise:
        model = NoiseModel()
        model.add_all_qubit_quantum_error(depolarizing_error(noise, 2), ["cx", "cz", "ecr"])
    sim = AerSimulator(noise_model=model, seed_simulator=seed, shot_branching_enable=bool(noise),
                       shot_branching_sampling_enable=bool(noise))
    return {str(k): int(v) for k, v in sim.run(circuit, shots=shots).result().get_counts().items()}


def run_aer(pubs: list, experiment: int, noise: float | None = None) -> list:
    """Counts of the pubs on Aer, in the layout of counts_of(). noise: depolarising error of every cx, cz and ecr gate.

    Seeds are point_seed(experiment, pub, binding, chunk). Noisy shots run in chunks of AER_CHUNK in a process pool.
    """
    tasks, place = [], []
    for i, (circuit, bindings, shots) in enumerate(pubs):
        for j, binding in enumerate(bindings or [None]):
            bound = circuit if binding is None else circuit.assign_parameters(
                {p: binding[p.name] for p in circuit.parameters})
            chunks = [shots] if not noise else [AER_CHUNK] * (shots // AER_CHUNK) + [shots % AER_CHUNK] * bool(shots % AER_CHUNK)
            for k, size in enumerate(chunks):
                tasks.append((bound, size, int(point_seed(experiment, i, j, k) % (2 ** 31 - 1)), noise))
                place.append((i, j))
    with ProcessPoolExecutor(max_workers=min(pool_workers(), len(tasks))) as pool:
        parts = list(pool.map(_aer_task, tasks))
    out = [[Counter() for _ in (bindings or [None])] for _, bindings, _ in pubs]
    for (i, j), part in zip(place, parts):
        out[i][j].update(part)
    return [[dict(counter) for counter in pub] for pub in out]
