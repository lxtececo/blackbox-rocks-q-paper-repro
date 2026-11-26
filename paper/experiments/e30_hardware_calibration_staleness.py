"""E30 - Cost of placing a circuit with old calibration data.

Analysis plan fixed before any run of this script.

Question. Does a placement chosen from hours-old calibration data give fewer constraint-satisfying
samples than the placement chosen from the calibration data read at submission, when all circuits
run in the same job?

Decisions
 1. Routing. The circuit of hardware round 2A (e14.qaoa_problem(): 18 qubits, QAOA at depth 1, free
    angles) is routed once on the device's coupling map WITHOUT error data: optimisation level 3
    given only the coupling map and the device's basis gates, transpiler seeds 0 to 7, the circuit
    with the fewest two-qubit gates, the lowest seed among equals. Every arm has this gate
    structure. The arms differ in the physical qubits and, for the null, in one angle.
 2. Placement. All isomorphic placements of the routed circuit's interaction graph on the coupling
    map are enumerated (VF2, directed, not induced). The score of a placement under a set of error
    data is the product of (1 - error) over every gate application and every measurement of the
    routed circuit. A placement that uses a gate or a readout whose error is missing or >= 1 is not
    eligible. The eligible placement with the highest score is chosen; ties go to the
    lexicographically smallest list of physical qubits (listed in the order of the routed circuit's
    qubits). The choice is a deterministic function of (routed circuit, coupling map, error data).
 3. Snapshots. --snapshot stores what the device reports at that moment: the error of every gate
    name and qubit tuple (the readout errors are those of "measure"), the device's
    last_update_date and the UTC time of reading, as gzip JSON in
    results/raw/e30_snapshot_<device>_<UTC time>.json.gz. A file is never replaced.
 4. Submission. --submit reads fresh error data (and stores them as a snapshot), loads the stored
    snapshots of this device and takes, for each requested age (default 6, 24 and 72 hours; --ages),
    the snapshot whose age is closest to it, the more recent of two equally close ones, if its age
    is within 25 percent of the requested age; otherwise that age is skipped and reported. The age
    of a snapshot is the time from its reading to the reading of the fresh data. ONE job:
      fresh      the exact-optimal angles on the placement chosen from the fresh data
      age_<H>h   the exact-optimal angles on the placement chosen from the snapshot taken for age H;
                 one arm per usable age whose placement differs from the fresh one
      null       (gamma*, beta = 0) on the fresh placement; its ideal output is exactly uniform
    with equal shots per arm (--shots, default 40 000). An age whose placement equals the fresh
    placement gets no circuit and is reported as "identical placement, no information", not as
    "no effect". Sampler settings as E14: XY4 dynamical decoupling, gate twirling.
 5. Analysis. Primary outcome: the fraction of constraint-satisfying samples per arm (E14's
    "feasible": one-hot, linking and site limit). One test per age: one-sided Fisher exact test,
    fresh greater than stale. Family-wise alpha 0.01 as in the paper, Bonferroni over the number of
    REQUESTED ages (3 by default, so 0.00333 per test), whether or not an age is skipped or
    uninformative. Also reported: the score of each placement under the fresh data (what the stale
    choice loses according to the device's own numbers), the number of physical qubits the
    placements share, the ratio of the stale rate to the fresh rate with a 95% interval, and, as a
    check outside the family, fresh against null (one-sided Fisher exact test): whether the fresh
    placement shows a signal at all.
 6. --plan prints the arms, the shots, the QPU-seconds estimate and the smallest relative drop of
    the rate that a test detects with power 0.8 at the per-test alpha, for 40 000, 80 000 and
    100 000 shots per arm at the Round 2A rate of 244 per 40 000.
 7. --dry-run uses synthetic error data: the packaged errors of the fake_fez device model as
    "fresh", and three "stale" sets, each the fresh errors multiplied per gate by exp(sigma z) with
    a seeded standard normal z: one aged 6.5 h with sigma small enough that the placement stays
    identical, one aged 22 h with sigma large enough that it changes, and one aged 40 h, so that
    the requested 72 h has no snapshot within tolerance. Each arm is sampled with 300 shots on Aer
    with the depolarising two-qubit errors and the readout errors of the fresh set (single-qubit
    gate errors are left out; the device noise model takes about 0.1 s per shot). With --quick the
    simulation is noiseless and uses the full number of shots. Dry-run counts carry no information
    about hardware; the result says so and its name ends in _dryrun.

No run is excluded. Every submitted job is listed in results/e30_submission.json and in every
result. Raw counts are archived before any analysis. A device result is never replaced. A device
that already has a finished job of this experiment gets a further one only with --again.

To have a usable snapshot for the ages 6, 24 and 72 hours, take snapshots between 54 and 90,
between 18 and 30, and between 4.5 and 7.5 hours before the submission.

Device, chosen on 20 November 2026 before the first snapshot: ibm_marrakesh, the device of hardware
rounds 2 and 3. The detectable drop that --plan prints is computed at the rate measured there.

Amendment of 26 November 2026, written and committed before any job of this experiment was
submitted and before any placement was computed from a stored snapshot. The workstation that takes
the snapshots was suspended from the evening of 21 November until 26 November. The thirteen
snapshots of 20 and 21 November, taken for the 72-hour age, are 110 to 135 hours old on
26 November; none exists for the 24-hour age; one was taken on 26 November at 11:39:51 UTC. The job
is submitted on 26 November between 16:10 and 19:10 UTC, when that snapshot is 4.5 to 7.5 hours
old, with --ages 6,24,120: the 72-hour age is replaced by 120 hours, which the snapshots of
21 November serve within the tolerance of decision 4, and the 24-hour age stays requested and is
reported as skipped. Three ages are requested, so the per-test alpha is 0.00333 as before. Nothing
else changes.

Usage
  e30_hardware_calibration_staleness.py --plan [--ages 6,24,72] [--shots N]      offline
  e30_hardware_calibration_staleness.py --dry-run [--quick]                      Aer, synthetic data
  e30_hardware_calibration_staleness.py --snapshot    store the errors the device reports now
  e30_hardware_calibration_staleness.py --status      device, account usage, jobs, stored snapshots
  e30_hardware_calibration_staleness.py --submit [--ages 6,24,72] [--shots N] [--again]
  e30_hardware_calibration_staleness.py --collect     archive the counts of finished jobs, then analyse
  e30_hardware_calibration_staleness.py --analyse     analyse archived counts that have no result yet
Environment: QFPO_IBM_BACKEND names the device; there is no default.
"""

from __future__ import annotations

import calendar
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import hardware_jobs as hj
import numpy as np
from common import point_seed, pool_workers, provenance
from scipy.stats import fisher_exact

e14 = hj.e14

EXPERIMENT = "e30"
STEM = "e30_hardware_calibration_staleness"
ALPHA = 0.01                                     # family-wise, over the requested ages
AGES = [6.0, 24.0, 72.0]                         # hours
TOLERANCE = 0.25                                 # a snapshot serves an age if within this fraction of it
SHOTS = 40_000                                   # per arm
PLAN_SHOTS = (40_000, 80_000, 100_000)
ROUTING_SEEDS = range(8)
ROUTING_BASIS = ("cx", "cz", "ecr", "rz", "sx", "x")
MAX_JOB_SECONDS = 300
DRY_SHOTS = 300                                  # per arm in a noisy dry run
DRY_CHUNK = 50                                   # noisy shots per Aer task
DRY_NOW = "2026-01-01T12:00:00Z"                 # synthetic time of the fresh reading in a dry run
DRY_SNAPSHOTS = ((6.5, True), (22.0, False), (40.0, False))   # (age in hours, placement stays identical)
TESTED = "tested"
IDENTICAL = "identical placement, no information"
NO_SNAPSHOT = "no snapshot within tolerance"
NOT_ELIGIBLE = "no eligible placement under the snapshot"
FACT_KEYS = ("requested_ages_h", "tolerance", "alpha_family", "alpha_per_test", "shots", "parameters", "qubits",
             "routing", "placements", "fresh")
DESIGN = __doc__.split("\nUsage\n")[0].strip()


# --------------------------------------------------------------------------- error data and snapshots


def utc(stamp=None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


def seconds(stamp: str) -> int:
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def error_data(backend) -> dict:
    """What the device reports now: the error of every gate name and qubit tuple, measurement included."""
    target, errors = backend.target, {}
    for name in sorted(target.operation_names):
        rows = sorted((qubits, props) for qubits, props in target[name].items() if qubits is not None)
        if rows:
            errors[name] = {",".join(map(str, qubits)): None if props is None or props.error is None else float(props.error)
                            for qubits, props in rows}
    try:
        updated = str(backend.properties().last_update_date)
    except Exception as exc:  # noqa: BLE001 - bookkeeping only
        updated = f"unavailable: {type(exc).__name__}"
    return {"experiment": EXPERIMENT, "device": backend.name, "read_utc": utc(), "last_update_date": updated,
            "num_qubits": target.num_qubits, "errors": errors}


def store_snapshot(record: dict) -> str:
    """Decision 3: the error data as a new file below results/raw. Returns the file name."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(seconds(record["read_utc"])))
    path = hj.OUT / "raw" / f"{EXPERIMENT}_snapshot_{record['device']}_{stamp}.json.gz"
    try:
        hj.write_archive(path, {**record, "provenance": provenance(), "code_sha256": hj.code_hash(__file__)})
    except FileExistsError:
        raise SystemExit(f"{path.name} exists and is not replaced: a snapshot was stored in the same second") from None
    return path.name


def load_snapshots(device=None) -> list:
    """The stored snapshots, of one device or of all, oldest first, each with its file name."""
    found = []
    for path in sorted((hj.OUT / "raw").glob(f"{EXPERIMENT}_snapshot_*.json.gz")):
        record = hj.read_archive(path)
        if device is None or record["device"] == device:
            found.append({**record, "file": path.name})
    return sorted(found, key=lambda record: record["read_utc"])


def pick_snapshots(snapshots: list, fresh_utc: str, ages: list) -> list:
    """Decision 4: per requested age the snapshot of the closest age and whether it is within the tolerance."""
    aged = [((seconds(fresh_utc) - seconds(record["read_utc"])) / 3600, record) for record in snapshots]
    aged = [(hours, record) for hours, record in aged if hours > 0]
    picks = []
    for age in ages:
        if not aged:
            picks.append(None)
            continue
        hours, record = min(aged, key=lambda item: (abs(item[0] - age), item[0]))
        picks.append({"hours": hours, "record": record, "usable": abs(hours - age) <= TOLERANCE * age})
    return picks


def snapshot_lines(ages: list, device=None) -> list:
    """The stored snapshots and the requested ages they would serve if the job were submitted now."""
    records, now = load_snapshots(device), utc()
    if not records:
        return [f"  no snapshot{'' if device is None else ' of ' + device} is stored (--snapshot stores one)"]
    lines = []
    for name in sorted({record["device"] for record in records}):
        mine = [record for record in records if record["device"] == name]
        held = ", ".join(f"{(seconds(now) - seconds(record['read_utc'])) / 3600:.1f}" for record in mine)
        lines.append(f"  {name}: {len(mine)} snapshots, now {held} h old")
        for age, pick in zip(ages, pick_snapshots(mine, now, ages)):
            usable = pick is not None and pick["usable"]
            lines.append(f"    age {age:g} h: " + (f"{pick['record']['file']}, {pick['hours']:.2f} h old" if usable else
                                                     f"{NO_SNAPSHOT} ({(1 - TOLERANCE) * age:g} to {(1 + TOLERANCE) * age:g} h)"))
    return lines


# --------------------------------------------------------------------------- routing and placement


def remap(circuit, to: dict, num_qubits: int):
    """The same instructions on other qubits; classical bits and parameters are unchanged."""
    from qiskit import QuantumCircuit

    out = QuantumCircuit(num_qubits, global_phase=circuit.global_phase)
    for creg in circuit.cregs:
        out.add_register(creg)
    for inst in circuit.data:
        out.append(inst.operation, [to[circuit.find_bit(q).index] for q in inst.qubits],
                   [circuit.find_bit(c).index for c in inst.clbits])
    return out


def gate_uses(circuit) -> dict:
    """(gate name, qubits) -> number of applications in the circuit, measurements included."""
    uses = Counter((inst.operation.name, tuple(circuit.find_bit(q).index for q in inst.qubits)) for inst in circuit.data)
    return dict(sorted(uses.items()))


def placements(uses: dict, edges: list, num_qubits: int) -> tuple:
    """Decision 2: every placement of the interaction graph on the coupling map, in lexicographic order.

    Returns (the qubits the routed circuit uses, per placement the physical qubit of each of them).
    """
    import rustworkx as rx

    active = sorted({q for _, qubits in uses for q in qubits})
    index = {q: i for i, q in enumerate(active)}
    pattern, device = rx.PyDiGraph(), rx.PyDiGraph()
    pattern.add_nodes_from(active)
    for _, qubits in uses:
        if len(qubits) == 2 and not pattern.has_edge(index[qubits[0]], index[qubits[1]]):
            pattern.add_edge(index[qubits[0]], index[qubits[1]], None)
    if not rx.is_weakly_connected(pattern):
        raise SystemExit("the interaction graph of the routed circuit is not connected; its placements are not enumerated")
    device.add_nodes_from(range(num_qubits))
    for a, b in edges:
        if not device.has_edge(a, b):
            device.add_edge(a, b, None)
    found = []
    for mapping in rx.vf2_mapping(device, pattern, subgraph=True, induced=False, id_order=True):
        physical = {i: q for q, i in mapping.items()}
        found.append([physical[i] for i in range(len(active))])
    return active, sorted(found)


def log_score(uses: dict, active: list, place: list, errors: dict):
    """Logarithm of the score of a placement under error data; None if the placement is not eligible."""
    to, total = dict(zip(active, place)), 0.0
    for (name, qubits), count in uses.items():
        error = errors.get(name, {}).get(",".join(str(to[q]) for q in qubits))
        if error is None or not error < 1:
            return None
        total += count * math.log1p(-error)
    return total


def choose(layout: dict, errors: dict):
    """Decision 2: (placement, logarithm of its score) of the best eligible placement, the first of equals; None if none."""
    best = None
    for place in layout["places"]:
        score = log_score(layout["uses"], layout["active"], place, errors)
        if score is not None and (best is None or score > best[1]):
            best = (place, score)
    return best


def route(problem: dict, backend) -> dict:
    """Decision 1, and the placements of decision 2, for a device or a device model."""
    basis = sorted(name for name in backend.target.operation_names if name in ROUTING_BASIS)
    seed, routed = hj.transpile_best(problem["circuit"], ROUTING_SEEDS, coupling_map=backend.coupling_map, basis_gates=basis)
    unknown = sorted(set(routed.count_ops()) - set(basis) - {"measure"})
    if unknown:
        raise SystemExit(f"the routed circuit contains {unknown}, for which the device reports no error")
    uses = gate_uses(routed)
    active, places = placements(uses, backend.coupling_map.get_edges(), backend.num_qubits)
    if active not in places:
        raise SystemExit("the placement of the routed circuit itself is not among the enumerated placements")
    return {"seed": seed, "basis": basis, "circuit": routed, "uses": uses, "active": active, "places": places}


def place_on(layout: dict, place: list, backend):
    """The routed circuit on the physical qubits of a placement, checked against the device's gates."""
    circuit = remap(layout["circuit"], dict(zip(layout["active"], place)), backend.num_qubits)
    for name, qubits in gate_uses(circuit):
        if not backend.target.instruction_supported(name, qubits):
            raise SystemExit(f"placement {place}: the device has no {name} on {qubits}")
    return circuit


def build_job(problem: dict, backend, layout: dict, fresh: dict, snapshots: list, ages: list, shots: int) -> dict:
    """Decisions 2 and 4: the placement under each set of error data, the arms and the pubs of the one job."""
    uses, active = layout["uses"], layout["active"]
    eligible = sorted(s for s in (log_score(uses, active, place, fresh["errors"]) for place in layout["places"])
                      if s is not None)
    if not eligible:
        raise SystemExit("not submitted: under the fresh error data no placement is eligible; every placement uses a "
                         "gate or a readout whose error is missing or >= 1")
    fresh_place, fresh_log = choose(layout, fresh["errors"])
    qaoa, null = problem["bindings"]["qaoa"], problem["bindings"]["null_beta"]
    pubs = [(place_on(layout, fresh_place, backend), [qaoa, null], shots)]
    arms = [{"arm": "fresh", "pub": 0, "binding": 0, "angles": "qaoa", "placement": fresh_place},
            {"arm": "null", "pub": 0, "binding": 1, "angles": "null_beta", "placement": fresh_place}]
    rows = []
    for age, pick in zip(ages, pick_snapshots(snapshots, fresh["read_utc"], ages)):
        row = {"age_h": age, "status": NO_SNAPSHOT}
        if pick is not None:
            record = pick["record"]
            row.update(snapshot=record.get("file"), snapshot_age_h=round(pick["hours"], 3),
                       snapshot_read_utc=record["read_utc"], snapshot_last_update_date=record["last_update_date"])
            stale = choose(layout, record["errors"]) if pick["usable"] else None
            if pick["usable"] and stale is None:
                row["status"] = NOT_ELIGIBLE
            elif pick["usable"]:
                under_fresh = log_score(uses, active, stale[0], fresh["errors"])
                row.update(status=IDENTICAL if stale[0] == fresh_place else TESTED, placement=stale[0],
                           score_under_snapshot=math.exp(stale[1]),
                           score_under_fresh_data=None if under_fresh is None else math.exp(under_fresh),
                           shared_physical_qubits=len(set(stale[0]) & set(fresh_place)))
        if row["status"] == TESTED:
            row["arm"] = f"age_{age:g}h"
            arms.append({"arm": row["arm"], "pub": len(pubs), "binding": 0, "angles": "qaoa", "placement": row["placement"]})
            pubs.append((place_on(layout, row["placement"], backend), [qaoa], shots))
        rows.append(row)
    routed = layout["circuit"]
    facts = {"requested_ages_h": list(ages), "tolerance": TOLERANCE, "alpha_family": ALPHA, "alpha_per_test": ALPHA / len(ages),
             "shots": shots, "parameters": dict(qaoa), "qubits": problem["n"],
             "routing": {"transpiler_seed": layout["seed"], "basis_gates": layout["basis"],
                         "two_qubit_gates": e14.two_qubit_gates(routed), "depth": routed.depth(), "qubits_in_use": active},
             "placements": {"enumerated": len(layout["places"]), "eligible_under_fresh_data": len(eligible),
                            "score_best": math.exp(eligible[-1]), "score_median": math.exp(float(np.median(eligible))),
                            "score_worst": math.exp(eligible[0])},
             "fresh": {"snapshot": fresh.get("file"), "read_utc": fresh["read_utc"],
                       "last_update_date": fresh["last_update_date"], "placement": fresh_place, "score": math.exp(fresh_log)},
             "ages": rows, "arms": arms}
    return {"pubs": pubs, "facts": facts}


def age_text(row: dict, fresh_score: float) -> str:
    text = f"age {row['age_h']:g} h: {row['status']}"
    if "snapshot" in row:
        text += f"; closest snapshot {row['snapshot']}, {row['snapshot_age_h']:.2f} h old"
    if "placement" in row:
        under = row["score_under_fresh_data"]
        text += (f"; its placement shares {row['shared_physical_qubits']} physical qubits with the fresh one; score under "
                 f"the fresh data {'not eligible' if under is None else format(under, '.4f')} against {fresh_score:.4f}")
    return text


def describe(label: str, job: dict) -> None:
    facts = job["facts"]
    routing, places, fresh = facts["routing"], facts["placements"], facts["fresh"]
    print(f"{label}: routed without error data: transpiler seed {routing['transpiler_seed']}, {routing['two_qubit_gates']} "
          f"two-qubit gates, depth {routing['depth']}, {len(routing['qubits_in_use'])} qubits in use")
    print(f"  placements: {places['enumerated']} enumerated, {places['eligible_under_fresh_data']} eligible under the fresh "
          f"data; score best {places['score_best']:.4f}, median {places['score_median']:.4f}, worst {places['score_worst']:.4f}")
    print(f"  fresh data read {fresh['read_utc']}, calibration data of {fresh['last_update_date']}; fresh placement "
          f"{fresh['placement']}")
    for row in facts["ages"]:
        print("  " + age_text(row, fresh["score"]))
    print(f"  arms: {', '.join(arm['arm'] for arm in facts['arms'])}; {facts['shots']:,} shots each; circuits (pubs): "
          f"{len(job['pubs'])}", flush=True)


# --------------------------------------------------------------------------- analysis (the same for device and Aer)


def one_sided(k: int, n: int, k0: int, n0: int) -> float:
    """One-sided Fisher exact p-value that the rate behind k of n exceeds the rate behind k0 of n0."""
    return float(fisher_exact([[k, n - k], [k0, n0 - k0]], alternative="greater").pvalue)


def analyse_counts(problem: dict, facts: dict, counts: list) -> dict:
    """Decision 5 on the counts of one job. facts: what build_job() returned, as the manifest keeps it."""
    arms = {}
    for arm in facts["arms"]:
        ideal = problem["ideal"][arm["angles"]]
        arms[arm["arm"]] = {"placement": arm["placement"], **e14.qaoa_arm(problem, counts[arm["pub"]][arm["binding"]], ideal)}
    k, n = arms["fresh"]["feasible"]["count"], arms["fresh"]["shots"]
    k0, n0 = arms["null"]["feasible"]["count"], arms["null"]["shots"]
    alpha, ages = facts["alpha_per_test"], []
    for row in facts["ages"]:
        row = dict(row)
        if row["status"] == TESTED:
            k1, n1 = arms[row["arm"]]["feasible"]["count"], arms[row["arm"]]["shots"]
            p = one_sided(k, n, k1, n1)
            row["test"] = {"hypothesis": "P(constraint-satisfying | fresh placement) > P(constraint-satisfying | this placement)",
                           "method": "one-sided Fisher exact test", "fresh": [k, n], "stale": [k1, n1], "p": p,
                           "alpha": alpha, "rejects": bool(p <= alpha)}
            row["stale_rate_over_fresh_rate"] = hj.rate_ratio(k1, n1, k, n)
        ages.append(row)
    drop = None if k == 0 else hj.detectable(lambda d: hj.fisher_power(n, k / n, n, k / n * (1 - d), alpha), 1.0)
    return {"arms": arms, "ages": ages,
            "signal_check": {"note": "outside the family of tests: whether the fresh placement shows a signal at all",
                             "method": "one-sided Fisher exact test, fresh against null", "fresh": [k, n], "null": [k0, n0],
                             "p": one_sided(k, n, k0, n0), "uniform_rate": float(problem["masks"]["feasible"].mean())},
            "sensitivity": {"note": "smallest relative drop from the observed fresh rate that one test detects with power "
                                    f"{hj.POWER} at the per-test alpha; None if even a rate of 0 is not detected",
                            "relative_drop": drop, "method": hj.POWER_METHOD}}


def report(label: str, facts: dict, result: dict) -> None:
    fresh, null, check = result["arms"]["fresh"], result["arms"]["null"], result["signal_check"]
    print(f"{label}: constraint-satisfying samples, fresh {fresh['feasible']['count']}/{fresh['shots']} (noiseless rate "
          f"{fresh['feasible']['ideal_rate']:.4f}), null {null['feasible']['count']}/{null['shots']} (uniform rate "
          f"{check['uniform_rate']:.5f}); check outside the family, fresh against null: p = {check['p']:.2e}")
    for row in result["ages"]:
        print("  " + age_text(row, facts["fresh"]["score"]))
        if "test" in row:
            test, ratio = row["test"], row["stale_rate_over_fresh_rate"]
            shown = "not defined (a count is 0)" if ratio["ratio"] is None else (
                f"{ratio['ratio']:.3f} [{ratio['ci95'][0]:.3f}, {ratio['ci95'][1]:.3f}]")
            print(f"    {test['stale'][0]}/{test['stale'][1]} against fresh {test['fresh'][0]}/{test['fresh'][1]}; stale rate "
                  f"over fresh rate {shown}; one-sided Fisher p = {test['p']:.3g}, per-test alpha {test['alpha']:.5f}: "
                  f"{'rejects' if test['rejects'] else 'does not reject'}")
    drop = result["sensitivity"]["relative_drop"]
    print("  smallest relative drop from the observed fresh rate detected with power "
          f"{hj.POWER}: {'none, not even a rate of 0' if drop is None else format(drop, '.3f')}", flush=True)


# --------------------------------------------------------------------------- --plan


def plan(ages: list, shots: int) -> None:
    problem, basis, old = e14.qaoa_problem(), hj.usage_basis(), hj.round_2a()
    alpha = ALPHA / len(ages)
    print(f"E30, cost of a placement chosen from old calibration data: one job, at most {len(ages) + 2} arms")
    print(f"  fresh, null and one arm per requested age ({', '.join(format(a, 'g') for a in ages)} h) whose snapshot is "
          f"within {TOLERANCE:.0%} of the age and gives another placement; {shots:,} shots per arm")
    for model in hj.device_models():
        job = build_job(problem, model, route(problem, model), error_data(model), [], ages, shots)
        describe(f"  {model.name}, packaged error data, no snapshot", job)
    full = [(None, [0, 1], shots)] + [(None, [0], shots)] * len(ages)
    estimate = hj.qpu_seconds(full, basis)
    print(f"With every age tested: {estimate['pubs']} pubs, {estimate['shots']:,} shots, at most {MAX_JOB_SECONDS} s of "
          "execution allowed")
    hj.print_estimate(estimate, basis)
    base = old["feasible"] / old["shots"]
    print(f"Test per age: one-sided Fisher exact, fresh greater than stale; family-wise alpha {ALPHA}, {len(ages)} requested "
          f"ages, alpha per test {alpha:.5f}.")
    print(f"  Smallest relative drop of the rate detected with power {hj.POWER}, fresh rate as in Round 2A "
          f"({old['feasible']} per {old['shots']:,}):")
    for n in PLAN_SHOTS:
        drop = hj.detectable(lambda d: hj.fisher_power(n, base, n, base * (1 - d), alpha), 1.0)
        seconds_n = hj.qpu_seconds([(None, [0, 1], n)] + [(None, [0], n)] * len(ages), basis)["seconds"]
        print(f"    {n:,} shots per arm: drop {drop:.3f} (stale rate {base * (1 - drop):.6f} = {base * (1 - drop) * n:.1f} "
              f"per {n:,}; power {hj.fisher_power(n, base, n, base * (1 - drop), alpha):.4f}); job {seconds_n:.1f} QPU seconds")
    print(f"  Method: {hj.POWER_METHOD}.")
    print("Stored snapshots and the ages they would serve if the job were submitted now:")
    print("\n".join(snapshot_lines(ages, os.environ.get("QFPO_IBM_BACKEND"))))


# --------------------------------------------------------------------------- --dry-run


def synthetic_snapshot(fresh: dict, layout: dict, hours: float, same: bool, index: int) -> dict:
    """Decision 7: the fresh errors times exp(sigma z); the first sigma of a fixed list that keeps or changes the placement."""
    fresh_place = choose(layout, fresh["errors"])[0]
    for sigma in ((0.05, 0.02, 0.01, 0.005, 0.002, 0.001) if same else (0.5, 1.0, 1.5, 2.0, 3.0)):
        rng = np.random.default_rng(point_seed(30, 9, index))
        errors = {name: {key: e if e is None or e == 0 or e >= 1 else min(e * math.exp(sigma * rng.standard_normal()), 0.5)
                         for key, e in rows.items()} for name, rows in fresh["errors"].items()}
        chosen = choose(layout, errors)
        if chosen is not None and (chosen[0] == fresh_place) == same:
            return {"device": fresh["device"], "read_utc": utc(seconds(fresh["read_utc"]) - round(hours * 3600)),
                    "last_update_date": "synthetic", "errors": errors, "file": f"synthetic (fresh errors x exp({sigma} z))"}
    raise SystemExit("dry run: no perturbation of the fixed list " + ("keeps" if same else "changes") + " the placement")


def _simulate(task) -> dict:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    circuit, shots, seed, noise = task
    model = None
    if noise is not None:
        model = NoiseModel()
        for name, qubits, error in noise["gates"]:
            model.add_quantum_error(depolarizing_error(error, 2), [name], list(qubits))
        for qubit, error in noise["readout"]:
            model.add_readout_error(ReadoutError([[1 - error, error], [error, 1 - error]]), [qubit])
    sim = AerSimulator(noise_model=model, seed_simulator=seed, shot_branching_enable=model is not None,
                       shot_branching_sampling_enable=model is not None)
    return {str(k): int(v) for k, v in sim.run(circuit, shots=shots).result().get_counts().items()}


def simulate(pubs: list, errors) -> list:
    """Counts of the pubs on Aer, each circuit on the qubits it uses. errors: None (noiseless) or the error data applied.

    Seeds are point_seed(30, pub, binding, chunk). Noisy shots run in chunks of DRY_CHUNK in a process pool.
    """
    tasks, where = [], []
    for i, (circuit, bindings, shots) in enumerate(pubs):
        active = sorted({q for _, qubits in gate_uses(circuit) for q in qubits})
        small = remap(circuit, {q: k for k, q in enumerate(active)}, len(active))
        noise = None
        if errors is not None:
            reported = {(name, qubits): errors[name][",".join(str(active[q]) for q in qubits)]
                        for name, qubits in gate_uses(small)}
            reported = {key: 1.0 if e is None else min(e, 1.0) for key, e in reported.items()}
            noise = {"gates": [(name, qubits, e) for (name, qubits), e in reported.items() if len(qubits) == 2],
                     "readout": [(qubits[0], e) for (name, qubits), e in reported.items() if name == "measure"]}
        for j, binding in enumerate(bindings):
            bound = small.assign_parameters({p: binding[p.name] for p in small.parameters})
            chunks = [shots]
            if errors is not None:
                chunks = [DRY_CHUNK] * (shots // DRY_CHUNK) + [shots % DRY_CHUNK] * bool(shots % DRY_CHUNK)
            for k, size in enumerate(chunks):
                tasks.append((bound, size, int(point_seed(30, i, j, k) % (2 ** 31 - 1)), noise))
                where.append((i, j))
    with ProcessPoolExecutor(max_workers=min(pool_workers(), len(tasks))) as pool:
        parts = list(pool.map(_simulate, tasks))
    out = [[Counter() for _ in bindings] for _, bindings, _ in pubs]
    for (i, j), part in zip(where, parts):
        out[i][j].update(part)
    return [[dict(counter) for counter in pub] for pub in out]


def dry_run(args) -> None:
    started = time.time()
    problem, model = e14.qaoa_problem(), hj.device_models()[0]
    layout = route(problem, model)
    fresh = {**error_data(model), "read_utc": DRY_NOW, "file": f"synthetic: packaged errors of {model.name}"}
    snapshots = [synthetic_snapshot(fresh, layout, hours, same, index) for index, (hours, same) in enumerate(DRY_SNAPSHOTS)]
    shots = SHOTS if args.quick else DRY_SHOTS
    job = build_job(problem, model, layout, fresh, snapshots, AGES, shots)
    describe(f"dry run on {model.name}, synthetic error data", job)
    counts = simulate(job["pubs"], None if args.quick else fresh["errors"])
    result = analyse_counts(problem, job["facts"], counts)
    execution = ("Aer, noiseless" if args.quick else
                 "Aer with the depolarising two-qubit errors and the readout errors of the synthetic fresh data")
    report(f"dry run ({execution}; {shots:,} shots per arm; these counts carry no information about hardware)",
           job["facts"], result)
    hj.save_result(hj.dry_run_name(STEM, args), {
        "design": DESIGN, "execution": execution,
        "note": "dry run: synthetic error data and simulated counts; they validate the pipeline and the analysis and "
                "carry no information about hardware",
        "synthetic_snapshots": [{"file": s["file"], "read_utc": s["read_utc"]} for s in snapshots],
        **{key: job["facts"][key] for key in FACT_KEYS}, **result, "wall_s": round(time.time() - started, 1)})


# --------------------------------------------------------------------------- the device


def snapshot() -> None:
    _, backend = hj.device()
    record = error_data(backend)
    name = store_snapshot(record)
    reported = sum(1 for rows in record["errors"].values() for e in rows.values() if e is not None)
    print(f"stored {name}: {backend.name}, read {record['read_utc']}, calibration data of {record['last_update_date']}, "
          f"{reported} reported errors over {len(record['errors'])} gate names")
    print("\n".join(snapshot_lines(AGES, backend.name)))


def submit(ages: list, shots: int, again: bool) -> None:
    if not os.environ.get("QFPO_IBM_BACKEND"):
        raise SystemExit("not submitted: set QFPO_IBM_BACKEND to the name of the device")
    hj.refuse_repeat(EXPERIMENT, os.environ["QFPO_IBM_BACKEND"], again)
    hj.e20.require_committed_code()
    opened, backend = hj.device()
    hj.refuse_unfinished(EXPERIMENT, opened)
    try:
        limit = backend.configuration().max_shots
    except Exception:  # noqa: BLE001 - a device that states no limit is not limited here
        limit = None
    if limit and shots > limit:
        raise SystemExit(f"not submitted: {shots:,} shots per arm, but {backend.name} accepts at most {limit:,} per circuit")
    problem = e14.qaoa_problem()
    layout = route(problem, backend)
    fresh = error_data(backend)
    job = build_job(problem, backend, layout, fresh, load_snapshots(backend.name), ages, shots)
    describe(f"E30 on {backend.name}", job)
    if not any(row["status"] == TESTED for row in job["facts"]["ages"]):
        raise SystemExit("not submitted: no requested age has a usable snapshot whose placement differs from the fresh one, "
                         "so the job could test nothing")
    estimate = hj.qpu_seconds(job["pubs"])
    if estimate["seconds"] > MAX_JOB_SECONDS / 1.5:
        raise SystemExit(f"not submitted: {estimate['seconds']:.0f} QPU seconds estimated, and a job is limited to "
                         f"{MAX_JOB_SECONDS} s of execution; reduce --shots")
    job["facts"]["fresh"]["snapshot"] = store_snapshot(fresh)
    hj.submit(EXPERIMENT, __file__, opened, backend, job["pubs"], MAX_JOB_SECONDS, job["facts"], again=again)


def analyse() -> None:
    todo = hj.to_analyse(EXPERIMENT, STEM)
    if not todo:
        print("E30: no archived job is without a result")
        return
    hj.e20.require_committed_code()
    problem = e14.qaoa_problem()
    for entry, record in todo:
        started = time.time()
        pubs = 1 + max(arm["pub"] for arm in entry["arms"])
        requested = [[entry["shots"]] * sum(arm["pub"] == pub for arm in entry["arms"]) for pub in range(pubs)]
        if record["shots"] != requested:
            raise SystemExit(f"job {entry['job_id']} not analysed: shots per pub {record['shots']}, requested {requested}")
        if entry["parameters"] != problem["bindings"]["qaoa"]:
            raise SystemExit(f"job {entry['job_id']} not analysed: the angles of e14.qaoa_problem() are not those submitted")
        result = analyse_counts(problem, entry, record["counts"])
        report(f"{entry['device']}, job {entry['job_id']}", entry, result)
        hj.save_result(f"{STEM}_{entry['tag']}", {
            "design": DESIGN, "execution": entry["device"], **{key: entry[key] for key in FACT_KEYS}, **result,
            **hj.job_summary(EXPERIMENT, entry, __file__), "wall_s": round(time.time() - started, 1)})


def main(argv=None) -> None:
    p = hj.parser(__doc__, {"snapshot": "store the errors the device reports now below results/raw"})
    p.add_argument("--ages", metavar="H,H,..", help="with --plan and --submit: requested ages in hours (default 6,24,72)")
    p.add_argument("--shots", type=int, metavar="N", help="with --plan and --submit: shots per arm (default 40000)")
    args = hj.parse(p, argv)
    if args.noise is not None:
        p.error("E30 has no --noise: its dry run applies the errors of the packaged device model")
    if (args.ages is not None or args.shots is not None) and not (args.plan or args.submit):
        p.error("--ages and --shots belong to --plan and --submit")
    try:
        ages = list(AGES) if args.ages is None else [float(x) for x in args.ages.split(",")]
    except ValueError:
        p.error("--ages: numbers of hours separated by commas")
    if not ages or min(ages) <= 0 or len(set(ages)) != len(ages):
        p.error("--ages: distinct positive numbers of hours")
    shots = SHOTS if args.shots is None else args.shots
    if shots < 1:
        p.error("--shots: a positive number")
    if args.plan:
        plan(ages, shots)
    elif args.dry_run:
        dry_run(args)
    elif args.snapshot:
        snapshot()
    elif args.status:
        opened, backend = hj.device()
        hj.status(EXPERIMENT, opened, backend)
        print("\n".join(snapshot_lines(AGES, backend.name)))
    elif args.submit:
        submit(ages, shots, args.again)
    elif args.collect:
        if not hj.load_manifest(EXPERIMENT)["jobs"]:
            raise SystemExit("E30: no job has been submitted")
        hj.collect(EXPERIMENT, hj.service())
        analyse()
    else:
        analyse()


if __name__ == "__main__":
    main()
