"""E32 - Mission resource allocation at equal time, and the time of the exact solver.

Analysis plan fixed before any run of this script on the evaluation instances; only --quick was run
before (instance seeds 2000 and 2001, which are the --quick seeds of E31; not saved).

Why. E31 compared the product's 64 annealed reads with as many uniform random bitstrings through the
same repair and fill and, described only, with 1 024 of them. That is a comparison at an equal
number of bitstrings. In E31 the annealing of 64 reads took four to nine times as long as the repair
and fill of 64 bitstrings, so at equal time the random arm would have had several hundred. E31 also
did not record how long the exact solver took, which the workspace runs next to every plan. This
experiment measures, on instances no run has touched,
  1  whether the annealed reads end closer to the optimum than random bitstrings that are given the
     time the annealed arm takes;
  2  how long the exact solver needs, next to the time of the sampling path.
No quantum hardware and no quantum simulator is involved.

Instances. The generator behind the workspace's Custom scenario. E31 used seeds 1000 to 1009, the
Research suites use those, both --quick modes use 2000 and 2001, and no test or stored run uses a
seed above 2001.
  workspace  scenario.build_custom_scenario at the six sizes of E31 part A, (4, 12), (6, 18),
             (8, 24), (10, 30), (12, 36), (14, 40), seeds 3000 to 3009: 60 instances.
  larger     the same generator called past the limits the workspace accepts (scenario._generate
             with the custom profile) at (20, 60), (40, 120) and (80, 240), seeds 3000 to 3004:
             15 instances. Described only.
What was known when the plan was fixed: the result of E31 on its own instances; two --quick runs on
seeds 2000 and 2001 (the harness repeats; on the 12 workspace instances the exact solver took at
most 0.01 s, t_S was 0.02 to 0.13 s, N(t_S) 242 to 444, and the mean gaps were 0.15 percent for S
and 0.28 percent for U_eq, with S ahead on 2 instances, behind on 6 and level on 4; at (20, 60) and
(40, 120) with seed 2000 the gaps were 0.72 and 2.77 percent for S and 3.10 and 5.70 percent for
U_eq); and a probe on
seed 2000 at (20, 60), (30, 90), (40, 120), (60, 180) and (80, 240), where the exact solver took
0.3, 2.4, 4.0, 54 and 23 s and the annealing 0.3 to 6.3 s. The tests below were written before the
--quick run. After it and the probe the larger family was changed: it was (16, 48) and (20, 60)
with ten instances each, sizes at which the exact solver still takes hundredths of a second. No
evaluation instance had been generated.
Referee. exact.solve_exact, the workspace's solver, with a time limit of 60 s at the workspace's
sizes (the workspace's own default is 20 s) and 300 s at the larger ones. An instance whose optimum
is not proven is reported and left out of every comparison of gaps.

Timing. One process, one thread per numerical library, the instances one after another, measured
with time.perf_counter on the workstation of the paper. Before the first instance every arm runs
once on the instance (8, 24, seed 2000) and the result is discarded. The times are reported as
measured, and the comparison at equal time is repeated at half and at twice the time, so that a
conclusion does not rest on one timing.

Arms, per instance and sampler seed 0 to 4. The structure of allowed pairs is built once per
instance and counted for no arm.
  S  the product's path at its default effort: qubo.build_qubo, sampling.sample_qubo (64 reads,
     1 000 sweeps), every read through sampling.stages. t_S is the time of the three together.
  U  uniform random bitstrings from numpy's default generator seeded with the sampler seed + 1,
     4 096 at the workspace's sizes and 8 192 at the larger ones, drawn as 64, then 960, then the
     rest (the first 64 are the product's null, the first 1 024 are U16 of E31), through
     sampling.stages in blocks of 64, each block timed; the time of drawing is spread evenly over
     the blocks. best(N) is the best filled objective among the first N.
     N(t) is the number of bitstrings finished within time t: 64 for every block whose cumulative
     time is at most t, plus the share of the next block in proportion to the time that is left,
     rounded down; at least 1 and at most all of them (a run that reaches the cap is counted and
     reported).
       U_eq = best(N(t_S)),  U_half = best(N(t_S / 2)),  U_double = best(N(2 t_S)).
  E  exact.solve_exact: the time of the call and whether the optimum is proven.
  G  greedy.solve_greedy: its gap and its time.
On every instance and sampler seed the run checks that S and the first 64 bitstrings of U give the
numbers of the sampler block of sampling.solve_sampling, as E31 does. A mismatch stops the run.

Outcomes. gap(x) = 100 (optimum - objective(x)) / optimum, in percent, of the best plan after repair
and fill. Per instance, the mean over the five sampler seeds for S, U_eq, U_half and U_double.
Tests, paired by instance on those means, Wilcoxon signed-rank as in E31 (exact unless a difference
is zero or two absolute differences are equal), on the workspace instances with a proven optimum:
  PRIMARY  gap(S) < gap(U_eq), one-sided, at 0.05.
  T1       gap(S) < gap(U_half), one-sided.
  T2       gap(S) < gap(U_double), one-sided.
  T3       gap(S) < gap(U_eq) on the instances of (12, 36) and (14, 40), one-sided. In E31 these are
           the sizes at which 1 024 random bitstrings fell behind the annealed reads.
  T4       gap(S) against gap(U_eq) on the instances of the four smaller sizes, two-sided.
The p-values of T1 to T4 are reported unadjusted next to the Bonferroni threshold 0.05 / 4.
Described, not tested: N(t_S) and the times of every arm per size; U at 64 and 1 024 bitstrings and
at all of them; the number of runs in which an arm returns an optimum; in how many instances the
exact solver proves the optimum in less time than the mean t_S of the instance, in how many within
the workspace's default limit of 20 s, and the ratio of its time to t_S per size; all arms on the
15 larger instances.

What the comparison is and is not. The times are those of this implementation on one machine: the
annealer is compiled code, the repair and fill are Python. "Equal time" means equal time of the
product's code, not equal work.

Decision rule, fixed before the run. The walkthrough's "What we measured" gets the result of the
primary test and the time of the exact solver, whatever they are. If the primary test does not
reject, the description of the sampling method says that random bitstrings given the same time end
as close to the optimum.

Usage: e32_mission_equal_time.py [--quick]
  --quick  instance seeds 2000 and 2001 (2000 only and two sizes in the larger family), two sampler
           seeds, a limit of 120 s at the larger sizes, every instance run twice (checks the
           harness and that the untimed part of a run repeats; not saved as a result)
"""

from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import mean_ci, save  # noqa: E402

import e20_portfolio_corrected as e20  # noqa: E402  (the guard on committed code)
import e31_mission_allocation as e31  # noqa: E402  (gap, stages, arm checks and the signed-rank test)

from core.optimizers.mission import sampling, scenario  # noqa: E402
from core.optimizers.mission.exact import solve_exact  # noqa: E402
from core.optimizers.mission.feasibility import objective  # noqa: E402
from core.optimizers.mission.greedy import solve_greedy  # noqa: E402
from core.optimizers.mission.models import build_structure  # noqa: E402
from core.optimizers.mission.qubo import DEFAULT_PENALTY_FACTOR, build_qubo  # noqa: E402

QUICK = "--quick" in sys.argv
INSTANCE_SEEDS = [2000, 2001] if QUICK else list(range(3000, 3010))
LARGER_SEEDS = [2000] if QUICK else list(range(3000, 3005))
SIZES = e31.SIZES_A
LARGER = [(20, 60), (40, 120)] if QUICK else [(20, 60), (40, 120), (80, 240)]
SMALLER_FOUR = SIZES[:4]
LARGER_TWO = SIZES[4:]
SAMPLER_SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
READS, SWEEPS = e31.READS, e31.SWEEPS
BLOCK = READS
DRAWS = {"workspace": (READS, e31.MORE_DRAWS - READS, 4096 - e31.MORE_DRAWS),
         "larger": (READS, e31.MORE_DRAWS - READS, 8192 - e31.MORE_DRAWS)}
EXACT_LIMIT_S = {"workspace": 60.0, "larger": 120.0 if QUICK else 300.0}
WORKSPACE_DEFAULT_LIMIT_S = 20.0
WARM_UP = (8, 24, 2000)
TOL = e31.TOL
ALPHA = 0.05
BONFERRONI = 0.05 / 4
REPEATS = 2 if QUICK else 1
TIMED = ("seconds", "block_seconds", "t_S", "N", "U_half", "U_eq", "U_double", "capped", "optimal_half",
         "optimal_eq", "optimal_double", "repeat")


def instance(kind: str, assets: int, tasks: int, seed: int):
    if kind == "workspace":
        return scenario.build_custom_scenario(assets, tasks, seed)
    return scenario._generate(scenario._CUSTOM_PROFILE, assets, tasks, seed)  # noqa: SLF001 - past the workspace's limits


def finished_within(cumulative: np.ndarray, t: float):
    """Bitstrings finished within time t: whole blocks, plus the share of the next block in proportion."""
    whole = int(np.searchsorted(cumulative, t, side="right"))
    if whole >= len(cumulative):
        return BLOCK * len(cumulative), True
    start = float(cumulative[whole - 1]) if whole else 0.0
    part = int(BLOCK * (t - start) / (float(cumulative[whole]) - start))
    return max(1, BLOCK * whole + part), False


def best_of_first(records: list, count: int) -> float:
    """The best filled objective among the first `count` bitstrings, from the records of improvements."""
    return max(value for index, value in records if index <= count)


def run_instance(task) -> dict:
    kind, assets, tasks, seed, repeat = task
    problem = instance(kind, assets, tasks, seed)
    st = build_structure(problem)
    t0 = time.perf_counter()
    exact_plan, diag = solve_exact(problem, time_limit_s=EXACT_LIMIT_S[kind])
    exact_seconds = time.perf_counter() - t0
    optimum = float(objective(problem, exact_plan))
    t0 = time.perf_counter()
    greedy_plan = solve_greedy(problem, st)
    greedy_seconds = time.perf_counter() - t0
    n = len(st.pairs)
    row = {"kind": kind, "assets": assets, "tasks": tasks, "seed": seed, "repeat": repeat, "pairs": n,
           "proven": bool(diag["proven_optimal"]), "optimum": optimum,
           "exact": {"seconds": exact_seconds, "status": diag["status"], "binaries": diag["binaries"],
                     "constraints": diag["constraints"], "time_limit_s": EXACT_LIMIT_S[kind]},
           "greedy": {"gap": e31.gap(optimum, objective(problem, greedy_plan)), "seconds": greedy_seconds},
           "runs": []}
    for s in SAMPLER_SEEDS:
        t0 = time.perf_counter()
        model = build_qubo(problem, DEFAULT_PENALTY_FACTOR, st)
        t1 = time.perf_counter()
        bits, _ = sampling.sample_qubo(model, reads=READS, sweeps=SWEEPS, seed=s)
        t2 = time.perf_counter()
        reads = e31.through(problem, st, bits)
        t3 = time.perf_counter()
        if len(model.variables) != n:
            raise AssertionError("the QUBO does not have one variable per allowed pair")

        t4 = time.perf_counter()
        rng = np.random.default_rng(s + 1)
        drawn = np.concatenate([rng.integers(0, 2, size=(count, n)) for count in DRAWS[kind]])
        draw_seconds = time.perf_counter() - t4
        total = len(drawn)
        filled, block_seconds = [], []
        for start in range(0, total, BLOCK):
            t5 = time.perf_counter()
            part = e31.through(problem, st, drawn[start:start + BLOCK])
            block_seconds.append(time.perf_counter() - t5 + draw_seconds * BLOCK / total)
            filled.append(part[3])
            if start == 0:
                null = part
        filled = np.concatenate(filled)
        cumulative = np.cumsum(block_seconds)

        _, shown = sampling.solve_sampling(problem, reads=READS, sweeps=SWEEPS, seed=s,
                                           penalty_factor=DEFAULT_PENALTY_FACTOR, qubo=model)
        mine = {"repaired": e31.as_shown(reads[2]), "filled": e31.as_shown(reads[3]),
                "null_repaired": e31.as_shown(null[2]), "null_filled": e31.as_shown(null[3]),
                "feasible_share": round(float(reads[0].sum()) / READS, 4)}
        theirs = {"repaired": shown["repaired"], "filled": shown["filled"],
                  "null_repaired": shown["null"]["repaired"], "null_filled": shown["null"]["filled"],
                  "feasible_share": shown["raw"]["feasible_share"]}
        if mine != theirs:
            raise AssertionError(f"sampler seed {s}: the arms are not the product's block: {mine} != {theirs}")

        records, best = [], -np.inf
        for index, value in enumerate(filled, start=1):
            if value > best:
                best = float(value)
                records.append([index, best])
        t_s = t3 - t0
        counts = {name: finished_within(cumulative, factor * t_s)
                  for name, factor in (("half", 0.5), ("eq", 1.0), ("double", 2.0))}
        run = {"sampler_seed": s, "S": e31.gap(optimum, reads[3].max()),
               "S_optimal": bool(reads[3].max() >= optimum - TOL),
               "seconds": {"build_qubo": t1 - t0, "anneal": t2 - t1, "stages": t3 - t2, "draw": draw_seconds,
                           "null_total": float(cumulative[-1])},
               "t_S": t_s, "block_seconds": [float(v) for v in block_seconds], "records": records,
               "bitstrings": total,
               "N": {name: count for name, (count, _) in counts.items()},
               "capped": [name for name, (_, capped) in counts.items() if capped]}
        for name, (count, _) in counts.items():
            value = best_of_first(records, count)
            run[f"U_{name}"] = e31.gap(optimum, value)
            run[f"optimal_{name}"] = bool(value >= optimum - TOL)
        for name, count in ((f"U{READS}", READS), (f"U{e31.MORE_DRAWS}", e31.MORE_DRAWS), ("U_all", total)):
            value = best_of_first(records, count)
            run[name] = e31.gap(optimum, value)
            run[f"{name}_optimal"] = bool(value >= optimum - TOL)
        row["runs"].append(run)
    return row


def per_instance(row: dict, field: str) -> float:
    return float(np.mean([run[field] for run in row["runs"]]))


def describe(rows) -> dict:
    used = [r for r in rows if r["proven"]]
    out = {"instances": len(rows), "proven": len(used),
           "not_proven": [[r["assets"], r["tasks"], r["seed"]] for r in rows if not r["proven"]],
           "pairs": [min(r["pairs"] for r in rows), max(r["pairs"] for r in rows)],
           "runs": int(sum(len(r["runs"]) for r in used))}
    if used:
        for field in ("S", "U_half", "U_eq", "U_double", f"U{READS}", f"U{e31.MORE_DRAWS}", "U_all"):
            out[field] = mean_ci([per_instance(r, field) for r in used])
        out["greedy"] = mean_ci([r["greedy"]["gap"] for r in used])
        out["optimal_runs"] = {name: int(sum(run[key] for r in used for run in r["runs"]))
                               for name, key in (("S", "S_optimal"), ("U_half", "optimal_half"),
                                                 ("U_eq", "optimal_eq"), ("U_double", "optimal_double"))}
    for name in ("half", "eq", "double"):
        counts = [run["N"][name] for r in rows for run in r["runs"]]
        out[f"N_{name}"] = {"mean": float(np.mean(counts)), "min": int(min(counts)), "max": int(max(counts))}
    out["capped_runs"] = int(sum(bool(run["capped"]) for r in rows for run in r["runs"]))
    seconds = {"t_S": [run["t_S"] for r in rows for run in r["runs"]],
               **{key: [run["seconds"][key] for r in rows for run in r["runs"]]
                  for key in ("build_qubo", "anneal", "stages", "null_total")},
               "exact": [r["exact"]["seconds"] for r in rows], "greedy": [r["greedy"]["seconds"] for r in rows]}
    out["seconds"] = {key: {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(min(v)),
                            "max": float(max(v))} for key, v in seconds.items()}
    mean_t_s = [per_instance(r, "t_S") for r in rows]
    out["exact_faster_than_t_S"] = int(sum(r["exact"]["seconds"] < t for r, t in zip(rows, mean_t_s) if r["proven"]))
    out["exact_within_default_limit"] = int(sum(r["proven"] and r["exact"]["seconds"] <= WORKSPACE_DEFAULT_LIMIT_S
                                                for r in rows))
    ratios = [r["exact"]["seconds"] / t for r, t in zip(rows, mean_t_s)]
    out["exact_over_t_S"] = {"median": float(np.median(ratios)), "min": float(min(ratios)), "max": float(max(ratios))}
    return out


def tests(rows) -> dict:
    used = [r for r in rows if r["proven"]]

    def column(part, field):
        return [per_instance(r, field) for r in part]

    larger = [r for r in used if (r["assets"], r["tasks"]) in LARGER_TWO]
    smaller = [r for r in used if (r["assets"], r["tasks"]) in SMALLER_FOUR]
    return {
        "PRIMARY": e31.signed_rank(column(used, "S"), column(used, "U_eq"), "less", ALPHA),
        "T1": e31.signed_rank(column(used, "S"), column(used, "U_half"), "less", BONFERRONI),
        "T2": e31.signed_rank(column(used, "S"), column(used, "U_double"), "less", BONFERRONI),
        "T3": e31.signed_rank(column(larger, "S"), column(larger, "U_eq"), "less", BONFERRONI),
        "T4": e31.signed_rank(column(smaller, "S"), column(smaller, "U_eq"), "two-sided", BONFERRONI),
    }


def without_time(value):
    if isinstance(value, dict):
        return {k: without_time(v) for k, v in value.items() if k not in TIMED}
    if isinstance(value, list):
        return [without_time(v) for v in value]
    return value


def main() -> None:
    if not QUICK:
        e20.require_committed_code()
    started = time.time()
    run_instance(("workspace", *WARM_UP, 0))
    tasks = [("workspace", a, t, seed, rep) for a, t in SIZES for seed in INSTANCE_SEEDS for rep in range(REPEATS)]
    tasks += [("larger", a, t, seed, rep) for a, t in LARGER for seed in LARGER_SEEDS for rep in range(REPEATS)]
    rows = []
    for task in tasks:
        rows.append(run_instance(task))
        r = rows[-1]
        print(f"{r['kind']} {r['assets']}x{r['tasks']} seed {r['seed']}: {r['pairs']} pairs, exact "
              f"{r['exact']['seconds']:.4f} s ({r['exact']['status']}), t_S {per_instance(r, 't_S'):.4f} s, "
              f"N_eq {np.mean([run['N']['eq'] for run in r['runs']]):.0f}, S {per_instance(r, 'S'):.3f} %, "
              f"U_eq {per_instance(r, 'U_eq'):.3f} %", flush=True)

    if QUICK:
        first = [r for r in rows if r["repeat"] == 0]
        again = [r for r in rows if r["repeat"] == 1]
        same = sum(without_time(a) == without_time(b) for a, b in zip(first, again))
        print(f"harness: the untimed part of {same} of {len(again)} repeated instances is identical", flush=True)
        rows = first

    workspace = [r for r in rows if r["kind"] == "workspace"]
    larger = [r for r in rows if r["kind"] == "larger"]
    summary = {
        "workspace": describe(workspace),
        "by_size": {f"{a}x{t}": describe([r for r in rows if (r["assets"], r["tasks"]) == (a, t)])
                    for a, t in SIZES + LARGER},
        "larger": describe(larger),
        "tests": tests(workspace),
    }
    for name, test in summary["tests"].items():
        print(f"{name:>8}: p = {test['p']}, first smaller/larger/zero {test['first_smaller']}/{test['first_larger']}/"
              f"{test['zero']}, median difference {test['median_difference']:.4g}, rejects at {test['alpha']:.4g}: "
              f"{test['rejects']}", flush=True)
    for label, part in (("workspace", summary["workspace"]), ("larger", summary["larger"])):
        if part["proven"]:
            print(f"{label}, {part['proven']} of {part['instances']} proven: S {part['S']['mean']:.4f} %, "
                  f"U_half {part['U_half']['mean']:.4f} %, U_eq {part['U_eq']['mean']:.4f} %, "
                  f"U_double {part['U_double']['mean']:.4f} %, greedy {part['greedy']['mean']:.3f} %; "
                  f"N_eq mean {part['N_eq']['mean']:.0f}; exact median {part['seconds']['exact']['median']:.4f} s, "
                  f"t_S median {part['seconds']['t_S']['median']:.4f} s; exact faster than t_S on "
                  f"{part['exact_faster_than_t_S']}", flush=True)
    print(f"{len(rows)} instances, {time.time() - started:.0f} s", flush=True)
    if QUICK:
        print("quick run: nothing saved", flush=True)
        return
    save("e32_mission_equal_time", {
        "experiment": "E32",
        "plan": {"instance_seeds": INSTANCE_SEEDS, "larger_seeds": LARGER_SEEDS, "sizes": SIZES, "larger": LARGER,
                 "sampler_seeds": SAMPLER_SEEDS, "reads": READS, "sweeps": SWEEPS,
                 "penalty_factor": DEFAULT_PENALTY_FACTOR, "draws": {k: list(v) for k, v in DRAWS.items()},
                 "block": BLOCK, "exact_time_limit_s": EXACT_LIMIT_S,
                 "workspace_default_limit_s": WORKSPACE_DEFAULT_LIMIT_S, "warm_up": list(WARM_UP),
                 "tolerance": TOL, "alpha": ALPHA, "bonferroni": BONFERRONI},
        **summary,
        "rows": rows,
        "seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
