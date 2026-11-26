"""E31 - Mission resource allocation: the QUBO route against random bitstrings through the same repair and fill.

Analysis plan fixed before any run of this script on the evaluation instances; only --quick was run
before (instances with other seeds, not saved).

Why. The Defense workspace (support and logistics allocation only) offers three methods: an exact
mixed-integer solve, a greedy baseline, and a QUBO sampled with a classical simulated annealer. The
QUBO has one binary per allowed asset-task pair and does not encode the duty limit. Every read is
repaired (pairs are dropped until no rule is violated) and filled (unassigned tasks are added
greedily where they fit), and the best filled plan is returned. Next to it the product shows a null:
as many uniform random bitstrings through the same repair and fill. On the four built-in scenarios
with the default seed the sampler's plan was 0.02 to 0.27 percent below the optimum, and on one of
the four the null ended above the sampler. Those four runs are all that is known. This experiment
measures, on instances no run has touched,
  A  what the annealed reads add over the null at each stage, at the sizes the workspace accepts;
  B  what a gate-based sampler could add: QAOA on the same QUBO, evaluated exactly on instances
     small enough to enumerate. The workspace has no gate-based method; B decides whether one is
     built.
Part A involves no quantum hardware and no quantum simulator. Part B is a noiseless state vector.

Instances. scenario.build_custom_scenario(assets, tasks, seed), the generator behind the workspace's
Custom scenario, with seeds 1000 to 1009. The product default is seed 0 and no test or stored run
uses a seed above 99.
  A  (assets, tasks) = (4, 12), (6, 18), (8, 24), (10, 30), (12, 36), (14, 40): 60 instances.
  B  (3, 6), (3, 8), (4, 8), (4, 12): 40 instances. The ten of (4, 12) are those of A. The run stops
     if one has more than 20 allowed pairs.
What was known of these instances when the plan was fixed: their numbers of allowed pairs, read to
choose the sizes, and the time the repair and fill take on random bitstrings of two of them. No
optimum, objective or sample of an evaluation instance had been computed or looked at.
Referee. exact.exact_reference_with_plan with a time limit of 60 s. An instance whose optimum is not
proven is reported and left out of every comparison.

Part A, arms. All on the product's QUBO (penalty factor 1.05) with the product's default effort
(64 reads, 1 000 sweeps) and sampler seeds 0 to 4.
  S    the product's path: sampling.sample_qubo, every read through sampling.stages.
  U    the product's null: 64 uniform random bitstrings from numpy's default generator seeded with
       the sampler seed + 1, through sampling.stages. This is what the product computes and shows.
  U16  1 024 uniform bitstrings: the 64 of U and 960 more from the same generator. Described only.
  G    greedy.solve_greedy, the workspace's baseline. Deterministic.
  F    greedy.fill applied to the empty plan: what the fill returns without a bitstring.
       Deterministic. Described only.
On every instance and sampler seed the run checks that S and U give the numbers of the sampler block
of sampling.solve_sampling (best and mean after repair and after fill, for the reads and for the
null, and the share of feasible raw reads). A mismatch stops the run.

Part A, outcomes. gap(x) = 100 (optimum - objective(x)) / optimum, in percent. Per instance and arm,
the mean over the five sampler seeds of
  best_filled    the gap of the best plan after repair and fill (what the user gets),
  best_repaired  the gap of the best plan after repair only,
  mean_filled    the gap of the mean objective after repair and fill (what one bitstring is worth).
Tests, paired by instance on those means: Wilcoxon signed-rank, exact unless a difference is zero or
two absolute differences are equal.
  PRIMARY  best_filled(S) < best_filled(U), one-sided, at 0.05.
  A1       best_repaired(S) < best_repaired(U), one-sided.
  A2       mean_filled(S) < mean_filled(U), one-sided.
  A3       best_filled(S) against gap(G), two-sided.
  A4       best_filled(U) against gap(G), two-sided.
Described, not tested: the same per size; the number of runs in which each arm returns an optimum;
the share of raw reads and of raw random bitstrings that satisfy every rule; gap(F);
best_filled(U16); the time of the annealing and of the repair and fill; the four built-in scenarios
with the same arms and seeds (their default-seed runs were seen before this plan was written).

Part B. On every instance all 2^n bitstrings go through sampling.stages. For a distribution over
bitstrings this gives exactly
  p_opt  the probability that the filled plan of one bitstring is an optimum,
  gap1   the expected gap of the filled plan of one bitstring,
  p_raw  the probability that one bitstring, as read, satisfies every rule and is an optimum.
The run stops if the filled plan of any bitstring is above the proven optimum.
Distributions
  uniform   2^-n.
  QAOA      depth 1 and 2. The circuit is the platform's (fleet.samplers.build_qaoa_circuit: cost
            Hamiltonian divided by its largest Ising coefficient, transverse-field mixer). The
            angles minimise the exact expectation of the QUBO energy: Nelder-Mead (xatol 1e-4, fatol
            1e-8 on the expectation scaled to [0, 1], at most 800 evaluations) from 30 uniform
            random starts in [0, pi]^(2p), the lowest value kept. That is the best this search finds
            for the objective a QAOA trains on. It is not the product's training and not a proven
            global minimum. The state is computed with the mixer of E10; on every instance and depth
            the run checks it against the state vector of the platform's circuit (largest
            difference of a probability at most 1e-9) and stops otherwise.
  annealer  the product's reads (64 reads, sampler seeds 0 to 4) as shares of the 320 reads.
            Described only.
Tests, paired by instance, Wilcoxon signed-rank as above, one-sided:
  B1  p_opt(QAOA depth 2) > p_opt(uniform).
  B2  gap1(QAOA depth 2) < gap1(uniform).
  B3  p_raw(QAOA depth 2) > p_raw(uniform).
Described: depth 1; the approximation ratio of the QUBO energy; 1 - (1 - p_opt)^64, the probability
that 64 bitstrings hold one whose filled plan is an optimum; all of it per size.

There are seven tests besides the primary one (A1 to A4, B1 to B3). Their p-values are reported
unadjusted next to the Bonferroni threshold 0.05 / 7.

Decision rule, fixed before the run.
  - The walkthrough's paragraph "What we measured" and the description of the sampling method are
    replaced by the result of A, whatever it is. The result card already shows the null next to the
    sampler on every run and says which ended higher.
  - A gate-based method is added to the workspace only if B1 holds at the Bonferroni threshold.
    Otherwise it is not built, and the paper says why.

Usage: e31_mission_allocation.py [--quick]
  --quick  instance seeds 2000 and 2001, two sampler seeds, 5 QAOA starts, no built-in scenarios,
           every task run twice (checks the harness and that a run repeats; not saved as a result)
"""

from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import mean_ci, point_seed, pool_workers, save  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import e20_portfolio_corrected as e20  # noqa: E402  (the guard on committed code)
from e10_qaoa_exact import _apply_mixer  # noqa: E402  (exp(-i beta X) on every qubit)

from core.optimizers.fleet.qubo import FleetQubo  # noqa: E402
from core.optimizers.fleet.samplers import build_qaoa_circuit, ising_scale  # noqa: E402
from core.optimizers.mission import sampling  # noqa: E402
from core.optimizers.mission.exact import exact_reference_with_plan  # noqa: E402
from core.optimizers.mission.feasibility import objective  # noqa: E402
from core.optimizers.mission.greedy import fill, solve_greedy  # noqa: E402
from core.optimizers.mission.models import build_structure  # noqa: E402
from core.optimizers.mission.qubo import DEFAULT_PENALTY_FACTOR, build_qubo, energy  # noqa: E402
from core.optimizers.mission.scenario import build_custom_scenario, get_scenario, list_scenarios  # noqa: E402
from core.optimizers.mission.service import SAMPLING_DEFAULTS  # noqa: E402

QUICK = "--quick" in sys.argv
INSTANCE_SEEDS = [2000, 2001] if QUICK else list(range(1000, 1010))
SIZES_A = [(4, 12), (6, 18), (8, 24), (10, 30), (12, 36), (14, 40)]
SIZES_B = [(3, 6), (3, 8), (4, 8), (4, 12)]
SAMPLER_SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
READS = int(SAMPLING_DEFAULTS["reads"])
SWEEPS = int(SAMPLING_DEFAULTS["sweeps"])
MORE_DRAWS = 1024
EXACT_LIMIT_S = 60.0
DEPTHS = (1, 2)
QAOA_STARTS = 5 if QUICK else 30
MAX_PAIRS = 20
TOL = 1e-6  # objectives are sums of a few dozen terms of the order 1e4
ALPHA = 0.05
BONFERRONI = 0.05 / 7
REPEATS = 2 if QUICK else 1
PRESETS = [] if QUICK else [s["id"] for s in list_scenarios()]


def gap(optimum: float, value: float) -> float:
    return 100.0 * (optimum - float(value)) / optimum


def through(problem, st, bits):
    """Every bitstring through the product's stages: feasible as read, raw, repaired and filled objective."""
    rows = [sampling.stages(problem, row, st) for row in bits]
    return (np.array([r["raw_feasible"] for r in rows], bool),
            np.array([r["raw_objective"] if r["raw_feasible"] else np.nan for r in rows], float),
            np.array([r["repaired_objective"] for r in rows], float),
            np.array([r["filled_objective"] for r in rows], float))


def arm(optimum: float, feasible, raw, repaired, filled) -> dict:
    return {"best_filled": gap(optimum, filled.max()), "best_repaired": gap(optimum, repaired.max()),
            "mean_filled": gap(optimum, filled.mean()), "mean_repaired": gap(optimum, repaired.mean()),
            "raw_feasible_share": float(feasible.mean()),
            "best_raw_feasible": gap(optimum, np.nanmax(raw)) if feasible.any() else None,
            "optimal": bool(filled.max() >= optimum - TOL)}


def as_shown(values) -> dict:
    """Best and mean as sampling._best_mean rounds them."""
    values = [float(v) for v in values]
    return {"best": round(max(values), 3), "mean": round(math.fsum(values) / len(values), 3)}


def run_a(task) -> dict:
    kind, assets, tasks, seed, repeat = task
    head = {"part": "A", "kind": kind, "assets": assets, "tasks": tasks, "seed": seed, "repeat": repeat}
    try:
        problem = build_custom_scenario(assets, tasks, seed) if kind == "custom" else get_scenario(kind)
        st = build_structure(problem)
        exact_plan, block = exact_reference_with_plan(problem, time_limit_s=EXACT_LIMIT_S)
        optimum = float(objective(problem, exact_plan))
        model = build_qubo(problem, DEFAULT_PENALTY_FACTOR, st)
        n = len(model.variables)
        row = {**head, "assets": len(problem.assets), "tasks": len(problem.tasks), "pairs": n,
               "proven": bool(block["proven_optimal"]), "optimum": optimum,
               "duty_limit_binding": block.get("duty_limit_binding"),
               "greedy": gap(optimum, objective(problem, solve_greedy(problem, st))),
               "fill_only": gap(optimum, objective(problem, fill(problem, [], st))), "runs": []}
        for s in SAMPLER_SEEDS:
            t0 = time.perf_counter()
            bits, _ = sampling.sample_qubo(model, reads=READS, sweeps=SWEEPS, seed=s)
            t1 = time.perf_counter()
            reads = through(problem, st, bits)
            t2 = time.perf_counter()
            rng = np.random.default_rng(s + 1)
            null_bits = rng.integers(0, 2, size=(READS, n))
            more_bits = rng.integers(0, 2, size=(MORE_DRAWS - READS, n))
            null = through(problem, st, null_bits)
            more = through(problem, st, more_bits)

            _, shown = sampling.solve_sampling(problem, reads=READS, sweeps=SWEEPS, seed=s,
                                               penalty_factor=DEFAULT_PENALTY_FACTOR, qubo=model)
            mine = {"repaired": as_shown(reads[2]), "filled": as_shown(reads[3]),
                    "null_repaired": as_shown(null[2]), "null_filled": as_shown(null[3]),
                    "feasible_share": round(float(reads[0].sum()) / READS, 4)}
            theirs = {"repaired": shown["repaired"], "filled": shown["filled"],
                      "null_repaired": shown["null"]["repaired"], "null_filled": shown["null"]["filled"],
                      "feasible_share": shown["raw"]["feasible_share"]}
            if mine != theirs:
                raise AssertionError(f"sampler seed {s}: the arms are not the product's block: {mine} != {theirs}")

            all_null = np.concatenate([null[3], more[3]])
            row["runs"].append({"sampler_seed": s, "S": arm(optimum, *reads), "U": arm(optimum, *null),
                                "U16_best_filled": gap(optimum, all_null.max()),
                                "U16_optimal": bool(all_null.max() >= optimum - TOL),
                                "anneal_seconds": t1 - t0, "stages_seconds": t2 - t1})
        return row
    except Exception as exc:  # noqa: BLE001 - reported, and the run stops
        return {**head, "failed": f"{type(exc).__name__}: {exc}"}


def run_b(task) -> dict:
    assets, tasks, seed, repeat = task
    head = {"part": "B", "assets": assets, "tasks": tasks, "seed": seed, "repeat": repeat}
    try:
        from qiskit.quantum_info import Statevector

        problem = build_custom_scenario(assets, tasks, seed)
        st = build_structure(problem)
        model = build_qubo(problem, DEFAULT_PENALTY_FACTOR, st)
        n = len(model.variables)
        if n > MAX_PAIRS:
            raise AssertionError(f"{n} allowed pairs, more than {MAX_PAIRS}")
        exact_plan, block = exact_reference_with_plan(problem, time_limit_s=EXACT_LIMIT_S)
        optimum = float(objective(problem, exact_plan))

        states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
        feasible, raw, _, filled = through(problem, st, states)
        if block["proven_optimal"] and filled.max() > optimum + TOL:
            raise AssertionError(f"a filled plan ({filled.max()}) is above the proven optimum ({optimum})")
        optimal = filled >= optimum - TOL
        raw_optimal = feasible & (np.nan_to_num(raw, nan=-np.inf) >= optimum - TOL)
        gaps = 100.0 * (optimum - filled) / optimum

        # The QUBO in the platform's pair form, its energies, and the Ising form the circuit applies.
        q = np.array([model.terms.get((i, i), 0.0) for i in range(n)], float)
        pair = np.zeros((n, n))
        for (i, j), value in model.terms.items():
            if i != j:
                pair[min(i, j), max(i, j)] += value
        x = states.astype(float)
        energies = x @ q + np.einsum("bi,ij,bj->b", x, pair, x, optimize=True)
        probe = np.random.default_rng(seed).integers(0, 2 ** n, 32)
        if max(abs(energies[k] - energy(model, states[k])) for k in probe) > 1e-6:
            raise AssertionError("the pair-form energy is not the workspace's QUBO energy")
        fq = FleetQubo(q=q, Q=pair, const=0.0, clusters=[], vehicle_ids=[], index={})
        h, coupling, offset = fq.to_ising()
        z = 1.0 - 2.0 * x
        ising = z @ np.asarray(h, float) + np.einsum("bi,ij,bj->b", z, np.asarray(coupling, float), z, optimize=True)
        if np.max(np.abs(ising + offset - energies)) > 1e-6 * max(1.0, float(np.abs(energies).max())):
            raise AssertionError("the Ising form does not reproduce the QUBO energy")
        diag = ising / ising_scale(fq)
        e_min, e_uniform = float(energies.min()), float(energies.mean())

        def state(params, p: int) -> np.ndarray:
            psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
            for layer in range(p):
                psi = psi * np.exp(-1j * params[layer] * diag)
                psi = _apply_mixer(psi, float(params[p + layer]), n)
            return psi

        def metrics(prob: np.ndarray) -> dict:
            p_opt = float(prob[optimal].sum())
            return {"p_opt": p_opt, "gap1": float(prob @ gaps), "p_raw": float(prob[raw_optimal].sum()),
                    "p_feasible": float(prob[feasible].sum()),
                    "approx_ratio": float((e_uniform - prob @ energies) / (e_uniform - e_min)),
                    "p_opt_64": float(1.0 - (1.0 - min(1.0, p_opt)) ** READS)}

        row = {**head, "pairs": n, "proven": bool(block["proven_optimal"]), "optimum": optimum,
               "optimal_bitstrings": int(optimal.sum()), "feasible_bitstrings": int(feasible.sum()),
               "uniform": metrics(np.full(2 ** n, 2.0 ** -n)), "qaoa": {}}
        for p in DEPTHS:
            rng = np.random.default_rng(point_seed(31, assets, tasks, seed, p))

            def shortfall(params, p=p) -> float:
                """One minus the approximation ratio of the QUBO energy; 0 at the QUBO minimum."""
                prob = np.abs(state(params, p)) ** 2
                return float((prob @ energies - e_min) / (e_uniform - e_min))

            best = None
            for _ in range(QAOA_STARTS):
                found = minimize(shortfall, rng.uniform(0.0, np.pi, 2 * p), method="Nelder-Mead",
                                 options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 800, "maxfev": 800})
                if best is None or found.fun < best.fun:
                    best = found
            prob = np.abs(state(best.x, p)) ** 2
            circuit = Statevector(build_qaoa_circuit(fq, np.asarray(best.x, float), p, measure=False)).probabilities()
            check = float(np.max(np.abs(circuit - prob)))
            if check > 1e-9:
                raise AssertionError(f"depth {p}: the state differs from the platform's circuit by {check}")
            row["qaoa"][str(p)] = {**metrics(prob), "angles": [float(v) for v in best.x],
                                   "circuit_check": check, "starts": QAOA_STARTS}

        hits = raw_hits = 0
        read_gaps = []
        for s in SAMPLER_SEEDS:
            bits, _ = sampling.sample_qubo(model, reads=READS, sweeps=SWEEPS, seed=s)
            index = bits.astype(np.int64) @ (1 << np.arange(n, dtype=np.int64))
            hits += int(optimal[index].sum())
            raw_hits += int(raw_optimal[index].sum())
            read_gaps.extend(float(g) for g in gaps[index])
        total = READS * len(SAMPLER_SEEDS)
        row["annealer"] = {"reads": total, "p_opt": hits / total, "gap1": float(np.mean(read_gaps)),
                           "p_raw": raw_hits / total}
        return row
    except Exception as exc:  # noqa: BLE001 - reported, and the run stops
        return {**head, "failed": f"{type(exc).__name__}: {exc}"}


def run_task(task) -> dict:
    return run_a(task[1:]) if task[0] == "A" else run_b(task[1:])


def signed_rank(first, second, alternative: str, alpha: float) -> dict:
    """Wilcoxon signed-rank test of first_k - second_k, as e29.signed_rank: exact unless a difference is zero
    or two absolute ones are equal."""
    d = np.asarray(first, float) - np.asarray(second, float)
    out = {"pairs": len(d), "alternative": alternative, "alpha": alpha, "first_smaller": int((d < -1e-12).sum()),
           "first_larger": int((d > 1e-12).sum()), "zero": int((np.abs(d) <= 1e-12).sum()),
           "median_difference": float(np.median(d)), "range": [float(d.min()), float(d.max())]}
    if out["zero"] == len(d):
        return {**out, "p": None, "rejects": False, "method": "no test: every difference is zero"}
    tied = bool(out["zero"] or len(np.unique(np.abs(d))) < len(d))
    p = float((wilcoxon(d, alternative=alternative) if tied else wilcoxon(d, alternative=alternative,
                                                                           method="exact")).pvalue)
    method = ("scipy.stats.wilcoxon with its default method: a difference is zero or two absolute differences "
              "are equal" if tied else "exact Wilcoxon signed-rank test (scipy.stats.wilcoxon, method exact)")
    return {**out, "p": p, "rejects": bool(p <= alpha), "method": method}


def per_instance(row: dict, side: str, field: str) -> float:
    return float(np.mean([run[side][field] for run in row["runs"]]))


def describe_a(rows) -> dict:
    out = {"instances": len(rows), "pairs": [min(r["pairs"] for r in rows), max(r["pairs"] for r in rows)]}
    for side in ("S", "U"):
        out[side] = {field: mean_ci([per_instance(r, side, field) for r in rows])
                     for field in ("best_filled", "best_repaired", "mean_filled", "mean_repaired", "raw_feasible_share")}
        out[side]["optimal_runs"] = int(sum(run[side]["optimal"] for r in rows for run in r["runs"]))
    out["runs"] = int(sum(len(r["runs"]) for r in rows))
    out["U16_best_filled"] = mean_ci([float(np.mean([run["U16_best_filled"] for run in r["runs"]])) for r in rows])
    out["U16_optimal_runs"] = int(sum(run["U16_optimal"] for r in rows for run in r["runs"]))
    out["greedy"] = mean_ci([r["greedy"] for r in rows])
    out["greedy_optimal_instances"] = int(sum(r["greedy"] <= 100.0 * TOL / r["optimum"] for r in rows))
    out["fill_only"] = mean_ci([r["fill_only"] for r in rows])
    out["duty_limit_binding_instances"] = int(sum(bool(r["duty_limit_binding"]) for r in rows))
    out["anneal_seconds"] = mean_ci([run["anneal_seconds"] for r in rows for run in r["runs"]])
    out["stages_seconds"] = mean_ci([run["stages_seconds"] for r in rows for run in r["runs"]])
    return out


def summarise_a(rows) -> dict:
    custom = [r for r in rows if r["kind"] == "custom"]
    used = [r for r in custom if r["proven"]]
    value = {(side, field): [per_instance(r, side, field) for r in used]
             for side in ("S", "U") for field in ("best_filled", "best_repaired", "mean_filled")}
    greedy = [r["greedy"] for r in used]
    return {
        "instances": len(custom), "proven": len(used),
        "not_proven": [[r["assets"], r["tasks"], r["seed"]] for r in custom if not r["proven"]],
        "all": describe_a(used) if used else None,
        "by_size": {f"{a}x{t}": describe_a([r for r in used if (r["assets"], r["tasks"]) == (a, t)])
                    for a, t in SIZES_A if any((r["assets"], r["tasks"]) == (a, t) for r in used)},
        "tests": {
            "PRIMARY": signed_rank(value["S", "best_filled"], value["U", "best_filled"], "less", ALPHA),
            "A1": signed_rank(value["S", "best_repaired"], value["U", "best_repaired"], "less", BONFERRONI),
            "A2": signed_rank(value["S", "mean_filled"], value["U", "mean_filled"], "less", BONFERRONI),
            "A3": signed_rank(value["S", "best_filled"], greedy, "two-sided", BONFERRONI),
            "A4": signed_rank(value["U", "best_filled"], greedy, "two-sided", BONFERRONI),
        } if used else None,
        "presets": [{"scenario": r["kind"], "assets": r["assets"], "tasks": r["tasks"], "pairs": r["pairs"],
                     "proven": r["proven"], "greedy": r["greedy"], "fill_only": r["fill_only"],
                     **{f"{side}_{field}": per_instance(r, side, field)
                        for side in ("S", "U") for field in ("best_filled", "best_repaired", "mean_filled")},
                     "S_optimal_runs": int(sum(run["S"]["optimal"] for run in r["runs"])),
                     "U_optimal_runs": int(sum(run["U"]["optimal"] for run in r["runs"])),
                     "U16_best_filled": float(np.mean([run["U16_best_filled"] for run in r["runs"]]))}
                    for r in rows if r["kind"] != "custom"],
    }


def summarise_b(rows) -> dict:
    used = [r for r in rows if r["proven"]]
    fields = ("p_opt", "gap1", "p_raw", "p_feasible", "approx_ratio", "p_opt_64")

    def column(source, field):
        return [float(source(r)[field]) for r in used]

    arms = {"uniform": lambda r: r["uniform"], "qaoa_1": lambda r: r["qaoa"]["1"], "qaoa_2": lambda r: r["qaoa"]["2"]}

    def describe(part):
        out = {name: {field: mean_ci([float(source(r)[field]) for r in part]) for field in fields}
               for name, source in arms.items()}
        out["annealer"] = {field: mean_ci([float(r["annealer"][field]) for r in part])
                           for field in ("p_opt", "gap1", "p_raw")}
        out["instances"] = len(part)
        out["pairs"] = [min(r["pairs"] for r in part), max(r["pairs"] for r in part)]
        return out

    described = describe(used) if used else None
    return {
        "instances": len(rows), "proven": len(used),
        "not_proven": [[r["assets"], r["tasks"], r["seed"]] for r in rows if not r["proven"]],
        "pairs": [min(r["pairs"] for r in used), max(r["pairs"] for r in used)] if used else None,
        "arms": described,
        "by_size": {f"{a}x{t}": describe([r for r in used if (r["assets"], r["tasks"]) == (a, t)])
                    for a, t in SIZES_B if any((r["assets"], r["tasks"]) == (a, t) for r in used)},
        "largest_circuit_check": max(r["qaoa"][str(p)]["circuit_check"] for r in used for p in DEPTHS) if used else None,
        "tests": {
            "B1": signed_rank(column(arms["qaoa_2"], "p_opt"), column(arms["uniform"], "p_opt"), "greater", BONFERRONI),
            "B2": signed_rank(column(arms["qaoa_2"], "gap1"), column(arms["uniform"], "gap1"), "less", BONFERRONI),
            "B3": signed_rank(column(arms["qaoa_2"], "p_raw"), column(arms["uniform"], "p_raw"), "greater", BONFERRONI),
        } if used else None,
    }


def without_time(value):
    if isinstance(value, dict):
        return {k: without_time(v) for k, v in value.items() if k not in ("anneal_seconds", "stages_seconds", "repeat")}
    if isinstance(value, list):
        return [without_time(v) for v in value]
    return value


def main() -> None:
    if not QUICK:
        e20.require_committed_code()
    started = time.time()
    tasks = [("A", "custom", a, t, seed, rep) for a, t in SIZES_A for seed in INSTANCE_SEEDS for rep in range(REPEATS)]
    tasks += [("A", preset, 0, 0, 0, 0) for preset in PRESETS]
    tasks += [("B", a, t, seed, rep) for a, t in SIZES_B for seed in INSTANCE_SEEDS for rep in range(REPEATS)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=1))
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} tasks, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed:
        print("  failed:", {k: v for k, v in r.items() if k != "failed"}, r["failed"], flush=True)
    if failed:
        sys.exit("a task failed: no result is written")

    if QUICK:
        first = [r for r in rows if r["repeat"] == 0]
        again = [r for r in rows if r["repeat"] == 1]
        same = sum(without_time(a) == without_time(b) for a, b in zip(first, again))
        print(f"harness: {same} of {len(again)} repeated tasks identical", flush=True)
        rows = first

    part_a = summarise_a([r for r in rows if r["part"] == "A"])
    part_b = summarise_b([r for r in rows if r["part"] == "B"])
    for name, test in {**(part_a["tests"] or {}), **(part_b["tests"] or {})}.items():
        print(f"{name:>8}: p = {test['p']}, first smaller/larger/zero {test['first_smaller']}/{test['first_larger']}/"
              f"{test['zero']}, median difference {test['median_difference']:.4g}, rejects at {test['alpha']:.4g}: "
              f"{test['rejects']}", flush=True)
    if part_a["all"]:
        a = part_a["all"]
        print(f"A, {part_a['proven']} instances: best filled gap S {a['S']['best_filled']['mean']:.4f} %, "
              f"U {a['U']['best_filled']['mean']:.4f} %, greedy {a['greedy']['mean']:.4f} %, "
              f"fill only {a['fill_only']['mean']:.4f} %, U16 {a['U16_best_filled']['mean']:.4f} %; optimal runs "
              f"S {a['S']['optimal_runs']}, U {a['U']['optimal_runs']}, U16 {a['U16_optimal_runs']} of {a['runs']}",
              flush=True)
    if part_b["arms"]:
        b = part_b["arms"]
        print(f"B, {part_b['proven']} instances: p_opt uniform {b['uniform']['p_opt']['mean']:.4f}, "
              f"QAOA-1 {b['qaoa_1']['p_opt']['mean']:.4f}, QAOA-2 {b['qaoa_2']['p_opt']['mean']:.4f}, "
              f"annealer {b['annealer']['p_opt']['mean']:.4f}; largest circuit check "
              f"{part_b['largest_circuit_check']:.2e}", flush=True)
    if QUICK:
        print("quick run: nothing saved", flush=True)
        return
    save("e31_mission_allocation", {
        "experiment": "E31",
        "plan": {"instance_seeds": INSTANCE_SEEDS, "sizes_a": SIZES_A, "sizes_b": SIZES_B, "sampler_seeds": SAMPLER_SEEDS,
                 "reads": READS, "sweeps": SWEEPS, "penalty_factor": DEFAULT_PENALTY_FACTOR, "more_draws": MORE_DRAWS,
                 "exact_time_limit_s": EXACT_LIMIT_S, "depths": list(DEPTHS), "qaoa_starts": QAOA_STARTS,
                 "max_pairs": MAX_PAIRS, "tolerance": TOL, "alpha": ALPHA, "bonferroni": BONFERRONI,
                 "presets": PRESETS},
        "part_a": {**part_a, "rows": [r for r in rows if r["part"] == "A"]},
        "part_b": {**part_b, "rows": [r for r in rows if r["part"] == "B"]},
        "seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
