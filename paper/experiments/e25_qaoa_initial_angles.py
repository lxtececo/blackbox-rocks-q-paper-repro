"""E25 - the initial angles of the portfolio QAOA, in the order the code intends.

Why. The product starts the training of its portfolio QAOA from two linear
ramps over the layers: gamma rising from pi/8 to pi/2 and beta falling from
pi/2 to pi/8. It writes them as (gamma_0, beta_0, gamma_1, beta_1). Qiskit's
QAOAAnsatz orders its parameters (beta_0, beta_1, gamma_0, gamma_1). With the
product's two layers the training therefore starts at beta = (pi/8, pi/2) and
gamma = (pi/2, pi/8): the two ramps are exchanged and each runs the other way.
The paper states this and says that the intended start was not measured. This
experiment measures it.

Arms, on the same instances with the same seeds
  shipped   the product code, unchanged. On the instances of E21 and E23 with
            their seeds it must return what those experiments stored. The
            number of rows that do is recorded; if one differs, the comparison
            below is not paired with the paper's numbers and is not to be used.
  intended  the product code with one function replaced, the one that builds
            the initial vector: the same two ramps, placed by parameter name,
            so that gamma_k and beta_k receive the values meant for them.
            Nothing else differs.

Instances and seeds: those of E21 (factor model, 504 days) and of E23 (the
product's simulated market data, 90 days), 10 instances at each of 8, 12 and
16 assets and 5 seeds each, in each family: 300 runs per arm. There is no null
arm: the null ansatz returns uniform samples for every parameter value, so its
result cannot depend on the initial angles. E21 and E23 report it.

Tested, intended against shipped, paired by instance on per-instance means,
Wilcoxon signed-rank, two-sided, exact unless absolute differences tie:
  (a) the gap of the returned selection to the exact optimum (what a user
      gets);
  (b) the approximation ratio of the exact final state (what the training
      reached).
Twelve comparisons (two outcomes x two families x three sizes). p-values are
reported unadjusted next to the Bonferroni threshold 0.05 / 12. An instance on
which the two arms tie does not count. With ten untied instances the threshold
is reached only at a rank sum of at most 1 for the smaller side, with nine only
at 0, and with fewer than nine it cannot be reached. Outcome (a) is zero in
most runs at 8 assets in E21 and E23, so it may tie on most instances there.

Described, not tested: the number of optimal runs; the probability of the
optimum in the final state against 2^-n; both state measures at the initial
angles; the number of runs in which training raised the approximation ratio.

Decision rule, fixed before the run. This experiment does not change the
product. A change to the intended order is proposed if the intended arm is
better at the Bonferroni threshold in at least one of the twelve comparisons
and worse at that threshold in none. It would require E21, E23 and E24 to be
run again, because every QAOA result of the paper was measured with the
shipped order. Otherwise the product keeps the order it has. In both cases the
comment of the function, which says that its vector matches the layout of the
ansatz, is corrected to say what the code does, and the paper reports the
comparison.

Usage: e25_qaoa_initial_angles.py [--quick]
  --quick   2 instances x 2 seeds at 8 assets, on instances outside the
            evaluation, each run twice (checks the harness and that a run
            repeats; not saved as a result)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import RESULTS, mean_ci, point_seed, pool_workers, save  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import e19_portfolio as e19  # noqa: E402  (instances, enumeration, scores)
import e20_portfolio_corrected as e20  # noqa: E402  (defaults, the guards)
import e23_qaoa_product_data as e23  # noqa: E402  (the stored product data)

QUICK = "--quick" in sys.argv
SIZES = [8] if QUICK else [8, 12, 16]
INSTANCES = 2 if QUICK else 10
SEEDS = 2 if QUICK else 5
REPEATS = 2 if QUICK else 1
FAMILIES = ("e21", "e23")
ARMS = ("shipped", "intended")
OUTCOMES = {"gap": ("gap", None, True), "approx_ratio": ("approx_ratio", "state", False)}  # field, sub, lower is better
BONFERRONI = 0.05 / 12
D = e20.DEFAULTS["qaoa"]
STORED = {"e21": "e21_qaoa_fresh.json", "e23": "e23_qaoa_product_data.json"}


def intended_init(num_params: int, reps: int) -> np.ndarray:
    """The product's two ramps, placed by parameter name: QAOAAnsatz orders beta_0.., then gamma_0.. ."""
    reps = max(1, int(reps))
    gamma = np.linspace(np.pi / 8, np.pi / 2, reps)
    beta = np.linspace(np.pi / 2, np.pi / 8, reps)
    return np.concatenate([beta, gamma])[:num_params]


def prices_of(family: str, n: int, instance: int) -> pd.DataFrame:
    if family == "e21":
        # Evaluation: instances 10..19 of E21. Harness check: 90, 91, as in E21's own check.
        return e19.make_prices(n, 1000 + 100 * (n // 4) + (90 + instance if QUICK else 10 + instance))
    if not QUICK:
        return e23.prices_of(n, instance)
    # Harness check: 8 symbols out of the block E23 uses for its 16-asset instances; no 8-asset instance of E23.
    e23.prices_of(16, 0)
    symbols = e23.SYMBOLS[200 + 8 * instance:208 + 8 * instance]
    return pd.DataFrame({s: e23._DATA["close"][s] for s in symbols}, index=pd.to_datetime(e23._DATA["dates"]))


def seed_of(family: str, n: int, instance: int, rep: int) -> int:
    if QUICK:
        return int(point_seed(25, 9, int(family[1:]), n, instance, rep) % (2 ** 31 - 1))
    stream, first = (21, 10) if family == "e21" else (23, 0)
    return int(point_seed(stream, 0, n, first + instance, rep, 0) % (2 ** 31 - 1))


def run_task(task):
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    family, n, instance, rep, arm, repeat = task
    logging.disable(logging.CRITICAL)
    prices = prices_of(family, n, instance)
    seed = seed_of(family, n, instance, rep)
    started = time.time()
    saved = qaoa._interp_init_params
    if arm == "intended":
        qaoa._interp_init_params = intended_init
    try:
        with e20.quiet():
            result = qaoa.quantum_optimize(prices, risk_aversion=D["risk_aversion"], qc_simulator_mode=True,
                                           estimator_shots=D["estimator_shots"], sampler_shots=D["sampler_shots"],
                                           max_iterations=D["max_iterations"], reps=D["reps"], seed=seed)
        linear, quadratic, _ = qaoa._build_qubo(prices, D["risk_aversion"])
        cost = e19.pair_costs(linear, quadratic, n)
        out = e19.score(cost, e19.index_of(result["solution"]))
        out["init_scheme"] = result["warm_start"]["init_scheme"]
        ansatz = QAOAAnsatz(cost_operator=qaoa._build_hamiltonian(linear, quadratic, n), reps=D["reps"])
        names = [p.name for p in ansatz.parameters]
        expected = [f"β[{k}]" for k in range(D["reps"])] + [f"γ[{k}]" for k in range(D["reps"])]
        if names != expected:
            raise RuntimeError(f"the ansatz orders its parameters {names}, not {expected}")
        initial = qaoa._interp_init_params(ansatz.num_parameters, D["reps"])
        out["initial_over_pi"] = dict(zip(names, (float(v / np.pi) for v in initial)))
        pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
        out["state"] = e19.state_metrics(e19.exact_probabilities(pm.run(ansatz.assign_parameters(result["final_parameters"]))), cost)
        out["state_initial"] = e19.state_metrics(e19.exact_probabilities(pm.run(ansatz.assign_parameters(initial))), cost)
    except Exception as exc:  # recorded, never dropped
        out = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        qaoa._interp_init_params = saved
    out.update(family=family, n=n, instance=instance, rep=rep, arm=arm, repeat=repeat, seed=seed,
               wall_s=round(time.time() - started, 2))
    return out


def paired(intended: dict, shipped: dict, lower_is_better: bool) -> dict:
    """Intended against shipped on per-instance means. Ties are dropped; exact unless absolute differences tie."""
    common = sorted(set(intended) & set(shipped))
    diff = [intended[i] - shipped[i] for i in common]
    untied = [d for d in diff if abs(d) > 1e-12]
    good = (lambda d: d < 0) if lower_is_better else (lambda d: d > 0)
    out = {"instances": len(common), "untied": len(untied), "intended_minus_shipped": mean_ci(diff),
           "intended_better": int(sum(good(d) for d in untied)), "shipped_better": int(sum(not good(d) for d in untied)),
           "wilcoxon_p": None, "wilcoxon_method": None, "favours": None}
    if untied:
        exact = len({round(abs(d), 12) for d in untied}) == len(untied)
        out["wilcoxon_method"] = "exact" if exact else "approx"
        out["wilcoxon_p"] = float(wilcoxon(untied, method=out["wilcoxon_method"]).pvalue)
        ranks = np.argsort(np.argsort(np.abs(untied))) + 1.0
        for_intended = float(sum(r for r, d in zip(ranks, untied) if good(d)))
        out["rank_sum_for_intended"], out["rank_sum_for_shipped"] = for_intended, float(ranks.sum() - for_intended)
        out["favours"] = ("intended" if for_intended > ranks.sum() - for_intended
                          else "shipped" if for_intended < ranks.sum() - for_intended else None)
    return out


def summarise(rows) -> dict:
    out = {}
    for family in FAMILIES:
        for n in SIZES:
            cell = {}
            for arm in ARMS:
                sel = [r for r in rows if (r["family"], r["n"], r["arm"], r["repeat"]) == (family, n, arm, 0)]
                ok = [r for r in sel if "failed" not in r]
                entry = {"runs": len(sel), "failed": len(sel) - len(ok),
                         "optimal_runs": int(sum(r["optimal"] for r in ok)),
                         "init_schemes": sorted({r["init_scheme"] for r in ok}),
                         "initial_over_pi": sorted({json.dumps(r["initial_over_pi"], ensure_ascii=False) for r in ok}),
                         "state_p_optimum_over_uniform_median": float(np.median([r["state"]["p_optimum_over_uniform"] for r in ok])),
                         "state_initial_p_optimum_over_uniform_median": float(
                             np.median([r["state_initial"]["p_optimum_over_uniform"] for r in ok])),
                         "state_initial_approx_ratio": mean_ci(list(e19.per_instance(ok, "approx_ratio", "state_initial").values())),
                         "training_raised_approx_ratio_runs": int(sum(
                             r["state"]["approx_ratio"] > r["state_initial"]["approx_ratio"] for r in ok))}
                for name, (field, sub, _) in OUTCOMES.items():
                    entry[f"{name}_by_instance"] = e19.per_instance(ok, field, sub)
                    entry[name] = mean_ci(list(entry[f"{name}_by_instance"].values()))
                cell[arm] = entry
            for name, (_, _, lower_is_better) in OUTCOMES.items():
                cell[f"{name}_intended_vs_shipped"] = paired(cell["intended"][f"{name}_by_instance"],
                                                             cell["shipped"][f"{name}_by_instance"], lower_is_better)
            out[f"{family}|{n}"] = cell
    return out


def decide(summary: dict) -> dict:
    better, worse = [], []
    for key, cell in summary.items():
        for name in OUTCOMES:
            test = cell[f"{name}_intended_vs_shipped"]
            if test["wilcoxon_p"] is not None and test["wilcoxon_p"] <= BONFERRONI and test["favours"]:
                (better if test["favours"] == "intended" else worse).append(f"{key}|{name}")
    return {"threshold": BONFERRONI, "intended_better_at_threshold": better, "intended_worse_at_threshold": worse,
            "outcome": "a change to the intended order is proposed" if better and not worse
                       else "the product keeps the order it has"}


def reproduction(rows) -> dict:
    """The shipped arm against the stored rows of E21 and E23 (same instance, same seed)."""
    out = {}
    for family in FAMILIES:
        stored = json.loads((RESULTS / STORED[family]).read_text())["rows"]
        first = 10 if family == "e21" else 0
        before = {(r["n"], r["instance"] - first, r["rep"]): r for r in stored if not r["null"] and "failed" not in r}
        cell = {"runs": 0, "found": 0, "same_seed": 0, "same_result": 0, "same_final_state": 0, "first_differences": []}
        for r in rows:
            if (r["family"], r["arm"], r["repeat"]) != (family, "shipped", 0) or "failed" in r:
                continue
            cell["runs"] += 1
            old = before.get((r["n"], r["instance"], r["rep"]))
            if old is None:
                continue
            cell["found"] += 1
            same_result = old["optimal"] == r["optimal"] and abs(old["gap"] - r["gap"]) <= 1e-12
            same_state = abs(old["state"]["approx_ratio"] - r["state"]["approx_ratio"]) <= 1e-9
            cell["same_seed"] += int(old["seed"] == r["seed"])
            cell["same_result"] += int(same_result)
            cell["same_final_state"] += int(same_state)
            if not (same_result and same_state) and len(cell["first_differences"]) < 5:
                cell["first_differences"].append({"n": r["n"], "instance": r["instance"], "rep": r["rep"],
                                                  "gap": [old["gap"], r["gap"]],
                                                  "approx_ratio": [old["state"]["approx_ratio"], r["state"]["approx_ratio"]]})
        cell["reproduces"] = cell["runs"] > 0 and cell["runs"] == cell["found"] == cell["same_seed"] == cell[
            "same_result"] == cell["same_final_state"]
        out[family] = cell
    return out


def repeats_agree(rows) -> dict:
    """Harness check only: the same task run twice must return the same result."""
    first = {(r["family"], r["n"], r["instance"], r["rep"], r["arm"]): r for r in rows if r["repeat"] == 0}
    again = [r for r in rows if r["repeat"] == 1]
    same = sum("failed" not in r and "failed" not in first[(r["family"], r["n"], r["instance"], r["rep"], r["arm"])]
               and r["gap"] == first[(r["family"], r["n"], r["instance"], r["rep"], r["arm"])]["gap"]
               and r["state"]["approx_ratio"] == first[(r["family"], r["n"], r["instance"], r["rep"], r["arm"])]["state"]["approx_ratio"]
               for r in again)
    return {"repeated": len(again), "identical": int(same)}


def main() -> None:
    e20.check_defaults()
    if not QUICK:
        e20.require_committed_code()
    started = time.time()
    tasks = [(family, n, instance, rep, arm, repeat) for family in FAMILIES for n in SIZES
             for instance in range(INSTANCES) for rep in range(SEEDS) for arm in ARMS for repeat in range(REPEATS)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=2))
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} runs, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed[:10]:
        print("  failed:", r["family"], r["n"], r["instance"], r["rep"], r["arm"], r["failed"], flush=True)
    summary = summarise(rows)
    for key, cell in summary.items():
        s, i = cell["shipped"], cell["intended"]
        g, a = cell["gap_intended_vs_shipped"], cell["approx_ratio_intended_vs_shipped"]
        print(f"{key:>7}: optimal {i['optimal_runs']}/{i['runs']} intended vs {s['optimal_runs']}/{s['runs']} shipped"
              f" | gap {i['gap']['mean']:.4f} vs {s['gap']['mean']:.4f}, better/worse/untied {g['intended_better']}/{g['shipped_better']}/{g['untied']} p={g['wilcoxon_p']}"
              f" | final ratio {i['approx_ratio']['mean']:.3f} vs {s['approx_ratio']['mean']:.3f}, better/worse {a['intended_better']}/{a['shipped_better']} p={a['wilcoxon_p']}"
              f" | initial ratio {i['state_initial_approx_ratio']['mean']:.3f} vs {s['state_initial_approx_ratio']['mean']:.3f}"
              f" | training raised the ratio in {i['training_raised_approx_ratio_runs']} vs {s['training_raised_approx_ratio_runs']} runs"
              f" | schemes {i['init_schemes']} {s['init_schemes']}", flush=True)
    if QUICK:
        print("repeats:", repeats_agree(rows), flush=True)
        print("initial angles over pi:", {arm: summary[f"e21|{SIZES[0]}"][arm]["initial_over_pi"] for arm in ARMS}, flush=True)
        return
    repro = reproduction(rows)
    verdict = decide(summary)
    for family, cell in repro.items():
        print(f"shipped arm against the stored rows of {family}: {cell['same_result']} of {cell['runs']} results and"
              f" {cell['same_final_state']} final states identical; reproduces: {cell['reproduces']}", flush=True)
    print("decision:", verdict, flush=True)
    save("e25_qaoa_initial_angles",
         {"design": __doc__.split("Usage:")[0].strip(), "defaults": D, "sizes": SIZES, "instances": INSTANCES, "seeds": SEEDS,
          "families": {"e21": "instances 10..19 and seeds of E21", "e23": "instances 0..9 and seeds of E23"},
          "bonferroni_threshold": BONFERRONI, "shipped_arm_reproduces_stored_rows": repro, "decision": verdict,
          "summary": summary, "rows": rows, "wall_s": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
