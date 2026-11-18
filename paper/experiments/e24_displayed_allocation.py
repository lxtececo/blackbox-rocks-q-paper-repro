"""E24 - What reaches the user: the QAOA selection after the product's classical weighting.

Why. E20 to E23 score the selection a method returns. The product does not
display the selection; it displays weights and key figures. Between the two,
the worker's executor (api/executors.py, execute_quantum_qaoa_optimization, and
api/_executor_helpers.py, apply_quantum_degeneracy_guard) runs three classical
steps:
  1. minimum-variance weights on the selected assets (the classical weights of
     the whole universe if the selection holds fewer than two assets);
  2. a long-only mean-variance program, lambda w'Sw - m'w at the risk aversion
     of the request, with a cap of 40 % per asset and a cap on the Herfindahl
     index, on the assets that step 1 left with a positive weight;
  3. if the allocation is still concentrated (fewer than three positions above
     1 %, one position above 40 %, or a Herfindahl index above its cap), it is
     replaced by the minimum-variance portfolio of the five assets with the
     highest Sharpe ratio.
On 2026-11-17, on the default instance of the panel, two QAOA runs and three
runs of the D-Wave methods displayed the same allocation to two decimals. One
instance proves nothing. This experiment asks two questions. Does the selection
reach what the product displays? And is what the product displays better with
the selection step than without it?

Path. The executor itself, called with a job as the worker builds it for a
default simulator run from the panel: no limit on the number of assets, the
defaults of the table the executors read, the forecast step switched off (it
acts only on the pre-selection under a limit on the number of assets). Nothing
downstream of the selection is re-implemented here.

Instances. The 30 instances of E23 (the product's simulated market data, 90
days; 10 instances at each of 8, 12 and 16 assets). The default instance of the
panel, on which the observation was made, is not among them.

Arms, each through the same executor:
  method    QAOA, 5 seeds per instance (the seeds of E23)
  null      the ansatz replaced by one whose measured distribution is uniform
            for every parameter value, 5 seeds (the seeds of E23)
  all       the selection "every asset", put in the place of the quantum
            result: the product's classical steps alone
  optimum   the exact minimum of the QUBO the method builds, put in the place
            of the quantum result: what a perfect selection method would hand
            to the classical steps
  uniform   selections drawn uniformly, put in the place of the quantum
            result: every non-empty selection at 8 assets (255 per instance);
            1000 per instance, drawn without replacement with a fixed seed, at
            12 and 16 assets

Outcomes, computed from the displayed weights with the estimators of step 2
(annualised mean of the daily returns, Ledoit-Wolf covariance):
  J       lambda w'Sw - m'w at the risk aversion of the run (0.5), the
          objective by which the product chooses the displayed weights.
          Primary. Lower is better.
  Sharpe  m'w / sqrt(w'Sw). Secondary.
Two allocations count as the same if no weight differs by more than 0.0005
(0.05 percentage points; the result card prints two decimals of a per cent).
Two values of J closer than 1e-6 count as a tie (1e-6 also for the Sharpe ratio).

Tests (stated before the run), per size, on per-instance means, Wilcoxon
signed-rank, two-sided:
  T1  J of the method against J of its null
  T2  J of the method against J of "all"
Six comparisons; p-values are reported unadjusted next to the Bonferroni
threshold 0.05 / 6. Both tests are two-sided: the selection step may help or
hurt.

What will be concluded.
  "The selection reaches the user" at a size if T1 is below the threshold with
  the method better.
  "The selection step improves what is displayed" at a size if T2 is below the
  threshold with the method better; "it worsens what is displayed" if T2 is
  below the threshold with "all" better. Otherwise neither is concluded.

Described, not tested: the share of runs whose displayed allocation is the
same as that of "all" (method, null, optimum, uniform); how often each
classical step rewrote the allocation and how often the displayed assets are
the selected ones; J of the optimum and of the null against "all"; the share of
uniform selections whose J is below that of "all", and below that of the
method; the rank correlation
(Spearman), within an instance, between the QUBO cost of a uniform selection
and the J of what is displayed for it; the Sharpe ratio under the same two
tests, marked secondary; and whether each method and null run returns the
selection that scored as E23 stored for the same seed.

Limits stated in advance. One method: the executors of CVaR-VQE, PCE and
BF-DCQO end in the same steps 2 and 3 after a step 1 of their own, and are not
measured. One instance family. No limit on the number of assets: under a limit
the executor pre-selects the pool classically and the question changes. J and
the Sharpe ratio are in-sample figures on 90 days of simulated prices; nothing
here says how any of these portfolios would perform.

Usage: e24_displayed_allocation.py [--quick]
  --quick   2 factor-model instances (not used by any experiment) x 2 seeds at 8 assets,
            40 uniform selections; checks the harness, saves nothing
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import RESULTS, mean_ci, point_seed, pool_workers, save  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

import e19_portfolio as e19  # noqa: E402  (enumeration, scores, the uniform ansatz)
import e20_portfolio_corrected as e20  # noqa: E402  (defaults, the referee check)
import e23_qaoa_product_data as e23  # noqa: E402  (the instances)

QUICK = "--quick" in sys.argv
SIZES = [8] if QUICK else [8, 12, 16]
INSTANCES = 2 if QUICK else 10
SEEDS = 2 if QUICK else 5
UNIFORM = {8: 40 if QUICK else None, 12: 1000, 16: 1000}  # None: every non-empty selection
BONFERRONI = 0.05 / 6
SAME_WEIGHT = 5e-4
TIE = 1e-6
D = e20.DEFAULTS["qaoa"]
LAMBDA = D["risk_aversion"]
RAW = RESULTS / "raw" / "e24_uniform_selections.json.gz"


def require_committed() -> None:
    """The evaluation measures committed code: the backend sources and the experiment scripts."""
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    watched = ["backend/src/main/python", "reports/paper/quantumfpo-arxiv/experiments"]
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *watched],
                                     text=True, cwd=root)
    changed = [line for line in status.splitlines() if line[3:].strip('"').endswith(".py")]
    if changed:
        sys.exit("not run: uncommitted code\n" + "\n".join(changed))


def instance_rows(n: int, instance: int):
    """The stock rows of the request, and the price table as the executor builds it from them."""
    source = e19.make_prices(n, 9000 + instance) if QUICK else e23.prices_of(n, instance)
    rows = [{"symbol": str(symbol), "date": str(pd.Timestamp(day).date()), "close": float(value)}
            for symbol in source.columns for day, value in source[symbol].items()]
    prices = pd.DataFrame(rows).pivot(index="date", columns="symbol", values="close").sort_index()
    return rows, prices


def job_for(rows, seed):
    from api.models import EnhancedOptimizationJob, JobPriority, JobStatus, OptimizationMethod

    request = {"stock_data": rows, "var_percent": 0.05, "qc_simulator": True, "enable_ml_prediction": False}
    if seed is not None:
        request["qaoaConfig"] = {"seed": seed}
    return EnhancedOptimizationJob(
        job_id="e24", method=OptimizationMethod.QUANTUM_QAOA, status=JobStatus.RUNNING, priority=JobPriority.NORMAL,
        created_at=datetime.utcnow(), started_at=datetime.utcnow(), tenant_id="00000000-0000-0000-0000-000000000001",
        request_data=request)


def displayed(api_result, symbols, mu, cov) -> dict:
    """What the product displays for one run, and the two outcomes computed from the displayed weights."""
    allocation = api_result["allocation"]
    w = np.array([float(allocation.get(s, 0.0) or 0.0) for s in symbols])
    ret, var = float(w @ mu), float(w @ cov @ w)
    provenance = api_result["selection_provenance"]
    guard = api_result.get("degeneracy_guard") or {}
    return {"weights": [round(float(x), 6) for x in w],
            "J": LAMBDA * var - ret, "sharpe": float(ret / np.sqrt(var)) if var > 0 else 0.0,
            "shown": {key: float(api_result[key]) for key in ("expected_return", "portfolio_risk", "sharpe_ratio")},
            "selected": len(provenance["quantum_selection"]),
            "held_above_1pct": int((w > 0.01).sum()),
            "displayed_assets_are_the_selection": bool(provenance["final_equals_quantum_selection"]),
            "rewritten_by": [step["step"] for step in provenance["allocation_rewritten_by"]],
            "guard_reason": guard.get("reason") if guard.get("applied") else None}


def same(a: dict, b: dict) -> bool:
    return bool(np.max(np.abs(np.asarray(a["weights"]) - np.asarray(b["weights"]))) <= SAME_WEIGHT)


def run_instance(task):
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa
    from api import executors
    from core.preprocessing import ledoit_wolf_covariance

    n, instance = task
    logging.disable(logging.CRITICAL)
    started = time.time()
    rows, prices = instance_rows(n, instance)
    symbols = list(prices.columns)
    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252.0
    cov = np.asarray(ledoit_wolf_covariance(returns, annualization_factor=252.0)[0], float)
    linear, quadratic, _ = qaoa._build_qubo(prices, LAMBDA)
    cost = e19.pair_costs(linear, quadratic, n)
    out = {"n": n, "instance": instance, "symbols": symbols, "failed": []}

    def call(seed=None, null=False, bits=None):
        """One executor call. ``bits``: a selection put in the place of the quantum result."""
        saved = qaoa.QAOAAnsatz, qaoa.quantum_optimize
        if null:
            qaoa.QAOAAnsatz = lambda cost_operator, reps=1, **_: e19.uniform_ansatz(cost_operator.num_qubits, 2 * reps)
        if bits is not None:
            qaoa.quantum_optimize = lambda *args, **kwargs: {"solution": [int(b) for b in bits]}
        try:
            with e20.quiet():
                return executors.execute_quantum_qaoa_optimization(job_for(rows, seed))
        finally:
            qaoa.QAOAAnsatz, qaoa.quantum_optimize = saved

    def injected(index: int):
        result = call(bits=[(index >> i) & 1 for i in range(n)])
        return {"index": int(index), "qubo_cost": float(cost[index]), **displayed(result, symbols, mu, cov)}

    try:
        out["all"] = injected(2 ** n - 1)
        out["optimum"] = {**injected(int(np.argmin(cost))), **e19.score(cost, int(np.argmin(cost)))}
    except Exception as exc:  # an instance without its references cannot be analysed
        out["failed"].append(f"references: {type(exc).__name__}: {exc}"[:300])
        out["wall_s"] = round(time.time() - started, 1)
        return out
    out["optimum"]["same_as_all"] = same(out["optimum"], out["all"])

    out["runs"] = []
    for null in (False, True):
        for rep in range(SEEDS):
            seed = int(point_seed(23, 0, n, instance, rep, int(null)) % (2 ** 31 - 1))
            row = {"arm": "null" if null else "method", "rep": rep, "seed": seed}
            try:
                result = call(seed=seed, null=null)
                index = e19.index_of(result["quantum_solution"])
                row.update(index=index, qubo_cost=float(cost[index]))
                row.update(e19.score(cost, index))
                row.update(displayed(result, symbols, mu, cov))
                row["referee_agrees"] = e20.referee_agrees(result["exact_reference"], row)
                row["same_as_all"] = same(row, out["all"])
            except Exception as exc:  # recorded, never dropped
                row["failed"] = f"{type(exc).__name__}: {exc}"[:300]
            out["runs"].append(row)

    count = UNIFORM[n]
    if count is None:
        indices = np.arange(1, 2 ** n)
    else:
        indices = np.random.default_rng(point_seed(24, 1, n, instance)).choice(2 ** n - 1, size=count, replace=False) + 1
    uniform = {"index": [], "qubo_cost": [], "J": [], "sharpe": [], "same_as_all": [], "guard": [], "selected": []}
    for index in indices:
        try:
            row = injected(int(index))
        except Exception as exc:  # recorded, never dropped
            out["failed"].append(f"uniform {int(index)}: {type(exc).__name__}: {exc}"[:300])
            continue
        uniform["index"].append(int(index))
        uniform["qubo_cost"].append(row["qubo_cost"])
        uniform["J"].append(row["J"])
        uniform["sharpe"].append(row["sharpe"])
        uniform["same_as_all"].append(int(same(row, out["all"])))
        uniform["guard"].append(int("degeneracy_guard" in row["rewritten_by"]))
        uniform["selected"].append(row["selected"])
    out["uniform_raw"] = uniform
    j, j_all = np.asarray(uniform["J"]), out["all"]["J"]
    rho = spearmanr(uniform["qubo_cost"], uniform["J"])
    out["uniform"] = {
        "selections": len(j), "exhaustive": count is None,
        "share_J_below_all": float(np.mean(j < j_all - TIE)),
        "share_J_tied_with_all": float(np.mean(np.abs(j - j_all) <= TIE)),
        "share_J_above_all": float(np.mean(j > j_all + TIE)),
        "share_same_allocation_as_all": float(np.mean(uniform["same_as_all"])),
        "share_guard_replaced": float(np.mean(uniform["guard"])),
        "J_median": float(np.median(j)), "J_min": float(j.min()),
        "spearman_qubo_cost_vs_J": float(rho.statistic)}
    out["wall_s"] = round(time.time() - started, 1)
    return out


def paired(a: dict, b: dict) -> dict:
    """Wilcoxon signed-rank on per-instance values, two-sided; differences within TIE are ties."""
    common = sorted(set(a) & set(b))
    diff = [0.0 if abs(a[i] - b[i]) <= TIE else a[i] - b[i] for i in common]
    return {"instances": len(common), "first_minus_second": mean_ci([a[i] - b[i] for i in common]),
            "first_lower": int(sum(d < 0 for d in diff)), "first_higher": int(sum(d > 0 for d in diff)),
            "ties": int(sum(d == 0 for d in diff)),
            "wilcoxon_p": float(wilcoxon(diff).pvalue) if any(diff) else None}


def summarise(instances) -> dict:
    stored = {} if QUICK else {(r["n"], r["instance"], r["rep"], r["null"]): r for r in json.loads(
        (RESULTS / "e23_qaoa_product_data.json").read_text())["rows"]}
    out = {}
    for n in SIZES:
        cells = [c for c in instances if c["n"] == n and "runs" in c]
        cell = {"instances": len(cells), "instances_without_references": int(sum(c["n"] == n and "runs" not in c for c in instances)),
                "failed_calls": int(sum(len(c["failed"]) for c in cells))}
        values = {"J": {}, "sharpe": {}}
        for arm in ("method", "null"):
            runs = [(c, r) for c in cells for r in c["runs"] if r["arm"] == arm]
            ok = [(c, r) for c, r in runs if "failed" not in r]
            entry = {"runs": len(runs), "failed": len(runs) - len(ok),
                     "optimal_selections": int(sum(r["optimal"] for _, r in ok)),
                     "referee_disagreements": int(sum(not r["referee_agrees"] for _, r in ok)),
                     "same_allocation_as_all": int(sum(r["same_as_all"] for _, r in ok)),
                     "displayed_assets_are_the_selection": int(sum(r["displayed_assets_are_the_selection"] for _, r in ok)),
                     "rewritten_by": {step: int(sum(step in r["rewritten_by"] for _, r in ok))
                                      for step in ("classical_weights", "classical_refinement", "degeneracy_guard")},
                     "selected_mean": float(np.mean([r["selected"] for _, r in ok])),
                     "held_above_1pct_mean": float(np.mean([r["held_above_1pct"] for _, r in ok]))}
            if stored:
                entry["scores_as_e23_stored"] = int(sum(
                    abs(r["gap"] - stored[(n, c["instance"], r["rep"], arm == "null")]["gap"]) <= 1e-12
                    and r["selected"] == stored[(n, c["instance"], r["rep"], arm == "null")]["selected"] for c, r in ok))
            for field in ("J", "sharpe"):
                by = {}
                for c, r in ok:
                    by.setdefault(c["instance"], []).append(r[field])
                values[field][arm] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
                entry[field] = mean_ci(list(values[field][arm].values()))
            cell[arm] = entry
        for arm in ("all", "optimum"):
            entry = {"held_above_1pct_mean": float(np.mean([c[arm]["held_above_1pct"] for c in cells])),
                     "selected_mean": float(np.mean([c[arm]["selected"] for c in cells])),
                     "rewritten_by": {step: int(sum(step in c[arm]["rewritten_by"] for c in cells))
                                      for step in ("classical_weights", "classical_refinement", "degeneracy_guard")}}
            for field in ("J", "sharpe"):
                values[field][arm] = {c["instance"]: float(c[arm][field]) for c in cells}
                entry[field] = mean_ci(list(values[field][arm].values()))
            cell[arm] = entry
        cell["optimum"]["same_allocation_as_all"] = int(sum(c["optimum"]["same_as_all"] for c in cells))
        for field in ("J", "sharpe"):
            cell[f"{field}_tests"] = {"T1_method_vs_null": paired(values[field]["method"], values[field]["null"]),
                                      "T2_method_vs_all": paired(values[field]["method"], values[field]["all"]),
                                      "described_null_vs_all": paired(values[field]["null"], values[field]["all"]),
                                      "described_optimum_vs_all": paired(values[field]["optimum"], values[field]["all"])}
        cell["uniform"] = {key: {"median": float(np.median([c["uniform"][key] for c in cells])),
                                 "min": float(min(c["uniform"][key] for c in cells)),
                                 "max": float(max(c["uniform"][key] for c in cells))}
                           for key in ("share_J_below_all", "share_J_tied_with_all", "share_J_above_all",
                                       "share_same_allocation_as_all", "share_guard_replaced", "spearman_qubo_cost_vs_J")}
        cell["uniform"]["selections_per_instance"] = sorted({c["uniform"]["selections"] for c in cells})
        cell["uniform"]["method_J_percentile_among_uniform"] = mean_ci([
            100.0 * float(np.mean(np.asarray(c["uniform_raw"]["J"]) < values["J"]["method"][c["instance"]] - TIE)) for c in cells])
        out[str(n)] = cell
    return out


def main() -> None:
    e20.check_defaults()
    if not QUICK:
        require_committed()
    started = time.time()
    tasks = [(n, instance) for n in SIZES for instance in range(INSTANCES)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        instances = list(pool.map(run_instance, tasks))
    print(f"{len(instances)} instances, {time.time() - started:.0f} s", flush=True)
    for c in instances:
        for failure in c["failed"][:3]:
            print("  failed:", c["n"], c["instance"], failure, flush=True)
        for r in c.get("runs", []):
            if "failed" in r:
                print("  failed:", c["n"], c["instance"], r["arm"], r["rep"], r["failed"], flush=True)
    summary = summarise(instances)
    for n, cell in summary.items():
        m, u, a, o, t, uni = cell["method"], cell["null"], cell["all"], cell["optimum"], cell["J_tests"], cell["uniform"]
        t1, t2 = t["T1_method_vs_null"], t["T2_method_vs_all"]
        print(f"{n:>2} assets: J method {m['J']['mean']:.4f} null {u['J']['mean']:.4f} all {a['J']['mean']:.4f} optimum {o['J']['mean']:.4f}"
              f" | T1 lower/higher/tied {t1['first_lower']}/{t1['first_higher']}/{t1['ties']} p={t1['wilcoxon_p']}"
              f" | T2 {t2['first_lower']}/{t2['first_higher']}/{t2['ties']} p={t2['wilcoxon_p']}"
              f" | same as all: method {m['same_allocation_as_all']}/{m['runs']} null {u['same_allocation_as_all']}/{u['runs']}"
              f" optimum {o['same_allocation_as_all']}/{cell['instances']} uniform median {uni['share_same_allocation_as_all']['median']:.3f}"
              f" | guard: method {m['rewritten_by']['degeneracy_guard']} null {u['rewritten_by']['degeneracy_guard']}"
              f" | uniform below all {uni['share_J_below_all']['median']:.3f}, spearman {uni['spearman_qubo_cost_vs_J']['median']:.2f}"
              f" | Sharpe method {m['sharpe']['mean']:.3f} null {u['sharpe']['mean']:.3f} all {a['sharpe']['mean']:.3f}"
              f" | as E23: {m.get('scores_as_e23_stored')}/{m['runs']}, {u.get('scores_as_e23_stored')}/{u['runs']}"
              f" | failed calls {cell['failed_calls']}", flush=True)
    if QUICK:
        return
    raw = {f"{c['n']}:{c['instance']}": c.pop("uniform_raw") for c in instances if "uniform_raw" in c}
    RAW.parent.mkdir(exist_ok=True)
    with gzip.open(RAW, "wt") as handle:
        json.dump(raw, handle)
    save("e24_displayed_allocation",
         {"design": __doc__.split("Usage:")[0].strip(), "defaults": D, "sizes": SIZES, "instances_per_size": INSTANCES,
          "seeds": SEEDS, "uniform_selections": {str(k): ("every non-empty selection" if v is None else v) for k, v in UNIFORM.items()},
          "same_weight_tolerance": SAME_WEIGHT, "tie_tolerance": TIE, "bonferroni_threshold": BONFERRONI,
          "uniform_raw_file": str(RAW.relative_to(RESULTS)), "summary": summary, "instances": instances,
          "wall_s": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
