"""E23 - QAOA on the simulated market data of the product itself.

Why. E19, E20 and E21 use one instance family: a factor model with 504 trading
days. A default run from the product's panel uses another: the simulated
market data of the gateway (90 days, daily volatility of 1.5 to 4 %, occasional
jumps, one series per symbol name). One such run, seen on 2026-11-17 while the
deployed panel was being checked, returned a selection with 8 cheaper
alternatives among 256, where 200 uniform samples read by the same rule leave
0.84 on average. One run proves nothing. This experiment asks the question
properly: does the result of E21 hold on the data a new user sees first?

Data. Simulated series fetched once from the gateway (POST /api/stocks/load,
simulated, 90 days) for 360 symbol names that exist nowhere else
(SIM_E23_000 .. SIM_E23_359); the generator seeds a series from its symbol
name. They are stored in data/e23_simulated_market.json with the time of the
fetch. The experiment reads that file and needs no cluster.

Instances: 10 at each of 8, 12 and 16 assets, on disjoint symbols.

Everything else is E21: the product's simulator path with its own ``seed``
argument and the defaults of the table the worker executors read, 5 seeds per
instance, the null (the ansatz replaced by one whose measured distribution is
uniform for every parameter value), the referee (exact minimum over all 2^n
selections of the QUBO the method builds), the gap of the method against its
null paired by instance, Wilcoxon signed-rank on per-instance means. Three
comparisons; p-values are reported unadjusted next to the Bonferroni
threshold 0.05 / 3. The test is two-sided: the method may be worse than its
null, and the run that prompted this experiment was.

Described, not tested: the share of optimal runs next to what exact uniform
sampling predicts; the exact final state (probability of the optimum against
2^-n, approximation ratio) and the state at the initial angles; and, for this
family and for the instances of E21, three numbers that describe the cost
function: the ratio of the largest linear to the largest pair coefficient of
the QUBO, the number of assets in the optimum, and the spread of the cost
(maximum minus minimum) in units of pi, which is the phase the first cost
layer can imprint at gamma = 1.

Usage: e23_qaoa_product_data.py [--fetch] [--quick]
  --fetch   fetch and store the data (needs the dev gateway); runs nothing
  --quick   2 instances x 2 seeds at 8 assets (checks the harness; not saved as a result)
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

import e19_portfolio as e19  # noqa: E402  (enumeration, scores, the uniform ansatz)
import e20_portfolio_corrected as e20  # noqa: E402  (defaults, the paired test, the guards)

QUICK = "--quick" in sys.argv
SIZES = [8] if QUICK else [8, 12, 16]
INSTANCES = 2 if QUICK else 10
SEEDS = 2 if QUICK else 5
BONFERRONI = 0.05 / 3
DAYS = 90
SYMBOLS = [f"SIM_E23_{k:03d}" for k in range(360)]
DATA = RESULTS.parent / "data" / "e23_simulated_market.json"
D = e20.DEFAULTS["qaoa"]
# --fetch only. The address of the development gateway and its demo account were constants here when the
# data were fetched on 2026-11-17. They were replaced by environment variables before the script was
# published; the stored data file records its source. The experiment itself reads that file only.
GATEWAY = os.environ.get("QFPO_GATEWAY", "")


def fetch() -> None:
    import httpx

    user, password = os.environ.get("QFPO_USER"), os.environ.get("QFPO_PASSWORD")
    if not (GATEWAY and user and password):
        raise SystemExit("--fetch needs QFPO_GATEWAY, QFPO_USER and QFPO_PASSWORD: a gateway of the platform and an account on it")
    with httpx.Client(base_url=GATEWAY, verify=False, timeout=300.0) as client:
        login = client.post("/api/v3/auth/login", json={"email": user, "password": password})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        series = {}
        for start in range(0, len(SYMBOLS), 40):
            chunk = SYMBOLS[start:start + 40]
            response = client.post("/api/stocks/load", headers=headers, json={
                "stocks": chunk, "period": DAYS, "stockAmount": len(chunk), "symbols": chunk, "simulated": True})
            response.raise_for_status()
            for row in response.json():
                series.setdefault(row["symbol"], []).append((row["date"], float(row["close"])))
    if sorted(series) != SYMBOLS or any(len(v) != DAYS for v in series.values()):
        raise SystemExit(f"expected {len(SYMBOLS)} series of {DAYS} days, got {len(series)}")
    dates = sorted({d for v in series.values() for d, _ in v})
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps({
        "source": f"{GATEWAY} POST /api/stocks/load, simulated, period {DAYS}",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dates": dates,
        "close": {symbol: [price for _, price in sorted(series[symbol])] for symbol in SYMBOLS}}))
    print(f"stored {len(SYMBOLS)} series of {len(dates)} dates in {DATA}")


_DATA = {}


def prices_of(n: int, instance: int) -> pd.DataFrame:
    """Instance ``instance`` of size n: its own n symbols, disjoint from every other instance."""
    if not _DATA:
        _DATA.update(json.loads(DATA.read_text()))
    first = {8: 0, 12: 80, 16: 200}[n] + n * instance
    symbols = SYMBOLS[first:first + n]
    return pd.DataFrame({s: _DATA["close"][s] for s in symbols}, index=pd.to_datetime(_DATA["dates"]))


def cost_shape(linear, quadratic, cost) -> dict:
    pair = np.abs(np.triu(np.asarray(quadratic, float), 1))
    best = int(np.argmin(cost))
    return {"linear_over_pair": float(np.abs(linear).max() / pair.max()),
            "assets_in_optimum": int(bin(best).count("1")),
            "cost_spread_over_pi": float((cost.max() - cost.min()) / np.pi)}


def run_task(task):
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    n, instance, rep, null = task
    logging.disable(logging.CRITICAL)
    prices = prices_of(n, instance)
    seed = int(point_seed(23, 0, n, instance, rep, int(null)) % (2 ** 31 - 1))
    started = time.time()
    saved = qaoa.QAOAAnsatz
    if null:
        qaoa.QAOAAnsatz = lambda cost_operator, reps=1, **_: e19.uniform_ansatz(cost_operator.num_qubits, 2 * reps)
    try:
        with e20.quiet():
            result = qaoa.quantum_optimize(prices, risk_aversion=D["risk_aversion"], qc_simulator_mode=True,
                                           estimator_shots=D["estimator_shots"], sampler_shots=D["sampler_shots"],
                                           max_iterations=D["max_iterations"], reps=D["reps"], seed=seed)
        linear, quadratic, _ = qaoa._build_qubo(prices, D["risk_aversion"])
        cost = e19.pair_costs(linear, quadratic, n)
        out = e19.score(cost, e19.index_of(result["solution"]))
        reference = result["exact_reference"]
        out["referee_agrees"] = e20.referee_agrees(reference, out)
        out["uniform_p_returns_an_optimum"] = reference["uniform_null"]["p_returns_an_optimum"]
        out["uniform_expected_rank_pct"] = reference["uniform_null"]["expected_better_selections_pct"]
        out["init_scheme"] = result["warm_start"]["init_scheme"]
        out["cost_shape"] = cost_shape(linear, quadratic, cost)
        hamiltonian = qaoa._build_hamiltonian(linear, quadratic, n)
        ansatz = (e19.uniform_ansatz(n, 2 * D["reps"]) if null else QAOAAnsatz(cost_operator=hamiltonian, reps=D["reps"]))
        pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
        out["state"] = e19.state_metrics(e19.exact_probabilities(pm.run(ansatz.assign_parameters(result["final_parameters"]))), cost)
        if not null:
            initial = qaoa._interp_init_params(ansatz.num_parameters, D["reps"])
            out["state_initial"] = e19.state_metrics(e19.exact_probabilities(pm.run(ansatz.assign_parameters(initial))), cost)
    except Exception as exc:  # recorded, never dropped
        out = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        qaoa.QAOAAnsatz = saved
    out.update(n=n, instance=instance, rep=rep, null=null, seed=seed, wall_s=round(time.time() - started, 2))
    return out


def e21_cost_shape(n: int) -> list:
    """The same three numbers for the ten instances of E21 at size n (no run needed)."""
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa

    out = []
    for instance in range(10, 20):
        prices = e19.make_prices(n, 1000 + 100 * (n // 4) + instance)
        linear, quadratic, _ = qaoa._build_qubo(prices, D["risk_aversion"])
        out.append(cost_shape(linear, quadratic, e19.pair_costs(linear, quadratic, n)))
    return out


def shape_summary(shapes: list) -> dict:
    return {key: {"median": float(np.median([s[key] for s in shapes])), "min": float(min(s[key] for s in shapes)),
                  "max": float(max(s[key] for s in shapes))} for key in shapes[0]}


def summarise(rows) -> dict:
    out = {}
    for n in SIZES:
        cell = {}
        for null in (False, True):
            sel = [r for r in rows if r["n"] == n and r["null"] == null]
            ok = [r for r in sel if "failed" not in r]
            entry = {"runs": len(sel), "failed": len(sel) - len(ok),
                     "optimal_runs": int(sum(r["optimal"] for r in ok)),
                     "referee_disagreements": int(sum(not r["referee_agrees"] for r in ok)),
                     "init_schemes": sorted({r["init_scheme"] for r in ok}),
                     "state_p_optimum_over_uniform_median": float(np.median([r["state"]["p_optimum_over_uniform"] for r in ok])),
                     "state_approx_ratio": mean_ci(list(e19.per_instance(ok, "approx_ratio", "state").values()))}
            for field in ("gap", "rank_pct", "optimal", "selected"):
                entry[field] = mean_ci(list(e19.per_instance(ok, field).values()))
                entry[f"{field}_by_instance"] = e19.per_instance(ok, field)
            if null:
                entry["optimal_runs_expected_from_exact_uniform_sampling"] = float(sum(r["uniform_p_returns_an_optimum"] for r in ok))
                entry["rank_pct_expected_from_exact_uniform_sampling"] = float(np.mean([r["uniform_expected_rank_pct"] for r in ok]))
            else:
                entry["state_initial_p_optimum_over_uniform_median"] = float(
                    np.median([r["state_initial"]["p_optimum_over_uniform"] for r in ok]))
                entry["state_initial_approx_ratio"] = mean_ci(list(e19.per_instance(ok, "approx_ratio", "state_initial").values()))
                entry["training_raised_approx_ratio_runs"] = int(sum(
                    r["state"]["approx_ratio"] > r["state_initial"]["approx_ratio"] for r in ok))
            cell["null" if null else "method"] = entry
        cell["method_vs_null"] = e20.paired_test(cell["method"]["gap_by_instance"], cell["null"]["gap_by_instance"])
        first = {}
        for r in rows:
            if r["n"] == n and "failed" not in r:
                first.setdefault(r["instance"], r["cost_shape"])
        cell["cost_shape"] = {"this_family": shape_summary(list(first.values())), "e21_family": shape_summary(e21_cost_shape(n))}
        out[str(n)] = cell
    return out


def main() -> None:
    if "--fetch" in sys.argv:
        fetch()
        return
    e20.check_defaults()
    if not QUICK:
        e20.require_committed_code()
    started = time.time()
    tasks = [(n, instance, rep, null) for n in SIZES for instance in range(INSTANCES)
             for rep in range(SEEDS) for null in (False, True)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=2))
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} runs, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed[:10]:
        print("  failed:", r["n"], r["instance"], r["rep"], r["null"], r["failed"], flush=True)
    summary = summarise(rows)
    for n, cell in summary.items():
        m, u, t, s = cell["method"], cell["null"], cell["method_vs_null"], cell["cost_shape"]
        print(f"{n:>2} assets: gap {m['gap']['mean']:.4f}±{m['gap']['ci95']:.4f} vs null {u['gap']['mean']:.4f}±{u['gap']['ci95']:.4f}"
              f" better/worse/tied {t['method_better']}/{t['null_better']}/{t['ties']} p={t['wilcoxon_p']}"
              f" optimal {m['optimal_runs']}/{m['runs']} vs {u['optimal_runs']}/{u['runs']}"
              f" (exact uniform sampling predicts {u['optimal_runs_expected_from_exact_uniform_sampling']:.1f})"
              f" | state P(opt)/uniform median {m['state_p_optimum_over_uniform_median']:.2f}"
              f" (initial {m['state_initial_p_optimum_over_uniform_median']:.2f}), ratio {m['state_approx_ratio']['mean']:.3f}"
              f" (initial {m['state_initial_approx_ratio']['mean']:.3f})"
              f" | linear/pair {s['this_family']['linear_over_pair']['median']:.1f} (E21 {s['e21_family']['linear_over_pair']['median']:.1f}),"
              f" spread/pi {s['this_family']['cost_spread_over_pi']['median']:.2f} (E21 {s['e21_family']['cost_spread_over_pi']['median']:.2f})"
              f" | referee disagreements {m['referee_disagreements'] + u['referee_disagreements']}", flush=True)
    if QUICK:
        return
    data = json.loads(DATA.read_text())
    save("e23_qaoa_product_data",
         {"design": __doc__.split("Usage:")[0].strip(), "defaults": D, "sizes": SIZES, "instances": INSTANCES, "seeds": SEEDS,
          "data": {"source": data["source"], "fetched_utc": data["fetched_utc"], "days": len(data["dates"]), "symbols": len(data["close"])},
          "bonferroni_threshold": BONFERRONI, "summary": summary, "rows": rows, "wall_s": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
