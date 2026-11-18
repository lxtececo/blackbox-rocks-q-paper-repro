"""E21 - QAOA with the lowest-cost rule on instances and seeds that no earlier experiment used.

Why: the rule the portfolio QAOA now applies (return the cheapest sampled
selection instead of the most frequent one) was chosen from the samples E19
recorded, and the QAOA arm of E20 re-scores those same samples by design. That
arm shows that the corrected code returns what the analysis of E19 predicted;
it is not an independent test of the rule. This experiment is one.

What is the same as in E19 and E20: the instance generator, the sizes (8, 12
and 16 assets), 10 instances per size and 5 seeds each, the referee (exact
minimum over all 2^n selections of the QUBO the method builds), the null
(the ansatz replaced by one whose measured distribution is uniform for every
parameter value, so that training, sampling and selection run unchanged), the
outcome measures and the test (gap of the method against its null, paired by
instance, Wilcoxon signed-rank on per-instance means).

What differs
 - Instances: generator seeds 1000 + 100 (n // 4) + i for i = 10..19. E19 and
   E20 used i = 0..9, the tuning grids 5000-based seeds, the harness checks
   900-based ones.
 - Seeds: the product's own ``seed`` argument, from point_seed(21, ...). Nothing
   is substituted around the product code except the ansatz of the null.
 - The defaults are those of the table the worker executors read; the script
   stops if they differ.

Three comparisons (one per size); p-values are reported unadjusted next to the
Bonferroni threshold 0.05 / 3. With ten instances the smallest two-sided
p-value the test can return is 2 / 1024.

Described, not tested: the share of optimal runs; the exact final state
(probability of the optimum against 2^-n, approximation ratio); for the null,
the number of optimal runs next to the number that the exact probability of
"200 uniform samples contain an optimum" predicts; whether the exact reference
in the product's result agrees with the score computed here.

Usage: e21_qaoa_fresh.py [--quick]
  --quick   2 instances x 2 seeds, n = 8, on other instances (checks the harness)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import mean_ci, point_seed, pool_workers, save  # noqa: E402

import e19_portfolio as e19  # noqa: E402  (instances, enumeration, scores, the uniform ansatz)
import e20_portfolio_corrected as e20  # noqa: E402  (defaults, the paired test, the guards)

QUICK = "--quick" in sys.argv
SIZES = [8] if QUICK else [8, 12, 16]
INSTANCES = 2 if QUICK else 10
SEEDS = 2 if QUICK else 5
FIRST_INSTANCE = 90 if QUICK else 10          # E19 and E20 used 0..9
INSTANCE_BASE = 1000
BONFERRONI = 0.05 / 3
D = e20.DEFAULTS["qaoa"]


def run_task(task):
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    n, instance, rep, null = task
    logging.disable(logging.CRITICAL)
    prices = e19.make_prices(n, INSTANCE_BASE + 100 * (n // 4) + instance)
    seed = int(point_seed(21, 0, n, instance, rep, int(null)) % (2 ** 31 - 1))
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
        out["sampled_shots"] = int(result["selection"]["shots"])
        out["most_frequent_is_selected"] = bool(result["selection"]["most_frequent_is_selected"])
        hamiltonian = qaoa._build_hamiltonian(linear, quadratic, n)
        # The circuit the product trained, without its measurements: same class, same parameter order.
        # Transpiled to basis gates as the product does it; the abstract evolution gates would otherwise
        # be exponentiated as matrices of dimension 2^n.
        ansatz = (e19.uniform_ansatz(n, 2 * D["reps"]) if null else QAOAAnsatz(cost_operator=hamiltonian, reps=D["reps"]))
        final = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1).run(
            ansatz.assign_parameters(result["final_parameters"]))
        out["state"] = e19.state_metrics(e19.exact_probabilities(final), cost)
    except Exception as exc:  # recorded, never dropped
        out = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        qaoa.QAOAAnsatz = saved
    out.update(n=n, instance=instance, rep=rep, null=null, seed=seed, wall_s=round(time.time() - started, 2))
    return out


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
                     "most_frequent_is_selected_runs": int(sum(r["most_frequent_is_selected"] for r in ok)),
                     "state_p_optimum_over_uniform_median": float(np.median([r["state"]["p_optimum_over_uniform"] for r in ok])),
                     "state_approx_ratio": mean_ci(list(e19.per_instance(ok, "approx_ratio", "state").values()))}
            for field in ("gap", "rank_pct", "optimal", "selected"):
                entry[field] = mean_ci(list(e19.per_instance(ok, field).values()))
                entry[f"{field}_by_instance"] = e19.per_instance(ok, field)
            if null:
                # What the exact probability of "the uniform samples contain an optimum" predicts for these runs.
                entry["optimal_runs_expected_from_exact_uniform_sampling"] = float(sum(r["uniform_p_returns_an_optimum"] for r in ok))
            cell["null" if null else "method"] = entry
        cell["method_vs_null"] = e20.paired_test(cell["method"]["gap_by_instance"], cell["null"]["gap_by_instance"])
        out[str(n)] = cell
    return out


def main() -> None:
    e20.check_defaults()
    if not QUICK:
        e20.require_committed_code()
    started = time.time()
    tasks = [(n, instance, rep, null) for n in SIZES for instance in range(FIRST_INSTANCE, FIRST_INSTANCE + INSTANCES)
             for rep in range(SEEDS) for null in (False, True)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=2))
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} runs, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed[:10]:
        print("  failed:", r["n"], r["instance"], r["rep"], r["null"], r["failed"], flush=True)
    summary = summarise(rows)
    for n, cell in summary.items():
        m, u, t = cell["method"], cell["null"], cell["method_vs_null"]
        print(f"{n:>2} assets: gap {m['gap']['mean']:.4f}±{m['gap']['ci95']:.4f} vs null {u['gap']['mean']:.4f}±{u['gap']['ci95']:.4f}"
              f" better/worse/tied {t['method_better']}/{t['null_better']}/{t['ties']} p={t['wilcoxon_p']}"
              f" optimal {m['optimal_runs']}/{m['runs']} vs {u['optimal_runs']}/{u['runs']}"
              f" (exact uniform sampling predicts {u['optimal_runs_expected_from_exact_uniform_sampling']:.1f})"
              f" referee disagreements {m['referee_disagreements'] + u['referee_disagreements']}", flush=True)
    save("e21_qaoa_fresh_quick" if QUICK else "e21_qaoa_fresh",
         {"design": __doc__.split("Usage:")[0].strip(), "defaults": D, "sizes": SIZES, "instances": INSTANCES,
          "first_instance": FIRST_INSTANCE, "seeds": SEEDS, "instance_seed_base": INSTANCE_BASE,
          "bonferroni_threshold": BONFERRONI, "summary": summary, "rows": rows,
          "wall_s": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
