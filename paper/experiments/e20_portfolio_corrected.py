"""E20 - Portfolio selection after the corrections: the design of E19 applied to the corrected methods.

Why: E19 measured four portfolio methods as shipped and found one of them
returning its samples by the wrong rule and three not implementing the
algorithm they are named after. The four modules were then corrected (commits
a11f7d18, 71ff43ee, 8210d720, 056587f5). A correction is a claim; this
experiment tests it with the referee and the null of E19, on the instances of
E19.

What is the same as in E19: the instance generator and the 30 instances (10
each at 8, 12 and 16 assets), 5 seeds per instance and arm, the referee (exact
minimum over all 2^n selections of the QUBO the method itself builds), the
null (uniform bits through the same platform code), the outcome measures
(optimal, rank_pct, gap) and the primary comparison (gap of a method against
its null, paired by instance, Wilcoxon signed-rank on per-instance means).

What differs, and why
 - Seeds. The corrected CVaR-VQE, BF-DCQO and PCE paths take a seed, so nothing
   is substituted around them. QAOA is run under E19's seeded sampler with
   E19's seeds: its training and its final samples are then the ones E19 saw,
   and only the selection rule differs. The script checks that run by run
   against the stored E19 records.
 - Nulls. QAOA and CVaR-VQE: the ansatz is replaced by E19's uniform ansatz.
   BF-DCQO: uniform counts through its sample-injection hook. PCE: its
   expectation values are computed from uniform counts in all three bases.
 - Cardinality. E19 ran CVaR-VQE with K = n because that was what the executor
   asked for; the executor now passes the user's target. CVaR-VQE and PCE are
   run with K = max(1, n // 3), the target BF-DCQO uses, and without a target.
 - Risk aversion of CVaR-VQE. E19 used 1.0, the executor's innermost fallback.
   Jobs submitted through the optimisation endpoint did not get it: they carry
   risk_tolerance (0.5 unless set) and the executor used that number as the
   weight. The corrected executor uses 0.5, the value of QAOA and PCE, unless
   the request names a risk aversion.
 - BF-DCQO is run at the executor's shots and at 32 shots per iteration. At the
   executor's shots a cluster of 8 qubits is sampled exhaustively and the
   method cannot differ from its null; 32 shots on 256 states is the regime in
   which the circuit can show.
 - PCE now returns a selection, so it is scored like the other three.

Arms
 qaoa            IBM QAOA, lowest-cost sampled selection
 cvar_free       IBM CVaR-VQE without a target
 cvar_k          IBM CVaR-VQE, K = max(1, n // 3)
 bfdcqo          IonQ BF-DCQO on the Aer device model, executor shots, K = max(1, n // 3)
 bfdcqo_lowshot  the same with 32 shots per iteration
 pce_free        IBM PCE without a target
 pce_k           IBM PCE, K = max(1, n // 3)
each with its null. Twenty-one primary comparisons (seven arms, three sizes)
on the returned selection; p-values are reported unadjusted next to the
Bonferroni threshold 0.05 / 21.

Added after the tuning grid and before any evaluation run
The grid (see Tuning) showed that for bfdcqo, bfdcqo_lowshot and pce_k the
returned selection was the optimum in every run of the method and of its null,
in every configuration: the classical search that follows the sampling finds
the optimum from any start at these sizes. The primary comparison is kept as
specified and is expected to tie for these arms. What the quantum part
contributes is therefore measured before the classical search, in a secondary
family of 21 comparisons (threshold 0.05 / 21, reported separately):
 - PCE, both arms: gap of the decoded selection (signs of the final
   correlators, before flips and swaps), method against null. 6 comparisons.
 - BF-DCQO sampler arms: one cluster holding all n assets (qmax = 16), 32 shots
   per iteration, K = max(1, n // 3), at ansatz depth 1, 2 and 3:
     bfdcqo_sampler_d1  no evolution. With the step times of the module the
                        only step of a depth-1 circuit is empty, so every
                        iteration samples the product state its bias fields
                        prepare (uniform in the first iteration). This is the
                        control for the bias feedback, which is classical.
     bfdcqo_sampler_d2  one counterdiabatic layer (the executor's depth)
     bfdcqo_sampler_d3  two layers
   Outcome: gap of the lowest-energy raw shot over all iterations. Each depth
   against its uniform null (9 comparisons); depth 2 and depth 3 against the
   depth-1 control (6 comparisons).
The raw shots of the second and later iterations depend on the bias fields,
which are computed from shots that classical sweeps have lowered. Only the
comparison with the depth-1 control separates the circuit from that feedback.

Described, not tested
 - QAOA, CVaR-VQE: the exact final state (probability of the optimum against
   2^-n, approximation ratio) next to the state at the initial parameters.
 - BF-DCQO on single-cluster instances: the lowest-energy raw shot of each
   iteration; the exact state of the noise-free circuit at zero bias and at
   the last bias.
 - PCE: qubits used; signs the final estimate does not resolve; bits the
   classical search changed.
 - CVaR-VQE: objective evaluations, parameters, how often the selection had to
   be repaired.
 - Arms with a target, and arms that never return the empty selection: the
   standing of the returned selection among the selections the method may
   return (exactly K assets, or at least one), next to its standing among all
   2^n. With a target the mean over all 2^n is dominated by the penalty, which
   makes the gap of E19 small for any selection of the right size.
 - For every run, whether the exact reference the product now attaches to its
   result agrees with the score computed here.
 - For every arm the share of runs whose QUBO optimum is the empty selection,
   which CVaR-VQE and PCE never return.

Defaults: what a job gets in simulator mode when its request carries no method
configuration (``DEFAULTS``). The script stops if they differ from the table
the worker executors read (core/optimizers/quantum/portfolio_defaults.py).
QAOA and BF-DCQO keep the values of E19. CVaR-VQE needed new ones: E19's
evaluation budget was below the optimiser's first simplex and its final sample
was one shot. PCE keeps the executor's six iterations and six rounds.

Tuning. Exploratory, on held-out instances (generator seeds
5000 + 100 (n // 4) + i, i = 0..3; the evaluation uses 1000 + 100 (n // 4) + i,
i = 0..9), which are not used below.
 - ``--tune`` (e20_tuning.json): 36 configurations, 2 seeds per instance, 8
   runs per cell. For pce_k, bfdcqo and bfdcqo_lowshot the returned selection
   was the optimum in all runs of the method and of the null in every
   configuration, so their budgets were chosen by run time. No CVaR-VQE
   configuration separated from its null on the returned selection (method
   mean gap below the null's in 10 of 36 cells); 1024 shots per evaluation
   gave a higher approximation ratio of the trained state than 256 in all six
   depth-size cells. Eight runs per cell do not rank configurations on the
   decoded selection of PCE: this grid had it better at 256 than at 1024
   shots, and the follow-up, with ten seeds, had it the other way round.
 - ``--tune-followup`` (e20_tuning_followup.json): 10 seeds per instance, with
   decision rules written into this script before the run (FOLLOWUP_NOTE).
   PCE at the executor's budget: against 100 shots, the mean decoded gap over
   the 12 instances was lower by 0.040 (SE 0.065) at 256 shots, by 0.131 (SE
   0.037) at 1024 and by 0.111 (SE 0.059) at 4096. By the rule the default is
   1024. The returned selection was the optimum in 120 of 120 runs at every
   shot count, for the method and for the null.
   CVaR-VQE at risk aversion 0.5: against depth 1 with 1024 shots, the mean
   gap of the returned selection was lower by 0.00067 (SE 0.00028) at depth 2
   with 1024 shots and higher at 256 shots at either depth. By the rule the
   default is depth 2. In no configuration was the method's mean gap below
   its null's: method minus null was +0.0033 (SE 0.0008) at depth 1 and
   +0.0022 (SE 0.0003) at depth 2 with 256 shots, +0.0011 (SE 0.0004) at
   depth 1 and +0.0003 (SE 0.0002) at depth 2 with 1024 shots; the returned
   selection was optimal in 36, 38, 54 and 57 of 120 runs against 57 to 60 of
   120 for the null.
Both files were written before this script and the module edits it relies on
were committed. Rows of each were recomputed on the code as committed and
compared with the stored ones; the commit message of the script states how
many and the outcome.

Usage: e20_portfolio_corrected.py [--tune | --tune-followup | --quick | --figure]
  --tune           first grid on the held-out instances; writes e20_tuning.json
  --tune-followup  second grid on the same instances; writes e20_tuning_followup.json
  --quick          2 instances x 2 seeds, n = 8, on other instances (checks the harness)
  --figure         redraw the figure from the stored results
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import FIGURES, RESULTS, mean_ci, mpl, point_seed, pool_workers, save  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import e19_portfolio as e19  # noqa: E402  (instances, enumeration, scores, the seeded sampler)

TUNE = "--tune" in sys.argv
FOLLOWUP = "--tune-followup" in sys.argv
QUICK = "--quick" in sys.argv
HELD_OUT = TUNE or FOLLOWUP
SIZES = [8] if QUICK else [8, 12, 16]
INSTANCES = 2 if QUICK else (4 if HELD_OUT else 10)
SEEDS = 2 if (QUICK or TUNE) else (10 if FOLLOWUP else 5)
INSTANCE_BASE = 900 if QUICK else (5000 if HELD_OUT else 1000)

# What a job gets in simulator mode when its request carries no method configuration.
DEFAULTS = {
    "qaoa": {"risk_aversion": 0.5, "estimator_shots": 100, "sampler_shots": 200, "max_iterations": 15, "reps": 2},
    "cvar": {"alpha": 0.25, "max_cvar_iterations": None, "measurement_budget": 1024, "risk_aversion": 0.5,
             "ansatz_depth": 2, "sampling_shots": 2048},
    "bfdcqo": {"qmax": 8, "n_iterations": 3, "shots": 8192, "ansatz_depth": 2, "noise_model": "aria-1",
               "risk_aversion": 0.5, "n_candidates": 100},
    "pce": {"compression_factor": 3, "ansatz_depth": 3, "shots": 1024, "max_iterations": 6,
            "max_no_improvement_rounds": 6, "restart_threshold": 3, "optimization_timeout": 300.0, "risk_aversion": 0.5},
}
LOW_SHOTS = 32
PRIMARY_ARMS = ["qaoa", "cvar_free", "cvar_k", "bfdcqo", "bfdcqo_lowshot", "pce_free", "pce_k"]
SAMPLER_ARMS = ["bfdcqo_sampler_d1", "bfdcqo_sampler_d2", "bfdcqo_sampler_d3"]
ARMS = PRIMARY_ARMS + SAMPLER_ARMS
ARM_ID = {name: i for i, name in enumerate(ARMS)}     # the primary arms keep the ids the tuning grid ran with
ARM_SETTINGS = {
    "bfdcqo_lowshot": {"shots": LOW_SHOTS},
    "bfdcqo_sampler_d1": {"shots": LOW_SHOTS, "qmax": 16, "ansatz_depth": 1},
    "bfdcqo_sampler_d2": {"shots": LOW_SHOTS, "qmax": 16, "ansatz_depth": 2},
    "bfdcqo_sampler_d3": {"shots": LOW_SHOTS, "qmax": 16, "ansatz_depth": 3},
}
BONFERRONI = 0.05 / (len(PRIMARY_ARMS) * 3)
SECONDARY_COMPARISONS = 2 * 3 + 5 * 3     # PCE decoded selection; sampler arms against null and against the depth-1 control
BONFERRONI_SECONDARY = 0.05 / SECONDARY_COMPARISONS


def target(n: int) -> int:
    return max(1, n // 3)


def score_among(cost: np.ndarray, idx: int, n: int, cardinality=None, min_assets: int = 0) -> dict:
    """Standing among the selections a method may return: exactly ``cardinality`` assets, else at least ``min_assets``."""
    held = e19.states(n).sum(axis=1)
    mask = held == cardinality if cardinality is not None else held >= min_assets
    pool = cost[mask]
    lo, mean = float(pool.min()), float(pool.mean())
    tol = 1e-9 * max(1.0, abs(lo))
    return {"admissible": bool(mask[idx]), "optimal": bool(mask[idx] and cost[idx] <= lo + tol),
            "rank_pct": 100.0 * float((pool < cost[idx] - tol).sum()) / len(pool),
            "gap": float((cost[idx] - lo) / (mean - lo))}


def referee_agrees(reference, mine: dict) -> bool:
    """The exact reference in the product's result against the score computed here from the experiment's own enumeration."""
    return bool(reference is not None and reference["is_optimal"] == mine["optimal"]
                and abs(reference["better_selections_pct"] - mine["rank_pct"]) <= 1e-9
                and abs(reference["gap"] - mine["gap"]) <= 1e-9)


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


# --------------------------------------------------------------------------- the four methods


@contextlib.contextmanager
def e19_sampler(seed: int, null: bool):
    """E19's seeded sampler around the QAOA path, so that its samples are the ones E19 recorded."""
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa

    e19._install()
    runtime = e19._PATCH["runtime"]
    e19._RUN.update(rng=np.random.default_rng(seed), sampler_last=None, sampler_first=None, estimator_first=None,
                    aer_last=None)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    saved = runtime.SamplerV2, qaoa.QAOAAnsatz
    runtime.SamplerV2 = e19._PATCH["SeededSampler"]
    if null:
        qaoa.QAOAAnsatz = lambda cost_operator, reps=1, **_: e19.uniform_ansatz(cost_operator.num_qubits, 2 * reps)
    try:
        with quiet():
            yield
    finally:
        runtime.SamplerV2, qaoa.QAOAAnsatz = saved


def run_qaoa(prices, seed: int, null: bool, d: dict) -> dict:
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa

    n = prices.shape[1]
    linear, quadratic, _ = qaoa._build_qubo(prices, d["risk_aversion"])
    cost = e19.pair_costs(linear, quadratic, n)
    with e19_sampler(seed, null):
        result = qaoa.quantum_optimize(prices, risk_aversion=d["risk_aversion"], qc_simulator_mode=True,
                                       estimator_shots=d["estimator_shots"], sampler_shots=d["sampler_shots"],
                                       max_iterations=d["max_iterations"], reps=d["reps"])
        circuit, final_seed, shots = e19._RUN["sampler_last"]
        first = e19._RUN["sampler_first"]
    out = e19.score(cost, e19.index_of(result["solution"]))
    out["referee_agrees"] = referee_agrees(result.get("exact_reference"), out)
    out["state"] = e19.state_metrics(e19.exact_probabilities(circuit), cost)
    out["state_initial"] = e19.state_metrics(e19.exact_probabilities(first), cost)
    # the final samples again, from the same circuit and seed, to score the rule E19 measured
    sampler = e19._PATCH["base_sampler"](mode=e19._PATCH["AerSimulator"]())
    sampler.options.simulator.seed_simulator = final_seed
    counts = sampler.run([circuit], shots=shots).result()[0].data.meas.get_int_counts()
    out["most_frequent_rule"] = e19.score(cost, max(counts, key=counts.get))
    out["returned_is_lowest_cost_sample"] = bool(
        abs(cost[e19.index_of(result["solution"])] - min(cost[k] for k in counts)) <= 1e-12)
    out["reported_rule"] = result["selection"]["rule"]
    out["optimum_cardinality"] = int(e19.states(n)[int(np.argmin(cost))].sum())
    return out


def run_cvar(prices, seed: int, null: bool, cardinality, d: dict) -> dict:
    import core.optimizers.quantum.ibm.cvar_vqe_adapter as adapter_module
    from core.optimizers.quantum.ibm.cvar_vqe_optimizer import CVaRConfig

    n = prices.shape[1]
    saved = adapter_module.RealAmplitudes
    if null:
        adapter_module.RealAmplitudes = lambda num_qubits, reps=1, **_: e19.uniform_ansatz(num_qubits, num_qubits * (reps + 1))
    try:
        with quiet():
            config = CVaRConfig(simulator_mode=True, alpha=d["alpha"], max_cvar_iterations=d["max_cvar_iterations"],
                                measurement_budget=d["measurement_budget"], seed=int(seed % (2 ** 31 - 1)))
            adapter = adapter_module.IBMCVaRVQEAdapter(config, risk_aversion=d["risk_aversion"],
                                                       ansatz_depth=d["ansatz_depth"], cardinality=cardinality,
                                                       use_real_hardware=False, sampling_shots=d["sampling_shots"],
                                                       enable_classical_reweight=False)
            result = adapter.optimize(prices)
            ansatz = adapter_module.RealAmplitudes(num_qubits=n, reps=d["ansatz_depth"])
    finally:
        adapter_module.RealAmplitudes = saved
    if not result.get("success"):
        return {"failed": str(result.get("error"))[:200]}
    mu, sigma, _, _ = adapter_module._annualised_returns_and_cov(prices)
    h, J, _ = adapter_module._build_markowitz_qubo(mu, sigma, risk_aversion=d["risk_aversion"], cardinality=cardinality,
                                                   cardinality_penalty=adapter.cardinality_penalty)
    cost = e19.pair_costs(h, J, n)
    idx = e19.index_of([1 if w > 0 else 0 for w in result["portfolio_weights"].values()])
    out = e19.score(cost, idx)
    out["admissible"] = score_among(cost, idx, n, cardinality, min_assets=1)
    out["referee_agrees"] = referee_agrees(result.get("exact_reference"), out["admissible"])
    trained = np.asarray(result["metadata"]["optimal_parameters"], float)
    initial = np.random.default_rng(config.seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    out["state"] = e19.state_metrics(e19.exact_probabilities(ansatz.assign_parameters(trained)), cost)
    out["state_initial"] = e19.state_metrics(e19.exact_probabilities(ansatz.assign_parameters(initial)), cost)
    out["objective_evaluations"] = int(result["convergence_metrics"]["total_evaluations"])
    out["parameters"] = int(result["resource_usage"]["parameter_count"])
    out["repaired"] = bool(result["metadata"]["selection"]["repaired"])
    out["feasible_shot_share"] = result["metadata"]["selection"]["feasible_shots"] / result["metadata"]["selection"]["shots"]
    out["optimum_cardinality"] = int(e19.states(n)[int(np.argmin(cost))].sum())
    return out


def run_bfdcqo(prices, seed: int, null: bool, d: dict) -> dict:
    import core.optimizers.quantum.ionq.ionq_optimizer as ionq
    from core.quantum_math import portfolio_qubo, qubo_pair_to_ising

    n = prices.shape[1]
    k = target(n)
    shots = d["shots"]
    mu, sigma = e19.bfdcqo_inputs(prices)
    q, Q, _, _ = portfolio_qubo(mu, sigma, k, d["risk_aversion"])
    h, J, _ = qubo_pair_to_ising(q, Q)
    cost = e19.ising_costs(h, J, n)
    run_seed = int(seed % (2 ** 31 - 1))
    with quiet():
        if null:
            rng = np.random.default_rng(seed)

            def uniform_counts(circuit, n_shots):
                width = circuit.num_qubits
                values, counts = np.unique(rng.integers(0, 2 ** width, n_shots), return_counts=True)
                return {format(int(v), f"0{width}b"): int(c) for v, c in zip(values, counts)}

            optimizer = ionq.IonQBFDCQOOptimizer(risk_aversion=d["risk_aversion"], execute_fn=uniform_counts,
                                                 backend_name="uniform-null", seed=run_seed)
        else:
            backend, name = ionq._get_backend(None, d["noise_model"], False)
            optimizer = ionq.IonQBFDCQOOptimizer(risk_aversion=d["risk_aversion"], backend=backend, backend_name=name,
                                                 seed=run_seed)
        result = optimizer.optimize(prices, cardinality_K=k, qmax=d["qmax"], n_iterations=d["n_iterations"], shots=shots,
                                    ansatz_depth=d["ansatz_depth"], n_candidates=d["n_candidates"])
    if not result.get("success"):
        return {"failed": str(result.get("error"))[:200]}
    metrics = result["quantum_metrics"]
    if any(c["iterations_with_backend_samples"] != d["n_iterations"] for c in metrics["cluster_metrics"]):
        return {"failed": "an iteration ran on a substituted sample"}
    idx = e19.index_of([1 if w > 0 else 0 for w in result["allocation"].values()])
    out = e19.score(cost, idx)
    out["admissible"] = score_among(cost, idx, n, k)
    out["referee_agrees"] = referee_agrees(result.get("exact_reference"), out["admissible"])
    out.update(cardinality=k, clusters=int(metrics["n_clusters"]), shots=shots,
               candidates=int(metrics["candidate_pool"]["distinct"]),
               shots_over_states=[c["shots_over_states"] for c in metrics["cluster_metrics"]])
    if metrics["n_clusters"] == 1:
        cluster = metrics["cluster_metrics"][0]
        lo, mean = float(cost.min()), float(cost.mean())
        sampled = cluster["best_sampled_energy_history"]
        out["sampled_gap_by_iteration"] = [float((e - lo) / (mean - lo)) for e in sampled]
        out["best_sampled_gap"] = min(out["sampled_gap_by_iteration"])
        out["sampler_found_optimum_before_any_classical_step"] = bool(min(sampled) <= lo + 1e-9 * max(1.0, abs(lo)))
        if not null:
            biases = [np.zeros(n)] + [np.asarray(b, float) for b in cluster["bias_history"]]
            for label, bias in (("state_initial", biases[0]), ("state", biases[len(sampled) - 1])):
                circuit = ionq._build_bfdcqo_circuit(n, h, J, bias, d["ansatz_depth"], measure=False)
                out[label] = e19.state_metrics(e19.exact_probabilities(circuit), cost)
            out["rzz_gates_last_circuit"] = int(circuit.count_ops().get("rzz", 0))
            if d["ansatz_depth"] == 1 and out["rzz_gates_last_circuit"]:
                return {"failed": "the depth-1 control ran an entangling layer"}
    out["optimum_cardinality"] = int(e19.states(n)[int(np.argmin(cost))].sum())
    return out


def run_pce(prices, seed: int, null: bool, cardinality, d: dict) -> dict:
    import core.optimizers.quantum.ibm.pce_portfolio_optimizer as pce

    class UniformPCE(pce.PCEPortfolioOptimizer):
        """Expectation values of uniform bits in all three bases, through the same training and decoding."""

        def _estimate(self, params, stage):
            draws = {basis: dict(Counter(int(v) for v in self._null_rng.integers(0, 2 ** self.num_qubits, self.config.shots)))
                     for basis in pce.BASES}
            return pce.correlators(draws, self.pauli_correlations, self.num_qubits)

    n = prices.shape[1]
    config = pce.PCEConfig(compression_factor=d["compression_factor"], ansatz_depth=d["ansatz_depth"], shots=d["shots"],
                           max_iterations=d["max_iterations"], max_no_improvement_rounds=d["max_no_improvement_rounds"],
                           restart_threshold=d["restart_threshold"], optimization_timeout=d["optimization_timeout"],
                           risk_aversion=d["risk_aversion"], cardinality=cardinality, seed=int(seed % (2 ** 31 - 1)))
    with quiet():
        optimizer = (UniformPCE if null else pce.PCEPortfolioOptimizer)(config)
        optimizer._null_rng = np.random.default_rng([int(seed % (2 ** 31 - 1)), 99])
        result = optimizer.optimize(prices)
        q, Q, _, _ = optimizer.build_portfolio_qubo(prices)
    if not result.get("success"):
        return {"failed": str(result.get("error"))[:200]}
    cost = e19.pair_costs(q, Q, n)
    idx = e19.index_of(result["selection"])
    out = e19.score(cost, idx)
    out["admissible"] = score_among(cost, idx, n, cardinality, min_assets=1)
    out["referee_agrees"] = referee_agrees(result.get("exact_reference"), out["admissible"])
    out["decoded"] = e19.score(cost, e19.index_of(result["decoded_selection"]))
    out.update(qubits=int(result["resource_usage"]["qubits_used"]),
               loss_evaluations=int(result["resource_usage"]["loss_evaluations"]),
               parameters=int(result["resource_usage"]["total_parameters"]),
               bits_changed=int(result["refinement"]["bits_changed"]),
               signs_unresolved=int(result["relaxation_metrics"]["signs_within_two_standard_errors"]),
               optimum_cardinality=int(e19.states(n)[int(np.argmin(cost))].sum()))
    return out


def run_task(task):
    arm, n, instance, rep, null, overrides = task
    logging.disable(logging.CRITICAL)
    prices = e19.make_prices(n, INSTANCE_BASE + 100 * (n // 4) + instance)
    family = arm.split("_")[0]
    d = {**DEFAULTS[family], **ARM_SETTINGS.get(arm, {}), **dict(overrides)}
    # QAOA keeps E19's seeds so that its samples are E19's.
    seed = point_seed(19, e19.ARM_ID["qaoa"], n, instance, rep, int(null)) if arm == "qaoa" \
        else point_seed(20, ARM_ID[arm], n, instance, rep, int(null))
    started = time.time()
    try:
        if arm == "qaoa":
            out = run_qaoa(prices, seed, null, d)
        elif family == "cvar":
            out = run_cvar(prices, seed, null, target(n) if arm == "cvar_k" else None, d)
        elif family == "bfdcqo":
            out = run_bfdcqo(prices, seed, null, d)
        else:
            out = run_pce(prices, seed, null, target(n) if arm == "pce_k" else None, d)
    except Exception as exc:  # recorded, never dropped
        out = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
    out.update(arm=arm, n=n, instance=instance, rep=rep, null=null, seed=seed, overrides=dict(overrides),
               wall_s=round(time.time() - started, 2))
    return out


# --------------------------------------------------------------------------- summary


def summarise(rows, arms=ARMS):
    out = {}
    for arm in arms:
        for n in SIZES:
            cell = {}
            for null in (False, True):
                sel = [r for r in rows if r["arm"] == arm and r["n"] == n and r["null"] == null]
                ok = [r for r in sel if "failed" not in r]
                entry = {"runs": len(sel), "failed": len(sel) - len(ok),
                         "mean_wall_s": float(np.mean([r["wall_s"] for r in sel])) if sel else None,
                         "optimal_runs": int(sum(r["optimal"] for r in ok)),
                         "referee_disagreements": int(sum(not r["referee_agrees"] for r in ok)),
                         "optimum_is_empty_selection_runs": int(sum(r["optimum_cardinality"] == 0 for r in ok))}
                among = [r for r in ok if "admissible" in r]
                if among:
                    entry["admissible_runs"] = int(sum(r["admissible"]["admissible"] for r in among))
                    entry["admissible_optimal_runs"] = int(sum(r["admissible"]["optimal"] for r in among))
                    for field in ("gap", "rank_pct"):
                        entry[f"admissible_{field}"] = mean_ci(list(e19.per_instance(among, field, "admissible").values()))
                if arm.endswith("_k") or arm.startswith("bfdcqo"):
                    # Above zero: the penalty in the QUBO is too weak to make a selection of the target size its minimum.
                    entry["qubo_minimum_misses_target_runs"] = int(sum(r["optimum_cardinality"] != target(n) for r in ok))
                for field in ("gap", "rank_pct", "optimal", "selected"):
                    entry[field] = mean_ci(list(e19.per_instance(ok, field).values()))
                    entry[f"{field}_by_instance"] = e19.per_instance(ok, field)
                for sub in ("state", "state_initial"):
                    with_state = [r for r in ok if sub in r]
                    if with_state:
                        for field in ("p_optimum_over_uniform", "approx_ratio"):
                            entry[f"{sub}_{field}"] = mean_ci(list(e19.per_instance(with_state, field, sub).values()))
                        entry[f"{sub}_p_optimum_over_uniform_median"] = float(
                            np.median([r[sub]["p_optimum_over_uniform"] for r in with_state]))
                        entry[f"{sub}_runs"] = len(with_state)
                if arm == "qaoa":
                    for field in ("gap", "optimal", "rank_pct"):
                        entry[f"most_frequent_rule_{field}"] = mean_ci(
                            list(e19.per_instance(ok, field, "most_frequent_rule").values()))
                    entry["most_frequent_rule_optimal_runs"] = int(sum(r["most_frequent_rule"]["optimal"] for r in ok))
                    entry["returned_is_lowest_cost_sample_runs"] = int(sum(r["returned_is_lowest_cost_sample"] for r in ok))
                if arm.startswith("cvar"):
                    entry["objective_evaluations"] = sorted({r["objective_evaluations"] for r in ok})
                    entry["parameters"] = sorted({r["parameters"] for r in ok})
                    entry["repaired_runs"] = int(sum(r["repaired"] for r in ok))
                    entry["feasible_shot_share"] = mean_ci([r["feasible_shot_share"] for r in ok])
                if arm.startswith("bfdcqo"):
                    entry["clusters"] = sorted({r["clusters"] for r in ok})
                    entry["cardinality"] = sorted({r["cardinality"] for r in ok})
                    entry["candidates"] = mean_ci([r["candidates"] for r in ok])
                    single = [r for r in ok if "sampler_found_optimum_before_any_classical_step" in r]
                    entry["sampler_found_optimum_before_any_classical_step"] = {
                        "runs": len(single),
                        "optimal": int(sum(r["sampler_found_optimum_before_any_classical_step"] for r in single))}
                    if single:
                        length = min(len(r["sampled_gap_by_iteration"]) for r in single)
                        entry["sampled_gap_by_iteration"] = [
                            mean_ci([r["sampled_gap_by_iteration"][i] for r in single]) for i in range(length)]
                        entry["best_sampled_gap"] = mean_ci(list(e19.per_instance(single, "best_sampled_gap").values()))
                        entry["best_sampled_gap_by_instance"] = e19.per_instance(single, "best_sampled_gap")
                        entry["rzz_gates_last_circuit"] = sorted({r["rzz_gates_last_circuit"] for r in single
                                                                  if "rzz_gates_last_circuit" in r})
                if arm.startswith("pce"):
                    for field in ("gap", "optimal", "rank_pct"):
                        entry[f"decoded_{field}"] = mean_ci(list(e19.per_instance(ok, field, "decoded").values()))
                    entry["decoded_optimal_runs"] = int(sum(r["decoded"]["optimal"] for r in ok))
                    entry["decoded_gap_by_instance"] = e19.per_instance(ok, "gap", "decoded")
                    entry["qubits"] = sorted({r["qubits"] for r in ok})
                    entry["parameters"] = sorted({r["parameters"] for r in ok})
                    entry["loss_evaluations"] = mean_ci([r["loss_evaluations"] for r in ok])
                    entry["bits_changed"] = mean_ci([r["bits_changed"] for r in ok])
                    entry["signs_unresolved"] = mean_ci([r["signs_unresolved"] for r in ok])
                cell["null" if null else "method"] = entry
            cell["method_vs_null"] = paired_test(cell["method"]["gap_by_instance"], cell["null"]["gap_by_instance"])
            if arm.startswith("pce"):
                cell["decoded_method_vs_null"] = paired_test(cell["method"]["decoded_gap_by_instance"],
                                                             cell["null"]["decoded_gap_by_instance"])
            if "best_sampled_gap_by_instance" in cell["method"] and "best_sampled_gap_by_instance" in cell["null"]:
                cell["sampler_vs_null"] = paired_test(cell["method"]["best_sampled_gap_by_instance"],
                                                      cell["null"]["best_sampled_gap_by_instance"])
            out[f"{arm}|{n}"] = cell
    return out


def sampler_comparisons(summary) -> dict:
    """Lowest-energy raw shot: the circuits with evolution against the depth-1 control, paired by instance."""
    out = {}
    for n in SIZES:
        control = summary.get(f"bfdcqo_sampler_d1|{n}", {}).get("method", {}).get("best_sampled_gap_by_instance")
        for depth in (2, 3):
            mine = summary.get(f"bfdcqo_sampler_d{depth}|{n}", {}).get("method", {}).get("best_sampled_gap_by_instance")
            if control and mine:
                out[f"d{depth}_vs_d1|{n}"] = paired_test(mine, control)
    return out


def paired_test(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    diff = [a[i] - b[i] for i in common]
    return {"instances": len(common), "method_minus_null": mean_ci(diff),
            "method_better": int(sum(d < -1e-12 for d in diff)), "null_better": int(sum(d > 1e-12 for d in diff)),
            "ties": int(sum(abs(d) <= 1e-12 for d in diff)),
            "wilcoxon_p": float(wilcoxon(diff).pvalue) if any(abs(d) > 1e-12 for d in diff) else None}


def compare_with_e19(rows) -> dict:
    """QAOA ran on E19's samples: its returned score must be E19's lowest-cost rule, its old rule E19's result."""
    stored = json.loads((RESULTS / "e19_portfolio.json").read_text())["rows"]
    before = {(r["n"], r["instance"], r["rep"], r["null"]): r for r in stored if r["arm"] == "qaoa" and "failed" not in r}
    out = {"runs": 0, "same_seed": 0, "returned_equals_e19_lowest_cost_rule": 0, "most_frequent_equals_e19_returned": 0,
           "same_final_state": 0}
    for r in rows:
        if r["arm"] != "qaoa" or "failed" in r:
            continue
        old = before.get((r["n"], r["instance"], r["rep"], r["null"]))
        if old is None:
            continue
        out["runs"] += 1
        out["same_seed"] += int(old["seed"] == r["seed"])
        same = lambda x, y: x["optimal"] == y["optimal"] and abs(x["gap"] - y["gap"]) <= 1e-9  # noqa: E731
        out["returned_equals_e19_lowest_cost_rule"] += int(same(r, old["lowest_cost_rule"]))
        out["most_frequent_equals_e19_returned"] += int(same(r["most_frequent_rule"], old))
        out["same_final_state"] += int(abs(r["state"]["approx_ratio"] - old["state"]["approx_ratio"]) <= 1e-9)
    return out


def figure(summary) -> None:
    """Rank of the returned selection, method against null, as in E19's figure."""
    plt = mpl()
    labels = ["QAOA", "CVaR-VQE\nfree", "CVaR-VQE\nK", "BF-DCQO", "BF-DCQO\n32 shots", "PCE\nfree", "PCE\nK"]
    fig, axes = plt.subplots(1, len(SIZES), figsize=(6.8, 2.5), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], SIZES):
        for offset, key, colour, marker, name in ((-0.13, "method", "C0", "o", "method"),
                                                  (0.13, "null", "0.45", "x", "uniform null")):
            cells = [summary[f"{a}|{n}"][key]["rank_pct"] for a in PRIMARY_ARMS]
            ax.errorbar(np.arange(len(PRIMARY_ARMS)) + offset, [c["mean"] for c in cells], yerr=[c["ci95"] for c in cells],
                        fmt=marker, color=colour, ms=3.5, lw=0.8, capsize=1.5, label=name)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(-0.15, 100)
        ax.set_yticks([0, 1, 10, 50, 100])
        ax.set_yticklabels(["0", "1", "10", "50", "100"])
        ax.set_xticks(np.arange(len(PRIMARY_ARMS)))
        ax.set_xticklabels(labels, fontsize=4.5)
        ax.set_title(f"{n} assets", fontsize=7)
    axes[0][0].set_ylabel("selections that are better (%)\n(0 = exact optimum)")
    axes[0][-1].legend(fontsize=5.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "portfolio_corrected.pdf")


# --------------------------------------------------------------------------- tuning on held-out instances

# The first grid ran when the CVaR-VQE default risk aversion was 1.0; it is pinned so that the grid stays repeatable.
TUNING_GRID = {
    "cvar_k": [(("risk_aversion", 1.0), ("ansatz_depth", depth), ("measurement_budget", shots), ("max_cvar_iterations", budget))
               for depth in (1, 2) for shots in (256, 1024) for budget in (None, 300, 1000)],
    "pce_k": [(("max_iterations", iterations), ("max_no_improvement_rounds", rounds), ("shots", shots),
               ("compression_factor", compression))
              for iterations, rounds in ((6, 6), (50, 6), (100, 12)) for shots in (256, 1024) for compression in (3, 8)],
    "bfdcqo_lowshot": [(("ansatz_depth", depth), ("n_iterations", iterations))
                       for depth in (2, 3) for iterations in (3, 10)],
    "bfdcqo": [(("ansatz_depth", depth), ("n_iterations", iterations), ("shots", shots))
               for depth in (2, 3) for iterations in (3, 10) for shots in (512, 8192)],
}
TUNING_NOTE = "exploratory grid on held-out instances; used only to choose defaults"

# Two questions the first grid left open, with ten seeds per instance.
FOLLOWUP_GRID = {
    "pce_k": [(("max_iterations", 6), ("max_no_improvement_rounds", 6), ("shots", shots)) for shots in (100, 256, 1024, 4096)],
    "cvar_k": [(("risk_aversion", 0.5), ("ansatz_depth", depth), ("measurement_budget", shots))
               for depth in (1, 2) for shots in (256, 1024)],
}
FOLLOWUP_NOTE = (
    "exploratory, held-out instances, ten seeds. Decision rules, written before the run. "
    "PCE shots (the first grid had the decoded selection better at 256 than at 1024 shots, on 8 runs per cell): "
    "the executor's 100 shots are kept unless another value has a lower mean decoded gap over the 12 instances "
    "(4 per size) by more than two standard errors of the paired per-instance difference. "
    "CVaR-VQE at risk aversion 0.5: depth 1 with 1024 shots is kept unless another configuration has a lower mean "
    "gap of the returned selection over the 12 instances by more than two standard errors of the paired difference."
)


def tune(grid=TUNING_GRID, name: str = "e20_tuning", note: str = TUNING_NOTE) -> None:
    started = time.time()
    tasks = [(arm, n, instance, rep, null, overrides) for arm, arm_grid in grid.items() for overrides in arm_grid
             for n in SIZES for instance in range(INSTANCES) for rep in range(SEEDS) for null in (False, True)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=2))
    table = []
    for arm, arm_grid in grid.items():
        for overrides in arm_grid:
            for n in SIZES:
                def cell(null):
                    sel = [r for r in rows if r["arm"] == arm and r["n"] == n and r["null"] == null
                           and r["overrides"] == dict(overrides) and "failed" not in r]
                    entry = {"runs": len(sel), "gap": float(np.mean([r["gap"] for r in sel])) if sel else None,
                             "optimal": int(sum(r["optimal"] for r in sel)),
                             "gap_by_instance": e19.per_instance(sel, "gap"),
                             "wall_s": float(np.mean([r["wall_s"] for r in sel])) if sel else None}
                    if sel and "decoded" in sel[0]:
                        entry["decoded_gap"] = float(np.mean([r["decoded"]["gap"] for r in sel]))
                        entry["decoded_optimal"] = int(sum(r["decoded"]["optimal"] for r in sel))
                        entry["decoded_gap_by_instance"] = e19.per_instance(sel, "gap", "decoded")
                    if sel and "state" in sel[0]:
                        entry["approx_ratio"] = float(np.mean([r["state"]["approx_ratio"] for r in sel if "state" in r]))
                        entry["p_opt_over_uniform_median"] = float(
                            np.median([r["state"]["p_optimum_over_uniform"] for r in sel if "state" in r]))
                    single = [r for r in sel if "sampler_found_optimum_before_any_classical_step" in r]
                    if single:
                        entry["sampler_optimal"] = int(sum(r["sampler_found_optimum_before_any_classical_step"] for r in single))
                        entry["sampled_gap_last"] = float(np.mean([r["sampled_gap_by_iteration"][-1] for r in single]))
                        entry["sampled_gap_first"] = float(np.mean([r["sampled_gap_by_iteration"][0] for r in single]))
                    return entry
                table.append({"arm": arm, "overrides": dict(overrides), "n": n, "method": cell(False), "null": cell(True)})
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} runs, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed[:8]:
        print("  failed:", r["arm"], r["n"], r["overrides"], r["failed"], flush=True)
    for entry in table:
        m, u = entry["method"], entry["null"]
        extras = " ".join(f"{k}={m[k]:.3f}/{u.get(k, float('nan')):.3f}" if isinstance(m[k], float) else f"{k}={m[k]}/{u.get(k)}"
                          for k in m if k not in ("runs", "gap", "optimal", "wall_s") and not k.endswith("_by_instance"))
        print(f"{entry['arm']:15s} n={entry['n']:2d} {str(entry['overrides']):95s} gap {m['gap']:.3f}/{u['gap']:.3f} "
              f"opt {m['optimal']}/{u['optimal']} of {m['runs']} t={m['wall_s']:.1f}s {extras}", flush=True)
    save(name, {"note": note,
                "instance_seed_base": INSTANCE_BASE, "sizes": SIZES, "instances": INSTANCES, "seeds": SEEDS,
                "defaults_at_the_time": DEFAULTS, "grid": {k: [dict(o) for o in v] for k, v in grid.items()},
                "table": table, "rows": rows, "wall_s": round(time.time() - started, 1)})


def check_defaults() -> None:
    """The values evaluated here are the ones the worker executors read from the product's table."""
    from core.optimizers.quantum.portfolio_defaults import SIMULATOR_DEFAULTS

    for mine, theirs in (("qaoa", "qaoa"), ("cvar", "cvar_vqe"), ("bfdcqo", "bfdcqo"), ("pce", "pce")):
        if DEFAULTS[mine] != SIMULATOR_DEFAULTS[theirs]:
            sys.exit(f"DEFAULTS[{mine!r}] differs from the product table: {DEFAULTS[mine]} != {SIMULATOR_DEFAULTS[theirs]}")


def require_committed_code() -> None:
    """The evaluation measures committed code: stop if this script or the code it loads is modified or untracked.

    The two directories are those of the loaded modules: the solver package and this directory. In the
    platform repository they are backend/src/main/python/core and reports/paper/quantumfpo-arxiv/experiments,
    the paths this function named when the stored results were produced; in the public reproduction package
    they are src/core and paper/experiments. Outside a git checkout (an archive download) the state of the
    code cannot be checked: the run goes on, says so, and its result records the commit as unknown.
    """
    import core

    here = os.path.dirname(os.path.abspath(__file__))
    watched = [os.path.dirname(os.path.abspath(core.__file__)), here]
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, cwd=here,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("not a git checkout: the state of the code cannot be checked; the result records the commit as unknown",
              flush=True)
        return
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *watched],
                                     text=True, cwd=root)
    changed = [line for line in status.splitlines() if line[3:].strip('"').endswith(".py")]
    if changed:
        sys.exit("not run: uncommitted code\n" + "\n".join(changed))


def main() -> None:
    if "--figure" in sys.argv:
        stored = json.loads((RESULTS / "e20_portfolio_corrected.json").read_text())
        SIZES[:] = stored["sizes"]
        figure(stored["summary"])
        return
    if TUNE:
        tune()
        return
    if FOLLOWUP:
        tune(FOLLOWUP_GRID, "e20_tuning_followup", FOLLOWUP_NOTE)
        return
    check_defaults()
    if not QUICK:
        require_committed_code()
    started = time.time()
    tasks = [(arm, n, instance, rep, null, ()) for arm in ARMS for n in SIZES for instance in range(INSTANCES)
             for rep in range(SEEDS) for null in (False, True)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=2))
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} runs, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed[:10]:
        print("  failed:", r["arm"], r["n"], r["instance"], r["rep"], r["null"], r["failed"], flush=True)
    summary = summarise(rows)
    for key, cell in summary.items():
        m, u, t = cell["method"], cell["null"], cell["method_vs_null"]
        print(f"{key:22s} gap {m['gap']['mean']:.4f}±{m['gap']['ci95']:.4f} vs null {u['gap']['mean']:.4f}±{u['gap']['ci95']:.4f}"
              f" p={t['wilcoxon_p']} optimal {m['optimal_runs']}/{m['runs']} vs {u['optimal_runs']}/{u['runs']}"
              f" referee disagreements {m['referee_disagreements'] + u['referee_disagreements']}", flush=True)
        for label in ("decoded_method_vs_null", "sampler_vs_null"):
            if label in cell:
                s = cell[label]
                print(f"    {label}: method - null {s['method_minus_null']['mean']:+.4f}±{s['method_minus_null']['ci95']:.4f}"
                      f" better/worse/tied {s['method_better']}/{s['null_better']}/{s['ties']} p={s['wilcoxon_p']}", flush=True)
    against_control = sampler_comparisons(summary)
    for key, s in against_control.items():
        print(f"sampler {key}: depth - control {s['method_minus_null']['mean']:+.4f}±{s['method_minus_null']['ci95']:.4f}"
              f" better/worse/tied {s['method_better']}/{s['null_better']}/{s['ties']} p={s['wilcoxon_p']}", flush=True)
    against_e19 = None if QUICK else compare_with_e19(rows)
    print("QAOA against the E19 records:", against_e19, flush=True)
    save("e20_portfolio_corrected_quick" if QUICK else "e20_portfolio_corrected",
         {"design": __doc__.split("Usage:")[0].strip(), "defaults": DEFAULTS, "low_shots": LOW_SHOTS, "sizes": SIZES,
          "primary_arms": PRIMARY_ARMS, "sampler_arms": SAMPLER_ARMS, "arm_settings": ARM_SETTINGS,
          "instances": INSTANCES, "seeds": SEEDS, "instance_seed_base": INSTANCE_BASE, "bonferroni_threshold": BONFERRONI,
          "bonferroni_threshold_secondary": BONFERRONI_SECONDARY, "sampler_arms_against_depth_1_control": against_control,
          "qaoa_against_e19_records": against_e19, "summary": summary, "rows": rows,
          "wall_s": round(time.time() - started, 1)})
    if not QUICK:
        figure(summary)


if __name__ == "__main__":
    main()
