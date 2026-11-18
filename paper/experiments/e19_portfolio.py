"""E19 - Portfolio selection: four gate-model methods against the exact optimum of their own objective
and against a uniform null that runs through the same platform code.

Why: the earlier experiments covered five workloads and left out the one the
platform was built around. Its classical baseline (continuous maximum-Sharpe
weights) solves a different problem from every QUBO the quantum methods
encode, so no portfolio run had ever been compared with an exact answer. At
8-16 assets the exact answer is an enumeration of 2^n selections.

History, stated because it bears on how the design was chosen. An exploratory
probe on three 8-asset instances (generator seeds 1-3, not used below) ran the
four methods once and found that BF-DCQO returned the first K assets on all
three. The cause was the classical refinement step (a private, wrong
QUBO-to-Ising conversion); it was fixed in commit d044fc74 and the module as
it shipped is kept byte-identical under ``legacy/``. The design below was
written after that probe and before any instance below was generated. It is
therefore not blind to the refinement defect; it is blind to everything else
it measures.

Design
------
Instances: seeded synthetic daily prices from a factor model (one market
factor, three sector factors, idiosyncratic noise; 504 trading days), n = 8,
12 and 16 assets, 10 instances per size, 5 seeds per instance and arm. No
market data, so the package is self-contained.

Each method runs through the platform's own optimiser entry point with the
defaults its worker executor passes in simulator mode (``DEFAULTS`` below,
with the executor lines they come from). Only two things are substituted, and
only around the platform code, never inside it:
 - the Aer primitives get a seed (the shipped paths expose none), and the last
   circuit they ran is kept so that its output distribution can be computed
   exactly;
 - in the null arms the ansatz is replaced by a Hadamard layer followed by
   parameterised Z rotations (same parameter count). Its measured distribution
   is uniform for every parameter value, so objective evaluation, optimiser,
   sampling, selection rule, repair and refinement all run unchanged on
   uniform bits. BF-DCQO has a sample-injection hook (``execute_fn``) and PCE a
   decode entry point (``_extract_portfolio_solution``); their nulls feed
   uniform counts through those.

Arms
 qaoa             IBM QAOA (ibm/qaoa_circuit.quantum_optimize)
 cvar_all         IBM CVaR-VQE with cardinality = n: what the executor asks
                  for after it has pre-selected max_assets columns
 cvar_free        IBM CVaR-VQE without cardinality (no max_assets)
 bfdcqo_shipped   IonQ BF-DCQO on the Aer device model, refinement as shipped
                  from 2026-05-08 to 2026-11-17
 bfdcqo_fixed     the same with the corrected refinement
 pce              IBM PCE
each with its null (suffix ``_null``).

Referee: for the three binary methods, the exact minimum over all 2^n
selections of the QUBO that method itself builds (each method has its own).
For PCE, whose output is a weight vector, the exact minimum of its own cost
over the simplex (a convex quadratic programme).

Outcome measures, fixed before the runs
 optimal      the returned selection is an exact optimum of the method's QUBO
 rank_pct     share of all 2^n selections that are strictly better (0 = best)
 gap          (cost - min) / (mean over all selections - min): 0 at the
              optimum, 1 for the average selection
 PCE          cost excess over the simplex optimum, next to that of equal
              weights
 primary comparison: gap of a method against its null, paired by instance
 (Wilcoxon signed-rank on per-instance means). Unit of analysis: the instance.
 Described, not tested: for QAOA and CVaR-VQE the exact final state
 (probability of the optimum against 2^-n; approximation ratio) next to the
 exact state at the initial parameters, which shows what training changed;
 for QAOA the lowest-cost rule applied to the same final samples; for
 BF-DCQO whether the answer changes when the columns are reordered; the
 time the exact enumeration takes.
 Eighteen primary comparisons (six arms, three sizes); p-values unadjusted.

Deterministic checks of what the shipped code computes are recorded under
``audit``. The paper's statements about the behaviour of the shipped code
refer to these checks or to the run records; the objective formulas it quotes
are those of the platform builders this script calls.

Usage: e19_portfolio.py [--quick | --figure]
  --quick    2 instances x 2 seeds, n = 8, on other instances (checks the harness)
  --figure   redraw the figure from the stored results
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import FIGURES, mean_ci, mpl, point_seed, pool_workers, save  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

QUICK = "--quick" in sys.argv
SIZES = [8] if QUICK else [8, 12, 16]
INSTANCES = 2 if QUICK else 10
SEEDS = 2 if QUICK else 5
INSTANCE_BASE = 900 if QUICK else 1000  # the probe used 1-3
DAYS = 504

# Simulator-mode defaults of the worker executors (api/executors.py at commit a9c51482).
DEFAULTS = {
    "qaoa": {"risk_aversion": 0.5, "estimator_shots": 100, "sampler_shots": 200, "max_iterations": 15, "reps": 2,
             "source": "execute_quantum_qaoa_optimization: estimatorShots 100, samplerShots 200, 15 iterations, reps 2; "
                       "risk_aversion is not passed (function default 0.5)"},
    "cvar": {"alpha": 0.25, "adaptive_alpha": True, "max_cvar_iterations": 20, "measurement_budget": 100,
             "noise_mitigation": True, "risk_aversion": 1.0, "ansatz_depth": 1, "sampling_shots": 1,
             "source": "execute_quantum_cvar_vqe_optimization: default_iterations_cvar 20, default_shots_cvar 100, "
                       "ansatz depth 1, risk_aversion 1.0, _effective_pce_shots 1 in simulator mode, "
                       "cardinality = max_assets or None"},
    "bfdcqo": {"qmax": 8, "n_iterations": 3, "shots": 8192, "ansatz_depth": 2, "noise_model": "aria-1",
               "risk_aversion": 0.5, "cardinality": "max(1, n // 3)",
               "source": "execute_ionq_bfdcqo_optimization: qmax 8, 3 iterations, 8192 shots, depth 2, K = n // 3"},
    "pce": {"compression_factor": 3, "ansatz_depth": 3, "shots": 100, "max_iterations": 6,
            "max_no_improvement_rounds": 6, "restart_threshold": 3, "optimization_timeout": 300.0, "risk_aversion": 0.5,
            "source": "execute_quantum_pce_optimization: 100 shots, 6 iterations and 6 outer rounds in simulator mode"},
}
ARMS = ["qaoa", "cvar_all", "cvar_free", "bfdcqo_shipped", "bfdcqo_fixed", "pce"]
ARM_ID = {name: i for i, name in enumerate(ARMS)}
LEGACY_FILE = Path(__file__).resolve().parent / "legacy" / "refinement_engine_shipped_2026_05_08_to_2026_11_17.py"


# --------------------------------------------------------------------------- instances and exact enumeration


def make_prices(n: int, seed: int) -> pd.DataFrame:
    """Factor-model daily prices: market + three sectors + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n)
    sector = rng.integers(0, 3, n)
    drift = rng.uniform(0.02, 0.18, n) / 252
    idio = rng.uniform(0.008, 0.02, n)
    market = rng.normal(0, 0.01, DAYS)
    sectors = rng.normal(0, 0.006, (DAYS, 3))
    returns = drift + beta * market[:, None] + sectors[:, sector] + rng.normal(0, 1, (DAYS, n)) * idio
    index = pd.bdate_range("2024-01-01", periods=DAYS)
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=[f"A{i:02d}" for i in range(n)])


_STATES = {}


def states(n: int) -> np.ndarray:
    """All 2^n selections; row index has asset i as bit i."""
    if n not in _STATES:
        _STATES[n] = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return _STATES[n]


def pair_costs(q, Q_upper, n: int) -> np.ndarray:
    s = states(n)
    return s @ np.asarray(q, float) + np.einsum("bi,ij,bj->b", s, np.triu(np.asarray(Q_upper, float), 1), s, optimize=True)


def ising_costs(h, J, n: int) -> np.ndarray:
    z = 1.0 - 2.0 * states(n)
    return z @ np.asarray(h, float) + np.einsum("bi,ij,bj->b", z, np.triu(np.asarray(J, float), 1), z, optimize=True)


def index_of(bits) -> int:
    return int(sum(int(round(float(b))) << i for i, b in enumerate(bits)))


def score(cost: np.ndarray, idx: int) -> dict:
    lo, mean = float(cost.min()), float(cost.mean())
    tol = 1e-9 * max(1.0, abs(lo))
    return {"optimal": bool(cost[idx] <= lo + tol), "rank_pct": 100.0 * float((cost < cost[idx] - tol).sum()) / len(cost),
            "gap": float((cost[idx] - lo) / (mean - lo)), "selected": int(bin(idx).count("1"))}


# --------------------------------------------------------------------------- seeded primitives and the uniform ansatz

_RUN = {"rng": None, "sampler_last": None, "sampler_first": None, "estimator_first": None, "aer_last": None}
_PATCH = {}


def _install() -> None:
    """Seeded subclasses of the Aer primitives; they also remember the last circuit they ran."""
    if _PATCH:
        return
    import qiskit_ibm_runtime as runtime
    from qiskit_aer import AerSimulator

    base_sampler, base_estimator = runtime.SamplerV2, runtime.EstimatorV2

    def next_seed() -> int:
        return int(_RUN["rng"].integers(1, 2 ** 31 - 1))

    class SeededSampler(base_sampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.options.simulator.seed_simulator = next_seed()

        def run(self, pubs, **kwargs):
            _RUN["sampler_last"] = (pubs[0], self.options.simulator.seed_simulator, kwargs.get("shots"))
            if _RUN["sampler_first"] is None:
                _RUN["sampler_first"] = pubs[0]
            return super().run(pubs, **kwargs)

    class SeededEstimator(base_estimator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.options.simulator.seed_simulator = next_seed()

        def run(self, pubs, **kwargs):
            if _RUN["estimator_first"] is None:
                _RUN["estimator_first"] = pubs[0][0]
            return super().run(pubs, **kwargs)

    class SeededAer(AerSimulator):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("seed_simulator", next_seed())
            super().__init__(*args, **kwargs)

        def run(self, circuits, **kwargs):
            _RUN["aer_last"] = circuits
            return super().run(circuits, **kwargs)

    _PATCH.update(runtime=runtime, base_sampler=base_sampler, SeededSampler=SeededSampler,
                  SeededEstimator=SeededEstimator, SeededAer=SeededAer, AerSimulator=AerSimulator)


def uniform_ansatz(num_qubits: int, num_parameters: int):
    """Hadamards, then one Z rotation per parameter: measured distribution uniform for every parameter value."""
    from qiskit.circuit import Parameter, QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for k in range(num_parameters):
        qc.rz(Parameter(f"null_{k}"), k % num_qubits)
    return qc


@contextlib.contextmanager
def patched(seed: int, null: bool):
    """Seed every Aer primitive the portfolio paths create; in null arms swap the ansatz factories."""
    _install()
    import core.optimizers.quantum.ibm.cvar_vqe_adapter as cvar_adapter
    import core.optimizers.quantum.ibm.cvar_vqe_optimizer as cvar_optimizer
    import core.optimizers.quantum.ibm.pce_portfolio_optimizer as pce
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa

    runtime = _PATCH["runtime"]
    _RUN.update(rng=np.random.default_rng(seed), sampler_last=None, sampler_first=None, estimator_first=None, aer_last=None)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    saved = [(runtime, "SamplerV2", runtime.SamplerV2), (pce, "Sampler", pce.Sampler),
             (cvar_optimizer, "Estimator", cvar_optimizer.Estimator), (cvar_adapter, "AerSimulator", cvar_adapter.AerSimulator),
             (qaoa, "QAOAAnsatz", qaoa.QAOAAnsatz), (cvar_adapter, "RealAmplitudes", cvar_adapter.RealAmplitudes)]
    runtime.SamplerV2 = _PATCH["SeededSampler"]
    pce.Sampler = _PATCH["SeededSampler"]
    cvar_optimizer.Estimator = _PATCH["SeededEstimator"]
    cvar_adapter.AerSimulator = _PATCH["SeededAer"]
    if null:
        qaoa.QAOAAnsatz = lambda cost_operator, reps=1, **_: uniform_ansatz(cost_operator.num_qubits, 2 * reps)
        cvar_adapter.RealAmplitudes = lambda num_qubits, reps=1, **_: uniform_ansatz(num_qubits, num_qubits * (reps + 1))
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


def exact_probabilities(circuit) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()


def state_metrics(prob: np.ndarray, cost: np.ndarray) -> dict:
    lo, mean = float(cost.min()), float(cost.mean())
    best = np.abs(cost - lo) <= 1e-9 * max(1.0, abs(lo))
    p_opt = float(prob[best].sum())
    return {"p_optimum": p_opt, "p_optimum_uniform": float(best.sum()) / len(cost),
            "p_optimum_over_uniform": p_opt * len(cost) / float(best.sum()),
            "approx_ratio": float((mean - prob @ cost) / (mean - lo))}


# --------------------------------------------------------------------------- the four methods


def run_qaoa(prices, seed: int, null: bool) -> dict:
    import core.optimizers.quantum.ibm.qaoa_circuit as qaoa

    d = DEFAULTS["qaoa"]
    n = prices.shape[1]
    linear, quadratic, _ = qaoa._build_qubo(prices, d["risk_aversion"])
    cost = pair_costs(linear, quadratic, n)
    with patched(seed, null):
        result = qaoa.quantum_optimize(prices, risk_aversion=d["risk_aversion"], qc_simulator_mode=True,
                                       estimator_shots=d["estimator_shots"], sampler_shots=d["sampler_shots"],
                                       max_iterations=d["max_iterations"], reps=d["reps"])
        circuit, final_seed, shots = _RUN["sampler_last"]
        first = _RUN["sampler_first"]
    out = score(cost, index_of(result["solution"]))
    out["state"] = state_metrics(exact_probabilities(circuit), cost)
    out["state_initial"] = state_metrics(exact_probabilities(first), cost)
    # the final samples again, from the same circuit and seed, to read both selection rules off them
    sampler = _PATCH["base_sampler"](mode=_PATCH["AerSimulator"]())
    sampler.options.simulator.seed_simulator = final_seed
    counts = sampler.run([circuit], shots=shots).result()[0].data.meas.get_int_counts()
    out["returned_is_most_frequent_sample"] = bool(max(counts, key=counts.get) == index_of(result["solution"]))
    out["lowest_cost_rule"] = score(cost, min(counts, key=lambda k: cost[k]))
    out["distinct_final_samples"] = len(counts)
    out["largest_sample_count"] = int(max(counts.values()))
    out["init_scheme"] = result["warm_start"]["init_scheme"]
    out["optimum_cardinality"] = int(states(n)[int(np.argmin(cost))].sum())
    return out


def run_cvar(prices, seed: int, null: bool, select_all: bool) -> dict:
    import core.optimizers.quantum.ibm.cvar_vqe_adapter as adapter_module
    from core.optimizers.quantum.ibm.cvar_vqe_optimizer import CVaRConfig

    d = DEFAULTS["cvar"]
    n = prices.shape[1]
    cardinality = n if select_all else None
    with patched(seed, null):
        config = CVaRConfig(simulator_mode=True, alpha=d["alpha"], adaptive_alpha=d["adaptive_alpha"],
                            max_cvar_iterations=d["max_cvar_iterations"], measurement_budget=d["measurement_budget"],
                            noise_mitigation=d["noise_mitigation"])
        # reweighting comes after the selection and does not change it; switched off to read the selection
        adapter = adapter_module.IBMCVaRVQEAdapter(config, risk_aversion=d["risk_aversion"], ansatz_depth=d["ansatz_depth"],
                                                   cardinality=cardinality, use_real_hardware=False,
                                                   sampling_shots=d["sampling_shots"], enable_classical_reweight=False)
        result = adapter.optimize(prices)
        circuit = _RUN["aer_last"]
        first = _RUN["estimator_first"]
        mu, sigma, _, _ = adapter_module._annualised_returns_and_cov(prices)
        h, J, _const = adapter_module._build_markowitz_qubo(mu, sigma, risk_aversion=d["risk_aversion"],
                                                             cardinality=cardinality,
                                                             cardinality_penalty=adapter.cardinality_penalty)
    if not result.get("success"):
        return {"failed": str(result.get("error"))[:200]}
    cost = pair_costs(h, J, n)
    selection = [1 if w > 0 else 0 for w in result["portfolio_weights"].values()]
    out = score(cost, index_of(selection))
    if circuit is not None:
        out["state"] = state_metrics(exact_probabilities(circuit), cost)
    if first is not None:
        out["state_initial"] = state_metrics(exact_probabilities(first), cost)
    out["objective_evaluations"] = int(result["convergence_metrics"]["total_evaluations"])
    out["parameters"] = int(result["resource_usage"]["parameter_count"])
    out["optimum_cardinality"] = int(states(n)[int(np.argmin(cost))].sum())
    return out


_LEGACY = {}


def legacy_engine():
    if not _LEGACY:
        spec = importlib.util.spec_from_file_location("refinement_engine_as_shipped", LEGACY_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LEGACY["module"] = module
    return _LEGACY["module"]


@contextlib.contextmanager
def refinement(shipped: bool):
    import core.postprocessing.refinement_engine as live

    saved = live.refine_binary_solution
    if shipped:
        live.refine_binary_solution = legacy_engine().refine_binary_solution
    try:
        yield
    finally:
        live.refine_binary_solution = saved


def bfdcqo_inputs(prices):
    """(mu, Sigma) exactly as IonQBFDCQOOptimizer.optimize computes them."""
    from core.preprocessing import ledoit_wolf_covariance

    log_returns = np.log(prices / prices.shift(1)).dropna()
    sigma, _ = ledoit_wolf_covariance(log_returns, annualization_factor=252.0)
    return log_returns.mean().values * 252, sigma


def run_bfdcqo(prices, seed: int, null: bool, shipped_refinement: bool, risk_aversion=None) -> dict:
    import core.optimizers.quantum.ionq.ionq_optimizer as ionq

    d = DEFAULTS["bfdcqo"]
    n = prices.shape[1]
    k = max(1, n // 3)
    mu, sigma = bfdcqo_inputs(prices)
    h, J = ionq._build_qubo(mu, sigma, k)  # the call _run_bfdcqo_cluster makes: risk_aversion left at its default
    cost = ising_costs(h, J, n)
    gamma = d["risk_aversion"] if risk_aversion is None else risk_aversion
    with patched(seed, null), refinement(shipped_refinement):
        if null:
            rng = np.random.default_rng(seed)

            def uniform_counts(circuit, shots):
                width = circuit.num_qubits
                values, counts = np.unique(rng.integers(0, 2 ** width, shots), return_counts=True)
                return {format(int(v), f"0{width}b"): int(c) for v, c in zip(values, counts)}

            optimizer = ionq.IonQBFDCQOOptimizer(risk_aversion=gamma, execute_fn=uniform_counts, backend_name="uniform-null")
        else:
            backend, name = ionq._get_backend(None, d["noise_model"], False)
            backend.set_options(seed_simulator=int(seed % (2 ** 31 - 1)) + 1)
            optimizer = ionq.IonQBFDCQOOptimizer(risk_aversion=gamma, backend=backend, backend_name=name)
        result = optimizer.optimize(prices, cardinality_K=k, qmax=d["qmax"], n_iterations=d["n_iterations"],
                                    shots=d["shots"], ansatz_depth=d["ansatz_depth"])
    if not result.get("success"):
        return {"failed": str(result.get("error"))[:200]}
    selection = [1 if w > 0 else 0 for w in result["allocation"].values()]
    out = score(cost, index_of(selection))
    metrics = result["quantum_metrics"]
    out["cardinality"] = k
    out["clusters"] = int(metrics["n_clusters"])
    out["is_first_k_assets"] = bool(selection == [1] * k + [0] * (n - k))
    out["selected_assets"] = [s for s, w in result["allocation"].items() if w > 0]
    if metrics["n_clusters"] == 1:
        # one cluster holds every asset: the sampler's best energy can be compared with the exact minimum
        out["sampler_found_optimum_before_refinement"] = bool(
            abs(metrics["cluster_metrics"][0]["best_energy"] - cost.min()) <= 1e-9 * max(1.0, abs(cost.min())))
    out["cluster_best_energies"] = [float(c["best_energy"]) for c in metrics["cluster_metrics"]]
    return out


def pce_reference(linear, quadratic) -> dict:
    """Exact minimum of PCE's own cost over the simplex, and the cost of equal weights."""
    n = len(linear)
    sym = (quadratic + quadratic.T) / 2.0

    def f(w):
        return float(linear @ w + w @ quadratic @ w)

    def grad(w):
        return linear + 2.0 * sym @ w

    constraint = ({"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},)
    rng = np.random.default_rng(0)
    best = None
    for start in [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(4)]:
        res = minimize(f, start, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=constraint,
                       options={"ftol": 1e-15, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return {"optimum": float(best.fun), "equal_weights": f(np.full(n, 1.0 / n)),
            "min_eigenvalue": float(np.linalg.eigvalsh(sym).min()), "optimum_weights": [float(v) for v in best.x]}


def run_pce(prices, seed: int, null: bool) -> dict:
    import core.optimizers.quantum.ibm.pce_portfolio_optimizer as pce

    d = DEFAULTS["pce"]
    n = prices.shape[1]
    with patched(seed, null):
        config = pce.PCEConfig(compression_factor=d["compression_factor"], ansatz_depth=d["ansatz_depth"], shots=d["shots"],
                               max_iterations=d["max_iterations"], max_no_improvement_rounds=d["max_no_improvement_rounds"],
                               restart_threshold=d["restart_threshold"], optimization_timeout=d["optimization_timeout"],
                               risk_aversion=d["risk_aversion"])
        optimizer = pce.PCEPortfolioOptimizer.create_deterministic(int(seed % (2 ** 31 - 1)), config)
        linear, quadratic, _ = optimizer.build_portfolio_qubo(prices)
        if null:
            optimizer.num_qubits = optimizer.calculate_required_qubits(n)
            draws = np.random.default_rng(seed).integers(0, 2 ** optimizer.num_qubits, config.shots)
            weights = optimizer._extract_portfolio_solution(dict(Counter(int(v) for v in draws)), linear, quadratic)
        else:
            result = optimizer.optimize(prices)
            if not result.get("success", False):
                return {"failed": str(result.get("error"))[:200]}
            weights = result["portfolio_weights"]
        w = np.array([float(v) for v in weights.values()])
        cost = float(optimizer._evaluate_portfolio_cost(list(w), linear, quadratic))
        qubits = int(optimizer.num_qubits)
    ref = pce_reference(np.asarray(linear, float), np.asarray(quadratic, float))
    return {"cost": cost, "excess": cost - ref["optimum"], "equal_weights_excess": ref["equal_weights"] - ref["optimum"],
            "convex": bool(ref["min_eigenvalue"] >= -1e-12), "qubits": qubits, "weight_sum": float(w.sum()),
            "largest_weight": float(w.max()), "distinct_weight_values": int(len({round(float(v), 9) for v in w}))}


def run_task(task):
    arm, n, instance, rep, null = task
    logging.disable(logging.CRITICAL)
    prices = make_prices(n, INSTANCE_BASE + 100 * (n // 4) + instance)
    seed = point_seed(19, ARM_ID[arm], n, instance, rep, int(null))
    started = time.time()
    try:
        if arm == "qaoa":
            out = run_qaoa(prices, seed, null)
        elif arm in ("cvar_all", "cvar_free"):
            out = run_cvar(prices, seed, null, select_all=arm == "cvar_all")
        elif arm in ("bfdcqo_shipped", "bfdcqo_fixed"):
            out = run_bfdcqo(prices, seed, null, shipped_refinement=arm == "bfdcqo_shipped")
        else:
            out = run_pce(prices, seed, null)
    except Exception as exc:  # recorded, never dropped
        out = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
    out.update(arm=arm, n=n, instance=instance, rep=rep, null=null, seed=seed, wall_s=round(time.time() - started, 2))
    return out


# --------------------------------------------------------------------------- deterministic checks of the shipped code


def audit() -> dict:
    logging.disable(logging.CRITICAL)
    import core.optimizers.quantum.ibm.cvar_vqe_optimizer as cvar_optimizer
    import core.optimizers.quantum.ibm.pce_portfolio_optimizer as pce
    import core.optimizers.quantum.ionq.ionq_optimizer as ionq
    import core.postprocessing.refinement_engine as live
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp

    legacy = legacy_engine()
    out = {}
    n, k = 8, 2
    engine = {"instances": 0, "shipped_minimum_is_first_k": 0, "shipped_moves_exact_optimum_to_first_k": 0,
              "shipped_energy_matches_qubo": 0, "fixed_energy_matches_qubo": 0, "fixed_keeps_exact_optimum": 0,
              "exact_optimum_is_first_k": 0}
    s = states(n)
    feasible = s.sum(axis=1) == k
    first_k = [1] * k + [0] * (n - k)
    returns_ignored = 0
    for instance in range(INSTANCES):
        prices = make_prices(n, INSTANCE_BASE + 100 * (n // 4) + instance)
        mu, sigma = bfdcqo_inputs(prices)
        h, J = ionq._build_qubo(mu, sigma, k)
        cost = ising_costs(h, J, n)
        optimum = s[int(np.argmin(cost))].astype(int)
        engine["instances"] += 1
        engine["exact_optimum_is_first_k"] += int(optimum.tolist() == first_k)
        for label, module in (("shipped", legacy), ("fixed", live)):
            hh, JJ = module.build_ising_from_portfolio(mu, sigma, k, 1.0)
            energy = np.array([module.ising_energy(x.astype(int), hh, JJ) for x in s])
            engine[f"{label}_energy_matches_qubo"] += int(np.allclose(energy - energy[0], cost - cost[0], atol=1e-8))
            if label == "shipped":
                best_feasible = s[feasible][int(np.argmin(energy[feasible]))].astype(int)
                engine["shipped_minimum_is_first_k"] += int(best_feasible.tolist() == first_k)
        moved = legacy.refine_binary_solution(optimum, mu, sigma, cardinality_K=k, risk_aversion=1.0, max_rounds=10)
        engine["shipped_moves_exact_optimum_to_first_k"] += int(moved.tolist() == first_k and optimum.tolist() != first_k)
        kept = live.refine_binary_solution(optimum, mu, sigma, cardinality_K=k, risk_aversion=1.0, max_rounds=10)
        engine["fixed_keeps_exact_optimum"] += int(kept.tolist() == optimum.tolist())
        # returns: with the penalty held fixed, the cluster QUBO must not change when the returns do
        h1, J1 = ionq._build_qubo(mu, sigma, k, penalty_lambda=3.0)
        h2, J2 = ionq._build_qubo(3.0 * mu + 0.2, sigma, k, penalty_lambda=3.0)
        returns_ignored += int(np.allclose(h1, h2) and np.allclose(J1, J2))
        variance = np.einsum("bi,ij,bj->b", s, sigma, s)
        engine.setdefault("qubo_optimum_is_min_variance_k_subset", 0)
        engine["qubo_optimum_is_min_variance_k_subset"] += int(
            optimum.tolist() == s[feasible][int(np.argmin(variance[feasible]))].astype(int).tolist())
    out["refinement_engine"] = engine

    prices = make_prices(n, INSTANCE_BASE + 100 * (n // 4))
    mu, sigma = bfdcqo_inputs(prices)
    h, J = ionq._build_qubo(mu, sigma, k)
    circuit = ionq._build_bfdcqo_circuit(n, h, J, np.zeros(n), DEFAULTS["bfdcqo"]["ansatz_depth"])
    same = []
    for gamma in (0.0, 0.5, 1.0):
        r = run_bfdcqo(prices, 12345, null=False, shipped_refinement=False, risk_aversion=gamma)
        same.append((r["selected_assets"], r["cluster_best_energies"]))
    out["bfdcqo"] = {
        "circuit_gates": sorted(circuit.count_ops().keys()),
        "has_y_type_gate": any(g in circuit.count_ops() for g in ("ry", "ryy", "rzy", "ryz", "y", "sdg", "s")),
        "cluster_qubo_unchanged_when_returns_change": {"instances": engine["instances"], "unchanged": returns_ignored},
        "risk_aversion_changes_result": bool(any(x != same[0] for x in same[1:])),
        "risk_aversion_values_tried": [0.0, 0.5, 1.0],
        "shots_per_iteration": DEFAULTS["bfdcqo"]["shots"], "largest_cluster_states": 2 ** DEFAULTS["bfdcqo"]["qmax"],
        "shots_over_states": DEFAULTS["bfdcqo"]["shots"] / 2 ** DEFAULTS["bfdcqo"]["qmax"],
        "p_uniform_misses_a_given_state_in_one_iteration": float((1 - 2.0 ** -DEFAULTS["bfdcqo"]["qmax"]) ** DEFAULTS["bfdcqo"]["shots"]),
    }

    d = DEFAULTS["cvar"]
    with patched(4242, null=False):
        config = cvar_optimizer.CVaRConfig(simulator_mode=True, alpha=d["alpha"], adaptive_alpha=d["adaptive_alpha"],
                                           max_cvar_iterations=d["max_cvar_iterations"], measurement_budget=d["measurement_budget"])
        optimizer = cvar_optimizer.CVaRVQEOptimizer(config)
        ansatz = RealAmplitudes(num_qubits=4, reps=1)
        observable = SparsePauliOp.from_list([("ZIII", 1.0), ("IZZI", 0.5), ("IIIZ", -0.7)])
        identical, samples = 0, []
        for trial in range(20):
            bound = ansatz.assign_parameters(np.random.default_rng(trial).uniform(-np.pi, np.pi, ansatz.num_parameters))
            value, metrics = optimizer.enhanced_energy_evaluation(bound, observable)
            identical += int(value == metrics["standard_energy"])
            samples.append(metrics["cvar_metrics"]["n_samples"])
    production = cvar_optimizer.CVaRConfig.__new__(cvar_optimizer.CVaRConfig)
    production.simulator_mode, production.measurement_budget = False, 8000
    out["cvar_vqe"] = {
        "simulator_measurement_rounds": config.measurement_rounds, "simulator_shots_per_circuit": config.shots_per_circuit,
        "objective_equals_mean_energy": {"evaluations": 20, "identical": identical},
        "values_entering_the_cvar": sorted(set(samples)),
        "production_measurement_rounds": production.measurement_rounds,
        "production_shots_per_circuit": production.shots_per_circuit,
        "final_sampling_shots_simulator": d["sampling_shots"], "max_objective_evaluations": d["max_cvar_iterations"],
    }

    with patched(4243, null=False):
        optimizer = pce.PCEPortfolioOptimizer.create_deterministic(1, pce.PCEConfig(shots=100))
        rng = np.random.default_rng(3)
        equal = 0
        for _ in range(20):
            counts = dict(Counter(int(v) for v in rng.integers(0, 16, 100)))
            x, y, z = optimizer._compute_pauli_expectations(counts, ["XIII", "YIII", "ZIII"])
            equal += int(x == y == z)
        prices12 = make_prices(12, INSTANCE_BASE + 300)
        linear, quadratic, _ = optimizer.build_portfolio_qubo(prices12)
        optimizer.num_qubits = optimizer.calculate_required_qubits(12)
        optimizer._bit_swap_post_processing = lambda weights, lin, quad: weights  # look at the decode before the swap search
        raw = [list(optimizer._extract_portfolio_solution({v: 1}, linear, quadratic).values()) for v in (0, 21, 63)]
    m = optimizer.num_qubits
    tail_ratio = [[w[i] / w[m] for i in range(m, 12)] for w in raw]
    out["pce"] = {
        "x_y_z_expectations_equal_from_z_counts": {"trials": 20, "equal": equal},
        "assets": 12, "qubits": m,
        "assets_whose_weight_depends_on_a_measured_bit": m,
        "remaining_assets_get_one_constant_value": bool(np.allclose(tail_ratio, 1.0)),
        "raw_weight_levels_before_normalisation": sorted({round(0.5 * (1 + float(np.tanh(sgn))), 4) for sgn in (-1, 1)} | {0.1}),
    }

    select_all = {}
    import core.optimizers.quantum.ibm.cvar_vqe_adapter as adapter_module

    for size in SIZES:
        hits = 0
        for instance in range(INSTANCES):
            prices = make_prices(size, INSTANCE_BASE + 100 * (size // 4) + instance)
            mu, sigma, _, _ = adapter_module._annualised_returns_and_cov(prices)
            h, J, _ = adapter_module._build_markowitz_qubo(mu, sigma, risk_aversion=d["risk_aversion"], cardinality=size,
                                                           cardinality_penalty=max(1.0, 5.0 * d["risk_aversion"]))
            hits += int(int(np.argmin(pair_costs(h, J, size))) == 2 ** size - 1)
        select_all[str(size)] = {"instances": INSTANCES, "optimum_is_all_assets": hits}
    out["cvar_cardinality_equals_n"] = select_all
    return out


def relabelling() -> dict:
    """BF-DCQO on reordered columns: a selection method must return the same assets."""
    logging.disable(logging.CRITICAL)
    n = 8
    out = {}
    for label, shipped in (("shipped", True), ("fixed", False)):
        same = 0
        for instance in range(INSTANCES):
            prices = make_prices(n, INSTANCE_BASE + 100 * (n // 4) + instance)
            order = list(np.random.default_rng(700 + instance).permutation(prices.columns))
            a = run_bfdcqo(prices, 555 + instance, null=False, shipped_refinement=shipped)
            b = run_bfdcqo(prices[order], 555 + instance, null=False, shipped_refinement=shipped)
            same += int(sorted(a["selected_assets"]) == sorted(b["selected_assets"]))
        out[label] = {"instances": INSTANCES, "same_assets_after_reordering": same}
    return out


# --------------------------------------------------------------------------- summary


def per_instance(rows, field, sub=None):
    by = {}
    for r in rows:
        value = r.get(sub, {}).get(field) if sub else r.get(field)
        if value is not None:
            by.setdefault(r["instance"], []).append(float(value))
    return {k: float(np.mean(v)) for k, v in sorted(by.items())}


def summarise(rows):
    out = {}
    for arm in ARMS:
        for n in SIZES:
            cell = {}
            for null in (False, True):
                sel = [r for r in rows if r["arm"] == arm and r["n"] == n and r["null"] == null]
                ok = [r for r in sel if "failed" not in r]
                entry = {"runs": len(sel), "failed": len(sel) - len(ok),
                         "mean_wall_s": float(np.mean([r["wall_s"] for r in sel])) if sel else None}
                fields = ["excess", "equal_weights_excess"] if arm == "pce" else ["gap", "rank_pct", "optimal", "selected"]
                for field in fields:
                    entry[field] = mean_ci(list(per_instance(ok, field).values()))
                    entry[f"{field}_by_instance"] = per_instance(ok, field)
                if arm != "pce":
                    entry["optimal_runs"] = int(sum(r["optimal"] for r in ok))
                if ok and "state" in ok[0]:
                    for field in ("p_optimum_over_uniform", "approx_ratio", "p_optimum"):
                        entry[f"state_{field}"] = mean_ci(list(per_instance(ok, field, "state").values()))
                    ratios = [r["state"]["p_optimum_over_uniform"] for r in ok if "state" in r]
                    entry["state_p_optimum_over_uniform_median"] = float(np.median(ratios))
                    entry["state_p_optimum_over_uniform_range"] = [float(min(ratios)), float(max(ratios))]
                if ok and "state_initial" in ok[0]:
                    for field in ("p_optimum_over_uniform", "approx_ratio"):
                        entry[f"state_initial_{field}"] = mean_ci(list(per_instance(ok, field, "state_initial").values()))
                    change = [r["state"]["approx_ratio"] - r["state_initial"]["approx_ratio"] for r in ok
                              if "state" in r and "state_initial" in r]
                    entry["training_change_in_approx_ratio"] = mean_ci(change)
                    entry["runs_where_training_raised_approx_ratio"] = int(sum(c > 1e-12 for c in change))
                if arm == "qaoa":
                    entry["returned_is_most_frequent_sample_runs"] = int(sum(r["returned_is_most_frequent_sample"] for r in ok))
                    for field in ("gap", "optimal", "rank_pct"):
                        entry[f"lowest_cost_rule_{field}"] = mean_ci(list(per_instance(ok, field, "lowest_cost_rule").values()))
                    entry["lowest_cost_rule_gap_by_instance"] = per_instance(ok, "gap", "lowest_cost_rule")
                    entry["distinct_final_samples"] = mean_ci([r["distinct_final_samples"] for r in ok])
                    entry["largest_sample_count"] = mean_ci([r["largest_sample_count"] for r in ok])
                    entry["init_schemes"] = dict(Counter(r["init_scheme"] for r in ok))
                    entry["optimum_cardinality"] = sorted({r["optimum_cardinality"] for r in ok})
                if arm.startswith("cvar"):
                    entry["objective_evaluations"] = sorted({r["objective_evaluations"] for r in ok})
                    entry["parameters"] = sorted({r["parameters"] for r in ok})
                    entry["optimum_cardinality"] = sorted({r["optimum_cardinality"] for r in ok})
                if arm.startswith("bfdcqo"):
                    entry["first_k_assets_runs"] = int(sum(r["is_first_k_assets"] for r in ok))
                    entry["clusters"] = sorted({r["clusters"] for r in ok})
                    entry["cardinality"] = sorted({r["cardinality"] for r in ok})
                    before = [r["sampler_found_optimum_before_refinement"] for r in ok if "sampler_found_optimum_before_refinement" in r]
                    entry["sampler_found_optimum_before_refinement"] = {"runs": len(before), "optimal": int(sum(before))}
                if arm == "pce":
                    entry["convex_runs"] = int(sum(r["convex"] for r in ok))
                    entry["qubits"] = sorted({r["qubits"] for r in ok})
                    entry["distinct_weight_values"] = sorted({r["distinct_weight_values"] for r in ok})
                cell["null" if null else "method"] = entry
            field = "excess" if arm == "pce" else "gap"
            a, b = cell["method"][f"{field}_by_instance"], cell["null"][f"{field}_by_instance"]
            common = sorted(set(a) & set(b))
            diff = [a[i] - b[i] for i in common]
            test = {"instances": len(common), "method_minus_null": mean_ci(diff),
                    "method_better": int(sum(d < -1e-12 for d in diff)), "null_better": int(sum(d > 1e-12 for d in diff)),
                    "ties": int(sum(abs(d) <= 1e-12 for d in diff))}
            test["wilcoxon_p"] = float(wilcoxon(diff).pvalue) if any(abs(d) > 1e-12 for d in diff) else None
            cell["method_vs_null"] = test
            if arm == "qaoa":
                a2 = cell["method"]["lowest_cost_rule_gap_by_instance"]
                b2 = cell["null"]["lowest_cost_rule_gap_by_instance"]
                d2 = [a2[i] - b2[i] for i in sorted(set(a2) & set(b2))]
                cell["lowest_cost_rule_method_vs_null"] = {
                    "method_minus_null": mean_ci(d2), "method_better": int(sum(d < -1e-12 for d in d2)),
                    "null_better": int(sum(d > 1e-12 for d in d2)),
                    "wilcoxon_p": float(wilcoxon(d2).pvalue) if any(abs(d) > 1e-12 for d in d2) else None}
            out[f"{arm}|{n}"] = cell
    return out


def figure(summary) -> None:
    """Rank of the returned selection, method against null.

    The gap is not drawn: in a QUBO with a cardinality penalty every selection of
    the right size has a gap near zero, so it cannot tell the BF-DCQO arms apart.
    """
    plt = mpl()
    arms = [a for a in ARMS if a != "pce"]
    labels = ["QAOA", "CVaR-VQE\nK = n", "CVaR-VQE\nfree", "BF-DCQO\nshipped", "BF-DCQO\nfixed"]
    fig, axes = plt.subplots(1, len(SIZES), figsize=(6.8, 2.5), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], SIZES):
        for offset, key, colour, marker, name in ((-0.13, "method", "C0", "o", "method"), (0.13, "null", "0.45", "x", "uniform null")):
            cells = [summary[f"{a}|{n}"][key]["rank_pct"] for a in arms]
            ax.errorbar(np.arange(len(arms)) + offset, [c["mean"] for c in cells], yerr=[c["ci95"] for c in cells],
                        fmt=marker, color=colour, ms=3.5, lw=0.8, capsize=1.5, label=name)
        ax.axhline(50.0, color="grey", lw=0.5, ls=":")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(-0.15, 100)
        ax.set_yticks([0, 1, 10, 50, 100])
        ax.set_yticklabels(["0", "1", "10", "50", "100"])
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_title(f"{n} assets", fontsize=7)
    axes[0][0].set_ylabel("selections that are better (%)\n(0 = exact optimum)")
    axes[0][-1].legend(fontsize=5.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "portfolio_referee.pdf")


def main() -> None:
    if "--figure" in sys.argv:  # redraw from the stored results, no runs
        import json

        from common import RESULTS

        stored = json.loads((RESULTS / "e19_portfolio.json").read_text())
        SIZES[:] = stored["sizes"]
        figure(stored["summary"])
        return
    started = time.time()
    tasks = [(arm, n, instance, rep, null) for arm in ARMS for n in SIZES for instance in range(INSTANCES)
             for rep in range(SEEDS) for null in (False, True)]
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        rows = list(pool.map(run_task, tasks, chunksize=4))
    failed = [r for r in rows if "failed" in r]
    print(f"{len(rows)} runs, {len(failed)} failed, {time.time() - started:.0f} s", flush=True)
    for r in failed[:10]:
        print("  failed:", r["arm"], r["n"], r["instance"], r["rep"], r["null"], r["failed"], flush=True)
    checks = audit()
    relabel = relabelling()
    summary = summarise(rows)
    enumeration = {}
    for n in SIZES:
        rng = np.random.default_rng(n)
        q, Q = rng.normal(size=n), rng.normal(size=(n, n))
        states(n)
        tick = time.perf_counter()
        for _ in range(5):
            pair_costs(q, Q, n).argmin()
        enumeration[str(n)] = {"selections": 2 ** n, "seconds": (time.perf_counter() - tick) / 5}
    for key, cell in summary.items():
        arm = key.split("|")[0]
        field = "excess" if arm == "pce" else "gap"
        m, u, t = cell["method"], cell["null"], cell["method_vs_null"]
        extra = "" if arm == "pce" else f" optimal {m['optimal_runs']}/{m['runs']} vs {u['optimal_runs']}/{u['runs']}"
        print(f"{key:20s} {field} {m[field]['mean']:.4f}±{m[field]['ci95']:.4f} vs null {u[field]['mean']:.4f}±{u[field]['ci95']:.4f}"
              f" p={t['wilcoxon_p']}{extra}", flush=True)
    print("audit:", checks, flush=True)
    print("relabelling:", relabel, flush=True)
    name = "e19_portfolio_quick" if QUICK else "e19_portfolio"
    save(name, {"design": __doc__.split("Design\n------\n")[1].split("Usage:")[0].strip(), "defaults": DEFAULTS,
                "sizes": SIZES, "instances": INSTANCES, "seeds": SEEDS, "instance_seed_base": INSTANCE_BASE,
                "exploratory_probe_instance_seeds": [1, 2, 3], "audit": checks, "relabelling": relabel,
                "exact_enumeration": enumeration,
                "summary": summary, "rows": rows, "wall_s": round(time.time() - started, 1)})
    if not QUICK:
        figure(summary)


if __name__ == "__main__":
    main()
