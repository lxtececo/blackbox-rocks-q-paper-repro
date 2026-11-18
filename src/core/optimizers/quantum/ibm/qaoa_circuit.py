"""
IBM Quantum QAOA Circuit Implementation

Provides QAOA (Quantum Approximate Optimization Algorithm) portfolio optimization
using Qiskit. Moved from the legacy core/optimizers/hybrid_portfolio_opt.py.
"""

import sys
import time
from typing import Optional
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import generate_preset_pass_manager
from scipy.optimize import minimize

from core.portfolio_referee import referee
from core.quantum_math import pauli_labels_for_qubit_pair, qubo_pair_to_ising


import logging
logger = logging.getLogger(__name__)


def classical_optimize(prices: pd.DataFrame):
    """
    Classical mean-variance optimization using PyPortfolioOpt.

    Args:
        prices: DataFrame of closing prices (dates x symbols).

    Returns:
        Tuple of (weights dict, performance tuple (return, volatility, sharpe)).
    """
    logger.info("[QAOA-Classical] Computing expected returns and covariance")
    mu = expected_returns.mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    logger.info("[QAOA-Classical] Optimization complete: return=%.4f, vol=%.4f, sharpe=%.4f", *perf)
    return weights, perf


def _build_qubo(prices: pd.DataFrame, risk_aversion: float = 0.5):
    """Build QUBO coefficient arrays for the portfolio problem.

    cost(x) = Σ_i linear_i x_i + Σ_{i<j} Q_ij x_i x_j with
      linear_i = −μ_i + γ S_ii      (variance diagonal folds into linear
                                     because x_i² = x_i for binary vars —
                                     previously the diagonal was silently
                                     DROPPED by _build_hamiltonian)
      Q_ij     = 2 γ S_ij           (upper-triangular pair coefficient;
                                     stored once per triangle in the full
                                     symmetric matrix for backwards compat
                                     with _build_hamiltonian's i<j reads)
    """
    num_assets = prices.shape[1]
    mu = expected_returns.mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    linear = np.array([-mu.iloc[i] + risk_aversion * S.iloc[i, i] for i in range(num_assets)])
    quadratic = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            # x^T(γS)x contributes 2γS_ij x_i x_j per pair; each triangle
            # carries the full pair coefficient (consumed once, i<j, by
            # _build_hamiltonian). Previously γS_ij — half the intended
            # risk coupling.
            quadratic[i, j] = 2.0 * risk_aversion * S.iloc[i, j]
            quadratic[j, i] = 2.0 * risk_aversion * S.iloc[i, j]
    return linear, quadratic, num_assets


def _build_hamiltonian(linear: np.ndarray, quadratic: np.ndarray, num_assets: int) -> SparsePauliOp:
    """Build cost Hamiltonian from QUBO coefficients.

    The QUBO is cost(x) = Σ_i linear_i x_i + Σ_{i<j} Q_ij x_i x_j with
    x_i ∈ {0,1}. ``quadratic`` is a full symmetric matrix where each pair
    (i,j) carries the full pair coefficient once in each triangle, so the
    pair coefficient is ``quadratic[i, j]`` (= ``quadratic[j, i]``).

    Conversion delegated to the shared authority in ``core.quantum_math``
    (F29) — the offset is included as an identity Pauli term so the
    reported ``objective_value`` equals the true QUBO cost of the optimum.
    """
    n = num_assets
    # Pair-form Q (upper-triangular) from the symmetric input.
    Q_pair = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Q_pair[i, j] = quadratic[i, j]
    z_coeffs, J, offset = qubo_pair_to_ising(np.asarray(linear, dtype=float), Q_pair)

    pauli_terms = []
    if abs(offset) > 1e-15:
        pauli_terms.append(('I' * n, offset))
    for i in range(n):
        if abs(z_coeffs[i]) < 1e-15:
            continue
        pauli_terms.append((pauli_labels_for_qubit_pair(n, i), z_coeffs[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) < 1e-15:
                continue
            pauli_terms.append((pauli_labels_for_qubit_pair(n, i, j), J[i, j]))
    return SparsePauliOp.from_list(pauli_terms)


def _qubo_cost_of_key(linear: np.ndarray, quadratic: np.ndarray, num_assets: int):
    """QUBO cost of a measured outcome given as an integer whose bit i is asset i."""
    lin = np.asarray(linear, dtype=float)
    upper = np.triu(np.asarray(quadratic, dtype=float), 1)

    def cost(key: int) -> float:
        x = np.array([(key >> i) & 1 for i in range(num_assets)], dtype=float)
        return float(lin @ x + x @ upper @ x)

    return cost


def _select_sample(counts: dict, cost) -> tuple:
    """The sampled outcome with the lowest cost; ties go to the more frequent one.

    Returns (key, details); the details also report the most frequent outcome.
    """
    costs = {key: cost(key) for key in counts}
    best = min(counts, key=lambda key: (costs[key], -counts[key]))
    most_frequent = max(counts, key=counts.get)
    return best, {
        'rule': 'lowest_cost_sample',
        'shots': int(sum(counts.values())),
        'distinct_samples': len(counts),
        'selected_cost': costs[best],
        'selected_sample_count': int(counts[best]),
        'most_frequent_cost': costs[most_frequent],
        'most_frequent_sample_count': int(counts[most_frequent]),
        'most_frequent_is_selected': bool(best == most_frequent),
    }


def _run_simulator(ansatz, hamiltonian, num_assets: int, init_params: np.ndarray, cost,
                   estimator_shots: int = 100, sampler_shots: int = 200,
                   max_iterations: int = 15, seed: Optional[int] = None) -> dict:
    """Run QAOA optimization on the Aer simulator."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    aer = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer, optimization_level=1)
    shot_seeds = None if seed is None else np.random.default_rng(seed)

    def _sampler():
        sampler = Sampler(mode=aer)
        if shot_seeds is not None:
            sampler.options.simulator.seed_simulator = int(shot_seeds.integers(1, 2 ** 31 - 1))
        return sampler

    def _bitstring_to_z(bitstr, n):
        bits = np.array([int(b) for b in format(bitstr, f'0{n}b')])
        return 1 - 2 * bits

    def _compute_expectation(counts, hamiltonian, n):
        exp = 0.0
        shots = sum(counts.values())
        for bitstr, prob in counts.items():
            z = _bitstring_to_z(bitstr, n)
            value = 0.0
            for label, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
                term = coeff.real
                for idx, op in enumerate(label):
                    if op == 'Z':
                        term *= z[idx]
                value += term
            exp += (prob / shots) * value
        return exp

    sampler_job_count = 0
    # 7-QAP.25 Slice 4a.2 — per-iteration cost trace for diagnostics.
    # We also capture the parameter vector (β/γ for QAOA) and the
    # per-iteration wallclock so the P3 (parameter trajectory) and P9
    # (wallclock-per-iter) charts have real data instead of P1-only.
    objective_func_vals: list[float] = []
    iter_params: list[list[float]] = []
    iter_wall_s: list[float] = []
    _last_t = time.perf_counter()

    def cost_func(params):
        nonlocal sampler_job_count, _last_t
        circ = ansatz.assign_parameters(params)
        isa_circ = pm.run(circ)
        sampler = _sampler()
        sampler_job_count += 1
        logger.debug("[QAOA-Simulator] Submitting sampler job #%d", sampler_job_count)
        job = sampler.run([isa_circ], shots=estimator_shots)
        result = job.result()
        counts = result[0].data.meas.get_int_counts()
        cost = _compute_expectation(counts, hamiltonian, num_assets)
        objective_func_vals.append(float(cost))
        try:
            iter_params.append([float(p) for p in params])
        except Exception:
            iter_params.append([])
        _now = time.perf_counter()
        iter_wall_s.append(round(_now - _last_t, 6))
        _last_t = _now
        return cost

    logger.info("[QAOA-Simulator] Starting COBYLA minimization (maxiter=%d)", max_iterations)
    opt_result = minimize(cost_func, init_params, method="COBYLA", options={"maxiter": max_iterations})
    logger.info("[QAOA-Simulator] Minimization complete, running final sampling")

    final_circ = ansatz.assign_parameters(opt_result.x)
    isa_final_circ = pm.run(final_circ)
    sampler = _sampler()
    sampler_job_count += 1
    job = sampler.run([isa_final_circ], shots=sampler_shots)
    result = job.result()
    counts = result[0].data.meas.get_int_counts()
    best_bitstr, selection = _select_sample(counts, cost)
    # Qiskit big-endian: leftmost bit = qubit n-1. Reverse so solution[i]
    # is asset i (previously index-mirrored).
    solution = [int(b) for b in format(best_bitstr, f'0{num_assets}b')][::-1]

    logger.info("[QAOA-Simulator] Total sampler jobs: %d, solution: %s", sampler_job_count, solution)
    return {
        'solution': solution,
        'objective_value': opt_result.fun,
        'selection': selection,
        'final_parameters': [float(p) for p in opt_result.x],
        'seed': seed,
        'simulator_sampler_jobs_executed': sampler_job_count,
        'optimization_history': [
            {
                'i': idx,
                'cost': float(c),
                'params': iter_params[idx] if idx < len(iter_params) else None,
                'wall_s': iter_wall_s[idx] if idx < len(iter_wall_s) else None,
            }
            for idx, c in enumerate(objective_func_vals)
        ],
    }


def _run_real_backend(ansatz, hamiltonian, num_assets: int, init_params: np.ndarray, cost,
                      max_iterations: int = 12) -> dict:
    """Run QAOA optimization on a real IBM Quantum backend."""
    import os
    import time
    from datetime import datetime, timezone
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, SamplerV2 as Sampler

    # BYOK: prefer the per-tenant IBM Quantum token from the request-scoped
    # context (set by the API layer when the user has saved a Bring-Your-Own-
    # Key credential); fall back to the pod-wide IBM_QUANTUM_TOKEN env var
    # for legacy deployments. Without a token, QiskitRuntimeService() raises
    # "Unable to find account. Please make sure an account with the channel
    # name 'ibm_quantum_platform' is saved." — surface a friendly error so the
    # UI can prompt the user to add their key.
    try:
        from core.quantum_credential_context import tenant_token_for
        ibm_token = tenant_token_for("ibm_quantum")
    except Exception:  # noqa: BLE001
        ibm_token = None
    if not ibm_token:
        ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
    if not ibm_token:
        raise RuntimeError(
            "No IBM Quantum API token available. Add one via "
            "Settings → Quantum Credentials (BYOK) or set IBM_QUANTUM_TOKEN "
            "on the backend pod. Channel: ibm_quantum_platform."
        )

    logger.info("[QAOA-RealBackend] Initializing QiskitRuntimeService (channel=ibm_quantum_platform, byok=%s)",
                bool(ibm_token))
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
    backend = service.least_busy()
    logger.info("[QAOA-RealBackend] Using backend: %s", backend)
    backend_name = str(getattr(backend, "name", backend))

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(ansatz)

    objective_func_vals = []
    estimator_job_count = 0
    sampler_job_count = 0
    iter_params: list[list[float]] = []
    iter_wall_s: list[float] = []
    # F24 Phase A — capture vendor-side job IDs so the JobHistoryView can
    # render "View on IBM Quantum portal" deep-links for each Estimator /
    # Sampler submit. Simulator runs (Aer) never populate this list.
    vendor_runs: list[dict] = []
    _last_t = time.perf_counter()

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        nonlocal estimator_job_count, _last_t
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        estimator_job_count += 1
        jid = None
        try:
            jid = job.job_id()
        except Exception:  # noqa: BLE001 — best-effort id capture  # nosec B110 - best-effort cleanup; exception intentionally swallowed
            pass
        logger.info("[QAOA-RealBackend] Estimator job #%d submitted: %s", estimator_job_count, jid)
        if jid:
            vendor_runs.append({
                "vendor": "ibm",
                "vendor_label": "IBM Quantum",
                "job_id": jid,
                "stage": f"estimator #{estimator_job_count}",
                "backend": backend_name,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
        while not job.done():
            status = getattr(job, 'status', lambda: None)()
            logger.debug("[QAOA-RealBackend] Waiting for estimator job %s, status: %s", jid, status)
            if status == 'ERROR':
                logger.error("[QAOA-RealBackend] Estimator job %s ended in ERROR", jid)
                return None
            time.sleep(5)
        results = job.result()[0]
        cost = results.data.evs
        objective_func_vals.append(cost)
        try:
            iter_params.append([float(p) for p in params])
        except Exception:
            iter_params.append([])
        _now = time.perf_counter()
        iter_wall_s.append(round(_now - _last_t, 6))
        _last_t = _now
        return cost

    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1000
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
        options={"maxiter": max_iterations},
    )
    logger.info("[QAOA-RealBackend] Minimization complete (maxiter=%d)", max_iterations)

    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 10000
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    job = sampler.run([(optimized_circuit,)], shots=int(1e4))
    sampler_job_count += 1
    sjid = None
    try:
        sjid = job.job_id()
    except Exception:  # noqa: BLE001  # nosec B110 - best-effort cleanup; exception intentionally swallowed
        pass
    logger.info("[QAOA-RealBackend] Sampler job #%d submitted: %s", sampler_job_count, sjid)
    if sjid:
        vendor_runs.append({
            "vendor": "ibm",
            "vendor_label": "IBM Quantum",
            "job_id": sjid,
            "stage": f"sampler #{sampler_job_count}",
            "backend": backend_name,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })

    while not job.done():
        status = getattr(job, 'status', lambda: None)()
        logger.debug("[QAOA-RealBackend] Waiting for sampler job %s, status: %s", sjid, status)
        if status == 'ERROR':
            logger.error("[QAOA-RealBackend] Sampler job %s ended in ERROR", sjid)
            return {'solution': None, 'objective_value': None}
        time.sleep(5)

    counts_int = job.result()[0].data.meas.get_int_counts()
    best_bitstr, selection = _select_sample(counts_int, cost)
    # Qiskit big-endian: leftmost bit = qubit n-1. Reverse so solution[i]
    # is asset i (previously index-mirrored).
    solution = [int(b) for b in format(best_bitstr, f'0{num_assets}b')][::-1]

    logger.info("[QAOA-RealBackend] Solution: %s, objective: %.4f", solution, result.fun)
    logger.info(
        "[QAOA-RealBackend] Summary: %d estimator jobs, %d sampler jobs",
        estimator_job_count, sampler_job_count,
    )
    return {
        'solution': solution,
        'objective_value': result.fun,
        'selection': selection,
        'final_parameters': [float(p) for p in result.x],
        'estimator_jobs_executed': estimator_job_count,
        'sampler_jobs_executed': sampler_job_count,
        # F24 Phase A — IBM Runtime job IDs (Estimator + Sampler) for portal deep-links.
        'vendor_runs': vendor_runs,
        'optimization_history': [
            {
                'i': idx,
                'cost': float(c),
                'params': iter_params[idx] if idx < len(iter_params) else None,
                'wall_s': iter_wall_s[idx] if idx < len(iter_wall_s) else None,
            }
            for idx, c in enumerate(objective_func_vals)
        ],
    }


def _classical_warm_start(
    prices: pd.DataFrame,
    linear: np.ndarray,
    quadratic: np.ndarray,
    num_assets: int,
) -> dict:
    """A classical selection and its QUBO cost, reported next to the QAOA result for comparison.

    Runs the classical mean-variance step, thresholds its weights into a
    selection and evaluates the QUBO the quantum path minimises on it.

    The circuit does not use this selection. Its initial state is the uniform
    superposition and its initial angles are a fixed vector
    (``_interp_init_params``). The only effect on the run is the choice of
    that vector: the fixed ramp when this step succeeds, random angles with a
    fixed seed when it fails. There is no fallback to the classical selection
    when the sampled one is worse.

    The name of the function, the result key ``warm_start`` and the scheme
    name ``interp_linear_ramp`` are historical: nothing is warm-started. They
    are kept because stored results and the experiment scripts of the paper
    read them.

    Any exception (missing history, PyPortfolioOpt convergence issue,
    all-NaN input) is caught and reported in ``fallback_reason``.
    """
    telemetry: dict = {
        "source": "classical_mvo",
        "bitstring": None,
        "qubo_value": None,
        "n_selected": 0,
        "init_scheme": "random_uniform",
        "fallback_reason": None,
    }
    try:
        weights, _perf = classical_optimize(prices)
        # Rank symbols by weight, pick strictly positive as selected.
        symbols = list(prices.columns)
        pairs = sorted(
            ((symbols.index(s), float(w)) for s, w in weights.items() if s in symbols),
            key=lambda p: -p[1],
        )
        bitstring = [0] * num_assets
        for idx, w in pairs:
            if w > 1e-4:
                bitstring[idx] = 1
        # QUBO objective on {0,1}^n; ``quadratic`` holds each pair coefficient in both triangles.
        x = np.asarray(bitstring, dtype=float)
        qubo_value = float(x @ linear + x @ np.triu(quadratic, 1) @ x)
        telemetry.update(
            bitstring=bitstring,
            qubo_value=qubo_value,
            n_selected=int(sum(bitstring)),
            init_scheme="interp_linear_ramp",
        )
    except Exception as exc:  # noqa: BLE001
        telemetry["fallback_reason"] = f"classical_warm_start_failed:{type(exc).__name__}"
        logger.warning(
            "[QAOA-Init] classical comparison step failed (%s); training starts from random angles with a fixed seed",
            exc,
        )
    return telemetry


def _interp_init_params(num_params: int, reps: int) -> np.ndarray:
    """The fixed initial angles of the training. They depend on the number of layers only.

    Two linear ramps over the layers, pi/8 to pi/2 and pi/2 to pi/8, written
    alternately: ``[a_0, b_0, a_1, b_1, ...]`` with ``a`` rising and ``b``
    falling. QAOAAnsatz orders its parameters ``[beta_0, .., gamma_0, ..]``,
    so the vector is not read layer by layer. With two layers the training
    starts at beta = (pi/8, pi/2), gamma = (pi/2, pi/8).

    An earlier comment said that ``a`` was gamma and ``b`` beta and that the
    order matched the ansatz. It did not. Experiment E25 of the paper measured
    the start that comment described (beta = (pi/2, pi/8), gamma = (pi/8,
    pi/2)) against this one on the instances and seeds of E21 and E23: it was
    better in none of twelve pre-specified comparisons and worse in four. By
    the rule fixed before that run the order below stays. Every QAOA result
    of the paper was measured with it. Neither start was tuned.

    The function name is historical: this is not the INTERP strategy of Zhou
    et al., which initialises depth p + 1 from optimised angles at depth p.
    """
    reps = max(1, int(reps))
    rising = np.linspace(np.pi / 8, np.pi / 2, reps)
    falling = np.linspace(np.pi / 2, np.pi / 8, reps)
    stacked = np.column_stack([rising, falling]).flatten()
    if stacked.size >= num_params:
        return stacked[:num_params]
    # Degenerate fallback (should not happen for standard QAOA).
    return np.resize(stacked, num_params)


def quantum_optimize(prices: pd.DataFrame, risk_aversion: float = 0.5, qc_simulator_mode: bool = True,
                     estimator_shots: int = 100, sampler_shots: int = 200,
                     max_iterations: Optional[int] = None,
                     reps: int = 2, seed: Optional[int] = None) -> dict:
    """
    Run QAOA portfolio optimization.

    Args:
        prices: DataFrame of closing prices (dates x symbols).
        risk_aversion: Risk-aversion coefficient for QUBO construction.
        qc_simulator_mode: If True, use Aer simulator; if False, use real IBM Quantum backend.
        estimator_shots: Shots per expectation-value evaluation (controls IBM Quantum usage).
        sampler_shots: Shots for the final solution sampling (controls IBM Quantum usage).
        max_iterations: COBYLA iteration cap; None = mode-appropriate default
            (15 simulator, 12 real backend) to keep interactive jobs snappy.
        reps: QAOA ansatz depth ``p`` (number of cost/mixer layers).
        seed: Simulator only; makes the run repeatable.

    Returns:
        Dict with keys: solution (binary list, the lowest-cost sampled selection),
        objective_value, selection (what was sampled) and job count stats.
    """
    logger.info("[QAOA] Starting quantum optimization (simulator=%s, maxiter=%s, reps=%d)",
                qc_simulator_mode, max_iterations, reps)
    linear, quadratic, num_assets = _build_qubo(prices, risk_aversion)
    hamiltonian = _build_hamiltonian(linear, quadratic, num_assets)

    reps = max(1, int(reps))
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)
    ansatz.measure_all()

    # A classical selection is computed for comparison and reported with the
    # result. The circuit does not use it. When that step succeeds the
    # training starts from the fixed ramp, otherwise from random angles with
    # a fixed seed (see _classical_warm_start and _interp_init_params; the
    # names are historical).
    warm_start = _classical_warm_start(prices, linear, quadratic, num_assets)
    if warm_start["init_scheme"] == "interp_linear_ramp":
        init_params = _interp_init_params(ansatz.num_parameters, reps)
        logger.info(
            "[QAOA-Init] fixed ramp; classical selection for comparison=%s qubo=%.6f n_selected=%d",
            warm_start["bitstring"], warm_start["qubo_value"], warm_start["n_selected"],
        )
    else:
        rng = np.random.default_rng(42)
        init_params = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)

    cost = _qubo_cost_of_key(linear, quadratic, num_assets)
    if qc_simulator_mode:
        result = _run_simulator(ansatz, hamiltonian, num_assets, init_params, cost,
                                estimator_shots=estimator_shots, sampler_shots=sampler_shots,
                                max_iterations=max_iterations if max_iterations is not None else 15,
                                seed=seed)
    else:
        result = _run_real_backend(ansatz, hamiltonian, num_assets, init_params, cost,
                                   max_iterations=max_iterations if max_iterations is not None else 12)
    # The classical selection and its cost travel with the result so that a
    # reader can compare the two. Nothing downstream replaces the sampled
    # selection by it.
    result["warm_start"] = warm_start
    result["risk_aversion"] = risk_aversion
    if result.get("solution") is not None:
        result["exact_reference"] = referee(linear, quadratic, result["solution"],
                                            sampled_shots=result["selection"]["shots"])
    return result
