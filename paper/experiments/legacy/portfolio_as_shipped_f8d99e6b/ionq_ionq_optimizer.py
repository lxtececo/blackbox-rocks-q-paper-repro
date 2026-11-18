"""
IonQ BF-DCQO Portfolio Optimizer

Implements Bias-Field Digitized Counterdiabatic Quantum Optimization (BF-DCQO)
for large-scale cardinality-constrained portfolio selection on trapped-ion hardware.

Reference: 'Large-scale portfolio optimization on a trapped-ion quantum computer'
           arXiv:2602.23976

Algorithm Pipeline:
1. RMT correlation denoising (Marchenko-Pastur spectral separation)
2. Hardware-aware clustering (Louvain community detection + greedy Qmax splitting)
3. QUBO/Ising formulation with cardinality penalty per cluster
4. BF-DCQO execution with iterative bias-field updates
5. Two-phase post-processing (gradient cardinality repair + first-improvement swap local search)

SDK: qiskit-ionq (IonQProvider) or Qiskit Aer with aria-1 noise model fallback
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

from core.preprocessing import ledoit_wolf_covariance
from core.quantum_math import (
    cardinality_penalty_lambda,
    ising_energy as _qm_ising_energy,
    qubo_pair_to_ising,
)

# ---------------------------------------------------------------------------
# IonQ noise model parameters (aria-1 calibration approximation)
# ---------------------------------------------------------------------------
_ARIA1_GATE_ERROR_1Q = 5e-4    # single-qubit gate error rate
_ARIA1_GATE_ERROR_2Q = 2e-3    # two-qubit gate error rate
_ARIA1_READOUT_ERROR  = 5e-3   # measurement error rate

# ---------------------------------------------------------------------------
# RMT denoising — thin wrappers around the shared core.preprocessing.rmt
# helpers. Kept as module-level aliases so any existing import paths
# (`from .ionq_optimizer import _rmt_denoise_correlation`) keep working.
# ---------------------------------------------------------------------------

from core.preprocessing import (
    marchenko_pastur_threshold as _marchenko_pastur_threshold,
)
from core.preprocessing import denoise_correlation as _shared_denoise_correlation


def _rmt_denoise_correlation(C: np.ndarray, n_obs: int) -> np.ndarray:
    """Denoise a correlation matrix for BF-DCQO clustering.

    Preserves the historical BF-DCQO behaviour: the largest eigenvalue
    (global / market mode) is dropped so Louvain sees only the residual
    sector structure, and noise eigenvalues are zeroed (not redistributed)
    to maximise spectral separation for community detection.
    """
    return _shared_denoise_correlation(
        C,
        n_obs,
        drop_market_mode=True,
        replace_noise_with_mean=False,
        renormalize_diagonal=False,
    )


# ---------------------------------------------------------------------------
# Hardware-aware clustering
# ---------------------------------------------------------------------------

def _louvain_clustering(affinity: np.ndarray) -> List[int]:
    """
    Community detection using modularity maximisation (Louvain).

    Falls back to a simple spectral bisection if `community` is unavailable.

    Args:
        affinity: n × n non-negative symmetric affinity matrix

    Returns:
        community_labels: list of integer community ids, length n
    """
    n = affinity.shape[0]

    try:
        import community as community_louvain  # python-louvain
        import networkx as nx

        # Build weighted graph from affinity
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                w = float(affinity[i, j])
                if w > 1e-8:
                    G.add_edge(i, j, weight=w)

        partition = community_louvain.best_partition(G, weight='weight')
        return [partition[i] for i in range(n)]

    except ImportError:
        logger.warning("[BF-DCQO] python-louvain not installed; using spectral bisection fallback")
        return _spectral_bisection(affinity, n)


def _spectral_bisection(affinity: np.ndarray, n: int) -> List[int]:
    """Fallback: recursive spectral bisection."""
    # Compute graph Laplacian
    D = np.diag(affinity.sum(axis=1))
    L = D - affinity
    _, V = np.linalg.eigh(L)
    # Fiedler vector
    fiedler = V[:, 1]
    return [0 if fiedler[i] < 0 else 1 for i in range(n)]


def _hardware_aware_clusters(
    C_star: np.ndarray,
    n_assets: int,
    qmax: int,
    C_raw: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """
    Algorithm 1 from the paper: Louvain → greedy split to enforce Qmax budget.

    Args:
        C_star:   denoised signal correlation matrix (n × n)
        n_assets: total number of assets
        qmax:     maximum qubits available per cluster run
        C_raw:    undenoised correlation matrix, used when denoising leaves
                  no residual signal (only the market mode was above the
                  Marchenko-Pastur edge)

    Returns:
        clusters: list of lists of asset indices
    """
    # Use absolute values as affinity (co-movement regardless of direction)
    affinity = np.abs(C_star)
    np.fill_diagonal(affinity, 0.0)
    if C_raw is not None and float(affinity.max(initial=0.0)) <= 1e-8:
        # Short histories often leave only the market eigenvalue above the
        # noise edge; dropping it zeroes the whole matrix and Louvain then
        # returns one community per asset.
        logger.info("[BF-DCQO] Denoised correlation has no residual signal; clustering on raw correlation")
        affinity = np.abs(C_raw)
        np.fill_diagonal(affinity, 0.0)

    labels = _louvain_clustering(affinity)
    n_communities = max(labels) + 1

    # Group assets by community
    communities: Dict[int, List[int]] = {k: [] for k in range(n_communities)}
    for asset_idx, comm_id in enumerate(labels):
        communities[comm_id].append(asset_idx)

    # Greedily split any community exceeding Qmax
    clusters: List[List[int]] = []
    for comm in communities.values():
        if len(comm) <= qmax:
            clusters.append(comm)
        else:
            # Split into chunks of ≤ qmax assets
            for start in range(0, len(comm), qmax):
                clusters.append(comm[start:start + qmax])

    logger.info(f"[BF-DCQO] Hardware-aware clustering: {n_assets} assets  {len(clusters)} clusters "
                f"(Qmax={qmax})")
    return clusters


# ---------------------------------------------------------------------------
# QUBO / Ising formulation
# ---------------------------------------------------------------------------

def _build_qubo(
    returns: np.ndarray,
    covariance: np.ndarray,
    cardinality_K: int,
    risk_aversion: float = 1.0,
    penalty_lambda: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the cluster QUBO and return its Ising coefficients.

    QUBO (binary selection x_i, diagonal folded into the linear part):
        q_i  = −μ_i (1 − γ) + γ Σ_ii + λ (1 − 2K)
        Q_ij = γ (Σ_ij + Σ_ji) + 2λ            (upper-triangular pairs)

    Converted to Ising via the shared authority in ``core.quantum_math``
    (F29 — the hand-rolled conversion here was sign-flipped until
    2026-07-29, producing complement portfolios n−K instead of K).

    Returns:
        h: linear (diagonal) Ising coefficients
        J: quadratic (off-diagonal) Ising interaction matrix (upper-tri zeros on diag)
    """
    n = len(returns)
    if penalty_lambda is None:
        penalty_lambda = cardinality_penalty_lambda(covariance, returns, n)

    # QUBO objective — diagonal in `q`, pair coefficients in `Q` (upper-tri).
    q = np.zeros(n)
    Q = np.zeros((n, n))
    for i in range(n):
        q[i] += -returns[i] * (1.0 - risk_aversion)
        q[i] += risk_aversion * covariance[i, i]
        q[i] += penalty_lambda * (1.0 - 2.0 * cardinality_K)
        for j in range(i + 1, n):
            Q[i, j] += risk_aversion * (covariance[i, j] + covariance[j, i])
            Q[i, j] += 2.0 * penalty_lambda  # cross terms of (Σ x_i – K)^2

    h, J, _offset = qubo_pair_to_ising(q, Q)
    return h, J


# ---------------------------------------------------------------------------
# BF-DCQO circuit builder
# ---------------------------------------------------------------------------

def _build_bfdcqo_circuit(
    n_qubits: int,
    h: np.ndarray,
    J: np.ndarray,
    bias_fields: np.ndarray,
    ansatz_depth: int,
    theta_cutoff: float = 0.01
):
    """
    Build Trotterised BF-DCQO ansatz circuit in Qiskit.

    Structure per Trotter step p:
      - RX rotation from driver Hamiltonian (bias-field modified)
      - RZZ interactions from Ising problem Hamiltonian (pruned by theta_cutoff)
      - RZ rotations from local fields h

    Args:
        n_qubits:    number of qubits
        h:           Ising linear coefficients (n,)
        J:           Ising quadratic coefficients (n × n, upper-tri)
        bias_fields: current bias field b_i for each qubit (n,)
        ansatz_depth: number of Trotter steps
        theta_cutoff: minimum |angle| to include a gate (circuit depth pruning)

    Returns:
        QuantumCircuit (Qiskit)
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition initialisation
    qc.h(range(n_qubits))

    # Trotter steps
    dt = 1.0 / ansatz_depth  # effective time step

    for _ in range(ansatz_depth):
        # Driver layer (bias-field modified X rotations)
        for i in range(n_qubits):
            # Bias field shifts the driver: effective angle = dt * (1 + b_i)
            angle_x = 2.0 * dt * (1.0 + bias_fields[i])
            if abs(angle_x) > theta_cutoff:
                qc.rx(angle_x, i)

        # Problem Hamiltonian layer: ZZ interactions
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                angle_zz = 2.0 * dt * J[i, j]
                if abs(angle_zz) > theta_cutoff:
                    qc.rzz(angle_zz, i, j)

        # Local field layer: Z rotations
        for i in range(n_qubits):
            angle_z = 2.0 * dt * h[i]
            if abs(angle_z) > theta_cutoff:
                qc.rz(angle_z, i)

    # Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ---------------------------------------------------------------------------
# Backend initialisation
# ---------------------------------------------------------------------------

def _get_backend(api_key: Optional[str], noise_model: str, use_real_hardware: bool = False):
    """
    Obtain a quantum backend.

    Priority:
    1. qiskit-ionq IonQProvider if api_key is set and qiskit_ionq is installed
    2. Qiskit Aer fake backend with aria-1 calibrated noise model (simulator fallback)

    Returns:
        (backend, backend_name: str)
    """
    # --- Try IonQ cloud/simulator via qiskit-ionq ---
    if api_key:
        try:
            from qiskit_ionq import IonQProvider  # type: ignore
            from qiskit_ionq.exceptions import IonQTranspileLevelWarning  # type: ignore
            provider = IonQProvider(token=api_key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IonQTranspileLevelWarning)
                if use_real_hardware:
                    from core.optimizers.quantum.shared.circuit_runner import ionq_qpu_backend
                    qpu = ionq_qpu_backend()
                    backend = provider.get_backend(qpu)
                    logger.info("[BF-DCQO] Using REAL IonQ QPU backend: %s", qpu)
                    return backend, qpu
                else:
                    backend = provider.get_backend("simulator")
                    backend.set_options(noise_model=noise_model)
                    logger.info(f"[BF-DCQO] Using IonQProvider simulator with noise_model={noise_model}")
                    return backend, f"ionq_simulator({noise_model})"
        except Exception as e:
            logger.warning(f"[BF-DCQO] qiskit-ionq unavailable ({e}); falling back to Aer")

    # --- Aer simulator fallback with approximate aria-1 noise ---
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import (
            NoiseModel, depolarizing_error, ReadoutError
        )

        noise_model_obj = NoiseModel()

        # Single-qubit gate depolarising noise
        err_1q = depolarizing_error(_ARIA1_GATE_ERROR_1Q, 1)
        noise_model_obj.add_all_qubit_quantum_error(err_1q, ['rx', 'rz', 'h'])

        # Two-qubit gate depolarising noise
        err_2q = depolarizing_error(_ARIA1_GATE_ERROR_2Q, 2)
        noise_model_obj.add_all_qubit_quantum_error(err_2q, ['rzz', 'cx'])

        # Readout error
        p0g1 = _ARIA1_READOUT_ERROR
        p1g0 = _ARIA1_READOUT_ERROR
        ro_err = ReadoutError([[1 - p0g1, p0g1], [p1g0, 1 - p1g0]])
        noise_model_obj.add_all_qubit_readout_error(ro_err)

        backend = AerSimulator(noise_model=noise_model_obj)
        logger.info("[BF-DCQO] Using Qiskit Aer with aria-1 approximate noise model")
        return backend, "aer_aria1_noise"

    except ImportError:
        # Last resort: noiseless Aer
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        logger.warning("[BF-DCQO] Noise model unavailable; using noiseless Aer simulator")
        return backend, "aer_noiseless"


# ---------------------------------------------------------------------------
# Cluster-level BF-DCQO
# ---------------------------------------------------------------------------

def _run_bfdcqo_cluster(
    returns_c: np.ndarray,
    covariance_c: np.ndarray,
    cardinality_K_c: int,
    backend,
    n_iterations: int,
    shots: int,
    ansatz_depth: int,
    n_low_energy_samples: int = 10,
    execute_fn=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Execute BF-DCQO for a single cluster (Algorithm 3 from the paper).

    Iteratively:
      1. Build circuit with current bias fields
      2. Run shots on backend
      3. Collect nl lowest-energy samples
      4. Update bias fields: h_j^(b) ← −⟨σ_z_j⟩

    Args:
        returns_c:         expected returns for cluster assets (n_c,)
        covariance_c:      covariance matrix for cluster (n_c × n_c)
        cardinality_K_c:   target cardinality within cluster
        backend:           Qiskit-compatible backend
        n_iterations:      number of BF-DCQO iterations R
        shots:             circuit shots per iteration
        ansatz_depth:      Trotter steps p
        n_low_energy_samples: nl – samples kept per iteration for bias update

    Returns:
        best_bitstring:  binary array (n_c,) – selected assets in cluster
        metrics:         dict with execution metadata
    """
    from qiskit import transpile

    n_c = len(returns_c)
    cardinality_K_c = max(1, min(cardinality_K_c, n_c))

    h, J = _build_qubo(returns_c, covariance_c, cardinality_K_c)
    bias_fields = np.zeros(n_c)

    best_bitstring = np.zeros(n_c, dtype=int)
    best_energy = float('inf')

    # Per-iteration diagnostics (consumed by ionq_adapter._emit_diagnostics)
    best_energy_history: list[float] = []
    mean_bias_history: list[float] = []
    # 7-QAP.25 — per-iter bias-field vector (“params” analogue for BF-DCQO)
    # + per-iter wallclock so the P3 (parameter trajectory) and P9
    # (wallclock-per-iter) diagnostics charts have data.
    bias_history: list[list[float]] = []
    iter_wall_s: list[float] = []
    _last_t = time.perf_counter()

    # F24 Phase B + I11 — IonQ Cloud job IDs collected when running on the
    # IonQ Cloud (both real QPU AND the IonQ Cloud Simulator backend) so
    # the UI can render "View on IonQ Cloud portal" deep-links. Local
    # qiskit-aer / simulated runs (which go through execute_fn) and
    # non-cloud paths are skipped. ``backend.name`` is e.g.
    # ``ionq_qpu.forte-1`` for hardware and ``ionq_simulator`` for the
    # cloud simulator — both expose a real cloud job id on ``job.id``.
    vendor_runs: list[dict] = []
    _backend_label = str(getattr(backend, "name", backend) or "ionq")
    _backend_label_lc = _backend_label.lower()
    _capture_ionq = (execute_fn is None) and (
        "qpu" in _backend_label_lc or "ionq" in _backend_label_lc
    )

    consecutive_failures = 0
    last_error: Optional[BaseException] = None
    successful_iterations = 0

    for iteration in range(n_iterations):
        # Build and transpile circuit
        qc = _build_bfdcqo_circuit(n_c, h, J, bias_fields, ansatz_depth)
        if execute_fn is None:
            try:
                qc_t = transpile(qc, backend, optimization_level=1)
            except Exception:
                qc_t = qc  # Use untranspiled if backend doesn't support transpilation
        else:
            # External executor (e.g. Braket shim) handles its own native
            # decomposition — skip Qiskit transpile to avoid double work.
            qc_t = qc

        # Execute
        try:
            if execute_fn is not None:
                counts = execute_fn(qc_t, shots)
            else:
                job_run = backend.run(qc_t, shots=shots)
                # F24 Phase B + I11 — record IonQ Cloud job id for both
                # real QPU runs and IonQ Cloud Simulator runs. The IonQ
                # portal lists simulator jobs too (with their compiled
                # circuit, queue + completion timestamps); only purely
                # local simulators (qiskit-aer / mock backends) are
                # excluded — those have no cloud-side record to link to.
                if _capture_ionq:
                    try:
                        # I13 — ``qiskit-ionq``'s ``IonQJob`` exposes a
                        # ``job_id()`` *method* (per ``qiskit.providers.JobV1``)
                        # rather than an ``.id`` attribute. Earlier I11 code
                        # only checked ``.id`` and always got ``None``, so
                        # deep-links were silently dropped for IonQ Cloud
                        # Simulator runs. Probe both, and call if callable.
                        jid = None
                        for _attr in ("job_id", "id"):
                            _val = getattr(job_run, _attr, None)
                            if _val is None:
                                continue
                            if callable(_val):
                                try:
                                    _val = _val()
                                except Exception:  # noqa: BLE001
                                    _val = None
                            if _val:
                                jid = _val
                                break
                        if jid:
                            from datetime import datetime, timezone
                            vendor_runs.append({
                                "vendor": "ionq",
                                "vendor_label": "IonQ Cloud",
                                "job_id": str(jid),
                                "stage": f"iteration #{iteration + 1}",
                                "backend": _backend_label,
                                "submitted_at": datetime.now(timezone.utc).isoformat(),
                            })
                    except Exception:  # noqa: BLE001  # nosec B110 - best-effort cleanup; exception intentionally swallowed
                        pass
                counts = job_run.result().get_counts()
            consecutive_failures = 0
            successful_iterations += 1
        except Exception as e:
            consecutive_failures += 1
            last_error = e
            err_repr = repr(e)
            # Fail-fast on auth / entitlement / quota errors — no point retrying.
            # IonQ returns 401 / 403 in the body; AWS Braket raises ClientError.
            fatal_markers = (
                "403", "401", "Forbidden", "Unauthorized",
                "not enabled", "not authorized", "AccessDenied",
                "InvalidSignature", "ExpiredToken",
            )
            is_fatal = any(m in err_repr for m in fatal_markers)
            if is_fatal or consecutive_failures >= 3:
                logger.error(
                    f"[BF-DCQO] Aborting cluster after {consecutive_failures} "
                    f"consecutive backend failure(s) (fatal={is_fatal}): {e}"
                )
                raise RuntimeError(
                    f"BF-DCQO backend execution failed: {e}"
                ) from e
            logger.warning(f"[BF-DCQO] Backend run failed at iteration {iteration}: {e}")
            # Soft-fallback only for transient errors (kept ≤2 in a row)
            counts = {_random_binary_string(n_c, cardinality_K_c): shots}

        # Evaluate energies of sampled bitstrings
        energies = []
        for bitstring, count in counts.items():
            # Qiskit returns bits in reverse order (qubit 0 = rightmost)
            bits = np.array([int(b) for b in reversed(bitstring)], dtype=int)
            if len(bits) != n_c:
                continue
            energy = _ising_energy(bits, h, J)
            energies.append((energy, bits, count))

        if not energies:
            continue

        # Sort by energy and keep nl lowest
        energies.sort(key=lambda x: x[0])
        low_energy_samples = energies[:n_low_energy_samples]

        # Update best solution
        if low_energy_samples[0][0] < best_energy:
            best_energy = low_energy_samples[0][0]
            best_bitstring = low_energy_samples[0][1].copy()

        # Update bias fields: h_j^(b) ← −⟨σ_z_j⟩
        # σ_z_j = 1 – 2 x_j  (x_j ∈ {0,1})
        total_count = sum(s[2] for s in low_energy_samples)
        for j in range(n_c):
            mean_z = sum(
                (1.0 - 2.0 * s[1][j]) * s[2]
                for s in low_energy_samples
            ) / max(total_count, 1)
            bias_fields[j] = -mean_z

        # Snapshot per-iteration diagnostics
        best_energy_history.append(float(best_energy))
        mean_bias_history.append(float(np.mean(np.abs(bias_fields))))
        bias_history.append([float(x) for x in bias_fields])
        _now = time.perf_counter()
        iter_wall_s.append(round(_now - _last_t, 6))
        _last_t = _now

        logger.debug(f"[BF-DCQO] Iteration {iteration + 1}/{n_iterations}: "
                     f"best_energy={best_energy:.6f}")

    if successful_iterations == 0:
        # Every iteration soft-failed (none crossed the consecutive=3 fatal trip,
        # e.g. n_iterations <= 2). Refuse to return a fabricated random result.
        raise RuntimeError(
            f"BF-DCQO backend execution failed: 0/{n_iterations} iterations "
            f"produced real samples. Last error: {last_error!r}"
        )

    metrics = {
        'n_iterations': n_iterations,
        'best_energy': float(best_energy),
        'n_assets_cluster': n_c,
        'cardinality_k_cluster': cardinality_K_c,
        'best_energy_history': best_energy_history,
        'mean_bias_history': mean_bias_history,
        # 7-QAP.25 — expose for P3/P9 diagnostics charts.
        'bias_history': bias_history,
        'iter_wall_s': iter_wall_s,
        # F24 Phase B — per-cluster IonQ Cloud job ids; the optimizer
        # aggregates these across clusters into the top-level vendor_runs.
        'vendor_runs': vendor_runs,
    }
    return best_bitstring, metrics


def _ising_energy(bits: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    """Evaluate Ising energy E = Σ_i h_i z_i + Σ_{i<j} J_ij z_i z_j.

    Thin wrapper over ``core.quantum_math.ising_energy`` (kept so existing
    imports keep working). J is upper-triangular — z@J@z counts each pair
    once, no /2.
    """
    return _qm_ising_energy(bits, h, J)


def _random_binary_string(n: int, k: int) -> str:
    """Generate a random binary string with exactly k ones."""
    bits = np.zeros(n, dtype=int)
    ones = np.random.choice(n, size=min(k, n), replace=False)
    bits[ones] = 1
    return ''.join(str(b) for b in reversed(bits))  # Qiskit convention


# ---------------------------------------------------------------------------
# Two-phase post-processing (Algorithm 4)
# ---------------------------------------------------------------------------

def _postprocess_allocation(
    allocation_bits: np.ndarray,
    returns: np.ndarray,
    covariance: np.ndarray,
    target_k: int,
    telemetry: dict | None = None,
) -> np.ndarray:
    """
    Two-phase post-processing (delegates to ``core.postprocessing.refinement_engine``).

    Kept as a thin wrapper so existing call sites in this module remain stable.
    Phase 1 – gradient-based cardinality repair.
    Phase 2 – first-improvement swap local search on Ising energy.

    When ``telemetry`` is provided, the refinement engine populates per-phase
    diagnostics in-place (used by Job History step-detail rendering).
    """
    from core.postprocessing.refinement_engine import refine_binary_solution

    return refine_binary_solution(
        allocation_bits,
        returns,
        covariance,
        cardinality_K=target_k,
        risk_aversion=1.0,
        max_rounds=10,
        telemetry=telemetry,
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_performance(
    weights: Dict[str, float],
    prices: pd.DataFrame
) -> Dict[str, Any]:
    """Compute annualised Sharpe ratio and expected return from equal weights."""
    if not weights or prices.empty:
        return {'sharpe_ratio': 0.0, 'expected_return': 0.0, 'annual_volatility': 0.0}

    selected = [s for s, w in weights.items() if w > 0]
    if not selected:
        return {'sharpe_ratio': 0.0, 'expected_return': 0.0, 'annual_volatility': 0.0}

    cols = [s for s in selected if s in prices.columns]
    if not cols:
        return {'sharpe_ratio': 0.0, 'expected_return': 0.0, 'annual_volatility': 0.0}

    sub = prices[cols].pct_change().dropna()
    w_arr = np.array([weights[s] for s in cols])
    w_arr = w_arr / (w_arr.sum() + 1e-12)

    mean_daily = sub.mean().values
    cov_daily = sub.cov().values

    exp_ret = float(np.dot(w_arr, mean_daily) * 252)
    vol = float(np.sqrt(w_arr @ cov_daily @ w_arr) * np.sqrt(252))
    sharpe = exp_ret / vol if vol > 1e-8 else 0.0

    return {
        'sharpe_ratio': round(sharpe, 4),
        'expected_return': round(exp_ret, 4),
        'annual_volatility': round(vol, 4)
    }


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class IonQBFDCQOOptimizer:
    """
    BF-DCQO portfolio optimizer for IonQ trapped-ion quantum hardware.

    Usage:
        optimizer = IonQBFDCQOOptimizer(api_key=os.getenv('IONQ_API_KEY'))
        result = optimizer.optimize(prices_df, cardinality_K=5)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        noise_model: str = 'aria-1',
        risk_aversion: float = 0.5,
        use_real_hardware: bool = False,
        backend=None,
        backend_name: Optional[str] = None,
        execute_fn=None,
    ):
        """
        Args:
            api_key:           IonQ API key (used only when ``backend`` is None)
            noise_model:       IonQ noise model name (used only when ``backend`` is None)
            risk_aversion:     QUBO risk-aversion coefficient
            use_real_hardware: route to IonQ QPU when ``backend`` is None
            backend:           optional externally-supplied Qiskit ``BackendV2``-compatible
                               backend. When provided, ``_init_backend`` skips the IonQ /
                               Aer factory entirely. Enables BF-DCQO on non-IonQ
                               platforms (IBM, Google, Rigetti, IQM) at the simulator
                               level via the platform adapter layer.
            backend_name:      optional human-readable label for logs / quantum_metrics.
                               Falls back to ``backend.name`` (callable or attribute) or
                               ``"external_backend"``.
        """
        self.api_key = api_key
        self.noise_model = noise_model
        self.risk_aversion = risk_aversion
        self.use_real_hardware = use_real_hardware
        self._backend = backend
        self._execute_fn = execute_fn
        if backend is not None and backend_name is None:
            try:
                name_attr = getattr(backend, "name", None)
                backend_name = name_attr() if callable(name_attr) else (name_attr or "external_backend")
            except Exception:
                backend_name = "external_backend"
        self._backend_name = backend_name

    def _init_backend(self):
        if self._backend is None and self._execute_fn is None:
            self._backend, self._backend_name = _get_backend(self.api_key, self.noise_model, self.use_real_hardware)

    def optimize(
        self,
        prices: pd.DataFrame,
        cardinality_K: Optional[int] = None,
        qmax: int = 8,
        n_iterations: int = 3,
        shots: int = 8192,
        ansatz_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Run BF-DCQO portfolio optimisation.

        Args:
            prices:        DataFrame of daily closing prices (rows=dates, cols=symbols)
            cardinality_K: target number of assets to select; defaults to n_assets // 3
            qmax:          maximum qubits per cluster (hardware qubit budget)
            n_iterations:  BF-DCQO iterations R
            shots:         circuit measurements per iteration
            ansatz_depth:  Trotter steps p

        Returns:
            dict with keys: success, allocation, performance, quantum_metrics
        """
        try:
            self._init_backend()
            symbols = list(prices.columns)
            n_assets = len(symbols)

            if cardinality_K is None:
                cardinality_K = max(1, n_assets // 3)
            cardinality_K = max(1, min(cardinality_K, n_assets))

            logger.info(f"[BF-DCQO] n_assets={n_assets}, K={cardinality_K}, "
                        f"qmax={qmax}, R={n_iterations}, shots={shots}, "
                        f"backend={self._backend_name}")

            # Compute returns and Ledoit-Wolf-shrunk covariance
            log_returns = np.log(prices / prices.shift(1)).dropna()
            mu = log_returns.mean().values * 252          # annualised
            Sigma, _cov_telemetry = ledoit_wolf_covariance(log_returns, annualization_factor=252.0)
            logger.info(
                "[BF-DCQO] covariance estimator=%s shrinkage=%s n_obs=%s n_assets=%s",
                _cov_telemetry.get("estimator"), _cov_telemetry.get("shrinkage"),
                _cov_telemetry.get("n_observations"), _cov_telemetry.get("n_assets"),
            )

            # Normalised correlation matrix for clustering
            sigma_diag = np.sqrt(np.diag(Sigma))
            sigma_diag = np.where(sigma_diag < 1e-10, 1.0, sigma_diag)
            C = Sigma / np.outer(sigma_diag, sigma_diag)

            # Step 1: RMT denoising
            n_obs = len(log_returns)
            C_star = _rmt_denoise_correlation(C, n_obs)

            # Step 2: Hardware-aware clustering
            clusters = _hardware_aware_clusters(C_star, n_assets, qmax, C_raw=C)

            # Step 3–4: BF-DCQO per cluster
            global_bits = np.zeros(n_assets, dtype=int)
            cluster_metrics = []
            total_cluster_K = 0

            for c_idx, cluster in enumerate(clusters):
                c_arr = np.array(cluster)
                mu_c = mu[c_arr]
                Sigma_c = Sigma[np.ix_(c_arr, c_arr)]

                # Proportional cardinality per cluster
                K_c = max(1, round(cardinality_K * len(cluster) / n_assets))
                K_c = min(K_c, len(cluster))
                total_cluster_K += K_c

                logger.info(f"[BF-DCQO] Cluster {c_idx + 1}/{len(clusters)}: "
                            f"assets={c_arr.tolist()}, K_c={K_c}")

                bits_c, c_metrics = _run_bfdcqo_cluster(
                    returns_c=mu_c,
                    covariance_c=Sigma_c,
                    cardinality_K_c=K_c,
                    backend=self._backend,
                    n_iterations=n_iterations,
                    shots=shots,
                    ansatz_depth=ansatz_depth,
                    execute_fn=self._execute_fn,
                )
                cluster_metrics.append(c_metrics)

                # Place cluster result back into global vector
                for local_i, global_i in enumerate(c_arr):
                    global_bits[global_i] = bits_c[local_i]

            # Step 5: Two-phase post-processing on the full portfolio
            refinement_telemetry: dict = {}
            global_bits = _postprocess_allocation(
                global_bits, mu, Sigma, target_k=cardinality_K,
                telemetry=refinement_telemetry,
            )

            # Convert binary selection to equal-weight allocation
            selected_indices = np.where(global_bits == 1)[0]
            if len(selected_indices) == 0:
                # Fallback: pick top-K by expected return
                selected_indices = np.argsort(mu)[-cardinality_K:]

            weight = 1.0 / len(selected_indices)
            allocation = {symbols[i]: 0.0 for i in range(n_assets)}
            for i in selected_indices:
                allocation[symbols[i]] = round(weight, 6)

            # Performance metrics
            performance = _compute_performance(allocation, prices)

            cardinality_satisfied = len(selected_indices) == cardinality_K

            return {
                'success': True,
                'allocation': allocation,
                'performance': performance,
                'quantum_metrics': {
                    'backend': self._backend_name,
                    'noise_model': self.noise_model,
                    'n_iterations': n_iterations,
                    'shots': shots,
                    'ansatz_depth': ansatz_depth,
                    'qmax': qmax,
                    'n_clusters': len(clusters),
                    'cardinality_k': cardinality_K,
                    'cardinality_satisfied': cardinality_satisfied,
                    'selected_assets': [symbols[i] for i in selected_indices],
                    'cluster_metrics': cluster_metrics,
                    'refinement': refinement_telemetry,
                    # F24 Phase B — flatten per-cluster IonQ Cloud job ids
                    # into a single top-level list so the executor /
                    # adapter can attach "View on IonQ Cloud portal"
                    # deep-link artifacts. Empty when running against the
                    # IonQ simulator (sim runs collect no real job ids).
                    #
                    # 2026-07-09 UX fix — `_run_bfdcqo_cluster` labels its
                    # per-iteration vendor runs as `iteration #1..R`, which
                    # restarts at 1 for every cluster and produces a
                    # repeating "iteration #1, iteration #2, iteration #1,
                    # iteration #2..." Stage column in the UI. Re-tag with
                    # the cluster ordinal so the column reads
                    # "cluster 1 / iter #1, cluster 1 / iter #2, cluster
                    # 2 / iter #1, ..." across the whole run.
                    'vendor_runs': [
                        {**vr, 'stage': f"cluster {c_idx + 1} / {vr.get('stage', 'iter')}"}
                        for c_idx, cm in enumerate(cluster_metrics)
                        for vr in (cm.get('vendor_runs') or [])
                    ],
                }
            }

        except Exception as e:
            logger.error(f"[BF-DCQO] Optimization failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'allocation': None,
                'performance': {},
                'quantum_metrics': {}
            }
