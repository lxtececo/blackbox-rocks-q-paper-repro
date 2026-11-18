"""
IBM CVaR-VQE Platform Adapter
=============================

Bridges the standalone :class:`CVaRVQEOptimizer` (which works on
``(ansatz, hamiltonian)``) and the executor contract used by
``execute_quantum_cvar_vqe_optimization``. Until this adapter existed,
that executor delegated to :class:`PCEPortfolioOptimizer` because there
was no portfolio-level CVaR-VQE entry-point — see the historical comment
at ``api/executors.py:execute_quantum_cvar_vqe_optimization``.

What this adapter does
----------------------

1. Compute expected returns + sample covariance from a prices DataFrame.
2. Optionally apply Marchenko-Pastur RMT denoising on the *sample*
   covariance (method key ``"cvar_vqe"``; opt-in by default).
3. Build a Markowitz QUBO over N binary asset-selection variables.
4. Convert QUBO -> Ising Hamiltonian as a :class:`SparsePauliOp` via the
   standard substitution ``x = (1 - z) / 2``.
5. Build a :class:`RealAmplitudes` ansatz over N qubits.
6. Run :meth:`CVaRVQEOptimizer.enhanced_vqe_optimization`.
7. Sample the trained ansatz, pick the lowest-energy / most-probable
   feasible bitstring, equal-weight the selected assets (with optional
   cardinality cap) and compute portfolio performance metrics on the
   denoised covariance.
8. Return a result dict shaped exactly like the one
   ``PCEPortfolioOptimizer.optimize`` returns, so the executor can stay
   agnostic to which backend produced it.

Hardware path
-------------

When ``use_real_hardware=True`` (or ``CVaRConfig.simulator_mode=False``),
:class:`CVaRVQEOptimizer` resolves an IBM Runtime backend via
``QiskitRuntimeService.least_busy`` (BYOK token first, ``IBM_QUANTUM_TOKEN``
env-var fallback) and runs both the EstimatorV2 energy-evaluation loop
*and* the final SamplerV2 bitstring sampling against that QPU. The
returned dict's ``vendor_runs`` list carries the IBM Runtime job IDs
so the Job History view can render "View on IBM Quantum portal"
deep-links (F24 Phase A.2). Falls back to :class:`AerSimulator` cleanly
if no token is available.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize as _scipy_minimize

from core.optimizers.quantum.ibm.cvar_vqe_optimizer import (
    CVaRConfig,
    CVaRVQEOptimizer,
)
from core.preprocessing import maybe_denoise_covariance, marchenko_pastur_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QUBO construction (sample-covariance Markowitz, NO Ledoit-Wolf shrinkage so
# RMT denoising remains theoretically clean).
# ---------------------------------------------------------------------------

def _annualised_returns_and_cov(
    prices: pd.DataFrame,
    *,
    trading_days: int = 252,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
    """Return (mu_annual, Sigma_annual, rmt_applied, rmt_info).

    ``rmt_info`` always carries the numbers the UI needs to render a
    "RMT denoising" preprocessing step (N, T, q, λ⁺), even when no
    cleaning was actually applied (in which case ``applied=False``).
    Uses simple returns and the per-method opt-in matrix in
    ``core.preprocessing.rmt``. Returned arrays are dense numpy arrays
    in the order of ``prices.columns``.
    """
    if prices.shape[0] < 2:
        raise ValueError(
            f"CVaR-VQE needs at least 2 price observations, got {prices.shape[0]}"
        )
    returns = prices.pct_change().dropna()
    mu_daily = returns.mean().to_numpy()
    cov_daily = np.cov(returns.to_numpy(), rowvar=False, ddof=1)
    # Ensure 2-D (np.cov returns 0-D for N=1)
    cov_daily = np.atleast_2d(cov_daily)

    n_assets = int(prices.shape[1])
    n_obs = max(returns.shape[0] - 1, 1)
    cleaned, applied = maybe_denoise_covariance(
        cov_daily, n_obs=n_obs, method="cvar_vqe",
    )
    cov_daily = cleaned

    rmt_info: Dict[str, Any] = {
        "method": "cvar_vqe",
        "applied": bool(applied),
        "n_assets": n_assets,
        "n_observations": int(n_obs),
        "q_ratio": float(n_assets) / float(n_obs) if n_obs else float("inf"),
        "lambda_plus": float(marchenko_pastur_threshold(n_obs, n_assets)),
        "scheme": "marchenko_pastur_clip",
    }

    mu_annual = mu_daily * trading_days
    sigma_annual = cov_daily * trading_days
    return mu_annual, sigma_annual, applied, rmt_info


def _build_markowitz_qubo(
    mu_annual: np.ndarray,
    sigma_annual: np.ndarray,
    *,
    risk_aversion: float,
    cardinality: Optional[int],
    cardinality_penalty: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build a Markowitz binary-selection QUBO.

    Cost(x) = - mu^T x + lambda * x^T Sigma x
              + penalty * (1^T x - k)^2     [if cardinality is set]

    Returns ``(h, J, const)`` where the QUBO is expanded as
    ``Cost(x) = const + sum_i h_i x_i + sum_{i<j} J_ij x_i x_j``.

    The diagonal of the quadratic part is absorbed into ``h`` because
    ``x_i^2 = x_i`` for binary variables.
    """
    n = mu_annual.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    const = 0.0

    # Expected return contribution (we maximise returns -> minimise -mu^T x)
    h -= mu_annual

    # Risk contribution: lambda * x^T Sigma x
    # Symmetrise Sigma (defensive) then split diag -> h, off-diag -> J
    Sigma = 0.5 * (sigma_annual + sigma_annual.T)
    for i in range(n):
        h[i] += risk_aversion * Sigma[i, i]
        for j in range(i + 1, n):
            # Counted once with weight 2 (from x_i x_j + x_j x_i)
            J[i, j] += 2.0 * risk_aversion * Sigma[i, j]

    # Cardinality penalty: penalty * (sum x - k)^2 expands to
    # penalty * (sum x_i + 2 sum_{i<j} x_i x_j - 2k sum x_i + k^2)
    if cardinality is not None and cardinality > 0:
        k = float(cardinality)
        for i in range(n):
            h[i] += cardinality_penalty * (1.0 - 2.0 * k)
        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] += 2.0 * cardinality_penalty
        const += cardinality_penalty * (k ** 2)

    return h, J, const


def _qubo_to_ising_sparse_pauli(
    h: np.ndarray, J: np.ndarray
) -> Tuple[SparsePauliOp, float]:
    """Map a QUBO ``h, J`` (with diagonal already in h) to a Pauli-Z Ising
    Hamiltonian via the substitution ``x_i = (1 - Z_i) / 2``.

    Returns ``(H, offset)`` such that minimising ``<H>`` over states
    minimises the QUBO cost up to the additive ``offset``.
    """
    n = h.shape[0]
    # x_i = (1 - z_i)/2  ->  x_i x_j = (1 - z_i)(1 - z_j)/4
    #                       = 1/4 - z_i/4 - z_j/4 + z_i z_j / 4
    # Aggregate Z and ZZ coefficients.
    offset = 0.0
    z_coeffs = np.zeros(n)
    zz_coeffs: Dict[Tuple[int, int], float] = {}

    # Linear h_i x_i = h_i * (1 - z_i)/2 = h_i/2 - h_i z_i / 2
    for i in range(n):
        offset += 0.5 * h[i]
        z_coeffs[i] -= 0.5 * h[i]

    # Quadratic J_ij x_i x_j  (only stored for i < j)
    for i in range(n):
        for j in range(i + 1, n):
            jij = J[i, j]
            if jij == 0.0:
                continue
            offset += 0.25 * jij
            z_coeffs[i] -= 0.25 * jij
            z_coeffs[j] -= 0.25 * jij
            zz_coeffs[(i, j)] = zz_coeffs.get((i, j), 0.0) + 0.25 * jij

    # Build Pauli strings — qiskit convention: leftmost char acts on qubit n-1.
    paulis: List[str] = []
    coeffs: List[float] = []

    for i in range(n):
        if z_coeffs[i] == 0.0:
            continue
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        paulis.append("".join(s))
        coeffs.append(float(z_coeffs[i]))

    for (i, j), v in zz_coeffs.items():
        if v == 0.0:
            continue
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        s[n - 1 - j] = "Z"
        paulis.append("".join(s))
        coeffs.append(float(v))

    if not paulis:
        # Degenerate: build an explicit zero operator on n qubits.
        paulis = ["I" * n]
        coeffs = [0.0]

    H = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
    return H, offset


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class IBMCVaRVQEAdapter:
    """Portfolio-level CVaR-VQE entry-point for the IBM platform."""

    def __init__(
        self,
        cvar_config: CVaRConfig,
        *,
        risk_aversion: float = 1.0,
        ansatz_depth: int = 3,
        cardinality: Optional[int] = None,
        cardinality_penalty: Optional[float] = None,
        use_real_hardware: bool = False,
        sampling_shots: int = 2048,
        enable_classical_reweight: bool = True,
    ):
        self.cvar_config = cvar_config
        self.use_real_hardware = bool(use_real_hardware)
        # F24 Phase A.2 — push real-HW flag through to the inner optimizer
        # so EstimatorV2.run() submits to IBM Runtime and exposes job_id()
        # for vendor-portal deep-links.
        self.cvar_optimizer = CVaRVQEOptimizer(
            cvar_config, use_real_hardware=self.use_real_hardware
        )
        self.risk_aversion = float(risk_aversion)
        self.ansatz_depth = int(ansatz_depth)
        self.cardinality = cardinality if cardinality and cardinality > 0 else None
        # Default penalty scales with risk_aversion to keep terms comparable.
        self.cardinality_penalty = float(
            cardinality_penalty if cardinality_penalty is not None
            else max(1.0, 5.0 * self.risk_aversion)
        )
        self.sampling_shots = int(sampling_shots)
        self.enable_classical_reweight = bool(enable_classical_reweight)

        # Vendor-side job IDs captured during ``_sample_best_bitstring``
        # when running on real IBM hardware. Merged with the optimizer's
        # vendor_runs in ``optimize`` so the executor can surface them.
        self._sampling_vendor_runs: list[dict] = []

        # Whether the inner optimizer actually got promoted off AerSimulator
        # (may differ from self.use_real_hardware if BYOK token is missing).
        self._real_active = bool(getattr(self.cvar_optimizer, "_is_real_ibm", False))
        if self.use_real_hardware and not self._real_active:
            logger.warning(
                "[IBM-CVAR-VQE] use_real_hardware=True requested but no IBM "
                "Runtime backend could be acquired (token missing or "
                "service unavailable). Falling back to AerSimulator."
            )

    # ------------------------------------------------------------------

    def optimize(self, prices: pd.DataFrame) -> Dict[str, Any]:
        """Run a portfolio-level CVaR-VQE optimisation.

        Returns a result dict shaped like ``PCEPortfolioOptimizer.optimize``
        so callers (in particular ``execute_quantum_cvar_vqe_optimization``)
        do not need to know which backend produced the result.
        """
        symbols = list(prices.columns)
        n_assets = len(symbols)
        if n_assets == 0:
            return {"success": False, "error": "No assets in prices",
                    "method": "IBM_CVAR_VQE", "timestamp": datetime.now().isoformat()}

        # Soft cap on qubits to keep simulator tractable.
        if n_assets > 20:
            logger.warning(
                "[IBM-CVAR-VQE] %d assets requested — CVaR-VQE allocates 1 "
                "qubit per asset; simulator runtime grows exponentially. "
                "Consider PCE for large portfolios.",
                n_assets,
            )

        start_time = time.time()
        logger.info(
            "[IBM-CVAR-VQE] Starting portfolio optimisation: %d assets, "
            "depth=%d, alpha=%.3f, adaptive_alpha=%s",
            n_assets, self.ansatz_depth, self.cvar_config.alpha,
            self.cvar_config.adaptive_alpha,
        )

        try:
            mu_annual, sigma_annual, rmt_applied, rmt_info = _annualised_returns_and_cov(prices)

            h, J, const = _build_markowitz_qubo(
                mu_annual, sigma_annual,
                risk_aversion=self.risk_aversion,
                cardinality=self.cardinality,
                cardinality_penalty=self.cardinality_penalty,
            )

            H, ising_offset = _qubo_to_ising_sparse_pauli(h, J)
            # Total additive constant when comparing <H> to QUBO cost
            energy_offset = const + ising_offset

            # Build ansatz: 1 qubit per asset. NB: do NOT add measurements
            # here — CVaRVQEOptimizer uses EstimatorV2 internally which
            # rejects measured circuits.
            ansatz = RealAmplitudes(num_qubits=n_assets, reps=self.ansatz_depth)

            logger.info(
                "[IBM-CVAR-VQE] Built Ising Hamiltonian: %d Pauli terms, "
                "%d qubits, ansatz_params=%d",
                len(H), n_assets, ansatz.num_parameters,
            )

            # Run the CVaR-VQE inner loop. Gradient-free COBYLA is
            # used: SLSQP would do ~num_params finite-diff evaluations per
            # gradient step which makes 20+-qubit portfolios take hours on
            # the Aer simulator. COBYLA converges in O(num_params)
            # total evaluations which is interactive-friendly for
            # portfolios with up to ~40 assets.
            vqe_result = self.cvar_optimizer.enhanced_vqe_optimization(
                ansatz=ansatz, hamiltonian=H,
                classical_optimizer='COBYLA',
            )

            # Sample the trained ansatz and pick best feasible bitstring
            best_params = np.asarray(vqe_result.get("optimal_parameters", []), dtype=float)
            selection = self._sample_best_bitstring(ansatz, best_params, h, J)

            portfolio_weights = self._selection_to_weights(symbols, selection)
            reweight_info: Dict[str, Any] = {
                "enabled": self.enable_classical_reweight,
                "applied": False,
                "scheme": "classical_mean_variance_qp",
            }
            if self.enable_classical_reweight:
                reweighted, info = self._classical_reweight_selected(
                    symbols, portfolio_weights, mu_annual, sigma_annual,
                )
                reweight_info.update(info)
                if reweighted is not None:
                    portfolio_weights = reweighted
            performance = self._performance_metrics(
                symbols, portfolio_weights, mu_annual, sigma_annual,
            )

            # Compute realised QUBO cost for the chosen selection (for reporting)
            x = np.array([1.0 if portfolio_weights[s] > 0 else 0.0 for s in symbols])
            qubo_cost = float(x @ h + x @ np.triu(J, 1) @ x + const)

            convergence = {
                "total_evaluations": int(vqe_result.get("optimization_iterations", 0)),
                "best_cost_achieved": float(vqe_result.get("cvar_energy", 0.0) + energy_offset),
                "convergence_efficiency": 1.0 if vqe_result.get("success") else 0.0,
                "optimization_stability": "stable" if vqe_result.get("success") else "unstable",
            }
            cvar_metrics = {
                "var_estimate": float(vqe_result.get("final_evaluation_metrics", {})
                                        .get("cvar_metrics", {}).get("var_value", 0.0)),
                "final_alpha": float(
                    (vqe_result.get("alpha_history") or [self.cvar_config.alpha])[-1]
                ),
                "confidence_interval": vqe_result.get("final_evaluation_metrics", {})
                                                   .get("cvar_metrics", {})
                                                   .get("cvar_confidence_interval", [0.0, 0.0]),
            }
            resource_usage = {
                "qubits_used": n_assets,
                "ansatz_depth": self.ansatz_depth,
                "parameter_count": ansatz.num_parameters,
                "total_measurements": int(vqe_result.get("resource_usage", {})
                                           .get("total_measurements", 0)),
            }
            optimization_history = [
                {
                    "round": idx,
                    "i": idx,
                    "cost": float(v),
                    "params": (vqe_result.get("params_history") or [None] * (idx + 1))[idx]
                              if idx < len(vqe_result.get("params_history") or []) else None,
                    "wall_s": (vqe_result.get("wall_history") or [None] * (idx + 1))[idx]
                              if idx < len(vqe_result.get("wall_history") or []) else None,
                }
                for idx, v in enumerate(vqe_result.get("cvar_history", []))
            ]

            total_time = time.time() - start_time

            logger.info(
                "[IBM-CVAR-VQE] Done in %.2fs: held=%d, qubo_cost=%.6f, "
                "Sharpe=%.3f",
                total_time,
                performance["num_assets_held"],
                qubo_cost,
                performance["sharpe_ratio"],
            )

            return {
                "success": True,
                "portfolio_weights": portfolio_weights,
                "performance_metrics": performance,
                "convergence_metrics": convergence,
                "cvar_metrics": cvar_metrics,
                "resource_usage": resource_usage,
                "optimization_history": optimization_history,
                # F24 Phase A.2 — union of IBM Runtime job IDs collected
                # inside the optimizer (per-iter Estimator.run()) and during
                # final bitstring sampling. Empty when AerSimulator is used.
                "vendor_runs": (
                    list(getattr(self.cvar_optimizer, "vendor_runs", []) or [])
                    + list(self._sampling_vendor_runs)
                ),
                "metadata": {
                    "num_assets": n_assets,
                    "rmt_applied": rmt_applied,
                    "rmt_info": rmt_info,
                    "risk_aversion": self.risk_aversion,
                    "cardinality": self.cardinality,
                    "energy_offset": energy_offset,
                    "qubo_cost": qubo_cost,
                    "classical_reweight": reweight_info,
                },
                # Top-level mirror so executors / UI can render a dedicated
                # \"RMT denoising\" preprocessing step without digging into
                # metadata. ``rmt_info`` is always populated; ``applied``
                # reflects whether any eigenvalue was actually clipped.
                "rmt_denoising": rmt_info,
                "randomness_info": {
                    "classical_optimizer": "COBYLA (via CVaRVQEOptimizer)",
                    "quantum_sampling_shots": self.cvar_config.shots_per_circuit,
                    "sources_of_randomness": [
                        "Initial parameter sampling",
                        "Quantum measurement outcomes (Aer)",
                        "Adaptive-alpha trajectory",
                    ],
                },
                "total_optimization_time": total_time,
                "method": "IBM_CVAR_VQE",
                "configuration": {
                    "alpha": self.cvar_config.alpha,
                    "adaptive_alpha": self.cvar_config.adaptive_alpha,
                    "ansatz_depth": self.ansatz_depth,
                    "risk_aversion": self.risk_aversion,
                    "cardinality": self.cardinality,
                    "measurement_budget": self.cvar_config.measurement_budget,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as exc:
            logger.exception("[IBM-CVAR-VQE] Optimisation failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "method": "IBM_CVAR_VQE",
                "timestamp": datetime.now().isoformat(),
            }

    # ------------------------------------------------------------------

    def _record_sampling_job(self, job, *, stage: str) -> None:
        """F24 Phase A.2 — record IBM Runtime sampler job_id (real-HW only).

        Errors are swallowed so a SDK quirk never breaks the optimisation.
        """
        if not self._real_active:
            return
        try:
            jid = job.job_id()
        except Exception:  # noqa: BLE001
            return
        if not jid:
            return
        try:
            from datetime import timezone
            self._sampling_vendor_runs.append({
                "vendor": "ibm",
                "vendor_label": "IBM Quantum",
                "job_id": jid,
                "stage": stage,
                "backend": getattr(self.cvar_optimizer, "_backend_name", "ibm"),
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001  # nosec B110 - best-effort cleanup; exception intentionally swallowed
            pass

    def _sample_best_bitstring(
        self,
        ansatz,
        params: np.ndarray,
        h: np.ndarray,
        J: np.ndarray,
    ) -> np.ndarray:
        """Sample the trained ansatz and return the bitstring (1 per asset)
        with the lowest classical QUBO cost. Falls back to all-ones if
        sampling fails. Indexing: result[i] corresponds to asset i (i.e.
        prices.columns[i]); qubit-order conversion is handled here.
        """
        n = ansatz.num_qubits
        try:
            if params.size != ansatz.num_parameters:
                raise ValueError(
                    f"Param shape mismatch: got {params.size}, need {ansatz.num_parameters}"
                )
            bound = ansatz.assign_parameters(params)
            # Add measurements for sampling (CVaRVQEOptimizer keeps the
            # ansatz unmeasured for the EstimatorV2 path).
            bound = bound.copy()
            bound.measure_all()

            # F24 Phase A.2 — when the inner optimizer promoted to a real
            # IBM backend, sample via SamplerV2(mode=<real-backend>) so the
            # final bitstring distribution comes from the QPU and the job
            # ID is surfaced as a vendor-portal deep-link. Otherwise fall
            # back to the local Aer path (fast, no network).
            if self._real_active:
                from qiskit_ibm_runtime import SamplerV2
                isa = self.cvar_optimizer.pm.run(bound)
                sampler = SamplerV2(mode=self.cvar_optimizer.simulator)
                job = sampler.run([isa], shots=self.sampling_shots)
                self._record_sampling_job(job, stage="sampler-final")
                pub_result = job.result()[0]
                # SamplerV2 returns a ``DataBin`` whose attribute matches
                # the classical register name created by ``measure_all`` ("meas").
                bit_array = getattr(pub_result.data, "meas", None)
                if bit_array is None:
                    raise RuntimeError("SamplerV2 result missing 'meas' bit array")
                counts = bit_array.get_counts()
            else:
                backend = AerSimulator()
                from qiskit import transpile

                qc = transpile(bound, backend, optimization_level=1)
                job = backend.run(qc, shots=self.sampling_shots)
                counts = job.result().get_counts()

            best_x: Optional[np.ndarray] = None
            best_cost = float("inf")
            for bitstring, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:64]:
                # Qiskit returns little-endian: rightmost char is qubit 0.
                bits = bitstring.replace(" ", "")
                x = np.array([int(bits[-(i + 1)]) for i in range(n)], dtype=float)
                cost = float(x @ h + x @ np.triu(J, 1) @ x)
                if cost < best_cost:
                    best_cost = cost
                    best_x = x

            if best_x is None:
                logger.warning("[IBM-CVAR-VQE] No samples produced; falling back to all-on")
                return np.ones(n, dtype=float)
            return best_x

        except Exception as exc:
            logger.warning(
                "[IBM-CVAR-VQE] Bitstring sampling failed (%s); equal-weighting all assets",
                exc,
            )
            return np.ones(n, dtype=float)

    def _selection_to_weights(
        self, symbols: List[str], selection: np.ndarray
    ) -> Dict[str, float]:
        """Equal-weight the selected assets, honouring cardinality cap."""
        x = selection.astype(bool)
        if self.cardinality is not None:
            # If too many selected, keep top-k by the QUBO-favoured order
            # (highest mu - lambda*sigma_ii implied by negative h_i).
            if x.sum() > self.cardinality:
                # Selection of `cardinality` assets at random from selected
                # set — deterministic via sorted indices.
                idxs = np.where(x)[0][: self.cardinality]
                x = np.zeros_like(x)
                x[idxs] = True
        if not x.any():
            # Empty selection -> fall back to equal weights so the job
            # returns something usable.
            logger.warning("[IBM-CVAR-VQE] Empty selection; equal-weighting all assets")
            x = np.ones_like(x)
        weights = x.astype(float) / x.sum()
        return {symbols[i]: float(weights[i]) for i in range(len(symbols))}

    def _classical_reweight_selected(
        self,
        symbols: List[str],
        portfolio_weights: Dict[str, float],
        mu_annual: np.ndarray,
        sigma_annual: np.ndarray,
    ) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
        """Delegate to the shared post-quantum classical QP refinement helper.

        Historical CVaR-VQE-specific implementation now lives in
        ``core.postprocessing.classical_refinement.refine_selected_weights``
        so QAOA / Cirq / IonQ / D-Wave executors can share the exact
        same behaviour. Signature and telemetry keys are unchanged so
        existing tests, AI-reviewer prompts and downstream metadata
        consumers continue to work.
        """
        from core.postprocessing.classical_refinement import refine_selected_weights
        return refine_selected_weights(
            symbols,
            portfolio_weights,
            mu_annual,
            sigma_annual,
            risk_aversion=self.risk_aversion,
            tag="IBM-CVAR-VQE",
        )

    def _performance_metrics(
        self,
        symbols: List[str],
        portfolio_weights: Dict[str, float],
        mu_annual: np.ndarray,
        sigma_annual: np.ndarray,
    ) -> Dict[str, float]:
        w = np.array([portfolio_weights.get(s, 0.0) for s in symbols])
        expected_return = float(w @ mu_annual)
        portfolio_variance = float(w @ sigma_annual @ w)
        portfolio_vol = float(np.sqrt(max(portfolio_variance, 0.0)))
        sharpe = (expected_return / portfolio_vol) if portfolio_vol > 1e-12 else 0.0

        # Parametric (Gaussian) VaR/CVaR at 95% for reporting.
        z95 = 1.6448536269514722
        var_95 = float(-(expected_return - z95 * portfolio_vol))
        # CVaR_alpha for Gaussian: mu - sigma * phi(z)/alpha (alpha=0.05)
        phi_z = float(np.exp(-0.5 * z95 ** 2) / np.sqrt(2.0 * np.pi))
        cvar_95 = float(-(expected_return - portfolio_vol * phi_z / 0.05))

        return {
            "expected_annual_return": expected_return,
            "annual_volatility": portfolio_vol,
            "sharpe_ratio": float(sharpe),
            "concentration_index": float(np.sum(w ** 2)),
            "max_weight": float(np.max(w)),
            "num_assets_held": int(np.sum(w > 1e-9)),
            "var_95": var_95,
            "cvar_95": cvar_95,
        }
