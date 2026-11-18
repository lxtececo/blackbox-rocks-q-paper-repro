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
6. Run :meth:`CVaRVQEOptimizer.enhanced_vqe_optimization` (CVaR of the
   sampled bitstring energies, fixed alpha).
7. Sample the trained ansatz, return the lowest-cost sampled bitstring
   that satisfies the cardinality (see ``_select``), equal-weight the
   selected assets and compute portfolio performance metrics on the
   denoised covariance.
8. Return a result dict shaped exactly like the one
   ``PCEPortfolioOptimizer.optimize`` returns, so the executor can stay
   agnostic to which backend produced it.

Hardware path
-------------

When ``use_real_hardware=True`` (or ``CVaRConfig.simulator_mode=False``),
:class:`CVaRVQEOptimizer` resolves an IBM Runtime backend via
``QiskitRuntimeService.least_busy`` (BYOK token first, ``IBM_QUANTUM_TOKEN``
env-var fallback) and runs every objective evaluation and the final
sampling as SamplerV2 jobs against that QPU. The
returned dict's ``vendor_runs`` list carries the IBM Runtime job IDs
so the Job History view can render "View on IBM Quantum portal"
deep-links (F24 Phase A.2). Without a token or a reachable device the
optimizer raises; a hardware request is not run on a simulator.
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
from core.portfolio_referee import referee
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
        backend=None,
    ):
        self.cvar_config = cvar_config
        self.use_real_hardware = bool(use_real_hardware)
        self.cvar_optimizer = CVaRVQEOptimizer(
            cvar_config, use_real_hardware=self.use_real_hardware, backend=backend
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
            "depth=%d, alpha=%.3f",
            n_assets, self.ansatz_depth, self.cvar_config.alpha,
        )

        try:
            mu_annual, sigma_annual, rmt_applied, rmt_info = _annualised_returns_and_cov(prices)

            # A target of n or more assets constrains nothing.
            cardinality = self.cardinality if self.cardinality and self.cardinality < n_assets else None
            h, J, const = _build_markowitz_qubo(
                mu_annual, sigma_annual,
                risk_aversion=self.risk_aversion,
                cardinality=cardinality,
                cardinality_penalty=self.cardinality_penalty,
            )

            H, ising_offset = _qubo_to_ising_sparse_pauli(h, J)
            # Total additive constant when comparing <H> to QUBO cost
            energy_offset = const + ising_offset

            ansatz = RealAmplitudes(num_qubits=n_assets, reps=self.ansatz_depth)

            logger.info(
                "[IBM-CVAR-VQE] Built Ising Hamiltonian: %d Pauli terms, "
                "%d qubits, ansatz_params=%d",
                len(H), n_assets, ansatz.num_parameters,
            )

            vqe_result = self.cvar_optimizer.enhanced_vqe_optimization(
                ansatz=ansatz, hamiltonian=H,
                classical_optimizer='COBYLA',
            )

            best_params = np.asarray(vqe_result["optimal_parameters"], dtype=float)
            counts = self.cvar_optimizer.sample_counts(best_params, self.sampling_shots, stage="sampler-final")
            selection, selection_info = self._select(counts, h, J, cardinality)
            exact_reference = referee(h, J, selection, const=const, cardinality=cardinality, min_assets=1,
                                      sampled_shots=selection_info["shots"])

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
                "optimizer_converged": bool(vqe_result.get("converged")),
                "evaluation_budget": vqe_result["configuration"]["evaluation_budget"],
                "evaluation_budget_raised": vqe_result["configuration"]["evaluation_budget_raised"],
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
                # IBM Runtime job IDs of every sampler job; empty on the simulator.
                "vendor_runs": list(self.cvar_optimizer.vendor_runs),
                # Standing of the returned selection among all admissible ones (None above the enumeration limit).
                "exact_reference": exact_reference,
                "metadata": {
                    "num_assets": n_assets,
                    "rmt_applied": rmt_applied,
                    "rmt_info": rmt_info,
                    "risk_aversion": self.risk_aversion,
                    "cardinality": cardinality,
                    "energy_offset": energy_offset,
                    "qubo_cost": qubo_cost,
                    "selection": selection_info,
                    "selected_assets": [symbols[i] for i in range(n_assets) if selection[i] > 0.5],
                    "optimal_parameters": vqe_result["optimal_parameters"],
                    "seed": self.cvar_config.seed,
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
                        "Quantum measurement outcomes",
                    ],
                },
                "total_optimization_time": total_time,
                "method": "IBM_CVAR_VQE",
                "configuration": {
                    "alpha": self.cvar_config.alpha,
                    "alpha_schedule": "fixed",
                    "objective": "cvar_of_sampled_bitstring_energies",
                    "ansatz_depth": self.ansatz_depth,
                    "risk_aversion": self.risk_aversion,
                    "cardinality": cardinality,
                    "measurement_budget": self.cvar_config.measurement_budget,
                    "sampling_shots": self.sampling_shots,
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

    def _select(
        self,
        counts: Dict[int, int],
        h: np.ndarray,
        J: np.ndarray,
        cardinality: Optional[int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Lowest-cost sampled selection that holds ``cardinality`` assets (at least one without a target).

        ``counts`` keys are integers whose bit i is asset i. Ties go to the more
        frequent sample. If no sample is feasible, the lowest-cost sample is
        repaired by the QUBO cost, never by column order.
        """
        n = h.shape[0]
        keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        freq = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
        bits = ((keys[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(float)
        upper = np.triu(J, 1)
        cost = bits @ h + np.einsum("bi,ij,bj->b", bits, upper, bits)
        held = bits.sum(axis=1)
        feasible = held == cardinality if cardinality is not None else held >= 1
        pool = np.flatnonzero(feasible) if feasible.any() else np.arange(len(keys))
        best = pool[np.lexsort((-freq[pool], cost[pool]))[0]]
        x = bits[best].copy()
        info = {
            "rule": "lowest_cost_feasible_sample",
            "shots": int(freq.sum()),
            "distinct_samples": int(len(keys)),
            "feasible_shots": int(freq[feasible].sum()),
            "selected_sample_count": int(freq[best]),
            "repaired": False,
        }
        if not feasible.any():
            x = self._repair(x, h, upper, cardinality if cardinality is not None else 1)
            info["repaired"] = True
        return x, info

    @staticmethod
    def _repair(x: np.ndarray, h: np.ndarray, upper: np.ndarray, target: int) -> np.ndarray:
        """Add or drop one asset at a time, each time the one that leaves the lowest QUBO cost."""
        x = x.copy()
        while int(x.sum()) != target:
            current = 1.0 if x.sum() > target else 0.0
            candidates = np.flatnonzero(x == current)
            trial_costs = []
            for i in candidates:
                y = x.copy()
                y[i] = 1.0 - current
                trial_costs.append(float(y @ h + y @ upper @ y))
            x[candidates[int(np.argmin(trial_costs))]] = 1.0 - current
        return x

    def _selection_to_weights(
        self, symbols: List[str], selection: np.ndarray
    ) -> Dict[str, float]:
        """Equal-weight the selected assets."""
        x = selection.astype(bool)
        if not x.any():
            raise ValueError("empty selection")
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
