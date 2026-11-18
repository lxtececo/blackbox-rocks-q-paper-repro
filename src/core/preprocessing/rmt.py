"""Random Matrix Theory (Marchenko-Pastur) preprocessing.

Generic, optimizer-agnostic helpers used to clean the noise bulk out of
sample correlation / covariance matrices before they enter a downstream
portfolio optimization routine.

Originally factored out of `core.optimizers.quantum.ionq.ionq_optimizer`
where it was used to denoise the affinity matrix fed into Louvain
clustering for BF-DCQO. The same machinery applies to any algorithm that
consumes Σ — classical mean-variance, QAOA / VQE / CVaR-VQE risk terms,
D-Wave CQM/DQM quadratic objectives, etc.

Marchenko-Pastur upper edge for an N×N sample correlation matrix
estimated from T observations of i.i.d. unit-variance noise:

    λ⁺ = σ²·(1 + 1/q + 2·√(1/q)),   q = T / N

Eigenvalues ≤ λ⁺ are statistically indistinguishable from noise and are
collapsed; eigenvalues > λ⁺ carry "signal" and are kept.

References
----------
* Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets
  of random matrices"
* Laloux et al. (1999), "Noise dressing of financial correlation matrices"
* Bun, Bouchaud & Potters (2017), "Cleaning large correlation matrices"
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrail — when NOT to apply MP denoising
# ---------------------------------------------------------------------------

def should_apply_rmt(
    n_vars: int,
    n_obs: int,
    *,
    min_assets: int = 8,
    min_q: float = 2.0,
) -> bool:
    """Return True iff a sample covariance is large enough for MP denoising.

    Args:
        n_vars:     N — number of assets in the matrix.
        n_obs:      T — number of return observations used to estimate it.
        min_assets: skip MP for tiny portfolios (default 8); below this the
                    eigenvalue bulk is dominated by edge effects and the cure
                    is worse than the disease.
        min_q:      require T/N ≥ min_q (default 2.0) so the MP regime is
                    well-defined; q < 1 makes the sample covariance singular.
    """
    if n_vars < min_assets:
        return False
    if n_obs <= 0:
        return False
    q = n_obs / float(n_vars)
    return q >= min_q


# ---------------------------------------------------------------------------
# Marchenko-Pastur upper edge
# ---------------------------------------------------------------------------

def marchenko_pastur_threshold(n_obs: int, n_vars: int, sigma2: float = 1.0) -> float:
    """Upper edge λ⁺ of the Marchenko-Pastur eigenvalue bulk.

    For a sample correlation matrix the per-asset variance is normalised to
    one, so `sigma2=1.0` is the right default; pass a different value when
    denoising a *covariance* matrix directly without rescaling.
    """
    if n_vars <= 0 or n_obs <= 0:
        raise ValueError("n_vars and n_obs must be positive")
    q = n_obs / float(n_vars)
    return sigma2 * (1.0 + 1.0 / q + 2.0 * np.sqrt(1.0 / q))


# ---------------------------------------------------------------------------
# Correlation denoising
# ---------------------------------------------------------------------------

def denoise_correlation(
    C: np.ndarray,
    n_obs: int,
    *,
    drop_market_mode: bool = False,
    replace_noise_with_mean: bool = True,
    renormalize_diagonal: bool = True,
    min_diag: float = 1e-8,
) -> np.ndarray:
    """Return a Marchenko-Pastur-cleaned correlation matrix.

    Args:
        C:                       N × N symmetric correlation matrix.
        n_obs:                   number of observations used to estimate C.
        drop_market_mode:        if True the largest eigenvalue (interpreted
                                 as the global / "market" mode) is zeroed
                                 in addition to the noise bulk. Useful when
                                 the downstream consumer is a *clustering*
                                 routine that benefits from removing the
                                 dominant common factor; set False when the
                                 caller still wants the full signal (e.g.
                                 mean-variance risk term).
        replace_noise_with_mean: if True, noise eigenvalues are replaced by
                                 their average (trace-preserving Laloux
                                 cleaning); if False they are zeroed out.
                                 Trace preservation keeps total variance
                                 unchanged and is recommended for QP
                                 objectives.
        renormalize_diagonal:    if True the cleaned matrix is rescaled so
                                 its diagonal is exactly 1.0 (proper
                                 correlation matrix; required for
                                 `denoise_covariance` to round-trip
                                 variances faithfully). Set False to keep
                                 a rank-reduced "affinity" matrix — useful
                                 for clustering pipelines (e.g. BF-DCQO)
                                 that explicitly want the dominant mode
                                 stripped without inflating residuals.
        min_diag:                lower clip applied to the cleaned diagonal
                                 to keep the matrix positive-definite for
                                 downstream Cholesky factorisations.

    Returns:
        N × N cleaned correlation matrix (symmetric, real).
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got shape {C.shape}")
    n_vars = C.shape[0]

    # I12 — emit a structured warning when the sample size T is so small
    # vs the number of assets N that the Marchenko-Pastur bulk dominates
    # the spectrum and the cleaned matrix is effectively all noise. This
    # path is reachable when callers bypass ``should_apply_rmt`` / 
    # ``maybe_denoise_covariance`` (e.g. BF-DCQO's affinity matrix for
    # Louvain clustering still calls ``denoise_correlation`` directly).
    # The AI peer-reviewer surfaces this warning to the user so a run
    # that was technically "denoised" but with q << 1 doesn't get a free
    # pass on the noise-bulk-dominated regime.
    if n_obs > 0:
        q = n_obs / float(n_vars)
        if q < 0.1:
            logger.warning(
                "RMT applied with severely under-determined sample "
                "(N=%d, T=%d, q=T/N=%.3f < 0.1) — the Marchenko-Pastur "
                "bulk dominates the spectrum and most eigenvalues will "
                "be treated as noise; consider a longer history or "
                "fewer assets before trusting this matrix downstream",
                n_vars, n_obs, q,
            )
        elif q < 1.0:
            logger.warning(
                "RMT applied in singular regime (N=%d, T=%d, q=T/N=%.3f "
                "< 1.0) — sample covariance is rank-deficient; cleaning "
                "still proceeds via eigh but signal/noise separation is "
                "weaker than the q ≥ 2 reference regime",
                n_vars, n_obs, q,
            )

    # Symmetrise input to suppress any small numerical asymmetry
    C_sym = (C + C.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(C_sym)

    lambda_plus = marchenko_pastur_threshold(n_obs, n_vars)
    signal_mask = eigenvalues > lambda_plus

    if not np.any(signal_mask):
        logger.debug(
            "RMT: all eigenvalues below MP threshold λ⁺=%.4f (N=%d, T=%d) — "
            "returning near-identity matrix",
            lambda_plus, n_vars, n_obs,
        )
        return np.eye(n_vars) * max(min_diag, 1e-6)

    cleaned = eigenvalues.copy()

    if replace_noise_with_mean:
        # Trace-preserving cleaning (Laloux): redistribute noise mass
        noise_eigs = eigenvalues[~signal_mask]
        if noise_eigs.size > 0:
            cleaned[~signal_mask] = float(np.mean(noise_eigs))
    else:
        cleaned[~signal_mask] = 0.0

    if drop_market_mode and np.sum(signal_mask) > 1:
        # The largest eigenvalue is the global / market mode. Zero it out
        # so clustering sees only the residual sector structure. Only do this
        # if at least one other signal eigenvalue survives — otherwise we'd
        # collapse the matrix to noise.
        max_idx = int(np.argmax(cleaned))
        cleaned[max_idx] = 0.0

    C_star = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    C_star = (C_star + C_star.T) / 2.0
    if renormalize_diagonal:
        # Re-normalise to a proper correlation matrix: each diagonal must
        # be 1.0. Spectral cleaning preserves trace but generally perturbs
        # individual diagonal entries; the standard Laloux post-processing
        # rescales the rows/cols by D = diag(C*)^(-1/2) so the result is a
        # valid correlation matrix again. This also makes
        # denoise_covariance round-trip exact for per-asset variances.
        diag = np.diag(C_star).copy()
        diag = np.maximum(diag, min_diag)
        inv_sqrt = 1.0 / np.sqrt(diag)
        C_star = C_star * np.outer(inv_sqrt, inv_sqrt)
        np.fill_diagonal(C_star, 1.0)
    else:
        # Rank-reduced affinity matrix (e.g. for clustering): keep the
        # spectral structure as-is, just clip negative / vanishing diagonal
        # entries so downstream Cholesky / linear solves don't blow up.
        np.fill_diagonal(C_star, np.maximum(np.diag(C_star), min_diag))
    return C_star


# ---------------------------------------------------------------------------
# Covariance denoising (round-trip via correlation)
# ---------------------------------------------------------------------------

def _cov_to_corr(Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert covariance Σ → correlation C and return the std-dev vector."""
    diag = np.diag(Sigma)
    # Clip negative / zero variances that can sneak in from float noise
    diag = np.maximum(diag, 1e-12)
    stds = np.sqrt(diag)
    inv = 1.0 / stds
    C = Sigma * np.outer(inv, inv)
    # Force exact-1 diagonal
    np.fill_diagonal(C, 1.0)
    return C, stds


def _corr_to_cov(C: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return C * np.outer(stds, stds)


def denoise_covariance(
    Sigma: np.ndarray,
    n_obs: int,
    *,
    drop_market_mode: bool = False,
    replace_noise_with_mean: bool = True,
) -> np.ndarray:
    """Marchenko-Pastur cleaning applied to a *covariance* matrix.

    Internally normalises Σ → correlation, cleans the eigenbulk, then rescales
    back to a covariance matrix using the original standard deviations. This
    keeps per-asset variances faithful and only removes spurious off-diagonal
    correlation noise.
    """
    C, stds = _cov_to_corr(Sigma)
    C_clean = denoise_correlation(
        C,
        n_obs,
        drop_market_mode=drop_market_mode,
        replace_noise_with_mean=replace_noise_with_mean,
    )
    return _corr_to_cov(C_clean, stds)


# ---------------------------------------------------------------------------
# High-level convenience wrapper used by optimizers — honours an env flag,
# a per-method allow-list and the small-portfolio guardrail so individual
# call sites can stay one line.
# ---------------------------------------------------------------------------

import os  # noqa: E402  (kept local to highlight the env-driven behaviour)


# Per-method default (True = denoise unless explicitly disabled). Most
# algorithms benefit; we default OFF for methods that already use a strong
# shrinkage estimator (e.g. Ledoit-Wolf) where double-cleaning is debatable.
_DEFAULT_METHOD_OPT_IN: dict[str, bool] = {
    # Classical
    "classical": True,
    "classical_markowitz": True,
    "markowitz": True,
    # Quantum — sample-covariance based
    "cirq_qaoa": True,
    "cirq_vqe": True,
    "cirq_cvar": True,
    "dwave_cqm": True,
    "dwave_dqm": True,
    "dwave_bqm": True,
    "ionq_bf_dcqo": True,
    "qaoa": True,
    "vqe": True,
    "cvar_vqe": True,
    "ibm_benchmark": True,
    # Quantum — already shrunk (Ledoit-Wolf). Default OPT-OUT to avoid
    # double-shrinkage: Ledoit-Wolf already pulls eigenvalues toward the
    # mean, and the Marchenko-Pastur threshold λ⁺=(1+1/√q)² is derived
    # for the sample covariance, not a shrunk one. Stack RMT on top only
    # explicitly via QFPO_RMT_METHOD_PCE=on.
    "pce": False,
    "advanced_qubo": False,
    "enhanced_dynamic": False,
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _method_opt_in(method: str | None) -> bool:
    """Resolve the per-method default, optionally overridden via env."""
    if not method:
        return True
    default = _DEFAULT_METHOD_OPT_IN.get(method, True)
    # Per-method env override: QFPO_RMT_METHOD_<METHOD>=on/off
    env_key = f"QFPO_RMT_METHOD_{method.upper()}"
    return _env_flag(env_key, default)


def maybe_denoise_covariance(
    cov: np.ndarray,
    n_obs: int,
    *,
    method: str | None = None,
    drop_market_mode: bool = False,
    min_assets: int = 8,
    min_q: float = 2.0,
) -> tuple[np.ndarray, bool]:
    """Apply MP denoising to a covariance matrix if configuration allows it.

    Returns
    -------
    (cleaned_cov, applied)
        `cleaned_cov` is the (possibly cleaned) covariance matrix and
        `applied` is True iff RMT was actually run. Callers can use the
        boolean to emit accurate step metadata / log lines without
        duplicating the gating logic.

    Configuration
    -------------
    * Global kill-switch:  ``QFPO_RMT_DENOISE=off`` disables RMT everywhere.
    * Per-method override: ``QFPO_RMT_METHOD_<METHOD>=on|off``.
    * Guardrail:           skipped when N < min_assets or T/N < min_q.
    """
    if cov is None:
        return cov, False

    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        return cov, False

    if not _env_flag("QFPO_RMT_DENOISE", True):
        return cov, False

    if not _method_opt_in(method):
        return cov, False

    n_vars = cov_arr.shape[0]
    if not should_apply_rmt(n_vars, n_obs, min_assets=min_assets, min_q=min_q):
        # INFO so operators can see the guardrail kicking in — otherwise a
        # quietly-skipped run looks identical to a successful denoise.
        q = (n_obs / n_vars) if n_vars else 0.0
        logger.info(
            "RMT skipped for %s: guardrail (N=%d, T=%d, q=%.2f, min_assets=%d, min_q=%.2f)",
            method or "?", n_vars, n_obs, q, min_assets, min_q,
        )
        return cov, False

    try:
        cleaned = denoise_covariance(
            cov_arr,
            n_obs,
            drop_market_mode=drop_market_mode,
        )
        logger.info(
            "RMT denoising applied for %s (N=%d, T=%d, λ⁺=%.4f)",
            method or "?", n_vars, n_obs,
            marchenko_pastur_threshold(n_obs, n_vars),
        )
        return cleaned, True
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("RMT denoising failed for %s: %s — using raw covariance",
                       method or "?", exc)
        return cov, False
