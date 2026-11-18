"""Shared QUBO / Ising mathematics for gate-based quantum optimizers.

Single authority for the QUBO → Ising conversion used across the IBM,
Google Cirq, and IonQ optimizer paths. Every gate-based optimizer that
encodes a portfolio selection problem as a QUBO must convert through
these helpers — never hand-roll the substitution again.

Canonical forms
---------------
Pair-form QUBO over binary variables ``x_i ∈ {0,1}``::

    cost(x) = Σ_i q_i x_i + Σ_{i<j} Q_ij x_i x_j

With the substitution ``x_i = (1 − z_i)/2`` (``z_i ∈ {+1, −1}``)::

    H(z) = Σ_i h_i z_i + Σ_{i<j} J_ij z_i z_j + offset

    h_i    = −q_i/2 − Σ_{j>i} Q_ij/4 − Σ_{j<i} Q_ji/4
    J_ij   = Q_ij/4                         (upper-triangular)
    offset = Σ_i q_i/2 + Σ_{i<j} Q_ij/4

Reference twin (kept in lock-step):
``core/optimizers/quantum/ibm/cvar_vqe_adapter._qubo_to_ising_sparse_pauli``.
Regression lock: ``backend/src/test/python/test_ising_conversion.py``
brute-forces ``cost(x) = H(z(x)) + offset`` over all 2^n bitstrings.

Traps (learned the hard way in F29, 2026-07-29)
-----------------------------------------------
1. Accumulate-vs-assign: when filling ``h`` in a loop whose inner j-loop
   already adds ``Q_ji/4`` shares into ``h[i]``, the diagonal term MUST
   accumulate (``h[i] -= q_i/2``), never assign.
2. ``z = 1 − 2·bits`` matches ``x = (1−z)/2`` (x=1 ↔ z=−1). Evaluating
   ``z @ J @ z`` with an upper-triangular J counts each pair ONCE — do
   not divide by 2.
3. Qiskit ``SparsePauliOp`` strings: leftmost char acts on qubit n−1,
   rightmost on qubit 0. Logical qubit i sits at string index ``n−1−i``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def qubo_pair_to_ising(
    q: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert a pair-form QUBO to Ising coefficients.

    Args:
        q: (n,) linear coefficients.
        Q: (n, n) matrix of PAIR coefficients. Only the upper triangle
           (i < j) is read; each pair's full coefficient lives in Q[i, j].
           The diagonal is ignored (diagonal of the QUBO belongs in ``q``
           for pair-form; see ``symmetric_matrix_to_pair``).

    Returns:
        (h, J, offset):
          h:      (n,) Z-field coefficients,
          J:      (n, n) upper-triangular ZZ coefficients,
          offset: constant term such that
                  ``cost(x) = h·z + zᵀJz + offset`` exactly for all x.
    """
    q = np.asarray(q, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n = q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        offset += 0.5 * q[i]
        h[i] -= 0.5 * q[i]  # accumulate: inner loop adds Q_ji/4 shares
        for j in range(i + 1, n):
            qij = Q[i, j]
            if qij == 0.0:
                continue
            offset += 0.25 * qij
            h[i] -= 0.25 * qij
            h[j] -= 0.25 * qij
            J[i, j] = 0.25 * qij

    return h, J, offset


def symmetric_matrix_to_pair(
    q: np.ndarray,
    S: np.ndarray,
    *,
    diagonal_into_linear: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-express a full symmetric quadratic form as pair-form QUBO.

    Given cost(x) = Σᵢ qᵢxᵢ + Σᵢⱼ Sᵢⱼ xᵢxⱼ with S symmetric (both
    triangles populated), the pair-form equivalent has:
      - pair coefficient Q[i,j] = 2·S[i,j] for i < j  (both triangles of
        the xᵢxⱼ term collapse into one pair term)
      - diagonal S[i,i] folds into the linear vector because xᵢ² = xᵢ for
        binary variables (when ``diagonal_into_linear``).

    Returns:
        (q_out, Q) — Q upper-triangular pair coefficients, ready for
        ``qubo_pair_to_ising``.
    """
    q = np.asarray(q, dtype=float).copy()
    S = np.asarray(S, dtype=float)
    n = q.shape[0]
    Q = np.zeros((n, n))
    for i in range(n):
        if diagonal_into_linear:
            q[i] += S[i, i]
        for j in range(i + 1, n):
            Q[i, j] = S[i, j] + S[j, i]
    return q, Q


def symmetric_matrix_to_ising(
    q: np.ndarray,
    S: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convenience: symmetric-matrix QUBO straight to Ising.

    Equivalent to ``qubo_pair_to_ising(*symmetric_matrix_to_pair(q, S))``.
    """
    return qubo_pair_to_ising(*symmetric_matrix_to_pair(q, S))


def ising_energy(
    bits: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    offset: float = 0.0,
) -> float:
    """Evaluate Ising energy for a bit vector.

    ``bits`` are x-values in {0,1}; internally mapped with z = 1 − 2·bits
    (x=1 ↔ z=−1, matching x = (1−z)/2). ``J`` is upper-triangular so
    ``z @ J @ z`` counts each pair exactly once — no /2.
    """
    bits = np.asarray(bits, dtype=int)
    z = 1.0 - 2.0 * bits
    return float(np.dot(h, z) + np.dot(z, J @ z) + offset)


def qubo_pair_cost(
    bits: np.ndarray,
    q: np.ndarray,
    Q: np.ndarray,
    const: float = 0.0,
) -> float:
    """Evaluate the pair-form QUBO directly (reference for tests)."""
    bits = np.asarray(bits, dtype=int)
    n = bits.shape[0]
    total = const + float(np.dot(q, bits))
    for i in range(n):
        for j in range(i + 1, n):
            total += Q[i, j] * bits[i] * bits[j]
    return float(total)


def cardinality_penalty_params(
    q: np.ndarray,
    Q: np.ndarray,
    K: int,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Add λ·(Σᵢ xᵢ − K)² to a pair-form QUBO.

    Expansion: λ(Σx − K)² = λΣᵢxᵢ + 2λΣᵢ<ⱼxᵢxⱼ − 2λKΣᵢxᵢ + λK²
      (using xᵢ² = xᵢ), i.e.
        q_i += λ(1 − 2K),  Q_ij += 2λ,  const += λK².

    Returns the updated (q, Q, const_add) — Q stays upper-triangular.
    """
    q = np.asarray(q, dtype=float).copy()
    Q = np.asarray(Q, dtype=float).copy()
    n = q.shape[0]
    Kf = float(K)
    for i in range(n):
        q[i] += lam * (1.0 - 2.0 * Kf)
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * lam
    return q, Q, lam * Kf * Kf


def cardinality_penalty_lambda(
    covariance: np.ndarray,
    returns: np.ndarray,
    n_assets: Optional[int] = None,
) -> float:
    """Heuristic λ scale for the cardinality penalty.

    Matches the historical IonQ heuristic: ``max(cov_scale, ret_scale) * n``
    — large enough to dominate any single covariance/return term so a
    violation of Σx = K always costs more than any plausible objective gain.
    """
    cov = np.asarray(covariance, dtype=float)
    ret = np.asarray(returns, dtype=float)
    n = n_assets if n_assets is not None else max(ret.shape[0], 1)
    cov_scale = float(np.max(np.abs(cov))) if cov.size else 1.0
    ret_abs_max = float(np.max(np.abs(ret))) if ret.size else 0.0
    ret_scale = ret_abs_max if ret_abs_max > 1e-10 else 1.0
    return max(cov_scale, ret_scale) * n


def cardinality_penalty_influence(q: np.ndarray, Q: np.ndarray) -> float:
    """Penalty weight Λ = 2·maxᵢ Iᵢ of arXiv:2602.23976, Eq. 10, for a pair-form objective.

    Iᵢ = |qᵢ| + ½ Σ_{j≠i} |Q_ij| is asset i's influence (the paper's
    |qᵢ + Q_ii| + Σ_{j≠i}|Q_ij| for a full symmetric matrix). No single flip
    can gain more than Λ, so a QUBO optimum always holds exactly K assets.
    """
    q = np.asarray(q, dtype=float)
    pairs = np.abs(np.triu(np.asarray(Q, dtype=float), 1))
    influence = np.abs(q) + 0.5 * (pairs.sum(axis=0) + pairs.sum(axis=1))
    return 2.0 * float(influence.max()) if influence.size else 0.0


def portfolio_qubo(
    returns: np.ndarray,
    covariance: np.ndarray,
    cardinality_K: int,
    risk_aversion: float,
    penalty_lambda: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Pair-form QUBO of fixed-cardinality portfolio selection (arXiv:2602.23976, Eq. 8).

        f(x) = −(1 − γ) μᵀx + γ xᵀCx + Λ (Σᵢ xᵢ − K)²,   γ = risk_aversion ∈ [0, 1]

    γ = 1 minimises variance, γ = 0 maximises return; the paper's weight
    γ' ∈ [−1, 1] is 2γ − 1, and its balanced choice γ' = 0 is γ = 0.5.

    Returns (q, Q, const, Λ) with f(x) = Σᵢ qᵢxᵢ + Σᵢ<ⱼ Q_ij xᵢxⱼ + const.
    """
    mu = np.asarray(returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    cov = 0.5 * (cov + cov.T)
    q = -(1.0 - risk_aversion) * mu + risk_aversion * np.diag(cov)
    Q = np.triu(2.0 * risk_aversion * cov, 1)
    lam = cardinality_penalty_influence(q, Q) if penalty_lambda is None else float(penalty_lambda)
    q, Q, const = cardinality_penalty_params(q, Q, cardinality_K, lam)
    return q, Q, const, lam


def pauli_labels_for_qubit_pair(
    n: int,
    i: int,
    j: Optional[int] = None,
) -> str:
    """Qiskit Pauli-string label with Z on logical qubit i (and j).

    Qiskit convention: leftmost char acts on qubit n−1, rightmost on
    qubit 0 — so logical qubit ``i`` sits at string index ``n−1−i``.
    """
    label = ["I"] * n
    label[n - 1 - i] = "Z"
    if j is not None:
        label[n - 1 - j] = "Z"
    return "".join(label)
