"""Block Lewis weights and Lewis-weighted geometry.

Implements the block Lewis weight computation from Theorem 2.3 / Appendix E
of Patel & Manoj (ICLR 2026), following (Manoj & Ovsiankin, 2025, Algorithm 2).

Block Lewis weights provide a data-dependent positive-definite matrix M such that
the M-norm approximates the group-infinity norm up to a factor of O(sqrt(rank(A))),
enabling geometry-adaptive trust regions for the ball oracle methods.
"""

from __future__ import annotations

import numpy as np


def leverage_scores(M: np.ndarray) -> np.ndarray:
    """Exact leverage scores of rows of M.

    For M of shape (n, d), the leverage score of row i is:
        tau_i = M_i^T (M^T M)^{-1} M_i
    """
    n, d = M.shape
    G = M.T @ M + 1e-12 * np.eye(d)
    G_inv = np.linalg.inv(G)
    return np.einsum("ij,jk,ik->i", M, G_inv, M)


def block_lewis_weights(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    p: float = np.inf,
    fudge: float = 1.1,
) -> np.ndarray:
    """Compute approximate block Lewis weights on augmented blocks [A_i | b_i].

    Uses the iterative reweighting scheme from (Manoj & Ovsiankin, 2025):
    run ~3 log(m) iterations of leverage-score reweighting on the augmented
    matrix [A | b] with block structure.

    Args:
        A_groups: List of m design matrices, each (n_i, d).
        b_groups: List of m response vectors, each (n_i,).
        p: The norm parameter (p = inf for robust DRO, p >= 2 for interpolating).
        fudge: Multiplicative safety factor (>= 1).

    Returns:
        Per-row weights of shape (n,) where n = sum_i n_i.
    """
    blocks = [np.hstack([A_i, b_i[:, None]]) for A_i, b_i in zip(A_groups, b_groups)]
    m = len(blocks)
    A_aug = np.vstack(blocks)
    n, d = A_aug.shape

    # Build row-to-block mapping.
    block_ids = np.concatenate([np.full(B.shape[0], i, dtype=int) for i, B in enumerate(blocks)])

    # Number of iterations: O(log m).
    T = int(np.ceil(3 * np.log(max(m, 2))))

    # Exponent for reweighting: 1/2 - 1/p.
    expo = 0.5 if p == np.inf else 0.5 - 1.0 / p

    # Initialize: average block weight ~ d/m per row.
    w = (d / m) * np.ones(n)
    w_history = [w.copy()]

    for _ in range(T):
        D = np.diag(np.power(np.maximum(w, 1e-16), expo))
        lev = leverage_scores(D @ A_aug)

        # Normalize so leverage scores sum to d.
        lev *= d / max(lev.sum(), 1e-16)

        # Aggregate to block level, then broadcast back.
        block_w = np.zeros(m)
        for i in range(m):
            block_w[i] = lev[block_ids == i].sum()
        w = block_w[block_ids]
        w_history.append(w.copy())

    # Average over iterations and apply fudge factor.
    w_avg = fudge * sum(w_history) / len(w_history)
    return np.maximum(w_avg, 1e-16)


def build_lewis_geometry(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    p: float = np.inf,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build the Lewis-weighted geometry matrix and return its Cholesky factor.

    Constructs M = A^T diag(w_geom) A + eps I where w_geom are the (transformed)
    block Lewis weights, and returns L such that M = L^T L.

    This L defines the Lewis-norm: ||x||_M = ||Lx||_2.

    Args:
        A_groups: List of m design matrices.
        b_groups: List of m response vectors.
        p: Norm parameter.
        eps: Regularization for positive-definiteness.

    Returns:
        L: Cholesky factor of shape (d, d).
    """
    w = block_lewis_weights(A_groups, b_groups, p=p)
    A = np.vstack(A_groups)
    d = A.shape[1]

    # Transform weights: w_geom = w^{1 - 2/p} for finite p, w for p = inf.
    w_geom = w if p == np.inf else np.power(w, 1.0 - 2.0 / p)

    M = A.T @ np.diag(w_geom) @ A + eps * np.eye(d)

    for _ in range(5):
        try:
            return np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            eps *= 10.0
            M += eps * np.eye(d)

    return np.eye(d)  # fallback


def project_onto_lewis_ball(
    x: np.ndarray,
    center: np.ndarray,
    R: float,
    L: np.ndarray,
) -> np.ndarray:
    """Project x onto the Lewis-norm ball {z : ||z - center||_M <= R}.

    The M-norm is ||v||_M = ||L v||_2 where M = L^T L.
    """
    diff = x - center
    y = L @ diff
    norm_y = np.linalg.norm(y)
    if norm_y <= R:
        return x
    if norm_y == 0.0:
        return center.copy()
    return center + np.linalg.solve(L, (R / norm_y) * y)
