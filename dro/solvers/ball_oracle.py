"""Ball-oracle methods for group DRO (Algorithm 1 of Patel & Manoj, ICLR 2026).

Implements trust-region methods that repeatedly solve the smoothed objective
within shrinking balls using a damped Newton solver:

    For k = 1, ..., n_outer:
        x_k = Newton-ball-oracle(center_k, R_k)
        center_{k+1} = x_k
        R_{k+1} = R_decay * R_k

Two geometry choices:
  - Naive (Euclidean): ||x - center||_2 <= R
  - Lewis: ||x - center||_M <= R, where M = L^T L from block Lewis weights

The paper shows that Lewis geometry gives O(min{rank(A), m}^{1/3} / eps^{2/3})
iteration complexity vs O(m^{1/3} / eps^{2/3}) for naive geometry (Theorem 1, Table 1).
"""

from __future__ import annotations

import numpy as np

from ..problem import SolverResult, max_group_loss
from ..smoothing import smooth_value, smooth_grad_and_hessian
from ..lewis_weights import project_onto_lewis_ball


# ---------------------------------------------------------------------------
# Inner Newton solver (shared logic for both geometries)
# ---------------------------------------------------------------------------

def _newton_ball_oracle(
    A_groups, b_groups, center, R, beta, delta,
    max_newton_steps, project_fn,
    backtrack_beta=0.5, backtrack_c=1e-4, tol=1e-6,
):
    """Damped Newton solver inside a trust-region ball.

    Approximately solves: min_x f_smooth(x)  s.t. x in Ball(center, R)
    using a few steps of Newton with backtracking and projection.

    Args:
        project_fn: callable(x_trial) -> projected x_trial onto the ball.

    Returns the approximate minimizer, or None on failure.
    """
    x = center.copy()
    F = smooth_value(A_groups, b_groups, x, beta, delta)
    if not np.isfinite(F):
        return None

    for k in range(1, max_newton_steps + 1):
        g, H = smooth_grad_and_hessian(A_groups, b_groups, x, beta, delta)
        if g is None or H is None:
            return None

        g_norm = np.linalg.norm(g)
        if g_norm <= tol * max(1.0, abs(F)):
            return x

        # Damped Newton direction.
        lam = 1e-8 * max(1.0, g_norm)
        H_reg = H + lam * np.eye(H.shape[0])
        try:
            dx = -np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            return x

        # Ensure descent direction.
        if g @ dx > 0:
            dx = -dx

        # Backtracking with projection.
        alpha = 1.0
        descent = g @ dx
        for _ in range(40):
            x_trial = project_fn(x + alpha * dx)
            F_trial = smooth_value(A_groups, b_groups, x_trial, beta, delta)
            if np.isfinite(F_trial) and F_trial <= F + backtrack_c * alpha * descent:
                x, F = x_trial, F_trial
                break
            alpha *= backtrack_beta
        else:
            return x  # line search failed, return current best

    return x


# ---------------------------------------------------------------------------
# Euclidean (naive) ball projection
# ---------------------------------------------------------------------------

def _project_euclidean(center, R, x_trial):
    diff = x_trial - center
    nrm = np.linalg.norm(diff)
    if nrm <= R:
        return x_trial
    return center + (R / max(nrm, 1e-16)) * diff


# ---------------------------------------------------------------------------
# Outer ball-oracle loop
# ---------------------------------------------------------------------------

def _run_ball_oracle(
    A_groups, b_groups, x0, beta, delta,
    R0, n_outer, R_decay, max_newton_steps, project_fn_factory,
):
    """Shared outer loop for both naive and Lewis ball-oracle methods.

    Args:
        project_fn_factory: callable(center, R) -> project_fn(x_trial)
    """
    x = x0.copy()
    center = x0.copy()
    R = float(R0)

    F_init = max_group_loss(A_groups, b_groups, x)
    if not np.isfinite(F_init):
        return None
    explode = 1e6 * max(F_init, 1.0)

    best = F_init
    iters, best_vals = [0], [best]

    for k in range(1, n_outer + 1):
        project_fn = project_fn_factory(center, R)
        x_new = _newton_ball_oracle(
            A_groups, b_groups, center, R, beta, delta,
            max_newton_steps, project_fn,
        )
        if x_new is None or not np.all(np.isfinite(x_new)):
            return None

        x = x_new
        center = x.copy()

        F_true = max_group_loss(A_groups, b_groups, x)
        if not np.isfinite(F_true) or F_true > explode:
            return None

        best = min(best, F_true)
        iters.append(k)
        best_vals.append(best)

        R *= R_decay
        if R <= 0:
            break

    return SolverResult(x_final=x, best_loss=best, iters=iters, best_values=best_vals)


def ball_oracle_naive(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    beta: float,
    delta: float,
    R0: float,
    n_outer: int = 100,
    R_decay: float = 0.9,
    max_newton_steps: int = 5,
) -> SolverResult | None:
    """Ball-oracle method with Euclidean (naive) geometry.

    Uses trust regions defined by ||x - center||_2 <= R.
    This gives m^{1/3} / eps^{2/3} iteration complexity (Table 1, row "naive").
    """
    def factory(center, R):
        return lambda x_trial: _project_euclidean(center, R, x_trial)

    return _run_ball_oracle(
        A_groups, b_groups, x0, beta, delta,
        R0, n_outer, R_decay, max_newton_steps, factory,
    )


def ball_oracle_lewis(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    L: np.ndarray,
    beta: float,
    delta: float,
    R0: float,
    n_outer: int = 100,
    R_decay: float = 0.9,
    max_newton_steps: int = 5,
) -> SolverResult | None:
    """Ball-oracle method with Lewis-weighted geometry (Algorithm 1).

    Uses trust regions defined by ||x - center||_M <= R where M = L^T L
    is constructed from block Lewis weights.

    This gives min{rank(A), m}^{1/3} / eps^{2/3} iteration complexity (Theorem 1).

    Args:
        L: Cholesky factor from lewis_weights.build_lewis_geometry().
    """
    def factory(center, R):
        return lambda x_trial: project_onto_lewis_ball(x_trial, center, R, L)

    return _run_ball_oracle(
        A_groups, b_groups, x0, beta, delta,
        R0, n_outer, R_decay, max_newton_steps, factory,
    )
