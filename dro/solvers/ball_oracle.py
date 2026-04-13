"""Ball-oracle methods for group DRO (Algorithm 1 of Patel & Manoj, ICLR 2026).

Two outer-loop strategies:

  Naive iteration (accelerated=False):
    For k = 1, ..., n_outer:
        x_k = Newton-ball-oracle(center_k, R_k)
        center_{k+1} = x_k
        R_{k+1} = R_decay * R_k

  MS acceleration (accelerated=True, Algorithm 1 lines 7-9):
    Build an MS oracle from the ball optimization oracle, then run the
    Monteiro-Svaiter accelerated scheme (Algorithm 3). This improves the
    outer iteration count from ||x0-x*||_M / eps  to  (||x0-x*||_M / eps)^{2/3}.

Two geometry choices:
  - Naive (Euclidean): ||x - center||_2 <= R
  - Lewis: ||x - center||_M <= R, where M = L^T L from block Lewis weights
"""

from __future__ import annotations

import numpy as np

from ..problem import SolverResult, max_group_loss
from ..smoothing import smooth_value, smooth_grad_and_hessian, smooth_value_and_grad
from ..lewis_weights import project_onto_lewis_ball
from ..acceleration import ms_accelerate, make_ms_oracle_robust


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
# Outer loop: naive (repeated) iteration
# ---------------------------------------------------------------------------

def _run_ball_oracle_naive(
    A_groups, b_groups, x0, beta, delta,
    R0, n_outer, R_decay, max_newton_steps, project_fn_factory,
):
    """Naive outer loop: iterate ball oracle calls with shrinking radius."""
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


# ---------------------------------------------------------------------------
# Outer loop: MS-accelerated iteration (Algorithm 1, lines 7-9)
# ---------------------------------------------------------------------------

def _run_ball_oracle_accelerated(
    A_groups, b_groups, x0, beta, delta,
    radius, n_outer, max_newton_steps, project_fn_factory, M,
):
    """MS-accelerated outer loop for p=inf (Algorithm 1, lines 7-9).

    Uses a fixed ball radius (not shrinking) — the acceleration framework
    handles convergence by choosing query points adaptively.

    Args:
        radius: Fixed ball radius for all oracle calls (r = C/eps from Alg. 1 line 7).
        M: Geometry matrix (L^T L for Lewis, or I for naive).
    """
    # Build the inner ball oracle callable.
    def ball_oracle_call(center, R):
        project_fn = project_fn_factory(center, R)
        return _newton_ball_oracle(
            A_groups, b_groups, center, R, beta, delta,
            max_newton_steps, project_fn,
        )

    # Gradient of the smoothed objective for the v-update in Algorithm 3.
    def f_grad(x):
        _, g = smooth_value_and_grad(A_groups, b_groups, x, beta, delta)
        return g

    # Build MS oracle from ball oracle (Carmon et al., 2020, Proposition 5).
    ms_oracle = make_ms_oracle_robust(
        ball_oracle_fn=ball_oracle_call,
        f_grad=f_grad,
        M=M,
        radius=radius,
    )

    # Value function for tracking.
    def f_value(x):
        return smooth_value(A_groups, b_groups, x, beta, delta)

    # Run Algorithm 3 (OptimalMSAcceleration).
    iterates = ms_accelerate(
        f_value=f_value,
        f_grad=f_grad,
        ms_oracle=ms_oracle,
        x0=x0,
        M=M,
        T=n_outer,
        s=np.inf,  # p = inf => s = inf movement bound
    )

    # Convert iterates to SolverResult, tracking best true max-loss.
    best = max_group_loss(A_groups, b_groups, x0)
    iters, best_vals = [0], [best]
    x_best = x0.copy()

    for k, x_k in enumerate(iterates[1:], 1):
        if not np.all(np.isfinite(x_k)):
            continue
        F_true = max_group_loss(A_groups, b_groups, x_k)
        if np.isfinite(F_true) and F_true < best:
            best = F_true
            x_best = x_k.copy()
        iters.append(k)
        best_vals.append(best)

    return SolverResult(x_final=x_best, best_loss=best, iters=iters, best_values=best_vals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    accelerated: bool = False,
) -> SolverResult | None:
    """Ball-oracle method with Euclidean (naive) geometry.

    Uses trust regions defined by ||x - center||_2 <= R.
    This gives m^{1/3} / eps^{2/3} iteration complexity (Table 1, row "naive").

    Args:
        accelerated: If True, uses MS acceleration (Algorithm 3) instead of
            naive repeated iteration. When accelerated, R0 is used as the
            fixed ball radius and R_decay is ignored.
    """
    def factory(center, R):
        return lambda x_trial: _project_euclidean(center, R, x_trial)

    if accelerated:
        d = A_groups[0].shape[1]
        M = np.eye(d)
        return _run_ball_oracle_accelerated(
            A_groups, b_groups, x0, beta, delta,
            radius=R0, n_outer=n_outer,
            max_newton_steps=max_newton_steps,
            project_fn_factory=factory, M=M,
        )
    else:
        return _run_ball_oracle_naive(
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
    accelerated: bool = False,
) -> SolverResult | None:
    """Ball-oracle method with Lewis-weighted geometry (Algorithm 1).

    Uses trust regions defined by ||x - center||_M <= R where M = L^T L
    is constructed from block Lewis weights.

    This gives min{rank(A), m}^{1/3} / eps^{2/3} iteration complexity (Theorem 1).

    Args:
        L: Cholesky factor from lewis_weights.build_lewis_geometry().
        accelerated: If True, uses MS acceleration (Algorithm 3) instead of
            naive repeated iteration. When accelerated, R0 is used as the
            fixed ball radius and R_decay is ignored.
    """
    def factory(center, R):
        return lambda x_trial: project_onto_lewis_ball(x_trial, center, R, L)

    if accelerated:
        M = L.T @ L
        return _run_ball_oracle_accelerated(
            A_groups, b_groups, x0, beta, delta,
            radius=R0, n_outer=n_outer,
            max_newton_steps=max_newton_steps,
            project_fn_factory=factory, M=M,
        )
    else:
        return _run_ball_oracle_naive(
            A_groups, b_groups, x0, beta, delta,
            R0, n_outer, R_decay, max_newton_steps, factory,
        )
