"""Log-barrier interior-point method for group DRO.

Solves the epigraph formulation:

    min_{x,t}  t   s.t.  ell_i(x) <= t  for all i

using a log-barrier approach with Newton steps on:

    phi(x, t; mu) = t - mu * sum_i log(t - ell_i(x))

The barrier parameter mu is decreased geometrically every inner_iters steps.
This is the standard IPM baseline from (Boyd & Vandenberghe, 2004, Section 6.4).
"""

from __future__ import annotations

import numpy as np

from ..problem import (
    SolverResult,
    group_losses,
    group_loss_gradients,
    group_loss_hessians,
    max_group_loss,
)


def _barrier_grad_hess(A_groups, b_groups, x, t, mu):
    """Gradient and Hessian of phi(x,t;mu) w.r.t. the joint variable [x; t].

    Returns (grad, H) of shapes (d+1,) and (d+1, d+1), or (None, None) on failure.
    """
    d = A_groups[0].shape[1]
    m = len(A_groups)

    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)) or not np.isfinite(t):
        return None, None

    slack = t - ell
    if np.any(slack <= 0):
        return None, None

    g_arr = group_loss_gradients(A_groups, b_groups, x)  # (m, d)
    H_list = group_loss_hessians(A_groups, b_groups)       # list of (d, d)

    inv = 1.0 / slack
    inv2 = inv ** 2

    # Gradient w.r.t. [x; t]
    grad_x = mu * (inv[:, None] * g_arr).sum(axis=0)
    grad_t = 1.0 - mu * inv.sum()

    # Hessian w.r.t. [x; t]
    H_xx = np.zeros((d, d))
    for i in range(m):
        H_xx += mu * (inv[i] * H_list[i] + inv2[i] * np.outer(g_arr[i], g_arr[i]))
    H_xt = -mu * (inv2[:, None] * g_arr).sum(axis=0)
    H_tt = mu * inv2.sum()

    H = np.zeros((d + 1, d + 1))
    H[:d, :d] = H_xx
    H[:d, d] = H_xt
    H[d, :d] = H_xt
    H[d, d] = H_tt

    grad = np.empty(d + 1)
    grad[:d] = grad_x
    grad[d] = grad_t

    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(H)):
        return None, None

    H += 1e-10 * np.eye(d + 1)
    return grad, H


def interior_point(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    mu0: float = 1e-3,
    tau: float = 0.9,
    inner_iters: int = 10,
    max_newton_steps: int = 100,
    backtrack_beta: float = 0.5,
    backtrack_c: float = 1e-4,
) -> SolverResult | None:
    """Log-barrier interior-point method.

    Args:
        x0: Initial parameter vector.
        mu0: Initial barrier parameter.
        tau: Barrier reduction factor (mu *= tau every inner_iters steps).
        inner_iters: Newton steps between barrier reductions.
        max_newton_steps: Total Newton step budget.
        backtrack_beta: Line search shrinkage factor.
        backtrack_c: Armijo sufficient decrease constant.

    Returns None on numerical failure.
    """
    d = A_groups[0].shape[1]
    x = x0.copy()

    ell0 = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell0)):
        return None

    max_ell0 = float(np.max(ell0))
    t = max_ell0 + max(10.0, 10.0 * abs(max_ell0))
    mu = float(mu0)

    F_true = max_group_loss(A_groups, b_groups, x)
    if not np.isfinite(F_true):
        return None

    best = F_true
    explode = 1e6 * max(F_true, 1.0)
    iters, best_vals = [0], [best]

    for k in range(1, max_newton_steps + 1):
        # Barrier value for line search reference.
        ell_cur = group_losses(A_groups, b_groups, x)
        slack_cur = t - ell_cur
        if np.any(slack_cur <= 0):
            return None
        phi_val = t - mu * np.sum(np.log(slack_cur))
        if not np.isfinite(phi_val):
            return None

        grad, H = _barrier_grad_hess(A_groups, b_groups, x, t, mu)
        if grad is None:
            return None

        try:
            delta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            return None

        dx, dt = delta[:d], delta[d]

        # Backtracking line search preserving strict feasibility.
        alpha = 1.0
        descent = grad @ delta
        success = False
        for _ in range(80):
            x_new = x + alpha * dx
            t_new = t + alpha * dt
            ell_new = group_losses(A_groups, b_groups, x_new)
            if np.all(np.isfinite(ell_new)):
                slack_new = t_new - ell_new
                if np.all(slack_new > 1e-10) and np.isfinite(t_new):
                    phi_new = t_new - mu * np.sum(np.log(slack_new))
                    if np.isfinite(phi_new) and phi_new <= phi_val + backtrack_c * alpha * descent:
                        success = True
                        break
            alpha *= backtrack_beta

        if not success:
            return None

        x, t = x_new, t_new

        F_true = max_group_loss(A_groups, b_groups, x)
        if not np.isfinite(F_true) or F_true > explode:
            return None

        best = min(best, F_true)
        iters.append(k)
        best_vals.append(best)

        if k % inner_iters == 0:
            mu *= tau

    return SolverResult(
        x_final=x,
        best_loss=best,
        iters=iters,
        best_values=best_vals,
        info={"t_final": t},
    )
