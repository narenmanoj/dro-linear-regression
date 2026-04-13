"""The Gp interpolating objective and its derivatives (Equation 4, Theorem 2).

Defines the family of objectives parameterized by p >= 2:

    F_p(x) = ( (1/m) sum_i  ell_i(x)^{p/2} )^{1/p}

where ell_i(x) = (1/n_i) ||A_i x - b_i||^2 is the per-group loss.

  - p = 2  gives the average (ERM) loss:  (1/m) sum_i ell_i(x)
  - p -> inf  recovers the robust objective: max_i ell_i(x)

For optimization we work with the pth-power (same minimizer, smoother):

    f_p(x) = sum_i  ell_i(x)^{p/2}

This is smooth for all p >= 2 and all x where ell_i > 0, unlike the
nonsmooth max-loss. The gradient and Hessian are derived by the chain rule
through ell_i^{p/2}.
"""

from __future__ import annotations

import numpy as np

from .problem import group_losses, group_loss_gradients, group_loss_hessians


def gp_value(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    p: float,
) -> float:
    """Evaluate F_p(x) = ( (1/m) sum_i ell_i(x)^{p/2} )^{1/p}.

    Returns np.inf on numerical failure.
    """
    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)):
        return np.inf
    m = len(A_groups)
    # Clamp ell >= 0 for safety (it's a squared norm, so always nonneg).
    ell = np.maximum(ell, 0.0)
    summand = np.sum(ell ** (p / 2.0))
    val = (summand / m) ** (1.0 / p)
    return float(val) if np.isfinite(val) else np.inf


def gp_power_value(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    p: float,
) -> float:
    """Evaluate the pth-power objective f_p(x) = sum_i ell_i(x)^{p/2}.

    This has the same minimizer as F_p but is smoother to work with.
    """
    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)):
        return np.inf
    ell = np.maximum(ell, 0.0)
    return float(np.sum(ell ** (p / 2.0)))


def gp_power_grad(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    p: float,
) -> np.ndarray | None:
    """Gradient of f_p(x) = sum_i ell_i(x)^{p/2}.

    grad f_p = (p/2) sum_i ell_i^{p/2 - 1} * grad ell_i

    Returns None on numerical failure.
    """
    ell = group_losses(A_groups, b_groups, x)  # (m,)
    if not np.all(np.isfinite(ell)):
        return None
    ell = np.maximum(ell, 1e-30)  # avoid 0^(negative) when p/2 - 1 < 0

    grads = group_loss_gradients(A_groups, b_groups, x)  # (m, d)
    if not np.all(np.isfinite(grads)):
        return None

    coeff = (p / 2.0) * ell ** (p / 2.0 - 1.0)  # (m,)
    grad = (coeff[:, None] * grads).sum(axis=0)

    return grad if np.all(np.isfinite(grad)) else None


def gp_power_grad_and_hessian(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    p: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Gradient and Hessian of f_p(x) = sum_i ell_i(x)^{p/2}.

    Hess f_p = (p/2) sum_i [
        (p/2 - 1) * ell_i^{p/2 - 2} * g_i g_i^T
        + ell_i^{p/2 - 1} * H_i
    ]

    where g_i = grad ell_i, H_i = (2/n_i) A_i^T A_i.

    Returns (None, None) on numerical failure.
    """
    m = len(A_groups)
    d = A_groups[0].shape[1]

    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)):
        return None, None
    ell = np.maximum(ell, 1e-30)

    grads = group_loss_gradients(A_groups, b_groups, x)  # (m, d)
    H_list = group_loss_hessians(A_groups, b_groups)       # list of (d, d)
    if not np.all(np.isfinite(grads)):
        return None, None

    half_p = p / 2.0

    # Gradient
    coeff_grad = half_p * ell ** (half_p - 1.0)
    grad = (coeff_grad[:, None] * grads).sum(axis=0)

    # Hessian
    H = np.zeros((d, d))
    for i in range(m):
        # Rank-1 curvature from the power chain rule
        if half_p != 1.0:
            H += half_p * (half_p - 1.0) * ell[i] ** (half_p - 2.0) * np.outer(grads[i], grads[i])
        # Weighted group Hessian
        H += half_p * ell[i] ** (half_p - 1.0) * H_list[i]

    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(H)):
        return None, None

    # Light regularization.
    lam = 1e-10 * max(1.0, np.linalg.norm(H, ord=2))
    H += lam * np.eye(d)

    return grad, H
