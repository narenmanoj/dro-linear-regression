"""Smoothed max-loss objective and its derivatives.

Implements the smoothing from Equation (7) of Patel & Manoj (ICLR 2026):

    f_beta,delta(x) = beta * log sum_i exp( (sqrt(delta^2 + ell_i^2) - delta) / beta )

where ell_i(x) = (1/n_i) ||A_i x - b_i||^2 is the group loss.

When delta = 0 this reduces to the standard log-sum-exp (softmax) smoothing.
The robust smoothing (delta > 0) makes the inner functions twice differentiable
at ell_i = 0, which is needed for quasi-self-concordance (Definition 2.1).
"""

from __future__ import annotations

import math

import numpy as np

from .problem import group_losses, group_loss_gradients, group_loss_hessians


def _softmax_weights(psi: np.ndarray, beta: float) -> tuple[np.ndarray, float] | None:
    """Compute softmax weights w_i = exp(psi_i / beta) / sum_j exp(psi_j / beta).

    Returns (weights, log_partition) or None on numerical failure.
    """
    z = psi / beta
    z_max = np.max(z)
    if not np.isfinite(z_max):
        return None
    exp_z = np.exp(z - z_max)
    Z = exp_z.sum()
    if Z <= 0 or not np.isfinite(Z):
        return None
    return exp_z / Z, z_max + math.log(Z)


def smooth_value(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    beta: float,
    delta: float,
) -> float:
    """Evaluate the smoothed max-loss objective f_beta,delta(x).

    Returns np.inf on numerical failure.
    """
    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)) or beta <= 0:
        return np.inf

    if delta > 0:
        s = np.sqrt(delta ** 2 + ell ** 2)
        psi = s - delta
    else:
        psi = ell

    result = _softmax_weights(psi, beta)
    if result is None:
        return np.inf
    _, log_Z = result
    val = beta * log_Z
    return float(val) if np.isfinite(val) else np.inf


def smooth_value_and_grad(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    beta: float,
    delta: float,
) -> tuple[float | None, np.ndarray | None]:
    """Smoothed max-loss value and gradient.

    The gradient is:
        grad f_beta,delta = sum_i  w_i * psi'_i * grad ell_i

    where psi'_i = ell_i / sqrt(delta^2 + ell_i^2) when delta > 0, else 1.

    Returns (None, None) on numerical failure.
    """
    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)) or beta <= 0:
        return None, None

    if delta > 0:
        s = np.sqrt(delta ** 2 + ell ** 2)
        if not np.all(np.isfinite(s)):
            return None, None
        psi = s - delta
        psi_prime = ell / np.maximum(s, 1e-16)
    else:
        psi = ell
        psi_prime = np.ones_like(ell)

    result = _softmax_weights(psi, beta)
    if result is None:
        return None, None
    w, log_Z = result

    F_smooth = float(beta * log_Z)
    if not np.isfinite(F_smooth):
        return None, None

    grads = group_loss_gradients(A_groups, b_groups, x)  # (m, d)
    if not np.all(np.isfinite(grads)):
        return None, None

    coeff = w * psi_prime  # (m,)
    grad = (coeff[:, None] * grads).sum(axis=0)
    if not np.all(np.isfinite(grad)):
        return None, None

    return F_smooth, grad


def smooth_grad_and_hessian(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x: np.ndarray,
    beta: float,
    delta: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Gradient and Hessian of the smoothed max-loss objective.

    The Hessian has three terms:
      1. Weighted average of group Hessians: sum_i w_i * psi'_i * H_i
      2. Curvature from the inner function: sum_i w_i * psi''_i * g_i g_i^T
      3. Softmax curvature: (1/beta) sum_i w_i * (v_i - v_bar)(v_i - v_bar)^T
         where v_i = psi'_i * g_i.

    Returns (None, None) on numerical failure.
    """
    m = len(A_groups)
    d = A_groups[0].shape[1]

    ell = group_losses(A_groups, b_groups, x)
    if not np.all(np.isfinite(ell)) or beta <= 0:
        return None, None

    grads = group_loss_gradients(A_groups, b_groups, x)  # (m, d)
    H_list = group_loss_hessians(A_groups, b_groups)       # list of (d, d)
    if not np.all(np.isfinite(grads)):
        return None, None

    if delta > 0:
        s = np.sqrt(delta ** 2 + ell ** 2)
        if not np.all(np.isfinite(s)):
            return None, None
        psi = s - delta
        psi_prime = ell / np.maximum(s, 1e-16)
        psi_second = delta ** 2 / np.maximum(s ** 3, 1e-16)
    else:
        psi = ell
        psi_prime = np.ones_like(ell)
        psi_second = np.zeros_like(ell)

    result = _softmax_weights(psi, beta)
    if result is None:
        return None, None
    w, _ = result

    # v_i = psi'_i * g_i
    v_arr = psi_prime[:, None] * grads  # (m, d)
    v_bar = (w[:, None] * v_arr).sum(axis=0)  # (d,)

    # Gradient
    grad = v_bar.copy()

    # Hessian: three terms
    H = np.zeros((d, d))
    for i in range(m):
        # Term 1: weighted group Hessian
        H += w[i] * psi_prime[i] * H_list[i]
        # Term 2: curvature from inner function
        H += w[i] * psi_second[i] * np.outer(grads[i], grads[i])
        # Term 3: softmax curvature
        diff = v_arr[i] - v_bar
        H += (w[i] / beta) * np.outer(diff, diff)

    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(H)):
        return None, None

    # Light regularization for numerical stability.
    lam = 1e-10 * max(1.0, np.linalg.norm(H, ord=2))
    H += lam * np.eye(d)

    return grad, H
