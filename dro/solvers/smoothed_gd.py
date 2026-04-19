"""Smoothed gradient methods for group DRO.

First-order baselines that optimize the smoothed objective f_beta,delta(x)
(Equation 7 of Patel & Manoj, ICLR 2026) using:
  - Plain gradient descent
  - Heavy-Ball (Polyak) momentum
  - Nesterov accelerated gradient

All methods track the true (nonsmooth) max-group loss as the evaluation metric.
"""

from __future__ import annotations

import time

import numpy as np

from ..problem import SolverResult, max_group_loss
from ..smoothing import smooth_value_and_grad


def _check_and_record(F_true, best, explode, iters, best_vals, times, t, elapsed):
    """Check for divergence and record progress. Returns (new_best, ok)."""
    if not np.isfinite(F_true) or F_true > explode:
        return best, False
    best = min(best, F_true)
    iters.append(t)
    best_vals.append(best)
    times.append(elapsed)
    return best, True


def smooth_gd(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    beta: float,
    delta: float,
    step: float,
    T: int = 100,
    time_budget: float | None = None,
) -> SolverResult | None:
    """Plain gradient descent on the smoothed objective.

    Returns None on divergence or numerical failure.
    """
    start = time.perf_counter()
    x = x0.copy()
    F_init = max_group_loss(A_groups, b_groups, x)
    best = F_init
    explode = 1e6 * max(F_init, 1.0)
    iters, best_vals, times = [0], [best], [0.0]

    for t in range(1, T + 1):
        _, g = smooth_value_and_grad(A_groups, b_groups, x, beta, delta)
        if g is None:
            return None

        x = x - step * g
        if not np.all(np.isfinite(x)):
            return None

        F_true = max_group_loss(A_groups, b_groups, x)
        elapsed = time.perf_counter() - start
        best, ok = _check_and_record(F_true, best, explode, iters, best_vals, times, t, elapsed)
        if not ok:
            return None

        if time_budget is not None and elapsed >= time_budget:
            break

    return SolverResult(x_final=x, best_loss=best,
                         iters=iters, best_values=best_vals, times=times)


def smooth_heavy_ball(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    beta: float,
    delta: float,
    step: float,
    momentum: float,
    T: int = 100,
    time_budget: float | None = None,
) -> SolverResult | None:
    """Heavy-Ball momentum on the smoothed objective.

    Update: v <- momentum * v + step * grad;  x <- x - v.

    Returns None on divergence or numerical failure.
    """
    start = time.perf_counter()
    x = x0.copy()
    v = np.zeros_like(x)
    F_init = max_group_loss(A_groups, b_groups, x)
    best = F_init
    explode = 1e6 * max(F_init, 1.0)
    iters, best_vals, times = [0], [best], [0.0]

    for t in range(1, T + 1):
        _, g = smooth_value_and_grad(A_groups, b_groups, x, beta, delta)
        if g is None:
            return None

        v = momentum * v + step * g
        x = x - v
        if not np.all(np.isfinite(x)):
            return None

        F_true = max_group_loss(A_groups, b_groups, x)
        elapsed = time.perf_counter() - start
        best, ok = _check_and_record(F_true, best, explode, iters, best_vals, times, t, elapsed)
        if not ok:
            return None

        if time_budget is not None and elapsed >= time_budget:
            break

    return SolverResult(x_final=x, best_loss=best,
                         iters=iters, best_values=best_vals, times=times)


def smooth_nesterov(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    beta: float,
    delta: float,
    step: float,
    momentum: float,
    T: int = 100,
    time_budget: float | None = None,
) -> SolverResult | None:
    """Nesterov accelerated gradient on the smoothed objective.

    Update: y <- x - momentum * v;  g <- grad f(y);
            v <- momentum * v + step * g;  x <- x - v.

    Returns None on divergence or numerical failure.
    """
    start = time.perf_counter()
    x = x0.copy()
    v = np.zeros_like(x)
    F_init = max_group_loss(A_groups, b_groups, x)
    best = F_init
    explode = 1e6 * max(F_init, 1.0)
    iters, best_vals, times = [0], [best], [0.0]

    for t in range(1, T + 1):
        y = x - momentum * v
        _, g = smooth_value_and_grad(A_groups, b_groups, y, beta, delta)
        if g is None:
            return None

        v = momentum * v + step * g
        x = x - v
        if not np.all(np.isfinite(x)):
            return None

        F_true = max_group_loss(A_groups, b_groups, x)
        elapsed = time.perf_counter() - start
        best, ok = _check_and_record(F_true, best, explode, iters, best_vals, times, t, elapsed)
        if not ok:
            return None

        if time_budget is not None and elapsed >= time_budget:
            break

    return SolverResult(x_final=x, best_loss=best,
                         iters=iters, best_values=best_vals, times=times)
