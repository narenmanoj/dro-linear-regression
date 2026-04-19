"""Subgradient descent on the (nonsmooth) max-group loss.

Implements fixed-step and diminishing-step subgradient methods as baselines.
The subgradient at x is the gradient of the maximally-active group loss:

    g(x) = grad ell_{i*}(x),   i* = argmax_i ell_i(x).
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..problem import SolverResult, group_losses, max_group_loss


def _subgradient(A_groups, b_groups, x):
    """Subgradient of max_i ell_i(x)."""
    ell = group_losses(A_groups, b_groups, x)
    i_star = int(np.argmax(ell))
    A_i, b_i = A_groups[i_star], b_groups[i_star]
    r_i = A_i @ x - b_i
    return (2.0 / len(b_i)) * (A_i.T @ r_i)


def subgradient_fixed(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    step: float,
    T: int = 100,
    time_budget: float | None = None,
) -> SolverResult | None:
    """Fixed-step subgradient descent on the true max-loss.

    Args:
        T: Maximum iteration count.
        time_budget: If set, stop after this many wall-clock seconds.

    Returns None if the iterates diverge or become non-finite.
    """
    start = time.perf_counter()
    x = x0.copy()
    F = max_group_loss(A_groups, b_groups, x)
    best = F
    explode = 1e6 * max(F, 1.0)
    iters, best_vals, times = [0], [best], [0.0]

    for t in range(1, T + 1):
        g = _subgradient(A_groups, b_groups, x)
        if not np.all(np.isfinite(g)):
            return None

        x = x - step * g
        if not np.all(np.isfinite(x)):
            return None

        F = max_group_loss(A_groups, b_groups, x)
        if not np.isfinite(F) or F > explode:
            return None

        best = min(best, F)
        elapsed = time.perf_counter() - start
        iters.append(t)
        best_vals.append(best)
        times.append(elapsed)

        if time_budget is not None and elapsed >= time_budget:
            break

    return SolverResult(x_final=x, best_loss=best,
                         iters=iters, best_values=best_vals, times=times)


def subgradient_diminishing(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    base_step: float,
    T: int = 100,
    time_budget: float | None = None,
) -> SolverResult | None:
    """Diminishing-step subgradient descent: eta_t = base_step / sqrt(t).

    Args:
        T: Maximum iteration count.
        time_budget: If set, stop after this many wall-clock seconds.

    Returns None if the iterates diverge or become non-finite.
    """
    start = time.perf_counter()
    x = x0.copy()
    F = max_group_loss(A_groups, b_groups, x)
    best = F
    explode = 1e6 * max(F, 1.0)
    iters, best_vals, times = [0], [best], [0.0]

    for t in range(1, T + 1):
        eta = base_step / math.sqrt(t)
        g = _subgradient(A_groups, b_groups, x)
        if not np.all(np.isfinite(g)):
            return None

        x = x - eta * g
        if not np.all(np.isfinite(x)):
            return None

        F = max_group_loss(A_groups, b_groups, x)
        if not np.isfinite(F) or F > explode:
            return None

        best = min(best, F)
        elapsed = time.perf_counter() - start
        iters.append(t)
        best_vals.append(best)
        times.append(elapsed)

        if time_budget is not None and elapsed >= time_budget:
            break

    return SolverResult(x_final=x, best_loss=best,
                         iters=iters, best_values=best_vals, times=times)
