"""Core types and loss utilities for group distributionally robust regression.

Implements the objective from Equation (2) of Patel & Manoj (ICLR 2026):

    min_x  max_{i in [m]}  (1/n_i) ||A_i x - b_i||^2

where each group i has design matrix A_i and response vector b_i.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class SolverResult:
    """Output returned by every solver.

    Attributes:
        x_final: The parameter vector at the last iteration.
        best_loss: The best (lowest) true max-group loss seen along the trajectory.
        iters: Iteration indices at which losses were recorded (length T+1, starting at 0).
        best_values: Best-so-far true max-group loss at each recorded iteration.
        info: Solver-specific extra information (e.g. IPM barrier variable t).
    """

    x_final: np.ndarray
    best_loss: float
    iters: Optional[list[int]] = None
    best_values: Optional[list[float]] = None
    info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def group_losses(A_groups: list[np.ndarray], b_groups: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Per-group mean squared errors: ell_i(x) = (1/n_i) ||A_i x - b_i||^2."""
    return np.array([
        np.sum((A_i @ x - b_i) ** 2) / A_i.shape[0]
        for A_i, b_i in zip(A_groups, b_groups)
    ])


def max_group_loss(A_groups: list[np.ndarray], b_groups: list[np.ndarray], x: np.ndarray) -> float:
    """Worst-group loss: F(x) = max_i ell_i(x)."""
    return float(group_losses(A_groups, b_groups, x).max())


def group_loss_gradients(A_groups: list[np.ndarray], b_groups: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Gradient of each group loss: grad ell_i(x) = (2/n_i) A_i^T (A_i x - b_i).

    Returns array of shape (m, d).
    """
    grads = []
    for A_i, b_i in zip(A_groups, b_groups):
        r_i = A_i @ x - b_i
        grads.append((2.0 / A_i.shape[0]) * (A_i.T @ r_i))
    return np.array(grads)


def group_loss_hessians(A_groups: list[np.ndarray], b_groups: list[np.ndarray]) -> list[np.ndarray]:
    """Hessian of each group loss: H_i = (2/n_i) A_i^T A_i.

    These are constant (independent of x) for quadratic losses.
    """
    return [(2.0 / A_i.shape[0]) * (A_i.T @ A_i) for A_i in A_groups]


def erm_solution(A_groups: list[np.ndarray], b_groups: list[np.ndarray]) -> np.ndarray:
    """Empirical risk minimizer (minimizes average group loss) via pseudoinverse."""
    A_all = np.vstack(A_groups)
    b_all = np.concatenate(b_groups)
    return np.linalg.pinv(A_all) @ b_all
