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


def summarize(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    solutions: dict[str, np.ndarray],
    group_names: list[str] | None = None,
) -> str:
    """Print a summary table of per-group losses and aggregate metrics.

    Args:
        solutions: {label: x} mapping solver names to parameter vectors.
        group_names: Optional readable names for each group.

    Returns:
        The formatted summary string (also printed to stdout).
    """
    m = len(A_groups)
    labels = list(solutions.keys())

    if group_names is None:
        group_names = [f"Group {i}" for i in range(m)]

    # Compute per-group losses for each solution.
    all_losses = {}
    for label, x in solutions.items():
        all_losses[label] = group_losses(A_groups, b_groups, x)

    # Column widths.
    name_w = max(len(g) for g in group_names)
    name_w = max(name_w, 5)  # at least "Group"
    n_w = 6  # for sample count
    col_w = max(max(len(l) for l in labels), 10)

    lines = []

    # Header.
    header = f"{'Group':<{name_w}}  {'n':>{n_w}}"
    for label in labels:
        header += f"  {label:>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Per-group rows.
    for i in range(m):
        n_i = A_groups[i].shape[0]
        row = f"{group_names[i]:<{name_w}}  {n_i:>{n_w}}"
        for label in labels:
            row += f"  {all_losses[label][i]:>{col_w}.4f}"
        lines.append(row)

    lines.append("-" * len(header))

    # Aggregate rows.
    for agg_name, agg_fn in [("Max", np.max), ("Mean", np.mean), ("Median", np.median), ("Min", np.min), ("Std", np.std)]:
        row = f"{agg_name:<{name_w}}  {'':>{n_w}}"
        for label in labels:
            row += f"  {agg_fn(all_losses[label]):>{col_w}.4f}"
        lines.append(row)

    # Max/mean ratio (measures how much worse the worst group is vs average).
    row = f"{'Max/Mean':<{name_w}}  {'':>{n_w}}"
    for label in labels:
        losses = all_losses[label]
        ratio = np.max(losses) / max(np.mean(losses), 1e-16)
        row += f"  {ratio:>{col_w}.4f}"
    lines.append(row)

    text = "\n".join(lines)
    print(text)
    return text
