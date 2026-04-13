"""Exact solver for the group DRO objective via CVXPY.

Solves the epigraph reformulation (Appendix F.2):

    min_{x,t} t   s.t.  (1/n_i) ||A_i x - b_i||^2 <= t  for all i.
"""

from __future__ import annotations

import numpy as np

from ..problem import SolverResult


def solve_exact(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    verbose: bool = False,
    solver: str | None = None,
) -> SolverResult:
    """Compute the exact robust optimum via convex programming.

    Requires cvxpy to be installed.

    Returns a SolverResult with iters=[0], best_values=[OPT].

    Raises:
        RuntimeError: If cvxpy is unavailable or the solve fails.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise RuntimeError("cvxpy not installed. Install with: pip install cvxpy")

    d = A_groups[0].shape[1]
    x = cp.Variable(d)
    t = cp.Variable()

    constraints = []
    for A_i, b_i in zip(A_groups, b_groups):
        n_i = A_i.shape[0]
        r_i = A_i @ x - b_i
        constraints.append((1.0 / n_i) * cp.sum_squares(r_i) <= t)

    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY solve status: {prob.status}")

    x_star = np.asarray(x.value, dtype=float).ravel()
    t_star = float(t.value)
    if x_star is None or not np.isfinite(t_star):
        raise RuntimeError("CVXPY did not return a valid solution.")

    return SolverResult(
        x_final=x_star,
        best_loss=t_star,
        iters=[0],
        best_values=[t_star],
    )
