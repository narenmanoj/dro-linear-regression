"""Solvers for the Gp interpolating objective (Equation 4, Theorem 2).

Minimizes F_p(x) = ( (1/m) sum_i ell_i(x)^{p/2} )^{1/p} for 2 <= p < inf,
which smoothly interpolates between ERM (p=2) and robust DRO (p->inf).

Three solver variants mirror the structure of the p=inf (max-loss) solvers:

  1. gp_newton: Plain damped Newton on the pth-power objective f_p.
  2. gp_ball_oracle_naive: Ball-oracle with Euclidean trust regions on f_p.
  3. gp_ball_oracle_lewis: Ball-oracle with Lewis-weighted trust regions on f_p
     (Algorithm 5). Achieves the min{rank(A), m}^{(p-2)/(3p-2)} rate of Theorem 2.

For the theoretical algorithm (Algorithm 5), the proximal subproblems (Eq. 8)
are solved via Newton steps, and the outer loop follows the accelerated
proximal scheme of Carmon et al. (2022). The practical implementations here
use direct Newton and ball-oracle approaches which are simpler but match
the paper's experimental setup.
"""

from __future__ import annotations

import numpy as np

from ..problem import SolverResult, max_group_loss
from ..gp_objective import gp_value, gp_power_value, gp_power_grad, gp_power_grad_and_hessian
from ..lewis_weights import project_onto_lewis_ball


# ---------------------------------------------------------------------------
# 1. Plain damped Newton on f_p
# ---------------------------------------------------------------------------

def gp_newton(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    p: float,
    max_steps: int = 100,
    tol: float = 1e-10,
    backtrack_beta: float = 0.5,
    backtrack_c: float = 1e-4,
) -> SolverResult | None:
    """Damped Newton's method on f_p(x) = sum_i ell_i(x)^{p/2}.

    Since f_p is smooth and convex for p >= 2, Newton converges quadratically
    near the optimum. Uses backtracking line search for global convergence.

    Returns None on numerical failure.
    """
    x = x0.copy()
    d = len(x)

    fp = gp_power_value(A_groups, b_groups, x, p)
    if not np.isfinite(fp):
        return None

    F_true = max_group_loss(A_groups, b_groups, x)
    Fp = gp_value(A_groups, b_groups, x, p)
    best_Fp = Fp
    iters, best_vals = [0], [Fp]

    for k in range(1, max_steps + 1):
        g, H = gp_power_grad_and_hessian(A_groups, b_groups, x, p)
        if g is None or H is None:
            return None

        g_norm = np.linalg.norm(g)
        if g_norm <= tol * max(1.0, abs(fp)):
            break

        # Newton direction with regularization.
        lam = 1e-8 * max(1.0, g_norm)
        try:
            dx = -np.linalg.solve(H + lam * np.eye(d), g)
        except np.linalg.LinAlgError:
            return None

        if g @ dx > 0:
            dx = -dx

        # Backtracking line search on f_p.
        alpha = 1.0
        descent = g @ dx
        for _ in range(40):
            x_trial = x + alpha * dx
            fp_trial = gp_power_value(A_groups, b_groups, x_trial, p)
            if np.isfinite(fp_trial) and fp_trial <= fp + backtrack_c * alpha * descent:
                x, fp = x_trial, fp_trial
                break
            alpha *= backtrack_beta
        else:
            break  # line search failed

        Fp = gp_value(A_groups, b_groups, x, p)
        best_Fp = min(best_Fp, Fp)
        iters.append(k)
        best_vals.append(best_Fp)

    return SolverResult(
        x_final=x, best_loss=best_Fp,
        iters=iters, best_values=best_vals,
        info={"p": p, "method": "newton"},
    )


# ---------------------------------------------------------------------------
# 2. Ball-oracle on f_p (shared inner Newton solver)
# ---------------------------------------------------------------------------

def _newton_ball_gp(
    A_groups, b_groups, center, R, p,
    max_newton_steps, project_fn,
    backtrack_beta=0.5, backtrack_c=1e-4, tol=1e-6,
):
    """Damped Newton inside a trust-region ball, minimizing f_p."""
    x = center.copy()
    d = len(x)
    fp = gp_power_value(A_groups, b_groups, x, p)
    if not np.isfinite(fp):
        return None

    for k in range(1, max_newton_steps + 1):
        g, H = gp_power_grad_and_hessian(A_groups, b_groups, x, p)
        if g is None or H is None:
            return None

        g_norm = np.linalg.norm(g)
        if g_norm <= tol * max(1.0, abs(fp)):
            return x

        lam = 1e-8 * max(1.0, g_norm)
        try:
            dx = -np.linalg.solve(H + lam * np.eye(d), g)
        except np.linalg.LinAlgError:
            return x

        if g @ dx > 0:
            dx = -dx

        alpha = 1.0
        descent = g @ dx
        for _ in range(40):
            x_trial = project_fn(x + alpha * dx)
            fp_trial = gp_power_value(A_groups, b_groups, x_trial, p)
            if np.isfinite(fp_trial) and fp_trial <= fp + backtrack_c * alpha * descent:
                x, fp = x_trial, fp_trial
                break
            alpha *= backtrack_beta
        else:
            return x

    return x


def _project_euclidean(center, R, x_trial):
    diff = x_trial - center
    nrm = np.linalg.norm(diff)
    if nrm <= R:
        return x_trial
    return center + (R / max(nrm, 1e-16)) * diff


def _run_gp_ball_oracle(
    A_groups, b_groups, x0, p,
    R0, n_outer, R_decay, max_newton_steps, project_fn_factory,
):
    """Shared outer loop for Gp ball-oracle methods."""
    x = x0.copy()
    center = x0.copy()
    R = float(R0)

    Fp_init = gp_value(A_groups, b_groups, x, p)
    if not np.isfinite(Fp_init):
        return None
    explode = 1e6 * max(Fp_init, 1.0)

    best = Fp_init
    iters, best_vals = [0], [best]

    for k in range(1, n_outer + 1):
        project_fn = project_fn_factory(center, R)
        x_new = _newton_ball_gp(
            A_groups, b_groups, center, R, p,
            max_newton_steps, project_fn,
        )
        if x_new is None or not np.all(np.isfinite(x_new)):
            return None

        x = x_new
        center = x.copy()

        Fp = gp_value(A_groups, b_groups, x, p)
        if not np.isfinite(Fp) or Fp > explode:
            return None

        best = min(best, Fp)
        iters.append(k)
        best_vals.append(best)

        R *= R_decay
        if R <= 0:
            break

    return SolverResult(
        x_final=x, best_loss=best,
        iters=iters, best_values=best_vals,
        info={"p": p, "method": "ball_oracle"},
    )


def gp_ball_oracle_naive(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    p: float,
    R0: float,
    n_outer: int = 100,
    R_decay: float = 0.9,
    max_newton_steps: int = 5,
) -> SolverResult | None:
    """Ball-oracle method for Gp objective with Euclidean geometry.

    Uses trust regions ||x - center||_2 <= R, giving m^{(p-2)/(3p-2)} rate.
    """
    def factory(center, R):
        return lambda x_trial: _project_euclidean(center, R, x_trial)

    return _run_gp_ball_oracle(
        A_groups, b_groups, x0, p,
        R0, n_outer, R_decay, max_newton_steps, factory,
    )


def gp_ball_oracle_lewis(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    L: np.ndarray,
    p: float,
    R0: float,
    n_outer: int = 100,
    R_decay: float = 0.9,
    max_newton_steps: int = 5,
) -> SolverResult | None:
    """Ball-oracle method for Gp objective with Lewis geometry (Algorithm 5).

    Uses trust regions ||x - center||_M <= R where M = L^T L from block Lewis
    weights computed for the Gp norm. Achieves min{rank(A), m}^{(p-2)/(3p-2)}
    iteration complexity (Theorem 2).

    Args:
        L: Cholesky factor from lewis_weights.build_lewis_geometry(p=p).
           Note: for best results, Lewis weights should be computed with the
           same p as the objective.
    """
    def factory(center, R):
        return lambda x_trial: project_onto_lewis_ball(x_trial, center, R, L)

    return _run_gp_ball_oracle(
        A_groups, b_groups, x0, p,
        R0, n_outer, R_decay, max_newton_steps, factory,
    )


# ---------------------------------------------------------------------------
# 3. Interpolation path: sweep over p to trace utility-robustness tradeoff
# ---------------------------------------------------------------------------

def interpolation_path(
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    x0: np.ndarray,
    p_values: list[float] | None = None,
    max_steps: int = 200,
    L: np.ndarray | None = None,
) -> list[dict]:
    """Sweep over p values to trace the Pareto frontier between ERM and DRO.

    For each p, solves the Gp objective using Newton's method (or ball-oracle
    with Lewis geometry if L is provided) and records both the Gp value and
    the max-group loss at the solution.

    Args:
        x0: Warm start (typically ERM solution).
        p_values: List of p values to try. Defaults to a log-spaced range
                  from p=2 (ERM) through large p (near-robust).
        max_steps: Newton iteration budget per p.
        L: If provided, uses Lewis ball-oracle instead of plain Newton.

    Returns:
        List of dicts, each with keys:
            "p": the p value
            "x": the solution
            "gp_value": F_p(x) at the solution
            "max_loss": max_i ell_i(x) at the solution
            "avg_loss": (1/m) sum_i ell_i(x) at the solution
    """
    from ..problem import group_losses as _group_losses

    if p_values is None:
        # Log-spaced from p=2 to p=128 (near-robust for typical m).
        p_values = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0]

    m = len(A_groups)
    results = []

    # Warm-start each p from the previous solution for faster convergence.
    x_prev = x0.copy()

    for p in sorted(p_values):
        if L is not None:
            # Use ball-oracle with Lewis geometry.
            xnorm = max(float(np.linalg.norm(L @ x_prev)), 1.0)
            res = gp_ball_oracle_lewis(
                A_groups, b_groups, x_prev, L=L, p=p,
                R0=xnorm, n_outer=max_steps, R_decay=0.95,
                max_newton_steps=8,
            )
        else:
            res = gp_newton(
                A_groups, b_groups, x_prev, p=p, max_steps=max_steps,
            )

        if res is None:
            # Fall back to plain Newton if ball-oracle fails.
            res = gp_newton(A_groups, b_groups, x_prev, p=p, max_steps=max_steps)

        if res is not None:
            x_sol = res.x_final
            ell = _group_losses(A_groups, b_groups, x_sol)
            results.append({
                "p": p,
                "x": x_sol,
                "gp_value": gp_value(A_groups, b_groups, x_sol, p),
                "max_loss": float(ell.max()),
                "avg_loss": float(ell.mean()),
            })
            x_prev = x_sol.copy()
        else:
            results.append({
                "p": p, "x": None,
                "gp_value": np.inf, "max_loss": np.inf, "avg_loss": np.inf,
            })

    return results
