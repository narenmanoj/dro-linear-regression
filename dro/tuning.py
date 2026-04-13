"""Hyperparameter tuning via grid search for all solver families.

Each tune_* function runs its solver over a grid of hyperparameters and returns
the best configuration (as a dict) and its score (best true max-group loss
achieved along the trajectory).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .problem import max_group_loss
from .solvers.subgradient import subgradient_fixed, subgradient_diminishing
from .solvers.smoothed_gd import smooth_gd, smooth_heavy_ball, smooth_nesterov
from .solvers.ipm import interior_point
from .solvers.ball_oracle import ball_oracle_naive, ball_oracle_lewis


def _best_of(results: list[tuple[dict, float]]) -> tuple[dict | None, float]:
    """Pick the config with the lowest score from a list of (config, score) pairs."""
    best_cfg, best_val = None, float("inf")
    for cfg, val in results:
        if np.isfinite(val) and val < best_val - 1e-9:
            best_val = val
            best_cfg = cfg
    return best_cfg, best_val


# ---------------------------------------------------------------------------
# Subgradient
# ---------------------------------------------------------------------------

def tune_subgradient(
    A_groups, b_groups, x0,
    T: int = 100,
    step_grid: np.ndarray | None = None,
    base_step_grid: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[dict | None, float]:
    """Tune fixed and diminishing step subgradient methods."""
    if step_grid is None:
        step_grid = np.logspace(-8, -5, 7)
    if base_step_grid is None:
        base_step_grid = np.logspace(-7, -4, 7)

    results = []

    for step in step_grid:
        res = subgradient_fixed(A_groups, b_groups, x0, step=step, T=T)
        val = res.best_loss if res else float("inf")
        if verbose:
            status = f"best={val:.6e}" if res else "FAILED"
            print(f"  [subgrad fixed] step={step:.2e} -> {status}")
        if res:
            results.append(({"mode": "fixed", "step": float(step)}, val))

    for base in base_step_grid:
        res = subgradient_diminishing(A_groups, b_groups, x0, base_step=base, T=T)
        val = res.best_loss if res else float("inf")
        if verbose:
            status = f"best={val:.6e}" if res else "FAILED"
            print(f"  [subgrad dim] base={base:.2e} -> {status}")
        if res:
            results.append(({"mode": "diminishing", "base_step": float(base)}, val))

    return _best_of(results)


# ---------------------------------------------------------------------------
# Smoothed gradient methods
# ---------------------------------------------------------------------------

def tune_smooth(
    A_groups, b_groups, x0,
    T: int = 100,
    verbose: bool = False,
) -> tuple[dict | None, float]:
    """Tune beta, delta, step, momentum across GD / Heavy-Ball / Nesterov."""
    F_init = max_group_loss(A_groups, b_groups, x0)

    beta_rel = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    delta_rel = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    step_grid = np.logspace(-6, -2, 7)
    momentum_grid = [0.5, 0.7, 0.9]

    results = []

    for br in beta_rel:
        beta = br * F_init
        if beta <= 0:
            continue
        for dr in delta_rel:
            delta = dr * F_init

            # Plain GD
            for step in step_grid:
                res = smooth_gd(A_groups, b_groups, x0, beta=beta, delta=delta, step=step, T=T)
                if res:
                    results.append(({
                        "variant": "gd", "beta": float(beta),
                        "delta": float(delta), "step": float(step),
                    }, res.best_loss))

            # Heavy-Ball and Nesterov
            for step in step_grid:
                for mom in momentum_grid:
                    for variant, fn in [("heavy_ball", smooth_heavy_ball), ("nesterov", smooth_nesterov)]:
                        res = fn(A_groups, b_groups, x0,
                                 beta=beta, delta=delta, step=step, momentum=mom, T=T)
                        if res:
                            results.append(({
                                "variant": variant, "beta": float(beta),
                                "delta": float(delta), "step": float(step),
                                "momentum": float(mom),
                            }, res.best_loss))

    cfg, val = _best_of(results)
    if verbose and cfg:
        print(f"  [smooth] best: {cfg['variant']}, val={val:.6e}")
    return cfg, val


# ---------------------------------------------------------------------------
# Interior-point method
# ---------------------------------------------------------------------------

def tune_ipm(
    A_groups, b_groups, x0,
    max_newton_steps: int = 100,
    verbose: bool = False,
) -> tuple[dict | None, float]:
    """Tune (mu0, tau, inner_iters) for the log-barrier IPM."""
    mu0_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    tau_list = [0.99, 0.95, 0.9, 0.7, 0.5]
    inner_list = [3, 5, 10, 20]

    results = []

    for mu0 in mu0_list:
        for tau in tau_list:
            for inner in inner_list:
                res = interior_point(
                    A_groups, b_groups, x0,
                    mu0=mu0, tau=tau, inner_iters=inner,
                    max_newton_steps=max_newton_steps,
                )
                val = res.best_loss if res else float("inf")
                if verbose:
                    status = f"best={val:.6e}" if res else "FAILED"
                    print(f"  [ipm] mu0={mu0:.1e},tau={tau:.2f},inner={inner} -> {status}")
                if res:
                    results.append(({"mu0": mu0, "tau": tau, "inner_iters": inner}, val))

    return _best_of(results)


# ---------------------------------------------------------------------------
# Ball-oracle (naive)
# ---------------------------------------------------------------------------

def tune_ball_oracle_naive(
    A_groups, b_groups, x0,
    n_outer: int = 100,
    verbose: bool = False,
) -> tuple[dict | None, float]:
    """Tune (beta, delta, R0, R_decay, max_newton_steps) for naive ball-oracle."""
    F_init = max_group_loss(A_groups, b_groups, x0)
    base = max(F_init, 1.0)
    xnorm = max(float(np.linalg.norm(x0)), 1.0)

    beta_grid = base * np.logspace(-3, 0, 5)
    delta_mults = [0.0, 1e-3, 1e-2]
    R0_grid = [f * xnorm for f in [0.25, 0.5, 1.0, 2.0]]
    R_decay_grid = [0.5, 0.7, 0.9, 0.99]
    newton_grid = [3, 5, 8]

    results = []

    for beta in beta_grid:
        for dm in delta_mults:
            delta = dm * beta
            for R0 in R0_grid:
                for R_decay in R_decay_grid:
                    for msteps in newton_grid:
                        res = ball_oracle_naive(
                            A_groups, b_groups, x0,
                            beta=beta, delta=delta, R0=R0,
                            n_outer=n_outer, R_decay=R_decay,
                            max_newton_steps=msteps,
                        )
                        if res is None:
                            continue

                        vals = np.asarray(res.best_values, float)
                        # Discard configs where all improvement is at k=1.
                        if vals.size >= 2 and vals[-1] >= 0.99 * vals[1]:
                            continue

                        results.append(({
                            "beta": float(beta), "delta": float(delta),
                            "R0": float(R0), "R_decay": float(R_decay),
                            "n_outer": n_outer, "max_newton_steps": msteps,
                        }, res.best_loss))

    return _best_of(results)


# ---------------------------------------------------------------------------
# Ball-oracle (Lewis)
# ---------------------------------------------------------------------------

def tune_ball_oracle_lewis(
    A_groups, b_groups, x0, L,
    n_outer: int = 100,
    verbose: bool = False,
) -> tuple[dict | None, float]:
    """Tune (beta, delta, R0, R_decay, max_newton_steps) for Lewis ball-oracle."""
    F_init = max_group_loss(A_groups, b_groups, x0)
    base = max(F_init, 1.0)
    xnorm_lewis = max(float(np.linalg.norm(L @ x0)), 1.0)

    beta_grid = base * np.logspace(-3, 0, 5)
    delta_mults = [0.0, 1e-3, 1e-2]
    R0_grid = [f * xnorm_lewis for f in [0.25, 0.5, 1.0, 2.0]]
    R_decay_grid = [0.9, 0.99]
    newton_grid = [8]

    results = []

    for beta in beta_grid:
        for dm in delta_mults:
            delta = dm * beta
            for R0 in R0_grid:
                for R_decay in R_decay_grid:
                    for msteps in newton_grid:
                        res = ball_oracle_lewis(
                            A_groups, b_groups, x0, L=L,
                            beta=beta, delta=delta, R0=R0,
                            n_outer=n_outer, R_decay=R_decay,
                            max_newton_steps=msteps,
                        )
                        if res is None:
                            continue

                        vals = np.asarray(res.best_values, float)
                        if vals.size >= 2 and vals[-1] >= 0.99 * vals[1]:
                            continue

                        results.append(({
                            "beta": float(beta), "delta": float(delta),
                            "R0": float(R0), "R_decay": float(R_decay),
                            "n_outer": n_outer, "max_newton_steps": msteps,
                        }, res.best_loss))

    return _best_of(results)


# ---------------------------------------------------------------------------
# Convenience: tune all methods at once
# ---------------------------------------------------------------------------

def tune_all(
    A_groups, b_groups, x0,
    L=None,
    T: int = 100,
    verbose: bool = False,
) -> dict[str, tuple[dict | None, float]]:
    """Tune all solver families and return a dict of {name: (best_cfg, best_val)}.

    Args:
        L: Cholesky factor from build_lewis_geometry (needed for Lewis ball-oracle).
           If None, skips Lewis ball-oracle tuning.
    """
    results = {}

    if verbose:
        print("Tuning subgradient...")
    results["subgradient"] = tune_subgradient(A_groups, b_groups, x0, T=T, verbose=verbose)

    if verbose:
        print("Tuning smoothed gradient methods...")
    results["smooth"] = tune_smooth(A_groups, b_groups, x0, T=T, verbose=verbose)

    if verbose:
        print("Tuning IPM...")
    results["ipm"] = tune_ipm(A_groups, b_groups, x0, max_newton_steps=T, verbose=verbose)

    if verbose:
        print("Tuning naive ball-oracle...")
    results["ball_oracle_naive"] = tune_ball_oracle_naive(
        A_groups, b_groups, x0, n_outer=T, verbose=verbose)

    if L is not None:
        if verbose:
            print("Tuning Lewis ball-oracle...")
        results["ball_oracle_lewis"] = tune_ball_oracle_lewis(
            A_groups, b_groups, x0, L=L, n_outer=T, verbose=verbose)

    return results
