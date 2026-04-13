"""Plotting utilities for DRO experiments.

Generates:
  - Iteration-based convergence plots (best-so-far max-loss - OPT vs iterations)
  - Wall-clock time vs accuracy plots
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from .problem import SolverResult, max_group_loss


def _monotone_best(vals):
    """Enforce monotone decrease (running minimum)."""
    vals = np.asarray(vals, float).copy()
    for i in range(1, len(vals)):
        vals[i] = min(vals[i], vals[i - 1])
    return vals


def plot_convergence(
    curves: dict[str, SolverResult],
    F_opt: float | None = None,
    F_erm: float | None = None,
    title: str = "Convergence on Max-Loss Objective",
    save_path: str | None = None,
):
    """Plot iteration-based convergence curves (log scale x-axis).

    Args:
        curves: {label: SolverResult} for each method.
        F_opt: Optimal value to subtract (if None, plots raw losses).
        F_erm: ERM baseline value (shown as horizontal dashed line).
        save_path: If provided, saves the figure to this path.
    """
    baseline = F_opt if (F_opt is not None and np.isfinite(F_opt)) else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))

    if F_erm is not None and np.isfinite(F_erm):
        gap = max(F_erm - baseline, 1e-16)
        ax.axhline(y=gap, linestyle="--", linewidth=1.5, color="gray",
                    label=f"ERM - OPT ({gap:.2e})")

    for label, result in curves.items():
        if result.iters is None or result.best_values is None:
            continue
        iters = np.asarray(result.iters, float)
        vals = np.asarray(result.best_values, float)

        mask = np.isfinite(iters) & np.isfinite(vals)
        iters, vals = iters[mask], vals[mask]
        if iters.size == 0:
            continue

        vals_shifted = np.maximum(vals - baseline, 1e-16)
        x_plot = iters + 1.0  # shift for log scale

        marker = "o" if len(x_plot) <= 40 else None
        ax.plot(x_plot, vals_shifted, label=label, linewidth=2.0,
                marker=marker, markersize=4 if marker else 0)

    ax.set_xscale("log")
    ax.set_xlim(left=1.0, right=max(iters.max() + 1, 101.0) if len(curves) > 0 else 101.0)
    ax.set_xlabel("Iterations (log scale)")
    ax.set_ylabel("Best-so-far (max-group loss - OPT)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def _scaled_times(iters, total_runtime):
    """Map iteration indices to proportional wall-clock times."""
    iters = np.asarray(iters, float)
    T_max = max(float(iters[-1]), 1.0) if iters.size > 0 else 1.0
    return (iters / T_max) * max(total_runtime, 1e-9)


def timed_run(solver_fn, *args, **kwargs) -> tuple[SolverResult | None, float]:
    """Run a solver and return (result, wall_clock_seconds)."""
    start = time.perf_counter()
    result = solver_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def plot_time_vs_accuracy(
    timed_curves: dict[str, tuple[SolverResult, float]],
    F_opt: float | None = None,
    F_erm: float | None = None,
    title: str = "Time vs Accuracy",
    save_path: str | None = None,
):
    """Plot wall-clock time vs best-so-far accuracy (log scale x-axis).

    Args:
        timed_curves: {label: (SolverResult, total_runtime_seconds)}.
        F_opt: Optimal value to subtract.
        F_erm: ERM baseline.
        save_path: If provided, saves the figure.
    """
    baseline = F_opt if (F_opt is not None and np.isfinite(F_opt)) else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))

    if F_erm is not None and np.isfinite(F_erm):
        gap = max(F_erm - baseline, 1e-16)
        ax.axhline(y=gap, linestyle="--", linewidth=1.5, color="gray",
                    label=f"ERM - OPT ({gap:.2e})")

    for label, (result, runtime) in timed_curves.items():
        if result is None or result.iters is None or result.best_values is None:
            continue

        times = _scaled_times(result.iters, runtime)
        vals = _monotone_best(result.best_values)

        mask = np.isfinite(times) & np.isfinite(vals)
        times, vals = times[mask], vals[mask]
        if times.size == 0:
            continue

        gaps = np.maximum(vals - baseline, 1e-16)
        times = np.maximum(times, 1e-7)

        marker = "o" if len(times) <= 40 else None
        ax.plot(times, gaps, label=label, linewidth=2.0,
                marker=marker, markersize=4 if marker else 0)

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_ylabel("Best-so-far (max-group loss - OPT)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_interpolation_path(
    path: list[dict],
    F_opt: float | None = None,
    F_erm: float | None = None,
    title: str = "Utility-Robustness Tradeoff (Gp Interpolation)",
    save_path: str | None = None,
):
    """Plot the interpolation path from ERM (p=2) to DRO (p->inf).

    Shows max-group loss and average loss as functions of p, illustrating
    the smooth tradeoff between utility and robustness (Equation 4, Theorem 2).

    Args:
        path: Output of solvers.interpolation_path().
        F_opt: Robust optimum (horizontal line).
        F_erm: ERM max-group loss (horizontal line).
        save_path: If provided, saves the figure.
    """
    valid = [r for r in path if np.isfinite(r["max_loss"])]
    if not valid:
        print("No valid interpolation results to plot.")
        return

    ps = np.array([r["p"] for r in valid])
    max_losses = np.array([r["max_loss"] for r in valid])
    avg_losses = np.array([r["avg_loss"] for r in valid])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ps, max_losses, "o-", linewidth=2.0, markersize=6,
            label="Max-group loss (worst case)", color="tab:red")
    ax.plot(ps, avg_losses, "s-", linewidth=2.0, markersize=6,
            label="Avg-group loss (utility)", color="tab:blue")

    if F_opt is not None and np.isfinite(F_opt):
        ax.axhline(y=F_opt, linestyle="--", linewidth=1.5, color="green",
                    label=f"Robust OPT ({F_opt:.2e})")

    if F_erm is not None and np.isfinite(F_erm):
        ax.axhline(y=F_erm, linestyle=":", linewidth=1.5, color="gray",
                    label=f"ERM max-loss ({F_erm:.2e})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("p (log\u2082 scale)")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
