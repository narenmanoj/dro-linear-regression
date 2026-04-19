"""Plotting utilities for DRO experiments.

Generates:
  - Iteration-based convergence plots (best-so-far max-loss - OPT vs iterations)
  - Wall-clock time vs accuracy plots
  - Interpolation path plots (Gp tradeoff)

Each convergence plot supports both log-scale and linear-scale x-axes via
the `log_scale` parameter.
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
    log_scale: bool = True,
):
    """Plot iteration-based convergence curves.

    Args:
        curves: {label: SolverResult} for each method.
        F_opt: Optimal value to subtract (if None, plots raw losses).
        F_erm: ERM baseline value (shown as horizontal dashed line).
        save_path: If provided, saves the figure to this path.
        log_scale: If True, use log scale on x-axis. If False, use linear.
    """
    baseline = F_opt if (F_opt is not None and np.isfinite(F_opt)) else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))

    if F_erm is not None and np.isfinite(F_erm):
        gap = max(F_erm - baseline, 1e-16)
        ax.axhline(y=gap, linestyle="--", linewidth=1.5, color="gray",
                    label=f"ERM - OPT ({gap:.2e})")

    max_iter = 0
    for label, result in curves.items():
        if result.iters is None or result.best_values is None:
            continue
        iters = np.asarray(result.iters, float)
        vals = np.asarray(result.best_values, float)

        mask = np.isfinite(iters) & np.isfinite(vals)
        iters, vals = iters[mask], vals[mask]
        if iters.size == 0:
            continue

        max_iter = max(max_iter, iters.max())
        vals_shifted = np.maximum(vals - baseline, 1e-16)

        if log_scale:
            x_plot = iters + 1.0  # shift so t=0 maps to x=1
        else:
            x_plot = iters

        marker = "o" if len(x_plot) <= 40 else None
        ax.plot(x_plot, vals_shifted, label=label, linewidth=2.0,
                marker=marker, markersize=4 if marker else 0)

    scale_label = "log" if log_scale else "linear"
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlim(left=1.0, right=max(max_iter + 1, 101.0))
    else:
        ax.set_xlim(left=0, right=max(max_iter, 100))

    ax.set_xlabel(f"Iterations ({scale_label} scale)")
    ax.set_ylabel("Best-so-far (max-group loss - OPT)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def timed_run(solver_fn, *args, **kwargs) -> tuple[SolverResult | None, float]:
    """Run a solver and return (result, wall_clock_seconds).

    Kept for backward compatibility; the solvers now record per-iteration
    timestamps in result.times directly, so this wrapper is rarely needed.
    """
    start = time.perf_counter()
    result = solver_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def plot_time_vs_accuracy(
    curves: dict[str, SolverResult],
    F_opt: float | None = None,
    F_erm: float | None = None,
    title: str = "Time vs Accuracy",
    save_path: str | None = None,
    log_scale: bool = True,
):
    """Plot wall-clock time vs best-so-far accuracy.

    Uses the per-iteration `times` field recorded by each solver.

    Args:
        curves: {label: SolverResult} with result.times populated.
        F_opt: Optimal value to subtract.
        F_erm: ERM baseline.
        save_path: If provided, saves the figure.
        log_scale: If True, use log scale on x-axis. If False, use linear.
    """
    baseline = F_opt if (F_opt is not None and np.isfinite(F_opt)) else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))

    if F_erm is not None and np.isfinite(F_erm):
        gap = max(F_erm - baseline, 1e-16)
        ax.axhline(y=gap, linestyle="--", linewidth=1.5, color="gray",
                    label=f"ERM - OPT ({gap:.2e})")

    for label, result in curves.items():
        if result is None or result.times is None or result.best_values is None:
            continue

        times = np.asarray(result.times, float)
        vals = _monotone_best(result.best_values)

        mask = np.isfinite(times) & np.isfinite(vals)
        times, vals = times[mask], vals[mask]
        if times.size == 0:
            continue

        gaps = np.maximum(vals - baseline, 1e-16)
        if log_scale:
            times = np.maximum(times, 1e-7)

        marker = "o" if len(times) <= 40 else None
        ax.plot(times, gaps, label=label, linewidth=2.0,
                marker=marker, markersize=4 if marker else 0)

    scale_label = "log" if log_scale else "linear"
    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(f"Runtime (seconds, {scale_label} scale)")
    ax.set_ylabel("Best-so-far (max-group loss - OPT)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_scaling(
    scaling_data: dict[str, dict[int, float]],
    ylabel: str = "Iterations to target accuracy",
    title: str = "Scaling with number of groups m",
    save_path: str | None = None,
    log_scale: bool = True,
    reference_slopes: dict[str, float] | None = None,
):
    """Plot some metric vs number of groups m, one curve per solver.

    Args:
        scaling_data: {solver_label: {m: metric_value}}.
        ylabel: Y-axis label (typically "iterations" or "runtime (s)").
        title: Plot title.
        save_path: If provided, save the figure.
        log_scale: If True, use log-log axes.
        reference_slopes: Optional {label: exponent} to overlay reference lines
            m^exponent. For example, {"m^(1/3)": 1/3, "m^(1/2)": 1/2}.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    all_m = set()
    for m_to_val in scaling_data.values():
        all_m.update(m_to_val.keys())
    all_m = sorted(all_m)

    for label, m_to_val in scaling_data.items():
        ms = sorted(m_to_val.keys())
        vals = [m_to_val[m] for m in ms]
        mask = np.isfinite(vals)
        ms_arr = np.array(ms)[mask]
        vals_arr = np.array(vals)[mask]
        if len(ms_arr) == 0:
            continue
        ax.plot(ms_arr, vals_arr, "o-", label=label, linewidth=2.0, markersize=6)

    # Reference slopes anchored at the first m.
    if reference_slopes and all_m:
        m0 = all_m[0]
        # Anchor y to the median value across solvers at m0 (if available).
        y_anchors = [m_to_val[m0] for m_to_val in scaling_data.values()
                      if m0 in m_to_val and np.isfinite(m_to_val[m0])]
        if y_anchors:
            y0 = float(np.median(y_anchors))
            ms_ref = np.array(all_m)
            for slope_label, slope in reference_slopes.items():
                y_ref = y0 * (ms_ref / m0) ** slope
                ax.plot(ms_ref, y_ref, "--", alpha=0.5, linewidth=1.0,
                        label=f"∝ {slope_label}")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Number of groups m" + (" (log)" if log_scale else ""))
    ax.set_ylabel(ylabel + (" (log)" if log_scale else ""))
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
