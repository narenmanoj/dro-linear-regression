#!/usr/bin/env python3
"""Reproduce the experimental comparison from Appendix F of Patel & Manoj (ICLR 2026).

Generates the adversarial synthetic instance, tunes all solver families via
grid search, and produces iteration-based and time-based convergence plots.

Usage:
    python run_experiment.py                         # default synthetic instance
    python run_experiment.py --csv data.csv \\
        --target y --group g                         # load from CSV
    python run_experiment.py --m 50 --d 5 --seed 42  # custom synthetic params
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import dro
from dro import solvers, tuning, plotting
from dro.lewis_weights import build_lewis_geometry
from dro.plotting import timed_run


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DRO linear regression experiment.")

    # Dataset source (mutually exclusive).
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--csv", type=str, default=None,
                      help="Path to CSV file with group regression data.")
    grp.add_argument("--synthetic", action="store_true", default=True,
                      help="Use synthetic adversarial instance (default).")

    # CSV options.
    p.add_argument("--target", type=str, default="y",
                    help="Target column name (for --csv).")
    p.add_argument("--group", type=str, default="group",
                    help="Group column name (for --csv).")
    p.add_argument("--features", nargs="*", default=None,
                    help="Feature column names (for --csv). Default: all others.")

    # Synthetic instance parameters.
    p.add_argument("--m", type=int, default=100, help="Total groups.")
    p.add_argument("--m-outlier", type=int, default=5, help="Outlier groups.")
    p.add_argument("--d", type=int, default=10, help="Dimension.")
    p.add_argument("--n-per-group", type=int, default=10, help="Samples per group.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Experiment parameters.
    p.add_argument("--T", type=int, default=100, help="Iteration budget.")
    p.add_argument("--warm-start", choices=["erm", "zero", "rand"], default="erm")
    p.add_argument("--save-iterations", type=str, default="iterations.png")
    p.add_argument("--save-time", type=str, default="time.png")
    p.add_argument("--verbose", action="store_true")

    return p


def main():
    args = make_parser().parse_args()

    # ------------------------------------------------------------------
    # 1. Load or generate data
    # ------------------------------------------------------------------
    if args.csv:
        print(f"Loading data from {args.csv}...")
        A_groups, b_groups = dro.from_csv(
            args.csv, target_col=args.target, group_col=args.group,
            feature_cols=args.features,
        )
    else:
        print("Generating synthetic adversarial instance...")
        A_groups, b_groups = dro.generate_hard_instance(
            m_total=args.m, m_outlier=args.m_outlier,
            d=args.d, n_per_group=args.n_per_group, seed=args.seed,
        )

    m = len(A_groups)
    d = A_groups[0].shape[1]

    # ------------------------------------------------------------------
    # 2. Baselines: ERM and (optionally) exact CVXPY solve
    # ------------------------------------------------------------------
    x_erm = dro.erm_solution(A_groups, b_groups)
    F_erm = dro.max_group_loss(A_groups, b_groups, x_erm)

    A_all = np.vstack(A_groups)
    G = A_all.T @ A_all + 1e-12 * np.eye(d)
    evals = np.linalg.eigvalsh(G)
    cond_stack = float(evals.max() / max(evals.min(), 1e-18))

    print(f"  m={m}, d={d}, cond(stack)={cond_stack:.2e}")
    print(f"  ERM max-group loss: {F_erm:.6e}")

    F_opt = None
    try:
        opt_result = solvers.solve_exact(A_groups, b_groups)
        F_opt = opt_result.best_loss
        print(f"  Robust optimum (CVXPY): {F_opt:.6e}")
        print(f"  ERM - OPT gap: {F_erm - F_opt:.6e}")
    except Exception as e:
        print(f"  [CVXPY unavailable: {e}]")

    # ------------------------------------------------------------------
    # 3. Warm start
    # ------------------------------------------------------------------
    if args.warm_start == "erm":
        x0 = x_erm.copy()
    elif args.warm_start == "zero":
        x0 = np.zeros(d)
    else:
        x0 = np.random.default_rng(1).standard_normal(d)

    print(f"  Warm start: {args.warm_start}, init max-loss: "
          f"{dro.max_group_loss(A_groups, b_groups, x0):.6e}")

    # ------------------------------------------------------------------
    # 4. Build Lewis geometry
    # ------------------------------------------------------------------
    print("Computing block Lewis weights...")
    t0 = time.perf_counter()
    L_lewis = build_lewis_geometry(A_groups, b_groups, p=np.inf)
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 5. Tune all methods
    # ------------------------------------------------------------------
    print("\nTuning all methods (this may take a while)...")
    configs = tuning.tune_all(A_groups, b_groups, x0, L=L_lewis,
                              T=args.T, verbose=args.verbose)

    print("\nBest configurations:")
    for name, (cfg, val) in configs.items():
        if cfg is not None:
            print(f"  {name}: val={val:.6e}, cfg={cfg}")
        else:
            print(f"  {name}: no stable configuration found")

    # ------------------------------------------------------------------
    # 6. Final runs for convergence curves
    # ------------------------------------------------------------------
    print("\nRunning final curves...")
    curves = {}

    # Subgradient
    cfg, _ = configs["subgradient"]
    if cfg:
        if cfg["mode"] == "fixed":
            res = solvers.subgradient_fixed(A_groups, b_groups, x0, step=cfg["step"], T=args.T)
            label = "Subgradient (fixed)"
        else:
            res = solvers.subgradient_diminishing(A_groups, b_groups, x0, base_step=cfg["base_step"], T=args.T)
            label = "Subgradient (diminishing)"
        if res:
            curves[label] = res

    # Smoothed gradient
    cfg, _ = configs["smooth"]
    if cfg:
        variant = cfg["variant"]
        common = dict(beta=cfg["beta"], delta=cfg["delta"], step=cfg["step"], T=args.T)
        if variant == "gd":
            res = solvers.smooth_gd(A_groups, b_groups, x0, **common)
            label = "Smoothed GD"
        elif variant == "heavy_ball":
            res = solvers.smooth_heavy_ball(A_groups, b_groups, x0, momentum=cfg["momentum"], **common)
            label = "Smoothed Heavy-Ball"
        else:
            res = solvers.smooth_nesterov(A_groups, b_groups, x0, momentum=cfg["momentum"], **common)
            label = "Smoothed Nesterov"
        if res:
            curves[label] = res

    # IPM
    cfg, _ = configs["ipm"]
    if cfg:
        res = solvers.interior_point(A_groups, b_groups, x0,
                                     mu0=cfg["mu0"], tau=cfg["tau"],
                                     inner_iters=cfg["inner_iters"],
                                     max_newton_steps=args.T)
        if res:
            curves["IPM"] = res

    # Ball-oracle naive
    cfg, _ = configs["ball_oracle_naive"]
    if cfg:
        res = solvers.ball_oracle_naive(A_groups, b_groups, x0,
                                        beta=cfg["beta"], delta=cfg["delta"],
                                        R0=cfg["R0"], n_outer=cfg["n_outer"],
                                        R_decay=cfg["R_decay"],
                                        max_newton_steps=cfg["max_newton_steps"])
        if res:
            curves["Ball-Oracle (Euclidean)"] = res

    # Ball-oracle Lewis
    if "ball_oracle_lewis" in configs:
        cfg, _ = configs["ball_oracle_lewis"]
        if cfg:
            res = solvers.ball_oracle_lewis(A_groups, b_groups, x0, L=L_lewis,
                                            beta=cfg["beta"], delta=cfg["delta"],
                                            R0=cfg["R0"], n_outer=cfg["n_outer"],
                                            R_decay=cfg["R_decay"],
                                            max_newton_steps=cfg["max_newton_steps"])
            if res:
                curves["Ball-Oracle (Lewis)"] = res

    # ------------------------------------------------------------------
    # 7. Plot iterations
    # ------------------------------------------------------------------
    print("\nPlotting iteration convergence...")
    plotting.plot_convergence(
        curves, F_opt=F_opt, F_erm=F_erm,
        title="Convergence on Max-Loss (0-{} iterations)".format(args.T),
        save_path=args.save_iterations,
    )

    # ------------------------------------------------------------------
    # 8. Timed runs and plot
    # ------------------------------------------------------------------
    print("Running timed comparisons...")
    timed_curves = {}

    cfg, _ = configs["subgradient"]
    if cfg:
        if cfg["mode"] == "fixed":
            res, rt = timed_run(solvers.subgradient_fixed, A_groups, b_groups, x0,
                                step=cfg["step"], T=args.T)
            label = "Subgradient (fixed)"
        else:
            res, rt = timed_run(solvers.subgradient_diminishing, A_groups, b_groups, x0,
                                base_step=cfg["base_step"], T=args.T)
            label = "Subgradient (diminishing)"
        if res:
            timed_curves[label] = (res, rt)

    cfg, _ = configs["smooth"]
    if cfg:
        variant = cfg["variant"]
        common = dict(beta=cfg["beta"], delta=cfg["delta"], step=cfg["step"], T=args.T)
        if variant == "gd":
            res, rt = timed_run(solvers.smooth_gd, A_groups, b_groups, x0, **common)
            label = "Smoothed GD"
        elif variant == "heavy_ball":
            res, rt = timed_run(solvers.smooth_heavy_ball, A_groups, b_groups, x0,
                                momentum=cfg["momentum"], **common)
            label = "Smoothed Heavy-Ball"
        else:
            res, rt = timed_run(solvers.smooth_nesterov, A_groups, b_groups, x0,
                                momentum=cfg["momentum"], **common)
            label = "Smoothed Nesterov"
        if res:
            timed_curves[label] = (res, rt)

    cfg, _ = configs["ipm"]
    if cfg:
        res, rt = timed_run(solvers.interior_point, A_groups, b_groups, x0,
                            mu0=cfg["mu0"], tau=cfg["tau"],
                            inner_iters=cfg["inner_iters"],
                            max_newton_steps=args.T)
        if res:
            timed_curves["IPM"] = (res, rt)

    cfg, _ = configs["ball_oracle_naive"]
    if cfg:
        res, rt = timed_run(solvers.ball_oracle_naive, A_groups, b_groups, x0,
                            beta=cfg["beta"], delta=cfg["delta"],
                            R0=cfg["R0"], n_outer=cfg["n_outer"],
                            R_decay=cfg["R_decay"],
                            max_newton_steps=cfg["max_newton_steps"])
        if res:
            timed_curves["Ball-Oracle (Euclidean)"] = (res, rt)

    if "ball_oracle_lewis" in configs:
        cfg, _ = configs["ball_oracle_lewis"]
        if cfg:
            res, rt = timed_run(solvers.ball_oracle_lewis, A_groups, b_groups, x0,
                                L=L_lewis, beta=cfg["beta"], delta=cfg["delta"],
                                R0=cfg["R0"], n_outer=cfg["n_outer"],
                                R_decay=cfg["R_decay"],
                                max_newton_steps=cfg["max_newton_steps"])
            if res:
                timed_curves["Ball-Oracle (Lewis)"] = (res, rt)

    plotting.plot_time_vs_accuracy(
        timed_curves, F_opt=F_opt, F_erm=F_erm,
        title="Time vs Accuracy (best tuned methods)",
        save_path=args.save_time,
    )

    print("\nDone. Plots saved to", args.save_iterations, "and", args.save_time)


if __name__ == "__main__":
    main()
