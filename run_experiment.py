#!/usr/bin/env python3
"""Reproduce the experimental comparison from Appendix F of Patel & Manoj (ICLR 2026).

Generates the adversarial synthetic instance, tunes all solver families via
grid search, and produces iteration-based and time-based convergence plots.

Usage:
    python run_experiment.py                         # default synthetic instance
    python run_experiment.py --csv data.csv \\
        --target y --group g                         # load from CSV
    python run_experiment.py --m 50 --d 5 --seed 42  # custom synthetic params
    python run_experiment.py --folktables             # ACS income (10 US states)
    python run_experiment.py --folktables \\
        --acs-states CA TX NY --acs-subsample 200     # custom ACS options
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import dro
from dro import solvers, tuning, plotting
from dro.lewis_weights import build_lewis_geometry


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DRO linear regression experiment.")

    # Dataset source (mutually exclusive).
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--csv", type=str, default=None,
                      help="Path to CSV file with group regression data.")
    grp.add_argument("--folktables", action="store_true", default=False,
                      help="Use ACS income data from Folktables (grouped by state).")
    grp.add_argument("--synthetic", action="store_true", default=True,
                      help="Use synthetic adversarial instance (default).")

    # CSV options.
    p.add_argument("--target", type=str, default="y",
                    help="Target column name (for --csv).")
    p.add_argument("--group", type=str, default="group",
                    help="Group column name (for --csv).")
    p.add_argument("--features", nargs="*", default=None,
                    help="Feature column names (for --csv). Default: all others.")

    # Folktables options.
    p.add_argument("--acs-states", nargs="*", default=None,
                    help="State abbreviations for Folktables (default: 10 most populous).")
    p.add_argument("--acs-all", action="store_true",
                    help="Use all 50 states + DC + PR for Folktables (overrides --acs-states).")
    p.add_argument("--acs-group-by", choices=["state", "race"], default="state",
                    help="Grouping for Folktables data.")
    p.add_argument("--acs-subsample", type=int, default=None,
                    help="Subsample per group for Folktables (for speed).")
    p.add_argument("--acs-year", type=str, default="2018",
                    help="ACS survey year.")

    # Synthetic instance parameters.
    p.add_argument("--m", type=int, default=100, help="Total groups.")
    p.add_argument("--m-outlier", type=int, default=5, help="Outlier groups.")
    p.add_argument("--d", type=int, default=10, help="Dimension.")
    p.add_argument("--n-per-group", type=int, default=10, help="Samples per group.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Experiment parameters.
    p.add_argument("--T", type=int, default=100, help="Iteration budget.")
    p.add_argument("--warm-start", choices=["erm", "zero", "rand"], default="erm")

    # Run artifact parameters.
    p.add_argument("--runs-root", type=str, default="runs",
                    help="Parent directory for run folders.")
    p.add_argument("--run-name", type=str, default=None,
                    help="Optional suffix for the run directory name.")
    p.add_argument("--verbose", action="store_true")

    return p


def main():
    args = make_parser().parse_args()

    # ------------------------------------------------------------------
    # 0. Create run directory
    # ------------------------------------------------------------------
    run_dir = dro.artifacts.create_run_dir(root=args.runs_root, name=args.run_name)
    print(f"Run directory: {run_dir}")

    # ------------------------------------------------------------------
    # 1. Load or generate data
    # ------------------------------------------------------------------
    if args.csv:
        print(f"Loading data from {args.csv}...")
        A_groups, b_groups = dro.from_csv(
            args.csv, target_col=args.target, group_col=args.group,
            feature_cols=args.features,
        )
    elif args.folktables:
        print("Loading ACS income data from Folktables...")
        if args.acs_all:
            from dro.datasets_folktables import ALL_STATES
            states = ALL_STATES
        else:
            states = args.acs_states
        A_groups, b_groups, acs_info = dro.load_acs_income(
            states=states,
            survey_year=args.acs_year,
            group_by=args.acs_group_by,
            subsample=args.acs_subsample,
        )
        print(f"  Groups ({acs_info['n_groups']}): {acs_info['group_names']}")
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

    # Persist tuned hyperparameters to the run directory.
    dro.artifacts.save_hyperparameters(run_dir, configs)

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
    # 7. Summary table and artifact persistence
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Per-group loss summary")
    print("=" * 60 + "\n")

    # Collect solutions for the summary.
    solutions = {"ERM": x_erm}
    if F_opt is not None:
        solutions["OPT (CVXPY)"] = opt_result.x_final
    for label, result in curves.items():
        solutions[label] = result.x_final

    # Group names (from Folktables info or generic).
    group_names = None
    if args.folktables:
        group_names = acs_info["group_names"]

    summary_text = dro.summarize(A_groups, b_groups, solutions, group_names=group_names)
    print()

    # Persist summary, losses CSV, solutions, curves, and config.
    dro.artifacts.save_summary_txt(run_dir, summary_text)
    stats = dro.artifacts.save_losses(run_dir, A_groups, b_groups, solutions,
                                        group_names=group_names)
    dro.artifacts.save_stats(run_dir, stats)
    dro.artifacts.save_solutions(run_dir, solutions)
    dro.artifacts.save_curves(run_dir, curves)

    # Config dump captures everything needed to reproduce the run.
    config_dump = {
        "cli_args": vars(args),
        "dataset": {
            "m": m,
            "d": d,
            "cond_stack": cond_stack,
            "group_names": group_names,
        },
        "baselines": {
            "F_erm": F_erm,
            "F_opt": F_opt,
            "warm_start": args.warm_start,
        },
    }
    if args.folktables:
        config_dump["dataset"]["folktables"] = {
            "group_names": acs_info["group_names"],
            "feature_names": acs_info["feature_names"],
            "n_total": acs_info["n_total"],
        }
    dro.artifacts.save_config(run_dir, config_dump)

    # ------------------------------------------------------------------
    # 8. Plot iterations (log + linear)
    # ------------------------------------------------------------------
    save_iterations = os.path.join(run_dir, "iterations.png")
    save_iterations_linear = os.path.join(run_dir, "iterations_linear.png")
    save_time = os.path.join(run_dir, "time.png")
    save_time_linear = os.path.join(run_dir, "time_linear.png")

    print("Plotting iteration convergence...")
    plotting.plot_convergence(
        curves, F_opt=F_opt, F_erm=F_erm,
        title="Convergence on Max-Loss (0-{} iterations, log scale)".format(args.T),
        save_path=save_iterations,
        log_scale=True,
    )
    plotting.plot_convergence(
        curves, F_opt=F_opt, F_erm=F_erm,
        title="Convergence on Max-Loss (0-{} iterations, linear scale)".format(args.T),
        save_path=save_iterations_linear,
        log_scale=False,
    )

    # ------------------------------------------------------------------
    # 8. Equal-runtime comparison
    # ------------------------------------------------------------------
    # Use the runtime of the slowest method (from step 6) as the shared
    # time budget. Each method gets enough iterations (up to a large cap)
    # to fill that budget, and fast methods exit early when it's up.
    slowest = 0.0
    for label, result in curves.items():
        if result.times:
            slowest = max(slowest, result.times[-1])
    time_budget = max(slowest, 1e-3)
    # Generous iteration cap — let fast methods iterate freely until time runs out.
    big_T = max(args.T * 1000, 100000)

    print(f"\nRunning equal-runtime comparison (budget = {time_budget*1000:.2f} ms)...")
    timed_curves = {}

    cfg, _ = configs["subgradient"]
    if cfg:
        if cfg["mode"] == "fixed":
            res = solvers.subgradient_fixed(A_groups, b_groups, x0,
                                             step=cfg["step"], T=big_T,
                                             time_budget=time_budget)
            label = "Subgradient (fixed)"
        else:
            res = solvers.subgradient_diminishing(A_groups, b_groups, x0,
                                                    base_step=cfg["base_step"],
                                                    T=big_T, time_budget=time_budget)
            label = "Subgradient (diminishing)"
        if res:
            timed_curves[label] = res
            print(f"  {label}: {len(res.iters)-1} iters in {res.times[-1]*1000:.1f} ms")

    cfg, _ = configs["smooth"]
    if cfg:
        variant = cfg["variant"]
        common = dict(beta=cfg["beta"], delta=cfg["delta"], step=cfg["step"],
                      T=big_T, time_budget=time_budget)
        if variant == "gd":
            res = solvers.smooth_gd(A_groups, b_groups, x0, **common)
            label = "Smoothed GD"
        elif variant == "heavy_ball":
            res = solvers.smooth_heavy_ball(A_groups, b_groups, x0,
                                             momentum=cfg["momentum"], **common)
            label = "Smoothed Heavy-Ball"
        else:
            res = solvers.smooth_nesterov(A_groups, b_groups, x0,
                                           momentum=cfg["momentum"], **common)
            label = "Smoothed Nesterov"
        if res:
            timed_curves[label] = res
            print(f"  {label}: {len(res.iters)-1} iters in {res.times[-1]*1000:.1f} ms")

    cfg, _ = configs["ipm"]
    if cfg:
        res = solvers.interior_point(A_groups, b_groups, x0,
                                      mu0=cfg["mu0"], tau=cfg["tau"],
                                      inner_iters=cfg["inner_iters"],
                                      max_newton_steps=big_T,
                                      time_budget=time_budget)
        if res:
            timed_curves["IPM"] = res
            print(f"  IPM: {len(res.iters)-1} iters in {res.times[-1]*1000:.1f} ms")

    cfg, _ = configs["ball_oracle_naive"]
    if cfg:
        res = solvers.ball_oracle_naive(A_groups, b_groups, x0,
                                         beta=cfg["beta"], delta=cfg["delta"],
                                         R0=cfg["R0"], n_outer=big_T,
                                         R_decay=cfg["R_decay"],
                                         max_newton_steps=cfg["max_newton_steps"],
                                         time_budget=time_budget)
        if res:
            timed_curves["Ball-Oracle (Euclidean)"] = res
            print(f"  Ball-Oracle (Euclidean): {len(res.iters)-1} iters in {res.times[-1]*1000:.1f} ms")

    if "ball_oracle_lewis" in configs:
        cfg, _ = configs["ball_oracle_lewis"]
        if cfg:
            res = solvers.ball_oracle_lewis(A_groups, b_groups, x0, L=L_lewis,
                                             beta=cfg["beta"], delta=cfg["delta"],
                                             R0=cfg["R0"], n_outer=big_T,
                                             R_decay=cfg["R_decay"],
                                             max_newton_steps=cfg["max_newton_steps"],
                                             time_budget=time_budget)
            if res:
                timed_curves["Ball-Oracle (Lewis)"] = res
                print(f"  Ball-Oracle (Lewis): {len(res.iters)-1} iters in {res.times[-1]*1000:.1f} ms")

    plotting.plot_time_vs_accuracy(
        timed_curves, F_opt=F_opt, F_erm=F_erm,
        title=f"Time vs Accuracy (equal {time_budget*1000:.1f} ms budget, log scale)",
        save_path=save_time,
        log_scale=True,
    )
    plotting.plot_time_vs_accuracy(
        timed_curves, F_opt=F_opt, F_erm=F_erm,
        title=f"Time vs Accuracy (equal {time_budget*1000:.1f} ms budget, linear scale)",
        save_path=save_time_linear,
        log_scale=False,
    )

    # Save equal-runtime curves too (distinguished by "timed:" prefix).
    timed_curves_prefixed = {f"timed_{label}": res for label, res in timed_curves.items()}
    merged_curves = {**curves, **timed_curves_prefixed}
    dro.artifacts.save_curves(run_dir, merged_curves)

    print(f"\nDone. All artifacts saved to: {run_dir}")
    print("  - config.json, hyperparameters.json, stats.json")
    print("  - losses.csv, summary.txt")
    print("  - solutions.npz, curves.npz")
    print("  - iterations.png, iterations_linear.png")
    print("  - time.png, time_linear.png")


if __name__ == "__main__":
    main()
