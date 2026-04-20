#!/usr/bin/env python3
"""Scaling experiment: iterations-to-accuracy vs number of groups m.

Loads PUMA-grouped ACS income data (thousands of geographic groups), then
sweeps over m by subsampling groups. For each m, runs all solvers and records
the number of iterations (and wall-clock time) required to reach a target
accuracy (relative gap to the robust optimum).

This isolates the m-dependence of each solver's iteration complexity:
  - Subgradient, smoothed GD : m-independent (but geometry-dependent)
  - IPM                      : ~ m^{1/2} log(1/eps)
  - Ball-Oracle (naive)      : ~ m^{1/3} / eps^{2/3}
  - Ball-Oracle (Lewis)      : ~ min(rank(A), m)^{1/3} / eps^{2/3}

The Lewis-geometry method is expected to *plateau* once m > rank(A) = d,
while IPM and the naive Euclidean version should grow with m.

Usage:
    python run_scaling_experiment.py                              # default: CA PUMAs
    python run_scaling_experiment.py --acs-states CA TX NY FL IL  # larger pool
    python run_scaling_experiment.py --m-values 50 100 200 500    # custom sweep
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import dro
from dro import solvers
from dro.lewis_weights import build_lewis_geometry


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scaling experiment: m vs iterations-to-accuracy.")

    # Data source (mutually exclusive).
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true",
                      help="Use the synthetic adversarial instance (pool of generated groups).")
    src.add_argument("--folktables", action="store_true", default=True,
                      help="Use PUMA-grouped ACS income data (default).")

    # Folktables (PUMA) options.
    p.add_argument("--acs-states", nargs="*", default=["CA"],
                    help="States to load PUMAs from (default: CA, gives ~265 PUMAs).")
    p.add_argument("--acs-all", action="store_true",
                    help="Use all 51 regions (~2400 PUMAs nationally).")
    p.add_argument("--acs-year", type=str, default="2018")
    p.add_argument("--acs-group-by", choices=["state", "race", "puma", "state_race"],
                    default="puma",
                    help="Grouping key for ACS data (default: puma, for thousands of groups).")
    p.add_argument("--samples-per-puma", type=int, default=50,
                    help="Fixed samples per group (subsample within each group).")

    # Synthetic-instance options.
    p.add_argument("--syn-pool-m", type=int, default=2500,
                    help="Total synthetic groups in the pool (subsets drawn from this).")
    p.add_argument("--syn-outlier-frac", type=float, default=0.05,
                    help="Fraction of pool groups that are adversarial outliers.")
    p.add_argument("--syn-d", type=int, default=10,
                    help="Dimension of the synthetic instance.")
    p.add_argument("--syn-n-per-group", type=int, default=10,
                    help="Samples per synthetic group.")
    p.add_argument("--syn-seed", type=int, default=0,
                    help="Seed for synthetic pool generation.")

    # Sweep.
    p.add_argument("--m-values", nargs="*", type=int,
                    default=[25, 50, 100, 200],
                    help="Values of m (number of groups) to sweep over.")
    p.add_argument("--target-rel-gap", type=float, default=0.02,
                    help="Stop criterion: (max_loss - OPT) / OPT <= target.")
    p.add_argument("--T-cap", type=int, default=200,
                    help="Max iterations per solver (cap on iteration budget).")
    p.add_argument("--max-solver-time", type=float, default=1.0,
                    help="Hard wall-clock limit (seconds) per solver call. "
                         "Prevents intrinsically slow methods (e.g. subgradient) "
                         "from inflating total runtime.")
    p.add_argument("--n-trials", type=int, default=3,
                    help="Number of random subsets per m (results averaged).")
    p.add_argument("--seed", type=int, default=0)

    # Artifacts.
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="scaling")

    # Solver hyperparameters (fixed across m to keep the sweep clean).
    p.add_argument("--beta-rel", type=float, default=0.01,
                    help="Smoothing beta as fraction of initial max-loss.")
    p.add_argument("--delta-rel", type=float, default=0.0,
                    help="Smoothing delta as fraction of initial max-loss.")
    p.add_argument("--newton-steps", type=int, default=5,
                    help="Inner Newton steps per ball-oracle call.")

    return p


def iters_to_accuracy(
    iters: list[int],
    best_values: list[float],
    F_opt: float,
    target_rel_gap: float,
) -> int | None:
    """Return first iteration index where (best - OPT) / OPT <= target, or None."""
    if not iters or not best_values or not np.isfinite(F_opt) or F_opt <= 0:
        return None
    for k, v in zip(iters, best_values):
        if (v - F_opt) / F_opt <= target_rel_gap:
            return k
    return None


def time_to_accuracy(
    times: list[float],
    best_values: list[float],
    F_opt: float,
    target_rel_gap: float,
) -> float | None:
    """Return first wall-clock second where target accuracy is reached, or None."""
    if not times or not best_values or not np.isfinite(F_opt) or F_opt <= 0:
        return None
    for t, v in zip(times, best_values):
        if (v - F_opt) / F_opt <= target_rel_gap:
            return float(t)
    return None


def run_one_size(
    A_groups_full,
    b_groups_full,
    m: int,
    args,
    rng,
) -> dict:
    """Run all solvers on a random subset of m groups. Returns per-solver stats."""
    # Subsample groups (without replacement).
    idx = rng.choice(len(A_groups_full), size=m, replace=False)
    A_groups = [A_groups_full[i] for i in idx]
    b_groups = [b_groups_full[i] for i in idx]

    d = A_groups[0].shape[1]

    # Baselines.
    x0 = dro.erm_solution(A_groups, b_groups)
    F_erm = dro.max_group_loss(A_groups, b_groups, x0)

    try:
        opt = solvers.solve_exact(A_groups, b_groups)
        F_opt = opt.best_loss
    except Exception as e:
        print(f"    [CVXPY failed at m={m}: {e}]")
        return {"m": m, "F_erm": F_erm, "F_opt": None, "solvers": {}}

    # Smoothing parameters: follow Algorithm 1 line 5 of the paper, scaled so
    # the smoothed objective is accurate to within the target additive error.
    # Choice: beta = eps / (4 log m), delta = eps / 4, where eps = target_rel_gap * F_opt.
    eps_additive = args.target_rel_gap * F_opt
    beta = eps_additive / (4.0 * max(np.log(m), 1.0))
    delta = eps_additive / 4.0

    # Lewis geometry.
    L = build_lewis_geometry(A_groups, b_groups, p=np.inf)
    xnorm_lewis = max(float(np.linalg.norm(L @ x0)), 1.0)
    xnorm_euc = max(float(np.linalg.norm(x0)), 1.0)

    results = {}
    trajectories = {}  # full SolverResult objects, for per-m plots
    tb = args.max_solver_time

    def _record(label, res):
        if res is None:
            return
        results[label] = {
            "iters_to_target": iters_to_accuracy(res.iters, res.best_values, F_opt, args.target_rel_gap),
            "time_to_target": time_to_accuracy(res.times, res.best_values, F_opt, args.target_rel_gap),
            "final_loss": res.best_loss,
        }
        trajectories[label] = res

    _record("Subgradient", solvers.subgradient_diminishing(
        A_groups, b_groups, x0,
        base_step=1e-5, T=args.T_cap, time_budget=tb,
    ))
    _record("Smoothed Heavy-Ball", solvers.smooth_heavy_ball(
        A_groups, b_groups, x0,
        beta=beta, delta=delta, step=1e-3, momentum=0.9,
        T=args.T_cap, time_budget=tb,
    ))
    _record("IPM", solvers.interior_point(
        A_groups, b_groups, x0,
        mu0=1e-3, tau=0.5, inner_iters=3,
        max_newton_steps=args.T_cap, time_budget=tb,
    ))
    _record("Ball-Oracle (Euclidean)", solvers.ball_oracle_naive(
        A_groups, b_groups, x0,
        beta=beta, delta=delta,
        R0=xnorm_euc, n_outer=args.T_cap, R_decay=0.9,
        max_newton_steps=args.newton_steps, time_budget=tb,
    ))
    _record("Ball-Oracle (Lewis)", solvers.ball_oracle_lewis(
        A_groups, b_groups, x0, L=L,
        beta=beta, delta=delta,
        R0=xnorm_lewis, n_outer=args.T_cap, R_decay=0.9,
        max_newton_steps=args.newton_steps, time_budget=tb,
    ))

    return {
        "m": m,
        "F_erm": float(F_erm),
        "F_opt": float(F_opt),
        "solvers": results,
        "trajectories": trajectories,
    }


def main():
    args = make_parser().parse_args()

    # Create run directory.
    run_dir = dro.artifacts.create_run_dir(root=args.runs_root, name=args.run_name)
    print(f"Run directory: {run_dir}")

    # Load or generate the full pool of groups once.
    if args.synthetic:
        m_outlier = int(round(args.syn_outlier_frac * args.syn_pool_m))
        m_outlier = max(1, m_outlier)
        print(f"Generating synthetic pool: m={args.syn_pool_m} "
              f"({m_outlier} outliers), d={args.syn_d}, "
              f"n_per_group={args.syn_n_per_group}")
        A_groups_full, b_groups_full = dro.generate_hard_instance(
            m_total=args.syn_pool_m,
            m_outlier=m_outlier,
            d=args.syn_d,
            n_per_group=args.syn_n_per_group,
            seed=args.syn_seed,
        )
        info = {
            "n_groups": len(A_groups_full),
            "d": A_groups_full[0].shape[1],
            "n_total": sum(A.shape[0] for A in A_groups_full),
            "source": "synthetic",
        }
    else:
        if args.acs_all:
            from dro.datasets_folktables import ALL_STATES
            states = ALL_STATES
        else:
            states = args.acs_states

        print(f"Loading ACS data for states: {states}, group_by={args.acs_group_by} ...")
        A_groups_full, b_groups_full, info = dro.load_acs_income(
            states=states,
            survey_year=args.acs_year,
            group_by=args.acs_group_by,
            subsample=args.samples_per_puma,
            min_group_size=args.samples_per_puma,
        )
        info["source"] = f"folktables_{args.acs_group_by}"

    print(f"  Pool: {info['n_groups']} groups, d={info['d']}, "
          f"n_total={info['n_total']}")

    # Filter requested m values to those <= available groups.
    m_values = sorted(m for m in args.m_values if m <= info["n_groups"])
    if not m_values:
        raise ValueError(f"No m values in {args.m_values} are <= n_groups={info['n_groups']}.")
    if max(args.m_values) > info["n_groups"]:
        print(f"  Capping m at {info['n_groups']} (requested up to {max(args.m_values)}).")

    print(f"\nSweeping m over {m_values} with target rel gap = {args.target_rel_gap:.3f}, "
          f"n_trials = {args.n_trials}")

    rng = np.random.default_rng(args.seed)
    sweep_results = []

    for m in m_values:
        print(f"\n--- m = {m} ---")
        t0 = time.perf_counter()

        # Run n_trials independent trials with different random subsets.
        trials = []
        for trial in range(args.n_trials):
            trial_result = run_one_size(A_groups_full, b_groups_full, m, args, rng)
            trials.append(trial_result)

        # Average metrics across trials (median is more robust than mean to failures).
        def _agg(solver, field):
            vals = [t["solvers"].get(solver, {}).get(field) for t in trials]
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else None

        solvers_agg = {}
        all_solver_names = set()
        for t in trials:
            all_solver_names.update(t["solvers"].keys())

        for solver in all_solver_names:
            iters_vals = [t["solvers"].get(solver, {}).get("iters_to_target") for t in trials]
            time_vals = [t["solvers"].get(solver, {}).get("time_to_target") for t in trials]
            iters_ok = [v for v in iters_vals if v is not None]
            time_ok = [v for v in time_vals if v is not None]
            solvers_agg[solver] = {
                "iters_to_target": float(np.median(iters_ok)) if iters_ok else None,
                "time_to_target": float(np.median(time_ok)) if time_ok else None,
                "n_trials_succeeded": len(iters_ok),
                "n_trials": args.n_trials,
            }

        result = {
            "m": m,
            "F_erm": float(np.median([t["F_erm"] for t in trials])),
            "F_opt": float(np.median([t["F_opt"] for t in trials if t["F_opt"] is not None])),
            "solvers": solvers_agg,
            # Keep trial 0's trajectories for the per-m plots (not serialized to JSON).
            "trajectories": trials[0].get("trajectories", {}),
        }

        print(f"  [total {time.perf_counter() - t0:.1f}s, "
              f"F_erm={result['F_erm']:.4f}, F_opt={result['F_opt']:.4f}]")
        for solver_name, stats in result["solvers"].items():
            its = stats["iters_to_target"]
            tms = stats["time_to_target"]
            its_str = f"{its:.1f}" if its is not None else "N/A"
            tms_str = f"{tms*1000:.1f}ms" if tms is not None else "N/A"
            n_ok = stats["n_trials_succeeded"]
            print(f"  {solver_name:30s}: iters={its_str:>6}, time={tms_str:>10}  ({n_ok}/{stats['n_trials']} trials ok)")

        sweep_results.append(result)

    # --------------------------------------------------------------------
    # Persist artifacts.
    # --------------------------------------------------------------------

    # Reshape into {solver: {m: metric}} for plotting.
    iter_data = {}
    time_data = {}
    for result in sweep_results:
        m = result["m"]
        for solver, stats in result["solvers"].items():
            iter_data.setdefault(solver, {})[m] = (
                stats["iters_to_target"] if stats["iters_to_target"] is not None else np.inf
            )
            time_data.setdefault(solver, {})[m] = (
                stats["time_to_target"] if stats["time_to_target"] is not None else np.inf
            )

    # CSV: one row per (m, solver) pair (median across trials).
    import csv
    with open(os.path.join(run_dir, "scaling.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "solver", "iters_to_target_median", "time_to_target_s_median",
                          "n_trials_succeeded", "n_trials", "F_erm_median", "F_opt_median"])
        for result in sweep_results:
            for solver, stats in result["solvers"].items():
                writer.writerow([
                    result["m"], solver,
                    stats["iters_to_target"] if stats["iters_to_target"] is not None else "",
                    stats["time_to_target"] if stats["time_to_target"] is not None else "",
                    stats["n_trials_succeeded"],
                    stats["n_trials"],
                    f"{result['F_erm']:.6f}",
                    f"{result['F_opt']:.6f}" if result['F_opt'] else "",
                ])

    # Config dump.
    dro.artifacts.save_config(run_dir, {"cli_args": vars(args), "m_values_used": m_values})

    # JSON dump of raw results (strip trajectories — they're SolverResult objects,
    # saved as PNG plots instead).
    import json
    jsonable = [{k: v for k, v in r.items() if k != "trajectories"}
                for r in sweep_results]
    with open(os.path.join(run_dir, "scaling_results.json"), "w") as f:
        json.dump(dro.artifacts._to_jsonable(jsonable), f, indent=2)

    # --------------------------------------------------------------------
    # Per-m convergence plots: 4 plots per m (iters log/linear, time log/linear).
    # --------------------------------------------------------------------
    per_m_dir = os.path.join(run_dir, "per_m_plots")
    os.makedirs(per_m_dir, exist_ok=True)

    for result in sweep_results:
        m = result["m"]
        curves = result.get("trajectories", {})
        if not curves:
            continue

        F_opt = result["F_opt"]
        F_erm = result["F_erm"]

        dro.plotting.plot_convergence(
            curves, F_opt=F_opt, F_erm=F_erm,
            title=f"m={m}: Convergence vs iterations (log scale)",
            save_path=os.path.join(per_m_dir, f"iterations_m{m}.png"),
            log_scale=True,
        )
        dro.plotting.plot_convergence(
            curves, F_opt=F_opt, F_erm=F_erm,
            title=f"m={m}: Convergence vs iterations (linear scale)",
            save_path=os.path.join(per_m_dir, f"iterations_m{m}_linear.png"),
            log_scale=False,
        )
        dro.plotting.plot_time_vs_accuracy(
            curves, F_opt=F_opt, F_erm=F_erm,
            title=f"m={m}: Convergence vs wall-clock time (log scale)",
            save_path=os.path.join(per_m_dir, f"time_m{m}.png"),
            log_scale=True,
        )
        dro.plotting.plot_time_vs_accuracy(
            curves, F_opt=F_opt, F_erm=F_erm,
            title=f"m={m}: Convergence vs wall-clock time (linear scale)",
            save_path=os.path.join(per_m_dir, f"time_m{m}_linear.png"),
            log_scale=False,
        )

    # Plots: m vs iterations, m vs time, with reference slopes.
    # Both log-log and linear-scale versions.
    reference = {"m^(1/3)": 1.0 / 3.0, "m^(1/2)": 1.0 / 2.0}

    dro.plotting.plot_scaling(
        iter_data,
        ylabel="Iterations to reach target gap",
        title=f"Scaling: iterations to relative gap ≤ {args.target_rel_gap} (log scale)",
        save_path=os.path.join(run_dir, "scaling_iters.png"),
        log_scale=True,
        reference_slopes=reference,
    )
    dro.plotting.plot_scaling(
        iter_data,
        ylabel="Iterations to reach target gap",
        title=f"Scaling: iterations to relative gap ≤ {args.target_rel_gap} (linear scale)",
        save_path=os.path.join(run_dir, "scaling_iters_linear.png"),
        log_scale=False,
    )
    dro.plotting.plot_scaling(
        time_data,
        ylabel="Wall-clock time to target gap (s)",
        title=f"Scaling: time to relative gap ≤ {args.target_rel_gap} (log scale)",
        save_path=os.path.join(run_dir, "scaling_time.png"),
        log_scale=True,
        reference_slopes=reference,
    )
    dro.plotting.plot_scaling(
        time_data,
        ylabel="Wall-clock time to target gap (s)",
        title=f"Scaling: time to relative gap ≤ {args.target_rel_gap} (linear scale)",
        save_path=os.path.join(run_dir, "scaling_time_linear.png"),
        log_scale=False,
    )

    print(f"\nDone. Artifacts saved to: {run_dir}")
    print("  Scaling plots: scaling_iters{,_linear}.png, scaling_time{,_linear}.png")
    print(f"  Per-m plots:   per_m_plots/ (iterations_m*.png, time_m*.png; log + linear)")


if __name__ == "__main__":
    main()
