"""Run artifact management.

Every invocation of run_experiment.py creates a timestamped subdirectory
under runs/ (or a user-specified root) containing:

    runs/<timestamp>/
        config.json          — CLI args, dataset metadata, random seeds
        hyperparameters.json — best-tuned hyperparameters per method
        losses.csv           — per-group losses for every solver, with aggregates
        summary.txt          — human-readable version of losses.csv
        solutions.npz        — parameter vectors x for every solver
        curves.npz           — iteration-wise convergence curves (iters, best_values, times)
        iterations.png       — log-scale convergence plot
        iterations_linear.png
        time.png             — log-scale time-vs-accuracy plot
        time_linear.png
        interpolation.png    — (optional) Gp interpolation path

Use `create_run_dir()` to make a new directory with a unique timestamp.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

from .problem import SolverResult, group_losses


def create_run_dir(root: str = "runs", name: str | None = None) -> str:
    """Create a new uniquely-named run directory under `root`.

    Args:
        root: Parent directory for run folders (default "runs/").
        name: Optional suffix to append to the timestamp.

    Returns:
        Absolute path to the created directory.
    """
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        dirname = f"{stamp}_{name}"
    else:
        dirname = stamp

    run_dir = os.path.join(root, dirname)
    # Break ties on same-second starts.
    suffix = 0
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = os.path.join(root, f"{dirname}_{suffix}")
    os.makedirs(run_dir)
    return os.path.abspath(run_dir)


# ---------------------------------------------------------------------------
# JSON / CSV serialization helpers
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """Convert numpy types and common containers to JSON-serializable forms."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_config(run_dir: str, config: dict) -> None:
    """Write config.json with CLI args, dataset metadata, etc."""
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(config), f, indent=2, sort_keys=True)


def save_hyperparameters(
    run_dir: str,
    configs: dict[str, tuple[dict | None, float]],
) -> None:
    """Write hyperparameters.json with best-tuned configs for every method.

    Args:
        configs: Output of tuning.tune_all(), i.e. {method: (cfg, best_val)}.
    """
    payload = {
        method: {"best_config": cfg, "best_value": val}
        for method, (cfg, val) in configs.items()
    }
    path = os.path.join(run_dir, "hyperparameters.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)


def save_losses(
    run_dir: str,
    A_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    solutions: dict[str, np.ndarray],
    group_names: list[str] | None = None,
) -> dict:
    """Write losses.csv with per-group losses + aggregates, return stats dict.

    The CSV has one row per group, plus aggregate rows (max, mean, median, min,
    std, max/mean) at the bottom. Columns are solver names.

    Returns:
        A dict with per-solver aggregate statistics (for downstream use).
    """
    m = len(A_groups)
    if group_names is None:
        group_names = [f"Group_{i}" for i in range(m)]

    labels = list(solutions.keys())

    # Compute all losses.
    all_losses = {
        label: group_losses(A_groups, b_groups, x)
        for label, x in solutions.items()
    }

    # Write CSV.
    csv_path = os.path.join(run_dir, "losses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "n"] + labels)

        for i in range(m):
            writer.writerow([group_names[i], A_groups[i].shape[0]] +
                            [f"{all_losses[l][i]:.6f}" for l in labels])

        writer.writerow([])  # blank separator

        for agg_name, agg_fn in [
            ("max", np.max), ("mean", np.mean), ("median", np.median),
            ("min", np.min), ("std", np.std),
        ]:
            writer.writerow([agg_name, ""] +
                            [f"{agg_fn(all_losses[l]):.6f}" for l in labels])

        writer.writerow(["max_over_mean", ""] +
                        [f"{np.max(all_losses[l]) / max(np.mean(all_losses[l]), 1e-16):.6f}"
                         for l in labels])

    # Build a stats dict for the return value.
    stats = {}
    for label, losses in all_losses.items():
        stats[label] = {
            "max": float(np.max(losses)),
            "mean": float(np.mean(losses)),
            "median": float(np.median(losses)),
            "min": float(np.min(losses)),
            "std": float(np.std(losses)),
            "max_over_mean": float(np.max(losses) / max(np.mean(losses), 1e-16)),
        }

    return stats


def save_summary_txt(run_dir: str, summary_text: str) -> None:
    """Save the human-readable summary table from problem.summarize()."""
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(summary_text + "\n")


def save_solutions(run_dir: str, solutions: dict[str, np.ndarray]) -> None:
    """Save parameter vectors (x) for every solver in a single .npz file."""
    path = os.path.join(run_dir, "solutions.npz")
    # np.savez requires valid keyword identifiers; sanitize labels.
    safe = {_sanitize(k): np.asarray(v) for k, v in solutions.items()}
    np.savez(path, **safe)
    # Also write a mapping from sanitized → original names.
    mapping = {_sanitize(k): k for k in solutions}
    with open(os.path.join(run_dir, "solutions_labels.json"), "w") as f:
        json.dump(mapping, f, indent=2)


def save_curves(
    run_dir: str,
    curves: dict[str, SolverResult],
) -> None:
    """Save per-iteration convergence data (iters, best_values, times) as NPZ.

    For each solver, stores three arrays: <label>__iters, <label>__best, <label>__times.
    """
    arrays = {}
    for label, result in curves.items():
        if result is None:
            continue
        key = _sanitize(label)
        if result.iters is not None:
            arrays[f"{key}__iters"] = np.asarray(result.iters)
        if result.best_values is not None:
            arrays[f"{key}__best"] = np.asarray(result.best_values)
        if result.times is not None:
            arrays[f"{key}__times"] = np.asarray(result.times)

    if arrays:
        np.savez(os.path.join(run_dir, "curves.npz"), **arrays)


def save_stats(run_dir: str, stats: dict) -> None:
    """Save aggregate stats (from save_losses) as stats.json for easy parsing."""
    path = os.path.join(run_dir, "stats.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(stats), f, indent=2, sort_keys=True)


def _sanitize(name: str) -> str:
    """Make a string safe as an NPZ key / filename."""
    return (name
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace("/", "_"))
