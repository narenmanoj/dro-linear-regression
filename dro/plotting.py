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


# ---------------------------------------------------------------------------
# US state heatmaps
# ---------------------------------------------------------------------------

# Full state name -> 2-letter abbreviation (for matching GeoJSON "name" field).
_STATE_NAME_TO_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR",
}

_US_STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/"
    "data/geojson/us-states.json"
)


def _ensure_cached_us_states_geojson() -> str:
    """Download and cache the US states GeoJSON on first use."""
    import os
    import urllib.request
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dro-linear-regression")
    cache_path = os.path.join(cache_dir, "us_states.geojson")
    if not os.path.exists(cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading US states GeoJSON to {cache_path} ...")
        urllib.request.urlretrieve(_US_STATES_GEOJSON_URL, cache_path)
    return cache_path


def _load_us_states_gdf(apply_insets: bool = True):
    """Load US states as a GeoDataFrame with abbreviations; optionally apply
    Alaska/Hawaii insets (translate + scale to fit near continental US).
    """
    import geopandas as gpd
    from shapely.affinity import translate, scale as shp_scale

    gdf = gpd.read_file(_ensure_cached_us_states_geojson())
    # The GeoJSON uses "name" with full state names.
    gdf["abbr"] = gdf["name"].map(_STATE_NAME_TO_ABBR)

    if apply_insets:
        # Target: AK as a scaled inset at the lower-left of the continental US,
        # HI just below SoCal. Offsets below are in geographic degrees.
        new_geoms = []
        for _, row in gdf.iterrows():
            g = row.geometry
            if row["abbr"] == "AK":
                g = shp_scale(g, xfact=0.35, yfact=0.35, origin="center")
                g = translate(g, xoff=32, yoff=-43)
            elif row["abbr"] == "HI":
                g = translate(g, xoff=51, yoff=3)
            new_geoms.append(g)
        gdf = gdf.set_geometry(new_geoms)

    return gdf


# Continental-US bounding box used for all state heatmaps so the map fills the axes.
_US_MAP_XLIM = (-128, -65)
_US_MAP_YLIM = (20, 52)


def plot_us_state_heatmap(
    state_to_value: dict[str, float],
    title: str = "Per-state value",
    cmap_name: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
    cbar_label: str = "per-state MSE",
    ax=None,
    show_labels: bool = False,
    _gdf=None,
):
    """Render a geographically accurate US state choropleth (via geopandas).

    States are colored by their `state_to_value` entry (states missing a value
    are light grey). Alaska and Hawaii are repositioned as insets near the
    continental US. Puerto Rico is skipped (not in the underlying GeoJSON).

    Args:
        show_labels: If True, overlay state abbreviation + value at each centroid.
            Defaults False (unlabeled, cleaner for presentations).

    Requires: geopandas, shapely.
    """
    gdf = _gdf.copy() if _gdf is not None else _load_us_states_gdf()
    gdf["value"] = gdf["abbr"].map(state_to_value)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    gdf.plot(
        column="value",
        cmap=cmap_name,
        vmin=vmin, vmax=vmax,
        ax=ax,
        edgecolor="white", linewidth=0.3,
        legend=True,
        legend_kwds={"label": cbar_label, "shrink": 0.7, "fraction": 0.035, "pad": 0.02},
        missing_kwds={"color": "lightgrey", "edgecolor": "white", "linewidth": 0.3},
    )

    if show_labels:
        for _, row in gdf.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            c = row.geometry.centroid
            val = row["value"]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                label = row["abbr"]
            else:
                label = f"{row['abbr']}\n{val:.1f}"
            ax.text(c.x, c.y, label, ha="center", va="center", fontsize=6)

    ax.set_xlim(_US_MAP_XLIM)
    ax.set_ylim(_US_MAP_YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return ax


def plot_us_state_heatmaps_grid(
    solver_to_state_values: dict[str, dict[str, float]],
    title_prefix: str = "",
    cmap_name: str = "RdYlGn_r",
    shared_scale: bool = True,
    cbar_label: str = "per-state MSE",
    save_path: str | None = None,
):
    """Grid of US choropleths, one per solver. Shared colormap for direct comparison."""
    labels = list(solver_to_state_values.keys())
    n = len(labels)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    # Load the GeoDataFrame once; all subplots reuse it.
    gdf = _load_us_states_gdf()

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    if shared_scale:
        all_vals = [v for d in solver_to_state_values.values() for v in d.values()]
        vmin, vmax = min(all_vals), max(all_vals)
    else:
        vmin = vmax = None

    for ax, label in zip(axes, labels):
        plot_us_state_heatmap(
            solver_to_state_values[label],
            title=f"{title_prefix}{label}" if title_prefix else label,
            cmap_name=cmap_name,
            vmin=vmin, vmax=vmax,
            cbar_label=cbar_label,
            ax=ax,
            _gdf=gdf,
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Tile-grid fallback (kept for offline use / as a simple alternative)
# ---------------------------------------------------------------------------

US_STATE_TILE_GRID: dict[str, tuple[int, int]] = {
    "AK": (0, 0), "ME": (0, 10),
    "VT": (1, 9), "NH": (1, 10),
    "WA": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "WI": (2, 5),
    "MI": (2, 7), "NY": (2, 8), "MA": (2, 9), "RI": (2, 10),
    "OR": (3, 1), "ID": (3, 2), "SD": (3, 3), "IA": (3, 4), "IL": (3, 5),
    "IN": (3, 6), "OH": (3, 7), "PA": (3, 8), "NJ": (3, 9), "CT": (3, 10),
    "CA": (4, 1), "NV": (4, 2), "WY": (4, 3), "NE": (4, 4), "MO": (4, 5),
    "KY": (4, 6), "WV": (4, 7), "VA": (4, 8), "MD": (4, 9), "DE": (4, 10),
    "UT": (5, 2), "CO": (5, 3), "KS": (5, 4), "AR": (5, 5), "TN": (5, 6),
    "NC": (5, 7), "SC": (5, 8), "DC": (5, 9),
    "AZ": (6, 2), "NM": (6, 3), "OK": (6, 4), "LA": (6, 5), "MS": (6, 6),
    "AL": (6, 7), "GA": (6, 8),
    "HI": (7, 0), "TX": (7, 4), "FL": (7, 8),
    "PR": (8, 9),
}


def plot_us_state_heatmap_tile(
    state_to_value: dict[str, float],
    title: str = "Per-state value",
    cmap_name: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
    label_fmt: str = "{:.1f}",
    cbar_label: str = "loss",
    ax=None,
):
    """Render a US-state tile-grid heatmap. States are colored by their value;
    states with no value are shown in light grey.

    Args:
        state_to_value: {state_abbrev: numeric value}.
        cmap_name: Matplotlib colormap (default: red-yellow-green reversed so
            small = green = good, large = red = bad).
        vmin, vmax: Color scale limits. If None, inferred from data.
        label_fmt: Format string for the in-tile numeric label.
        cbar_label: Colorbar label.
        ax: Optional existing Axes to draw into.
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    values = list(state_to_value.values())
    if not values:
        ax.set_title(title + " (no data)")
        return ax
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcm.get_cmap(cmap_name)

    max_row = max(r for r, _ in US_STATE_TILE_GRID.values())
    max_col = max(c for _, c in US_STATE_TILE_GRID.values())

    for abbr, (r, c) in US_STATE_TILE_GRID.items():
        val = state_to_value.get(abbr)
        y = max_row - r  # invert so row 0 appears at top

        if val is None:
            color = "lightgrey"
            text_color = "dimgrey"
            label = abbr
        else:
            rgba = cmap(norm(val))
            color = rgba
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if lum < 0.5 else "black"
            label = f"{abbr}\n{label_fmt.format(val)}"

        ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                    facecolor=color,
                                    edgecolor="white", linewidth=1.5))
        ax.text(c + 0.5, y + 0.5, label,
                ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xlim(-0.3, max_col + 1.3)
    ax.set_ylim(-0.3, max_row + 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label=cbar_label)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return ax


def plot_us_state_heatmaps_grid_tile(
    solver_to_state_values: dict[str, dict[str, float]],
    title_prefix: str = "",
    cmap_name: str = "RdYlGn_r",
    shared_scale: bool = True,
    cbar_label: str = "per-state loss",
    label_fmt: str = "{:.1f}",
    save_path: str | None = None,
):
    """Tile-grid variant of plot_us_state_heatmaps_grid. No geopandas needed."""
    labels = list(solver_to_state_values.keys())
    n = len(labels)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    if shared_scale:
        all_vals = [v for d in solver_to_state_values.values() for v in d.values()]
        vmin, vmax = min(all_vals), max(all_vals)
    else:
        vmin = vmax = None

    for ax, label in zip(axes, labels):
        plot_us_state_heatmap_tile(
            solver_to_state_values[label],
            title=f"{title_prefix}{label}" if title_prefix else label,
            cmap_name=cmap_name,
            vmin=vmin, vmax=vmax,
            label_fmt=label_fmt,
            cbar_label=cbar_label,
            ax=ax,
        )

    for ax in axes[n:]:
        ax.set_visible(False)

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
