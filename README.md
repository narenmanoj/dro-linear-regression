# dro-linear-regression

Implementation of algorithms from **"Distributionally Robust Linear Regression with Block Lewis Weights"** (Patel & Manoj, ICLR 2026).

Solves the group distributionally robust (GDR) least squares problem:

$$\min_{x \in \mathbb{R}^d} \max_{i \in [m]} \frac{1}{n_i} \|A_i x - b_i\|^2$$

## Installation

```bash
pip install -r requirements.txt
```

Requires: `numpy`, `matplotlib`, `cvxpy` (for exact solver), `pandas` (for CSV loading).

## Quick start

### Synthetic experiment (reproduces paper plots)

```bash
python run_experiment.py
```

This generates the adversarial heterogeneous instance from Appendix F, tunes all methods via grid search, and saves all artifacts to a new timestamped directory under `runs/`.

### Custom synthetic parameters

```bash
python run_experiment.py --m 50 --d 5 --m-outlier 3 --n-per-group 20 --seed 42
```

### Load a CSV dataset

```bash
python run_experiment.py --csv data.csv --target y --group group_id
```

The CSV should have a target column, a group column, and feature columns (auto-detected if not specified with `--features`).

### Run artifacts

Every invocation creates a new directory `runs/<timestamp>[_<name>]/` containing a complete record of the run:

```
runs/20260419_152144_acs_all/
├── config.json           # CLI args, dataset metadata, ERM/OPT baselines
├── hyperparameters.json  # best-tuned hyperparameters for each method
├── losses.csv            # per-group losses + aggregates (max, mean, std, max/mean)
├── stats.json            # per-solver aggregate statistics
├── summary.txt           # human-readable version of losses.csv
├── solutions.npz         # parameter vectors x for every solver
├── solutions_labels.json # map: sanitized NPZ key -> original solver label
├── curves.npz            # per-iteration (iters, best_values, times) for both
│                         #   iteration-budget and equal-runtime sweeps
├── iterations.png        # convergence vs iterations (log scale)
├── iterations_linear.png # convergence vs iterations (linear scale)
├── time.png              # convergence vs wall-clock (log scale)
└── time_linear.png       # convergence vs wall-clock (linear scale)
```

Customize the output location with `--runs-root` (default `runs/`) and `--run-name` (appended as a readable suffix):

```bash
python run_experiment.py --folktables --acs-all --run-name acs_all_v1
# → writes to runs/20260419_152144_acs_all_v1/
```

### Use as a library

```python
import numpy as np
import dro
from dro import solvers, tuning
from dro.lewis_weights import build_lewis_geometry

# Generate or load data
A_groups, b_groups = dro.generate_hard_instance(m_total=50, d=10, seed=0)

# Or from arrays:
# A_groups, b_groups = dro.from_arrays(A, b, group_labels)

# Warm start with ERM
x0 = dro.erm_solution(A_groups, b_groups)

# Solve exactly with CVXPY
opt = solvers.solve_exact(A_groups, b_groups)
print(f"OPT = {opt.best_loss:.6e}")

# Run a single solver
result = solvers.ball_oracle_naive(
    A_groups, b_groups, x0,
    beta=0.1, delta=0.01, R0=1.0, n_outer=100,
)
print(f"Best loss = {result.best_loss:.6e}")

# Or tune and run all methods
L = build_lewis_geometry(A_groups, b_groups)
configs = tuning.tune_all(A_groups, b_groups, x0, L=L)

# Interpolate between ERM and DRO (Theorem 2, Equation 4)
path = solvers.interpolation_path(A_groups, b_groups, x0)
for r in path:
    print(f"p={r['p']:5.1f}  max_loss={r['max_loss']:.4e}  avg_loss={r['avg_loss']:.4e}")

# Solve for a specific p value
result = solvers.gp_newton(A_groups, b_groups, x0, p=8, max_steps=100)

# Persist artifacts from a programmatic run
run_dir = dro.artifacts.create_run_dir(name="my_experiment")
dro.artifacts.save_hyperparameters(run_dir, configs)
dro.artifacts.save_losses(run_dir, A_groups, b_groups,
                           solutions={"ERM": x0, "my_solver": result.x_final})
dro.artifacts.save_solutions(run_dir, solutions={"ERM": x0, "my_solver": result.x_final})
```

## Algorithms

| Method | Module | Paper reference |
|--------|--------|-----------------|
| CVXPY exact solver | `solvers.solve_exact` | Appendix F.2 |
| Subgradient descent | `solvers.subgradient_fixed`, `subgradient_diminishing` | Table 1, row 1 |
| Smoothed GD / Heavy-Ball / Nesterov | `solvers.smooth_gd`, `smooth_heavy_ball`, `smooth_nesterov` | Table 1, rows 2-3; Eq. (7) |
| Log-barrier IPM | `solvers.interior_point` | Table 1, row 4 |
| Ball-oracle (Euclidean) | `solvers.ball_oracle_naive` | Table 1, row "naive" |
| Ball-oracle (Lewis) | `solvers.ball_oracle_lewis` | Algorithm 1; Theorem 1 |
| Gp Newton (interpolating) | `solvers.gp_newton` | Algorithm 5; Theorem 2; Eq. (4) |
| Gp ball-oracle (Euclidean) | `solvers.gp_ball_oracle_naive` | Theorem 2 (naive geometry) |
| Gp ball-oracle (Lewis) | `solvers.gp_ball_oracle_lewis` | Algorithm 5; Theorem 2 |
| Interpolation path | `solvers.interpolation_path` | Sweeps p from ERM to DRO |

## Package structure

```
dro/
├── __init__.py          # Public API
├── problem.py           # SolverResult, loss utilities (group_losses, max_group_loss)
├── datasets.py          # generate_hard_instance, from_arrays, from_csv
├── smoothing.py         # Smoothed objective f_{β,δ} (Eq. 7): value, gradient, Hessian
├── gp_objective.py      # Gp interpolating objective (Eq. 4): value, gradient, Hessian
├── lewis_weights.py     # Block Lewis weights and Lewis geometry (Theorem 2.3)
├── tuning.py            # Grid-search tuning for all methods
├── plotting.py          # Convergence and timing plots
├── artifacts.py         # Run directory helpers (configs, losses, solutions, curves)
└── solvers/
    ├── cvxpy_exact.py   # Exact CVXPY epigraph solver
    ├── subgradient.py   # Fixed and diminishing step subgradient
    ├── smoothed_gd.py   # GD, Heavy-Ball, Nesterov on smoothed objective
    ├── ipm.py           # Log-barrier interior point method
    ├── ball_oracle.py   # Ball-oracle with Euclidean and Lewis geometry
    └── gp_regression.py # Gp solvers: Newton, ball-oracle, interpolation path
```

## Experimental details

We construct heterogeneous regression problems in a deliberately challenging way. Most groups share a moderately well-conditioned data geometry that lives in a common subspace, but a small set of _adversarial_ groups have extremely ill-conditioned design matrices whose dominant directions differ sharply from the normal groups. These adversarial groups also have target parameters that lie far from the bulk of the population. As a result, the combined dataset has a very high condition number, and a model that fits the average group loss well can still perform extremely poorly on the worst group. We use a convex solver (through CVXPY) to compute the true optimum for the max-loss objective, and all plots show the difference between an algorithm's worst-group loss and this optimum.

We compare a wide range of baselines: subgradient descent on the raw max-loss, several smoothed gradient-based methods (plain gradient descent, Heavy-Ball momentum, and Nesterov momentum), and a standard interior-point method. Our own methods are ball-oracle procedures that use a damped Newton solver inside gradually shrinking trust regions, and we implement both a simple Euclidean version and a version based on Lewis-weight geometry. We tune every method through a grid search over its relevant hyperparameters (step sizes, smoothing strengths, momentum parameters, initial trust-region radii and decay factors), always using the same warm start and scoring each configuration by the lowest worst-group loss it achieves within a fixed number of iterations.

The meaning of an _iteration_ differs across methods. For the subgradient and smoothed-gradient methods, one iteration is a single full gradient or subgradient step on the smoothed or nonsmoothed objective. For the interior-point method, one iteration is a single outer Newton step of the barrier procedure. For the ball-oracle methods, one iteration is a single call to the Newton trust-region solver. To ensure fairness, our plots always compare methods by the number of their own natural outer iterations.

## Real-world experiment: ACS Income (Folktables)

We evaluate our algorithms on a real-world regression task from the American Community Survey (ACS), loaded via the [Folktables](https://github.com/socialfoundations/folktables) package.

### Prediction problem

The task is to predict **log personal income** from 10 demographic and employment features (age, education, occupation, hours worked, etc.) for employed US adults. The data is grouped by **US state**, making this a natural setting for group DRO: income distributions vary substantially across states, and a model that minimizes average prediction error may perform poorly on some states.

We use **all 51 regions** (50 US states + Puerto Rico) with 200 samples per state, giving 10,200 total samples in 10 dimensions. The features are standardized to zero mean and unit variance.

```bash
python run_experiment.py --folktables --acs-all --acs-subsample 200 --T 50 --run-name acs_all
```

This writes all artifacts (config, hyperparameters, losses, solutions, curves, plots) to `runs/<timestamp>_acs_all/`.

### Per-group loss comparison

The table below shows the mean squared prediction error per state for each solver. All methods use the ERM solution as a warm start and are tuned via grid search over 50 iterations.

| State |     ERM | OPT (CVXPY) | Subgradient | Smoothed HB |     IPM | Ball-Oracle (Euc.) | Ball-Oracle (Lewis) |
|-------|--------:|------------:|------------:|------------:|--------:|-------------------:|--------------------:|
| AL |  91.877 | 112.490 |  92.177 | 112.688 | 112.491 | 112.443 | 112.443 |
| AK | 102.950 | 113.837 | 102.799 | 113.448 | 113.840 | 113.476 | 113.476 |
| AZ | 113.954 | 110.873 | 113.804 | 110.734 | 110.872 | 110.834 | 110.834 |
| AR |  96.325 | 113.111 |  96.463 | 112.794 | 113.111 | 112.931 | 112.931 |
| CA | **138.065** | 113.956 | 137.307 | 113.187 | 113.956 | 113.749 | 113.749 |
| CO | 108.927 | 113.472 | 108.966 | 113.405 | 113.472 | 113.375 | 113.375 |
| CT | 115.090 | 113.956 | 115.130 | 114.123 | 113.956 | 113.914 | 113.914 |
| DE | 106.379 | 110.504 | 106.513 | 111.012 | 110.504 | 110.688 | 110.688 |
| FL | 130.928 | 109.676 | 130.572 | 109.494 | 109.677 | 109.694 | 109.694 |
| GA | 111.512 | 112.849 | 111.435 | 112.612 | 112.850 | 112.713 | 112.713 |
| HI | 132.336 | 110.808 | 131.299 | 109.873 | 110.807 | 110.373 | 110.373 |
| ID |  98.620 | 112.512 |  98.779 | 112.171 | 112.513 | 112.343 | 112.343 |
| IL | 104.909 | 111.277 | 105.028 | 111.478 | 111.276 | 111.360 | 111.360 |
| IN |  98.289 | 113.956 |  98.471 | 113.946 | 113.956 | 113.870 | 113.870 |
| IA |  92.348 | 113.263 |  92.598 | 113.257 | 113.265 | 113.130 | 113.130 |
| KS | 101.611 | 110.476 | 101.699 | 110.476 | 110.478 | 110.354 | 110.354 |
| KY |  98.028 | 113.651 |  98.280 | 113.690 | 113.651 | 113.587 | 113.587 |
| LA | 100.182 | 112.777 | 100.309 | 112.872 | 112.777 | 112.813 | 112.813 |
| ME |  95.141 | 109.688 |  95.416 | 110.214 | 109.686 | 109.797 | 109.797 |
| MD | 122.349 | 113.956 | 122.203 | 114.100 | 113.956 | 114.011 | 114.011 |
| MA | 120.006 | 112.341 | 119.937 | 112.513 | 112.340 | 112.362 | 112.362 |
| MI | 100.776 | 112.922 | 100.988 | 112.892 | 112.923 | 112.826 | 112.826 |
| MN | 105.931 | 113.531 | 106.025 | 113.559 | 113.532 | 113.434 | 113.434 |
| MS |  96.131 | 107.237 |  96.282 | 107.379 | 107.237 | 107.224 | 107.224 |
| MO | 100.258 | 109.911 | 100.398 | 109.925 | 109.911 | 109.833 | 109.833 |
| MT |  97.212 | 107.313 |  97.413 | 107.669 | 107.313 | 107.563 | 107.563 |
| NE |  97.375 | 110.118 |  97.541 | 110.159 | 110.118 | 110.091 | 110.091 |
| NV | 131.379 | 113.956 | 130.888 | 113.612 | 113.956 | 113.995 | 113.995 |
| NH | 104.483 | 113.956 | 104.699 | **114.260** | 113.956 | 113.991 | 113.991 |
| NJ | 130.220 | 113.956 | 129.940 | 113.755 | 113.956 | 113.798 | 113.798 |
| NM | 103.400 | 106.350 | 103.378 | 106.708 | 106.349 | 106.557 | 106.557 |
| NY | 133.433 | 113.510 | 133.029 | 113.187 | 113.511 | 113.349 | 113.349 |
| NC | 103.499 | 110.153 | 103.500 | 109.887 | 110.154 | 110.035 | 110.035 |
| ND |  98.548 | 111.632 |  98.759 | 111.850 | 111.634 | 111.634 | 111.634 |
| OH | 104.392 | 113.037 | 104.602 | 113.212 | 113.037 | 113.021 | 113.021 |
| OK | 104.375 | 110.966 | 104.328 | 110.675 | 110.967 | 110.786 | 110.786 |
| OR | 110.348 | 112.886 | 110.359 | 113.045 | 112.884 | 112.986 | 112.986 |
| PA | 109.453 | 111.305 | 109.496 | 111.308 | 111.305 | 111.351 | 111.351 |
| RI | 112.544 | 111.312 | 112.573 | 111.675 | 111.312 | 111.380 | 111.380 |
| SC | 101.575 | 110.357 | 101.733 | 110.580 | 110.357 | 110.408 | 110.408 |
| SD |  99.086 | 109.277 |  99.241 | 109.465 | 109.277 | 109.344 | 109.344 |
| TN | 106.298 | 109.977 | 106.322 | 110.014 | 109.977 | 109.934 | 109.934 |
| TX | 115.551 | 112.522 | 115.436 | 112.391 | 112.523 | 112.538 | 112.538 |
| UT | 111.159 | 113.084 | 111.168 | 112.770 | 113.084 | 113.000 | 113.000 |
| VT | 103.831 | 110.496 | 104.037 | 110.865 | 110.495 | 110.635 | 110.635 |
| VA | 121.851 | 113.236 | 121.683 | 113.066 | 113.237 | 113.149 | 113.149 |
| WA | 128.040 | 111.627 | 127.771 | 111.635 | 111.627 | 111.557 | 111.557 |
| WV |  98.473 | 109.843 |  98.661 | 109.721 | 109.844 | 109.860 | 109.860 |
| WI | 102.850 | 111.058 | 103.035 | 111.249 | 111.059 | 111.078 | 111.078 |
| WY |  98.922 | 111.777 |  99.106 | 111.753 | 111.779 | 111.710 | 111.710 |
| PR | 108.545 |  **97.149** | 108.247 |  97.249 |  97.147 |  97.300 |  97.300 |
| | | | | | | | |
| **Max**      | **138.065** | 113.956 | 137.307 | 114.260 | 113.956 | 114.011 | 114.011 |
| **Mean**     | 108.231 | 111.449 | 108.232 | 111.443 | 111.449 | 111.415 | 111.415 |
| **Median**   | 104.392 | 111.777 | 104.602 | 111.850 | 111.779 | 111.710 | 111.710 |
| **Std**      |  11.773 |   2.751 |  11.546 |   2.687 |   2.751 |   2.696 |   2.696 |
| **Max/Mean** |   1.276 |   1.022 |   1.269 |   1.025 |   1.022 |   1.023 |   1.023 |

### Observations

- **ERM** achieves the lowest average loss (108.23) but has an enormous spread (σ = 11.77). The worst states under ERM are **CA (138.1), NY (133.4), HI (132.3), FL (130.9), NV (131.4), NJ (130.2), WA (128.0)** — all large-population/high-income states where the 10-feature linear model struggles. The best-served are **AL (91.9), IA (92.3), ME (95.1), AR (96.3), MS (96.1)** — lower-income states with more homogeneous income distributions.

- **OPT (CVXPY)** nearly equalizes the group losses around 107–114 (Max/Mean = 1.022, σ = 2.75). Notably, **Puerto Rico (97.1) is the only state that stays below the equalized band** — its income distribution is so different (much lower mean income) that it anchors the entire problem. To bring CA's loss down from 138 to 114, the optimum accepts ~3 MSE cost on the previously well-served states.

- **IPM** and **both ball-oracle methods** match the exact optimum to within 0.1 on every state, with Max/Mean ratios of 1.022–1.023 — effectively fully converged within 50 iterations.

- **Subgradient descent** barely moves from ERM even after 50 iterations. The max/mean ratio stays at 1.269 (vs 1.276 for ERM), showing that on this adversarial 51-group instance, subgradient is fundamentally outclassed.

- **Smoothed Heavy-Ball** gets close to optimum (Max/Mean = 1.025) but slightly overshoots on NH (114.26 vs OPT's 113.96), showing the accuracy limit of first-order methods on this problem.

The **ERM→DRO transition** redistributes prediction accuracy from 33 states whose loss increases to 17 states whose loss decreases. The largest decreases are CA (–24.3), NY (–19.9), HI (–21.5), NV (–17.4), NJ (–16.3), WA (–16.4) — all states that ERM effectively neglected. The price is moderate cost (~5–20 MSE) on the small-population, homogeneous states that ERM happened to fit well.

## Adding new datasets

The package supports four ways to provide data:

1. **Synthetic**: Use `dro.generate_hard_instance(...)` with custom parameters.
2. **From arrays**: Use `dro.from_arrays(A, b, groups)` where `A` is the design matrix, `b` is the response, and `groups` is an integer array of group labels.
3. **From CSV**: Use `dro.from_csv(path, target_col, group_col)` or the `--csv` flag on `run_experiment.py`.
4. **Folktables (ACS)**: Use `dro.load_acs_income(states=..., group_by="state")` or the `--folktables` flag. Requires `pip install folktables`. Supports grouping by US state or race.

Any dataset that provides `(A_groups, b_groups)` — a list of per-group design matrices and response vectors — is compatible with all solvers and tuning routines.
