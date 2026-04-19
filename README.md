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

This generates the adversarial heterogeneous instance from Appendix F, tunes all methods via grid search, and saves convergence plots to `iterations.png` and `time.png`.

### Custom synthetic parameters

```bash
python run_experiment.py --m 50 --d 5 --m-outlier 3 --n-per-group 20 --seed 42
```

### Load a CSV dataset

```bash
python run_experiment.py --csv data.csv --target y --group group_id
```

The CSV should have a target column, a group column, and feature columns (auto-detected if not specified with `--features`).

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

We use 5 states (CA, TX, NY, FL, IL) with 200 samples per state, giving 1000 total samples in 10 dimensions. The features are standardized to zero mean and unit variance.

```bash
python run_experiment.py --folktables --acs-states CA TX NY FL IL --acs-subsample 200 --T 20
```

### Per-group loss comparison

The table below shows the mean squared error per group (state) for each solver, along with aggregate statistics. All methods use the ERM solution as a warm start and are tuned via grid search over 20 iterations.

| Group |   n |     ERM | OPT (CVXPY) | Subgradient | Smoothed HB |     IPM | Ball-Oracle (Euc.) | Ball-Oracle (Lewis) |
|-------|-----|---------|-------------|-------------|-------------|---------|--------------------:|--------------------:|
| CA    | 200 | 116.048 |     111.123 |     115.955 |     111.088 | 111.123 |            111.130  |            111.120  |
| FL    | 200 | 107.737 |     110.533 |     107.765 |     110.152 | 110.533 |            110.447  |            110.538  |
| IL    | 200 | 107.781 |     111.123 |     107.828 |     111.035 | 111.123 |            111.059  |            111.066  |
| NY    | 200 | 112.438 |     111.123 |     112.430 |     111.226 | 111.123 |            111.147  |            111.150  |
| TX    | 200 | 108.680 |     111.123 |     108.706 |     111.137 | 111.123 |            111.126  |            111.126  |
| | | | | | | | | |
| **Max**  |  | 116.048 | 111.123 | 115.955 | 111.226 | 111.123 | 111.147 | 111.150 |
| **Mean** |  | 110.537 | 111.005 | 110.537 | 110.928 | 111.005 | 110.982 | 111.000 |
| **Std**  |  |   3.252 |   0.236 |   3.204 |   0.393 |   0.236 |   0.269 |   0.233 |
| **Max/Mean** | | 1.050 | 1.001 | 1.049 | 1.003 | 1.001 | 1.001 | 1.001 |

### Observations

- **ERM** achieves the lowest average loss (110.54) but has the highest worst-group loss (116.05 for CA). The max/mean ratio of 1.050 means the worst state's error is 5% higher than average.

- **OPT (CVXPY)** equalizes all group losses to ~111.12, with a max/mean ratio of 1.001. It sacrifices ~0.5 on average loss to bring the worst group down by ~5. This is the optimal tradeoff — the price of fairness.

- **IPM** and **both ball-oracle methods** closely match the exact optimum within 20 iterations, achieving max/mean ratios of ~1.001. The per-group losses are nearly equalized across states.

- **Subgradient descent** barely moves from ERM within 20 iterations — the max/mean ratio remains at 1.049.

- **Smoothed Heavy-Ball** makes significant progress but slightly overshoots on NY (111.23 > 111.12), leaving a small gap.

The key takeaway: on real census data, the robust (DRO) solution successfully redistributes prediction accuracy from well-served states (FL, IL, TX) to the worst-served state (CA), at a modest cost to average performance.

## Adding new datasets

The package supports four ways to provide data:

1. **Synthetic**: Use `dro.generate_hard_instance(...)` with custom parameters.
2. **From arrays**: Use `dro.from_arrays(A, b, groups)` where `A` is the design matrix, `b` is the response, and `groups` is an integer array of group labels.
3. **From CSV**: Use `dro.from_csv(path, target_col, group_col)` or the `--csv` flag on `run_experiment.py`.
4. **Folktables (ACS)**: Use `dro.load_acs_income(states=..., group_by="state")` or the `--folktables` flag. Requires `pip install folktables`. Supports grouping by US state or race.

Any dataset that provides `(A_groups, b_groups)` — a list of per-group design matrices and response vectors — is compatible with all solvers and tuning routines.
