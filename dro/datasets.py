"""Dataset generation and loading for group DRO regression.

Provides:
  - generate_hard_instance: The adversarial synthetic dataset from Appendix F.
  - from_arrays: Build group data from a design matrix + response + group labels.
  - from_csv: Load group data from a CSV file.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def generate_hard_instance(
    m_total: int = 20,
    m_outlier: int = 3,
    d: int = 20,
    n_per_group: int = 200,
    noise_std_normal: float = 0.1,
    noise_std_outlier: float = 1e-3,
    cond_normal: float = 10.0,
    cond_outlier: float = 1e8,
    spread_normal: float = 3.0,
    spread_outlier: float = 60.0,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate an adversarial heterogeneous regression instance.

    Constructs m_total groups where most are benign (well-conditioned, clustered
    near a common center) and m_outlier groups are adversarial (extremely
    ill-conditioned along distinct directions, with far-away optima and tiny noise).

    This is the construction from Appendix F.1 of Patel & Manoj (ICLR 2026).

    Returns:
        (A_groups, b_groups) where A_groups[i] has shape (n_per_group, d)
        and b_groups[i] has shape (n_per_group,).
    """
    rng = np.random.default_rng(seed)
    m_normal = m_total - m_outlier
    assert m_normal > 0 and m_outlier > 0

    # Shared orthonormal basis for controlling curvature direction-by-direction.
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))

    A_groups: list[np.ndarray] = []
    b_groups: list[np.ndarray] = []

    # Common reference center in parameter space.
    x_center = rng.standard_normal(d)

    # --- Normal groups: moderate condition, clustered optima ---
    for _ in range(m_normal):
        logs = rng.uniform(0.0, math.log10(cond_normal), size=d)
        lambdas = 10.0 ** logs
        Sigma = U @ np.diag(lambdas) @ U.T
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))

        Z = rng.standard_normal((n_per_group, d))
        A_i = (Z @ L.T) / math.sqrt(d)

        x_i_star = x_center + spread_normal * rng.standard_normal(d)
        noise = noise_std_normal * rng.standard_normal(n_per_group)
        b_i = A_i @ x_i_star + noise

        A_groups.append(A_i)
        b_groups.append(b_i)

    # --- Outlier groups: extreme curvature along distinct directions ---
    # When m_outlier > d, we cycle through directions (multiple outliers share
    # a direction but with independent sampling, noise, and target offsets).
    for j in range(m_outlier):
        idx = (d - 1 - j) % d
        lambdas = np.ones(d)
        lambdas[idx] = cond_outlier

        # Mild variability on other coordinates.
        logs = rng.uniform(0.0, 1.0, size=d)
        for t in range(d):
            if t != idx:
                lambdas[t] *= 10.0 ** (0.2 * logs[t])

        Sigma = U @ np.diag(lambdas) @ U.T
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))

        Z = rng.standard_normal((n_per_group, d))
        A_i = (Z @ L.T) / math.sqrt(d)

        v = U[:, idx]
        x_i_star = x_center + spread_outlier * v
        noise = noise_std_outlier * rng.standard_normal(n_per_group)
        b_i = A_i @ x_i_star + noise

        A_groups.append(A_i)
        b_groups.append(b_i)

    return A_groups, b_groups


def from_arrays(
    A: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build group data from a single design matrix, response vector, and group labels.

    Args:
        A: Design matrix of shape (n, d).
        b: Response vector of shape (n,).
        groups: Integer group labels of shape (n,), values in {0, ..., m-1}.

    Returns:
        (A_groups, b_groups) split by group.
    """
    unique_groups = np.unique(groups)
    A_groups = []
    b_groups = []
    for g in unique_groups:
        mask = groups == g
        A_groups.append(A[mask])
        b_groups.append(b[mask])
    return A_groups, b_groups


def from_csv(
    path: str,
    target_col: str,
    group_col: str,
    feature_cols: Optional[list[str]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load group regression data from a CSV file.

    Args:
        path: Path to the CSV file.
        target_col: Name of the response/target column.
        group_col: Name of the column containing group labels.
        feature_cols: Names of feature columns. If None, uses all columns
                      except target_col and group_col.

    Returns:
        (A_groups, b_groups) split by group.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (target_col, group_col)]

    A = df[feature_cols].values.astype(float)
    b = df[target_col].values.astype(float)
    groups = df[group_col].values

    # Encode group labels as integers.
    _, group_ints = np.unique(groups, return_inverse=True)
    return from_arrays(A, b, group_ints)
