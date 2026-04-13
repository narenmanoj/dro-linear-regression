"""Solver implementations for group DRO regression."""

from .cvxpy_exact import solve_exact
from .subgradient import subgradient_fixed, subgradient_diminishing
from .smoothed_gd import smooth_gd, smooth_heavy_ball, smooth_nesterov
from .ipm import interior_point
from .ball_oracle import ball_oracle_naive, ball_oracle_lewis
from .gp_regression import (
    gp_newton,
    gp_ball_oracle_naive,
    gp_ball_oracle_lewis,
    interpolation_path,
)

__all__ = [
    "solve_exact",
    "subgradient_fixed",
    "subgradient_diminishing",
    "smooth_gd",
    "smooth_heavy_ball",
    "smooth_nesterov",
    "interior_point",
    "ball_oracle_naive",
    "ball_oracle_lewis",
    "gp_newton",
    "gp_ball_oracle_naive",
    "gp_ball_oracle_lewis",
    "interpolation_path",
]
