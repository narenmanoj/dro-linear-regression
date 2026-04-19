"""DRO Linear Regression: Distributionally Robust Linear Regression with Block Lewis Weights.

Implementation of algorithms from Patel & Manoj (ICLR 2026).
"""

from .problem import SolverResult, group_losses, max_group_loss, erm_solution, summarize
from .datasets import generate_hard_instance, from_arrays, from_csv
from .datasets_folktables import load_acs_income
from .lewis_weights import build_lewis_geometry
from .smoothing import smooth_value, smooth_value_and_grad, smooth_grad_and_hessian
from .gp_objective import gp_value, gp_power_value
from . import solvers
from . import tuning
from . import plotting
from . import artifacts

__all__ = [
    "SolverResult",
    "group_losses",
    "max_group_loss",
    "erm_solution",
    "summarize",
    "generate_hard_instance",
    "from_arrays",
    "from_csv",
    "load_acs_income",
    "build_lewis_geometry",
    "smooth_value",
    "smooth_value_and_grad",
    "smooth_grad_and_hessian",
    "gp_value",
    "gp_power_value",
    "solvers",
    "tuning",
    "plotting",
    "artifacts",
]
