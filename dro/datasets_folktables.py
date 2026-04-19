"""ACS income regression dataset via Folktables.

Loads US Census (American Community Survey) data for a regression task:
predicting log personal income from demographic and employment features,
grouped by US state. This is a natural setting for group DRO — different
states have heterogeneous income distributions, and a model that minimizes
average error may perform poorly on some states.

Requires: pip install folktables

Reference:
    Ding, Hartford, Leqi, Vieira, "Retiring Adult: New Datasets for Fair
    Machine Learning", NeurIPS 2021.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# The feature set from ACSIncome (Folktables), good for income regression.
ACS_INCOME_FEATURES = [
    "AGEP",   # Age
    "COW",    # Class of worker
    "SCHL",   # Educational attainment
    "MAR",    # Marital status
    "OCCP",   # Occupation code
    "POBP",   # Place of birth
    "RELP",   # Relationship to householder
    "WKHP",   # Hours worked per week
    "SEX",    # Sex
    "RAC1P",  # Race
]

# The 10 most populous US states (good default for heterogeneous groups).
DEFAULT_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

# All 50 US states + DC + PR.
ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR",
]

# FIPS codes to state abbreviations for readable group names.
_FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL",
    18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND",
    39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY", 72: "PR",
}


def load_acs_income(
    states: list[str] | None = None,
    survey_year: str = "2018",
    group_by: str = "state",
    log_income: bool = True,
    standardize: bool = True,
    min_group_size: int = 50,
    subsample: int | None = None,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Load ACS income regression data grouped by US state (or race).

    Filters to employed adults with positive income, uses standard ACSIncome
    features, and predicts (log) personal income.

    Args:
        states: List of state abbreviations to include.
            Defaults to the 10 most populous states.
        survey_year: ACS survey year (e.g., "2018", "2019").
        group_by: How to define groups. Options:
            "state" — group by US state (FIPS code). Natural for DRO since
                      income distributions vary substantially across states.
            "race"  — group by race (RAC1P). Common fairness benchmark.
        log_income: If True, predict log(1 + income) instead of raw income.
            Log-transform reduces skew and makes regression better-conditioned.
        standardize: If True, standardize features to zero mean, unit variance.
        min_group_size: Drop groups with fewer than this many samples.
        subsample: If set, subsample this many rows per group (for speed).
        seed: Random seed for subsampling.

    Returns:
        (A_groups, b_groups, info) where:
            A_groups[i]: design matrix for group i, shape (n_i, d).
            b_groups[i]: response vector for group i, shape (n_i,).
            info: dict with metadata:
                "group_names": list of group labels (state abbrevs or race codes).
                "feature_names": list of feature column names.
                "n_groups": number of groups.
                "n_total": total number of samples.
                "d": feature dimension.
    """
    try:
        from folktables import ACSDataSource
    except ImportError:
        raise ImportError(
            "folktables is required for ACS data. Install with: pip install folktables"
        )

    if states is None:
        states = DEFAULT_STATES

    # Download ACS data.
    data_source = ACSDataSource(
        survey_year=survey_year, horizon="1-Year", survey="person"
    )
    data = data_source.get_data(states=states, download=True)

    # Filter to employed adults with positive income.
    df = data.copy()
    df = df[df["AGEP"] >= 18]                    # adults
    df = df[df["ESR"].isin([1, 2, 4, 5])]        # employed (civilian or armed forces)
    df = df[df["PINCP"] > 0]                      # positive income
    df = df.dropna(subset=ACS_INCOME_FEATURES + ["PINCP", "ST"])

    # Features and target.
    X = df[ACS_INCOME_FEATURES].values.astype(np.float64)
    if log_income:
        y = np.log1p(df["PINCP"].values.astype(np.float64))
    else:
        y = df["PINCP"].values.astype(np.float64)

    # Group labels.
    if group_by == "state":
        group_codes = df["ST"].values.astype(int)
        code_to_name = _FIPS_TO_STATE
    elif group_by == "race":
        group_codes = df["RAC1P"].values.astype(int)
        code_to_name = {
            1: "White", 2: "Black", 3: "AIAN", 4: "Alaska Native",
            5: "AIAN+", 6: "Asian", 7: "NHPI", 8: "Other", 9: "Two+",
        }
    else:
        raise ValueError(f"Unknown group_by={group_by!r}. Use 'state' or 'race'.")

    # Standardize features.
    if standardize:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-10] = 1.0
        X = (X - mu) / sigma

    # Split by group.
    rng = np.random.default_rng(seed)
    unique_codes = sorted(set(group_codes))

    A_groups = []
    b_groups = []
    group_names = []

    for code in unique_codes:
        mask = group_codes == code
        X_g, y_g = X[mask], y[mask]

        if len(y_g) < min_group_size:
            continue

        if subsample is not None and len(y_g) > subsample:
            idx = rng.choice(len(y_g), size=subsample, replace=False)
            X_g, y_g = X_g[idx], y_g[idx]

        A_groups.append(X_g)
        b_groups.append(y_g)
        group_names.append(code_to_name.get(code, str(code)))

    info = {
        "group_names": group_names,
        "feature_names": list(ACS_INCOME_FEATURES),
        "n_groups": len(A_groups),
        "n_total": sum(A.shape[0] for A in A_groups),
        "d": X.shape[1],
    }

    return A_groups, b_groups, info
