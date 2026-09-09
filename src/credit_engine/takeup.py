"""Take-up model: quantify the acceptance trade-off created by friction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TAKE_UP_NUMERIC_FEATURES = [
    "requested_amount",
    "document_requirement_count",
    "decision_time_minutes",
    "trailing_cashflow_amount",
]
TAKE_UP_CATEGORICAL_FEATURES = ["bureau_missing", "country_code"]
TAKE_UP_FEATURES = TAKE_UP_NUMERIC_FEATURES + TAKE_UP_CATEGORICAL_FEATURES


def fit_take_up_model(training_data: pd.DataFrame) -> Pipeline:
    """Fit an interpretable acceptance model; this is predictive, not causal."""
    missing = set(TAKE_UP_FEATURES + ["taken_up"]).difference(training_data.columns)
    if missing:
        raise ValueError(f"Missing take-up model fields: {sorted(missing)}")
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                TAKE_UP_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                TAKE_UP_CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1_000, random_state=42))]
    ).fit(training_data[TAKE_UP_FEATURES], training_data["taken_up"])
