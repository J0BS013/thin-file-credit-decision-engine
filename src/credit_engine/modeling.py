"""Interpretable default scorecard with strictly out-of-time evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "bureau_score",
    "trailing_cashflow_amount",
    "cashflow_transaction_count",
]
CATEGORICAL_FEATURES = ["bureau_missing", "country_code"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class OutOfTimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_timestamp: pd.Timestamp


def split_out_of_time(data: pd.DataFrame, train_fraction: float = 0.75) -> OutOfTimeSplit:
    """Split on application time so validation occurs after all training decisions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    timestamps = pd.to_datetime(data["application_timestamp"], utc=True)
    unique_timestamps = sorted(timestamps.unique())
    if len(unique_timestamps) < 2:
        raise ValueError("Out-of-time validation requires at least two decision timestamps.")
    cutoff_index = min(max(int(len(unique_timestamps) * train_fraction), 1), len(unique_timestamps) - 1)
    cutoff = pd.Timestamp(unique_timestamps[cutoff_index])
    train = data.loc[timestamps < cutoff].copy()
    test = data.loc[timestamps >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Out-of-time split produced an empty partition.")
    return OutOfTimeSplit(train=train, test=test, cutoff_timestamp=cutoff)


def fit_logistic_scorecard(training_data: pd.DataFrame, target: str = "defaulted") -> Pipeline:
    """Fit a regularized, interpretable default model using decision-time features."""
    if target not in training_data:
        raise ValueError(f"Target column {target!r} is required.")
    missing = set(MODEL_FEATURES).difference(training_data.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    if training_data[target].nunique() < 2:
        raise ValueError("The target must contain both outcome classes.")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42)),
        ]
    ).fit(training_data[MODEL_FEATURES], training_data[target])


def evaluate_binary_model(model: Pipeline, validation_data: pd.DataFrame, target: str = "defaulted") -> dict[str, float]:
    """Evaluate discrimination and probability quality on an untouched time period."""
    if validation_data[target].nunique() < 2:
        raise ValueError("Validation data must contain both outcome classes.")
    probabilities = model.predict_proba(validation_data[MODEL_FEATURES])[:, 1]
    return evaluate_probabilities(validation_data[target], probabilities)


def evaluate_probabilities(outcomes: pd.Series, probabilities) -> dict[str, float]:
    """Score a probability vector without assuming a particular model implementation."""
    if outcomes.nunique() < 2:
        raise ValueError("Validation data must contain both outcome classes.")
    return {
        "roc_auc": float(roc_auc_score(outcomes, probabilities)),
        "pr_auc": float(average_precision_score(outcomes, probabilities)),
        "brier_score": float(brier_score_loss(outcomes, probabilities)),
    }
