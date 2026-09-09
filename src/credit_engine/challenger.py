"""XGBoost challenger and time-respecting probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from credit_engine.modeling import MODEL_FEATURES, split_out_of_time


@dataclass(frozen=True)
class CalibratedChallenger:
    model: XGBClassifier
    calibrator: LogisticRegression
    feature_columns: list[str]
    calibration_cutoff: pd.Timestamp

    def predict_proba(self, data: pd.DataFrame) -> list[float]:
        raw_probability = self.model.predict_proba(_prepare_features(data, self.feature_columns))[:, 1]
        return self.calibrator.predict_proba(pd.DataFrame({"raw_probability": raw_probability}))[:, 1]


def _prepare_features(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Encode fixed, decision-time inputs for a tree-based model."""
    features = data[MODEL_FEATURES].copy()
    features["bureau_score"] = features["bureau_score"].fillna(-1)
    features["bureau_missing"] = features["bureau_missing"].astype(int)
    encoded = pd.get_dummies(features, columns=["country_code"], dtype=int)
    if columns is not None:
        return encoded.reindex(columns=columns, fill_value=0)
    return encoded


def fit_calibrated_xgboost(training_data: pd.DataFrame, target: str = "defaulted") -> CalibratedChallenger:
    """Fit on early data and calibrate on a later pre-holdout period."""
    split = split_out_of_time(training_data, train_fraction=0.8)
    fit_features = _prepare_features(split.train)
    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    ).fit(fit_features, split.train[target])
    calibration_features = _prepare_features(split.test, fit_features.columns.tolist())
    raw_probability = model.predict_proba(calibration_features)[:, 1]
    calibrator = LogisticRegression(random_state=42).fit(
        pd.DataFrame({"raw_probability": raw_probability}), split.test[target]
    )
    return CalibratedChallenger(
        model=model,
        calibrator=calibrator,
        feature_columns=fit_features.columns.tolist(),
        calibration_cutoff=split.cutoff_timestamp,
    )
