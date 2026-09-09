"""Contracts that prevent decision-time features from using future information."""

from __future__ import annotations

import pandas as pd


class PointInTimeViolation(ValueError):
    """Raised when a candidate feature was not available at decision time."""


def assert_available_at_decision_time(frame: pd.DataFrame) -> None:
    """Require each feature record to be available on or before its application."""
    required = {"application_id", "available_at", "application_timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing timing columns: {sorted(missing)}")

    available_at = pd.to_datetime(frame["available_at"], utc=True)
    application_timestamp = pd.to_datetime(frame["application_timestamp"], utc=True)
    invalid = available_at > application_timestamp
    if invalid.any():
        example_ids = frame.loc[invalid, "application_id"].head(3).tolist()
        raise PointInTimeViolation(
            "Feature records must be available at decision time; "
            f"violations include {example_ids}."
        )
