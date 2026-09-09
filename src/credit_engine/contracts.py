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


def assert_event_happened_before_decision(frame: pd.DataFrame) -> None:
    """Reject future events even when a malformed source claims early availability."""
    required = {"application_id", "event_timestamp", "application_timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing timing columns: {sorted(missing)}")

    event_timestamp = pd.to_datetime(frame["event_timestamp"], utc=True)
    application_timestamp = pd.to_datetime(frame["application_timestamp"], utc=True)
    invalid = event_timestamp > application_timestamp
    if invalid.any():
        example_ids = frame.loc[invalid, "application_id"].head(3).tolist()
        raise PointInTimeViolation(
            "Feature events must happen on or before decision time; "
            f"violations include {example_ids}."
        )


def assert_no_label_columns(frame: pd.DataFrame) -> None:
    """Keep future repayment and fraud outcomes out of decision-time features."""
    forbidden = {"defaulted", "fraud_confirmed", "taken_up", "outcome_available_at"}.intersection(frame.columns)
    if forbidden:
        raise PointInTimeViolation(
            f"Decision-time feature data cannot contain outcome columns: {sorted(forbidden)}."
        )
