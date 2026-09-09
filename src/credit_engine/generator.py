"""Deterministic synthetic fixtures for thin-file credit decisioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    application_count: int


PROFILES = {
    "smoke": GenerationProfile(name="smoke", application_count=1_000),
    "full": GenerationProfile(name="full", application_count=50_000),
}
COUNTRIES = (("BR", "BRL"), ("MX", "MXN"), ("PH", "PHP"))


def generate_synthetic_data(profile: str = "smoke", seed: int = 20260909) -> dict[str, pd.DataFrame]:
    """Generate source tables with decision-time and future-outcome timestamps."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {sorted(PROFILES)}.")
    rng = np.random.default_rng(seed)
    count = PROFILES[profile].application_count
    application_ids = np.array([f"app_{index:06d}" for index in range(count)])
    applicant_ids = np.array([f"person_{index:06d}" for index in range(count)])
    application_timestamp = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 365, size=count), unit="D"
    )
    country_index = rng.integers(0, len(COUNTRIES), size=count)
    countries = np.array([COUNTRIES[index][0] for index in country_index])
    currencies = np.array([COUNTRIES[index][1] for index in country_index])
    thin_file = rng.random(count) < 0.45
    latent_risk = np.clip(rng.beta(2.2, 7.0, size=count) + thin_file * 0.05, 0.01, 0.85)

    applicants = pd.DataFrame(
        {
            "applicant_id": applicant_ids,
            "country_code": countries,
            "currency_code": currencies,
            "created_at": application_timestamp - pd.to_timedelta(rng.integers(30, 1_500, size=count), unit="D"),
        }
    )
    applications = pd.DataFrame(
        {
            "application_id": application_ids,
            "applicant_id": applicant_ids,
            "application_timestamp": application_timestamp,
            "requested_amount": rng.choice([50, 100, 200, 300], size=count),
            "country_code": countries,
        }
    )
    bureau_scores = np.where(thin_file, np.nan, np.round(850 - latent_risk * 430 + rng.normal(0, 35, count), 0))
    bureau_snapshots = pd.DataFrame(
        {
            "application_id": application_ids,
            "event_timestamp": application_timestamp - pd.to_timedelta(rng.integers(1, 45, size=count), unit="D"),
            "available_at": application_timestamp - pd.to_timedelta(rng.integers(0, 2, size=count), unit="D"),
            "bureau_score": bureau_scores,
        }
    )

    transaction_count = 3
    repeated_applicant_ids = np.repeat(applicant_ids, transaction_count)
    repeated_application_timestamp = np.repeat(application_timestamp.to_numpy(), transaction_count)
    cashflow_transactions = pd.DataFrame(
        {
            "transaction_id": [f"cash_{index:07d}" for index in range(count * transaction_count)],
            "applicant_id": repeated_applicant_ids,
            "event_timestamp": pd.to_datetime(repeated_application_timestamp, utc=True)
            - pd.to_timedelta(rng.integers(2, 90, size=count * transaction_count), unit="D"),
            "available_at": pd.to_datetime(repeated_application_timestamp, utc=True)
            - pd.to_timedelta(rng.integers(0, 2, size=count * transaction_count), unit="D"),
            "amount": np.round(rng.normal(90, 45, size=count * transaction_count), 2),
        }
    )
    device_events = pd.DataFrame(
        {
            "event_id": [f"device_{index:06d}" for index in range(count)],
            "application_id": application_ids,
            "event_timestamp": application_timestamp - pd.to_timedelta(rng.integers(0, 5, size=count), unit="D"),
            "available_at": application_timestamp,
            "device_stability_days": rng.integers(1, 1_000, size=count),
        }
    )
    employment_statements = pd.DataFrame(
        {
            "application_id": application_ids,
            "statement_text": np.where(
                rng.random(count) < 0.5,
                "Salaried worker with regular monthly income.",
                "Self-employed business owner with variable weekly income.",
            ),
            "event_timestamp": application_timestamp - pd.to_timedelta(rng.integers(0, 3, size=count), unit="D"),
            "available_at": application_timestamp,
        }
    )
    loan_outcomes = pd.DataFrame(
        {
            "application_id": application_ids,
            "outcome_available_at": application_timestamp + pd.to_timedelta(90, unit="D"),
            "defaulted": rng.binomial(1, latent_risk),
        }
    )
    fraud_probability = np.clip(0.01 + latent_risk * 0.07 + thin_file * 0.02, 0, 0.25)
    fraud_outcomes = pd.DataFrame(
        {
            "application_id": application_ids,
            "outcome_available_at": application_timestamp + pd.to_timedelta(30, unit="D"),
            "fraud_confirmed": rng.binomial(1, fraud_probability),
        }
    )
    return {
        "applicants": applicants,
        "applications": applications,
        "bureau_snapshots": bureau_snapshots,
        "cashflow_transactions": cashflow_transactions,
        "device_events": device_events,
        "employment_statements": employment_statements,
        "loan_outcomes": loan_outcomes,
        "fraud_outcomes": fraud_outcomes,
    }


def write_synthetic_data(output_dir: str | Path, profile: str = "smoke", seed: int = 20260909) -> None:
    """Write a deterministic generated bundle as individually inspectable Parquet files."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in generate_synthetic_data(profile=profile, seed=seed).items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)
