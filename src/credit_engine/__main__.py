"""CLI entrypoint for generating synthetic decisioning data."""

from __future__ import annotations

import argparse

from credit_engine.generator import PROFILES, write_synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic thin-file credit decisioning data.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output-dir", default="data/generated/smoke")
    arguments = parser.parse_args()
    write_synthetic_data(arguments.output_dir, profile=arguments.profile, seed=arguments.seed)


if __name__ == "__main__":
    main()
