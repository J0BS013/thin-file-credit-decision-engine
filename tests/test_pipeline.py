from __future__ import annotations

import json

from credit_engine.pipeline import run_smoke_pipeline


def test_smoke_pipeline_generates_a_reproducible_report(tmp_path) -> None:
    report = run_smoke_pipeline(tmp_path, seed=101)
    saved = json.loads((tmp_path / "mvp_summary.json").read_text(encoding="utf-8"))
    assert report["applications"] == 1_000
    assert len(saved["policy_backtest"]) == 2
    assert {row["policy"] for row in saved["policy_backtest"]} == {"rules_v1", "expected_value_v1"}
