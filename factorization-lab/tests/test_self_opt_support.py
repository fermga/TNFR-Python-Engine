"""Unit tests for tnfr_factorization.self_opt_support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from tnfr_factorization.self_opt_support import (  # type: ignore[import]
    attach_self_opt_sequences,
    run_partition_self_optimization,
)

import scripts.run_self_opt_validation as validator_mod
import scripts.run_self_optimization as runner_mod


def _fake_runner_summary() -> Dict[str, Any]:
    return {
        "partition_results": [
            {
                "success": True,
                "partition_id": "p0",
                "telemetry": {"delta_c": 0.15, "delta_phi_s": -0.02, "delta_si": 0.04},
                "engine": {
                    "validation": {
                        "passed": True,
                        "canonical_tokens": ["IL", "THOL"],
                        "tokens": ["IL", "THOL"],
                    },
                    "signature": "sig-123",
                    "snapshot_path": "snapshots/p0.json",
                },
                "candidate_factors": [13],
            }
        ]
    }


def _fake_validation_summary() -> Dict[str, Any]:
    return {
        "results": [
            {
                "status": "validated",
                "metadata": {"partition_id": "p0"},
            }
        ]
    }


def test_run_partition_self_optimization_invokes_clis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    summary = tmp_path / "_manifest_summary.json"
    summary.write_text("{}", encoding="utf-8")

    captured_runner_args: Dict[str, Any] = {}
    captured_validator_args: Dict[str, Any] = {}

    def fake_runner_parse(args: list[str]) -> list[str]:
        captured_runner_args["args"] = args
        return args

    def fake_runner_run(args: list[str]) -> Dict[str, Any]:
        captured_runner_args["run_called_with"] = args
        return _fake_runner_summary()

    def fake_validator_parse(args: list[str]) -> list[str]:
        captured_validator_args["args"] = args
        return args

    def fake_validator_run(args: list[str]) -> Dict[str, Any]:
        captured_validator_args["run_called_with"] = args
        return _fake_validation_summary()

    monkeypatch.setattr(runner_mod, "parse_args", fake_runner_parse)
    monkeypatch.setattr(runner_mod, "run", fake_runner_run)
    monkeypatch.setattr(validator_mod, "parse_args", fake_validator_parse)
    monkeypatch.setattr(validator_mod, "run", fake_validator_run)

    payload = run_partition_self_optimization(
        manifest_path=manifest,
        manifest_summary_path=summary,
        base_name="unit_certificate",
        operation_type="paley_partition",
        output_root=tmp_path / "self_opt",
    )

    assert payload is not None
    assert "promotable" in payload
    assert "p0" in payload["promotable"]
    assert payload["promotable"]["p0"]["telemetry"]["delta_c"] == pytest.approx(0.15)
    assert "--manifest" in captured_runner_args["args"]
    assert captured_validator_args["run_called_with"][0] == "--payload-root"


def test_attach_self_opt_sequences_merges_summary() -> None:
    summary = {
        "promotable": {
            "p0": {
                "telemetry": {
                    "delta_c": 0.12,
                    "delta_phi_s": -0.03,
                    "delta_si": 0.02,
                },
                "engine": {
                    "validation": {
                        "passed": True,
                        "canonical_tokens": ["IL"],
                        "tokens": ["IL"],
                    },
                    "signature": "sig",
                    "snapshot_path": "snap.json",
                },
            }
        }
    }

    merged = attach_self_opt_sequences({}, summary)
    assert merged is not None
    seq_block = merged["self_optimization"]["validated_sequences"]
    assert "p0" in seq_block
    assert seq_block["p0"]["delta_c"] == pytest.approx(0.12)
