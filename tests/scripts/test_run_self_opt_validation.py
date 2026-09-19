"""Tests for scripts.run_self_opt_validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_self_opt_validation as validator

DATA_ROOT = Path("tests/data/self_opt_validation")


def _build_args(tmp_path: Path, *, extra: list[str] | None = None) -> list[str]:
    args = [
        "--payload-root",
        str(DATA_ROOT),
        "--report",
        str(tmp_path / "report.json"),
        "--quiet",
    ]
    if extra:
        args.extend(extra)
    return args


def test_validation_status_classification(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        validator,
        "OPERATION_TESTS",
        {
            "paley_partition": ["tests/pass_suite.py"],
            "integration_pipeline": ["tests/fail_suite.py"],
        },
    )

    calls: list[tuple[str, ...]] = []

    def fake_run_pytest(
        tests: list[str], *, pytest_cmd: str | None, pytest_args: str
    ) -> int:
        calls.append(tuple(tests))
        return 0 if "tests/pass_suite.py" in tests else 1

    monkeypatch.setattr(validator, "_run_pytest", fake_run_pytest)

    args = validator.parse_args(_build_args(tmp_path))
    summary = validator.run(args)

    assert summary["status_counts"]["validated"] == 1
    assert summary["status_counts"]["regressed"] == 1
    assert summary["status_counts"]["pending"] == 1
    assert len(calls) == 2

    report_payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    statuses = {
        entry["operation_type"]: entry["status"] for entry in report_payload["results"]
    }
    assert statuses["paley_partition"] == "validated"
    assert statuses["integration_pipeline"] == "regressed"
    assert statuses["orbitals_demo"] == "pending"


def test_validation_fail_on_regression(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        validator, "OPERATION_TESTS", {"paley_partition": ["tests/failure.py"]}
    )

    def fake_run_pytest(
        tests: list[str], *, pytest_cmd: str | None, pytest_args: str
    ) -> int:  # noqa: ARG001
        return 1

    monkeypatch.setattr(validator, "_run_pytest", fake_run_pytest)

    args = validator.parse_args(_build_args(tmp_path, extra=["--fail-on-regression"]))
    with pytest.raises(SystemExit):
        validator.run(args)
