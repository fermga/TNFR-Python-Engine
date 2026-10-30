"""Load example entry points and verify their report wiring without a producer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def load_example(path: Path) -> ModuleType:
    """Load one explicit example without registering or executing its protocol."""
    spec = importlib.util.spec_from_file_location(f"tnfr_example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_prebuilt_report_main(
    example: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check one producer-to-renderer-to-JSON call chain using a sentinel.

    Numerical and provenance assertions belong to the real protocol tests.
    This independent entry-point check must not create a scientific execution.
    """
    protocol = object()
    report = {"claim": "synthetic wiring control", "scope": {"runtime": False}}
    calls = []

    def run_protocol():
        calls.append("run_protocol")
        return protocol

    def build_report(received):
        assert received is protocol
        calls.append("build_report")
        return report

    monkeypatch.setattr(example, "run_protocol", run_protocol)
    monkeypatch.setattr(example, "build_report", build_report)

    example.main()

    assert calls == ["run_protocol", "build_report"]
    assert json.loads(capsys.readouterr().out) == report
