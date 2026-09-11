"""Regression checks for the executed P2 Reception-stage example."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tnfr.physics as physics
import tnfr.physics.runtime_p2_reception_stage as runtime_p2_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "175_runtime_p2_reception_stage.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "runtime_p2_reception_stage_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_protocol_report():
    example = _load_example()
    protocol = example.run_protocol()
    return example, protocol, example.build_report(protocol)


def test_module_stub_and_facade_expose_the_runtime_api() -> None:
    expected = {
        "ExecutedP2HalfReceptionStageCertificate",
        "certify_executed_p2_half_reception_stage",
    }
    assert set(runtime_p2_module.__all__) == expected
    stub = Path(runtime_p2_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class ExecutedP2HalfReceptionStageCertificate" in stub
    assert "def certify_executed_p2_half_reception_stage" in stub
    assert expected <= set(physics.__all__)
    for name in expected:
        assert getattr(physics, name) is getattr(runtime_p2_module, name)


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.physics; import tnfr.operators",
        "import tnfr.operators; import tnfr.physics",
    ),
)
def test_facade_is_cold_import_order_safe(imports: str) -> None:
    environment = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join((source_path, existing))
    )
    completed = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_example_reports_the_executor_bound_zero_gain(example_protocol_report) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["certificate_valid"]
    assert report["event_index"] == 0
    assert report["node_order"] == [0, 1]
    assert report["epi_before"] == [-1.0, 0.5]
    assert report["epi_after"] == [-0.25, -0.25]
    assert report["normalized_metric"] == ["1/2", "1/2"]
    assert report["energy_before"] == "9/32"
    assert report["energy_after"] == report["q"] == "0/1"


def test_example_keeps_runtime_scope_explicit(example_protocol_report) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["scope"] == {
        "finite_executor_bound_epi_stage": True,
        "finite_grammar_admission": True,
        "two_phase_reception_stage": True,
        "observed_numeric_consensus": True,
        "source_q_zero_observed_epi": True,
        "schedule_graph_state_atomicity": True,
        "executed_remesh_configuration": False,
        "future_or_repeated_runtime": False,
        "solver_accuracy": False,
        "full_tnfr_stability": False,
    }


def test_main_emits_the_prebuilt_finite_report(
    example_protocol_report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _value: report)

    example.main()

    assert json.loads(capsys.readouterr().out) == report
