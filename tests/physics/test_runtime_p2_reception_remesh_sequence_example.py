"""Regression checks for the causal P2 Reception/REMESH example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.runtime_p2_reception_remesh_sequence as sequence_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "176_runtime_p2_reception_remesh_sequence.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "runtime_p2_reception_remesh_sequence_example",
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


def test_module_stub_and_facade_expose_the_sequence_api() -> None:
    expected = {
        "ExecutedP2HalfReceptionRemeshSequenceCertificate",
        "certify_executed_p2_half_reception_remesh_sequence",
    }
    assert set(sequence_module.__all__) == expected
    stub = Path(sequence_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class ExecutedP2HalfReceptionRemeshSequenceCertificate" in stub
    assert "def certify_executed_p2_half_reception_remesh_sequence" in stub
    assert expected <= set(physics.__all__)
    for name in expected:
        assert getattr(physics, name) is getattr(sequence_module, name)


def test_example_reports_finite_active_history_extinction(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["certificate_valid"]
    assert report["cycle_count"] == report["extinction_horizon"] == 2
    assert report["extinction_cycle_index"] == 1
    assert report["post_reception_energies"] == ["0/1", "0/1"]
    assert report["post_remesh_energies"][0] != "0/1"
    assert report["post_remesh_energies"][1] == "0/1"


def test_example_keeps_finite_scope_explicit(example_protocol_report) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["scope"] == {
        "same_invocation_causal_provenance": True,
        "whole_sequence_graph_state_atomicity": True,
        "runtime_telescope_included": False,
        "every_reception_stage_q_zero": True,
        "every_remesh_global_delay_copy_eta_zero": True,
        "observed_active_history_extinction": True,
        "future_runtime_stability": False,
        "unobserved_repetition_stability": False,
        "auxiliary_state_stability": False,
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
