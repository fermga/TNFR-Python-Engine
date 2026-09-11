"""Regression checks for the transactional P2 policy example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.runtime_p2_reception_remesh_policy as policy_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "177_runtime_p2_reception_remesh_policy.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "runtime_p2_reception_remesh_policy_example",
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


def test_module_stub_and_facade_expose_the_policy_api() -> None:
    name = "execute_p2_half_reception_remesh_policy_invocation"
    assert policy_module.__all__ == (name,)
    stub = Path(policy_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert f"def {name}" in stub
    assert name in physics.__all__
    assert getattr(physics, name) is getattr(policy_module, name)


def test_example_reports_two_separately_validated_invocations(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["certificate_valid"] == [True, True]
    assert report["distinct_executions"]
    assert report["cycle_counts"] == [2, 2]
    assert report["post_remesh_energies"][0][-1] == "0/1"
    assert report["post_remesh_energies"][1] == ["0/1", "0/1"]
    assert report["retained_history_length"] == 5


def test_example_keeps_future_and_auxiliary_claims_false(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["scope"] == {
        "finite_causal_extinction": [True, True],
        "future_runtime_stability": False,
        "unobserved_repetition_stability": False,
        "auxiliary_state_stability": False,
        "full_tnfr_stability": False,
    }


def test_main_emits_the_prebuilt_report(
    example_protocol_report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _value: report)

    example.main()

    assert json.loads(capsys.readouterr().out) == report
