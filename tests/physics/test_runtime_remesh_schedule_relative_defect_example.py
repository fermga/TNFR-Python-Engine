"""Regression checks for the finite relative-defect REMESH example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tnfr.physics as physics
import tnfr.physics.remesh_schedule_relative_defect_stability as pure_module
import tnfr.physics.runtime_remesh_schedule_relative_defect as runtime_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "172_runtime_remesh_relative_defect.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "runtime_remesh_relative_defect_example",
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


def test_modules_stubs_and_facade_expose_both_apis() -> None:
    pure_names = {
        "UniformRemeshScheduleRelativeDefectStabilityCertificate",
        "certify_uniform_remesh_schedule_relative_defect_stability",
    }
    runtime_names = {
        "RuntimeRemeshScheduleRelativeDefectBlockObservation",
        "observe_executed_event_remesh_relative_defect_block",
    }
    assert set(pure_module.__all__) == pure_names
    assert set(runtime_module.__all__) == runtime_names
    expected = pure_names | runtime_names
    assert expected <= set(physics.__all__)
    for name in pure_names:
        assert getattr(physics, name) is getattr(pure_module, name)
    for name in runtime_names:
        assert getattr(physics, name) is getattr(runtime_module, name)
    assert EXAMPLE_PATH.is_file()

    pure_stub = Path(pure_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    runtime_stub = Path(runtime_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class UniformRemeshScheduleRelativeDefect" in pure_stub
    assert "class RuntimeRemeshScheduleRelativeDefect" in runtime_stub


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.physics; import tnfr.operators",
        "import tnfr.operators; import tnfr.physics",
    ),
)
def test_relative_defect_facade_is_cold_import_order_safe(imports: str) -> None:
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


def test_example_binds_zero_and_positive_defect_blocks(
    example_protocol_report,
) -> None:
    _example, protocol, _report = example_protocol_report
    dyadic_certificate = protocol["dyadic_certificate"]
    dyadic = protocol["dyadic_observation"]
    positive_certificate = protocol["positive_certificate"]
    positive = protocol["positive_observation"]

    assert dyadic_certificate.exact_effective_head_energy_gain_upper_bound == (
        Fraction(9, 16)
    )
    assert dyadic.exact_pre_schedule_energy_defects == (0, 0)
    assert dyadic.exact_finite_endpoint_energy_gain_upper_bound == Fraction(
        9,
        16,
    )
    assert dyadic.relative_defect_block_observation_certified

    assert positive.exact_pre_schedule_energy_defects[0] > 0
    assert positive.exact_relative_energy_defect_ratios == (
        positive_certificate.pre_schedule_relative_energy_defect_upper_bound,
    )
    assert positive.exact_relative_energy_defect_slacks == (0,)
    assert positive_certificate.exact_effective_head_energy_gain_upper_bound < 1
    assert positive.relative_defect_block_observation_certified


def test_report_keeps_the_finite_scope_explicit(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["claim"] == (
        "finite causal verification of a relative-defect envelope"
    )
    zero = report["zero_defect_complete_block"]
    positive = report["positive_binary64_defect"]
    assert zero["q_eff"] == "9/16"
    assert zero["endpoint_gain_upper_bound"] == "9/16"
    assert positive["defects"][0] != "0/1"
    assert report["scope"] == {
        "same_execution_provenance": True,
        "runtime_forward_invariant_class": False,
        "repeated_binary64_stability": False,
        "future_runtime_stability": False,
        "solver_accuracy": False,
        "full_tnfr_stability": False,
    }


def test_main_prints_the_prebuilt_report(
    example_protocol_report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _value: report)

    example.main()

    decoded = json.loads(capsys.readouterr().out)
    assert decoded == report
