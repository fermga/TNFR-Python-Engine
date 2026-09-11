"""Regression checks for the executed reversible P3 reference example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.runtime_eigenmode_reference as runtime_module


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "168_runtime_reversible_eigenmode_reference.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "runtime_reversible_eigenmode_reference_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_protocol_report():
    example = _load_example()
    observation = example.run_protocol()
    return example, observation, example.build_report(observation)


def test_module_stub_and_facade_expose_runtime_reference_api() -> None:
    expected = {
        "ExecutedReversibleSingleEigenmodeEulerPartitionObservation",
        "ExecutedReversibleSingleEigenmodeEulerReferenceObservation",
        "observe_executed_reversible_single_eigenmode_euler_reference",
    }

    assert set(runtime_module.__all__) == expected
    stub = Path(runtime_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert (
        "class ExecutedReversibleSingleEigenmodeEulerPartitionObservation"
        in stub
    )
    assert (
        "class ExecutedReversibleSingleEigenmodeEulerReferenceObservation"
        in stub
    )
    assert (
        "def observe_executed_reversible_single_eigenmode_euler_reference"
        in stub
    )
    assert expected <= set(physics.__all__)
    assert (
        physics.ExecutedReversibleSingleEigenmodeEulerPartitionObservation
        is runtime_module.ExecutedReversibleSingleEigenmodeEulerPartitionObservation
    )
    assert (
        physics.ExecutedReversibleSingleEigenmodeEulerReferenceObservation
        is runtime_module.ExecutedReversibleSingleEigenmodeEulerReferenceObservation
    )
    assert (
        physics.observe_executed_reversible_single_eigenmode_euler_reference
        is runtime_module.observe_executed_reversible_single_eigenmode_euler_reference
    )


def test_example_binds_three_executed_p3_partitions(
    example_protocol_report,
) -> None:
    _, observation, report = example_protocol_report

    assert observation.runtime_reference_binding_certified
    assert observation.nodes == (0, 1, 2)
    assert observation.reference_certificate.exact_mode_eigenvalue == 1
    assert observation.reference_certificate.exact_reversible_metric == (
        1,
        2,
        1,
    )
    assert [row["segment_count"] for row in report["partitions"]] == [
        2,
        4,
        8,
    ]
    assert all(row["binding_certified"] for row in report["partitions"])
    assert all(row["nonzero_rho"] for row in report["partitions"])
    assert all(row["nonzero_eta"] for row in report["partitions"])
    assert all(row["nonzero_epsilon"] for row in report["partitions"])


def test_report_withholds_runtime_convergence_and_causal_scope(
    example_protocol_report,
) -> None:
    _, _, report = example_protocol_report
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))

    assert len(encoded) < 8_000
    assert report["scope"] == {
        "binary64_asymptotic_convergence": False,
        "runtime_mesh_convergence": False,
        "solver_accuracy": False,
        "solver_order": False,
        "common_causal_execution_provenance": False,
        "glyph_or_remesh_dynamics": False,
        "future_or_repeated_stability": False,
        "full_tnfr_stability": False,
    }


def test_main_emits_finite_json(
    example_protocol_report,
    monkeypatch,
    capsys,
) -> None:
    example, observation, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: observation)
    monkeypatch.setattr(example, "build_report", lambda _: report)

    example.main()
    decoded = json.loads(capsys.readouterr().out)

    assert decoded["claim"] == (
        "finite executed reversible single-eigenmode binding on P3"
    )
    assert decoded["runtime_reference_binding_certified"]
    assert [row["segment_count"] for row in decoded["partitions"]] == [2, 4, 8]
