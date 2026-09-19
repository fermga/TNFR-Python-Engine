"""Regression checks for the causal finite-block margin example."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.runtime_remesh_schedule_block_margin as block_module
from tests.example_protocol_helpers import assert_prebuilt_report_main, load_example

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "170_runtime_remesh_block_margin.py"
)


@pytest.fixture(scope="module")
def example_protocol_report():
    example = load_example(EXAMPLE_PATH)
    protocol = example.run_protocol()
    return example, protocol, example.build_report(protocol)


def test_module_stub_and_facade_expose_block_margin_api() -> None:
    expected = {
        "RuntimeRemeshScheduleBlockMarginObservation",
        "observe_executed_event_remesh_block_margin",
    }

    assert set(block_module.__all__) == expected
    stub = Path(block_module.__file__).with_suffix(".pyi").read_text(encoding="utf-8")
    assert "class RuntimeRemeshScheduleBlockMarginObservation" in stub
    assert "def observe_executed_event_remesh_block_margin" in stub
    assert expected <= set(physics.__all__)
    for name in expected:
        assert getattr(physics, name) is getattr(block_module, name)


def test_example_reports_exact_positive_and_zero_margins(
    example_protocol_report,
) -> None:
    _, protocol, report = example_protocol_report
    positive = protocol["positive_margin"]
    zero = protocol["zero_margin"]

    assert positive.block_observation_certified
    assert positive.same_graph_execution_provenance_certified
    assert positive.whole_sequence_graph_state_atomic
    assert positive.exact_gain_based_energy_drop_fraction_lower_bound == Fraction(
        139,
        256,
    )
    assert positive.exact_endpoint_energy_gain_upper_bound == Fraction(117, 256)
    assert positive.positive_normalized_block_margin_certified
    assert positive.strict_energy_contraction_observed

    assert zero.block_observation_certified
    assert zero.exact_gain_based_energy_drop_fraction_lower_bound == 0
    assert zero.exact_endpoint_energy_gain_upper_bound == 1
    assert not zero.positive_normalized_block_margin_certified
    assert not zero.strict_energy_contraction_observed
    assert (
        report["positive_block"]["exact_gain_based_energy_drop_fraction_lower_bound"]
        == "139/256"
    )
    assert (
        report["alpha_one_boundary"][
            "exact_gain_based_energy_drop_fraction_lower_bound"
        ]
        == "0/1"
    )

    # These claims belong to the real report, independently of CLI wiring.
    decoded = report
    assert decoded["claim"] == ("exact margins for two causally executed finite blocks")
    assert decoded["positive_block"]["positive_normalized_block_margin_certified"]
    assert not decoded["alpha_one_boundary"][
        "positive_normalized_block_margin_certified"
    ]


def test_example_withholds_uniform_and_repeated_scope(
    example_protocol_report,
) -> None:
    _, protocol, report = example_protocol_report
    positive = protocol["positive_margin"]

    assert report["scope"] == {
        "absolute_uniform_margin": False,
        "uniform_normalized_forward_invariant_class_margin": False,
        "intrablock_prefix_bound": False,
        "repeated_runtime_stability": False,
        "future_stability": False,
        "runtime_global_gain": False,
        "solver_accuracy": False,
        "solver_order": False,
        "mesh_convergence": False,
        "full_tnfr_stability": False,
    }
    assert not positive.uniform_class_coercivity_certified
    assert not positive.uniform_repeated_margin_certified
    assert not positive.repeated_runtime_stability_certified
    assert not positive.future_stability_certified


def test_main_emits_finite_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert_prebuilt_report_main(load_example(EXAMPLE_PATH), monkeypatch, capsys)
