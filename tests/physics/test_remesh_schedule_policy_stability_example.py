"""Regression checks for the exact REMESH/schedule policy example."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.remesh_schedule_policy_stability as policy_module
from tests.example_protocol_helpers import assert_prebuilt_report_main, load_example

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "171_remesh_schedule_policy_stability.py"
)


@pytest.fixture(scope="module")
def example_protocol_report():
    example = load_example(EXAMPLE_PATH)
    protocol = example.run_protocol()
    return example, protocol, example.build_report(protocol)


def test_module_stub_and_facade_expose_policy_api() -> None:
    expected = {
        "UniformRemeshSchedulePolicyStabilityCertificate",
        "certify_uniform_remesh_schedule_policy_stability",
    }
    assert set(policy_module.__all__) == expected
    stub = Path(policy_module.__file__).with_suffix(".pyi").read_text(encoding="utf-8")
    assert "class UniformRemeshSchedulePolicyStabilityCertificate" in stub
    assert "def certify_uniform_remesh_schedule_policy_stability" in stub
    assert expected <= set(physics.__all__)
    for name in expected:
        assert getattr(physics, name) is getattr(policy_module, name)


def test_example_reports_uniform_exact_bounds(example_protocol_report) -> None:
    _, protocol, report = example_protocol_report
    mixed = protocol["mixed_delay_strict"]
    pure_strict = protocol["pure_delay_strict"]
    boundary = protocol["pure_delay_boundary"]

    assert mixed.universal_block_horizon == 5
    assert mixed.exact_uniform_normalized_block_margin_lower_bound == Fraction(3, 4)
    assert mixed.exact_cycle_energy_gain_upper_bound(12) == Fraction(1, 16)
    assert pure_strict.universal_block_horizon == 3
    assert pure_strict.alpha_one_spatial_disagreement_decay_certified
    assert pure_strict.exact_cycle_energy_gain_upper_bound(7) == Fraction(1, 16)
    assert boundary.exact_uniform_normalized_block_margin_lower_bound == 0
    assert boundary.q_one_zero_margin_boundary_certified
    assert not boundary.geometric_spatial_disagreement_convergence_certified
    assert report["mixed_delay_strict"]["horizon_is_claimed_minimal"] is False
    assert report["mixed_delay_strict"]["prefix_gain_upper_bound"] == "1/1"

    # These claims belong to the real report, independently of CLI wiring.
    decoded = report
    assert decoded["claim"] == (
        "conditional uniform exact REMESH/schedule spatial-disagreement " "stability"
    )
    assert decoded["mixed_delay_strict"]["cycle_gain_upper_bound"] == "1/16"
    assert decoded["q_one_boundary"]["uniform_margin"] == "0/1"


def test_example_withholds_runtime_and_general_scope(
    example_protocol_report,
) -> None:
    _, _, report = example_protocol_report
    assert report["scope"] == {
        "schedule_maps_verified": False,
        "binary64_runtime": False,
        "solver_accuracy": False,
        "adaptive_grammar": False,
        "full_tnfr_stability": False,
    }


def test_main_emits_finite_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert_prebuilt_report_main(load_example(EXAMPLE_PATH), monkeypatch, capsys)
