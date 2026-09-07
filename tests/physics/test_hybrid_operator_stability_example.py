"""Regression checks for the affine hybrid pure-EPI example."""

from __future__ import annotations

import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from tnfr.physics import compose_hybrid_epi_stability


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "162_hybrid_epi_stability.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "hybrid_epi_stability_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_and_protocol():
    example = _load_example()
    return example, example.run_protocol()


def test_flow_has_the_declared_represented_common_metric(example_and_protocol):
    example, protocol = example_and_protocol
    flow = protocol["flow"]

    assert tuple(flow.nodes) == example.NODES
    assert tuple(flow.metric_weights) == example.EXPECTED_METRIC
    assert flow.exponential_rate == pytest.approx(8.0)
    assert flow.certified_exponential_rate_lower_bound == pytest.approx(8.0)
    assert flow.is_certified


def test_diffusion_overcomes_the_exact_affine_amplification(example_and_protocol):
    _, protocol = example_and_protocol
    flow = protocol["flow"]
    jump = protocol["amplification"]
    word = protocol["amplification_word"]

    assert jump.exact_consensus_subspace_preservation
    assert jump.exact_weighted_mean_preservation
    assert jump.exact_weighted_frobenius_energy_bound == Fraction(4, 1)
    assert jump.sharp_quotient_energy_gain_estimate == pytest.approx(4.0)
    assert word.energy_multiplier_bound == pytest.approx(4.0 * math.exp(-2.0))
    assert word.disagreement_contracts_over_declared_horizon
    assert word.repeated_schedule_disagreement_convergence_certified
    assert not flow.exact_weighted_mean_preservation
    assert not word.initial_weighted_mean_preserved
    assert not word.repeated_schedule_initial_weighted_consensus_convergence_certified

    insufficient = compose_hybrid_epi_stability(
        flow,
        (jump,),
        (0.0, 0.125),
        repeat_schedule=True,
    )
    assert insufficient.energy_multiplier_bound > 1.0
    assert not insufficient.disagreement_contracts_over_declared_horizon
    assert not insufficient.repeated_schedule_disagreement_convergence_certified


def test_uniform_translation_separates_disagreement_from_consensus(
    example_and_protocol,
):
    _, protocol = example_and_protocol
    jump = protocol["uniform_translation"]
    word = protocol["translation_word"]

    assert jump.exact_consensus_subspace_preservation
    assert jump.finite_global_energy_gain
    assert jump.exact_weighted_frobenius_energy_bound == Fraction(1, 1)
    assert not jump.exact_weighted_mean_preservation
    assert word.energy_multiplier_bound == pytest.approx(math.exp(-2.0))
    assert word.repeated_schedule_disagreement_convergence_certified
    assert not word.initial_weighted_mean_preserved
    assert not word.repeated_schedule_initial_weighted_consensus_convergence_certified


def test_local_offset_exposes_an_exact_zero_to_positive_counterexample(
    example_and_protocol,
):
    _, protocol = example_and_protocol
    jump = protocol["local_offset"]

    assert not jump.exact_consensus_subspace_preservation
    assert not jump.finite_global_energy_gain
    assert math.isinf(jump.global_energy_gain_bound)
    assert jump.consensus_counterexample_level == 0.0
    assert jump.exact_consensus_counterexample_energy_after == Fraction(3, 32)
    assert jump.consensus_counterexample_energy_after == pytest.approx(3.0 / 32.0)


def test_report_is_finite_compact_and_records_the_scope_boundary(
    example_and_protocol, monkeypatch
):
    example, protocol = example_and_protocol
    monkeypatch.setattr(
        example,
        "current_git_source_provenance",
        lambda *_args, **_kwargs: ("deadbeef", False, None),
    )

    report = example.build_report(protocol)
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))

    assert len(encoded) < 5000
    assert report["manifest"]["result_status"] == "derived"
    assert report["cases"]["local_offset"]["global_energy_gain"] == "infinite"
    assert report["cases"]["local_offset"]["exact_output_energy"] == "3/32"
    assert report["flow"]["exact_consensus_subspace_preservation"] is True
    assert report["flow"]["exact_uniform_fixed_point_preservation"] is True
    assert report["flow"]["exact_weighted_mean_preservation"] is False
    assert report["scope"]["operator_names_are_metadata_only"] is True
    assert report["scope"]["runtime_operator_realization_claimed"] is False
    assert report["scope"]["grammar_conclusion_claimed"] is False
