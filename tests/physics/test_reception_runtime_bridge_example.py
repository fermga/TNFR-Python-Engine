"""Regression checks for the local Reception realization example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "163_reception_runtime_bridge.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "reception_runtime_bridge_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_and_protocol():
    example = _load_example()
    return example, example.run_protocol()


def test_example_separates_reception_and_transport_means(example_and_protocol):
    _, protocol = example_and_protocol
    result = protocol["baseline"]

    assert result.unweighted_runtime_neighbor_mean == pytest.approx(0.5)
    assert result.transport_weighted_neighbor_mean == pytest.approx(0.9)
    assert result.runtime_target_value == pytest.approx(0.25)
    assert result.runtime_matches_represented_affine_exactly_at_snapshot


def test_example_exposes_mean_drift_and_pressure_refresh(example_and_protocol):
    _, protocol = example_and_protocol
    result = protocol["baseline"]

    assert result.represented_affine_jump_eligible
    assert result.affine_jump_certificate is not None
    assert not result.ideal_real_global_weighted_mean_preservation
    assert result.exact_current_weighted_mean_shift == Fraction(1, 12)
    assert result.pressure_refresh_required
    assert result.exact_pressure_refresh_iff_nontrivial_theorem
    assert result.post_reset_pressure_manifold_defect_norm == pytest.approx(0.25)


def test_example_recovery_uses_the_exact_hybrid_decision(example_and_protocol):
    _, protocol = example_and_protocol
    baseline = protocol["baseline"]
    recovered = protocol["recovered"]

    jump = baseline.affine_jump_certificate
    assert jump is not None
    assert jump.exact_quotient_energy_gain_upper_bound == Fraction(1)
    assert baseline.recovery_break_even_duration_estimate == 0.0
    rate = baseline.diffusion_certificate.certified_exponential_rate_lower_bound
    assert protocol["recovery_duration"] == pytest.approx(1.0 / rate)
    assert protocol["recovery_duration"] > 0.0
    assert recovered.hybrid_certificate is not None
    assert recovered.represented_hybrid_recovery_certified is True
    assert recovered.hybrid_certificate.disagreement_contracts_over_declared_horizon
    assert not recovered.hybrid_certificate.initial_weighted_mean_preserved


def test_example_report_is_finite_and_declares_scope(
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
    assert report["runtime"]["unweighted_neighbor_mean"] == pytest.approx(0.5)
    assert report["runtime"]["conductance_weighted_neighbor_mean"] == pytest.approx(
        0.9
    )
    assert report["affine_boundary"]["exact_quotient_gain_upper_bound"] == "1/1"
    assert report["consensus_and_pressure"]["pressure_refresh_required"]
    assert report["recovery"]["exact_hybrid_contraction_decision"]
    assert report["scope"]["uniform_real_scalar_embedding"]
    assert report["affine_boundary"][
        "ideal_real_hard_clipping_inactive_by_convexity"
    ]
    assert report["affine_boundary"][
        "runtime_hard_clipping_inactive_at_snapshot"
    ]
    assert report["scope"]["ideal_hard_clipping_controlled_by_convexity"]
    assert report["scope"]["runtime_clip_observation_is_snapshot_only"]
