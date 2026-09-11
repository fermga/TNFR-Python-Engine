"""Regression checks for the executable restricted S16 trajectory example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "161_core_research_trajectory.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "core_research_trajectory_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_and_protocol():
    example = _load_example()
    return example, example.run_protocol()


def test_example_builds_two_stable_identity_aligned_meshes(example_and_protocol):
    example, protocol = example_and_protocol
    comparison = protocol["comparison"]
    coarse = protocol["coarse_certificate"]
    fine = protocol["fine_certificate"]
    exact_reference = protocol["exact_reference"]

    assert protocol["finite_protocol_pass"]
    assert protocol["verdicts_agree"]
    assert coarse.joint_temporal_conditions_pass
    assert fine.joint_temporal_conditions_pass
    assert comparison.joint_refinement_conditions_pass
    assert comparison.all_coarse_times_matched
    assert comparison.fine_grid_is_strict_refinement
    assert comparison.maximum_fine_step < comparison.maximum_coarse_step
    assert 0.0 < comparison.maximum_scaled_direct_epi_error
    assert (
        comparison.maximum_scaled_direct_epi_error
        <= comparison.agreement_tolerance
    )
    assert all(
        tuple(node for node, _ in sample.direct_epi_differences)
        == example.NODES
        for sample in comparison.samples
    )
    assert all(
        interval.euler_relaxation.is_euler_stable
        for certificate in (coarse, fine)
        for interval in certificate.intervals
    )
    assert all(
        interval.euler_relaxation.spectral_relative_tolerance
        == pytest.approx(example.SPECTRAL_RELATIVE_TOLERANCE)
        for certificate in (coarse, fine)
        for interval in certificate.intervals
    )
    assert exact_reference[
        "fine_strictly_closer_at_every_noninitial_common_time"
    ]
    assert (
        exact_reference["maximum_fine_error_to_exact_linf"]
        < exact_reference["maximum_coarse_error_to_exact_linf"]
    )


def test_every_example_snapshot_has_exact_pure_epi_telemetry(
    example_and_protocol,
):
    example, protocol = example_and_protocol
    snapshots = (
        protocol["coarse_snapshots"] + protocol["fine_snapshots"]
    )

    assert protocol["maximum_pressure_residual"] == pytest.approx(0.0)
    assert protocol["maximum_nodal_residual"] == pytest.approx(0.0)
    assert example.exact_pure_epi_state(
        protocol["initial"], 0.0
    ) == pytest.approx(example.INITIAL_EPI)
    for snapshot in snapshots:
        assert tuple(snapshot) == example.NODES
        assert all(
            "delta_nfr" in snapshot.nodes[node]
            and "dEPI_dt" in snapshot.nodes[node]
            for node in snapshot
        )


def test_example_report_records_new_temporal_assumptions_and_budgets(
    example_and_protocol, monkeypatch
):
    example, protocol = example_and_protocol
    monkeypatch.setattr(
        example,
        "current_git_source_provenance",
        lambda *_args, **_kwargs: ("deadbeef", False, None),
    )

    report = example.build_report(protocol)

    assert report["design"]["same_dynamics_declared"] is True
    assert report["refinement"]["same_dynamics_declared"] is True
    assert report["refinement"]["spectral_relative_tolerance"] == pytest.approx(
        example.SPECTRAL_RELATIVE_TOLERANCE
    )
    for mesh in ("coarse", "fine"):
        diagnostic = report["euler_diagnostics"][mesh]
        assert diagnostic["spectral_relative_tolerance"] == pytest.approx(
            example.SPECTRAL_RELATIVE_TOLERANCE
        )
        assert diagnostic["spectral_zero_threshold"] > 0.0
        trajectory = report[f"{mesh}_trajectory"]
        assert trajectory[
            "cumulative_scaled_common_lyapunov_positive_variation"
        ] == pytest.approx(0.0)
