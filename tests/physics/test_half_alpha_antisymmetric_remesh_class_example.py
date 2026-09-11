"""Regression checks for the half-alpha antisymmetric REMESH example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "178_half_alpha_antisymmetric_remesh_class.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "half_alpha_antisymmetric_remesh_class_example",
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


def test_example_exposes_the_class_and_strict_schedule_composition(
    example_protocol_report,
) -> None:
    _example, protocol, report = example_protocol_report
    certificate = protocol["certificate"]
    strict = protocol["strict_policy"]

    assert (
        certificate.half_alpha_antisymmetric_hard_clip_class_certificate_certified
    )
    assert certificate.exact_uniform_relative_defect_upper_bound == Fraction(
        135, 124
    )
    assert certificate.exact_strict_schedule_gain_threshold == Fraction(
        124, 259
    )
    assert strict.exact_effective_head_energy_gain_upper_bound == Fraction(259, 279)
    assert strict.exact_uniform_normalized_block_margin_lower_bound == Fraction(
        20, 279
    )
    assert report["class"]["uniform_eta"] == "135/124"
    assert report["class"]["strict_q_threshold"] == "124/259"
    proof = report["class"]["global_bound_proof"]
    assert Fraction(proof["large_norm_tail_bound"]) < Fraction(135, 124)
    assert proof["large_norm_tail_is_strict"]
    assert proof["finite_core_candidates"] == 6615
    assert proof["finite_core_admissible"] == 3890
    assert proof["finite_core_maximizer"] == [-3, -2, -3]
    assert report["strict_schedule_composition"] == {
        "q": "4/9",
        "q_effective": "259/279",
        "normalized_block_margin": "20/279",
        "geometric_spatial_disagreement_convergence": True,
    }


def test_example_keeps_the_non_strict_threshold_boundary_explicit(
    example_protocol_report,
) -> None:
    _example, protocol, report = example_protocol_report
    boundary = protocol["boundary_policy"]

    assert boundary.exact_effective_head_energy_gain_upper_bound == 1
    assert boundary.exact_uniform_normalized_block_margin_lower_bound == 0
    assert not boundary.geometric_spatial_disagreement_convergence_certified
    assert report["threshold_boundary"] == {
        "q": "124/259",
        "q_effective": "1/1",
        "normalized_block_margin": "0/1",
        "geometric_spatial_disagreement_convergence": False,
    }


def test_example_records_sharpness_and_both_excluded_generalizations(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["sharp_subnormal_witness"] == {
        "current_in_smallest_subnormal_units": [-3, 3],
        "local_in_smallest_subnormal_units": [-2, 2],
        "global_in_smallest_subnormal_units": [-3, 3],
        "ideal_left_in_smallest_subnormal_units": "-11/4",
        "runtime_hex": [
            "-0x0.0000000000004p-1022",
            "0x0.0000000000004p-1022",
        ],
        "observed_eta": "135/124",
        "attains_uniform_bound": True,
    }
    excluded = report["excluded_generalizations"]
    assert excluded["general_metric_centered_input_centers"] == [
        "0/1",
        "0/1",
        "0/1",
    ]
    assert excluded["general_metric_centered_output_center"] == (
        f"-1/{2**84}"
    )
    assert not excluded["general_metric_centering_forward_invariant"]
    assert excluded["unit_lattice_runtime_output"] == [
        "0x1.0000000000000p-2",
        "-0x1.0000000000000p-2",
    ]
    assert not excluded["unit_lattice_forward_invariant"]


def test_example_scope_does_not_promote_runtime_or_future_claims(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["class"]["antisymmetry_preserved"]
    assert report["class"]["remesh_forward_invariant"]
    assert not report["class"]["repeated_binary64_runtime"]
    assert not report["class"]["future_execution"]
    assert not report["class"]["full_tnfr_stability"]


def test_main_prints_the_prebuilt_report(
    example_protocol_report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _value: report)

    example.main()

    assert json.loads(capsys.readouterr().out) == report
