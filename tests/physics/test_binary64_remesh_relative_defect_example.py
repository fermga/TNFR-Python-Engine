"""Regression checks for the binary64 REMESH defect-boundary example."""

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
    / "173_binary64_remesh_relative_defect.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "binary64_remesh_relative_defect_example",
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


def test_example_exposes_the_exact_normal_binary64_counterexample(
    example_protocol_report,
) -> None:
    _example, protocol, report = example_protocol_report
    observation = protocol["counterexample"]
    exact_eta = Fraction(2**210) - Fraction(1, 4)

    assert observation.pair_relative_defect_observation_certified
    assert observation.exact_input_pairwise_jensen_denominator == Fraction(
        1,
        2**316,
    )
    assert observation.exact_ideal_squared_separation == Fraction(1, 2**318)
    assert observation.exact_bounded_squared_separation == Fraction(1, 2**106)
    assert (
        observation.exact_minimum_nonnegative_relative_defect_bound
        == exact_eta
    )

    witness = report["normal_binary64_counterexample"]
    assert witness["observation_valid"]
    assert witness["alpha"] == "1/2"
    assert witness["coefficients"] == {
        "beta": "1/4",
        "gamma": "1/4",
        "delta": "1/2",
    }
    assert witness["minimum_eta"] == (
        f"{exact_eta.numerator}/{exact_eta.denominator}"
    )
    assert witness["eta_identity"] == "2^210 - 1/4"
    assert witness["eta_identity_verified"]
    assert not witness["uniform_bound_certified"]
    assert not witness["future_bound_certified"]


def test_example_certifies_only_the_alpha_one_remesh_class(
    example_protocol_report,
) -> None:
    _example, protocol, report = example_protocol_report
    certificate = protocol["alpha_one_class"]

    assert certificate.alpha_one_hard_clip_class_certificate_certified
    assert certificate.exact_uniform_relative_defect_upper_bound == 0
    assert certificate.binary64_global_delay_numeric_copy_certified
    assert certificate.hard_clip_identity_on_class_certified
    assert certificate.remesh_class_forward_invariant_certified

    boundary = report["alpha_one_hard_clip_class"]
    assert boundary == {
        "certificate_valid": True,
        "alpha": "1/1",
        "uniform_eta": "0/1",
        "numeric_global_delay_copy": True,
        "hard_clip_identity": True,
        "remesh_class_forward_invariant": True,
        "schedule_family_certified": False,
        "repeated_binary64_stability": False,
        "future_binary64_execution": False,
        "solver_accuracy": False,
        "full_tnfr_stability": False,
    }


def test_report_keeps_the_claim_and_scope_explicit(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["claim"] == (
        "a bounded binary64 box alone does not imply a useful uniform "
        "REMESH relative-defect budget"
    )
    assert report["normal_binary64_counterexample"]["current_hex"] == [
        "0x1.0000000000000p-52",
        "0x1.0000000000000p-52",
    ]
    assert report["normal_binary64_counterexample"]["local_hex"] == [
        "0x1.0000000000000p-105",
        "0x1.0000000000001p-105",
    ]
    assert report["normal_binary64_counterexample"]["global_hex"] == [
        "0x1.0000000000000p+0",
        "0x1.0000000000000p+0",
    ]


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
