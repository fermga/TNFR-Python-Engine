"""Regression checks for the exact reversible P3 eigenmode example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.reversible_eigenmode_reference as reference_module


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "167_reversible_eigenmode_reference.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "reversible_eigenmode_reference_example",
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


def test_module_stub_and_facade_expose_the_general_reference_api() -> None:
    expected = {
        "ReversibleSingleEigenmodeEulerReferenceCertificate",
        "certify_reversible_single_eigenmode_euler_reference",
    }

    assert set(reference_module.__all__) == expected
    stub = Path(reference_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class ReversibleSingleEigenmodeEulerReferenceCertificate" in stub
    assert "def certify_reversible_single_eigenmode_euler_reference" in stub
    assert expected <= set(physics.__all__)
    assert (
        physics.ReversibleSingleEigenmodeEulerReferenceCertificate
        is reference_module.ReversibleSingleEigenmodeEulerReferenceCertificate
    )
    assert (
        physics.certify_reversible_single_eigenmode_euler_reference
        is reference_module.certify_reversible_single_eigenmode_euler_reference
    )


def test_example_certifies_both_nonregular_p3_modes(
    example_protocol_report,
) -> None:
    _, protocol, _ = example_protocol_report
    by_name = dict(protocol)

    assert tuple(by_name) == ("antisymmetric", "alternating")
    assert all(
        certificate.reference_certificate_certified
        for certificate in by_name.values()
    )
    assert by_name["antisymmetric"].exact_mode_eigenvalue == 1
    assert by_name["alternating"].exact_mode_eigenvalue == 2
    assert all(
        certificate.exact_degrees
        == (Fraction(1), Fraction(2), Fraction(1))
        for certificate in by_name.values()
    )
    assert by_name["antisymmetric"].exact_euler_factors == (
        Fraction(9, 16),
        Fraction(2_401, 4_096),
        Fraction(2_562_890_625, 4_294_967_296),
    )
    assert by_name["alternating"].exact_euler_factors == (
        Fraction(1, 4),
        Fraction(81, 256),
        Fraction(5_764_801, 16_777_216),
    )


def test_report_separates_the_conditional_theorem_from_runtime(
    example_protocol_report,
) -> None:
    _, _, report = example_protocol_report
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))

    assert len(encoded) < 8_000
    assert report["graph"] == {
        "node_count": 3,
        "regular": False,
        "degrees": ["1/1", "2/1", "1/1"],
    }
    assert [mode["mu"] for mode in report["modes"]] == ["1/1", "2/1"]
    for mode in report["modes"]:
        assert mode["reference_certificate_certified"]
        assert mode["strict_subdivision_improvement"]
        assert mode["scope"] == {
            "conditional_exact_real_partition_convergence": True,
            "binary64_asymptotic_convergence": False,
            "arbitrary_or_mixed_mode_initial_data": False,
            "directed_or_nonreversible_generator": False,
            "changing_generator_or_metric": False,
            "glyph_or_remesh_dynamics": False,
            "solver_order": False,
            "full_tnfr_stability": False,
        }


def test_main_emits_finite_json(
    example_protocol_report,
    monkeypatch,
    capsys,
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _: report)

    example.main()
    decoded = json.loads(capsys.readouterr().out)

    assert decoded["claim"] == (
        "exact reversible single-eigenmode Euler references on P3"
    )
    assert [mode["mu"] for mode in decoded["modes"]] == ["1/1", "2/1"]
