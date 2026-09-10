"""Regression checks for the finite P2 event/REMESH reference example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.event_remesh_reference as reference_module


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "166_event_remesh_reference_family.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "event_remesh_reference_family_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_and_protocol():
    example = _load_example()
    protocol = example.run_protocol()
    return example, protocol, example.build_report(protocol)


def test_module_and_stub_expose_the_reference_api() -> None:
    expected = {
        "P2EventRemeshMeshReferenceObservation",
        "P2EventRemeshReferenceFamilyObservation",
        "observe_p2_event_remesh_reference_family",
    }

    assert set(reference_module.__all__) == expected
    stub = Path(reference_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class P2EventRemeshMeshReferenceObservation" in stub
    assert "class P2EventRemeshReferenceFamilyObservation" in stub
    assert "def observe_p2_event_remesh_reference_family" in stub
    assert expected <= set(physics.__all__)
    assert (
        physics.P2EventRemeshMeshReferenceObservation
        is reference_module.P2EventRemeshMeshReferenceObservation
    )
    assert (
        physics.P2EventRemeshReferenceFamilyObservation
        is reference_module.P2EventRemeshReferenceFamilyObservation
    )
    assert (
        physics.observe_p2_event_remesh_reference_family
        is reference_module.observe_p2_event_remesh_reference_family
    )


def test_example_executes_the_expected_two_four_eight_meshes(
    example_and_protocol,
) -> None:
    _, protocol, _ = example_and_protocol
    reference = protocol["reference"]

    assert all(passed for _, passed in reference.conditions)
    assert tuple(
        len(mesh.exact_segment_durations) for mesh in reference.meshes
    ) == (2, 4, 8)
    assert reference.exact_euler_factors == (
        Fraction(1, 4),
        Fraction(81, 256),
        Fraction(5_764_801, 16_777_216),
    )
    assert reference.exact_quadratic_factor_error_upper_bounds == (
        Fraction(1, 4),
        Fraction(1, 8),
        Fraction(1, 16),
    )


def test_report_separates_the_finite_result_from_open_claims(
    example_and_protocol,
) -> None:
    _, _, report = example_and_protocol
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))

    assert len(encoded) < 7000
    assert report["reference_family_certified"]
    assert [item["segment_count"] for item in report["meshes"]] == [2, 4, 8]
    assert report["strict_proper_subdivision_improvement"]
    assert all(
        item["runtime_residual_linf"] == "0/1" for item in report["meshes"]
    )
    assert report["scope"] == {
        "compatible_finite_p2_problem": True,
        "binary64_asymptotic_convergence": False,
        "arbitrary_glyph_or_mixed_mode": False,
        "soft_clipping": False,
        "changing_support_or_metric": False,
        "generic_mesh_convergence": False,
        "solver_order": False,
        "repeated_runtime_stability": False,
        "future_stability": False,
    }


def test_main_emits_finite_json(example_and_protocol, monkeypatch, capsys) -> None:
    example, protocol, report = example_and_protocol
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _: report)

    example.main()
    report = json.loads(capsys.readouterr().out)

    assert report["claim"] == "finite exact P2 event/REMESH reference family"
    assert report["reference_family_certified"]
