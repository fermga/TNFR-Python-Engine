"""Frozen source wiring only; the analytical acquisition is not evaluated."""

from fractions import Fraction as Q

from benchmarks import organized_alternative_outer_acquisition as owner


def test_original_five_source_allowances_reach_the_assessor(monkeypatch):
    from tnfr.physics import _organized_alternative_outer_acquisition as model

    calls = []
    expected = object()

    def assess(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        model, "_assess_organized_alternative_outer_acquisition", assess
    )
    assert owner._assess() is expected
    assert calls == [
        {
            "form_error_bounds": (Q(1, 10**10),) * 5,
            "phase_error_bounds": (Q(1, 10**10),) * 5,
        }
    ]


def test_cli_delegates_exclusive_lifecycle_without_evaluation(monkeypatch, tmp_path):
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return "fixture_not_evaluated"

    monkeypatch.setattr(owner, "_run_single_analytic_assessment", record)
    output = tmp_path / "evidence"
    assert owner.main(["--output", str(output)]) == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["repository"] == owner.ROOT
    assert call["output"] == output
    assert call["assess"] is owner._assess
    assert call["declaration_path"] == (
        "theory/nodal/ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md"
    )
    assert call["evaluator_path"] == (
        "benchmarks/organized_alternative_outer_acquisition.py"
    )
    assert call["admitted_verdict"] == "alternative_outer_acquisition_certified"
    assert call["unavailable_verdict"] == "alternative_outer_acquisition_unavailable"
    assert not output.exists()
