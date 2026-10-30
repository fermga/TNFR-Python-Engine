"""Static law-robustness bounds reuse retained tubes without a producer run."""

import shutil
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr._exact_time import exp_unit_bounds
from tnfr.physics import relational_transit as transit
from tnfr.research import relational_formation_robustness as owner
from tnfr.utils.io import json_dumps, json_loads

REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/assets/relational_capture_response/continuous-transit.audit.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError("a retained-record bound must not run a trajectory producer")


@pytest.fixture(scope="module")
def robustness():
    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "certify_relational_transit_capture",
            "_validated_step",
            "_tube",
            "_flow_jets",
        ):
            patch.setattr(transit, name, forbidden)
        result = owner.audit_relational_formation_robustness(REFERENCE)
    assert result.admitted, result.unavailable_reasons
    return result


def test_retained_corridor_and_endpoint_have_independent_strict_margins(robustness):
    evidence = robustness.evidence
    assert robustness.source_file_count == 604
    assert len(robustness.reference_sha256) == 3
    assert evidence.horizon == 32 and len(evidence.comparison_steps) == 256
    assert evidence.corridor_radius == Q(1, 1024)
    assert evidence.minimum_picard_margin > 0
    for old, inflated, multiplier in zip(
        evidence.reference_resultant_lower_bounds,
        evidence.inflated_resultant_lower_bounds,
        (4, 3, 2),
        strict=True,
    ):
        assert inflated == old - multiplier * evidence.corridor_radius
    assert all(
        value > threshold
        for value, threshold in zip(
            evidence.inflated_resultant_lower_bounds,
            (Q(97, 100), Q(21, 25), Q(29, 1000)),
            strict=True,
        )
    )
    assert evidence.inflated_coordinate_hull[0].abs_max < Q(101, 100)
    assert evidence.inflated_coordinate_hull[1].abs_max < Q(1, 4)
    assert all(value.lo > Q(17, 100) for value in evidence.endpoint_rectangle_margins)
    assert evidence.endpoint_storage.hi < 7
    assert evidence.endpoint_storage_margin == 7 - evidence.endpoint_storage.hi
    # The endpoint admits the reflected E<7 theorem. It is above the
    # different full-state acute-sector threshold, not a certificate thereof.
    assert evidence.endpoint_storage.lo > Q(6925, 1000)
    assert all(passed for _, passed in evidence.checks)


def test_logarithmic_growth_and_separate_parameter_caps_close_first_exit(robustness):
    evidence = robustness.evidence
    time = Q(0)
    integrated = Q(0)
    for step in evidence.comparison_steps:
        assert step.time == time and step.duration == Q(1, 8)
        matrix = step.comparison_matrix
        assert all(matrix[i][j] >= 0 for i in range(4) for j in range(4) if i != j)
        assert step.logarithmic_norm_upper_bound == max(Q(0), max(map(sum, matrix)))
        integrated += step.duration * step.logarithmic_norm_upper_bound
        time += step.duration
    assert time == evidence.horizon
    assert integrated == evidence.integrated_logarithmic_norm_upper_bound < 9
    assert exp_unit_bounds(Q(1))[1] < Q(11, 4)
    assert Q(11, 4) ** 9 < evidence.exponential_upper_bound == 2**14
    limits = {bound.law: bound for bound in evidence.law_bounds}
    assert limits["rho_phase_loss_only"].parameter_upper_bound == Q(1, 2**29)
    assert limits["eta_nonlinear_form_only"].parameter_upper_bound == Q(1, 2**24)
    for bound in limits.values():
        expected = (
            32 * 2**14 * bound.parameter_upper_bound * bound.forcing_factor_upper_bound
        )
        assert expected == bound.deviation_upper_bound == Q(1, 2048)
        assert bound.corridor_slack == evidence.corridor_radius - expected > 0
    # Adding both advertised forcing budgets would consume the whole corridor;
    # these separate conclusions cannot be advertised as a simultaneous box.
    assert (
        sum(bound.deviation_upper_bound for bound in limits.values())
        == evidence.corridor_radius
    )


def test_nonlinear_bound_and_exact_export_do_not_encode_a_fitted_parameter(robustness):
    s = pytest.importorskip("sympy")
    u = s.Symbol("u", nonnegative=True)
    h = u**3 / (1 + u**2)
    assert s.factor(u - h - u / (1 + u**2)) == 0
    assert (u / (1 + u**2)).is_nonnegative
    assert s.simplify(h.subs(u, -u) + h) == 0
    assert Q(101, 7200) < Q(1, 64) and Q(1, 192) < Q(1, 64)
    projected = robustness.to_dict()
    evidence = projected["report"]["evidence"]
    assert evidence["corridor_radius"] == {"numerator": 1, "denominator": 1024}
    assert evidence["law_bounds"][0]["deviation_upper_bound"] == {
        "numerator": 1,
        "denominator": 2048,
    }
    assert json_loads(json_dumps(projected)) == projected
    assert (
        "parameter_caps_do_not_authorize_simultaneous_rho_and_eta_terms"
        in projected["report"]["scope"]
    )


def test_missing_or_changed_artifacts_abstain_before_derivative_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(transit, "_comparison_matrix", forbidden)
    missing = owner.audit_relational_formation_robustness(tmp_path / REFERENCE.name)
    assert missing.status == "unavailable" and missing.evidence is None
    assert missing.unavailable_reasons[0] == "missing_artifacts"
    for path in (
        REFERENCE,
        REFERENCE.with_suffix(".protocol.json"),
        REFERENCE.with_suffix(".source.zip"),
    ):
        shutil.copyfile(path, tmp_path / path.name)
    changed = tmp_path / REFERENCE.name
    changed.write_bytes(changed.read_bytes() + b"\n")
    result = owner.audit_relational_formation_robustness(changed)
    assert result.status == "inconsistent" and result.evidence is None
    assert "immutable reference changed" in result.unavailable_reasons[0]


def test_current_formula_changes_cannot_borrow_the_retained_flow(tmp_path, monkeypatch):
    changed_source = tmp_path / "relational_transit.py"
    source = Path(transit.__file__).read_text(encoding="utf-8")
    source = source.replace("q, r, a, b = state", "q, r, a, b = state\n    q = -q")
    changed_source.write_text(source, encoding="utf-8")
    monkeypatch.setattr(transit, "__file__", str(changed_source))
    monkeypatch.setattr(transit, "_comparison_matrix", forbidden)
    result = owner.audit_relational_formation_robustness(REFERENCE)
    assert result.status == "inconsistent" and result.evidence is None
    assert "reference formula changed: _flow" in result.unavailable_reasons[0]


@pytest.mark.parametrize("delegate", ("_comparison_matrix", "comparison_matrix"))
def test_changed_comparison_cannot_borrow_the_retained_formula(monkeypatch, delegate):
    identity = tuple(tuple(Q(i == j) for j in range(4)) for i in range(4))
    monkeypatch.setattr(transit, delegate, lambda *args: identity)
    result = owner.audit_relational_formation_robustness(REFERENCE)
    assert result.status == "inconsistent" and result.evidence is None
    assert result.unavailable_reasons == (
        "delegated comparison differs from reference formula",
    )


def test_changed_oracle_cannot_replace_the_archived_formula(tmp_path, monkeypatch):
    changed_source = tmp_path / "relational_formation_robustness.py"
    source = Path(owner.__file__).read_text(encoding="utf-8")
    source = source.replace(
        "columns[j][i].hi if i == j", "columns[j][i].abs_max if i == j"
    )
    changed_source.write_text(source, encoding="utf-8")
    monkeypatch.setattr(owner, "__file__", str(changed_source))
    monkeypatch.setattr(transit, "_comparison_matrix", forbidden)
    result = owner.audit_relational_formation_robustness(REFERENCE)
    assert result.status == "inconsistent" and result.evidence is None
    assert result.unavailable_reasons == (
        "reference formula changed: _comparison_matrix",
    )


def test_unresolved_majorant_abstains_despite_an_admitted_endpoint(monkeypatch):
    monkeypatch.setattr(owner, "exp_unit_bounds", lambda value: (Q(1), Q(3)))
    result = owner.audit_relational_formation_robustness(REFERENCE)
    assert result.status == "unavailable" and not result.admitted
    assert result.evidence.integrated_logarithmic_norm_upper_bound < 9
    checks = dict(result.evidence.checks)
    assert checks["inflated_endpoint_in_Rplus"]
    assert checks["inflated_endpoint_storage_below_seven"]
    assert not checks["exponential_rational_majorant"]
    assert result.unavailable_reasons == ("exponential_rational_majorant",)
