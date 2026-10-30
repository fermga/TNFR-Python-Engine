"""Finite adapter controls; no reserved trajectory or producer is executed.

These tests evaluate only detached rows/Jets, domain predicates, retained
target admission and conditional endpoint arithmetic after their own review
and freeze. A synthetic target-near endpoint is never an actual flow image.
"""

import copy
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, cos, sin
from tnfr.research import formed_two_port_reference as owner
from tnfr.utils.io import json_loads

DIRECTORY = Path(__file__).parents[2] / "docs/assets/sine_formed_classes"


def _overlap(left, right):
    return max(left.lo, right.lo) <= min(left.hi, right.hi)


def _neighbors(node):
    """Independent fine support in the original labels, including both ports."""
    start, local = 9 * (node // 9), node % 9
    result = [start + (local - 1) % 9, start + (local + 1) % 9]
    if node in (4, 5):
        result.append(node + 9)
    elif node in (13, 14):
        result.append(node - 9)
    return tuple(result)


@pytest.fixture(scope="module", autouse=True)
def no_producers_or_steps():
    from tnfr.mathematics import _validated_metric
    from tnfr.physics import (
        relational_sine_two_port_capture,
        relational_sine_two_port_compatibility,
        relational_sine_two_port_transit,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("finite reference controls must not run a producer or step")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_validated_metric, "validated_metric_taylor_step", forbidden)
        for module in (
            relational_sine_two_port_capture,
            relational_sine_two_port_compatibility,
            relational_sine_two_port_transit,
        ):
            for name in tuple(vars(module)):
                if name.startswith("assess_") or name in (
                    "validated_metric_taylor_step",
                    "_root_enclosures",
                    "_enclose_decreasing_phase_root",
                ):
                    patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def reference():
    return owner.build_formed_two_port_reference()


@pytest.fixture(scope="module")
def admitted_target(reference):
    """One fresh sign admission via the existing owner, no new root search."""
    calls = []
    original = owner.sine_two_port_handoff._target

    def retained_admission(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner.sine_two_port_handoff, "_target", retained_admission)
        result = owner.admit_formed_two_port_target(reference, DIRECTORY)
    assert calls == [1]
    return result


def test_original_source_geometry_mean_and_fixed_lifts(reference):
    assert reference.oriented_edges[-2:] == ((4, 13), (5, 14))
    assert reference.oriented_edges[8] == (8, 0)
    assert reference.oriented_edges[17] == (17, 9)
    assert reference.degrees == tuple(len(_neighbors(i)) for i in range(18))
    assert sum(reference.degrees) == 40
    assert len(reference.metric) == 18
    for i, row in enumerate(reference.metric):
        assert row == tuple(Q(reference.degrees[i] if i == j else 0) for j in range(18))
    for coefficients in (reference.nominal_pi_coefficients, reference.nominal_offsets):
        assert sum(d * c for d, c in zip(reference.degrees, coefficients)) == 0
    assert reference.initial_center == (Q(0),) * 18
    assert reference.initial_radius == 0
    assert reference.reference_margin == Q(1, 2048)
    angles = reference.angles(reference.initial_center)
    assert _overlap(angles[8], 4 * reference.pi / 9)
    assert _overlap(angles[17], 2 * reference.pi / 9)
    assert angles[-2].contains(Q(1, 1000))
    assert _overlap(angles[-1], Q(1, 1000) - 2 * reference.pi / 9)
    # The inherited source is not the midpoint source with opposed equal gaps.
    assert angles[-2].hi < Q(1, 500)
    assert angles[-1].hi < -Q(2, 3)


def test_field_agrees_with_all_independent_neighbor_rows(reference):
    values = tuple(Q(i % 5 - 2, 2**20) for i in range(18))
    actual = reference.field(values)
    for i, rate in enumerate(actual):
        expected = sum(
            (
                sin(
                    reference.nominal_phases[j]
                    - reference.nominal_phases[i]
                    + values[j]
                    - values[i]
                )
                for j in _neighbors(i)
            ),
            I(0),
        ) / len(_neighbors(i))
        assert _overlap(rate, expected)
        assert rate.width < Q(1, 2**90)
    conserved = sum((d * value for d, value in zip(reference.degrees, actual)), I(0))
    assert conserved.contains(0)
    origin = reference.field(reference.initial_center)
    assert all(origin[i].contains(0) for i in range(18) if i not in (4, 5, 13, 14))
    assert origin[4].lo > 0 and origin[5].hi < 0
    assert origin[13].hi < 0 and origin[14].lo > 0


def test_jet_derivative_matches_independent_cosine_jacobian(reference):
    center = tuple(Q(i % 3 - 1, 2**20) for i in range(18))
    direction = tuple(Q(i - 8, 19) for i in range(18))
    values = tuple(Jet((I(c), I(h), I(0))) for c, h in zip(center, direction))
    result = reference.field(values)
    for i, rate in enumerate(result):
        expected = sum(
            (
                cos(
                    reference.nominal_phases[j]
                    - reference.nominal_phases[i]
                    + center[j]
                    - center[i]
                )
                * (direction[j] - direction[i])
                for j in _neighbors(i)
            ),
            I(0),
        ) / len(_neighbors(i))
        assert rate.order == 2
        assert _overlap(rate.coeffs[1], expected)
        assert rate.coeffs[1].width < Q(1, 2**90)


def test_domain_returns_slacks_and_never_wraps_escaped_lifts(reference):
    slacks = reference.domain(reference.initial_center)
    assert len(slacks) == 20
    for slack, angle in zip(slacks, reference.angles(reference.initial_center)):
        assert slack == (reference.pi / 2 - abs(angle)).lo - Q(1, 2048)
    assert min(slacks) > Q(1, 60) - Q(1, 2048)
    escaped = [I(0)] * 18
    escaped[4] = 2 * reference.pi
    assert min(reference.domain(escaped)) < 0
    wide = [I(0)] * 18
    wide[5] = I(-2, 2)
    assert min(reference.domain(wide)) < 0


@pytest.mark.parametrize("bad", [True, 1.0, float("nan")])
def test_authoritative_reference_coordinates_are_not_coerced(reference, bad):
    values = [Q(0)] * 18
    values[3] = bad
    with pytest.raises((TypeError, ValueError)):
        reference.field(values)


def test_field_and_domain_reject_wrong_shape_or_jet_contract(reference):
    with pytest.raises(ValueError, match="18"):
        reference.field((Q(0),) * 17)
    values = [Jet.constant(0, 2)] * 18
    values[5] = Jet.constant(0, 1)
    with pytest.raises(ValueError, match="common order"):
        reference.field(values)
    with pytest.raises(ValueError, match="domain needs intervals"):
        reference.domain((Jet.constant(0, 1),) * 18)


def test_retained_target_is_transported_to_original_labels_and_gauge(
    reference, admitted_target
):
    target = admitted_target
    assert target.acute_margin_lower_bound > Q(1, 8)
    assert len(target.target_deviations) == 18
    for column in range(3):
        assert (
            sum(
                reference.degrees[i] * target.target_turn_affine_coefficients[i][column]
                for i in range(18)
            )
            == 0
        )
    assert all(
        _overlap(phase - nominal, deviation)
        for phase, nominal, deviation in zip(
            target.target_phases, reference.nominal_phases, target.target_deviations
        )
    )
    gaps = reference.angles(target.target_deviations)
    assert (4 * reference.pi / 9 - gaps[4]).lo > Q(1, 5)
    assert gaps[8].lo > (4 * reference.pi / 9).hi
    assert min(reference.domain(target.target_deviations)) > Q(1, 8) - Q(1, 2048)
    names = {Path(item["path"]).name for item in target.evidence_receipts}
    assert names == {
        "two-port-compatibility-v1.manifest.json",
        "two-port-compatibility-v1.json",
        "two-port-compatibility-v1.protocol.json",
        "two-port-compatibility-v1.source.zip",
    }
    assert all(
        item["path"].startswith("docs/assets/sine_formed_classes/")
        for item in target.evidence_receipts
    )


def test_target_digest_failure_stops_before_signs(reference, tmp_path, monkeypatch):
    for suffix in (".manifest.json", ".protocol.json", ".source.zip", ".json"):
        name = "two-port-compatibility-v1" + suffix
        (tmp_path / name).write_bytes((DIRECTORY / name).read_bytes())
    report = tmp_path / "two-port-compatibility-v1.json"
    report.write_bytes(report.read_bytes() + b" ")

    def forbidden(*args, **kwargs):
        pytest.fail("altered source bytes must fail before target signs")

    monkeypatch.setattr(owner.sine_two_port_handoff, "_target", forbidden)
    with pytest.raises(ValueError, match="size or digest"):
        owner.admit_formed_two_port_target(reference, tmp_path)


def test_boolean_target_classes_do_not_inherit_report_success():
    record = json_loads((DIRECTORY / "two-port-compatibility-v1.json").read_bytes())
    protocol = json_loads(
        (DIRECTORY / "two-port-compatibility-v1.protocol.json").read_bytes()
    )
    changed = copy.deepcopy(record)
    changed["report"]["classes"][1] = True
    with pytest.raises(ValueError, match="target classes"):
        owner._admit_primary_primitives(changed, protocol)


def test_cached_target_geometry_and_verdicts_are_not_primitive_inputs():
    record = json_loads((DIRECTORY / "two-port-compatibility-v1.json").read_bytes())
    protocol = json_loads(
        (DIRECTORY / "two-port-compatibility-v1.protocol.json").read_bytes()
    )
    changed = copy.deepcopy(record)
    changed["report"].update(
        {
            "status": "unavailable",
            "target_phase_bounds": None,
            "acute_margin_turns_bounds": None,
            "implicit_equilibrium_certified": False,
        }
    )
    report, _, _, _ = owner._admit_primary_primitives(changed, protocol)
    assert report["target_phase_bounds"] is None
    # The separate sign reader must still reconstruct the target; a status flag
    # is neither needed nor accepted as a replacement for those primitives.


def test_comparison_retains_new_source_constants_and_absolute_clock():
    bounds = owner.formed_two_port_comparison_bounds()
    assert bounds.source_error_radius == Q(1, 10**32)
    assert bounds.source_excess_storage_upper_bound == Q(1, 4)
    assert bounds.source_reference_speed_upper_bound == Q(3, 5)
    assert bounds.phase_comparison_error_upper_bound < Q(9, 25000) < Q(1, 2048)
    assert bounds.endpoint_form_norm_upper_bound < Q(1, 8192)
    assert bounds.endpoint_phase_distance_upper_bound < Q(1, 1024)
    assert bounds.endpoint_excess_storage_upper_bound < Q(1, 648000)
    assert bounds.absolute_tau_rational_part == 100 + 10**13
    assert bounds.absolute_tau_pi_squared_coefficient == 1025 * 1023**2
    assert bounds.absolute_t_rational_part * Q(1023, 1024) == 100 + 10**13
    assert bounds.absolute_t_pi_squared_coefficient * Q(1023, 1024) == 1025 * 1023**2


def test_synthetic_endpoint_is_conditional_and_keeps_common_mode(
    reference, admitted_target
):
    center = tuple(value.midpoint for value in admitted_target.target_deviations)
    result = owner.admit_formed_two_port_endpoint(
        reference, admitted_target, center, Q(0)
    )
    assert result.certified and result.conditional_on_acute_reference_prefix
    assert result.reference_distance_margin >= 0
    assert result.form_margin > 0 and result.phase_margin > 0
    assert result.capture_storage_margin > 0
    assert result.capture_radius_squared_margin > 0
    shifted = tuple(value + Q(1, 1000) for value in center)
    shifted_result = owner.admit_formed_two_port_endpoint(
        reference, admitted_target, shifted, 0
    )
    assert not shifted_result.certified
    assert shifted_result.reference_distance_upper_bound > Q(1, 2048)
    uncertain = owner.admit_formed_two_port_endpoint(
        reference, admitted_target, center, Q(1, 2048)
    )
    assert not uncertain.certified
    assert uncertain.reasons == ("reference_target_distance_exceeds_rho",)


@pytest.mark.parametrize("radius", [True, 0.0, -1])
def test_endpoint_radius_admission_precedes_comparison(
    reference, admitted_target, radius
):
    center = tuple(value.midpoint for value in admitted_target.target_deviations)
    with pytest.raises(ValueError):
        owner.admit_formed_two_port_endpoint(reference, admitted_target, center, radius)
