"""Finite adapter controls; no reserved trajectory or producer is executed.

These tests evaluate only detached rows/Jets, domain predicates, retained
target admission and conditional endpoint arithmetic after their own review
and freeze. A synthetic target-near endpoint is never an actual flow image.
"""

import copy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, cos, sin
from tnfr.research import formed_two_port_reference as owner
from tnfr.utils.io import json_loads

DIRECTORY = Path(__file__).parents[2] / "docs/assets/sine_formed_classes"
PREFIX_DIRECTORY = (
    Path(__file__).parents[2] / "artifacts/formed-two-port-attempt-20261029"
)


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


def test_corridor_series_bulk_identities_and_curvature_exit_obstruction(reference):
    """Verify (C1)--(C9) in SINE_TWO_PORT_CAPTURE without running a step."""
    pi = reference.pi
    delta = pi / 9
    phi = I(Q(1, 1000))

    # (C4): degree-2 interior donor bulk vector v vanishes on ports/receiver
    # and isolates edge 5->6 against the 7 remaining donor bulk edges.
    v = [Q(0)] * 18
    for k in range(1, 8):
        v[(5 + k) % 9] = Q(8 - k, 7)
    diffs = tuple(v[right] - v[left] for left, right in reference.oriented_edges)
    assert diffs[5] == Q(1)
    bulk_rest = (0, 1, 2, 3, 6, 7, 8)
    assert all(diffs[idx] == -Q(1, 7) for idx in bulk_rest)
    assert diffs[4] == Q(0)
    assert all(diffs[idx] == Q(0) for idx in range(9, 20))

    # (C5)--(C9): state on the extended path (D3) at u_crit = 5*pi/576.
    u_crit = 5 * pi / 576
    h = [-u_crit if k != 5 else 8 * u_crit for k in range(9)]
    eta_d = [I(0)]
    acc = I(0)
    for k in range(8):
        acc = acc + h[k]
        eta_d.append(acc)
    eta_r = [delta - phi - Q(9, 2) * u_crit for _ in range(9)]
    eta = eta_d + eta_r
    mean = sum((d * val for d, val in zip(reference.degrees, eta)), I(0)) / 40
    xi_crit = tuple(val - mean for val in eta)
    assert sum((d * val for d, val in zip(reference.degrees, xi_crit)), I(0)).contains(
        0
    )

    gaps = reference.angles(xi_crit)
    assert gaps[5].lo > (pi / 2).hi and gaps[5].hi < (5 * pi / 9).lo
    rates = reference.field(xi_crit)
    g_prime = tuple(
        rates[right] - rates[left] for left, right in reference.oriented_edges
    )

    assert g_prime[5].lo > Q(31, 576) > 0
    assert g_prime[4].hi < -Q(13, 96) < 0
    assert (g_prime[6] + g_prime[3]).lo > Q(41, 576) > 0
    assert all(g_prime[idx].contains(0) for idx in (0, 1, 2, 7, 8))

    c_edges = tuple(cos(g) for g in gaps)
    q_bulk = sum((c_edges[idx] * (diffs[idx] ** 2) for idx in range(20)), I(0))
    assert q_bulk.hi < -Q(181, 14112) < -Q(1, 96) < 0

    dq_bulk = -sin(gaps[5]) * g_prime[5] - (sin(gaps[6]) / 49) * (
        g_prime[6] + g_prime[3]
    )
    assert dq_bulk.hi < -Q(72145, 1354752) < -Q(1, 19) < 0


def test_centralized_edge_duals_corridor_and_fifth_step_acute_exit_certificate(
    reference,
):
    """Verify C/P bounds after admitting the complete retained four-step prefix."""

    duals = reference.edge_dual_norms()
    assert len(duals) == 20
    # All 12 degree-two bulk edges have exact dual norm 1.
    bulk_degree_two = (0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17)
    assert all(duals[idx] == Q(1) for idx in bulk_degree_two)
    # Port-to-bulk edges have dual norm sqrt(5/6) < 1; port-to-port have sqrt(2/3) < 1.
    assert all(Q(5, 6) <= duals[idx] ** 2 < Q(1) for idx in (3, 5, 12, 14))
    assert all(Q(2, 3) <= duals[idx] ** 2 < Q(5, 6) for idx in (4, 13, 18, 19))

    slacks = reference.corridor_domain(reference.initial_center)
    assert len(slacks) == 20 and min(slacks) > 0
    with pytest.raises(ValueError, match="path-compensation"):
        reference.corridor_domain(
            reference.initial_center, kappa=Q(1, 16), donor_margin=Q(1, 64)
        )

    cert = owner.certify_formed_two_port_fifth_step_acute_exit(
        reference, PREFIX_DIRECTORY
    )
    assert cert.certified is True
    assert cert.active_edge_index == 5
    assert cert.reference_speed_upper_at_one < Q(1, 4)
    assert cert.second_derivative_coarse_upper < Q(11, 64)
    assert cert.second_derivative_tight_upper < Q(9, 64)
    assert cert.corridor_contradiction_excess_lower > reference.reference_margin
    assert cert.actual_full_law_excess_over_pi_half_lower > 0
    assert cert.hessian_exit_horizon == Q(7, 4)
    assert cert.psd_contradiction_slope_upper < Q(1, 10)
    assert cert.psd_contradiction_gap_excess_lower > Q(1, 32)
    assert cert.psd_contradiction_bulk_cosine_upper < Q(3, 100)
    assert cert.psd_contradiction_q_bulk_upper < -Q(1, 256) < 0
    assert min(cert.prefix_corridor_slacks) > 0
    assert len(cert.prefix_evidence_receipts) == 8
    assert not hasattr(cert, "hessian_exit_q_bulk_upper")

    origin_cert = owner.admit_formed_two_port_origin_window(reference, Q(355, 1017))
    assert origin_cert.certified is True
    assert origin_cert.receiver_origin_rational_offset == Q(355, 1017)
    assert origin_cert.receiver_origin_pi_coefficient == 0
    assert origin_cert.nodewise_form_error_upper <= Q(1, 65536)
    assert origin_cert.nodewise_phase_error_upper <= Q(1, 65536)
    assert origin_cert.contact_work_upper_bound < Q(1, 8) < Q(1, 4)
    rejected_origin = owner.admit_formed_two_port_origin_window(reference, Q(1, 1000))
    assert rejected_origin.certified is False


def test_prefix_admission_recomputes_all_four_corridor_tubes(reference, monkeypatch):
    observed = []
    original = owner.FormedTwoPortReference.corridor_domain

    def record(self, values, **kwargs):
        observed.append(values)
        return original(self, values, **kwargs)

    monkeypatch.setattr(owner.FormedTwoPortReference, "corridor_domain", record)
    prefix = owner.admit_formed_two_port_acute_exit_prefix(reference, PREFIX_DIRECTORY)
    assert len(observed) == 4
    assert min(prefix.corridor_slacks) > 0
    assert prefix.endpoint_radius < Q(1, 2**26)
    assert len(prefix.endpoint_center) == 18
    # A dataclass carrying invented cached values cannot enter the public API.
    with pytest.raises(TypeError):
        owner.certify_formed_two_port_fifth_step_acute_exit(reference, prefix)


def test_prefix_receipt_corruption_rejected_before_corridor_or_field(
    reference, monkeypatch
):
    read = owner.read_bytes_bounded

    def corrupt(path, **kwargs):
        data = read(path, **kwargs)
        return data[:-1] + b" " if Path(path).name == "0000.json" else data

    def forbidden(*args, **kwargs):
        pytest.fail("corrupt evidence must not reach domain or field evaluation")

    monkeypatch.setattr(owner, "read_bytes_bounded", corrupt)
    monkeypatch.setattr(owner.FormedTwoPortReference, "corridor_domain", forbidden)
    monkeypatch.setattr(owner.FormedTwoPortReference, "field", forbidden)
    with pytest.raises(ValueError, match="prefix receipt differs"):
        owner.certify_formed_two_port_fifth_step_acute_exit(reference, PREFIX_DIRECTORY)


def test_prefix_does_not_accept_cached_old_acute_slacks_for_new_corridor(
    reference, monkeypatch
):
    def outside(self, values, **kwargs):
        return (Q(-1),) + (Q(1),) * 19

    monkeypatch.setattr(owner.FormedTwoPortReference, "corridor_domain", outside)
    with pytest.raises(ValueError, match="buffered corridor"):
        owner.admit_formed_two_port_acute_exit_prefix(reference, PREFIX_DIRECTORY)


@pytest.mark.parametrize(
    "field", ("initial_center", "endpoint_center", "endpoint_radius")
)
def test_prefix_checks_reconstructed_step_primitives(reference, monkeypatch, field):
    from tnfr.research import formed_two_port_run

    read = formed_two_port_run.read_step_certificate

    def inconsistent(directory, index):
        result = read(directory, index)
        if index == 0:
            result[field] = (
                (Q(1),) + result[field][1:] if field.endswith("center") else Q(0)
            )
        return result

    monkeypatch.setattr(formed_two_port_run, "read_step_certificate", inconsistent)
    with pytest.raises(ValueError, match="chain differs|projection differs"):
        owner.admit_formed_two_port_acute_exit_prefix(reference, PREFIX_DIRECTORY)


@pytest.mark.parametrize("offset, coefficient", ((Q(355, 1017), Q(0)), (Q(0), Q(1, 9))))
def test_origin_transport_full_state_lifts_gauge_and_member_means(
    reference, offset, coefficient
):
    epsilon = Q(1, 10**32)
    vd, ud, vr, ur = ([Q(0)] * 9 for _ in range(4))
    vd[4], vd[0] = epsilon / 4, -epsilon / 4
    vr[5], vr[1] = epsilon / 5, -epsilon / 5
    ud[4], ud[0] = epsilon / 2, -epsilon / 2
    ur[5], ur[1] = epsilon / 3, -epsilon / 3
    transported = owner.transport_formed_two_port_origin_source(
        reference, offset, vd, ud, vr, ur, phi_pi_coefficient=coefficient
    )
    pi = reference.pi
    delta = (coefficient - Q(1, 9)) * pi + offset
    for old in range(18):
        component, local = divmod(old, 9)
        new = 9 * component + (local + 4) % 9
        assert transported.original_indices[old] == new
        assert transported.form[old] == (vd if component == 0 else vr)[new % 9]
        winding = 2 if component == 0 else 1
        lift = 2 * winding * pi if local >= 5 else I(0)
        independently_transported = (
            transported.original_phase[new]
            + lift
            - (offset / 2 + (coefficient / 2 + Q(73, 60)) * pi)
        )
        assert _overlap(transported.phase[old], independently_transported)
        expected_error = (ud if component == 0 else ur)[new % 9] + (
            -delta / 2 if component == 0 else delta / 2
        )
        assert transported.phase_deviation[old] == expected_error
        assert expected_error.abs_max <= Q(1, 65536)
    assert transported.joined_phase_residual_mean == epsilon / 48 != 0
    assert transported.joined_form_mean == (epsilon / 4 + epsilon / 5) / 40 != 0
    if coefficient == Q(1, 9):
        assert transported.window.origin_deviation_from_midpoint_upper == 0
    else:
        assert not transported.phase_deviation[2].contains(0)


def test_origin_rejects_nonexact_scalars_invalid_residuals_and_changed_reference(
    reference,
):
    zero = (Q(0),) * 9
    for value in (True, 0.3):
        with pytest.raises(ValueError, match="exact record"):
            owner.admit_formed_two_port_origin_window(reference, value)
    with pytest.raises(ValueError, match="reference primitives"):
        owner.admit_formed_two_port_origin_window(replace(reference, pi=I(0)), Q(0))
    with pytest.raises(ValueError, match="A1 residual bounds"):
        owner.transport_formed_two_port_origin_source(
            reference, Q(355, 1017), (Q(1, 10**32),) + zero[1:], zero, zero, zero
        )
    with pytest.raises(ValueError, match="outside the admitted window"):
        owner.transport_formed_two_port_origin_source(
            reference, Q(1, 1000), zero, zero, zero, zero
        )


@pytest.mark.parametrize(
    "change",
    ("boolean_center", "float_radius", "float_metric", "boolean_edge", "float_margin"),
)
def test_fixed_reference_rejects_equal_but_unadmitted_scalar_types(reference, change):
    if change == "boolean_center":
        altered = replace(reference, initial_center=(False,) * 18)
    elif change == "float_radius":
        altered = replace(reference, initial_radius=0.0)
    elif change == "float_metric":
        metric = ((2.0,) + reference.metric[0][1:],) + reference.metric[1:]
        altered = replace(reference, metric=metric)
    elif change == "boolean_edge":
        altered = replace(
            reference, oriented_edges=((False, 1),) + reference.oriented_edges[1:]
        )
    else:
        altered = replace(reference, reference_margin=1 / 2048)
    assert altered == reference  # Python equality alone cannot admit physical scalars.
    with pytest.raises(ValueError, match="reference primitives"):
        owner.admit_formed_two_port_origin_window(altered)
    with pytest.raises(ValueError, match="reference primitives"):
        owner.certify_formed_two_port_fifth_step_acute_exit(altered, PREFIX_DIRECTORY)


def test_reference_dataclass_requires_materialized_scalars_not_json_tags(reference):
    altered = replace(reference, initial_radius={"numerator": 0, "denominator": 1})
    with pytest.raises(ValueError, match="reference primitives"):
        owner.admit_formed_two_port_origin_window(altered)


def test_post_acute_bound_rebuilds_short_prefix_without_field_or_cached_comparison(
    reference, monkeypatch
):
    from tnfr.research import formed_two_port_run

    read = formed_two_port_run.read_step_certificate
    observed = []

    def record(directory, index):
        observed.append(index)
        return read(directory, index)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "post-acute arithmetic must not evaluate a field or reuse M6 caches"
        )

    monkeypatch.setattr(formed_two_port_run, "read_step_certificate", record)
    monkeypatch.setattr(owner.FormedTwoPortReference, "field", forbidden)
    monkeypatch.setattr(owner, "formed_two_port_comparison_bounds", forbidden)
    result = owner.bound_formed_two_port_post_acute_window(reference, PREFIX_DIRECTORY)
    epsilon = Q(1, 10**32)
    z0 = epsilon / 1023
    v0 = 3 * epsilon + z0
    assert observed == [0, 1, 2, 3]
    assert result.certified
    assert (result.start_slow_time, result.end_slow_time) == (Q(1), Q(7, 4))
    assert result.source_scaled_form_upper_bound == z0
    assert result.source_mixed_error_upper_bound == v0
    assert result.initial_mixed_error_upper_bound - v0 == Q(1, 100000)
    assert result.initial_mixed_error_upper_bound < Q(1, 90000)
    # The initial value solves the retained short-prefix feedback inequality.
    assert (1 - Q(1, 50000)) * result.initial_scaled_form_upper_bound == (
        z0 + Q(1, 100000) * (Q(3, 5) + 2 * result.initial_mixed_error_upper_bound)
    )
    assert result.initial_scaled_form_upper_bound < Q(1, 100000)
    assert result.scaled_form_norm_upper_bound == Q(7, 100000)
    assert result.phase_error_upper_bound < Q(73, 200000) < reference.reference_margin
    assert result.mixed_error_upper_bound + result.scaled_form_norm_upper_bound == (
        result.phase_error_upper_bound
    )
    assert result.centered_original_form_norm_upper_bound == Q(1407, 6250)
    assert 0 < result.reference_numerical_radius_at_start < Q(1, 2**26)
    assert len(result.prefix_evidence_receipts) == 8
    assert min(result.prefix_corridor_slacks) > 0
    assert "no_capture_winding_change_or_actual_Hessian_sign_conclusion" in result.scope


def test_post_acute_exponential_majorant_has_exact_geometric_tail(reference):
    from math import factorial

    result = owner.bound_formed_two_port_post_acute_window(reference, PREFIX_DIRECTORY)
    # Independent tighter degree-six remainder: after its first omitted term,
    # every exponential term ratio is <= (1/2)/8. This checks the declared
    # coarse majorant without an exponential call or dynamical evaluation.
    partial = sum((Q(1, 2**k * factorial(k)) for k in range(7)), Q(0))
    tail = Q(1, 2**7 * factorial(7)) / (1 - Q(1, 16))
    assert partial < partial + tail < result.half_exponential_upper_bound
    assert result.half_exponential_upper_bound == Q(277, 168)
    assert result.half_exponential_upper_bound**3 == result.growth_factor_upper_bound
    assert result.growth_factor_upper_bound < Q(33, 20) ** 3 < Q(9, 2)
    assert result.phase_error_design_ceiling == Q(9, 2) * (Q(1, 90000) + Q(7, 100000))


def test_post_acute_bound_rejects_wrong_reference_and_unbound_evidence(
    reference, monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="reference primitives"):
        owner.bound_formed_two_port_post_acute_window(
            replace(reference, initial_radius=False), PREFIX_DIRECTORY
        )
    with pytest.raises(OSError):
        owner.bound_formed_two_port_post_acute_window(reference, tmp_path)
    read = owner.read_bytes_bounded

    def corrupt(path, **kwargs):
        data = read(path, **kwargs)
        return data[:-1] + b" " if Path(path).name == "0003.json" else data

    monkeypatch.setattr(owner, "read_bytes_bounded", corrupt)
    with pytest.raises(ValueError, match="prefix receipt differs"):
        owner.bound_formed_two_port_post_acute_window(reference, PREFIX_DIRECTORY)
