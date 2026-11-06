"""Full-field local derivative controls; no source or trajectory evaluation."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_alternative_contact as contact_owner
from tnfr.physics import _organized_alternative_outer_acquisition as source_owner
from tnfr.physics import _organized_contact_preparation_sensitivity as owner

_ZERO = (Q(0),) * 54
_INTERNAL = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
_OUTER = tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for j in range(9) for c in range(5))
_CONTACT = tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
_EDGES = _INTERNAL + _OUTER + _CONTACT
_WITNESS = _ZERO[:9] + (Q(1), Q(-1)) + _ZERO[11:]


@pytest.fixture(scope="module", autouse=True)
def no_acquired_source_or_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "local preparation derivative attempted source or trajectory evaluation"
        )

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, attribute in (
            (contact_owner, "_assess_organized_alternative_contact"),
            (contact_owner, "_assess_organized_alternative_outer_acquisition"),
            (contact_owner, "_assess_constitutive_recovery"),
            (source_owner, "_assess_organized_alternative_outer_acquisition"),
            (source_owner, "_assess_constitutive_recovery"),
        ):
            patch.setattr(module, attribute, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module", params=(0, 1), ids=("sine", "alternative"))
def deformation(request):
    return request.param


@pytest.fixture(scope="module")
def phase_witness(deformation):
    return owner._observe_organized_contact_preparation_sensitivity(
        deformation=deformation,
        beta_turns=0,
        form_direction=_ZERO,
        phase_direction=_WITNESS,
        time_order=1,
    )


@pytest.fixture(scope="module")
def form_witness(deformation):
    return owner._observe_organized_contact_preparation_sensitivity(
        deformation=deformation,
        beta_turns=0,
        form_direction=_WITNESS,
        phase_direction=_ZERO,
        time_order=3,
    )


@pytest.fixture(scope="module")
def mixed_directions():
    u = tuple(Q((7 * i) % 19 - 9, 64) for i in range(54))
    v = tuple(Q((11 * i) % 23 - 11, 128) for i in range(54))
    return tuple(value - sum(u) / 54 for value in u), tuple(
        value - sum(v) / 54 for value in v
    )


@pytest.fixture(scope="module")
def mixed(deformation, mixed_directions):
    u, v = mixed_directions
    # General full-field wiring control, separate from the named witnesses.
    return owner._observe_organized_contact_preparation_sensitivity(
        deformation=deformation,
        beta_turns=Q(1, 17),
        form_direction=u,
        phase_direction=v,
        time_order=3,
    )


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, box, value):
    if box == I(0):
        assert abs(value) < mp.mpf("1e-80")
    else:
        assert _mp(mp, box.lo) <= value <= _mp(mp, box.hi)


def _laplacian(row):
    result = [0] * 54
    for a, b in _EDGES:
        gap = row[a] - row[b]
        result[a] += gap
        result[b] -= gap
    return result


def _current(mp, angle, law, derivative=0):
    if derivative == 0:
        return mp.sin(angle) + law * mp.sin(9 * angle) * (1 - mp.cos(9 * angle)) / 2187
    if derivative == 1:
        return mp.cos(angle) + law * (mp.cos(9 * angle) - mp.cos(18 * angle)) / 243
    return -mp.sin(angle) + law * (-mp.sin(9 * angle) + 2 * mp.sin(18 * angle)) / 27


def _potential(mp, angle, law):
    return 1 - mp.cos(angle) + law * (1 - mp.cos(9 * angle)) ** 2 / (6 * 9**4)


def _complete_local_derivatives(mp, epsilon, law, beta, u, v):
    """Differentiate complete state rows in time before varying preparation.

    This route does not use the production amplitude recurrence. It evaluates
    F, G, dF/dtau and d²F/dtau² at one supplied state, not a trajectory.
    mpmath then differentiates this vector with respect to epsilon at zero.
    """
    gamma = 1 / (1023 * mp.pi)
    form = [epsilon * _mp(mp, value) for value in u]
    phase = [
        2 * mp.pi * (_mp(mp, Q(j - 4, 9) + (Q(c - 2, 5) if c < 5 else beta)))
        + epsilon * _mp(mp, v[9 * c + j])
        for c in range(6)
        for j in range(9)
    ]

    def derivative(order, *directions):
        result = [mp.mpf(0)] * 54
        for a, b in _EDGES:
            term = _current(mp, phase[b] - phase[a], law, order)
            for direction in directions:
                term *= direction[b] - direction[a]
            result[a] += term
            result[b] -= term
        return result

    lf, s = _laplacian(form), derivative(0)
    f = [(-value + gamma * current) / 4 for value, current in zip(lf, s)]
    g = [gamma * value / 4 for value in lf]
    lf1, jg = _laplacian(f), derivative(1, g)
    f2 = [(-value + gamma * current) / 4 for value, current in zip(lf1, jg)]
    g2 = [gamma * value / 4 for value in lf1]
    lf2, hgg, jg2 = _laplacian(f2), derivative(2, g, g), derivative(1, g2)
    f3 = [
        (-value + gamma * (second + first)) / 4
        for value, second, first in zip(lf2, hgg, jg2)
    ]
    g3 = [gamma * value / 4 for value in lf2]
    costs = [
        (form[a] - form[b]) ** 2 / 2 + _potential(mp, phase[b] - phase[a], law)
        for a, b in _EDGES
    ]
    return mp.matrix(
        [value for row in (form, f, f2, f3, phase, g, g2, g3) for value in row]
        + [sum(costs), sum(costs[99:])]
    )


@pytest.fixture(scope="module")
def independent_derivatives(mp, mixed, mixed_directions):
    u, v = mixed_directions

    def field(epsilon):
        return _complete_local_derivatives(
            mp, epsilon, int(mixed.deformation), mixed.beta_turns, u, v
        )

    return field(mp.mpf(0)), mp.diff(field, 0), mp.diff(field, 0, 2) / 2


def test_geometry_and_complete_local_time_rows_match_independent_field(
    mp, mixed, independent_derivatives
):
    assert tuple((a, b) for a, b, _ in mixed.geometry.weighted_edges) == _EDGES
    assert mixed.geometry.support_degrees == (5,) * 45 + (7,) * 9
    assert mixed.geometry.clock_factor == Q(1, 4)
    assert mixed.geometry.held_capacities == (Q(1),) * 54
    assert not any("pi_over_two" in scope for scope in mixed.geometry.scope)
    for rows, expected in (
        (
            mixed.linear_form_time_derivative_bounds
            + mixed.linear_phase_time_derivative_bounds,
            independent_derivatives[1],
        ),
        (
            mixed.quadratic_form_time_derivative_bounds
            + mixed.quadratic_phase_time_derivative_bounds,
            independent_derivatives[2],
        ),
    ):
        for index, interval in enumerate(value for row in rows for value in row):
            _contains(mp, interval, expected[index])
    for rows in (
        mixed.linear_form_time_derivative_bounds,
        mixed.linear_phase_time_derivative_bounds,
        mixed.quadratic_form_time_derivative_bounds,
        mixed.quadratic_phase_time_derivative_bounds,
    ):
        assert all(sum(row, I(0)).contains(0) for row in rows)


def test_storage_and_attachment_coefficients_use_half_second_amplitude_derivative(
    mp, mixed, independent_derivatives
):
    for amplitude, expected in enumerate(independent_derivatives):
        _contains(
            mp, mixed.initial_storage_amplitude_coefficients[amplitude], expected[432]
        )
        _contains(
            mp, mixed.contact_work_amplitude_coefficients[amplitude], expected[433]
        )
    expected_full = sum(
        (mixed.form_direction[a] - mixed.form_direction[b]) ** 2 / 2 for a, b in _EDGES
    )
    expected_contact = sum(
        (mixed.form_direction[a] - mixed.form_direction[b]) ** 2 / 2
        for a, b in _CONTACT
    )
    assert mixed.initial_form_storage_quadratic_coefficient == expected_full
    assert mixed.contact_form_storage_quadratic_coefficient == expected_contact
    assert expected_full > expected_contact > 0


def test_mean_projection_and_contact_covariances_retain_primitive_information(
    mp, mixed
):
    for raw, means, internal in (
        (
            mixed.form_direction,
            mixed.child_form_direction_means,
            mixed.internal_form_direction,
        ),
        (
            mixed.phase_direction,
            mixed.child_phase_direction_means,
            mixed.internal_phase_direction,
        ),
    ):
        assert tuple(value + means[i // 9] for i, value in enumerate(internal)) == raw
        assert all(sum(internal[c * 9 : c * 9 + 9]) == 0 for c in range(6))
    assert not mixed.pure_internal_direction
    assert not mixed.linear_donor_identically_zero_via_slot_symmetry
    assert not mixed.donor_identically_zero_via_reflection
    assert mixed.reflection_permutation is None
    gamma = 1 / (1023 * mp.pi)
    expected_linear = -mp.mpf(3) / 2 * _mp(mp, mixed.child_form_direction_means[5])
    expected_quadratic = mp.mpf(0)
    for c in range(5):
        gap = 2 * mp.pi * (_mp(mp, mixed.beta_turns) - mp.mpf(c - 2) / 5)
        means = mixed.child_phase_direction_means
        expected_linear -= (
            gamma
            * _current(mp, gap, int(mixed.deformation), 1)
            * _mp(mp, means[5] - means[c])
            / 4
        )
        expected_quadratic -= (
            gamma
            * _current(mp, gap, int(mixed.deformation), 2)
            * _mp(mp, mixed.contact_phase_gap_second_moments[c])
            / 8
        )
    for interval in (
        mixed.closed_linear_donor_slope_bounds,
        mixed.linear_donor_mean_time_derivative_bounds[1],
    ):
        _contains(mp, interval, expected_linear)
    for interval in (
        mixed.closed_quadratic_donor_slope_bounds,
        mixed.quadratic_donor_mean_time_derivative_bounds[1],
    ):
        _contains(mp, interval, expected_quadratic)


def test_phase_internal_witness_has_quadratic_first_time_leakage(mp, phase_witness):
    report = phase_witness
    gamma = 1 / (1023 * mp.pi)
    expected = -gamma * _current(mp, 2 * mp.pi / 5, int(report.deformation), 2) / 36
    assert expected > 0
    assert report.computed_time_order == 1
    assert len(report.quadratic_form_time_derivative_bounds) == 2
    assert report.certified_quadratic_nonzero_time_orders == (1,)
    _contains(mp, report.quadratic_donor_mean_time_derivative_bounds[1], expected)
    assert report.quadratic_donor_mean_time_derivative_bounds[1].lo > 0
    assert report.contact_phase_gap_second_moments == (0, Q(2, 9), 0, 0, 0)
    assert report.contact_form_gap_second_moments == (0,) * 5
    assert report.pure_internal_direction
    assert report.linear_donor_identically_zero_via_slot_symmetry
    assert all(
        value.contains(0) for value in report.linear_donor_mean_time_derivative_bounds
    )
    assert not report.donor_identically_zero_via_reflection


def test_form_internal_witness_leaks_first_at_third_time_derivative(mp, form_witness):
    report = form_witness
    gamma = 1 / (1023 * mp.pi)
    q0 = _current(mp, 4 * mp.pi / 5, int(report.deformation), 2)
    q1 = _current(mp, 2 * mp.pi / 5, int(report.deformation), 2)
    expected = gamma**3 * (q0 - 49 * q1) / 288
    assert expected > 0
    assert report.quadratic_donor_mean_time_derivative_bounds[:3] == (I(0),) * 3
    assert report.certified_quadratic_nonzero_time_orders == (3,)
    _contains(mp, report.quadratic_donor_mean_time_derivative_bounds[3], expected)
    assert report.quadratic_donor_mean_time_derivative_bounds[3].lo > 0
    assert report.form_direction_laplacian == tuple(_laplacian(_WITNESS))
    assert report.contact_form_gap_second_moments == (0, Q(2, 9), 0, 0, 0)
    assert report.contact_laplacian_form_gap_second_moments == (
        0,
        Q(100, 9),
        0,
        Q(2, 9),
        Q(2, 9),
    )
    assert report.pure_internal_direction
    assert report.linear_donor_identically_zero_via_slot_symmetry
    assert all(
        value.contains(0) for value in report.linear_donor_mean_time_derivative_bounds
    )
    assert not report.donor_identically_zero_via_reflection


def test_witness_initial_winding_has_an_explicit_nonzero_amplitude_domain(
    mp, phase_witness, form_witness
):
    assert phase_witness.sufficient_initial_winding_amplitude_radius >= Q(1, 10)
    assert form_witness.sufficient_initial_winding_amplitude_radius == 1
    for edges, ideal_gap in ((_INTERNAL, 2 * mp.pi / 9), (_OUTER, 2 * mp.pi / 5)):
        for a, b in edges:
            gap = ideal_gap + mp.mpf("0.1") * (_WITNESS[b] - _WITNESS[a])
            assert 0 < gap < mp.pi / 2
    assert phase_witness.marked_phase_direction_gap_maximum == 2
    assert phase_witness.initial_marked_acute_margin_lower_bound > 0


def test_reflection_is_an_exact_stronger_direction_condition():
    direction = list(_ZERO)
    direction[18], direction[26] = Q(1), Q(-1)
    report = owner._observe_organized_contact_preparation_sensitivity(
        deformation=1,
        beta_turns=0,
        form_direction=direction,
        phase_direction=direction,
        time_order=1,
    )
    assert report.pure_internal_direction
    assert report.donor_identically_zero_via_reflection
    assert report.direction_preserves_reflection
    permutation = report.reflection_permutation
    assert permutation is not None
    assert all(permutation[permutation[i]] == i for i in range(54))
    assert all(direction[i] == -direction[permutation[i]] for i in range(54))
    assert report.certified_quadratic_nonzero_time_orders == ()


def test_large_integer_origin_turns_preserve_periodic_rows_but_retain_the_lift(mixed):
    turn_shift = 10**80
    shifted = owner._observe_organized_contact_preparation_sensitivity(
        deformation=mixed.deformation,
        beta_turns=mixed.beta_turns + turn_shift,
        form_direction=mixed.form_direction,
        phase_direction=mixed.phase_direction,
        time_order=1,
    )
    assert shifted.beta_turns - mixed.beta_turns == turn_shift
    assert shifted.beta_bounds != mixed.beta_bounds
    assert shifted.geometry.external_phase_origin_bounds == shifted.beta_bounds
    assert shifted.ideal_phase_turns[:45] == mixed.ideal_phase_turns[:45]
    assert all(
        after - before == turn_shift
        for after, before in zip(
            shifted.ideal_phase_turns[45:], mixed.ideal_phase_turns[45:]
        )
    )
    for attribute in (
        "ideal_current_bounds",
        "edge_current_derivative_bounds",
        "carrier_initial_phase_acceleration_bounds",
        "initial_storage_amplitude_coefficients",
        "contact_work_amplitude_coefficients",
        "closed_linear_donor_slope_bounds",
        "closed_quadratic_donor_slope_bounds",
        "reflection_permutation",
    ):
        assert getattr(shifted, attribute) == getattr(mixed, attribute)
    for attribute in (
        "linear_form_time_derivative_bounds",
        "linear_phase_time_derivative_bounds",
        "quadratic_form_time_derivative_bounds",
        "quadratic_phase_time_derivative_bounds",
        "linear_donor_mean_time_derivative_bounds",
        "quadratic_donor_mean_time_derivative_bounds",
    ):
        assert getattr(shifted, attribute) == getattr(mixed, attribute)[:2]


def test_nonzero_reflection_origin_and_unprotected_origin_use_exact_sector_index(
    deformation,
):
    # At beta=pi/5 the protected child is c=0, unlike c=2 at beta=0.
    direction = (Q(1),) + _ZERO[1:8] + (Q(-1),) + _ZERO[9:]
    protected = owner._observe_organized_contact_preparation_sensitivity(
        deformation=deformation,
        beta_turns=Q(1, 10),
        form_direction=direction,
        phase_direction=direction,
        time_order=1,
    )
    unprotected = owner._observe_organized_contact_preparation_sensitivity(
        deformation=deformation,
        beta_turns=Q(1, 20),
        form_direction=direction,
        phase_direction=direction,
        time_order=1,
    )
    assert protected.direction_preserves_reflection
    assert protected.donor_identically_zero_via_reflection
    assert protected.reflection_permutation[:9] == tuple(range(8, -1, -1))
    assert protected.certified_quadratic_nonzero_time_orders == ()
    assert unprotected.reflection_permutation is None
    assert not unprotected.direction_preserves_reflection
    assert not unprotected.donor_identically_zero_via_reflection
    assert protected.linear_donor_identically_zero_via_slot_symmetry
    assert unprotected.linear_donor_identically_zero_via_slot_symmetry


def test_tiny_internal_directions_retain_exact_primitives_without_false_nonzero():
    tiny = Q(1, 10**400)
    direction = tuple(tiny * value for value in _WITNESS)
    report = owner._observe_organized_contact_preparation_sensitivity(
        deformation=0,
        beta_turns=0,
        form_direction=_ZERO,
        phase_direction=direction,
        time_order=1,
    )
    assert report.phase_direction == direction
    assert report.internal_phase_direction == direction
    assert report.contact_phase_gap_second_moments[1] == 2 * tiny**2 / 9
    assert report.pure_internal_direction
    assert report.linear_donor_identically_zero_via_slot_symmetry
    assert not report.donor_identically_zero_via_reflection
    assert report.quadratic_donor_mean_time_derivative_bounds[1].contains(0)
    assert report.certified_quadratic_nonzero_time_orders == ()


@pytest.mark.parametrize(
    "field,value",
    (
        ("deformation", True),
        ("deformation", Q(1, 2)),
        ("deformation", float("nan")),
        ("beta_turns", False),
        ("beta_turns", float("inf")),
        ("form_direction", _ZERO[:-1]),
        ("phase_direction", _ZERO + (0,)),
        ("form_direction", (True,) + _ZERO[1:]),
        ("phase_direction", (float("nan"),) + _ZERO[1:]),
        ("form_direction", (Q(1, 10**400),) + _ZERO[1:]),
        ("phase_direction", (Q(1, 10**400),) + _ZERO[1:]),
        ("time_order", True),
        ("time_order", 0),
        ("time_order", 4),
        ("time_order", 1.0),
        ("time_order", Q(1)),
    ),
)
def test_invalid_primitives_reject_before_geometry_or_trigonometry(
    monkeypatch, field, value
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached geometry or trigonometry")

    for attribute in ("_geometry", "pi_interval", "sin", "cos", "_edge_law"):
        monkeypatch.setattr(owner, attribute, forbidden)
    arguments = dict(
        deformation=0,
        beta_turns=0,
        form_direction=_ZERO,
        phase_direction=_ZERO,
        time_order=1,
    )
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_contact_preparation_sensitivity(**arguments)


def test_report_retains_scope_stopping_rule_and_immutable_metadata(
    phase_witness, form_witness
):
    for report in (phase_witness, form_witness):
        assert report.maximum_time_order == 3
        assert report.maximum_amplitude_order == 2
        assert report.protocol_revision == "b2801170"
        assert "divided by two" in report.amplitude_convention
        assert "not divided by time factorial" in report.time_convention
        assert (
            "interval_containment_of_zero_is_not_an_exact_vanishing_claim"
            in report.scope
        )
        assert (
            "no_acquired_source_residual_distribution_trajectory_or_finite_discriminator"
            in report.scope
        )
        with pytest.raises(FrozenInstanceError):
            report.computed_time_order = 12
