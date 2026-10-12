"""Exact algebra and independent full-field checks for the bounded contact jets."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.physics import _organized_alternative_contact as contact_owner
from tnfr.physics import _organized_alternative_outer_acquisition as source_owner
from tnfr.physics import _organized_contact_dynamic_cancellation as owner
from tnfr.physics._phase_boundary_information import _evaluate_boundary_response
from tnfr.physics._sine_class_constitutive_comparison import (
    _current_derivative_bound,
    _current_fourier_coefficients,
    _edge_law,
)


@pytest.fixture(scope="module", autouse=True)
def no_acquired_or_finite_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("local symbolic jet attempted source or finite response evaluation")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(contact_owner, "_assess_organized_alternative_contact", forbidden)
        patch.setattr(
            source_owner, "_assess_organized_alternative_outer_acquisition", forbidden
        )
        yield


@pytest.fixture(scope="module")
def sine():
    return owner._derive_organized_contact_dynamic_cancellation(deformation=0)


@pytest.fixture(scope="module")
def alternative():
    return owner._derive_organized_contact_dynamic_cancellation(deformation=1)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _q(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _evaluate(mp, terms, *, gamma, beta):
    return sum(
        (
            _q(mp, coefficient)
            * gamma**g
            * mp.exp(mp.j * beta * b)
            * mp.exp(mp.j * mp.pi * r / 10)
            for g, b, r, coefficient in terms
        ),
        mp.mpc(0),
    )


def _polynomial(terms):
    return {(g, b, r): coefficient for g, b, r, coefficient in terms}


def _sqrt_five():
    # sqrt(5)=1+2*zeta20^4−2*zeta20^6, since 2*cos(2pi/5)=(sqrt5−1)/2.
    return owner._add(
        owner._monomial(), owner._monomial(Q(2), root=4), owner._monomial(Q(-2), root=6)
    )


def _sine_monomial(harmonic):
    return owner._scale(
        owner._shift(
            owner._add(
                owner._monomial(beta=harmonic), owner._monomial(Q(-1), beta=-harmonic)
            ),
            root=5,
        ),
        Q(-1, 2),
    )


def test_root_basis_enforces_actual_cyclotomic_relation_and_imaginary_unit():
    one = owner._monomial()
    root = owner._monomial(root=1)
    assert owner._multiply(root, owner._monomial(root=19)) == one
    assert owner._multiply(
        owner._monomial(root=5), owner._monomial(root=5)
    ) == owner._monomial(Q(-1))
    assert owner._add(*(owner._monomial(root=4 * k) for k in range(5))) == {}
    assert (
        owner._add(
            owner._monomial(root=8),
            owner._monomial(Q(-1), root=6),
            owner._monomial(root=4),
            owner._monomial(Q(-1), root=2),
            one,
        )
        == {}
    )
    assert owner._multiply(_sqrt_five(), _sqrt_five()) == owner._monomial(Q(5))
    for power in range(20):
        assert all(0 <= degree <= 7 for degree, _ in owner._root_reduction(power))


@pytest.mark.parametrize("deformation", (Q(0), Q(1), Q(1, 2), Q(1, 10**400)))
def test_shared_fourier_law_matches_independent_factored_edge_owner(deformation):
    # Static identity controls for the existing law family do not launch a
    # third-law dynamical investigation or alter the frozen 0/1 jet domain.
    coefficients = _current_fourier_coefficients(deformation)
    phasor = Q(3, 5), Q(-4, 5)
    response = _evaluate_boundary_response(
        coefficients, gap_phasors=(phasor,), query_phasor=(1, 0)
    )
    assert response.current == _edge_law(*phasor, deformation)[1][0]
    for order in range(4):
        assert sum(
            abs(scale) * harmonic**order for harmonic, scale in coefficients
        ) == _current_derivative_bound(deformation, order)
    if deformation:
        assert coefficients[-1][1] < 0 < coefficients[-2][1]
    else:
        assert coefficients == ((1, Q(1)),)


def test_generic_first_orders_are_exact_not_rounding_thresholds(sine, alternative):
    for jet, order in (
        (sine.generic, 9),
        (alternative.generic, 3),
        (alternative.exceptional, 5),
    ):
        assert jet.computed_order == jet.first_nonzero_order == order
        assert jet.maximum_order == 12
        assert jet.donor_taylor_coefficients[:order] == ((),) * order
        assert jet.donor_derivatives[:order] == ((),) * order
        assert jet.donor_derivatives[order]
        assert jet.all_preceding_donor_derivatives_zero
        assert jet.status == "first_nonzero_donor_derivative"
        assert (
            len(jet.form_taylor_coefficients)
            == len(jet.phase_offset_taylor_coefficients)
            == 6
        )
        assert all(len(row) == order + 1 for row in jet.form_taylor_coefficients)
    assert sine.exceptional is None
    assert alternative.exceptional.origin_root_power == 1


def test_generic_alternative_coefficient_matches_chain_rule_root_sum(alternative):
    ell1 = owner._scale(
        owner._add(owner._monomial(Q(7)), owner._scale(_sqrt_five(), Q(-1))), Q(1, 2)
    )
    expected = owner._scale(
        owner._multiply(owner._shift(ell1, gamma=3), _sine_monomial(10)),
        Q(25, 64 * 2187),
    )
    assert _polynomial(alternative.generic.donor_derivatives[3]) == expected


def test_exceptional_fifth_coefficient_matches_independent_chain_rule_factor(
    alternative,
):
    # The complete-field fourth derivative of aggregate current gives this
    # cubic harmonic contraction; this is not fitted to the computed jet.
    sqrt5 = _sqrt_five()
    ell1 = owner._scale(
        owner._add(owner._monomial(Q(7)), owner._scale(sqrt5, Q(-1))), Q(1, 2)
    )
    ell2 = owner._scale(owner._add(owner._monomial(Q(7)), sqrt5), Q(1, 2))
    u = Q(1, 2187)
    bracket = owner._add(
        owner._scale(ell1, 2250 + 2430 * u), owner._scale(ell2, -15 + 405 * u)
    )
    expected = owner._scale(
        owner._shift(owner._multiply(ell2, bracket), gamma=5), -Q(5, 16 * 1024) * u**2
    )
    assert _polynomial(alternative.exceptional.donor_derivatives[5]) == expected
    closed = owner._scale(
        owner._shift(
            owner._add(owner._scale(sqrt5, Q(1750)), owner._monomial(Q(-828975))),
            gamma=5,
        ),
        Q(1, 528958107648),
    )
    assert expected == closed
    assert 1750 * 3 - 828975 < 0  # 2<sqrt5<3 gives a strict negative coefficient.


def test_sine_ninth_coefficient_has_the_proved_c5_selected_harmonic(sine):
    coefficient = owner._scale(
        owner._add(owner._scale(_sqrt_five(), Q(14595)), owner._monomial(Q(-52060))),
        Q(1, 524288),
    )
    expected = owner._multiply(owner._shift(coefficient, gamma=9), _sine_monomial(5))
    assert _polynomial(sine.generic.donor_derivatives[9]) == expected
    assert {row[1] for row in sine.generic.donor_derivatives[9]} == {-5, 5}
    assert 14595 * 3 - 52060 < 0


def _full_field(mp, form, phase, gamma, deformation):
    # Independent full54 edge enumeration and trigonometric law. This tests
    # the carrier embedding without reusing the symbolic wheel recurrence.
    edges = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
    edges += tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9)
    )
    edges += tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
    gradient, current = [mp.mpf(0)] * 54, [mp.mpf(0)] * 54
    for a, b in edges:
        difference = form[a] - form[b]
        angle = phase[b] - phase[a]
        force = (
            mp.sin(angle)
            + deformation * mp.sin(9 * angle) * (1 - mp.cos(9 * angle)) / 2187
        )
        gradient[a] += difference
        gradient[b] -= difference
        current[a] += force
        current[b] -= force
    return tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current)), tuple(
        gamma * a / 4 for a in gradient
    )


@pytest.mark.parametrize("which", ("sine", "alternative", "exceptional"))
def test_entire_jet_satisfies_both_full54_rows_to_its_declared_order(
    mp, sine, alternative, which
):
    jet = (
        sine.generic
        if which == "sine"
        else alternative.generic if which == "alternative" else alternative.exceptional
    )
    beta = (
        mp.mpf(2) / 7
        if jet.origin_root_power is None
        else mp.pi * jet.origin_root_power / 10
    )
    gamma = mp.mpf(3) / 11
    coefficients = [
        [
            [_evaluate(mp, terms, gamma=gamma, beta=beta) for terms in row]
            for row in field
        ]
        for field in (
            jet.form_taylor_coefficients,
            jet.phase_offset_taylor_coefficients,
        )
    ]
    assert max(
        abs(mp.im(value)) for field in coefficients for row in field for value in row
    ) < mp.mpf("1e-80")
    coefficients = [
        [[mp.re(value) for value in row] for row in field] for field in coefficients
    ]
    residuals = []
    for time in (mp.mpf(1) / 1000, mp.mpf(1) / 2000):
        values = [
            [
                sum(value * time**order for order, value in enumerate(row))
                for row in field
            ]
            for field in coefficients
        ]
        derivatives = [
            [
                sum(
                    order * value * time ** (order - 1)
                    for order, value in enumerate(row)
                    if order
                )
                for row in field
            ]
            for field in coefficients
        ]
        form = tuple(value for value in values[0] for _ in range(9))
        phase = tuple(
            2 * mp.pi * (j - 4) / 9
            + (2 * mp.pi * (c - 2) / 5 if c < 5 else beta)
            + values[1][c]
            for c in range(6)
            for j in range(9)
        )
        actual = _full_field(mp, form, phase, gamma, int(jet.deformation))
        residual = max(
            abs(actual[field][9 * c + j] - derivatives[field][c])
            for field in range(2)
            for c in range(6)
            for j in range(9)
        )
        assert residual < time**jet.computed_order
        residuals.append(residual)
    # The direct nonlinear full-field residual has the first omitted order,
    # rather than a clock, orientation or missing reaction error at low order.
    ratio = residuals[0] / residuals[1]
    assert (
        mp.mpf(".95") * 2**jet.computed_order
        < ratio
        < mp.mpf("1.05") * 2**jet.computed_order
    )


def test_exact_common_form_and_phase_sums_are_retained_in_every_coefficient(
    sine, alternative
):
    for jet in (sine.generic, alternative.generic, alternative.exceptional):
        for order in range(jet.computed_order + 1):
            assert (
                owner._add(
                    *(_polynomial(row[order]) for row in jet.form_taylor_coefficients)
                )
                == {}
            )
            assert (
                owner._add(
                    *(
                        _polynomial(row[order])
                        for row in jet.phase_offset_taylor_coefficients
                    )
                )
                == {}
            )


def test_protected_reflection_is_an_exact_complete_field_symmetry(mp):
    # One arbitrary state in the beta=0 fixed sector, with nonzero currents:
    # invariance follows from the odd law and support automorphism, not this
    # finite sample. The check independently exercises its full54 execution.
    forms = tuple(mp.mpf(value) / 13 for value in (2, -3, 0, 3, -2, 0))
    offsets = tuple(mp.mpf(value) / 17 for value in (-8, -4, 0, 4, 8, 0))
    phase = tuple(
        2 * mp.pi * (j - 4) / 9 + offsets[c] for c in range(6) for j in range(9)
    )
    x = tuple(value for value in forms for _ in range(9))
    permutation = tuple(
        9 * (4 - c if c < 5 else 5) + 8 - j for c in range(6) for j in range(9)
    )
    for deformation in (0, 1):
        dx, dt = _full_field(mp, x, phase, mp.mpf(1) / 9, deformation)
        assert max(abs(dx[i] + dx[permutation[i]]) for i in range(54)) < mp.mpf("1e-80")
        assert max(abs(dt[i] + dt[permutation[i]]) for i in range(54)) < mp.mpf("1e-80")
        assert abs(sum(dx[45:]) / 9) < mp.mpf("1e-80")
        assert max(map(abs, dx[:45])) > mp.mpf(".01")


@pytest.mark.parametrize(
    "bad", (True, None, "0", complex(0), float("nan"), float("inf"), -1, Q(1, 2), 2)
)
def test_law_admission_precedes_any_algebra(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_initial_edge_phasor",
        lambda *args: pytest.fail("invalid law reached algebra"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner._derive_organized_contact_dynamic_cancellation(deformation=bad)


@pytest.mark.parametrize("bad", (True, False, -1, 20, Q(1), 1.0, "1"))
def test_special_origin_requires_bounded_ordinary_integer(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_initial_edge_phasor",
        lambda *args: pytest.fail("invalid origin reached algebra"),
    )
    with pytest.raises(ValueError):
        owner._derive_contact_cancellation_jet(deformation=1, origin_root_power=bad)


@pytest.mark.parametrize(
    "extra",
    ("maximum_order", "horizon", "gamma", "source", "trajectory", "coefficients"),
)
def test_source_and_larger_derivative_budget_cannot_be_substituted(extra):
    with pytest.raises(TypeError):
        owner._derive_organized_contact_dynamic_cancellation(
            deformation=0, **{extra: 13}
        )


def test_reports_are_immutable(sine):
    with pytest.raises(FrozenInstanceError):
        sine.generic.computed_order = 12


def test_unresolved_zero_jet_stops_at_fixed_budget_on_synthetic_empty_support(
    monkeypatch,
):
    # Software stopping control only: this deliberately altered support is
    # not the wheel and establishes no protected sector for the actual law.
    monkeypatch.setattr(owner, "_EDGES", ())
    result = owner._derive_contact_cancellation_jet(deformation=0)
    assert result.computed_order == 12 and result.first_nonzero_order is None
    assert result.donor_derivatives == ((),) * 13
    assert result.status == "unresolved_zero_jet_at_order_twelve"
