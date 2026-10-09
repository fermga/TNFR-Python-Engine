"""Independent static controls for the composition-compatible counterfamily."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _composition_compatible_constitutive as owner


@pytest.fixture(scope="module", autouse=True)
def no_response_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert _mp(mp, interval.hi - interval.lo) < mp.mpf("1e-30")


def _potential(mp, angle, deformation):
    return (
        1 - mp.cos(angle) + deformation * mp.cos(angle) * (1 - mp.cos(9 * angle)) ** 2
    )


def _current(mp, angle, deformation):
    high = 1 - mp.cos(9 * angle)
    return mp.sin(angle) + deformation * (
        -mp.sin(angle) * high**2 + 18 * mp.cos(angle) * high * mp.sin(9 * angle)
    )


def _factor_coefficients(deformation):
    """Expand the factored potential as a Laurent polynomial independently."""

    def multiply(left, right):
        result = {}
        for first, a in left.items():
            for second, b in right.items():
                result[first + second] = result.get(first + second, Q(0)) + a * b
        return result

    cosine = {-1: Q(1, 2), 1: Q(1, 2)}
    high = {0: Q(1), -9: Q(-1, 2), 9: Q(-1, 2)}
    perturbation = multiply(cosine, multiply(high, high))
    potential = {0: Q(1), -1: Q(-1, 2), 1: Q(-1, 2)}
    for n, value in perturbation.items():
        potential[n] = potential.get(n, Q(0)) + deformation * value
    return {n: -2 * n * value for n, value in potential.items() if n > 0}


def _geometry(interface):
    offsets = (0,) if interface == "matching" else (-1, 0, 1)
    cycles = tuple(
        (9 * component + local, 9 * component + (local + 1) % 9)
        for component in range(2)
        for local in range(9)
    )
    cross = tuple(
        (local, 9 + (local + offset) % 9) for local in range(9) for offset in offsets
    )
    edges = cycles + cross
    degrees = tuple(sum(node in edge for edge in edges) for node in range(18))
    return edges, cross, degrees


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 324)))
def test_exact_moments_match_complete_targets_without_discarding_high_modes(
    deformation, mp
):
    law = owner._derive_composition_compatible_law(deformation=deformation)
    coefficients = dict(law.current_coefficients)
    assert coefficients == _factor_coefficients(deformation)
    assert all(n % 9 in (1, 8) for n in coefficients)

    def signed(n):
        return 1 if n % 9 == 1 else -1

    moments = (
        sum((coefficient / n for n, coefficient in coefficients.items()), Q(0)),
        sum((signed(n) * coefficient for n, coefficient in coefficients.items()), Q(0)),
        sum((n * coefficient for n, coefficient in coefficients.items()), Q(0)),
        sum(
            (signed(n) * n**2 * coefficient for n, coefficient in coefficients.items()),
            Q(0),
        ),
    )
    assert law.target_moments == moments == (Q(1),) * 4
    alpha, eps = 2 * mp.pi / 9, _mp(mp, deformation)
    for angle in (mp.mpf(0), alpha, 2 * alpha):
        potential = sum(
            _mp(mp, coefficient / n) * (1 - mp.cos(n * angle))
            for n, coefficient in coefficients.items()
        )
        assert abs(potential - (1 - mp.cos(angle))) < mp.mpf("1e-110")
        for order in range(3):
            from_coefficients = sum(
                _mp(mp, coefficient) * n**order * mp.sin(n * angle + order * mp.pi / 2)
                for n, coefficient in coefficients.items()
            )
            expected = mp.diff(mp.sin, angle, order)
            assert abs(from_coefficients - expected) < mp.mpf("1e-110")
        actual_cubic = mp.diff(lambda t: _potential(mp, t, eps), angle, 4)
        assert abs(actual_cubic + mp.cos(angle) - 39366 * eps * mp.cos(angle)) < mp.mpf(
            "1e-110"
        )
    if deformation:
        assert coefficients[8] and coefficients[19]


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(1, 1000), Q(1, 324)))
def test_edge_jet_agrees_with_independently_differentiated_factor_potential(
    deformation, mp
):
    phasor = (Q(5, 13), Q(-12, 13))
    observation = owner._observe_composition_compatible_edge(
        deformation=deformation, relative_phase_phasor=phasor
    )
    angle, eps = mp.atan2(-12, 5), _mp(mp, deformation)

    def potential(t):
        return _potential(mp, t, eps)

    assert abs(_mp(mp, observation.potential) - potential(angle)) < mp.mpf("1e-110")
    for order, actual in enumerate(observation.current_derivatives):
        assert abs(_mp(mp, actual) - mp.diff(potential, angle, order + 1)) < mp.mpf(
            "1e-105"
        )
    law = owner._derive_composition_compatible_law(deformation=deformation)
    for bound, actual in zip(
        law.current_derivative_bounds, observation.current_derivatives
    ):
        assert abs(actual) <= bound
    alpha, multiplier = 2 * mp.pi / 9, (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3
    averaged = sum(_current(mp, angle + s * alpha, eps) for s in (-1, 0, 1)) / 3
    assert abs(averaged - multiplier * _current(mp, angle, eps)) < mp.mpf("1e-110")


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 324)))
def test_exact_quarter_turn_separates_laws_despite_identical_target_jets(deformation):
    observation = owner._observe_composition_compatible_edge(
        deformation=deformation, relative_phase_phasor=(0, 1)
    )
    assert observation.potential == 1
    assert observation.current_derivatives == (
        1 - deformation,
        -36 * deformation,
        -1 - 485 * deformation,
        5904 * deformation,
    )


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(1, 324)))
def test_admitted_boundary_keeps_positive_storage_and_tiny_represented_angles(
    deformation,
):
    small = Q(1, 10**50)
    phasors = (
        (Q(1), Q(0)),
        (Q(-1), Q(0)),
        (Q(3, 5), Q(4, 5)),
        ((1 - small**2) / (1 + small**2), 2 * small / (1 + small**2)),
    )
    law = owner._derive_composition_compatible_law(deformation=deformation)
    assert law.potential_relative_lower_bound == Q(1, 2)
    for cosine, sine in phasors:
        observation = owner._observe_composition_compatible_edge(
            deformation=deformation, relative_phase_phasor=(cosine, sine)
        )
        assert observation.potential >= law.potential_relative_lower_bound * (
            1 - cosine
        )
        assert (observation.potential > 0) == (cosine != 1)


def test_target_moments_do_not_imply_nonnegative_storage():
    coefficients = _factor_coefficients(Q(2))
    moments = (
        sum(b / n for n, b in coefficients.items()),
        sum((1 if n % 9 == 1 else -1) * b for n, b in coefficients.items()),
        sum(n * b for n, b in coefficients.items()),
        sum((1 if n % 9 == 1 else -1) * n**2 * b for n, b in coefficients.items()),
    )
    assert moments == (1, 1, 1, 1)
    # At pi the defining factors give cos(pi)=cos(9*pi)=-1.
    from_factors = 1 - (-1) + 2 * (-1) * (1 - (-1)) ** 2
    from_harmonics = sum(b * (1 - (-1) ** n) / n for n, b in coefficients.items())
    assert from_harmonics == from_factors == -6
    with pytest.raises(ValueError):
        owner._derive_composition_compatible_law(deformation=2)


@pytest.mark.parametrize("winding", (1, 2))
def test_whole_target_matching_does_not_require_matching_each_edge_current(winding, mp):
    perturbation = {1: Q(57), 8: Q(-708), 10: Q(185), 17: Q(221)}
    signs = {n: 1 if n % 9 == 1 else -1 for n in perturbation}
    assert sum(b / n for n, b in perturbation.items()) == 0
    assert sum(n * b for n, b in perturbation.items()) == 0
    assert sum(signs[n] * n**2 * b for n, b in perturbation.items()) == 0
    assert sum(signs[n] * b for n, b in perturbation.items()) == 729
    relative_bound = sum(n * abs(b) for n, b in perturbation.items())
    epsilon = Q(1, 22656)
    assert epsilon * relative_bound == Q(1, 2)
    coefficients = {n: epsilon * b for n, b in perturbation.items()}
    coefficients[1] += 1

    def current(angle, derivative=0):
        return sum(
            _mp(mp, b) * n**derivative * mp.sin(n * angle + derivative * mp.pi / 2)
            for n, b in coefficients.items()
        )

    alpha = 2 * mp.pi * winding / 9
    phases = tuple(alpha * (node - 4) for node in range(9))
    divergence = [mp.mpf(0)] * 9
    for node in range(9):
        other = (node + 1) % 9
        gap = phases[other] - phases[node]
        edge_current = current(gap)
        divergence[node] += edge_current
        divergence[other] -= edge_current
        assert abs(current(gap, 1) - mp.cos(gap)) < mp.mpf("1e-110")
        assert abs(current(gap, 2) + mp.sin(gap)) < mp.mpf("1e-110")
        potential = sum(
            _mp(mp, b / n) * (1 - mp.cos(n * gap)) for n, b in coefficients.items()
        )
        assert abs(potential - (1 - mp.cos(gap))) < mp.mpf("1e-110")
    assert max(map(abs, divergence)) < mp.mpf("1e-110")
    assert abs(current(alpha) - mp.sin(alpha)) > mp.mpf("0.01")


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 324)))
def test_full_fine_edge_rows_storage_loss_and_reciprocal_chart(
    interface, deformation, mp
):
    forms, phasor = (Q(3, 7), Q(-2, 5)), (Q(3, 5), Q(4, 5))
    report = owner._observe_composition_compatible_pair(
        interface=interface,
        deformation=deformation,
        form=forms,
        relative_phase_phasor=phasor,
    )
    edges, cross, degrees = _geometry(interface)
    geometry = report.geometry
    assert geometry.edges == edges
    assert geometry.cross_edges == cross
    assert degrees == (geometry.degree,) * 18
    assert report.law.deformation == deformation
    values = tuple(forms[node // 9] for node in range(18))
    alpha, origin, delta = 2 * mp.pi / 9, mp.pi / 17, mp.atan2(4, 3)
    phases = tuple(
        origin + (node // 9) * delta + alpha * (node % 9 - 4) for node in range(18)
    )
    gamma, eps = 1 / (1023 * mp.pi), _mp(mp, deformation)
    gradient, current = [Q(0)] * 18, [mp.mpf(0)] * 18
    energy = mp.mpf(0)
    for first, second in edges:
        difference = values[first] - values[second]
        gradient[first] += difference
        gradient[second] -= difference
        gap = phases[second] - phases[first]
        flow = _current(mp, gap, eps)
        current[first] += flow
        current[second] -= flow
        energy += _mp(mp, difference**2 / 2) + _potential(mp, gap, eps)
    rate_from_rows = mp.mpf(0)
    for node, (grad, flow, degree) in enumerate(zip(gradient, current, degrees)):
        form_rate = (-_mp(mp, grad) + gamma * flow) / degree
        phase_rate = gamma * _mp(mp, grad) / degree
        _contains(mp, report.form_rate_bounds[node // 9], form_rate)
        _contains(mp, report.phase_rate_bounds[node // 9], phase_rate)
        rate_from_rows += _mp(mp, grad) * form_rate - flow * phase_rate
    expected_loss = sum(
        (grad**2 / degree for grad, degree in zip(gradient, degrees)), Q(0)
    )
    assert report.loss == expected_loss
    assert report.storage_rate == -expected_loss
    _contains(mp, report.storage_rate_from_rows_bounds, rate_from_rows)
    _contains(mp, report.full_storage_bounds, energy)
    q = 1 if interface == "matching" else 3
    k = mp.mpf(q) / (2 + q)
    multiplier = mp.mpf(1) if interface == "matching" else (1 + 2 * mp.cos(alpha)) / 3
    scale, exchange = mp.sqrt(multiplier), gamma * mp.sqrt(multiplier)
    _contains(mp, geometry.effective_gamma_bounds, exchange)
    z = tuple(_mp(mp, value) / scale for value in forms)
    inherited_flow = _current(mp, delta, eps)
    form_rates = (
        z[1] - z[0] + exchange * inherited_flow,
        z[0] - z[1] - exchange * inherited_flow,
    )
    phase_rates = (exchange * (z[0] - z[1]), exchange * (z[1] - z[0]))
    for component in range(2):
        _contains(mp, report.chart_form_bounds[component], z[component])
        _contains(mp, report.chart_form_rate_bounds[component], form_rates[component])
        _contains(mp, report.chart_phase_rate_bounds[component], phase_rates[component])
        assert mp.almosteq(scale * z[component], _mp(mp, forms[component]))
    target = 18 * (1 - mp.cos(alpha)) + 9 * q * (1 - multiplier)
    normalized_energy = (z[1] - z[0]) ** 2 / 2 + _potential(mp, delta, eps)
    _contains(
        mp, report.full_storage_bounds, target + 9 * q * multiplier * normalized_energy
    )
    assert mp.almosteq(
        _mp(mp, report.loss) / k, 18 * q * multiplier * (z[1] - z[0]) ** 2
    )


@pytest.mark.parametrize(
    "bad", (True, "0", float("nan"), float("inf"), Q(1, 324) + Q(1, 10**90))
)
def test_deformation_admission_rejects_invalid_or_exactly_out_of_range_values(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._derive_composition_compatible_law(deformation=bad)


@pytest.mark.parametrize(
    "change",
    (
        {"interface": True},
        {"interface": "all_to_all"},
        {"form": (0,)},
        {"form": (0, True)},
        {"form": (0, float("inf"))},
        {"relative_phase_phasor": (1, 1)},
        {"relative_phase_phasor": (True, 0)},
    ),
)
def test_collective_coordinates_reject_invalid_primitives(change):
    inputs = dict(
        interface="matching", deformation=0, form=(0, 0), relative_phase_phasor=(1, 0)
    )
    with pytest.raises((TypeError, ValueError)):
        owner._observe_composition_compatible_pair(**(inputs | change))


def test_reports_or_tuned_trajectory_inputs_do_not_substitute_for_law_primitives():
    for extra in ({"source": object()}, {"horizon": 1}, {"target_order": 3}):
        with pytest.raises(TypeError):
            owner._derive_composition_compatible_law(deformation=0, **extra)
