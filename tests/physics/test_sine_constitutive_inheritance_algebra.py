"""Independent full-support checks of organized constitutive inheritance."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_inheritance as owner


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


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
@pytest.mark.parametrize("deformation", (Q(0), Q(1), Q(3, 2)))
def test_full_eighteen_node_rows_storage_and_loss_restrict_exactly(
    interface, deformation, mp
):
    forms, phasor = (Q(3, 7), Q(-2, 5)), (Q(3, 5), Q(4, 5))
    report = owner._observe_inherited_constitutive_pair(
        interface=interface,
        deformation=deformation,
        form=forms,
        relative_phase_phasor=phasor,
    )
    edges, cross, degrees = _geometry(interface)
    assert report.law.edges == edges
    assert report.law.cross_edges == cross
    assert len({frozenset(edge) for edge in edges}) == len(edges)
    assert degrees == (report.law.degree,) * 18
    assert report.law.collective_clock_factor == Q(len(cross), 9 * degrees[0])
    values = tuple(forms[node // 9] for node in range(18))
    alpha, origin, delta = 2 * mp.pi / 9, mp.pi / 17, mp.atan2(4, 3)
    phases = tuple(
        origin + (node // 9) * delta + alpha * (node % 9 - 4) for node in range(18)
    )
    gamma, lam = 1 / (1023 * mp.pi), _mp(mp, deformation)
    gradient = [Q(0)] * 18
    current = [mp.mpf(0)] * 18
    energy, cross_energy = mp.mpf(0), mp.mpf(0)
    for first, second in edges:
        difference = values[first] - values[second]
        gradient[first] += difference
        gradient[second] -= difference
        gap = phases[second] - phases[first]
        potential = 1 - mp.cos(gap) + lam * (1 - mp.cos(9 * gap)) ** 2 / (6 * 9**4)
        flow = mp.sin(gap) + lam * mp.sin(9 * gap) * (1 - mp.cos(9 * gap)) / 2187
        current[first] += flow
        current[second] -= flow
        edge_energy = _mp(mp, difference**2 / 2) + potential
        energy += edge_energy
        if (first, second) in cross:
            cross_energy += edge_energy
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
    relative_cross = cross_energy - 9 * report.law.cross_degree * (
        1 - (1 if interface == "matching" else (1 + 2 * mp.cos(alpha)) / 3)
    )
    _contains(
        mp,
        report.form_storage
        + 9 * report.law.cross_degree * report.relative_potential_bounds,
        relative_cross,
    )


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
@pytest.mark.parametrize("deformation", (Q(0), Q(1), Q(3, 2)))
def test_regular_chart_transforms_both_rows_inputs_storage_and_observation(
    interface, deformation, mp
):
    form = (Q(-3, 8), Q(5, 11))
    report = owner._observe_inherited_constitutive_pair(
        interface=interface,
        deformation=deformation,
        form=form,
        relative_phase_phasor=(Q(3, 5), Q(4, 5)),
    )
    q = 1 if interface == "matching" else 3
    k, alpha, delta = mp.mpf(q) / (2 + q), 2 * mp.pi / 9, mp.atan2(4, 3)
    r = mp.mpf(1) if interface == "matching" else (1 + 2 * mp.cos(alpha)) / 3
    scale, gamma = mp.sqrt(r), 1 / (1023 * mp.pi)
    mapped, exchange = _mp(mp, deformation) / r, gamma * scale
    _contains(mp, report.law.effective_deformation_bounds, mapped)
    _contains(mp, report.law.effective_gamma_bounds, exchange)
    _contains(mp, report.law.form_chart_scale_bounds, scale)
    _contains(mp, report.law.normalized_pair_storage_factor_bounds, 9 * q * r)
    _contains(mp, report.law.form_input_pushforward_bounds, 1 / (k * scale))
    assert report.law.phase_input_pushforward == Q(2 + q, q)
    z = tuple(_mp(mp, value) / scale for value in form)
    flow = mp.sin(delta) + mapped * mp.sin(9 * delta) * (1 - mp.cos(9 * delta)) / 2187
    form_rates = (z[1] - z[0] + exchange * flow, z[0] - z[1] - exchange * flow)
    phase_rates = (exchange * (z[0] - z[1]), exchange * (z[1] - z[0]))
    for index in range(2):
        _contains(mp, report.chart_form_bounds[index], z[index])
        _contains(mp, report.chart_form_rate_bounds[index], form_rates[index])
        _contains(mp, report.chart_phase_rate_bounds[index], phase_rates[index])
        # A raw form sensor still reads X, even when the state uses Z.
        assert mp.almosteq(scale * z[index], _mp(mp, form[index]))
    potential = 1 - mp.cos(delta) + mapped * (1 - mp.cos(9 * delta)) ** 2 / (6 * 9**4)
    target = 18 * (1 - mp.cos(alpha)) + 9 * q * (1 - r)
    normalized_energy = (z[1] - z[0]) ** 2 / 2 + potential
    _contains(mp, report.full_storage_bounds, target + 9 * q * r * normalized_energy)
    expected_loss_in_sigma = 18 * q * r * (z[1] - z[0]) ** 2
    assert mp.almosteq(_mp(mp, report.loss) / k, expected_loss_in_sigma)
    form_input, phase_input = mp.mpf(3) / 17, -mp.mpf(2) / 19
    pushed_form, pushed_phase = form_input / (k * scale), phase_input / k
    assert mp.almosteq(k * scale * pushed_form, form_input)
    assert mp.almosteq(k * pushed_phase, phase_input)


def test_reciprocal_map_distinguishes_current_shape_from_fixed_exchange_family(mp):
    sine = owner._derive_constitutive_inheritance(interface="symmetric", deformation=0)
    altered = owner._derive_constitutive_inheritance(
        interface="symmetric", deformation=1
    )
    matching = owner._derive_constitutive_inheritance(
        interface="matching", deformation=1
    )
    r = (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3
    gamma = 1 / (1023 * mp.pi)
    assert sine.same_current_profile_up_to_scale
    assert not sine.retains_fixed_exchange_parameter
    assert not altered.same_current_profile_up_to_scale
    assert not altered.retains_fixed_exchange_parameter
    assert altered.effective_deformation_within_unit_interval is False
    assert matching.retains_fixed_exchange_parameter
    assert matching.effective_deformation_within_unit_interval is True
    assert _mp(mp, sine.effective_gamma_bounds.hi) < gamma
    # Equal reciprocal coefficients force a^2=r after diffusion fixes the clock.
    a = mp.sqrt(r)
    assert mp.almosteq(gamma * a, gamma * r / a)
    assert gamma * a < gamma
    # Forcing the original phase coefficient instead forces a=1,
    # whose first harmonic retains a strictly different coefficient.
    assert gamma * r < gamma
    delta = mp.pi / 18
    fine_current = mp.sin(delta) + mp.mpf(1) / 2187
    inherited_current = r * mp.sin(delta) + mp.mpf(1) / 2187
    assert abs(inherited_current - r * fine_current) > mp.mpf("1e-5")


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
def test_ninth_and_eighteenth_moments_do_not_retain_registered_current(interface, mp):
    alpha = 2 * mp.pi / 9
    offsets = (0,) if interface == "matching" else (-1, 0, 1)
    phases = tuple(alpha * (local - 4) for local in range(9))
    for harmonic in (1, 9, 18):
        original = (
            sum((mp.exp(1j * harmonic * phase) for phase in phases), mp.mpc(0)) / 9
        )
        shifted = (
            sum(
                (mp.exp(1j * harmonic * (phase + alpha)) for phase in phases), mp.mpc(0)
            )
            / 9
        )
        assert abs(original - shifted) < mp.mpf("1e-110")
        assert abs(original - (0 if harmonic == 1 else 1)) < mp.mpf("1e-110")
    # Both endpoint laws have zero higher current at each multiple of alpha.
    for deformation in (0, 1):

        def averaged_current(delta):
            return sum(
                (
                    mp.sin(delta + offset * alpha)
                    + deformation
                    * mp.sin(9 * (delta + offset * alpha))
                    * (1 - mp.cos(9 * (delta + offset * alpha)))
                    / 2187
                    for offset in offsets
                ),
                mp.mpf(0),
            ) / len(offsets)

        assert abs(averaged_current(0)) < mp.mpf("1e-110")
        assert averaged_current(alpha) > mp.mpf("0.5")


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
@pytest.mark.parametrize("deformation", (Q(0), Q(1)))
def test_generic_off_family_means_retain_linear_rows_and_bounded_current_defect(
    interface, deformation, mp
):
    """A supplied perturbed state tests the lemma, not acquired reachability."""
    radius = Q(1, 10**4)
    means = (Q(1, 7), Q(-2, 11))
    report = owner._observe_inherited_constitutive_pair(
        interface=interface,
        deformation=deformation,
        form=means,
        relative_phase_phasor=(Q(3, 5), Q(4, 5)),
    )
    edges, _, degrees = _geometry(interface)
    values = tuple(
        means[component] + radius * Q((2 * local + component) % 9 - 4, 4)
        for component in range(2)
        for local in range(9)
    )
    fluctuations = tuple(
        radius * Q((5 * local + 3 * component) % 9 - 4, 2)
        for component in range(2)
        for local in range(9)
    )
    for component in range(2):
        block = slice(9 * component, 9 * component + 9)
        assert sum(values[block], Q(0)) / 9 == means[component]
        assert sum(fluctuations[block], Q(0)) == 0
    assert max(map(abs, fluctuations)) <= 2 * radius
    alpha, delta, gamma = 2 * mp.pi / 9, mp.atan2(4, 3), 1 / (1023 * mp.pi)
    phases = tuple(
        mp.mpf(1) / 13
        + (node // 9) * delta
        + (node % 9 - 4) * alpha
        + _mp(mp, fluctuations[node])
        for node in range(18)
    )
    gradient, currents = [Q(0)] * 18, [mp.mpf(0)] * 18
    lam = _mp(mp, deformation)
    for first, second in edges:
        difference = values[first] - values[second]
        gradient[first] += difference
        gradient[second] -= difference
        gap = phases[second] - phases[first]
        flow = mp.sin(gap) + lam * mp.sin(9 * gap) * (1 - mp.cos(9 * gap)) / 2187
        currents[first] += flow
        currents[second] -= flow
    q, degree = report.law.cross_degree, degrees[0]
    k = mp.mpf(q) / degree
    r = mp.mpf(1) if interface == "matching" else (1 + 2 * mp.cos(alpha)) / 3
    current = (
        r * mp.sin(delta) + lam * mp.sin(9 * delta) * (1 - mp.cos(9 * delta)) / 2187
    )
    nominal = k * (_mp(mp, means[1] - means[0]) + gamma * current)
    residuals = []
    for component in range(2):
        block = slice(9 * component, 9 * component + 9)
        grad_mean = sum(gradient[block], Q(0)) / 9
        assert grad_mean == q * (means[component] - means[1 - component])
        phase_rate = gamma * _mp(mp, grad_mean) / degree
        _contains(mp, report.phase_rate_bounds[component], phase_rate)
        form_rate = (-_mp(mp, grad_mean) + gamma * sum(currents[block]) / 9) / degree
        residuals.append(form_rate - (nominal if component == 0 else -nominal))
    bound = 4 * gamma * k * (1 + 2 * lam / 243) * _mp(mp, radius)
    assert abs(sum(residuals)) < mp.mpf("1e-110")
    assert mp.mpf("1e-30") < abs(residuals[0]) <= bound
