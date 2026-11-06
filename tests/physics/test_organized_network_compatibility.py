"""Independent full-state controls for the registered three-C9 triangle."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_network_compatibility as owner

_CONTACTS = ((0, 1), (1, 2), (2, 0))
_EQUAL = (Q(1, 2),) * 3
_UNEQUAL = (Q(1, 4), Q(1, 2), Q(3, 4))
_FORMS = (Q(3, 7), Q(-2, 5), Q(4, 9))
_PHASORS = ((Q(1), Q(0)), (Q(3, 5), Q(4, 5)), (Q(5, 13), Q(-12, 13)))


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
    assert _mp(mp, interval.width) < mp.mpf("1e-28")


def _support(mixing):
    internal = tuple(
        (9 * component + local, 9 * component + (local + 1) % 9, Q(1))
        for component in range(3)
        for local in range(9)
    )
    cross = tuple(
        (9 * tail + local, 9 * head + (local + offset) % 9, weight)
        for (tail, head), s in zip(_CONTACTS, mixing)
        for local in range(9)
        for offset, weight in ((0, 1 - s), (1, s))
        if weight
    )
    return internal + cross, cross


def _product(first, second):
    a, b = first
    c, d = second
    return a * c - b * d, a * d + b * c


def _relative(first, second):
    return _product((first[0], -first[1]), second)


@pytest.mark.parametrize("mixing", (_EQUAL, _UNEQUAL, (Q(0),) * 3, (Q(1),) * 3))
def test_full_twenty_seven_node_rows_storage_and_loss(mixing, mp):
    report = owner._observe_organized_triangle(
        mixing=mixing, form=_FORMS, phase_phasors=_PHASORS
    )
    geometry = report.geometry
    edges, cross = _support(mixing)
    assert geometry.mixing == mixing
    assert geometry.weighted_edges == edges
    assert geometry.weighted_degrees == (Q(4),) * 27
    assert geometry.support_degrees == tuple(
        sum(node in (first, second) for first, second, _ in edges) for node in range(27)
    )
    assert geometry.clock_factor == Q(1, 4)
    assert report.form == _FORMS
    assert report.phase_phasors == _PHASORS
    relative = tuple(_relative(_PHASORS[a], _PHASORS[b]) for a, b in _CONTACTS)
    assert report.relative_phase_phasors == relative
    assert _product(_product(relative[0], relative[1]), relative[2]) == (1, 0)

    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    origins = tuple(mp.atan2(_mp(mp, s), _mp(mp, c)) for c, s in _PHASORS)
    phases = tuple(origins[node // 9] + alpha * (node % 9 - 4) for node in range(27))
    values = tuple(_FORMS[node // 9] for node in range(27))
    gradient, current = [Q(0)] * 27, [mp.mpf(0)] * 27
    potential, form_energy, contact_work = mp.mpf(0), Q(0), mp.mpf(0)
    for first, second, weight in edges:
        difference = values[first] - values[second]
        gradient[first] += weight * difference
        gradient[second] -= weight * difference
        gap = phases[second] - phases[first]
        flow = _mp(mp, weight) * mp.sin(gap)
        current[first] += flow
        current[second] -= flow
        elastic = weight * difference**2 / 2
        angular = _mp(mp, weight) * (1 - mp.cos(gap))
        form_energy += elastic
        potential += angular
        if (first, second, weight) in cross:
            contact_work += _mp(mp, elastic) + angular

    storage_rate = mp.mpf(0)
    for node, (grad, flow) in enumerate(zip(gradient, current)):
        component = node // 9
        dx = (-_mp(mp, grad) + gamma * flow) / 4
        dphase = gamma * _mp(mp, grad) / 4
        assert report.form_gradient[component] == grad
        _contains(mp, report.phase_current_sums_bounds[component], flow)
        _contains(mp, report.form_rate_bounds[component], dx)
        _contains(mp, report.phase_rate_bounds[component], dphase)
        _contains(mp, report.collective_form_rate_bounds[component], 4 * dx)
        _contains(mp, report.collective_phase_rate_bounds[component], 4 * dphase)
        storage_rate += _mp(mp, grad) * dx - flow * dphase

    loss = sum((grad**2 / 4 for grad in gradient), Q(0))
    differences = tuple(_FORMS[b] - _FORMS[a] for a, b in _CONTACTS)
    assert report.form_storage == form_energy == 9 * sum(u**2 for u in differences) / 2
    assert report.loss == loss == Q(27, 4) * sum(u**2 for u in differences)
    assert report.storage_rate == -loss
    # Full-support gradients are squared after summing incident forces.
    assert loss != 6 * sum(u**2 for u in differences)
    _contains(mp, report.storage_rate_from_rows_bounds, storage_rate)
    _contains(mp, report.full_storage_bounds, _mp(mp, form_energy) + potential)
    _contains(mp, report.phase_storage_bounds, potential)
    _contains(mp, report.attachment_work_bounds, contact_work)
    _contains(mp, geometry.internal_storage_bounds, 27 * (1 - mp.cos(alpha)))
    for index, ((tail, head), s) in enumerate(zip(_CONTACTS, mixing)):
        delta = origins[head] - origins[tail]
        s = _mp(mp, s)
        inherited = (1 - s) * mp.sin(delta) + s * mp.sin(delta + alpha)
        energy = (1 - s) * (1 - mp.cos(delta)) + s * (1 - mp.cos(delta + alpha))
        _contains(mp, report.edge_current_bounds[index], inherited)
        _contains(mp, report.edge_potential_bounds[index], energy)


@pytest.mark.parametrize("mixing", (_EQUAL, _UNEQUAL))
def test_cycle_mismatch_and_distinct_amplitude_obstructions(mixing, mp):
    geometry = owner._derive_organized_triangle(mixing=mixing)
    alpha = 2 * mp.pi / 9
    factors = tuple(1 - _mp(mp, s) + _mp(mp, s) * mp.exp(1j * alpha) for s in mixing)
    amplitudes = tuple(abs(value) for value in factors)
    angles = tuple(mp.arg(value) for value in factors)
    product = mp.mpc(1)
    for contact, value, rho, angle in zip(
        geometry.contacts, factors, amplitudes, angles
    ):
        _contains(mp, contact.real_multiplier_bounds, value.real)
        _contains(mp, contact.imaginary_multiplier_bounds, value.imag)
        _contains(mp, contact.amplitude_bounds, rho)
        _contains(mp, contact.phase_origin_shift_bounds, angle)
        product *= value / rho
    # Complementary quarter contacts have the same total phase mismatch as
    # the equal contact control; unequal amplitudes give a separate failure.
    assert abs(sum(angles) - mp.pi / 3) < mp.mpf("1e-110")
    _contains(mp, geometry.cycle_phase_bounds, mp.pi / 3)
    _contains(mp, geometry.cycle_phasor_bounds[0], mp.mpf("0.5"))
    _contains(mp, geometry.cycle_phasor_bounds[1], mp.sqrt(3) / 2)
    assert abs(product - mp.exp(1j * mp.pi / 3)) < mp.mpf("1e-110")
    assert geometry.phase_compatible is False
    assert geometry.common_network_chart is False
    assert geometry.amplitude_compatible is (mixing == _EQUAL)
    if mixing == _EQUAL:
        _contains(mp, geometry.common_form_scale_bounds, mp.sqrt(amplitudes[0]))
        _contains(
            mp, geometry.common_gamma_bounds, mp.sqrt(amplitudes[0]) / (1023 * mp.pi)
        )
    else:
        assert geometry.common_form_scale_bounds is None
        assert geometry.common_gamma_bounds is None
        assert abs(amplitudes[0] - amplitudes[2]) < mp.mpf("1e-110")
        assert abs(
            amplitudes[0] ** 2 - amplitudes[1] ** 2 - (1 - mp.cos(alpha)) / 8
        ) < mp.mpf("1e-110")
    _contains(
        mp, geometry.aligned_attachment_work_bounds, Q(27, 2) * (1 - mp.cos(alpha))
    )
    assert geometry.aligned_attachment_work_bounds.lo > 0


@pytest.mark.parametrize("mixing", (_EQUAL, _UNEQUAL))
def test_tree_chart_retains_cycle_and_original_observations(mixing, mp):
    report = owner._observe_organized_triangle(
        mixing=mixing, form=_FORMS, phase_phasors=_PHASORS
    )
    geometry, alpha = report.geometry, 2 * mp.pi / 9
    angles = tuple(
        mp.arg(1 - _mp(mp, s) + _mp(mp, s) * mp.exp(1j * alpha)) for s in mixing
    )
    shifts = (mp.mpf(0), angles[0], angles[0] + angles[1])
    origins = tuple(mp.atan2(_mp(mp, s), _mp(mp, c)) for c, s in _PHASORS)
    transformed = tuple(phi + shift for phi, shift in zip(origins, shifts))
    for bounds, expected in zip(geometry.tree_phase_origins_bounds, shifts):
        _contains(mp, bounds, expected)
    for bounds, psi, phi, shift in zip(
        report.tree_phase_phasors_bounds, transformed, origins, shifts
    ):
        _contains(mp, bounds[0], mp.cos(psi))
        _contains(mp, bounds[1], mp.sin(psi))
        assert abs((psi - shift) - phi) < mp.mpf("1e-110")
    for edge, ((tail, head), s) in enumerate(zip(_CONTACTS, mixing)):
        rho = abs(1 - _mp(mp, s) + _mp(mp, s) * mp.exp(1j * alpha))
        retained = mp.mpf(0) if edge < 2 else sum(angles)
        current = rho * mp.sin(transformed[head] - transformed[tail] + retained)
        _contains(mp, report.edge_current_bounds[edge], current)

    # A further arbitrary node-origin change cancels from edge currents and
    # telescopes around the cycle only when its boundary coefficients change.
    extra = (mp.pi / 11, -mp.pi / 7, mp.pi / 13)
    new_angles = tuple(
        angle - extra[b] + extra[a] for angle, (a, b) in zip(angles, _CONTACTS)
    )
    assert abs(sum(new_angles) - sum(angles)) < mp.mpf("1e-110")
    for index, ((tail, head), s, beta) in enumerate(zip(_CONTACTS, mixing, new_angles)):
        rho = abs(1 - _mp(mp, s) + _mp(mp, s) * mp.exp(1j * alpha))
        gap = origins[head] + extra[head] - origins[tail] - extra[tail]
        _contains(mp, report.edge_current_bounds[index], rho * mp.sin(gap + beta))


@pytest.mark.parametrize("mixing", (_EQUAL, _UNEQUAL))
def test_inputs_transform_both_rows_and_keep_full_storage_work(mixing, mp):
    report = owner._observe_organized_triangle(
        mixing=mixing, form=_FORMS, phase_phasors=_PHASORS
    )
    form_input = (Q(2, 7), Q(-1, 5), Q(4, 11))
    phase_input = (Q(-3, 8), Q(1, 9), Q(5, 12))
    alpha = 2 * mp.pi / 9
    origins = tuple(mp.atan2(_mp(mp, s), _mp(mp, c)) for c, s in _PHASORS)
    edges, _ = _support(mixing)
    work = mp.mpf(0)
    for first, second, weight in edges:
        left, right = first // 9, second // 9
        gap = origins[right] - origins[left] + alpha * (second % 9 - first % 9)
        work += _mp(mp, weight) * (
            _mp(
                mp,
                (_FORMS[left] - _FORMS[right]) * (form_input[left] - form_input[right]),
            )
            + mp.sin(gap) * _mp(mp, phase_input[right] - phase_input[left])
        )
    raw_dx = tuple(rate + f for rate, f in zip(report.form_rate_bounds, form_input))
    raw_dphase = tuple(
        rate + g for rate, g in zip(report.phase_rate_bounds, phase_input)
    )
    raw_rate = 9 * sum(
        gradient * dx - current * dphase
        for gradient, current, dx, dphase in zip(
            report.form_gradient, report.phase_current_sums_bounds, raw_dx, raw_dphase
        )
    )
    _contains(mp, raw_rate, -_mp(mp, report.loss) + work)
    clock_dx = tuple(
        rate + 4 * f for rate, f in zip(report.collective_form_rate_bounds, form_input)
    )
    clock_dphase = tuple(
        rate + 4 * g
        for rate, g in zip(report.collective_phase_rate_bounds, phase_input)
    )
    clock_rate = 9 * sum(
        gradient * dx - current * dphase
        for gradient, current, dx, dphase in zip(
            report.form_gradient,
            report.phase_current_sums_bounds,
            clock_dx,
            clock_dphase,
        )
    )
    _contains(mp, clock_rate, 4 * (-_mp(mp, report.loss) + work))


def test_equal_alignment_has_circulating_current_but_no_nodal_rate(mp):
    report = owner._observe_organized_triangle(
        mixing=_EQUAL, form=(0, 0, 0), phase_phasors=((1, 0),) * 3
    )
    for edge in report.edge_current_bounds:
        _contains(mp, edge, mp.sin(2 * mp.pi / 9) / 2)
        assert edge.lo > 0
    for rows in (
        report.phase_current_sums_bounds,
        report.form_rate_bounds,
        report.phase_rate_bounds,
    ):
        assert all(value.contains(0) for value in rows)
    assert report.loss == 0
    assert report.attachment_work_bounds.lo > 0


def test_unequal_alignment_has_nonzero_rates_despite_zero_instantaneous_loss(mp):
    report = owner._observe_organized_triangle(
        mixing=_UNEQUAL, form=(0, 0, 0), phase_phasors=((1, 0),) * 3
    )
    gamma, sine = 1 / (1023 * mp.pi), mp.sin(2 * mp.pi / 9)
    for rate, factor in zip(report.form_rate_bounds, (Q(-1, 2), Q(1, 4), Q(1, 4))):
        _contains(mp, rate, gamma * sine * _mp(mp, factor) / 4)
    assert report.form_rate_bounds[0].hi < 0
    assert report.form_rate_bounds[1].lo > 0
    assert report.form_rate_bounds[2].lo > 0
    assert report.loss == 0


def test_nonuniform_node_form_units_do_not_preserve_uniform_form_kernel():
    # Z=(1,1,1) has zero phase rate in every difference-based reciprocal law.
    # X=diag(1,2,3) Z does not; a common clock cannot remove that mismatch.
    report = owner._observe_organized_triangle(
        mixing=_EQUAL, form=(1, 2, 3), phase_phasors=((1, 0),) * 3
    )
    assert report.form_gradient == (-3, 0, 3)
    assert report.phase_rate_bounds[0].hi < 0
    assert report.phase_rate_bounds[2].lo > 0


@pytest.mark.parametrize("mixing", (_EQUAL, (Q(0),) * 3))
def test_common_form_scale_changes_reciprocity_sources_and_sensor_together(mixing, mp):
    report = owner._observe_organized_triangle(
        mixing=mixing, form=_FORMS, phase_phasors=_PHASORS
    )
    geometry = report.geometry
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    rho = abs(1 - _mp(mp, mixing[0]) + _mp(mp, mixing[0]) * mp.exp(1j * alpha))
    scale, exchange = mp.sqrt(rho), gamma * mp.sqrt(rho)
    z = tuple(_mp(mp, form) / scale for form in _FORMS)
    lz = tuple(3 * value - sum(z) for value in z)
    form_input, phase_input = (Q(2, 7), Q(-1, 5), Q(4, 11)), (
        Q(-3, 8),
        Q(1, 9),
        Q(5, 12),
    )
    origins = tuple(mp.atan2(_mp(mp, s), _mp(mp, c)) for c, s in _PHASORS)
    beta = mp.arg(1 - _mp(mp, mixing[0]) + _mp(mp, mixing[0]) * mp.exp(1j * alpha))
    reduced_current = tuple(
        mp.sin(origins[b] - origins[a] + beta) for a, b in _CONTACTS
    )
    net = tuple(reduced_current[r] - reduced_current[(r - 1) % 3] for r in range(3))
    chart_dx, chart_dphase = [], []
    for r in range(3):
        dx = report.collective_form_rate_bounds[r] / geometry.common_form_scale_bounds
        dphase = report.collective_phase_rate_bounds[r]
        _contains(mp, dx, -lz[r] + exchange * net[r])
        _contains(mp, dphase, exchange * lz[r])
        assert abs(scale * z[r] - _mp(mp, _FORMS[r])) < mp.mpf("1e-110")
        chart_dx.append(dx + 4 * form_input[r] / geometry.common_form_scale_bounds)
        chart_dphase.append(dphase + 4 * phase_input[r])
    chart_rate = 9 * sum(
        geometry.common_form_scale_bounds * grad * dx - current * dphase
        for grad, current, dx, dphase in zip(
            report.form_gradient,
            report.phase_current_sums_bounds,
            chart_dx,
            chart_dphase,
        )
    )
    expected_work = 9 * sum(
        _mp(mp, grad * f) - rho * current * _mp(mp, g)
        for grad, current, f, g in zip(
            report.form_gradient, net, form_input, phase_input
        )
    )
    _contains(mp, chart_rate, 4 * (-_mp(mp, report.loss) + expected_work))
    energy = 27 * (1 - mp.cos(alpha)) + 9 * (
        3 * (1 - rho)
        + rho
        * (
            sum((z[b] - z[a]) ** 2 / 2 for a, b in _CONTACTS)
            + sum(1 - mp.cos(origins[b] - origins[a] + beta) for a, b in _CONTACTS)
        )
    )
    _contains(mp, report.full_storage_bounds, energy)
    # Equal amplitudes alone do not remove the residual cycle at nonzero s.
    assert geometry.common_network_chart is (mixing[0] == 0)


def test_tiny_joint_phase_keeps_exact_state_and_unresolved_enclosures():
    t = Q(1, 10**80)
    small = ((1 - t**2) / (1 + t**2), 2 * t / (1 + t**2))
    report = owner._observe_organized_triangle(
        mixing=(0, 0, 0), form=(0, 0, 0), phase_phasors=((1, 0), small, (1, 0))
    )
    assert report.phase_phasors[1] == small
    assert report.relative_phase_phasors == (small, (small[0], -small[1]), (1, 0))
    assert report.edge_current_bounds[0].contains(small[1])
    assert report.edge_current_bounds[0].hi > 0
    assert report.edge_current_bounds[1].contains(-small[1])
    assert report.edge_current_bounds[1].lo < 0
    # The declared 128-bit outward grid cannot resolve this tiny response;
    # retaining zero in its interval is not an exact equilibrium verdict.
    assert report.edge_current_bounds[0].contains(0)
    assert all(rate.contains(0) for rate in report.form_rate_bounds)
    expected_work = 18 * (1 - small[0])
    assert report.attachment_work_bounds.contains(expected_work)
    assert report.attachment_work_bounds.hi > 0
    assert expected_work > 0


@pytest.mark.parametrize(
    "mixing,amplitude,phase",
    (
        ((0, 0, 0), True, True),
        ((1, 1, 1), True, False),
        ((0, 1, 0), True, False),
        ((Q(1, 4), Q(3, 4), Q(1, 4)), True, False),
        ((Q(1, 10**80), 0, 0), False, False),
        ((Q(1, 2), Q(1, 2) + Q(1, 10**80), Q(1, 2)), False, False),
    ),
)
def test_exact_compatibility_has_no_empirical_tolerance(mixing, amplitude, phase):
    geometry = owner._derive_organized_triangle(mixing=mixing)
    assert geometry.amplitude_compatible is amplitude
    assert geometry.phase_compatible is phase
    assert geometry.common_network_chart is (amplitude and phase)
    edges, _ = _support(mixing)
    assert geometry.weighted_edges == edges
    assert geometry.weighted_degrees == (4,) * 27


@pytest.mark.parametrize(
    "bad",
    (
        (0, 0),
        (0, 0, 0, 0),
        (True, 0, 0),
        ("0.5", 0, 0),
        (float("nan"), 0, 0),
        (float("inf"), 0, 0),
        (-Q(1, 10**80), 0, 0),
        (1 + Q(1, 10**80), 0, 0),
    ),
)
def test_invalid_triangle_geometry_is_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._derive_organized_triangle(mixing=bad)


@pytest.mark.parametrize(
    "change",
    (
        {"mixing": (True, 0, 0)},
        {"form": (0, 0)},
        {"form": (0, 0, True)},
        {"form": (0, float("inf"), 0)},
        {"phase_phasors": ((1, 0),) * 2},
        {"phase_phasors": ((1, 0), (1, 0), (True, 0))},
        {"phase_phasors": ((1, 0), (1, 0), (1, 1))},
    ),
)
def test_observation_admits_all_primitives_before_geometry(change, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached geometry construction")

    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    inputs = dict(mixing=_EQUAL, form=_FORMS, phase_phasors=_PHASORS)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_triangle(**(inputs | change))


def test_reports_and_trajectory_controls_cannot_replace_primitives():
    for extra in ({"geometry": object()}, {"source": object()}, {"horizon": 1}):
        with pytest.raises(TypeError):
            owner._observe_organized_triangle(
                mixing=_EQUAL, form=_FORMS, phase_phasors=_PHASORS, **extra
            )
