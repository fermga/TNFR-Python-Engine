"""Independent controls for the conditional five-child electrical protocol.

These are static arithmetic and field checks, not trajectories, source
assessments or laboratory records. The amplitude controls eliminate unknown
phase from raw current quadratures without using a predicted amplitude as a
normalization or a phase measurement.
"""

import hashlib
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import pi_interval


@pytest.fixture(scope="module", autouse=True)
def no_reserved_response_or_archive_changes():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
            "src/tnfr/physics/_organized_electrical_source.py",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_sine_regeneration():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _laplacian_action(values, edges):
    result = [0] * len(values)
    for left, right in edges:
        difference = values[left] - values[right]
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def _sine_current(mp, phases, edges):
    result = [mp.mpf(0)] * len(phases)
    for left, right in edges:
        transfer = mp.sin(phases[right] - phases[left])
        result[left] += transfer
        result[right] -= transfer
    return tuple(result)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def test_existing_source_coarse_bounds_follow_by_exact_positive_majorants():
    # Enlarge gamma and duration before arithmetic; this does not reevaluate
    # any fixed source corner or consume its cached verdict.
    gamma = Q(1, 1023) / pi_interval()
    g, duration, q = Q(1, 3000), Q(16), Q(1, 10**8)
    assert gamma.hi < g and 15 * (1 + Q(1, 10**5)) < duration
    phase = 6 * q * (1 + g)
    form = 3 * q + 24 * g * phase
    assert form < Q(31, 10**9) and phase < Q(601, 10**10)
    coupling = 2 * g * duration
    hidden_form = (form + coupling * phase) / (1 - coupling**2)
    hidden_phase = (phase + coupling * form) / (1 - coupling**2)
    # Check the two independently obtained integral inequalities at equality.
    assert hidden_form == form + coupling * hidden_phase
    assert hidden_phase == phase + coupling * hidden_form
    assert hidden_form < Q(32, 10**9)
    assert hidden_phase < Q(61, 10**9)
    rho = g * hidden_phase**2 / 9
    collective_coupling = g * duration
    form_error = duration * rho / (1 - collective_coupling**2)
    phase_error = collective_coupling * form_error
    assert form_error == duration * rho + collective_coupling * phase_error
    assert form_error < Q(23, 10**19)
    # A distinct physical error cannot be replaced by the intrinsic error.
    initial_form_error = Q(1, 10**8)
    changed = (initial_form_error + duration * rho) / (1 - collective_coupling**2)
    assert changed > initial_form_error > form_error


@pytest.mark.parametrize("angle", (Q(-6, 5), Q(0), Q(9, 10)))
@pytest.mark.parametrize("signs", tuple(product((-1, 1), repeat=3)))
def test_raw_charge_quadratures_enclose_unfitted_amplitude(mp, angle, signs):
    epsilon, h = _mp(mp, Q(1, 10)), _mp(mp, Q(1, 10**5))
    clock, kappa = _mp(mp, Q(3, 7)), _mp(mp, Q(2, 13))
    # Choose amplitude independently: this static identity must not normalize
    # observations by a separately predicted value, including a wrong one.
    amplitude = _mp(mp, Q(7, 11))
    duration, beta = h / clock, _mp(mp, Q(1, 10**12))
    zeta = _mp(mp, Q(1, 10**6))
    offsets = tuple(zeta * (slot - 4) / 2 for slot in range(9))
    angles = tuple(_mp(mp, angle) + value for value in offsets)
    assert sum(offsets) == 0 and max(abs(value) for value in offsets) == 2 * zeta
    cosine = sum(mp.cos(value) for value in angles) / 9
    sine = sum(mp.sin(value) for value in angles) / 9
    background = mp.sin(_mp(mp, Q(4, 7)))
    exact = tuple(
        amplitude
        * duration
        * (background - sum(mp.sin(value - level * epsilon) for value in angles) / 9)
        for level in (0, 1, -1)
    )
    measured = tuple(
        value + 9 * kappa * beta * sign for value, sign in zip(exact, signs)
    )
    odd = (measured[1] - measured[2]) / (2 * duration * mp.sin(epsilon))
    even = (measured[1] + measured[2] - 2 * measured[0]) / (
        2 * duration * (1 - mp.cos(epsilon))
    )
    odd_error = 9 * clock * kappa * beta / (h * mp.sin(epsilon))
    even_error = 18 * clock * kappa * beta / (h * (1 - mp.cos(epsilon)))
    tolerance = mp.mpf("1e-75")
    assert abs(odd - amplitude * cosine) <= odd_error + tolerance
    assert abs(even - amplitude * sine) <= even_error + tolerance
    radial_error = mp.sqrt(odd_error**2 + even_error**2)
    measured_radius = mp.sqrt(odd**2 + even**2)
    true_radius = mp.sqrt(cosine**2 + sine**2)
    assert mp.cos(2 * zeta) <= true_radius <= 1
    assert amplitude * mp.cos(2 * zeta) - radial_error <= measured_radius
    assert measured_radius <= amplitude + radial_error
    lower = max(0, measured_radius - radial_error)
    upper = (measured_radius + radial_error) / mp.cos(2 * zeta)
    assert lower <= amplitude <= upper
    # These synthetic readings exclude an independently supplied doubled
    # amplitude; no claim about its own-law source history follows.
    assert upper < 2 * amplitude


def test_same_voltage_neutrality_and_winding_do_not_initialize_the_circuit(mp):
    edges = tuple((child, (child + 1) % 5) for child in range(5))
    delta = _mp(mp, Q(1, 10))
    first, second = (mp.mpf(0),) * 5, (delta, -delta, 0, 0, 0)
    form = (Q(0),) * 5
    assert sum(first) == sum(second) == sum(form) == 0
    assert _laplacian_action(form, edges) == (0,) * 5
    for phases in (first, second):
        gaps = tuple(phases[right] - phases[left] for left, right in edges)
        assert max(abs(value) for value in gaps) < mp.pi / 2
        assert sum(gaps) == 0  # All gaps already principal: winding zero.
    first_current = _sine_current(mp, first, edges)
    second_current = _sine_current(mp, second, edges)
    assert first_current == (0,) * 5
    assert max(abs(value) for value in second_current) > delta


@pytest.fixture(scope="module")
def off_carrier_state(mp):
    coarse_edges = tuple((child, (child + 1) % 5) for child in range(5))
    fine_edges = tuple(
        (9 * child + slot, 9 * child + (slot + 1) % 9)
        for child in range(5)
        for slot in range(9)
    ) + tuple(
        (9 * child + slot, 9 * ((child + 1) % 5) + slot)
        for child in range(5)
        for slot in range(9)
    )
    forms = tuple(Q(child - 2, 7) for child in range(5))
    hidden_forms = tuple(
        Q((child + 1) * (slot - 4), 101) for child in range(5) for slot in range(9)
    )
    hidden_phases = tuple(
        _mp(mp, Q((child + 1) * (slot - 4), 10**5))
        for child in range(5)
        for slot in range(9)
    )
    origins = tuple(_mp(mp, Q(child - 2, 11)) for child in range(5))
    fine_forms = tuple(
        forms[child] + hidden_forms[9 * child + slot]
        for child in range(5)
        for slot in range(9)
    )
    phases = tuple(
        2 * mp.pi * (slot - 4) / 9 + origins[child] + hidden_phases[9 * child + slot]
        for child in range(5)
        for slot in range(9)
    )
    return (
        coarse_edges,
        fine_edges,
        forms,
        hidden_forms,
        hidden_phases,
        origins,
        fine_forms,
        phases,
    )


def test_actual_hidden_storage_loss_and_electrical_input_work_are_distinct(
    mp, off_carrier_state
):
    (
        coarse_edges,
        fine_edges,
        forms,
        hidden_forms,
        hidden_phases,
        origins,
        fine_forms,
        phases,
    ) = off_carrier_state
    fine_laplacian = _laplacian_action(fine_forms, fine_edges)
    hidden_laplacian = _laplacian_action(hidden_forms, fine_edges)
    coarse_laplacian = _laplacian_action(forms, coarse_edges)
    forcing = tuple(Q(child - 2, 13) for child in range(5))
    copied_forcing = tuple(value for value in forcing for _ in range(9))
    structural_power = _dot(fine_laplacian, copied_forcing)
    assert structural_power == 9 * _dot(coarse_laplacian, forcing) != 0
    fine_loss = _dot(fine_laplacian, fine_laplacian) / 4
    hidden_loss = _dot(hidden_laplacian, hidden_laplacian) / 4
    coarse_loss = Q(9, 4) * _dot(coarse_laplacian, coarse_laplacian)
    assert fine_loss == coarse_loss + hidden_loss and hidden_loss > 0
    gamma, a0, clock, kappa = 1 / (1023 * mp.pi), mp.mpf(2), mp.mpf(3), mp.mpf(5)
    conversion = a0 * clock * kappa * gamma / 4
    voltage = tuple(
        a0 * clock * gamma * _mp(mp, value) / 4 for value in coarse_laplacian
    )
    current = tuple(9 * clock * kappa * _mp(mp, value) for value in forcing)
    conductance = 9 * kappa / (a0 * gamma)
    tolerance = mp.mpf("1e-75")
    assert (
        abs(_dot(voltage, current) / clock - conversion * _mp(mp, structural_power))
        < tolerance
    )
    joule_rate = conductance * _dot(voltage, voltage)
    assert abs(joule_rate / clock - conversion * _mp(mp, coarse_loss)) < tolerance
    assert joule_rate / clock < conversion * _mp(mp, fine_loss)
    fine_storage = _mp(mp, _dot(fine_forms, fine_laplacian) / 2) + sum(
        1 - mp.cos(phases[right] - phases[left]) for left, right in fine_edges
    )
    coarse_storage = _mp(mp, _dot(forms, coarse_laplacian) / 2) + sum(
        1 - mp.cos(origins[right] - origins[left]) for left, right in coarse_edges
    )
    internal_storage = 45 * (1 - mp.cos(2 * mp.pi / 9))
    remainder = fine_storage - internal_storage - 9 * coarse_storage
    form_norm_squared = _mp(mp, _dot(hidden_forms, hidden_forms))
    phase_norm_squared = _dot(hidden_phases, hidden_phases)
    assert (
        -4 * phase_norm_squared
        <= remainder
        <= 4 * (form_norm_squared + phase_norm_squared)
    )
    assert conversion * fine_storage != 9 * conversion * coarse_storage


def test_general_storage_remainder_can_be_negative_outside_the_acute_source(mp):
    # This controls the general departure identity only. The nonacute
    # collective state is deliberately outside the admitted observation prior.
    origins = (mp.pi, 0, 0, 0, 0)
    residuals = tuple(
        mp.cos(2 * mp.pi * slot / 9) / 100 if child == 0 else 0
        for child in range(5)
        for slot in range(9)
    )
    phases = tuple(
        2 * mp.pi * (slot - 4) / 9 + origins[child] + residuals[9 * child + slot]
        for child in range(5)
        for slot in range(9)
    )
    internal = sum(
        1 - mp.cos(phases[9 * child + (slot + 1) % 9] - phases[9 * child + slot])
        for child in range(5)
        for slot in range(9)
    )
    external = sum(
        1 - mp.cos(phases[9 * ((child + 1) % 5) + slot] - phases[9 * child + slot])
        for child in range(5)
        for slot in range(9)
    )
    carrier = 45 * (1 - mp.cos(2 * mp.pi / 9)) + 9 * sum(
        1 - mp.cos(origins[(child + 1) % 5] - origins[child]) for child in range(5)
    )
    remainder = internal + external - carrier
    assert -4 * _dot(residuals, residuals) <= remainder < 0
