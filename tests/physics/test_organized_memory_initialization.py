"""Static preparation and information controls; no trajectories or observations.

The source trap is an explicit hypothesis. Arithmetic certificates below do
not reacquire a source, replay an assessor or manufacture a measured record.
"""

from fractions import Fraction as Q
import hashlib
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sqrt
from tnfr.physics._organized_electrical_source import _transport_hidden_deviations
from tnfr.physics._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)
from tnfr.physics._sine_preparation import _prepared_heat_error_radii


@pytest.fixture(scope="module", autouse=True)
def preserve_frozen_evidence_and_producers():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_joint_resources/attempt.json",
            "docs/assets/organized_joint_resources/result.json",
            "benchmarks/organized_joint_resources.py",
            "src/tnfr/physics/_organized_joint_resources.py",
            "src/tnfr/physics/_sine_recovery_resources.py",
            "src/tnfr/physics/_organized_outer_winding.py",
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_organized_response_evaluation():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


def _means(values):
    return tuple(sum(values[9 * c : 9 * c + 9]) / 9 for c in range(5))


def _variances(values):
    return tuple(
        sum((values[9 * ((c + 1) % 5) + j] - values[9 * c + j]) ** 2 for j in range(9))
        / 9
        for c in range(5)
    )


@pytest.fixture(scope="module")
def recovery():
    pi, radius = pi_interval(), Q(1, 12)
    eta = 1 / (1023 * pi) ** 2
    assert Q(9, 10**8) < eta.lo <= eta.hi < Q(1, 10**7)
    assert cos(2 * pi / 9 + sqrt(I(2)) * radius).lo > Q(2, 3)
    assert (1 - cos(2 * pi / 9)).lo > Q(1, 5)
    mu, maximum, lower, upper, rate = _sine_lyapunov_coefficients(
        eta_lower=Q(9, 10**8),
        eta_upper=Q(1, 10**7),
        cosine_lower=Q(2, 3),
        gap_lower=Q(1, 5),
        rate_upper=Q(2),
    )
    assert mu > 0 and maximum > mu
    assert lower > Q(1, 400) and upper < Q(3, 50)
    assert rate > Q(2, 10**9)
    # The original half-law decay uses half the unchanged quarter-law dwell.
    assert rate * (8 * 10**10) / 2 > 80
    initial = _sine_lyapunov_initial_upper(
        eta_upper=Q(1, 10**7),
        rate_upper=Q(2),
        gap_lower=Q(1, 5),
        position_upper=upper,
        form_norm_upper=radius,
        phase_norm_upper=radius,
    )
    assert initial < Q(3, 1000)
    return initial, lower


def test_same_recovery_retains_distinct_form_and_phase_radii(recovery):
    initial, lower = recovery
    energy, form_square, phase_square = _sine_lyapunov_return_squared(
        initial_upper=initial,
        decay_upper=Q(1, 2**80),
        eta_lower=Q(9, 10**8),
        gap_lower=Q(1, 5),
        rate_upper=Q(2),
        position_lower=lower,
    )
    # Recover the energy independently from each metric; the denominators
    # differ and replacing both by the old q loses useful phase information.
    assert energy == initial / 2**80
    assert form_square * Q(9, 10**8) * Q(1, 5) / 4 == energy
    assert phase_square * lower / 2 == energy
    assert 0 < form_square < Q(1, 10**18)
    assert 0 < phase_square < Q(4, 10**24)
    assert 5 * Q(1, 10**18) < Q(3, 10**9) ** 2
    assert 5 * Q(4, 10**24) < Q(6, 10**12) ** 2


@pytest.mark.parametrize("active_common_form", (False, True))
def test_zero_sum_child_errors_give_exact_attachment_means(active_common_form):
    raw = tuple(Q((7 * i) % 13 - 6, 10**14) for i in range(45))
    means = _means(raw)
    residual = tuple(value - means[i // 9] for i, value in enumerate(raw))
    origins = tuple(Q(4092 * (c - 2)) if active_common_form else Q(0) for c in range(5))
    form = tuple(origins[i // 9] + value for i, value in enumerate(residual))
    assert _means(form) == origins
    assert _means(residual) == (Q(0),) * 5
    # Arbitrary isolated edge currents conserve each lifted child mean. This
    # is an incidence cancellation, without a sampled sine trajectory.
    row = [Q(0)] * 45
    for c in range(5):
        for j in range(9):
            a, b = 9 * c + j, 9 * c + (j + 1) % 9
            row[a] += raw[a]
            row[b] -= raw[a]
    assert _means(row) == (Q(0),) * 5


def test_full_history_heat_bound_supplies_zero_variance_initialization():
    g, horizon = Q(1, 3000), Q(216)
    pi = pi_interval()
    assert (1 / (1023 * pi)).hi < g
    assert Q(200) + Q(1500015, 100000) < horizon
    # C9-by-C5 spectral gap exceeds 1/3; the heat integral factor is <24.
    assert min((2 - 2 * cos(2 * pi / n)).lo for n in (5, 9)) > Q(1, 3)
    growth_exponent = 2 * g**2 * horizon
    assert 0 < growth_exponent < Q(1, 2)
    assert 1 / (1 - growth_exponent) < 2
    form, phase = _prepared_heat_error_radii(
        initial_form_error=Q(3, 10**9),
        initial_phase_error=Q(6, 10**12),
        gamma_upper=g,
        phase_growth_upper=Q(2),
        form_convolution_factor=Q(24),
    )
    assert phase == 2 * (Q(6, 10**12) + g * Q(3, 10**9)) == Q(14, 10**12)
    assert form == Q(3, 10**9) + 24 * g * phase
    assert Q(4, 9) * phase**2 < Q(1, 10**17)
    assert 2 * phase < pi.lo / 2


def test_variance_sum_is_bounded_by_phase_norm_without_reading_phases():
    raw = tuple(Q((11 * i) % 17 - 8, 10**15) for i in range(45))
    means = _means(raw)
    z = tuple(value - means[i // 9] for i, value in enumerate(raw))
    norm_squared = sum(value**2 for value in z)
    assert _means(z) == (Q(0),) * 5
    assert norm_squared <= sum(value**2 for value in raw)
    assert norm_squared < Q(14, 10**12) ** 2
    assert 0 < sum(_variances(z)) <= Q(4, 9) * norm_squared
    assert sum(_variances(z)) < Q(1, 10**17)


def test_known_attachment_means_propagate_the_nonlinear_carrier_error():
    g, horizon, phase = Q(1, 3000), Q(216), Q(14, 10**12)
    rho = g * phase**2 / 9
    coupling = g * horizon
    inverse = exact_matrix_inverse(((Q(1), -coupling), (-coupling, Q(1))))
    error, phase_error = tuple(row[0] * horizon * rho for row in inverse)
    assert all(value >= 0 for row in inverse for value in row)
    assert error == horizon * rho + coupling * phase_error
    assert phase_error == coupling * error
    assert error < Q(16, 10**25) < Q(15, 10**21)
    assert phase_error < Q(12, 10**26)
    # Initial collective errors would add a separate positive contribution.
    assert inverse[0][0] * Q(1, 10**14) > error


def test_observation_class_can_have_disjoint_variance_targets():
    amplitude, mu = Q(1, 2**26), Q(1, 10**17)
    zero = (Q(0),) * 45
    perturbed = (amplitude, -amplitude) + zero[2:]
    assert _means(zero) == _means(perturbed) == (Q(0),) * 5
    assert sum(value**2 for value in perturbed) < Q(601, 10**10) ** 2
    assert sum(value**2 for value in perturbed) > Q(14, 10**12) ** 2
    assert max(abs(value) for value in perturbed) < Q(1, 10**6)
    assert (2 * pi_interval() / 5).hi < Q(7, 5)
    difference = sum(
        abs(a - b) for a, b in zip(_variances(zero), _variances(perturbed))
    )
    assert difference == 4 * amplitude**2 / 9 > 2 * mu


def test_uniform_carrier_envelope_gives_joint_noise_box_overlap():
    g, horizon, noise = Q(1, 3000), Q(16), Q(1, 10**14)
    transport = _transport_hidden_deviations(
        initial_form=Q(31, 10**9),
        initial_phase=Q(601, 10**10),
        gamma_upper=g,
        laplacian_upper=Q(8),
        duration=horizon,
    )
    residual = g * transport.phase_upper**2 / 9
    error = horizon * residual / (1 - (g * horizon) ** 2)
    assert 0 < error < Q(23, 10**19) < noise
    # After translating by any carrier reading, every possible discrepancy
    # lies inside the allowed noise interval. Cartesian products retain this
    # containment at all 150 times/coordinates, including correlated errors.
    assert I(-noise, noise).contains(I(-error, error))
    for discrepancy in (-error, Q(0), error):
        allowed = I(discrepancy - noise, discrepancy + noise)
        assert allowed.contains(0)
        assert allowed.contains(discrepancy / 2)
        assert I(-noise, noise).contains(discrepancy / 2)
