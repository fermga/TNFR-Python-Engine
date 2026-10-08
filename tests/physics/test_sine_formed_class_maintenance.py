"""Full-row Lyapunov and uniform half-neighborhood return controls for C9."""

from fractions import Fraction as Q
from pathlib import Path
from typing import get_type_hints

import mpmath
import pytest

from tnfr.physics import relational_sine_formed_class_maintenance as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

ERROR, DWELL = Q(1, 10**10), Q(10**12)
FIELDS = (
    "formation_time",
    "probe_time",
    "probe_duration",
    "phase_increment",
    "form_error_bound",
    "phase_error_bound",
    "readout_error_bound",
    "radius",
    "common_dwell",
)


def _assess(**changes):
    values = dict(
        formation_time=Q(100),
        probe_time=Q(200),
        probe_duration=Q(1),
        phase_increment=Q(1, 100),
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        readout_error_bound=ERROR,
        radius=Q(1, 12),
        common_dwell=DWELL,
    )
    return owner.assess_sine_formed_class_maintenance(**(values | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _a(values):
    return [
        (2 * values[i] - values[(i - 1) % 9] - values[(i + 1) % 9]) / 2
        for i in range(9)
    ]


def _forcing(mp, theta):
    return [
        sum(mp.sin(theta[j] - theta[i]) for j in ((i - 1) % 9, (i + 1) % 9)) / 2
        for i in range(9)
    ]


def _potential(mp, theta):
    return sum(1 - mp.cos(theta[(i + 1) % 9] - theta[i]) for i in range(9))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _inverse_quadratic(mp, values):
    total = mp.mpf(0)
    for mode in range(1, 9):
        angle = 2 * mp.pi * mode / 9
        coefficient = sum(values[j] * mp.exp(-mp.j * angle * j) for j in range(9))
        total += abs(coefficient) ** 2 / (1 - mp.cos(angle))
    return total / 9


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize(
    "x_scale,y_scale",
    [(0, Q(1, 3)), (Q(1, 3), 0), (Q(1, 4), Q(1, 3)), (-Q(1, 4), Q(1, 3))],
)
def test_complete_nonlinear_rows_prove_lyapunov_loss_and_norm_equivalence(
    frozen, mp, k, x_scale, y_scale
):
    gamma, eta, epsilon = 1 / (1023 * mp.pi), 1 / (1023 * mp.pi) ** 2, mp.mpf(1) / 20
    radius = _mp(mp, frozen.reference_certificate.radius)
    direction = [(mp.mpf(i) - 4) / mp.sqrt(60) for i in range(9)]
    other = [mp.sin(4 * mp.pi * i / 9) * mp.sqrt(mp.mpf(2) / 9) for i in range(9)]
    x = [radius * _mp(mp, Q(x_scale)) * value for value in direction]
    y = [radius * _mp(mp, Q(y_scale)) * value for value in other]
    target = [(i - 4) * 2 * k * mp.pi / 9 for i in range(9)]
    theta = [value + error for value, error in zip(target, y)]
    assert _dot(x, x) + _dot(y, y) < radius**2
    ax, f = _a(x), _forcing(mp, theta)
    dx, dy = [-a + gamma * force for a, force in zip(ax, f)], [gamma * a for a in ax]
    energy = _potential(mp, theta) - _potential(mp, target)
    lyapunov = (
        eta * (_dot(x, ax) + energy) / 2
        + epsilon * gamma * _dot(y, x)
        + epsilon * _dot(y, y) / 2
    )
    grad_x = [eta * a + epsilon * gamma * b for a, b in zip(ax, y)]
    grad_y = [
        -eta * force + epsilon * gamma * a + epsilon * b for force, a, b in zip(f, x, y)
    ]
    direct_derivative = _dot(grad_x, dx) + _dot(grad_y, dy)
    cancellation = (
        -eta * _dot(ax, ax) + epsilon * eta * _dot(x, ax) + epsilon * eta * _dot(y, f)
    )
    assert abs(direct_derivative - cancellation) < mp.mpf("1e-110")
    assert direct_derivative <= -_mp(mp, frozen.lyapunov_decay_rate) * lyapunov
    velocity_square = eta * _dot(x, ax)
    position_square = _inverse_quadratic(mp, y)
    lower = (
        velocity_square / 4
        + _mp(mp, frozen.lyapunov_position_lower_coefficient) * position_square
    )
    upper = (
        3 * velocity_square / 4
        + _mp(mp, frozen.lyapunov_position_upper_coefficient) * position_square
    )
    assert lower <= lyapunov <= upper


def test_exact_constants_and_uniform_return_preserve_half_radii(frozen, mp):
    reference = frozen.reference_certificate
    low, high = frozen.gamma_squared_bounds.lo, frozen.gamma_squared_bounds.hi
    gap, rate, epsilon = Q(1, 5), Q(2), Q(1, 20)
    cosine = min(reference.coercivity_lower_bounds_by_class) / gap
    mu, upper = low * cosine * gap**2, high * rate**2
    a_minus, a_plus = mu / 2 + gap**2 / 16, upper / 2 + epsilon * rate / 2 + epsilon**2
    assert frozen.phase_stiffness_lower_bound == mu
    assert frozen.phase_stiffness_upper_bound == upper
    assert frozen.lyapunov_position_lower_coefficient == a_minus
    assert frozen.lyapunov_position_upper_coefficient == a_plus
    assert frozen.lyapunov_decay_rate == min(
        4 * (gap - epsilon) / 3, epsilon * mu / a_plus
    )
    exponent = frozen.lyapunov_decay_rate * DWELL
    assert 0 < exponent <= 4096 and DWELL * gap > 4096
    decay = mp.exp(-_mp(mp, exponent))
    assert (
        _mp(mp, frozen.exponential_decay_bounds.lo)
        <= decay
        <= _mp(mp, frozen.exponential_decay_bounds.hi)
    )
    for i in (0, 1):
        bx, theta = (
            reference.warmup_form_norm_upper_bounds[i],
            reference.warmup_target_phase_radius_upper_bounds[i],
        )
        post_theta = reference.post_probe_phase_radius_upper_bounds[i]
        initial = Q(3, 4) * high * rate * bx**2 + a_plus * post_theta**2 / gap
        assert frozen.post_probe_lyapunov_upper_bounds[i] == initial
        threshold = min(low * gap * bx**2 / 16, a_minus * theta**2 / (4 * rate))
        assert frozen.return_energy_thresholds[i] == threshold
        returned = initial * frozen.exponential_decay_bounds.hi
        assert returned == frozen.returned_lyapunov_upper_bounds[i] < threshold
        x_squared, y_squared = 4 * returned / (low * gap), rate * returned / a_minus
        assert (
            frozen.returned_form_norm_squared_upper_bounds[i] == x_squared < bx**2 / 4
        )
        assert (
            frozen.returned_phase_norm_squared_upper_bounds[i]
            == y_squared
            < theta**2 / 4
        )
        assert frozen.form_return_margin_bounds[i].lo > 0
        assert frozen.phase_return_margin_bounds[i].lo > 0
    assert frozen.status == "certified_repeated_probe_maintenance"
    assert frozen.return_certified_by_class == (True, True)
    assert frozen.return_radius_factor == Q(1, 2)
    assert frozen.unavailable_reasons == ()
    assert frozen.original_common_dwell == DWELL * Q(1024, 1023)


def test_recurrent_evidence_retains_original_common_probe_report(frozen):
    path = (
        Path(__file__).parents[2] / "docs/assets/sine_formed_classes/response-v1.json"
    )
    original = json_loads(path.read_bytes())
    assert frozen.reference_certificate.to_dict() == original
    reference = frozen.reference_certificate
    assert reference.recorded_contrast_bounds.lo > Q(1, 10**7)
    assert all(work.lo > 0 for work in reference.probe_work_bounds_by_class)
    # Uniform membership reinstates the same per-cycle errors; neither a cycle
    # index nor cumulative readout-error feedback is a premise of this API.
    assert "common_dwell" in get_type_hints(owner.SineFormedClassMaintenance)
    with pytest.raises(TypeError):
        owner.assess_sine_formed_class_maintenance(probe_count=2)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"common_dwell": 2}, "strict_half_neighborhood_return_not_certified"),
        ({"radius": 1}, "positive_common_acute_decay_rate_unavailable"),
        ({"phase_increment": 1}, "post_probe_recovery_prerequisite_unavailable"),
        (
            {"readout_error_bound": Q(1, 100)},
            "formed_class_response_prerequisite_unavailable",
        ),
        ({"formation_time": 0}, "formed_class_response_prerequisite_unavailable"),
    ],
)
def test_insufficient_prerequisites_do_not_claim_maintenance(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable"
    assert any(reason in item for item in result.unavailable_reasons)
    if "radius" in changes or "phase_increment" in changes:
        assert result.exponential_decay_bounds is None
        assert result.returned_form_norm_squared_upper_bounds is None
        assert result.returned_phase_norm_squared_upper_bounds is None
        assert result.form_return_margin_bounds is None
    if "readout_error_bound" in changes or "formation_time" in changes:
        assert result.return_certified_by_class == (True, True)


def test_wider_valid_exponential_enclosure_cannot_false_certify_half_return(
    monkeypatch, frozen
):
    # Widen an already valid exponential enclosure to the energy threshold.
    # This represents insufficient precision, not a changed scientific law.
    radius = (
        frozen.return_energy_thresholds[0] / frozen.post_probe_lyapunov_upper_bounds[0]
    )
    assert frozen.exponential_decay_bounds.hi < radius < 1
    monkeypatch.setattr(owner, "_negative_exp_bounds", lambda exponent: (Q(0), radius))
    result = _assess()
    assert result.return_certified_by_class[0] is False
    assert (
        result.form_return_margin_bounds[0].lo <= 0
        or result.phase_return_margin_bounds[0].lo <= 0
    )
    assert result.status == "unavailable"


def test_exponential_work_cap_applies_to_slow_lyapunov_rate(frozen):
    maximum = Q(4096) / frozen.lyapunov_decay_rate
    admitted = _assess(common_dwell=maximum)
    assert admitted.common_dwell == maximum
    assert admitted.return_certified_by_class == (True, True)
    with pytest.raises(ValueError, match="lyapunov_decay_rate"):
        _assess(common_dwell=maximum + 1 / frozen.lyapunov_decay_rate)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_raw_admission_precedes_the_fresh_reference(monkeypatch, field, bad):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached prerequisite")

    monkeypatch.setattr(owner, "assess_sine_formed_class_response", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"formation_time": -1},
        {"probe_time": 99},
        {"probe_time": 20481},
        {"probe_duration": -1},
        {"probe_duration": 2049},
        {"common_dwell": 1},
        {"common_dwell": 0},
        {"common_dwell": -1},
        {"phase_increment": -1},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
        {"readout_error_bound": -1},
        {"radius": 0},
    ],
)
def test_physical_domains_reject_before_the_fresh_reference(monkeypatch, changes):
    def forbidden(**kwargs):
        pytest.fail("invalid domain reached prerequisite")

    monkeypatch.setattr(owner, "assess_sine_formed_class_response", forbidden)
    with pytest.raises(ValueError):
        _assess(**changes)


def test_fresh_primitive_reference_is_rebuilt_on_each_call(monkeypatch):
    seen = []
    original = owner.assess_sine_formed_class_response

    def reference(**kwargs):
        result = original(**kwargs)
        seen.append((kwargs, result))
        return result

    monkeypatch.setattr(owner, "assess_sine_formed_class_response", reference)
    tiny = Q(1, 2**300)
    first, second = _assess(form_error_bound=0), _assess(form_error_bound=tiny)
    assert len(seen) == 2 and seen[0][1] is not seen[1][1]
    assert seen[0][0]["form_error_bound"] == 0
    assert seen[1][0]["form_error_bound"] == tiny
    assert second.reference_certificate.form_error_bound == tiny
    assert first.reference_certificate.form_error_bound == 0
    with pytest.raises(TypeError):
        owner.assess_sine_formed_class_maintenance(
            reference_certificate=first.reference_certificate
        )


def test_direct_and_generic_export_preserve_inputs_and_availability(frozen, tmp_path):
    direct = frozen.to_dict()
    assert direct["schema"] == "tnfr.sine-formed-class-maintenance.v1"
    assert relational_report_to_dict(frozen)["report"] == direct["report"]
    path = tmp_path / "maintenance.json"
    export_to_json(frozen, path)
    assert json_loads(path.read_bytes())["report"] == direct["report"]
    dwell = direct["report"]["common_dwell"]
    assert Q(dwell["numerator"], dwell["denominator"]) == DWELL
    get_type_hints(owner.SineFormedClassMaintenance)
    get_type_hints(owner.assess_sine_formed_class_maintenance)
