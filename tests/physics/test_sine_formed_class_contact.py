"""Independent complete-row and event controls for the frozen C9 contact.

The fixed candidate is evaluated only after protocol/source archival. Tests
use exact matrices and analytic inequalities; no nonlinear trajectory or
37-coordinate flow enclosure is installed or sampled.
"""

import random
from fractions import Fraction as Q
from math import factorial
from pathlib import Path
from typing import get_type_hints

import mpmath
import pytest

from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.physics import relational_sine_formed_class_contact as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

ERROR, EPSILON, READOUT = Q(1, 10**10), Q(1, 10**32), Q(1, 10**30)
PHI, H, RADIUS = Q(1, 1000), Q(1, 100), Q(1, 12)
REAL_FIELDS = (
    "formation_time",
    "relaxation_duration",
    "phase_origin_difference",
    "contact_duration",
    "form_error_bound",
    "phase_error_bound",
    "endpoint_radius",
    "readout_error_bound",
    "radius",
    "work_allowance",
)
EDGES = tuple(
    (start + i, start + (i + 1) % 9) for start in (0, 9) for i in range(9)
) + ((4, 13),)
DEGREES = tuple(sum(i in edge for edge in EDGES) for i in range(18))


def _assess(**changes):
    inputs = dict(
        formation_time=Q(100),
        relaxation_duration=Q(10**13),
        phase_origin_difference=PHI,
        contact_duration=H,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        endpoint_radius=EPSILON,
        readout_error_bound=READOUT,
        radius=RADIUS,
        work_allowance=Q(1, 10**6),
        decay_power=512,
    )
    return owner.assess_sine_formed_class_contact(**(inputs | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


@pytest.fixture(scope="module", autouse=True)
def no_probe_reset_or_trajectory():
    from tnfr.dynamics import relational
    from tnfr.physics import (
        relational_sine_forecast,
        relational_sine_formed_class_maintenance,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("contact must retain unprobed sources and use analytic bounds")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
        patch.setattr(
            relational_sine_formed_classes,
            "assess_sine_formed_class_response",
            forbidden,
        )
        patch.setattr(
            relational_sine_formed_class_maintenance,
            "assess_sine_formed_class_maintenance",
            forbidden,
        )
        yield


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _laplacian(edges=EDGES):
    matrix = [[Q(0) for _ in range(18)] for _ in range(18)]
    for i, j in edges:
        matrix[i][i] += 1
        matrix[j][j] += 1
        matrix[i][j] -= 1
        matrix[j][i] -= 1
    return tuple(map(tuple, matrix))


def _mv(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _potential(mp, phases, edges=EDGES):
    return sum(1 - mp.cos(phases[j] - phases[i]) for i, j in edges)


def test_complete_joined_laplacian_has_the_declared_exact_gap(frozen):
    laplacian = _laplacian()
    shifted = tuple(
        tuple(laplacian[i][j] - Q(2, 81) * (Q(i == j) - Q(1, 18)) for j in range(18))
        for i in range(18)
    )
    assert exact_symmetric_semidefinite(shifted)
    assert frozen.joined_gap_lower_bound == Q(2, 81)
    assert frozen.joined_degrees == DEGREES
    assert DEGREES[4] == DEGREES[13] == 3 and sum(DEGREES) == 38
    assert set(frozen.edges) == {tuple(sorted(edge)) for edge in EDGES}
    # A component-constant test vector also excludes an implausible gap1;
    # the test does not merely accept every matrix as positive semidefinite.
    direction = (Q(1),) * 9 + (Q(-1),) * 9
    assert sum(a * b for a, b in zip(direction, _mv(laplacian, direction))) == 4
    assert sum(value**2 for value in direction) == 18


def test_full_node_derivative_chain_gives_first_class_difference_at_order_four(
    frozen, mp
):
    laplacian = _laplacian()
    donor_laplacian = _laplacian(EDGES[:9])
    a = tuple(
        tuple(value / DEGREES[i] for value in row) for i, row in enumerate(laplacian)
    )
    donor = tuple(
        tuple(value / DEGREES[i] for value in row)
        for i, row in enumerate(donor_laplacian)
    )
    u = tuple(Q((i == 4) - (i == 13), 3) for i in range(18))
    au = _mv(a, u)
    # B2-B1=-(cos(alpha)-cos(2alpha))*K*L_donor. Both complete
    # evolution rows contribute: -(A*dB*A+dB*A^2)u at derivative4.
    first_path = _mv(a, _mv(donor, au))
    second_path = _mv(donor, _mv(a, au))
    assert -first_path[13] - second_path[13] == Q(11, 81)
    assert _mv(donor, au)[13] == 0  # The receiver cubic derivative still agrees.
    gamma = 1 / (1023 * mp.pi)
    phi = _mp(mp, PHI)
    sine = mp.sin(phi)
    receiver_derivatives = []
    for k in (1, 2):
        theta = [(i - 4) * 2 * k * mp.pi / 9 for i in range(9)] + [
            (i - 13) * 2 * mp.pi / 9 + phi for i in range(9, 18)
        ]
        force = [mp.mpf(0)] * 18
        jacobian = [[mp.mpf(0) for _ in range(18)] for _ in range(18)]
        for i, j in EDGES:
            current, curvature = mp.sin(theta[j] - theta[i]), mp.cos(
                theta[j] - theta[i]
            )
            force[i] += current / DEGREES[i]
            force[j] -= current / DEGREES[j]
            jacobian[i][i] -= curvature / DEGREES[i]
            jacobian[i][j] += curvature / DEGREES[i]
            jacobian[j][j] -= curvature / DEGREES[j]
            jacobian[j][i] += curvature / DEGREES[j]
        assert max(abs(force[i] - sine * _mp(mp, u[i])) for i in range(18)) < mp.mpf(
            "1e-110"
        )
        am = tuple(tuple(_mp(mp, v) for v in row) for row in a)
        x1 = tuple(gamma * v for v in force)
        x2 = tuple(-v for v in _mv(am, x1))
        theta2 = tuple(gamma * v for v in _mv(am, x1))
        x3 = tuple(-d + gamma * f for d, f in zip(_mv(am, x2), _mv(jacobian, theta2)))
        theta3 = tuple(gamma * v for v in _mv(am, x2))
        x4 = tuple(-d + gamma * f for d, f in zip(_mv(am, x3), _mv(jacobian, theta3)))
        receiver_derivatives.append(tuple(row[13] for row in (x1, x2, x3, x4)))
    assert all(
        abs(a - b) < mp.mpf("1e-108")
        for a, b in zip(receiver_derivatives[0][:3], receiver_derivatives[1][:3])
    )
    expected = (
        mp.mpf(11)
        / 81
        * gamma**3
        * sine
        * (mp.cos(2 * mp.pi / 9) - mp.cos(4 * mp.pi / 9))
    )
    assert abs(
        receiver_derivatives[1][3] - receiver_derivatives[0][3] - expected
    ) < mp.mpf("1e-108")
    _contains(mp, frozen.ideal_fourth_derivative_contrast_bounds, expected)
    _contains(mp, frozen.ideal_leading_contrast_bounds, expected * _mp(mp, H) ** 4 / 24)


def _zero_sum_error(random_source, epsilon):
    raw = tuple(Q(random_source.randint(-9, 9)) for _ in range(9))
    centered = tuple(value - sum(raw) / 9 for value in raw)
    scale = sum(map(abs, centered))
    return tuple(epsilon * value / scale for value in centered)


@pytest.mark.parametrize("seed", [11, 29, 71])
def test_event_work_storage_and_new_weighted_means_cover_zero_sum_errors(
    frozen, mp, seed
):
    generator = random.Random(seed)
    eps, phi = _mp(mp, EPSILON), _mp(mp, PHI)
    for k in (1, 2):
        x_blocks = tuple(_zero_sum_error(generator, EPSILON) for _ in range(2))
        y_blocks = tuple(_zero_sum_error(generator, EPSILON) for _ in range(2))
        for block in (*x_blocks, *y_blocks):
            assert sum(block) == 0 and sum(value**2 for value in block) <= EPSILON**2
        x = tuple(value for block in x_blocks for value in block)
        y = tuple(value for block in y_blocks for value in block)
        target = [(i - 4) * 2 * k * mp.pi / 9 for i in range(9)] + [
            (i - 13) * 2 * mp.pi / 9 for i in range(9, 18)
        ]
        phases = [
            value + _mp(mp, y[i]) + (phi if i >= 9 else 0)
            for i, value in enumerate(target)
        ]
        mean_x = sum(DEGREES[i] * x[i] for i in range(18)) / 38
        expected_mean_x = (x[4] + x[13]) / 38
        assert mean_x == expected_mean_x
        assert frozen.joined_form_mean_bounds.contains(mean_x)
        phase_mean = sum(DEGREES[i] * phases[i] for i in range(18)) / 38
        _contains(mp, frozen.joined_phase_mean_bounds, phase_mean)
        assert abs(
            phase_mean - phi / 2 - (_mp(mp, y[4]) + _mp(mp, y[13])) / 38
        ) < mp.mpf("1e-110")
        bridge_work = (
            _mp(mp, x[4] - x[13]) ** 2 / 2 + 1 - mp.cos(phases[13] - phases[4])
        )
        _contains(mp, frozen.bridge_work_bounds, bridge_work)
        form_storage = sum(_mp(mp, x[j] - x[i]) ** 2 / 2 for i, j in EDGES)
        energy = form_storage + _potential(mp, phases) - _potential(mp, target)
        assert energy <= _mp(mp, frozen.joined_excess_storage_upper_bound)
        # The phase-origin displacement is orthogonal to each component's
        # zero-sum error. Quotient geometry uses the ordinary common mean.
        z2 = sum(_mp(mp, value) ** 2 for value in x) + sum(
            (phases[i] - target[i] - phi / 2) ** 2 for i in range(18)
        )
        assert z2 <= _mp(mp, frozen.joined_radius_squared_upper_bound)
        assert z2 <= 4 * eps**2 + mp.mpf(9) / 2 * phi**2
    assert frozen.joined_barrier_lower_bound == Q(2, 81) * Q(1, 20) * RADIUS**2 / 2


def test_unprobed_recovery_keeps_exact_tiny_bounds_and_original_uncertainty(frozen):
    source = frozen.formation_certificate
    assert source.form_error_bound == source.phase_error_bound == ERROR
    assert source.preparation_dimension == 16 and source.full_relative_preparation
    assert source.status == "certified_two_formed_classes"
    assert frozen.exact_decay_upper_bound == Q(1, 2**512)
    assert 512 <= frozen.decay_exponent <= 4096
    assert frozen.handoff_certified_by_class == (True, True)
    assert all(
        0 < value <= EPSILON**2
        for value in frozen.endpoint_form_norm_squared_upper_bounds
    )
    assert all(
        0 < value <= EPSILON**2
        for value in frozen.endpoint_phase_norm_squared_upper_bounds
    )
    assert frozen.contact_time == 100 + 10**13
    assert frozen.original_contact_time == frozen.contact_time * Q(1024, 1023)
    assert frozen.preparation_response_error_upper_bound == 2 * EPSILON / (1 - 3 * H)
    assert frozen.joined_form_mean_bounds.contains(EPSILON / 19)
    assert frozen.joined_phase_mean_bounds.contains(PHI / 2 - EPSILON / 19)


def test_response_remainders_have_independent_high_precision_bounds(frozen, mp):
    gamma = 1 / (1023 * mp.pi)
    h, phi = _mp(mp, H), _mp(mp, PHI)
    sine, dc = mp.sin(phi), mp.cos(2 * mp.pi / 9) - mp.cos(4 * mp.pi / 9)
    leading = mp.mpf(11) / 1944 * gamma**3 * sine * dc * h**4
    tail = 4 * sine * dc * gamma**3 * h**5 / (45 * (1 - h / 2))
    nonlinear = mp.mpf(8) / 15 * sine * gamma**5 * h**5
    assert _mp(mp, frozen.semigroup_tail_upper_bound) >= tail
    assert _mp(mp, frozen.nonlinear_remainder_upper_bound) >= nonlinear
    radius = (
        tail + nonlinear + 2 * _mp(mp, EPSILON) / (1 - 3 * h) + 2 * _mp(mp, READOUT)
    )
    assert _mp(mp, frozen.recorded_contrast_bounds.lo) <= leading - radius
    assert leading + radius <= _mp(mp, frozen.recorded_contrast_bounds.hi)
    assert frozen.recorded_contrast_bounds.lo > Q(1, 10**25)
    assert frozen.status == "certified_formed_class_contact"
    assert (
        frozen.response_certified
        and frozen.identity_certified
        and frozen.work_within_allowance
    )
    assert frozen.unavailable_reasons == ()


def test_correlated_semigroup_series_obeys_the_declared_tail(frozen, mp):
    laplacian, donor_laplacian = _laplacian(), _laplacian(EDGES[:9])
    a = tuple(
        tuple(value / DEGREES[i] for value in row) for i, row in enumerate(laplacian)
    )
    donor = tuple(
        tuple(value / DEGREES[i] for value in row)
        for i, row in enumerate(donor_laplacian)
    )
    u = tuple(Q((i == 4) - (i == 13), 3) for i in range(18))
    powers = [u]
    for _ in range(13):
        powers.append(_mv(a, powers[-1]))
    correction = Q(0)
    scalar_tail = Q(0)
    for n in range(12):
        coefficient = Q(0)
        for p in range(n + 1):
            value = _mv(donor, powers[n - p + 1])
            for _ in range(p):
                value = _mv(a, value)
            coefficient += (-1) ** n * value[13]
        term = coefficient * H ** (n + 3) / factorial(n + 3)
        if n == 0:
            assert term == 0
        elif n == 1:
            assert term == Q(11, 1944) * H**4
        else:
            majorant = Q(n * 2 ** (n + 2), 3 * factorial(n + 3)) * H ** (n + 3)
            assert abs(term) <= majorant
            scalar_tail += majorant
            correction += term
    geometric_tail = Q(4, 45) * H**5 / (1 - H / 2)
    assert 0 < scalar_tail < geometric_tail and abs(correction) <= scalar_tail
    # The same bound follows from the exact successive-term ratios, without
    # diagonalizing the nonsymmetric degree-normalized matrix.
    for n in range(2, 40):
        assert 2 * H * Q(n + 1, n * (n + 4)) <= H / 2
    scale = (
        mp.sin(_mp(mp, PHI))
        * (mp.cos(2 * mp.pi / 9) - mp.cos(4 * mp.pi / 9))
        / (1023 * mp.pi) ** 3
    )
    assert _mp(mp, geometric_tail) * scale <= _mp(mp, frozen.semigroup_tail_upper_bound)


def test_complete_joined_rows_have_the_declared_storage_loss(mp):
    # Nonzero forms and irregular phases exercise both mixed terms; checking
    # only the ideal zero-form contact state would make this identity vacuous.
    x = tuple(Q((7 * i) % 11 - 5, 13) for i in range(18))
    theta = tuple(mp.mpf(i * i - 5 * i) / 17 for i in range(18))
    lx = _mv(_laplacian(), x)
    sine = [mp.mpf(0)] * 18
    for i, j in EDGES:
        current = mp.sin(theta[j] - theta[i])
        sine[i] += current
        sine[j] -= current
    gamma = 1 / (1023 * mp.pi)
    form_rate = tuple(
        (-_mp(mp, lx[i]) + gamma * sine[i]) / DEGREES[i] for i in range(18)
    )
    phase_rate = tuple(gamma * _mp(mp, lx[i]) / DEGREES[i] for i in range(18))
    actual = sum(
        _mp(mp, lx[i]) * form_rate[i] - sine[i] * phase_rate[i] for i in range(18)
    )
    exact_loss = -sum(value**2 / DEGREES[i] for i, value in enumerate(lx))
    assert exact_loss < 0
    assert abs(actual - _mp(mp, exact_loss)) < mp.mpf("1e-110")
    assert abs(sum(DEGREES[i] * form_rate[i] for i in range(18))) < mp.mpf("1e-110")
    assert abs(sum(DEGREES[i] * phase_rate[i] for i in range(18))) < mp.mpf("1e-110")


def test_disconnected_observation_band_retains_independent_errors(frozen):
    radius = 2 * EPSILON / (1 - 3 * H) + 2 * READOUT
    bounds = frozen.disconnected_recorded_contrast_bounds
    assert bounds.contains(-radius) and bounds.contains(radius)
    assert bounds.lo < 0 < bounds.hi
    assert bounds.hi < frozen.recorded_contrast_bounds.lo
    assert bounds.hi - radius < Q(1, 2**128)


def test_work_allowance_is_a_closed_budget_rebuilt_from_primitives():
    work_upper = 2 * EPSILON**2 + (PHI + 2 * EPSILON) ** 2 / 2
    touching = _assess(work_allowance=work_upper)
    below = _assess(work_allowance=work_upper - Q(1, 10**80))
    assert touching.work_within_allowance and touching.work_margin_bounds.contains(0)
    assert touching.status == "certified_formed_class_contact"
    assert not below.work_within_allowance and below.status == "unavailable"
    assert "supplied_contact_work_allowance_not_certified" in below.unavailable_reasons


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"formation_time": 0}, "formation_unavailable"),
        ({"relaxation_duration": 0}, "unprobed_endpoint_budget_not_certified"),
        ({"decay_power": 0}, "unprobed_endpoint_budget_not_certified"),
        ({"endpoint_radius": Q(1, 2**500)}, "unprobed_endpoint_budget_not_certified"),
        (
            {"phase_origin_difference": 0},
            "recorded_receiver_contrast_not_strictly_positive",
        ),
        ({"contact_duration": 0}, "recorded_receiver_contrast_not_strictly_positive"),
        (
            {"readout_error_bound": Q(1, 10**10)},
            "recorded_receiver_contrast_not_strictly_positive",
        ),
        ({"work_allowance": 0}, "supplied_contact_work_allowance_not_certified"),
        ({"phase_origin_difference": 1}, "whole_joined_identity_not_certified"),
    ],
)
def test_invalid_sufficient_bounds_remain_unavailable(changes, reason):
    result = _assess(**changes)
    assert result.status == "unavailable" and reason in result.unavailable_reasons
    if not all(result.handoff_certified_by_class):
        for name in (
            "preparation_response_error_upper_bound",
            "recorded_contrast_bounds",
            "disconnected_recorded_contrast_bounds",
            "joined_radius_squared_upper_bound",
            "joined_excess_storage_upper_bound",
            "joined_radius_margin_bounds",
            "joined_storage_margin_bounds",
            "bridge_work_bounds",
            "work_margin_bounds",
            "joined_form_mean_bounds",
            "joined_phase_mean_bounds",
        ):
            assert getattr(result, name) is None
        assert (
            not result.response_certified
            and not result.identity_certified
            and not result.work_within_allowance
        )


@pytest.mark.parametrize("field", REAL_FIELDS)
@pytest.mark.parametrize("bad", [True, "1", 1j, float("inf"), float("nan")])
def test_raw_scalar_admission_precedes_formation(monkeypatch, field, bad):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached formation")

    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("bad", [True, -1, 4097, Q(512), 512.0, "512", float("nan")])
def test_decay_power_is_an_original_nonboolean_bounded_integer(monkeypatch, bad):
    def forbidden(**kwargs):
        pytest.fail("invalid decay policy reached formation")

    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", forbidden)
    with pytest.raises(ValueError):
        _assess(decay_power=bad)


@pytest.mark.parametrize(
    "changes",
    [
        {"formation_time": -1},
        {"relaxation_duration": -1},
        {"contact_duration": -1},
        {"phase_origin_difference": -1},
        {"phase_origin_difference": Q(1001, 1000)},
        {"contact_duration": Q(251, 1000)},
        {"form_error_bound": -1},
        {"phase_error_bound": -1},
        {"endpoint_radius": 0},
        {"readout_error_bound": -1},
        {"radius": 0},
        {"radius": Q(1, 11)},
        {"work_allowance": -1},
    ],
)
def test_physical_domain_rejects_before_formation(monkeypatch, changes):
    def forbidden(**kwargs):
        pytest.fail("invalid domain reached formation")

    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", forbidden)
    with pytest.raises(ValueError):
        _assess(**changes)


def test_slow_relaxation_work_cap_is_distinct_from_fast_formation_clock(frozen):
    assert frozen.relaxation_duration / 5 > 4096
    with pytest.raises(ValueError, match="lyapunov_decay_rate"):
        _assess(relaxation_duration=Q(4097) / frozen.lyapunov_decay_rate)
    with pytest.raises(ValueError, match="scaled_time"):
        _assess(formation_time=20481)


def test_fresh_original_formation_primitives_and_no_report_input(monkeypatch):
    calls = []
    original = owner.assess_sine_formed_class_pair

    def assess(**kwargs):
        report = original(**kwargs)
        calls.append((kwargs, report))
        return report

    monkeypatch.setattr(owner, "assess_sine_formed_class_pair", assess)
    first = _assess(form_error_bound=0)
    tiny = Q(1, 2**300)
    second = _assess(form_error_bound=tiny)
    assert len(calls) == 2 and calls[0][1] is not calls[1][1]
    assert calls[0][0] == dict(
        scaled_time=Q(100),
        form_error_bound=Q(0),
        phase_error_bound=ERROR,
        radius=RADIUS,
    )
    assert calls[1][0]["form_error_bound"] == tiny
    assert first.formation_certificate.form_error_bound == 0
    assert second.formation_certificate.form_error_bound == tiny
    with pytest.raises(TypeError):
        owner.assess_sine_formed_class_contact(
            formation_certificate=first.formation_certificate
        )


def test_direct_sdk_json_and_frozen_report_projection(frozen, tmp_path):
    direct = frozen.to_dict()
    assert direct["schema"] == "tnfr.sine-formed-class-contact.v1"
    assert relational_report_to_dict(frozen)["report"] == direct["report"]
    path = tmp_path / "contact.json"
    export_to_json(frozen, path)
    assert json_loads(path.read_bytes())["report"] == direct["report"]
    artifact = (
        Path(__file__).parents[2] / "docs/assets/sine_formed_classes/contact-v1.json"
    )
    assert json_loads(artifact.read_bytes()) == direct
    assert direct["report"]["decay_power"] == 512
    for name in (
        "relaxation_duration",
        "phase_origin_difference",
        "contact_duration",
        "endpoint_radius",
        "readout_error_bound",
        "work_allowance",
    ):
        value = direct["report"][name]
        assert Q(value["numerator"], value["denominator"]) == getattr(frozen, name)
    assert (
        get_type_hints(owner.assess_sine_formed_class_contact)["return"]
        is owner.SineFormedClassContact
    )
    get_type_hints(owner.SineFormedClassContact)
