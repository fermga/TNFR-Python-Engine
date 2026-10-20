"""Independent complete-row and information controls for the organized port."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_reservoir_exchange as auxiliary
from tnfr.physics import _organized_source_port as owner

_ZERO = (Q(0),) * 45
_ERROR = Q(1, 10**10)
_HOST = (
    tuple((9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(4) for j in range(9))
    + tuple((18 * g + j, 18 * g + 9 + j, Q(1)) for g in range(2) for j in range(9))
    + tuple((9 * a + j, 18 + 9 * a + j, Q(1)) for a in range(2) for j in range(9))
)
_DONOR = tuple((36 + j, 36 + (j + 1) % 9, Q(1)) for j in range(9))
_PORT = tuple((j, 36 + j, Q(1)) for j in range(9))
_EDGES = _HOST + _DONOR + _PORT
_COLLECTIVE_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
_FORMS = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
_PHASES = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _fields(mp, form, theta, edges):
    gradient = [0] * len(form)
    current = [mp.mpf(0)] * len(form)
    storage = mp.mpf(0)
    for left, right, weight in edges:
        difference = form[left] - form[right]
        gap = theta[right] - theta[left]
        gradient[left] += weight * difference
        gradient[right] -= weight * difference
        sine = _mp(mp, weight) * mp.sin(gap)
        current[left] += sine
        current[right] -= sine
        storage += _mp(mp, weight) * (difference**2 / 2 + 1 - mp.cos(gap))
    return tuple(gradient), tuple(current), storage


def _observe(**changes):
    return owner._observe_organized_source_port(
        **(dict(form=_FORMS, phase_offsets=_PHASES) | changes)
    )


def _collective(**changes):
    return owner._observe_organized_source_collective(
        **(
            dict(
                form_means=(Q(1, 7), -Q(1, 11), Q(1, 13), -Q(1, 17), Q(1, 19)),
                phase_origins=(Q(1, 9), -Q(1, 10), Q(1, 12), -Q(1, 14), Q(1, 8)),
            )
            | changes
        )
    )


def _assess(**changes):
    return owner._assess_organized_source_attachment(
        **(
            dict(
                host_form_error_bounds=(_ERROR,) * 4,
                host_phase_error_bounds=(_ERROR,) * 4,
                donor_form_error_bound=_ERROR,
                donor_phase_error_bound=_ERROR,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def observation():
    return _observe()


@pytest.fixture(scope="module")
def collective():
    return _collective()


@pytest.fixture(scope="module")
def attachment():
    return _assess()


def test_support_uses_one_uniform_clock_on_nonuniform_degrees(observation):
    geometry = observation.geometry
    assert geometry.host_edges == _HOST
    assert geometry.donor_edges == _DONOR
    assert geometry.port_edges == _PORT
    assert geometry.weighted_edges == _EDGES
    assert len({tuple(sorted((a, b))) for a, b, _ in _EDGES}) == 90
    expected_degrees = (Q(5),) * 9 + (Q(4),) * 27 + (Q(3),) * 9
    assert geometry.weighted_degrees == expected_degrees
    assert geometry.support_degrees == expected_degrees
    assert geometry.node_order == tuple(range(45))
    assert geometry.held_capacities == (Q(1),) * 45
    assert geometry.clock_factor == Q(1, 4)


def test_complete_fine_rows_and_storage_are_independently_enclosed(observation, mp):
    report = observation
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    x = tuple(_mp(mp, value) for value in report.form)
    theta = tuple(
        alpha * (i % 9 - 4) + _mp(mp, value)
        for i, value in enumerate(report.phase_offsets)
    )
    q, current, energy = _fields(mp, x, theta, _EDGES)
    dx = tuple((-lap + gamma * sine) / 4 for lap, sine in zip(q, current))
    dtheta = tuple(gamma * lap / 4 for lap in q)
    for i in range(45):
        _contains(mp, report.phase_lift_bounds[i], theta[i])
        assert abs(_mp(mp, report.form_gradient[i]) - q[i]) < mp.mpf("1e-140")
        _contains(mp, report.phase_current_bounds[i], current[i])
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("slow", 2 / gamma**2),
        ("original", mp.mpf(1023) / 1024),
    ):
        for i in range(45):
            _contains(
                mp, getattr(report, f"form_{clock}_rate_bounds")[i], factor * dx[i]
            )
            _contains(
                mp, getattr(report, f"phase_{clock}_rate_bounds")[i], factor * dtheta[i]
            )
    for child in range(5):
        start = 9 * child
        assert report.child_form_means[child] == sum(report.form[start : start + 9]) / 9
        assert (
            report.child_phase_offset_means[child]
            == sum(report.phase_offsets[start : start + 9]) / 9
        )
        _contains(
            mp, report.child_form_tau_rate_bounds[child], sum(dx[start : start + 9]) / 9
        )
        _contains(
            mp,
            report.child_phase_tau_rate_bounds[child],
            sum(dtheta[start : start + 9]) / 9,
        )
    assert abs(sum(dx)) < mp.mpf("1e-140")
    assert abs(sum(dtheta)) < mp.mpf("1e-140")
    loss = sum(value**2 for value in q) / 4
    assert abs(_mp(mp, report.joint_dissipation_tau) - loss) < mp.mpf("1e-140")
    assert report.joint_storage_tau_rate == -report.joint_dissipation_tau < 0
    assert (
        report.joint_storage_original_rate
        == Q(1023, 1024) * report.joint_storage_tau_rate
    )
    _contains(mp, report.joint_storage_slow_rate_bounds, -2 * loss / gamma**2)
    _contains(mp, report.joint_storage_bounds, energy)
    direct = sum(a * b - c * d for a, b, c, d in zip(q, dx, current, dtheta))
    _contains(mp, report.storage_rate_from_rows_bounds, direct)
    assert abs(direct + loss) < mp.mpf("1e-140")
    assert report.balance_residual_bounds.contains(0)
    # Differentiate the complete storage, independently of the gradient identity.
    step = mp.mpf("1e-55")
    advanced_x = tuple(value + mp.j * step * rate for value, rate in zip(x, dx))
    advanced_t = tuple(value + mp.j * step * rate for value, rate in zip(theta, dtheta))
    energy_derivative = mp.im(_fields(mp, advanced_x, advanced_t, _EDGES)[2]) / step
    assert abs(energy_derivative + loss) < mp.mpf("1e-100")


def test_actual_port_has_two_opposite_channels_and_separate_contact_work(
    observation, mp
):
    report = observation
    gamma = 1 / (1023 * mp.pi)
    x = tuple(_mp(mp, value) for value in report.form)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value)
        for i, value in enumerate(report.phase_offsets)
    )
    for j in range(9):
        u, delta = x[36 + j] - x[j], theta[36 + j] - theta[j]
        form, phase = (u + gamma * mp.sin(delta)) / 4, -gamma * u / 4
        _contains(mp, report.host_port_form_tau_bounds[j], form)
        _contains(mp, report.host_port_phase_tau_bounds[j], phase)
        _contains(mp, report.donor_port_form_tau_bounds[j], -form)
        _contains(mp, report.donor_port_phase_tau_bounds[j], -phase)
    for field, edges in (
        ("host_storage_bounds", _HOST),
        ("donor_storage_bounds", _DONOR),
        ("interaction_storage_bounds", _PORT),
    ):
        _contains(mp, getattr(report, field), _fields(mp, x, theta, edges)[2])


def test_regional_powers_include_phase_and_actual_contact_storage(observation, mp):
    report = observation
    gamma = 1 / (1023 * mp.pi)
    x = tuple(_mp(mp, value) for value in report.form)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value)
        for i, value in enumerate(report.phase_offsets)
    )
    q, current, _ = _fields(mp, x, theta, _EDGES)
    dx = tuple((-lap + gamma * sine) / 4 for lap, sine in zip(q, current))
    dt = tuple(gamma * lap / 4 for lap in q)
    port_x = tuple(
        (x[36 + j] - x[j] + gamma * mp.sin(theta[36 + j] - theta[j])) / 4
        for j in range(9)
    )
    port_t = tuple(-gamma * (x[36 + j] - x[j]) / 4 for j in range(9))
    regional_rates = []
    for name, edges, start, stop, sign in (
        ("host", _HOST, 0, 36, 1),
        ("donor", _DONOR, 36, 45, -1),
    ):
        gradient, sine, _ = _fields(mp, x, theta, edges)
        regional_rate = sum(a * b - c * d for a, b, c, d in zip(gradient, dx, sine, dt))
        regional_rates.append(regional_rate)
        loss = sum(gradient[i] ** 2 for i in range(start, stop)) / 4
        power = sign * sum(
            gradient[start + j] * port_x[j] - sine[start + j] * port_t[j]
            for j in range(9)
        )
        _contains(mp, getattr(report, f"{name}_port_power_tau_bounds"), power)
        _contains(mp, getattr(report, f"{name}_storage_tau_rate_bounds"), regional_rate)
        assert abs(
            _mp(mp, getattr(report, f"{name}_internal_dissipation_tau")) - loss
        ) < mp.mpf("1e-140")
        assert abs(regional_rate + loss - power) < mp.mpf("1e-140")
        for i, source_i in enumerate(range(start, stop)):
            assert abs(
                _mp(mp, getattr(report, f"{name}_internal_form_gradient")[i])
                - gradient[source_i]
            ) < mp.mpf("1e-140")
            _contains(
                mp,
                getattr(report, f"{name}_internal_phase_current_bounds")[i],
                sine[source_i],
            )
    contact_rate = sum(
        (x[36 + j] - x[j]) * (dx[36 + j] - dx[j])
        + mp.sin(theta[36 + j] - theta[j]) * (dt[36 + j] - dt[j])
        for j in range(9)
    )
    _contains(mp, report.interaction_storage_tau_rate_bounds, contact_rate)
    assert abs(
        sum(regional_rates) + contact_rate + _mp(mp, report.joint_dissipation_tau)
    ) < mp.mpf("1e-140")
    assert report.regional_balance_residual_bounds.contains(0)
    # Isolated component losses do not account for energy in the added edges.
    assert (
        report.host_internal_dissipation_tau + report.donor_internal_dissipation_tau
        != report.joint_dissipation_tau
    )


def test_carrier_rows_pull_back_the_full_law_without_a_degree_change(collective, mp):
    report = collective
    x = tuple(_mp(mp, value) for value in report.form_means)
    phi = tuple(_mp(mp, value) for value in report.phase_origins)
    gamma = 1 / (1023 * mp.pi)
    coarse_edges = tuple((a, b, Q(1)) for a, b in _COLLECTIVE_EDGES)
    q, current, interface_energy = _fields(mp, x, phi, coarse_edges)
    full = _observe(
        form=tuple(value for value in report.form_means for _ in range(9)),
        phase_offsets=tuple(value for value in report.phase_origins for _ in range(9)),
    )
    for i in range(5):
        dx, dt = (-q[i] + gamma * current[i]) / 4, gamma * q[i] / 4
        for clock, factor in (
            ("tau", 1),
            ("slow", 2 / gamma**2),
            ("original", mp.mpf(1023) / 1024),
        ):
            _contains(mp, getattr(report, f"form_{clock}_rate_bounds")[i], factor * dx)
            _contains(mp, getattr(report, f"phase_{clock}_rate_bounds")[i], factor * dt)
        for j in range(9):
            _contains(mp, full.form_tau_rate_bounds[9 * i + j], dx)
            _contains(mp, full.phase_tau_rate_bounds[9 * i + j], dt)
    expected_storage = 45 * (1 - mp.cos(2 * mp.pi / 9)) + 9 * interface_energy
    _contains(mp, report.joint_storage_bounds, expected_storage)
    _contains(mp, full.joint_storage_bounds, expected_storage)
    expected_loss = 9 * sum(value**2 for value in q) / 4
    assert abs(_mp(mp, report.joint_dissipation_tau) - expected_loss) < mp.mpf("1e-140")
    assert report.joint_storage_tau_rate == -report.joint_dissipation_tau
    assert (
        report.joint_storage_original_rate
        == -Q(1023, 1024) * report.joint_dissipation_tau
    )
    _contains(mp, report.joint_storage_slow_rate_bounds, -2 * expected_loss / gamma**2)
    _contains(mp, report.storage_rate_from_rows_bounds, -expected_loss)
    assert report.balance_residual_bounds.contains(0)
    # The one-child attachment breaks the old equal-children parent family.
    equal_host = _collective(form_means=(0, 0, 0, 0, 1), phase_origins=(0,) * 5)
    assert equal_host.form_tau_rate_bounds[0].lo > 0
    assert equal_host.form_tau_rate_bounds[1].contains(0)
    assert equal_host.phase_tau_rate_bounds[0].hi < 0


def test_retained_totals_reconstruct_donor_on_the_carrier(collective):
    restored = owner._reconstruct_organized_source_collective(
        host_form_means=collective.form_means[:4],
        host_phase_origins=collective.phase_origins[:4],
        conserved_form_total=sum(collective.form_means),
        conserved_phase_total=sum(collective.phase_origins),
    )
    assert restored == collective


@pytest.mark.parametrize("field", ("conserved_form_total", "conserved_phase_total"))
@pytest.mark.parametrize("bad", (True, "0", None, float("inf"), complex(0)))
def test_reconstruction_totals_are_primitive_information(field, bad, collective):
    arguments = dict(
        host_form_means=collective.form_means[:4],
        host_phase_origins=collective.phase_origins[:4],
        conserved_form_total=sum(collective.form_means),
        conserved_phase_total=sum(collective.phase_origins),
    )
    arguments[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._reconstruct_organized_source_collective(**arguments)


def test_all_ten_means_and_totals_do_not_close_off_family_current(mp):
    delta, h = Q(1, 10), Q(1, 100)
    original = (Q(0),) * 36 + (delta,) * 9
    perturbed = original[:36] + (delta + h, delta - h) + (delta,) * 7
    first = _observe(form=_ZERO, phase_offsets=original)
    second = _observe(form=_ZERO, phase_offsets=perturbed)
    assert first.form == second.form
    assert first.phase_offsets[:36] == second.phase_offsets[:36]
    assert first.child_form_means == second.child_form_means
    assert first.child_phase_offset_means == second.child_phase_offset_means
    assert sum(first.phase_offsets) == sum(second.phase_offsets)
    rate_difference = (
        second.child_form_tau_rate_bounds[0] - first.child_form_tau_rate_bounds[0]
    )
    expected = mp.sin(_mp(mp, delta)) * (mp.cos(_mp(mp, h)) - 1) / (18 * 1023 * mp.pi)
    _contains(mp, rate_difference, expected)
    assert rate_difference.hi < 0


def test_form_only_donor_observation_loses_phase_and_old_port_loses_reaction(mp):
    # Totals are not held in this comparison: they would restore the missing origin.
    first = _collective(form_means=(0,) * 5, phase_origins=(0,) * 5)
    changed_phase = _collective(
        form_means=(0,) * 5, phase_origins=(0,) * 4 + (Q(1, 10),)
    )
    assert first.form_means == changed_phase.form_means
    assert first.phase_origins[:4] == changed_phase.phase_origins[:4]
    assert changed_phase.form_tau_rate_bounds[0].lo > first.form_tau_rate_bounds[0].hi
    phase_reaction = _observe(form=(Q(0),) * 36 + (Q(1),) * 9, phase_offsets=_ZERO)
    old = auxiliary._observe_parent_reservoir_exchange(
        form=(Q(0),) * 36,
        phase_offsets=(Q(0),) * 36,
        reservoir_state=Q(1),
        port="connected",
    )
    assert phase_reaction.phase_tau_rate_bounds[0].hi < 0
    assert old.phase_tau_rate_bounds[0].lo == old.phase_tau_rate_bounds[0].hi == 0
    # Complete port derivative has rank two on the acute independent-source domain.
    gamma, delta = 1 / (1023 * mp.pi), mp.mpf("0.1")

    def port(u, angle):
        return (u + gamma * mp.sin(angle)) / 4, -gamma * u / 4

    jacobian = mp.matrix(
        [
            [
                mp.diff(lambda u: port(u, delta)[row], 0),
                mp.diff(lambda angle: port(0, angle)[row], delta),
            ]
            for row in range(2)
        ]
    )
    assert abs(mp.det(jacobian) - gamma**2 * mp.cos(delta) / 16) < mp.mpf("1e-140")
    assert mp.det(jacobian) > 0


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0))
)
def test_invalid_fine_primitives_reject_before_geometry(field, bad, monkeypatch):
    def forbidden():
        pytest.fail("invalid primitive reached geometry")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: (Q(0),) * 44 + (bad,)})


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", ((), (0,) * 44, (0,) * 46, "0" * 45, {0}))
def test_fine_vector_shapes_do_not_coerce_or_truncate(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _observe(**{field: bad})


@pytest.mark.parametrize("field", ("form_means", "phase_origins"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), complex(0)))
def test_collective_primitives_have_the_same_admission(field, bad):
    with pytest.raises((TypeError, ValueError)):
        _collective(**{field: (Q(0),) * 4 + (bad,)})


def test_tiny_exact_form_and_lift_are_retained_before_interval_rounding():
    tiny = Q(1, 10**1000)
    report = _observe(form=(tiny,) + _ZERO[1:], phase_offsets=(tiny,) + _ZERO[1:])
    assert report.form[0] == report.phase_offsets[0] == tiny
    assert report.form_gradient[0] == 5 * tiny
    assert report.child_form_means[0] == report.child_phase_offset_means[0] == tiny / 9
    assert report.joint_dissipation_tau == Q(30, 4) * tiny**2 > 0
    lifted = _observe(phase_offsets=(Q(7),) + _ZERO[1:])
    assert lifted.phase_offsets[0] == 7
    assert lifted.child_phase_offset_means[0] == Q(7, 9)


@pytest.mark.parametrize(
    "extra",
    (
        "source_report",
        "trajectory",
        "capacity",
        "gamma",
        "duration",
        "degree_normalized",
        "reservoir_state",
    ),
)
def test_fixed_model_cannot_be_replaced_by_a_report_or_alternative_law(extra):
    with pytest.raises(TypeError):
        _observe(**{extra: 0})


def test_reports_are_immutable_and_carry_no_future_identity_verdict(observation):
    with pytest.raises(FrozenInstanceError):
        observation.form = _ZERO
    for field in (
        "source_certified",
        "entry_certified",
        "maintenance_certified",
        "trajectory",
    ):
        assert not hasattr(observation, field)


def test_actual_five_sources_keep_preparation_clock_and_eighty_directions(attachment):
    report = attachment
    sources = report.host.child_sources + (report.donor,)
    assert len({id(source) for source in sources}) == 5
    assert len({id(source.source) for source in sources}) == 5
    assert report.relative_preparation_dimension == 5 * 2 * (9 - 1)
    for source in sources:
        assert source.form_error_bound == source.phase_error_bound == _ERROR
        assert source.deformation == 0
        assert source.recovery_certified
        assert source.endpoint_form_norm_squared_upper_bound < Q(1, 10**140)
        assert source.endpoint_phase_norm_squared_upper_bound < Q(1, 10**140)
    assert report.host.acquisition_slow_time == 4
    assert report.donor_clock_dilation == 2
    assert report.donor_formation_tau == 200
    assert report.donor_recovery_duration_tau == 2 * 10**13
    assert report.donor_handoff_tau == 2 * (100 + 10**13)
    assert report.donor_handoff_original == Q(1024, 1023) * report.donor_handoff_tau
    assert report.donor_phase_origin == Q(1, 8)
    assert report.donor_form_mean == report.conserved_form_total == 0
    assert report.conserved_phase_total == Q(1, 8)
    assert 0 < report.host_fine_form_error_upper_bound < Q(1, 10**60)
    assert 0 < report.host_fine_phase_error_upper_bound < Q(1, 10**60)
    assert report.donor_recovered_norm_bound == Q(1, 10**70)
    assert report.sources_certified and report.initial_child_identities_certified
    assert report.positive_attachment_work_certified and report.admitted
    assert not report.unresolved_conditions
    assert report.protocol_revision == "02a6415f"
    with pytest.raises(FrozenInstanceError):
        report.admitted = False
    for field in (
        "maintenance_certified",
        "response_certified",
        "capture_certified",
        "trajectory",
    ):
        assert not hasattr(report, field)


def test_source_costs_and_new_contact_work_are_separate(attachment, mp):
    report = attachment
    gamma = 1 / (1023 * mp.pi)
    slope = Q(2046, 9) * Q(355, 113) ** 2
    ramp = tuple(slope * (j - 4) for j in range(9))
    ramp_cost = sum((ramp[(j + 1) % 9] - ramp[j]) ** 2 / 2 for j in range(9))
    # Eight ordinary ramp edges and its retained long closing edge.
    assert ramp_cost == 36 * slope**2
    initial_bound = ramp_cost + 32 * slope * _ERROR + 36 * _ERROR**2
    assert report.donor_original_preparation_storage_upper_bound == initial_bound
    assert report.original_preparation_storage_upper_bound == 5 * initial_bound
    assert initial_bound < 10**9
    donor_target = 9 * (1 - mp.cos(2 * mp.pi / 9))
    _contains(mp, report.donor_pre_attachment_storage_bounds, donor_target)
    assert report.donor_pre_attachment_excess_storage_upper_bound == 4 * Q(1, 10**140)
    assert (
        report.host_pre_attachment_storage_upper_bound
        >= report.host.complete_initial_storage_bounds.hi
    )
    assert report.joined_initial_storage_upper_bound >= (
        report.host_pre_attachment_storage_upper_bound
        + report.donor_pre_attachment_storage_bounds.hi
        + report.attachment_work_upper_bound
    )
    nominal_phase = (
        report.host.nominal_inner_gap_at_entry_bounds.hi
        + report.host.nominal_outer_gap_at_entry_bounds.hi
    ) / 2
    assert nominal_phase < Q(11, 1280)
    xerror = report.host_fine_form_error_upper_bound + report.donor_recovered_norm_bound
    terror = (
        report.host_fine_phase_error_upper_bound + report.donor_recovered_norm_bound
    )
    expected_form_bound = gamma * (1 + 2 * gamma**2) * _mp(mp, nominal_phase) + _mp(
        mp, xerror
    )
    assert _mp(mp, report.contact_form_gap_absolute_upper_bound) >= expected_form_bound
    assert report.contact_phase_gap_lower_bound == Q(1, 8) - terror > Q(3, 25)
    assert (
        report.contact_phase_gap_upper_bound
        == Q(1, 8) + nominal_phase + terror
        < Q(1, 4)
    )
    assert (
        Q(1, 16)
        < report.attachment_work_lower_bound
        < report.attachment_work_upper_bound
        < Q(1, 10)
    )
    # Independent global trigonometric bound on each of the nine actual edges.
    phase_lower = _mp(mp, report.contact_phase_gap_lower_bound)
    assert _mp(mp, report.attachment_work_lower_bound) <= 9 * (1 - mp.cos(phase_lower))
    assert report.attachment_work_upper_bound == Q(9, 2) * (
        report.contact_form_gap_absolute_upper_bound**2
        + report.contact_phase_gap_upper_bound**2
    )


@pytest.mark.parametrize(
    "field",
    (
        "host_form_error_bounds",
        "host_phase_error_bounds",
        "donor_form_error_bound",
        "donor_phase_error_bound",
    ),
)
@pytest.mark.parametrize("bad", (True, "0", float("nan"), -Q(1, 10**1000)))
def test_all_ten_source_primitives_are_admitted_before_any_rebuild(
    field, bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("invalid allowance reached a source owner")

    monkeypatch.setattr(owner, "_assess_organized_parent_acquisition", forbidden)
    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    value = (_ERROR,) * 3 + (bad,) if field.startswith("host_") else bad
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: value})


@pytest.mark.parametrize("field", ("host_form_error_bounds", "host_phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (_ERROR,) * 3, (_ERROR,) * 5, "0000", {_ERROR}))
def test_source_vector_shapes_reject_before_any_rebuild(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid allowance shape reached a source owner")

    monkeypatch.setattr(owner, "_assess_organized_parent_acquisition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "field, value",
    (
        ("recovery_certified", False),
        ("endpoint_form_norm_squared_upper_bound", None),
        ("endpoint_phase_norm_squared_upper_bound", Q(1, 10**140)),
    ),
)
def test_actual_donor_recovery_and_strict_radius_are_required(
    field, value, attachment, monkeypatch
):
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: attachment.host
    )
    monkeypatch.setattr(
        owner,
        "_assess_constitutive_recovery",
        lambda **kwargs: replace(attachment.donor, **{field: value}),
    )
    report = _assess()
    assert (
        not report.sources_certified and not report.initial_child_identities_certified
    )
    assert not report.admitted and report.status == "unavailable"
    assert (
        "donor_clock_transformed_original_recovery_not_certified"
        in report.unresolved_conditions
    )


@pytest.mark.parametrize("field", ("entry_certified", "child_identities_retained"))
def test_host_handoff_is_required_before_joined_source_admission(
    field, attachment, monkeypatch
):
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_acquisition",
        lambda **kwargs: replace(attachment.host, **{field: False}),
    )
    monkeypatch.setattr(
        owner, "_assess_constitutive_recovery", lambda **kwargs: attachment.donor
    )
    report = _assess()
    assert not report.sources_certified and not report.admitted
    assert "host_actual_acquired_source_not_certified" in report.unresolved_conditions


def test_unrelated_host_future_prediction_does_not_decide_attachment(
    attachment, monkeypatch
):
    host = replace(
        attachment.host,
        prediction_certified=False,
        parent_identity_retained=False,
        admitted=False,
        status="unavailable",
    )
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: host
    )
    monkeypatch.setattr(
        owner, "_assess_constitutive_recovery", lambda **kwargs: attachment.donor
    )
    report = _assess()
    assert report.sources_certified and report.admitted


@pytest.mark.parametrize(
    "extra",
    (
        "source_report",
        "host",
        "donor",
        "donor_phase_origin",
        "duration",
        "clock_dilation",
        "trajectory",
    ),
)
def test_source_reports_and_protocol_replacements_are_not_inputs(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})
