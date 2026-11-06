"""Static independent controls for retained reservoir leaves and local traps."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_reservoir_exchange as exchange
from tnfr.physics import _organized_parent_reservoir_memory as owner

_EDGES = tuple(
    sorted(
        {
            tuple(sorted((18 * g + 9 * a + j, 18 * g + 9 * a + (j + 1) % 9)))
            for g in range(2)
            for a in range(2)
            for j in range(9)
        }
        | {(18 * g + j, 18 * g + 9 + j) for g in range(2) for j in range(9)}
        | {(9 * a + j, 18 + 9 * a + j) for a in range(2) for j in range(9)}
    )
)
_SIGN = tuple(2 * (i // 9 % 2) - 1 for i in range(36))
_ZERO = (Q(0),) * 36
_FORMS = tuple(
    Q(2 * g - 1, 10000) + Q(2 * a - 1, 9000) + Q((g + a + 1) * (j - 4), 100000)
    for g in range(2)
    for a in range(2)
    for j in range(9)
)
_PHASES = tuple(
    Q(2 * g - 1, 60) + Q(2 * a - 1, 20) + Q((g + a + 1) * (3 * j * j - 68), 300000)
    for g in range(2)
    for a in range(2)
    for j in range(9)
)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _assess(**changes):
    return owner._assess_parent_reservoir_memory(
        **(
            dict(
                form=_ZERO,
                phase_offsets=_ZERO,
                reservoir_state=Q(0),
                leakage="zero",
            )
            | changes
        )
    )


def _laplacian(values):
    result = [0] * 36
    for left, right in _EDGES:
        difference = values[left] - values[right]
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def _current(mp, phases):
    result = [mp.mpf(0)] * 36
    for left, right in _EDGES:
        current = mp.sin(phases[right] - phases[left])
        result[left] += current
        result[right] -= current
    return tuple(result)


def _potential(mp, phases):
    return sum(1 - mp.cos(phases[j] - phases[i]) for i, j in _EDGES)


def _near_equilibrium(mp, retained, *, scale=Q(0), leakage="zero"):
    target = mp.findroot(lambda b: 6 * b + 3 * mp.sin(b) - retained, 0)
    gap = Q(mp.nstr(target, 110))
    offsets = tuple(
        sign * gap / 2 + scale * Q((i // 9 + 1) * (i % 9 - 4))
        for i, sign in enumerate(_SIGN)
    )
    reservoir = Q(retained) - sum(
        sign * value / 3 for sign, value in zip(_SIGN, offsets)
    )
    report = _assess(
        form=tuple(scale * (i % 9 - 4) for i in range(36)),
        phase_offsets=offsets,
        reservoir_state=reservoir,
        leakage=leakage,
    )
    return report, target


@pytest.fixture(scope="module")
def equilibrium_reports(mp):
    return {retained: _near_equilibrium(mp, retained) for retained in (-3, -1, 0, 1, 3)}


@pytest.mark.parametrize("leakage", ("zero", "original"))
@pytest.mark.parametrize("reservoir", (-Q(4, 5), Q(0), Q(4, 5)))
def test_complete_fine_rows_retained_derivative_and_balance_all_clocks(
    leakage, reservoir, mp
):
    report = _assess(
        form=_FORMS, phase_offsets=_PHASES, reservoir_state=reservoir, leakage=leakage
    )
    observation = report.observation
    gamma = 1 / (1023 * mp.pi)
    decay = gamma**2 / 2 if leakage == "original" else mp.mpf(0)
    x = tuple(_mp(mp, value) for value in _FORMS)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(_PHASES)
    )
    r = _mp(mp, reservoir)
    lap, current = _laplacian(x), _current(mp, theta)
    coupling = tuple(gamma * sign / 12 for sign in _SIGN)
    dx = tuple(
        (-q + gamma * sine) / 4 + b * r for q, sine, b in zip(lap, current, coupling)
    )
    dtheta = tuple(gamma * q / 4 for q in lap)
    dr = -sum(b * q for b, q in zip(coupling, lap)) - decay * r
    d_retained = dr + sum(sign * rate / 3 for sign, rate in zip(_SIGN, dtheta))
    assert abs(d_retained + decay * r) < mp.mpf("1e-175")
    assert report.retained_quantity == reservoir + sum(
        sign * value / 3 for sign, value in zip(_SIGN, _PHASES)
    )
    loss = sum(q**2 for q in lap) / 4 + decay * r**2
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("slow", 2 / gamma**2),
        ("original", mp.mpf(1023) / 1024),
    ):
        for i in range(36):
            _contains(
                mp, getattr(observation, f"form_{clock}_rate_bounds")[i], factor * dx[i]
            )
            _contains(
                mp,
                getattr(observation, f"phase_{clock}_rate_bounds")[i],
                factor * dtheta[i],
            )
        _contains(
            mp, getattr(observation, f"reservoir_{clock}_rate_bounds"), factor * dr
        )
        _contains(
            mp, getattr(report, f"retained_{clock}_rate_bounds"), -factor * decay * r
        )
        _contains(
            mp,
            getattr(observation, f"joint_storage_{clock}_rate_bounds"),
            -factor * loss,
        )
    _contains(mp, observation.reservoir_decay_bounds, decay)
    _contains(mp, observation.joint_storage_rate_from_rows_bounds, -loss)
    assert (
        observation.balance_residual_bounds.lo
        <= 0
        <= observation.balance_residual_bounds.hi
    )
    if reservoir == 0:
        # A zero instantaneous J derivative does not freeze the reservoir row.
        assert observation.reservoir_tau_rate_bounds.hi < 0
    if leakage == "zero":
        assert (
            report.retained_tau_rate_bounds.lo
            == report.retained_tau_rate_bounds.hi
            == 0
        )


def test_original_observer_default_is_unchanged():
    primitives = dict(form=_FORMS, phase_offsets=_PHASES, reservoir_state=Q(4, 5))
    old = exchange._observe_parent_reservoir_exchange(**primitives, port="connected")
    new = _assess(**primitives, leakage="original").observation
    assert new == old
    with pytest.raises(TypeError):
        exchange._observe_parent_reservoir_exchange(
            **primitives, port="connected", leakage="zero"
        )


@pytest.mark.parametrize("retained", (-3, -1, 0, 1, 3))
def test_root_full_equilibrium_rows_and_registration(retained, equilibrium_reports, mp):
    report, gap = equilibrium_reports[retained]
    assert report.retained_quantity == retained
    assert report.common_modes_admitted and report.chart_admitted
    assert report.retained_range_admitted and report.equilibrium_available
    _contains(mp, report.equilibrium_phase_gap_bounds, gap)
    _contains(mp, report.equilibrium_reservoir_bounds, 3 * mp.sin(gap))
    assert report.equilibrium_phase_gap_bounds.width < Q(1, 10**18)
    assert abs(gap) < mp.mpf("0.5")
    for sign, bounds in zip(_SIGN, report.equilibrium_phase_offsets_bounds):
        _contains(mp, bounds, sign * gap / 2)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + sign * gap / 2 for i, sign in enumerate(_SIGN)
    )
    r, gamma = 3 * mp.sin(gap), 1 / (1023 * mp.pi)
    current = _current(mp, theta)
    assert max(
        abs(gamma * sine / 4 + gamma * sign * r / 12)
        for sign, sine in zip(_SIGN, current)
    ) < mp.mpf("1e-175")
    _contains(mp, report.equilibrium_storage_bounds, _potential(mp, theta) + r**2 / 2)
    _contains(
        mp,
        report.registration_threshold_bounds,
        mp.mpf("1.5") + 3 * mp.sin(mp.mpf("0.25")),
    )
    assert report.equilibrium_registration == (
        "registered" if abs(gap) < mp.mpf("0.25") else "unregistered"
    )
    assert report.phase_ball_admitted and report.energy_barrier_admitted
    assert report.trap_certified and report.attraction_certified


def test_fixed_leaf_hessian_includes_arbitrary_fine_directions(equilibrium_reports, mp):
    # This direction varies inside every child and between both matching factors.
    direction = tuple(
        Q((i // 9 + 1) * (3 * (i % 9) ** 2 - 68), 100) + Q(_SIGN[i], 9)
        for i in range(36)
    )
    assert sum(direction) == 0
    y = tuple(_mp(mp, value) for value in direction)
    c_y = sum(sign * value / 3 for sign, value in zip(_SIGN, y))
    assert c_y != 0
    lap = np.zeros((36, 36))
    for i, j in _EDGES:
        lap[i, i] += 1
        lap[j, j] += 1
        lap[i, j] -= 1
        lap[j, i] -= 1
    assert np.linalg.eigvalsh(lap)[1] > 1 / 3
    for retained, (_, gap) in equilibrium_reports.items():
        theta = tuple(
            2 * mp.pi * (i % 9 - 4) / 9 + sign * gap / 2 for i, sign in enumerate(_SIGN)
        )
        expected = (
            sum(mp.cos(theta[j] - theta[i]) * (y[j] - y[i]) ** 2 for i, j in _EDGES)
            + c_y**2
        )

        def restricted_storage(t):
            phases = tuple(angle + t * value for angle, value in zip(theta, y))
            r = retained - sum(sign * angle / 3 for sign, angle in zip(_SIGN, phases))
            return _potential(mp, phases) + r**2 / 2

        assert abs(mp.diff(restricted_storage, 0)) < mp.mpf("1e-170")
        assert abs(mp.diff(restricted_storage, 0, 2) - expected) < mp.mpf("1e-170")
        assert expected > sum(value**2 for value in y) / 6
        # The common phase null is removed by the declared mean, not by c c^T.
        assert sum(_SIGN) == 0


@pytest.mark.parametrize("retained", (-3, 0, 3))
def test_transverse_local_trap_and_energy_floor_use_full_fine_state(retained, mp):
    report, gap = _near_equilibrium(mp, retained, scale=Q(1, 100000))
    observation = report.observation
    offsets = observation.network.phase_offsets
    phase_square = sum(
        (_mp(mp, value) - sign * gap / 2) ** 2 for sign, value in zip(_SIGN, offsets)
    )
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(offsets)
    )
    target_theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + sign * gap / 2 for i, sign in enumerate(_SIGN)
    )
    r = _mp(mp, observation.reservoir_state)
    energy = (
        _mp(mp, observation.network.form_storage)
        + _potential(mp, theta)
        + r**2 / 2
        - _potential(mp, target_theta)
        - (3 * mp.sin(gap)) ** 2 / 2
    )
    _contains(mp, report.phase_distance_squared_bounds, phase_square)
    _contains(mp, report.relative_energy_bounds, energy)
    assert energy > phase_square / 12
    assert report.phase_ball_admitted and report.energy_barrier_admitted
    assert report.trap_certified and report.attraction_certified


@pytest.mark.parametrize("leakage", ("zero", "original"))
@pytest.mark.parametrize(
    "scale,inside", ((Q(1, 49), True), (Q(1, 48), False), (Q(1, 47), False))
)
def test_phase_ball_boundary_is_strict_even_inside_acute_chart(leakage, scale, inside):
    offsets = tuple((2 * (i // 18) - 1) * scale for i in range(36))
    report = _assess(phase_offsets=offsets, leakage=leakage)
    assert report.chart_admitted and report.equilibrium_available
    assert report.retained_quantity == 0
    assert report.phase_ball_admitted is inside
    if not inside:
        assert not report.trap_certified and not report.attraction_certified


@pytest.mark.parametrize("leakage", ("zero", "original"))
@pytest.mark.parametrize("scale,inside", ((Q(1, 200), True), (Q(1, 150), False)))
def test_energy_barrier_remains_separate_from_phase_ball(leakage, scale, inside):
    form = tuple((2 * (i // 18) - 1) * scale for i in range(36))
    report = _assess(form=form, leakage=leakage)
    assert report.phase_ball_admitted and report.chart_admitted
    assert report.observation.network.form_storage == 36 * scale**2
    assert report.energy_barrier_admitted is inside
    assert report.trap_certified is inside
    assert report.attraction_certified is inside


@pytest.mark.parametrize("leakage", ("zero", "original"))
@pytest.mark.parametrize(
    "shift,inside", ((-Q(1, 10**12), True), (Q(0), False), (Q(1, 10**12), False))
)
def test_exact_joint_energy_boundary_is_not_a_trap(leakage, shift, inside, mp):
    # An alternating square in the K2 x K2 factors has exact Dirichlet cost12a².
    scale = Q(1, 96) + shift
    form = tuple(
        (2 * (i // 18) - 1) * _SIGN[i] * scale if i % 9 == 0 else Q(0)
        for i in range(36)
    )
    report = _assess(form=form, leakage=leakage)
    expected_energy = 12 * scale**2
    assert report.observation.network.form_storage == expected_energy
    _contains(mp, report.relative_energy_bounds, _mp(mp, expected_energy))
    assert report.phase_ball_admitted and report.chart_admitted
    assert report.energy_barrier_admitted is inside
    assert report.trap_certified is inside
    if shift == 0:
        assert expected_energy == report.trap_energy_barrier


def test_positive_leakage_has_its_own_target_and_requires_joint_storage(mp):
    state, _ = _near_equilibrium(mp, 3)
    primitives = dict(
        form=state.observation.network.form,
        phase_offsets=state.observation.network.phase_offsets,
        reservoir_state=state.observation.reservoir_state,
    )
    original = _assess(**primitives, leakage="original")
    assert state.trap_certified and state.equilibrium_registration == "unregistered"
    assert (
        original.equilibrium_phase_gap_bounds.lo
        == original.equilibrium_phase_gap_bounds.hi
        == 0
    )
    assert (
        original.equilibrium_reservoir_bounds.lo
        == original.equilibrium_reservoir_bounds.hi
        == 0
    )
    assert original.equilibrium_registration == "registered"
    assert not original.trap_certified and not original.attraction_certified
    near = _assess(reservoir_state=Q(1, 100), leakage="original")
    assert near.phase_ball_admitted and near.energy_barrier_admitted
    assert near.trap_certified and near.attraction_certified
    wide = _assess(reservoir_state=Q(4), leakage="original")
    assert wide.equilibrium_available and wide.retained_range_admitted
    assert wide.phase_ball_admitted and not wide.energy_barrier_admitted
    assert not wide.trap_certified


@pytest.mark.parametrize("retained", (-Q(3) - Q(1, 10**100), Q(3) + Q(1, 10**100)))
def test_zero_leakage_range_is_not_rounded_or_extended(retained):
    report = _assess(reservoir_state=retained)
    assert report.retained_quantity == retained
    assert not report.retained_range_admitted and not report.equilibrium_available
    assert report.equilibrium_phase_gap_bounds is None
    assert report.equilibrium_reservoir_bounds is None
    assert report.equilibrium_phase_offsets_bounds is None
    assert report.equilibrium_storage_bounds is None
    assert report.phase_distance_squared_bounds is None
    assert report.relative_energy_bounds is None
    assert report.equilibrium_registration == "unavailable"
    assert not report.trap_certified and not report.attraction_certified
    assert report.unresolved_conditions


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
def test_nonzero_common_mode_is_unavailable_without_recentering(field):
    values = [Q(0)] * 36
    values[0] = Q(1, 10**1000)
    report = _assess(**{field: values})
    assert not report.common_modes_admitted
    assert getattr(report.observation.network, field)[0] == values[0]
    assert not report.trap_certified and not report.attraction_certified
    assert report.unresolved_conditions


def test_lift_change_is_retained_instead_of_silently_wrapped():
    offsets = list(_ZERO)
    offsets[0], offsets[9] = Q(7), Q(-7)
    report = _assess(phase_offsets=offsets)
    assert report.common_modes_admitted
    assert report.retained_quantity == -Q(14, 3)
    assert report.observation.network.phase_offsets == tuple(offsets)
    assert not report.chart_admitted and not report.retained_range_admitted
    assert not report.trap_certified and not report.attraction_certified


def test_closing_child_edge_turn_is_part_of_chart_admission():
    offsets = list(_ZERO)
    offsets[0], offsets[8] = -Q(2, 5), Q(2, 5)
    report = _assess(phase_offsets=offsets)
    assert report.common_modes_admitted and report.retained_quantity == 0
    assert report.equilibrium_available and not report.chart_admitted
    margins = dict(report.chart_margins)
    assert margins["child_positive_gap"] < 0
    assert margins["child_acute_gap"] > 0
    assert margins["contact_acute_gap"] > 0
    assert report.state_parent_registration == ("unavailable", "unavailable")
    assert not report.trap_certified


def test_contact_chart_is_independent_of_child_winding():
    offsets = tuple((2 * (i // 18) - 1) * Q(4, 5) for i in range(36))
    report = _assess(phase_offsets=offsets)
    margins = dict(report.chart_margins)
    assert margins["child_positive_gap"] > 0
    assert margins["child_acute_gap"] > 0
    assert margins["contact_acute_gap"] < 0
    assert report.equilibrium_available and not report.chart_admitted
    assert not report.trap_certified


@pytest.mark.parametrize(
    "gap,registration",
    (
        (Q(1, 4) - Q(1, 10**100), "registered"),
        (Q(1, 4), "boundary"),
        (Q(1, 4) + Q(1, 10**100), "unregistered"),
    ),
)
def test_stored_parent_registration_has_an_exact_strict_boundary(gap, registration):
    report = _assess(phase_offsets=tuple(sign * gap / 2 for sign in _SIGN))
    assert report.chart_admitted
    assert report.state_parent_registration == (registration, registration)


@pytest.mark.parametrize("side,registration", ((-1, "registered"), (1, "unregistered")))
def test_equilibrium_registration_uses_root_not_reservoir_alone(side, registration, mp):
    threshold = mp.mpf("1.5") + 3 * mp.sin(mp.mpf("0.25"))
    retained = Q(mp.nstr(threshold, 100)) + side * Q(1, 10**12)
    report = _assess(reservoir_state=retained)
    assert report.state_parent_registration == ("registered", "registered")
    assert report.equilibrium_registration == registration
    assert not report.trap_certified


def test_overlap_near_irrational_registration_boundary_remains_unresolved():
    threshold = _assess().registration_threshold_bounds
    report = _assess(reservoir_state=threshold.midpoint)
    assert report.equilibrium_available
    assert report.equilibrium_registration == "unresolved"
    assert report.state_parent_registration == ("registered", "registered")
    assert not report.trap_certified


def test_exact_zero_excess_is_enclosed_without_clipping_signed_bounds():
    report = _assess()
    assert report.relative_energy_bounds.lo < 0 < report.relative_energy_bounds.hi
    assert report.trap_certified


def test_independent_reservoir_perturbation_changes_leaf_and_target():
    original, perturbed = _assess(), _assess(reservoir_state=Q(1, 100))
    assert original.retained_quantity == 0
    assert perturbed.retained_quantity == Q(1, 100)
    assert perturbed.equilibrium_phase_gap_bounds.lo > 0
    assert original.trap_certified and perturbed.trap_certified
    assert (
        original.observation.network.phase_offsets
        == perturbed.observation.network.phase_offsets
    )


def test_tiny_retained_information_is_preserved_before_interval_materialization(mp):
    tiny = Q(1, 10**1000)
    report = _assess(reservoir_state=tiny)
    assert report.retained_quantity == report.observation.reservoir_state == tiny
    assert report.observation.reservoir_storage == tiny**2 / 2
    # The outward enclosure may cross zero; the primitive and leaf never do.
    root = mp.findroot(
        lambda b: 6 * b + 3 * mp.sin(b) - _mp(mp, tiny), _mp(mp, tiny) / 9
    )
    _contains(mp, report.equilibrium_phase_gap_bounds, root)
    assert report.equilibrium_registration == "registered"


@pytest.mark.parametrize("field", ("form", "phase_offsets", "reservoir_state"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), complex(0)))
def test_all_primitives_are_admitted_before_shared_observation(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached shared observation")

    monkeypatch.setattr(owner, "_observe_admitted_parent_reservoir_exchange", forbidden)
    values = bad if field == "reservoir_state" else [Q(0)] * 35 + [bad]
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: values})


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", ((), (0,) * 35, (0,) * 37, "0" * 36, {0}))
def test_vector_shapes_reject_before_shared_observation(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid shape reached shared observation")

    monkeypatch.setattr(owner, "_observe_admitted_parent_reservoir_exchange", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("bad", (True, 0, None, "positive", "ZERO", "zero "))
def test_only_declared_leakage_laws_are_admitted(bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid law reached shared observation")

    monkeypatch.setattr(owner, "_observe_admitted_parent_reservoir_exchange", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(leakage=bad)


@pytest.mark.parametrize("bad", (True, "0", float("nan"), Q(3) + Q(1, 10**100)))
def test_root_helper_admits_its_own_exact_domain(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._equilibrium_gap_bounds(bad)


@pytest.mark.parametrize(
    "extra",
    (
        "source_report",
        "equilibrium_report",
        "phase_lifts",
        "coupling",
        "lambda",
        "duration",
        "trajectory",
        "capacity",
    ),
)
def test_reports_cannot_replace_state_or_extend_the_frozen_model(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_reports_are_immutable_and_scope_does_not_claim_source_entry(
    equilibrium_reports,
):
    report, _ = equilibrium_reports[0]
    with pytest.raises(FrozenInstanceError):
        report.retained_quantity = Q(1)
    assert report.protocol_revision == "53cc7b68"
    assert report.phase_ball_radius == Q(1, 8)
    assert report.trap_energy_barrier == Q(1, 768)
    assert not hasattr(report, "source_certified")
    assert not hasattr(report, "entry_certified")
