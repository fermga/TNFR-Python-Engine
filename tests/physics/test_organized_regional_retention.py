"""Independent full-law regional balances and conditional first-exit budgets."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics import BEPIElement
from tnfr.physics import _organized_regional_retention as owner

_ERROR = Q(1, 10**10)
_ZERO = (0,) * 27


@pytest.fixture(scope="module", autouse=True)
def no_response_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, expected):
    assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)
    assert _mp(mp, bounds.width) < mp.mpf("1e-28")


def _support():
    return tuple(
        (9 * region + j, 9 * region + (j + 1) % 9, Q(1))
        for region in range(3)
        for j in range(9)
    ) + tuple(
        (9 * left + j, 9 * right + (j + offset) % 9, Q(1, 2))
        for left, right in ((0, 1), (1, 2), (2, 0))
        for j in range(9)
        for offset in (0, 1)
    )


def _assess(**changes):
    return owner._assess_organized_regional_retention(
        **(dict(form_error_bound=_ERROR, phase_error_bound=_ERROR) | changes)
    )


@pytest.fixture(scope="module")
def source_reports():
    missing = _assess()
    return {
        "missing": missing,
        "zero": _assess(donor_net_work_prefix_upper_bound=0),
        "boundary": _assess(
            donor_net_work_prefix_upper_bound=missing.net_work_reserves[0]
        ),
        "failed": _assess(phase_error_bound=2, donor_net_work_prefix_upper_bound=0),
    }


@pytest.fixture(scope="module")
def arbitrary_state():
    # Dyadic forms make the expected consumed state independent of rounding.
    form = tuple(Q((11 * i) % 17 - 8, 16) for i in range(27))
    phase = tuple(Q((7 * i) % 19 - 9, 5) for i in range(27))
    return form, phase, owner._observe_organized_regional_work(form=form, phase=phase)


def test_complete_rows_and_regional_contact_balances_match_edge_differentiation(
    arbitrary_state, mp
):
    form, phase, report = arbitrary_state
    x, theta = tuple(_mp(mp, value) for value in form), tuple(
        _mp(mp, value) for value in phase
    )
    edges = _support()
    gradient, current = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
    gamma = 1 / (1023 * mp.pi)
    for a, b, weight in edges:
        w = _mp(mp, weight)
        flow, sine = w * (x[a] - x[b]), w * mp.sin(theta[b] - theta[a])
        gradient[a] += flow
        gradient[b] -= flow
        current[a] += sine
        current[b] -= sine
    dx = tuple((-g + gamma * s) / 4 for g, s in zip(gradient, current))
    dtheta = tuple(gamma * g / 4 for g in gradient)
    assert report.form == form
    assert report.phase_lifts == phase
    assert report.geometry.weighted_edges == edges
    for i in range(27):
        assert _mp(mp, report.form_gradient[i]) == gradient[i]
        _contains(mp, report.phase_current_bounds[i], current[i])
        _contains(mp, report.form_rate_bounds[i], dx[i])
        _contains(mp, report.phase_rate_bounds[i], dtheta[i])

    storage_groups, rate_groups, form_groups, phase_groups = [], [], [], []
    for group in range(4):
        selected = tuple(
            (a, b, weight)
            for a, b, weight in edges
            if (a // 9 == b // 9 == group if group < 3 else a // 9 != b // 9)
        )
        local_gradient, local_current = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
        form_energy, phase_energy, rate = mp.mpf(0), mp.mpf(0), mp.mpf(0)
        for a, b, weight in selected:
            w = _mp(mp, weight)
            difference, angle = x[b] - x[a], theta[b] - theta[a]
            form_energy += w * difference**2 / 2
            phase_energy += w * (1 - mp.cos(angle))
            # Direct edge differentiation uses both full rows, independently
            # of the regional power cancellation used by the implementation.
            rate += w * (
                difference * (dx[b] - dx[a]) + mp.sin(angle) * (dtheta[b] - dtheta[a])
            )
            local_gradient[a] -= w * difference
            local_gradient[b] += w * difference
            local_current[a] += w * mp.sin(angle)
            local_current[b] -= w * mp.sin(angle)
        storage_groups.append(form_energy + phase_energy)
        rate_groups.append(rate)
        form_groups.append(form_energy)
        phase_groups.append(phase_energy)
        if group == 3:
            assert _mp(mp, report.contact_form_storage) == form_energy
            _contains(mp, report.contact_phase_storage_bounds, phase_energy)
            _contains(mp, report.contact_storage_bounds, storage_groups[-1])
            _contains(mp, report.contact_storage_rate_from_rows_bounds, rate)
            continue
        region = report.regions[group]
        loss = sum(value**2 for value in local_gradient) / 4
        assert _mp(mp, region.form_storage) == form_energy
        assert _mp(mp, region.dissipation) == loss
        _contains(mp, region.phase_storage_bounds, phase_energy)
        _contains(mp, region.storage_bounds, storage_groups[-1])
        _contains(mp, region.storage_rate_from_rows_bounds, rate)
        _contains(mp, region.storage_rate_bounds, rate)
        _contains(mp, region.boundary_power_bounds, rate + loss)
        assert region.balance_residual_bounds.contains(0)
        for local, node in enumerate(range(9 * group, 9 * (group + 1))):
            assert _mp(mp, region.form_gradient[local]) == local_gradient[node]
            assert _mp(mp, region.contact_form_gradient[local]) == (
                gradient[node] - local_gradient[node]
            )
            _contains(mp, region.internal_current_bounds[local], local_current[node])
            _contains(
                mp,
                region.contact_current_bounds[local],
                current[node] - local_current[node],
            )
    loss = sum(value**2 for value in gradient) / 4
    assert _mp(mp, report.dissipation) == loss
    assert _mp(mp, report.storage_rate) == -loss < 0
    assert abs(sum(rate_groups) + loss) < mp.mpf("1e-90")
    assert _mp(mp, report.form_storage) == sum(form_groups)
    _contains(mp, report.phase_storage_bounds, sum(phase_groups))
    _contains(mp, report.storage_bounds, sum(storage_groups))
    _contains(mp, report.storage_rate_from_rows_bounds, -loss)
    for residual in (
        report.balance_residual_bounds,
        report.storage_partition_residual_bounds,
        report.rate_partition_residual_bounds,
    ):
        assert residual.contains(0)
    assert not report.accumulated_work_observed


def test_global_loss_does_not_make_each_regional_store_decrease(mp):
    form = tuple(scale * (j - 4) for scale in (1, 8, 8) for j in range(9))
    report = owner._observe_organized_regional_work(form=form, phase=_ZERO)
    region = report.regions[0]
    assert region.dissipation == Q(81, 2)
    _contains(mp, region.boundary_power_bounds, mp.mpf(90))
    _contains(mp, region.storage_rate_bounds, mp.mpf(99) / 2)
    assert region.storage_rate_bounds.lo > 0
    assert report.storage_rate < 0


@pytest.mark.parametrize("sign", (-1, 1))
def test_same_phase_has_opposite_phase_normal_rates_from_retained_form(sign, mp):
    kappa = Q(1, 8)
    form = [Q(0)] * 27
    form[0], form[1] = -sign * kappa, sign * kappa
    phase = tuple(Q(i % 9, 4) for i in range(27))
    report = owner._observe_organized_regional_work(form=form, phase=phase)
    normal = report.phase_rate_bounds[1] - report.phase_rate_bounds[0]
    # b=e1-e0: degree(0)+degree(1)+2*weight(01)=10, so b^T Lb=10.
    expected = sign * 5 * _mp(mp, kappa) / (2 * 1023 * mp.pi)
    _contains(mp, normal, expected)
    assert normal.lo > 0 if sign > 0 else normal.hi < 0


def test_release_reserves_bound_independent_source_and_cosine_calculation(
    source_reports, mp
):
    report = source_reports["missing"]
    gamma, norm = 1 / (1023 * mp.pi), mp.sqrt(27)
    ramp = mp.mpf(2046) / 9 * (mp.mpf(355) / 113) ** 2
    error = _mp(mp, _ERROR)
    initial = gamma * (ramp * mp.sqrt(120) + error * norm)
    scaled = initial * mp.exp(-40) + 10 * gamma**2 * norm
    form = scaled / gamma
    phase = (
        mp.sqrt(120) * abs(gamma * ramp - 2 * mp.pi / 9)
        + norm * error * (1 + gamma)
        + scaled
        + 400 * gamma**2 * norm
    )
    correction = 2 * (form**2 + phase**2)
    donor = 9 * (1 - mp.cos(2 * mp.pi / 9)) + correction
    assert report.first_exit_barrier == Q(9, 4)
    assert report.release_windings == (1, 1, 0)
    assert report.source_admitted
    for upper, reserve, expected in zip(
        report.release_storage_upper_bounds,
        report.net_work_reserves,
        (donor, donor, correction),
    ):
        assert expected <= _mp(mp, upper) < expected + mp.mpf("1e-25")
        assert reserve == Q(9, 4) - upper
        assert 0 < _mp(mp, reserve) <= mp.mpf(9) / 4 - expected
    assert Q(14, 100) < report.net_work_reserves[0] < Q(15, 100)
    assert Q(224, 100) < report.net_work_reserves[2] < Q(225, 100)


def test_absent_and_supplied_budgets_never_become_observed_retention(source_reports):
    missing, zero, boundary = (
        source_reports[name] for name in ("missing", "zero", "boundary")
    )
    assert missing.donor_net_work_prefix_upper_bound is None
    assert missing.donor_budget_margin is None
    assert missing.donor_budget_inequality_passes is None
    assert missing.status == "unavailable"
    assert zero.donor_budget_margin > 0
    assert zero.donor_budget_inequality_passes is True
    assert zero.status == "conditional_donor_budget_passes"
    assert boundary.donor_budget_margin == 0
    assert boundary.donor_budget_inequality_passes is False
    assert boundary.status == "unavailable"
    for report in source_reports.values():
        assert not report.accumulated_work_observed
        assert not report.donor_retention_measured
        assert report.receiver_acquisition == "unresolved"
    failed = source_reports["failed"]
    assert not failed.source_admitted
    assert failed.release_windings is None
    assert failed.donor_budget_inequality_passes is None
    assert failed.status == "unavailable"


@pytest.mark.parametrize(
    "key",
    ("form_error_bound", "phase_error_bound", "donor_net_work_prefix_upper_bound"),
)
@pytest.mark.parametrize(
    "value", (True, float("nan"), float("inf"), -Q(1, 10**80), "0")
)
def test_invalid_budget_primitives_reject_before_source(key, value, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached source reconstruction")

    monkeypatch.setattr(owner, "_assess_organized_receiver_slow_admission", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{key: value})


@pytest.mark.parametrize("coordinate", ("form", "phase"))
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), "0", object()))
def test_invalid_state_rejects_before_geometry(coordinate, value, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached support construction")

    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    inputs = dict(form=_ZERO, phase=_ZERO)
    inputs[coordinate] = (*_ZERO[:26], value)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_regional_work(**inputs)


@pytest.mark.parametrize("coordinate", ("form", "phase"))
@pytest.mark.parametrize("values", ((), (0,) * 26, (0,) * 28, "0" * 27, {0}))
def test_invalid_state_shape_rejects_before_geometry(coordinate, values, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid state shape reached support construction")

    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_regional_work(
            **(dict(form=_ZERO, phase=_ZERO) | {coordinate: values})
        )


def test_uniform_signed_bepi_and_consumed_rational_representation():
    uniform = BEPIElement((-2.0, -2.0), (-2.0, -2.0), (0.0, 1.0))
    form = (uniform, Q(1, 3), *_ZERO[2:])
    phase = (Q(1, 3), *_ZERO[1:])
    report = owner._observe_organized_regional_work(form=form, phase=phase)
    assert report.form[0] == -2
    assert report.form[1] == Q.from_float(float(Q(1, 3))) != Q(1, 3)
    assert report.phase_lifts[0] == Q(1, 3)
    expected_gradient = Q(-8) - report.form[1]
    assert report.form_gradient[0] == expected_gradient
    with pytest.raises(FrozenInstanceError):
        report.form = _ZERO


@pytest.mark.parametrize(
    "value",
    (
        BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0)),
        BEPIElement((1j, 1j), (1j, 1j), (0.0, 1.0)),
        Q(1, 10**1000),
    ),
)
def test_rich_form_or_lost_nonzero_rejects_before_geometry(value, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid signed form reached support construction")

    monkeypatch.setattr(owner, "_derive_organized_triangle", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_regional_work(form=(value, *_ZERO[1:]), phase=_ZERO)
