"""Independent field/domain controls; no paired trajectory is run here."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.physics import relational_transit as native
from tnfr.research import native_capacity_selection as owner


def _box(q=Q(3, 8), r=Q(-1, 8), s=Q(9, 8), t=Q(7, 8)):
    return tuple(map(I, (q, r, 0, 0, s, t)))


def _neighbors():
    rows = [set() for _ in range(10)]
    edges = [
        (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
    ] + [(0, 5), (1, 6)]
    for i, j in edges:
        rows[i].add(j)
        rows[j].add(i)
    return rows


def _narrow(interval):
    assert interval.width < Q(1, 1 << 90)


def test_interval_field_matches_independent_full_native_rows():
    # At (a,b)=(pi/3,pi/6), direct phasor addition gives port Arg=-pi/4,
    # H_port=2(1+sqrt(3)), and H_interior=pi*sqrt(3). This is not a lock.
    q, r, s, t = Q(3, 8), Q(-1, 8), Q(9, 8), Q(7, 8)
    pi, root_three = pi_interval(), sqrt(I(3))
    state = (I(q), I(r), pi / 3, pi / 6, I(s), I(t))
    actual = owner.flow(state, dynamic=True)
    held = owner.flow(state, dynamic=False)
    a_form, b_form = (2 * q + r) / 5, (q + 3 * r) / 5
    forms = (a_form, -a_form, -b_form, Q(0), b_form) * 2
    capacities = (s, s, t, 5 - 2 * s - 2 * t, t) * 2
    phase_source = (Q(-1, 4), Q(1, 4), Q(0), Q(0), Q(0)) * 2
    neighbors = _neighbors()
    gradient = tuple(
        sum((forms[i] - forms[j] for j in row), Q(0)) for i, row in enumerate(neighbors)
    )
    native_rates = tuple(
        capacities[i] * (-gradient[i] / len(row) + phase_source[i]) / 2
        for i, row in enumerate(neighbors)
    )
    capacity_rates = tuple(
        -sum((capacities[i] - capacities[j] for j in row), Q(0)) / 2
        for i, row in enumerate(neighbors)
    )
    assert actual[0].contains(3 * native_rates[0] - native_rates[4])
    assert actual[1].contains(2 * native_rates[4] - native_rates[0])
    assert (actual[2] * (4 * (1 + root_three))).contains(s * gradient[0])
    assert (actual[3] * (2 * pi * root_three)).contains(t * gradient[4])
    assert actual[4].contains(capacity_rates[0])
    assert actual[5].contains(capacity_rates[4])
    assert (-2 * actual[4] - 2 * actual[5]).contains(capacity_rates[3])
    assert actual[:4] == held[:4]
    assert held[4:] == (I(0), I(0))
    for value in actual:
        _narrow(value)


def test_jet_field_keeps_capacity_native_mixed_derivatives():
    # Independently at phase consensus: q'=-s*q/2+t*r/4,
    # r'=s*q/6-t*r/2, a'=s*q/(6*pi), b'=t*r/(4*pi).
    state = list(Jet.constant(value, 2) for value in _box())
    for index in (0, 4):
        state[index] = Jet((state[index].coeffs[0], I(1), I(0)))
    actual = owner.flow(state, dynamic=True)
    held = owner.flow(state, dynamic=False)
    assert all(value.order == 2 for value in actual)
    assert actual[0].coeffs[2].contains(Q(-1, 2))
    assert actual[1].coeffs[2].contains(Q(1, 6))
    assert (actual[2].coeffs[2] * pi_interval()).contains(Q(1, 6))
    assert actual[3].coeffs[2].contains(0)
    assert actual[4].coeffs[1].contains(Q(-1, 2))
    assert actual[5].coeffs[1].contains(Q(-1, 2))
    assert actual[:4] == held[:4]
    assert held[4:] == (Jet.constant(0, 2), Jet.constant(0, 2))
    for value in actual:
        for coefficient in value.coeffs:
            _narrow(coefficient)


def test_uniform_capacity_reduction_and_transport_balance():
    uniform = _box(s=Q(1), t=Q(1))
    base = native._flow(uniform[:4], Q(1, 2), Q(1, 2), Q(1))
    actual = owner.flow(uniform, dynamic=True)
    for value, reference in zip(actual[:4], base):
        assert value.contains(reference)
        _narrow(value)
    assert actual[4:] == (I(0), I(0))
    s, t = Q(9, 8), Q(15, 16)
    u = 5 - 2 * s - 2 * t
    released = owner.flow(_box(s=s, t=t), dynamic=True)
    ds, dt = released[4:]
    du = -2 * ds - 2 * dt
    assert ds.contains((t - s) / 2)
    assert dt.contains((s + u - 2 * t) / 2)
    assert du.contains(t - u)
    variance_rate = 2 * (s - 1) * ds + 2 * (t - 1) * dt + (u - 1) * du
    assert variance_rate.contains(-((s - t) ** 2 + (t - u) ** 2))
    assert variance_rate.hi < 0


def test_domain_rejects_hidden_capacity_and_phase_boundary():
    good = _box()
    assert min(owner.domain(good)) > 0
    bad_capacity = good[:4] + (I(1, Q(3, 2)), I(1))
    assert owner.domain(bad_capacity)[5] <= 0
    # The two active resultants stay positive; the omitted central one does not.
    bad_central = good[:2] + (I(2), I(Q(7, 4))) + good[4:]
    margins = owner.domain(bad_central)
    assert min(margins[:2]) > 0 and margins[2] < 0
    bad_sinc = good[:2] + (I(0), I(Q(-9, 8))) + good[4:]
    assert min(owner.domain(bad_sinc)[:-1]) > 0
    assert owner.domain(bad_sinc)[-1] < 0
    for bad in (bad_capacity, bad_central, bad_sinc):
        with pytest.raises(ValueError, match="domain"):
            owner.flow(bad, dynamic=True)
        assert owner.capture(bad)["sector"] is None
    with pytest.raises(ValueError, match="exactly"):
        owner.flow(good[:-1], dynamic=True)
    with pytest.raises(TypeError, match="Boolean"):
        owner.flow(good, dynamic=1)
    with pytest.raises(TypeError):
        owner.domain((0.5,) + good[1:])
    with pytest.raises(TypeError, match="all six"):
        owner.flow((Jet.constant(0, 1),) + good[1:], dynamic=True)
    mixed_orders = (Jet.constant(0, 1),) + tuple(
        Jet.constant(value, 2) for value in good[1:]
    )
    with pytest.raises(ValueError, match="same order"):
        owner.flow(mixed_orders, dynamic=True)


def test_capture_requires_whole_box_storage_and_geometry():
    pi = pi_interval()
    zero = _box(q=Q(0), r=Q(0))
    positive = zero[:2] + (4 * pi / 5, 2 * pi / 5) + zero[4:]
    negative = zero[:2] + (-4 * pi / 5, -2 * pi / 5) + zero[4:]
    for state, sector in ((zero, 0), (positive, 1), (negative, -1)):
        report = owner.capture(state)
        assert report["sector"] == sector
        assert report["storage"].hi < 7
        assert all(value.lo > 0 for value in report["margins"])
        assert min(report["domain"]) > 0
    uncertain_form = (I(0, 4),) + zero[1:]
    assert owner.capture((I(2),) + zero[1:])["sector"] == 0
    assert owner.capture(uncertain_form)["sector"] is None
    assert owner.capture(uncertain_form)["storage"].hi > 7
    boundary = zero[:2] + (I(2, Q(11, 5)), I(1)) + zero[4:]
    report = owner.capture(boundary)
    assert not any(
        all(margin.lo > 0 for margin in margins) for _, margins in report["candidates"]
    )
    assert report["sector"] is None


def test_step_delegates_once_and_retains_failed_evidence(monkeypatch):
    initial = _box()
    failed = tuple(value.hull(I(0)) for value in initial)
    retained = (None, failed, "declared enclosure failure")
    calls = []

    def unavailable(box, duration, field, domain, **kwargs):
        calls.append((box, duration, kwargs))
        assert field(box)[4:] == (I(0), I(0))
        assert domain(box) == owner.domain(initial)
        return retained

    monkeypatch.setattr(owner, "validated_taylor_step", unavailable)
    actual = owner.step(initial, dynamic=False, time=Q(3, 8))
    assert actual is retained
    assert calls == [
        (
            initial,
            Q(1, 8),
            {
                "order": 14,
                "time": Q(3, 8),
                "domain_failure": "whole_reflected_capacity_domain_not_admitted",
            },
        )
    ]
