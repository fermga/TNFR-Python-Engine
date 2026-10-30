"""Independent finite waiting controls for the actual common UM source.

Static points verify complete laws, not reached endpoints. The waiting bound
comes from a uniform metric and integer inequalities, with no settling trace.
The source association and elapsed model time remain conditional premises.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_circulation_readiness as owner
from tnfr.physics import _sine_coupling_successor as source_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
RADIUS = Q(1, 32)
SCALE = Q(6, 5)
AUXILIARY = Q(4, 9)
GAMMA_LO, GAMMA_HI = Q(1, 2750), Q(1, 2500)
CURVATURE = Q(5, 72)
WAIT = Q(10**10)
HANDOFF = Q(1, 2**28)
JUMP_ERROR = Q(1, 10**8)
NOISE = Q(1, 10**6)


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert interval.width < Q(1, 10**30)


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support():
    bins = tuple(int((9 * value) % 9) for value in _turns())
    pairs = {
        (i, j)
        for i in range(18)
        for j in range(i + 1, 18)
        if min((bins[j] - bins[i]) % 9, (bins[i] - bins[j]) % 9) in (1, 2)
        or (i // 9 != j // 9 and bins[i] == bins[j])
    }
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in (0, 1)
            for j in range(9)
        )
    )
    return old, tuple(sorted(pairs - set(old)))


def _weights():
    return tuple(Q(3, 4) + Q(i % 5, 16) for i in range(63))


def _edges():
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, _weights())
    )


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _laplacian(edges):
    rows = [[0] * 18 for _ in range(18)]
    for a, b, weight in edges:
        rows[a][a] += weight
        rows[b][b] += weight
        rows[a][b] -= weight
        rows[b][a] -= weight
    return tuple(tuple(row) for row in rows)


def _product(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(18)) for j in range(18))
        for i in range(18)
    )


def _state(forms, errors, gauge=Q(0)):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + error + gauge
        for turn, error in zip(_turns(), errors)
    )


def _oracle(mp, forms, errors, gauge=Q(0)):
    x = tuple(_mp(mp, value) for value in forms)
    y = tuple(_mp(mp, value) for value in errors)
    target = tuple(2 * mp.pi * _mp(mp, turn) for turn in _turns())
    phase = tuple(a + b + _mp(mp, gauge) for a, b in zip(target, y))
    gradient, sine = [mp.mpf(0)] * 18, [mp.mpf(0)] * 18
    secants = []
    for a, b, weight in _edges():
        q = _mp(mp, weight) * (x[a] - x[b])
        current = mp.sin(phase[b] - phase[a])
        gradient[a] += q
        gradient[b] -= q
        sine[a] += current
        sine[b] -= current
        displacement = y[b] - y[a]
        base = target[b] - target[a]
        coefficient = (
            (mp.sin(base + displacement) - mp.sin(base)) / displacement
            if displacement
            else mp.cos(base)
        )
        secants.append((a, b, coefficient))
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-q + gamma * current) / 9 for q, current in zip(gradient, sine))
    dy = tuple(gamma * value / 9 for value in gradient)
    return x, y, tuple(gradient), tuple(sine), dx, dy, _laplacian(secants)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("waiting controls cannot advance or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


def _assess(**changes):
    arguments = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        phase_jump_error_bound=JUMP_ERROR,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_circulation_readiness(**arguments)


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def field():
    return source_owner._coupling_successor_field(
        completion="common_sense",
        new_edge_conductances=_weights(),
        loss_normalization="bare_degree",
    )


def test_original_36_rows_scaled_law_and_original_clock_are_all_retained(mp, field):
    forms = _center(tuple(Q((i % 7) - 3, 8192) for i in range(18)))
    errors = _center(tuple(Q((i % 5) - 2, 16384) for i in range(18)))
    x, y, q, sine, dx, dy, stiffness = _oracle(mp, forms, errors)
    rates = field(_state(forms, errors))
    for interval, value in zip(rates, dx + dy):
        _contains(mp, interval, value)
    gamma, scale, auxiliary = 1 / (1023 * mp.pi), _mp(mp, SCALE), _mp(mp, AUXILIARY)
    b = tuple(scale * value for value in y)
    kb = tuple(value / scale for value in _matvec(stiffness, y))
    scaled_x = tuple(
        -auxiliary * (hi + scale * gamma * ki) / 4 for hi, ki in zip(q, kb)
    )
    scaled_b = tuple(scale * gamma * auxiliary * hi / 4 for hi in q)
    assert max(abs(a - c) for a, c in zip(scaled_x, dx)) < mp.mpf("1e-190")
    assert max(abs(a - scale * c) for a, c in zip(scaled_b, dy)) < mp.mpf("1e-190")
    assert _dot(b, b) > _dot(y, y)
    clock = _mp(mp, Q(1023, 1024))
    original_x = tuple(
        -clock * hi / 9 + current / (1024 * mp.pi * 9) for hi, current in zip(q, sine)
    )
    original_y = tuple(hi / (1024 * mp.pi * 9) for hi in q)
    assert max(
        abs(a / clock - c) for a, c in zip(original_x + original_y, dx + dy)
    ) < mp.mpf("1e-190")
    assert abs(sum(dx)) < mp.mpf("1e-190") and abs(sum(dy)) < mp.mpf("1e-190")
    assert max(abs(value) for value in dy) > 0
    # Unknown common phase is a conserved source gauge, not a fitted zero.
    shifted = field(_state(forms, errors, gauge=Q(7, 16)))
    for interval, value in zip(shifted, dx + dy):
        _contains(mp, interval, value)


def test_integrated_bare_hessian_and_weighted_transport_need_not_commute(mp):
    errors = _center(tuple(Q((i % 9) ** 2, 65536) for i in range(18)))
    forms = tuple(value / 2 for value in errors)
    x, y, _, sine, _, _, k = _oracle(mp, forms, errors)
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges()))
    ky = _matvec(k, y)
    assert max(abs(a + b) for a, b in zip(ky, sine)) < mp.mpf("1e-190")
    hk, kh = _product(h, k), _product(k, h)
    assert max(
        abs(a - b) for left, right in zip(hk, kh) for a, b in zip(left, right)
    ) > mp.mpf("1e-6")
    assert abs(_dot(x, y)) > mp.mpf("1e-12")
    # Edge-square positivity proves the Loewner comparison for all vectors,
    # without requiring either matrix to share eigenvectors with the other.
    for a, b, weight in _edges():
        coefficient = -k[a][b]
        assert mp.mpf(1) / 10 < coefficient <= 1
        assert _mp(mp, Q(3, 4)) <= _mp(mp, weight) <= 1
        scaled = coefficient / _mp(mp, SCALE**2)
        assert _mp(mp, CURVATURE * weight) <= scaled <= _mp(mp, Q(25, 27) * weight)
    assert Q(1, 10) / SCALE**2 == CURVATURE
    assert Q(4, 3) / SCALE**2 == Q(25, 27) < 1
    assert SCALE / 3300 == GAMMA_LO and SCALE / 3000 == GAMMA_HI


def test_metric_derivative_keeps_signed_cross_term_for_noncommuting_matrices(mp):
    errors = _center(tuple(Q((i % 9) ** 2, 65536) for i in range(18)))
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges()))
    for sign in (-1, 1):
        forms = tuple(sign * value / 2 for value in errors)
        x, y, hx, sine, dx, dy, k = _oracle(mp, forms, errors)
        assert _dot(forms, forms) + _dot(errors, errors) < RADIUS**2
        gamma, scale = 1 / (1023 * mp.pi), _mp(mp, SCALE)
        b, db, alpha = (
            tuple(scale * value for value in y),
            tuple(scale * value for value in dy),
            scale * gamma,
        )
        square = _dot(x, x) + _dot(b, b)
        mixed = 2 * alpha * _dot(x, b)
        metric = mp.mpf(9) / 4 * (square + mixed)
        assert sign * mixed > 0
        derivative = (
            mp.mpf(9)
            / 2
            * sum(
                a * da + c * dc + alpha * (da * c + a * dc)
                for a, c, da, dc in zip(x, b, dx, db)
            )
        )
        ky = _matvec(k, y)
        identity = (
            -(1 - scale**2 * gamma**2) * _dot(x, hx) / 2
            - gamma * _dot(x, ky) / 2
            - scale**2 * gamma**2 * _dot(y, ky) / 2
        )
        assert abs(derivative - identity) < mp.mpf("1e-190")
        assert max(abs(a + b) for a, b in zip(ky, sine)) < mp.mpf("1e-190")
        low, high = Q(9, 4) * (1 - GAMMA_HI), Q(9, 4) * (1 + GAMMA_HI)
        assert _mp(mp, low) * square < metric < _mp(mp, high) * square
        rate = Q(5) / (216 * 2750**2 * (1 + GAMMA_HI))
        assert derivative < -_mp(mp, rate) * metric
        # H is the actual weighted form matrix; replacing it by the bare
        # phase Hessian would change the independently evaluated rows.
        assert max(abs(a - c) for a, c in zip(_matvec(h, x), _matvec(k, x))) > mp.mpf(
            "1e-7"
        )


def test_initial_image_upper_bound_includes_fine_errors_and_both_cross_signs(report):
    phase = Q(9, 2) * DELTA**2 + 54 * DELTA * ETA + 162 * ETA**2
    initial = phase + 2 * ETA**2
    assert initial - Q(9, 2) * DELTA**2 == 54 * DELTA * ETA + 164 * ETA**2 > 0
    assert initial < 5 * DELTA**2
    low, high = Q(9, 4) * (1 - GAMMA_HI), Q(9, 4) * (1 + GAMMA_HI)
    assert report.initial_norm_squared_upper_bounds == (initial, initial)
    for source in report.sources:
        common = source.completion_reports[0]
        assert source.precontact_reference.initial_form_means == (Q(0), Q(0))
        assert common.degree_form_limit_bounds == ((Q(0), Q(0)),)
        assert common.initial_phase_norm_squared_upper_bound == phase
        assert common.initial_quotient_radius_margin == RADIUS**2 - initial > 0
        assert common.initial_energy_upper_bound < common.storage_barrier == Q(1, 5120)
    for sign in (-1, 1):
        y = (-DELTA / 2 - ETA / 2,) * 9 + (DELTA / 2 + ETA / 2,) * 9
        x = (-sign * ETA / 3,) * 9 + (sign * ETA / 3,) * 9
        assert sum(x) == sum(y) == 0
        assert _dot(x, x) == 2 * ETA**2
        assert _dot(y, y) < phase
        square = _dot(x, x) + SCALE**2 * _dot(y, y)
        assert square <= SCALE**2 * initial
        for gamma in (Q(1, 3300), Q(1, 3000)):
            mixed = 2 * SCALE**2 * gamma * _dot(x, y)
            assert sign * mixed > 0
            metric = Q(9, 4) * (square + mixed)
            assert low * square <= metric <= high * square
            assert metric < high * SCALE**2 * initial
    assert report.initial_scaled_norm_squared_upper_bounds == (SCALE**2 * initial,) * 2
    assert report.initial_metric_upper_bounds == (high * SCALE**2 * initial,) * 2
    assert report.physical_norm_prefactors == (high / low * SCALE**2 * initial,) * 2
    # Exact source envelopes retain sub-grid eta terms and the unknown signed
    # mixed product. These convenient points do not authenticate endpoints.


def test_shared_kernel_receives_only_auxiliary_capacity_and_fixed_clock(monkeypatch):
    original = owner._sine_common_quadratic_coefficients
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(owner, "_sine_common_quadratic_coefficients", capture)
    report = _assess()
    assert calls == [
        dict(
            gamma_lower=GAMMA_LO,
            gamma_upper=GAMMA_HI,
            cosine_lower=CURVATURE,
            gap_lower=Q(3),
            capacity_lower=AUXILIARY,
            capacity_upper=AUXILIARY,
        )
    ]
    assert report.metric_lower_bound == (1 - GAMMA_HI) / AUXILIARY
    assert report.metric_upper_bound == (1 + GAMMA_HI) / AUXILIARY
    assert report.metric_decay_rate == Q(5) / (216 * 2750**2 * (1 + GAMMA_HI))
    assert report.phase_scale == SCALE
    assert report.scaled_stiffness_upper_ratio == Q(25, 27)
    assert (
        report.physical_capacities
        == report.source_geometry.held_capacities
        == (Q(1),) * 18
    )
    assert report.source_geometry.degree == 9
    assert report.original_clock_waiting_duration == Q(1024, 1023) * WAIT
    assert report.original_clock_waiting_duration * Q(1023, 1024) == WAIT


def test_integer_decay_chain_gives_positive_maturity_for_every_later_age(
    report, record_property
):
    # This independent rational certificate evaluates neither exp nor sqrt.
    rate = Q(5) / (216 * 2750**2 * (1 + GAMMA_HI))
    assert rate > Q(3, 10**9)
    assert report.exponent_lower_bound == rate * WAIT > 30
    assert report.exponent_margin == rate * WAIT - 30 > 0
    assert Q(1) + 1 + Q(1, 2) + Q(1, 6) > Q(5, 2)
    assert 5**10 > 2**23
    assert Q(5, 2) ** 30 > 2**39
    assert report.decay_upper_bound == Q(1, 2**39)
    assert Q(2501, 2499) * SCALE**2 * 5 < 8
    assert 8 * DELTA**2 == Q(1, 2**17)
    assert Q(1, 2**17) * Q(1, 2**39) == Q(1, 2**56) == HANDOFF**2
    assert 0 < HANDOFF < Q(1, 10**8)
    for initial, prefactor, returned, margin in zip(
        report.initial_norm_squared_upper_bounds,
        report.physical_norm_prefactors,
        report.returned_norm_squared_upper_bounds,
        report.readiness_squared_margins,
    ):
        assert prefactor == Q(2501, 2499) * SCALE**2 * initial < Q(1, 2**17)
        assert returned == prefactor * Q(1, 2**39) < HANDOFF**2
        assert margin == HANDOFF**2 - returned > 0
    assert report.readiness_certified == (True, True)
    assert report.later_isolated_age_guarantee_certified
    assert report.all_time_source_identity_certified == (True, True)
    # For any additional isolated duration, W is nonincreasing in the
    # already trapped chart. There is no new endpoint or finite exact zero.
    record_property("uniform_structural_wait", str(WAIT))
    record_property(
        "same_wait_original_clock", str(report.original_clock_waiting_duration)
    )
    record_property("exact_decay_upper_bound", str(report.decay_upper_bound))
    record_property(
        "returned_norm_squared_upper_bound",
        str(report.returned_norm_squared_upper_bounds[0]),
    )
    record_property("derived_positive_maturity", str(HANDOFF))


def test_handoff_is_derived_automatically_and_retains_the_closed_response(
    monkeypatch, report
):
    original = owner._assess_mixed_two_channel_circulation
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(owner, "_assess_mixed_two_channel_circulation", capture)
    composed = _assess()
    assert calls == [
        dict(
            form_error_bounds=(ERROR, ERROR),
            phase_error_bounds=(ERROR, ERROR),
            mature_norm_upper_bounds=(HANDOFF, HANDOFF),
            phase_jump_error_bound=JUMP_ERROR,
            readout_error_bound=NOISE,
        )
    ]
    assert composed == report
    downstream = report.circulation
    assert (
        report.handoff_norm_upper_bounds
        == downstream.mature_norm_upper_bounds
        == (HANDOFF, HANDOFF)
    )
    assert downstream.charge_amplitude == 0
    assert downstream.charge_duration == downstream.prefix_leak_duration == Q(1, 32)
    assert downstream.phase_jump_amplitude == Q(1, 8192)
    assert downstream.contact_duration == 512
    assert downstream.readout_error_bound == downstream.readout_error_cap == NOISE
    assert downstream.classifier_threshold == Q(1, 400000)
    assert downstream.contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert downstream.independently_observed_channels == 1
    assert downstream.record_separation_margin >= Q(1503, 400000000)
    assert report.conservative_record_separation_floor == Q(1503, 400000000)
    assert (
        report.handoff_association_certified
        and report.composed_response_certified
        and report.certified
    )
    assert not report.caller_maturity_required
    assert not report.runtime_source_authenticated
    assert not report.attained_waiting_age_authenticated
    assert (
        not report.source_trajectory_evaluated
        and not report.minimum_waiting_time_claimed
    )


def test_fresh_common_source_mean_curvature_gap_and_initial_inclusion_are_required(
    monkeypatch, report
):
    first, second = report.sources

    def install(left, right):
        rows = iter((left, right))
        monkeypatch.setattr(
            owner, "_assess_sine_coupling_successor", lambda **kwargs: next(rows)
        )

    for side, source in enumerate(report.sources):
        common = source.completion_reports[0]
        for changed in (
            replace(source, conditional_image_certified=False),
            replace(source, phase_curvature_bounds=I(0)),
            replace(source, new_weight_range=(Q(1, 2), Q(1))),
            replace(source, form_error_bound=Q(0)),
            replace(
                source,
                completion_reports=(
                    replace(common, degree_form_limit_bounds=((Q(1), Q(1)),)),
                )
                + source.completion_reports[1:],
            ),
            replace(
                source,
                completion_reports=(
                    replace(common, initial_quotient_norm_squared_upper_bound=Q(0)),
                )
                + source.completion_reports[1:],
            ),
            replace(
                source,
                completion_reports=(
                    replace(common, initial_quotient_radius_margin=Q(0)),
                )
                + source.completion_reports[1:],
            ),
            replace(
                source,
                completion_reports=(replace(common, storage_margin=Q(0)),)
                + source.completion_reports[1:],
            ),
        ):
            install(changed if side == 0 else first, changed if side == 1 else second)
            unavailable = _assess()
            assert not unavailable.readiness_certified[side]
            assert (
                unavailable.handoff_norm_upper_bounds is None
                and unavailable.circulation is None
            )
            assert not unavailable.certified
    install(first, second)
    original = owner._derive_coupling_successor_geometry

    def corrupt(**kwargs):
        return replace(original(**kwargs), weighted_gap_lower_bound=Q(2))

    monkeypatch.setattr(owner, "_derive_coupling_successor_geometry", corrupt)
    unavailable = _assess()
    assert unavailable.source_certified == (False, False)
    assert unavailable.handoff_norm_upper_bounds is None


def test_cached_success_neither_substitutes_for_source_nor_changes_handoff(
    monkeypatch, report
):
    # The old strength-loss contrast/top-level verdict is not a premise of
    # this new common degree-loss metric. Its consumed source bounds are.
    rows = iter(
        tuple(
            replace(
                source,
                certified=False,
                normalization_common_form_separation_certified=False,
            )
            for source in report.sources
        )
    )
    monkeypatch.setattr(
        owner, "_assess_sine_coupling_successor", lambda **kwargs: next(rows)
    )
    assert _assess().certified
    monkeypatch.undo()
    downstream = report.circulation
    for altered in (
        replace(downstream, mature_norm_upper_bounds=(Q(1, 10**8),) * 2),
        replace(downstream, contact_duration=Q(513)),
        replace(downstream, classifier_threshold=Q(0)),
        replace(downstream, record_separation_margin=Q(0)),
        replace(downstream, readout_error_bound=Q(0)),
    ):
        monkeypatch.setattr(
            owner, "_assess_mixed_two_channel_circulation", lambda **kwargs: altered
        )
        unavailable = _assess()
        assert unavailable.readiness_certified == (True, True)
        assert not unavailable.composed_response_certified and not unavailable.certified
    for side, name in (
        (0, "edge_weight_deviation_upper_bound"),
        (1, "gauge_phase_deviation_upper_bound"),
    ):
        source = downstream.sources[side]
        changed = replace(source, **{name: 2 * getattr(source, name)})
        altered = replace(
            downstream,
            sources=tuple(
                changed if i == side else downstream.sources[i] for i in range(2)
            ),
        )
        assert altered.certified and all(value.certified for value in altered.sources)
        monkeypatch.setattr(
            owner, "_assess_mixed_two_channel_circulation", lambda **kwargs: altered
        )
        unavailable = _assess()
        assert unavailable.readiness_certified == (True, True)
        assert not unavailable.handoff_association_certified
        assert not unavailable.composed_response_certified and not unavailable.certified


def test_waiting_and_downstream_actuator_observation_admission_remain_separate():
    for side in (0, 1):
        for name in ("form_error_bounds", "phase_error_bounds"):
            outside = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            unavailable = _assess(**{name: outside})
            assert (
                not unavailable.source_certified[side]
                and not unavailable.readiness_certified[side]
            )
            assert unavailable.circulation is None and not unavailable.certified
    for name, value in (
        ("phase_jump_error_bound", JUMP_ERROR + Q(1, 10**100)),
        ("readout_error_bound", NOISE + Q(1, 10**100)),
    ):
        unavailable = _assess(**{name: value})
        assert unavailable.readiness_certified == (True, True)
        assert unavailable.handoff_norm_upper_bounds == (HANDOFF, HANDOFF)
        assert (
            unavailable.circulation is not None
            and not unavailable.composed_response_certified
        )
        assert not unavailable.certified
    assert _assess(phase_jump_error_bound=0, readout_error_bound=0).certified


def test_invalid_primitives_precede_sources_geometry_metric_and_handoff(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed primitive reached source or waiting arithmetic")

    for name in (
        "_assess_sine_coupling_successor",
        "_derive_coupling_successor_geometry",
        "_sine_common_quadratic_coefficients",
        "_assess_mixed_two_channel_circulation",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for value in (
        True,
        np.bool_(False),
        "0",
        float("nan"),
        float("inf"),
        complex(0, 1),
        None,
    ):
        for side in (0, 1):
            pair = tuple(value if i == side else Q(0) for i in range(2))
            for name in ("form_error_bounds", "phase_error_bounds"):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        for name in ("phase_jump_error_bound", "readout_error_bound"):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: value})
    for name in ("form_error_bounds", "phase_error_bounds"):
        for values in (
            (0,),
            (0, 0, 0),
            repeat(Q(0)),
            {0, 1},
            {0: 0, 1: 0},
            (Q(0), -Q(1, 10**100)),
        ):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: values})
    for name in ("phase_jump_error_bound", "readout_error_bound"):
        with pytest.raises(ValueError):
            _assess(**{name: -Q(1, 10**100)})


def test_one_shot_fixed_width_and_no_caller_radius_age_or_endpoint(report):
    def fixed(value):
        return Q(np.int64(Q(value).numerator), np.int64(Q(value).denominator))

    normalized = _assess(
        form_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_jump_error_bound=fixed(JUMP_ERROR),
        readout_error_bound=fixed(NOISE),
    )
    assert normalized == report
    assert report.status == "conditional_finite_readiness_and_circulation"
    with pytest.raises(FrozenInstanceError):
        report.attained_waiting_age_authenticated = True
    for name in (
        "sources",
        "initial_state",
        "mature_norm_upper_bounds",
        "starting_gap_upper_bounds",
        "waiting_duration",
        "elapsed_time",
        "source_age",
        "actual_transfer",
        "circulation",
        "phase_scale",
        "capacity",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
