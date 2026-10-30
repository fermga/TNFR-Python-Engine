"""A changed port can close one aggregate without closing the complete state.

Independent static edge sums test actual unequal weights, complete phase rows,
storage and the information retained by the boundary protocol. Constructed
states are algebraic witnesses, not newly acquired source endpoints. No
exponential, source replay, numerical root or trajectory is evaluated.
"""

import json
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
from tnfr.physics import _mixed_renewed_interface as owner
from tnfr.physics import _mixed_two_channel_circulation as field_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**9)
THRESHOLD = Q(1, 10**8)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
MATURE = Q(1, 2**28)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
QLO, QHI = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
G = Q(1, 3000)
SWITCH_AGE = Q(3375000)
HORIZON = Q(8)
DECAY = Q(1, 2**16)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    old = tuple(
        sorted(
            tuple(sorted((9 * p + j, 9 * p + (j + 1) % 9)))
            for p in (0, 1)
            for j in range(9)
        )
    )
    edges = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return old, tuple(sorted(edges - set(old)))


def _nominal_weights(mp):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if j < 9 else 1) * _mp(mp, DELTA) / 2
        for j, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _weights(mp, side):
    values = tuple(
        Q(mp.nstr(value, 210))
        + (1 if value < _mp(mp, Q(7, 8)) else -1)
        * WEIGHT_ERROR
        * Q(1 + (j + side) % 3, 6)
        for j, value in enumerate(_nominal_weights(mp))
    )
    assert all(Q(3, 4) <= value <= 1 for value in values)
    return values


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, weight) for (a, b), weight in zip(new, weights, strict=True)
    )


def _h_apply(mp, values, weights):
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            current = _mp(mp, weight) * (values[a] - values[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _sine(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            a, b = a + 18 * side, b + 18 * side
            current = mp.sin(theta[b] - theta[a])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _exchange(values, slots=18):
    first = tuple(values[j] - values[18 + j] if j < slots else 0 for j in range(18))
    return first + tuple(-value for value in first)


def _relative_common(values):
    whole = sum(values) / 36
    return tuple(
        sum(values[18 * side : 18 * (side + 1)]) / 18 - whole
        for side in (0, 1)
        for _ in range(18)
    )


def _field(mp, forms, phases, weights, slots=18):
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine, gx = _h_apply(mp, x, weights), _sine(mp, phases), _exchange(x, slots)
    gamma = 1 / (1023 * mp.pi)
    return tuple(
        (-h + gamma * s) / 9 - port for h, s, port in zip(hx, sine, gx, strict=True)
    ), tuple(gamma * h / 9 for h in hx)


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        full_observation_error_bound=NOISE,
        continued_observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_renewed_interface(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("renewed-interface controls cannot replay or advance a source")

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
        # Permit the existing row builder for static full-state comparisons;
        # every integrator, acquisition producer and operator remains blocked.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220

    def forbidden(*args, **kwargs):
        pytest.fail("static interface proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_actual_full_rows_close_the_original_aggregate_with_unequal_weights(mp):
    forms = _center(tuple(Q((j * 7) % 17 - 8, 10**5) for j in range(36)))
    phases = tuple(
        value
        for side in (0, 1)
        for value in _center(
            tuple(Q((j * 5 + side) % 11 - 5, 10**5) for j in range(18))
        )
    )
    weights = (_weights(mp, 0), _weights(mp, 1))
    assert weights[0] != weights[1]
    dx, dy = _field(mp, forms, phases, weights)
    state = tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + I(value)
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )
    before = tuple(state)
    field = field_owner._mixed_two_channel_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
        law="full",
    )
    actual_rows = field(state)
    assert len(actual_rows) == 72 and tuple(state) == before
    for interval, value in zip(actual_rows, dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    q = -sum(forms[:18])
    assert q != 0
    assert abs(-sum(dx[:18]) + 2 * _mp(mp, q)) < mp.mpf("1e-200")
    assert abs(sum(dx)) < mp.mpf("1e-200")
    assert all(abs(sum(dy[18 * p : 18 * (p + 1)])) < mp.mpf("1e-200") for p in (0, 1))
    # Hidden phases keep evolving although this single form observable closes.
    assert _dot(dy, dy) > 0
    gauged = tuple(
        value + (Q(2, 7) if j < 18 else -Q(3, 11)) for j, value in enumerate(phases)
    )
    shifted = tuple(value + Q(2, 9) for value in forms)
    transformed = _field(mp, shifted, gauged, weights)
    assert max(
        abs(a - b)
        for row, other in zip((dx, dy), transformed, strict=True)
        for a, b in zip(row, other, strict=True)
    ) < mp.mpf("1e-200")
    # A weighted biregular cross-port explains the aggregate cancellation;
    # this is an algebraic mechanism, not a newly admitted response protocol.
    permutation = tuple((5 * j + 3) % 18 for j in range(18))
    cross = [Q(0)] * 36
    strength = Q(3, 7)
    for j, target in enumerate(permutation):
        flow = strength * (forms[j] - forms[18 + target])
        cross[j] += flow
        cross[18 + target] -= flow
    assert sum(cross) == 0
    assert sum(cross[:18]) == -2 * strength * q


def test_partial_or_unbalanced_ports_do_not_inherit_the_full_scalar_law(mp):
    weights = (_weights(mp, 0), _weights(mp, 1))
    phases = (Q(0),) * 36
    first = (-Q(1, 10**6),) * 18 + (Q(1, 10**6),) * 18
    hidden = (-Q(1, 10**7),) * 9 + (Q(1, 10**7),) * 9 + (Q(0),) * 18
    second = tuple(a + b for a, b in zip(first, hidden, strict=True))
    assert sum(first[:18]) == sum(second[:18])
    derivatives = []
    for forms in (first, second):
        dx, _ = _field(mp, forms, phases, weights, slots=9)
        q = -sum(forms[:18])
        vd = (sum(forms[9:18]) - sum(forms[:9])) / 18
        vr = (sum(forms[27:]) - sum(forms[18:27])) / 18
        b = vr - vd
        rate = -sum(dx[:18])
        assert abs(rate - _mp(mp, -q + 9 * b)) < mp.mpf("1e-200")
        derivatives.append(rate)
    assert abs(derivatives[0] - derivatives[1]) > mp.mpf("1e-8")
    # Larger-class static witness only: not two attained current-source states.
    # Removing one full-port slot loses constant row sums and scalar closure.
    for forms in (first, second):
        gf = _exchange(forms)
        broken = list(gf)
        removed = forms[0] - forms[18]
        broken[0] -= removed
        broken[18] += removed
        assert sum(broken) == 0
        assert sum(broken[:18]) != sum(gf[:18])


def test_new_port_total_loss_work_and_storage_jump_are_distinct(mp, report):
    x = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 5) % 17 - 8, 10**5) for j in range(36)))
    )
    phases = tuple(Q((j % 3) - 1, 10**7) for j in range(36))
    nominal = (_nominal_weights(mp),) * 2
    actual = (_weights(mp, 0), _weights(mp, 1))
    hx, h0x = _h_apply(mp, x, actual), _h_apply(mp, x, nominal)
    gx, common = _exchange(x), _relative_common(x)
    c = 9 * (mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi))
    effort = tuple(h + 2 * c * p for h, p in zip(hx, common, strict=True))
    nominal_effort = tuple(h + 2 * c * p for h, p in zip(h0x, common, strict=True))
    nominal_commutator = tuple(
        a - b for a, b in zip(_h_apply(mp, gx, nominal), _exchange(h0x), strict=True)
    )
    actual_commutator = tuple(
        a - b for a, b in zip(_h_apply(mp, gx, actual), _exchange(hx), strict=True)
    )
    assert max(map(abs, nominal_commutator)) < mp.mpf("1e-200")
    assert _dot(actual_commutator, actual_commutator) > mp.mpf("1e-160")
    loss = _dot(hx, hx) / 9 + _dot(effort, gx)
    nominal_loss = _dot(h0x, h0x) / 9 + _dot(nominal_effort, gx)
    norm_squared = _dot(x, x)
    assert nominal_loss >= norm_squared
    assert abs(loss - nominal_loss) <= _mp(mp, 84 * WEIGHT_ERROR) * norm_squared
    assert loss >= _mp(mp, 1 - 84 * WEIGHT_ERROR) * norm_squared
    power = -_dot(effort, gx)
    assert power <= _mp(mp, 28 * WEIGHT_ERROR) * norm_squared
    sine = _sine(mp, phases)
    dx, dy = _field(mp, x, phases, actual)
    derivative = _dot(effort, dx) - _dot(sine, dy)
    assert abs(derivative + loss) < mp.mpf("1e-200")
    old_dx, old_dy = _field(mp, x, phases, actual, slots=9)
    old_derivative = _dot(effort, old_dx) - _dot(sine, old_dy)
    added = tuple(a - b for a, b in zip(gx, _exchange(x, 9), strict=True))
    assert abs(derivative - old_derivative + _dot(effort, added)) < mp.mpf("1e-200")
    assert dy == old_dy
    assert report.storage_jump == 0
    assert report.actual_dissipation_lower_bound == 1 - 84 * WEIGHT_ERROR > 0
    assert (
        report.port_power_upper_bound
        == 28 * WEIGHT_ERROR * report.post_norm_upper_bound**2
    )
    assert report.positive_work_upper_bound == HORIZON * report.port_power_upper_bound
    assert report.changed_port_storage_certified and report.work_allowance_certified
    # A full-port kernel/range witness shows why actual port passivity was
    # not inferred from total loss. These are legal static weights/states,
    # not an assertion of attainability from the acquired source family.
    vcopy = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q(j % 5 - 2, 10**5) for j in range(18)))
    )
    v = vcopy * 2
    assert _exchange(v) == (0,) * 36
    w = _exchange(_h_apply(mp, v, actual))
    assert _dot(w, w) > mp.mpf("1e-160")
    witness = tuple(a - b / 72 for a, b in zip(v, w, strict=True))
    witness_effort = tuple(
        a + 2 * c * b
        for a, b in zip(
            _h_apply(mp, witness, actual), _relative_common(witness), strict=True
        )
    )
    positive_power = -_dot(witness_effort, _exchange(witness))
    assert positive_power >= _dot(w, w) / 144 > 0
    assert positive_power <= _mp(mp, 28 * WEIGHT_ERROR) * _dot(witness, witness)


def test_new_window_identity_uses_a_fresh_norm_and_phase_budget(mp, report):
    entry = 4 * QHI + 200 * MATURE
    coupling = 2 * G * HORIZON
    upper = entry / (1 - coupling)
    # The integral bootstrap solves N_h <= N_entry + 2*g*h*N_h.
    assert upper == entry + coupling * upper
    assert report.entry_norm_upper_bound == entry
    assert report.post_norm_upper_bound == upper < Q(1, 100) < Q(1, 32)
    assert report.phase_drift_upper_bound == 3 * G * HORIZON * upper
    assert (
        report.marked_gap_upper_bound
        == QHI + 300 * MATURE + report.phase_drift_upper_bound
    )
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    assert report.finite_identity_certified
    # Check the complete weighted rows underlying the norm differential bound.
    x = _center(tuple(Q((j * 7) % 13 - 6, 10**5) for j in range(36)))
    y = tuple(Q((j % 3) - 1, 10**5) for j in range(36))
    dx, dy = _field(mp, x, y, (_weights(mp, 0), _weights(mp, 1)))
    xm, ym = tuple(_mp(mp, value) for value in x), tuple(_mp(mp, value) for value in y)
    norm_squared = _dot(xm, xm) + _dot(ym, ym)
    assert _dot(xm, dx) + _dot(ym, dy) <= 2 * _mp(mp, G) * norm_squared
    assert _dot(dy, dy) <= 4 * _mp(mp, G) ** 2 * _dot(xm, xm)
    assert 8 < 9  # (2*sqrt(2))^2 < 3^2 bounds every marked edge drift.


def test_protocol_aware_scalar_response_preserves_initial_error_and_clock(report):
    # Formal attenuation factors verify the exact semigroup without an exp call.
    q0, error, first, second = Q(7, 10**8), Q(1, 10**10), Q(2, 3), Q(3, 5)
    continued = second * (first * q0)
    assert continued == (first * second) * q0
    assert abs(continued - first * second * (q0 + error)) == first * second * error
    increment = continued - q0
    assert -2 * continued == -2 * (increment + q0)
    assert -2 * increment != -2 * continued
    assert report.scalar_decay_rate == 2
    assert report.scalar_decay_upper_bound == DECAY
    assert (
        report.scalar_initialization_error_upper_bound
        == report.inherited_output_error_upper_bound
    )
    assert (
        report.scalar_output_error_upper_bound
        == report.inherited_output_error_upper_bound * DECAY
    )
    assert report.switch_age == SWITCH_AGE
    assert report.activation_time == 2010000
    assert report.switch_time == report.activation_time + SWITCH_AGE == 5385000
    assert report.total_duration == report.switch_time + HORIZON == 5385008
    assert report.original_clock_horizon * Q(1023, 1024) == report.total_duration
    assert report.sum_closure_certified and report.scalar_response_certified


def test_fixed_renewed_window_separates_full_and_continued_records(
    report, record_property
):
    amplitude_lower, amplitude_upper = report.slow_amplitude_bounds
    rho_upper = report.slow_scale_bounds[1]
    allowance = report.inherited_output_error_upper_bound
    full_upper = (amplitude_upper + allowance) * DECAY
    continued_lower = amplitude_lower * (Q(1, 2) - 2 * rho_upper * HORIZON) - allowance
    assert amplitude_lower / 2 - allowance > 0
    assert report.full_true_bounds == (Q(0), full_upper)
    assert report.continued_true_bounds == (
        continued_lower,
        amplitude_upper + allowance,
    )
    assert report.full_record_bounds == (-NOISE, full_upper + NOISE)
    assert report.continued_record_bounds == (
        continued_lower - NOISE,
        amplitude_upper + allowance + NOISE,
    )
    assert report.full_return_margin == NOISE - full_upper > 0
    assert report.full_threshold_margin == THRESHOLD - full_upper - NOISE > 0
    assert report.continued_threshold_margin == continued_lower - NOISE - THRESHOLD > 0
    assert (
        report.record_separation_margin == continued_lower - full_upper - 2 * NOISE > 0
    )
    assert report.full_return_certified and report.continued_response_certified
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    assert report.certified
    for name in (
        "inherited_output_error_upper_bound",
        "scalar_initialization_error_upper_bound",
        "scalar_output_error_upper_bound",
        "switch_age",
        "post_duration",
        "switch_time",
        "total_duration",
        "entry_norm_upper_bound",
        "post_norm_upper_bound",
        "phase_drift_upper_bound",
        "marked_gap_upper_bound",
        "registration_margin",
        "actual_dissipation_lower_bound",
        "port_power_upper_bound",
        "positive_work_upper_bound",
        "full_return_margin",
        "full_threshold_margin",
        "continued_threshold_margin",
        "record_separation_margin",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "slow_amplitude_bounds",
        "slow_scale_bounds",
        "full_true_bounds",
        "continued_true_bounds",
        "full_record_bounds",
        "continued_record_bounds",
    ):
        record_property(
            name, json.dumps([str(value) for value in getattr(report, name)])
        )


def test_independent_readout_errors_do_not_change_true_response_claims(report):
    zero = _assess(
        full_observation_error_bound=Q(0), continued_observation_error_bound=Q(0)
    )
    assert zero.full_record_bounds == zero.full_true_bounds
    assert zero.continued_record_bounds == zero.continued_true_bounds
    assert zero.record_separation_margin == report.record_separation_margin + 2 * NOISE
    for name in ("full_observation_error_bound", "continued_observation_error_bound"):
        changed = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not changed.observation_allowances_admitted and not changed.certified
        assert not changed.fixed_threshold_certified
        assert not changed.independent_record_sets_separated
        assert changed.full_return_certified and changed.continued_response_certified
        assert changed.scalar_response_certified and changed.finite_identity_certified


def test_parent_bounds_are_rebuilt_before_the_new_interface_claim(monkeypatch, report):
    source = report.source
    cases = (
        ("source_association_certified", False),
        ("uniform_approximation_certified", False),
        ("amplitude_certified", False),
        ("total_output_error_upper_bound", Q(0)),
        ("original_output_error_upper_bound", Q(0)),
        ("nonlinear_output_error_upper_bound", Q(0)),
        ("fast_output_tail_upper_bound", Q(0)),
        ("slow_amplitude_bounds", (Q(0), Q(1))),
        ("slow_scale_bounds", (Q(0), Q(1))),
        ("nominal_q0_bounds", (Q(0), Q(0))),
        ("early_age", Q(64)),
        ("correlated_initialization_retained", False),
        ("same_actual_history_required", False),
        ("state_reset_performed", True),
    )
    for name, value in cases:
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_slow_response", lambda **kwargs: changed
        )
        assert not _assess().certified


def test_actual_source_geometry_history_and_storage_are_readmitted(monkeypatch, report):
    source = report.source
    for name, value in (
        ("clock", "wrong slow-response clock"),
        ("activation_time", Q(0)),
        ("early_observation_error_bound", NOISE),
        ("late_observation_error_bound", NOISE),
    ):
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_slow_response", lambda **kwargs: changed
        )
        assert not _assess().certified
    collective, persistent = source.source, source.source.source
    for name, value in (
        ("clock", "wrong persistent clock"),
        ("uniform_tube_certified", False),
        ("nominal_cone_certified", False),
        ("nominal_norm_upper_bound", Q(0)),
        ("whole_form_mean", Q(1)),
        ("physical_capacities", (Q(0),) + (Q(1),) * 35),
        ("original_phase_gauges_and_branches_retained", False),
    ):
        changed = replace(
            source,
            source=replace(collective, source=replace(persistent, **{name: value})),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_slow_response", lambda **kwargs: changed
        )
        assert not _assess().certified
    storage = persistent.source
    for name, value in (
        ("storage_domain_certified", False),
        ("selected_mu", Q(2)),
        ("origin_reference", "uncentered form origin"),
    ):
        changed = replace(
            source,
            source=replace(
                collective,
                source=replace(persistent, source=replace(storage, **{name: value})),
            ),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_slow_response", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(owner, "_assess_mixed_slow_response", lambda **kwargs: source)
    registered = owner._derive_registered_phase_probe_geometry()
    for name, value in (
        ("bare_gap_lower_bound", Q(3)),
        ("spectral_gap_certified", False),
        ("weighted_gap_lower_bound", Q(0)),
    ):
        altered = replace(
            registered, successor=replace(registered.successor, **{name: value})
        )
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().certified


def test_old_record_and_late_return_verdicts_do_not_admit_the_new_task(
    monkeypatch, report
):
    source = replace(
        report.source,
        certified=False,
        early_response_certified=False,
        early_record_certified=False,
        late_return_certified=False,
        late_record_certified=False,
        correlated_true_drop_certified=False,
        correlated_drop_certified=False,
        early_true_bounds=(Q(0), Q(0)),
        early_record_bounds=(Q(0), Q(0)),
        late_true_bounds=(Q(0), Q(0)),
        late_record_bounds=(Q(0), Q(0)),
        status="unrelated_previous_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(owner, "_assess_mixed_slow_response", lambda **kwargs: source)
    changed = _assess()
    assert changed.certified and changed.scalar_response_certified
    assert changed.full_record_bounds == report.full_record_bounds
    assert changed.continued_record_bounds == report.continued_record_bounds


def test_raw_primitives_are_normalized_before_any_source_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or derived arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_slow_response", forbidden)
    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
        -Q(1, 10**100),
    )
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in invalid:
            for side in (0, 1):
                value = tuple(bad if index == side else Q(0) for index in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: value})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
    for name in ("full_observation_error_bound", "continued_observation_error_bound"):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})


def test_fixed_width_admission_retains_the_exact_task_scope(report):
    error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    noise = Q(np.int64(NOISE.numerator), np.int64(NOISE.denominator))
    assert (
        _assess(
            form_error_bounds=(error for _ in range(2)),
            phase_error_bounds=(error for _ in range(2)),
            full_observation_error_bound=noise,
            continued_observation_error_bound=noise,
        )
        == report
    )
    assert report.post_duration == HORIZON
    assert report.threshold == THRESHOLD
    assert report.block_total_generator == ((Q(-1), Q(1)), (Q(1), Q(-1)))
    assert report.full_contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert report.added_contact_pairs == tuple((j, 18 + j) for j in range(9, 18))
    assert report.full_port_row_sums == report.full_port_column_sums == (Q(1),) * 18
    assert report.observation_covector == (-Q(1),) * 18 + (Q(0),) * 54
    assert report.original_donor_form_total == report.whole_form_mean == 0
    assert report.scalar_equation == "Q_prime=-2*Q"
    assert report.reduced_response == "Q_hat(s)=Q_s(3375000)*exp(-2*s)"
    assert report.rebased_equation == "Z_prime=-2*(Z+Q_entry)"
    assert (
        report.same_input_scalar_model and report.continued_A_is_boundary_effect_control
    )
    assert report.counterfactuals_share_reached_actual_member
    assert report.inherited_amplitude_rate_correlation_retained
    assert (
        report.original_baseline_retained and report.internal_rows_and_clock_unchanged
    )
    assert (
        report.independent_record_errors
        and not report.observation_error_cancellation_assumed
    )
    assert not report.full_nonlinear_state_closure
    assert not report.inherited_slow_rate_used_after_switch
    assert not report.all_future_changed_port_identity_claimed
    assert not report.port_only_passivity_claimed
    assert not report.zero_physical_switching_cost_claimed
    assert not report.original_source_reacquired and not report.state_reset_performed
    assert not report.observation_rebased_at_switch and report.switch_observations == 0
    assert report.supplied_external_protocol
    assert not report.runtime_event_executed and not report.operator_occurrence_derived
    assert not report.source_trajectory_evaluated and not report.exponentials_evaluated
    assert (
        not report.roots_numerically_evaluated
        and not report.physical_timescale_identified
    )
    assert not report.empirical_validation and report.newly_observed_aggregates == 0
    assert (
        report.required_endpoint_records == 2
        and report.numerical_response_allowance == 0
    )
    with pytest.raises(FrozenInstanceError):
        report.post_duration = Q(0)
    for name in (
        "source",
        "endpoint",
        "amplitude",
        "slow_rate",
        "horizon",
        "switch_age",
        "initial_velocity",
        "initial_phase",
        "observed_record",
        "threshold",
        "port_strength",
        "capacity",
        "initial_reading",
        "baseline",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
