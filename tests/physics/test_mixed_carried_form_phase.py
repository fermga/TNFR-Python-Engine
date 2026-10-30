"""A carried reference predicts a phase trace after its form port is removed.

The frozen fine-state oracle checks static algebra only. Constructed hidden
states illustrate information loss in a larger inference class; they are not
claimed to be acquired donor/recipient endpoints. No trajectory is advanced.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.physics import test_mixed_resource_transfer_obstruction as fine
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_carried_form_phase as owner
from tnfr.physics import _mixed_two_channel_circulation as field_owner
from tnfr.physics import _sine_coupling_successor as source_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
MATURE = Q(1, 2**28)
PREFIX = Q(2000000)
CONTINUATION = Q(10000)
NOISE = Q(1, 10**6)
THRESHOLD = Q(1, 800000)
GAMMA_LO, GAMMA_HI = Q(1, 3300), Q(1, 3000)
PHASE_ROW = (-Q(1, 9),) * 9 + (Q(1, 9),) * 9
FORM_ROW = tuple(x / 2 for x in PHASE_ROW)


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_carried_form_phase(**values)


def _project(mp, row, values):
    return sum(
        fine._mp(mp, a) * fine._mp(mp, b) for a, b in zip(row, values, strict=True)
    )


def _replace_reference(prefix, **changes):
    return replace(prefix, reference=replace(prefix.reference, **changes))


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("carried-reference controls cannot advance or replay a source")

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
        # The old immutable oracle needs this factory for static field rows,
        # never for a step, source replay or evaluated trajectory.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", factory)
            yield


@pytest.fixture(scope="module")
def mp():
    result = mpmath.mp.clone()
    result.dps = 200
    return result


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_actual_projection_retains_weight_defects_in_both_rows(mp, report):
    forms = tuple(
        value
        for side in (0, 1)
        for value in fine._center(
            tuple(Q((5 * j + side) % 19 - 9, 8192) for j in range(18))
        )
    )
    hidden = tuple(
        value
        for region in range(4)
        for value in fine._center(
            tuple(Q((3 * j + region) % 7 - 3, 10**7) for j in range(9))
        )
    )
    carrier = fine._offset(SECTION) + fine._offset(-SECTION / 7)
    phases = tuple(a + b for a, b in zip(carrier, hidden, strict=True))
    weights = (fine._weights(0), fine._weights(1))
    x, hx, sine, dx, dy = fine._oracle(mp, forms, phases, weights)
    nominal_weights = fine._nominal_weights(mp)
    nominal_h = fine._laplacian(
        tuple((a, b, fine._mp(mp, w)) for a, b, w in fine._edges(nominal_weights))
    )
    h0x = fine._block(nominal_h, x)
    carrier_sine = fine._oracle(mp, forms, carrier, weights)[2]
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - fine._mp(mp, DELTA) / (18 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    v = tuple(_project(mp, FORM_ROW, x[18 * p : 18 * (p + 1)]) for p in (0, 1))
    for p in (0, 1):
        a, b = 18 * p, 18 * (p + 1)
        q = _project(mp, PHASE_ROW, tuple(fine._mp(mp, y) for y in phases[a:b]))
        weight = _project(mp, PHASE_ROW, tuple(hx[i] - h0x[i] for i in range(a, b)))
        rw = -weight / 18
        rs = (
            gamma
            * _project(
                mp, PHASE_ROW, tuple(sine[i] - carrier_sine[i] for i in range(a, b))
            )
            / 18
        )
        t = gamma * weight / 9
        assert abs(t + 2 * gamma * rw) < mp.mpf("1e-190")
        assert abs(rw) > mp.mpf("1e-8")  # broad legal static weights, not UM images
        assert abs(_project(mp, PHASE_ROW, h0x[a:b]) - 36 * k * v[p]) < mp.mpf("1e-190")
        expected_v = -2 * k * v[p] - gamma * coefficient * mp.sin(q) / 9
        expected_v += -(v[p] - v[1 - p]) + rw + rs
        assert abs(_project(mp, FORM_ROW, dx[a:b]) - expected_v) < mp.mpf("1e-190")
        assert abs(
            _project(mp, PHASE_ROW, dy[a:b]) - (4 * gamma * k * v[p] + t)
        ) < mp.mpf("1e-190")
    assert report.projected_form_covector == FORM_ROW
    assert report.projected_phase_covector == PHASE_ROW
    assert report.projected_rows_certified
    assert not report.exact_actual_four_coordinate_closure


def test_hidden_phase_defect_is_quadratic_and_same_means_do_not_close(mp):
    gamma = 1 / (1023 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    q, h = fine._mp(mp, SECTION), fine._mp(mp, Q(1, 10**6))
    weights = (fine._nominal_weights(mp),) * 2
    forms = (Q(0),) * 36
    base = fine._offset(q) + fine._offset(mp.mpf(0))
    hidden = (h, -h) + (mp.mpf(0),) * 34
    changed = tuple(a + b for a, b in zip(base, hidden, strict=True))
    baseline = fine._oracle(mp, forms, base, weights)
    altered = fine._oracle(mp, forms, changed, weights)
    assert abs(_project(mp, PHASE_ROW, changed[:18]) - q) < mp.mpf("1e-190")
    assert _project(mp, FORM_ROW, forms[:18]) == 0
    residual = _project(
        mp,
        FORM_ROW,
        tuple(a - b for a, b in zip(altered[3][:18], baseline[3][:18], strict=True)),
    )
    expected = 2 * gamma * coefficient * mp.sin(q) * (1 - mp.cos(h)) / 81
    assert expected > 0 and abs(residual - expected) < mp.mpf("1e-190")
    assert abs(residual) <= 5 * gamma * (2 * h**2) / 81
    # Independent edge sums establish cancellation of the linear projection
    # and the quadratic bound for non-symmetric zero-regional-mean errors.
    eta = tuple(
        fine._mp(mp, value)
        for region in (0, 1)
        for value in fine._center(
            tuple(Q((j + 2 * region) % 5 - 2, 10**5) for j in range(9))
        )
    )
    angles = tuple(
        2 * mp.pi * fine._mp(mp, t) + y
        for t, y in zip(fine._turns(), base[:18], strict=True)
    )
    linear = mp.mpf(0)
    remainder = mp.mpf(0)
    edge_square = mp.mpf(0)
    for a, b, _ in fine._edges(weights[0]):
        if a < 9 <= b:
            angle, shift = angles[b] - angles[a], eta[b] - eta[a]
            linear += mp.cos(angle) * shift
            remainder += mp.sin(angle + shift) - mp.sin(angle) - mp.cos(angle) * shift
            edge_square += shift**2
    assert abs(linear) < mp.mpf("1e-190")
    assert edge_square <= 10 * fine._dot(eta, eta)
    assert abs(-gamma * remainder / 81) <= 5 * gamma * fine._dot(eta, eta) / 81
    # This pair is a larger-inference-class witness. Neither preparation nor
    # attainability from the frozen donor/recipient family is asserted.


def test_cut_changes_only_external_form_rows_at_the_same_complete_state(mp):
    forms = tuple(
        value
        for p in (0, 1)
        for value in fine._center(
            tuple(Q((j + 3 * p) % 7 - 3, 8192) for j in range(18))
        )
    )
    errors = fine._offset(SECTION / 2) + fine._offset(-SECTION / 11)
    weights = (fine._weights(0), fine._weights(1))
    state = fine._state(forms, errors)
    before = tuple(state)
    full = field_owner._mixed_two_channel_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
        law="full",
    )
    cut = tuple(
        source_owner._coupling_successor_field(
            completion="common_sense",
            new_edge_conductances=w,
            loss_normalization="bare_degree",
        )
        for w in weights
    )
    full_rates = full(state)
    cut_rates = tuple(
        cut[p](state[18 * p : 18 * (p + 1)] + state[36 + 18 * p : 36 + 18 * (p + 1)])
        for p in (0, 1)
    )
    oracle_full = fine._oracle(mp, forms, errors, weights)
    oracle_cut = fine._oracle(mp, forms, errors, weights, exchange=0)
    for interval, value in zip(
        full_rates, oracle_full[3] + oracle_full[4], strict=True
    ):
        fine._contains(mp, interval, value)
    for p in (0, 1):
        a, b = 18 * p, 18 * (p + 1)
        for interval, value in zip(
            cut_rates[p], oracle_cut[3][a:b] + oracle_cut[4][a:b], strict=True
        ):
            fine._contains(mp, interval, value)
    for i, gx in enumerate(fine._exchange(tuple(fine._mp(mp, x) for x in forms))):
        assert abs(oracle_cut[3][i] - oracle_full[3][i] - gx) < mp.mpf("1e-190")
    assert oracle_full[4] == oracle_cut[4]
    assert state == before  # no form, phase, gauge or hidden-state jump


def test_common_modes_and_original_gauges_survive_both_pieces(mp, report):
    forms = tuple(
        value
        for p in (0, 1)
        for value in fine._center(tuple(Q((j + p) % 11 - 5, 8192) for j in range(18)))
    )
    errors = fine._offset(SECTION / 3) + fine._offset(-SECTION / 13)
    gauges = (Q(2, 7), -Q(3, 11))
    shifted = tuple(
        value + gauges[p] for p in (0, 1) for value in errors[18 * p : 18 * (p + 1)]
    )
    for exchange in (0, 1):
        base = fine._oracle(
            mp, forms, errors, (fine._weights(0), fine._weights(1)), exchange=exchange
        )
        changed = fine._oracle(
            mp, forms, shifted, (fine._weights(0), fine._weights(1)), exchange=exchange
        )
        for a, b in zip(base[3] + base[4], changed[3] + changed[4], strict=True):
            assert abs(a - b) < mp.mpf("1e-190")
        for p in (0, 1):
            assert abs(sum(base[3][18 * p : 18 * (p + 1)])) < mp.mpf("1e-190")
            assert abs(sum(base[4][18 * p : 18 * (p + 1)])) < mp.mpf("1e-190")
    assert report.prefix.reference.initial_form_means == (Q(0), Q(0))
    assert report.prefix.reference.original_supplied_phase_origins == (
        -DELTA / 2,
        DELTA / 2,
    )
    assert report.reference_coordinates == ("q_D", "q_R", "v_D", "v_R")


def test_cut_metric_drops_and_carries_the_single_prefix_error(mp, report):
    h = mp.matrix(
        fine._laplacian(
            tuple(
                (a, b, fine._mp(mp, w))
                for a, b, w in fine._edges(fine._nominal_weights(mp))
            )
        )
    )
    regularized = h + mp.ones(18) / 18
    centered = mp.matrix(
        tuple(
            fine._mp(mp, x)
            for x in fine._center(tuple(Q((j * 3) % 11 - 5, 10**7) for j in range(18)))
        )
    )
    other = -centered / 3
    difference = centered - other
    extra = (
        mp.mpf(81)
        / 4
        * (mp.mpf(6) / 5) ** 2
        * (difference.T * regularized**-1 * difference)[0]
    )
    assert extra > 0  # W_plus - W_0 on the same error, never a state reset
    reference = report.prefix.reference
    rho = reference.base_metric_decay_rate
    lam = rho / 8
    e0, forcing = MATURE + SECTION_ERROR, 2 * WEIGHT_ERROR
    expected = 4 * (e0 + 2 * forcing / lam)
    assert report.metric_decay_rate == lam
    assert report.metric_forcing_upper_bound == forcing
    assert report.carried_metric_upper_bound == (
        reference.metric_upper_bound * Q(6, 5) ** 2 * (e0 + 2 * forcing / lam) ** 2
    )
    assert report.metric_lower_bound == reference.metric_lower_bound
    assert report.metric_upper_bound == reference.metric_upper_bound
    assert report.carried_error_upper_bound == expected < 8 * MATURE
    assert report.actual_observation_allowance == 8 * MATURE
    assert report.carried_error_upper_bound > 4 * e0 > 0
    assert report.switched_metric_certified and report.handoff_certified
    assert report.reference_family_carried and not report.state_reset_performed
    assert report.handoff_observations == 0


def test_reached_nonlinear_wedges_survive_cut_without_velocity_reset(mp, report):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - fine._mp(mp, DELTA) / (18 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    stiffness = 4 * gamma**2 * k * coefficient / 9
    qhi = fine._mp(mp, SECTION + SECTION_ERROR / 2)
    for s, z in ((qhi, qhi), (qhi / 3, 2 * qhi / 3)):
        for sf, zf in ((0, 0), (0, 1), (1, 0), (1, 1)):
            sd, zd = -sf * gamma**2 * s, -zf * gamma**2 * z
            sdd = -2 * k * sd - 2 * stiffness * mp.sin(s / 2) * mp.cos(z / 2)
            zdd = -2 * k * zd - 2 * stiffness * mp.sin(z / 2) * mp.cos(s / 2)
            for r, speed, acceleration, face in ((s, sd, sdd, sf), (z, zd, zdd, zf)):
                if face:
                    assert acceleration + gamma**2 * speed > 0
                else:
                    assert acceleration < 0
            assert abs((sd - zd) / 2) <= gamma**2 * qhi / 2
    # These are continuous reached coordinates, not (q_D,q_R)=(q0,0)
    # installed for a fresh isolated source at the switch.
    assert report.nominal_q0_bounds == (
        SECTION - SECTION_ERROR / 2,
        SECTION + SECTION_ERROR / 2,
    )
    assert report.continuation_wedges_certified
    assert (
        report.continuation_drift_upper_bound
        == CONTINUATION * GAMMA_HI**2 * report.nominal_q0_bounds[1] / 2
    )


def test_independent_late_records_retain_a_phase_trace_after_removal(report):
    qlo, qhi = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
    stiffness_lo = Q(32, 81) * GAMMA_LO**2
    bracket = stiffness_lo * (Q(21, 52) * PREFIX - Q(243, 256)) - Q(9, 26) * GAMMA_HI**2
    before = qlo * bracket / 2
    drift = CONTINUATION * GAMMA_HI**2 * qhi / 2
    late = before - drift
    state = 8 * MATURE
    assert report.nominal_prefix_response_lower_bound == before
    assert report.nominal_late_response_lower_bound == late
    assert report.switched_true_bounds == (late - state, qhi / 2 + state)
    assert report.never_contact_true_bounds == (-state, state)
    assert report.switched_record_bounds == (
        late - state - NOISE,
        qhi / 2 + state + NOISE,
    )
    assert report.never_contact_record_bounds == (-state - NOISE, state + NOISE)
    assert report.record_separation_margin == late - 2 * state - 2 * NOISE > 0
    assert (
        report.never_contact_record_bounds[1]
        < THRESHOLD
        < report.switched_record_bounds[0]
    )
    assert (
        report.switched_threshold_margin
        == report.switched_record_bounds[0] - THRESHOLD
        > 0
    )
    assert (
        report.never_contact_threshold_margin
        == THRESHOLD - report.never_contact_record_bounds[1]
        > 0
    )
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    assert report.certified


def test_noise_is_one_endpoint_budget_not_an_integrated_or_handoff_reading(report):
    clean = _assess(observation_error_bound=Q(0))
    assert clean.switched_true_bounds == report.switched_true_bounds
    assert clean.never_contact_true_bounds == report.never_contact_true_bounds
    assert clean.record_separation_margin - report.record_separation_margin == 2 * NOISE
    assert clean.carried_error_upper_bound == report.carried_error_upper_bound
    too_noisy = _assess(observation_error_bound=NOISE + Q(1, 10**100))
    assert too_noisy.source_association_certified and too_noisy.handoff_certified
    assert not too_noisy.certified
    assert report.handoff_observations == 0
    assert report.numerical_response_allowance == 0


def test_prefix_global_record_and_old_target_verdicts_are_not_premises(
    monkeypatch, report
):
    prefix = report.prefix
    source = replace(
        prefix.reference,
        target_domain_certified=False,
        connected_target_unreachable=False,
        disconnected_target_unreachable=False,
        positive_donor_resource_certified=False,
        certified=False,
        status="old_target_unused",
        unresolved_conditions=("old_target_unused",),
    )
    changed = replace(
        prefix,
        reference=source,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        observation_allowance_admitted=False,
        connected_record_bounds=(Q(0), Q(0)),
        cut_record_bounds=(Q(0), Q(0)),
        status="prefix_record_verdict_unused",
        unresolved_conditions=("unused_record_verdict",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_recipient_phase_response", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified and result.handoff_certified
    assert result.switched_record_bounds == report.switched_record_bounds
    assert result.record_separation_margin == report.record_separation_margin


def test_source_branch_handoff_and_metric_premises_are_rebuilt(monkeypatch, report):
    prefix = report.prefix
    for name, value in (
        ("original_relative_phase_gap", SECTION),
        ("initial_form_means", (Q(0), Q(1))),
        ("inherited_donor_velocity_retained", False),
        ("inherited_nominal_velocity_relation", "reset_zero_velocity"),
        ("donor_dwell_applied", True),
        ("phase_jump_performed", True),
        ("recipient_mature_radius", 2 * MATURE),
        ("comparison_forcing_upper_bound", WEIGHT_ERROR),
        ("initial_comparison_norm_upper_bound", MATURE),
        ("uniform_comparison_upper_bound", 4 * MATURE),
        ("conservative_metric_decay_rate", prefix.reference.base_metric_decay_rate / 4),
        ("conservative_metric_decay_rate", Q(0)),
        ("nominal_wedges_certified", False),
        ("nominal_q0_bounds", (SECTION, SECTION)),
    ):
        changed = _replace_reference(prefix, **{name: value})
        assert changed.certified  # a stale positive verdict cannot substitute
        monkeypatch.setattr(
            owner, "_assess_mixed_recipient_phase_response", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified and not result.independent_record_sets_separated
    geometry = prefix.source_geometry
    for altered in (
        replace(geometry, held_capacities=(Q(2),) + geometry.held_capacities[1:]),
        replace(
            geometry,
            marked_branch_turns=(geometry.marked_branch_turns[0] + 1,)
            + geometry.marked_branch_turns[1:],
        ),
        replace(
            geometry,
            target_phase_turns=(geometry.target_phase_turns[0] + Q(1, 9),)
            + geometry.target_phase_turns[1:],
        ),
    ):
        changed = replace(prefix, source_geometry=altered)
        monkeypatch.setattr(
            owner, "_assess_mixed_recipient_phase_response", lambda **kwargs: changed
        )
        assert not _assess().certified
    for side in (0, 1):
        sources = list(prefix.reference.sources)
        old_source = sources[side]
        sources[side] = replace(
            old_source, edge_weight_deviation_upper_bound=2 * WEIGHT_ERROR
        )
        links = {"sources": tuple(sources)}
        if side == 0:
            links["donor"] = replace(prefix.reference.donor, source=sources[side])
        else:
            links["recipient_readiness"] = replace(
                prefix.reference.recipient_readiness,
                sources=(
                    sources[side],
                    prefix.reference.recipient_readiness.sources[1],
                ),
            )
        changed = _replace_reference(prefix, **links)
        monkeypatch.setattr(
            owner, "_assess_mixed_recipient_phase_response", lambda **kwargs: changed
        )
        assert not _assess().certified


def test_raw_pairs_and_noise_precede_dependency_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        for side in (0, 1):
            values = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            result = _assess(**{name: values})
            assert not result.source_association_certified and not result.certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached a source or derived calculation")

    monkeypatch.setattr(owner, "_assess_mixed_recipient_phase_response", forbidden)
    for bad in (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
        -Q(1, 10**100),
    ):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(observation_error_bound=bad)
        for name in ("form_error_bounds", "phase_error_bounds"):
            for side in (0, 1):
                values = tuple(bad if i == side else Q(0) for i in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: values})
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})


def test_one_shot_fixed_protocol_has_no_endpoint_or_handoff_state_setter(report):
    fixed_error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    result = _assess(
        form_error_bounds=(fixed_error for _ in range(2)),
        phase_error_bounds=(fixed_error for _ in range(2)),
        observation_error_bound=Q(np.int64(1), np.int64(10**6)),
    )
    assert result == report
    assert report.prefix_duration == PREFIX
    assert report.continuation_duration == CONTINUATION
    assert report.total_duration == PREFIX + CONTINUATION
    assert report.threshold == THRESHOLD
    assert report.reference_family_carried and report.handoff_observations == 0
    assert not report.exact_actual_four_coordinate_closure
    assert not report.state_reset_performed
    assert not report.source_trajectory_evaluated and not report.runtime_event_executed
    with pytest.raises(FrozenInstanceError):
        report.threshold = Q(0)
    for name in (
        "prefix",
        "state",
        "handoff_state",
        "regional_means",
        "phase_reading",
        "observed_endpoint",
        "contact_duration",
        "continuation_duration",
        "threshold",
        "source_weights",
        "source_gauges",
        "donor_velocity",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
