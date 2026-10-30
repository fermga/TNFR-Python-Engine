"""Independent algebraic controls for the reached pair's regional form port.

Static constructions test projections and missing hypotheses. They are not
acquired source endpoints. Existing independent edge oracles are reused without
replaying a source, advancing a trajectory or installing an operator event.
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
from tnfr.physics import _mixed_positive_interface as owner
from tnfr.physics import _mixed_reciprocal_partner as field_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
MATURE = Q(1, 2**28)
FULL, CUT, HORIZON = Q(2000000), Q(10000), Q(1, 2)
NOISE, THRESHOLD = Q(1, 10**9), Q(1, 10**8)
GAMMA_LO, GAMMA_HI = Q(1, 3300), Q(1, 3000)
E, FORCING = 8 * MATURE, 2 * WEIGHT_ERROR
PHASE_ROW = (-Q(1, 9),) * 9 + (Q(1, 9),) * 9
FORM_ROW = tuple(x / 2 for x in PHASE_ROW)
CURRENT_ROW = (Q(1),) * 9 + (Q(0),) * 9 + (-Q(1),) * 9 + (Q(0),) * 9


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_positive_interface(**values)


def _project(mp, row, values):
    return sum(
        fine._mp(mp, a) * fine._mp(mp, b) for a, b in zip(row, values, strict=True)
    )


def _a_exchange(values):
    first = tuple(values[j] - values[18 + j] for j in range(9)) + (0,) * 9
    return first + tuple(-value for value in first)


def _oracle(mp, forms, phases, weights, connected=True):
    x, hx, sine, dx, dy = fine._oracle(mp, forms, phases, weights, exchange=0)
    gx = _a_exchange(x)
    if connected:
        dx = tuple(value - current for value, current in zip(dx, gx, strict=True))
    return x, hx, sine, dx, dy


def _replace_reference(handoff, **changes):
    return replace(
        handoff,
        prefix=replace(
            handoff.prefix, reference=replace(handoff.prefix.reference, **changes)
        ),
    )


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("positive-interface controls cannot advance or replay a source")

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


def test_complete_regional_projection_retains_live_common_form_and_both_defects(
    mp, report
):
    forms = tuple(Q((j * 5) % 17 - 8, 8192) for j in range(36))
    forms = fine._center(forms)
    phases = tuple(
        base + perturb
        for p, q in enumerate((SECTION / 2, -SECTION / 7))
        for base, perturb in zip(
            fine._offset(q),
            tuple(
                value
                for region in (0, 1)
                for value in fine._center(
                    tuple(Q((j + p + region) % 7 - 3, 10**7) for j in range(9))
                )
            ),
            strict=True,
        )
    )
    weights = (fine._weights(0), fine._weights(1))
    x, hx, sine, dx, dy = _oracle(mp, forms, phases, weights)
    k = mp.mpf(13) / 27 - fine._mp(mp, DELTA) / (18 * mp.pi)
    gamma, coefficient = 1 / (1023 * mp.pi), 1 + 2 * mp.cos(mp.pi / 9)
    h0 = fine._laplacian(
        tuple(
            (a, b, fine._mp(mp, w))
            for a, b, w in fine._edges(fine._nominal_weights(mp))
        )
    )
    h0x = fine._block(h0, x)
    means = tuple(sum(x[18 * p : 18 * (p + 1)]) / 18 for p in (0, 1))
    velocities = tuple(_project(mp, FORM_ROW, x[18 * p : 18 * (p + 1)]) for p in (0, 1))
    gaps = tuple(_project(mp, PHASE_ROW, phases[18 * p : 18 * (p + 1)]) for p in (0, 1))
    carrier = fine._offset(gaps[0]) + fine._offset(gaps[1])
    carrier_sine = _oracle(mp, forms, carrier, weights)[2]
    relative_mean = means[0] - means[1]
    relative_v = velocities[0] - velocities[1]
    difference = relative_mean - relative_v
    for p in (0, 1):
        a, b = 18 * p, 18 * (p + 1)
        sign = 1 if p == 0 else -1
        rw = -_project(mp, PHASE_ROW, tuple(hx[j] - h0x[j] for j in range(a, b))) / 18
        rs = (
            gamma
            * _project(
                mp, PHASE_ROW, tuple(sine[j] - carrier_sine[j] for j in range(a, b))
            )
            / 18
        )
        expected_v = (
            -2 * k * velocities[p]
            - gamma * coefficient * mp.sin(gaps[p]) / 9
            + sign * difference / 2
            + rw
            + rs
        )
        assert abs(_project(mp, FORM_ROW, dx[a:b]) - expected_v) < mp.mpf("1e-190")
        assert abs(
            _project(mp, PHASE_ROW, dy[a:b])
            - (4 * gamma * k * velocities[p] - 2 * gamma * rw)
        ) < mp.mpf("1e-190")
        assert abs(sum(dx[a:b]) / 18 + sign * difference / 2) < mp.mpf("1e-190")
    assert abs(
        (sum(dx[:18]) - sum(dx[18:])) / 18 + relative_mean - relative_v
    ) < mp.mpf("1e-190")
    field = field_owner._mixed_reciprocal_partner_field(
        first_new_edge_conductances=weights[0], second_new_edge_conductances=weights[1]
    )
    state = fine._state(forms, phases)
    before = tuple(state)
    for interval, value in zip(field(state), dx + dy, strict=True):
        fine._contains(mp, interval, value)
    assert state == before
    assert report.projected_rows_certified


def test_current_is_one_endpoint_total_with_exact_baseline_and_conserved_global_form(
    mp,
):
    forms = fine._center(tuple(Q((j * 3) % 11 - 5, 8192) for j in range(36)))
    phases = fine._offset(SECTION / 3) + fine._offset(-SECTION / 13)
    weights = (fine._weights(0), fine._weights(1))
    for connected in (False, True):
        x, _, _, dx, dy = _oracle(mp, forms, phases, weights, connected=connected)
        current = _project(mp, CURRENT_ROW, x) if connected else mp.mpf(0)
        assert abs(sum(dx[:18]) + current) < mp.mpf("1e-190")
        assert abs(sum(dx[18:]) - current) < mp.mpf("1e-190")
        assert abs(sum(dx)) < mp.mpf("1e-190")
        for p in (0, 1):
            assert abs(sum(dy[18 * p : 18 * (p + 1)])) < mp.mpf("1e-190")
        # A common form-origin change and independent phase gauges leave
        # currents and rates unchanged; a relative form shift does not.
        shifted_forms = tuple(value + Q(2, 7) for value in forms)
        shifted_phases = tuple(
            value + (Q(3, 11) if j < 18 else -Q(2, 9)) for j, value in enumerate(phases)
        )
        altered = _oracle(
            mp, shifted_forms, shifted_phases, weights, connected=connected
        )
        assert max(
            abs(a - b) for a, b in zip(dx + dy, altered[3] + altered[4], strict=True)
        ) < mp.mpf("1e-190")
    assert sum(value**2 for value in CURRENT_ROW) == 18
    assert mp.sqrt(18) < fine._mp(mp, Q(9, 2))
    shifted_relative = tuple(
        value + (Q(1, 1024) if j < 18 else -Q(1, 1024)) for j, value in enumerate(forms)
    )
    assert sum(
        coefficient * (shifted - original)
        for coefficient, shifted, original in zip(
            CURRENT_ROW, shifted_relative, forms, strict=True
        )
    ) == Q(18, 1024)
    assert abs(
        _project(mp, CURRENT_ROW, shifted_relative)
        - _project(mp, CURRENT_ROW, forms)
        - fine._mp(mp, Q(18, 1024))
    ) < mp.mpf("1e-190")


def test_regional_port_breaks_old_leaf_commutation_and_storage_monotonicity(mp):
    weights = (fine._nominal_weights(mp),) * 2
    h0 = fine._laplacian(
        tuple((a, b, fine._mp(mp, w)) for a, b, w in fine._edges(weights[0]))
    )
    centered = (-mp.mpf(1),) * 9 + (mp.mpf(1),) * 9 + (mp.mpf(0),) * 18
    gx = _a_exchange(centered)
    assert sum(centered[:18]) == 0 and sum(gx[:18]) != 0
    commutator = tuple(
        a - b
        for a, b in zip(
            fine._block(h0, gx), _a_exchange(fine._block(h0, centered)), strict=True
        )
    )
    assert max(abs(value) for value in commutator) > 1
    epsilon = fine._mp(mp, Q(1, 10**6))
    first = (epsilon / 2,) * 9 + (3 * epsilon / 2,) * 9
    forms = first + tuple(-value for value in first)
    _, hx, sine, dx, dy = _oracle(mp, forms, (Q(0),) * 36, weights)
    c = mp.mpf(13) / 3 - fine._mp(mp, DELTA) / (2 * mp.pi)
    work = -fine._dot(hx, _a_exchange(forms))
    storage_rate = fine._dot(hx, dx) - fine._dot(sine, dy)
    assert abs(work - 18 * c * epsilon**2) < mp.mpf("1e-190")
    assert abs(storage_rate - 2 * c * (9 - 2 * c) * epsilon**2) < mp.mpf("1e-190")
    assert work > 0 and storage_rate > 0
    # This arbitrarily small static state has nonzero relative common form.
    # It disproves a blanket storage claim, not acquisition from our sources.


def test_paid_cut_sharpens_form_error_without_a_new_reading(mp, report):
    a = E / (1 + CUT / 3) + 6 * GAMMA_HI * E + 3 * FORCING
    exact_scalar_envelope = fine._mp(mp, E) * mp.exp(-fine._mp(mp, CUT) / 3)
    exact_scalar_envelope += fine._mp(mp, 6 * GAMMA_HI * E + 3 * FORCING) * (
        1 - mp.exp(-fine._mp(mp, CUT) / 3)
    )
    assert exact_scalar_envelope < fine._mp(mp, a) < fine._mp(mp, E)
    assert report.cut_form_error_upper_bound == a
    assert report.handoff_phase_error_upper_bound == E
    assert report.cut_form_error_certified
    assert report.handoff.handoff_observations == 0
    assert report.handoff.observation_error_bound == 0


def test_component_bootstrap_prices_new_means_without_commuting_metrics(mp, report):
    a = E / (1 + CUT / 3) + 6 * GAMMA_HI * E + 3 * FORCING
    s = 2 * GAMMA_HI * HORIZON
    ux = (a + FORCING * HORIZON + s * (E + FORCING * HORIZON)) / (1 - s * s)
    uy = (E + FORCING * HORIZON + s * (a + FORCING * HORIZON)) / (1 - s * s)
    assert 0 < s < 1
    assert ux == a + FORCING * HORIZON + s * uy
    assert uy == E + FORCING * HORIZON + s * ux
    assert report.positive_form_error_upper_bound == ux
    assert report.positive_phase_error_upper_bound == uy
    assert report.transfer_error_upper_bound == Q(9, 2) * HORIZON * ux
    assert report.component_comparison_certified
    assert ux > a > 0 and uy > E
    # Inspect the differential norm inequalities on noncommuting matrices
    # and an error with live per-copy common form. No old centered inverse
    # or quadratic-exchange metric is available for this point.
    h0 = fine._laplacian(
        tuple(
            (i, j, fine._mp(mp, w))
            for i, j, w in fine._edges(fine._nominal_weights(mp))
        )
    )
    bare = fine._laplacian(
        tuple((i, j, mp.mpf(1)) for i, j, _ in fine._edges(fine._weights(0)))
    )
    ex = tuple(fine._mp(mp, Q((j * 7) % 13 - 6, 10**12)) for j in range(36))
    ey = tuple(fine._mp(mp, Q((j * 3) % 11 - 5, 10**10)) for j in range(36))
    hx, hy = fine._block(h0, ex), fine._block(bare, ey)
    gx = _a_exchange(ex)
    gamma = 1 / (1023 * mp.pi)
    dx = tuple(
        -hv / 9 - gv - gamma * kv / 18 for hv, gv, kv in zip(hx, gx, hy, strict=True)
    )
    dy = tuple(gamma * hv / 9 for hv in hx)
    xnorm, ynorm = mp.sqrt(fine._dot(ex, ex)), mp.sqrt(fine._dot(ey, ey))
    assert sum(ex[:18]) != 0
    assert fine._dot(ex, hx) / 9 + fine._dot(ex, gx) >= 0
    assert fine._dot(ex, dx) / xnorm <= fine._mp(mp, 2 * GAMMA_HI) * ynorm
    assert fine._dot(ey, dy) / ynorm <= fine._mp(mp, 2 * GAMMA_HI) * xnorm


def test_reached_negative_form_contrast_comes_from_the_paid_cut(mp, report):
    qlo = SECTION - SECTION_ERROR / 2
    qhi = SECTION + SECTION_ERROR / 2
    fraction = 1 - GAMMA_HI**2 * (FULL + CUT)
    assert fraction == Q(233, 300)
    assert 1 - qhi * qhi / 6 > Q(99, 100)
    assert fraction * Q(99, 100) > Q(3, 4)
    assert 1 - Q(1) / (1 + Q(8, 9) * CUT) > Q(99, 100)
    # Exact damping kernels are algebraic controls, not the actual source
    # velocity at the handoff. The nonpositive inherited term is retained.
    for damping in (Q(8, 9), Q(9, 10), Q(1)):
        dm = fine._mp(mp, damping)
        kernel = (1 - mp.exp(-dm * fine._mp(mp, CUT))) / dm
        assert kernel > fine._mp(mp, Q(99, 100) / damping)
    floor = Q(3, 4) * Q(99, 100) * GAMMA_LO * 2 * qlo / (18 * Q(1, 2))
    assert floor == qlo / 20000
    assert report.inherited_form_contrast_lower_bound == floor
    assert report.inherited_orientation_certified


def test_new_window_trap_restoring_sign_and_cooperative_transfer(mp, report):
    qlo = SECTION - SECTION_ERROR / 2
    qhi = SECTION + SECTION_ERROR / 2
    norm = Q(1, 100) / (1 - 2 * GAMMA_HI * HORIZON)
    assert norm < Q(1, 32)
    assert qhi + 12 * MATURE + 4 * GAMMA_HI * norm * HORIZON < DELTA / 4
    assert Q(233, 300) * qlo - Q(4, 3) * GAMMA_HI * HORIZON * norm > 0
    assert 2 * norm / 3 < 1
    assert Q(14, 9) * WEIGHT_ERROR * norm * (1 + GAMMA_HI) < FORCING
    # The Metzler lower bound keeps feedback u in b' nonnegative. Neither
    # previous per-copy centeredness nor old wedges are assumed on this port.
    h = fine._mp(mp, HORIZON)
    exact_kernel = mp.exp(-h) * (1 - mp.exp(-h))
    rational_kernel = (1 - HORIZON) * (HORIZON - HORIZON**2 / 2)
    assert rational_kernel == Q(3, 16)
    assert exact_kernel > fine._mp(mp, rational_kernel)
    assert report.nominal_transfer_lower_bound == 9 * (qlo / 20000) * rational_kernel
    assert report.new_trap_certified and report.identity_certified
    assert report.nominal_transfer_certified


def test_independent_form_records_separate_from_remaining_cut_with_one_noise_budget(
    report,
):
    norm = Q(1, 100) / (1 - 2 * GAMMA_HI * HORIZON)
    lower = report.nominal_transfer_lower_bound - report.transfer_error_upper_bound
    upper = 54 * HORIZON * norm
    assert report.positive_true_bounds == (lower, upper)
    assert report.cut_true_bounds == (Q(0), Q(0))
    assert report.positive_record_bounds == (lower - NOISE, upper + NOISE)
    assert report.cut_record_bounds == (-NOISE, NOISE)
    assert report.record_separation_margin == lower - 2 * NOISE > 0
    assert NOISE < THRESHOLD < lower - NOISE
    assert report.positive_threshold_margin == lower - NOISE - THRESHOLD > 0
    assert report.cut_threshold_margin == THRESHOLD - NOISE > 0
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    assert report.certified
    clean = _assess(observation_error_bound=0)
    assert clean.positive_true_bounds == report.positive_true_bounds
    assert clean.record_separation_margin - report.record_separation_margin == 2 * NOISE
    assert clean.handoff == report.handoff
    noisy = _assess(observation_error_bound=NOISE + Q(1, 10**100))
    assert noisy.source_association_certified and not noisy.certified


def test_prior_phase_reading_and_target_conclusions_do_not_decide_form_transfer(
    monkeypatch, report
):
    handoff = report.handoff
    old = handoff.prefix
    reference = replace(
        old.reference,
        positive_donor_resource_certified=False,
        connected_target_unreachable=False,
        disconnected_target_unreachable=False,
        certified=False,
        status="old_target_unused",
        unresolved_conditions=("unused",),
    )
    prefix = replace(
        old,
        reference=reference,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        status="old_phase_record_unused",
        unresolved_conditions=("unused",),
    )
    changed = replace(
        handoff,
        prefix=prefix,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        switched_record_bounds=(Q(0), Q(0)),
        never_contact_record_bounds=(Q(0), Q(0)),
        status="late_phase_record_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_carried_form_phase", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified and result.source_association_certified
    assert result.positive_record_bounds == report.positive_record_bounds
    assert result.record_separation_margin == report.record_separation_margin


def test_source_clock_branch_and_carried_error_premises_are_rebuilt(
    monkeypatch, report
):
    handoff = report.handoff
    for name, value in (
        ("initial_form_means", (Q(0), Q(1))),
        ("original_relative_phase_gap", SECTION),
        ("recipient_mature_radius", 2 * MATURE),
        ("recipient_isolated_deadline", Q(10**9)),
        ("inherited_donor_velocity_retained", False),
        ("inherited_nominal_velocity_relation", "reset_zero_velocity"),
        ("comparison_forcing_upper_bound", WEIGHT_ERROR),
        ("uniform_comparison_upper_bound", 4 * MATURE),
        ("clock", "unscaled"),
    ):
        changed = _replace_reference(handoff, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_carried_form_phase", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified and not result.independent_record_sets_separated
    for name, value in (
        ("prefix_duration", FULL + 1),
        ("continuation_duration", CUT + 1),
        ("handoff_observations", 1),
        ("state_reset_performed", True),
        ("reference_family_carried", False),
        ("carried_error_upper_bound", Q(0)),
        ("metric_forcing_upper_bound", FORCING / 2),
        ("metric_decay_rate", Q(0)),
    ):
        changed = replace(handoff, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_carried_form_phase", lambda **kwargs: changed
        )
        assert not _assess().certified
    geometry = handoff.prefix.source_geometry
    altered = replace(
        geometry,
        marked_branch_turns=(geometry.marked_branch_turns[0] + 1,)
        + geometry.marked_branch_turns[1:],
    )
    changed = replace(handoff, prefix=replace(handoff.prefix, source_geometry=altered))
    monkeypatch.setattr(
        owner, "_assess_mixed_carried_form_phase", lambda **kwargs: changed
    )
    assert not _assess().certified
    reference = handoff.prefix.reference
    for side in (0, 1):
        sources = list(reference.sources)
        sources[side] = replace(
            sources[side], edge_weight_deviation_upper_bound=2 * WEIGHT_ERROR
        )
        links = {"sources": tuple(sources)}
        if side == 0:
            links["donor"] = replace(reference.donor, source=sources[side])
        else:
            links["recipient_readiness"] = replace(
                reference.recipient_readiness,
                sources=(sources[side], reference.recipient_readiness.sources[1]),
            )
        changed = _replace_reference(handoff, **links)
        monkeypatch.setattr(
            owner, "_assess_mixed_carried_form_phase", lambda **kwargs: changed
        )
        assert not _assess().certified


def test_raw_pairs_and_new_form_noise_precede_source_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        for side in (0, 1):
            values = tuple(
                ERROR + Q(1, 10**100) if j == side else ERROR for j in range(2)
            )
            result = _assess(**{name: values})
            assert not result.source_association_certified and not result.certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_carried_form_phase", forbidden)
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
                values = tuple(bad if j == side else Q(0) for j in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: values})
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})


def test_one_shot_fixed_protocol_does_not_install_an_endpoint_or_operator(report):
    error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    result = _assess(
        form_error_bounds=(error for _ in range(2)),
        phase_error_bounds=(error for _ in range(2)),
        observation_error_bound=Q(np.int64(1), np.int64(10**9)),
    )
    assert result == report
    assert report.prefix_full_duration == FULL and report.prefix_cut_duration == CUT
    assert report.positive_duration == HORIZON
    assert report.handoff_time == FULL + CUT
    assert report.threshold == THRESHOLD
    assert report.numerical_response_allowance == 0
    assert report.observation_covector == (-Q(1),) * 18 + (Q(0),) * 54
    assert report.outgoing_current_covector == CURRENT_ROW + (Q(0),) * 36
    assert report.outgoing_current_covector_norm_squared == 18
    assert report.initial_donor_form_total == report.whole_form_mean == 0
    assert report.exact_zero_entry_total_from_prior_laws
    assert report.relative_common_form_mode_retained
    assert report.nominal_coordinates == ("a", "v_D", "v_R", "q_D", "q_R")
    assert not report.old_phase_reading_consumed
    assert not report.old_commuting_metric_transferred
    assert not report.per_copy_zero_form_leaf_assumed_after_switch
    assert not report.storage_monotonicity_claimed_after_switch
    assert not report.exact_actual_five_coordinate_closure
    assert not report.minimal_collective_state_established
    assert not report.state_reset_performed
    assert not report.observed_record_supplied
    assert not report.source_trajectory_evaluated
    assert not report.runtime_event_executed
    assert not report.empirical_validation
    with pytest.raises(FrozenInstanceError):
        report.threshold = Q(0)
    for name in (
        "handoff",
        "state",
        "regional_means",
        "relative_common_form",
        "donor_velocity",
        "observed_transfer",
        "source_weights",
        "phase_reading",
        "positive_duration",
        "threshold",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
