"""Independent admission controls for registered regional phase interventions.

Constructed phase points and work calculations are not attained endpoints or
an implemented controller. The theorem excludes a conditionally retained
class; it does not certify all-window registration of a supplied intervention.
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
from tnfr.physics import _mixed_phase_feedback as comparator_owner
from tnfr.physics import _mixed_reciprocal_partner as full_owner
from tnfr.physics import _mixed_registered_phase_probe as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
REGISTRATION = DELTA / 4
GAMMA = Q(1, 3000)
NOISE = Q(1, 10**6)
WEIGHT_ERROR = 10**9 * ETA
BOUND = Q(45, 4) * GAMMA * REGISTRATION + Q(675, 2) * WEIGHT_ERROR


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


def _weights(side):
    return tuple(Q(3, 4) + Q((i + 2 * side) % 5, 16) for i in range(63))


def _edges(side, weights=None):
    old, new = _support()
    weights = _weights(side) if weights is None else weights
    return tuple((18 * side + a, 18 * side + b, Q(1)) for a, b in old) + tuple(
        (18 * side + a, 18 * side + b, w)
        for (a, b), w in zip(new, weights, strict=True)
    )


def _state(forms, errors):
    return tuple(I(x) for x in forms) + tuple(
        2 * pi_interval() * turn + error
        for turn, error in zip(_turns() * 2, errors, strict=True)
    )


def _oracle(mp, forms, errors, weights=None):
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns() * 2, errors, strict=True)
    )
    q, sine = [mp.mpf(0)] * 36, [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(side, None if weights is None else weights[side]):
            gradient = _mp(mp, weight) * (x[a] - x[b])
            current = mp.sin(phase[b] - phase[a])
            q[a] += gradient
            q[b] -= gradient
            sine[a] += current
            sine[b] -= current
    gamma = 1 / (1023 * mp.pi)
    intrinsic = tuple((-a + gamma * b) / 9 for a, b in zip(q, sine))
    inputs = [mp.mpf(0)] * 36
    for j in range(9):
        difference = x[j] - x[18 + j]
        inputs[j], inputs[18 + j] = -difference, difference
    dx = tuple(a + b for a, b in zip(intrinsic, inputs))
    phases = tuple(gamma * value / 9 for value in q)
    return x, tuple(q), tuple(sine), tuple(inputs), dx + phases


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("reciprocal controls cannot advance or replay a source")

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
        starting_gap_upper_bounds=(DELTA / 8, DELTA / 8),
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_registered_phase_probe(**arguments)


@pytest.fixture(scope="module")
def report():
    return _assess()


def _partners():
    return tuple(9 + (5 * (j - 4) + 4) % 9 for j in range(9))


def _cross_matrix():
    edges = set(_support()[1])
    return tuple(tuple(int((i, j) in edges) for j in _partners()) for i in range(9))


def _currents(mp, phases):
    result = [mp.mpf(0)] * 18
    for a, b, _ in _edges(0):
        current = mp.sin(phases[b] - phases[a])
        result[a] += current
        result[b] -= current
    return tuple(result)


def test_marked_relabeling_makes_the_actual_cross_support_symmetric(report):
    geometry = report.geometry
    cross = _cross_matrix()
    assert len(_support()[0]) == 18 and len(_support()[1]) == 63
    assert sum(a < 9 <= b for a, b in _support()[1]) == 45
    assert all(sum(row) == 5 for row in cross)
    assert all(cross[i][j] == cross[j][i] for i in range(9) for j in range(9))
    assert all(
        cross[i][j] == int((i - j) % 9 in (0, 1, 2, 7, 8))
        for i in range(9)
        for j in range(9)
    )
    turns = _turns()
    branches = tuple(turns[b] - turns[a] for a, b in enumerate(_partners()))
    assert all(value.denominator == 1 for value in branches)
    assert geometry.cross_incidence == cross
    assert geometry.marked_partner_indices == _partners()
    assert geometry.marked_branch_turns == branches
    assert (
        geometry.cross_symmetry_certified
        and geometry.connected_support_certified
        and geometry.certified
    )


def test_nonlinear_cross_current_cancels_without_an_equitable_phase_carrier(mp):
    cross = _cross_matrix()
    angles = tuple(_mp(mp, Q(j * j - 9, 512)) for j in range(9))
    assert len(set(angles[i + 1] - angles[i] for i in range(8))) > 1
    branches = tuple(_turns()[b] - _turns()[a] for a, b in enumerate(_partners()))
    for gaps in ((Q(0),) * 9, tuple(REGISTRATION * Q(1 + j % 3, 4) for j in range(9))):
        phases = [mp.mpf(0)] * 18
        for i, b in enumerate(_partners()):
            phases[i] = angles[i]
            phases[b] = angles[i] + _mp(mp, gaps[i]) + 2 * mp.pi * _mp(mp, branches[i])
        currents = _currents(mp, phases)
        explicit = sum(
            mp.sin(phases[_partners()[j]] - phases[i])
            for i in range(9)
            for j in range(9)
            if cross[i][j]
        )
        assert abs(sum(currents[:9]) - explicit) < mp.mpf("1e-190")
        assert abs(sum(currents)) < mp.mpf("1e-190")
        assert abs(explicit) <= _mp(mp, 45 * REGISTRATION)
        if not any(gaps):
            assert abs(explicit) < mp.mpf("1e-190")
        else:
            assert abs(explicit) > mp.mpf("1e-10")
    # These deliberately nonuniform phase points test an all-phase algebraic
    # cancellation. They are not asserted to be reached acquired states.


def test_spanning_tree_variance_supplies_the_uniform_form_difference_bound(report):
    geometry = report.geometry
    expected = tuple(
        (a, b, Q(1, 9) if (a % 18, b % 18) in _support()[0] else Q(1, 12))
        for side in (0, 1)
        for a, b, _ in _edges(side)
    )
    expected += tuple((j, 18 + j, Q(1)) for j in range(9))
    assert geometry.combined_weighted_edges == expected
    assert len(expected) == 171
    assert (
        geometry.minimum_weight_lower_bound
        == min(w for _, _, w in expected)
        == Q(1, 12)
    )
    adjacency = [[] for _ in range(36)]
    for a, b, _ in expected:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = {0: None}
    queue = [0]
    for node in queue:
        for other in adjacency[node]:
            if other not in parent:
                parent[other] = node
                queue.append(other)
    assert len(parent) == 36
    tree = tuple(
        (node, ancestor) for node, ancestor in parent.items() if ancestor is not None
    )
    assert len(tree) == 35
    assert geometry.combined_components == (tuple(range(36)),)
    vector = tuple(Q(2 * i - 35, 8192) for i in range(36))
    assert sum(vector) == 0
    norm_squared = sum(value**2 for value in vector)
    variance = (
        sum((vector[i] - vector[j]) ** 2 for i in range(36) for j in range(i + 1, 36))
        / 36
    )
    assert norm_squared == variance
    tree_energy = sum((vector[a] - vector[b]) ** 2 for a, b in tree)
    # Every simple tree path uses at most35 edges. Cauchy for every pair,
    # followed by the exact pairwise variance, yields (35^2/2)*tree_energy.
    assert norm_squared <= Q(35**2, 2) * tree_energy
    energy = sum(w * (vector[a] - vector[b]) ** 2 for a, b, w in expected)
    assert energy >= Q(1, 12) * tree_energy
    gap = 2 * Q(1, 12) / 35**2
    assert gap == geometry.spanning_tree_gap_lower_bound == Q(1, 7350)
    assert energy >= gap * norm_squared
    assert report.full_sine_force_norm_upper_bound == 6 * GAMMA
    assert report.form_difference_upper_bound == 6 * GAMMA / gap == Q(147, 10)
    assert report.form_difference_upper_bound < report.form_difference_envelope == 15
    # This is a paired form difference bound, not a norm or a lifetime
    # certificate for either controlled phase state.


def test_phase_rate_work_jump_work_and_gauge_are_separately_accounted(mp):
    original_clock = Q(1023, 1024)
    regional_gap = REGISTRATION / 4
    errors = (-regional_gap / 2,) * 9 + (regional_gap / 2,) * 9 + (Q(0),) * 18
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(36))
    x, gradient, sine, inputs, full = _oracle(mp, forms, errors)
    gamma = 1 / (1023 * mp.pi)
    phase_input = (Q(1, 128),) * 9 + (-Q(1, 256),) * 9 + (Q(0),) * 18
    rate_work = -sum(value * _mp(mp, rate) for value, rate in zip(sine, phase_input))
    relative_rate = phase_input[0] - phase_input[9]
    assert abs(rate_work + _mp(mp, relative_rate) * sum(sine[:9])) < mp.mpf("1e-190")
    assert abs(rate_work) > mp.mpf("1e-10")
    original_time_work = -sum(
        value * _mp(mp, original_clock * rate) for value, rate in zip(sine, phase_input)
    )
    assert abs(original_time_work / _mp(mp, original_clock) - rate_work) < mp.mpf(
        "1e-190"
    )
    phase_rates = tuple(
        value + _mp(mp, rate) for value, rate in zip(full[36:], phase_input)
    )
    comp_forms = tuple(-q / 9 + added for q, added in zip(gradient, inputs))
    arguments = dict(
        first_new_edge_conductances=_weights(0),
        second_new_edge_conductances=_weights(1),
    )
    fields = (
        full_owner._mixed_reciprocal_partner_field(**arguments),
        comparator_owner._mixed_without_form_feedback_field(**arguments),
    )
    for field, expected in zip(fields, (full, comp_forms + full[36:])):
        for interval, value in zip(field(_state(forms, errors)), expected):
            _contains(mp, interval, value)
    base = -sum(value**2 for value in gradient) / 9 + sum(
        a * b for a, b in zip(gradient, inputs)
    )
    full_derivative = sum(a * b for a, b in zip(gradient, full[:36])) - sum(
        a * b for a, b in zip(sine, phase_rates)
    )
    comparator_derivative = sum(a * b for a, b in zip(gradient, comp_forms)) - sum(
        a * b for a, b in zip(sine, phase_rates)
    )
    assert abs(full_derivative - base - rate_work) < mp.mpf("1e-190")
    assert abs(
        comparator_derivative
        - base
        - rate_work
        + gamma * sum(a * b for a, b in zip(gradient, sine)) / 9
    ) < mp.mpf("1e-190")
    phases = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns(), errors[:18])
    )
    for shift, sign in ((regional_gap, -1), (-regional_gap, 1)):
        jumps = (shift / 2,) * 9 + (-shift / 2,) * 9
        work = sum(
            mp.cos(phases[b] - phases[a])
            - mp.cos(phases[b] - phases[a] + _mp(mp, jumps[b] - jumps[a]))
            for a, b, _ in _edges(0)
        )
        cross_only = sum(
            mp.cos(phases[b] - phases[a])
            - mp.cos(phases[b] - phases[a] - _mp(mp, shift))
            for a, b in _support()[1]
            if a < 9 <= b
        )
        assert abs(work - cross_only) < mp.mpf("1e-190")
        assert sign * work > 0
        assert abs(regional_gap - shift) <= REGISTRATION
    gauge = (Q(17, 32),) * 18
    assert all(gauge[b] - gauge[a] == 0 for a, b, _ in _edges(0))
    # Uniform regional jumps preserve internal-ring gaps; gauge jumps leave
    # every edge gap unchanged. Neither calculation executes an intervention
    # or authenticates that an actual source can meet its required identity.


def test_output_ode_has_a_positive_kernel_with_finite_total_mass(mp, report):
    # Eliminate the two regional differences directly, before using a
    # closed scalar impulse response. This is not a sampled trajectory.
    k = Q(13, 27)
    a, b, forcing = Q(-7, 13), Q(5, 17), Q(11, 19)
    da = -(k + 2) * a + k * b + forcing
    db = k * a - k * b - forcing
    z, dz, ddz = a + b, da + db, -2 * da
    assert dz == -2 * a
    assert ddz + (2 * k + 2) * dz + 2 * k * z == -2 * forcing
    # Q=-9z/2 gives forcing +9v, not -9v or 18v.
    assert -Q(9, 2) * (ddz + (2 * k + 2) * dz + 2 * k * z) == 9 * forcing
    assert -(k + 2) * 0 + k * 0 + forcing + k * 0 - k * 0 - forcing == 0
    exact_k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    assert exact_k > _mp(mp, Q(4, 9))
    assert Q(13, 27) - DELTA / 54 - Q(4, 9) == report.nominal_coupling_lower_margin > 0
    root = mp.sqrt(exact_k**2 + 1)
    slow, fast = exact_k + 1 - root, exact_k + 1 + root
    assert 0 < slow < fast
    assert abs(slow * fast - 2 * exact_k) < mp.mpf("1e-190")
    assert abs(slow + fast - (2 * exact_k + 2)) < mp.mpf("1e-190")
    for time in (Q(0), Q(1, 32), Q(1)):
        t = _mp(mp, time)
        low, high = mp.exp(-slow * t), mp.exp(-fast * t)
        value = 9 * (low - high) / (fast - slow)
        first = 9 * (-slow * low + fast * high) / (fast - slow)
        second = 9 * (slow**2 * low - fast**2 * high) / (fast - slow)
        assert value >= 0
        assert abs(second + (2 * exact_k + 2) * first + 2 * exact_k * value) < mp.mpf(
            "1e-190"
        )
        if time == 0:
            assert value == 0 and abs(first - 9) < mp.mpf("1e-190")
        else:
            assert value > 0
    # Integrating the two exponentials analytically, not quadrature, gives
    # total mass9/(slow*fast). Its finite value is the all-horizon step.
    mass = 9 * (1 / slow - 1 / fast) / (fast - slow)
    assert abs(mass - 9 / (2 * exact_k)) < mp.mpf("1e-190")
    assert mass < _mp(mp, report.output_kernel_mass_upper_bound)
    assert report.output_kernel_mass_upper_bound == Q(81, 8)


def test_uniform_bound_keeps_the_actual_weight_residual_and_external_identity(
    report, record_property
):
    phase_force = 5 * GAMMA * REGISTRATION / 9
    weight_force = Q(50, 3) * WEIGHT_ERROR
    assert report.cross_current_upper_bound == 45 * REGISTRATION
    assert report.regional_sine_force_upper_bound == phase_force
    assert report.weight_residual_upper_bound == weight_force > 0
    assert report.regional_residual_upper_bound == phase_force + weight_force
    assert (
        report.transfer_difference_upper_bound
        == 2 * Q(81, 8) * (phase_force + weight_force)
        == BOUND
    )
    assert BOUND - Q(3, 3276800) == Q(675, 2) * WEIGHT_ERROR > 0
    assert Q(3, 3276800) < Q(19, 20) * NOISE
    assert Q(675, 2) * WEIGHT_ERROR < NOISE / 20
    assert BOUND < NOISE < 2 * NOISE
    assert report.source_association_certified and report.difference_bound_certified
    assert report.shared_record_overlap_certified and report.certified
    assert report.status == "conditional_shared_record_overlap"
    assert report.overlap_margin == 2 * NOISE - BOUND > 0
    assert report.all_window_registration_is_external_condition
    assert not report.all_window_registration_certified
    assert not report.identity_retention_certified
    assert not report.intervention_admitted
    assert report.matched_initial_state_assumed
    assert not report.comparator_own_law_source_acquisition_claimed
    assert report.registration_gap_bound == REGISTRATION
    assert report.edge_weight_deviation_upper_bound == WEIGHT_ERROR
    record_property("uniform_true_transfer_difference_upper_bound", str(BOUND))
    record_property("per_record_error_allowance", str(NOISE))
    record_property("shared_record_margin", str(report.overlap_margin))
    record_property("all_window_registration", "external condition; not certified")
    record_property("selected_control_or_horizon", "none")


def test_true_record_midpoint_is_sufficient_including_equality_not_low_noise():
    # These are abstract real-output lemmas, not generated source endpoints.
    # A proved bound on true outputs, unlike overlap of outer forecasts,
    # places their exact midpoint in BOTH admissible measurement intervals.
    for first in (Q(-3), Q(0), Q(7, 5)):
        for difference in (-BOUND, Q(0), BOUND):
            second = first + difference
            shared_record = (first + second) / 2
            assert abs(shared_record - first) <= BOUND / 2
            assert abs(shared_record - second) <= BOUND / 2
    equality = _assess(readout_error_bound=BOUND / 2)
    assert equality.overlap_margin == 0
    assert equality.certified and equality.shared_record_overlap_certified
    for noise in (Q(0), BOUND / 2 - Q(1, 10**100)):
        unavailable = _assess(readout_error_bound=noise)
        assert unavailable.difference_bound_certified
        assert unavailable.readout_allowance_admitted
        assert unavailable.overlap_margin < 0
        assert (
            not unavailable.certified
            and not unavailable.shared_record_overlap_certified
        )
        assert unavailable.status == "unavailable"
        assert (
            "reading_allowance_does_not_certify_overlap"
            in unavailable.unresolved_conditions
        )
    outside = _assess(readout_error_bound=NOISE + Q(1, 10**100))
    assert outside.difference_bound_certified and outside.overlap_margin > 0
    assert not outside.readout_allowance_admitted and not outside.certified
    # Smaller noise has no separation verdict: the bound is sufficient only.


def test_source_and_weight_associations_are_fresh_but_contact_tubes_are_not_extended(
    monkeypatch, report
):
    original = report.reference

    def install(value):
        monkeypatch.setattr(
            owner, "_assess_mixed_reciprocal_partner", lambda **kwargs: value
        )

    install(
        replace(
            original,
            certified=False,
            positive_information_certified=False,
            partner_history_obstruction_certified=False,
            full_state_tube_certified=False,
            constituent_identity_retained_certified=False,
            joint_norm_upper_bound=Q(100),
            uniform_arithmetic_error_upper_bound=Q(1),
        )
    )
    assert _assess().certified
    for side in (0, 1):
        for field in (
            "source_certified",
            "finite_protocol_ready_ages_exist",
            "conditional_ready_sources",
        ):
            install(replace(original, **{field: tuple(i != side for i in range(2))}))
            assert not _assess().difference_bound_certified
        prefix = original.prefix_reports[side]
        for corrupt in (
            replace(prefix, full_state_tube_certified=False),
            replace(prefix, edge_weight_deviation_upper_bound=2 * WEIGHT_ERROR),
            replace(
                prefix,
                successor=replace(prefix.successor, conditional_image_certified=False),
            ),
            replace(
                prefix,
                successor=replace(
                    prefix.successor, new_weight_range=(Q(3, 4), Q(5, 4))
                ),
            ),
        ):
            install(
                replace(
                    original,
                    prefix_reports=tuple(
                        corrupt if i == side else original.prefix_reports[i]
                        for i in range(2)
                    ),
                )
            )
            assert not _assess().source_association_certified
    for changed in (
        replace(original, contact_rate=Q(0)),
        replace(original, contact_pairs=tuple((j, 18 + (j + 1) % 9) for j in range(9))),
        replace(original, form_error_bounds=(Q(0), ERROR)),
    ):
        install(changed)
        assert not _assess().difference_bound_certified
    # The common prefix is still a source obligation. Only the old future
    # short-contact tube and unrelated prediction verdicts are discarded.


def test_unavailable_geometry_is_not_repaired_by_cached_success(monkeypatch, report):
    original = report.geometry.successor
    monkeypatch.setattr(
        owner, "_assess_mixed_reciprocal_partner", lambda **kwargs: report.reference
    )
    for altered in (
        replace(original, held_capacities=(Q(2),) * 18),
        replace(
            original, marked_branch_turns=(Q(1, 2),) + original.marked_branch_turns[1:]
        ),
        replace(
            original,
            marked_pairs=tuple((i, _partners()[(i + 1) % 9]) for i in range(9)),
        ),
    ):
        monkeypatch.setattr(
            owner, "_derive_coupling_successor_geometry", lambda **kwargs: altered
        )
        fresh = owner._derive_registered_phase_probe_geometry()
        assert not fresh.certified
        assert not _assess().difference_bound_certified


def test_ready_age_existence_and_source_caps_do_not_authenticate_future_registration():
    for side in (0, 1):
        enlarged = tuple(
            DELTA / 8 + Q(1, 10**100) if i == side else DELTA / 8 for i in range(2)
        )
        waiting = _assess(starting_gap_upper_bounds=enlarged)
        assert (
            waiting.source_certified[side]
            and waiting.finite_protocol_ready_ages_exist[side]
        )
        assert not waiting.conditional_ready_sources[side] and not waiting.certified
        for field in ("form_error_bounds", "phase_error_bounds"):
            errors = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            unavailable = _assess(**{field: errors})
            assert not unavailable.source_certified[side] and not unavailable.certified
    zero = _assess(starting_gap_upper_bounds=(0, 0))
    assert zero.certified and zero.conditional_ready_sources == (True, True)
    assert zero.existence_gap_threshold == DELTA / 8 > 0
    assert not zero.all_window_registration_certified and not zero.intervention_admitted
    # Supplied zero is conditional readiness, never a finite exact-zero
    # acquisition theorem. Existence retains its fixed positive threshold.


def test_all_primitive_pairs_precede_source_geometry_or_component_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed primitive reached source or geometry work")

    for name in (
        "_assess_mixed_reciprocal_partner",
        "_derive_registered_phase_probe_geometry",
        "_derive_coupling_successor_geometry",
        "_support_components",
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
            for name in (
                "form_error_bounds",
                "phase_error_bounds",
                "starting_gap_upper_bounds",
            ):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(readout_error_bound=value)
    for name in (
        "form_error_bounds",
        "phase_error_bounds",
        "starting_gap_upper_bounds",
    ):
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
    with pytest.raises(ValueError):
        _assess(readout_error_bound=-Q(1, 10**100))


def test_one_shot_fixed_width_and_report_boundary_do_not_supply_a_control(report):
    def fixed(value):
        return Q(np.int64(Q(value).numerator), np.int64(Q(value).denominator))

    normalized = _assess(
        form_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_error_bounds=(fixed(ERROR) for _ in range(2)),
        starting_gap_upper_bounds=(fixed(DELTA / 8) for _ in range(2)),
        readout_error_bound=fixed(NOISE),
    )
    assert normalized == report
    with pytest.raises(FrozenInstanceError):
        report.all_window_registration_certified = True
    for name in (
        "reference",
        "initial_state",
        "actual_transfer",
        "comparator_transfer",
        "phase_input",
        "phase_jumps",
        "elapsed_time",
        "ready_ages",
        "registration_gap_bound",
        "all_window_registration_certified",
        "intervention_admitted",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
