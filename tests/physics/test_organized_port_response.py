"""Matched single-port laws and analytical whole-window response controls.

The acquired source theorem and the represented boundary fixtures are distinct.
No trajectory, source producer or committed Coupling event is evaluated here.
"""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from types import SimpleNamespace

import mpmath
import pytest

# Import only pure helpers, never the previous module's autouse fixtures.
from tests.operators.test_organized_coupling_port import (
    _constructed,
    _phases,
    _precommit,
)
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics._repeated_organized_composition import _bounded_sine_edge_fields
from tnfr.physics._sine_preparation import _prepared_form_costs
from tnfr.physics.relational_sine_regional import _branches

Q_SOURCE = Q(1, 10**70)
ERROR = Q(1, 10**10)
HORIZON = Q(1, 1000)
FORM_START = Q(1401, 10**6)
FORM_WINDOW = Q(1402, 10**6)
GAMMA_UPPER = Q(1, 3000)
FORM_RATE = 5 * FORM_WINDOW / 2 + 5 * GAMMA_UPPER / 4
PHASE_RATE = 5 * GAMMA_UPPER * FORM_WINDOW / 2
CASES = tuple(product((0, 1), (-1, 1), (-1, 1)))


@pytest.fixture(scope="module", autouse=True)
def no_committed_event_or_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("single-port response controls cannot execute events or flows")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return external._derive_organized_external_interaction()


@pytest.fixture(scope="module")
def source():
    return external._assess_organized_external_interaction(
        form_error_bounds=(ERROR,) * 6, phase_error_bounds=(ERROR,) * 6
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


@pytest.fixture(scope="module")
def paired(geometry):
    result = {}
    for origin, host_sign, donor_sign in CASES:
        connected = _constructed(geometry, origin, host_sign, donor_sign)
        control = deepcopy(connected)
        control.graph["UM_FUNCTIONAL_LINKS"] = False
        assert connected.graph.get("UM_FUNCTIONAL_LINKS", True) is True
        assert tuple(connected.nodes(data=True)) == tuple(control.nodes(data=True))
        assert tuple(connected.edges(data=True)) == tuple(control.edges(data=True))
        assert {
            key: value
            for key, value in connected.graph.items()
            if key != "UM_FUNCTIONAL_LINKS"
        } == {
            key: value
            for key, value in control.graph.items()
            if key != "UM_FUNCTIONAL_LINKS"
        }
        _, on, on_factors = _precommit(connected)
        _, off, off_factors = _precommit(control)
        assert on_factors == off_factors
        assert on.node_updates == off.node_updates
        assert len(on.edges) == 1 and off.edges == ()
        edge = on.edges[0]
        assert (edge.left, edge.right) == (18, 45)
        weight = Q(edge.weight)
        assert Q(5, 6) < weight <= 1
        old_edges = geometry.internal_edges + geometry.outer_edges
        result[origin, host_sign, donor_sign] = SimpleNamespace(
            graph=connected,
            control=control,
            on=on,
            off=off,
            factors=on_factors,
            weight=weight,
            old_edges=old_edges,
            joined_edges=old_edges + ((18, 45, weight),),
            form=tuple(Q(connected.nodes[node]["EPI"]) for node in range(54)),
            before_phase=tuple(Q(connected.nodes[node]["theta"]) for node in range(54)),
            phase=tuple(Q(_phases(connected, on)[node]) for node in range(54)),
        )
    return result


def _laplacian(edges):
    matrix = [[Q(0)] * 54 for _ in range(54)]
    for left, right, weight in edges:
        matrix[left][left] += weight
        matrix[right][right] += weight
        matrix[left][right] -= weight
        matrix[right][left] -= weight
    return tuple(tuple(row) for row in matrix)


def _fields(form, phase, edges, gamma):
    costs, gradient = _prepared_form_costs(form=form, weighted_edges=edges)
    # Raw lifted coordinates are already supplied: zero carrier avoids any
    # claim that these represented controls are exact acquired carrier states.
    current, potential, edge_current = _bounded_sine_edge_fields(
        phase_offsets=phase, weighted_edges=edges, alpha_bounds=I(0)
    )
    return SimpleNamespace(
        gradient=gradient,
        current=current,
        edge_current=edge_current,
        potential=potential,
        form_cost=sum(costs),
        storage=sum(costs) + sum(potential, I(0)),
        dx=tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current)),
        dt=tuple(gamma * value / 4 for value in gradient),
    )


@pytest.mark.parametrize("origin,host_sign,donor_sign", CASES)
def test_matched_proposals_retain_the_same_reset_and_the_actual_weight(
    paired, geometry, record_property, origin, host_sign, donor_sign
):
    pair = paired[origin, host_sign, donor_sign]
    assert pair.on.targets == pair.off.targets == (45,)
    assert pair.on.node_updates == pair.off.node_updates
    assert {update.node for update in pair.on.node_updates} == {45, 46, 53}
    assert all(
        update.vf_after is None and update.dnfr_after is None
        for update in pair.on.node_updates
    )
    assert _phases(pair.graph, pair.on) == _phases(pair.control, pair.off)
    assert any(pair.phase[node] != pair.before_phase[node] for node in (46, 53))
    assert pair.graph.number_of_edges() == pair.control.number_of_edges() == 99
    assert not pair.graph.has_edge(18, 45) and not pair.control.has_edge(18, 45)
    assert all(pair.graph.nodes[node]["nu_f"] == 1 for node in range(54))
    assert len(pair.joined_edges) == 100
    assert pair.weight != Q(1, 100000) and pair.weight != 1
    old, joined = _laplacian(pair.old_edges), _laplacian(pair.joined_edges)
    incidence = tuple(int(node == 18) - int(node == 45) for node in range(54))
    assert all(
        joined[i][j] - old[i][j] == pair.weight * incidence[i] * incidence[j]
        for i in range(54)
        for j in range(54)
    )
    assert tuple(old[i][i] for i in range(54)) == (4,) * 45 + (2,) * 9
    assert max(joined[i][i] for i in range(54)) == 4 + pair.weight <= 5
    # The complete old identities concern closed cycles, not a pair inventory.
    for cycle in geometry.internal_cycles + geometry.outer_cycles:
        raw = tuple(
            pair.phase[b] - pair.phase[a] for a, b in zip(cycle, cycle[1:] + cycle[:1])
        )
        assert sum(raw) == 0
        turns, gaps, winding, margin = _branches(tuple(I(value) for value in raw))
        assert gaps is not None and winding == -sum(turns) == 1 and margin > 0
    record_property("generated_weight", str(pair.weight))
    record_property("old_edges", 99)
    record_property("joined_edges", 100)
    record_property("retained_old_cycles", 15)


@pytest.mark.parametrize("origin", (0, 1))
def test_complete_asymmetric_rows_and_reciprocal_means_use_each_own_laplacian(
    paired, geometry, mp, origin
):
    pair = paired[origin, 1, 1]
    # An unrelated nonuniform static form control exercises every row. It is
    # not substituted into the acquired-source theorem or Coupling proposal.
    form = tuple(Q((7 * node + 3) % 19 - 9, 64) for node in range(54))
    phase = pair.phase
    gamma_mp = 1 / (1023 * mp.pi)
    for edges in (pair.old_edges, pair.joined_edges):
        field = _fields(form, phase, edges, geometry.gamma_bounds)
        laplacian = _laplacian(edges)
        expected_gradient = tuple(
            sum(laplacian[i][j] * form[j] for j in range(54)) for i in range(54)
        )
        assert field.gradient == expected_gradient
        assert (
            field.form_cost
            == sum(form[i] * expected_gradient[i] for i in range(54)) / 2
        )
        assert sum(expected_gradient) == 0
        currents = [mp.mpf(0)] * 54
        energy = _mp(mp, field.form_cost)
        # Independent high-precision directed adjacency sums check the shared
        # undirected field, including the actual non-unit port conductance.
        for node in range(54):
            for left, right, weight in edges:
                if node not in (left, right):
                    continue
                neighbor = right if node == left else left
                currents[node] += _mp(mp, weight) * mp.sin(
                    _mp(mp, phase[neighbor] - phase[node])
                )
        for left, right, weight in edges:
            energy += _mp(mp, weight) * (
                1 - mp.cos(_mp(mp, phase[right] - phase[left]))
            )
        dx = tuple(
            (-_mp(mp, grad) + gamma_mp * current) / 4
            for grad, current in zip(expected_gradient, currents)
        )
        dt = tuple(gamma_mp * _mp(mp, grad) / 4 for grad in expected_gradient)
        for node in range(54):
            _contains(mp, field.current[node], currents[node])
            _contains(mp, field.dx[node], dx[node])
            _contains(mp, field.dt[node], dt[node])
        _contains(mp, field.storage, energy)
        loss = -sum(grad**2 for grad in expected_gradient) / 4
        direct = sum(
            _mp(mp, grad) * x - current * theta
            for grad, current, x, theta in zip(expected_gradient, currents, dx, dt)
        )
        assert abs(direct - _mp(mp, loss)) < mp.mpf("1e-160")
        assert loss < 0
        # For every independent edge current, its two incidence coefficients
        # cancel globally; donor/host coefficients vanish except at the port.
        for left, right, weight in edges:
            donor_coefficient = int(left >= 45) - int(right >= 45)
            host_coefficient = int(left < 45) - int(right < 45)
            assert donor_coefficient + host_coefficient == 0
            assert donor_coefficient == (-1 if (left, right) == (18, 45) else 0)
        attached = edges is pair.joined_edges
        coefficient = pair.weight if attached else Q(0)
        delta = form[18] - form[45]
        angle = _mp(mp, phase[45] - phase[18])
        donor_dx = (
            _mp(mp, coefficient) * (_mp(mp, delta) - gamma_mp * mp.sin(angle)) / 36
        )
        donor_dt = -_mp(mp, coefficient) * gamma_mp * _mp(mp, delta) / 36
        assert abs(sum(dx[45:]) / 9 - donor_dx) < mp.mpf("1e-160")
        assert abs(sum(dt[45:]) / 9 - donor_dt) < mp.mpf("1e-160")
        assert abs(sum(dx[:45]) / 45 + donor_dx / 5) < mp.mpf("1e-160")
        assert abs(sum(dt[:45]) / 45 + donor_dt / 5) < mp.mpf("1e-160")
        assert abs(sum(dx)) < mp.mpf("1e-160")
        assert abs(sum(dt)) < mp.mpf("1e-160")
        # Uniform quarter mobility must survive the changed degree at node18.
        if attached:
            assert abs(dx[18] - 4 * dx[18] / _mp(mp, laplacian[18][18])) > mp.mpf(
                "1e-8"
            )


@pytest.mark.parametrize("origin", (0, 1))
def test_reset_work_and_attachment_work_are_separate_on_the_same_primary_state(
    paired, geometry, mp, record_property, origin
):
    pair = paired[origin, 1, 1]
    before = _fields(
        pair.form, pair.before_phase, pair.old_edges, geometry.gamma_bounds
    )
    reset = _fields(pair.form, pair.phase, pair.old_edges, geometry.gamma_bounds)
    joined = _fields(pair.form, pair.phase, pair.joined_edges, geometry.gamma_bounds)
    changed = tuple(
        (left, right)
        for left, right, _ in pair.old_edges
        if {left, right} & {45, 46, 53}
    )
    assert set(map(frozenset, changed)) == {
        frozenset(edge) for edge in ((45, 46), (46, 47), (52, 53), (53, 45))
    }
    reset_exact = sum(
        mp.cos(_mp(mp, pair.before_phase[b] - pair.before_phase[a]))
        - mp.cos(_mp(mp, pair.phase[b] - pair.phase[a]))
        for a, b in changed
    )
    added_exact = _mp(mp, pair.weight) * (
        _mp(mp, pair.form[18] - pair.form[45]) ** 2 / 2
        + 1
        - mp.cos(_mp(mp, pair.phase[45] - pair.phase[18]))
    )
    _contains(mp, reset.storage - before.storage, reset_exact)
    _contains(mp, joined.storage - reset.storage, added_exact)
    _contains(mp, joined.storage - before.storage, reset_exact + added_exact)
    assert reset_exact > 0 and added_exact > 0
    assert before.form_cost == reset.form_cost
    assert joined.form_cost - reset.form_cost == (
        pair.weight * (pair.form[18] - pair.form[45]) ** 2 / 2
    )
    # A reset-only control is not stationary merely because donor means stay
    # constant: its fresh current has nonzero entries after the phase reset.
    assert any(box.lo > 0 or box.hi < 0 for box in reset.current[45:])
    record_property("reset_work_lower", str((reset.storage - before.storage).lo))
    record_property("attachment_work_lower", str((joined.storage - reset.storage).lo))


def test_full_source_history_and_global_form_majorant_are_retained(source):
    assert all(source.source_admission_flags)
    assert source.relative_preparation_dimension == 96
    assert source.form_error_bounds == source.phase_error_bounds == (ERROR,) * 6
    host, donor, protocol = (
        source.host_acquisition,
        source.donor_recovery,
        source.observation_protocol,
    )
    assert host.full_form_error_upper_bound < 4 * Q_SOURCE
    assert host.full_phase_error_upper_bound < 7 * Q_SOURCE
    assert donor.endpoint_form_norm_squared_upper_bound < Q_SOURCE**2
    assert donor.endpoint_phase_norm_squared_upper_bound < Q_SOURCE**2
    assert protocol.marked_child_order == tuple(range(5))
    assert protocol.event_multipliers == (1, -2, 1)
    assert protocol.window_command_multipliers == (0, 1, -1, 0)
    assert protocol.total_elapsed_scaled_time == Q(1, 5000)
    initial = (
        max(box.abs_max for box in host.primitive_entry.endpoint_form_bounds)
        + host.full_form_error_upper_bound
    )
    postword = (
        initial + source.geometry.gamma_bounds.hi * protocol.total_elapsed_scaled_time
    )
    assert postword < FORM_START
    assert Q(1, 3300) < source.geometry.gamma_bounds.lo
    assert source.geometry.gamma_bounds.hi < GAMMA_UPPER
    # Heat contraction is in the max norm for either symmetric weighted
    # Laplacian. The sine forcing has magnitude at most degree*gamma/4.
    assert FORM_START + 5 * GAMMA_UPPER * HORIZON / 4 < FORM_WINDOW
    assert FORM_RATE == (2 * 5 * FORM_WINDOW + 5 * GAMMA_UPPER) / 4
    assert PHASE_RATE == GAMMA_UPPER * (2 * 5 * FORM_WINDOW) / 4
    assert 2 * PHASE_RATE * HORIZON < Q(1, 10**8)


def test_whole_window_old_cycle_bounds_follow_the_acquired_source_and_reset(source):
    pi, host = pi_interval(), source.host_acquisition
    phase_error = (
        host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    )
    offsets = tuple(
        box + I(-phase_error, phase_error)
        for box in host.primitive_entry.endpoint_phase_bounds
    )
    assert offsets == source.outer_one_host_phase_offset_bounds
    for child, box in enumerate(offsets):
        assert (box - 4 * (child - 2) / pi).abs_max < Q(1, 40000)
    alpha = 2 * pi / 9
    host_internal = alpha + I(-Q(1, 20000), Q(1, 20000))
    outer = (
        4 / pi + I(-Q(1, 20000), Q(1, 20000)),
        2 * pi - 16 / pi + I(-Q(1, 20000), Q(1, 20000)),
    )
    donor_min = 3 * alpha.lo / 4 - 4 * Q_SOURCE
    donor_max = 5 * alpha.hi / 4 + 4 * Q_SOURCE
    assert host_internal.lo > Q(1, 2) and host_internal.hi < 1
    assert all(box.lo > 1 and box.hi < Q(13, 10) for box in outer)
    assert donor_min > Q(1, 2) and donor_max < 1
    drift = 2 * PHASE_RATE * HORIZON
    assert min(host_internal.lo, *(box.lo for box in outer), donor_min) - drift > Q(
        2, 5
    )
    assert max(host_internal.hi, *(box.hi for box in outer), donor_max) + drift < Q(
        7, 5
    )
    assert Q(7, 5) < pi.lo / 2
    assert len(source.geometry.internal_cycles) == 6
    assert len(source.geometry.outer_cycles) == 9
    # Every oriented old gap stays in (0,pi/2), hence never reaches a cut;
    # continuity preserves each of the already established winding-one cycles.


@pytest.mark.parametrize("origin", (0, 1))
def test_whole_window_port_current_has_a_strict_noise_resolved_sign(
    source, record_property, origin
):
    pi = pi_interval()
    host18 = source.outer_one_host_phase_offset_bounds[2]
    beta = I(Q(3, 5)) if origin == 0 else 2 * pi / 5 - Q(3, 5)
    origin_box = beta + I(-Q(1, 1000), Q(1, 1000))
    initial_gap = origin_box - host18 + I(-2 * Q_SOURCE, 2 * Q_SOURCE)
    assert initial_gap.lo > Q(59, 100) and initial_gap.hi < Q(7, 10)
    drift = 2 * PHASE_RATE * HORIZON
    whole_gap = initial_gap + I(-drift, drift)
    assert whole_gap.lo > Q(1, 2) and whole_gap.hi < 1
    assert sin(I(Q(1, 2), 1)).lo > Q(2, 5)
    # The preceding center-word theorem supplies the initial form difference;
    # here both complete post-event rows contribute to its whole-window drift.
    form_difference = Q(2, 10**10) + Q_SOURCE + 2 * FORM_RATE * HORIZON
    assert form_difference < Q(1, 100000)
    boundary_bracket = form_difference - Q(1, 3300) * Q(2, 5)
    assert boundary_bracket < -Q(1, 10000)
    donor_rate_upper = Q(5, 6) * (-Q(1, 10000)) / 36
    assert donor_rate_upper == -Q(1, 432000)
    increment_upper = donor_rate_upper * HORIZON
    assert increment_upper == -Q(1, 432000000)
    observed_upper = increment_upper + 4 * ERROR
    assert observed_upper < -Q(1, 10**9)
    record_property("horizon_tau", str(HORIZON))
    record_property("port_phase_drift_upper", str(drift))
    record_property("true_increment_strict_upper", str(increment_upper))
    record_property("observed_increment_strict_upper", str(observed_upper))


def test_source_reset_work_budget_keeps_every_endpoint_perturbation(mp):
    pi = pi_interval()
    alpha, gain = 2 * pi / 9, 1 / (pi + 1)
    assert 0 < gain.lo < gain.hi < Q(1, 4)
    nominal_work = 4 * cos(alpha) * (1 - cos(gain * alpha))
    assert nominal_work.lo > 22 * Q_SOURCE
    exact_alpha, exact_gain = 2 * mp.pi / 9, 1 / (mp.pi + 1)
    direct_work = 2 * (mp.cos(exact_alpha) - mp.cos((1 - exact_gain) * exact_alpha))
    direct_work += 2 * (mp.cos(exact_alpha) - mp.cos((1 + exact_gain) * exact_alpha))
    _contains(mp, nominal_work, direct_work)
    assert abs(
        direct_work - 4 * mp.cos(exact_alpha) * (1 - mp.cos(exact_gain * exact_alpha))
    ) < mp.mpf("1e-160")
    assert (1 + 2 * cos(alpha)).lo > 2
    assert 3 * Q_SOURCE / (2 - 3 * Q_SOURCE) < 2 * Q_SOURCE
    changed = ((45, 46), (46, 47), (52, 53), (53, 45))
    before_error = {node: Q_SOURCE for node in range(45, 54)}
    after_error = before_error | {node: 2 * Q_SOURCE for node in (45, 46, 53)}
    # Unit Lipschitz cosine retains both endpoint errors on all four changed
    # edges before and after the ideal source reset, including untouched nodes.
    error = sum(
        before_error[a] + before_error[b] + after_error[a] + after_error[b]
        for a, b in changed
    )
    assert error == 22 * Q_SOURCE
    # Adding the port has its own positive phase potential; no continuous
    # dissipation or previously paid preparation cost is used as a reservoir.
    assert (1 - cos(I(Q(59, 100), Q(7, 10)))).lo > 0


def test_four_raw_reading_errors_do_not_cancel_with_equal_exact_start_means():
    # (on_end-on_start)-(off_end-off_start): all correlations are allowed.
    errors = tuple(
        end_on - start_on - end_off + start_off
        for end_on, start_on, end_off, start_off in product((-ERROR, ERROR), repeat=4)
    )
    assert min(errors) == -4 * ERROR and max(errors) == 4 * ERROR
    assert -Q(1, 432000000) + max(errors) < -Q(1, 10**9)
