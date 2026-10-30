"""Independent full-state controls for conditional Coupling successors.

Static generic law states and coefficient-family members are not acquired
endpoints or actual proposed weights. No trajectory, event, root search or earlier
scientific producer is evaluated. Convergence is analytical and asymptotic;
these tests supply no finite arrival time.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import networkx as nx
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _sine_coupling_successor as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
RADIUS = Q(1, 32)
COMPLETIONS = ("common_sense", "split_sense")
LAWS = ("bare_degree", "transport_strength")


def _assess(**changes):
    arguments = dict(form_error_bound=ERROR, phase_error_bound=ERROR)
    arguments.update(changes)
    return owner._assess_sine_coupling_successor(**arguments)


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert _mp(mp, interval.width) < mp.mpf("1e-30")


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support(completion):
    """Generate all pairs by circular phase bins, independently of chord code."""
    bins = tuple(int((9 * value) % 9) for value in _turns())
    pairs = set()
    for i in range(18):
        for j in range(i + 1, 18):
            distance = min((bins[i] - bins[j]) % 9, (bins[j] - bins[i]) % 9)
            if i // 9 == j // 9:
                accepted = distance in (1, 2)
            else:
                accepted = completion == COMPLETIONS[0] and distance <= 2
            if accepted:
                pairs.add((i, j))
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in range(2)
            for j in range(9)
        )
    )
    new = tuple(sorted(pairs - set(old)))
    return old, new


def _weights(completion):
    return tuple(
        Q(3, 4) + Q(index % 5, 16) for index in range(len(_support(completion)[1]))
    )


def _edges(completion, weights=None):
    old, new = _support(completion)
    weights = _weights(completion) if weights is None else weights
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, value) for (a, b), value in zip(new, weights, strict=True)
    )


def _laplacian(edges, size=18):
    rows = [[0 for _ in range(size)] for _ in range(size)]
    for i, j, weight in edges:
        rows[i][i] += weight
        rows[j][j] += weight
        rows[i][j] -= weight
        rows[j][i] -= weight
    return tuple(tuple(row) for row in rows)


def _matvec(matrix, values):
    return tuple(sum(a * b for a, b in zip(row, values, strict=True)) for row in matrix)


def _product(first, second):
    return tuple(
        tuple(
            sum(a * b for a, b in zip(row, column, strict=True))
            for column in zip(*second)
        )
        for row in first
    )


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _state(mp, forms, errors=None, gap=Q(0)):
    errors = (Q(0),) * 18 if errors is None else errors
    phase = tuple(
        2 * mp.pi * _mp(mp, turn)
        + _mp(mp, error)
        + (-1 if i < 9 else 1) * _mp(mp, gap) / 2
        for i, (turn, error) in enumerate(zip(_turns(), errors, strict=True))
    )
    return tuple(_mp(mp, value) for value in forms), phase


def _interval_state(forms, errors=None, gap=Q(0)):
    errors = (Q(0),) * 18 if errors is None else errors
    phases = tuple(
        2 * pi_interval() * turn + error + (-1 if i < 9 else 1) * gap / 2
        for i, (turn, error) in enumerate(zip(_turns(), errors, strict=True))
    )
    return tuple(I(value) for value in forms) + phases


def _edge_oracle(mp, completion, x, theta, weights=None):
    gradient, sine, weighted_sine = ([mp.mpf(0)] * 18 for _ in range(3))
    strengths = [Q(0)] * 18
    energy = mp.mpf(0)
    for i, j, raw_weight in _edges(completion, weights):
        weight = _mp(mp, raw_weight)
        difference = x[i] - x[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        current = mp.sin(theta[j] - theta[i])
        sine[i] += current
        sine[j] -= current
        weighted_sine[i] += weight * current
        weighted_sine[j] -= weight * current
        strengths[i] += raw_weight
        strengths[j] += raw_weight
        energy += weight * difference**2 / 2 + 1 - mp.cos(theta[j] - theta[i])
    return tuple(gradient), tuple(sine), tuple(weighted_sine), tuple(strengths), energy


def _rows(mp, completion, law, x, theta, weights=None):
    q, sine, _, strengths, _ = _edge_oracle(mp, completion, x, theta, weights)
    degree = 9 if completion == COMPLETIONS[0] else 4
    gamma = 1 / (1023 * mp.pi)
    loss = strengths if law == LAWS[1] else (Q(degree),) * 18
    return (
        tuple(
            -value / _mp(mp, denominator) + gamma * current / degree
            for value, denominator, current in zip(q, loss, sine, strict=True)
        ),
        tuple(gamma * value / degree for value in q),
    )


def _components(completion):
    return (
        (tuple(range(18)),)
        if completion == COMPLETIONS[0]
        else (tuple(range(9)), tuple(range(9, 18)))
    )


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("successor controls cannot execute events or evolve sources")

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
        # Grant only the shared static row factory, unwinding it before the
        # outer scientific guards. Its returned field never advances a state.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def fields():
    return {
        (completion, law): owner._coupling_successor_field(
            completion=completion,
            new_edge_conductances=_weights(completion),
            loss_normalization=law,
        )
        for completion in COMPLETIONS
        for law in LAWS
    }


def test_full_support_and_phase_index_blocks_are_independent(report, record_property):
    bins = tuple(int((9 * value) % 9) for value in _turns())
    order = tuple(
        next(i for i in range(9 * r, 9 * r + 9) if bins[i] == b)
        for r in range(2)
        for b in range(9)
    )
    base_edges = tuple(
        (i, j, Q(1))
        for i in range(9)
        for j in range(i + 1, 9)
        if min((j - i) % 9, (i - j) % 9) in (1, 2)
    )
    base = _laplacian(base_edges, 9)
    for completion_report in report.completion_reports:
        geometry = completion_report.geometry
        common = geometry.completion == COMPLETIONS[0]
        old, new = _support(geometry.completion)
        assert geometry.nodes == tuple(range(18))
        assert geometry.internal_edges == tuple((a, b, Q(1)) for a, b in old)
        assert geometry.new_edge_pairs == new
        assert geometry.edge_pairs == old + new
        assert geometry.target_phase_turns == _turns()
        assert geometry.phase_indices == bins
        assert geometry.held_capacities == (Q(1),) * 18
        graph = nx.Graph(geometry.edge_pairs)
        assert graph.number_of_edges() == (81 if common else 36)
        assert nx.number_connected_components(graph) == (1 if common else 2)
        assert tuple(dict(graph.degree()).values()) == (9 if common else 4,) * 18
        assert len(nx.cycle_basis(graph)) == (64 if common else 20)
        matrix = _laplacian(tuple((a, b, Q(1)) for a, b in old + new))
        rows = tuple(tuple(matrix[i][j] for j in order) for i in order)
        for i in range(9):
            for j in range(9):
                if common:
                    assert rows[i][j] + rows[i][9 + j] == 2 * base[i][j]
                    assert rows[i][j] - rows[i][9 + j] == 10 * int(i == j)
                    assert rows[9 + i][9 + j] == rows[i][j]
                    assert rows[9 + i][j] == rows[i][9 + j]
                else:
                    assert rows[i][j] == rows[9 + i][9 + j] == base[i][j]
                    assert rows[i][9 + j] == rows[9 + i][j] == 0
        assert geometry.support_certified and geometry.reference_stationary_certified
        record_property(geometry.completion + "_edges", graph.number_of_edges())
        record_property(geometry.completion + "_cycle_rank", len(nx.cycle_basis(graph)))


def test_exact_spectral_and_curvature_margins_do_not_use_a_sampled_spectrum(mp, report):
    alpha = 2 * mp.pi / 9
    identities = (
        mp.cos(mp.pi / 9),
        -mp.cos(2 * mp.pi / 9),
        -mp.mpf(1),
        -mp.cos(4 * mp.pi / 9),
    )
    for m, expected in enumerate(identities, 1):
        assert abs(mp.cos(m * alpha) + mp.cos(2 * m * alpha) - expected) < mp.mpf(
            "1e-190"
        )
        assert expected < 1
    for row in report.completion_reports:
        geometry = row.geometry
        assert geometry.spectral_gap_certified
        for m, bounds in enumerate(geometry.internal_eigenvalue_bounds, 1):
            value = 4 - 2 * mp.cos(m * alpha) - 2 * mp.cos(2 * m * alpha)
            _contains(mp, bounds, value)
            assert bounds.lo > 2
        common = geometry.completion == COMPLETIONS[0]
        assert geometry.bare_gap_lower_bound == (4 if common else 2)
        assert geometry.weighted_gap_lower_bound == (3 if common else Q(3, 2))
    _contains(mp, report.phase_curvature_bounds, mp.cos(2 * alpha + _mp(mp, Q(1, 16))))
    assert report.phase_curvature_margin > 0
    assert report.phase_cosine_lower_bound == Q(1, 10)
    # sin(pi/18-1/16)>sin(5/48)>1/10, using pi>3 and its cubic lower bound.
    assert Q(5, 48) - Q(5, 48) ** 3 / 6 > Q(1, 10)


def test_stationary_reference_uses_all_rows_and_each_component_gauge(mp, fields):
    for completion in COMPLETIONS:
        forms = (
            (Q(-1, 8),) * 18
            if completion == COMPLETIONS[0]
            else (Q(-1, 8),) * 9 + (Q(1, 4),) * 9
        )
        gap = Q(0) if completion == COMPLETIONS[0] else DELTA
        x, theta = _state(mp, forms, gap=gap)
        interval_state = _interval_state(forms, gap=gap)
        q, sine, _, _, _ = _edge_oracle(mp, completion, x, theta)
        assert max(abs(value) for value in q + sine) < mp.mpf("1e-190")
        for law in LAWS:
            actual = fields[completion, law](interval_state)
            assert len(actual) == 36
            for value in actual:
                _contains(mp, value, mp.mpf(0))
        for ring, winding in ((0, 1), (1, 2)):
            turns = _turns()[9 * ring : 9 * ring + 9]
            principal = tuple(
                ((turns[(j + 1) % 9] - turns[j] + Q(1, 2)) % 1) - Q(1, 2)
                for j in range(9)
            )
            assert sum(principal) == winding
            assert max(abs(value) for value in principal) < Q(1, 4)


def test_complete_rows_keep_weighted_gradient_bare_sine_and_declared_clock(mp, fields):
    forms = tuple(Q(-1, 4) + Q(i - 8, 2048) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 2048) for i in range(18))
    x, theta = _state(mp, forms, errors)
    state = _interval_state(forms, errors)
    for completion in COMPLETIONS:
        q, sine, weighted_sine, strengths, _ = _edge_oracle(mp, completion, x, theta)
        assert max(abs(a - b) for a, b in zip(sine, weighted_sine)) > 0
        degree = 9 if completion == COMPLETIONS[0] else 4
        assert any(value != degree for value in strengths)
        for law in LAWS:
            expected = sum(_rows(mp, completion, law, x, theta), ())
            actual = fields[completion, law](state)
            for bounds, value in zip(actual, expected, strict=True):
                _contains(mp, bounds, value)
            # Exact component translations preserve edge differences. These
            # are law symmetries, not source transformations used for admission.
            shifted = tuple(
                value + (Q(1, 2) if i < 18 else Q(1, 16))
                for i, value in enumerate(state)
            )
            assert fields[completion, law](shifted) == actual
        first = fields[completion, LAWS[0]](state)
        second = fields[completion, LAWS[1]](state)
        assert first[18:] == second[18:]
        assert any(a != b for a, b in zip(first[:18], second[:18]))


def test_shared_directed_gradient_and_interval_input_are_retained(
    mp, fields, monkeypatch
):
    forms = tuple(Q((i % 7) - 3, 1024) for i in range(18))
    errors = tuple(Q((i % 3) - 1, 2048) for i in range(18))
    original = owner._laplacian
    calls = []

    def checked(edges, values):
        edges = tuple(edges)
        calls.append(edges)
        return original(edges, values)

    monkeypatch.setattr(owner, "_laplacian", checked)
    radius = Q(1, 10**12)
    for completion in COMPLETIONS:
        state = list(_interval_state(forms, errors))
        state[0] += I(-radius, radius)
        actual = fields[completion, LAWS[1]](state)
        expected_edges = _edges(completion)
        assert calls[-1] == tuple(
            row for a, b, w in expected_edges for row in ((a, b, w), (b, a, w))
        )
        for sign in (-1, 1):
            changed = (forms[0] + sign * radius,) + forms[1:]
            x, theta = _state(mp, changed, errors)
            expected = sum(_rows(mp, completion, LAWS[1], x, theta), ())
            for interval, value in zip(actual, expected, strict=True):
                assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_jets_enclose_independent_directional_derivatives_of_both_rows(mp, fields):
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    a = tuple(Q((i % 3) - 1, 32) for i in range(18))
    b = tuple(Q((i % 4) - 1, 64) for i in range(18))
    state = _interval_state(forms, errors)
    jets = tuple(
        Jet((value, I(direction)))
        for value, direction in zip(state, a + b, strict=True)
    )
    x, theta = _state(mp, forms, errors)
    gamma = 1 / (1023 * mp.pi)
    for completion in COMPLETIONS:
        edges = _edges(completion)
        qa = _matvec(_laplacian(edges), a)
        ds = [mp.mpf(0)] * 18
        for i, j, _ in edges:
            current = mp.cos(theta[j] - theta[i]) * _mp(mp, b[j] - b[i])
            ds[i] += current
            ds[j] -= current
        strengths = _edge_oracle(mp, completion, x, theta)[3]
        degree = 9 if completion == COMPLETIONS[0] else 4
        for law in LAWS:
            loss = strengths if law == LAWS[1] else (Q(degree),) * 18
            derivative = tuple(
                -_mp(mp, value / denominator) + gamma * current / degree
                for value, denominator, current in zip(qa, loss, ds, strict=True)
            )
            derivative += tuple(gamma * _mp(mp, value) / degree for value in qa)
            constants = sum(_rows(mp, completion, law, x, theta), ())
            for jet, constant, first in zip(
                fields[completion, law](jets), constants, derivative, strict=True
            ):
                assert isinstance(jet, Jet) and jet.order == 1
                _contains(mp, jet.coeffs[0], constant)
                _contains(mp, jet.coeffs[1], first)


def test_common_initial_registration_has_live_phase_feedback_not_a_static_hold(
    mp, fields
):
    forms = (Q(0),) * 18
    x, theta = _state(mp, forms, gap=DELTA)
    state = _interval_state(forms, gap=DELTA)
    gamma = 1 / (1023 * mp.pi)
    gain = 1 + 2 * mp.cos(2 * mp.pi / 9) + 2 * mp.cos(4 * mp.pi / 9)
    forcing = gamma * gain * mp.sin(_mp(mp, DELTA)) / 9
    z = (Q(1),) * 9 + (Q(-1),) * 9
    gradient = _matvec(_laplacian(_edges(COMPLETIONS[0])), z)
    jets = tuple(
        Jet((value, I(direction)))
        for value, direction in zip(state, z + (Q(0),) * 18, strict=True)
    )
    assert forcing > 0 and any(value != 0 for value in gradient)
    for law in LAWS:
        actual = fields[COMPLETIONS[0], law](jets)
        for i in range(18):
            _contains(mp, actual[i].coeffs[0], forcing * _mp(mp, z[i]))
            _contains(mp, actual[18 + i].coeffs[0], mp.mpf(0))
            _contains(mp, actual[18 + i].coeffs[1], gamma * _mp(mp, gradient[i]) / 9)
    # The actual initial phase acceleration is forcing times this nonzero
    # directional response. No time integration or arrival time is inferred.
    assert max(abs(forcing * gamma * _mp(mp, value) / 9) for value in gradient) > 0


def test_storage_loss_and_inventory_are_full_state_without_commutation(mp):
    errors = tuple(Q((i % 5) - 2, 2048) for i in range(18))
    _, reference = _state(mp, (Q(0),) * 18)
    gamma = 1 / (1023 * mp.pi)
    for completion in COMPLETIONS:
        exact_strengths = [Q(0)] * 18
        for i, j, weight in _edges(completion):
            exact_strengths[i] += weight
            exact_strengths[j] += weight
        mobility = tuple(1 / strength for strength in exact_strengths)
        # This source-independent probe gives sum(x')=-B^T L_W B/1024<0
        # under strength loss. Its common form is not translated away.
        forms = tuple(Q(-1, 8) + value / 1024 for value in mobility)
        x, theta = _state(mp, forms, errors)
        q, sine, _, strengths, energy = _edge_oracle(mp, completion, x, theta)
        assert strengths == tuple(exact_strengths)
        degree = 9 if completion == COMPLETIONS[0] else 4
        y = [_mp(mp, value) for value in errors]
        v = list(x)
        for component in _components(completion):
            mx = sum(x[i] for i in component) / len(component)
            my = sum(y[i] for i in component) / len(component)
            for i in component:
                v[i] -= mx
                y[i] -= my
        secant_edges = []
        for i, j, _ in _edges(completion):
            difference = y[j] - y[i]
            base = reference[j] - reference[i]
            coefficient = (
                (mp.sin(base + difference) - mp.sin(base)) / difference
                if difference
                else mp.cos(base)
            )
            assert coefficient > mp.mpf("0.1")
            secant_edges.append((i, j, coefficient))
        k = _laplacian(secant_edges)
        ky = _matvec(k, y)
        assert max(abs(a + b) for a, b in zip(ky, sine)) < mp.mpf("1e-190")
        weighted = _laplacian(
            tuple((i, j, _mp(mp, w)) for i, j, w in _edges(completion))
        )
        hk, kh = _product(weighted, k), _product(k, weighted)
        assert any(
            abs(hk[i][j] - kh[i][j]) > mp.mpf("1e-100")
            for i in range(18)
            for j in range(18)
        )
        baseline = _edge_oracle(mp, completion, (mp.mpf(0),) * 18, reference)[-1]
        norm = _dot(v, v) + _dot(y, y)
        coercivity = Q(1, 5) if completion == COMPLETIONS[0] else Q(1, 10)
        assert energy - baseline >= _mp(mp, coercivity) * norm
        for law in LAWS:
            dx, dt = _rows(mp, completion, law, x, theta)
            loss = strengths if law == LAWS[1] else (Q(degree),) * 18
            derivative = _dot(q, dx) - _dot(sine, dt)
            expected = -sum(value**2 / _mp(mp, d) for value, d in zip(q, loss))
            assert abs(derivative - expected) < mp.mpf("1e-190") and expected < 0
            for component in _components(completion):
                assert abs(sum(dt[i] for i in component)) < mp.mpf("1e-190")
                inventory_rate = sum(
                    dx[i] + degree / gamma * dt[i] / _mp(mp, loss[i]) for i in component
                )
                assert abs(inventory_rate) < mp.mpf("1e-190")
                if law == LAWS[0]:
                    assert abs(sum(dx[i] for i in component)) < mp.mpf("1e-190")
        expected_mean_rate = (
            -_dot(mobility, _matvec(_laplacian(_edges(completion)), mobility)) / 1024
        )
        assert expected_mean_rate < 0
        actual_mean_rate = sum(_rows(mp, completion, LAWS[1], x, theta)[0])
        assert abs(actual_mean_rate - _mp(mp, expected_mean_rate)) < mp.mpf("1e-190")


def test_unit_coefficient_family_makes_both_complete_laws_identical():
    state = _interval_state(tuple(Q(i - 8, 2048) for i in range(18)))
    for completion in COMPLETIONS:
        weights = (Q(1),) * len(_support(completion)[1])
        degree = owner._coupling_successor_field(
            completion=completion, new_edge_conductances=weights
        )
        strength = owner._coupling_successor_field(
            completion=completion,
            new_edge_conductances=weights,
            loss_normalization=LAWS[1],
        )
        assert degree(state) == strength(state)
    # This enclosing-family control is not an attained proposal or a reason
    # to replace the actual image's distinct strength-law common-form limit.


def test_source_image_traps_and_registration_are_all_time_without_deadline(
    report, record_property
):
    assert report.source_certified and report.isolated_waiting_certified
    assert report.geometry_certified and report.conditional_image_certified
    assert (
        report.certified
        and report.status == "certified"
        and not report.unresolved_conditions
    )
    assert report.waiting_radius == ETA and report.phase_reset_error_bound == 3 * ETA
    assert report.phase_radius == RADIUS and report.new_weight_range == (Q(3, 4), Q(1))
    assert tuple(
        row.target_winding for row in report.precontact_reference.recoveries
    ) == (1, 2)
    for row in report.completion_reports:
        common = row.geometry.completion == COMPLETIONS[0]
        phase = (Q(9, 2) * DELTA**2 + 54 * DELTA * ETA if common else 0) + 162 * ETA**2
        energy = (
            Q(27, 2) * DELTA**2 + 486 * ETA if common else 216 * ETA
        ) + 18 * ETA**2
        assert row.initial_phase_norm_squared_upper_bound == phase
        assert row.initial_quotient_norm_squared_upper_bound == phase + 2 * ETA**2
        assert row.initial_phase_radius_margin == RADIUS**2 - phase > 0
        assert (
            row.initial_quotient_radius_margin == RADIUS**2 - (phase + 2 * ETA**2) > 0
        )
        assert row.initial_energy_upper_bound == energy
        assert row.storage_barrier == (Q(1, 5120) if common else Q(1, 10240))
        assert row.storage_margin == row.storage_barrier - energy > 0
        assert row.initial_registration_bounds == (DELTA - 6 * ETA, DELTA + 6 * ETA)
        assert row.initial_registration_bounds[0] > 7 * DELTA / 8
        assert (
            row.constituent_identity_certified and row.asymptotic_convergence_certified
        )
        assert row.joint_registration_certified is common
        assert row.disconnected_registration_certified is (not common)
        assert row.registration_limit_bounds == (
            (Q(0), Q(0)) if common else row.initial_registration_bounds
        )
        assert row.degree_form_limit_bounds == ((Q(0), Q(0)),) * len(
            row.geometry.components
        )
        if not common:
            assert row.strength_form_limit_bounds is None
        record_property(row.geometry.completion + "_energy_upper", str(energy))
        record_property(
            row.geometry.completion + "_storage_margin", str(row.storage_margin)
        )
    assert (
        report.connected_joint_registration_certified
        and report.disconnected_nonregistration_certified
    )
    for name in ("deadline", "recovery_duration", "decay_rate", "elapsed_time"):
        assert not hasattr(report, name)
    record_property("finite_arrival_deadline_asserted", False)


def test_nominal_strength_limit_follows_actual_full_weights_and_inventory(
    mp, report, record_property
):
    zero = (Q(0),) * 18
    _, theta = _state(mp, zero, gap=DELTA)
    _, reference = _state(mp, zero)
    old, new = _support(COMPLETIONS[0])
    strengths = [mp.mpf(2)] * 18
    for i, j in new:
        gap = (theta[j] - theta[i] + mp.pi) % (2 * mp.pi) - mp.pi
        weight = 1 - abs(gap) / (2 * mp.pi)
        assert mp.mpf("0.75") < weight <= 1
        strengths[i] += weight
        strengths[j] += weight
    expected = (
        mp.mpf(71) / 9 - _mp(mp, DELTA) / (2 * mp.pi),
        mp.mpf(73) / 9 - _mp(mp, DELTA) / (2 * mp.pi),
    )
    for i, value in enumerate(strengths):
        assert abs(value - expected[i // 9]) < mp.mpf("1e-190")
    for interval, value in zip(report.nominal_strength_bounds, expected, strict=True):
        _contains(mp, interval, value)
        assert Q(31, 4) < interval.lo <= interval.hi < Q(33, 4)
    gamma = 1 / (1023 * mp.pi)
    inventory_limit = (
        9
        / (18 * gamma)
        * sum((p - t) / s for p, t, s in zip(theta, reference, strengths, strict=True))
    )
    closed = -_mp(mp, DELTA) / (2 * gamma * expected[0] * expected[1])
    assert abs(inventory_limit - closed) < mp.mpf("1e-190")
    _contains(mp, report.nominal_common_strength_form_limit_bounds, closed)
    assert -_mp(mp, Q(7, 256)) < closed < -_mp(mp, Q(5, 256))
    assert report.normalization_common_form_separation_certified
    lower, upper = report.common_strength_form_limit_bounds
    assert -Q(1, 32) < lower <= upper < -Q(1, 64)
    assert report.completion_reports[0].strength_form_limit_bounds == ((lower, upper),)
    record_property("connected_strength_form_limit_lower", str(lower))
    record_property("connected_strength_form_limit_upper", str(upper))
    record_property("connected_degree_form_limit", 0)


def test_actual_image_weight_mobility_and_gauge_errors_preserve_limit_margin(report):
    epsilon = Q(_coupling_stage_kernel._EPI_SIMILARITY_EPSILON)
    assert epsilon > Q(9, 10**10)
    assert report.form_similarity_regularizer == epsilon
    assert report.similarity_regularizer_margin == epsilon - Q(9, 10**10) > 0
    assert report.proposal_regularizer_certified
    assert 6 * ETA / (2 * Q(3)) + ETA / (2 * epsilon) < 10**9 * ETA
    assert report.edge_weight_deviation_upper_bound == 10**9 * ETA
    assert report.strength_deviation_upper_bound == 7 * 10**9 * ETA
    assert 7 * 10**9 / (Q(29, 4) * Q(31, 4)) < 2 * 10**8
    assert report.mobility_deviation_upper_bound == 2 * 10**8 * ETA
    assert report.gauge_phase_deviation_upper_bound == 6 * ETA
    limit_error = 9 * 3300 * (2 * 10**8 * ETA * DELTA / 2 + 6 * ETA / Q(29, 4))
    assert (
        limit_error
        < report.common_form_limit_error_upper_bound
        == 10**12 * ETA
        < Q(1, 256)
    )
    magnitude = (
        DELTA / (2 * Q(1, 3000) * Q(33, 4) ** 2),
        DELTA / (2 * Q(1, 3300) * Q(31, 4) ** 2),
    )
    assert report.nominal_form_magnitude_bounds == magnitude
    assert Q(5, 256) < magnitude[0] <= magnitude[1] < Q(7, 256)
    assert min(report.common_form_target_margins) > 0


def test_source_association_is_fresh_and_old_joined_verdicts_do_not_transfer(
    report, monkeypatch
):
    reference = report.precontact_reference
    ignored = replace(
        reference,
        constituent_identity_certified=False,
        joint_identity_certified=False,
        response_certified=False,
        certified=False,
        status="unavailable",
    )
    monkeypatch.setattr(owner, "_assess_sine_mixed_capture", lambda **kwargs: ignored)
    assert _assess().certified
    corrupted = (
        replace(reference, source_certified=False),
        replace(reference, isolated_waiting_certified=False),
        replace(reference, form_error_bound=2 * ERROR),
        replace(reference, recoveries=reference.recoveries[::-1]),
        replace(
            reference,
            geometry=replace(reference.geometry, target_phase_turns=(Q(0),) * 18),
        ),
    )
    for value in corrupted:
        monkeypatch.setattr(
            owner, "_assess_sine_mixed_capture", lambda _value=value, **kwargs: _value
        )
        result = _assess()
        assert not result.certified and result.status == "unavailable"
        assert result.unresolved_conditions


def test_nonnegative_outside_source_domain_keeps_unavailable_candidate_bounds():
    for key in ("form_error_bound", "phase_error_bound"):
        report = _assess(**{key: ERROR + Q(1, 10**100)})
        assert not report.source_certified and not report.conditional_image_certified
        assert not report.constituent_identity_certified
        assert not report.connected_joint_registration_certified
        assert not report.normalization_common_form_separation_certified
        assert not report.certified and report.status == "unavailable"
        assert all(
            not row.asymptotic_convergence_certified
            for row in report.completion_reports
        )


def test_malformed_source_and_field_primitives_reject_before_numerical_work(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitives reached geometry, source or numerical work")

    for name in (
        "_derive_mixed_constitutive_pair",
        "_assess_sine_mixed_capture",
        "_held_sine_neighbor_field",
        "I",
        "cos",
        "pi_interval",
        "_laplacian",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for value in (
        True,
        False,
        float("nan"),
        float("inf"),
        "0",
        complex(0),
        -Q(1, 10**100),
    ):
        for key in ("form_error_bound", "phase_error_bound"):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{key: value})
    for completion in COMPLETIONS:
        count = len(_support(completion)[1])
        invalid_weights = (
            (),
            (1,) * (count - 1),
            (1,) * (count + 1),
            repeat(1),
            {0: 1},
            "weights",
            (True,) + (1,) * (count - 1),
            (float("nan"),) + (1,) * (count - 1),
            (Q(3, 4) - Q(1, 10**100),) + (1,) * (count - 1),
            (Q(1) + Q(1, 10**100),) + (1,) * (count - 1),
        )
        for weights in invalid_weights:
            with pytest.raises((TypeError, ValueError)):
                owner._coupling_successor_field(
                    completion=completion, new_edge_conductances=weights
                )
    for invalid in (None, True, 0, b"common_sense", "COMMON_SENSE", "common_sense "):
        with pytest.raises((TypeError, ValueError)):
            owner._derive_coupling_successor_geometry(completion=invalid)
    for invalid in (None, False, b"bare_degree", "native", "transport_strength "):
        with pytest.raises((TypeError, ValueError)):
            owner._coupling_successor_field(
                completion=COMPLETIONS[0],
                new_edge_conductances=(1,) * 63,
                loss_normalization=invalid,
            )


def test_state_shape_and_jet_order_reject_before_added_gradient(fields, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid state reached the added transport gradient")

    monkeypatch.setattr(owner, "_laplacian", forbidden)
    field = fields[COMPLETIONS[0], LAWS[1]]
    bad = (
        (Q(0),) * 35,
        (Q(0),) * 37,
        (True,) + (Q(0),) * 35,
        (float("nan"),) + (Q(0),) * 35,
        (Jet.constant(0, 1),) + (Q(0),) * 35,
        (Jet.constant(0, 1),) + (Jet.constant(0, 2),) * 35,
    )
    for values in bad:
        with pytest.raises((TypeError, ValueError)):
            field(values)


def test_fixed_width_fraction_inputs_iterators_and_no_report_or_deadline_argument(
    report,
):
    fixed = Q(np.int64(1), np.int64(10**10))
    assert _assess(form_error_bound=fixed, phase_error_bound=fixed) == report
    state = _interval_state(tuple(Q(i - 8, 2048) for i in range(18)))
    for completion in COMPLETIONS:
        weights = _weights(completion)
        fixed_weights = (
            Q(np.int64(w.numerator), np.int64(w.denominator)) for w in weights
        )
        first = owner._coupling_successor_field(
            completion=completion,
            new_edge_conductances=fixed_weights,
            loss_normalization=LAWS[1],
        )
        second = owner._coupling_successor_field(
            completion=completion,
            new_edge_conductances=weights,
            loss_normalization=LAWS[1],
        )
        assert first(iter(state)) == second(state)
    with pytest.raises(FrozenInstanceError):
        report.certified = False
    for key in (
        "source",
        "report",
        "elapsed_time",
        "recovery_duration",
        "observation_error_bound",
    ):
        with pytest.raises(TypeError):
            _assess(**{key: report if key in {"source", "report"} else 1})
