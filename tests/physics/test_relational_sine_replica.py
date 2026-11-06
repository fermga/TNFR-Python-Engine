"""Full-node static controls for exact replica inheritance and its limits."""

import json
import pickle
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_scale import (
    assess_sine_mixed_pair_state,
    assess_sine_mobility_geometry,
    assess_sine_pair_support_symmetry,
    assess_sine_pairing_mobility,
    assess_sine_pairing_transition,
    assess_sine_pairing_window,
    assess_sine_replica_capacity,
    assess_sine_replica_equilibria,
    assess_sine_replica_persistence,
    assess_sine_replica_pulse,
    assess_sine_replica_pulse_splitting,
    assess_sine_replica_pulse_variation,
    assess_sine_replica_scale,
    assess_sine_state_pairing,
    observe_phase_pairs,
)
from tnfr.sdk import export_to_json, relational_report_to_dict

PAIRS = tuple((2 * i, 2 * i + 1) for i in range(5))
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    if bound.lo == bound.hi:
        assert abs(value - _mp(bound.lo)) < mp.mpf("1e-80")
    else:
        assert _mp(bound.lo) <= value <= _mp(bound.hi)


def test_prepared_pulse_retains_exact_stored_exchange_and_storage_coefficients():
    model = replace(MODEL)
    object.__setattr__(model, "phase_weight", Q(5, 17))
    object.__setattr__(model, "storage_scale", Q(11, 13))
    report = assess_sine_replica_pulse(
        reference_model=model,
        form_half_difference=Q(1, 8),
        phase_half_difference=0,
        capacity=3,
    )
    assert report.model is model
    assert report.nonlinear_periodic_exchange_certified
    with mp.workdps(90):
        _contains(report.internal_form_rate_bounds, mp.mpf(0))
        _contains(report.internal_phase_rate_bounds, mp.mpf(195) / (1496 * mp.pi))
        _contains(report.internal_energy_bounds, mp.mpf(1) / 64)


def _graph(
    *,
    form=(0,) * 5,
    phase=tuple(Q(5 * i, 4) for i in range(5)),
    internal_form=(0,) * 5,
    internal_phase=(0,) * 5,
    capacity=(1,) * 5,
):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for i, pair in enumerate(PAIRS):
        for sign, node in zip((1, -1), pair):
            graph.nodes[node].update(
                EPI=Q(form[i]) + sign * Q(internal_form[i]),
                theta=Q(phase[i]) + sign * Q(internal_phase[i]),
                nu_f=capacity[i],
            )
        graph.add_edges_from(
            (left, right) for left in pair for right in PAIRS[(i + 1) % 5]
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _assessment(graph=None, **changes):
    arguments = dict(reference_model=MODEL, pairs=PAIRS)
    arguments.update(changes)
    return assess_sine_replica_scale(_graph() if graph is None else graph, **arguments)


def _fine_field(graph, state, *, beta=1, capacities=None):
    """Independent complete sine rows, in graph node order and structural t."""
    nodes = tuple(graph)
    indices = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    capacity = (
        [_mp(graph.nodes[node]["nu_f"]) for node in nodes]
        if capacities is None
        else capacities
    )
    form, phase = [], []
    for i, node in enumerate(nodes):
        neighbors = tuple(indices[neighbor] for neighbor in graph[node])
        q = sum(state[i] - state[j] for j in neighbors)
        current = sum(mp.sin(state[n + j] - state[n + i]) for j in neighbors)
        form.append(capacity[i] * current / (len(neighbors) * mp.pi))
        phase.append(capacity[i] * q / (_mp(beta) * len(neighbors) * mp.pi))
    return mp.matrix(form + phase)


def _state(graph, phase_turns=None):
    turns = (0,) * len(graph) if phase_turns is None else phase_turns
    return mp.matrix(
        [_mp(graph.nodes[node]["EPI"]) for node in graph]
        + [
            _mp(graph.nodes[node]["theta"]) + 2 * mp.pi * turn
            for node, turn in zip(graph, turns)
        ]
    )


def _means_and_internal(values):
    return (
        tuple((values[left] + values[right]) / 2 for left, right in PAIRS),
        tuple((values[left] - values[right]) / 2 for left, right in PAIRS),
    )


def _naive_form_rates(theta, capacity):
    return tuple(
        _mp(capacity[i])
        / (2 * mp.pi)
        * sum(mp.sin(theta[j] - theta[i]) for j in ((i - 1) % 5, (i + 1) % 5))
        for i in range(5)
    )


def test_full_node_rows_factor_without_discarding_internal_coordinates():
    centers = (Q(1, 4), Q(-1, 2), Q(3, 4), Q(0), Q(1, 8))
    angles = tuple(Q(i, 4) for i in range(5))
    internal_form = (Q(1, 8), Q(0), Q(-1, 16), Q(1, 32), Q(-1, 8))
    internal_phase = (Q(1, 8), Q(-1, 16), Q(3, 32), Q(0), Q(1, 16))
    capacities = (Q(1), Q(3, 2), Q(1, 2), Q(2), Q(3, 4))
    beta = Q(3, 2)
    graph = _graph(
        form=centers,
        phase=angles,
        internal_form=internal_form,
        internal_phase=internal_phase,
        capacity=capacities,
    )
    report = _assessment(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    assert len(report.comparison.nodes) == 10
    assert len(report.comparison.edges) == 20
    assert report.form_means == centers
    assert report.form_half_differences == internal_form
    assert report.phase_mean_radian_parts == angles
    assert report.phase_half_difference_radian_parts == internal_phase
    assert report.pair_capacity == capacities
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state, beta=beta)
        mean_form, internal_form_rate = _means_and_internal(field[:10, 0])
        mean_phase, internal_phase_rate = _means_and_internal(field[10:, 0])
        for direct, factored, residuals, expected in (
            (
                report.mean_form_rates,
                report.factored_mean_form_rates,
                report.mean_form_factorization_residual,
                mean_form,
            ),
            (
                report.mean_phase_rates,
                report.factored_mean_phase_rates,
                report.mean_phase_factorization_residual,
                mean_phase,
            ),
            (
                report.internal_form_rates,
                report.factored_internal_form_rates,
                report.internal_form_factorization_residual,
                internal_form_rate,
            ),
            (
                report.internal_phase_rates,
                report.factored_internal_phase_rates,
                report.internal_phase_factorization_residual,
                internal_phase_rate,
            ),
        ):
            for actual, identity, residual, value in zip(
                direct, factored, residuals, expected
            ):
                _contains(actual, value)
                _contains(identity, value)
                assert residual.contains(0)
        for i, (left, right) in enumerate(PAIRS):
            for sign, node in ((1, left), (-1, right)):
                reconstructed_form = (
                    report.form_means[i] + sign * report.form_half_differences[i]
                )
                reconstructed_phase = (
                    _mp(report.phase_mean_radian_parts[i])
                    + 2 * mp.pi * _mp(report.phase_mean_turn_parts[i])
                    + sign
                    * (
                        _mp(report.phase_half_difference_radian_parts[i])
                        + 2 * mp.pi * _mp(report.phase_half_difference_turn_parts[i])
                    )
                )
                assert reconstructed_form == graph.nodes[node]["EPI"]
                assert abs(reconstructed_phase - state[10 + node]) < mp.mpf("1e-90")
            delta, u = _mp(internal_phase[i]), _mp(internal_form[i])
            _contains(report.resultant_magnitude_bounds[i], mp.cos(delta))
            _contains(
                report.resultant_magnitude_rate_bounds[i],
                -_mp(capacities[i] / beta) * u * mp.sin(delta) / mp.pi,
            )
        fine_energy = sum(
            (state[left] - state[right]) ** 2 / 2
            + _mp(beta) * (1 - mp.cos(state[10 + right] - state[10 + left]))
            for left, right in graph.edges
        )
        coarse_energy = sum(
            _mp(centers[i] - centers[(i + 1) % 5]) ** 2 / 2
            + _mp(beta) * (1 - mp.cos(_mp(angles[(i + 1) % 5] - angles[i])))
            for i in range(5)
        )
        internal_form_energy = 4 * sum(_mp(u) ** 2 for u in internal_form)
        internal_phase_energy = (
            4
            * _mp(beta)
            * sum(
                mp.cos(_mp(angles[(i + 1) % 5] - angles[i]))
                * (
                    1
                    - mp.cos(_mp(internal_phase[i]))
                    * mp.cos(_mp(internal_phase[(i + 1) % 5]))
                )
                for i in range(5)
            )
        )
        _contains(report.comparison.storage, fine_energy)
        _contains(report.coarse_storage, coarse_energy)
        assert report.internal_form_storage == 4 * sum(u**2 for u in internal_form)
        _contains(report.internal_phase_storage_correction, internal_phase_energy)
        assert abs(
            fine_energy
            - 4 * coarse_energy
            - internal_form_energy
            - internal_phase_energy
        ) < mp.mpf("1e-90")
        assert report.storage_decomposition_residual.contains(0)


def test_exact_synchrony_inherits_the_same_rows_and_fourfold_storage():
    graph = _graph(form=(Q(1, 4), Q(-1, 2), 0, Q(1, 8), 1), capacity=(1, 2, 3, 1, 2))
    report = _assessment(graph)
    assert report.synchronized_submanifold_invariant
    assert report.source_in_synchronized_submanifold
    assert report.same_law_reduced_flow_certified_for_source
    assert report.all_state_coarse_closure_obstructed
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state)
        theta, _ = _means_and_internal(state[10:, 0])
        coarse_form = _naive_form_rates(theta, report.pair_capacity)
        for i, (left, right) in enumerate(PAIRS):
            _contains(report.naive_form_rates[i], coarse_form[i])
            _contains(report.mean_form_rates[i], coarse_form[i])
            _contains(report.naive_phase_rates[i], field[10 + left])
            assert abs(field[left] - field[right]) < mp.mpf("1e-90")
            assert abs(field[10 + left] - field[10 + right]) < mp.mpf("1e-90")
            _contains(report.internal_form_rates[i], 0)
            _contains(report.internal_phase_rates[i], 0)
            _contains(report.coarse_form_defect[i], 0)
            _contains(report.coarse_phase_defect[i], 0)
        energy = sum(
            (state[i] - state[j]) ** 2 / 2 + 1 - mp.cos(state[10 + j] - state[10 + i])
            for i, j in graph.edges
        )
        _contains(report.coarse_storage, energy / 4)
        assert report.internal_form_storage == 0
        _contains(report.internal_phase_storage_correction, 0)


def test_equal_observed_means_have_different_response_with_hidden_phase_geometry():
    base = _assessment()
    epsilon = Q(1, 8)
    hidden = _assessment(_graph(internal_phase=(epsilon, 0, 0, 0, 0)))
    assert base.form_means == hidden.form_means
    assert base.phase_mean_radian_parts == hidden.phase_mean_radian_parts
    assert base.phase_mean_turn_parts == hidden.phase_mean_turn_parts
    assert base.source_in_synchronized_submanifold
    assert not hidden.source_in_synchronized_submanifold
    assert not hidden.same_law_reduced_flow_certified_for_source
    assert hidden.coarse_form_defect[1].lo > 0
    assert hidden.coarse_form_defect[4].hi < 0
    with mp.workdps(100):
        for i in (1, 4):
            delta = -_mp(base.phase_mean_radian_parts[i])
            expected = mp.sin(delta) * (mp.cos(_mp(epsilon)) - 1) / (2 * mp.pi)
            _contains(hidden.coarse_form_defect[i], expected)
        # The two captures share represented phases5j/4. They are not labeled
        # as the exact 2pi*j/5 equilibrium, so compare the response defect.
        for i in range(5):
            _contains(hidden.coarse_phase_defect[i], 0)


def test_zero_snapshot_defect_with_internal_form_does_not_certify_future_closure():
    amplitude = Q(1, 8)
    graph = _graph(internal_form=(amplitude, 0, 0, 0, 0))
    report = _assessment(graph)
    assert report.synchronized_submanifold_invariant
    assert not report.source_in_synchronized_submanifold
    assert not report.same_law_reduced_flow_certified_for_source
    with mp.workdps(100):
        initial = _state(graph)
        initial_rate = _fine_field(graph, initial)

        def defect(value):
            form_mean_rate, _ = _means_and_internal(_fine_field(graph, value)[:10, 0])
            theta_mean, _ = _means_and_internal(value[10:, 0])
            bare_rate = _naive_form_rates(theta_mean, (1,) * 5)
            return tuple(
                actual - bare for actual, bare in zip(form_mean_rate, bare_rate)
            )

        for i in range(5):
            _contains(report.coarse_form_defect[i], 0)
            assert abs(defect(initial)[i]) < mp.mpf("1e-90")
            assert abs(
                mp.diff(lambda t: defect(initial + t * initial_rate)[i], 0)
            ) < mp.mpf("1e-90")
        for i in (1, 4):
            second_derivative = mp.diff(
                lambda t: defect(initial + t * initial_rate)[i], 0, 2
            )
            expected = -mp.sin(-_mp(Q(5 * i, 4))) * _mp(amplitude) ** 2 / (2 * mp.pi**3)
            assert abs(second_derivative - expected) < mp.mpf("1e-90")
            assert abs(second_derivative) > 0
        # At delta=0 the full defect gradient vanishes. Its second derivative
        # along F(z0) is therefore also the exact temporal Lie derivative;
        # no trajectory or constant-velocity future approximation is used.


def test_replica_midpoint_chart_requires_explicit_lifts_and_keeps_pair_order():
    graph = _graph(phase=(0,) * 5, internal_phase=(2, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        _assessment(graph)
    turns = (-1,) + (0,) * 9
    report = _assessment(graph, phase_turns=turns)
    assert report.phase_mean_radian_parts[0] == 0
    assert report.phase_mean_turn_parts[0] == Q(-1, 2)
    with mp.workdps(100):
        _contains(report.phase_half_difference_bounds[0], 2 - mp.pi)
        _contains(report.resultant_magnitude_bounds[0], mp.cos(2 - mp.pi))
        assert report.phase_chart_margins[0].lo > 0
        field = _fine_field(graph, _state(graph, turns))
        mean_form, internal_form = _means_and_internal(field[:10, 0])
        for i in range(5):
            _contains(report.factored_mean_form_rates[i], mean_form[i])
            _contains(report.factored_internal_form_rates[i], internal_form[i])
    reversed_pairs = tuple((right, left) for left, right in PAIRS)
    reversed_report = _assessment(graph, pairs=reversed_pairs, phase_turns=turns)
    assert reversed_report.form_means == report.form_means
    assert reversed_report.phase_mean_radian_parts == report.phase_mean_radian_parts
    assert reversed_report.phase_mean_turn_parts == report.phase_mean_turn_parts
    assert reversed_report.phase_half_difference_radian_parts == tuple(
        -value for value in report.phase_half_difference_radian_parts
    )
    assert reversed_report.phase_half_difference_turn_parts == tuple(
        -value for value in report.phase_half_difference_turn_parts
    )
    for invalid in ((0,) * 9, (False,) + (0,) * 9, (Q(1, 2),) + (0,) * 9):
        with pytest.raises((ValueError, TypeError)):
            _assessment(_graph(), phase_turns=invalid)
    tiny_phase = _assessment(_graph(internal_phase=(Q(1, 2**100), 0, 0, 0, 0)))
    assert not tiny_phase.source_in_synchronized_submanifold


@pytest.mark.parametrize(
    "change",
    (
        "missing_edge",
        "within_pair",
        "extra_edge",
        "capacity",
        "zero_capacity",
        "pair_partition",
    ),
)
def test_replica_inheritance_requires_its_complete_support_and_capacity_premises(
    change,
):
    graph = _graph()
    arguments = {}
    if change == "missing_edge":
        graph.remove_edge(0, 2)
    elif change == "within_pair":
        graph.add_edge(0, 1)
    elif change == "extra_edge":
        graph.add_edge(0, 4)
    elif change == "capacity":
        graph.nodes[0]["nu_f"] = 2
    elif change == "zero_capacity":
        graph.nodes[0]["nu_f"] = graph.nodes[1]["nu_f"] = 0
    else:
        arguments["pairs"] = PAIRS[:-1] + ((8, 0),)
    with pytest.raises((ValueError, TypeError)):
        _assessment(graph, **arguments)


def test_replica_report_is_detached_and_tiny_capacity_is_not_an_automatic_freeze(
    tmp_path,
):
    tiny = Q(1, 2**200)
    graph = _graph(capacity=(tiny,) * 5, internal_form=(Q(1, 8), 0, 0, 0, 0))
    before = pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )
    report = _assessment(graph)
    assert report.pair_capacity == (tiny,) * 5
    assert report.synchronized_submanifold_invariant
    assert not report.source_in_synchronized_submanifold
    assert len(report.comparison.epi) == len(report.comparison.phase) == 10
    assert (
        pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
        )
        == before
    )
    with mp.workdps(110):
        _contains(report.internal_phase_rates[0], _mp(tiny) / (8 * mp.pi))
    with pytest.raises(ValueError):
        _assessment(
            graph, reference_model=RelationalExchangeModel(1, phase_domain="regular")
        )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-replica-scale.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "retained-replica-scale.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload

    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    forged = replace(report, pairs=((OpaqueLabel(1), 1),) + report.pairs[1:])
    with pytest.raises((ValueError, TypeError)):
        forged.to_dict()


@pytest.mark.parametrize("angle", (Q(1, 2), Q(2)))
def test_single_base_edge_nonclosure_and_signed_storage_keep_generic_scope(angle):
    pairs = (("a+", "a-"), ("b+", "b-"))
    graph = nx.Graph()
    graph.add_nodes_from(("b-", "a+", "b+", "a-"))
    graph.add_edges_from((left, right) for left in pairs[0] for right in pairs[1])
    epsilon = Q(1, 8)
    for i, pair in enumerate(pairs):
        for sign, node in zip((1, -1), pair):
            graph.nodes[node].update(
                EPI=0,
                theta=angle + sign * epsilon if i else 0,
                nu_f=i + 1,
            )
    report = _assessment(graph, pairs=pairs)
    assert report.base_edges == ((0, 1),)
    assert report.base_degrees == (1, 1)
    assert report.comparison.nodes == tuple(graph)
    assert report.all_state_coarse_closure_obstructed
    assert not report.same_law_reduced_flow_certified_for_source
    assert report.coarse_form_defect[0].hi < 0
    if angle == 2:
        assert report.internal_phase_storage_correction.hi < 0
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state)
        index = {node: i for i, node in enumerate(graph)}
        for i, (left, right) in enumerate(pairs):
            mean_rate = (field[index[left]] + field[index[right]]) / 2
            _contains(report.mean_form_rates[i], mean_rate)
        expected = (mp.cos(_mp(epsilon)) - 1) * mp.sin(_mp(angle)) / mp.pi
        _contains(report.coarse_form_defect[0], expected)
        _contains(
            report.internal_phase_storage_correction,
            4 * mp.cos(_mp(angle)) * (1 - mp.cos(_mp(epsilon))),
        )


@pytest.mark.parametrize("winding", (1, 2))
def test_same_full_law_has_oscillatory_or_unstable_transverse_geometry(winding):
    graph = _graph(capacity=(Q(3, 2),) * 5)
    with mp.workdps(100):
        alpha, nu = 2 * mp.pi * winding / 5, mp.mpf("1.5")
        reference = mp.matrix([0] * 10 + [alpha * (i // 2) for i in graph])
        # This is the exact mathematical critical target used in the
        # derivative oracle, not rounded phases represented as exact data.
        for column in (0, 1):
            direction = mp.matrix([0] * 20)
            direction[10 * column] = 1
            direction[10 * column + 1] = -1
            image = mp.matrix(
                [
                    mp.diff(
                        lambda t: _fine_field(graph, reference + t * direction)[row],
                        0,
                    )
                    for row in range(20)
                ]
            )
            expected_form = -nu * mp.cos(alpha) / mp.pi if column else 0
            expected_phase = 0 if column else nu / mp.pi
            assert abs((image[0] - image[1]) / 2 - expected_form) < mp.mpf("1e-90")
            assert abs((image[10] - image[11]) / 2 - expected_phase) < mp.mpf("1e-90")
            assert all(abs(image[i]) < mp.mpf("1e-90") for i in range(2, 10))
            assert all(abs(image[i]) < mp.mpf("1e-90") for i in range(12, 20))
        squared_eigenvalue = -((nu / mp.pi) ** 2) * mp.cos(alpha)
        assert (squared_eigenvalue < 0) if winding == 1 else (squared_eigenvalue > 0)
    # Imaginary transverse eigenvalues do not imply attraction. A positive
    # squared eigenvalue supplies a positive real linear eigenvalue; no
    # finite-amplitude trajectory or physical identification is inferred.


def test_unordered_invariant_rates_are_the_pushforward_of_all_fine_rows():
    internal_form = (Q(1, 8), Q(-1, 16), Q(1, 32), Q(0), Q(-1, 8))
    internal_phase = (Q(1, 16), Q(1, 8), Q(-1, 16), Q(1, 32), Q(-1, 8))
    capacities = (Q(1), Q(3, 2), Q(1, 2), Q(2), Q(3, 4))
    graph = _graph(
        form=(Q(1, 4), Q(-1, 8), 0, Q(1, 2), Q(-1, 4)),
        internal_form=internal_form,
        internal_phase=internal_phase,
        capacity=capacities,
    )
    report = _assessment(graph)
    assert report.unordered_pair_state_closure_certified
    assert report.unordered_pair_state_identifies_swap_orbits
    assert report.internal_form_squared == tuple(u**2 for u in internal_form)
    assert len(report.comparison.epi) == len(report.comparison.phase) == 10
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state)
        _, form_internal_rates = _means_and_internal(field[:10, 0])
        _, phase_internal_rates = _means_and_internal(field[10:, 0])
        theta_means, _ = _means_and_internal(state[10:, 0])
        for i in range(5):
            u, delta = _mp(internal_form[i]), _mp(internal_phase[i])
            r, variance, correlation = mp.cos(delta), u**2, u * mp.sin(delta)
            r_rate = -mp.sin(delta) * phase_internal_rates[i]
            variance_rate = 2 * u * form_internal_rates[i]
            correlation_rate = (
                mp.sin(delta) * form_internal_rates[i]
                + u * mp.cos(delta) * phase_internal_rates[i]
            )
            restoring = (
                _mp(capacities[i])
                / (2 * mp.pi)
                * sum(
                    mp.cos(_mp(internal_phase[j]))
                    * mp.cos(theta_means[j] - theta_means[i])
                    for j in ((i - 1) % 5, (i + 1) % 5)
                )
            )
            _contains(report.form_phase_correlation_bounds[i], correlation)
            _contains(report.internal_restoring_coefficients[i], restoring)
            _contains(report.internal_phase_coefficients[i], _mp(capacities[i]) / mp.pi)
            for direct, closed, residual, expected in (
                (
                    report.resultant_magnitude_rate_bounds[i],
                    report.closed_resultant_magnitude_rates[i],
                    report.resultant_rate_closure_residual[i],
                    r_rate,
                ),
                (
                    report.internal_form_squared_rates[i],
                    report.closed_internal_form_squared_rates[i],
                    report.internal_form_squared_rate_closure_residual[i],
                    variance_rate,
                ),
                (
                    report.form_phase_correlation_rates[i],
                    report.closed_form_phase_correlation_rates[i],
                    report.form_phase_correlation_rate_closure_residual[i],
                    correlation_rate,
                ),
            ):
                _contains(direct, expected)
                _contains(closed, expected)
                assert residual.contains(0)
            constraint = correlation**2 - variance * (1 - r**2)
            constraint_rate = (
                2 * correlation * correlation_rate
                - variance_rate * (1 - r**2)
                + 2 * variance * r * r_rate
            )
            assert abs(constraint) < mp.mpf("1e-90")
            assert abs(constraint_rate) < mp.mpf("1e-90")
            _contains(report.internal_constraint_residual[i], constraint)
            _contains(report.internal_constraint_rate_residual[i], constraint_rate)

        def fine_energy(value):
            return sum(
                (value[i] - value[j]) ** 2 / 2
                + 1
                - mp.cos(value[10 + j] - value[10 + i])
                for i, j in graph.edges
            )

        energy, energy_rate = fine_energy(state), mp.diff(
            lambda t: fine_energy(state + t * field), 0
        )
        assert abs(energy_rate) < mp.mpf("1e-90")
        _contains(report.unordered_storage_bounds, energy)
        _contains(report.unordered_storage_rate_bounds, energy_rate)
        assert report.unordered_storage_residual.contains(0)
        assert report.unordered_storage_rate_residual.contains(0)
    # Residual intervals are executable checks, not proofs that arbitrary
    # independent interval triples satisfy the exact nonlinear constraint.


@pytest.mark.parametrize("cyclic", (False, True))
def test_inherited_poisson_bracket_pushes_forward_fine_storage_and_field(cyclic):
    beta = Q(3, 2)
    graph = _graph(
        form=(Q(1, 4), Q(-1, 8), 0, Q(1, 2), Q(-1, 4)),
        internal_form=(Q(1, 8), Q(-1, 16), Q(1, 32), Q(0), Q(-1, 8)),
        internal_phase=(Q(1, 16), Q(1, 8), Q(-1, 16), Q(1, 32), Q(-1, 8)),
        capacity=(Q(1), Q(3, 2), Q(1, 2), Q(2), Q(3, 4)),
    )
    if not cyclic:
        graph.remove_edges_from(product(PAIRS[0], PAIRS[-1]))
    report = _assessment(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    assert report.poisson_coordinate_order == ("X", "Theta", "R", "U", "Q")
    assert report.poisson_coefficient_pairs == (
        ("X", "Theta"),
        ("R", "U"),
        ("R", "Q"),
        ("U", "Q"),
    )
    with mp.workdps(100):
        state = _state(graph)

        def observe(value):
            observed = []
            for left, right in PAIRS:
                u = (value[left] - value[right]) / 2
                delta = (value[10 + left] - value[10 + right]) / 2
                observed.extend(
                    (
                        (value[left] + value[right]) / 2,
                        (value[10 + left] + value[10 + right]) / 2,
                        mp.cos(delta),
                        u**2,
                        u * mp.sin(delta),
                    )
                )
            return mp.matrix(observed)

        # Differentiate the actual observation map and push forward the fine
        # bracket. This independently tests the pair-mean and degree factors.
        jacobian = mp.matrix(25, 20)
        fine_bracket = mp.matrix(20)
        for column in range(20):
            direction = mp.matrix([int(i == column) for i in range(20)])
            image = mp.diff(lambda t: observe(state + t * direction), 0)
            for row in range(25):
                jacobian[row, column] = image[row]
        for node in graph:
            coefficient = _mp(graph.nodes[node]["nu_f"]) / (
                _mp(beta) * mp.pi * graph.degree[node]
            )
            fine_bracket[node, 10 + node] = -coefficient
            fine_bracket[10 + node, node] = coefficient
        inherited = jacobian * fine_bracket * jacobian.T
        observed = observe(state)
        fine_field = _fine_field(graph, state, beta=beta)
        direct_rates = jacobian * fine_field
        base_edges = sorted({tuple(sorted((i // 2, j // 2))) for i, j in graph.edges})

        def retained_energy(value):
            # Sum each complete block's four fine edges before differentiating
            # the invariant-coordinate extension of that same storage.
            return sum(
                2 * (value[5 * i] - value[5 * j]) ** 2
                + 2 * (value[5 * i + 3] + value[5 * j + 3])
                + 4
                * _mp(beta)
                * (
                    1
                    - value[5 * i + 2]
                    * value[5 * j + 2]
                    * mp.cos(value[5 * j + 1] - value[5 * i + 1])
                )
                for i, j in base_edges
            )

        fine_energy = sum(
            (state[i] - state[j]) ** 2 / 2
            + _mp(beta) * (1 - mp.cos(state[10 + j] - state[10 + i]))
            for i, j in graph.edges
        )
        assert abs(retained_energy(observed) - fine_energy) < mp.mpf("1e-90")

        def invariant_gradient(function):
            result = mp.matrix(25, 1)
            for column in range(25):
                direction = mp.matrix([int(i == column) for i in range(25)])
                result[column] = mp.diff(
                    lambda t: function(observed + t * direction), 0
                )
            return result

        storage_gradient = invariant_gradient(retained_energy)
        generated_rates = inherited * storage_gradient
        assert mp.norm(generated_rates - direct_rates) < mp.mpf("1e-90")
        coordinate_index = dict(zip(report.poisson_coordinate_order, range(5)))
        for pair in range(5):
            sparse = dict(
                zip(report.poisson_coefficient_pairs, report.poisson_coefficients[pair])
            )
            for (left, right), bound in sparse.items():
                i, j = coordinate_index[left], coordinate_index[right]
                _contains(bound, inherited[5 * pair + i, 5 * pair + j])
            for row in range(5):
                global_row = 5 * pair + row
                _contains(
                    report.unordered_storage_gradients[pair][row],
                    storage_gradient[global_row],
                )
                _contains(
                    report.poisson_generated_rates[pair][row], direct_rates[global_row]
                )
                assert report.poisson_rate_residuals[pair][row].contains(0)
                for other in range(25):
                    if other // 5 != pair:
                        assert inherited[global_row, other] == 0

            def casimir(value):
                r, u, q = value[5 * pair + 2 : 5 * pair + 5, 0]
                return q**2 - u * (1 - r**2)

            casimir_gradient = invariant_gradient(casimir)
            assert mp.norm(inherited * casimir_gradient) < mp.mpf("1e-90")
            assert all(
                bound.contains(0) for bound in report.poisson_casimir_residuals[pair]
            )


@pytest.mark.parametrize("angle", (Q(0), Q(1, 8)))
def test_exact_resultant_jets_recover_internal_state_from_the_full_field(angle):
    beta, capacity = Q(3, 2), Q(5, 4)
    graph = _graph(
        form=(Q(1, 4), Q(-1, 8), 0, Q(1, 2), Q(-1, 4)),
        internal_form=(Q(1, 8), Q(-1, 16), Q(0), Q(1, 32), Q(-1, 8)),
        internal_phase=(angle, Q(1, 16), Q(-1, 8), Q(1, 32), Q(-1, 16)),
        capacity=(capacity, Q(3, 2), Q(1, 2), Q(2), Q(3, 4)),
    )
    report = _assessment(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state, beta=beta)
        means, deltas = _means_and_internal(state[10:, 0])
        u = (state[0] - state[1]) / 2
        delta_rate = (field[10] - field[11]) / 2
        delta_acceleration = mp.diff(
            lambda t: (
                _fine_field(graph, state + t * field, beta=beta)[10]
                - _fine_field(graph, state + t * field, beta=beta)[11]
            )
            / 2,
            0,
        )
        r, sine = mp.cos(deltas[0]), mp.sin(deltas[0])
        r_rate = -sine * delta_rate
        r_acceleration = -r * delta_rate**2 - sine * delta_acceleration
        c = _mp(capacity) / (_mp(beta) * mp.pi)
        restoring = (
            _mp(capacity)
            / (2 * mp.pi)
            * sum(mp.cos(deltas[j]) * mp.cos(means[j] - means[0]) for j in (1, 4))
        )
        _contains(report.internal_restoring_coefficients[0], restoring)
        reconstructed_q = -r_rate / c
        reconstructed_u_squared = (c * restoring * (1 - r**2) - r_acceleration) / (
            c**2 * r
        )
        assert abs(reconstructed_q - u * sine) < mp.mpf("1e-90")
        assert abs(reconstructed_u_squared - u**2) < mp.mpf("1e-90")
        if angle == 0:
            assert r == 1 and r_rate == 0
            assert r_acceleration < 0 and reconstructed_u_squared > 0
            # The first jet is blind here; the actual second derivative of
            # the full phase row recovers the nonzero internal form variance.


def test_all_independent_within_pair_swaps_preserve_the_unordered_state_and_flow():
    graph = _graph(
        internal_form=(Q(1, 8), Q(-1, 16), Q(1, 32), Q(0), Q(-1, 8)),
        internal_phase=(Q(1, 16), Q(1, 8), Q(-1, 16), Q(1, 32), Q(-1, 8)),
    )
    original = _assessment(graph)
    invariant_fields = (
        "form_means",
        "phase_mean_radian_parts",
        "phase_mean_turn_parts",
        "resultant_magnitude_bounds",
        "internal_form_squared",
        "form_phase_correlation_bounds",
        "mean_form_rates",
        "mean_phase_rates",
        "closed_resultant_magnitude_rates",
        "closed_internal_form_squared_rates",
        "closed_form_phase_correlation_rates",
        "unordered_storage_bounds",
        "unordered_storage_rate_bounds",
        "poisson_coefficients",
        "unordered_storage_gradients",
        "poisson_generated_rates",
        "poisson_rate_residuals",
        "poisson_casimir_residuals",
    )
    for reverse in product((False, True), repeat=5):
        pairs = tuple(
            pair[::-1] if flip else pair for pair, flip in zip(PAIRS, reverse)
        )
        permuted = _assessment(graph, pairs=pairs)
        for field in invariant_fields:
            assert getattr(permuted, field) == getattr(original, field)
        assert permuted.comparison.nodes == original.comparison.nodes
        assert permuted.comparison.epi == original.comparison.epi
        assert permuted.comparison.phase == original.comparison.phase


def test_unordered_invariants_and_rates_ignore_common_full_turns_inside_each_pair():
    graph = _graph(
        internal_form=(Q(1, 8), Q(-1, 16), Q(0), Q(1, 8), Q(1, 32)),
        internal_phase=(Q(1, 8), Q(-1, 16), Q(1, 32), Q(0), Q(1, 16)),
    )
    original = _assessment(graph)
    block_turns = (2, -1, 0, 1, -2)
    shifted = _assessment(
        graph, phase_turns=tuple(turn for turn in block_turns for _ in range(2))
    )
    assert shifted.phase_mean_turn_parts == block_turns
    assert shifted.internal_form_squared == original.internal_form_squared
    assert shifted.resultant_magnitude_bounds == original.resultant_magnitude_bounds
    assert (
        shifted.form_phase_correlation_bounds == original.form_phase_correlation_bounds
    )
    assert (
        shifted.closed_resultant_magnitude_rates
        == original.closed_resultant_magnitude_rates
    )
    assert (
        shifted.closed_internal_form_squared_rates
        == original.closed_internal_form_squared_rates
    )
    assert (
        shifted.closed_form_phase_correlation_rates
        == original.closed_form_phase_correlation_rates
    )


@pytest.mark.parametrize("stratum", ("unit_resultant", "zero_variance", "tip"))
def test_unordered_boundary_rows_are_regular_and_preserve_hidden_response(stratum):
    amplitude, angle = Q(1, 8), Q(1, 8)
    internal_form = (amplitude, 0, 0, 0, 0) if stratum == "unit_resultant" else (0,) * 5
    internal_phase = (angle, 0, 0, 0, 0) if stratum == "zero_variance" else (0,) * 5
    graph = _graph(
        form=(Q(1, 4), 0, 0, 0, 0),
        phase=(0,) * 5,
        internal_form=internal_form,
        internal_phase=internal_phase,
    )
    report = _assessment(graph)
    with mp.workdps(100):
        state, field = _state(graph), _fine_field(graph, _state(graph))
        _contains(report.form_phase_correlation_bounds[0], 0)
        _contains(report.closed_resultant_magnitude_rates[0], 0)
        _contains(report.closed_internal_form_squared_rates[0], 0)
        bracket = dict(
            zip(report.poisson_coefficient_pairs, report.poisson_coefficients[0])
        )
        assert bracket[("X", "Theta")].hi < 0
        assert all(bound.contains(0) for bound in report.poisson_casimir_residuals[0])
        if stratum == "unit_resultant":
            assert bracket[("R", "U")].lo == bracket[("R", "U")].hi == 0
            assert bracket[("R", "Q")].lo == bracket[("R", "Q")].hi == 0
            assert bracket[("U", "Q")].hi < 0
            assert report.internal_form_squared[0] == amplitude**2
            _contains(report.resultant_magnitude_bounds[0], 1)
            _contains(
                report.closed_form_phase_correlation_rates[0],
                _mp(amplitude) ** 2 / mp.pi,
            )
            second = mp.diff(
                lambda t: mp.cos(
                    ((state + t * field)[10] - (state + t * field)[11]) / 2
                ),
                0,
                2,
            )
            assert abs(second + (_mp(amplitude) / mp.pi) ** 2) < mp.mpf("1e-90")
            assert second < 0
            assert not report.source_in_synchronized_submanifold
        elif stratum == "zero_variance":
            assert bracket[("R", "U")].lo == bracket[("R", "U")].hi == 0
            assert bracket[("R", "Q")].hi < 0
            assert bracket[("U", "Q")].lo == bracket[("U", "Q")].hi == 0
            assert report.internal_form_squared[0] == 0
            r = mp.cos(_mp(angle))
            _contains(
                report.closed_form_phase_correlation_rates[0], -(1 - r**2) / mp.pi
            )
            second = mp.diff(
                lambda t: (((state + t * field)[0] - (state + t * field)[1]) / 2) ** 2,
                0,
                2,
            )
            assert abs(second - 2 * (1 - r**2) / mp.pi**2) < mp.mpf("1e-90")
            assert second > 0
        else:
            assert all(
                bound.lo == bound.hi == 0
                for bound in report.poisson_coefficients[0][1:]
            )
            _contains(report.closed_form_phase_correlation_rates[0], 0)
            assert report.source_in_synchronized_submanifold
            assert report.mean_phase_rates[0].lo > 0
            # The internal tip stays fixed while collective state can move;
            # neither division by U nor division by 1-R^2 is admissible here.


def test_mixed_form_phase_correlation_is_information_the_closed_state_needs():
    positive = _assessment(
        _graph(
            phase=(0,) * 5,
            internal_form=(Q(1, 8), 0, 0, 0, 0),
            internal_phase=(Q(1, 8), 0, 0, 0, 0),
        )
    )
    negative = _assessment(
        _graph(
            phase=(0,) * 5,
            internal_form=(Q(-1, 8), 0, 0, 0, 0),
            internal_phase=(Q(1, 8), 0, 0, 0, 0),
        )
    )
    assert positive.form_means == negative.form_means
    assert positive.phase_mean_radian_parts == negative.phase_mean_radian_parts
    assert positive.resultant_magnitude_bounds == negative.resultant_magnitude_bounds
    assert positive.internal_form_squared == negative.internal_form_squared
    assert positive.form_phase_correlation_bounds[0].lo > 0
    assert negative.form_phase_correlation_bounds[0].hi < 0
    assert positive.closed_resultant_magnitude_rates[0].hi < 0
    assert negative.closed_resultant_magnitude_rates[0].lo > 0
    assert positive.closed_internal_form_squared_rates[0].hi < 0
    assert negative.closed_internal_form_squared_rates[0].lo > 0


def test_unordered_pair_is_reconstructible_on_each_admitted_stratum_and_exported(
    tmp_path,
):
    graph = _graph(
        internal_form=(Q(-1, 8), Q(0), Q(1, 8), Q(0), Q(1, 32)),
        internal_phase=(Q(1, 16), Q(-1, 8), Q(0), Q(0), Q(-1, 16)),
    )
    report = _assessment(graph)
    with mp.workdps(100):
        state = _state(graph)
        for i, (left, right) in enumerate(PAIRS):
            center = (state[left] + state[right]) / 2
            angle = (state[10 + left] + state[10 + right]) / 2
            u = (state[left] - state[right]) / 2
            delta = (state[10 + left] - state[10 + right]) / 2
            r, variance, correlation = mp.cos(delta), u**2, u * mp.sin(delta)
            _contains(report.resultant_magnitude_bounds[i], r)
            assert (
                report.internal_form_squared[i] == report.form_half_differences[i] ** 2
            )
            _contains(report.form_phase_correlation_bounds[i], correlation)
            reconstructed_u = mp.sqrt(variance)
            reconstructed_delta = (
                mp.asin(correlation / reconstructed_u) if variance > 0 else mp.acos(r)
            )
            reconstructed = sorted(
                (
                    (center + reconstructed_u, angle + reconstructed_delta),
                    (center - reconstructed_u, angle - reconstructed_delta),
                )
            )
            original = sorted(
                ((state[left], state[10 + left]), (state[right], state[10 + right]))
            )
            assert max(
                abs(a - b)
                for actual, expected in zip(reconstructed, original)
                for a, b in zip(actual, expected)
            ) < mp.mpf("1e-90")
    payload = report.to_dict()
    for name in (
        "internal_form_squared",
        "form_phase_correlation_bounds",
        "internal_constraint_residual",
        "unordered_storage_rate_bounds",
        "poisson_coordinate_order",
        "poisson_coefficient_pairs",
        "poisson_coefficients",
        "unordered_storage_gradients",
        "poisson_generated_rates",
        "poisson_rate_residuals",
        "poisson_casimir_residuals",
    ):
        assert name in payload["report"]
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "unordered-replica-state.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload


def _pulse(**changes):
    arguments = dict(
        reference_model=MODEL,
        form_half_difference=Q(1, 16),
        phase_half_difference=0,
        capacity=1,
    )
    arguments.update(changes)
    return assess_sine_replica_pulse(**arguments)


def test_prepared_pulse_reduction_matches_all_ten_nonlinear_rows_and_full_storage():
    beta, nu, u, delta = Q(3, 2), Q(3, 2), Q(1, 16), Q(1, 32)
    report = _pulse(
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
        capacity=nu,
        form_half_difference=u,
        phase_half_difference=delta,
    )
    assert report.target_phase_turns == tuple(Q(i, 5) for i in range(5))
    assert report.symbolic_family_invariant and report.collective_means_constant
    assert not report.graph_membership_certified
    assert report.nonlinear_periodic_exchange_certified
    graph = _graph(capacity=(nu,) * 5)
    with mp.workdps(100):
        alpha = 2 * mp.pi / 5
        cosine = mp.cos(alpha)
        state = mp.matrix(
            [3 + sign * _mp(u) for _ in range(5) for sign in (1, -1)]
            + [
                alpha * i - mp.mpf("0.25") + sign * _mp(delta)
                for i in range(5)
                for sign in (1, -1)
            ]
        )
        field = _fine_field(graph, state, beta=beta)
        for left, right in PAIRS:
            assert abs(field[left] + field[right]) < mp.mpf("1e-90")
            assert abs(field[10 + left] + field[10 + right]) < mp.mpf("1e-90")
            _contains(
                report.internal_form_rate_bounds, (field[left] - field[right]) / 2
            )
            _contains(
                report.internal_phase_rate_bounds,
                (field[10 + left] - field[10 + right]) / 2,
            )
        full_energy = sum(
            (state[i] - state[j]) ** 2 / 2
            + _mp(beta) * (1 - mp.cos(state[10 + j] - state[10 + i]))
            for i, j in graph.edges
        )
        internal_energy = _mp(u) ** 2 + _mp(beta) * cosine * mp.sin(_mp(delta)) ** 2
        parameter = internal_energy / (_mp(beta) * cosine)
        frequency_squared = _mp(nu) ** 2 * cosine / (_mp(beta) * mp.pi**2)
        _contains(report.internal_energy_bounds, internal_energy)
        _contains(report.normalized_energy_bounds, parameter)
        _contains(report.full_storage_bounds, full_energy)
        _contains(report.natural_angular_frequency_squared_bounds, frequency_squared)
        assert abs(
            full_energy - 20 * _mp(beta) * (1 - cosine) - 20 * internal_energy
        ) < mp.mpf("1e-90")
    # Mathematical alpha is supplied in this oracle. No rounded graph phase
    # or small residual is promoted to membership in the exact family.


def test_exact_elliptic_solution_returns_labels_and_swaps_the_unordered_pairs():
    nu, initial_u = Q(3, 2), Q(1, 16)
    report = _pulse(capacity=nu, form_half_difference=initial_u)
    graph = _graph(capacity=(nu,) * 5)
    with mp.workdps(100):
        alpha, cosine = 2 * mp.pi / 5, mp.cos(2 * mp.pi / 5)
        parameter = _mp(initial_u) ** 2 / cosine
        omega = _mp(nu) * mp.sqrt(cosine) / mp.pi
        quarter_argument = mp.ellipk(parameter)
        period = 4 * quarter_argument / omega

        def internal(time):
            argument = omega * time
            return (
                _mp(initial_u) * mp.ellipfun("cn", argument, parameter),
                mp.asin(mp.sqrt(parameter) * mp.ellipfun("sn", argument, parameter)),
            )

        def state_at(time):
            u, delta = internal(time)
            return mp.matrix(
                [sign * u for _ in range(5) for sign in (1, -1)]
                + [alpha * i + sign * delta for i in range(5) for sign in (1, -1)]
            )

        _contains(report.period_bounds, period)
        _contains(report.unordered_period_bounds, period / 2)
        _contains(report.small_amplitude_period_bounds, 2 * mp.pi / omega)
        assert (
            2 * mp.pi / omega < period <= 2 * mp.pi / (omega * mp.sqrt(1 - parameter))
        )
        for time in (mp.mpf(0), period / 7, period / 4):
            state = state_at(time)
            field = _fine_field(graph, state)
            analytic_derivative = mp.matrix(
                [mp.diff(lambda t: state_at(t)[i], time) for i in range(20)]
            )
            assert mp.norm(field - analytic_derivative) < mp.mpf("1e-90")
            energy = sum(
                (state[i] - state[j]) ** 2 / 2
                + 1
                - mp.cos(state[10 + j] - state[10 + i])
                for i, j in graph.edges
            )
            _contains(report.full_storage_bounds, energy)
        time = period / 7
        u, delta = internal(time)
        swapped_u, swapped_delta = internal(time + period / 2)
        assert abs(swapped_u + u) < mp.mpf("1e-90")
        assert abs(swapped_delta + delta) < mp.mpf("1e-90")
        before = (mp.cos(delta), u**2, u * mp.sin(delta))
        after = (mp.cos(swapped_delta), swapped_u**2, swapped_u * mp.sin(swapped_delta))
        assert max(abs(left - right) for left, right in zip(before, after)) < mp.mpf(
            "1e-90"
        )
        fine, swapped = state_at(time), state_at(time + period / 2)
        for left, right in PAIRS:
            assert abs(swapped[left] - fine[right]) < mp.mpf("1e-90")
            assert abs(swapped[10 + left] - fine[10 + right]) < mp.mpf("1e-90")
        assert mp.norm(state_at(period) - state_at(0)) < mp.mpf("1e-90")
        turn_u, turn_delta = internal(period / 4)
        assert abs(turn_u) < mp.mpf("1e-90")
        assert mp.cos(turn_delta) < 1
        assert turn_u**2 < _mp(initial_u) ** 2
    # These fixed evaluations check a closed elliptic solution against the
    # full rows; no numerical initial-value trajectory or fitted period is used.


@pytest.mark.parametrize(
    "amplitude,acute_status", ((Q(1, 16), "certified"), (Q(1, 8), "excluded"))
)
def test_internal_libration_and_all_time_fine_acuity_are_separate_conditions(
    amplitude, acute_status
):
    report = _pulse(form_half_difference=amplitude)
    assert report.status == "libration_certified"
    assert report.period_bounds is not None
    assert report.all_fine_edges_acute_status == acute_status
    with mp.workdps(100):
        alpha = 2 * mp.pi / 5
        parameter = _mp(amplitude) ** 2 / mp.cos(alpha)
        maximum_delta = mp.asin(mp.sqrt(parameter))
        threshold = mp.sin(mp.pi / 20) ** 2
        _contains(report.acute_energy_threshold_bounds, threshold)
        _contains(report.all_fine_edges_acute_margin_bounds, threshold - parameter)
        assert maximum_delta < mp.pi / 2
        widest_edge = alpha + 2 * maximum_delta
        assert (widest_edge < mp.pi / 2) == (acute_status == "certified")
    assert not report.graph_membership_certified


def test_pulse_stationary_over_barrier_and_unresolved_energy_do_not_get_a_period():
    stationary = _pulse(form_half_difference=0, phase_half_difference=0)
    high_energy = _pulse(form_half_difference=1)
    with mp.workdps(120):
        near_barrier = Q(mp.nstr(mp.sqrt(mp.cos(2 * mp.pi / 5)), 110))
    unresolved = _pulse(form_half_difference=near_barrier)
    assert stationary.status == "equilibrium"
    assert high_energy.status == "over_barrier_out_of_scope"
    assert unresolved.status == "energy_classification_unresolved"
    assert unresolved.normalized_energy_bounds.contains(1)
    for report in (stationary, high_energy, unresolved):
        assert not report.nonlinear_periodic_exchange_certified
        assert report.period_bounds is None
        assert report.unordered_period_bounds is None
        assert report.period_unavailable_reasons
    # The rounded rational near the separatrix is not an exact separatrix
    # claim. Overlap with the threshold must remain numerically unresolved.
    tiny_phase = _pulse(form_half_difference=0, phase_half_difference=Q(1, 2**200))
    assert tiny_phase.internal_energy_bounds.lo == 0
    assert tiny_phase.status == "libration_certified"
    assert tiny_phase.nonlinear_periodic_exchange_certified
    assert tiny_phase.period_bounds is not None
    assert tiny_phase.all_fine_edges_acute_status == "certified"
    # Exact nonzero delta in the strict pair chart proves positive energy
    # even when outward arithmetic cannot resolve its positive lower bound.


def test_replica_pulse_clock_and_form_unit_scalings_preserve_the_dimensionless_orbit():
    tiny = Q(1, 2**200)
    ordinary = _pulse()
    slow = _pulse(capacity=tiny)
    rescaled = _pulse(
        reference_model=RelationalExchangeModel(
            4, epi_weight=0, phase_domain="regular"
        ),
        form_half_difference=Q(1, 8),
    )
    assert slow.capacity == tiny
    assert slow.nonlinear_periodic_exchange_certified
    assert slow.small_amplitude_period_bounds.lo > 0
    assert ordinary.normalized_energy_bounds == slow.normalized_energy_bounds
    assert ordinary.normalized_energy_bounds == rescaled.normalized_energy_bounds
    with mp.workdps(110):
        cosine = mp.cos(2 * mp.pi / 5)
        parameter = mp.mpf(1) / (256 * cosine)
        period = 4 * mp.pi * mp.ellipk(parameter) / mp.sqrt(cosine)
        _contains(ordinary.period_bounds, period)
        _contains(slow.period_bounds, period / _mp(tiny))
        _contains(rescaled.period_bounds, 2 * period)
        _contains(slow.unordered_period_bounds, period / (2 * _mp(tiny)))


def test_pure_phase_libration_near_chart_boundary_does_not_fabricate_a_period_bound():
    with mp.workdps(120):
        delta = Q(mp.nstr(mp.pi / 2 - mp.mpf("1e-25"), 110))
    report = _pulse(form_half_difference=0, phase_half_difference=delta)
    assert report.phase_chart_margin_bounds.lo > 0
    assert report.normalized_energy_bounds.contains(1)
    assert report.status == "libration_certified"
    assert report.nonlinear_periodic_exchange_certified
    assert report.period_bounds is None
    assert report.unordered_period_bounds is None
    assert report.period_unavailable_reasons == (
        "positive_period_bound_denominator_not_resolved",
    )
    with mp.workdps(120):
        parameter = mp.sin(_mp(delta)) ** 2
        assert 0 < parameter < 1
        _contains(report.normalized_energy_bounds, parameter)
        assert 0 < mp.cos(_mp(delta)) ** 2 < mp.mpf("1e-49")
    # Exact chart membership and u=0 prove a libration, while the tiny
    # numerical margin remains unresolved. These are separate certificates.


@pytest.mark.parametrize(
    "change",
    (
        {"capacity": 0},
        {"capacity": -1},
        {"capacity": True},
        {"form_half_difference": True},
        {"phase_half_difference": float("nan")},
        {"phase_half_difference": 2},
        {"reference_model": RelationalExchangeModel(1, phase_domain="regular")},
    ),
)
def test_prepared_pulse_rejects_invalid_state_clock_and_law_declarations(change):
    with pytest.raises((ValueError, TypeError)):
        _pulse(**change)


def test_prepared_pulse_export_preserves_symbolic_family_and_both_period_meanings(
    tmp_path,
):
    report = _pulse()
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-replica-pulse.v1"
    assert payload["report"]["graph_membership_certified"] is False
    assert payload["report"]["nonlinear_periodic_exchange_certified"] is True
    assert "period_bounds" in payload["report"]
    assert "unordered_period_bounds" in payload["report"]
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "prepared-internal-pulse.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload


def _variation(**changes):
    arguments = dict(
        reference_model=MODEL,
        form_half_difference=Q(1, 16),
        phase_half_difference=Q(1, 20),
        capacity=1,
    )
    arguments.update(changes)
    return assess_sine_replica_pulse_variation(**arguments)


def _pulse_state(u, delta):
    return mp.matrix(
        [sign * u for _ in range(5) for sign in (1, -1)]
        + [2 * mp.pi * i / 5 + sign * delta for i in range(5) for sign in (1, -1)]
    )


def _fine_jacobian(graph, state, *, beta):
    """Differentiate the twenty independent fine rows, before any reduction."""
    columns = [
        mp.diff(
            lambda shift: _fine_field(
                graph, state + shift * mp.eye(20)[:, j], beta=beta
            ),
            0,
        )
        for j in range(20)
    ]
    return mp.matrix([[columns[j][i] for j in range(20)] for i in range(20)])


def _fourier_embedding(mode):
    # Complex Fourier amplitudes use exp(+i*q*j); member signs encode the
    # actual means and half-differences, not an assumed reduced evolution.
    embedding = mp.matrix(20, 4)
    for i, (left, right) in enumerate(PAIRS):
        wave = mp.exp(2j * mp.pi * mode * i / 5) / mp.sqrt(5)
        for sign, node in ((1, left), (-1, right)):
            embedding[node, 0] = wave
            embedding[10 + node, 1] = wave
            embedding[node, 2] = sign * wave
            embedding[10 + node, 3] = sign * wave
    return embedding


def _real_mode_block(jacobian, mode):
    embedding = _fourier_embedding(mode)
    raw = embedding.H * jacobian * embedding / 2
    internal_factor = 1 if mode == 0 else (1j if mode in (1, 2) else -1j)
    change = mp.diag((1, 1, internal_factor, internal_factor))
    block = change * raw * change**-1
    assert max(abs(mp.im(value)) for value in block) < mp.mpf("1e-90")
    return mp.matrix([[mp.re(block[i, j]) for j in range(4)] for i in range(4)])


@pytest.fixture(scope="module")
def fine_variation_oracle():
    beta, nu, u, delta = Q(3, 2), Q(5, 4), Q(1, 16), Q(1, 20)
    report = _variation(
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
        capacity=nu,
        form_half_difference=u,
        phase_half_difference=delta,
    )
    with mp.workdps(100):
        graph = _graph(capacity=(nu,) * 5)
        state = _pulse_state(_mp(u), _mp(delta))
        jacobian = _fine_jacobian(graph, state, beta=beta)
        blocks = tuple(_real_mode_block(jacobian, mode) for mode in range(5))
    return report, graph, state, jacobian, blocks


def test_variational_fourier_blocks_retain_the_full_twenty_node_coordinates(
    fine_variation_oracle,
):
    report, _, _, jacobian, blocks = fine_variation_oracle
    assert report.mode_indices == (0, 1, 2)
    assert report.mode_multiplicities == (1, 2, 2)
    assert 4 * sum(report.mode_multiplicities) == report.full_real_dimension == 20
    with mp.workdps(100):
        # The five complex embeddings form a complete basis; conjugate modes
        # contribute separate real directions, not discarded fine states.
        embeddings = [_fourier_embedding(mode) for mode in range(5)]
        basis = mp.matrix(
            [
                [embeddings[k][i, j] for k in range(5) for j in range(4)]
                for i in range(20)
            ]
        )
        assert mp.norm(basis.H * basis / 2 - mp.eye(20)) < mp.mpf("1e-90")
        for mode in range(5):
            representative = min(mode, 5 - mode)
            for i in range(4):
                for j in range(4):
                    _contains(
                        report.mode_blocks[representative][i][j], blocks[mode][i, j]
                    )
            for other in range(5):
                if other != mode:
                    coupling = embeddings[other].H * jacobian * embeddings[mode] / 2
                    assert mp.norm(coupling) < mp.mpf("1e-90")
        assert blocks[1][0, 3] < 0
        assert abs(blocks[1][0, 3] - blocks[1][2, 1]) < mp.mpf("1e-90")


def test_dimensionless_variation_is_the_full_jacobian_similarity_not_a_new_law(
    fine_variation_oracle,
):
    report, _, _, _, blocks = fine_variation_oracle
    rescaled = _variation(
        reference_model=RelationalExchangeModel(
            6, epi_weight=0, phase_domain="regular"
        ),
        capacity=Q(7, 3),
        form_half_difference=Q(1, 8),
    )
    assert (
        report.pulse.normalized_energy_bounds == rescaled.pulse.normalized_energy_bounds
    )
    assert report.dimensionless_mode_blocks == rescaled.dimensionless_mode_blocks
    with mp.workdps(100):
        beta, nu = mp.mpf(3) / 2, mp.mpf(5) / 4
        cosine = mp.cos(2 * mp.pi / 5)
        form_scale = mp.sqrt(beta * cosine)
        omega = nu * mp.sqrt(cosine / beta) / mp.pi
        _contains(report.form_normalization_bounds, form_scale)
        _contains(report.dimensionless_clock_rate_bounds, omega)
        scale = mp.diag((1 / form_scale, 1, 1 / form_scale, 1))
        for mode in range(3):
            normalized = scale * blocks[mode] * scale**-1 / omega
            for i in range(4):
                for j in range(4):
                    _contains(
                        report.dimensionless_mode_blocks[mode][i][j], normalized[i, j]
                    )
    # A beta change at fixed raw u changes m. Covariance above required the
    # matching form-unit change and does not assert equal physical clocks.
    fixed_u = _variation(
        reference_model=RelationalExchangeModel(6, epi_weight=0, phase_domain="regular")
    )
    assert (
        fixed_u.pulse.normalized_energy_bounds != report.pulse.normalized_energy_bounds
    )


def test_zero_amplitude_full_spectrum_has_collective_and_internal_frequencies():
    report = _variation(form_half_difference=0, phase_half_difference=0)
    assert report.equilibrium_reference
    assert not report.periodic_reference_certified
    with mp.workdps(100):
        graph = _graph()
        jacobian = _fine_jacobian(graph, _pulse_state(0, 0), beta=1)
        # Direct normalized fine Laplacian eigenvalues retain five internal
        # directions as well as the base cycle modes.
        laplacian = mp.matrix(10, 10)
        for i in graph:
            laplacian[i, i] = 1
            for j in graph[i]:
                laplacian[i, j] = -mp.mpf(1) / graph.degree[i]
        spatial_values, vectors = mp.eigsy(laplacian)
        expected = sorted(
            [0]
            + [(5 - mp.sqrt(5)) / 4] * 2
            + [mp.mpf(1)] * 5
            + [(5 + mp.sqrt(5)) / 4] * 2
        )
        omega_squared = mp.cos(2 * mp.pi / 5) / mp.pi**2
        _contains(
            report.zero_amplitude_internal_frequency_squared_bounds, omega_squared
        )
        for k, expected_value in enumerate(expected):
            assert abs(spatial_values[k] - expected_value) < mp.mpf("1e-90")
            vector = mp.matrix(list(vectors[:, k]) + [0] * 10)
            squared_action = jacobian * jacobian * vector
            assert mp.norm(
                squared_action + omega_squared * spatial_values[k] ** 2 * vector
            ) < mp.mpf("1e-90")
        for mode in range(3):
            eigenvalue = 1 - mp.cos(2 * mp.pi * mode / 5)
            _contains(
                report.zero_amplitude_collective_frequency_squared_bounds[mode],
                omega_squared * eigenvalue**2,
            )
    assert report.orbital_stability_status == "not_assessed"


def test_half_return_conjugacy_and_hamiltonian_structure_do_not_claim_half_periodicity(
    fine_variation_oracle,
):
    report, _, _, _, blocks = fine_variation_oracle
    reverse = _variation(
        reference_model=report.pulse.model,
        capacity=report.pulse.capacity,
        form_half_difference=-report.pulse.form_half_difference,
        phase_half_difference=-report.pulse.phase_half_difference,
    )
    assert report.periodic_reference_certified
    assert report.half_return_swap_covariance_certified
    with mp.workdps(100):
        swap = mp.diag((1, 1, -1, -1))
        symplectic = mp.matrix(
            [[int(value) for value in row] for row in report.symplectic_form]
        )
        assert mp.norm(swap.T * symplectic * swap - symplectic) == 0
        for mode in range(3):
            block = blocks[mode]
            assert mp.norm(block.T * symplectic + symplectic * block) < mp.mpf("1e-90")
            half_return = swap * block * swap
            for i in range(4):
                for j in range(4):
                    _contains(reverse.mode_blocks[mode][i][j], half_return[i, j])
                    assert report.hamiltonian_structure_residuals[mode][i][j].contains(
                        0
                    )
        assert blocks[1][0, 3] < 0
        assert (swap * blocks[1] * swap)[0, 3] > 0
    assert report.mode_blocks[1] != reverse.mode_blocks[1]
    assert (
        report.orbital_stability_status
        == reverse.orbital_stability_status
        == "not_assessed"
    )


def test_exact_pulse_time_tangent_and_amplitude_shear_are_neutral_not_attraction():
    # Fixed evaluations of the exact elliptic family provide the time and
    # energy derivative identities without computing a variational propagator.
    with mp.workdps(100):
        cosine = mp.cos(2 * mp.pi / 5)
        omega = mp.sqrt(cosine) / mp.pi
        energy = mp.mpf(1) / 256

        def state(time, h):
            parameter = h / cosine
            return mp.matrix(
                [
                    mp.sqrt(h) * mp.ellipfun("cn", omega * time, parameter),
                    mp.asin(
                        mp.sqrt(parameter) * mp.ellipfun("sn", omega * time, parameter)
                    ),
                ]
            )

        def period(h):
            return 4 * mp.ellipk(h / cosine) / omega

        duration, period_slope = period(energy), mp.diff(period, energy)
        lower_slope = mp.pi / (2 * omega * cosine)
        upper_slope = lower_slope / (1 - energy / cosine) ** mp.mpf("1.5")
        assert 0 < lower_slope < period_slope <= upper_slope
        time = duration / 7
        _, delta = state(time, energy)
        block = mp.matrix([[0, -cosine * mp.cos(2 * delta) / mp.pi], [1 / mp.pi, 0]])
        tangent = mp.diff(lambda t: state(t, energy), time)
        assert mp.norm(
            mp.diff(lambda t: state(t, energy), time, 2) - block * tangent
        ) < mp.mpf("1e-90")
        initial_energy_direction = mp.diff(lambda h: state(0, h), energy)
        initial_time_direction = mp.diff(lambda t: state(t, energy), 0)
        full_energy_direction = mp.diff(lambda h: state(duration, h), energy)
        half_energy_direction = -mp.diff(lambda h: state(duration / 2, h), energy)
        assert mp.norm(
            full_energy_direction
            - initial_energy_direction
            + period_slope * initial_time_direction
        ) < mp.mpf("1e-90")
        assert mp.norm(
            half_energy_direction
            - initial_energy_direction
            + period_slope * initial_time_direction / 2
        ) < mp.mpf("1e-90")
        # Differentiation holds the evaluation time fixed. Differentiating
        # state(T(h),h) would cancel the shear and test only periodicity.


def test_variation_admission_and_export_separate_instantaneous_identity_from_pulse(
    tmp_path,
):
    periodic = _variation()
    stationary = _variation(form_half_difference=0, phase_half_difference=0)
    over_barrier = _variation(form_half_difference=1)
    with mp.workdps(120):
        near_barrier = Q(mp.nstr(mp.sqrt(mp.cos(2 * mp.pi / 5)), 110))
    unresolved = _variation(form_half_difference=near_barrier, phase_half_difference=0)
    for report in (stationary, over_barrier, unresolved):
        assert report.variation_identity_certified
        assert len(report.mode_blocks) == 3
        assert not report.periodic_reference_certified
        assert not report.half_return_swap_covariance_certified
        assert report.orbital_stability_status == "not_assessed"
    for change in (
        {"capacity": 0},
        {"phase_half_difference": 2},
        {"form_half_difference": True},
    ):
        with pytest.raises((ValueError, TypeError)):
            _variation(**change)
    # A moving pulse crosses delta=0 and shares that snapshot Jacobian with
    # equilibrium. The snapshot alone is not a constant-coefficient proof.
    crossing = _variation(phase_half_difference=0)
    assert crossing.mode_blocks == stationary.mode_blocks
    assert crossing.periodic_reference_certified
    payload = periodic.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-replica-pulse-variation.v1"
    assert payload["report"]["full_real_dimension"] == 20
    assert payload["report"]["pulse"]["graph_membership_certified"] is False
    assert payload["report"]["orbital_stability_status"] == "not_assessed"
    assert relational_report_to_dict(periodic)["report"] == payload["report"]
    destination = tmp_path / "pulse-variation.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload


@pytest.mark.parametrize("mode", (1, 2))
def test_small_amplitude_splitting_comes_from_forced_harmonics_and_elliptic_clock(mode):
    report = assess_sine_replica_pulse_splitting(reference_model=MODEL, capacity=1)
    index = mode - 1
    with mp.workdps(100):
        alpha, wave = 2 * mp.pi / 5, 2 * mp.pi * mode / 5
        rho = 1 - mp.cos(wave)
        gamma = mp.sqrt(rho) * mp.tan(alpha) * mp.sin(wave)

        def clock_squared(parameter):
            return (mp.pi / (2 * mp.ellipk(parameter))) ** 2

        assert abs(mp.diff(clock_squared, 0) + mp.mpf(1) / 2) < mp.mpf("1e-90")

        def exact_coefficients(phase, parameter):
            clock = mp.pi / (2 * mp.ellipk(parameter))
            tau = phase / clock
            sn = mp.ellipfun("sn", tau, parameter)
            dn = mp.ellipfun("dn", tau, parameter)
            return (
                rho**2 * dn**2 / clock**2,
                (1 - (2 - rho) * parameter * sn**2) / clock**2,
                gamma * sn * dn / clock**2,
            )

        def diagonal_correction(phase):
            return (rho - 1) / 2 + (2 - rho) * mp.cos(2 * phase) / 2

        for phase in (mp.mpf("0.37"), mp.mpf("1.1"), mp.mpf("2.4")):
            collective_slope = mp.diff(lambda m: exact_coefficients(phase, m)[0], 0)
            internal_slope = mp.diff(lambda m: exact_coefficients(phase, m)[1], 0)
            assert abs(collective_slope - rho**2 * mp.cos(2 * phase) / 2) < mp.mpf(
                "1e-90"
            )
            assert abs(internal_slope - diagonal_correction(phase)) < mp.mpf("1e-90")
            assert abs(
                exact_coefficients(phase, 0)[2] - gamma * mp.sin(phase)
            ) < mp.mpf("1e-90")

        # Invert the collective oscillator separately on its zero and second
        # Fourier harmonics. rho is neither zero nor two, so both are admitted.
        denominator = rho**2 - 4

        def collective_cos(phase):
            return -gamma * mp.sin(2 * phase) / (2 * denominator)

        def collective_sin(phase):
            return -gamma / (2 * rho**2) + gamma * mp.cos(2 * phase) / (2 * denominator)

        projections = []
        for initial_mode, response in (
            (mp.cos, collective_cos),
            (mp.sin, collective_sin),
        ):
            for phase in (mp.mpf("0.37"), mp.mpf("1.1")):
                residual = mp.diff(response, phase, 2) + rho**2 * response(phase)
                residual += gamma * mp.sin(phase) * initial_mode(phase)
                assert abs(residual) < mp.mpf("1e-90")

            def forcing(phase):
                return diagonal_correction(phase) * initial_mode(
                    phase
                ) + gamma * mp.sin(phase) * response(phase)

            projections.append(
                tuple(
                    mp.quad(
                        lambda phase: test_mode(phase) * forcing(phase),
                        [0, mp.pi, 2 * mp.pi],
                    )
                    / mp.pi
                    for test_mode in (mp.cos, mp.sin)
                )
            )
        p, q = projections[0][0], projections[1][1]
        assert abs(projections[0][1]) < mp.mpf("1e-90")
        assert abs(projections[1][0]) < mp.mpf("1e-90")
        _contains(report.p_bounds[index], p)
        _contains(report.q_bounds[index], q)
        # Dropping feedback or the +1/2 clock correction changes the resonant
        # coefficients. Neither is an optional effective parameter.
        uncoupled_p, uncoupled_q = rho / 4, (3 * rho - 4) / 4
        assert abs(p - uncoupled_p) > mp.mpf("0.1")
        assert abs(q - uncoupled_q) > mp.mpf("0.1")
        for coefficients, expected in (
            (report.p_exact_coefficients[index], p),
            (report.q_exact_coefficients[index], q),
        ):
            exact_value = _mp(coefficients[0]) + _mp(coefficients[1]) * mp.sqrt(5)
            assert abs(exact_value - expected) < mp.mpf("1e-90")
        generator = mp.matrix([[0, q / 2], [-p / 2, 0]])
        for i in range(2):
            for j in range(2):
                _contains(report.slow_generators[index][i][j], generator[i, j])
        squared_exponent = -p * q / 4
        _contains(report.squared_slow_exponent_bounds[index], squared_exponent)
        assert (squared_exponent > 0) == (mode == 1)
        magnitude = mp.sqrt(abs(squared_exponent))
        _contains(report.leading_exponent_magnitude_bounds[index], magnitude)
        _contains(
            report.labeled_log_multiplier_slope_magnitude_bounds[index],
            2 * mp.pi * magnitude,
        )
        _contains(
            report.unordered_log_multiplier_slope_magnitude_bounds[index],
            mp.pi * magnitude,
        )


def test_small_amplitude_result_is_dimensionless_and_not_a_finite_preparation_verdict():
    report = assess_sine_replica_pulse_splitting(reference_model=MODEL, capacity=1)
    rescaled = assess_sine_replica_pulse_splitting(
        reference_model=RelationalExchangeModel(
            4, epi_weight=0, phase_domain="regular"
        ),
        capacity=Q(1, 2**200),
    )
    assert report.reference.equilibrium_reference
    assert not report.reference.periodic_reference_certified
    assert rescaled.reference.pulse.capacity == Q(1, 2**200)
    assert report.p_exact_coefficients == rescaled.p_exact_coefficients
    assert report.q_exact_coefficients == rescaled.q_exact_coefficients
    assert report.slow_generators == rescaled.slow_generators
    assert report.squared_slow_exponent_bounds == rescaled.squared_slow_exponent_bounds
    assert report.existential_amplitude_interval_certified
    assert report.mode_classifications == ("hyperbolic", "elliptic")
    assert report.squared_slow_exponent_bounds[0].lo > 0
    assert report.squared_slow_exponent_bounds[1].hi < 0
    assert report.sufficiently_small_nonlinear_orbital_instability_certified
    assert report.amplitude_upper_bound is None
    assert report.finite_amplitude_remainder_bound is None
    assert not report.finite_preparation_assessed
    assert not report.return_multipliers_computed
    # The exact equilibrium is not a periodic pulse; the theorem concerns
    # sufficiently small positive amplitudes and supplies no numerical radius.
    with pytest.raises(TypeError):
        assess_sine_replica_pulse_splitting(
            reference_model=MODEL, capacity=1, form_half_difference=Q(1, 16)
        )


def test_small_amplitude_admission_and_export_do_not_install_a_new_model(tmp_path):
    for capacity in (0, -1, True):
        with pytest.raises((ValueError, TypeError)):
            assess_sine_replica_pulse_splitting(
                reference_model=MODEL, capacity=capacity
            )
    with pytest.raises(ValueError):
        assess_sine_replica_pulse_splitting(
            reference_model=RelationalExchangeModel(1, phase_domain="regular"),
            capacity=1,
        )
    report = assess_sine_replica_pulse_splitting(reference_model=MODEL, capacity=1)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-replica-pulse-splitting.v1"
    assert payload["report"]["amplitude_upper_bound"] is None
    assert payload["report"]["finite_amplitude_remainder_bound"] is None
    assert payload["report"]["finite_preparation_assessed"] is False
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "small-amplitude-splitting.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload


def _persistent_graph(**changes):
    arguments = dict(
        phase=tuple(Q(1287 * i, 1024) for i in range(5)),
        internal_form=(Q(1, 1024),) * 5,
    )
    arguments.update(changes)
    return _graph(**arguments)


def _persistence(graph=None, **changes):
    arguments = dict(
        reference_model=MODEL,
        pairs=PAIRS,
        radius=Q(1, 16),
        excess_ceiling=Q(1, 32768),
        form_mean_bounds=(-1, 1),
    )
    arguments.update(changes)
    return assess_sine_replica_persistence(
        _persistent_graph() if graph is None else graph, **arguments
    )


@pytest.mark.parametrize("beta,capacity", ((Q(1), Q(1)), (Q(3, 2), Q(5, 4))))
def test_persistence_full_geometry_storage_and_rotation_match_fine_field(
    monkeypatch, beta, capacity
):
    import tnfr.physics.relational_sine_scale as owner

    calls = []
    original = owner.bound_relational_sine_exchange

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", capture)
    graph = _persistent_graph(
        internal_phase=(0, Q(1, 2048), Q(-1, 4096), Q(1, 4096), Q(-1, 2048)),
        capacity=(capacity,) * 5,
    )
    owned_before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = _persistence(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    assert calls == [1]
    assert owned_before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    assert report.family_admitted and report.source_set_trapping_certified
    assert report.source_family_membership == "certified_inside"
    assert report.source_joint_persistence_certified
    assert report.family_almost_everywhere_recurrence_certified
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
    assert report.pair_source_status == ("exact_nontip",) * 5
    assert all(report.pair_all_time_activity_certified)
    assert all(report.pair_all_time_circulation_certified)
    assert report.divergence == 0
    assert report.weighted_mean_rate_residual_bounds.contains(0)
    assert report.replica.comparison.storage_rate.contains(0)
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state, beta=beta)
        target = mp.matrix([2 * mp.pi * (i // 2) / 5 for i in range(10)])
        deviation = state[10:, 0] - target
        form_mean = sum(state[:10, 0]) / 10
        phase_mean = sum(deviation) / 10
        norm = sum(
            (state[i] - form_mean) ** 2 + (deviation[i] - phase_mean) ** 2
            for i in range(10)
        )
        energy = sum(
            (state[i] - state[j]) ** 2 / 2
            + _mp(beta) * (1 - mp.cos(state[10 + j] - state[10 + i]))
            for i, j in graph.edges
        )
        target_energy = 20 * _mp(beta) * (1 - mp.cos(2 * mp.pi / 5))
        assert norm <= _mp(report.norm_squared_upper_bound) < _mp(report.radius) ** 2
        assert 0 < energy - target_energy <= _mp(report.excess_storage_upper_bound)
        assert energy - target_energy < _mp(report.excess_ceiling)
        _contains(report.target_storage_bounds, target_energy)
        exact_gap = 5 - mp.sqrt(5)
        assert 0 < _mp(report.spectral_gap_lower_bound) <= exact_gap
        for pair, (left, right) in enumerate(PAIRS):
            u = (state[left] - state[right]) / 2
            delta = (state[10 + left] - state[10 + right]) / 2
            u_rate = (field[left] - field[right]) / 2
            delta_rate = (field[10 + left] - field[10 + right]) / 2
            y, y_rate = u / mp.sqrt(_mp(beta)), u_rate / mp.sqrt(_mp(beta))
            rotation = (y * delta_rate - delta * y_rate) / (delta**2 + y**2)
            _contains(report.internal_angular_speed_bounds, rotation)
            assert rotation > 0
            # Each pair can have a different instantaneous angular speed.
            assert report.pair_all_time_circulation_certified[pair]
        _contains(report.internal_phase_radius_bounds, _mp(report.radius) / mp.sqrt(2))
        assert _mp(report.internal_full_turn_time_bounds.lo) <= (
            2 * mp.pi**2 * mp.sqrt(_mp(beta)) / _mp(capacity)
        )
        assert report.internal_full_turn_time_bounds.lo > 0


def test_persistence_family_and_captured_trapping_have_separate_obligations():
    baseline = _persistence()
    broad = _persistence(excess_ceiling=baseline.barrier_lower_bound)
    assert not broad.family_admitted
    assert not broad.family_almost_everywhere_recurrence_certified
    assert broad.source_set_trapping_certified
    assert broad.source_joint_persistence_certified
    assert all(broad.pair_all_time_circulation_certified)
    assert broad.internal_full_turn_time_bounds is not None
    assert (
        "strict_family_excess_barrier_not_certified"
        in broad.family_unresolved_conditions
    )
    small = _persistence(excess_ceiling=Q(1, 2**40))
    assert small.family_admitted and small.source_set_trapping_certified
    assert small.source_family_membership == "unresolved"
    # An unsuccessful upper estimate is not a proof of nonmembership.
    mean_boundary = _persistence(form_mean_bounds=(0, 1))
    assert mean_boundary.family_admitted and mean_boundary.source_set_trapping_certified
    assert mean_boundary.source_family_membership == "outside"
    assert mean_boundary.source_mean_margins[0] == 0
    assert all(mean_boundary.pair_all_time_activity_certified)
    assert mean_boundary.individual_recurrence_status == "unavailable_for_chosen_state"
    distant = _persistence(_persistent_graph(form=(1, 0, 0, 0, 0)))
    assert distant.family_admitted
    assert not distant.source_set_trapping_certified
    assert not any(distant.pair_all_time_activity_certified)
    assert not any(distant.pair_all_time_circulation_certified)
    assert distant.source_family_membership == "unresolved"
    no_chart = _persistence(radius=1)
    assert not no_chart.family_admitted and not no_chart.source_set_trapping_certified
    assert no_chart.internal_angular_speed_bounds is None
    assert no_chart.internal_full_turn_time_bounds is None


def test_persistence_retains_tip_nontip_and_subquantum_distinctions():
    tip = _persistence(
        _persistent_graph(
            internal_form=(0, Q(1, 1024), Q(1, 1024), Q(1, 1024), Q(1, 1024))
        )
    )
    assert tip.family_admitted and tip.source_set_trapping_certified
    assert tip.pair_source_status[0] == "synchronized_tip"
    assert tip.pair_all_time_activity_certified == (False, True, True, True, True)
    assert tip.source_family_membership == "outside"
    assert not tip.source_joint_persistence_certified
    assert tip.replica.closed_form_phase_correlation_rates[0].lo == 0
    assert tip.replica.closed_form_phase_correlation_rates[0].hi == 0
    tiny = Q(1, 2**200)
    small = _persistence(
        _persistent_graph(
            internal_form=(tiny, Q(1, 1024), Q(1, 1024), Q(1, 1024), Q(1, 1024)),
            capacity=(tiny,) * 5,
        )
    )
    assert small.pair_source_status[0] == "exact_nontip"
    assert all(small.pair_all_time_activity_certified)
    assert all(small.pair_all_time_circulation_certified)
    assert small.internal_angular_speed_bounds.lo == 0
    assert small.internal_angular_speed_bounds.hi > 0
    assert small.normalized_angular_speed_lower_bound > 0
    assert small.internal_full_turn_time_bounds.lo > 0
    phase_active = _persistence(
        _persistent_graph(
            internal_form=(0, Q(1, 1024), Q(1, 1024), Q(1, 1024), Q(1, 1024)),
            internal_phase=(tiny, 0, 0, 0, 0),
        )
    )
    assert phase_active.pair_source_status[0] == "exact_nontip"
    assert phase_active.source_joint_persistence_certified


def test_persistence_preserves_origins_orientation_and_all_constituent_labels():
    original = _persistence()
    shifted = _persistence(
        _persistent_graph(
            form=(7,) * 5, phase=tuple(2 + Q(1287 * i, 1024) for i in range(5))
        ),
        phase_turns=(10**40,) * 10,
    )
    assert shifted.weighted_form_mean == 7
    assert shifted.source_family_membership == "outside"
    assert shifted.source_set_trapping_certified
    assert shifted.target_phase_turns == original.target_phase_turns
    assert shifted.norm_squared_upper_bound == original.norm_squared_upper_bound
    assert shifted.excess_storage_upper_bound == original.excess_storage_upper_bound
    negative = _persistence(
        _persistent_graph(phase=tuple(-Q(1287 * i, 1024) for i in range(5))), winding=-1
    )
    assert negative.family_admitted and negative.source_joint_persistence_certified
    assert negative.norm_squared_upper_bound == original.norm_squared_upper_bound
    swapped = _persistence(pairs=tuple(pair[::-1] for pair in PAIRS))
    assert swapped.source_joint_persistence_certified
    assert swapped.replica.comparison == original.replica.comparison
    assert swapped.pair_source_status == original.pair_source_status
    assert swapped.norm_squared_upper_bound == original.norm_squared_upper_bound


@pytest.mark.parametrize(
    "changes",
    (
        {"winding": True},
        {"winding": 0},
        {"radius": False},
        {"radius": float("nan")},
        {"excess_ceiling": 0},
        {"form_mean_bounds": (1, 1)},
        {"reference_model": RelationalExchangeModel(1, phase_domain="regular")},
        {"pairs": (PAIRS[0], PAIRS[2], PAIRS[1], PAIRS[3], PAIRS[4])},
    ),
)
def test_persistence_rejects_unsupported_declared_domains(changes):
    with pytest.raises((TypeError, ValueError)):
        _persistence(**changes)


@pytest.mark.parametrize("kind", ("nonuniform", "zero", "not_cycle"))
def test_persistence_requires_its_actual_full_support_and_common_capacity(kind):
    graph = _persistent_graph()
    if kind == "not_cycle":
        graph.remove_edges_from(product(PAIRS[0], PAIRS[-1]))
    else:
        for node in PAIRS[0]:
            graph.nodes[node]["nu_f"] = 2 if kind == "nonuniform" else 0
    with pytest.raises(ValueError):
        _persistence(graph)


def test_persistence_export_retains_scope_and_reuses_label_validation(tmp_path):
    report = _persistence()
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-replica-persistence.v1"
    assert (
        payload["report"]["individual_recurrence_status"]
        == "unavailable_for_chosen_state"
    )
    assert len(payload["report"]["replica"]["comparison"]["epi"]) == 10
    assert relational_report_to_dict(report)["report"] == payload["report"]
    path = tmp_path / "replica-persistence.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text()) == payload

    @dataclass(frozen=True)
    class Opaque:
        value: int

    invalid_replica = replace(report.replica, pairs=((Opaque(0), 1),) + PAIRS[1:])
    with pytest.raises(TypeError):
        replace(report, replica=invalid_replica).to_dict()


def _capacity(graph=None, **changes):
    arguments = dict(reference_model=MODEL, pairs=PAIRS)
    arguments.update(changes)
    return assess_sine_replica_capacity(
        _persistent_graph() if graph is None else graph, **arguments
    )


def _unequal_capacities(graph, values):
    for pair, capacities in zip(PAIRS, values):
        for node, capacity in zip(pair, capacities):
            graph.nodes[node]["nu_f"] = capacity
    return graph


@pytest.mark.parametrize("cyclic", (False, True))
def test_capacity_rows_and_joint_gram_are_exact_full_field_pushforwards(cyclic):
    beta = Q(3, 2)
    graph = _unequal_capacities(
        _graph(
            form=(Q(1, 4), Q(-1, 8), 0, Q(1, 2), Q(-1, 4)),
            internal_form=(Q(1, 8), Q(-1, 16), 0, Q(1, 32), Q(-1, 8)),
            internal_phase=(Q(1, 16), 0, Q(-1, 16), Q(1, 32), Q(-1, 8)),
        ),
        ((Q(3, 2), Q(1, 2)), (Q(1, 2), 1), (2, Q(1, 4)), (Q(3, 4), Q(5, 4)), (1, 1)),
    )
    if not cyclic:
        graph.remove_edges_from(product(PAIRS[0], PAIRS[-1]))
    report = _capacity(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    assert report.family is None
    assert report.capacity_aware_closure_certified
    assert report.all_time_internal_activity_status == "not_certified_by_this_reader"
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state, beta=beta)
        rho = [
            _mp(graph.degree[node]) / _mp(graph.nodes[node]["nu_f"]) for node in graph
        ]
        assert sum(report.normalized_form_mean_weights, Q(0)) == 1
        expected_mean = sum(weight * state[i] for i, weight in enumerate(rho)) / sum(
            rho
        )
        assert abs(_mp(report.weighted_form_mean) - expected_mean) < mp.mpf("1e-90")
        weighted_rate = sum(weight * field[i] for i, weight in enumerate(rho)) / sum(
            rho
        )
        assert abs(weighted_rate) < mp.mpf("1e-90")
        _contains(report.weighted_mean_rate_residual_bounds, weighted_rate)
        _contains(report.arithmetic_mean_rate_bounds, sum(field[:10, 0]) / 10)
        assert abs(sum(field[:10, 0]) / 10) > mp.mpf("1e-6")
        for block, (left, right) in enumerate(PAIRS):
            eta = (_mp(graph.nodes[left]["nu_f"]) - _mp(graph.nodes[right]["nu_f"])) / 2

            def observe(value):
                x = (value[left] + value[right]) / 2
                theta = (value[10 + left] + value[10 + right]) / 2
                u = (value[left] - value[right]) / 2
                delta = (value[10 + left] - value[10 + right]) / 2
                return mp.matrix(
                    (
                        x,
                        theta,
                        mp.cos(delta),
                        u * u,
                        u * mp.sin(delta),
                        eta * u,
                        eta * mp.sin(delta),
                        eta * eta,
                    )
                )

            expected = mp.diff(lambda time: observe(state + time * field), 0)
            values = observe(state)
            for direct, closed, residual, value in zip(
                report.capacity_aware_direct_rates[block],
                report.capacity_aware_generated_rates[block],
                report.capacity_aware_rate_residuals[block],
                expected,
            ):
                _contains(direct, value)
                _contains(closed, value)
                assert residual.contains(0)
            ordered_expected = (
                (field[left] + field[right]) / 2,
                (field[10 + left] + field[10 + right]) / 2,
                (field[left] - field[right]) / 2,
                (field[10 + left] - field[10 + right]) / 2,
            )
            for direct, closed, residual, value in zip(
                report.ordered_direct_rates[block],
                report.ordered_generated_rates[block],
                report.ordered_rate_residuals[block],
                ordered_expected,
            ):
                _contains(direct, value)
                _contains(closed, value)
                assert residual.contains(0)
            r, u_squared, q, p, t, e = values[2:8, 0]
            gram = mp.matrix([[u_squared, q, p], [q, 1 - r * r, t], [p, t, e]])
            factors = mp.matrix(
                [
                    (state[left] - state[right]) / 2,
                    mp.sin((state[10 + left] - state[10 + right]) / 2),
                    eta,
                ]
            )
            assert mp.norm(gram - factors * factors.T) < mp.mpf("1e-90")
            assert all(
                bound.contains(0)
                for bound in report.capacity_aware_constraint_residuals[block]
            )
            assert all(
                bound.contains(0)
                for bound in report.ordered_poisson_rate_residuals[block]
            )
        # This is a coupled rank-one Gram image, not independent admission of
        # squared identities for unconstrained proposed invariant coordinates.


def test_capacity_ordered_poisson_structure_generates_the_retained_fine_energy():
    beta = Q(5, 4)
    graph = _unequal_capacities(
        _graph(
            internal_form=(Q(1, 8), Q(-1, 16), 0, Q(1, 32), Q(-1, 8)),
            internal_phase=(Q(1, 16), 0, Q(-1, 16), Q(1, 32), Q(-1, 8)),
        ),
        ((Q(3, 2), Q(1, 2)), (Q(1, 2), 1), (2, Q(1, 4)), (Q(3, 4), Q(5, 4)), (1, 1)),
    )
    graph.remove_edges_from(product(PAIRS[0], PAIRS[-1]))
    report = _capacity(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    with mp.workdps(100):
        state = _state(graph)
        transform = mp.matrix(20)
        fine_bracket = mp.matrix(20)
        for pair, (left, right) in enumerate(PAIRS):
            for slot, offset, sign in ((0, 0, 1), (1, 10, 1), (2, 0, -1), (3, 10, -1)):
                transform[4 * pair + slot, offset + left] = mp.mpf(".5")
                transform[4 * pair + slot, offset + right] = sign * mp.mpf(".5")
        for node in graph:
            mobility = _mp(graph.nodes[node]["nu_f"]) / (
                _mp(beta) * mp.pi * graph.degree[node]
            )
            fine_bracket[node, 10 + node] = -mobility
            fine_bracket[10 + node, node] = mobility
        ordered = transform * state
        bracket = transform * fine_bracket * transform.T
        inverse = transform**-1

        def energy(coordinates):
            values = inverse * coordinates
            return sum(
                (values[i] - values[j]) ** 2 / 2
                + _mp(beta) * (1 - mp.cos(values[10 + j] - values[10 + i]))
                for i, j in graph.edges
            )

        gradient = mp.matrix(20, 1)
        for column in range(20):
            direction = mp.matrix([int(i == column) for i in range(20)])
            gradient[column] = mp.diff(
                lambda time: energy(ordered + time * direction), 0
            )
        generated = bracket * gradient
        actual = transform * _fine_field(graph, state, beta=beta)
        assert mp.norm(generated - actual) < mp.mpf("1e-90")
        for pair in range(5):
            diagonal, cross = report.ordered_poisson_coefficients[pair]
            _contains(diagonal, bracket[4 * pair, 4 * pair + 1])
            _contains(diagonal, bracket[4 * pair + 2, 4 * pair + 3])
            _contains(cross, bracket[4 * pair, 4 * pair + 3])
            _contains(cross, bracket[4 * pair + 2, 4 * pair + 1])
            for slot in range(4):
                _contains(
                    report.ordered_storage_gradients[pair][slot],
                    gradient[4 * pair + slot],
                )
        assert report.ordered_poisson_coefficients[0][1].hi < 0


@pytest.mark.parametrize("eta", (Q(1, 2), Q(1, 2**50)))
def test_exact_represented_asymmetry_can_stall_a_nontip_without_equilibrium(eta):
    epsilon, h = Q(1, 1024), Q(1287, 1024)
    graph = _unequal_capacities(
        _graph(
            form=(epsilon, 0, 0, 0, 0),
            phase=(0, h, 2 * h, -2 * h, -h),
            internal_form=(-eta * epsilon, 0, 0, 0, 0),
        ),
        ((1 + eta, 1 - eta),) + ((1, 1),) * 4,
    )
    turns = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1)
    report = _capacity(
        graph,
        phase_turns=turns,
        radius=Q(1, 16),
        excess_ceiling=Q(1, 32768),
        form_mean_bounds=(-1, 1),
    )
    assert report.pair_capacity_half_differences[0] == eta
    assert report.internal_form_squared[0] > 0
    assert report.family.family_admitted and report.family.source_set_trapping_certified
    assert report.family.source_family_membership == "certified_inside"
    assert report.angular_circulation_status == "not_certified_by_this_reader"
    with mp.workdps(100):
        # Raw neighbors +h and -h have exact odd-sine cancellation at pair0.
        # No rounded full graph is being identified with an exact pi twist.
        field = _fine_field(graph, _state(graph))
        assert abs((field[0] - field[1]) / 2) < mp.mpf("1e-90")
        assert abs((field[10] - field[11]) / 2) < mp.mpf("1e-90")
        mean_phase = (field[10] + field[11]) / 2
        assert mean_phase > 0
        assert abs(mean_phase - _mp(epsilon) * (1 - _mp(eta) ** 2) / mp.pi) < mp.mpf(
            "1e-90"
        )
        _contains(report.ordered_direct_rates[0][2], 0)
        _contains(report.ordered_direct_rates[0][3], 0)
        _contains(report.ordered_generated_rates[0][1], mean_phase)
        for bound in report.capacity_aware_direct_rates[0][2:]:
            _contains(bound, 0)
    with pytest.raises(ValueError):
        _assessment(graph, phase_turns=turns)
    with pytest.raises(ValueError):
        _persistence(graph, phase_turns=turns)


def test_capacity_tags_distinguish_tip_escape_and_state_only_swaps():
    graph = _unequal_capacities(
        _graph(
            form=(Q(1, 8), 0, 0, 0, 0),
            phase=(0,) * 5,
        ),
        ((Q(3, 2), Q(1, 2)),) + ((1, 1),) * 4,
    )
    tip = _capacity(graph)
    assert tip.ordered_direct_rates[0][3].lo > 0
    assert tip.capacity_aware_generated_rates[0][6].lo > 0
    assert all(
        bound.contains(0) for bound in tip.capacity_aware_generated_rates[0][2:5]
    )
    graph.nodes[0]["EPI"] += Q(1, 16)
    graph.nodes[1]["EPI"] -= Q(1, 16)
    original = _capacity(graph)
    states_swapped = graph.copy()
    for name in ("EPI", "theta"):
        states_swapped.nodes[0][name], states_swapped.nodes[1][name] = (
            states_swapped.nodes[1][name],
            states_swapped.nodes[0][name],
        )
    fixed = _capacity(states_swapped)
    for name in (
        "form_means",
        "resultant_magnitude_bounds",
        "internal_form_squared",
        "form_phase_correlation_bounds",
        "capacity_half_difference_squared",
    ):
        assert getattr(fixed, name) == getattr(original, name)
    assert fixed.capacity_form_correlation[0] == -original.capacity_form_correlation[0]
    assert (
        fixed.ordered_generated_rates[0][1].hi
        < original.ordered_generated_rates[0][1].lo
    )
    joint = states_swapped.copy()
    joint.nodes[0]["nu_f"], joint.nodes[1]["nu_f"] = (
        joint.nodes[1]["nu_f"],
        joint.nodes[0]["nu_f"],
    )
    relabeled = _capacity(joint)
    for name in (
        "capacity_half_difference_squared",
        "capacity_form_correlation",
        "capacity_phase_correlation_bounds",
        "capacity_aware_generated_rates",
        "weighted_form_mean",
    ):
        assert getattr(relabeled, name) == getattr(original, name)


def test_capacity_family_uses_weighted_mean_and_includes_synchronized_tips():
    graph = _unequal_capacities(_persistent_graph(), ((Q(3, 2), Q(1, 2)),) * 5)
    arguments = dict(
        radius=Q(1, 16), excess_ceiling=Q(1, 32768), form_mean_bounds=(-1, 1)
    )
    report = _capacity(graph, **arguments)
    assert report.weighted_form_mean == -Q(1, 2048)
    assert report.family.family_admitted and report.family.source_set_trapping_certified
    assert report.family.family_almost_everywhere_recurrence_certified
    assert report.family.individual_recurrence_status == "unavailable_for_chosen_state"
    restricted = _capacity(
        graph, **{**arguments, "form_mean_bounds": (-Q(1, 4096), Q(1, 4096))}
    )
    assert restricted.family.source_family_membership == "outside"
    assert restricted.family.source_set_trapping_certified
    assert sum(restricted.comparison.epi, Q(0)) == 0
    broad = _capacity(
        graph, **{**arguments, "excess_ceiling": report.family.barrier_lower_bound}
    )
    assert (
        not broad.family.family_admitted and broad.family.source_set_trapping_certified
    )
    tip_graph = _unequal_capacities(
        _persistent_graph(internal_form=(0,) * 5), ((Q(3, 2), Q(1, 2)),) * 5
    )
    tips = _capacity(tip_graph, **arguments)
    assert tips.family.source_family_membership == "certified_inside"
    assert tips.capacity_form_correlation == (0,) * 5
    symmetric = _persistence(_persistent_graph(internal_form=(0,) * 5))
    assert symmetric.source_family_membership == "outside"
    with mp.workdps(100):
        radius = mp.sqrt(2) * _mp(report.family.radius)
        assert _mp(report.family.family_form_coordinate_bounds.lo) <= -1 - radius
        assert _mp(report.family.family_form_coordinate_bounds.hi) >= 1 + radius


def test_capacity_equal_limit_and_raw_input_materialization_keep_existing_contracts():
    graph = _graph(
        internal_form=(Q(1, 8), 0, 0, 0, 0),
        internal_phase=(Q(1, 8), 0, 0, 0, 0),
        capacity=(1, 2, Q(1, 2), Q(3, 2), Q(3, 4)),
    )
    old, new = _assessment(graph), _capacity(graph)
    assert new.pair_capacity_means == old.pair_capacity
    assert new.pair_capacity_half_differences == (0,) * 5
    for name in (
        "form_means",
        "form_half_differences",
        "resultant_magnitude_bounds",
        "internal_form_squared",
        "form_phase_correlation_bounds",
    ):
        assert getattr(old, name) == getattr(new, name)
    assert new.ordered_direct_rates == tuple(
        zip(
            old.mean_form_rates,
            old.mean_phase_rates,
            old.internal_form_rates,
            old.internal_phase_rates,
        )
    )
    assert all(
        value == 0
        for value in new.capacity_half_difference_squared
        + new.capacity_form_correlation
    )
    raw = _persistent_graph()
    raw.nodes[0]["nu_f"], raw.nodes[1]["nu_f"] = 1 + Q(1, 2**200), 1 - Q(1, 2**200)
    captured = _capacity(raw)
    assert captured.comparison.capacity[:2] == (1, 1)
    assert captured.pair_capacity_half_differences[0] == 0
    assert _assessment(raw).pair_capacity[0] == 1
    # The input contrast was lost in shared binary staging; this is not an
    # asymmetric proof or a claim that arbitrary raw fractions are retained.


def test_capacity_optional_admission_export_and_single_capture(monkeypatch, tmp_path):
    import tnfr.physics.relational_sine_scale as owner

    calls = []
    capture = owner.bound_relational_sine_exchange

    def counted(*args, **kwargs):
        calls.append(1)
        return capture(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    report = _capacity(
        radius=Q(1, 16), excess_ceiling=Q(1, 32768), form_mean_bounds=(-1, 1)
    )
    assert calls == [1]
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-replica-capacity.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "capacity-replica.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    for missing in (
        dict(radius=Q(1, 16)),
        dict(excess_ceiling=Q(1, 32768), form_mean_bounds=(-1, 1)),
    ):
        with pytest.raises(ValueError, match="together"):
            _capacity(**missing)
    path = _persistent_graph()
    path.remove_edges_from(product(PAIRS[0], PAIRS[-1]))
    assert _capacity(path).family is None
    with pytest.raises(ValueError, match="C5"):
        _capacity(
            path, radius=Q(1, 16), excess_ceiling=Q(1, 32768), form_mean_bounds=(-1, 1)
        )
    path.nodes[0]["nu_f"] = 0
    with pytest.raises(ValueError, match="positive"):
        _capacity(path)


def _observed_partition(report):
    return {frozenset(pair) for pair in report.candidate_pairs or ()}


def test_phase_only_pairing_matches_independent_circular_nearest_without_other_state():
    labels = ("reed", 17, ("a", 2), "stone", -3, "cloud", ("b", 9), "elm", 21, "pond")
    phases = tuple(
        Q(1287 * i, 1024) + sign * Q(i + 1, 16384) for i in range(5) for sign in (1, -1)
    )
    report = observe_phase_pairs(nodes=labels, phases=phases)
    expected = {frozenset(labels[2 * i : 2 * i + 2]) for i in range(5)}
    assert report.status == "certified" and _observed_partition(report) == expected
    assert all(margin > 0 for margin in report.nearest_separation_margins)
    with mp.workdps(100):
        for i, j in enumerate(report.nearest_partner_indices):
            separations = {
                k: abs(
                    mp.atan2(
                        mp.sin(_mp(phases[k] - phases[i])),
                        mp.cos(_mp(phases[k] - phases[i])),
                    )
                )
                for k in range(10)
                if k != i
            }
            ordered = sorted(separations, key=separations.get)
            assert j == ordered[0] and separations[j] < separations[ordered[1]]
            assert separations[j] > 0  # The positive case is not just duplicates.
        for (i, j), bound in zip(report.distance_indices, report.squared_chord_bounds):
            _contains(bound, 2 * (1 - mp.cos(_mp(phases[j] - phases[i]))))
    # No graph, capacity, EPI or predeclared partition was supplied.
    shifted = observe_phase_pairs(
        nodes=labels, phases=tuple(value + Q(10**80, 3) for value in phases)
    )
    assert shifted.squared_chord_bounds == report.squared_chord_bounds
    assert shifted.nearest_partner_indices == report.nearest_partner_indices
    permutation = (7, 0, 9, 2, 4, 1, 8, 5, 3, 6)
    relabel = {label: f"id_{i*7+3}" for i, label in enumerate(labels)}
    shuffled = observe_phase_pairs(
        nodes=tuple(relabel[labels[i]] for i in permutation),
        phases=tuple(phases[i] for i in permutation),
    )
    assert _observed_partition(shuffled) == {
        frozenset(relabel[node] for node in pair) for pair in expected
    }


@pytest.mark.parametrize(
    "phases,reason",
    (
        ((0, 0, 0, 2), "exact_nearest_tie"),
        ((0, 1, -1, 3), "exact_nearest_tie"),
        ((0, Q(1, 2**200), Q(1, 2**199), 1), "strict_nearest_comparison_unresolved"),
        ((0, Q(1, 10), Q(3, 10), 1), "nearest_relation_not_mutual"),
        ((0, 1, 2), "odd_node_count_cannot_form_complete_pairs"),
        ((0,), "odd_node_count_cannot_form_complete_pairs"),
    ),
)
def test_phase_pairing_abstains_on_ties_interval_ambiguity_or_unmatched_nodes(
    phases, reason
):
    report = observe_phase_pairs(nodes=tuple(range(len(phases))), phases=phases)
    assert report.status == "unavailable" and report.candidate_pairs is None
    assert reason in report.reasons
    if reason == "strict_nearest_comparison_unresolved":
        assert report.node_statuses[0] == "unresolved_nearest_comparison"
        assert report.nearest_partner_indices[0] is None
    reordered = observe_phase_pairs(
        nodes=tuple(reversed(range(len(phases)))), phases=tuple(reversed(phases))
    )
    assert reordered.candidate_pairs is None


def test_observed_pairing_then_same_capture_law_and_family_admission(monkeypatch):
    import tnfr.physics.relational_sine_scale as owner

    graph = _persistent_graph(internal_phase=(Q(1, 4096),) * 5)
    graph = _unequal_capacities(graph, ((Q(3, 2), Q(1, 2)),) * 5)
    original = owner.bound_relational_sine_exchange
    calls = []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", capture)
    data = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = assess_sine_state_pairing(
        graph,
        reference_model=MODEL,
        pair_order=PAIRS,
        radius=Q(1, 16),
        excess_ceiling=Q(1, 32768),
        form_mean_bounds=(-1, 1),
    )
    assert calls == [1]
    assert data == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    assert report.observation.status == "certified"
    assert _observed_partition(report.observation) == {
        frozenset(pair) for pair in PAIRS
    }
    assert report.capacity_admission_status == "admitted"
    assert report.capacity.comparison is report.comparison
    assert report.capacity.family.source_set_trapping_certified
    assert report.all_time_pairing_persistence_certified
    assert (
        report.capacity.family.individual_recurrence_status
        == "unavailable_for_chosen_state"
    )
    assert (
        report.capacity.all_time_internal_activity_status
        == "not_certified_by_this_reader"
    )
    assert report.pair_order_source == "explicit"
    restricted = assess_sine_state_pairing(
        graph,
        reference_model=MODEL,
        pair_order=PAIRS,
        radius=Q(1, 16),
        excess_ceiling=Q(1, 32768),
        form_mean_bounds=(2, 3),
    )
    assert restricted.capacity.family.source_family_membership == "outside"
    assert restricted.capacity.family.source_set_trapping_certified
    assert restricted.all_time_pairing_persistence_certified
    # The optional family, not instantaneous phase matching, supplies persistence.
    instant = assess_sine_state_pairing(graph, reference_model=MODEL)
    assert instant.capacity_admission_status == "admitted"
    assert instant.pair_order_source == "capture_order_display_only"
    assert not instant.all_time_pairing_persistence_certified
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["theta"] += 2
        shifted.nodes[node]["EPI"] += 3
        shifted.nodes[node]["nu_f"] *= 2
    same = assess_sine_state_pairing(shifted, reference_model=MODEL)
    assert (
        same.observation.squared_chord_bounds
        == instant.observation.squared_chord_bounds
    )
    assert same.observation.candidate_pairs == instant.observation.candidate_pairs
    relabel = {i: f"node_{29-3*i}" for i in graph}
    renamed = assess_sine_state_pairing(
        nx.relabel_nodes(graph, relabel), reference_model=MODEL
    )
    assert _observed_partition(renamed.observation) == {
        frozenset(relabel[node] for node in pair) for pair in PAIRS
    }


def test_state_only_phase_permutation_keeps_candidate_and_reports_support_rejection():
    graph = _persistent_graph()
    graph.nodes[1]["theta"], graph.nodes[3]["theta"] = (
        graph.nodes[3]["theta"],
        graph.nodes[1]["theta"],
    )
    report = assess_sine_state_pairing(graph, reference_model=MODEL)
    expected = {frozenset((0, 3)), frozenset((1, 2))} | {
        frozenset(pair) for pair in PAIRS[2:]
    }
    assert report.observation.status == "certified"
    assert _observed_partition(report.observation) == expected
    assert report.capacity is None and report.capacity_admission_status == "rejected"
    assert any(
        "within-pair edges" in reason for reason in report.capacity_admission_reasons
    )
    assert not report.all_time_pairing_persistence_certified
    assert {frozenset(pair) for pair in report.pair_order} == expected
    with pytest.raises(ValueError, match="phase-observed"):
        assess_sine_state_pairing(graph, reference_model=MODEL, pair_order=PAIRS)
    # Supplying old topological twins cannot replace the observed partition.


def test_circular_pairing_does_not_infer_a_lift_and_small_support_is_separate():
    graph = _graph(phase=(0, 1, 2, -2, -1))
    graph.nodes[0]["theta"], graph.nodes[1]["theta"] = Q(25, 8), -Q(25, 8)
    refused = assess_sine_state_pairing(graph, reference_model=MODEL)
    assert refused.observation.status == "certified"
    assert refused.capacity_admission_status == "rejected"
    assert any("half-gaps" in reason for reason in refused.capacity_admission_reasons)
    lifted = assess_sine_state_pairing(
        graph, reference_model=MODEL, phase_turns=(0, 1) + (0,) * 8
    )
    assert lifted.observation == refused.observation
    assert lifted.capacity_admission_status == "admitted"
    assert lifted.phase_turns == (0, 1) + (0,) * 8
    assert not lifted.all_time_pairing_persistence_certified
    pair = nx.Graph()
    pair.add_edge("a", "b")
    nx.set_node_attributes(pair, {node: dict(EPI=0, theta=0, nu_f=1) for node in pair})
    pair.graph["GAMMA"] = {"type": "none"}
    small = assess_sine_state_pairing(pair, reference_model=MODEL)
    assert small.observation.status == "certified"
    assert small.capacity_admission_status == "rejected"
    assert any(
        "at least two pairs" in reason for reason in small.capacity_admission_reasons
    )


def test_pairing_invalid_inputs_are_errors_and_observer_work_is_bounded():
    consumed = []

    def endless_labels():
        index = 0
        while True:
            consumed.append(index)
            yield index
            index += 1

    with pytest.raises(ValueError, match="budget"):
        observe_phase_pairs(nodes=endless_labels(), phases=())
    assert len(consumed) == 65
    for phases in ((False, 0), (float("nan"), 0), (0,), (0, float("inf"))):
        with pytest.raises((ValueError, TypeError)):
            observe_phase_pairs(nodes=("a", "b"), phases=phases)
    with pytest.raises(ValueError, match="distinct"):
        observe_phase_pairs(nodes=("a", "a"), phases=(0, 1))
    graph = _persistent_graph()
    for changes in (
        dict(radius=Q(1, 16)),
        dict(phase_turns=(False,) * 10),
        dict(phase_turns=(0,) * 9),
        dict(pair_order=((0, 1),)),
        dict(pair_order=PAIRS, radius=-1, excess_ceiling=1, form_mean_bounds=(-1, 1)),
    ):
        with pytest.raises((ValueError, TypeError)):
            assess_sine_state_pairing(graph, reference_model=MODEL, **changes)
    with pytest.raises(ValueError, match="explicit pair_order"):
        assess_sine_state_pairing(
            graph,
            reference_model=MODEL,
            radius=Q(1, 16),
            excess_ceiling=Q(1, 32768),
            form_mean_bounds=(-1, 1),
        )


def test_unavailable_observation_never_falls_back_to_order_and_errors_propagate(
    monkeypatch,
):
    import tnfr.physics.relational_sine_scale as owner

    graph = _graph(phase=(0,) * 5)
    report = assess_sine_state_pairing(graph, reference_model=MODEL, pair_order=PAIRS)
    assert report.observation.candidate_pairs is None
    assert report.pair_order == PAIRS
    assert (
        report.capacity is None and report.capacity_admission_status == "not_attempted"
    )
    assert not report.all_time_pairing_persistence_certified
    positive_loss = assess_sine_state_pairing(
        _persistent_graph(),
        reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    )
    assert positive_loss.observation.status == "certified"
    assert positive_loss.capacity_admission_status == "rejected"

    def fail(*args, **kwargs):
        raise ArithmeticError("independent arithmetic failure")

    monkeypatch.setattr(owner, "_assess_sine_replica_capacity_comparison", fail)
    with pytest.raises(ArithmeticError, match="arithmetic failure"):
        assess_sine_state_pairing(_persistent_graph(), reference_model=MODEL)


def test_phase_pairing_reports_export_independent_scope(tmp_path):
    observation = observe_phase_pairs(
        nodes=("a", "b", "c", "d"), phases=(0, Q(1, 32), 2, Q(65, 32))
    )
    assert observation.status == "certified"
    payload = observation.to_dict()
    assert payload["schema"] == "tnfr.phase-pairs.v1"
    assert relational_report_to_dict(observation)["report"] == payload["report"]
    combined = assess_sine_state_pairing(_persistent_graph(), reference_model=MODEL)
    result = combined.to_dict()
    assert result["schema"] == "tnfr.relational-sine-state-pairing.v1"
    assert relational_report_to_dict(combined)["report"] == result["report"]
    path = tmp_path / "state-phase-pairing.json"
    export_to_json(result, path)
    assert json.loads(path.read_text()) == result


def test_acute_equilibrium_targets_reconstruct_and_satisfy_full_fine_rows():
    graph = _unequal_capacities(
        _persistent_graph(),
        ((Q(1, 2), Q(3, 2)), (2, 3), (Q(1, 4), 1), (4, Q(3, 4)), (1, 2)),
    )
    beta = Q(7, 3)
    model = RelationalExchangeModel(beta, epi_weight=0, phase_domain="regular")
    report = assess_sine_replica_equilibria(graph, reference_model=model, pairs=PAIRS)
    assert report.acute_equilibrium_classification_certified
    assert tuple(target.winding for target in report.targets) == (-1, 0, 1)
    assert report.source_equilibrium_status == "certified_not_equilibrium"
    with mp.workdps(100):
        nodes, n = report.comparison.nodes, len(graph)
        for target in report.targets:
            assert (
                target.target_geometry.sine_balance_status
                == "proved_by_odd_cancellation"
            )
            assert not any(target.target_geometry.symbolic_sine_coefficients)
            angles = [
                2 * mp.pi * _mp(value) + mp.mpf("0.375")
                for value in target.target_phase_turns
            ]
            state = mp.matrix([mp.mpf("-2.25")] * n + angles)
            field = _fine_field(graph, state, beta=model.storage_scale)
            assert max(abs(value) for value in field) < mp.mpf("1e-98")
            energy = mp.mpf(0)
            for edge, turn, bound in zip(
                graph.edges, target.target_edge_turns, target.edge_cosine_bounds
            ):
                i, j = (nodes.index(node) for node in edge)
                cosine = mp.cos(angles[j] - angles[i])
                _contains(bound, cosine)
                assert bound.lo > 0 and abs(turn) < Q(1, 4)
                energy += _mp(model.storage_scale) * (1 - cosine)
            _contains(target.storage_bounds, energy)
            assert abs(
                energy
                - 20
                * _mp(model.storage_scale)
                * (1 - mp.cos(2 * mp.pi * target.winding / 5))
            ) < mp.mpf("1e-98")
            reconstructed = target.target_geometry.nodal_turns
            for i in range(n):
                assert (
                    target.target_phase_turns[i]
                    - target.target_phase_turns[0]
                    - reconstructed[i]
                ) % 1 == 0
    assert report.targets[0].storage_bounds == report.targets[-1].storage_bounds
    assert (
        report.targets[1].storage_bounds.lo == report.targets[1].storage_bounds.hi == 0
    )


def test_captured_equilibrium_is_exact_consensus_not_small_rates_or_near_twist():
    consensus = _graph(form=(Q(-3, 2),) * 5, phase=(Q(7, 8),) * 5)
    report = assess_sine_replica_equilibria(
        consensus, reference_model=MODEL, pairs=PAIRS
    )
    assert report.source_acute_status == "fully_acute"
    assert report.source_equilibrium_status == "certified_consensus_equilibrium"
    assert report.source_equilibrium_winding == 0
    assert report.source_form_uniform and report.source_raw_phase_uniform
    observation = observe_phase_pairs(
        nodes=report.comparison.nodes, phases=report.comparison.phase
    )
    assert observation.status == "unavailable" and observation.candidate_pairs is None
    assert all(
        bound.lo == bound.hi == 0
        for bound in report.comparison.form_rates + report.comparison.phase_rates
    )
    tiny = Q(1, 2**200)
    for attribute, expected_reason in (
        ("theta", "acute_classification_excludes_nonuniform_rational_radian_phases"),
        ("EPI", "connected_positive_capacity_phase_row_requires_uniform_form"),
    ):
        graph = _graph(phase=(0,) * 5)
        graph.nodes[0][attribute] = tiny
        small = assess_sine_replica_equilibria(
            graph, reference_model=MODEL, pairs=PAIRS
        )
        assert small.source_equilibrium_status == "certified_not_equilibrium"
        assert small.source_equilibrium_reasons == (expected_reason,)
        assert small.source_equilibrium_winding is None
        assert all(
            bound.contains(0)
            for bound in small.comparison.form_rates + small.comparison.phase_rates
        )
    # Rounded twists remain distinct from exact symbolic turns.
    with mp.workdps(100):
        graph = _graph(phase=tuple(float(2 * mp.pi * i / 5) for i in range(5)))
        near = assess_sine_replica_equilibria(graph, reference_model=MODEL, pairs=PAIRS)
        assert near.source_acute_status == "fully_acute"
        assert near.source_equilibrium_status == "certified_not_equilibrium"
        field = _fine_field(graph, _state(graph))
        assert mp.mpf(0) < max(abs(value) for value in field) < mp.mpf("1e-15")


def test_equilibrium_classification_is_not_a_nonacute_classification_or_pair_chart():
    graph = _graph(phase=(0,) * 5)
    # An exact symbolic antipodal equilibrium has unequal structural twins.
    with mp.workdps(100):
        state = mp.matrix([mp.mpf(0)] * 20)
        state[10] = mp.pi
        assert max(abs(value) for value in _fine_field(graph, state)) < mp.mpf("1e-98")
        assert mp.cos(state[10] - state[12]) < 0
    graph.nodes[0]["theta"] = 4
    report = assess_sine_replica_equilibria(graph, reference_model=MODEL, pairs=PAIRS)
    assert report.source_acute_status == "outside_fully_acute_sector"
    assert report.source_equilibrium_status == "unavailable"
    assert report.acute_equilibrium_classification_certified
    with pytest.raises(ValueError, match="half-gaps"):
        assess_sine_replica_scale(graph, reference_model=MODEL, pairs=PAIRS)
    graph.nodes[4]["EPI"] = Q(1, 128)
    moving = assess_sine_replica_equilibria(graph, reference_model=MODEL, pairs=PAIRS)
    assert moving.source_equilibrium_status == "certified_not_equilibrium"
    assert moving.source_equilibrium_reasons == (
        "connected_positive_capacity_phase_row_requires_uniform_form",
    )
    # Further cycle windings are critical configurations outside the sector.
    with mp.workdps(100):
        graph = _graph()
        state = mp.matrix(
            [mp.mpf(0)] * 10 + [4 * mp.pi * (i // 2) / 5 for i in range(10)]
        )
        assert max(abs(value) for value in _fine_field(graph, state)) < mp.mpf("1e-98")
        assert mp.cos(4 * mp.pi / 5) < 0


def test_acute_source_boundary_and_numerical_unavailability_are_distinct(monkeypatch):
    import math

    import tnfr.physics.relational_sine_scale as owner

    before = math.nextafter(math.pi / 2, -math.inf)
    after = math.nextafter(math.pi / 2, math.inf)
    below = assess_sine_replica_equilibria(
        _graph(phase=(0, before, 0, 0, 0)), reference_model=MODEL, pairs=PAIRS
    )
    above = assess_sine_replica_equilibria(
        _graph(phase=(0, after, 0, 0, 0)), reference_model=MODEL, pairs=PAIRS
    )
    assert below.source_acute_status == "fully_acute"
    assert below.source_equilibrium_status == "certified_not_equilibrium"
    assert above.source_acute_status == "outside_fully_acute_sector"
    assert above.source_equilibrium_status == "unavailable"
    # A legitimate wider enclosure withholds admission, not a new verdict.
    original = owner.certified_cosine_bounds
    monkeypatch.setattr(
        owner,
        "certified_cosine_bounds",
        lambda value: (Q(-1), Q(1)) if value else original(value),
    )
    unresolved = assess_sine_replica_equilibria(
        _graph(phase=(0, before, 0, 0, 0)), reference_model=MODEL, pairs=PAIRS
    )
    assert unresolved.source_acute_status == "unresolved"
    assert unresolved.source_equilibrium_status == "unavailable"
    assert unresolved.acute_equilibrium_classification_certified


def test_equilibrium_reader_reuses_one_capture_and_is_capacity_and_label_covariant(
    monkeypatch,
):
    import tnfr.physics.relational_sine_scale as owner

    graph = _graph(phase=(0,) * 5)
    _unequal_capacities(graph, ((Q(1, 2**200), Q(3, 2)),) * 5)
    original = owner.bound_relational_sine_exchange
    calls = []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", capture)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = assess_sine_replica_equilibria(graph, reference_model=MODEL, pairs=PAIRS)
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    assert report.source_equilibrium_status == "certified_consensus_equilibrium"
    relabel = {i: f"constituent_{71 - 7*i}" for i in graph}
    renamed_graph = nx.relabel_nodes(graph, relabel)
    renamed = assess_sine_replica_equilibria(
        renamed_graph,
        reference_model=MODEL,
        pairs=tuple(tuple(relabel[node] for node in pair) for pair in PAIRS),
    )
    assert tuple(target.target_phase_turns for target in renamed.targets) == tuple(
        target.target_phase_turns for target in report.targets
    )
    reversed_order = assess_sine_replica_equilibria(
        graph, reference_model=MODEL, pairs=tuple(reversed(PAIRS))
    )
    old_positive, new_negative = (
        report.targets[-1].target_phase_turns,
        reversed_order.targets[0].target_phase_turns,
    )
    assert len({(old - new) % 1 for old, new in zip(old_positive, new_negative)}) == 1


def test_equilibrium_classification_domain_and_export(tmp_path):
    graph = _graph()
    report = assess_sine_replica_equilibria(graph, reference_model=MODEL, pairs=PAIRS)
    specialized = report.to_dict()
    assert specialized["schema"] == "tnfr.relational-sine-replica-equilibria.v1"
    assert relational_report_to_dict(report)["report"] == specialized["report"]
    destination = tmp_path / "acute-equilibria.json"
    export_to_json(specialized, destination)
    assert json.loads(destination.read_text()) == specialized
    zero = graph.copy()
    zero.nodes[0]["nu_f"] = 0
    with pytest.raises(ValueError, match="positive held capacity"):
        assess_sine_replica_equilibria(zero, reference_model=MODEL, pairs=PAIRS)
    with pytest.raises(ValueError, match="zero form loss"):
        assess_sine_replica_equilibria(
            graph,
            reference_model=RelationalExchangeModel(1, phase_domain="regular"),
            pairs=PAIRS,
        )
    with pytest.raises(ValueError, match="regular reference"):
        assess_sine_replica_equilibria(graph, reference_model=None, pairs=PAIRS)
    with pytest.raises(ValueError, match="four fine cross-edges"):
        missing = graph.copy()
        missing.remove_edge(0, 2)
        assess_sine_replica_equilibria(missing, reference_model=MODEL, pairs=PAIRS)
    with pytest.raises(ValueError, match="complete doubled C5"):
        assess_sine_replica_equilibria(
            graph,
            reference_model=MODEL,
            pairs=(PAIRS[0], PAIRS[2], PAIRS[1], PAIRS[3], PAIRS[4]),
        )
    # The prior public trapping API still requires a nonzero twist.
    with pytest.raises(ValueError, match="-1 or 1"):
        assess_sine_replica_persistence(
            graph,
            reference_model=MODEL,
            pairs=PAIRS,
            radius=Q(1, 16),
            excess_ceiling=Q(1, 32768),
            form_mean_bounds=(-1, 1),
            winding=0,
        )


def _frozen_grouping_transition(form_sign=1, amplitude=Q(1, 8)):
    graph = _graph()
    phases = (0, Q(1, 8), Q(1, 4), Q(3, 8), 1, 1, 2, 2, 3, 3)
    forms = (amplitude, -amplitude, amplitude, -amplitude, 0, 0, 0, 0, 0, 0)
    for node, phase, form in zip(graph, phases, forms):
        graph.nodes[node]["theta"] = phase
        graph.nodes[node]["EPI"] = form_sign * form
    return graph


def test_frozen_transition_is_complete_local_matching_and_opposite_nonmutuality():
    graph = _frozen_grouping_transition()
    report = assess_sine_pairing_transition(graph, reference_model=MODEL)
    assert report.observation.status == "unavailable"
    assert report.exact_nearest_groups == (
        (1,),
        (0, 2),
        (1, 3),
        (2,),
        (5,),
        (4,),
        (7,),
        (6,),
        (9,),
        (8,),
    )
    assert all(margin > 0 for margin in report.outside_group_distance_margins)
    assert report.forward.status == "certified_matching"
    assert report.forward.candidate_pairs == PAIRS
    assert report.forward.support_admission_status == "admitted"
    assert report.backward.status == "certified_nonmutual"
    assert report.backward.nearest_partner_indices == (1, 2, 1, 2, 5, 4, 7, 6, 9, 8)
    assert report.backward.candidate_pairs is None
    assert report.backward.support_admission_status == "not_attempted"
    for side in (report.backward, report.forward):
        assert side.local_interval_existence_certified
        assert side.certified_time_horizon is None
        assert side.tied_rate_coefficient_margins[1:3] == (Q(1, 2), Q(1, 2))
        assert all(side.tied_derivative_margin_bounds[i].lo > 0 for i in (1, 2))
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state)
        for numerator, phase_rate in zip(report.phase_rate_numerators, field[10:]):
            assert abs(_mp(numerator) / mp.pi - phase_rate) < mp.mpf("1e-99")
        for (i, j), bound in zip(
            report.observation.distance_indices, report.squared_chord_rate_bounds
        ):
            direct = (
                2
                * mp.sin(state[10 + j] - state[10 + i])
                * (field[10 + j] - field[10 + i])
            )
            _contains(bound, direct)
        for i in (1, 2):
            _contains(
                report.forward.tied_derivative_margin_bounds[i],
                mp.sin(mp.mpf(1) / 8) / mp.pi,
            )

        def distance(i, j):
            return 2 * (1 - mp.cos(state[10 + j] - state[10 + i]))

        for i, group in enumerate(report.exact_nearest_groups):
            for j in group:
                for k in range(10):
                    if k != i and k not in group:
                        assert distance(i, k) - distance(i, j) >= _mp(
                            report.outside_group_distance_margins[i]
                        )


def test_transition_direction_reversal_uses_form_hidden_from_phase_observer():
    positive = assess_sine_pairing_transition(
        _frozen_grouping_transition(), reference_model=MODEL
    )
    negative_graph = _frozen_grouping_transition(-1)
    negative = assess_sine_pairing_transition(negative_graph, reference_model=MODEL)
    assert positive.observation == negative.observation
    assert negative.phase_rate_numerators == tuple(
        -value for value in positive.phase_rate_numerators
    )
    assert negative.squared_chord_rate_bounds == tuple(
        -value for value in positive.squared_chord_rate_bounds
    )
    assert negative.backward.status == "certified_matching"
    assert negative.backward.candidate_pairs == PAIRS
    assert negative.forward.status == "certified_nonmutual"
    assert (
        negative.backward.nearest_partner_indices
        == positive.forward.nearest_partner_indices
    )
    assert (
        negative.forward.nearest_partner_indices
        == positive.backward.nearest_partner_indices
    )
    with mp.workdps(100):
        plus = _fine_field(
            _frozen_grouping_transition(), _state(_frozen_grouping_transition())
        )
        minus = _fine_field(negative_graph, _state(negative_graph))
        for i in range(10):
            assert abs(plus[i] - minus[i]) < mp.mpf("1e-98")
            assert abs(plus[10 + i] + minus[10 + i]) < mp.mpf("1e-98")


def test_first_order_transition_abstains_at_zero_derivative_and_distance_ambiguity():
    frozen_rates = assess_sine_pairing_transition(
        _frozen_grouping_transition(amplitude=0), reference_model=MODEL
    )
    assert frozen_rates.forward.status == frozen_rates.backward.status == "unavailable"
    assert frozen_rates.forward.node_statuses[1] == "first_derivative_order_unresolved"
    graph = _frozen_grouping_transition()
    for node in graph:
        graph.nodes[node]["theta"] = 0
    coincident = assess_sine_pairing_transition(graph, reference_model=MODEL)
    assert any(coincident.phase_rate_numerators)
    assert all(
        bound.lo == bound.hi == 0 for bound in coincident.squared_chord_rate_bounds
    )
    assert coincident.forward.status == coincident.backward.status == "unavailable"
    assert not coincident.forward.local_interval_existence_certified
    ambiguous = _frozen_grouping_transition()
    ambiguous.nodes[1]["theta"] = Q(1, 2**200)
    ambiguous.nodes[2]["theta"] = Q(1, 2**199)
    report = assess_sine_pairing_transition(ambiguous, reference_model=MODEL)
    assert report.exact_nearest_groups[0] is None
    assert report.forward.node_statuses[0] == "distance_comparison_unresolved"
    assert report.forward.status == report.backward.status == "unavailable"


def test_transition_exact_shared_factor_retains_tiny_strict_derivative_order():
    tiny = assess_sine_pairing_transition(
        _frozen_grouping_transition(amplitude=Q(1, 2**200)), reference_model=MODEL
    )
    assert tiny.forward.status == "certified_matching"
    assert tiny.backward.status == "certified_nonmutual"
    for i in (1, 2):
        assert tiny.forward.tied_rate_factor_bounds[i].lo > 0
        assert tiny.forward.tied_rate_coefficient_margins[i] > 0
        assert tiny.forward.tied_derivative_margin_bounds[i].lo == 0
    assert tiny.forward.certified_time_horizon is None


def test_transition_support_is_independent_and_all_competitors_are_required():
    graph = _persistent_graph()
    graph.nodes[1]["theta"], graph.nodes[3]["theta"] = (
        graph.nodes[3]["theta"],
        graph.nodes[1]["theta"],
    )
    rejected = assess_sine_pairing_transition(graph, reference_model=MODEL)
    for side in (rejected.forward, rejected.backward):
        assert side.status == "certified_matching"
        assert side.support_admission_status == "rejected"
        assert any(
            "within-pair edges" in reason for reason in side.support_admission_reasons
        )
    incomplete = _frozen_grouping_transition()
    for i in (4, 5, 6):
        incomplete.nodes[i]["theta"] = 1
    partial = assess_sine_pairing_transition(incomplete, reference_model=MODEL)
    assert partial.forward.nearest_partner_indices[:4] == (1, 0, 3, 2)
    assert partial.forward.status == "unavailable"
    assert partial.forward.candidate_pairs is None


def test_transition_single_capture_scope_covariance_and_export(monkeypatch, tmp_path):
    import tnfr.physics.relational_sine_scale as owner

    graph = _frozen_grouping_transition()
    original, calls = owner.bound_relational_sine_exchange, []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", capture)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = assess_sine_pairing_transition(graph, reference_model=MODEL)
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["theta"] += 4
        shifted.nodes[node]["EPI"] += 2
        shifted.nodes[node]["nu_f"] *= 2
    doubled = assess_sine_pairing_transition(shifted, reference_model=MODEL)
    assert (
        doubled.observation.squared_chord_bounds
        == report.observation.squared_chord_bounds
    )
    assert doubled.phase_rate_numerators == tuple(
        2 * value for value in report.phase_rate_numerators
    )
    assert doubled.forward.candidate_pairs == report.forward.candidate_pairs
    assert (
        doubled.backward.nearest_partner_indices
        == report.backward.nearest_partner_indices
    )
    labels = {i: f"n{37-3*i}" for i in graph}
    renamed = assess_sine_pairing_transition(
        nx.relabel_nodes(graph, labels), reference_model=MODEL
    )
    assert renamed.forward.candidate_pairs == tuple(
        tuple(labels[node] for node in pair) for pair in PAIRS
    )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-pairing-transition.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "grouping-transition.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload

    @dataclass(frozen=True)
    class OpaqueTransitionLabel:
        index: int

    forged_observation = replace(
        report.observation, candidate_pairs=((OpaqueTransitionLabel(0), 1),)
    )
    with pytest.raises(TypeError):
        replace(report, observation=forged_observation).to_dict()
    with pytest.raises(ValueError, match="regular reference"):
        assess_sine_pairing_transition(graph, reference_model=None)
    invalid = graph.copy()
    invalid.nodes[0]["EPI"] = float("nan")
    with pytest.raises(ValueError):
        assess_sine_pairing_transition(invalid, reference_model=MODEL)


def _frozen_grouping_window(graph=None, **overrides):
    graph = _frozen_grouping_transition() if graph is None else graph
    arguments = dict(
        reference_model=MODEL,
        form_error_bounds=(Q(1, 2**20),) * len(graph),
        phase_error_bounds=(Q(1, 2**20),) * len(graph),
        window_start=Q(1, 128),
        window_end=Q(1, 64),
    )
    arguments.update(overrides)
    return assess_sine_pairing_window(graph, **arguments)


def test_frozen_full_box_window_certifies_both_complete_nearest_relations():
    positive = _frozen_grouping_window()
    negative = _frozen_grouping_window(_frozen_grouping_transition(-1))
    assert positive.window == negative.window == (Q(1, 128), Q(1, 64))
    assert (
        positive.form_error_bounds == positive.phase_error_bounds == (Q(1, 2**20),) * 10
    )
    assert positive.status == "certified_matching"
    assert positive.candidate_pairs == PAIRS
    assert positive.support_admission_status == "admitted"
    assert negative.status == "certified_nonmutual"
    assert negative.nearest_partner_indices == (1, 2, 1, 2, 5, 4, 7, 6, 9, 8)
    assert negative.candidate_pairs is None
    assert negative.support_admission_status == "not_attempted"
    for report in (positive, negative):
        assert report.whole_window_nearest_relation_certified
        assert all(margin > 0 for margin in report.nearest_separation_margins)
        assert len(report.initial_form_bounds) + len(report.initial_phase_bounds) == 20
        assert all(
            bound.width > 0
            for bound in report.initial_form_bounds + report.initial_phase_bounds
        )
        assert set(report.chord_bound_methods) == {
            "monotone_scalar_endpoints_on_absolute_gap_within_pi"
        }
        lookup = dict(zip(report.distance_indices, report.squared_chord_window_bounds))
        for i, j in enumerate(report.nearest_partner_indices):
            chosen = lookup[min(i, j), max(i, j)]
            for k in range(10):
                if k not in (i, j):
                    other = lookup[min(i, k), max(i, k)]
                    assert (
                        other.lo - chosen.hi >= report.nearest_separation_margins[i] > 0
                    )
    assert negative.phase_rate_numerators == tuple(
        -value for value in positive.phase_rate_numerators
    )
    assert (
        negative.phase_remainder_upper_bounds == positive.phase_remainder_upper_bounds
    )
    # The old qualitative reader is not silently promoted to this time budget.
    local = assess_sine_pairing_transition(
        _frozen_grouping_transition(), reference_model=MODEL
    )
    assert local.forward.certified_time_horizon is None


def test_full_field_remainder_and_uncertain_rows_have_independent_controls():
    graph = nx.path_graph(4)
    for i in graph:
        graph.nodes[i].update(
            EPI=Q(2 * i - 1, 8), theta=Q(i * i - 2, 4), nu_f=(Q(1, 2), Q(3, 2), 2, 0)[i]
        )
    graph.graph["GAMMA"] = {"type": "none"}
    model = RelationalExchangeModel(Q(3, 2), epi_weight=0, phase_domain="regular")
    rx, rt = (Q(1, 1024), Q(3, 2048), Q(1, 512), Q(0)), (
        Q(1, 512),
        Q(1, 1024),
        Q(1, 256),
        Q(1, 2048),
    )
    report = assess_sine_pairing_window(
        graph,
        reference_model=model,
        form_error_bounds=rx,
        phase_error_bounds=rt,
        window_start=Q(1, 128),
        window_end=Q(1, 64),
    )
    with mp.workdps(100):
        base = _state(graph)
        center_field = _fine_field(graph, base, beta=model.storage_scale)
        for i in graph:
            assert abs(
                center_field[4 + i] - _mp(report.phase_rate_numerators[i]) / mp.pi
            ) < mp.mpf("1e-98")
            # Independent corner maximizing this actual linear phase row.
            corner = base.copy()
            for j in graph:
                corner[j] += _mp(rx[j]) * (1 if j == i else -1)
            field = _fine_field(graph, corner, beta=model.storage_scale)
            difference = field[4 + i] - center_field[4 + i]
            expected = (
                _mp(graph.nodes[i]["nu_f"])
                / (_mp(model.storage_scale) * mp.pi)
                * (_mp(rx[i]) + sum(_mp(rx[j]) for j in graph[i]) / graph.degree[i])
            )
            assert abs(difference - expected) < mp.mpf("1e-98")
            assert difference <= _mp(report.initial_phase_rate_error_bounds[i])
        # Several fixed full-coordinate corners exercise nonlinear feedback.
        for form_sign, phase_sign in product((-1, 1), repeat=2):
            corner = base.copy()
            for i in graph:
                corner[i] += form_sign * (-1) ** i * _mp(rx[i])
                corner[4 + i] += phase_sign * (-1) ** i * _mp(rt[i])
            field = _fine_field(graph, corner, beta=model.storage_scale)
            for i in graph:
                acceleration = (
                    _mp(graph.nodes[i]["nu_f"])
                    / (_mp(model.storage_scale) * mp.pi)
                    * (field[i] - sum(field[j] for j in graph[i]) / graph.degree[i])
                )
                assert abs(field[i]) <= _mp(report.form_speed_upper_bounds[i])
                assert abs(acceleration) <= _mp(
                    report.phase_acceleration_upper_bounds[i]
                )
        assert (
            report.form_speed_upper_bounds[-1]
            == report.initial_phase_rate_error_bounds[-1]
            == report.phase_acceleration_upper_bounds[-1]
            == 0
        )
        # Independent unit-capacity formula recovers the frozen analytic budget.
        fixed = _frozen_grouping_window()
        rho, t = mp.mpf(2) ** -20, mp.mpf(1) / 64
        exact = rho + 2 * rho * t / mp.pi + t * t / mp.pi**2
        assert (
            exact
            <= _mp(fixed.phase_remainder_upper_bounds[0])
            < exact + mp.mpf("1e-36")
        )


def test_window_chords_cover_full_affine_error_boxes_and_do_not_hold_pairs_exact():
    report = _frozen_grouping_window()
    with mp.workdps(100):
        for t in (Q(1, 128), Q(3, 256), Q(1, 64)):
            for sign in (-1, 1):
                phase = [
                    _mp(center)
                    + _mp(rate) * _mp(t) / mp.pi
                    + sign * (-1) ** i * _mp(report.phase_remainder_upper_bounds[i])
                    for i, (center, rate) in enumerate(
                        zip(report.comparison.phase, report.phase_rate_numerators)
                    )
                ]
                assert phase[4] != phase[5] and phase[6] != phase[7]
                for i, bound in enumerate(report.phase_window_bounds):
                    _contains(bound, phase[i])
                for (i, j), gap, chord in zip(
                    report.distance_indices,
                    report.phase_gap_window_bounds,
                    report.squared_chord_window_bounds,
                ):
                    _contains(gap, phase[j] - phase[i])
                    _contains(chord, 2 * (1 - mp.cos(phase[j] - phase[i])))
                for i, j in enumerate(report.nearest_partner_indices):
                    chosen = 2 * (1 - mp.cos(phase[j] - phase[i]))
                    assert all(
                        chosen < 2 * (1 - mp.cos(phase[k] - phase[i]))
                        for k in range(10)
                        if k not in (i, j)
                    )


def test_window_wide_bounds_abstain_and_support_remains_independent():
    wide = _frozen_grouping_window(
        form_error_bounds=(Q(1, 8),) * 10, phase_error_bounds=(Q(1, 8),) * 10
    )
    assert wide.status == "unavailable"
    assert not wide.whole_window_nearest_relation_certified
    assert wide.candidate_pairs is None
    initial_ties = _frozen_grouping_window(window_start=0)
    assert initial_ties.status == "unavailable"
    assert not initial_ties.whole_window_nearest_relation_certified
    graph = _persistent_graph()
    graph.nodes[1]["theta"], graph.nodes[3]["theta"] = (
        graph.nodes[3]["theta"],
        graph.nodes[1]["theta"],
    )
    rejected = _frozen_grouping_window(graph)
    assert rejected.status == "certified_matching"
    assert rejected.support_admission_status == "rejected"
    assert any(
        "within-pair edges" in reason for reason in rejected.support_admission_reasons
    )
    # A larger raw chart invokes global trigonometry, never an inferred unwrap.
    lifted = _frozen_grouping_transition()
    lifted.nodes[0]["theta"] = 10
    fallback = _frozen_grouping_window(lifted)
    assert (
        "shared_interval_cosine_without_lift_inference" in fallback.chord_bound_methods
    )
    with mp.workdps(100):
        for gap, chord, method in zip(
            fallback.phase_gap_window_bounds,
            fallback.squared_chord_window_bounds,
            fallback.chord_bound_methods,
        ):
            if method.startswith("shared_interval"):
                for endpoint in (gap.lo, gap.midpoint, gap.hi):
                    _contains(chord, 2 * (1 - mp.cos(_mp(endpoint))))


def test_window_admission_single_capture_source_purity_and_export(
    monkeypatch, tmp_path
):
    import tnfr.physics.relational_sine_scale as owner

    graph = _frozen_grouping_transition()
    original, calls = owner.bound_relational_sine_exchange, []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", capture)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = _frozen_grouping_window(graph)
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["theta"] += 4
        shifted.nodes[node]["EPI"] += 8
    same = _frozen_grouping_window(shifted)
    assert same.phase_gap_window_bounds == report.phase_gap_window_bounds
    assert same.squared_chord_window_bounds == report.squared_chord_window_bounds
    assert same.candidate_pairs == report.candidate_pairs
    nondyadic = _frozen_grouping_window(
        form_error_bounds=(Q(1, 3),) * 10,
        phase_error_bounds=(Q(1, 7),) * 10,
    )
    assert nondyadic.form_error_bounds == (Q(1, 3),) * 10
    assert nondyadic.phase_error_bounds == (Q(1, 7),) * 10
    # Exact radii define the source; grid-rounded display intervals are wider.
    assert nondyadic.initial_form_bounds[4].lo < -Q(1, 3)
    assert nondyadic.initial_form_bounds[4].hi > Q(1, 3)
    assert nondyadic.initial_phase_bounds[0].lo < -Q(1, 7)
    assert nondyadic.initial_phase_bounds[0].hi > Q(1, 7)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-pairing-window.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "pairing-window.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload

    @dataclass(frozen=True)
    class OpaqueWindowLabel:
        index: int

    with pytest.raises(TypeError):
        replace(
            report, candidate_pairs=((OpaqueWindowLabel(0), 1),) + PAIRS[1:]
        ).to_dict()
    for changes in (
        dict(window_start=-1),
        dict(window_start=Q(1, 64)),
        dict(window_end=0),
        dict(window_start=True),
        dict(window_end=float("inf")),
        dict(form_error_bounds=(0,) * 9),
        dict(phase_error_bounds=(0,) * 11),
        dict(form_error_bounds=(-1,) * 10),
        dict(phase_error_bounds=(True,) * 10),
        dict(form_error_bounds=(float("nan"),) * 10),
    ):
        with pytest.raises((TypeError, ValueError)):
            _frozen_grouping_window(**changes)
    with pytest.raises(ValueError, match="zero form loss"):
        _frozen_grouping_window(
            reference_model=RelationalExchangeModel(1, phase_domain="regular")
        )
    with pytest.raises(ValueError, match="regular reference"):
        _frozen_grouping_window(reference_model=None)


def test_missing_edge_swap_keeps_pair_orbits_but_changes_full_law_mean_rate():
    graph = _frozen_grouping_transition()
    graph.remove_edge(0, 8)
    report = assess_sine_pair_support_symmetry(
        graph, reference_model=MODEL, pairs=PAIRS, witness_pair=0
    )
    assert not report.independent_pair_swaps_equivariant
    assert report.pair_swap_symmetry == (False, True, True, True, False)
    assert report.incomplete_cross_blocks == ((0, 4, 3),)
    assert report.within_pair_edge_indices == ()
    assert report.strict_replica_admission_status == "rejected"
    witness = report.witness
    assert witness.source_orbit_equal
    assert witness.collective_phase_rate_obstruction_certified
    assert witness.source_pair_mean_phase_rate_numerators[4] == Q(1, 48)
    assert witness.swapped_pair_mean_phase_rate_numerators[4] == -Q(1, 48)
    assert witness.pair_mean_phase_rate_numerator_difference[4] == -Q(1, 24)
    assert witness.swapped_phase_rate_numerators[8] == -Q(1, 24)
    with mp.workdps(100):
        original = _state(graph)
        swapped = mp.matrix(
            [_mp(value) for value in witness.swapped_form + witness.swapped_phase]
        )
        original_field, swapped_field = _fine_field(graph, original), _fine_field(
            graph, swapped
        )
        for i in range(10):
            _contains(witness.swapped_form_rate_bounds[i], swapped_field[i])
            _contains(witness.swapped_phase_rate_bounds[i], swapped_field[10 + i])
            assert abs(
                swapped_field[10 + i]
                - _mp(witness.swapped_phase_rate_numerators[i]) / mp.pi
            ) < mp.mpf("1e-98")
        for pair_index, (i, j) in enumerate(PAIRS):
            difference = (
                swapped_field[10 + i]
                + swapped_field[10 + j]
                - original_field[10 + i]
                - original_field[10 + j]
            ) / 2
            _contains(
                witness.pair_mean_phase_rate_difference_bounds[pair_index], difference
            )

            # All five existing unordered coordinates have the same values.
            def invariants(state):
                u = (state[i] - state[j]) / 2
                delta = (state[10 + i] - state[10 + j]) / 2
                return (
                    (state[i] + state[j]) / 2,
                    (state[10 + i] + state[10 + j]) / 2,
                    mp.cos(delta),
                    u * u,
                    u * mp.sin(delta),
                )

            assert max(
                abs(left - right)
                for left, right in zip(invariants(original), invariants(swapped))
            ) < mp.mpf("1e-98")
        assert abs(
            (
                swapped_field[18]
                + swapped_field[19]
                - original_field[18]
                - original_field[19]
            )
            / 2
            + 1 / (24 * mp.pi)
        ) < mp.mpf("1e-98")
        # These exact-tie source observations do not certify future grouping;
        # the separate whole-window assessment below supplies that evidence.
    before = observe_phase_pairs(
        nodes=report.comparison.nodes, phases=report.comparison.phase
    )
    after = observe_phase_pairs(
        nodes=report.comparison.nodes, phases=witness.swapped_phase
    )
    assert before.squared_chord_bounds != after.squared_chord_bounds
    # Reuse the frozen window unchanged on this actual broken support.
    # Observable grouping can remain robust while collective closure fails.
    window = _frozen_grouping_window(graph)
    assert window.window == (Q(1, 128), Q(1, 64))
    assert window.form_error_bounds == window.phase_error_bounds == (Q(1, 2**20),) * 10
    assert window.status == "certified_matching"
    assert window.candidate_pairs == PAIRS
    assert window.support_admission_status == "rejected"
    assert sorted(
        tuple(
            sorted((report.comparison.epi[i], report.comparison.phase[i]) for i in pair)
        )
        for pair in PAIRS
    ) == sorted(
        tuple(sorted((witness.swapped_form[i], witness.swapped_phase[i]) for i in pair))
        for pair in PAIRS
    )


def test_swap_symmetry_allows_internal_edges_without_promoting_old_replica_rows():
    graph = _frozen_grouping_transition()
    graph.add_edge(0, 1)
    graph.add_edge(6, 7)
    report = assess_sine_pair_support_symmetry(
        graph, reference_model=MODEL, pairs=PAIRS, witness_pair=0
    )
    assert report.independent_pair_swaps_equivariant
    assert report.within_pair_edge_indices == (0, 3)
    assert report.incomplete_cross_blocks == ()
    assert report.phase_row_commutator_witnesses == (None,) * 5
    assert report.strict_replica_admission_status == "rejected"
    assert any(
        "within-pair edges" in reason
        for reason in report.strict_replica_admission_reasons
    )
    with pytest.raises(ValueError, match="within-pair edges"):
        assess_sine_replica_scale(graph, reference_model=MODEL, pairs=PAIRS)
    with mp.workdps(100):
        source = _state(graph)
        source_field = _fine_field(graph, source)
        # Each independent generator, with all other labels and support fixed.
        for left, right in PAIRS:
            swapped = source.copy()
            swapped[left], swapped[right] = source[right], source[left]
            swapped[10 + left], swapped[10 + right] = (
                source[10 + right],
                source[10 + left],
            )
            field = _fine_field(graph, swapped)
            permutation = list(range(10))
            permutation[left], permutation[right] = right, left
            for i, j in enumerate(permutation):
                assert abs(field[i] - source_field[j]) < mp.mpf("1e-98")
                assert abs(field[10 + i] - source_field[10 + j]) < mp.mpf("1e-98")
    assert not report.witness.collective_phase_rate_obstruction_certified
    assert report.witness.phase_equivariance_numerator_residual == (Q(0),) * 10
    pair = nx.Graph()
    pair.add_edge("left", "right")
    pair.graph["GAMMA"] = {"type": "none"}
    nx.set_node_attributes(
        pair,
        {
            "left": dict(EPI=1, theta=Q(1, 4), nu_f=1),
            "right": dict(EPI=-1, theta=-Q(1, 4), nu_f=1),
        },
    )
    single = assess_sine_pair_support_symmetry(
        pair, reference_model=MODEL, pairs=(("left", "right"),), witness_pair=0
    )
    assert (
        single.independent_pair_swaps_equivariant
        and single.within_pair_edge_indices == (0,)
    )
    assert single.strict_replica_admission_status == "rejected"


def test_equitable_support_is_not_independent_swap_equivariance():
    graph = _frozen_grouping_transition()
    graph.remove_edges_from(tuple(graph.edges))
    for i, (left, right) in enumerate(PAIRS):
        next_left, next_right = PAIRS[(i + 1) % 5]
        graph.add_edges_from(((left, right), (left, next_left), (right, next_right)))
    # Neighbor counts match inside every block, but attachments do not.
    for left, right in PAIRS:
        assert tuple(
            sum(node in graph[left] for node in pair) for pair in PAIRS
        ) == tuple(sum(node in graph[right] for node in pair) for pair in PAIRS)
    report = assess_sine_pair_support_symmetry(
        graph, reference_model=MODEL, pairs=PAIRS, witness_pair=0
    )
    assert report.pair_swap_symmetry == (False,) * 5
    assert report.unordered_pair_quotient_status == "obstructed_by_fixed_support"
    assert all(item[2] == 2 for item in report.incomplete_cross_blocks)
    # Verify necessity directly from each exact normalized phase matrix.
    for pair, witness in zip(PAIRS, report.phase_row_commutator_witnesses):
        row, column, numerator = witness
        permutation = list(range(10))
        permutation[pair[0]], permutation[pair[1]] = pair[1], pair[0]

        def entry(i, j):
            return Q(int(i == j)) - Q(int(j in graph[i]), graph.degree[i])

        assert numerator == entry(permutation[row], permutation[column]) - entry(
            row, column
        )
        assert numerator != 0
    # This particular projected phase mean can fail to detect the obstruction.
    assert not report.witness.collective_phase_rate_obstruction_certified
    assert any(report.witness.phase_equivariance_numerator_residual)
    uniform = graph.copy()
    for node in uniform:
        uniform.nodes[node].update(EPI=0, theta=0)
    stationary = assess_sine_pair_support_symmetry(
        uniform, reference_model=MODEL, pairs=PAIRS, witness_pair=0
    )
    assert not stationary.independent_pair_swaps_equivariant
    assert not stationary.witness.collective_phase_rate_obstruction_certified
    assert stationary.witness.phase_equivariance_numerator_residual == (Q(0),) * 10


def test_support_symmetry_capture_relabeling_domains_and_export(monkeypatch, tmp_path):
    import tnfr.physics.relational_sine_scale as owner

    graph = _frozen_grouping_transition()
    graph.remove_edge(0, 8)
    original, calls = owner.bound_relational_sine_exchange, []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", capture)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = assess_sine_pair_support_symmetry(
        graph, reference_model=MODEL, pairs=PAIRS, witness_pair=0
    )
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    labels = {i: f"member_{71-3*i}" for i in graph}
    renamed = assess_sine_pair_support_symmetry(
        nx.relabel_nodes(graph, labels),
        reference_model=MODEL,
        pairs=tuple(tuple(labels[node] for node in pair) for pair in PAIRS),
        witness_pair=0,
    )
    assert renamed.pair_swap_symmetry == report.pair_swap_symmetry
    assert (
        renamed.phase_row_commutator_witnesses == report.phase_row_commutator_witnesses
    )
    assert (
        renamed.witness.pair_mean_phase_rate_numerator_difference
        == report.witness.pair_mean_phase_rate_numerator_difference
    )
    absent = assess_sine_pair_support_symmetry(
        graph, reference_model=MODEL, pairs=PAIRS
    )
    assert absent.witness is None
    assert (
        absent.unordered_pair_quotient_status == report.unordered_pair_quotient_status
    )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-pair-support-symmetry.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "support-symmetry.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    for invalid in (True, -1, 5, Q(0), "0"):
        with pytest.raises(ValueError, match="nonboolean pair index"):
            assess_sine_pair_support_symmetry(
                graph, reference_model=MODEL, pairs=PAIRS, witness_pair=invalid
            )
    for bad_pairs in (PAIRS[:-1], PAIRS[:-1] + ((0, 9),), PAIRS[:-1] + ((8, 9, 7),)):
        with pytest.raises(ValueError):
            assess_sine_pair_support_symmetry(
                graph, reference_model=MODEL, pairs=bad_pairs
            )
    with pytest.raises(ValueError, match="zero form loss"):
        assess_sine_pair_support_symmetry(
            graph,
            reference_model=RelationalExchangeModel(1, phase_domain="regular"),
            pairs=PAIRS,
        )
    for value in (0, Q(3, 2)):
        invalid = graph.copy()
        invalid.nodes[0]["nu_f"] = value
        with pytest.raises(ValueError, match="common positive"):
            assess_sine_pair_support_symmetry(
                invalid, reference_model=MODEL, pairs=PAIRS
            )

    @dataclass(frozen=True)
    class OpaqueSupportLabel:
        index: int

    with pytest.raises(TypeError):
        replace(report, pairs=((OpaqueSupportLabel(0), 1),) + PAIRS[1:]).to_dict()


def _mixed(graph=None, **changes):
    if graph is None:
        graph = _frozen_grouping_transition()
        graph.remove_edge(0, 8)
    arguments = dict(reference_model=MODEL, pairs=PAIRS)
    arguments.update(changes)
    return assess_sine_mixed_pair_state(graph, **arguments)


def test_mixed_state_repairs_the_known_derivative_collision_and_ordered_tip():
    graph = _frozen_grouping_transition()
    graph.remove_edge(0, 8)
    original = _mixed(graph)
    swapped = graph.copy()
    for name in ("EPI", "theta"):
        swapped.nodes[0][name], swapped.nodes[1][name] = (
            graph.nodes[1][name],
            graph.nodes[0][name],
        )
    other = _mixed(swapped)
    assert tuple(item.mode for item in original.coordinates) == (
        "ordered",
        "unordered",
        "unordered",
        "unordered",
        "ordered",
    )
    assert original.exact_mixed_coordinates_identify_allowed_swap_orbits
    assert original.mixed_coordinate_rates_well_defined
    assert not original.independent_interval_coordinates_admitted
    assert original.support_symmetry.strict_replica_admission_status == "rejected"
    assert original.coordinates[0] != other.coordinates[0]
    assert original.coordinates[1:] == other.coordinates[1:]
    assert original.coordinates[0].form_half_difference == Q(1, 8)
    assert other.coordinates[0].form_half_difference == -Q(1, 8)
    for report, sign in ((original, 1), (other, -1)):
        # The receiver starts at an ordered tip, but unequal attachments split
        # its members. Symmetric-tip invariance must not be transferred here.
        receiver, rates = report.coordinates[4], report.rates[4]
        assert receiver.form_half_difference == 0
        assert receiver.phase_half_difference_radian_part == 0
        assert rates.phase_mean_numerator == sign * Q(1, 48)
        assert rates.phase_half_difference_numerator == sign * Q(1, 48)
        assert receiver.resultant_magnitude_bounds is None
        assert receiver.internal_form_squared is None
        assert receiver.form_phase_correlation_bounds is None
        for item in report.coordinates[1:4]:
            assert item.form_half_difference is None
            assert item.phase_half_difference_radian_part is None
            assert item.phase_half_difference_turn_part is None
    with mp.workdps(100):
        for source, report in ((graph, original), (swapped, other)):
            field = _fine_field(source, _state(source))
            for (i, j), row in zip(PAIRS, report.rates):
                _contains(row.form_mean_bounds, (field[i] + field[j]) / 2)
                assert abs(
                    _mp(row.phase_mean_numerator) / mp.pi
                    - (field[10 + i] + field[10 + j]) / 2
                ) < mp.mpf("1e-98")


def test_mixed_state_reconstruction_and_rates_use_the_complete_fine_field():
    graph = _graph(
        form=(Q(1, 4), Q(-1, 4), Q(3, 8), 0, Q(-1, 16)),
        phase=(0, Q(1, 4), Q(1, 2), Q(3, 4), 1),
        internal_form=(Q(-1, 4), Q(-1, 8), Q(-1, 16), 0, Q(1, 32)),
        internal_phase=(Q(-1, 16), Q(1, 8), 0, 0, Q(1, 32)),
        capacity=(Q(3, 2),) * 5,
    )
    graph.remove_edge(0, 8)
    graph.add_edge(2, 3)
    # Pair (2,3) is symmetric while its external neighbor set contains only
    # one member of the ordered pair (0,1). Complete-block formulas do not apply.
    graph.remove_edges_from(((1, 2), (1, 3)))
    beta = Q(3, 2)
    report = _mixed(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=0, phase_domain="regular"
        ),
    )
    assert tuple(item.reconstruction_stratum for item in report.coordinates) == (
        "ordered",
        "unordered_split_phase",
        "unordered_equal_phase",
        "unordered_tip",
        "ordered",
    )
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state, beta=beta)
        reconstructed = state.copy()
        for k, ((i, j), coordinate, rate) in enumerate(
            zip(PAIRS, report.coordinates, report.rates)
        ):
            u, delta = (state[i] - state[j]) / 2, (state[10 + i] - state[10 + j]) / 2
            du, dd = (field[i] - field[j]) / 2, (field[10 + i] - field[10 + j]) / 2
            center = _mp(coordinate.form_mean)
            angle = _mp(coordinate.phase_mean_radian_part) + (
                2 * mp.pi * _mp(coordinate.phase_mean_turn_part)
            )
            _contains(rate.form_mean_bounds, (field[i] + field[j]) / 2)
            assert abs(
                _mp(rate.phase_mean_numerator) / mp.pi
                - (field[10 + i] + field[10 + j]) / 2
            ) < mp.mpf("1e-98")
            if coordinate.mode == "ordered":
                recovered_u = _mp(coordinate.form_half_difference)
                recovered_delta = _mp(coordinate.phase_half_difference_radian_part) + (
                    2 * mp.pi * _mp(coordinate.phase_half_difference_turn_part)
                )
                _contains(rate.form_half_difference_bounds, du)
                assert abs(
                    _mp(rate.phase_half_difference_numerator) / mp.pi - dd
                ) < mp.mpf("1e-98")
            else:
                r, variance, correlation = mp.cos(delta), u * u, u * mp.sin(delta)
                _contains(coordinate.resultant_magnitude_bounds, r)
                _contains(coordinate.form_phase_correlation_bounds, correlation)
                assert abs(_mp(coordinate.internal_form_squared) - variance) < mp.mpf(
                    "1e-98"
                )
                _contains(rate.resultant_magnitude_rate_bounds, -mp.sin(delta) * dd)
                _contains(rate.internal_form_squared_rate_bounds, 2 * u * du)
                _contains(
                    rate.form_phase_correlation_rate_bounds,
                    du * mp.sin(delta) + u * mp.cos(delta) * dd,
                )
                # Reconstruct exact mathematical invariants, not independent
                # rounded interval endpoints. The proof owns the exact inverse.
                if delta:
                    recovered_delta = mp.acos(r)
                    recovered_u = correlation / mp.sqrt(1 - r * r)
                else:
                    recovered_delta, recovered_u = mp.mpf(0), mp.sqrt(variance)
            reconstructed[i], reconstructed[j] = (
                center + recovered_u,
                center - recovered_u,
            )
            reconstructed[10 + i], reconstructed[10 + j] = (
                angle + recovered_delta,
                angle - recovered_delta,
            )
            if coordinate.mode == "ordered":
                assert abs(reconstructed[i] - state[i]) < mp.mpf("1e-95")
                assert abs(reconstructed[10 + i] - state[10 + i]) < mp.mpf("1e-95")
            else:
                actual = sorted(
                    (
                        (reconstructed[i], reconstructed[10 + i]),
                        (reconstructed[j], reconstructed[10 + j]),
                    )
                )
                expected = sorted(
                    ((state[i], state[10 + i]), (state[j], state[10 + j]))
                )
                assert max(
                    abs(a - b)
                    for left, right in zip(actual, expected)
                    for a, b in zip(left, right)
                ) < mp.mpf("1e-95")
        reconstructed_field = _fine_field(graph, reconstructed, beta=beta)
        for i, j in PAIRS:
            # All retained phase means and their exact instantaneous rates
            # survive reconstruction up to the permitted independent swaps.
            assert abs(
                reconstructed_field[10 + i]
                + reconstructed_field[10 + j]
                - field[10 + i]
                - field[10 + j]
            ) < mp.mpf("1e-94")
        # Independent inherited invariant rows with an internal pair edge and
        # partially attached ordered neighbors. D includes the internal edge;
        # restoring terms contain twice its contribution.
        i, j = PAIRS[1]
        external = tuple(node for node in graph[i] if node != j)
        assert external == (0, 4, 5)
        angle, u = (state[10 + i] + state[10 + j]) / 2, (state[i] - state[j]) / 2
        delta = (state[10 + i] - state[10 + j]) / 2
        r, correlation = mp.cos(delta), u * mp.sin(delta)
        cosine_sum = sum(mp.cos(state[10 + k] - angle) for k in external)
        a = _mp(Q(3, 2)) * (cosine_sum + 2 * r) / (mp.pi * graph.degree[i])
        c = _mp(Q(3, 2)) * (len(external) + 2) / (_mp(beta) * mp.pi * graph.degree[i])
        _contains(report.rates[1].resultant_magnitude_rate_bounds, -c * correlation)
        _contains(
            report.rates[1].internal_form_squared_rate_bounds, -2 * a * correlation
        )
        _contains(
            report.rates[1].form_phase_correlation_rate_bounds,
            -a * (1 - r * r) + c * u * u * r,
        )
        for row in (report.rates[3],):
            assert row.resultant_magnitude_rate_bounds.contains(0)
            assert row.internal_form_squared_rate_bounds.contains(0)
            assert row.form_phase_correlation_rate_bounds.contains(0)


def test_mixed_state_preserves_permitted_swaps_and_tracks_ordered_pair_relabeling():
    graph = _graph(
        phase=(0, Q(1, 4), Q(1, 2), Q(3, 4), 1),
        internal_form=(Q(1, 3), Q(-1, 8), Q(1, 6), Q(-1, 9), Q(1, 7)),
        internal_phase=(Q(1, 16), Q(-1, 8), Q(1, 7), Q(1, 9), Q(-1, 11)),
    )
    graph.remove_edge(0, 8)
    source = _mixed(graph)
    for flags in product((False, True), repeat=3):
        swapped = graph.copy()
        for pair, flag in zip(PAIRS[1:4], flags):
            if flag:
                i, j = pair
                for name in ("EPI", "theta"):
                    swapped.nodes[i][name], swapped.nodes[j][name] = (
                        graph.nodes[j][name],
                        graph.nodes[i][name],
                    )
        report = _mixed(swapped)
        assert report.coordinates == source.coordinates
        assert tuple(row.phase_mean_numerator for row in report.rates) == tuple(
            row.phase_mean_numerator for row in source.rates
        )
        for left, right in zip(report.rates, source.rates):
            for name in (
                "form_mean_bounds",
                "form_half_difference_bounds",
                "resultant_magnitude_rate_bounds",
                "internal_form_squared_rate_bounds",
                "form_phase_correlation_rate_bounds",
            ):
                a, b = getattr(left, name), getattr(right, name)
                assert (a is None and b is None) or max(a.lo, b.lo) <= min(a.hi, b.hi)
    labels = {i: f"mixed_member_{71-3*i}" for i in graph}
    renamed = _mixed(
        nx.relabel_nodes(graph, labels),
        pairs=tuple(tuple(labels[node] for node in pair) for pair in PAIRS),
    )
    assert renamed.coordinates == source.coordinates
    assert renamed.rates == source.rates
    reversed_pairs = tuple((j, i) for i, j in PAIRS)
    reversed_report = _mixed(graph, pairs=reversed_pairs)
    for i, (original, reversed_row) in enumerate(
        zip(source.coordinates, reversed_report.coordinates)
    ):
        assert original.form_mean == reversed_row.form_mean
        if i in (0, 4):
            assert original.form_half_difference == -reversed_row.form_half_difference
            assert (
                original.phase_half_difference_radian_part
                == -reversed_row.phase_half_difference_radian_part
            )
        else:
            assert original == reversed_row


def test_mixed_state_chart_and_exact_strata_do_not_use_rounded_resultants():
    graph = _graph(phase=(0,) * 5, internal_phase=(0, 2, 0, 0, 0))
    graph.remove_edge(0, 8)
    with pytest.raises(ValueError, match="half-gaps"):
        _mixed(graph)
    turns = (0, 0, -1) + (0,) * 7
    lifted = _mixed(graph, phase_turns=turns)
    assert lifted.coordinates[1].phase_mean_turn_part == -Q(1, 2)
    with mp.workdps(100):
        _contains(lifted.coordinates[1].resultant_magnitude_bounds, mp.cos(2 - mp.pi))
    swapped = graph.copy()
    for name in ("EPI", "theta"):
        swapped.nodes[2][name], swapped.nodes[3][name] = (
            graph.nodes[3][name],
            graph.nodes[2][name],
        )
    swapped_turns = (0, 0, 0, -1) + (0,) * 6
    # A whole-member swap carries its declared lift; it does not infer an
    # unwrap or mistake a different midpoint chart for the same observation.
    swapped_report = _mixed(swapped, phase_turns=swapped_turns)
    assert swapped_report.coordinates == lifted.coordinates
    tiny = _graph(phase=(0,) * 5, internal_phase=(0, Q(1, 2**300), 0, 0, 0))
    tiny.remove_edge(0, 8)
    observed = _mixed(tiny)
    assert observed.coordinates[1].reconstruction_stratum == "unordered_split_phase"
    assert observed.coordinates[2].reconstruction_stratum == "unordered_tip"
    for value in ((0,) * 9, (True,) + (0,) * 9, (Q(1, 2),) + (0,) * 9):
        with pytest.raises(ValueError, match="nonboolean integer"):
            _mixed(phase_turns=value)


def test_mixed_state_capture_domains_and_export(monkeypatch, tmp_path):
    import tnfr.physics.relational_sine_scale as owner

    graph = _frozen_grouping_transition()
    graph.remove_edge(0, 8)
    capture, calls = owner.bound_relational_sine_exchange, []

    def counted(*args, **kwargs):
        calls.append(1)
        return capture(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = _mixed(graph)
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-mixed-pair-state.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    assert "comparison" not in payload["report"]
    assert "comparison" in payload["report"]["support_symmetry"]
    destination = tmp_path / "mixed-pair-state.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    with pytest.raises(ValueError, match="zero form loss"):
        _mixed(reference_model=RelationalExchangeModel(1, phase_domain="regular"))
    for value in (0, Q(3, 2)):
        invalid = graph.copy()
        invalid.nodes[0]["nu_f"] = value
        with pytest.raises(ValueError, match="common positive"):
            _mixed(invalid)
    for pairs in (PAIRS[:-1], PAIRS[:-1] + ((0, 9),)):
        with pytest.raises(ValueError):
            _mixed(pairs=pairs)

    @dataclass(frozen=True)
    class OpaqueMixedLabel:
        index: int

    with pytest.raises(TypeError):
        replace(
            report,
            support_symmetry=replace(
                report.support_symmetry, pairs=((OpaqueMixedLabel(0), 1),) + PAIRS[1:]
            ),
        ).to_dict()


MOBILITY_MARGIN_TRIPLES = ((1, 2, 0), (2, 1, 3))


def _mobility_pairing(graph=None, **changes):
    arguments = dict(
        reference_model=MODEL, epsilon=1, margin_triples=MOBILITY_MARGIN_TRIPLES
    )
    arguments.update(changes)
    return assess_sine_pairing_mobility(
        _frozen_grouping_transition() if graph is None else graph, **arguments
    )


def test_frozen_mobility_discriminator_uses_both_rows_and_is_not_a_clock_rescaling():
    graph = _frozen_grouping_transition()
    report = _mobility_pairing(graph)
    zero = _mobility_pairing(graph, epsilon=0)
    assert len(report.comparison.edges) == 20
    assert report.initial_exact_ties == (True, True)
    assert report.initial_margin_bounds == (I(0), I(0))
    assert report.source_margin_rate_difference_exact_zero
    assert report.sine_contrast_numerator_bounds == I(0)
    assert report.sine_normalized_contrast_bounds == I(0)
    assert zero.mobility_normalized_contrast_bounds == I(0)
    assert zero.mobility_margin_rate_bounds == zero.sine_margin_rate_bounds
    assert report.mobility_contrast_status == "available"
    assert Q(1, 500) < report.mobility_normalized_contrast_bounds.lo
    assert report.mobility_normalized_contrast_bounds.hi < Q(1, 400)
    assert report.certified_time_horizon is None
    with mp.workdps(100):
        state = _state(graph)
        base_field = _fine_field(graph, state)
        changed_field = base_field.copy()
        for i in graph:
            current = sum(mp.sin(state[10 + j] - state[10 + i]) for j in graph[i])
            factor = 1 + (current / graph.degree[i]) ** 2
            changed_field[i] *= factor
            changed_field[10 + i] *= factor
            _contains(report.mobility.form_rates[i], changed_field[i])
            _contains(report.mobility.phase_rates[i], changed_field[10 + i])
            assert report.mobility.node_balance_residual[i].contains(0)
        source_rates, changed_rates = [], []
        for (i, alternative, preferred), baseline_bound, changed_bound in zip(
            MOBILITY_MARGIN_TRIPLES,
            report.sine_margin_rate_bounds,
            report.mobility_margin_rate_bounds,
        ):

            def margin_rate(field):
                return 2 * mp.sin(state[10 + alternative] - state[10 + i]) * (
                    field[10 + alternative] - field[10 + i]
                ) - 2 * mp.sin(state[10 + preferred] - state[10 + i]) * (
                    field[10 + preferred] - field[10 + i]
                )

            baseline, changed = margin_rate(base_field), margin_rate(changed_field)
            source_rates.append(baseline)
            changed_rates.append(changed)
            _contains(baseline_bound, baseline)
            _contains(changed_bound, changed)
        assert abs(source_rates[0] - source_rates[1]) < mp.mpf("1e-95")
        _contains(
            report.mobility_normalized_contrast_bounds,
            (changed_rates[0] - changed_rates[1])
            / (changed_rates[0] + changed_rates[1]),
        )
        # The observable ratio removes a common clock factor. A single change
        # of time units cannot repair the unequal alternative rates.
        for rate in changed_rates:
            assert rate > 0
        assert changed_rates[0] != changed_rates[1]


def test_mobility_margin_reversal_and_common_capacity_scaling_keep_the_contrast():
    report = _mobility_pairing()
    reversed_report = _mobility_pairing(_frozen_grouping_transition(form_sign=-1))
    assert reversed_report.mobility.mobility_factors == report.mobility.mobility_factors
    assert reversed_report.mobility.form_rates == report.mobility.form_rates
    assert reversed_report.mobility.phase_rates == tuple(
        -rate for rate in report.mobility.phase_rates
    )
    assert reversed_report.mobility_margin_rate_bounds == tuple(
        -rate for rate in report.mobility_margin_rate_bounds
    )
    assert (
        reversed_report.mobility_normalized_contrast_bounds
        == report.mobility_normalized_contrast_bounds
    )
    assert reversed_report.mobility_contrast_denominator_bounds.hi < 0
    faster = _frozen_grouping_transition()
    for node in faster:
        faster.nodes[node]["nu_f"] = 2
    scaled = _mobility_pairing(faster)
    for rate, original in zip(
        scaled.mobility_margin_rate_bounds, report.mobility_margin_rate_bounds
    ):
        assert (rate - 2 * original).contains(0)
    assert (
        scaled.mobility_normalized_contrast_bounds
        - report.mobility_normalized_contrast_bounds
    ).contains(0)


def test_mobility_margin_unavailable_denominator_and_non_tie_are_not_fabricated():
    silent = _frozen_grouping_transition(amplitude=0)
    report = _mobility_pairing(silent)
    assert report.initial_exact_ties == (True, True)
    assert report.sine_margin_rate_bounds == (I(0),) * 2
    assert report.mobility_margin_rate_bounds == (I(0),) * 2
    assert report.source_margin_rate_difference_exact_zero
    assert report.sine_normalized_contrast_bounds is None
    assert report.mobility_normalized_contrast_bounds is None
    assert (
        report.sine_contrast_status
        == report.mobility_contrast_status
        == "denominator_not_separated_from_zero"
    )
    non_tie = _mobility_pairing(margin_triples=((0, 2, 1), (2, 1, 3)))
    assert non_tie.initial_exact_ties == (False, True)
    assert non_tie.initial_margin_bounds[0].lo > 0
    assert not non_tie.source_margin_rate_difference_exact_zero


def test_mobility_pairing_single_capture_labels_domains_and_export(
    monkeypatch, tmp_path
):
    import tnfr.physics.relational_sine_scale as owner

    graph = _frozen_grouping_transition()
    capture, calls = owner.bound_relational_sine_exchange, []

    def counted(*args, **kwargs):
        calls.append(1)
        return capture(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = _mobility_pairing(graph)
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    labels = {i: f"mobility_{71-3*i}" for i in graph}
    renamed = _mobility_pairing(
        nx.relabel_nodes(graph, labels),
        margin_triples=tuple(
            tuple(labels[node] for node in triple) for triple in MOBILITY_MARGIN_TRIPLES
        ),
    )
    assert renamed.mobility_margin_rate_bounds == report.mobility_margin_rate_bounds
    assert (
        renamed.mobility_normalized_contrast_bounds
        == report.mobility_normalized_contrast_bounds
    )
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-pairing-mobility.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    destination = tmp_path / "pairing-mobility.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    for triples in (
        (),
        ((1, 2, 0),),
        ((1, 2, 0),) * 3,
        ((1, 2, 0), (2, 1, 1)),
        ((1, 2, 0), (2, 1, 10)),
        ((1, 2, 0), (2, 1)),
    ):
        with pytest.raises((TypeError, ValueError)):
            _mobility_pairing(margin_triples=triples)

    @dataclass(frozen=True)
    class OpaqueMobilityLabel:
        index: int

    for corrupted in (
        replace(report, margin_triples=((OpaqueMobilityLabel(0), 2, 0), (2, 1, 3))),
        replace(
            report,
            observation=replace(
                report.observation,
                nodes=(OpaqueMobilityLabel(0),) + report.observation.nodes[1:],
            ),
        ),
        replace(
            report,
            observation=replace(
                report.observation, candidate_pairs=((OpaqueMobilityLabel(0), 1),)
            ),
        ),
    ):
        with pytest.raises(TypeError):
            corrupted.to_dict()


def _mobility_geometry(graph=None, **changes):
    arguments = dict(
        reference_model=MODEL,
        pairs=PAIRS,
        radius=Q(1, 16),
        excess_ceiling=Q(1, 32768),
        epsilon=1,
    )
    arguments.update(changes)
    return assess_sine_mobility_geometry(
        _persistent_graph() if graph is None else graph, **arguments
    )


def test_changed_mobility_reuses_relative_barrier_without_mean_slab_or_recurrence_transfer():
    graph = _persistent_graph(
        internal_phase=(0, Q(1, 2048), Q(-1, 4096), Q(1, 4096), Q(-1, 2048))
    )
    existing = _persistence(graph)
    alternative = _mobility_geometry(graph)
    original = _mobility_geometry(graph, epsilon=0)
    assert alternative.family_admitted and alternative.source_set_trapping_certified
    assert alternative.source_relative_family_membership == "certified_inside"
    assert alternative.pair_source_status == ("exact_nontip",) * 5
    assert alternative.synchronized_tip_sets_invariant
    assert (
        alternative.relative_family_recurrence_status
        == "unavailable_invariant_measure_unproved"
    )
    assert (
        alternative.invariant_measure_status
        == "equivalent_finite_invariant_measure_not_supplied"
    )
    assert (
        original.relative_family_recurrence_status
        == "certified_almost_everywhere_in_relative_family"
    )
    assert original.invariant_measure_status == "relative_Euclidean_volume"
    assert alternative.individual_recurrence_status == "unavailable_for_chosen_state"
    assert alternative.full_state_recurrence_status == "not_assessed_removed_origins"
    assert (
        alternative.internal_circulation_status == "not_assessed_for_changed_mobility"
    )
    assert alternative.radius == Q(1, 16)
    assert alternative.excess_ceiling == Q(1, 32768)
    assert alternative.target_phase_turns == tuple(Q(i // 2, 5) for i in range(10))
    for name in (
        "spectral_gap_lower_bound",
        "spectral_gap_method",
        "target_storage_bounds",
        "acute_radius_margin_bounds",
        "cosine_lower_bound",
        "coercivity_lower_bound",
        "barrier_lower_bound",
        "form_norm_squared_upper_bound",
        "phase_norm_squared_upper_bound",
        "norm_squared_upper_bound",
        "excess_storage_upper_bound",
        "norm_margin",
        "energy_margin",
        "family_barrier_margin",
        "source_family_excess_margin",
    ):
        assert (
            getattr(alternative, name)
            == getattr(original, name)
            == getattr(existing, name)
        )
    reflected = graph.copy()
    for node in reflected:
        reflected.nodes[node]["theta"] *= -1
    negative = _mobility_geometry(reflected, winding=-1)
    assert negative.family_admitted and negative.source_set_trapping_certified
    assert negative.target_phase_turns == tuple(
        -value for value in alternative.target_phase_turns
    )
    assert negative.norm_squared_upper_bound == alternative.norm_squared_upper_bound
    assert negative.excess_storage_upper_bound == alternative.excess_storage_upper_bound
    assert negative.mobility.form_rates == tuple(
        -rate for rate in alternative.mobility.form_rates
    )
    assert negative.mobility.phase_rates == alternative.mobility.phase_rates
    common_turn = _mobility_geometry(graph, phase_turns=(1,) * 10)
    assert common_turn.phase_turns == (1,) * 10
    assert common_turn.norm_squared_upper_bound == alternative.norm_squared_upper_bound
    assert (
        common_turn.excess_storage_upper_bound == alternative.excess_storage_upper_bound
    )
    with mp.workdps(100):
        state = _state(graph)
        field = _fine_field(graph, state)
        for i in graph:
            s = sum(mp.sin(state[10 + j] - state[10 + i]) for j in graph[i])
            factor = 1 + (s / graph.degree[i]) ** 2
            field[i] *= factor
            field[10 + i] *= factor
            _contains(alternative.mobility.form_rates[i], field[i])
            _contains(alternative.mobility.phase_rates[i], field[10 + i])
        for k, i in enumerate(alternative.balance.relative_indices):
            _contains(alternative.balance.relative_form_rates[k], field[i] - field[0])
            _contains(
                alternative.balance.relative_phase_rates[k], field[10 + i] - field[10]
            )
        _contains(alternative.relative_coordinate_deviation_bound, mp.sqrt(2) / 16)


def test_relative_geometry_preserves_trapping_when_absolute_origin_or_family_membership_changes():
    original = _mobility_geometry()
    shifted = _persistent_graph(form=(16,) * 5)
    changed = _mobility_geometry(shifted)
    assert changed.source_set_trapping_certified
    assert changed.source_relative_family_membership == "certified_inside"
    assert (
        changed.balance.weighted_form_mean == original.balance.weighted_form_mean + 16
    )
    assert changed.balance.relative_form == original.balance.relative_form
    assert changed.excess_storage_upper_bound == original.excess_storage_upper_bound
    old_slab = _persistence(shifted)
    assert old_slab.source_family_membership == "outside"
    assert (
        "captured_mean_outside_open_slab" in old_slab.source_family_membership_reasons
    )
    tips = _mobility_geometry(_persistent_graph(internal_form=(0,) * 5))
    assert tips.family_admitted and tips.source_set_trapping_certified
    assert tips.source_relative_family_membership == "outside"
    assert tips.pair_source_status == ("synchronized_tip",) * 5
    assert tips.source_family_membership_reasons == (
        "captured_source_has_synchronized_pair_tip",
    )
    small_family = _mobility_geometry(excess_ceiling=Q(1, 2**40))
    assert small_family.family_admitted and small_family.source_set_trapping_certified
    assert small_family.source_relative_family_membership == "unresolved"
    invalid_family = _mobility_geometry(excess_ceiling=1)
    assert not invalid_family.family_admitted
    assert invalid_family.source_set_trapping_certified
    assert (
        invalid_family.relative_family_recurrence_status
        == "unavailable_family_not_admitted"
    )
    no_acute_barrier = _mobility_geometry(radius=1)
    assert not no_acute_barrier.family_admitted
    assert not no_acute_barrier.source_set_trapping_certified
    assert no_acute_barrier.barrier_lower_bound is None


def test_mobility_geometry_single_capture_admission_chart_labels_and_export(
    monkeypatch, tmp_path
):
    import tnfr.physics.relational_sine_scale as owner

    graph = _persistent_graph()
    capture, calls = owner.bound_relational_sine_exchange, []

    def counted(*args, **kwargs):
        calls.append(1)
        return capture(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    before = pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    report = _mobility_geometry(graph)
    assert calls == [1]
    assert before == pickle.dumps(
        (dict(graph.graph), dict(graph.nodes), list(graph.edges(data=True)))
    )
    labels = {i: f"relative_geometry_{71-3*i}" for i in graph}
    relabeled = _mobility_geometry(
        nx.relabel_nodes(graph, labels),
        pairs=tuple(tuple(labels[node] for node in pair) for pair in PAIRS),
    )
    assert relabeled.balance.reference_node == labels[0]
    assert relabeled.balance.relative_form == report.balance.relative_form
    assert relabeled.balance.relative_phase_rates == report.balance.relative_phase_rates
    assert relabeled.energy_margin == report.energy_margin
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-mobility-geometry.v1"
    assert relational_report_to_dict(report)["report"] == payload["report"]
    assert "form_mean_bounds" not in payload["report"]
    assert "family_almost_everywhere_recurrence_certified" not in payload["report"]
    destination = tmp_path / "mobility-geometry.json"
    export_to_json(payload, destination)
    assert json.loads(destination.read_text()) == payload
    unequal = graph.copy()
    for node in PAIRS[0]:
        unequal.nodes[node]["nu_f"] = 2
    with pytest.raises(ValueError, match="common positive"):
        _mobility_geometry(unequal)
    missing = graph.copy()
    missing.remove_edge(0, 8)
    with pytest.raises(ValueError, match="four fine cross-edges"):
        _mobility_geometry(missing)
    for changes in (
        dict(radius=0),
        dict(excess_ceiling=0),
        dict(winding=True),
        dict(winding=0),
        dict(phase_turns=(0,) * 9),
        dict(phase_turns=(True,) + (0,) * 9),
        dict(epsilon=-1),
        dict(reference_model=RelationalExchangeModel(1, phase_domain="regular")),
    ):
        with pytest.raises((TypeError, ValueError)):
            _mobility_geometry(**changes)
    with pytest.raises(ValueError, match="half-gaps"):
        _mobility_geometry(_persistent_graph(internal_phase=(2, 0, 0, 0, 0)))

    @dataclass(frozen=True)
    class OpaqueGeometryLabel:
        index: int

    for corrupted in (
        replace(report, pairs=((OpaqueGeometryLabel(0), 1),) + PAIRS[1:]),
        replace(
            report,
            balance=replace(report.balance, reference_node=OpaqueGeometryLabel(0)),
        ),
    ):
        with pytest.raises(TypeError):
            corrupted.to_dict()
