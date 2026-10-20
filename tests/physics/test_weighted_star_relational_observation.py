"""Static relational observations of the retained complete weighted sine law.

No event or trajectory is replayed. Neighboring states are supplied controls,
not claims about what the original word prepares or a later trajectory reaches.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.physics.relational_sine_star_retention import (
    assess_sine_weighted_star_support_retention,
)
from tnfr.physics.relational_sine_support import (
    _rebuild_sine_support_comparison,
    _support_comparison_from_rows,
    bound_relational_sine_support_exchange,
)
from tnfr.research import weighted_star_contact, weighted_star_finite_episode
from tnfr.research.artifact_io import decode_exact_tree, read_bytes_bounded
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.research.weighted_star_constitutive_handoff import (
    compare_weighted_star_constitutive_handoff,
)
from tnfr.utils.io import json_loads

REGIONS = (tuple(range(5)), tuple(range(5, 10)))


@pytest.fixture(scope="module", autouse=True)
def no_event_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("relational observation must not replay events or trajectories")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                weighted_star_contact,
                (
                    "contact_source",
                    "execute_contact_word",
                    "execute_network_operator_stage",
                ),
            ),
            (
                weighted_star_finite_episode,
                (
                    "prepare_source",
                    "evaluate_protocol",
                    "execute_contact_word",
                    "rk4_coupled_step",
                ),
            ),
            (integrators, ("rk4_coupled_step",)),
            (relational, ("step_relational_exchange", "_advance")),
            (word_execution, ("execute_network_operator_stage",)),
            (network_stage, ("execute_pointwise_stage", "execute_coupling_stage")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/weighted_star_registration_episode/preparation.json"
    )
    preparation = decode_exact_tree(
        json_loads(read_bytes_bounded(path, max_bytes=256 * 1024))
    )
    admitted = compare_weighted_star_constitutive_handoff(preparation)
    prefix = admitted.prefixes[-1].source
    # The prefix type admits disconnected earlier states; this gate uses its
    # connected successor and re-admits the complete weighted support law.
    source = _support_comparison_from_rows(
        **{
            name: getattr(prefix, name)
            for name in (
                "reference_model",
                "nodes",
                "edges",
                "conductances",
                "epi",
                "phase",
                "capacity",
            )
        }
    )
    graph = restore_recorded_graph_state(preparation["static_states"][-1]).graph
    return source, graph


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _observe(source, *, regions=REGIONS, receiver=(5, 6)):
    """Test-only linear observation of freshly rebuilt complete rate rows."""
    source = _rebuild_sine_support_comparison(source)
    positions = {node: i for i, node in enumerate(source.nodes)}
    groups = tuple(tuple(positions[node] for node in row) for row in regions)
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))

    def contrast(values, zero):
        means = tuple(
            sum((weights[i] * values[i] for i in group), zero)
            / sum((weights[i] for i in group), Q(0))
            for group in groups
        )
        return means[1] - means[0]

    i, j = (positions[node] for node in receiver)
    return (
        source.epi[i] - source.epi[j],
        contrast(source.epi, Q(0)),
        source.form_rates[i] - source.form_rates[j],
        contrast(source.form_rates, I(0)),
    )


def _edge_form_direction(source, direction):
    """Differentiate oriented bare-edge sine currents, independently of rows."""
    positions = {node: i for i, node in enumerate(source.nodes)}
    degree, currents = [0] * len(source.nodes), [I(0) for _ in source.nodes]
    for left, right in source.edges:
        i, j = positions[left], positions[right]
        degree[i] += 1
        degree[j] += 1
        derivative = cos(I(source.phase[j] - source.phase[i])) * (
            direction[j] - direction[i]
        )
        currents[i] += derivative
        currents[j] -= derivative
    assert tuple(degree) == source.degrees
    return tuple(
        current * nu / (d * pi_interval())
        for current, nu, d in zip(currents, source.capacity, degree)
    )


def test_central_exact_region_weights_cut_current_and_slope(retained):
    source = retained[0]
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    left, right = sum(weights[:5]), sum(weights[5:])
    assert (left, right) == (Q(84, 13), Q(51, 8))
    k = 1 / left + 1 / right
    assert k == Q(445, 1428)
    r, d, rd, dd = _observe(source)
    assert (r, d) == (Q(0), Q(113, 6528))
    alpha = source.phase[5]
    assert alpha == Q(201, 512)
    # All five oriented cut currents equal -sin(alpha)/pi into the right.
    current = sin(I(alpha)) / pi_interval()
    assert (rd + Q(2, 15) * current).contains(0)
    assert (dd + 5 * k * current).contains(0)
    assert dd.hi < 0
    slope = rd / dd
    assert slope.contains(Q(952, 11125))
    assert slope.width < Q(1, 10**16)


def test_hidden_phase_direction_has_positive_slope_derivative(retained, mp):
    source = retained[0]
    _, _, rd, dd = _observe(source)
    direction = (Q(0),) * 5 + (Q(1), Q(0), Q(0), Q(0), Q(-2))
    rates = _edge_form_direction(source, direction)
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    rd_prime = rates[5] - rates[6]
    dd_prime = sum((weights[i] * rates[i] for i in range(5, 10)), I(0)) / sum(
        weights[5:]
    ) - sum((weights[i] * rates[i] for i in range(5)), I(0)) / sum(weights[:5])
    alpha = _mp(mp, source.phase[5])
    k = Q(445, 1428)
    _contains(mp, rd_prime, (-mp.mpf(62) / 15 - mp.mpf(4) / 5 * mp.cos(alpha)) / mp.pi)
    _contains(mp, dd_prime, _mp(mp, k) * mp.cos(alpha) / mp.pi)
    derivative = (rd_prime * dd - rd * dd_prime) / dd**2
    expected = 62 * (5 + mp.cos(alpha)) / (375 * _mp(mp, k) * mp.sin(alpha))
    _contains(mp, derivative, expected)
    assert derivative.lo > 0


def test_same_observations_and_weighted_phase_mean_have_distinct_slopes(retained, mp):
    source = retained[0]
    epsilon = Q(1, 1024)
    phases = list(source.phase)
    phases[5] += epsilon
    phases[9] -= 2 * epsilon
    changed = _rebuild_sine_support_comparison(replace(source, phase=tuple(phases)))
    before, after = _observe(source), _observe(changed)
    assert before[:2] == after[:2]
    assert changed.epi == source.epi
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    assert (
        sum((w * (a - b) for w, a, b in zip(weights, changed.phase, source.phase))) == 0
    )
    assert len(set(changed.phase[i] for i in (0, 1, 2, 3))) == 1
    assert len(set(changed.phase[i] for i in (6, 7, 8))) == 1
    for state in (source, changed):
        retention = assess_sine_weighted_star_support_retention(state, stars=REGIONS)
        assert retention.source_identity_admitted
        assert retention.all_time_identity_retention_certified
        assert state.storage.hi < Q(1, 2)
    assert before[3].hi < 0 and after[3].hi < 0
    assert (after[2] / after[3]).lo > (before[2] / before[3]).hi
    # Independent oriented-edge formulas at the finite supplied perturbation.
    a, e = _mp(mp, source.phase[5]), _mp(mp, epsilon)
    numerator = (
        -mp.mpf(46) / 15 * mp.sin(e)
        - mp.mpf(4) / 5 * mp.sin(3 * e)
        + mp.mpf(2) / 3 * mp.sin(2 * e)
        - mp.mpf(4) / 5 * mp.sin(a + e)
        + mp.mpf(2) / 3 * mp.sin(a)
    )
    denominator = -_mp(mp, Q(445, 1428)) * (
        mp.sin(a + e) + 3 * mp.sin(a) + mp.sin(a - 2 * e)
    )
    _contains(mp, after[2], numerator / mp.pi)
    _contains(mp, after[3], denominator / mp.pi)
    _contains(mp, after[2] / after[3], numerator / denominator)


def test_zero_denominator_with_live_phase_motion_cannot_supply_a_slope(retained):
    source = _rebuild_sine_support_comparison(replace(retained[0], phase=(Q(0),) * 10))
    _, _, rd, dd = _observe(source)
    assert rd.contains(0) and dd.contains(0)
    assert any(not rate.contains(0) for rate in source.phase_rates)
    with pytest.raises(ZeroDivisionError):
        rd / dd


@pytest.mark.parametrize("clock_rate", (Q(3, 2), Q(1, 4)))
def test_common_constant_clock_change_transports_both_rows(retained, clock_rate):
    source = retained[0]
    # In this held law capacity appears only as a common nodal prefactor.
    # nu_tau=nu/alpha therefore represents tau=alpha*t in both rows. This
    # does not justify the substitution for capacity-dependent pressure laws.
    transformed = _rebuild_sine_support_comparison(
        replace(source, capacity=tuple(nu / clock_rate for nu in source.capacity))
    )
    for old, new in zip(
        source.form_rates + source.phase_rates,
        transformed.form_rates + transformed.phase_rates,
    ):
        assert (clock_rate * new - old).contains(0)
    before, after = _observe(source), _observe(transformed)
    assert before[:2] == after[:2]
    assert after[3].hi < 0
    assert (after[2] / after[3]).contains(Q(952, 11125))
    assert transformed.storage == source.storage


def test_unused_lengths_and_joint_relabeling_preserve_the_observation(retained):
    source, recorded_graph = retained
    graph = deepcopy(recorded_graph)
    for index, edge in enumerate(graph.edges):
        graph.edges[edge]["length"] = index + 2
    captured = bound_relational_sine_support_exchange(
        graph, reference_model=source.reference_model
    )
    assert captured == source
    names = {node: f"port-{(7 * node + 3) % 10}" for node in graph}
    relabeled = nx.relabel_nodes(graph, names, copy=True)
    # Change insertion order too; observation ports and regions follow labels.
    reordered = nx.Graph()
    reordered.graph.update(deepcopy(relabeled.graph))
    reordered.add_nodes_from(reversed(tuple(relabeled.nodes(data=True))))
    reordered.add_edges_from(relabeled.edges(data=True))
    result = bound_relational_sine_support_exchange(
        reordered, reference_model=source.reference_model
    )
    observed = _observe(
        result,
        regions=tuple(tuple(names[node] for node in row) for row in REGIONS),
        receiver=(names[5], names[6]),
    )
    expected = _observe(source)
    assert observed[:2] == expected[:2]
    for actual, reference in zip(observed[2:], expected[2:]):
        assert (actual - reference).contains(0)
        assert actual.width < Q(1, 10**16)
    assert (observed[2] / observed[3]).contains(Q(952, 11125))


@pytest.fixture(scope="module")
def matched_form_probes(retained):
    """Supplied post-action states, without authenticating an AL/SHA word."""
    source = retained[0]
    result = {}
    for donor in (5, 0):
        branches = []
        for boost in (Q(1, 1024), Q(1, 512)):
            form, capacity = list(source.epi), list(source.capacity)
            form[donor] += boost
            capacity[donor] /= 2
            branches.append(
                _rebuild_sine_support_comparison(
                    replace(source, epi=tuple(form), capacity=tuple(capacity))
                )
            )
        result[donor] = tuple(branches)
    return result


def _phase_acceleration_from_edges(source):
    """Independent static K*x_dot on the actual complete weighted support."""
    positions = {node: i for i, node in enumerate(source.nodes)}
    gradient = [I(0) for _ in source.nodes]
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = positions[left], positions[right]
        difference = weight * (source.form_rates[i] - source.form_rates[j])
        gradient[i] += difference
        gradient[j] -= difference
    return tuple(
        nu * value / (degree * pi_interval())
        for nu, value, degree in zip(source.capacity, gradient, source.degrees)
    )


def test_matched_positive_probes_rebuild_full_state_and_receiver_ratio(
    retained, matched_form_probes
):
    source = retained[0]
    weights = {
        frozenset(edge): weight
        for edge, weight in zip(source.edges, source.conductances)
    }
    differences = {}
    for donor, (low, high) in matched_form_probes.items():
        for branch, boost in zip((low, high), (Q(1, 1024), Q(1, 512))):
            assert tuple(a - b for a, b in zip(branch.epi, source.epi)) == tuple(
                boost if node == donor else Q(0) for node in source.nodes
            )
            assert branch.capacity == tuple(
                value / 2 if node == donor else value
                for node, value in zip(source.nodes, source.capacity)
            )
            assert branch.capacity[4] == source.capacity[4]
            assert branch.phase == source.phase
            assert branch.edges == source.edges
            assert branch.conductances == source.conductances
            retention = assess_sine_weighted_star_support_retention(
                branch, stars=REGIONS
            )
            assert retention.all_time_identity_retention_certified
        assert low.capacity == high.capacity
        assert low.form_rates == high.form_rates
        assert _phase_acceleration_from_edges(low) == _phase_acceleration_from_edges(
            high
        )
        low_rates = low.phase_rate_numerators()
        high_rates = high.phase_rate_numerators()
        difference = high_rates[4] - low_rates[4]
        expected = -Q(13, 32) * weights[frozenset((4, donor))] / 1024
        assert difference == expected < 0
        assert low_rates != high_rates
        differences[donor] = difference
    assert weights[frozenset((4, 0))] == 2
    assert differences[5] / differences[0] == weights[frozenset((4, 5))] / 2
    # Donor zero belonged to the four-member block. Its form and capacity
    # reset break that family's constraints; all ten fine nodes were retained.
    for branch in matched_form_probes[0]:
        assert branch.epi[0] != branch.epi[1]
        assert (
            branch.capacity[0] / branch.degrees[0]
            != branch.capacity[1] / branch.degrees[1]
        )


def test_unequal_pair_capacities_break_acceleration_cancellation(
    retained, matched_form_probes, mp
):
    low, high = matched_form_probes[5]
    capacities = list(high.capacity)
    capacities[5] /= 2
    mismatch = _rebuild_sine_support_comparison(
        replace(high, capacity=tuple(capacities))
    )
    # Receiver capacity and form changes still give the same ideal first-rate
    # difference. That alone does not establish the matched finite-phase lemma.
    assert mismatch.phase_rate_numerators()[4] == high.phase_rate_numerators()[4]
    assert mismatch.form_rates != low.form_rates
    acceleration_difference = (
        _phase_acceleration_from_edges(mismatch)[4]
        - _phase_acceleration_from_edges(low)[4]
    )
    assert acceleration_difference.hi < 0
    source = retained[0]
    weight = dict(zip(source.edges, source.conductances))[(4, 5)]
    expected = (
        -_mp(mp, Q(13, 160) * weight) * mp.sin(_mp(mp, source.phase[5])) / mp.pi**2
    )
    _contains(mp, acceleration_difference, expected)
    assert assess_sine_weighted_star_support_retention(
        mismatch, stars=REGIONS
    ).all_time_identity_retention_certified
