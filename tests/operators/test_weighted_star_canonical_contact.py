"""Prospectively declared static AL/UM/SHA controls on two weighted stars.

SINE_WEIGHTED_STAR_CANONICAL_CONTACT section1 fixed the three dyadic sources,
action, arithmetic allowance and rejection controls before these evaluations.
These are current-engine domain controls, not an acquired-source campaign.
"""

import platform
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction as Q
from hashlib import sha256
from math import pi
from pathlib import Path

import mpmath
import networkx as nx
import pytest

from tests.physics.test_sine_consensus_observation import _apply as _tangent_apply
from tests.physics.test_sine_consensus_observation import _assert_exact_realization
from tests.physics.test_sine_consensus_observation import _identity as _tangent_identity
from tests.physics.test_sine_consensus_observation import _independent_tangent
from tests.physics.test_sine_consensus_observation import _product as _tangent_product
from tnfr.alias import set_dnfr
from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.mathematics import unified_numerical
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.metrics import phase_compatibility
from tnfr.operators import _coupling_stage_kernel, _phase_gate, network_stage
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_consensus_observation import (
    assess_sine_consensus_observation,
)
from tnfr.physics.relational_sine_star_retention import (
    assess_sine_weighted_star_support_retention,
)
from tnfr.physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from tnfr.physics.relational_sine_weighted_partition import (
    assess_sine_weighted_phase_offset_partition,
)
from tnfr.research.weighted_star_contact import contact_source as _source
from tnfr.research.weighted_star_contact import contact_word as _word
from tnfr.research.weighted_star_contact import (
    execute_contact_word as _execute_static_word,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.sdk._state import copy_graph_state
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils.io import json_loads

STARS = (tuple(range(5)), tuple(range(5, 10)))
COLLECTIVE_BLOCKS = ((0, 1, 2, 3), (4,), (5,), (6, 7, 8), (9,))
STAR_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
OLD_EDGES = STAR_EDGES + tuple((i + 5, j + 5) for i, j in STAR_EDGES)
OLD_WEIGHTS = (3, 3, 3, 2, 2, 2, 2) * 2
NEW_EDGES = tuple((4, other) for other in range(5, 10))
CAPACITIES = (4, 2, 2, 2, 8) * 2
POST_DEGREES = (4, 2, 2, 2, 9, 5, 3, 3, 3, 5)
PROBES = ((Q(1, 4), Q(201, 512)), (Q(15, 64), Q(3, 8)), (Q(17, 64), Q(409, 1024)))
BOOST = Q(1, 1024)
AMPLITUDE = Q(1, 16)
SCORE_TOLERANCE = Q(1, 2**46)
NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
CORE = {"compatibility_profile": "core"}


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


@pytest.fixture(scope="module", autouse=True)
def no_continuous_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("the static canonical-contact protocol permits no trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _form(graph, node):
    return Q(require_finite_real_scalar_epi(graph.nodes[node]["EPI"], "contact form"))


def _component(graph, nodes):
    """Capture a connected scientific component without runtime cache objects."""
    result = nx.Graph()
    result.graph["GAMMA"] = deepcopy(graph.graph["GAMMA"])
    result.add_nodes_from((node, deepcopy(graph.nodes[node])) for node in nodes)
    result.add_edges_from(
        (i, j, deepcopy(data))
        for i, j, data in graph.edges(data=True)
        if i in nodes and j in nodes
    )
    return result


def _evaluate_contact(probe):
    mean, phase = probe
    states, stages, graph = _execute_static_word(_source(mean, phase))
    final = bound_relational_sine_support_exchange(states[-1], reference_model=MODEL)
    retained = assess_sine_weighted_star_support_retention(final, stars=STARS)
    return mean, phase, states, stages, graph, final, retained


@pytest.fixture(scope="module")
def contacts(record_testsuite_property):
    backend = unified_numerical.np if unified_numerical.NUMPY_AVAILABLE else None
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "phase_backend": "numpy" if backend is not None else "math",
        "phase_backend_version": (
            backend.__version__ if backend is not None else "stdlib"
        ),
        "independent_mpmath_version": mpmath.__version__,
        "static_control_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for key, value in provenance.items():
        record_testsuite_property("canonical_contact_" + key, value)
    return tuple(_evaluate_contact(probe) for probe in PROBES)


@pytest.fixture(scope="module", params=range(3), ids=("central", "lower", "upper"))
def contact(request, contacts):
    return contacts[request.param]


def _edge_storage(mp, graph, edges):
    return sum(
        _mp(
            mp,
            Q(graph.edges[i, j]["weight"])
            * (_form(graph, i) - _form(graph, j)) ** 2
            / 2,
        )
        + 1
        - mp.cos(_mp(mp, Q(graph.nodes[j]["theta"]) - Q(graph.nodes[i]["theta"])))
        for i, j in edges
    )


def _independent_field(mp, graph):
    currents = [mp.mpf(0) for _ in graph]
    gradient = [Q(0) for _ in graph]
    for i, j in OLD_EDGES + NEW_EDGES:
        difference = _form(graph, i) - _form(graph, j)
        weight = Q(graph.edges[i, j]["weight"])
        phase = _mp(mp, Q(graph.nodes[j]["theta"]) - Q(graph.nodes[i]["theta"]))
        currents[i] += mp.sin(phase)
        currents[j] -= mp.sin(phase)
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
    pressure = tuple(
        value / (degree * mp.pi) for value, degree in zip(currents, POST_DEGREES)
    )
    capacities = tuple(Q(graph.nodes[node]["nu_f"]) for node in graph)
    form_rates = tuple(
        _mp(mp, capacity) * value for capacity, value in zip(capacities, pressure)
    )
    phase_numerators = tuple(
        capacity * value / degree
        for capacity, value, degree in zip(capacities, gradient, POST_DEGREES)
    )
    phase_rates = tuple(_mp(mp, value) / mp.pi for value in phase_numerators)
    return pressure, form_rates, phase_numerators, phase_rates


def _collective_partition(source, blocks=COLLECTIVE_BLOCKS):
    return assess_sine_weighted_phase_offset_partition(
        source, blocks=blocks, phase_offset_turns=(0,) * 10
    )


def _assert_fine_and_collective_state(mp, state, graph):
    pressure, form_rates, phase_numerators, phase_rates = _independent_field(mp, graph)
    assert state.full_row_equality_certified
    assert state.full_source.nodes == tuple(graph.nodes)
    assert state.full_source.epi == tuple(_form(graph, node) for node in graph)
    assert state.full_source.phase == tuple(
        Q(graph.nodes[node]["theta"]) for node in graph
    )
    assert state.full_source.capacity == tuple(
        Q(graph.nodes[node]["nu_f"]) for node in graph
    )
    assert state.full_phase_rate_numerators == phase_numerators
    for i, block in enumerate(state.partition.node_blocks):
        _contains(mp, state.full_source.pressure[i], pressure[i])
        _contains(mp, state.full_source.form_rates[i], form_rates[i])
        _contains(mp, state.full_source.phase_rates[i], phase_rates[i])
        _contains(mp, state.block_form_rates[block], form_rates[i])
        _contains(mp, state.block_phase_rates[block], phase_rates[i])
        assert state.block_phase_rate_numerators[block] == phase_numerators[i]
    form_storage = sum(
        (
            Q(graph.edges[i, j]["weight"])
            * (_form(graph, i) - _form(graph, j)) ** 2
            / 2
            for i, j in OLD_EDGES + NEW_EDGES
        ),
        Q(0),
    )
    phase_storage = sum(
        1 - mp.cos(_mp(mp, Q(graph.nodes[j]["theta"]) - Q(graph.nodes[i]["theta"])))
        for i, j in OLD_EDGES + NEW_EDGES
    )
    assert state.full_source.form_storage == state.quotient_form_storage == form_storage
    _contains(mp, state.full_source.storage, _mp(mp, form_storage) + phase_storage)
    _contains(mp, state.quotient_phase_storage_bounds, phase_storage)
    assert state.storage_residual_bounds.contains(0)
    assert not any(state.phase_rate_residuals)
    assert all(value.contains(0) for value in state.form_rate_residual_bounds)
    weights = tuple(
        Q(graph.degree[node]) / Q(graph.nodes[node]["nu_f"]) for node in graph
    )
    total = sum(weights)
    assert state.mean_weights == weights
    assert (
        state.form_mean
        == sum(w * _form(graph, node) for node, w in enumerate(weights)) / total
    )
    assert (
        state.phase_mean_radians
        == sum(w * Q(graph.nodes[node]["theta"]) for node, w in enumerate(weights))
        / total
    )
    assert state.form_mean_rate_bounds.contains(0)
    assert state.phase_mean_rate_numerator == 0


def test_actual_endpoint_has_exact_five_block_state_and_inherits_finite_response(
    contact, mp
):
    mean, alpha, states, _, _, source, retention = contact
    partition = _collective_partition(source)
    state = partition.observe_radian_state()
    assert partition.invariance_certified
    assert state.source_membership_certified
    assert state.collective_form == (
        mean + AMPLITUDE,
        mean + BOOST,
        mean + AMPLITUDE,
        mean + AMPLITUDE,
        mean,
    )
    assert state.collective_phase_radians == (0, 0, alpha, alpha, alpha)
    assert partition.block_mean_weights == (4, Q(32, 13), Q(5, 4), Q(9, 2), Q(5, 8))
    assert sum(partition.block_mean_weights) == Q(1335, 104)
    assert state.phase_mean_radians == Q(221, 445) * alpha
    _assert_fine_and_collective_state(mp, state, states[-1])
    # The observer retains the complete admitted fine law. Inheriting its
    # previous analytical bound performs no numerical evolution or new word.
    inherited = state.full_source.finite_form_response_bounds(duration=Q(1, 128))
    original = source.finite_form_response_bounds(duration=Q(1, 128))
    assert inherited == original
    assert (
        inherited.endpoint_form_bounds[5] - inherited.endpoint_form_bounds[6]
    ).hi < -Q(1, 32768)
    assert state.collective_form[2] - state.collective_form[3] == 0
    assert state.block_form_rates[2].hi < state.block_form_rates[3].lo
    assert retention.all_time_identity_retention_certified
    assert state.full_source.storage == retention.full_storage_bounds


def test_five_block_coefficients_preserve_multiplicity_and_exclude_four_blocks(contact):
    source = contact[5]
    partition = _collective_partition(source)
    state = partition.observe_radian_state()
    weights = dict(zip(source.edges, source.conductances))
    w_a, w_b = weights[4, 5], weights[4, 9]
    counts = (
        (0, 4, 0, 0, 0),
        (4, 0, 1, 3, 1),
        (0, 1, 0, 3, 1),
        (0, 3, 3, 0, 3),
        (0, 1, 1, 3, 0),
    )
    conductances = (
        (0, 8, 0, 0, 0),
        (8, 0, w_a, 3 * w_a, w_b),
        (0, w_a, 0, 9, 2),
        (0, 3 * w_a, 9, 0, 6),
        (0, w_b, 2, 6, 0),
    )
    assert partition.cross_bare_counts == counts
    assert partition.cross_conductance_sums == conductances
    for a in range(5):
        for b in range(5):
            mu = partition.block_mean_weights[a]
            assert mu * state.quotient_bare_rows[a][b] == counts[a][b]
            assert mu * partition.quotient_weighted_rows[a][b] == conductances[a][b]
            assert counts[a][b] == counts[b][a]
            assert conductances[a][b] == conductances[b][a]
    assert state.quotient_bare_rows[1][0] == Q(13, 8)
    assert state.quotient_bare_rows[3][2] == Q(2, 3)
    old = _collective_partition(source, ((0, 1, 2, 3), (4,), (5, 6, 7, 8), (9,)))
    assert old.status == "excluded"
    assert not old.invariance_certified
    assert old.internal_balance_certified
    assert not old.weighted_compatibility_certified
    assert not old.moment_compatibility_certified
    assert old.cross_weighted_residuals[6][1] == -Q(2, 15) * w_a
    assert old.row_cosine_expressions[5][1] == ((Q(1, 4), Q(4, 5)),)
    assert old.row_cosine_expressions[6][1] == ((Q(1, 4), Q(2, 3)),)
    difference = old.row_cosine_bounds[5][1] - old.row_cosine_bounds[6][1]
    assert difference.contains(Q(2, 15)) and difference.lo > 0
    with pytest.raises(ValueError, match="certified"):
        old.observe_radian_state()


def test_detached_five_block_state_keeps_right_center_and_leaf_coordinates(
    contacts, mp
):
    source = contacts[0][5]
    partition = _collective_partition(source)
    forms = (Q(3, 8), Q(1, 4), Q(5, 16), Q(9, 32), Q(7, 32))
    phases = (Q(1, 16), Q(-1, 32), Q(3, 8), Q(11, 32), Q(5, 16))
    state = partition.evaluate_radians(forms, phases)
    assert not state.source_membership_certified
    assert state.collective_form[2] != state.collective_form[3]
    assert state.collective_phase_radians[2] != state.collective_phase_radians[3]
    graph = copy_graph_state(contacts[0][2][-1])
    for block, form, phase in zip(COLLECTIVE_BLOCKS, forms, phases):
        for node in block:
            graph.nodes[node]["EPI"] = float(form)
            graph.nodes[node]["theta"] = float(phase)
    _assert_fine_and_collective_state(mp, state, graph)
    # The hypothetical point does not overwrite the actual source association.
    assert state.partition.source.epi == source.epi
    assert state.partition.source.phase == source.phase


def test_complete_actual_word_matches_independent_reset_and_five_link_inventory(
    contact, mp
):
    mean, phase, states, stages, graph, final, _ = contact
    source, emitted, coupled, silenced = states
    assert len(stages) == 3
    assert all(
        stage.schedule == TWO_PHASE_JACOBI and stage.nodes_processed == 1
        for stage in stages
    )
    assert tuple(stage.glyph for stage in stages) == ("AL", "UM", "SHA")
    assert set(coupled.edges) - set(source.edges) == set(NEW_EDGES)
    assert final.degrees == POST_DEGREES
    assert final.capacity == (4, 2, 2, 2, Q(117, 32), 4, 2, 2, 2, 8)
    assert Q(emitted.nodes[4]["nu_f"]) == 8
    assert Q(coupled.nodes[4]["nu_f"]) == Q(117, 16)
    assert _form(emitted, 4) == mean + BOOST
    assert _form(coupled, 4) == _form(silenced, 4) == mean + BOOST
    for edge, weight in zip(OLD_EDGES, OLD_WEIGHTS):
        assert (
            coupled.edges[edge]
            == source.edges[edge]
            == {"weight": float(weight), "length": 1.0}
        )
    for edge in NEW_EDGES:
        target_form = mean + (AMPLITUDE if edge[1] != 9 else 0)
        expected_weight = (
            1
            - _mp(mp, phase) / (2 * mp.pi)
            - _mp(mp, abs(mean + BOOST - target_form))
            / (4 * _mp(mp, abs(mean + BOOST) + abs(target_form) + Q(1e-9)))
        )
        actual = Q(coupled.edges[edge]["weight"])
        assert set(coupled.edges[edge]) == {"weight"}
        assert Q(3, 4) < actual < 1
        assert abs(_mp(mp, actual) - expected_weight) < _mp(mp, SCORE_TOLERANCE)
    assert len({coupled.edges[4, i]["weight"] for i in range(5, 9)}) == 1
    assert coupled.edges[4, 9]["weight"] > coupled.edges[4, 5]["weight"]
    for state in states:
        assert state.graph["_t"] == 7 and state.graph["RANDOM_SEED"] == 17
        assert "_node_sample" not in state.graph
        assert (
            tuple(Q(state.nodes[node]["theta"]) for node in state)
            == (Q(0),) * 5 + (phase,) * 5
        )
        assert all(state.nodes[node]["delta_nfr"] == 0 for node in state)
    for node in graph:
        data = graph.nodes[node]
        if node != 4:
            assert _form(graph, node) == _form(source, node)
            assert data["nu_f"] == source.nodes[node]["nu_f"]
            assert data["glyph_history"] == []
            assert "_operator_step" not in data
            assert "emission_timestamp" not in data and "latent" not in data
    target = graph.nodes[4]
    assert tuple(target["glyph_history"]) == ("AL", "UM", "SHA")
    assert target["_operator_step"] == 3 and target["_grammar_u2_debt"] == 0
    assert target["emission_timestamp"] == NOW.isoformat()
    assert target["_emission_origin"] == NOW.isoformat()
    assert target["_emission_activated"] is True
    assert target["_structural_lineage"]["activation_count"] == 1
    assert target["_structural_lineage"]["parent_emission"] is None
    assert target["_structural_lineage"]["derived_nodes"] == []
    assert target["latent"] is True
    assert target["preserved_epi"] == float(mean + BOOST)
    assert target["latency_start_time"] == NOW.isoformat()
    assert getattr(graph, "_last_operator_applied") == "silence"
    assert all("epi_time_history" not in data for _, data in graph.nodes(data=True))


def test_complete_work_includes_al_old_edge_change_actual_um_edges_and_zero_sha_jump(
    contact, mp
):
    _, phase, states, _, _, final, retention = contact
    source, emitted, coupled, silenced = states
    source_components = tuple(
        bound_relational_sine_support_exchange(
            _component(source, nodes), reference_model=MODEL
        )
        for nodes in STARS
    )
    assert tuple(row.form_storage for row in source_components) == (Q(1, 64),) * 2
    assert all(row.storage == I(Q(1, 64)) for row in source_components)
    old_before = _edge_storage(mp, source, OLD_EDGES)
    old_emitted = _edge_storage(mp, emitted, OLD_EDGES)
    old_coupled = _edge_storage(mp, coupled, OLD_EDGES)
    attachment = _edge_storage(mp, coupled, NEW_EDGES)
    expected_al_work = -8 * AMPLITUDE * BOOST + 4 * BOOST**2
    assert expected_al_work < 0
    assert abs(old_emitted - old_before - _mp(mp, expected_al_work)) < mp.mpf("1e-140")
    assert old_coupled == old_emitted
    assert attachment > 0
    assert (
        _edge_storage(mp, silenced, OLD_EDGES + NEW_EDGES) == old_coupled + attachment
    )
    assert _edge_storage(mp, coupled, OLD_EDGES + NEW_EDGES) == _edge_storage(
        mp, silenced, OLD_EDGES + NEW_EDGES
    )
    _contains(mp, final.storage, old_coupled + attachment)
    # The actual affinity weights enter form storage; every new bare edge
    # contributes its complete phase cost, independently of that weight.
    expected_attachment = (
        2 * _mp(mp, Q(coupled.edges[4, 5]["weight"]) * (AMPLITUDE - BOOST) ** 2)
        + _mp(mp, Q(coupled.edges[4, 9]["weight"]) * BOOST**2 / 2)
        + 5 * (1 - mp.cos(_mp(mp, phase)))
    )
    assert abs(attachment - expected_attachment) < mp.mpf("1e-140")
    assert retention.all_time_identity_retention_certified
    assert retention.source_identity_admitted
    assert retention.storage_margin > 0
    assert retention.full_storage_bounds.hi < Q(35, 64)
    assert retention.source.capacity == final.capacity
    assert retention.protected_edges == OLD_EDGES
    assert retention.protected_weights == OLD_WEIGHTS
    assert all(retention.protected_edge_checks)


def test_actual_successor_rows_break_right_family_tangency_and_exchange_full_storage(
    contact, mp
):
    _, alpha, states, _, _, final, _ = contact
    graph = states[-1]
    pressure, dx, phase_numerators, dtheta = _independent_field(mp, graph)
    assert final.phase_rate_numerators() == phase_numerators
    for intervals, expected in (
        (final.pressure, pressure),
        (final.form_rates, dx),
        (final.phase_rates, dtheta),
    ):
        for index, (interval, value) in enumerate(zip(intervals, expected)):
            if index < 4 and intervals in (final.pressure, final.form_rates):
                assert interval == I(0)
            else:
                _contains(mp, interval, value)
    work = []
    for selected in (OLD_EDGES[:7], OLD_EDGES[7:], NEW_EDGES):
        work.append(
            sum(
                _mp(
                    mp,
                    Q(graph.edges[i, j]["weight"])
                    * (_form(graph, i) - _form(graph, j)),
                )
                * (dx[i] - dx[j])
                + mp.sin(
                    _mp(mp, Q(graph.nodes[j]["theta"]) - Q(graph.nodes[i]["theta"]))
                )
                * (dtheta[j] - dtheta[i])
                for i, j in selected
            )
        )
    sine = mp.sin(_mp(mp, alpha))
    expected_left = -mp.mpf(4095) * sine / (4096 * mp.pi)
    expected_right = 9 * sine / (20 * mp.pi)
    assert abs(work[0] - expected_left) < mp.mpf("1e-140")
    assert abs(work[1] - expected_right) < mp.mpf("1e-140")
    assert work[0] < 0 < work[1] and work[2] > 0
    assert abs(sum(work)) < mp.mpf("1e-140")
    assert final.storage_rate.contains(0) and final.balance_residual.contains(0)
    # The right A members still have identical coordinates, yet their
    # actual degree-normalized velocities immediately leave that family.
    assert len({_form(graph, node) for node in range(5, 9)}) == 1
    assert len({graph.nodes[node]["theta"] for node in range(5, 9)}) == 1
    contrast = -2 * sine / (15 * mp.pi)
    assert abs(dx[5] - dx[6] - contrast) < mp.mpf("1e-140")
    assert final.form_rates[5].hi < final.form_rates[6].lo
    assert all(graph.nodes[node]["delta_nfr"] == 0 for node in graph)
    assert final.pressure[4].lo > 0 and all(
        final.pressure[node].hi < 0 for node in range(5, 10)
    )
    refreshed = copy_graph_state(graph)
    for node, enclosure in zip(final.nodes, final.pressure):
        set_dnfr(refreshed, node, float(enclosure.midpoint))
    after = bound_relational_sine_support_exchange(refreshed, reference_model=MODEL)
    assert after == final
    assert refreshed.nodes[4]["delta_nfr"] > 0
    assert graph.nodes[4]["delta_nfr"] == 0
    assert refreshed.nodes[4]["glyph_history"] == graph.nodes[4]["glyph_history"]


def _structural_snapshot(graph):
    attributes = dict(graph.graph)
    if "_node_cache" in attributes:
        for node, adapter in attributes["_node_cache"].items():
            assert adapter.G is graph and adapter.n == node
        attributes["_node_cache"] = tuple(sorted(attributes["_node_cache"]))
    return deepcopy(
        (
            dict(graph.nodes),
            dict(graph.edges),
            attributes,
            getattr(graph, "_last_operator_applied", None),
        )
    )


@pytest.mark.parametrize("fault", ("untranslated", "existing_phase", "candidate_sense"))
def test_rejected_um_is_atomic_after_the_committed_al_prefix(fault):
    graph = _source(Q(0) if fault == "untranslated" else PROBES[0][0], PROBES[0][1])
    if fault == "existing_phase":
        graph.nodes[4]["theta"] = pi
    elif fault == "candidate_sense":
        graph.nodes[9]["Si"] = -0.5
    operators, sequence = _word()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        execute_network_operator_stage(
            graph, operators[0], (4,), sequence_context=sequence.step(0)
        )
        before = _structural_snapshot(graph)
        error = (
            TNFRValueError if fault == "candidate_sense" else OperatorPreconditionError
        )
        with pytest.raises(error):
            execute_network_operator_stage(
                graph, operators[1], (4,), sequence_context=sequence.step(1)
            )
    assert _structural_snapshot(graph) == before
    assert tuple(graph.nodes[4]["glyph_history"]) == ("AL",)
    assert graph.nodes[4]["_operator_step"] == 1
    assert graph.nodes[4]["nu_f"] == 8
    assert set(graph.edges) == set(OLD_EDGES)
    assert "latent" not in graph.nodes[4]
    assert getattr(graph, "_last_operator_applied") == "emission"


def test_actual_support_retention_ignores_cached_fields_and_exports_fresh(
    contacts, tmp_path
):
    source, retention = contacts[0][-2:]
    poisoned = replace(
        source,
        degrees=(999,) * 10,
        pressure=(I(999),) * 10,
        form_rates=(I(999),) * 10,
        storage=I(999),
    )
    rebuilt = assess_sine_weighted_star_support_retention(poisoned, stars=STARS)
    assert rebuilt == retention
    assert rebuilt.source.degrees == POST_DEGREES
    # A fabricated favorable report cannot hide a primitive form boundary
    # violation. No operator is run on this deliberately altered source.
    excluded_source = replace(
        source, epi=(Q(10),) + source.epi[1:], storage=I(0), form_storage=Q(0)
    )
    excluded = assess_sine_weighted_star_support_retention(excluded_source, stars=STARS)
    assert excluded.source_identity_status == "excluded"
    assert excluded.storage_margin < 0
    assert not excluded.all_time_identity_retention_certified
    forged = replace(
        excluded,
        full_storage_bounds=I(0),
        storage_margin=Q(999),
        source_identity_admitted=True,
        all_time_identity_retention_certified=True,
        status="certified_identity_retention",
    )
    generic = relational_report_to_dict(forged)
    assert generic["report"] == excluded.to_dict()["report"]
    assert generic["report"]["all_time_identity_retention_certified"] is False
    assert generic["report"]["identity_storage_boundary_lower_bound"] == {
        "numerator": 35,
        "denominator": 64,
    }
    assert generic["report"]["source"]["degrees"] == list(POST_DEGREES)
    destination = tmp_path / "weighted-star-support-retention.json"
    export_to_json(generic, destination)
    assert json_loads(destination.read_text(encoding="utf-8")) == generic
    before = destination.read_bytes()
    invalid = replace(
        forged, source=replace(source, capacity=(False,) + source.capacity[1:])
    )
    with pytest.raises((TypeError, ValueError)):
        export_to_json(relational_report_to_dict(invalid), destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize(
    "defect",
    ("missing", "weakened", "capacity_zero", "phase_boolean", "form_nan", "law_beta"),
)
def test_retention_requires_actual_protected_support_law_and_admitted_primitives(
    contacts, defect
):
    source = contacts[0][-2]
    index = source.edges.index((0, 1))
    if defect == "missing":
        source = replace(
            source,
            edges=source.edges[:index] + source.edges[index + 1 :],
            conductances=source.conductances[:index] + source.conductances[index + 1 :],
        )
    elif defect == "weakened":
        source = replace(
            source,
            conductances=source.conductances[:index]
            + (Q(3) - Q(1, 2**60),)
            + source.conductances[index + 1 :],
        )
    elif defect == "capacity_zero":
        source = replace(source, capacity=(Q(0),) + source.capacity[1:])
    elif defect == "phase_boolean":
        source = replace(source, phase=(False,) + source.phase[1:])
    elif defect == "form_nan":
        source = replace(source, epi=(float("nan"),) + source.epi[1:])
    elif defect == "law_beta":
        source = replace(
            source,
            reference_model=RelationalExchangeModel(
                2, epi_weight=0, phase_domain="regular"
            ),
        )
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_support_retention(source, stars=STARS)


@pytest.mark.parametrize(
    "marks",
    (
        STARS[:1],
        (STARS[0], STARS[0]),
        (STARS[0], (5, 6, 7, 8)),
        (STARS[0], (5, 6, 7, 8, 99)),
    ),
)
def test_marked_stars_must_cover_the_actual_ten_nodes_once(contacts, marks):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_weighted_star_support_retention(contacts[0][-2], stars=marks)


def test_retention_support_orientation_and_zero_added_conductance_keep_bare_incidence(
    contacts,
):
    source, retained = contacts[0][-2:]
    reversed_source = replace(source, edges=tuple((j, i) for i, j in source.edges))
    reversed_retained = assess_sine_weighted_star_support_retention(
        reversed_source, stars=STARS
    )
    assert reversed_retained.protected_weights == retained.protected_weights
    assert reversed_retained.all_time_identity_retention_certified
    index = source.edges.index((4, 5))
    zero_extra = replace(
        source,
        conductances=source.conductances[:index]
        + (Q(0),)
        + source.conductances[index + 1 :],
    )
    changed = assess_sine_weighted_star_support_retention(zero_extra, stars=STARS)
    assert changed.protected_weights == retained.protected_weights
    assert changed.source.degrees == POST_DEGREES
    assert changed.source.pressure == retained.source.pressure
    assert changed.source.phase_storage == retained.source.phase_storage
    assert changed.source.form_storage < retained.source.form_storage
    assert changed.all_time_identity_retention_certified


def test_retention_report_does_not_dispatch_untrusted_source_types(contacts):
    class Fake:
        def to_dict(self):
            raise AssertionError("untrusted dispatch")

    with pytest.raises(TypeError):
        assess_sine_weighted_star_support_retention(Fake(), stars=STARS)
    with pytest.raises(TypeError):
        type(contacts[0][-1]).to_dict(Fake())


def test_retention_strict_boundary_and_large_energy_abstention_are_not_escape(contacts):
    source = contacts[0][-2]
    first = (Q(1, 2), Q(0), Q(7, 16), Q(7, 16), Q(11, 32))
    boundary_source = replace(
        source,
        epi=first + (first[4],) * 5,
        phase=(Q(0),) * 10,
        storage=I(0),
    )
    boundary = assess_sine_weighted_star_support_retention(boundary_source, stars=STARS)
    # Every actual added edge has zero cost in this harmonic extension.
    # The original star reaches its form boundary at exactly35/64.
    assert boundary.full_storage_bounds == I(Q(35, 64))
    assert boundary.storage_margin == 0
    assert boundary.source_identity_status == "excluded"
    assert not boundary.all_time_identity_retention_certified
    coherent_source = replace(
        source, epi=((Q(1, 2),) * 4 + (Q(0),)) * 2, phase=(Q(0),) * 10
    )
    coherent = assess_sine_weighted_star_support_retention(coherent_source, stars=STARS)
    assert coherent.internal_form_gaps == ((0, 0, 0),) * 2
    assert coherent.source_identity_status == "admitted"
    assert coherent.storage_margin < 0
    assert coherent.status == "unavailable"
    assert not coherent.all_time_identity_retention_certified
    assert coherent.unavailable_reasons == ("strict_full_storage_margin_not_certified",)


@pytest.fixture(scope="module")
def matched_controls(contacts):
    """Section8's only policy change; reuse each attached endpoint once."""
    controls = []
    for mean, phase, *_ in contacts:
        graph = _source(mean, phase)
        graph.graph["UM_FUNCTIONAL_LINKS"] = False
        states, stages, graph = _execute_static_word(graph)
        components = tuple(
            bound_relational_sine_support_exchange(
                _component(states[-1], nodes), reference_model=MODEL
            )
            for nodes in STARS
        )
        controls.append((states, stages, graph, components))
    return tuple(controls)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_matched_disabled_link_word_retains_same_node_reset_and_two_complete_laws(
    contacts,
    matched_controls,
    index,
):
    attached = contacts[index]
    states, stages, graph, components = matched_controls[index]
    assert states[0].graph == attached[2][0].graph | {"UM_FUNCTIONAL_LINKS": False}
    assert stages == attached[3]
    assert dict(graph.nodes) == dict(attached[4].nodes)
    assert getattr(graph, "_last_operator_applied") == "silence"
    assert set(graph.edges) == set(OLD_EDGES)
    assert tuple(graph.degree(node) for node in graph) == (4, 2, 2, 2, 4) * 2
    for edge in OLD_EDGES:
        assert graph.edges[edge] == attached[4].edges[edge]
    assert graph.graph["_t"] == 7 and graph.graph["RANDOM_SEED"] == 17
    assert tuple(graph.nodes[4]["glyph_history"]) == ("AL", "UM", "SHA")
    assert graph.nodes[4]["_operator_step"] == 3
    assert graph.nodes[4]["_grammar_u2_debt"] == 0
    assert all("epi_time_history" not in data for _, data in graph.nodes(data=True))
    assert components[0].capacity == (4, 2, 2, 2, Q(117, 32))
    assert components[1].capacity == (4, 2, 2, 2, 8)
    assert components[0].storage == I(4 * (AMPLITUDE - BOOST) ** 2)
    assert components[1].storage == I(Q(1, 64))
    # The original positive weighted subgraphs and held positive capacities
    # retain the per-star Dirichlet barrier, without a connected graph fiction.
    for component in components:
        assert component.storage.hi < Q(35, 64)
        assert all(value > 0 for value in component.capacity)
        assert len(set(component.epi[:4])) == 1
        assert len(set(component.phase[:4])) == 1
        assert component.degrees == (4, 2, 2, 2, 4)
    with pytest.raises(ValueError, match="connected"):
        bound_relational_sine_support_exchange(graph, reference_model=MODEL)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_finite_attached_contrast_is_strictly_separated_from_exact_control_family(
    contacts,
    matched_controls,
    index,
):
    attached, retention = contacts[index][-2:]
    right = matched_controls[index][-1][1]
    duration = Q(1, 128)
    response = attached.finite_form_response_bounds(duration=duration)
    assert response.duration == duration
    assert response.clock == "t"
    assert attached.epi[5] == attached.epi[6]
    contrast = response.endpoint_form_bounds[5] - response.endpoint_form_bounds[6]
    assert contrast.hi < Q(-1, 32768)
    assert response.form_remainder_upper_bounds[5] > 0
    assert response.form_remainder_upper_bounds[6] > 0
    assert (
        response.whole_window_form_radius_upper_bound
        > response.initial_centered_form_radius_upper_bound
    )
    assert retention.all_time_identity_retention_certified
    # This is a whole-family criterion plus actual membership, not an
    # extrapolation of this control's zero initial form velocities.
    family = assess_sine_weighted_phase_offset_partition(
        right,
        blocks=(STARS[1][:4], STARS[1][4:]),
        phase_offset_turns=(0,) * 5,
    )
    assert family.invariance_certified
    assert len(set(right.epi[:4])) == len(set(right.phase[:4])) == 1
    assert right.nodes == STARS[1]
    assert right.form_rates == (I(0),) * 5
    assert right.phase_rate_numerators() == (Q(1, 8),) * 4 + (Q(-1),)
    assert right.storage.hi < Q(35, 64)
    assert response.source.edges == attached.edges
    assert response.source.capacity == attached.capacity


def test_zero_initial_control_rates_do_not_certify_a_null_without_capacity_admission(
    matched_controls,
    mp,
):
    right = matched_controls[0][-1][1]
    altered = replace(right, capacity=(Q(8),) + right.capacity[1:])
    family = assess_sine_weighted_phase_offset_partition(
        altered,
        blocks=(STARS[1][:4], STARS[1][4:]),
        phase_offset_turns=(0,) * 5,
    )
    assert not family.invariance_certified
    fresh = family.source
    assert fresh.form_rates == (I(0),) * 5
    assert fresh.phase_rate_numerators() == (Q(1, 4), Q(1, 8), Q(1, 8), Q(1, 8), Q(-1))
    acceleration = fresh.form_acceleration_bounds()
    # Different phase velocities produce a nonzero second derivative even
    # though all initial form velocities vanish: (-52a)-(-16a)=-36a.
    expected = -36 * _mp(mp, AMPLITUDE) / mp.pi**2
    _contains(mp, acceleration[0] - acceleration[1], expected)
    assert (acceleration[0] - acceleration[1]).hi < 0


@pytest.fixture(scope="module")
def perturbation_reports(contacts, matched_controls):
    """Section18 boxes reuse the existing word outputs without new execution."""
    epsilon, duration = Q(1, 2**20), Q(1, 128)
    result = []
    for contact, control in zip(contacts, matched_controls):
        source = contact[5]
        attached = source.initial_state_perturbation_bounds(
            duration=duration,
            form_error_bounds=(epsilon,) * 10,
            phase_error_bounds=(epsilon,) * 10,
        )
        components = tuple(
            component.initial_state_perturbation_bounds(
                duration=duration,
                form_error_bounds=(epsilon,) * 5,
                phase_error_bounds=(epsilon,) * 5,
            )
            for component in control[-1]
        )
        result.append((attached, components))
    return tuple(result)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_full_independent_error_box_has_same_law_bounds_and_retained_identities(
    contacts, matched_controls, perturbation_reports, mp, index
):
    epsilon = Q(1, 2**20)
    attached, components = perturbation_reports[index]
    source = contacts[index][5]
    reference = _collective_partition(source).observe_radian_state()
    assert reference.source_membership_certified
    assert reference.full_source == attached.source
    assert tuple(report.source for report in components) == matched_controls[index][-1]
    assert attached.source.degrees == POST_DEGREES
    assert components[1].source.degrees == (4, 2, 2, 2, 4)
    assert attached.source.capacity[4] == components[0].source.capacity[4] == Q(117, 32)
    for report in (attached, *components):
        size = len(report.source.nodes)
        assert (
            report.form_error_bounds == report.phase_error_bounds == (epsilon,) * size
        )
        assert report.status == "enclosed" and report.comparison_denominator > 0
        assert report.phase_error_units == "radians_in_declared_real_lifts"
        assert report.duration == Q(1, 128) and report.clock == "t"
        assert report.max_form_phase_lipschitz_upper_bound < Q(16, 3)
        assert report.max_phase_form_lipschitz_upper_bound < Q(32, 3)
        assert epsilon < report.form_error_supremum_upper_bound < Q(21, 20) * epsilon
        assert epsilon < report.phase_error_supremum_upper_bound < Q(11, 10) * epsilon
        assert len(report.whole_window_form_error_bounds) == size
        assert len(report.whole_window_phase_error_bounds) == size
        assert all(
            epsilon <= bound <= report.form_error_supremum_upper_bound
            for bound in report.whole_window_form_error_bounds
        )
        assert all(
            epsilon <= bound <= report.phase_error_supremum_upper_bound
            for bound in report.whole_window_phase_error_bounds
        )

        # Independently bound every edge of the entire initial box. This
        # covers arbitrary signs and means, not a projected coherent sample.
        positions = {node: i for i, node in enumerate(report.source.nodes)}
        expected_form, expected_phase = Q(0), mp.mpf(0)
        for (left, right), weight in zip(
            report.source.edges, report.source.conductances
        ):
            i, j = positions[left], positions[right]
            expected_form += (
                weight
                * (abs(report.source.epi[i] - report.source.epi[j]) + 2 * epsilon) ** 2
                / 2
            )
            expected_phase += min(
                mp.mpf(2),
                1
                - mp.cos(_mp(mp, report.source.phase[j] - report.source.phase[i]))
                + _mp(mp, 2 * epsilon),
            )
        assert report.initial_form_storage_upper_bound == expected_form
        assert (
            0
            <= _mp(mp, report.initial_phase_storage_upper_bound) - expected_phase
            < mp.mpf("1e-30")
        )
        assert (
            report.initial_storage_upper_bound
            == expected_form + report.initial_phase_storage_upper_bound
        )
        assert report.initial_storage_upper_bound < Q(35, 64)
        actual_weights = dict(zip(report.source.edges, report.source.conductances))
        for edge, minimum in zip(OLD_EDGES, OLD_WEIGHTS):
            if edge[0] in positions and edge[1] in positions:
                assert actual_weights[edge] == minimum
    # The control boundary has its own larger weighted normalization. The
    # attached coefficient cannot replace it even though initial nodes match.
    assert (
        components[1].max_phase_form_lipschitz_upper_bound
        > attached.max_phase_form_lipschitz_upper_bound
    )
    assert 2 * epsilon < Q(1, 2)  # Strict initial form identity for every member.
    assert 2 * epsilon < 1  # Radian phase gaps stay strictly below pi/2 initially.


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_matched_contrast_stays_strict_without_assuming_perturbed_control_is_null(
    contacts, perturbation_reports, index
):
    epsilon, duration = Q(1, 2**20), Q(1, 128)
    attached, (_, right_control) = perturbation_reports[index]
    nominal = contacts[index][5].finite_form_response_bounds(duration=duration)
    nominal_contrast = nominal.endpoint_form_bounds[5] - nominal.endpoint_form_bounds[6]
    control_partition = assess_sine_weighted_phase_offset_partition(
        right_control.source,
        blocks=((5, 6, 7, 8), (9,)),
        phase_offset_turns=(0,) * 5,
    )
    assert control_partition.observe_radian_state().source_membership_certified
    # Only the unperturbed reference is exactly null. Each perturbed contrast
    # can differ from its own reference by two form-error suprema.
    on_error = attached.form_error_supremum_upper_bound
    off_error = right_control.form_error_supremum_upper_bound
    assert on_error > 0 and off_error > 0
    control_contrast_error = 2 * off_error
    assert control_contrast_error > 0
    robust_upper = nominal_contrast.hi + 2 * (on_error + off_error)
    assert robust_upper < -Q(1, 32768)
    # Independent rational margin from the already proved nominal theorem
    # and the declared uniform full-state error bound, without any samples.
    assert -Q(401, 8110080) + Q(21, 5) * epsilon < -Q(1, 32768)
    assert 2 * (on_error + off_error) < Q(21, 5) * epsilon


PREPARATION_EPSILON = Q(1, 2**24)
PREPARATION_SIGMA = Q(1, 2**42)
PREPARATION_FORM_SIGNS = (1, -1, 1, -1, 1, -1, 1, -1, 1, -1)
PREPARATION_PHASE_SIGNS = tuple(-value for value in PREPARATION_FORM_SIGNS)
PREPARATION_LIFT_TURNS = (1, 0, 1, 0, 1, 0, 0, 0, 0, 0)


def _execute_recorded_preparation_word(graph):
    """Record consumed scalar arithmetic during the one declared execution."""
    differences, consensuses = [], []
    original_difference = _coupling_stage_kernel.angle_diff
    original_consensus = _coupling_stage_kernel.compute_consensus_phase

    def difference(first, second):
        result = original_difference(first, second)
        differences.append((Q(first), Q(second), Q(result)))
        return result

    def consensus(phases):
        phases = tuple(phases)
        result = original_consensus(phases)
        consensuses.append((tuple(Q(value) for value in phases), Q(result)))
        return result

    with pytest.MonkeyPatch.context() as patch:
        for module in (_coupling_stage_kernel, _phase_gate, phase_compatibility):
            patch.setattr(module, "angle_diff", difference)
        patch.setattr(_coupling_stage_kernel, "compute_consensus_phase", consensus)
        response = _execute_static_word(graph)
    return response, (tuple(differences), tuple(consensuses))


@pytest.fixture(scope="module")
def preparation_words(contacts, matched_controls, record_testsuite_property):
    """The six static words fixed by section22, each evaluated exactly once."""
    record_testsuite_property("preparation_epsilon", str(PREPARATION_EPSILON))
    record_testsuite_property("preparation_sigma", str(PREPARATION_SIGMA))
    record_testsuite_property("preparation_form_signs", str(PREPARATION_FORM_SIGNS))
    record_testsuite_property("preparation_phase_signs", str(PREPARATION_PHASE_SIGNS))
    record_testsuite_property("preparation_lift_turns", str(PREPARATION_LIFT_TURNS))
    records = []
    for mean, alpha in PROBES:
        pair = []
        for functional in (True, False):
            graph = _source(mean, alpha)
            graph.graph["UM_FUNCTIONAL_LINKS"] = functional
            for node in graph:
                graph.nodes[node]["EPI"] = float(
                    _form(graph, node)
                    + PREPARATION_EPSILON * PREPARATION_FORM_SIGNS[node]
                )
                graph.nodes[node]["theta"] = float(
                    Q(graph.nodes[node]["theta"])
                    + PREPARATION_EPSILON * PREPARATION_PHASE_SIGNS[node]
                )
            pair.append(_execute_recorded_preparation_word(graph))
        records.append(tuple(pair))
    return tuple(records)


def _ideal_preparation_phases(mp, source):
    left = tuple(_mp(mp, Q(source.nodes[node]["theta"])) for node in range(5))
    consensus = mp.atan2(
        sum(mp.sin(value) for value in left), sum(mp.cos(value) for value in left)
    )
    return tuple(
        value + _mp(mp, BOOST) * (consensus - value) for value in left
    ) + tuple(_mp(mp, Q(source.nodes[node]["theta"])) for node in range(5, 10))


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_preparation_word_has_independent_arithmetic_and_full_context_association(
    contacts, matched_controls, preparation_words, mp, index
):
    on, off = preparation_words[index]
    (states, stages, graph), arithmetic = on
    (off_states, off_stages, off_graph), off_arithmetic = off
    initial, emitted, coupled, final = states
    nominal = contacts[index]
    assert initial.graph == nominal[2][0].graph
    assert off_states[0].graph == initial.graph | {"UM_FUNCTIONAL_LINKS": False}
    assert off_stages == stages == nominal[3]
    assert dict(off_graph.nodes) == dict(graph.nodes)
    assert set(off_graph.edges) == set(OLD_EDGES)
    assert set(graph.edges) == set(OLD_EDGES + NEW_EDGES)
    ideal_phases = _ideal_preparation_phases(mp, initial)
    for differences, consensuses in (arithmetic, off_arithmetic):
        assert differences and len(consensuses) == 1
        for first, second, result in differences:
            delta = _mp(mp, first - second)
            ideal = mp.atan2(mp.sin(delta), mp.cos(delta))
            assert abs(_mp(mp, result) - ideal) < _mp(mp, PREPARATION_SIGMA)
        for phases, result in consensuses:
            assert len(phases) == 5
            ideal = mp.atan2(
                sum(mp.sin(_mp(mp, value)) for value in phases),
                sum(mp.cos(_mp(mp, value)) for value in phases),
            )
            assert abs(_mp(mp, result) - ideal) < _mp(mp, PREPARATION_SIGMA)
    for node in graph:
        original = nominal[2][0].nodes[node]
        expected_form = (
            Q(original["EPI"]) + PREPARATION_FORM_SIGNS[node] * PREPARATION_EPSILON
        )
        expected_phase = (
            Q(original["theta"]) + PREPARATION_PHASE_SIGNS[node] * PREPARATION_EPSILON
        )
        assert _form(initial, node) == expected_form
        assert Q(initial.nodes[node]["theta"]) == expected_phase
        assert _form(final, node) == expected_form + (BOOST if node == 4 else 0)
        lifted = (
            _mp(mp, Q(final.nodes[node]["theta"]))
            - 2 * mp.pi * PREPARATION_LIFT_TURNS[node]
        )
        assert abs(lifted - ideal_phases[node]) < _mp(mp, PREPARATION_SIGMA)
        assert final.nodes[node]["nu_f"] == nominal[4].nodes[node]["nu_f"]
        assert final.nodes[node]["delta_nfr"] == 0
        assert final.nodes[node]["Si"] == Q(1, 2)
        assert "epi_time_history" not in final.nodes[node]
        if node != 4:
            assert graph.nodes[node]["glyph_history"] == []
            assert "_operator_step" not in graph.nodes[node]
            assert "latent" not in graph.nodes[node]
    target = graph.nodes[4]
    assert tuple(target["glyph_history"]) == ("AL", "UM", "SHA")
    assert target["_operator_step"] == 3 and target["_grammar_u2_debt"] == 0
    assert (
        target["emission_timestamp"] == target["latency_start_time"] == NOW.isoformat()
    )
    assert target["_emission_activated"] and target["latent"]
    assert target["_structural_lineage"] == nominal[4].nodes[4]["_structural_lineage"]
    assert Q(target["preserved_epi"]) == _form(final, 4)
    assert getattr(graph, "_last_operator_applied") == "silence"
    assert Q(emitted.nodes[4]["nu_f"]) == 8
    assert Q(coupled.nodes[4]["nu_f"]) == Q(117, 16)
    assert Q(final.nodes[4]["nu_f"]) == Q(117, 32)
    for state in states + off_states:
        assert state.graph["_t"] == 7 and state.graph["RANDOM_SEED"] == 17
        assert "_node_sample" not in state.graph
    for edge, weight in zip(OLD_EDGES, OLD_WEIGHTS):
        assert (
            graph.edges[edge]
            == initial.edges[edge]
            == {"weight": float(weight), "length": 1.0}
        )
    for _, node in NEW_EDGES:
        separation = abs(ideal_phases[node] - ideal_phases[4])
        left, right = _form(final, 4), _form(final, node)
        ideal = (
            1
            - separation / (2 * mp.pi)
            - _mp(mp, abs(left - right))
            / (4 * _mp(mp, abs(left) + abs(right) + Q(1e-9)))
        )
        actual = Q(graph.edges[4, node]["weight"])
        assert Q(3, 4) < actual < 1
        assert abs(_mp(mp, actual) - ideal) < _mp(mp, PREPARATION_SIGMA)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_preparation_work_and_full_law_do_not_restore_exact_collective_closure(
    preparation_words, mp, index
):
    (states, _, graph), _ = preparation_words[index][0]
    initial, emitted, coupled, final = states
    source = bound_relational_sine_support_exchange(final, reference_model=MODEL)
    retained = assess_sine_weighted_star_support_retention(source, stars=STARS)
    pressure, form_rates, phase_numerators, phase_rates = _independent_field(mp, final)
    assert source.phase == tuple(Q(final.nodes[node]["theta"]) for node in final)
    assert source.degrees == POST_DEGREES
    assert source.phase_rate_numerators() == phase_numerators
    for intervals, exacts in (
        (source.pressure, pressure),
        (source.form_rates, form_rates),
        (source.phase_rates, phase_rates),
    ):
        for interval, exact in zip(intervals, exacts):
            _contains(mp, interval, exact)
    al_work = sum(
        Q(initial.edges[4, node]["weight"])
        * (BOOST * (_form(initial, 4) - _form(initial, node)) + BOOST**2 / 2)
        for node in initial.neighbors(4)
    )
    h_initial = _edge_storage(mp, initial, OLD_EDGES)
    h_emitted = _edge_storage(mp, emitted, OLD_EDGES)
    h_old_coupled = _edge_storage(mp, coupled, OLD_EDGES)
    inserted = _edge_storage(mp, coupled, NEW_EDGES)
    assert abs(h_emitted - h_initial - _mp(mp, al_work)) < mp.mpf("1e-140")
    assert h_old_coupled < h_emitted  # Actual old-edge phase work is no longer zero.
    assert inserted > 0
    h_final = _edge_storage(mp, final, OLD_EDGES + NEW_EDGES)
    assert abs(h_final - h_old_coupled - inserted) < mp.mpf("1e-140")
    assert h_final == _edge_storage(mp, coupled, OLD_EDGES + NEW_EDGES)
    _contains(mp, source.storage, h_final)
    assert retained.all_time_identity_retention_certified
    assert retained.protected_weights == OLD_WEIGHTS
    # The frozen alternating pattern splits the four former equal affinities.
    assert len({graph.edges[4, node]["weight"] for node in range(5, 9)}) > 1
    partition = _collective_partition(source)
    assert not partition.weighted_compatibility_certified
    assert not partition.invariance_certified
    assert partition.status == "excluded"
    with pytest.raises(ValueError, match="certified"):
        partition.observe_radian_state()
    assert len(set(source.epi[:4])) > 1 and len(set(source.phase[:4])) > 1


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_preparation_associates_actual_changed_laws_then_transfers_finite_response(
    contacts, matched_controls, preparation_words, mp, index
):
    epsilon, duration = PREPARATION_EPSILON, Q(1, 128)
    (on_states, _, _), _ = preparation_words[index][0]
    (off_states, _, _), _ = preparation_words[index][1]
    nominal = contacts[index][5]
    actual = bound_relational_sine_support_exchange(
        on_states[-1], reference_model=MODEL
    )
    reports = []
    candidates = (actual,) + tuple(
        bound_relational_sine_support_exchange(
            _component(off_states[-1], nodes), reference_model=MODEL
        )
        for nodes in STARS
    )
    references = (nominal,) + matched_controls[index][-1]
    for candidate, reference in zip(candidates, references):
        assert candidate.nodes == reference.nodes and candidate.edges == reference.edges
        assert candidate.capacity == reference.capacity
        assert candidate.reference_model == reference.reference_model
        errors = tuple(
            4 * epsilon if edge in NEW_EDGES else Q(0) for edge in reference.edges
        )
        report = reference.state_and_conductance_perturbation_bounds(
            duration=duration,
            form_error_bounds=(2 * epsilon,) * len(reference.nodes),
            phase_error_bounds=(2 * epsilon,) * len(reference.nodes),
            conductance_error_bounds=errors,
        )
        for i, node in enumerate(candidate.nodes):
            assert abs(candidate.epi[i] - reference.epi[i]) <= 2 * epsilon
            # This is an interval around an exact mathematical 2*pi lift.
            # Raw phases remain in the source; Q(math.tau) is not substituted.
            lift = (
                I(candidate.phase[i] - reference.phase[i])
                - 2 * PREPARATION_LIFT_TURNS[node] * pi_interval()
            )
            assert lift.abs_max <= 2 * epsilon
            exact = (
                _mp(mp, candidate.phase[i] - reference.phase[i])
                - 2 * PREPARATION_LIFT_TURNS[node] * mp.pi
            )
            _contains(mp, lift, exact)
        for value, old, error in zip(
            candidate.conductances, reference.conductances, errors
        ):
            assert value >= 0 and abs(value - old) <= error
        assert report.status == "enclosed" and report.comparison_denominator > 0
        assert report.initial_storage_upper_bound < Q(35, 64)
        assert candidate.storage.hi < Q(35, 64)
        assert report.source.phase == reference.phase
        assert candidate.phase == tuple(
            Q(
                (on_states if candidate is actual else off_states)[-1].nodes[node][
                    "theta"
                ]
            )
            for node in candidate.nodes
        )
        reports.append(report)
    on_report, _, off_right = reports
    assert on_report.max_phase_law_forcing_upper_bound > 0
    assert off_right.max_phase_law_forcing_upper_bound == 0
    assert off_right.form_error_supremum_upper_bound > 0
    # Nominal control alone is exactly null. The perturbed control has its own
    # nonzero full-state budget and receives no coherent-family projection.
    assert len(set(candidates[2].epi[:4])) > 1
    response = nominal.finite_form_response_bounds(duration=duration)
    nominal_contrast = (
        response.endpoint_form_bounds[5] - response.endpoint_form_bounds[6]
    )
    upper = nominal_contrast.hi + 2 * (
        on_report.form_error_supremum_upper_bound
        + off_right.form_error_supremum_upper_bound
    )
    assert upper < -Q(1, 32768)


@pytest.fixture(scope="module")
def consensus_observations(contacts, matched_controls):
    """Section28 reuses six nominal laws, with no additional operator word."""
    result = []
    for contact, control in zip(contacts, matched_controls):
        attached = assess_sine_consensus_observation(
            contact[5], regions=STARS, max_rank_calls=128
        )
        disconnected_source = bound_relational_sine_frozen_isolates(
            control[0][-1], reference_model=MODEL
        )
        disconnected = assess_sine_consensus_observation(
            disconnected_source, regions=STARS, max_rank_calls=128
        )
        result.append((attached, disconnected))
    return tuple(result)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_relative_tangent_uses_full_actual_law_and_each_own_mean_metric(
    contacts, matched_controls, consensus_observations, mp, index
):
    attached, disconnected = consensus_observations[index]
    assert attached.source == contacts[index][5]
    assert attached.source.degrees == POST_DEGREES
    assert disconnected.source.degrees == (4, 2, 2, 2, 4) * 2
    assert attached.source.capacity == disconnected.source.capacity
    assert attached.region_mean_weights == (Q(84, 13), Q(51, 8))
    assert disconnected.region_mean_weights == (Q(596, 117), Q(9, 2))
    assert attached.region_mean_weights != disconnected.region_mean_weights
    weights = dict(zip(attached.source.edges, attached.source.conductances))
    assert len({weights[4, node] for node in range(5, 9)}) == 1
    assert all(Q(9, 10) < weights[edge] < 1 for edge in NEW_EDGES)
    for report, expected_dimension in ((attached, 8), (disconnected, 2)):
        generator, output, mu, masses, bare, weighted = _independent_tangent(
            report.source, STARS
        )
        assert report.rational_generator == generator
        assert report.output_rows == output
        assert report.inverse_capacity_weights == mu
        assert report.region_mean_weights == masses
        assert report.bare_laplacian == bare and report.weighted_laplacian == weighted
        assert report.linear_observation.full_state_dimension == 20
        assert report.linear_observation.dimension == expected_dimension
        assert report.max_rank_calls == report.linear_observation.max_rank_calls == 128
        assert report.linear_observation.rank_calls <= 128
        _assert_exact_realization(report.linear_observation, generator, output)
        zero, one = (Q(0),) * 10, (Q(1),) * 10
        for direction in (one + zero, zero + one):
            assert _tangent_apply(generator, direction) == (0,) * 20
            assert _tangent_apply(output, direction) == (0, 0)
            assert (
                _tangent_apply(report.linear_observation.observation, direction)
                == (0,) * expected_dimension
            )
        assert _tangent_product((mu + zero, zero + mu), generator) == ((0,) * 20,) * 2
        assert not report.source_in_consensus_family
        assert report.clock == "t"
        assert report.exchange_coefficient == report.storage_coefficient == 1
        _contains(mp, report.original_time_multiplier_bounds, 1 / mp.pi)
    assert attached.linear_observation.extra_coordinates == 6
    assert not attached.two_coordinate_closure_certified
    assert disconnected.two_coordinate_closure_certified
    assert disconnected.linear_observation.reduced_generator == ((0, 0), (0, 0))
    assert disconnected.source.epi == attached.source.epi
    assert disconnected.source.phase == attached.source.phase
    assert set(disconnected.source.edges) == set(OLD_EDGES)
    assert dict(matched_controls[index][2].nodes) == dict(contacts[index][4].nodes)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_full_fine_tangent_projects_to_five_blocks_and_retains_invisible_directions(
    contacts, consensus_observations, index
):
    report = consensus_observations[index][0]
    partition = _collective_partition(contacts[index][5])
    mu, masses = report.inverse_capacity_weights, partition.block_mean_weights
    spatial = tuple(
        tuple(mu[node] / masses[a] if node in block else Q(0) for node in range(10))
        for a, block in enumerate(COLLECTIVE_BLOCKS)
    )
    inflation = tuple(
        tuple(Q(node in block) for block in COLLECTIVE_BLOCKS) for node in range(10)
    )
    projection = tuple(row + (Q(0),) * 10 for row in spatial) + tuple(
        (Q(0),) * 10 + row for row in spatial
    )
    extension = tuple(row + (Q(0),) * 5 for row in inflation) + tuple(
        (Q(0),) * 5 + row for row in inflation
    )
    assert _tangent_product(projection, extension) == _tangent_identity(10)
    coarse = _tangent_product(
        _tangent_product(projection, report.rational_generator), extension
    )
    # Family invariance alone is insufficient: prove the arbitrary-fine-state
    # output projection and the family inflation as two separate identities.
    assert _tangent_product(projection, report.rational_generator) == _tangent_product(
        coarse, projection
    )
    assert _tangent_product(report.rational_generator, extension) == _tangent_product(
        extension, coarse
    )
    for a in range(5):
        for b in range(5):
            if a != b:
                assert coarse[a][5 + b] == partition.cross_bare_counts[a][b] / masses[a]
                assert (
                    coarse[5 + a][b]
                    == -partition.cross_conductance_sums[a][b] / masses[a]
                )
        assert coarse[a][5 + a] == -sum(coarse[a][5 + b] for b in range(5) if b != a)
        assert coarse[5 + a][a] == -sum(coarse[5 + a][b] for b in range(5) if b != a)
    lifted_projection = _tangent_product(extension, projection)
    invisible = tuple(
        tuple(Q(i == j) - lifted_projection[i][j] for j in range(20)) for i in range(20)
    )
    assert _tangent_product(projection, invisible) == ((0,) * 20,) * 10
    assert _tangent_product(report.output_rows, invisible) == ((0,) * 20,) * 2
    assert (
        _tangent_product(report.linear_observation.observation, invisible)
        == ((0,) * 20,) * 8
    )
    # An arbitrary full state need not be constant inside any block.
    fine = tuple(Q((-1) ** node * (node + 2), node + 3) for node in range(20))
    rates = _tangent_apply(report.rational_generator, fine)
    assert _tangent_apply(projection, rates) == _tangent_apply(
        coarse, _tangent_apply(projection, fine)
    )


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_zero_relative_readings_hide_contact_sensitive_internal_initialization(
    consensus_observations, index
):
    attached, disconnected = consensus_observations[index]
    mu = attached.inverse_capacity_weights
    assert mu[5] == Q(5, 4) and mu[6] == Q(3, 2)
    internal = (Q(0),) * 5 + (Q(6), Q(-5), Q(0), Q(0), Q(0))
    zero = (Q(0),) * 10
    rate_scale = sum(1 / mass for mass in attached.region_mean_weights)
    w_a = dict(zip(attached.source.edges, attached.source.conductances))[4, 5]
    for witness, expected in (
        (zero + internal, (rate_scale, Q(0))),
        (internal + zero, (Q(0), -w_a * rate_scale)),
    ):
        assert _tangent_apply(attached.output_rows, witness) == (0, 0)
        assert _tangent_apply((mu + zero, zero + mu), witness) == (0, 0)
        actual = _tangent_apply(
            attached.output_rows, _tangent_apply(attached.rational_generator, witness)
        )
        assert actual == expected and actual != (0, 0)
        assert any(_tangent_apply(attached.linear_observation.observation, witness))
    # The same vector need not have zero control reading because its metric
    # differs; nevertheless every disconnected regional reading is constant.
    assert (
        _tangent_product(disconnected.output_rows, disconnected.rational_generator)
        == ((0,) * 20,) * 2
    )


def test_declared_weight_family_has_independent_symbolic_observation_minor():
    """Check the analytic family identity, without another rank or modal scan."""
    import sympy as sym

    u, v = sym.symbols("u v")
    node_block = {
        node: index for index, block in enumerate(COLLECTIVE_BLOCKS) for node in block
    }
    counts, weighted = sym.zeros(5), sym.zeros(5)
    edge_weights = OLD_WEIGHTS + (u, u, u, u, v)
    for (left, right), weight in zip(OLD_EDGES + NEW_EDGES, edge_weights):
        a, b = node_block[left], node_block[right]
        if a == b:
            continue
        for matrix, coefficient in ((counts, 1), (weighted, weight)):
            matrix[a, a] += coefficient
            matrix[b, b] += coefficient
            matrix[a, b] -= coefficient
            matrix[b, a] -= coefficient
    capacities = CAPACITIES[:4] + (Q(117, 32),) + CAPACITIES[5:]
    mu = tuple(
        Q(degree) / capacity for degree, capacity in zip(POST_DEGREES, capacities)
    )
    masses = tuple(sum(mu[node] for node in block) for block in COLLECTIVE_BLOCKS)
    inverse_mass = sym.diag(
        *(sym.Rational(value.denominator, value.numerator) for value in masses)
    )
    a, b = inverse_mass * counts, inverse_mass * weighted
    region_masses = (sum(masses[:2]), sum(masses[2:]))
    relative = tuple(
        value / region_masses[0] if index < 2 else -value / region_masses[1]
        for index, value in enumerate(masses)
    )
    ell = sym.Matrix(
        [[sym.Rational(value.numerator, value.denominator) for value in relative]]
    )
    rows = sym.Matrix.vstack(ell, ell * b, ell * a * b, ell * b * a * b)
    determinant = rows[:, :4].det(method="domain-ge")
    polynomial = sym.Poly(
        sym.expand(determinant * sym.Rational(16633085543424, 9164597)), u, v
    )
    expected = {
        (3, 1): 147968,
        (3, 0): 1300644,
        (2, 2): -208896,
        (2, 1): -484895,
        (2, 0): 11594856,
        (1, 2): -1763766,
        (1, 1): -14815038,
        (0, 2): -330078,
    }
    assert dict(polynomial.terms()) == expected
    assert (rows * sym.ones(5, 1)).applyfunc(sym.expand) == sym.zeros(4, 1)
    # Termwise bounds over the whole closed rectangle, not sampled ranks.
    lower_u = sum(
        coefficient * (sym.Rational(9, 10) ** sum(powers) if coefficient > 0 else 1)
        for powers, coefficient in sym.Poly(polynomial.diff(u), u, v).terms()
    )
    upper_v = sum(
        coefficient * (1 if coefficient > 0 else sym.Rational(9, 10) ** sum(powers))
        for powers, coefficient in sym.Poly(polynomial.diff(v), u, v).terms()
    )
    assert lower_u == sym.Rational(798565717, 125) > 0
    assert upper_v < 0
    assert polynomial.eval({u: 1, v: sym.Rational(9, 10)}) == -sym.Rational(
        26064879, 10
    )
    # This minor stays strictly negative throughout the rectangle. These four
    # form rows and their injective one-step phase images give eight rows.


@pytest.fixture(scope="module")
def nonlinear_preparations(contacts, matched_controls, record_testsuite_property):
    """Section33's two detached preparations on each reused pair of held laws."""
    epsilon, amplitude = Q(1, 16), Q(1, 64)
    record_testsuite_property("nonlinear_consensus_epsilon", "1/16")
    record_testsuite_property("nonlinear_original_time_horizon", "1/16")
    record_testsuite_property("nonlinear_hidden_phase_amplitude", "1/64")
    record_testsuite_property(
        "nonlinear_hidden_phase_pattern",
        "both signs; nodes 6=a, 7=a, 8=-2a; all other phases and all forms zero",
    )
    record_testsuite_property(
        "nonlinear_preparation_semantics",
        "detached source on each reused held law; not an acquired successor",
    )
    records = []
    for contact, control in zip(contacts, matched_controls):
        signs = []
        for sign in (1, -1):
            laws = []
            for original, reader in (
                (contact[2][-1], bound_relational_sine_support_exchange),
                (control[0][-1], bound_relational_sine_frozen_isolates),
            ):
                graph = copy_graph_state(original)
                for node in graph:
                    graph.nodes[node]["EPI"] = 0.0
                    graph.nodes[node]["theta"] = float(
                        sign
                        * amplitude
                        * (1 if node in (6, 7) else -2 if node == 8 else 0)
                    )
                source = reader(graph, reference_model=MODEL)
                observation = assess_sine_consensus_observation(
                    source, regions=STARS, max_rank_calls=128
                )
                bound = observation.nonlinear_response_bounds(
                    duration=epsilon,
                    consensus_form=0,
                    consensus_phase=0,
                    form_amplitude_bounds=(epsilon,) * 10,
                    phase_amplitude_bounds=(epsilon,) * 10,
                )
                laws.append((source, observation, bound, graph))
            signs.append(tuple(laws))
        records.append(tuple(signs))
    return tuple(records)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_hidden_fine_phases_have_same_complete_tangent_state_and_storage_but_opposite_nonlinear_rates(
    contacts, matched_controls, nonlinear_preparations, mp, index
):
    amplitude, epsilon = Q(1, 64), Q(1, 16)
    positive, negative = nonlinear_preparations[index]
    on_positive, on_negative = positive[0], negative[0]
    assert on_positive[0].epi == on_negative[0].epi == (0,) * 10
    assert on_positive[0].phase == tuple(-value for value in on_negative[0].phase)
    assert on_positive[0].storage == on_negative[0].storage
    expected_storage = 3 * (
        2 * (1 - mp.cos(_mp(mp, amplitude))) + 1 - mp.cos(_mp(mp, 2 * amplitude))
    )
    kappa = Q(445, 1428)
    expected_rate = (
        _mp(mp, kappa)
        * (2 * mp.sin(_mp(mp, amplitude)) - mp.sin(_mp(mp, 2 * amplitude)))
        / mp.pi
    )
    assert expected_rate > 0
    for sign, laws in zip((1, -1), (positive, negative)):
        for law_index, (source, observation, bound, graph) in enumerate(laws):
            assert source.epi == (0,) * 10
            assert source.phase == (Q(0),) * 6 + (
                sign * amplitude,
                sign * amplitude,
                -2 * sign * amplitude,
                Q(0),
            )
            assert max(map(abs, source.phase)) < epsilon
            assert bound.initial_outputs == (0, 0)
            assert (
                bound.initial_observation_state
                == (0,) * observation.linear_observation.dimension
            )
            assert any(bound.initial_phase_displacements)
            assert not observation.source_in_consensus_family
            assert observation.max_rank_calls == 128
            assert source.capacity == contacts[index][5].capacity
            original = (
                contacts[index][5] if law_index == 0 else matched_controls[index][-1]
            )
            if law_index == 0:
                assert source.edges == original.edges
                assert source.conductances == original.conductances
                assert observation.linear_observation.dimension == 8
                pressure, dx, _, dtheta = _independent_field(mp, graph)
                for intervals, exacts in (
                    (source.pressure, pressure),
                    (source.form_rates, dx),
                    (source.phase_rates, dtheta),
                ):
                    for interval, exact in zip(intervals, exacts):
                        _contains(mp, interval, exact)
                ydot = sum(
                    _mp(mp, coefficient) * value
                    for coefficient, value in zip(observation.relative_mean_row, dx)
                )
                assert abs(ydot - sign * expected_rate) < mp.mpf("1e-140")
                _contains(mp, source.storage, expected_storage)
                retained = assess_sine_weighted_star_support_retention(
                    source, stars=STARS
                )
                assert retained.all_time_identity_retention_certified
            else:
                assert set(source.edges) == set(OLD_EDGES)
                assert source.conductances == OLD_WEIGHTS
                assert observation.linear_observation.dimension == 2
                assert observation.two_coordinate_closure_certified
                # Full regional edge-flux cancellation proves constant means
                # for the nonlinear law, even though fine nodes do move.
                assert any(interval.abs_max > 0 for interval in source.form_rates)
                for region in STARS:
                    assert all(
                        (left in region) == (right in region)
                        for left, right in source.edges
                    )
                    assert all(
                        observation.inverse_capacity_weights[node]
                        * source.capacity[node]
                        / source.degrees[node]
                        == 1
                        for node in region
                    )
                    summed = sum(
                        (
                            source.form_rates[node]
                            * observation.inverse_capacity_weights[node]
                            for node in region
                        ),
                        I(0),
                    )
                    assert summed.contains(0)
                _contains(mp, source.storage, 2 * expected_storage / 3)
            weights = observation.inverse_capacity_weights
            assert sum(value * x for value, x in zip(weights, source.epi)) == 0
            assert (
                sum(value * theta for value, theta in zip(weights, source.phase)) == 0
            )
            assert source.phase_rates == (I(0),) * 10
    # Neither the new zero-form preparations nor their origins overwrite the
    # old captured finite source, whose phase separation exceeds this box.
    assert contacts[index][5].phase == (0,) * 5 + (PROBES[index][1],) * 5
    assert all(value > 0 for value in contacts[index][5].epi)


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_complete_near_consensus_box_certifies_finite_output_accuracy_and_both_identities(
    contacts, consensus_observations, nonlinear_preparations, index
):
    epsilon, horizon = Q(1, 16), Q(1, 16)
    for laws in nonlinear_preparations[index]:
        source, observation, bound, _ = laws[0]
        assert bound.duration == horizon and bound.status == "enclosed"
        assert bound.consensus_form == bound.consensus_phase == 0
        assert (
            bound.form_amplitude_bounds
            == bound.phase_amplitude_bounds
            == (epsilon,) * 10
        )
        assert bound.max_form_phase_lipschitz_upper_bound < Q(16, 3)
        assert bound.max_phase_form_lipschitz_upper_bound < Q(48, 5)
        assert bound.comparison_denominator > Q(4, 5)
        assert bound.form_amplitude_supremum_upper_bound < Q(5, 3) * epsilon
        assert bound.phase_amplitude_supremum_upper_bound < 2 * epsilon
        assert bound.max_form_cubic_remainder_upper_bound < Q(256, 9) * epsilon**3
        assert bound.form_error_supremum_upper_bound < Q(20, 9) * epsilon**3
        assert bound.phase_error_supremum_upper_bound < Q(4, 3) * epsilon**3
        assert (
            bound.output_error_upper_bounds[0] < Q(40, 9) * epsilon**3 < 5 * epsilon**3
        )
        assert (
            bound.output_error_upper_bounds[1] < Q(8, 3) * epsilon**3 < 3 * epsilon**3
        )
        errors = (
            bound.whole_window_form_error_bounds + bound.whole_window_phase_error_bounds
        )
        assert bound.selected_state_error_upper_bounds == tuple(
            sum(abs(coefficient) * error for coefficient, error in zip(row, errors))
            for row in observation.linear_observation.observation
        )
        assert bound.output_error_upper_bounds == tuple(
            sum(abs(coefficient) * error for coefficient, error in zip(row, errors))
            for row in observation.output_rows
        )
        # Every edge has |dx|,|dtheta|<=2epsilon initially. This elementary
        # full-box storage bound covers all signs, not only the witness.
        box_storage = 2 * epsilon**2 * (sum(source.conductances) + len(source.edges))
        assert box_storage < 116 * epsilon**2 == Q(29, 64) < Q(35, 64)
        for region in STARS:
            control = laws[1][0]
            pairs = [
                (edge, weight)
                for edge, weight in zip(control.edges, control.conductances)
                if edge[0] in region
            ]
            assert (
                2 * epsilon**2 * (sum(weight for _, weight in pairs) + len(pairs))
                == 48 * epsilon**2
                == Q(3, 16)
                < Q(35, 64)
            )
        assert laws[1][2].status == "enclosed"
        assert laws[1][2].initial_outputs == (0, 0)
    # Actual old forms fit around their declared old mean, but their phase
    # separation does not. The new reader must reject that source transfer.
    with pytest.raises(ValueError):
        consensus_observations[index][0].nonlinear_response_bounds(
            duration=horizon,
            consensus_form=contacts[index][0],
            consensus_phase=0,
            form_amplitude_bounds=(epsilon,) * 10,
            phase_amplitude_bounds=(epsilon,) * 10,
        )


@pytest.fixture(scope="module")
def recurrence_reports(contacts, record_testsuite_property):
    """Section39's invariant families reuse the three original held laws."""
    provenance = {
        "target": "Q <= 1/4; both marked constituent identities retained",
        "form_scale": "one existing form unit",
        "phase_scale": "one radian; one common circular origin",
        "energy_ceiling": "3/4",
        "own_law_form_mean_slab": "[-1,1]",
        "full_state_box_radius": "2^-20 in each of ten forms and ten radians",
        "measure_scope": "ambient form Lebesgue and circle Haar; not nominal surface",
        "individual_recurrence": "unavailable for every selected captured point",
    }
    for key, value in provenance.items():
        record_testsuite_property("permanent_registration_" + key, value)
    return tuple(
        contact[5].recurrence_bounds(energy_ceiling=Q(3, 4), form_mean_bounds=(-1, 1))
        for contact in contacts
    )


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_actual_held_law_has_compact_recurrence_family_without_a_pointwise_verdict(
    contacts, recurrence_reports, mp, index
):
    mean, _, _, _, _, source, retained = contacts[index]
    report = recurrence_reports[index]
    assert report.source == source
    assert report.energy_ceiling == Q(3, 4)
    assert report.form_mean_bounds == (-1, 1)
    weights = tuple(Q(degree) / nu for degree, nu in zip(POST_DEGREES, source.capacity))
    total = sum(weights)
    assert total == Q(1335, 104)
    assert report.form_mean_weights == weights
    assert report.normalized_mean_weights == tuple(weight / total for weight in weights)
    expected_mean = mean + (1014 * AMPLITUDE + 256 * BOOST) / 1335
    assert report.weighted_form_mean == expected_mean
    assert (
        expected_mean
        == sum(weight * form for weight, form in zip(weights, source.epi)) / total
    )
    assert -1 < expected_mean < 1
    assert report.weighted_mean_rate_residual_bounds.contains(0)
    assert report.divergence == 0
    assert all(weight > 0 for weight in source.conductances)
    assert nx.is_connected(nx.Graph(source.edges))
    assert report.positive_conductance_edge_indices == tuple(range(19))
    assert report.positive_conductance_weight_floor == min(source.conductances)
    assert report.path_length_upper_bound == len(source.nodes) - 1 == 9
    radius = mp.sqrt(2 * 9 * _mp(mp, Q(3, 4) / min(source.conductances)))
    _contains(mp, report.form_radius_bounds, radius)
    for form in source.epi:
        assert abs(_mp(mp, form - expected_mean)) <= radius
        assert report.form_coordinate_bounds.contains(form)
    assert report.form_coordinate_bounds == I(
        -1 - report.form_radius_bounds.hi, 1 + report.form_radius_bounds.hi
    )
    assert report.captured_energy_exact is None
    assert report.captured_storage_bounds == source.storage
    assert source.storage.hi < Q(35, 64) < report.energy_ceiling
    assert report.snapshot_family_membership == "inside"
    assert report.snapshot_membership_reasons == ()
    assert retained.all_time_identity_retention_certified
    assert report.almost_everywhere_recurrence_certified
    assert report.finite_positive_family_volume_certified
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
    assert (
        "no_transfer_to_exact_energy_fixed_mean_or_other_lower_dimensional_preparations"
        in report.scope
    )


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_full_dimensional_box_is_inside_invariant_family_but_outside_circular_target(
    contacts, perturbation_reports, recurrence_reports, index
):
    _, alpha, _, _, _, source, _ = contacts[index]
    report = recurrence_reports[index]
    box = perturbation_reports[index][0]
    zeta = Q(1, 2**20)
    assert box.source == source
    assert box.form_error_bounds == box.phase_error_bounds == (zeta,) * 10
    assert source.phase == (0,) * 5 + (alpha,) * 5
    assert Q(3, 8) <= alpha <= Q(2, 5) < pi_interval().lo

    # Five disjoint cross pairs have circular distance alpha. For every
    # common origin phi, d_i(phi)^2+d_j(phi)^2 >= alpha^2/2 by the circle
    # triangle inequality. The midpoint alpha/2 attains the summed lower
    # bound here. This establishes the circle minimum for this source;
    # arbitrary real-lift variance is not used as a global phase metric.
    pair_gaps = tuple(source.phase[i + 5] - source.phase[i] for i in range(5))
    phase_cost = sum(gap**2 / 2 for gap in pair_gaps)
    midpoint_cost = sum((phase - alpha / 2) ** 2 for phase in source.phase)
    assert phase_cost == midpoint_cost == Q(5, 2) * alpha**2
    assert phase_cost >= Q(45, 128) > Q(1, 4)
    arithmetic_mean = sum(source.epi) / 10
    form_cost = sum((form - arithmetic_mean) ** 2 for form in source.epi)
    assert form_cost >= 0
    assert form_cost + phase_cost > Q(1, 4)

    # The circular triangle inequality retains at least alpha-2*zeta in
    # every pair throughout the independent initial box. This is the same
    # lower bound as (sqrt(5/2)*alpha-sqrt(10)*zeta)^2, without rounding.
    assert alpha > 2 * zeta
    phase_box_lower_bound = sum((gap - 2 * zeta) ** 2 / 2 for gap in pair_gaps)
    assert phase_box_lower_bound == Q(5, 2) * (alpha - 2 * zeta) ** 2
    assert phase_box_lower_bound >= Q(5, 2) * (Q(3, 8) - 2 * zeta) ** 2 > Q(1, 4)

    # A positive normalized mean has error at most zeta for every choice
    # of signs. The independent full-edge budget admits the entire box,
    # including all source correlations rather than a selected corner.
    assert sum(report.normalized_mean_weights) == 1
    assert all(weight > 0 for weight in report.normalized_mean_weights)
    assert -1 < report.weighted_form_mean - zeta
    assert report.weighted_form_mean + zeta < 1
    assert box.initial_storage_upper_bound < Q(35, 64) < report.energy_ceiling
    for star in STARS:
        center = star[0]
        for leaf in star[1:4]:
            assert abs(source.epi[center] - source.epi[leaf]) + 2 * zeta < Q(1, 2)
            assert (
                abs(source.phase[center] - source.phase[leaf]) + 2 * zeta
                < pi_interval().lo / 2
            )
    assert 0 < 2 * zeta < 2 * pi_interval().lo
    assert (2 * zeta) ** 20 > 0
    # Its open interior has positive ambient volume. The recurrence
    # theorem concerns almost every member, not any selected nominal point
    # or the lower-dimensional two-parameter preparation surface.
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"


@pytest.fixture(scope="module")
def finite_registration_static_sources(
    contacts, matched_controls, perturbation_reports, record_testsuite_property
):
    """Reuse the sources fixed by the finite-identity owner's section1."""
    provenance = {
        "target": "Q <= 1/4 with one common circular phase center",
        "form_scale": "one existing form unit",
        "phase_scale": "one radian",
        "full_state_box_radius": "2^-20 per form and radian coordinate",
        "joined_claim": "same displayed storage and two lifted means; not reachability",
        "disconnected_claim": "circular target excluded by own component invariants",
        "tangent_claim": "separate real-lift law and its own quadratic storage",
        "execution": "no new word, source pattern, trajectory or episode timing",
    }
    for key, value in provenance.items():
        record_testsuite_property("finite_registration_admission_" + key, value)
    return tuple(zip(contacts, matched_controls, perturbation_reports, strict=True))


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_disconnected_component_invariants_exclude_the_common_circular_target(
    finite_registration_static_sources, index
):
    contact, control, boxes = finite_registration_static_sources[index]
    alpha, zeta = contact[1], Q(1, 2**20)
    components = control[-1]
    normalized = []
    phase_means = []
    for component, box in zip(components, boxes[1], strict=True):
        assert box.source == component
        weights = tuple(
            Q(degree) / nu
            for degree, nu in zip(component.degrees, component.capacity, strict=True)
        )
        probabilities = tuple(weight / sum(weights) for weight in weights)
        assert sum(probabilities) == 1
        assert all(weight > 0 for weight in probabilities)
        normalized.append(probabilities)
        phase_means.append(
            sum(weight * phase for weight, phase in zip(probabilities, component.phase))
        )
        # Both full rows cancel with the actual inverse-mobility metric.
        # The phase numerator uses the held conductance gradient, not the
        # bare sine incidence or the joined graph's degrees.
        assert (
            sum(
                weight * rate
                for weight, rate in zip(weights, component.phase_rate_numerators())
            )
            == 0
        )
        assert sum(
            (weight * rate for weight, rate in zip(weights, component.form_rates)),
            I(0),
        ).contains(0)
        assert box.form_error_bounds == box.phase_error_bounds == (zeta,) * 5
        assert component.storage.hi <= box.initial_storage_upper_bound < Q(1, 32)
        assert 1 - box.initial_storage_upper_bound > 0
        assert all(
            abs(component.phase[i] - component.phase[j]) + 2 * zeta
            < pi_interval().lo / 2
            for i, j in (
                (component.nodes.index(left), component.nodes.index(right))
                for left, right in component.edges
            )
        )
        assert nx.is_connected(nx.Graph(component.edges))
    assert normalized[0] == (Q(117, 596),) * 4 + (Q(32, 149),)
    assert normalized[1] == (Q(2, 9),) * 4 + (Q(1, 9),)
    assert phase_means == [Q(0), alpha]
    relative_mean_row = tuple(-weight for weight in normalized[0]) + normalized[1]
    assert sum(relative_mean_row) == 0
    cauchy_coefficient = sum(weight**2 for weight in relative_mean_row)
    assert cauchy_coefficient == Q(17785, 88804) + Q(17, 81) < Q(4, 9)

    # At any hypothetical circular cost J<=1, closest common-center lifts
    # have |delta_i|<=1. Along each protected edge their discrepancy from
    # the continuous acute lifts is <pi/2+2<2pi. Hence its integer-turn
    # difference is zero; connectedness leaves one integer per component.
    # This is the required lift association, not a raw-variance shortcut.
    assert pi_interval().hi / 2 + 2 < 2 * pi_interval().lo
    assert 0 < alpha - 2 * zeta < alpha + 2 * zeta < pi_interval().lo
    phase_floor = (alpha - 2 * zeta) ** 2 / cauchy_coefficient
    assert phase_floor > Q(9, 4) * (Q(3, 8) - 2 * zeta) ** 2 > Q(1, 4)
    assert phase_floor < 1
    # Cauchy therefore gives J>=phase_floor when J<=1. If J>1, the same
    # bound holds immediately. Conserved component means and the uniform
    # initial energy budget make this an all-time obstruction on the full
    # independent source box, without selecting or computing a trajectory.


@pytest.mark.parametrize("index", range(3), ids=("central", "lower", "upper"))
def test_joined_displayed_invariants_and_tangent_energy_admit_correlated_target(
    finite_registration_static_sources, mp, index
):
    contact, _, boxes = finite_registration_static_sources[index]
    _, alpha, _, _, _, source, _ = contact
    zeta = Q(1, 2**20)
    weights = tuple(
        Q(degree) / nu
        for degree, nu in zip(source.degrees, source.capacity, strict=True)
    )
    mass = sum(weights)
    form_mean = sum(weight * form for weight, form in zip(weights, source.epi)) / mass
    phase_mean = (
        sum(weight * phase for weight, phase in zip(weights, source.phase)) / mass
    )
    assert phase_mean == Q(221, 445) * alpha
    q = tuple(Q(node == 4) - weights[4] / mass for node in source.nodes)
    assert sum(weight * value for weight, value in zip(weights, q)) == 0
    arithmetic_q_mean = sum(q) / len(q)
    centered_q_square = sum((value - arithmetic_q_mean) ** 2 for value in q)
    assert centered_q_square == Q(9, 10)
    quadratic_q = sum(
        weight * (q[left] - q[right]) ** 2 / 2
        for (left, right), weight in zip(source.edges, source.conductances)
    )
    strengths = sum(
        weight for edge, weight in zip(source.edges, source.conductances) if 4 in edge
    )
    actual_weights = dict(zip(source.edges, source.conductances))
    hub_neighbors = {
        right if left == 4 else left
        for left, right in source.edges
        if 4 in (left, right)
    }
    assert hub_neighbors == set(source.nodes) - {4}
    assert len(hub_neighbors) == 9
    assert all(weight > 0 for weight in source.conductances)
    assert all(
        abs(source.phase[node] - source.phase[4]) + 2 * zeta < pi_interval().lo / 3
        for node in hub_neighbors
    )
    assert strengths == 8 + 4 * actual_weights[4, 5] + actual_weights[4, 9]
    assert strengths > Q(25, 2)
    assert quadratic_q == strengths / 2
    assert (2 / strengths) * quadratic_q == 1

    # For the exact (unrounded) source energy H, r^2=2H/s4 and
    # x*=m_mu+r*q, theta*=m_theta conserve the displayed storage and means.
    # The source-energy interval is propagated only through this correlated
    # formula; its endpoints do not define independent target coordinates.
    q_coefficient = (2 / strengths) * centered_q_square
    assert q_coefficient == 9 / (5 * strengths)
    target_cost = q_coefficient * source.storage
    assert 0 < target_cost.lo <= target_cost.hi < Q(81, 1250) < Q(1, 4)
    assert source.storage.hi < Q(9, 20)
    assert boxes[0].initial_storage_upper_bound < Q(1, 2)
    # Conservation and each hub edge's bare phase cost prevent a first
    # crossing of +/-pi/3. The retained common lifts therefore have span
    # <2*pi/3<pi for every box source, which admits the circle-cost lemma.
    # This is an energy/branch implication, not a sampled phase history.
    assert 2 * pi_interval().hi / 3 < pi_interval().lo
    assert q_coefficient * boxes[0].initial_storage_upper_bound < Q(9, 125) < Q(1, 4)
    for star in STARS:
        assert all(q[star[0]] == q[leaf] for leaf in star[1:4])
    for block in COLLECTIVE_BLOCKS:
        assert len({q[node] for node in block}) == 1
        assert len({source.epi[node] for node in block}) == 1
        assert len({source.phase[node] for node in block}) == 1

    # Uniform algebraic path positivity, without sampling a proposed path:
    # v(s)=(1-s)(x-m_mu)+s*q has v4-v9=(1-s)(x4-x9)+s.
    # Every initial box member has x4-x9>=b-2*zeta>0, so this retained
    # weighted edge alone keeps the denominator F(v(s)) strictly positive.
    assert source.epi[4] - source.epi[9] == BOOST
    assert 0 < BOOST - 2 * zeta <= BOOST + 2 * zeta < 1
    path_form_floor = actual_weights[4, 9] * (BOOST - 2 * zeta) ** 2 / 2
    assert path_form_floor > 0
    assert (
        sum(weight * (form - form_mean) for weight, form in zip(weights, source.epi))
        == 0
    )
    for left, right in source.edges:
        assert (
            abs(source.phase[right] - source.phase[left]) + 2 * zeta
            < pi_interval().lo / 2
        )
    # Scaling those admitted acute gaps toward zero cannot increase their
    # phase energy. Thus H-V(theta(s)) stays >=F(x0)>0. Normalizing v(s)
    # by that correlated energy gives a configuration path, not a flow.

    fine_form_energy = sum(
        weight * (source.epi[left] - source.epi[right]) ** 2 / 2
        for (left, right), weight in zip(source.edges, source.conductances)
    )
    tangent_phase_energy = sum(
        (source.phase[right] - source.phase[left]) ** 2 / 2
        for left, right in source.edges
    )
    assert fine_form_energy == source.form_storage < Q(1, 25)
    assert tangent_phase_energy == Q(5, 2) * alpha**2
    tangent_energy = fine_form_energy + tangent_phase_energy
    assert tangent_energy < Q(11, 25)
    tangent_box_energy = sum(
        weight * (abs(source.epi[left] - source.epi[right]) + 2 * zeta) ** 2 / 2
        + (abs(source.phase[right] - source.phase[left]) + 2 * zeta) ** 2 / 2
        for (left, right), weight in zip(source.edges, source.conductances)
    )
    assert tangent_energy < tangent_box_energy < Q(49, 100) < Q(1, 2)
    assert q_coefficient * tangent_box_energy < Q(9, 125) < Q(1, 4)
    # The tangent uses its own quadratic edge energy. Each hub gap has
    # square/2 below 1/2 for all time, hence its full lift span is <2<pi;
    # the same circular observation is valid without transferring H.
    assert 2 < pi_interval().lo
    assert q_coefficient * tangent_energy < Q(198, 3125) < Q(1, 4)
    nonlinear_exact = _mp(mp, fine_form_energy) + 5 * (1 - mp.cos(_mp(mp, alpha)))
    _contains(mp, source.storage, nonlinear_exact)
    assert _mp(mp, tangent_energy) > nonlinear_exact
    # The full tangent has its own larger quadratic energy; the nonlinear
    # energy is not transferred. Neither construction proves that these
    # are all invariants, a reached state, or a finite-lived collective NFR.
