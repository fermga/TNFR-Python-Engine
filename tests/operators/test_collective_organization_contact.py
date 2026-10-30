"""Frozen actual-UM contact between two separately maintained organizations.

SINE_ORGANIZATION_CONTACT section 1 fixes both sources, complete word and
static discriminators before response evaluation. These controls compare
the executed reset to independent 100-digit edge/consensus arithmetic and
check analytical identity bounds; they never integrate a trajectory.

SINE_PROXIMITY_CONTACT section 1 separately freezes the same source/word
with candidate count three. Its independent controls retain the actual
five-edge attachment and the full twenty-node maintenance identity.

Section 10 separately declares eight independent relative-origin controls;
they reuse the same word and assertions on their actual shifted inputs.
Section 17 reuses the opposite-phase pair for a finite-exchange proof's
static ingredients and an independently admitted disconnected symmetry null.
"""

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as Q
from functools import lru_cache
from math import pi

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.glyph_history import push_glyph
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import (
    I,
    cos,
    exp_neg_unit,
    pi_interval,
    sin,
    sqrt,
)
from tnfr.mathematics.krylov import exact_rank
from tnfr.operators import network_stage
from tnfr.operators.definitions import Coupling, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_scale import assess_sine_replica_capacity
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.types import require_finite_real_scalar_epi

_A = tuple(range(10))
_B = tuple(range(10, 20))
_TARGETS = (0, 1)
_NEIGHBORS = (2, 3, 8, 9)
_RECEIVERS = (10, 11, 12, 13, 18, 19)
_NEW_EDGES = {(0, 1)} | {(left, right) for left in _TARGETS for right in _RECEIVERS}
_PROXIMITY_EDGES = {(0, 1)} | {(left, right) for left in _TARGETS for right in (10, 11)}
_MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
_TWIST, _FORM = Q(674651885, 536870912), Q(5, 16)
_BOOST, _PUSH = Q(1, 1024), Q(1, 1024)
_NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_ORIGIN = "2029-01-01T00:00:00+00:00"
_TOLERANCE = Q(1, 2**46)


def _source(*, split):
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    for offset in (0, 10):
        pairs = tuple((offset + 2 * k, offset + 2 * k + 1) for k in range(5))
        for index, pair in enumerate(pairs):
            graph.add_edges_from(
                (left, right, {"weight": 1, "length": 1})
                for left in pair
                for right in pairs[(index + 1) % 5]
            )
            u = Q(1, 8192) if split and offset == index == 0 else Q(0)
            delta = Q(1, 32768) if split and offset == index == 0 else Q(0)
            for sign, node in zip((1, -1), pair):
                graph.nodes[node].update(
                    EPI=float(_FORM + sign * u),
                    theta=float(index * _TWIST + sign * delta),
                    nu_f=1.0,
                    delta_nfr=sign / 16,
                    Si=0.5,
                    glyph_history=[],
                )
    graph.graph.update(
        _t=7,
        RANDOM_SEED=17,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_LATENT_MAX=1,
        VF_BASAL_THRESHOLD=1 / 8,
        SHA_MIN_VF=1 / 8,
        UM_MIN_EPI=1 / 8,
        UM_MIN_VF=1 / 8,
        DELTA_PHI_MAX=pi / 2,
        UM_MAX_PHASE_DIFF=pi / 2,
        MIN_NETWORK_DEGREE_COUPLING=1,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=3 / 4,
        UM_CANDIDATE_COUNT=0,
        UM_CANDIDATE_MODE="proximity",
        EPI_MIN=-1,
        EPI_MAX=1,
        CLIP_MODE="hard",
        GLYPH_HYSTERESIS_WINDOW=8,
        GLYPH_FACTORS={
            "AL_boost": float(_BOOST),
            "UM_theta_push": float(_PUSH),
            "UM_vf_sync": 1 / 8,
            "UM_dnfr_reduction": 1 / 8,
            "SHA_vf_factor": 1 / 2,
        },
    )
    for _ in range(2):
        push_glyph(graph.nodes[1], "AL", window=8)
    graph.nodes[1].update(
        emission_timestamp=_ORIGIN,
        _emission_activated=True,
        _emission_origin=_ORIGIN,
        _structural_lineage={
            "origin": _ORIGIN,
            "activation_count": 2,
            "derived_nodes": [],
            "parent_emission": None,
        },
    )
    return graph


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is None else _NOW.astimezone(tz)


def _detached(graph):
    result = nx.Graph()
    result.graph["GAMMA"] = deepcopy(graph.graph["GAMMA"])
    result.add_nodes_from(
        (node, deepcopy(data)) for node, data in graph.nodes(data=True)
    )
    result.add_edges_from(
        (left, right, deepcopy(data)) for left, right, data in graph.edges(data=True)
    )
    return result


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("the organization-contact protocol permits no trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _execute_contact_word(graph):
    operators = (Emission(), Coupling(), Silence())
    context = {"compatibility_profile": "core"}
    admission = validate_sequence(
        [operator.name for operator in operators], context=context
    )
    assert admission.passed
    sequence = ValidatedSequence(operators, context=context)
    states, results = [_detached(graph)], []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        for index, operator in enumerate(operators):
            results.append(
                execute_network_operator_stage(
                    graph, operator, _TARGETS, sequence_context=sequence.step(index)
                )
            )
            states.append(_detached(graph))
    return tuple(states), tuple(results), graph


@pytest.fixture(scope="module", params=(True, False), ids=("split", "tips"))
def contact(request):
    return request.param, *_execute_contact_word(_source(split=request.param))


@pytest.fixture(scope="module", params=(True, False), ids=("split", "tips"))
def count3_contact(request):
    graph = _source(split=request.param)
    graph.graph["UM_CANDIDATE_COUNT"] = 3
    return request.param, *_execute_contact_word(graph)


@pytest.fixture(scope="module")
def count3_continuation(count3_contact):
    _, (_, _, _, final), _, _ = count3_contact
    return bound_relational_sine_support_exchange(final, reference_model=_MODEL)


def _registered_preparation(signs):
    form_offset, phase_offset = (sign * Q(3, 4096) for sign in signs)
    graph = _source(split=True)
    graph.graph["UM_CANDIDATE_COUNT"] = 3
    for node in _B:
        data = graph.nodes[node]
        data["EPI"] = float(Q(data["EPI"]) + form_offset)
        data["theta"] = float(Q(data["theta"]) + phase_offset)
    contact = (True, *_execute_contact_word(graph))
    report = bound_relational_sine_support_exchange(
        contact[1][-1], reference_model=_MODEL
    )
    return form_offset, phase_offset, contact, report


@pytest.fixture(scope="module")
def registered_preparations():
    return lru_cache(maxsize=8)(_registered_preparation)


@pytest.fixture(
    scope="module",
    params=((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)),
    ids=("++", "+-", "-+", "--", "+form", "-form", "+phase", "-phase"),
)
def registered_contact(request, registered_preparations):
    return registered_preparations(request.param)


@pytest.fixture(scope="module")
def opposite_phase_contacts(registered_preparations):
    return registered_preparations((0, 1)), registered_preparations((0, -1))


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _form(graph, node):
    return Q(require_finite_real_scalar_epi(graph.nodes[node]["EPI"], "contact form"))


def _angle(mp, value):
    return mp.atan2(mp.sin(value), mp.cos(value))


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _reset_oracle(mp, source):
    phases = {node: _mp(mp, data["theta"]) for node, data in source.nodes(data=True)}
    consensus = {
        node: mp.arg(sum(mp.exp(mp.j * phases[other]) for other in (node, *_NEIGHBORS)))
        for node in _TARGETS
    }
    final = dict(phases)
    for node in (*_TARGETS, *_NEIGHBORS):
        writers = (node,) if node in _TARGETS else _TARGETS
        final[node] += (
            _mp(mp, _PUSH)
            * sum(_angle(mp, consensus[writer] - phases[node]) for writer in writers)
            / len(writers)
        )
    candidates = {
        node: {
            other
            for other in source
            if other != node
            and not source.has_edge(node, other)
            and abs(_angle(mp, phases[other] - phases[node])) <= mp.pi / 2
        }
        for node in _TARGETS
    }
    forms = {
        node: _form(source, node) + (_BOOST if node in _TARGETS else 0)
        for node in source
    }
    weights = {}
    for node, selected in candidates.items():
        for other in selected:
            phase_gap = abs(_angle(mp, final[node] - final[other]))
            weight = (
                1
                - phase_gap / (2 * mp.pi)
                - _mp(mp, abs(forms[node] - forms[other]))
                / (4 * _mp(mp, abs(forms[node]) + abs(forms[other]) + Q(1e-9)))
            )
            assert phase_gap < mp.pi / 2 and weight > mp.mpf(3) / 4
            weights[tuple(sorted((node, other)))] = weight
    pressures = {}
    for node in _TARGETS:
        affinity = (
            sum(
                1 - abs(_angle(mp, final[node] - final[other])) / mp.pi
                for other in _NEIGHBORS
            )
            / 4
        )
        pressures[node] = _mp(mp, source.nodes[node]["delta_nfr"]) * (1 - affinity / 8)
    return candidates, final, weights, pressures


def test_sources_are_separately_maintained_and_union_is_not_admitted(contact):
    _, (source, _, _, _), _, _ = contact
    assert {frozenset(nodes) for nodes in nx.connected_components(source)} == {
        frozenset(_A),
        frozenset(_B),
    }
    for component in (_A, _B):
        graph = source.subgraph(component).copy()
        report = bound_relational_sine_exchange(graph, reference_model=_MODEL)
        assert report.degrees == (4,) * 10
        pairs = tuple(
            (component[0] + 2 * k, component[0] + 2 * k + 1) for k in range(5)
        )
        barrier = assess_sine_replica_capacity(
            graph,
            reference_model=_MODEL,
            pairs=pairs,
            radius=Q(1, 16),
            excess_ceiling=Q(1, 16384),
            form_mean_bounds=(Q(1, 4), Q(1, 2)),
        )
        assert barrier.family.source_set_trapping_certified
        assert barrier.family.norm_squared_upper_bound < Q(1, 512) ** 2
    with pytest.raises(ValueError, match="connected"):
        bound_relational_sine_support_exchange(source, reference_model=_MODEL)


def test_actual_word_preserves_context_and_matches_independent_contact_oracle(
    contact, mp
):
    split, (source, _, _, final), results, runtime = contact
    candidates, phases, weights, pressures = _reset_oracle(mp, source)
    assert all(
        result.schedule == TWO_PHASE_JACOBI and result.nodes_processed == 2
        for result in results
    )
    assert runtime.graph["_t"] == 7 and runtime.graph["RANDOM_SEED"] == 17
    assert "_trig_version" not in runtime.graph
    assert runtime.graph["_edge_version"] == len(_NEW_EDGES)
    assert candidates == {node: {1 - node, *_RECEIVERS} for node in _TARGETS}
    assert set(weights) == _NEW_EDGES
    assert set(final.edges) - set(source.edges) == _NEW_EDGES
    assert nx.is_connected(final)
    for edge in source.edges:
        assert final.edges[edge] == source.edges[edge]
    for edge, expected in weights.items():
        assert set(final.edges[edge]) == {"weight"}
        assert abs(_mp(mp, final.edges[edge]["weight"]) - expected) < _mp(
            mp, _TOLERANCE
        )
        assert 3 / 4 < final.edges[edge]["weight"] <= 1
    assert (final.edges[0, 1]["weight"] < 1) is split
    for node in final:
        data = final.nodes[node]
        assert abs(_angle(mp, _mp(mp, data["theta"]) - phases[node])) < _mp(
            mp, _TOLERANCE
        )
        if node not in _TARGETS:
            assert _form(final, node) == _form(source, node)
            assert data["nu_f"] == source.nodes[node]["nu_f"]
            assert data["delta_nfr"] == source.nodes[node]["delta_nfr"]
            assert data["glyph_history"] == []
            assert "_operator_step" not in data
            if node in _NEIGHBORS:
                assert data["theta"] != source.nodes[node]["theta"]
            else:
                assert data == source.nodes[node]
            continue
        assert _form(final, node) == _form(source, node) + _BOOST
        assert data["nu_f"] == 0.5
        assert abs(_mp(mp, data["delta_nfr"]) - pressures[node]) < _mp(mp, _TOLERANCE)
        assert tuple(data["glyph_history"]) == ("AL",) * (2 * node) + (
            "AL",
            "UM",
            "SHA",
        )
        assert data["_operator_step"] == 2 * node + 3
        assert data["_grammar_u2_debt"] == 0
        assert data["_structural_lineage"]["activation_count"] == 2 * node + 1
        assert data["emission_timestamp"] == (_ORIGIN if node else _NOW.isoformat())
        assert data["latent"] is True
        assert data["preserved_epi"] == float(_form(final, node))
        assert data["latency_start_time"] == _NOW.isoformat()
    assert all(final.nodes[node] == source.nodes[node] for node in _B)


def _storage(mp, graph, edges):
    return sum(
        _mp(
            mp,
            Q(graph.edges[left, right]["weight"])
            * (_form(graph, left) - _form(graph, right)) ** 2
            / 2,
        )
        + 1
        - mp.cos(
            _mp(mp, graph.nodes[right]["theta"]) - _mp(mp, graph.nodes[left]["theta"])
        )
        for left, right in edges
    )


def test_actual_weighted_rows_and_full_hybrid_work_use_all_created_edges(contact, mp):
    _, (source, emitted, coupled, final), _, _ = contact
    report = bound_relational_sine_support_exchange(final, reference_model=_MODEL)
    assert report.degrees == tuple(
        11 if node in _TARGETS else 6 if node in _RECEIVERS else 4 for node in final
    )
    assert report.capacity == (Q(1, 2),) * 2 + (Q(1),) * 18
    with pytest.raises(ValueError, match="unit conductances"):
        bound_relational_sine_exchange(final, reference_model=_MODEL)
    for node in final:
        gradient = sum(
            (
                Q(final.edges[node, other]["weight"])
                * (_form(final, node) - _form(final, other))
                for other in final[node]
            ),
            Q(0),
        )
        current = sum(
            mp.sin(
                _mp(mp, final.nodes[other]["theta"])
                - _mp(mp, final.nodes[node]["theta"])
            )
            for other in final[node]
        )
        mobility = _mp(mp, Q(final.nodes[node]["nu_f"])) / (mp.pi * final.degree(node))
        assert report.form_gradient[node] == gradient
        _contains(mp, report.form_rates[node], mobility * current)
        _contains(mp, report.phase_rates[node], mobility * _mp(mp, gradient))
        assert report.node_balance_residual[node].contains(0)
    old = tuple(source.edges)
    al_work = _storage(mp, emitted, old) - _storage(mp, source, old)
    pair_form = tuple(
        (_form(source, 2 * index) + _form(source, 2 * index + 1)) / 2
        for index in range(5)
    )
    expected_al = (
        4 * _BOOST * (2 * pair_form[0] - pair_form[1] - pair_form[4]) + 4 * _BOOST**2
    )
    assert abs(al_work - _mp(mp, expected_al)) < mp.mpf("1e-90")
    phase_work = sum(
        mp.cos(
            _mp(mp, emitted.nodes[right]["theta"])
            - _mp(mp, emitted.nodes[left]["theta"])
        )
        - mp.cos(
            _mp(mp, coupled.nodes[right]["theta"])
            - _mp(mp, coupled.nodes[left]["theta"])
        )
        for left, right in old
    )
    added_work = _storage(mp, coupled, _NEW_EDGES)
    total = _storage(mp, final, tuple(final.edges)) - _storage(mp, source, old)
    assert added_work > 0
    assert abs(total - (al_work + phase_work + added_work)) < mp.mpf("1e-90")
    assert _storage(mp, final, tuple(final.edges)) == _storage(
        mp, coupled, tuple(coupled.edges)
    )
    _contains(mp, report.storage, _storage(mp, final, tuple(final.edges)))
    assert report.continuous_loss == 0
    assert report.storage_rate.contains(0)
    assert report.balance_residual.contains(0)


def test_boundary_association_is_needed_and_joint_relabeling_is_covariant(contact, mp):
    split, (_, _, _, final), _, _ = contact
    original = bound_relational_sine_support_exchange(final, reference_model=_MODEL)
    state_swap = deepcopy(final)
    for node in _TARGETS:
        state_swap.nodes[node].clear()
        state_swap.nodes[node].update(deepcopy(final.nodes[1 - node]))
    changed = bound_relational_sine_support_exchange(state_swap, reference_model=_MODEL)
    assert dict(state_swap.edges) == dict(final.edges)

    # The old unordered pair coordinates are invariant under this state-only
    # swap, including the signed form/phase correlation Q=u*sin(delta).
    def coordinates(graph):
        x0, x1 = (_form(graph, node) for node in _TARGETS)
        phases = tuple(
            _angle(mp, _mp(mp, graph.nodes[node]["theta"])) for node in _TARGETS
        )
        u, delta = _mp(mp, (x0 - x1) / 2), (phases[0] - phases[1]) / 2
        return ((x0 + x1) / 2, sum(phases) / 2, mp.cos(delta), u**2, u * mp.sin(delta))

    assert coordinates(state_swap) == coordinates(final)
    u = (_form(final, 0) - _form(final, 1)) / 2
    row_contrast = sum(
        (
            Q(final.edges[0, node]["weight"]) - Q(final.edges[1, node]["weight"])
            for node in (10, 11)
        ),
        Q(0),
    )
    predicted = u * row_contrast / 6
    original_mean = sum(original.phase_rate_numerators()[node] for node in (10, 11)) / 2
    changed_mean = sum(changed.phase_rate_numerators()[node] for node in (10, 11)) / 2
    assert changed_mean - original_mean == predicted
    assert (predicted != 0) is split
    mapping = {node: 1 - node if node in _TARGETS else node for node in final}
    relabeled = nx.relabel_nodes(final, mapping, copy=True)
    covariant = bound_relational_sine_support_exchange(
        relabeled, reference_model=_MODEL
    )
    positions = {node: index for index, node in enumerate(covariant.nodes)}
    for index, node in enumerate(original.nodes):
        transported = positions[mapping[node]]
        for name in (
            "epi",
            "phase",
            "capacity",
            "degrees",
            "form_gradient",
            "form_rates",
            "phase_rates",
        ):
            assert (
                getattr(covariant, name)[transported] == getattr(original, name)[index]
            )
        assert relabeled.nodes[mapping[node]] == final.nodes[node]
    assert covariant.storage == original.storage


def test_finite_retention_and_later_receiver_exit_have_strict_analytic_margins():
    z, radius, short, late = Q(1, 512), Q(1, 16), Q(1, 64), Q(3, 8)
    a0, form_range = 5 * z / 3, 5 * z / 2
    circular_pi = pi_interval()
    displacement = (
        sqrt(I(10))
        * short
        / circular_pi
        * sqrt(1 + (form_range + short / circular_pi) ** 2)
    )
    assert (13 * z / 4 + displacement).hi < radius
    assert (z + displacement).hi < radius
    # Bound the exact mathematical twist, not its dyadic test representation.
    assert sin(2 * circular_pi / 5).lo > Q(19, 20)
    assert circular_pi.lo > 3 and circular_pi.hi < Q(22, 7)
    receiver_upper = (
        z
        - 2 * Q(19, 20) * late / (3 * Q(22, 7))
        + 4 * a0 * late / 3
        + 2 * form_range * late**2 / 3**2
        + 4 * late**3 / (3 * 3**3)
    )
    assert receiver_upper < -radius
    # The receiver contrast is a centered linear observation of unit norm;
    # exceeding radius in this coordinate forces exit of the full identity.
    coefficients = (
        Q(0),
        Q(0),
        Q(1, 2),
        Q(1, 2),
        Q(0),
        Q(0),
        Q(0),
        Q(0),
        Q(-1, 2),
        Q(-1, 2),
    )
    assert sum(coefficients) == 0
    assert sum(coefficient**2 for coefficient in coefficients) == 1


def _assert_proximity_word_selects_three_nearest_and_preserves_actual_resets(
    count3_contact, mp
):
    split, (source, emitted, coupled, final), results, runtime = count3_contact
    candidates, phases, all_weights, pressures = _reset_oracle(mp, source)
    selected = {}
    for node in _TARGETS:
        gaps = {
            other: abs(
                _angle(
                    mp,
                    _mp(mp, source.nodes[other]["theta"])
                    - _mp(mp, source.nodes[node]["theta"]),
                )
            )
            for other in candidates[node]
        }
        selected[node] = set(sorted(gaps, key=lambda other: (gaps[other], other))[:3])
        excluded = candidates[node] - selected[node]
        assert max(gaps[other] for other in selected[node]) < min(
            gaps[other] for other in excluded
        )
    expected_edges = {
        tuple(sorted((node, other)))
        for node, neighbors in selected.items()
        for other in neighbors
    }
    assert selected == {node: {1 - node, 10, 11} for node in _TARGETS}
    assert expected_edges == _PROXIMITY_EDGES
    assert set(final.edges) - set(source.edges) == expected_edges
    assert len(final.edges) == 45 and nx.is_connected(final)
    assert runtime.graph["UM_CANDIDATE_COUNT"] == 3
    assert runtime.graph["_edge_version"] == 5
    assert runtime.graph["_t"] == 7 and runtime.graph["RANDOM_SEED"] == 17
    assert "_trig_version" not in runtime.graph
    assert all(
        result.schedule == TWO_PHASE_JACOBI and result.nodes_processed == 2
        for result in results
    )
    for edge in source.edges:
        assert final.edges[edge] == source.edges[edge]
    for edge in expected_edges:
        assert set(final.edges[edge]) == {"weight"}
        actual = _mp(mp, final.edges[edge]["weight"])
        assert abs(actual - all_weights[edge]) < _mp(mp, _TOLERANCE)
        assert mp.mpf(3) / 4 < actual <= 1
    assert (final.edges[0, 1]["weight"] < 1) is split
    for node in final:
        data = final.nodes[node]
        assert abs(_angle(mp, _mp(mp, data["theta"]) - phases[node])) < _mp(
            mp, _TOLERANCE
        )
        if node in _TARGETS:
            assert _form(emitted, node) == _form(source, node) + _BOOST
            assert _form(final, node) == _form(emitted, node)
            assert coupled.nodes[node]["nu_f"] == 1 and data["nu_f"] == 0.5
            assert abs(_mp(mp, data["delta_nfr"]) - pressures[node]) < _mp(
                mp, _TOLERANCE
            )
            assert tuple(data["glyph_history"]) == ("AL",) * (2 * node) + (
                "AL",
                "UM",
                "SHA",
            )
            assert data["_operator_step"] == 2 * node + 3
            assert data["_grammar_u2_debt"] == 0
            assert data["_structural_lineage"]["activation_count"] == 2 * node + 1
            assert data["emission_timestamp"] == (_ORIGIN if node else _NOW.isoformat())
            assert data["latent"] is True
            assert data["preserved_epi"] == float(_form(final, node))
            assert data["latency_start_time"] == _NOW.isoformat()
        else:
            expected = deepcopy(source.nodes[node])
            expected["theta"] = data["theta"]
            if node in _NEIGHBORS:
                # The existing graph-level phase setter maintains this mirror.
                expected["phase"] = data["theta"]
            assert data == expected
            if node not in _NEIGHBORS:
                assert data == source.nodes[node]
    assert all(final.nodes[node] == source.nodes[node] for node in _B)


def _assert_proximity_full_rows_weighted_means_and_hybrid_work(
    count3_contact, count3_continuation, mp
):
    _, (source, emitted, coupled, final), _, _ = count3_contact
    report = count3_continuation
    assert report.degrees == tuple(
        7 if node in _TARGETS else 6 if node in (10, 11) else 4 for node in final
    )
    assert report.capacity == (Q(1, 2),) * 2 + (Q(1),) * 18
    assert dict(zip(report.edges, report.conductances)) == {
        edge: Q(final.edges[edge]["weight"]) for edge in final.edges
    }
    rho = tuple(
        Q(degree) / capacity
        for degree, capacity in zip(report.degrees, report.capacity)
    )
    assert sum(rho[:10]) == 60 and sum(rho[10:]) == 44 and sum(rho) == 104
    gradient, currents = [Q(0)] * 20, [mp.mpf(0)] * 20
    for left, right, data in final.edges(data=True):
        flux = Q(data["weight"]) * (_form(final, left) - _form(final, right))
        current = mp.sin(
            _mp(mp, final.nodes[right]["theta"]) - _mp(mp, final.nodes[left]["theta"])
        )
        gradient[left] += flux
        gradient[right] -= flux
        currents[left] += current
        currents[right] -= current
    assert tuple(gradient) == report.form_gradient
    rates = tuple(
        (
            currents[node] / (_mp(mp, rho[node]) * mp.pi),
            _mp(mp, gradient[node]) / (_mp(mp, rho[node]) * mp.pi),
        )
        for node in final
    )
    for node, (form_rate, phase_rate) in enumerate(rates):
        _contains(mp, report.form_rates[node], form_rate)
        _contains(mp, report.phase_rates[node], phase_rate)
        _contains(mp, report.pressure[node], form_rate / _mp(mp, report.capacity[node]))
        assert report.node_balance_residual[node].contains(0)
    for row in (0, 1):
        assert abs(
            sum(_mp(mp, rho[node]) * rates[node][row] for node in final)
        ) < mp.mpf("1e-90")
    assert abs(
        sum(
            _mp(mp, gradient[node]) * rates[node][0] - currents[node] * rates[node][1]
            for node in final
        )
    ) < mp.mpf("1e-90")
    with pytest.raises(ValueError, match="unit conductances"):
        bound_relational_sine_exchange(final, reference_model=_MODEL)
    old = tuple(source.edges)
    al_work = _storage(mp, emitted, old) - _storage(mp, source, old)
    expected_al = sum(
        _BOOST * (_form(source, node) - _form(source, other)) + _BOOST**2 / 2
        for node in _TARGETS
        for other in source[node]
    )
    assert abs(al_work - _mp(mp, expected_al)) < mp.mpf("1e-90")
    old_phase_work = sum(
        mp.cos(
            _mp(mp, emitted.nodes[right]["theta"])
            - _mp(mp, emitted.nodes[left]["theta"])
        )
        - mp.cos(
            _mp(mp, coupled.nodes[right]["theta"])
            - _mp(mp, coupled.nodes[left]["theta"])
        )
        for left, right in old
    )
    attached_work = _storage(mp, coupled, _PROXIMITY_EDGES)
    final_storage = _storage(mp, final, tuple(final.edges))
    assert attached_work > 0
    assert abs(
        final_storage
        - _storage(mp, source, old)
        - al_work
        - old_phase_work
        - attached_work
    ) < mp.mpf("1e-90")
    assert final_storage == _storage(mp, coupled, tuple(coupled.edges))
    _contains(mp, report.storage, final_storage)
    assert report.continuous_loss == 0
    assert report.storage_rate.contains(0) and report.balance_residual.contains(0)


def test_proximity_word_selects_three_nearest_and_preserves_actual_resets(
    count3_contact, mp
):
    _assert_proximity_word_selects_three_nearest_and_preserves_actual_resets(
        count3_contact, mp
    )


def test_proximity_full_rows_weighted_means_and_hybrid_work(
    count3_contact, count3_continuation, mp
):
    _assert_proximity_full_rows_weighted_means_and_hybrid_work(
        count3_contact, count3_continuation, mp
    )


def _exact_laplacian(size, weighted_edges):
    result = [[Q(0)] * size for _ in range(size)]
    for left, right, weight in weighted_edges:
        result[left][left] += Q(weight)
        result[right][right] += Q(weight)
        result[left][right] -= Q(weight)
        result[right][left] -= Q(weight)
    return tuple(map(tuple, result))


def _shift_matrix(matrix, scalar, *, centered=False):
    size = len(matrix)
    return tuple(
        tuple(
            matrix[i][j] - scalar * (Q(i == j) - (Q(1, size) if centered else 0))
            for j in range(size)
        )
        for i in range(size)
    )


def test_proximity_exact_support_gap_and_source_metric_bounds():
    source = _source(split=False)
    old = _exact_laplacian(20, ((i, j, 1) for i, j in source.edges))
    added = _exact_laplacian(20, ((i, j, 1) for i, j in _PROXIMITY_EDGES))
    full = tuple(
        tuple(a + b for a, b in zip(left, right)) for left, right in zip(old, added)
    )
    for matrix, bound in ((old, 8), (added, 4)):
        upper = tuple(
            tuple(-value for value in row) for row in _shift_matrix(matrix, bound)
        )
        assert exact_symmetric_semidefinite(upper)
    gap = _shift_matrix(full, Q(2, 5), centered=True)
    assert exact_symmetric_semidefinite(gap)
    assert exact_rank(gap) == 19
    assert not exact_symmetric_semidefinite(_shift_matrix(old, Q(2, 5), centered=True))

    # Resolve every fine pair mode, including the internal-edge contribution;
    # no pair-constant dynamics or loss of the relative A/B origin is assumed.
    basis = tuple(
        tuple(
            (
                Q(int(node // 2 == column))
                if column < 10
                else Q((1 if node % 2 == 0 else -1) * int(node // 2 == column - 10))
            )
            for column in range(20)
        )
        for node in range(20)
    )
    transpose = tuple(zip(*basis))
    reduced = exact_matrix_product(transpose, exact_matrix_product(full, basis))
    base_edges = [
        (offset + k, offset + (k + 1) % 5, 1) for offset in (0, 5) for k in range(5)
    ] + [(0, 5, 1)]
    base = _exact_laplacian(10, base_edges)
    assert all(reduced[i][j] == 4 * base[i][j] for i in range(10) for j in range(10))
    assert all(reduced[i][j] == 0 for i in range(10) for j in range(10, 20))
    assert (
        tuple(reduced[i][i] / 2 for i in range(10, 20))
        == (Q(8),) + (Q(4),) * 4 + (Q(6),) + (Q(4),) * 4
    )
    assert all(
        reduced[i][j] == 0 for i in range(10, 20) for j in range(10, 20) if i != j
    )
    relative_origin = (Q(1),) * 10 + (Q(-1),) * 10
    assert sum(value * value for value in relative_origin) == 20
    assert (
        sum(
            relative_origin[i] * full[i][j] * relative_origin[j]
            for i in range(20)
            for j in range(20)
        )
        == 16
    )
    # The two separate constituent projections would erase this costly mode.
    assert all(
        value == sum(relative_origin[:10]) / 10 for value in relative_origin[:10]
    )
    assert all(
        value == sum(relative_origin[10:]) / 10 for value in relative_origin[10:]
    )


def test_proximity_stationary_reference_and_full_weighted_hessian(count3_contact, mp):
    _, (_, _, _, final), _, _ = count3_contact
    weighted = _exact_laplacian(
        20, ((i, j, Q(data["weight"])) for i, j, data in final.edges(data=True))
    )
    unit = _exact_laplacian(20, ((i, j, 1) for i, j in final.edges))
    comparison = tuple(
        tuple(weighted[i][j] - Q(3, 4) * unit[i][j] for j in range(20))
        for i in range(20)
    )
    assert exact_symmetric_semidefinite(comparison)
    assert exact_rank(weighted) == 19
    phases = tuple((node % 10 // 2) * 2 * mp.pi / 5 for node in final)
    currents = [mp.mpf(0)] * 20
    hessian = [[mp.mpf(0)] * 20 for _ in range(20)]
    for left, right in final.edges:
        difference = phases[right] - phases[left]
        current, curvature = mp.sin(difference), mp.cos(difference)
        currents[left] += current
        currents[right] -= current
        hessian[left][left] += curvature
        hessian[right][right] += curvature
        hessian[left][right] -= curvature
        hessian[right][left] -= curvature
        assert curvature > mp.mpf(3) / 10
    assert max(map(abs, currents)) < mp.mpf("1e-90")
    assert all(sum(row) == 0 for row in weighted)
    assert max(abs(sum(row)) for row in hessian) < mp.mpf("1e-90")
    added = _exact_laplacian(20, ((a, b, 1) for a, b in _PROXIMITY_EDGES))
    for i in range(20):
        for j in range(20):
            expected = _mp(mp, unit[i][j]) * mp.cos(2 * mp.pi / 5)
            expected += _mp(mp, added[i][j]) * (1 - mp.cos(2 * mp.pi / 5))
            assert abs(hessian[i][j] - expected) < mp.mpf("1e-90")

    captured = deepcopy(final)
    for node in captured:
        captured.nodes[node].update(EPI=_FORM, theta=float(phases[node]))
    report = bound_relational_sine_support_exchange(captured, reference_model=_MODEL)
    assert report.form_gradient == (Q(0),) * 20
    assert report.form_storage == 0
    for node in captured:
        represented_current = sum(
            mp.sin(
                _mp(mp, captured.nodes[other]["theta"])
                - _mp(mp, captured.nodes[node]["theta"])
            )
            for other in captured[node]
        )
        rate = (
            represented_current
            * _mp(mp, report.capacity[node])
            / (mp.pi * report.degrees[node])
        )
        _contains(mp, report.form_rates[node], rate)
        assert abs(rate) < _mp(mp, _TOLERANCE)
        assert abs(_mp(mp, report.form_rates[node].midpoint) - rate) < _mp(
            mp, _TOLERANCE
        )
        assert report.phase_rates[node].contains(0)


def test_proximity_full_source_energy_and_identity_bounds():
    z, h, radius = Q(1, 512), _PUSH, Q(1, 16)
    alpha = 2 * pi_interval() / 5
    # Bounds come from the exact edge metrics checked above. Reset form work
    # is recomputed from its eight old and four cross edges, not assigned.
    source = _source(split=False)
    increment = tuple(_BOOST if node in _TARGETS else Q(0) for node in source)
    reset_form = sum(
        (increment[i] - increment[j]) ** 2 / 2
        for i, j in (*source.edges, *_PROXIMITY_EDGES)
    )
    assert reset_form == 6 * h**2
    assert (sqrt(I(2)) * z + 3 * h).hi < 6 * h
    assert (sqrt(I(2)) * 6 * h).hi < Q(1, 64)
    assert cos(alpha).hi < Q(5, 16)
    assert Q(5, 16) + Q(1, 64) < Q(1, 3)
    target_limit = 3 * h * z
    target_corner = (target_limit, -target_limit) + (Q(0),) * 18
    new_phase_reset = sum(
        (target_corner[i] - target_corner[j]) ** 2 / 2 for i, j in _PROXIMITY_EDGES
    )
    assert new_phase_reset == 36 * h**2 * z**2
    initial_metric = 12 * z**2
    reset_metric = reset_form + 12 * h**2 + new_phase_reset
    excess = (sqrt(I(initial_metric)) + sqrt(I(reset_metric))) ** 2
    assert excess.hi < Q(1, 8192)
    assert cos(alpha).lo > Q(3, 10)
    assert (sqrt(I(2)) * radius).hi < Q(3, 32)
    assert Q(3, 10) - Q(3, 32) > Q(1, 5)
    lower_curvature = Q(1, 5) * Q(2, 5)
    boundary = lower_curvature * radius**2 / 2
    assert boundary == Q(1, 6400) and excess.hi < boundary
    # This is the full source-family norm bound, before any response reading.
    source_radius = sqrt(I(2)) * z + sqrt(I(Q(9, 5))) * _BOOST + 3 * h
    assert source_radius.hi < radius


def test_proximity_represented_sources_lie_inside_the_joint_energy_barrier(
    count3_contact, count3_continuation, mp
):
    _, (source, _, _, final), _, _ = count3_contact
    alpha = 2 * mp.pi / 5
    for component in (_A, _B):
        forms = [_mp(mp, _form(source, node)) for node in component]
        errors = [
            _mp(mp, source.nodes[node]["theta"]) - (node % 10 // 2) * alpha
            for node in component
        ]
        mean_form, mean_error = sum(forms) / 10, sum(errors) / 10
        norm = sum(
            (x - mean_form) ** 2 + (eta - mean_error) ** 2
            for x, eta in zip(forms, errors)
        )
        assert norm < _mp(mp, Q(1, 512) ** 2)
        assert abs(mean_error) < _mp(
            mp, 8 * abs(I(_TWIST) - 2 * pi_interval() / 5).abs_max
        )
    forms = [_mp(mp, _form(final, node)) for node in final]
    # Choose the unique lift within pi of the original twist at each node;
    # the checked small joint norm retains this original branch chart.
    errors = [
        _angle(mp, _mp(mp, final.nodes[node]["theta"]) - (node % 10 // 2) * alpha)
        for node in final
    ]
    mean_form, mean_error = sum(forms) / 20, sum(errors) / 20
    norm = sum(
        (x - mean_form) ** 2 + (eta - mean_error) ** 2 for x, eta in zip(forms, errors)
    )
    assert norm < _mp(mp, Q(1, 16) ** 2)
    reference_storage = 40 * (1 - mp.cos(alpha))
    actual_excess = _storage(mp, final, tuple(final.edges)) - reference_storage
    assert 0 < norm / 25 <= actual_excess < _mp(mp, Q(1, 8192))
    _contains(mp, count3_continuation.storage, actual_excess + reference_storage)
    assert (_mp(mp, count3_continuation.storage.hi) - reference_storage) < _mp(
        mp, Q(1, 6400)
    )


def test_proximity_retains_weight_to_member_association(
    count3_contact, count3_continuation
):
    split, (_, _, _, final), _, _ = count3_contact
    swapped = deepcopy(final)
    for node in _TARGETS:
        swapped.nodes[node].clear()
        swapped.nodes[node].update(deepcopy(final.nodes[1 - node]))
    changed = bound_relational_sine_support_exchange(swapped, reference_model=_MODEL)
    contrast = sum(
        Q(final.edges[0, node]["weight"]) - Q(final.edges[1, node]["weight"])
        for node in (10, 11)
    )
    u = (_form(final, 0) - _form(final, 1)) / 2
    original = count3_continuation.phase_rate_numerators()
    transported = changed.phase_rate_numerators()
    assert (
        sum(transported[node] - original[node] for node in (10, 11)) / 2
        == u * contrast / 6
    )
    assert (u * contrast != 0) is split
    mapping = {node: 1 - node if node in _TARGETS else node for node in final}
    relabeled = bound_relational_sine_support_exchange(
        nx.relabel_nodes(final, mapping, copy=True), reference_model=_MODEL
    )
    locations = {node: index for index, node in enumerate(relabeled.nodes)}
    for node in final:
        assert (
            relabeled.form_gradient[locations[mapping[node]]]
            == count3_continuation.form_gradient[node]
        )
        assert (
            relabeled.form_rates[locations[mapping[node]]]
            == count3_continuation.form_rates[node]
        )
    assert relabeled.storage == count3_continuation.storage


def test_registered_contact_preserves_full_word_and_joint_identity(
    registered_contact, mp
):
    form_offset, phase_offset, contact, report = registered_contact
    _, (source, _, _, final), _, _ = contact
    _assert_proximity_word_selects_three_nearest_and_preserves_actual_resets(
        contact, mp
    )
    _assert_proximity_full_rows_weighted_means_and_hybrid_work(contact, report, mp)
    alpha = 2 * mp.pi / 5
    rho = tuple(
        Q(degree) / capacity
        for degree, capacity in zip(report.degrees, report.capacity)
    )
    source_forms, source_errors, forms, errors = [], [], [], []
    for node in source:
        selected_sign = 1 if node == 0 else -1 if node == 1 else 0
        initial_form = _FORM + selected_sign * Q(1, 8192)
        initial_phase = (node % 10 // 2) * _TWIST + selected_sign * Q(1, 32768)
        if node in _B:
            initial_form += form_offset
            initial_phase += phase_offset
        # The offset materialization is exact for these declared dyadics;
        # the original dyadic twist still differs from the mathematical one.
        assert _form(source, node) == initial_form
        assert Q(source.nodes[node]["theta"]) == initial_phase
        source_forms.append(_mp(mp, initial_form))
        source_errors.append(_mp(mp, initial_phase) - (node % 10 // 2) * alpha)
        forms.append(_mp(mp, _form(final, node)))
        # Select the original local lift, not a global circular resultant.
        errors.append(
            _angle(mp, _mp(mp, final.nodes[node]["theta"]) - (node % 10 // 2) * alpha)
        )
    for component in (_A, _B):
        mean_form = sum(source_forms[node] for node in component) / 10
        mean_error = sum(source_errors[node] for node in component) / 10
        internal_norm = sum(
            (source_forms[node] - mean_form) ** 2
            + (source_errors[node] - mean_error) ** 2
            for node in component
        )
        assert internal_norm < _mp(mp, Q(1, 512) ** 2)
        supplied_phase = phase_offset if component == _B else 0
        defect = 2 * (_mp(mp, _TWIST) - alpha)
        assert abs(mean_error - _mp(mp, supplied_phase) - defect) < mp.mpf("1e-90")
    assert (
        sum(_form(source, node) for node in _B) / 10
        - sum(_form(source, node) for node in _A) / 10
        == form_offset
    )
    assert (
        sum(_form(final, node) for node in _B) / 10
        - sum(_form(final, node) for node in _A) / 10
        == form_offset - _BOOST / 5
    )
    weighted_form = sum(rho[node] * _form(final, node) for node in final) / sum(rho)
    assert weighted_form == _FORM + (7 * _BOOST + 11 * form_offset) / 26
    # Remove only the supplied B translation for this geometric comparison;
    # this does not claim that the shifted conductances are zero-offset output.
    unshifted_forms = tuple(
        forms[node] - (_mp(mp, form_offset) if node in _B else 0) for node in final
    )
    unshifted_errors = tuple(
        errors[node] - (_mp(mp, phase_offset) if node in _B else 0) for node in final
    )

    def norm_squared(x, eta):
        mean_x, mean_eta = sum(x) / 20, sum(eta) / 20
        return sum(
            (value - mean_x) ** 2 + (angle - mean_eta) ** 2
            for value, angle in zip(x, eta)
        )

    norm = norm_squared(forms, errors)
    base_norm = norm_squared(unshifted_forms, unshifted_errors)
    phase_reset_mean = (
        sum(
            _angle(
                mp,
                _mp(mp, final.nodes[node]["theta"])
                - _mp(mp, source.nodes[node]["theta"]),
            )
            for node in _A
        )
        / 10
    )
    expected_norm = base_norm + 5 * _mp(mp, form_offset**2 + phase_offset**2)
    expected_norm -= 2 * _mp(mp, _BOOST * form_offset)
    expected_norm -= 10 * phase_reset_mean * _mp(mp, phase_offset)
    assert abs(norm - expected_norm) < mp.mpf("1e-90")
    weighted_error_shift = sum(
        _mp(mp, rho[node]) * (errors[node] - unshifted_errors[node]) for node in final
    ) / _mp(mp, sum(rho))
    assert abs(weighted_error_shift - _mp(mp, 11 * phase_offset / 26)) < mp.mpf("1e-90")
    assert norm < _mp(mp, Q(1, 16) ** 2)
    reference_storage = 40 * (1 - mp.cos(alpha))
    actual_excess = _storage(mp, final, tuple(final.edges)) - reference_storage
    assert norm / 25 <= actual_excess < _mp(mp, Q(127, 10) ** 2 * _PUSH**2)
    _contains(mp, report.storage, reference_storage + actual_excess)
    assert _mp(mp, report.storage.hi) - reference_storage < _mp(mp, Q(1, 6400))


def test_registered_contact_axes_resolve_rigid_form_and_phase_storage(
    registered_contact, mp
):
    form_offset, phase_offset, contact, report = registered_contact
    _, (_, _, _, final), _, _ = contact
    rigid_form = tuple(form_offset if node in _B else Q(0) for node in final)
    rigid_phase = tuple(
        (node % 10 // 2) * 2 * mp.pi / 5 + (_mp(mp, phase_offset) if node in _B else 0)
        for node in final
    )
    direct_form = sum(
        Q(final.edges[i, j]["weight"]) * (rigid_form[i] - rigid_form[j]) ** 2 / 2
        for i, j in final.edges
    )
    direct_phase = sum(
        1 - mp.cos(rigid_phase[j] - rigid_phase[i]) for i, j in final.edges
    )
    cross_weight = sum(
        Q(final.edges[i, j]["weight"]) for i, j in final.edges if i in _A and j in _B
    )
    expected_phase = 4 * (1 - mp.cos(_mp(mp, phase_offset)))
    assert direct_form == cross_weight * form_offset**2 / 2
    assert abs(
        direct_phase - 40 * (1 - mp.cos(2 * mp.pi / 5)) - expected_phase
    ) < mp.mpf("1e-90")
    assert (direct_form == 0) is (form_offset == 0)
    assert (expected_phase == 0) is (phase_offset == 0)
    assert cross_weight > 3
    assert cross_weight == sum(
        weight
        for (i, j), weight in zip(report.edges, report.conductances)
        if i in _A and j in _B
    )


def test_independent_registration_square_has_uniform_strict_energy_margin():
    z, h, radius = Q(1, 512), _PUSH, Q(1, 16)
    source = _source(split=False)
    old = _exact_laplacian(20, ((i, j, 1) for i, j in source.edges))
    added = _exact_laplacian(20, ((i, j, 1) for i, j in _PROXIMITY_EDGES))
    old_upper = tuple(
        tuple(-entry for entry in row) for row in _shift_matrix(old, Q(15, 2))
    )
    assert exact_symmetric_semidefinite(old_upper, strict=True)
    added_upper = tuple(
        tuple(-entry for entry in row) for row in _shift_matrix(added, Q(4))
    )
    assert exact_symmetric_semidefinite(added_upper)
    old_max, added_max = Q(15, 2), Q(4)
    form_max, phase_max = old_max + added_max, old_max / 3 + added_max
    initial_metric = max(form_max, phase_max) * (2 * z**2) / 2
    assert initial_metric == 46 * h**2
    displacement_squared = (Q(64, 9) + 18 * z**2) * h**2
    reset_metric = 6 * h**2 + old_max * displacement_squared / 6 + 36 * h**2 * z**2
    assert reset_metric == (Q(134, 9) + Q(117, 2) * z**2) * h**2
    assert reset_metric < 15 * h**2
    # Derive the two independent offset costs from actual support incidence.
    rigid_offset = tuple(h if node in _B else Q(0) for node in source)
    assert sum((rigid_offset[i] - rigid_offset[j]) ** 2 for i, j in source.edges) == 0
    offset_metric = sum(
        (rigid_offset[i] - rigid_offset[j]) ** 2 for i, j in _PROXIMITY_EDGES
    )
    assert offset_metric == 4 * h**2
    excess = (
        sqrt(I(initial_metric)) + sqrt(I(15 * h**2)) + sqrt(I(offset_metric))
    ) ** 2
    ceiling = Q(127, 10) ** 2 * h**2
    boundary = radius**2 / 25
    assert excess.hi < ceiling < boundary
    assert boundary - ceiling == Q(51, 20) * h**2
    # The open source square enters the same radius before using conservation.
    norm_upper = sqrt(I(2)) * z + sqrt(I(11)) * h + sqrt(I(10)) * h
    assert norm_upper.hi < radius
    # Admission retains translated B forms and the separated nearest set.
    circular_pi = pi_interval()
    alpha = 2 * circular_pi / 5
    assert Q(9, 32) - h - z > Q(1, 4)
    assert (alpha - h - 2 * z).lo > h + 2 * z
    assert h + 2 * z + 3 * h * z < circular_pi.lo / 2
    score_lower = (
        1 - (h + 2 * z + 3 * h * z) / (2 * circular_pi) - (h + 2 * z + _BOOST) / 2
    )
    assert score_lower.lo > Q(3, 4)


def test_registration_family_allows_zero_postword_form_storage():
    b, z = _BOOST, Q(1, 512)
    offset = b / 5
    deviations = tuple(-4 * b / 5 if node in _TARGETS else b / 5 for node in _A)
    assert 0 < offset < _PUSH
    assert sum(deviations) == 0
    assert sum(value**2 for value in deviations) == 8 * b**2 / 5 < z**2
    initial = tuple(
        _FORM + deviations[node] if node in _A else _FORM + offset for node in range(20)
    )
    final = tuple(
        value + (b if node in _TARGETS else 0) for node, value in enumerate(initial)
    )
    assert final == (_FORM + offset,) * 20
    source = _source(split=False)
    # UM and SHA preserve these forms. Every edge difference vanishes, so
    # no assumption about the generated positive conductances is needed.
    assert all(final[i] - final[j] == 0 for i, j in (*source.edges, *_PROXIMITY_EDGES))


def _joined_readout(report):
    rho = tuple(
        Q(degree) / capacity
        for degree, capacity in zip(report.degrees, report.capacity)
    )
    totals = {component: sum(rho[node] for node in component) for component in (_A, _B)}
    assert totals == {_A: Q(60), _B: Q(44)}
    return tuple(
        -rho[node] / totals[_A] if node in _A else rho[node] / totals[_B]
        for node in report.nodes
    )


def test_exchange_pair_reuses_matched_states_and_distinct_actual_weights(
    opposite_phase_contacts, registered_preparations, mp
):
    plus, minus = opposite_phase_contacts
    assert plus is registered_preparations((0, 1))
    assert minus is registered_preparations((0, -1))
    assert plus is not minus
    a_plus, g, contact_plus, report_plus = plus
    a_minus, opposite_g, contact_minus, report_minus = minus
    assert a_plus == a_minus == 0 and g == -opposite_g == Q(3, 4096)
    states_plus, states_minus = contact_plus[1], contact_minus[1]
    for positive, negative in zip(states_plus, states_minus):
        assert all(positive.nodes[node] == negative.nodes[node] for node in _A)
        for node in _B:
            positive_data, negative_data = dict(positive.nodes[node]), dict(
                negative.nodes[node]
            )
            difference = Q(positive_data.pop("theta")) - Q(negative_data.pop("theta"))
            assert difference == 2 * g
            assert positive_data == negative_data
    assert report_plus.epi == report_minus.epi
    assert report_plus.capacity == report_minus.capacity
    assert report_plus.degrees == report_minus.degrees
    positive, negative = states_plus[-1], states_minus[-1]
    cross = tuple(edge for edge in positive.edges if edge[0] in _A and edge[1] in _B)
    assert len(cross) == 4
    assert all(
        positive.edges[edge] == negative.edges[edge]
        for edge in positive.edges
        if edge not in cross
    )
    oracle_plus = _reset_oracle(mp, states_plus[0])[2]
    oracle_minus = _reset_oracle(mp, states_minus[0])[2]
    observed = []
    for edge in cross:
        ideal_difference = oracle_plus[edge] - oracle_minus[edge]
        actual_difference = _mp(
            mp, Q(positive.edges[edge]["weight"]) - Q(negative.edges[edge]["weight"])
        )
        assert abs(ideal_difference) <= _mp(mp, g) / mp.pi
        assert abs(actual_difference - ideal_difference) < 2 * _mp(mp, _TOLERANCE)
        observed.append(actual_difference)
    assert any(value != 0 for value in observed)


def test_exchange_readout_is_exactly_the_complete_cut_current(
    opposite_phase_contacts, mp
):
    derivatives, currents, phases = [], [], []
    for _, _, contact, report in opposite_phase_contacts:
        final = contact[1][-1]
        coefficients = _joined_readout(report)
        assert (
            sum(coefficients[node] * report.epi[node] for node in final)
            == -7 * _BOOST / 15
        )
        mobility_coefficients = tuple(
            coefficients[node] * report.capacity[node] / report.degrees[node]
            for node in final
        )
        row_currents = [mp.mpf(0)] * 20
        form_gradients = [Q(0)] * 20
        cut_current = mp.mpf(0)
        weighted_cut = Q(0)
        phase = tuple(_mp(mp, final.nodes[node]["theta"]) for node in final)
        for left, right in final.edges:
            flux = mp.sin(phase[right] - phase[left])
            form_flux = Q(final.edges[left, right]["weight"]) * (
                _form(final, left) - _form(final, right)
            )
            row_currents[left] += flux
            row_currents[right] -= flux
            form_gradients[left] += form_flux
            form_gradients[right] -= form_flux
            coefficient = mobility_coefficients[left] - mobility_coefficients[right]
            if left in _A and right in _B:
                assert coefficient == -Q(13, 330)
                cut_current += flux
                weighted_cut -= form_flux
            else:
                assert coefficient == 0
        derivative = (
            sum(
                _mp(mp, mobility_coefficients[node]) * row_currents[node]
                for node in final
            )
            / mp.pi
        )
        expected = -_mp(mp, Q(13, 330)) * cut_current / mp.pi
        assert abs(derivative - expected) < mp.mpf("1e-90")
        interval = sum(
            (coefficients[node] * report.form_rates[node] for node in final), I(0)
        )
        _contains(mp, interval, derivative)
        assert abs(_mp(mp, interval.midpoint) - derivative) < _mp(mp, _TOLERANCE)
        phase_derivative = (
            _mp(
                mp,
                sum(
                    mobility_coefficients[node] * form_gradients[node] for node in final
                ),
            )
            / mp.pi
        )
        expected_phase_derivative = _mp(mp, Q(13, 330) * weighted_cut) / mp.pi
        assert abs(phase_derivative - expected_phase_derivative) < mp.mpf("1e-90")
        phase_interval = sum(
            (coefficients[node] * report.phase_rates[node] for node in final), I(0)
        )
        _contains(mp, phase_interval, phase_derivative)
        assert abs(_mp(mp, phase_interval.midpoint) - phase_derivative) < _mp(
            mp, _TOLERANCE
        )
        derivatives.append(derivative)
        currents.append(cut_current)
        phases.append(phase)
    positive, negative = opposite_phase_contacts
    final = positive[2][1][-1]
    g = positive[1]
    cos_sum = sum(
        mp.cos((phases[0][right] + phases[1][right]) / 2 - phases[0][left])
        for left, right in final.edges
        if left in _A and right in _B
    )
    expected_current = 2 * mp.sin(_mp(mp, g)) * cos_sum
    assert abs(currents[0] - currents[1] - expected_current) < mp.mpf("1e-90")
    assert cos_sum > mp.mpf(18) / 5
    expected_derivative = -_mp(mp, Q(13, 330)) * expected_current / mp.pi
    assert abs(derivatives[0] - derivatives[1] - expected_derivative) < mp.mpf("1e-90")
    assert expected_derivative < 0
    assert _joined_readout(positive[3]) == _joined_readout(negative[3])


def test_disconnected_exchange_null_keeps_the_joined_observation(
    opposite_phase_contacts,
):
    coefficients = _joined_readout(opposite_phase_contacts[0][3])
    components = []
    for _, _, contact, _ in opposite_phase_contacts:
        final = contact[1][-1]
        reports = []
        for component in (_A, _B):
            graph = final.subgraph(component).copy()
            assert nx.is_connected(graph)
            report = bound_relational_sine_support_exchange(
                graph, reference_model=_MODEL
            )
            expected_degrees = tuple(5 if node in _TARGETS else 4 for node in component)
            assert report.degrees == expected_degrees
            own_rho = tuple(
                Q(degree) / capacity
                for degree, capacity in zip(report.degrees, report.capacity)
            )
            assert sum(own_rho) == (52 if component == _A else 40)
            # Fixed joined readout weights are not the null model's invariant
            # mean weights: their mobility coefficients vary on an old edge.
            row_coefficients = tuple(
                coefficients[node] / weight for node, weight in zip(component, own_rho)
            )
            assert row_coefficients[0] != row_coefficients[2]
            reports.append(report)
        components.append(reports)
    for positive, negative in zip(*components):
        assert positive.edges == negative.edges
        assert positive.conductances == negative.conductances
        assert positive.epi == negative.epi
        assert positive.capacity == negative.capacity
        assert positive.form_gradient == negative.form_gradient
        assert positive.form_rates == negative.form_rates
        assert positive.phase_rates == negative.phase_rates
        positions = {node: index for index, node in enumerate(positive.nodes)}
        rotation = tuple(
            left - right for left, right in zip(positive.phase, negative.phase)
        )
        for left, right in positive.edges:
            i, j = positions[left], positions[right]
            # This incidence identity holds for every internal phase state:
            # a component-constant rotation cannot change either complete row.
            assert rotation[j] - rotation[i] == 0
            assert (
                positive.phase[j] - positive.phase[i]
                == negative.phase[j] - negative.phase[i]
            )
        assert sum(
            coefficients[node] * value
            for node, value in zip(positive.nodes, positive.epi)
        ) == sum(
            coefficients[node] * value
            for node, value in zip(negative.nodes, negative.epi)
        )


def test_full_family_exchange_error_bounds_reach_the_frozen_window(
    opposite_phase_contacts,
):
    _, g, contact, report = opposite_phase_contacts[0]
    final = contact[1][-1]
    radius, horizon, window_start = Q(1, 16), Q(1, 8), Q(1, 16)
    circular_pi = pi_interval()
    assert (sqrt(I(2)) * radius).hi < Q(1, 10)
    cross_counts = tuple(
        sum((node in _A) != (other in _A) for other in final[node]) for node in final
    )
    row_factors = tuple(
        report.capacity[node] * count / report.degrees[node]
        for node, count in enumerate(cross_counts)
    )
    assert max(row_factors) == Q(1, 3)
    # The forcing bound uses only the actual two-contact port count, held
    # mobility, score Lipschitz factor g/pi, and maintained form range<1/10.
    forcing = max(row_factors) / (10 * circular_pi**2)
    assert (forcing - 1 / (30 * circular_pi**2)).contains(0)
    growth = 1 / exp_neg_unit(2 * horizon / circular_pi)
    norm_bound = (2 + 1 / (60 * circular_pi)) * growth - 1 / (60 * circular_pi)
    assert norm_bound.hi < Q(9, 4)
    phase_defect = 9 * horizon**2 / (2 * circular_pi**2) + horizon / (
        30 * circular_pi**2
    )
    simple_defect = horizon**2 / 2 + horizon / 270
    assert phase_defect.hi < simple_defect == Q(143, 17280) < Q(1, 100)
    gap_separation = 2 - 2 * simple_defect
    assert gap_separation > Q(99, 50)
    assert cos(I(Q(1, 10))).lo > Q(9, 10)
    current_separation = 4 * Q(99, 50) * Q(9, 10)
    assert current_separation == Q(891, 125) > 7
    kappa = Q(13, 330) / circular_pi
    assert (7 * kappa).lo > Q(1, 12)
    assert g * horizon / 12 == Q(1, 2**17)
    assert g * window_start / 12 == Q(1, 2**18)
