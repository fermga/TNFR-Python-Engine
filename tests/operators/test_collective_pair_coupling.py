"""Frozen static AL/UM/SHA controls for the declared collective support gate.

The source, flags, factors and two probes follow SINE_PAIR_COUPLING section 1.
Before response evaluation, materialize its twist as 674651885/536870912,
the existing pair-word control's dyadic approximation. Independent 100-digit
arithmetic checks the real reset on these represented primitives to 2**-46;
it does not assert bitwise equivariance for every mathematical source.
No continuous trajectory or autonomous event selector is evaluated here.
"""

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import pi

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.glyph_history import push_glyph
from tnfr.operators import network_stage
from tnfr.operators.definitions import Coupling, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_scale import (
    assess_sine_pair_support_symmetry,
    assess_sine_replica_capacity,
    assess_sine_replica_scale,
)
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.types import require_finite_real_scalar_epi

_PAIRS = tuple((2 * index, 2 * index + 1) for index in range(5))
_TARGETS = _PAIRS[0]
_NEIGHBORS = (2, 3, 8, 9)
_NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_ORIGIN = "2029-01-01T00:00:00+00:00"
_CORE = {"compatibility_profile": "core"}
_MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
_FORM, _TWIST = Q(5, 16), Q(674651885, 536870912)
_BOOST = _PUSH = Q(1, 1024)
_SYNC = _REDUCTION = Q(1, 8)
_ATTENUATION = Q(1, 2)
_TOLERANCE = Q(1, 2**46)


def _source(*, split):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for index, pair in enumerate(_PAIRS):
        graph.add_edges_from(
            (left, right, {"weight": 1, "length": 1})
            for left in pair
            for right in _PAIRS[(index + 1) % 5]
        )
        u = Q(1, 8192) if split and index == 0 else Q(0)
        delta = Q(1, 32768) if split and index == 0 else Q(0)
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
            "UM_vf_sync": float(_SYNC),
            "UM_dnfr_reduction": float(_REDUCTION),
            "SHA_vf_factor": float(_ATTENUATION),
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


def _word():
    operators = (Emission(), Coupling(), Silence())
    admission = validate_sequence(
        [operator.name for operator in operators], context=_CORE
    )
    assert admission.passed
    assert admission.metadata["compatibility_profile"] == "core"
    return operators, ValidatedSequence(operators, context=_CORE)


class _SuppliedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW if tz is None else _NOW.astimezone(tz)


@pytest.fixture
def supplied_clock(monkeypatch):
    monkeypatch.setattr(network_stage, "datetime", _SuppliedClock)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    rational = Q(value)
    return mp.mpf(rational.numerator) / rational.denominator


def _form(graph, node):
    return Q(require_finite_real_scalar_epi(graph.nodes[node]["EPI"], "word form"))


def _angle(mp, value):
    return mp.atan2(mp.sin(value), mp.cos(value))


def _independent_reset(mp, source):
    """Use complex resultants and the stated real merge, not stage proposals."""
    theta = {node: _mp(mp, data["theta"]) for node, data in source.nodes(data=True)}
    centers = {
        node: mp.arg(sum(mp.exp(mp.j * theta[other]) for other in (node, *_NEIGHBORS)))
        for node in _TARGETS
    }
    final = dict(theta)
    for node in (*_TARGETS, *_NEIGHBORS):
        sources = (node,) if node in _TARGETS else _TARGETS
        final[node] += (
            _mp(mp, _PUSH)
            * sum(_angle(mp, centers[target] - theta[node]) for target in sources)
            / len(sources)
        )
    pressure = {}
    for node in _TARGETS:
        affinity = (
            sum(
                1 - abs(_angle(mp, final[node] - final[other])) / mp.pi
                for other in _NEIGHBORS
            )
            / 4
        )
        pressure[node] = _mp(mp, source.nodes[node]["delta_nfr"]) * (
            1 - _mp(mp, _REDUCTION) * affinity
        )
    forms = tuple(_mp(mp, _form(source, node) + _BOOST) for node in _TARGETS)
    phase_distance = abs(_angle(mp, final[0] - final[1]))
    # Both supplied Si values are equal. The epsilon is the represented
    # production constant, retained explicitly rather than replaced by zero.
    weight = (
        1
        - phase_distance / (2 * mp.pi)
        - abs(forms[0] - forms[1])
        / (4 * (abs(forms[0]) + abs(forms[1]) + _mp(mp, 1e-9)))
    )
    return final, pressure, weight


def _execute(graph, targets=_TARGETS):
    operators, sequence = _word()
    return tuple(
        execute_network_operator_stage(
            graph, operator, targets, sequence_context=sequence.step(index)
        )
        for index, operator in enumerate(operators)
    )


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


def _detached_state(graph):
    """Copy scientific primitives, without graph-bound runtime cache objects."""
    result = nx.Graph()
    result.graph["GAMMA"] = deepcopy(graph.graph["GAMMA"])
    result.add_nodes_from(
        (node, deepcopy(data)) for node, data in graph.nodes(data=True)
    )
    result.add_edges_from(
        (left, right, deepcopy(data)) for left, right, data in graph.edges(data=True)
    )
    return result


@pytest.mark.parametrize("split", (True, False))
def test_complete_word_matches_reset_support_and_associated_history(
    split, supplied_clock, mp
):
    graph = _source(split=split)
    source = deepcopy(graph)
    expected_phase, expected_pressure, expected_weight = _independent_reset(mp, source)
    results = _execute(graph)
    assert all(result.schedule == TWO_PHASE_JACOBI for result in results)
    assert all(result.nodes_processed == 2 for result in results)
    assert graph.graph["_t"] == 7
    assert graph.graph["RANDOM_SEED"] == 17
    # Adding support invalidates the trig counter after the phase writes.
    assert "_trig_version" not in graph.graph
    assert graph.graph["_edge_version"] == 1
    assert set(graph.edges) - set(source.edges) == {(0, 1)}
    assert set(graph.edges[0, 1]) == {"weight"}
    for edge in source.edges:
        assert graph.edges[edge] == source.edges[edge]
    weight = graph.edges[0, 1]["weight"]
    assert abs(_mp(mp, weight) - expected_weight) < _mp(mp, _TOLERANCE)
    assert 3 / 4 < weight <= 1
    assert (weight < 1) is split
    for node in graph:
        assert abs(
            _angle(mp, _mp(mp, graph.nodes[node]["theta"]) - expected_phase[node])
        ) < _mp(mp, _TOLERANCE)
        if node not in _TARGETS:
            assert _form(graph, node) == _form(source, node)
            assert graph.nodes[node]["nu_f"] == source.nodes[node]["nu_f"]
            assert graph.nodes[node]["delta_nfr"] == source.nodes[node]["delta_nfr"]
            assert graph.nodes[node]["glyph_history"] == []
            assert "_operator_step" not in graph.nodes[node]
            if node in _NEIGHBORS:
                assert graph.nodes[node]["theta"] != source.nodes[node]["theta"]
            else:
                assert graph.nodes[node] == source.nodes[node]
            continue
        data = graph.nodes[node]
        assert _form(graph, node) == _form(source, node) + _BOOST
        assert Q(data["nu_f"]) == _ATTENUATION
        assert abs(_mp(mp, data["delta_nfr"]) - expected_pressure[node]) < _mp(
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
        assert data["preserved_epi"] == float(_form(graph, node))
        assert data["latency_start_time"] == _NOW.isoformat()
    if split:
        with pytest.raises(ValueError, match="unit conductances"):
            bound_relational_sine_exchange(graph, reference_model=_MODEL)


def test_pair_swap_transports_state_and_unequal_histories(supplied_clock):
    source = _source(split=True)
    forward, swapped = deepcopy(source), deepcopy(source)
    for node in _TARGETS:
        swapped.nodes[1 - node].clear()
        swapped.nodes[1 - node].update(deepcopy(source.nodes[node]))
    assert _execute(forward) == _execute(swapped, tuple(reversed(_TARGETS)))
    for node in source:
        partner = 1 - node if node in _TARGETS else node
        assert forward.nodes[node] == swapped.nodes[partner]
    assert dict(forward.edges) == dict(swapped.edges)
    assert forward.graph["_t"] == swapped.graph["_t"]
    assert "_trig_version" not in forward.graph and "_trig_version" not in swapped.graph
    assert forward.graph["_edge_version"] == swapped.graph["_edge_version"]


@pytest.mark.parametrize("condition", ("phase", "sense"))
def test_rejected_um_retains_only_the_committed_emission_prefix(
    condition, supplied_clock
):
    graph = _source(split=True)
    if condition == "phase":
        graph.nodes[0]["theta"] = pi
        error, message = OperatorPreconditionError, "U3 phase gate"
    else:
        graph.nodes[1]["Si"] = -0.5
        error, message = TNFRValueError, "sense index must be nonnegative"
    operators, sequence = _word()
    execute_network_operator_stage(
        graph, operators[0], _TARGETS, sequence_context=sequence.step(0)
    )
    before = _structural_snapshot(graph)
    with pytest.raises(error, match=message):
        execute_network_operator_stage(
            graph, operators[1], _TARGETS, sequence_context=sequence.step(1)
        )
    assert _structural_snapshot(graph) == before
    assert not graph.has_edge(0, 1)
    assert all(graph.nodes[node]["glyph_history"][-1] == "AL" for node in _TARGETS)


@pytest.fixture(scope="module")
def unit_prefixes():
    graph = _source(split=False)
    states = [_detached_state(graph)]
    operators, sequence = _word()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        for index, operator in enumerate(operators):
            execute_network_operator_stage(
                graph, operator, _TARGETS, sequence_context=sequence.step(index)
            )
            states.append(_detached_state(graph))
    return tuple(states)


def _fine_storage(mp, graph, edges):
    return sum(
        (_mp(mp, _form(graph, left) - _form(graph, right))) ** 2 / 2
        + 1
        - mp.cos(
            _mp(mp, graph.nodes[left]["theta"]) - _mp(mp, graph.nodes[right]["theta"])
        )
        for left, right in edges
    )


def test_unit_contact_uses_actual_degree_and_signed_support_work(unit_prefixes, mp):
    source, emitted, coupled, silenced = unit_prefixes
    assert coupled.edges[0, 1] == {"weight": 1.0}
    symmetry = assess_sine_pair_support_symmetry(
        coupled, reference_model=_MODEL, pairs=_PAIRS
    )
    assert symmetry.independent_pair_swaps_equivariant
    assert symmetry.within_pair_edge_indices == (0,)
    assert symmetry.strict_replica_admission_status == "rejected"
    actual = bound_relational_sine_exchange(silenced, reference_model=_MODEL)
    assert actual.degrees == (5, 5) + (4,) * 8
    assert actual.capacity == (Q(1, 2),) * 2 + (Q(1),) * 8
    with pytest.raises(ValueError, match="within-pair edges"):
        assess_sine_replica_scale(coupled, reference_model=_MODEL, pairs=_PAIRS)
    old_edges = tuple(source.edges)
    old_before = _fine_storage(mp, emitted, old_edges)
    old_after = _fine_storage(mp, coupled, old_edges)
    attachment = _fine_storage(mp, coupled, ((0, 1),))
    whole_jump = _fine_storage(mp, coupled, tuple(coupled.edges)) - old_before
    assert attachment == 0
    assert old_after - old_before > 0
    assert abs(whole_jump - (old_after - old_before + attachment)) < mp.mpf("1e-90")
    assert _fine_storage(mp, silenced, tuple(silenced.edges)) == _fine_storage(
        mp, coupled, tuple(coupled.edges)
    )
    assert _fine_storage(mp, coupled, tuple(coupled.edges)) > _fine_storage(
        mp, source, old_edges
    )
    for index, node in enumerate(actual.nodes):
        pressure = sum(
            mp.sin(
                _mp(mp, silenced.nodes[neighbor]["theta"])
                - _mp(mp, silenced.nodes[node]["theta"])
            )
            for neighbor in silenced[node]
        ) / (mp.pi * silenced.degree(node))
        phase_rate = (
            _mp(mp, actual.capacity[index])
            * sum(
                _mp(mp, _form(silenced, node) - _form(silenced, neighbor))
                for neighbor in silenced[node]
            )
            / (mp.pi * silenced.degree(node))
        )
        for bounds, value in (
            (actual.form_rates[index], _mp(mp, actual.capacity[index]) * pressure),
            (actual.phase_rates[index], phase_rate),
        ):
            assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)
        if node in _TARGETS:
            assert abs(_mp(mp, silenced.nodes[node]["delta_nfr"]) - pressure) > mp.mpf(
                "0.01"
            )


def test_unit_tip_has_explicit_auxiliary_mobility_equivalence(unit_prefixes, mp):
    actual_graph = unit_prefixes[-1]
    before = _structural_snapshot(actual_graph)
    actual = bound_relational_sine_exchange(actual_graph, reference_model=_MODEL)
    auxiliary = deepcopy(actual_graph)
    auxiliary.remove_edge(0, 1)
    for node in _TARGETS:
        auxiliary.nodes[node]["nu_f"] = Q(4, 5) * Q(actual_graph.nodes[node]["nu_f"])
    mapped = bound_relational_sine_exchange(auxiliary, reference_model=_MODEL)
    assert mapped.capacity[:2] == (Q(float(Q(2, 5))),) * 2
    for index, node in enumerate(actual.nodes):
        neighbors = tuple(auxiliary[node])
        current = sum(
            mp.sin(
                _mp(mp, actual_graph.nodes[other]["theta"])
                - _mp(mp, actual_graph.nodes[node]["theta"])
            )
            for other in neighbors
        )
        gradient = sum(
            (
                _form(actual_graph, node) - _form(actual_graph, other)
                for other in neighbors
            ),
            Q(0),
        )
        exact_capacity = (
            Q(4, 5) * actual.capacity[index]
            if node in _TARGETS
            else actual.capacity[index]
        )
        assert exact_capacity / 4 == actual.capacity[index] / actual.degrees[index]
        old_pressure = current / (4 * mp.pi)
        pressure_scale = Q(4, 5) if node in _TARGETS else Q(1)
        actual_pressure = _mp(mp, pressure_scale) * old_pressure
        assert (
            _mp(mp, actual.pressure[index].lo)
            <= actual_pressure
            <= _mp(mp, actual.pressure[index].hi)
        )
        assert (
            _mp(mp, mapped.pressure[index].lo)
            <= old_pressure
            <= _mp(mp, mapped.pressure[index].hi)
        )
        exact_rates = (
            _mp(mp, exact_capacity) * current / (4 * mp.pi),
            _mp(mp, exact_capacity * gradient) / (4 * mp.pi),
        )
        for original, represented, exact in zip(
            (actual.form_rates[index], actual.phase_rates[index]),
            (mapped.form_rates[index], mapped.phase_rates[index]),
            exact_rates,
        ):
            assert _mp(mp, original.lo) <= exact <= _mp(mp, original.hi)
            # A graph capture materializes 2/5 to binary64: keep that defect
            # explicit rather than transferring exact coefficient equality.
            # Enclosure width is separate from that coefficient error.
            represented_exact = exact * _mp(mp, mapped.capacity[index] / exact_capacity)
            assert (
                _mp(mp, represented.lo) <= represented_exact <= _mp(mp, represented.hi)
            )
            assert abs(represented_exact - exact) < _mp(mp, Q(1, 2**55))
    phase_turns = tuple(
        -1 if node in _TARGETS and auxiliary.nodes[node]["theta"] > pi else 0
        for node in auxiliary
    )
    barrier = assess_sine_replica_capacity(
        auxiliary,
        reference_model=_MODEL,
        pairs=_PAIRS,
        phase_turns=phase_turns,
        radius=Q(1, 16),
        excess_ceiling=Q(1, 16384),
        form_mean_bounds=(Q(1, 4), Q(1, 2)),
    )
    # This is the explicitly mapped represented auxiliary model. The owner
    # proves the positive-capacity-independent trap and the exact reduction;
    # this captured verdict alone does not authenticate either theorem.
    assert barrier.family.source_set_trapping_certified
    assert _structural_snapshot(actual_graph) == before


def test_actual_nonunit_contact_has_declared_weighted_continuation_and_pair_rows(
    supplied_clock, mp
):
    graph = _source(split=True)
    _execute(graph)
    weight = Q(graph.edges[0, 1]["weight"])
    assert Q(3, 4) < weight < 1
    before = deepcopy((dict(graph.nodes), dict(graph.edges)))
    report = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    assert deepcopy((dict(graph.nodes), dict(graph.edges))) == before
    assert report.conductances[report.edges.index((0, 1))] == weight
    assert report.degrees == (5, 5) + (4,) * 8
    assert report.law == "conservative_sine_support_exchange"
    with pytest.raises(ValueError, match="unit conductances"):
        bound_relational_sine_exchange(graph, reference_model=_MODEL)
    # All pair means retain the supplied continuous lifts around the winding.
    # The actual graph and report retain their stored circular representatives.
    phase = tuple(
        _mp(mp, report.phase[node])
        - (2 * mp.pi if node in _TARGETS and report.phase[node] > pi else 0)
        for node in report.nodes
    )
    means = tuple((report.epi[left] + report.epi[right]) / 2 for left, right in _PAIRS)
    internal = tuple(
        (report.epi[left] - report.epi[right]) / 2 for left, right in _PAIRS
    )
    centers = tuple((phase[left] + phase[right]) / 2 for left, right in _PAIRS)
    widths = tuple((phase[left] - phase[right]) / 2 for left, right in _PAIRS)
    coherence = tuple(mp.cos(delta) for delta in widths)
    for index, (left, right) in enumerate(_PAIRS):
        neighbors = ((index - 1) % 5, (index + 1) % 5)
        sine = sum(
            coherence[other] * mp.sin(centers[other] - centers[index])
            for other in neighbors
        )
        cosine = sum(
            coherence[other] * mp.cos(centers[other] - centers[index])
            for other in neighbors
        )
        gradient = sum((means[index] - means[other] for other in neighbors), Q(0))
        degree = report.degrees[left]
        mobility = 2 * _mp(mp, report.capacity[left]) / (mp.pi * degree)
        attachment = int(index == 0)
        conductance = _mp(mp, weight) if attachment else mp.mpf(0)
        expected = (
            mobility * coherence[index] * sine,
            mobility * _mp(mp, gradient),
            -mobility
            * mp.sin(widths[index])
            * (cosine + attachment * coherence[index]),
            mobility * (2 + conductance) * _mp(mp, internal[index]),
        )
        observed = (
            (report.form_rates[left] + report.form_rates[right]) / 2,
            (report.phase_rates[left] + report.phase_rates[right]) / 2,
            (report.form_rates[left] - report.form_rates[right]) / 2,
            (report.phase_rates[left] - report.phase_rates[right]) / 2,
        )
        for bounds, value in zip(observed, expected):
            assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)
    old_edges = tuple(edge for edge in graph.edges if edge != (0, 1))
    old_storage = _fine_storage(mp, graph, old_edges)
    attachment_storage = (
        2 * _mp(mp, weight * internal[0] ** 2) + 2 * mp.sin(widths[0]) ** 2
    )
    assert attachment_storage > 0
    assert (
        _mp(mp, report.storage.lo)
        <= old_storage + attachment_storage
        <= _mp(mp, report.storage.hi)
    )
    assert report.continuous_loss == 0
    assert report.storage_rate.contains(0)
    assert report.balance_residual.contains(0)
