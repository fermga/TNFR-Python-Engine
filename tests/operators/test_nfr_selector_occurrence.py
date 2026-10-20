"""Actual native occurrence needs supplied policy memory beyond pair coordinates.

The ten-node doubled-C5 source and coordinate projection are reused from the
collective-context study. Direct-batch controls isolate primitive glyphs;
runtime timing controls also execute the actual integrator and native policies
on the uniform source. No formed-source producer or continuous sine-law
continuation is invoked. Supplied trace/counter states are admitted inputs,
not reconstructed past trajectories or a derived occurrence law.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_SI
from tnfr.dynamics import runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import prepare_integration_params
from tnfr.dynamics.selectors import DefaultGlyphSelector, _apply_glyphs
from tnfr.glyph_history import push_glyph
from tnfr.metrics.sense_index import compute_Si
from tnfr.operators.grammar_debt import PRIOR_COHERENCE_KEY, U2_DEBT_KEY
from tnfr.physics.relational_sine_pair import derive_sine_global_pair_state
from tnfr.types import real_scalar_epi

NODES = tuple(range(10))


def _prepared(al_lag=0, *, target_history=("AL", "IL", "EN"), en_limit=None):
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    for block in range(5):
        following = (block + 1) % 5
        for member in (2 * block, 2 * block + 1):
            for neighbor in (2 * following, 2 * following + 1):
                graph.add_edge(member, neighbor, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(_t=1.0, RANDOM_SEED=17, GRAMMAR_REJECTION_MODE="raise")
    if en_limit is not None:
        graph.graph["EN_MAX_LAG"] = en_limit
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            glyph_history=[],
        )
        history = target_history if node == 0 else ("AL", "IL", "EN")
        for token in history:
            # Record supplied grammar context without replaying physical maps.
            push_glyph(graph.nodes[node], token, 1)
    default_compute_delta_nfr(graph)
    compute_Si(graph, inplace=True)
    assert graph.graph["GRAMMAR_CANON"]["enabled"] is True
    assert graph.graph["AL_MAX_LAG"] == 5
    memory = {
        "since_AL": {node: al_lag if node == 0 else 0 for node in NODES},
        "since_EN": dict.fromkeys(NODES, 0),
    }
    return graph, memory


def _forms(graph):
    return tuple(
        real_scalar_epi(
            get_attr(
                graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
            )
        )
        for node in NODES
    )


def _physical_state(graph):
    return (
        _forms(graph),
        tuple(
            tuple(graph.nodes[node][key] for key in ("theta", "nu_f", "delta_nfr"))
            for node in NODES
        ),
        tuple((a, b, data["weight"]) for a, b, data in graph.edges(data=True)),
    )


def _coordinates(graph):
    assert all(graph.nodes[node]["theta"] == 0.0 for node in NODES)
    result = derive_sine_global_pair_state(_forms(graph), ((Q(1), Q(0)),) * 10)
    # Retain only the coordinate map. Its sine-law rates and storage do not
    # certify the native hybrid path exercised by this test.
    return (
        result.form_means,
        result.resultants,
        result.phase_products,
        result.internal_form_squared,
        result.form_phase_moments,
    )


def _base_proposals(graph):
    selector = DefaultGlyphSelector()
    selector.prepare(graph, NODES)
    proposals = tuple(selector(graph, node) for node in NODES)
    selector.clear()
    return proposals


def _realized(graph):
    return tuple(graph.nodes[node]["glyph_history"][-1] for node in NODES)


def test_same_pair_coordinates_and_default_proposal_have_distinct_realized_actions():
    quiet, quiet_memory = _prepared(0)
    due, due_memory = _prepared(5)
    assert _physical_state(quiet) == _physical_state(due)
    assert _coordinates(quiet) == _coordinates(due)
    assert dict(quiet.nodes(data=True)) == dict(due.nodes(data=True))
    for key in (
        "GLYPH_FACTORS",
        "GRAMMAR_CANON",
        "GRAMMAR_REJECTION_MODE",
        "SELECTOR_THRESHOLDS",
        "DNFR_WEIGHTS",
        "AL_MAX_LAG",
        "EN_MAX_LAG",
        "CLIP_MODE",
        "EPI_MIN",
        "EPI_MAX",
        "RANDOM_SEED",
        "_t",
    ):
        assert quiet.graph[key] == due.graph[key]
    assert all(
        get_attr(graph.nodes[node], ALIAS_SI) > 0.9
        for graph in (quiet, due)
        for node in NODES
    )
    assert _base_proposals(quiet) == _base_proposals(due) == ("IL",) * 10
    assert quiet_memory["since_EN"] == due_memory["since_EN"]
    before = _physical_state(quiet)
    boost = due.graph["GLYPH_FACTORS"]["AL_boost"]
    expected_form = 0.25 + boost
    assert 0.25 < expected_form < due.graph["EPI_MAX"]

    _apply_glyphs(quiet, DefaultGlyphSelector(), quiet_memory)
    _apply_glyphs(due, DefaultGlyphSelector(), due_memory)

    assert _realized(quiet) == ("IL",) * 10
    assert _realized(due) == ("AL",) + ("IL",) * 9
    assert _physical_state(quiet) == before
    assert _forms(due) == (expected_form,) + (0.25,) * 9
    assert _physical_state(due)[1:] == before[1:]
    represented_increment = Q(expected_form) - Q(1, 4)
    assert represented_increment > 0
    assert _coordinates(due)[0] == (
        Q(1, 4) + represented_increment / 2,
        *(Q(1, 4),) * 4,
    )
    assert _coordinates(due)[3] == ((represented_increment / 2) ** 2,) + (Q(0),) * 4
    assert quiet_memory["since_AL"] == dict.fromkeys(NODES, 1)
    assert due_memory["since_AL"] == {0: 0, **dict.fromkeys(NODES[1:], 1)}
    assert quiet_memory["since_EN"] == due_memory["since_EN"] == dict.fromkeys(NODES, 1)
    for graph in (quiet, due):
        assert graph.graph["_t"] == 1.0
        assert all(graph.nodes[node][U2_DEBT_KEY] == 0 for node in NODES)


@pytest.mark.parametrize(
    "lag,mark,next_lag", [(0, "IL", 1), (4, "IL", 5), (5, "AL", 0), (6, "AL", 0)]
)
def test_lag_boundary_counts_native_batches_before_strict_limit(lag, mark, next_lag):
    graph, memory = _prepared(lag)
    assert _base_proposals(graph) == ("IL",) * 10
    _apply_glyphs(graph, DefaultGlyphSelector(), memory)
    assert _realized(graph) == (mark,) + ("IL",) * 9
    assert memory["since_AL"][0] == next_lag
    assert memory["since_EN"][0] == 1
    assert graph.graph["_t"] == 1.0


def test_different_admission_memory_can_still_give_the_same_realized_action():
    unpaid, unpaid_memory = _prepared(target_history=("AL", "IL", "VAL", "VAL", "EN"))
    paid, paid_memory = _prepared(target_history=("AL", "IL", "VAL", "IL", "EN"))
    assert _physical_state(unpaid) == _physical_state(paid)
    assert _coordinates(unpaid) == _coordinates(paid)
    assert unpaid.nodes[0]["glyph_history"] == paid.nodes[0]["glyph_history"]
    assert tuple(unpaid.nodes[0]["glyph_history"]) == ("EN",)
    assert (
        unpaid.nodes[0][PRIOR_COHERENCE_KEY]
        is paid.nodes[0][PRIOR_COHERENCE_KEY]
        is True
    )
    assert (unpaid.nodes[0][U2_DEBT_KEY], paid.nodes[0][U2_DEBT_KEY]) == (2, 0)
    before = _physical_state(unpaid)

    for graph, memory in ((unpaid, unpaid_memory), (paid, paid_memory)):
        assert _base_proposals(graph) == ("IL",) * 10
        _apply_glyphs(graph, DefaultGlyphSelector(), memory)
        assert _realized(graph) == ("IL",) * 10
        assert _physical_state(graph) == before
    # A common action does not erase the difference in complete bookkeeping.
    assert (unpaid.nodes[0][U2_DEBT_KEY], paid.nodes[0][U2_DEBT_KEY]) == (1, 0)


def test_one_current_eligibility_bit_does_not_predict_repeated_batch_occurrence():
    # EN is held away from its lag limit for all five declared batches. This
    # supplied policy differs from the default-EN single-batch controls above.
    early, early_memory = _prepared(0, en_limit=8)
    later, later_memory = _prepared(1, en_limit=8)
    assert _coordinates(early) == _coordinates(later)
    assert early_memory["since_AL"][0] < 5 and later_memory["since_AL"][0] < 5
    traces = [[], []]
    for batch in range(1, 6):
        for index, (graph, memory) in enumerate(
            ((early, early_memory), (later, later_memory))
        ):
            default_compute_delta_nfr(graph)
            compute_Si(graph, inplace=True)
            assert _base_proposals(graph) == ("IL",) * 10
            _apply_glyphs(graph, DefaultGlyphSelector(), memory)
            traces[index].append(_realized(graph)[0])
            assert _realized(graph)[1:] == ("IL",) * 9
            assert graph.graph["_t"] == 1.0
            assert memory["since_EN"][0] == batch
        if batch < 5:
            assert _physical_state(early) == _physical_state(later)
    assert traces == [["IL"] * 5, ["IL"] * 4 + ["AL"]]
    assert early_memory["since_AL"][0] == 5
    assert later_memory["since_AL"][0] == 0
    assert _forms(early) == (0.25,) * 10
    assert _forms(later)[0] == 0.25 + later.graph["GLYPH_FACTORS"]["AL_boost"]


def _runtime_source(*, subdivision_floor=0.0):
    graph, memory = _prepared()
    graph.graph.update(
        DT_MIN=subdivision_floor,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        history=memory,
    )
    # Keep the actual phase and capacity policies. Uniform phases/capacities
    # are their fixed points, although their bookkeeping still advances.
    assert graph.graph["PHASE_ADAPT"]["enabled"] is True
    assert graph.graph["VF_ADAPT_TAU"] == 5
    assert graph.graph["VF_ADAPT_MU"] == 0.1
    assert graph.graph["EN_MAX_LAG"] == 3
    assert graph.graph["REMESH_STABILITY_WINDOW"] == 25
    assert not graph.graph.get("MATH_ENGINE", {}).get("enabled", False)
    assert not graph.graph.get("callbacks")
    return graph, memory


def _capture_actual_native_glyphs(monkeypatch):
    records = []
    actual = selectors.apply_glyph

    def capture(graph, node, glyph, **kwargs):
        event_time = graph.graph["_t"]
        actual(graph, node, glyph, **kwargs)
        records.append(
            (graph, node, event_time, graph.nodes[node]["glyph_history"][-1])
        )

    monkeypatch.setattr(selectors, "apply_glyph", capture)
    return records


def _target_events(records, graph):
    return tuple(
        (time, glyph)
        for owner, node, time, glyph in records
        if owner is graph and node == 0
    )


def _assert_uniform_runtime_coordinates(graph, form):
    assert _forms(graph) == (form,) * 10
    assert all(
        tuple(graph.nodes[node][key] for key in ("theta", "nu_f", "delta_nfr"))
        == (0.0, 1.0, 0.0)
        for node in NODES
    )
    assert graph.number_of_edges() == 20
    assert all(data["weight"] == 1.0 for _, _, data in graph.edges(data=True))
    assert graph.graph["history"]["stable_frac"] == []
    assert "_last_remesh_step" not in graph.graph


def test_outer_decision_refinement_changes_events_at_the_same_structural_time(
    monkeypatch,
):
    coarse, coarse_memory = _runtime_source()
    refined, refined_memory = _runtime_source()
    assert _physical_state(coarse) == _physical_state(refined)
    assert coarse_memory == refined_memory
    records = _capture_actual_native_glyphs(monkeypatch)

    for graph, count, duration in ((coarse, 3, 0.25), (refined, 6, 0.125)):
        for _ in range(count):
            runtime.step(graph, dt=duration)

    assert coarse.graph["_t"] == refined.graph["_t"] == 1.75
    assert _target_events(records, coarse) == ((1.0, "IL"), (1.25, "IL"), (1.5, "IL"))
    assert _target_events(records, refined) == tuple(
        (1.0 + index / 8, glyph)
        for index, glyph in enumerate(("IL", "IL", "IL", "EN", "IL", "AL"))
    )
    # Every node executes the same mark, so the final AL is a uniform jump;
    # there is no hidden post-emission pressure response in this comparison.
    for graph, expected in ((coarse, "IL"), (refined, "AL")):
        assert _realized(graph) == (expected,) * 10
        assert len([record for record in records if record[0] is graph]) == (
            30 if graph is coarse else 60
        )
    _assert_uniform_runtime_coordinates(coarse, 0.25)
    _assert_uniform_runtime_coordinates(
        refined, 0.25 + refined.graph["GLYPH_FACTORS"]["AL_boost"]
    )
    assert coarse_memory["since_AL"] == dict.fromkeys(NODES, 3)
    assert coarse_memory["since_EN"] == dict.fromkeys(NODES, 3)
    assert refined_memory["since_AL"] == dict.fromkeys(NODES, 0)
    assert refined_memory["since_EN"] == dict.fromkeys(NODES, 2)


def test_internal_flow_subdivision_does_not_add_native_decisions(monkeypatch):
    coarse, coarse_memory = _runtime_source()
    subdivided, subdivided_memory = _runtime_source(subdivision_floor=0.125)
    assert prepare_integration_params(coarse, dt=0.25)[:2] == (0.25, 1)
    assert prepare_integration_params(subdivided, dt=0.25)[:2] == (0.125, 2)
    records = _capture_actual_native_glyphs(monkeypatch)
    for graph in (coarse, subdivided):
        for _ in range(3):
            runtime.step(graph, dt=0.25)
        _assert_uniform_runtime_coordinates(graph, 0.25)

    assert (
        _target_events(records, coarse)
        == _target_events(records, subdivided)
        == ((1.0, "IL"), (1.25, "IL"), (1.5, "IL"))
    )
    assert coarse.graph["_t"] == subdivided.graph["_t"] == 1.75
    assert (
        coarse_memory["since_AL"]
        == subdivided_memory["since_AL"]
        == dict.fromkeys(NODES, 3)
    )
    assert (
        coarse_memory["since_EN"]
        == subdivided_memory["since_EN"]
        == dict.fromkeys(NODES, 3)
    )


def test_fixed_decision_lattice_preserves_marks_but_not_all_runtime_bookkeeping(
    monkeypatch,
):
    coarse, coarse_memory = _runtime_source()
    refined, refined_memory = _runtime_source()
    records = _capture_actual_native_glyphs(monkeypatch)
    for _ in range(3):
        runtime.step(coarse, dt=0.25)
    for index in range(6):
        runtime.step(refined, dt=0.125, apply_glyphs=index % 2 == 0)

    assert _target_events(records, coarse) == _target_events(records, refined)
    assert coarse.graph["_t"] == refined.graph["_t"] == 1.75
    assert (
        coarse_memory["since_AL"]
        == refined_memory["since_AL"]
        == dict.fromkeys(NODES, 3)
    )
    assert (
        coarse_memory["since_EN"]
        == refined_memory["since_EN"]
        == dict.fromkeys(NODES, 3)
    )
    for graph in (coarse, refined):
        _assert_uniform_runtime_coordinates(graph, 0.25)
    # Disabling glyph selection is not disabling the other per-call policies.
    assert coarse.nodes[0]["stable_count"] == 3
    assert refined.nodes[0]["stable_count"] == 6
    assert len(coarse_memory["phase_state"]) == 3
    assert len(refined_memory["phase_state"]) == 6


def test_zero_duration_runtime_calls_still_consume_native_decision_batches(monkeypatch):
    graph, memory = _runtime_source()
    records = _capture_actual_native_glyphs(monkeypatch)
    for _ in range(6):
        runtime.step(graph, dt=0.0)

    assert graph.graph["_t"] == 1.0
    assert _target_events(records, graph) == tuple(
        (1.0, glyph) for glyph in ("IL", "IL", "IL", "EN", "IL", "AL")
    )
    _assert_uniform_runtime_coordinates(
        graph, 0.25 + graph.graph["GLYPH_FACTORS"]["AL_boost"]
    )
    assert memory["since_AL"] == dict.fromkeys(NODES, 0)
    assert memory["since_EN"] == dict.fromkeys(NODES, 2)
