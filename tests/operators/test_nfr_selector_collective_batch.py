"""Restricted complete native batches descend on retained doubled-C5 blocks.

The sources, lag profiles and checks below are fixed analytical controls, not
search results or a trajectory campaign. Fraction arithmetic evaluates the
ideal real five-block recurrence independently of the native dispatcher. The
binary64 comparison budget is absolute 1e-14; exact twin equality, event marks,
history and bookkeeping have no tolerance. Every batch uses fresh default
pressure/Si, followed by actual strict grammar and direct primitive glyphs.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics import selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.selectors import DefaultGlyphSelector, _apply_glyphs
from tnfr.glyph_history import push_glyph
from tnfr.metrics import sense_index
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.types import real_scalar_epi

NODES = tuple(range(10))
PAIRS = tuple((2 * block, 2 * block + 1) for block in range(5))
ABS_ERROR = 1e-14
FORM_CASES = (
    (0.125, 0.375, 0.125, 0.125, 0.125),
    (-0.375, 0.25, -0.125, 0.5, 0.0),
    (0.96875, 0.375, 0.125, 0.25, 0.5),
)
PROFILES = (("IL",) * 5, ("EN",) * 5, ("AL", "EN", "IL", "EN", "AL"))


def _source(forms, marks, *, order=NODES):
    graph = nx.Graph()
    graph.add_nodes_from(order)
    for block in range(5):
        for node in PAIRS[block]:
            for neighbor in PAIRS[(block + 1) % 5]:
                graph.add_edge(node, neighbor, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(_t=1.0, RANDOM_SEED=17, GRAMMAR_REJECTION_MODE="raise")
    memory = {"since_AL": {}, "since_EN": {}}
    for node in graph:
        block = node // 2
        data = graph.nodes[node]
        data.update(
            EPI=forms[block],
            epi_kind="seed",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            glyph_history=[],
        )
        for glyph in ("AL", "IL", "EN"):
            # Supplied, bounded context; this does not replay a physical past.
            push_glyph(data, glyph, 1)
        data[U2_DEBT_KEY] = block % 2
        memory["since_AL"][node] = 5 if marks[block] == "AL" else 0
        # Both limits are due on AL blocks: Emission has actual priority.
        memory["since_EN"][node] = 3 if marks[block] in {"AL", "EN"} else 0
    return graph, memory


def _form(graph, node):
    return real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _physical_row(graph, node):
    data = graph.nodes[node]
    return (_form(graph, node), *(data[key] for key in ("delta_nfr", "theta", "nu_f")))


def _pressure(forms, coefficient):
    values = tuple(map(Q, forms))
    return tuple(
        coefficient * ((values[(block - 1) % 5] + values[(block + 1) % 5]) / 2 - x)
        for block, x in enumerate(values)
    )


def _normalized_weight(graph, key, channel):
    weights = {name: Q(value) for name, value in graph.graph[key].items()}
    return weights[channel] / sum(weights.values())


def _refresh(graph):
    default_compute_delta_nfr(graph)
    sense_index.compute_Si(graph, inplace=True)


def _assert_fresh_observations(graph, forms):
    pressure = _pressure(forms, _normalized_weight(graph, "DNFR_WEIGHTS", "epi"))
    maximum = max(map(abs, pressure))
    alpha, beta, gamma = (
        _normalized_weight(graph, "SI_WEIGHTS", key)
        for key in ("alpha", "beta", "gamma")
    )
    for block, pair in enumerate(PAIRS):
        ratio = abs(pressure[block]) / maximum if maximum else Q(0)
        expected_si = alpha + beta + gamma * (1 - ratio)
        for node in pair:
            assert graph.nodes[node]["delta_nfr"] == pytest.approx(
                float(pressure[block]), abs=ABS_ERROR, rel=0
            )
            assert graph.nodes[node]["Si"] == pytest.approx(
                float(expected_si), abs=ABS_ERROR, rel=0
            )
            assert (
                graph.nodes[node]["Si"] >= graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]
            )
    selector = DefaultGlyphSelector()
    selector.prepare(graph, tuple(graph))
    assert tuple(selector(graph, node) for node in graph) == ("IL",) * 10
    selector.clear()
    return pressure


def _block_oracle(graph, forms, marks):
    """Ideal real ordered block map; no native glyph or realization kernel."""
    values = list(map(Q, forms))
    pressure = list(_pressure(forms, _normalized_weight(graph, "DNFR_WEIGHTS", "epi")))
    factors = graph.graph["GLYPH_FACTORS"]
    boost, mix, retention = (
        Q(factors[key]) for key in ("AL_boost", "EN_mix", "IL_dnfr_factor")
    )
    lower, upper = Q(graph.graph["EPI_MIN"]), Q(graph.graph["EPI_MAX"])
    for block, mark in enumerate(marks):
        if mark == "AL":
            values[block] = min(upper, max(lower, values[block] + boost))
        elif mark == "EN":
            mean = (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2
            values[block] = (1 - mix) * values[block] + mix * mean
        else:
            assert mark == "IL"
            pressure[block] *= retention
    return tuple(values), tuple(pressure)


def _capture_admitted_prefixes(monkeypatch):
    actual = selectors.apply_glyph
    records = []

    def capture(graph, node, glyph, **kwargs):
        # This observes each actual fine-event boundary, not just initial state.
        assert validate_candidate(graph, node, glyph).allowed
        actual(graph, node, glyph, **kwargs)
        records.append(
            (
                node,
                glyph,
                tuple(_physical_row(graph, target) for target in NODES),
            )
        )

    monkeypatch.setattr(selectors, "apply_glyph", capture)
    return records


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("forms", FORM_CASES)
@pytest.mark.parametrize("marks", PROFILES)
def test_complete_contiguous_batch_matches_five_block_map(
    forms, marks, scalar, monkeypatch
):
    graph, memory = _source(forms, marks)
    if scalar:
        graph.graph["vectorized_dnfr"] = False
        monkeypatch.setattr(sense_index, "np", None)
        monkeypatch.setattr(selectors, "np", None)
    assert graph.graph["CLIP_MODE"] == "hard"
    assert graph.graph["GRAMMAR_CANON"]["enabled"] is True
    assert graph.graph["GLYPH_HYSTERESIS_WINDOW"] == 7
    _refresh(graph)
    initial_pressure = _assert_fresh_observations(graph, forms)
    initial_si = tuple(graph.nodes[node]["Si"] for node in NODES)
    expected_forms, expected_pressure = _block_oracle(graph, forms, marks)
    records = _capture_admitted_prefixes(monkeypatch)
    selector = DefaultGlyphSelector()

    _apply_glyphs(graph, selector, memory)

    assert tuple((node, glyph) for node, glyph, _ in records) == tuple(
        (node, marks[node // 2]) for node in NODES
    )
    assert selector._preselection is selector._prepared_graph is None
    # Fine events can leave the pair stratum; every completed pair returns.
    assert any(rows[node] != rows[node + 1] for node, _, rows in records[::2])
    for _, _, rows in records[1::2]:
        assert all(rows[first] == rows[second] for first, second in PAIRS)
    for block, pair in enumerate(PAIRS):
        mark = marks[block]
        expected_lags = (0, 3) if mark == "AL" else (1, 0 if mark == "EN" else 1)
        for node in pair:
            data = graph.nodes[node]
            assert _form(graph, node) == pytest.approx(
                float(expected_forms[block]), abs=ABS_ERROR, rel=0
            )
            assert data["delta_nfr"] == pytest.approx(
                float(expected_pressure[block]), abs=ABS_ERROR, rel=0
            )
            assert (data["theta"], data["nu_f"], data["d2EPI"]) == (0.0, 1.0, 0.0)
            assert data["epi_kind"] == "seed"
            assert data["source_glyph"] == mark
            assert tuple(data["glyph_history"]) == ("EN", mark)
            assert data[OPERATOR_STEP_KEY] == 4
            assert data[PRIOR_COHERENCE_KEY] is True
            assert data[U2_DEBT_KEY] == (0 if mark == "IL" else block % 2)
            assert (memory["since_AL"][node], memory["since_EN"][node]) == expected_lags
            assert "_reception_sources" not in data
            assert "emission_timestamp" not in data
        assert read_grammar_state(graph.nodes[pair[0]]) == read_grammar_state(
            graph.nodes[pair[1]]
        )
    assert graph.graph["_t"] == 1.0
    assert graph.number_of_edges() == 20
    assert all(data["weight"] == 1.0 for _, _, data in graph.edges(data=True))
    assert tuple(graph.nodes[node]["Si"] for node in NODES) == initial_si
    if marks[0] == "AL" and forms[0] == 0.96875:
        assert _form(graph, 0) == _form(graph, 1) == 1.0

    # Stored post-event pressure is not silently identified with the fresh law.
    raw_pressure = tuple(graph.nodes[node]["delta_nfr"] for node in range(0, 10, 2))
    actual_forms = tuple(_form(graph, node) for node in range(0, 10, 2))
    _refresh(graph)
    refreshed_pressure = _assert_fresh_observations(graph, actual_forms)
    assert max(
        abs(Q(value) - target)
        for value, target in zip(raw_pressure, refreshed_pressure)
    ) > Q(1, 10000)
    if marks == ("IL",) * 5:
        assert refreshed_pressure == initial_pressure


def test_sequential_reception_is_not_an_atomic_neighbor_snapshot():
    forms = FORM_CASES[0]
    graph, memory = _source(forms, ("EN",) * 5)
    _refresh(graph)
    expected_forms, _ = _block_oracle(graph, forms, ("EN",) * 5)
    mix = Q(graph.graph["GLYPH_FACTORS"]["EN_mix"])
    # The preceding block's new form enters Reception of block one.
    jacobi_block_one = Q(forms[1]) * (1 - mix) + mix * Q(forms[0])
    assert expected_forms[1] - jacobi_block_one == mix**2 / 16

    _apply_glyphs(graph, DefaultGlyphSelector(), memory)

    assert _form(graph, 2) == pytest.approx(
        float(expected_forms[1]), abs=ABS_ERROR, rel=0
    )
    assert _form(graph, 2) - float(jacobi_block_one) > 0.003


def test_interleaved_reception_splits_twins_despite_equal_requests_and_memory():
    order = (0, 2, 1, 3, 4, 5, 6, 7, 8, 9)
    graph, memory = _source(FORM_CASES[0], ("EN",) * 5, order=order)
    _refresh(graph)
    _assert_fresh_observations(graph, FORM_CASES[0])
    mix = Q(graph.graph["GLYPH_FACTORS"]["EN_mix"])
    predicted_gap = mix**2 * (mix - 8) / 128
    assert predicted_gap < 0

    _apply_glyphs(graph, DefaultGlyphSelector(), memory)

    assert tuple(graph.nodes[node]["source_glyph"] for node in NODES) == ("EN",) * 10
    assert _form(graph, 1) - _form(graph, 0) == pytest.approx(
        float(predicted_gap), abs=ABS_ERROR, rel=0
    )
    assert _form(graph, 1) != _form(graph, 0)
    assert read_grammar_state(graph.nodes[0]) == read_grammar_state(graph.nodes[1])
    assert memory["since_AL"][0] == memory["since_AL"][1] == 1
    assert memory["since_EN"][0] == memory["since_EN"][1] == 0


def test_live_grammar_failure_preserves_executed_prefix_without_a_batch_return():
    forms = (0.125, 0.0, 0.25, 0.375, 0.5)
    graph, memory = _source(forms, ("AL", "IL", "IL", "IL", "IL"))
    for node in PAIRS[1]:
        graph.nodes[node]["glyph_history"] = []
    _refresh(graph)
    _assert_fresh_observations(graph, forms)
    before = tuple(_physical_row(graph, node) for node in NODES)
    boost = graph.graph["GLYPH_FACTORS"]["AL_boost"]
    selector = DefaultGlyphSelector()
    assert validate_candidate(graph, 0, "AL").allowed
    assert not validate_candidate(graph, 2, "IL").allowed

    with pytest.raises(StructuralGrammarError, match="requires a generator") as error:
        _apply_glyphs(graph, selector, memory)

    assert error.value.rule == "U1a"
    assert _form(graph, 0) == _form(graph, 1) == forms[0] + boost
    assert tuple(_physical_row(graph, node) for node in NODES[2:]) == before[2:]
    assert tuple(graph.nodes[node]["source_glyph"] for node in PAIRS[0]) == ("AL", "AL")
    assert all("source_glyph" not in graph.nodes[node] for node in NODES[2:])
    # All counters increment before live execution; only completed AL resets.
    assert memory["since_AL"] == {node: 0 if node < 2 else 1 for node in NODES}
    assert memory["since_EN"] == {node: 3 if node < 2 else 1 for node in NODES}
    assert selector._preselection is selector._prepared_graph is None
    assert graph.graph["_t"] == 1.0


@pytest.mark.parametrize(
    "forms,mark,endpoint,lags",
    [
        ((0.0,) * 5, "EN", 0.0, (1, 0)),
        ((31 / 32, 63 / 64, 1.0, 15 / 16, 127 / 128), "AL", 1.0, (0, 3)),
    ],
)
def test_uniform_boundary_return_retains_admitted_neutral_debt(
    forms, mark, endpoint, lags, monkeypatch
):
    graph, memory = _source(forms, (mark,) * 5)
    for node in graph:
        graph.nodes[node][U2_DEBT_KEY] = 2
    _refresh(graph)
    _assert_fresh_observations(graph, forms)
    initial_pressure = tuple(graph.nodes[node]["delta_nfr"] for node in NODES)
    initial_si = tuple(graph.nodes[node]["Si"] for node in NODES)
    if mark == "AL":
        assert any(value != 0.0 for value in initial_pressure)
    records = _capture_admitted_prefixes(monkeypatch)

    _apply_glyphs(graph, DefaultGlyphSelector(), memory)

    assert tuple((node, glyph) for node, glyph, _ in records) == tuple(
        (node, mark) for node in NODES
    )
    assert tuple(_form(graph, node) for node in NODES) == (endpoint,) * 10
    assert tuple(graph.nodes[node]["delta_nfr"] for node in NODES) == initial_pressure
    assert tuple(graph.nodes[node]["Si"] for node in NODES) == initial_si
    for node in NODES:
        data = graph.nodes[node]
        assert data["epi_kind"] == "seed"
        assert data["source_glyph"] == mark
        assert tuple(data["glyph_history"]) == ("EN", mark)
        assert data[U2_DEBT_KEY] == 2
        assert data[OPERATOR_STEP_KEY] == 4
        assert data[PRIOR_COHERENCE_KEY] is True
        assert (memory["since_AL"][node], memory["since_EN"][node]) == lags

    _refresh(graph)

    assert _assert_fresh_observations(graph, (endpoint,) * 5) == (Q(0),) * 5
    assert tuple(graph.nodes[node]["delta_nfr"] for node in NODES) == (0.0,) * 10
    assert tuple(graph.nodes[node]["Si"] for node in NODES) == pytest.approx(
        (1.0,) * 10, abs=ABS_ERROR, rel=0
    )
    assert graph.graph["_t"] == 1.0
