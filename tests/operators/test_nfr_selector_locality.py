"""Native occurrence can consume disconnected diagnostic normalization state.

Two supplied disjoint edges isolate the default selector's graph-wide capacity
normalizer. Local pressure is freshly evaluated by the unchanged default law;
no pressure sample, requested glyph, admission result or event map is forged.
The zero-duration step isolates an actual jump before the runtime's later
phase/capacity policies. This is a configured-controller counterexample, not
physical remote interaction or an autonomous occurrence theorem.
"""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_VF
from tnfr.dynamics import runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.glyph_history import push_glyph
from tnfr.metrics import sense_index
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.grammar_dynamics import validate_candidate

LOCAL = (0, 1)
NODES = (0, 1, 2, 3)


def _source(remote_capacity):
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(((0, 1), (2, 3)), weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=1.0,
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
    )
    for node, capacity in zip(NODES, (1.0, 1.0, *remote_capacity), strict=True):
        graph.nodes[node].update(
            EPI=0.5,
            nu_f=capacity,
            theta=3 * math.pi / 4 if node == 1 else 0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            glyph_history=[],
            epi_time_history=[(0.75, 0.5), (1.0, 0.5)],
        )
        # Supplied, admitted context/evidence, not reconstructed past execution.
        for glyph in ("AL", "IL", "EN"):
            push_glyph(graph.nodes[node], glyph, 8)
    graph.graph["history"] = {
        "since_AL": dict.fromkeys(NODES, 0),
        "since_EN": dict.fromkeys(NODES, 0),
    }
    assert graph.graph["GRAMMAR_CANON"]["enabled"] is True
    assert graph.graph["PHASE_ADAPT"]["enabled"] is True
    assert not graph.graph.get("MATH_ENGINE", {}).get("enabled", False)
    assert not graph.graph.get("callbacks")
    assert graph.graph["AL_MAX_LAG"] == 5
    assert graph.graph["EN_MAX_LAG"] == 3
    assert all(graph.degree(node) == 1 for node in NODES)
    assert set(nx.node_connected_component(graph, 0)) == set(LOCAL)
    assert max(remote_capacity) < graph.graph["VF_MAX"]
    return graph


def _physical_row(graph, node):
    attrs = graph.nodes[node]
    return (
        get_attr(attrs, ALIAS_EPI),
        get_attr(attrs, ALIAS_VF),
        attrs["theta"],
        get_attr(attrs, ALIAS_DNFR),
    )


def _refresh(graph):
    default_compute_delta_nfr(graph)
    profile = {}
    sense_index.compute_Si(graph, inplace=True, profile=profile)
    return profile["path"]


def _capture_actual_events(monkeypatch):
    actual = selectors.apply_glyph
    records = []

    def capture(graph, node, glyph, **kwargs):
        before = _physical_row(graph, node)
        context = read_grammar_state(graph.nodes[node])
        time = graph.graph["_t"]
        actual(graph, node, glyph, **kwargs)
        records.append(
            (
                graph,
                node,
                time,
                before,
                _physical_row(graph, node),
                context,
                graph.nodes[node]["glyph_history"][-1],
            )
        )

    monkeypatch.setattr(selectors, "apply_glyph", capture)
    return records


def _events(records, graph):
    return tuple(record[1:] for record in records if record[0] is graph)


@pytest.mark.parametrize("scalar_si", [False, True])
def test_disconnected_capacity_changes_an_actual_admitted_local_event(
    monkeypatch, scalar_si
):
    if scalar_si:
        monkeypatch.setattr(sense_index, "np", None)
    quiet = _source((1.0, 1.0))
    remote = _source((4.0, 4.0))
    assert quiet.graph == remote.graph
    assert tuple(quiet.edges(data=True)) == tuple(remote.edges(data=True))
    assert all(quiet.nodes[node] == remote.nodes[node] for node in LOCAL)
    local_history = deepcopy(quiet.nodes[0]["epi_time_history"])
    context = read_grammar_state(quiet.nodes[0])

    for graph in (quiet, remote):
        assert _refresh(graph) == ("fallback" if scalar_si else "vectorized")
    weight = quiet.graph["_DNFR_META"]["weights_effective"]["phase"]
    pressure = 0.75 * weight
    # Only the phase channel is nonzero: all edge-local forms and capacities
    # agree, and the remote phases agree. No zero-degree fallback is involved.
    expected = (pressure, -pressure, 0.0, 0.0)
    for graph in (quiet, remote):
        assert tuple(get_attr(graph.nodes[n], ALIAS_DNFR) for n in NODES) == (
            pytest.approx(expected, abs=2e-16, rel=0)
        )
        assert graph.graph["_dnfrmax"] == pytest.approx(pressure, abs=2e-16, rel=0)
        assert all(validate_candidate(graph, node, "IL").allowed for node in LOCAL)
        assert all(validate_candidate(graph, node, "OZ").allowed for node in LOCAL)
    assert tuple(_physical_row(quiet, n) for n in LOCAL) == tuple(
        _physical_row(remote, n) for n in LOCAL
    )
    assert (quiet.graph["_vfmax"], remote.graph["_vfmax"]) == (1.0, 4.0)

    alpha, beta, _gamma = sense_index.get_Si_weights(quiet)
    thresholds = quiet.graph["SELECTOR_THRESHOLDS"]
    for graph, maximum, mark in ((quiet, 1.0, "IL"), (remote, 4.0, "OZ")):
        expected_si = alpha / maximum + beta / 4
        assert tuple(get_attr(graph.nodes[n], ALIAS_SI) for n in LOCAL) == (
            pytest.approx((expected_si, expected_si), abs=2e-16, rel=0)
        )
        if mark == "IL":
            assert expected_si > thresholds["si_hi"]
        else:
            assert expected_si < thresholds["si_lo"]
            assert 1.0 > thresholds["dnfr_hi"]
        assert tuple(selectors.DefaultGlyphSelector()(graph, n) for n in LOCAL) == (
            mark,
            mark,
        )

    records = _capture_actual_events(monkeypatch)
    for graph, mark in ((quiet, "IL"), (remote, "OZ")):
        runtime.step(graph, dt=0.0)
        events = _events(records, graph)
        assert tuple(event[0] for event in events) == NODES
        assert tuple(event[-1] for event in events) == (mark, mark, "IL", "IL")
        factor = graph.graph["GLYPH_FACTORS"][f"{mark}_dnfr_factor"]
        for event, signed_pressure in zip(events[:2], expected[:2], strict=True):
            node, time, before, after, prior_context, _mark = event
            assert time == graph.graph["_t"] == 1.0
            assert prior_context == context
            assert before[-1] == pytest.approx(signed_pressure, abs=2e-16, rel=0)
            assert after[:-1] == before[:-1]
            assert after[-1] == pytest.approx(
                signed_pressure * factor, abs=2e-16, rel=0
            )
            assert list(graph.nodes[node]["epi_time_history"]) == local_history
            assert graph.graph["history"]["since_AL"][node] == 1
            assert graph.graph["history"]["since_EN"][node] == 1
        # Native phase policies still execute after the captured jump. The
        # event comparison above does not freeze the complete runtime state.
        assert graph.nodes[0]["theta"] != 0.0
        assert get_attr(graph.nodes[0], ALIAS_EPI) == 0.5
        assert get_attr(graph.nodes[0], ALIAS_VF) == 1.0
    assert abs(_events(records, quiet)[0][3][-1]) < pressure
    assert abs(_events(records, remote)[0][3][-1]) > pressure


def test_matched_fresh_global_normalizers_preserve_the_immediate_local_event(
    monkeypatch,
):
    uniform = _source((4.0, 4.0))
    contrast = _source((2.0, 4.0))
    assert all(uniform.nodes[node] == contrast.nodes[node] for node in LOCAL)
    for graph in (uniform, contrast):
        _refresh(graph)
    weights = contrast.graph["_DNFR_META"]["weights_effective"]
    remote_pressure = 2 * weights["vf"]
    local_pressure = 0.75 * weights["phase"]
    assert 0 < remote_pressure < local_pressure
    assert tuple(get_attr(uniform.nodes[n], ALIAS_DNFR) for n in (2, 3)) == (0.0, 0.0)
    assert tuple(get_attr(contrast.nodes[n], ALIAS_DNFR) for n in (2, 3)) == (
        pytest.approx((remote_pressure, -remote_pressure), abs=2e-16, rel=0)
    )
    assert uniform.graph["_vfmax"] == contrast.graph["_vfmax"] == 4.0
    assert uniform.graph["_dnfrmax"] == contrast.graph["_dnfrmax"]
    assert all(
        _physical_row(uniform, n) == _physical_row(contrast, n)
        and get_attr(uniform.nodes[n], ALIAS_SI)
        == get_attr(contrast.nodes[n], ALIAS_SI)
        for n in LOCAL
    )
    records = _capture_actual_events(monkeypatch)
    for graph in (uniform, contrast):
        runtime.step(graph, dt=0.0)
    assert _events(records, uniform)[:2] == _events(records, contrast)[:2]
    assert tuple(event[-1] for event in _events(records, uniform)[:2]) == ("OZ", "OZ")


def test_without_selection_disconnected_capacity_leaves_local_pressure_unchanged():
    quiet = _source((1.0, 1.0))
    remote = _source((4.0, 4.0))
    for graph in (quiet, remote):
        runtime.step(graph, dt=0.0, apply_glyphs=False)
        assert all(
            tuple(graph.nodes[n]["glyph_history"]) == ("AL", "IL", "EN") for n in LOCAL
        )
        assert graph.graph["history"]["since_AL"] == dict.fromkeys(NODES, 0)
        assert graph.graph["history"]["since_EN"] == dict.fromkeys(NODES, 0)
    assert tuple(_physical_row(quiet, n) for n in LOCAL) == tuple(
        _physical_row(remote, n) for n in LOCAL
    )
    pressure = 0.75 * quiet.graph["_DNFR_META"]["weights_effective"]["phase"]
    assert get_attr(quiet.nodes[0], ALIAS_DNFR) == pytest.approx(
        pressure, abs=2e-16, rel=0
    )
