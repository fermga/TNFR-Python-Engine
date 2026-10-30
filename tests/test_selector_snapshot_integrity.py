"""Prepared selection retains its graph and representable decision inputs."""

import gc
import pickle
import weakref
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DNFR, ALIAS_SI
from tnfr.dynamics import selectors
from tnfr.mathematics import unified_numerical


def _graph():
    graph = nx.empty_graph(3)
    inject_defaults(graph)
    graph.graph.update(
        SELECTOR_THRESHOLDS={"si_hi": 0.8, "si_lo": 0.2, "dnfr_hi": 0.5},
        SELECTOR_WEIGHTS={"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0},
        GLYPH_SELECTOR_MARGIN=0.0,
        GRAMMAR_CANON={"enabled": False},
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
    )
    for node in graph:
        graph.nodes[node].update(
            {ALIAS_SI[0]: 0.5, ALIAS_DNFR[0]: 0.0, ALIAS_D2EPI[0]: 0.0}
        )
    return graph


@pytest.mark.parametrize(
    "selector_type", [selectors.DefaultGlyphSelector, selectors.ParametricGlyphSelector]
)
@pytest.mark.parametrize("channel", [ALIAS_DNFR[0], ALIAS_D2EPI[0]])
@pytest.mark.parametrize("path", ["scalar", "array", "python"])
def test_nonzero_normalized_inputs_cannot_silently_become_zero(
    monkeypatch, selector_type, channel, path
):
    graph = _graph()
    graph.nodes[0][channel] = 5e-324
    graph.nodes[1][channel] = 1e308
    selector = selector_type()
    if path == "python":
        monkeypatch.setattr(selectors, "np", None)
        monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", False)
    with pytest.raises(ValueError, match="underflow"):
        if path == "scalar":
            selector(graph, 0)
        else:
            selector.prepare(graph, tuple(graph))
    assert selector._preselection is None


def test_positive_pressure_at_zero_cut_cannot_be_reclassified_as_zero():
    graph = _graph()
    graph.graph["SELECTOR_THRESHOLDS"].update(dnfr_hi=0.0, dnfr_lo=0.0)
    graph.nodes[0].update({ALIAS_SI[0]: 0.1, ALIAS_DNFR[0]: 5e-324})
    graph.nodes[1][ALIAS_DNFR[0]] = 1e308
    # The mathematical normalized pressure is positive, so the configured
    # strict comparison requests Dissonance. Rounding it to zero must not
    # silently request Mutation; the unavailable binary64 ratio rejects.
    with pytest.raises(ValueError, match="underflow"):
        selectors.DefaultGlyphSelector()(graph, 0)


@pytest.mark.parametrize("channel", ["dnfr", "accel"])
@pytest.mark.parametrize("numerator,denominator", [(5e-324, 1e308), (1e308, 5e-324)])
@pytest.mark.parametrize("path", ["scalar", "array", "python", "worker"])
def test_normalization_rejects_unrepresentable_outputs(
    monkeypatch, channel, numerator, denominator, path
):
    graph = _graph()
    aliases = ALIAS_DNFR if channel == "dnfr" else ALIAS_D2EPI
    graph.nodes[0][aliases[0]] = numerator
    norms = {"dnfr_max": 1.0, "accel_max": 1.0, f"{channel}_max": denominator}
    if path in {"python", "worker"}:
        monkeypatch.setattr(selectors, "np", None)
        monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", False)
    with pytest.raises(ValueError, match="overflows|underflows"):
        if path == "scalar":
            selectors._selector_normalized_metrics(graph.nodes[0], norms)
        elif path == "worker":
            selectors._selector_metrics_chunk(
                (
                    [0.5],
                    [numerator if channel == "dnfr" else 0.0],
                    [numerator if channel == "accel" else 0.0],
                    norms["dnfr_max"],
                    norms["accel_max"],
                )
            )
        else:
            selectors._collect_selector_metrics(graph, [0], norms)


@pytest.mark.parametrize("path", ["scalar", "array", "python", "worker"])
def test_valid_ratio_remains_unclipped_with_zero_and_unit_scale_conventions(
    monkeypatch, path
):
    graph = _graph()
    graph.nodes[0].update({ALIAS_DNFR[0]: -2.0, ALIAS_D2EPI[0]: 0.0})
    norms = {"dnfr_max": 0.0, "accel_max": 0.0}
    if path in {"python", "worker"}:
        monkeypatch.setattr(selectors, "np", None)
        monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", False)
    if path == "scalar":
        actual = selectors._selector_normalized_metrics(graph.nodes[0], norms)
    elif path == "worker":
        actual = tuple(
            values[0]
            for values in selectors._selector_metrics_chunk(
                ([0.5], [-2.0], [0.0], 1.0, 1.0)
            )
        )
    else:
        actual = selectors._collect_selector_metrics(graph, [0], norms)[0]
    assert actual == (0.5, 2.0, 0.0)


@pytest.mark.parametrize("prepared", [False, True])
def test_remote_normalizer_inputs_remain_admitted_outside_selected_targets(prepared):
    graph = _graph()
    graph.nodes[2][ALIAS_DNFR[0]] = True
    selector = selectors.DefaultGlyphSelector()
    with pytest.raises((TypeError, ValueError)):
        if prepared:
            selector.prepare(graph, [0])
        else:
            selector(graph, 0)


@pytest.mark.parametrize(
    "selector_type", [selectors.DefaultGlyphSelector, selectors.ParametricGlyphSelector]
)
def test_worker_payload_preserves_prepared_snapshot_without_global_rescans(
    monkeypatch, selector_type
):
    graph = _graph()
    selector = selector_type()
    selector.prepare(graph, tuple(graph))
    # Process workers serialize this payload. Graph identity must refer to the
    # reconstructed graph, not an integer address in the parent process.
    nodes, worker_graph, worker_selector, context = pickle.loads(
        pickle.dumps((list(graph), graph, selector, None))
    )
    for data in worker_graph.nodes.values():
        data[ALIAS_SI[0]] = 0.95

    with monkeypatch.context() as patch:
        patch.setattr(
            selectors,
            "_selector_norms",
            lambda *_: pytest.fail("prepared worker rescanned the entire graph"),
        )
        assert selectors._glyph_proposal_worker(
            (nodes, worker_graph, worker_selector, context)
        ) == [(node, "RA") for node in nodes]

    worker_selector.prepare(worker_graph, nodes)
    assert [worker_selector(worker_graph, node) for node in nodes] == ["IL"] * 3
    worker_selector.clear()
    assert worker_selector._preselection is None


@pytest.mark.parametrize(
    "selector_type", [selectors.DefaultGlyphSelector, selectors.ParametricGlyphSelector]
)
def test_process_worker_uses_the_prepared_decision_inputs(selector_type):
    graph = _graph()
    selector = selector_type()
    selector.prepare(graph, tuple(graph))
    for data in graph.nodes.values():
        data[ALIAS_SI[0]] = 0.95
    with ProcessPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            selectors._glyph_proposal_worker, (list(graph), graph, selector, None)
        ).result(timeout=30)
    assert result == [(node, "RA") for node in graph]


class _CustomDefault(selectors.DefaultGlyphSelector):
    def select(self, graph, node):
        return "SHA"


class _CustomParametric(selectors.ParametricGlyphSelector):
    def select(self, graph, node):
        return "SHA"


@pytest.mark.parametrize("selector_type", [_CustomDefault, _CustomParametric])
def test_serialized_batch_keeps_custom_selection_dispatch(selector_type):
    graph = _graph()
    selector = selector_type()
    selector.prepare(graph, tuple(graph))
    payload = pickle.loads(pickle.dumps((list(graph), graph, selector, None)))
    assert selectors._glyph_proposal_worker(payload) == [
        (node, "SHA") for node in graph
    ]


@pytest.mark.parametrize(
    "selector_type", [selectors.DefaultGlyphSelector, selectors.ParametricGlyphSelector]
)
def test_prepared_identity_is_graph_specific_and_clear_releases_it(selector_type):
    graph = _graph()
    selector = selector_type()
    selector.prepare(graph, tuple(graph))
    other = _graph()
    for data in other.nodes.values():
        data[ALIAS_SI[0]] = 0.95
    assert selector(other, 0) == "IL"
    assert selector(graph, 0) == "RA"
    reference = weakref.ref(graph)
    del graph
    assert reference() is not None
    selector.clear()
    gc.collect()
    assert reference() is None
