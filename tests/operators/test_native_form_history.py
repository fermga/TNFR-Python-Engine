"""Native jump/history contracts using finite supplied operator states.

No flow, producer or historical test fixture is imported. The independent
participation module owns the real three-batch default-selector roundtrip.
"""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import selectors
from tnfr.dynamics._operator_form_history import (
    capture_native_form_history,
    finish_native_form_history,
)
from tnfr.operators.grammar_debt import reset_grammar_state_from_history
from tnfr.types import require_finite_real_scalar_epi


def _source():
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(
        _t=2.0,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
        history={"since_AL": {0: 0, 1: 0}, "since_EN": {0: 0, 1: 0}},
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        VAL_scale=1.5,
        NUL_scale=0.5,
        NUL_densification_factor=2.0,
    )
    for node, epi in enumerate((0.5, 0.25)):
        data = graph.nodes[node]
        data.update(
            EPI=epi,
            epi_kind="form",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            Si=0.5,
            glyph_history=["AL", "IL"],
            epi_time_history=[(0.0, epi / 2.0), (2.0, epi)],
            epi_history=[epi / 2.0, epi],
        )
        reset_grammar_state_from_history(data)
    return graph


def _run(graph, glyph, *, traced=True):
    graph.graph["glyph_selector"] = lambda _graph, node: glyph if node == 0 else "IL"
    trace = [] if traced else None
    selectors.apply_native_operator_batch(
        graph, graph.graph["history"], decision_trace=trace
    )
    return trace


def _epi(graph, node=0):
    return require_finite_real_scalar_epi(graph.nodes[node]["EPI"], "test EPI")


@pytest.mark.parametrize("glyph", ["AL", "EN", "RA", "VAL", "NUL"])
def test_actual_native_form_writers_invalidate_only_changed_target(glyph):
    graph = _source()
    target = graph.nodes[0]
    old_own = target["epi_time_history"]
    peer_own = graph.nodes[1]["epi_time_history"]
    legacy = target["epi_history"]
    before = _epi(graph)
    trace = _run(graph, glyph)
    assert trace[0].applied_glyph == glyph
    assert _epi(graph) != before
    assert target["epi_time_history"] == ()
    assert target["epi_time_history"] is not old_own
    assert target["epi_history"] is legacy
    assert graph.nodes[1]["epi_time_history"] is peer_own
    assert trace[0].physical_history_invalidated is True
    assert trace[1].physical_history_invalidated is False
    assert graph.graph["_t"] == 2.0


@pytest.mark.parametrize("physical", ["absent", "none"])
def test_changed_native_form_does_not_create_physical_history(physical):
    graph = _source()
    target = graph.nodes[0]
    if physical == "absent":
        target.pop("epi_time_history")
    else:
        target["epi_time_history"] = None
    legacy = target["epi_history"]
    trace = _run(graph, "AL")
    assert _epi(graph) == 0.625
    assert ("epi_time_history" in target) is (physical == "none")
    assert target.get("epi_time_history") is None
    assert target["epi_history"] is legacy
    assert trace[0].physical_history_invalidated is False


@pytest.mark.parametrize("glyph", ["AL", "EN", "VAL", "NUL", "IL", "SHA"])
def test_numerically_unchanged_or_nonform_native_action_preserves_exact_object(glyph):
    graph = _source()
    if glyph == "AL":
        graph.nodes[0]["EPI"] = 1.0
    elif glyph == "EN":
        graph.nodes[1]["EPI"] = 0.5
    elif glyph in ("VAL", "NUL"):
        graph.graph["EDGE_AWARE_ENABLED"] = False
    # This deliberately invalid history must not be inspected by a no-op.
    physical = object()
    graph.nodes[0]["epi_time_history"] = physical
    before = _epi(graph)
    trace = _run(graph, glyph)
    assert trace[0].applied_glyph == glyph
    assert _epi(graph) == before
    assert graph.nodes[0]["epi_time_history"] is physical
    assert trace[0].physical_history_invalidated is False


@pytest.mark.parametrize(
    "glyph", ["IL", "SHA", "OZ", "UM", "THOL", "ZHIR", "NAV", "REMESH"]
)
def test_nonform_role_does_not_admit_unconsumed_scalar_or_history(glyph):
    data = {"EPI": object(), "epi_time_history": object()}
    before = dict(data)
    assert capture_native_form_history(data, glyph) is None
    assert finish_native_form_history(data, None) is False
    assert all(data[key] is value for key, value in before.items())


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, object()])
def test_form_boundary_admits_authoritative_scalar_before_dispatch(bad):
    data = {"EPI": bad, "epi": 0.5, "epi_time_history": ()}
    with pytest.raises((TypeError, ValueError)):
        capture_native_form_history(data, "AL")
    assert data["EPI"] is bad


def test_shared_history_alias_is_not_mutated_in_place():
    graph = _source()
    physical = graph.nodes[0]["epi_time_history"]
    graph.nodes[1]["epi_time_history"] = physical
    original = list(physical)
    _run(graph, "AL")
    assert graph.nodes[0]["epi_time_history"] == ()
    assert graph.nodes[1]["epi_time_history"] is physical
    assert physical == original


def test_signed_zero_is_not_a_fictitious_form_jump():
    physical = [(0.0, -0.0), (1.0, -0.0)]
    data = {"EPI": -0.0, "epi_time_history": physical}
    before = capture_native_form_history(data, "EN")
    data["EPI"] = 0.0
    assert finish_native_form_history(data, before) is False
    assert data["epi_time_history"] is physical


@pytest.mark.parametrize("new_epi", [0.75, math.nan, object()])
def test_failed_partial_dispatch_invalidates_without_masking_error(
    monkeypatch, new_epi
):
    graph = _source()
    original = RuntimeError("declared partial dispatch")

    def partial(graph, node, glyph, **kwargs):
        graph.nodes[node]["EPI"] = new_epi
        raise original

    monkeypatch.setattr(selectors, "apply_glyph", partial)
    trace = []
    graph.graph["glyph_selector"] = lambda _graph, _node: "AL"
    with pytest.raises(RuntimeError) as caught:
        selectors.apply_native_operator_batch(
            graph, graph.graph["history"], decision_trace=trace
        )
    assert caught.value is original
    assert graph.nodes[0]["epi_time_history"] == ()
    assert trace[0].status == "failed" and trace[0].applied_glyph is None
    assert trace[0].physical_history_invalidated is True
    assert trace[0].failure_stage == "dispatch"
    assert trace[1].status == "not_reached"


def test_failed_dispatch_without_form_write_preserves_history(monkeypatch):
    graph = _source()
    physical = graph.nodes[0]["epi_time_history"]
    original = RuntimeError("declared rejection")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(selectors, "apply_glyph", fail)
    with pytest.raises(RuntimeError) as caught:
        _run(graph, "AL")
    assert caught.value is original
    assert graph.nodes[0]["epi_time_history"] is physical


def test_cleanup_failure_retains_original_dispatch_error_and_reports_limit(monkeypatch):
    graph = _source()
    original = RuntimeError("primary dispatch failure")

    def partial(graph, node, glyph, **kwargs):
        graph.nodes[node]["EPI"] = 0.75
        raise original

    def cleanup_failure(*args):
        raise TypeError("external cleanup failure")

    monkeypatch.setattr(selectors, "apply_glyph", partial)
    monkeypatch.setattr(selectors, "finish_native_form_history", cleanup_failure)
    trace = []
    graph.graph["glyph_selector"] = lambda _graph, _node: "AL"
    with pytest.raises(RuntimeError) as caught:
        selectors.apply_native_operator_batch(
            graph, graph.graph["history"], decision_trace=trace
        )
    assert caught.value is original
    if hasattr(original, "add_note"):
        assert any("history cleanup failed" in note for note in original.__notes__)
    assert trace[0].physical_history_invalidated is None
    assert trace[0].applied_glyph is None


def test_live_grammar_replacement_role_owns_cleanup(monkeypatch):
    graph = _source()
    monkeypatch.setattr(selectors, "enforce_canonical_grammar", lambda *a, **kw: "AL")
    trace = _run(graph, "IL")
    assert trace[0].requested_glyph == "IL"
    assert trace[0].applied_glyph == "AL"
    assert trace[0].physical_history_invalidated is True
    assert graph.nodes[0]["epi_time_history"] == ()


def test_cleanup_uses_actual_postdispatch_endpoint_not_requested_factor(monkeypatch):
    graph = _source()
    physical = graph.nodes[0]["epi_time_history"]
    monkeypatch.setattr(selectors, "apply_glyph", lambda *a, **kw: None)
    trace = _run(graph, "AL")
    assert trace[0].applied_glyph == "AL"
    assert trace[0].physical_history_invalidated is False
    assert graph.nodes[0]["epi_time_history"] is physical


def test_optional_trace_does_not_control_invalidation_or_native_state():
    traced, plain = _source(), _source()
    for glyph in ("AL", "EN"):
        _run(traced, glyph)
        _run(plain, glyph, traced=False)
    assert dict(traced.nodes(data=True)) == dict(plain.nodes(data=True))
    assert traced.graph["history"] == plain.graph["history"]
    assert traced.nodes[0]["epi_time_history"] == ()


def test_committed_prefix_remains_invalidated_after_later_target_failure(monkeypatch):
    graph = _source()
    canonical = selectors.apply_glyph
    peer_history = graph.nodes[1]["epi_time_history"]

    def prefix(graph, node, glyph, **kwargs):
        if node == 1:
            raise RuntimeError("later target rejection")
        canonical(graph, node, glyph, **kwargs)

    monkeypatch.setattr(selectors, "apply_glyph", prefix)
    with pytest.raises(RuntimeError, match="later target rejection"):
        _run(graph, "AL")
    assert _epi(graph) == 0.625
    assert graph.nodes[0]["epi_time_history"] == ()
    assert graph.nodes[1]["epi_time_history"] is peer_history


def test_old_consumption_and_legacy_memory_are_not_erased():
    graph = _source()
    data = graph.nodes[0]
    consumed = (("epi_time_history", 0.0, 2.0),)
    data["_native_zhir_consumed_intervals"] = consumed
    legacy = deepcopy(data["epi_history"])
    _run(graph, "AL")
    assert data["_native_zhir_consumed_intervals"] is consumed
    assert data["epi_history"] == legacy
