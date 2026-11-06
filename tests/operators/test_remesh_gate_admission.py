"""Stability evidence admission before the separate delayed REMESH operation."""

from collections import deque
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators import remesh
from tnfr.operators.preconditions import OperatorPreconditionError

_TINY = Fraction(1, 2**1200)
_ERRORS = (TNFRValueError, OperatorPreconditionError)


def _graph():
    graph = nx.path_graph(2)
    graph.graph.update(
        REMESH_STABILITY_WINDOW=1,
        REMESH_REQUIRE_STABILITY=True,
        REMESH_COOLDOWN_WINDOW=0,
        REMESH_COOLDOWN_TS=0,
        FRACTION_STABLE_REMESH=0.5,
        history={"stable_frac": [1.0]},
        _t=2.0,
    )
    return graph


@pytest.fixture
def requests(monkeypatch):
    calls = []

    def capture(graph):
        calls.append(graph)
        return SimpleNamespace(applied=False)

    monkeypatch.setattr(remesh, "apply_network_remesh", capture)
    return calls


def _assert_rejected_without_writes(graph, calls, *, match, **kwargs):
    before = deepcopy(graph)
    with pytest.raises(_ERRORS, match=match):
        remesh.apply_remesh_if_globally_stable(graph, **kwargs)
    assert not calls
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("value", (0, -1, True, False, 1.5, "1", float("nan"), _TINY))
@pytest.mark.parametrize("explicit", (False, True))
def test_stability_window_requires_positive_integer(value, explicit, requests):
    graph = _graph()
    graph.graph["history"]["stable_frac"] = []
    kwargs = {"stable_step_window": value} if explicit else {}
    if not explicit:
        graph.graph["REMESH_STABILITY_WINDOW"] = value
    _assert_rejected_without_writes(
        graph,
        requests,
        match="stable_step_window|REMESH_STABILITY_WINDOW",
        **kwargs,
    )


@pytest.mark.parametrize("value", (0, 1, "false", None))
def test_optional_gate_flag_requires_boolean(value, requests):
    graph = _graph()
    graph.graph["REMESH_REQUIRE_STABILITY"] = value
    _assert_rejected_without_writes(graph, requests, match="REMESH_REQUIRE_STABILITY")


@pytest.mark.parametrize(
    "key",
    (
        "REMESH_MIN_PHASE_SYNC",
        "REMESH_MAX_GLYPH_DISR",
        "REMESH_MIN_SIGMA_MAG",
        "REMESH_MIN_KURAMOTO_R",
        "REMESH_MIN_SI_HI_FRAC",
        "REMESH_COOLDOWN_TS",
        "FRACTION_STABLE_REMESH",
        "_t",
    ),
)
@pytest.mark.parametrize("value", (True, float("inf"), float("nan"), _TINY))
def test_consumed_gate_scalars_reject_invalid_raw_values(key, value, requests):
    graph = _graph()
    graph.graph[key] = value
    _assert_rejected_without_writes(graph, requests, match=key)


@pytest.mark.parametrize("value", (-1, True, 1.5, "2"))
def test_cooldown_window_is_nonnegative_integer(value, requests):
    graph = _graph()
    graph.graph["REMESH_COOLDOWN_WINDOW"] = value
    _assert_rejected_without_writes(graph, requests, match="REMESH_COOLDOWN_WINDOW")


def test_negative_physical_cooldown_rejects(requests):
    graph = _graph()
    graph.graph["REMESH_COOLDOWN_TS"] = -1
    _assert_rejected_without_writes(graph, requests, match="REMESH_COOLDOWN_TS")


@pytest.mark.parametrize(
    "history", (None, [], {"stable_frac": None}, {"stable_frac": "1"})
)
def test_present_malformed_history_is_not_missing_evidence(history, requests):
    graph = _graph()
    graph.graph["history"] = history
    _assert_rejected_without_writes(graph, requests, match="history")


@pytest.mark.parametrize(
    "key",
    (
        "stable_frac",
        "phase_sync",
        "glyph_load_disr",
        "sense_sigma_mag",
        "kuramoto_R",
        "Si_hi_frac",
    ),
)
@pytest.mark.parametrize("value", (True, float("inf"), float("nan"), _TINY))
def test_present_selected_evidence_is_finite_nonboolean(key, value, requests):
    graph = _graph()
    graph.graph["history"][key] = [value]
    _assert_rejected_without_writes(graph, requests, match=key)


@pytest.mark.parametrize("history", ("absent", {}, {"stable_frac": []}))
def test_missing_stability_history_abstains_without_creating_samples(history, requests):
    graph = _graph()
    if history == "absent":
        graph.graph.pop("history")
    else:
        graph.graph["history"] = history
    before = deepcopy(graph)
    remesh.apply_remesh_if_globally_stable(graph)
    assert not requests
    assert nx.utils.graphs_equal(graph, before)


def test_zero_sample_is_observed_and_distinct_from_no_sample(requests):
    graph = _graph()
    graph.graph["FRACTION_STABLE_REMESH"] = 0
    graph.graph["history"]["stable_frac"] = [0.0]
    remesh.apply_remesh_if_globally_stable(graph)
    assert len(requests) == 1
    assert "_last_remesh_step" not in graph.graph


@pytest.mark.parametrize("series", ([0.1, 1.0, 1.0], deque((0.1, 1.0, 1.0), maxlen=3)))
def test_selected_tail_and_explicit_window_preserve_valid_admission(series, requests):
    graph = _graph()
    graph.graph["REMESH_STABILITY_WINDOW"] = 3
    graph.graph["history"]["stable_frac"] = series
    remesh.apply_remesh_if_globally_stable(graph)
    assert not requests
    remesh.apply_remesh_if_globally_stable(graph, stable_step_window=2)
    assert len(requests) == 1


def test_disabled_optional_gate_does_not_consume_its_history(requests):
    graph = _graph()
    graph.graph["REMESH_REQUIRE_STABILITY"] = False
    graph.graph["history"]["phase_sync"] = [float("nan")]
    remesh.apply_remesh_if_globally_stable(graph)
    assert len(requests) == 1


def test_optional_evidence_uses_finite_exact_mean_without_sum_overflow(requests):
    graph = _graph()
    graph.graph["REMESH_STABILITY_WINDOW"] = 2
    graph.graph["REMESH_MAX_GLYPH_DISR"] = 1e308
    graph.graph["history"] = {
        "stable_frac": [1.0, 1.0],
        "glyph_load_disr": [1e308, 1e308],
    }
    remesh.apply_remesh_if_globally_stable(graph)
    assert len(requests) == 1


def test_invalid_optional_evidence_is_not_hidden_by_an_earlier_false_gate(requests):
    graph = _graph()
    graph.graph["history"].update(phase_sync=[0.0], sense_sigma_mag=[float("nan")])
    _assert_rejected_without_writes(graph, requests, match="sense_sigma_mag")


@pytest.mark.parametrize(
    "key,value",
    (
        ("_last_remesh_step", True),
        ("_last_remesh_step_basis", "wrong"),
        ("_last_remesh_ts", _TINY),
    ),
)
def test_consumed_cooldown_record_is_admitted_before_map(key, value, requests):
    graph = _graph()
    graph.graph["REMESH_COOLDOWN_TS"] = 1
    graph.graph[key] = value
    _assert_rejected_without_writes(graph, requests, match=key)


def test_valid_gate_executes_actual_delayed_map_and_records_cooldown():
    graph = _graph()
    graph.graph.update(
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10,
        EPI_MAX=10,
        CLIP_MODE="hard",
        _epi_hist=deque(({0: 0.0, 1: 0.0}, {0: 0.5, 1: 0.25}), maxlen=8),
    )
    for node, form in enumerate((0.5, 0.25)):
        graph.nodes[node]["EPI"] = form
    remesh.apply_remesh_if_globally_stable(graph)
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (0.125, 0.0625)
    assert graph.graph["_last_remesh_step"] == 1
    assert graph.graph["_last_remesh_step_basis"] == "stable_samples"
    assert graph.graph["_last_remesh_ts"] == 2.0
