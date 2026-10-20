"""Public runtime rejects invalid supplied policies before scientific progress."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics import runtime
from tnfr.dynamics.integrators import DefaultIntegrator
from tnfr.dynamics.selectors import AbstractSelector
from tnfr.errors import NetworkConfigError
from tnfr.utils import CallbackEvent, callback_manager


def _pair():
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph["PHASE_ADAPT"] = {"enabled": False}
    for node in graph:
        graph.nodes[node].update(
            {
                "EPI": 0.1 + 0.2 * node,
                "νf": 1.0,
                "theta": 0.25 * node,
                "Si": 0.6,
                "ΔNFR": 0.0,
            }
        )
    return graph


def _advance(graph, entry, *, apply_glyphs=False, use_Si=False, dt=0.125):
    kwargs = dict(dt=dt, apply_glyphs=apply_glyphs, use_Si=use_Si)
    if entry == "step":
        runtime.step(graph, **kwargs)
    else:
        runtime.run(graph, 1, **kwargs)


@pytest.mark.parametrize("entry", ["step", "run"])
@pytest.mark.parametrize("apply_glyphs", [False, True])
@pytest.mark.parametrize(
    "override",
    [
        {"SELECTOR_THRESHOLDS": {"si_hi": float("nan")}},
        {"SELECTOR_THRESHOLDS": {"si_hi": True}},
        {"SELECTOR_THRESHOLDS": []},
        {"PHASE_ADAPT": {"enabled": True, "up": 2.0}},
        {"PHASE_ADAPT": {"enabled": "false"}},
        {"VF_ADAPT_MU": 2.0},
        {"VF_ADAPT_TAU": 0},
        {"EPS_DNFR_STABLE": -1.0},
        {"VF_MIN": 2.0, "VF_MAX": 1.0},
        {"INTEGRATOR_METHOD": "unknown"},
        {"DT_MIN": float("inf")},
    ],
)
def test_invalid_supplied_policy_rejects_before_callbacks_or_state_writes(
    entry, apply_glyphs, override
):
    graph = _pair()
    graph.graph.update(override)
    calls = []
    callback_manager.register_callback(
        graph,
        CallbackEvent.BEFORE_STEP,
        lambda _graph, _context: calls.append("before"),
    )
    before_graph = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises((ValueError, TypeError, NetworkConfigError)):
        _advance(graph, entry, apply_glyphs=apply_glyphs)

    assert calls == []
    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes


@pytest.mark.parametrize("entry", ["step", "run"])
@pytest.mark.parametrize(
    "dt", [-1.0, float("nan"), float("inf"), True, Fraction(1, 2**2000)]
)
def test_invalid_default_integrator_span_rejects_before_mutation(entry, dt):
    graph = _pair()
    before_graph = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(NetworkConfigError):
        _advance(graph, entry, dt=dt)

    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes


@pytest.mark.parametrize("entry", ["step", "run"])
@pytest.mark.parametrize("use_Si", [False, True])
def test_valid_partial_policy_matches_explicit_overlay_through_real_runtime(
    entry, use_Si
):
    partial, full = _pair(), _pair()
    partial.graph["SELECTOR_THRESHOLDS"] = {}
    partial.graph["PHASE_ADAPT"] = {"enabled": True}
    full.graph["PHASE_ADAPT"] = dict(DEFAULTS["PHASE_ADAPT"])
    for graph in (partial, full):
        _advance(graph, entry, use_Si=use_Si)

    assert partial.graph["_t"] == full.graph["_t"] == 0.125
    for node in partial:
        for key in ("EPI", "νf", "theta", "Si", "ΔNFR", "stable_count"):
            assert partial.nodes[node][key] == full.nodes[node][key]
    assert partial.nodes[0]["EPI"] != 0.1
    assert partial.graph["PHASE_K_GLOBAL"] == full.graph["PHASE_K_GLOBAL"]
    assert partial.graph["PHASE_K_LOCAL"] == full.graph["PHASE_K_LOCAL"]


def test_disabled_glyphs_do_not_instantiate_or_prepare_a_selector():
    graph = _pair()

    class DisabledSelector(AbstractSelector):
        def __init__(self):
            raise AssertionError("disabled selector must not be instantiated")

        def select(self, graph, node):
            return "IL"

    graph.graph["glyph_selector"] = DisabledSelector
    _advance(graph, "step", apply_glyphs=False)
    assert graph.graph["_t"] == 0.125
    assert graph.graph["glyph_selector"] is DisabledSelector


def test_consumption_checks_still_reject_callback_configuration_changes():
    graph = _pair()
    calls = []

    def change_policy(live_graph, _context):
        calls.append("before")
        live_graph.graph["VF_ADAPT_MU"] = 2.0

    callback_manager.register_callback(graph, CallbackEvent.BEFORE_STEP, change_policy)

    with pytest.raises(ValueError, match="VF_ADAPT_MU"):
        _advance(graph, "step")

    assert calls == ["before"]
    # Preflight covers the supplied configuration, not later callback writes
    # or a full-step transaction. The consumed invalid policy still rejects.
    assert graph.graph["_t"] == 0.125
    assert graph.nodes[0]["EPI"] != 0.1
    assert "stable_count" not in graph.nodes[0]


@pytest.mark.parametrize("entry", ["step", "run"])
@pytest.mark.parametrize("writer", ["callback", "pressure"])
@pytest.mark.parametrize("invalid", [True, "0.125", Fraction(1, 2**2000)])
def test_runtime_does_not_coerce_changed_clock_before_integrator_admission(
    entry, writer, invalid
):
    graph = _pair()
    graph.graph.update(DT=0.125, _t=0.0)
    initial_form = tuple(graph.nodes[node]["EPI"] for node in graph)
    calls = []

    def replace_clock(live_graph, *_args, **_kwargs):
        calls.append(writer)
        live_graph.graph["DT"] = invalid

    if writer == "callback":
        callback_manager.register_callback(
            graph, CallbackEvent.BEFORE_STEP, replace_clock
        )
    else:
        graph.graph["compute_delta_nfr"] = replace_clock

    with pytest.raises(NetworkConfigError, match="dt"):
        _advance(graph, entry, dt=None)

    assert calls == [writer]
    assert graph.graph["DT"] is invalid
    assert graph.graph["_t"] == 0.0
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == initial_form


def test_callback_exact_timestep_remains_a_valid_shared_clock():
    graph = _pair()
    graph.graph["DT"] = 0.125
    callback_manager.register_callback(
        graph,
        CallbackEvent.BEFORE_STEP,
        lambda live_graph, _context: live_graph.graph.update(DT=Fraction(1, 16)),
    )
    reference = _pair()
    _advance(reference, "step", dt=0.0625)
    _advance(graph, "step", dt=None)
    assert graph.graph["_t"] == reference.graph["_t"] == 0.0625
    for node in graph:
        for key in ("EPI", "νf", "theta", "ΔNFR"):
            assert graph.nodes[node][key] == reference.nodes[node][key]


def test_custom_integrator_factory_retains_its_own_method_contract():
    graph = _pair()
    calls = []

    class CustomIntegrator(DefaultIntegrator):
        def integrate(self, graph, *, dt=None, t=None, method=None, n_jobs=None):
            calls.append(("integrate", method))
            return {}

    def factory(live_graph):
        calls.append(("factory", live_graph is graph))
        return CustomIntegrator()

    graph.graph["integrator"] = factory
    graph.graph["INTEGRATOR_METHOD"] = "custom-method"
    _advance(graph, "step")
    assert calls == [("factory", True), ("integrate", "custom-method")]


@pytest.mark.parametrize("entry", ["step", "run"])
@pytest.mark.parametrize(
    "kind",
    ["window", "limit", "grammar_flag", "lag", "trace", "debt", "operator_index"],
)
def test_active_batch_admission_precedes_runtime_callbacks_and_observations(
    entry, kind
):
    graph = _pair()
    if kind == "window":
        graph.graph["GLYPH_HYSTERESIS_WINDOW"] = 1.5
    elif kind == "limit":
        graph.graph["AL_MAX_LAG"] = True
    elif kind == "grammar_flag":
        graph.graph["GRAMMAR_CANON"] = {"enabled": "unknown"}
    elif kind == "lag":
        graph.graph["history"] = {"since_AL": {0: 2, 1: -1}}
    elif kind == "trace":
        graph.nodes[1]["glyph_history"] = None
    elif kind == "debt":
        graph.nodes[1]["_grammar_u2_debt"] = True
    else:
        graph.nodes[1]["_operator_step"] = -1
    calls = []
    callback_manager.register_callback(
        graph, CallbackEvent.BEFORE_STEP, lambda *_: calls.append("before")
    )
    before_graph = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises((TypeError, ValueError)):
        _advance(graph, entry, apply_glyphs=True)
    assert calls == []
    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes
    assert "_runtime_step_next" not in graph.graph


def test_disabled_batch_does_not_consume_unused_lag_or_grammar_policy():
    graph = _pair()
    graph.graph.update(AL_MAX_LAG=True, GRAMMAR_CANON={"enabled": "unknown"})
    graph.graph["history"] = {"since_AL": {0: -1}}
    _advance(graph, "step", apply_glyphs=False)
    assert graph.graph["_t"] == 0.125
    assert graph.graph["history"]["since_AL"] == {0: -1}


def test_active_batch_rechecks_policy_changed_after_initial_admission():
    graph = _pair()
    forms = tuple(graph.nodes[node]["EPI"] for node in graph)
    callback_manager.register_callback(
        graph,
        CallbackEvent.BEFORE_STEP,
        lambda live, _: live.graph.update(AL_MAX_LAG=True),
    )
    with pytest.raises(TypeError, match="AL_MAX_LAG"):
        _advance(graph, "step", apply_glyphs=True)
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == forms
    assert graph.graph["_runtime_step_next"] == 1
    assert graph.graph["AL_MAX_LAG"] is True
    assert all("epi_time_history" in graph.nodes[node] for node in graph)
    assert all("glyph_history" not in graph.nodes[node] for node in graph)
