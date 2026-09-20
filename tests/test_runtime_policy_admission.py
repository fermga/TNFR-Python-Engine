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
