"""Admission and lifetime of the opt-in episode-interior selector channel.

Local grammar and pressure are declared engineering inputs. The independent
participation fixture acquires its actual pressure/context through IL and OZ.
No boundary control rewrites physical evidence to force an operator request.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_EPI
from tnfr.dynamics.selectors import (
    DefaultGlyphSelector,
    ParametricGlyphSelector,
    apply_native_operator_batch,
)
from tnfr.operators import emission_observation as observation_owner
from tnfr.operators.emission_observation import (
    _execute_emission_for_selection,
    observe_emission_episode,
)
from tnfr.operators.grammar_debt import (
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.native_emission_window import execute_native_emission_window
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
RECEIPTS = "_native_zhir_consumed_intervals"


def _graph():
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        RANDOM_SEED=53,
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        EPI_LATENT_MAX=0.5,
        ZHIR_THRESHOLD_XI=0.0625,
        NATIVE_ZHIR_OCCURRENCE_POLICY="consume_timestamped_interval",
        NATIVE_ZHIR_ABSTENTION_POLICY="skip",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        glyph_selector=ParametricGlyphSelector(),
        history={
            "since_AL": dict.fromkeys(NODES, 0),
            "since_EN": dict.fromkeys(NODES, 0),
        },
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        ZHIR_theta_shift=0.25,
    )
    for node, capacity, pressure, sense, history in (
        (CORE, 8.0, 0.0625, 0.0, ["IL", "OZ", "IL"]),
        (PEER, 1.0, 0.5, 1.0, ["IL", "OZ"]),
    ):
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=capacity,
            delta_nfr=pressure,
            theta=0.0,
            Si=sense,
            d2EPI=0.0,
            epi_kind="form",
            glyph_history=history,
        )
        reset_grammar_state_from_history(graph.nodes[node])
    return graph


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _state(graph):
    return (
        deepcopy(dict(graph.nodes(data=True))),
        "history" in graph.graph,
        deepcopy(graph.graph.get("history")),
        graph.graph["_t"],
        tuple(graph.edges(data="weight")),
    )


def _window(graph, **kwargs):
    return execute_native_emission_window(
        graph, NODES, duration=0.25, method="euler", n_jobs=1, **kwargs
    )


@pytest.mark.parametrize("policy", [None, True, "", "interior", "Emission_Interior", 0])
def test_invalid_selector_channel_rejects_before_source_and_history_installation(
    policy,
):
    graph = _graph()
    del graph.graph["history"]
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        _window(graph, selector_acceleration_source=policy)
    assert _state(graph) == before
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)
    assert all("_structural_lineage" not in graph.nodes[node] for node in graph)


class _ParametricSubclass(ParametricGlyphSelector):
    pass


@pytest.mark.parametrize("selector", [None, DefaultGlyphSelector, _ParametricSubclass])
def test_typed_channel_requires_the_exact_supported_parametric_selector(selector):
    graph = _graph()
    graph.graph["glyph_selector"] = None if selector is None else selector()
    before = _state(graph)
    with pytest.raises((TypeError, ValueError)):
        _window(graph, selector_acceleration_source="emission_interior")
    assert _state(graph) == before


def _effect_projection(graph, result):
    return (
        graph.graph["_t"],
        tuple(
            (
                _form(graph, node),
                graph.nodes[node]["nu_f"],
                graph.nodes[node]["delta_nfr"],
                graph.nodes[node]["theta"],
                get_attr(graph.nodes[node], ALIAS_D2EPI),
                read_grammar_state(graph.nodes[node]),
                tuple(graph.nodes[node]["epi_time_history"]),
                graph.nodes[node].get(RECEIPTS),
            )
            for node in graph
        ),
        result.selected_targets,
        tuple(item.applied_glyph for item in result.source_result.native_decisions),
    )


def test_omitted_channel_matches_explicit_solver_diagnostic_without_new_observation():
    omitted, explicit = _graph(), _graph()
    first = _window(omitted)
    second = _window(explicit, selector_acceleration_source="solver_diagnostic")
    assert _effect_projection(omitted, first) == _effect_projection(explicit, second)
    for result in (first, second):
        assert result.source_result.selector_observation is None
        assert result.source_result.selector_acceleration_source == "solver_diagnostic"


def test_typed_selection_uses_full_native_population_despite_candidate_subset():
    graph = _graph()
    result = _window(
        graph, candidates=(PEER,), selector_acceleration_source="emission_interior"
    )
    source = result.source_result
    observation = source.selector_observation
    assert result.candidates == (PEER,)
    assert observation.nodes == NODES
    assert observation.exact_interior_accelerations == (Q(0), Q(0))
    assert tuple(item.node for item in source.native_decisions) == NODES
    assert tuple(
        item.selection.acceleration_source for item in source.native_decisions
    ) == ("emission_interior", "emission_interior")
    assert all(
        item.selection.acceleration_window == (Q(0), Q(1, 4))
        for item in source.native_decisions
    )
    assert all(
        item.selection.normalized_metrics[2] == 0.0 for item in source.native_decisions
    )


def test_zero_row_nonparticipant_still_belongs_to_the_native_observation_basis():
    graph = _graph()
    graph.nodes[PEER]["delta_nfr"] = 0.0
    result = execute_native_emission_window(
        graph,
        (CORE,),
        duration=0.25,
        candidates=(PEER,),
        method="euler",
        n_jobs=1,
        selector_acceleration_source="emission_interior",
    )
    observation = result.source_result.selector_observation
    assert observation.targets == (CORE,) and observation.nodes == NODES
    assert observation.exact_held_rates == (Q(1, 2), Q(0))
    assert observation.exact_interior_accelerations == (Q(0), Q(0))
    assert observation.solver_accelerations == (2.0, 0.0)
    assert tuple(item.node for item in result.source_result.native_decisions) == NODES
    assert all(
        item.selection.acceleration_source == "emission_interior"
        for item in result.source_result.native_decisions
    )
    assert _form(graph, PEER) == 0.25
    assert result.selected_targets == ()


def test_passive_observation_and_typed_channel_preserve_actual_solver_cache_and_samples():
    graph = _graph()
    result = _window(graph, selector_acceleration_source="emission_interior")
    before = _state(graph)
    report = observe_emission_episode(result.source_result.episode_result)
    assert _state(graph) == before
    assert report.nodes == NODES
    assert report.exact_observed_rates == report.exact_held_rates == (Q(1, 2), Q(1, 2))
    assert report.exact_interior_accelerations == (Q(0), Q(0))
    assert report.solver_rates == (0.5, 0.5)
    assert report.solver_accelerations == (2.0, 2.0)
    assert report.source == "emission_episode_interior"
    assert report.acceleration_basis == "held_emission_law"
    assert not report.episode_sampled_acceleration_available
    assert not report.boundary_transition_available
    assert get_attr(graph.nodes[CORE], ALIAS_D2EPI) == 2.0
    # The peer's actual supplementary staged ZHIR commits its own history-derived
    # compatibility diagnostic after source capture. Selection did not write it.
    assert get_attr(graph.nodes[PEER], ALIAS_D2EPI) == 0.0
    for node in graph:
        assert tuple(graph.nodes[node]["epi_time_history"]) == (
            (0.0, 0.25),
            (0.25, 0.375),
        )
    with pytest.raises(FrozenInstanceError):
        report.exact_interior_accelerations = (Q(2), Q(2))


def test_returned_observation_is_not_an_activation_argument_for_another_graph():
    produced, foreign = _graph(), _graph()
    result = _window(produced, selector_acceleration_source="emission_interior")
    report = result.source_result.selector_observation
    before = _state(foreign)
    with pytest.raises(TypeError):
        _window(
            foreign,
            selector_acceleration_source="emission_interior",
            selector_observation=report,
        )
    assert _state(foreign) == before


def test_late_native_failure_preserves_prefix_and_retires_the_typed_channel():
    graph = _graph()
    selector = graph.graph["glyph_selector"]
    graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"] = math.nan
    with pytest.raises(ValueError, match="IL_dnfr_factor"):
        _window(graph, selector_acceleration_source="emission_interior")
    assert graph.graph["_t"] == 0.25
    assert graph.nodes[CORE]["theta"] == 0.25
    assert graph.nodes[CORE][RECEIPTS] == (("epi_time_history", 0.0, 0.25),)
    assert graph.nodes[PEER]["theta"] == 0.0
    assert all(_form(graph, node) == 0.375 for node in graph)
    # No supplementary stage was reached; the typed channel and direct native
    # ZHIR have preserved both actual completion caches.
    assert all(get_attr(graph.nodes[node], ALIAS_D2EPI) == 2.0 for node in graph)
    assert selector._preselection is None and selector._prepared_graph is None
    graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"] = 0.5
    trace = []
    apply_native_operator_batch(graph, graph.graph["history"], decision_trace=trace)
    assert all(
        item.selection.acceleration_source == "solver_diagnostic" for item in trace
    )
    assert all(item.selection.acceleration_window is None for item in trace)
    assert selector._preselection is None and selector._prepared_graph is None


def _actual_context(graph):
    return _execute_emission_for_selection(
        graph, NODES, duration=0.25, method="euler", n_jobs=1
    )


def test_actual_selector_context_rejects_foreign_owner_and_retires_after_exit():
    graph = _graph()
    with _actual_context(graph) as context:
        assert context.validate_live(graph) is context.observation
        with pytest.raises(ValueError, match="foreign|owner"):
            context.validate_live(graph.copy())
    with pytest.raises(ValueError, match="inactive"):
        context.validate_live(graph)


@pytest.mark.parametrize("change", ["pressure", "clock", "history", "population"])
def test_stale_live_source_cannot_prepare_a_typed_batch(change):
    graph = _graph()
    with _actual_context(graph) as context:
        if change == "pressure":
            graph.nodes[CORE]["delta_nfr"] += 0.125
        elif change == "clock":
            graph.graph["_t"] += 0.25
        elif change == "history":
            graph.graph["history"]["since_AL"][CORE] += 1
        else:
            graph.add_node("foreign", **deepcopy(graph.nodes[PEER]))
        with pytest.raises(ValueError, match="stale|population|source"):
            context.admit_preparation(graph, tuple(graph))


@pytest.mark.parametrize("nodes", [(CORE,), (PEER, CORE)])
def test_channel_preparation_cannot_use_a_subset_or_reordered_native_population(nodes):
    graph = _graph()
    with _actual_context(graph) as context:
        with pytest.raises(ValueError, match="full ordered population"):
            context.admit_preparation(graph, nodes)


def test_actual_context_lends_its_observation_to_one_preparation_only():
    graph = _graph()
    with _actual_context(graph) as context:
        assert context.admit_preparation(graph, NODES) is context.observation
        context.finish_preparation(graph)
        with pytest.raises(ValueError, match="already consumed"):
            context.admit_preparation(graph, NODES)
        with pytest.raises(ValueError, match="unfinished preparation"):
            context.finish_preparation(graph)


def test_copied_episode_fields_do_not_make_an_actual_observation_source():
    graph = _graph()
    with _actual_context(graph) as context:
        copied = replace(context.result)
        assert copied.exact_epi_after == context.result.exact_epi_after
        with pytest.raises(ValueError, match="intact actual episode"):
            observe_emission_episode(copied)


def test_equal_foreign_actual_episode_cannot_replace_this_selector_source():
    graph, foreign = _graph(), _graph()
    with _actual_context(graph) as context, _actual_context(foreign) as other:
        assert context.result is not other.result
        assert (
            context.observation.exact_observed_rates
            == other.observation.exact_observed_rates
        )
        object.__setattr__(context, "_result", other.result)
        with pytest.raises(ValueError, match="source|identity|changed"):
            context.admit_preparation(graph, NODES)


def test_corrupted_passive_observation_does_not_replace_the_produced_channel():
    graph = _graph()
    with _actual_context(graph) as context:
        object.__setattr__(
            context.observation, "exact_interior_accelerations", (Q(2), Q(2))
        )
        with pytest.raises(ValueError, match="observation changed"):
            context.admit_preparation(graph, NODES)


def test_context_construction_failure_retains_completed_episode_and_new_history(
    monkeypatch,
):
    graph = _graph()
    del graph.graph["history"]

    def failed_context(*args, **kwargs):
        raise RuntimeError("declared context construction failure")

    monkeypatch.setattr(observation_owner, "_EmissionSelectionContext", failed_context)
    with pytest.raises(RuntimeError, match="declared context construction failure"):
        _window(graph, selector_acceleration_source="emission_interior")
    assert graph.graph["_t"] == 0.25
    assert type(graph.graph["history"]) is dict
    assert graph.graph["history"] == {}
    for node in graph:
        assert _form(graph, node) == 0.375
        assert graph.nodes[node]["theta"] == 0.0
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 1
        assert read_grammar_state(graph.nodes[node]).history[-1] == "AL"
        assert RECEIPTS not in graph.nodes[node]
        assert tuple(graph.nodes[node]["epi_time_history"]) == (
            (0.0, 0.25),
            (0.25, 0.375),
        )


class _FailedEpisodeCompletion:
    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator == "emission" and node == PEER:
            assert graph.graph["_t"] == 0.25
            raise RuntimeError("declared episode completion failure")


def test_typed_source_episode_failure_restores_flow_lifecycle_and_missing_history():
    graph = _graph()
    del graph.graph["history"]
    graph.graph["integrity_monitor"] = _FailedEpisodeCompletion()
    before = _state(graph)
    with pytest.raises(RuntimeError, match="declared episode completion failure"):
        _window(graph, selector_acceleration_source="emission_interior")
    assert _state(graph) == before
    assert all("_structural_lineage" not in graph.nodes[node] for node in graph)
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_ordinary_opaque_selector_does_not_claim_an_unevaluated_acceleration_source():
    graph = _graph()

    def opaque_choice(subject, node):
        return "IL"

    graph.graph["glyph_selector"] = opaque_choice
    trace = []
    apply_native_operator_batch(graph, graph.graph["history"], decision_trace=trace)
    assert tuple(item.applied_glyph for item in trace) == ("IL", "IL")
    for item in trace:
        assert item.selection.kind == "opaque"
        assert item.selection.acceleration_source is None
        assert item.selection.acceleration_window is None
    assert graph.graph["_t"] == 0.0
