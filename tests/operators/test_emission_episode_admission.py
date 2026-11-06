"""Finite-duration Emission admission, lifecycle and live-source boundaries.

Local unit grammar and initial driver coordinates are supplied inputs. Real flow
owns physical samples; numerical steps are not invented operator occurrences.
The independent participation fixture owns the positive acquired preparation.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics import _mutation_occurrence as occurrence
from tnfr.errors import FrequencyError, NetworkConfigError
from tnfr.operators import native_emission_window as consumer
from tnfr.operators._operator_growth_trigger import (
    _native_emission_window_source_context,
)
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.grammar_debt import (
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.native_emission_window import execute_native_emission_window
from tnfr.operators.network_stage import _detached_stage_graph
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
PHYSICAL_RECEIPTS = "_native_zhir_consumed_intervals"
HYBRID_RECEIPT = "_operator_growth_mutation_end"
INTERVAL = ("epi_time_history", 0.0, 0.25)


def _graph():
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        RANDOM_SEED=43,
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        EPI_LATENT_MAX=0.5,
        ZHIR_THRESHOLD_XI=0.125,
        NATIVE_ZHIR_OCCURRENCE_POLICY="consume_timestamped_interval",
        NATIVE_ZHIR_ABSTENTION_POLICY="skip",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
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
        deepcopy(graph.graph["history"]),
        graph.graph["_t"],
        tuple(graph.edges(data="weight")),
        deepcopy(graph.graph.get("hybrid_event_log")),
    )


def _episode(graph, **kwargs):
    return execute_emission_episode(graph, NODES, duration=0.25, **kwargs)


def _window(graph, **kwargs):
    return execute_native_emission_window(graph, NODES, duration=0.25, **kwargs)


@pytest.mark.parametrize("duration", [True, 0.0, -0.25, math.nan, math.inf])
def test_invalid_episode_duration_never_creates_time_or_activation(duration):
    graph = _graph()
    before = _state(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        execute_emission_episode(graph, NODES, duration=duration)
    assert _state(graph) == before
    assert all("_emission_activated" not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("targets", [[], (), (CORE, CORE), ("foreign",)])
def test_invalid_episode_targets_leave_source_unchanged(targets):
    graph = _graph()
    before = _state(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        execute_emission_episode(graph, targets, duration=0.25)
    assert _state(graph) == before


def test_positive_unowned_row_cannot_hide_inside_an_emission_interval():
    graph = _graph()
    before = _state(graph)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        execute_emission_episode(graph, (CORE,), duration=0.25)
    assert _state(graph) == before


@pytest.mark.parametrize("pressure", [0.0, -0.0625, True])
def test_target_requires_its_own_admitted_positive_rate(pressure):
    graph = _graph()
    graph.nodes[CORE]["delta_nfr"] = pressure
    before = _state(graph)
    expected_error = (
        NetworkConfigError
        if pressure is True
        else (TypeError, ValueError, OperatorPreconditionError)
    )
    with pytest.raises(expected_error):
        _episode(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("field,value", [("nu_f", True), ("nu_f", 0.0), ("EPI", True)])
def test_malformed_or_inactive_episode_coordinate_is_not_a_noop(field, value):
    graph = _graph()
    graph.nodes[CORE][field] = value
    before = _state(graph)
    expected_error = (
        FrequencyError
        if field == "nu_f" and value is True
        else (TypeError, ValueError, OperatorPreconditionError)
    )
    with pytest.raises(expected_error):
        _episode(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("domain", ["gamma", "extended", "soft_projection"])
def test_unsupported_flow_domain_does_not_create_an_episode(domain):
    graph = _graph()
    if domain == "gamma":
        graph.graph["GAMMA"] = {"type": "kuramoto", "K": 0.0}
    elif domain == "extended":
        graph.graph["use_extended_dynamics"] = True
    else:
        graph.graph["CLIP_MODE"] = "soft"
        graph.graph["CLIP_SOFT_K"] = 0.1
    before = _state(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _episode(graph)
    assert _state(graph) == before


def test_disabled_grammar_mapping_is_not_admitted_by_its_truthiness():
    graph = _graph()
    graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    before = _state(graph)
    with pytest.raises((ValueError, OperatorPreconditionError), match="grammar"):
        _episode(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("boundary", ["entry", "completion"])
def test_entire_positive_episode_stays_in_the_strict_al_activity_chart(boundary):
    graph = _graph()
    graph.graph["EPI_LATENT_MAX"] = 0.25 if boundary == "entry" else 0.375
    before = _state(graph)
    with pytest.raises(
        (ValueError, OperatorPreconditionError), match="latent|AL precondition"
    ):
        _episode(graph)
    assert _state(graph) == before
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("projection", ["initial", "endpoint"])
def test_episode_excludes_hard_projection_even_when_latent_chart_admits(projection):
    graph = _graph()
    graph.graph["EPI_LATENT_MAX"] = 1.0
    if projection == "initial":
        # The gain would return to the hard interval; it must not legitimize
        # an initial projected coordinate retrospectively.
        graph.graph["EPI_MIN"] = 0.0
        graph.nodes[CORE]["EPI"] = -0.0625
    else:
        graph.graph["EPI_MAX"] = 0.3125
    before = _state(graph)
    with pytest.raises(
        (ValueError, OperatorPreconditionError), match="project|clip|rail"
    ):
        _episode(graph)
    assert _state(graph) == before
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("dt_min", [0.0, 0.0625])
def test_real_subdivision_is_one_episode_and_does_not_double_add_al(dt_min):
    graph = _graph()
    graph.graph["DT_MIN"] = dt_min
    prior = {node: read_grammar_state(graph.nodes[node]) for node in graph}
    memory = deepcopy(graph.graph["history"])
    result = _episode(graph, method="euler", n_jobs=1)
    assert result.targets == result.nodes == NODES
    assert result.exact_rates == (Q(1, 2), Q(1, 2))
    assert result.effective_boosts == (0.125, 0.125)
    assert result.exact_epi_before == (Q(1, 4), Q(1, 4))
    assert result.exact_epi_after == (Q(3, 8), Q(3, 8))
    assert result.start_time == 0.0 and result.final_time == 0.25
    assert graph.graph["_t"] == 0.25
    assert graph.graph["history"] == memory
    for node in graph:
        assert _form(graph, node) == 0.375
        state = read_grammar_state(graph.nodes[node])
        assert state.operator_step == prior[node].operator_step + 1
        assert state.history == (*prior[node].history, "AL")
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 1
        assert graph.nodes[node]["_emission_activated"] is True
        assert tuple(graph.nodes[node]["epi_time_history"])[-2:] == (
            (0.0, 0.25),
            (0.25, 0.375),
        )
    # Solver diagnostic channels need not be subdivision invariant.
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)


def test_episode_completion_preserves_existing_native_waiting_policy():
    graph = _graph()
    graph.graph["history"]["since_AL"][CORE] = graph.graph["AL_MAX_LAG"]
    graph.graph["history"]["since_EN"][PEER] = graph.graph["EN_MAX_LAG"]
    before = deepcopy(graph.graph["history"])
    _episode(graph)
    assert graph.graph["history"] == before
    assert read_grammar_state(graph.nodes[CORE]).history[-1] == "AL"


@pytest.mark.parametrize("policy", [None, "level", "invalid"])
def test_composition_requires_explicit_consumption_before_emission(policy):
    graph = _graph()
    if policy is None:
        del graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"]
    else:
        graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = policy
    before = _state(graph)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        _window(graph)
    assert _state(graph) == before


def test_malformed_physical_receipt_cannot_run_a_new_source():
    graph = _graph()
    graph.nodes[PEER][PHYSICAL_RECEIPTS] = (("epi_time_history", True, 0.25),)
    before = _state(graph)
    with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
        _window(graph)
    assert _state(graph) == before


class _CompletionFailure:
    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator == "emission" and node == PEER:
            assert graph.graph["_t"] == 0.25
            assert _form(graph, node) == 0.375
            graph.nodes[node]["declared_completion_fault"] = True
            raise RuntimeError("declared episode completion failure")


@pytest.mark.parametrize("composed", [False, True])
def test_failed_completion_restores_real_flow_and_all_episode_occurrences(composed):
    graph = _graph()
    graph.graph["integrity_monitor"] = _CompletionFailure()
    before = _state(graph)
    with pytest.raises(RuntimeError, match="declared episode completion failure"):
        (_window if composed else _episode)(graph)
    assert _state(graph) == before
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)
    assert all("_structural_lineage" not in graph.nodes[node] for node in graph)


def test_failed_episode_does_not_leave_new_graph_owned_native_history():
    graph = _graph()
    del graph.graph["history"]
    graph.graph["integrity_monitor"] = _CompletionFailure()
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(RuntimeError, match="declared episode completion failure"):
        _window(graph)
    assert "history" not in graph.graph
    assert graph.graph["_t"] == 0.0
    assert dict(graph.nodes(data=True)) == before


class _SilentLifecycleCorruption:
    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator == "emission" and node == PEER:
            graph.nodes[node]["_structural_lineage"]["activation_count"] = 99


def test_silent_completion_lifecycle_corruption_rolls_back_the_entire_episode():
    graph = _graph()
    graph.graph["integrity_monitor"] = _SilentLifecycleCorruption()
    before = _state(graph)
    with pytest.raises(RuntimeError, match="lifecycle|activation|completion"):
        _episode(graph)
    assert _state(graph) == before
    assert all("_structural_lineage" not in graph.nodes[node] for node in graph)
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_preactivated_missing_lineage_cannot_claim_a_new_counted_episode():
    graph = _graph()
    graph.nodes[CORE]["_emission_activated"] = True
    before = _state(graph)
    with pytest.raises(
        (ValueError, OperatorPreconditionError), match="lineage|activation"
    ):
        _episode(graph)
    assert _state(graph) == before
    assert all("epi_time_history" not in graph.nodes[node] for node in graph)


def test_late_native_failure_keeps_completed_episode_and_actual_mutation_prefix():
    graph = _graph()
    graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"] = math.nan
    with pytest.raises(ValueError, match="IL_dnfr_factor"):
        _window(graph)
    assert graph.graph["_t"] == 0.25
    assert all(_form(graph, node) == 0.375 for node in graph)
    assert graph.nodes[CORE]["theta"] == 0.25
    assert graph.nodes[CORE][PHYSICAL_RECEIPTS] == (INTERVAL,)
    assert graph.nodes[PEER]["theta"] == 0.0
    assert read_grammar_state(graph.nodes[CORE]).history[-2:] == ("AL", "ZHIR")
    assert read_grammar_state(graph.nodes[PEER]).history[-1] == "AL"
    assert all(
        graph.nodes[node]["_structural_lineage"]["activation_count"] == 1
        for node in graph
    )
    assert all(HYBRID_RECEIPT not in graph.nodes[node] for node in graph)
    assert PHYSICAL_RECEIPTS not in graph.nodes[PEER]


class _ResponseFailure:
    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator == "mutation" and node == PEER:
            graph.nodes[node]["declared_response_fault"] = True
            raise RuntimeError("declared response failure")


@pytest.mark.parametrize("fault", ["callback", "physical_receipt_commit"])
def test_response_failure_rolls_back_only_response_including_mirrored_receipt(
    monkeypatch, fault
):
    graph = _graph()
    checkpoints, commits = [], []
    actual_stage = consumer._execute_growth_stage

    def record_stage(subject, *args, **kwargs):
        checkpoints.append(_state(subject))
        return actual_stage(subject, *args, **kwargs)

    monkeypatch.setattr(consumer, "_execute_growth_stage", record_stage)
    if fault == "callback":
        graph.graph["integrity_monitor"] = _ResponseFailure()
    else:
        actual_commit = occurrence.commit_native_mutation_interval

        def fail_after_commit(data, admission):
            actual_commit(data, admission)
            commits.append(data[PHYSICAL_RECEIPTS])
            raise RuntimeError("declared physical receipt commit failure")

        # The native runtime retains its original imported helper. This seam
        # faults only the response's shared receipt commit after actual ZHIR.
        monkeypatch.setattr(
            occurrence, "commit_native_mutation_interval", fail_after_commit
        )
    with pytest.raises(RuntimeError, match="declared.*failure"):
        _window(graph)
    assert len(checkpoints) == 1 and _state(graph) == checkpoints[0]
    assert commits == ([(INTERVAL,)] if fault == "physical_receipt_commit" else [])
    assert graph.graph["_t"] == 0.25
    assert graph.nodes[CORE]["theta"] == 0.25
    assert graph.nodes[CORE][PHYSICAL_RECEIPTS] == (INTERVAL,)
    assert graph.nodes[PEER]["theta"] == 0.0
    assert PHYSICAL_RECEIPTS not in graph.nodes[PEER]
    assert HYBRID_RECEIPT not in graph.nodes[PEER]
    assert read_grammar_state(graph.nodes[PEER]).history[-2:] == ("AL", "IL")


def test_declared_consumed_memory_blocks_same_actual_physical_interval():
    graph = _graph()
    # Supplied memory control; the episode still produces all real samples.
    graph.nodes[PEER][PHYSICAL_RECEIPTS] = (INTERVAL,)
    result = _window(graph, candidates=(PEER,))
    assert result.selected_targets == () and result.stage_result is None
    assert result.decisions[0].reasons == ("mutation_interval_already_consumed",)
    assert graph.nodes[PEER][PHYSICAL_RECEIPTS] == (INTERVAL,)
    assert graph.nodes[PEER]["theta"] == 0.0
    assert HYBRID_RECEIPT not in graph.nodes[PEER]
    assert graph.nodes[CORE]["theta"] == 0.25
    assert tuple(graph.nodes[PEER]["epi_time_history"])[-2:] == (
        (0.0, 0.25),
        (0.25, 0.375),
    )


def test_cached_selection_cannot_waive_strict_same_interval_consumption(monkeypatch):
    graph = _graph()
    graph.nodes[PEER][PHYSICAL_RECEIPTS] = (INTERVAL,)
    select = consumer._select_growth_candidates

    def force_cached_selection(*args, **kwargs):
        actual = select(*args, **kwargs)
        assert actual[0].reasons == ("mutation_interval_already_consumed",)
        return tuple(replace(item, reasons=()) for item in actual)

    monkeypatch.setattr(consumer, "_select_growth_candidates", force_cached_selection)
    with pytest.raises(ValueError, match="mutation_interval_already_consumed"):
        _window(graph, candidates=(PEER,))
    assert graph.graph["_t"] == 0.25
    assert graph.nodes[CORE]["theta"] == 0.25
    assert graph.nodes[PEER]["theta"] == 0.0
    assert graph.nodes[PEER][PHYSICAL_RECEIPTS] == (INTERVAL,)
    assert HYBRID_RECEIPT not in graph.nodes[PEER]


def test_real_terminal_form_jump_invalidates_samples_without_fabricating_receipt():
    graph = _graph()
    graph.graph["history"]["since_AL"][PEER] = graph.graph["AL_MAX_LAG"]
    result = _window(graph)
    assert tuple(
        item.applied_glyph for item in result.source_result.native_decisions
    ) == ("ZHIR", "AL")
    assert result.selected_targets == (PEER,)
    assert result.growth.exact_net_growth_rate == (Q(1, 2), Q(1))
    assert _form(graph, PEER) == 0.5
    assert graph.nodes[PEER]["theta"] == 0.25
    assert graph.nodes[PEER]["epi_time_history"] == ()
    assert PHYSICAL_RECEIPTS not in graph.nodes[PEER]
    assert graph.nodes[PEER][HYBRID_RECEIPT] == Q(1, 4)


def _actual_context(graph):
    schedule = build_operator_event_schedule(
        (), start_time=graph.graph["_t"], flow_durations=(0.25,)
    )
    return _native_emission_window_source_context(
        graph, schedule, NODES, NODES, graph.graph["history"], method="euler", n_jobs=1
    )


@pytest.mark.parametrize(
    "fault", ["disabled_grammar", "nonpositive_row", "level_policy"]
)
def test_private_producer_lifetime_readmits_its_own_constitutive_domain(fault):
    graph = _graph()
    if fault == "disabled_grammar":
        graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    elif fault == "nonpositive_row":
        graph.nodes[PEER]["delta_nfr"] = 0.0
    else:
        graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = "level"
    before = _state(graph)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        with _actual_context(graph):
            pytest.fail("unsupported source minted live response authority")
    assert _state(graph) == before


def test_private_producer_cannot_replace_a_stale_requested_clock_with_live_time():
    graph = _graph()
    stale = build_operator_event_schedule((), start_time=-0.25, flow_durations=(0.25,))
    before = _state(graph)
    with pytest.raises(ValueError, match="clock|start"):
        with _native_emission_window_source_context(
            graph, stale, NODES, NODES, graph.graph["history"], method="euler", n_jobs=1
        ):
            pytest.fail("stale requested source interval was replaced implicitly")
    assert _state(graph) == before


def test_episode_context_is_owner_bound_single_stage_and_retired():
    graph = _graph()
    with _actual_context(graph) as source:
        snapshot = _detached_stage_graph(graph)
        with pytest.raises(ValueError, match="authorized stage snapshot"):
            source.gate_for(snapshot, PEER)
        with pytest.raises(ValueError, match="owner"):
            source.validate_live_before_commit(graph.copy())
        source.bind_snapshot(graph, snapshot)
        gate = source.gate_for(snapshot, PEER)
        assert gate.threshold.certificate.source == "native_emission_window_net_growth"
        with pytest.raises(ValueError, match="native_mutation_precedence"):
            source.gate_for(snapshot, CORE)
        with pytest.raises(ValueError, match="already bound"):
            source.bind_snapshot(graph, _detached_stage_graph(graph))
    with pytest.raises(ValueError, match="inactive"):
        source.validate_live_before_commit(graph)


def test_equal_foreign_episode_cannot_replace_this_actual_production():
    graph, other_graph = _graph(), _graph()
    with _actual_context(graph) as source, _actual_context(other_graph) as other:
        assert (
            source.result.episode_result.exact_epi_after
            == other.result.episode_result.exact_epi_after
        )
        assert source.result.episode_result is not other.result.episode_result
        object.__setattr__(source.result, "episode_result", other.result.episode_result)
        with pytest.raises(ValueError, match="actual episode binding|source changed"):
            source.validate_live_before_commit(graph)


def test_observation_is_frozen_and_not_a_detached_activation_api():
    graph = _graph()
    result = _window(graph)
    with pytest.raises(FrozenInstanceError):
        result.source_result.episode_result.exact_rates = (Q(9), Q(9))
    before = _state(graph)
    with pytest.raises(TypeError):
        execute_native_emission_window(
            graph, NODES, duration=0.25, source_result=result.source_result
        )
    assert _state(graph) == before
