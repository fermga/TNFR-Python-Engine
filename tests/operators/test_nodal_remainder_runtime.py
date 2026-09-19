"""Independent live-pressure, carry-continuity and rollback checks for B23."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from fractions import Fraction
from random import Random

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_EPI, ALIAS_THETA
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators import nodal_remainder_runtime as runtime
from tnfr.operators._mutation_gate import mutation_threshold_sample
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.utils._structural_signature import structural_proof_signature

SLOT = "_nodal_remainder_runtime_state"


def _graph(epi=(0.5, 0.75)):
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=17,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=0.05,
        EPI_MAX=1.0,
        compute_delta_nfr=default_compute_delta_nfr,
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
    )
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node].update(
            EPI=value,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=[],
        )
    graph.edges[0, 1].update(weight=1.0, length=1.0)
    return graph


def _schedule(graph, names=(), durations=None):
    return build_operator_event_schedule(
        names,
        start_time=graph.graph["_t"],
        flow_durations=(
            durations if durations is not None else (0.0,) * (len(names) + 1)
        ),
    )


def _run(graph, duration=0.1):
    return runtime.execute_nodal_remainder_event_schedule(
        graph,
        _schedule(graph, durations=(duration,)),
    )


def _signature(graph):
    return structural_proof_signature((graph.graph, graph._node, graph._adj))


def _epi(graph):
    return tuple(graph.nodes[node][ALIAS_EPI[0]] for node in graph)


def test_pressure_is_recomputed_from_each_visible_endpoint():
    graph = _graph()
    schedule = _schedule(graph, durations=(0.5,))
    partition = build_physical_flow_partition(schedule.intervals[0], (0.25, 0.25))

    result = runtime.execute_nodal_remainder_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
    )

    assert _epi(graph) == (0.59375, 0.65625)
    assert _epi(graph) != (0.625, 0.625)  # Frozen initial pressure would give this.
    assert graph.graph["_t"] == 0.5
    assert len(result.flows) == 2
    first, second = (flow.step for flow in result.flows)
    assert first.pressure == (0.25, -0.25)
    assert first.after.epi == (0.5625, 0.6875)
    assert second.pressure == (0.125, -0.125)
    assert second.before == first.after
    assert tuple(graph.nodes[i]["delta_nfr"] for i in graph) == (0.0625, -0.0625)
    assert result.runtime_provenance_certified
    for i in graph:
        assert tuple(graph.nodes[i]["epi_time_history"]) == (
            (0.0, (0.5, 0.75)[i]),
            (0.25, first.after.epi[i]),
            (0.5, second.after.epi[i]),
        )


def test_exact_nodal_area_and_carry_telescope_are_independently_reconstructed():
    graph = _graph()
    result = _run(graph)
    step = result.flows[0].step
    assert any(step.after.remainder)
    for i in graph:
        expected = Fraction((0.5, 0.75)[i]) + Fraction(0.1) * Fraction((0.25, -0.25)[i])
        assert step.after.exact_epi[i] == expected
        assert step.after.epi[i] == float(expected)
        assert step.after.remainder[i] == expected - Fraction(float(expected))
        assert Fraction(step.after.epi[i]) - Fraction(step.before.epi[i]) == (
            step.exact_increment[i] + step.before.remainder[i] - step.after.remainder[i]
        )
    assert result.final_binding.state == step.after
    assert graph.graph[SLOT] is result.final_binding


def test_zero_duration_and_positive_continuation_use_existing_carry():
    graph = _graph()
    first = _run(graph)
    carry = first.final_binding.state.remainder
    assert any(carry)
    stationary = _run(graph, 0.0)
    assert stationary.initial_binding is first.final_binding
    assert stationary.final_binding.state.remainder == carry
    assert not stationary.flows
    second = _run(graph, 0.1)
    assert second.initial_binding.state == stationary.final_binding.state
    step = second.flows[0].step
    for i in graph:
        expected = first.final_binding.state.exact_epi[i] + Fraction(0.1) * Fraction(
            step.pressure[i]
        )
        assert step.after.exact_epi[i] == expected
    assert second.final_binding.time == 0.2
    assert (
        first.runtime_provenance_certified
    )  # Historical evidence survives later evolution.
    assert stationary.runtime_provenance_certified


def test_mutation_uses_visible_secants_not_invisible_carried_area():
    graph = _graph((0.5, math.nextafter(0.5, math.inf)))
    result = _run(graph, 0.125)
    step = result.flows[0].step
    assert step.after.epi == step.before.epi
    assert step.exact_increment[0] > 0
    assert step.after.remainder[0] > 0
    sample = mutation_threshold_sample(graph.nodes[0], graph.graph)
    assert sample.history_key == "epi_time_history"
    assert sample.sample_interval == 0.125
    assert sample.depi_dt == 0.0
    assert not sample.crossed


def test_actual_um_il_sha_preserve_nonzero_carry_and_visible_epi():
    graph = _graph()
    initial_flow = _run(graph)
    initial = initial_flow.final_binding.state
    assert any(initial.remainder)
    names = ("coupling", "coherence", "silence")
    result = runtime.execute_nodal_remainder_event_schedule(
        graph, _schedule(graph, names)
    )
    assert len(result.events) == 3
    for item, name in zip(result.events, names, strict=True):
        assert item.event.operator_name == name
        assert item.stage.schedule == "two_phase_jacobi"
        assert item.stage.nodes_processed == 2
        assert item.before_binding.state == initial
        assert item.after_binding.state == initial
    assert result.refresh_records
    for refresh in result.refresh_records:
        assert refresh.boundary == "operator_stage_pressure_refresh"
        assert refresh.callback_identity == id(default_compute_delta_nfr)
        assert all(value for _name, value in refresh.checks)
        left, right = refresh.after_snapshot.epi
        assert refresh.after_snapshot.delta_nfr == (right - left, left - right)
        assert refresh.before_snapshot.epi == refresh.after_snapshot.epi == initial.epi
        assert refresh.binding.state is initial
    assert result.final_binding.state == initial
    capacity = tuple(graph.nodes[i]["nu_f"] for i in graph)
    assert all(0.0 < value < 1.0 for value in capacity)
    assert all(graph.nodes[i]["glyph_history"] for i in graph)
    following = _run(graph, 0.125)
    step = following.flows[0].step
    assert step.before == initial
    assert step.capacity == capacity
    for i in graph:
        assert step.after.exact_epi[i] == (
            initial.exact_epi[i]
            + Fraction(0.125) * Fraction(capacity[i]) * Fraction(step.pressure[i])
        )


def test_declared_zero_capacity_continuation_preserves_nonzero_encoding():
    graph = _graph()
    result = _run(graph)
    initial = result.final_binding.state
    assert any(initial.remainder)
    # Capacity is an independently supplied live input of each invocation.
    for i in graph:
        graph.nodes[i]["nu_f"] = 0.0
    following = _run(graph, 0.125)
    assert following.final_binding.state == initial
    assert following.flows[0].step.capacity == (0.0, 0.0)
    assert following.flows[0].step.exact_increment == (0, 0)


@pytest.mark.parametrize("change", ("epi", "time", "edge", "weight", "nodes", "band"))
def test_stale_persistent_binding_is_rejected_without_reset(change):
    graph = _graph()
    _run(graph)
    binding = graph.graph[SLOT]
    kwargs = {}
    if change == "epi":
        graph.nodes[0][ALIAS_EPI[0]] = math.nextafter(_epi(graph)[0], math.inf)
    elif change == "time":
        graph.graph["_t"] = 0.2
    elif change == "edge":
        graph.remove_edge(0, 1)
    elif change == "weight":
        graph.edges[0, 1]["weight"] = 2.0
    elif change == "nodes":
        graph.add_node(2, **dict(graph.nodes[0]))
    else:
        kwargs["epi_lower"] = 0.1
    before = _signature(graph)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        runtime.execute_nodal_remainder_event_schedule(
            graph, _schedule(graph), **kwargs
        )
    assert _signature(graph) == before
    assert graph.graph[SLOT] is binding


@pytest.mark.parametrize(
    "names",
    (
        ("reception", "silence"),
        ("emission", "silence"),
        ("recursivity", "silence"),
        ("coupling", "coupling", "silence"),
    ),
)
def test_unsupported_or_grammar_refused_word_does_not_attach_carry(names):
    graph = _graph()
    schedule = _schedule(graph, names)
    before = _signature(graph)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        runtime.execute_nodal_remainder_event_schedule(graph, schedule)
    assert _signature(graph) == before
    assert SLOT not in graph.graph


@pytest.mark.parametrize(
    "field,value",
    (
        ("GAMMA", {"type": "constant", "value": 0.1}),
        ("CLIP_MODE", "soft"),
        ("INTEGRATOR_METHOD", "rk4"),
    ),
)
def test_unsupported_solver_configuration_is_rejected_before_attachment(field, value):
    graph = _graph()
    graph.graph[field] = value
    before = _signature(graph)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _run(graph)
    assert _signature(graph) == before
    assert SLOT not in graph.graph


def test_custom_pressure_callback_is_rejected_without_execution():
    graph = _graph()
    calls = []

    def custom(live):
        calls.append(live)
        default_compute_delta_nfr(live)

    graph.graph["compute_delta_nfr"] = custom
    before = _signature(graph)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _run(graph)
    assert not calls
    assert _signature(graph) == before


@pytest.mark.parametrize("graph_type", (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
def test_non_simple_undirected_graphs_are_outside_the_runtime_contract(graph_type):
    graph = graph_type(_graph())
    before = _signature(graph)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _run(graph)
    assert _signature(graph) == before


def test_carried_binding_cannot_be_transplanted_into_a_graph_copy():
    graph = _graph()
    _run(graph)
    copied = graph.copy()
    before = _signature(copied)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _run(copied, 0.0)
    assert _signature(copied) == before
    assert copied.graph[SLOT] is graph.graph[SLOT]


def test_first_present_alias_precedence_matches_the_shared_scalar_owner():
    graph = _graph()
    for i in graph:
        for alias in ALIAS_EPI:
            graph.nodes[i][alias] = (0.5, 0.75)[i]
        for alias in (*ALIAS_DEPI, *ALIAS_D2EPI):
            graph.nodes[i][alias] = 0.0
    result = _run(graph, 0.25)
    step = result.flows[0].step
    for i in graph:
        assert graph.nodes[i][ALIAS_EPI[0]] == step.after.epi[i]
        assert graph.nodes[i][ALIAS_DEPI[0]] == (0.25, -0.25)[i]
        assert graph.nodes[i][ALIAS_D2EPI[0]] == (1.0, -1.0)[i]
        assert all(graph.nodes[i][key] == (0.5, 0.75)[i] for key in ALIAS_EPI[1:])
        assert all(
            graph.nodes[i][key] == 0.0 for key in (*ALIAS_DEPI[1:], *ALIAS_D2EPI[1:])
        )


def test_late_exact_band_failure_rolls_back_history_carry_and_aliases():
    graph = _graph()
    _run(graph)
    binding = graph.graph[SLOT]
    shared = ["before"]
    generator = Random(71)
    graph.graph.update(shared=shared, shared_alias=shared, local_rng=generator)
    before = _signature(graph)
    schedule = _schedule(graph, durations=(5.125,))
    partition = build_physical_flow_partition(schedule.intervals[0], (0.125, 5.0))
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        runtime.execute_nodal_remainder_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
        )
    assert _signature(graph) == before
    assert graph.graph[SLOT] is binding
    assert graph.graph["shared"] is shared is graph.graph["shared_alias"]
    assert graph.graph["local_rng"] is generator


def test_result_copy_cannot_recreate_executor_provenance():
    graph = _graph()
    result = _run(graph)
    assert result.runtime_provenance_certified
    assert result.solver_accuracy_certified is False
    assert result.future_or_repeated_schedule_stability_certified is False
    detached = replace(result)
    assert detached.runtime_provenance_certified is False


@pytest.mark.parametrize(
    "target,field",
    (
        ("result", "execution_identity"),
        ("result", "initial_binding"),
        ("binding", "state"),
        ("binding", "binding_identity"),
        ("flow", "step"),
        ("refresh", "after_snapshot"),
    ),
)
def test_missing_evidence_slots_fail_closed(target, field):
    graph = _graph()
    result = _run(graph)
    owner = {
        "result": result,
        "binding": result.final_binding,
        "flow": result.flows[0],
        "refresh": result.refresh_records[0],
    }[target]
    object.__delattr__(owner, field)
    assert result.runtime_provenance_certified is False


def test_nested_flow_input_tampering_invalidates_the_complete_runtime_result():
    graph = _graph()
    result = _run(graph)
    flow = result.flows[0]
    changed = replace(flow.step, pressure=(0.0, 0.0))
    object.__setattr__(flow, "step", changed)
    assert result.runtime_provenance_certified is False


def test_phase_observations_bind_two_live_c6_cycles_to_the_existing_projection():
    from tnfr.physics.c6_phase_orbit import observe_c6_coupling_coherence_phase_step

    phase = tuple(i * math.pi / 3 for i in range(6))
    expected_first = observe_c6_coupling_coherence_phase_step(phase=phase)
    expected_second = observe_c6_coupling_coherence_phase_step(
        phase=expected_first.phase_after_coherence
    )
    graph = nx.cycle_graph(6)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0, RANDOM_SEED=17, compute_delta_nfr=default_compute_delta_nfr
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=0.5,
            nu_f=1.0,
            theta=phase[node],
            delta_nfr=0.0,
            dEPI=0.0,
            SI=0.5,
            glyph_history=[],
        )
    for edge in graph.edges:
        graph.edges[edge].update(weight=1.0, length=1.0)
    names = ("coupling", "coherence") * 2 + ("silence",)
    schedule = _schedule(graph, names, (0.0, 0.0, 0.25, 0.0, 0.25, 0.0))
    partitions = tuple(
        build_physical_flow_partition(interval, (0.0625,) * 4)
        for interval in schedule.intervals
        if interval.duration > 0
    )
    result = runtime.execute_nodal_remainder_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=partitions,
        epi_lower=0.375,
        epi_upper=0.625,
    )
    for cycle, expected in enumerate((expected_first, expected_second)):
        um, il = result.events[2 * cycle : 2 * cycle + 2]
        assert um.phase_before == expected.phase_before
        assert um.phase_after == il.phase_before == expected.phase_after_coupling
        assert il.phase_after == expected.phase_after_coherence
        for flow in result.flows[4 * cycle : 4 * cycle + 4]:
            assert flow.phase_before == flow.phase_after == il.phase_after
            assert flow.step.capacity == (1.0,) * 6
    terminal = result.events[-1]
    assert (
        terminal.phase_before
        == terminal.phase_after
        == expected_second.phase_after_coherence
    )
    assert terminal.phase_after == tuple(graph.nodes[i]["theta"] for i in graph)
    assert terminal.before_snapshot.nu_f == (1.0,) * 6
    assert all(value < 1.0 for value in terminal.after_snapshot.nu_f)
    assert result.runtime_provenance_certified


@pytest.mark.parametrize(
    "location,field",
    (
        ("flow", "phase_before"),
        ("flow", "phase_after"),
        ("event", "phase_before"),
        ("event", "phase_after"),
    ),
)
def test_phase_evidence_tampering_invalidates_its_executor_seal(location, field):
    graph = _graph()
    names = ("coupling", "coherence", "silence")
    result = runtime.execute_nodal_remainder_event_schedule(
        graph,
        _schedule(graph, names, (0.125, 0.0, 0.0, 0.0)),
    )
    assert result.runtime_provenance_certified
    record = result.flows[0] if location == "flow" else result.events[0]
    values = getattr(record, field)
    object.__setattr__(
        record, field, (math.nextafter(values[0], math.inf),) + values[1:]
    )
    assert result.runtime_provenance_certified is False


def test_flow_phase_capture_preserves_the_stored_signed_zero():
    graph = _graph()
    graph.nodes[0]["theta"] = -0.0
    result = _run(graph)
    flow = result.flows[0]
    assert tuple(value.hex() for value in flow.phase_before) == (
        "-0x0.0p+0",
        "0x0.0p+0",
    )
    assert tuple(value.hex() for value in flow.phase_after) == tuple(
        value.hex() for value in flow.phase_before
    )
    assert result.runtime_provenance_certified


@pytest.mark.parametrize(
    "values,default,expected",
    (
        ({"theta": 0.25, "phase": 0.75}, 0.0, 0.25),
        ({"phase": 0.75}, 0.0, 0.75),
        ({}, 0.0, 0.0),
        ({}, 0.125, 0.125),
        ({"theta": -0.0}, 0.0, -0.0),
    ),
)
def test_scalar_capture_preserves_alias_priority_without_virtual_mapping_reads(
    values, default, expected
):
    class ReadTrap(dict):
        def __contains__(self, key):
            raise AssertionError("scalar capture dispatched virtual membership")

        def __getitem__(self, key):
            raise AssertionError("scalar capture dispatched virtual item lookup")

    observed = runtime._scalar(ReadTrap(values), ALIAS_THETA, "phase", default)
    assert type(observed) is float and observed.hex() == expected.hex()


@pytest.mark.parametrize(
    "invalid", (True, 0, Fraction(1, 2), float("nan"), float("inf"), -float("inf"))
)
def test_scalar_capture_rejects_invalid_preferred_alias_instead_of_using_fallback(
    invalid,
):
    with pytest.raises(ValueError, match="actual finite scalar float"):
        runtime._scalar({"theta": invalid, "phase": 0.75}, ALIAS_THETA, "phase")


@pytest.mark.parametrize("alias", ("theta", "phase"))
def test_string_subclass_core_alias_cannot_silently_become_a_default_phase(alias):
    class CoreKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            raise AssertionError("capture dispatched virtual string equality")

        def __str__(self):
            raise AssertionError("capture dispatched virtual string conversion")

    with pytest.raises(ValueError, match="exact string keys"):
        runtime._scalar({CoreKey(alias): 0.25}, ALIAS_THETA, "phase")
    # A non-core auxiliary string key has no alias semantics and is not read.
    assert (
        runtime._scalar({CoreKey("auxiliary"): 7, "phase": 0.75}, ALIAS_THETA, "phase")
        == 0.75
    )


@pytest.mark.parametrize("hook", ("contains", "getitem"))
def test_phase_capture_cannot_mutate_auxiliary_graph_state_via_mapping_hooks(hook):
    graph = _graph()
    graph.graph["phase_capture_side_effect"] = 0

    def mutate_during_capture(key, active_hook):
        if key != "theta" or active_hook != hook:
            return
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "_phase_tuple":
                graph.graph["phase_capture_side_effect"] += 1
                return
            frame = frame.f_back

    class PhaseReadMutation(dict):
        def __contains__(self, key):
            mutate_during_capture(key, "contains")
            return dict.__contains__(self, key)

        def __getitem__(self, key):
            mutate_during_capture(key, "getitem")
            return dict.__getitem__(self, key)

    graph._node[0] = PhaseReadMutation(graph._node[0])
    result = _run(graph, 0.125)
    assert result.runtime_provenance_certified
    assert graph.graph["phase_capture_side_effect"] == 0
    assert result.flows[0].phase_before == result.flows[0].phase_after == (0.0, 0.0)
    assert result.final_binding.state.epi == (0.53125, 0.71875)


@pytest.mark.parametrize("change", ("epi", "kind", "support", "carry", "carry_payload"))
def test_actual_event_with_forbidden_side_effect_restores_the_complete_graph(
    monkeypatch, change
):
    graph = _graph()
    _run(graph)
    binding = graph.graph[SLOT]
    before = _signature(graph)
    original = runtime.execute_network_operator_stage
    reached = []

    def faulty_stage(live, *args, **kwargs):
        stage = original(live, *args, **kwargs)
        reached.append(stage)
        if change == "epi":
            live.nodes[0][ALIAS_EPI[0]] = math.nextafter(_epi(live)[0], math.inf)
        elif change == "kind":
            live.nodes[0]["epi_kind"] = "changed_kind"
        elif change == "support":
            live.remove_edge(0, 1)
        elif change == "carry":
            live.graph[SLOT] = replace(live.graph[SLOT])
        else:
            state = live.graph[SLOT].state
            object.__setattr__(state, "remainder", (Fraction(0), Fraction(0)))
        return stage

    monkeypatch.setattr(runtime, "execute_network_operator_stage", faulty_stage)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        runtime.execute_nodal_remainder_event_schedule(
            graph,
            _schedule(graph, ("coupling", "coherence", "silence")),
        )
    assert len(reached) == 1
    assert _signature(graph) == before
    assert graph.graph[SLOT] is binding
    assert binding.intact
