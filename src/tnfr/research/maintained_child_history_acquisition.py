"""One frozen numerical child continuation, canonical OZ and own history.

The caller freezes this producer and its budget before evaluation. No THOL or
closure is executed. Independent retained enclosures certify errors; interval
midpoints never initialize the numerical solution or supply runtime samples.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from time import perf_counter

from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from ..dynamics.integrators import rk4_coupled_step
from ..dynamics.runtime import _record_mutation_flow_boundary
from ..operators.definitions import Dissonance, SelfOrganization, Silence
from ..operators.grammar_debt import node_debt
from ..operators.grammar_dynamics import validate_candidate
from ..operators.grammar_execution import ValidatedSequence
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from ..operators.word_execution import execute_network_operator_stage
from .local_capacity_history import (
    CONTEXT,
    _state,
    capture,
    encode_result,
    numerical_field,
    refresh_pressure,
)
from .maintained_child_history import CHILD, CHILD_INDEX
from .maintained_child_oz_timing import _admit_child_oz_timing_source
from .maintained_child_sample_certificate import (
    _assess_child_boundary,
    _assess_child_samples,
    prepare_child_sample_certificate,
)
from .maintained_parent_thol import _represented, _signature
from .maintained_parent_um_execution import _pressure_preservation_signature
from .retained_graph_state import record_graph_state, restore_recorded_graph_state

INTEGRATION_STEP = Q(1, 2**14)
MAXIMUM_STEPS = 416
WALL_BUDGET_SECONDS = 600


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _primary(graph):
    field = capture(graph)
    return (
        field.nodes,
        field.epi,
        field.phase,
        field.capacity,
        tuple((node, tuple(graph[node])) for node in graph),
        tuple((a, b, dict(data)) for a, b, data in graph.edges(data=True)),
        _represented(graph.graph["_t"], "structural clock"),
    )


def _pressures(graph):
    return tuple(float(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in graph)


def _guarded_refresh(graph):
    protected = _pressure_preservation_signature(graph, runtime=True)
    refresh_pressure(graph)
    _require(
        _pressure_preservation_signature(graph, runtime=True) == protected,
        "child history refresh changed non-pressure state",
    )


def _install_state(graph, state, time):
    count = len(graph)
    _require(len(state) == 2 * count, "both complete rows required")
    for index, node in enumerate(graph):
        set_attr(graph.nodes[node], ALIAS_EPI, state[index])
        set_attr(graph.nodes[node], ALIAS_THETA, state[count + index])
    graph.graph["_t"] = float(time)
    _require(Q(graph.graph["_t"]) == time, "boundary clock was rounded")


def _advance(state, field, start, end, *, first_step, emit, check_budget):
    count = (end - start) / INTEGRATION_STEP
    _require(count.denominator == 1 and count >= 0, "fixed dyadic schedule required")
    _require(
        first_step + count <= MAXIMUM_STEPS, "fixed numerical step budget exceeded"
    )
    for offset in range(int(count)):
        check_budget()
        time = start + offset * INTEGRATION_STEP
        stages = []

        def recorded_field(stage_time, values):
            rates = field(stage_time, values)
            stages.append(
                {
                    "time": Q(stage_time),
                    "state": tuple(map(Q, values)),
                    "rates": tuple(map(Q, rates)),
                }
            )
            return rates

        next_state = rk4_coupled_step(
            state, recorded_field, t=float(time), dt=float(INTEGRATION_STEP)
        )
        _require(len(stages) == 4, "classical RK4 must evaluate four complete rows")
        residuals = tuple(
            Q(value)
            - Q(old)
            - INTEGRATION_STEP
            * sum(
                weight * row["rates"][index]
                for weight, row in zip((1, 2, 2, 1), stages, strict=True)
            )
            / 6
            for index, (old, value) in enumerate(zip(state, next_state, strict=True))
        )
        emit(
            "step",
            {
                "index": first_step + offset,
                "start": time,
                "end": time + INTEGRATION_STEP,
                "stages": tuple(stages),
                "endpoint": tuple(map(Q, next_state)),
                "represented_quadrature_residuals": residuals,
            },
        )
        state = next_state
    return state, first_step + int(count)


def _acquire_child_history(certificate, *, on_record=None):
    """Internal execution on one fresh preparation, with a detached raw sink.

    The public entry point reloads source evidence. A sink exception or any
    failed admission stops this attempt. Already written raw evidence remains;
    no suffix completion or external-side-effect rollback is inferred.
    """
    if on_record is not None and not callable(on_record):
        raise TypeError("record sink must be callable")
    started = perf_counter()
    source = certificate.source
    original = source.retention.admitted.graph
    original_signature = _signature(original)
    timing = _admit_child_oz_timing_source(source)
    graph = restore_recorded_graph_state(record_graph_state(original)).graph
    _require(_signature(graph) == original_signature, "acquisition clone differs")

    def check_budget():
        if perf_counter() - started > WALL_BUDGET_SECONDS:
            raise TimeoutError("frozen child acquisition wall budget exceeded")

    def emit(kind, record):
        protected = _signature(graph)
        if on_record is not None:
            on_record(kind, deepcopy(record))
        _require(_signature(graph) == protected, "record sink changed live graph")

    def associate(index):
        result = _assess_child_boundary(
            certificate,
            index=index,
            state=_state(graph),
            pressures=_pressures(graph),
            sample_time=graph.graph["_t"],
        )
        _require(result.passed, f"child boundary {index} failed error association")
        return result

    emit(
        "source",
        {
            "endpoint": record_graph_state(graph),
            "timing_admission": encode_result(timing),
            "source_receipts": source.receipts,
        },
    )
    protected = _signature(graph)
    field = numerical_field(capture(graph))
    _require(_signature(graph) == protected, "field construction changed live graph")
    state, steps = _state(graph), 0
    start = timing.source_time
    states, pressures, observations = [], [], []
    word = ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()), context=CONTEXT
    )
    stage, raw_event = None, []
    for index, time in enumerate(certificate.sample_times):
        protected = _signature(graph)
        state, steps = _advance(
            state,
            field,
            start,
            time,
            first_step=steps,
            emit=emit,
            check_budget=check_budget,
        )
        _require(
            _signature(graph) == protected, "numerical continuation changed live graph"
        )
        _install_state(graph, state, time)
        _guarded_refresh(graph)
        association = associate(index)
        if index == 0:
            before = _primary(graph)
            emit("event_before", record_graph_state(graph))

            def handoff(live):
                _require(not raw_event, "one OZ permits exactly one pressure handoff")
                _require(_primary(live) == before, "OZ changed continuous coordinates")
                raw = record_graph_state(live)
                raw_event.append(raw)
                emit("event_raw", raw)
                _guarded_refresh(live)

            stage = execute_network_operator_stage(
                graph,
                Dissonance(),
                (CHILD,),
                sequence_context=word.step(0),
                compute_delta_nfr=handoff,
            )
            _require(len(raw_event) == 1, "OZ omitted its raw pressure handoff")
            _require(_primary(graph) == before, "OZ continuation changed primary state")
            _require(
                tuple(graph.nodes[CHILD].get("glyph_history", ())) == ("OZ",)
                and node_debt(graph.nodes[CHILD]) == 1,
                "OZ did not supply the declared child-owned prefix",
            )
            association = associate(index)
            emit(
                "event_after",
                {"endpoint": record_graph_state(graph), "stage": encode_result(stage)},
            )
        # All nodes retain their actual boundary observations, so the parent's
        # older acquired history is extended rather than left apparently live.
        # The child's first sample is written only after OZ/refresh commits.
        _record_mutation_flow_boundary(graph)
        states.append(_state(graph))
        pressures.append(_pressures(graph))
        observations.append(encode_result(association))
        emit(
            f"sample_{index}",
            {
                "index": index,
                "time": time,
                "endpoint": record_graph_state(graph),
                "state": tuple(map(Q, states[-1])),
                "pressures": tuple(map(Q, pressures[-1])),
                "association": encode_result(association),
            },
        )
        state, start = states[-1], time
    _require(steps == MAXIMUM_STEPS, "acquisition did not complete the fixed schedule")
    check_budget()
    child = graph.nodes[CHILD]
    history = tuple(child.get("epi_time_history", ()))
    expected = tuple(
        (float(time), state[CHILD_INDEX])
        for time, state in zip(certificate.sample_times, states, strict=True)
    )
    _require(history == expected, "child history differs from acquired states")
    _require(
        history[-1][0] == graph.graph["_t"], "child history has a stale live clock"
    )
    before_observers = _signature(graph)
    observation = observe_structural_acceleration(graph, CHILD)
    validate_self_organization_strict(graph, CHILD, emit_warnings=False)
    grammar = validate_candidate(graph, CHILD, "THOL", sequence_context=word.step(1))
    _require(
        _signature(graph) == before_observers, "detached readiness checks changed state"
    )
    assessment = _assess_child_samples(
        certificate,
        states=tuple(states),
        pressures=tuple(pressures),
        sample_times=certificate.sample_times,
    )
    _require(assessment.passed, "complete acquired history failed certification")
    exact_stencil = (
        Q(states[0][CHILD_INDEX])
        - 2 * Q(states[1][CHILD_INDEX])
        + Q(states[2][CHILD_INDEX])
    ) / (certificate.sample_times[1] - certificate.sample_times[0]) ** 2
    _require(
        observation.available
        and Q(observation.value) == exact_stencil
        and exact_stencil > timing.threshold
        and grammar.allowed,
        "acquired reader or grammar differs from the admitted contract",
    )
    _require(
        _signature(original) == original_signature,
        "acquisition changed source evidence",
    )
    return {
        "schema": "tnfr.maintained-child-history-response.v1",
        "status": "acquired",
        "numerical_steps": steps,
        "field_evaluations": 4 * steps,
        "sample_times": certificate.sample_times,
        "states": tuple(tuple(map(Q, value)) for value in states),
        "pressures": tuple(tuple(map(Q, value)) for value in pressures),
        "boundary_associations": tuple(observations),
        "assessment": encode_result(assessment),
        "observation": encode_result(observation),
        "grammar": encode_result(grammar),
        "endpoint": record_graph_state(graph),
        "stage": encode_result(stage),
        "executed_prefix": tuple(child["glyph_history"]),
        "outstanding_u2_debt": node_debt(child),
        "complete_word_executed": False,
        "second_birth_executed": False,
        "physical_observation": False,
        "source_receipts": source.receipts,
    }


def evaluate_child_history(evidence_dir, *, repository_root, on_record=None):
    """Acquire once after the caller freezes source, law, schedule and budget."""
    certificate = prepare_child_sample_certificate(
        evidence_dir, repository_root=repository_root
    )
    return _acquire_child_history(certificate, on_record=on_record)
