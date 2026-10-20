"""One prospectively frozen THOL reset on a certified maintained parent.

Occurrence is supplied. The post-event law explicitly freezes isolated nodes;
the connected-only reader remains a rejecting control. No flow is advanced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction as Q
from unittest.mock import patch

from ..alias import get_attr, set_dnfr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..operators import network_stage
from ..operators.definitions import SelfOrganization
from ..operators.grammar_debt import node_debt
from ..operators.nodal_equation import observe_structural_acceleration
from ..operators.word_execution import execute_network_operator_stage
from ..physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from .local_capacity_history import MODEL, PARENT, encode_result, refresh_pressure
from .maintained_parent_thol import validate_live_parent
from .retained_graph_state import record_graph_state, restore_recorded_graph_state


class _EventClock:
    @classmethod
    def now(cls, tz=None):
        return datetime(2030, 1, 2, 3, 4, 6, tzinfo=timezone.utc)


def _primary(graph, nodes):
    return tuple(
        tuple(
            Q(get_attr(graph.nodes[node], alias))
            for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
        )
        for node in nodes
    )


def _pressure(graph, nodes):
    return tuple(Q(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in nodes)


def _contains(interval, value):
    return interval.lo <= Q(value) <= interval.hi


def refresh_frozen_successor(graph):
    """Refresh each admitted component and explicitly assign isolate p=0.

    Reuse the original refined sine-pressure evaluator on each nontrivial
    component. Neither this callback nor the detached reader installs a native
    flow dispatcher or advances the structural clock.
    """
    source = bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
    for indices in source.component_indices:
        nodes = tuple(source.nodes[index] for index in indices)
        if len(nodes) == 1:
            set_dnfr(graph, nodes[0], 0.0)
        else:
            component = graph.subgraph(nodes).copy()
            refresh_pressure(component)
            for node in nodes:
                set_dnfr(graph, node, get_attr(component.nodes[node], ALIAS_DNFR))
    return source


def evaluate_thol_event(admitted, prediction):
    """Execute exactly one atomic stage after caller freezes all inputs.

    The admission object comes from the retained-evidence reader. Prediction
    is independently computed before the reserved event, then archived by the
    caller. Exceptions remain failures of this one attempt, not retry signals.
    """
    admitted = validate_live_parent(admitted)
    graph = admitted.graph
    old_nodes = tuple(graph)
    before = record_graph_state(graph)
    old_primary = _primary(graph, old_nodes)
    old_pressure = _pressure(graph, old_nodes)
    old_edges = tuple((u, v, dict(data)) for u, v, data in graph.edges(data=True))
    old_history = tuple(graph.nodes[PARENT]["epi_time_history"])
    before_fields = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    captured = []

    def handoff(successor):
        if captured:
            raise ValueError("one THOL event permits exactly one pressure handoff")
        captured.append(record_graph_state(successor))
        refresh_frozen_successor(successor)

    with patch.object(network_stage, "datetime", _EventClock):
        stage = execute_network_operator_stage(
            graph,
            SelfOrganization(),
            (PARENT,),
            sequence_context=admitted.sequence.step(1),
            compute_delta_nfr=handoff,
        )
    if len(captured) != 1:
        raise ValueError("the atomic event did not retain its raw pressure state")
    raw = restore_recorded_graph_state(captured[0]).graph
    successor = bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
    connected_rejection = None
    try:
        bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    except ValueError as exc:
        connected_rejection = {"type": type(exc).__name__, "message": str(exc)}

    child = prediction.child
    child_data = graph.nodes[child]
    parent = graph.nodes[PARENT]
    child_observation = observe_structural_acceleration(graph, child)
    parent_raw = Q(get_attr(raw.nodes[PARENT], ALIAS_DNFR))
    child_form = Q(get_attr(child_data, ALIAS_EPI))
    primary_same = _primary(graph, old_nodes) == old_primary
    support_same = (
        tuple((u, v, dict(data)) for u, v, data in graph.edges(data=True)) == old_edges
    )
    restriction = graph.subgraph(old_nodes).copy()
    old_after = bound_relational_sine_support_exchange(
        restriction, reference_model=MODEL
    )
    rho = tuple(
        Q(d) / nu for d, nu in zip(before_fields.degrees, before_fields.capacity)
    )
    mean_changes = tuple(
        sum(w * (after - previous) for w, after, previous in zip(rho, right, left))
        / sum(rho)
        for left, right in (
            (before_fields.epi, old_after.epi),
            (before_fields.phase, old_after.phase),
        )
    )
    child_index = successor.nodes.index(child)
    record = parent["sub_epis"][-1]
    checks = {
        "one_canonical_atomic_event": stage.glyph == "THOL"
        and stage.nodes_processed == 1
        and stage.schedule == "two_phase_jacobi",
        "one_predicted_child": tuple(graph) == old_nodes + (child,),
        "old_primary_coordinates_unchanged": primary_same,
        "old_support_and_conductances_unchanged": support_same,
        "child_amplitude_predicted": _contains(
            prediction.child_form_bounds, child_form
        ),
        "child_capacity_and_phase_predicted": (
            Q(get_attr(child_data, ALIAS_VF)) == prediction.child_capacity
            and Q(get_attr(child_data, ALIAS_THETA)) == prediction.child_phase
        ),
        "raw_parent_pressure_predicted": _contains(
            prediction.raw_pressure_bounds, parent_raw
        ),
        "raw_other_pressures_unchanged": all(
            node == PARENT
            or Q(get_attr(raw.nodes[node], ALIAS_DNFR)) == old_pressure[index]
            for index, node in enumerate(old_nodes)
        ),
        "child_isolated_with_zero_pressure": graph.degree(child) == 0
        and Q(get_attr(raw.nodes[child], ALIAS_DNFR))
        == Q(get_attr(child_data, ALIAS_DNFR))
        == 0,
        "hierarchy_committed": parent["sub_nodes"] == [child]
        and graph.graph["hierarchy"][PARENT] == [child]
        and child_data["parent_node"] == PARENT
        and child_data["hierarchy_level"] == child_data["_bifurcation_level"] == 1
        and child_data["_hierarchy_path"] == [PARENT]
        and record["node_id"] == child
        and record["metabolized"] is True,
        "child_lifecycle_prediction": record["timestamp"]
        == prediction.child_record_operator_step
        and Q(record["d2_epi"]) == prediction.observed_acceleration
        and Q(record["tau"]) == admitted.threshold
        and record["bifurcation_level"] == prediction.child_hierarchy_level
        and tuple(record["hierarchy_path"]) == prediction.child_hierarchy_path
        and child_data["glyph_history"] == [],
        "metabolic_signals_predicted": (
            record["network_signals"]["neighbor_count"]
            == prediction.metabolic.neighbor_count
            and abs(
                Q(record["network_signals"]["mean_neighbor_epi"])
                - prediction.metabolic.mean_neighbor_epi
            )
            <= prediction.numerical_allowance
            and abs(
                Q(record["network_signals"]["epi_gradient"])
                - prediction.metabolic.epi_gradient
            )
            <= prediction.numerical_allowance
            and prediction.metabolic.phase_variance.lo - prediction.numerical_allowance
            <= Q(record["network_signals"]["phase_variance"])
            <= prediction.metabolic.phase_variance.hi + prediction.numerical_allowance
            and prediction.metabolic.coupling_strength_mean.lo
            - prediction.numerical_allowance
            <= Q(record["network_signals"]["coupling_strength_mean"])
            <= prediction.metabolic.coupling_strength_mean.hi
            + prediction.numerical_allowance
        ),
        "actual_grammar_prefix": tuple(parent["glyph_history"]) == ("VAL", "THOL")
        and node_debt(parent) == 0,
        "clock_and_parent_record_retained": Q(graph.graph["_t"])
        == Q(old_history[-1][0])
        and tuple(parent["epi_time_history"]) == old_history,
        "child_temporal_baseline_unavailable": not child_observation.available
        and "epi_time_history" not in child_data
        and child_data["epi_history"] == [float(child_form)],
        "connected_only_reader_rejects": connected_rejection is not None,
        "fresh_old_pressure_restored": _pressure(graph, old_nodes) == old_pressure,
        "old_continuous_rows_preserved": old_after.form_rates
        == before_fields.form_rates
        and old_after.phase_rates == before_fields.phase_rates,
        "isolate_rows_explicitly_frozen": successor.form_rates[child_index].lo
        == successor.form_rates[child_index].hi
        == successor.phase_rates[child_index].lo
        == successor.phase_rates[child_index].hi
        == 0,
        "zero_edge_storage_jump_by_unchanged_arguments": primary_same
        and support_same
        and successor.form_storage == before_fields.form_storage
        and successor.phase_storage == before_fields.phase_storage,
        "old_component_means_preserved": mean_changes == (Q(0), Q(0)),
    }
    return {
        "schema": "tnfr.maintained-parent-thol-response.v1",
        "scope": "One supplied THOL occurrence on a certified retained parent; explicit frozen-isolate successor",
        "before": before,
        "raw_after": captured[0],
        "endpoint": record_graph_state(graph),
        "prediction": encode_result(prediction),
        "stage": encode_result(stage),
        "retained_source_receipts": admitted.receipts,
        "source_certificate": encode_result(admitted.certificate),
        "successor_fields": encode_result(successor),
        "connected_reader_rejection": connected_rejection,
        "child_observation": encode_result(child_observation),
        "parent_raw_pressure": parent_raw,
        "parent_pressure_increment": parent_raw - old_pressure[old_nodes.index(PARENT)],
        "old_component_mean_changes": mean_changes,
        "edge_storage_jump": Q(0) if primary_same and support_same else None,
        "global_positive_mean_metric_available": False,
        "numerical_u5_observation_available": False,
        "cross_birth_u6_observation_available": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
