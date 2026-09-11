"""Coherence snapshot-stage, telemetry, and lifecycle regressions."""

from __future__ import annotations

from copy import deepcopy
import math
import sys
import warnings
from typing import Any

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import COHERENCE_RETENTION
from tnfr.operators._coherence_stage_kernel import propose_coherence_pressure
from tnfr.operators.definitions import Coherence
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.metrics.local_coherence import compute_radius_structural_coherence
from tnfr.operators.network_stage import (
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_pointwise_stage,
)


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph["NAV_RANDOM"] = False
    for node, (pressure, phase) in enumerate(((-0.8, 0.0), (0.4, 1.0))):
        graph.nodes[node].update(
            **{
                ALIAS_EPI[0]: 0.5,
                ALIAS_VF[0]: 1.0,
                ALIAS_DNFR[0]: pressure,
                "dEPI_dt": 0.0,
                ALIAS_THETA[0]: phase,
                "glyph_history": ["AL", "OZ"],
            },
        )
    return graph


def _primary(graph: nx.Graph) -> dict[Any, tuple[float, float]]:
    return {
        node: (
            float(get_attr(data, ALIAS_DNFR)),
            float(get_attr(data, ALIAS_THETA)),
        )
        for node, data in graph.nodes(data=True)
    }


def _plain_state(graph: nx.Graph) -> tuple[object, ...]:
    return (
        tuple((node, deepcopy(dict(data))) for node, data in graph.nodes(data=True)),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(dict(graph.graph)),
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


def test_two_node_phase_lock_is_jacobi_and_reverse_order_invariant() -> None:
    forward = _graph()
    reverse = _graph()

    forward_result = execute_pointwise_stage(
        forward, Coherence(), (0, 1), phase_locking_coefficient=0.3
    )
    reverse_result = execute_pointwise_stage(
        reverse, Coherence(), (1, 0), phase_locking_coefficient=0.3
    )

    assert forward_result.schedule == TWO_PHASE_JACOBI
    assert reverse_result.schedule == TWO_PHASE_JACOBI
    for node, expected in {
        0: (-0.8 * COHERENCE_RETENTION, 0.3),
        1: (0.4 * COHERENCE_RETENTION, 0.7),
    }.items():
        assert _primary(forward)[node] == pytest.approx(expected)
        assert _primary(reverse)[node] == pytest.approx(expected)
    assert [
        event["node"] for event in reverse.graph["IL_phase_locking"]
    ] == [1, 0]


def test_stage_telemetry_preserves_legacy_dispersion_and_adds_structural_c() -> None:
    graph = _graph()

    execute_pointwise_stage(graph, Coherence(), (1, 0))

    events = graph.graph["IL_coherence_tracking"]
    assert [event["node"] for event in events] == [1, 0]
    assert all(event["scope"] == "stage" for event in events)
    for event in events:
        assert event["C_global_before"] == event["C_dispersion_global_before"]
        assert event["C_global_after"] == event["C_dispersion_global_after"]
        assert event["C_local_before"] == event["C_dispersion_local_before"]
        assert event["C_local_after"] == event["C_dispersion_local_after"]
        assert event["legacy_C_fields_deprecated"] is True
        assert event["C_t_global_before"] == pytest.approx(0.625)
        assert event["C_t_global_after"] == pytest.approx(
            1.0 / (1.0 + 0.6 * COHERENCE_RETENTION)
        )
        assert 0.0 <= event["C_structural_local_after"] <= 1.0


def test_direct_single_target_matches_shared_kernel_and_stage() -> None:
    direct = _graph()
    staged = _graph()

    Coherence()(direct, 0, phase_locking_coefficient=0.4)
    execute_pointwise_stage(
        staged, Coherence(), (0,), phase_locking_coefficient=0.4
    )

    assert _primary(direct)[0] == pytest.approx(_primary(staged)[0])
    assert direct.graph["IL_phase_locking"] == staged.graph["IL_phase_locking"]
    assert direct.graph["IL_dnfr_reductions"] == staged.graph["IL_dnfr_reductions"]
    direct_tracking = dict(direct.graph["IL_coherence_tracking"][-1])
    staged_tracking = dict(staged.graph["IL_coherence_tracking"][-1])
    assert direct_tracking.pop("scope") == "direct"
    assert staged_tracking.pop("scope") == "stage"
    assert direct_tracking == staged_tracking


def test_negative_pressure_reports_magnitude_reduction_in_metrics() -> None:
    graph = _graph()

    Coherence()(graph, 0, collect_metrics=True)

    reduction = graph.graph["IL_dnfr_reductions"][-1]
    assert reduction["before"] == pytest.approx(-0.8)
    assert reduction["after"] == pytest.approx(-0.8 * COHERENCE_RETENTION)
    assert reduction["reduction"] == pytest.approx(0.8 * (1.0 - COHERENCE_RETENTION))
    assert reduction["reduction_factor"] == pytest.approx(1.0 - COHERENCE_RETENTION)
    assert reduction["signed_delta"] == pytest.approx(
        0.8 * (1.0 - COHERENCE_RETENTION)
    )
    metrics = graph.graph["operator_metrics"][-1]
    expected_reduction = 0.8 * (1.0 - COHERENCE_RETENTION)
    assert metrics["dnfr_reduction"] == pytest.approx(expected_reduction)
    assert metrics["dnfr_reduction_pct"] == pytest.approx(
        100.0 * (1.0 - COHERENCE_RETENTION)
    )
    assert metrics["stability_gain"] == pytest.approx(expected_reduction)


def test_radius_structural_coherence_avoids_finite_mean_overflow() -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node][ALIAS_DNFR[0]] = sys.float_info.max

    value = compute_radius_structural_coherence(graph, 0, radius=1)

    assert math.isfinite(value)
    assert value > 0.0


def test_public_pressure_kernel_rejects_noncontracting_endpoint() -> None:
    with pytest.raises(GlyphFactorValidationError, match=r"\[0.0, 1.0\)"):
        propose_coherence_pressure(-0.8, 1.0)


class _TimingMonitor:
    def __init__(self) -> None:
        self.events: list[tuple[str, float, int, int, int]] = []

    def before_operator(self, graph: nx.Graph, node: Any) -> None:
        self.events.append(
            (
                "before",
                float(get_attr(graph.nodes[node], ALIAS_THETA)),
                len(graph.graph.get("IL_phase_locking", ())),
                len(graph.graph.get("IL_dnfr_reductions", ())),
                len(graph.graph.get("IL_coherence_tracking", ())),
            )
        )

    def after_operator(self, graph: nx.Graph, node: Any, operator: str) -> None:
        assert operator == "coherence"
        self.events.append(
            (
                "after",
                float(get_attr(graph.nodes[node], ALIAS_THETA)),
                len(graph.graph.get("IL_phase_locking", ())),
                len(graph.graph.get("IL_dnfr_reductions", ())),
                len(graph.graph.get("IL_coherence_tracking", ())),
            )
        )


def test_direct_monitor_observes_phase_and_telemetry_commit() -> None:
    graph = _graph()
    monitor = _TimingMonitor()
    graph.graph["integrity_monitor"] = monitor

    Coherence()(graph, 0)

    assert monitor.events[0] == ("before", 0.0, 0, 0, 0)
    assert monitor.events[1][0] == "after"
    assert monitor.events[1][1:] == pytest.approx((0.3, 1, 1, 1))


def test_strict_readiness_accepts_negative_pressure_and_ignores_epi_headroom() -> None:
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.9
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.graph["IL_PRECONDITIONS"] = {
        "max_epi": 0.5,
        "dnfr_critical_threshold": 1.0,
    }

    Coherence()(graph, 0)

    assert float(get_attr(graph.nodes[0], ALIAS_DNFR)) == pytest.approx(
        -0.8 * COHERENCE_RETENTION
    )


def test_late_pressure_failure_rolls_back_complete_il_stage_without_warning() -> None:
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.graph["IL_PRECONDITIONS"] = {"warn_zero_dnfr": True}
    for node in graph:
        graph.nodes[node][ALIAS_DNFR[0]] = 0.0
    before = _plain_state(graph)

    def reject_refresh(subject: nx.Graph) -> None:
        subject.graph["refresh_side_effect"] = True
        subject.nodes[0][ALIAS_THETA[0]] = 5.0
        raise RuntimeError("late IL refresh failure")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="late IL refresh failure"):
            execute_pointwise_stage(
                graph,
                Coherence(),
                (0, 1),
                compute_delta_nfr=reject_refresh,
            )

    assert caught == []
    assert _plain_state(graph) == before


def test_il_warnings_emit_once_per_target_only_after_successful_refresh() -> None:
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    for node in graph:
        graph.nodes[node][ALIAS_DNFR[0]] = 0.0
    callback_calls = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def refresh(subject: nx.Graph) -> None:
            nonlocal callback_calls
            callback_calls += 1
            assert subject is graph
            assert caught == []

        execute_pointwise_stage(
            graph,
            Coherence(),
            (1, 0),
            compute_delta_nfr=refresh,
        )

    assert callback_calls == 1
    assert len(caught) == 2
    assert all("|ΔNFR|=0" in str(item.message) for item in caught)


def test_il_warning_promoted_to_error_rolls_back_committed_stage() -> None:
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    for node in graph:
        graph.nodes[node][ALIAS_DNFR[0]] = 0.0
    before = _plain_state(graph)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=r"\|ΔNFR\|=0"):
            execute_pointwise_stage(graph, Coherence(), (0, 1))

    assert _plain_state(graph) == before


def test_grammar_replacement_uses_transactional_gs_instead_of_il_jacobi() -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = 0.0
        graph.nodes[node]["glyph_history"] = []

    result = execute_pointwise_stage(graph, Coherence(), (0, 1))

    assert result.schedule == OPERATOR_MAJOR_GAUSS_SEIDEL
    assert graph.graph[STAGE_SCHEDULE_KEY]["schedule"] == (
        OPERATOR_MAJOR_GAUSS_SEIDEL
    )
    assert all(graph.nodes[node]["glyph_history"][-1] != "IL" for node in graph)
