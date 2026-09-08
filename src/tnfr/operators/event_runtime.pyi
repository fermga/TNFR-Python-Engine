from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import networkx as nx

from ..types import Glyph
from .event_timing import OperatorEventSchedule, ScheduledOperatorEvent
from .network_stage import NetworkStageResult


@dataclass(frozen=True, slots=True)
class ExecutedOperatorEvent:
    event_index: int
    cycle_index: int
    word_position: int
    operator_name: str
    glyph: Glyph
    event_time: float
    event_offset: Fraction
    exact_event_time: Fraction
    stage_schedule: str
    nodes_processed: int
    zero_duration: bool = field(default=..., init=False)
    history_channel: str = field(default=..., init=False)
    feeds_epi_time_history: bool = field(default=..., init=False)

    @classmethod
    def from_stage(
        cls,
        event: ScheduledOperatorEvent,
        result: NetworkStageResult,
    ) -> ExecutedOperatorEvent: ...
    def as_record(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class OperatorEventExecutionResult:
    schedule: OperatorEventSchedule
    target_nodes: tuple[Any, ...]
    flow_interval_indices: tuple[int, ...]
    positive_flow_interval_indices: tuple[int, ...]
    events: tuple[ExecutedOperatorEvent, ...]
    final_time: float
    integrator_name: str | None
    pressure_refresh_callback_invocations: int
    runtime_clock_checked: bool = field(default=..., init=False)
    flow_provenance: str = field(default=..., init=False)
    nodal_flow_inputs: str = field(default=..., init=False)
    whole_schedule_graph_state_atomic: bool = field(
        default=..., init=False
    )
    operator_jumps_have_zero_duration: bool = field(
        default=..., init=False
    )
    solver_accuracy_certified: bool = field(default=..., init=False)
    adaptive_u2_u4_policy: bool = field(default=..., init=False)
    external_side_effects_rolled_back: bool = field(
        default=..., init=False
    )
    flow_scope: str = field(default=..., init=False)


def execute_operator_event_schedule(
    graph: nx.Graph,
    schedule: OperatorEventSchedule,
    *,
    context: Mapping[str, Any] | None = ...,
    method: str | None = ...,
    n_jobs: int | None = ...,
    suppress_birth_warnings: bool = ...,
) -> OperatorEventExecutionResult: ...


__all__: tuple[str, str, str]