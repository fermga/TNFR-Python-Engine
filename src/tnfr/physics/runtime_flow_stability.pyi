from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class NodalFlowStateSnapshot:
    nodes: tuple[Any, ...]
    epi: tuple[float, ...]
    nu_f: tuple[float, ...]
    delta_nfr: tuple[float, ...]
    exact_epi: ExactVector
    exact_nu_f: ExactVector
    exact_delta_nfr: ExactVector
    conductance: ExactMatrix
    binary64_pure_epi_pressure: tuple[float, ...]
    exact_binary64_pure_epi_pressure: ExactVector


@dataclass(frozen=True, slots=True)
class NodalFlowIntervalCertificate:
    left: NodalFlowStateSnapshot
    right: NodalFlowStateSnapshot
    duration: float
    exact_duration: Fraction
    exact_nodal_equation_residual: ExactVector | None
    exact_nodal_equation_realized: bool
    stable_node_support: bool
    fixed_conductance: bool
    symmetric_nonnegative_conductance: bool
    positive_row_strength: bool
    positive_capacity: bool
    capacity_unchanged: bool
    pressure_unchanged: bool
    exact_pure_epi_pressure: ExactVector | None
    exact_pressure_residual: ExactVector | None
    exact_pure_epi_pressure_realized: bool
    exact_binary64_pressure_replay_residual: ExactVector | None
    binary64_pure_epi_pressure_realized: bool
    diffusion_conditions: tuple[tuple[str, bool], ...]
    pure_epi_diffusion_eligible: bool
    integrator_name: str | None
    method: str | None
    substeps: int | None
    gamma_is_none: bool | None
    clipping_applied: bool | None
    extended_dynamics_requested: bool | None
    runtime_conditions: tuple[tuple[str, bool], ...]
    runtime_euler_eligible: bool
    binary64_euler_replay: tuple[float, ...] | None
    exact_binary64_euler_replay_residual: ExactVector | None
    binary64_euler_replay_matches: bool
    binary64_runtime_interval_identified: bool
    exact_metric_weights: ExactVector | None
    exact_explicit_euler_map: ExactMatrix | None
    exact_explicit_euler_endpoint_residual: ExactVector | None
    explicit_euler_map_identified: bool
    exact_left_disagreement_energy: Fraction | None
    exact_right_disagreement_energy: Fraction | None
    exact_observed_disagreement_energy_gain: Fraction | None
    observed_disagreement_nonincrease: bool | None
    exact_quotient_energy_gain_upper_bound: Fraction | None
    _global_disagreement_contraction_certified: bool
    integrator_provenance_certified: bool
    future_or_repeated_schedule_stability_certified: bool
    scope: str
    binary64_substep_duration: float | None = ...
    exact_binary64_substep_duration: Fraction | None = ...
    exact_binary64_substep_duration_sum: Fraction | None = ...
    exact_substep_duration_sum_matches_interval: bool = ...
    binary64_held_pressure_replay: tuple[float, ...] | None = ...
    exact_binary64_held_pressure_replay_residual: ExactVector | None = ...
    binary64_held_pressure_replay_matches: bool = ...
    held_pressure_runtime_conditions: tuple[tuple[str, bool], ...] = ...
    _binary64_held_pressure_runtime_identified: bool = field(...)
    _proof_stamp: tuple[Any, ...] = field(...)

    def _proof_fields_are_intact(self) -> bool: ...

    @property
    def binary64_held_pressure_runtime_identified(self) -> bool: ...
    @property
    def nodes(self) -> tuple[Any, ...]: ...
    @property
    def left_epi(self) -> tuple[float, ...]: ...
    @property
    def right_epi(self) -> tuple[float, ...]: ...
    @property
    def left_nu_f(self) -> tuple[float, ...]: ...
    @property
    def left_delta_nfr(self) -> tuple[float, ...]: ...
    @property
    def global_disagreement_contraction_certified(self) -> bool: ...
    @property
    def failed_diffusion_conditions(self) -> tuple[str, ...]: ...
    @property
    def failed_runtime_conditions(self) -> tuple[str, ...]: ...
    @property
    def failed_held_pressure_runtime_conditions(self) -> tuple[str, ...]: ...
    @property
    def euler_map_abstention_reasons(self) -> tuple[str, ...]: ...


def capture_nodal_flow_state(
    graph: Any,
    *,
    nodes: Iterable[Any] | None = ...,
) -> NodalFlowStateSnapshot: ...


def certify_observed_nodal_flow_interval(
    left: NodalFlowStateSnapshot,
    right: NodalFlowStateSnapshot,
    *,
    duration: Any,
    integrator_name: str | None = ...,
    method: str | None = ...,
    substeps: int | None = ...,
    gamma_is_none: bool | None = ...,
    clipping_applied: bool | None = ...,
    extended_dynamics_requested: bool | None = ...,
) -> NodalFlowIntervalCertificate: ...


__all__: list[str]
