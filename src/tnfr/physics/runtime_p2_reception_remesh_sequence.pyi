from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from ..operators.event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence,
)
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
)
from .runtime_p2_reception_stage import (
    ExecutedP2HalfReceptionStageCertificate,
)
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
)

ExactPair = tuple[Fraction, Fraction]

@dataclass(frozen=True, slots=True)
class ExecutedP2HalfReceptionRemeshSequenceCertificate:
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate = field(
        repr=False
    )
    execution: ExecutedEventRemeshCycleSequence = field(repr=False)
    cycle_indices: tuple[int, ...]
    reception_event_indices: tuple[int, ...]
    reception_stage_certificates: tuple[
        ExecutedP2HalfReceptionStageCertificate, ...
    ] = field(repr=False)
    remesh_history_bridges: tuple[
        RuntimeRemeshHistoryBridgeObservation, ...
    ] = field(repr=False)
    node_order: tuple[Any, Any]
    exact_normalized_metric: ExactPair
    cycle_count: int
    active_history_extinction_horizon: int
    guaranteed_extinction_cycle_index: int
    exact_schedule_input_energies: tuple[Fraction, ...]
    exact_post_reception_energies: tuple[Fraction, ...]
    exact_post_remesh_energies: tuple[Fraction, ...]
    exact_active_history_suffixes: tuple[tuple[ExactPair, ...], ...]
    exact_final_post_remesh_energy: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def executed_p2_half_reception_remesh_sequence_certificate_certified(
        self,
    ) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def finite_causal_extinction_certified(self) -> bool: ...
    @property
    def same_invocation_causal_provenance_certified(self) -> bool: ...
    @property
    def every_reception_stage_q_zero_certified(self) -> bool: ...
    @property
    def every_remesh_global_delay_copy_eta_zero_certified(self) -> bool: ...
    @property
    def observed_active_history_extinction_certified(self) -> bool: ...
    @property
    def whole_sequence_graph_state_atomicity_certified(self) -> bool: ...
    @property
    def future_runtime_stability_certified(self) -> bool: ...
    @property
    def unobserved_repetition_stability_certified(self) -> bool: ...
    @property
    def auxiliary_state_stability_certified(self) -> bool: ...
    @property
    def current_live_graph_state_bound(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...

def certify_executed_p2_half_reception_remesh_sequence(
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate,
    execution: ExecutedEventRemeshCycleSequence,
    *,
    reception_event_indices: tuple[int, ...] | None = ...,
) -> ExecutedP2HalfReceptionRemeshSequenceCertificate: ...

__all__: tuple[str, ...]
