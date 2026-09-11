from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Any, Literal

from .._remesh_contract import DelayedRemeshConfiguration
from .remesh_history_stability import UniformRemeshHistoryStabilityCertificate

ExactPair = tuple[Fraction, Fraction]
Binary64Pair = tuple[float, float]

@dataclass(frozen=True, slots=True)
class Binary64RemeshPairRelativeDefectObservation:
    binary64_alpha: float
    alpha: Fraction
    beta: Fraction
    gamma: Fraction
    delta: Fraction
    binary64_epi_min: float
    binary64_epi_max: float
    epi_min: Fraction
    epi_max: Fraction
    clip_mode: Literal["hard", "soft"]
    binary64_current_pair: Binary64Pair
    binary64_local_pair: Binary64Pair
    binary64_global_pair: Binary64Pair
    exact_current_pair: ExactPair
    exact_local_pair: ExactPair
    exact_global_pair: ExactPair
    exact_ideal_pair: ExactPair
    runtime_raw_pair: Binary64Pair
    runtime_bounded_pair: Binary64Pair
    exact_runtime_raw_pair: ExactPair
    exact_runtime_bounded_pair: ExactPair
    clipping_intervened: tuple[bool, bool]
    exact_input_pairwise_jensen_denominator: Fraction
    exact_ideal_squared_separation: Fraction
    exact_raw_squared_separation: Fraction
    exact_bounded_squared_separation: Fraction
    exact_rounding_signed_squared_separation_defect: Fraction
    exact_clipping_signed_squared_separation_defect: Fraction
    exact_total_signed_squared_separation_defect: Fraction
    exact_relative_signed_defect_ratio: Fraction | None
    exact_minimum_nonnegative_relative_defect_bound: Fraction | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def pair_relative_defect_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def hard_clipping_pairwise_nonexpansive_certified(self) -> bool: ...
    @property
    def uniform_binary64_relative_defect_bound_certified(self) -> bool: ...
    @property
    def future_binary64_relative_defect_bound_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class UniformAlphaOneHardClipRemeshClassCertificate:
    node_order: tuple[Hashable, ...]
    exact_normalized_metric: tuple[Fraction, ...]
    configuration: DelayedRemeshConfiguration = field(repr=False)
    tau_local: int
    tau_global: int
    history_maxlen: int
    required_history_length: int
    epi_min: Fraction
    epi_max: Fraction
    alpha: Fraction
    clip_mode: Literal["hard"]
    remesh_certificate: UniformRemeshHistoryStabilityCertificate = field(
        repr=False
    )
    exact_uniform_relative_defect_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def alpha_one_hard_clip_class_certificate_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def binary64_global_delay_numeric_copy_certified(self) -> bool: ...
    @property
    def hard_clip_identity_on_class_certified(self) -> bool: ...
    @property
    def remesh_class_forward_invariant_certified(self) -> bool: ...
    def represented_history_belongs_to_class(self, history: Any) -> bool: ...
    @property
    def schedule_family_certificate_certified(self) -> bool: ...
    @property
    def repeated_binary64_stability_certified(self) -> bool: ...
    @property
    def future_binary64_execution_certified(self) -> bool: ...
    @property
    def solver_accuracy_certified(self) -> bool: ...
    @property
    def full_tnfr_stability_certified(self) -> bool: ...

def observe_binary64_remesh_pair_relative_defect(
    current_pair: Binary64Pair | list[float],
    local_pair: Binary64Pair | list[float],
    global_pair: Binary64Pair | list[float],
    *,
    alpha: Real,
    epi_min: Real,
    epi_max: Real,
    clip_mode: Literal["hard", "soft"] = "hard",
) -> Binary64RemeshPairRelativeDefectObservation: ...

def certify_alpha_one_hard_clip_remesh_class(
    nodes: Iterable[Hashable],
    metric_weights: Mapping[Hashable, Real] | Sequence[Real] | None = None,
    *,
    tau_local: int,
    tau_global: int,
    epi_min: Real,
    epi_max: Real,
) -> UniformAlphaOneHardClipRemeshClassCertificate: ...

__all__: tuple[str, ...]
