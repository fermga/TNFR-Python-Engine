from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral, Real
from typing import Any

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[ExactVector, ...]
ExactHistory = tuple[ExactVector, ...]

@dataclass(frozen=True, slots=True)
class UniformRemeshHistoryStabilityCertificate:
    alpha: Fraction
    tau_local: int
    tau_global: int
    beta: Fraction
    gamma: Fraction
    delta: Fraction
    combined_delay_coefficients: tuple[tuple[int, Fraction], ...]
    active_delays: tuple[int, ...]
    active_max_delay: int
    companion_matrix: ExactMatrix
    stationary_denominator: Fraction
    stationary_distribution: ExactVector
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def stability_certificate_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def companion_row_stochastic_certified(self) -> bool: ...
    @property
    def stationary_distribution_certified(self) -> bool: ...
    @property
    def jensen_lyapunov_nonincrease_certified(self) -> bool: ...
    @property
    def equality_characterization_certified(self) -> bool: ...
    @property
    def alpha_zero_identity_map_certified(self) -> bool: ...
    @property
    def alpha_one_pure_delay_map_certified(self) -> bool: ...
    @property
    def alpha_one_augmented_energy_conservation_certified(self) -> bool: ...
    @property
    # Order of the companion permutation; orbit periods may divide it.
    def pure_delay_period(self) -> int | None: ...
    @property
    def periodic_temporal_cycles_possible(self) -> bool: ...
    @property
    def strict_mixing_companion_primitive_certified(self) -> bool: ...
    @property
    def strict_mixing_pointwise_temporal_convergence_certified(self) -> bool: ...
    @property
    def spatial_consensus_certified(self) -> bool: ...
    @property
    def zero_pressure_equilibrium_certified(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class UniformRemeshHistoryTransitionObservation:
    certificate: UniformRemeshHistoryStabilityCertificate
    nodes: tuple[Hashable, ...]
    exact_metric_weights: ExactVector
    exact_history: ExactHistory
    exact_centered_history: ExactHistory
    exact_history_energies: ExactVector
    exact_next_field: ExactVector
    exact_next_centered_field: ExactVector
    exact_next_energy: Fraction
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_energy_drop: Fraction
    exact_jensen_dissipation: Fraction
    active_centered_fields: tuple[tuple[int, ExactVector], ...]
    active_centered_fields_pairwise_equal: bool
    lyapunov_nonincreasing: bool
    lyapunov_equality: bool
    exact_stationary_history_barycenter: ExactVector
    exact_post_transition_stationary_history_barycenter: ExactVector
    exact_strict_mixing_temporal_limit: ExactVector | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(...)
    def _proof_fields_are_intact(self) -> bool: ...
    @property
    def scope(self) -> str: ...
    @property
    def transition_observation_certified(self) -> bool: ...
    @property
    def failed_conditions(self) -> tuple[str, ...]: ...
    @property
    def exact_dissipation_identity_certified(self) -> bool: ...
    @property
    def equality_iff_active_centered_fields_agree_certified(self) -> bool: ...
    @property
    def spatial_consensus_certified(self) -> bool: ...
    @property
    def zero_pressure_equilibrium_certified(self) -> bool: ...

def certify_uniform_remesh_history_stability(
    *,
    alpha: Real,
    tau_local: Integral,
    tau_global: Integral,
) -> UniformRemeshHistoryStabilityCertificate: ...

def observe_uniform_remesh_history_transition(
    certificate: UniformRemeshHistoryStabilityCertificate,
    history: Iterable[Iterable[Real]],
    metric_weights: Iterable[Real],
    *,
    nodes: Iterable[Hashable] | None = None,
) -> UniformRemeshHistoryTransitionObservation: ...

__all__: tuple[str, ...]
