r"""Uniform exact spatial-disagreement stability for REMESH and schedules.

Let ``P`` be the exact row-stochastic companion from
``remesh_history_stability`` and let every schedule in a possibly varying
sequence preserve spatial consensus and satisfy, in one fixed positive metric,

    E_H(S_k x) <= q E_H(x),          0 <= q <= 1.

Histories are indexed immediately after each schedule.  One abstract cycle
therefore mixes the history through REMESH and then applies the next schedule
to the new head.  The componentwise energy envelope is

    e[k + 1] <= B_q e[k],            B_q = diag(q, 1, ..., 1) P.

For ``L = active_max_delay + 1``, every length-``L`` companion path visits the
head.  Exact nonnegative-matrix domination then gives

    B_q**s <= P**s                    for every s >= 0,
    B_q**L <= q P**L,

and stationarity of ``pi`` gives prefix nonexpansion and the repeated bound

    V[k + n] <= q**floor(n/L) V[k].

This is a conditional exact-model theorem.  It does not inspect schedules,
identify a binary64 execution, or control rounding, clipping, changing support,
changing metrics, solver error, adaptive grammar, or full TNFR dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Rational, Real
from typing import Any

from ..errors import TNFRValueError
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._exact_linear_algebra import (
    ExactSquareMatrix,
    exact_square_matrix_power,
)
from .remesh_history_stability import UniformRemeshHistoryStabilityCertificate

__all__ = (
    "UniformRemeshSchedulePolicyStabilityCertificate",
    "certify_uniform_remesh_schedule_policy_stability",
)

ExactVector = tuple[Fraction, ...]

_PROOF_VERSION = "uniform_remesh_schedule_policy_stability_v1"
_SCOPE = (
    "Conditional exact-rational stability for post-schedule histories under "
    "one fixed uniform-alpha unclipped REMESH companion and any sequence of "
    "consensus-preserving exact schedule maps whose disagreement-energy gain "
    "in one fixed positive spatial metric is at most the supplied common q. "
    "The theorem proves prefix nonexpansion and a uniform q gain every "
    "active_max_delay + 1 cycles. It does not verify any schedule map, bind a "
    "binary64 runtime, control rounding or clipping defects, admit changing "
    "support or metric, prove solver accuracy or order, derive adaptive U2/U4 "
    "policy, or establish full multichannel TNFR stability."
)

_CONDITION_NAMES = (
    "source_remesh_certificate_intact",
    "history_dimension_matches_active_delay",
    "common_schedule_gain_in_unit_interval",
    "remesh_companion_nonnegative_row_stochastic",
    "stationary_distribution_positive_normalized",
    "stationary_distribution_left_invariant",
    "schedule_energy_envelope_is_Dq_times_P",
    "head_avoidance_matrix_nilpotent_over_universal_horizon",
    "one_step_envelope_is_entrywise_dominated_by_remesh",
    "universal_block_is_entrywise_dominated_by_q_times_remesh",
    "one_step_stationary_energy_nonexpansive",
    "universal_block_stationary_energy_gain_at_most_q",
    "uniform_margin_and_gain_are_complements",
)


def _proof_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not UniformRemeshSchedulePolicyStabilityCertificate:
        raise TypeError("proof value must have its canonical result type")
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(UniformRemeshSchedulePolicyStabilityCertificate)
        if item.name != "_proof_stamp"
    )
    return (_PROOF_VERSION, structural_proof_signature(payload))


def _seal(
    value: UniformRemeshSchedulePolicyStabilityCertificate,
) -> UniformRemeshSchedulePolicyStabilityCertificate:
    return replace(value, _proof_stamp=_proof_stamp(value))


def _sealed(value: Any) -> bool:
    try:
        return proof_stamps_are_identical(
            object.__getattribute__(value, "_proof_stamp"),
            _proof_stamp(value),
        )
    except BaseException:
        return False


def _same(left: Any, right: Any) -> bool:
    try:
        return proof_stamps_are_identical(
            structural_proof_signature(left),
            structural_proof_signature(right),
        )
    except BaseException:
        return False


def _exact_unit_gain(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(
            "schedule_energy_gain_upper_bound must be a finite real scalar"
        )
    if isinstance(value, Rational):
        result = Fraction(value)
    else:
        try:
            source_nonzero = bool(value != 0)
            floating = float(value)
        except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise TNFRValueError(
                "schedule_energy_gain_upper_bound must be a finite real scalar"
            ) from exc
        if not math.isfinite(floating):
            raise TNFRValueError(
                "schedule_energy_gain_upper_bound must be finite"
            )
        if floating == 0.0 and source_nonzero:
            raise TNFRValueError(
                "schedule_energy_gain_upper_bound contains a nonzero value "
                "below binary64 range"
            )
        result = Fraction.from_float(floating)
    if not Fraction(0) <= result <= Fraction(1):
        raise TNFRValueError(
            "schedule_energy_gain_upper_bound must be in [0, 1]"
        )
    return result


def _strict_exact_matrix(value: Any, dimension: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == dimension
        and all(
            type(row) is tuple
            and len(row) == dimension
            and all(type(entry) is Fraction for entry in row)
            for row in value
        )
    )


def _left_action(
    vector: ExactVector,
    matrix: ExactSquareMatrix,
) -> ExactVector:
    return tuple(
        sum(
            (
                vector[row] * matrix[row][column]
                for row in range(len(matrix))
            ),
            Fraction(0),
        )
        for column in range(len(matrix))
    )


def _entrywise_at_most(
    left: ExactSquareMatrix,
    right: ExactSquareMatrix,
) -> bool:
    return all(
        left_value <= right_value
        for left_row, right_row in zip(left, right, strict=True)
        for left_value, right_value in zip(left_row, right_row, strict=True)
    )


def _scaled_matrix(
    scalar: Fraction,
    matrix: ExactSquareMatrix,
) -> ExactSquareMatrix:
    return tuple(tuple(scalar * entry for entry in row) for row in matrix)


def _head_scaled_matrix(
    scalar: Fraction,
    matrix: ExactSquareMatrix,
) -> ExactSquareMatrix:
    return tuple(
        tuple((scalar if row == 0 else Fraction(1)) * entry for entry in values)
        for row, values in enumerate(matrix)
    )


def _zero_matrix(dimension: int) -> ExactSquareMatrix:
    return tuple(
        tuple(Fraction(0) for _column in range(dimension))
        for _row in range(dimension)
    )


@dataclass(frozen=True, slots=True)
class _PolicyModel:
    history_length: int
    universal_block_horizon: int
    remesh_companion_matrix: ExactSquareMatrix
    schedule_energy_domination_matrix: ExactSquareMatrix
    head_avoidance_matrix: ExactSquareMatrix
    remesh_block_power: ExactSquareMatrix
    schedule_block_domination_power: ExactSquareMatrix
    head_avoidance_block_power: ExactSquareMatrix
    exact_uniform_normalized_block_margin_lower_bound: Fraction
    exact_uniform_block_energy_gain_upper_bound: Fraction
    exact_intrablock_prefix_energy_gain_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]


def _derive_policy_model(
    source: UniformRemeshHistoryStabilityCertificate,
    q: Fraction,
) -> _PolicyModel:
    companion = source.companion_matrix
    stationary = source.stationary_distribution
    dimension = len(companion)
    horizon = source.active_max_delay + 1
    domination = _head_scaled_matrix(q, companion)
    head_avoidance = _head_scaled_matrix(Fraction(0), companion)
    remesh_block = exact_square_matrix_power(companion, horizon)
    domination_block = exact_square_matrix_power(domination, horizon)
    head_avoidance_block = exact_square_matrix_power(head_avoidance, horizon)
    q_remesh_block = _scaled_matrix(q, remesh_block)
    one_step_entrywise = _entrywise_at_most(domination, companion)
    one_step_stationary = all(
        observed <= reference
        for observed, reference in zip(
            _left_action(stationary, domination),
            stationary,
            strict=True,
        )
    )
    block_stationary = all(
        observed <= q * reference
        for observed, reference in zip(
            _left_action(stationary, domination_block),
            stationary,
            strict=True,
        )
    )
    row_stochastic = all(
        all(entry >= 0 for entry in row)
        and sum(row, Fraction(0)) == 1
        for row in companion
    )
    stationary_invariant = _left_action(stationary, companion) == stationary
    margin = Fraction(1) - q
    conditions = (
        ("source_remesh_certificate_intact", source.stability_certificate_certified),
        (
            "history_dimension_matches_active_delay",
            dimension == horizon and dimension > 0,
        ),
        ("common_schedule_gain_in_unit_interval", Fraction(0) <= q <= Fraction(1)),
        (
            "remesh_companion_nonnegative_row_stochastic",
            row_stochastic,
        ),
        (
            "stationary_distribution_positive_normalized",
            len(stationary) == dimension
            and all(entry > 0 for entry in stationary)
            and sum(stationary, Fraction(0)) == 1,
        ),
        (
            "stationary_distribution_left_invariant",
            stationary_invariant,
        ),
        (
            "schedule_energy_envelope_is_Dq_times_P",
            domination == _head_scaled_matrix(q, companion),
        ),
        (
            "head_avoidance_matrix_nilpotent_over_universal_horizon",
            head_avoidance_block == _zero_matrix(dimension),
        ),
        (
            "one_step_envelope_is_entrywise_dominated_by_remesh",
            one_step_entrywise,
        ),
        (
            "universal_block_is_entrywise_dominated_by_q_times_remesh",
            _entrywise_at_most(domination_block, q_remesh_block),
        ),
        (
            "one_step_stationary_energy_nonexpansive",
            one_step_stationary,
        ),
        (
            "universal_block_stationary_energy_gain_at_most_q",
            block_stationary,
        ),
        (
            "uniform_margin_and_gain_are_complements",
            margin >= 0 and margin + q == 1,
        ),
    )
    return _PolicyModel(
        history_length=dimension,
        universal_block_horizon=horizon,
        remesh_companion_matrix=companion,
        schedule_energy_domination_matrix=domination,
        head_avoidance_matrix=head_avoidance,
        remesh_block_power=remesh_block,
        schedule_block_domination_power=domination_block,
        head_avoidance_block_power=head_avoidance_block,
        exact_uniform_normalized_block_margin_lower_bound=margin,
        exact_uniform_block_energy_gain_upper_bound=q,
        exact_intrablock_prefix_energy_gain_upper_bound=Fraction(1),
        conditions=conditions,
    )


@dataclass(frozen=True, slots=True)
class UniformRemeshSchedulePolicyStabilityCertificate:
    """Sealed conditional theorem for a uniformly contractive schedule family."""

    remesh_certificate: UniformRemeshHistoryStabilityCertificate = field(
        repr=False
    )
    schedule_energy_gain_upper_bound: Fraction
    history_length: int
    universal_block_horizon: int
    remesh_companion_matrix: ExactSquareMatrix
    schedule_energy_domination_matrix: ExactSquareMatrix
    head_avoidance_matrix: ExactSquareMatrix
    remesh_block_power: ExactSquareMatrix
    schedule_block_domination_power: ExactSquareMatrix
    head_avoidance_block_power: ExactSquareMatrix
    exact_uniform_normalized_block_margin_lower_bound: Fraction
    exact_uniform_block_energy_gain_upper_bound: Fraction
    exact_intrablock_prefix_energy_gain_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            if not _sealed(self):
                return False
            source = object.__getattribute__(self, "remesh_certificate")
            q = object.__getattribute__(self, "schedule_energy_gain_upper_bound")
            if (
                type(source) is not UniformRemeshHistoryStabilityCertificate
                or type(q) is not Fraction
                or not source.stability_certificate_certified
                or not Fraction(0) <= q <= Fraction(1)
            ):
                return False
            dimension = source.active_max_delay + 1
            if (
                type(self.history_length) is not int
                or type(self.universal_block_horizon) is not int
                or not _strict_exact_matrix(
                    self.remesh_companion_matrix,
                    dimension,
                )
                or not _strict_exact_matrix(
                    self.schedule_energy_domination_matrix,
                    dimension,
                )
                or not _strict_exact_matrix(self.head_avoidance_matrix, dimension)
                or not _strict_exact_matrix(
                    self.remesh_block_power,
                    dimension,
                )
                or not _strict_exact_matrix(
                    self.schedule_block_domination_power,
                    dimension,
                )
                or not _strict_exact_matrix(
                    self.head_avoidance_block_power,
                    dimension,
                )
            ):
                return False
            expected = _derive_policy_model(source, q)
            observed_payload = tuple(
                (item.name, object.__getattribute__(self, item.name))
                for item in fields(_PolicyModel)
            )
            expected_payload = tuple(
                (item.name, object.__getattribute__(expected, item.name))
                for item in fields(_PolicyModel)
            )
            return _same(observed_payload, expected_payload)
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def policy_stability_certificate_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and type(self.conditions) is tuple
            and tuple(name for name, _passed in self.conditions)
            == _CONDITION_NAMES
            and all(type(passed) is bool and passed for _, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("remesh_schedule_policy_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def conditional_policy_family_spatial_disagreement_nonincrease_certified(
        self,
    ) -> bool:
        return self.policy_stability_certificate_certified

    @property
    def uniform_intrablock_prefix_bound_certified(self) -> bool:
        return self.policy_stability_certificate_certified

    @property
    def uniform_positive_normalized_block_margin_certified(self) -> bool:
        return bool(
            self.policy_stability_certificate_certified
            and self.schedule_energy_gain_upper_bound < 1
        )

    @property
    def repeated_exact_model_spatial_disagreement_stability_certified(
        self,
    ) -> bool:
        return self.policy_stability_certificate_certified

    @property
    def geometric_spatial_disagreement_convergence_certified(self) -> bool:
        return self.uniform_positive_normalized_block_margin_certified

    @property
    def alpha_one_spatial_disagreement_decay_certified(self) -> bool:
        return bool(
            self.uniform_positive_normalized_block_margin_certified
            and self.remesh_certificate.alpha == 1
        )

    @property
    def q_one_zero_margin_boundary_certified(self) -> bool:
        return bool(
            self.policy_stability_certificate_certified
            and self.schedule_energy_gain_upper_bound == 1
            and self.exact_uniform_normalized_block_margin_lower_bound == 0
        )

    def exact_cycle_energy_gain_upper_bound(self, cycle_count: int) -> Fraction:
        """Return ``q**floor(cycle_count/L)`` for the exact model family."""

        if type(cycle_count) is not int or cycle_count < 0:
            raise TNFRValueError("cycle_count must be a nonnegative integer")
        if not self.policy_stability_certificate_certified:
            raise TNFRValueError(
                "REMESH/schedule policy certificate is unsealed or inconsistent"
            )
        blocks = cycle_count // self.universal_block_horizon
        return self.schedule_energy_gain_upper_bound**blocks

    @property
    def runtime_schedule_maps_verified(self) -> bool:
        return False

    @property
    def binary64_runtime_stability_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def adaptive_grammar_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def certify_uniform_remesh_schedule_policy_stability(
    remesh_certificate: UniformRemeshHistoryStabilityCertificate,
    schedule_energy_gain_upper_bound: Real,
) -> UniformRemeshSchedulePolicyStabilityCertificate:
    """Certify the conditional uniform exact REMESH/schedule energy theorem.

    The caller is responsible for proving that every schedule in the intended
    family preserves consensus and has disagreement-energy gain no greater than
    ``schedule_energy_gain_upper_bound`` in one common fixed positive metric.
    """

    if type(remesh_certificate) is not UniformRemeshHistoryStabilityCertificate:
        raise TNFRValueError(
            "remesh_certificate must be an exact uniform REMESH certificate"
        )
    if not remesh_certificate.stability_certificate_certified:
        raise TNFRValueError(
            "remesh_certificate is unsealed, tampered, or inconsistent"
        )
    q = _exact_unit_gain(schedule_energy_gain_upper_bound)
    model = _derive_policy_model(remesh_certificate, q)
    if not all(passed for _, passed in model.conditions):
        failed = tuple(name for name, passed in model.conditions if not passed)
        raise TNFRValueError(f"REMESH/schedule policy theorem failed: {failed}")
    value = UniformRemeshSchedulePolicyStabilityCertificate(
        remesh_certificate=remesh_certificate,
        schedule_energy_gain_upper_bound=q,
        history_length=model.history_length,
        universal_block_horizon=model.universal_block_horizon,
        remesh_companion_matrix=model.remesh_companion_matrix,
        schedule_energy_domination_matrix=(
            model.schedule_energy_domination_matrix
        ),
        head_avoidance_matrix=model.head_avoidance_matrix,
        remesh_block_power=model.remesh_block_power,
        schedule_block_domination_power=model.schedule_block_domination_power,
        head_avoidance_block_power=model.head_avoidance_block_power,
        exact_uniform_normalized_block_margin_lower_bound=(
            model.exact_uniform_normalized_block_margin_lower_bound
        ),
        exact_uniform_block_energy_gain_upper_bound=(
            model.exact_uniform_block_energy_gain_upper_bound
        ),
        exact_intrablock_prefix_energy_gain_upper_bound=(
            model.exact_intrablock_prefix_energy_gain_upper_bound
        ),
        conditions=model.conditions,
    )
    return _seal(value)
