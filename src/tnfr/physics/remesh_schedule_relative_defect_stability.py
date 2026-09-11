r"""Uniform exact REMESH/schedule stability under a relative head defect.

Let ``y`` be the ideal REMESH head, ``z`` the bounded head presented to the
schedule, and

    J = sum_d c_d E_H(history[d]).

Assume, for every transition in the declared exact-model family,

    E_H(z) - E_H(y) <= eta J,         eta >= 0,
    E_H(S_k z) <= q E_H(z),           0 <= q <= 1,

where every ``S_k`` preserves spatial consensus and uses the same fixed
positive metric and support as the REMESH theorem.  Jensen gives
``E_H(y) <= J`` and hence

    E_H(S_k z) <= q (1 + eta) J.

Thus the existing policy-envelope theorem applies with the combined head gain
``q_eff = q * (1 + eta)``.  This module delegates all companion matrices,
block powers and prefix bounds to that theorem; it does not duplicate their
linear algebra.

The relative defect bound and schedule hypotheses are caller declarations.
This pure certificate does not inspect runtime records, establish forward
invariance of a runtime family, or certify future executions, binary64
behavior, solver properties, adaptive grammar, or full TNFR dynamics.
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
from ._exact_linear_algebra import ExactSquareMatrix
from .remesh_history_stability import UniformRemeshHistoryStabilityCertificate
from .remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
    certify_uniform_remesh_schedule_policy_stability,
)

__all__ = (
    "UniformRemeshScheduleRelativeDefectStabilityCertificate",
    "certify_uniform_remesh_schedule_relative_defect_stability",
)

_PROOF_VERSION = "uniform_remesh_schedule_relative_defect_stability_v1"
_SCOPE = (
    "Conditional exact-rational spatial-disagreement stability for "
    "post-schedule histories under one fixed uniform-alpha unclipped REMESH "
    "companion. For every transition, the caller must establish in one "
    "fixed positive spatial metric and support that the bounded pre-schedule "
    "head z satisfies E_H(z)-E_H(y)<=eta*J, where y is the ideal REMESH head "
    "and J=sum_d c_d E_H(history[d]), and that every schedule preserves "
    "consensus and has global disagreement-energy gain at most q, including "
    "on z. The theorem proves the exact envelope q_eff=q*(1+eta), prefix "
    "nonexpansion, and a q_eff gain every active_max_delay+1 cycles when "
    "q_eff<=1. These hypotheses are declarations: the certificate does not "
    "verify runtime defects or schedule maps, establish runtime forward "
    "invariance, bind or predict future executions, certify binary64 runtime "
    "rounding or clipping, admit changing support or metric, prove solver "
    "accuracy or order, derive adaptive U2/U4 policy, or establish full "
    "multichannel TNFR stability."
)

_CONDITION_NAMES = (
    "source_policy_certificate_intact",
    "declared_pre_schedule_relative_energy_defect_is_nonnegative_finite",
    "effective_head_gain_is_q_times_one_plus_eta",
    "effective_head_gain_in_unit_interval",
    "effective_head_gain_envelope_certificate_intact",
    "effective_head_gain_envelope_reuses_source_remesh_model",
    "effective_head_gain_envelope_uses_combined_gain",
    "effective_block_gain_and_margin_match_combined_gain",
    "effective_intrablock_prefix_bound_is_one",
)


def _proof_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not UniformRemeshScheduleRelativeDefectStabilityCertificate:
        raise TypeError("proof value must have its canonical result type")
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(
            UniformRemeshScheduleRelativeDefectStabilityCertificate
        )
        if item.name != "_proof_stamp"
    )
    return (_PROOF_VERSION, structural_proof_signature(payload))


def _seal(
    value: UniformRemeshScheduleRelativeDefectStabilityCertificate,
) -> UniformRemeshScheduleRelativeDefectStabilityCertificate:
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


def _strict_conditions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_CONDITION_NAMES)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == _CONDITION_NAMES[index]
            for index, item in enumerate(value)
        )
    )


def _exact_nonnegative_relative_defect(value: Any) -> Fraction:
    argument = "pre_schedule_relative_energy_defect_upper_bound"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(f"{argument} must be a finite real scalar")
    if isinstance(value, Rational):
        try:
            result = Fraction(value)
        except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise TNFRValueError(
                f"{argument} must be a finite real scalar"
            ) from exc
    else:
        try:
            source_nonzero = bool(value != 0)
            floating = float(value)
        except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise TNFRValueError(
                f"{argument} must be a finite real scalar"
            ) from exc
        if not math.isfinite(floating):
            raise TNFRValueError(f"{argument} must be finite")
        if floating == 0.0 and source_nonzero:
            raise TNFRValueError(
                f"{argument} contains a nonzero value below binary64 range"
            )
        result = Fraction.from_float(floating)
    if result < 0:
        raise TNFRValueError(f"{argument} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class _RelativeDefectModel:
    exact_effective_head_energy_gain_upper_bound: Fraction
    effective_head_gain_envelope_certificate: (
        UniformRemeshSchedulePolicyStabilityCertificate
    )
    conditions: tuple[tuple[str, bool], ...]


def _derive_relative_defect_model(
    policy: UniformRemeshSchedulePolicyStabilityCertificate,
    eta: Fraction,
) -> _RelativeDefectModel:
    q = policy.schedule_energy_gain_upper_bound
    q_eff = q * (Fraction(1) + eta)
    if q_eff > 1:
        raise TNFRValueError(
            "effective head energy gain q*(1+eta) must be at most 1"
        )
    envelope = certify_uniform_remesh_schedule_policy_stability(
        policy.remesh_certificate,
        q_eff,
    )
    conditions = (
        (
            "source_policy_certificate_intact",
            policy.policy_stability_certificate_certified,
        ),
        (
            "declared_pre_schedule_relative_energy_defect_is_nonnegative_finite",
            eta >= 0,
        ),
        (
            "effective_head_gain_is_q_times_one_plus_eta",
            q_eff == q * (Fraction(1) + eta),
        ),
        (
            "effective_head_gain_in_unit_interval",
            Fraction(0) <= q_eff <= Fraction(1),
        ),
        (
            "effective_head_gain_envelope_certificate_intact",
            envelope.policy_stability_certificate_certified,
        ),
        (
            "effective_head_gain_envelope_reuses_source_remesh_model",
            _same(envelope.remesh_certificate, policy.remesh_certificate),
        ),
        (
            "effective_head_gain_envelope_uses_combined_gain",
            envelope.schedule_energy_gain_upper_bound == q_eff,
        ),
        (
            "effective_block_gain_and_margin_match_combined_gain",
            envelope.exact_uniform_block_energy_gain_upper_bound == q_eff
            and envelope.exact_uniform_normalized_block_margin_lower_bound
            == Fraction(1) - q_eff,
        ),
        (
            "effective_intrablock_prefix_bound_is_one",
            envelope.exact_intrablock_prefix_energy_gain_upper_bound
            == Fraction(1),
        ),
    )
    return _RelativeDefectModel(
        exact_effective_head_energy_gain_upper_bound=q_eff,
        effective_head_gain_envelope_certificate=envelope,
        conditions=conditions,
    )


@dataclass(frozen=True, slots=True)
class UniformRemeshScheduleRelativeDefectStabilityCertificate:
    """Sealed conditional exact theorem for relative pre-schedule defects."""

    policy_certificate: UniformRemeshSchedulePolicyStabilityCertificate = field(
        repr=False
    )
    pre_schedule_relative_energy_defect_upper_bound: Fraction
    exact_effective_head_energy_gain_upper_bound: Fraction
    effective_head_gain_envelope_certificate: (
        UniformRemeshSchedulePolicyStabilityCertificate
    ) = field(repr=False)
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            if not _sealed(self):
                return False
            policy = object.__getattribute__(self, "policy_certificate")
            eta = object.__getattribute__(
                self,
                "pre_schedule_relative_energy_defect_upper_bound",
            )
            q_eff = object.__getattribute__(
                self,
                "exact_effective_head_energy_gain_upper_bound",
            )
            envelope = object.__getattribute__(
                self,
                "effective_head_gain_envelope_certificate",
            )
            if (
                type(policy) is not UniformRemeshSchedulePolicyStabilityCertificate
                or type(eta) is not Fraction
                or type(q_eff) is not Fraction
                or type(envelope)
                is not UniformRemeshSchedulePolicyStabilityCertificate
                or eta < 0
                or not Fraction(0) <= q_eff <= Fraction(1)
                or not policy.policy_stability_certificate_certified
                or not envelope.policy_stability_certificate_certified
                or not _strict_conditions(self.conditions)
            ):
                return False
            expected = _derive_relative_defect_model(policy, eta)
            return bool(
                q_eff
                == expected.exact_effective_head_energy_gain_upper_bound
                and proof_stamps_are_identical(
                    object.__getattribute__(envelope, "_proof_stamp"),
                    object.__getattribute__(
                        expected.effective_head_gain_envelope_certificate,
                        "_proof_stamp",
                    ),
                )
                and self.conditions == expected.conditions
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def relative_defect_stability_certificate_certified(self) -> bool:
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
            return ("remesh_schedule_relative_defect_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def remesh_certificate(self) -> UniformRemeshHistoryStabilityCertificate:
        return self.policy_certificate.remesh_certificate

    @property
    def schedule_energy_gain_upper_bound(self) -> Fraction:
        """Return the declared schedule-only gain ``q``."""

        return self.policy_certificate.schedule_energy_gain_upper_bound

    @property
    def effective_head_energy_gain_upper_bound(self) -> Fraction:
        """Return the combined REMESH-defect/schedule gain ``q_eff``."""

        return self.exact_effective_head_energy_gain_upper_bound

    @property
    def history_length(self) -> int:
        return self.effective_head_gain_envelope_certificate.history_length

    @property
    def universal_block_horizon(self) -> int:
        return (
            self.effective_head_gain_envelope_certificate.universal_block_horizon
        )

    @property
    def remesh_companion_matrix(self) -> ExactSquareMatrix:
        return self.effective_head_gain_envelope_certificate.remesh_companion_matrix

    @property
    def effective_head_energy_domination_matrix(self) -> ExactSquareMatrix:
        """Return ``diag(q_eff, 1, ..., 1) P`` as an energy envelope."""

        return (
            self.effective_head_gain_envelope_certificate
            .schedule_energy_domination_matrix
        )

    @property
    def remesh_block_power(self) -> ExactSquareMatrix:
        return self.effective_head_gain_envelope_certificate.remesh_block_power

    @property
    def effective_head_block_domination_power(self) -> ExactSquareMatrix:
        return (
            self.effective_head_gain_envelope_certificate
            .schedule_block_domination_power
        )

    @property
    def exact_uniform_normalized_block_margin_lower_bound(self) -> Fraction:
        return (
            self.effective_head_gain_envelope_certificate
            .exact_uniform_normalized_block_margin_lower_bound
        )

    @property
    def exact_uniform_block_energy_gain_upper_bound(self) -> Fraction:
        return (
            self.effective_head_gain_envelope_certificate
            .exact_uniform_block_energy_gain_upper_bound
        )

    @property
    def exact_intrablock_prefix_energy_gain_upper_bound(self) -> Fraction:
        return (
            self.effective_head_gain_envelope_certificate
            .exact_intrablock_prefix_energy_gain_upper_bound
        )

    @property
    def conditional_exact_model_spatial_disagreement_nonincrease_certified(
        self,
    ) -> bool:
        return self.relative_defect_stability_certificate_certified

    @property
    def uniform_intrablock_prefix_bound_certified(self) -> bool:
        return self.relative_defect_stability_certificate_certified

    @property
    def uniform_positive_normalized_block_margin_certified(self) -> bool:
        return bool(
            self.relative_defect_stability_certificate_certified
            and self.exact_effective_head_energy_gain_upper_bound < 1
        )

    @property
    def repeated_exact_model_spatial_disagreement_stability_certified(
        self,
    ) -> bool:
        return self.relative_defect_stability_certificate_certified

    @property
    def geometric_spatial_disagreement_convergence_certified(self) -> bool:
        return self.uniform_positive_normalized_block_margin_certified

    @property
    def q_zero_preschedule_defect_absorption_certified(self) -> bool:
        return bool(
            self.relative_defect_stability_certificate_certified
            and self.schedule_energy_gain_upper_bound == 0
            and self.exact_effective_head_energy_gain_upper_bound == 0
        )

    @property
    def effective_gain_one_zero_margin_boundary_certified(self) -> bool:
        return bool(
            self.relative_defect_stability_certificate_certified
            and self.exact_effective_head_energy_gain_upper_bound == 1
            and self.exact_uniform_normalized_block_margin_lower_bound == 0
        )

    def exact_cycle_energy_gain_upper_bound(self, cycle_count: int) -> Fraction:
        """Return ``q_eff**floor(cycle_count/L)`` for the exact envelope."""

        if not self.relative_defect_stability_certificate_certified:
            raise TNFRValueError(
                "REMESH/schedule relative-defect certificate is unsealed or "
                "inconsistent"
            )
        return (
            self.effective_head_gain_envelope_certificate
            .exact_cycle_energy_gain_upper_bound(cycle_count)
        )

    @property
    def runtime_relative_defect_bound_verified(self) -> bool:
        return False

    @property
    def runtime_schedule_maps_verified(self) -> bool:
        return False

    @property
    def runtime_forward_invariance_certified(self) -> bool:
        return False

    @property
    def future_runtime_stability_certified(self) -> bool:
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


def certify_uniform_remesh_schedule_relative_defect_stability(
    policy_certificate: UniformRemeshSchedulePolicyStabilityCertificate,
    pre_schedule_relative_energy_defect_upper_bound: Real,
) -> UniformRemeshScheduleRelativeDefectStabilityCertificate:
    """Certify the conditional exact theorem with ``q_eff=q*(1+eta)``.

    The caller is responsible for establishing the relative pre-schedule
    energy-defect bound at every transition and for ensuring that the source
    policy's schedule-only gain applies to each bounded head in the declared
    fixed-support, fixed-metric exact-model family.
    """

    if type(policy_certificate) is not UniformRemeshSchedulePolicyStabilityCertificate:
        raise TNFRValueError(
            "policy_certificate must be an exact REMESH/schedule policy "
            "certificate"
        )
    if not policy_certificate.policy_stability_certificate_certified:
        raise TNFRValueError(
            "policy_certificate is unsealed, tampered, or inconsistent"
        )
    eta = _exact_nonnegative_relative_defect(
        pre_schedule_relative_energy_defect_upper_bound
    )
    model = _derive_relative_defect_model(policy_certificate, eta)
    if not all(passed for _, passed in model.conditions):
        failed = tuple(name for name, passed in model.conditions if not passed)
        raise TNFRValueError(
            f"REMESH/schedule relative-defect theorem failed: {failed}"
        )
    value = UniformRemeshScheduleRelativeDefectStabilityCertificate(
        policy_certificate=policy_certificate,
        pre_schedule_relative_energy_defect_upper_bound=eta,
        exact_effective_head_energy_gain_upper_bound=(
            model.exact_effective_head_energy_gain_upper_bound
        ),
        effective_head_gain_envelope_certificate=(
            model.effective_head_gain_envelope_certificate
        ),
        conditions=model.conditions,
    )
    sealed = _seal(value)
    if not sealed.relative_defect_stability_certificate_certified:
        raise TNFRValueError(
            "REMESH/schedule relative-defect certificate sealing failed"
        )
    return sealed
