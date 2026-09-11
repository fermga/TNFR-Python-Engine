r"""Global binary64 P2 half-Reception EPI-kernel stability.

Fix an abstract two-node support with declared mutual singleton neighbour sets
and evaluate one all-target Reception EPI proposal from an immutable snapshot
with the exact binary64 factor ``0.5``.  For every finite binary64 pair
``(x, y)``, the two unclipped proposals are

    fl(fl(0.5*x) + fl(0.5*y)),
    fl(fl(0.5*y) + fl(0.5*x)).

The singleton ``fmean`` calls reproduce the opposite numeric operand and
binary64 addition is commutative.  Both proposals are therefore numerically
equal.  Applying one common hard interval projection keeps them equal and in
the declared interval.  Consequently the post-kernel centered energy is zero
in every positive diagonal metric: the global schedule-only gain is ``q=0``.

The source ``alpha=1`` REMESH class supplies a uniform binary64 relative
defect ``eta=0`` and exact global-delay class closure.  This module composes
those proved facts with the existing common-q and relative-defect theorems.
For ``L=tau_global+1``, the active post-schedule companion history therefore
has exactly zero spatial-disagreement energy after every ``n >= L`` cycles.

The certificate concerns the restricted numeric EPI kernels only.  It does
not certify the complete Reception stage, grammar admission, graph callbacks,
history/event ownership, concurrent execution, signed-zero bit preservation,
an exact affine realization of the binary64 average, solver properties, or
full TNFR stability.  In particular, underflow can make the runtime consensus
value differ from the exact-real half average while leaving ``q=0`` intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any, Hashable, Literal

from ..dynamics.structural_clip import structural_clip
from ..errors import TNFRValueError
from ..operators._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
)
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .binary64_remesh_relative_defect import (
    UniformAlphaOneHardClipRemeshClassCertificate,
)
from .remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
    certify_uniform_remesh_schedule_policy_stability,
)
from .remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
    certify_uniform_remesh_schedule_relative_defect_stability,
)

__all__ = (
    "P2HalfReceptionRemeshStabilityCertificate",
    "certify_p2_half_reception_remesh_stability",
)

Binary64Pair = tuple[float, float]
ExactMatrix2 = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]

_PROOF_VERSION = "binary64_p2_half_reception_remesh_stability_v1"
_SCOPE = (
    "One fixed abstract two-node mutual-singleton EPI-kernel family: every "
    "cycle declares the shared Reception singleton mean and blend from one "
    "immutable all-target snapshot, exact binary64 mix 0.5, and one common "
    "hard clamp. The support is part of the abstract kernel declaration and "
    "is not validated against a graph. "
    "For every finite represented pair in the source interval, the bounded "
    "outputs are numerically equal, so the global centered-energy gain is "
    "q=0 in the source class's fixed positive metric. The alpha=1 source "
    "REMESH class has eta=0 and is forward invariant; composition through "
    "the existing common-q theorem gives zero active post-schedule companion "
    "energy after tau_global+1 cycles. This proves arbitrary finite "
    "repetition of the restricted binary64 EPI kernels. It does not certify "
    "the complete EN stage, grammar or event execution, callbacks, graph "
    "transactions, signed-zero bits, global binary64 affinity, solver "
    "properties, or full TNFR stability."
)

_CONDITION_NAMES = (
    "source_alpha_one_hard_clip_class_is_intact",
    "ordered_support_is_exactly_two_nodes",
    "source_metric_is_fixed_positive_and_normalized",
    "abstract_p2_support_declares_mutual_singleton_neighbor_sets",
    "abstract_reception_epi_kernel_declares_one_immutable_all_target_snapshot",
    "binary64_mix_factor_is_exactly_one_half",
    "singleton_means_reproduce_opposite_numeric_operands",
    "reversed_half_blends_are_numerically_equal_for_all_finite_pairs",
    "common_hard_clamp_preserves_numeric_consensus_and_interval",
    "restricted_epi_kernel_preserves_support_metric_and_remesh_configuration",
    "global_centered_energy_gain_is_zero",
    "source_binary64_remesh_relative_defect_is_zero",
    "common_q_policy_reuses_source_remesh_with_q_zero",
    "relative_defect_composition_has_q_eff_zero",
    "active_history_extinction_horizon_is_tau_global_plus_one",
)


def _proof_stamp(value: Any) -> tuple[Any, ...]:
    if type(value) is not P2HalfReceptionRemeshStabilityCertificate:
        raise TypeError("proof value must have its canonical result type")
    opaque = (
        object.__getattribute__(value, "remesh_class_certificate"),
        object.__getattribute__(value, "policy_certificate"),
        object.__getattribute__(value, "relative_defect_certificate"),
    )
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(P2HalfReceptionRemeshStabilityCertificate)
        if item.name != "_proof_stamp"
    )
    return (
        _PROOF_VERSION,
        structural_proof_signature(payload, opaque_references=opaque),
    )


def _seal(
    value: P2HalfReceptionRemeshStabilityCertificate,
) -> P2HalfReceptionRemeshStabilityCertificate:
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


def _evaluate_binary64_half_reception_pair(
    values: tuple[float, float],
    *,
    lower: float,
    upper: float,
) -> Binary64Pair:
    """Replay the production half-Reception kernel on one validated pair.

    This private kernel is shared by the abstract family certificate and the
    finite executor adapter.  It validates only represented values and the
    interval; callers remain responsible for validating their proof objects.
    """

    if (
        type(values) is not tuple
        or len(values) != 2
        or type(lower) is not float
        or type(upper) is not float
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower > upper
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < lower
            or value > upper
            for value in values
        )
    ):
        raise TNFRValueError(
            "pair values must be finite binary64 coordinates in the "
            "certified interval"
        )
    left, right = values
    left_mean = neighbor_epi_unweighted_mean((right,))
    right_mean = neighbor_epi_unweighted_mean((left,))
    left_raw = neighbor_epi_blend_value(left, left_mean, 0.5)
    right_raw = neighbor_epi_blend_value(right, right_mean, 0.5)
    left_bounded = float(
        structural_clip(
            left_raw,
            lo=lower,
            hi=upper,
            mode="hard",
            record_stats=False,
        )
    )
    right_bounded = float(
        structural_clip(
            right_raw,
            lo=lower,
            hi=upper,
            mode="hard",
            record_stats=False,
        )
    )
    if left_bounded != right_bounded:
        raise RuntimeError(
            "production P2 half-Reception kernel violated numeric consensus"
        )
    if not (
        math.isfinite(left_bounded)
        and lower <= left_bounded <= upper
        and lower <= right_bounded <= upper
    ):
        raise RuntimeError(
            "production P2 half-Reception kernel violated its interval"
        )
    return left_bounded, right_bounded


@dataclass(frozen=True, slots=True)
class _P2HalfReceptionModel:
    node_order: tuple[Hashable, Hashable]
    exact_normalized_metric: tuple[Fraction, Fraction]
    mutual_singleton_neighbor_indices: tuple[tuple[int], tuple[int]]
    operator_name: Literal["Reception"]
    operator_glyph: Literal["EN"]
    stage_schedule: Literal["two_phase_jacobi"]
    binary64_mix_factor: float
    exact_mix_factor: Fraction
    exact_ideal_consensus_projector: ExactMatrix2
    exact_schedule_energy_gain_upper_bound: Fraction
    exact_pre_schedule_relative_energy_defect_upper_bound: Fraction
    exact_effective_head_energy_gain_upper_bound: Fraction
    active_history_extinction_horizon: int
    policy_certificate: UniformRemeshSchedulePolicyStabilityCertificate
    relative_defect_certificate: (
        UniformRemeshScheduleRelativeDefectStabilityCertificate
    )
    conditions: tuple[tuple[str, bool], ...]


def _model_from_validated_dependencies(
    source: UniformAlphaOneHardClipRemeshClassCertificate,
    policy: UniformRemeshSchedulePolicyStabilityCertificate,
    relative: UniformRemeshScheduleRelativeDefectStabilityCertificate,
) -> _P2HalfReceptionModel:
    nodes = source.node_order
    metric = source.exact_normalized_metric
    if len(nodes) != 2 or len(metric) != 2:
        raise TNFRValueError(
            "P2 half-Reception stability requires exactly two ordered nodes"
        )
    typed_nodes = (nodes[0], nodes[1])
    typed_metric = (metric[0], metric[1])
    neighbors = ((1,), (0,))
    projector = (
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
    )
    horizon = source.remesh_certificate.active_max_delay + 1
    conditions = (
        (
            "source_alpha_one_hard_clip_class_is_intact",
            source.alpha_one_hard_clip_class_certificate_certified,
        ),
        ("ordered_support_is_exactly_two_nodes", len(nodes) == 2),
        (
            "source_metric_is_fixed_positive_and_normalized",
            len(metric) == 2
            and all(type(weight) is Fraction and weight > 0 for weight in metric)
            and sum(metric, Fraction(0)) == 1,
        ),
        (
            "abstract_p2_support_declares_mutual_singleton_neighbor_sets",
            neighbors == ((1,), (0,)),
        ),
        (
            "abstract_reception_epi_kernel_declares_one_immutable_all_target_snapshot",
            True,
        ),
        (
            "binary64_mix_factor_is_exactly_one_half",
            Fraction.from_float(0.5) == Fraction(1, 2),
        ),
        (
            "singleton_means_reproduce_opposite_numeric_operands",
            True,
        ),
        (
            "reversed_half_blends_are_numerically_equal_for_all_finite_pairs",
            True,
        ),
        (
            "common_hard_clamp_preserves_numeric_consensus_and_interval",
            source.clip_mode == "hard"
            and source.epi_min <= source.epi_max,
        ),
        (
            "restricted_epi_kernel_preserves_support_metric_and_remesh_configuration",
            True,
        ),
        ("global_centered_energy_gain_is_zero", True),
        (
            "source_binary64_remesh_relative_defect_is_zero",
            source.binary64_global_delay_numeric_copy_certified
            and source.exact_uniform_relative_defect_upper_bound == 0,
        ),
        (
            "common_q_policy_reuses_source_remesh_with_q_zero",
            policy.policy_stability_certificate_certified
            and policy.schedule_energy_gain_upper_bound == 0
            and proof_stamps_are_identical(
                object.__getattribute__(
                    policy.remesh_certificate,
                    "_proof_stamp",
                ),
                object.__getattribute__(
                    source.remesh_certificate,
                    "_proof_stamp",
                ),
            ),
        ),
        (
            "relative_defect_composition_has_q_eff_zero",
            relative.relative_defect_stability_certificate_certified
            and relative.pre_schedule_relative_energy_defect_upper_bound == 0
            and relative.exact_effective_head_energy_gain_upper_bound == 0,
        ),
        (
            "active_history_extinction_horizon_is_tau_global_plus_one",
            horizon == source.tau_global + 1
            and horizon == relative.universal_block_horizon,
        ),
    )
    return _P2HalfReceptionModel(
        node_order=typed_nodes,
        exact_normalized_metric=typed_metric,
        mutual_singleton_neighbor_indices=neighbors,
        operator_name="Reception",
        operator_glyph="EN",
        stage_schedule="two_phase_jacobi",
        binary64_mix_factor=0.5,
        exact_mix_factor=Fraction(1, 2),
        exact_ideal_consensus_projector=projector,
        exact_schedule_energy_gain_upper_bound=Fraction(0),
        exact_pre_schedule_relative_energy_defect_upper_bound=Fraction(0),
        exact_effective_head_energy_gain_upper_bound=Fraction(0),
        active_history_extinction_horizon=horizon,
        policy_certificate=policy,
        relative_defect_certificate=relative,
        conditions=conditions,
    )


def _derive_model(
    source: UniformAlphaOneHardClipRemeshClassCertificate,
) -> _P2HalfReceptionModel:
    policy = certify_uniform_remesh_schedule_policy_stability(
        source.remesh_certificate,
        Fraction(0),
    )
    relative = certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        source.exact_uniform_relative_defect_upper_bound,
    )
    return _model_from_validated_dependencies(source, policy, relative)


@dataclass(frozen=True, slots=True)
class P2HalfReceptionRemeshStabilityCertificate:
    """Sealed global certificate for the restricted P2 binary64 kernels."""

    remesh_class_certificate: UniformAlphaOneHardClipRemeshClassCertificate = field(
        repr=False
    )
    node_order: tuple[Hashable, Hashable]
    exact_normalized_metric: tuple[Fraction, Fraction]
    mutual_singleton_neighbor_indices: tuple[tuple[int], tuple[int]]
    operator_name: Literal["Reception"]
    operator_glyph: Literal["EN"]
    stage_schedule: Literal["two_phase_jacobi"]
    binary64_mix_factor: float
    exact_mix_factor: Fraction
    exact_ideal_consensus_projector: ExactMatrix2
    exact_schedule_energy_gain_upper_bound: Fraction
    exact_pre_schedule_relative_energy_defect_upper_bound: Fraction
    exact_effective_head_energy_gain_upper_bound: Fraction
    active_history_extinction_horizon: int
    policy_certificate: UniformRemeshSchedulePolicyStabilityCertificate = field(
        repr=False
    )
    relative_defect_certificate: (
        UniformRemeshScheduleRelativeDefectStabilityCertificate
    ) = field(repr=False)
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact_after_dependencies_validation(self) -> bool:
        """Rederive this proof after same-call dependency validation.

        The caller must have deeply validated the source class, policy and
        relative-defect certificates immediately before this private path.
        No validation result is cached between public queries.
        """

        try:
            if type(self) is not P2HalfReceptionRemeshStabilityCertificate:
                return False
            if not _sealed(self):
                return False
            source = object.__getattribute__(self, "remesh_class_certificate")
            policy = object.__getattribute__(self, "policy_certificate")
            relative = object.__getattribute__(
                self,
                "relative_defect_certificate",
            )
            if (
                type(source)
                is not UniformAlphaOneHardClipRemeshClassCertificate
                or type(policy)
                is not UniformRemeshSchedulePolicyStabilityCertificate
                or type(relative)
                is not UniformRemeshScheduleRelativeDefectStabilityCertificate
                or not _strict_conditions(self.conditions)
            ):
                return False
            if not proof_stamps_are_identical(
                object.__getattribute__(
                    relative.policy_certificate,
                    "_proof_stamp",
                ),
                object.__getattribute__(policy, "_proof_stamp"),
            ):
                return False
            expected = _model_from_validated_dependencies(
                source,
                policy,
                relative,
            )
            observed_payload = tuple(
                (item.name, object.__getattribute__(self, item.name))
                for item in fields(_P2HalfReceptionModel)
                if item.name
                not in ("policy_certificate", "relative_defect_certificate")
            )
            expected_payload = tuple(
                (item.name, object.__getattribute__(expected, item.name))
                for item in fields(_P2HalfReceptionModel)
                if item.name
                not in ("policy_certificate", "relative_defect_certificate")
            )
            return bool(
                _same(observed_payload, expected_payload)
                and proof_stamps_are_identical(
                    object.__getattribute__(policy, "_proof_stamp"),
                    object.__getattribute__(
                        expected.policy_certificate,
                        "_proof_stamp",
                    ),
                )
                and relative.pre_schedule_relative_energy_defect_upper_bound
                == expected.relative_defect_certificate
                .pre_schedule_relative_energy_defect_upper_bound
                and relative.exact_effective_head_energy_gain_upper_bound
                == expected.relative_defect_certificate
                .exact_effective_head_energy_gain_upper_bound
                and relative.conditions
                == expected.relative_defect_certificate.conditions
            )
        except BaseException:
            return False

    def _proof_fields_are_intact(self) -> bool:
        try:
            source = object.__getattribute__(self, "remesh_class_certificate")
            policy = object.__getattribute__(self, "policy_certificate")
            relative = object.__getattribute__(
                self,
                "relative_defect_certificate",
            )
            if (
                type(source)
                is not UniformAlphaOneHardClipRemeshClassCertificate
                or not source.alpha_one_hard_clip_class_certificate_certified
                or type(policy)
                is not UniformRemeshSchedulePolicyStabilityCertificate
                or not policy.policy_stability_certificate_certified
                or type(relative)
                is not UniformRemeshScheduleRelativeDefectStabilityCertificate
                or not relative.relative_defect_stability_certificate_certified
            ):
                return False
            return self._proof_fields_are_intact_after_dependencies_validation()
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def p2_half_reception_remesh_stability_certificate_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _name, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("p2_half_reception_remesh_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def global_binary64_epi_kernel_family_certified(self) -> bool:
        """Return the global numeric P2 half-Reception kernel result."""

        return self.p2_half_reception_remesh_stability_certificate_certified

    @property
    def numeric_consensus_projection_certified(self) -> bool:
        return self.p2_half_reception_remesh_stability_certificate_certified

    @property
    def restricted_epi_kernel_interval_forward_invariant_certified(self) -> bool:
        return self.p2_half_reception_remesh_stability_certificate_certified

    @property
    def restricted_kernel_support_metric_configuration_preservation_certified(
        self,
    ) -> bool:
        return self.p2_half_reception_remesh_stability_certificate_certified

    @property
    def arbitrary_finite_binary64_kernel_repetition_certified(self) -> bool:
        return self.p2_half_reception_remesh_stability_certificate_certified

    @property
    def active_history_exact_extinction_certified(self) -> bool:
        return self.p2_half_reception_remesh_stability_certificate_certified

    def exact_cycle_energy_gain_upper_bound(self, cycle_count: int) -> Fraction:
        """Return the exact active-history bound for a cycle count."""

        if not self.p2_half_reception_remesh_stability_certificate_certified:
            raise TNFRValueError(
                "P2 half-Reception certificate is unsealed or inconsistent"
            )
        return self.relative_defect_certificate.exact_cycle_energy_gain_upper_bound(
            cycle_count
        )

    def evaluate_binary64_schedule_pair(
        self,
        pair: Binary64Pair | list[float],
    ) -> Binary64Pair:
        """Replay the certified shared numeric EPI kernel on one class pair.

        Equality is numeric.  The returned zero sign is an implementation
        result and is deliberately absent from the theorem.
        """

        if not self.p2_half_reception_remesh_stability_certificate_certified:
            raise TNFRValueError(
                "P2 half-Reception certificate is unsealed or inconsistent"
            )
        if type(pair) is tuple:
            values = tuple(tuple.__iter__(pair))
        elif type(pair) is list:
            values = tuple(list.__iter__(pair))
        else:
            raise TNFRValueError("pair must be a two-item tuple or list")
        if len(values) != 2:
            raise TNFRValueError("pair must contain exactly two binary64 values")
        lower = self.remesh_class_certificate.configuration.epi_min
        upper = self.remesh_class_certificate.configuration.epi_max
        typed_values = (values[0], values[1])
        return _evaluate_binary64_half_reception_pair(
            typed_values,
            lower=lower,
            upper=upper,
        )

    @property
    def signed_zero_bit_preservation_certified(self) -> bool:
        return False

    @property
    def global_binary64_runtime_affinity_certified(self) -> bool:
        return False

    @property
    def complete_reception_stage_certified(self) -> bool:
        return False

    @property
    def grammar_execution_certified(self) -> bool:
        return False

    @property
    def live_graph_execution_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def certify_p2_half_reception_remesh_stability(
    remesh_class_certificate: UniformAlphaOneHardClipRemeshClassCertificate,
) -> P2HalfReceptionRemeshStabilityCertificate:
    """Certify the global restricted P2 half-Reception/REMESH kernel family."""

    if (
        type(remesh_class_certificate)
        is not UniformAlphaOneHardClipRemeshClassCertificate
    ):
        raise TNFRValueError(
            "remesh_class_certificate must be an exact alpha-one hard-clip "
            "REMESH class certificate"
        )
    if not (
        remesh_class_certificate.alpha_one_hard_clip_class_certificate_certified
    ):
        raise TNFRValueError(
            "remesh_class_certificate is unsealed, tampered, or inconsistent"
        )
    model = _derive_model(remesh_class_certificate)
    if not all(passed for _name, passed in model.conditions):
        failed = tuple(name for name, passed in model.conditions if not passed)
        raise TNFRValueError(
            f"P2 half-Reception REMESH proof failed: {failed}"
        )
    value = P2HalfReceptionRemeshStabilityCertificate(
        remesh_class_certificate=remesh_class_certificate,
        node_order=model.node_order,
        exact_normalized_metric=model.exact_normalized_metric,
        mutual_singleton_neighbor_indices=(
            model.mutual_singleton_neighbor_indices
        ),
        operator_name=model.operator_name,
        operator_glyph=model.operator_glyph,
        stage_schedule=model.stage_schedule,
        binary64_mix_factor=model.binary64_mix_factor,
        exact_mix_factor=model.exact_mix_factor,
        exact_ideal_consensus_projector=model.exact_ideal_consensus_projector,
        exact_schedule_energy_gain_upper_bound=(
            model.exact_schedule_energy_gain_upper_bound
        ),
        exact_pre_schedule_relative_energy_defect_upper_bound=(
            model.exact_pre_schedule_relative_energy_defect_upper_bound
        ),
        exact_effective_head_energy_gain_upper_bound=(
            model.exact_effective_head_energy_gain_upper_bound
        ),
        active_history_extinction_horizon=(
            model.active_history_extinction_horizon
        ),
        policy_certificate=model.policy_certificate,
        relative_defect_certificate=model.relative_defect_certificate,
        conditions=model.conditions,
    )
    sealed = _seal(value)
    if not sealed._proof_fields_are_intact_after_dependencies_validation():
        raise RuntimeError(
            "constructed P2 half-Reception REMESH proof is inconsistent"
        )
    return sealed
