r"""Exact pairwise diagnostics and the binary64 ``alpha=1`` REMESH class.

The pairwise observer evaluates the production nested binary64 recurrence and
compares its squared spatial separation with the exact-rational REMESH model.
Its denominator

    D_c = beta*d_current**2 + gamma*d_local**2 + delta*d_global**2

is the scalar contribution that lifts, through the pairwise variance identity,
to the Jensen budget used by the REMESH stability theorem.  One observation is
a local witness; it is not a uniform bound over a runtime state class.

The second certificate isolates the exact positive boundary ``alpha=1`` with
hard clipping.  For every sufficient represented history whose finite
binary64 entries lie in the configured interval, the nested recurrence is
numerically equal to the global delayed row and the clamp is the identity.
Thus its signed centered-energy defect is uniformly zero and the REMESH step
preserves the represented class.  The certificate does not supply a global
schedule family, repeated Event/REMESH execution, or future stability.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from numbers import Real
from typing import Any, Literal

from .._remesh_contract import (
    DelayedRemeshConfiguration,
    materialize_delayed_remesh_configuration,
    materialize_positive_diagonal_metric,
)
from ..dynamics.structural_clip import structural_clip
from ..errors import TNFRValueError
from ..operators._delayed_remesh_kernel import (
    _evaluate_delayed_remesh_binary64,
)
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate,
    certify_uniform_remesh_history_stability,
)

__all__ = (
    "Binary64RemeshPairRelativeDefectObservation",
    "UniformAlphaOneHardClipRemeshClassCertificate",
    "certify_alpha_one_hard_clip_remesh_class",
    "observe_binary64_remesh_pair_relative_defect",
)

ExactPair = tuple[Fraction, Fraction]
Binary64Pair = tuple[float, float]

_PAIR_PROOF_VERSION = "binary64_remesh_pair_relative_defect_v1"
_CLASS_PROOF_VERSION = "uniform_alpha_one_hard_clip_remesh_class_v1"
_PAIR_SCOPE = (
    "One exact pairwise comparison between the exact-rational REMESH head and "
    "the production nested binary64 evaluation followed by the configured "
    "clip. The result is a local scalar witness. It does not maximize over an "
    "admissible alphabet, prove a uniform binary64 relative-defect bound, bind "
    "a graph execution, certify future behavior, establish solver accuracy, "
    "or prove full TNFR stability."
)
_CLASS_SCOPE = (
    "Uniform binary64 REMESH-only state class for alpha=1, one fixed nonempty "
    "ordered support, one fixed positive normalized metric, fixed positive "
    "delays and one fixed finite hard-clipping interval. Every sufficient "
    "represented chronological history with binary64 rows in the interval "
    "maps numerically to its global delayed row, so the relative centered-"
    "energy defect is zero and the REMESH step preserves the class. Signed-"
    "zero bits need not be preserved. The certificate does not establish a "
    "global schedule family, repeated or future Event/REMESH execution, solver "
    "properties, adaptive grammar, or full TNFR stability."
)

_PAIR_CONDITION_NAMES = (
    "canonical_nonnegative_coefficient_partition",
    "all_inputs_are_finite_binary64",
    "runtime_raw_pair_replays_shared_nested_kernel",
    "runtime_bounded_pair_replays_configured_clip",
    "pairwise_jensen_denominator_exact_nonnegative",
    "signed_squared_separation_defects_telescope",
    "relative_defect_ratio_uses_pairwise_jensen_denominator",
    "zero_denominator_branch_avoids_division",
    "hard_clipping_does_not_increase_pairwise_separation",
)
_CLASS_CONDITION_NAMES = (
    "ordered_support_is_nonempty_and_unique",
    "exact_metric_is_positive_and_normalized",
    "configuration_is_alpha_one_hard_clip",
    "runtime_history_capacity_covers_both_delays",
    "exact_remesh_model_is_pure_global_delay",
    "nested_binary64_map_is_numerically_global_delay",
    "hard_clip_is_identity_on_declared_interval",
    "uniform_relative_centered_energy_defect_is_zero",
    "remesh_output_preserves_declared_interval_and_support",
)


def _proof_stamp(value: Any, expected_type: type[Any], version: str) -> tuple[Any, ...]:
    if type(value) is not expected_type:
        raise TypeError("proof value must have its canonical result type")
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(expected_type)
        if item.name != "_proof_stamp"
    )
    return (version, structural_proof_signature(payload))


def _seal(value: Any, expected_type: type[Any], version: str) -> Any:
    return replace(
        value,
        _proof_stamp=_proof_stamp(value, expected_type, version),
    )


def _sealed(value: Any, expected_type: type[Any], version: str) -> bool:
    try:
        observed = object.__getattribute__(value, "_proof_stamp")
        expected = _proof_stamp(value, expected_type, version)
        return proof_stamps_are_identical(observed, expected)
    except BaseException:
        return False


def _strict_conditions(value: Any, names: tuple[str, ...]) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(names)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == names[index]
            for index, item in enumerate(value)
        )
    )


def _stored_binary64_pair_is_canonical(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is float and math.isfinite(item) for item in value)
    )


def _stored_exact_pair_is_canonical(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is Fraction for item in value)
    )


def _pair_payload_types_are_canonical(value: Any) -> bool:
    exact_fraction_fields = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epi_min",
        "epi_max",
        "exact_input_pairwise_jensen_denominator",
        "exact_ideal_squared_separation",
        "exact_raw_squared_separation",
        "exact_bounded_squared_separation",
        "exact_rounding_signed_squared_separation_defect",
        "exact_clipping_signed_squared_separation_defect",
        "exact_total_signed_squared_separation_defect",
    )
    optional_fraction_fields = (
        "exact_relative_signed_defect_ratio",
        "exact_minimum_nonnegative_relative_defect_bound",
    )
    binary64_pair_fields = (
        "binary64_current_pair",
        "binary64_local_pair",
        "binary64_global_pair",
        "runtime_raw_pair",
        "runtime_bounded_pair",
    )
    exact_pair_fields = (
        "exact_current_pair",
        "exact_local_pair",
        "exact_global_pair",
        "exact_ideal_pair",
        "exact_runtime_raw_pair",
        "exact_runtime_bounded_pair",
    )
    binary64_scalar_fields = (
        "binary64_alpha",
        "binary64_epi_min",
        "binary64_epi_max",
    )
    try:
        if not all(
            type(object.__getattribute__(value, name)) is Fraction
            for name in exact_fraction_fields
        ):
            return False
        if not all(
            (
                object.__getattribute__(value, name) is None
                or type(object.__getattribute__(value, name)) is Fraction
            )
            for name in optional_fraction_fields
        ):
            return False
        if not all(
            _stored_binary64_pair_is_canonical(
                object.__getattribute__(value, name)
            )
            for name in binary64_pair_fields
        ):
            return False
        if not all(
            _stored_exact_pair_is_canonical(
                object.__getattribute__(value, name)
            )
            for name in exact_pair_fields
        ):
            return False
        if not all(
            type(object.__getattribute__(value, name)) is float
            and math.isfinite(object.__getattribute__(value, name))
            for name in binary64_scalar_fields
        ):
            return False
        clipping = object.__getattribute__(value, "clipping_intervened")
        clip_mode = object.__getattribute__(value, "clip_mode")
        conditions = object.__getattribute__(value, "conditions")
        return bool(
            type(clipping) is tuple
            and len(clipping) == 2
            and all(type(item) is bool for item in clipping)
            and type(clip_mode) is str
            and clip_mode in ("hard", "soft")
            and _strict_conditions(conditions, _PAIR_CONDITION_NAMES)
        )
    except BaseException:
        return False


def _strict_binary64(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TNFRValueError(f"{label} must be a finite binary64 float")
    return value


def _binary64_pair(value: Any, label: str) -> Binary64Pair:
    if type(value) not in (tuple, list):
        raise TNFRValueError(f"{label} must contain two binary64 floats")
    try:
        if type(value) is tuple:
            items = tuple(tuple.__iter__(value))
        else:
            items = tuple(list.__iter__(value))
    except BaseException as exc:
        raise TNFRValueError(f"{label} must contain two binary64 floats") from exc
    if len(items) != 2:
        raise TNFRValueError(f"{label} must contain two binary64 floats")
    return (
        _strict_binary64(items[0], f"{label}[0]"),
        _strict_binary64(items[1], f"{label}[1]"),
    )


def _exact_pair(value: Binary64Pair) -> ExactPair:
    return (Fraction.from_float(value[0]), Fraction.from_float(value[1]))


def _square_separation(value: ExactPair) -> Fraction:
    difference = value[1] - value[0]
    return difference * difference


def _pair_values(
    current_pair: Any,
    local_pair: Any,
    global_pair: Any,
    *,
    alpha: Any,
    epi_min: Any,
    epi_max: Any,
    clip_mode: Any,
) -> dict[str, Any]:
    current = _binary64_pair(current_pair, "current_pair")
    local = _binary64_pair(local_pair, "local_pair")
    global_ = _binary64_pair(global_pair, "global_pair")
    configuration = materialize_delayed_remesh_configuration(
        tau_local=1,
        tau_global=2,
        alpha=alpha,
        alpha_source="pairwise_relative_defect_observation",
        epi_min=epi_min,
        epi_max=epi_max,
        clip_mode=clip_mode,
    )
    lower = configuration.epi_min
    upper = configuration.epi_max
    all_inputs = current + local + global_

    alpha_float = configuration.alpha
    alpha_q = Fraction.from_float(alpha_float)
    one_minus = Fraction(1) - alpha_q
    beta = one_minus * one_minus
    gamma = alpha_q * one_minus
    delta = alpha_q
    current_q = _exact_pair(current)
    local_q = _exact_pair(local)
    global_q = _exact_pair(global_)
    ideal = tuple(
        beta * current_value
        + gamma * local_value
        + delta * global_value
        for current_value, local_value, global_value in zip(
            current_q,
            local_q,
            global_q,
            strict=True,
        )
    )
    raw = tuple(
        _evaluate_delayed_remesh_binary64(
            current_value,
            local_value,
            global_value,
            alpha_float,
        )
        for current_value, local_value, global_value in zip(
            current,
            local,
            global_,
            strict=True,
        )
    )
    if any(not math.isfinite(value) for value in raw):
        raise TNFRValueError("binary64 REMESH pair evaluation is not finite")
    bounded = tuple(
        structural_clip(
            value,
            lo=lower,
            hi=upper,
            mode=configuration.clip_mode,
            record_stats=False,
        )
        for value in raw
    )
    if any(not math.isfinite(value) for value in bounded):
        raise TNFRValueError("bounded binary64 REMESH pair is not finite")
    raw_q = _exact_pair(raw)  # type: ignore[arg-type]
    bounded_q = _exact_pair(bounded)  # type: ignore[arg-type]

    current_difference = current_q[1] - current_q[0]
    local_difference = local_q[1] - local_q[0]
    global_difference = global_q[1] - global_q[0]
    denominator = (
        beta * current_difference * current_difference
        + gamma * local_difference * local_difference
        + delta * global_difference * global_difference
    )
    ideal_squared = _square_separation(ideal)  # type: ignore[arg-type]
    raw_squared = _square_separation(raw_q)
    bounded_squared = _square_separation(bounded_q)
    rounding_defect = raw_squared - ideal_squared
    clipping_defect = bounded_squared - raw_squared
    total_defect = bounded_squared - ideal_squared
    ratio = None if denominator == 0 else total_defect / denominator
    minimum_eta = (
        Fraction(0)
        if denominator == 0 and total_defect <= 0
        else None
        if denominator == 0
        else max(Fraction(0), ratio)
    )
    clipping_intervened = tuple(
        bounded_value != raw_value
        for raw_value, bounded_value in zip(raw, bounded, strict=True)
    )
    conditions = (
        (
            "canonical_nonnegative_coefficient_partition",
            beta >= 0
            and gamma >= 0
            and delta >= 0
            and beta + gamma + delta == 1,
        ),
        ("all_inputs_are_finite_binary64", all(map(math.isfinite, all_inputs))),
        (
            "runtime_raw_pair_replays_shared_nested_kernel",
            all(
                observed.hex()
                == _evaluate_delayed_remesh_binary64(
                    current_value,
                    local_value,
                    global_value,
                    alpha_float,
                ).hex()
                for observed, current_value, local_value, global_value in zip(
                    raw,
                    current,
                    local,
                    global_,
                    strict=True,
                )
            ),
        ),
        (
            "runtime_bounded_pair_replays_configured_clip",
            all(
                observed.hex()
                == structural_clip(
                    raw_value,
                    lo=lower,
                    hi=upper,
                    mode=configuration.clip_mode,
                    record_stats=False,
                ).hex()
                for observed, raw_value in zip(bounded, raw, strict=True)
            ),
        ),
        (
            "pairwise_jensen_denominator_exact_nonnegative",
            denominator >= 0,
        ),
        (
            "signed_squared_separation_defects_telescope",
            rounding_defect + clipping_defect == total_defect,
        ),
        (
            "relative_defect_ratio_uses_pairwise_jensen_denominator",
            ratio == (None if denominator == 0 else total_defect / denominator),
        ),
        (
            "zero_denominator_branch_avoids_division",
            denominator != 0 or (ratio is None and total_defect <= 0),
        ),
        (
            "hard_clipping_does_not_increase_pairwise_separation",
            configuration.clip_mode != "hard" or clipping_defect <= 0,
        ),
    )
    return {
        "binary64_alpha": alpha_float,
        "alpha": alpha_q,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "binary64_epi_min": lower,
        "binary64_epi_max": upper,
        "epi_min": Fraction.from_float(lower),
        "epi_max": Fraction.from_float(upper),
        "clip_mode": configuration.clip_mode,
        "binary64_current_pair": current,
        "binary64_local_pair": local,
        "binary64_global_pair": global_,
        "exact_current_pair": current_q,
        "exact_local_pair": local_q,
        "exact_global_pair": global_q,
        "exact_ideal_pair": ideal,
        "runtime_raw_pair": raw,
        "runtime_bounded_pair": bounded,
        "exact_runtime_raw_pair": raw_q,
        "exact_runtime_bounded_pair": bounded_q,
        "clipping_intervened": clipping_intervened,
        "exact_input_pairwise_jensen_denominator": denominator,
        "exact_ideal_squared_separation": ideal_squared,
        "exact_raw_squared_separation": raw_squared,
        "exact_bounded_squared_separation": bounded_squared,
        "exact_rounding_signed_squared_separation_defect": rounding_defect,
        "exact_clipping_signed_squared_separation_defect": clipping_defect,
        "exact_total_signed_squared_separation_defect": total_defect,
        "exact_relative_signed_defect_ratio": ratio,
        "exact_minimum_nonnegative_relative_defect_bound": minimum_eta,
        "conditions": conditions,
    }


@dataclass(frozen=True, slots=True)
class Binary64RemeshPairRelativeDefectObservation:
    """Sealed exact diagnostic for one pair of REMESH input tuples."""

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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            if type(self) is not Binary64RemeshPairRelativeDefectObservation:
                return False
            if not _sealed(
                self,
                Binary64RemeshPairRelativeDefectObservation,
                _PAIR_PROOF_VERSION,
            ):
                return False
            if not _pair_payload_types_are_canonical(self):
                return False
            current = object.__getattribute__(self, "binary64_current_pair")
            local = object.__getattribute__(self, "binary64_local_pair")
            global_ = object.__getattribute__(self, "binary64_global_pair")
            binary64_alpha = object.__getattribute__(self, "binary64_alpha")
            binary64_epi_min = object.__getattribute__(self, "binary64_epi_min")
            binary64_epi_max = object.__getattribute__(self, "binary64_epi_max")
            clip_mode = object.__getattribute__(self, "clip_mode")
            conditions = object.__getattribute__(self, "conditions")
            expected = _pair_values(
                current,
                local,
                global_,
                alpha=binary64_alpha,
                epi_min=binary64_epi_min,
                epi_max=binary64_epi_max,
                clip_mode=clip_mode,
            )
            expected_value = Binary64RemeshPairRelativeDefectObservation(
                **expected
            )
            return proof_stamps_are_identical(
                _proof_stamp(
                    self,
                    Binary64RemeshPairRelativeDefectObservation,
                    _PAIR_PROOF_VERSION,
                ),
                _proof_stamp(
                    expected_value,
                    Binary64RemeshPairRelativeDefectObservation,
                    _PAIR_PROOF_VERSION,
                ),
            ) and _strict_conditions(conditions, _PAIR_CONDITION_NAMES)
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _PAIR_SCOPE

    @property
    def pair_relative_defect_observation_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _name, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return (
                "binary64_remesh_pair_relative_defect_proof_fields_intact",
            )
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def hard_clipping_pairwise_nonexpansive_certified(self) -> bool:
        return bool(
            self.pair_relative_defect_observation_certified
            and self.clip_mode == "hard"
            and self.exact_clipping_signed_squared_separation_defect <= 0
        )

    @property
    def uniform_binary64_relative_defect_bound_certified(self) -> bool:
        return False

    @property
    def future_binary64_relative_defect_bound_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def observe_binary64_remesh_pair_relative_defect(
    current_pair: Binary64Pair | list[float],
    local_pair: Binary64Pair | list[float],
    global_pair: Binary64Pair | list[float],
    *,
    alpha: Real,
    epi_min: Real,
    epi_max: Real,
    clip_mode: Literal["hard", "soft"] = "hard",
) -> Binary64RemeshPairRelativeDefectObservation:
    """Observe one exact pairwise binary64 REMESH relative defect."""

    try:
        values = _pair_values(
            current_pair,
            local_pair,
            global_pair,
            alpha=alpha,
            epi_min=epi_min,
            epi_max=epi_max,
            clip_mode=clip_mode,
        )
    except TNFRValueError:
        raise
    except BaseException as exc:
        raise TNFRValueError(
            "invalid binary64 REMESH pair observation inputs"
        ) from exc
    value = Binary64RemeshPairRelativeDefectObservation(**values)
    result = _seal(value, type(value), _PAIR_PROOF_VERSION)
    if not result.pair_relative_defect_observation_certified:
        raise RuntimeError("constructed binary64 REMESH pair proof is inconsistent")
    return result


def _node_order(nodes: Any) -> tuple[Hashable, ...]:
    if isinstance(nodes, (str, bytes, bytearray, Mapping)):
        raise TNFRValueError("nodes must be a nonempty iterable of identifiers")
    try:
        result = tuple(nodes)
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(
            "nodes must be a nonempty iterable of identifiers"
        ) from exc
    if not result:
        raise TNFRValueError("nodes must be nonempty")
    try:
        if len(frozenset(result)) != len(result):
            raise TNFRValueError("nodes must contain unique identifiers")
        tuple(structural_proof_signature(node) for node in result)
    except TNFRValueError:
        raise
    except BaseException as exc:
        raise TNFRValueError("nodes contain invalid identifiers") from exc
    return result


def _normalized_metric(
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None,
    nodes: tuple[Hashable, ...],
) -> tuple[Fraction, ...]:
    materialized = materialize_positive_diagonal_metric(metric_weights, nodes)
    exact = tuple(Fraction.from_float(value) for value in materialized)
    total = sum(exact, Fraction(0))
    return tuple(value / total for value in exact)


def _class_conditions(
    node_order: tuple[Hashable, ...],
    metric: tuple[Fraction, ...],
    configuration: DelayedRemeshConfiguration,
    remesh: UniformRemeshHistoryStabilityCertificate,
) -> tuple[tuple[str, bool], ...]:
    required = max(configuration.tau_local, configuration.tau_global) + 1
    node_signatures = tuple(
        structural_proof_signature(node) for node in node_order
    )
    support_valid = bool(
        node_order
        and len(node_order) == len(set(node_signatures))
    )
    metric_valid = bool(
        len(metric) == len(node_order)
        and all(type(value) is Fraction and value > 0 for value in metric)
        and sum(metric, Fraction(0)) == 1
    )
    pure_global = bool(
        remesh.stability_certificate_certified
        and remesh.alpha == 1
        and remesh.beta == 0
        and remesh.gamma == 0
        and remesh.delta == 1
        and remesh.combined_delay_coefficients
        == ((configuration.tau_global, Fraction(1)),)
    )
    return (
        ("ordered_support_is_nonempty_and_unique", support_valid),
        ("exact_metric_is_positive_and_normalized", metric_valid),
        (
            "configuration_is_alpha_one_hard_clip",
            configuration.alpha == 1.0
            and configuration.clip_mode == "hard"
            and math.isfinite(configuration.epi_min)
            and math.isfinite(configuration.epi_max)
            and configuration.epi_min <= configuration.epi_max,
        ),
        (
            "runtime_history_capacity_covers_both_delays",
            configuration.history_maxlen >= required,
        ),
        ("exact_remesh_model_is_pure_global_delay", pure_global),
        ("nested_binary64_map_is_numerically_global_delay", True),
        ("hard_clip_is_identity_on_declared_interval", True),
        ("uniform_relative_centered_energy_defect_is_zero", True),
        ("remesh_output_preserves_declared_interval_and_support", True),
    )


@dataclass(frozen=True, slots=True)
class UniformAlphaOneHardClipRemeshClassCertificate:
    """Sealed REMESH-only forward-invariant class with uniform ``eta=0``."""

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
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            if type(self) is not UniformAlphaOneHardClipRemeshClassCertificate:
                return False
            if not _sealed(
                self,
                UniformAlphaOneHardClipRemeshClassCertificate,
                _CLASS_PROOF_VERSION,
            ):
                return False
            configuration = object.__getattribute__(self, "configuration")
            node_order = object.__getattribute__(self, "node_order")
            metric = object.__getattribute__(self, "exact_normalized_metric")
            remesh = object.__getattribute__(self, "remesh_certificate")
            conditions = object.__getattribute__(self, "conditions")
            if (
                type(configuration) is not DelayedRemeshConfiguration
                or type(node_order) is not tuple
                or not node_order
                or type(metric) is not tuple
                or len(metric) != len(node_order)
                or not all(type(value) is Fraction and value > 0 for value in metric)
                or sum(metric, Fraction(0)) != 1
                or type(remesh) is not UniformRemeshHistoryStabilityCertificate
                or not _strict_conditions(conditions, _CLASS_CONDITION_NAMES)
                or type(self.tau_local) is not int
                or self.tau_local <= 0
                or type(self.tau_global) is not int
                or self.tau_global <= 0
                or type(self.history_maxlen) is not int
                or self.history_maxlen <= 0
                or type(self.required_history_length) is not int
                or self.required_history_length <= 0
                or type(self.epi_min) is not Fraction
                or type(self.epi_max) is not Fraction
                or type(self.alpha) is not Fraction
                or type(self.clip_mode) is not str
                or type(self.exact_uniform_relative_defect_upper_bound)
                is not Fraction
                or type(configuration.tau_local) is not int
                or configuration.tau_local <= 0
                or type(configuration.tau_global) is not int
                or configuration.tau_global <= 0
                or type(configuration.history_maxlen) is not int
                or configuration.history_maxlen <= 0
                or type(configuration.alpha) is not float
                or not math.isfinite(configuration.alpha)
                or type(configuration.alpha_source) is not str
                or not configuration.alpha_source
                or type(configuration.epi_min) is not float
                or not math.isfinite(configuration.epi_min)
                or type(configuration.epi_max) is not float
                or not math.isfinite(configuration.epi_max)
                or type(configuration.clip_mode) is not str
            ):
                return False
            canonical = materialize_delayed_remesh_configuration(
                tau_local=self.tau_local,
                tau_global=self.tau_global,
                alpha=1.0,
                alpha_source=configuration.alpha_source,
                epi_min=configuration.epi_min,
                epi_max=configuration.epi_max,
                clip_mode="hard",
            )
            expected_remesh = certify_uniform_remesh_history_stability(
                alpha=Fraction(1),
                tau_local=self.tau_local,
                tau_global=self.tau_global,
            )
            expected_conditions = _class_conditions(
                node_order,
                metric,
                canonical,
                expected_remesh,
            )
            return bool(
                configuration == canonical
                and self.history_maxlen == canonical.history_maxlen
                and self.required_history_length
                == max(self.tau_local, self.tau_global) + 1
                and self.epi_min == Fraction.from_float(canonical.epi_min)
                and self.epi_max == Fraction.from_float(canonical.epi_max)
                and self.alpha == 1
                and self.clip_mode == "hard"
                and self.exact_uniform_relative_defect_upper_bound == 0
                and remesh.stability_certificate_certified
                and proof_stamps_are_identical(
                    object.__getattribute__(remesh, "_proof_stamp"),
                    object.__getattribute__(expected_remesh, "_proof_stamp"),
                )
                and conditions == expected_conditions
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _CLASS_SCOPE

    @property
    def alpha_one_hard_clip_class_certificate_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _name, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("alpha_one_hard_clip_remesh_class_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def binary64_global_delay_numeric_copy_certified(self) -> bool:
        return self.alpha_one_hard_clip_class_certificate_certified

    @property
    def hard_clip_identity_on_class_certified(self) -> bool:
        """Return numeric clamp identity; signed-zero bits may canonicalize."""

        return self.alpha_one_hard_clip_class_certificate_certified

    @property
    def remesh_class_forward_invariant_certified(self) -> bool:
        return self.alpha_one_hard_clip_class_certificate_certified

    def represented_history_belongs_to_class(self, history: Any) -> bool:
        """Check one chronological row-vector view against the pure class."""

        if not self.alpha_one_hard_clip_class_certificate_certified:
            return False
        if type(history) not in (tuple, list):
            return False
        try:
            if type(history) is tuple:
                rows = tuple(tuple.__iter__(history))
            else:
                rows = tuple(list.__iter__(history))
        except BaseException:
            return False
        if not (
            self.required_history_length <= len(rows) <= self.history_maxlen
        ):
            return False
        width = len(self.node_order)
        lower = self.configuration.epi_min
        upper = self.configuration.epi_max
        for row in rows:
            if type(row) is tuple:
                values = tuple(tuple.__iter__(row))
            elif type(row) is list:
                values = tuple(list.__iter__(row))
            else:
                return False
            if len(values) != width:
                return False
            if any(
                type(value) is not float
                or not math.isfinite(value)
                or value < lower
                or value > upper
                for value in values
            ):
                return False
        return True

    @property
    def schedule_family_certificate_certified(self) -> bool:
        return False

    @property
    def repeated_binary64_stability_certified(self) -> bool:
        return False

    @property
    def future_binary64_execution_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False


def certify_alpha_one_hard_clip_remesh_class(
    nodes: Iterable[Hashable],
    metric_weights: Mapping[Hashable, Real] | Sequence[Real] | None = None,
    *,
    tau_local: int,
    tau_global: int,
    epi_min: Real,
    epi_max: Real,
) -> UniformAlphaOneHardClipRemeshClassCertificate:
    """Certify the REMESH-only ``alpha=1`` hard-clipped binary64 class."""

    try:
        node_order = _node_order(nodes)
        metric = _normalized_metric(metric_weights, node_order)
        configuration = materialize_delayed_remesh_configuration(
            tau_local=tau_local,
            tau_global=tau_global,
            alpha=1.0,
            alpha_source="alpha_one_hard_clip_class_certificate",
            epi_min=epi_min,
            epi_max=epi_max,
            clip_mode="hard",
        )
    except TNFRValueError:
        raise
    except BaseException as exc:
        raise TNFRValueError(
            "invalid alpha-one REMESH class inputs"
        ) from exc
    remesh = certify_uniform_remesh_history_stability(
        alpha=Fraction(1),
        tau_local=configuration.tau_local,
        tau_global=configuration.tau_global,
    )
    conditions = _class_conditions(node_order, metric, configuration, remesh)
    if not all(passed for _name, passed in conditions):
        failed = tuple(name for name, passed in conditions if not passed)
        raise TNFRValueError(f"alpha-one REMESH class proof failed: {failed}")
    value = UniformAlphaOneHardClipRemeshClassCertificate(
        node_order=node_order,
        exact_normalized_metric=metric,
        configuration=configuration,
        tau_local=configuration.tau_local,
        tau_global=configuration.tau_global,
        history_maxlen=configuration.history_maxlen,
        required_history_length=max(
            configuration.tau_local,
            configuration.tau_global,
        )
        + 1,
        epi_min=Fraction.from_float(configuration.epi_min),
        epi_max=Fraction.from_float(configuration.epi_max),
        alpha=Fraction(1),
        clip_mode="hard",
        remesh_certificate=remesh,
        exact_uniform_relative_defect_upper_bound=Fraction(0),
        conditions=conditions,
    )
    result = _seal(value, type(value), _CLASS_PROOF_VERSION)
    if not result.alpha_one_hard_clip_class_certificate_certified:
        raise RuntimeError("constructed alpha-one REMESH class proof is inconsistent")
    return result
