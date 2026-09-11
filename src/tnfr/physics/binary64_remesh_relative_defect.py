r"""Exact pairwise diagnostics and scoped binary64 REMESH classes.

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

The third certificate restricts the production ``alpha=1/2`` recurrence to
two-node antisymmetric represented histories and one symmetric hard-clipping
interval ``[-B, B]`` with ``B >= 4 * 2**-1074``.  Odd binary64 halving and
symmetric clipping preserve this class.  Writing ``H(x)=RN(x/2)``, its scalar
amplitude is evaluated by ``a=H(c)``, ``b=H(l)``, ``m=RN(a+b)``, ``d=H(m)``,
``e=H(g)`` and ``r=RN(d+e)``, while the ideal amplitude is
``y=(c+l)/4+g/2``.  Let ``s=2**-1074``, ``u=2**-53``,
``D=c**2+l**2+2*g**2`` and ``x=sqrt(D)``.  Binary64 halving has absolute error
at most ``s/2`` and every finite rounded sum has error at most
``u*abs(z)+s/2``.  Propagating those errors through the four halvings and two
sums gives the following bound; the sums cannot overflow because each combines
halved finite operands:

    abs(r-y) <= A*x + C*s,
    A = 3*u/2 + u**2/2,
    C = 9/4 + 9*u/4 + u**2/2.

When ``x >= 11*s``, putting ``z=A+C/11`` gives the exact strict inequality
``4*(z+z**2) < 135/124``.  When ``x < 11*s``, every input is an integer
multiple of ``s`` with ``abs(c/s),abs(l/s)<=10`` and ``abs(g/s)<=7``.  The
module exhausts those 6,615 integer triples exactly (3,890 have ``0<D<121``)
and obtains

    4 * (r**2 - y**2) / D <= 135/124.

Equality occurs at ``(c,l,g)=(-3,-2,-3)*s``.  Symmetric hard clipping can only
reduce the runtime squared separation.  Composition with the existing robust
envelope is strict exactly when ``q < 124/259``.  These are REMESH-only numeric
claims, not runtime execution or full TNFR stability claims.
"""

from __future__ import annotations

import math
import sys
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
from .remesh_schedule_policy_stability import (
    certify_uniform_remesh_schedule_policy_stability,
)
from .remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
    certify_uniform_remesh_schedule_relative_defect_stability,
)

__all__ = (
    "Binary64RemeshPairRelativeDefectObservation",
    "UniformAlphaOneHardClipRemeshClassCertificate",
    "UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate",
    "certify_alpha_one_hard_clip_remesh_class",
    "certify_half_alpha_antisymmetric_hard_clip_remesh_class",
    "observe_binary64_remesh_pair_relative_defect",
)

ExactPair = tuple[Fraction, Fraction]
Binary64Pair = tuple[float, float]

_PAIR_PROOF_VERSION = "binary64_remesh_pair_relative_defect_v1"
_CLASS_PROOF_VERSION = "uniform_alpha_one_hard_clip_remesh_class_v1"
_HALF_CLASS_PROOF_VERSION = (
    "uniform_half_alpha_antisymmetric_hard_clip_remesh_class_v1"
)
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
_HALF_CLASS_SCOPE = (
    "Uniform binary64 REMESH-only state class on one fixed ordered P2 support, "
    "one fixed positive normalized diagonal metric, fixed positive delays and "
    "one symmetric hard-clipping interval [-B, B] with represented "
    "B >= 4*2**-1074. Every sufficient represented chronological history in "
    "the class has antisymmetric rows (a, -a). The production alpha=1/2 "
    "nested recurrence and symmetric clamp preserve numeric antisymmetry and "
    "the interval. Its optimal uniform signed centered-energy relative-defect "
    "bound is eta=135/124, so the existing robust envelope is strictly "
    "contractive exactly for a declared schedule gain q<124/259. The "
    "certificate does not verify a schedule family, graph or event execution, "
    "future runtime behavior, solver properties, adaptive grammar, or full "
    "TNFR stability."
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
_HALF_CLASS_CONDITION_NAMES = (
    "ordered_support_is_exactly_two_unique_nodes",
    "exact_metric_is_positive_and_normalized",
    "configuration_is_half_alpha_symmetric_hard_clip",
    "represented_bound_contains_sharp_subnormal_witness",
    "runtime_history_capacity_covers_both_delays",
    "exact_remesh_coefficients_are_one_quarter_one_quarter_one_half",
    "runtime_uses_required_ieee_binary64_rounding_model",
    "ieee_sign_symmetry_preserves_numeric_antisymmetry",
    "symmetric_hard_clip_preserves_antisymmetry_and_interval",
    "pairwise_variance_identity_cancels_every_positive_metric_factor",
    "analytic_large_norm_tail_is_strictly_below_uniform_eta",
    "exact_subnormal_core_enumeration_has_sharp_uniform_eta",
    "stored_subnormal_witness_attains_uniform_eta",
    "remesh_output_preserves_antisymmetric_history_class",
    "strict_robust_schedule_gain_threshold_is_reciprocal_one_plus_eta",
    "four_ninths_example_has_positive_exact_robust_margin",
)

_MIN_BINARY64_SUBNORMAL = math.ulp(0.0)
_MIN_HALF_CLASS_BOUND = 4.0 * _MIN_BINARY64_SUBNORMAL
_HALF_CLASS_ETA = Fraction(135, 124)
_HALF_CLASS_STRICT_Q_THRESHOLD = Fraction(124, 259)
_HALF_CLASS_EXAMPLE_Q = Fraction(4, 9)
_HALF_CLASS_EXAMPLE_Q_EFF = Fraction(259, 279)
_HALF_CLASS_EXAMPLE_MARGIN = Fraction(20, 279)
_BINARY64_UNIT_ROUNDOFF = Fraction(1, 2**53)
_HALF_CLASS_TAIL_ERROR_LINEAR_COEFFICIENT = (
    Fraction(3, 2) * _BINARY64_UNIT_ROUNDOFF
    + Fraction(1, 2) * _BINARY64_UNIT_ROUNDOFF**2
)
_HALF_CLASS_TAIL_ERROR_ABSOLUTE_COEFFICIENT = (
    Fraction(9, 4)
    + Fraction(9, 4) * _BINARY64_UNIT_ROUNDOFF
    + Fraction(1, 2) * _BINARY64_UNIT_ROUNDOFF**2
)
_HALF_CLASS_TAIL_ERROR_RATIO = (
    _HALF_CLASS_TAIL_ERROR_LINEAR_COEFFICIENT
    + _HALF_CLASS_TAIL_ERROR_ABSOLUTE_COEFFICIENT / 11
)
_HALF_CLASS_TAIL_RELATIVE_DEFECT_BOUND = 4 * (
    _HALF_CLASS_TAIL_ERROR_RATIO + _HALF_CLASS_TAIL_ERROR_RATIO**2
)
_HALF_CLASS_CLEAN_TAIL_ERROR_RATIO_BOUND = Fraction(21, 100)
_HALF_CLASS_CLEAN_TAIL_RELATIVE_DEFECT_BOUND = Fraction(2541, 2500)


def _round_half_integer_ties_even(value: int) -> int:
    """Round ``value/2`` to the nearest integer with ties to even."""

    sign = -1 if value < 0 else 1
    quotient, remainder = divmod(abs(value), 2)
    if remainder and quotient % 2:
        quotient += 1
    return sign * quotient


def _enumerate_half_class_subnormal_core(
) -> tuple[Fraction, tuple[int, int, int], int, int]:
    """Exhaust the exact ``sqrt(D) < 11*s`` integer reduction."""

    maximum: Fraction | None = None
    maximizer = (0, 0, 0)
    candidate_count = 0
    admissible_count = 0
    for current in range(-10, 11):
        for local in range(-10, 11):
            for global_ in range(-7, 8):
                candidate_count += 1
                denominator = (
                    current * current
                    + local * local
                    + 2 * global_ * global_
                )
                if denominator == 0 or denominator >= 121:
                    continue
                admissible_count += 1
                runtime = _round_half_integer_ties_even(
                    _round_half_integer_ties_even(current)
                    + _round_half_integer_ties_even(local)
                ) + _round_half_integer_ties_even(global_)
                ideal = Fraction(current + local + 2 * global_, 4)
                ratio = Fraction(4) * (runtime * runtime - ideal * ideal)
                ratio /= denominator
                if maximum is None or ratio > maximum:
                    maximum = ratio
                    maximizer = (current, local, global_)
    if maximum is None:
        raise RuntimeError("half-alpha subnormal core enumeration is empty")
    return maximum, maximizer, candidate_count, admissible_count


(
    _HALF_CLASS_CORE_MAXIMUM,
    _HALF_CLASS_CORE_MAXIMIZER,
    _HALF_CLASS_CORE_CANDIDATE_COUNT,
    _HALF_CLASS_CORE_ADMISSIBLE_COUNT,
) = _enumerate_half_class_subnormal_core()


def _runtime_uses_required_binary64_rounding_model() -> bool:
    smallest = _MIN_BINARY64_SUBNORMAL
    try:
        return bool(
            sys.float_info.radix == 2
            and sys.float_info.mant_dig == 53
            and sys.float_info.min_exp == -1021
            and sys.float_info.max_exp == 1024
            and sys.float_info.rounds == 1
            and float.__getformat__("double").startswith("IEEE")
            and smallest == float.fromhex("0x0.0000000000001p-1022")
            and 0.5 * smallest == 0.0
            and 0.5 * (3.0 * smallest) == 2.0 * smallest
            and 0.5 * (-3.0 * smallest) == -2.0 * smallest
        )
    except BaseException:
        return False


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


def _positive_half_class_bound(value: Any) -> float:
    label = "epi_bound"
    if isinstance(value, (bool, str, bytes, bytearray, complex)):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    if not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    if result < _MIN_HALF_CLASS_BOUND:
        raise TNFRValueError(
            "epi_bound must be at least four minimum binary64 subnormals"
        )
    return result


def _half_class_sharpness_witness(
    configuration: DelayedRemeshConfiguration,
) -> Binary64RemeshPairRelativeDefectObservation:
    smallest = _MIN_BINARY64_SUBNORMAL
    return observe_binary64_remesh_pair_relative_defect(
        (-3.0 * smallest, 3.0 * smallest),
        (-2.0 * smallest, 2.0 * smallest),
        (-3.0 * smallest, 3.0 * smallest),
        alpha=0.5,
        epi_min=configuration.epi_min,
        epi_max=configuration.epi_max,
        clip_mode="hard",
    )


def _half_class_conditions(
    node_order: tuple[Hashable, ...],
    metric: tuple[Fraction, ...],
    configuration: DelayedRemeshConfiguration,
    remesh: UniformRemeshHistoryStabilityCertificate,
    witness: Binary64RemeshPairRelativeDefectObservation,
) -> tuple[tuple[str, bool], ...]:
    required = max(configuration.tau_local, configuration.tau_global) + 1
    node_signatures = tuple(
        structural_proof_signature(node) for node in node_order
    )
    support_valid = bool(
        len(node_order) == 2
        and len(node_order) == len(set(node_signatures))
    )
    metric_valid = bool(
        len(metric) == 2
        and all(type(value) is Fraction and value > 0 for value in metric)
        and sum(metric, Fraction(0)) == 1
    )
    configuration_valid = bool(
        configuration.alpha == 0.5
        and configuration.clip_mode == "hard"
        and math.isfinite(configuration.epi_min)
        and math.isfinite(configuration.epi_max)
        and configuration.epi_min == -configuration.epi_max
        and configuration.epi_max >= _MIN_HALF_CLASS_BOUND
    )
    exact_coefficients = bool(
        remesh.stability_certificate_certified
        and remesh.alpha == Fraction(1, 2)
        and remesh.beta == Fraction(1, 4)
        and remesh.gamma == Fraction(1, 4)
        and remesh.delta == Fraction(1, 2)
    )
    witness_valid = bool(
        witness.pair_relative_defect_observation_certified
        and witness.alpha == Fraction(1, 2)
        and witness.binary64_current_pair
        == (
            -3.0 * _MIN_BINARY64_SUBNORMAL,
            3.0 * _MIN_BINARY64_SUBNORMAL,
        )
        and witness.binary64_local_pair
        == (
            -2.0 * _MIN_BINARY64_SUBNORMAL,
            2.0 * _MIN_BINARY64_SUBNORMAL,
        )
        and witness.binary64_global_pair
        == (
            -3.0 * _MIN_BINARY64_SUBNORMAL,
            3.0 * _MIN_BINARY64_SUBNORMAL,
        )
        and witness.runtime_bounded_pair
        == (
            -4.0 * _MIN_BINARY64_SUBNORMAL,
            4.0 * _MIN_BINARY64_SUBNORMAL,
        )
        and witness.exact_relative_signed_defect_ratio == _HALF_CLASS_ETA
        and witness.exact_minimum_nonnegative_relative_defect_bound
        == _HALF_CLASS_ETA
    )
    return (
        ("ordered_support_is_exactly_two_unique_nodes", support_valid),
        ("exact_metric_is_positive_and_normalized", metric_valid),
        (
            "configuration_is_half_alpha_symmetric_hard_clip",
            configuration_valid,
        ),
        (
            "represented_bound_contains_sharp_subnormal_witness",
            configuration.epi_max >= _MIN_HALF_CLASS_BOUND,
        ),
        (
            "runtime_history_capacity_covers_both_delays",
            configuration.history_maxlen >= required,
        ),
        (
            "exact_remesh_coefficients_are_one_quarter_one_quarter_one_half",
            exact_coefficients,
        ),
        (
            "runtime_uses_required_ieee_binary64_rounding_model",
            _runtime_uses_required_binary64_rounding_model(),
        ),
        (
            "ieee_sign_symmetry_preserves_numeric_antisymmetry",
            _runtime_uses_required_binary64_rounding_model()
            and _round_half_integer_ties_even(-3)
            == -_round_half_integer_ties_even(3),
        ),
        (
            "symmetric_hard_clip_preserves_antisymmetry_and_interval",
            configuration_valid,
        ),
        (
            "pairwise_variance_identity_cancels_every_positive_metric_factor",
            metric_valid
            and metric[0] * metric[1] / 2 > 0,
        ),
        (
            "analytic_large_norm_tail_is_strictly_below_uniform_eta",
            _HALF_CLASS_TAIL_ERROR_RATIO
            < _HALF_CLASS_CLEAN_TAIL_ERROR_RATIO_BOUND
            and 4
            * (
                _HALF_CLASS_CLEAN_TAIL_ERROR_RATIO_BOUND
                + _HALF_CLASS_CLEAN_TAIL_ERROR_RATIO_BOUND**2
            )
            == _HALF_CLASS_CLEAN_TAIL_RELATIVE_DEFECT_BOUND
            and _HALF_CLASS_CLEAN_TAIL_RELATIVE_DEFECT_BOUND
            < _HALF_CLASS_ETA,
        ),
        (
            "exact_subnormal_core_enumeration_has_sharp_uniform_eta",
            _HALF_CLASS_CORE_CANDIDATE_COUNT == 6615
            and _HALF_CLASS_CORE_ADMISSIBLE_COUNT == 3890
            and _HALF_CLASS_CORE_MAXIMUM == _HALF_CLASS_ETA
            and _HALF_CLASS_CORE_MAXIMIZER == (-3, -2, -3),
        ),
        ("stored_subnormal_witness_attains_uniform_eta", witness_valid),
        (
            "remesh_output_preserves_antisymmetric_history_class",
            configuration_valid
            and exact_coefficients
            and _runtime_uses_required_binary64_rounding_model(),
        ),
        (
            "strict_robust_schedule_gain_threshold_is_reciprocal_one_plus_eta",
            _HALF_CLASS_STRICT_Q_THRESHOLD
            == Fraction(1, 1) / (Fraction(1, 1) + _HALF_CLASS_ETA),
        ),
        (
            "four_ninths_example_has_positive_exact_robust_margin",
            _HALF_CLASS_EXAMPLE_Q
            * (Fraction(1, 1) + _HALF_CLASS_ETA)
            == _HALF_CLASS_EXAMPLE_Q_EFF
            and Fraction(1, 1) - _HALF_CLASS_EXAMPLE_Q_EFF
            == _HALF_CLASS_EXAMPLE_MARGIN
            and _HALF_CLASS_EXAMPLE_MARGIN > 0,
        ),
    )


@dataclass(frozen=True, slots=True)
class UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate:
    """Sealed sharp ``alpha=1/2`` antisymmetric P2 REMESH class."""

    node_order: tuple[Hashable, Hashable]
    exact_normalized_metric: tuple[Fraction, Fraction]
    configuration: DelayedRemeshConfiguration = field(repr=False)
    tau_local: int
    tau_global: int
    history_maxlen: int
    required_history_length: int
    epi_bound: Fraction
    epi_min: Fraction
    epi_max: Fraction
    alpha: Fraction
    clip_mode: Literal["hard"]
    remesh_certificate: UniformRemeshHistoryStabilityCertificate = field(
        repr=False
    )
    exact_uniform_relative_defect_upper_bound: Fraction
    exact_tail_error_linear_coefficient: Fraction
    exact_tail_error_absolute_coefficient: Fraction
    exact_tail_error_ratio_at_eleven_subnormals: Fraction
    exact_tail_relative_defect_upper_bound: Fraction
    finite_core_candidate_count: int
    finite_core_admissible_count: int
    finite_core_maximizer_amplitudes: tuple[int, int, int]
    exact_strict_schedule_gain_threshold: Fraction
    example_schedule_energy_gain_upper_bound: Fraction
    exact_example_effective_head_energy_gain_upper_bound: Fraction
    exact_example_normalized_block_margin_lower_bound: Fraction
    sharpness_witness: Binary64RemeshPairRelativeDefectObservation = field(
        repr=False
    )
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            if (
                type(self)
                is not UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate
            ):
                return False
            if not _sealed(
                self,
                UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate,
                _HALF_CLASS_PROOF_VERSION,
            ):
                return False
            configuration = object.__getattribute__(self, "configuration")
            node_order = object.__getattribute__(self, "node_order")
            metric = object.__getattribute__(self, "exact_normalized_metric")
            remesh = object.__getattribute__(self, "remesh_certificate")
            witness = object.__getattribute__(self, "sharpness_witness")
            conditions = object.__getattribute__(self, "conditions")
            exact_fraction_fields = (
                "epi_bound",
                "epi_min",
                "epi_max",
                "alpha",
                "exact_uniform_relative_defect_upper_bound",
                "exact_tail_error_linear_coefficient",
                "exact_tail_error_absolute_coefficient",
                "exact_tail_error_ratio_at_eleven_subnormals",
                "exact_tail_relative_defect_upper_bound",
                "exact_strict_schedule_gain_threshold",
                "example_schedule_energy_gain_upper_bound",
                "exact_example_effective_head_energy_gain_upper_bound",
                "exact_example_normalized_block_margin_lower_bound",
            )
            if (
                type(configuration) is not DelayedRemeshConfiguration
                or type(node_order) is not tuple
                or len(node_order) != 2
                or type(metric) is not tuple
                or len(metric) != 2
                or not all(
                    type(value) is Fraction and value > 0 for value in metric
                )
                or sum(metric, Fraction(0)) != 1
                or type(remesh) is not UniformRemeshHistoryStabilityCertificate
                or type(witness)
                is not Binary64RemeshPairRelativeDefectObservation
                or not _strict_conditions(
                    conditions,
                    _HALF_CLASS_CONDITION_NAMES,
                )
                or type(self.tau_local) is not int
                or self.tau_local <= 0
                or type(self.tau_global) is not int
                or self.tau_global <= 0
                or type(self.history_maxlen) is not int
                or self.history_maxlen <= 0
                or type(self.required_history_length) is not int
                or self.required_history_length <= 0
                or type(self.finite_core_candidate_count) is not int
                or type(self.finite_core_admissible_count) is not int
                or type(self.finite_core_maximizer_amplitudes) is not tuple
                or len(self.finite_core_maximizer_amplitudes) != 3
                or not all(
                    type(value) is int
                    for value in self.finite_core_maximizer_amplitudes
                )
                or not all(
                    type(object.__getattribute__(self, name)) is Fraction
                    for name in exact_fraction_fields
                )
                or type(self.clip_mode) is not str
                or type(configuration.tau_local) is not int
                or configuration.tau_local <= 0
                or type(configuration.tau_global) is not int
                or configuration.tau_global <= 0
                or type(configuration.history_maxlen) is not int
                or configuration.history_maxlen <= 0
                or type(configuration.alpha) is not float
                or type(configuration.alpha_source) is not str
                or not configuration.alpha_source
                or type(configuration.epi_min) is not float
                or type(configuration.epi_max) is not float
                or type(configuration.clip_mode) is not str
            ):
                return False
            canonical = materialize_delayed_remesh_configuration(
                tau_local=self.tau_local,
                tau_global=self.tau_global,
                alpha=0.5,
                alpha_source=configuration.alpha_source,
                epi_min=-configuration.epi_max,
                epi_max=configuration.epi_max,
                clip_mode="hard",
            )
            expected_remesh = certify_uniform_remesh_history_stability(
                alpha=Fraction(1, 2),
                tau_local=self.tau_local,
                tau_global=self.tau_global,
            )
            expected_witness = _half_class_sharpness_witness(canonical)
            expected_conditions = _half_class_conditions(
                node_order,
                metric,
                canonical,
                expected_remesh,
                expected_witness,
            )
            return bool(
                configuration == canonical
                and self.history_maxlen == canonical.history_maxlen
                and self.required_history_length
                == max(self.tau_local, self.tau_global) + 1
                and self.epi_bound == Fraction.from_float(canonical.epi_max)
                and self.epi_bound
                >= Fraction.from_float(_MIN_HALF_CLASS_BOUND)
                and self.epi_min == -self.epi_bound
                and self.epi_max == self.epi_bound
                and self.alpha == Fraction(1, 2)
                and self.clip_mode == "hard"
                and self.exact_uniform_relative_defect_upper_bound
                == _HALF_CLASS_ETA
                and self.exact_tail_error_linear_coefficient
                == _HALF_CLASS_TAIL_ERROR_LINEAR_COEFFICIENT
                and self.exact_tail_error_absolute_coefficient
                == _HALF_CLASS_TAIL_ERROR_ABSOLUTE_COEFFICIENT
                and self.exact_tail_error_ratio_at_eleven_subnormals
                == _HALF_CLASS_TAIL_ERROR_RATIO
                and self.exact_tail_relative_defect_upper_bound
                == _HALF_CLASS_TAIL_RELATIVE_DEFECT_BOUND
                and self.finite_core_candidate_count
                == _HALF_CLASS_CORE_CANDIDATE_COUNT
                and self.finite_core_admissible_count
                == _HALF_CLASS_CORE_ADMISSIBLE_COUNT
                and self.finite_core_maximizer_amplitudes
                == _HALF_CLASS_CORE_MAXIMIZER
                and self.exact_strict_schedule_gain_threshold
                == _HALF_CLASS_STRICT_Q_THRESHOLD
                and self.example_schedule_energy_gain_upper_bound
                == _HALF_CLASS_EXAMPLE_Q
                and self.exact_example_effective_head_energy_gain_upper_bound
                == _HALF_CLASS_EXAMPLE_Q_EFF
                and self.exact_example_normalized_block_margin_lower_bound
                == _HALF_CLASS_EXAMPLE_MARGIN
                and remesh.stability_certificate_certified
                and witness.pair_relative_defect_observation_certified
                and proof_stamps_are_identical(
                    object.__getattribute__(remesh, "_proof_stamp"),
                    object.__getattribute__(expected_remesh, "_proof_stamp"),
                )
                and proof_stamps_are_identical(
                    object.__getattribute__(witness, "_proof_stamp"),
                    object.__getattribute__(expected_witness, "_proof_stamp"),
                )
                and conditions == expected_conditions
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _HALF_CLASS_SCOPE

    @property
    def half_alpha_antisymmetric_hard_clip_class_certificate_certified(
        self,
    ) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(passed for _name, passed in self.conditions)
        )

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return (
                "half_alpha_antisymmetric_hard_clip_remesh_class_"
                "proof_fields_intact",
            )
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def binary64_antisymmetry_preserved_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    @property
    def hard_clip_preserves_antisymmetric_interval_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    @property
    def uniform_binary64_relative_defect_bound_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    @property
    def uniform_relative_defect_bound_is_sharp_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    @property
    def remesh_class_forward_invariant_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    @property
    def strict_schedule_composition_threshold_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    @property
    def example_schedule_composition_certified(self) -> bool:
        return self.half_alpha_antisymmetric_hard_clip_class_certificate_certified

    def represented_history_belongs_to_class(self, history: Any) -> bool:
        """Check one chronological P2 history against the closed class."""

        if not self.half_alpha_antisymmetric_hard_clip_class_certificate_certified:
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
        lower = self.configuration.epi_min
        upper = self.configuration.epi_max
        for row in rows:
            if type(row) is tuple:
                values = tuple(tuple.__iter__(row))
            elif type(row) is list:
                values = tuple(list.__iter__(row))
            else:
                return False
            if len(values) != 2:
                return False
            if any(
                type(value) is not float
                or not math.isfinite(value)
                or value < lower
                or value > upper
                for value in values
            ):
                return False
            if values[1] != -values[0]:
                return False
        return True

    def certify_schedule_relative_defect_stability(
        self,
        schedule_energy_gain_upper_bound: Real,
    ) -> UniformRemeshScheduleRelativeDefectStabilityCertificate:
        """Compose this ``eta`` with one declared common schedule gain."""

        if not self.half_alpha_antisymmetric_hard_clip_class_certificate_certified:
            raise TNFRValueError(
                "half-alpha REMESH class certificate is unsealed or inconsistent"
            )
        policy = certify_uniform_remesh_schedule_policy_stability(
            self.remesh_certificate,
            schedule_energy_gain_upper_bound,
        )
        return certify_uniform_remesh_schedule_relative_defect_stability(
            policy,
            self.exact_uniform_relative_defect_upper_bound,
        )

    @property
    def schedule_family_certificate_certified(self) -> bool:
        return False

    @property
    def repeated_binary64_stability_certified(self) -> bool:
        return False

    @property
    def runtime_forward_invariance_certified(self) -> bool:
        return False

    @property
    def binary64_runtime_stability_certified(self) -> bool:
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


def certify_half_alpha_antisymmetric_hard_clip_remesh_class(
    nodes_pair: Iterable[Hashable],
    metric_weights: Mapping[Hashable, Real] | Sequence[Real] | None = None,
    *,
    tau_local: int,
    tau_global: int,
    epi_bound: Real,
) -> UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate:
    """Certify the sharp ``alpha=1/2`` antisymmetric P2 REMESH class."""

    try:
        node_order = _node_order(nodes_pair)
        if len(node_order) != 2:
            raise TNFRValueError(
                "nodes_pair must contain exactly two unique identifiers"
            )
        typed_node_order = (node_order[0], node_order[1])
        metric_raw = _normalized_metric(metric_weights, typed_node_order)
        typed_metric = (metric_raw[0], metric_raw[1])
        bound = _positive_half_class_bound(epi_bound)
        configuration = materialize_delayed_remesh_configuration(
            tau_local=tau_local,
            tau_global=tau_global,
            alpha=0.5,
            alpha_source=(
                "half_alpha_antisymmetric_hard_clip_class_certificate"
            ),
            epi_min=-bound,
            epi_max=bound,
            clip_mode="hard",
        )
    except TNFRValueError:
        raise
    except BaseException as exc:
        raise TNFRValueError(
            "invalid half-alpha antisymmetric REMESH class inputs"
        ) from exc
    remesh = certify_uniform_remesh_history_stability(
        alpha=Fraction(1, 2),
        tau_local=configuration.tau_local,
        tau_global=configuration.tau_global,
    )
    witness = _half_class_sharpness_witness(configuration)
    conditions = _half_class_conditions(
        typed_node_order,
        typed_metric,
        configuration,
        remesh,
        witness,
    )
    if not all(passed for _name, passed in conditions):
        failed = tuple(name for name, passed in conditions if not passed)
        raise TNFRValueError(
            f"half-alpha antisymmetric REMESH class proof failed: {failed}"
        )
    bound_q = Fraction.from_float(configuration.epi_max)
    value = UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate(
        node_order=typed_node_order,
        exact_normalized_metric=typed_metric,
        configuration=configuration,
        tau_local=configuration.tau_local,
        tau_global=configuration.tau_global,
        history_maxlen=configuration.history_maxlen,
        required_history_length=max(
            configuration.tau_local,
            configuration.tau_global,
        )
        + 1,
        epi_bound=bound_q,
        epi_min=-bound_q,
        epi_max=bound_q,
        alpha=Fraction(1, 2),
        clip_mode="hard",
        remesh_certificate=remesh,
        exact_uniform_relative_defect_upper_bound=_HALF_CLASS_ETA,
        exact_tail_error_linear_coefficient=(
            _HALF_CLASS_TAIL_ERROR_LINEAR_COEFFICIENT
        ),
        exact_tail_error_absolute_coefficient=(
            _HALF_CLASS_TAIL_ERROR_ABSOLUTE_COEFFICIENT
        ),
        exact_tail_error_ratio_at_eleven_subnormals=(
            _HALF_CLASS_TAIL_ERROR_RATIO
        ),
        exact_tail_relative_defect_upper_bound=(
            _HALF_CLASS_TAIL_RELATIVE_DEFECT_BOUND
        ),
        finite_core_candidate_count=_HALF_CLASS_CORE_CANDIDATE_COUNT,
        finite_core_admissible_count=_HALF_CLASS_CORE_ADMISSIBLE_COUNT,
        finite_core_maximizer_amplitudes=_HALF_CLASS_CORE_MAXIMIZER,
        exact_strict_schedule_gain_threshold=_HALF_CLASS_STRICT_Q_THRESHOLD,
        example_schedule_energy_gain_upper_bound=_HALF_CLASS_EXAMPLE_Q,
        exact_example_effective_head_energy_gain_upper_bound=(
            _HALF_CLASS_EXAMPLE_Q_EFF
        ),
        exact_example_normalized_block_margin_lower_bound=(
            _HALF_CLASS_EXAMPLE_MARGIN
        ),
        sharpness_witness=witness,
        conditions=conditions,
    )
    result = _seal(value, type(value), _HALF_CLASS_PROOF_VERSION)
    if not result.half_alpha_antisymmetric_hard_clip_class_certificate_certified:
        raise RuntimeError(
            "constructed half-alpha antisymmetric REMESH class proof is "
            "inconsistent"
        )
    return result
