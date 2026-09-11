"""Continuous-time relaxation across one declared operator-flow interval.

This module maps a physical flow duration to the existing frozen,
connected, symmetric pure-EPI diffusion theorem.  It does not inspect solver
steps, derive another Laplacian, execute operators, or alter grammar policy.
The certified rate concerns disagreement energy, so ``target_fraction`` is an
energy fraction under the fixed certificate's weighted metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Real
from typing import Any

from .._exact_time import (
    exact_log_bounds,
    exp_upper_float,
    finite_represented_real,
    fraction_upper_float,
    nonnegative_represented_time,
    represented_fraction,
)
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    verify_heterogeneous_diffusion_stability,
)

__all__ = (
    "ContinuousRelaxationDurationDiagnostic",
    "diagnose_continuous_relaxation_duration",
)


_DEFAULT_TARGET_FRACTION = 1.0 / (math.pi + 1.0)
_SCOPE = (
    "frozen connected symmetric positive-capacity continuous-time pure-EPI "
    "disagreement-energy relaxation; rational log/exp enclosures drive every "
    "certified duration conclusion, while libm and spectral values remain "
    "unsealed estimates"
)


def _optional_float_signature(value: float | None) -> str | None:
    """Encode an optional binary64 result for an in-memory proof stamp."""

    return None if value is None else value.hex()


def _duration_proof_stamp(
    *,
    nodes: tuple[Any, ...],
    flow_duration: float,
    exact_flow_duration: Fraction,
    target_fraction: float,
    exact_target_fraction: Fraction,
    tolerance: float,
    certificate_available: bool,
    theorem_certified: bool,
    exact_rate: Fraction | None,
    certified_rate: float | None,
    log_target_upper: Fraction | None,
    exact_required_duration_upper: Fraction | None,
    required_duration: float | None,
    certified_decay_factor_upper: float | None,
    reaches_target: bool | None,
    abstained: bool,
    abstention_reason: str | None,
    abstention_detail: str | None,
    source_certificate_stamp: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Bind every field that can promote the continuous-time conclusion."""

    return (
        "continuous_relaxation_duration_v2",
        structural_proof_signature(nodes),
        flow_duration.hex(),
        exact_flow_duration,
        target_fraction.hex(),
        exact_target_fraction,
        tolerance.hex(),
        certificate_available,
        theorem_certified,
        exact_rate,
        _optional_float_signature(certified_rate),
        log_target_upper,
        exact_required_duration_upper,
        _optional_float_signature(required_duration),
        _optional_float_signature(certified_decay_factor_upper),
        reaches_target,
        abstained,
        abstention_reason,
        abstention_detail,
        source_certificate_stamp,
    )


def _require_float(
    value: Any,
    label: str,
    *,
    allow_infinity: bool = False,
    nonnegative: bool = False,
) -> None:
    """Validate one exact binary64 payload field without coercion."""

    if type(value) is not float:
        raise TypeError(f"{label} must be a binary64 float")
    if math.isnan(value) or (math.isinf(value) and not allow_infinity):
        raise ValueError(f"{label} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{label} must use canonical positive zero")


def _require_optional_float(
    value: Any,
    label: str,
    *,
    allow_infinity: bool = False,
    nonnegative: bool = False,
) -> None:
    if value is not None:
        _require_float(
            value,
            label,
            allow_infinity=allow_infinity,
            nonnegative=nonnegative,
        )


def _require_optional_fraction(value: Any, label: str) -> None:
    if value is not None and type(value) is not Fraction:
        raise TypeError(f"{label} must be an exact Fraction or None")


def _unsealed_spectral_rate(value: Any) -> float | None:
    """Retain only a well-formed binary64 spectral estimate."""

    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        return None
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationDurationDiagnostic:
    """Fixed-flow energy relaxation over one physical duration.

    ``exact_energy_decay_rate_lower_bound`` is the rational theorem quantity.
    ``certified_energy_decay_rate_lower_bound`` is its downward-rounded
    operational binary64 value. ``spectral_energy_decay_rate_estimate`` remains
    the existing eigensolver diagnostic and never drives the target decision.
    A positive exact rate may survive when the operational display underflows
    to zero; in that case the libm estimates are unavailable.

    ``exact_log_target_upper_bound`` encloses ``log(1 / target)`` from above.
    The certified decision compares it rationally with the exact theorem-rate
    lower bound times ``exact_flow_duration``. ``required_flow_duration`` is
    the upward-rounded display of that sufficient rational threshold, and
    ``certified_decay_factor_upper_bound`` is a rigorous exponential upper
    bound. The two fields ending in ``_estimate`` use ordinary libm and never
    drive a proof. No field contains a numerical integration timestep.
    """

    nodes: tuple[Any, ...]
    flow_duration: float
    exact_flow_duration: Fraction
    target_fraction: float
    exact_target_fraction: Fraction
    tolerance: float
    fixed_flow_certificate_available: bool
    fixed_flow_theorem_certified: bool
    exact_energy_decay_rate_lower_bound: Fraction | None
    certified_energy_decay_rate_lower_bound: float | None
    spectral_energy_decay_rate_estimate: float | None
    exact_log_target_upper_bound: Fraction | None
    exact_required_flow_duration_upper_bound: Fraction | None
    required_flow_duration: float | None
    required_flow_duration_estimate: float | None
    certified_decay_factor_upper_bound: float | None
    decay_factor_estimate: float | None
    duration_reaches_target: bool | None
    abstained: bool
    abstention_reason: str | None
    abstention_detail: str | None
    solver_timestep_independent: bool = field(default=True, init=False)
    spectral_estimate_is_proof_input: bool = field(default=False, init=False)
    spectral_estimate_provenance: str = field(
        default="unsealed_source_eigensolver_diagnostic", init=False
    )
    scope: str = field(default=_SCOPE, init=False)
    _source_certificate_stamp: tuple[Any, ...] = field(
        default=(), repr=False, compare=False
    )
    _proof_stamp: tuple[Any, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Reject type coercions and mutable replacements in sealed fields."""

        if type(self.nodes) is not tuple:
            raise TypeError("nodes must be an immutable tuple")
        represented_duration = represented_fraction(
            self.flow_duration, "flow_duration"
        )
        if represented_duration < 0:
            raise ValueError("flow_duration must be nonnegative")
        _require_float(self.target_fraction, "target_fraction")
        _require_float(self.tolerance, "tolerance")
        if type(self.exact_flow_duration) is not Fraction:
            raise TypeError("exact_flow_duration must be an exact Fraction")
        if type(self.exact_target_fraction) is not Fraction:
            raise TypeError("exact_target_fraction must be an exact Fraction")
        if self.exact_flow_duration != represented_duration:
            raise ValueError("exact_flow_duration must match flow_duration")
        if self.exact_target_fraction != Fraction.from_float(
            self.target_fraction
        ):
            raise ValueError("exact_target_fraction must match target_fraction")
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("target_fraction must lie strictly between zero and one")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        boolean_fields = (
            (self.fixed_flow_certificate_available, "certificate availability"),
            (self.fixed_flow_theorem_certified, "theorem certification"),
            (self.abstained, "abstained"),
        )
        for value, label in boolean_fields:
            if type(value) is not bool:
                raise TypeError(f"{label} must be a boolean")
        if self.duration_reaches_target is not None:
            if type(self.duration_reaches_target) is not bool:
                raise TypeError("duration_reaches_target must be a boolean or None")
        _require_optional_fraction(
            self.exact_energy_decay_rate_lower_bound,
            "exact_energy_decay_rate_lower_bound",
        )
        _require_optional_fraction(
            self.exact_log_target_upper_bound,
            "exact_log_target_upper_bound",
        )
        _require_optional_fraction(
            self.exact_required_flow_duration_upper_bound,
            "exact_required_flow_duration_upper_bound",
        )
        _require_optional_float(
            self.certified_energy_decay_rate_lower_bound,
            "certified_energy_decay_rate_lower_bound",
            nonnegative=True,
        )
        _require_optional_float(
            self.spectral_energy_decay_rate_estimate,
            "spectral_energy_decay_rate_estimate",
            nonnegative=True,
        )
        _require_optional_float(
            self.required_flow_duration,
            "required_flow_duration",
            allow_infinity=True,
            nonnegative=True,
        )
        _require_optional_float(
            self.required_flow_duration_estimate,
            "required_flow_duration_estimate",
            allow_infinity=True,
            nonnegative=True,
        )
        _require_optional_float(
            self.certified_decay_factor_upper_bound,
            "certified_decay_factor_upper_bound",
            nonnegative=True,
        )
        _require_optional_float(
            self.decay_factor_estimate,
            "decay_factor_estimate",
            nonnegative=True,
        )
        if self.abstention_reason is not None:
            if type(self.abstention_reason) is not str:
                raise TypeError("abstention_reason must be a string or None")
        if self.abstention_detail is not None:
            if type(self.abstention_detail) is not str:
                raise TypeError("abstention_detail must be a string or None")
        if type(self._source_certificate_stamp) is not tuple:
            raise TypeError("source certificate stamp must be an immutable tuple")
        if type(self._proof_stamp) is not tuple:
            raise TypeError("proof stamp must be an immutable tuple")

    def _proof_fields_are_intact(self) -> bool:
        """Check conclusion fields; unsealed estimates are excluded."""

        try:
            expected = _duration_proof_stamp(
                nodes=self.nodes,
                flow_duration=self.flow_duration,
                exact_flow_duration=self.exact_flow_duration,
                target_fraction=self.target_fraction,
                exact_target_fraction=self.exact_target_fraction,
                tolerance=self.tolerance,
                certificate_available=self.fixed_flow_certificate_available,
                theorem_certified=self.fixed_flow_theorem_certified,
                exact_rate=self.exact_energy_decay_rate_lower_bound,
                certified_rate=self.certified_energy_decay_rate_lower_bound,
                log_target_upper=self.exact_log_target_upper_bound,
                exact_required_duration_upper=(
                    self.exact_required_flow_duration_upper_bound
                ),
                required_duration=self.required_flow_duration,
                certified_decay_factor_upper=(
                    self.certified_decay_factor_upper_bound
                ),
                reaches_target=self.duration_reaches_target,
                abstained=self.abstained,
                abstention_reason=self.abstention_reason,
                abstention_detail=self.abstention_detail,
                source_certificate_stamp=self._source_certificate_stamp,
            )
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        return proof_stamps_are_identical(observed, expected)


def _diagnostic(
    *,
    nodes: tuple[Any, ...],
    flow_duration: float,
    exact_flow_duration: Fraction,
    target_fraction: float,
    exact_target_fraction: Fraction,
    tolerance: float,
    certificate_available: bool,
    theorem_certified: bool,
    exact_rate: Fraction | None = None,
    certified_rate: float | None = None,
    spectral_rate: float | None = None,
    log_target_upper: Fraction | None = None,
    exact_required_duration_upper: Fraction | None = None,
    required_duration: float | None = None,
    required_duration_estimate: float | None = None,
    certified_decay_factor_upper: float | None = None,
    decay_factor_estimate: float | None = None,
    reaches_target: bool | None = None,
    abstention_reason: str | None = None,
    abstention_detail: str | None = None,
    source_certificate_stamp: tuple[Any, ...] = (),
) -> ContinuousRelaxationDurationDiagnostic:
    """Construct one sealed success or abstention record."""

    abstained = abstention_reason is not None
    stamp = _duration_proof_stamp(
        nodes=nodes,
        flow_duration=flow_duration,
        exact_flow_duration=exact_flow_duration,
        target_fraction=target_fraction,
        exact_target_fraction=exact_target_fraction,
        tolerance=tolerance,
        certificate_available=certificate_available,
        theorem_certified=theorem_certified,
        exact_rate=exact_rate,
        certified_rate=certified_rate,
        log_target_upper=log_target_upper,
        exact_required_duration_upper=exact_required_duration_upper,
        required_duration=required_duration,
        certified_decay_factor_upper=certified_decay_factor_upper,
        reaches_target=reaches_target,
        abstained=abstained,
        abstention_reason=abstention_reason,
        abstention_detail=abstention_detail,
        source_certificate_stamp=source_certificate_stamp,
    )
    return ContinuousRelaxationDurationDiagnostic(
        nodes=nodes,
        flow_duration=flow_duration,
        exact_flow_duration=exact_flow_duration,
        target_fraction=target_fraction,
        exact_target_fraction=exact_target_fraction,
        tolerance=tolerance,
        fixed_flow_certificate_available=certificate_available,
        fixed_flow_theorem_certified=theorem_certified,
        exact_energy_decay_rate_lower_bound=exact_rate,
        certified_energy_decay_rate_lower_bound=certified_rate,
        spectral_energy_decay_rate_estimate=spectral_rate,
        exact_log_target_upper_bound=log_target_upper,
        exact_required_flow_duration_upper_bound=(
            exact_required_duration_upper
        ),
        required_flow_duration=required_duration,
        required_flow_duration_estimate=required_duration_estimate,
        certified_decay_factor_upper_bound=certified_decay_factor_upper,
        decay_factor_estimate=decay_factor_estimate,
        duration_reaches_target=reaches_target,
        abstained=abstained,
        abstention_reason=abstention_reason,
        abstention_detail=abstention_detail,
        _source_certificate_stamp=source_certificate_stamp,
        _proof_stamp=stamp,
    )


def diagnose_continuous_relaxation_duration(
    graph: Any,
    *,
    flow_duration: Real,
    target_fraction: Real = _DEFAULT_TARGET_FRACTION,
    tolerance: Real = 1e-10,
) -> ContinuousRelaxationDurationDiagnostic:
    """Assess a fixed pure-EPI continuous flow over ``flow_duration``.

    A domain failure from the existing fixed-flow verifier is returned as an
    explicit abstention. Invalid scalar controls still raise. The result uses
    only the verifier's proved rate and existing spectral estimate; it does not
    derive transport data independently.
    """

    duration, exact_duration = nonnegative_represented_time(
        flow_duration, "flow_duration"
    )
    target, exact_target = finite_represented_real(
        target_fraction, "target_fraction"
    )
    tol, _ = finite_represented_real(tolerance, "tolerance")
    if not 0.0 < target < 1.0:
        raise ValueError("target_fraction must lie strictly between zero and one")
    if tol <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    try:
        certificate = verify_heterogeneous_diffusion_stability(
            graph, tolerance=tol
        )
    except ValueError as exc:
        try:
            nodes = tuple(graph.nodes())
        except (AttributeError, TypeError):
            nodes = ()
        return _diagnostic(
            nodes=nodes,
            flow_duration=duration,
            exact_flow_duration=exact_duration,
            target_fraction=target,
            exact_target_fraction=exact_target,
            tolerance=tol,
            certificate_available=False,
            theorem_certified=False,
            abstention_reason="fixed_flow_domain_error",
            abstention_detail=str(exc),
        )

    if type(certificate) is not HeterogeneousDiffusionStabilityCertificate:
        try:
            nodes = tuple(graph.nodes())
        except (AttributeError, TypeError):
            nodes = ()
        return _diagnostic(
            nodes=nodes,
            flow_duration=duration,
            exact_flow_duration=exact_duration,
            target_fraction=target,
            exact_target_fraction=exact_target,
            tolerance=tol,
            certificate_available=True,
            theorem_certified=False,
            abstention_reason="fixed_flow_certificate_integrity_failed",
        )

    raw_nodes = getattr(certificate, "nodes", ())
    nodes = raw_nodes if type(raw_nodes) is tuple else ()
    raw_source_stamp = getattr(certificate, "_proof_stamp", ())
    source_stamp = (
        raw_source_stamp if type(raw_source_stamp) is tuple else ()
    )
    spectral_rate = _unsealed_spectral_rate(
        getattr(certificate, "exponential_rate", None)
    )
    try:
        source_intact = certificate._proof_fields_are_intact()
    except BaseException:
        source_intact = False
    if type(source_intact) is not bool or not source_intact:
        return _diagnostic(
            nodes=nodes,
            flow_duration=duration,
            exact_flow_duration=exact_duration,
            target_fraction=target,
            exact_target_fraction=exact_target,
            tolerance=tol,
            certificate_available=True,
            theorem_certified=False,
            spectral_rate=spectral_rate,
            source_certificate_stamp=source_stamp,
            abstention_reason="fixed_flow_certificate_integrity_failed",
        )

    nodes = certificate.nodes
    exact_rate = 2 * certificate.exact_quotient_gap_lower_bound
    certified_rate = certificate.certified_exponential_rate_lower_bound
    if certified_rate == 0.0:
        certified_rate = 0.0
    source_stamp = certificate._proof_stamp
    common = {
        "nodes": nodes,
        "flow_duration": duration,
        "exact_flow_duration": exact_duration,
        "target_fraction": target,
        "exact_target_fraction": exact_target,
        "tolerance": tol,
        "certificate_available": True,
        "exact_rate": exact_rate,
        "certified_rate": certified_rate,
        "spectral_rate": spectral_rate,
        "source_certificate_stamp": source_stamp,
    }
    if not certificate.exact_uniform_fixed_point_preservation:
        return _diagnostic(
            **common,
            theorem_certified=False,
            abstention_reason="fixed_flow_theorem_not_certified",
            abstention_detail="represented uniform fixed points are not exact",
        )
    if exact_rate <= 0:
        return _diagnostic(
            **common,
            theorem_certified=False,
            abstention_reason="exact_rate_not_positive",
        )
    if not math.isfinite(certified_rate) or certified_rate < 0.0:
        return _diagnostic(
            **common,
            theorem_certified=False,
            abstention_reason="certified_rate_display_invalid",
        )

    log_target_lower, _ = exact_log_bounds(exact_target)
    log_target_upper = -log_target_lower
    if log_target_upper <= 0 or exact_rate <= 0:
        return _diagnostic(
            **common,
            theorem_certified=False,
            abstention_reason="exact_rate_or_log_bound_not_positive",
        )
    exact_required_duration_upper = log_target_upper / exact_rate
    required_duration = fraction_upper_float(exact_required_duration_upper)
    reaches_target = bool(
        exact_rate * exact_duration >= log_target_upper
    )
    certified_decay_factor_upper = exp_upper_float(
        -exact_rate * exact_duration
    )
    if reaches_target:
        # The rational log comparison independently proves the target itself
        # is an upper bound and can tighten the separately enclosed exponential.
        certified_decay_factor_upper = min(
            certified_decay_factor_upper, target
        )
    required_duration_estimate = None
    decay_factor_estimate = None
    if certified_rate > 0.0:
        logarithmic_target_estimate = -math.log(target)
        required_duration_estimate = (
            logarithmic_target_estimate / certified_rate
        )
        decay_exponent_estimate = certified_rate * duration
        decay_factor_estimate = math.exp(-decay_exponent_estimate)
    return _diagnostic(
        **common,
        theorem_certified=True,
        log_target_upper=log_target_upper,
        exact_required_duration_upper=exact_required_duration_upper,
        required_duration=required_duration,
        required_duration_estimate=required_duration_estimate,
        certified_decay_factor_upper=certified_decay_factor_upper,
        decay_factor_estimate=decay_factor_estimate,
        reaches_target=reaches_target,
    )
