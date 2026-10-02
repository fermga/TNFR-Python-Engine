"""Configured validation reports for scalar, string and array inputs.

This adapter reuses shared represented-real and coherence admission. Additional
bounds (including the default frequency cap) are configured application policy,
not laws derived from the nodal equation. Operator, trajectory and scientific
admission remain with their corresponding owners.

Optional memoization retains typed immutable inputs, consumed policy and detached
results; it never authenticates a nodal trajectory or operator execution.
"""

from __future__ import annotations

import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable

from .._coherence_validation import validate_structural_coherence
from .._exact_time import finite_represented_real

# Unified configuration integration
from ..config import get_config
from ..errors import TNFRValueError
from ..errors.contextual import (  # noqa: F401 – re-exported via __init__
    TNFRSecurityError,
)
from ..mathematics.unified_numerical import np

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for all validation errors."""


@dataclass
class ValidationResult:
    """Result of validation operation with detailed feedback."""

    is_valid: bool
    error_messages: list[str]
    warnings: list[str]
    validated_value: Any = None
    validation_metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.validation_metadata is None:
            self.validation_metadata = {}


@dataclass
class ValidationConfig:
    """Configuration for unified validation system."""

    # Validation strictness
    strict_mode: bool = True
    enable_warnings: bool = True

    # Configured report bounds; the frequency cap is not a nodal law.
    max_structural_frequency: float = 1000.0  # Hz_str
    min_structural_frequency: float = 0.0
    max_phase_value: float = 2 * math.pi
    min_phase_value: float = 0.0

    # Coherence and stability bounds
    min_coherence: float = 0.0
    max_coherence: float = 1.0
    min_sense_index: float = 0.0
    max_sense_index: float = float("inf")

    # Security validation
    enable_security_checks: bool = True
    max_string_length: int = 1000
    forbidden_patterns: list[str] = None

    # Performance settings
    enable_caching: bool = True
    cache_validation_results: bool = True

    def __post_init__(self):
        """Initialize default forbidden patterns."""
        if self.forbidden_patterns is None:
            self.forbidden_patterns = [
                r"<script",
                r"javascript:",
                r"eval\(",
                r"exec\(",
                r"\$\{",
                r"`.*`",
            ]


class TNFRValidationError(TNFRValueError):
    """Unified validation error for TNFR structural constraints."""

    def __init__(
        self,
        message: str,
        field_name: str = None,
        validation_context: dict[str, Any] = None,
        suggestion: str = None,
    ):
        context = validation_context or {}
        if field_name:
            context["field"] = field_name

        super().__init__(message=message, context=context, suggestion=suggestion)
        self.field_name = field_name
        self.validation_context = context


class TNFRUnifiedValidationSystem:
    """Report input admission under shared scalar contracts and supplied policy.

    Usage:
        # Configured input-reporting entry point
        validator = TNFRUnifiedValidationSystem()

        # Structural parameter validation
        result = validator.validate_structural_frequency(0.5)
        assert result.is_valid

        # Security validation
        result = validator.validate_string_input("user_input")

        # Batch validation
        results = validator.validate_multiple({
            "vf": 1.2,
            "phase": 3.14,
            "coherence": 0.85
        })

    This class does not replace live operator preconditions or trajectory
    certificates. A successful input report has only its stated local scope.
    """

    def __init__(self, config: ValidationConfig | None = None):
        """Initialize unified validation system."""
        self.config = config or ValidationConfig()

        # Validation cache for performance
        self._validation_cache: dict[tuple[Any, ...], ValidationResult] = {}
        self._cache_stats = {"hits": 0, "misses": 0}

        # Global configuration integration
        self.global_config = get_config()

        self._pattern_sources: tuple[str, ...] | None = None
        self._compiled_security_patterns: list[re.Pattern[str]] = []
        self._security_patterns()

        logger.info(f"Initialized unified validation system with config: {self.config}")

    def _cache_key(self, *parts: Any) -> tuple[Any, ...] | None:
        """Cache only exact immutable builtins, with their types and policy."""
        if not (self.config.enable_caching and self.config.cache_validation_results):
            return None
        if any(type(part) not in (str, int, float, bool, type(None)) for part in parts):
            return None
        if any(type(part) is float and not math.isfinite(part) for part in parts):
            return None
        return tuple((type(part), part) for part in parts)

    def _cached_result(self, key: tuple[Any, ...] | None) -> ValidationResult | None:
        if key is not None and key in self._validation_cache:
            self._cache_stats["hits"] += 1
            return deepcopy(self._validation_cache[key])
        self._cache_stats["misses"] += 1
        return None

    def _store_result(
        self, key: tuple[Any, ...] | None, result: ValidationResult
    ) -> ValidationResult:
        if key is not None:
            self._validation_cache[key] = deepcopy(result)
        return result

    def _security_patterns(self) -> tuple[str, ...]:
        """Compile the current pattern policy without retaining stale rules."""
        sources = tuple(self.config.forbidden_patterns)
        if any(not isinstance(pattern, str) for pattern in sources):
            raise TypeError("forbidden_patterns must contain strings")
        if sources != self._pattern_sources:
            compiled = [re.compile(pattern, re.IGNORECASE) for pattern in sources]
            self._compiled_security_patterns = compiled
            self._pattern_sources = sources
        return sources

    def _policy_bounds(
        self, lower_name: str, upper_name: str, *, allow_unbounded: bool = False
    ) -> tuple[Any, Any]:
        """Admit the live bounds while preserving their original ordering."""
        lower = getattr(self.config, lower_name)
        upper = getattr(self.config, upper_name)
        for bound, name, open_end in (
            (lower, lower_name, -math.inf),
            (upper, upper_name, math.inf),
        ):
            if allow_unbounded and isinstance(bound, Real) and bound == open_end:
                continue
            finite_represented_real(bound, name)
        if lower > upper:
            raise ValueError(f"{lower_name} must not exceed {upper_name}")
        return lower, upper

    def validate_structural_frequency(
        self, vf: float | int, field_name: str = "vf"
    ) -> ValidationResult:
        """Validate structural frequency (νf) parameter.

        Reuses represented-real admission; the configured maximum is a policy.

        Parameters
        ----------
        vf : float or int
            Structural frequency value in Hz_str units
        field_name : str
            Name of the field being validated for error reporting

        Returns
        -------
        ValidationResult
            Validation result with detailed feedback
        """
        cache_key = self._cache_key(
            "vf",
            vf,
            field_name,
            self.config.min_structural_frequency,
            self.config.max_structural_frequency,
            self.config.strict_mode,
        )

        cached = self._cached_result(cache_key)
        if cached is not None:
            return cached

        errors = []
        warnings = []
        validated_value = vf

        try:
            lower, upper = self._policy_bounds(
                "min_structural_frequency",
                "max_structural_frequency",
                allow_unbounded=True,
            )
            if lower < 0:
                raise ValueError("min_structural_frequency must be nonnegative")
            validated_value, _ = finite_represented_real(vf, field_name)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if vf < lower:
                errors.append(f"{field_name} must be >= {lower}, got {validated_value}")

            if vf > upper:
                if self.config.strict_mode:
                    errors.append(
                        f"{field_name} exceeds maximum {upper}, got {validated_value}"
                    )
                else:
                    warnings.append(
                        f"{field_name} is very large ({validated_value}), consider checking units"
                    )

        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={
                "field_type": "structural_frequency",
                "units": "Hz_str",
            },
        )

        return self._store_result(cache_key, result)

    def validate_phase_value(
        self, phase: float | int, field_name: str = "phase", normalize: bool = True
    ) -> ValidationResult:
        """Validate phase (φ/θ) parameter.

        Reuses represented-real admission before optional circular normalization.

        Parameters
        ----------
        phase : float or int
            Phase value in radians
        field_name : str
            Name of the field being validated
        normalize : bool
            Whether to normalize phase to [0, 2π) range

        Returns
        -------
        ValidationResult
            Validation result with normalized phase value
        """
        cache_key = self._cache_key(
            "phase",
            phase,
            field_name,
            normalize,
            *(
                ()
                if normalize is True
                else (self.config.min_phase_value, self.config.max_phase_value)
            ),
        )

        cached = self._cached_result(cache_key)
        if cached is not None:
            return cached

        errors = []
        warnings = []
        validated_value = phase

        try:
            if not isinstance(normalize, bool):
                raise TypeError("normalize must be a boolean")
            validated_value, _ = finite_represented_real(phase, field_name)
            if not normalize:
                lower, upper = self._policy_bounds(
                    "min_phase_value", "max_phase_value", allow_unbounded=True
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if normalize:
                period = 2 * math.pi
                validated_value %= period
                # Negative near-zero inputs can round the modulo result to 2π.
                if validated_value == period:
                    validated_value = 0.0
            elif phase < lower or phase > upper:
                warnings.append(
                    f"{field_name} outside configured range [{lower}, {upper}], "
                    f"got {validated_value}"
                )

        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={
                "field_type": "phase",
                "units": "radians",
                "normalized": normalize,
            },
        )

        return self._store_result(cache_key, result)

    def validate_coherence(
        self, coherence: float | int, field_name: str = "coherence"
    ) -> ValidationResult:
        """Validate coherence C(t) parameter.

        Reuses the shared original-value and represented-value domain checks.
        """
        cache_key = self._cache_key(
            "coherence",
            coherence,
            field_name,
            self.config.min_coherence,
            self.config.max_coherence,
        )

        cached = self._cached_result(cache_key)
        if cached is not None:
            return cached

        errors = []
        warnings = []
        validated_value = coherence

        try:
            lower, upper = self._policy_bounds("min_coherence", "max_coherence")
            validate_structural_coherence(lower, name="min_coherence")
            validate_structural_coherence(upper, name="max_coherence")
            validated_value = validate_structural_coherence(coherence, name=field_name)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if coherence < lower:
                errors.append(
                    f"{field_name} must be >= {self.config.min_coherence}, "
                    f"got {validated_value}"
                )
            elif coherence > upper:
                errors.append(
                    f"{field_name} must be <= {self.config.max_coherence}, "
                    f"got {validated_value}"
                )

        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={
                "field_type": "coherence",
                "bounds": [self.config.min_coherence, self.config.max_coherence],
            },
        )

        return self._store_result(cache_key, result)

    def validate_string_input(
        self,
        input_string: str,
        field_name: str = "input",
        max_length: int | None = None,
    ) -> ValidationResult:
        """Validate string input with security checks.

        Applies the current configured length and regular-expression policy.
        This is an input filter, not a general injection-safety certificate.

        Parameters
        ----------
        input_string : str
            String to validate
        field_name : str
            Name of the field being validated
        max_length : int, optional
            Maximum allowed string length (uses config default if not provided)

        Returns
        -------
        ValidationResult
            Validation result with security assessment
        """
        if not self.config.enable_security_checks:
            return ValidationResult(
                is_valid=True,
                error_messages=[],
                warnings=[],
                validated_value=input_string,
            )

        max_len = self.config.max_string_length if max_length is None else max_length
        patterns = self._security_patterns()
        cache_key = self._cache_key(
            "string", input_string, field_name, max_len, *patterns
        )

        cached = self._cached_result(cache_key)
        if cached is not None:
            return cached

        errors = []
        warnings = []
        validated_value = input_string

        if (
            isinstance(max_len, bool)
            or not isinstance(max_len, Integral)
            or max_len < 0
        ):
            errors.append("max_length must be a nonnegative integer")
        elif not isinstance(input_string, str):
            errors.append(
                f"{field_name} must be a string, got {type(input_string).__name__}"
            )
        else:
            # Length validation
            if len(input_string) > max_len:
                errors.append(
                    f"{field_name} exceeds maximum length {max_len}, got {len(input_string)}"
                )

            # Security pattern validation
            for pattern in self._compiled_security_patterns:
                if pattern.search(input_string):
                    errors.append(
                        f"{field_name} contains potentially unsafe pattern: {pattern.pattern}"
                    )
                    break

            # Additional security checks
            if "<" in input_string and ">" in input_string:
                warnings.append(
                    f"{field_name} contains angle brackets, verify if intended"
                )

            if input_string.strip() != input_string:
                warnings.append(f"{field_name} has leading/trailing whitespace")

        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={
                "field_type": "string",
                "security_checked": True,
                "max_length": max_len,
            },
        )

        return self._store_result(cache_key, result)

    def validate_array_input(
        self,
        array: np.ndarray,
        field_name: str = "array",
        expected_shape: tuple | None = None,
        expected_dtype: type | None = None,
    ) -> ValidationResult:
        """Validate NumPy array input for TNFR operations.

        Parameters
        ----------
        array : np.ndarray
            Array to validate
        field_name : str
            Name of the field being validated
        expected_shape : tuple, optional
            Expected array shape
        expected_dtype : type, optional
            Expected array data type

        Returns
        -------
        ValidationResult
            Validation result with array information
        """
        errors = []
        warnings = []
        validated_value = array

        # type validation
        if not isinstance(array, np.ndarray):
            errors.append(
                f"{field_name} must be a NumPy array, got {type(array).__name__}"
            )
        else:
            # Shape validation
            if expected_shape is not None and array.shape != expected_shape:
                errors.append(
                    f"{field_name} shape mismatch: expected {expected_shape}, got {array.shape}"
                )

            # Data type validation
            if expected_dtype is not None and array.dtype != expected_dtype:
                warnings.append(
                    f"{field_name} dtype mismatch: expected {expected_dtype}, got {array.dtype}"
                )

            # Special values validation
            if np.any(np.isnan(array)):
                errors.append(f"{field_name} contains NaN values")
            elif np.any(np.isinf(array)):
                errors.append(f"{field_name} contains infinite values")

            # Size validation (prevent memory issues)
            if array.size > 1e8:  # 100M elements
                warnings.append(
                    f"{field_name} is very large ({array.size} elements), may cause memory issues"
                )

        result = ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            validated_value=validated_value,
            validation_metadata={
                "field_type": "array",
                "shape": array.shape if isinstance(array, np.ndarray) else None,
                "dtype": str(array.dtype) if isinstance(array, np.ndarray) else None,
            },
        )

        return result

    def validate_multiple(
        self,
        values: dict[str, Any],
        validation_rules: dict[str, Callable] | None = None,
    ) -> dict[str, ValidationResult]:
        """Validate multiple values with unified error handling.

        Parameters
        ----------
        values : dict
            Dictionary of field names to values to validate
        validation_rules : dict, optional
            Custom validation rules for specific fields

        Returns
        -------
        dict
            Dictionary of field names to validation results
        """
        results = {}

        # Default validation rules
        default_rules = {
            "vf": self.validate_structural_frequency,
            "phase": self.validate_phase_value,
            "coherence": self.validate_coherence,
            "structural_frequency": self.validate_structural_frequency,
            "phi": self.validate_phase_value,
            "theta": self.validate_phase_value,
        }

        # Merge with custom rules
        rules = {**default_rules, **(validation_rules or {})}

        for field_name, value in values.items():
            if field_name in rules:
                results[field_name] = rules[field_name](value, field_name)
            else:
                # Generic validation for unknown fields
                if isinstance(value, str):
                    results[field_name] = self.validate_string_input(value, field_name)
                elif isinstance(value, (int, float)):
                    # Basic number validation
                    results[field_name] = ValidationResult(
                        is_valid=(
                            not (math.isnan(value) or math.isinf(value))
                            if isinstance(value, float)
                            else True
                        ),
                        error_messages=(
                            ["Value cannot be NaN or infinite"]
                            if isinstance(value, float)
                            and (math.isnan(value) or math.isinf(value))
                            else []
                        ),
                        warnings=[],
                        validated_value=value,
                        validation_metadata={"field_type": "generic_number"},
                    )
                elif isinstance(value, np.ndarray):
                    results[field_name] = self.validate_array_input(value, field_name)
                else:
                    # Unknown type - basic validation
                    results[field_name] = ValidationResult(
                        is_valid=True,
                        error_messages=[],
                        warnings=[
                            f"Unknown field type {type(value).__name__} for {field_name}"
                        ],
                        validated_value=value,
                        validation_metadata={"field_type": "unknown"},
                    )

        return results

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get validation cache statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            (self._cache_stats["hits"] / total_requests * 100.0)
            if total_requests > 0
            else 0.0
        )

        return {
            **self._cache_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._validation_cache),
            "cache_enabled": (
                self.config.enable_caching and self.config.cache_validation_results
            ),
        }

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        logger.info("Cleared unified validation cache")


# ============================================================================
# PUBLIC API - Unified Validation Interface
# ============================================================================

# Global unified validation system instance
_unified_validation_system: TNFRUnifiedValidationSystem | None = None


def get_unified_validation_system(
    config: ValidationConfig | None = None,
) -> TNFRUnifiedValidationSystem:
    """Get or create global unified validation system.

    This singleton owns the configured input-reporting adapter. Live operators
    and scientific certificates retain their separate admission owners.

    Parameters
    ----------
    config : ValidationConfig, optional
        Configuration for system (only used on first call)

    Returns
    -------
    TNFRUnifiedValidationSystem
        Global unified validation system instance
    """
    global _unified_validation_system

    if _unified_validation_system is None:
        _unified_validation_system = TNFRUnifiedValidationSystem(config)
        logger.info("Created global unified validation system")

    return _unified_validation_system


# Convenience functions for direct validation operations
def validate_structural_frequency(
    vf: float | int, field_name: str = "vf"
) -> ValidationResult:
    """Validate structural frequency - convenience function."""
    return get_unified_validation_system().validate_structural_frequency(vf, field_name)


def validate_phase_value(
    phase: float | int, field_name: str = "phase"
) -> ValidationResult:
    """Validate phase value - convenience function."""
    return get_unified_validation_system().validate_phase_value(phase, field_name)


def validate_coherence(
    coherence: float | int, field_name: str = "coherence"
) -> ValidationResult:
    """Validate coherence - convenience function."""
    return get_unified_validation_system().validate_coherence(coherence, field_name)


def validate_string_input(
    input_string: str, field_name: str = "input"
) -> ValidationResult:
    """Validate string input - convenience function."""
    return get_unified_validation_system().validate_string_input(
        input_string, field_name
    )


def get_unified_validation_stats() -> dict[str, Any]:
    """Get unified validation statistics - convenience function."""
    if _unified_validation_system is not None:
        return _unified_validation_system.get_cache_statistics()
    return {"status": "system_not_initialized"}
