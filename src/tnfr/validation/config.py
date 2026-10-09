"""Global policy flags consumed by the structural validation runner.

Only flags and severity consumed by ``structural.run_sequence`` belong here.
Scalar admission, live U3 policy and validation caching retain their separate
owners. Construction and ``configure_validation`` normalize declared settings;
an invalid update is rejected before changing the shared configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ..config.parsing import parse_bool
from ..errors import TNFRValueError
from .invariants import InvariantSeverity

__all__ = [
    "ValidationConfig",
    "validation_config",
    "configure_validation",
]


@dataclass
class ValidationConfig:
    """Policy for the semantic and invariant checks in structural word execution."""

    # Validation levels
    validate_invariants: bool = True
    validate_each_step: bool = False  # Expensive, only for debugging
    min_severity: InvariantSeverity = InvariantSeverity.ERROR

    # Semantic validation
    enable_semantic_validation: bool = True
    allow_semantic_warnings: bool = True

    def __post_init__(self) -> None:
        settings = {
            field.name: getattr(self, field.name) for field in fields(ValidationConfig)
        }
        for name, value in _normalize_settings(settings).items():
            setattr(self, name, value)


def _normalize_settings(
    settings: dict[str, object],
) -> dict[str, bool | InvariantSeverity]:
    """Admit actual dataclass fields through shared flag and enum parsing."""
    available = {field.name for field in fields(ValidationConfig)}
    unknown = settings.keys() - available
    if unknown:
        raise TNFRValueError(
            f"Unknown validation config key: {', '.join(sorted(unknown))}",
            context={"keys": sorted(unknown), "available": sorted(available)},
            suggestion="Use a declared ValidationConfig field.",
        )
    normalized: dict[str, bool | InvariantSeverity] = {}
    for key, value in settings.items():
        try:
            normalized[key] = (
                InvariantSeverity(value) if key == "min_severity" else parse_bool(value)
            )
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(
                f"Invalid validation config value for {key}", context={"key": key}
            ) from exc
    return normalized


# Global configuration
validation_config = ValidationConfig()


def configure_validation(**kwargs: object) -> None:
    """Normalize a declared update before committing it to the global policy.

    Parameters
    ----------
    **kwargs
        Configuration parameters to update. Valid keys match
        ValidationConfig fields. Flags use shared Boolean parsing; severity
        accepts an InvariantSeverity member or its exact string value.

    Raises
    ------
    TNFRValueError
        If a key or value is invalid. No supplied setting is then committed.

    Examples
    --------
    >>> from tnfr.validation.config import configure_validation
    >>> configure_validation(validate_each_step=True)
    >>> configure_validation(enable_semantic_validation=True)
    """
    for key, value in _normalize_settings(kwargs).items():
        setattr(validation_config, key, value)
