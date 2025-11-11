"""Configuration for TNFR validation system.

This module provides configuration options for controlling validation behavior,
including thresholds, performance settings, and validation levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .invariants import InvariantSeverity

__all__ = [
    "ValidationConfig",
    "validation_config",
    "configure_validation",
]


@dataclass
class ValidationConfig:
    """TNFR validation system configuration."""

    # Validation levels
    validate_invariants: bool = True
    validate_each_step: bool = False  # Expensive, only for debugging
    min_severity: InvariantSeverity = InvariantSeverity.ERROR

    # Numerical thresholds (can be overridden from graph.graph config)
    epi_range: tuple[float, float] = (0.0, 1.0)
    vf_range: tuple[float, float] = (0.001, 1000.0)  # Hz_str
    phase_coupling_threshold: float = math.pi / 2

    # Semantic validation
    enable_semantic_validation: bool = True
    allow_semantic_warnings: bool = True

    # Performance
    cache_validation_results: bool = False  # Future optimization
    max_validation_time_ms: float = 1000.0  # Timeout (not implemented yet)


# Global configuration
validation_config = ValidationConfig()


def configure_validation(**kwargs: object) -> None:
    """Updates global validation configuration.

    Parameters
    ----------
    **kwargs
        Configuration parameters to update. Valid keys match
        ValidationConfig attributes.

    Raises
    ------
    ValueError
        If an unknown configuration key is provided.

    Examples
    --------
    >>> from tnfr.validation.config import configure_validation
    >>> configure_validation(validate_each_step=True)
    >>> configure_validation(phase_coupling_threshold=3.14159/3)
    """
    for key, value in kwargs.items():
        if hasattr(validation_config, key):
            setattr(validation_config, key, value)
        else:
            raise ValueError(f"Unknown validation config key: {key}")
