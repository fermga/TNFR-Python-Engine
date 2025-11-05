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
    """Configuración del sistema de validación TNFR."""

    # Niveles de validación
    validate_invariants: bool = True
    validate_each_step: bool = False  # Costoso, solo para debugging
    min_severity: InvariantSeverity = InvariantSeverity.ERROR

    # Umbrales numéricos (se pueden sobrescribir desde graph.graph config)
    epi_range: tuple[float, float] = (0.0, 1.0)
    vf_range: tuple[float, float] = (0.001, 1000.0)  # Hz_str
    phase_coupling_threshold: float = math.pi / 2

    # Validación semántica
    enable_semantic_validation: bool = True
    allow_semantic_warnings: bool = True

    # Performance
    cache_validation_results: bool = False  # Future optimization
    max_validation_time_ms: float = 1000.0  # Timeout (not implemented yet)


# Configuración global
validation_config = ValidationConfig()


def configure_validation(**kwargs: object) -> None:
    """Actualiza configuración global de validación.

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
    global validation_config
    for key, value in kwargs.items():
        if hasattr(validation_config, key):
            setattr(validation_config, key, value)
        else:
            raise ValueError(f"Unknown validation config key: {key}")
