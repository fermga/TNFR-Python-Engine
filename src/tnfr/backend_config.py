"""Configuration for optional execution services.

This module provides unified configuration for all TNFR subsystems, including:
- Mathematics backend selection (JAX, PyTorch, NumPy)
- GPU acceleration preferences
- Validation levels and thresholds
- Operator sequence optimization
- Memory management settings
- Structural field computation parameters

This service configuration is distinct from the graph parameters in
``tnfr.config``. Its integration defaults are recommendations returned by
``get_integration_config``; setting them does not rewrite a graph's DT or
INTEGRATOR_METHOD. Environment overrides are read when this object is created.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    XI_C_CRITICAL_RATIO,
)
from .errors import TNFRValueError
from .config.parsing import parse_bool

__all__ = [
    "TNFRConfig",
    "get_config",
    "configure",
    "reset_config",
]

BackendType = Literal["auto", "jax", "torch", "numpy"]
ValidationLevel = Literal["strict", "normal", "minimal", "disabled"]
GPUMode = Literal["auto", "force", "disabled"]


@dataclass
class TNFRConfig:
    """Optional backend/service settings, separate from graph configuration."""

    # Mathematics Backend Configuration
    math_backend: BackendType = "auto"
    enable_autodiff: bool = True
    backend_cache_enabled: bool = True

    # GPU Acceleration Settings
    gpu_mode: GPUMode = "auto"
    cuda_enabled: bool = True
    gpu_memory_pool_size_mb: int | None = None
    max_gpu_memory_percent: float = 0.9
    enable_mixed_precision: bool = False

    # Validation Configuration (coherent with nodal equation)
    validation_level: ValidationLevel = "normal"
    validate_invariants: bool = True
    validate_each_step: bool = False  # Expensive
    cache_validation_results: bool = True
    max_validation_time_ms: float = 1000.0

    # Structural Field Tetrad Parameters (audit 2026: only π genuine)
    structural_potential_threshold: float = (
        PHI_S_VON_KOCH_THRESHOLD  # π/4 selected per-node warning policy
    )
    phase_gradient_threshold: float = GRAD_PHI_CANONICAL_THRESHOLD  # π/16 policy
    phase_curvature_threshold: float = (
        K_PHI_CANONICAL_THRESHOLD  # selected 0.9π margin; exact bound π
    )
    coherence_length_critical: float = XI_C_CRITICAL_RATIO  # finite-size comparison

    # Nodal Equation Integration Parameters
    default_dt: float = 0.1
    integration_method: Literal["euler", "rk4"] = "rk4"
    max_integration_steps: int = 10000
    convergence_tolerance: float = 1e-6

    # Operator Sequence Optimization
    enable_operator_caching: bool = True
    max_sequence_cache_size: int = 1000
    grammar_validation_strict: bool = True
    optimize_sequences: bool = True

    # Memory Management
    enable_memory_pooling: bool = True
    max_cache_size_mb: int = 512
    gc_frequency: int = 100  # steps

    # Performance Monitoring
    enable_telemetry: bool = True
    log_performance_metrics: bool = False
    profile_operations: bool = False

    # Environment Variable Overrides
    _env_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Apply environment variable overrides following TNFR conventions."""
        env_mappings: dict[str, tuple[str, Any]] = {
            "TNFR_MATH_BACKEND": ("math_backend", str),
            "TNFR_CUDA_ENABLED": ("cuda_enabled", self._parse_bool),
            "TNFR_GPU_MODE": ("gpu_mode", str),
            "TNFR_VALIDATION_LEVEL": ("validation_level", str),
            "TNFR_DISABLE_OPTIMIZER": (
                "optimize_sequences",
                lambda x: not self._parse_bool(x),
            ),
            "TNFR_ENABLE_TELEMETRY": ("enable_telemetry", self._parse_bool),
            "TNFR_MAX_MEMORY_MB": ("max_cache_size_mb", int),
            "TNFR_INTEGRATION_DT": ("default_dt", float),
        }

        for env_var, (attr_name, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    setattr(self, attr_name, converted_value)
                    self._env_overrides[attr_name] = converted_value
                except (ValueError, TypeError) as e:
                    # Log warning but continue with defaults
                    import warnings

                    warnings.warn(
                        f"Invalid TNFR environment variable {env_var}={env_value}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse boolean from environment string."""
        return parse_bool(value)

    def get_backend_config(self) -> dict[str, Any]:
        """Get configuration dict for mathematics backend."""
        return {
            "backend": self.math_backend,
            "enable_cuda": self.cuda_enabled and self.gpu_mode != "disabled",
            "enable_autodiff": self.enable_autodiff,
            "memory_pool_size": self.gpu_memory_pool_size_mb,
            "mixed_precision": self.enable_mixed_precision,
        }

    def get_validation_config(self) -> dict[str, Any]:
        """Get configuration dict for validation system."""
        return {
            "level": self.validation_level,
            "validate_invariants": self.validate_invariants,
            "validate_each_step": self.validate_each_step,
            "cache_results": self.cache_validation_results,
            "max_time_ms": self.max_validation_time_ms,
            "strict_grammar": self.grammar_validation_strict,
        }

    def get_structural_config(self) -> dict[str, Any]:
        """Get structural field thresholds (tetrad; audit 2026: π genuine, rest overlay)."""
        return {
            "phi_s_threshold": self.structural_potential_threshold,
            "phase_gradient_threshold": self.phase_gradient_threshold,
            "phase_curvature_threshold": self.phase_curvature_threshold,
            "coherence_length_critical": self.coherence_length_critical,
        }

    def get_integration_config(self) -> dict[str, Any]:
        """Get nodal equation integration parameters."""
        return {
            "dt": self.default_dt,
            "method": self.integration_method,
            "max_steps": self.max_integration_steps,
            "tolerance": self.convergence_tolerance,
        }


# Global configuration instance
_global_config: TNFRConfig | None = None


def get_config() -> TNFRConfig:
    """Get global TNFR configuration instance.

    Returns:
        TNFRConfig: Global configuration with environment overrides applied
    """
    global _global_config
    if _global_config is None:
        _global_config = TNFRConfig()
    return _global_config


def configure(**kwargs: Any) -> None:
    """Update global TNFR configuration.

    Parameters
    ----------
    **kwargs
        Configuration parameters to update

    Examples
    --------
    >>> from tnfr.backend_config import configure
    >>> configure(math_backend="torch", cuda_enabled=True)
    >>> configure(validation_level="strict", enable_telemetry=False)
    """
    config = get_config()

    allowed = {item.name for item in fields(config) if not item.name.startswith("_")}
    for key in kwargs:
        if key not in allowed:
            raise TNFRValueError(
                f"Unknown configuration parameter: {key}",
                context={"parameter": key},
                suggestion="Check available configuration options in TNFRConfig.",
            )
    for key, value in kwargs.items():
        setattr(config, key, value)


def reset_config() -> None:
    """Reset global configuration to defaults with fresh environment scan."""
    global _global_config
    _global_config = None  # Force recreation on next access
