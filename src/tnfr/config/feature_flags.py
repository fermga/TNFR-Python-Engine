"""Math feature flag configuration helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator

__all__ = ("MathFeatureFlags", "get_flags", "context_flags")


@dataclass(frozen=True)
class MathFeatureFlags:
    """Toggle optional mathematical behaviours in the engine."""

    enable_math_validation: bool = False
    enable_math_dynamics: bool = False
    log_performance: bool = False
    math_backend: str = "numpy"


_TRUE_VALUES = {"1", "true", "on", "yes", "y", "t"}
_FALSE_VALUES = {"0", "false", "off", "no", "n", "f"}

_BASE_FLAGS: MathFeatureFlags | None = None
_FLAGS_STACK: list[MathFeatureFlags] = []


def _parse_env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return default


def _load_base_flags() -> MathFeatureFlags:
    global _BASE_FLAGS
    if _BASE_FLAGS is None:
        backend = os.getenv("TNFR_MATH_BACKEND")
        backend_choice = backend.strip() if backend else "numpy"
        _BASE_FLAGS = MathFeatureFlags(
            enable_math_validation=_parse_env_flag("TNFR_ENABLE_MATH_VALIDATION", False),
            enable_math_dynamics=_parse_env_flag("TNFR_ENABLE_MATH_DYNAMICS", False),
            log_performance=_parse_env_flag("TNFR_LOG_PERF", False),
            math_backend=backend_choice or "numpy",
        )
    return _BASE_FLAGS


def get_flags() -> MathFeatureFlags:
    """Return the currently active feature flags."""

    if _FLAGS_STACK:
        return _FLAGS_STACK[-1]
    return _load_base_flags()


@contextmanager
def context_flags(**overrides: bool) -> Iterator[MathFeatureFlags]:
    """Temporarily override math feature flags."""

    invalid = set(overrides) - set(MathFeatureFlags.__annotations__)
    if invalid:
        invalid_names = ", ".join(sorted(invalid))
        raise TypeError(f"Unknown flag overrides: {invalid_names}")

    previous = get_flags()
    next_flags = replace(previous, **overrides)
    _FLAGS_STACK.append(next_flags)
    try:
        yield next_flags
    finally:
        _FLAGS_STACK.pop()
