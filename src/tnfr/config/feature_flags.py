"""Math feature flag configuration helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator

from .parsing import parse_bool

__all__ = ("MathFeatureFlags", "get_flags", "context_flags")


@dataclass(frozen=True)
class MathFeatureFlags:
    """Toggle optional mathematical behaviours in the engine."""

    enable_math_validation: bool = False
    enable_math_dynamics: bool = False
    log_performance: bool = False
    math_backend: str = "numpy"


_BASE_FLAGS: MathFeatureFlags | None = None
_CONTEXT_FLAGS: ContextVar[MathFeatureFlags | None] = ContextVar(
    "tnfr_math_feature_flags", default=None
)


def _parse_env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return parse_bool(value)
    except ValueError:
        return default


def _load_base_flags() -> MathFeatureFlags:
    global _BASE_FLAGS
    if _BASE_FLAGS is None:
        backend = os.getenv("TNFR_MATH_BACKEND")
        backend_choice = backend.strip() if backend else "numpy"
        _BASE_FLAGS = MathFeatureFlags(
            enable_math_validation=_parse_env_flag(
                "TNFR_ENABLE_MATH_VALIDATION", False
            ),
            enable_math_dynamics=_parse_env_flag("TNFR_ENABLE_MATH_DYNAMICS", False),
            log_performance=_parse_env_flag("TNFR_LOG_PERF", False),
            math_backend=backend_choice or "numpy",
        )
    return _BASE_FLAGS


def get_flags() -> MathFeatureFlags:
    """Return the currently active feature flags."""

    flags = _CONTEXT_FLAGS.get()
    return flags if flags is not None else _load_base_flags()


@contextmanager
def context_flags(**overrides: bool | str) -> Iterator[MathFeatureFlags]:
    """Override flags in the current context, isolated from other threads/tasks."""

    invalid = set(overrides) - set(MathFeatureFlags.__annotations__)
    if invalid:
        invalid_names = ", ".join(sorted(invalid))
        raise TypeError(f"Unknown flag overrides: {invalid_names}")

    previous = get_flags()
    parsed = {
        key: value if key == "math_backend" else parse_bool(value)
        for key, value in overrides.items()
    }
    next_flags = replace(previous, **parsed)
    token = _CONTEXT_FLAGS.set(next_flags)
    try:
        yield next_flags
    finally:
        _CONTEXT_FLAGS.reset(token)
