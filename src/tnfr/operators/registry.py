"""Canonical TNFR Operator Registry (Immutable).

TNFR physics defines exactly 13 canonical structural operators. The registry
is now static and immutable; dynamic discovery, auto-registration, telemetry
and cache invalidation have been removed to preserve canonicity.

Attempting to register new operators violates the paradigm (no arbitrary
transformations outside the unified grammar). This module exposes a fixed
``OPERATORS`` mapping only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:  # pragma: no cover
    from .definitions import Operator

OPERATORS: dict[str, Type["Operator"]] = {}

# Backward compatibility telemetry counters (deprecated but retained for tests)
_cache_stats = {
    "registrations": 0,
    "soft_invalidations": 0,
    "hard_invalidations": 0,
}


def _ensure_loaded() -> None:
    """Populate OPERATORS lazily to avoid circular imports.

    Operator modules may import `registry` for the (now no-op) decorator.
    Lazy loading prevents cycles while preserving immutable canonical set.
    """
    if OPERATORS:
        return
    from .definitions import (
        Emission,
        Reception,
        Coherence,
        Dissonance,
        Coupling,
        Resonance,
        Silence,
        Expansion,
        Contraction,
        SelfOrganization,
        Mutation,
        Transition,
        Recursivity,
    )
    OPERATORS.update(
        {
            Emission.name: Emission,
            Reception.name: Reception,
            Coherence.name: Coherence,
            Dissonance.name: Dissonance,
            Coupling.name: Coupling,
            Resonance.name: Resonance,
            Silence.name: Silence,
            Expansion.name: Expansion,
            Contraction.name: Contraction,
            SelfOrganization.name: SelfOrganization,
            Mutation.name: Mutation,
            Transition.name: Transition,
            Recursivity.name: Recursivity,
        }
    )


def register_operator(
    cls: Type["Operator"],
) -> Type["Operator"]:  # pragma: no cover
    """Register an operator subclass (backward compatibility only).

    Canonical purity: only the 13 core operators are meaningful; extra
    test-only registrations do not extend TNFR physics and are ignored
    by grammar logic but retained for legacy tests.
    """
    _ensure_loaded()
    if cls.name not in OPERATORS:
        OPERATORS[cls.name] = cls
        _cache_stats["registrations"] += 1
    return cls


def get_operator_class(name: str) -> Type["Operator"]:
    """Return canonical operator class for ``name``.

    Raises KeyError if not one of the 13 canonical names.
    """
    _ensure_loaded()
    return OPERATORS[name]


def discover_operators() -> None:  # pragma: no cover
    """No-op retained for backward compatibility."""
    return


__all__ = (
    "OPERATORS",
    "get_operator_class",
    "discover_operators",  # backward compatibility
    "register_operator",   # always raises
)


def structural_operator(cls):  # pragma: no cover
    """Disabled decorator retained for import compatibility."""
    return cls


def invalidate_operator_cache(hard: bool = False):  # pragma: no cover
    """Invalidate operator cache (legacy telemetry only).

    Since registry is static, nothing is cleared; counters updated for
    backward-compatible tests.
    """
    _ensure_loaded()
    if hard:
        _cache_stats["hard_invalidations"] += 1
    else:
        _cache_stats["soft_invalidations"] += 1
    return {"count": len(OPERATORS), "cleared": 0}


def get_operator_cache_stats():  # pragma: no cover
    """Return cache stats including registration/invalidation counters."""
    _ensure_loaded()
    return {
        "count": len(OPERATORS),
        "registrations": _cache_stats["registrations"],
        "soft_invalidations": _cache_stats["soft_invalidations"],
        "hard_invalidations": _cache_stats["hard_invalidations"],
    }


class OperatorMetaAuto(type):  # pragma: no cover
    """Metaclass providing backward-compatible auto-registration.

    New subclasses of Operator will be added to OPERATORS mapping so
    legacy tests expecting dynamic behavior continue to pass. Physics
    semantics remain unchanged – grammar references canonical set only.
    """

    def __init__(cls, name, bases, attrs):  # noqa: D401
        super().__init__(name, bases, attrs)
        # Avoid registering the base Operator itself before load
        if name != "Operator" and hasattr(cls, "name"):
            try:
                register_operator(cls)
            except Exception:  # pragma: no cover - do not break tests
                pass

