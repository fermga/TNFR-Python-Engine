"""Lazy operator-class lookup with compatibility registration.

The lookup loads the 13 built-in classes and accepts distinct-name legacy
registrations. The canonical inventory belongs to ``operator_contracts``;
adding a class here does not add a canonical contract, executable glyph,
grammar role or atomic-stage guarantee. ``TNFR.operators()`` reads that
contract inventory rather than this mutable compatibility mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .definitions import Operator

OPERATORS: dict[str, type["Operator"]] = {}


def _ensure_loaded() -> None:
    """Populate OPERATORS lazily to avoid circular imports.

    The operator base class imports the registration metaclass from this module.
    Lazy loading avoids an eager dependency on the built-in definitions.
    """
    if OPERATORS:
        return
    from .definitions import (
        Coherence,
        Contraction,
        Coupling,
        Dissonance,
        Emission,
        Expansion,
        Mutation,
        Reception,
        Recursivity,
        Resonance,
        SelfOrganization,
        Silence,
        Transition,
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
    cls: type["Operator"],
) -> type["Operator"]:  # pragma: no cover
    """Register an operator subclass (backward compatibility only).

    Add a previously unused name to the compatibility lookup. Existing names
    are retained. Registration alone supplies no canonical execution contract
    or grammar role.
    """
    _ensure_loaded()
    if cls.name not in OPERATORS:
        OPERATORS[cls.name] = cls
    return cls


def get_operator_class(name: str) -> type["Operator"]:
    """Return the registered operator class for ``name``.

    Raise KeyError if the name is absent from the compatibility lookup.
    """
    _ensure_loaded()
    return OPERATORS[name]


__all__ = (
    "OPERATORS",
    "get_operator_class",
    "register_operator",  # backward compatibility
)


class OperatorMetaAuto(type):  # pragma: no cover
    """Metaclass providing opt-out-compatible legacy registration.

    Subclasses are registered for backward compatibility unless their class
    body explicitly sets ``__register__ = False``. This guard lets internal
    probes and application-only subclasses avoid changing the lookup. The
    separate canonical contract inventory is unaffected either way.
    """

    def __init__(cls, name, bases, attrs):  # noqa: D401
        super().__init__(name, bases, attrs)
        # Read the class body rather than the inherited base-class marker:
        # canonical operators rely on the legacy default, while an explicit
        # opt-out must never mutate the module-level registry.
        registration_enabled = attrs.get("__register__", True) is not False
        if name != "Operator" and registration_enabled and hasattr(cls, "name"):
            try:
                register_operator(cls)
            except Exception:  # pragma: no cover - do not break imports
                pass
