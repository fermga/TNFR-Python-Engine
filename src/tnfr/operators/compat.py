"""Legacy Spanish operator wrappers."""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, TypeVar

from ..config.operator_names import (
    ACOPLAMIENTO,
    AUTOORGANIZACION,
    COHERENCIA,
    CONTRACCION,
    DISONANCIA,
    EMISION,
    MUTACION,
    RECEPCION,
    RECURSIVIDAD,
    RESONANCIA,
    SILENCIO,
    TRANSICION,
)
from ..types import Glyph
from .registry import register_operator

if False:  # pragma: no cover - imported for type checkers only
    from .definitions import (  # noqa: F401
        Operator,
        Coupling,
        Coherence,
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
        Contraction,
    )

from . import definitions as _definitions

Operator = _definitions.Operator
Emission = _definitions.Emission
Reception = _definitions.Reception
Coherence = _definitions.Coherence
Dissonance = _definitions.Dissonance
Coupling = _definitions.Coupling
Resonance = _definitions.Resonance
Silence = _definitions.Silence
Contraction = _definitions.Contraction
SelfOrganization = _definitions.SelfOrganization
Mutation = _definitions.Mutation
Transition = _definitions.Transition
Recursivity = _definitions.Recursivity

LegacyOperatorT = TypeVar("LegacyOperatorT", bound="LegacyOperatorMixin")


class LegacyOperatorMixin:
    """Mixin that warns when legacy operators are instantiated."""

    legacy_name: ClassVar[str]
    canonical_cls: ClassVar[type[Operator]]

    def __init__(self: LegacyOperatorT, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            (
                "Spanish operator class '%s' is deprecated and will be removed "
                "in a future release; import '%s' instead"
            )
            % (self.__class__.__name__, self.canonical_cls.__name__),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class Operador(LegacyOperatorMixin, Operator):
    """Deprecated Spanish base class alias."""

    __slots__ = ()
    legacy_name: ClassVar[str] = "operador"
    canonical_cls: ClassVar[type[Operator]] = Operator
    glyph: ClassVar[Glyph | None] = Operator.glyph


@register_operator
class Emision(LegacyOperatorMixin, Emission):
    """Spanish alias for :class:`tnfr.operators.definitions.Emission`."""

    __slots__ = ()
    name: ClassVar[str] = EMISION
    legacy_name: ClassVar[str] = EMISION
    canonical_cls: ClassVar[type[Operator]] = Emission


@register_operator
class Recepcion(LegacyOperatorMixin, Reception):
    __slots__ = ()
    name: ClassVar[str] = RECEPCION
    legacy_name: ClassVar[str] = RECEPCION
    canonical_cls: ClassVar[type[Operator]] = Reception


@register_operator
class Coherencia(LegacyOperatorMixin, Coherence):
    __slots__ = ()
    name: ClassVar[str] = COHERENCIA
    legacy_name: ClassVar[str] = COHERENCIA
    canonical_cls: ClassVar[type[Operator]] = Coherence


@register_operator
class Disonancia(LegacyOperatorMixin, Dissonance):
    __slots__ = ()
    name: ClassVar[str] = DISONANCIA
    legacy_name: ClassVar[str] = DISONANCIA
    canonical_cls: ClassVar[type[Operator]] = Dissonance


@register_operator
class Acoplamiento(LegacyOperatorMixin, Coupling):
    __slots__ = ()
    name: ClassVar[str] = ACOPLAMIENTO
    legacy_name: ClassVar[str] = ACOPLAMIENTO
    canonical_cls: ClassVar[type[Operator]] = Coupling


@register_operator
class Resonancia(LegacyOperatorMixin, Resonance):
    __slots__ = ()
    name: ClassVar[str] = RESONANCIA
    legacy_name: ClassVar[str] = RESONANCIA
    canonical_cls: ClassVar[type[Operator]] = Resonance


@register_operator
class Silencio(LegacyOperatorMixin, Silence):
    __slots__ = ()
    name: ClassVar[str] = SILENCIO
    legacy_name: ClassVar[str] = SILENCIO
    canonical_cls: ClassVar[type[Operator]] = Silence


@register_operator
class Contraccion(LegacyOperatorMixin, Contraction):
    __slots__ = ()
    name: ClassVar[str] = CONTRACCION
    legacy_name: ClassVar[str] = CONTRACCION
    canonical_cls: ClassVar[type[Operator]] = Contraction


@register_operator
class Autoorganizacion(LegacyOperatorMixin, SelfOrganization):
    __slots__ = ()
    name: ClassVar[str] = AUTOORGANIZACION
    legacy_name: ClassVar[str] = AUTOORGANIZACION
    canonical_cls: ClassVar[type[Operator]] = SelfOrganization


@register_operator
class Mutacion(LegacyOperatorMixin, Mutation):
    __slots__ = ()
    name: ClassVar[str] = MUTACION
    legacy_name: ClassVar[str] = MUTACION
    canonical_cls: ClassVar[type[Operator]] = Mutation


@register_operator
class Transicion(LegacyOperatorMixin, Transition):
    __slots__ = ()
    name: ClassVar[str] = TRANSICION
    legacy_name: ClassVar[str] = TRANSICION
    canonical_cls: ClassVar[type[Operator]] = Transition


@register_operator
class Recursividad(LegacyOperatorMixin, Recursivity):
    __slots__ = ()
    name: ClassVar[str] = RECURSIVIDAD
    legacy_name: ClassVar[str] = RECURSIVIDAD
    canonical_cls: ClassVar[type[Operator]] = Recursivity


__all__ = (
    "Operador",
    "Emision",
    "Recepcion",
    "Coherencia",
    "Disonancia",
    "Acoplamiento",
    "Resonancia",
    "Silencio",
    "Contraccion",
    "Autoorganizacion",
    "Mutacion",
    "Transicion",
    "Recursividad",
)
