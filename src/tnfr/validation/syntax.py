"""Syntax validation for TNFR operator sequences."""

from __future__ import annotations

from ..operators.registry import OPERADORES
from ..config.operator_names import (
    ACOPLAMIENTO,
    AUTOORGANIZACION,
    CIERRE_VALIDO,
    COHERENCIA,
    CONTRACCION,
    DISONANCIA,
    EMISION,
    INICIO_VALIDOS,
    RECURSIVIDAD,
    RECEPCION,
    RESONANCIA,
    SILENCIO,
    TRANSICION,
    canonical_operator_name,
    operator_display_name,
)

__all__ = ("validate_sequence",)


_CANONICAL_START = (EMISION, RECURSIVIDAD)
_CANONICAL_INTERMEDIATE = (DISONANCIA, ACOPLAMIENTO, RESONANCIA)
_CANONICAL_END = (SILENCIO, TRANSICION, RECURSIVIDAD)
_AUTO_CLOSURES = (SILENCIO, CONTRACCION)


def _format_token_group(tokens: tuple[str, ...]) -> str:
    return ", ".join(operator_display_name(token) for token in tokens)


def _validate_start(token: str) -> tuple[bool, str]:
    """Ensure the sequence begins with a valid structural operator."""

    if not isinstance(token, str):
        return False, "tokens must be str"
    if token not in INICIO_VALIDOS:
        valid_tokens = _format_token_group(_CANONICAL_START)
        return False, f"must start with {valid_tokens}"
    return True, ""


def _validate_intermediate(
    found_recepcion: bool, found_coherencia: bool, seen_intermedio: bool
) -> tuple[bool, str]:
    """Check that the central TNFR segment is present."""

    if not (found_recepcion and found_coherencia):
        return False, f"missing {RECEPCION}→{COHERENCIA} segment"
    if not seen_intermedio:
        intermedio_tokens = _format_token_group(_CANONICAL_INTERMEDIATE)
        return False, f"missing {intermedio_tokens} segment"
    return True, ""


def _validate_end(last_token: str, open_thol: bool) -> tuple[bool, str]:
    """Validate closing operator and any pending THOL blocks."""

    if last_token not in CIERRE_VALIDO:
        cierre_tokens = _format_token_group(_CANONICAL_END)
        return False, f"sequence must end with {cierre_tokens}"
    if open_thol:
        return False, "THOL block without closure"
    return True, ""


def _validate_known_tokens(nombres_set: set[str]) -> tuple[bool, str]:
    """Ensure all tokens map to canonical operators."""

    operadores_canonicos = set(OPERADORES.keys())
    desconocidos = nombres_set - operadores_canonicos
    if desconocidos:
        tokens_ordenados = ", ".join(sorted(desconocidos))
        return False, f"unknown tokens: {tokens_ordenados}"
    return True, ""


def _validate_token_sequence(nombres: list[str]) -> tuple[bool, str]:
    """Validate token format and logical coherence in one pass."""

    if not nombres:
        return False, "empty sequence"

    ok, msg = _validate_start(nombres[0])
    if not ok:
        return False, msg

    nombres_set: set[str] = set()
    found_recepcion = False
    found_coherencia = False
    seen_intermedio = False
    open_thol = False

    for n in nombres:
        if not isinstance(n, str):
            return False, "tokens must be str"
        nombres_set.add(n)

        canonical = canonical_operator_name(n)

        if canonical == RECEPCION and not found_recepcion:
            found_recepcion = True
        elif found_recepcion and canonical == COHERENCIA and not found_coherencia:
            found_coherencia = True
        elif (
            found_coherencia
            and not seen_intermedio
            and canonical in _CANONICAL_INTERMEDIATE
        ):
            seen_intermedio = True

        if canonical == AUTOORGANIZACION:
            open_thol = True
        elif open_thol and canonical in _AUTO_CLOSURES:
            open_thol = False

    ok, msg = _validate_known_tokens(nombres_set)
    if not ok:
        return False, msg
    ok, msg = _validate_intermediate(found_recepcion, found_coherencia, seen_intermedio)
    if not ok:
        return False, msg
    ok, msg = _validate_end(nombres[-1], open_thol)
    if not ok:
        return False, msg
    return True, "ok"


def validate_sequence(nombres: list[str]) -> tuple[bool, str]:
    """Validate minimal TNFR syntax rules."""

    return _validate_token_sequence(nombres)
