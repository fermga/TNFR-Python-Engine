"""Structural analysis."""

from __future__ import annotations
from typing import Iterable
import networkx as nx  # type: ignore[import-untyped]

from .dynamics import (
    set_delta_nfr_hook,
    dnfr_epi_vf_mixed,
)
from .grammar import apply_glyph_with_grammar
from .types import Glyph
from .constants import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY


# ---------------------------------------------------------------------------
# 1) Factoría NFR
# ---------------------------------------------------------------------------


def create_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: nx.Graph | None = None,
    dnfr_hook=dnfr_epi_vf_mixed,
) -> tuple[nx.Graph, str]:
    """Create a graph with an initialised NFR node.

    Returns the tuple ``(G, name)`` for convenience.
    """
    G = graph if graph is not None else nx.Graph()
    G.add_node(
        name,
        **{
            EPI_PRIMARY: float(epi),
            VF_PRIMARY: float(vf),
            THETA_PRIMARY: float(theta),
        },
    )
    set_delta_nfr_hook(G, dnfr_hook)
    return G, name


# ---------------------------------------------------------------------------
# 2) Operadores estructurales como API de primer orden
# ---------------------------------------------------------------------------


class Operador:
    """Base class for TNFR operators.

    Each operator defines ``name`` (ASCII identifier) and ``glyph``
    (símbolo TNFR canónico). Calling an instance applies the corresponding
    symbol to the node.
    """

    name = "operador"
    glyph = None  # tipo: str

    def __call__(self, G: nx.Graph, node, **kw) -> None:
        if self.glyph is None:
            raise NotImplementedError("Operador sin glyph asignado")
        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


class Emision(Operador):
    """Aplicación del operador de emisión (símbolo ``AL``)."""

    __slots__ = ()
    name = "emision"
    glyph = Glyph.AL.value


class Recepcion(Operador):
    """Operador de recepción (símbolo ``EN``)."""

    __slots__ = ()
    name = "recepcion"
    glyph = Glyph.EN.value


class Coherencia(Operador):
    """Operador de coherencia (símbolo ``IL``)."""

    __slots__ = ()
    name = "coherencia"
    glyph = Glyph.IL.value


class Disonancia(Operador):
    """Operador de disonancia (símbolo ``OZ``)."""

    __slots__ = ()
    name = "disonancia"
    glyph = Glyph.OZ.value


class Acoplamiento(Operador):
    """Operador de acoplamiento (símbolo ``UM``)."""

    __slots__ = ()
    name = "acoplamiento"
    glyph = Glyph.UM.value


class Resonancia(Operador):
    """Operador de resonancia (símbolo ``RA``)."""

    __slots__ = ()
    name = "resonancia"
    glyph = Glyph.RA.value


class Silencio(Operador):
    """Operador de silencio (símbolo ``SHA``)."""

    __slots__ = ()
    name = "silencio"
    glyph = Glyph.SHA.value


class Expansion(Operador):
    """Operador de expansión (símbolo ``VAL``)."""

    __slots__ = ()
    name = "expansion"
    glyph = Glyph.VAL.value


class Contraccion(Operador):
    """Operador de contracción (símbolo ``NUL``)."""

    __slots__ = ()
    name = "contraccion"
    glyph = Glyph.NUL.value


class Autoorganizacion(Operador):
    """Operador de autoorganización (símbolo ``THOL``)."""

    __slots__ = ()
    name = "autoorganizacion"
    glyph = Glyph.THOL.value


class Mutacion(Operador):
    """Operador de mutación (símbolo ``ZHIR``)."""

    __slots__ = ()
    name = "mutacion"
    glyph = Glyph.ZHIR.value


class Transicion(Operador):
    """Operador de transición (símbolo ``NAV``)."""

    __slots__ = ()
    name = "transicion"
    glyph = Glyph.NAV.value


class Recursividad(Operador):
    """Operador de recursividad (símbolo ``REMESH``)."""

    __slots__ = ()
    name = "recursividad"
    glyph = Glyph.REMESH.value


OPERADORES: dict[str, type[Operador]] = {
    Emision.name: Emision,
    Recepcion.name: Recepcion,
    Coherencia.name: Coherencia,
    Disonancia.name: Disonancia,
    Acoplamiento.name: Acoplamiento,
    Resonancia.name: Resonancia,
    Silencio.name: Silencio,
    Expansion.name: Expansion,
    Contraccion.name: Contraccion,
    Autoorganizacion.name: Autoorganizacion,
    Mutacion.name: Mutacion,
    Transicion.name: Transicion,
    Recursividad.name: Recursividad,
}


__all__ = (
    "create_nfr",
    "Operador",
    "Emision",
    "Recepcion",
    "Coherencia",
    "Disonancia",
    "Acoplamiento",
    "Resonancia",
    "Silencio",
    "Expansion",
    "Contraccion",
    "Autoorganizacion",
    "Mutacion",
    "Transicion",
    "Recursividad",
    "OPERADORES",
    "validate_sequence",
    "run_sequence",
)
# ---------------------------------------------------------------------------
# 3) Motor de secuencias + validador sintáctico
# ---------------------------------------------------------------------------


_INICIO_VALIDOS = {"emision", "recursividad"}
_TRAMO_INTERMEDIO = {"disonancia", "acoplamiento", "resonancia"}
_CIERRE_VALIDO = {"silencio", "transicion", "recursividad"}


def _validate_start(token: str) -> tuple[bool, str]:
    """Ensure the sequence begins with a valid structural operator."""

    if not isinstance(token, str):
        return False, "tokens must be str"
    if token not in _INICIO_VALIDOS:
        return False, "must start with emission or recursion"
    return True, ""


def _validate_intermediate(
    found_recepcion: bool, found_coherencia: bool, seen_intermedio: bool
) -> tuple[bool, str]:
    """Check that the central TNFR segment is present."""

    if not (found_recepcion and found_coherencia):
        return False, "missing input→coherence segment"
    if not seen_intermedio:
        return False, "missing tension/coupling/resonance segment"
    return True, ""


def _validate_end(last_token: str, open_thol: bool) -> tuple[bool, str]:
    """Validate closing operator and any pending THOL blocks."""

    if last_token not in _CIERRE_VALIDO:
        return False, "sequence must end with silence/transition/recursion"
    if open_thol:
        return False, "THOL block without closure"
    return True, ""


def _validate_known_tokens(nombres_set: set[str]) -> tuple[bool, str]:
    """Ensure all tokens map to canonical operators."""

    desconocidos = nombres_set - OPERADORES.keys()
    if desconocidos:
        return False, f"unknown tokens: {', '.join(desconocidos)}"
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

        if n == "recepcion" and not found_recepcion:
            found_recepcion = True
        elif found_recepcion and n == "coherencia" and not found_coherencia:
            found_coherencia = True
        elif found_coherencia and not seen_intermedio and n in _TRAMO_INTERMEDIO:
            seen_intermedio = True

        if n == "autoorganizacion":
            open_thol = True
        elif open_thol and n in {"silencio", "contraccion"}:
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


def run_sequence(G: nx.Graph, node, ops: Iterable[Operador]) -> None:
    """Execute a sequence of operators on ``node`` after validation."""

    compute = G.graph.get("compute_delta_nfr")
    ops_list = list(ops)
    nombres = [op.name for op in ops_list]

    ok, msg = validate_sequence(nombres)
    if not ok:
        raise ValueError(f"Invalid sequence: {msg}")

    for op in ops_list:
        op(G, node)
        if callable(compute):
            compute(G)
        # ``update_epi_via_nodal_equation`` was previously invoked here to
        # recalculate the EPI value after each operator. The responsibility for
        # updating EPI now lies with the dynamics hook configured in
        # ``compute_delta_nfr`` or with external callers.
