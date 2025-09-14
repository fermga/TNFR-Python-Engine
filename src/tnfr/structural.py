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
    (canonical glyph). Calling an instance applies the corresponding glyph
    to the node.
    """

    name = "operador"
    glyph = None  # tipo: str

    def __call__(self, G: nx.Graph, node, **kw) -> None:
        if self.glyph is None:
            raise NotImplementedError("Operador sin glyph asignado")
        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


# Derivados concretos -------------------------------------------------------
#
def operador_factory(*pairs: tuple[str, str]) -> dict[str, type[Operador]]:
    """Dynamically build ``Operador`` subclasses.

    Each ``(name, glyph)`` pair produces a concrete subclass with the
    corresponding attributes. Generated classes are exposed in the module as
    ``CamelCase`` versions of the original name and registered in a dictionary
    for easy access by name.
    """

    registry: dict[str, type[Operador]] = {}
    for nombre, glyph in pairs:
        class_name = nombre.title().replace("_", "").replace(" ", "")
        cls = type(
            class_name,
            (Operador,),
            {
                "name": nombre,
                "glyph": glyph,
                "__module__": __name__,
                "__slots__": (),
            },
        )
        registry[nombre] = cls
    return registry


OPERADORES = operador_factory(
    ("emision", Glyph.AL.value),
    ("recepcion", Glyph.EN.value),
    ("coherencia", Glyph.IL.value),
    ("disonancia", Glyph.OZ.value),
    ("acoplamiento", Glyph.UM.value),
    ("resonancia", Glyph.RA.value),
    ("silencio", Glyph.SHA.value),
    ("expansion", Glyph.VAL.value),
    ("contraccion", Glyph.NUL.value),
    ("autoorganizacion", Glyph.THOL.value),
    ("mutacion", Glyph.ZHIR.value),
    ("transicion", Glyph.NAV.value),
    ("recursividad", Glyph.REMESH.value),
)

# Exposición dinámica de clases concretas en el espacio global
__all__ = ("create_nfr", "Operador", "OPERADORES")
for _cls in OPERADORES.values():
    globals()[_cls.__name__] = _cls
    __all__ += (_cls.__name__,)
__all__ += ("validate_sequence", "run_sequence")
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
