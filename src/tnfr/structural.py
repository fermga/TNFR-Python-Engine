"""Structural analysis."""

from __future__ import annotations
from typing import Iterable, Tuple, List
import networkx as nx

from .dynamics import (
    set_delta_nfr_hook,
    dnfr_epi_vf_mixed,
)
from .grammar import apply_glyph_with_grammar
from .types import Glyph
from .constants import ALIAS_EPI, ALIAS_VF, ALIAS_THETA


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
) -> Tuple[nx.Graph, str]:
    """Create a graph with an initialised NFR node.

    Returns the tuple ``(G, name)`` for convenience.
    """
    G = graph if graph is not None else nx.Graph()
    G.add_node(
        name,
        **{
            ALIAS_EPI[0]: float(epi),
            ALIAS_VF[0]: float(vf),
            ALIAS_THETA[0]: float(theta),
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
def operador_factory(*pairs: Tuple[str, str]) -> dict[str, type[Operador]]:
    """Dynamically build ``Operador`` subclasses.

    Each ``(name, glyph)`` pair produces a concrete subclass with the
    corresponding attributes. Generated classes are exposed in the module as
    ``CamelCase`` versions of the original name and registered in a dictionary
    for easy access by name.
    """

    registry: dict[str, type[Operador]] = {}
    for nombre, glyph in pairs:
        class_name = nombre.title().replace("_", "").replace(" ", "")
        cls = type(class_name, (Operador,), {"name": nombre, "glyph": glyph})
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

# Exposición explícita de clases concretas
Emision = OPERADORES["emision"]
Recepcion = OPERADORES["recepcion"]
Coherencia = OPERADORES["coherencia"]
Disonancia = OPERADORES["disonancia"]
Acoplamiento = OPERADORES["acoplamiento"]
Resonancia = OPERADORES["resonancia"]
Silencio = OPERADORES["silencio"]
Expansion = OPERADORES["expansion"]
Contraccion = OPERADORES["contraccion"]
Autoorganizacion = OPERADORES["autoorganizacion"]
Mutacion = OPERADORES["mutacion"]
Transicion = OPERADORES["transicion"]
Recursividad = OPERADORES["recursividad"]

__all__ = [
    "create_nfr",
    "Operador",
    "OPERADORES",
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
    "validate_sequence",
    "run_sequence",
]
# ---------------------------------------------------------------------------
# 3) Motor de secuencias + validador sintáctico
# ---------------------------------------------------------------------------


_INICIO_VALIDOS = {"emision", "recursividad"}
_TRAMO_INTERMEDIO = {"disonancia", "acoplamiento", "resonancia"}
_CIERRE_VALIDO = {"silencio", "transicion", "recursividad"}


def _verify_token_format(nombres: List[str]) -> Tuple[bool, str]:
    """Check basic type and format of the token list."""
    if not nombres:
        return False, "empty sequence"
    if any(not isinstance(n, str) for n in nombres):
        return False, "tokens must be str"
    if nombres[0] not in _INICIO_VALIDOS:
        return False, "must start with emission or recursion"
    desconocidos = [n for n in nombres if n not in OPERADORES]
    if desconocidos:
        return False, f"unknown tokens: {', '.join(desconocidos)}"
    return True, "ok"


def _validate_logical_coherence(nombres: List[str]) -> Tuple[bool, str]:
    """Validate logical coherence of the sequence."""
    i_rec = i_coh = -1
    found_intermedio = False
    cierre_ok = False
    thol_open = False
    total = len(nombres)
    for idx, n in enumerate(nombres):
        if n == "autoorganizacion":
            thol_open = True
        elif thol_open and n in {"silencio", "contraccion"}:
            thol_open = False
        if i_rec == -1 and n == "recepcion":
            i_rec = idx
        elif i_rec != -1 and i_coh == -1 and n == "coherencia":
            i_coh = idx
        elif i_coh != -1 and not found_intermedio and n in _TRAMO_INTERMEDIO:
            found_intermedio = True
        if idx >= total - 2 and n in _CIERRE_VALIDO:
            cierre_ok = True
    if i_rec == -1 or i_coh == -1:
        return False, "missing input→coherence segment"
    if not found_intermedio:
        return False, "missing tension/coupling/resonance segment"
    if not cierre_ok:
        return False, "missing closure (silence/transition/recursion)"
    if thol_open:
        return False, "THOL block without closure"
    return True, "ok"


def validate_sequence(nombres: List[str]) -> Tuple[bool, str]:
    """Validate minimal TNFR syntax rules."""
    ok, msg = _verify_token_format(nombres)
    if not ok:
        return False, msg
    ok, msg = _validate_logical_coherence(nombres)
    if not ok:
        return False, msg
    return True, "ok"


def run_sequence(G: nx.Graph, node, ops: Iterable[Operador]) -> None:
    """Execute a validated sequence of operators on the given node."""
    ops_list = list(ops)
    nombres = [op.name for op in ops_list]
    ok, msg = validate_sequence(nombres)
    if not ok:
        raise ValueError(f"Invalid sequence: {msg}")
    compute = G.graph.get("compute_delta_nfr")
    for op in ops_list:
        op(G, node)
        if callable(compute):
            compute(G)
        # ``update_epi_via_nodal_equation`` was previously invoked here to
        # recalculate the EPI value after each operator. The responsibility for
        # updating EPI now lies with the dynamics hook configured in
        # ``compute_delta_nfr`` or with external callers.
