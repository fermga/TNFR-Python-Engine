"""Análisis estructural."""
from __future__ import annotations
from typing import Iterable, Tuple, List
import networkx as nx

from .dynamics import (
    set_delta_nfr_hook,
    update_epi_via_nodal_equation,
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
    """Crea una red (graph) con un nodo NFR inicializado.

    Devuelve la tupla ``(G, name)`` para conveniencia.
    """
    G = graph if graph is not None else nx.Graph()
    G.add_node(
        name,
        **{
            ALIAS_EPI[0]: float(epi),
            ALIAS_VF[0]: float(vf),
            ALIAS_THETA[0]: float(theta),
        }
    )
    set_delta_nfr_hook(G, dnfr_hook)
    return G, name


# ---------------------------------------------------------------------------
# 2) Operadores estructurales como API de primer orden
# ---------------------------------------------------------------------------


class Operador:
    """Base para operadores TNFR.

    Cada operador define ``name`` (identificador ASCII) y ``glyph`` (glifo
    canónico). La llamada ejecuta el glifo correspondiente sobre el nodo.
    """

    name = "operador"
    glyph = None  # tipo: str

    def __call__(self, G: nx.Graph, node, **kw) -> None:
        if self.glyph is None:
            raise NotImplementedError("Operador sin glifo asignado")
        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


# Derivados concretos -------------------------------------------------------
#
def operador_factory(*pairs: Tuple[str, str]) -> dict[str, type[Operador]]:
    """Construye dinámicamente clases ``Operador``.

    Cada par ``(nombre, glifo)`` produce una subclase concreta con los
    atributos correspondientes. Las clases generadas se exponen en el módulo
    como ``CamelCase`` del nombre original y se registran en un diccionario
    para fácil acceso por nombre.
    """

    registry: dict[str, type[Operador]] = {}
    for nombre, glifo in pairs:
        class_name = nombre.title().replace("_", "").replace(" ", "")
        cls = type(class_name, (Operador,), {"name": nombre, "glyph": glifo})
        globals()[class_name] = cls
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
# ---------------------------------------------------------------------------
# 3) Motor de secuencias + validador sintáctico
# ---------------------------------------------------------------------------


_INICIO_VALIDOS = {"emision", "recursividad"}
_TRAMO_INTERMEDIO = {"disonancia", "acoplamiento", "resonancia"}
_CIERRE_VALIDO = {"silencio", "transicion", "recursividad"}


def _verify_token_format(nombres: List[str]) -> Tuple[bool, str]:
    """Comprueba tipo y formato básicos de la lista de tokens."""
    if not nombres:
        return False, "secuencia vacía"
    if any(not isinstance(n, str) for n in nombres):
        return False, "tokens deben ser str"
    if nombres[0] not in _INICIO_VALIDOS:
        return False, "debe iniciar en emisión o recursividad"
    desconocidos = [n for n in nombres if n not in OPERADORES]
    if desconocidos:
        return False, f"tokens desconocidos: {', '.join(desconocidos)}"
    return True, "ok"


def _validate_logical_coherence(nombres: List[str]) -> Tuple[bool, str]:
    """Valida la coherencia lógica de la secuencia."""
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
        return False, "falta tramo entrada→coherencia"
    if not found_intermedio:
        return False, "falta tramo de tensión/acoplamiento/resonancia"
    if not cierre_ok:
        return False, "falta cierre (silencio/transición/recursividad)"
    if thol_open:
        return False, "bloque THOL sin cierre"
    return True, "ok"


def validate_sequence(nombres: List[str]) -> Tuple[bool, str]:
    """Valida reglas mínimas de la sintaxis TNFR."""
    ok, msg = _verify_token_format(nombres)
    if not ok:
        return False, msg
    ok, msg = _validate_logical_coherence(nombres)
    if not ok:
        return False, msg
    return True, "ok"


def run_sequence(G: nx.Graph, node, ops: Iterable[Operador]) -> None:
    """Ejecuta una secuencia validada de operadores sobre el nodo dado."""
    ops_list = list(ops)
    nombres = [op.name for op in ops_list]
    ok, msg = validate_sequence(nombres)
    if not ok:
        raise ValueError(f"Secuencia no válida: {msg}")
    compute = G.graph.get("compute_delta_nfr")
    for op in ops_list:
        op(G, node)
        if callable(compute):
            compute(G)
        update_epi_via_nodal_equation(G)

