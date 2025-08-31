from __future__ import annotations
"""API de operadores estructurales y secuencias TNFR.

Este módulo ofrece:
    - Factoría `create_nfr` para inicializar redes/nodos TNFR.
    - Clases de operador (`Operador` y derivados) con interfaz común.
    - Registro de operadores `OPERADORES`.
    - Utilidades `validate_sequence` y `run_sequence` para ejecutar
      secuencias canónicas de operadores.
"""
from typing import Iterable, Tuple, List
import networkx as nx

from .dynamics import (
    set_delta_nfr_hook,
    update_epi_via_nodal_equation,
    dnfr_epi_vf_mixed,
)
from .operators import aplicar_glifo
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
    G = graph or nx.Graph()
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
        aplicar_glifo(G, node, self.glyph, **kw)


# Derivados concretos -------------------------------------------------------
class Emision(Operador):
    name = "emision"
    glyph = "A’L"


class Recepcion(Operador):
    name = "recepcion"
    glyph = "E’N"


class Coherencia(Operador):
    name = "coherencia"
    glyph = "I’L"


class Disonancia(Operador):
    name = "disonancia"
    glyph = "O’Z"


class Acoplamiento(Operador):
    name = "acoplamiento"
    glyph = "U’M"


class Resonancia(Operador):
    name = "resonancia"
    glyph = "R’A"


class Silencio(Operador):
    name = "silencio"
    glyph = "SH’A"


class Expansion(Operador):
    name = "expansion"
    glyph = "VA’L"


class Contraccion(Operador):
    name = "contraccion"
    glyph = "NU’L"


class Autoorganizacion(Operador):
    name = "autoorganizacion"
    glyph = "T’HOL"


class Mutacion(Operador):
    name = "mutacion"
    glyph = "Z’HIR"


class Transicion(Operador):
    name = "transicion"
    glyph = "NA’V"


class Recursividad(Operador):
    name = "recursividad"
    glyph = "RE’MESH"


OPERADORES = {
    op().name: op
    for op in [
        Emision,
        Recepcion,
        Coherencia,
        Disonancia,
        Acoplamiento,
        Resonancia,
        Silencio,
        Expansion,
        Contraccion,
        Autoorganizacion,
        Mutacion,
        Transicion,
        Recursividad,
    ]
}


# ---------------------------------------------------------------------------
# 3) Motor de secuencias + validador sintáctico
# ---------------------------------------------------------------------------


_INICIO_VALIDOS = {"emision", "recursividad"}
_TRAMO_INTERMEDIO = {"disonancia", "acoplamiento", "resonancia"}
_CIERRE_VALIDO = {"silencio", "transicion", "recursividad"}


def validate_sequence(nombres: List[str]) -> Tuple[bool, str]:
    """Valida reglas mínimas de la sintaxis TNFR."""
    if not nombres:
        return False, "secuencia vacía"
    if nombres[0] not in _INICIO_VALIDOS:
        return False, "debe iniciar en emisión o recursividad"
    try:
        i_rec = nombres.index("recepcion")
        i_coh = nombres.index("coherencia", i_rec + 1)
    except ValueError:
        return False, "falta tramo entrada→coherencia"
    if not any(n in _TRAMO_INTERMEDIO for n in nombres[i_coh + 1 :]):
        return False, "falta tramo de tensión/acoplamiento/resonancia"
    if not any(n in _CIERRE_VALIDO for n in nombres[-2:]):
        return False, "falta cierre (silencio/transición/recursividad)"
    return True, "ok"


def run_sequence(G: nx.Graph, node, ops: Iterable[Operador]) -> None:
    """Ejecuta una secuencia validada de operadores sobre el nodo dado."""
    ops_list = list(ops)
    nombres = [op.name for op in ops_list]
    ok, msg = validate_sequence(nombres)
    if not ok:
        raise ValueError(f"Secuencia no válida: {msg}")
    for op in ops_list:
        op(G, node)
        compute = G.graph.get("compute_delta_nfr")
        if callable(compute):
            compute(G)
        update_epi_via_nodal_equation(G)

