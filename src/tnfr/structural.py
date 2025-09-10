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
__all__ = ["create_nfr", "Operador", "OPERADORES"]
for _cls in OPERADORES.values():
    globals()[_cls.__name__] = _cls
    __all__.append(_cls.__name__)
__all__ += ["validate_sequence", "run_sequence"]
# ---------------------------------------------------------------------------
# 3) Motor de secuencias + validador sintáctico
# ---------------------------------------------------------------------------


_INICIO_VALIDOS = {"emision", "recursividad"}
_TRAMO_INTERMEDIO = {"disonancia", "acoplamiento", "resonancia"}
_CIERRE_VALIDO = {"silencio", "transicion", "recursividad"}


def _verify_token_format(nombres: list[str]) -> tuple[bool, str]:
    """Check basic type and format of the token list."""
    if not nombres:
        return False, "empty sequence"
    if any(not isinstance(n, str) for n in nombres):
        return False, "tokens must be str"
    if nombres[0] not in _INICIO_VALIDOS:
        return False, "must start with emission or recursion"
    nombres_set = set(nombres)
    desconocidos = nombres_set - OPERADORES.keys()
    if desconocidos:
        return False, f"unknown tokens: {', '.join(desconocidos)}"
    return True, "ok"


def _find_recepcion(nombres: list[str]) -> int:
    """Return index of the first ``recepcion`` token or ``-1``."""

    try:
        return nombres.index("recepcion")
    except ValueError:
        return -1


def _find_coherencia(nombres: list[str], start: int) -> int:
    """Return index of the first ``coherencia`` token after ``start`` or ``-1``."""

    try:
        return nombres.index("coherencia", start)
    except ValueError:
        return -1


def _has_tramo_intermedio(nombres: list[str], start: int) -> bool:
    """Check if any intermediate segment token appears after ``start``."""

    return any(n in _TRAMO_INTERMEDIO for n in nombres[start:])


def _has_cierre(nombres: list[str]) -> bool:
    """Check if the sequence ends with a valid closure token."""

    return any(n in _CIERRE_VALIDO for n in nombres[-2:])


def _thol_blocks_closed(nombres: list[str]) -> bool:
    """Ensure ``autoorganizacion`` blocks are closed with silence or contraction."""

    open_block = False
    for n in nombres:
        if n == "autoorganizacion":
            open_block = True
        elif open_block and n in {"silencio", "contraccion"}:
            open_block = False
    return not open_block


def _validate_logical_coherence(nombres: list[str]) -> tuple[bool, str]:
    """Validate logical coherence of the sequence."""
    i_rec = _find_recepcion(nombres)
    i_coh = _find_coherencia(nombres, i_rec + 1 if i_rec != -1 else 0)
    if i_rec == -1 or i_coh == -1:
        return False, "missing input→coherence segment"
    if not _has_tramo_intermedio(nombres, i_coh + 1):
        return False, "missing tension/coupling/resonance segment"
    if not _has_cierre(nombres):
        return False, "missing closure (silence/transition/recursion)"
    if not _thol_blocks_closed(nombres):
        return False, "THOL block without closure"
    return True, "ok"


def validate_sequence(nombres: list[str]) -> tuple[bool, str]:
    """Validate minimal TNFR syntax rules."""
    ok, msg = _verify_token_format(nombres)
    if not ok:
        return False, msg
    ok, msg = _validate_logical_coherence(nombres)
    if not ok:
        return False, msg
    return True, "ok"


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
