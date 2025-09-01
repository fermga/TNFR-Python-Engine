"""Lenguaje de programación TNFR."""
from __future__ import annotations
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from collections.abc import Collection
from dataclasses import dataclass
from contextlib import contextmanager
from collections import deque

from .constants import get_param
from .grammar import apply_glyph_with_grammar
from .sense import GLYPHS_CANONICAL_SET
from .types import Glyph

# Tipos básicos
Node = Any
AdvanceFn = Callable[[Any], None]  # normalmente dynamics.step

# ---------------------
# Construcciones del DSL
# ---------------------

@dataclass
class WAIT:
    """Esperar cierto número de pasos sin aplicar glifos.

    Attributes:
        steps: número de pasos que avanza el sistema.
    """
    steps: int = 1

@dataclass
class TARGET:
    """Selecciona el subconjunto de nodos para los glifos siguientes.

    Attributes:
        nodes: iterable de nodos; ``None`` para todos los nodos.
    """
    nodes: Optional[Iterable[Node]] = None   # None = todos los nodos

@dataclass
class THOL:
    """Bloque THOL que abre la autoorganización.

    Attributes:
        body: secuencia de tokens a ejecutar dentro del bloque.
        repeat: cuántas veces repetir ``body``.
        force_close: ``Glyph.SHA`` o ``Glyph.NUL`` para forzar cierre.
    """
    body: Sequence[Any]
    repeat: int = 1                # cuántas veces repetir el cuerpo
    force_close: Optional[Glyph] = None  # None → cierre automático (gramática); SHA o NUL para forzar

Token = Union[Glyph, WAIT, TARGET, THOL]

# ---------------------
# Utilidades internas
# ---------------------

@contextmanager
def _forced_selector(G, glyph: Glyph):
    """Sobrescribe temporalmente el selector glífico para forzar `glyph`.
    Pasa por la gramática canónica antes de aplicar.
    """
    prev = G.graph.get("glyph_selector")
    def selector_forced(_G, _n):
        return glyph
    G.graph["glyph_selector"] = selector_forced
    try:
        yield
    finally:
        if prev is None:
            G.graph.pop("glyph_selector", None)
        else:
            G.graph["glyph_selector"] = prev

def _window(G) -> int:
    return int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))

def _all_nodes(G):
    """Return a view or generator over all nodes in ``G``.

    Using ``G.nodes()`` avoids creating an intermediate list and allows
    callers to materialise the collection only when strictly necessary.
    """
    return G.nodes()

# ---------------------
# Núcleo de ejecución
# ---------------------

def _apply_glyph_to_targets(G, g: Glyph | str, nodes: Optional[Iterable[Node]] = None):
    """Apply ``g`` to ``nodes`` (or all nodes) respecting the grammar.

    ``nodes`` may be any iterable of nodes, including a ``NodeView`` or
    other :class:`~collections.abc.Collection`. Non-collection iterables
    are materialised as a ``list`` only when the active selector requires
    index-based access.
    """
    if nodes is None:
        nodes = _all_nodes(G)
    elif not isinstance(nodes, Collection):
        nodes = list(nodes)
    w = _window(G)
    apply_glyph_with_grammar(G, nodes, g, w)

def _advance(G, step_fn: Optional[AdvanceFn] = None):
    if step_fn is None:
        from .dynamics import step as step_fn
    step_fn(G)

# ---------------------
# Compilación de secuencia → lista de operaciones atómicas
# ---------------------

def _flatten(seq: Sequence[Token]) -> List[Tuple[str, Any]]:
    """Devuelve lista de operaciones (op, payload).
    op ∈ { 'GLYPH', 'WAIT', 'TARGET' }.

    Implementación iterativa usando una pila explícita para evitar
    recursión profunda cuando se anidan bloques ``THOL``.
    """
    ops: List[Tuple[str, Any]] = []
    stack: List[Token] = list(reversed(seq))

    while stack:
        item = stack.pop()
        if isinstance(item, TARGET):
            ops.append(("TARGET", item))
            continue
        if isinstance(item, WAIT):
            ops.append(("WAIT", item.steps))
            continue
        if isinstance(item, THOL):
            # Abrimos bloque THOL y apilamos su cuerpo de forma iterativa
            ops.append(("GLYPH", Glyph.THOL.value))

            repeats = max(1, int(item.repeat))
            closing = item.force_close if item.force_close in (Glyph.SHA, Glyph.NUL) else None
            # El cierre explícito debe ejecutarse al final, por lo que se
            # coloca en la pila antes de expandir el cuerpo.
            if closing is not None:
                stack.append(closing)

            # Insertamos los cuerpos en orden para que la primera repetición
            # sea procesada antes.
            for _ in range(repeats):
                for tok in reversed(item.body):
                    stack.append(tok)
            continue

        # item debería ser un glifo
        g = item.value if isinstance(item, Glyph) else str(item)
        if g not in GLYPHS_CANONICAL_SET:
            raise ValueError(f"Glifo no canónico: {g}")
        ops.append(("GLYPH", g))
    return ops

# ---------------------
# API pública
# ---------------------

def play(G, sequence: Sequence[Token], step_fn: Optional[AdvanceFn] = None) -> None:
    """Ejecuta una secuencia canónica sobre el grafo `G`.

    Reglas:
      - Usa `TARGET(nodes=...)` para cambiar el subconjunto de aplicación.
      - `WAIT(k)` avanza k pasos con el selector vigente (no fuerza glifo).
      - `THOL([...], repeat=r, force_close=…)` abre un bloque autoorganizativo,
        repite el cuerpo y (opcional) fuerza cierre con SHA/NUL.
      - Los glifos se aplican pasando por `enforce_canonical_grammar`.
    """
    ops = _flatten(sequence)
    curr_target: Optional[List[Node]] = None

    # Traza de programa en history
    history = G.graph.setdefault("history", {})
    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    trace = history.get("program_trace")
    if not isinstance(trace, deque) or trace.maxlen != maxlen:
        trace = deque(trace or [], maxlen=maxlen)
        history["program_trace"] = trace

    for op, payload in ops:
        if op == "TARGET":
            if payload.nodes is None:
                curr_target = list(_all_nodes(G))
            else:
                curr_target = list(payload.nodes)
            trace.append({"t": float(G.graph.get("_t", 0.0)), "op": "TARGET", "n": len(curr_target)})
            continue
        if op == "WAIT":
            for _ in range(max(1, int(payload))):
                _advance(G, step_fn)
            trace.append({"t": float(G.graph.get("_t", 0.0)), "op": "WAIT", "k": int(payload)})
            continue
        if op == "GLYPH":
            g = str(payload)
            # aplicar + avanzar 1 paso del sistema
            _apply_glyph_to_targets(G, g, curr_target)
            _advance(G, step_fn)
            trace.append({"t": float(G.graph.get("_t", 0.0)), "op": "GLYPH", "g": g})
            continue

# ---------------------
# Helpers para construir secuencias de manera cómoda
# ---------------------

def seq(*tokens: Token) -> List[Token]:
    return list(tokens)

def block(*tokens: Token, repeat: int = 1, close: Optional[Glyph] = None) -> THOL:
    return THOL(body=list(tokens), repeat=repeat, force_close=close)

def target(nodes: Optional[Iterable[Node]] = None) -> TARGET:
    return TARGET(nodes=nodes)

def wait(steps: int = 1) -> WAIT:
    return WAIT(steps=max(1, int(steps)))


def ejemplo_canonico_basico() -> List[Token]:
    """Secuencia canónica de referencia.

    SHA → AL → RA → ZHIR → NUL → THOL
    """
    return seq(Glyph.SHA, Glyph.AL, Glyph.RA, Glyph.ZHIR, Glyph.NUL, Glyph.THOL)
