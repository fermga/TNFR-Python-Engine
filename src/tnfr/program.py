from __future__ import annotations
"""program.py — API de secuencias canónicas con T’HOL como primera clase."""
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager

from .constants import DEFAULTS
from .helpers import register_callback
from .grammar import enforce_canonical_grammar, on_applied_glifo
from .operators import aplicar_glifo
from .sense import GLYPHS_CANONICAL

# Tipos básicos
Glyph = str
Node = Any
AdvanceFn = Callable[[Any], None]  # normalmente dynamics.step

# ---------------------
# Construcciones del DSL
# ---------------------

@dataclass
class WAIT:
    steps: int = 1

@dataclass
class TARGET:
    nodes: Optional[Iterable[Node]] = None   # None = todos los nodos

@dataclass
class THOL:
    body: Sequence[Any]
    repeat: int = 1                # cuántas veces repetir el cuerpo
    force_close: Optional[Glyph] = None  # None → cierre automático (gramática); 'SH’A' o 'NU’L' para forzar

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
    return int(G.graph.get("GLYPH_HYSTERESIS_WINDOW", DEFAULTS.get("GLYPH_HYSTERESIS_WINDOW", 1)))

def _all_nodes(G):
    return list(G.nodes())

# ---------------------
# Núcleo de ejecución
# ---------------------

def _apply_glyph_to_targets(G, g: Glyph, nodes: Optional[Iterable[Node]] = None):
    nodes = list(nodes) if nodes is not None else _all_nodes(G)
    w = _window(G)
    # Pasamos por la gramática antes de aplicar
    for n in nodes:
        g_eff = enforce_canonical_grammar(G, n, g)
        aplicar_glifo(G, n, g_eff, window=w)
        on_applied_glifo(G, n, g_eff)

def _advance(G, step_fn: Optional[AdvanceFn] = None):
    if step_fn is None:
        from .dynamics import step as step_fn
    step_fn(G)

# ---------------------
# Compilación de secuencia → lista de operaciones atómicas
# ---------------------

def _flatten(seq: Sequence[Token], current_target: Optional[TARGET] = None) -> List[Tuple[str, Any]]:
    """Devuelve lista de operaciones (op, payload).
    op ∈ { 'GLYPH', 'WAIT', 'TARGET' }.
    """
    ops: List[Tuple[str, Any]] = []
    for item in seq:
        if isinstance(item, TARGET):
            ops.append(("TARGET", item))
        elif isinstance(item, WAIT):
            ops.append(("WAIT", item.steps))
        elif isinstance(item, THOL):
            # abrir bloque T’HOL
            ops.append(("GLYPH", "T’HOL"))
            for _ in range(max(1, int(item.repeat))):
                ops.extend(_flatten(item.body, current_target))
            # cierre explícito si se pidió; si no, la gramática puede cerrarlo
            if item.force_close in ("SH’A", "NU’L"):
                ops.append(("GLYPH", item.force_close))
        else:
            # item debería ser un glifo
            g = str(item)
            if g not in GLYPHS_CANONICAL:
                # Permitimos glifos no listados (compat futuros), pero no forzamos
                pass
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
        repite el cuerpo y (opcional) fuerza cierre con SH’A/NU’L.
      - Los glifos se aplican pasando por `enforce_canonical_grammar`.
    """
    ops = _flatten(sequence)
    curr_target: Optional[Iterable[Node]] = None

    # Traza de programa en history
    if "history" not in G.graph:
        G.graph["history"] = {}
    trace = G.graph["history"].setdefault("program_trace", [])

    for op, payload in ops:
        if op == "TARGET":
            curr_target = list(payload.nodes) if payload.nodes is not None else None
            trace.append({"t": float(G.graph.get("_t", 0.0)), "op": "TARGET", "n": len(curr_target or _all_nodes(G))})
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
