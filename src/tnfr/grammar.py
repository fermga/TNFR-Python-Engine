"""Reglas de gramática."""
from __future__ import annotations
from typing import Dict, Any, Set, Iterable, Optional

from .constants import (
    DEFAULTS,
    ALIAS_SI, ALIAS_DNFR,
    get_param,
)
from .helpers import get_attr, clamp01, reciente_glifo
from .types import Glyph

# Glifos nominales (para evitar typos)
AL = Glyph.AL
EN = Glyph.EN
IL = Glyph.IL
OZ = Glyph.OZ
UM = Glyph.UM
RA = Glyph.RA
SHA = Glyph.SHA
VAL = Glyph.VAL
NUL = Glyph.NUL
THOL = Glyph.THOL
ZHIR = Glyph.ZHIR
NAV = Glyph.NAV
REMESH = Glyph.REMESH

# -------------------------
# Estado de gramática por nodo
# -------------------------

def _gram_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Crea/retorna el estado de gramática nodal.
    Campos:
      - thol_open (bool)
      - thol_len (int)
    """
    return nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})

# -------------------------
# Compatibilidades canónicas (siguiente permitido)
# -------------------------
CANON_COMPAT: Dict[Glyph, Set[Glyph]] = {
    # Inicio / apertura
    Glyph.AL:   {Glyph.EN, Glyph.RA, Glyph.NAV, Glyph.VAL, Glyph.UM},
    Glyph.EN:   {Glyph.IL, Glyph.UM, Glyph.RA, Glyph.NAV},
    # Estabilización / difusión / acople
    Glyph.IL:   {Glyph.RA, Glyph.VAL, Glyph.UM, Glyph.SHA},
    Glyph.UM:   {Glyph.RA, Glyph.IL, Glyph.VAL, Glyph.NAV},
    Glyph.RA:   {Glyph.IL, Glyph.VAL, Glyph.UM, Glyph.NAV},
    Glyph.VAL:  {Glyph.UM, Glyph.RA, Glyph.IL, Glyph.NAV},
    # Disonancia → transición → mutación
    Glyph.OZ:   {Glyph.ZHIR, Glyph.NAV},
    Glyph.ZHIR: {Glyph.IL, Glyph.NAV},
    Glyph.NAV:  {Glyph.OZ, Glyph.ZHIR, Glyph.RA, Glyph.IL, Glyph.UM},
    # Cierres / latencias
    Glyph.SHA:  {Glyph.AL, Glyph.EN},
    Glyph.NUL:  {Glyph.AL, Glyph.IL},
    # Bloques autoorganizativos
    Glyph.THOL: {Glyph.OZ, Glyph.ZHIR, Glyph.NAV, Glyph.RA, Glyph.IL, Glyph.UM, Glyph.SHA, Glyph.NUL},
}

# Fallbacks canónicos si una transición no está permitida
CANON_FALLBACK: Dict[Glyph, Glyph] = {
    Glyph.AL: Glyph.EN,
    Glyph.EN: Glyph.IL,
    Glyph.IL: Glyph.RA,
    Glyph.NAV: Glyph.RA,
    Glyph.NUL: Glyph.AL,
    Glyph.OZ: Glyph.ZHIR,
    Glyph.RA: Glyph.IL,
    Glyph.SHA: Glyph.AL,
    Glyph.THOL: Glyph.NAV,
    Glyph.UM: Glyph.RA,
    Glyph.VAL: Glyph.RA,
    Glyph.ZHIR: Glyph.IL,
}

# -------------------------
# Cierres THOL y precondiciones ZHIR
# -------------------------

def _dnfr_norm(G, nd) -> float:
    # Normalizador robusto: usa historial de |ΔNFR| máx guardado por dynamics (si existe)
    norms = G.graph.get("_sel_norms") or {}
    dmax = float(norms.get("dnfr_max", 1.0)) or 1.0
    return clamp01(abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dmax)


def _si(G, nd) -> float:
    return clamp01(get_attr(nd, ALIAS_SI, 0.5))

# -------------------------
# Núcleo: forzar gramática sobre un candidato
# -------------------------

def enforce_canonical_grammar(G, n, cand: str) -> str:
    """Valida/ajusta el glifo candidato según la gramática canónica.

    Reglas clave:
      - Compatibilidades de transición glífica (recorrido TNFR).
      - OZ→ZHIR: la mutación requiere disonancia reciente o |ΔNFR| alto.
      - THOL[...]: obliga cierre con SHA o NUL cuando el campo se estabiliza
        o se alcanza el largo del bloque; mantiene estado por nodo.

    Devuelve el glifo efectivo a aplicar.
    """
    nd = G.nodes[n]
    st = _gram_state(nd)
    cfg = G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {}))

    # 0) Si vienen glifos fuera del alfabeto, no tocamos
    if cand not in CANON_COMPAT:
        return cand

    # 1) Precondición OZ→ZHIR: mutación requiere disonancia reciente o campo fuerte
    if cand == Glyph.ZHIR:
        win = int(cfg.get("zhir_requires_oz_window", 3))
        dn_min = float(cfg.get("zhir_dnfr_min", 0.05))
        if not reciente_glifo(nd, Glyph.OZ, win) and _dnfr_norm(G, nd) < dn_min:
            cand = Glyph.OZ  # forzamos paso por OZ

    # 2) Si estamos dentro de THOL, control de cierre obligado
    if st.get("thol_open", False):
        st["thol_len"] = int(st.get("thol_len", 0))
        st["thol_len"] += 1
        minlen = int(cfg.get("thol_min_len", 2))
        maxlen = int(cfg.get("thol_max_len", 6))
        close_dn = float(cfg.get("thol_close_dnfr", 0.15))
        if st["thol_len"] >= maxlen or (st["thol_len"] >= minlen and _dnfr_norm(G, nd) <= close_dn):
            cand = Glyph.NUL if _si(G, nd) >= float(cfg.get("si_high", 0.66)) else Glyph.SHA

    # 3) Compatibilidades: si el anterior restringe el siguiente
    hist = nd.get("hist_glifos")
    prev = hist[-1] if hist else None
    if prev in CANON_COMPAT and cand not in CANON_COMPAT[prev]:
        cand = CANON_FALLBACK.get(prev, cand)

    return cand

# -------------------------
# Post-selección: actualizar estado de gramática
# -------------------------

def on_applied_glifo(G, n, applied: str) -> None:
    nd = G.nodes[n]
    st = _gram_state(nd)
    if applied == Glyph.THOL:
        st["thol_open"] = True
        st["thol_len"] = 0
    elif applied in (Glyph.SHA, Glyph.NUL):
        st["thol_open"] = False
        st["thol_len"] = 0
    else:
        pass

# -------------------------
# Aplicación directa con gramática canónica
# -------------------------

def apply_glyph_with_grammar(G, nodes: Optional[Iterable[Any]], glyph: Glyph | str, window: Optional[int] = None) -> None:
    """Aplica ``glyph`` a ``nodes`` pasando por la gramática canónica.

    ``nodes`` admite ``NodeView`` y cualquier iterable. Se itera directamente
    sobre ``nodes`` para evitar materializaciones innecesarias; corresponde al
    llamador materializarlo si necesita indexación.
    """

    from .operators import aplicar_glifo

    if window is None:
        window = get_param(G, "GLYPH_HYSTERESIS_WINDOW")

    g_str = glyph.value if isinstance(glyph, Glyph) else str(glyph)
    iter_nodes = G.nodes() if nodes is None else nodes
    for n in iter_nodes:
        g_eff = enforce_canonical_grammar(G, n, g_str)
        aplicar_glifo(G, n, g_eff, window=window)
        on_applied_glifo(G, n, g_eff)

# -------------------------
# Integración con dynamics.step: helper de selección+aplicación
# -------------------------

def select_and_apply_with_grammar(G, n, selector, window: int) -> None:
    """Aplica gramática canónica sobre la propuesta del selector.

    El selector puede incluir una gramática **suave** (pre–filtro) como
    `parametric_glyph_selector`; la presente función garantiza que la
    gramática canónica tenga precedencia final.
    """
    cand = selector(G, n)
    apply_glyph_with_grammar(G, [n], cand, window)
