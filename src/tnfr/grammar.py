from __future__ import annotations
from typing import Dict, Any, Set

from .constants import (
    DEFAULTS,
    ALIAS_SI, ALIAS_DNFR, ALIAS_EPI,
)
from .helpers import _get_attr, clamp01, reciente_glifo
from collections import deque

# Glifos nominales (para evitar typos)
AL = "A’L"; EN = "E’N"; IL = "I’L"; OZ = "O’Z"; UM = "U’M"; RA = "R’A"; SHA = "SH’A"; VAL = "VA’L"; NUL = "NU’L"; THOL = "T’HOL"; ZHIR = "Z’HIR"; NAV = "NA’V"; REMESH = "RE’MESH"

# -------------------------
# Estado de gramática por nodo
# -------------------------

def _gram_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Crea/retorna el estado de gramática nodal.
    Campos:
      - thol_open (bool)
      - thol_len (int)
    """
    st = nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})
    st.setdefault("thol_open", False)
    st.setdefault("thol_len", 0)
    return st

# -------------------------
# Compatibilidades canónicas (siguiente permitido)
# -------------------------
CANON_COMPAT: Dict[str, Set[str]] = {
    # Inicio / apertura
    AL:   {EN, RA, NAV, VAL, UM},
    EN:   {IL, UM, RA, NAV},
    # Estabilización / difusión / acople
    IL:   {RA, VAL, UM, SHA},
    UM:   {RA, IL, VAL, NAV},
    RA:   {IL, VAL, UM, NAV},
    VAL:  {UM, RA, IL, NAV},
    # Disonancia → transición → mutación
    OZ:   {ZHIR, NAV},
    ZHIR: {IL, NAV},
    NAV:  {OZ, ZHIR, RA, IL, UM},
    # Cierres / latencias
    SHA:  {AL, EN},
    NUL:  {AL, IL},
    # Bloques autoorganizativos
    THOL: {OZ, ZHIR, NAV, RA, IL, UM, SHA, NUL},
}

# Fallbacks canónicos si una transición no está permitida
CANON_FALLBACK: Dict[str, str] = {
    AL: EN, EN: IL, IL: RA, UM: RA, RA: IL, VAL: RA, OZ: ZHIR, ZHIR: IL, NAV: RA, SHA: AL, NUL: AL, THOL: NAV,
}

# -------------------------
# Cierres T’HOL y precondiciones Z’HIR
# -------------------------

def _dnfr_norm(G, nd) -> float:
    # Normalizador robusto: usa historial de |ΔNFR| máx guardado por dynamics (si existe)
    norms = G.graph.get("_sel_norms") or {}
    dmax = float(norms.get("dnfr_max", 1.0)) or 1.0
    return clamp01(abs(_get_attr(nd, ALIAS_DNFR, 0.0)) / dmax)


def _si(G, nd) -> float:
    return clamp01(_get_attr(nd, ALIAS_SI, 0.5))

# -------------------------
# Núcleo: forzar gramática sobre un candidato
# -------------------------

def enforce_canonical_grammar(G, n, cand: str) -> str:
    """Valida/ajusta el glifo candidato según la gramática canónica.

    Reglas clave:
      - Compatibilidades de transición glífica (recorrido TNFR).
      - O’Z→Z’HIR: la mutación requiere disonancia reciente o |ΔNFR| alto.
      - T’HOL[...]: obliga cierre con SH’A o NU’L cuando el campo se estabiliza
        o se alcanza el largo del bloque; mantiene estado por nodo.

    Devuelve el glifo efectivo a aplicar.
    """
    nd = G.nodes[n]
    st = _gram_state(nd)
    cfg = G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {}))

    # 0) Si vienen glifos fuera del alfabeto, no tocamos
    if cand not in CANON_COMPAT:
        return cand

    # 1) Precondición O’Z→Z’HIR: mutación requiere disonancia reciente o campo fuerte
    if cand == ZHIR:
        win = int(cfg.get("zhir_requires_oz_window", 3))
        dn_min = float(cfg.get("zhir_dnfr_min", 0.05))
        if not reciente_glifo(nd, OZ, win) and _dnfr_norm(G, nd) < dn_min:
            cand = OZ  # forzamos paso por O’Z

    # 2) Si estamos dentro de T’HOL, control de cierre obligado
    if st.get("thol_open", False):
        st["thol_len"] = int(st.get("thol_len", 0))
        st["thol_len"] += 1
        minlen = int(cfg.get("thol_min_len", 2))
        maxlen = int(cfg.get("thol_max_len", 6))
        close_dn = float(cfg.get("thol_close_dnfr", 0.15))
        if st["thol_len"] >= maxlen or (st["thol_len"] >= minlen and _dnfr_norm(G, nd) <= close_dn):
            cand = NUL if _si(G, nd) >= float(cfg.get("si_high", 0.66)) else SHA

    # 3) Compatibilidades: si el anterior restringe el siguiente
    prev = None
    hist = nd.get("hist_glifos")
    if hist:
        try:
            prev = list(hist)[-1]
        except Exception:
            prev = None
    if prev in CANON_COMPAT and cand not in CANON_COMPAT[prev]:
        cand = CANON_FALLBACK.get(prev, cand)

    return cand

# -------------------------
# Post-selección: actualizar estado de gramática
# -------------------------

def on_applied_glifo(G, n, applied: str) -> None:
    nd = G.nodes[n]
    st = _gram_state(nd)
    if applied == THOL:
        st["thol_open"] = True
        st["thol_len"] = 0
    elif applied in (SHA, NUL):
        st["thol_open"] = False
        st["thol_len"] = 0
    else:
        pass

# -------------------------
# Integración con dynamics.step: helper de selección+aplicación
# -------------------------

def select_and_apply_with_grammar(G, n, selector, window: int) -> None:
    from .operators import aplicar_glifo
    cand = selector(G, n)
    cand = enforce_canonical_grammar(G, n, cand)
    aplicar_glifo(G, n, cand, window=window)
    on_applied_glifo(G, n, cand)
