"""Grammar rules."""

from __future__ import annotations
from typing import Dict, Any, Set, Iterable, Optional, Callable

from .constants import (
    DEFAULTS,
    ALIAS_SI,
    ALIAS_DNFR,
    ALIAS_D2EPI,
    get_param,
)
from .alias import get_attr
from .helpers import clamp01
from .glyph_history import recent_glyph
from .types import Glyph
from .operators import apply_glyph  # avoid repeated import inside functions

# Nominal glyphs (to avoid typos)
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
# Per-node grammar state
# -------------------------


def _gram_state(nd: Dict[str, Any]) -> Dict[str, Any]:
    """Create or return the node grammar state.

    Fields:
      - thol_open (bool)
      - thol_len (int)
    """
    return nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})


# -------------------------
# Canonical compatibilities (allowed next glyphs)
# -------------------------
CANON_COMPAT: Dict[Glyph, Set[Glyph]] = {
    # Inicio / apertura
    Glyph.AL: {Glyph.EN, Glyph.RA, Glyph.NAV, Glyph.VAL, Glyph.UM},
    Glyph.EN: {Glyph.IL, Glyph.UM, Glyph.RA, Glyph.NAV},
    # Estabilización / difusión / acople
    Glyph.IL: {Glyph.RA, Glyph.VAL, Glyph.UM, Glyph.SHA},
    Glyph.UM: {Glyph.RA, Glyph.IL, Glyph.VAL, Glyph.NAV},
    Glyph.RA: {Glyph.IL, Glyph.VAL, Glyph.UM, Glyph.NAV},
    Glyph.VAL: {Glyph.UM, Glyph.RA, Glyph.IL, Glyph.NAV},
    # Disonancia → transición → mutación
    Glyph.OZ: {Glyph.ZHIR, Glyph.NAV},
    Glyph.ZHIR: {Glyph.IL, Glyph.NAV},
    Glyph.NAV: {Glyph.OZ, Glyph.ZHIR, Glyph.RA, Glyph.IL, Glyph.UM},
    # Cierres / latencias
    Glyph.SHA: {Glyph.AL, Glyph.EN},
    Glyph.NUL: {Glyph.AL, Glyph.IL},
    # Bloques autoorganizativos
    Glyph.THOL: {
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.NAV,
        Glyph.RA,
        Glyph.IL,
        Glyph.UM,
        Glyph.SHA,
        Glyph.NUL,
    },
}

# Canonical fallbacks when a transition is not allowed
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
# THOL closures and ZHIR preconditions
# -------------------------


def _norm_attr(G, nd, attr_alias: str, norm_key: str) -> float:
    """Normalise ``attr_alias`` using the global maximum ``norm_key``.

    ``_sel_norms`` is stored in ``G.graph`` and contains per-attribute
    maxima computed by selectors. If ``norm_key`` is missing the default
    value ``1.0`` is used to avoid division by zero.
    """

    norms = G.graph.get("_sel_norms") or {}
    max_val = float(norms.get(norm_key, 1.0)) or 1.0
    return clamp01(abs(get_attr(nd, attr_alias, 0.0)) / max_val)


def _dnfr_norm(G, nd) -> float:
    """Robust normaliser for |ΔNFR|."""
    return _norm_attr(G, nd, ALIAS_DNFR, "dnfr_max")


def _si(G, nd) -> float:
    return clamp01(get_attr(nd, ALIAS_SI, 0.5))


def _accel_norm(G, nd) -> float:
    """Normalise acceleration using the global maximum."""
    return _norm_attr(G, nd, ALIAS_D2EPI, "accel_max")


def _check_repeats(G, n, cand: str, cfg: Dict[str, Any]) -> str:
    """Avoid recent repetitions according to ``cfg``."""
    nd = G.nodes[n]
    gwin = int(cfg.get("window", 0))
    avoid = set(cfg.get("avoid_repeats", []))
    fallbacks = cfg.get("fallbacks", {})
    cand_key = cand.value if isinstance(cand, Glyph) else str(cand)
    if gwin > 0 and cand_key in avoid and recent_glyph(nd, cand_key, gwin):
        try:
            glyph_key = Glyph(cand_key)
        except ValueError:
            glyph_key = cand_key
        fb = fallbacks.get(cand_key, CANON_FALLBACK.get(glyph_key, cand_key))
        try:
            return Glyph(fb)
        except (ValueError, TypeError):
            return fb
    return cand


def _check_force(
    G,
    n,
    cand: str,
    original: str,
    cfg: Dict[str, Any],
    accessor: Callable[[Any, Dict[str, Any]], float],
    cfg_key: str,
) -> str:
    """If repetition was blocked but ``accessor`` exceeds the threshold,
    restore ``original``."""
    if cand == original:
        return cand
    force_th = float(cfg.get(cfg_key, 0.60))
    if accessor(G, G.nodes[n]) >= force_th:
        return original
    return cand


def _check_oz_to_zhir(G, n, cand: str, cfg: Dict[str, Any]) -> str:
    nd = G.nodes[n]
    if cand == Glyph.ZHIR:
        win = int(cfg.get("zhir_requires_oz_window", 3))
        dn_min = float(cfg.get("zhir_dnfr_min", 0.05))
        if not recent_glyph(nd, Glyph.OZ, win) and _dnfr_norm(G, nd) < dn_min:
            return Glyph.OZ
    return cand


def _check_thol_closure(
    G, n, cand: str, cfg: Dict[str, Any], st: Dict[str, Any]
) -> str:
    nd = G.nodes[n]
    if st.get("thol_open", False):
        st["thol_len"] = int(st.get("thol_len", 0)) + 1
        minlen = int(cfg.get("thol_min_len", 2))
        maxlen = int(cfg.get("thol_max_len", 6))
        close_dn = float(cfg.get("thol_close_dnfr", 0.15))
        if st["thol_len"] >= maxlen or (
            st["thol_len"] >= minlen and _dnfr_norm(G, nd) <= close_dn
        ):
            return (
                Glyph.NUL
                if _si(G, nd) >= float(cfg.get("si_high", 0.66))
                else Glyph.SHA
            )
    return cand


def _check_compatibility(G, n, cand: str) -> str:
    nd = G.nodes[n]
    hist = nd.get("glyph_history")
    prev = hist[-1] if hist else None
    if prev in CANON_COMPAT and cand not in CANON_COMPAT[prev]:
        return CANON_FALLBACK.get(prev, cand)
    return cand


# -------------------------
# Core: enforce grammar on a candidate
# -------------------------


def enforce_canonical_grammar(G, n, cand: str) -> str:
    """Validate and adjust a candidate glyph according to canonical grammar.

    Key rules:
      - Repeat window with forces based on |ΔNFR| and acceleration.
      - Transition compatibilities (TNFR path).
      - OZ→ZHIR: mutation requires recent dissonance or high |ΔNFR|.
      - THOL[...]: forces closure with SHA or NUL when the field stabilises
        or block length is reached; maintains per-node state.

    Returns the effective glyph to apply.
    """
    nd = G.nodes[n]
    st = _gram_state(nd)
    cfg_canon = G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {}))
    cfg_soft = G.graph.get("GRAMMAR", DEFAULTS.get("GRAMMAR", {}))

    # 0) If glyphs outside the alphabet arrive, leave untouched
    if cand not in CANON_COMPAT:
        return cand

    original = cand
    cand = _check_repeats(G, n, cand, cfg_soft)
    cand = _check_force(
        G, n, cand, original, cfg_soft, _dnfr_norm, "force_dnfr"
    )
    cand = _check_force(
        G, n, cand, original, cfg_soft, _accel_norm, "force_accel"
    )
    cand = _check_oz_to_zhir(G, n, cand, cfg_canon)
    cand = _check_thol_closure(G, n, cand, cfg_canon, st)
    cand = _check_compatibility(G, n, cand)

    return cand


# -------------------------
# Post-selection: update grammar state
# -------------------------


def on_applied_glyph(G, n, applied: str) -> None:
    nd = G.nodes[n]
    st = _gram_state(nd)
    if applied == Glyph.THOL:
        st["thol_open"] = True
        st["thol_len"] = 0
    elif applied in (Glyph.SHA, Glyph.NUL):
        st["thol_open"] = False
        st["thol_len"] = 0


# -------------------------
# Direct application with canonical grammar
# -------------------------


def apply_glyph_with_grammar(
    G,
    nodes: Optional[Iterable[Any]],
    glyph: Glyph | str,
    window: Optional[int] = None,
) -> None:
    """Apply ``glyph`` to ``nodes`` enforcing the canonical grammar.

    ``nodes`` may be a ``NodeView`` or any iterable. The iterable is consumed
    directly to avoid unnecessary materialisation; callers must materialise if
    they need indexing.
    """
    if window is None:
        window = get_param(G, "GLYPH_HYSTERESIS_WINDOW")

    g_str = glyph.value if isinstance(glyph, Glyph) else str(glyph)
    iter_nodes = G.nodes() if nodes is None else nodes
    for n in iter_nodes:
        g_eff = enforce_canonical_grammar(G, n, g_str)
        apply_glyph(G, n, g_eff, window=window)
        on_applied_glyph(G, n, g_eff)
