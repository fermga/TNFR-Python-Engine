"""Grammar rules."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Callable

from .constants import DEFAULTS, get_aliases, get_param
from .alias import get_attr
from .helpers.numeric import clamp01
from .glyph_history import recent_glyph
from .types import Glyph
from .operators import apply_glyph  # avoid repeated import inside functions
from .metrics_utils import normalize_dnfr

ALIAS_SI = get_aliases("SI")
ALIAS_D2EPI = get_aliases("D2EPI")


@dataclass
class GrammarContext:
    """Shared context for grammar helpers.

    Collects graph-level settings to reduce positional parameters across
    helper functions.
    """

    G: Any
    cfg_soft: dict[str, Any]
    cfg_canon: dict[str, Any]
    norms: dict[str, Any]

    @classmethod
    def from_graph(cls, G: Any) -> "GrammarContext":
        """Create a :class:`GrammarContext` for ``G``."""
        return cls(
            G=G,
            cfg_soft=G.graph.get("GRAMMAR", DEFAULTS.get("GRAMMAR", {})),
            cfg_canon=G.graph.get(
                "GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {})
            ),
            norms=G.graph.get("_sel_norms") or {},
        )


__all__ = (
    "CANON_COMPAT",
    "CANON_FALLBACK",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "apply_glyph_with_grammar",
    "GrammarContext",
)

# -------------------------
# Per-node grammar state
# -------------------------


def _gram_state(nd: dict[str, Any]) -> dict[str, Any]:
    """Create or return the node grammar state.

    Fields:
      - thol_open (bool)
      - thol_len (int)
    """
    return nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})


# -------------------------
# Canonical compatibilities (allowed next glyphs)
# -------------------------
CANON_COMPAT: dict[Glyph, set[Glyph]] = {
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
CANON_FALLBACK: dict[Glyph, Glyph] = {
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


def _coerce_glyph(val: Any) -> Glyph | Any:
    """Return ``val`` as ``Glyph`` when possible."""
    try:
        return Glyph(val)
    except (ValueError, TypeError):
        return val


def _glyph_fallback(cand_key: str, fallbacks: dict[str, Any]) -> Glyph | str:
    """Determine fallback glyph for ``cand_key`` and return converted value."""
    glyph_key = _coerce_glyph(cand_key)
    canon_fb = (
        CANON_FALLBACK.get(glyph_key, cand_key)
        if isinstance(glyph_key, Glyph)
        else cand_key
    )
    fb = fallbacks.get(cand_key, canon_fb)
    return _coerce_glyph(fb)


# -------------------------
# THOL closures and ZHIR preconditions
# -------------------------


def get_norm(ctx: GrammarContext, key: str) -> float:
    """Retrieve a global normalisation value from ``ctx.norms``."""
    return float(ctx.norms.get(key, 1.0)) or 1.0


def _norm_attr(ctx: GrammarContext, nd, attr_alias: str, norm_key: str) -> float:
    """Normalise ``attr_alias`` using the global maximum ``norm_key``."""

    max_val = get_norm(ctx, norm_key)
    return clamp01(abs(get_attr(nd, attr_alias, 0.0)) / max_val)


def _si(nd) -> float:
    return clamp01(get_attr(nd, ALIAS_SI, 0.5))


def _accel_norm(ctx: GrammarContext, nd) -> float:
    """Normalise acceleration using the global maximum."""
    return _norm_attr(ctx, nd, ALIAS_D2EPI, "accel_max")


def _check_repeats(ctx: GrammarContext, n, cand: Glyph | str) -> Glyph | str:
    """Avoid recent repetitions according to ``ctx.cfg_soft``."""
    nd = ctx.G.nodes[n]
    cfg = ctx.cfg_soft
    gwin = int(cfg.get("window", 0))
    avoid = set(cfg.get("avoid_repeats", []))
    fallbacks = cfg.get("fallbacks", {})
    cand_key = cand.value if isinstance(cand, Glyph) else str(cand)
    if gwin > 0 and cand_key in avoid and recent_glyph(nd, cand_key, gwin):
        return _glyph_fallback(cand_key, fallbacks)
    return cand


def _maybe_force(
    ctx: GrammarContext,
    n,
    cand: str,
    original: str,
    accessor: Callable[[GrammarContext, dict[str, Any]], float],
    key: str,
) -> str:
    """Restore ``original`` if ``accessor`` exceeds ``key`` threshold."""
    if cand == original:
        return cand
    force_th = float(ctx.cfg_soft.get(key, 0.60))
    if accessor(ctx, ctx.G.nodes[n]) >= force_th:
        return original
    return cand


def _check_oz_to_zhir(ctx: GrammarContext, n, cand: str) -> str:
    nd = ctx.G.nodes[n]
    if cand == Glyph.ZHIR:
        cfg = ctx.cfg_canon
        win = int(cfg.get("zhir_requires_oz_window", 3))
        dn_min = float(cfg.get("zhir_dnfr_min", 0.05))
        dnfr_max = get_norm(ctx, "dnfr_max")
        if (
            not recent_glyph(nd, Glyph.OZ, win)
            and normalize_dnfr(nd, dnfr_max) < dn_min
        ):
            return Glyph.OZ
    return cand


def _check_thol_closure(
    ctx: GrammarContext, n, cand: str, st: dict[str, Any]
) -> str:
    nd = ctx.G.nodes[n]
    if st.get("thol_open", False):
        st["thol_len"] = int(st.get("thol_len", 0)) + 1
        cfg = ctx.cfg_canon
        minlen = int(cfg.get("thol_min_len", 2))
        maxlen = int(cfg.get("thol_max_len", 6))
        close_dn = float(cfg.get("thol_close_dnfr", 0.15))
        dnfr_max = get_norm(ctx, "dnfr_max")
        if st["thol_len"] >= maxlen or (
            st["thol_len"] >= minlen
            and normalize_dnfr(nd, dnfr_max) <= close_dn
        ):
            return (
                Glyph.NUL
                if _si(nd) >= float(cfg.get("si_high", 0.66))
                else Glyph.SHA
            )
    return cand


def _check_compatibility(ctx: GrammarContext, n, cand: str) -> str:
    nd = ctx.G.nodes[n]
    hist = nd.get("glyph_history")
    prev = hist[-1] if hist else None
    if prev in CANON_COMPAT and cand not in CANON_COMPAT[prev]:
        return CANON_FALLBACK.get(prev, cand)
    return cand


# -------------------------
# Core: enforce grammar on a candidate
# -------------------------


def enforce_canonical_grammar(
    G, n, cand: str, ctx: Optional[GrammarContext] = None
) -> str:
    """Validate and adjust a candidate glyph according to canonical grammar.

    Key rules:
      - Repeat window with forces based on |ΔNFR| and acceleration.
      - Transition compatibilities (TNFR path).
      - OZ→ZHIR: mutation requires recent dissonance or high |ΔNFR|.
      - THOL[...]: forces closure with SHA or NUL when the field stabilises
        or block length is reached; maintains per-node state.

    Returns the effective glyph to apply.
    """
    if ctx is None:
        ctx = GrammarContext.from_graph(G)

    nd = ctx.G.nodes[n]
    st = _gram_state(nd)

    # 0) If glyphs outside the alphabet arrive, leave untouched
    if cand not in CANON_COMPAT:
        return cand

    original = cand
    cand = _check_repeats(ctx, n, cand)

    dnfr_accessor = lambda ctx, nd: normalize_dnfr(nd, get_norm(ctx, "dnfr_max"))
    cand = _maybe_force(ctx, n, cand, original, dnfr_accessor, "force_dnfr")
    cand = _maybe_force(ctx, n, cand, original, _accel_norm, "force_accel")
    cand = _check_oz_to_zhir(ctx, n, cand)
    cand = _check_thol_closure(ctx, n, cand, st)
    cand = _check_compatibility(ctx, n, cand)

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
    ctx = GrammarContext.from_graph(G)
    for n in iter_nodes:
        g_eff = enforce_canonical_grammar(G, n, g_str, ctx)
        apply_glyph(G, n, g_eff, window=window)
        on_applied_glyph(G, n, g_eff)
