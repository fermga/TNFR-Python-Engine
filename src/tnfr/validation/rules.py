"""Validation helpers grouped by rule type.

These utilities implement the canonical checks required by
:mod:`tnfr.validation.grammar`.  They are organised here to make it
explicit which pieces enforce repetition control, transition
compatibility or stabilisation thresholds.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TYPE_CHECKING

from ..alias import get_attr
from ..constants import get_aliases
from ..glyph_history import recent_glyph
from ..helpers.numeric import clamp01
from ..metrics.common import normalize_dnfr
from ..types import Glyph
from .compatibility import CANON_COMPAT, CANON_FALLBACK

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from .grammar import GrammarContext

ALIAS_SI = get_aliases("SI")
ALIAS_D2EPI = get_aliases("D2EPI")

__all__ = [
    "coerce_glyph",
    "glyph_fallback",
    "get_norm",
    "normalized_dnfr",
    "_norm_attr",
    "_si",
    "_accel_norm",
    "_check_repeats",
    "_maybe_force",
    "_check_oz_to_zhir",
    "_check_thol_closure",
    "_check_compatibility",
]


def coerce_glyph(val: Any) -> Glyph | Any:
    """Return ``val`` coerced to :class:`Glyph` when possible."""

    try:
        return Glyph(val)
    except (ValueError, TypeError):
        return val


def glyph_fallback(cand_key: str, fallbacks: Mapping[str, Any]) -> Glyph | str:
    """Determine fallback glyph for ``cand_key`` considering canon tables."""

    glyph_key = coerce_glyph(cand_key)
    canon_fb = (
        CANON_FALLBACK.get(glyph_key, cand_key)
        if isinstance(glyph_key, Glyph)
        else cand_key
    )
    fb = fallbacks.get(cand_key, canon_fb)
    return coerce_glyph(fb)


# -------------------------
# Normalisation helpers
# -------------------------

def get_norm(ctx: "GrammarContext", key: str) -> float:
    """Retrieve a global normalisation value from ``ctx.norms``."""

    return float(ctx.norms.get(key, 1.0)) or 1.0


def _norm_attr(ctx: "GrammarContext", nd, attr_alias: str, norm_key: str) -> float:
    """Normalise ``attr_alias`` using the global maximum ``norm_key``."""

    max_val = get_norm(ctx, norm_key)
    return clamp01(abs(get_attr(nd, attr_alias, 0.0)) / max_val)


def _si(nd) -> float:
    """Return the structural sense index for ``nd`` clamped to ``[0, 1]``."""

    return clamp01(get_attr(nd, ALIAS_SI, 0.5))


def _accel_norm(ctx: "GrammarContext", nd) -> float:
    """Normalise acceleration using the global maximum."""

    return _norm_attr(ctx, nd, ALIAS_D2EPI, "accel_max")


def normalized_dnfr(ctx: "GrammarContext", nd) -> float:
    """Normalise |ΔNFR| using the configured global maximum."""

    return normalize_dnfr(nd, get_norm(ctx, "dnfr_max"))


# -------------------------
# Validation rules
# -------------------------

def _check_repeats(ctx: "GrammarContext", n, cand: Glyph | str) -> Glyph | str:
    """Avoid recent repetitions according to ``ctx.cfg_soft``."""

    nd = ctx.G.nodes[n]
    cfg = ctx.cfg_soft
    gwin = int(cfg.get("window", 0))
    avoid = set(cfg.get("avoid_repeats", []))
    fallbacks = cfg.get("fallbacks", {})
    cand_key = cand.value if isinstance(cand, Glyph) else str(cand)
    if gwin > 0 and cand_key in avoid and recent_glyph(nd, cand_key, gwin):
        return glyph_fallback(cand_key, fallbacks)
    return cand


def _maybe_force(
    ctx: "GrammarContext",
    n,
    cand: Glyph | str,
    original: Glyph | str,
    accessor: Callable[["GrammarContext", dict[str, Any]], float],
    key: str,
) -> Glyph | str:
    """Restore ``original`` if ``accessor`` exceeds ``key`` threshold."""

    if cand == original:
        return cand
    force_th = float(ctx.cfg_soft.get(key, 0.60))
    if accessor(ctx, ctx.G.nodes[n]) >= force_th:
        return original
    return cand


def _check_oz_to_zhir(ctx: "GrammarContext", n, cand: Glyph | str) -> Glyph | str:
    """Enforce OZ precedents before allowing ZHIR mutations."""

    nd = ctx.G.nodes[n]
    cand_glyph = coerce_glyph(cand)
    if cand_glyph == Glyph.ZHIR:
        cfg = ctx.cfg_canon
        win = int(cfg.get("zhir_requires_oz_window", 3))
        dn_min = float(cfg.get("zhir_dnfr_min", 0.05))
        if (
            not recent_glyph(nd, Glyph.OZ, win)
            and normalized_dnfr(ctx, nd) < dn_min
        ):
            return Glyph.OZ
    return cand


def _check_thol_closure(
    ctx: "GrammarContext", n, cand: Glyph | str, st: dict[str, Any]
) -> Glyph | str:
    """Close THOL blocks with canonical glyphs once stabilised."""

    nd = ctx.G.nodes[n]
    if st.get("thol_open", False):
        st["thol_len"] = int(st.get("thol_len", 0)) + 1
        cfg = ctx.cfg_canon
        minlen = int(cfg.get("thol_min_len", 2))
        maxlen = int(cfg.get("thol_max_len", 6))
        close_dn = float(cfg.get("thol_close_dnfr", 0.15))
        if st["thol_len"] >= maxlen or (
            st["thol_len"] >= minlen
            and normalized_dnfr(ctx, nd) <= close_dn
        ):
            return (
                Glyph.NUL
                if _si(nd) >= float(cfg.get("si_high", 0.66))
                else Glyph.SHA
            )
    return cand


def _check_compatibility(ctx: "GrammarContext", n, cand: Glyph | str) -> Glyph | str:
    """Verify canonical transition compatibility for ``cand``."""

    nd = ctx.G.nodes[n]
    hist = nd.get("glyph_history")
    prev = hist[-1] if hist else None
    prev_glyph = coerce_glyph(prev)
    cand_glyph = coerce_glyph(cand)
    if isinstance(prev_glyph, Glyph):
        allowed = CANON_COMPAT.get(prev_glyph)
        if allowed is None:
            return cand
        if isinstance(cand_glyph, Glyph):
            if cand_glyph not in allowed:
                return CANON_FALLBACK.get(prev_glyph, cand_glyph)
        else:
            return CANON_FALLBACK.get(prev_glyph, cand)
    return cand
