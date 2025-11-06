"""Validation helpers grouped by rule type.

These utilities implement the canonical checks required by
:mod:`tnfr.validation`.  They are organised here to make it
explicit which pieces enforce repetition control, transition
compatibility or stabilisation thresholds.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping

from ..alias import get_attr
from ..constants.aliases import ALIAS_SI
from ..config.operator_names import (
    CONTRACTION,
    DISSONANCE,
    MUTATION,
    SELF_ORGANIZATION,
    SILENCE,
    canonical_operator_name,
    operator_display_name,
)
from ..utils import clamp01
from ..metrics.common import normalize_dnfr
from ..types import Glyph

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from ..operators.grammar import GrammarContext

__all__ = [
    "coerce_glyph",
    "glyph_fallback",
    "get_norm",
    "normalized_dnfr",
    "_norm_attr",
    "_si",
    "_check_oz_to_zhir",
    "_check_thol_closure",
    "_check_compatibility",
]


def coerce_glyph(val: Any) -> Glyph | Any:
    """Return ``val`` coerced to :class:`Glyph` when possible."""

    try:
        return Glyph(val)
    except (ValueError, TypeError):
        if isinstance(val, str) and val.startswith("Glyph."):
            _, _, candidate = val.partition(".")
            if candidate:
                try:
                    return Glyph(candidate)
                except ValueError:
                    pass  # Invalid glyph candidate, return as-is
        return val


def glyph_fallback(cand_key: str, fallbacks: Mapping[str, Any]) -> Glyph | str:
    """Determine fallback glyph for ``cand_key`` considering canon tables."""

    glyph_key = coerce_glyph(cand_key)
    fb_override = fallbacks.get(cand_key)
    if fb_override is not None:
        return coerce_glyph(fb_override)

    glyph_to_name, name_to_glyph = _functional_translators()
    _, fallback_table = _structural_tables()
    cand_name = glyph_to_name(glyph_key if isinstance(glyph_key, Glyph) else cand_key)
    if cand_name is None:
        return coerce_glyph(cand_key)
    fb_name = fallback_table.get(cand_name)
    if fb_name is None:
        return coerce_glyph(cand_key)
    fb_glyph = name_to_glyph(fb_name)
    return fb_glyph if fb_glyph is not None else coerce_glyph(cand_key)


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


def normalized_dnfr(ctx: "GrammarContext", nd) -> float:
    """Normalise |ΔNFR| using the configured global maximum."""

    return normalize_dnfr(nd, get_norm(ctx, "dnfr_max"))


# -------------------------
# Translation helpers
# -------------------------


def _structural_label(value: object) -> str:
    """Return the canonical structural name for ``value`` when possible."""

    glyph_to_name = _functional_translators()[0]
    coerced = coerce_glyph(value)
    if isinstance(coerced, Glyph):
        name = glyph_to_name(coerced)
        if name is not None:
            return name
    name = glyph_to_name(value if isinstance(value, Glyph) else value)
    if name is not None:
        return name
    if value is None:
        return "unknown"
    return canonical_operator_name(str(value))


# -------------------------
# Validation rules
# -------------------------


def _check_oz_to_zhir(ctx: "GrammarContext", n, cand: Glyph | str) -> Glyph | str:
    """Enforce OZ precedents before allowing ZHIR mutations.

    When mutation is attempted without recent dissonance and low ΔNFR,
    returns DISSONANCE as a fallback glyph (structural requirement).
    """

    from ..glyph_history import recent_glyph

    nd = ctx.G.nodes[n]
    cand_glyph = coerce_glyph(cand)
    glyph_to_name, name_to_glyph = _functional_translators()
    cand_name = glyph_to_name(cand_glyph if isinstance(cand_glyph, Glyph) else cand)
    if cand_name == MUTATION:
        cfg = ctx.cfg_canon
        win = int(cfg.get("zhir_requires_oz_window", 3))
        dn_min = float(cfg.get("zhir_dnfr_min", 0.05))
        dissonance_glyph = name_to_glyph(DISSONANCE)
        if dissonance_glyph is None:
            return cand
        norm_dn = normalized_dnfr(ctx, nd)
        recent_glyph(nd, dissonance_glyph.value, win)
        history = tuple(_structural_label(item) for item in nd.get("glyph_history", ()))
        has_recent_dissonance = any(entry == DISSONANCE for entry in history[-win:])
        if not has_recent_dissonance and norm_dn < dn_min:
            # Return dissonance as fallback - structural requirement for mutation
            # Maintains TNFR invariant: mutation requires prior dissonance (§3.4 operator closure)
            return dissonance_glyph
    return cand


def _check_thol_closure(
    ctx: "GrammarContext", n, cand: Glyph | str, st: dict[str, Any]
) -> Glyph | str:
    """Close THOL blocks with canonical glyphs once stabilised."""

    nd = ctx.G.nodes[n]
    if st.get("thol_open", False):
        glyph_to_name, name_to_glyph = _functional_translators()
        cand_glyph = coerce_glyph(cand)
        cand_name = glyph_to_name(cand_glyph if isinstance(cand_glyph, Glyph) else cand)

        # Allow nested THOL (self_organization) blocks without incrementing length
        # TNFR invariant: operational fractality (§3.7)
        if cand_name == SELF_ORGANIZATION:
            return cand

        st["thol_len"] = int(st.get("thol_len", 0)) + 1
        cfg = ctx.cfg_canon
        minlen = int(cfg.get("thol_min_len", 2))
        maxlen = int(cfg.get("thol_max_len", 6))
        close_dn = float(cfg.get("thol_close_dnfr", 0.15))
        requires_close = st["thol_len"] >= maxlen or (
            st["thol_len"] >= minlen and normalized_dnfr(ctx, nd) <= close_dn
        )
        if requires_close:
            si_high = float(cfg.get("si_high", 0.66))
            si = _si(nd)
            target_name = SILENCE if si >= si_high else CONTRACTION
            target_glyph = name_to_glyph(target_name)

            if cand_name == target_name and isinstance(cand_glyph, Glyph):
                return cand_glyph

            if target_glyph is not None and cand_name in {CONTRACTION, SILENCE}:
                return target_glyph

            history = tuple(
                _structural_label(item) for item in nd.get("glyph_history", ())
            )
            cand_label = cand_name if cand_name is not None else _structural_label(cand)
            order = (*history[-st["thol_len"] :], cand_label)
            from ..operators import grammar as _grammar

            raise _grammar.TholClosureError(
                rule="thol-closure",
                candidate=cand_label,
                message=(
                    f"{operator_display_name(SELF_ORGANIZATION)} block requires {operator_display_name(target_name)} closure"
                ),
                window=st["thol_len"],
                threshold=close_dn,
                order=order,
                context={
                    "thol_min_len": minlen,
                    "thol_max_len": maxlen,
                    "si": si,
                    "si_high": si_high,
                    "required_closure": target_name,
                },
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
        glyph_to_name, name_to_glyph = _functional_translators()
        compat, fallback = _structural_tables()
        prev_name = glyph_to_name(prev_glyph)
        if prev_name is None:
            return cand
        allowed = compat.get(prev_name)
        if allowed is None:
            return cand
        cand_name = glyph_to_name(cand_glyph if isinstance(cand_glyph, Glyph) else cand)
        if cand_name in allowed:
            return cand
        fb_name = fallback.get(prev_name)
        if fb_name is not None:
            fb_glyph = name_to_glyph(fb_name)
            if fb_glyph is not None:
                return fb_glyph
        order = (prev_name, cand_name or str(cand))
        from ..operators import grammar as _grammar

        raise _grammar.TransitionCompatibilityError(
            rule="transition-compatibility",
            candidate=cand_name or str(cand),
            message=f"{cand_name or cand} incompatible after {prev_name}",
            order=order,
        )
    return cand


@lru_cache(maxsize=1)
def _functional_translators():
    from ..operators import grammar as _grammar

    return _grammar.glyph_function_name, _grammar.function_name_to_glyph


@lru_cache(maxsize=1)
def _structural_tables():
    from . import compatibility as _compat

    return _compat._STRUCTURAL_COMPAT_TABLE, _compat._STRUCTURAL_FALLBACK_TABLE
