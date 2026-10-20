"""Validation helpers grouped by rule type.

These utilities retain shared scalar observations and explicit fallback
configuration. Word and live-state admission belong to the operator grammar
owners; no competing mutation or block-closure selector is defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_SI
from ..metrics.common import normalize_dnfr
from ..types import Glyph
from ..utils import clamp01
from ..utils.numeric import _clipped_ratio

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from ..operators.grammar import GrammarContext

__all__ = [
    "coerce_glyph",
    "glyph_fallback",
    "get_norm",
    "normalized_dnfr",
    "_norm_attr",
    "_si",
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
    """Apply an explicit fallback override, or retain the candidate.

    This configured substitution does not establish word admission or satisfy
    an operator's live preconditions.
    """

    candidate = _admitted_glyph(cand_key, "candidate glyph")
    if not isinstance(fallbacks, Mapping):
        raise TypeError("GRAMMAR.fallbacks must be a mapping")
    if cand_key in fallbacks:
        return _admitted_glyph(fallbacks[cand_key], "configured fallback glyph")
    if candidate.value in fallbacks:
        return _admitted_glyph(fallbacks[candidate.value], "configured fallback glyph")

    # No configured substitution; admission remains a separate check.
    return candidate


def _admitted_glyph(value: Any, label: str) -> Glyph:
    """Resolve an identifier through the shared operator registry boundary."""
    from ..operators.grammar_debt import canonical_operator_code

    try:
        return Glyph(canonical_operator_code(value))
    except ValueError as exc:
        raise ValueError(f"{label} must identify a registered operator") from exc


# -------------------------
# Normalisation helpers
# -------------------------


def get_norm(ctx: "GrammarContext", key: str) -> float:
    """Admit the supplied scalar norm, with absent/zero values using one.

    Each consumer retains its sign policy: pressure requires a nonnegative
    maximum, while the generic clipped ratio retains signed denominators.
    This reader does not refresh a global maximum from the graph.
    """

    if not isinstance(ctx.norms, Mapping):
        raise TypeError("grammar norms must be a mapping")
    return finite_represented_real(ctx.norms.get(key, 1.0), key)[0] or 1.0


def _norm_attr(
    ctx: "GrammarContext", nd, attr_alias: str | tuple[str, ...], norm_key: str
) -> float:
    """Normalise ``attr_alias`` using the global maximum ``norm_key``."""

    max_val = get_norm(ctx, norm_key)
    value = get_attr(nd, attr_alias, 0.0, strict=True, conv=lambda item: item)
    return _clipped_ratio(value, max_val, absolute=True)


def _si(nd) -> float:
    """Return the structural sense index for ``nd`` clamped to ``[0, 1]``."""

    value = get_attr(nd, ALIAS_SI, 0.5, strict=True, conv=lambda item: item)
    return clamp01(value)


def normalized_dnfr(ctx: "GrammarContext", nd) -> float:
    """Normalise |ΔNFR| using the configured global maximum."""

    return normalize_dnfr(nd, get_norm(ctx, "dnfr_max"))
