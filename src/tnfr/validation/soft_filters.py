"""Soft grammar filters harmonising canonical selector heuristics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..alias import get_attr
from ..constants.aliases import ALIAS_D2EPI
from ..glyph_history import recent_glyph
from ..types import Glyph
from ..utils import clamp01
from .rules import glyph_fallback, get_norm, normalized_dnfr

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from collections.abc import Mapping
    from ..operators.grammar import GrammarContext

__all__ = (
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
)


def acceleration_norm(ctx: "GrammarContext", nd: "Mapping[str, Any]") -> float:
    """Return the node acceleration normalised to ``[0, 1]``.

    The computation uses the canonical ``accel_max`` bound stored in
    :class:`~tnfr.validation.GrammarContext`.  Values beyond the bound are
    clamped to preserve structural comparability with ΔNFR-based heuristics.
    """

    max_val = get_norm(ctx, "accel_max")
    return clamp01(abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / max_val)


def check_repeats(ctx: "GrammarContext", n: Any, cand: Glyph | str) -> Glyph | str:
    """Swap ``cand`` when it breaches the configured repetition window.

    The rule honours the soft grammar configuration stored in ``ctx.cfg_soft``:

    * ``window`` specifies the history length inspected using
      :func:`tnfr.glyph_history.recent_glyph`.
    * ``avoid_repeats`` enumerates glyph codes to dodge within that window.
    * ``fallbacks`` optionally map avoided glyph codes to explicit
      replacements, defaulting to canonical fallbacks when unspecified.
    """

    nd = ctx.G.nodes[n]
    cfg = ctx.cfg_soft
    gwin = int(cfg.get("window", 0))
    avoid = set(cfg.get("avoid_repeats", []))
    fallbacks = cfg.get("fallbacks", {})
    cand_key = cand.value if isinstance(cand, Glyph) else str(cand)
    if gwin > 0 and cand_key in avoid and recent_glyph(nd, cand_key, gwin):
        fallback = glyph_fallback(cand_key, fallbacks)
        fallback_key = fallback.value if isinstance(fallback, Glyph) else str(fallback)
        if fallback_key != cand_key:
            return fallback
        history: list[str] = []
        for item in nd.get("glyph_history", ()):
            if isinstance(item, Glyph):
                history.append(item.value)
            else:
                try:
                    history.append(Glyph(str(item)).value)
                except (TypeError, ValueError):
                    history.append(str(item))
        order = (*history[-gwin:], cand_key)
        from ..operators import grammar as _grammar

        def to_structural(value: Glyph | str) -> str:
            default = value.value if isinstance(value, Glyph) else str(value)
            result = _grammar.glyph_function_name(value, default=default)
            return default if result is None else result

        cand_name = to_structural(cand_key)
        fallback_name = to_structural(fallback_key)
        order_names = tuple(to_structural(item) for item in order)

        raise _grammar.RepeatWindowError(
            rule="repeat-window",
            candidate=cand_name,
            message=f"{cand_name} repeats within window {gwin}",
            window=gwin,
            order=order_names,
            context={"fallback": fallback_name},
        )
    return cand


def maybe_force(
    ctx: "GrammarContext",
    n: Any,
    cand: Glyph | str,
    original: Glyph | str,
    accessor: Callable[["GrammarContext", "Mapping[str, Any]"], float],
    key: str,
) -> Glyph | str:
    """Return ``original`` when ``accessor`` crosses the soft threshold ``key``.

    ``accessor`` receives the grammar context and node attributes.  Whenever the
    resulting score is greater than or equal to the configured threshold the
    original candidate is restored, ensuring soft filters never override
    high-confidence canonical choices.
    """

    if cand == original:
        return cand
    force_th = float(ctx.cfg_soft.get(key, 0.60))
    if accessor(ctx, ctx.G.nodes[n]) >= force_th:
        return original
    return cand


def _match_template(result: Glyph | str, template: Glyph | str) -> Glyph | str:
    """Coerce ``result`` to mirror ``template``'s representation when possible."""

    if isinstance(template, str) and isinstance(result, Glyph):
        return result.value
    if isinstance(template, Glyph) and isinstance(result, str):
        try:
            return Glyph(result)
        except (TypeError, ValueError):
            return result
    return result


def soft_grammar_filters(
    ctx: "GrammarContext",
    n: Any,
    cand: Glyph | str,
    *,
    original: Glyph | str | None = None,
    template: Glyph | str | None = None,
) -> Glyph | str:
    """Apply the canonical soft grammar pipeline for ``cand``.

    The pipeline performs three ordered checks:

    1. :func:`check_repeats` swaps recent repetitions for the configured
       fallback glyphs.
    2. :func:`maybe_force` with :func:`tnfr.validation.rules.normalized_dnfr`
       restores the original glyph when ΔNFR already exceeds ``force_dnfr``.
    3. :func:`maybe_force` with :func:`acceleration_norm` performs the same
       safeguard using the ``force_accel`` threshold.

    Parameters
    ----------
    ctx:
        Active grammar context providing node access and configuration.
    n:
        Node identifier inside ``ctx.G``.
    cand:
        Candidate glyph (``Glyph`` or code string) to validate softly.
    original:
        Optional glyph used as the reference for :func:`maybe_force`.  When
        omitted, ``cand`` before filtering is used as the anchor value.
    template:
        Optional value whose type will be preserved in the returned result.  By
        default the original ``cand`` representation is used.
    """

    anchor = cand if original is None else original
    filtered = check_repeats(ctx, n, cand)
    filtered = maybe_force(ctx, n, filtered, anchor, normalized_dnfr, "force_dnfr")
    filtered = maybe_force(ctx, n, filtered, anchor, acceleration_norm, "force_accel")
    base_template = cand if template is None else template
    return _match_template(filtered, base_template)
