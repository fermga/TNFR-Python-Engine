"""Configured repeat avoidance and threshold restoration for glyph proposals."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, Callable

from .._exact_time import bounded_represented_real
from ..constants.aliases import ALIAS_D2EPI
from ..types import Glyph
from .rules import _admitted_glyph, _norm_attr, glyph_fallback, normalized_dnfr
from .window import validate_window

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..operators.grammar import GrammarContext

__all__ = (
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
)


def _soft_config(ctx: "GrammarContext") -> Mapping[str, Any]:
    """Read the supplied soft policy without replacing malformed containers."""
    if not isinstance(ctx.cfg_soft, Mapping):
        raise TypeError("GRAMMAR soft configuration must be a mapping")
    return ctx.cfg_soft


def acceleration_norm(ctx: "GrammarContext", nd: "Mapping[str, Any]") -> float:
    """Return the node acceleration normalised to ``[0, 1]``.

    The computation uses the supplied ``accel_max`` scale stored in
    :class:`~tnfr.validation.GrammarContext`, with its unit-default convention.
    Values beyond a positive scale saturate at one; this is a policy score.
    """

    return _norm_attr(ctx, nd, ALIAS_D2EPI, "accel_max")


def check_repeats(ctx: "GrammarContext", n: Any, cand: Glyph | str) -> Glyph | str:
    """Swap ``cand`` when it breaches the configured repetition window.

    The rule honours the soft grammar configuration stored in ``ctx.cfg_soft``:

    * ``window`` specifies the recent suffix of a freshly admitted history.
    * ``avoid_repeats`` enumerates glyph codes to dodge within that window.
    * ``fallbacks`` optionally map avoided glyph codes to explicit registered
      replacements. An absent substitution retains the candidate and a repeat
      with no distinct replacement raises ``RepeatWindowError``.
    """

    nd = ctx.G.nodes[n]
    cfg = _soft_config(ctx)
    gwin = validate_window(cfg.get("window", 0))
    raw_avoid = cfg.get("avoid_repeats", ())
    if not isinstance(raw_avoid, Collection) or isinstance(
        raw_avoid, (str, bytes, bytearray, Mapping)
    ):
        raise TypeError("GRAMMAR.avoid_repeats must be a replayable glyph collection")
    avoid = {_admitted_glyph(value, "avoided glyph").value for value in raw_avoid}
    fallbacks = cfg.get("fallbacks", {})
    if not isinstance(fallbacks, Mapping):
        raise TypeError("GRAMMAR.fallbacks must be a mapping")
    cand_key = _admitted_glyph(cand, "candidate glyph").value
    if gwin <= 0 or cand_key not in avoid:
        return cand

    from ..operators.grammar_debt import read_grammar_state

    history = read_grammar_state(nd).history
    if cand_key in history[-gwin:]:
        fallback = glyph_fallback(cand_key, fallbacks)
        fallback_key = fallback.value if isinstance(fallback, Glyph) else str(fallback)
        if fallback_key != cand_key:
            return fallback
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
    original candidate is restored. Threshold and consumed score must be
    represented reals in [0, 1]; malformed configured thresholds reject even
    when no replacement is needed. This policy supplies neither word admission
    nor the operator's live preconditions.
    """

    force_th = bounded_represented_real(
        _soft_config(ctx).get(key, 0.60), f"GRAMMAR.{key}", minimum=0.0, maximum=1.0
    )[0]
    candidate = _admitted_glyph(cand, "candidate glyph")
    anchor = _admitted_glyph(original, "original glyph")
    if candidate is anchor:
        return cand
    score = bounded_represented_real(
        accessor(ctx, ctx.G.nodes[n]),
        f"{key} normalized score",
        minimum=0.0,
        maximum=1.0,
    )[0]
    if score >= force_th:
        return original
    return cand


def _match_template(result: Glyph | str, template: Glyph | str) -> Glyph | str:
    """Coerce ``result`` to mirror ``template``'s representation when possible."""

    admitted = _admitted_glyph(result, "filtered glyph")
    # Glyph subclasses str, so its representation must be checked first.
    if isinstance(template, Glyph):
        return admitted
    if isinstance(template, str):
        return admitted.value
    raise TypeError("glyph template must be a Glyph or string")


def soft_grammar_filters(
    ctx: "GrammarContext",
    n: Any,
    cand: Glyph | str,
    *,
    original: Glyph | str | None = None,
    template: Glyph | str | None = None,
) -> Glyph | str:
    """Apply the configured soft grammar pipeline for ``cand``.

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
