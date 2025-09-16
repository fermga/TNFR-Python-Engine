"""Backward compatible façade for the TNFR programming helpers."""

from __future__ import annotations

from .execution import (
    AdvanceFn,
    HANDLERS,
    _apply_glyph_to_targets,
    _handle_glyph,
    _handle_target,
    _handle_thol,
    _handle_wait,
    _record_trace,
    basic_canonical_example,
    block,
    get_step_fn,
    play,
    seq,
    target,
    wait,
)
from .flatten import (
    THOLEvaluator,
    _flatten,
    _flatten_glyph,
    _flatten_target,
    _flatten_wait,
)
from .token_parser import validate_token
from .tokens import OpTag, TARGET, THOL, WAIT, THOL_SENTINEL, Token

__all__ = (
    "WAIT",
    "TARGET",
    "THOL",
    "THOLEvaluator",
    "OpTag",
    "Token",
    "THOL_SENTINEL",
    "validate_token",
    "get_step_fn",
    "seq",
    "block",
    "target",
    "wait",
    "play",
    "basic_canonical_example",
)
