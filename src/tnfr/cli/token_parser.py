from __future__ import annotations

from typing import Any, Callable
from functools import partial

from ..program import block, wait, target
from ..types import Glyph
from ..token_parser import (
    validate_token as _tp_validate_token,
    _parse_tokens as _tp_parse_tokens,
    _flatten_tokens as _tp_flatten_tokens,
)

__all__ = (
    "validate_token",
    "_parse_tokens",
    "_flatten_tokens",
    "parse_thol",
    "TOKEN_MAP",
)


def parse_thol(spec: dict[str, Any]) -> Any:
    """Parse the specification of a ``THOL`` block."""
    close = spec.get("close")
    if isinstance(close, str):
        close_enum = Glyph.__members__.get(close)
        if close_enum is None:
            raise ValueError(f"Glyph de cierre desconocido: {close!r}")
        close = close_enum

    return block(
        *_parse_tokens(spec.get("body", [])),
        repeat=int(spec.get("repeat", 1)),
        close=close,
    )


TOKEN_MAP: dict[str, Callable[[Any], Any]] = {
    "WAIT": lambda v: wait(int(v)),
    "TARGET": lambda v: target(v),
    "THOL": parse_thol,
}


validate_token = partial(_tp_validate_token, token_map=TOKEN_MAP)
_parse_tokens = partial(_tp_parse_tokens, token_map=TOKEN_MAP)
_flatten_tokens = _tp_flatten_tokens
