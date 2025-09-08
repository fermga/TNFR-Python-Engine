from __future__ import annotations

from typing import Any, Callable

from ..program import block, wait, target
from ..types import Glyph
from ..token_parser import (
    validate_token as _tp_validate_token,
    _parse_tokens as _tp_parse_tokens,
    _flatten_tokens as _tp_flatten_tokens,
)


def validate_token(tok: Any, pos: int) -> Any:
    return _tp_validate_token(tok, pos, TOKEN_MAP)


def _parse_tokens(obj: Any) -> list[Any]:
    return _tp_parse_tokens(obj, TOKEN_MAP)


def _flatten_tokens(obj: Any):
    return _tp_flatten_tokens(obj)


def parse_thol(spec: dict[str, Any]) -> Any:
    """Parse the specification of a ``THOL`` block."""
    close = spec.get("close")
    if isinstance(close, str):
        if close not in Glyph.__members__:
            raise ValueError(f"Glyph de cierre desconocido: {close!r}")
        close = Glyph[close]

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
