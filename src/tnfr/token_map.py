from __future__ import annotations

from typing import Any, Callable

from .types import Glyph

__all__ = ("TOKEN_MAP", "parse_thol")


def parse_thol(spec: dict[str, Any]) -> Any:
    """Parse the specification of a ``THOL`` block."""
    from .execution import block
    from .token_parser import _parse_tokens

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


def _wait_handler(v: Any) -> Any:
    from .execution import wait

    return wait(int(v))


def _target_handler(v: Any) -> Any:
    from .execution import target

    return target(v)


TOKEN_MAP: dict[str, Callable[[Any], Any]] = {
    "WAIT": _wait_handler,
    "TARGET": _target_handler,
    "THOL": parse_thol,
}
