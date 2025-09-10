"""Shared helper for parsing tokens."""

from __future__ import annotations

from typing import Any, Callable
from collections import deque
from collections.abc import Sequence

__all__ = ("_flatten_tokens", "validate_token", "_parse_tokens")


def _flatten_tokens(obj: Any):
    """Yield each token in order using a deque for clarity."""

    stack = deque([obj])
    while stack:
        item = stack.pop()
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            stack.extend(reversed(item))
        else:
            yield item


def validate_token(
    tok: Any, pos: int, token_map: dict[str, Callable[[Any], Any]]
) -> Any:
    """Validate a token and wrap handler errors with context.

    The handler retrieved from ``token_map`` may raise ``KeyError``,
    ``ValueError`` or ``TypeError``. These exceptions are intercepted and
    re-raised as :class:`ValueError` with additional positional context.
    """

    tok_info = f"(pos {pos}, token {tok!r})"

    if isinstance(tok, dict):
        if len(tok) != 1:
            raise ValueError(f"Invalid token: {tok} (pos {pos})")
        key, val = next(iter(tok.items()))
        handler = token_map.get(key)
        if handler is None:
            raise ValueError(f"Unrecognized token: {key} {tok_info}")
        try:
            return handler(val)
        except (KeyError, ValueError, TypeError) as e:
            msg = f"{type(e).__name__}: {e} {tok_info}"
            raise ValueError(msg) from e
    if isinstance(tok, str):
        return tok
    raise ValueError(f"Invalid token: {tok} (pos {pos})")


def _parse_tokens(
    obj: Any, token_map: dict[str, Callable[[Any], Any]]
) -> list[Any]:
    return [
        validate_token(tok, pos, token_map)
        for pos, tok in enumerate(_flatten_tokens(obj), start=1)
    ]
