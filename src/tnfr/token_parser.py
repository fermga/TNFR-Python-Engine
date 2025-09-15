"""Shared helper for parsing tokens."""

from __future__ import annotations

from typing import Any, Callable, Iterator

from .collections_utils import flatten_structure
from .token_map import TOKEN_MAP, parse_thol

__all__ = ("validate_token", "_parse_tokens", "parse_thol", "TOKEN_MAP")


def validate_token(
    tok: Any,
    pos: int,
    token_map: dict[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Validate a token and wrap handler errors with context.

    The handler retrieved from ``token_map`` may raise ``KeyError``,
    ``ValueError`` or ``TypeError``. These exceptions are intercepted and
    re-raised as :class:`ValueError` with additional positional context.
    """

    token_map = TOKEN_MAP if token_map is None else token_map
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
    obj: Any, token_map: dict[str, Callable[[Any], Any]] | None = None
) -> Iterator[Any]:
    """Yield validated tokens from ``obj`` lazily."""

    token_map = TOKEN_MAP if token_map is None else token_map
    for pos, tok in enumerate(flatten_structure(obj), start=1):
        yield validate_token(tok, pos, token_map)
