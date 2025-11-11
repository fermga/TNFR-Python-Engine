"""Encoded legacy tokens for the Spanish language guard."""

from __future__ import annotations

from base64 import b64decode
from typing import Iterable, Sequence, Any

LEGACY_KEYWORD_CODEPOINTS: tuple[tuple[int, ...], ...] = (
    (101, 115, 116, 97, 98, 108, 101),
    (116, 114, 97, 110, 115, 105, 99, 105, 111, 110),
    (116, 114, 97, 110, 115, 105, 99, 105, 243, 110),
    (100, 105, 115, 111, 110, 97, 110, 116, 101),
    (111, 112, 101, 114, 97, 100, 111, 114, 101, 115),
    (111, 112, 101, 114, 97, 100, 111, 114),
    (101, 106, 101, 109, 112, 108, 111),
    (111, 112, 99, 105, 111, 110, 97, 108, 101, 115),
    (100, 101, 112, 101, 110, 100, 101, 110, 99, 105, 97),
    (99, 111, 109, 112, 97, 116, 105, 98, 105, 108, 105, 100, 97, 100),
    (118, 97, 108, 111, 114, 101, 115),
    (100, 101, 98, 101),
    (114, 101, 99, 111, 109, 112, 117, 116, 97, 114),
    (109, 111, 116, 111, 114),
    (112, 111, 114, 95, 100, 101, 102, 101, 99, 116, 111),
)

ACCENT_CODEPOINTS: tuple[int, ...] = (
    225,
    233,
    237,
    243,
    250,
    252,
    241,
    193,
    201,
    205,
    211,
    218,
    220,
    209,
    191,
    161,
)


def _coerce_code_sequence(candidate: Any) -> tuple[int, ...] | None:
    """Return a tuple of integers from ``candidate`` when possible."""

    if isinstance(candidate, (str, bytes, bytearray)):
        return None
    if isinstance(candidate, int):
        return (candidate,)
    try:
        return tuple(int(part) for part in candidate)
    except (TypeError, ValueError):
        return None


def decode_keyword_codes(
    encoded: Iterable[Iterable[int] | Sequence[int]],
) -> tuple[str, ...]:
    """Decode an iterable of integer sequences into keyword strings."""

    tokens: list[str] = []
    for item in encoded:
        codepoints = _coerce_code_sequence(item)
        if codepoints is None:
            continue
        tokens.append("".join(chr(codepoint) for codepoint in codepoints))
    return tuple(tokens)


def decode_keyword_base64(encoded: Iterable[str]) -> tuple[str, ...]:
    """Decode base64-encoded keyword strings."""

    tokens: list[str] = []
    for item in encoded:
        if not isinstance(item, str):
            continue
        try:
            tokens.append(b64decode(item).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
    return tuple(tokens)


def decode_accent_codepoints(encoded: Iterable[int]) -> tuple[str, ...]:
    """Decode integer code points representing accented characters."""

    characters: list[str] = []
    for item in encoded:
        try:
            characters.append(chr(int(item)))
        except (TypeError, ValueError):
            continue
    return tuple(characters)


def default_disallowed_keywords() -> tuple[str, ...]:
    """Return the decoded default disallowed keywords."""

    return decode_keyword_codes(LEGACY_KEYWORD_CODEPOINTS)


def default_accented_characters() -> tuple[str, ...]:
    """Return the decoded default accented characters."""

    return decode_accent_codepoints(ACCENT_CODEPOINTS)


__all__ = [
    "ACCENT_CODEPOINTS",
    "LEGACY_KEYWORD_CODEPOINTS",
    "decode_accent_codepoints",
    "decode_keyword_base64",
    "decode_keyword_codes",
    "default_accented_characters",
    "default_disallowed_keywords",
]
