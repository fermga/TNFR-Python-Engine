from ..types import Glyph

__all__ = ("CANON_COMPAT", "CANON_FALLBACK")

CANON_COMPAT: dict[Glyph, set[Glyph]]
CANON_FALLBACK: dict[Glyph, Glyph]
