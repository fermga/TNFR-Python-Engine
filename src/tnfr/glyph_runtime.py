"""Runtime helpers for structural operator glyph provenance.

Glyph history remains the authoritative ordered trace. The source_glyph field
and its legacy alias last_glyph provide a single-value provenance fallback for
serialized or migrated nodes without a history. Neither field is structural EPI
identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .constants.aliases import ALIAS_SOURCE_GLYPH

__all__ = ("last_glyph",)


def last_glyph(nd: Mapping[str, Any]) -> str | None:
    """Return a canonical glyph code from admitted history or provenance.

    A nonempty glyph_history takes precedence. When no history is available,
    source_glyph/last_glyph is read through its dedicated alias tuple.
    Malformed stored history rejects before any provenance fallback. Neither
    fallback metadata nor this observation establishes an executed occurrence.
    epi_kind is deliberately outside this lookup.
    """

    from .operators.grammar_debt import canonical_operator_code, read_grammar_state

    history = read_grammar_state(nd).history
    if history:
        return history[-1]
    value = next((nd[key] for key in ALIAS_SOURCE_GLYPH if key in nd), None)
    return None if value is None or value == "" else canonical_operator_code(value)
