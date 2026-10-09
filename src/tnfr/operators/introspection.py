"""Operator introspection metadata (Phase 3).

Provides a lightweight, immutable metadata registry describing each
canonical structural operator's physics category, grammar roles, and
contracts for tooling (telemetry enrichment, validation messaging,
documentation generation).

Design Constraints
------------------
1. Read-only: No mutation of operator classes or graph state.
2. Traceability: Grammar roles reference U1-U5 identifiers verbatim.
3. Fidelity: Shared fields derive from operator_contracts and grammar_canon.
4. Backward compatibility: Optional; absence of this module should not
   break existing imports.

Public API
----------
get_operator_meta(name_or_glyph) -> OperatorMeta
iter_operator_meta() -> iterator[OperatorMeta]
OPERATOR_METADATA: dict[str, OperatorMeta]

Fields
------
OperatorMeta.name          Title-case display/class name (e.g. Emission)
OperatorMeta.mnemonic      Glyph code (AL, EN, ...)
OperatorMeta.category      High-level functional category
OperatorMeta.grammar_roles list of grammar rule roles (U1a, U1b, U2, ...)
OperatorMeta.contracts     Short, stable contract statements
OperatorMeta.doc           Canonical purpose from operator_contracts

Note: Grammar rule U6 (confinement) is telemetry-only and not included
as an active role. The ``grammar_roles`` tuples are the canonical per-operator
U1-U5 roles derived from the operator's role set in
:mod:`tnfr.operators.grammar_canon` (single source of truth); the agreement is
pinned by ``test_grammar_canon.py`` (``u_rules_for_operator``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

__all__ = [
    "OperatorMeta",
    "OPERATOR_METADATA",
    "get_operator_meta",
    "iter_operator_meta",
]


@dataclass(frozen=True, slots=True)
class OperatorMeta:
    name: str
    mnemonic: str
    category: str
    grammar_roles: tuple[str, ...]
    contracts: tuple[str, ...]
    doc: str


# These legacy presentation categories are distinct from canonical U-rule roles.
# Preserve their labels and iteration order while deriving every shared field.
_CATEGORIES = {
    "AL": "generator",
    "EN": "integrator",
    "IL": "stabilizer",
    "OZ": "destabilizer",
    "UM": "coupling",
    "RA": "propagation",
    "SHA": "closure",
    "VAL": "destabilizer",
    "NUL": "simplifier",
    "THOL": "stabilizer",
    "ZHIR": "transformer",
    "NAV": "generator",
    "REMESH": "generator",
}


def _build_operator_metadata() -> dict[str, OperatorMeta]:
    from .grammar_canon import u_rules_for_operator
    from .operator_contracts import contract_for

    table = {}
    for mnemonic, category in _CATEGORIES.items():
        spec = contract_for(mnemonic)
        table[mnemonic] = OperatorMeta(
            name=spec.english_name,
            mnemonic=spec.glyph,
            category=category,
            grammar_roles=u_rules_for_operator(spec.name),
            contracts=(spec.postcondition,),
            doc=spec.purpose,
        )
    return table


OPERATOR_METADATA: Mapping[str, OperatorMeta] = _build_operator_metadata()


def get_operator_meta(identifier: str) -> OperatorMeta:
    """Return metadata for an internal glyph or title-case display/class name.

    Resolution order:
    1. Exact mnemonic key (AL, EN, ...)
    2. Search by title-case display/class name (Emission, Coherence, ...)
    Raises KeyError if not found.
    """

    # Direct mnemonic
    meta = OPERATOR_METADATA.get(identifier)
    if meta is not None:
        return meta
    # Title-case display/class-name lookup
    for m in OPERATOR_METADATA.values():
        if m.name == identifier:
            return m
    raise KeyError(identifier)


def iter_operator_meta() -> Iterator[OperatorMeta]:
    """Iterate all operator metadata objects."""
    return iter(OPERATOR_METADATA.values())
