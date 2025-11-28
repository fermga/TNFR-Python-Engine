"""Operator introspection metadata (Phase 3).

Provides a lightweight, immutable metadata registry describing each
canonical structural operator's physics category, grammar roles, and
contracts for tooling (telemetry enrichment, validation messaging,
documentation generation).

Design Constraints
------------------
1. Read-only: No mutation of operator classes or graph state.
2. Traceability: Grammar roles reference U1-U4 identifiers verbatim.
3. Fidelity: Contracts reflect AGENTS.md canonical operator summaries.
4. Backward compatibility: Optional; absence of this module should not
   break existing imports.

Public API
----------
get_operator_meta(name_or_glyph) -> OperatorMeta
iter_operator_meta() -> iterator[OperatorMeta]
OPERATOR_METADATA: dict[str, OperatorMeta]

Fields
------
OperatorMeta.name          English class name (e.g. Emission)
OperatorMeta.mnemonic      Glyph code (AL, EN, ...)
OperatorMeta.category      High-level functional category
OperatorMeta.grammar_roles List of grammar rule roles (U1a, U1b, U2, ...)
OperatorMeta.contracts     Short, stable contract statements
OperatorMeta.doc           Concise physics rationale (1-2 sentences)

Note: Grammar rule U6 (confinement) is telemetry-only and not included
as an active role.
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


OPERATOR_METADATA: Mapping[str, OperatorMeta] = {
    # Generators ---------------------------------------------------------
    "AL": OperatorMeta(
        name="Emission",
        mnemonic="AL",
        category="generator",
        grammar_roles=("U1a",),
        contracts=(
            "Initialises νf",
            "Positive ΔNFR",
            "Irreversible activation",
        ),
        doc="Starts coherent emission; begins structural reorganization.",
    ),
    "EN": OperatorMeta(
        name="Reception",
        mnemonic="EN",
        category="integrator",
        grammar_roles=(),
        contracts=("Integrates incoming resonance", "Does not reduce C(t)"),
        doc="Integrates external resonance without coherence loss.",
    ),
    "IL": OperatorMeta(
        name="Coherence",
        mnemonic="IL",
        category="stabilizer",
        grammar_roles=("U2", "U4a"),
        contracts=(
            "Reduces |ΔNFR|",
            "Monotonic C(t) unless test",
            "Bifurcation handler",
        ),
        doc="Negative feedback preserving bounded evolution and coherence.",
    ),
    "OZ": OperatorMeta(
        name="Dissonance",
        mnemonic="OZ",
        category="destabilizer",
        grammar_roles=("U2", "U4a"),
        contracts=(
            "Increases |ΔNFR|",
            "May trigger bifurcation",
            "Needs IL/THOL handler",
        ),
        doc="Controlled instability elevating structural pressure.",
    ),
    "UM": OperatorMeta(
        name="Coupling",
        mnemonic="UM",
        category="coupling",
        grammar_roles=("U3",),
        contracts=("Phase compatibility", "Establishes link"),
        doc="Phase-sync enabling resonance exchange.",
    ),
    "RA": OperatorMeta(
        name="Resonance",
        mnemonic="RA",
        category="propagation",
        grammar_roles=("U3",),
        contracts=("Amplifies identity", "Phase compatibility"),
        doc="Propagates coherent pattern maintaining identity.",
    ),
    "SHA": OperatorMeta(
        name="Silence",
        mnemonic="SHA",
        category="closure",
        grammar_roles=("U1b",),
        contracts=("νf→0 temporary", "Preserves EPI"),
        doc="Freezes evolution for observation window.",
    ),
    "VAL": OperatorMeta(
        name="Expansion",
        mnemonic="VAL",
        category="destabilizer",
        grammar_roles=("U2",),
        contracts=("Raises dimensionality", "Needs stabilizer"),
        doc="Adds degrees of freedom increasing complexity.",
    ),
    "NUL": OperatorMeta(
        name="Contraction",
        mnemonic="NUL",
        category="simplifier",
        grammar_roles=(),
        contracts=("Reduces dimensionality", "Aids stabilization"),
        doc="Simplifies complexity by removing degrees of freedom.",
    ),
    "THOL": OperatorMeta(
        name="SelfOrganization",
        mnemonic="THOL",
        category="stabilizer",
        grammar_roles=("U2", "U4a", "U4b"),
        contracts=(
            "Creates sub-EPIs",
            "Preserves form",
            "Bifurcation handler",
        ),
        doc="Autopoietic structuring creating fractal sub-forms.",
    ),
    "ZHIR": OperatorMeta(
        name="Mutation",
        mnemonic="ZHIR",
        category="transformer",
        grammar_roles=("U4a", "U4b"),
        contracts=(
            "Phase transform threshold",
            "Requires prior IL",
            "Recent destabilizer",
        ),
        doc="Threshold-driven phase change altering regime.",
    ),
    "NAV": OperatorMeta(
        name="Transition",
        mnemonic="NAV",
        category="generator",
        grammar_roles=("U1a", "U1b"),
        contracts=("Activates latent EPI", "Closes sequences"),
        doc="Regime shift navigating attractors.",
    ),
    "REMESH": OperatorMeta(
        name="Recursivity",
        mnemonic="REMESH",
        category="generator",
        grammar_roles=("U1a", "U1b"),
        contracts=("Cross-scale echoing", "Supports fractality"),
        doc="Echoes patterns across scales for memory/nesting.",
    ),
}


def get_operator_meta(identifier: str) -> OperatorMeta:
    """Return metadata for glyph mnemonic or class name.

    Resolution order:
    1. Exact mnemonic key (AL, EN, ...)
    2. Search by English name (Emission, Coherence, ...)
    Raises KeyError if not found.
    """

    # Direct mnemonic
    meta = OPERATOR_METADATA.get(identifier)
    if meta is not None:
        return meta
    # English name lookup
    for m in OPERATOR_METADATA.values():
        if m.name == identifier:
            return m
    raise KeyError(identifier)


def iter_operator_meta() -> Iterator[OperatorMeta]:
    """Iterate all operator metadata objects."""
    return iter(OPERATOR_METADATA.values())
