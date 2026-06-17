"""TNFR Canonical Grammar Specification — the single source of truth.

This module is the authoritative, physics-grounded, TNFR.pdf-anchored
specification of the grammar of the 13 structural operators. It does NOT
re-implement validation (that lives in :mod:`grammar_core` / :mod:`grammar_validate`)
and it does NOT re-define the classification sets (those are derived in
:mod:`tnfr.config.physics_derivation` and re-exported by :mod:`grammar_types`).
Instead it *materialises*, in one place, the canonical knowledge that was
previously scattered across modules and prose:

1. ``OPERATOR_ROLES`` — the per-operator grammatical role table (the 13 operators
   × their U1-U6 roles), derived directly from the nodal-equation predicates in
   :mod:`physics_derivation`. One query point instead of eight separate sets.

2. ``GRAMMAR_RULES`` — the U1-U6 rule registry as data (id, name, physics basis,
   operator sets involved, canonical invariant, TNFR.pdf reference). The
   declarative spec that the validator, the error factory, and the docs share.

3. ``STRUCTURAL_TYPOLOGY`` — the canonical structural typology from TNFR.pdf §2.3
   "Tabla comparativa de estructuras glíficas": the five structure shapes
   (LINEAR, BIFURCATED, FRACTAL, CYCLIC, HIERARCHICAL) with their combinator and
   Chomsky class (established in examples 143-144), required glyphs, activation
   conditions and common errors (TNFR.pdf "Validación estructural de las
   tipologías glíficas").

4. ``CANONICAL_GLYPHIC_FUNCTIONS`` — the canonical glyphic functions / macros from
   TNFR.pdf §2.3 "Tabla de funciones glíficas operativas" and "Macros glíficas".
   These are structural FRAGMENTS (words to compose), not standalone valid
   sequences (see example 143).

Theoretical anchor (TNFR.pdf §2.3.3 "Reglas sintácticas glíficas")
------------------------------------------------------------------
The PDF formalises the glyphic syntax with an "Esquema formal de sintaxis":

    Inicio válido (valid start):      AL, NAV   (+ REMESH as structural reactivator)
    Desarrollo necesario (develop):   IL, THOL, UM
    Transición opcional (optional):   OZ, ZHIR, REMESH
    Cierre requerido (required close): SHA, NUL

plus the rules: order is non-commutative (AL→IL ≠ IL→AL); ZHIR must be preceded
by OZ (no mutation without dissonance); brackets THOL[...] encapsulate nested
nodes; every coherent sequence closes with a latency/containment glyph; OZ
triggers bifurcation OZ→[ZHIR|NUL].

Theory↔engine note (NUL as closure)
-----------------------------------
TNFR.pdf lists ``NUL`` (contraction, "retorno al estado potencial") among the
required closures. The engine, deriving closures from the nodal equation, does
NOT treat NUL as a closure: contraction reduces dim(EPI) (removes degrees of
freedom) but does not force ∂EPI/∂t → 0 the way SILENCE (SHA) does. The engine's
``CLOSURES`` = {SHA, NAV, REMESH, OZ} is the physics-grounded set (see
``physics_derivation.achieves_operational_closure`` /
``can_stabilize_reorganization``). This module documents the PDF nuance without
overriding the physics derivation.

All of this is DERIVED, not hand-maintained: the role table is built by querying
the physics predicates, and a self-check (:func:`verify_canon_consistency`)
asserts that the materialised roles reproduce the canonical sets in
:mod:`grammar_types` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from ..config.physics_derivation import (
    achieves_operational_closure,
    can_activate_latent_epi,
    can_generate_epi_from_null,
    can_stabilize_reorganization,
    executes_bifurcation,
    handles_bifurcation,
    increases_structural_pressure,
    provides_negative_feedback,
    triggers_bifurcation,
)
from .grammar_types import (
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    CLOSURES,
    COUPLING_RESONANCE,
    DESTABILIZERS,
    GENERATORS,
    RECURSIVE_GENERATORS,
    STABILIZERS,
    TRANSFORMERS,
    FUNCTION_TO_GLYPH,
    StructuralPattern,
)

__all__ = [
    "GrammarRole",
    "OperatorGrammar",
    "OPERATOR_ROLES",
    "operator_grammar",
    "GrammarRule",
    "GRAMMAR_RULES",
    "rule",
    "StructuralType",
    "StructuralTypeSpec",
    "STRUCTURAL_TYPOLOGY",
    "ChomskyClass",
    "GlyphicFunction",
    "CANONICAL_GLYPHIC_FUNCTIONS",
    "FORMAL_SYNTAX_SCHEMA",
    "STRUCTURAL_PATTERN_TO_TYPE",
    "canonical_structural_type",
    "verify_canon_consistency",
]


# ===========================================================================
# 1. Per-operator grammatical role table (derived from physics_derivation)
# ===========================================================================


class GrammarRole(str, Enum):
    """The grammatical roles an operator can carry across U1-U6.

    Each role corresponds to a per-operator nodal-equation predicate in
    :mod:`physics_derivation`; an operator may carry several roles.
    """

    GENERATOR = "generator"          # U1a — can start (create/activate EPI)
    CLOSURE = "closure"              # U1b — can end (stabilize / close cycle)
    STABILIZER = "stabilizer"        # U2  — reduces |ΔNFR| (negative feedback)
    DESTABILIZER = "destabilizer"    # U2  — raises |ΔNFR| (positive feedback)
    COUPLING = "coupling"            # U3  — requires phase verification
    TRIGGER = "trigger"              # U4a — may push ∂²EPI/∂t² past τ
    HANDLER = "handler"              # U4a — absorbs a triggered bifurcation
    TRANSFORMER = "transformer"      # U4b — executes a threshold-gated bifurcation
    RECURSIVE = "recursive"          # U5  — echoes structure across scales


@dataclass(frozen=True)
class OperatorGrammar:
    """The complete grammatical role signature of one canonical operator."""

    name: str
    glyph: str
    roles: frozenset[GrammarRole]

    def has(self, role: GrammarRole) -> bool:
        return role in self.roles


def _derive_roles(op: str) -> frozenset[GrammarRole]:
    """Materialise an operator's roles from the nodal-equation predicates."""
    roles: set[GrammarRole] = set()
    if can_generate_epi_from_null(op) or can_activate_latent_epi(op):
        roles.add(GrammarRole.GENERATOR)
    if can_stabilize_reorganization(op) or achieves_operational_closure(op):
        roles.add(GrammarRole.CLOSURE)
    if provides_negative_feedback(op):
        roles.add(GrammarRole.STABILIZER)
    if increases_structural_pressure(op):
        roles.add(GrammarRole.DESTABILIZER)
    if op in COUPLING_RESONANCE:
        roles.add(GrammarRole.COUPLING)
    if triggers_bifurcation(op):
        roles.add(GrammarRole.TRIGGER)
    if handles_bifurcation(op):
        roles.add(GrammarRole.HANDLER)
    if executes_bifurcation(op):
        roles.add(GrammarRole.TRANSFORMER)
    if op in RECURSIVE_GENERATORS:
        roles.add(GrammarRole.RECURSIVE)
    return frozenset(roles)


def _glyph_of(op: str) -> str:
    glyph = FUNCTION_TO_GLYPH.get(op)
    return getattr(glyph, "name", str(op))


#: The 13 operators in canonical order (matches the nodal-equation operator set).
CANONICAL_ORDER: tuple[str, ...] = (
    EMISSION,
    RECEPTION,
    COHERENCE,
    DISSONANCE,
    COUPLING,
    RESONANCE,
    SILENCE,
    EXPANSION,
    CONTRACTION,
    SELF_ORGANIZATION,
    MUTATION,
    TRANSITION,
    RECURSIVITY,
)

#: The single materialised per-operator grammatical role table.
OPERATOR_ROLES: dict[str, OperatorGrammar] = {
    op: OperatorGrammar(name=op, glyph=_glyph_of(op), roles=_derive_roles(op))
    for op in CANONICAL_ORDER
}


def operator_grammar(op: str) -> OperatorGrammar:
    """Return the grammatical role signature of a canonical operator."""
    return OPERATOR_ROLES[op]


# ===========================================================================
# 2. The U1-U6 rule registry as data
# ===========================================================================


@dataclass(frozen=True)
class GrammarRule:
    """One canonical grammar rule (U1-U6), declared as data.

    The validator (:mod:`grammar_core`, :mod:`grammar_u6`) implements these; the
    error factory and the documentation reference them. This registry is the
    shared declarative description, anchored to the nodal equation and to the
    TNFR.pdf formal syntax (§2.3.3).
    """

    rule_id: str            # "U1a", "U2", "U4b", ...
    name: str
    physics: str            # the nodal-equation rationale
    operator_sets: tuple[str, ...]   # names of the classification sets it uses
    invariant: int          # canonical invariant 1-6 it maps to
    pdf_reference: str      # TNFR.pdf §2.3.3 anchor


GRAMMAR_RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        rule_id="U1a",
        name="Structural Initiation",
        physics="∂EPI/∂t is undefined at EPI=0; a generator must create or "
                "activate EPI from the null/latent state before evolution.",
        operator_sets=("GENERATORS",),
        invariant=1,
        pdf_reference="§2.3.3 'Esquema formal de sintaxis' — valid start: AL, "
                      "NAV (+ REMESH reactivator)",
    ),
    GrammarRule(
        rule_id="U1b",
        name="Structural Closure",
        physics="A coherent sequence must terminate in a stable attractor: "
                "either ∂EPI/∂t → 0 (silence) or an operational cycle close.",
        operator_sets=("CLOSURES",),
        invariant=1,
        pdf_reference="§2.3.3 'Cierre estructural' — close with a latency glyph "
                      "(SHA, NUL)",
    ),
    GrammarRule(
        rule_id="U2",
        name="Convergence & Boundedness",
        physics="∫νf·ΔNFR dt must converge: every destabilizer (raises |ΔNFR|) "
                "needs a stabilizer (reduces |ΔNFR|) or the integral diverges.",
        operator_sets=("DESTABILIZERS", "STABILIZERS"),
        invariant=1,
        pdf_reference="Compatibilidad entre glifos / Bifurcación y mutación",
    ),
    GrammarRule(
        rule_id="U3",
        name="Resonant Coupling",
        physics="Resonance requires phase compatibility |φᵢ - φⱼ| ≤ Δφ_max; "
                "antiphase coupling produces destructive interference.",
        operator_sets=("COUPLING_RESONANCE",),
        invariant=2,
        pdf_reference="§2.3.3 'Compatibilidad' — phase compatibility of "
                      "coupling operators",
    ),
    GrammarRule(
        rule_id="U4a",
        name="Bifurcation Dynamics — triggers need handlers",
        physics="∂²EPI/∂t² > τ (a bifurcation) must be absorbed by a handler "
                "or the cascade becomes chaotic.",
        operator_sets=("BIFURCATION_TRIGGERS", "BIFURCATION_HANDLERS"),
        invariant=4,
        pdf_reference="Bifurcación y mutación — OZ → [ZHIR / NUL]",
    ),
    GrammarRule(
        rule_id="U4b",
        name="Bifurcation Dynamics — transformers need context",
        physics="A threshold crossing needs elevated |ΔNFR|: a transformer "
                "(ZHIR/THOL) requires a recent destabilizer; ZHIR also a prior IL.",
        operator_sets=("TRANSFORMERS", "DESTABILIZERS"),
        invariant=4,
        pdf_reference="§2.3.3 'Compatibilidad entre glifos' — ZHIR must be "
                      "preceded by OZ (no mutation without dissonance)",
    ),
    GrammarRule(
        rule_id="U5",
        name="Multi-Scale Coherence",
        physics="Hierarchical coupling: nested EPIs need stabilizers at each "
                "scale so aggregate child reorganization stays bounded "
                "(C_parent ≥ α·Σ C_child).",
        operator_sets=("RECURSIVE_GENERATORS", "STABILIZERS"),
        invariant=3,
        pdf_reference="§2.3.3 'Agrupamiento y jerarquía' — THOL[...] nesting",
    ),
    GrammarRule(
        rule_id="U6",
        name="Structural Potential Confinement",
        physics="The emergent field Φ_s = Σ ΔNFR_j / d² stays confined: "
                "ΔΦ_s < φ ≈ 1.618 (golden-ratio harmonic confinement).",
        operator_sets=(),  # telemetry-based, not a sequence constraint
        invariant=5,
        pdf_reference="§2.3 'Validación estructural' — coherence thresholds",
    ),
)


def rule(rule_id: str) -> GrammarRule:
    """Look up a canonical grammar rule by id (e.g. ``"U4b"``)."""
    for r in GRAMMAR_RULES:
        if r.rule_id == rule_id:
            return r
    raise KeyError(f"Unknown grammar rule id: {rule_id!r}")


#: The TNFR.pdf §2.3.3 "Esquema formal de sintaxis" positions (theory anchor).
#: Quoted Spanish terms are verbatim citations of the source schema headers.
FORMAL_SYNTAX_SCHEMA: dict[str, tuple[str, ...]] = {
    "start": ("AL", "NAV", "REMESH"),       # valid start ("Inicio válido") + REMESH reactivator
    "development": ("IL", "THOL", "UM"),     # required development ("Desarrollo necesario")
    "optional_transition": ("OZ", "ZHIR", "REMESH"),  # optional transition ("Transición opcional")
    "closure": ("SHA", "NUL"),               # required closure ("Cierre requerido"); see NUL note
}


# ===========================================================================
# 3. The canonical structural typology (TNFR.pdf "Tabla comparativa")
# ===========================================================================


class ChomskyClass(str, Enum):
    """Chomsky-hierarchy class of a glyphic structure (examples 139-144)."""

    REGULAR = "regular"              # concatenation / union / Kleene star
    CONTEXT_FREE = "context_free"    # nesting (Dyck), THOL[...]


class StructuralType(str, Enum):
    """The five canonical glyphic structure types (TNFR.pdf "Tabla comparativa").

    This is the canonical structural typology — a sequence's shape, determinable
    from the operator stream alone (examples 143-144). It is distinct from the
    application *domain* of a sequence (therapeutic, educational, …), which is a
    separate axis tracked by the ``domain`` metadata field.
    """

    LINEAR = "linear"            # Lineal — simple concatenation, latency close
    BIFURCATED = "bifurcated"    # Bifurcada — OZ → [ZHIR | NUL] branch (union)
    FRACTAL = "fractal"          # Fractal — self-similar repeat (Kleene star)
    CYCLIC = "cyclic"            # Cíclica — close-and-reopen feedback cycle
    HIERARCHICAL = "hierarchical"  # Jerárquica — nested THOL[...] (Dyck/CF)
    UNKNOWN = "unknown"          # not a recognised canonical structure


@dataclass(frozen=True)
class StructuralTypeSpec:
    """Canonical metadata for one structural type (TNFR.pdf §2.3)."""

    type: StructuralType
    pdf_term: str                   # verbatim term from TNFR.pdf (Spanish source)
    combinator: str                 # concatenation / union / star / nesting
    chomsky_class: ChomskyClass
    example: tuple[str, ...]        # canonical glyphic example
    required_glyphs: tuple[str, ...]
    activation_condition: str       # English (paraphrase of the PDF condition)
    common_error: str               # English (paraphrase of the PDF error)
    pdf_reference: str              # verbatim section-title citation


STRUCTURAL_TYPOLOGY: dict[StructuralType, StructuralTypeSpec] = {
    StructuralType.LINEAR: StructuralTypeSpec(
        type=StructuralType.LINEAR,
        pdf_term="Lineal",
        combinator="concatenation",
        chomsky_class=ChomskyClass.REGULAR,
        example=("AL", "IL", "RA", "SHA"),
        required_glyphs=("AL", "IL", "RA", "SHA"),
        activation_condition="νf > ν0 with initial coherence θ_min",
        common_error="Missing closure or stabilization",
        pdf_reference="Tabla comparativa de estructuras glíficas — Lineal",
    ),
    StructuralType.BIFURCATED: StructuralTypeSpec(
        type=StructuralType.BIFURCATED,
        pdf_term="Bifurcada",
        combinator="union (alternation)",
        chomsky_class=ChomskyClass.REGULAR,
        example=("OZ", "ZHIR"),  # OZ → [ZHIR | NUL]
        required_glyphs=("OZ",),
        activation_condition="OZ generates a bifurcation threshold (U4a)",
        common_error="Bifurcation without a handler (uncontained cascade)",
        pdf_reference="Tabla comparativa — Bifurcada — OZ → [ZHIR / NUL]",
    ),
    StructuralType.FRACTAL: StructuralTypeSpec(
        type=StructuralType.FRACTAL,
        pdf_term="Fractal",
        combinator="Kleene star (self-similar repeat)",
        chomsky_class=ChomskyClass.REGULAR,
        example=("NAV", "IL", "UM", "NAV"),
        required_glyphs=("NAV", "UM", "IL"),
        activation_condition="EPI replicable across scales without phase loss",
        common_error="Cycles without restructuring: nodal entropy",
        pdf_reference="Tabla comparativa — Fractal",
    ),
    StructuralType.CYCLIC: StructuralTypeSpec(
        type=StructuralType.CYCLIC,
        pdf_term="Cíclica",
        combinator="Kleene star of nested cycles",
        chomsky_class=ChomskyClass.CONTEXT_FREE,
        example=("THOL", "NAV", "THOL"),  # THOL[...] → NAV → THOL[...]
        required_glyphs=("THOL", "NAV"),
        activation_condition="SHA or NUL closure + restart via NAV",
        common_error="Feedback without an intermediate closure",
        pdf_reference="Tabla comparativa — Cíclica",
    ),
    StructuralType.HIERARCHICAL: StructuralTypeSpec(
        type=StructuralType.HIERARCHICAL,
        pdf_term="Jerárquica",
        combinator="nesting (Dyck)",
        chomsky_class=ChomskyClass.CONTEXT_FREE,
        example=("THOL", "AL", "ZHIR", "IL"),  # THOL[ AL → ZHIR → IL ]
        required_glyphs=("THOL",),
        activation_condition="Valid encapsulation with sustained internal coherence",
        common_error="Nesting without closure, or incompatible glyphs inside the node",
        pdf_reference="Tabla comparativa — Jerárquica — THOL[ ... ]",
    ),
}


# ===========================================================================
# 4. The canonical glyphic functions / macros (TNFR.pdf §2.3)
# ===========================================================================


@dataclass(frozen=True)
class GlyphicFunction:
    """A canonical glyphic function / macro (TNFR.pdf §2.3).

    These are structural FRAGMENTS (named, reusable words to COMPOSE), not
    standalone grammar-valid sequences. A fragment becomes a valid word by
    adding the grammar glue: a U1a generator prefix and a U1b closure suffix
    (plus the U4b context a transformer needs). See example 143.
    """

    name: str
    glyphs: tuple[str, ...]
    description: str
    structural_type: StructuralType
    pdf_reference: str
    nested: bool = False            # contains a THOL[...] sub-EPI body
    branches: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


CANONICAL_GLYPHIC_FUNCTIONS: dict[str, GlyphicFunction] = {
    "simple_activation": GlyphicFunction(
        name="simple_activation",
        glyphs=("AL", "IL", "RA"),
        description="Stabilized emission that propagates.",
        structural_type=StructuralType.LINEAR,
        pdf_reference="Tabla de funciones glíficas operativas — Activación simple",
    ),
    "mutational_stabilization": GlyphicFunction(
        name="mutational_stabilization",
        glyphs=("OZ", "ZHIR", "IL"),
        description="Dissonance transformed into coherence.",
        structural_type=StructuralType.LINEAR,
        pdf_reference="Tabla de funciones glíficas operativas — "
                      "Estabilización mutacional / MOD ESTABILIZADOR",
    ),
    "regenerative_cycle": GlyphicFunction(
        name="regenerative_cycle",
        glyphs=("NAV", "THOL", "SHA"),
        description="Self-organized node that returns to latency.",
        structural_type=StructuralType.CYCLIC,
        pdf_reference="Tabla de funciones glíficas operativas — Ciclo regenerativo",
        nested=True,
    ),
    "adaptive_interface": GlyphicFunction(
        name="adaptive_interface",
        glyphs=("THOL", "ZHIR", "UM", "NAV", "RA"),
        description="Glyphic network that reorganizes and expands.",
        structural_type=StructuralType.HIERARCHICAL,
        pdf_reference="Tabla de funciones glíficas operativas — Interfaz adaptativa",
        nested=True,
    ),
    "macro_init": GlyphicFunction(
        name="macro_init",
        glyphs=("AL", "IL", "UM"),
        description="Initialization macro (emission, coherence, coupling).",
        structural_type=StructuralType.LINEAR,
        pdf_reference="Macros glíficas — MACRO INIT",
    ),
    "mutational_bifurcation": GlyphicFunction(
        name="mutational_bifurcation",
        glyphs=("OZ",),
        description="Dissonance-triggered bifurcation: OZ opens two real "
                    "structural trajectories, mutation (ZHIR) or collapse (NUL).",
        structural_type=StructuralType.BIFURCATED,
        pdf_reference="Bifurcación y mutación — OZ → [ZHIR / NUL]",
        branches=(("ZHIR",), ("NUL",)),
    ),
}


# ===========================================================================
# 5. Legacy StructuralPattern → canonical StructuralType reduction
# ===========================================================================
#
# The legacy ``StructuralPattern`` enum mixes three axes (structural shape,
# application domain, learning process). Only the structural-shape axis is the
# canonical grammar typology. This mapping reduces every legacy label to its
# canonical structural type: the five shape members map directly; the
# operational-meta members map to their dominant shape; the domain/learning
# members are NOT structural shapes and map to ``UNKNOWN`` (their information
# lives on a separate, non-grammar axis — the ``domain`` metadata field).

STRUCTURAL_PATTERN_TO_TYPE: dict[StructuralPattern, StructuralType] = {
    # canonical structural typology (direct)
    StructuralPattern.LINEAR: StructuralType.LINEAR,
    StructuralPattern.BIFURCATED: StructuralType.BIFURCATED,
    StructuralPattern.FRACTAL: StructuralType.FRACTAL,
    StructuralPattern.CYCLIC: StructuralType.CYCLIC,
    StructuralPattern.HIERARCHICAL: StructuralType.HIERARCHICAL,
    # operational-meta labels → dominant canonical shape
    StructuralPattern.BOOTSTRAP: StructuralType.LINEAR,    # AL→…→close pulse
    StructuralPattern.STABILIZE: StructuralType.LINEAR,    # IL→close
    StructuralPattern.RESONATE: StructuralType.LINEAR,     # RA/UM propagation
    StructuralPattern.COMPRESS: StructuralType.LINEAR,     # NUL contraction line
    StructuralPattern.EXPLORE: StructuralType.BIFURCATED,  # OZ/ZHIR branch
    StructuralPattern.COMPLEX: StructuralType.HIERARCHICAL,  # composite/nested
    # domain / learning axes are not structural shapes
    StructuralPattern.THERAPEUTIC: StructuralType.UNKNOWN,
    StructuralPattern.EDUCATIONAL: StructuralType.UNKNOWN,
    StructuralPattern.ORGANIZATIONAL: StructuralType.UNKNOWN,
    StructuralPattern.CREATIVE: StructuralType.UNKNOWN,
    StructuralPattern.REGENERATIVE: StructuralType.UNKNOWN,
    StructuralPattern.BASIC_LEARNING: StructuralType.UNKNOWN,
    StructuralPattern.DEEP_LEARNING: StructuralType.UNKNOWN,
    StructuralPattern.EXPLORATORY_LEARNING: StructuralType.UNKNOWN,
    StructuralPattern.CONSOLIDATION_CYCLE: StructuralType.UNKNOWN,
    StructuralPattern.ADAPTIVE_MUTATION: StructuralType.UNKNOWN,
    StructuralPattern.UNKNOWN: StructuralType.UNKNOWN,
}


def canonical_structural_type(pattern: StructuralPattern) -> StructuralType:
    """Reduce a legacy ``StructuralPattern`` to its canonical structural type.

    Domain/learning labels (not a structural shape) reduce to
    ``StructuralType.UNKNOWN``; their non-structural information belongs to a
    separate application-metadata axis, not the canonical grammar typology.
    """
    return STRUCTURAL_PATTERN_TO_TYPE.get(pattern, StructuralType.UNKNOWN)


# ===========================================================================
# Self-consistency check
# ===========================================================================


def verify_canon_consistency() -> bool:
    """Assert the materialised role table reproduces the canonical sets exactly.

    The per-operator role table is derived from the same nodal-equation
    predicates as :mod:`grammar_types`; this check pins that the two views agree,
    so the canon cannot silently drift from the single source of truth.
    """
    derived_generators = {
        op for op, g in OPERATOR_ROLES.items()
        if g.has(GrammarRole.GENERATOR)
    }
    derived_closures = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.CLOSURE)
    }
    derived_stabilizers = {
        op for op, g in OPERATOR_ROLES.items()
        if g.has(GrammarRole.STABILIZER)
    }
    derived_destabilizers = {
        op for op, g in OPERATOR_ROLES.items()
        if g.has(GrammarRole.DESTABILIZER)
    }
    derived_transformers = {
        op for op, g in OPERATOR_ROLES.items()
        if g.has(GrammarRole.TRANSFORMER)
    }
    derived_triggers = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.TRIGGER)
    }
    derived_handlers = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.HANDLER)
    }
    checks = (
        derived_generators == set(GENERATORS),
        derived_closures == set(CLOSURES),
        derived_stabilizers == set(STABILIZERS),
        derived_destabilizers == set(DESTABILIZERS),
        derived_transformers == set(TRANSFORMERS),
        derived_triggers == set(BIFURCATION_TRIGGERS),
        derived_handlers == set(BIFURCATION_HANDLERS),
        # The structural typology has exactly the five canonical types, and the
        # legacy-pattern reduction covers every StructuralPattern member.
        {t for t in STRUCTURAL_TYPOLOGY} == {
            StructuralType.LINEAR,
            StructuralType.BIFURCATED,
            StructuralType.FRACTAL,
            StructuralType.CYCLIC,
            StructuralType.HIERARCHICAL,
        },
        set(STRUCTURAL_PATTERN_TO_TYPE) == set(StructuralPattern),
    )
    return all(checks)
