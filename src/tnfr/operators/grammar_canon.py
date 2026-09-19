"""TNFR Canonical Grammar Specification — the single source of truth.

This module is the authoritative, physics-grounded, TNFR.pdf-anchored
specification of the grammar of the 13 structural operators. It does NOT
re-implement validation (that lives in :mod:`grammar_core` / :mod:`grammar_validate`)
and it does NOT re-define the classification sets (those are derived in
:mod:`tnfr.config.physics_derivation` and re-exported by :mod:`grammar_types`).
Instead it *materialises*, in one place, the canonical knowledge that was
previously scattered across modules and prose:

1. ``OPERATOR_ROLES`` — the per-operator grammatical role table (the 13 operators
   × their U1-U6 roles), materialized from the contract-role predicates in
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
required closures. The engine's supported endpoint policy does NOT include
NUL. Its ``CLOSURES`` = {SHA, NAV, REMESH, OZ} is a contract-role set (see
``physics_derivation.achieves_operational_closure`` /
``can_stabilize_reorganization``). This module documents the PDF nuance without
overriding the supported policy. A single SHA attenuates capacity without
generally setting it to zero; bounded pressure and repeated rate suppression
are additional premises for asymptotic inactivity. NAV, REMESH and OZ closure
labels likewise do not certify a stationary endpoint. The nodal equation alone
does not uniquely choose between the historical and implemented closure sets.

The role table is derived rather than hand-maintained: it is built by querying
the shared classification predicates, and a self-check
(:func:`verify_canon_consistency`) asserts that the materialised roles reproduce
the canonical sets in :mod:`grammar_types` exactly.  The rule descriptions and
PDF typology are declarative specifications with explicit policy scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

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
    FUNCTION_TO_GLYPH,
    GENERATORS,
    RECURSIVE_GENERATORS,
    STABILIZERS,
    TRANSFORMERS,
    StructuralPattern,
)

__all__ = [
    "GrammarRole",
    "OperatorGrammar",
    "OPERATOR_ROLES",
    "operator_grammar",
    "GrammarRule",
    "GRAMMAR_RULES",
    "GrammarBasisKind",
    "GrammarBasis",
    "GRAMMAR_BASES",
    "grammar_basis",
    "operator_role_metadata",
    "rule",
    "GRAMMAR_COMPLIANCE_INVARIANT",
    "related_invariants",
    "ROLE_TO_URULE",
    "u_rules_for_operator",
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

    GENERATOR = "generator"  # U1a — can start (create/activate EPI)
    CLOSURE = "closure"  # U1b — can end (stabilize / close cycle)
    STABILIZER = "stabilizer"  # U2  — reduces |ΔNFR| (negative feedback)
    # U2 debt: OZ perturbs pressure, ZHIR phase, and VAL capacity.
    DESTABILIZER = "destabilizer"
    COUPLING = "coupling"  # U3  — requires phase verification
    TRIGGER = "trigger"  # U4a — may push ∂²EPI/∂t² past τ
    HANDLER = "handler"  # U4a — absorbs a triggered bifurcation
    TRANSFORMER = "transformer"  # U4b — executes a threshold-gated bifurcation
    RECURSIVE = "recursive"  # U5  — echoes structure across scales


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

    rule_id: str  # "U1a", "U2", "U4b", ...
    name: str
    physics: str  # the nodal-equation rationale
    operator_sets: tuple[str, ...]  # names of the classification sets it uses
    invariant: int  # canonical invariant 1-6 it maps to
    pdf_reference: str  # TNFR.pdf §2.3.3 anchor


GRAMMAR_RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        rule_id="U1a",
        name="Structural Initiation",
        physics="At EPI=0 the nodal derivative remains νf·ΔNFR whenever its "
        "channels are defined. U1a is the standalone-sequence policy that "
        "requires an explicit generator to create or activate form.",
        operator_sets=("GENERATORS",),
        invariant=1,
        pdf_reference="§2.3.3 'Esquema formal de sintaxis' — valid start: AL, "
        "NAV (+ REMESH reactivator)",
    ),
    GrammarRule(
        rule_id="U1b",
        name="Structural Closure",
        physics="A standalone sequence must end with a registered closure. SHA "
        "suppresses the rate; NAV and REMESH close operational cycles; OZ is "
        "retained as a compatible terminal label. Closure membership alone "
        "does not prove convergence to a stable attractor.",
        operator_sets=("CLOSURES",),
        invariant=1,
        pdf_reference="§2.3.3 'Cierre estructural' — close with a latency glyph "
        "(SHA, NUL)",
    ),
    GrammarRule(
        rule_id="U2",
        name="Stabilization Coverage & Debt",
        physics="U2 assigns finite calibrated debt to declared perturbations: "
        "OZ directly raises |ΔNFR|, ZHIR transforms phase, and VAL raises νf. "
        "The configured debt capacity and IL/THOL coverage form a sequence "
        "policy; they do not prove convergence of ∫νf·ΔNFR dt.",
        operator_sets=("DESTABILIZERS", "STABILIZERS"),
        invariant=1,
        pdf_reference="Compatibilidad entre glifos / Bifurcación y mutación",
    ),
    GrammarRule(
        rule_id="U3",
        name="Resonant Coupling",
        physics="Resonance requires phase compatibility |wrap(φᵢ - φⱼ)| ≤ Δφ_max; "
        "antiphase coupling produces destructive interference.",
        operator_sets=("COUPLING_RESONANCE",),
        invariant=2,
        pdf_reference="§2.3.3 'Compatibilidad' — phase compatibility of "
        "coupling operators",
    ),
    GrammarRule(
        rule_id="U4a",
        name="Bifurcation Dynamics — triggers need handlers",
        physics="Operators assigned the U4a trigger role require a registered "
        "handler. The label-level rule does not itself establish a measured "
        "crossing of ∂²EPI/∂t² > τ or its absorption.",
        operator_sets=("BIFURCATION_TRIGGERS", "BIFURCATION_HANDLERS"),
        invariant=4,
        pdf_reference="Bifurcación y mutación — OZ → [ZHIR / NUL]",
    ),
    GrammarRule(
        rule_id="U4b",
        name="Bifurcation Dynamics — transformers need context",
        physics="A transformer (ZHIR/THOL) requires recent declared perturbation "
        "context from the pressure, phase, or capacity debt channels; ZHIR "
        "also requires a prior IL. The labels do not measure a threshold.",
        operator_sets=("TRANSFORMERS", "DESTABILIZERS"),
        invariant=4,
        pdf_reference="§2.3.3 'Compatibilidad entre glifos' — ZHIR must be "
        "preceded by OZ (no mutation without dissonance)",
    ),
    GrammarRule(
        rule_id="U5",
        name="Multi-Scale Coherence",
        physics="Hierarchical coupling requires stabilizer coverage at each "
        "nested scale. C_parent ≥ α·Σ C_child is a configured target "
        "whose interpretation requires a specified α and normalization.",
        operator_sets=("RECURSIVE_GENERATORS", "STABILIZERS"),
        invariant=3,
        pdf_reference="§2.3.3 'Agrupamiento y jerarquía' — THOL[...] nesting",
    ),
    GrammarRule(
        rule_id="U6",
        name="Structural Potential Confinement",
        physics="Monitor the reference-state drift of Φ_s = Σ ΔNFR_j / d² "
        "against the selected policy threshold ΔΦ_s < π/2. This is a "
        "read-only check, not a graph-independent field bound.",
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


class GrammarBasisKind(str, Enum):
    """Kinds of structural justification, separate from evidence status."""

    IDENTITY = "identity"
    CONDITIONAL_THEOREM = "conditional_theorem"
    CONTRACT = "contract"
    POLICY = "policy"


@dataclass(frozen=True)
class GrammarBasis:
    """One declarative basis; neither its label nor owner verifies premises.

    A rule can combine several bases. ``configured_choices`` names policies
    or formulas, not a measurement of the configuration of a particular graph.
    Hypotheses require separate domain-specific evidence before application.
    """

    kind: GrammarBasisKind
    statement: str
    hypotheses: tuple[str, ...]
    configured_choices: tuple[tuple[str, str], ...]
    owner: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GrammarBasisKind(self.kind))
        if isinstance(self.hypotheses, (str, bytes)):
            raise TypeError("hypotheses must be a sequence of text")
        hypotheses = tuple(self.hypotheses)
        raw_choices = tuple(self.configured_choices)
        if any(isinstance(pair, (str, bytes)) for pair in raw_choices):
            raise TypeError("configured choices must be name/value pairs")
        choices = tuple(tuple(pair) for pair in raw_choices)
        if any(len(pair) != 2 for pair in choices):
            raise ValueError("configured choices must be name/value pairs")
        text = (
            self.statement,
            self.owner,
            *hypotheses,
            *(item for pair in choices for item in pair),
        )
        if any(type(item) is not str or not item.strip() for item in text):
            raise ValueError("basis metadata must contain nonempty text")
        if len({pair[0] for pair in choices}) != len(choices):
            raise ValueError("configured choice names must be unique")
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "configured_choices", choices)

    def as_dict(self) -> dict:
        """Return detached metadata without asserting checked hypotheses."""
        return {
            "kind": self.kind.value,
            "statement": self.statement,
            "hypotheses": list(self.hypotheses),
            "configured_choices": dict(self.configured_choices),
            "owner": self.owner,
            "scope": "declarative_basis_not_execution_evidence",
        }


# The historical GrammarRule schema stays unchanged. These compositional
# descriptions do not alter role membership, validators, gains or admission.
_TETRAD_OBSERVATION_BASIS = GrammarBasis(
    GrammarBasisKind.CONTRACT,
    "The full tetrad combines pressure aggregation Phi_s, wrapped phase gradient/curvature, "
    "and nonlocal coherence length xi_C; these readouts do not certify full-state closure.",
    (
        "declared graph and state snapshot",
        "field-specific conventions and time coverage",
        "separate state-closure evidence for dynamical prediction",
    ),
    (),
    "tnfr.metrics.observations.observe_graph_tetrad",
)

GRAMMAR_BASES = MappingProxyType(
    {
        "U1a": (
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "The nodal rate is nu_f * DeltaNFR, including at EPI = 0.",
                ("defined finite nodal channels", "differentiable flow interval"),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#1-existence-boundedness-and-convergence",
            ),
            GrammarBasis(
                GrammarBasisKind.CONTRACT,
                "Generation or activation is an operator effect with live preconditions.",
                ("admitted operator input", "actual operator realization"),
                (),
                "tnfr.operators.operator_contracts.contract_for",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "A word starting from null form requires a registered generator.",
                ("declared initial EPI", "standalone word context"),
                (("generators", "GENERATORS"),),
                "tnfr.operators.grammar_core.GrammarValidator.validate_initiation",
            ),
        ),
        "U1b": (
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "Instantaneous stationarity requires nu_f * DeltaNFR = 0.",
                ("defined nodal channels", "differentiable flow interval"),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#1-existence-boundedness-and-convergence",
            ),
            GrammarBasis(
                GrammarBasisKind.CONDITIONAL_THEOREM,
                "Repeated nu_k = alpha**k * nu_0 with bounded pressure has rate tending to zero.",
                (
                    "fixed 0 <= alpha < 1",
                    "bounded pressure",
                    "no intervening capacity writes",
                ),
                (),
                "tnfr.config.physics_derivation.can_stabilize_reorganization",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "A registered closure ends a word without certifying a stationary endpoint.",
                ("standalone word context",),
                (("closures", "CLOSURES"),),
                "tnfr.config.physics_derivation.derive_end_operators_from_physics",
            ),
        ),
        "U2": (
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "Hybrid EPI change is the nodal flow integral plus the sum of EPI jumps.",
                (
                    "absolutely continuous flow segments",
                    "locally finite recorded jumps",
                ),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#9-u2-and-u4-in-the-actual-flowjump-model",
            ),
            GrammarBasis(
                GrammarBasisKind.CONDITIONAL_THEOREM,
                "Absolute integrability of the rate and absolute summability of jumps imply a finite EPI limit.",
                (
                    "defined scalar hybrid trajectory on the full time tail",
                    "absolutely integrable nodal rate",
                    "absolutely summable EPI jumps",
                ),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#9-u2-and-u4-in-the-actual-flowjump-model",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "Registered destabilizer debt and stabilizer coverage constrain words.",
                ("accepted history and carried debt",),
                (
                    ("debt_capacity", "U2_DEBT_CAPACITY"),
                    ("calibration", "floor(1/(nu_f*dt*rho)); rho=1 surrogate"),
                ),
                "tnfr.operators.grammar_debt.advance_debt",
            ),
        ),
        "U3": (
            _TETRAD_OBSERVATION_BASIS,
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "The squared two-phasor magnitude is a*a + b*b + 2*a*b*cos(delta).",
                ("nonnegative phasor amplitudes", "circular separation delta"),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#10-u3-exact-geometric-content-and-a-strict-gate-counterexample",
            ),
            GrammarBasis(
                GrammarBasisKind.CONTRACT,
                "Concrete UM/RA requires a compatible existing neighbor; merged UM relations are rechecked.",
                (
                    "finite live phases",
                    "actual graph support",
                    "admitted phase limits",
                    "merged stage proposals for a stage-level claim",
                ),
                (),
                "tnfr.operators._phase_gate.resolve_u3_phase_neighbors",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "The hard phase limit is selected in [0, pi/2]; UM can tighten it.",
                ("finite graph configuration",),
                (
                    ("hard_limit", "DELTA_PHI_MAX"),
                    ("optional_tightening", "UM_MAX_PHASE_DIFF"),
                ),
                "tnfr.operators._phase_gate.resolve_u3_phase_limits",
            ),
        ),
        "U4a": (
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "On a smooth flow segment EPI'' = nu_f' * DeltaNFR + nu_f * DeltaNFR'.",
                (
                    "differentiable capacity and pressure",
                    "twice differentiable EPI",
                    "no jump at the differentiation point",
                ),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#9-u2-and-u4-in-the-actual-flowjump-model",
            ),
            GrammarBasis(
                GrammarBasisKind.CONTRACT,
                "Threshold and birth evidence must come from the realized operator-specific observation.",
                (
                    "valid timestamped nodal history",
                    "operator-specific threshold and proposal",
                ),
                (),
                "tnfr.operators.self_organization_selection.observe_self_organization_eligibility",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "Registered triggers require handler coverage, not a claimed measured threshold crossing.",
                ("word or incremental execution context",),
                (
                    ("triggers", "BIFURCATION_TRIGGERS"),
                    ("handlers", "BIFURCATION_HANDLERS"),
                ),
                "tnfr.config.physics_derivation.derive_bifurcation_handlers_from_physics",
            ),
        ),
        "U4b": (
            GrammarBasis(
                GrammarBasisKind.CONDITIONAL_THEOREM,
                "For 0 <= q < 1 there exists n with q**n below any fixed positive band.",
                (
                    "scalar relaxation surrogate",
                    "0 <= q = 1-nu_f*dt*rho < 1",
                    "specified band strictly between zero and one",
                ),
                (
                    ("rho", "1"),
                    ("band", "1/(pi+1)"),
                    (
                        "implementation_scope",
                        "one-step fallback and 64-step cap do not certify the inequality",
                    ),
                ),
                "tnfr.config.physics_derivation.derive_bifurcation_window_from_physics",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "Transformers require recent destabilization; Mutation also requires prior Coherence.",
                ("accepted history", "retained prior-Coherence fact"),
                (("recency_window", "BIFURCATION_WINDOW"),),
                "tnfr.operators.grammar_dynamics._check_u4b",
            ),
        ),
        "U5": (
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "A differentiable parent representation obeys the chain rule on compatible flows.",
                (
                    "specified differentiable parent map",
                    "compatible differentiable child/parent dynamics",
                ),
                (),
                "theory/UNIFIED_GRAMMAR_RULES.md#6-u5--multi-scale-coherence",
            ),
            GrammarBasis(
                GrammarBasisKind.CONDITIONAL_THEOREM,
                "A fixed affine model projects autonomously under R when RA = Abar R; the source projects as Rb.",
                (
                    "fixed model x'=-Ax+b",
                    "fixed linear observation R",
                    "verified RA=Abar R",
                ),
                (),
                "tnfr.physics.epi_memory.observe_forced_support_closure",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "Declared deep recursion requires nearby scale stabilizer coverage.",
                ("declared recursion depth", "accepted word context"),
                (
                    ("scale_stabilizers", "STABILIZERS"),
                    ("recency_window", "BIFURCATION_WINDOW"),
                ),
                "tnfr.operators.grammar_core.GrammarValidator.validate_multiscale_coherence",
            ),
        ),
        "U6": (
            _TETRAD_OBSERVATION_BASIS,
            GrammarBasis(
                GrammarBasisKind.IDENTITY,
                "For aligned snapshots DeltaPhi = B_after DeltaPressure + (B_after-B_before) Pressure_before.",
                (
                    "aligned node order",
                    "declared distance kernels",
                    "finite pressure snapshots",
                ),
                (),
                "tnfr.operators.grammar_u6.structural_potential_change_terms",
            ),
            GrammarBasis(
                GrammarBasisKind.CONDITIONAL_THEOREM,
                "The fixed linear field satisfies norm_inf(Phi) <= norm_inf(B) * norm_inf(pressure).",
                ("fixed finite distance kernel", "bounded pressure"),
                (),
                "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#4-structural-potential-and-topology-dependent-bounds",
            ),
            GrammarBasis(
                GrammarBasisKind.POLICY,
                "Mean absolute nodewise potential drift must be strictly below the selected threshold.",
                (
                    "complete aligned reference and observed snapshots",
                    "declared canonical field provenance",
                ),
                (
                    ("threshold", "U6_STRUCTURAL_POTENTIAL_LIMIT"),
                    ("aggregation", "mean_absolute_nodewise_drift"),
                    ("time_coverage", "two_snapshot_finite_observation"),
                ),
                "tnfr.operators.grammar_u6.validate_structural_potential_confinement",
            ),
        ),
    }
)


def grammar_basis(rule_id: str) -> tuple[GrammarBasis, ...]:
    """Return immutable basis metadata, without evaluating any hypothesis."""
    return GRAMMAR_BASES[rule_id]


#: Canonical invariant index for "Grammar Compliance" (AGENTS.md §Canonical
#: Invariants, the 6-invariant model). Every grammar-rule violation relates to
#: this invariant by definition, in addition to the rule's primary physics
#: invariant.
GRAMMAR_COMPLIANCE_INVARIANT = 4


def related_invariants(rule_id: str) -> tuple[int, ...]:
    """Canonical invariants a violation of ``rule_id`` relates to.

    Returns the rule's primary physics invariant plus Grammar Compliance (#4),
    sorted and de-duplicated. This is the single source of the rule→invariant
    annotation, reconciled to the 6-invariant canon (AGENTS.md §Canonical
    Invariants); it replaces the stale pre-optimization 10-invariant numbering.
    """
    try:
        primary = rule(rule_id).invariant
    except KeyError:
        return (GRAMMAR_COMPLIANCE_INVARIANT,)
    return tuple(sorted({primary, GRAMMAR_COMPLIANCE_INVARIANT}))


#: Map each grammatical role to the active U1-U5 rule id it participates in.
#: (U6 confinement is telemetry-only and is not an active operator role.)
ROLE_TO_URULE: dict[GrammarRole, str] = {
    GrammarRole.GENERATOR: "U1a",
    GrammarRole.CLOSURE: "U1b",
    GrammarRole.STABILIZER: "U2",
    GrammarRole.DESTABILIZER: "U2",
    GrammarRole.COUPLING: "U3",
    GrammarRole.TRIGGER: "U4a",
    GrammarRole.HANDLER: "U4a",
    GrammarRole.TRANSFORMER: "U4b",
    GrammarRole.RECURSIVE: "U5",
}

#: Resolve a glyph mnemonic (e.g. "ZHIR") to its executable identifier.
_OPERATOR_BY_GLYPH: dict[str, str] = {g.glyph: op for op, g in OPERATOR_ROLES.items()}


def u_rules_for_operator(op: str) -> tuple[str, ...]:
    """The active U1-U5 rule ids an operator participates in (sorted, unique).

    Derived from the operator's canonical role set. ``op`` may be a function
    name (e.g. ``"mutation"``) or a glyph mnemonic (e.g. ``"ZHIR"``). U6
    (confinement) is telemetry-only and is not an active operator role, so it
    never appears here. Single source of the per-operator grammar-role table.
    """
    name = _OPERATOR_BY_GLYPH.get(op, op)
    grammar = OPERATOR_ROLES.get(name)
    if grammar is None:
        return ()
    return tuple(sorted({ROLE_TO_URULE[r] for r in grammar.roles}))


def operator_role_metadata(op: str) -> dict:
    """Serialize canonical roles and bases with the legacy SDK role view.

    The legacy ``roles`` list preserves its spelling and order. Complete roles
    are exposed separately; neither list constitutes execution evidence.
    """
    name = _OPERATOR_BY_GLYPH.get(op, op)
    grammar = operator_grammar(name)
    legacy = (
        (GrammarRole.GENERATOR, "generator"),
        (GrammarRole.CLOSURE, "closure"),
        (GrammarRole.STABILIZER, "stabilizer"),
        (GrammarRole.DESTABILIZER, "destabilizer"),
        (GrammarRole.TRANSFORMER, "transformer"),
        (GrammarRole.COUPLING, "coupling/resonance"),
    )
    rule_ids = u_rules_for_operator(name)
    return {
        "roles": [label for role, label in legacy if grammar.has(role)],
        "canonical_roles": [role.value for role in GrammarRole if grammar.has(role)],
        "u_rules": list(rule_ids),
        "grammar_basis": {
            rule_id: [basis.as_dict() for basis in grammar_basis(rule_id)]
            for rule_id in rule_ids
        },
    }


#: The TNFR.pdf §2.3.3 "Esquema formal de sintaxis" positions (theory anchor).
#: Quoted Spanish terms are verbatim citations of the source schema headers.
FORMAL_SYNTAX_SCHEMA: dict[str, tuple[str, ...]] = {
    "start": (
        "AL",
        "NAV",
        "REMESH",
    ),  # valid start ("Inicio válido") + REMESH reactivator
    "development": (
        "IL",
        "THOL",
        "UM",
    ),  # required development ("Desarrollo necesario")
    "optional_transition": (
        "OZ",
        "ZHIR",
        "REMESH",
    ),  # optional transition ("Transición opcional")
    "closure": ("SHA", "NUL"),  # required closure ("Cierre requerido"); see NUL note
}


# ===========================================================================
# 3. The canonical structural typology (TNFR.pdf "Tabla comparativa")
# ===========================================================================


class ChomskyClass(str, Enum):
    """Chomsky-hierarchy class of a glyphic structure (examples 139-144)."""

    REGULAR = "regular"  # concatenation / union / Kleene star
    CONTEXT_FREE = "context_free"  # nesting (Dyck), THOL[...]


class StructuralType(str, Enum):
    """The five canonical glyphic structure types (TNFR.pdf "Tabla comparativa").

    This is the canonical structural typology — a sequence's shape, determinable
    from the operator stream alone (examples 143-144). It is distinct from the
    application *domain* of a sequence (therapeutic, educational, …), which is a
    separate axis tracked by the ``domain`` metadata field.
    """

    LINEAR = "linear"  # Lineal — simple concatenation, latency close
    BIFURCATED = "bifurcated"  # Bifurcada — OZ → [ZHIR | NUL] branch (union)
    FRACTAL = "fractal"  # Fractal — self-similar repeat (Kleene star)
    CYCLIC = "cyclic"  # Cíclica — close-and-reopen feedback cycle
    HIERARCHICAL = "hierarchical"  # Jerárquica — nested THOL[...] (Dyck/CF)
    UNKNOWN = "unknown"  # not a recognised canonical structure


@dataclass(frozen=True)
class StructuralTypeSpec:
    """Canonical metadata for one structural type (TNFR.pdf §2.3)."""

    type: StructuralType
    pdf_term: str  # verbatim term from TNFR.pdf (Spanish source)
    combinator: str  # concatenation / union / star / nesting
    chomsky_class: ChomskyClass
    example: tuple[str, ...]  # canonical glyphic example
    required_glyphs: tuple[str, ...]
    activation_condition: str  # English (paraphrase of the PDF condition)
    common_error: str  # English (paraphrase of the PDF error)
    pdf_reference: str  # verbatim section-title citation


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
    nested: bool = False  # contains a THOL[...] sub-EPI body
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
    StructuralPattern.BOOTSTRAP: StructuralType.LINEAR,  # AL→…→close pulse
    StructuralPattern.STABILIZE: StructuralType.LINEAR,  # IL→close
    StructuralPattern.RESONATE: StructuralType.LINEAR,  # RA/UM propagation
    StructuralPattern.COMPRESS: StructuralType.LINEAR,  # NUL contraction line
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
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.GENERATOR)
    }
    derived_closures = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.CLOSURE)
    }
    derived_stabilizers = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.STABILIZER)
    }
    derived_destabilizers = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.DESTABILIZER)
    }
    derived_transformers = {
        op for op, g in OPERATOR_ROLES.items() if g.has(GrammarRole.TRANSFORMER)
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
        {t for t in STRUCTURAL_TYPOLOGY}
        == {
            StructuralType.LINEAR,
            StructuralType.BIFURCATED,
            StructuralType.FRACTAL,
            StructuralType.CYCLIC,
            StructuralType.HIERARCHICAL,
        },
        set(STRUCTURAL_PATTERN_TO_TYPE) == set(StructuralPattern),
    )
    return all(checks)
