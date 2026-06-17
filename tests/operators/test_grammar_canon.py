"""Canonical consistency of the unified grammar specification (grammar_canon).

These tests guard the single-source-of-truth grammar specification introduced by
``tnfr.operators.grammar_canon``. They pin that:

1. the materialised per-operator role table reproduces the canonical
   classification sets in ``grammar_types`` exactly (no drift);
2. the canonical structural typology has exactly the five TNFR.pdf types and
   agrees with ``grammar_types.CANONICAL_STRUCTURAL_TYPES`` and
   ``StructuralPattern``;
3. the legacy ``StructuralPattern`` → canonical ``StructuralType`` reduction is
   total (covers every enum member);
4. the canonical glyphic functions are well-formed fragments that COMPOSE into
   grammar-valid words (TNFR.pdf §2.3);
5. the formal-syntax schema and the U1-U6 rule registry are internally coherent.
"""

from __future__ import annotations

from tnfr.operators import grammar_canon as gc
from tnfr.operators import grammar_types as gt
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling, Resonance,
    Silence, Expansion, Contraction, SelfOrganization, Mutation,
    Transition, Recursivity,
)
from tnfr.operators.grammar_validate import validate_grammar

_INST = {
    "AL": Emission(), "EN": Reception(), "IL": Coherence(),
    "OZ": Dissonance(), "UM": Coupling(), "RA": Resonance(),
    "SHA": Silence(), "VAL": Expansion(), "NUL": Contraction(),
    "THOL": SelfOrganization(), "ZHIR": Mutation(), "NAV": Transition(),
    "REMESH": Recursivity(),
}


def _valid(glyphs):
    return validate_grammar([_INST[g] for g in glyphs], 0.0)


class TestRoleTableReproducesCanon:
    """The materialised role table must equal the canonical sets exactly."""

    def test_self_consistency_check_passes(self) -> None:
        assert gc.verify_canon_consistency() is True

    def test_generators(self) -> None:
        derived = {
            op for op, g in gc.OPERATOR_ROLES.items()
            if g.has(gc.GrammarRole.GENERATOR)
        }
        assert derived == set(gt.GENERATORS)

    def test_destabilizers(self) -> None:
        derived = {
            op for op, g in gc.OPERATOR_ROLES.items()
            if g.has(gc.GrammarRole.DESTABILIZER)
        }
        assert derived == set(gt.DESTABILIZERS)

    def test_transformers(self) -> None:
        derived = {
            op for op, g in gc.OPERATOR_ROLES.items()
            if g.has(gc.GrammarRole.TRANSFORMER)
        }
        assert derived == set(gt.TRANSFORMERS)

    def test_all_thirteen_operators_present(self) -> None:
        assert len(gc.OPERATOR_ROLES) == 13

    def test_mutation_is_destabilizer_and_transformer(self) -> None:
        roles = gc.OPERATOR_ROLES["mutation"].roles
        assert gc.GrammarRole.DESTABILIZER in roles
        assert gc.GrammarRole.TRANSFORMER in roles

    def test_contraction_is_not_a_closure(self) -> None:
        # Engine derivation (NUL reduces dim(EPI), does not force ∂EPI/∂t→0).
        assert not gc.OPERATOR_ROLES["contraction"].has(gc.GrammarRole.CLOSURE)


class TestCanonicalTypology:
    """Exactly five canonical structural types, agreeing across modules."""

    def test_five_types(self) -> None:
        assert set(gc.STRUCTURAL_TYPOLOGY) == {
            gc.StructuralType.LINEAR,
            gc.StructuralType.BIFURCATED,
            gc.StructuralType.FRACTAL,
            gc.StructuralType.CYCLIC,
            gc.StructuralType.HIERARCHICAL,
        }

    def test_canonical_structural_types_match(self) -> None:
        names = {t.value for t in gt.CANONICAL_STRUCTURAL_TYPES}
        assert names == {s.value for s in gc.STRUCTURAL_TYPOLOGY}

    def test_chomsky_classes(self) -> None:
        T = gc.StructuralType
        C = gc.ChomskyClass
        assert gc.STRUCTURAL_TYPOLOGY[T.LINEAR].chomsky_class == C.REGULAR
        assert gc.STRUCTURAL_TYPOLOGY[T.BIFURCATED].chomsky_class == C.REGULAR
        assert gc.STRUCTURAL_TYPOLOGY[T.FRACTAL].chomsky_class == C.REGULAR
        assert (
            gc.STRUCTURAL_TYPOLOGY[T.HIERARCHICAL].chomsky_class
            == C.CONTEXT_FREE
        )
        assert gc.STRUCTURAL_TYPOLOGY[T.CYCLIC].chomsky_class == C.CONTEXT_FREE


class TestLegacyReductionIsTotal:
    """Every legacy StructuralPattern reduces to a canonical type."""

    def test_mapping_covers_every_member(self) -> None:
        assert set(gc.STRUCTURAL_PATTERN_TO_TYPE) == set(gt.StructuralPattern)

    def test_canonical_members_map_to_themselves(self) -> None:
        for p in gt.CANONICAL_STRUCTURAL_TYPES:
            assert gc.canonical_structural_type(p).value == p.value

    def test_domain_labels_are_not_structural(self) -> None:
        # Domain labels are a separate, non-grammar axis → UNKNOWN shape.
        for p in (
            gt.StructuralPattern.THERAPEUTIC,
            gt.StructuralPattern.EDUCATIONAL,
            gt.StructuralPattern.CREATIVE,
        ):
            assert (
                gc.canonical_structural_type(p) == gc.StructuralType.UNKNOWN
            )


class TestGlyphicFunctionsCompose:
    """Canonical glyphic functions are fragments that compose into valid words."""

    def test_fragments_are_not_standalone_words(self) -> None:
        # Per TNFR.pdf they are macros to COMPOSE, not standalone valid words
        # (see example 143). At least the non-trivial ones are fragments.
        frag = gc.CANONICAL_GLYPHIC_FUNCTIONS["macro_init"]
        assert not _valid(list(frag.glyphs))  # [AL, IL, UM] lacks a closure

    def test_linear_fragments_compose_with_grammar_glue(self) -> None:
        # A U1a generator prefix + U1b closure suffix turns a linear,
        # transformer-free fragment into a grammar-valid word.
        for key in ("simple_activation", "macro_init"):
            fn = gc.CANONICAL_GLYPHIC_FUNCTIONS[key]
            word = ["AL"] + list(fn.glyphs) + ["SHA"]
            assert _valid(word), f"{key} should compose into a valid word"

    def test_all_glyphs_are_canonical(self) -> None:
        for fn in gc.CANONICAL_GLYPHIC_FUNCTIONS.values():
            assert all(g in _INST for g in fn.glyphs)

    def test_branches_are_canonical(self) -> None:
        fn = gc.CANONICAL_GLYPHIC_FUNCTIONS["mutational_bifurcation"]
        assert fn.branches == (("ZHIR",), ("NUL",))


class TestRuleRegistry:
    """The U1-U6 rule registry is coherent and complete."""

    def test_all_rules_present(self) -> None:
        ids = {r.rule_id for r in gc.GRAMMAR_RULES}
        assert ids == {"U1a", "U1b", "U2", "U3", "U4a", "U4b", "U5", "U6"}

    def test_rule_lookup(self) -> None:
        assert gc.rule("U4b").name.startswith("Bifurcation")
        assert "TRANSFORMERS" in gc.rule("U4b").operator_sets

    def test_formal_syntax_schema_positions(self) -> None:
        schema = gc.FORMAL_SYNTAX_SCHEMA
        assert "AL" in schema["start"]
        assert "SHA" in schema["closure"]


class TestRelatedInvariants:
    """The rule→invariant annotation is canonical (6-invariant model)."""

    def test_every_rule_relates_to_grammar_compliance(self) -> None:
        # Grammar Compliance (#4) is in every grammar-rule's related set.
        for r in gc.GRAMMAR_RULES:
            assert gc.GRAMMAR_COMPLIANCE_INVARIANT in gc.related_invariants(
                r.rule_id
            )

    def test_invariants_are_in_the_six_invariant_canon(self) -> None:
        # No stale references to the old 10-invariant numbering (7, 9, …).
        for r in gc.GRAMMAR_RULES:
            for inv in gc.related_invariants(r.rule_id):
                assert 1 <= inv <= 6

    def test_primary_invariant_is_included(self) -> None:
        for r in gc.GRAMMAR_RULES:
            assert r.invariant in gc.related_invariants(r.rule_id)

    def test_error_factory_mapping_is_derived(self) -> None:
        # The error factory's _RULE_INVARIANTS must come from grammar_canon.
        from tnfr.operators import grammar_error_factory as gef

        for r in gc.GRAMMAR_RULES:
            assert gef._RULE_INVARIANTS[r.rule_id] == gc.related_invariants(
                r.rule_id
            )
        # The U6 confinement alias maps to the canonical U6 rule.
        assert gef._RULE_INVARIANTS["U6_CONFINEMENT"] == gc.related_invariants(
            "U6"
        )


class TestOperatorMetadataRolesAreCanonical:
    """introspection.OPERATOR_METADATA.grammar_roles == the canonical roles."""

    def test_every_operator_metadata_matches_canon(self) -> None:
        from tnfr.operators.introspection import OPERATOR_METADATA

        for mnemonic, meta in OPERATOR_METADATA.items():
            expected = gc.u_rules_for_operator(mnemonic)
            assert tuple(meta.grammar_roles) == expected, (
                f"{mnemonic}: metadata {meta.grammar_roles} != canon {expected}"
            )

    def test_u_rules_accepts_function_name_and_glyph(self) -> None:
        # Same result whether queried by function name or glyph mnemonic.
        assert gc.u_rules_for_operator("mutation") == gc.u_rules_for_operator(
            "ZHIR"
        )
        assert gc.u_rules_for_operator("ZHIR") == ("U2", "U4a", "U4b")

    def test_remesh_carries_the_recursive_rule(self) -> None:
        # REMESH is the U5 recursive generator (was historically omitted).
        assert "U5" in gc.u_rules_for_operator("REMESH")

    def test_dissonance_is_also_a_closure(self) -> None:
        # OZ is a closure (∈ CLOSURES) → U1b (was historically omitted).
        assert "U1b" in gc.u_rules_for_operator("OZ")
