"""Integration tests for canonical operator sequences.

Tests the 6 archetypal sequences with OZ (Dissonance) from TNFR theory.
Validates registry, application, filtering, and coherence outcomes.
"""

import pytest

from tnfr.sdk import TNFRNetwork, NetworkConfig
from tnfr.operators.canonical_patterns import (
    CANONICAL_SEQUENCES,
    BIFURCATED_BASE,
    BIFURCATED_COLLAPSE,
    THERAPEUTIC_PROTOCOL,
    THEORY_SYSTEM,
    FULL_DEPLOYMENT,
    MOD_STABILIZER,
)
from tnfr.types import Glyph


class TestCanonicalSequencesRegistry:
    """Test canonical sequences registry and metadata."""

    def test_registry_has_all_sequences(self):
        """Verify all 6 canonical sequences are registered."""
        required = [
            "bifurcated_base",
            "bifurcated_collapse",
            "therapeutic_protocol",
            "theory_system",
            "full_deployment",
            "mod_stabilizer",
        ]

        for name in required:
            assert name in CANONICAL_SEQUENCES, f"Missing sequence: {name}"

    def test_all_sequences_have_oz(self):
        """Verify all canonical sequences contain OZ (Dissonance)."""
        for name, seq in CANONICAL_SEQUENCES.items():
            assert Glyph.OZ in seq.glyphs, f"Sequence {name} missing OZ"

    def test_sequence_metadata_complete(self):
        """Verify all sequences have complete metadata."""
        for name, seq in CANONICAL_SEQUENCES.items():
            assert seq.name == name
            assert len(seq.glyphs) > 0
            assert seq.pattern_type is not None
            assert len(seq.description) > 0
            assert len(seq.use_cases) > 0
            assert seq.domain in ["general", "biomedical", "cognitive", "social"]
            assert len(seq.references) > 0

    def test_bifurcated_sequences(self):
        """Test bifurcated pattern sequences."""
        assert BIFURCATED_BASE.pattern_type.value == "bifurcated"
        assert BIFURCATED_COLLAPSE.pattern_type.value == "bifurcated"
        assert Glyph.ZHIR in BIFURCATED_BASE.glyphs  # Mutation path
        assert Glyph.NUL in BIFURCATED_COLLAPSE.glyphs  # Collapse path


class TestApplyCanonicalSequence:
    """Test applying canonical sequences to networks."""

    def test_apply_bifurcated_base(self):
        """Apply bifurcated base sequence (mutation path)."""
        net = TNFRNetwork("test_bifurc", NetworkConfig(random_seed=42))
        net.add_nodes(1)
        net.apply_canonical_sequence("bifurcated_base")

        results = net.measure()
        assert results.coherence > 0.8, "Bifurcated base should maintain high coherence"
        assert len(results.sense_indices) == 1

    def test_apply_bifurcated_collapse(self):
        """Apply bifurcated collapse sequence (collapse path)."""
        net = TNFRNetwork("test_collapse", NetworkConfig(random_seed=42))
        net.add_nodes(1)
        net.apply_canonical_sequence("bifurcated_collapse")

        results = net.measure()
        assert results.coherence > 0.8, "Bifurcated collapse should maintain high coherence"

    def test_apply_therapeutic_protocol(self):
        """Apply therapeutic protocol sequence."""
        net = TNFRNetwork("test_therapy", NetworkConfig(random_seed=42))
        net.add_nodes(3)
        net.connect_nodes(0.4, "random")
        net.apply_canonical_sequence("therapeutic_protocol")

        results = net.measure()
        assert results.coherence > 0.5, "Therapeutic protocol should achieve moderate coherence"
        assert len(results.sense_indices) == 3

    def test_apply_theory_system(self):
        """Apply theory system sequence."""
        net = TNFRNetwork("test_theory", NetworkConfig(random_seed=42))
        net.add_nodes(2)
        net.connect_nodes(0.5, "ring")
        net.apply_canonical_sequence("theory_system")

        results = net.measure()
        assert results.coherence > 0.7, "Theory system should achieve good coherence"

    def test_apply_full_deployment(self):
        """Apply full deployment sequence."""
        net = TNFRNetwork("test_deploy", NetworkConfig(random_seed=42))
        net.add_nodes(4)
        net.connect_nodes(0.5, "random")
        net.apply_canonical_sequence("full_deployment")

        results = net.measure()
        assert results.coherence > 0.5, "Full deployment should achieve coherence"
        assert len(results.sense_indices) == 4

    def test_apply_mod_stabilizer(self):
        """Apply MOD_STABILIZER macro."""
        net = TNFRNetwork("test_mod", NetworkConfig(random_seed=42))
        net.add_nodes(1)
        net.apply_canonical_sequence("mod_stabilizer")

        results = net.measure()
        assert results.coherence >= 0.0, "MOD_STABILIZER should complete"

    def test_apply_invalid_sequence_raises(self):
        """Applying invalid sequence name should raise ValueError."""
        net = TNFRNetwork("test_invalid")
        net.add_nodes(1)

        with pytest.raises(ValueError, match="Unknown canonical sequence"):
            net.apply_canonical_sequence("nonexistent_sequence")

    def test_apply_without_nodes_raises(self):
        """Applying sequence without nodes should raise ValueError."""
        net = TNFRNetwork("test_empty")

        with pytest.raises(ValueError, match="No nodes in graph"):
            net.apply_canonical_sequence("bifurcated_base")

    def test_apply_to_specific_node(self):
        """Apply sequence to a specific node."""
        net = TNFRNetwork("test_specific", NetworkConfig(random_seed=42))
        net.add_nodes(3)
        nodes = list(net.graph.nodes())

        # Apply to first node explicitly
        net.apply_canonical_sequence("mod_stabilizer", node=nodes[0])

        results = net.measure()
        assert len(results.sense_indices) == 3

    def test_apply_to_invalid_node_raises(self):
        """Applying to non-existent node should raise ValueError."""
        net = TNFRNetwork("test_badnode", NetworkConfig(random_seed=42))
        net.add_nodes(1)

        with pytest.raises(ValueError, match="Node .* not found"):
            net.apply_canonical_sequence("bifurcated_base", node=9999)


class TestListCanonicalSequences:
    """Test listing and filtering canonical sequences."""

    def test_list_all_sequences(self):
        """List all canonical sequences."""
        net = TNFRNetwork("test_list")
        sequences = net.list_canonical_sequences()

        assert len(sequences) == 6
        assert all(isinstance(seq.glyphs, list) for seq in sequences.values())

    def test_filter_by_oz(self):
        """Filter sequences containing OZ."""
        net = TNFRNetwork("test_oz_filter")
        oz_sequences = net.list_canonical_sequences(with_oz=True)

        assert len(oz_sequences) == 6, "All canonical sequences should have OZ"
        for seq in oz_sequences.values():
            assert Glyph.OZ in seq.glyphs

    def test_filter_by_domain_biomedical(self):
        """Filter by biomedical domain."""
        net = TNFRNetwork("test_bio")
        bio_sequences = net.list_canonical_sequences(domain="biomedical")

        assert len(bio_sequences) >= 1
        for seq in bio_sequences.values():
            assert seq.domain == "biomedical"
        assert "therapeutic_protocol" in bio_sequences

    def test_filter_by_domain_cognitive(self):
        """Filter by cognitive domain."""
        net = TNFRNetwork("test_cognitive")
        cog_sequences = net.list_canonical_sequences(domain="cognitive")

        assert len(cog_sequences) >= 1
        for seq in cog_sequences.values():
            assert seq.domain == "cognitive"
        assert "theory_system" in cog_sequences

    def test_filter_by_domain_general(self):
        """Filter by general domain."""
        net = TNFRNetwork("test_general")
        gen_sequences = net.list_canonical_sequences(domain="general")

        assert len(gen_sequences) >= 3  # Most sequences are general
        for seq in gen_sequences.values():
            assert seq.domain == "general"

    def test_combined_filters(self):
        """Combine domain and OZ filters."""
        net = TNFRNetwork("test_combined")
        filtered = net.list_canonical_sequences(domain="general", with_oz=True)

        for seq in filtered.values():
            assert seq.domain == "general"
            assert Glyph.OZ in seq.glyphs


class TestSequenceOperatorComposition:
    """Test that sequences correctly compose operators."""

    def test_glyph_to_operator_mapping(self):
        """Verify glyph to operator mapping is complete."""
        from tnfr.operators.definitions import (
            Emission,
            Reception,
            Coherence,
            Dissonance,
            Coupling,
            Resonance,
            Silence,
            Expansion,
            Contraction,
            SelfOrganization,
            Mutation,
            Transition,
            Recursivity,
        )

        # All glyphs used in canonical sequences should map to operators
        all_glyphs = set()
        for seq in CANONICAL_SEQUENCES.values():
            all_glyphs.update(seq.glyphs)

        # Verify we have operators for all glyphs
        assert Glyph.AL in all_glyphs  # Emission
        assert Glyph.OZ in all_glyphs  # Dissonance
        assert Glyph.ZHIR in all_glyphs  # Mutation
        assert Glyph.IL in all_glyphs  # Coherence
        assert Glyph.SHA in all_glyphs  # Silence

    def test_sequence_grammar_structure(self):
        """Verify sequences follow reception→coherence grammar."""
        for name, seq in CANONICAL_SEQUENCES.items():
            # All sequences should have EN and IL (grammar requirement)
            assert Glyph.EN in seq.glyphs, f"{name} missing EN (reception)"
            assert Glyph.IL in seq.glyphs, f"{name} missing IL (coherence)"


class TestSequenceCoherenceOutcomes:
    """Test that sequences produce expected coherence outcomes."""

    def test_all_sequences_maintain_coherence(self):
        """All sequences should maintain reasonable coherence."""
        for seq_name in CANONICAL_SEQUENCES.keys():
            net = TNFRNetwork(f"test_{seq_name}", NetworkConfig(random_seed=42))
            net.add_nodes(2)
            net.connect_nodes(0.5, "random")
            net.apply_canonical_sequence(seq_name)

            results = net.measure()
            assert results.coherence >= 0.0, f"{seq_name} failed coherence check"
            assert results.coherence <= 1.0, f"{seq_name} coherence out of bounds"

    def test_sequences_produce_valid_sense_indices(self):
        """All sequences should produce valid sense indices."""
        for seq_name in CANONICAL_SEQUENCES.keys():
            net = TNFRNetwork(f"test_{seq_name}_si", NetworkConfig(random_seed=42))
            net.add_nodes(1)
            net.apply_canonical_sequence(seq_name)

            results = net.measure()
            assert len(results.sense_indices) > 0
            for si_value in results.sense_indices.values():
                assert si_value >= 0.0, f"{seq_name} produced negative Si"


class TestSequenceRepeatability:
    """Test that sequences are deterministic with seeds."""

    def test_same_seed_same_result(self):
        """Same seed should produce same coherence."""
        # First run
        net1 = TNFRNetwork("test_repeat1", NetworkConfig(random_seed=123))
        net1.add_nodes(3)
        net1.connect_nodes(0.4, "random")
        net1.apply_canonical_sequence("therapeutic_protocol")
        results1 = net1.measure()

        # Second run with same seed
        net2 = TNFRNetwork("test_repeat2", NetworkConfig(random_seed=123))
        net2.add_nodes(3)
        net2.connect_nodes(0.4, "random")
        net2.apply_canonical_sequence("therapeutic_protocol")
        results2 = net2.measure()

        assert (
            abs(results1.coherence - results2.coherence) < 0.01
        ), "Same seed should produce consistent coherence"
