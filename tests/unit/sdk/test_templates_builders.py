"""Tests for TNFR SDK templates and builders."""

import pytest

from tnfr.sdk import TNFRTemplates, TNFRExperimentBuilder, NetworkResults


class TestTNFRTemplates:
    """Test TNFRTemplates pre-configured patterns."""

    def test_social_network_simulation(self):
        """Test social network simulation template."""
        results = TNFRTemplates.social_network_simulation(
            people=15, simulation_steps=3, random_seed=42
        )

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 15
        assert results.coherence >= 0.0
        assert results.coherence <= 1.0

    def test_neural_network_model(self):
        """Test neural network modeling template."""
        results = TNFRTemplates.neural_network_model(
            neurons=20, connectivity=0.15, activation_cycles=5, random_seed=42
        )

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 20

    def test_ecosystem_dynamics(self):
        """Test ecosystem dynamics template."""
        results = TNFRTemplates.ecosystem_dynamics(species=12, evolution_steps=10, random_seed=42)

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 12

    def test_creative_process_model(self):
        """Test creative process modeling template."""
        results = TNFRTemplates.creative_process_model(
            ideas=10, development_cycles=6, random_seed=42
        )

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 10

    def test_organizational_network(self):
        """Test organizational network template."""
        results = TNFRTemplates.organizational_network(
            agents=20, coordination_steps=5, random_seed=42
        )

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 20


class TestTNFRExperimentBuilder:
    """Test TNFRExperimentBuilder research patterns."""

    def test_small_world_study(self):
        """Test small-world network study."""
        results = TNFRExperimentBuilder.small_world_study(
            nodes=15, rewiring_prob=0.1, steps=2, random_seed=42
        )

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 15

    def test_synchronization_study(self):
        """Test synchronization study."""
        results = TNFRExperimentBuilder.synchronization_study(
            nodes=12, coupling_strength=0.4, steps=5, random_seed=42
        )

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 12

    def test_creativity_emergence(self):
        """Test creativity emergence study."""
        results = TNFRExperimentBuilder.creativity_emergence(nodes=10, steps=3, random_seed=42)

        assert isinstance(results, NetworkResults)
        assert len(results.sense_indices) == 10

    def test_compare_topologies(self):
        """Test topology comparison."""
        comparison = TNFRExperimentBuilder.compare_topologies(
            node_count=12, steps=2, random_seed=42
        )

        assert isinstance(comparison, dict)
        assert "random" in comparison
        assert "ring" in comparison
        assert "small_world" in comparison

        for topology, results in comparison.items():
            assert isinstance(results, NetworkResults)
            assert len(results.sense_indices) == 12

    def test_phase_transition_study(self):
        """Test phase transition study."""
        try:
            import numpy as np

            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False

        if not HAS_NUMPY:
            pytest.skip("NumPy not available")

        transition = TNFRExperimentBuilder.phase_transition_study(
            nodes=15, coupling_levels=3, steps_per_level=2, random_seed=42
        )

        assert isinstance(transition, dict)
        assert len(transition) == 3

        for coupling, results in transition.items():
            assert isinstance(results, NetworkResults)
            assert len(results.sense_indices) == 15

    def test_resilience_study(self):
        """Test network resilience study."""
        resilience = TNFRExperimentBuilder.resilience_study(
            nodes=15,
            initial_steps=2,
            perturbation_steps=1,
            recovery_steps=2,
            random_seed=42,
        )

        assert isinstance(resilience, dict)
        assert "initial" in resilience
        assert "perturbed" in resilience
        assert "recovered" in resilience

        for state, results in resilience.items():
            assert isinstance(results, NetworkResults)
            assert len(results.sense_indices) == 15

        # Check that resilience shows recovery
        initial_c = resilience["initial"].coherence
        perturbed_c = resilience["perturbed"].coherence
        recovered_c = resilience["recovered"].coherence

        # All should be valid coherence values
        assert 0.0 <= initial_c <= 1.0
        assert 0.0 <= perturbed_c <= 1.0
        assert 0.0 <= recovered_c <= 1.0
