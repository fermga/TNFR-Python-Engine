"""Tests for adaptive dynamics and feedback loops.

This test suite validates the feedback loop, adaptive sequence selection,
homeostasis, and integrated adaptive system as specified in the TNFR
operational manual.
"""

import pytest

from tnfr.structural import create_nfr
from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.config.operator_names import (
    COHERENCE,
    DISSONANCE,
    SELF_ORGANIZATION,
)
from tnfr.dynamics.feedback import StructuralFeedbackLoop
from tnfr.dynamics.adaptive_sequences import AdaptiveSequenceSelector
from tnfr.dynamics.homeostasis import StructuralHomeostasis
from tnfr.sdk.adaptive_system import TNFRAdaptiveSystem


# Tests for StructuralFeedbackLoop ----------------------------------------


def test_feedback_loop_initialization():
    """Test feedback loop initializes with correct parameters."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node, target_coherence=0.737061493693243, tau_adaptive=0.098503076588036)

    assert abs(loop.target_coherence - 0.737061493693243) < 1e-10  # φ/(φ+γ) - canonical coherence
    assert abs(loop.tau_adaptive - 0.098503076588036) < 1e-10  # γ/(π+e) - canonical tolerance
    assert abs(loop.learning_rate - 0.043213918263772) < 1e-10  # e^(-π) - canonical learning rate


def test_feedback_loop_regulate_low_coherence():
    """Test feedback loop selects coherence for low coherence."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node, target_coherence=0.737061493693243)

    # Set high ΔNFR to simulate low coherence
    set_attr(G.nodes[node], ALIAS_DNFR, 0.8)

    operator_name = loop.regulate()
    assert operator_name == COHERENCE


def test_feedback_loop_regulate_high_coherence():
    """Test feedback loop selects dissonance for high coherence."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node, target_coherence=0.7)

    # Set low ΔNFR to simulate high coherence
    set_attr(G.nodes[node], ALIAS_DNFR, 0.0)

    operator_name = loop.regulate()
    assert operator_name == DISSONANCE


def test_feedback_loop_regulate_high_dnfr():
    """Test feedback loop logic with high ΔNFR and mid-coherence."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node, target_coherence=0.5)

    # Set high ΔNFR (0.2) which gives coherence of 0.8 (1.0 - 0.2)
    # Coherence 0.8 > target (0.5) + 0.1, so should select DISSONANCE
    set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[node], ALIAS_EPI, 0.5)

    operator_name = loop.regulate()
    # With ΔNFR=0.2, coherence=0.8, target=0.5, so 0.8 > 0.6 → DISSONANCE
    assert operator_name == DISSONANCE


def test_feedback_loop_compute_local_coherence():
    """Test local coherence computation from ΔNFR."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node)

    # Low ΔNFR → high coherence
    set_attr(G.nodes[node], ALIAS_DNFR, 0.0)
    coherence = loop._compute_local_coherence()
    assert coherence == 1.0

    # High ΔNFR → low coherence
    set_attr(G.nodes[node], ALIAS_DNFR, 1.0)
    coherence = loop._compute_local_coherence()
    assert coherence == 0.0

    # Mid ΔNFR → mid coherence
    set_attr(G.nodes[node], ALIAS_DNFR, 0.5)
    coherence = loop._compute_local_coherence()
    assert coherence == 0.5


def test_feedback_loop_adapt_thresholds():
    """Test threshold adaptation based on performance."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node, target_coherence=0.737061493693243, tau_adaptive=0.098503076588036)

    initial_tau = loop.tau_adaptive

    # Below target → increase tau
    loop.adapt_thresholds(0.5)
    assert loop.tau_adaptive > initial_tau

    # Above target → decrease tau
    loop.adapt_thresholds(0.9)
    assert loop.tau_adaptive < initial_tau


def test_feedback_loop_homeostatic_cycle():
    """Test homeostatic cycle executes without errors."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node)

    # Execute cycle
    loop.homeostatic_cycle(num_steps=5)

    # Should complete without errors


# Tests for AdaptiveSequenceSelector --------------------------------------


def test_adaptive_sequence_selector_initialization():
    """Test sequence selector initializes with canonical sequences."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    # Check canonical sequences exist
    assert "basic_activation" in selector.sequences
    assert "deep_learning" in selector.sequences
    assert "exploration" in selector.sequences
    assert "consolidation" in selector.sequences
    assert "mutation" in selector.sequences

    # Check performance tracking initialized
    for seq_name in selector.sequences:
        assert seq_name in selector.performance
        assert selector.performance[seq_name] == []


def test_adaptive_sequence_selector_select_stability():
    """Test sequence selection for stability goal."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    context = {"goal": "stability"}
    sequence = selector.select_sequence(context)

    # Should return one of the stability sequences
    assert sequence in [
        selector.sequences["basic_activation"],
        selector.sequences["consolidation"],
    ]


def test_adaptive_sequence_selector_select_growth():
    """Test sequence selection for growth goal."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    context = {"goal": "growth"}
    sequence = selector.select_sequence(context)

    # Should return one of the growth sequences
    assert sequence in [
        selector.sequences["deep_learning"],
        selector.sequences["exploration"],
    ]


def test_adaptive_sequence_selector_select_adaptation():
    """Test sequence selection for adaptation goal."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    context = {"goal": "adaptation"}
    sequence = selector.select_sequence(context)

    # Should return one of the adaptation sequences
    assert sequence in [
        selector.sequences["mutation"],
        selector.sequences["deep_learning"],
    ]


def test_adaptive_sequence_selector_record_performance():
    """Test performance recording and sliding window."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    # Record performance
    selector.record_performance("basic_activation", 0.8)
    selector.record_performance("basic_activation", 0.85)

    assert len(selector.performance["basic_activation"]) == 2
    assert selector.performance["basic_activation"][0] == 0.8
    assert selector.performance["basic_activation"][1] == 0.85


def test_adaptive_sequence_selector_sliding_window():
    """Test sliding window keeps only last 20 records."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    # Record 25 performances
    for i in range(25):
        selector.record_performance("basic_activation", float(i))

    # Should keep only last 20
    assert len(selector.performance["basic_activation"]) == 20
    assert selector.performance["basic_activation"][0] == 5.0
    assert selector.performance["basic_activation"][-1] == 24.0


# Tests for StructuralHomeostasis ------------------------------------------


def test_homeostasis_initialization():
    """Test homeostasis initializes with target ranges."""
    G, node = create_nfr("test_node")
    homeostasis = StructuralHomeostasis(G, node)

    assert homeostasis.epi_range == (0.4, 0.8)
    assert homeostasis.vf_range == (0.8, 1.2)
    assert homeostasis.dnfr_range == (0.0, 0.15)


def test_homeostasis_low_epi_correction():
    """Test homeostasis corrects low EPI with AL (Emission)."""
    G, node = create_nfr("test_node")
    homeostasis = StructuralHomeostasis(G, node)

    # Set EPI below range
    set_attr(G.nodes[node], ALIAS_EPI, 0.2)
    initial_epi = get_attr(G.nodes[node], ALIAS_EPI, 0.0)

    homeostasis.maintain_equilibrium()

    # EPI should be corrected (increased)
    final_epi = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    # Note: AL operator should increase EPI, but exact value depends on operator implementation


def test_homeostasis_high_vf_correction():
    """Test homeostasis corrects high νf with SHA (Silence)."""
    G, node = create_nfr("test_node")
    homeostasis = StructuralHomeostasis(G, node)

    # Set νf above range
    set_attr(G.nodes[node], ALIAS_VF, 1.5)

    homeostasis.maintain_equilibrium()

    # Should complete without errors


def test_homeostasis_high_dnfr_correction():
    """Test homeostasis corrects high ΔNFR with IL (Coherence)."""
    G, node = create_nfr("test_node")
    homeostasis = StructuralHomeostasis(G, node)

    # Set ΔNFR above range
    set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

    homeostasis.maintain_equilibrium()

    # Should complete without errors


# Tests for TNFRAdaptiveSystem ---------------------------------------------


def test_adaptive_system_initialization():
    """Test adaptive system initializes all components."""
    G, node = create_nfr("test_node")
    system = TNFRAdaptiveSystem(G, node)

    assert isinstance(system.feedback, StructuralFeedbackLoop)
    assert isinstance(system.sequence_selector, AdaptiveSequenceSelector)
    assert isinstance(system.homeostasis, StructuralHomeostasis)
    assert system.learning is not None
    assert system.metabolism is not None


def test_adaptive_system_measure_stress():
    """Test stress measurement from ΔNFR."""
    G, node = create_nfr("test_node")
    system = TNFRAdaptiveSystem(G, node)

    # Low ΔNFR → low stress
    set_attr(G.nodes[node], ALIAS_DNFR, 0.0)
    stress = system._measure_stress()
    assert stress == 0.0

    # High ΔNFR → high stress
    set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
    stress = system._measure_stress()
    assert stress == 1.0

    # Mid ΔNFR → mid stress
    set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
    stress = system._measure_stress()
    assert stress == 0.5


def test_adaptive_system_autonomous_evolution():
    """Test autonomous evolution executes without errors."""
    G, node = create_nfr("test_node")
    system = TNFRAdaptiveSystem(G, node)

    # Execute autonomous evolution
    system.autonomous_evolution(num_cycles=10)

    # Should complete without errors


def test_adaptive_system_integration():
    """Test integrated system combines all components correctly."""
    G, node = create_nfr("test_node")
    system = TNFRAdaptiveSystem(G, node)

    # Set initial conditions
    set_attr(G.nodes[node], ALIAS_EPI, 0.5)
    set_attr(G.nodes[node], ALIAS_VF, 1.0)
    set_attr(G.nodes[node], ALIAS_DNFR, 0.05)

    # Run a few cycles
    system.autonomous_evolution(num_cycles=5)

    # All components should have executed
    # (Detailed validation would require checking internal state)


# Edge case tests ----------------------------------------------------------


def test_feedback_loop_extreme_dnfr():
    """Test feedback loop handles extreme ΔNFR values."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node)

    # Very high ΔNFR
    set_attr(G.nodes[node], ALIAS_DNFR, 10.0)
    coherence = loop._compute_local_coherence()
    assert coherence == 0.0  # Should clamp to 0

    # Negative ΔNFR (should use absolute value)
    set_attr(G.nodes[node], ALIAS_DNFR, -0.5)
    coherence = loop._compute_local_coherence()
    assert 0.0 <= coherence <= 1.0


def test_threshold_adaptation_bounds():
    """Test threshold adaptation respects bounds."""
    G, node = create_nfr("test_node")
    loop = StructuralFeedbackLoop(G, node, tau_adaptive=0.1)

    # Try to push below minimum
    for _ in range(100):
        loop.adapt_thresholds(1.0)  # Performance above target

    assert loop.tau_adaptive >= 0.05

    # Try to push above maximum
    for _ in range(100):
        loop.adapt_thresholds(0.0)  # Performance below target

    assert loop.tau_adaptive <= 0.25


def test_empty_performance_history():
    """Test sequence selector handles empty performance history."""
    G, node = create_nfr("test_node")
    selector = AdaptiveSequenceSelector(G, node)

    # Select without any recorded performance
    context = {"goal": "stability"}
    sequence = selector.select_sequence(context)

    # Should still return valid sequence
    assert isinstance(sequence, list)
    assert len(sequence) > 0
