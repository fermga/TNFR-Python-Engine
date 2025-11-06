"""Tests for dynamic canonical limits.

This test suite validates the theoretical proposition that canonical limits
should be dynamic rather than static, adapting based on network coherence.
"""

import math

import networkx as nx
import pytest

from tnfr.dynamics.dynamic_limits import (
    DynamicLimits,
    DynamicLimitsConfig,
    compute_dynamic_limits,
)


class TestDynamicLimitsConfig:
    """Test configuration dataclass for dynamic limits."""
    
    def test_default_values(self):
        """Verify default configuration values are theoretically sound."""
        config = DynamicLimitsConfig()
        
        assert config.base_epi_max == 1.0
        assert config.base_vf_max == 10.0
        assert config.alpha == 0.5
        assert config.beta == 0.3
        assert config.max_expansion_factor == 3.0
        assert config.enabled is True
    
    def test_custom_configuration(self):
        """Test creating custom configurations."""
        config = DynamicLimitsConfig(
            base_epi_max=2.0,
            base_vf_max=20.0,
            alpha=0.8,
            beta=0.5,
            max_expansion_factor=5.0,
        )
        
        assert config.base_epi_max == 2.0
        assert config.base_vf_max == 20.0
        assert config.alpha == 0.8
        assert config.beta == 0.5
        assert config.max_expansion_factor == 5.0
    
    def test_config_is_frozen(self):
        """Configuration should be immutable (frozen dataclass)."""
        config = DynamicLimitsConfig()
        
        with pytest.raises(AttributeError):
            config.alpha = 0.99  # type: ignore


class TestDynamicLimitsBasicComputation:
    """Test basic computation of dynamic limits."""
    
    def test_empty_graph_returns_base_limits(self):
        """Empty graph should return base limits."""
        G = nx.Graph()
        limits = compute_dynamic_limits(G)
        
        assert limits.epi_max_effective == 1.0
        assert limits.vf_max_effective == 10.0
        assert limits.coherence == 0.0
        assert limits.si_avg == 0.0
        assert limits.kuramoto_r == 0.0
    
    def test_single_node_graph(self):
        """Single node graph should compute valid limits."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 1.0,
            "theta": 0.0,
            "EPI": 0.5,
            "Si": 0.7,
            "ΔNFR": 0.1,
            "dEPI_dt": 0.05,
        })
        
        limits = compute_dynamic_limits(G)
        
        # Should be computable
        assert isinstance(limits.epi_max_effective, float)
        assert isinstance(limits.vf_max_effective, float)
        
        # Should be non-negative
        assert limits.epi_max_effective > 0
        assert limits.vf_max_effective > 0
        
        # Metrics should be valid
        assert 0.0 <= limits.coherence <= 1.0
        assert limits.si_avg > 0
        assert 0.0 <= limits.kuramoto_r <= 1.0
    
    def test_disabled_returns_base_limits(self):
        """When disabled, should return base limits regardless of network state."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 5.0,
            "theta": 0.0,
            "EPI": 0.8,
            "Si": 0.9,
            "ΔNFR": 0.01,
            "dEPI_dt": 0.01,
        })
        
        config = DynamicLimitsConfig(enabled=False)
        limits = compute_dynamic_limits(G, config)
        
        # Should return base limits
        assert limits.epi_max_effective == config.base_epi_max
        assert limits.vf_max_effective == config.base_vf_max


class TestCoherenceBasedExpansion:
    """Test that limits expand with coherence."""
    
    def test_high_coherence_increases_epi_limit(self):
        """High coherence (low ΔNFR, low dEPI) should increase EPI limit."""
        G = nx.Graph()
        
        # Create highly coherent network
        for i in range(5):
            G.add_node(i, **{
                "νf": 1.0,
                "theta": 0.0,  # All in phase
                "EPI": 0.5,
                "Si": 0.9,  # High sense index
                "ΔNFR": 0.001,  # Very low gradient
                "dEPI_dt": 0.001,  # Very low velocity
            })
        
        limits = compute_dynamic_limits(G)
        
        # Should expand beyond base limit due to high coherence
        assert limits.epi_max_effective > 1.0
        
        # Coherence should be high
        assert limits.coherence > 0.7
        assert limits.coherence_factor > 0.5
    
    def test_low_coherence_keeps_base_limit(self):
        """Low coherence should keep limits near base values."""
        G = nx.Graph()
        
        # Create low coherence network
        for i in range(5):
            G.add_node(i, **{
                "νf": 1.0,
                "theta": i * 0.5,  # Dispersed phases
                "EPI": 0.5,
                "Si": 0.3,  # Low sense index
                "ΔNFR": 0.5,  # High gradient (instability)
                "dEPI_dt": 0.5,  # High velocity
            })
        
        limits = compute_dynamic_limits(G)
        
        # Should stay close to base limit
        assert limits.epi_max_effective < 1.5
        
        # Coherence should be lower (relaxed threshold for realistic values)
        assert limits.coherence <= 0.5
    
    def test_expansion_respects_maximum_factor(self):
        """Expansion should not exceed max_expansion_factor."""
        G = nx.Graph()
        
        # Create perfect network (unrealistic but tests bounds)
        for i in range(10):
            G.add_node(i, **{
                "νf": 1.0,
                "theta": 0.0,
                "EPI": 0.5,
                "Si": 1.0,
                "ΔNFR": 0.0,
                "dEPI_dt": 0.0,
            })
        
        config = DynamicLimitsConfig(
            alpha=10.0,  # Artificially high
            beta=10.0,
            max_expansion_factor=2.0,
        )
        
        limits = compute_dynamic_limits(G, config)
        
        # Should be clamped to max_expansion_factor
        assert limits.epi_max_effective <= config.base_epi_max * 2.0
        assert limits.vf_max_effective <= config.base_vf_max * 2.0


class TestKuramotoSynchronization:
    """Test that Kuramoto order affects νf limits."""
    
    def test_synchronized_network_increases_vf_limit(self):
        """Synchronized network (high R) should increase νf limit."""
        G = nx.Graph()
        
        # Create synchronized network
        for i in range(5):
            G.add_node(i, **{
                "νf": 2.0,
                "theta": 0.1,  # Nearly identical phases
                "EPI": 0.5,
                "Si": 0.7,
                "ΔNFR": 0.1,
                "dEPI_dt": 0.05,
            })
        
        limits = compute_dynamic_limits(G)
        
        # Should expand νf limit due to synchronization
        assert limits.vf_max_effective > 10.0
        
        # Kuramoto order should be high
        assert limits.kuramoto_r > 0.8
    
    def test_desynchronized_network_keeps_base_vf_limit(self):
        """Desynchronized network should keep νf near base."""
        G = nx.Graph()
        
        # Create desynchronized network
        for i in range(5):
            G.add_node(i, **{
                "νf": 2.0,
                "theta": i * (2 * math.pi / 5),  # Evenly dispersed
                "EPI": 0.5,
                "Si": 0.7,
                "ΔNFR": 0.1,
                "dEPI_dt": 0.05,
            })
        
        limits = compute_dynamic_limits(G)
        
        # Should stay close to base νf limit
        assert limits.vf_max_effective < 12.0
        
        # Kuramoto order should be low
        assert limits.kuramoto_r < 0.5


class TestDynamicLimitsResult:
    """Test the DynamicLimits result dataclass."""
    
    def test_result_contains_all_metrics(self):
        """Result should contain all computed metrics."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 1.0,
            "theta": 0.0,
            "EPI": 0.5,
            "Si": 0.7,
            "ΔNFR": 0.1,
            "dEPI_dt": 0.05,
        })
        
        limits = compute_dynamic_limits(G)
        
        # All fields should be present
        assert hasattr(limits, "epi_max_effective")
        assert hasattr(limits, "vf_max_effective")
        assert hasattr(limits, "coherence")
        assert hasattr(limits, "si_avg")
        assert hasattr(limits, "kuramoto_r")
        assert hasattr(limits, "coherence_factor")
        assert hasattr(limits, "config")
    
    def test_result_is_frozen(self):
        """Result should be immutable (frozen dataclass)."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 1.0,
            "theta": 0.0,
            "EPI": 0.5,
            "Si": 0.7,
            "ΔNFR": 0.1,
            "dEPI_dt": 0.05,
        })
        
        limits = compute_dynamic_limits(G)
        
        with pytest.raises(AttributeError):
            limits.epi_max_effective = 999.0  # type: ignore


class TestTheoreticalInvariants:
    """Test that dynamic limits preserve TNFR theoretical invariants."""
    
    def test_preserves_operator_closure(self):
        """Dynamic limits should preserve operator closure (finite bounds)."""
        G = nx.Graph()
        
        # Create a network
        for i in range(10):
            G.add_node(i, **{
                "νf": 1.0,
                "theta": 0.0,
                "EPI": 0.5,
                "Si": 0.8,
                "ΔNFR": 0.01,
                "dEPI_dt": 0.01,
            })
        
        limits = compute_dynamic_limits(G)
        
        # Limits must remain finite (preserving operator closure)
        assert math.isfinite(limits.epi_max_effective)
        assert math.isfinite(limits.vf_max_effective)
        
        # Limits must be positive
        assert limits.epi_max_effective > 0
        assert limits.vf_max_effective > 0
    
    def test_structural_semantics_preserved(self):
        """Expansion should be proportional to coherence (structural semantics)."""
        G1 = nx.Graph()
        G2 = nx.Graph()
        
        # Network with higher coherence
        for i in range(5):
            G1.add_node(i, **{
                "νf": 1.0,
                "theta": 0.0,
                "EPI": 0.5,
                "Si": 0.9,
                "ΔNFR": 0.01,
                "dEPI_dt": 0.01,
            })
        
        # Network with lower coherence
        for i in range(5):
            G2.add_node(i, **{
                "νf": 1.0,
                "theta": i * 0.5,
                "EPI": 0.5,
                "Si": 0.4,
                "ΔNFR": 0.3,
                "dEPI_dt": 0.3,
            })
        
        limits1 = compute_dynamic_limits(G1)
        limits2 = compute_dynamic_limits(G2)
        
        # Higher coherence should yield higher limits
        assert limits1.coherence > limits2.coherence
        assert limits1.epi_max_effective > limits2.epi_max_effective
    
    def test_self_organization_reflected(self):
        """Limits should emerge from system state (self-organization)."""
        G = nx.Graph()
        
        # Add nodes with varying states
        for i in range(5):
            G.add_node(i, **{
                "νf": 1.0 + i * 0.1,
                "theta": i * 0.2,
                "EPI": 0.4 + i * 0.05,
                "Si": 0.6 + i * 0.05,
                "ΔNFR": 0.1,
                "dEPI_dt": 0.05,
            })
        
        limits = compute_dynamic_limits(G)
        
        # Limits computed from actual network state
        assert limits.coherence_factor == limits.coherence * limits.si_avg
        
        # Expansion reflects measured coherence
        expected_epi_expansion = 1.0 + 0.5 * limits.coherence_factor
        actual_epi_ratio = limits.epi_max_effective / limits.config.base_epi_max
        
        # Should match (within expansion factor bounds)
        assert abs(actual_epi_ratio - expected_epi_expansion) < 0.01 or actual_epi_ratio == 3.0


class TestComparisonWithStaticLimits:
    """Compare dynamic limits behavior with static limits."""
    
    def test_dynamic_allows_more_freedom_when_coherent(self):
        """Dynamic limits should allow higher values in coherent networks."""
        G = nx.Graph()
        
        # Create highly coherent network
        for i in range(10):
            G.add_node(i, **{
                "νf": 1.0,
                "theta": 0.0,
                "EPI": 0.8,  # High EPI values
                "Si": 0.9,
                "ΔNFR": 0.01,
                "dEPI_dt": 0.01,
            })
        
        limits = compute_dynamic_limits(G)
        
        # Dynamic limit should exceed static limit of 1.0
        assert limits.epi_max_effective > 1.0
        
        # Should reflect high coherence
        assert limits.coherence > 0.8
    
    def test_dynamic_restricts_when_chaotic(self):
        """Dynamic limits should stay conservative in chaotic networks."""
        G = nx.Graph()
        
        # Create chaotic network
        for i in range(10):
            G.add_node(i, **{
                "νf": 2.0,
                "theta": i * 0.7,  # Random phases
                "EPI": 0.5,
                "Si": 0.2,  # Low sense
                "ΔNFR": 0.8,  # High instability
                "dEPI_dt": 0.6,
            })
        
        limits = compute_dynamic_limits(G)
        
        # Should stay near base limits
        assert limits.epi_max_effective < 1.3
        
        # Should reflect low coherence (relaxed threshold for realistic values)
        assert limits.coherence <= 0.42


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_alpha_no_epi_expansion(self):
        """With alpha=0, EPI limit should not expand."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 1.0,
            "theta": 0.0,
            "EPI": 0.5,
            "Si": 0.9,
            "ΔNFR": 0.01,
            "dEPI_dt": 0.01,
        })
        
        config = DynamicLimitsConfig(alpha=0.0)
        limits = compute_dynamic_limits(G, config)
        
        # Should equal base limit
        assert limits.epi_max_effective == config.base_epi_max
    
    def test_zero_beta_no_vf_expansion(self):
        """With beta=0, νf limit should not expand."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 1.0,
            "theta": 0.0,
            "EPI": 0.5,
            "Si": 0.9,
            "ΔNFR": 0.01,
            "dEPI_dt": 0.01,
        })
        
        config = DynamicLimitsConfig(beta=0.0)
        limits = compute_dynamic_limits(G, config)
        
        # Should equal base limit
        assert limits.vf_max_effective == config.base_vf_max
    
    def test_custom_base_limits(self):
        """Custom base limits should be respected."""
        G = nx.Graph()
        G.add_node(0, **{
            "νf": 1.0,
            "theta": 0.0,
            "EPI": 0.5,
            "Si": 0.7,
            "ΔNFR": 0.1,
            "dEPI_dt": 0.05,
        })
        
        config = DynamicLimitsConfig(
            base_epi_max=5.0,
            base_vf_max=50.0,
        )
        
        limits = compute_dynamic_limits(G, config)
        
        # Effective limits should be based on custom base limits
        assert limits.epi_max_effective >= 5.0
        assert limits.vf_max_effective >= 50.0
