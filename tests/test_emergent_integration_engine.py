"""Test suite for TNFR Emergent Integration Engine."""

import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tnfr.dynamics.emergent_integration_engine import (
        TNFREmergentIntegrationEngine,
        get_emergent_integration_engine,
        discover_and_apply_integrations,
        IntegrationOpportunity,
        IntegrationPattern
    )
    HAS_INTEGRATION_ENGINE = True
except ImportError:
    HAS_INTEGRATION_ENGINE = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

pytest_mark_integration = pytest.mark.skipif(
    not HAS_INTEGRATION_ENGINE, 
    reason="Integration engine not available"
)


@pytest_mark_integration
def test_integration_engine_initialization():
    """Test that integration engine initializes correctly."""
    engine = TNFREmergentIntegrationEngine()
    assert engine is not None
    
    # Check engine availability tracking
    stats = engine.get_integration_statistics()
    assert "engines_available" in stats
    assert isinstance(stats["engines_available"], dict)


@pytest_mark_integration 
@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available")
def test_discover_integration_opportunities():
    """Test discovery of integration opportunities."""
    engine = TNFREmergentIntegrationEngine()
    
    # Create test graph
    G = nx.path_graph(10)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['EPI'] = float(i * 0.1)
        G.nodes[node]['nu_f'] = 1.0 + float(i * 0.05) 
        G.nodes[node]['phase'] = float(i * 0.2)
    
    # Discover opportunities
    opportunities = engine.discover_integration_opportunities(G)
    
    assert isinstance(opportunities, list)
    # Should discover at least some opportunities for a 10-node graph
    if opportunities:
        for opp in opportunities:
            assert isinstance(opp, IntegrationPattern)
            assert hasattr(opp, 'opportunity_type')
            assert hasattr(opp, 'confidence_score')
            assert 0.0 <= opp.confidence_score <= 1.0


@pytest_mark_integration
@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available") 
def test_apply_integration_pattern():
    """Test applying integration patterns."""
    engine = TNFREmergentIntegrationEngine()
    
    # Create test graph
    G = nx.complete_graph(5)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['EPI'] = float(i * 0.1)
        G.nodes[node]['nu_f'] = 1.0
        G.nodes[node]['phase'] = 0.0
        
    # Create mock integration pattern
    pattern = IntegrationPattern(
        pattern_id="test_pattern",
        opportunity_type=IntegrationOpportunity.SPECTRAL_SHARING,
        mathematical_basis="Test mathematical basis",
        involved_engines={"test_engine"},
        integration_strategy={"method": "test_method"},
        expected_benefit={"computation_time_reduction": 0.1},
        mathematical_requirements=["test_requirement"],
        confidence_score=0.8
    )
    
    # Apply pattern
    result = engine.apply_integration_pattern(pattern, G, validate_mathematics=False)
    
    assert hasattr(result, 'pattern_applied')
    assert hasattr(result, 'success')
    assert hasattr(result, 'mathematical_consistency_maintained')
    assert result.pattern_applied == "test_pattern"


@pytest_mark_integration
def test_global_integration_engine():
    """Test global integration engine accessor."""
    engine1 = get_emergent_integration_engine()
    engine2 = get_emergent_integration_engine()
    
    # Should return same instance (singleton pattern)
    assert engine1 is engine2


@pytest_mark_integration
@pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX not available")
def test_discover_and_apply_integrations():
    """Test convenience function for discovery and application."""
    # Create test graph
    G = nx.cycle_graph(8)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['EPI'] = float(i * 0.05)
        G.nodes[node]['nu_f'] = 1.0 + float(i * 0.02)
        G.nodes[node]['phase'] = float(i * 0.1)
        
    # Test discovery without auto-apply
    results = discover_and_apply_integrations(G, auto_apply=False)
    
    assert "opportunities_discovered" in results
    assert "opportunity_details" in results
    assert "integration_results" in results
    assert "engine_statistics" in results
    
    assert isinstance(results["opportunities_discovered"], int)
    assert results["opportunities_discovered"] >= 0


@pytest_mark_integration
def test_integration_statistics():
    """Test integration statistics collection."""
    engine = TNFREmergentIntegrationEngine()
    
    stats = engine.get_integration_statistics()
    
    required_fields = [
        "total_opportunities_discovered",
        "total_patterns_discovered", 
        "total_integrations_attempted",
        "successful_integrations",
        "success_rate",
        "engines_available"
    ]
    
    for field in required_fields:
        assert field in stats
        
    # Check engines availability tracking
    engines_available = stats["engines_available"]
    assert isinstance(engines_available, dict)
    
    expected_engines = [
        "cache_orchestrator",
        "optimization_orchestrator",
        "self_optimizer",
        "spectral_fusion",
        "centralization",
        "nodal_optimizer",
        "structural_cache",
        "fft_cache"
    ]
    
    for engine_name in expected_engines:
        assert engine_name in engines_available
        assert isinstance(engines_available[engine_name], bool)


def test_integration_engine_without_dependencies():
    """Test integration engine behavior when dependencies unavailable."""
    # This test should pass even without optional dependencies
    if HAS_INTEGRATION_ENGINE:
        engine = TNFREmergentIntegrationEngine()
        
        # Should handle missing dependencies gracefully
        stats = engine.get_integration_statistics()
        assert stats is not None
        assert "engines_available" in stats
        
        # Test with None graph
        opportunities = engine.discover_integration_opportunities(None)
        assert isinstance(opportunities, list)


if __name__ == "__main__":
    # Run basic tests
    if HAS_INTEGRATION_ENGINE:
        test_integration_engine_initialization()
        print("✓ Integration engine initialization test passed")
        
        test_integration_engine_without_dependencies()
        print("✓ Missing dependencies handling test passed")
        
        test_global_integration_engine()  
        print("✓ Global integration engine test passed")
        
        test_integration_statistics()
        print("✓ Integration statistics test passed")
        
        if HAS_NETWORKX:
            test_discover_integration_opportunities()
            print("✓ Opportunity discovery test passed")
            
            test_apply_integration_pattern()
            print("✓ Pattern application test passed")
            
            test_discover_and_apply_integrations()
            print("✓ Convenience function test passed")
            
        print("\n🎯 All emergent integration engine tests passed!")
    else:
        print("⚠️ Integration engine not available - skipping tests")