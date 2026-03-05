#!/usr/bin/env python3
"""
Simple Signal Debug - Test with Mock Data
========================================
Tests signal generation with known good data to isolate the issue.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent / "future-research" / "crypto-lab" / "src"
sys.path.insert(0, str(project_root))

from gmx_trading_engine import GMXTradingSignal
from gmx_data_source import GMXMarketData
from gmx_tnfr_analyzer import TNFRStructuralMetrics

def create_mock_market_data():
    """Create realistic mock market data for testing."""
    return GMXMarketData(
        symbol="BTC",
        market_address="0x123",
        oracle_price=49950.0,
        mark_price=50000.0,
        index_price=49975.0,
        price_deviation=50.0,
        open_interest_long=1000000,
        open_interest_short=800000,
        oi_imbalance=2.0,  # Above φ (1.618) - indicates long heavy
        funding_rate=0.01,  # 1% funding rate - should trigger funding arbitrage
        funding_rate_velocity=0.001,
        pool_value=10000000,
        utilization_rate=0.8,
        available_liquidity=2000000,
        liquidations_24h=100,
        liquidation_volume_24h=5000000,
        liquidation_cascade_risk=0.35,  # Above 0.2904 limit - should trigger liquidation signal
        timestamp=datetime.now(),
        block_number=12345678
    )

def create_mock_tnfr_metrics():
    """Create TNFR metrics that should generate signals."""
    return TNFRStructuralMetrics(
        # Core TNFR Fields
        structural_potential=0.5,  # Below 2.0 threshold
        phase_gradient=0.1,  # Below 0.2904 limit  
        phase_curvature=1.0,
        coherence_length=1.0,
        
        # Unified Complex Fields
        complex_geometric_magnitude=0.8,
        chirality_field=0.1,  # Within limits
        symmetry_breaking_field=0.1,  # Within limits
        coherence_coupling=0.6,
        energy_density=1.0,
        topological_charge=0.0,
        
        # GMX-specific derived fields
        funding_coherence=0.7,
        liquidation_proximity=0.4,
        oi_structural_balance=0.8,
        oracle_phase_lock=0.8,  # Should be above threshold
        
        # Network topology
        market_connectivity=5,
        arbitrage_pathways=3,
        
        # Temporal dynamics  
        coherence_velocity=0.1,
        bifurcation_probability=0.5,  # Above 1/φ² = 0.382
        timestamp=datetime.now()
    )

def test_funding_arbitrage():
    """Test funding arbitrage signal generation."""
    print("🧪 TESTING FUNDING ARBITRAGE SIGNAL GENERATION")
    print("=" * 50)
    
    # Import directly to avoid initialization issues
    from gmx_trading_engine import GMXTNFRTradingEngine
    
    # Create engine (will fail to connect but we'll use internal methods)
    try:
        engine = GMXTNFRTradingEngine()
        
        # Create mock data
        market_data = {"BTC": create_mock_market_data()}
        metrics_map = {"BTC": create_mock_tnfr_metrics()}
        
        print("✅ Mock data created:")
        btc_data = market_data["BTC"]
        btc_metrics = metrics_map["BTC"]
        
        print(f"  BTC Mark Price: ${btc_data.mark_price:,.0f}")
        print(f"  Funding Rate: {btc_data.funding_rate:.2%}")
        print(f"  Liquidation Risk: {btc_data.liquidation_cascade_risk:.3f}")
        print(f"  OI Imbalance: {btc_data.oi_imbalance:.2f}")
        print(f"  Structural Potential: {btc_metrics.structural_potential:.3f}")
        print(f"  Phase Gradient: {btc_metrics.phase_gradient:.3f}")
        print(f"  Bifurcation Prob: {btc_metrics.bifurcation_probability:.3f}")
        
        # Test individual strategy methods
        print("\n🔍 TESTING INDIVIDUAL STRATEGIES:")
        
        # Test funding arbitrage
        print("\n1. Funding Arbitrage:")
        funding_signal = engine._analyze_funding_arbitrage("BTC", btc_data, btc_metrics)
        if funding_signal:
            print(f"✅ Generated funding signal:")
            print(f"   Action: {funding_signal.action}")
            print(f"   Confidence: {funding_signal.confidence:.3f}")
            print(f"   Basis: {funding_signal.structural_basis}")
        else:
            print("❌ No funding signal generated")
            
            # Debug why
            from tnfr.constants.canonical import PHI, INV_PHI, INV_E
            structural_stress = max(btc_metrics.phase_gradient, 1 / PHI**9)
            funding_threshold = structural_stress * INV_PHI
            noise_threshold = funding_threshold * INV_E
            
            print(f"   Debug: Funding {btc_data.funding_rate:.4%} vs threshold {funding_threshold:.4%}")
            print(f"   Debug: Noise threshold {noise_threshold:.4%}")
            print(f"   Debug: Min confidence: {engine.min_confidence}")
        
        # Test liquidation opportunity
        print("\n2. Liquidation Opportunity:")
        liquidation_signal = engine._analyze_liquidation_opportunity("BTC", btc_data, btc_metrics)
        if liquidation_signal:
            print(f"✅ Generated liquidation signal:")
            print(f"   Action: {liquidation_signal.action}")
            print(f"   Confidence: {liquidation_signal.confidence:.3f}")
            print(f"   Basis: {liquidation_signal.structural_basis}")
        else:
            print("❌ No liquidation signal generated")
            
            # Debug why
            STRUCTURAL_LIMIT_RISK = 0.2904
            print(f"   Debug: Liquidation risk {btc_data.liquidation_cascade_risk:.4f} vs limit {STRUCTURAL_LIMIT_RISK:.4f}")
            print(f"   Debug: Bifurcation prob {btc_metrics.bifurcation_probability:.4f} vs threshold {(1 / PHI**2):.4f}")
        
        # Test full signal generation
        print("\n3. Full Signal Generation:")
        all_signals = engine._generate_trading_signals(market_data, metrics_map)
        if all_signals:
            print(f"✅ Generated {len(all_signals)} signals:")
            for i, signal in enumerate(all_signals):
                print(f"   Signal {i+1}: {signal.symbol} {signal.action} (conf: {signal.confidence:.3f})")
        else:
            print("❌ No signals from full generation method")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_funding_arbitrage()