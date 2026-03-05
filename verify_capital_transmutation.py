
import sys
import os
import asyncio
from datetime import datetime
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "future-research", "crypto-lab", "src"))

from gmx_trading_engine import GMXTNFRTradingEngine, GMXTradingSignal
from gmx_tnfr_analyzer import TNFRStructuralMetrics
from gmx_data_source import GMXMarketData

def create_mock_metrics(potential=1.0, coherence=50.0, gradient=0.1, bifurcation=0.1):
    return TNFRStructuralMetrics(
        structural_potential=potential,
        phase_gradient=gradient,
        phase_curvature=0.5,
        coherence_length=coherence,
        complex_geometric_magnitude=1.0,
        chirality_field=0.1,
        symmetry_breaking_field=0.1,
        coherence_coupling=0.5,
        energy_density=1.0,
        topological_charge=0.0,
        funding_coherence=0.8,
        liquidation_proximity=0.2,
        oi_structural_balance=0.5,
        oracle_phase_lock=0.9,
        market_connectivity=5,
        arbitrage_pathways=3,
        coherence_velocity=0.01,
        bifurcation_probability=bifurcation,
        timestamp=datetime.now()
    )

def test_capital_states():
    print("🧪 Testing Capital Transmutation Logic...")
    engine = GMXTNFRTradingEngine()
    
    # Test 1: SOLID STATE
    # High Coherence (>50), Low Gradient (<0.15)
    metrics_solid = create_mock_metrics(coherence=60.0, gradient=0.1)
    state_solid = engine._determine_capital_state(metrics_solid)
    print(f"Test 1 (SOLID): Expected 'SOLID', Got '{state_solid}' -> {'✅' if state_solid == 'SOLID' else '❌'}")
    
    # Test 2: GASEOUS STATE
    # High Potential (>1.618)
    metrics_gaseous_pot = create_mock_metrics(potential=1.7, coherence=20.0)
    state_gaseous_pot = engine._determine_capital_state(metrics_gaseous_pot)
    print(f"Test 2a (GASEOUS - Potential): Expected 'GASEOUS', Got '{state_gaseous_pot}' -> {'✅' if state_gaseous_pot == 'GASEOUS' else '❌'}")
    
    # High Bifurcation (>0.6)
    metrics_gaseous_bif = create_mock_metrics(bifurcation=0.7, coherence=20.0)
    state_gaseous_bif = engine._determine_capital_state(metrics_gaseous_bif)
    print(f"Test 2b (GASEOUS - Bifurcation): Expected 'GASEOUS', Got '{state_gaseous_bif}' -> {'✅' if state_gaseous_bif == 'GASEOUS' else '❌'}")
    
    # Test 3: LIQUID STATE
    # Default (Low Coherence, Low Potential)
    metrics_liquid = create_mock_metrics(coherence=30.0, potential=1.2, gradient=0.2)
    state_liquid = engine._determine_capital_state(metrics_liquid)
    print(f"Test 3 (LIQUID): Expected 'LIQUID', Got '{state_liquid}' -> {'✅' if state_liquid == 'LIQUID' else '❌'}")

def test_signal_generation():
    print("\n🧪 Testing Signal Generation by State...")
    engine = GMXTNFRTradingEngine()
    
    # Mock Analyzer
    engine.analyzer = MagicMock()
    engine.analyzer.calculate_fractal_metrics.return_value = {'fractal_coherence': 0.8}
    engine.analyzer.ecosystem_metrics = None
    
    # Mock Optimization Engine
    engine.optimization_engine = MagicMock()
    # IMPORTANT: Make apply_policies return the signal as-is, otherwise it returns a Mock which breaks logic
    engine.optimization_engine.apply_policies.side_effect = lambda sig, *args: sig
    
    # Mock Market Data
    market_data = {
        "BTC": GMXMarketData(
            symbol="BTC",
            market_address="0x123",
            oracle_price=50000.0,
            mark_price=50000.0,
            index_price=50000.0,
            price_deviation=0.0,
            open_interest_long=1000.0,
            open_interest_short=1000.0,
            oi_imbalance=1.0,
            funding_rate=0.0001,
            funding_rate_velocity=0.0,
            pool_value=1e6,
            utilization_rate=0.5,
            available_liquidity=5e5,
            liquidations_24h=0,
            liquidation_volume_24h=0.0,
            liquidation_cascade_risk=0.1,
            timestamp=datetime.now(),
            block_number=100
        )
    }
    
    # Case 1: SOLID -> Should generate LP Signal
    metrics_solid = create_mock_metrics(coherence=60.0, gradient=0.1)
    signals = engine._generate_trading_signals(market_data, metrics_solid)
    
    lp_signal = next((s for s in signals if s.signal_type == "LIQUIDITY_PROVISION"), None)
    if lp_signal:
        print(f"Case 1 (SOLID): Generated LP Signal -> ✅")
        print(f"   Action: {lp_signal.action}")
        print(f"   Basis: {lp_signal.structural_basis}")
    else:
        print(f"Case 1 (SOLID): No LP Signal -> ❌")
        
    # Reset active signals to prevent hysteresis from blocking the new signal
    engine.active_signals = {}
        
    # Case 2: LIQUID -> Should generate Funding Signal (if arb exists)
    # We need to mock _analyze_funding_arbitrage to return something
    engine._analyze_funding_arbitrage = MagicMock(return_value=GMXTradingSignal(
        symbol="BTC", action="LONG", confidence=0.8, signal_type="FUNDING_ARBITRAGE", structural_basis="Mock"
    ))
    
    metrics_liquid = create_mock_metrics(coherence=30.0, gradient=0.2)
    metrics_liquid.pool_coherence = 0.1 # Disable LP signal to ensure Funding Signal dominates
    signals_liquid = engine._generate_trading_signals(market_data, metrics_liquid)
    
    fund_signal = next((s for s in signals_liquid if s.signal_type == "FUNDING_ARBITRAGE"), None)
    if fund_signal:
        print(f"Case 2 (LIQUID): Generated Funding Signal -> ✅")
    else:
        print(f"Case 2 (LIQUID): No Funding Signal -> ❌")

def test_signal_instantiation():
    print("\n🧪 Testing Signal Instantiation...")
    try:
        s = GMXTradingSignal(
            symbol="TEST",
            action="MINT",
            confidence=0.9,
            signal_type="LIQUIDITY_PROVISION",
            structural_basis="Test",
            primary_metric="coherence_length",
            invalidation_threshold=40.0,
            exhaustion_threshold=0.0,
            entry_price=100.0,
            target_price=None,
            stop_loss=None,
            position_size_pct=0.5,
            time_horizon_hours=24,
            execution_window_minutes=120,
            gmx_execution_strategy="MINT_GM",
            structural_potential=1.0,
            coherence_length=60.0,
            bifurcation_probability=0.1,
            capital_state="SOLID",
            metadata={'fractal_coherence': 0.8}
        )
        print(f"Signal Instantiation -> ✅")
    except Exception as e:
        print(f"Signal Instantiation -> ❌ Error: {e}")

if __name__ == "__main__":
    test_capital_states()
    test_signal_instantiation()
    test_signal_generation()
