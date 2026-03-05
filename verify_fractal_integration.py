import sys
import os
import asyncio
import traceback
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'future-research', 'crypto-lab', 'src'))

from gmx_trading_engine import GMXTNFRTradingEngine as GMXTradingEngine, GMXMarketData, TNFRStructuralMetrics

async def verify_integration():
    print("🚀 Verifying Fractal Integration in GMX Trading Engine...")
    
    engine = GMXTradingEngine()
    
    # Mock Data
    data = GMXMarketData(
        symbol="BTC",
        mark_price=50000.0,
        index_price=50000.0,
        open_interest_long=1000000.0,
        open_interest_short=800000.0,
        borrowing_rate_long=0.0001,
        borrowing_rate_short=0.0001,
        funding_rate=0.0001,
        price_deviation=0.0,
        liquidation_cascade_risk=0.1,
        impact_factor_long=0.0001,
        impact_factor_short=0.0001,
        oi_imbalance=1.25,
        timestamp=datetime.now(),
        oracle_price=50000.0,
        market_address="0x123",
        funding_rate_velocity=0.0,
        pool_value=1000000.0,
        utilization_rate=0.5,
        available_liquidity=500000.0,
        liquidations_24h=0,
        liquidation_volume_24h=0.0,
        block_number=123456,
        long_token_amount=0.0,
        short_token_amount=0.0,
        pair_name="BTC-USDC",
        long_token_symbol="BTC",
        short_token_symbol="USDC"
    )
    
    # Fix field names
    data.open_interest_short = 800000.0
    
    # Mock Metrics
    metrics = TNFRStructuralMetrics(
        structural_potential=1.5,
        phase_gradient=0.1,
        phase_curvature=0.0,
        coherence_length=100.0,
        
        complex_geometric_magnitude=0.1,
        chirality_field=0.1,
        symmetry_breaking_field=0.1,
        coherence_coupling=0.1,
        energy_density=0.1,
        topological_charge=0.0,
        
        funding_coherence=0.8,
        liquidation_proximity=0.5,
        oi_structural_balance=0.5,
        oracle_phase_lock=0.9,
        
        market_connectivity=1,
        arbitrage_pathways=1,
        
        coherence_velocity=0.0,
        bifurcation_probability=0.2,
        
        timestamp=datetime.now(),
        
        pool_coherence=0.8,
        arbitrage_potential=0.0,
        liquidity_stress_index=0.5,
        impact_asymmetry=0.06, # Set to trigger signal (> 0.05)
        structural_inertia=1000.0,
        structural_friction=0.01
    )
    
    # Test _calculate_structural_targets directly
    print("\nTesting _calculate_structural_targets...")
    sl, tp, horizon, window = engine._calculate_structural_targets(50000.0, "LONG", metrics, data)
    print(f"Targets: SL={sl:.2f}, TP={tp:.2f}, Horizon={horizon}h, Window={window}m")
    
    # Test Strategy Methods
    print("\nTesting Strategy Methods...")
    
    # 3. Structural Advantage (LSI / Impact) - This one we modified
    print("- Testing Structural Advantage (Impact Asymmetry)...")
    try:
        sig = engine._analyze_structural_advantage("BTC", data, metrics)
        if sig:
            print(f"  ✅ OK. Signal: {sig.signal_type}")
            print(f"     Primary Metric: {sig.primary_metric}")
            print(f"     Invalidation: {sig.invalidation_threshold}")
            print(f"     Exhaustion: {sig.exhaustion_threshold}")
        else:
            print("  ⚠️ No Signal generated (check thresholds)")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    asyncio.run(verify_integration())
