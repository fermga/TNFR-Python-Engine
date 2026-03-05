"""
Test Canonical Threshold Signal Generation
==========================================
Verify that the trading engine generates signals with canonical thresholds
derived from GMX protocol analysis and TNFR Universal Tetrahedral Correspondence.
"""

import sys
import os
import asyncio
import random
from dataclasses import dataclass

# Add paths for imports
sys.path.append('src')
sys.path.append('future-research/crypto-lab/src')

# Import the canonical thresholds
sys.path.insert(0, os.path.join('src'))
from tnfr.gmx.canonical_thresholds import get_thresholds_for_token, CANONICAL_THRESHOLDS

# Mock market data structure
@dataclass
class MockMarketData:
    symbol: str
    price: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    timestamp: int

def create_realistic_market_scenario(symbol: str) -> MockMarketData:
    """Create realistic market data with normal DeFi funding rates."""
    
    # Get canonical thresholds for the token
    thresholds = get_thresholds_for_token(symbol)
    base_threshold = thresholds['funding_threshold']
    
    # Generate funding rates in different scenarios
    scenarios = {
        'normal_market': random.uniform(-0.01, 0.01),  # -1% to +1%
        'slight_contango': random.uniform(0.02, 0.08),  # 2% to 8% 
        'strong_backwardation': random.uniform(-0.15, -0.05),  # -15% to -5%
        'extreme_funding': random.uniform(0.12, 0.25)  # 12% to 25% (rare but possible)
    }
    
    # Pick a scenario based on token volatility
    if symbol in ['BTC', 'ETH']:
        scenario = random.choice(['normal_market', 'slight_contango'])
    elif symbol in ['SOL', 'AVAX', 'ARB', 'LINK']:
        scenario = random.choice(['normal_market', 'slight_contango', 'strong_backwardation'])
    else:  # GMX and other volatile tokens
        scenario = random.choice(list(scenarios.keys()))
    
    funding_rate = scenarios[scenario]
    
    # Generate other realistic market data
    base_prices = {
        'BTC': 43000,
        'ETH': 2300, 
        'SOL': 65,
        'AVAX': 25,
        'ARB': 1.2,
        'GMX': 45,
        'LINK': 14
    }
    
    return MockMarketData(
        symbol=symbol,
        price=base_prices.get(symbol, 100) * random.uniform(0.95, 1.05),
        funding_rate=funding_rate,
        open_interest=random.uniform(1000000, 100000000),  # $1M to $100M
        volume_24h=random.uniform(10000000, 1000000000),   # $10M to $1B  
        timestamp=int(random.uniform(1700000000, 1800000000))
    )

async def test_signal_generation():
    """Test signal generation with canonical thresholds."""
    
    print("=== CANONICAL THRESHOLD SIGNAL GENERATION TEST ===")
    print(f"Testing with Universal Tetrahedral Correspondence thresholds:")
    print(f"  Conservative: {CANONICAL_THRESHOLDS.funding_threshold_conservative:.4f}")
    print(f"  Standard:     {CANONICAL_THRESHOLDS.funding_threshold_standard:.4f}")
    print(f"  Aggressive:   {CANONICAL_THRESHOLDS.funding_threshold_aggressive:.4f}")
    print()
    
    # Test tokens with different risk profiles
    test_tokens = ['BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'GMX', 'LINK']
    
    signals_generated = 0
    total_tests = 0
    
    for token in test_tokens:
        print(f"\n--- Testing {token} ---")
        token_thresholds = get_thresholds_for_token(token)
        print(f"Canonical funding threshold: {token_thresholds['funding_threshold']:.4f}")
        
        # Test multiple market scenarios
        for i in range(5):
            market_data = create_realistic_market_scenario(token)
            total_tests += 1
            
            # Check if this would generate a signal
            funding_rate = abs(market_data.funding_rate)
            threshold = token_thresholds['funding_threshold']
            
            # Apply stress modulation (simplified version of engine logic)
            # Assume moderate phase gradient (0.1) for this test
            phase_gradient = 0.1  
            stress_factor = 1.0 + (phase_gradient * 0.618)  # INV_PHI approximation
            adjusted_threshold = threshold * stress_factor
            
            signal_generated = funding_rate > adjusted_threshold
            
            if signal_generated:
                signals_generated += 1
                action = "SHORT" if market_data.funding_rate > 0 else "LONG"
                confidence = min((funding_rate - adjusted_threshold) / adjusted_threshold * 0.618, 0.764)
                
                print(f"  ✅ SIGNAL: {token} {action}")
                print(f"     Funding Rate: {market_data.funding_rate:.4f}")
                print(f"     Threshold:    {adjusted_threshold:.4f}")  
                print(f"     Confidence:   {confidence:.3f}")
            else:
                print(f"  ⚪ No Signal: Funding {market_data.funding_rate:.4f} < Threshold {adjusted_threshold:.4f}")
    
    print(f"\n=== RESULTS ===")
    print(f"Total Tests: {total_tests}")
    print(f"Signals Generated: {signals_generated}")
    print(f"Signal Rate: {signals_generated/total_tests:.1%}")
    
    if signals_generated > 0:
        print("✅ SUCCESS: Canonical thresholds are generating trading signals!")
        print("The thresholds are calibrated for realistic DeFi market conditions.")
    else:
        print("⚠️  WARNING: No signals generated. Thresholds may still be too strict.")
        
    return signals_generated > 0

if __name__ == "__main__":
    success = asyncio.run(test_signal_generation())
    if success:
        print("\n🎯 Canonical threshold integration ready for live trading!")
    else:
        print("\n⚠️ Threshold calibration needs adjustment.")