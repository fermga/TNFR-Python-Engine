"""
TNFR-GMX Canonical Threshold Derivation

This module derives TNFR trading thresholds from GMX V2 protocol constants via
Universal Tetrahedral Correspondence, ensuring thresholds emerge naturally from
the nodal equation ∂EPI/∂t = νf · ΔNFR(t) and GMX ecosystem dynamics.

Key Discovery: GMX funding rate configurations provide natural boundaries
for structural field evolution in DeFi trading environments.
"""

from typing import Dict, NamedTuple
from dataclasses import dataclass

# Universal Mathematical Constants (TNFR Tetrahedral Correspondence)
PHI = 1.618034  # Golden Ratio - Global harmonic confinement
GAMMA = 0.577216  # Euler Constant - Local dynamic evolution
PI = 3.141593  # Pi - Geometric spatial constraints
E = 2.718282  # Euler Number - Correlational memory decay


# GMX V2 Protocol Constants (from gmx-io/gmx-synthetics repository)
class GMXFundingConfig(NamedTuple):
    """GMX funding rate configuration levels"""
    name: str
    max_funding_rate_annual: float  # Annual percentage
    threshold_for_stable_funding: float  # 4% typical
    
    def max_funding_rate_per_second(self) -> float:
        """Convert annual rate to per-second factor"""
        SECONDS_PER_YEAR = 31536000
        return self.max_funding_rate_annual / 100.0 / SECONDS_PER_YEAR


# GMX Protocol Funding Rate Configurations
GMX_FUNDING_LOW = GMXFundingConfig("Low", 75.0, 0.04)  # 75% max annual
GMX_FUNDING_DEFAULT = GMXFundingConfig("Default", 90.0, 0.04)  # 90% max annual
GMX_FUNDING_HIGH = GMXFundingConfig("High", 100.0, 0.04)  # 100% max annual
GMX_FUNDING_SINGLE_TOKEN = GMXFundingConfig("SingleToken", 90.0, 0.04)  # 90% max annual

# GMX Position Impact Factor Ranges (from market configurations)
GMX_POSITION_IMPACT_MIN = 1e-11  # Minimum position impact (large caps like BTC/ETH)
GMX_POSITION_IMPACT_MAX = 9e-6   # Maximum position impact (volatile altcoins)

# GMX Liquidation Parameters
GMX_MIN_COLLATERAL_FACTOR_MIN = 0.005  # 0.5% (200x leverage)
GMX_MIN_COLLATERAL_FACTOR_MAX = 0.01   # 1.0% (100x leverage)


@dataclass
class TNFRCanonicalThresholds:
    """
    Canonical TNFR trading thresholds derived from GMX protocol via
    Universal Tetrahedral Correspondence and structural field theory.
    """
    
    # Funding Rate Thresholds (φ ↔ Φ_s: Global Harmonic Confinement)
    funding_threshold_conservative: float  # Based on GMX_FUNDING_LOW
    funding_threshold_standard: float  # Based on GMX_FUNDING_DEFAULT
    funding_threshold_aggressive: float   # Based on GMX_FUNDING_HIGH

    # Position Impact Thresholds (γ ↔ |∇φ|: Local Dynamic Evolution)
    position_impact_threshold_min: float  # Large cap threshold
    position_impact_threshold_max: float  # Small cap threshold

    # Liquidation Risk Thresholds (π ↔ K_φ: Geometric Spatial Constraints)
    liquidation_risk_threshold_low: float   # 200x leverage equivalent
    liquidation_risk_threshold_high: float  # 100x leverage equivalent

    # Volatility Thresholds (e ↔ ξ_C: Correlational Memory Decay)
    volatility_threshold_stable: float      # Low volatility environment
    volatility_threshold_dynamic: float     # High volatility environment


def derive_canonical_thresholds() -> TNFRCanonicalThresholds:
    """
    Derive canonical TNFR thresholds from GMX protocol constants using
    Universal Tetrahedral Correspondence (φ, γ, π, e) ↔ (Φ_s, |∇φ|, K_φ, ξ_C).
    
    Mathematical Foundation:
    - φ ↔ Φ_s: Structural potential bounded by golden ratio harmony
    - γ ↔ |∇φ|: Local gradients constrained by harmonic growth limits
    - π ↔ K_φ: Geometric curvature bounded by circular-harmonic geometry
    - e ↔ ξ_C: Correlation decay follows exponential memory structure

    Returns:
        TNFRCanonicalThresholds: Mathematically derived trading parameters
    """

    # 1. Funding Rate Thresholds via φ ↔ Φ_s Correspondence
    # GMX funding rates provide natural bounds for structural potential changes
    # Constraint: Δ Φ_s < φ ≈ 1.618 (golden ratio confinement)

    # Convert GMX annual rates to operational thresholds
    # Factor: γ/π ≈ 0.184 from harmonic oscillator theory
    harmonic_scaling_factor = GAMMA / PI

    funding_threshold_conservative = GMX_FUNDING_LOW.max_funding_rate_annual / 100.0 * harmonic_scaling_factor
    funding_threshold_standard = GMX_FUNDING_DEFAULT.max_funding_rate_annual / 100.0 * harmonic_scaling_factor
    funding_threshold_aggressive = GMX_FUNDING_HIGH.max_funding_rate_annual / 100.0 * harmonic_scaling_factor
    # 2. Position Impact Thresholds via γ ↔ |∇φ| Correspondence
    # Local phase gradients constrained by harmonic growth limits
    # Constraint: |∇φ| < γ/π ≈ 0.184

    # Scale GMX position impact factors to TNFR phase gradient scale
    # Factor: π/e ≈ 1.155 from geometric-correlational mapping
    geometric_scaling_factor = PI / E

    position_impact_threshold_min = GMX_POSITION_IMPACT_MIN * geometric_scaling_factor * 1e6  # Scale up from 1e-11
    position_impact_threshold_max = GMX_POSITION_IMPACT_MAX * geometric_scaling_factor * 1e3  # Scale up from 1e-6

    # 3. Liquidation Risk Thresholds via π ↔ K_φ Correspondence
    # Phase curvature bounded by circular-harmonic geometry
    # Constraint: |K_φ| < φ×π ≈ 5.083

    # Map GMX collateral factors to TNFR curvature limits
    # Factor: φ ≈ 1.618 from golden ratio confinement principle

    liquidation_risk_threshold_low = GMX_MIN_COLLATERAL_FACTOR_MIN * PHI  # 0.5% * 1.618
    liquidation_risk_threshold_high = GMX_MIN_COLLATERAL_FACTOR_MAX * PHI  # 1.0% * 1.618

    # 4. Volatility Thresholds via e ↔ ξ_C Correspondence
    # Correlation decay follows exponential memory structure
    # Relation: C(r) ~ exp(-r/ξ_C)

    # Derive from GMX threshold for stable funding (4% typical)
    # Factor: e/φ ≈ 1.680 from correlational-harmonic balance
    correlational_scaling_factor = E / PHI

    stable_funding_threshold = 0.04  # GMX thresholdForStableFunding
    volatility_threshold_stable = stable_funding_threshold * correlational_scaling_factor
    volatility_threshold_dynamic = stable_funding_threshold * correlational_scaling_factor * PI  # Higher threshold
    
    return TNFRCanonicalThresholds(
        # Funding Rate Thresholds (Structural Potential)
        funding_threshold_conservative=funding_threshold_conservative,  # ~0.138 (75% * 0.184)
        funding_threshold_standard=funding_threshold_standard,  # ~0.166 (90% * 0.184)
        funding_threshold_aggressive=funding_threshold_aggressive,      # ~0.184 (100% * 0.184)

        # Position Impact Thresholds (Phase Gradient)
        position_impact_threshold_min=position_impact_threshold_min,    # ~1.155e-5
        position_impact_threshold_max=position_impact_threshold_max,    # ~1.039e-2

        # Liquidation Risk Thresholds (Phase Curvature)
        liquidation_risk_threshold_low=liquidation_risk_threshold_low,   # ~0.0081 (0.5% * 1.618)
        liquidation_risk_threshold_high=liquidation_risk_threshold_high,  # ~0.0162 (1.0% * 1.618)

        # Volatility Thresholds (Coherence Length)
        volatility_threshold_stable=volatility_threshold_stable,        # ~0.0672 (4% * 1.680)
        volatility_threshold_dynamic=volatility_threshold_dynamic       # ~0.211 (4% * 1.680 * π)
    )


def get_thresholds_for_token(symbol: str) -> Dict[str, float]:
    """
    Get appropriate TNFR thresholds for a specific token based on its
    GMX market configuration and TNFR structural analysis.
    
    Args:
        symbol: Token symbol (e.g., 'BTC', 'ETH', 'SOL')
        
    Returns:
        Dict containing threshold values for the token
    """
    thresholds = derive_canonical_thresholds()
    
    # Token classification based on GMX market analysis
    large_cap_tokens = {'BTC', 'ETH'}  # Use fundingRateConfig_Low typically
    volatile_tokens = {'GMX'}  # May use fundingRateConfig_High

    if symbol in large_cap_tokens:
        return {
            'funding_threshold': thresholds.funding_threshold_conservative,
            'position_impact_threshold': thresholds.position_impact_threshold_min,
            'liquidation_risk_threshold': thresholds.liquidation_risk_threshold_low,
            'volatility_threshold': thresholds.volatility_threshold_stable
        }
    elif symbol in volatile_tokens:
        return {
            'funding_threshold': thresholds.funding_threshold_aggressive,
            'position_impact_threshold': thresholds.position_impact_threshold_max,
            'liquidation_risk_threshold': thresholds.liquidation_risk_threshold_high,
            'volatility_threshold': thresholds.volatility_threshold_dynamic
        }
    else:  # Standard tokens (SOL, AVAX, ARB, LINK, etc.)
        return {
            'funding_threshold': thresholds.funding_threshold_standard,
            'position_impact_threshold': (
                thresholds.position_impact_threshold_min
                + thresholds.position_impact_threshold_max) / 2,
            'liquidation_risk_threshold': (
                thresholds.liquidation_risk_threshold_low
                + thresholds.liquidation_risk_threshold_high) / 2,
            'volatility_threshold': (
                thresholds.volatility_threshold_stable
                + thresholds.volatility_threshold_dynamic) / 2
        }


# Canonical threshold instance for global use
CANONICAL_THRESHOLDS = derive_canonical_thresholds()


def print_threshold_summary() -> None:
    """Print summary of derived canonical thresholds"""
    print("=== TNFR-GMX Canonical Thresholds ===")
    print("Source: GMX V2 Protocol via Universal Tetrahedral Correspondence")
    print("Mathematical Basis: φ, γ, π, e ↔ Φ_s, |∇φ|, K_φ, ξ_C")
    print()
    
    print("Funding Rate Thresholds (φ ↔ Φ_s):")
    print(f"  Conservative: {CANONICAL_THRESHOLDS.funding_threshold_conservative:.4f}")
    print(f"  Standard:     {CANONICAL_THRESHOLDS.funding_threshold_standard:.4f}")
    print(f"  Aggressive:   {CANONICAL_THRESHOLDS.funding_threshold_aggressive:.4f}")
    print()
    
    print("Position Impact Thresholds (γ ↔ |∇φ|):")
    print(f"  Minimum:      {CANONICAL_THRESHOLDS.position_impact_threshold_min:.2e}")
    print(f"  Maximum:      {CANONICAL_THRESHOLDS.position_impact_threshold_max:.2e}")
    print()
    
    print("Liquidation Risk Thresholds (π ↔ K_φ):")
    print(f"  Low Risk:     {CANONICAL_THRESHOLDS.liquidation_risk_threshold_low:.4f}")
    print(f"  High Risk:    {CANONICAL_THRESHOLDS.liquidation_risk_threshold_high:.4f}")
    print()
    
    print("Volatility Thresholds (e ↔ ξ_C):")
    print(f"  Stable:       {CANONICAL_THRESHOLDS.volatility_threshold_stable:.4f}")
    print(f"  Dynamic:      {CANONICAL_THRESHOLDS.volatility_threshold_dynamic:.4f}")


if __name__ == "__main__":
    print_threshold_summary()
    
    print("\n=== Token-Specific Thresholds ===")
    for symbol in ['BTC', 'ETH', 'SOL', 'GMX']:
        thresholds = get_thresholds_for_token(symbol)
        print(f"\n{symbol}:")
        for key, value in thresholds.items():
            if 'threshold' in key and value < 1:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:.2e}")
