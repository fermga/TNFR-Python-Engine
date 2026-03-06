"""
Canonical Constants for TNFR Primality Testing

All constants derived from the four fundamental mathematical constants
(φ, γ, π, e) via the Universal Tetrahedral Correspondence. Zero empirical
fitting — every value traces back to first-principles derivation.

Source: TNFR-Python-Engine canonical.py (CanonicalArithmeticParameters)

Mathematical Foundation:
    The ΔNFR arithmetic pressure equation uses three coefficients:
        ΔNFR(n) = ζ·(Ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))

    Where ζ, η, θ are derived from (φ, γ, π, e):
        ζ = φ×γ  (zeta coupling strength)
        η = (γ/φ)×π  (phase coupling × geometric factor)
        θ = 1/φ  (coherence scaling)

    Ω(n) = prime factor count WITH multiplicity (big Omega)
    τ(n) = divisor count
    σ(n) = divisor sum

Changelog:
    v1.0: Original empirical constants (ζ=1.0, η=0.8, θ=0.6)
    v2.0: Canonical derivation from (φ, γ, π, e) — zero fitting
"""
from __future__ import annotations

import math

# --- Fundamental mathematical constants ---
PHI = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618033988749895
GAMMA = 0.5772156649015329             # γ (Euler-Mascheroni constant)
PI = math.pi                           # π ≈ 3.141592653589793
E = math.e                             # e ≈ 2.718281828459045

# --- Derived TNFR canonical coefficients ---
# Factorization pressure coefficient: ζ = φ × γ
ZETA_CANONICAL = PHI * GAMMA                  # ≈ 0.9340

# Divisor pressure coefficient: η = (γ/φ) × π
ETA_CANONICAL = (GAMMA / PHI) * PI            # ≈ 1.1207

# Abundance pressure coefficient: θ = 1/φ
THETA_CANONICAL = 1.0 / PHI                   # ≈ 0.6180

# --- Detection thresholds ---
# Structural significance threshold: γ/(e×π)
# From canonical theory: structural pressure resolution limit
DELTA_NFR_THRESHOLD = GAMMA / (E * PI)        # ≈ 0.0676

# Zero-detection tolerance for primality testing.
# ΔNFR(p) = 0 exactly for primes; this is the numerical tolerance
# for floating-point comparisons (not a physics threshold).
PRIMALITY_TOLERANCE = 1e-10

# --- Structural triad constants ---
# EPI parameters (form characterization)
ALPHA_EPI = 1.0 / PHI                         # α = 1/φ ≈ 0.618
BETA_EPI = GAMMA / (PI + GAMMA)               # β = γ/(π+γ) ≈ 0.155
GAMMA_EPI = GAMMA / PI                         # γ_epi = γ/π ≈ 0.1837

# Frequency parameters (νf characterization)
NU_0 = (PHI / GAMMA) / PI                     # ν₀ = (φ/γ)/π ≈ 0.8925
DELTA_FREQ = GAMMA / (PHI * PI)               # δ = γ/(φ×π) ≈ 0.1137
EPSILON_FREQ = math.exp(-PI)                   # ε = e^(-π) ≈ 0.0432

# --- Tetrad field thresholds (for arithmetic networks) ---
PHI_S_THRESHOLD = 0.7711                       # von Koch fractal bound
GRAD_PHI_THRESHOLD = GAMMA / PI                # γ/π ≈ 0.1837
K_PHI_THRESHOLD = 0.9 * PI                     # 0.9π ≈ 2.8274

# --- Legacy constants (backward compatibility) ---
ZETA_LEGACY = 1.0
ETA_LEGACY = 0.8
THETA_LEGACY = 0.6
