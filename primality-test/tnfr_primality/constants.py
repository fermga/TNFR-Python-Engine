"""
Constants for TNFR Primality Testing

The arithmetic structural triad uses canonical UNIT coefficients. Per AGENTS.md
§3 only π is a genuine structural scale; φ, γ and e are not. By the Coefficient
Independence theorem (TNFR_NUMBER_THEORY.md §4.2) the primality criterion
ΔNFR(n) = 0 holds for any positive coefficients, so the canonical choice is
unity — the structural content lives entirely in the arithmetic invariants
(Ω, τ, σ, n), with no fitted/numerological scale.

Source: TNFR-Python-Engine number_theory.py (ArithmeticTNFRParameters)

Mathematical Foundation:
    The ΔNFR arithmetic pressure equation uses three coefficients:
        ΔNFR(n) = ζ·(Ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))

    Canonically ζ = η = θ = 1 (unit weights, §4.2 coefficient independence).

    Ω(n) = prime factor count WITH multiplicity (big Omega)
    τ(n) = divisor count
    σ(n) = divisor sum

Changelog:
    v1.0: Original empirical constants (ζ=1.0, η=0.8, θ=0.6)
    v2.0: Rewritten as (φ, γ, π, e) combinations approximating those values
          (notational; audit 2026: NOT a derivation — values barely moved).
    v3.0: Canonicalized to UNIT coefficients (ζ=η=θ=1, EPI/νf weights to unity).
          Removes the φ/γ/e overlay; only π remains an admissible structural
          scale. Primality preserved exactly (§4.2 theorem).
"""

from __future__ import annotations

import math

# --- Fundamental mathematical constants ---
PHI = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895
GAMMA = 0.5772156649015329  # γ (Euler-Mascheroni constant)
PI = math.pi  # π ≈ 3.141592653589793
E = math.e  # e ≈ 2.718281828459045

# --- Canonical ΔNFR pressure coefficients (unit weights, §4.2) ---
# Only π is a genuine structural scale (AGENTS.md §3); by the coefficient-
# independence theorem the primality criterion ΔNFR=0 is invariant to any
# positive rescaling, so the canonical weights are unity.
ZETA_CANONICAL = 1.0  # factorization pressure (Ω − 1)
ETA_CANONICAL = 1.0  # divisor pressure (τ − 2)
THETA_CANONICAL = 1.0  # abundance pressure (σ/n − (1 + 1/n))

# --- Detection thresholds ---
# Structural significance threshold: γ/(e×π)
# From canonical theory: structural pressure resolution limit
DELTA_NFR_THRESHOLD = GAMMA / (E * PI)  # ≈ 0.0676

# Zero-detection tolerance for primality testing.
# ΔNFR(p) = 0 exactly for primes; this is the numerical tolerance
# for floating-point comparisons (not a physics threshold).
PRIMALITY_TOLERANCE = 1e-10

# --- Structural triad constants (canonical unit weights) ---
# EPI parameters: EPI = 1 + α·Ω + β·ln τ + γ·(σ/n − 1)
ALPHA_EPI = 1.0  # factorization-complexity weight (Ω)
BETA_EPI = 1.0  # divisor-complexity weight (ln τ)
GAMMA_EPI = 1.0  # abundance-deviation weight (σ/n − 1)

# Frequency parameters: νf = ν₀·(1 + δ·τ/n + ε·Ω/ln n)
NU_0 = 1.0  # base structural frequency
DELTA_FREQ = 1.0  # divisor-density modulation (τ/n)
EPSILON_FREQ = 1.0  # factorization modulation (Ω/ln n)

# --- Tetrad field thresholds (for arithmetic networks) ---
PHI_S_THRESHOLD = 0.7711  # von Koch fractal bound
GRAD_PHI_THRESHOLD = GAMMA / PI  # γ/π ≈ 0.1837
K_PHI_THRESHOLD = 0.9 * PI  # 0.9π ≈ 2.8274

# --- Legacy constants (backward compatibility) ---
ZETA_LEGACY = 1.0
ETA_LEGACY = 0.8
THETA_LEGACY = 0.6
