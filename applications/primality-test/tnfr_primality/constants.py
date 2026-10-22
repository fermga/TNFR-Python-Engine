"""Compatibility defaults for static arithmetic-pressure diagnostics.

For n>=2 the exact-real zero set of
zeta*(Omega-1)+eta*(tau-2)+theta*(sigma/n-(1+1/n)) is the primes
for every positive coefficient triple. This leaves the coefficients
underdetermined; unit weights are the package's normalization choice.

EPI/capacity weights and diagnostic thresholds below are retained API
configuration, not consequences of the nodal equation. Pi is the exact
angle-wrap scale; it does not fix arithmetic pressure or coherence bands.
Legacy mathematical constants remain exported for compatibility."""

from __future__ import annotations

import math

# --- Fundamental mathematical constants ---
PHI = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895
GAMMA = 0.5772156649015329  # γ (Euler-Mascheroni constant)
PI = math.pi  # π ≈ 3.141592653589793
E = math.e  # e ≈ 2.718281828459045

# --- Compatibility pressure defaults ---
# Positive coefficients preserve the exact arithmetic zero set.
# Unit weights select a normalization; the theorem does not require them.
ZETA_CANONICAL = 1.0  # factorization pressure (Ω − 1)
ETA_CANONICAL = 1.0  # divisor pressure (τ − 2)
THETA_CANONICAL = 1.0  # abundance pressure (σ/n − (1 + 1/n))

# --- Detection thresholds ---
# Legacy configured significance threshold, not a derived resolution limit.
DELTA_NFR_THRESHOLD = GAMMA / (E * PI)  # ≈ 0.0676

# Zero-detection tolerance for primality testing.
# ΔNFR(p) = 0 exactly for primes; this is the numerical tolerance
# for floating-point comparisons (not a physics threshold).
PRIMALITY_TOLERANCE = 1e-10

# --- Static arithmetic EPI/capacity configuration (no phase channel) ---
# EPI parameters: EPI = 1 + α·Ω + β·ln τ + γ·(σ/n − 1)
ALPHA_EPI = 1.0  # factorization-complexity weight (Ω)
BETA_EPI = 1.0  # divisor-complexity weight (ln τ)
GAMMA_EPI = 1.0  # abundance-deviation weight (σ/n − 1)

# Frequency parameters: νf = ν₀·(1 + δ·τ/n + ε·Ω/ln n)
NU_0 = 1.0  # base structural frequency
DELTA_FREQ = 1.0  # divisor-density modulation (τ/n)
EPSILON_FREQ = 1.0  # factorization modulation (Ω/ln n)

# --- Legacy diagnostic policies (not universal tetrad field bounds) ---
PHI_S_THRESHOLD = 0.7711  # retained empirical policy
GRAD_PHI_THRESHOLD = GAMMA / PI  # γ/π ≈ 0.1837
K_PHI_THRESHOLD = 0.9 * PI  # 0.9π ≈ 2.8274

# --- Legacy constants (backward compatibility) ---
ZETA_LEGACY = 1.0
ETA_LEGACY = 0.8
THETA_LEGACY = 0.6
