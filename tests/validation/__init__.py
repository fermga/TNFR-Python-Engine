"""Quantitative validation test suite for TNFR n-body systems.

This module implements the validation experiments described in:
docs/source/theory/09_classical_mechanics_numerical_validation.md

Test suites cover:
- Mass-frequency scaling validation (m = 1/νf)
- Conservation laws (energy, momentum, angular momentum)
- Known analytical solutions (Kepler orbits, Lagrange points)
- Chaos detection (Lyapunov exponents, Poincaré sections)
- Coherence metrics (C(t), Si) for structural multitudes

All tests follow TNFR canonical invariants and produce reproducible results.
"""

__all__ = [
    "test_nbody_validation",
]
