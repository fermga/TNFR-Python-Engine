#!/usr/bin/env python3
"""
Example 154 — Conductor-Annotated QR Spectrum: Factorized Count and Scalar Wall
================================================================================

Examples 119 and 120 established the phase-sector fact: the canonical
structural diffusion operator on the directed quadratic-residue Cayley graph has
exactly three distinct complex eigenvalues precisely for odd primes, while the
per-node substrate remains blind by vertex-transitivity. Example 153
(07_number_theory) measures the structural-frequency RANK and the cyclotomy law.

This example adds the arithmetic refinement needed after the scalar/global
investigation: the clean multiplicative object is not the unannotated scalar
count alone.  It is the conductor-annotated distinct-value count

    #{(F_m(k), gcd(k,m)) : k mod m},  with m odd,

where F_m(k) is the Fourier sum of the nonzero quadratic residues modulo m.
For m = Product_i p_i^e_i this count factors exactly as

    Product_i (e_i + ceiling(e_i/2) + 1).

This is the multiplicative A(m) of the engine
(``tnfr.mathematics.number_theory.quadratic_residue_annotated_rank``); the
example verifies the spectral conductor-annotated count against that closed form
and exhibits the scalar wall where the unannotated projection collides.

Doctrine compliance
-------------------
This example CONSUMES the canonical residue-network API
(``quadratic_residue_set``, ``residue_network_rank``,
``quadratic_residue_annotated_rank``) and the canonical
``structural_diffusion_operator`` (through ``residue_network_rank``). The exact
CRT counterexample arithmetic (local Gauss-sum values in the multiquadratic
field) is example-specific and mirrors
``publish/quadratic-residue-digraph-spectrum/proof_note.md``.

Measured results
----------------
R1 PHASE PRIME SIGNATURE.  For odd m in [5,119], the FFT spectrum and the
   canonical operator (``residue_network_rank``) have the same distinct-count
   signature on sampled moduli, and "3 distinct scalar values" detects odd
   primes on the full sweep.

R2 CONDUCTOR-ANNOTATED PRODUCT.  For odd m in [3,119], the conductor-annotated
   count equals ``quadratic_residue_annotated_rank(m)`` =
   Product_i (e_i + ceiling(e_i/2) + 1).  Prime powers give the local ladder
   3, 4, 6, 7, 9, 10, 12, ...

R3 SCALAR WALL.  The unannotated scalar product rule is false globally:
   m = 3^7 * 5^2 * 41^2 has product count 192 but exact scalar count 191.
   The conductor/gcd annotation separates the colliding states and restores
   the exact product count 192.

Honest scope
------------
This is a compact bridge between the TNFR phase-sector examples and the OEIS
arithmetic sequence.  It does not introduce a faster factorization algorithm,
does not derive the primes, does not close the Riemann obstruction, and does
not claim the scalar count is multiplicative.  The result is a precise
structural classification: the phase spectrum detects primality locally, the
conductor-annotated count factors globally, and the scalar projection can lose
information through CRT collisions.

References
----------
- examples/08_emergent_geometry/119_phase_sector_directed_residue.py
- examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py
- examples/07_number_theory/153_structural_frequency_rank_cyclotomy.py
- src/tnfr/mathematics/number_theory.py (quadratic_residue_annotated_rank)
- publish/quadratic-residue-digraph-spectrum/proof_note.md
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction
from itertools import product
from math import prod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
from sympy import factorint, isprime

from tnfr.mathematics.number_theory import (
    quadratic_residue_annotated_rank,
    quadratic_residue_set,
    residue_network_rank,
)


COUNTEREXAMPLE_FACTORS = [(3, 7), (5, 2), (41, 2)]


def normalized_fft_spectrum(modulus: int) -> np.ndarray:
    """Return the row-normalized diffusion spectrum of the QR circulant."""
    residues = quadratic_residue_set(modulus)
    first_row = np.zeros(modulus, dtype=complex)
    for residue in residues:
        first_row[residue] = 1.0
    return 1.0 - (np.fft.fft(first_row) / len(residues))


def complex_key(value: complex, decimals: int = 8) -> tuple[float, float]:
    """Rounded key for stable distinct-value counting."""
    return (round(float(value.real), decimals), round(float(value.imag), decimals))


def scalar_spectrum_count(modulus: int) -> int:
    """Count distinct unannotated scalar spectrum values."""
    return len({complex_key(value) for value in normalized_fft_spectrum(modulus)})


def conductor_annotated_count(modulus: int) -> int:
    """Count distinct pairs (spectrum value, gcd(character, modulus))."""
    return len(
        {
            (complex_key(value), math.gcd(character, modulus))
            for character, value in enumerate(normalized_fft_spectrum(modulus))
        }
    )


def local_values(prime: int, exponent: int) -> list[tuple[str, tuple[Fraction, Fraction]]]:
    """Return exact local values as a + b*sqrt(epsilon*p)."""
    active_layers = [
        layer for layer in range(1, exponent + 1) if layer % 2 == exponent % 2
    ]
    values: list[tuple[str, tuple[Fraction, Fraction]]] = []

    for valuation_depth in range(exponent):
        rational_base = Fraction(1, 1) + sum(
            Fraction((prime - 1) * prime ** (layer - 1), 2)
            for layer in active_layers
            if layer <= valuation_depth
        )
        if valuation_depth + 1 in active_layers:
            rational_part = rational_base - Fraction(prime**valuation_depth, 2)
            sqrt_part = Fraction(prime**valuation_depth, 2)
            values.append((f"b{valuation_depth}+", (rational_part, sqrt_part)))
            values.append((f"b{valuation_depth}-", (rational_part, -sqrt_part)))
        else:
            values.append((f"b{valuation_depth}", (rational_base, Fraction(0, 1))))

    zero_character_value = Fraction(1, 1) + sum(
        Fraction((prime - 1) * prime ** (layer - 1), 2) for layer in active_layers
    )
    values.append(("zero", (zero_character_value, Fraction(0, 1))))

    distinct_values: list[tuple[str, tuple[Fraction, Fraction]]] = []
    seen = set()
    for label, value in values:
        if value not in seen:
            seen.add(value)
            distinct_values.append((label, value))
    return distinct_values


def local_value_items(
    prime: int, exponent: int
) -> list[tuple[str, tuple[Fraction, Fraction], int]]:
    """Return exact local values with their p-adic conductor depth."""
    active_layers = [
        layer for layer in range(1, exponent + 1) if layer % 2 == exponent % 2
    ]
    items: list[tuple[str, tuple[Fraction, Fraction], int]] = []

    for valuation_depth in range(exponent):
        rational_base = Fraction(1, 1) + sum(
            Fraction((prime - 1) * prime ** (layer - 1), 2)
            for layer in active_layers
            if layer <= valuation_depth
        )
        if valuation_depth + 1 in active_layers:
            rational_part = rational_base - Fraction(prime**valuation_depth, 2)
            sqrt_part = Fraction(prime**valuation_depth, 2)
            items.append((f"b{valuation_depth}+", (rational_part, sqrt_part), valuation_depth))
            items.append((f"b{valuation_depth}-", (rational_part, -sqrt_part), valuation_depth))
        else:
            items.append((f"b{valuation_depth}", (rational_base, Fraction(0, 1)), valuation_depth))

    zero_character_value = Fraction(1, 1) + sum(
        Fraction((prime - 1) * prime ** (layer - 1), 2) for layer in active_layers
    )
    items.append(("zero", (zero_character_value, Fraction(0, 1)), exponent))
    return items


def multiply_by_local_value(
    element: dict[int, Fraction],
    local_value: tuple[Fraction, Fraction],
    factor_index: int,
    radicand: int,
) -> dict[int, Fraction]:
    """Multiply a multiquadratic element by a local quadratic-field value."""
    rational_part, sqrt_part = local_value
    factor_bit = 1 << factor_index
    result: dict[int, Fraction] = {}

    for basis_mask, coefficient in element.items():
        if rational_part:
            result[basis_mask] = result.get(basis_mask, Fraction(0, 1)) + (
                coefficient * rational_part
            )
        if sqrt_part:
            if basis_mask & factor_bit:
                next_mask = basis_mask ^ factor_bit
                next_coefficient = coefficient * sqrt_part * radicand
            else:
                next_mask = basis_mask | factor_bit
                next_coefficient = coefficient * sqrt_part
            result[next_mask] = result.get(next_mask, Fraction(0, 1)) + next_coefficient

    return {
        basis_mask: coefficient
        for basis_mask, coefficient in result.items()
        if coefficient
    }


def exact_scalar_product_count(
    factors: list[tuple[int, int]],
) -> tuple[int, list[tuple[tuple[tuple[int, Fraction], ...], list[tuple], list[tuple]]]]:
    """Count distinct global scalar products exactly."""
    radicands = [prime if prime % 4 == 1 else -prime for prime, _ in factors]
    local_value_sets = [local_values(prime, exponent) for prime, exponent in factors]
    seen: dict[tuple[tuple[int, Fraction], ...], list[tuple]] = {}
    collisions = []

    for local_choice in product(*local_value_sets):
        element = {0: Fraction(1, 1)}
        diagnostic_label = []
        for factor_index, ((label, local_value), (prime, exponent), radicand) in enumerate(
            zip(local_choice, factors, radicands)
        ):
            element = multiply_by_local_value(
                element, local_value, factor_index, radicand
            )
            diagnostic_label.append((prime, exponent, label, local_value))

        product_key = tuple(sorted(element.items()))
        if product_key in seen and seen[product_key] != diagnostic_label:
            collisions.append((product_key, seen[product_key], diagnostic_label))
        else:
            seen[product_key] = diagnostic_label

    return len(seen), collisions


def exact_conductor_annotated_product_count(factors: list[tuple[int, int]]) -> int:
    """Count exact products after retaining the global conductor depth."""
    radicands = [prime if prime % 4 == 1 else -prime for prime, _ in factors]
    local_item_sets = [
        local_value_items(prime, exponent) for prime, exponent in factors
    ]
    seen = set()

    for local_choice in product(*local_item_sets):
        element = {0: Fraction(1, 1)}
        conductor_depth = 1
        for factor_index, (
            (_label, local_value, valuation_depth),
            (prime, _exponent),
            radicand,
        ) in enumerate(zip(local_choice, factors, radicands)):
            element = multiply_by_local_value(
                element, local_value, factor_index, radicand
            )
            conductor_depth *= prime**valuation_depth
        seen.add((tuple(sorted(element.items())), conductor_depth))

    return len(seen)


def experiment_phase_prime_signature() -> None:
    """Check the phase-sector prime signature and canonical agreement."""
    print("=" * 78)
    print("EXPERIMENT 1: Phase-sector prime signature")
    print("=" * 78)

    prime_signature_mismatches = [
        modulus
        for modulus in range(5, 120, 2)
        if (scalar_spectrum_count(modulus) == 3) != isprime(modulus)
    ]
    canonical_samples = [7, 9, 11, 15, 25, 29, 49]
    canonical_mismatches = [
        modulus
        for modulus in canonical_samples
        if residue_network_rank(modulus, "quadratic") != scalar_spectrum_count(modulus)
    ]

    print(f"  odd m in [5,119]: scalar_count(m)=3 iff prime: "
          f"{58 - len(prime_signature_mismatches)}/58")
    print("  canonical operator sample check:")
    print(f"    {'m':>4} {'factorization':>16} {'FFT count':>10} {'TNFR count':>10}")
    for modulus in canonical_samples:
        print(
            f"    {modulus:>4} {str(dict(factorint(modulus))):>16} "
            f"{scalar_spectrum_count(modulus):>10} "
            f"{residue_network_rank(modulus, 'quadratic'):>10}"
        )
    print()

    if prime_signature_mismatches or canonical_mismatches:
        raise AssertionError(
            "Unexpected phase-signature or canonical-operator mismatch"
        )


def experiment_annotated_product() -> None:
    """Check the conductor-annotated product formula on a small sweep."""
    print("=" * 78)
    print("EXPERIMENT 2: Conductor-annotated count factors exactly")
    print("=" * 78)

    annotated_mismatches = [
        modulus
        for modulus in range(3, 120, 2)
        if conductor_annotated_count(modulus)
        != quadratic_residue_annotated_rank(modulus)
    ]
    print(f"  odd m in [3,119]: annotated_count(m)=A(m): "
          f"{59 - len(annotated_mismatches)}/59")
    print("  local ladder for m=3^e:")
    print(f"    {'e':>2} {'m':>5} {'scalar':>8} {'annotated':>10} {'A(m)':>8}")
    for exponent in range(1, 8):
        modulus = 3**exponent
        print(
            f"    {exponent:>2} {modulus:>5} {scalar_spectrum_count(modulus):>8} "
            f"{conductor_annotated_count(modulus):>10} "
            f"{quadratic_residue_annotated_rank(modulus):>8}"
        )
    print()

    if annotated_mismatches:
        raise AssertionError("Unexpected conductor-annotated product mismatch")


def experiment_scalar_wall() -> None:
    """Show the first known scalar/product collision and annotated repair."""
    print("=" * 78)
    print("EXPERIMENT 3: Scalar projection has CRT collisions")
    print("=" * 78)

    modulus = prod(prime**exponent for prime, exponent in COUNTEREXAMPLE_FACTORS)
    predicted = quadratic_residue_annotated_rank(modulus)
    exact_scalar, collisions = exact_scalar_product_count(COUNTEREXAMPLE_FACTORS)
    exact_annotated = exact_conductor_annotated_product_count(COUNTEREXAMPLE_FACTORS)

    print(f"  m = {modulus} = 3^7 * 5^2 * 41^2")
    print(f"  product formula count A(m):     {predicted}")
    print(f"  exact unannotated scalar count: {exact_scalar}")
    print(f"  exact conductor-annotated count:{exact_annotated}")
    print("  collision witness:")
    print("    11 * 1 * 821 = 821 * 11 * 1 = 9031")
    print("    the scalar value collides, but the gcd/conductor depths differ")
    print()

    if predicted != 192 or exact_scalar != 191 or exact_annotated != 192 or not collisions:
        raise AssertionError("Unexpected scalar-wall counterexample result")


def main() -> None:
    print()
    print("TNFR Example 154: Conductor-annotated QR spectrum")
    print("Phase prime signature, exact product count, and scalar CRT wall")
    print()
    experiment_phase_prime_signature()
    experiment_annotated_product()
    experiment_scalar_wall()
    print("=" * 78)
    print("STRUCTURAL READING")
    print("=" * 78)
    print("  The phase spectrum detects odd primes through the directed QR graph.")
    print("  The conductor annotation retains the local support depth and factors.")
    print("  The scalar projection can alias distinct CRT-local states, so it is")
    print("  the wrong object for a global product theorem.")
    print()


if __name__ == "__main__":
    main()
