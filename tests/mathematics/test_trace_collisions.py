r"""Tests for trace collisions — loss of observability under the trace (R5, N10).

Exact fiber counts, the character formula (Fourier inversion on F_p), and Galois
invariance of the trace observable. Prime fields reproduce R2 (identity trace, no
collisions); extensions collapse H onto fewer residues.
"""

from __future__ import annotations

from math import gcd

from tnfr.mathematics.finite_fields import FiniteField
from tnfr.mathematics import trace_collisions as tc
from tnfr.mathematics.trace_collisions import (
    certify_trace_collisions,
    character_formula_residual,
    observed_trace_values,
    power_image,
    trace_fiber_counts,
    trace_fiber_counts_via_characters,
    trace_is_galois_invariant,
)


# --------------------------------------------------------------------------- #
# Power-image structure
# --------------------------------------------------------------------------- #
def test_power_image_size_is_index_formula():
    for p, f in [(5, 1), (7, 1), (2, 3), (3, 2), (5, 2)]:
        field = FiniteField(p, f)
        for k in (2, 3, 4):
            h = power_image(field, k)
            assert len(h) == (field.q - 1) // gcd(k, field.q - 1)


def test_fiber_counts_sum_to_subset_size():
    field = FiniteField(5, 2)
    h = power_image(field, 2)
    counts = trace_fiber_counts(field, h)
    assert sum(counts.values()) == len(h)
    assert set(counts) == set(range(field.p))


# --------------------------------------------------------------------------- #
# Character formula = exact count (Fourier inversion)
# --------------------------------------------------------------------------- #
def test_character_formula_matches_exact_counts():
    for p, f in [(5, 1), (7, 1), (2, 3), (3, 2), (5, 2)]:
        field = FiniteField(p, f)
        for k in (2, 3):
            h = power_image(field, k)
            assert trace_fiber_counts_via_characters(field, h) == \
                trace_fiber_counts(field, h)
            assert character_formula_residual(field, h) < 1e-9


# --------------------------------------------------------------------------- #
# Galois invariance — the histogram is representation-free
# --------------------------------------------------------------------------- #
def test_trace_is_galois_invariant():
    for p, f in [(5, 1), (2, 3), (3, 2), (5, 2)]:
        field = FiniteField(p, f)
        for k in (2, 3, 4):
            assert trace_is_galois_invariant(field, power_image(field, k))


# --------------------------------------------------------------------------- #
# Prime field vs extension: R2 recovery vs collisions
# --------------------------------------------------------------------------- #
def test_prime_field_has_no_trace_collisions():
    # in F_p the trace is the identity: every h has its own residue
    field = FiniteField(7, 1)
    for k in (2, 3, 4):
        h = power_image(field, k)
        cert = certify_trace_collisions(field, k)
        assert not cert.has_collisions
        assert cert.observed_values == len(h)
        assert cert.max_fiber == 1


def test_extension_field_collapses_information():
    # F_2^3: the 7 nonzero cubes/squares collapse onto {0,1}
    field = FiniteField(2, 3)
    cert = certify_trace_collisions(field, 2)
    assert cert.subset_size == 7
    assert cert.observed_values == 2
    assert cert.has_collisions
    assert cert.max_fiber > 1


def test_incomplete_support_when_observed_less_than_p():
    field = FiniteField(5, 2)
    cert = certify_trace_collisions(field, 2)
    assert cert.observed_values < field.p     # some residue is never a square-trace
    assert not cert.full_support
    assert cert.has_collisions


# --------------------------------------------------------------------------- #
# Certificate + exports
# --------------------------------------------------------------------------- #
def test_certificate_fields_consistent():
    field = FiniteField(3, 2)
    cert = certify_trace_collisions(field, 3)
    assert cert.character_formula_residual < 1e-9
    assert cert.galois_invariant
    assert cert.min_fiber <= cert.max_fiber
    assert "NT-P05c" in cert.claim_status      # type detector NOT claimed


def test_observed_values_helper_matches_certificate():
    field = FiniteField(5, 2)
    h = power_image(field, 3)
    assert observed_trace_values(field, h) == \
        certify_trace_collisions(field, 3).observed_values


def test_module_exports_complete():
    expected = {
        "power_image", "trace_fiber_counts", "observed_trace_values",
        "character_sum", "trace_fiber_counts_via_characters",
        "character_formula_residual", "trace_is_galois_invariant",
        "uniform_deviation", "TraceCollisionCertificate",
        "certify_trace_collisions",
    }
    assert expected <= set(tc.__all__)
