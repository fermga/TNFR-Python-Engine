r"""Tests for the R5 Gaussian-integer residue networks.

The additive ``k``-th power Cayley network on ``Z[i]/(p)`` has a spectrum whose
distinct-eigenvalue count, at ``k = 2``, separates the classical decomposition
types — ramified ``2``, inert ``3``, split ``6`` — on the tested primes.  The
classical type is a ground-truth label only.  The separation is ``k``-sensitive
(``NT-P05`` stays CONJECTURAL).
"""

from __future__ import annotations

import pytest

from tnfr.mathematics import algebraic_residue_networks as arn
from tnfr.mathematics.algebraic_residue_networks import (
    decomposition_signature,
    decomposition_type,
    gaussian_cayley_spectrum_count,
    gaussian_is_unit,
    gaussian_kth_power_set,
    gaussian_mul,
    signature_separates_types,
)

RAMIFIED = [2]
INERT = [3, 7, 11, 19, 23]
SPLIT = [5, 13, 17, 29]


# --------------------------------------------------------------------------- #
# Classical labels (ground truth)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", RAMIFIED)
def test_ramified_label(p):
    assert decomposition_type(p) == "ramified"


@pytest.mark.parametrize("p", INERT)
def test_inert_label(p):
    assert decomposition_type(p) == "inert"


@pytest.mark.parametrize("p", SPLIT)
def test_split_label(p):
    assert decomposition_type(p) == "split"


# --------------------------------------------------------------------------- #
# Ring arithmetic in Z[i]/(p)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [3, 5, 7])
def test_i_squared_is_minus_one(p):
    # (0+1i)^2 = -1 = (p-1, 0)
    assert gaussian_mul((0, 1), (0, 1), p) == ((p - 1) % p, 0)


@pytest.mark.parametrize("p", [3, 5, 7, 11, 13])
def test_unit_count_matches_norm_criterion(p):
    units = [e for e in arn.gaussian_elements(p)
             if e != (0, 0) and gaussian_is_unit(e, p)]
    if decomposition_type(p) == "inert":
        # Z[i]/(p) is the field F_{p^2}: every non-zero element is a unit
        assert len(units) == p * p - 1
    else:  # split: units are (F_p^*)^2
        assert len(units) == (p - 1) ** 2


@pytest.mark.parametrize("p", [3, 5, 7])
def test_kth_power_set_nonempty_and_units(p):
    S = gaussian_kth_power_set(p, 2, units_only=True)
    assert S
    assert (0, 0) not in S


# --------------------------------------------------------------------------- #
# Required test: Gaussian decomposition signature (k = 2 separates types)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", RAMIFIED)
def test_ramified_signature_is_two(p):
    assert decomposition_signature(p, 2)[1] == 2


@pytest.mark.parametrize("p", INERT)
def test_inert_signature_is_three(p):
    assert decomposition_signature(p, 2)[1] == 3


@pytest.mark.parametrize("p", SPLIT)
def test_split_signature_is_six(p):
    assert decomposition_signature(p, 2)[1] == 6


def test_signature_separates_all_three_types_at_k2():
    assert signature_separates_types(RAMIFIED + INERT + SPLIT, 2) is True


# --------------------------------------------------------------------------- #
# Control: the separation is k-sensitive (NT-P05 is conjectural)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [3, 4])
def test_separation_is_k_sensitive(k):
    # higher powers do not give a universal detector across the three types
    assert signature_separates_types([2, 3, 5, 7, 11, 13, 17], k) is False


def test_inert_matches_finite_field_f_p_squared():
    # an inert p gives Z[i]/(p) = F_{p^2}: its k=2 count equals the F_{p^2} one
    from tnfr.mathematics.finite_fields import (
        FiniteField,
        distinct_period_count,
    )
    for p in (3, 7, 11):
        field_count = distinct_period_count(FiniteField(p, 2), 2)
        gaussian_count = gaussian_cayley_spectrum_count(p, 2, units_only=True)
        assert field_count == gaussian_count == 3


# --------------------------------------------------------------------------- #
# Guards and exports
# --------------------------------------------------------------------------- #
def test_kth_power_rejects_nonpositive():
    with pytest.raises(ValueError):
        gaussian_kth_power_set(5, 0)


def test_decomposition_type_rejects_small():
    with pytest.raises(ValueError):
        decomposition_type(1)


def test_module_exports_complete():
    expected = {
        "decomposition_type",
        "gaussian_mul",
        "gaussian_elements",
        "gaussian_is_unit",
        "gaussian_kth_power_set",
        "gaussian_cayley_spectrum_count",
        "decomposition_signature",
        "signature_separates_types",
    }
    assert expected <= set(arn.__all__)
