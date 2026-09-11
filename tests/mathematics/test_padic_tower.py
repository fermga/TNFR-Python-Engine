r"""Tests for the R4 projective p-adic network tower.

For the reduction-compatible family ``S_e = {x : x mod p ∈ base}`` on the tower
``ℤ/pℤ ← ℤ/p²ℤ ← ⋯`` the random-walk transport is projective, exactly over ℚ:

    R_e P_{e+1} = P_e R_e,   P_{e+1} Lift_e = Lift_e P_e,   R_e Lift_e = I,

so ``spec(L_e) ⊆ spec(L_{e+1})`` (surviving modes).  A non-uniform fine set is
the control (commutation fails).  REMESH is not claimed: the contract audit
stays unverified (``NT-P04`` CONJECTURAL).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics import padic_tower as pt
from tnfr.mathematics.padic_tower import (
    RemeshContractAudit,
    compatible_connection_set,
    lift_intertwining_residual,
    lift_reduction_residual,
    laplacian_commutation_residual,
    padic_lift_map,
    padic_laplacian,
    padic_spectral_gaps,
    padic_transition,
    projective_commutation_residual,
    projective_scale_map,
    remesh_contract_audit,
    surviving_spectrum_containment,
)
from tnfr.physics.spectral_projectors import derived_tolerance

# Small primes and low exponents; both a single-generator base and the units.
PRIMES = [2, 3, 5, 7]
LEVELS = [1, 2]
BASES = {2: [frozenset({1})],
         3: [frozenset({1}), frozenset({1, 2})],
         5: [frozenset({1}), frozenset({1, 2, 3, 4})],
         7: [frozenset({1}), frozenset({1, 2, 3, 4, 5, 6})]}


def _cases():
    for p in PRIMES:
        for e in LEVELS:
            if p ** (e + 1) > 128:
                continue
            for base in BASES[p]:
                yield p, e, base


CASES = list(_cases())


# --------------------------------------------------------------------------- #
# Scale maps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,e", [(2, 1), (3, 1), (3, 2), (5, 1)])
def test_reduction_map_is_stochastic(p, e):
    R = projective_scale_map(p, e)
    assert len(R) == p ** e
    assert len(R[0]) == p ** (e + 1)
    for row in R:
        assert sum(row) == Fraction(1)  # each fiber averages to 1
        for x in row:
            assert x in (Fraction(0), Fraction(1, p))


@pytest.mark.parametrize("p,e", [(2, 1), (3, 1), (3, 2), (5, 1)])
def test_lift_map_is_constant_on_fibers(p, e):
    Lift = padic_lift_map(p, e)
    assert len(Lift) == p ** (e + 1)
    assert len(Lift[0]) == p ** e
    for x in range(p ** (e + 1)):
        # row x is the indicator of the coarse node x mod p^e
        assert Lift[x][x % (p ** e)] == Fraction(1)
        assert sum(Lift[x]) == Fraction(1)


# --------------------------------------------------------------------------- #
# Required test 1: projective commutation (exact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,e,base", CASES)
def test_projective_commutation_is_exact(p, e, base):
    assert projective_commutation_residual(p, e, base) == Fraction(0)


@pytest.mark.parametrize("p,e,base", CASES)
def test_laplacian_commutation_is_exact(p, e, base):
    assert laplacian_commutation_residual(p, e, base) == Fraction(0)


# --------------------------------------------------------------------------- #
# Required test 2: lift/reduction consistency (exact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,e", [(2, 1), (2, 2), (3, 1), (3, 2), (5, 1),
                                 (7, 1)])
def test_lift_is_right_inverse_of_reduction(p, e):
    assert lift_reduction_residual(p, e) == Fraction(0)


@pytest.mark.parametrize("p,e,base", CASES)
def test_lift_intertwines_levels(p, e, base):
    # P_{e+1} Lift = Lift P_e  ->  spec(L_e) subset spec(L_{e+1}).
    assert lift_intertwining_residual(p, e, base) == Fraction(0)


@pytest.mark.parametrize("p,e,base", CASES)
def test_surviving_spectrum_containment(p, e, base):
    dist = surviving_spectrum_containment(p, e, base)
    tol = derived_tolerance(
        np.array(
            [[float(x) for x in row]
             for row in padic_laplacian(
                 p, e + 1, compatible_connection_set(p, e + 1, base))],
            dtype=float,
        )
    )
    assert dist <= tol


# --------------------------------------------------------------------------- #
# Required test 3: scale reproducibility (deterministic, exact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,e,base", CASES)
def test_transport_is_deterministic(p, e, base):
    c = compatible_connection_set(p, e, base)
    P1 = padic_transition(p, e, c)
    P2 = padic_transition(p, e, c)
    assert P1 == P2
    assert all(isinstance(x, Fraction) for row in P1 for x in row)


def test_spectral_gaps_are_stable_and_reproducible():
    g1 = padic_spectral_gaps(3, 3, frozenset({1}))
    g2 = padic_spectral_gaps(3, 3, frozenset({1}))
    assert g1 == g2
    assert [e for e, _ in g1] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Control: a non-uniform fine set breaks commutation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,e", [(3, 2), (5, 2), (2, 3)])
def test_non_uniform_fine_set_breaks_commutation(p, e):
    base = frozenset(range(1, p))
    R = projective_scale_map(p, e)
    Pe = padic_transition(p, e, compatible_connection_set(p, e, base))
    full = compatible_connection_set(p, e + 1, base)
    broken = set(full)
    broken.discard(max(full))  # drop one fiber element -> non-uniform
    Pe1 = padic_transition(p, e + 1, broken)
    resid = pt._maxabs(pt._sub(pt._matmul(R, Pe1), pt._matmul(Pe, R)))
    assert resid > Fraction(0)


# --------------------------------------------------------------------------- #
# REMESH contract audit — not claimed until it passes
# --------------------------------------------------------------------------- #
def test_remesh_not_claimed_by_default():
    audit = remesh_contract_audit()
    assert isinstance(audit, RemeshContractAudit)
    assert audit.realizes_remesh is False
    d = audit.to_dict()
    assert d["realizes_remesh"] is False
    assert not any(
        d[k] for k in (
            "epi_recursion_verified",
            "network_scale_verified",
            "identity_preserved_verified",
            "u5_multiscale_verified",
        )
    )


def test_remesh_realized_only_when_all_conditions_hold():
    partial = RemeshContractAudit(epi_recursion_verified=True,
                                  network_scale_verified=True)
    assert partial.realizes_remesh is False
    full = RemeshContractAudit(True, True, True, True)
    assert full.realizes_remesh is True


# --------------------------------------------------------------------------- #
# Guards and exports
# --------------------------------------------------------------------------- #
def test_empty_base_rejected():
    with pytest.raises(ValueError):
        compatible_connection_set(3, 2, frozenset())


def test_zero_residue_rejected():
    with pytest.raises(ValueError):
        compatible_connection_set(3, 2, frozenset({0, 1}))


def test_module_exports_complete():
    expected = {
        "projective_scale_map",
        "padic_lift_map",
        "compatible_connection_set",
        "padic_transition",
        "padic_laplacian",
        "projective_commutation_residual",
        "laplacian_commutation_residual",
        "lift_intertwining_residual",
        "lift_reduction_residual",
        "surviving_spectrum_containment",
        "padic_spectral_gaps",
        "RemeshContractAudit",
        "remesh_contract_audit",
    }
    assert expected <= set(pt.__all__)
