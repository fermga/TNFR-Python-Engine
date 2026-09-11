r"""Tests for the R9 metric layer and U2 integral semantics (N03).

The Euclidean transient gain of a non-normal diffusion generator can exceed 1,
but in the stationary-weighted norm ``L²(π)`` the semigroup is a contraction
(Jensen), so its gain is ``≤ 1``. This layer reports both norms and both U2
integral readings (signed displacement vs total variation) WITHOUT deciding which
is the canonical U2 quantity — that decision is gated (AGENTS.md §6).
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics import directed_diffusion as dd
from tnfr.physics.directed_diffusion import (
    NormKind,
    directed_cayley_adjacency,
    directed_rw_laplacian,
    induced_operator_norm,
    is_stationary_contraction,
    state_norm,
    stationary_distribution,
    stationary_transient_gain,
    transient_gain_in_norm,
    u2_integral_readings,
)

# Strongly connected, non-normal digraphs with Euclidean transient gain > 1.
NON_NORMAL_SC = [
    np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]],
             dtype=float),
    np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]],
             dtype=float),
]


# --------------------------------------------------------------------------- #
# Stationary distribution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("W", NON_NORMAL_SC)
def test_stationary_distribution_is_left_perron(W):
    pi = stationary_distribution(W)
    P = np.eye(len(W)) - directed_rw_laplacian(W)
    assert np.allclose(pi @ P, pi, atol=1e-9)  # πᵀ P = πᵀ
    assert abs(pi.sum() - 1.0) < 1e-12
    assert np.min(pi) > 0.0


def test_stationary_distribution_rejects_non_strongly_connected():
    # node 2 is absorbing (row [0,0,1]) -> not irreducible
    W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError):
        stationary_distribution(W)


# --------------------------------------------------------------------------- #
# Required: weighted-Markov contraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("W", NON_NORMAL_SC)
def test_weighted_markov_semigroup_is_contractive(W):
    pi = stationary_distribution(W)
    P = np.eye(len(W)) - directed_rw_laplacian(W)
    # ||P||_{2,π} ≤ 1 (Jensen)
    assert induced_operator_norm(P, kind=NormKind.STATIONARY, pi=pi) <= 1.0 + 1e-9
    # and the whole semigroup contracts: peak stationary gain ≤ 1
    assert is_stationary_contraction(W)
    assert stationary_transient_gain(W) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# Required: Euclidean and stationary gains reported separately
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("W", NON_NORMAL_SC)
def test_euclidean_and_stationary_gain_reported_separately(W):
    L = directed_rw_laplacian(W)
    eucl = transient_gain_in_norm(-L, kind=NormKind.EUCLIDEAN)
    stat = stationary_transient_gain(W)
    # the non-normal Euclidean gain exceeds 1; the stationary gain does not
    assert eucl > 1.0
    assert stat <= 1.0 + 1e-6
    assert stat < eucl  # the metrics genuinely disagree


def test_normal_circulant_gain_one_in_both_norms():
    W = directed_cayley_adjacency(7, {1, 2})  # normal circulant
    L = directed_rw_laplacian(W)
    assert transient_gain_in_norm(-L, kind=NormKind.EUCLIDEAN) <= 1.0 + 1e-6
    assert stationary_transient_gain(W) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# U2 integral semantics: signed displacement vs total variation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", [NormKind.EUCLIDEAN, NormKind.STATIONARY])
def test_u2_integral_net_le_total(kind):
    W = NON_NORMAL_SC[1]
    x0 = np.array([1.0, -1.0, 0.5, -0.5])
    r = u2_integral_readings(W, x0, kind=kind)
    assert r.norm_kind == kind.value
    assert r.net <= r.total + 1e-9
    assert r.net_le_total


def test_total_variation_strictly_exceeds_net_when_it_reverses():
    # a perturbation that reverses accumulates more variation than net displacement
    W = NON_NORMAL_SC[1]
    x0 = np.array([2.0, -2.0, 0.0, 0.0])
    r = u2_integral_readings(W, x0, kind=NormKind.EUCLIDEAN)
    assert r.total > r.net  # no cancellation in the total variation


# --------------------------------------------------------------------------- #
# Norm helpers
# --------------------------------------------------------------------------- #
def test_state_norm_euclidean_matches_numpy():
    v = np.array([3.0, 4.0, 0.0])
    assert state_norm(v, kind=NormKind.EUCLIDEAN) == pytest.approx(5.0)


def test_state_norm_stationary_requires_pi():
    with pytest.raises(ValueError):
        state_norm(np.array([1.0, 2.0]), kind=NormKind.STATIONARY)


def test_induced_norm_stationary_requires_pi():
    with pytest.raises(ValueError):
        induced_operator_norm(np.eye(2), kind=NormKind.STATIONARY)


def test_module_exports_complete():
    expected = {
        "NormKind", "stationary_distribution", "state_norm",
        "induced_operator_norm", "transient_gain_in_norm",
        "stationary_transient_gain", "is_stationary_contraction",
        "net_reorganization", "total_reorganization",
        "U2IntegralReadings", "u2_integral_readings",
    }
    assert expected <= set(dd.__all__)
