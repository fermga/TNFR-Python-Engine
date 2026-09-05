r"""Tests for the R9 structural-time theorem (N04).

For a scalar frequency ``ν_f(t) ≥ 0`` the linear EPI transport ``ẋ = −ν_f(t) L x``
has the exact solution ``x(t) = e^{−s(t)L} x₀`` with ``s(t) = ∫ ν_f`` — a clock
change, not a mass. On the non-consensus subspace ``Q = I − 1πᵀ`` the semigroup
decays, so the total reorganization is finite: ``J ≤ M‖LQ‖‖x₀‖/ω``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics import directed_diffusion as dd
from tnfr.physics.directed_diffusion import (
    certify_structural_time,
    clock_change_residual,
    consensus_projection,
    directed_cayley_adjacency,
    directed_rw_laplacian,
    nonconsensus_abscissa,
    reorganization_time_invariance_residual,
    stationary_distribution,
    structural_time,
    sustained_gain,
    total_variation_bound,
)

NON_NORMAL_SC = np.array(
    [[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]], dtype=float
)
X0 = np.array([1.0, -1.0, 0.5, -0.5])


def _vf(t):
    return 1.0 + 0.5 * np.sin(t)


# --------------------------------------------------------------------------- #
# Consensus projection Q = I - 1 pi^T
# --------------------------------------------------------------------------- #
def test_consensus_projection_properties():
    W = NON_NORMAL_SC
    Q = consensus_projection(W)
    L = directed_rw_laplacian(W)
    assert np.allclose(Q @ Q, Q)                 # idempotent
    assert np.allclose(Q @ np.ones(len(W)), 0)   # annihilates consensus
    assert np.allclose(L @ Q, L)                 # LQ = L (Q commutes with L)
    assert np.allclose(Q @ L, L)                 # QL = L
    pi = stationary_distribution(W)
    assert np.allclose(pi @ Q, 0)                # pi^T Q = 0


def test_structural_time_is_cumulative_integral():
    t = np.linspace(0.0, 4.0, 500)
    s = structural_time(lambda x: 2.0, t)  # constant vf=2 -> s(t)=2t
    assert s[0] == 0.0
    assert np.allclose(s, 2.0 * t, atol=1e-9)


# --------------------------------------------------------------------------- #
# Required: scalar-vf structural-time reparameterization (clock change)
# --------------------------------------------------------------------------- #
def test_scalar_vf_structural_time_reparameterization():
    t = np.linspace(0.0, 10.0, 1500)
    # RK4 of the time-varying ODE converges to e^{-s(t)L} x0
    resid = clock_change_residual(NON_NORMAL_SC, X0, _vf, t)
    assert resid < 1e-5


def test_clock_change_residual_shrinks_with_refinement():
    coarse = clock_change_residual(NON_NORMAL_SC, X0, _vf,
                                   np.linspace(0.0, 8.0, 400))
    fine = clock_change_residual(NON_NORMAL_SC, X0, _vf,
                                 np.linspace(0.0, 8.0, 3200))
    assert fine < coarse  # RK4 is convergent


# --------------------------------------------------------------------------- #
# Required: integrated-reorganization change of variables
# --------------------------------------------------------------------------- #
def test_integrated_reorganization_change_of_variables():
    t = np.linspace(0.0, 25.0, 4000)
    resid = reorganization_time_invariance_residual(NON_NORMAL_SC, X0, _vf, t)
    # int vf(t) g(s(t)) dt == int g(s) ds : total reorganization is clock-invariant
    assert resid < 1e-3


# --------------------------------------------------------------------------- #
# Required: normal M=1 vs non-normal M>1 (sustained gain)
# --------------------------------------------------------------------------- #
def test_normal_case_peak_gain_one():
    m = sustained_gain(directed_cayley_adjacency(7, {1, 2}))
    assert m == pytest.approx(1.0, abs=1e-6)


def test_nonnormal_case_sustained_gain_exceeds_one():
    assert sustained_gain(NON_NORMAL_SC) > 1.0


# --------------------------------------------------------------------------- #
# Required: transient bound dominates the measured reorganization
# --------------------------------------------------------------------------- #
def test_transient_bound_dominates_measured_reorganization():
    j, bound, holds = total_variation_bound(NON_NORMAL_SC, X0)
    assert holds
    assert j <= bound * (1.0 + 1e-6)
    assert np.isfinite(bound)


def test_nonconsensus_abscissa_is_positive_spectral_gap():
    omega = nonconsensus_abscissa(NON_NORMAL_SC)
    assert omega > 0.0
    # equals the smallest non-zero |Re λ| of L
    eig = np.linalg.eigvals(directed_rw_laplacian(NON_NORMAL_SC))
    nz = eig[np.abs(eig) > 1e-9]
    assert omega == pytest.approx(float(np.min(nz.real)), abs=1e-9)


# --------------------------------------------------------------------------- #
# Required: certificate relabel invariance
# --------------------------------------------------------------------------- #
def test_certificate_relabel_invariant():
    t = np.linspace(0.0, 8.0, 1200)
    base = certify_structural_time(NON_NORMAL_SC, X0, _vf, t)
    # permute nodes: W'[p[i]][p[j]] = W[i][j], x0'[p[i]] = x0[i]
    perm = [2, 0, 3, 1]
    n = len(perm)
    P = np.zeros((n, n))
    for i, pi_ in enumerate(perm):
        P[pi_, i] = 1.0
    Wp = P @ NON_NORMAL_SC @ P.T
    x0p = P @ X0
    relabelled = certify_structural_time(Wp, x0p, _vf, t)
    assert relabelled.sustained_gain == pytest.approx(base.sustained_gain,
                                                      abs=1e-6)
    assert relabelled.nonconsensus_abscissa == pytest.approx(
        base.nonconsensus_abscissa, abs=1e-6)
    assert relabelled.total_reorganization == pytest.approx(
        base.total_reorganization, abs=1e-4)
    assert relabelled.bound_holds == base.bound_holds


def test_certificate_bundle_fields():
    t = np.linspace(0.0, 8.0, 1000)
    cert = certify_structural_time(NON_NORMAL_SC, X0, _vf, t)
    assert cert.sustained_gain >= 1.0
    assert cert.clock_change_residual < 1e-4
    assert cert.bound_holds is True


def test_module_exports_complete():
    expected = {
        "consensus_projection", "nonconsensus_abscissa", "sustained_gain",
        "structural_time", "clock_change_residual",
        "reorganization_time_invariance_residual", "total_variation_bound",
        "StructuralTimeCertificate", "certify_structural_time",
    }
    assert expected <= set(dd.__all__)
