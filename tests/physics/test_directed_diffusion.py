r"""Tests for the R9 directed non-normal structural dynamics.

A directed circulant (R2 residue digraph) is normal with unit transient gain; a
general directed graph is non-normal and can amplify transiently even with a
stable spectrum. The C4 projector path rejects the non-normal operator; the R9
certificates (transient gain, pseudospectral / Kreiss bound, gated Schur) handle
it without ever calling ``eigh`` on a non-symmetric matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics import directed_diffusion as dd
from tnfr.physics.directed_diffusion import (
    DirectedDynamicsCertificate,
    certify_directed_dynamics,
    directed_cayley_adjacency,
    directed_rw_laplacian,
    is_directed_circulant,
)
from tnfr.physics.spectral_projectors import (
    NonNormalOperatorError,
    matrix_exponential,
    schur_residual,
    scipy_available,
    spectral_abscissa,
    spectral_clusters,
    transient_gain,
)

# A strongly non-normal generator: feed-forward chain with self-loops.
NON_NORMAL = np.array(
    [[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float
)


# --------------------------------------------------------------------------- #
# Directed random-walk Laplacian
# --------------------------------------------------------------------------- #
def test_directed_rw_laplacian_row_sums_zero_for_nonsinks():
    w = directed_cayley_adjacency(6, {1, 2})
    laplacian = directed_rw_laplacian(w)
    # each row of I - D^-1 W sums to zero (row-stochastic transition)
    assert np.allclose(laplacian.sum(axis=1), 0.0)


def test_directed_rw_laplacian_handles_sinks():
    # node 2 is a sink (no out-edges); its row must be all-zero, not NaN
    w = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    laplacian = directed_rw_laplacian(w)
    assert np.all(np.isfinite(laplacian))
    assert np.allclose(laplacian[2], 0.0)


# --------------------------------------------------------------------------- #
# Required test 1: normal / non-normal classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,conn", [(7, {1, 2}), (5, {1}), (8, {1, 3})])
def test_directed_circulant_is_normal(n, conn):
    w = directed_cayley_adjacency(n, conn)
    assert is_directed_circulant(w)
    cert = certify_directed_dynamics(w)
    assert cert.normal
    assert cert.commutator < 1e-9


def test_general_digraph_is_non_normal():
    assert not is_directed_circulant(NON_NORMAL)
    cert = certify_directed_dynamics(NON_NORMAL)
    assert cert.normal is False
    assert cert.commutator > 1e-6


# --------------------------------------------------------------------------- #
# Required test 2: transient-gain controls
# --------------------------------------------------------------------------- #
def test_normal_circulant_has_no_transient_growth():
    cert = certify_directed_dynamics(directed_cayley_adjacency(7, {1, 2}))
    assert cert.transient_gain <= 1.0 + 1e-6
    assert cert.has_transient_amplification is False


def test_non_normal_generator_amplifies_despite_stability():
    cert = certify_directed_dynamics(NON_NORMAL)
    # asymptotically stable spectrum ...
    assert cert.asymptotically_stable
    assert cert.abscissa <= 1e-6
    # ... yet the transient gain exceeds one
    assert cert.transient_gain > 1.0
    assert cert.has_transient_amplification is True


def test_pseudospectral_bound_is_a_lower_bound_on_gain():
    # Kreiss matrix theorem: K(A) <= sup_t ||e^{tA}||
    cert = certify_directed_dynamics(NON_NORMAL)
    assert cert.pseudospectral_bound <= cert.transient_gain + 1e-6
    assert cert.pseudospectral_bound > 1.0  # certifies growth from the resolvent


# --------------------------------------------------------------------------- #
# Required test 3: Schur residual (SciPy-gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not scipy_available(), reason="SciPy not installed")
def test_schur_residual_is_zero():
    laplacian = directed_rw_laplacian(NON_NORMAL)
    assert schur_residual(laplacian) < 1e-9


def test_schur_residual_raises_without_scipy(monkeypatch):
    import tnfr.physics.spectral_projectors as sp

    monkeypatch.setattr(sp, "_HAS_SCIPY", False)
    with pytest.raises(NonNormalOperatorError):
        sp.schur_residual(directed_rw_laplacian(NON_NORMAL))


def test_certificate_schur_field_gated():
    cert = certify_directed_dynamics(NON_NORMAL)
    if scipy_available():
        assert cert.schur_residual is not None
        assert cert.schur_residual < 1e-9
    else:  # pragma: no cover
        assert cert.schur_residual is None


# --------------------------------------------------------------------------- #
# Required test 4: no eigh on non-symmetric matrices
# --------------------------------------------------------------------------- #
def test_c4_projector_path_rejects_non_normal():
    laplacian = directed_rw_laplacian(NON_NORMAL)
    assert not np.allclose(laplacian, laplacian.T)  # non-symmetric
    with pytest.raises(NonNormalOperatorError):
        spectral_clusters(laplacian)


def test_spectral_abscissa_uses_eigvals_on_non_symmetric():
    # spectral_abscissa must work on a non-symmetric matrix (no eigh)
    laplacian = directed_rw_laplacian(NON_NORMAL)
    alpha = spectral_abscissa(-laplacian)
    assert np.isfinite(alpha)
    assert alpha <= 1e-6  # stable generator


# --------------------------------------------------------------------------- #
# Matrix exponential helper
# --------------------------------------------------------------------------- #
def test_matrix_exponential_of_zero_is_identity():
    assert np.allclose(matrix_exponential(np.zeros((3, 3))), np.eye(3))


def test_matrix_exponential_matches_diagonal():
    d = np.diag([-1.0, -2.0, 0.5])
    expected = np.diag([np.exp(-1.0), np.exp(-2.0), np.exp(0.5)])
    assert np.allclose(matrix_exponential(d), expected, atol=1e-9)


def test_transient_gain_of_zero_generator_is_one():
    assert transient_gain(np.zeros((3, 3))) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Certificate and exports
# --------------------------------------------------------------------------- #
def test_certificate_type_and_fields():
    cert = certify_directed_dynamics(directed_cayley_adjacency(5, {1}))
    assert isinstance(cert, DirectedDynamicsCertificate)
    assert isinstance(cert.normal, bool)
    assert isinstance(cert.transient_gain, float)


def test_module_exports_complete():
    expected = {
        "directed_rw_laplacian",
        "directed_cayley_adjacency",
        "is_directed_circulant",
        "DirectedDynamicsCertificate",
        "certify_directed_dynamics",
    }
    assert expected <= set(dd.__all__)
