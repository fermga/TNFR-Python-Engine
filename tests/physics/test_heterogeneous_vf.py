r"""Tests for heterogeneous nodal frequency (R9b, N13).

The scalar clock-change theorem (N04) holds for a common ν_f but **fails** for a
heterogeneous D_{ν_f}(t): the generators D_{ν_f}(t)·L no longer commute, so
x(t) ≠ e^{-s(t)L} x₀. A fixed positive D_{ν_f} is still stable; uniform
time-varying stability stays open.
"""

from __future__ import annotations

import numpy as np

from tnfr.physics import heterogeneous_vf as hv
from tnfr.physics.directed_diffusion import directed_cayley_adjacency
from tnfr.physics.heterogeneous_vf import (
    certify_heterogeneous_vf,
    fixed_generator_abscissa,
    generator_commutator_norm,
    heterogeneous_schedule,
    scalar_schedule,
    scalar_time_ansatz_residual,
)

W_SC = np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]],
                dtype=float)
X4 = np.array([1.0, -1.0, 0.5, -0.5]) / np.linalg.norm([1.0, -1.0, 0.5, -0.5])
T = np.linspace(0.0, 10.0, 600)


# --------------------------------------------------------------------------- #
# Commutators: scalar commutes, heterogeneous does not
# --------------------------------------------------------------------------- #
def test_scalar_frequency_generators_commute():
    # any two scalar multiples of L commute
    assert generator_commutator_norm(2.0 * np.ones(4), 3.0 * np.ones(4),
                                     W_SC) < 1e-9


def test_heterogeneous_frequency_generators_do_not_commute():
    assert generator_commutator_norm(np.array([0.5, 1.5, 1.0, 2.0]),
                                     np.array([2.0, 0.5, 1.5, 1.0]),
                                     W_SC) > 1e-3


# --------------------------------------------------------------------------- #
# The key result: the scalar-time theorem does not extend
# --------------------------------------------------------------------------- #
def test_heterogeneous_vf_does_not_use_scalar_time_theorem():
    common = scalar_schedule(4)
    hetero = heterogeneous_schedule(4)
    res_common = scalar_time_ansatz_residual(W_SC, X4, common, T)
    res_hetero = scalar_time_ansatz_residual(W_SC, X4, hetero, T)
    assert res_common < 1e-3            # common ν_f: clock change holds
    assert res_hetero > 1e-2            # heterogeneous: ansatz fails
    assert res_hetero > 10 * res_common


def test_common_schedule_is_an_exact_clock_change():
    res = scalar_time_ansatz_residual(
        directed_cayley_adjacency(7, {1, 2}),
        np.ones(7) / np.sqrt(7) * np.array([1, -1, 1, -1, 1, -1, 1]),
        scalar_schedule(7), T)
    assert res < 1e-3


# --------------------------------------------------------------------------- #
# Fixed-D stability (frozen generator)
# --------------------------------------------------------------------------- #
def test_fixed_heterogeneous_generator_is_stable():
    # a frozen positive D keeps -D L stable (consensus preserved, rest decays)
    for vf in ([0.5, 1.5, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [0.3, 2.0, 0.7, 1.1]):
        assert fixed_generator_abscissa(np.array(vf), W_SC) <= 1e-6


# --------------------------------------------------------------------------- #
# Certificate + exports
# --------------------------------------------------------------------------- #
def test_certificate_marks_theorem_boundary():
    cert = certify_heterogeneous_vf(W_SC, X4, T)
    assert cert.commutator_scalar < 1e-9
    assert cert.commutator_heterogeneous > 1e-3
    assert cert.scalar_time_residual_common < 1e-3
    assert cert.scalar_time_residual_heterogeneous > 1e-2
    assert cert.scalar_time_theorem_extends is False
    assert cert.fixed_generator_stable is True
    assert "OPEN" in cert.claim_status
    assert "unmodified" in cert.claim_status


def test_module_exports_complete():
    expected = {
        "heterogeneous_generator", "generator_commutator_norm",
        "scalar_schedule", "heterogeneous_schedule", "structural_time_mean",
        "scalar_time_ansatz_residual", "fixed_generator_abscissa",
        "nonconsensus_transient_gain", "HeterogeneousVfCertificate",
        "certify_heterogeneous_vf",
    }
    assert expected <= set(hv.__all__)
