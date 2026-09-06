r"""Tests for the transient U2/U6 certificate.

These tests cover contracting fixtures and a weighted counterexample. The
ambient gain can additionally include the oblique projection factor ``‖Q‖``;
stationary-weighted contraction is tested separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from tnfr.physics import transient_u2 as tu2
from tnfr.physics.directed_diffusion import (
    consensus_projection,
    directed_cayley_adjacency,
)
from tnfr.physics.transient_u2 import (
    certify_transient_u2,
    kreiss_lower_bound,
    nonconsensus_basis,
    peak_transient_gain,
    potential_operator,
    potential_operator_from_graph,
    restricted_generator,
    structural_potential_peak,
    symmetric_part_min_eig,
)

NORMAL = directed_cayley_adjacency(7, {1, 2})
NON_NORMAL = np.array(
    [[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]], dtype=float
)
STAR_IN = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                   dtype=float)
NON_NORMALS = [NON_NORMAL, STAR_IN]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


X4 = _unit([1.0, -1.0, 0.5, -0.5])


# --------------------------------------------------------------------------- #
# Non-consensus basis
# --------------------------------------------------------------------------- #
def test_nonconsensus_basis_orthonormal_and_spans_subspace():
    v = nonconsensus_basis(NON_NORMAL)
    assert v.shape == (4, 3)
    assert np.allclose(v.T @ v, np.eye(3))                 # orthonormal
    from tnfr.physics.directed_diffusion import stationary_distribution
    pi = stationary_distribution(NON_NORMAL)
    assert np.allclose(pi @ v, 0.0, atol=1e-9)             # spans {pi^T y = 0}


def test_restricted_generator_shape():
    assert restricted_generator(NON_NORMAL).shape == (3, 3)


# --------------------------------------------------------------------------- #
# The canonical finding: Euclidean per-node contraction (no transient)
# --------------------------------------------------------------------------- #
def test_symmetric_part_positive_definite_normal_and_nonnormal():
    assert symmetric_part_min_eig(NORMAL) > 0.0
    for w in NON_NORMALS:
        assert symmetric_part_min_eig(w) > 0.0  # Euclidean contraction


def test_peak_gain_is_one_no_transient_amplification():
    peak_n, _ = peak_transient_gain(NORMAL)
    assert peak_n == pytest.approx(1.0, abs=1e-6)
    for w in NON_NORMALS:
        peak, s_star = peak_transient_gain(w)
        assert peak == pytest.approx(1.0, abs=1e-6)  # no per-node-energy growth
        assert s_star == pytest.approx(0.0, abs=1e-6)


def test_kreiss_lower_bound_at_most_peak_gain():
    for w in [NORMAL, *NON_NORMALS]:
        peak, _ = peak_transient_gain(w)
        assert kreiss_lower_bound(w) <= peak * (1.0 + 1e-6)


def test_ambient_gain_equals_consensus_projection_norm_artifact():
    # the naive ambient "transient > 1" is exactly ‖Q‖ (peak at s = 0)
    for w in NON_NORMALS:
        c = certify_transient_u2(w, X4)
        q_norm = float(np.linalg.norm(consensus_projection(w), 2))
        assert c.consensus_projection_norm == pytest.approx(q_norm, abs=1e-9)
        assert c.ambient_oblique_gain == pytest.approx(q_norm, abs=1e-3)
        assert c.ambient_oblique_gain > 1.0            # naive number > 1
        assert c.no_transient_amplification            # but per-node energy = 1


def test_normal_graph_has_unit_ambient_gain():
    c = certify_transient_u2(NORMAL, _unit([1, -1, 0.5, -0.5, 0.3, -0.2, -0.1]))
    assert c.consensus_projection_norm == pytest.approx(1.0, abs=1e-6)
    assert c.ambient_oblique_gain == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Structural potential operator (U6)
# --------------------------------------------------------------------------- #
def test_potential_operator_inverse_square_kernel():
    # undirected support path 0-1-2: d(0,1)=1, d(1,2)=1, d(0,2)=2
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = potential_operator(w)
    assert np.allclose(np.diag(b), 0.0)
    assert b[0, 1] == pytest.approx(1.0)
    assert b[1, 2] == pytest.approx(1.0)
    assert b[0, 2] == pytest.approx(0.25)   # 1 / 2^2
    assert np.allclose(b, b.T)              # symmetric distance kernel


def test_graph_potential_kernel_preserves_directed_weighted_convention():
    graph = __import__("networkx").DiGraph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (2, 0, 4.0)])
    nodes, kernel = potential_operator_from_graph(graph)
    assert nodes == [0, 1, 2]
    assert kernel[0, 1] == pytest.approx(0.25)
    assert kernel[0, 2] == pytest.approx(1.0 / 25.0)
    assert kernel[1, 0] == pytest.approx(1.0 / 49.0)


def test_structural_potential_peak_within_operator_bound():
    for w in [NORMAL, *NON_NORMALS]:
        x = _unit([(-1) ** i * (1.0 / (i + 1)) for i in range(len(w))])
        peak, bound = structural_potential_peak(w, x)
        assert peak <= bound * (1.0 + 1e-6)


def test_u6_confined_for_small_perturbation():
    # a small non-consensus perturbation keeps Φ_s below the π/2 drift limit
    c = certify_transient_u2(NON_NORMAL, 0.1 * X4)
    assert c.peak_structural_potential < U6_STRUCTURAL_POTENTIAL_LIMIT
    assert c.u6_confined


@pytest.mark.parametrize("kwargs", [{"t_max": -1.0}, {"samples": 1}])
def test_finite_transient_scans_reject_degenerate_windows(kwargs):
    x = _unit([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125])
    with pytest.raises(ValueError):
        peak_transient_gain(NORMAL, **kwargs)
    with pytest.raises(ValueError):
        structural_potential_peak(NORMAL, x, **kwargs)


def test_transient_certificate_reports_unassessed_tail():
    x = _unit([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125])
    certificate = certify_transient_u2(NORMAL, x)
    assert certificate.observation_window_structural == pytest.approx(40.0)
    assert certificate.tail_status == "UNASSESSED_FINITE_WINDOW"
    assert certificate.continuous_u6_status == (
        "INCONCLUSIVE_NO_INTERVAL_OR_TAIL_BOUND"
    )


# --------------------------------------------------------------------------- #
# Certificate bundle
# --------------------------------------------------------------------------- #
def test_certificate_bounds_hold_and_status_open():
    for w in NON_NORMALS:
        c = certify_transient_u2(w, X4)
        assert c.consensus_projection_residual < 1e-9   # LQ = L
        assert c.spectral_abscissa <= 1e-9              # spectrally stable
        assert c.bounds_hold
        assert "OPEN" in c.claim_status                 # U2 metric not decided


def test_certificate_relabel_invariant():
    base = certify_transient_u2(NON_NORMAL, X4)
    perm = [2, 0, 3, 1]
    p = np.zeros((4, 4))
    for i, pi_ in enumerate(perm):
        p[pi_, i] = 1.0
    c = certify_transient_u2(p @ NON_NORMAL @ p.T, p @ X4)
    assert c.peak_gain == pytest.approx(base.peak_gain, abs=1e-6)
    assert c.kreiss_lower_bound == pytest.approx(base.kreiss_lower_bound,
                                                 abs=1e-6)
    assert c.symmetric_part_min_eig == pytest.approx(
        base.symmetric_part_min_eig, abs=1e-6)
    assert c.consensus_projection_norm == pytest.approx(
        base.consensus_projection_norm, abs=1e-6)


def test_weighted_nonconsensus_euclidean_contraction_has_counterexample():
    """The weighted graph class is not universally Euclidean-contracting."""
    weights = np.array([
        [0, 1, 0, 0, 0, 12],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 15, 0, 0],
        [0, 9, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 10],
        [1, 0, 0, 0, 0, 0],
    ], dtype=float)
    vector = np.array([4341, -4028, -4275, -4065, 2273, 4998], dtype=float)
    basis = nonconsensus_basis(weights)
    restricted = basis.T @ tu2.directed_rw_laplacian(weights) @ basis
    symmetric_part = (restricted + restricted.T) / 2.0
    assert np.min(np.linalg.eigvalsh(symmetric_part)) < -1e-4
    assert np.linalg.norm(restricted @ (basis.T @ vector)) > 0.0


def test_module_exports_complete():
    expected = {
        "nonconsensus_basis", "restricted_generator", "symmetric_part_min_eig",
        "peak_transient_gain", "kreiss_lower_bound", "potential_operator",
        "structural_potential_peak", "TransientU2Certificate",
        "certify_transient_u2",
    }
    assert expected <= set(tu2.__all__)
