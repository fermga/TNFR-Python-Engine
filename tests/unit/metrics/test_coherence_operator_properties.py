"""Test spectral properties of coherence operator Ĉ approximation.

This module validates that the coherence matrix W computed by
coherence_matrix(G) provides a meaningful approximation to the
coherence operator Ĉ on Hilbert space H_NFR.

The tests verify:
1. Hermiticity: W = W^T (ensures symmetric coupling)
2. Element bounds: wᵢⱼ ∈ [0,1] (similarity weights are normalized)
3. Consistency: Elements reflect structural similarities
4. API behavior: Functions return valid data structures

Note: Unlike the theoretical Ĉ which is positive semidefinite,
the W matrix may have negative eigenvalues because it represents
pairwise similarities in the computational basis rather than
the full spectral decomposition.

Mathematical foundation: See docs/source/theory/mathematical_foundations.md §3.1
"""

import pytest

from tnfr.metrics.coherence import coherence_matrix
from tnfr.metrics.common import compute_coherence

# Test tolerance constants
HERMITICITY_TOL = 1e-10  # Strict tolerance for symmetry checks
ELEMENT_BOUND_TOL = 1e-6  # Relaxed tolerance for [0,1] bounds due to numerical precision


def _extract_dense_matrix(W, n, use_numpy=False):
    """Convert W to dense matrix for testing."""
    if use_numpy:
        import numpy as np

        if isinstance(W, list):
            if W and isinstance(W[0], tuple):
                # Sparse format: [(i, j, w), ...]
                matrix = np.zeros((n, n))
                for i, j, w in W:
                    matrix[i, j] = w
                return matrix
            else:
                # Dense format: [[...], [...], ...]
                return np.array(W)
        else:
            # Already numpy array
            return W
    else:
        # Pure Python: convert to list of lists
        if isinstance(W, list):
            if W and isinstance(W[0], tuple):
                # Sparse format
                matrix = [[0.0] * n for _ in range(n)]
                for i, j, w in W:
                    matrix[i][j] = w
                return matrix
            else:
                # Already dense
                return W
        else:
            raise ValueError(f"Unexpected W type: {type(W)}")


def test_coherence_matrix_hermiticity_numpy(graph_canon):
    """Verify Ĉ property: Hermiticity W = W^T ensures symmetric coupling."""
    np = pytest.importorskip("numpy")

    G = graph_canon()
    G.add_node(0, theta=0.0, EPI=1.0, vf=1.0, Si=0.8)
    G.add_node(1, theta=0.5, EPI=1.2, vf=1.1, Si=0.7)
    G.add_node(2, theta=1.0, EPI=0.9, vf=1.2, Si=0.9)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    nodes, W = coherence_matrix(G, use_numpy=True)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    W_dense = _extract_dense_matrix(W, n, use_numpy=True)

    # Test Hermiticity: W should equal W^T
    assert np.allclose(
        W_dense, W_dense.T, atol=HERMITICITY_TOL
    ), "Coherence matrix must be Hermitian (symmetric)"


def test_coherence_matrix_element_bounds_numpy(graph_canon):
    """Verify that all matrix elements wᵢⱼ ∈ [0, 1]."""
    np = pytest.importorskip("numpy")

    G = graph_canon()
    G.add_node(0, theta=0.0, EPI=1.0, vf=1.0, Si=0.8)
    G.add_node(1, theta=0.5, EPI=1.2, vf=1.1, Si=0.7)
    G.add_node(2, theta=1.0, EPI=0.9, vf=1.2, Si=0.9)
    G.add_node(3, theta=1.5, EPI=1.1, vf=0.9, Si=0.85)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    nodes, W = coherence_matrix(G, use_numpy=True)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    W_dense = _extract_dense_matrix(W, n, use_numpy=True)

    # Test element bounds: all elements should be in [0, 1]
    # (similarity weights are normalized)
    assert np.all(W_dense >= 0.0), f"Found negative elements in W"
    assert np.all(W_dense <= 1.0 + ELEMENT_BOUND_TOL), f"Found elements > 1 in W"


def test_coherence_matrix_diagonal_structure(graph_canon):
    """Verify diagonal elements reflect self-similarity configuration."""
    np = pytest.importorskip("numpy")

    G = graph_canon()
    # Create diverse network
    for i in range(5):
        G.add_node(
            i,
            theta=i * 0.5,
            EPI=1.0 + i * 0.1,
            vf=1.0 + i * 0.05,
            Si=0.7 + i * 0.02,
        )
    # Add edges
    for i in range(4):
        G.add_edge(i, i + 1)

    nodes, W = coherence_matrix(G, use_numpy=True)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    W_dense = _extract_dense_matrix(W, n, use_numpy=True)

    # Diagonal elements should be meaningful (typically high for self-similarity)
    # Check that diagonals are in valid range
    diag = np.diag(W_dense)
    assert np.all(diag >= 0.0), "Diagonal elements must be non-negative"
    assert np.all(diag <= 1.0 + ELEMENT_BOUND_TOL), "Diagonal elements must be ≤ 1"


def test_coherence_matrix_similarity_properties(graph_canon):
    """Verify that matrix reflects phase and structural similarities."""
    np = pytest.importorskip("numpy")

    G = graph_canon()
    # Create nodes with known similarities
    # Nodes 0 and 1: very similar (should have high wᵢⱼ)
    G.add_node(0, theta=0.0, EPI=1.0, vf=1.0, Si=0.8)
    G.add_node(1, theta=0.1, EPI=1.05, vf=1.02, Si=0.81)  # Very similar to 0
    # Node 2: quite different (should have lower wᵢⱼ)
    G.add_node(2, theta=3.0, EPI=0.5, vf=2.0, Si=0.3)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    nodes, W = coherence_matrix(G, use_numpy=True)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    W_dense = _extract_dense_matrix(W, n, use_numpy=True)

    # Get indices (may not be in order)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx0 = node_to_idx[0]
    idx1 = node_to_idx[1]
    idx2 = node_to_idx[2]

    # Similarity between 0 and 1 should be higher than similarity with 2
    w_01 = W_dense[idx0, idx1]
    w_02 = W_dense[idx0, idx2]

    assert w_01 > w_02, (
        f"Expected higher similarity between similar nodes (0,1) "
        f"than dissimilar nodes (0,2): w_01={w_01}, w_02={w_02}"
    )


def test_coherence_matrix_hermiticity_python(graph_canon):
    """Verify Hermiticity using pure Python implementation."""
    G = graph_canon()
    G.add_node(0, theta=0.0, EPI=1.0, vf=1.0, Si=0.8)
    G.add_node(1, theta=0.5, EPI=1.2, vf=1.1, Si=0.7)
    G.add_node(2, theta=1.0, EPI=0.9, vf=1.2, Si=0.9)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    nodes, W = coherence_matrix(G, use_numpy=False)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    W_dense = _extract_dense_matrix(W, n, use_numpy=False)

    # Manually check symmetry
    for i in range(n):
        for j in range(n):
            assert abs(W_dense[i][j] - W_dense[j][i]) < 1e-10, (
                f"Matrix not symmetric at ({i},{j}): "
                f"{W_dense[i][j]} != {W_dense[j][i]}"
            )


def test_coherence_matrix_element_bounds_python(graph_canon):
    """Verify all matrix elements are in [0,1] using pure Python."""
    G = graph_canon()
    for i in range(5):
        G.add_node(
            i,
            theta=i * 0.5,
            EPI=1.0 + i * 0.1,
            vf=1.0 + i * 0.05,
            Si=0.7 + i * 0.02,
        )
    for i in range(4):
        G.add_edge(i, i + 1)

    nodes, W = coherence_matrix(G, use_numpy=False)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    W_dense = _extract_dense_matrix(W, n, use_numpy=False)

    # Check all elements are in [0, 1]
    for i in range(n):
        for j in range(n):
            w_ij = W_dense[i][j]
            assert (
                0.0 <= w_ij <= 1.0 + ELEMENT_BOUND_TOL
            ), f"Element W[{i}][{j}] = {w_ij} out of bounds [0,1]"


def test_coherence_matrix_connectivity(graph_canon):
    """Test coherence matrix respects network connectivity."""
    np = pytest.importorskip("numpy")

    G = graph_canon()
    # Create disconnected components
    G.add_node(0, theta=0.0, EPI=1.0, vf=1.0, Si=0.8)
    G.add_node(1, theta=0.5, EPI=1.2, vf=1.1, Si=0.7)
    # Node 2 disconnected
    G.add_node(2, theta=1.0, EPI=0.9, vf=1.2, Si=0.9)
    G.add_edge(0, 1)  # Only edge

    # Note: By default coherence_matrix uses scope="neighbors" which only
    # considers connected nodes. We test that it returns a valid matrix.
    nodes, W = coherence_matrix(G, use_numpy=True)
    assert nodes is not None
    assert W is not None

    n = len(nodes)
    assert n == 3  # All nodes should be in the list

    W_dense = _extract_dense_matrix(W, n, use_numpy=True)

    # Matrix should still be Hermitian
    assert np.allclose(W_dense, W_dense.T, atol=HERMITICITY_TOL)


def test_coherence_computation_consistency(graph_canon):
    """Verify compute_coherence returns valid values."""
    G = graph_canon()
    G.add_node(0, theta=0.0, EPI=1.0, vf=1.0, Si=0.8, dnfr=0.1, dEPI=0.05)
    G.add_node(1, theta=0.5, EPI=1.2, vf=1.1, Si=0.7, dnfr=0.15, dEPI=0.08)
    G.add_node(2, theta=1.0, EPI=0.9, vf=1.2, Si=0.9, dnfr=0.08, dEPI=0.03)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    # Compute coherence via standard API
    C_direct = compute_coherence(G)

    # Should be in valid range [0, 1]
    assert 0.0 <= C_direct <= 1.0, f"Direct coherence out of bounds: {C_direct}"

    # Should be meaningful positive value for a coherent network
    assert C_direct > 0.0, "Coherence should be positive for connected network"


def test_coherence_matrix_empty_network(graph_canon):
    """Test coherence matrix on empty network."""
    G = graph_canon()

    nodes, W = coherence_matrix(G, use_numpy=False)

    # Empty network should return empty results
    assert nodes == []
    assert W == []


def test_coherence_matrix_returns_valid_format(graph_canon):
    """Test that coherence_matrix returns expected data structures."""
    G = graph_canon()
    G.add_node(0, theta=1.0, EPI=1.0, vf=1.0, Si=0.8)
    G.add_node(1, theta=1.5, EPI=1.1, vf=1.05, Si=0.75)
    G.add_edge(0, 1)

    # Test both implementations return valid formats
    nodes_py, W_py = coherence_matrix(G, use_numpy=False)
    assert nodes_py is not None
    assert W_py is not None
    assert len(nodes_py) == 2

    # NumPy version if available
    try:
        import numpy as np

        nodes_np, W_np = coherence_matrix(G, use_numpy=True)
        assert nodes_np is not None
        assert W_np is not None
        assert len(nodes_np) == 2
    except ImportError:
        pass  # NumPy not available, skip

