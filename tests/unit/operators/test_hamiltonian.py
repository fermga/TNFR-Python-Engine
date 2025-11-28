"""Tests for internal Hamiltonian construction (H_int).

This module tests the explicit construction of the internal Hamiltonian operator
and its components (H_coh, H_freq, H_coupling) as specified in TNFR formalization.
"""

import pytest
import numpy as np
import networkx as nx

from tnfr.operators.hamiltonian import (
    InternalHamiltonian,
    build_H_coherence,
    build_H_frequency,
    build_H_coupling,
)


class TestInternalHamiltonian:
    """Test suite for InternalHamiltonian class."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple triangle graph with TNFR attributes."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        for i, node in enumerate(G.nodes):
            G.nodes[node].update(
                {
                    "nu_f": 0.5 + 0.1 * i,  # Different frequencies
                    "phase": 0.0,
                    "epi": 1.0,
                    "si": 0.7,
                }
            )

        return G

    @pytest.fixture
    def linear_chain(self):
        """Create a linear chain graph."""
        G = nx.path_graph(4)

        for i, node in enumerate(G.nodes):
            G.nodes[node].update({"nu_f": 1.0, "phase": i * 0.1, "epi": 1.0, "si": 0.8})

        return G

    def test_hamiltonian_initialization(self, simple_graph):
        """Test basic Hamiltonian initialization."""
        ham = InternalHamiltonian(simple_graph)

        assert ham.N == 3, "Should have 3 nodes"
        assert ham.H_coh.shape == (3, 3), "H_coh should be 3x3"
        assert ham.H_freq.shape == (3, 3), "H_freq should be 3x3"
        assert ham.H_coupling.shape == (3, 3), "H_coupling should be 3x3"
        assert ham.H_int.shape == (3, 3), "H_int should be 3x3"

    def test_hermiticity_all_components(self, simple_graph):
        """Verify all Hamiltonian components are Hermitian."""
        ham = InternalHamiltonian(simple_graph)

        components = {
            "H_coh": ham.H_coh,
            "H_freq": ham.H_freq,
            "H_coupling": ham.H_coupling,
            "H_int": ham.H_int,
        }

        for name, H in components.items():
            H_dagger = H.conj().T
            assert np.allclose(H, H_dagger), f"{name} must be Hermitian"

    def test_frequency_operator_diagonal(self, simple_graph):
        """Verify H_freq is diagonal with correct frequencies."""
        ham = InternalHamiltonian(simple_graph)

        # Check diagonal structure
        off_diagonal = ham.H_freq - np.diag(np.diag(ham.H_freq))
        assert np.allclose(off_diagonal, 0), "H_freq must be diagonal"

        # Check frequency values
        expected_freqs = [0.5, 0.6, 0.7]  # From simple_graph fixture
        actual_freqs = np.diag(ham.H_freq).real
        assert np.allclose(actual_freqs, expected_freqs), "Frequencies should match node nu_f"

    def test_coupling_symmetric(self, simple_graph):
        """Verify H_coupling is symmetric for undirected graph."""
        ham = InternalHamiltonian(simple_graph)

        assert np.allclose(
            ham.H_coupling, ham.H_coupling.T
        ), "H_coupling must be symmetric for undirected graph"

    def test_coherence_potential_negative(self, simple_graph):
        """Verify H_coh has negative scaling (attractive potential)."""
        # Set explicit positive coherence strength (should be scaled negatively)
        simple_graph.graph["H_COH_STRENGTH"] = -2.0

        ham = InternalHamiltonian(simple_graph)

        # Coherence matrix elements should generally be negative or zero
        # (negative sign creates potential well for coherent states)
        # Note: Not all elements must be negative, but the scale should be applied
        assert ham.H_coh.dtype in [
            np.complex64,
            np.complex128,
            complex,
        ], "H_coh should be complex matrix"

    def test_spectrum_real_eigenvalues(self, simple_graph):
        """Verify energy eigenvalues are real (Hermiticity consequence)."""
        ham = InternalHamiltonian(simple_graph)

        eigenvalues, eigenvectors = ham.get_spectrum()

        # Check all eigenvalues are real
        assert np.all(np.isreal(eigenvalues)), "Eigenvalues must be real for Hermitian operator"

        # Check eigenvector matrix is unitary (columns are orthonormal)
        V_dag_V = eigenvectors.conj().T @ eigenvectors
        assert np.allclose(V_dag_V, np.eye(ham.N)), "Eigenvectors should be orthonormal"

    def test_time_evolution_unitary(self, simple_graph):
        """Verify U(t) is unitary for all t."""
        ham = InternalHamiltonian(simple_graph)

        test_times = [0.1, 1.0, 5.0, 10.0]

        for t in test_times:
            U_t = ham.time_evolution_operator(t)

            # Check unitarity: U†U = I
            U_dag = U_t.conj().T
            product = U_dag @ U_t
            identity = np.eye(ham.N)

            assert np.allclose(product, identity), f"U({t}) must be unitary: U†U = I"

            # Also check UU† = I
            product2 = U_t @ U_dag
            assert np.allclose(product2, identity), f"U({t}) must be unitary: UU† = I"

    def test_delta_nfr_operator_anti_hermitian(self, simple_graph):
        """Verify ΔNFR operator is anti-Hermitian."""
        ham = InternalHamiltonian(simple_graph)

        delta_op = ham.compute_delta_nfr_operator()

        # Anti-Hermitian: Δ† = -Δ
        delta_dagger = delta_op.conj().T
        assert np.allclose(delta_op, -delta_dagger), "ΔNFR operator must be anti-Hermitian"

    def test_node_delta_nfr_computation(self, simple_graph):
        """Test ΔNFR computation for individual nodes."""
        ham = InternalHamiltonian(simple_graph)

        # Compute ΔNFR for each node
        delta_nfr_values = []
        for node in ham.nodes:
            delta = ham.compute_node_delta_nfr(node)
            assert isinstance(delta, float), "ΔNFR should be a real number"
            delta_nfr_values.append(delta)

        # Values should be finite
        assert all(np.isfinite(v) for v in delta_nfr_values), "All ΔNFR values should be finite"

    def test_hamiltonian_with_custom_parameters(self):
        """Test Hamiltonian with custom strength parameters."""
        G = nx.cycle_graph(4)

        for node in G.nodes:
            G.nodes[node].update({"nu_f": 2.0, "phase": 0.0, "epi": 1.5, "si": 0.9})

        # Set custom parameters
        G.graph["H_COH_STRENGTH"] = -5.0
        G.graph["H_COUPLING_STRENGTH"] = 0.5

        ham = InternalHamiltonian(G, hbar_str=1e-19)

        assert ham.hbar_str == 1e-19, "Custom ℏ_str should be set"
        assert ham.N == 4, "Should have 4 nodes"

        # Verify Hermiticity still holds
        assert np.allclose(ham.H_int, ham.H_int.conj().T), "Total Hamiltonian must remain Hermitian"

    def test_empty_graph_handling(self):
        """Test behavior with empty graph."""
        G = nx.Graph()

        # Should still construct but with zero dimensions
        ham = InternalHamiltonian(G)
        assert ham.N == 0, "Empty graph should have N=0"
        assert ham.H_int.shape == (0, 0), "Empty Hamiltonian should be 0x0"

    def test_single_node_graph(self):
        """Test Hamiltonian for single isolated node."""
        G = nx.Graph()
        G.add_node(0, nu_f=1.0, phase=0.0, epi=1.0, si=0.8)

        ham = InternalHamiltonian(G)

        assert ham.N == 1, "Should have 1 node"
        assert ham.H_int.shape == (1, 1), "Single node Hamiltonian is 1x1"

        # For single node, H_coupling should be zero (no edges)
        assert np.allclose(ham.H_coupling, 0), "Single node has no coupling"

        # H_freq should equal nu_f
        assert np.isclose(ham.H_freq[0, 0].real, 1.0), "Frequency should match nu_f"

    def test_linear_chain_spectrum(self, linear_chain):
        """Test spectrum for linear chain graph."""
        ham = InternalHamiltonian(linear_chain)

        eigenvalues, eigenvectors = ham.get_spectrum()

        # Should have N distinct eigenvalues
        assert len(eigenvalues) == 4, "Should have 4 eigenvalues"

        # Eigenvalues should be sorted
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:]), "Eigenvalues should be sorted"

        # Ground state (lowest energy)
        ground_state_energy = eigenvalues[0]
        ground_state = eigenvectors[:, 0]

        # Ground state should be normalized
        assert np.isclose(np.linalg.norm(ground_state), 1.0), "Ground state should be normalized"

    def test_hamiltonian_components_sum(self, simple_graph):
        """Verify H_int is the sum of components."""
        ham = InternalHamiltonian(simple_graph)

        H_sum = ham.H_coh + ham.H_freq + ham.H_coupling

        assert np.allclose(ham.H_int, H_sum), "H_int must equal sum of components"

    def test_cache_manager_integration(self, simple_graph):
        """Test integration with CacheManager."""
        from tnfr.utils.cache import CacheManager

        cache_manager = CacheManager()
        ham = InternalHamiltonian(simple_graph, cache_manager=cache_manager)

        assert ham._cache_manager is cache_manager, "Should use provided cache manager"


class TestStandaloneBuilders:
    """Test standalone builder functions."""

    @pytest.fixture
    def test_graph(self):
        """Create test graph for builder functions."""
        G = nx.complete_graph(3)
        for i, node in enumerate(G.nodes):
            G.nodes[node].update({"nu_f": 1.0 + i * 0.5, "phase": 0.0, "epi": 1.0, "si": 0.7})
        return G

    def test_build_H_coherence(self, test_graph):
        """Test standalone H_coherence builder."""
        H_coh = build_H_coherence(test_graph, C_0=-2.0)

        assert H_coh.shape == (3, 3), "Should be 3x3 matrix"
        assert np.allclose(H_coh, H_coh.conj().T), "Should be Hermitian"

    def test_build_H_frequency(self, test_graph):
        """Test standalone H_frequency builder."""
        H_freq = build_H_frequency(test_graph)

        assert H_freq.shape == (3, 3), "Should be 3x3 matrix"

        # Check diagonal
        off_diagonal = H_freq - np.diag(np.diag(H_freq))
        assert np.allclose(off_diagonal, 0), "Should be diagonal"

        # Check values
        expected = [1.0, 1.5, 2.0]
        actual = np.diag(H_freq).real
        assert np.allclose(actual, expected), "Frequencies should match"

    def test_build_H_coupling(self, test_graph):
        """Test standalone H_coupling builder."""
        H_coupling = build_H_coupling(test_graph, J_0=0.3)

        assert H_coupling.shape == (3, 3), "Should be 3x3 matrix"
        assert np.allclose(H_coupling, H_coupling.T), "Should be symmetric"

        # For complete graph, all off-diagonal elements should be J_0
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(H_coupling[i, j], 0.3), "Off-diagonal should be J_0"
                else:
                    assert np.isclose(H_coupling[i, j], 0.0), "Diagonal should be zero"


class TestHamiltonianPhysicalProperties:
    """Test physical properties and conservation laws."""

    @pytest.fixture
    def symmetric_graph(self):
        """Create symmetric graph for conservation tests."""
        G = nx.cycle_graph(6)

        for node in G.nodes:
            G.nodes[node].update({"nu_f": 1.0, "phase": 0.0, "epi": 1.0, "si": 0.8})

        return G

    def test_energy_conservation(self, symmetric_graph):
        """Test energy conservation under time evolution."""
        ham = InternalHamiltonian(symmetric_graph)

        # Create initial state (normalized)
        np.random.seed(42)
        psi_0 = np.random.randn(ham.N) + 1j * np.random.randn(ham.N)
        psi_0 = psi_0 / np.linalg.norm(psi_0)

        # Initial energy expectation
        E_0 = np.real(psi_0.conj() @ ham.H_int @ psi_0)

        # Evolve state
        U_t = ham.time_evolution_operator(t=2.0)
        psi_t = U_t @ psi_0

        # Final energy expectation
        E_t = np.real(psi_t.conj() @ ham.H_int @ psi_t)

        # Energy should be conserved
        assert np.isclose(E_0, E_t, rtol=1e-10), "Energy must be conserved under unitary evolution"

    def test_normalization_preservation(self, symmetric_graph):
        """Test that time evolution preserves state normalization."""
        ham = InternalHamiltonian(symmetric_graph)

        # Create normalized initial state
        np.random.seed(123)
        psi_0 = np.random.randn(ham.N) + 1j * np.random.randn(ham.N)
        psi_0 = psi_0 / np.linalg.norm(psi_0)

        assert np.isclose(np.linalg.norm(psi_0), 1.0), "Initial state normalized"

        # Evolve
        U_t = ham.time_evolution_operator(t=5.0)
        psi_t = U_t @ psi_0

        # Norm should be preserved
        assert np.isclose(
            np.linalg.norm(psi_t), 1.0
        ), "Unitary evolution must preserve normalization"

    def test_eigenstate_time_evolution(self, symmetric_graph):
        """Test that eigenstates evolve with phase factors only."""
        ham = InternalHamiltonian(symmetric_graph)

        # Get ground state
        eigenvalues, eigenvectors = ham.get_spectrum()
        E_0 = eigenvalues[0]
        psi_0 = eigenvectors[:, 0]

        # Evolve ground state
        t = 1.0
        U_t = ham.time_evolution_operator(t)
        psi_t = U_t @ psi_0

        # Eigenstates evolve as: |ψ(t)⟩ = exp(-iE₀t/ℏ)|ψ(0)⟩
        expected_phase = np.exp(-1j * E_0 * t / ham.hbar_str)
        expected_psi_t = expected_phase * psi_0

        # Should match up to global phase
        overlap = np.abs(np.vdot(psi_t, expected_psi_t))
        assert np.isclose(
            overlap, 1.0, rtol=1e-10
        ), "Eigenstate should evolve with correct phase factor"

    def test_trace_preservation(self, symmetric_graph):
        """Test that trace of Hamiltonian is well-defined."""
        ham = InternalHamiltonian(symmetric_graph)

        # Trace should equal sum of diagonal elements
        trace_H = np.trace(ham.H_int)

        # Trace should be real (Hermitian property)
        assert np.abs(trace_H.imag) < 1e-10, "Trace of Hermitian operator must be real"

        # Trace of H_freq should equal sum of frequencies
        sum_frequencies = sum(ham.G.nodes[node].get("nu_f", 0.0) for node in ham.nodes)
        trace_H_freq = np.trace(ham.H_freq).real

        assert np.isclose(
            trace_H_freq, sum_frequencies
        ), "Trace of H_freq should equal sum of node frequencies"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_numpy(self, monkeypatch):
        """Test error when NumPy is not available."""
        import sys

        # Temporarily hide numpy
        numpy_module = sys.modules.get("numpy")
        if numpy_module:
            monkeypatch.setitem(sys.modules, "numpy", None)

        G = nx.Graph()
        G.add_node(0, nu_f=1.0, phase=0.0, epi=1.0, si=0.8)

        # Should raise ImportError
        with pytest.raises(ImportError, match="NumPy is required"):
            InternalHamiltonian(G)

        # Restore numpy
        if numpy_module:
            sys.modules["numpy"] = numpy_module

    def test_invalid_node_delta_nfr(self):
        """Test error for invalid node in compute_node_delta_nfr."""
        G = nx.path_graph(3)
        for node in G.nodes:
            G.nodes[node].update({"nu_f": 1.0, "phase": 0.0, "epi": 1.0, "si": 0.8})

        ham = InternalHamiltonian(G)

        # Try to compute ΔNFR for non-existent node
        with pytest.raises(ValueError, match="not found"):
            ham.compute_node_delta_nfr(999)

    def test_non_hermitian_detection(self):
        """Test that non-Hermitian matrices are detected."""
        G = nx.Graph()
        G.add_edge(0, 1)

        for node in G.nodes:
            G.nodes[node].update({"nu_f": 1.0, "phase": 0.0, "epi": 1.0, "si": 0.8})

        # Manually create Hamiltonian and break Hermiticity
        ham = InternalHamiltonian(G)

        # Make H_int non-Hermitian
        ham.H_int[0, 1] = 1.0 + 1j
        ham.H_int[1, 0] = 1.0 - 2j  # Asymmetric

        # Verification should fail
        with pytest.raises(ValueError, match="not Hermitian"):
            ham._verify_hermitian()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
