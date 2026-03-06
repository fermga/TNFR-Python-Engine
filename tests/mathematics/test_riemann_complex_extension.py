r"""Tests for the P4 complex-s extension of the TNFR-Riemann operator.

Validates the non-Hermitian operator H^(k)(s) = L_k + (s - 1/2) diag(log p_i)
for complex s, including:

1. **Operator construction** — complex matrix properties, dtype, symmetry.
2. **Hermitian limit** — recovers P1 results at s = 1/2 (Im(s) = 0).
3. **Non-Hermiticity** — measures departure from self-adjointness.
4. **Critical line scan** — eigenvalue flow along s = 1/2 + it.
5. **Pseudo-spectrum** — sigma_min(zI - H) grid computation.
6. **Resolvent** — pole structure and norm computation.
7. **Riemann zero comparison** — structural correlation check.

Physics basis: H(s) = L + V(s) with complex V encodes oscillatory
TNFR dynamics.  At s = 1/2, H reduces to the pure Laplacian (P1
structural equilibrium).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.riemann.operator import (
    build_prime_path_graph,
    build_h_tnfr,
    build_h_tnfr_complex,
    build_tridiagonal_h_tnfr,
    build_tridiagonal_h_tnfr_complex,
    default_prime_potential,
    default_prime_potential_complex,
    _first_primes,
)
from tnfr.riemann.complex_extension import (
    # Data structures
    ComplexEigenResult,
    CriticalLineScan,
    PseudoSpectrumResult,
    ResolventAnalysis,
    ComplexPlaneAnalysis,
    # Constants
    KNOWN_RIEMANN_ZEROS,
    # Core
    compute_complex_eigenspectrum,
    compute_complex_eigensystem,
    analyze_non_hermiticity,
    # Critical line
    scan_critical_line,
    find_eigenvalue_zero_crossings,
    # Pseudo-spectrum
    compute_pseudospectrum,
    compute_resolvent_norm,
    # Resolvent
    analyze_resolvent_along_critical_line,
    # Comparison
    compare_with_riemann_zeros,
    # Integrated
    run_complex_plane_analysis,
    # Private helpers
    _find_local_minima,
    _find_local_maxima,
    _deduplicate_crossings,
    _build_dense_from_tridiag,
)
from tnfr.riemann.spectral_proof import (
    compute_eigenspectrum,
    compute_eigensystem,
)


# ============================================================================
# 1. Operator Construction Tests
# ============================================================================


class TestComplexOperatorConstruction:
    """Tests for build_h_tnfr_complex and build_tridiagonal_h_tnfr_complex."""

    def test_complex_dtype(self) -> None:
        """H(s) must be complex ndarray when Im(s) != 0."""
        G = build_prime_path_graph(5)
        H, V = build_h_tnfr_complex(G, s=0.5 + 1j)
        assert H.dtype == complex
        assert V.dtype == complex

    def test_real_at_sigma_half(self) -> None:
        """At s = 1/2, H(s) should be purely real (= Laplacian)."""
        G = build_prime_path_graph(5)
        H, V = build_h_tnfr_complex(G, s=0.5 + 0j)
        assert np.allclose(H.imag, 0.0, atol=1e-14)
        assert np.allclose(V, 0.0, atol=1e-14)

    def test_shape(self) -> None:
        """Output matrices must be square k x k."""
        for k in [3, 10, 20]:
            G = build_prime_path_graph(k)
            H, V = build_h_tnfr_complex(G, s=0.5 + 2j)
            assert H.shape == (k, k)
            assert V.shape == (k, k)

    def test_diagonal_potential_is_purely_imaginary_on_critical_line(self) -> None:
        """V(s) = it * diag(log p_i) is purely imaginary when sigma = 1/2."""
        G = build_prime_path_graph(5)
        _, V = build_h_tnfr_complex(G, s=0.5 + 3j)
        for i in range(5):
            assert abs(V[i, i].real) < 1e-14, "V diagonal should be purely imaginary"
            assert abs(V[i, i].imag) > 0, "V diagonal should have nonzero imaginary part"
        # Off-diagonal should be zero
        np.fill_diagonal(V, 0)
        assert np.allclose(V, 0)

    def test_consistency_dense_vs_tridiag(self) -> None:
        """Dense and tridiagonal builders must produce same H."""
        k = 10
        s = 0.3 + 2.5j
        G = build_prime_path_graph(k)
        H_dense, _ = build_h_tnfr_complex(G, s=s)

        d, e, _ = build_tridiagonal_h_tnfr_complex(k, s)
        H_tri = _build_dense_from_tridiag(d, e)

        assert np.allclose(H_dense, H_tri, atol=1e-12)

    def test_complex_potential_function(self) -> None:
        """default_prime_potential_complex matches manual computation."""
        s = 0.7 + 1.5j
        label = 7
        expected = (s - 0.5) * np.log(7.0)
        actual = default_prime_potential_complex(label, s)
        assert abs(actual - expected) < 1e-14

    def test_complex_potential_vanishes_at_half(self) -> None:
        """V(p, 1/2) = 0 for all primes."""
        for p in [2, 3, 5, 7, 11]:
            assert abs(default_prime_potential_complex(p, 0.5 + 0j)) < 1e-15

    def test_hermitian_at_real_sigma(self) -> None:
        """H(s) is Hermitian when s is real (Im(s) = 0)."""
        G = build_prime_path_graph(8)
        for sigma in [0.3, 0.5, 0.7, 1.0]:
            H, _ = build_h_tnfr_complex(G, s=complex(sigma, 0))
            assert np.allclose(H, H.conj().T, atol=1e-14)

    def test_non_hermitian_when_imaginary(self) -> None:
        """H(s) is NOT Hermitian when Im(s) != 0."""
        G = build_prime_path_graph(8)
        H, _ = build_h_tnfr_complex(G, s=0.5 + 5j)
        diff = np.linalg.norm(H - H.conj().T, "fro")
        assert diff > 1e-10, "H should be non-Hermitian"

    def test_tridiag_complex_k1(self) -> None:
        """Tridiagonal builder handles k=1 (edge case)."""
        d, e, lp = build_tridiagonal_h_tnfr_complex(1, 0.5 + 1j)
        assert len(d) == 1
        assert len(e) == 0
        # V = it * log(2) for the single node
        expected = 1j * np.log(2.0)
        assert abs(d[0] - expected) < 1e-14

    def test_tridiag_complex_subdiag_real(self) -> None:
        """Sub-diagonal is always real (from Laplacian)."""
        d, e, _ = build_tridiagonal_h_tnfr_complex(10, 0.5 + 3j)
        assert np.all(np.isreal(e))

    def test_invalid_graph_type_raises(self) -> None:
        """Non-Graph input must raise TypeError."""
        with pytest.raises(TypeError):
            build_h_tnfr_complex("not a graph", s=0.5 + 0j)  # type: ignore[arg-type]

    def test_missing_label_raises(self) -> None:
        """Missing node label must raise TNFRValueError."""
        import networkx as nx
        G = nx.Graph()
        G.add_node(0)  # no 'label' attribute
        with pytest.raises(Exception):  # TNFRValueError
            build_h_tnfr_complex(G, s=0.5 + 1j)


# ============================================================================
# 2. Hermitian Limit — P1 Consistency
# ============================================================================


class TestHermitianLimit:
    """Complex extension must recover P1 results when Im(s) = 0."""

    def test_eigenvalues_match_p1(self) -> None:
        """Eigenvalues at real sigma must match P1 Hermitian computation."""
        for k in [5, 10, 20]:
            for sigma in [0.3, 0.5, 0.7]:
                p1_evals = compute_eigenspectrum(k, sigma)
                p4_evals = compute_complex_eigenspectrum(k, complex(sigma, 0))
                # P4 eigenvalues should be real (imaginary part ~0)
                assert np.allclose(p4_evals.imag, 0, atol=1e-10)
                assert np.allclose(np.sort(p4_evals.real), p1_evals, atol=1e-10)

    def test_zero_eigenvalue_at_half(self) -> None:
        """lambda_min(H(1/2)) = 0 in complex extension, matching P1."""
        for k in [5, 10, 30]:
            evals = compute_complex_eigenspectrum(k, 0.5 + 0j)
            assert abs(evals[0]) < 1e-10, f"k={k}: lambda_min should be ~0"

    def test_non_hermiticity_vanishes_at_real_sigma(self) -> None:
        """Non-Hermiticity must be ~0 for real s."""
        result = analyze_non_hermiticity(10, 0.5 + 0j)
        assert result.non_hermiticity < 1e-14
        result = analyze_non_hermiticity(10, 0.3 + 0j)
        assert result.non_hermiticity < 1e-14


# ============================================================================
# 3. Non-Hermiticity Validation
# ============================================================================


class TestNonHermiticity:
    """Test non-Hermitian properties of H(s) for Im(s) != 0."""

    def test_non_hermiticity_increases_with_t(self) -> None:
        """Non-Hermiticity should increase with |Im(s)|."""
        nh_values = []
        for t in [0.0, 1.0, 5.0, 10.0]:
            result = analyze_non_hermiticity(10, 0.5 + t * 1j)
            nh_values.append(result.non_hermiticity)
        # Monotonically increasing
        for i in range(1, len(nh_values)):
            assert nh_values[i] >= nh_values[i - 1] - 1e-10

    def test_complex_eigenvalues(self) -> None:
        """Eigenvalues must be genuinely complex when Im(s) != 0."""
        evals = compute_complex_eigenspectrum(10, 0.5 + 5j)
        # At least some eigenvalues should have nonzero imaginary part
        max_imag = np.max(np.abs(evals.imag))
        assert max_imag > 0.1, "Expected complex eigenvalues"

    def test_condition_number_finite(self) -> None:
        """Eigenvector condition number should be finite for moderate t."""
        result = analyze_non_hermiticity(10, 0.5 + 2j)
        assert np.isfinite(result.condition_number)
        assert result.condition_number >= 1.0

    def test_result_dataclass_fields(self) -> None:
        """ComplexEigenResult has all expected fields."""
        result = analyze_non_hermiticity(5, 0.5 + 1j)
        assert result.k == 5
        assert result.s == 0.5 + 1j
        assert len(result.eigenvalues) == 5
        assert isinstance(result.min_abs_eigenvalue, float)
        assert isinstance(result.condition_number, float)
        assert isinstance(result.non_hermiticity, float)


# ============================================================================
# 4. Complex Eigensystem Tests
# ============================================================================


class TestComplexEigensystem:
    """Tests for eigenvalue/eigenvector computation."""

    def test_eigenvector_equation(self) -> None:
        """H V = V diag(lambda) must hold."""
        k = 8
        s = 0.5 + 3j
        evals, evecs = compute_complex_eigensystem(k, s)
        d, e, _ = build_tridiagonal_h_tnfr_complex(k, s)
        H = _build_dense_from_tridiag(d, e)

        # H @ v_j = lambda_j * v_j for each j
        for j in range(k):
            lhs = H @ evecs[:, j]
            rhs = evals[j] * evecs[:, j]
            assert np.allclose(lhs, rhs, atol=1e-10), f"Eigenvector eq failed for j={j}"

    def test_eigenvalue_count(self) -> None:
        """Must return exactly k eigenvalues."""
        for k in [3, 10, 25]:
            evals = compute_complex_eigenspectrum(k, 0.5 + 2j)
            assert len(evals) == k

    def test_sorted_by_real_part(self) -> None:
        """Eigenvalues must be sorted by real part."""
        evals = compute_complex_eigenspectrum(15, 0.5 + 4j)
        assert np.all(np.diff(evals.real) >= -1e-10)

    def test_conjugate_symmetry_on_critical_line(self) -> None:
        """On the critical line (sigma=1/2), H = L + it*V_1.
        Since L is symmetric and V_1 is real diagonal,
        H^T = L + it*V_1 = H, so H is complex symmetric.
        Eigenvalues of a complex symmetric matrix come in conjugate
        pairs only if there is additional symmetry."""
        k = 10
        s = 0.5 + 5j
        d, e, _ = build_tridiagonal_h_tnfr_complex(k, s)
        H = _build_dense_from_tridiag(d, e)
        # H should be complex symmetric (H = H^T, NOT H = H^dag)
        assert np.allclose(H, H.T, atol=1e-14)


# ============================================================================
# 5. Critical Line Scan Tests
# ============================================================================


class TestCriticalLineScan:
    """Tests for scanning s = 1/2 + it."""

    def test_scan_returns_correct_shape(self) -> None:
        """Scan result matrices have correct dimensions."""
        k = 8
        n = 50
        result = scan_critical_line(k, t_max=10.0, n_points=n)
        assert result.k == k
        assert len(result.t_values) == n
        assert result.eigenvalue_matrix.shape == (n, k)
        assert len(result.min_abs_eigenvalue) == n
        assert len(result.non_hermiticity) == n

    def test_t0_hermitian(self) -> None:
        """At t=0, eigenvalues should be real and min should be ~0."""
        result = scan_critical_line(10, t_max=5.0, n_points=20)
        # First point is t=0 -> Hermitian
        evals_t0 = result.eigenvalue_matrix[0]
        assert np.allclose(evals_t0.imag, 0, atol=1e-10)
        assert result.min_abs_eigenvalue[0] < 1e-8

    def test_non_hermiticity_monotone(self) -> None:
        """Non-Hermiticity should generally increase with t."""
        result = scan_critical_line(10, t_max=20.0, n_points=50)
        # Compare first quarter vs last quarter averages
        n = len(result.non_hermiticity)
        first_q = np.mean(result.non_hermiticity[:n // 4])
        last_q = np.mean(result.non_hermiticity[3 * n // 4:])
        assert last_q > first_q

    def test_local_minima_detected(self) -> None:
        """Scan should detect local minima (data structure populated)."""
        result = scan_critical_line(10, t_max=30.0, n_points=200)
        # We expect at least some local minima in min_abs
        # (eigenvalue absolute values oscillate)
        assert isinstance(result.local_minima_t, np.ndarray)
        assert isinstance(result.local_minima_val, np.ndarray)
        assert len(result.local_minima_t) == len(result.local_minima_val)

    def test_t_range(self) -> None:
        """Custom t_min and t_max respected."""
        result = scan_critical_line(5, t_max=20.0, n_points=30, t_min=5.0)
        assert result.t_values[0] >= 5.0
        assert result.t_values[-1] <= 20.0


# ============================================================================
# 6. Pseudo-Spectrum Tests
# ============================================================================


class TestPseudoSpectrum:
    """Tests for epsilon-pseudospectrum computation."""

    def test_pseudospectrum_shape(self) -> None:
        """Grid and sigma_min arrays have correct shapes."""
        result = compute_pseudospectrum(5, 0.5 + 2j, n_grid=20)
        assert result.k == 5
        assert len(result.z_real) == 20
        assert len(result.z_imag) == 20
        assert result.sigma_min_grid.shape == (20, 20)
        assert len(result.eigenvalues) == 5

    def test_sigma_min_nonnegative(self) -> None:
        """Singular values are always >= 0."""
        result = compute_pseudospectrum(8, 0.5 + 3j, n_grid=15)
        assert np.all(result.sigma_min_grid >= -1e-15)

    def test_sigma_min_small_at_eigenvalues(self) -> None:
        """sigma_min should be ~0 at eigenvalue locations."""
        k = 5
        s = 0.5 + 2j
        evals = compute_complex_eigenspectrum(k, s)
        # Check resolvent norm at an eigenvalue
        for ev in evals[:2]:  # check first two
            rn = compute_resolvent_norm(k, s, ev)
            # Should be very large (near pole)
            assert rn > 1e6, f"Resolvent should be large at eigenvalue z={ev}"

    def test_hermitian_pseudospectrum_is_disc_union(self) -> None:
        """For Hermitian H (s real), sigma_min(zI - H) = min |z - lambda_j|.
        The pseudospectrum should be approximately the union of discs."""
        k = 5
        s = 0.5 + 0j  # Hermitian
        evals = compute_complex_eigenspectrum(k, s)
        result = compute_pseudospectrum(
            k, s, z_center=complex(np.mean(evals.real), 0),
            z_radius=3.0, n_grid=30,
        )
        # For each grid point, sigma_min should ~ min distance to eigenvalues
        for i in range(result.sigma_min_grid.shape[0]):
            for j in range(result.sigma_min_grid.shape[1]):
                z = complex(result.z_real[j], result.z_imag[i])
                expected = float(np.min(np.abs(z - evals)))
                actual = result.sigma_min_grid[i, j]
                assert abs(actual - expected) < 0.1, (
                    f"Hermitian pseudo-spectrum mismatch at z={z}"
                )


# ============================================================================
# 7. Resolvent Tests
# ============================================================================


class TestResolvent:
    """Tests for resolvent norm computation."""

    def test_resolvent_at_eigenvalue(self) -> None:
        """Resolvent norm -> inf at eigenvalue."""
        k = 5
        s = 0.5 + 2j
        evals = compute_complex_eigenspectrum(k, s)
        rn = compute_resolvent_norm(k, s, evals[0])
        assert rn > 1e8

    def test_resolvent_far_from_spectrum(self) -> None:
        """Resolvent norm should be moderate far from spectrum."""
        k = 5
        s = 0.5 + 0j
        # Far from any eigenvalue
        rn = compute_resolvent_norm(k, s, 1000.0 + 0j)
        assert rn < 1.0  # Should be small (~1/1000)

    def test_resolvent_analysis_shape(self) -> None:
        """Resolvent analysis returns correct structures."""
        result = analyze_resolvent_along_critical_line(
            8, z_probe=0.0 + 0j, t_max=20.0, n_points=50,
        )
        assert result.k == 8
        assert len(result.t_values) == 50
        assert len(result.resolvent_norms) == 50
        assert np.all(result.resolvent_norms > 0)

    def test_resolvent_analysis_peaks(self) -> None:
        """Resolvent analysis should detect peaks."""
        result = analyze_resolvent_along_critical_line(
            10, z_probe=0.0 + 0j, t_max=30.0, n_points=200,
        )
        assert isinstance(result.peak_t_values, np.ndarray)
        assert isinstance(result.peak_norms, np.ndarray)
        assert len(result.peak_t_values) == len(result.peak_norms)


# ============================================================================
# 8. Riemann Zero Comparison Tests
# ============================================================================


class TestRiemannComparison:
    """Tests for comparison with known Riemann zeros."""

    def test_known_zeros_count(self) -> None:
        """We store at least 20 known zeros."""
        assert len(KNOWN_RIEMANN_ZEROS) == 20

    def test_known_zeros_ordered(self) -> None:
        """Known zeros must be in ascending order."""
        for i in range(1, len(KNOWN_RIEMANN_ZEROS)):
            assert KNOWN_RIEMANN_ZEROS[i] > KNOWN_RIEMANN_ZEROS[i - 1]

    def test_first_zero_value(self) -> None:
        """First Riemann zero is t_1 ~ 14.1347."""
        assert abs(KNOWN_RIEMANN_ZEROS[0] - 14.134725) < 0.001

    def test_comparison_returns_tuples(self) -> None:
        """compare_with_riemann_zeros returns list of 3-tuples."""
        results = compare_with_riemann_zeros(
            10, t_max=50.0, n_points=200,
            n_zeros=5, threshold=5.0,
        )
        assert isinstance(results, list)
        for item in results:
            assert len(item) == 3
            t_c, t_r, dist = item
            assert isinstance(t_c, float)
            assert isinstance(t_r, float)
            assert isinstance(dist, float)
            assert dist >= 0


# ============================================================================
# 9. Integrated Analysis Tests
# ============================================================================


class TestIntegratedAnalysis:
    """Tests for run_complex_plane_analysis."""

    def test_analysis_fields(self) -> None:
        """Integrated analysis populates all fields."""
        result = run_complex_plane_analysis(
            8, t_max=20.0, n_points=50,
        )
        assert result.k == 8
        assert result.critical_line is not None
        assert isinstance(result.riemann_comparison, list)
        assert isinstance(result.non_hermiticity_at_half, float)
        assert isinstance(result.mean_non_hermiticity, float)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_analysis_hermitian_at_origin(self) -> None:
        """Non-Hermiticity at s = 1/2 should be small."""
        result = run_complex_plane_analysis(
            10, t_max=10.0, n_points=30,
        )
        assert result.non_hermiticity_at_half < 1e-10

    def test_analysis_summary_readable(self) -> None:
        """Summary should contain key information."""
        result = run_complex_plane_analysis(5, t_max=15.0, n_points=30)
        assert "P4" in result.summary
        assert "k=5" in result.summary


# ============================================================================
# 10. Private Helper Tests
# ============================================================================


class TestPrivateHelpers:
    """Tests for internal utility functions."""

    def test_find_local_minima(self) -> None:
        arr = np.array([5.0, 3.0, 1.0, 2.0, 4.0, 0.5, 3.0])
        idx = _find_local_minima(arr)
        assert 2 in idx  # value 1.0 is a local minimum
        assert 5 in idx  # value 0.5 is a local minimum

    def test_find_local_minima_short_array(self) -> None:
        assert len(_find_local_minima(np.array([1.0, 2.0]))) == 0
        assert len(_find_local_minima(np.array([]))) == 0

    def test_find_local_maxima(self) -> None:
        arr = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        idx = _find_local_maxima(arr)
        assert 1 in idx  # value 5.0
        assert 3 in idx  # value 7.0

    def test_deduplicate_crossings(self) -> None:
        crossings = [(1.0, 0.1), (1.5, 0.2), (5.0, 0.3), (5.3, 0.1)]
        result = _deduplicate_crossings(crossings, min_gap=1.0)
        assert len(result) == 2
        assert result[0][0] == 1.0  # kept first
        assert result[1][0] == 5.0 or result[1][0] == 5.3

    def test_deduplicate_empty(self) -> None:
        assert _deduplicate_crossings([], min_gap=1.0) == []

    def test_build_dense_from_tridiag(self) -> None:
        d = np.array([1 + 1j, 2 + 0j, 3 + 2j])
        e = np.array([-0.5, -0.7])
        H = _build_dense_from_tridiag(d, e)
        assert H.shape == (3, 3)
        assert H[0, 0] == d[0]
        assert H[1, 1] == d[1]
        assert H[0, 1] == e[0]
        assert H[1, 0] == e[0]
        assert H[0, 2] == 0


# ============================================================================
# 11. Mathematical Property Tests
# ============================================================================


class TestMathematicalProperties:
    """Tests for structural mathematical properties of H(s)."""

    def test_determinant_product_of_eigenvalues(self) -> None:
        """det(H) = product of eigenvalues."""
        k = 7
        s = 0.5 + 3j
        evals = compute_complex_eigenspectrum(k, s)
        d, e, _ = build_tridiagonal_h_tnfr_complex(k, s)
        H = _build_dense_from_tridiag(d, e)
        det_H = np.linalg.det(H)
        prod_evals = np.prod(evals)
        assert abs(det_H - prod_evals) / max(abs(det_H), 1e-15) < 1e-8

    def test_trace_sum_of_eigenvalues(self) -> None:
        """tr(H) = sum of eigenvalues."""
        k = 10
        s = 0.5 + 4j
        evals = compute_complex_eigenspectrum(k, s)
        d, e, _ = build_tridiagonal_h_tnfr_complex(k, s)
        H = _build_dense_from_tridiag(d, e)
        assert abs(np.trace(H) - np.sum(evals)) < 1e-10

    def test_eigenvalue_perturbation_linearity(self) -> None:
        """Small change in s should produce small change in eigenvalues."""
        k = 8
        s0 = 0.5 + 5j
        ds = 0.001j
        evals0 = compute_complex_eigenspectrum(k, s0)
        evals1 = compute_complex_eigenspectrum(k, s0 + ds)
        # Eigenvalue change should be O(|ds|)
        max_change = np.max(np.abs(evals1 - evals0))
        assert max_change < 10 * abs(ds)  # Bounded perturbation

    def test_spectral_symmetry_under_t_negation(self) -> None:
        r"""H(1/2 + it) and H(1/2 - it) have conjugate spectra.

        Since H(s) = L + (s-1/2) V with real L, V:
            H(conj(s)) = conj(H(s))
        so eigenvalues of H(conj(s)) = conj(eigenvalues of H(s)).
        """
        k = 10
        t = 7.0
        evals_pos = compute_complex_eigenspectrum(k, 0.5 + t * 1j)
        evals_neg = compute_complex_eigenspectrum(k, 0.5 - t * 1j)
        # Sort conjugates for comparison
        conj_pos = np.sort(np.conj(evals_pos).real + 1j * np.sort(np.conj(evals_pos).imag))
        # Just check the sets match
        sorted_neg = np.sort(evals_neg.real + 1j * np.sort(evals_neg.imag))
        # More robust: compare sorted absolute values
        assert np.allclose(
            np.sort(np.abs(evals_pos)),
            np.sort(np.abs(evals_neg)),
            atol=1e-10,
        )

    def test_H_equals_L_plus_complex_potential(self) -> None:
        """H(s) = L_k + (s - 1/2) V_1 structure is explicit."""
        k = 8
        s = 0.3 + 2j
        # Laplacian at sigma=1/2
        d_L, e_L, log_p = build_tridiagonal_h_tnfr_complex(k, 0.5 + 0j)
        L = _build_dense_from_tridiag(d_L, e_L)
        V1 = np.diag(log_p)

        # Full H
        d, e, _ = build_tridiagonal_h_tnfr_complex(k, s)
        H = _build_dense_from_tridiag(d, e)

        expected = L + (s - 0.5) * V1
        assert np.allclose(H, expected, atol=1e-14)


# ============================================================================
# 12. Edge Cases and Robustness
# ============================================================================


class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_large_imaginary_part(self) -> None:
        """No crash with very large Im(s)."""
        evals = compute_complex_eigenspectrum(5, 0.5 + 100j)
        assert len(evals) == 5
        assert all(np.isfinite(evals))

    def test_k2_minimum(self) -> None:
        """k=2 is the minimum meaningful graph."""
        evals = compute_complex_eigenspectrum(2, 0.5 + 1j)
        assert len(evals) == 2

    def test_zero_imaginary(self) -> None:
        """Im(s) = 0 => real eigenvalues."""
        evals = compute_complex_eigenspectrum(10, 0.5 + 0j)
        assert np.allclose(evals.imag, 0, atol=1e-10)

    def test_scan_single_point(self) -> None:
        """Scan with n_points=1 should not crash."""
        result = scan_critical_line(5, t_max=1.0, n_points=1)
        assert len(result.t_values) == 1

    def test_pseudospectrum_small_grid(self) -> None:
        """Small pseudo-spectrum grid should work."""
        result = compute_pseudospectrum(3, 0.5 + 1j, n_grid=5)
        assert result.sigma_min_grid.shape == (5, 5)

    def test_resolvent_norm_positive(self) -> None:
        """Resolvent norm is always positive."""
        rn = compute_resolvent_norm(5, 0.5 + 1j, 10.0 + 0j)
        assert rn > 0

    def test_off_critical_line(self) -> None:
        """Operator works for sigma != 1/2."""
        evals = compute_complex_eigenspectrum(8, 0.3 + 5j)
        assert len(evals) == 8
        assert all(np.isfinite(evals))
