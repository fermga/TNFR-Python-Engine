"""Tests for U6 relaxation time estimation with Liouvillian integration.

Validates the dual-path τ_relax estimation:
1. Liouvillian slow-mode path: τ = 1/|Re(λ_slow)|
2. Spectral topological fallback: τ = (k_top/νf)·k_op·ln(1/ε)

Also tests the full Liouvillian spectrum computation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import networkx as nx

from tnfr.mathematics.generators import build_lindblad_delta_nfr
from tnfr.mathematics.liouville import (
    compute_liouvillian_spectrum,
    get_slow_relaxation_mode,
    store_liouvillian_spectrum,
    get_liouvillian_spectrum,
)
from tnfr.operators.metrics_u6 import measure_tau_relax_observed


def test_liouvillian_spectrum_computation():
    """Test basic Liouvillian spectrum computation."""
    print("Testing Liouvillian spectrum computation...")

    # Simple 2-level system
    H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L1 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # decay operator

    liouv = build_lindblad_delta_nfr(
        hamiltonian=H,
        collapse_operators=[L1],
        nu_f=1.0,
        ensure_contractive=True,
    )

    eigs = compute_liouvillian_spectrum(liouv, sort=True, validate_contractivity=True)

    assert len(eigs) == 4, f"Expected 4 eigenvalues for 2-level system, got {len(eigs)}"
    assert np.all(eigs.real <= 1e-9), f"Eigenvalues must be non-positive: {eigs}"

    # Check for steady state (λ ≈ 0)
    steady_state_eig = eigs[np.argmax(eigs.real)]
    assert abs(steady_state_eig.real) < 1e-9, f"Expected steady state near zero: {steady_state_eig}"

    print(f"  ✓ Computed {len(eigs)} eigenvalues")
    print(f"  ✓ All eigenvalues satisfy contractivity (Re(λ) ≤ 0)")
    print(f"  ✓ Steady state eigenvalue: {steady_state_eig:.6f}")


def test_slow_relaxation_mode_extraction():
    """Test extraction of slowest relaxation eigenvalue."""
    print("\nTesting slow relaxation mode extraction...")

    # Artificial spectrum with known slow mode
    eigs = np.array([0.0 + 0j, -0.05 + 0.02j, -1.2 - 0.1j, -5.0 + 0j])

    slow = get_slow_relaxation_mode(eigs, tolerance=1e-12)

    assert slow is not None, "Should find slow mode in test spectrum"
    expected_slow_real = -0.05  # Least negative real part
    assert (
        abs(slow.real - expected_slow_real) < 1e-10
    ), f"Expected slow mode Re(λ) ≈ {expected_slow_real}, got {slow.real}"

    tau = 1.0 / abs(slow.real)
    print(f"  ✓ Extracted slow mode: λ_slow = {slow}")
    print(f"  ✓ Relaxation time: τ = {tau:.2f}")


def test_graph_metadata_storage():
    """Test storing and retrieving Liouvillian spectrum in graph."""
    print("\nTesting graph metadata storage...")

    G = nx.Graph()
    eigs = np.array([0.0 + 0j, -0.1 + 0.05j, -2.3 - 0.1j])

    store_liouvillian_spectrum(G, eigs)
    assert "LIOUVILLIAN_EIGS" in G.graph, "Spectrum not stored in graph metadata"

    retrieved = get_liouvillian_spectrum(G)
    assert retrieved is not None, "Failed to retrieve stored spectrum"
    assert len(retrieved) == len(eigs), "Retrieved spectrum has wrong length"
    assert np.allclose(retrieved, eigs), "Retrieved spectrum does not match stored"

    print(f"  ✓ Stored {len(eigs)} eigenvalues in graph metadata")
    print(f"  ✓ Retrieved spectrum matches stored values")


def test_u6_tau_relax_with_liouvillian():
    """Test U6 measure_tau_relax_observed with Liouvillian spectrum."""
    print("\nTesting U6 τ_relax with Liouvillian integration...")

    G = nx.Graph()
    G.add_node("n0")
    G.nodes["n0"]["DELTA_NFR"] = 0.3
    G.nodes["n0"]["nu_f"] = 1.0

    # Build and store Liouvillian spectrum
    H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L1 = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=complex)
    liouv = build_lindblad_delta_nfr(hamiltonian=H, collapse_operators=[L1])
    eigs = compute_liouvillian_spectrum(liouv)
    store_liouvillian_spectrum(G, eigs)

    # Measure relaxation time
    result = measure_tau_relax_observed(G, "n0")

    assert result["metric_type"] == "u6_relaxation_time"
    assert "estimated_tau_relax" in result
    assert "estimated_tau_relax_spectral" in result
    assert "estimated_tau_relax_liouvillian" in result
    assert "liouvillian_slow_mode_real" in result

    # Should have computed Liouvillian-based τ
    liouv_tau = result["estimated_tau_relax_liouvillian"]
    assert liouv_tau is not None, "Liouvillian τ should be computed when spectrum available"
    assert liouv_tau > 0, f"Relaxation time must be positive, got {liouv_tau}"

    # Final estimate should prefer Liouvillian
    tau_final = result["estimated_tau_relax"]
    assert tau_final == liouv_tau, "Final τ should use Liouvillian when available"

    print(f"  ✓ Liouvillian τ_relax: {liouv_tau:.3f}")
    print(f"  ✓ Spectral fallback τ: {result['estimated_tau_relax_spectral']:.3f}")
    print(f"  ✓ Final τ estimate: {tau_final:.3f} (Liouvillian preferred)")


def test_u6_tau_relax_fallback():
    """Test U6 fallback to spectral estimate when Liouvillian unavailable."""
    print("\nTesting U6 τ_relax fallback to spectral estimate...")

    G = nx.Graph()
    G.add_node("n0")
    G.nodes["n0"]["DELTA_NFR"] = 0.2
    G.nodes["n0"]["nu_f"] = 0.8
    # No Liouvillian spectrum stored

    result = measure_tau_relax_observed(G, "n0")

    assert (
        result["estimated_tau_relax_liouvillian"] is None
    ), "Liouvillian τ should be None when spectrum unavailable"

    spectral_tau = result["estimated_tau_relax_spectral"]
    assert spectral_tau is not None, "Spectral fallback should always compute"
    assert spectral_tau > 0, "Spectral τ must be positive"

    tau_final = result["estimated_tau_relax"]
    assert tau_final == spectral_tau, "Should fall back to spectral when Liouvillian missing"

    print(f"  ✓ Liouvillian τ: None (as expected)")
    print(f"  ✓ Spectral fallback τ: {spectral_tau:.3f}")
    print(f"  ✓ Final τ estimate: {tau_final:.3f} (spectral fallback)")


def test_contractivity_validation():
    """Test that non-contractive Liouvillian raises error."""
    print("\nTesting contractivity validation...")

    # Construct artificial non-contractive Liouvillian
    bad_liouv = np.diag([0.5 + 0j, -1.0 + 0j, -2.0 + 0j, -3.0 + 0j])

    try:
        eigs = compute_liouvillian_spectrum(bad_liouv, validate_contractivity=True)
        assert False, "Should have raised ValueError for non-contractive Liouvillian"
    except ValueError as e:
        assert "contractivity" in str(e).lower(), f"Expected contractivity error, got: {e}"
        print(f"  ✓ Correctly rejected non-contractive spectrum")

    # Without validation, should compute
    eigs_unchecked = compute_liouvillian_spectrum(bad_liouv, validate_contractivity=False)
    assert len(eigs_unchecked) == 4
    print(f"  ✓ Can compute spectrum without validation")


def test_relaxation_time_physical_bounds():
    """Test that computed relaxation times satisfy physical bounds."""
    print("\nTesting relaxation time physical bounds...")

    G = nx.Graph()
    G.add_node("n0")
    G.nodes["n0"]["DELTA_NFR"] = 0.5
    G.nodes["n0"]["nu_f"] = 1.5

    # Build realistic Liouvillian
    H = np.array([[2.0, 0.1], [0.1, -2.0]], dtype=complex)
    L1 = np.array([[0.0, 0.8], [0.0, 0.0]], dtype=complex)
    L2 = np.array([[0.0, 0.0], [0.3, 0.0]], dtype=complex)
    liouv = build_lindblad_delta_nfr(hamiltonian=H, collapse_operators=[L1, L2], nu_f=1.2)
    eigs = compute_liouvillian_spectrum(liouv)
    store_liouvillian_spectrum(G, eigs)

    result = measure_tau_relax_observed(G, "n0")

    tau = result["estimated_tau_relax"]
    vf = result["vf"]

    # Physical bounds:
    # 1. τ > 0 (positive relaxation time)
    # 2. τ should be O(1/νf) typically
    # 3. τ should not be excessively large (< 1000 for test systems)

    assert tau > 0, f"Relaxation time must be positive: {tau}"
    assert tau < 1000, f"Relaxation time unreasonably large: {tau}"

    # Rough order of magnitude check
    expected_order = 1.0 / vf
    assert (
        0.1 * expected_order < tau < 100 * expected_order
    ), f"Relaxation time {tau:.2f} outside expected range ~{expected_order:.2f}"

    print(f"  ✓ Relaxation time τ = {tau:.3f} is positive")
    print(f"  ✓ τ within reasonable bounds for νf = {vf:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("U6 Relaxation Time Estimation Tests")
    print("=" * 60)

    tests = [
        test_liouvillian_spectrum_computation,
        test_slow_relaxation_mode_extraction,
        test_graph_metadata_storage,
        test_u6_tau_relax_with_liouvillian,
        test_u6_tau_relax_fallback,
        test_contractivity_validation,
        test_relaxation_time_physical_bounds,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_func.__name__} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
