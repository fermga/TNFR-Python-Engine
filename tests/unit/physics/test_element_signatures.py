"""Test element signature utilities in TNFR physics module."""
from __future__ import annotations

import pytest

from tnfr.physics.patterns import build_element_radial_pattern
from tnfr.physics.signatures import compute_element_signature, compute_au_like_signature


def test_compute_element_signature_basic():
    """Test basic element signature computation with expected keys."""
    G = build_element_radial_pattern(Z=6, seed=42)  # Carbon-like
    
    signature = compute_element_signature(G, apply_synthetic_step=False)
    
    # Verify expected keys
    expected_keys = {
        "xi_c", "mean_phase_gradient", "mean_phase_curvature_abs", "max_phase_curvature_abs",
        "phi_s_before", "phi_s_after", "phi_s_drift", "phase_gradient_ok",
        "curvature_hotspots_ok", "coherence_length_category", "signature_class"
    }
    assert set(signature.keys()) == expected_keys
    
    # Verify types and ranges
    assert isinstance(signature["xi_c"], float) and signature["xi_c"] >= 0.0
    assert isinstance(signature["mean_phase_gradient"], float) and signature["mean_phase_gradient"] >= 0.0
    assert isinstance(signature["max_phase_curvature_abs"], float) and signature["max_phase_curvature_abs"] >= 0.0
    assert isinstance(signature["phase_gradient_ok"], bool)
    assert isinstance(signature["curvature_hotspots_ok"], bool)
    assert signature["coherence_length_category"] in {"localized", "medium", "extended"}
    assert signature["signature_class"] in {"stable", "marginal", "unstable"}


def test_compute_element_signature_with_synthetic_step():
    """Test signature with synthetic step for drift computation."""
    G = build_element_radial_pattern(Z=1, seed=42)  # Hydrogen-like
    
    signature = compute_element_signature(G, apply_synthetic_step=True)
    
    # With synthetic step, drift should be computed
    assert isinstance(signature["phi_s_drift"], float)
    assert signature["phi_s_drift"] >= 0.0
    
    # Phi_s before and after should be different (or at least computed)
    assert isinstance(signature["phi_s_before"], float)
    assert isinstance(signature["phi_s_after"], float)


def test_compute_au_like_signature():
    """Test Au-specific signature computation."""
    G = build_element_radial_pattern(Z=79, seed=42)  # Au-like
    
    signature = compute_au_like_signature(G)
    
    # Should have all element signature keys plus is_au_like
    expected_keys = {
        "xi_c", "mean_phase_gradient", "mean_phase_curvature_abs", "max_phase_curvature_abs",
        "phi_s_before", "phi_s_after", "phi_s_drift", "phase_gradient_ok",
        "curvature_hotspots_ok", "coherence_length_category", "signature_class", "is_au_like"
    }
    assert set(signature.keys()) == expected_keys
    
    # is_au_like should be boolean
    assert isinstance(signature["is_au_like"], bool)


def test_signature_reproducibility():
    """Test that same seed produces same signature."""
    G1 = build_element_radial_pattern(Z=8, seed=123)  # Oxygen-like
    G2 = build_element_radial_pattern(Z=8, seed=123)  # Same
    
    sig1 = compute_element_signature(G1, apply_synthetic_step=False)
    sig2 = compute_element_signature(G2, apply_synthetic_step=False)
    
    # Key metrics should be identical (floating point precision)
    assert abs(sig1["xi_c"] - sig2["xi_c"]) < 1e-10
    assert abs(sig1["mean_phase_gradient"] - sig2["mean_phase_gradient"]) < 1e-10
    assert sig1["signature_class"] == sig2["signature_class"]


def test_different_elements_different_signatures():
    """Test that different Z values produce different signatures."""
    G_light = build_element_radial_pattern(Z=1, seed=42)  # Hydrogen
    G_heavy = build_element_radial_pattern(Z=79, seed=42)  # Gold
    
    sig_light = compute_element_signature(G_light, apply_synthetic_step=False)
    sig_heavy = compute_element_signature(G_heavy, apply_synthetic_step=False)
    
    # Should have different coherence lengths (heavy elements are larger)
    assert sig_heavy["xi_c"] != sig_light["xi_c"]
    
    # Network sizes should be different, so signatures should differ
    assert (
        sig_heavy["mean_phase_gradient"] != sig_light["mean_phase_gradient"]
        or sig_heavy["mean_phase_curvature_abs"] != sig_light["mean_phase_curvature_abs"]
    )


@pytest.mark.parametrize("Z", [1, 6, 8, 79])
def test_signature_consistency_across_elements(Z):
    """Test signature computation is consistent across different atomic numbers."""
    G = build_element_radial_pattern(Z=Z, seed=42)
    
    signature = compute_element_signature(G, apply_synthetic_step=True)
    
    # Basic sanity checks
    assert signature["xi_c"] >= 0.0  # Allow 0.0 for cases where fit fails
    assert signature["mean_phase_gradient"] >= 0.0
    assert signature["max_phase_curvature_abs"] >= signature["mean_phase_curvature_abs"]
    assert signature["phi_s_drift"] >= 0.0
    
    # Classification should be consistent with threshold checks
    if signature["phase_gradient_ok"] and signature["curvature_hotspots_ok"]:
        # If both main thresholds pass, should be at least marginal
        assert signature["signature_class"] in {"stable", "marginal"}