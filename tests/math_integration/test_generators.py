"""Integration tests for ΔNFR generator construction.

This module contains baseline validation against expected ring topology values.
All other operator generation tests have been consolidated into:
- tests/integration/test_unified_operator_validation.py (Hermitian, reproducibility, input validation)
- tests/integration/test_operator_generation_critical_paths.py (critical path coverage)

See:
- tests/README_TEST_OPTIMIZATION.md for usage guidelines
- tests/TEST_CONSOLIDATION_SUMMARY.md for detailed consolidation mapping
"""
from __future__ import annotations

import numpy as np

from tnfr.mathematics import build_delta_nfr


def _expected_ring_adjacency(dim: int) -> np.ndarray:
    """Compute expected ring adjacency matrix for baseline validation."""
    adjacency = np.zeros((dim, dim), dtype=float)
    if dim == 1:
        return adjacency

    indices = np.arange(dim)
    adjacency[indices, (indices + 1) % dim] = 1.0
    adjacency[(indices + 1) % dim, indices] = 1.0
    return adjacency


def test_build_delta_nfr_scaling_matches_ring_baselines():
    """Verify ΔNFR operator scaling matches expected ring topology baselines.
    
    This test provides specific baseline validation against expected ring topology 
    values that serve as a reference for operator generation correctness.
    """
    dim = 4
    nu_f = 1.5
    scale = 0.25

    expected_adjacency = _expected_ring_adjacency(dim) * (nu_f * scale)
    adjacency = build_delta_nfr(dim, topology="adjacency", nu_f=nu_f, scale=scale)
    assert np.allclose(adjacency, expected_adjacency)

    ring_laplacian = np.diag([2.0] * dim) - _expected_ring_adjacency(dim)
    expected_laplacian = ring_laplacian * (nu_f * scale)
    laplacian = build_delta_nfr(dim, topology="laplacian", nu_f=nu_f, scale=scale)
    assert np.allclose(laplacian, expected_laplacian)
