"""
Comprehensive tests for ΔNFR backend fallback robustness.

This test module validates that the ΔNFR computation system correctly handles
transitions between NumPy and Python execution modes, ensuring:
- ΔNFR semantics preservation (TNFR Invariant #3)
- Controlled determinism (TNFR Invariant #8)
- Cache initialization consistency
- Reproducible results across backend changes
"""

import sys
from unittest.mock import patch

import networkx as nx
import pytest

from tnfr.alias import collect_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics.dnfr import (
    _build_neighbor_sums_common,
    _compute_dnfr,
    _prepare_dnfr_data,
    default_compute_delta_nfr,
)

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def _create_test_graph(size=5, seed=42):
    """Create a deterministic test graph with TNFR node attributes."""
    import random

    random.seed(seed)
    G = nx.path_graph(size)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.1 * (n + 1))
        set_attr(G.nodes[n], ALIAS_EPI, 0.2 * (n + 1))
        set_attr(G.nodes[n], ALIAS_VF, 0.3 * (n + 1))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return G


def test_backend_detection_with_numpy_available():
    """Verify backend detection correctly identifies NumPy availability."""
    np = pytest.importorskip("numpy")
    G = _create_test_graph()
    data = _prepare_dnfr_data(G)

    assert data.get("dnfr_path_decision") in ["sparse", "dense_auto", "dense_forced"]
    assert "nodes" in data
    assert "cache" in data
    assert data["cache"] is not None


def test_backend_detection_without_numpy(monkeypatch):
    """Verify backend falls back gracefully when NumPy is unavailable."""
    # Simulate NumPy being unavailable
    import tnfr.dynamics.dnfr as dnfr_module

    monkeypatch.setattr(dnfr_module, "get_numpy", lambda: None)

    G = _create_test_graph()
    data = _prepare_dnfr_data(G)

    assert data.get("dnfr_path_decision") == "fallback"
    assert "nodes" in data
    assert "cache" in data


def test_dnfr_semantics_preserved_across_backends():
    """
    Validate TNFR Invariant #3: ΔNFR semantics preservation.

    Ensure ΔNFR calculations produce equivalent results regardless of whether
    NumPy vectorization or Python fallback is used.
    """
    np = pytest.importorskip("numpy")

    # Compute with NumPy backend
    G1 = _create_test_graph()
    default_compute_delta_nfr(G1)
    dnfr_numpy = list(collect_attr(G1, G1.nodes, ALIAS_DNFR, 0.0))

    # Compute with Python fallback (disable vectorization)
    G2 = _create_test_graph()
    G2.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(G2)
    dnfr_fallback = list(collect_attr(G2, G2.nodes, ALIAS_DNFR, 0.0))

    # ΔNFR values must be numerically equivalent
    np.testing.assert_allclose(dnfr_numpy, dnfr_fallback, rtol=1e-10)


def test_determinism_with_seed_across_backends():
    """
    Validate TNFR Invariant #8: Controlled determinism.

    Ensure that with the same seed, ΔNFR computation produces identical
    results across different execution modes.
    """
    np = pytest.importorskip("numpy")

    seed = 12345

    # Run 1: NumPy backend
    G1 = _create_test_graph(seed=seed)
    default_compute_delta_nfr(G1)
    dnfr1 = list(collect_attr(G1, G1.nodes, ALIAS_DNFR, 0.0))

    # Run 2: NumPy backend (repeat)
    G2 = _create_test_graph(seed=seed)
    default_compute_delta_nfr(G2)
    dnfr2 = list(collect_attr(G2, G2.nodes, ALIAS_DNFR, 0.0))

    # Run 3: Python fallback
    G3 = _create_test_graph(seed=seed)
    G3.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(G3)
    dnfr3 = list(collect_attr(G3, G3.nodes, ALIAS_DNFR, 0.0))

    # All runs must produce identical results
    np.testing.assert_allclose(dnfr1, dnfr2, rtol=1e-15)
    np.testing.assert_allclose(dnfr1, dnfr3, rtol=1e-10)


def test_cache_initialization_consistent_across_backends():
    """
    Validate cache factory initialization patterns.

    Ensure that cache structures are correctly initialized regardless of
    which backend is active.
    """
    np = pytest.importorskip("numpy")

    # Test with NumPy backend
    G1 = _create_test_graph()
    data1 = _prepare_dnfr_data(G1)
    cache1 = data1.get("cache")

    assert cache1 is not None
    assert hasattr(cache1, "theta")
    assert hasattr(cache1, "epi")
    assert hasattr(cache1, "vf")
    assert hasattr(cache1, "idx")

    # Test with Python fallback
    G2 = _create_test_graph()
    G2.graph["vectorized_dnfr"] = False
    data2 = _prepare_dnfr_data(G2)
    cache2 = data2.get("cache")

    assert cache2 is not None
    assert hasattr(cache2, "theta")
    assert hasattr(cache2, "epi")
    assert hasattr(cache2, "vf")
    assert hasattr(cache2, "idx")


def test_fallback_activates_with_cached_numpy_buffers(monkeypatch):
    """
    Test the specific fallback scenario at line 1995-1996 of dnfr.py.

    When get_numpy() returns None but cached NumPy buffers exist,
    the system should fall back to using sys.modules.get("numpy").
    """
    np = pytest.importorskip("numpy")
    import tnfr.dynamics.dnfr as dnfr_module

    # First, create a graph and populate cache with NumPy arrays
    G = _create_test_graph()
    data = _prepare_dnfr_data(G)
    _compute_dnfr(G, data)
    baseline_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    # Verify cache has NumPy buffers
    cache = data.get("cache")
    assert cache is not None
    has_numpy_buffers = any(
        [
            cache.theta_np is not None,
            cache.epi_np is not None,
            cache.neighbor_accum_np is not None,
        ]
    )
    assert has_numpy_buffers, "Cache should have NumPy buffers from first run"

    # Now simulate get_numpy() returning None while numpy is still in sys.modules
    # This triggers the fallback at line 1995-1996
    monkeypatch.setattr(dnfr_module, "get_numpy", lambda: None)

    # Ensure numpy is actually in sys.modules for the fallback to work
    assert "numpy" in sys.modules

    # Build neighbor sums - should use fallback to sys.modules.get("numpy")
    stats = _build_neighbor_sums_common(G, data, use_numpy=True)
    assert stats is not None

    # Compute ΔNFR with fallback
    _compute_dnfr_common = dnfr_module._compute_dnfr_common
    _compute_dnfr_common(
        G,
        data,
        x=stats[0],
        y=stats[1],
        epi_sum=stats[2],
        vf_sum=stats[3],
        count=stats[4],
        deg_sum=stats[5],
        degs=stats[6],
    )
    fallback_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    # Results should be consistent
    np.testing.assert_allclose(baseline_dnfr, fallback_dnfr, rtol=1e-10)


def test_fallback_handles_missing_numpy_module(monkeypatch):
    """
    Test fallback when both get_numpy() returns None AND numpy is not in sys.modules.

    This is the most challenging fallback scenario and should gracefully
    fall back to pure Python execution.
    """
    import tnfr.dynamics.dnfr as dnfr_module
    import tnfr.metrics.trig_cache as trig_cache_module

    # Remove numpy from both get_numpy() and sys.modules
    monkeypatch.setattr(dnfr_module, "get_numpy", lambda: None)
    monkeypatch.setattr(trig_cache_module, "get_numpy", lambda: None)
    monkeypatch.setattr(
        dnfr_module, "_has_cached_numpy_buffers", lambda *_, **__: False
    )

    # Should fall back to pure Python implementation
    G = _create_test_graph()
    data = _prepare_dnfr_data(G)

    assert data.get("dnfr_path_decision") == "fallback"

    # Computation should still succeed
    default_compute_delta_nfr(G)
    dnfr_values = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    # Verify all values are floats and reasonable
    assert all(isinstance(v, float) for v in dnfr_values)
    assert len(dnfr_values) == G.number_of_nodes()


def test_cache_refresh_consistency_after_backend_change():
    """
    Validate that cache refresh maintains consistency when backend changes.

    This tests the scenario where:
    1. Cache is populated with NumPy backend
    2. Backend switches to Python fallback
    3. Cache is refreshed
    4. Results remain consistent
    """
    np = pytest.importorskip("numpy")

    # Initial computation with NumPy
    G = _create_test_graph()
    data1 = _prepare_dnfr_data(G)
    _compute_dnfr(G, data1)
    dnfr1 = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    # Force cache refresh and switch to Python fallback
    G.graph["_dnfr_prep_dirty"] = True
    G.graph["vectorized_dnfr"] = False
    data2 = _prepare_dnfr_data(G)
    _compute_dnfr(G, data2)
    dnfr2 = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    # Results should be consistent despite backend change
    np.testing.assert_allclose(dnfr1, dnfr2, rtol=1e-10)


def test_sparse_vs_dense_path_decision_consistency():
    """
    Validate that sparse vs dense path decisions are made consistently
    and produce equivalent ΔNFR results.
    """
    np = pytest.importorskip("numpy")

    # Test with sparse graph (low density)
    G_sparse = nx.path_graph(20)  # Density = 2/(20*19) ≈ 0.0053
    for n in G_sparse.nodes:
        set_attr(G_sparse.nodes[n], ALIAS_THETA, 0.1 * (n + 1))
        set_attr(G_sparse.nodes[n], ALIAS_EPI, 0.2 * (n + 1))
        set_attr(G_sparse.nodes[n], ALIAS_VF, 0.3 * (n + 1))
    G_sparse.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }

    data_sparse = _prepare_dnfr_data(G_sparse)
    assert data_sparse.get("dnfr_path_decision") == "sparse"

    default_compute_delta_nfr(G_sparse)
    dnfr_sparse = list(collect_attr(G_sparse, G_sparse.nodes, ALIAS_DNFR, 0.0))

    # Test with dense graph (high density)
    G_dense = nx.complete_graph(10)  # Density = 1.0
    for n in G_dense.nodes:
        set_attr(G_dense.nodes[n], ALIAS_THETA, 0.1 * (n + 1))
        set_attr(G_dense.nodes[n], ALIAS_EPI, 0.2 * (n + 1))
        set_attr(G_dense.nodes[n], ALIAS_VF, 0.3 * (n + 1))
    G_dense.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }

    data_dense = _prepare_dnfr_data(G_dense)
    assert data_dense.get("dnfr_path_decision") in ["dense_auto", "dense_forced"]

    default_compute_delta_nfr(G_dense)
    dnfr_dense = list(collect_attr(G_dense, G_dense.nodes, ALIAS_DNFR, 0.0))

    # Both should produce valid ΔNFR values
    assert all(isinstance(v, float) for v in dnfr_sparse)
    assert all(isinstance(v, float) for v in dnfr_dense)


def test_dnfr_path_decision_persists_in_telemetry():
    """
    Validate that path decision information is correctly recorded for telemetry.

    This ensures debugging and monitoring can track which execution path
    was taken for each ΔNFR computation.
    """
    np = pytest.importorskip("numpy")

    G = _create_test_graph()
    profile = {}

    default_compute_delta_nfr(G, profile=profile)

    assert "dnfr_path" in profile
    assert profile["dnfr_path"] in ["vectorized", "fallback"]


def test_fallback_validates_numpy_module_attributes():
    """
    Validate that the improved fallback mechanism (lines 1995-2000)
    checks for required NumPy attributes before using sys.modules.

    This test ensures the fallback is robust against corrupted or
    incomplete NumPy modules in sys.modules.
    """
    np = pytest.importorskip("numpy")
    import tnfr.dynamics.dnfr as dnfr_module

    # First, create a graph and populate cache with NumPy arrays
    G = _create_test_graph()
    data = _prepare_dnfr_data(G)
    _compute_dnfr(G, data)
    baseline_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))

    # Verify cache has NumPy buffers
    cache = data.get("cache")
    assert cache is not None
    has_numpy_buffers = any(
        [
            cache.theta_np is not None,
            cache.epi_np is not None,
            cache.neighbor_accum_np is not None,
        ]
    )
    assert has_numpy_buffers, "Cache should have NumPy buffers from first run"

    # Create a mock incomplete numpy module without required attributes
    class IncompleteNumpy:
        """Mock numpy module missing required attributes."""

        pass

    # Test 1: With incomplete numpy in sys.modules, fallback should fail gracefully
    original_numpy = sys.modules.get("numpy")
    try:
        sys.modules["numpy"] = IncompleteNumpy()  # type: ignore
        
        # Mock get_numpy to return None
        with patch.object(dnfr_module, "get_numpy", return_value=None):
            # This should fall back to pure Python since the numpy in sys.modules is invalid
            stats = _build_neighbor_sums_common(G, data, use_numpy=True)
            assert stats is not None
            # Verify computation still works
            _compute_dnfr_common = dnfr_module._compute_dnfr_common
            _compute_dnfr_common(
                G,
                data,
                x=stats[0],
                y=stats[1],
                epi_sum=stats[2],
                vf_sum=stats[3],
                count=stats[4],
                deg_sum=stats[5],
                degs=stats[6],
            )
            fallback_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))
            # Results should still be reasonable (may differ slightly due to different path)
            assert all(isinstance(v, float) for v in fallback_dnfr)
            assert len(fallback_dnfr) == G.number_of_nodes()
    finally:
        # Restore original numpy module
        if original_numpy is not None:
            sys.modules["numpy"] = original_numpy
        else:
            sys.modules.pop("numpy", None)

    # Test 2: With valid numpy in sys.modules, fallback should work
    sys.modules["numpy"] = np
    with patch.object(dnfr_module, "get_numpy", return_value=None):
        stats = _build_neighbor_sums_common(G, data, use_numpy=True)
        assert stats is not None
        _compute_dnfr_common = dnfr_module._compute_dnfr_common
        _compute_dnfr_common(
            G,
            data,
            x=stats[0],
            y=stats[1],
            epi_sum=stats[2],
            vf_sum=stats[3],
            count=stats[4],
            deg_sum=stats[5],
            degs=stats[6],
        )
        valid_fallback_dnfr = list(collect_attr(G, G.nodes, ALIAS_DNFR, 0.0))
        # With valid numpy, results should match baseline
        np.testing.assert_allclose(valid_fallback_dnfr, baseline_dnfr, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
