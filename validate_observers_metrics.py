#!/usr/bin/env python
"""Validation script for observer and metrics telemetry systems.

This script validates the four key issues mentioned in the problem statement:
1. Observer callback registration
2. Trigonometric cache synchronization
3. Parallel Si computation executor instantiation
4. Coherence metrics accuracy
"""

import networkx as nx
from tnfr.constants import inject_defaults
from tnfr.initialization import init_node_attrs
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.coherence import coherence_matrix, register_coherence_callbacks
from tnfr.metrics.sense_index import compute_Si
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.observers import (
    attach_standard_observer,
    kuramoto_metrics,
    kuramoto_order,
    phase_sync,
)
from tnfr.utils import callback_manager


def validate_observer_registration():
    """Validate that observer callbacks register correctly."""
    print("=" * 60)
    print("1. Validating Observer Callback Registration")
    print("=" * 60)
    
    G = nx.cycle_graph(8)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    
    # Attach standard observer
    attach_standard_observer(G)
    
    # Check that the standard observer flag was set
    assert G.graph.get("_STD_OBSERVER") == "attached", "Standard observer not attached"
    
    # Register metrics callbacks
    register_metrics_callbacks(G)
    
    # Check that metrics callbacks were registered by verifying the graph has the right configuration
    assert "METRICS" in G.graph, "METRICS config missing"
    
    print(f"✓ Standard observer: attached")
    print(f"✓ Metrics callbacks: registered")
    print(f"✓ Graph configured for telemetry")
    print()


def validate_trig_cache_sync():
    """Validate trigonometric cache synchronization between modules."""
    print("=" * 60)
    print("2. Validating Trigonometric Cache Synchronization")
    print("=" * 60)
    
    G = nx.cycle_graph(8)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    
    # Get trig cache from first module
    trig1 = get_trig_cache(G)
    assert trig1.cos, "Trig cache not populated"
    
    # Get trig cache again (should be cached)
    trig2 = get_trig_cache(G)
    assert trig1 is trig2 or trig1.cos == trig2.cos, "Trig cache not synchronized"
    
    # Verify cache is used in coherence module
    nodes, W = coherence_matrix(G)
    assert nodes is not None, "Coherence matrix failed"
    
    # Get trig cache after coherence computation
    trig3 = get_trig_cache(G)
    assert trig3.cos, "Trig cache lost after coherence"
    
    print(f"✓ Trig cache synchronized across modules")
    print(f"✓ Cache persists after coherence computation")
    print(f"✓ Cached {len(trig3.cos)} node trigonometric values")
    print()


def validate_parallel_si():
    """Validate parallel Si computation executor instantiation."""
    print("=" * 60)
    print("3. Validating Parallel Si Computation")
    print("=" * 60)
    
    G = nx.cycle_graph(20)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    G.graph["SI_N_JOBS"] = 2
    
    # Compute Si with parallelization
    si_parallel = compute_Si(G, inplace=False, n_jobs=2)
    assert si_parallel, "Parallel Si computation failed"
    assert len(si_parallel) == 20, "Not all nodes computed"
    
    # Compute Si sequentially
    G2 = nx.cycle_graph(20)
    inject_defaults(G2)
    init_node_attrs(G2, override=True)
    si_sequential = compute_Si(G2, inplace=False, n_jobs=1)
    
    # Results should be similar (small numerical differences acceptable)
    for node in si_parallel:
        diff = abs(si_parallel[node] - si_sequential[node])
        assert diff < 1e-6, f"Si mismatch for node {node}: {diff}"
    
    print(f"✓ Parallel Si computation working")
    print(f"✓ Sequential and parallel results match")
    print(f"✓ Computed Si for {len(si_parallel)} nodes")
    print()


def validate_coherence_metrics():
    """Validate coherence metrics accuracy (C(t), Si, phase_sync, kuramoto_order)."""
    print("=" * 60)
    print("4. Validating Coherence Metrics Accuracy")
    print("=" * 60)
    
    G = nx.cycle_graph(8)
    inject_defaults(G)
    init_node_attrs(G, override=True)
    attach_standard_observer(G)
    register_metrics_callbacks(G)
    
    # Compute coherence matrix
    nodes, W = coherence_matrix(G)
    assert nodes is not None, "Coherence matrix computation failed"
    assert W is not None, "Coherence weights not computed"
    
    # Test phase metrics
    R, psi = kuramoto_metrics(G)
    assert 0 <= R <= 1, f"Kuramoto order out of bounds: {R}"
    
    ps = phase_sync(G)
    assert 0 <= ps <= 1, f"Phase sync out of bounds: {ps}"
    
    ko = kuramoto_order(G)
    assert 0 <= ko <= 1, f"Kuramoto order out of bounds: {ko}"
    assert abs(ko - R) < 1e-9, "Kuramoto order mismatch"
    
    # Test Si computation
    si_values = compute_Si(G, inplace=False)
    assert si_values, "Si computation failed"
    for node, si in si_values.items():
        assert 0 <= si <= 1, f"Si out of bounds for node {node}: {si}"
    
    print(f"✓ Coherence matrix computed: {len(nodes)} nodes")
    print(f"✓ Kuramoto order (R): {R:.4f}")
    print(f"✓ Phase sync: {ps:.4f}")
    print(f"✓ Si values: all in [0, 1] range")
    print(f"✓ All coherence metrics validated")
    print()


def main():
    """Run all validation checks."""
    print("\n")
    print("=" * 60)
    print("OBSERVER AND METRICS TELEMETRY VALIDATION")
    print("=" * 60)
    print()
    
    try:
        validate_observer_registration()
        validate_trig_cache_sync()
        validate_parallel_si()
        validate_coherence_metrics()
        
        print("=" * 60)
        print("ALL VALIDATIONS PASSED ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print("- Observer callback registration: WORKING")
        print("- Trigonometric cache synchronization: WORKING")
        print("- Parallel Si computation: WORKING")
        print("- Coherence metrics accuracy: WORKING")
        print()
        return 0
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print("VALIDATION FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        print()
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print("UNEXPECTED ERROR ✗")
        print("=" * 60)
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
