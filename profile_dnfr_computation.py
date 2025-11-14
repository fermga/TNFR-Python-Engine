"""
Profile ΔNFR computation and validation pipeline after Phase 3 optimizations.

Focus: Identify new bottlenecks now that eccentricity is cached.
Target: Functions taking >10% of validation time.
"""

import cProfile
import pstats
import io
from pstats import SortKey
import networkx as nx
import numpy as np

from tnfr.operators.definitions import Emission, Coupling, Coherence, Silence
from tnfr.operators.grammar import apply_glyph_with_grammar
from tnfr.validation.aggregator import run_structural_validation
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)


def setup_test_graph(n=500):
    """Create scale-free graph for profiling."""
    np.random.seed(42)
    G = nx.barabasi_albert_graph(n, m=3, seed=42)
    
    # Initialize TNFR attributes
    for node in G.nodes():
        G.nodes[node]['EPI'] = np.random.randn(10)
        G.nodes[node]['nu_f'] = 1.0 + np.random.rand()
        G.nodes[node]['theta'] = np.random.uniform(0, 2 * np.pi)
        G.nodes[node]['DELTA_NFR'] = np.random.randn()
    
    return G


def profile_validation_pipeline():
    """Profile full validation pipeline with field computations."""
    G = setup_test_graph(500)
    
    # Valid sequence
    sequence = ["AL", "UM", "IL", "SHA"]
    
    # Run validation 10 times (simulates repeated calls)
    for i in range(10):
        _ = run_structural_validation(
            G,
            sequence=sequence,
            max_delta_phi_s=2.0,
            max_phase_gradient=0.38,
            k_phi_flag_threshold=3.0,
            xi_c_critical_multiplier=1.0,
        )


def profile_dnfr_computation():
    """Profile ΔNFR computation in isolation."""
    G = setup_test_graph(500)
    
    from tnfr.operators.metrics import default_compute_delta_nfr
    
    # Compute ΔNFR 100 times
    for i in range(100):
        for node in G.nodes():
            _ = default_compute_delta_nfr(G, node)


def profile_field_computations():
    """Profile canonical field tetrad computation."""
    G = setup_test_graph(500)
    
    # Compute each field 20 times (test caching)
    for i in range(20):
        _ = compute_structural_potential(G, alpha=2.0)
        _ = compute_phase_gradient(G)
        _ = compute_phase_curvature(G)
        _ = estimate_coherence_length(G)


def profile_operator_application():
    """Profile operator application sequence."""
    G = setup_test_graph(500)
    
    # Apply operator sequence 50 times
    ops = [Emission(), Coupling(), Coherence(), Silence()]
    
    for run in range(50):
        for op in ops:
            for node in list(G.nodes())[:10]:  # Sample of nodes
                apply_glyph_with_grammar(G, node, op)


def run_comprehensive_profile():
    """Run all profiling scenarios."""
    print("=" * 80)
    print("TNFR DNFR Computation Profiling - Post Phase 3 Optimizations")
    print("=" * 80)
    
    scenarios = [
        (
            "Full Validation Pipeline (10 runs, 500 nodes)",
            profile_validation_pipeline,
        ),
        ("DNFR Computation (100 iterations)", profile_dnfr_computation),
        (
            "Field Computations (20 iterations, test cache)",
            profile_field_computations,
        ),
        ("Operator Application (50 sequences)", profile_operator_application),
    ]
    
    for name, func in scenarios:
        print(f"\n{'=' * 80}")
        print(f"Profiling: {name}")
        print('=' * 80)
        
        # Profile
        profiler = cProfile.Profile()
        profiler.enable()
        func()
        profiler.disable()
        
        # Create stats object
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
        
        # Print top 30 functions
        ps.print_stats(30)
        output = s.getvalue()
        
        print(output)
        
        # Also print by self-time (internal time)
        print(f"\n{'-' * 80}")
        print(f"Top 15 by SELF TIME (internal execution):")
        print('-' * 80)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.TIME)
        ps.print_stats(15)
        print(s.getvalue())
        
        # Save detailed stats to file
        filename = f"profile_dnfr_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.stats"
        profiler.dump_stats(filename)
        print(f"\nDetailed stats saved to: {filename}")


if __name__ == "__main__":
    run_comprehensive_profile()
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print("\nAnalysis:")
    print("- Review cumulative time to find high-level bottlenecks")
    print("- Review self-time to find tight loops consuming CPU")
    print("- Look for functions >10% of total time")
    print("\nVisualize with snakeviz:")
    print("  pip install snakeviz")
    print("  snakeviz profile_dnfr_*.stats")
