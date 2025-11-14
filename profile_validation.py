"""Profile validation aggregator to identify hot paths."""
import cProfile
import pstats
import io
from pstats import SortKey
import networkx as nx

from tnfr.validation.aggregator import run_structural_validation
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

# Create test graph (moderate size)
print("Creating test graph (500 nodes, scale-free)...")
G = nx.barabasi_albert_graph(500, 3, seed=42)

# Initialize node attributes
for n in G.nodes():
    G.nodes[n]['delta_nfr'] = 0.5
    G.nodes[n]['phase'] = 0.3
    G.nodes[n]['vf'] = 1.0
    G.nodes[n]['coherence'] = 0.8
    G.nodes[n]['EPI'] = [0.0] * 10

sequence = ["AL", "UM", "IL", "OZ", "THOL", "IL", "SHA"]

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Sequence: {sequence}")
print("\n" + "=" * 80)

# Profile validation
print("\n1. PROFILING: Full Validation (with grammar + fields)")
print("-" * 80)

pr = cProfile.Profile()
pr.enable()

# Run validation 10 times to get meaningful stats
for _ in range(10):
    report = run_structural_validation(
        G,
        sequence=sequence,
        max_delta_phi_s=2.0,
        max_phase_gradient=0.38,
    )

pr.disable()

# Print stats
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats(30)  # Top 30 functions
print(s.getvalue())

print("\n" + "=" * 80)
print("\n2. PROFILING: Fields Only (no grammar)")
print("-" * 80)

pr2 = cProfile.Profile()
pr2.enable()

# Run field computations 10 times
for _ in range(10):
    phi_s = compute_structural_potential(G)
    grad = compute_phase_gradient(G)
    curv = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

pr2.disable()

s2 = io.StringIO()
ps2 = pstats.Stats(pr2, stream=s2).sort_stats(SortKey.CUMULATIVE)
ps2.print_stats(30)
print(s2.getvalue())

print("\n" + "=" * 80)
print("\nProfiling complete. Key findings:")
print("- Check 'cumtime' column for total time in function + children")
print("- Functions with high 'tottime' are bottlenecks (self time)")
print("- Focus optimization on top 5-10 functions by cumtime")
