"""TNFR Self-Optimization — Intrinsic Agency on the Structural Manifold.

Demonstrates that TNFR networks possess intrinsic agency: the ability to
analyse their own mathematical structure and select optimal transformation
strategies via gradient descent on the structural manifold.

Key results shown:
1. Mathematical optimisation landscape analysis (unified fields, conservation,
   graph structure, nodal-equation analysis, recommendations)
2. Automatic strategy recommendation from learned policies
3. Experience-based learning loop (record → policy extraction → adaptive config)
4. Exported knowledge: policies, adaptive configuration, performance statistics
5. Dry-run optimisation with structural telemetry snapshots before / after
6. Conservation integrity feedback driving strategy reordering

Physics basis:
  This is NOT "AI magic."  The self-optimising engine performs *gradient
  descent on the structural manifold* driven by the pressure term ΔNFR
  in the nodal equation ∂EPI/∂t = νf · ΔNFR(t).  Unified-field telemetry
  (Ψ, χ, S, C) and conservation invariants (Noether charge, Lyapunov
  derivative) close the feedback loop.

See: AGENTS.md § Self-Optimizing Dynamics
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.operators import apply_glyph
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.dynamics.self_optimizing_engine import (
    TNFRSelfOptimizingEngine,
    OptimizationObjective,
    LearningStrategy,
    OptimizationExperience,
    SelfOptimizationResult,
    create_self_optimizing_engine,
    auto_optimize_tnfr_computation,
)

SEED = 42
np.random.seed(SEED)

# ── helpers ──────────────────────────────────────────────────────────
HEADER = "=" * 72
SECTION = "-" * 60

def banner(title: str) -> None:
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


def build_network(n: int = 20, p: float = 0.3, seed: int = SEED) -> nx.Graph:
    """Create and initialise a random TNFR network."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    inject_defaults(G)
    # Bootstrap a few nodes so the graph has non-trivial EPI / ΔNFR
    for node in list(G.nodes())[:5]:
        apply_glyph(G, node, "AL")   # Emission
    for node in list(G.nodes())[:5]:
        apply_glyph(G, node, "IL")   # Coherence
    return G


def print_dict(d: dict, indent: int = 2) -> None:
    """Pretty-print a dict with controlled depth."""
    prefix = " " * indent
    for key, val in d.items():
        if isinstance(val, dict):
            print(f"{prefix}{key}:")
            print_dict(val, indent + 4)
        elif isinstance(val, (list, tuple)):
            if len(val) == 0:
                print(f"{prefix}{key}: []")
            elif len(val) <= 6:
                print(f"{prefix}{key}: {val}")
            else:
                print(f"{prefix}{key}: [{val[0]}, ... ({len(val)} items)]")
        elif isinstance(val, float):
            print(f"{prefix}{key}: {val:.6f}")
        else:
            print(f"{prefix}{key}: {val}")


# ── § 1  Optimisation Landscape Analysis ──────────────────────────────
banner("§ 1  Mathematical Optimisation Landscape")

G = build_network()
engine = TNFRSelfOptimizingEngine(
    learning_strategy=LearningStrategy.MATHEMATICAL_ANALYSIS,
    optimization_objective=OptimizationObjective.BALANCE_ALL,
)

landscape = engine.analyze_mathematical_optimization_landscape(G, "general")

print("\nLandscape keys:", sorted(landscape.keys()))

if "unified_field_analysis" in landscape:
    print(f"\n{SECTION}")
    print("  Unified Field Analysis  (Ψ, χ, S, C, E, Q)")
    print(SECTION)
    print_dict(landscape["unified_field_analysis"])

if "conservation_feedback" in landscape:
    print(f"\n{SECTION}")
    print("  Conservation Integrity Feedback")
    print(SECTION)
    print_dict(landscape["conservation_feedback"])

if "graph_structure" in landscape:
    print(f"\n{SECTION}")
    print("  Graph Structure")
    print(SECTION)
    print_dict(landscape["graph_structure"])

if "nodal_equation_analysis" in landscape:
    print(f"\n{SECTION}")
    print("  Nodal Equation Analysis")
    print(SECTION)
    nea = landscape["nodal_equation_analysis"]
    print(f"  EPI variance:    {nea.get('epi_variance', 0):.6f}")
    print(f"  νf range:        {nea.get('vf_range', 0):.6f}")
    print(f"  |ΔNFR| mean:    {nea.get('dnfr_magnitude', 0):.6f}")
    recs = nea.get("optimization_recommendations", [])
    print(f"  Recommendations ({len(recs)}):")
    for r in recs:
        print(f"    • {r}")

if "pattern_optimization_hints" in landscape:
    hints = landscape["pattern_optimization_hints"]
    print(f"\n  Pattern hints ({len(hints)}):")
    for h in hints:
        print(f"    • {h}")


# ── § 2  Strategy Recommendation ──────────────────────────────────────
banner("§ 2  Strategy Recommendation (SelfOptimizationResult)")

result: SelfOptimizationResult = engine.recommend_optimization_strategy(
    G, operation_type="general"
)

print(f"\n  Recommended strategies ({len(result.recommended_strategies)}):")
for s in result.recommended_strategies[:10]:
    speedup = result.predicted_speedups.get(s)
    extra = f"  (predicted speedup: {speedup:.2f}×)" if speedup else ""
    print(f"    • {s}{extra}")

print(f"\n  Learned policies (matching): {len(result.learned_policies)}")
print(f"  Adaptive configuration:")
print_dict(result.adaptive_configurations)
print(f"  Analysis time: {result.execution_time * 1000:.1f} ms")

if result.conservation_feedback is not None:
    print(f"\n  Conservation feedback:")
    print_dict(result.conservation_feedback)


# ── § 3  Experience-Based Learning Loop ───────────────────────────────
banner("§ 3  Experience-Based Learning Loop")

print("\n  Simulating 15 optimisation experiments ...\n")

topologies = [
    ("ring", lambda n, s: nx.cycle_graph(n)),
    ("random", lambda n, s: nx.erdos_renyi_graph(n, 0.3, seed=s)),
    ("star", lambda n, s: nx.star_graph(n - 1)),
]

strategies = ["spectral", "vectorized", "cache", "structural", "hybrid"]

for i in range(15):
    topo_name, topo_fn = topologies[i % len(topologies)]
    n = 10 + i * 5
    G_exp = topo_fn(n, SEED + i)
    inject_defaults(G_exp)

    # Bootstrap
    for node in list(G_exp.nodes())[:3]:
        apply_glyph(G_exp, node, "AL")   # Emission
        apply_glyph(G_exp, node, "IL")   # Coherence

    num_nodes = len(G_exp.nodes())
    num_edges = len(G_exp.edges())
    density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    strategy = strategies[i % len(strategies)]
    speedup = 1.0 + np.random.exponential(0.5)

    exp = OptimizationExperience(
        graph_properties={"nodes": num_nodes, "edges": num_edges, "density": density},
        operation_type="general",
        strategy_used=strategy,
        parameters={"topology": topo_name},
        performance_metrics={"speedup_factor": float(speedup), "execution_time": 0.01 * n},
        timestamp=time.time(),
        success=speedup > 1.05,
    )
    engine.learn_from_experience(exp)

print(f"  Experiences recorded: {len(engine.experience_history)}")
print(f"  Successful:           {engine.successful_optimizations} / {engine.optimization_attempts}")
print(f"  Learned policies:     {len(engine.learned_policies)}")

for p in engine.learned_policies[:5]:
    print(
        f"\n  Policy: {p.policy_name}"
        f"\n    conditions: {p.conditions}"
        f"\n    action:     {p.actions}"
        f"\n    confidence: {p.confidence:.2f}  |  avg improvement: {p.average_improvement:.3f}×"
    )


# ── § 4  Exported Knowledge ──────────────────────────────────────────
banner("§ 4  Exported Knowledge")

knowledge = engine.export_learned_knowledge()

print(f"\n  Performance statistics:")
print_dict(knowledge["performance_statistics"])

print(f"\n  Adaptive configuration:")
print_dict(knowledge["adaptive_configuration"])

print(f"\n  Policies exported: {len(knowledge['learned_policies'])}")
for p in knowledge["learned_policies"][:3]:
    print(f"    • {p['name']}  (confidence {p['confidence']:.2f})")


# ── § 5  Dry-Run Automatic Optimisation ───────────────────────────────
banner("§ 5  Dry-Run Automatic Optimisation")

G2 = build_network(n=30, p=0.25, seed=SEED + 100)
C_before = compute_coherence(G2)
si_result = compute_Si(G2)
Si_mean = float(np.mean(list(si_result.values()))) if isinstance(si_result, dict) else float(np.mean(si_result))

print(f"\n  Pre-optimisation telemetry:")
print(f"    C(t)  = {C_before:.6f}")
print(f"    Si    = {Si_mean:.6f}")
print(f"    Nodes = {len(G2.nodes())}  |  Edges = {len(G2.edges())}")

dry_result = engine.optimize_automatically(
    G2, operation_type="general", dry_run=True, seed=SEED,
)

print(f"\n  Dry-run mode: {dry_result.get('dry_run', False)}")
snapshot_path = dry_result.get("snapshot_path")
if snapshot_path:
    print(f"  Snapshot saved to: {snapshot_path}")
sig = dry_result.get("signature")
if sig:
    print(f"  Signature:         {sig[:40]}...")

recs_obj = dry_result.get("recommendations")
if recs_obj is not None and hasattr(recs_obj, "recommended_strategies"):
    print(f"\n  Recommended strategies:")
    for s in recs_obj.recommended_strategies[:8]:
        print(f"    • {s}")


# ── § 6  Convenience API ─────────────────────────────────────────────
banner("§ 6  Convenience API — auto_optimize_tnfr_computation()")

G3 = build_network(n=15, p=0.35, seed=SEED + 200)

auto_result = auto_optimize_tnfr_computation(G3, "general", dry_run=True, seed=SEED)

print(f"\n  dry_run: {auto_result.get('dry_run', '?')}")
recs_auto = auto_result.get("recommendations")
if recs_auto is not None and hasattr(recs_auto, "recommended_strategies"):
    strats = recs_auto.recommended_strategies
    print(f"  Strategies ({len(strats)}):")
    for s in strats[:6]:
        print(f"    • {s}")

print(f"\n  Conservation closed-loop:")
cf = auto_result.get("conservation")
if cf is not None:
    print_dict(cf)
else:
    print("    (no live conservation data in dry-run)")

# ── Summary ──────────────────────────────────────────────────────────
banner("Summary")
print("""
  Self-optimisation in TNFR is gradient descent on the structural manifold.
  The engine:
    1. Analyses the mathematical landscape (unified fields, conservation,
       graph structure, nodal equation properties).
    2. Recommends strategies — from learned policies and from mathematical
       analysis of Ψ, χ, S, C, E, Q.
    3. Records experiences and extracts optimisation policies.
    4. Reorders strategies when conservation health is stressed.
    5. Persists knowledge for reuse across sessions.

  Physics:  ∂EPI/∂t = νf · ΔNFR(t)  →  natural gradient on structural
  manifold.  Grammar rules (U1-U6) define the constraint sub-manifold.
  Conservation laws (Noether charge, Lyapunov derivative) close the
  feedback loop.

  See: AGENTS.md § Self-Optimizing Dynamics
""")
