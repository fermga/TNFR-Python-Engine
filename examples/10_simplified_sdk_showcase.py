#!/usr/bin/env python3
"""🌊 **TNFR Simplified SDK Example** - New Optimized API ⭐

This example demonstrates the new, simplified TNFR SDK that makes creating
and analyzing networks incredibly easy while maintaining full theoretical power.

**INSTALLATION**: pip install tnfr
**IMPORT**: from tnfr.sdk import TNFR
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("🌊 TNFR SIMPLIFIED SDK - Quick Start Guide")
print("=" * 50)
print()

# === 1. INSTANT NETWORK CREATION ===
print("🚀 1. INSTANT NETWORK CREATION")
print("-" * 30)

from tnfr.sdk.simple import TNFR

# Create networks with one line
small_net = TNFR.create(5)
print(f"Empty network:     {small_net.summary()}")

ring_net = TNFR.create(8).ring()
print(f"Ring network:      {ring_net.summary()}")

random_net = TNFR.create(12).random(0.3)
print(f"Random network:    {random_net.summary()}")

star_net = TNFR.create(10).star()
print(f"Star network:      {star_net.summary()}")

complete_net = TNFR.create(6).complete()
print(f"Complete network:  {complete_net.summary()}")

print()

# === 2. TEMPLATES FOR COMMON PATTERNS ===
print("📋 2. READY-MADE TEMPLATES")
print("-" * 30)

molecule = TNFR.template('molecule')
print(f"Molecule template: {molecule.summary()}")

small_world = TNFR.template('medium')  # 15 nodes, small-world-like
print(f"Small world:       {small_world.summary()}")

large_random = TNFR.template('large')  # 50 nodes, random
print(f"Large random:      {large_random.summary()}")

print()

# === 3. EVOLUTION & OPTIMIZATION ===
print("🧬 3. EVOLUTION & OPTIMIZATION")
print("-" * 30)

# Create and evolve
net = TNFR.create(15).ring()
before = net.coherence()
print(f"Before evolution: C={before:.3f}")

net.evolve(5)
after_evolution = net.coherence()
print(f"After evolution:  C={after_evolution:.3f}")

net.auto_optimize()
after_optimization = net.coherence()
print(f"After auto-opt:   C={after_optimization:.3f}")

print()

# === 4. CHAIN OPERATIONS ===
print("⛓️ 4. CHAIN OPERATIONS (FLUENT API)")
print("-" * 30)

# Everything in one line!
result = TNFR.create(20).random(0.3).evolve(3).auto_optimize().results()
print(f"One-liner result: {result.summary()}")

# Template + evolution + optimization
optimized = TNFR.template('molecule').evolve(5).auto_optimize()
print(f"Template pipeline: {optimized.summary()}")

print()

# === 5. NETWORK COMPARISON ===
print("⚖️ 5. NETWORK ANALYSIS & COMPARISON")
print("-" * 30)

# Create different topologies
networks = {
    'ring': TNFR.create(10).ring().evolve(3),
    'star': TNFR.create(10).star().evolve(3),
    'random': TNFR.create(10).random(0.4).evolve(3),
    'complete': TNFR.create(10).complete().evolve(3)
}

# Compare them
comparison = TNFR.compare(*networks.values())

print("Topology comparison (by coherence):")
for i, result in enumerate(comparison['ranking'], 1):
    name = list(networks.keys())[result['index']]
    coherence = result['coherence']
    nodes = result['nodes']
    edges = result['edges']
    print(f"  {i}. {name:8s}: C={coherence:.3f} (N={nodes}, E={edges})")

print()

# === 6. POWER USER SHORTCUTS ===
print("⚡ 6. POWER USER SHORTCUTS")
print("-" * 30)

from tnfr.sdk.simple import T  # Ultra-short alias

# Ultra-compact syntax
result = T.create(8).complete().results()
print(f"Ultra-short (T): {result.summary()}")

# Check coherence quickly
net = T.template('star')
if net.results().is_coherent():
    print("✅ Network is coherent!")
else:
    print("❌ Network needs work")

# Get detailed info
info = net.info()
print(f"Network info: {info['nodes']} nodes, density={info['density']:.2f}")

print()

# === 7. REAL-WORLD EXAMPLE ===
print("🌍 7. REAL-WORLD EXAMPLE - Social Network Analysis")
print("-" * 30)

# Simulate different social network structures
social_networks = {
    'family_group': T.create(6).complete(),  # Everyone knows everyone
    'friend_circle': T.create(12).ring().random(0.2),  # Ring + random connections
    'hierarchical': T.create(15).star(),  # Central leader
    'community': T.create(20).random(0.15)  # Sparse random connections
}

print("Social network coherence analysis:")
for name, net in social_networks.items():
    # Evolve to see natural dynamics
    evolved = net.evolve(3)
    result = evolved.results()
    
    status = "👍 Stable" if result.is_stable() else "⚠️  Needs attention"
    print(f"  {name:12s}: C={result.coherence:.3f}, {status}")

print()

print("🎉 TNFR SDK EXPLORATION COMPLETE!")
print("=" * 50)
print()
print("🚀 NEXT STEPS:")
print("  • Try your own network topologies")
print("  • Experiment with evolution parameters")  
print("  • Compare different optimization strategies")
print("  • Use templates as starting points")
print("  • Explore the full TNFR theory in AGENTS.md")
print()
print("📚 LEARN MORE:")
print("  • Repository: https://github.com/fermga/TNFR-Python-Engine")
print("  • Theory: Read AGENTS.md for complete guide")
print("  • Examples: Check examples/ directory")
print("  • Install: pip install tnfr")