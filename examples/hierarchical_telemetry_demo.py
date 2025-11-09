"""Demonstration of hierarchical depth telemetry for nested THOL bifurcations.

This example showcases the new hierarchical tracking features for THOL
(Self-organization) operator, demonstrating operational fractality across
multiple nested levels.

New Features Demonstrated:
1. bifurcation_level tracking in sub_epi records
2. hierarchy_path for full ancestor chain
3. compute_hierarchical_depth() for recursive depth measurement
4. print_bifurcation_hierarchy() for ASCII visualization
5. Depth validation warnings for excessive nesting
"""

from tnfr.structural import create_nfr
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.metabolism import compute_hierarchical_depth
from tnfr.visualization import print_bifurcation_hierarchy, get_hierarchy_info


def demo_single_level_bifurcation():
    """Demonstrate single-level bifurcation with hierarchy telemetry."""
    print("=" * 60)
    print("DEMO 1: Single-Level Bifurcation")
    print("=" * 60)
    
    # Create node
    G, node = create_nfr("root", epi=0.50, vf=1.0, theta=0.1)
    
    # Set up for bifurcation (proper acceleration)
    G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
    
    # Apply THOL
    SelfOrganization()(G, node, tau=0.05)
    
    # Check hierarchy metadata
    sub_epi = G.nodes[node]["sub_epis"][0]
    print(f"\n📊 Sub-EPI Metadata:")
    print(f"  - Bifurcation level: {sub_epi['bifurcation_level']}")
    print(f"  - Hierarchy path: {sub_epi['hierarchy_path']}")
    print(f"  - Node ID: {sub_epi['node_id']}")
    
    # Compute depth
    depth = compute_hierarchical_depth(G, node)
    print(f"\n📏 Maximum depth: {depth}")
    
    # Visualize
    print(f"\n🌳 Hierarchy Visualization:")
    print_bifurcation_hierarchy(G, node)
    
    # Get info
    info = get_hierarchy_info(G, node)
    print(f"\n📈 Hierarchy Info:")
    print(f"  - Max depth: {info['max_depth']}")
    print(f"  - Total descendants: {info['total_descendants']}")
    print()


def demo_two_level_nested_bifurcation():
    """Demonstrate two-level nested bifurcation."""
    print("=" * 60)
    print("DEMO 2: Two-Level Nested Bifurcation")
    print("=" * 60)
    
    # Create root node
    G, node = create_nfr("root", epi=0.50, vf=1.0, theta=0.1)
    
    # Level 1 bifurcation
    print("\n🔹 Creating Level 1 bifurcation...")
    G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
    SelfOrganization()(G, node, tau=0.05)
    
    # Get sub-node
    sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
    print(f"  Sub-node created: {sub_node}")
    print(f"  Bifurcation level: {G.nodes[sub_node]['_bifurcation_level']}")
    
    # Level 2 bifurcation (nested)
    print("\n🔹 Creating Level 2 bifurcation (nested)...")
    G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
    SelfOrganization()(G, sub_node, tau=0.05)
    
    # Get nested sub-node
    nested_sub_node = G.nodes[sub_node]["sub_epis"][0]["node_id"]
    print(f"  Nested sub-node created: {nested_sub_node}")
    print(f"  Bifurcation level: {G.nodes[nested_sub_node]['_bifurcation_level']}")
    
    # Check depth
    depth = compute_hierarchical_depth(G, node)
    print(f"\n📏 Maximum depth: {depth}")
    
    # Visualize full hierarchy
    print(f"\n🌳 Complete Hierarchy:")
    print_bifurcation_hierarchy(G, node)
    
    # Show hierarchy paths
    print(f"\n🗺️ Hierarchy Paths:")
    for i, se in enumerate(G.nodes[node]["sub_epis"], 1):
        print(f"  Level 1 Sub-EPI {i}: {se['hierarchy_path']}")
    
    for i, se in enumerate(G.nodes[sub_node]["sub_epis"], 1):
        print(f"  Level 2 Sub-EPI {i}: {se['hierarchy_path']}")
    print()


def demo_three_level_nested_bifurcation():
    """Demonstrate three-level nested bifurcation."""
    print("=" * 60)
    print("DEMO 3: Three-Level Nested Bifurcation")
    print("=" * 60)
    
    # Create root
    G, node = create_nfr("alpha", epi=0.50, vf=1.0, theta=0.1)
    
    # Build three levels
    print("\n🔹 Building three-level hierarchy...")
    
    # Level 1
    G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
    SelfOrganization()(G, node, tau=0.05)
    sub_1 = G.nodes[node]["sub_epis"][0]["node_id"]
    print(f"  Level 1: {sub_1} (level={G.nodes[sub_1]['_bifurcation_level']})")
    
    # Level 2
    G.nodes[sub_1]["epi_history"] = [0.05, 0.15, 0.35]
    SelfOrganization()(G, sub_1, tau=0.05)
    sub_2 = G.nodes[sub_1]["sub_epis"][0]["node_id"]
    print(f"  Level 2: {sub_2} (level={G.nodes[sub_2]['_bifurcation_level']})")
    
    # Level 3
    G.nodes[sub_2]["epi_history"] = [0.02, 0.08, 0.20]
    SelfOrganization()(G, sub_2, tau=0.05)
    sub_3 = G.nodes[sub_2]["sub_epis"][0]["node_id"]
    print(f"  Level 3: {sub_3} (level={G.nodes[sub_3]['_bifurcation_level']})")
    
    # Compute depth
    depth = compute_hierarchical_depth(G, node)
    print(f"\n📏 Maximum depth: {depth}")
    
    # Visualize
    print(f"\n🌳 Three-Level Hierarchy:")
    print_bifurcation_hierarchy(G, node)
    
    # Get comprehensive info
    info = get_hierarchy_info(G, node)
    print(f"\n📊 Comprehensive Info:")
    print(f"  - Root node: {info['node']}")
    print(f"  - Root EPI: {info['epi']:.3f}")
    print(f"  - Max depth: {info['max_depth']}")
    print(f"  - Total descendants: {info['total_descendants']}")
    print()


def demo_depth_validation():
    """Demonstrate depth validation warnings."""
    print("=" * 60)
    print("DEMO 4: Depth Validation (Warning System)")
    print("=" * 60)
    
    # Create node with low max depth
    G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)
    G.graph["THOL_MAX_BIFURCATION_DEPTH"] = 1  # Low threshold for demo
    
    print("\n⚙️  Configuration:")
    print(f"  THOL_MAX_BIFURCATION_DEPTH = 1")
    
    # Level 1 - no warning
    print("\n🔹 Creating Level 1 (no warning expected)...")
    G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
    SelfOrganization()(G, node, tau=0.05)
    print("  ✅ Level 1 created successfully")
    
    # Level 2 - should warn
    print("\n🔹 Creating Level 2 (warning expected)...")
    sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
    G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
    
    # This will trigger warning
    print("  ⚠️  Attempting bifurcation at max depth...")
    SelfOrganization()(G, sub_node, tau=0.05)
    
    # Check warning was recorded
    if G.nodes[sub_node].get("_thol_max_depth_warning"):
        print("  ✅ Depth warning recorded in node")
    
    events = G.graph.get("thol_depth_warnings", [])
    if events:
        print(f"  ✅ Depth warning recorded in graph (count: {len(events)})")
        print(f"     Event details: {events[0]}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HIERARCHICAL DEPTH TELEMETRY DEMONSTRATION")
    print("Nested THOL Bifurcation Tracking")
    print("=" * 60 + "\n")
    
    # Run all demos
    demo_single_level_bifurcation()
    demo_two_level_nested_bifurcation()
    demo_three_level_nested_bifurcation()
    demo_depth_validation()
    
    print("=" * 60)
    print("✨ All demonstrations completed successfully!")
    print("=" * 60)
