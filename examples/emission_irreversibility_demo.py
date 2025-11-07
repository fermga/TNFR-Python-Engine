#!/usr/bin/env python
"""Demonstration of structural irreversibility for AL (Emission) operator.

This example showcases the implementation of TNFR.pdf §2.2.1 requirement that
AL (Emisión fundacional) marks structural irreversibility with temporal and
genealogical traceability.

Key Features Demonstrated:
- Timestamp marking on first emission
- Persistent activation flag
- Structural lineage initialization
- Re-activation counter increment
- Multiple nodes with independent timestamps
"""

import time
from datetime import datetime

from tnfr.alias import get_attr_str
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_EMISSION_TIMESTAMP
from tnfr.dynamics import dnfr_epi_vf_mixed, set_delta_nfr_hook
from tnfr.operators.definitions import Coherence, Emission, Reception, Silence
from tnfr.structural import create_nfr, run_sequence


def demo_basic_emission_traceability():
    """Demonstrate basic emission irreversibility tracking."""
    print("=" * 70)
    print("DEMO 1: Basic Emission Irreversibility")
    print("=" * 70)

    # Create a nascent node
    G, node = create_nfr("genesis", epi=0.2, vf=0.9)
    print(f"\n✓ Created node '{node}' with EPI=0.2, νf=0.9")

    # Apply emission with valid TNFR sequence
    print("\n→ Applying AL → EN → IL → SHA sequence...")
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify irreversibility markers
    timestamp = get_attr_str(G.nodes[node], ALIAS_EMISSION_TIMESTAMP, default=None)
    activated = G.nodes[node]["_emission_activated"]
    origin = G.nodes[node]["_emission_origin"]
    lineage = G.nodes[node]["_structural_lineage"]

    print("\n✓ Emission completed - Irreversibility markers established:")
    print(f"  • Activation Flag: {activated}")
    print(f"  • Emission Timestamp: {timestamp}")
    print(f"  • Origin (immutable): {origin}")
    print(f"  • Activation Count: {lineage['activation_count']}")
    print(f"  • Derived Nodes: {lineage['derived_nodes']}")

    # Parse and display timestamp details
    emission_time = datetime.fromisoformat(timestamp)
    print(f"\n  Timestamp Details:")
    print(f"  • Format: ISO 8601")
    print(f"  • Timezone: {emission_time.tzinfo}")
    print(f"  • Human-readable: {emission_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    return G, node


def demo_emission_reactivation():
    """Demonstrate re-activation behavior."""
    print("\n" + "=" * 70)
    print("DEMO 2: Emission Re-activation")
    print("=" * 70)

    # Create and activate node
    G, node = create_nfr("reactivation_test", epi=0.25, vf=1.0)
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    original_timestamp = G.nodes[node]["_emission_origin"]
    original_count = G.nodes[node]["_structural_lineage"]["activation_count"]

    print(f"\n✓ First activation at: {original_timestamp}")
    print(f"  • Activation count: {original_count}")

    # Re-apply emission
    print("\n→ Re-applying AL operator...")
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    new_timestamp = G.nodes[node]["_emission_origin"]
    new_count = G.nodes[node]["_structural_lineage"]["activation_count"]

    print(f"\n✓ Re-activation completed:")
    print(f"  • Original timestamp preserved: {new_timestamp == original_timestamp}")
    print(f"  • Activation count incremented: {original_count} → {new_count}")

    # Third activation
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    final_count = G.nodes[node]["_structural_lineage"]["activation_count"]

    print(f"  • After 3rd activation: count = {final_count}")

    return G, node


def demo_multiple_nodes_genealogy():
    """Demonstrate independent timestamps for multiple nodes."""
    print("\n" + "=" * 70)
    print("DEMO 3: Multiple Nodes - Independent Genealogy")
    print("=" * 70)

    # Create first node
    G, node1 = create_nfr("node_alpha", epi=0.3, vf=1.2)
    run_sequence(G, node1, [Emission(), Reception(), Coherence(), Silence()])

    ts1 = get_attr_str(G.nodes[node1], ALIAS_EMISSION_TIMESTAMP, default=None)
    print(f"\n✓ Node 'alpha' activated at: {ts1}")

    # Small delay
    time.sleep(0.05)

    # Create second node
    G.add_node("node_beta")
    G.nodes["node_beta"][EPI_PRIMARY] = 0.4
    G.nodes["node_beta"][VF_PRIMARY] = 0.8
    G.nodes["node_beta"]["theta"] = 0.0
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    run_sequence(G, "node_beta", [Emission(), Reception(), Coherence(), Silence()])

    ts2 = get_attr_str(G.nodes["node_beta"], ALIAS_EMISSION_TIMESTAMP, default=None)
    print(f"✓ Node 'beta' activated at:  {ts2}")

    # Verify independence
    print(f"\n✓ Timestamps are independent: {ts1 != ts2}")
    print(f"  • Delta: ~{(datetime.fromisoformat(ts2) - datetime.fromisoformat(ts1)).total_seconds():.3f} seconds")

    # Display lineage for both nodes
    print("\n✓ Structural lineages:")
    for node_name in [node1, "node_beta"]:
        lineage = G.nodes[node_name]["_structural_lineage"]
        print(f"  • {node_name}:")
        print(f"    - Origin: {lineage['origin'][:26]}...")
        print(f"    - Activation count: {lineage['activation_count']}")
        print(f"    - Parent: {lineage['parent_emission']}")

    return G


def demo_backward_compatibility():
    """Demonstrate backward compatibility with legacy nodes."""
    print("\n" + "=" * 70)
    print("DEMO 4: Backward Compatibility")
    print("=" * 70)

    # Create node
    G, node = create_nfr("legacy_sim", epi=0.2, vf=1.0)

    # Simulate legacy node (remove any emission metadata)
    print("\n→ Simulating legacy node (pre-feature)...")
    for key in ["_emission_activated", "_emission_origin", "_structural_lineage"]:
        if key in G.nodes[node]:
            del G.nodes[node][key]

    has_metadata = "_emission_activated" in G.nodes[node]
    print(f"  • Has emission metadata: {has_metadata}")

    # Apply emission to legacy node
    print("\n→ Applying AL to legacy node...")
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    # Verify metadata was added
    has_metadata_after = "_emission_activated" in G.nodes[node]
    print(f"\n✓ Legacy node upgraded successfully:")
    print(f"  • Has emission metadata: {has_metadata_after}")
    print(f"  • Timestamp: {G.nodes[node].get('emission_timestamp', 'N/A')[:26]}...")
    print(f"  • Lineage initialized: {G.nodes[node].get('_structural_lineage') is not None}")

    return G, node


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║  TNFR Structural Irreversibility Demonstration (AL Operator)     ║")
    print("║  Implementation of TNFR.pdf §2.2.1 - Emisión fundacional        ║")
    print("╚" + "=" * 68 + "╝")

    # Run demonstrations
    demo_basic_emission_traceability()
    demo_emission_reactivation()
    demo_multiple_nodes_genealogy()
    demo_backward_compatibility()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ All demonstrations completed successfully!")
    print("\nKey Principles Demonstrated:")
    print("  1. Temporal Irreversibility: Timestamps establish 'time zero'")
    print("  2. Activation Tracking: Persistent flags mark AL application")
    print("  3. Genealogical Traceability: Lineage records enable EPI analysis")
    print("  4. Re-activation Support: Multiple AL applications tracked")
    print("  5. Backward Compatibility: Legacy nodes work without errors")
    print("\nStructural Benefits:")
    print("  • Complete traceability of EPI reorganization")
    print("  • Definitive chronology of node activation")
    print("  • Foundation for tracking EPI derivation and emergence")
    print("  • Enhanced debugging and analysis capabilities")
    print("  • Strict alignment with TNFR canonical principles")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
