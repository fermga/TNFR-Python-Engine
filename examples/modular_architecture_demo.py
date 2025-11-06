"""Example demonstrating TNFR modular architecture.

This example shows how to use the new orchestration layer with dependency
injection to execute operator sequences with clean separation of concerns.
"""

from tnfr.core import TNFRContainer
from tnfr.services import TNFROrchestrator
from tnfr.structural import create_nfr
from tnfr.dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed


def main():
    """Demonstrate modular architecture with orchestrator."""

    print("=" * 60)
    print("TNFR Modular Architecture Demo")
    print("=" * 60)

    # 1. Create a graph with a node
    print("\n1. Creating TNFR node...")
    G, node = create_nfr("demo_node", epi=1.0, vf=0.8, theta=0.0)
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)
    print(f"   ✓ Created node '{node}' with EPI=1.0, νf=0.8")

    # 2. Create orchestrator with default services
    print("\n2. Setting up orchestrator with default services...")
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)
    print("   ✓ Orchestrator ready")
    print("     - ValidationService: Default (TNFR grammar)")
    print("     - OperatorRegistry: Default (canonical operators)")
    print("     - DynamicsEngine: Default (ΔNFR + nodal equation)")
    print("     - TelemetryCollector: Default (C(t), Si)")

    # 3. Validate sequence before execution
    print("\n3. Validating operator sequence...")
    sequence = [
        "emission",
        "reception",
        "coherence",
        "coupling",
        "dissonance",
        "resonance",
        "silence",
    ]
    try:
        orchestrator.validate_only(sequence)
        print(f"   ✓ Sequence validated: {len(sequence)} operators")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
        return

    # 4. Get initial metrics
    print("\n4. Initial state metrics...")
    initial_coherence = orchestrator.get_coherence(G)
    initial_si = orchestrator.get_sense_index(G)
    print(f"   Coherence C(t): {initial_coherence:.4f}")
    print(f"   Sense Index Si: {initial_si.get('Si', 'N/A')}")

    # 5. Execute sequence with telemetry
    print("\n5. Executing sequence with telemetry enabled...")
    orchestrator.execute_sequence(G, node, sequence, enable_telemetry=True)
    print("   ✓ Sequence executed")

    # 6. Show final metrics
    print("\n6. Final state metrics...")
    final_coherence = orchestrator.get_coherence(G)
    final_si = orchestrator.get_sense_index(G)
    print(f"   Coherence C(t): {final_coherence:.4f}")
    print(f"   Sense Index Si: {final_si.get('Si', 'N/A')}")
    print(f"   ΔCoherence: {final_coherence - initial_coherence:+.4f}")

    # 7. Show captured transitions
    print("\n7. Telemetry transitions captured...")
    if "_trace_transitions" in G.graph:
        transitions = G.graph["_trace_transitions"]
        print(f"   Total transitions: {len(transitions)}")
        for i, trans in enumerate(transitions[:3], 1):  # Show first 3
            op = trans["operator"]
            delta_c = trans["delta_coherence"]
            print(f"   {i}. {op:12s} → ΔC = {delta_c:+.6f}")
        if len(transitions) > 3:
            print(f"   ... and {len(transitions) - 3} more transitions")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nKey benefits demonstrated:")
    print("  ✓ Clean separation: validation → execution → telemetry")
    print("  ✓ Dependency injection: services configured via container")
    print("  ✓ Detailed telemetry: transitions captured per operator")
    print("  ✓ TNFR invariants: all canonical constraints maintained")


def demo_custom_validator():
    """Demonstrate custom validation service."""
    from tnfr.core.interfaces import ValidationService

    print("\n" + "=" * 60)
    print("Custom Validator Demo")
    print("=" * 60)

    class VerboseValidator:
        """Custom validator with verbose output."""

        def validate_sequence(self, sequence):
            print(f"\n   [VerboseValidator] Checking {len(sequence)} operators...")
            from tnfr.validation import validate_sequence as _validate

            outcome = _validate(sequence)
            if not outcome.passed:
                msg = outcome.summary.get("message", "failed")
                print(f"   [VerboseValidator] ✗ Failed: {msg}")
                raise ValueError(f"Invalid sequence: {msg}")
            print("   [VerboseValidator] ✓ All operators valid")

        def validate_graph_state(self, graph):
            print(f"   [VerboseValidator] Graph has {graph.number_of_nodes()} nodes")

    # Use custom validator
    print("\nCreating orchestrator with custom validator...")
    container = TNFRContainer.create_default()
    container.register_singleton(ValidationService, VerboseValidator())
    orchestrator = TNFROrchestrator.from_container(container)
    print("✓ Custom validator registered")

    # Execute with custom validator
    G, node = create_nfr("test", epi=1.0, vf=0.8)
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    print("\nExecuting sequence...")
    orchestrator.execute_sequence(
        G,
        node,
        ["emission", "reception", "coherence", "coupling", "resonance", "silence"],
    )
    print("\n✓ Execution complete with custom validation")


if __name__ == "__main__":
    main()
    demo_custom_validator()
