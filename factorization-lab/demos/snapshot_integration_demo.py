"""
TNFR Partition Snapshot Integration Demo

Demonstrates complete snapshot integration with factorization pipeline.
Shows state capture at key verification points for reproducibility and analysis.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from snapshot_system import (
    NodalState,
    PartitionSnapshotManager,
    PartitionState,
    StructuralFieldSnapshot,
    create_mock_network_topology,
    create_mock_nodal_state,
    create_mock_structural_fields,
)


class MockFactorizationEngine:
    """Mock factorization engine for demonstration."""

    def __init__(self, modulus_n: int, snapshot_manager: PartitionSnapshotManager):
        self.modulus_n = modulus_n
        self.snapshot_manager = snapshot_manager
        self.node_count = 10
        self.partition_strategy = "spectral_paley_snapshot_demo"

        # Initialize system state
        self.nodal_states = [create_mock_nodal_state(i) for i in range(self.node_count)]
        self.structural_fields = create_mock_structural_fields(self.node_count)
        self.network_topology = create_mock_network_topology(self.node_count)

        # Performance tracking
        self.start_time = time.time()
        self.verification_confidence = 0.5
        self.coherence_evolution = []

        print(f"🔧 Initialized factorization engine for n={modulus_n}")

    def run_verification_with_snapshots(self) -> tuple[bool, list[str]]:
        """Run complete verification with snapshot capture at each stage."""

        snapshot_ids = []
        stages = [
            ("initialization", "System initialization and network setup"),
            ("partitioning", "Partition boundary identification"),
            ("spectral_analysis", "Eigenvalue decomposition and clustering"),
            ("verification", "Structural coherence verification"),
            ("completion", "Final result validation and cleanup"),
        ]

        print(
            f"\n🚀 Starting verification for n={self.modulus_n} with snapshot capture"
        )

        for i, (stage_name, description) in enumerate(stages):
            print(f"\n📸 Stage {i+1}/{len(stages)}: {stage_name}")
            print(f"   {description}")

            # Simulate stage processing
            self._simulate_verification_stage(stage_name, i)

            # Capture snapshot
            snapshot_id = self._capture_stage_snapshot(stage_name, i)
            snapshot_ids.append(snapshot_id)

            print(f"   ✅ Snapshot captured: {snapshot_id}")

        # Final analysis
        success = self.verification_confidence > 0.8
        final_coherence = (
            self.coherence_evolution[-1] if self.coherence_evolution else 0
        )

        print(f"\n🎯 Verification {'SUCCEEDED' if success else 'FAILED'}")
        print(f"   Final coherence: {final_coherence:.3f}")
        print(f"   Confidence: {self.verification_confidence:.3f}")

        return success, snapshot_ids

    def _simulate_verification_stage(self, stage_name: str, stage_index: int):
        """Simulate verification stage with realistic state changes."""

        # Simulate different behaviors per stage
        if stage_name == "initialization":
            # Start with lower coherence
            coherence = 0.65
            self._update_nodal_states(coherence_target=coherence)

        elif stage_name == "partitioning":
            # Create partition boundaries, slight coherence improvement
            coherence = 0.72
            self._create_partition_boundaries()
            self._update_nodal_states(coherence_target=coherence)

        elif stage_name == "spectral_analysis":
            # Spectral clustering improves organization
            coherence = 0.81
            self._update_spectral_properties()
            self._update_nodal_states(coherence_target=coherence)

        elif stage_name == "verification":
            # Verification process with slight variation
            coherence = 0.83 + (0.1 if self._is_factorizable() else -0.15)
            self._update_verification_metrics(coherence)
            self._update_nodal_states(coherence_target=coherence)

        elif stage_name == "completion":
            # Final state depends on success
            coherence = self.coherence_evolution[-1] + 0.02
            self._finalize_verification(coherence)
            self._update_nodal_states(coherence_target=coherence)

        # Record coherence evolution
        self.coherence_evolution.append(coherence)

        # Simulate processing time
        time.sleep(0.1)

    def _update_nodal_states(self, coherence_target: float):
        """Update nodal states to reflect current verification stage."""

        for i, node_state in enumerate(self.nodal_states):
            # Adjust structural frequency based on coherence
            node_state.structural_frequency = 0.5 + (coherence_target * 1.2)

            # Update phase relationships
            node_state.phase = (node_state.phase + 0.1) % (2 * 3.14159)

            # Adjust ΔNFR based on coherence (lower is better)
            node_state.dnfr_magnitude = max(0.05, 0.8 - coherence_target)

            # Update stability based on coherence
            node_state.stability_index = min(0.95, coherence_target + 0.1)

            # Set appropriate operator
            if coherence_target > 0.8:
                node_state.last_operator = "coherence"
            elif coherence_target < 0.7:
                node_state.last_operator = "dissonance"
            else:
                node_state.last_operator = "coupling"

            node_state.operator_timestamp = time.time()

    def _create_partition_boundaries(self):
        """Create realistic partition state based on factorization structure."""

        # For demonstration, create two partitions
        mid_point = self.node_count // 2

        partition_1 = PartitionState(
            partition_id=f"partition_1_{self.modulus_n}",
            boundary_nodes=[mid_point - 1, mid_point],
            internal_nodes=list(range(0, mid_point - 1)),
            coherence_ratio=0.78,
            phase_gradient=0.15,
            curvature_delta=0.22,
            coupling_strength=0.72,
            fragmentation_risk=0.12 if self._is_factorizable() else 0.3,
            operator_sequence=["emission", "coupling", "resonance"],
        )

        partition_2 = PartitionState(
            partition_id=f"partition_2_{self.modulus_n}",
            boundary_nodes=[mid_point, mid_point + 1],
            internal_nodes=list(range(mid_point + 1, self.node_count)),
            coherence_ratio=0.74,
            phase_gradient=0.18,
            curvature_delta=0.28,
            coupling_strength=0.68,
            fragmentation_risk=0.15 if self._is_factorizable() else 0.35,
            operator_sequence=["reception", "coupling", "coherence"],
        )

        self.partition_states = [partition_1, partition_2]

    def _update_spectral_properties(self):
        """Update network topology with spectral analysis results."""

        # Simulate eigenvalue changes from spectral clustering
        self.network_topology.algebraic_connectivity = 0.6
        self.network_topology.spectral_gap = 0.35

        # Better clustering after spectral analysis
        self.network_topology.clustering_coefficient = 0.25

        if self._is_factorizable():
            # Factorizable numbers show clearer component separation
            self.network_topology.resonant_components = 2
            self.network_topology.fragmented_regions = 0
        else:
            # Prime numbers remain more integrated
            self.network_topology.resonant_components = 1
            self.network_topology.fragmented_regions = 0

    def _update_verification_metrics(self, coherence: float):
        """Update verification confidence based on structural analysis."""

        if self._is_factorizable():
            # Clear factor structure increases confidence
            self.verification_confidence = 0.85 + (coherence - 0.8) * 2
        else:
            # Prime numbers show lower confidence
            self.verification_confidence = 0.65 + (coherence - 0.8) * 1.5

        # Clamp confidence to valid range
        self.verification_confidence = max(0.0, min(1.0, self.verification_confidence))

    def _finalize_verification(self, coherence: float):
        """Finalize verification state."""

        # Update structural fields for final state
        self.structural_fields.phi_s_global = 1.1 + (coherence - 0.8) * 2
        self.structural_fields.energy_density = 2.2 + coherence * 1.5

        if self._is_factorizable() and self.verification_confidence > 0.8:
            # Successful factorization
            self.structural_fields.topological_charge = 0.4
            self.structural_fields.symmetry_breaking = 0.2
        else:
            # No clear factor structure
            self.structural_fields.topological_charge = 0.1
            self.structural_fields.symmetry_breaking = 0.05

    def _capture_stage_snapshot(self, stage_name: str, stage_index: int) -> str:
        """Capture snapshot of current verification state."""

        # Performance metrics
        elapsed_time = (time.time() - self.start_time) * 1000  # ms
        performance_metrics = {
            "coherence": (
                self.coherence_evolution[-1] if self.coherence_evolution else 0.5
            ),
            "sense_index": 0.6 + (stage_index * 0.05),  # Gradual improvement
            "confidence": self.verification_confidence,
            "dnfr_budget": max(10.0, 100.0 - (stage_index * 20)),  # Budget consumption
            "iterations": stage_index + 1,
            "elapsed_ms": elapsed_time,
            "memory_mb": 35.0 + (stage_index * 5),
            "cpu_percent": 20.0 + (stage_index * 8),
        }

        # Determine candidate factor
        candidate_factor = None
        if (
            stage_name == "completion"
            and self._is_factorizable()
            and self.verification_confidence > 0.8
        ):
            candidate_factor = self._get_candidate_factor()

        # Create snapshot
        return self.snapshot_manager.create_snapshot(
            verification_stage=stage_name,
            modulus_n=self.modulus_n,
            candidate_factor=candidate_factor,
            partition_strategy=self.partition_strategy,
            nodal_states=self.nodal_states,
            partition_states=getattr(self, "partition_states", []),
            structural_fields=self.structural_fields,
            network_topology=self.network_topology,
            performance_metrics=performance_metrics,
        )

    def _is_factorizable(self) -> bool:
        """Check if current modulus is factorizable (for demo purposes)."""

        # Simple factorability check for demo
        test_factors = [7, 11, 13, 17, 19]
        return any(self.modulus_n % factor == 0 for factor in test_factors)

    def _get_candidate_factor(self) -> int:
        """Get candidate factor for successful factorization."""

        test_factors = [7, 11, 13, 17, 19]
        for factor in test_factors:
            if self.modulus_n % factor == 0:
                return factor
        return 7  # Fallback


def demonstrate_snapshot_integration():
    """Run complete snapshot integration demonstration."""

    print("TNFR PARTITION SNAPSHOT INTEGRATION DEMO")
    print("=" * 60)
    print("Demonstrating state capture during factorization verification")

    # Create snapshot manager
    demo_db_path = Path("integration_demo_snapshots.db")
    snapshot_manager = PartitionSnapshotManager(demo_db_path)

    # Test with different numbers (factorizable and prime)
    test_numbers = [77, 91, 143, 89, 97]  # Mix of composites and primes

    all_results = []

    for modulus_n in test_numbers:
        print(f"\n" + "=" * 60)
        print(f"TESTING n = {modulus_n}")
        print("=" * 60)

        # Create factorization engine
        engine = MockFactorizationEngine(modulus_n, snapshot_manager)

        # Run verification with snapshots
        success, snapshot_ids = engine.run_verification_with_snapshots()

        # Analyze trajectory
        analysis = snapshot_manager.analyze_trajectory_coherence(snapshot_ids)

        print(f"\n📊 TRAJECTORY ANALYSIS:")
        print(f"   Coherence trend: {analysis['coherence_trend']:.4f}")
        print(f"   Coherence range: {analysis['coherence_range']}")
        print(f"   Final coherence: {analysis['final_coherence']:.3f}")
        print(f"   Snapshots captured: {len(snapshot_ids)}")

        # Store results
        all_results.append(
            {
                "modulus_n": modulus_n,
                "success": success,
                "snapshot_count": len(snapshot_ids),
                "coherence_trend": analysis["coherence_trend"],
                "final_coherence": analysis["final_coherence"],
                "snapshot_ids": snapshot_ids,
            }
        )

        # Export trajectory for this number
        export_path = Path(f"trajectory_{modulus_n}.json")
        snapshot_manager.export_trajectory_data(snapshot_ids, export_path)
        print(f"   📁 Trajectory exported: {export_path}")

    # Summary analysis
    print(f"\n" + "=" * 60)
    print("INTEGRATION DEMO SUMMARY")
    print("=" * 60)

    successful_factorizations = sum(1 for r in all_results if r["success"])
    total_snapshots = sum(r["snapshot_count"] for r in all_results)

    print(f"Numbers tested: {len(test_numbers)}")
    print(f"Successful factorizations: {successful_factorizations}")
    print(f"Total snapshots captured: {total_snapshots}")
    print(f"Average snapshots per number: {total_snapshots / len(test_numbers):.1f}")

    # Show coherence trend analysis
    positive_trends = sum(1 for r in all_results if r["coherence_trend"] > 0)
    print(f"Numbers with positive coherence trend: {positive_trends}")

    # Best and worst performers
    best_result = max(all_results, key=lambda x: x["final_coherence"])
    worst_result = min(all_results, key=lambda x: x["final_coherence"])

    print(
        f"\n🏆 Best performance: n={best_result['modulus_n']} (C={best_result['final_coherence']:.3f})"
    )
    print(
        f"🔍 Needs improvement: n={worst_result['modulus_n']} (C={worst_result['final_coherence']:.3f})"
    )

    # List all available snapshots
    print(f"\n📸 SNAPSHOT DATABASE CONTENTS:")
    all_snapshots = snapshot_manager.list_snapshots(limit=50)

    by_number = {}
    for snapshot in all_snapshots:
        n = snapshot["modulus_n"]
        if n not in by_number:
            by_number[n] = []
        by_number[n].append(snapshot)

    for n in sorted(by_number.keys()):
        snapshots = by_number[n]
        print(f"   n={n}: {len(snapshots)} snapshots")
        for snapshot in snapshots:
            stage = snapshot["verification_stage"]
            coherence = snapshot["overall_coherence"]
            print(f"     - {stage}: C={coherence:.3f}")

    print(f"\n✅ Integration demo completed successfully!")
    print(f"   Database: {demo_db_path}")
    print(f"   Contains {len(all_snapshots)} total snapshots")

    # Cleanup database for demo
    import os

    if demo_db_path.exists():
        os.remove(demo_db_path)
        print(f"   🧹 Demo database cleaned up")

    return True


if __name__ == "__main__":
    success = demonstrate_snapshot_integration()
    print(f"\nDemo {'completed successfully' if success else 'failed'}")
    exit(0 if success else 1)
