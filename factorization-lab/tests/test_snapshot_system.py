"""
Test Suite for TNFR Partition Invariant Snapshot Schema

Validates snapshot creation, storage, retrieval, and analysis capabilities.
Tests integrity, compression, trajectory analysis, and reproducibility.
"""

import unittest
import tempfile
import time
from pathlib import Path
import json
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from snapshot_system import (
    PartitionSnapshotManager, NodalState, PartitionState, 
    StructuralFieldSnapshot, NetworkTopologySnapshot, VerificationSnapshot,
    SnapshotCompressor, create_mock_nodal_state, create_mock_structural_fields,
    create_mock_network_topology
)


class TestSnapshotDataStructures(unittest.TestCase):
    """Test snapshot data structure integrity."""
    
    def test_nodal_state_creation(self):
        """Test NodalState dataclass creation and validation."""
        
        nodal_state = NodalState(
            node_id=5,
            epi_vector=[0.1, 0.2, 0.3],
            structural_frequency=1.5,
            phase=0.8,
            theta=1.2,
            dnfr_magnitude=0.3,
            coupling_count=4,
            last_operator="coherence",
            operator_timestamp=time.time(),
            stability_index=0.7
        )
        
        self.assertEqual(nodal_state.node_id, 5)
        self.assertEqual(len(nodal_state.epi_vector), 3)
        self.assertGreater(nodal_state.structural_frequency, 0)
        self.assertIn(nodal_state.last_operator, ["coherence", "dissonance", "emission", "reception"])
    
    def test_partition_state_creation(self):
        """Test PartitionState dataclass creation and validation."""
        
        partition_state = PartitionState(
            partition_id="test_partition",
            boundary_nodes=[1, 3, 5],
            internal_nodes=[0, 2, 4, 6],
            coherence_ratio=0.85,
            phase_gradient=0.12,
            curvature_delta=0.25,
            coupling_strength=0.7,
            fragmentation_risk=0.1,
            operator_sequence=["emission", "coupling", "coherence"]
        )
        
        self.assertEqual(partition_state.partition_id, "test_partition")
        self.assertEqual(len(partition_state.boundary_nodes), 3)
        self.assertLessEqual(partition_state.coherence_ratio, 1.0)
        self.assertGreaterEqual(partition_state.fragmentation_risk, 0.0)
    
    def test_structural_field_snapshot_creation(self):
        """Test StructuralFieldSnapshot with TNFR tetrad fields."""
        
        snapshot = StructuralFieldSnapshot(
            phi_s_global=1.2,
            phi_s_distribution=[0.5, 0.6, 0.7],
            phase_gradient_field=[0.1, 0.15, 0.12],
            curvature_field=[0.2, 0.18, 0.22],
            coherence_length=5.0,
            coherence_field=[0.8, 0.75, 0.82],
            energy_density=2.5,
            topological_charge=0.3,
            symmetry_breaking=0.1
        )
        
        # Validate TNFR tetrad components
        self.assertIsInstance(snapshot.phi_s_global, float)  # Φ_s global
        self.assertEqual(len(snapshot.phase_gradient_field), 3)  # |∇φ| field
        self.assertEqual(len(snapshot.curvature_field), 3)  # K_φ field
        self.assertGreater(snapshot.coherence_length, 0)  # ξ_C > 0
        
        # Validate derived invariants
        self.assertGreater(snapshot.energy_density, 0)  # ℰ > 0
        self.assertIsInstance(snapshot.topological_charge, float)  # 𝒬
        self.assertIsInstance(snapshot.symmetry_breaking, float)  # 𝒮


class TestSnapshotCompression(unittest.TestCase):
    """Test snapshot compression and decompression."""
    
    def setUp(self):
        """Set up test data."""
        
        self.nodal_states = [create_mock_nodal_state(i) for i in range(5)]
        self.partition_states = [
            PartitionState(
                partition_id="test_partition",
                boundary_nodes=[1, 2],
                internal_nodes=[0, 3, 4],
                coherence_ratio=0.8,
                phase_gradient=0.1,
                curvature_delta=0.2,
                coupling_strength=0.7,
                fragmentation_risk=0.05,
                operator_sequence=["emission", "coherence"]
            )
        ]
        self.structural_fields = create_mock_structural_fields(5)
        self.network_topology = create_mock_network_topology(5)
        
        self.test_snapshot = VerificationSnapshot(
            snapshot_id="test_snap_001",
            timestamp=time.time(),
            verification_stage="testing",
            modulus_n=35,
            candidate_factor=5,
            partition_strategy="test_strategy",
            nodal_states=self.nodal_states,
            partition_states=self.partition_states,
            structural_fields=self.structural_fields,
            network_topology=self.network_topology,
            overall_coherence=0.85,
            sense_index=0.72,
            verification_confidence=0.9,
            dnfr_budget_remaining=75.0,
            convergence_iterations=3,
            elapsed_time_ms=1200.0,
            memory_usage_mb=32.5,
            cpu_usage_percent=45.2,
            state_hash="test_hash"
        )
    
    def test_compression_roundtrip(self):
        """Test compression and decompression preserve data."""
        
        # Compress
        compressed_data = SnapshotCompressor.compress_snapshot(self.test_snapshot)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)
        
        # Decompress
        recovered_snapshot = SnapshotCompressor.decompress_snapshot(compressed_data)
        
        # Validate recovery
        self.assertEqual(recovered_snapshot.snapshot_id, self.test_snapshot.snapshot_id)
        self.assertEqual(recovered_snapshot.modulus_n, self.test_snapshot.modulus_n)
        self.assertEqual(len(recovered_snapshot.nodal_states), len(self.test_snapshot.nodal_states))
        self.assertAlmostEqual(recovered_snapshot.overall_coherence, self.test_snapshot.overall_coherence)
        
        # Validate nested structures
        self.assertEqual(len(recovered_snapshot.partition_states), 1)
        self.assertEqual(recovered_snapshot.partition_states[0].partition_id, "test_partition")
        self.assertAlmostEqual(recovered_snapshot.structural_fields.phi_s_global, 
                               self.test_snapshot.structural_fields.phi_s_global)
    
    def test_compression_efficiency(self):
        """Test compression achieves reasonable size reduction."""
        
        import pickle
        
        # Raw pickle size
        raw_size = len(pickle.dumps(self.test_snapshot))
        
        # Compressed size
        compressed_data = SnapshotCompressor.compress_snapshot(self.test_snapshot)
        compressed_size = len(compressed_data)
        
        # Should achieve some compression
        compression_ratio = compressed_size / raw_size
        self.assertLess(compression_ratio, 1.0)  # Some compression achieved
        self.assertGreater(compression_ratio, 0.3)  # Not too aggressive to maintain speed


class TestPartitionSnapshotManager(unittest.TestCase):
    """Test snapshot manager functionality."""
    
    def setUp(self):
        """Set up test manager with temporary database."""
        
        self.temp_dir = tempfile.mkdtemp(prefix="test_snapshot_db_")
        self.db_path = Path(self.temp_dir) / "test_snapshots.db"
        self.manager = PartitionSnapshotManager(self.db_path)
        
        # Test data
        self.nodal_states = [create_mock_nodal_state(i) for i in range(10)]
        self.partition_states = [
            PartitionState(
                partition_id="part_A",
                boundary_nodes=[2, 5, 8],
                internal_nodes=[0, 1, 3, 4, 6, 7, 9],
                coherence_ratio=0.82,
                phase_gradient=0.14,
                curvature_delta=0.21,
                coupling_strength=0.68,
                fragmentation_risk=0.08,
                operator_sequence=["emission", "coupling", "resonance", "coherence"]
            )
        ]
        self.structural_fields = create_mock_structural_fields(10)
        self.network_topology = create_mock_network_topology(10)
        self.performance_metrics = {
            "coherence": 0.82,
            "sense_index": 0.75,
            "confidence": 0.88,
            "elapsed_ms": 1500.0,
            "memory_mb": 48.2,
            "cpu_percent": 32.5
        }
    
    def tearDown(self):
        """Clean up test files."""
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_snapshot_creation_and_retrieval(self):
        """Test creating and retrieving snapshots."""
        
        # Create snapshot
        snapshot_id = self.manager.create_snapshot(
            verification_stage="initialization",
            modulus_n=77,
            candidate_factor=None,
            partition_strategy="spectral_paley",
            nodal_states=self.nodal_states,
            partition_states=self.partition_states,
            structural_fields=self.structural_fields,
            network_topology=self.network_topology,
            performance_metrics=self.performance_metrics
        )
        
        self.assertIsInstance(snapshot_id, str)
        self.assertTrue(snapshot_id.startswith("snap_"))
        
        # Retrieve snapshot
        retrieved_snapshot = self.manager.load_snapshot(snapshot_id)
        self.assertIsNotNone(retrieved_snapshot)
        self.assertEqual(retrieved_snapshot.modulus_n, 77)
        self.assertEqual(retrieved_snapshot.verification_stage, "initialization")
        self.assertEqual(len(retrieved_snapshot.nodal_states), 10)
    
    def test_snapshot_listing_and_filtering(self):
        """Test listing snapshots with filtering."""
        
        # Create multiple snapshots
        for i, stage in enumerate(["init", "partition", "verify", "complete"]):
            self.manager.create_snapshot(
                verification_stage=stage,
                modulus_n=77 + i,
                candidate_factor=7 if stage == "complete" else None,
                partition_strategy="spectral_paley",
                nodal_states=self.nodal_states,
                partition_states=self.partition_states,
                structural_fields=self.structural_fields,
                network_topology=self.network_topology,
                performance_metrics=self.performance_metrics
            )
        
        # List all snapshots
        all_snapshots = self.manager.list_snapshots()
        self.assertEqual(len(all_snapshots), 4)
        
        # Filter by modulus
        filtered_snapshots = self.manager.list_snapshots(modulus_n=79)
        self.assertEqual(len(filtered_snapshots), 1)
        self.assertEqual(filtered_snapshots[0]["verification_stage"], "verify")
        
        # Filter by stage
        stage_snapshots = self.manager.list_snapshots(verification_stage="complete")
        self.assertEqual(len(stage_snapshots), 1)
        self.assertEqual(stage_snapshots[0]["modulus_n"], 80)
    
    def test_trajectory_analysis(self):
        """Test verification trajectory analysis."""
        
        # Create trajectory snapshots
        modulus_n = 91
        strategy = "test_strategy"
        snapshot_ids = []
        
        for i, stage in enumerate(["init", "partition", "verify", "complete"]):
            # Simulate coherence improvement
            metrics = self.performance_metrics.copy()
            metrics["coherence"] = 0.7 + (i * 0.05)  # Increasing coherence
            
            snapshot_id = self.manager.create_snapshot(
                verification_stage=stage,
                modulus_n=modulus_n,
                candidate_factor=7 if stage == "complete" else None,
                partition_strategy=strategy,
                nodal_states=self.nodal_states,
                partition_states=self.partition_states,
                structural_fields=self.structural_fields,
                network_topology=self.network_topology,
                performance_metrics=metrics
            )
            snapshot_ids.append(snapshot_id)
        
        # Get trajectory
        trajectory = self.manager.get_verification_trajectory(modulus_n, strategy)
        self.assertEqual(len(trajectory), 4)
        
        # Analyze trajectory
        analysis = self.manager.analyze_trajectory_coherence(trajectory)
        
        self.assertEqual(analysis["trajectory_length"], 4)
        self.assertGreater(analysis["coherence_trend"], 0)  # Positive trend
        self.assertAlmostEqual(analysis["final_coherence"], 0.85, places=2)
        self.assertEqual(analysis["coherence_range"], (0.7, 0.85))
    
    def test_trajectory_export(self):
        """Test trajectory data export."""
        
        # Create test trajectory
        snapshot_ids = []
        for stage in ["init", "verify"]:
            snapshot_id = self.manager.create_snapshot(
                verification_stage=stage,
                modulus_n=35,
                candidate_factor=None,
                partition_strategy="test_export",
                nodal_states=self.nodal_states[:3],  # Smaller for export test
                partition_states=self.partition_states,
                structural_fields=self.structural_fields,
                network_topology=create_mock_network_topology(3),
                performance_metrics=self.performance_metrics
            )
            snapshot_ids.append(snapshot_id)
        
        # Export trajectory
        export_path = Path(self.temp_dir) / "test_export.json"
        success = self.manager.export_trajectory_data(snapshot_ids, export_path)
        
        self.assertTrue(success)
        self.assertTrue(export_path.exists())
        
        # Validate export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertEqual(export_data["metadata"]["trajectory_length"], 2)
        self.assertEqual(export_data["metadata"]["modulus_n"], 35)
        self.assertEqual(len(export_data["snapshots"]), 2)
    
    def test_integrity_verification(self):
        """Test snapshot integrity checking."""
        
        # Create snapshot
        snapshot_id = self.manager.create_snapshot(
            verification_stage="integrity_test",
            modulus_n=143,
            candidate_factor=11,
            partition_strategy="integrity_check",
            nodal_states=self.nodal_states,
            partition_states=self.partition_states,
            structural_fields=self.structural_fields,
            network_topology=self.network_topology,
            performance_metrics=self.performance_metrics
        )
        
        # Retrieve and verify integrity (should succeed)
        snapshot = self.manager.load_snapshot(snapshot_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.modulus_n, 143)
        
        # Test hash computation consistency
        hash1 = self.manager._compute_state_hash(snapshot)
        hash2 = self.manager._compute_state_hash(snapshot)
        self.assertEqual(hash1, hash2)  # Deterministic hashing
    
    def test_cleanup_old_snapshots(self):
        """Test cleanup of old snapshots."""
        
        # Create test snapshots
        for i in range(3):
            self.manager.create_snapshot(
                verification_stage="cleanup_test",
                modulus_n=100 + i,
                candidate_factor=None,
                partition_strategy="cleanup",
                nodal_states=self.nodal_states,
                partition_states=self.partition_states,
                structural_fields=self.structural_fields,
                network_topology=self.network_topology,
                performance_metrics=self.performance_metrics
            )
        
        # Verify snapshots exist
        initial_count = len(self.manager.list_snapshots())
        self.assertEqual(initial_count, 3)
        
        # Cleanup (very old threshold to not delete our test data)
        deleted_count = self.manager.cleanup_old_snapshots(max_age_hours=0.001)  # 3.6 seconds
        
        # Should not delete recent snapshots
        final_count = len(self.manager.list_snapshots())
        self.assertEqual(final_count, 3)  # No deletion for recent snapshots


class TestSnapshotUtilities(unittest.TestCase):
    """Test utility functions for snapshot creation."""
    
    def test_mock_nodal_state_creation(self):
        """Test mock nodal state utility."""
        
        # Coherent state
        coherent_state = create_mock_nodal_state(5, coherent=True)
        self.assertEqual(coherent_state.node_id, 5)
        self.assertGreater(coherent_state.structural_frequency, 1.0)
        self.assertEqual(coherent_state.last_operator, "coherence")
        self.assertGreater(coherent_state.stability_index, 0.5)
        
        # Dissonant state
        dissonant_state = create_mock_nodal_state(7, coherent=False)
        self.assertEqual(dissonant_state.node_id, 7)
        self.assertLess(dissonant_state.structural_frequency, 0.5)
        self.assertEqual(dissonant_state.last_operator, "dissonance")
        self.assertLess(dissonant_state.stability_index, 0.5)
    
    def test_mock_structural_fields_creation(self):
        """Test mock structural fields utility."""
        
        fields = create_mock_structural_fields(8)
        
        # Validate tetrad fields
        self.assertGreater(fields.phi_s_global, 0)  # Φ_s > 0
        self.assertEqual(len(fields.phi_s_distribution), 8)
        self.assertEqual(len(fields.phase_gradient_field), 8)  # |∇φ|
        self.assertEqual(len(fields.curvature_field), 8)  # K_φ
        self.assertGreater(fields.coherence_length, 0)  # ξ_C > 0
        
        # Validate invariants
        self.assertGreater(fields.energy_density, 0)  # ℰ > 0
        self.assertIsInstance(fields.topological_charge, float)
        self.assertIsInstance(fields.symmetry_breaking, float)
    
    def test_mock_network_topology_creation(self):
        """Test mock network topology utility."""
        
        topology = create_mock_network_topology(6)
        
        self.assertEqual(topology.node_count, 6)
        self.assertEqual(len(topology.adjacency_matrix), 6)
        self.assertEqual(len(topology.adjacency_matrix[0]), 6)
        self.assertGreater(topology.edge_count, 0)
        
        # Ring topology properties
        self.assertEqual(topology.clustering_coefficient, 0.0)  # No triangles in ring
        self.assertGreater(topology.characteristic_path_length, 0)
        self.assertEqual(topology.resonant_components, 1)  # Single component


def run_snapshot_tests():
    """Run all snapshot system tests."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotDataStructures))
    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotCompression))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionSnapshotManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotUtilities))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n" + "="*60)
    print(f"SNAPSHOT SYSTEM TEST SUMMARY")
    print(f"="*60)
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Success rate: {((tests_run - failures - errors) / tests_run * 100):.1f}%")
    
    if failures == 0 and errors == 0:
        print("✅ All snapshot system tests passed!")
        return True
    else:
        print("❌ Some tests failed - check output above")
        return False


if __name__ == "__main__":
    success = run_snapshot_tests()
    exit(0 if success else 1)