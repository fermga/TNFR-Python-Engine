"""
Test Suite for TNFR Seed Management System

Validates seed capture, restoration, reproducibility, and experiment context management.
Tests hierarchical seeding, cross-platform compatibility, and integrity verification.
"""

import unittest
import tempfile
import time
import os
import random
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from seed_management import (
    TNFRSeedManager, ExperimentParameters, SystemEnvironment,
    RandomSeedState, ReproducibilityMetadata, create_demo_experiment_params
)


class TestSeedManagerCore(unittest.TestCase):
    """Test core seed manager functionality."""
    
    def setUp(self):
        """Set up test seed manager."""
        self.seed_manager = TNFRSeedManager(master_seed=12345)
        
    def test_deterministic_initialization(self):
        """Test deterministic initialization from master seed."""
        
        # Create two managers with same master seed
        manager1 = TNFRSeedManager(master_seed=999)
        manager2 = TNFRSeedManager(master_seed=999)
        
        # Should have identical derived seeds
        self.assertEqual(manager1.global_network_seed, manager2.global_network_seed)
        self.assertEqual(manager1.node_initialization_seed, manager2.node_initialization_seed)
        self.assertEqual(manager1.coupling_dynamics_seed, manager2.coupling_dynamics_seed)
        self.assertEqual(manager1.spectral_analysis_seed, manager2.spectral_analysis_seed)
        
    def test_master_seed_generation(self):
        """Test master seed generation is reasonable."""
        
        # Test without providing seed
        manager = TNFRSeedManager()
        
        self.assertIsInstance(manager.master_seed, int)
        self.assertGreater(manager.master_seed, 0)
        self.assertLess(manager.master_seed, 2**31)
        
        # Multiple generations should be different
        manager2 = TNFRSeedManager()
        self.assertNotEqual(manager.master_seed, manager2.master_seed)
        
    def test_seeded_random_generators(self):
        """Test seeded random generator creation."""
        
        # Test Python random
        rng1 = self.seed_manager.get_seeded_random("node_initialization")
        rng2 = self.seed_manager.get_seeded_random("node_initialization")
        
        # Should be independent but reproducible
        self.assertEqual(rng1.random(), rng2.random())
        
        # Different seed types should be different
        coupling_rng = self.seed_manager.get_seeded_random("coupling_dynamics")
        self.assertNotEqual(rng1.random(), coupling_rng.random())
        
        # Test NumPy random
        np_rng1 = self.seed_manager.get_seeded_numpy_random("spectral_analysis")
        np_rng2 = self.seed_manager.get_seeded_numpy_random("spectral_analysis")
        
        # Should generate identical sequences
        seq1 = np_rng1.random(5)
        seq2 = np_rng2.random(5)
        np.testing.assert_array_equal(seq1, seq2)
        
    def test_invalid_seed_type(self):
        """Test handling of invalid seed types."""
        
        with self.assertRaises(ValueError):
            self.seed_manager.get_seeded_random("invalid_seed_type")
            
        with self.assertRaises(ValueError):
            self.seed_manager.get_seeded_numpy_random("another_invalid_type")


class TestStateCapture(unittest.TestCase):
    """Test state capture and restoration."""
    
    def setUp(self):
        """Set up test environment."""
        self.seed_manager = TNFRSeedManager(master_seed=54321)
        
    def test_state_capture_completeness(self):
        """Test complete state capture includes all components."""
        
        state = self.seed_manager.capture_complete_state()
        
        # Check required top-level keys
        self.assertIn("environment", state)
        self.assertIn("seed_state", state)
        self.assertIn("capture_timestamp", state)
        self.assertIn("master_seed", state)
        
        # Check environment components
        env = state["environment"]
        self.assertIn("platform_system", env)
        self.assertIn("python_version", env)
        self.assertIn("numpy_version", env)
        self.assertIn("process_id", env)
        
        # Check seed state components
        seeds = state["seed_state"]
        self.assertIn("master_seed", seeds)
        self.assertIn("python_random_state", seeds)
        self.assertIn("numpy_random_state", seeds)
        self.assertIn("node_initialization_seed", seeds)
        self.assertIn("spectral_analysis_seed", seeds)
        
    def test_state_restoration(self):
        """Test state restoration preserves randomness."""
        
        # Capture initial state
        initial_state = self.seed_manager.capture_complete_state()
        
        # Generate some random numbers
        random.seed(self.seed_manager.master_seed)
        original_sequence = [random.random() for _ in range(5)]
        
        # Modify random state
        random.seed(99999)
        modified_sequence = [random.random() for _ in range(5)]
        
        # Should be different
        self.assertNotEqual(original_sequence, modified_sequence)
        
        # Restore state
        success = self.seed_manager.restore_complete_state(initial_state)
        self.assertTrue(success)
        
        # Generate sequence again - should match original
        restored_sequence = [random.random() for _ in range(5)]
        
        # Check if sequences match (allowing small floating point differences)
        for orig, restored in zip(original_sequence, restored_sequence):
            self.assertAlmostEqual(orig, restored, places=15)
    
    def test_state_restoration_failure_handling(self):
        """Test graceful handling of invalid state data."""
        
        # Test with malformed state data
        invalid_states = [
            {},  # Empty dict
            {"master_seed": 123},  # Missing components
            {"seed_state": "not_a_dict"},  # Wrong types
        ]
        
        for invalid_state in invalid_states:
            success = self.seed_manager.restore_complete_state(invalid_state)
            self.assertFalse(success)


class TestExperimentContext(unittest.TestCase):
    """Test experiment context creation and management."""
    
    def setUp(self):
        """Set up test environment."""
        self.seed_manager = TNFRSeedManager(master_seed=11111)
        self.test_params = create_demo_experiment_params(143)
        
        # Use temporary directory for contexts
        self.temp_dir = tempfile.mkdtemp(prefix="test_contexts_")
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
    def tearDown(self):
        """Clean up test files."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_experiment_context_creation(self):
        """Test experiment context creation and storage."""
        
        experiment_id = self.seed_manager.create_experiment_context(self.test_params)
        
        # Should generate valid experiment ID
        self.assertIsInstance(experiment_id, str)
        self.assertTrue(experiment_id.startswith("exp_"))
        self.assertEqual(len(experiment_id), 16)  # exp_ + 12 hex chars
        
        # Should create context file
        context_file = Path("experiment_contexts") / f"{experiment_id}.json"
        self.assertTrue(context_file.exists())
        
        # Should be able to load context
        import json
        with open(context_file, 'r') as f:
            context_data = json.load(f)
            
        self.assertIn("metadata", context_data)
        self.assertIn("parameters", context_data)
        self.assertIn("reproducibility_state", context_data)
        
    def test_experiment_id_uniqueness(self):
        """Test experiment ID generation produces unique IDs."""
        
        # Create multiple experiments
        ids = []
        for i in range(10):
            # Slight parameter variation to ensure uniqueness
            params = create_demo_experiment_params(100 + i)
            exp_id = self.seed_manager.create_experiment_context(params)
            ids.append(exp_id)
        
        # All IDs should be unique
        self.assertEqual(len(ids), len(set(ids)))
        
    def test_experiment_context_loading(self):
        """Test loading experiment context."""
        
        # Create experiment
        experiment_id = self.seed_manager.create_experiment_context(self.test_params)
        
        # Load context
        context_data = self.seed_manager._load_experiment_context(experiment_id)
        
        self.assertIsNotNone(context_data)
        self.assertEqual(context_data["parameters"]["modulus_n"], self.test_params.modulus_n)
        self.assertEqual(context_data["parameters"]["partition_strategy"], self.test_params.partition_strategy)
        
        # Test loading non-existent context
        missing_context = self.seed_manager._load_experiment_context("exp_nonexistent")
        self.assertIsNone(missing_context)


class TestReproducibilityValidation(unittest.TestCase):
    """Test reproducibility validation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.seed_manager = TNFRSeedManager(master_seed=77777)
        self.test_params = create_demo_experiment_params(91)
        
        # Use temporary directory  
        self.temp_dir = tempfile.mkdtemp(prefix="test_repro_")
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
    def tearDown(self):
        """Clean up test files."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_reproducibility_validation_success(self):
        """Test reproducibility validation with consistent results."""
        
        # Create experiment context
        experiment_id = self.seed_manager.create_experiment_context(self.test_params)
        
        # Validate reproducibility
        validation_result = self.seed_manager.validate_reproducibility(experiment_id, test_iterations=3)
        
        self.assertIn("valid", validation_result)
        self.assertIn("test_iterations", validation_result)
        self.assertIn("consistency_score", validation_result)
        self.assertIn("results", validation_result)
        
        # Should be valid and consistent
        self.assertTrue(validation_result["valid"])
        self.assertEqual(validation_result["test_iterations"], 3)
        self.assertAlmostEqual(validation_result["consistency_score"], 1.0, places=6)
        
        # Results should be identical
        results = validation_result["results"]
        self.assertEqual(len(results), 3)
        
        # Check first result structure
        first_result = results[0]
        self.assertIn("final_coherence", first_result)
        self.assertIn("sense_index", first_result)
        self.assertIn("verification_confidence", first_result)
        
        # All results should be identical
        for i in range(1, len(results)):
            for key in first_result:
                self.assertAlmostEqual(first_result[key], results[i][key], places=15)
    
    def test_reproducibility_validation_missing_experiment(self):
        """Test validation with missing experiment context."""
        
        validation_result = self.seed_manager.validate_reproducibility("exp_nonexistent")
        
        self.assertFalse(validation_result["valid"])
        self.assertIn("error", validation_result)


class TestDataStructures(unittest.TestCase):
    """Test data structure creation and validation."""
    
    def test_experiment_parameters_creation(self):
        """Test ExperimentParameters dataclass creation."""
        
        params = create_demo_experiment_params(35)
        
        # Check required fields
        self.assertEqual(params.modulus_n, 35)
        self.assertIsInstance(params.partition_strategy, str)
        self.assertIsInstance(params.node_count, int)
        self.assertGreater(params.node_count, 0)
        
        # Check ranges
        self.assertIsInstance(params.coupling_strength_range, tuple)
        self.assertEqual(len(params.coupling_strength_range), 2)
        self.assertLess(params.coupling_strength_range[0], params.coupling_strength_range[1])
        
        # Check boolean flags
        self.assertIsInstance(params.adaptive_threshold_enabled, bool)
        self.assertIsInstance(params.feedback_learning_enabled, bool)
        
    def test_random_seed_state_structure(self):
        """Test RandomSeedState has all required components."""
        
        # Create mock seed state
        seed_state = RandomSeedState(
            python_random_state=(1, tuple(range(625)), None),
            numpy_random_state={"generator": "MT19937", "state": list(range(625)), "pos": 0, "has_gauss": 0, "cached_gaussian": None},
            master_seed=12345,
            node_initialization_seed=23456,
            coupling_dynamics_seed=34567,
            phase_evolution_seed=45678,
            operator_sequence_seed=56789,
            global_network_seed=67890,
            partition_level_seed=78901,
            node_level_seed=89012,
            spectral_analysis_seed=90123,
            clustering_seed=1234,
            threshold_jitter_seed=5678
        )
        
        # Check all seeds are present and positive
        self.assertGreater(seed_state.master_seed, 0)
        self.assertGreater(seed_state.node_initialization_seed, 0)
        self.assertGreater(seed_state.coupling_dynamics_seed, 0)
        self.assertGreater(seed_state.spectral_analysis_seed, 0)
        
        # Check hierarchical seeds
        self.assertGreater(seed_state.global_network_seed, 0)
        self.assertGreater(seed_state.partition_level_seed, 0)
        self.assertGreater(seed_state.node_level_seed, 0)


class TestHierarchicalSeeding(unittest.TestCase):
    """Test hierarchical seeding functionality."""
    
    def test_seed_hierarchy_independence(self):
        """Test different seed types produce independent sequences."""
        
        seed_manager = TNFRSeedManager(master_seed=33333)
        
        # Get different seeded generators
        node_rng = seed_manager.get_seeded_random("node_initialization")
        coupling_rng = seed_manager.get_seeded_random("coupling_dynamics")
        spectral_rng = seed_manager.get_seeded_numpy_random("spectral_analysis")
        
        # Generate sequences
        node_seq = [node_rng.random() for _ in range(10)]
        coupling_seq = [coupling_rng.random() for _ in range(10)]
        spectral_seq = spectral_rng.random(10).tolist()
        
        # Sequences should be different
        self.assertNotEqual(node_seq, coupling_seq)
        self.assertNotEqual(node_seq, spectral_seq)
        self.assertNotEqual(coupling_seq, spectral_seq)
        
        # But reproducible from same seeds
        node_rng2 = seed_manager.get_seeded_random("node_initialization")
        node_seq2 = [node_rng2.random() for _ in range(10)]
        self.assertEqual(node_seq, node_seq2)
    
    def test_master_seed_determinism(self):
        """Test master seed completely determines all derived seeds."""
        
        master_seed = 88888
        
        # Create multiple managers with same master seed
        managers = [TNFRSeedManager(master_seed=master_seed) for _ in range(3)]
        
        # All derived seeds should be identical
        for i in range(1, len(managers)):
            self.assertEqual(managers[0].global_network_seed, managers[i].global_network_seed)
            self.assertEqual(managers[0].node_initialization_seed, managers[i].node_initialization_seed)
            self.assertEqual(managers[0].coupling_dynamics_seed, managers[i].coupling_dynamics_seed)
            self.assertEqual(managers[0].spectral_analysis_seed, managers[i].spectral_analysis_seed)


def run_seed_management_tests():
    """Run all seed management system tests."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSeedManagerCore))
    suite.addTests(loader.loadTestsFromTestCase(TestStateCapture))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentContext))
    suite.addTests(loader.loadTestsFromTestCase(TestReproducibilityValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataStructures))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalSeeding))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n" + "="*60)
    print(f"SEED MANAGEMENT SYSTEM TEST SUMMARY")
    print(f"="*60)
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Success rate: {((tests_run - failures - errors) / tests_run * 100):.1f}%")
    
    if failures == 0 and errors == 0:
        print("✅ All seed management system tests passed!")
        return True
    else:
        print("❌ Some tests failed - check output above")
        if failures > 0:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        if errors > 0:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
        return False


if __name__ == "__main__":
    success = run_seed_management_tests()
    exit(0 if success else 1)