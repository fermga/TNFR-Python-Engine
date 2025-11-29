"""
TNFR Repository Test Suite Validation
=====================================

Comprehensive validation that all test categories are present and functional.
Ensures complete test coverage across the TNFR repository.

This meta-test validates the test infrastructure itself.
"""

import os
import pytest
from pathlib import Path


def find_test_files(test_dir: str) -> list:
    """Find all test files in a directory."""
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    return test_files


class TestRepositoryTestSuite:
    """Validate complete test suite structure and coverage."""
    
    def test_grammar_physics_tests_present(self):
        """Verify grammar physics tests exist."""
        grammar_tests = find_test_files('tests/grammar')
        assert len(grammar_tests) > 0, "Grammar physics tests missing"
        
        # Look for specific grammar test files
        grammar_files = [os.path.basename(f) for f in grammar_tests]
        expected_files = ['test_tnfr_grammar_physics.py']
        
        for expected in expected_files:
            assert expected in grammar_files, f"Missing grammar test: {expected}"
        
        print(f"✅ Grammar Physics Tests: {len(grammar_tests)} files found")
    
    def test_canonical_operator_tests_present(self):
        """Verify canonical operator tests exist."""
        operator_tests = find_test_files('tests/operators')
        assert len(operator_tests) > 0, "Canonical operator tests missing"
        
        # Look for modern operator tests
        operator_files = [os.path.basename(f) for f in operator_tests]
        expected_files = ['test_canonical_operators_modern.py']
        
        for expected in expected_files:
            assert expected in operator_files, f"Missing operator test: {expected}"
        
        print(f"✅ Canonical Operator Tests: {len(operator_tests)} files found")
    
    def test_structural_field_tests_present(self):
        """Verify structural field tests exist."""
        physics_tests = find_test_files('tests/physics')
        assert len(physics_tests) > 0, "Structural field tests missing"
        
        # Look for tetrad field tests
        physics_files = [os.path.basename(f) for f in physics_tests]
        expected_files = ['test_tetrad_fields_production.py']
        
        for expected in expected_files:
            assert expected in physics_files, f"Missing physics test: {expected}"
        
        print(f"✅ Structural Field Tests: {len(physics_tests)} files found")
    
    def test_integration_workflow_tests_present(self):
        """Verify integration workflow tests exist."""
        integration_tests = find_test_files('tests/integration')
        assert len(integration_tests) > 0, "Integration workflow tests missing"
        
        # Look for workflow tests
        integration_files = [os.path.basename(f) for f in integration_tests]
        expected_files = ['test_tnfr_workflows.py']
        
        for expected in expected_files:
            assert expected in integration_files, f"Missing integration test: {expected}"
        
        print(f"✅ Integration Workflow Tests: {len(integration_tests)} files found")
    
    def test_performance_regression_tests_present(self):
        """Verify performance and regression tests exist."""
        performance_tests = find_test_files('tests/performance')
        assert len(performance_tests) > 0, "Performance/regression tests missing"
        
        # Look for performance tests
        performance_files = [os.path.basename(f) for f in performance_tests]
        expected_files = ['test_tnfr_performance.py']
        
        for expected in expected_files:
            assert expected in performance_files, f"Missing performance test: {expected}"
        
        print(f"✅ Performance & Regression Tests: {len(performance_tests)} files found")
    
    def test_example_validation_tests_present(self):
        """Verify example validation tests exist (if any)."""
        # This would check for example/showcase validation
        # For now, just verify examples directory exists
        examples_dir = Path('examples')
        if examples_dir.exists():
            example_files = list(examples_dir.glob('*.py'))
            print(f"✅ Examples Available: {len(example_files)} Python files found")
        else:
            print("📝 Examples Directory: Not found (optional)")
    
    def test_all_test_directories_accessible(self):
        """Verify all test directories are accessible."""
        required_dirs = [
            'tests/grammar',
            'tests/operators', 
            'tests/physics',
            'tests/integration',
            'tests/performance'
        ]
        
        missing_dirs = []
        for test_dir in required_dirs:
            if not os.path.exists(test_dir):
                missing_dirs.append(test_dir)
        
        assert len(missing_dirs) == 0, f"Missing test directories: {missing_dirs}"
        
        print(f"✅ Test Directory Structure: All {len(required_dirs)} directories present")
    
    def test_complete_coverage_achieved(self):
        """Verify comprehensive test coverage across all categories."""
        test_categories = {
            'Grammar Physics': 'tests/grammar',
            'Canonical Operators': 'tests/operators',
            'Structural Fields': 'tests/physics', 
            'Integration Workflows': 'tests/integration',
            'Performance & Regression': 'tests/performance'
        }
        
        coverage_summary = {}
        total_test_files = 0
        
        for category, directory in test_categories.items():
            if os.path.exists(directory):
                test_files = find_test_files(directory)
                coverage_summary[category] = len(test_files)
                total_test_files += len(test_files)
            else:
                coverage_summary[category] = 0
        
        # Verify minimum coverage requirements
        assert coverage_summary['Grammar Physics'] >= 1, "Need grammar physics tests"
        assert coverage_summary['Canonical Operators'] >= 1, "Need operator tests"
        assert coverage_summary['Structural Fields'] >= 1, "Need field tests"
        assert coverage_summary['Integration Workflows'] >= 1, "Need integration tests"
        assert coverage_summary['Performance & Regression'] >= 1, "Need performance tests"
        
        print(f"\n🎯 TNFR Test Suite Coverage Summary:")
        for category, count in coverage_summary.items():
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {category}: {count} test file(s)")
        
        print(f"\n📊 Total Test Files: {total_test_files}")
        print(f"📈 Coverage Status: {'COMPLETE' if total_test_files >= 5 else 'INCOMPLETE'}")
        
        # All categories must have at least one test file
        assert all(count > 0 for count in coverage_summary.values()), \
            "All test categories must have at least one test file"


class TestTestInfrastructure:
    """Validate testing infrastructure and dependencies."""
    
    def test_pytest_available(self):
        """Verify pytest is available and functional."""
        import pytest
        assert pytest.__version__, "pytest not properly installed"
        print(f"✅ pytest version: {pytest.__version__}")
    
    def test_numpy_available(self):
        """Verify numpy is available for mathematical tests."""
        import numpy as np
        assert np.__version__, "numpy not properly installed"
        print(f"✅ numpy version: {np.__version__}")
    
    def test_networkx_available(self):
        """Verify NetworkX is available for graph operations."""
        import networkx as nx
        assert nx.__version__, "NetworkX not properly installed"
        print(f"✅ NetworkX version: {nx.__version__}")
    
    def test_tnfr_core_importable(self) -> None:
        """Verify TNFR core modules are importable."""
        try:
            # Test core imports
            from tnfr.alias import set_attr
            from tnfr.constants.aliases import ALIAS_EPI
            from tnfr.operators.definitions import Emission
            from tnfr.metrics.common import compute_coherence
            
            # Use imports to satisfy linter
            assert set_attr is not None
            assert ALIAS_EPI is not None
            assert Emission is not None
            assert compute_coherence is not None
            
            print("✅ TNFR Core Modules: All imports successful")
            
        except ImportError as e:
            pytest.fail(f"TNFR core import failed: {e}")
    
    def test_test_utilities_functional(self) -> None:
        """Verify test utility functions work correctly."""
        import networkx as nx
        from tnfr.alias import set_attr
        from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA
        
        # Create test network
        G = nx.path_graph(5)
        
        # Initialize with test data
        for i, node in enumerate(G.nodes()):
            set_attr(G.nodes[node], ALIAS_EPI, 0.5)
            set_attr(G.nodes[node], ALIAS_VF, 1.0)
            set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(G.nodes[node], ALIAS_THETA, 0.0)
        
        # Verify network is properly initialized
        assert G.number_of_nodes() == 5
        assert all(ALIAS_EPI[0] in node_data for node_data in G.nodes.values())
        
        print("✅ Test Utilities: Network creation and initialization functional")


if __name__ == "__main__":
    # Run repository test validation
    pytest.main([__file__, "-v", "-s", "--tb=short"])
