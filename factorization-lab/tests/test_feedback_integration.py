"""
Test Suite for TNFR Self-Optimization Feedback Integration

Comprehensive testing of the feedback learning system and
integration with factorization workflows.
"""

import json
import sqlite3

# Import the feedback system components
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

LAB_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_PATH))

from tnfr_factorization.feedback_adapter import (
    FeedbackIntegratedFactorizer,
    create_feedback_integrated_factorizer,
)
from tnfr_factorization.feedback_integration import (
    FeedbackAnalysis,
    OptimizationFeedbackLearner,
    OptimizationStrategy,
    VerificationFeedback,
)


class TestOptimizationFeedbackLearner(unittest.TestCase):
    """Test the core feedback learning functionality."""

    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_feedback.db"
        self.learner = OptimizationFeedbackLearner(self.db_path)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        """Test that database is properly initialized."""

        # Check that database file exists
        self.assertTrue(self.db_path.exists())

        # Check that tables are created
        with sqlite3.connect(self.db_path) as conn:
            tables = conn.execute(
                """
                SELECT name FROM sqlite_master WHERE type='table'
            """
            ).fetchall()

            table_names = [table[0] for table in tables]
            self.assertIn("verification_feedback", table_names)
            self.assertIn("optimization_strategies", table_names)

    def test_record_verification_feedback(self):
        """Test recording verification feedback."""

        feedback = VerificationFeedback(
            n=35,
            modulus=35,
            node_count=10,
            candidate_factor=5,
            was_certified=True,
            verification_score=0.8,
            dnfr_gain=0.2,
            coherence_ratio=0.9,
            phi_delta_parent=0.1,
            gradient_delta=0.15,
            curvature_delta=0.2,
            periodicity_confidence=0.7,
            partition_strategy="default",
            operator_sequence=["emission", "coupling", "coherence", "silence"],
            optimization_budget=10.0,
            runtime_ms=500.0,
            convergence_iterations=5,
            number_type="semiprime",
            factor_pattern="close_factors",
            timestamp=1000000.0,
        )

        # Record feedback
        self.learner.record_verification_feedback(feedback)

        # Verify it was stored
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM verification_feedback").fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row["n"], 35)
            self.assertEqual(row["candidate_factor"], 5)
            self.assertEqual(row["was_certified"], 1)  # SQLite stores as integer
            self.assertEqual(row["number_type"], "semiprime")

    def test_number_classification(self):
        """Test number type classification."""

        # Test different number types
        test_cases = [
            (35, "semiprime"),  # 5 × 7
            (105, "triprime"),  # 3 × 5 × 7
            (49, "prime_power"),  # 7²
            (17, "prime"),
            (60, "highly_composite"),  # 2² × 3 × 5
        ]

        for n, expected_type in test_cases:
            classified_type = self.learner._classify_number_type(n)
            self.assertEqual(
                classified_type,
                expected_type,
                f"Failed for n={n}: got {classified_type}, expected {expected_type}",
            )

    def test_adaptive_strategy_recommendation(self):
        """Test adaptive strategy recommendations."""

        # Add some feedback data
        for i in range(10):
            feedback = VerificationFeedback(
                n=35 + i,
                modulus=35 + i,
                node_count=10,
                candidate_factor=5,
                was_certified=i % 3 == 0,  # 33% success rate
                verification_score=0.7,
                dnfr_gain=0.15,
                coherence_ratio=0.8,
                phi_delta_parent=0.2,
                gradient_delta=0.3,
                curvature_delta=0.25,
                periodicity_confidence=0.6,
                partition_strategy="test_strategy",
                operator_sequence=["emission", "resonance", "coherence", "silence"],
                optimization_budget=8.0,
                runtime_ms=400.0 + i * 10,
                convergence_iterations=3,
                number_type="semiprime",
                factor_pattern="close_factors",
                timestamp=1000000.0 + i,
            )
            self.learner.record_verification_feedback(feedback)

        # Get recommendation
        strategy = self.learner.get_adaptive_strategy_recommendation(77, "semiprime")

        self.assertIsInstance(strategy, OptimizationStrategy)
        self.assertGreater(strategy.confidence, 0.0)
        self.assertIsInstance(strategy.recommended_sequence, list)
        self.assertGreater(len(strategy.recommended_sequence), 0)

    def test_feedback_pattern_analysis(self):
        """Test feedback pattern analysis."""

        # Add diverse feedback data
        strategies = ["strategy_a", "strategy_b", "strategy_c"]

        for i in range(30):
            strategy = strategies[i % 3]
            # Strategy A: 80% success, Strategy B: 50% success, Strategy C: 20% success
            success_rates = {"strategy_a": 0.8, "strategy_b": 0.5, "strategy_c": 0.2}
            was_certified = (i % 10) < (success_rates[strategy] * 10)

            feedback = VerificationFeedback(
                n=100 + i,
                modulus=100 + i,
                node_count=15,
                candidate_factor=None,
                was_certified=was_certified,
                verification_score=0.6 if was_certified else 0.3,
                dnfr_gain=0.2 if was_certified else 0.05,
                coherence_ratio=0.9 if was_certified else 0.4,
                phi_delta_parent=0.1 if was_certified else 0.5,
                gradient_delta=0.2,
                curvature_delta=0.3,
                periodicity_confidence=0.7 if was_certified else 0.2,
                partition_strategy=strategy,
                operator_sequence=["emission", "coupling", "silence"],
                optimization_budget=12.0,
                runtime_ms=300.0 + i * 5,
                convergence_iterations=2,
                number_type="composite",
                factor_pattern="moderate_gap_factors",
                timestamp=2000000.0 + i,
            )
            self.learner.record_verification_feedback(feedback)

        # Analyze patterns
        analysis = self.learner.analyze_feedback_patterns()

        self.assertIsInstance(analysis, FeedbackAnalysis)
        self.assertGreater(len(analysis.success_rate_by_strategy), 0)

        # Strategy A should have highest success rate
        if "strategy_a" in analysis.success_rate_by_strategy:
            self.assertGreater(analysis.success_rate_by_strategy["strategy_a"], 0.7)

        # Should have some recommendations
        self.assertGreater(len(analysis.optimization_recommendations), 0)


class TestFeedbackIntegration(unittest.TestCase):
    """Test the feedback integration with factorizer."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_integration.db"

        # Create mock factorizer
        self.mock_factorizer = Mock()
        self.mock_factorizer.factor = Mock()

        # Create integrated factorizer
        self.integrated_factorizer = FeedbackIntegratedFactorizer(
            self.mock_factorizer, self.db_path
        )

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_factorizer_integration(self):
        """Test basic factorizer integration."""

        # Mock factorization result
        mock_result = Mock()
        mock_result.tnfr_certified_factors = [5, 7]
        mock_result.tnfr_verification = {
            "per_factor_summary": {
                "5": {
                    "average_dnfr_gain": 0.2,
                    "average_coherence_ratio": 0.85,
                    "average_phi_delta": 0.15,
                    "average_gradient_delta": 0.25,
                    "average_curvature_delta": 0.3,
                    "average_periodicity_confidence": 0.65,
                }
            }
        }
        mock_result.modulus = 35
        mock_result.node_count = 12
        mock_result.self_optimization_summary = None

        self.mock_factorizer.factor.return_value = mock_result

        # Test factorization with feedback
        result = self.integrated_factorizer.factor_with_feedback(35)

        # Verify base factorizer was called
        self.mock_factorizer.factor.assert_called_once_with(35)

        # Verify result has feedback metadata
        self.assertTrue(hasattr(result, "feedback_metadata"))

        # Verify feedback was recorded
        performance = (
            self.integrated_factorizer.feedback_learner.get_performance_summary()
        )
        self.assertGreater(performance["total_feedback_records"], 0)

    def test_learning_summary(self):
        """Test learning summary generation."""

        summary = self.integrated_factorizer.get_learning_summary()

        self.assertIn("learning_status", summary)
        self.assertIn("performance_metrics", summary)
        self.assertIn("feedback_analysis", summary)

        # Check learning status
        learning_status = summary["learning_status"]
        self.assertIn("adaptive_strategies_enabled", learning_status)
        self.assertIn("feedback_recording_enabled", learning_status)

    def test_convenience_function(self):
        """Test convenience function for creating integrated factorizer."""

        mock_factorizer = Mock()
        integrated = create_feedback_integrated_factorizer(
            mock_factorizer, self.db_path
        )

        self.assertIsInstance(integrated, FeedbackIntegratedFactorizer)
        self.assertEqual(integrated.base_factorizer, mock_factorizer)


class TestFeedbackDataStructures(unittest.TestCase):
    """Test the feedback data structures."""

    def test_verification_feedback_creation(self):
        """Test VerificationFeedback dataclass creation."""

        feedback = VerificationFeedback(
            n=77,
            modulus=77,
            node_count=8,
            candidate_factor=7,
            was_certified=True,
            verification_score=0.9,
            dnfr_gain=0.25,
            coherence_ratio=0.88,
            phi_delta_parent=0.12,
            gradient_delta=0.18,
            curvature_delta=0.22,
            periodicity_confidence=0.75,
            partition_strategy="optimized",
            operator_sequence=["emission", "coupling", "resonance", "silence"],
            optimization_budget=15.0,
            runtime_ms=650.0,
            convergence_iterations=4,
            number_type="semiprime",
            factor_pattern="moderate_gap_factors",
            timestamp=3000000.0,
        )

        # Test basic properties
        self.assertEqual(feedback.n, 77)
        self.assertEqual(feedback.candidate_factor, 7)
        self.assertTrue(feedback.was_certified)
        self.assertEqual(feedback.number_type, "semiprime")

    def test_optimization_strategy_creation(self):
        """Test OptimizationStrategy dataclass creation."""

        strategy = OptimizationStrategy(
            context_pattern="semiprime_close_factors",
            recommended_sequence=["emission", "resonance", "coherence", "silence"],
            expected_success_rate=0.75,
            avg_runtime_ms=450.0,
            confidence=0.85,
            sample_count=20,
            last_updated=4000000.0,
        )

        # Test properties
        self.assertEqual(strategy.context_pattern, "semiprime_close_factors")
        self.assertEqual(len(strategy.recommended_sequence), 4)
        self.assertAlmostEqual(strategy.expected_success_rate, 0.75)
        self.assertGreater(strategy.confidence, 0.8)


def run_feedback_integration_tests():
    """Run all feedback integration tests."""

    print("TNFR SELF-OPTIMIZATION FEEDBACK INTEGRATION TESTS")
    print("=" * 60)

    # Create test suite
    test_classes = [
        TestOptimizationFeedbackLearner,
        TestFeedbackIntegration,
        TestFeedbackDataStructures,
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL FEEDBACK INTEGRATION TESTS PASSED")
        print("Self-optimization feedback integration system is ready!")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")

        # Print detailed failure info
        for test, traceback in result.failures + result.errors:
            print(f"\n❌ {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_feedback_integration_tests()
    sys.exit(0 if success else 1)
