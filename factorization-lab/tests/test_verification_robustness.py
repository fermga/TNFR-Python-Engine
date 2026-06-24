"""
TNFR Verification Robustness Test

Focused testing of TNFR verification criteria robustness.
Tests specific verification thresholds and edge cases.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

# Setup paths
LAB_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_PATH))

from tnfr_factorization.spectral_paley import _TNFR_VERIFICATION_CRITERIA


class VerificationRobustnessTest(unittest.TestCase):
    """Test TNFR verification criteria robustness and edge cases."""

    def test_verification_criteria_values(self):
        """Test that verification criteria have appropriate values for robustness."""

        criteria = _TNFR_VERIFICATION_CRITERIA

        # Test minimum partition flags requirement (should be strict)
        self.assertGreaterEqual(
            criteria["min_partition_flags"],
            3,
            "Should require at least 3 partition conditions",
        )
        self.assertLessEqual(
            criteria["min_partition_flags"],
            6,
            "Should not be too restrictive (max 6 conditions)",
        )

        # Test ΔNFR gain minimum (should require significant reduction)
        self.assertGreaterEqual(
            criteria["dnfr_gain_min"], 0.1, "Should require at least 10% ΔNFR reduction"
        )
        self.assertLessEqual(
            criteria["dnfr_gain_min"],
            0.5,
            "Should not require unrealistic ΔNFR reduction",
        )

        # Test coherence bounds (should be reasonable around 1.0)
        self.assertGreaterEqual(
            criteria["coherence_min"], 0.5, "Coherence minimum should be reasonable"
        )
        self.assertLessEqual(
            criteria["coherence_max"], 2.0, "Coherence maximum should be reasonable"
        )
        self.assertLess(
            criteria["coherence_min"],
            criteria["coherence_max"],
            "Coherence min should be less than max",
        )

        # Test structural potential deviation limit
        self.assertGreaterEqual(
            criteria["phi_delta_max"], 0.2, "Phi delta should allow some deviation"
        )
        self.assertLessEqual(
            criteria["phi_delta_max"], 0.8, "Phi delta should not be too permissive"
        )

        # Test gradient and curvature limits
        self.assertGreaterEqual(
            criteria["gradient_delta_max"],
            0.2,
            "Gradient delta should allow reasonable variation",
        )
        self.assertLessEqual(
            criteria["gradient_delta_max"],
            0.8,
            "Gradient delta should not be too permissive",
        )

        self.assertGreaterEqual(
            criteria["curvature_delta_max"],
            0.2,
            "Curvature delta should allow reasonable variation",
        )
        self.assertLessEqual(
            criteria["curvature_delta_max"],
            0.8,
            "Curvature delta should not be too permissive",
        )

        # Test periodicity confidence minimum
        self.assertGreaterEqual(
            criteria["periodicity_confidence_min"],
            0.3,
            "Should require reasonable periodicity confidence",
        )
        self.assertLessEqual(
            criteria["periodicity_confidence_min"],
            0.8,
            "Should not require unrealistic periodicity confidence",
        )

        # Test partition ratio requirements
        self.assertGreaterEqual(
            criteria["required_partition_ratio"],
            0.2,
            "Should require substantial partition endorsement",
        )
        self.assertLessEqual(
            criteria["required_partition_ratio"],
            0.8,
            "Should not require excessive partition endorsement",
        )

        # Test stabilization requirements
        self.assertGreaterEqual(
            criteria["min_stabilized_fraction"],
            0.1,
            "Should require some partition stabilization",
        )
        self.assertLessEqual(
            criteria["min_stabilized_fraction"],
            0.8,
            "Should not require excessive stabilization",
        )

        # Test coverage requirements
        self.assertGreaterEqual(
            criteria["min_coverage_fraction"],
            0.05,
            "Should require minimum modulus coverage",
        )
        self.assertLessEqual(
            criteria["min_coverage_fraction"],
            0.5,
            "Should not require excessive coverage",
        )

        print("✓ All verification criteria values within appropriate ranges")

    def test_criteria_interactions(self):
        """Test interactions between different verification criteria."""

        criteria = _TNFR_VERIFICATION_CRITERIA

        # Test that the combination of requirements is balanced
        # High minimum flags + high confidence should be balanced by reasonable thresholds
        strictness_score = (
            criteria["min_partition_flags"] / 6.0  # Normalize to [0,1]
            + criteria["dnfr_gain_min"] / 0.5
            + criteria["periodicity_confidence_min"] / 0.8
            + criteria["required_partition_ratio"] / 0.8
        ) / 4.0

        # Strictness should be moderate (not too lenient, not too harsh)
        self.assertGreaterEqual(
            strictness_score, 0.3, "Verification should be reasonably strict"
        )
        self.assertLessEqual(
            strictness_score, 0.8, "Verification should not be overly harsh"
        )

        print(f"✓ Verification strictness score: {strictness_score:.3f} (balanced)")

    def test_false_positive_resistance_parameters(self):
        """Test parameters specifically designed to resist false positives."""

        criteria = _TNFR_VERIFICATION_CRITERIA

        # Multiple flag requirement helps prevent single spurious condition
        self.assertGreaterEqual(
            criteria["min_partition_flags"],
            3,
            "Multiple flags prevent single spurious matches",
        )

        # ΔNFR gain requirement ensures structural improvement
        self.assertGreaterEqual(
            criteria["dnfr_gain_min"], 0.15, "ΔNFR gain prevents weak correlations"
        )

        # Periodicity confidence prevents accidental patterns
        self.assertGreaterEqual(
            criteria["periodicity_confidence_min"],
            0.5,
            "High confidence prevents accidental periodicity",
        )

        # Partition ratio prevents isolated spurious partitions
        self.assertGreaterEqual(
            criteria["required_partition_ratio"],
            0.3,
            "Partition ratio prevents isolated false matches",
        )

        print("✓ False-positive resistance parameters validated")

    def test_edge_case_boundary_conditions(self):
        """Test boundary conditions that might lead to edge case failures."""

        criteria = _TNFR_VERIFICATION_CRITERIA

        # Test zero and negative boundary handling
        self.assertGreater(criteria["dnfr_gain_min"], 0, "ΔNFR gain should be positive")
        self.assertGreater(
            criteria["periodicity_confidence_min"],
            0,
            "Periodicity confidence should be positive",
        )
        self.assertGreater(
            criteria["required_partition_ratio"],
            0,
            "Partition ratio should be positive",
        )

        # Test upper bound sanity
        self.assertLess(
            criteria["required_partition_ratio"],
            1,
            "Partition ratio should be less than 100%",
        )
        self.assertLess(
            criteria["min_stabilized_fraction"],
            1,
            "Stabilized fraction should be less than 100%",
        )
        self.assertLess(
            criteria["min_coverage_fraction"],
            1,
            "Coverage fraction should be less than 100%",
        )

        print("✓ Boundary condition robustness validated")

    def test_criteria_consistency(self):
        """Test internal consistency of verification criteria."""

        criteria = _TNFR_VERIFICATION_CRITERIA

        # Ensure coherence bounds are consistent
        self.assertLess(
            criteria["coherence_min"],
            criteria["coherence_max"],
            "Coherence bounds should be properly ordered",
        )

        # Ensure minimum endorsements is reasonable relative to partition requirements
        min_endorsements = criteria["min_endorsements"]
        self.assertGreaterEqual(
            min_endorsements, 1, "Should require at least 1 endorsement"
        )

        # Verify that criteria work together logically
        # If we require high partition ratio, we shouldn't require too many absolute endorsements
        if criteria["required_partition_ratio"] > 0.5:
            self.assertLessEqual(
                criteria["min_endorsements"],
                3,
                "High ratio requirement should not need many absolute endorsements",
            )

        print("✓ Verification criteria consistency validated")


def run_robustness_tests():
    """Run verification robustness tests."""

    print("TNFR VERIFICATION ROBUSTNESS TESTS")
    print("=" * 45)

    # Create and run test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(VerificationRobustnessTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("✓ All robustness tests passed")
        return True
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_robustness_tests()
    sys.exit(0 if success else 1)
