"""Tests for organizational domain examples.

Validates that all organizational patterns, case studies, and diagnostic
tools meet acceptance criteria according to TNFR Grammar 2.0.
"""

import pytest
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

# Import organizational examples
import sys
from pathlib import Path

examples_path = Path(__file__).parent.parent.parent / "examples" / "domain_applications"
sys.path.insert(0, str(examples_path))

from organizational_patterns import (
    get_crisis_management_sequence,
    get_strategic_planning_sequence,
    get_team_formation_sequence,
    get_organizational_transformation_sequence,
    get_innovation_cycle_sequence,
    get_change_resistance_resolution_sequence,
)

from organizational_case_studies import (
    case_digital_transformation,
    case_merger_integration,
    case_cultural_change,
    case_innovation_lab,
    case_agile_transformation,
)

from organizational_diagnostics import (
    map_health_to_organizational_kpis,
    detect_structural_dysfunctions,
    recommend_interventions,
    generate_diagnostic_report,
)


# =============================================================================
# Test Organizational Patterns
# =============================================================================


class TestOrganizationalPatterns:
    """Test suite for organizational_patterns.py."""

    def test_all_patterns_valid(self):
        """Test that all organizational patterns pass validation."""
        patterns = {
            "crisis_management": get_crisis_management_sequence(),
            "strategic_planning": get_strategic_planning_sequence(),
            "team_formation": get_team_formation_sequence(),
            "organizational_transformation": get_organizational_transformation_sequence(),
            "innovation_cycle": get_innovation_cycle_sequence(),
            "change_resistance_resolution": get_change_resistance_resolution_sequence(),
        }

        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation: {result.message}"

    def test_all_health_scores_above_threshold(self):
        """Test that all patterns have health scores > 0.75."""
        patterns = {
            "crisis_management": get_crisis_management_sequence(),
            "strategic_planning": get_strategic_planning_sequence(),
            "team_formation": get_team_formation_sequence(),
            "organizational_transformation": get_organizational_transformation_sequence(),
            "innovation_cycle": get_innovation_cycle_sequence(),
            "change_resistance_resolution": get_change_resistance_resolution_sequence(),
        }

        threshold = 0.75

        for name, sequence in patterns.items():
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} did not pass validation"

            health = result.health_metrics.overall_health
            assert (
                health >= threshold
            ), f"{name} health score {health:.3f} below threshold {threshold}"

    def test_minimum_pattern_count(self):
        """Test that we have at least 6 specialized organizational patterns."""
        patterns = [
            get_crisis_management_sequence(),
            get_strategic_planning_sequence(),
            get_team_formation_sequence(),
            get_organizational_transformation_sequence(),
            get_innovation_cycle_sequence(),
            get_change_resistance_resolution_sequence(),
        ]
        assert len(patterns) >= 6

    def test_crisis_management_rapid_response(self):
        """Test crisis management pattern has appropriate characteristics."""
        sequence = get_crisis_management_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should have good sustainability for rapid response
        assert health.sustainability_index >= 0.7

        # Should include key operators for crisis
        assert "dissonance" in sequence
        assert "transition" in sequence
        assert "silence" in sequence  # Consolidation

    def test_strategic_planning_completeness(self):
        """Test strategic planning has comprehensive transformation."""
        sequence = get_strategic_planning_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should be comprehensive with self-organization
        assert "self_organization" in sequence
        assert "expansion" in sequence

        # Should be long enough for full planning cycle
        assert len(sequence) >= 10

    def test_team_formation_coupling_strength(self):
        """Test team formation emphasizes coupling."""
        sequence = get_team_formation_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed

        # Should include coupling for team synchronization
        assert "coupling" in sequence

        # Should include transformation operators
        assert "mutation" in sequence or "self_organization" in sequence

    def test_organizational_transformation_comprehensive(self):
        """Test organizational transformation is comprehensive."""
        sequence = get_organizational_transformation_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed
        health = result.health_metrics

        # Should include key organizational operators
        assert "self_organization" in sequence
        assert "recursivity" in sequence
        assert "expansion" in sequence

        # Should be comprehensive (10+ operators)
        assert len(sequence) >= 10

    def test_innovation_cycle_exploration(self):
        """Test innovation cycle includes exploration operators."""
        sequence = get_innovation_cycle_sequence()
        result = validate_sequence_with_health(sequence)

        assert result.passed

        # Should include exploration and ideation
        assert "dissonance" in sequence  # Creative tension
        assert "self_organization" in sequence  # Emergent solutions

        # Should include transition for scaling
        assert "transition" in sequence


# =============================================================================
# Test Organizational Case Studies
# =============================================================================


class TestOrganizationalCaseStudies:
    """Test suite for organizational_case_studies.py."""

    def test_all_case_studies_valid(self):
        """Test that all case study sequences pass validation."""
        cases = {
            "digital_transformation": case_digital_transformation(),
            "merger_integration": case_merger_integration(),
            "cultural_change": case_cultural_change(),
            "innovation_lab": case_innovation_lab(),
            "agile_transformation": case_agile_transformation(),
        }

        for name, case_data in cases.items():
            sequence = case_data["sequence"]
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} failed validation: {result.message}"

    def test_all_case_studies_health_above_threshold(self):
        """Test that all case studies have health scores > 0.75."""
        cases = {
            "digital_transformation": case_digital_transformation(),
            "merger_integration": case_merger_integration(),
            "cultural_change": case_cultural_change(),
            "innovation_lab": case_innovation_lab(),
            "agile_transformation": case_agile_transformation(),
        }

        threshold = 0.75

        for name, case_data in cases.items():
            sequence = case_data["sequence"]
            result = validate_sequence_with_health(sequence)
            assert result.passed, f"{name} did not pass validation"

            health = result.health_metrics.overall_health
            assert (
                health >= threshold
            ), f"{name} health score {health:.3f} below threshold {threshold}"

    def test_minimum_case_count(self):
        """Test that we have at least 5 business case studies."""
        cases = [
            case_digital_transformation(),
            case_merger_integration(),
            case_cultural_change(),
            case_innovation_lab(),
            case_agile_transformation(),
        ]
        assert len(cases) >= 5

    def test_case_studies_business_relevance(self):
        """Test that all case studies have recognizable business context."""
        cases = [
            case_digital_transformation(),
            case_merger_integration(),
            case_cultural_change(),
            case_innovation_lab(),
            case_agile_transformation(),
        ]

        for case_data in cases:
            # Must have business metadata
            assert "name" in case_data
            assert "challenge" in case_data
            assert "transformation_goal" in case_data
            assert "key_operators" in case_data
            assert "timeline_expected" in case_data

            # Must have meaningful content
            assert len(case_data["challenge"]) > 20
            assert len(case_data["transformation_goal"]) > 20

    def test_digital_transformation_self_organization(self):
        """Test digital transformation includes self-organizing teams."""
        case_data = case_digital_transformation()
        sequence = case_data["sequence"]

        assert "self_organization" in sequence
        assert "contraction" in sequence  # Focus after self-organization
        assert "expansion" in sequence  # Scaling

    def test_merger_integration_coupling(self):
        """Test merger integration emphasizes coupling."""
        case_data = case_merger_integration()
        sequence = case_data["sequence"]

        # Must include coupling for integration
        assert "coupling" in sequence

        # Must include dissonance (cultural tensions)
        assert "dissonance" in sequence

        # Must include stabilization
        assert "coherence" in sequence or "silence" in sequence

    def test_cultural_change_mutation(self):
        """Test cultural change includes mutation (phase change)."""
        case_data = case_cultural_change()
        sequence = case_data["sequence"]

        # Cultural transformation requires mutation
        assert "mutation" in sequence

        # Should include recursivity to embed in systems
        assert "recursivity" in sequence


# =============================================================================
# Test Organizational Diagnostics
# =============================================================================


class TestOrganizationalDiagnostics:
    """Test suite for organizational_diagnostics.py."""

    def test_kpi_mapping_completeness(self):
        """Test that KPI mapping returns all expected metrics."""
        sequence = ["emission", "reception", "coherence", "silence"]
        result = validate_sequence_with_health(sequence)

        kpis = map_health_to_organizational_kpis(result.health_metrics)

        # Must include all key KPIs
        assert "strategic_alignment" in kpis
        assert "stability_agility_balance" in kpis
        assert "institutional_resilience" in kpis
        assert "operational_efficiency" in kpis
        assert "overall_health" in kpis

        # Must include ratings
        assert "alignment_rating" in kpis
        assert "balance_rating" in kpis
        assert "resilience_rating" in kpis
        assert "efficiency_rating" in kpis
        assert "health_rating" in kpis

    def test_dysfunction_detection_for_healthy_sequence(self):
        """Test that healthy sequences show no or few dysfunctions."""
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "contraction",
            "coherence",
            "silence",
        ]
        result = validate_sequence_with_health(sequence)

        dysfunctions = detect_structural_dysfunctions(sequence, result.health_metrics)

        # Healthy sequence should have few dysfunctions
        assert len(dysfunctions) <= 2

    def test_dysfunction_detection_for_chaotic_sequence(self):
        """Test that chaotic sequences are detected."""
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "mutation",
            "coherence",
            "dissonance",
            "mutation",
            "transition",
        ]
        result = validate_sequence_with_health(sequence)

        dysfunctions = detect_structural_dysfunctions(sequence, result.health_metrics)

        # Should detect chaos or imbalance
        dysfunction_types = [d["type"] for d in dysfunctions]
        assert any("Chaos" in dtype or "Imbalance" in dtype for dtype in dysfunction_types)

    def test_intervention_recommendations_prioritized(self):
        """Test that interventions are properly prioritized."""
        sequence = ["emission", "reception", "coherence", "dissonance", "mutation", "transition"]
        result = validate_sequence_with_health(sequence)
        dysfunctions = detect_structural_dysfunctions(sequence, result.health_metrics)

        interventions = recommend_interventions(sequence, result.health_metrics, dysfunctions)

        if interventions:
            # Must have priority field
            for intervention in interventions:
                assert "priority" in intervention
                assert intervention["priority"] in ["High", "Medium", "Low"]

            # Should be sorted by priority
            priorities = [i["priority"] for i in interventions]
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            assert all(
                priority_order[priorities[i]] <= priority_order[priorities[i + 1]]
                for i in range(len(priorities) - 1)
            )

    def test_diagnostic_report_generation(self):
        """Test comprehensive diagnostic report generation."""
        sequence = [
            "emission",
            "reception",
            "coherence",
            "dissonance",
            "self_organization",
            "contraction",
            "coherence",
            "transition",
        ]

        report = generate_diagnostic_report(sequence)

        # Must be valid
        assert report["valid"] is True

        # Must include all sections
        assert "overall_health" in report
        assert "kpis" in report
        assert "dysfunctions" in report
        assert "interventions" in report
        assert "summary" in report

        # KPIs must be populated
        assert len(report["kpis"]) > 0

        # Summary must have key fields
        assert "health_status" in report["summary"]
        assert "critical_issues" in report["summary"]

    def test_diagnostic_report_invalid_sequence(self):
        """Test diagnostic report handles invalid sequences."""
        invalid_sequence = ["emission", "silence"]  # Missing required operators

        report = generate_diagnostic_report(invalid_sequence)

        # Should indicate invalid
        assert report["valid"] is False
        assert "error" in report


# =============================================================================
# Integration Tests
# =============================================================================


class TestOrganizationalIntegration:
    """Integration tests across organizational domain examples."""

    def test_all_examples_load_successfully(self):
        """Test that all organizational modules load without errors."""
        # If we got here, all imports succeeded
        assert True

    def test_patterns_and_cases_use_similar_operators(self):
        """Test that patterns and case studies use similar operator sets."""
        from tnfr.config.operator_names import (
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COUPLING,
            SELF_ORGANIZATION,
            TRANSITION,
            EXPANSION,
        )

        # Get all patterns
        patterns = [
            get_crisis_management_sequence(),
            get_strategic_planning_sequence(),
            get_team_formation_sequence(),
        ]

        # Get all case studies
        cases = [
            case_digital_transformation()["sequence"],
            case_merger_integration()["sequence"],
            case_cultural_change()["sequence"],
        ]

        # Flatten and get unique operators
        pattern_ops = set(op for seq in patterns for op in seq)
        case_ops = set(op for seq in cases for op in seq)

        # Should have significant overlap
        overlap = pattern_ops & case_ops
        assert len(overlap) >= 8  # At least 8 operators in common

    def test_diagnostics_work_with_pattern_sequences(self):
        """Test that diagnostics work with organizational patterns."""
        patterns = [
            get_crisis_management_sequence(),
            get_strategic_planning_sequence(),
            get_team_formation_sequence(),
        ]

        for sequence in patterns:
            report = generate_diagnostic_report(sequence)
            assert report["valid"] is True
            assert report["overall_health"] > 0.7

    def test_average_health_meets_threshold(self):
        """Test that average health across all examples meets threshold."""
        all_sequences = []

        # Add patterns
        all_sequences.extend(
            [
                get_crisis_management_sequence(),
                get_strategic_planning_sequence(),
                get_team_formation_sequence(),
                get_organizational_transformation_sequence(),
                get_innovation_cycle_sequence(),
                get_change_resistance_resolution_sequence(),
            ]
        )

        # Add case studies
        all_sequences.extend(
            [
                case_digital_transformation()["sequence"],
                case_merger_integration()["sequence"],
                case_cultural_change()["sequence"],
                case_innovation_lab()["sequence"],
                case_agile_transformation()["sequence"],
            ]
        )

        # Calculate average health
        health_scores = []
        for sequence in all_sequences:
            result = validate_sequence_with_health(sequence)
            if result.passed:
                health_scores.append(result.health_metrics.overall_health)

        avg_health = sum(health_scores) / len(health_scores)

        # Average should be above threshold
        assert avg_health >= 0.75, f"Average health {avg_health:.3f} below 0.75 threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
