from __future__ import annotations

"""Tests for the interactive CLI validator."""

import io
from contextlib import redirect_stdout

import pytest

from tnfr.cli.interactive_validator import TNFRInteractiveValidator


class TestInteractiveValidator:
    """Test suite for TNFRInteractiveValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = TNFRInteractiveValidator()
        assert validator.generator is not None
        assert validator.analyzer is not None
        assert validator.running is True

    def test_initialization_with_seed(self):
        """Test validator initialization with deterministic seed."""
        validator = TNFRInteractiveValidator(seed=42)
        assert validator.generator is not None

    def test_parse_sequence_space_separated(self):
        """Test parsing space-separated sequences."""
        validator = TNFRInteractiveValidator()
        seq = validator._parse_sequence_input("emission reception coherence")
        assert seq == ["emission", "reception", "coherence"]

    def test_parse_sequence_comma_separated(self):
        """Test parsing comma-separated sequences."""
        validator = TNFRInteractiveValidator()
        seq = validator._parse_sequence_input("emission,reception,coherence")
        assert seq == ["emission", "reception", "coherence"]

    def test_parse_sequence_mixed_separators(self):
        """Test parsing mixed separator sequences."""
        validator = TNFRInteractiveValidator()
        seq = validator._parse_sequence_input("emission, reception coherence")
        assert seq == ["emission", "reception", "coherence"]

    def test_parse_sequence_with_extra_spaces(self):
        """Test parsing with extra whitespace."""
        validator = TNFRInteractiveValidator()
        seq = validator._parse_sequence_input("  emission   reception  ")
        assert seq == ["emission", "reception"]

    def test_health_icon_excellent(self):
        """Test health icon for excellent scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_icon(0.9) == "✓"
        assert validator._health_icon(0.8) == "✓"

    def test_health_icon_moderate(self):
        """Test health icon for moderate scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_icon(0.7) == "⚠"
        assert validator._health_icon(0.6) == "⚠"

    def test_health_icon_poor(self):
        """Test health icon for poor scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_icon(0.5) == "✗"
        assert validator._health_icon(0.3) == "✗"

    def test_health_status_excellent(self):
        """Test health status text for excellent scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_status(0.85) == "Excellent"

    def test_health_status_good(self):
        """Test health status text for good scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_status(0.75) == "Good"

    def test_health_status_moderate(self):
        """Test health status text for moderate scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_status(0.65) == "Moderate"

    def test_health_status_needs_improvement(self):
        """Test health status text for poor scores."""
        validator = TNFRInteractiveValidator()
        assert validator._health_status(0.5) == "Needs Improvement"

    def test_health_bar_full(self):
        """Test health bar for maximum value."""
        validator = TNFRInteractiveValidator()
        bar = validator._health_bar(1.0, width=10)
        assert bar == "█" * 10

    def test_health_bar_half(self):
        """Test health bar for half value."""
        validator = TNFRInteractiveValidator()
        bar = validator._health_bar(0.5, width=10)
        assert bar == "█" * 5 + "░" * 5

    def test_health_bar_empty(self):
        """Test health bar for zero value."""
        validator = TNFRInteractiveValidator()
        bar = validator._health_bar(0.0, width=10)
        assert bar == "░" * 10

    def test_display_health_metrics(self, capsys):
        """Test health metrics display."""
        from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

        validator = TNFRInteractiveValidator()
        analyzer = SequenceHealthAnalyzer()

        health = analyzer.analyze_health(["emission", "reception", "coherence", "silence"])

        validator._display_health_metrics(health)

        captured = capsys.readouterr()
        assert "Health Metrics" in captured.out
        assert "Overall Health:" in captured.out
        assert "Coherence Index:" in captured.out
        assert "Balance Score:" in captured.out
        assert "Pattern Detected:" in captured.out

    def test_display_generated_sequence(self, capsys):
        """Test generated sequence display."""
        from tnfr.tools.sequence_generator import GenerationResult

        validator = TNFRInteractiveValidator()

        result = GenerationResult(
            sequence=["emission", "coherence", "silence"],
            health_score=0.75,
            detected_pattern="simple",
            domain="therapeutic",
            objective="stabilization",
            method="template",
            recommendations=["Good sequence"],
            metadata={},
        )

        validator._display_generated_sequence(result)

        captured = capsys.readouterr()
        assert "GENERATED SEQUENCE" in captured.out
        assert "emission" in captured.out
        assert "coherence" in captured.out
        assert "0.75" in captured.out

    def test_ask_yes_no_yes(self, monkeypatch):
        """Test yes/no prompt with yes answer."""
        validator = TNFRInteractiveValidator()

        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert validator._ask_yes_no("Continue?") is True

        monkeypatch.setattr("builtins.input", lambda _: "yes")
        assert validator._ask_yes_no("Continue?") is True

    def test_ask_yes_no_no(self, monkeypatch):
        """Test yes/no prompt with no answer."""
        validator = TNFRInteractiveValidator()

        monkeypatch.setattr("builtins.input", lambda _: "n")
        assert validator._ask_yes_no("Continue?") is False

        monkeypatch.setattr("builtins.input", lambda _: "no")
        assert validator._ask_yes_no("Continue?") is False


class TestValidationMode:
    """Test validation mode functionality."""

    def test_validate_valid_sequence(self, monkeypatch, capsys):
        """Test validation of a valid sequence."""
        validator = TNFRInteractiveValidator()

        # Simulate user input
        inputs = iter(["emission reception coherence silence"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        validator._interactive_validate()

        captured = capsys.readouterr()
        assert "VALID SEQUENCE" in captured.out
        assert "Health Metrics" in captured.out

    def test_validate_invalid_sequence(self, monkeypatch, capsys):
        """Test validation of an invalid sequence."""
        validator = TNFRInteractiveValidator()

        # Invalid - silence can't be first
        inputs = iter(["silence emission"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        validator._interactive_validate()

        captured = capsys.readouterr()
        assert "INVALID SEQUENCE" in captured.out or "Error" in captured.out

    def test_validate_empty_sequence(self, monkeypatch, capsys):
        """Test validation with empty input."""
        validator = TNFRInteractiveValidator()

        inputs = iter([""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        validator._interactive_validate()

        captured = capsys.readouterr()
        assert "Empty sequence" in captured.out


class TestGenerationMode:
    """Test generation mode functionality."""

    def test_list_domains(self, capsys):
        """Test listing available domains."""
        validator = TNFRInteractiveValidator()
        validator._list_domains()

        captured = capsys.readouterr()
        assert "Available Domains" in captured.out
        assert "therapeutic" in captured.out.lower()

    def test_list_objectives_valid_domain(self, monkeypatch, capsys):
        """Test listing objectives for valid domain."""
        validator = TNFRInteractiveValidator()

        monkeypatch.setattr("builtins.input", lambda _: "therapeutic")
        validator._list_objectives_for_domain()

        captured = capsys.readouterr()
        assert "Objectives" in captured.out

    def test_list_objectives_invalid_domain(self, monkeypatch, capsys):
        """Test listing objectives for invalid domain."""
        validator = TNFRInteractiveValidator()

        monkeypatch.setattr("builtins.input", lambda _: "nonexistent_domain")
        validator._list_objectives_for_domain()

        captured = capsys.readouterr()
        assert "Unknown domain" in captured.out

    def test_explain_patterns(self, capsys):
        """Test pattern explanation display."""
        validator = TNFRInteractiveValidator()
        validator._explain_patterns()

        captured = capsys.readouterr()
        assert "Structural Patterns" in captured.out
        assert "BOOTSTRAP" in captured.out
        assert "THERAPEUTIC" in captured.out


class TestOptimizationMode:
    """Test optimization mode functionality."""

    def test_display_optimization_result(self, capsys):
        """Test optimization result display."""
        from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

        validator = TNFRInteractiveValidator()
        analyzer = SequenceHealthAnalyzer()

        current = ["emission", "coherence", "silence"]
        improved = ["emission", "reception", "coherence", "silence"]

        current_health = analyzer.analyze_health(current)
        improved_health = analyzer.analyze_health(improved)
        recommendations = ["Added reception for better balance"]

        validator._display_optimization_result(
            current, improved, current_health, improved_health, recommendations
        )

        captured = capsys.readouterr()
        assert "OPTIMIZATION COMPLETE" in captured.out
        assert "Original:" in captured.out
        assert "Improved:" in captured.out
        assert "Delta:" in captured.out


class TestDisplayMethods:
    """Test display helper methods."""

    def test_show_welcome(self, capsys):
        """Test welcome banner display."""
        validator = TNFRInteractiveValidator()
        validator._show_welcome()

        captured = capsys.readouterr()
        assert "TNFR Interactive Sequence Validator" in captured.out
        assert "Grammar 2.0" in captured.out

    def test_show_main_menu(self, monkeypatch):
        """Test main menu display and input."""
        validator = TNFRInteractiveValidator()

        monkeypatch.setattr("builtins.input", lambda _: "v")
        choice = validator._show_main_menu()

        assert choice == "v"

    def test_show_help(self, capsys):
        """Test help display."""
        validator = TNFRInteractiveValidator()
        validator._show_help()

        captured = capsys.readouterr()
        assert "HELP & DOCUMENTATION" in captured.out
        assert "emission" in captured.out
        assert "Health Metrics" in captured.out


class TestErrorHandling:
    """Test error handling in interactive validator."""

    def test_display_exception(self, capsys):
        """Test exception display."""
        validator = TNFRInteractiveValidator()
        error = ValueError("Test error")

        validator._display_exception(error)

        captured = capsys.readouterr()
        assert "Unexpected error" in captured.out
        assert "Test error" in captured.out

    def test_suggest_fixes(self, capsys):
        """Test fix suggestions display."""
        validator = TNFRInteractiveValidator()

        validator._suggest_fixes(["invalid"], None)

        captured = capsys.readouterr()
        assert "Suggestions" in captured.out


def test_run_interactive_validator_success(monkeypatch):
    """Test successful interactive validator run."""
    from tnfr.cli.interactive_validator import run_interactive_validator

    # Mock to quit immediately
    call_count = [0]

    def mock_run_session(self):
        call_count[0] += 1
        self.running = False

    monkeypatch.setattr(TNFRInteractiveValidator, "run_interactive_session", mock_run_session)

    exit_code = run_interactive_validator()
    assert exit_code == 0
    assert call_count[0] == 1


def test_run_interactive_validator_with_seed():
    """Test validator run with seed parameter."""
    from tnfr.cli.interactive_validator import TNFRInteractiveValidator

    validator = TNFRInteractiveValidator(seed=12345)
    assert validator.generator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
