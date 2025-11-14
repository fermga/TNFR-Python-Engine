#!/usr/bin/env python3
"""Interactive CLI validator for TNFR operator sequences.

Provides a user-friendly terminal interface for validating, analyzing,
optimizing, and exploring TNFR operator sequences without requiring
programming knowledge.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..operators.grammar import SequenceValidationResult
    from ..operators.health_analyzer import SequenceHealthMetrics
    from ..tools.sequence_generator import GenerationResult

from ..operators.grammar import validate_sequence_with_health
from ..operators.health_analyzer import SequenceHealthAnalyzer
from ..tools.sequence_generator import ContextualSequenceGenerator
from ..tools.domain_templates import list_domains, list_objectives

__all__ = ["TNFRInteractiveValidator", "run_interactive_validator"]


class TNFRInteractiveValidator:
    """Interactive validator for TNFR operator sequences.

    Provides a conversational interface for users to validate, generate,
    optimize, and explore TNFR operator sequences with real-time feedback
    and visual health metrics.

    Examples
    --------
    >>> validator = TNFRInteractiveValidator()
    >>> validator.run_interactive_session()
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the interactive validator.

        Parameters
        ----------
        seed : int, optional
            Random seed for deterministic sequence generation.
        """
        self.generator = ContextualSequenceGenerator(seed=seed)
        self.analyzer = SequenceHealthAnalyzer()
        self.running = True

    def run_interactive_session(self) -> None:
        """Run the main interactive session with menu navigation."""
        self._show_welcome()

        while self.running:
            try:
                choice = self._show_main_menu()

                if choice == "v":
                    self._interactive_validate()
                elif choice == "g":
                    self._interactive_generate()
                elif choice == "o":
                    self._interactive_optimize()
                elif choice == "e":
                    self._interactive_explore()
                elif choice == "h":
                    self._show_help()
                elif choice == "q":
                    self.running = False
                    print("\nThank you for using TNFR Interactive Validator!")
                else:
                    print(f"\n⚠ Invalid choice: '{choice}'. Please try again.\n")

            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted. Returning to main menu...\n")
            except EOFError:
                print("\n\nGoodbye!")
                self.running = False

    def _show_welcome(self) -> None:
        """Display welcome banner."""
        print()
        print("┌" + "─" * 58 + "┐")
        print("│" + " " * 10 + "TNFR Interactive Sequence Validator" + " " * 13 + "│")
        print("│" + " " * 15 + "Grammar 2.0 - Full Capabilities" + " " * 12 + "│")
        print("└" + "─" * 58 + "┘")
        print()

    def _show_main_menu(self) -> str:
        """Show main menu and get user choice.

        Returns
        -------
        str
            User's menu choice.
        """
        print("Main Menu:")
        print("  [v] Validate a sequence")
        print("  [g] Generate new sequence")
        print("  [o] Optimize existing sequence")
        print("  [e] Explore patterns and domains")
        print("  [h] Help and documentation")
        print("  [q] Quit")
        print()

        choice = input("Select option: ").strip().lower()
        return choice

    def _interactive_validate(self) -> None:
        """Interactive sequence validation with visual feedback."""
        print("\n" + "─" * 60)
        print("VALIDATE SEQUENCE")
        print("─" * 60)
        print("Enter operators separated by spaces or commas.")
        print("Example: emission reception coherence silence")
        print()

        sequence_input = input("Sequence: ").strip()
        if not sequence_input:
            print("⚠ Empty sequence. Returning to menu.\n")
            return

        # Parse sequence (handle both space and comma separation)
        sequence = self._parse_sequence_input(sequence_input)

        try:
            result = validate_sequence_with_health(sequence)

            if result.passed:
                self._display_success(result, sequence)

                # Suggest improvements if health is moderate
                if result.health_metrics and result.health_metrics.overall_health < 0.8:
                    self._suggest_improvements(sequence, result.health_metrics)
            else:
                self._display_error(result)
                self._suggest_fixes(sequence, result.error)

        except Exception as e:
            self._display_exception(e)

        print()

    def _interactive_generate(self) -> None:
        """Interactive sequence generation with guided menus."""
        print("\n" + "─" * 60)
        print("GENERATE SEQUENCE")
        print("─" * 60)
        print()

        # Ask generation mode
        print("Generation mode:")
        print("  [d] By domain and objective")
        print("  [p] By structural pattern")
        print("  [b] Back to main menu")
        print()

        mode = input("Select mode: ").strip().lower()

        if mode == "b":
            return
        elif mode == "d":
            self._generate_by_domain()
        elif mode == "p":
            self._generate_by_pattern()
        else:
            print(f"⚠ Invalid mode: '{mode}'\n")

    def _generate_by_domain(self) -> None:
        """Generate sequence by selecting domain and objective."""
        # Select domain
        domains = list_domains()
        print("\nAvailable domains:")
        for i, domain in enumerate(domains, 1):
            print(f"  {i}. {domain}")
        print()

        try:
            domain_idx = int(input("Select domain (number): ").strip()) - 1
            if domain_idx < 0 or domain_idx >= len(domains):
                print("⚠ Invalid selection.\n")
                return
            domain = domains[domain_idx]
        except (ValueError, EOFError):
            print("⚠ Invalid input.\n")
            return

        # Select objective
        try:
            objectives = list_objectives(domain)
            print(f"\nObjectives for '{domain}':")
            for i, obj in enumerate(objectives, 1):
                print(f"  {i}. {obj}")
            print()

            obj_idx = int(input("Select objective (number, or 0 for any): ").strip())
            if obj_idx == 0:
                objective = None
            else:
                obj_idx -= 1
                if obj_idx < 0 or obj_idx >= len(objectives):
                    print("⚠ Invalid selection.\n")
                    return
                objective = objectives[obj_idx]
        except (ValueError, EOFError):
            print("⚠ Invalid input.\n")
            return

        # Generate
        print("\nGenerating sequence...")
        try:
            result = self.generator.generate_for_context(
                domain=domain, objective=objective, min_health=0.70
            )
            self._display_generated_sequence(result)

            # Offer to analyze
            if self._ask_yes_no("\nAnalyze this sequence in detail?"):
                self._analyze_sequence(result.sequence)

        except Exception as e:
            print(f"✗ Generation failed: {e}\n")

    def _generate_by_pattern(self) -> None:
        """Generate sequence by selecting structural pattern."""
        print("\nCommon structural patterns:")
        patterns = [
            "BOOTSTRAP",
            "THERAPEUTIC",
            "STABILIZE",
            "REGENERATIVE",
            "EXPLORATION",
            "TRANSFORMATIVE",
            "COUPLING",
            "SIMPLE",
        ]
        for i, pattern in enumerate(patterns, 1):
            print(f"  {i}. {pattern}")
        print()

        try:
            pattern_idx = int(input("Select pattern (number): ").strip()) - 1
            if pattern_idx < 0 or pattern_idx >= len(patterns):
                print("⚠ Invalid selection.\n")
                return
            pattern = patterns[pattern_idx]
        except (ValueError, EOFError):
            print("⚠ Invalid input.\n")
            return

        # Generate
        print(f"\nGenerating {pattern} sequence...")
        try:
            result = self.generator.generate_for_pattern(target_pattern=pattern, min_health=0.70)
            self._display_generated_sequence(result)

        except Exception as e:
            print(f"✗ Generation failed: {e}\n")

    def _interactive_optimize(self) -> None:
        """Interactive sequence optimization."""
        print("\n" + "─" * 60)
        print("OPTIMIZE SEQUENCE")
        print("─" * 60)
        print("Enter the sequence you want to improve.")
        print()

        sequence_input = input("Current sequence: ").strip()
        if not sequence_input:
            print("⚠ Empty sequence. Returning to menu.\n")
            return

        current = self._parse_sequence_input(sequence_input)

        # Ask for target health
        try:
            target_input = input("Target health score (0.0-1.0, or Enter for default): ").strip()
            target_health = float(target_input) if target_input else None
        except ValueError:
            print("⚠ Invalid health score. Using default.\n")
            target_health = None

        print("\nOptimizing...")
        try:
            improved, recommendations = self.generator.improve_sequence(
                current, target_health=target_health
            )

            # Show results
            current_health = self.analyzer.analyze_health(current)
            improved_health = self.analyzer.analyze_health(improved)

            self._display_optimization_result(
                current, improved, current_health, improved_health, recommendations
            )

        except Exception as e:
            print(f"✗ Optimization failed: {e}\n")

    def _interactive_explore(self) -> None:
        """Interactive exploration of patterns and domains."""
        print("\n" + "─" * 60)
        print("EXPLORE")
        print("─" * 60)
        print()
        print("  [d] List all domains")
        print("  [o] List objectives for a domain")
        print("  [p] Learn about structural patterns")
        print("  [b] Back to main menu")
        print()

        choice = input("Select option: ").strip().lower()

        if choice == "d":
            self._list_domains()
        elif choice == "o":
            self._list_objectives_for_domain()
        elif choice == "p":
            self._explain_patterns()
        elif choice == "b":
            return
        else:
            print(f"⚠ Invalid choice: '{choice}'\n")

    def _show_help(self) -> None:
        """Show help and documentation."""
        print("\n" + "═" * 60)
        print("HELP & DOCUMENTATION")
        print("═" * 60)
        print()
        print("TNFR (Resonant Fractal Nature Theory) Operators:")
        print()
        print("  emission      - Initiate resonant pattern (AL)")
        print("  reception     - Receive and integrate patterns (EN)")
        print("  coherence     - Stabilize structure (IL)")
        print("  dissonance    - Introduce controlled instability (OZ)")
        print("  coupling      - Create structural links (UM)")
        print("  resonance     - Amplify and propagate (RA)")
        print("  silence       - Freeze evolution temporarily (SHA)")
        print("  expansion     - Increase complexity (VAL)")
        print("  contraction   - Reduce complexity (NUL)")
        print("  self_organization - Spontaneous pattern formation (THOL)")
        print("  mutation      - Phase transformation (ZHIR)")
        print("  transition    - Movement between states (NAV)")
        print("  recursivity   - Nested operations (REMESH)")
        print()
        print("Health Metrics:")
        print()
        print("  Overall Health    - Composite quality score (0.0-1.0)")
        print("  Coherence Index   - Sequential flow quality")
        print("  Balance Score     - Stability/instability equilibrium")
        print("  Sustainability    - Long-term maintenance capacity")
        print()
        print("For more information, visit:")
        print("  https://github.com/fermga/TNFR-Python-Engine")
        print()

    # Helper methods

    def _parse_sequence_input(self, sequence_input: str) -> list[str]:
        """Parse sequence from user input, handling multiple separators."""
        # Replace commas with spaces and split
        sequence_input = sequence_input.replace(",", " ")
        return [op.strip() for op in sequence_input.split() if op.strip()]

    def _display_success(self, result: SequenceValidationResult, sequence: list[str]) -> None:
        """Display successful validation with health metrics."""
        print()
        print("✓ VALID SEQUENCE")
        print()

        if result.health_metrics:
            self._display_health_metrics(result.health_metrics)

    def _display_health_metrics(self, health: SequenceHealthMetrics) -> None:
        """Display health metrics with visual formatting."""
        print("┌─ Health Metrics " + "─" * 41 + "┐")

        # Overall health
        icon = self._health_icon(health.overall_health)
        bar = self._health_bar(health.overall_health)
        status = self._health_status(health.overall_health)
        print(f"│ Overall Health:      {bar} {health.overall_health:.2f} {icon} ({status})")

        # Individual metrics
        print(
            f"│ Coherence Index:     {self._health_bar(health.coherence_index)} {health.coherence_index:.2f}"
        )
        print(
            f"│ Balance Score:       {self._health_bar(health.balance_score)} {health.balance_score:.2f}"
        )
        print(
            f"│ Sustainability:      {self._health_bar(health.sustainability_index)} {health.sustainability_index:.2f}"
        )

        # Pattern
        print(f"│ Pattern Detected:    {health.dominant_pattern.upper()}")
        print(f"│ Sequence Length:     {health.sequence_length}")

        print("└" + "─" * 58 + "┘")

    def _health_bar(self, value: float, width: int = 10) -> str:
        """Generate ASCII bar chart for health metric."""
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    def _health_icon(self, value: float) -> str:
        """Get icon for health value."""
        if value >= 0.8:
            return "✓"
        elif value >= 0.6:
            return "⚠"
        else:
            return "✗"

    def _health_status(self, value: float) -> str:
        """Get status text for health value."""
        if value >= 0.8:
            return "Excellent"
        elif value >= 0.7:
            return "Good"
        elif value >= 0.6:
            return "Moderate"
        else:
            return "Needs Improvement"

    def _display_error(self, result: SequenceValidationResult) -> None:
        """Display validation error with details."""
        print()
        print("✗ INVALID SEQUENCE")
        print()
        logger.error(f" {result.message}")
        if result.error:
            print(f"Type: {type(result.error).__name__}")
        print()

    def _suggest_improvements(self, sequence: list[str], health: SequenceHealthMetrics) -> None:
        """Suggest improvements for moderate health sequences."""
        if not health.recommendations:
            return

        print()
        print("💡 Recommendations:")
        for i, rec in enumerate(health.recommendations[:3], 1):
            print(f"  {i}. {rec}")
        print()

    def _suggest_fixes(self, sequence: list[str], error: Optional[Exception]) -> None:
        """Suggest fixes for validation errors."""
        print("💡 Suggestions:")
        print("  - Check operator spelling (e.g., 'emission' not 'emmision')")
        print("  - Ensure sequence starts with emission or reception")
        print("  - End with a stabilizer (coherence, silence, self_organization)")
        print()

    def _display_exception(self, error: Exception) -> None:
        """Display unexpected exception."""
        print()
        print(f"✗ Unexpected error: {error}")
        print()

    def _display_generated_sequence(self, result: GenerationResult) -> None:
        """Display generated sequence with details."""
        print()
        print("✓ GENERATED SEQUENCE")
        print()
        print(f"Sequence:  {' → '.join(result.sequence)}")
        print(f"Health:    {result.health_score:.2f} {self._health_icon(result.health_score)}")
        print(f"Pattern:   {result.detected_pattern.upper()}")

        if result.domain:
            print(f"Domain:    {result.domain}")
        if result.objective:
            print(f"Objective: {result.objective}")

        if result.recommendations:
            print()
            print("💡 Recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"  {i}. {rec}")
        print()

    def _analyze_sequence(self, sequence: list[str]) -> None:
        """Perform detailed analysis on a sequence."""
        print("\n" + "─" * 60)
        print("DETAILED ANALYSIS")
        print("─" * 60)

        health = self.analyzer.analyze_health(sequence)
        self._display_health_metrics(health)

        if health.recommendations:
            print()
            print("All Recommendations:")
            for i, rec in enumerate(health.recommendations, 1):
                print(f"  {i}. {rec}")
        print()

    def _display_optimization_result(
        self,
        current: list[str],
        improved: list[str],
        current_health: SequenceHealthMetrics,
        improved_health: SequenceHealthMetrics,
        recommendations: list[str],
    ) -> None:
        """Display optimization result with before/after comparison."""
        print()
        print("✓ OPTIMIZATION COMPLETE")
        print()
        print(f"Original:  {' → '.join(current)}")
        print(
            f"  Health:  {current_health.overall_health:.2f} {self._health_icon(current_health.overall_health)}"
        )
        print()
        print(f"Improved:  {' → '.join(improved)}")
        print(
            f"  Health:  {improved_health.overall_health:.2f} {self._health_icon(improved_health.overall_health)}"
        )

        delta = improved_health.overall_health - current_health.overall_health
        if delta > 0:
            print(f"  Delta:   +{delta:.2f} ✓")
        else:
            print(f"  Delta:   {delta:.2f}")

        if recommendations:
            print()
            print("Changes made:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        print()

    def _list_domains(self) -> None:
        """List all available domains."""
        domains = list_domains()
        print()
        print("Available Domains:")
        for domain in domains:
            print(f"  • {domain}")
        print()

    def _list_objectives_for_domain(self) -> None:
        """List objectives for a specific domain."""
        domain = input("\nDomain name: ").strip()
        try:
            objectives = list_objectives(domain)
            print()
            print(f"Objectives for '{domain}':")
            for obj in objectives:
                print(f"  • {obj}")
            print()
        except KeyError:
            print(f"\n⚠ Unknown domain: '{domain}'\n")

    def _explain_patterns(self) -> None:
        """Explain structural patterns."""
        print()
        print("Structural Patterns:")
        print()
        print("  BOOTSTRAP       - Initialize new nodes/systems")
        print("  THERAPEUTIC     - Healing and stabilization")
        print("  STABILIZE       - Maintain coherent structure")
        print("  REGENERATIVE    - Self-renewal and growth")
        print("  EXPLORATION     - Discovery with dissonance")
        print("  TRANSFORMATIVE  - Phase transitions")
        print("  COUPLING        - Network formation")
        print("  SIMPLE          - Minimal effective sequences")
        print()

    def _ask_yes_no(self, prompt: str) -> bool:
        """Ask yes/no question."""
        response = input(f"{prompt} (y/n): ").strip().lower()
        return response in ("y", "yes")


def run_interactive_validator(seed: Optional[int] = None) -> int:
    """Run the interactive validator session.

    Parameters
    ----------
    seed : int, optional
        Random seed for deterministic generation.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    validator = TNFRInteractiveValidator(seed=seed)
    try:
        validator.run_interactive_session()
        return 0
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_interactive_validator())
