"""
Tests for the TNFR Sequence Optimizer.
"""

import pytest
from tnfr.types import Glyph
from tnfr.math import optimizer


# A simple objective function for testing purposes
def simple_length_objective(sequence: optimizer.Sequence) -> float:
    """Rewards longer sequences."""
    return float(len(sequence))


def test_sample_objective_function():
    """Test the provided sample objective function."""
    # A valid sequence
    seq_good = [Glyph.AL, Glyph.UM, Glyph.IL, Glyph.SHA]
    score_good = optimizer.sample_objective_function(seq_good)
    assert score_good > 0

    # An invalid sequence (destabilizer without stabilizer)
    seq_bad_conv = [Glyph.AL, Glyph.OZ]
    score_bad_conv = optimizer.sample_objective_function(seq_bad_conv)
    assert score_bad_conv == -1000.0

    # A risky sequence
    seq_risk = [Glyph.AL, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.SHA]
    # This sequence is also net-destabilizing under default params
    # (IL: -0.15, OZ: +0.1, ZHIR: +0.1 => -0.15 + 0.2 = +0.05)
    # So it should fail the convergence check first.
    score_risk = optimizer.sample_objective_function(seq_risk)
    assert score_risk == -1000.0


class TestGreedyOptimizer:
    """Tests for the find_optimal_sequence_greedy function."""

    def test_finds_longer_sequence_with_simple_objective(self):
        """
        Verify the optimizer finds a longer sequence when the objective
        is simply to maximize length.
        """
        initial_sequence = [Glyph.AL]
        possible_glyphs = [Glyph.UM, Glyph.IL, Glyph.SHA]
        
        best_sequence, best_score = optimizer.find_optimal_sequence_greedy(
            initial_sequence=initial_sequence,
            possible_glyphs=possible_glyphs,
            objective_fn=simple_length_objective,
            max_iterations=3
        )

        assert best_score == 4.0  # Initial (1) + 3 iterations
        assert len(best_sequence) == 4
        # The exact sequence might vary, but it should be longer
        assert best_sequence[0] == Glyph.AL

    def test_stops_when_no_improvement_is_made(self):
        """
        Verify the optimizer terminates if no single addition improves the score.
        """
        # An objective that only rewards a specific glyph
        def specific_glyph_objective(sequence: optimizer.Sequence) -> float:
            if Glyph.RA in sequence:
                return 10.0
            return -1.0

        initial_sequence = [Glyph.AL]
        # RA is not in the possible glyphs, so no improvement is possible
        possible_glyphs = [Glyph.UM, Glyph.IL]
        
        best_sequence, best_score = optimizer.find_optimal_sequence_greedy(
            initial_sequence=initial_sequence,
            possible_glyphs=possible_glyphs,
            objective_fn=specific_glyph_objective,
            max_iterations=5
        )

        assert best_score == -1.0
        assert best_sequence == initial_sequence # Should not have changed

    def test_respects_grammar_rules_via_objective(self):
        """
        Verify that the optimizer avoids bad sequences when the objective
        function encodes grammar rules.
        """
        initial_sequence = [Glyph.AL, Glyph.OZ] # Starts with a violation
        possible_glyphs = [Glyph.IL, Glyph.SHA, Glyph.UM]

        # The sample objective function heavily penalizes this sequence
        initial_score = optimizer.sample_objective_function(initial_sequence)
        assert initial_score == -1000.0

        best_sequence, best_score = optimizer.find_optimal_sequence_greedy(
            initial_sequence=initial_sequence,
            possible_glyphs=possible_glyphs,
            objective_fn=optimizer.sample_objective_function,
            max_iterations=5
        )

        # The best move is to add IL to fix the convergence issue.
        # The optimizer should find a sequence that is valid and has a positive score.
        assert best_score > initial_score
        assert best_score > 0  # Should be a positive score now

        # Verify the final sequence is grammatically correct
        converges, _, _ = optimizer.verify_convergence_for_sequence(best_sequence)
        is_safe, risk, _ = optimizer.verify_bifurcation_risk_for_sequence(
            best_sequence
        )
        assert converges is True
        assert is_safe is True
        assert risk <= 0.5
