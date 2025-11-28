"""Tests for NAV (Transition) comprehensive precondition validation.

This module validates the enhanced implementation of TNFR.pdf §2.3.11 precondition
requirements for the Transition (NAV) operator:

1. νf > minimum (structural frequency for transition capacity)
2. |ΔNFR| < maximum (controlled reorganization pressure)
3. Regime validation (warns on deep latency transitions)
4. Sequence compatibility (optional strict checking)

These tests ensure NAV operates only under appropriate structural conditions,
preventing unstable regime transitions and providing actionable feedback for
problematic states.
"""

import warnings

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import Coherence, Emission, Silence, Transition
from tnfr.operators.preconditions import OperatorPreconditionError, validate_transition
from tnfr.structural import create_nfr, run_sequence


class TestTransitionBasicPreconditions:
    """Test basic NAV precondition validation (νf minimum)."""

    def test_validate_transition_success_normal_state(self):
        """Validation passes for normal active node."""
        G, node = create_nfr("active", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Should not raise
        validate_transition(G, node)

    def test_validate_transition_success_minimum_vf(self):
        """Validation passes at minimum νf threshold."""
        G, node = create_nfr("min_vf", epi=0.4, vf=0.01)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.2)

        # Should pass (0.01 >= 0.01 default minimum)
        validate_transition(G, node)

    def test_validate_transition_fails_zero_vf(self):
        """Validation fails when νf = 0."""
        G, node = create_nfr("zero_vf", epi=0.5, vf=0.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "Transition" in error_msg
        assert "Structural frequency too low" in error_msg
        assert "νf=0.000" in error_msg

    def test_validate_transition_fails_low_vf(self):
        """Validation fails when νf < minimum."""
        G, node = create_nfr("low_vf", epi=0.5, vf=0.005)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "νf=0.005" in error_msg
        assert "< 0.010" in error_msg

    def test_validate_transition_custom_min_vf(self):
        """Custom NAV_MIN_VF threshold should be respected."""
        G, node = create_nfr("custom", epi=0.5, vf=0.05)
        G.graph["NAV_MIN_VF"] = 0.1  # Higher threshold
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "νf=0.050" in error_msg
        assert "< 0.100" in error_msg


class TestTransitionDNFRValidation:
    """Test ΔNFR bounds validation for NAV."""

    def test_validate_transition_success_controlled_dnfr(self):
        """Validation passes with controlled ΔNFR."""
        G, node = create_nfr("controlled", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.5)

        # Should pass (0.5 < 1.0 default maximum)
        validate_transition(G, node)

    def test_validate_transition_success_zero_dnfr(self):
        """Validation passes with ΔNFR = 0."""
        G, node = create_nfr("zero_dnfr", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.0)

        # Should pass (|0| < 1.0)
        validate_transition(G, node)

    def test_validate_transition_success_negative_dnfr(self):
        """Validation passes with negative ΔNFR within bounds."""
        G, node = create_nfr("negative", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, -0.8)

        # Should pass (|-0.8| = 0.8 < 1.0)
        validate_transition(G, node)

    def test_validate_transition_fails_high_positive_dnfr(self):
        """Validation fails when positive ΔNFR exceeds maximum."""
        G, node = create_nfr("high_dnfr", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 1.5)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "ΔNFR too high for stable transition" in error_msg
        assert "|ΔNFR|=1.500" in error_msg
        assert "> 1.0" in error_msg
        assert "IL (Coherence)" in error_msg

    def test_validate_transition_fails_high_negative_dnfr(self):
        """Validation fails when negative ΔNFR magnitude exceeds maximum."""
        G, node = create_nfr("high_neg", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, -1.5)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "|ΔNFR|=1.500" in error_msg
        assert "> 1.0" in error_msg

    def test_validate_transition_fails_at_dnfr_boundary(self):
        """Validation fails when |ΔNFR| exactly exceeds maximum."""
        G, node = create_nfr("boundary", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 1.001)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "ΔNFR too high" in error_msg

    def test_validate_transition_custom_max_dnfr(self):
        """Custom NAV_MAX_DNFR threshold should be respected."""
        G, node = create_nfr("custom", epi=0.5, vf=0.6)
        G.graph["NAV_MAX_DNFR"] = 0.5  # Lower threshold
        set_attr(G.nodes[node], ALIAS_DNFR, 0.8)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "|ΔNFR|=0.800" in error_msg
        assert "> 0.5" in error_msg


class TestTransitionRegimeValidation:
    """Test regime origin validation and warnings."""

    def test_validate_transition_warns_deep_latency(self):
        """Warns when transitioning from deep latency (low EPI)."""
        G, node = create_nfr("latent", epi=0.03, vf=0.6)
        G.nodes[node]["latent"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with pytest.warns(UserWarning) as warnings_list:
            validate_transition(G, node)

        assert len(warnings_list) == 1
        warning_msg = str(warnings_list[0].message)
        assert "deep latency" in warning_msg
        assert "EPI=0.030" in warning_msg
        assert "< 0.050" in warning_msg
        assert "AL (Emission)" in warning_msg

    def test_validate_transition_no_warning_adequate_epi(self):
        """No warning when transitioning from latency with adequate EPI."""
        G, node = create_nfr("latent_ok", epi=0.15, vf=0.6)
        G.nodes[node]["latent"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors
            # Should not warn (0.15 >= 0.05 default minimum)
            validate_transition(G, node)

    def test_validate_transition_no_warning_non_latent(self):
        """No warning for non-latent nodes even with low EPI."""
        G, node = create_nfr("active_low", epi=0.03, vf=0.6)
        # latent flag not set
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not warn (only warns if latent=True)
            validate_transition(G, node)

    def test_validate_transition_custom_min_epi_latency(self):
        """Custom NAV_MIN_EPI_FROM_LATENCY threshold should be respected."""
        G, node = create_nfr("custom", epi=0.08, vf=0.6)
        G.graph["NAV_MIN_EPI_FROM_LATENCY"] = 0.1  # Higher threshold
        G.nodes[node]["latent"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with pytest.warns(UserWarning) as warnings_list:
            validate_transition(G, node)

        warning_msg = str(warnings_list[0].message)
        assert "EPI=0.080" in warning_msg
        assert "< 0.100" in warning_msg


class TestTransitionSequenceValidation:
    """Test optional sequence compatibility checking."""

    def test_validate_transition_no_warning_when_check_disabled(self):
        """No sequence warning when NAV_STRICT_SEQUENCE_CHECK=False."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply incompatible operator before NAV
        from tnfr.operators.definitions import Dissonance

        dissonance = Dissonance()
        dissonance(G, node)

        # Should not warn (strict checking disabled by default)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_transition(G, node)

    def test_validate_transition_warns_after_dissonance_when_strict(self):
        """Warns when NAV applied after dissonance with strict checking."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply dissonance (not a valid predecessor)
        from tnfr.operators.definitions import Dissonance

        dissonance = Dissonance()
        dissonance(G, node)

        with pytest.warns(UserWarning) as warnings_list:
            validate_transition(G, node)

        warning_msg = str(warnings_list[0].message)
        assert "NAV applied after dissonance" in warning_msg
        assert "emission" in warning_msg
        assert "coherence" in warning_msg
        assert "silence" in warning_msg
        assert "self_organization" in warning_msg

    def test_validate_transition_no_warning_after_emission(self):
        """No warning when NAV applied after emission (valid predecessor)."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply emission (valid predecessor)
        emission = Emission()
        emission(G, node)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_transition(G, node)

    def test_validate_transition_no_warning_after_coherence(self):
        """No warning when NAV applied after coherence (valid predecessor)."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply coherence (valid predecessor)
        coherence = Coherence()
        coherence(G, node)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_transition(G, node)

    def test_validate_transition_no_warning_after_silence(self):
        """No warning when NAV applied after silence (valid predecessor)."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply silence (valid predecessor)
        silence = Silence()
        silence(G, node)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_transition(G, node)

    def test_validate_transition_no_warning_empty_history(self):
        """No warning when history is empty (first operator)."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # No history - NAV as first operator (valid as generator)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_transition(G, node)


class TestTransitionIntegration:
    """Test NAV validation in realistic operator sequences."""

    def test_transition_after_coherence_reduces_dnfr(self):
        """IL → NAV sequence should pass validation with reduced ΔNFR."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 1.2)  # Initially too high

        # Apply coherence to reduce ΔNFR
        coherence = Coherence()
        coherence(G, node)

        # Now ΔNFR should be reduced, allowing NAV
        validate_transition(G, node)

    def test_transition_sha_nav_sequence_canonical(self):
        """NAV canonical sequence (SHA cannot start sequences per U1a)."""
        G, node = create_nfr("test", epi=0.3, vf=0.8)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.4)

        # Apply NAV + stabilizer + closure sequence (per U1a, U2, U1b)
        run_sequence(G, node, [Transition(), Coherence(), Silence()])

        # Should complete without errors
        assert True

    def test_transition_al_nav_il_sequence_canonical(self):
        """AL → NAV → IL canonical sequence should pass all validations."""
        G, node = create_nfr("test", epi=0.2, vf=0.8)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply AL → NAV → IL → closure sequence (per U1b)
        run_sequence(
            G, node, [Emission(), Transition(), Coherence(), Silence()]
        )

        # Should complete without errors
        assert True

    def test_transition_multiple_warnings_combined(self):
        """Multiple warnings should be issued when multiple issues present."""
        G, node = create_nfr("test", epi=0.03, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        G.nodes[node]["latent"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        # Apply incompatible operator
        from tnfr.operators.definitions import Dissonance

        dissonance = Dissonance()
        dissonance(G, node)

        with pytest.warns(UserWarning) as warnings_list:
            validate_transition(G, node)

        # Should have warnings for both deep latency and sequence
        assert len(warnings_list) == 2
        warning_messages = [str(w.message) for w in warnings_list]

        # Check for deep latency warning
        assert any("deep latency" in msg for msg in warning_messages)

        # Check for sequence warning
        assert any("NAV applied after" in msg for msg in warning_messages)


class TestTransitionErrorMessages:
    """Test that error messages provide actionable guidance."""

    def test_error_message_suggests_il_for_high_dnfr(self):
        """High ΔNFR error should suggest IL (Coherence)."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        set_attr(G.nodes[node], ALIAS_DNFR, 1.5)

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_transition(G, node)

        error_msg = str(exc_info.value)
        assert "Apply IL (Coherence) first" in error_msg
        assert "reduce reorganization pressure" in error_msg

    def test_warning_suggests_al_for_deep_latency(self):
        """Deep latency warning should suggest AL (Emission)."""
        G, node = create_nfr("test", epi=0.03, vf=0.6)
        G.nodes[node]["latent"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        with pytest.warns(UserWarning) as warnings_list:
            validate_transition(G, node)

        warning_msg = str(warnings_list[0].message)
        assert "Consider AL (Emission)" in warning_msg
        assert "smoother activation" in warning_msg

    def test_warning_lists_valid_predecessors(self):
        """Sequence warning should list all valid predecessors."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        G.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
        set_attr(G.nodes[node], ALIAS_DNFR, 0.3)

        from tnfr.operators.definitions import Dissonance

        dissonance = Dissonance()
        dissonance(G, node)

        with pytest.warns(UserWarning) as warnings_list:
            validate_transition(G, node)

        warning_msg = str(warnings_list[0].message)
        # Should list all valid predecessors
        assert "emission" in warning_msg
        assert "coherence" in warning_msg
        assert "silence" in warning_msg
        assert "self_organization" in warning_msg
