"""Tests for physics-based operator derivation from TNFR principles."""

import pytest

from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
    VALID_START_OPERATORS,
    VALID_END_OPERATORS,
    validate_physics_derivation,
)
from tnfr.config.physics_derivation import (
    can_generate_epi_from_null,
    can_activate_latent_epi,
    can_stabilize_reorganization,
    achieves_operational_closure,
    derive_start_operators_from_physics,
    derive_end_operators_from_physics,
)


class TestEPIGenerationCapacity:
    """Test which operators can generate EPI from null state."""

    def test_emission_can_generate_epi_from_null(self):
        """EMISSION (AL) can create EPI from null state via field emission."""
        assert can_generate_epi_from_null(EMISSION)

    def test_reception_cannot_generate_epi_from_null(self):
        """RECEPTION (EN) requires external source and existing EPI."""
        assert not can_generate_epi_from_null(RECEPTION)

    def test_coherence_cannot_generate_epi_from_null(self):
        """COHERENCE (IL) stabilizes existing form, cannot create from null."""
        assert not can_generate_epi_from_null(COHERENCE)

    def test_dissonance_cannot_generate_epi_from_null(self):
        """DISSONANCE (OZ) perturbs existing structure, needs EPI > 0."""
        assert not can_generate_epi_from_null(DISSONANCE)

    def test_silence_cannot_generate_epi_from_null(self):
        """SILENCE (SHA) suspends reorganization, needs active νf."""
        assert not can_generate_epi_from_null(SILENCE)


class TestEPIActivationCapacity:
    """Test which operators can activate latent/dormant EPI."""

    def test_recursivity_can_activate_latent_epi(self):
        """RECURSIVITY (REMESH) can replicate/activate existing patterns."""
        assert can_activate_latent_epi(RECURSIVITY)

    def test_transition_can_activate_latent_epi(self):
        """TRANSITION (NAV) can activate node from another phase."""
        assert can_activate_latent_epi(TRANSITION)

    def test_emission_not_activator(self):
        """EMISSION is a generator, not an activator of latent structure."""
        # Note: emission creates from null, doesn't activate latent
        assert not can_activate_latent_epi(EMISSION)

    def test_reception_not_activator(self):
        """RECEPTION requires external input, not latent activation."""
        assert not can_activate_latent_epi(RECEPTION)


class TestStabilizationCapacity:
    """Test which operators can stabilize reorganization (∂EPI/∂t → 0)."""

    def test_silence_stabilizes_reorganization(self):
        """SILENCE (SHA) forces ∂EPI/∂t → 0 via νf → 0."""
        assert can_stabilize_reorganization(SILENCE)

    def test_coherence_does_not_guarantee_stabilization(self):
        """COHERENCE (IL) reduces ΔNFR but doesn't force ∂EPI/∂t = 0."""
        assert not can_stabilize_reorganization(COHERENCE)

    def test_emission_does_not_stabilize(self):
        """EMISSION (AL) generates activation, opposite of stabilization."""
        assert not can_stabilize_reorganization(EMISSION)

    def test_dissonance_does_not_stabilize(self):
        """DISSONANCE (OZ) increases ΔNFR, opposite of stabilization."""
        assert not can_stabilize_reorganization(DISSONANCE)


class TestOperationalClosure:
    """Test which operators achieve operational closure."""

    def test_transition_achieves_closure(self):
        """TRANSITION (NAV) hands off to next phase, completes cycle."""
        assert achieves_operational_closure(TRANSITION)

    def test_recursivity_achieves_closure(self):
        """RECURSIVITY (REMESH) creates fractal closure."""
        assert achieves_operational_closure(RECURSIVITY)

    def test_dissonance_questionable_closure(self):
        """DISSONANCE (OZ) included but questionable as closure operator."""
        # Backward compatibility: included but represents edge case
        assert achieves_operational_closure(DISSONANCE)

    def test_emission_no_closure(self):
        """EMISSION (AL) generates activation, not closure."""
        assert not achieves_operational_closure(EMISSION)

    def test_coherence_no_closure(self):
        """COHERENCE (IL) stabilizes but doesn't complete cycle."""
        assert not achieves_operational_closure(COHERENCE)


class TestStartOperatorsDerivation:
    """Test derivation of valid start operators from physics."""

    def test_derived_start_operators_include_emission(self):
        """Derived set must include EMISSION (EPI generator)."""
        derived = derive_start_operators_from_physics()
        assert EMISSION in derived

    def test_derived_start_operators_include_recursivity(self):
        """Derived set must include RECURSIVITY (latent activator)."""
        derived = derive_start_operators_from_physics()
        assert RECURSIVITY in derived

    def test_derived_start_operators_include_transition(self):
        """Derived set must include TRANSITION (phase activator)."""
        derived = derive_start_operators_from_physics()
        assert TRANSITION in derived

    def test_derived_start_operators_exclude_reception(self):
        """RECEPTION cannot start (needs external source + existing EPI)."""
        derived = derive_start_operators_from_physics()
        assert RECEPTION not in derived

    def test_derived_start_operators_exclude_coherence(self):
        """COHERENCE cannot start (stabilizes existing, cannot create)."""
        derived = derive_start_operators_from_physics()
        assert COHERENCE not in derived

    def test_derived_start_operators_exclude_dissonance(self):
        """DISSONANCE cannot start (perturbs existing structure)."""
        derived = derive_start_operators_from_physics()
        assert DISSONANCE not in derived

    def test_derived_start_operators_exclude_coupling(self):
        """COUPLING cannot start (requires both nodes active)."""
        derived = derive_start_operators_from_physics()
        assert COUPLING not in derived

    def test_derived_start_operators_exclude_silence(self):
        """SILENCE cannot start (suspends reorganization)."""
        derived = derive_start_operators_from_physics()
        assert SILENCE not in derived


class TestEndOperatorsDerivation:
    """Test derivation of valid end operators from physics."""

    def test_derived_end_operators_include_silence(self):
        """Derived set must include SILENCE (stabilizer)."""
        derived = derive_end_operators_from_physics()
        assert SILENCE in derived

    def test_derived_end_operators_include_transition(self):
        """Derived set must include TRANSITION (closure via hand-off)."""
        derived = derive_end_operators_from_physics()
        assert TRANSITION in derived

    def test_derived_end_operators_include_recursivity(self):
        """Derived set must include RECURSIVITY (fractal closure)."""
        derived = derive_end_operators_from_physics()
        assert RECURSIVITY in derived

    def test_derived_end_operators_include_dissonance(self):
        """DISSONANCE included for backward compatibility (questionable)."""
        derived = derive_end_operators_from_physics()
        assert DISSONANCE in derived

    def test_derived_end_operators_exclude_emission(self):
        """EMISSION cannot end (generates activation)."""
        derived = derive_end_operators_from_physics()
        assert EMISSION not in derived

    def test_derived_end_operators_exclude_reception(self):
        """RECEPTION cannot end (captures input, ongoing process)."""
        derived = derive_end_operators_from_physics()
        assert RECEPTION not in derived

    def test_derived_end_operators_exclude_coherence(self):
        """COHERENCE cannot end alone (doesn't guarantee ∂EPI/∂t = 0)."""
        derived = derive_end_operators_from_physics()
        assert COHERENCE not in derived

    def test_derived_end_operators_exclude_expansion(self):
        """EXPANSION cannot end (active growth)."""
        derived = derive_end_operators_from_physics()
        assert EXPANSION not in derived

    def test_derived_end_operators_exclude_mutation(self):
        """MUTATION cannot end (active transformation)."""
        derived = derive_end_operators_from_physics()
        assert MUTATION not in derived


class TestPhysicsValidation:
    """Test validation of operator sets against physics derivation."""

    def test_validate_physics_derivation_returns_dict(self):
        """validate_physics_derivation returns structured report."""
        result = validate_physics_derivation()
        assert isinstance(result, dict)
        assert "start_operators_valid" in result
        assert "end_operators_valid" in result

    def test_current_start_operators_match_physics(self):
        """Current VALID_START_OPERATORS should match physics derivation."""
        result = validate_physics_derivation()
        assert result[
            "start_operators_valid"
        ], f"VALID_START_OPERATORS mismatch: {result['discrepancies']}"

    def test_current_end_operators_match_physics(self):
        """Current VALID_END_OPERATORS should match physics derivation."""
        result = validate_physics_derivation()
        assert result[
            "end_operators_valid"
        ], f"VALID_END_OPERATORS mismatch: {result['discrepancies']}"

    def test_start_operators_expected_equals_derived(self):
        """Expected start operators should match derived set."""
        result = validate_physics_derivation()
        expected = result["start_operators_expected"]
        derived = derive_start_operators_from_physics()
        assert expected == derived

    def test_end_operators_expected_equals_derived(self):
        """Expected end operators should match derived set."""
        result = validate_physics_derivation()
        expected = result["end_operators_expected"]
        derived = derive_end_operators_from_physics()
        assert expected == derived


class TestOperatorSetsConsistency:
    """Test consistency between VALID_*_OPERATORS and derived sets."""

    def test_valid_start_operators_matches_physics(self):
        """VALID_START_OPERATORS should equal physics-derived set."""
        derived = derive_start_operators_from_physics()
        assert VALID_START_OPERATORS == derived, (
            f"VALID_START_OPERATORS={VALID_START_OPERATORS} " f"!= derived={derived}"
        )

    def test_valid_end_operators_matches_physics(self):
        """VALID_END_OPERATORS should equal physics-derived set."""
        derived = derive_end_operators_from_physics()
        assert VALID_END_OPERATORS == derived, (
            f"VALID_END_OPERATORS={VALID_END_OPERATORS} " f"!= derived={derived}"
        )


class TestPhysicsRationale:
    """Test that physics rationale is documented and accessible."""

    def test_can_generate_epi_has_docstring(self):
        """can_generate_epi_from_null has explanatory docstring."""
        assert can_generate_epi_from_null.__doc__ is not None
        assert "EPI" in can_generate_epi_from_null.__doc__
        assert "null" in can_generate_epi_from_null.__doc__

    def test_derive_start_operators_has_docstring(self):
        """derive_start_operators_from_physics has physics explanation."""
        assert derive_start_operators_from_physics.__doc__ is not None
        assert "TNFR" in derive_start_operators_from_physics.__doc__
        assert "physics" in derive_start_operators_from_physics.__doc__.lower()

    def test_derive_end_operators_has_docstring(self):
        """derive_end_operators_from_physics has physics explanation."""
        assert derive_end_operators_from_physics.__doc__ is not None
        assert "TNFR" in derive_end_operators_from_physics.__doc__
        assert "∂EPI/∂t" in derive_end_operators_from_physics.__doc__
