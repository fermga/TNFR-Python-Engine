"""Tests for validation configuration."""

import math

import pytest

from tnfr.validation import (
    InvariantSeverity,
    ValidationConfig,
    configure_validation,
    validation_config,
)


class TestValidationConfig:
    """Test validation configuration."""

    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config = ValidationConfig()

        assert config.validate_invariants is True
        assert config.validate_each_step is False
        assert config.min_severity == InvariantSeverity.ERROR
        assert config.epi_range == (0.0, 1.0)
        assert config.vf_range == (0.001, 1000.0)
        assert config.phase_coupling_threshold == math.pi / 2
        assert config.enable_semantic_validation is True
        assert config.allow_semantic_warnings is True

    def test_configure_validation_updates_global(self):
        """Test that configure_validation updates global config."""
        original_value = validation_config.validate_each_step

        try:
            configure_validation(validate_each_step=not original_value)
            assert validation_config.validate_each_step == (not original_value)
        finally:
            # Restore original value
            configure_validation(validate_each_step=original_value)

    def test_configure_validation_multiple_params(self):
        """Test updating multiple configuration parameters."""
        original_invariants = validation_config.validate_invariants
        original_semantic = validation_config.enable_semantic_validation

        try:
            configure_validation(
                validate_invariants=not original_invariants,
                enable_semantic_validation=not original_semantic,
            )

            assert validation_config.validate_invariants == (not original_invariants)
            assert validation_config.enable_semantic_validation == (
                not original_semantic
            )
        finally:
            # Restore original values
            configure_validation(
                validate_invariants=original_invariants,
                enable_semantic_validation=original_semantic,
            )

    def test_configure_validation_numeric_params(self):
        """Test updating numeric configuration parameters."""
        original_threshold = validation_config.phase_coupling_threshold

        try:
            new_threshold = math.pi / 3
            configure_validation(phase_coupling_threshold=new_threshold)

            assert validation_config.phase_coupling_threshold == new_threshold
        finally:
            # Restore original value
            configure_validation(phase_coupling_threshold=original_threshold)

    def test_configure_validation_invalid_key_raises(self):
        """Test that invalid configuration keys raise ValueError."""
        with pytest.raises(ValueError, match="Unknown validation config key"):
            configure_validation(invalid_key=True)

    def test_configure_validation_min_severity(self):
        """Test configuring minimum severity level."""
        original_severity = validation_config.min_severity

        try:
            configure_validation(min_severity=InvariantSeverity.WARNING)
            assert validation_config.min_severity == InvariantSeverity.WARNING

            configure_validation(min_severity=InvariantSeverity.CRITICAL)
            assert validation_config.min_severity == InvariantSeverity.CRITICAL
        finally:
            # Restore original value
            configure_validation(min_severity=original_severity)

    def test_validation_config_dataclass(self):
        """Test that ValidationConfig is a proper dataclass."""
        config1 = ValidationConfig()
        config2 = ValidationConfig()

        # Should be equal with same values
        assert config1.validate_invariants == config2.validate_invariants
        assert config1.epi_range == config2.epi_range

        # Should be able to create with custom values
        custom_config = ValidationConfig(
            validate_invariants=False, phase_coupling_threshold=math.pi
        )

        assert custom_config.validate_invariants is False
        assert custom_config.phase_coupling_threshold == math.pi

    def test_epi_range_tuple(self):
        """Test EPI range is a tuple."""
        config = ValidationConfig()

        assert isinstance(config.epi_range, tuple)
        assert len(config.epi_range) == 2
        assert config.epi_range[0] < config.epi_range[1]

    def test_vf_range_tuple(self):
        """Test vf range is a tuple."""
        config = ValidationConfig()

        assert isinstance(config.vf_range, tuple)
        assert len(config.vf_range) == 2
        assert config.vf_range[0] < config.vf_range[1]
        assert config.vf_range[0] > 0  # Hz_str must be positive
