"""Input-report caches preserve type, live policy and detached evidence."""

from fractions import Fraction

import numpy as np
import pytest

from tnfr.validation.unified_validation_system import (
    TNFRUnifiedValidationSystem,
    ValidationConfig,
)


@pytest.mark.parametrize(
    "method",
    ["validate_coherence", "validate_structural_frequency", "validate_phase_value"],
)
def test_numeric_cache_entry_cannot_admit_text(method):
    validator = TNFRUnifiedValidationSystem()
    validate = getattr(validator, method)
    assert validate(0.5).is_valid
    assert not validate("0.5").is_valid
    assert validate(0.5).is_valid
    assert validator.get_cache_statistics()["hits"] == 1


def test_cached_evidence_is_detached_on_store_and_read():
    validator = TNFRUnifiedValidationSystem()
    first = validator.validate_coherence(0.5)
    first.is_valid = False
    first.error_messages.append("caller mutation")
    first.validation_metadata["bounds"][0] = -1.0
    second = validator.validate_coherence(0.5)
    assert second.is_valid
    assert second.error_messages == []
    assert second.validation_metadata["bounds"] == [0.0, 1.0]
    second.validation_metadata["bounds"][1] = 7.0
    assert validator.validate_coherence(0.5).validation_metadata["bounds"] == [0.0, 1.0]


def test_live_numeric_policy_cannot_reuse_old_admission():
    config = ValidationConfig()
    validator = TNFRUnifiedValidationSystem(config)
    assert validator.validate_coherence(0.5).is_valid
    config.max_coherence = 0.4
    assert not validator.validate_coherence(0.5).is_valid
    config.max_coherence = float("nan")
    assert not validator.validate_coherence(0.5).is_valid
    config.max_coherence = 1.0
    assert validator.validate_coherence(0.5).is_valid
    assert validator.get_cache_statistics()["hits"] == 1


@pytest.mark.parametrize("flag", ["enable_caching", "cache_validation_results"])
def test_either_cache_switch_disables_both_reads_and_writes(flag):
    config = ValidationConfig()
    validator = TNFRUnifiedValidationSystem(config)
    assert validator.validate_coherence(0.5).is_valid
    setattr(config, flag, False)
    assert validator.validate_coherence(0.5).is_valid
    assert validator.validate_coherence(0.6).is_valid
    stats = validator.get_cache_statistics()
    assert stats["hits"] == 0
    assert stats["cache_size"] == 1
    assert not stats["cache_enabled"]


def test_string_policy_uses_exact_input_zero_limit_and_current_patterns():
    config = ValidationConfig(forbidden_patterns=[])
    validator = TNFRUnifiedValidationSystem(config)
    assert not validator.validate_string_input([]).is_valid
    assert validator.validate_string_input("", max_length=0).is_valid
    assert not validator.validate_string_input("x", max_length=0).is_valid
    assert not validator.validate_string_input("x", max_length=True).is_valid
    assert validator.validate_string_input("reserved").is_valid
    config.forbidden_patterns.append("reserved")
    assert not validator.validate_string_input("reserved").is_valid


@pytest.mark.parametrize(
    "method", ["validate_structural_frequency", "validate_phase_value"]
)
@pytest.mark.parametrize(
    "value", [True, np.bool_(False), np.array(0.5), 10**400, Fraction(1, 2**1075)]
)
def test_scalar_reports_reject_invalid_representation_without_coercion(method, value):
    result = getattr(TNFRUnifiedValidationSystem(), method)(value)
    assert not result.is_valid
    assert result.error_messages


def test_raw_configured_bound_cannot_be_rounded_into_acceptance():
    validator = TNFRUnifiedValidationSystem(
        ValidationConfig(max_structural_frequency=1.0)
    )
    assert not validator.validate_structural_frequency(
        Fraction(2**55 + 1, 2**55)
    ).is_valid
    validator.config.strict_mode = False
    result = validator.validate_structural_frequency(2.0)
    assert result.is_valid and result.warnings
    assert not validator.validate_structural_frequency(-1.0).is_valid


def test_phase_report_retains_normalization_and_configured_warning_scope():
    validator = TNFRUnifiedValidationSystem()
    assert validator.validate_phase_value(-0.5).is_valid
    result = validator.validate_phase_value(-0.5, normalize=False)
    assert result.validated_value == -0.5 and result.warnings
    validator.config.min_phase_value = -1.0
    assert not validator.validate_phase_value(-0.5, normalize=False).warnings
    assert not validator.validate_phase_value(0.5, normalize="yes").is_valid


def test_unbounded_policy_does_not_admit_infinite_samples():
    validator = TNFRUnifiedValidationSystem(
        ValidationConfig(
            max_structural_frequency=float("inf"),
            min_phase_value=-float("inf"),
            max_phase_value=float("inf"),
        )
    )
    assert validator.validate_structural_frequency(1e100).is_valid
    assert not validator.validate_structural_frequency(float("inf")).is_valid
    phase = validator.validate_phase_value(-1e100, normalize=False)
    assert phase.is_valid and not phase.warnings
    assert not validator.validate_phase_value(float("inf"), normalize=False).is_valid


def test_normalized_phase_does_not_consume_unused_warning_bounds():
    validator = TNFRUnifiedValidationSystem()
    expected = validator.validate_phase_value(-0.5).validated_value
    validator.config.min_phase_value = "unused"
    validator.config.max_phase_value = float("nan")
    result = validator.validate_phase_value(-0.5)
    assert result.is_valid and result.validated_value == expected
    assert validator.get_cache_statistics()["hits"] == 1
    assert validator.validate_phase_value(-1.0).is_valid
    assert not validator.validate_phase_value(-0.5, normalize=False).is_valid
