"""All public C(t) validators share the canonical finite unit interval."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.metrics.common import validate_structural_coherence
from tnfr.security.validation import validate_coherence_value
from tnfr.validation.unified_validation_system import TNFRUnifiedValidationSystem


@pytest.mark.parametrize(
    "value", [0.0, 0.25, 1.0, Fraction(1, 4), np.float64(0.5), Fraction(1, 2**1074)]
)
def test_coherence_validators_accept_the_closed_unit_interval(value: float) -> None:
    assert validate_structural_coherence(value) == value
    assert validate_coherence_value(value) == value
    result = TNFRUnifiedValidationSystem().validate_coherence(value)
    assert result.is_valid is True
    assert result.validated_value == value


@pytest.mark.parametrize("value", [True, False, -0.1, 1.1, float("nan"), float("inf")])
def test_coherence_validators_reject_the_same_invalid_domain(value: object) -> None:
    expected = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(expected):
        validate_structural_coherence(value)
    with pytest.raises(TNFRValueError):
        validate_coherence_value(value)

    result = TNFRUnifiedValidationSystem().validate_coherence(value)
    assert result.is_valid is False
    assert result.error_messages


@pytest.mark.parametrize(
    "value",
    [Fraction(1, 2**1075), Fraction(-1, 2**1075), Fraction(2**55 + 1, 2**55)],
)
def test_coherence_values_cannot_round_into_an_admitted_boundary(value):
    for validate in (validate_structural_coherence, validate_coherence_value):
        with pytest.raises((ValueError, TNFRValueError)):
            validate(value)
    assert not TNFRUnifiedValidationSystem().validate_coherence(value).is_valid


@pytest.mark.parametrize("value", [np.bool_(False), "0.5", np.array(0.5), 0.5 + 0j])
def test_coherence_admission_uses_the_shared_raw_real_domain(value):
    with pytest.raises(TypeError):
        validate_structural_coherence(value)
    with pytest.raises(TNFRValueError):
        validate_coherence_value(value)
