"""All public C(t) validators share the canonical finite unit interval."""

from __future__ import annotations

import pytest

from tnfr.errors import TNFRValueError
from tnfr.metrics.common import validate_structural_coherence
from tnfr.security.validation import validate_coherence_value
from tnfr.validation.unified_validation_system import TNFRUnifiedValidationSystem


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
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