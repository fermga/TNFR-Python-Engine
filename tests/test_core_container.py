"""Service replacement must take effect after a dependency has been resolved."""

from __future__ import annotations

import pytest

from tnfr.core import TNFRContainer
from tnfr.core.interfaces import TelemetryCollector, ValidationService


@pytest.mark.parametrize("original_kind", ["singleton", "factory"])
@pytest.mark.parametrize("replacement_kind", ["singleton", "factory"])
def test_replacing_a_resolved_service_preserves_the_new_lifetime(
    original_kind, replacement_kind
):
    container = TNFRContainer()
    original = object()
    if original_kind == "singleton":
        container.register_singleton(ValidationService, original)
    else:
        container.register_factory(ValidationService, lambda: original)
    assert container.get(ValidationService) is original

    independent = object()
    container.register_singleton(TelemetryCollector, independent)
    assert container.get(TelemetryCollector) is independent

    if replacement_kind == "singleton":
        replacement = object()
        container.register_singleton(ValidationService, replacement)
        first = container.get(ValidationService)
        second = container.get(ValidationService)
        assert first is second is replacement
    else:
        created = []

        def create_service():
            instance = object()
            created.append(instance)
            return instance

        container.register_factory(ValidationService, create_service)
        assert not created  # Registration itself must not evaluate a factory.
        first = container.get(ValidationService)
        second = container.get(ValidationService)
        assert first is not second
        assert created == [first, second]
    assert first is not original
    assert container.get(TelemetryCollector) is independent


def test_default_container_can_override_an_already_used_service():
    container = TNFRContainer.create_default()
    original = container.get(ValidationService)
    telemetry = container.get(TelemetryCollector)
    replacement = object()
    container.register_singleton(ValidationService, replacement)
    assert container.get(ValidationService) is replacement
    assert container.get(ValidationService) is not original
    assert container.get(TelemetryCollector) is telemetry
