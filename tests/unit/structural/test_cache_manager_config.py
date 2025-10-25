"""Configuration tests for :class:`tnfr.cache.CacheManager`."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from tnfr.cache import CacheManager


@pytest.fixture()
def capacity_payload() -> dict[str, object]:
    manager = CacheManager(default_capacity=32)
    manager.configure(
        default_capacity=128,
        overrides={"alpha": 16, "beta": None, "gamma": 0},
    )
    exported = manager.export_config()
    return asdict(exported)


def test_export_config_returns_expected_mapping(capacity_payload: dict[str, object]) -> None:
    assert capacity_payload == {
        "default_capacity": 128,
        "overrides": {"alpha": 16, "beta": None, "gamma": 0},
    }


def test_configure_from_mapping_applies_overrides(capacity_payload: dict[str, object]) -> None:
    manager = CacheManager()
    manager.configure_from_mapping(capacity_payload)

    assert manager.get_capacity("alpha", requested=4) == 16
    assert manager.get_capacity("beta", requested=4) is None
    assert manager.get_capacity("delta", requested=None, fallback=7) == 128


def test_configure_from_mapping_ignores_non_mapping_overrides(
    capacity_payload: dict[str, object]
) -> None:
    manager = CacheManager()
    manager.configure_from_mapping(capacity_payload)

    previous = manager.export_config()

    invalid_payload = {"overrides": ["alpha", 32]}
    manager.configure_from_mapping(invalid_payload)

    assert asdict(manager.export_config()) == asdict(previous)
    assert manager.get_capacity("alpha", requested=1) == 16
    assert manager.get_capacity("beta", requested=1) is None
