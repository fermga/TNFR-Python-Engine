from __future__ import annotations

import pytest

from tnfr.config.presets import (
    LEGACY_PRESET_NAMES,
    PREFERRED_PRESET_NAMES,
    PRESET_NAME_ALIASES,
    get_preset,
)


@pytest.mark.parametrize("name", PREFERRED_PRESET_NAMES)
def test_get_preset_accepts_preferred_names(name: str) -> None:
    tokens = get_preset(name)
    assert tokens, f"El preset '{name}' no debería estar vacío"


@pytest.mark.parametrize("legacy", LEGACY_PRESET_NAMES)
def test_legacy_aliases_resolve_to_preferred_names(legacy: str) -> None:
    preferred = PRESET_NAME_ALIASES[legacy]
    assert get_preset(legacy) == get_preset(preferred)
