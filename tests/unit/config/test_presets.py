from __future__ import annotations

import pytest

from tnfr.config.presets import PREFERRED_PRESET_NAMES, get_preset

from tests.legacy_tokens import LEGACY_PRESET_TOKENS


@pytest.mark.parametrize("name", PREFERRED_PRESET_NAMES)
def test_get_preset_accepts_preferred_names(name: str) -> None:
    tokens = get_preset(name)
    assert tokens, f"Preset '{name}' should not be empty"


@pytest.mark.parametrize("legacy", LEGACY_PRESET_TOKENS)
def test_removed_presets_no_longer_receive_guidance(legacy: str) -> None:
    with pytest.raises(KeyError) as excinfo:
        get_preset(legacy)

    assert excinfo.value.args == (f"Preset not found: {legacy}",)
