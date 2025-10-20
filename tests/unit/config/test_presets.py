from __future__ import annotations

from contextlib import nullcontext

import pytest

from tnfr.config.presets import (
    LEGACY_PRESET_NAMES,
    PREFERRED_PRESET_NAMES,
    PRESET_NAME_ALIASES,
    SPANISH_PRESET_ALIASES,
    get_preset,
)


@pytest.mark.parametrize("name", PREFERRED_PRESET_NAMES)
def test_get_preset_accepts_preferred_names(name: str) -> None:
    tokens = get_preset(name)
    assert tokens, f"El preset '{name}' no debería estar vacío"


@pytest.mark.parametrize("legacy", LEGACY_PRESET_NAMES)
def test_legacy_aliases_resolve_to_preferred_names(legacy: str) -> None:
    preferred = PRESET_NAME_ALIASES[legacy]
    context = (
        pytest.warns(FutureWarning, match="Spanish preset identifier")
        if legacy in SPANISH_PRESET_ALIASES
        else nullcontext()
    )
    with context:
        assert get_preset(legacy) == get_preset(preferred)


@pytest.mark.parametrize("legacy", SPANISH_PRESET_ALIASES.keys())
def test_spanish_aliases_announce_removal_timeline(legacy: str) -> None:
    preferred = SPANISH_PRESET_ALIASES[legacy]
    warning_message = (
        f"Spanish preset identifier '{legacy}' is deprecated and will be removed "
        f"in TNFR 7.0. Use '{preferred}' instead."
    )
    with pytest.warns(FutureWarning, match="will be removed in TNFR 7.0") as record:
        assert get_preset(legacy) == get_preset(preferred)

    assert any(w.message.args[0] == warning_message for w in record)
