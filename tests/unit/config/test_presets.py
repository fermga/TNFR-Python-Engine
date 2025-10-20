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


@pytest.mark.parametrize(
    ("legacy", "preferred"),
    (
        ("arranque_resonante", "resonant_bootstrap"),
        ("mutacion_contenida", "contained_mutation"),
        ("exploracion_acople", "coupling_exploration"),
    ),
)
def test_spanish_aliases_are_removed(legacy: str, preferred: str) -> None:
    with pytest.raises(KeyError) as excinfo:
        get_preset(legacy)

    message = excinfo.value.args[0]
    assert message == (
        f"Spanish preset identifier '{legacy}' was removed in TNFR 7.0. "
        f"Use '{preferred}' instead."
    )
