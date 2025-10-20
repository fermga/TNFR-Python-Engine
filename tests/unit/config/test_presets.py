from __future__ import annotations

import pytest

from tnfr.execution import CANONICAL_PRESET_NAME
from tnfr.config.presets import (
    PREFERRED_PRESET_NAMES,
    get_preset,
)


@pytest.mark.parametrize("name", PREFERRED_PRESET_NAMES)
def test_get_preset_accepts_preferred_names(name: str) -> None:
    tokens = get_preset(name)
    assert tokens, f"El preset '{name}' no debería estar vacío"


@pytest.mark.parametrize(
    ("legacy", "preferred", "version"),
    (
        ("arranque_resonante", "resonant_bootstrap", "TNFR 7.0"),
        ("mutacion_contenida", "contained_mutation", "TNFR 7.0"),
        ("exploracion_acople", "coupling_exploration", "TNFR 7.0"),
        ("ejemplo_canonico", CANONICAL_PRESET_NAME, "TNFR 9.0"),
    ),
)
def test_removed_presets_raise_with_guidance(
    legacy: str, preferred: str, version: str
) -> None:
    with pytest.raises(KeyError) as excinfo:
        get_preset(legacy)

    message = excinfo.value.args[0]
    assert message == (
        f"Legacy preset identifier '{legacy}' was removed in {version}. "
        f"Use '{preferred}' instead."
    )
