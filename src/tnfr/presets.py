from __future__ import annotations
from .program import seq, block, wait


_PRESETS = {
    "arranque_resonante": seq("A’L", "E’N", "I’L", "R’A", "VA’L", "U’M", wait(3), "SH’A"),
    "mutacion_contenida": seq("A’L", "E’N", block("O’Z", "Z’HIR", "I’L", repeat=2), "R’A", "SH’A"),
    "exploracion_acople": seq(
        "A’L",
        "E’N",
        "I’L",
        "VA’L",
        "U’M",
        block("O’Z", "NA’V", "I’L", repeat=1),
        "R’A",
        "SH’A",
    ),
}


def get_preset(name: str):
    if name not in _PRESETS:
        raise KeyError(f"Preset no encontrado: {name}")
    return _PRESETS[name]
