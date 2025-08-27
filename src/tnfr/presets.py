from __future__ import annotations
from .program import seq, block, wait, ejemplo_canonico_basico


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
    "ejemplo_canonico": ejemplo_canonico_basico(),
    # Topologías fractales: expansión/contracción modular
    "fractal_expand": seq(block("T’HOL", "VA’L", "U’M", repeat=2, close="NU’L"), "R’A"),
    "fractal_contract": seq(block("T’HOL", "NU’L", "U’M", repeat=2, close="SH’A"), "R’A"),
}


def get_preset(name: str):
    if name not in _PRESETS:
        raise KeyError(f"Preset no encontrado: {name}")
    return _PRESETS[name]
