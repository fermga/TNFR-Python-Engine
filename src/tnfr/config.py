"""Carga e inyección de configuraciones externas.

Permite definir parámetros en JSON o YAML y aplicarlos sobre ``G.graph``
reutilizando :func:`tnfr.constants.inject_defaults`.
"""

from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from .helpers import read_structured_file

from .constants import inject_defaults


def load_config(path: Path) -> Dict[str, Any]:
    """Lee un archivo JSON/YAML y devuelve un ``dict`` con los parámetros."""
    data = read_structured_file(path)
    if not isinstance(data, dict):
        raise ValueError("El archivo de configuración debe contener un objeto")
    return data


def apply_config(G, path: Path) -> None:
    """Inyecta parámetros desde ``path`` sobre ``G.graph``.

    Se reutiliza :func:`inject_defaults` para mantener la semántica de los
    *defaults* canónicos.
    """
    cfg = load_config(path)
    inject_defaults(G, cfg, override=True)
