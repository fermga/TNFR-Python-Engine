"""Carga e inyección de configuraciones externas.

Permite definir parámetros en JSON o YAML y aplicarlos sobre ``G.graph``
reutilizando :func:`tnfr.constants.inject_defaults`.
"""

from __future__ import annotations
from typing import Any, Dict
import json

try:  # pragma: no cover - dependencia opcional
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from .constants import inject_defaults


def load_config(path: str) -> Dict[str, Any]:
    """Lee un archivo JSON/YAML y devuelve un ``dict`` con los parámetros."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        if not yaml:  # pragma: no cover - fallo en entorno sin pyyaml
            raise RuntimeError("pyyaml no está instalado")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("El archivo de configuración debe contener un objeto")
    return data


def apply_config(G, path: str) -> None:
    """Inyecta parámetros desde ``path`` sobre ``G.graph``.

    Se reutiliza :func:`inject_defaults` para mantener la semántica de los
    *defaults* canónicos.
    """
    cfg = load_config(path)
    inject_defaults(G, cfg, override=True)
