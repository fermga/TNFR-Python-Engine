"""Compatibility exports for canonical initialization defaults.

Values and dataclass definitions are owned by ``tnfr.config.defaults_init``.
The legacy public helper imports remain available for import compatibility.
"""

from __future__ import annotations

from ..config.defaults_init import INIT_DEFAULTS as INIT_DEFAULTS
from ..config.defaults_init import InitDefaults as InitDefaults
from ..config.defaults_init import asdict as asdict
from ..config.defaults_init import dataclass as dataclass
from ..config.defaults_init import math as math

# Preserve the complete historical wildcard surface, including helper imports.
__all__ = [
    "annotations",
    "INIT_DEFAULTS",
    "InitDefaults",
    "asdict",
    "dataclass",
    "math",
]
