from __future__ import annotations

"""
TNFR — Teoría de la Naturaleza Fractal Resonante
API canónica del paquete.

Principio operativo (ecuación nodal):
    ∂EPI/∂t = νf · ΔNFR(t)

Este paquete expone utilidades para preparar una red (preparar_red),
ejecutar la dinámica (step, run) y observar coherencia (coherencia_global,
orden_kuramoto), alineado con la TNFR.
"""

__version__ = "3.0.0"

# Re-exports de la API pública
from .ontosim import preparar_red
from .dynamics import step, run, set_delta_nfr_hook
from .observers import attach_standard_observer, coherencia_global, orden_kuramoto

__all__ = [
    "preparar_red",
    "step", "run", "set_delta_nfr_hook",
    "attach_standard_observer", "coherencia_global", "orden_kuramoto",
    "__version__",
]

# --- Adaptador de imports internos (compatibilidad sin tocar tus módulos) ---
# Varios archivos del paquete usan imports absolutos como:
#     from constants import DEFAULTS
# en lugar de imports relativos:
#     from .constants import DEFAULTS
# Para no reescribirlos, registramos alias en sys.modules.
import sys as _sys
from . import constants as _constants
from . import helpers as _helpers
from . import observers as _observers
from . import dynamics as _dynamics
from . import operators as _operators
from . import ontosim as _ontosim

_sys.modules.setdefault("constants", _constants)
_sys.modules.setdefault("helpers", _helpers)
_sys.modules.setdefault("observers", _observers)
_sys.modules.setdefault("dynamics", _dynamics)
_sys.modules.setdefault("operators", _operators)
_sys.modules.setdefault("ontosim", _ontosim)
