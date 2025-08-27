from __future__ import annotations
"""
TNFR — Teoría de la Naturaleza Fractal Resonante
API pública del paquete.

Ecuación nodal:
    ∂EPI/∂t = νf · ΔNFR(t)
"""

__version__ = "3.0.3"

import sys as _sys

# ------------------------------------------------------------
# 1) Crear alias para módulos con imports absolutos entre sí
#    (los submódulos usan cosas tipo `from constants import ...`)
#    Por eso registramos primero: constants → helpers → operators → observers
# ------------------------------------------------------------
from . import constants as _constants
_sys.modules.setdefault("constants", _constants)

from . import helpers as _helpers
_sys.modules.setdefault("helpers", _helpers)

from . import operators as _operators
_sys.modules.setdefault("operators", _operators)

from . import observers as _observers
_sys.modules.setdefault("observers", _observers)

# ------------------------------------------------------------
# 2) IMPORTAR dynamics y ALIAS antes de ontosim
#    (porque ontosim hace `from dynamics import ...`)
# ------------------------------------------------------------
from . import dynamics as _dynamics
_sys.modules.setdefault("dynamics", _dynamics)

# ------------------------------------------------------------
# 3) Ahora sí, importar ontosim y alias
# ------------------------------------------------------------
from . import ontosim as _ontosim
_sys.modules.setdefault("ontosim", _ontosim)

# ------------------------------------------------------------
# 4) Re-exports de la API pública
# ------------------------------------------------------------
from .dynamics import step, run, set_delta_nfr_hook
from .ontosim import preparar_red
from .observers import attach_standard_observer, coherencia_global, orden_kuramoto

__all__ = [
    "preparar_red",
    "step", "run", "set_delta_nfr_hook",
    "attach_standard_observer", "coherencia_global", "orden_kuramoto",
    "__version__",
]
