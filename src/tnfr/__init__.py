
from __future__ import annotations
"""
TNFR — Teoría de la Naturaleza Fractal Resonante
API pública del paquete.

Ecuación nodal:
    ∂EPI/∂t = νf · ΔNFR(t)
"""

__version__ = "3.0.3"

# Re-exports de la API pública
from .dynamics import step, run, set_delta_nfr_hook
from .ontosim import preparar_red
from .observers import attach_standard_observer, coherencia_global, orden_kuramoto

__all__ = [
    "preparar_red",
    "step", "run", "set_delta_nfr_hook",
    "attach_standard_observer", "coherencia_global", "orden_kuramoto",
    "__version__",
]
