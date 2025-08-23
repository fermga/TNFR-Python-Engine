# Query: 
# ContextLines: 1

"""
observers.py — Observador nodal TNFR
-------------------------------------
- El observador es también nodo: al "observar", reorganiza.
- Permite inyectar eventos externos (símbolos, sensores) en la red.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Optional

class ObservadorBase:
    """Interfaz mínima de observador."""
    def step(self, G, t: int, context: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover
        return

class ObservadorEventosSimbolicos(ObservadorBase):
    """Observador que aplica funciones/operaciones en tiempos dados.

    events: dict[int, list[Callable[[Any], None]]]
      - Cada callable recibe el grafo G y puede modificar nodos/atributos.
    """
    def __init__(self, events: Optional[Dict[int, Iterable[Callable[[Any], None]]]] = None):
        self.events: Dict[int, list[Callable[[Any], None]]] = {int(k): list(v) for k, v in (events or {}).items()}

    def step(self, G, t: int, context: Optional[Dict[str, Any]] = None) -> None:
        for fn in self.events.get(int(t), []) or []:
            try:
                fn(G)
            except Exception:
                pass
