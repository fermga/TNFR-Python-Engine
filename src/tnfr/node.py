from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional, Protocol
from collections import deque

from .constants import (
    DEFAULTS,
    ALIAS_EPI,
    ALIAS_VF,
    ALIAS_THETA,
    ALIAS_SI,
    ALIAS_EPI_KIND,
    ALIAS_DNFR,
    ALIAS_D2EPI,
)
from .helpers import (
    push_glifo,
    get_attr,
    get_attr_str,
    set_attr,
    set_attr_str,
    set_vf,
    set_dnfr,
)

from .operators import aplicar_glifo_obj


def _nx_attr_property(
    aliases,
    *,
    default=0.0,
    getter=get_attr,
    setter=set_attr,
    to_python=float,
    to_storage=float,
    use_graph_setter=False,
):
    """Generate ``NodoNX`` property descriptors.

    Parameters
    ----------
    aliases:
        Alias or tuple of aliases used to access the attribute in the
        underlying ``networkx`` node.
    default:
        Value returned when the attribute is missing.
    getter, setter:
        Helper functions used to retrieve or store the value. ``setter`` can
        either accept ``(mapping, aliases, value)`` or, when
        ``use_graph_setter`` is ``True``, ``(G, n, value)``.
    to_python, to_storage:
        Conversion helpers applied when getting or setting the value,
        respectively.
    use_graph_setter:
        Whether ``setter`` expects ``(G, n, value)`` instead of
        ``(mapping, aliases, value)``.
    """

    def fget(self):
        return to_python(getter(self.G.nodes[self.n], aliases, default))

    def fset(self, value):
        value = to_storage(value)
        if use_graph_setter:
            setter(self.G, self.n, value)
        else:
            setter(self.G.nodes[self.n], aliases, value)

    return property(fget, fset)


class NodoProtocol(Protocol):
    """Protocolo mínimo para nodos TNFR."""

    EPI: float
    vf: float
    theta: float
    Si: float
    epi_kind: str
    dnfr: float
    d2EPI: float
    graph: Dict[str, object]

    def neighbors(self) -> Iterable["NodoProtocol"]:
        ...

    def push_glifo(self, glifo: str, window: int) -> None:
        ...

    def has_edge(self, other: "NodoProtocol") -> bool:
        ...

    def add_edge(
        self, other: "NodoProtocol", weight: float, *, overwrite: bool = False
    ) -> None:
        ...

    def offset(self) -> int:
        ...

    def all_nodes(self) -> Iterable["NodoProtocol"]:
        ...


@dataclass(eq=False)
class NodoTNFR:
    """Representa un nodo TNFR autónomo.

    Para cada vecino se almacena el peso de la conexión. Aunque las
    operaciones actuales no usan los pesos, se preservan para posibles
    cálculos futuros.
    """

    EPI: float = 0.0
    vf: float = 0.0
    theta: float = 0.0
    Si: float = 0.0
    epi_kind: str = ""
    dnfr: float = 0.0
    d2EPI: float = 0.0
    graph: Dict[str, object] = field(default_factory=dict)
    _neighbors: Dict["NodoTNFR", float] = field(default_factory=dict)
    _hist_glifos: Deque[str] = field(default_factory=lambda: deque(maxlen=DEFAULTS.get("GLYPH_HYSTERESIS_WINDOW", 7)))

    def neighbors(self) -> Iterable["NodoTNFR"]:
        return self._neighbors.keys()

    def has_edge(self, other: "NodoTNFR") -> bool:
        return other in self._neighbors

    def edge_weight(self, other: "NodoTNFR") -> float:
        """Devuelve el peso de la arista hacia ``other`` o ``0.0`` si no existe."""
        return self._neighbors.get(other, 0.0)

    def add_edge(
        self, other: "NodoTNFR", weight: float = 1.0, *, overwrite: bool = False
    ) -> None:
        """Conecta este nodo con ``other``.

        Si la arista ya existe, el peso almacenado se conserva a menos que
        ``overwrite`` sea ``True``, en cuyo caso se actualiza al nuevo
        ``weight``.
        """

        if other is self:
            return
        if other in self._neighbors and not overwrite:
            return
        self._neighbors[other] = weight
        other._neighbors[self] = weight

    def push_glifo(self, glifo: str, window: int) -> None:
        nd = {"hist_glifos": self._hist_glifos}
        push_glifo(nd, glifo, window)
        self._hist_glifos = nd["hist_glifos"]
        self.epi_kind = glifo

    def offset(self) -> int:
        return 0

    def all_nodes(self) -> Iterable["NodoTNFR"]:
        return list(self.graph.get("_all_nodes", [self]))

    def aplicar_glifo(self, glifo: str, window: Optional[int] = None) -> None:
        aplicar_glifo_obj(self, glifo, window=window)

    def integrar(self, dt: float) -> None:
        self.EPI += self.dnfr * dt


class NodoNX(NodoProtocol):
    """Adaptador para nodos ``networkx``."""

    def __init__(self, G, n):
        self.G = G
        self.n = n
        self.graph = G.graph

    EPI = _nx_attr_property(ALIAS_EPI)
    vf = _nx_attr_property(ALIAS_VF, setter=set_vf, use_graph_setter=True)
    theta = _nx_attr_property(ALIAS_THETA)
    Si = _nx_attr_property(ALIAS_SI)
    epi_kind = _nx_attr_property(
        ALIAS_EPI_KIND,
        default="",
        getter=get_attr_str,
        setter=set_attr_str,
        to_python=str,
        to_storage=str,
    )
    dnfr = _nx_attr_property(ALIAS_DNFR, setter=set_dnfr, use_graph_setter=True)
    d2EPI = _nx_attr_property(ALIAS_D2EPI)

    def neighbors(self) -> Iterable[NodoProtocol]:
        return (NodoNX(self.G, v) for v in self.G.neighbors(self.n))

    def push_glifo(self, glifo: str, window: int) -> None:
        push_glifo(self.G.nodes[self.n], glifo, window)
        self.epi_kind = glifo

    def has_edge(self, other: NodoProtocol) -> bool:
        if isinstance(other, NodoNX):
            return self.G.has_edge(self.n, other.n)
        raise NotImplementedError

    def add_edge(
        self, other: NodoProtocol, weight: float, *, overwrite: bool = False
    ) -> None:
        if other is self:
            return
        if isinstance(other, NodoNX):
            if self.G.has_edge(self.n, other.n) and not overwrite:
                return
            self.G.add_edge(self.n, other.n, weight=float(weight))
        else:
            raise NotImplementedError

    def offset(self) -> int:
        from .operators import _node_offset
        return _node_offset(self.G, self.n)

    def all_nodes(self) -> Iterable[NodoProtocol]:
        return (NodoNX(self.G, v) for v in self.G.nodes())
