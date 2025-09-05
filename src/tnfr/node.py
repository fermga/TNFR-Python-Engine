"""Operaciones sobre nodos."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional, Protocol, TypeVar
from collections import deque
from collections.abc import Hashable

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
from .glyph_history import push_glyph
from .helpers import (
    get_attr,
    get_attr_str,
    set_attr,
    set_attr_str,
    set_vf,
    set_dnfr,
    increment_edge_version,
)

from .operators import apply_glyph_obj

T = TypeVar("T")

__all__ = ["NodoTNFR", "NodoNX", "NodoProtocol"]


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
        Tupla inmutable de aliases usados para acceder al atributo en el nodo
        ``networkx`` subyacente.
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

    def fget(self) -> T:
        return to_python(getter(self.G.nodes[self.n], aliases, default))

    def fset(self, value: T) -> None:
        value = to_storage(value)
        if use_graph_setter:
            setter(self.G, self.n, value)
        else:
            setter(self.G.nodes[self.n], aliases, value)

    return property(fget, fset)


def _add_edge_common(n1, n2, weight, overwrite):
    """Validate basic edge constraints.

    Returns the parsed weight if the edge can be added. ``None`` is returned
    when the edge should be ignored (e.g. self-connections).
    """

    if n1 == n2:
        return None

    weight = float(weight)
    if weight < 0:
        raise ValueError("Edge weight must be non-negative")

    return weight


def _add_edge_nx(G, n1, n2, weight, overwrite):
    """Add an edge between ``n1`` and ``n2`` in a ``networkx`` graph."""

    weight = _add_edge_common(n1, n2, weight, overwrite)
    if weight is None:
        return

    if G.has_edge(n1, n2) and not overwrite:
        return
    G.add_edge(n1, n2, weight=weight)
    increment_edge_version(G)


def _add_edge_tnfr(graph, n1, n2, weight, overwrite):
    """Add an edge between ``n1`` and ``n2`` in a TNFR-style graph."""

    weight = _add_edge_common(n1, n2, weight, overwrite)
    if weight is None:
        return

    if n2 in n1._neighbors and not overwrite:
        return
    n1._neighbors[n2] = weight
    n2._neighbors[n1] = weight
    increment_edge_version(graph)


class NodoProtocol(Protocol):
    """Minimal protocol for TNFR nodes."""

    EPI: float
    vf: float
    theta: float
    Si: float
    epi_kind: str
    dnfr: float
    d2EPI: float
    graph: Dict[str, object]

    def neighbors(self) -> Iterable[NodoProtocol | Hashable]: ...

    def push_glyph(self, glyph: str, window: int) -> None: ...

    def has_edge(self, other: "NodoProtocol") -> bool: ...

    def add_edge(
        self, other: "NodoProtocol", weight: float, *, overwrite: bool = False
    ) -> None: ...

    def offset(self) -> int: ...

    def all_nodes(self) -> Iterable["NodoProtocol"]: ...


@dataclass(eq=False)
class NodoTNFR:
    """Autonomous TNFR node representation.

    Each neighbour stores the connection weight. Although current operations
    do not use the weights, they are preserved for potential future
    calculations.
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
    _glyph_history: Deque[str] = field(
        default_factory=lambda: deque(maxlen=DEFAULTS.get("GLYPH_HYSTERESIS_WINDOW", 7))
    )

    def neighbors(self) -> Iterable["NodoTNFR"]:
        return self._neighbors.keys()

    def has_edge(self, other: "NodoTNFR") -> bool:
        return other in self._neighbors

    def edge_weight(self, other: "NodoTNFR") -> float:
        """Return the edge weight towards ``other`` or ``0.0`` if absent."""
        return self._neighbors.get(other, 0.0)

    def add_edge(
        self, other: "NodoTNFR", weight: float = 1.0, *, overwrite: bool = False
    ) -> None:
        """Connect this node with ``other``."""

        _add_edge_tnfr(self.graph, self, other, weight, overwrite)

    def push_glyph(self, glyph: str, window: int) -> None:
        push_glyph({"glyph_history": self._glyph_history}, glyph, window)
        self.epi_kind = glyph

    def offset(self) -> int:
        return 0

    def all_nodes(self) -> Iterable["NodoTNFR"]:
        return list(self.graph.get("_all_nodes", [self]))

    def apply_glyph(self, glyph: str, window: Optional[int] = None) -> None:
        apply_glyph_obj(self, glyph, window=window)

    def integrar(self, dt: float) -> None:
        self.EPI += self.dnfr * dt


class NodoNX(NodoProtocol):
    """Adaptador para nodos ``networkx``."""

    def __init__(self, G, n):
        self.G = G
        self.n = n
        self.graph = G.graph
        G.graph.setdefault("_node_cache", {})[n] = self

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

    @classmethod
    def from_graph(cls, G, n):
        """Return cached ``NodoNX`` for ``(G, n)``."""
        cache = G.graph.setdefault("_node_cache", {})
        node = cache.get(n)
        if node is None:
            node = cls(G, n)
        return node

    def neighbors(self) -> Iterable[Hashable]:
        """Itera identificadores de vecinos.

        Usa :meth:`from_graph` para obtener instancias ``NodoNX`` cacheadas.
        """
        return self.G.neighbors(self.n)

    def push_glyph(self, glyph: str, window: int) -> None:
        push_glyph(self.G.nodes[self.n], glyph, window)
        self.epi_kind = glyph

    def has_edge(self, other: NodoProtocol) -> bool:
        if isinstance(other, NodoNX):
            return self.G.has_edge(self.n, other.n)
        raise NotImplementedError

    def add_edge(
        self, other: NodoProtocol, weight: float, *, overwrite: bool = False
    ) -> None:
        if isinstance(other, NodoNX):
            _add_edge_nx(self.G, self.n, other.n, weight, overwrite)
        else:
            raise NotImplementedError

    def offset(self) -> int:
        from .operators import _node_offset

        return _node_offset(self.G, self.n)

    def all_nodes(self) -> Iterable[NodoProtocol]:
        return (NodoNX.from_graph(self.G, v) for v in self.G.nodes())
