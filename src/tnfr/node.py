"""Node operations."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional, Protocol, TypeVar
from collections import deque
from collections.abc import Hashable
import math

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
from .alias import (
    get_attr,
    get_attr_str,
    set_attr,
    set_attr_str,
    set_vf,
    set_dnfr,
)
from .helpers import increment_edge_version

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
        Immutable tuple of aliases used to access the attribute in the
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

    def fget(self) -> T:
        return to_python(getter(self.G.nodes[self.n], aliases, default))

    def fset(self, value: T) -> None:
        value = to_storage(value)
        if use_graph_setter:
            setter(self.G, self.n, value)
        else:
            setter(self.G.nodes[self.n], aliases, value)

    return property(fget, fset)


def _add_edge_common(n1, n2, weight) -> Optional[float]:
    """Validate basic edge constraints.

    Returns the parsed weight if the edge can be added. ``None`` is returned
    when the edge should be ignored (e.g. self-connections).
    """

    if n1 == n2:
        return None

    weight = float(weight)
    if math.isnan(weight):
        raise ValueError("Edge weight must be a number")
    if weight < 0:
        raise ValueError("Edge weight must be non-negative")

    return weight


def _exists_nx(graph, n1, n2):
    return graph.has_edge(n1, n2)


def _set_nx(graph, n1, n2, w: float) -> None:
    graph.add_edge(n1, n2, weight=w)


def _exists_tnfr(graph, n1, n2):
    return n2 in n1._neighbors


def _set_tnfr(graph, n1, n2, w: float) -> None:
    n1._neighbors[n2] = w
    n2._neighbors[n1] = w


_STRATEGY_CBS = {
    "nx": (_exists_nx, _set_nx),
    "tnfr": (_exists_tnfr, _set_tnfr),
}


def add_edge(
    graph,
    n1,
    n2,
    weight,
    overwrite,
    *,
    strategy: str | None = None,
    exists_cb=None,
    set_cb=None,
):
    """Add an edge between ``n1`` and ``n2`` using the given strategy.

    ``strategy`` can be ``"nx"`` for :mod:`networkx` graphs or ``"tnfr"`` for
    TNFR nodes. Custom callbacks may be supplied via ``exists_cb`` and
    ``set_cb``.
    """

    weight = _add_edge_common(n1, n2, weight)
    if weight is None:
        return

    if exists_cb is None or set_cb is None:
        if strategy is None:
            strategy = "nx" if hasattr(graph, "add_edge") else "tnfr"
        try:
            exists_cb, set_cb = _STRATEGY_CBS[strategy]
        except KeyError:
            raise ValueError("Unknown edge strategy")

    if exists_cb(graph, n1, n2) and not overwrite:
        return

    set_cb(graph, n1, n2, weight)
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
        default_factory=lambda: deque(
            maxlen=DEFAULTS.get("GLYPH_HYSTERESIS_WINDOW", 7)
        )
    )

    def neighbors(self) -> Iterable["NodoTNFR"]:
        return self._neighbors.keys()

    def has_edge(self, other: "NodoTNFR") -> bool:
        return other in self._neighbors

    def edge_weight(self, other: "NodoTNFR") -> float:
        """Return the edge weight towards ``other`` or ``0.0`` if absent."""
        return self._neighbors.get(other, 0.0)

    def add_edge(
        self,
        other: "NodoTNFR",
        weight: float = 1.0,
        *,
        overwrite: bool = False,
    ) -> None:
        """Connect this node with ``other``."""

        add_edge(
            self.graph,
            self,
            other,
            weight,
            overwrite,
            strategy="tnfr",
        )

    def push_glyph(self, glyph: str, window: int) -> None:
        push_glyph({"glyph_history": self._glyph_history}, glyph, window)
        self.epi_kind = glyph

    def offset(self) -> int:
        return 0

    def all_nodes(self) -> Iterable["NodoTNFR"]:
        return list(self.graph.get("_all_nodes", [self]))

    def apply_glyph(self, glyph: str, window: Optional[int] = None) -> None:
        apply_glyph_obj(self, glyph, window=window)


class NodoNX(NodoProtocol):
    """Adapter for ``networkx`` nodes."""

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
    dnfr = _nx_attr_property(
        ALIAS_DNFR, setter=set_dnfr, use_graph_setter=True
    )
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
        """Iterate neighbour identifiers (IDs).

        Wrap each resulting ID with :meth:`from_graph` to obtain the cached
        ``NodoNX`` instance when actual node objects are required.
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
            add_edge(
                self.G,
                self.n,
                other.n,
                weight,
                overwrite,
                strategy="nx",
            )
        else:
            raise NotImplementedError

    def offset(self) -> int:
        from .operators import _node_offset

        return _node_offset(self.G, self.n)

    def all_nodes(self) -> Iterable[NodoProtocol]:
        return (NodoNX.from_graph(self.G, v) for v in self.G.nodes())
