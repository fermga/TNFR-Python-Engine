"""Node utilities and structures for TNFR graphs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable, Optional, Protocol, TypeVar
from enum import Enum
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
    set_theta,
)
from .helpers.cache import increment_edge_version

from .operators import apply_glyph_obj

T = TypeVar("T")

__all__ = ("NodoTNFR", "NodoNX", "NodoProtocol", "EdgeStrategy")


def _nx_attr_property(
    aliases: tuple[str, ...],
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
    if not math.isfinite(weight):
        raise ValueError("Edge weight must be a finite number")
    if weight < 0:
        raise ValueError("Edge weight must be non-negative")

    return weight


class EdgeStrategy(Enum):
    NX = "nx"
    TNFR = "tnfr"


class EdgeOps(Protocol):
    """Interface for edge operations."""

    def exists(self, graph, n1, n2): ...

    def set(self, graph, n1, n2, w: float) -> None: ...


@dataclass(frozen=True, slots=True)
class NXEdgeOps:
    def exists(self, graph, n1, n2):
        return graph.has_edge(n1, n2)

    def set(self, graph, n1, n2, w: float) -> None:
        graph.add_edge(n1, n2, weight=w)


@dataclass(frozen=True, slots=True)
class TNFREdgeOps:
    def exists(self, graph, n1, n2):
        return n2 in n1._neighbors

    def set(self, graph, n1, n2, w: float) -> None:
        n1._neighbors[n2] = w
        n2._neighbors[n1] = w


@dataclass(frozen=True, slots=True)
class _CallbackEdgeOps:
    exists_cb: Callable
    set_cb: Callable

    def exists(self, graph, n1, n2):
        return self.exists_cb(graph, n1, n2)

    def set(self, graph, n1, n2, w: float) -> None:
        self.set_cb(graph, n1, n2, w)


_EDGE_OPS: dict[EdgeStrategy, EdgeOps] = {
    EdgeStrategy.NX: NXEdgeOps(),
    EdgeStrategy.TNFR: TNFREdgeOps(),
}


def _validate_callbacks(exists_cb, set_cb) -> None:
    """Validate callback pair provided to :func:`add_edge`."""

    if (exists_cb is None) != (set_cb is None):
        raise ValueError("exists_cb and set_cb must be provided together")

    if exists_cb is not None and set_cb is not None:
        if not callable(exists_cb) or not callable(set_cb):
            raise TypeError("exists_cb and set_cb must be callables")


def _resolve_edge_ops(graph, strategy, exists_cb, set_cb):
    if exists_cb is not None and set_cb is not None:
        return _CallbackEdgeOps(exists_cb, set_cb)
    if strategy is None:
        strategy = (
            EdgeStrategy.NX
            if hasattr(graph, "add_edge")
            else EdgeStrategy.TNFR
        )
    ops = _EDGE_OPS.get(strategy)
    if ops is None:
        raise ValueError("Unknown edge strategy")
    return ops


def add_edge(
    graph,
    n1,
    n2,
    weight,
    overwrite: bool = False,
    *,
    strategy: EdgeStrategy | None = None,
    exists_cb=None,
    set_cb=None,
):
    """Add an edge between ``n1`` and ``n2`` using the given strategy.

    ``strategy`` can be ``"nx"`` for :mod:`networkx` graphs or ``"tnfr"`` for
    TNFR nodes. Custom callbacks may be supplied via ``exists_cb`` and
    ``set_cb``; both must be callables and provided together.
    """

    _validate_callbacks(exists_cb, set_cb)

    ops = _resolve_edge_ops(graph, strategy, exists_cb, set_cb)

    if ops.exists(graph, n1, n2) and not overwrite:
        return

    weight = _add_edge_common(n1, n2, weight)
    if weight is None:
        return

    ops.set(graph, n1, n2, weight)
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
    graph: dict[str, object]

    def neighbors(self) -> Iterable[NodoProtocol | Hashable]: ...

    def push_glyph(self, glyph: str, window: int) -> None: ...

    def has_edge(self, other: "NodoProtocol") -> bool: ...

    def add_edge(
        self, other: "NodoProtocol", weight: float, *, overwrite: bool = False
    ) -> None: ...

    def offset(self) -> int: ...

    def all_nodes(self) -> Iterable["NodoProtocol"]: ...


@dataclass(eq=False, slots=True)
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
    graph: dict[str, object] = field(default_factory=dict)
    _neighbors: dict["NodoTNFR", float] = field(default_factory=dict)
    _glyph_history: Deque[str] = field(
        default_factory=lambda: deque(
            maxlen=DEFAULTS.get("GLYPH_HYSTERESIS_WINDOW", 7)
        )
    )

    def neighbors(self) -> Iterable["NodoTNFR"]:
        return self._neighbors.keys()

    def has_edge(self, other: "NodoTNFR") -> bool:
        return other in self._neighbors

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
            strategy=EdgeStrategy.TNFR,
        )

    def push_glyph(self, glyph: str, window: int) -> None:
        push_glyph({"glyph_history": self._glyph_history}, glyph, window)
        self.epi_kind = glyph

    def offset(self) -> int:
        return 0

    def all_nodes(self) -> Iterable["NodoTNFR"]:
        nodes = self.graph.get("_all_nodes")
        return nodes if nodes is not None else [self]

    def apply_glyph(self, glyph: str, window: Optional[int] = None) -> None:
        apply_glyph_obj(self, glyph, window=window)


class NodoNX(NodoProtocol):
    """Adapter for ``networkx`` nodes."""

    def __init__(self, G, n):
        self.G = G
        self.n = n
        self.graph = G.graph
        G.graph.setdefault("_node_cache", {})[n] = self

    EPI = _nx_attr_property(aliases=ALIAS_EPI)
    vf = _nx_attr_property(
        aliases=ALIAS_VF, setter=set_vf, use_graph_setter=True
    )
    theta = _nx_attr_property(
        aliases=ALIAS_THETA, setter=set_theta, use_graph_setter=True
    )
    Si = _nx_attr_property(aliases=ALIAS_SI)
    epi_kind = _nx_attr_property(
        aliases=ALIAS_EPI_KIND,
        default="",
        getter=get_attr_str,
        setter=set_attr_str,
        to_python=str,
        to_storage=str,
    )
    dnfr = _nx_attr_property(
        aliases=ALIAS_DNFR, setter=set_dnfr, use_graph_setter=True
    )
    d2EPI = _nx_attr_property(aliases=ALIAS_D2EPI)

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
                strategy=EdgeStrategy.NX,
            )
        else:
            raise NotImplementedError

    def offset(self) -> int:
        from .operators import _node_offset

        return _node_offset(self.G, self.n)

    def all_nodes(self) -> Iterable[NodoProtocol]:
        return (NodoNX.from_graph(self.G, v) for v in self.G.nodes())
