"""Node utilities and structures for TNFR graphs."""

from __future__ import annotations
from typing import Iterable, MutableMapping, Optional, Protocol, TypeVar
from collections.abc import Hashable
import math

from .constants import get_aliases
from .alias import (
    get_attr,
    get_attr_str,
    set_attr,
    set_attr_str,
    set_vf,
    set_dnfr,
    set_theta,
)
from .cache import (
    cached_node_list,
    ensure_node_offset_map,
    increment_edge_version,
)
from .utils import supports_add_edge
from .locking import get_lock

ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_THETA = get_aliases("THETA")
ALIAS_SI = get_aliases("SI")
ALIAS_EPI_KIND = get_aliases("EPI_KIND")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_D2EPI = get_aliases("D2EPI")

# Mapping of NodoNX attribute specifications used to generate property
# descriptors. Each entry defines the keyword arguments passed to
# ``_nx_attr_property`` for a given attribute name.
ATTR_SPECS: dict[str, dict] = {
    "EPI": {"aliases": ALIAS_EPI},
    "vf": {
        "aliases": ALIAS_VF,
        "setter": set_vf,
        "use_graph_setter": True,
    },
    "theta": {
        "aliases": ALIAS_THETA,
        "setter": set_theta,
        "use_graph_setter": True,
    },
    "Si": {"aliases": ALIAS_SI},
    "epi_kind": {
        "aliases": ALIAS_EPI_KIND,
        "default": "",
        "getter": get_attr_str,
        "setter": set_attr_str,
        "to_python": str,
        "to_storage": str,
    },
    "dnfr": {
        "aliases": ALIAS_DNFR,
        "setter": set_dnfr,
        "use_graph_setter": True,
    },
    "d2EPI": {"aliases": ALIAS_D2EPI},
}

T = TypeVar("T")

__all__ = ("NodoNX", "NodoProtocol", "add_edge")


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


def add_edge(
    graph,
    n1,
    n2,
    weight,
    overwrite: bool = False,
):
    """Add an edge between ``n1`` and ``n2`` in a ``networkx`` graph."""

    weight = _add_edge_common(n1, n2, weight)
    if weight is None:
        return

    if not supports_add_edge(graph):
        raise TypeError("add_edge only supports networkx graphs")

    if graph.has_edge(n1, n2) and not overwrite:
        return

    graph.add_edge(n1, n2, weight=weight)
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

    def _glyph_storage(self) -> MutableMapping[str, object]: ...

    def has_edge(self, other: "NodoProtocol") -> bool: ...

    def add_edge(
        self, other: "NodoProtocol", weight: float, *, overwrite: bool = False
    ) -> None: ...

    def offset(self) -> int: ...

    def all_nodes(self) -> Iterable["NodoProtocol"]: ...


class NodoNX(NodoProtocol):
    """Adapter for ``networkx`` nodes."""

    # Statically defined property descriptors for ``NodoNX`` attributes.
    # Declaring them here makes the attributes discoverable by type checkers
    # and IDEs, avoiding the previous runtime ``setattr`` loop.
    EPI: float = _nx_attr_property(**ATTR_SPECS["EPI"])
    vf: float = _nx_attr_property(**ATTR_SPECS["vf"])
    theta: float = _nx_attr_property(**ATTR_SPECS["theta"])
    Si: float = _nx_attr_property(**ATTR_SPECS["Si"])
    epi_kind: str = _nx_attr_property(**ATTR_SPECS["epi_kind"])
    dnfr: float = _nx_attr_property(**ATTR_SPECS["dnfr"])
    d2EPI: float = _nx_attr_property(**ATTR_SPECS["d2EPI"])

    def __init__(self, G, n):
        self.G = G
        self.n = n
        self.graph = G.graph
        G.graph.setdefault("_node_cache", {})[n] = self

    def _glyph_storage(self):
        return self.G.nodes[self.n]

    @classmethod
    def from_graph(cls, G, n):
        """Return cached ``NodoNX`` for ``(G, n)`` with thread safety."""
        lock = get_lock(f"nodonx_cache_{id(G)}")
        with lock:
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
            )
        else:
            raise NotImplementedError

    def offset(self) -> int:
        mapping = ensure_node_offset_map(self.G)
        return mapping.get(self.n, 0)

    def all_nodes(self) -> Iterable[NodoProtocol]:
        override = self.graph.get("_all_nodes")
        if override is not None:
            return override

        nodes = cached_node_list(self.G)
        return tuple(NodoNX.from_graph(self.G, v) for v in nodes)


