"""Node utilities and structures for TNFR graphs."""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Iterable,
    MutableMapping,
    Optional,
    Protocol,
    SupportsFloat,
    TypeVar,
)
from collections.abc import Hashable
import math
from dataclasses import dataclass

from .constants import get_aliases
from .alias import (
    get_attr,
    get_theta_attr,
    get_attr_str,
    set_attr,
    set_attr_str,
    set_vf,
    set_dnfr,
    set_theta,
)
from .types import (
    CouplingWeight,
    DeltaNFR,
    EPIValue,
    NodeId,
    Phase,
    SecondDerivativeEPI,
    SenseIndex,
    StructuralFrequency,
    TNFRGraph,
)
from .utils import (
    cached_node_list,
    ensure_node_offset_map,
    increment_edge_version,
    supports_add_edge,
)
from .locking import get_lock

ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_THETA = get_aliases("THETA")
ALIAS_SI = get_aliases("SI")
ALIAS_EPI_KIND = get_aliases("EPI_KIND")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_D2EPI = get_aliases("D2EPI")

T = TypeVar("T")

__all__ = ("NodeNX", "NodeProtocol", "add_edge")


@dataclass(frozen=True)
class AttrSpec:
    """Configuration required to expose a ``networkx`` node attribute.

    ``AttrSpec`` mirrors the defaults previously used by
    :func:`_nx_attr_property` and centralises the descriptor generation
    logic to keep a single source of truth for NodeNX attribute access.
    """

    aliases: tuple[str, ...]
    default: Any = 0.0
    getter: Callable[[MutableMapping[str, Any], tuple[str, ...], Any], Any] = get_attr
    setter: Callable[..., None] = set_attr
    to_python: Callable[[Any], Any] = float
    to_storage: Callable[[Any], Any] = float
    use_graph_setter: bool = False

    def build_property(self) -> property:
        """Create the property descriptor for ``NodeNX`` attributes."""

        def fget(instance: "NodeNX") -> T:
            return self.to_python(
                self.getter(instance.G.nodes[instance.n], self.aliases, self.default)
            )

        def fset(instance: "NodeNX", value: T) -> None:
            value = self.to_storage(value)
            if self.use_graph_setter:
                self.setter(instance.G, instance.n, value)
            else:
                self.setter(instance.G.nodes[instance.n], self.aliases, value)

        return property(fget, fset)


# Mapping of NodeNX attribute specifications used to generate property
# descriptors. Each entry defines the keyword arguments passed to
# ``AttrSpec.build_property`` for a given attribute name.
ATTR_SPECS: dict[str, AttrSpec] = {
    "EPI": AttrSpec(aliases=ALIAS_EPI),
    "vf": AttrSpec(aliases=ALIAS_VF, setter=set_vf, use_graph_setter=True),
    "theta": AttrSpec(
        aliases=ALIAS_THETA,
        getter=lambda mapping, _aliases, default: get_theta_attr(mapping, default),
        setter=set_theta,
        use_graph_setter=True,
    ),
    "Si": AttrSpec(aliases=ALIAS_SI),
    "epi_kind": AttrSpec(
        aliases=ALIAS_EPI_KIND,
        default="",
        getter=get_attr_str,
        setter=set_attr_str,
        to_python=str,
        to_storage=str,
    ),
    "dnfr": AttrSpec(aliases=ALIAS_DNFR, setter=set_dnfr, use_graph_setter=True),
    "d2EPI": AttrSpec(aliases=ALIAS_D2EPI),
}


def _add_edge_common(
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
) -> Optional[CouplingWeight]:
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
    graph: TNFRGraph,
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
    overwrite: bool = False,
) -> None:
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


class NodeProtocol(Protocol):
    """Minimal protocol for TNFR nodes."""

    EPI: EPIValue
    vf: StructuralFrequency
    theta: Phase
    Si: SenseIndex
    epi_kind: str
    dnfr: DeltaNFR
    d2EPI: SecondDerivativeEPI
    graph: MutableMapping[str, Any]

    def neighbors(self) -> Iterable[NodeProtocol | Hashable]:
        ...

    def _glyph_storage(self) -> MutableMapping[str, object]:
        ...

    def has_edge(self, other: "NodeProtocol") -> bool:
        ...

    def add_edge(
        self,
        other: NodeProtocol,
        weight: CouplingWeight,
        *,
        overwrite: bool = False,
    ) -> None:
        ...

    def offset(self) -> int:
        ...

    def all_nodes(self) -> Iterable[NodeProtocol]:
        ...


class NodeNX(NodeProtocol):
    """Adapter for ``networkx`` nodes."""

    # Statically defined property descriptors for ``NodeNX`` attributes.
    # Declaring them here makes the attributes discoverable by type checkers
    # and IDEs, avoiding the previous runtime ``setattr`` loop.
    EPI: EPIValue = ATTR_SPECS["EPI"].build_property()
    vf: StructuralFrequency = ATTR_SPECS["vf"].build_property()
    theta: Phase = ATTR_SPECS["theta"].build_property()
    Si: SenseIndex = ATTR_SPECS["Si"].build_property()
    epi_kind: str = ATTR_SPECS["epi_kind"].build_property()
    dnfr: DeltaNFR = ATTR_SPECS["dnfr"].build_property()
    d2EPI: SecondDerivativeEPI = ATTR_SPECS["d2EPI"].build_property()

    def __init__(self, G: TNFRGraph, n: NodeId) -> None:
        self.G: TNFRGraph = G
        self.n: NodeId = n
        self.graph: MutableMapping[str, Any] = G.graph
        G.graph.setdefault("_node_cache", {})[n] = self

    def _glyph_storage(self) -> MutableMapping[str, Any]:
        return self.G.nodes[self.n]

    @classmethod
    def from_graph(cls, G: TNFRGraph, n: NodeId) -> "NodeNX":
        """Return cached ``NodeNX`` for ``(G, n)`` with thread safety."""
        lock = get_lock(f"node_nx_cache_{id(G)}")
        with lock:
            cache = G.graph.setdefault("_node_cache", {})
            node = cache.get(n)
            if node is None:
                node = cls(G, n)
            return node

    def neighbors(self) -> Iterable[NodeId]:
        """Iterate neighbour identifiers (IDs).

        Wrap each resulting ID with :meth:`from_graph` to obtain the cached
        ``NodeNX`` instance when actual node objects are required.
        """
        return self.G.neighbors(self.n)

    def has_edge(self, other: NodeProtocol) -> bool:
        if isinstance(other, NodeNX):
            return self.G.has_edge(self.n, other.n)
        raise NotImplementedError

    def add_edge(
        self,
        other: NodeProtocol,
        weight: CouplingWeight,
        *,
        overwrite: bool = False,
    ) -> None:
        if isinstance(other, NodeNX):
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

    def all_nodes(self) -> Iterable[NodeProtocol]:
        override = self.graph.get("_all_nodes")
        if override is not None:
            return override

        nodes = cached_node_list(self.G)
        return tuple(NodeNX.from_graph(self.G, v) for v in nodes)
