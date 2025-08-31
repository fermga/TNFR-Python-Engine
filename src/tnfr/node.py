from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Protocol
from collections import deque

from .constants import DEFAULTS
from .helpers import push_glifo


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

    def add_edge(self, other: "NodoProtocol", weight: float) -> None:
        ...

    def offset(self) -> int:
        ...

    def all_nodes(self) -> Iterable["NodoProtocol"]:
        ...


@dataclass
class NodoTNFR:
    """Representa un nodo TNFR autónomo."""

    EPI: float = 0.0
    vf: float = 0.0
    theta: float = 0.0
    Si: float = 0.0
    epi_kind: str = ""
    dnfr: float = 0.0
    d2EPI: float = 0.0
    graph: Dict[str, object] = field(default_factory=dict)
    _neighbors: List["NodoTNFR"] = field(default_factory=list)
    _hist_glifos: Deque[str] = field(default_factory=lambda: deque(maxlen=DEFAULTS.get("GLYPH_HYSTERESIS_WINDOW", 7)))

    def neighbors(self) -> Iterable["NodoTNFR"]:
        return list(self._neighbors)

    def has_edge(self, other: "NodoTNFR") -> bool:
        return other in self._neighbors

    def add_edge(self, other: "NodoTNFR", weight: float = 1.0) -> None:
        if other not in self._neighbors:
            self._neighbors.append(other)
            other._neighbors.append(self)

    def push_glifo(self, glifo: str, window: int) -> None:
        self._hist_glifos.append(glifo)
        while len(self._hist_glifos) > window:
            self._hist_glifos.popleft()
        self.epi_kind = glifo

    def offset(self) -> int:
        return 0

    def all_nodes(self) -> Iterable["NodoTNFR"]:
        return list(getattr(self.graph, "_all_nodes", [self]))

    def aplicar_glifo(self, glifo: str, window: Optional[int] = None) -> None:
        from .operators import aplicar_glifo_obj
        aplicar_glifo_obj(self, glifo, window=window)

    def integrar(self, dt: float) -> None:
        self.EPI += self.dnfr * dt


class NodoNX(NodoProtocol):
    """Adaptador para nodos ``networkx``."""

    def __init__(self, G, n):
        self.G = G
        self.n = n
        self.graph = G.graph

    @property
    def EPI(self) -> float:
        from .helpers import _get_attr
        from .constants import ALIAS_EPI
        return float(_get_attr(self.G.nodes[self.n], ALIAS_EPI, 0.0))

    @EPI.setter
    def EPI(self, v: float) -> None:
        from .helpers import _set_attr
        from .constants import ALIAS_EPI
        _set_attr(self.G.nodes[self.n], ALIAS_EPI, float(v))

    @property
    def vf(self) -> float:
        from .helpers import _get_attr
        from .constants import ALIAS_VF
        return float(_get_attr(self.G.nodes[self.n], ALIAS_VF, 0.0))

    @vf.setter
    def vf(self, v: float) -> None:
        from .helpers import _set_attr
        from .constants import ALIAS_VF
        _set_attr(self.G.nodes[self.n], ALIAS_VF, float(v))

    @property
    def theta(self) -> float:
        from .helpers import _get_attr
        from .constants import ALIAS_THETA
        return float(_get_attr(self.G.nodes[self.n], ALIAS_THETA, 0.0))

    @theta.setter
    def theta(self, v: float) -> None:
        from .helpers import _set_attr
        from .constants import ALIAS_THETA
        _set_attr(self.G.nodes[self.n], ALIAS_THETA, float(v))

    @property
    def Si(self) -> float:
        from .helpers import _get_attr
        from .constants import ALIAS_SI
        return float(_get_attr(self.G.nodes[self.n], ALIAS_SI, 0.0))

    @Si.setter
    def Si(self, v: float) -> None:
        from .helpers import _set_attr
        from .constants import ALIAS_SI
        _set_attr(self.G.nodes[self.n], ALIAS_SI, float(v))

    @property
    def epi_kind(self) -> str:
        from .helpers import _get_attr_str
        from .constants import ALIAS_EPI_KIND
        return _get_attr_str(self.G.nodes[self.n], ALIAS_EPI_KIND, "")

    @epi_kind.setter
    def epi_kind(self, v: str) -> None:
        from .helpers import _set_attr_str
        from .constants import ALIAS_EPI_KIND
        _set_attr_str(self.G.nodes[self.n], ALIAS_EPI_KIND, str(v))

    @property
    def dnfr(self) -> float:
        from .helpers import _get_attr
        from .constants import ALIAS_DNFR
        return float(_get_attr(self.G.nodes[self.n], ALIAS_DNFR, 0.0))

    @dnfr.setter
    def dnfr(self, v: float) -> None:
        from .helpers import _set_attr
        from .constants import ALIAS_DNFR
        _set_attr(self.G.nodes[self.n], ALIAS_DNFR, float(v))

    @property
    def d2EPI(self) -> float:
        from .helpers import _get_attr
        from .constants import ALIAS_D2EPI
        return float(_get_attr(self.G.nodes[self.n], ALIAS_D2EPI, 0.0))

    @d2EPI.setter
    def d2EPI(self, v: float) -> None:
        from .helpers import _set_attr
        from .constants import ALIAS_D2EPI
        _set_attr(self.G.nodes[self.n], ALIAS_D2EPI, float(v))

    def neighbors(self) -> Iterable[NodoProtocol]:
        return [NodoNX(self.G, v) for v in self.G.neighbors(self.n)]

    def push_glifo(self, glifo: str, window: int) -> None:
        push_glifo(self.G.nodes[self.n], glifo, window)
        self.epi_kind = glifo

    def has_edge(self, other: NodoProtocol) -> bool:
        if isinstance(other, NodoNX):
            return self.G.has_edge(self.n, other.n)
        raise NotImplementedError

    def add_edge(self, other: NodoProtocol, weight: float) -> None:
        if isinstance(other, NodoNX):
            self.G.add_edge(self.n, other.n, weight=float(weight))
        else:
            raise NotImplementedError

    def offset(self) -> int:
        from .operators import _node_offset
        return _node_offset(self.G, self.n)

    def all_nodes(self) -> Iterable[NodoProtocol]:
        return [NodoNX(self.G, v) for v in self.G.nodes()]
