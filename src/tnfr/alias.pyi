from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, Hashable, TypeVar

from .types import FloatArray, NodeId

if TYPE_CHECKING:
    import networkx as nx

T = TypeVar("T")

__all__: list[str]

def __getattr__(name: str) -> Any: ...

class AbsMaxResult:
    max_value: float
    node: Hashable | None

SCALAR_SETTERS: dict[str, dict[str, Any]]

def get_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: T | None = ...,
    *,
    strict: bool = ...,
    log_level: int | None = ...,
    conv: Callable[[Any], T] = ...,
) -> T | None: ...
def get_theta_attr(
    d: Mapping[str, Any],
    default: T | None = ...,
    *,
    strict: bool = ...,
    log_level: int | None = ...,
    conv: Callable[[Any], T] = ...,
) -> T | None: ...
def collect_attr(
    G: "nx.Graph",
    nodes: Iterable[NodeId],
    aliases: Iterable[str],
    default: float = ...,
    *,
    np: ModuleType | None = ...,
) -> FloatArray | list[float]: ...
def collect_theta_attr(
    G: "nx.Graph",
    nodes: Iterable[NodeId],
    default: float = ...,
    *,
    np: ModuleType | None = ...,
) -> FloatArray | list[float]: ...
def set_attr_generic(
    d: dict[str, Any],
    aliases: Iterable[str],
    value: Any,
    *,
    conv: Callable[[Any], T],
) -> T: ...
def set_attr(
    d: dict[str, Any],
    aliases: Iterable[str],
    value: Any,
    conv: Callable[[Any], T] = ...,
) -> T: ...
def get_attr_str(
    d: dict[str, Any],
    aliases: Iterable[str],
    default: str | None = ...,
    *,
    strict: bool = ...,
    log_level: int | None = ...,
    conv: Callable[[Any], str] = ...,
) -> str | None: ...
def set_attr_str(d: dict[str, Any], aliases: Iterable[str], value: Any) -> str: ...
def set_theta_attr(d: MutableMapping[str, Any], value: Any) -> float: ...
def multi_recompute_abs_max(
    G: "nx.Graph", alias_map: Mapping[str, tuple[str, ...]]
) -> dict[str, float]: ...
def set_attr_and_cache(
    G: "nx.Graph",
    n: Hashable,
    aliases: tuple[str, ...],
    value: float,
    *,
    cache: str | None = ...,
    extra: Callable[["nx.Graph", Hashable, float], None] | None = ...,
) -> AbsMaxResult | None: ...
def set_attr_with_max(
    G: "nx.Graph", n: Hashable, aliases: tuple[str, ...], value: float, *, cache: str
) -> AbsMaxResult: ...
def set_scalar(
    G: "nx.Graph",
    n: Hashable,
    alias: tuple[str, ...],
    value: float,
    *,
    cache: str | None = ...,
    extra: Callable[["nx.Graph", Hashable, float], None] | None = ...,
) -> AbsMaxResult | None: ...
def set_vf(
    G: "nx.Graph", n: Hashable, value: float, *, update_max: bool = ...
) -> AbsMaxResult | None: ...
def set_dnfr(G: "nx.Graph", n: Hashable, value: float) -> AbsMaxResult | None: ...
def set_theta(G: "nx.Graph", n: Hashable, value: float) -> AbsMaxResult | None: ...
