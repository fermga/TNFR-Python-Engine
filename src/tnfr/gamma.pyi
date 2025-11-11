from __future__ import annotations

from typing import Callable, NamedTuple

from .types import GammaSpec, NodeId, TNFRGraph

__all__: tuple[str, ...]

class GammaEntry(NamedTuple):
    fn: Callable[[TNFRGraph, NodeId, float | int, GammaSpec], float]
    needs_kuramoto: bool

GAMMA_REGISTRY: dict[str, GammaEntry]

def kuramoto_R_psi(G: TNFRGraph) -> tuple[float, float]: ...
def gamma_none(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float: ...
def gamma_kuramoto_linear(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float: ...
def gamma_kuramoto_bandpass(
    G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec
) -> float: ...
def gamma_kuramoto_tanh(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float: ...
def gamma_harmonic(G: TNFRGraph, node: NodeId, t: float | int, cfg: GammaSpec) -> float: ...
def eval_gamma(
    G: TNFRGraph,
    node: NodeId,
    t: float | int,
    *,
    strict: bool = ...,
    log_level: int | None = ...,
) -> float: ...
