from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from .types import NodeInitAttrMap

__all__: tuple[str, str] = ("InitParams", "init_node_attrs")

@dataclass
class InitParams:
    seed: int | None
    init_rand_phase: bool
    th_min: float
    th_max: float
    vf_mode: str
    vf_min_lim: float
    vf_max_lim: float
    vf_uniform_min: float | None
    vf_uniform_max: float | None
    vf_mean: float
    vf_std: float
    clamp_to_limits: bool
    si_min: float
    si_max: float
    epi_val: float

    @classmethod
    def from_graph(cls, G: nx.Graph) -> InitParams: ...

def _init_phase(
    nd: NodeInitAttrMap,
    rng: random.Random,
    *,
    override: bool,
    random_phase: bool,
    th_min: float,
    th_max: float,
) -> None: ...
def _init_vf(
    nd: NodeInitAttrMap,
    rng: random.Random,
    *,
    override: bool,
    mode: str,
    vf_uniform_min: float,
    vf_uniform_max: float,
    vf_mean: float,
    vf_std: float,
    vf_min_lim: float,
    vf_max_lim: float,
    clamp_to_limits: bool,
) -> None: ...
def _init_si_epi(
    nd: NodeInitAttrMap,
    rng: random.Random,
    *,
    override: bool,
    si_min: float,
    si_max: float,
    epi_val: float,
) -> None: ...
def init_node_attrs(G: nx.Graph, *, override: bool = True) -> nx.Graph: ...
