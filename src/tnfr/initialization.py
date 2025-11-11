"""Node initialization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .constants import THETA_KEY, VF_KEY, get_graph_param
from .utils import clamp
from .rng import make_rng
from .types import NodeInitAttrMap

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

__all__ = ("InitParams", "init_node_attrs")


@dataclass
class InitParams:
    """Parameters governing node initialisation."""

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
    def from_graph(cls, G: "nx.Graph") -> "InitParams":
        """Construct ``InitParams`` from ``G.graph`` configuration."""

        return cls(
            seed=get_graph_param(G, "RANDOM_SEED", int),
            init_rand_phase=get_graph_param(G, "INIT_RANDOM_PHASE", bool),
            th_min=get_graph_param(G, "INIT_THETA_MIN"),
            th_max=get_graph_param(G, "INIT_THETA_MAX"),
            vf_mode=str(get_graph_param(G, "INIT_VF_MODE", str)).lower(),
            vf_min_lim=get_graph_param(G, "VF_MIN"),
            vf_max_lim=get_graph_param(G, "VF_MAX"),
            vf_uniform_min=get_graph_param(G, "INIT_VF_MIN"),
            vf_uniform_max=get_graph_param(G, "INIT_VF_MAX"),
            vf_mean=get_graph_param(G, "INIT_VF_MEAN"),
            vf_std=get_graph_param(G, "INIT_VF_STD"),
            clamp_to_limits=get_graph_param(G, "INIT_VF_CLAMP_TO_LIMITS", bool),
            si_min=get_graph_param(G, "INIT_SI_MIN"),
            si_max=get_graph_param(G, "INIT_SI_MAX"),
            epi_val=get_graph_param(G, "INIT_EPI_VALUE"),
        )


def _init_phase(
    nd: NodeInitAttrMap,
    rng: random.Random,
    *,
    override: bool,
    random_phase: bool,
    th_min: float,
    th_max: float,
) -> None:
    """Initialise ``θ`` in ``nd``."""
    if random_phase:
        if override or THETA_KEY not in nd:
            nd[THETA_KEY] = rng.uniform(th_min, th_max)
    else:
        if override:
            nd[THETA_KEY] = 0.0
        else:
            nd.setdefault(THETA_KEY, 0.0)


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
) -> None:
    """Initialise ``νf`` in ``nd``."""
    if mode == "uniform":
        vf = rng.uniform(vf_uniform_min, vf_uniform_max)
    elif mode == "normal":
        for _ in range(16):
            cand = rng.normalvariate(vf_mean, vf_std)
            if vf_min_lim <= cand <= vf_max_lim:
                vf = cand
                break
        else:
            vf = min(
                max(rng.normalvariate(vf_mean, vf_std), vf_min_lim),
                vf_max_lim,
            )
    else:
        vf = float(nd.get(VF_KEY, 0.5))
    if clamp_to_limits:
        vf = clamp(vf, vf_min_lim, vf_max_lim)
    if override or VF_KEY not in nd:
        nd[VF_KEY] = vf


def _init_si_epi(
    nd: NodeInitAttrMap,
    rng: random.Random,
    *,
    override: bool,
    si_min: float,
    si_max: float,
    epi_val: float,
) -> None:
    """Initialise ``Si`` and ``EPI`` in ``nd``."""
    if override or "EPI" not in nd:
        nd["EPI"] = epi_val

    si = rng.uniform(si_min, si_max)
    if override or "Si" not in nd:
        nd["Si"] = si


def init_node_attrs(G: "nx.Graph", *, override: bool = True) -> "nx.Graph":
    """Initialise EPI, θ, νf and Si on the nodes of ``G``.

    Parameters can be customised via ``G.graph`` entries:
    ``RANDOM_SEED``, ``INIT_RANDOM_PHASE``, ``INIT_THETA_MIN/MAX``,
    ``INIT_VF_MODE``, ``VF_MIN``, ``VF_MAX``, ``INIT_VF_MIN/MAX``,
    ``INIT_VF_MEAN``, ``INIT_VF_STD`` and ``INIT_VF_CLAMP_TO_LIMITS``.
    Ranges for ``Si`` are added via ``INIT_SI_MIN`` and ``INIT_SI_MAX``, and
    for ``EPI`` via ``INIT_EPI_VALUE``. If ``INIT_VF_MIN`` is greater than
    ``INIT_VF_MAX``, values are swapped and clamped to ``VF_MIN``/``VF_MAX``.
    When clamping results in an invalid range (min > max), both bounds
    collapse to ``VF_MIN``, ensuring ``VF_MIN``/``VF_MAX`` are hard limits.
    """
    params = InitParams.from_graph(G)

    vf_uniform_min = params.vf_uniform_min
    vf_uniform_max = params.vf_uniform_max
    vf_min_lim = params.vf_min_lim
    vf_max_lim = params.vf_max_lim
    if vf_uniform_min is None:
        vf_uniform_min = vf_min_lim
    if vf_uniform_max is None:
        vf_uniform_max = vf_max_lim
    if vf_uniform_min > vf_uniform_max:
        vf_uniform_min, vf_uniform_max = vf_uniform_max, vf_uniform_min
    params.vf_uniform_min = max(vf_uniform_min, vf_min_lim)
    params.vf_uniform_max = min(vf_uniform_max, vf_max_lim)
    # After clamping to VF_MIN/VF_MAX, ensure min <= max
    if params.vf_uniform_min > params.vf_uniform_max:
        # Collapse to VF_MIN when the requested range is entirely below the limit
        params.vf_uniform_min = params.vf_uniform_max = vf_min_lim

    rng = make_rng(params.seed, -1, G)
    for _, nd in G.nodes(data=True):
        node_attrs = cast(NodeInitAttrMap, nd)

        _init_phase(
            node_attrs,
            rng,
            override=override,
            random_phase=params.init_rand_phase,
            th_min=params.th_min,
            th_max=params.th_max,
        )
        _init_vf(
            node_attrs,
            rng,
            override=override,
            mode=params.vf_mode,
            vf_uniform_min=params.vf_uniform_min,
            vf_uniform_max=params.vf_uniform_max,
            vf_mean=params.vf_mean,
            vf_std=params.vf_std,
            vf_min_lim=params.vf_min_lim,
            vf_max_lim=params.vf_max_lim,
            clamp_to_limits=params.clamp_to_limits,
        )
        _init_si_epi(
            node_attrs,
            rng,
            override=override,
            si_min=params.si_min,
            si_max=params.si_max,
            epi_val=params.epi_val,
        )

    return G
