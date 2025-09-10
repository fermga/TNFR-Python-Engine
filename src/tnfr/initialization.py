"""Node initialization."""

from __future__ import annotations
import random
from typing import TYPE_CHECKING

from .constants import DEFAULTS, INIT_DEFAULTS, VF_KEY, THETA_KEY
from .helpers.numeric import clamp
from .rng import make_rng

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

__all__ = ["init_node_attrs"]


def _init_phase(
    nd: dict,
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
    nd: dict,
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
        vf = rng.uniform(float(vf_uniform_min), float(vf_uniform_max))
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
        nd[VF_KEY] = float(vf)


def _init_si_epi(
    nd: dict,
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
        nd["Si"] = float(si)


def init_node_attrs(G: "nx.Graph", *, override: bool = True) -> "nx.Graph":
    """Initialise EPI, θ, νf and Si on the nodes of ``G``.

    Parameters can be customised via ``G.graph`` entries:
    ``RANDOM_SEED``, ``INIT_RANDOM_PHASE``, ``INIT_THETA_MIN/MAX``,
    ``INIT_VF_MODE``, ``VF_MIN``, ``VF_MAX``, ``INIT_VF_MIN/MAX``,
    ``INIT_VF_MEAN``, ``INIT_VF_STD`` and ``INIT_VF_CLAMP_TO_LIMITS``.
    Ranges for ``Si`` are added via ``INIT_SI_MIN`` and ``INIT_SI_MAX``, and
    for ``EPI`` via ``INIT_EPI_VALUE``. If ``INIT_VF_MIN`` is greater than
    ``INIT_VF_MAX``, values are swapped and clamped to ``VF_MIN``/``VF_MAX``.
    """
    seed = int(G.graph.get("RANDOM_SEED", 0))
    init_rand_phase = bool(
        G.graph.get("INIT_RANDOM_PHASE", INIT_DEFAULTS["INIT_RANDOM_PHASE"])
    )

    th_min = float(
        G.graph.get("INIT_THETA_MIN", INIT_DEFAULTS["INIT_THETA_MIN"])
    )
    th_max = float(
        G.graph.get("INIT_THETA_MAX", INIT_DEFAULTS["INIT_THETA_MAX"])
    )

    vf_mode = str(
        G.graph.get("INIT_VF_MODE", INIT_DEFAULTS["INIT_VF_MODE"])
    ).lower()
    vf_min_lim = float(G.graph.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max_lim = float(G.graph.get("VF_MAX", DEFAULTS["VF_MAX"]))

    vf_uniform_min = G.graph.get(
        "INIT_VF_MIN", INIT_DEFAULTS.get("INIT_VF_MIN")
    )
    vf_uniform_max = G.graph.get(
        "INIT_VF_MAX", INIT_DEFAULTS.get("INIT_VF_MAX")
    )
    if vf_uniform_min is None:
        vf_uniform_min = vf_min_lim
    if vf_uniform_max is None:
        vf_uniform_max = vf_max_lim
    if vf_uniform_min > vf_uniform_max:
        vf_uniform_min, vf_uniform_max = vf_uniform_max, vf_uniform_min
    vf_uniform_min = max(vf_uniform_min, vf_min_lim)
    vf_uniform_max = min(vf_uniform_max, vf_max_lim)

    vf_mean = float(G.graph.get("INIT_VF_MEAN", INIT_DEFAULTS["INIT_VF_MEAN"]))
    vf_std = float(G.graph.get("INIT_VF_STD", INIT_DEFAULTS["INIT_VF_STD"]))
    clamp_to_limits = bool(
        G.graph.get(
            "INIT_VF_CLAMP_TO_LIMITS", INIT_DEFAULTS["INIT_VF_CLAMP_TO_LIMITS"]
        )
    )

    si_min = float(G.graph.get("INIT_SI_MIN", 0.4))
    si_max = float(G.graph.get("INIT_SI_MAX", 0.7))
    epi_val = float(G.graph.get("INIT_EPI_VALUE", 0.0))

    rng_template = make_rng(seed, -1)
    rng = random.Random()
    rng.setstate(rng_template.getstate())
    for _, nd in G.nodes(data=True):

        _init_phase(
            nd,
            rng,
            override=override,
            random_phase=init_rand_phase,
            th_min=th_min,
            th_max=th_max,
        )
        _init_vf(
            nd,
            rng,
            override=override,
            mode=vf_mode,
            vf_uniform_min=float(vf_uniform_min),
            vf_uniform_max=float(vf_uniform_max),
            vf_mean=vf_mean,
            vf_std=vf_std,
            vf_min_lim=vf_min_lim,
            vf_max_lim=vf_max_lim,
            clamp_to_limits=clamp_to_limits,
        )
        _init_si_epi(
            nd,
            rng,
            override=override,
            si_min=si_min,
            si_max=si_max,
            epi_val=epi_val,
        )

    return G
