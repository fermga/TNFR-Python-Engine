"""initialization.py — TNFR canónica

Lógica compartida para inicializar atributos nodales básicos (EPI, θ, νf y Si).
"""
from __future__ import annotations
import math
import random
import networkx as nx

from .constants import DEFAULTS


def init_node_attrs(G: nx.Graph, *, override: bool = True) -> nx.Graph:
    """Inicializa EPI, θ, νf y Si en los nodos de ``G``.

    Los parámetros pueden personalizarse mediante entradas en ``G.graph``:
    ``RANDOM_SEED``, ``INIT_RANDOM_PHASE``, ``INIT_THETA_MIN/MAX``,
    ``INIT_VF_MODE``, ``VF_MIN``, ``VF_MAX``, ``INIT_VF_MIN/MAX``,
    ``INIT_VF_MEAN``, ``INIT_VF_STD`` y ``INIT_VF_CLAMP_TO_LIMITS``.
    Se añaden rangos para ``Si`` vía ``INIT_SI_MIN`` y ``INIT_SI_MAX``, y para
    ``EPI`` mediante ``INIT_EPI_VALUE``.
    """
    seed = int(G.graph.get("RANDOM_SEED", 0))
    init_rand_phase = bool(G.graph.get("INIT_RANDOM_PHASE", DEFAULTS.get("INIT_RANDOM_PHASE", True)))

    th_min = float(G.graph.get("INIT_THETA_MIN", DEFAULTS.get("INIT_THETA_MIN", -math.pi)))
    th_max = float(G.graph.get("INIT_THETA_MAX", DEFAULTS.get("INIT_THETA_MAX", math.pi)))

    vf_mode = str(G.graph.get("INIT_VF_MODE", DEFAULTS.get("INIT_VF_MODE", "uniform"))).lower()
    vf_min_lim = float(G.graph.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max_lim = float(G.graph.get("VF_MAX", DEFAULTS["VF_MAX"]))

    vf_uniform_min = G.graph.get("INIT_VF_MIN", DEFAULTS.get("INIT_VF_MIN", None))
    vf_uniform_max = G.graph.get("INIT_VF_MAX", DEFAULTS.get("INIT_VF_MAX", None))
    if vf_uniform_min is None:
        vf_uniform_min = vf_min_lim
    if vf_uniform_max is None:
        vf_uniform_max = vf_max_lim

    vf_mean = float(G.graph.get("INIT_VF_MEAN", DEFAULTS.get("INIT_VF_MEAN", 0.5)))
    vf_std = float(G.graph.get("INIT_VF_STD", DEFAULTS.get("INIT_VF_STD", 0.15)))
    clamp_to_limits = bool(G.graph.get("INIT_VF_CLAMP_TO_LIMITS", DEFAULTS.get("INIT_VF_CLAMP_TO_LIMITS", True)))

    si_min = float(G.graph.get("INIT_SI_MIN", 0.4))
    si_max = float(G.graph.get("INIT_SI_MAX", 0.7))
    epi_val = float(G.graph.get("INIT_EPI_VALUE", 0.0))

    rand = random.Random(seed)

    for idx, n in enumerate(G.nodes()):
        nd = G.nodes[n]

        if override or "EPI" not in nd:
            nd["EPI"] = epi_val

        rand.seed(seed + idx)

        if init_rand_phase:
            if override or "θ" not in nd:
                nd["θ"] = rand.uniform(th_min, th_max)
        else:
            if override:
                nd["θ"] = 0.0
            else:
                nd.setdefault("θ", 0.0)

        if vf_mode == "uniform":
            vf = rand.uniform(float(vf_uniform_min), float(vf_uniform_max))
        elif vf_mode == "normal":
            for _ in range(16):
                cand = rand.normalvariate(vf_mean, vf_std)
                if vf_min_lim <= cand <= vf_max_lim:
                    vf = cand
                    break
            else:
                vf = min(max(rand.normalvariate(vf_mean, vf_std), vf_min_lim), vf_max_lim)
        else:
            vf = float(nd.get("νf", 0.5))
        if clamp_to_limits:
            vf = min(max(vf, vf_min_lim), vf_max_lim)
        if override or "νf" not in nd:
            nd["νf"] = float(vf)

        si = rand.uniform(si_min, si_max)
        if override or "Si" not in nd:
            nd["Si"] = float(si)

    return G
