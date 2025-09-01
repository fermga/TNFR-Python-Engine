"""Métricas de diagnóstico."""
from __future__ import annotations

from statistics import fmean
from typing import Dict

from ..constants import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_SI, DIAGNOSIS, COHERENCE
from ..helpers import (
    register_callback,
    ensure_history,
    get_attr,
    clamp01,
    compute_dnfr_accel_max,
)
from .coherence import local_phase_sync_weighted, _similarity_abs


def _dnfr_norm(nd, dnfr_max):
    val = abs(float(get_attr(nd, ALIAS_DNFR, 0.0)))
    if dnfr_max <= 0:
        return 0.0
    x = val / dnfr_max
    return 1.0 if x > 1 else x


def _symmetry_index(G, n, k=3, epi_min=None, epi_max=None):
    nd = G.nodes[n]
    epi_i = float(get_attr(nd, ALIAS_EPI, 0.0))
    vec = list(G.neighbors(n))
    if not vec:
        return 1.0
    epi_bar = fmean(float(get_attr(G.nodes[v], ALIAS_EPI, epi_i)) for v in vec)
    if epi_min is None or epi_max is None:
        epis = [float(get_attr(G.nodes[v], ALIAS_EPI, 0.0)) for v in G.nodes()]
        epi_min, epi_max = min(epis), max(epis)
    return _similarity_abs(epi_i, epi_bar, epi_min, epi_max)


def _state_from_thresholds(Rloc, dnfr_n, cfg):
    stb = cfg.get("stable", {"Rloc_hi": 0.8, "dnfr_lo": 0.2, "persist": 3})
    dsr = cfg.get("dissonance", {"Rloc_lo": 0.4, "dnfr_hi": 0.5, "persist": 3})
    if (Rloc >= float(stb["Rloc_hi"])) and (dnfr_n <= float(stb["dnfr_lo"])):
        return "estable"
    if (Rloc <= float(dsr["Rloc_lo"])) and (dnfr_n >= float(dsr["dnfr_hi"])):
        return "disonante"
    return "transicion"


def _recommendation(state, cfg):
    adv = cfg.get("advice", {})
    key = {"estable": "stable", "transicion": "transition", "disonante": "dissonant"}[state]
    return list(adv.get(key, []))


def _diagnosis_step(G, ctx=None):
    dcfg = G.graph.get("DIAGNOSIS", DIAGNOSIS)
    if not dcfg.get("enabled", True):
        return

    hist = ensure_history(G)
    key = dcfg.get("history_key", "nodal_diag")

    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    epi_vals = [float(get_attr(G.nodes[v], ALIAS_EPI, 0.0)) for v in G.nodes()]
    epi_min, epi_max = (min(epi_vals) if epi_vals else 0.0), (max(epi_vals) if epi_vals else 1.0)

    CfgW = G.graph.get("COHERENCE", COHERENCE)
    Wkey = CfgW.get("Wi_history_key", "W_i")
    Wm_key = CfgW.get("history_key", "W_sparse")
    Wi_series = hist.get(Wkey, [])
    Wi_last = Wi_series[-1] if Wi_series else None
    Wm_series = hist.get(Wm_key, [])
    Wm_last = Wm_series[-1] if Wm_series else None

    nodes = list(G.nodes())
    node_to_index = {v: i for i, v in enumerate(nodes)}
    diag = {}
    for i, n in enumerate(nodes):
        nd = G.nodes[n]
        Si = clamp01(get_attr(nd, ALIAS_SI, 0.0))
        EPI = float(get_attr(nd, ALIAS_EPI, 0.0))
        vf = get_attr(nd, ALIAS_VF, 0.0)
        dnfr_n = _dnfr_norm(nd, dnfr_max)

        Rloc = 0.0
        if Wm_last is not None:
            if Wm_last and isinstance(Wm_last[0], list):
                row = Wm_last[i]
            else:
                row = Wm_last
            Rloc = local_phase_sync_weighted(
                G, n, nodes_order=nodes, W_row=row, node_to_index=node_to_index
            )
        else:
            Rloc = local_phase_sync_weighted(
                G, n, nodes_order=nodes, node_to_index=node_to_index
            )

        symm = _symmetry_index(G, n, epi_min=epi_min, epi_max=epi_max) if dcfg.get("compute_symmetry", True) else None
        state = _state_from_thresholds(Rloc, dnfr_n, dcfg)

        alerts = []
        if state == "disonante" and dnfr_n >= float(dcfg.get("dissonance", {}).get("dnfr_hi", 0.5)):
            alerts.append("tensión estructural alta")

        advice = _recommendation(state, dcfg)

        rec = {
            "node": n,
            "Si": Si,
            "EPI": EPI,
            "νf": vf,
            "dnfr_norm": dnfr_n,
            "W_i": (Wi_last[i] if (Wi_last and i < len(Wi_last)) else None),
            "R_local": Rloc,
            "symmetry": symm,
            "state": state,
            "advice": advice,
            "alerts": alerts,
        }
        diag[n] = rec

    hist.setdefault(key, []).append(diag)


def dissonance_events(G, ctx=None):
    """Emite eventos de inicio/fin de disonancia estructural por nodo.

    Los eventos se registran como ``"dissonance_start"`` y
    ``"dissonance_end"``.
    """
    hist = ensure_history(G)
    evs = hist.setdefault("events", [])
    norms = G.graph.get("_sel_norms", {})
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    step_idx = len(hist.get("C_steps", []))
    nodes = list(G.nodes())
    node_to_index = {v: i for i, v in enumerate(nodes)}
    for n in nodes:
        nd = G.nodes[n]
        dn = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
        Rloc = local_phase_sync_weighted(
            G, n, nodes_order=nodes, node_to_index=node_to_index
        )
        st = bool(nd.get("_disr_state", False))
        if (not st) and dn >= 0.5 and Rloc <= 0.4:
            nd["_disr_state"] = True
            evs.append(("dissonance_start", {"node": n, "step": step_idx}))
        elif st and dn <= 0.2 and Rloc >= 0.7:
            nd["_disr_state"] = False
            evs.append(("dissonance_end", {"node": n, "step": step_idx}))


def register_diagnosis_callbacks(G) -> None:
    register_callback(G, event="after_step", func=_diagnosis_step, name="diagnosis_step")
    register_callback(G, event="after_step", func=dissonance_events, name="dissonance_events")
