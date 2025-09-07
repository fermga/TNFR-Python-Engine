"""Diagnostic metrics."""

from __future__ import annotations

from statistics import fmean, StatisticsError
from operator import ge, le

from ..constants import (
    ALIAS_EPI,
    ALIAS_VF,
    ALIAS_DNFR,
    ALIAS_SI,
    DIAGNOSIS,
    COHERENCE,
    VF_KEY,
)
from ..callback_utils import register_callback
from ..glyph_history import ensure_history, append_metric
from ..alias import get_attr
from ..helpers import clamp01
from ..metrics_utils import compute_dnfr_accel_max, min_max_range
from .coherence import local_phase_sync, local_phase_sync_weighted, _similarity_abs


def _dnfr_norm(nd, dnfr_max):
    val = abs(get_attr(nd, ALIAS_DNFR, 0.0))
    if dnfr_max <= 0:
        return 0.0
    x = val / dnfr_max
    return 1.0 if x > 1 else x


def _symmetry_index(
    G, n, epi_min: float | None = None, epi_max: float | None = None
):
    """Compute the symmetry index for node ``n`` based on EPI values."""
    nd = G.nodes[n]
    epi_i = get_attr(nd, ALIAS_EPI, 0.0)
    vec = G.neighbors(n)
    try:
        epi_bar = fmean(get_attr(G.nodes[v], ALIAS_EPI, epi_i) for v in vec)
    except StatisticsError:
        return 1.0
    if epi_min is None or epi_max is None:
        epi_iter = (get_attr(G.nodes[v], ALIAS_EPI, 0.0) for v in G.nodes())
        epi_min, epi_max = min_max_range(epi_iter, default=(0.0, 1.0))
    return _similarity_abs(epi_i, epi_bar, epi_min, epi_max)


def _state_from_thresholds(Rloc, dnfr_n, cfg):
    stb = cfg.get("stable", {"Rloc_hi": 0.8, "dnfr_lo": 0.2, "persist": 3})
    dsr = cfg.get("dissonance", {"Rloc_lo": 0.4, "dnfr_hi": 0.5, "persist": 3})

    stable_checks = {
        "Rloc": (Rloc, float(stb["Rloc_hi"]), ge),
        "dnfr": (dnfr_n, float(stb["dnfr_lo"]), le),
    }
    if all(comp(val, thr) for val, thr, comp in stable_checks.values()):
        return "estable"

    dissonant_checks = {
        "Rloc": (Rloc, float(dsr["Rloc_lo"]), le),
        "dnfr": (dnfr_n, float(dsr["dnfr_hi"]), ge),
    }
    if all(comp(val, thr) for val, thr, comp in dissonant_checks.values()):
        return "disonante"

    return "transicion"


def _recommendation(state, cfg):
    adv = cfg.get("advice", {})
    key = {
        "estable": "stable",
        "transicion": "transition",
        "disonante": "dissonant",
    }[state]
    return list(adv.get(key, []))


def _get_last_weights(G, hist):
    """Return last Wi and Wm matrices from history."""
    CfgW = G.graph.get("COHERENCE", COHERENCE)
    Wkey = CfgW.get("Wi_history_key", "W_i")
    Wm_key = CfgW.get("history_key", "W_sparse")
    Wi_series = hist.get(Wkey, [])
    Wm_series = hist.get(Wm_key, [])
    Wi_last = Wi_series[-1] if Wi_series else None
    Wm_last = Wm_series[-1] if Wm_series else None
    return Wi_last, Wm_last


def _node_diagnostics(
    G,
    n,
    i,
    nodes,
    node_to_index,
    Wi_last,
    Wm_last,
    epi_min,
    epi_max,
    dnfr_max,
    dcfg,
):
    nd = G.nodes[n]
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.0))
    EPI = get_attr(nd, ALIAS_EPI, 0.0)
    vf = get_attr(nd, ALIAS_VF, 0.0)
    dnfr_n = _dnfr_norm(nd, dnfr_max)

    if Wm_last is not None:
        if Wm_last and isinstance(Wm_last[0], list):
            row = Wm_last[i]
        else:
            row = Wm_last
        Rloc = local_phase_sync_weighted(
            G, n, nodes_order=nodes, W_row=row, node_to_index=node_to_index
        )
    else:
        Rloc = local_phase_sync(G, n)

    symm = (
        _symmetry_index(G, n, epi_min=epi_min, epi_max=epi_max)
        if dcfg.get("compute_symmetry", True)
        else None
    )
    state = _state_from_thresholds(Rloc, dnfr_n, dcfg)

    alerts = []
    if state == "disonante" and dnfr_n >= float(
        dcfg.get("dissonance", {}).get("dnfr_hi", 0.5)
    ):
        alerts.append("high structural tension")

    advice = _recommendation(state, dcfg)

    return {
        "node": n,
        "Si": Si,
        "EPI": EPI,
        VF_KEY: vf,
        "dnfr_norm": dnfr_n,
        "W_i": (Wi_last[i] if (Wi_last and i < len(Wi_last)) else None),
        "R_local": Rloc,
        "symmetry": symm,
        "state": state,
        "advice": advice,
        "alerts": alerts,
    }


def _diagnosis_step(G, ctx=None):
    dcfg = G.graph.get("DIAGNOSIS", DIAGNOSIS)
    if not dcfg.get("enabled", True):
        return

    hist = ensure_history(G)
    key = dcfg.get("history_key", "nodal_diag")

    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    epi_iter = (get_attr(nd, ALIAS_EPI, 0.0) for _, nd in G.nodes(data=True))
    epi_min, epi_max = min_max_range(epi_iter, default=(0.0, 1.0))

    Wi_last, Wm_last = _get_last_weights(G, hist)

    nodes = list(G.nodes())
    node_to_index = {v: i for i, v in enumerate(nodes)}
    diag = {}
    for i, n in enumerate(nodes):
        diag[n] = _node_diagnostics(
            G,
            n,
            i,
            nodes,
            node_to_index,
            Wi_last,
            Wm_last,
            epi_min,
            epi_max,
            dnfr_max,
            dcfg,
        )

    append_metric(hist, key, diag)


def dissonance_events(G, ctx=None):
    """Emit per-node structural dissonance start/end events.

    Events are recorded as ``"dissonance_start"`` and ``"dissonance_end"``.
    """
    hist = ensure_history(G)
    # eventos de disonancia se registran en ``history['events']``
    norms = G.graph.get("_sel_norms", {})
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    step_idx = len(hist.get("C_steps", []))
    nodes = list(G.nodes())
    node_to_index = {v: i for i, v in enumerate(nodes)}
    for n in nodes:
        nd = G.nodes[n]
        dn = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
        Rloc = local_phase_sync(G, n)
        st = bool(nd.get("_disr_state", False))
        if (not st) and dn >= 0.5 and Rloc <= 0.4:
            nd["_disr_state"] = True
            append_metric(
                hist, "events", ("dissonance_start", {"node": n, "step": step_idx})
            )
        elif st and dn <= 0.2 and Rloc >= 0.7:
            nd["_disr_state"] = False
            append_metric(
                hist, "events", ("dissonance_end", {"node": n, "step": step_idx})
            )


def register_diagnosis_callbacks(G) -> None:
    register_callback(
        G, event="after_step", func=_diagnosis_step, name="diagnosis_step"
    )
    register_callback(
        G, event="after_step", func=dissonance_events, name="dissonance_events"
    )
