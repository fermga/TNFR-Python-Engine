# operators.py — TNFR canónica (ASCII-safe)
from __future__ import annotations
from typing import Dict, Any, Optional
import math
import random
import hashlib

from .constants import DEFAULTS, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR, ALIAS_EPI, ALIAS_D2EPI
from .helpers import _get_attr, _set_attr, clamp, clamp01, list_mean, fase_media, push_glifo, invoke_callbacks
from collections import deque

"""
Este módulo implementa:
- Los 13 glifos como operadores locales suaves.
- Un dispatcher `aplicar_glifo` que mapea el nombre del glifo (con apóstrofo tipográfico) a su función.
- RE’MESH de red: `aplicar_remesh_red` y `aplicar_remesh_si_estabilización_global`.

Nota sobre α (alpha) de RE’MESH: se toma por prioridad de
1) G.graph["GLYPH_FACTORS"]["REMESH_alpha"]
2) G.graph["REMESH_ALPHA"]
3) DEFAULTS["REMESH_ALPHA"]
"""


def _node_offset(G, n) -> int:
    """Deterministic node index used for jitter seeds."""
    mapping = G.graph.get("_node_offset_map")
    if mapping is None or len(mapping) != G.number_of_nodes():
        mapping = {node: idx for idx, node in enumerate(sorted(G.nodes(), key=lambda x: str(x)))}
        G.graph["_node_offset_map"] = mapping
    return int(mapping.get(n, 0))

# -------------------------
# Glifos (operadores locales)
# -------------------------

def op_AL(G, n):  # A’L — Emisión
    f = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("AL_boost", 0.05))
    nd = G.nodes[n]
    epi = _get_attr(nd, ALIAS_EPI, 0.0)
    _set_attr(nd, ALIAS_EPI, epi + f)


def op_EN(G, n):  # E’N — Recepción
    mix = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("EN_mix", 0.25))
    nd = G.nodes[n]
    epi = _get_attr(nd, ALIAS_EPI, 0.0)
    if G.degree(n) == 0:
        return  # sin vecinos no hay mezcla
    epi_bar = list_mean(_get_attr(G.nodes[v], ALIAS_EPI, epi) for v in G.neighbors(n))
    _set_attr(nd, ALIAS_EPI, (1 - mix) * epi + mix * epi_bar)


def op_IL(G, n):  # I’L — Coherencia (reduce ΔNFR)
    factor = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("IL_dnfr_factor", 0.7))
    nd = G.nodes[n]
    dnfr = _get_attr(nd, ALIAS_DNFR, 0.0)
    _set_attr(nd, ALIAS_DNFR, factor * dnfr)

def op_OZ(G, n):  # O’Z — Disonancia (aumenta ΔNFR o añade ruido)
    factor = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("OZ_dnfr_factor", 1.3))
    nd = G.nodes[n]
    dnfr = _get_attr(nd, ALIAS_DNFR, 0.0)
    if bool(G.graph.get("OZ_NOISE_MODE", False)):
        base_seed = int(G.graph.get("RANDOM_SEED", 0))
        step_idx = len(G.graph.get("history", {}).get("C_steps", []))
        rnd = random.Random(base_seed + step_idx*1000003 + _node_offset(G, n) % 1009)
        sigma = float(G.graph.get("OZ_SIGMA", 0.1))
        noise = sigma * (2.0 * rnd.random() - 1.0)
        _set_attr(nd, ALIAS_DNFR, dnfr + noise)
    else:
        _set_attr(nd, ALIAS_DNFR, factor * dnfr if abs(dnfr) > 1e-9 else 0.1)

def op_UM(G, n):  # U’M — Acoplamiento (empuja fase a la media local)
    k = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("UM_theta_push", 0.25))
    nd = G.nodes[n]
    th = _get_attr(nd, ALIAS_THETA, 0.0)
    thL = fase_media(G, n)
    d = ((thL - th + math.pi) % (2 * math.pi) - math.pi)
    _set_attr(nd, ALIAS_THETA, th + k * d)


def op_RA(G, n):  # R’A — Resonancia (difusión EPI)
    diff = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("RA_epi_diff", 0.15))
    nd = G.nodes[n]
    epi = _get_attr(nd, ALIAS_EPI, 0.0)
    if G.degree(n) == 0:
        return
    epi_bar = list_mean(_get_attr(G.nodes[v], ALIAS_EPI, epi) for v in G.neighbors(n))
    _set_attr(nd, ALIAS_EPI, epi + diff * (epi_bar - epi))


def op_SHA(G, n):  # SH’A — Silencio (baja νf)
    factor = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("SHA_vf_factor", 0.85))
    nd = G.nodes[n]
    vf = _get_attr(nd, ALIAS_VF, 0.0)
    _set_attr(nd, ALIAS_VF, factor * vf)


def op_VAL(G, n):  # VA’L — Expansión (escala EPI)
    s = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("VAL_scale", 1.15))
    nd = G.nodes[n]
    epi = _get_attr(nd, ALIAS_EPI, 0.0)
    _set_attr(nd, ALIAS_EPI, s * epi)


def op_NUL(G, n):  # NU’L — Contracción (escala EPI)
    s = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("NUL_scale", 0.85))
    nd = G.nodes[n]
    epi = _get_attr(nd, ALIAS_EPI, 0.0)
    _set_attr(nd, ALIAS_EPI, s * epi)


def op_THOL(G, n):  # T’HOL — Autoorganización (inyecta aceleración)
    a = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("THOL_accel", 0.10))
    nd = G.nodes[n]
    d2 = _get_attr(nd, ALIAS_D2EPI, 0.0)
    _set_attr(nd, ALIAS_DNFR, _get_attr(nd, ALIAS_DNFR, 0.0) + a * d2)


def op_ZHIR(G, n):  # Z’HIR — Mutación (desplaza fase)
    shift = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("ZHIR_theta_shift", 1.57079632679))
    nd = G.nodes[n]
    th = _get_attr(nd, ALIAS_THETA, 0.0)
    _set_attr(nd, ALIAS_THETA, th + shift)

def op_NAV(G, n):  # NA’V — Transición (jitter suave de ΔNFR)
    nd = G.nodes[n]
    dnfr = _get_attr(nd, ALIAS_DNFR, 0.0)
    j = float(G.graph.get("GLYPH_FACTORS", DEFAULTS["GLYPH_FACTORS"]).get("NAV_jitter", 0.05))
    if bool(G.graph.get("NAV_RANDOM", True)):
        # jitter uniforme en [-j, j] con semilla reproducible
        base_seed = int(G.graph.get("RANDOM_SEED", 0))
        # opcional: pequeño offset para evitar misma secuencia en todos los nodos/pasos
        step_idx = len(G.graph.get("history", {}).get("C_steps", []))
        rnd = random.Random(base_seed + step_idx*1000003 + _node_offset(G, n) % 1009)
        jitter = j * (2.0 * rnd.random() - 1.0)
    else:
        # comportamiento determinista (compatibilidad previa)
        jitter = j * (1 if dnfr >= 0 else -1)
    _set_attr(nd, ALIAS_DNFR, dnfr + jitter)

def op_REMESH(G, n):  # RE’MESH — se realiza a escala de red (no-op local con aviso)
    # Loguea solo 1 vez por paso para no spamear
    step_idx = len(G.graph.get("history", {}).get("C_steps", []))
    last_warn = G.graph.get("_remesh_warn_step", None)
    if last_warn != step_idx:
        msg = "RE’MESH es a escala de red. Usa aplicar_remesh_si_estabilizacion_global(G) o aplicar_remesh_red(G)."
        G.graph.setdefault("history", {}).setdefault("events", []).append(("warn", {"step": step_idx, "node": n, "msg": msg}))
        G.graph["_remesh_warn_step"] = step_idx
    # no cambia estado local
    return

# -------------------------
# Dispatcher
# -------------------------

_NAME_TO_OP = {
    "A’L": op_AL, "E’N": op_EN, "I’L": op_IL, "O’Z": op_OZ, "U’M": op_UM,
    "R’A": op_RA, "SH’A": op_SHA, "VA’L": op_VAL, "NU’L": op_NUL,
    "T’HOL": op_THOL, "Z’HIR": op_ZHIR, "NA’V": op_NAV, "RE’MESH": op_REMESH,
}


def aplicar_glifo(G, n, glifo: str, *, window: Optional[int] = None) -> None:
    """Aplica un glifo TNFR al nodo `n` con histéresis `window`.

    Los 13 glifos implementan reorganizadores canónicos:
      A’L (emisión), E’N (recepción), I’L (coherencia), O’Z (disonancia),
      U’M (acoplamiento), R’A (resonancia), SH’A (silencio), VA’L (expansión),
      NU’L (contracción), T’HOL (autoorganización), Z’HIR (mutación),
      NA’V (transición), RE’MESH (recursividad).

    Relación con la gramática: la selección previa debe pasar por
    `enforce_canonical_grammar` (grammar.py) para respetar compatibilidades,
    precondición O’Z→Z’HIR y cierres T’HOL[...].
    """
    glifo = str(glifo)
    op = _NAME_TO_OP.get(glifo)
    if not op:
        return
    if window is None:
        window = int(G.graph.get("GLYPH_HYSTERESIS_WINDOW", DEFAULTS["GLYPH_HYSTERESIS_WINDOW"]))
    push_glifo(G.nodes[n], glifo, window)
    op(G, n)


# -------------------------
# RE’MESH de red (usa _epi_hist capturado en dynamics.step)
# -------------------------

def _remesh_alpha_info(G):
    """Devuelve `(alpha, source)` con precedencia explícita."""
    if bool(G.graph.get("REMESH_ALPHA_HARD", DEFAULTS.get("REMESH_ALPHA_HARD", False))):
        val = float(G.graph.get("REMESH_ALPHA", DEFAULTS["REMESH_ALPHA"]))
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return float(DEFAULTS["REMESH_ALPHA"]), "DEFAULTS.REMESH_ALPHA"


def aplicar_remesh_red(G) -> None:
    """RE’MESH a escala de red usando _epi_hist con memoria multi-escala."""
    tau_g = int(G.graph.get("REMESH_TAU_GLOBAL", G.graph.get("REMESH_TAU", DEFAULTS["REMESH_TAU_GLOBAL"])))
    tau_l = int(G.graph.get("REMESH_TAU_LOCAL", G.graph.get("REMESH_TAU", DEFAULTS["REMESH_TAU_LOCAL"])))
    tau_req = max(tau_g, tau_l)
    alpha, alpha_src = _remesh_alpha_info(G)
    G.graph["_REMESH_ALPHA_SRC"] = alpha_src
    hist = G.graph.get("_epi_hist", deque())
    if len(hist) < tau_req + 1:
        return

    past_g = hist[-(tau_g + 1)]
    past_l = hist[-(tau_l + 1)]

    # --- Topología + snapshot EPI (ANTES) ---
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degs = sorted(d for _, d in G.degree())
        topo_str = f"n={n_nodes};m={n_edges};deg=" + ",".join(map(str, degs))
        topo_hash = hashlib.sha1(topo_str.encode()).hexdigest()[:12]
    except Exception:
        topo_hash = None

    def _epi_items():
        for node in G.nodes():
            yield node, _get_attr(G.nodes[node], ALIAS_EPI, 0.0)

    epi_mean_before = list_mean(v for _, v in _epi_items())
    epi_checksum_before = hashlib.sha1(
        str(sorted((str(k), round(v, 6)) for k, v in _epi_items())).encode()
    ).hexdigest()[:12]

    # --- Mezcla (1-α)·now + α·old ---
    for n in G.nodes():
        nd = G.nodes[n]
        epi_now = _get_attr(nd, ALIAS_EPI, 0.0)
        epi_old_l = float(past_l.get(n, epi_now))
        epi_old_g = float(past_g.get(n, epi_now))
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g
        _set_attr(nd, ALIAS_EPI, mixed)

    # --- Snapshot EPI (DESPUÉS) ---
    epi_mean_after = list_mean(_get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes())
    epi_checksum_after = hashlib.sha1(
        str(sorted((str(n), round(_get_attr(G.nodes[n], ALIAS_EPI, 0.0), 6)) for n in G.nodes())).encode()
    ).hexdigest()[:12]

    # --- Metadatos y logging de evento ---
    step_idx = len(G.graph.get("history", {}).get("C_steps", []))
    meta = {
        "alpha": alpha,
        "alpha_source": alpha_src,
        "tau_global": tau_g,
        "tau_local": tau_l,
        "step": step_idx,
        # firmas
        "topo_hash": topo_hash,
        "epi_mean_before": float(epi_mean_before),
        "epi_mean_after": float(epi_mean_after),
        "epi_checksum_before": epi_checksum_before,
        "epi_checksum_after": epi_checksum_after,
    }

    # Snapshot opcional de métricas recientes
    h = G.graph.get("history", {})
    if h:
        if h.get("stable_frac"): meta["stable_frac_last"] = h["stable_frac"][-1]
        if h.get("phase_sync"):  meta["phase_sync_last"]  = h["phase_sync"][-1]
        if h.get("glyph_load_disr"): meta["glyph_disr_last"] = h["glyph_load_disr"][-1]

    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", DEFAULTS["REMESH_LOG_EVENTS"]):
        ev = G.graph.setdefault("history", {}).setdefault("remesh_events", [])
        ev.append(dict(meta))

    # Callbacks Γ(R)
    invoke_callbacks(G, "on_remesh", dict(meta))

def aplicar_remesh_si_estabilizacion_global(G, pasos_estables_consecutivos: Optional[int] = None) -> None:
    # Ventanas y umbrales
    w_estab = (
        pasos_estables_consecutivos
        if pasos_estables_consecutivos is not None
        else int(G.graph.get("REMESH_STABILITY_WINDOW", DEFAULTS["REMESH_STABILITY_WINDOW"]))
    )
    frac_req = float(G.graph.get("FRACTION_STABLE_REMESH", DEFAULTS["FRACTION_STABLE_REMESH"]))
    req_extra = bool(G.graph.get("REMESH_REQUIRE_STABILITY", DEFAULTS["REMESH_REQUIRE_STABILITY"]))
    min_sync = float(G.graph.get("REMESH_MIN_PHASE_SYNC", DEFAULTS["REMESH_MIN_PHASE_SYNC"]))
    max_disr = float(G.graph.get("REMESH_MAX_GLYPH_DISR", DEFAULTS["REMESH_MAX_GLYPH_DISR"]))
    min_sigma = float(G.graph.get("REMESH_MIN_SIGMA_MAG", DEFAULTS["REMESH_MIN_SIGMA_MAG"]))
    min_R = float(G.graph.get("REMESH_MIN_KURAMOTO_R", DEFAULTS["REMESH_MIN_KURAMOTO_R"]))
    min_sihi = float(G.graph.get("REMESH_MIN_SI_HI_FRAC", DEFAULTS["REMESH_MIN_SI_HI_FRAC"]))

    hist = G.graph.setdefault("history", {"stable_frac": []})
    sf = hist.get("stable_frac", [])
    if len(sf) < w_estab:
        return
    # 1) Estabilidad por fracción de nodos estables
    win_sf = sf[-w_estab:]
    cond_sf = all(v >= frac_req for v in win_sf)
    if not cond_sf:
        return
    # 2) Gating adicional (si está activado)
    if req_extra:
        # sincronía de fase (mayor mejor)
        ps_ok = True
        if "phase_sync" in hist and len(hist["phase_sync"]) >= w_estab:
            win_ps = hist["phase_sync"][-w_estab:]
            ps_ok = (sum(win_ps)/len(win_ps)) >= min_sync
        # carga glífica disruptiva (menor mejor)
        disr_ok = True
        if "glyph_load_disr" in hist and len(hist["glyph_load_disr"]) >= w_estab:
            win_disr = hist["glyph_load_disr"][-w_estab:]
            disr_ok = (sum(win_disr)/len(win_disr)) <= max_disr
        # magnitud de sigma (mayor mejor)
        sig_ok = True
        if "sense_sigma_mag" in hist and len(hist["sense_sigma_mag"]) >= w_estab:
            win_sig = hist["sense_sigma_mag"][-w_estab:]
            sig_ok = (sum(win_sig)/len(win_sig)) >= min_sigma
        # orden de Kuramoto R (mayor mejor)
        R_ok = True
        if "kuramoto_R" in hist and len(hist["kuramoto_R"]) >= w_estab:
            win_R = hist["kuramoto_R"][-w_estab:]
            R_ok = (sum(win_R)/len(win_R)) >= min_R
        # fracción de nodos con Si alto (mayor mejor)
        sihi_ok = True
        if "Si_hi_frac" in hist and len(hist["Si_hi_frac"]) >= w_estab:
            win_sihi = hist["Si_hi_frac"][-w_estab:]
            sihi_ok = (sum(win_sihi)/len(win_sihi)) >= min_sihi
        if not (ps_ok and disr_ok and sig_ok and R_ok and sihi_ok):
            return
    # 3) Cooldown
    last = G.graph.get("_last_remesh_step", -10**9)
    step_idx = len(sf)
    cooldown = int(G.graph.get("REMESH_COOLDOWN_VENTANA", DEFAULTS["REMESH_COOLDOWN_VENTANA"]))
    if step_idx - last < cooldown:
        return
    t_now = float(G.graph.get("_t", 0.0))
    last_ts = float(G.graph.get("_last_remesh_ts", -1e12))
    cooldown_ts = float(G.graph.get("REMESH_COOLDOWN_TS", DEFAULTS.get("REMESH_COOLDOWN_TS", 0.0)))
    if cooldown_ts > 0 and (t_now - last_ts) < cooldown_ts:
        return
    # 4) Aplicar y registrar
    aplicar_remesh_red(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now
