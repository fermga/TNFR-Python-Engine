"""
ontosim.py — TNFR canónica

Módulo de orquestación mínima que encadena:
ΔNFR (campo) → Si → glifos → ecuación nodal → clamps → U’M → observadores → RE’MESH
"""
from __future__ import annotations
import networkx as nx
import math
import random

from constants import DEFAULTS, attach_defaults
from dynamics import step as _step, run as _run
from dynamics import default_compute_delta_nfr

# API de alto nivel

def preparar_red(G: nx.Graph, *, override_defaults: bool = False, **overrides) -> nx.Graph:
    attach_defaults(G, override=override_defaults)
    if overrides:
        from constants import merge_overrides
        merge_overrides(G, **overrides)
    # Inicializaciones blandas
    G.graph.setdefault("history", {
        "C_steps": [],
        "stable_frac": [],
        "phase_sync": [],
        "kuramoto_R": [], 
        "sense_sigma_x": [],
        "sense_sigma_y": [],
        "sense_sigma_mag": [],
        "sense_sigma_angle": [],
        "iota": [],
        "glyph_load_estab": [],
        "glyph_load_disr": [],
        "Si_mean": [],
        "Si_hi_frac": [],
        "Si_lo_frac": [],
        "W_bar": [],
        "phase_kG": [],
        "phase_kL": [], 
        "phase_state": [],
        "phase_R": [], 
        "phase_disr": [],
    })
    G.graph.setdefault("_epi_hist", [])
    # Auto-attach del observador estándar si se pide
    if G.graph.get("ATTACH_STD_OBSERVER", False):
        try:
            from observers import attach_standard_observer
            attach_standard_observer(G)
        except Exception as e:
            G.graph.setdefault("_callback_errors", []).append(
                {"event":"attach_std_observer","error":repr(e)}
            )
    # Hook explícito para ΔNFR (se puede sustituir luego con dynamics.set_delta_nfr_hook)
    G.graph.setdefault("compute_delta_nfr", default_compute_delta_nfr)
    G.graph.setdefault("_dnfr_hook_name", "default_compute_delta_nfr")
    # Callbacks Γ(R): before_step / after_step / on_remesh
    G.graph.setdefault("callbacks", {
        "before_step": [],
        "after_step": [],
        "on_remesh": [],
    })
    G.graph.setdefault("_CALLBACKS_DOC",
        "Interfaz Γ(R): registrar funciones (G, ctx) en callbacks['before_step'|'after_step'|'on_remesh']")
    
    # --- Inicialización configurable de θ y νf ---
    seed = int(G.graph.get("RANDOM_SEED", 0))
    init_rand_phase = bool(G.graph.get("INIT_RANDOM_PHASE", DEFAULTS.get("INIT_RANDOM_PHASE", True)))

    th_min = float(G.graph.get("INIT_THETA_MIN",  DEFAULTS.get("INIT_THETA_MIN", -math.pi)))
    th_max = float(G.graph.get("INIT_THETA_MAX",  DEFAULTS.get("INIT_THETA_MAX",  math.pi)))

    vf_mode = str(G.graph.get("INIT_VF_MODE", DEFAULTS.get("INIT_VF_MODE", "uniform"))).lower()
    vf_min_lim = float(G.graph.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max_lim = float(G.graph.get("VF_MAX", DEFAULTS["VF_MAX"]))

    vf_uniform_min = G.graph.get("INIT_VF_MIN", DEFAULTS.get("INIT_VF_MIN", None))
    vf_uniform_max = G.graph.get("INIT_VF_MAX", DEFAULTS.get("INIT_VF_MAX", None))
    if vf_uniform_min is None: vf_uniform_min = vf_min_lim
    if vf_uniform_max is None: vf_uniform_max = vf_max_lim

    vf_mean = float(G.graph.get("INIT_VF_MEAN", DEFAULTS.get("INIT_VF_MEAN", 0.5)))
    vf_std  = float(G.graph.get("INIT_VF_STD",  DEFAULTS.get("INIT_VF_STD",  0.15)))
    clamp_to_limits = bool(G.graph.get("INIT_VF_CLAMP_TO_LIMITS", DEFAULTS.get("INIT_VF_CLAMP_TO_LIMITS", True)))

    for idx, n in enumerate(G.nodes()):
        nd = G.nodes[n]
        # EPI canónico
        nd.setdefault("EPI", 0.0)

        # θ aleatoria (opt-in por flag)
        if init_rand_phase:
            th_rng = random.Random(seed + 1009 * idx)
            nd["θ"] = th_rng.uniform(th_min, th_max)
        else:
            nd.setdefault("θ", 0.0)

        # νf distribuida
        if vf_mode == "uniform":
            vf_rng = random.Random(seed * 1000003 + idx)
            vf = vf_rng.uniform(float(vf_uniform_min), float(vf_uniform_max))
        elif vf_mode == "normal":
            vf_rng = random.Random(seed * 1000003 + idx)
            # normal truncada simple (rechazo)
            for _ in range(16):
                cand = vf_rng.normalvariate(vf_mean, vf_std)
                if vf_min_lim <= cand <= vf_max_lim:
                    vf = cand
                    break
            else:
                # fallback: clamp del último candidato
                vf = min(max(vf_rng.normalvariate(vf_mean, vf_std), vf_min_lim), vf_max_lim)
        else:
            # fallback: conserva si existe, si no 0.5
            vf = float(nd.get("νf", 0.5))

        if clamp_to_limits:
            vf = min(max(vf, vf_min_lim), vf_max_lim)

        nd["νf"] = float(vf)
        
    return G

def step(G: nx.Graph, *, dt: float | None = None, use_Si: bool = True, apply_glyphs: bool = True) -> None:
    _step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)

def run(G: nx.Graph, steps: int, *, dt: float | None = None, use_Si: bool = True, apply_glyphs: bool = True) -> None:
    _run(G, steps=steps, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)

# Helper rápido para pruebas manuales
if __name__ == "__main__":
    G = nx.erdos_renyi_graph(30, 0.15)
    preparar_red(G)
    run(G, 100)
    print("C(t) muestras:", G.graph["history"]["C_steps"][-5:])
