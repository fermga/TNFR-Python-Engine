"""Orquesta la simulación canónica."""

from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING

from .constants import METRIC_DEFAULTS, attach_defaults, get_param
from .dynamics import step as _step, run as _run
from .dynamics import default_compute_delta_nfr
from .initialization import init_node_attrs
from .glyph_history import append_metric
from .import_utils import optional_import

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

# API de alto nivel


def preparar_red(
    G: "nx.Graph",
    *,
    init_attrs: bool = True,
    override_defaults: bool = False,
    **overrides,
) -> "nx.Graph":
    """Prepara ``G`` para simulación.

    Parameters
    ----------
    init_attrs:
        Ejecuta ``init_node_attrs`` si es ``True`` (por defecto),
        dejando los atributos de nodos intactos cuando es ``False``.
    override_defaults:
        Si ``True``, `attach_defaults` sobreescribe entradas existentes.
    **overrides:
        Parámetros para aplicar tras la fase de defaults.
    """
    attach_defaults(G, override=override_defaults)
    if overrides:
        from .constants import merge_overrides

        merge_overrides(G, **overrides)
    # Inicializaciones blandas
    ph_len = int(
        G.graph.get(
            "PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"]
        )
    )
    G.graph.setdefault(
        "history",
        {
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
            "phase_state": deque(maxlen=ph_len),
            "phase_R": deque(maxlen=ph_len),
            "phase_disr": deque(maxlen=ph_len),
        },
    )
    # REMESH_TAU: alias legado resuelto por ``get_param``
    tau = int(get_param(G, "REMESH_TAU_GLOBAL"))
    maxlen = max(2 * tau + 5, 64)
    G.graph.setdefault("_epi_hist", deque(maxlen=maxlen))
    # Auto-attach del observador estándar si se pide
    if G.graph.get("ATTACH_STD_OBSERVER", False):
        attach_standard_observer = optional_import(
            "tnfr.observers.attach_standard_observer"
        )
        if attach_standard_observer is not None:
            attach_standard_observer(G)
        else:
            append_metric(
                G.graph,
                "_callback_errors",
                {"event": "attach_std_observer", "error": "ImportError"},
            )
    # Hook explícito para ΔNFR (se puede sustituir luego con
    # dynamics.set_delta_nfr_hook)
    G.graph.setdefault("compute_delta_nfr", default_compute_delta_nfr)
    G.graph.setdefault("_dnfr_hook_name", "default_compute_delta_nfr")
    # Callbacks Γ(R): before_step / after_step / on_remesh
    G.graph.setdefault(
        "callbacks",
        {
            "before_step": [],
            "after_step": [],
            "on_remesh": [],
        },
    )
    G.graph.setdefault(
        "_CALLBACKS_DOC",
        "Interfaz Γ(R): registrar pares (name, func) con firma (G, ctx) "
        "en callbacks['before_step'|'after_step'|'on_remesh']",
    )

    if init_attrs:
        init_node_attrs(G, override=True)
    return G


def step(
    G: "nx.Graph",
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None:
    _step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)


def run(
    G: "nx.Graph",
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None:
    _run(G, steps=steps, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
