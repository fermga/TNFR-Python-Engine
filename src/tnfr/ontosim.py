"""Orchestrate the canonical simulation."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .utils import CallbackEvent
from .constants import METRIC_DEFAULTS, get_param, inject_defaults
from .dynamics import default_compute_delta_nfr
from .dynamics import run as _run
from .dynamics import step as _step
from .glyph_history import append_metric
from .initialization import init_node_attrs
from .utils import cached_import

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

# High-level API exports
__all__ = ("prepare_network", "step", "run")


def prepare_network(
    G: "nx.Graph",
    *,
    init_attrs: bool = True,
    override_defaults: bool = False,
    **overrides,
) -> "nx.Graph":
    """Prepare ``G`` for simulation.

    Parameters
    ----------
    init_attrs:
        Run ``init_node_attrs`` when ``True`` (default), leaving node
        attributes untouched when ``False``.
    override_defaults:
        If ``True``, :func:`inject_defaults` overwrites existing entries.
    **overrides:
        Parameters applied after the defaults phase.
    """
    inject_defaults(G, override=override_defaults)
    if overrides:
        from .constants import merge_overrides

        merge_overrides(G, **overrides)
    # Initialize history buffers
    ph_len = int(G.graph.get("PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"]))
    hist_keys = [
        "C_steps",
        "stable_frac",
        "phase_sync",
        "kuramoto_R",
        "sense_sigma_x",
        "sense_sigma_y",
        "sense_sigma_mag",
        "sense_sigma_angle",
        "iota",
        "glyph_load_stabilizers",
        "glyph_load_disr",
        "Si_mean",
        "Si_hi_frac",
        "Si_lo_frac",
        "W_bar",
        "phase_kG",
        "phase_kL",
    ]
    history = {k: [] for k in hist_keys}
    history.update(
        {
            "phase_state": deque(maxlen=ph_len),
            "phase_R": deque(maxlen=ph_len),
            "phase_disr": deque(maxlen=ph_len),
        }
    )
    G.graph.setdefault("history", history)
    # Global REMESH memory
    tau = int(get_param(G, "REMESH_TAU_GLOBAL"))
    maxlen = max(2 * tau + 5, 64)
    G.graph.setdefault("_epi_hist", deque(maxlen=maxlen))
    # Auto-attach the standard observer when requested
    if G.graph.get("ATTACH_STD_OBSERVER", False):
        attach_standard_observer = cached_import(
            "tnfr.observers",
            "attach_standard_observer",
        )
        if attach_standard_observer is not None:
            attach_standard_observer(G)
        else:
            append_metric(
                G.graph,
                "_callback_errors",
                {"event": "attach_std_observer", "error": "ImportError"},
            )
    # Explicit hook for ΔNFR (can later be replaced with
    # dynamics.set_delta_nfr_hook)
    G.graph.setdefault("compute_delta_nfr", default_compute_delta_nfr)
    G.graph.setdefault("_dnfr_hook_name", "default_compute_delta_nfr")
    # Callbacks Γ(R): before_step / after_step / on_remesh
    G.graph.setdefault(
        "callbacks",
        {
            CallbackEvent.BEFORE_STEP.value: [],
            CallbackEvent.AFTER_STEP.value: [],
            CallbackEvent.ON_REMESH.value: [],
        },
    )
    G.graph.setdefault(
        "_CALLBACKS_DOC",
        "Γ(R) interface: register (name, func) pairs with signature (G, ctx) "
        "in callbacks['before_step'|'after_step'|'on_remesh']",
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
    """Advance the ontosim runtime by a single step."""

    _step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)


def run(
    G: "nx.Graph",
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None:
    """Advance the ontosim runtime ``steps`` times with optional overrides."""

    _run(G, steps=steps, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
