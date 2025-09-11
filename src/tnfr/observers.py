"""Observer management."""

from __future__ import annotations
from functools import partial
import statistics

from .constants import ALIAS_THETA, get_param
from .alias import get_attr
from .helpers.numeric import angle_diff, list_pvariance
from .metrics_utils import compute_coherence
from .callback_utils import register_callback
from .glyph_history import (
    ensure_history,
    count_glyphs,
    append_metric,
    validate_window,
)
from .collections_utils import normalize_counter, mix_groups
from .constants_glyphs import GLYPH_GROUPS
from .gamma import kuramoto_R_psi
from .logging_utils import get_logger
from .import_utils import get_numpy


__all__ = (
    "attach_standard_observer",
    "std_before",
    "std_after",
    "std_on_remesh",
    "phase_sync",
    "kuramoto_order",
    "glyph_load",
    "wbar",
)


logger = get_logger(__name__)


# -------------------------
# Observador estándar Γ(R)
# -------------------------
def _std_log(kind: str, G, ctx: dict):
    """Store compact events in ``history['events']``."""
    h = ensure_history(G)
    append_metric(h, "events", (kind, dict(ctx)))


_STD_CALLBACKS = {
    "before_step": partial(_std_log, "before"),
    "after_step": partial(_std_log, "after"),
    "on_remesh": partial(_std_log, "remesh"),
}

# alias conservados por compatibilidad
std_before = _STD_CALLBACKS["before_step"]
std_after = _STD_CALLBACKS["after_step"]
std_on_remesh = _STD_CALLBACKS["on_remesh"]


def attach_standard_observer(G):
    """Register standard callbacks: before_step, after_step, on_remesh."""
    if G.graph.get("_STD_OBSERVER"):
        return G
    for event, fn in _STD_CALLBACKS.items():
        register_callback(G, event, fn)
    G.graph["_STD_OBSERVER"] = "attached"
    return G


def _get_R_psi(
    G, R: float | None = None, psi: float | None = None
) -> tuple[float, float]:
    """Return ``(R, ψ)`` using cached values if provided."""
    if R is None or psi is None:
        R_calc, psi_calc = kuramoto_R_psi(G)
        if R is None:
            R = R_calc
        if psi is None:
            psi = psi_calc
    return R, psi


def phase_sync(G, R: float | None = None, psi: float | None = None) -> float:
    if not G.number_of_nodes():
        return 1.0
    _, psi = _get_R_psi(G, R, psi)

    if (np := get_numpy()) is not None:
        th = np.fromiter(
            (
                get_attr(data, ALIAS_THETA, 0.0)
                for _, data in G.nodes(data=True)
            ),
            dtype=float,
        )
        diff = (th - psi + np.pi) % (2 * np.pi) - np.pi
        var = float(np.var(diff)) if diff.size else 0.0
    else:
        diffs = (
            angle_diff(get_attr(data, ALIAS_THETA, 0.0), psi)
            for _, data in G.nodes(data=True)
        )
        var = list_pvariance(diffs, default=0.0)

    return 1.0 / (1.0 + var)


def kuramoto_order(
    G, R: float | None = None, psi: float | None = None
) -> float:
    """R in [0,1], 1 means perfectly aligned phases."""
    if not G.number_of_nodes():
        return 1.0
    R, _ = _get_R_psi(G, R, psi)
    return float(R)


def glyph_load(G, window: int | None = None) -> dict:
    """Return distribution of glyphs applied in the network.

    - ``window``: if provided, count only the last ``window`` events per node;
      otherwise use the deque's maxlen.
    Returns a dict with proportions per glyph and useful aggregates.
    """
    if window == 0:
        return {"_count": 0}
    window_int: int | None = None
    if window is not None:
        window_int = validate_window(window, positive=True)
    total = count_glyphs(G, window=window_int, last_only=(window_int == 1))
    dist, count = normalize_counter(total)
    if count == 0:
        return {"_count": 0}
    dist = mix_groups(dist, GLYPH_GROUPS)
    dist["_count"] = count
    return dist


def wbar(G, window: int | None = None) -> float:
    """Return W̄ = mean of ``C(t)`` over a recent window.

    Uses :func:`ensure_history` to obtain ``G.graph['history']`` and falls back
    to the instantaneous coherence when ``"C_steps"`` is missing or empty.
    """
    hist = ensure_history(G)
    cs = list(hist.get("C_steps", []))
    if not cs:
        # fallback: coherencia instantánea
        return compute_coherence(G)
    w_param = get_param(G, "WBAR_WINDOW") if window is None else window
    w = validate_window(w_param, positive=True)
    w = min(len(cs), w)
    return float(statistics.fmean(cs[-w:]))
