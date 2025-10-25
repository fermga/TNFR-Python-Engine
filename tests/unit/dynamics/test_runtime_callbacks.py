"""Tests for runtime callback helpers."""

from __future__ import annotations

from collections import deque

from tnfr.callback_utils import CallbackEvent, callback_manager
from tnfr.dynamics.runtime import _run_after_callbacks, _run_before_callbacks
from tnfr.glyph_history import ensure_history
from tnfr.structural import create_nfr


def test_run_after_callbacks_exposes_latest_history_metrics() -> None:
    """Registered AFTER_STEP callbacks should see the latest history values."""

    G, _ = create_nfr("seed", epi=0.1, vf=1.0)
    G.graph["HISTORY_MAXLEN"] = 6
    hist = ensure_history(G)

    hist["C_steps"] = deque([0.42, 0.5, 0.58], maxlen=6)
    hist["stable_frac"] = deque([0.8, 0.83, 0.86], maxlen=6)
    hist["phase_sync"] = [0.49, 0.53, 0.61]
    hist["glyph_load_disr"] = [0.12, 0.14, 0.18]
    hist["Si_mean"] = [0.21, 0.24, 0.27]

    captured: list[dict[str, float | int]] = []

    def capture_context(graph, ctx):
        captured.append(dict(ctx))

    callback_manager.register_callback(
        G,
        CallbackEvent.AFTER_STEP,
        capture_context,
        name="capture_latest_metrics",
    )

    _run_after_callbacks(G, step_idx=3)

    assert captured == [
        {
            "step": 3,
            "C": 0.58,
            "stable_frac": 0.86,
            "phase_sync": 0.61,
            "glyph_disr": 0.18,
            "Si_mean": 0.27,
        }
    ]


def test_run_before_callbacks_provides_execution_context() -> None:
    """Registered BEFORE_STEP callbacks should receive the execution context."""

    G, _ = create_nfr("seed", epi=0.2, vf=2.0)

    captured: list[dict[str, float | int | bool | None]] = []

    def capture_context(graph, ctx):
        captured.append(dict(ctx))

    callback_manager.register_callback(
        G,
        CallbackEvent.BEFORE_STEP,
        capture_context,
        name="capture_before_context",
    )

    _run_before_callbacks(
        G,
        step_idx=5,
        dt=0.125,
        use_Si=True,
        apply_glyphs=False,
    )

    assert captured == [
        {
            "step": 5,
            "dt": 0.125,
            "use_Si": True,
            "apply_glyphs": False,
        }
    ]
