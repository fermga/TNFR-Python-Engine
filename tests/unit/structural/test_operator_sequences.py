"""Tests for multi-operator TNFR sequences involving dissonance and stabilisation."""

from __future__ import annotations

import pytest

from tnfr.constants import D2EPI_PRIMARY, DNFR_PRIMARY, EPI_KIND_PRIMARY, inject_defaults
from tnfr.dynamics import set_delta_nfr_hook
from tnfr import glyph_history
from tnfr.structural import (
    Coherence,
    Dissonance,
    Emission,
    Reception,
    Resonance,
    Silence,
    Transition,
    create_nfr,
    run_sequence,
)
from tnfr.types import Glyph


def test_dissonance_sequence_tracks_bifurcation_pressure_and_stabilises() -> None:
    """ΔNFR sequences log bifurcation pressure and honour grammar cut-offs."""

    G, node = create_nfr("probe", epi=0.2, vf=1.0, theta=0.05)
    inject_defaults(G)
    G.graph["GLYPH_HYSTERESIS_WINDOW"] = 12
    G.graph["HISTORY_MAXLEN"] = 4
    G.graph["_sel_norms"] = {"dnfr_max": 1.0, "accel_max": 1.0}

    nd = G.nodes[node]
    initial_dnfr = 0.12
    nd[DNFR_PRIMARY] = initial_dnfr
    nd[D2EPI_PRIMARY] = 0.0

    dnfr_progression: list[float] = []
    accel_progression: list[float] = []
    pending_cutoff = False
    ready_for_silence = False
    oz_count = 0

    def scripted_delta(graph) -> None:
        nonlocal pending_cutoff, ready_for_silence, oz_count
        nd_local = graph.nodes[node]
        history = list(nd_local.get("glyph_history", ()))
        last = history[-1] if history else None
        record = False
        if last == Glyph.OZ.value:
            record = True
            pending_cutoff = True
            ready_for_silence = False
            oz_count += 1
            nd_local[D2EPI_PRIMARY] = 0.8 if oz_count == 1 else 0.3
        elif last == Glyph.IL.value and pending_cutoff:
            record = True
            pending_cutoff = False
            ready_for_silence = True
            nd_local[DNFR_PRIMARY] = nd_local[DNFR_PRIMARY] * 0.4
            nd_local[D2EPI_PRIMARY] = 0.1
        elif last == Glyph.SHA.value and ready_for_silence:
            record = True
            ready_for_silence = False
            nd_local[DNFR_PRIMARY] = nd_local[DNFR_PRIMARY] * 0.2
            nd_local[D2EPI_PRIMARY] = 0.05
        if record:
            dnfr_progression.append(nd_local[DNFR_PRIMARY])
            accel_progression.append(nd_local[D2EPI_PRIMARY])
            hist_store = glyph_history.ensure_history(graph)
            glyph_history.append_metric(
                hist_store,
                "bifurcation_pressure",
                nd_local[D2EPI_PRIMARY],
            )

    set_delta_nfr_hook(G, scripted_delta)

    run_sequence(
        G,
        node,
        [
            Emission(),
            Reception(),
            Coherence(),
            Resonance(),
            Transition(),
            Dissonance(),
            Transition(),
            Dissonance(),
            Transition(),
            Coherence(),
            Silence(),
        ],
    )

    nd_final = G.nodes[node]
    history = list(nd_final.get("glyph_history", ()))
    assert history.count(Glyph.OZ.value) == 2
    last_oz = max(idx for idx, glyph in enumerate(history) if glyph == Glyph.OZ.value)
    assert last_oz < history.index(Glyph.IL.value)
    assert abs(dnfr_progression[0]) > abs(initial_dnfr)
    assert abs(nd_final[DNFR_PRIMARY]) < abs(dnfr_progression[0])

    hist_store = G.graph["history"]
    assert "bifurcation_pressure" in hist_store
    assert list(hist_store["bifurcation_pressure"]) == pytest.approx([0.8, 0.3, 0.1, 0.05])

    assert history[-2:] == [Glyph.IL.value, Glyph.SHA.value]
    assert max(accel_progression) == pytest.approx(0.8)
    assert accel_progression[-1] < 0.6  # cutoff after stabilisation
    assert nd_final[EPI_KIND_PRIMARY] == Glyph.SHA.value
