"""Direct RA tracking must record canonical network C(t)."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.metrics.common import compute_coherence
from tnfr.operators.definitions import Resonance


def test_direct_resonance_records_canonical_coherence_samples() -> None:
    graph = nx.path_graph(2)
    graph.nodes[0].update(
        EPI=0.2,
        nu_f=1.0,
        theta=0.0,
        delta_nfr=0.2,
        dEPI_dt=0.2,
    )
    graph.nodes[1].update(
        EPI=0.8,
        nu_f=1.0,
        theta=0.2,
        delta_nfr=-0.4,
        dEPI_dt=-0.4,
    )
    graph.graph["TRACK_NETWORK_COHERENCE"] = True
    before = float(compute_coherence(graph))

    Resonance()(graph, 0)

    samples = graph.graph["_ra_c_tracking"]
    assert len(samples) == 1
    assert samples[0]["node"] == 0
    assert samples[0]["c_before"] == pytest.approx(before)
    assert samples[0]["c_after"] == pytest.approx(compute_coherence(graph))
    assert samples[0]["c_delta"] == pytest.approx(
        samples[0]["c_after"] - samples[0]["c_before"]
    )