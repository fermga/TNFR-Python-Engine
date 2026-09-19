"""The optional additive source is a distinct model, including at zero capacity."""

import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.constants.aliases import ALIAS_DEPI
from tnfr.dynamics import integrators


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("forced", [False, True])
def test_zero_capacity_freezes_only_the_unforced_nodal_channel(
    monkeypatch, vectorized, forced
):
    if not vectorized:
        monkeypatch.setattr(integrators, "np", None)
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.graph.update(DT_MIN=0.0, CLIP_MODE="hard", use_extended_dynamics=False)
    graph.nodes[0].update({EPI_PRIMARY: 0.25, VF_PRIMARY: 0.0, DNFR_PRIMARY: 2.0})
    graph.graph["GAMMA"] = (
        {"type": "harmonic", "beta": 0.125, "omega": 0.0, "phi": math.pi / 2}
        if forced
        else {"type": "none"}
    )
    integrators.update_epi_via_nodal_equation(graph, dt=0.5, method="euler")
    expected_rate = 0.125 if forced else 0.0
    assert graph.nodes[0][EPI_PRIMARY] == 0.25 + 0.5 * expected_rate
    assert get_attr(graph.nodes[0], ALIAS_DEPI) == expected_rate
    assert graph.nodes[0][VF_PRIMARY] == 0.0
    assert graph.nodes[0][DNFR_PRIMARY] == 2.0
