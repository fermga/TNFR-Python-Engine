"""One shared U3 policy resolution governs each simultaneous phase proposal."""

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.errors import TNFRValueError
from tnfr.operators import _phase_gate


@pytest.mark.parametrize("nodes", [0, 3])
@pytest.mark.parametrize("parameter", ["DELTA_PHI_MAX", "UM_MAX_PHASE_DIFF"])
def test_invalid_gate_rejected_even_without_nodes(nodes, parameter):
    graph = nx.empty_graph(nodes)
    graph.graph[parameter] = True
    with pytest.raises(TNFRValueError, match="U3 phase gate"):
        propose_u3_gated_phase_step(
            graph,
            tuple(graph),
            [0.0] * nodes,
            [1.0] * nodes,
            dt=0.25,
            coupling_strength=0.5,
        )


def test_policy_resolved_once_with_snapshot_mean_and_um_tightening(monkeypatch):
    graph = nx.path_graph(3)
    graph.graph.update(DELTA_PHI_MAX=0.5, UM_MAX_PHASE_DIFF=0.25)
    calls = []
    original = _phase_gate.resolve_u3_phase_limits

    def resolve(attributes, *, operator_code):
        calls.append(operator_code)
        return original(attributes, operator_code=operator_code)

    monkeypatch.setattr(_phase_gate, "resolve_u3_phase_limits", resolve)
    proposal = propose_u3_gated_phase_step(
        graph,
        (0, 1, 2),
        [-0.25, 0.0, 0.5],
        [0.0, 0.5, 1.0],
        dt=0.25,
        coupling_strength=0.125,
    )
    expected = np.mod(
        [-0.25 + math.sin(0.25) / 32, 0.125 - math.sin(0.25) / 32, 0.75], math.tau
    )
    np.testing.assert_array_equal(proposal, expected)
    assert calls == ["UM"]
