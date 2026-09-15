"""Finite production checks of the P5 reflection quotient under REMESH.

The exact linear projection intertwines uniform unclipped REMESH. Binary64
rounding and clipping are measured separately; no assertion concerns future
runtime stability, physical memory, or a completed operator-event schedule.
"""

from collections import deque
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.operators import apply_network_remesh
from tnfr.physics.p5_reduction import reduce_p5_state


def _exact(values):
    return tuple(Fraction.from_float(float(value)) for value in values)


def _project(values):
    """Exact reflection-orbit averages, including asymmetric microstates."""
    return reduce_p5_state(values).orbit_epi


def _difference(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _graph(current, local, older, *, alpha=0.5, bounds=(-10.0, 10.0)):
    graph = nx.path_graph(len(current))
    weight = 1.0 if len(current) == 5 else 2.0
    for edge in graph.edges:
        graph.edges[edge].update(weight=weight, length=1.0)
    graph.graph.update(
        _t=0.0,
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=2,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=bounds[0],
        EPI_MAX=bounds[1],
        CLIP_MODE="hard",
    )
    for node, value in enumerate(current):
        graph.nodes[node].update(
            EPI=float(value), nu_f=1.0, theta=0.0, delta_nfr=0.0
        )
    graph.graph["_epi_hist"] = deque(
        [dict(enumerate(row)) for row in (older, local, current)], maxlen=8
    )
    return graph


def _pair(current, local, older, **kwargs):
    rows = (current, local, older)
    coarse = tuple(tuple(float(v) for v in _project(_exact(row))) for row in rows)
    # Isolate arithmetic in the map from any rounding of prepared projections.
    for fine_row, coarse_row in zip(rows, coarse, strict=True):
        assert _project(_exact(fine_row)) == _exact(coarse_row)
    return _graph(*rows, **kwargs), _graph(*coarse, **kwargs)


def _apply(graph):
    metric = (1.0, 2.0, 2.0, 2.0, 1.0) if len(graph) == 5 else (2.0, 4.0, 2.0)
    return apply_network_remesh(
        graph, include_stability_evidence=True, metric_weights=metric
    )


def _parts(result):
    alpha = Fraction.from_float(result.plan.alpha)
    beta, gamma, delta = (1 - alpha) ** 2, alpha * (1 - alpha), alpha
    ideal = tuple(
        beta * Fraction.from_float(proposal.epi_now)
        + gamma * Fraction.from_float(proposal.epi_local)
        + delta * Fraction.from_float(proposal.epi_global)
        for proposal in result.proposals
    )
    raw = _exact(proposal.raw_epi for proposal in result.proposals)
    bounded = _exact(proposal.bounded_epi for proposal in result.proposals)
    return ideal, _difference(raw, ideal), _difference(bounded, raw)


def _assert_signed_defect_identity(fine, coarse):
    fine_ideal, fine_rounding, fine_clipping = _parts(fine)
    coarse_ideal, coarse_rounding, coarse_clipping = _parts(coarse)
    assert _project(fine_ideal) == coarse_ideal
    projected = _project(_exact(p.bounded_epi for p in fine.proposals))
    coarse_output = _exact(p.bounded_epi for p in coarse.proposals)
    rounding = _difference(_project(fine_rounding), coarse_rounding)
    clipping = _difference(_project(fine_clipping), coarse_clipping)
    defect = _difference(projected, coarse_output)
    assert defect == tuple(a + b for a, b in zip(rounding, clipping, strict=True))
    return defect, rounding, clipping


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 1.0])
def test_dyadic_unclipped_remesh_commutes_with_reflection_projection(alpha):
    fine, coarse = _pair(
        (1.0, 2.0, -1.0, 0.0, -3.0),
        (-2.0, 0.0, 1.0, 4.0, 2.0),
        (3.0, -2.0, 2.0, 2.0, -1.0),
        alpha=alpha,
    )
    histories = [deepcopy(graph.graph["_epi_hist"]) for graph in (fine, coarse)]
    inputs = [
        tuple(
            (data["nu_f"], data["theta"], data["delta_nfr"])
            for _, data in graph.nodes(data=True)
        )
        for graph in (fine, coarse)
    ]
    results = [_apply(graph) for graph in (fine, coarse)]
    assert _assert_signed_defect_identity(*results) == ((0, 0, 0),) * 3
    for graph, history, before, result in zip(
        (fine, coarse), histories, inputs, results, strict=True
    ):
        assert result.applied
        assert result.plan.required_history_length == 3
        assert result.plan.history_length == 3
        assert list(graph.graph["_epi_hist"]) == list(history)
        assert tuple(
            (data["nu_f"], data["theta"], data["delta_nfr"])
            for _, data in graph.nodes(data=True)
        ) == before
        assert result.evidence.max_raw_affine_rounding_residual == 0.0
        assert not result.plan.any_clipping_intervention
        assert result.epi_time_boundary_recorded


def test_nonzero_binary64_commutator_is_explained_by_signed_rounding():
    fine, coarse = _pair(
        (0.1, 0.0, 0.0, 0.0, 0.1),
        (0.1, 0.0, 0.0, 0.0, 0.1),
        (0.2, 0.0, 0.0, 0.0, -0.1),
        alpha=0.3,
    )
    results = [_apply(graph) for graph in (fine, coarse)]
    defect, rounding, clipping = _assert_signed_defect_identity(*results)
    assert defect == rounding == (Fraction(1, 2**57), 0, 0)
    assert clipping == (0, 0, 0)
    assert all(not result.plan.any_clipping_intervention for result in results)


def test_hard_clipping_outside_source_interval_breaks_projection():
    # The oldest row deliberately lies outside the target interval. Convex
    # mixing of three rows inside one hard interval would never need clipping.
    fine, coarse = _pair(
        (0.0,) * 5,
        (0.0,) * 5,
        (4.0, 0.0, 0.0, 0.0, -2.0),
        alpha=1.0,
        bounds=(-1.0, 1.0),
    )
    fine_result, coarse_result = _apply(fine), _apply(coarse)
    defect, rounding, clipping = _assert_signed_defect_identity(
        fine_result, coarse_result
    )
    assert defect == clipping == (-1, 0, 0)
    assert rounding == (0, 0, 0)
    assert fine_result.plan.any_clipping_intervention
    assert not coarse_result.plan.any_clipping_intervention


def test_same_coarse_consensus_does_not_imply_fine_spatial_consensus():
    hidden = (1.0, 2.0, 0.0, -2.0, -1.0)
    fine, coarse = _pair(hidden, hidden, hidden)
    fine_result, coarse_result = _apply(fine), _apply(coarse)
    assert _assert_signed_defect_identity(fine_result, coarse_result)[0] == (0, 0, 0)
    fine_values = tuple(proposal.bounded_epi for proposal in fine_result.proposals)
    coarse_values = tuple(proposal.bounded_epi for proposal in coarse_result.proposals)
    assert fine_values == hidden
    assert coarse_values == (0.0, 0.0, 0.0)
    assert fine_result.evidence.observed_bounded_disagreement_energy > 0.0
    assert coarse_result.evidence.observed_bounded_disagreement_energy == 0.0


def test_causal_diffusion_history_echo_is_not_a_forward_diffusion_step():
    # For the unit-capacity P5, v=(1,0,-1,0,1) satisfies A v=v.
    # These rows are one exact diffusion orbit sampled at elapsed log(2):
    # older=4+4v, local=4+2v, current=4+v. No fabricated past is needed.
    current = (5.0, 4.0, 3.0, 4.0, 5.0)
    local = (6.0, 4.0, 2.0, 4.0, 6.0)
    older = (8.0, 4.0, 0.0, 4.0, 8.0)
    fine, coarse = _pair(current, local, older, bounds=(0.0, 8.0))
    default_compute_delta_nfr(fine)
    assert _exact(
        get_attr(fine.nodes[node], ALIAS_DNFR, None) for node in fine
    ) == (-1, 0, 1, 0, -1)
    fine_result, coarse_result = _apply(fine), _apply(coarse)
    assert _assert_signed_defect_identity(fine_result, coarse_result)[0] == (0, 0, 0)
    output = _exact(proposal.bounded_epi for proposal in fine_result.proposals)
    echo_amplitude = Fraction(11, 4)
    expected_echo = tuple(4 + echo_amplitude * v for v in (1, 0, -1, 0, 1))
    forward_diffusion = tuple(4 + Fraction(v, 2) for v in (1, 0, -1, 0, 1))
    assert output == expected_echo
    assert output != forward_diffusion
    assert fine_result.evidence.observed_current_disagreement_energy == 2.0
    assert fine_result.evidence.observed_bounded_disagreement_energy == 121 / 8
    assert not fine_result.plan.any_clipping_intervention
    # This spatial-energy increase is not a claim about augmented-history
    # energy, nor does direct invocation certify executor-owned causal history.


def test_refreshed_nodal_euler_then_remesh_transports_one_finite_cycle():
    fine, coarse = _pair(
        (0.0, 1.0, 2.0, -1.0, 3.0),
        (1.0, -2.0, 0.0, 2.0, -1.0),
        (2.0, 0.0, -1.0, 4.0, 0.0),
    )
    for graph in (fine, coarse):
        default_compute_delta_nfr(graph)
    pressure = [
        _exact(get_attr(graph.nodes[node], ALIAS_DNFR, None) for node in graph)
        for graph in (fine, coarse)
    ]
    assert _project(pressure[0]) == pressure[1]
    for graph in (fine, coarse):
        update_epi_via_nodal_equation(graph, dt=0.125, t=0.0, method="euler")
        default_compute_delta_nfr(graph)
    states = [
        _exact(get_attr(graph.nodes[node], ALIAS_EPI, None) for node in graph)
        for graph in (fine, coarse)
    ]
    assert _project(states[0]) == states[1]
    # The separately invoked map reads retained history without sampling it.
    # This finite witness does not stand in for the event/REMESH executor.
    results = [_apply(graph) for graph in (fine, coarse)]
    assert _assert_signed_defect_identity(*results) == ((0, 0, 0),) * 3
