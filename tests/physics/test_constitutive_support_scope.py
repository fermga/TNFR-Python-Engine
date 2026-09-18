"""Scale freedom of declared conductance, not an inferred support evolution law.

Rational identities and finite dyadic binary64 read-outs are distinct controls.
No graph trajectory, operator, topology update or pressure write is executed.
"""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_derivative,
)

F = Fraction


def _graph(scale=1, *, explicit_lengths=False):
    graph = nx.Graph()
    graph.add_nodes_from(("center", "left", "right", "leaf"))
    for neighbor, weight in (("left", 1), ("right", 2), ("leaf", 4)):
        attributes = {"weight": float(scale * weight)}
        if explicit_lengths:
            attributes["length"] = float(weight)
        graph.add_edge("center", neighbor, **attributes)
    for node, epi, capacity, phase, pressure in zip(
        graph,
        (1.0, -0.5, 0.25, 2.0),
        (1.0, 2.0, 0.5, 4.0),
        (0.0, 0.25, -0.5, 0.75),
        (0.5, 1.0, -0.25, 0.125),
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi, nu_f=capacity, theta=phase, delta_nfr=pressure
        )
    graph.graph.update(
        DNFR_WEIGHTS={name: 1.0 for name in ("phase", "epi", "vf", "topo")},
        compute_delta_nfr=default_compute_delta_nfr,
    )
    return graph


def _normalized_rows(entries, count):
    rows = [[F(0) for _ in range(count)] for _ in range(count)]
    for i, j, weight in entries:
        rows[i][j] = weight
    return tuple(
        tuple(value / sum(row) if sum(row) else F(0) for value in row) for row in rows
    )


@pytest.mark.parametrize("scale", (F(1, 7), F(3, 2), F(17)))
def test_exact_positive_scale_leaves_every_normalized_row_including_isolate(scale):
    # General rational coefficients, deliberately not a claim about float casts.
    entries = (
        (0, 0, F(1, 3)),
        (0, 1, F(2, 7)),
        (1, 0, F(2, 7)),
        (1, 2, F(5, 11)),
        (2, 1, F(5, 11)),
    )
    scaled = tuple((i, j, scale * weight) for i, j, weight in entries)
    assert _normalized_rows(scaled, 4) == _normalized_rows(entries, 4)
    assert _normalized_rows(entries, 4)[3] == (0, 0, 0, 0)


@pytest.mark.parametrize("scale", (F(1, 2), F(2), F(8)))
def test_fixed_support_full_pressure_is_invariant_in_finite_dyadic_fixture(scale):
    first, second = _graph(), _graph(scale)
    saved = [
        (deepcopy(dict(g.nodes(data=True))), deepcopy(dict(g.edges)), deepcopy(g.graph))
        for g in (first, second)
    ]
    baseline, scaled = (capture_non_epi_forcing(g) for g in (first, second))
    assert scaled.snapshot.epi_gradient == baseline.snapshot.epi_gradient
    assert scaled.snapshot.capacity_gradient == baseline.snapshot.capacity_gradient
    assert scaled.snapshot.topology_gradient == baseline.snapshot.topology_gradient
    assert scaled.phase_gradient == baseline.phase_gradient
    assert scaled.forcing == baseline.forcing
    assert scaled.normalized_weights == baseline.normalized_weights
    assert scaled.full_kernel_pressure == baseline.full_kernel_pressure
    assert scaled.kernel_pressure_defect == baseline.kernel_pressure_defect
    assert (
        scaled.snapshot.dirichlet_energy == scale * baseline.snapshot.dirichlet_energy
    )
    for graph, expected in zip((first, second), saved, strict=True):
        assert (
            dict(graph.nodes(data=True)),
            dict(graph.edges),
            graph.graph,
        ) == expected


def test_same_initial_support_has_distinct_declared_tangents_and_energy_work():
    source = observe_support_transport(_graph())
    # W_1(t)=W0 and W_2(t)=(1+3t)W0 agree at t=0. Both are positive
    # near zero. Their distinct supplied derivatives are not runtime laws.
    static = observe_support_transport_derivative(
        source,
        conductance_rates=(F(0),) * len(source.conductance),
    )
    growing = observe_support_transport_derivative(
        source,
        conductance_rates=tuple(3 * w for _, _, w in source.conductance),
    )
    assert static.source == growing.source == source
    assert static.conductance_rates != growing.conductance_rates
    assert growing.geometry_gradient_rate == static.geometry_gradient_rate == (0,) * 4
    assert growing.epi_gradient_rate == static.epi_gradient_rate
    assert growing.nodal_work == static.nodal_work
    assert static.conductance_work == 0
    assert growing.conductance_work == 3 * source.dirichlet_energy > 0
    assert growing.energy_rate - static.energy_rate == 3 * source.dirichlet_energy


@pytest.mark.parametrize("explicit_lengths", (False, True))
def test_potential_distinguishes_metric_length_from_conductance(explicit_lengths):
    first = _graph(explicit_lengths=explicit_lengths)
    second = _graph(2, explicit_lengths=explicit_lengths)
    before = compute_structural_potential(first)
    after = compute_structural_potential(second)
    factor = 1 if explicit_lengths else F(1, 4)
    # Dyadic scaling of the same positive paths gives an exact finite result.
    assert after == {node: factor * value for node, value in before.items()}
    assert any(before.values())
    assert compute_phase_gradient(first) == compute_phase_gradient(second)
    assert compute_phase_curvature(first) == compute_phase_curvature(second)
    # Four nodes cannot supply the correlation fit's ten distinct pairs.
    # This explicitly tests the normalized-spectrum fallback, not every xi fit.
    xi_first = estimate_coherence_length_with_provenance(first)
    xi_second = estimate_coherence_length_with_provenance(second)
    assert xi_first.method == xi_second.method == "spectral_gap"
    assert xi_first.value == xi_second.value


def test_zero_conductance_support_still_changes_non_epi_channels():
    connected = _graph()
    connected.edges["center", "leaf"]["weight"] = 0.0
    absent = _graph()
    absent.remove_edge("center", "leaf")
    with_support = capture_non_epi_forcing(connected)
    without_support = capture_non_epi_forcing(absent)
    assert with_support.snapshot.conductance == without_support.snapshot.conductance
    assert with_support.snapshot.epi_gradient == without_support.snapshot.epi_gradient
    assert (
        with_support.snapshot.capacity_gradient
        != without_support.snapshot.capacity_gradient
    )
    assert with_support.phase_gradient != without_support.phase_gradient
    assert with_support.full_kernel_pressure != without_support.full_kernel_pressure
