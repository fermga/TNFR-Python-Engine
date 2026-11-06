"""Fine-field dependencies beyond one closed internal-mode observation.

These are detached prism snapshots, not trajectories or a new dynamics.
The tests retain stored-pressure, path-length and numerical provenance.
"""

from collections import Counter
from fractions import Fraction as Q
from math import atan2, pi, sqrt

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import NODES, PROJECTION, _apply, _graph
from tnfr.physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing

_BASE_PRESSURE = (Q(-1, 4), Q(1, 4), Q(0)) * 2
_BASE_POTENTIAL = (Q(1, 16), Q(-1, 16), Q(0)) * 2


def _prepare_pressure(graph):
    # Populate detached input data from the actual fresh pressure owner.
    # This is fixture construction, not a pressure-refresh runtime event.
    captured = capture_non_epi_forcing(graph)
    for node, pressure in zip(NODES, captured.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(pressure)
    prepared = capture_non_epi_forcing(graph)
    assert prepared.full_kernel_pressure == captured.full_kernel_pressure
    assert prepared.stored_pressure_residual == (0,) * 6
    return prepared


def _ordered(field):
    return tuple(field[node] for node in NODES)


def _c5(observation):
    """Four internal coefficients and the omitted fiber-mean contrast."""
    epi = observation.snapshot.epi
    return (*_apply(PROJECTION, epi), (sum(epi[:3]) - sum(epi[3:])) / 3)


def _tetrad(graph):
    return (
        compute_structural_potential(graph),
        compute_phase_gradient(graph),
        compute_phase_curvature(graph),
        estimate_coherence_length_with_provenance(graph),
    )


def test_omitted_fiber_mean_contrast_changes_fresh_pressure_and_potential():
    first_graph = _graph()
    second_graph = _graph(means=(Q(11, 16), Q(5, 16)))
    first, second = map(_prepare_pressure, (first_graph, second_graph))
    assert (
        _apply(PROJECTION, first.snapshot.epi)
        == _apply(PROJECTION, second.snapshot.epi)
        == (Q(1, 4), 0, Q(1, 4), 0)
    )
    assert _c5(first)[-1] == 0
    assert _c5(second)[-1] == Q(3, 8)
    assert first.kernel_pressure_defect == (0,) * 6
    assert first.full_kernel_pressure == first.snapshot.epi_gradient == _BASE_PRESSURE
    ideal_second_pressure = (
        Q(-3, 8),
        Q(1, 8),
        Q(-1, 8),
        Q(-1, 8),
        Q(3, 8),
        Q(1, 8),
    )
    pressure_defect = (Q(1, 2**54), 0, 0, 0, Q(-1, 2**54), 0)
    assert second.snapshot.epi_gradient == ideal_second_pressure
    assert second.kernel_pressure_defect == pressure_defect
    assert second.full_kernel_pressure == tuple(
        value + defect
        for value, defect in zip(ideal_second_pressure, pressure_defect, strict=True)
    )
    assert (
        tuple(
            b - a
            for a, b in zip(
                first.snapshot.epi_gradient,
                second.snapshot.epi_gradient,
                strict=True,
            )
        )
        == (Q(-1, 8),) * 3 + (Q(1, 8),) * 3
    )
    first_phi = _ordered(compute_structural_potential(first_graph))
    second_phi = _ordered(compute_structural_potential(second_graph))
    assert first_phi == _BASE_POTENTIAL
    ideal_second_phi = (0, Q(-1, 8), Q(-1, 16), Q(1, 8), 0, Q(1, 16))
    # This dyadic potential error is exactly the inverse-square aggregation
    # of the two measured pressure defects; do not erase it as model pressure.
    potential_defect = (
        Q(-1, 2**56),
        0,
        Q(3, 2**56),
        0,
        Q(1, 2**56),
        Q(-3, 2**56),
    )
    assert second_phi == tuple(
        value + defect
        for value, defect in zip(ideal_second_phi, potential_defect, strict=True)
    )
    assert tuple(b - a for a, b in zip(first_phi, second_phi, strict=True)) == tuple(
        value + defect
        for value, defect in zip(
            (Q(-1, 16),) * 3 + (Q(1, 16),) * 3,
            potential_defect,
            strict=True,
        )
    )
    # The exact model's internal rate still closes; diagnostic reconstruction
    # and captured binary64 defects are separate from that identity.
    assert _apply(PROJECTION, first.snapshot.epi_gradient) == _apply(
        PROJECTION, second.snapshot.epi_gradient
    )


def test_common_epi_shift_preserves_the_tetrad_on_this_exact_dyadic_fixture():
    first_graph = _graph()
    second_graph = _graph(means=(Q(11, 16), Q(11, 16)))
    first, second = map(_prepare_pressure, (first_graph, second_graph))
    assert (
        tuple(
            b - a
            for a, b in zip(
                first.snapshot.epi,
                second.snapshot.epi,
                strict=True,
            )
        )
        == (Q(3, 16),) * 6
    )
    assert _c5(first) == _c5(second)
    assert first.kernel_pressure_defect == second.kernel_pressure_defect == (0,) * 6
    assert first.full_kernel_pressure == second.full_kernel_pressure == _BASE_PRESSURE
    assert _tetrad(first_graph) == _tetrad(second_graph)
    # This fixture also realizes the exact-real invariance in binary64.
    # It supplies no general rounding or arbitrary pressure-law theorem.


def test_primitive_phase_changes_fields_without_changing_internal_modal_angles():
    first_graph, second_graph = _graph(), _graph()
    for a, i in NODES:
        second_graph.nodes[a, i]["theta"] = a * pi / 3
    first, second = map(_prepare_pressure, (first_graph, second_graph))
    assert first.snapshot.epi == second.snapshot.epi
    assert _c5(first) == _c5(second)
    coefficients = _apply(PROJECTION, first.snapshot.epi)
    angles = tuple(
        atan2(sqrt(6) * coefficients[2 * a + 1], sqrt(2) * coefficients[2 * a])
        for a in range(2)
    )
    assert angles == (0, 0)
    assert first.kernel_pressure_defect == second.kernel_pressure_defect == (0,) * 6
    assert first.full_kernel_pressure == second.full_kernel_pressure == _BASE_PRESSURE
    assert compute_structural_potential(first_graph) == compute_structural_potential(
        second_graph
    )
    assert _ordered(compute_phase_gradient(first_graph)) == (0,) * 6
    assert _ordered(compute_phase_curvature(first_graph)) == (0,) * 6
    assert _ordered(compute_phase_gradient(second_graph)) == pytest.approx(
        (pi / 9,) * 6
    )
    curvature = atan2(sqrt(3), 5)
    assert _ordered(compute_phase_curvature(second_graph)) == pytest.approx(
        (-curvature,) * 3 + (curvature,) * 3
    )
    evidence = observe_phase_curvature(second_graph)
    assert tuple(row.status for row in evidence.rows) == ("defined",) * 6
    assert all(not row.resultant.joint_zero for row in evidence.rows)
    # Fine intrafiber neighbors affect both phase fields. The same primitive
    # phase pair on a coarse P2 does not inherit those fine read-outs.
    coarse = nx.path_graph(2)
    nx.set_node_attributes(coarse, {0: 0.0, 1: pi / 3}, "theta")
    assert tuple(compute_phase_gradient(coarse).values()) == pytest.approx(
        (pi / 3,) * 2
    )
    assert tuple(compute_phase_curvature(coarse).values()) == pytest.approx(
        (-pi / 3, pi / 3)
    )
    # Pressure independence here uses phase_weight=0; it is not a statement
    # about a general multichannel pressure realization.


def test_prism_coherence_length_is_a_spectral_fallback_not_a_resolved_fit():
    graphs = (_graph(), _graph(means=(Q(11, 16), Q(5, 16))), _graph())
    for a, i in NODES:
        graphs[-1].nodes[a, i]["theta"] = a * pi / 3
    observed = []
    for graph in graphs:
        _prepare_pressure(graph)
        distances = dict(nx.all_pairs_shortest_path_length(graph))
        bins = Counter(
            distances[left][right]
            for i, left in enumerate(NODES)
            for right in NODES[i + 1 :]
        )
        assert bins == {1: 9, 2: 6}
        result = estimate_coherence_length_with_provenance(graph)
        assert result.method == "spectral_gap"
        assert result.fit_available is False
        assert result.fit_quality == "autocorrelation fit unavailable"
        assert result.graph_regime == "undirected; connected"
        assert result.value == pytest.approx(sqrt(Q(3, 2)))
        assert "dimensionless normalized-generator" in result.distance_weighting
        observed.append(result)
    assert observed[0] == observed[1] == observed[2]
    # The shared fit requires three distance bins; this unit prism has two.
    # Its unchanged fallback is not evidence for an unchanged fitted range.


def test_same_five_epi_coordinates_do_not_authenticate_stored_pressure():
    first_graph, second_graph = _graph(), _graph()
    first = _prepare_pressure(first_graph)
    _prepare_pressure(second_graph)
    offset = Q(1, 8)
    for node in NODES:
        second_graph.nodes[node]["delta_nfr"] += float(offset)
    second = capture_non_epi_forcing(second_graph)
    assert _c5(first) == _c5(second)
    assert first.snapshot.epi == second.snapshot.epi
    assert first.full_kernel_pressure == second.full_kernel_pressure == _BASE_PRESSURE
    assert second.kernel_pressure_defect == (0,) * 6
    assert second.stored_pressure_residual == (offset,) * 6
    first_phi = _ordered(compute_structural_potential(first_graph))
    second_phi = _ordered(compute_structural_potential(second_graph))
    assert (
        tuple(b - a for a, b in zip(first_phi, second_phi, strict=True))
        == (Q(7, 16),) * 6
    )
    # Phi_s reads stored pressure. A reconstruction using fresh model pressure
    # needs that premise or the retained residual, even when C5 is sufficient
    # for the exact pure-EPI model under fixed capacities and support.


def test_explicit_path_length_is_a_separate_input_to_potential_reconstruction():
    first_graph, second_graph = _graph(), _graph()
    nx.set_edge_attributes(second_graph, 2.0, "length")
    first, second = map(_prepare_pressure, (first_graph, second_graph))
    assert _c5(first) == _c5(second)
    assert first.snapshot.conductance == second.snapshot.conductance
    assert first.full_kernel_pressure == second.full_kernel_pressure == _BASE_PRESSURE
    first_phi = _ordered(compute_structural_potential(first_graph))
    second_phi = _ordered(compute_structural_potential(second_graph))
    assert first_phi == _BASE_POTENTIAL
    assert second_phi == tuple(value / 4 for value in first_phi)
    # alpha=2 turns this distance doubling into a factor 1/4. Transport
    # conductance and the normalized-generator fallback stay unchanged.
    assert estimate_coherence_length_with_provenance(first_graph) == (
        estimate_coherence_length_with_provenance(second_graph)
    )


def test_exact_five_coordinate_pressure_retains_the_fresh_thirds_rounding_defect():
    coefficients = (Q(1, 8), Q(1, 24), Q(0), Q(0))
    first = capture_non_epi_forcing(_graph(coefficients, means=(Q(7, 12), Q(1, 2))))
    shifted = capture_non_epi_forcing(_graph(coefficients, means=(Q(1, 3), Q(1, 4))))
    assert first.snapshot.epi == (Q(3, 4),) + (Q(1, 2),) * 5
    assert shifted.snapshot.epi == (Q(1, 2),) + (Q(1, 4),) * 5
    assert _c5(first) == _c5(shifted)
    exact = (Q(-1, 4), Q(1, 12), Q(1, 12), Q(1, 12), Q(0), Q(0))
    assert first.snapshot.epi_gradient == shifted.snapshot.epi_gradient == exact
    defect = (Q(0),) + (Q(-1, 3 * 2**56),) * 3 + (Q(0),) * 2
    assert first.kernel_pressure_defect == shifted.kernel_pressure_defect == defect
    assert (
        first.full_kernel_pressure
        == shifted.full_kernel_pressure
        == tuple(
            pressure + error for pressure, error in zip(exact, defect, strict=True)
        )
    )
    # The actual capture owner accumulates contrasts: this pair preserves
    # translation equality, despite its nonzero exact-model rounding defect.
    # A separately rounded neighbor mean is not this kernel's arithmetic.
