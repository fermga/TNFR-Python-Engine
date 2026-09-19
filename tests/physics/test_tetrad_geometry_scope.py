"""Tetrad dependencies beyond closed means and averaged model potential.

The P5 reflection observation reuses the existing exact reduction. Inputs
are detached declared snapshots, not executed trajectories or new phase laws.
Exact pressure factorization and numerical fit provenance are kept separate.
"""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction as F
from math import isfinite, log, pi, sqrt

import networkx as nx
import pytest

from tests.physics.test_forced_epi_closure import _apply, _identity, _product, _subtract
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.epi_memory import observe_affine_nodal_realization
from tnfr.physics.geometry_realization import observe_forced_support_tetrad_dependencies
from tnfr.physics.p5_reduction import p5_reduction_geometry, reduce_p5_state

P5_BLOCKS = ((0, 4), (1, 3), (2,))
P3_BLOCKS = ((0, 2), (1,))


def _path(size=5, *, amplitude=1, stored_amplitude=None):
    graph = nx.path_graph(size)
    if stored_amplitude is None:
        stored_amplitude = amplitude
    for node in graph:
        pressure = (
            stored_amplitude
            if node == 0
            else -stored_amplitude if node == size - 1 else 0
        )
        graph.nodes[node].update(
            EPI=amplitude * (node - (size - 1) // 2),
            nu_f=1,
            theta=0,
            delta_nfr=pressure,
        )
    for edge in graph.edges:
        graph.edges[edge].update(weight=1, length=1)
    graph.graph["DNFR_WEIGHTS"] = {"phase": 0, "epi": 1, "vf": 0, "topo": 0}
    graph.graph["unrelated_record"] = {"retained": [1, 2]}
    return graph


def _observe(graph, blocks=P5_BLOCKS):
    before = deepcopy(graph)
    result = observe_forced_support_tetrad_dependencies(graph, blocks)
    # This includes the fallback's spectral cache: it must stay on its copy.
    assert graph.graph == before.graph
    assert tuple(graph.nodes(data=True)) == tuple(before.nodes(data=True))
    assert tuple(graph.edges(data=True)) == tuple(before.edges(data=True))
    return result


def _pressure_bins(pressure):
    """Independent exact unordered pairs for the unit path geometry."""
    coherence = tuple(1 / (1 + abs(F(p))) for p in pressure)
    bins = defaultdict(list)
    for i, first in enumerate(coherence):
        for j in range(i + 1, len(coherence)):
            bins[j - i].append(first * coherence[j])
    return {
        distance: (len(products), sum(products, F(0)) / len(products))
        for distance, products in bins.items()
    }


def test_p5_closed_geometry_has_two_unresolved_pressure_directions_at_zero_state():
    result = _observe(_path(amplitude=0))
    geometry = result.geometry
    realization = geometry.realization
    c, t = realization.observation, realization.right_inverse
    known = p5_reduction_geometry()
    assert c == known.orbit_projection == geometry.closure.projection
    assert geometry.closure.micro_generator == known.micro_generator
    assert geometry.closure.all_state_affine_closed
    assert realization.dimension == 3
    assert realization.reduced_state == (0, 0, 0)
    assert geometry.model_output == (0, 0, 0)

    pressure_rows = tuple(tuple(-a for a in row) for row in known.micro_generator)
    hidden = _subtract(_identity(5), _product(t, c))
    assert result.pressure_rows == pressure_rows
    assert result.pressure_decoder == _product(pressure_rows, t)
    assert result.pressure_hidden_residual == _product(pressure_rows, hidden)
    assert exact_rank(result.pressure_hidden_residual) == 2
    assert not result.pressure_observation_closed
    # A zero current residual is not an all-state closure certificate.
    assert result.decoded_model_pressure == result.unresolved_model_pressure == (0,) * 5
    witness = result.pressure_witness
    assert witness is not None
    assert _apply(c, witness.hidden_delta) == (0,) * 3
    assert _apply(pressure_rows, witness.hidden_delta) == witness.pressure_difference
    assert any(witness.pressure_difference)


def test_identical_p5_observed_states_have_different_successful_coherence_fits():
    first = _observe(_path(amplitude=1))
    second = _observe(_path(amplitude=2))
    assert (
        first.geometry.realization.observation
        == second.geometry.realization.observation
    )
    assert (
        first.geometry.realization.reduced_state
        == second.geometry.realization.reduced_state
    )
    assert (
        first.geometry.realization.output_state
        == second.geometry.realization.output_state
    )
    assert (
        first.geometry.realization.output_rate
        == second.geometry.realization.output_rate
    )
    assert first.geometry.model_output == second.geometry.model_output == (0,) * 3
    assert first.geometry.model_pressure == (1, 0, 0, 0, -1)
    assert second.geometry.model_pressure == (2, 0, 0, 0, -2)
    assert first.phase_observation == second.phase_observation
    assert first.mean_phase_gradient == first.mean_phase_curvature == (0,) * 3

    for result in (first, second):
        epi = result.geometry.fine_capture.snapshot.epi
        assert reduce_p5_state(epi).discarded_epi == epi
        assert result.decoded_model_pressure == (0,) * 5
        assert result.unresolved_model_pressure == result.geometry.model_pressure
        assert result.geometry.fine_capture.kernel_pressure_defect == (0,) * 5
        assert result.geometry.fine_capture.stored_pressure_residual == (0,) * 5
        fit = result.observed_coherence_length
        assert fit.method == "autocorrelation_fit" and fit.fit_available
        assert fit.sample_selection == "all unordered node pairs"
        assert "path-length units" in fit.distance_weighting

    first_bins = _pressure_bins(first.geometry.model_pressure)
    second_bins = _pressure_bins(second.geometry.model_pressure)
    assert first_bins == {
        1: (4, F(3, 4)),
        2: (3, F(2, 3)),
        3: (2, F(1, 2)),
        4: (1, F(1, 4)),
    }
    assert second_bins == {
        1: (4, F(2, 3)),
        2: (3, F(5, 9)),
        3: (2, F(1, 3)),
        4: (1, F(1, 9)),
    }
    assert sum(count for count, _ in first_bins.values()) == 10
    # Distance four is a singleton. On the three equally spaced retained
    # bins, the log-linear slope is (log(mean_3)-log(mean_1))/2.
    assert first.observed_coherence_length.value == pytest.approx(
        2 / log(1.5), rel=2e-14
    )
    assert second.observed_coherence_length.value == pytest.approx(
        2 / log(2), rel=2e-14
    )
    assert (
        first.observed_coherence_length.value != second.observed_coherence_length.value
    )

    reflected = _observe(_path(amplitude=-1))
    assert reflected.geometry.model_pressure != first.geometry.model_pressure
    assert reflected.observed_coherence_length == first.observed_coherence_length


def test_retaining_p5_model_pressure_uses_full_form_dimension_via_shared_realization():
    result = _observe(_path())
    base = result.geometry.realization
    rows = (*base.output_rows, *result.pressure_rows)
    offset = (*base.output_offset, *result.geometry.fine_capture.forcing)
    complete = observe_affine_nodal_realization(
        base.reference, rows, output_offset=offset
    )
    assert exact_rank((*base.observation, *result.pressure_rows)) == 5
    assert (
        complete.output_rank == complete.dimension == complete.full_state_dimension == 5
    )
    assert complete.extra_coordinates == 0
    c, t = complete.observation, complete.right_inverse
    assert _product(t, c) == _identity(5)
    decoder = _product(result.pressure_rows, t)
    assert _product(decoder, c) == result.pressure_rows
    assert _apply(decoder, complete.reduced_state) == result.geometry.model_pressure
    assert complete.output_state[:6] == base.output_state
    assert complete.output_state[6:] == result.geometry.model_pressure
    # Five coordinates here are a change of chart, not further compression.


def test_p3_pressure_completeness_is_not_necessary_for_its_fixed_fallback():
    zero = _observe(_path(3, amplitude=0), P3_BLOCKS)
    hidden = _observe(_path(3, amplitude=1), P3_BLOCKS)
    for result in (zero, hidden):
        assert result.geometry.realization.dimension == 2
        assert not result.pressure_observation_closed
        assert exact_rank(result.pressure_hidden_residual) == 1
        assert result.pressure_witness is not None
        bins = _pressure_bins(result.geometry.model_pressure)
        # This count is fixed for every pressure on P3, not only these inputs.
        assert sum(count for count, _ in bins.values()) == 3 < 10
        fit = result.observed_coherence_length
        assert fit.method == "spectral_gap" and not fit.fit_available
        assert fit.value == pytest.approx(1, rel=2e-14)
        assert "dimensionless" in fit.distance_weighting
    assert (
        zero.geometry.realization.reduced_state
        == hidden.geometry.realization.reduced_state
    )
    assert zero.geometry.model_pressure != hidden.geometry.model_pressure
    assert zero.observed_coherence_length == hidden.observed_coherence_length


def test_stored_pressure_changes_observed_fit_without_changing_the_held_model():
    fresh = _observe(_path(amplitude=1))
    stale = _observe(_path(amplitude=1, stored_amplitude=0))
    first, second = fresh.geometry.realization, stale.geometry.realization
    assert first.observation == second.observation
    assert first.reduced_generator == second.reduced_generator
    assert first.reduced_source == second.reduced_source
    assert first.reduced_state == second.reduced_state
    assert first.output_state == second.output_state
    assert first.output_rate == second.output_rate
    assert fresh.geometry.model_pressure == stale.geometry.model_pressure
    assert fresh.pressure_hidden_residual == stale.pressure_hidden_residual
    assert fresh.unresolved_model_pressure == stale.unresolved_model_pressure
    assert stale.geometry.fine_capture.stored_pressure_residual == (-1, 0, 0, 0, 1)
    assert fresh.observed_coherence_length.method == "autocorrelation_fit"
    assert stale.observed_coherence_length.method == "spectral_gap"
    assert not stale.observed_coherence_length.fit_available
    # P5 normalized-Laplacian lambda_2=1-cos(pi/4).
    assert stale.observed_coherence_length.value == pytest.approx(
        1 / sqrt(1 - sqrt(2) / 2), rel=2e-14
    )
    assert fresh.phase_observation == stale.phase_observation
    assert stale.phase_observation.primitive_phases == (0,) * 5
    assert all(row.curvature == 0 for row in stale.phase_observation.rows)


def test_asymmetric_p3_geometry_retains_all_model_pressure_without_a_witness():
    graph = _path(3)
    graph.edges[1, 2]["length"] = 2
    result = _observe(graph, P3_BLOCKS)
    state = result.geometry.realization
    assert state.dimension == state.full_state_dimension == 3
    assert result.pressure_observation_closed
    assert result.pressure_hidden_residual == ((0,) * 3,) * 3
    assert result.pressure_witness is None
    assert result.unresolved_model_pressure == (0,) * 3
    assert result.decoded_model_pressure == result.geometry.model_pressure == (1, 0, -1)
    assert _product(result.pressure_decoder, state.observation) == result.pressure_rows


def test_represented_zero_curvature_only_invalidates_its_containing_block_mean():
    graph = nx.star_graph(4)
    blocks = ((0,), (1, 2), (3,), (4,))
    for node, phase in enumerate((0.37, 0, 0, pi, -pi)):
        graph.nodes[node].update(EPI=node / 8, nu_f=1, theta=phase, delta_nfr=7)
    nx.set_edge_attributes(graph, 1, "weight")
    graph.graph["DNFR_WEIGHTS"] = dict.fromkeys(("phase", "epi", "vf", "topo"), 1)
    result = _observe(graph, blocks)
    rows = result.phase_observation.rows
    center = rows[0]
    # This is cancellation of the represented phasor components, not a
    # transcendental-zero proof or a strict-U3/autonomous phase fixture.
    assert center.node == 0 and center.status == "undefined_represented_resultant"
    assert center.resultant.joint_zero and center.curvature is None
    assert result.mean_phase_curvature[0] is None
    assert all(value is not None for value in result.mean_phase_curvature[1:])
    assert all(row.curvature is not None for row in rows[1:])
    assert all(isfinite(row.gradient) for row in rows)
    assert all(isfinite(value) for value in result.mean_phase_gradient)
    assert result.mean_phase_gradient == _apply(
        result.geometry.closure.projection,
        tuple(F.from_float(row.gradient) for row in rows),
    )
    fit = result.observed_coherence_length
    assert fit.method == "spectral_gap" and not fit.fit_available
    assert fit.value == pytest.approx(1, rel=2e-14)
