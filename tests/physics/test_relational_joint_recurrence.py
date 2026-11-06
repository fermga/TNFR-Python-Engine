"""Join the local identity theorem to its actual ambient recurrence owner.

This checks one analytical family admission, not a recurrence trajectory or a
numerical lifetime. The exact critical source is never made a graph attribute.
"""

from fractions import Fraction as Q

from tests.physics.test_relational_sine_replica import (
    MODEL,
    _frozen_grouping_transition,
)
from tnfr.mathematics._phase_resultant_chamber import certified_cosine_bounds
from tnfr.mathematics._rational_interval import I, sqrt
from tnfr.physics.relational_sine_resonance import assess_sine_recurrence


def test_joint_boundary_neighborhood_uses_same_ambient_family_not_point_recurrence():
    # The old rational preparation is only a represented API witness. It is
    # neither the exact critical preparation nor the unknown crossing orbit.
    graph = _frozen_grouping_transition()
    nodes, edges = tuple(graph), tuple(graph.edges())
    assert len(nodes) == 10 and len(edges) == 20
    assert all(graph.degree[node] == 4 for node in nodes)
    assert MODEL.effective_weights == (0, 1)
    assert MODEL.storage_scale == 1

    # This bound covers every signed form in the full box and every circular
    # phase, rather than selected corners or a putative numerical orbit.
    form_radius = Q(1, 4)
    edge_form_bound = (2 * form_radius) ** 2 / 2
    form_storage_bound = len(edges) * edge_form_bound
    phase_storage_bound = 2 * len(edges)
    energy_ceiling = Q(43)
    mean_bounds = (Q(-1, 2), Q(1, 2))
    assert form_storage_bound == Q(5, 2)
    assert form_storage_bound + phase_storage_bound == Q(85, 2) < energy_ceiling
    assert mean_bounds[0] < -form_radius < form_radius < mean_bounds[1]

    # Both exact critical orientations lie strictly inside that form box.
    # Continuity, not this snapshot reader, supplies a small pre-crossing U.
    d = Q(1, 8)
    cosine_d = I(*certified_cosine_bounds(d))
    cosine_2d = I(*certified_cosine_bounds(2 * d))
    critical_amplitude = sqrt((cosine_d - cosine_2d) / 2)
    assert 0 < critical_amplitude.lo < critical_amplitude.hi < d < form_radius

    report = assess_sine_recurrence(
        graph,
        reference_model=MODEL,
        energy_ceiling=energy_ceiling,
        form_mean_bounds=mean_bounds,
    )
    captured = report.comparison
    assert captured.nodes == nodes and captured.edges == edges
    assert captured.capacity == (Q(1),) * len(nodes)
    assert report.form_mean_weights == (Q(4),) * len(nodes)
    assert report.normalized_mean_weights == (Q(1, 10),) * len(nodes)
    assert all(abs(value) < form_radius for value in captured.epi)
    expected_form_storage = sum(
        (captured.epi[i] - captured.epi[j]) ** 2 / 2 for i, j in edges
    )
    assert captured.form_storage == expected_form_storage <= form_storage_bound
    assert captured.storage.hi < energy_ceiling
    assert report.weighted_form_mean == 0
    assert report.divergence == 0
    assert report.snapshot_family_membership == "inside"
    assert report.snapshot_motion_status == "nonstationary"
    assert report.almost_everywhere_recurrence_certified
    assert report.finite_positive_family_volume_certified
    assert report.individual_recurrence_status == "unavailable_for_chosen_state"
