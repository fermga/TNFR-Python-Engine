"""Reuse the conservative pair barrier across supplied primary AL/SHA maps.

These static controls test reset algebra, a nonzero source neighborhood and
post-reset complete-law observations. They execute no operator word, install
no event selector and evolve no trajectory.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.operators.al_sha_stage_proposals import (
    emission_epi_proposal,
    silence_capacity_proposal,
)
from tnfr.physics.relational_sine_scale import assess_sine_replica_capacity

PAIRS = tuple((2 * i, 2 * i + 1) for i in range(5))
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
RADIUS, SOURCE_RADIUS = Q(1, 16), Q(1, 128)
SOURCE_EXCESS, BOOST = Q(1, 32768), Q(1, 1024)
FACTOR, COMMON_FORM = Q(1, 2), Q(1, 4)


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for pair_index, pair in enumerate(PAIRS):
        for sign, node in zip((1, -1), pair):
            graph.nodes[node].update(
                EPI=COMMON_FORM + sign * Q(1, 1024),
                theta=Q(1287 * pair_index, 1024) + sign * Q(1, 2048),
                nu_f=1,
                delta_nfr="unconsumed stale pressure",
            )
        graph.add_edges_from(
            (left, right) for left in pair for right in PAIRS[(pair_index + 1) % 5]
        )
    graph.graph.update(GAMMA={"type": "none"}, EPI_MIN=-1, EPI_MAX=1, CLIP_MODE="hard")
    return graph


def _assess(graph):
    return assess_sine_replica_capacity(
        graph,
        reference_model=MODEL,
        pairs=PAIRS,
        radius=RADIUS,
        excess_ceiling=Q(3, 32768),
        form_mean_bounds=(-1, 1),
    )


@pytest.fixture(scope="module")
def primary_reset_reference():
    source = _graph()
    emitted = source.copy()
    for node in PAIRS[0]:
        before, after = emission_epi_proposal(
            emitted.graph, source.nodes[node]["EPI"], float(BOOST)
        )
        assert Q(after) - Q(before) == BOOST
        emitted.nodes[node]["EPI"] = after
    silenced = emitted.copy()
    for node in PAIRS[0]:
        before, after = silence_capacity_proposal(
            emitted.nodes[node]["nu_f"], float(FACTOR)
        )
        assert Q(after) == FACTOR * Q(before) > 0
        silenced.nodes[node]["nu_f"] = after
    graphs = source, emitted, silenced
    return graphs, tuple(_assess(graph) for graph in graphs)


def test_pair_jump_constants_and_work_follow_from_independent_fine_edges():
    symbolic = pytest.importorskip("sympy")
    graph = _graph()
    basis = symbolic.eye(10)
    laplacian = symbolic.zeros(10)
    for left, right in graph.edges:
        incidence = basis[:, left] - basis[:, right]
        laplacian += incidence * incidence.T
    indicator = symbolic.Matrix([int(node in PAIRS[0]) for node in graph])
    centered = indicator - symbolic.ones(10, 1) / 5
    image = laplacian * indicator
    assert tuple(image) == (4, 4, -2, -2, 0, 0, 0, 0, -2, -2)
    assert (indicator.T * image)[0] == 8
    assert (image.T * image)[0] == 48
    assert (centered.T * centered)[0] == symbolic.Rational(8, 5)

    forms = symbolic.Matrix(symbolic.symbols("x0:10", real=True))
    boost = symbolic.Symbol("b", positive=True)
    jump = boost * indicator
    energy_jump = symbolic.expand(
        ((forms + jump).T * laplacian * (forms + jump))[0] / 2
        - (forms.T * laplacian * forms)[0] / 2
    )
    means = [(forms[left] + forms[right]) / 2 for left, right in PAIRS]
    expected = 4 * boost * (2 * means[0] - means[1] - means[4]) + 4 * boost**2
    assert symbolic.expand(energy_jump - expected) == 0
    # Cauchy--Schwarz uses ||L h||=sqrt(48)*b on the centered fine form.
    assert sum(image) == 0


def test_positive_source_neighborhood_fits_existing_post_reset_barrier(
    primary_reset_reference,
):
    _, (source, emitted, silenced) = primary_reset_reference
    assert source.family.norm_squared_upper_bound < SOURCE_RADIUS**2
    assert source.family.excess_storage_upper_bound < SOURCE_EXCESS
    # The common form coordinate is separately bounded: a relative norm alone
    # cannot guarantee that the actual AL proposal avoids boundary projection.
    mean_min, mean_max = Q(1, 8), Q(3, 8)
    assert mean_min < COMMON_FORM < mean_max
    assert mean_min - SOURCE_RADIUS > -1
    assert mean_max + SOURCE_RADIUS + BOOST < 1

    norm_bound = SOURCE_RADIUS + sqrt(I(Q(8, 5))) * BOOST
    excess_bound = SOURCE_EXCESS + 4 * sqrt(I(3)) * BOOST * SOURCE_RADIUS + 4 * BOOST**2
    assert norm_bound.hi < RADIUS
    assert excess_bound.hi < silenced.family.barrier_lower_bound
    assert excess_bound.hi < silenced.family.excess_ceiling
    # The source lies strictly inside the declared finite reference domain.
    # Its neighborhood conclusion uses the owner's analytic inequalities,
    # rather than sampling endpoints of a box or integrating a trajectory.
    for report in (emitted, silenced):
        family = report.family
        assert family.acute_radius_margin_bounds.lo > 0
        assert family.norm_squared_upper_bound < norm_bound.hi**2
        assert family.excess_storage_upper_bound < excess_bound.hi
        assert family.family_admitted and family.source_set_trapping_certified
        assert family.source_family_membership == "certified_inside"
        assert family.individual_recurrence_status == "unavailable_for_chosen_state"
        assert (
            report.all_time_internal_activity_status == "not_certified_by_this_reader"
        )


def test_capacity_reset_preserves_storage_but_reweights_the_inventory(
    primary_reset_reference,
):
    graphs, (source, emitted, silenced) = primary_reset_reference
    form_storage = tuple(
        sum(
            ((Q(graph.nodes[i]["EPI"]) - Q(graph.nodes[j]["EPI"])) ** 2 / 2)
            for i, j in graph.edges
        )
        for graph in graphs
    )
    assert form_storage[1] - form_storage[0] == 4 * BOOST**2
    assert form_storage[2] == form_storage[1]
    assert emitted.comparison.phase == silenced.comparison.phase
    assert emitted.comparison.storage == silenced.comparison.storage
    assert (emitted.comparison.storage - source.comparison.storage).contains(
        4 * BOOST**2
    )
    assert emitted.form_means == silenced.form_means
    assert emitted.internal_form_squared == silenced.internal_form_squared
    assert (
        emitted.form_phase_correlation_bounds == silenced.form_phase_correlation_bounds
    )
    assert silenced.pair_capacity_means == (FACTOR, 1, 1, 1, 1)
    assert silenced.pair_capacity_half_differences == (0,) * 5

    def inventory(graph):
        weights = tuple(
            Q(graph.degree[node]) / Q(graph.nodes[node]["nu_f"]) for node in graph
        )
        charge = sum(
            weight * Q(graph.nodes[node]["EPI"]) for node, weight in zip(graph, weights)
        )
        return weights, charge, sum(weights)

    old_weights, old_charge, old_weight = inventory(graphs[1])
    new_weights, new_charge, new_weight = inventory(graphs[2])
    old_mean, new_mean = old_charge / old_weight, new_charge / new_weight
    direct_reweighting = (
        sum(
            (new - old) * (Q(graphs[1].nodes[node]["EPI"]) - old_mean)
            for node, old, new in zip(graphs[1], old_weights, new_weights)
        )
        / new_weight
    )
    assert emitted.weighted_form_mean == old_mean == COMMON_FORM + BOOST / 5
    assert silenced.weighted_form_mean == new_mean == COMMON_FORM + BOOST / 3
    assert new_mean - old_mean == direct_reweighting == 2 * BOOST / 15
    assert new_charge - old_charge == 8 * (COMMON_FORM + BOOST)
    assert emitted.weighted_mean_rate_residual_bounds.contains(0)
    assert silenced.weighted_mean_rate_residual_bounds.contains(0)


def test_selected_pair_mobility_changes_both_rows_in_the_original_clock(
    primary_reset_reference,
):
    _, (_, emitted, silenced) = primary_reset_reference
    for index, (before, after) in enumerate(
        zip(
            emitted.capacity_aware_generated_rates,
            silenced.capacity_aware_generated_rates,
        )
    ):
        factor = FACTOR if index == 0 else 1
        for left, right in zip(before[:5], after[:5]):
            assert (right - factor * left).contains(0)
        for residual in silenced.capacity_aware_rate_residuals[index]:
            assert residual.contains(0)
    # With X_0 increased by b and all other centers unchanged, the fine
    # neighbor sums give pi*Theta'_0=b and pi*Theta'_1=-b/2 before SHA.
    # A changed nonzero row and an unchanged nonzero row exclude one global
    # clock factor. The actual row observations retain their finite bounds.
    before_selected = emitted.ordered_generated_rates[0][1]
    after_selected = silenced.ordered_generated_rates[0][1]
    before_neighbor = emitted.ordered_generated_rates[1][1]
    after_neighbor = silenced.ordered_generated_rates[1][1]
    assert (before_selected - BOOST / pi_interval()).contains(0)
    assert (after_selected - FACTOR * BOOST / pi_interval()).contains(0)
    assert (before_neighbor + BOOST / (2 * pi_interval())).contains(0)
    assert 0 < after_selected.hi < before_selected.lo
    assert before_neighbor.hi < 0
    assert before_neighbor == after_neighbor
