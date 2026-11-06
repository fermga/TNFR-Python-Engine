"""All-duration form obstruction for the unchanged organized C9 source.

Rebuild primitive source allowances once, retaining the acquired host, recovered
donor and full attached support. Exact graph, norm and scalar comparison
controls do not evaluate a trajectory or regenerate an earlier response.
Failure of the necessary child-mean form increment suffices to exclude the
full Emission endpoint; it does not assert a grammar or lifecycle execution.
"""

from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.constants import inject_defaults
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import cos
from tnfr.operators.al_sha_stage_proposals import emission_epi_proposal
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.physics import _organized_source_port as owner

ERROR = Q(1, 10**10)
CHILD_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))


@pytest.fixture(scope="module", autouse=True)
def no_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Source Emission controls cannot execute a native trajectory")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _assess(**changes):
    return owner._assess_organized_source_emission(
        **(
            dict(
                host_form_error_bounds=(ERROR,) * 4,
                host_phase_error_bounds=(ERROR,) * 4,
                donor_form_error_bound=ERROR,
                donor_phase_error_bound=ERROR,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def report():
    return _assess()


def _laplacian(graph):
    return tuple(
        tuple(
            Q(graph.degree(i) if i == j else -int(graph.has_edge(i, j))) for j in graph
        )
        for i in graph
    )


def _positive_pivots(matrix):
    """Exact symmetric elimination, with no fitted spectral tolerance."""
    remaining = [list(row) for row in matrix]
    pivots = []
    for i in range(len(matrix)):
        pivot = remaining[i][i]
        assert pivot > 0
        pivots.append(pivot)
        for j in range(i + 1, len(matrix)):
            for k in range(i + 1, len(matrix)):
                remaining[j][k] -= remaining[j][i] * remaining[i][k] / pivot
    return tuple(pivots)


def test_actual_support_is_the_full_product_with_a_certified_positive_gap(report):
    child = nx.Graph()
    child.add_nodes_from(range(5))
    child.add_edges_from(CHILD_EDGES)
    product = nx.relabel_nodes(
        nx.cartesian_product(child, nx.cycle_graph(9)), lambda n: 9 * n[0] + n[1]
    )
    geometry = report.source.geometry
    assert {frozenset((a, b)) for a, b, w in geometry.weighted_edges if w == 1} == {
        frozenset(edge) for edge in product.edges
    }
    assert len(geometry.weighted_edges) == product.number_of_edges() == 90
    assert geometry.node_order == tuple(range(45))
    degrees = tuple(product.degree(i) for i in range(45))
    assert degrees == (5,) * 9 + (4,) * 27 + (3,) * 9
    assert (
        sum(d * d for d in degrees)
        == report.sine_current_norm_squared_upper_bound
        == 738
    )
    assert 738 < report.sine_current_norm_upper_bound**2 == 28**2

    diameter = nx.diameter(child)
    gap = Q(4, len(child) * diameter)
    assert diameter == report.collective_diameter == 3
    assert gap == report.collective_gap_lower_bound == Q(4, 15)
    # Independent quadratic-form certificate on the zero-sum basis e_i-e_4.
    laplacian = _laplacian(child)
    restricted = tuple(
        tuple(
            laplacian[i][j]
            - laplacian[i][4]
            - laplacian[4][j]
            + laplacian[4][4]
            - gap * (1 + int(i == j))
            for j in range(4)
        )
        for i in range(4)
    )
    assert len(_positive_pivots(restricted)) == 4
    cycle_gap = (2 - 2 * cos(geometry.alpha_bounds)).lo
    assert cycle_gap > Q(1, 3) > gap
    assert gap <= report.cycle_gap_lower_bound <= cycle_gap
    assert report.form_laplacian_gap_lower_bound == gap

    # Test the Kronecker-sum identity on a nonseparable exact vector. This
    # retains all45 fine directions rather than substituting a carrier row.
    values = tuple(Q((17 * i) % 23 - 11, 128) for i in range(45))
    for c in range(5):
        for j in range(9):
            node = 9 * c + j
            full = sum(values[node] - values[n] for n in product.neighbors(node))
            separated = (
                2 * values[node]
                - values[9 * c + (j - 1) % 9]
                - values[9 * c + (j + 1) % 9]
            )
            separated += sum(
                values[node] - values[9 * neighbor + j]
                for neighbor in child.neighbors(c)
            )
            assert full == separated


def test_envelope_is_rebuilt_from_the_reached_source_and_both_error_budgets(report):
    source = report.source
    host, gamma = source.host, source.geometry.gamma_bounds
    inner = host.nominal_inner_gap_at_entry_bounds.hi
    outer = host.nominal_outer_gap_at_entry_bounds.hi
    eta_upper = (gamma**2).hi
    assert eta_upper < Q(1, 10**7)
    scale = gamma.hi * (1 + 2 * eta_upper)
    initial = (
        3 * scale * (inner + outer)
        + host.full_form_error_upper_bound
        + source.donor_recovered_norm_bound
    )
    assert (
        source.host_form_error_bounds == source.host_phase_error_bounds == (ERROR,) * 4
    )
    assert source.donor_form_error_bound == source.donor_phase_error_bound == ERROR
    assert source.host.acquisition_slow_time == 4
    assert source.donor_recovery_duration_tau == 2 * 10**13
    assert source.conserved_form_total == 0
    assert source.geometry.held_capacities == (Q(1),) * 45
    assert source.geometry.clock_factor == Q(1, 4)
    assert report.target_nodes == tuple(range(9))
    assert host.full_form_error_upper_bound > 0
    assert source.donor_recovered_norm_bound > 0
    assert initial == report.initial_form_norm_upper_bound < Q(1, 50000)
    assert initial > 3 * scale * (inner + outer)
    forced = 28 * gamma.hi / Q(4, 15)
    assert forced == report.forced_form_norm_upper_bound
    assert report.all_time_form_norm_upper_bound == max(initial, forced)
    assert report.form_decay_tau_lower_bound == Q(1, 15)
    assert (
        report.target_mean_displacement_upper_bound
        == (max(initial, forced) + initial) / 3
    )
    assert gamma.hi < Q(1, 3000)
    assert report.all_time_form_norm_upper_bound < Q(7, 200)
    assert report.target_mean_displacement_upper_bound < Q(1751, 150000) < Q(1, 80)
    assert report.source_certified and report.geometry_certified
    assert report.form_target_admitted and report.envelope_certified
    assert report.realization_excluded


@pytest.mark.parametrize("larger_initial", (False, True))
def test_duhamel_comparison_keeps_the_initial_term_and_the_quarter_clock(
    report, larger_initial
):
    forced = report.forced_form_norm_upper_bound
    initial = 2 * forced if larger_initial else report.initial_form_norm_upper_bound
    rate = report.form_laplacian_gap_lower_bound / 4
    forcing = 28 * report.source.geometry.gamma_bounds.hi / 4
    assert rate * forced == forcing
    values = []
    for decay in (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1)):
        value = decay * initial + (1 - decay) * forced
        derivative = -rate * decay * (initial - forced)
        assert derivative == -rate * value + forcing
        assert value <= max(initial, forced)
        values.append(value)
    assert values[0] == forced and values[-1] == initial
    assert max(values) == max(initial, forced)
    # The larger initial value tests sharpness of the scalar comparison
    # class, not a new organized source. No decay/trajectory is evaluated.
    if larger_initial:
        assert max(values) > forced


def test_nine_slot_mean_requires_the_operator_increment_not_a_peak(report):
    values = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    # Independent exact variance identity proves the relevant 1/sqrt(9)
    # observation factor. A large individual slot need not shift this mean.
    target = values[:9]
    variance = sum(value**2 for value in target) - sum(target) ** 2 / 9
    pairwise = sum((target[i] - target[j]) ** 2 for i in range(9) for j in range(i)) / 9
    assert variance == pairwise >= 0
    assert (sum(target) / 9) ** 2 <= sum(value**2 for value in values) / 9
    cancellation = (Q(1), Q(-1)) + (Q(0),) * 7
    assert max(cancellation) > report.emission_boost and sum(cancellation) == 0
    shifted = tuple(value + report.emission_boost for value in target)
    assert (sum(shifted) - sum(target)) / 9 == report.emission_boost
    assert report.target_mean_displacement_upper_bound < report.emission_boost


def test_actual_default_emission_has_a_nonclipped_target_on_the_whole_source_domain(
    report,
):
    graph = nx.Graph()
    inject_defaults(graph)
    factors = resolve_runtime_operator_factors(
        graph.graph["GLYPH_FACTORS"], "AL", graph.graph
    )
    boost = Q.from_float(factors["AL_boost"])
    radius = report.initial_form_norm_upper_bound
    assert report.emission_boost == boost > Q(1, 16)
    assert Q(graph.graph["EPI_MIN"]) < -radius
    assert radius + boost < Q(graph.graph["EPI_MAX"])
    for value in (-radius, Q(0), radius):
        represented = float(value)
        before, after = emission_epi_proposal(graph.graph, represented, float(boost))
        assert before == represented
        assert after == before + float(boost)
    assert (
        report.obstruction_margin
        == boost - report.target_mean_displacement_upper_bound
        > 0
    )
    # This scalar contract check is not the supplied source flow, a target
    # acquisition or an operator occurrence at the acquired organization.


@pytest.mark.parametrize(
    "field",
    (
        "host_form_error_bounds",
        "host_phase_error_bounds",
        "donor_form_error_bound",
        "donor_phase_error_bound",
    ),
)
def test_new_observer_admits_all_primitives_before_rebuilding_sources(
    field, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("invalid raw allowance reached expensive source rebuilding")

    monkeypatch.setattr(owner, "_assess_organized_source_attachment", forbidden)
    value = (ERROR, ERROR, ERROR, True) if field.startswith("host_") else True
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: value})


def test_unavailable_source_suppresses_actual_realization_flags(report, monkeypatch):
    unavailable = replace(report.source, admitted=False, sources_certified=False)
    monkeypatch.setattr(
        owner, "_assess_organized_source_attachment", lambda **kwargs: unavailable
    )
    reading = _assess()
    assert reading.geometry_certified
    assert not reading.source_certified
    assert not reading.form_target_admitted
    assert not reading.envelope_certified and not reading.realization_excluded
    assert reading.initial_form_norm_upper_bound == report.initial_form_norm_upper_bound
    assert reading.unresolved_conditions


def test_nonzero_conserved_form_cannot_inherit_the_mean_zero_envelope(
    report, monkeypatch
):
    changed = replace(report.source, conserved_form_total=Q(1))
    assert changed.admitted and changed.sources_certified
    monkeypatch.setattr(
        owner, "_assess_organized_source_attachment", lambda **kwargs: changed
    )
    reading = _assess()
    assert reading.geometry_certified
    assert not reading.source_certified
    assert not reading.envelope_certified and not reading.realization_excluded
    assert reading.unresolved_conditions


def test_cached_contact_summaries_cannot_replace_the_reached_source_envelope(
    report, monkeypatch
):
    changed = replace(
        report.source,
        contact_form_gap_absolute_upper_bound=Q(0),
        host_fine_form_error_upper_bound=Q(0),
    )
    monkeypatch.setattr(
        owner, "_assess_organized_source_attachment", lambda **kwargs: changed
    )
    reading = _assess()
    assert reading.initial_form_norm_upper_bound == report.initial_form_norm_upper_bound
    assert (
        reading.target_mean_displacement_upper_bound
        == report.target_mean_displacement_upper_bound
    )
    assert reading.obstruction_margin == report.obstruction_margin


def test_changed_actual_support_cannot_inherit_the_cached_product_certificate(
    report, monkeypatch
):
    geometry = replace(
        report.source.geometry,
        weighted_edges=report.source.geometry.weighted_edges[:-1],
    )
    changed = replace(report.source, geometry=geometry)
    monkeypatch.setattr(
        owner, "_assess_organized_source_attachment", lambda **kwargs: changed
    )
    reading = _assess()
    assert not reading.geometry_certified
    assert not reading.envelope_certified and not reading.realization_excluded
    assert reading.unresolved_conditions


@pytest.mark.parametrize("extra", ("source", "horizon", "emission_boost", "clock"))
def test_reports_or_changed_target_parameters_cannot_replace_the_frozen_protocol(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})
