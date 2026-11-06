"""Independent exact consensus-tangent observations of the complete sine law."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_consensus_observation import (
    assess_sine_consensus_observation,
)
from tnfr.physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

REGIONS = ((0, 1), (2, 3))


def _product(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _apply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _identity(size):
    return tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))


def _independent_tangent(source, regions):
    """Direct edge sums with no cached degree, field or observation inputs."""
    size = len(source.nodes)
    positions = {node: i for i, node in enumerate(source.nodes)}
    degrees = [0] * size
    bare = [[Q(0)] * size for _ in range(size)]
    weighted = [[Q(0)] * size for _ in range(size)]
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = positions[left], positions[right]
        for a, b in ((i, j), (j, i)):
            degrees[a] += 1
            bare[a][a] += 1
            bare[a][b] -= 1
            weighted[a][a] += weight
            weighted[a][b] -= weight
    mu = tuple(
        Q(degree) / capacity for degree, capacity in zip(degrees, source.capacity)
    )
    masses = tuple(sum(mu[positions[node]] for node in block) for block in regions)
    coefficients = tuple(
        mu[i] / masses[0] if node in regions[0] else -mu[i] / masses[1]
        for i, node in enumerate(source.nodes)
    )
    zero = (Q(0),) * size
    beta = Q(source.reference_model.storage_scale)
    generator = tuple(
        zero + tuple(-value / mass for value in row) for row, mass in zip(bare, mu)
    ) + tuple(
        tuple(value / (beta * mass) for value in row) + zero
        for row, mass in zip(weighted, mu)
    )
    output = (coefficients + zero, zero + coefficients)
    return (
        generator,
        output,
        mu,
        masses,
        tuple(map(tuple, bare)),
        tuple(map(tuple, weighted)),
    )


def _assert_exact_realization(result, generator, output):
    """Check closure/decoding and each declared derivative coordinate directly."""
    c, t, g, d = (
        result.observation,
        result.right_inverse,
        result.reduced_generator,
        result.output_map,
    )
    assert result.generator == generator and result.output_rows == output
    assert _product(c, t) == _identity(result.dimension)
    assert _product(c, generator) == _product(g, c)
    assert _product(d, c) == output
    assert result.dimension == len(c)
    assert result.output_rank == 2
    assert result.extra_coordinates == result.dimension - 2
    assert result.full_state_dimension == len(generator)
    assert result.rank_calls <= result.max_rank_calls
    assert result.level_records[-1].rank_before == result.level_records[-1].rank_after
    level = output
    selected = []
    for power, record in enumerate(result.level_records):
        assert record.power == power and record.candidate_rows == 2
        selected.extend(level[i] for i in record.selected_row_indices)
        level = _product(level, generator)
    assert tuple(selected) == c
    mixed = tuple(Q((-1) ** i * (i + 1), i + 2) for i in range(len(generator)))
    observed = _apply(c, mixed)
    assert _apply(d, observed) == _apply(output, mixed)
    assert _apply(g, observed) == _apply(c, _apply(generator, mixed))


def _source(*, disconnected=False, zero_link=False, capacities=(1, 2, 3, 4)):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, capacity in enumerate(capacities):
        graph.add_node(
            node, EPI=(node - 1) / 8, theta=(2 * node - 1) / 16, nu_f=float(capacity)
        )
    graph.add_edge(0, 1, weight=2)
    graph.add_edge(2, 3, weight=3)
    if not disconnected:
        graph.add_edge(1, 2, weight=0 if zero_link else 1)
    model = RelationalExchangeModel(2, epi_weight=0, phase_domain="regular")
    reader = (
        bound_relational_sine_frozen_isolates
        if disconnected
        else bound_relational_sine_support_exchange
    )
    return reader(graph, reference_model=model)


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("consensus observation permits no trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def report(source):
    return assess_sine_consensus_observation(
        source, regions=REGIONS, max_rank_calls=128
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def test_direct_full_edge_tangent_and_original_time_multiplier(source, report, mp):
    generator, output, mu, masses, bare, weighted = _independent_tangent(
        source, REGIONS
    )
    assert report.rational_generator == generator
    assert report.output_rows == output
    assert report.inverse_capacity_weights == mu == (1, 1, Q(2, 3), Q(1, 4))
    assert report.region_mean_weights == masses == (2, Q(11, 12))
    assert report.relative_mean_row == (Q(1, 2), Q(1, 2), -Q(8, 11), -Q(3, 11))
    assert report.bare_laplacian == bare and report.weighted_laplacian == weighted
    assert report.mobility == tuple(1 / value for value in mu)
    assert report.exchange_coefficient == 1 and report.storage_coefficient == 2
    assert report.clock == "t"
    assert report.generator_rule == "J=(exchange_coefficient/pi)*rational_generator"
    multiplier = report.original_time_multiplier_bounds
    assert _mp(mp, multiplier.lo) <= 1 / mp.pi <= _mp(mp, multiplier.hi)
    # Every basis direction is independently differentiated at consensus.
    for j in range(8):
        x = tuple(Q(i == j) for i in range(4))
        theta = tuple(Q(i + 4 == j) for i in range(4))
        dx, dtheta = [Q(0)] * 4, [Q(0)] * 4
        for (i, k), weight in zip(source.edges, source.conductances):
            dx[i] += (theta[k] - theta[i]) / mu[i]
            dx[k] += (theta[i] - theta[k]) / mu[k]
            dtheta[i] += weight * (x[i] - x[k]) / (2 * mu[i])
            dtheta[k] += weight * (x[k] - x[i]) / (2 * mu[k])
        assert tuple(row[j] for row in generator) == tuple(dx + dtheta)
    _assert_exact_realization(report.linear_observation, generator, output)
    assert not report.source_in_consensus_family
    assert report.source.epi == source.epi and report.source.phase == source.phase
    with pytest.raises(FrozenInstanceError):
        report.regions = tuple(reversed(REGIONS))


def test_nonunit_exchange_changes_every_physical_row_but_not_rational_minimal_state(
    source, report, mp
):
    model = copy(source.reference_model)
    object.__setattr__(model, "phase_weight", Q(3, 2))
    scaled = assess_sine_consensus_observation(
        replace(source, reference_model=model), regions=REGIONS
    )
    assert scaled.rational_generator == report.rational_generator
    assert scaled.output_rows == report.output_rows
    assert (
        scaled.linear_observation.observation == report.linear_observation.observation
    )
    assert (
        scaled.linear_observation.reduced_generator
        == report.linear_observation.reduced_generator
    )
    assert scaled.exchange_coefficient == Q(3, 2)
    interval = scaled.original_time_multiplier_bounds
    assert _mp(mp, interval.lo) <= 3 / (2 * mp.pi) <= _mp(mp, interval.hi)
    assert source.reference_model.phase_weight == 1


def test_conserved_global_means_and_common_origins_remain_distinct_from_relative_output(
    report,
):
    generator, output, mu, *_ = _independent_tangent(report.source, REGIONS)
    zero = (Q(0),) * 4
    constants = ((Q(1),) * 4 + zero, zero + (Q(1),) * 4)
    conserved = (mu + zero, zero + mu)
    assert _product(conserved, generator) == ((0,) * 8,) * 2
    for direction in constants:
        assert _apply(generator, direction) == (0,) * 8
        assert _apply(output, direction) == (0, 0)
        assert (
            _apply(report.linear_observation.observation, direction)
            == (0,) * report.linear_observation.dimension
        )
    # An internal left phase perturbation has the same two readings as zero
    # and zero total phase mean, but a distinct relative-form derivative.
    hidden = zero + (Q(1), Q(-1), Q(0), Q(0))
    assert _apply(output, hidden) == (0, 0)
    derivative = _apply(output, _apply(generator, hidden))
    assert derivative == (Q(35, 22), Q(0))
    assert not report.two_coordinate_closure_certified
    assert report.linear_observation.dimension > 2


def test_disconnected_control_uses_own_masses_and_has_constant_relative_means():
    source = _source(disconnected=True)
    report = assess_sine_consensus_observation(source, regions=REGIONS)
    generator, output, mu, masses, *_ = _independent_tangent(source, REGIONS)
    assert mu == (1, Q(1, 2), Q(1, 3), Q(1, 4))
    assert masses == (Q(3, 2), Q(7, 12))
    assert report.source.component_indices == ((0, 1), (2, 3))
    assert _product(output, generator) == ((0,) * 8,) * 2
    assert report.two_coordinate_closure_certified
    assert report.linear_observation.dimension == 2
    assert report.linear_observation.reduced_generator == ((0, 0), (0, 0))
    assert report.linear_observation.observation == output
    _assert_exact_realization(report.linear_observation, generator, output)


def test_zero_weight_supported_link_is_not_a_disconnected_control():
    source = _source(zero_link=True)
    report = assess_sine_consensus_observation(source, regions=REGIONS)
    disconnected = assess_sine_consensus_observation(
        _source(disconnected=True), regions=REGIONS
    )
    assert report.weighted_laplacian == disconnected.weighted_laplacian
    assert report.bare_laplacian != disconnected.bare_laplacian
    assert report.inverse_capacity_weights != disconnected.inverse_capacity_weights
    assert not report.two_coordinate_closure_certified
    assert (
        report.linear_observation.dimension > disconnected.linear_observation.dimension
    )


def test_two_node_complete_relative_pair_is_exactly_closed_without_oscillator_fitting():
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    graph.add_node("left", EPI=0.25, theta=0.125, nu_f=2)
    graph.add_node("right", EPI=-0.25, theta=-0.125, nu_f=5)
    graph.add_edge("left", "right", weight=3)
    source = bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_domain="regular"
        ),
    )
    report = assess_sine_consensus_observation(source, regions=(("left",), ("right",)))
    assert report.two_coordinate_closure_certified
    assert report.linear_observation.dimension == 2
    assert report.linear_observation.reduced_generator == ((0, -7), (Q(21, 2), 0))
    assert report.linear_observation.observation == ((1, -1, 0, 0), (0, 0, 1, -1))
    zero_weight = assess_sine_consensus_observation(
        replace(source, conductances=(Q(0),)), regions=(("left",), ("right",))
    )
    assert zero_weight.linear_observation.reduced_generator == ((0, -7), (0, 0))
    assert zero_weight.two_coordinate_closure_certified


def test_equilibrium_membership_is_separate_from_held_law_admission(source, report):
    equilibrium = replace(source, epi=(Q(7, 16),) * 4, phase=(Q(-3, 8),) * 4)
    current = assess_sine_consensus_observation(equilibrium, regions=REGIONS)
    assert current.source_in_consensus_family
    assert current.rational_generator == report.rational_generator
    assert current.output_rows == report.output_rows
    assert current.source.epi == equilibrium.epi
    assert source.epi != equilibrium.epi
    lifted = replace(
        equilibrium,
        phase=(Q(-3, 8),) * 3 + (Q(-3, 8) + Q.from_float(6.283185307179586),),
    )
    assert not assess_sine_consensus_observation(
        lifted, regions=REGIONS
    ).source_in_consensus_family


@pytest.mark.parametrize(
    "regions",
    (
        ((0, 1, 2, 3),),
        ((0,), (1,), (2, 3)),
        ((), (0, 1, 2, 3)),
        ((0, 1), (1, 2, 3)),
        ((0, 0), (1, 2, 3)),
        ((0, 1), (2,)),
        ((0, 1), (2, 99)),
        {0, 1, 2, 3},
    ),
)
def test_invalid_region_partition_rejects(source, regions):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_consensus_observation(source, regions=regions)


@pytest.mark.parametrize("budget", (True, 0, -1, Q(128), 1.5))
def test_invalid_rank_budget_rejects_without_fallback(source, budget):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_consensus_observation(
            source, regions=REGIONS, max_rank_calls=budget
        )


def test_exhausted_rank_budget_is_unresolved_not_a_partial_realization(source, report):
    with pytest.raises(ValueError, match="budget"):
        assess_sine_consensus_observation(
            source,
            regions=REGIONS,
            max_rank_calls=report.linear_observation.rank_calls - 1,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", 0),
        ("capacity", True),
        ("phase", float("nan")),
        ("epi", float("inf")),
        ("conductances", -1),
    ),
)
def test_invalid_source_primitives_cannot_be_hidden_by_cached_rows(
    source, field, value
):
    forged = replace(
        source,
        **{field: (value,) + getattr(source, field)[1:]},
        storage=I(0),
        form_rates=(I(0),) * 4,
    )
    with pytest.raises((TypeError, ValueError)):
        assess_sine_consensus_observation(forged, regions=REGIONS)


def test_actual_isolate_has_no_inverse_mobility_mean_even_under_frozen_extension():
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node in range(4):
        graph.add_node(node, EPI=0, theta=0, nu_f=1)
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(1, 2, weight=1)
    source = bound_relational_sine_frozen_isolates(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    with pytest.raises(ValueError):
        assess_sine_consensus_observation(source, regions=REGIONS)


def test_fresh_sdk_export_rebuilds_source_generator_and_minimality_claims(
    source, report, tmp_path
):
    poisoned_source = replace(
        source,
        degrees=(99,) * 4,
        form_gradient=(Q(0),) * 4,
        pressure=(I(0),) * 4,
        form_rates=(I(0),) * 4,
        storage=I(0),
    )
    poisoned = replace(
        report,
        source=poisoned_source,
        rational_generator=((Q(0),) * 8,) * 8,
        relative_mean_row=(Q(0),) * 4,
        two_coordinate_closure_certified=True,
        source_in_consensus_family=True,
        linear_observation=replace(
            report.linear_observation, dimension=2, reduced_generator=((Q(0),) * 2,) * 2
        ),
    )
    expected = relational_report_to_dict(report)
    assert relational_report_to_dict(poisoned) == expected
    path = tmp_path / "consensus-observation.json"
    export_to_json(relational_report_to_dict(poisoned), path)
    assert json_loads(path.read_text(encoding="utf-8")) == expected
    before = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(
                replace(poisoned, source=replace(source, capacity=(True, 2, 3, 4)))
            ),
            path,
        )
    assert path.read_bytes() == before


def test_untrusted_source_cannot_dispatch_custom_observation(source, report):
    class Fake:
        def to_dict(self):
            raise AssertionError("unadmitted dispatch")

    with pytest.raises(TypeError):
        assess_sine_consensus_observation(Fake(), regions=REGIONS)
    with pytest.raises(TypeError):
        replace(report, source=Fake()).to_dict()
