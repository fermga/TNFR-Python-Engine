"""Independent weighted closure, represented capacity maps and admission controls."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_phase_offset_partition
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_weighted_partition import (
    assess_sine_weighted_capacity_reset,
    assess_sine_weighted_phase_offset_partition,
)
from tnfr.types import Glyph

STAR_EDGES = ((0, 1), (0, 2), (0, 3)) + tuple((i, 4) for i in range(4))
STAR_WEIGHTS = (3,) * 3 + (2,) * 4
STAR_CAPACITIES = (4, 2, 2, 2, 8)
STAR_BLOCKS = ((0, 1, 2, 3), (4,))


def _graph(edges, *, capacities, weights=None):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(capacities)))
    if weights is None:
        weights = (1,) * len(edges)
    for edge, weight in zip(edges, weights):
        graph.add_edge(*edge, weight=weight)
    for node, capacity in enumerate(capacities):
        # A partition certificate describes a family; this preparation need
        # not itself have the blockwise common forms of that family.
        graph.nodes[node].update(EPI=node - 3, theta=0, nu_f=capacity)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _model():
    return RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )


def _source(edges, *, capacities, weights=None):
    return bound_relational_sine_support_exchange(
        _graph(edges, capacities=capacities, weights=weights),
        reference_model=_model(),
    )


def _assess(source, *, blocks=STAR_BLOCKS, offsets=(0,) * 5):
    return assess_sine_weighted_phase_offset_partition(
        source, blocks=blocks, phase_offset_turns=offsets
    )


@pytest.fixture(scope="module")
def source():
    return _source(STAR_EDGES, capacities=STAR_CAPACITIES, weights=STAR_WEIGHTS)


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    exact_decimal = Q(mp.nstr(value, 160))
    assert interval.lo <= exact_decimal <= interval.hi


def _direct_rows_and_storage(report, forms, turns):
    """Independent scalar edge sums, without the quotient or symbolic kernel."""
    source = report.source
    node_index = {node: i for i, node in enumerate(source.nodes)}
    x = [Q(forms[block]) for block in report.node_blocks]
    theta = [
        Q(turns[block]) + offset
        for block, offset in zip(report.node_blocks, report.phase_offset_turns)
    ]
    pressure_sums = [mp.mpf(0) for _ in x]
    gradients = [Q(0) for _ in x]
    storage = mp.mpf(0)
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = node_index[left], node_index[right]
        phase = 2 * mp.pi * _mp(theta[j] - theta[i])
        sine = mp.sin(phase)
        pressure_sums[i] += sine
        pressure_sums[j] -= sine
        gradient = weight * (x[i] - x[j])
        gradients[i] += gradient
        gradients[j] -= gradient
        storage += _mp(weight * (x[i] - x[j]) ** 2 / 2) + 1 - mp.cos(phase)
    mobility = [
        Q(capacity, degree) for capacity, degree in zip(source.capacity, source.degrees)
    ]
    form_rates = [_mp(a) * p / mp.pi for a, p in zip(mobility, pressure_sums)]
    phase_numerators = [a * g for a, g in zip(mobility, gradients)]
    phase_rates = [_mp(value) / mp.pi for value in phase_numerators]
    return form_rates, phase_rates, phase_numerators, storage


def test_positive_weighted_star_has_nonuniform_capacity_inside_one_block(report):
    assert report.invariance_certified and report.status == "certified"
    assert report.weighted_compatibility_certified
    assert report.moment_compatibility_certified
    assert report.internal_balance_certified
    assert report.reasons == ()
    assert report.source.capacity == STAR_CAPACITIES
    assert report.source.degrees == (4, 2, 2, 2, 4)
    assert report.mobility == (1, 1, 1, 1, 2)
    assert report.inverse_capacity_weights == (1, 1, 1, 1, Q(1, 2))
    assert report.block_mean_weights == (4, Q(1, 2))
    assert report.weighted_rows == ((9, 2), (3, 2), (3, 2), (3, 2), (16, 0))
    assert report.internal_current_expressions == ((),) * 5
    assert report.cross_bare_counts == ((0, 4), (4, 0))
    assert report.cross_conductance_sums == ((0, 8), (8, 0))
    assert report.cross_cosine_expressions == (
        ((), ((Q(1, 4), Q(4)),)),
        (((Q(1, 4), Q(4)),), ()),
    )
    assert report.cross_sine_expressions == (((), ()), ((), ()))
    for a, b in ((0, 1), (1, 0)):
        assert (
            report.block_mean_weights[a] * report.quotient_weighted_rows[a][b]
            == report.cross_conductance_sums[a][b]
        )
    with pytest.raises(FrozenInstanceError):
        report.status = "excluded"


def test_full_equations_storage_and_means_match_independent_edge_sums(report):
    forms, turns = (Q(3, 8), Q(1, 4)), (Q(1, 16), Q(3, 16))
    state = report.evaluate(forms, turns)
    assert state.full_row_equality_certified
    assert state.full_phase_rate_numerators == (Q(1, 4),) * 4 + (-2,)
    assert state.block_phase_rate_numerators == (Q(1, 4), -2)
    assert state.mean_weights == (1, 1, 1, 1, Q(1, 2))
    assert state.form_mean == Q(13, 36)
    assert state.phase_mean_turns == Q(11, 144)
    assert state.phase_mean_rate_numerator == 0
    assert state.internal_form_storage == (0, 0)
    assert state.internal_phase_storage_bounds == (I(0), I(0))
    assert state.cross_form_storage == state.full_form_storage == Q(1, 16)
    assert state.quotient_form_storage == Q(1, 16)
    assert state.clock == "t"
    with mp.workdps(180):
        dx, dtheta, numerators, storage = _direct_rows_and_storage(report, forms, turns)
        assert tuple(numerators) == state.full_phase_rate_numerators
        for i, block in enumerate(report.node_blocks):
            _contains(state.full_form_rates[i], dx[i])
            _contains(state.block_form_rates[block], dx[i])
            _contains(state.full_phase_rates[i], dtheta[i])
            _contains(state.block_phase_rates[block], dtheta[i])
        _contains(state.full_storage_bounds, storage)
        _contains(state.full_storage_bounds, mp.mpf(1) / 16 + 4 - 2 * mp.sqrt(2))
        _contains(state.quotient_phase_storage_bounds, 4 - 2 * mp.sqrt(2))
    assert state.form_mean_rate_bounds == I(0)
    assert state.storage_residual_bounds.lo <= 0 <= state.storage_residual_bounds.hi


def test_bare_phase_moment_is_not_bounded_by_the_conductance_row():
    # The phase channel sees four bare edges even when their weights are <1.
    report = _assess(
        _source(
            STAR_EDGES,
            capacities=STAR_CAPACITIES,
            weights=(3,) * 3 + (Q(1, 8),) * 4,
        )
    )
    assert report.invariance_certified
    assert report.quotient_weighted_rows[0][1] == Q(1, 8)
    assert report.row_cosine_bounds[0][1] == I(1)
    assert report.cross_bare_counts[0][1] == 4
    assert report.cross_conductance_sums[0][1] == Q(1, 2)
    state = report.evaluate((1, 0), (0, Q(1, 4)))
    assert state.block_phase_rate_numerators == (Q(1, 8), -1)
    assert state.full_form_storage == Q(1, 4)
    assert state.full_phase_storage_bounds == I(4)


def test_represented_default_sha_on_whole_block_preserves_family(report):
    factor = resolve_runtime_operator_factors({}, Glyph.SHA, {})["SHA_vf_factor"]
    exact_factor = Q.from_float(factor)
    after = tuple(float(value) * factor for value in STAR_CAPACITIES[:4]) + (8.0,)
    reset = assess_sine_weighted_capacity_reset(report, capacity_after=after)
    assert reset.preservation_certified
    assert reset.blockwise_scaling_certified
    assert reset.block_ratios == (exact_factor, 1)
    assert reset.capacity_ratios == (exact_factor,) * 4 + (1,)
    assert reset.after.block_mean_weights == (4 / exact_factor, Q(1, 2))
    assert reset.common_clock_scale is None
    assert reset.storage_jump == 0
    original = report.evaluate((1, 0), (0, Q(1, 4)))
    changed = reset.after.evaluate((1, 0), (0, Q(1, 4)))
    assert changed.full_storage_bounds == original.full_storage_bounds
    assert changed.block_phase_rate_numerators == (2 * exact_factor, -16)
    assert changed.fine_form == original.fine_form
    assert changed.fine_phase_turns == original.fine_phase_turns
    # This is a primary-state map only: no operator, grammar or history is run.
    assert report.source.capacity == STAR_CAPACITIES


def test_singleton_sha_breaks_the_vector_field_on_an_open_set(report):
    factor = resolve_runtime_operator_factors({}, Glyph.SHA, {})["SHA_vf_factor"]
    after = list(map(float, STAR_CAPACITIES))
    after[1] *= factor
    reset = report.capacity_reset(after)
    assert not reset.preservation_certified
    assert not reset.blockwise_scaling_certified
    assert reset.block_ratios is None
    assert reset.after.status == "excluded"
    assert not reset.after.weighted_compatibility_certified
    assert not reset.after.moment_compatibility_certified
    # At X_A-X_B=1 and DeltaTheta=pi/2 the two residuals are nonzero.
    # Their continuous dependence also excludes an open neighborhood, despite
    # the capacity-only reset leaving the instantaneous x/theta family intact.
    ratio = Q.from_float(factor)
    assert reset.after.weighted_rows[1][1] - reset.after.weighted_rows[0][1] == 2 * (
        ratio - 1
    )
    assert reset.after.row_cosine_bounds[1][1] - reset.after.row_cosine_bounds[0][
        1
    ] == I(ratio - 1)
    with pytest.raises(ValueError, match="certified"):
        reset.after.evaluate((1, 0), (0, Q(1, 4)))


def test_uniform_represented_capacity_map_has_one_clock_scale(report):
    reset = report.capacity_reset(tuple(Q(value, 2) for value in STAR_CAPACITIES))
    assert reset.preservation_certified
    assert reset.common_clock_scale == Q(1, 2)
    assert reset.block_ratios == (Q(1, 2), Q(1, 2))
    assert reset.after.block_mean_weights == (8, 1)


def test_common_ideal_factor_can_lose_closure_after_separate_binary64_products():
    # Degrees 2,3,2 require capacity ratios that are not powers of two.
    source = _source(((0, 1), (1, 2), (0, 3), (1, 3), (2, 3)), capacities=(2, 3, 2, 3))
    report = _assess(source, blocks=((0, 1, 2), (3,)), offsets=(0,) * 4)
    assert report.invariance_certified
    factor = 0.1
    after = tuple(float(value) * factor for value in source.capacity)
    reset = report.capacity_reset(after)
    assert Q.from_float(after[0]) / 2 != Q.from_float(after[1]) / 3
    assert not reset.blockwise_scaling_certified
    assert not reset.preservation_certified
    assert reset.after.status == "excluded"
    assert reset.status == "excluded"


def test_external_complex_mismatch_cannot_be_replaced_by_equal_moduli():
    source = _source(
        ((0, 2), (0, 3), (1, 2), (1, 3)), capacities=(2,) * 4, weights=(2,) * 4
    )
    report = _assess(source, blocks=((0, 1), (2, 3)), offsets=(0, Q(1, 4), 0, 0))
    assert report.weighted_compatibility_certified
    assert report.internal_balance_certified
    assert not report.moment_compatibility_certified
    assert report.status == "excluded"
    assert report.row_cosine_bounds[0][1] == I(2)
    assert report.row_sine_bounds[0][1] == I(0)
    assert report.row_cosine_bounds[1][1] == I(0)
    assert report.row_sine_bounds[1][1] == I(-2)


def test_nonzero_complex_moment_respects_edge_orientation_and_lift_gauge(source):
    offsets = (0, 0, 0, 0, Q(1, 4))
    report = _assess(source, offsets=offsets)
    assert report.invariance_certified
    assert report.row_cosine_bounds[0][1] == I(0)
    assert report.row_sine_bounds[0][1] == I(1)
    assert report.row_sine_bounds[4][0] == I(-8)
    state = report.evaluate((1, 0), (0, Q(1, 8)))
    with mp.workdps(180):
        dx, dtheta, _, storage = _direct_rows_and_storage(report, (1, 0), (0, Q(1, 8)))
        for i in range(5):
            _contains(state.full_form_rates[i], dx[i])
            _contains(state.full_phase_rates[i], dtheta[i])
        _contains(state.full_storage_bounds, storage)
        _contains(state.quotient_phase_storage_bounds, 4 + 2 * mp.sqrt(2))
    reversed_source = replace(
        source,
        edges=tuple((j, i) for i, j in source.edges[::-1]),
        conductances=source.conductances[::-1],
    )
    reversed_report = _assess(
        reversed_source, offsets=tuple(turn + 3 for turn in offsets)
    )
    reversed_state = reversed_report.evaluate((1, 0), (0, Q(1, 8)))
    assert reversed_report.invariance_certified
    assert reversed_state.full_form_rates == state.full_form_rates
    assert reversed_state.full_phase_rates == state.full_phase_rates
    assert reversed_state.full_storage_bounds == state.full_storage_bounds
    assert reversed_state.phase_mean_turns == state.phase_mean_turns + 3


def test_internal_sine_current_is_an_independent_obstruction():
    edges = tuple((i, j) for i in range(4) for j in range(4, 8)) + (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    )
    source = _source(edges, capacities=(5,) * 8, weights=(2,) * 16 + (3,) * 4)
    report = _assess(
        source,
        blocks=((0, 1, 2, 3), (4, 5, 6, 7)),
        offsets=(0, Q(1, 4), Q(1, 2), Q(3, 4)) * 2,
    )
    assert report.weighted_compatibility_certified
    assert report.moment_compatibility_certified
    assert not report.internal_balance_certified
    assert report.status == "excluded"
    assert report.internal_current_bounds == (I(1), I(-1)) * 4


def test_internal_real_moments_and_lift_offsets_remain_in_their_own_roles():
    edges = ((0, 1), (1, 2), (3, 4), (3, 5), (0, 3), (1, 3), (1, 4), (2, 5))
    source = _source(edges, capacities=(2, 2, 2, 3, 3, 3), weights=(7,) * 4 + (2,) * 4)
    offsets = (0, 0, Q(1, 2), 0, 0, Q(1, 2))
    report = _assess(source, blocks=((0, 1, 2), (3, 4, 5)), offsets=offsets)
    assert report.invariance_certified
    assert tuple(
        report.row_cosine_bounds[i][report.node_blocks[i]] for i in range(6)
    ) == tuple(map(I, (1, 0, -1, 0, Q(3, 2), -Q(3, 2))))
    state = report.evaluate((2, -3), (Q(1, 7), Q(1, 9)))
    assert state.internal_phase_storage_bounds == (I(2), I(2))
    assert state.internal_form_storage == (0, 0)
    weights = tuple(Q(d, nu) for d, nu in zip(source.degrees, source.capacity))
    total = sum(weights)
    offset_mean = (
        sum(weight * offset for weight, offset in zip(weights, offsets)) / total
    )
    assert state.phase_offset_mean_turns == offset_mean
    assert state.phase_mean_turns == state.collective_phase_mean_turns + offset_mean
    with mp.workdps(180):
        dx, dtheta, _, storage = _direct_rows_and_storage(
            report, (2, -3), (Q(1, 7), Q(1, 9))
        )
        for i in range(6):
            _contains(state.full_form_rates[i], dx[i])
            _contains(state.full_phase_rates[i], dtheta[i])
        _contains(state.full_storage_bounds, storage)


def test_unit_law_reduction_converts_the_old_tau_clock_to_t():
    edges = tuple((i, (i + 1) % 5) for i in range(5)) + tuple(
        (i, i + 5) for i in range(5)
    )
    graph = _graph(edges, capacities=(1,) * 10)
    blocks = (tuple(range(5)), tuple(range(5, 10)))
    offsets = tuple(Q(i, 5) for i in range(5)) * 2
    old = assess_sine_phase_offset_partition(
        bound_relational_sine_exchange(graph, reference_model=_model()),
        blocks=blocks,
        phase_offset_turns=offsets,
    )
    new = _assess(
        bound_relational_sine_support_exchange(graph, reference_model=_model()),
        blocks=blocks,
        offsets=offsets,
    )
    assert new.invariance_certified
    old_state, new_state = old.evaluate((2, -1), (0, Q(1, 4))), new.evaluate(
        (2, -1), (0, Q(1, 4))
    )
    assert old_state.clock == "tau=t/pi"
    assert new_state.clock == "t"
    assert new_state.full_phase_rate_numerators == old_state.full_phase_rates
    assert new_state.full_storage_bounds == old_state.full_storage_bounds
    for new_rate, old_rate in zip(
        new_state.block_form_rates, old_state.block_form_rates
    ):
        assert new_rate == old_rate / pi_interval()


def test_unresolved_symbolic_identities_are_not_false_counterexamples():
    source = _source(
        tuple((i, j) for i in range(4) for j in range(4, 7)),
        capacities=(3,) * 4 + (4,) * 3,
        weights=(2,) * 12,
    )
    report = _assess(
        source,
        blocks=((0, 1, 2, 3), (4, 5, 6)),
        offsets=(0, Q(1, 4), Q(1, 2), Q(3, 4), 0, Q(1, 3), Q(2, 3)),
    )
    assert report.weighted_compatibility_certified
    assert report.internal_balance_certified
    assert report.status == "unavailable"
    assert any("unresolved" in row for row in report.cross_moment_statuses)
    assert not any("excluded" in row for row in report.cross_moment_statuses)


def test_evaluation_and_capacity_reset_rebuild_poisoned_derived_reports(report):
    poisoned = replace(
        report,
        source=replace(
            report.source,
            form_rates=(I(999),),
            phase_rates=(I(-999),),
            storage=I(-1),
            form_gradient=(Q(99),),
        ),
        status="excluded",
        invariance_certified=False,
        mobility=(),
        inverse_capacity_weights=(),
        block_mean_weights=(),
        weighted_rows=(),
        node_blocks=(),
        block_indices=(),
        quotient_weighted_rows=((True,),),
        quotient_cosine_expressions=(),
        quotient_sine_expressions=(),
        cross_bare_counts=(),
        cross_conductance_sums=(),
    )
    actual = poisoned.evaluate((1, -1), (0, Q(1, 8)))
    expected = report.evaluate((1, -1), (0, Q(1, 8)))
    assert actual.partition.invariance_certified
    assert actual.block_form_rates == expected.block_form_rates
    assert actual.block_phase_rates == expected.block_phase_rates
    assert actual.full_storage_bounds == expected.full_storage_bounds
    reset = poisoned.capacity_reset(STAR_CAPACITIES)
    assert reset.before.invariance_certified and reset.after.invariance_certified
    assert reset.blockwise_scaling_certified and reset.preservation_certified


@pytest.mark.parametrize("cached_degrees", ((True, 2, 2, 2, 4), (5, 2, 2, 2, 4)))
def test_degrees_are_rebuilt_from_bare_support_not_the_cached_vector(
    source, cached_degrees
):
    rebuilt = _assess(replace(source, degrees=cached_degrees))
    assert rebuilt.source.degrees == (4, 2, 2, 2, 4)
    assert rebuilt.mobility == (1, 1, 1, 1, 2)
    assert rebuilt.invariance_certified


def test_exports_rebuild_fields_and_admit_the_partition_type_before_dispatch(report):
    state = report.evaluate((1, 0), (0, Q(1, 8)))
    reset = report.capacity_reset(STAR_CAPACITIES)
    poisoned_report = replace(report, status="excluded", invariance_certified=False)
    assert poisoned_report.to_dict() == report.to_dict()
    assert replace(state, full_storage_bounds=I(-123)).to_dict() == state.to_dict()
    assert (
        replace(reset, preservation_certified=False, storage_jump=Q(99)).to_dict()
        == reset.to_dict()
    )

    class NotAPartition:
        def evaluate(self, *args):
            raise AssertionError("unadmitted polymorphic dispatch")

    with pytest.raises(TypeError):
        replace(state, partition=NotAPartition()).to_dict()


@pytest.mark.parametrize("field", ("epi", "phase", "capacity", "conductances"))
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), complex(1, 2)))
def test_invalid_authoritative_primitives_cannot_use_cached_favorable_fields(
    source, field, value
):
    bad = replace(source, **{field: (value,) + getattr(source, field)[1:]})
    with pytest.raises((TypeError, ValueError)):
        _assess(bad)


@pytest.mark.parametrize("field", ("capacity", "conductances"))
@pytest.mark.parametrize("value", (0, -1))
def test_strictly_positive_theorem_domain(source, field, value):
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, **{field: (value,) + getattr(source, field)[1:]}))


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_model_boolean_is_not_a_matching_coefficient(source, field):
    model = copy(source.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, reference_model=model))


@pytest.mark.parametrize(
    "field,value", (("epi_weight", 0.5), ("phase_weight", 0.5), ("storage_scale", 2))
)
def test_complete_law_coefficients_are_not_transferred_from_other_models(
    source, field, value
):
    model = copy(source.reference_model)
    object.__setattr__(model, field, value)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, reference_model=model))


@pytest.mark.parametrize(
    "mutation",
    (
        {"law": "native"},
        {"edges": STAR_EDGES + ((1, 0),)},
        {"edges": ((0, 0),) + STAR_EDGES[1:]},
    ),
)
def test_primitive_support_and_degree_consistency(source, mutation):
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, **mutation))


@pytest.mark.parametrize(
    "offsets", ((0,) * 4, (0,) * 6, (False, 0, 0, 0, 0), (0.0,) * 5)
)
def test_exact_offset_admission(source, offsets):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, offsets=offsets)


@pytest.mark.parametrize(
    "blocks", (((0, 1, 2), (3,)), ((0, 1, 2, 3), (3, 4)), ((0, 1, 2, 9), (4,)))
)
def test_partition_must_cover_once_and_only_the_actual_nodes(source, blocks):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, blocks=blocks)


@pytest.mark.parametrize(
    "forms,turns",
    (
        ((True, 0), (0, 0)),
        ((1, float("inf")), (0, 0)),
        ((1,), (0, 0)),
        ((1, 0), (0, True)),
        ((1, 0), (0, 0.125)),
        ((1, 0), (0,)),
    ),
)
def test_collective_coordinate_admission(report, forms, turns):
    with pytest.raises((TypeError, ValueError)):
        report.evaluate(forms, turns)


@pytest.mark.parametrize(
    "value", (0, -1, True, float("inf"), float("nan"), Q(1, 3), Q(1, 2**1075))
)
def test_capacity_map_requires_positive_actual_represented_outputs(report, value):
    with pytest.raises((TypeError, ValueError)):
        report.capacity_reset((value,) + STAR_CAPACITIES[1:])


def test_forged_positive_verdict_cannot_override_invalid_or_changed_primitives(report):
    invalid = replace(
        report, source=replace(report.source, capacity=(0,) + STAR_CAPACITIES[1:])
    )
    with pytest.raises((TypeError, ValueError)):
        invalid.evaluate((1, 0), (0, Q(1, 4)))
    with pytest.raises((TypeError, ValueError)):
        invalid.capacity_reset(STAR_CAPACITIES)
    changed = replace(report, phase_offset_turns=(0, Q(1, 4), 0, 0, 0))
    with pytest.raises(ValueError, match="certified"):
        changed.evaluate((1, 0), (0, Q(1, 4)))
