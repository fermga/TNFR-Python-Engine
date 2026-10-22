"""Independent complete-law controls for exact zero-offset radian states."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_weighted_partition import (
    assess_sine_weighted_phase_offset_partition,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

BLOCKS = ((0, 1), (2, 3), (4, 5))
FORMS = (Q(1, 4), Q(-1, 8), Q(3, 8))
PHASES = (Q(3, 8), Q(-1, 4), Q(5, 8))
CAPACITIES = (5, 5, 10, 10, 15, 15)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.fixture(scope="module")
def source():
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, capacity in enumerate(CAPACITIES):
        graph.add_node(
            node,
            EPI=float(FORMS[node // 2]),
            theta=float(PHASES[node // 2]),
            nu_f=float(capacity),
        )
    internal = (7, 11, 13)
    cross = {(0, 1): 2, (0, 2): 3, (1, 2): 5}
    for i in range(6):
        for j in range(i + 1, 6):
            a, b = i // 2, j // 2
            graph.add_edge(i, j, weight=internal[a] if a == b else cross[a, b])
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )


@pytest.fixture(scope="module")
def partition(source):
    return assess_sine_weighted_phase_offset_partition(
        source, blocks=BLOCKS, phase_offset_turns=(0,) * 6
    )


@pytest.fixture(scope="module")
def observed(partition):
    return partition.observe_radian_state()


def _direct_field(mp, source):
    """Sum actual edges without consuming derived degrees, gradients or rows."""
    positions = {node: i for i, node in enumerate(source.nodes)}
    degrees = [0] * len(source.nodes)
    currents = [mp.mpf(0) for _ in source.nodes]
    gradient = [Q(0) for _ in source.nodes]
    form_storage, phase_storage = Q(0), mp.mpf(0)
    for (left, right), weight in zip(source.edges, source.conductances):
        i, j = positions[left], positions[right]
        degrees[i] += 1
        degrees[j] += 1
        difference = source.epi[i] - source.epi[j]
        angle = _mp(mp, source.phase[j] - source.phase[i])
        current = mp.sin(angle)
        currents[i] += current
        currents[j] -= current
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        form_storage += weight * difference**2 / 2
        phase_storage += 1 - mp.cos(angle)
    pressure = tuple(
        current / (degree * mp.pi) for current, degree in zip(currents, degrees)
    )
    form_rates = tuple(_mp(mp, nu) * p for nu, p in zip(source.capacity, pressure))
    phase_numerators = tuple(
        nu * q / d for nu, q, d in zip(source.capacity, gradient, degrees)
    )
    weights = tuple(Q(d) / nu for d, nu in zip(degrees, source.capacity))
    return pressure, form_rates, phase_numerators, form_storage, phase_storage, weights


def test_three_block_radian_capture_matches_independent_complete_edge_rows(
    observed, mp
):
    state, full = observed, observed.full_source
    pressure, dx, dtheta, form_storage, phase_storage, weights = _direct_field(mp, full)
    assert state.source_membership_certified and state.full_row_equality_certified
    assert state.collective_form == FORMS
    assert state.collective_phase_radians == PHASES
    assert state.quotient_bare_rows == ((0, 2, 2), (4, 0, 4), (6, 6, 0))
    assert state.partition.block_mean_weights == (2, 1, Q(2, 3))
    assert state.block_phase_rate_numerators == (Q(3, 4), -13, Q(69, 4))
    assert state.full_phase_rate_numerators == dtheta
    assert (
        form_storage == full.form_storage == state.quotient_form_storage == Q(101, 32)
    )
    assert state.mean_weights == weights == (1, 1, Q(1, 2), Q(1, 2), Q(1, 3), Q(1, 3))
    assert state.form_mean == Q(15, 88)
    assert state.phase_mean_radians == Q(1, 4)
    assert state.form_mean_rate_bounds.contains(0)
    assert state.phase_mean_rate_numerator == 0
    assert state.clock == "t"
    assert state.phase_rate_units == "radians_per_original_structural_time"
    for i in range(6):
        block = i // 2
        _contains(mp, full.pressure[i], pressure[i])
        _contains(mp, full.form_rates[i], dx[i])
        _contains(mp, state.block_form_rates[block], dx[i])
        _contains(mp, full.phase_rates[i], _mp(mp, dtheta[i]) / mp.pi)
        _contains(mp, state.block_phase_rates[block], _mp(mp, dtheta[i]) / mp.pi)
    _contains(mp, full.phase_storage, phase_storage)
    _contains(mp, state.quotient_phase_storage_bounds, phase_storage)
    _contains(mp, full.storage, _mp(mp, form_storage) + phase_storage)
    assert not any(state.phase_rate_residuals)
    assert all(value.contains(0) for value in state.form_rate_residual_bounds)
    assert state.storage_residual_bounds.contains(0)
    with pytest.raises(FrozenInstanceError):
        state.source_membership_certified = False


def test_explicit_other_family_state_is_not_a_capture_of_the_original(partition, mp):
    state = partition.evaluate_radians(
        (Q(1, 8), Q(-3, 8), Q(7, 16)), (0, Q(1, 10), Q(-2, 7))
    )
    assert not state.source_membership_certified
    assert state.full_row_equality_certified
    assert state.partition.source.epi != state.full_source.epi
    assert state.partition.source.phase != state.full_source.phase
    _, rates, numerators, form_storage, phase_storage, _ = _direct_field(
        mp, state.full_source
    )
    for i, block in enumerate(state.partition.node_blocks):
        _contains(mp, state.block_form_rates[block], rates[i])
        assert state.block_phase_rate_numerators[block] == numerators[i]
    assert state.quotient_form_storage == form_storage
    _contains(mp, state.quotient_phase_storage_bounds, phase_storage)


@pytest.mark.parametrize("field", ("epi", "phase"))
def test_exact_off_family_capture_is_rejected_without_tolerance_projection(
    partition, field
):
    values = list(getattr(partition.source, field))
    values[1] += Q(1, 2**80)
    changed = replace(
        partition, source=replace(partition.source, **{field: tuple(values)})
    )
    with pytest.raises(ValueError, match="block|family|equal|member"):
        changed.observe_radian_state()
    # Explicit family evaluation remains a different, honestly labelled state.
    evaluated = changed.evaluate_radians(FORMS, PHASES)
    assert evaluated.full_row_equality_certified
    assert not evaluated.source_membership_certified


@pytest.mark.parametrize("offset", (Q(1, 4), Q(1)))
def test_radian_interface_does_not_silently_discard_nonzero_turn_offsets(
    source, offset
):
    partition = assess_sine_weighted_phase_offset_partition(
        source, blocks=BLOCKS, phase_offset_turns=(offset,) * 6
    )
    assert partition.invariance_certified
    with pytest.raises(ValueError, match="zero|offset"):
        partition.observe_radian_state()
    with pytest.raises(ValueError, match="zero|offset"):
        partition.evaluate_radians(FORMS, PHASES)
    assert partition.evaluate(FORMS, (0, 0, 0)).full_row_equality_certified


def test_radian_lifts_are_not_the_same_numeric_values_in_turns(partition, mp):
    radians = partition.evaluate_radians(FORMS, PHASES)
    turns = partition.evaluate(FORMS, PHASES)
    expected_radian = 2 * (mp.sin(-mp.mpf(5) / 8) + mp.sin(mp.mpf(1) / 4)) / mp.pi
    expected_turn = 2 * (mp.sin(-5 * mp.pi / 4) + mp.sin(mp.pi / 2)) / mp.pi
    _contains(mp, radians.block_form_rates[0], expected_radian)
    _contains(mp, turns.block_form_rates[0], expected_turn)
    assert radians.block_form_rates[0].hi < 0 < turns.block_form_rates[0].lo
    # The reciprocal phase row consumes form, so its radian/time unit is unchanged.
    assert radians.block_phase_rate_numerators == turns.block_phase_rate_numerators


def test_common_lift_and_form_translation_preserve_rows_and_shift_means(
    partition, observed
):
    form_shift, phase_shift = Q(7, 8), Q(-5, 16)
    shifted = partition.evaluate_radians(
        tuple(x + form_shift for x in FORMS),
        tuple(theta + phase_shift for theta in PHASES),
    )
    assert shifted.block_form_rates == observed.block_form_rates
    assert shifted.block_phase_rates == observed.block_phase_rates
    assert shifted.full_source.storage == observed.full_source.storage
    assert shifted.form_mean == observed.form_mean + form_shift
    assert shifted.phase_mean_radians == observed.phase_mean_radians + phase_shift
    assert not shifted.source_membership_certified


def test_edge_orientation_and_declared_block_order_do_not_change_fine_state(
    source, observed, mp
):
    source = replace(
        source,
        edges=tuple((j, i) for i, j in reversed(source.edges)),
        conductances=source.conductances[::-1],
    )
    partition = assess_sine_weighted_phase_offset_partition(
        source, blocks=BLOCKS[::-1], phase_offset_turns=(0,) * 6
    )
    state = partition.observe_radian_state()
    assert state.source_membership_certified
    assert state.collective_form == FORMS[::-1]
    assert state.collective_phase_radians == PHASES[::-1]
    # Reversing a phase difference can change outward rounding endpoints;
    # both evaluations must enclose the same independently evaluated rows.
    _, rates, _, form_storage, phase_storage, _ = _direct_field(mp, source)
    for i, block in enumerate(state.partition.node_blocks):
        _contains(mp, state.block_form_rates[block], rates[i])
        _contains(mp, state.full_source.form_rates[i], rates[i])
        _contains(mp, observed.full_source.form_rates[i], rates[i])
    assert state.block_phase_rates == observed.block_phase_rates[::-1]
    assert state.full_source.phase_rates == observed.full_source.phase_rates
    _contains(mp, state.full_source.storage, _mp(mp, form_storage) + phase_storage)
    assert state.form_mean == observed.form_mean
    assert state.phase_mean_radians == observed.phase_mean_radians


def test_source_membership_uses_normalized_exact_or_represented_coordinates(partition):
    state = partition.evaluate_radians(
        tuple(map(float, FORMS)), tuple(map(float, PHASES))
    )
    assert state.source_membership_certified
    exact = partition.evaluate_radians(FORMS, (Q(1, 10), 0, 0))
    represented = partition.evaluate_radians(FORMS, (0.1, 0, 0))
    assert exact.collective_phase_radians[0] == Q(1, 10)
    assert represented.collective_phase_radians[0] == Q.from_float(0.1)
    assert exact.collective_phase_radians != represented.collective_phase_radians


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi", True),
        ("phase", float("nan")),
        ("phase", complex(1, 2)),
        ("capacity", 0),
        ("conductances", -1),
    ),
)
def test_invalid_source_primitives_override_favorable_derived_claims(
    partition, field, value
):
    values = (value,) + getattr(partition.source, field)[1:]
    changed = replace(
        partition,
        source=replace(partition.source, **{field: values}),
        invariance_certified=True,
    )
    with pytest.raises((TypeError, ValueError)):
        changed.observe_radian_state()
    with pytest.raises((TypeError, ValueError)):
        changed.evaluate_radians(FORMS, PHASES)


@pytest.mark.parametrize(
    "forms,phases",
    (
        ((True, 0, 0), PHASES),
        ((0, float("inf"), 0), PHASES),
        ((0, 0), PHASES),
        (FORMS, (False, 0, 0)),
        (FORMS, (0, float("inf"), 0)),
        (FORMS, (0, 0)),
    ),
)
def test_radian_coordinate_admission_precedes_numeric_conversion(
    partition, forms, phases
):
    with pytest.raises((TypeError, ValueError)):
        partition.evaluate_radians(forms, phases)


def test_cached_partition_fields_cannot_promote_a_nonclosed_law(partition):
    changed = replace(
        partition,
        source=replace(partition.source, capacity=(6,) + partition.source.capacity[1:]),
        invariance_certified=True,
        status="certified",
    )
    with pytest.raises(ValueError, match="certified|closure"):
        changed.observe_radian_state()
    with pytest.raises(ValueError, match="certified|closure"):
        changed.evaluate_radians(FORMS, PHASES)


def test_export_rebuilds_all_consumed_derived_fields_and_uses_atomic_sdk(
    partition, observed, tmp_path
):
    poisoned_partition = replace(
        partition,
        source=replace(
            partition.source,
            degrees=(99,) * 6,
            form_rates=(I(999),),
            phase_rates=(I(-999),),
            storage=I(-1),
        ),
        invariance_certified=False,
        status="excluded",
        block_indices=(),
        node_blocks=(),
        mobility=(),
        block_mean_weights=(),
        cross_bare_counts=(),
        cross_conductance_sums=(),
    )
    state = poisoned_partition.observe_radian_state()
    assert state.to_dict() == observed.to_dict()
    poisoned = replace(
        state,
        full_source=replace(state.full_source, storage=I(-999)),
        source_membership_certified=False,
        quotient_bare_rows=(),
        block_form_rates=(I(99),),
        full_row_equality_certified=False,
        form_mean=Q(99),
        phase_mean_radians=Q(99),
    )
    expected = relational_report_to_dict(observed)
    assert relational_report_to_dict(poisoned) == expected
    path = tmp_path / "radian-state.json"
    export_to_json(relational_report_to_dict(poisoned), path)
    assert json_loads(path.read_text(encoding="utf-8")) == expected
    before = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(
                replace(poisoned, collective_phase_radians=(True, 0, 0))
            ),
            path,
        )
    assert path.read_bytes() == before


def test_export_does_not_dispatch_to_an_unadmitted_partition(observed):
    class NotAPartition:
        def evaluate_radians(self, *args, **kwargs):
            raise AssertionError("unadmitted polymorphic dispatch")

    with pytest.raises(TypeError):
        replace(observed, partition=NotAPartition()).to_dict()
