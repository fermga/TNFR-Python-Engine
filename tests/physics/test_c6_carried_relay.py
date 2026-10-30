"""Exact relay pressures, tie ownership and bounded first passive-cell exit."""

import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import (
    NodalRemainderState,
    _validate_nodal_remainder_state,
)
from tnfr.physics.binary64_nodal_flow import _rounding_cell
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_carried_relay import (
    _formula,
    derive_c6_carried_relay,
    observe_c6_carried_relay_exit,
)
from tnfr.physics.c6_pressure_lattice import (
    _observe_rebuilt_c6_pressure_lattice,
    derive_c6_pressure_lattice,
)

PHASE = tuple(
    map(
        float.fromhex,
        (
            "0x1.0b8fb3e3956cbp-55",
            "0x1.0c152382d7365p+0",
            "0x1.0c152382d7365p+1",
            "0x1.921fb54442d18p+1",
            "0x1.0c152382d7365p+2",
            "0x1.4f1a6c638d03fp+2",
        ),
    )
)
DELTA, M, GRID = F(1, 2**54), F(1, 2**113), F(1, 2**3222)


def _origin():
    # Captured B43 pre-SHA data; no graph, event word or live cycle is run.
    terms = (
        (-35698968670925291, 110),
        (229688483940001907, 113),
        (-203177807149452541, 112),
        (-17846977754793875, 109),
        (-139871603228525, 101),
        (41950776505050555, 111),
    )
    epi = tuple(float(F(1, 2) + offset * DELTA) for offset in (-3, 0, 2, 0, 8, -5))
    return NodalRemainderState(epi, tuple(F(a, 2**b) for a, b in terms), 0.375, 0.625)


@pytest.fixture(scope="module")
def profile():
    reference = derive_c6_pressure_lattice(
        phase=PHASE,
        epi_weight=CHANNEL_WEIGHT_SECONDARY,
        phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    return derive_c6_carried_profile(reference)


def _closure(profile, state=None, timestep=0.0625):
    return derive_c6_carried_closure(
        profile, state=_origin() if state is None else state, timestep=timestep
    )


@pytest.fixture(scope="module")
def relay(profile):
    return derive_c6_carried_relay(_closure(profile))


@pytest.fixture(scope="module")
def passage(relay):
    return observe_c6_carried_relay_exit(relay, step_budget=10)


def test_four_pressure_rows_have_exact_independent_opposite_responses(relay, profile):
    expected_offsets = (
        (-3, 0, 2, 0, 8, -5),
        (-4, 0, 2, 0, 8, -5),
        (-3, 0, 2, -1, 8, -5),
        (-4, 0, 2, -1, 8, -5),
    )
    assert relay.visible_rows == tuple(
        tuple(float(F(1, 2) + value * DELTA) for value in row)
        for row in expected_offsets
    )
    independent = tuple(
        _observe_rebuilt_c6_pressure_lattice(profile.lattice, row).pressure
        for row in relay.visible_rows
    )
    assert relay.pressure_rows == independent
    exact = tuple(tuple(map(F, row)) for row in independent)
    assert all(
        exact[3][i] - exact[1][i] - exact[2][i] + exact[0][i] == 0 for i in range(6)
    )
    assert tuple(i for i in range(6) if exact[1][i] != exact[0][i]) == (0, 1, 5)
    assert tuple(i for i in range(6) if exact[2][i] != exact[0][i]) == (2, 3, 4)


def test_actual_coefficients_keep_pressure_rounding_instead_of_ideal_half(relay):
    first, second = relay.axes
    assert (first.decrement / M, first.increment / M) == (
        3039824576180305,
        3558973984143090,
    )
    assert (second.decrement / M, second.increment / M) == (
        1803052714421816,
        4795745845901584,
    )
    assert first.correction == (
        F(1),
        F(-3299399280161697, 6598798560323395),
        F(0),
        F(0),
        F(0),
        F(-3299399280161696, 6598798560323395),
    )
    assert second.correction == (
        F(0),
        F(0),
        F(-412424910020212, 824849820040425),
        F(1),
        F(-137474970006737, 274949940013475),
        F(0),
    )
    assert first.correction[1] != first.correction[5]
    assert second.correction[2] != second.correction[4]
    assert relay.drift[0] == relay.drift[3] == 0
    assert tuple((value > 0) - (value < 0) for value in relay.drift) == (
        0,
        -1,
        1,
        0,
        -1,
        -1,
    )


def test_complementary_ties_and_full_strip_outer_extents(relay):
    assert tuple((axis.lower_closed, axis.upper_closed) for axis in relay.axes) == (
        (False, True),
        (True, False),
    )
    for axis in relay.axes:
        assert axis.lower_visible == math.nextafter(axis.upper_visible, -math.inf)
        assert axis.facet == (F(axis.upper_visible) + F(axis.lower_visible)) / 2
        owner = axis.upper_visible if axis.lower_closed else axis.lower_visible
        assert float(axis.facet) == owner
        lower = _rounding_cell(axis.lower_visible, axis.facet)
        upper = _rounding_cell(axis.upper_visible, axis.facet)
        assert (
            lower.lower
            < axis.facet - axis.decrement
            < axis.facet + axis.increment
            < upper.upper
        )
        assert (
            F(0.375)
            <= axis.facet - axis.decrement
            < axis.facet + axis.increment
            <= F(0.625)
        )


def test_signed_deadline_is_derived_from_the_favorable_correction(relay):
    assert relay.exit_deadlines == (None, 205, 299, None, 10, 2899)
    assert relay.deadline == 10 and relay.deadline_nodes == (4,)
    exact = _validate_nodal_remainder_state(relay.closure.base_tube.state)
    for node in relay.nonrelay_nodes:
        deadline, velocity = relay.exit_deadlines[node], relay.drift[node]
        cell = _rounding_cell(relay.visible_rows[0][node], exact[node])
        if velocity > 0:
            bound = exact[node] + relay.correction_lower[node]
            assert bound + deadline * velocity > cell.upper
            assert bound + (deadline - 1) * velocity <= cell.upper
        else:
            bound = exact[node] + relay.correction_upper[node]
            assert bound + deadline * velocity < cell.lower
            assert bound + (deadline - 1) * velocity >= cell.lower


def test_first_passive_exit_has_canonical_prefix_and_exact_nodal_telescope(
    relay, passage, profile
):
    assert passage.exit_step == 7 and passage.exiting_nodes == (4,)
    assert tuple(
        relay.visible_rows.index(step.before.epi) for step in passage.steps
    ) == (0, 1, 2, 0, 1, 0, 3)
    assert len(passage.points) == 8 and len(passage.steps) == 7
    assert passage.steps[0].before == _origin()
    assert not passage.band_failure and passage.endpoint == passage.steps[-1].after
    assert passage.endpoint.epi[4] == float(F(1, 2) + 6 * DELTA)
    for ordinal, step in enumerate(passage.steps):
        assert step.before.epi in relay.visible_rows
        assert (
            step.pressure
            == _observe_rebuilt_c6_pressure_lattice(
                profile.lattice, step.before.epi
            ).pressure
        )
        assert step.capacity == (1.0,) * 6 and step.timestep == 0.0625
        assert step.before.exact_epi == passage.points[ordinal].exact_epi
        assert step.after.exact_epi == passage.points[ordinal + 1].exact_epi
        assert not any(step.nodal_balance_residual)
        if ordinal < passage.exit_step - 1:
            assert all(
                step.after.epi[node] == _origin().epi[node]
                for node in relay.nonrelay_nodes
            )
    area = tuple(
        sum((step.exact_increment[i] for step in passage.steps), F(0)) for i in range(6)
    )
    assert area == passage.total_nodal_area
    assert tuple(value / M for value in area) == (
        -1482376352291950,
        -17082760244853979,
        22807680901284836,
        576228119694088,
        -4451855519549880,
        -366916904283112,
    )
    assert (
        passage.mean_area == F(1, 2**114) and sum(area, F(0)) / 6 == passage.mean_area
    )
    assert (
        tuple(b - a for a, b in zip(_origin().exact_epi, passage.exact_endpoint))
        == area
    )
    assert not any(passage.nodal_balance_residual)


def test_switch_counts_match_actual_pre_step_cell_ownership(passage):
    counts = [0, 0]
    assert passage.points[0].lower_visits == (0, 0)
    for step, point in zip(passage.steps, passage.points[1:]):
        for index, axis in enumerate(passage.relay.axes):
            counts[index] += int(step.before.epi[axis.node] == axis.lower_visible)
        assert point.lower_visits == tuple(counts)
        assert not any(point.drift_identity_residual)


def test_large_scalar_periods_are_exact_modular_identities_not_runtime_extensions(
    relay,
):
    periods = []
    for axis in relay.axes:
        a, b = int(axis.decrement / M), int(axis.increment / M)
        periods.append((a + b) // math.gcd(a, b))
    assert tuple(periods) == (1319759712064679, 824849820040425)
    for ordinal in (1, 7, *periods, math.lcm(*periods)):
        point = _formula(relay, ordinal)
        for index, axis in enumerate(relay.axes):
            shifted = _formula(relay, ordinal + periods[index])
            assert shifted.relay_offsets[index] == point.relay_offsets[index]
            value = point.relay_offsets[index]
            assert -axis.decrement <= value <= axis.increment
            assert value != (-axis.decrement if axis.upper_closed else axis.increment)
            assert (
                value - axis.initial_offset + ordinal * axis.decrement
            ) == point.lower_visits[index] * axis.length
        assert not any(point.drift_identity_residual)
    # Only seven steps were validated as an actual conditional six-node prefix.
    assert relay.deadline < min(periods) and not relay.future_runtime_certified


@pytest.mark.parametrize("node", (0, 3))
def test_exact_central_facet_arrival_obeys_both_nearest_even_rules(
    relay, profile, node
):
    axis = next(axis for axis in relay.axes if axis.node == node)
    source = _origin()
    carry = list(source.remainder)
    carry[node] = axis.facet + axis.decrement - F(source.epi[node])
    modified = replace(source, remainder=tuple(carry))
    altered = derive_c6_carried_relay(_closure(profile, modified))
    observed = observe_c6_carried_relay_exit(altered)
    after = observed.steps[0].after
    assert after.exact_epi[node] == axis.facet
    assert after.epi[node] == (
        axis.upper_visible if axis.lower_closed else axis.lower_visible
    )
    assert observed.points[1].relay_offsets[altered.relay_nodes.index(node)] == 0


def test_closed_upper_strip_endpoint_is_admitted_and_open_one_rejected(relay, profile):
    source = _origin()
    first, second = relay.axes
    carry = list(source.remainder)
    carry[first.node] = first.facet + first.increment - F(source.epi[first.node])
    included = derive_c6_carried_relay(
        _closure(profile, replace(source, remainder=tuple(carry)))
    )
    assert included.axes[0].initial_offset == first.increment
    carry = list(source.remainder)
    carry[second.node] = second.facet + second.increment - F(source.epi[second.node])
    with pytest.raises(ValueError, match="outside a derived relay strip"):
        derive_c6_carried_relay(
            _closure(profile, replace(source, remainder=tuple(carry)))
        )
    carry[second.node] -= GRID
    admitted = derive_c6_carried_relay(
        _closure(profile, replace(source, remainder=tuple(carry)))
    )
    assert admitted.axes[1].initial_offset == second.increment - GRID


def test_zero_carry_cannot_be_substituted_for_the_actual_origin(profile):
    source = replace(_origin(), remainder=(F(0),) * 6)
    with pytest.raises(ValueError, match="outside a derived relay strip"):
        derive_c6_carried_relay(_closure(profile, source))


def test_reversing_opposite_axis_order_preserves_the_physical_passage(relay, passage):
    reverse = derive_c6_carried_relay(relay.closure, relay_nodes=(3, 0))
    observed = observe_c6_carried_relay_exit(reverse, step_budget=10)
    assert (
        reverse.drift == relay.drift and reverse.exit_deadlines == relay.exit_deadlines
    )
    assert (
        observed.steps == passage.steps
        and observed.exact_endpoint == passage.exact_endpoint
    )
    assert tuple(point.lower_visits for point in observed.points) == tuple(
        point.lower_visits[::-1] for point in passage.points
    )


def test_band_failure_retains_exact_candidate_without_an_invalid_state(profile):
    source = _origin()
    carry = list(source.remainder)
    carry[5] = GRID
    source = replace(source, remainder=tuple(carry), epi_lower=source.epi[5])
    bound = derive_c6_carried_relay(_closure(profile, source))
    observed = observe_c6_carried_relay_exit(bound)
    assert observed.exit_step == 2 and observed.band_failure
    assert (
        observed.endpoint is None
        and len(observed.steps) == 1
        and len(observed.points) == 3
    )
    assert observed.exact_endpoint[5] < F(source.epi_lower)
    assert observed.exiting_nodes == ()
    assert not any(observed.nodal_balance_residual)


def test_public_cache_tampering_is_rebuilt_before_the_deadline_or_pressure_is_used(
    relay, passage
):
    forged = replace(
        relay,
        pressure_rows=((0.0,) * 6,) * 4,
        drift=(F(0),) * 6,
        base_increment=(F(0),) * 6,
        deadline=1,
        exit_deadlines=(None,) * 6,
        axes=tuple(
            replace(axis, correction=(F(1),) * 6, decrement=F(1)) for axis in relay.axes
        ),
    )
    observed = observe_c6_carried_relay_exit(forged, step_budget=10)
    assert observed == passage
    with pytest.raises(ValueError, match="exceeds step_budget"):
        observe_c6_carried_relay_exit(forged, step_budget=9)


def test_budget_must_cover_the_proved_deadline_even_when_observed_exit_is_earlier(
    relay, monkeypatch
):
    import tnfr.physics.c6_carried_relay as owner

    def no_step(*_args, **_kwargs):
        raise AssertionError("budget admission must precede every shared nodal step")

    monkeypatch.setattr(owner, "advance_nodal_remainder", no_step)
    with pytest.raises(ValueError, match="exceeds step_budget"):
        owner.observe_c6_carried_relay_exit(relay, step_budget=7)


@pytest.mark.parametrize("budget", (0, -1, True, 10.0, F(10)))
def test_work_budget_requires_a_positive_actual_integer(relay, budget):
    with pytest.raises(ValueError, match="positive exact integer"):
        observe_c6_carried_relay_exit(relay, step_budget=budget)


@pytest.mark.parametrize("nodes", ((0,), (0, 0), (0, 1), [0, 3], (False, 3), (0, 6)))
def test_relay_nodes_require_two_opposite_exact_indices(relay, nodes):
    with pytest.raises(ValueError, match="opposite ordered C6 nodes"):
        derive_c6_carried_relay(relay.closure, relay_nodes=nodes)


@pytest.mark.parametrize(
    "nodes,reason",
    (
        ((1, 4), "outside a derived relay strip"),
        ((2, 5), "strictly inward"),
    ),
)
def test_other_opposite_pairs_do_not_inherit_the_admitted_relay(relay, nodes, reason):
    with pytest.raises(ValueError, match=reason):
        derive_c6_carried_relay(relay.closure, relay_nodes=nodes)


def test_scope_and_original_carry_remain_unchanged(relay, passage):
    assert relay.closure.base_tube.state == _origin()
    assert (
        relay.initial_membership_verified and relay.conditional_product_relay_inclusion
    )
    assert not relay.full_region_invariant and not relay.future_runtime_certified
    assert (
        not passage.future_runtime_certified
        and not passage.indefinite_trapping_certified
    )
    with pytest.raises(FrozenInstanceError):
        relay.deadline = 1


def test_observer_rejects_noncertificate_input():
    with pytest.raises(TypeError, match="C6CarriedRelay"):
        observe_c6_carried_relay_exit(None)
