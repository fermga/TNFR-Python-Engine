"""Finite live-contact averaging, full-law identities and admission boundaries.

The derivative controls evaluate algebra at fixed states. They do not run a
trajectory, scan preparations or replay a retained research response.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES, _source
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.phase_cycle_geometry import C5_PHASE_SECTOR_BARRIER
from tnfr.physics.relational_sine_partition import assess_sine_contact_averaging

LEAF_PHASE = (Q(7, 20), Q(7, 20), Q(3, 20), Q(3, 20), Q(1, 4))


@pytest.fixture(scope="module")
def source():
    return replace(
        _source(CYCLE_LEAF_EDGES),
        epi=(Q(5),) * 5 + (Q(-15),) * 5,
        phase=(Q(0),) * 5 + LEAF_PHASE,
    )


def _assess(source, **changes):
    arguments = dict(cycle=tuple(range(5)), scaled_horizon=Q(1))
    return assess_sine_contact_averaging(source, **(arguments | changes))


def _with_gap(source, gap):
    return replace(source, epi=(gap / 4,) * 5 + (-3 * gap / 4,) * 5)


def test_high_energy_does_not_remove_finite_averaging_obstruction(source):
    result = _assess(source)
    assert result.form_gap == 20
    assert result.contact_speed_lower_bound == Q(68, 3)
    assert result.integrated_contact_current_upper_bound == Q(111, 1156)
    assert result.receiver_phase_storage_upper_bound == Q(54760, 751689)
    assert result.initial_full_storage_bounds.lo > 1000
    assert result.initial_full_storage_bounds.lo > result.phase_sector_barrier
    assert result.phase_sector_barrier == C5_PHASE_SECTOR_BARRIER
    assert result.speed_separation_certified
    assert result.whole_window_acute_winding_excluded
    assert result.status == "excluded" and result.reasons == ()
    assert result.clock == "tau=t/pi"
    assert result.scaled_horizon == 1
    assert result.contact_acceleration_upper_bound == 4
    assert result.to_dict()["schema"] == "tnfr.sine-contact-averaging.v1"
    with pytest.raises(FrozenInstanceError):
        result.status = "unavailable"


def test_signed_gap_and_arbitrary_leaf_phases_keep_same_bound(source):
    original = _assess(source)
    reversed_gap = _assess(_with_gap(source, Q(-20)))
    arbitrary_lags = _assess(
        replace(source, phase=(Q(0),) * 5 + (Q(-27), Q(19), Q(0), Q(4), Q(35)))
    )
    assert reversed_gap.form_gap == -20
    for result in (reversed_gap, arbitrary_lags):
        assert result.whole_window_acute_winding_excluded
        assert result.receiver_phase_storage_upper_bound == (
            original.receiver_phase_storage_upper_bound
        )
    shifted = _assess(
        replace(
            source,
            epi=tuple(value + Q(13, 7) for value in source.epi),
            phase=tuple(value + Q(11, 3) for value in source.phase),
        )
    )
    assert shifted.receiver_phase_storage_upper_bound == (
        original.receiver_phase_storage_upper_bound
    )
    assert shifted.initial_full_storage_bounds == original.initial_full_storage_bounds


@pytest.mark.parametrize("gap", (Q(0), Q(3), Q(-3)))
def test_zero_or_negative_speed_margin_is_unavailable_including_equality(source, gap):
    result = _assess(_with_gap(source, gap))
    assert result.contact_speed_lower_bound <= 0
    assert not result.speed_separation_certified
    assert result.integrated_contact_current_upper_bound is None
    assert result.receiver_phase_storage_upper_bound is None
    assert not result.whole_window_acute_winding_excluded
    assert result.status == "unavailable"
    assert result.reasons == ("contact_speed_separation_not_certified",)


def test_valid_fast_contact_bound_can_still_fail_to_exclude_formation(source):
    result = _assess(_with_gap(source, Q(4)))
    assert result.speed_separation_certified
    assert result.receiver_phase_storage_upper_bound > C5_PHASE_SECTOR_BARRIER
    assert not result.whole_window_acute_winding_excluded
    assert result.status == "unavailable"
    assert result.reasons == ("phase_storage_bound_does_not_exclude_acute_winding",)


def test_exact_comparison_resolves_both_sides_of_the_barrier_without_tolerance(source):
    # The threshold gap is irrational. Two nearby exact rational sources
    # distinguish the strict inequality without claiming approximate equality.
    with mp.workdps(90):
        c = mp.sqrt(mp.mpf(7) / 2 * 81 / 640)
        speed_threshold = (1 + mp.sqrt(1 + 4 * c)) / c
        gap_threshold = (speed_threshold + 4) * 3 / 4
        scale = 10**40
        below = Q(int(mp.floor(gap_threshold * scale)), scale)
        above = below + Q(1, scale)
    lower = _assess(_with_gap(source, below))
    upper = _assess(_with_gap(source, above))
    assert lower.receiver_phase_storage_upper_bound > C5_PHASE_SECTOR_BARRIER
    assert upper.receiver_phase_storage_upper_bound < C5_PHASE_SECTOR_BARRIER
    assert lower.status == "unavailable"
    assert upper.status == "excluded"


def test_cached_derived_evidence_is_rebuilt_and_labels_keep_their_order(source):
    ordinary = _assess(source)
    poisoned = replace(
        source,
        form_gradient=(Q(999),),
        form_rates=(),
        phase_rates=(I(555),),
        storage=I(-1),
        relative_resultant=(),
    )
    actual = _assess(poisoned)
    assert actual.initial_full_storage_bounds == ordinary.initial_full_storage_bounds
    assert actual.receiver_phase_storage_upper_bound == (
        ordinary.receiver_phase_storage_upper_bound
    )
    order = (8, 3, 0, 9, 4, 2, 6, 1, 5, 7)
    reordered = replace(
        _source(CYCLE_LEAF_EDGES, order=order, label=lambda i: f"n:{i}"),
        epi=tuple(source.epi[i] for i in order),
        phase=tuple(source.phase[i] for i in order),
    )
    result = _assess(reordered, cycle=tuple(f"n:{i}" for i in range(5)))
    assert result.cycle_indices == tuple(order.index(i) for i in range(5))
    assert result.contact_indices == tuple(
        (order.index(i), order.index(i + 5)) for i in range(5)
    )
    assert result.initial_full_storage_bounds == ordinary.initial_full_storage_bounds
    assert result.receiver_phase_storage_upper_bound == (
        ordinary.receiver_phase_storage_upper_bound
    )


@pytest.mark.parametrize(
    "horizon", (False, True, 0, -1, float("nan"), float("inf"), None)
)
def test_invalid_horizon_is_rejected_before_evaluation(source, horizon):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, scaled_horizon=horizon)


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_invalid_primitive_cannot_hide_behind_cached_report(source, field):
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, **{field: (False,) + getattr(source, field)[1:]}))


@pytest.mark.parametrize("node", (0, 5))
def test_each_form_block_must_be_exactly_uniform(source, node):
    forms = list(source.epi)
    forms[node] += Q(1, 10**100)
    with pytest.raises(ValueError, match="uniform"):
        _assess(replace(source, epi=tuple(forms)))


def test_receiver_phase_lifts_are_exact_and_other_domains_do_not_fall_through(source):
    with pytest.raises(ValueError, match="equal represented lifts"):
        _assess(replace(source, phase=(Q(1, 10**100),) + source.phase[1:]))
    with pytest.raises(ValueError):
        _assess(source, cycle=(0, 2, 1, 3, 4))
    with pytest.raises(ValueError):
        _assess(replace(source, capacity=(Q(0),) + (Q(1),) * 9))
    with pytest.raises(ValueError):
        _assess(
            replace(
                source,
                reference_model=RelationalExchangeModel(
                    1, epi_weight=1, phase_weight=1, phase_domain="regular"
                ),
            )
        )
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, law="native"))


def test_exact_receiver_transformation_and_storage_identity_from_complete_rows():
    # A generic instant, not another uniform initial state, exercises the
    # nonlinear ring terms and live leaves retained by the transformation.
    with mp.workdps(90):
        x = [mp.mpf(v) / 7 for v in (2, -5, 7, 1, -2, 4, 3, -8, 5, 0)]
        theta = [mp.mpf(v) / 9 for v in (1, 9, -17, 3, 27, 6, -2, 11, 8, -15)]
        neighbors = [set() for _ in range(10)]
        for i, j in CYCLE_LEAF_EDGES:
            neighbors[i].add(j)
            neighbors[j].add(i)
        x_rate = [
            mp.fsum(mp.sin(theta[j] - theta[i]) for j in neighbors[i])
            / len(neighbors[i])
            for i in range(10)
        ]
        phase_rate = [
            mp.fsum(x[i] - x[j] for j in neighbors[i]) / len(neighbors[i])
            for i in range(10)
        ]
        phase_acceleration = [
            mp.fsum(x_rate[i] - x_rate[j] for j in neighbors[i]) / len(neighbors[i])
            for i in range(10)
        ]
        a, b = mp.mpf(5), mp.mpf(-15)
        current = mp.matrix([b - x[i + 5] for i in range(5)])
        current_rate = mp.matrix([-x_rate[i + 5] for i in range(5)])
        y = mp.matrix([x[i] - a - current[i] / 3 for i in range(5)])
        y_rate = mp.matrix([x_rate[i] - current_rate[i] / 3 for i in range(5)])
        z_rate = mp.matrix([phase_rate[i] - (a - b) / 3 for i in range(5)])
        laplacian = mp.matrix(5)
        for i in range(5):
            laplacian[i, i] = 2
            laplacian[i, (i - 1) % 5] = -1
            laplacian[i, (i + 1) % 5] = -1
        sine = mp.matrix(
            [
                mp.sin(theta[(i - 1) % 5] - theta[i])
                + mp.sin(theta[(i + 1) % 5] - theta[i])
                for i in range(5)
            ]
        )
        a_matrix = laplacian + mp.eye(5)
        forcing = (laplacian + 4 * mp.eye(5)) * current / 9
        tolerance = mp.mpf("1e-85")
        assert mp.norm(y_rate - sine / 3) < tolerance
        assert mp.norm(z_rate - a_matrix * y / 3 - forcing) < tolerance
        phase_energy = mp.fsum(
            1 - mp.cos(theta[(i + 1) % 5] - theta[i]) for i in range(5)
        )
        phase_energy_rate = mp.fsum(
            mp.sin(theta[(i + 1) % 5] - theta[i])
            * (phase_rate[(i + 1) % 5] - phase_rate[i])
            for i in range(5)
        )
        direct = (y_rate.T * a_matrix * y)[0] + phase_energy_rate
        reconstructed = -(sine.T * forcing)[0]
        assert abs(direct - reconstructed) < tolerance
        storage = (y.T * a_matrix * y)[0] / 2 + phase_energy
        assert storage >= phase_energy >= 0
        assert mp.norm(sine) ** 2 <= 8 * phase_energy
        assert abs(direct) <= mp.sqrt(8 * storage) * 8 * mp.norm(current) / 9
        assert max(abs(v) for v in phase_acceleration) <= 2
        assert (
            max(
                abs(phase_acceleration[i + 5] - phase_acceleration[i]) for i in range(5)
            )
            <= 4
        )
