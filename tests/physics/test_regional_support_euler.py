"""Exact finite regional endpoint accounting, without trajectory execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.physics.support_transport import _from_data, observe_regional_support_euler


def _path(*, epi=(0, 1, 5), capacity=(1, 2, 1), pressure=(1, F(3, 2), -4)):
    return _from_data(
        ("a", "b", "outside"),
        ((0, 1, 1), (1, 0, 1), (1, 2, 1), (2, 1, 1)),
        ((1,), (0, 2), (1,)),
        epi,
        capacity,
        pressure,
    )


def _observe(
    before=None, after=None, *, dt=F(1, 2), region=("a", "b"), forcing=(0, 0, 0)
):
    before = _path() if before is None else before
    after = replace(before, epi=(F(1, 2), F(5, 2), F(3))) if after is None else after
    return observe_regional_support_euler(
        before, after, region, dt=dt, epi_weight=1, forcing=forcing
    )


def test_hand_path_exact_euler_keeps_boundary_growth_and_quadratic_term():
    result = _observe()
    assert result.expected_epi == (F(1, 2), F(5, 2), 3)
    assert result.state_defect == (0, 0, 0)
    assert result.balance.internal_dissipation == 1
    assert result.balance.variance_boundary_rate == 2
    assert result.balance.stored_variance_rate == 1
    assert result.regional_rate_mean == 2 and result.expected_mean == F(3, 2)
    assert result.after_weighted_total == 3 and result.after_variance == 1
    assert result.mass_drift_term == result.mass_change == 2
    assert result.mass_defect_term == 0
    assert result.variance_drift_term == F(1, 2)
    assert result.variance_quadratic_term == F(1, 4)
    assert (
        result.variance_defect_linear_term == result.variance_defect_quadratic_term == 0
    )
    assert result.variance_change == F(3, 4)
    assert result.mass_identity_residual == result.variance_identity_residual == 0


def test_uniform_endpoint_shift_changes_mean_only_relative_to_euler_reference():
    before = _path()
    exact = _observe(before)
    shifted = _observe(
        before, replace(before, epi=tuple(x + 7 for x in exact.expected_epi))
    )
    assert shifted.state_defect == (7, 7, 7) and shifted.defect_mean == 7
    assert shifted.mass_defect_term == 14
    assert shifted.after_mean == exact.after_mean + 7
    assert shifted.after_variance == exact.after_variance
    assert shifted.variance_change == exact.variance_change
    assert (
        shifted.variance_defect_linear_term
        == shifted.variance_defect_quadratic_term
        == 0
    )


def test_clipped_endpoint_exposes_signed_linear_and_nonnegative_quadratic_defects():
    before = _path()
    clipped = _observe(before, replace(before, epi=(F(1, 2), F(2), F(3))))
    assert clipped.state_defect == (0, F(-1, 2), 0)
    assert clipped.defect_mean == F(-1, 4) and clipped.mass_defect_term == F(-1, 2)
    assert clipped.mass_change == F(3, 2)
    assert clipped.variance_defect_linear_term == F(-1, 2)
    assert clipped.variance_defect_quadratic_term == F(1, 16)
    assert clipped.after_variance == F(9, 16)
    assert clipped.variance_change == F(5, 16)
    assert clipped.mass_identity_residual == clipped.variance_identity_residual == 0


def test_stored_pressure_discrepancy_is_separate_from_exact_endpoint_defect():
    before = _path(pressure=(0, 0, 0))
    result = _observe(before, before)
    assert result.expected_epi == before.epi and result.state_defect == (0, 0, 0)
    assert result.balance.stored_pressure_defect != (0, 0, 0)
    assert result.balance.model_variance_rate == 1
    assert result.balance.variance_defect_rate == -1
    assert result.mass_change == result.variance_change == 0
    assert "No causal execution" in result.scope


def test_after_pressure_is_not_used_to_replace_the_held_initial_rate():
    ordinary = _observe()
    updated = _observe(
        after=replace(ordinary.after, stored_pressure=(F(100), F(-20), F(7)))
    )
    for field in (
        "expected_epi",
        "state_defect",
        "mass_drift_term",
        "variance_drift_term",
        "variance_quadratic_term",
        "mass_change",
        "variance_change",
    ):
        assert getattr(ordinary, field) == getattr(updated, field)
    assert ordinary.after.rate != updated.after.rate


def test_zero_duration_can_account_for_a_detached_event_without_claiming_integration():
    before = _path()
    after = replace(before, epi=(F(2), F(1), F(5)))
    result = _observe(before, after, dt=0)
    assert result.expected_epi == before.epi and result.state_defect == (2, 0, 0)
    assert (
        result.mass_drift_term
        == result.variance_drift_term
        == result.variance_quadratic_term
        == 0
    )
    assert result.mass_change == result.mass_defect_term == 2
    assert result.variance_defect_linear_term == -1
    assert result.variance_defect_quadratic_term == 1
    assert result.variance_change == 0


def test_nonuniform_metric_direct_endpoint_quadratic_matches_complete_decomposition():
    before = _path(
        epi=(F(-2, 3), F(7, 5), F(11, 7)),
        capacity=(2, 3, 5),
        pressure=(F(1, 3), F(-2, 5), F(4, 7)),
    )
    after = replace(before, epi=(F(3, 11), F(-5, 13), F(17, 19)))
    result = _observe(before, after, dt=F(7, 11), forcing=(F(1, 5), F(-2, 3), F(4, 9)))
    # For two coordinates the weighted centered variance is
    # h0*h1/(2*(h0+h1))*(x0-x1)^2, independently of the observer's centering.
    h0, h1 = F(1, 2), F(2, 3)
    coefficient = h0 * h1 / (2 * (h0 + h1))
    direct = coefficient * (
        (after.epi[0] - after.epi[1]) ** 2 - (before.epi[0] - before.epi[1]) ** 2
    )
    assert result.variance_change == direct
    assert direct == (
        result.variance_drift_term
        + result.variance_quadratic_term
        + result.variance_defect_linear_term
        + result.variance_defect_quadratic_term
    )
    assert result.mass_change == h0 * (after.epi[0] - before.epi[0]) + h1 * (
        after.epi[1] - before.epi[1]
    )
    assert result.mass_identity_residual == result.variance_identity_residual == 0


def test_singleton_region_has_no_centered_variance_despite_total_change():
    result = _observe(region=("outside",))
    assert result.mass_change == -2
    assert result.after_variance == result.variance_change == 0
    assert result.variance_drift_term == result.variance_quadratic_term == 0
    assert (
        result.variance_defect_linear_term == result.variance_defect_quadratic_term == 0
    )


@pytest.mark.parametrize(
    "change",
    (
        {"nodes": ("b", "a", "outside")},
        {"capacity": (F(1), F(3), F(1))},
        {"conductance": ((0, 1, F(2)), (1, 0, F(2)), (1, 2, F(1)), (2, 1, F(1)))},
        {"support_neighbors": ((0, 1), (0, 2), (1,))},
        {"support_neighbors": ((1,), (2, 0), (1,))},
    ),
)
def test_changed_primitive_node_space_support_or_metric_is_rejected(change):
    before = _path()
    with pytest.raises(ValueError, match="fixed full"):
        _observe(before, replace(before, **change))


@pytest.mark.parametrize("dt", (-1, float("nan"), float("inf"), True, "0.5"))
def test_dt_must_be_finite_nonnegative_real(dt):
    with pytest.raises((TypeError, ValueError)):
        _observe(dt=dt)


@pytest.mark.parametrize("region", ((), ("a", "b", "outside"), ("a", "a"), {"a", "b"}))
def test_region_admission_is_reused(region):
    with pytest.raises((TypeError, ValueError)):
        _observe(region=region)


def test_forged_caches_are_ignored_but_bad_endpoint_primitives_fail():
    ordinary = _observe()
    cached = {
        "rate": (F(99),) * 3,
        "epi_gradient": (F(99),) * 3,
        "dirichlet_energy": F(99),
        "energy_rate": F(99),
    }
    assert (
        _observe(replace(ordinary.before, **cached), replace(ordinary.after, **cached))
        == ordinary
    )
    with pytest.raises(ValueError):
        _observe(after=replace(ordinary.after, epi=(0, float("nan"), 0)))
    with pytest.raises(TypeError, match="SupportTransportSnapshot"):
        _observe(after={"epi": ordinary.after.epi})


def test_inputs_remain_unchanged_and_frozen_output_detaches_region_and_forcing():
    before = _path()
    after = replace(before, epi=(F(1, 2), F(5, 2), F(3)))
    saved = deepcopy((before, after))
    region, forcing = ["a", "b"], [0, 0, 0]
    result = _observe(before, after, region=region, forcing=forcing)
    assert (before, after) == saved
    region.reverse()
    forcing[0] = 9
    assert result.balance.region == ("a", "b") and result.balance.forcing == (0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        result.dt = F(3)
