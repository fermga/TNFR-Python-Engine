"""Analytic entry storage and prospective finite-exit handoff obstruction."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tests.physics.test_sine_conservative_source_geometry import (
    _source as geometry_source,
)
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import assess_sine_conservative_handoff


def _source(*, omega=64, **kwargs):
    return geometry_source(
        forms=tuple(-3 * omega * (i - 2) for i in range(5)), **kwargs
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def report(source):
    return assess_sine_conservative_handoff(source, cycle=range(5))


def test_independent_exact_time_storage_and_exit_formulas(report):
    assert report.omega == 64 and report.orientation == 1
    assert report.initial_phase_velocity[:5] == (-128, -64, 0, 64, 128)
    assert report.scaled_exit_time == Q(5, 192)
    assert report.entry_acute_certified
    assert report.entry_subbarrier_certified
    assert report.forced_exit_certified
    assert report.handoff_obstruction_certified
    assert report.status == "certified_handoff_obstruction"
    assert report.unresolved == ()
    with mp.workdps(100):
        tau = 2 * mp.pi / (5 * 64)
        phase_minimum = 5 * (1 - mp.cos(2 * mp.pi / 5))
        barrier = 5 - 4 * mp.cos(3 * mp.pi / 8)
        expected = {
            "entry_cycle_gap_error_bound": 2 * tau**2,
            "entry_form_storage_upper_bound": 10 * tau**2,
            "entry_phase_storage_upper_bound": phase_minimum + 10 * tau**4,
            "entry_storage_upper_bound": phase_minimum + 10 * tau**2 + 10 * tau**4,
        }
        for name, value in expected.items():
            assert value <= _mp(getattr(report, name)) < value + mp.mpf("1e-33")
        assert report.scaled_entry_time_bounds.contains(Q(str(tau)))
        assert report.original_entry_time_bounds.contains(Q(str(mp.pi * tau)))
        assert report.original_exit_time_bounds.contains(Q(str(mp.pi * 5 / 192)))
        assert report.target_phase_storage_bounds.contains(Q(str(phase_minimum)))
        assert report.boundary_storage_bounds.contains(Q(str(barrier)))
        assert 0 < _mp(report.entry_acute_margin_lower_bound) <= mp.pi / 10 - 2 * tau**2
        assert (
            0
            < _mp(report.unavoidable_boundary_work_lower_bound)
            <= barrier - expected["entry_storage_upper_bound"]
        )
        assert mp.pi / 2 < _mp(report.exit_oriented_gap_bounds.lo)
        assert _mp(report.exit_oriented_gap_bounds.hi) < mp.pi
    assert report.entry_subbarrier_margin_lower_bound > Q(1, 100)
    assert (
        report.unavoidable_boundary_work_lower_bound
        == report.entry_subbarrier_margin_lower_bound
    )
    assert report.scaled_entry_time_bounds.hi < report.scaled_exit_time
    radius = 4 * report.scaled_entry_time_bounds.hi
    assert report.entry_relative_phase_rate_bounds == (
        I(64 - radius, 64 + radius),
    ) * 4 + (
        I(-256 - radius, -256 + radius),
    )
    assert all(bound.lo > 0 for bound in report.entry_relative_phase_rate_bounds[:4])
    assert report.entry_relative_phase_rate_bounds[-1].hi < 0


def test_zero_sum_phase_error_cancels_linear_storage_term(report):
    # This verifies the correlated target inequality with independent Taylor
    # algebra, not independent rectangular phase intervals with lost closure.
    with mp.workdps(100):
        alpha = 2 * mp.pi / 5
        radius = _mp(report.entry_cycle_gap_error_bound)
        errors = (radius, -radius, radius, -radius / 2, -radius / 2)
        assert abs(sum(errors)) < mp.mpf("1e-95")
        value = sum(1 - mp.cos(alpha + error) for error in errors)
        upper = 5 * (1 - mp.cos(alpha)) + sum(error**2 for error in errors) / 2
        assert value <= upper <= _mp(report.entry_phase_storage_upper_bound)


def test_reversal_and_signed_environment_preserve_the_same_obstruction(source, report):
    reversed_report = assess_sine_conservative_handoff(source, cycle=(4, 3, 2, 1, 0))
    sign_reversed = replace(source, epi=tuple(-value for value in source.epi))
    signed_report = assess_sine_conservative_handoff(sign_reversed, cycle=range(5))
    for actual in (reversed_report, signed_report):
        assert actual.orientation == -1
        assert actual.handoff_obstruction_certified
        assert actual.entry_storage_upper_bound == report.entry_storage_upper_bound
        assert actual.exit_oriented_gap_bounds == report.exit_oriented_gap_bounds
        assert actual.entry_relative_phase_rate_bounds == tuple(
            -bound for bound in report.entry_relative_phase_rate_bounds
        )


def test_common_origins_and_full_node_order_are_preserved(source, report):
    shifted = replace(
        source,
        epi=tuple(value + Q(3, 7) for value in source.epi),
        phase=(Q(4, 9),) * 10,
    )
    shifted_report = assess_sine_conservative_handoff(shifted, cycle=range(5))
    assert shifted_report.initial_phase_velocity == report.initial_phase_velocity
    assert shifted_report.entry_storage_upper_bound == report.entry_storage_upper_bound
    renamed = _source(order=tuple(reversed(range(10))), label=lambda i: f"n{i}")
    actual = assess_sine_conservative_handoff(
        renamed, cycle=tuple(f"n{i}" for i in range(5))
    )
    assert actual.handoff_obstruction_certified
    assert actual.cycle_indices == (9, 8, 7, 6, 5)
    assert actual.initial_phase_velocity == tuple(
        reversed(report.initial_phase_velocity)
    )
    assert (
        actual.unavoidable_boundary_work_lower_bound
        == report.unavoidable_boundary_work_lower_bound
    )


def test_low_speed_sufficient_bounds_remain_unavailable_without_rejecting_law():
    report = assess_sine_conservative_handoff(_source(omega=1), cycle=range(5))
    assert report.status == "unavailable"
    assert not report.entry_acute_certified
    assert not report.entry_subbarrier_certified
    assert not report.forced_exit_certified
    assert not report.handoff_obstruction_certified
    assert report.unavoidable_boundary_work_lower_bound is None
    assert "strict_subbarrier_entry_storage_unresolved" in report.unresolved


def test_predicates_are_independent_and_no_hardcoded_speed_threshold_is_used():
    report = assess_sine_conservative_handoff(_source(omega=10), cycle=range(5))
    assert report.entry_acute_certified
    assert report.forced_exit_certified
    assert not report.entry_subbarrier_certified
    assert not report.handoff_obstruction_certified
    assert report.unavoidable_boundary_work_lower_bound is None
    assert report.unresolved == ("strict_subbarrier_entry_storage_unresolved",)


def test_cached_fields_cannot_supply_handoff_evidence(source, report):
    poisoned = replace(
        source,
        storage=I(0),
        form_gradient=(999,) * 10,
        form_rates=(I(999),) * 10,
        phase_rates=(I(0),) * 10,
    )
    actual = assess_sine_conservative_handoff(poisoned, cycle=range(5))
    assert actual.initial_phase_velocity == report.initial_phase_velocity
    assert actual.entry_storage_upper_bound == report.entry_storage_upper_bound
    assert actual.handoff_obstruction_certified


def test_internal_chord_is_outside_induced_cycle_theorem(source):
    graph = nx.Graph(source.edges)
    graph.add_edge(0, 2)
    for i in graph:
        graph.nodes[i].update(EPI=source.epi[i], theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    chorded = bound_relational_sine_exchange(
        graph, reference_model=source.reference_model
    )
    with pytest.raises(ValueError, match="induced five-node"):
        assess_sine_conservative_handoff(chorded, cycle=range(5))


@pytest.mark.parametrize("forms", [(0,) * 10, (0,) * 5 + (384, 193, 0, -192, -384)])
def test_nonzero_exact_affine_velocity_profile_is_required(source, forms):
    with pytest.raises(ValueError, match="nonzero affine"):
        assess_sine_conservative_handoff(replace(source, epi=forms), cycle=range(5))


@pytest.mark.parametrize(
    "cycle",
    [
        (),
        (0, 1, 0),
        (0, 2, 1, 3, 4),
        (0, 1, 2),
        {0, 1, 2, 3, 4},
        (0, 1, 2, 3, "missing"),
    ],
)
def test_malformed_or_unsupported_cycle_rejects(source, cycle):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_conservative_handoff(source, cycle=cycle)


@pytest.mark.parametrize(
    "changes",
    [
        {"epi": (True,) + (0,) * 9},
        {"epi": (1,) + (0,) * 9},
        {"phase": (False,) + (0,) * 9},
        {"phase": (1,) + (0,) * 9},
        {"phase": (float("nan"),) + (0,) * 9},
        {"capacity": (0,) + (1,) * 9},
        {"capacity": (True,) * 10},
        {"degrees": (1,) * 10},
        {"law": "native"},
    ],
)
def test_authoritative_primitive_errors_are_not_repaired(source, changes):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_conservative_handoff(replace(source, **changes), cycle=range(5))


@pytest.mark.parametrize(
    "field,value",
    [
        ("epi_weight", False),
        ("epi_weight", Q(1, 2)),
        ("phase_weight", True),
        ("storage_scale", True),
        ("phase_domain", "acute"),
    ],
)
def test_complete_model_rejects_poisoned_or_different_law(source, field, value):
    model = copy(source.reference_model)
    object.__setattr__(model, field, value)
    with pytest.raises((TypeError, ValueError)):
        assess_sine_conservative_handoff(
            replace(source, reference_model=model), cycle=range(5)
        )


def test_report_projection_is_detached_and_preserves_running_work_scope(report):
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.sine-conservative-handoff.v1"
    assert payload["report"]["handoff_obstruction_certified"] is True
    assert (
        "positive_running_net_work_is_attained_by_first_exit_not_necessarily_at_deadline"
        in payload["report"]["scope"]
    )
    payload["report"]["cycle"].append(99)
    assert report.cycle == (0, 1, 2, 3, 4)
    with pytest.raises(FrozenInstanceError):
        report.status = "different"
