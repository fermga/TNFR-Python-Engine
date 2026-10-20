"""Exact-solution and independent-envelope controls without integration."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_support as support_owner
from tnfr.physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_support_exchange,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

FORMS = (Q(-1, 4), Q(1, 8), Q(1, 2))
PHASES = (Q(0), Q(1, 4), Q(-1, 8))


def _source(*, conductances=(2, 3), capacities=(2, 3, 5)):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, (form, phase, capacity) in enumerate(zip(FORMS, PHASES, capacities)):
        graph.add_node(node, EPI=float(form), theta=float(phase), nu_f=float(capacity))
    for edge, weight in zip(((0, 1), (1, 2)), conductances):
        graph.add_edge(*edge, weight=float(weight))
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_domain="regular"
        ),
    )


@pytest.fixture(scope="module")
def weighted_source():
    return _source()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


@pytest.mark.parametrize("exchange", (Q(1), Q(3, 2)))
def test_global_speed_radius_and_weighted_acceleration_match_independent_coefficients(
    weighted_source,
    mp,
    exchange,
):
    duration = Q(1, 3)
    # Constructor normalization and later exact coefficient admission are
    # distinct contracts. Supply a separate raw declaration without changing
    # the shared model; the consumer must retain its actual exchange scale.
    model = copy(weighted_source.reference_model)
    object.__setattr__(model, "phase_weight", exchange)
    source = replace(weighted_source, reference_model=model)
    report = source.finite_form_response_bounds(duration=duration)
    # The forms center to (-3/8,0,3/8), and the global capacity norm is sqrt38.
    expected_radius = 3 * mp.sqrt(2) / 8
    expected_speed = _mp(mp, exchange) * mp.sqrt(38) / mp.pi
    for upper, expected in (
        (report.initial_centered_form_radius_upper_bound, expected_radius),
        (report.global_form_speed_upper_bound, expected_speed),
        (
            report.whole_window_form_radius_upper_bound,
            expected_radius + expected_speed / 3,
        ),
    ):
        assert 0 <= _mp(mp, upper) - expected < mp.mpf("1e-30")
    # Independent weighted path calculation: d=(1,2,1), M=(2,3/2,5),
    # weighted strengths=(2,5,3). M_i*(sum_j M_j*s_j+d_i*M_i*s_i)
    # equals (23,51,225/2), retaining beta2 in the reciprocal phase row.
    coefficients = (Q(23), Q(51), Q(225, 2))
    radius = _mp(mp, report.whole_window_form_radius_upper_bound)
    unit_rates = (
        2 * mp.sin(mp.mpf(1) / 4) / mp.pi,
        3 * (-mp.sin(mp.mpf(1) / 4) - mp.sin(mp.mpf(3) / 8)) / (2 * mp.pi),
        5 * mp.sin(mp.mpf(3) / 8) / mp.pi,
    )
    initial_rates = tuple(_mp(mp, exchange) * value for value in unit_rates)
    for index, coefficient in enumerate(coefficients):
        expected_acceleration_bound = (
            mp.sqrt(2) * radius * _mp(mp, exchange**2 * coefficient) / (2 * mp.pi**2)
        )
        bounds = report.acceleration_bounds[index]
        assert bounds.lo == -bounds.hi
        assert 0 <= _mp(mp, bounds.hi) - expected_acceleration_bound < mp.mpf("1e-30")
        expected_remainder = expected_acceleration_bound / 18
        assert (
            0
            <= _mp(mp, report.form_remainder_upper_bounds[index]) - expected_remainder
            < mp.mpf("1e-30")
        )
        _contains(mp, report.initial_form_rates[index], initial_rates[index])
        tangent = _mp(mp, FORMS[index]) + initial_rates[index] / 3
        _contains(mp, report.endpoint_form_bounds[index], tangent - expected_remainder)
        _contains(mp, report.endpoint_form_bounds[index], tangent + expected_remainder)
    assert report.source.reference_model.storage_scale == 2
    assert report.source.reference_model.phase_weight == exchange
    assert weighted_source.reference_model.phase_weight == 1
    assert report.duration == duration and report.clock == "t"
    assert report.status == "enclosed"
    with pytest.raises(FrozenInstanceError):
        report.duration = Q(1)


@pytest.mark.parametrize("duration", (Q(1, 3), Q(5, 2)))
def test_zero_conductance_bare_path_has_exact_affine_solution_at_any_duration(
    duration, mp
):
    source = _source(conductances=(0, 0), capacities=(2, 0, 3))
    report = source.finite_form_response_bounds(duration=duration)
    # Zero weighted gradient fixes every phase exactly. Bare sine support
    # remains active, so the complete nonlinear law has this exact solution.
    velocities = (
        2 * mp.sin(mp.mpf(1) / 4) / mp.pi,
        mp.mpf(0),
        3 * mp.sin(mp.mpf(3) / 8) / mp.pi,
    )
    exact_endpoint = tuple(
        _mp(mp, form) + _mp(mp, duration) * velocity
        for form, velocity in zip(FORMS, velocities)
    )
    assert source.degrees == (1, 2, 1)
    assert source.phase_rate_numerators() == (0, 0, 0)
    assert report.acceleration_bounds == (I(0),) * 3
    assert report.form_remainder_upper_bounds == (0,) * 3
    assert report.initial_form_rates[0].lo > 0
    assert report.initial_form_rates[2].lo > 0
    assert report.endpoint_form_bounds[1] == I(FORMS[1])
    for enclosure, value in zip(report.endpoint_form_bounds, exact_endpoint):
        _contains(mp, enclosure, value)
    mean = sum(exact_endpoint) / 3
    centered_radius = mp.sqrt(sum((value - mean) ** 2 for value in exact_endpoint))
    assert centered_radius <= _mp(mp, report.whole_window_form_radius_upper_bound)
    # The arithmetic mean is moving; the radius argument must use a norm-one
    # centering projection rather than assert conservation of this mean.
    assert sum(velocities) / 3 > 0
    assert source.pressure[1].hi < 0


@pytest.mark.parametrize("capacities", ((0, 0, 0), (2, 0, 5)))
def test_zero_capacity_form_rows_are_exactly_frozen_despite_nonzero_pressure(
    capacities,
):
    source = _source(capacities=capacities)
    report = source.finite_form_response_bounds(duration=Q(3, 7))
    for index, capacity in enumerate(capacities):
        if capacity == 0:
            assert source.pressure[index].abs_max > 0
            assert report.initial_form_rates[index] == I(0)
            assert report.acceleration_bounds[index] == I(0)
            assert report.form_remainder_upper_bounds[index] == 0
            assert report.endpoint_form_bounds[index] == I(FORMS[index])
    if not any(capacities):
        assert report.global_form_speed_upper_bound == 0
        assert (
            report.whole_window_form_radius_upper_bound
            == report.initial_centered_form_radius_upper_bound
        )


def test_exact_common_form_and_phase_translations_transport_the_same_flow_enclosure(
    weighted_source,
):
    original = weighted_source.finite_form_response_bounds(duration=Q(1, 128))
    translated = replace(
        weighted_source,
        epi=tuple(value + Q(7, 8) for value in weighted_source.epi),
        phase=tuple(value + Q(3, 8) for value in weighted_source.phase),
    ).finite_form_response_bounds(duration=Q(1, 128))
    for field in (
        "initial_centered_form_radius_upper_bound",
        "global_form_speed_upper_bound",
        "whole_window_form_radius_upper_bound",
        "initial_form_rates",
        "acceleration_bounds",
        "form_remainder_upper_bounds",
    ):
        assert getattr(translated, field) == getattr(original, field)
    assert translated.endpoint_form_bounds == tuple(
        value + Q(7, 8) for value in original.endpoint_form_bounds
    )


@pytest.mark.parametrize(
    "duration", (0, -1, True, False, float("nan"), float("inf"), complex(1, 0))
)
def test_duration_requires_positive_finite_nonboolean_real(weighted_source, duration):
    with pytest.raises((TypeError, ValueError)):
        weighted_source.finite_form_response_bounds(duration=duration)


def test_exact_and_represented_durations_remain_distinct(weighted_source):
    exact = weighted_source.finite_form_response_bounds(duration=Q(1, 10))
    represented = weighted_source.finite_form_response_bounds(duration=0.1)
    assert exact.duration == Q(1, 10)
    assert represented.duration == Q.from_float(0.1)
    assert exact.duration != represented.duration


@pytest.mark.parametrize(
    "defect", ("capacity", "phase", "form", "conductance", "law", "loss")
)
def test_invalid_primitive_source_cannot_be_repaired_by_cached_bounds(
    weighted_source, defect
):
    changes = {
        "capacity": {"capacity": (True,) + weighted_source.capacity[1:]},
        "phase": {"phase": (float("inf"),) + weighted_source.phase[1:]},
        "form": {"epi": (float("nan"),) + weighted_source.epi[1:]},
        "conductance": {"conductances": (Q(-1), Q(3))},
        "law": {"law": "different_complete_law"},
        "loss": {
            "reference_model": RelationalExchangeModel(
                2, epi_weight=1, phase_domain="regular"
            )
        },
    }
    forged = replace(
        weighted_source, **changes[defect], form_rates=(I(0),) * 3, storage=I(0)
    )
    with pytest.raises((TypeError, ValueError)):
        forged.finite_form_response_bounds(duration=Q(1, 128))


def test_fresh_sdk_export_rebuilds_every_consumed_bound_and_keeps_atomic_failure(
    weighted_source, tmp_path
):
    report = weighted_source.finite_form_response_bounds(duration=Q(1, 128))
    source = replace(
        weighted_source,
        degrees=(99,) * 3,
        form_gradient=(Q(0),) * 3,
        form_rates=(I(0),) * 3,
        phase_rates=(I(0),) * 3,
        pressure=(I(0),) * 3,
        storage=I(0),
    )
    assert source.finite_form_response_bounds(duration=report.duration) == report
    forged = replace(
        report,
        source=source,
        initial_centered_form_radius_upper_bound=Q(0),
        global_form_speed_upper_bound=Q(0),
        whole_window_form_radius_upper_bound=Q(0),
        initial_form_rates=(I(0),) * 3,
        acceleration_bounds=(I(0),) * 3,
        form_remainder_upper_bounds=(Q(0),) * 3,
        endpoint_form_bounds=(I(999),) * 3,
        status="unavailable",
    )
    generic = relational_report_to_dict(forged)
    assert generic["report"] == report.to_dict()["report"]
    assert generic["report"]["duration"] == {"numerator": 1, "denominator": 128}
    assert generic["report"]["status"] == "enclosed"
    destination = tmp_path / "sine-finite-form-response.json"
    export_to_json(generic, destination)
    assert json_loads(destination.read_text(encoding="utf-8")) == generic
    original_bytes = destination.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(replace(forged, duration=False)), destination
        )
    assert destination.read_bytes() == original_bytes


def test_each_public_response_boundary_rebuilds_source_once(
    weighted_source, monkeypatch
):
    original = support_owner._rebuild_sine_support_comparison
    calls = []

    def counted(source):
        calls.append(source)
        return original(source)

    monkeypatch.setattr(support_owner, "_rebuild_sine_support_comparison", counted)
    poisoned = replace(
        weighted_source,
        degrees=(99,) * 3,
        form_gradient=(Q(0),) * 3,
        form_rates=(I(0),) * 3,
    )
    report = poisoned.finite_form_response_bounds(duration=Q(1, 128))
    assert len(calls) == 1 and calls[0] is poisoned
    assert report.source.degrees == (1, 2, 1)
    assert report.initial_form_rates[0].lo > 0

    calls.clear()
    exported = replace(report, source=poisoned).to_dict()
    assert len(calls) == 1 and calls[0] is poisoned
    assert exported["report"]["source"]["degrees"] == [1, 2, 1]

    calls.clear()
    accelerations = poisoned.conditional_form_acceleration_bounds(
        form_radius=report.whole_window_form_radius_upper_bound
    )
    assert len(calls) == 1 and calls[0] is poisoned
    assert accelerations == report.acceleration_bounds


def test_report_export_rejects_untrusted_source_without_custom_method_dispatch(
    weighted_source,
):
    class Fake:
        def finite_form_response_bounds(self, *, duration):
            raise AssertionError("untrusted finite-response dispatch")

    report = weighted_source.finite_form_response_bounds(duration=Q(1, 128))
    with pytest.raises(TypeError):
        SineSupportExchangeComparison.finite_form_response_bounds(
            Fake(), duration=Q(1, 128)
        )
    with pytest.raises(TypeError):
        replace(report, source=Fake()).to_dict()
    with pytest.raises(TypeError):
        type(report).to_dict(Fake())
