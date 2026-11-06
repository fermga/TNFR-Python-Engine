"""Exact export keeps asymptotic memory and finite-family capture distinct."""

from fractions import Fraction as Q

import pytest

from tnfr.physics.relational_cycle_memory import (
    bound_relational_cycle_memory,
    certify_relational_cycle_memory,
    certify_relational_cycle_memory_readout,
)
from tnfr.physics.relational_memory_contact import (
    certify_relational_memory_contact,
    certify_relational_memory_retention,
)
from tnfr.sdk import RelationalExchangeModel, relational_report_to_dict
from tnfr.utils.io import json_dumps, json_loads


@pytest.mark.parametrize("radius, admitted", ((Q(1, 64), True), (Q(1), False)))
def test_export_preserves_coefficients_basin_verdict_and_unknown_remainder(
    radius, admitted
):
    report = bound_relational_cycle_memory(
        model=RelationalExchangeModel(1),
        form_direction=(1, -1, 0, 0, 0),
        phase_direction=(0, 1, -1, 0, 0),
        capacity=Q(3, 2),
        amplitude_radius=radius,
    )
    projection = relational_report_to_dict(report)
    assert projection["report_type"] == "RelationalCycleMemoryBounds"
    data = projection["report"]
    assert data["basin_admitted"] is admitted
    assert data["capacity"] == {"numerator": 3, "denominator": 2}
    assert data["remainder_order"] == 3
    assert data["remainder_bound"] is None
    assert bool(data["unavailable_reasons"]) is (not admitted)
    assert report.quadratic_phase_shift_coefficient_bounds[0] > 0
    assert report.quadratic_left_port_form_rate_coefficient_bounds[1] < 0
    lower, upper = report.quadratic_phase_shift_coefficient_bounds
    assert data["quadratic_phase_shift_coefficient_bounds"] == [
        {"numerator": lower.numerator, "denominator": lower.denominator},
        {"numerator": upper.numerator, "denominator": upper.denominator},
    ]
    assert json_loads(json_dumps(projection)) == projection
    data["form_direction"][0]["numerator"] = 99
    assert report.form_direction[0] == 1


@pytest.mark.parametrize("blocks, admitted", ((0, False), (40, True)))
def test_finite_time_readout_export_keeps_clock_background_and_event_work(
    blocks, admitted
):
    certificate = certify_relational_cycle_memory_readout(decay_blocks=blocks)
    projection = relational_report_to_dict(certificate)
    assert projection["report_type"] == "RelationalMemoryReadoutCertificate"
    data = projection["report"]
    assert data["horizon"] == {"numerator": 4128 * blocks, "denominator": 1}
    radius = Q(8, 2 ** (20 + blocks))
    tail = 4096 * radius**2
    assert data["state_norm_upper_bound"] == {
        "numerator": radius.numerator,
        "denominator": radius.denominator,
    }
    assert data["mean_phase_tail_upper_bound"] == {
        "numerator": tail.numerator,
        "denominator": tail.denominator,
    }
    assert data["admitted"] is admitted
    assert data["memory"]["status"] == "admitted"
    assert data["rate_change_signs_separated"] is admitted
    assert bool(data["unavailable_reasons"]) is (not admitted)
    for case in data["cases"]:
        background = case["left_no_contact_form_rate_bounds"]
        low, high = (Q(item["numerator"], item["denominator"]) for item in background)
        assert low < 0 < high
        assert case["right_no_contact_form_rate_bounds"] == [
            {"numerator": 0, "denominator": 1},
            {"numerator": 0, "denominator": 1},
        ]
        assert case["contact_storage_bounds"] is not None
    assert json_loads(json_dumps(projection)) == projection


@pytest.mark.parametrize(
    "amplitude, phase_available, readout_available",
    (
        (Q(1, 2**20), True, True),
        (Q(1, 1024), False, False),
        (Q(1, 2**200), True, False),
    ),
)
def test_finite_certificate_export_retains_exact_error_and_separate_availability(
    amplitude, phase_available, readout_available
):
    certificate = certify_relational_cycle_memory(amplitude=amplitude)
    projection = relational_report_to_dict(certificate)
    assert projection["report_type"] == "RelationalFiniteMemoryCertificate"
    data = projection["report"]
    expected_error = 2**15 * amplitude**3
    assert data["phase_remainder_bound"] == {
        "numerator": expected_error.numerator,
        "denominator": expected_error.denominator,
    }
    assert data["phase_signs_separated"] is phase_available
    assert data["contact_signs_separated"] is readout_available
    assert data["admitted"] is (phase_available and readout_available)
    assert bool(data["unavailable_reasons"]) is (not data["admitted"])
    for case in data["cases"]:
        assert case["asymptotic"]["remainder_bound"] is None
        assert case["phase_sign_certified"] is phase_available
        assert case["contact_sign_certified"] is readout_available
        if phase_available:
            low, high = (
                Q(value["numerator"], value["denominator"])
                for value in case["limiting_phase_shift_bounds"]
            )
            assert low > 0 if case["form_sign"] == 1 else high < 0
    assert json_loads(json_dumps(projection)) == projection


@pytest.mark.parametrize("duration, admitted", ((Q(1, 4096), True), (Q(1, 3), False)))
def test_contact_export_retains_continuous_error_control_and_independent_work(
    duration, admitted
):
    certificate = certify_relational_memory_contact(duration=duration)
    projection = relational_report_to_dict(certificate)
    assert projection["report_type"] == "RelationalMemoryContactCertificate"
    data = projection["report"]
    assert data["duration"] == {
        "numerator": duration.numerator,
        "denominator": duration.denominator,
    }
    end_time = Q(165120) + duration
    assert data["end_time"] == {
        "numerator": end_time.numerator,
        "denominator": end_time.denominator,
    }
    assert data["readout"]["status"] == "admitted"
    assert data["admitted"] is admitted
    assert data["windings_preserved"]
    assert data["contact_induced_signs_separated"] is admitted
    assert bool(data["unavailable_reasons"]) is (not admitted)
    for case, original in zip(data["cases"], certificate.cases):
        remainder = case["accumulated_form_remainder_bound"]
        assert (
            Q(remainder["numerator"], remainder["denominator"])
            == (original.accumulated_form_remainder_bound)
            > 0
        )
        assert case["right_no_contact_form_change_bounds"] == [
            {"numerator": 0, "denominator": 1},
            {"numerator": 0, "denominator": 1},
        ]
        cost = tuple(
            Q(value["numerator"], value["denominator"])
            for value in case["contact_storage_bounds"]
        )
        loss = case["continuous_loss_upper_bound"]
        assert cost == original.contact_storage_bounds
        assert (
            Q(loss["numerator"], loss["denominator"])
            == original.continuous_loss_upper_bound
        )
    work = data["required_event_work_upper_bound"]
    assert Q(work["numerator"], work["denominator"]) > 0
    assert json_loads(json_dumps(projection)) == projection


def test_retention_export_keeps_regional_record_capture_and_cut_budget():
    certificate = certify_relational_memory_retention()
    projection = relational_report_to_dict(certificate)
    assert projection["report_type"] == "RelationalMemoryRetentionCertificate"
    data = projection["report"]
    assert data["admitted"]
    assert data["contact"]["status"] == "admitted"
    assert data["receiver_mean_signs_separated"] and data["both_rings_captured"]
    for case, original in zip(data["cases"], certificate.cases):
        mean = tuple(
            Q(value["numerator"], value["denominator"])
            for value in case["receiver_persistent_mean_bounds"]
        )
        assert mean == original.receiver_persistent_mean_bounds
        assert mean[0] > 0 if case["form_sign"] == 1 else mean[1] < 0
        assert case["no_contact_receiver_mean_bounds"] == [
            {"numerator": 0, "denominator": 1},
            {"numerator": 0, "denominator": 1},
        ]
        jump = tuple(
            Q(value["numerator"], value["denominator"])
            for value in case["removal_storage_change_bounds"]
        )
        assert jump == original.removal_storage_change_bounds
        assert jump[1] < 0
    assert json_loads(json_dumps(projection)) == projection
    data["cases"][0]["receiver_persistent_mean_bounds"][0]["numerator"] = 0
    assert certificate.cases[0].receiver_persistent_mean_bounds[0] > 0
