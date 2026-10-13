"""Manufactured-record controls for the certified free-probe reader.

These small analytic records test a numerical observation functional. They
are not TNFR responses, acquired sources, fitted laws or physical readings.
"""

from collections import UserDict
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial

import mpmath as mp
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics._organized_free_phase_record import _read_organized_free_phase_record


@pytest.fixture(scope="module", autouse=True)
def no_response_source_or_trajectory_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _manufactured_record(*, intervals=4, residual=Q(0), origin=Q(0)):
    """Exact rational samples of an independently bounded Taylor polynomial."""
    gamma = (1 / (1023 * pi_interval())).midpoint
    times = tuple(Q(index, intervals) for index in range(intervals + 1))
    # On [0,1], the alternating remainder is <= 1/41!, independently
    # of the exponential/interval implementation consumed by the reader.
    heat = tuple(
        12 * sum(((-time) ** degree / factorial(degree) for degree in range(41)), Q(0))
        for time in times
    )
    return dict(
        sample_times=times,
        raw_form_samples=tuple((value + residual) / gamma for value in heat),
        gamma_hat=gamma,
        initial_phase_readout=origin,
        form_error_bound=Q(0),
        phase_error_bound=Q(1, 10**6),
        quadrature_subdivisions=4,
        raw_form_error_bound=Q(1, 10000),
        exposure_time_error_bound=Q(0),
        kinetic_relative_error_bound=Q(1, 10**18),
        initial_phase_readout_error_bound=abs(origin),
        numerical_coefficient_error_bound=Q(1, 10**10),
        harmonic=3,
    )


@pytest.fixture(scope="module")
def heat_reading():
    arguments = _manufactured_record()
    return arguments, _read_organized_free_phase_record(**arguments)


@pytest.fixture(scope="module")
def shifted_reading():
    arguments = _manufactured_record(residual=Q(1, 10**8), origin=Q(1, 7))
    return arguments, _read_organized_free_phase_record(**arguments)


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _constant_residual_midpoint_oracle(arguments, endpoint, residual):
    """Evaluate closed analytic expressions, with no state evolution/root fit.

    The supplied endpoint comes from a certified bracket. The ideal constant
    residual differs from the polynomial samples by at most 12/41!, far
    below the reader's declared numerical allowance.
    """
    with mp.workdps(80):
        gamma = _mp(arguments["gamma_hat"])
        origin = _mp(arguments["initial_phase_readout"])
        constant = _mp(residual)
        stop = _mp(endpoint)
        n = arguments["harmonic"]
        total_cells = (len(arguments["sample_times"]) - 1) * arguments[
            "quadrature_subdivisions"
        ]

        def values(time):
            heat = 12 * mp.exp(-time)
            velocity = heat + constant
            phase = origin + 12 * (1 - mp.exp(-time)) + constant * time
            energy = (heat * constant + constant**2 / 2) / gamma**2
            factor = n * velocity * mp.cos(n * phase) - 2 * mp.sin(n * phase)
            return energy * factor, energy * mp.sin(n * phase)

        total = mp.mpf(0)
        for index in range(total_cells):
            left = mp.mpf(index) / total_cells
            if left >= stop:
                break
            right = min(mp.mpf(index + 1) / total_cells, stop)
            total += (right - left) * values((left + right) / 2)[0]
        endpoint_term = values(stop)[1] - values(mp.mpf(0))[1]
        return +(total - endpoint_term) / mp.pi, +endpoint_term / mp.pi


def test_heat_record_certifies_numerics_without_asserting_a_law_response(heat_reading):
    arguments, report = heat_reading
    assert report.status == "certified"
    assert report.endpoint_bounds is not None
    assert 0 < report.endpoint_bounds.lo < report.endpoint_bounds.hi < 1
    assert report.observation_bounds.contains(0)
    assert report.coefficient_bounds.contains(report.observation_bounds)
    assert (
        report.numerical_error_upper_bound
        <= arguments["numerical_coefficient_error_bound"]
    )
    assert report.quadrature_evaluations <= 16
    assert report.budget.sample_count == 5
    assert report.sample_times == arguments["sample_times"]
    assert report.raw_form_samples == arguments["raw_form_samples"]
    # The manufactured heat record does not supply the missing current law.
    # Wide observation uncertainty is separate from successful computation.
    assert report.coefficient_bounds.width > Q(1, 2187)
    with pytest.raises(FrozenInstanceError):
        report.status = "physical_measurement"


def test_nonzero_endpoint_correction_matches_independent_clipped_midpoints(
    shifted_reading,
):
    arguments, report = shifted_reading
    assert report.status == "certified"
    endpoint = report.endpoint_bounds.midpoint
    observed, endpoint_term = _constant_residual_midpoint_oracle(
        arguments, endpoint, Q(1, 10**8)
    )
    with mp.workdps(80):
        assert (
            _mp(report.observation_bounds.lo)
            <= observed
            <= _mp(report.observation_bounds.hi)
        )
        assert abs(endpoint_term) > mp.mpf("0.01")
        assert not (
            _mp(report.observation_bounds.lo)
            <= observed + endpoint_term
            <= _mp(report.observation_bounds.hi)
        )
        # This bracket check uses the independent exact analytic increment.
        lower, upper = map(_mp, (report.endpoint_bounds.lo, report.endpoint_bounds.hi))
        constant = mp.mpf(1) / 10**8
        assert 12 * (1 - mp.exp(-lower)) + constant * lower < 2 * mp.pi
        assert 12 * (1 - mp.exp(-upper)) + constant * upper > 2 * mp.pi
    # R'+2R=c*v/gamma_hat^2 implies zero continuous full-turn harmonic
    # integral for this manufactured curve. Quadrature and model errors
    # remain outside the numerical-functional interval itself.
    assert report.coefficient_bounds.contains(0)


def test_observation_interval_does_not_add_the_numerical_allowance_twice(heat_reading):
    _, report = heat_reading
    budget = report.budget
    mathematical_error = (
        budget.fine_harmonic_error_upper_bound
        + (
            budget.observation_numerator_error_upper_bound
            + budget.quadrature_numerator_error_upper_bound
        )
        / pi_interval().lo
    )
    lower_added = report.observation_bounds.lo - report.coefficient_bounds.lo
    upper_added = report.coefficient_bounds.hi - report.observation_bounds.hi
    assert lower_added >= mathematical_error
    assert upper_added >= mathematical_error
    assert (
        lower_added < mathematical_error + budget.numerical_coefficient_error_bound / 2
    )
    assert (
        upper_added < mathematical_error + budget.numerical_coefficient_error_bound / 2
    )


def test_record_values_cannot_borrow_bound_constants_when_residuals_violate_them():
    arguments = _manufactured_record()
    raw = list(arguments["raw_form_samples"])
    raw[0] += Q(1, 1000) / arguments["gamma_hat"]
    report = _read_organized_free_phase_record(**dict(arguments, raw_form_samples=raw))
    assert report.status == "incompatible_record"
    assert "initial decay" in report.reason
    assert report.coefficient_bounds is None

    # These alternating residuals each fit C+E, while the adjacent jump
    # exceeds 2*C/N+2E; an amplitude-only check would be insufficient.
    raw = list(arguments["raw_form_samples"])
    raw[1] += Q(1, 10**7) / arguments["gamma_hat"]
    raw[2] -= Q(1, 10**7) / arguments["gamma_hat"]
    report = _read_organized_free_phase_record(**dict(arguments, raw_form_samples=raw))
    assert report.status == "incompatible_record"
    assert "adjacent" in report.reason
    assert report.coefficient_bounds is None


def test_calibration_separation_and_boundary_overlap_have_distinct_outcomes():
    arguments = _manufactured_record()
    incompatible = _read_organized_free_phase_record(
        **dict(arguments, gamma_hat=2 * arguments["gamma_hat"])
    )
    assert incompatible.status == "incompatible_record"
    assert incompatible.coefficient_bounds is None
    kappa = Q(1, 10**5)
    unresolved = _read_organized_free_phase_record(
        **dict(
            arguments,
            gamma_hat=(1 + kappa) / (1023 * pi_interval().midpoint),
            kinetic_relative_error_bound=kappa,
        )
    )
    assert unresolved.status == "unresolved_numeric"
    assert unresolved.coefficient_bounds is None


def test_initial_lift_contradiction_is_not_repaired_by_wrapping():
    arguments = _manufactured_record()
    report = _read_organized_free_phase_record(
        **dict(arguments, initial_phase_readout=2 * pi_interval().midpoint)
    )
    assert report.status == "incompatible_record"
    assert "initial phase" in report.reason
    assert report.coefficient_bounds is None


def test_resource_limits_return_unavailable_coefficients_without_relabeling_data():
    arguments = _manufactured_record()
    endpoint = _read_organized_free_phase_record(**arguments, max_endpoint_iterations=1)
    assert endpoint.status == "unresolved_endpoint"
    assert endpoint.coefficient_bounds is None
    assert endpoint.endpoint_iterations <= 1
    limited = _read_organized_free_phase_record(
        **arguments, max_quadrature_evaluations=1
    )
    assert limited.status == "work_limit"
    assert limited.coefficient_bounds is None
    assert limited.quadrature_evaluations <= 1
    precision = _read_organized_free_phase_record(
        **dict(arguments, numerical_coefficient_error_bound=Q(1, 10**80))
    )
    assert precision.status in {"unresolved_numeric", "unresolved_endpoint"}
    assert precision.coefficient_bounds is None


def test_primitive_fractions_and_input_copies_are_normalized(heat_reading):
    arguments, expected = heat_reading
    altered = dict(arguments)
    for name in ("sample_times", "raw_form_samples"):
        # Counts and records need not have fixed-width denominators here;
        # normalize a representable zero/one endpoint separately below.
        altered[name] = list(arguments[name])
    altered["sample_times"][0] = Q(np.int64(0), np.int64(1))
    altered["phase_error_bound"] = Q(np.int64(1), np.int64(10**6))
    report = _read_organized_free_phase_record(**altered)
    assert report == expected
    assert type(report.sample_times[0].numerator) is int
    assert type(report.budget.phase_error_bound.denominator) is int
    altered["raw_form_samples"][0] = Q(0)
    assert report.raw_form_samples[0] == arguments["raw_form_samples"][0]


def test_prior_reports_cannot_replace_primitive_readings_or_error_bounds(heat_reading):
    arguments, report = heat_reading
    with pytest.raises(TypeError):
        _read_organized_free_phase_record(**dict(arguments, raw_form_samples=report))
    with pytest.raises(TypeError):
        _read_organized_free_phase_record(
            **dict(arguments, raw_form_error_bound=report.budget)
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("gamma_hat", False),
        ("gamma_hat", 0),
        ("gamma_hat", -1),
        ("gamma_hat", float("nan")),
        ("initial_phase_readout", float("inf")),
        ("raw_form_samples", ()),
        ("raw_form_samples", (1,)),
        ("sample_times", (0, 1)),
        ("sample_times", (0, Q(1, 4), Q(1, 4), Q(3, 4), 1)),
        ("sample_times", (0, Q(1, 4), Q(1, 2), 1, Q(3, 4))),
        ("quadrature_subdivisions", True),
        ("harmonic", Q(3)),
        ("max_endpoint_iterations", False),
        ("max_quadrature_evaluations", 0),
    ),
)
def test_malformed_primitive_reader_inputs_reject_before_evaluation(field, value):
    with pytest.raises((TypeError, ValueError)):
        _read_organized_free_phase_record(
            **dict(_manufactured_record(), **{field: value})
        )


@pytest.mark.parametrize("bad_reading", (True, float("nan"), float("inf")))
def test_each_authoritative_reading_is_admitted(bad_reading):
    arguments = _manufactured_record()
    raw = list(arguments["raw_form_samples"])
    raw[2] = bad_reading
    with pytest.raises((TypeError, ValueError)):
        _read_organized_free_phase_record(**dict(arguments, raw_form_samples=raw))


def test_represented_timestamps_are_not_snapped_to_the_declared_grid():
    arguments = _manufactured_record(intervals=10)
    timestamps = [index / 10 for index in range(11)]
    assert Q(timestamps[1]) != Q(1, 10)
    with pytest.raises(ValueError):
        _read_organized_free_phase_record(**dict(arguments, sample_times=timestamps))


@pytest.mark.parametrize("field", ("sample_times", "raw_form_samples"))
@pytest.mark.parametrize("container", (set, frozenset, UserDict))
def test_unordered_or_mapping_records_cannot_acquire_a_sample_order(field, container):
    arguments = _manufactured_record(intervals=1)
    values = arguments[field]
    source = container({value: None for value in values})
    with pytest.raises(TypeError, match="ordered sequence"):
        _read_organized_free_phase_record(**dict(arguments, **{field: source}))
