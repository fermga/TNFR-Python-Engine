"""Finite-record budgets and detached readout for the fixed free phase probe.

The proof owner is ORGANIZED_FREE_PHASE_PROBE.md#free-phase-probe-finite-records.
All bounds are uniform over the supplied 0 <= deformation <= 1 family. The
reader consumes primitive samples; it neither generates a trajectory nor
authenticates the supplied preparation, law or observation error evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import (
    EXP_NEG_UNIT_METHOD,
    INTERVAL_METHOD,
    I,
    cos,
    exp_neg_unit,
    pi_interval,
    sin,
)
from ._organized_free_phase_probe import (
    _derive_organized_free_phase_probe,
    _OrganizedFreePhaseProbe,
)


@dataclass(frozen=True)
class _OrganizedFreePhaseRecordBudget:
    """Conditional coefficient uncertainty and cost without observed readings.

    Raw form error concerns the observed relative child mean X_A-X_B.
    Exposure-time error uses s=a*tau/2; every actual exposure must remain in
    [0,1]. The constant kinetic readout calibration obeys
    abs(gamma_hat/gamma-1) <= kinetic_relative_error_bound and does not change
    the supplied dynamics or authorize an unknown sampling-clock scale.

    The exact preparation still has v(0)=12 and delta(0)=0. Its initial phase
    readout error affects harmonic registration, while the reconstructed turn
    is an increment from that same initial readout. Numerical coefficient
    error must cover certified exponential, trigonometric, endpoint and
    arithmetic errors of the declared algorithm; it is not inferred here.
    A zero numerical allowance describes exact ideal evaluation only, not a
    generally attainable finite floating-point calculation.

    The family bound stored in ``probe_bound`` uses deformation=1 only to
    obtain uniform majorants, not as a fitted or identified constitutive law.
    A finite coefficient error is not a law-discrimination verdict.
    """

    form_error_bound: Q
    phase_error_bound: Q
    sample_intervals: int
    quadrature_subdivisions: int
    raw_form_error_bound: Q
    exposure_time_error_bound: Q
    kinetic_relative_error_bound: Q
    initial_phase_readout_error_bound: Q
    numerical_coefficient_error_bound: Q
    harmonic: int
    probe_bound: _OrganizedFreePhaseProbe
    sample_step: Q
    quadrature_step: Q
    scaled_current_forcing_upper_bound: Q
    sample_velocity_error_upper_bound: Q
    decay_residual_second_derivative_upper_bound: Q
    velocity_reconstruction_error_upper_bound: Q
    phase_reconstruction_error_upper_bound: Q
    endpoint_time_error_upper_bound: Q
    reconstructed_velocity_lower_bound: Q
    full_turn_reconstruction_margin: Q
    kinetic_gamma_squared_lower_bound: Q
    true_scaled_energy_residual_upper_bound: Q
    normalization_relative_error_upper_bound: Q
    energy_residual_error_upper_bound: Q
    interpolated_decay_residual_upper_bound: Q
    interpolated_decay_residual_slope_upper_bound: Q
    reconstructed_velocity_upper_bound: Q
    reconstructed_velocity_rate_upper_bound: Q
    reconstructed_energy_residual_upper_bounds: tuple[Q, Q, Q]
    harmonic_factor_derivative_upper_bounds: tuple[Q, Q, Q]
    integrand_second_derivative_upper_bound: Q
    quadrature_numerator_error_upper_bound: Q
    observation_numerator_error_upper_bound: Q
    fine_harmonic_error_upper_bound: Q
    coefficient_error_upper_bound: Q
    sample_count: int
    quadrature_evaluation_count_upper_bound: int
    conditional_on_preparation_and_error_bounds: bool = True
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "uniform_over_the_supplied_deformation_interval_zero_to_one",
        "unchanged_two_C9_dynamics_support_preparation_and_fine_identity_premises",
        "uniform_nominal_grid_on_zero_to_one_in_the_declared_scaled_clock",
        "all_actual_sample_exposures_remain_in_zero_to_one",
        "reading_and_constant_kinetic_calibration_bounds_are_supplied_not_inferred",
        "no_unknown_clock_scale_is_admitted_by_the_exposure_time_allowance",
        "subtract_known_homogeneous_decay_before_piecewise_linear_interpolation",
        "reconstruct_phase_increment_and_stop_at_its_first_complete_turn",
        "retain_initial_phase_readout_error_and_both_energy_boundary_terms",
        "composite_midpoint_cells_never_cross_interpolation_knots",
        "numerical_allowance_covers_endpoint_and_function_evaluation_errors",
        "prospective_budget_without_record_evaluation_or_source_authentication",
        "no_law_selection_sensor_capability_or_physical_admission_verdict",
    )


def _positive_count(value, label):
    if type(value) is not int:
        raise TypeError(f"{label} must be a positive ordinary integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _bound_organized_free_phase_record(
    *,
    form_error_bound,
    phase_error_bound,
    sample_intervals,
    quadrature_subdivisions,
    raw_form_error_bound,
    exposure_time_error_bound,
    kinetic_relative_error_bound,
    initial_phase_readout_error_bound,
    numerical_coefficient_error_bound,
    harmonic,
) -> _OrganizedFreePhaseRecordBudget:
    """Rebuild a sufficient finite-record budget from primitive allowances.

    Counts and harmonic are positive ordinary integers. All errors are
    nonnegative exact or represented reals, with kinetic relative error < 1.
    The shared fine-state admission and sufficient reconstructed-turn margins
    must hold strictly. Failure rejects this certificate, not the underlying
    model or all possible observation schemes. No tolerance, derivative bound,
    numerical accuracy or unknown constitutive parameter is fitted to data.
    """
    intervals = _positive_count(sample_intervals, "sample_intervals")
    subdivisions = _positive_count(quadrature_subdivisions, "quadrature_subdivisions")
    n = _positive_count(harmonic, "harmonic")
    raw_error = _exact(raw_form_error_bound, "raw_form_error_bound")
    exposure_error = _exact(exposure_time_error_bound, "exposure_time_error_bound")
    calibration_error = _exact(
        kinetic_relative_error_bound, "kinetic_relative_error_bound"
    )
    initial_error = _exact(
        initial_phase_readout_error_bound, "initial_phase_readout_error_bound"
    )
    numerical_error = _exact(
        numerical_coefficient_error_bound, "numerical_coefficient_error_bound"
    )
    if (
        min(
            raw_error, exposure_error, calibration_error, initial_error, numerical_error
        )
        < 0
    ):
        raise ValueError("record error bounds must be nonnegative")
    if calibration_error >= 1:
        raise ValueError("kinetic_relative_error_bound must be strictly below one")

    probe = _derive_organized_free_phase_probe(
        deformation=1,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
    )
    gamma = probe.gamma_bounds
    pi = pi_interval()
    j0, j1, j2 = probe.current_derivative_bounds
    fine_energy = probe.fine_energy_upper_bound
    h = Q(1, intervals)
    forcing = gamma.hi**2 * j0
    sample_error = 12 * calibration_error + (1 + calibration_error) * (
        gamma.hi * raw_error + (12 + forcing) * exposure_error
    )
    residual_second = gamma.hi**2 * (2 * j0 + 12 * j1) + Q(5000, 3) * j2 * fine_energy
    velocity_error = sample_error + residual_second * h**2 / 8
    phase_error = initial_error + velocity_error
    velocity_margin = probe.scaled_form_lower_bound - velocity_error
    turn_margin = probe.endpoint_phase_lower_bound - 2 * pi.hi - velocity_error
    if min(velocity_margin, turn_margin) <= 0:
        raise ValueError(
            "record error budget does not support the strict reconstructed-turn bounds"
        )

    gamma_hat_squared_lower = ((1 - calibration_error) * gamma.lo) ** 2
    residual_bound = 6 * j0
    normalization_error = 1 / (1 - calibration_error) ** 2 - 1
    energy_error = (24 + velocity_error) * velocity_error / (
        2 * gamma_hat_squared_lower
    ) + residual_bound * normalization_error
    decay_residual = forcing + sample_error
    decay_residual_slope = 2 * forcing + 2 * sample_error / h
    velocity_upper = 12 + decay_residual
    velocity_rate_upper = 12 + decay_residual_slope
    energy_bounds = (
        (12 * decay_residual + decay_residual**2 / 2) / gamma_hat_squared_lower,
        (12 * decay_residual + velocity_upper * decay_residual_slope)
        / gamma_hat_squared_lower,
        (12 * decay_residual + 24 * decay_residual_slope + decay_residual_slope**2)
        / gamma_hat_squared_lower,
    )
    factor_bounds = (
        n * velocity_upper + 2,
        n * velocity_rate_upper + n**2 * velocity_upper**2 + 2 * n * velocity_upper,
        12 * n
        + 3 * n**2 * velocity_upper * velocity_rate_upper
        + n**3 * velocity_upper**3
        + 2 * n * velocity_rate_upper
        + 2 * n**2 * velocity_upper**2,
    )
    integrand_second = (
        energy_bounds[2] * factor_bounds[0]
        + 2 * energy_bounds[1] * factor_bounds[1]
        + energy_bounds[0] * factor_bounds[2]
    )
    evaluation_count = intervals * subdivisions
    quadrature_error = integrand_second / (24 * evaluation_count**2)
    observation_error = (
        energy_error * (n * velocity_upper + 4)
        + residual_bound * (n * velocity_error + (12 * n**2 + 4 * n) * phase_error)
        + j0 * velocity_error
    )
    fine_error = probe.harmonic_defect_upper_bound
    total_error = (
        fine_error + (observation_error + quadrature_error) / pi.lo + numerical_error
    )
    return _OrganizedFreePhaseRecordBudget(
        form_error_bound=probe.form_error_bound,
        phase_error_bound=probe.phase_error_bound,
        sample_intervals=intervals,
        quadrature_subdivisions=subdivisions,
        raw_form_error_bound=raw_error,
        exposure_time_error_bound=exposure_error,
        kinetic_relative_error_bound=calibration_error,
        initial_phase_readout_error_bound=initial_error,
        numerical_coefficient_error_bound=numerical_error,
        harmonic=n,
        probe_bound=probe,
        sample_step=h,
        quadrature_step=h / subdivisions,
        scaled_current_forcing_upper_bound=forcing,
        sample_velocity_error_upper_bound=sample_error,
        decay_residual_second_derivative_upper_bound=residual_second,
        velocity_reconstruction_error_upper_bound=velocity_error,
        phase_reconstruction_error_upper_bound=phase_error,
        endpoint_time_error_upper_bound=velocity_error / probe.scaled_form_lower_bound,
        reconstructed_velocity_lower_bound=velocity_margin,
        full_turn_reconstruction_margin=turn_margin,
        kinetic_gamma_squared_lower_bound=gamma_hat_squared_lower,
        true_scaled_energy_residual_upper_bound=residual_bound,
        normalization_relative_error_upper_bound=normalization_error,
        energy_residual_error_upper_bound=energy_error,
        interpolated_decay_residual_upper_bound=decay_residual,
        interpolated_decay_residual_slope_upper_bound=decay_residual_slope,
        reconstructed_velocity_upper_bound=velocity_upper,
        reconstructed_velocity_rate_upper_bound=velocity_rate_upper,
        reconstructed_energy_residual_upper_bounds=energy_bounds,
        harmonic_factor_derivative_upper_bounds=factor_bounds,
        integrand_second_derivative_upper_bound=integrand_second,
        quadrature_numerator_error_upper_bound=quadrature_error,
        observation_numerator_error_upper_bound=observation_error,
        fine_harmonic_error_upper_bound=fine_error,
        coefficient_error_upper_bound=total_error,
        sample_count=intervals + 1,
        quadrature_evaluation_count_upper_bound=evaluation_count,
    )


@dataclass(frozen=True)
class _OrganizedFreePhaseRecord:
    """Numerical readout and conditional coefficient enclosure.

    ``observation_bounds`` encloses the declared reconstructed midpoint
    functional, including its unknown exact stopping endpoint. Sensor and
    model errors are added only in ``coefficient_bounds``. Certification
    establishes the numerical allowance and necessary record checks; it
    does not establish that the samples were produced by the declared law.
    Unavailable coefficients remain None for every noncertified status.
    """

    sample_times: tuple[Q, ...]
    raw_form_samples: tuple[Q, ...]
    gamma_hat: Q
    initial_phase_readout: Q
    budget: _OrganizedFreePhaseRecordBudget
    max_endpoint_iterations: int
    max_quadrature_evaluations: int
    status: str
    reason: str
    endpoint_bounds: I | None = None
    observation_bounds: I | None = None
    coefficient_bounds: I | None = None
    numerical_error_upper_bound: Q | None = None
    endpoint_iterations: int = 0
    quadrature_evaluations: int = 0
    interval_method: str = INTERVAL_METHOD
    exponential_method: str = EXP_NEG_UNIT_METHOD
    conditional_on_preparation_and_observation: bool = True


def _record_scalars(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} requires an ordered sequence of real scalars")
    try:
        source = tuple(values)
    except TypeError as exc:
        raise TypeError(
            f"{label} requires an ordered sequence of real scalars"
        ) from exc
    return tuple(
        _exact(value, f"{label}[{index}]") for index, value in enumerate(source)
    )


class _FreePhaseReconstruction:
    """Rebuild interval cell formulas from already admitted raw primitives."""

    def __init__(self, raw, gamma_hat):
        self.intervals = len(raw) - 1
        self.step = Q(1, self.intervals)
        self.gamma_squared = gamma_hat**2
        decay = exp_neg_unit(self.step)
        heat = [I(12)]
        for _ in range(self.intervals):
            heat.append(heat[-1] * decay)
        self.heat = tuple(heat)
        self.residual = tuple(
            I(gamma_hat * value) - reference for value, reference in zip(raw, heat)
        )
        self.slopes = tuple(
            (right - left) / self.step
            for left, right in zip(self.residual, self.residual[1:])
        )
        prefix = [I(0)]
        for left, right in zip(self.residual, self.residual[1:]):
            prefix.append(prefix[-1] + (left + right) * self.step / 2)
        self.prefix = tuple(prefix)

    def at(self, time, *, cell=None, heat=None):
        if cell is None:
            cell = min((time * self.intervals).__floor__(), self.intervals - 1)
        if heat is None:
            heat = 12 * exp_neg_unit(time)
        offset = I(time - cell * self.step)
        residual = self.residual[cell] + self.slopes[cell] * offset
        increment = (
            12
            - heat
            + self.prefix[cell]
            + self.residual[cell] * offset
            + self.slopes[cell] * offset**2 / 2
        )
        velocity = heat + residual
        energy = (heat * residual + residual**2 / 2) / self.gamma_squared
        return velocity, increment, energy


def _read_organized_free_phase_record(
    *,
    sample_times,
    raw_form_samples,
    gamma_hat,
    initial_phase_readout,
    form_error_bound,
    phase_error_bound,
    quadrature_subdivisions,
    raw_form_error_bound,
    exposure_time_error_bound,
    kinetic_relative_error_bound,
    initial_phase_readout_error_bound,
    numerical_coefficient_error_bound,
    harmonic,
    max_endpoint_iterations=128,
    max_quadrature_evaluations=1000000,
) -> _OrganizedFreePhaseRecord:
    """Enclose the admitted harmonic readout from an exact uniform-grid record.

    Times must equal i/N as represented, including 0 and 1; no sorting,
    snapping or imputation occurs. Gamma_hat is a positive supplied scalar,
    not an interval or an inferred kinetic law. All budget inputs are rebuilt
    through the shared owner. Invalid primitive domains and unsupported
    sufficient budget margins retain its TypeError/ValueError contract.

    Necessary record contradictions return ``incompatible_record``; interval
    ambiguity or insufficient precision returns an explicit unresolved status.
    Work caps return ``work_limit`` or ``unresolved_endpoint``. The fixed
    dyadic128 backend cannot promise arbitrarily small numerical allowances.
    Nominal times cannot authenticate actual exposure-time error or provenance.
    """
    times = _record_scalars(sample_times, "sample_times")
    raw = _record_scalars(raw_form_samples, "raw_form_samples")
    if len(raw) < 2 or len(times) != len(raw):
        raise ValueError(
            "record requires equal sample/time counts and at least two samples"
        )
    intervals = len(raw) - 1
    if times != tuple(Q(index, intervals) for index in range(intervals + 1)):
        raise ValueError(
            "sample_times must be the exact ordered uniform grid i/N on [0,1]"
        )
    calibration = _exact(gamma_hat, "gamma_hat")
    origin = _exact(initial_phase_readout, "initial_phase_readout")
    if calibration <= 0:
        raise ValueError("gamma_hat must be positive")
    root_limit = _positive_count(max_endpoint_iterations, "max_endpoint_iterations")
    work_limit = _positive_count(
        max_quadrature_evaluations, "max_quadrature_evaluations"
    )
    budget = _bound_organized_free_phase_record(
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        sample_intervals=intervals,
        quadrature_subdivisions=quadrature_subdivisions,
        raw_form_error_bound=raw_form_error_bound,
        exposure_time_error_bound=exposure_time_error_bound,
        kinetic_relative_error_bound=kinetic_relative_error_bound,
        initial_phase_readout_error_bound=initial_phase_readout_error_bound,
        numerical_coefficient_error_bound=numerical_coefficient_error_bound,
        harmonic=harmonic,
    )

    def result(status, reason, **observed):
        return _OrganizedFreePhaseRecord(
            sample_times=times,
            raw_form_samples=raw,
            gamma_hat=calibration,
            initial_phase_readout=origin,
            budget=budget,
            max_endpoint_iterations=root_limit,
            max_quadrature_evaluations=work_limit,
            status=status,
            reason=reason,
            **observed,
        )

    pi = pi_interval()
    ratio = calibration * 1023 * pi
    kappa = budget.kinetic_relative_error_bound
    if ratio.hi < 1 - kappa or ratio.lo > 1 + kappa:
        return result(
            "incompatible_record", "calibration contradicts its relative error bound"
        )
    if ratio.lo < 1 - kappa or ratio.hi > 1 + kappa:
        return result(
            "unresolved_numeric", "calibration lies on an unresolved interval boundary"
        )
    if abs(origin) > budget.initial_phase_readout_error_bound:
        return result(
            "incompatible_record", "initial phase readout contradicts its error bound"
        )

    reconstruction = _FreePhaseReconstruction(raw, calibration)
    sample_error = budget.sample_velocity_error_upper_bound
    amplitude = budget.interpolated_decay_residual_upper_bound
    difference = (
        2 * budget.scaled_current_forcing_upper_bound * budget.sample_step
        + 2 * sample_error
    )
    checks = [(reconstruction.residual[0], sample_error, "initial decay residual")]
    checks.extend(
        (value, amplitude, f"decay residual at sample {i}")
        for i, value in enumerate(reconstruction.residual)
    )
    checks.extend(
        (right - left, difference, f"adjacent decay residual at cell {i}")
        for i, (left, right) in enumerate(
            zip(reconstruction.residual, reconstruction.residual[1:])
        )
    )
    for value, allowance, label in checks:
        if value.lo > allowance or value.hi < -allowance:
            return result(
                "incompatible_record", f"{label} contradicts the declared record bounds"
            )
        if value.lo < -allowance or value.hi > allowance:
            return result(
                "unresolved_numeric", f"{label} has an unresolved interval boundary"
            )

    lower_speed = min(
        reconstruction.heat[i + 1].lo + min(left.lo, right.lo)
        for i, (left, right) in enumerate(
            zip(reconstruction.residual, reconstruction.residual[1:])
        )
    )
    endpoint_increment = reconstruction.at(Q(1))[1]
    if lower_speed <= 0 or endpoint_increment.lo <= 2 * pi.hi:
        return result(
            "unresolved_endpoint",
            "the reconstructed positive speed or full turn is unresolved",
        )

    r0, r1, _ = budget.reconstructed_energy_residual_upper_bounds
    a0, a1, _ = budget.harmonic_factor_derivative_upper_bounds
    endpoint_lipschitz = (
        r0 * a0
        + budget.quadrature_step * (r1 * a0 + r0 * a1) / 2
        + r1
        + harmonic * r0 * budget.reconstructed_velocity_upper_bound
    ) / pi.lo
    lo, hi = Q(0), Q(1)
    iterations = 0
    target = budget.numerical_coefficient_error_bound
    while endpoint_lipschitz * (hi - lo) / 2 > target / 4:
        if iterations >= root_limit:
            return result(
                "unresolved_endpoint",
                "endpoint iteration budget exhausted",
                endpoint_bounds=I(lo, hi),
                endpoint_iterations=iterations,
            )
        midpoint = (lo + hi) / 2
        residual = reconstruction.at(midpoint)[1] - 2 * pi
        iterations += 1
        if residual.lo > 0:
            hi = midpoint
        elif residual.hi < 0:
            lo = midpoint
        else:
            radius = residual.abs_max / lower_speed
            new_lo, new_hi = max(lo, midpoint - radius), min(hi, midpoint + radius)
            if new_hi - new_lo >= hi - lo:
                return result(
                    "unresolved_endpoint",
                    "endpoint exceeds the fixed interval precision",
                    endpoint_bounds=I(lo, hi),
                    endpoint_iterations=iterations,
                )
            lo, hi = new_lo, new_hi

    endpoint = I(lo, hi)
    stop = endpoint.midpoint
    panel_count = intervals * budget.quadrature_subdivisions
    used_panels = (stop * panel_count).__ceil__()
    if used_panels > work_limit:
        return result(
            "work_limit",
            "midpoint evaluation budget is insufficient",
            endpoint_bounds=endpoint,
            endpoint_iterations=iterations,
        )

    total = I(0)
    step = budget.quadrature_step
    decay = exp_neg_unit(step)
    half_decay = exp_neg_unit(step / 2)
    heat = None
    for index in range(used_panels):
        left, right = Q(index, panel_count), min(Q(index + 1, panel_count), stop)
        midpoint = (left + right) / 2
        cell = index // budget.quadrature_subdivisions
        if right - left < step:
            heat = 12 * exp_neg_unit(midpoint)
        elif index % budget.quadrature_subdivisions == 0:
            heat = reconstruction.heat[cell] * half_decay
        else:
            heat = heat * decay
        velocity, increment, energy = reconstruction.at(midpoint, cell=cell, heat=heat)
        angle = harmonic * (origin + increment)
        integrand = energy * (harmonic * velocity * cos(angle) - 2 * sin(angle))
        total = total + (right - left) * integrand

    _, stop_increment, stop_energy = reconstruction.at(stop)
    initial_energy = reconstruction.at(Q(0))[2]
    boundary = stop_energy * sin(
        harmonic * (origin + stop_increment)
    ) - initial_energy * sin(harmonic * I(origin))
    fixed_endpoint_observation = (total - boundary) / pi
    endpoint_error = endpoint_lipschitz * endpoint.radius
    observation = fixed_endpoint_observation + I(-endpoint_error, endpoint_error)
    common = dict(
        endpoint_bounds=endpoint,
        observation_bounds=observation,
        numerical_error_upper_bound=observation.radius,
        endpoint_iterations=iterations,
        quadrature_evaluations=used_panels,
    )
    if observation.radius > target:
        return result(
            "unresolved_numeric",
            "numerical enclosure exceeds the coefficient allowance",
            **common,
        )
    model_error = budget.coefficient_error_upper_bound - target
    coefficient = observation + I(-model_error, model_error)
    return result(
        "certified",
        "conditional readout within the declared numerical allowance",
        coefficient_bounds=coefficient,
        **common,
    )
