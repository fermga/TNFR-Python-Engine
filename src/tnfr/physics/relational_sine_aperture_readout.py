"""Validated finite boxcar readouts with an explicitly supplied affine clock.

The complete nodal state, clock rate and cumulative observation integral
are carried through four fixed windows. Averages use the shared Taylor
increment to cancel the same accumulator baseline symbolically.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import MAX_ORDER, Jet
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics._validated_taylor import (
    ValidatedBoxTaylorStep,
    validated_box_taylor_step,
)
from ._sine_flow import _full_sine_field
from .phase_cycle_geometry import PhaseCycleGeometry, _derive
from .relational_observations import _interval, _ordered
from .relational_sine_two_port_compatibility import _EDGES, _NODES

__all__ = ("SineApertureReadout", "bound_sine_aperture_readout")


def _affine_clock_sine_field(model, geometry, degrees, clock_slope):
    """Augment the shared complete structural law with rate and integral rows."""
    structural, _ = _full_sine_field(model, geometry, degrees)

    def flow(state):
        nodal_rates = structural(state[:36])
        slope = (
            Jet.constant(clock_slope, state[0].order)
            if isinstance(state[0], Jet)
            else I(clock_slope)
        )
        return tuple(state[36] * rate for rate in nodal_rates) + (
            slope,
            state[4] - state[5],
        )

    def domain(_):
        # This polynomial/sine field is smooth on all R^38, even if an
        # outward tube for rho includes zero. Positivity of the prescribed
        # affine clock is proved separately from its exact endpoint rates.
        return (Q(1),)

    return flow, domain


@dataclass(frozen=True)
class SineApertureReadout:
    """Four full-law boxcar enclosures from one carried history, or its prefix.

    The source consists of eighteen forms and eighteen continuous phase
    lifts. The supplied affine clock is rho(s)=initial_clock_rate+clock_slope*s.
    Its positivity is admitted analytically on [0,2H]; the numerical tube's
    global smooth-domain certificate does not assert a positive tube rate,
    an acute phase chart or identity retention.

    The cumulative passive integral starts at zero once and is carried with
    all other coordinates. Each average is its same-window symbolic Taylor
    increment divided by that exact positive aperture width. Failed windows
    have no fabricated average; later windows are not attempted.
    """

    initial_form_bounds: tuple[I, ...]
    initial_phase_bounds: tuple[I, ...]
    phase_increments: tuple[Q, Q]
    probe_duration: Q
    initial_clock_rate: Q
    clock_slope: Q
    order: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    dipole: tuple[Q, ...]
    aperture_windows: tuple[tuple[Q, Q], ...]
    phase_event_increments: tuple[Q, Q, Q, Q]
    clock_rate_bounds: tuple[Q, Q]
    clock_positivity_certified: bool
    cumulative_structural_exposures: tuple[Q, ...]
    initial_augmented_box: tuple[I, ...]
    window_initial_boxes: tuple[tuple[I, ...], ...]
    baseline_readout_bounds: I
    steps: tuple[ValidatedBoxTaylorStep, ...]
    averaged_readout_bounds: tuple[I, ...]
    completed_window_count: int
    completed_observed_time: Q
    completed_endpoint_box: tuple[I, ...] | None
    final_state_bounds: tuple[I, ...] | None
    failed_window_index: int | None
    failed_tube: tuple[I, ...] | None
    status: str
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "d_tau/d_s=rho(s)=initial_clock_rate+clock_slope*s"
    capacity: tuple[Q, ...] = (Q(1),) * 18
    state_order: tuple[str, ...] = (
        tuple(f"x_{i}" for i in range(18))
        + tuple(f"theta_{i}" for i in range(18))
        + ("rho", "cumulative_readout_integral")
    )
    complete_observed_rows: tuple[str, ...] = (
        "dx/ds=rho*(-A*x+gamma*f(theta))",
        "dtheta/ds=rho*gamma*A*x",
        "drho/ds=clock_slope",
        "dz/ds=x_4-x_5",
    )
    scope: tuple[str, ...] = (
        "primitive_full_thirty_six_coordinate_source_box_on_fixed_two_port_C9_support",
        "same_complete_shared_sine_law_with_both_nodal_rows_transformed_by_rho",
        "explicit_positive_affine_forward_clock_not_inferred_clock_profile",
        "four_normalized_boxcars_at_fixed_thirds_and_full_second_window",
        "phase_only_events_at_zero_and_H_with_zero_increment_meaning_no_event",
        "all_thirty_eight_endpoint_coordinates_carried_without_state_or_integral_reset",
        "passive_observed_time_integral_row_has_no_extra_clock_rate_factor",
        "per_window_integral_increment_cancels_its_same_initial_coordinate_symbolically",
        "four_fixed_shared_Picard_Taylor_steps_without_retry_or_budget_adaptation",
        "first_failed_step_stops_acquisition_and_retains_successful_prefix_only",
        "exact_affine_clock_positivity_is_separate_from_global_smooth_tube_domain",
        "no_acute_chart_identity_acquisition_or_physical_sensor_claim",
        "no_inverse_bounds_target_root_gain_offset_or_noise_input",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-aperture-readout.v1", "report": _project(self)}


def bound_sine_aperture_readout(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    phase_increments,
    probe_duration,
    initial_clock_rate,
    clock_slope,
    order,
) -> SineApertureReadout:
    """Enclose four boxcar averages under one supplied affine-clock history.

    All seven inputs are mandatory primitives. Source channels each contain
    eighteen admitted interval endpoint pairs; phases are continuous lifts
    in radians. Cumulative amplitudes satisfy 0<=a1<=a2<=1. Require H>0,
    positive affine rates at zero and 2H, and max_endpoint_rate*H<=1/2.
    There is no independent upper bound on observed H. Order is an ordinary
    integer in 1..16 and applies unchanged to every fixed window.

    Four shared source-box Taylor steps evolve the complete 38-coordinate
    augmented state. No sensor law or inverse result is consumed. A numerical
    failure returns its attempted initial box and last tube, all completed
    certificates and averages, and no full-horizon state or later response.
    """
    channels = []
    for raw, label in (
        (initial_form_bounds, "initial_form_bounds"),
        (initial_phase_bounds, "initial_phase_bounds"),
    ):
        rows = _ordered(raw, label, limit=19)
        if len(rows) != 18:
            raise ValueError(f"{label} must contain exactly eighteen endpoint pairs")
        channels.append(
            tuple(_interval(row, f"{label}[{index}]") for index, row in enumerate(rows))
        )
    form, phase = channels
    raw_amplitudes = _ordered(phase_increments, "phase_increments", limit=3)
    if len(raw_amplitudes) != 2:
        raise ValueError(
            "phase_increments must contain exactly two cumulative amplitudes"
        )
    amplitudes = tuple(
        exact_or_represented_real(value, f"phase_increments[{index}]")
        for index, value in enumerate(raw_amplitudes)
    )
    horizon, rate, slope = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (probe_duration, "probe_duration"),
            (initial_clock_rate, "initial_clock_rate"),
            (clock_slope, "clock_slope"),
        )
    )
    if not 0 <= amplitudes[0] <= amplitudes[1] <= 1:
        raise ValueError("cumulative phase_increments must satisfy 0<=a1<=a2<=1")
    if horizon <= 0:
        raise ValueError("probe_duration must be strictly positive")
    end_rate = rate + 2 * slope * horizon
    rates = (min(rate, end_rate), max(rate, end_rate))
    if rates[0] <= 0:
        raise ValueError(
            "affine clock rates at zero and twice probe_duration must be positive"
        )
    if rates[1] * horizon > Q(1, 2):
        raise ValueError(
            "maximum affine clock rate times probe_duration must not exceed 1/2"
        )
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("order must be an ordinary integer in 1..16")

    geometry = _derive(_NODES, _EDGES)
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in _NODES)
    boundaries = (Q(0), horizon / 3, 2 * horizon / 3, horizon, 2 * horizon)
    windows = tuple(zip(boundaries, boundaries[1:]))
    jumps = (amplitudes[0], Q(0), Q(0), amplitudes[1] - amplitudes[0])
    exposures = tuple(rate * time + slope * time * time / 2 for time in boundaries)
    initial = form + phase + (I(rate), I(0))
    state = initial
    sources, steps, averages = [], [], []
    failed_index = failed_tube = reason = None
    flow, domain = _affine_clock_sine_field(model, geometry, degrees, slope)
    for index, ((start, end), jump) in enumerate(zip(windows, jumps)):
        if jump:
            state = (
                state[:18]
                + tuple(
                    value + jump * coefficient
                    for value, coefficient in zip(state[18:36], q)
                )
                + state[36:]
            )
        sources.append(state)
        width = end - start
        step, failed_tube, reason = validated_box_taylor_step(
            state, width, flow, domain, order=order, time=start
        )
        if step is None:
            failed_index = index
            break
        steps.append(step)
        integral = step.increment[37]
        # The aperture width is an exact positive primitive expression.
        # Divide rational endpoints before dyadic interval materialization.
        averages.append(I(integral.lo / width, integral.hi / width))
        state = step.endpoint
    completed = len(steps)
    return SineApertureReadout(
        initial_form_bounds=form,
        initial_phase_bounds=phase,
        phase_increments=amplitudes,
        probe_duration=horizon,
        initial_clock_rate=rate,
        clock_slope=slope,
        order=order,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        dipole=q,
        aperture_windows=windows,
        phase_event_increments=jumps,
        clock_rate_bounds=rates,
        clock_positivity_certified=True,
        cumulative_structural_exposures=exposures,
        initial_augmented_box=initial,
        window_initial_boxes=tuple(sources),
        baseline_readout_bounds=form[4] - form[5],
        steps=tuple(steps),
        averaged_readout_bounds=tuple(averages),
        completed_window_count=completed,
        completed_observed_time=boundaries[completed],
        completed_endpoint_box=steps[-1].endpoint if steps else None,
        final_state_bounds=state if completed == 4 else None,
        failed_window_index=failed_index,
        failed_tube=failed_tube,
        status="admitted" if completed == 4 else "unavailable",
        unavailable_reasons=() if reason is None else (reason,),
    )
