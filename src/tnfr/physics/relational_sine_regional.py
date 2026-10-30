"""Regional geometry and supplied complete conservative sine enclosures.

The reader checks the declared law, chain and primitive interval evidence,
including recomputed Picard inclusion. It does not replay Taylor endpoint
proofs or authenticate a producer. All conclusions are conditional on the
supplied numerical endpoint enclosures, retaining their full uncertainty.
The separate full-state cycle barrier uses conservation and phase geometry,
without an enclosure, forecast, trajectory or presumed flat preparation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from math import isfinite

from .._exact_time import exact_or_represented_real, exp_upper_float
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ..mathematics._validated_taylor import ValidatedTaylorStep
from ._sine_admission import _admit_sine_source, _sine_model_coefficients
from .phase_cycle_geometry import C5_PHASE_SECTOR_BARRIER
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    SineRegionalChannelBalance,
    _validate_comparison_labels,
)
from .relational_sine_forecast import SineForecast, _admit_support, _sine_flow

__all__ = (
    "SineRegionalOrganizationStep",
    "SineRegionalOrganization",
    "assess_sine_regional_organization",
    "SineRegionalChannelStep",
    "SineRegionalChannelHistory",
    "assess_sine_regional_channels",
    "SineCycleBarrier",
    "SineCycleBarrierSector",
    "assess_sine_cycle_barrier",
    "SineCycleRetention",
    "assess_sine_cycle_retention",
    "SineReversiblePreparationStep",
    "SineReversiblePreparation",
    "assess_sine_reversible_preparation",
)


def _scalar(value, label):
    return exact_or_represented_real(value, label)


def _interval(value, label):
    if isinstance(value, I):
        return I(_scalar(value.lo, label), _scalar(value.hi, label))
    return I(_scalar(value, label))


def _interval_row(values, size, label):
    values = _ordered(values, label, limit=size + 1)
    if len(values) != size:
        raise ValueError(f"{label} must retain every full-state coordinate")
    return tuple(_interval(value, f"{label}[{i}]") for i, value in enumerate(values))


def _unit_box(values, size, label, *, point_capacity=False):
    values = _interval_row(values, size, label)
    if not values[-1].contains(1) or point_capacity and values[-1] != I(1):
        raise ValueError(f"{label} must retain the held unit capacity")
    return values


def _flow(box, forecast):
    return _sine_flow(
        box,
        neighbors=forecast.neighbors,
        visible_capacity=forecast.visible_capacity,
        model=forecast.model,
    )


def _admit_forecast(forecast):
    if not isinstance(forecast, SineForecast):
        raise TypeError("an exact SineForecast is required")
    if _sine_model_coefficients(forecast.model) != (0, 1, 1):
        raise ValueError(
            "regional organization requires zero loss and unit exchange and beta"
        )
    neighbors, capacity = _admit_support(
        forecast.neighbors, forecast.visible_capacity, forecast.model
    )
    if any(value != 1 for value in capacity):
        raise ValueError("regional organization requires held unit capacities")
    if forecast.freeze_hidden is not False:
        raise ValueError("changed-capacity frozen forecasts are unsupported")
    if forecast.method != "full_sine_held_capacity_Picard_Taylor_Metzler128_v1":
        raise ValueError("the declared complete sine forecast method is required")
    size = 2 * len(neighbors) + 1
    initial = _unit_box(forecast.initial_box, size, "initial_box", point_capacity=True)
    endpoint = _unit_box(forecast.endpoint, size, "endpoint", point_capacity=True)
    at, end, actual, step_size = (
        _scalar(getattr(forecast, name), name)
        for name in ("observation_time", "end_time", "validated_end_time", "time_step")
    )
    if not 0 <= at <= actual <= end or at == end or step_size <= 0:
        raise ValueError("forecast times must retain their ordered positive horizon")
    if (end - at) / step_size > 256:
        raise ValueError("forecast must respect the shared 256-step budget")
    if type(forecast.order) is not int or not 1 <= forecast.order <= 16:
        raise ValueError("forecast Taylor order must lie between 1 and 16")
    reasons = _ordered(forecast.reasons, "reasons")
    if any(not isinstance(value, str) or not value for value in reasons):
        raise ValueError("forecast reasons must be nonempty strings")
    complete = actual == end
    if forecast.status not in ("admitted", "unavailable"):
        raise ValueError("unknown forecast status")
    if (forecast.status == "admitted") != complete or bool(reasons) == complete:
        raise ValueError("forecast status and reasons must match actual coverage")
    failed = (
        None
        if forecast.failed_tube is None
        else _unit_box(forecast.failed_tube, size, "failed_tube")
    )
    if complete and failed is not None:
        raise ValueError("a complete forecast cannot retain a failed tube")
    start = forecast.forecast_start
    if start is not None:
        start = _scalar(start, "forecast_start")
        if not at <= start <= end:
            raise ValueError("forecast_start must lie in the declared horizon")
    normalized = replace(
        forecast,
        neighbors=neighbors,
        visible_capacity=capacity,
        initial_box=initial,
        endpoint=endpoint,
        observation_time=at,
        end_time=end,
        validated_end_time=actual,
        time_step=step_size,
        reasons=reasons,
        failed_tube=failed,
        forecast_start=start,
    )
    time, incoming, steps = at, initial, []
    raw_steps = _ordered(forecast.steps, "steps", limit=257)
    if len(raw_steps) > 256:
        raise ValueError("forecast exceeds the shared step budget")
    for index, step in enumerate(raw_steps):
        if not isinstance(step, ValidatedTaylorStep):
            raise TypeError("forecast steps must be ValidatedTaylorStep reports")
        step_time = _scalar(step.time, "step.time")
        duration = _scalar(step.duration, "step.duration")
        if step_time != time or duration <= 0 or duration != min(step_size, end - time):
            raise ValueError(
                "forecast steps must retain the contiguous declared time grid"
            )
        if time + duration > actual:
            raise ValueError("a step extends beyond the validated endpoint time")
        tube = _unit_box(step.tube, size, "step.tube")
        final = _unit_box(step.endpoint, size, "step.endpoint", point_capacity=True)
        if any(not value.subset_of(bound) for value, bound in zip(incoming, tube)):
            raise ValueError("incoming endpoint must lie in the next whole-time tube")
        if any(not value.subset_of(bound) for value, bound in zip(final, tube)):
            raise ValueError("step endpoint must lie in its whole-time tube")
        margin = _scalar(step.picard_interior_margin, "step.picard_interior_margin")
        domain = tuple(
            _scalar(value, "step.domain_lower_bounds")
            for value in _ordered(step.domain_lower_bounds, "step.domain_lower_bounds")
        )
        if margin <= 0 or not domain or min(domain) <= 0:
            raise ValueError("step Picard and domain margins must be strictly positive")
        radii = _ordered(
            step.propagated_initial_radii,
            "step.propagated_initial_radii",
            limit=size + 1,
        )
        if len(radii) != size:
            raise ValueError("propagated radii must retain every state coordinate")
        radii = tuple(
            _scalar(value, "step.propagated_initial_radii") for value in radii
        )
        if any(value < 0 for value in radii):
            raise ValueError("propagated radii must be nonnegative")
        remainder = _interval_row(
            step.local_remainder_bounds, size, "step.local_remainder_bounds"
        )
        # This cheap check proves inclusion of the declared tube from the
        # supplied incoming endpoint set. It does not revalidate that incoming
        # Taylor endpoint's derivation, which remains a numerical premise.
        rates = _flow(tube, normalized)
        images = tuple(
            value + I(0, duration) * rate for value, rate in zip(incoming, rates)
        )
        recomputed = min(
            min(image.lo - bound.lo, bound.hi - image.hi)
            for image, bound in zip(images, tube)
        )
        if recomputed <= 0 or margin > recomputed:
            raise ValueError("step does not retain its claimed strict Picard inclusion")
        steps.append(
            replace(
                step,
                time=step_time,
                duration=duration,
                tube=tube,
                endpoint=final,
                picard_interior_margin=margin,
                domain_lower_bounds=domain,
                propagated_initial_radii=radii,
                local_remainder_bounds=remainder,
            )
        )
        time, incoming = time + duration, final
    if time != actual or incoming != endpoint:
        raise ValueError(
            "forecast endpoint and validated time must match the complete step chain"
        )
    return replace(normalized, steps=tuple(steps))


def _cycle_on_support(neighbors, cycle_indices, *, induced):
    cycle = _ordered(cycle_indices, "cycle_indices", limit=6)
    size = len(neighbors)
    if (
        len(cycle) != 5
        or any(type(i) is not int or not 0 <= i < size for i in cycle)
        or len(set(cycle)) != 5
    ):
        raise ValueError(
            "cycle_indices must contain five distinct full-source node indices"
        )
    pairs = tuple(zip(cycle, cycle[1:] + cycle[:1]))
    expected = {frozenset(pair) for pair in pairs}
    members = set(cycle)
    actual = {frozenset((i, j)) for i in cycle for j in neighbors[i] if j in members}
    if not expected <= actual or induced and expected != actual:
        qualification = "an induced" if induced else "a simple"
        raise ValueError(f"the receiver must be {qualification} C5 in the full support")
    return cycle


def _cycle(forecast, cycle_indices):
    return _cycle_on_support(forecast.neighbors, cycle_indices, induced=True)


def _branches(gaps):
    pi = pi_interval()
    offsets, principal = [], []
    for gap in gaps:
        candidate = (gap + pi) / (2 * pi)
        left, right = candidate.lo // 1, candidate.hi // 1
        if left != right:
            return None, None, None, None
        shifted = gap - 2 * left * pi
        if shifted.abs_max >= pi.lo:
            return None, None, None, None
        offsets.append(left)
        principal.append(shifted)
    principal = tuple(principal)
    margin = min(pi.lo / 2 - gap.abs_max for gap in principal)
    return tuple(offsets), principal, -sum(offsets), margin


def _snapshot(box, forecast, cycle):
    size = len(forecast.neighbors)
    forms, phases = box[:size], box[size : 2 * size]
    members = set(cycle)
    regional_form, regional_phase, full = I(0), I(0), I(0)
    # The actual held capacity is exactly1 even where a whole-time Picard
    # tube includes additional artificial values in that augmented coordinate.
    rates = _flow(box[:-1] + (I(1),), forecast)
    boundary_power = I(0)
    for i, row in enumerate(forecast.neighbors):
        for j in row:
            if i >= j:
                continue
            form = (forms[j] - forms[i]) ** 2 / 2
            phase = 1 - cos(phases[j] - phases[i])
            full += form + phase
            if i in members and j in members:
                regional_form += form
                regional_phase += phase
            elif (i in members) != (j in members):
                inside, outside = (i, j) if i in members else (j, i)
                boundary_power += (forms[outside] - forms[inside]) * rates[
                    inside
                ] + sin(phases[outside] - phases[inside]) * rates[size + inside]
    anchor = cycle[0]
    relative_rates = tuple(
        I(0) if i == anchor else rates[size + i] - rates[size + anchor]
        for i in range(size)
    )
    return (
        regional_form,
        regional_phase,
        regional_form + regional_phase,
        full,
        relative_rates,
        boundary_power,
        rates[size : 2 * size],
    )


@dataclass(frozen=True)
class SineRegionalChannelStep:
    """Whole-time storage rates and cumulative integrals at the step endpoint."""

    time: Q
    duration: Q
    channels: SineRegionalChannelBalance
    cumulative_internal_conversion: I
    cumulative_boundary_form_input: I
    cumulative_boundary_phase_input: I
    form_storage_change: I
    phase_storage_change: I
    form_integral_residual: I
    phase_integral_residual: I
    phase_storage_upper_bound: Q
    running_phase_work_upper_bound: Q


@dataclass(frozen=True)
class SineRegionalChannelHistory:
    """Retrospective or prospective conditional analysis of supplied enclosures."""

    forecast: SineForecast
    cycle_indices: tuple[int, ...]
    initial_form_storage: I
    initial_phase_storage: I
    endpoint_form_storage: I
    endpoint_phase_storage: I
    steps: tuple[SineRegionalChannelStep, ...]
    cumulative_internal_conversion: I
    cumulative_boundary_form_input: I
    cumulative_boundary_phase_input: I
    phase_storage_upper_bound: Q
    running_phase_work_upper_bound: Q
    initial_phase_flat_certified: bool
    acute_accessibility_barrier: Q
    acute_entry_excluded_on_validated_prefix: bool
    horizon_complete: bool
    clock: str = "structural_t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_conservative_sine_law_and_unit_held_capacity",
        "shared_regional_storage_channel_kernel_original_structural_t",
        "positive_internal_conversion_transfers_phase_storage_to_form_storage",
        "boundary_storage_inputs_are_not_conjugate_boundary_work_components",
        "cumulative_bounds_integrate_whole_time_rates_not_endpoint_samples",
        "primitive_forecast_readmitted_and_Picard_inclusion_recomputed",
        "Taylor_endpoint_enclosures_remain_supplied_premises_no_replay_or_authentication",
        "phase_flat_admission_requires_equal_exact_represented_lifts",
        "C5_phase_minimax_excludes_acute_unit_winding_entry_below_7_over_2",
        "barrier_is_necessary_running_phase_storage_not_sufficient_entry_or_arbitrary_winding",
        "partial_prefix_does_not_certify_unreached_horizon",
        "no_source_selection_damping_new_trajectory_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-regional-channel-history.v1",
            "report": _project(self),
        }


def assess_sine_regional_channels(forecast, *, cycle_indices):
    """Integrate signed channel bounds without rerunning the supplied forecast.

    The same strict complete-law and endpoint-evidence admission used by the
    regional organization observer applies. The C5 minimax conclusion requires
    exactly flat initial receiver phase; inability to certify flatness is not
    a claim that phase is nonflat. The barrier concerns reaching any fully
    acute unit-winding state, not arbitrary transient winding.
    """
    from .relational_sine_comparison import _sine_regional_channel_balance

    forecast = _admit_forecast(forecast)
    cycle = _cycle(forecast, cycle_indices)
    size = len(forecast.neighbors)
    edges = tuple(
        (i, j) for i, row in enumerate(forecast.neighbors) for j in row if i < j
    )
    degrees = tuple(len(row) for row in forecast.neighbors)
    initial = _snapshot(forecast.initial_box, forecast, cycle)
    endpoint = _snapshot(forecast.endpoint, forecast, cycle)
    initial_phase = tuple(forecast.initial_box[size + i] for i in cycle)
    flat = all(
        value == initial_phase[0] and value.width == 0 for value in initial_phase
    )
    conversion, boundary_form, boundary_phase = I(0), I(0), I(0)
    phase_upper = initial[1].hi
    running_phase_work_upper = Q(0)
    steps = []
    for step in forecast.steps:
        channels = _sine_regional_channel_balance(
            reference_model=forecast.model,
            edges=edges,
            degrees=degrees,
            capacity=(Q(1),) * size,
            epi=step.tube[:size],
            phase=step.tube[size : 2 * size],
            region_indices=cycle,
        )
        phase_work_tube = (
            -conversion
            + boundary_phase
            + I(0, step.duration)
            * (-channels.internal_conversion + channels.boundary_phase_input)
        )
        running_phase_work_upper = max(running_phase_work_upper, phase_work_tube.hi)
        conversion += step.duration * channels.internal_conversion
        boundary_form += step.duration * channels.boundary_form_input
        boundary_phase += step.duration * channels.boundary_phase_input
        end = _snapshot(step.endpoint, forecast, cycle)
        tube = _snapshot(step.tube, forecast, cycle)
        form_change, phase_change = end[0] - initial[0], end[1] - initial[1]
        form_residual = form_change - (conversion + boundary_form)
        phase_residual = phase_change - (-conversion + boundary_phase)
        if not form_residual.contains(0) or not phase_residual.contains(0):
            raise ValueError("endpoint storage contradicts integrated channel bounds")
        phase_upper = max(phase_upper, tube[1].hi)
        steps.append(
            SineRegionalChannelStep(
                time=step.time,
                duration=step.duration,
                channels=channels,
                cumulative_internal_conversion=conversion,
                cumulative_boundary_form_input=boundary_form,
                cumulative_boundary_phase_input=boundary_phase,
                form_storage_change=form_change,
                phase_storage_change=phase_change,
                form_integral_residual=form_residual,
                phase_integral_residual=phase_residual,
                phase_storage_upper_bound=tube[1].hi,
                running_phase_work_upper_bound=running_phase_work_upper,
            )
        )
    barrier = C5_PHASE_SECTOR_BARRIER
    return SineRegionalChannelHistory(
        forecast=forecast,
        cycle_indices=cycle,
        initial_form_storage=initial[0],
        initial_phase_storage=initial[1],
        endpoint_form_storage=endpoint[0],
        endpoint_phase_storage=endpoint[1],
        steps=tuple(steps),
        cumulative_internal_conversion=conversion,
        cumulative_boundary_form_input=boundary_form,
        cumulative_boundary_phase_input=boundary_phase,
        phase_storage_upper_bound=phase_upper,
        running_phase_work_upper_bound=running_phase_work_upper,
        initial_phase_flat_certified=flat,
        acute_accessibility_barrier=barrier,
        acute_entry_excluded_on_validated_prefix=flat
        and min(phase_upper, initial[1].hi + running_phase_work_upper) < barrier,
        horizon_complete=forecast.validated_end_time == forecast.end_time,
    )


@dataclass(frozen=True)
class SineRegionalOrganizationStep:
    """Regional properties holding throughout one supplied Picard tube."""

    time: Q
    duration: Q
    cycle_raw_gap_bounds: tuple[I, ...]
    cycle_principal_gap_bounds: tuple[I, ...] | None
    edge_turn_offsets: tuple[int, ...] | None
    winding: int | None
    acute_margin_lower_bound: Q | None
    requested_acute_margin_met: bool
    acute_winding_excluded: bool
    exclusion_reasons: tuple[str, ...]
    regional_form_storage: I
    regional_phase_storage: I
    regional_storage: I
    integrated_regional_work: I
    full_storage: I
    relative_phase_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    regional_boundary_work: I


@dataclass(frozen=True)
class SineRegionalOrganization:
    """Conditional whole-time regional assessment without endpoint proof replay."""

    forecast: SineForecast
    cycle_indices: tuple[int, ...]
    minimum_duration: Q
    acute_margin: Q
    initial_regional_storage: I
    endpoint_regional_storage: I
    endpoint_integrated_regional_work: I
    initial_full_storage: I
    endpoint_full_storage: I
    steps: tuple[SineRegionalOrganizationStep, ...]
    certified_windows: tuple[tuple[Q, Q], ...]
    horizon_complete: bool
    outcome: str
    unresolved: tuple[str, ...]
    clock: str = "structural_t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_sine_law_original_structural_t_clock",
        "fixed_full_support_and_exact_unit_held_capacity",
        "induced_receiver_C5_and_all_environmental_coordinates_retained",
        "source_chain_scalars_and_boxes_readmitted_with_recomputed_Picard_inclusion",
        "conditional_on_supplied_Taylor_endpoint_enclosures_no_solver_replay_or_authentication",
        "all_regional_fields_rebuilt_from_complete_boxes_without_midpoint_replacement",
        "known_held_unit_capacity_tightens_artificial_Picard_capacity_width",
        "whole_time_fixed_branch_winding_uses_exact_telescoping_not_rounded_samples",
        "requested_margin_and_duration_are_observation_policies_not_physical_absence",
        "edge_cosine_two_hop_cosine_and_zero_winding_are_sufficient_exclusions",
        "integrated_net_boundary_work_equals_internal_storage_change_by_exact_zero_loss_balance",
        "regional_boundary_work_is_instantaneous_power_not_integrated_work",
        "partial_horizon_remains_explicitly_unresolved_without_extrapolation",
        "finite_retention_does_not_establish_initially_unorganized_preparation_capture_or_identity",
        "no_forcing_support_events_constitutive_selection_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-regional-organization.v1",
            "report": _project(self),
        }


def assess_sine_regional_organization(
    forecast, *, cycle_indices, minimum_duration, acute_margin
) -> SineRegionalOrganization:
    """Observe finite acute C5 winding or sufficient absence on supplied tubes.

    ``minimum_duration`` is strictly positive in the forecast's original t
    clock. ``acute_margin`` is a nonnegative requested margin in radians.
    Failing either requested criterion does not prove acute winding absent.
    Returned work is net inward boundary work since the initial observation,
    obtained from the same model's exact conservative regional energy identity.
    """
    forecast = _admit_forecast(forecast)
    cycle = _cycle(forecast, cycle_indices)
    minimum = _scalar(minimum_duration, "minimum_duration")
    margin = _scalar(acute_margin, "acute_margin")
    if minimum <= 0 or margin < 0:
        raise ValueError(
            "minimum_duration must be positive and acute_margin nonnegative"
        )
    size = len(forecast.neighbors)
    initial = _snapshot(forecast.initial_box, forecast, cycle)
    endpoint = _snapshot(forecast.endpoint, forecast, cycle)
    observations, windows = [], []
    run_start, run_end, run_offsets = None, None, None

    def finish_run():
        if run_start is not None and run_end - run_start >= minimum:
            windows.append((run_start, run_end))

    for step in forecast.steps:
        phases = step.tube[size : 2 * size]
        gaps = tuple(
            phases[j] - phases[i] for i, j in zip(cycle, cycle[1:] + cycle[:1])
        )
        offsets, principal, winding, lower = _branches(gaps)
        reasons = []
        if any(cos(gap).hi <= 0 for gap in gaps):
            reasons.append("cycle_edge_nonpositive_cosine")
        if any(
            cos(phases[cycle[(i + 2) % 5]] - phases[cycle[i]]).lo >= 0 for i in range(5)
        ):
            reasons.append("two_hop_pair_nonnegative_cosine")
        if winding == 0:
            reasons.append("fixed_branch_zero_winding")
        meets = winding in (-1, 1) and lower is not None and lower > margin
        if meets:
            if run_start is None or run_offsets != offsets:
                finish_run()
                run_start, run_offsets = step.time, offsets
            run_end = step.time + step.duration
        else:
            finish_run()
            run_start, run_end, run_offsets = None, None, None
        form, phase, storage, full, relative, power, rates = _snapshot(
            step.tube, forecast, cycle
        )
        observations.append(
            SineRegionalOrganizationStep(
                step.time,
                step.duration,
                gaps,
                principal,
                offsets,
                winding,
                lower,
                meets,
                bool(reasons),
                tuple(reasons),
                form,
                phase,
                storage,
                storage - initial[2],
                full,
                relative,
                rates,
                power,
            )
        )
    finish_run()
    complete = forecast.validated_end_time == forecast.end_time
    if windows:
        outcome = "finite_acute_retention_certified"
    elif (
        complete
        and observations
        and all(step.acute_winding_excluded for step in observations)
    ):
        outcome = "acute_winding_excluded_on_horizon"
    else:
        outcome = "unresolved"
    unresolved = []
    if not complete:
        unresolved.append("requested_horizon_not_fully_enclosed")
    if outcome == "unresolved":
        unresolved.append("no_requested_window_or_complete_acute_absence_certificate")
    return SineRegionalOrganization(
        forecast=forecast,
        cycle_indices=cycle,
        minimum_duration=minimum,
        acute_margin=margin,
        initial_regional_storage=initial[2],
        endpoint_regional_storage=endpoint[2],
        endpoint_integrated_regional_work=endpoint[2] - initial[2],
        initial_full_storage=initial[3],
        endpoint_full_storage=endpoint[3],
        steps=tuple(observations),
        certified_windows=tuple(windows),
        horizon_complete=complete,
        outcome=outcome,
        unresolved=tuple(unresolved),
    )


@dataclass(frozen=True)
class SineCycleBarrierSector:
    """One oriented sector's membership and conservation consequences."""

    winding: int
    source_membership: str
    sector_invariance_certified: bool
    acute_acquisition_excluded: bool
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SineCycleBarrier:
    """Conditional all-time C5 sector barrier for an actual full source.

    Sectors have winding +/-1 and all principal gaps less than 2*pi/3 in
    magnitude. They are wider than the acute sectors. Each boundary costs
    at least 7/2 in unit receiver phase storage, independent of the live
    environment. Conserved full storage at most this boundary value prevents
    finite-time passage in either direction: equality at a boundary forces
    full equilibrium, which another orbit cannot reach in finite time.
    """

    source: SineExchangeComparison
    cycle: tuple[object, ...]
    cycle_indices: tuple[int, ...]
    cycle_raw_gap_bounds: tuple[I, ...]
    cycle_principal_gap_bounds: tuple[I, ...] | None
    edge_turn_offsets: tuple[int, ...] | None
    initial_winding: int | None
    initial_acute_margin_lower_bound: Q | None
    initial_receiver_phase_flat_certified: bool
    sector_half_width_bounds: I
    sector_margin_lower_bound: Q | None
    regional_phase_storage_bounds: I
    full_storage_bounds: I
    energy_barrier: Q
    energy_margin_lower_bound: Q
    strict_energy_bound_certified: bool
    sectors: tuple[SineCycleBarrierSector, ...]
    status: str
    reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_unit_sine_state_support_and_coefficients_readmitted",
        "selected_simple_C5_all_other_edges_nodes_and_storage_retained",
        "full_storage_rebuilt_from_primitives_not_cached_comparison_evidence",
        "principal_branch_ambiguity_does_not_select_an_integer_winding",
        "wider_sector_requires_winding_plus_or_minus_one_and_abs_gaps_below_two_pi_over_three",
        "full_storage_at_most_barrier_blocks_finite_time_sector_passage_in_both_directions",
        "boundary_energy_equality_forces_full_equilibrium_and_smooth_uniqueness_blocks_crossing",
        "outside_sector_excludes_acute_acquisition_without_assuming_initial_phase_consensus",
        "inside_sector_preserves_winding_not_strict_acuteness_or_a_prepared_reference_tube",
        "upper_storage_endpoint_must_not_exceed_barrier_no_inferred_exact_source_energy",
        "no_trajectory_forcing_event_source_selection_or_physical_identification",
    )
    finite_time_energy_bound_certified: bool = False

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.source)
        for node in self.cycle:
            _validate_label(node)
        return {"schema": "tnfr.sine-cycle-barrier.v1", "report": _project(self)}


def _admit_unit_cycle_source(source, cycle):
    """Admit one live conservative unit source and a selected simple C5."""
    if not isinstance(source, SineExchangeComparison):
        raise TypeError("a SineExchangeComparison is required")
    source, edges = _admit_sine_source(source)
    if _sine_model_coefficients(source.reference_model) != (0, 1, 1):
        raise ValueError("cycle analysis requires zero loss and unit exchange and beta")
    if any(value != 1 for value in source.capacity):
        raise ValueError("cycle analysis requires held unit capacities")
    labels = _ordered(cycle, "cycle", limit=6)
    positions = {node: i for i, node in enumerate(source.nodes)}
    if (
        len(labels) != 5
        or len(set(labels)) != 5
        or any(node not in positions for node in labels)
    ):
        raise ValueError("cycle must contain five distinct full-source node labels")
    rows = [[] for _ in source.nodes]
    for i, j in edges:
        rows[i].append(j)
        rows[j].append(i)
    neighbors = tuple(tuple(row) for row in rows)
    indices = _cycle_on_support(
        neighbors, tuple(positions[node] for node in labels), induced=False
    )
    return source, edges, neighbors, indices


def assess_sine_cycle_barrier(source, *, cycle) -> SineCycleBarrier:
    """Assess a supplied C5's wider winding sectors from conserved full storage.

    The complete source has zero loss, unit exchange/storage coefficients and
    held unit capacities. ``cycle`` supplies five distinct source labels in
    traversal order. Extra edges, including chords, remain in the full law
    and storage. No receiver, environmental or contact phase is presumed flat.

    Sector membership is independent of the energy test. With H<=7/2,
    certified membership preserves that wider sector for both time directions;
    certified nonmembership excludes every acute state of that winding at
    finite time. At a boundary equality forces vanishing full form differences,
    aligned extra-edge phases and balanced cycle currents: the complete field
    is zero. Uniqueness then excludes finite arrival from a distinct orbit.
    The legacy strict flag still requires H<7/2. No source equilibrium is
    inferred merely from its energy; failed bounds are not counterexamples.
    """
    from .relational_sine_comparison import (
        _comparison_from_state,
        _sine_state_from_rows,
    )

    original = source
    source, edges, neighbors, indices = _admit_unit_cycle_source(source, cycle)
    gaps = tuple(
        I(source.phase[j] - source.phase[i])
        for i, j in zip(indices, indices[1:] + indices[:1])
    )
    offsets, principal, winding, acute_margin = _branches(gaps)
    width = 2 * pi_interval() / 3
    sector_margin = (
        None if principal is None else min(width.lo - gap.abs_max for gap in principal)
    )
    # cos(delta)<=-1/2 proves exclusion from both open sectors even when an
    # antipodal gap prevents choosing a principal winding convention.
    gap_outside = any(cos(gap).hi <= -Q(1, 2) for gap in gaps)
    rebuilt = _comparison_from_state(
        _sine_state_from_rows(
            source.nodes,
            source.edges,
            source.epi,
            source.phase,
            source.capacity,
            neighbors,
        ),
        source.reference_model,
    )
    phase_storage = sum((1 - cos(gap) for gap in gaps), I(0))
    margin = C5_PHASE_SECTOR_BARRIER - rebuilt.storage.hi
    strict = margin > 0
    finite_time_bound = margin >= 0
    sectors = []
    for target in (-1, 1):
        if gap_outside or winding is not None and winding != target:
            membership = "outside"
        elif winding == target and sector_margin is not None and sector_margin > 0:
            membership = "inside"
        else:
            membership = "unavailable"
        protected = finite_time_bound and membership == "inside"
        excluded = finite_time_bound and membership == "outside"
        reasons = []
        if not finite_time_bound:
            reasons.append("full_storage_at_most_barrier_not_certified")
        if membership == "unavailable":
            reasons.append("initial_sector_membership_unresolved")
        status = (
            "sector_invariance_certified"
            if protected
            else "acute_acquisition_excluded" if excluded else "unavailable"
        )
        sectors.append(
            SineCycleBarrierSector(
                winding=target,
                source_membership=membership,
                sector_invariance_certified=protected,
                acute_acquisition_excluded=excluded,
                status=status,
                reasons=tuple(reasons),
            )
        )
    reasons = tuple(
        dict.fromkeys(reason for sector in sectors for reason in sector.reasons)
    )
    return SineCycleBarrier(
        source=original,
        cycle=tuple(source.nodes[i] for i in indices),
        cycle_indices=indices,
        cycle_raw_gap_bounds=gaps,
        cycle_principal_gap_bounds=principal,
        edge_turn_offsets=offsets,
        initial_winding=winding,
        initial_acute_margin_lower_bound=acute_margin,
        initial_receiver_phase_flat_certified=all(
            source.phase[i] == source.phase[indices[0]] for i in indices
        ),
        sector_half_width_bounds=width,
        sector_margin_lower_bound=sector_margin,
        regional_phase_storage_bounds=phase_storage,
        full_storage_bounds=rebuilt.storage,
        energy_barrier=C5_PHASE_SECTOR_BARRIER,
        energy_margin_lower_bound=margin,
        strict_energy_bound_certified=strict,
        finite_time_energy_bound_certified=finite_time_bound,
        sectors=tuple(sectors),
        status="certified" if not reasons else "unavailable",
        reasons=reasons,
    )


@dataclass(frozen=True)
class SineCycleRetention:
    """Finite acute-cycle retention from the conserved complete storage.

    The initial set is defined by exact independent form and radian phase
    errors at every source node. Returned interval observations enclose that
    exact set; they do not declare a larger independent preparation box.
    The same certificate holds forward and backward for the stated duration.
    """

    source: SineExchangeComparison
    cycle: tuple[object, ...]
    cycle_indices: tuple[int, ...]
    scaled_duration: Q
    source_error_bound: Q
    nominal_storage_bounds: I
    direct_storage_bounds: I
    taylor_storage_upper_bound: Q
    full_storage_bounds: I
    phase_storage_floor_bounds: I
    cycle_raw_gap_bounds: tuple[I, ...]
    cycle_principal_gap_bounds: tuple[I, ...] | None
    edge_turn_offsets: tuple[int, ...] | None
    initial_winding: int | None
    initial_acute_margin_lower_bounds: tuple[Q, ...] | None
    edge_cauchy_factors: tuple[Q, ...]
    available_form_storage_upper_bound: Q | None
    edge_phase_speed_upper_bounds: tuple[Q, ...] | None
    edge_phase_transport_upper_bounds: tuple[Q, ...] | None
    edge_retention_margin_lower_bounds: tuple[Q, ...] | None
    retention_margin_lower_bound: Q | None
    initial_acute_unit_winding_certified: bool
    whole_window_retention_certified: bool
    status: str
    reasons: tuple[str, ...]
    clock: str = "tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_conservative_unit_sine_law_with_held_support_and_capacity",
        "source_error_is_an_exact_independent_bound_for_every_form_and_radian_phase",
        "selected_simple_C5_can_have_live_chords_and_environmental_edges",
        "all_full_source_storage_is_rebuilt_without_using_cached_evidence",
        "direct_edge_and_correlated_taylor_upper_bounds_cover_the_same_exact_source_set",
        "acute_unit_cycle_phase_storage_is_at_least_five_times_one_minus_cos_two_pi_over_five",
        "full_degree_edge_cauchy_factors_bound_phase_speed_until_a_first_acute_exit",
        "strict_margin_excludes_first_exit_for_both_positive_and_negative_scaled_duration",
        "returned_interval_observations_do_not_enlarge_the_declared_source_error_set",
        "finite_acute_winding_retention_not_rigidity_attraction_or_indefinite_maintenance",
        "no_formation_claim_source_selection_new_law_solver_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.source)
        for node in self.cycle:
            _validate_label(node)
        return {"schema": "tnfr.sine-cycle-retention.v1", "report": _project(self)}


def assess_sine_cycle_retention(
    source, *, cycle, scaled_duration, source_error_bound=0
) -> SineCycleRetention:
    """Bound finite retention of an already acute unit-winding C5.

    Under x'=K S(theta), theta'=K L x in tau=t/pi, an oriented edge has
    |delta'|**2 <= 2*kappa*F, where kappa=1/d_i+1/d_j+2/(d_i*d_j).
    While the selected cycle is acute with winding +/-1, its phase storage
    is at least V5=5*(1-cos(2*pi/5)), so F<=H_upper-V5. Strict positive
    initial acute margins after subtracting duration times this speed bound
    prevent a first exit in either time direction.

    The full source's independently varying form and phase coordinates have
    error at most eps. Its storage is bounded both directly over edges and
    by H0+eps*(sum(abs(Lx))+sum(abs(S)))+4*m*eps**2 for m full edges.
    No loss, environmental freeze, external forcing or acquisition is assumed.
    """
    original = source
    source, edges, _, indices = _admit_unit_cycle_source(source, cycle)
    duration = _scalar(scaled_duration, "scaled_duration")
    error = _scalar(source_error_bound, "source_error_bound")
    if duration <= 0:
        raise ValueError("scaled_duration must be strictly positive")
    if error < 0:
        raise ValueError("source_error_bound must be nonnegative")
    nominal, direct = I(0), I(0)
    gradient = [Q(0)] * len(source.nodes)
    currents = [I(0)] * len(source.nodes)
    for i, j in edges:
        form_gap = source.epi[j] - source.epi[i]
        phase_gap = source.phase[j] - source.phase[i]
        nominal += form_gap**2 / 2 + 1 - cos(I(phase_gap))
        direct += (
            I(form_gap - 2 * error, form_gap + 2 * error) ** 2 / 2
            + 1
            - cos(I(phase_gap - 2 * error, phase_gap + 2 * error))
        )
        gradient[i] -= form_gap
        gradient[j] += form_gap
        current = sin(I(phase_gap))
        currents[i] += current
        currents[j] -= current
    taylor_upper = (
        nominal.hi
        + error
        * (
            sum(map(abs, gradient), Q(0))
            + sum((value.abs_max for value in currents), Q(0))
        )
        + 4 * len(edges) * error**2
    )
    storage = I(max(Q(0), direct.lo), min(direct.hi, taylor_upper))
    pi = pi_interval()
    floor = 5 * (1 - cos(2 * pi / 5))
    pairs = tuple(zip(indices, indices[1:] + indices[:1]))
    gaps = tuple(
        I(
            source.phase[j] - source.phase[i] - 2 * error,
            source.phase[j] - source.phase[i] + 2 * error,
        )
        for i, j in pairs
    )
    offsets, principal, winding, _ = _branches(gaps)
    margins = (
        None
        if principal is None
        else tuple(pi.lo / 2 - gap.abs_max for gap in principal)
    )
    factors = tuple(
        Q(1, source.degrees[i])
        + Q(1, source.degrees[j])
        + Q(2, source.degrees[i] * source.degrees[j])
        for i, j in pairs
    )
    initial = winding in (-1, 1) and margins is not None and min(margins) > 0
    budget = storage.hi - floor.lo if initial else None
    speeds = (
        None
        if budget is None
        else tuple(sqrt(I(2 * factor * budget)).hi for factor in factors)
    )
    transports = None if speeds is None else tuple(duration * speed for speed in speeds)
    retained_margins = (
        None
        if transports is None
        else tuple(margin - transport for margin, transport in zip(margins, transports))
    )
    retention_margin = None if retained_margins is None else min(retained_margins)
    certified = initial and retention_margin is not None and retention_margin > 0
    reasons = []
    if principal is None:
        reasons.append("initial_cycle_branches_not_certified")
    elif winding not in (-1, 1):
        reasons.append("initial_unit_winding_not_certified")
    elif min(margins) <= 0:
        reasons.append("initial_strict_acuteness_not_certified")
    elif not certified:
        reasons.append("strict_finite_retention_margin_not_certified")
    return SineCycleRetention(
        source=original,
        cycle=tuple(source.nodes[i] for i in indices),
        cycle_indices=indices,
        scaled_duration=duration,
        source_error_bound=error,
        nominal_storage_bounds=nominal,
        direct_storage_bounds=direct,
        taylor_storage_upper_bound=taylor_upper,
        full_storage_bounds=storage,
        phase_storage_floor_bounds=floor,
        cycle_raw_gap_bounds=gaps,
        cycle_principal_gap_bounds=principal,
        edge_turn_offsets=offsets,
        initial_winding=winding,
        initial_acute_margin_lower_bounds=margins,
        edge_cauchy_factors=factors,
        available_form_storage_upper_bound=budget,
        edge_phase_speed_upper_bounds=speeds,
        edge_phase_transport_upper_bounds=transports,
        edge_retention_margin_lower_bounds=retained_margins,
        retention_margin_lower_bound=retention_margin,
        initial_acute_unit_winding_certified=initial,
        whole_window_retention_certified=certified,
        status="certified" if certified else "unavailable",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class SineReversiblePreparationStep:
    """One predeclared endpoint's conditional preparation-to-target check."""

    index: int
    time: Q
    scaled_time_bounds: I
    source_form_center: tuple[Q, ...]
    source_phase_center: tuple[Q, ...]
    endpoint_error_bound: Q
    lipschitz_amplification_upper_bound: Q | None
    propagated_target_error_bound: Q | None
    target_radius_margin_lower_bound: Q | None
    source_cycle_principal_gap_bounds: tuple[I, ...] | None
    source_edge_turn_offsets: tuple[int, ...] | None
    source_initial_winding: int | None
    source_full_storage_bounds: I
    source_zero_winding_certified: bool
    source_target_energy_overlap: bool
    time_separation_certified: bool
    target_entry_certified: bool
    formation_retention_certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SineReversiblePreparation:
    """Conditional preparation from time-reversed validated endpoint evidence.

    The complete positive-time forecast starts at R(checkpoint), where
    R(x,theta)=(-x,theta). Endpoint midpoints supply exact source centers,
    never replacements for their retained enclosure radii. The endpoint
    Taylor proofs remain a declared numerical premise of this consumer.
    """

    forecast: SineForecast
    checkpoint: SineExchangeComparison
    cycle: tuple[object, ...]
    cycle_indices: tuple[int, ...]
    target_error_bound: Q
    source_error_bound: Q
    scaled_retention_duration: Q
    target_retention: SineCycleRetention
    steps: tuple[SineReversiblePreparationStep, ...]
    selected_step_index: int | None
    formation_time: Q | None
    horizon_complete: bool
    outcome: str
    reasons: tuple[str, ...]
    clock: str = "forecast_structural_t_retention_tau=t/pi"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_conservative_unit_sine_law_support_and_node_order",
        "R_reverses_all_forms_and_preserves_all_phase_lifts",
        "positive_time_forecast_initial_box_contains_exact_R_checkpoint",
        "full_forecast_chain_and_Picard_inclusions_readmitted_without_Taylor_proof_replay",
        "conclusions_conditional_on_supplied_validated_Taylor_endpoint_enclosures",
        "source_center_is_exact_reversed_endpoint_midpoint_with_all_halfwidths_retained",
        "source_and_target_are_exact_independent_coordinate_balls_not_rounded_output_boxes",
        "global_lipschitz_two_in_tau_propagates_endpoint_and_source_uncertainty",
        "every_certified_source_member_has_zero_winding_and_enters_the_retained_target",
        "energy_interval_overlap_is_necessary_only_not_a_reachability_proof",
        "first_success_on_the_declared_endpoint_grid_survives_a_later_unavailable_suffix",
        "no_backward_integrator_adaptive_source_scan_forecast_replay_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.checkpoint)
        _validate_comparison_labels(self.target_retention.source)
        for node in self.cycle:
            _validate_label(node)
        for node in self.target_retention.cycle:
            _validate_label(node)
        return {
            "schema": "tnfr.sine-reversible-preparation.v1",
            "report": _project(self),
        }


def assess_sine_reversible_preparation(
    forecast,
    checkpoint,
    *,
    cycle,
    target_error_bound,
    source_error_bound,
    scaled_retention_duration=1,
) -> SineReversiblePreparation:
    """Read a fixed reversible forecast as a conditional robust preparation.

    In original structural time the full field has Lipschitz constant 2/pi.
    For endpoint time A, reverse its midpoint to c and retain the largest
    form/phase halfwidth eta. Every point of the exact source ball B_delta(c)
    reaches B_epsilon(checkpoint) at time A if
    exp(2*A/pi)*(eta+delta)<epsilon. This conclusion uses reversibility and
    the endpoint inclusion, not a midpoint-only trajectory or negative-time
    integration. The target's acute retention is rebuilt independently.

    A source ball must have certified winding zero, compatible conserved
    storage, and A>pi*T for the target retention duration T in tau. All
    declared endpoints are assessed; a failed sufficient test is unavailable,
    not a proof that the corresponding actual trajectory fails to acquire.
    """
    original = checkpoint
    checkpoint, _, neighbors, indices = _admit_unit_cycle_source(checkpoint, cycle)
    target_error = _scalar(target_error_bound, "target_error_bound")
    source_error = _scalar(source_error_bound, "source_error_bound")
    retention_time = _scalar(scaled_retention_duration, "scaled_retention_duration")
    if target_error <= 0 or source_error <= 0 or retention_time <= 0:
        raise ValueError(
            "source and target errors and retention duration must be positive"
        )
    forecast = _admit_forecast(forecast)
    if forecast.observation_time != 0 or forecast.prior_admission is not None:
        raise ValueError(
            "reversible preparation requires an unconditioned forecast from time zero"
        )
    if forecast.forecast_start is not None:
        raise ValueError(
            "reversible preparation does not use a separate forecast start"
        )
    if tuple(tuple(sorted(row)) for row in forecast.neighbors) != tuple(
        tuple(sorted(row)) for row in neighbors
    ):
        raise ValueError("forecast support must match the checkpoint in its node order")
    size = len(checkpoint.nodes)
    reversed_checkpoint = (
        tuple(-value for value in checkpoint.epi) + checkpoint.phase + (Q(1),)
    )
    if any(
        not bound.contains(value)
        for bound, value in zip(forecast.initial_box, reversed_checkpoint)
    ):
        raise ValueError(
            "forecast initial box must contain every exact R(checkpoint) coordinate"
        )
    labels = tuple(checkpoint.nodes[i] for i in indices)
    target = assess_sine_cycle_retention(
        checkpoint,
        cycle=labels,
        scaled_duration=retention_time,
        source_error_bound=target_error,
    )
    pi = pi_interval()
    observations = []
    for index, step in enumerate(forecast.steps):
        time = step.time + step.duration
        endpoint = step.endpoint[: 2 * size]
        center = tuple(value.midpoint for value in endpoint)
        forms, phases = tuple(-value for value in center[:size]), center[size:]
        eta = max(value.radius for value in endpoint)
        amplified = exp_upper_float(2 * time / pi.lo)
        amplification = Q.from_float(amplified) if isfinite(amplified) else None
        propagated = (
            None if amplification is None else amplification * (eta + source_error)
        )
        radius_margin = None if propagated is None else target_error - propagated
        candidate = assess_sine_cycle_retention(
            replace(checkpoint, epi=forms, phase=phases),
            cycle=labels,
            scaled_duration=retention_time,
            source_error_bound=source_error,
        )
        zero = candidate.initial_winding == 0
        overlap = max(
            candidate.full_storage_bounds.lo, target.full_storage_bounds.lo
        ) <= min(candidate.full_storage_bounds.hi, target.full_storage_bounds.hi)
        separated = time > pi.hi * retention_time
        entered = radius_margin is not None and radius_margin > 0
        reasons = []
        for passed, reason in (
            (target.whole_window_retention_certified, "target_retention_not_certified"),
            (separated, "formation_time_not_beyond_backward_retention_window"),
            (amplification is not None, "lipschitz_amplification_not_representable"),
            (entered, "strict_source_to_target_radius_not_certified"),
            (zero, "whole_source_ball_zero_winding_not_certified"),
            (overlap, "source_target_storage_intervals_disjoint"),
        ):
            if not passed:
                reasons.append(reason)
        observations.append(
            SineReversiblePreparationStep(
                index=index,
                time=time,
                scaled_time_bounds=I(time) / pi,
                source_form_center=forms,
                source_phase_center=phases,
                endpoint_error_bound=eta,
                lipschitz_amplification_upper_bound=amplification,
                propagated_target_error_bound=propagated,
                target_radius_margin_lower_bound=radius_margin,
                source_cycle_principal_gap_bounds=candidate.cycle_principal_gap_bounds,
                source_edge_turn_offsets=candidate.edge_turn_offsets,
                source_initial_winding=candidate.initial_winding,
                source_full_storage_bounds=candidate.full_storage_bounds,
                source_zero_winding_certified=zero,
                source_target_energy_overlap=overlap,
                time_separation_certified=separated,
                target_entry_certified=entered,
                formation_retention_certified=not reasons,
                reasons=tuple(reasons),
            )
        )
    selected = next(
        (step for step in observations if step.formation_retention_certified), None
    )
    complete = forecast.validated_end_time == forecast.end_time
    outcome = (
        "certified"
        if selected is not None
        else ("no_certificate_on_declared_grid" if complete else "unavailable")
    )
    reasons = (
        ()
        if selected is not None
        else tuple(
            dict.fromkeys(
                (
                    *(
                        ("target_retention_not_certified",)
                        if not target.whole_window_retention_certified
                        else ()
                    ),
                    *(reason for step in observations for reason in step.reasons),
                    *(
                        ("forecast_validated_prefix_incomplete",)
                        if not complete
                        else ()
                    ),
                )
            )
        )
    )
    return SineReversiblePreparation(
        forecast=forecast,
        checkpoint=original,
        cycle=labels,
        cycle_indices=indices,
        target_error_bound=target_error,
        source_error_bound=source_error,
        scaled_retention_duration=retention_time,
        target_retention=target,
        steps=tuple(observations),
        selected_step_index=None if selected is None else selected.index,
        formation_time=None if selected is None else selected.time,
        horizon_complete=complete,
        outcome=outcome,
        reasons=reasons,
    )
