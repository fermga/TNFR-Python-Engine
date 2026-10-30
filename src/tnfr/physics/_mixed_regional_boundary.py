"""Conditional regional prediction for the acquired connected UM image.

The two-mean reference keeps the actual initial regional form and bounds its
discarded sine and conductance effects. It is not an exact coarse law. The
source, readiness and observation associations remain independent premises.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._comparison_flow import (
    MetzlerExponentialBounds,
    metzler_exponential_bounds,
)
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, pi_interval
from ._sine_coupling_successor import (
    _assess_sine_coupling_successor,
    _coupling_successor_field,
    _SineCouplingSuccessor,
)
from .relational_observations import _ordered

_COUNT = 18
_REGION_SIZE = 9
_CHARGE_TIME = _PROBE_TIME = Q(1, 32)
_AMPLITUDE = Q(1, 16)
_DELTA = Q(1, 1024)
_RADIUS = Q(1, 32)
_GAMMA_UPPER = Q(1, 3000)
_ETA = Q(20, 10**70)
_WEIGHT_DEVIATION = 10**9 * _ETA
_INITIAL_NORM = Q(9, 1024)
_NORM = Q(9, 512)
_MEAN_RADIUS = _NORM / 3
_RESIDUAL = Q(1, 10**5)
_SOURCE_CAP = Q(1, 10**10)
_MEAN_ERROR_CAP = _READOUT_ERROR_CAP = Q(1, 10**6)
_ARITHMETIC_BUDGET = Q(1, 10**35)
_PREDICTION_TARGET = Q(2, 10**6)
_SEPARATION_TARGET = Q(1, 1000)


def _nonnegative(value, name):
    value = _exact(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class _RegionalBoundaryReference:
    """A rational reference and its priced deviation from the actual projection."""

    coupling_bounds: tuple[Q, Q]
    reference_coupling: Q
    coupling_radius: Q
    generator: tuple[tuple[Q, ...], ...]
    exponential: MetzlerExponentialBounds
    gain_bounds: tuple[Q, Q]
    residual_calculated_upper_bound: Q
    residual_upper_bound: Q
    coefficient_domain_certified: bool
    residual_certified: bool
    gain_certified: bool


def _regional_reference():
    """Enclose the reference observation, not a nonlinear acquired trajectory."""
    coupling = Q(13, 27) - _DELTA / (18 * pi_interval())
    bounds = (coupling.lo, coupling.hi)
    midpoint = (bounds[0] + bounds[1]) / 2
    radius = (bounds[1] - bounds[0]) / 2
    coefficient_ok = 0 < bounds[0] <= midpoint <= bounds[1] < Q(1, 2)
    if not coefficient_ok:
        raise ValueError("regional reference coupling domain is unavailable")
    matrix = (
        (-1 - midpoint, midpoint, Q(0)),
        (midpoint, -midpoint, Q(0)),
        (Q(9), Q(0), Q(0)),
    )
    exponential = metzler_exponential_bounds(matrix, _PROBE_TIME)
    gain = (
        exponential.lower[2][0] - exponential.upper[2][1],
        exponential.upper[2][0] - exponential.lower[2][1],
    )
    residual = Q(10, 9) * _NORM * (_GAMMA_UPPER + _WEIGHT_DEVIATION)
    residual += Q(2, 3) * _NORM * radius
    return _RegionalBoundaryReference(
        coupling_bounds=bounds,
        reference_coupling=midpoint,
        coupling_radius=radius,
        generator=matrix,
        exponential=exponential,
        gain_bounds=gain,
        residual_calculated_upper_bound=residual,
        residual_upper_bound=_RESIDUAL,
        coefficient_domain_certified=coefficient_ok,
        residual_certified=residual < _RESIDUAL,
        gain_certified=(
            9 * _PROBE_TIME * (1 - _PROBE_TIME) <= gain[0] <= gain[1] <= 9 * _PROBE_TIME
        ),
    )


def _mixed_regional_boundary_field(
    *, new_edge_conductances, stage, charge_amplitude=Q(0)
):
    """Add balanced charging or a regional leak to the complete 36-row law.

    Charge supplies +a on nodes0..8 and -a on nodes9..17. Probe supplies -x
    only on nodes0..8. Phase, weights, capacity and bare degree9 are unchanged.
    The generic coefficient family does not authenticate an actual UM image.
    """
    if not isinstance(stage, str):
        raise TypeError("stage must be a string")
    if stage not in ("charge", "probe"):
        raise ValueError("stage must be 'charge' or 'probe'")
    amplitude = _exact(charge_amplitude, "charge_amplitude")
    if abs(amplitude) > _AMPLITUDE:
        raise ValueError("charge_amplitude must lie in [-1/16,1/16]")
    if stage == "probe" and amplitude != 0:
        raise ValueError("probe does not consume a charge amplitude")
    internal = _coupling_successor_field(
        completion="common_sense",
        new_edge_conductances=new_edge_conductances,
        loss_normalization="bare_degree",
    )

    def flow(values):
        state = _ordered(values, "state", limit=2 * _COUNT + 1)
        rates = internal(state)
        # The shared full-state reader admits all coordinates before arithmetic.
        if stage == "charge":
            form = tuple(
                rate + (amplitude if i < _REGION_SIZE else -amplitude)
                for i, rate in enumerate(rates[:_COUNT])
            )
        else:
            forms = state[:_REGION_SIZE]
            if not isinstance(forms[0], Jet):
                forms = tuple(I.coerce(value) for value in forms)
            form = (
                tuple(
                    rate - value
                    for rate, value in zip(rates[:_REGION_SIZE], forms, strict=True)
                )
                + rates[_REGION_SIZE:_COUNT]
            )
        return form + rates[_COUNT:]

    return flow


@dataclass(frozen=True)
class _RegionalTransferPrediction:
    """A conditional prediction from one associated regional mean aggregate.

    The original source and balanced input give mB=-mA at probe start. Neither
    that association nor acquisition is authenticated by interval consistency.
    The supplied observation interval is never clipped to its preparation box.
    """

    regional_mean_reading: Q
    regional_mean_error_bound: Q
    readout_error_bound: Q
    regional_mean_reading_bounds: tuple[Q, Q]
    preparation_mean_bounds: tuple[Q, Q]
    reference: _RegionalBoundaryReference
    gain_bounds: tuple[Q, Q]
    forecast: Q
    mean_error_contribution: Q
    model_error_upper_bound: Q
    arithmetic_error_upper_bound: Q
    record_prediction_error_upper_bound: Q
    true_transfer_bounds: tuple[Q, Q]
    recorded_transfer_bounds: tuple[Q, Q]
    mean_observation_consistent: bool
    allowances_admitted: bool
    arithmetic_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    probe_duration: Q = _PROBE_TIME
    regional_mean_error_cap: Q = _MEAN_ERROR_CAP
    readout_error_cap: Q = _READOUT_ERROR_CAP
    prediction_target: Q = _PREDICTION_TARGET
    arithmetic_budget: Q = _ARITHMETIC_BUDGET
    decoder_inputs: tuple[str, ...] = (
        "associated_probe_start_regional_mean_and_its_absolute_error",
        "independent_reserved_regional_transfer_readout_error_bound",
        "fixed_law_regions_actual_weight_association_and_balanced_preparation",
    )


def _predict_mixed_regional_transfer(
    *, regional_mean_reading, regional_mean_error_bound, readout_error_bound
):
    """Predict Q_A without a fine state, current or response as an input."""
    mean = _exact(regional_mean_reading, "regional_mean_reading")
    mean_error = _nonnegative(regional_mean_error_bound, "regional_mean_error_bound")
    readout = _nonnegative(readout_error_bound, "readout_error_bound")
    mean_bounds = (mean - mean_error, mean + mean_error)
    consistent = mean_bounds[0] <= _MEAN_RADIUS and mean_bounds[1] >= -_MEAN_RADIUS
    allowances = mean_error <= _MEAN_ERROR_CAP and readout <= _READOUT_ERROR_CAP
    reference = _regional_reference()
    gain = reference.gain_bounds
    midpoint, radius = (gain[0] + gain[1]) / 2, (gain[1] - gain[0]) / 2
    forecast = mean * midpoint
    arithmetic = abs(mean) * radius
    mean_contribution = 9 * _PROBE_TIME * mean_error
    model_error = 9 * _PROBE_TIME**2 * _RESIDUAL / 2
    error = mean_contribution + model_error + arithmetic + readout
    products = tuple(m * g for m in mean_bounds for g in gain)
    true_bounds = (min(products) - model_error, max(products) + model_error)
    record_bounds = (true_bounds[0] - readout, true_bounds[1] + readout)
    arithmetic_ok = arithmetic <= _ARITHMETIC_BUDGET
    unresolved = tuple(
        label
        for passed, label in (
            (allowances, "observation_allowance_outside_protocol"),
            (consistent, "regional_mean_inconsistent_with_preparation"),
            (reference.residual_certified, "projected_residual_allowance_unavailable"),
            (reference.gain_certified, "reference_gain_unavailable"),
            (arithmetic_ok, "arithmetic_allowance_unavailable"),
            (error < _PREDICTION_TARGET, "record_prediction_target_unavailable"),
        )
        if not passed
    )
    return _RegionalTransferPrediction(
        regional_mean_reading=mean,
        regional_mean_error_bound=mean_error,
        readout_error_bound=readout,
        regional_mean_reading_bounds=mean_bounds,
        preparation_mean_bounds=(-_MEAN_RADIUS, _MEAN_RADIUS),
        reference=reference,
        gain_bounds=gain,
        forecast=forecast,
        mean_error_contribution=mean_contribution,
        model_error_upper_bound=model_error,
        arithmetic_error_upper_bound=arithmetic,
        record_prediction_error_upper_bound=error,
        true_transfer_bounds=true_bounds,
        recorded_transfer_bounds=record_bounds,
        mean_observation_consistent=consistent,
        allowances_admitted=allowances,
        arithmetic_certified=arithmetic_ok,
        certified=not unresolved,
        status="certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )


@dataclass(frozen=True)
class _MixedRegionalBoundary:
    """Source-inclusive regional task bounds under independently supplied readiness.

    The tube includes common form created by the probe. Relative storage need
    not decrease under the regional input; its supplied work is separately
    bounded. The negative histories share the pre-probe global mean, not their
    phases, regional form or charging command. No finite age is evaluated.
    """

    form_error_bound: Q
    phase_error_bound: Q
    starting_gap_upper_bound: Q
    readout_error_bound: Q
    successor: _SineCouplingSuccessor
    reference: _RegionalBoundaryReference
    source_certified: bool
    finite_protocol_ready_age_exists: bool
    starting_gap_domain_admitted: bool
    conditional_ready_source: bool
    initial_norm_squared_upper_bound: Q
    initial_norm_margin: Q
    norm_tube_upper_bound: Q
    norm_tube_margin: Q
    full_state_tube_certified: bool
    marked_gap_change_upper_bound: Q
    retained_gap_upper_bound: Q
    registration_margin: Q
    joint_registration_retained_certified: bool
    charge_storage_work_upper_bound: Q
    probe_storage_work_upper_bound: Q
    gain_bounds: tuple[Q, Q]
    uniform_arithmetic_error_upper_bound: Q
    model_error_upper_bound: Q
    record_prediction_error_upper_bound: Q
    prediction_margin: Q
    charged_regional_mean_gap_lower_bound: Q
    actual_history_transfer_gap_lower_bound: Q
    recorded_history_transfer_gap_lower_bound: Q
    weak_prediction_error_lower_bound: Q
    separation_margin: Q
    positive_information_certified: bool
    global_history_obstruction_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    regions: tuple[tuple[int, ...], ...] = (tuple(range(9)), tuple(range(9, 18)))
    charge_duration: Q = _CHARGE_TIME
    probe_duration: Q = _PROBE_TIME
    charge_amplitude_bounds: tuple[Q, Q] = (-_AMPLITUDE, _AMPLITUDE)
    preparation_mean_bounds: tuple[Q, Q] = (-_MEAN_RADIUS, _MEAN_RADIUS)
    supplied_signed_form_amount: Q = Q(0)
    supplied_absolute_form_amount_upper_bound: Q = 18 * _AMPLITUDE * _CHARGE_TIME
    existence_gap_threshold: Q = _DELTA / 8
    retained_gap_threshold: Q = _DELTA / 4
    phase_radius: Q = _RADIUS
    gamma_upper_bound: Q = _GAMMA_UPPER
    edge_weight_deviation_upper_bound: Q = _WEIGHT_DEVIATION
    regional_mean_error_cap: Q = _MEAN_ERROR_CAP
    readout_error_cap: Q = _READOUT_ERROR_CAP
    prediction_target: Q = _PREDICTION_TARGET
    separation_target: Q = _SEPARATION_TARGET
    arithmetic_budget: Q = _ARITHMETIC_BUDGET
    controlled_charge_channels: int = 18
    controlled_probe_channels: int = 9
    predictor_regional_mean_coordinates: int = 1
    clock: str = "tau=(1023/1024)*t; both rows and supplied inputs use tau"
    scope: tuple[str, ...] = (
        "known_degree_loss_law_on_the_connected_actual_ideal_UM_image",
        "correlated_frozen_weights_not_an_arbitrary_equitable_carrier",
        "conditional_finite_ready_source_not_an_authenticated_endpoint",
        "balanced_supplied_charging_and_nine_channel_regional_form_leak",
        "bounded_reference_prediction_not_exact_scalar_or_two_mean_closure",
        "common_form_created_during_probe_is_retained_in_the_full_state_tube",
        "weak_history_excludes_command_and_regional_state_and_probe_global_mean",
        "no_event_source_replay_finite_reservoir_or_empirical_identification",
    )


def _assess_mixed_regional_boundary(
    *,
    form_error_bound,
    phase_error_bound,
    starting_gap_upper_bound,
    readout_error_bound,
):
    """Rebuild the original source and assess the fixed regional protocol.

    Starting gap and mean readings require independent association to the same
    actual source history. Existence concerns the fixed positive Delta/8 entry
    threshold; no finite attainment of a supplied zero bound is inferred.
    """
    rx, rt, gap, readout = (
        _nonnegative(value, name)
        for value, name in (
            (form_error_bound, "form_error_bound"),
            (phase_error_bound, "phase_error_bound"),
            (starting_gap_upper_bound, "starting_gap_upper_bound"),
            (readout_error_bound, "readout_error_bound"),
        )
    )
    successor = _assess_sine_coupling_successor(
        form_error_bound=rx, phase_error_bound=rt
    )
    common = successor.completion_reports[0]
    geometry = common.geometry
    source_ok = (
        max(rx, rt) <= _SOURCE_CAP
        and successor.form_error_bound == rx
        and successor.phase_error_bound == rt
        and successor.source_certified
        and successor.isolated_waiting_certified
        and successor.geometry_certified
        and successor.conditional_image_certified
        and successor.phase_radius == _RADIUS
        and successor.relative_phase_gap == _DELTA
        and successor.registered_gap_target == _DELTA / 8
        and successor.gamma_range[1] <= _GAMMA_UPPER
        and successor.waiting_radius <= _ETA
        and successor.edge_weight_deviation_upper_bound <= _WEIGHT_DEVIATION
        and successor.new_weight_range == (Q(3, 4), Q(1))
        and geometry.completion == "common_sense"
        and geometry.degree == 9
        and geometry.degrees == (9,) * _COUNT
        and geometry.nodes == tuple(range(_COUNT))
        and geometry.held_capacities == (Q(1),) * _COUNT
        and geometry.components == (tuple(range(_COUNT)),)
        and len(geometry.edge_pairs) == 81
        and len(geometry.new_edge_pairs) == 63
        and geometry.support_certified
        and geometry.reference_stationary_certified
        and geometry.spectral_gap_certified
        and geometry.quotient_coercivity_lower_bound >= Q(1, 5)
        and common.constituent_identity_certified
        and common.initial_quotient_radius_margin > 0
        and common.storage_margin > 0
        and common.initial_energy_upper_bound < Q(1, 65536)
        and common.degree_form_limit_bounds == ((Q(0), Q(0)),)
        and successor.precontact_reference.initial_form_means == (Q(0), Q(0))
    )
    exists = (
        source_ok
        and common.asymptotic_convergence_certified
        and common.joint_registration_certified
    )
    gap_admitted = gap <= _DELTA / 8
    ready = source_ok and gap_admitted
    initial_squared = 5 * common.initial_energy_upper_bound
    initial_margin = _INITIAL_NORM**2 - initial_squared
    exponent = 2 * _GAMMA_UPPER * (_CHARGE_TIME + _PROBE_TIME)
    growth = Q(36, 35)
    tube = (_INITIAL_NORM + Q(17, 4) * _AMPLITUDE * _CHARGE_TIME) * growth
    tube_margin = _RADIUS - tube
    tube_ok = (
        source_ok
        and initial_margin > 0
        and exponent < 1
        and 1 / (1 - exponent) < growth
        and tube <= _NORM
        and tube_margin > 0
    )
    motion = 4 * _GAMMA_UPPER * _RADIUS * (_CHARGE_TIME + _PROBE_TIME)
    retained_gap = gap + motion
    registration_margin = _DELTA / 4 - retained_gap
    registration_ok = ready and tube_ok and registration_margin > 0
    reference = _regional_reference()
    gain = reference.gain_bounds
    arithmetic = (_MEAN_RADIUS + _MEAN_ERROR_CAP) * (gain[1] - gain[0]) / 2
    model_error = 9 * _PROBE_TIME**2 * _RESIDUAL / 2
    prediction_error = (
        9 * _PROBE_TIME * _MEAN_ERROR_CAP + model_error + readout + arithmetic
    )
    prediction_margin = _PREDICTION_TARGET - prediction_error
    mean_gap = (
        2 * _AMPLITUDE * _CHARGE_TIME * (1 - _CHARGE_TIME / 2)
        - 2 * _RESIDUAL * _CHARGE_TIME
    )
    actual_gap = (
        18
        * _AMPLITUDE
        * _CHARGE_TIME
        * _PROBE_TIME
        * (1 - _CHARGE_TIME / 2)
        * (1 - _PROBE_TIME)
        - (18 * _PROBE_TIME * _CHARGE_TIME + 9 * _PROBE_TIME**2) * _RESIDUAL
    )
    recorded_gap = actual_gap - 2 * readout
    weak_error = recorded_gap / 2
    separation_margin = recorded_gap - _SEPARATION_TARGET
    observation_ok = readout <= _READOUT_ERROR_CAP
    arithmetic_ok = arithmetic <= _ARITHMETIC_BUDGET
    reference_ok = reference.residual_certified and reference.gain_certified
    positive = (
        registration_ok
        and reference_ok
        and observation_ok
        and arithmetic_ok
        and prediction_margin > 0
    )
    obstruction = (
        registration_ok
        and reference_ok
        and observation_ok
        and separation_margin > 0
        and weak_error > _PREDICTION_TARGET
    )
    unresolved = tuple(
        label
        for passed, label in (
            (source_ok, "original_connected_source_unavailable"),
            (exists, "finite_protocol_ready_age_existence_unavailable"),
            (gap_admitted, "supplied_starting_gap_outside_protocol"),
            (tube_ok, "full_state_tube_unavailable"),
            (registration_ok, "finite_joint_registration_retention_unavailable"),
            (reference_ok, "regional_reference_allowance_unavailable"),
            (observation_ok, "transfer_readout_allowance_outside_protocol"),
            (arithmetic_ok, "uniform_arithmetic_allowance_unavailable"),
            (positive, "regional_mean_prediction_unavailable"),
            (obstruction, "global_history_obstruction_unavailable"),
        )
        if not passed
    )
    return _MixedRegionalBoundary(
        form_error_bound=rx,
        phase_error_bound=rt,
        starting_gap_upper_bound=gap,
        readout_error_bound=readout,
        successor=successor,
        reference=reference,
        source_certified=source_ok,
        finite_protocol_ready_age_exists=exists,
        starting_gap_domain_admitted=gap_admitted,
        conditional_ready_source=ready,
        initial_norm_squared_upper_bound=initial_squared,
        initial_norm_margin=initial_margin,
        norm_tube_upper_bound=tube,
        norm_tube_margin=tube_margin,
        full_state_tube_certified=tube_ok,
        marked_gap_change_upper_bound=motion,
        retained_gap_upper_bound=retained_gap,
        registration_margin=registration_margin,
        joint_registration_retained_certified=registration_ok,
        charge_storage_work_upper_bound=18
        * _NORM
        * Q(17, 4)
        * _AMPLITUDE
        * _CHARGE_TIME,
        probe_storage_work_upper_bound=18 * _NORM**2 * _PROBE_TIME,
        gain_bounds=gain,
        uniform_arithmetic_error_upper_bound=arithmetic,
        model_error_upper_bound=model_error,
        record_prediction_error_upper_bound=prediction_error,
        prediction_margin=prediction_margin,
        charged_regional_mean_gap_lower_bound=mean_gap,
        actual_history_transfer_gap_lower_bound=actual_gap,
        recorded_history_transfer_gap_lower_bound=recorded_gap,
        weak_prediction_error_lower_bound=weak_error,
        separation_margin=separation_margin,
        positive_information_certified=positive,
        global_history_obstruction_certified=obstruction,
        certified=not unresolved,
        status="certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
