"""A conditional finite uniform-boundary task for the actual UM composite.

The known degree-loss law and supplied input balance close this particular
transfer observation. They do not turn its scalar predictor into a complete
NFR state, authenticate a ready endpoint or select the boundary interaction.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from .._exact_time import exp_unit_bounds
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I
from ._sine_coupling_successor import (
    _assess_sine_coupling_successor,
    _coupling_successor_field,
    _SineCouplingSuccessor,
)
from .relational_observations import _ordered

_COUNT = 18
_CHARGE_TIME = Q(1, 32)
_PROBE_TIME = Q(1, 32)
_AMPLITUDE = Q(1, 16)
_MEAN_RADIUS = _AMPLITUDE * _CHARGE_TIME
_DELTA = Q(1, 1024)
_PHASE_RADIUS = Q(1, 32)
_SOURCE_CAP = Q(1, 10**10)
_MEAN_ERROR_CAP = Q(1, 10**6)
_READOUT_ERROR_CAP = Q(1, 10**6)
_PREDICTION_TARGET = Q(2, 10**6)
_SEPARATION_TARGET = Q(1, 1000)
_ARITHMETIC_BUDGET = Q(1, 10**40)


def _nonnegative(value, name):
    value = _exact(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _transfer_gain_bounds():
    """Exact rational enclosure of the fixed task gain, not a trajectory."""
    lower, upper = exp_unit_bounds(_PROBE_TIME)
    return _COUNT * (1 - 1 / lower), _COUNT * (1 - 1 / upper)


def _mixed_composite_boundary_field(
    *, new_edge_conductances, stage, charge_amplitude=Q(0)
):
    """Add the declared form input to all 36 unchanged internal law rows.

    ``charge`` supplies amplitude times the all-ones form vector;
    ``probe`` supplies minus the current form. There is no phase input or
    new neighbor. Both stages use structural time tau=(1023/1024)*t.
    Generic legal coefficients and states are not authenticated UM images.
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
        # Shared full-state admission precedes any extra row arithmetic.
        if stage == "charge":
            form = tuple(value + amplitude for value in rates[:_COUNT])
        else:
            forms = state[:_COUNT]
            if not isinstance(forms[0], Jet):
                forms = tuple(I.coerce(value) for value in forms)
            form = tuple(
                rate - value for rate, value in zip(rates[:_COUNT], forms, strict=True)
            )
        return form + rates[_COUNT:]

    return flow


@dataclass(frozen=True)
class _BoundaryTransferPrediction:
    """A prediction conditional on the mean reading's charged-state association.

    Interval consistency is necessary but does not authenticate that association.
    Preparation bounds are not used to clamp the supplied reading interval.
    A mean is one aggregate datum, not one free nodal measurement.
    """

    mean_reading: Q
    mean_error_bound: Q
    readout_error_bound: Q
    mean_reading_bounds: tuple[Q, Q]
    preparation_mean_bounds: tuple[Q, Q]
    gain_bounds: tuple[Q, Q]
    forecast: Q
    mean_error_contribution: Q
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
    node_count: int = _COUNT
    probe_duration: Q = _PROBE_TIME
    boundary_rate: Q = Q(1)
    boundary_reference: Q = Q(0)
    mean_error_cap: Q = _MEAN_ERROR_CAP
    readout_error_cap: Q = _READOUT_ERROR_CAP
    prediction_target: Q = _PREDICTION_TARGET
    arithmetic_budget: Q = _ARITHMETIC_BUDGET
    decoder_inputs: tuple[str, ...] = (
        "associated_charged_state_mean_reading_and_its_absolute_error",
        "independent_reserved_transfer_readout_error_bound",
        "fixed_complete_law_boundary_clock_and_preparation_metadata",
    )


def _predict_mixed_composite_transfer(
    *, mean_reading, mean_error_bound, readout_error_bound
):
    """Predict the fixed transfer record without any fine state or response input.

    The actual mean must differ from the supplied reading by at most its
    declared error. The report is conditional on that independent premise.
    Nonnegative unsupported errors or inconsistent observations are unavailable.
    """
    mean = _exact(mean_reading, "mean_reading")
    mean_error = _nonnegative(mean_error_bound, "mean_error_bound")
    readout_error = _nonnegative(readout_error_bound, "readout_error_bound")
    mean_bounds = (mean - mean_error, mean + mean_error)
    consistent = mean_bounds[0] <= _MEAN_RADIUS and mean_bounds[1] >= -_MEAN_RADIUS
    allowances = mean_error <= _MEAN_ERROR_CAP and readout_error <= _READOUT_ERROR_CAP
    gain = _transfer_gain_bounds()
    midpoint, radius = (gain[0] + gain[1]) / 2, (gain[1] - gain[0]) / 2
    forecast = mean * midpoint
    arithmetic = abs(mean) * radius
    mean_contribution = gain[1] * mean_error
    error = mean_contribution + arithmetic + readout_error
    products = tuple(m * g for m in mean_bounds for g in gain)
    true_bounds = (min(products), max(products))
    record_bounds = (true_bounds[0] - readout_error, true_bounds[1] + readout_error)
    arithmetic_ok = arithmetic <= _ARITHMETIC_BUDGET
    unresolved = tuple(
        label
        for passed, label in (
            (allowances, "observation_allowance_outside_protocol"),
            (consistent, "mean_observation_inconsistent_with_preparation"),
            (arithmetic_ok, "arithmetic_allowance_unavailable"),
            (error <= _PREDICTION_TARGET, "record_prediction_target_unavailable"),
        )
        if not passed
    )
    return _BoundaryTransferPrediction(
        mean_reading=mean,
        mean_error_bound=mean_error,
        readout_error_bound=readout_error,
        mean_reading_bounds=mean_bounds,
        preparation_mean_bounds=(-_MEAN_RADIUS, _MEAN_RADIUS),
        gain_bounds=gain,
        forecast=forecast,
        mean_error_contribution=mean_contribution,
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
class _MixedCompositeBoundary:
    """Source-inclusive task bounds, conditional on independently supplied readiness.

    Existence concerns the fixed Delta/8 readiness threshold, not attainment of
    an arbitrary tighter supplied bound (including zero). No numerical age,
    endpoint authentication, complete word or physical exchange is certified.
    """

    form_error_bound: Q
    phase_error_bound: Q
    starting_gap_upper_bound: Q
    readout_error_bound: Q
    successor: _SineCouplingSuccessor
    source_certified: bool
    finite_protocol_ready_age_exists: bool
    starting_gap_domain_admitted: bool
    conditional_ready_source: bool
    quotient_retention_certified: bool
    joint_registration_retained_certified: bool
    marked_gap_change_upper_bound: Q
    retained_gap_upper_bound: Q
    registration_margin: Q
    gain_bounds: tuple[Q, Q]
    uniform_arithmetic_error_upper_bound: Q
    record_prediction_error_upper_bound: Q
    prediction_margin: Q
    actual_history_transfer_gap_bounds: tuple[Q, Q]
    recorded_history_transfer_gap_lower_bound: Q
    analytic_history_gap_lower_bound: Q
    weak_prediction_error_lower_bound: Q
    separation_margin: Q
    positive_information_certified: bool
    relative_history_obstruction_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    charge_duration: Q = _CHARGE_TIME
    probe_duration: Q = _PROBE_TIME
    charge_amplitude_bounds: tuple[Q, Q] = (-_AMPLITUDE, _AMPLITUDE)
    preparation_mean_bounds: tuple[Q, Q] = (-_MEAN_RADIUS, _MEAN_RADIUS)
    supplied_absolute_form_amount_upper_bound: Q = _COUNT * _MEAN_RADIUS
    existence_gap_threshold: Q = _DELTA / 8
    retained_gap_threshold: Q = _DELTA / 4
    phase_radius: Q = _PHASE_RADIUS
    gamma_upper_bound: Q = Q(1, 3000)
    mean_error_cap: Q = _MEAN_ERROR_CAP
    readout_error_cap: Q = _READOUT_ERROR_CAP
    prediction_target: Q = _PREDICTION_TARGET
    separation_target: Q = _SEPARATION_TARGET
    arithmetic_budget: Q = _ARITHMETIC_BUDGET
    boundary_rate: Q = Q(1)
    boundary_reference: Q = Q(0)
    controlled_form_channels: int = _COUNT
    predictor_mean_coordinates: int = 1
    clock: str = "tau=(1023/1024)*t; both rows and supplied inputs use tau"
    scope: tuple[str, ...] = (
        "known_bare_degree_law_on_the_connected_actual_ideal_UM_image",
        "conditional_finite_ready_source_not_a_sampled_or_limiting_endpoint",
        "supplied_uniform_charge_and_state_dependent_form_only_boundary",
        "exact_task_balance_not_a_complete_scalar_NFR_or_winding_caused_closure",
        "retained_signed_charge_integral_can_replace_an_associated_mean_reading",
        "weaker_information_discards_both_charge_integral_and_absolute_mean",
        "internal_gradient_storage_does_not_price_common_form_as_physical_energy",
        "no_source_replay_event_execution_finite_reservoir_or_empirical_bridge",
    )


def _assess_mixed_composite_boundary(
    *,
    form_error_bound,
    phase_error_bound,
    starting_gap_upper_bound,
    readout_error_bound,
):
    """Rebuild source premises and assess the fixed two-stage boundary protocol.

    The supplied starting bound must concern all nine actual marked gaps at
    the same finite starting state. Its value alone supplies no observation
    evidence. All original fine errors, held weights and current form survive.
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
    source_ok = (
        rx <= _SOURCE_CAP
        and rt <= _SOURCE_CAP
        and successor.form_error_bound == rx
        and successor.phase_error_bound == rt
        and successor.source_certified
        and successor.isolated_waiting_certified
        and successor.geometry_certified
        and successor.conditional_image_certified
        and successor.phase_radius == _PHASE_RADIUS
        and successor.relative_phase_gap == _DELTA
        and successor.registered_gap_target == _DELTA / 8
        and successor.gamma_range[1] <= Q(1, 3000)
        and common.geometry.completion == "common_sense"
        and common.geometry.degree == 9
        and common.geometry.nodes == tuple(range(_COUNT))
        and common.geometry.held_capacities == (Q(1),) * _COUNT
        and common.geometry.components == (tuple(range(_COUNT)),)
        and len(common.geometry.edge_pairs) == 81
        and common.constituent_identity_certified
        and common.initial_quotient_radius_margin > 0
        and common.storage_margin > 0
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
    motion = 4 * Q(1, 3000) * _PHASE_RADIUS * (_CHARGE_TIME + _PROBE_TIME)
    retained_gap = gap + motion
    registration_margin = _DELTA / 4 - retained_gap
    registration_ok = ready and registration_margin > 0
    gain = _transfer_gain_bounds()
    arithmetic = (_MEAN_RADIUS + _MEAN_ERROR_CAP) * (gain[1] - gain[0]) / 2
    prediction_error = gain[1] * _MEAN_ERROR_CAP + readout + arithmetic
    prediction_margin = _PREDICTION_TARGET - prediction_error
    actual_gap = tuple(2 * _MEAN_RADIUS * value for value in gain)
    recorded_gap = actual_gap[0] - 2 * readout
    analytic_gap = (
        2 * _COUNT * _MEAN_RADIUS * (_PROBE_TIME - _PROBE_TIME**2 / 2) - 2 * readout
    )
    weak_error = recorded_gap / 2
    separation_margin = recorded_gap - _SEPARATION_TARGET
    observation_ok = readout <= _READOUT_ERROR_CAP
    arithmetic_ok = arithmetic <= _ARITHMETIC_BUDGET
    positive = (
        registration_ok and observation_ok and arithmetic_ok and prediction_margin >= 0
    )
    obstruction = (
        registration_ok
        and observation_ok
        and analytic_gap > _SEPARATION_TARGET
        and separation_margin > 0
        and weak_error > _PREDICTION_TARGET
    )
    unresolved = tuple(
        label
        for passed, label in (
            (source_ok, "original_connected_source_unavailable"),
            (exists, "finite_protocol_ready_age_existence_unavailable"),
            (gap_admitted, "supplied_starting_gap_outside_protocol"),
            (registration_ok, "finite_joint_registration_retention_unavailable"),
            (observation_ok, "transfer_readout_allowance_outside_protocol"),
            (arithmetic_ok, "uniform_arithmetic_allowance_unavailable"),
            (positive, "mean_information_prediction_unavailable"),
            (obstruction, "relative_history_obstruction_unavailable"),
        )
        if not passed
    )
    return _MixedCompositeBoundary(
        form_error_bound=rx,
        phase_error_bound=rt,
        starting_gap_upper_bound=gap,
        readout_error_bound=readout,
        successor=successor,
        source_certified=source_ok,
        finite_protocol_ready_age_exists=exists,
        starting_gap_domain_admitted=gap_admitted,
        conditional_ready_source=ready,
        quotient_retention_certified=source_ok,
        joint_registration_retained_certified=registration_ok,
        marked_gap_change_upper_bound=motion,
        retained_gap_upper_bound=retained_gap,
        registration_margin=registration_margin,
        gain_bounds=gain,
        uniform_arithmetic_error_upper_bound=arithmetic,
        record_prediction_error_upper_bound=prediction_error,
        prediction_margin=prediction_margin,
        actual_history_transfer_gap_bounds=actual_gap,
        recorded_history_transfer_gap_lower_bound=recorded_gap,
        analytic_history_gap_lower_bound=analytic_gap,
        weak_prediction_error_lower_bound=weak_error,
        separation_margin=separation_margin,
        positive_information_certified=positive,
        relative_history_obstruction_certified=obstruction,
        certified=not unresolved,
        status="certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
