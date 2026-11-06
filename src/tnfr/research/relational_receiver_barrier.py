"""One frozen full-state prefix and dissipative-tail receiver test.

The source is the existing localized H=2 preparation. No source, horizon or
numerical parameter is fitted to its response. A failed enclosure or margin
does not predict receiver passage. Evidence transport belongs to benchmarks.
"""

from fractions import Fraction as Q

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval
from ..physics.relational_sine_forecast import SineForecast, bound_sine_flow
from ..physics.relational_sine_formation import assess_sine_mediated_formation
from ..sdk.relational_reports import _project
from .relational_acquisition import _exact, _require

__all__ = (
    "prepare_receiver_barrier",
    "assess_receiver_barrier_forecast",
    "evaluate_receiver_barrier",
)

_HORIZON, _STEP, _ORDER = Q(32), Q(1, 8), 16
_BARRIER = Q(7, 2)


def _preparation():
    model = RelationalExchangeModel(1, phase_domain="regular")
    source = assess_sine_mediated_formation(
        model=model, amplitude=2, capacity_contrast=Q(1, 2)
    )
    initial = (
        tuple(I(value) for value in source.initial_epi)
        + tuple(2 * pi_interval() * turn for turn in source.initial_phase_turns)
        + (I(source.capacity[-1]),)
    )
    return source, initial


def prepare_receiver_barrier():
    """Declare the complete source and numerical verdict before propagation."""
    source, initial = _preparation()
    return {
        "schema": "tnfr.sine-receiver-barrier-protocol.v1",
        "source": "localized_H2_donor_plus_one_flat_receiver",
        "nodes": source.nodes,
        "edges": source.edges,
        "neighbors": source.neighbors,
        "initial_form": _project(source.initial_epi),
        "initial_phase_turns": _project(source.initial_phase_turns),
        "held_capacity": _project(source.capacity),
        "initial_box": _project(initial),
        "state_layout": "x0..x10,theta0..theta10,held_nu10",
        "model": {
            "law": source.law,
            "effective_weights": _project(source.model.effective_weights),
            "storage_scale": 1,
            "support": "fixed_simple_connected_unit_edges",
            "input": "absent",
            "events": "none",
            "capacity": "held",
        },
        "clock": "original_structural_clock",
        "observation_time": 0,
        "end_time": _project(_HORIZON),
        "time_step": _project(_STEP),
        "order": _ORDER,
        "maximum_steps": 256,
        "arithmetic": INTERVAL_METHOD,
        "rounding": "exact_phase_turns_enclosed_with_mathematical_pi_bounds",
        "receiver_nodes": source.cycles[1],
        "barrier": _project(_BARRIER),
        "acceptance": (
            "complete_frozen_horizon; every_whole_time_receiver_potential_upper<7/2; "
            "full_endpoint_total_storage_upper<7/2"
        ),
        "failure_policy": (
            "retain_unavailable_or_unresolved_result; no_adaptive_retries; "
            "no_early_tail_closure_or_horizon_reinterpretation"
        ),
        "conclusion_scope": (
            "conditional_all_future_receiver_potential_barrier_exclusion; "
            "not_all_time_zero_winding_or_a_selected_donor_endpoint; "
            "not_physical_validation"
        ),
    }


def _storage(box, edges, *, phase_only=False):
    """Full-edge interval storage with beta=1 in the frozen coordinate layout."""
    _require(
        len(box) == 23 and all(isinstance(value, I) for value in box),
        "expected the full interval state",
    )
    potential = sum((1 - cos(box[11 + j] - box[11 + i]) for i, j in edges), I(0))
    if phase_only:
        return potential
    return potential + sum(((box[j] - box[i]) ** 2 / 2 for i, j in edges), I(0))


def assess_receiver_barrier_forecast(forecast):
    """Check coverage and margins of shared solver evidence, without replaying it.

    Public dataclass records do not authenticate their production. This reader
    checks the frozen declaration and consumed enclosure structure; the source
    archive and producer retain numerical provenance separately.
    """
    _require(isinstance(forecast, SineForecast), "a shared sine forecast is required")
    source, initial = _preparation()
    _require(
        forecast.model == source.model
        and forecast.neighbors == source.neighbors
        and tuple(_exact(value) for value in forecast.visible_capacity)
        == source.capacity[:-1]
        and forecast.initial_box == initial
        and _exact(forecast.observation_time) == 0
        and _exact(forecast.end_time) == _HORIZON
        and _exact(forecast.time_step) == _STEP
        and type(forecast.order) is int
        and forecast.order == _ORDER
        and forecast.prior_admission is None
        and forecast.forecast_start is None
        and forecast.freeze_hidden is False,
        "forecast does not match the frozen source and budget",
    )
    _require(forecast.status in ("admitted", "unavailable"), "invalid forecast status")
    _require(
        forecast.status != "admitted" or not forecast.reasons,
        "an admitted forecast cannot retain failure reasons",
    )
    _require(len(forecast.steps) <= 256, "forecast exceeds the frozen step budget")
    receiver = frozenset(source.cycles[1])
    receiver_edges = tuple(
        (i, j) for i, j in source.edges if i in receiver and j in receiver
    )
    at, previous, receiver_upper = Q(0), initial, []
    for step in forecast.steps:
        _require(
            _exact(step.time) == at and _exact(step.duration) == _STEP,
            "step coverage differs from the frozen grid",
        )
        _require(
            _exact(step.picard_interior_margin) > 0
            and step.domain_lower_bounds
            and min(_exact(value) for value in step.domain_lower_bounds) > 0,
            "step lacks positive enclosure admission",
        )
        _require(
            len(step.tube) == len(step.endpoint) == len(previous) == 23,
            "step drops a full-state coordinate",
        )
        _require(
            all(isinstance(value, I) for value in step.tube + step.endpoint)
            and all(
                tube.lo <= value.lo <= value.hi <= tube.hi
                for box in (previous, step.endpoint)
                for value, tube in zip(box, step.tube)
            ),
            "step tube does not retain its complete initial and endpoint boxes",
        )
        _require(
            step.endpoint[-1] == initial[-1],
            "step endpoint changes the exactly held intermediary capacity",
        )
        receiver_upper.append(_storage(step.tube, receiver_edges, phase_only=True).hi)
        at, previous = at + _STEP, step.endpoint
    _require(
        _exact(forecast.validated_end_time) == at and forecast.endpoint == previous,
        "forecast endpoint does not match retained step coverage",
    )
    complete = forecast.status == "admitted" and at == _HORIZON
    _require(
        forecast.status != "admitted" or complete,
        "an admitted forecast must reach its declared horizon",
    )
    peak = max(receiver_upper) if receiver_upper else None
    prefix_margin = None if peak is None else _BARRIER - peak
    endpoint_storage = _storage(forecast.endpoint, source.edges)
    tail_margin = _BARRIER - endpoint_storage.hi
    prefix_passed = complete and prefix_margin is not None and prefix_margin > 0
    tail_passed = complete and tail_margin > 0
    certified = prefix_passed and tail_passed
    if not complete:
        status = "unavailable_prefix"
    elif not prefix_passed:
        status = "unresolved_receiver_prefix"
    elif not tail_passed:
        status = "unresolved_tail"
    else:
        status = "certified_receiver_barrier_exclusion"
    return {
        "status": status,
        "validated_end_time": at,
        "validated_steps": len(forecast.steps),
        "full_horizon_completed": complete,
        "receiver_potential_upper_by_tube": tuple(receiver_upper),
        "receiver_potential_upper_on_validated_prefix": peak,
        "receiver_prefix_margin": prefix_margin,
        "endpoint_total_storage_bounds": endpoint_storage,
        "endpoint_tail_margin": tail_margin,
        "full_prefix_passed": prefix_passed,
        "full_horizon_tail_passed": tail_passed,
        "all_future_receiver_potential_barrier_excluded": certified,
        "maintained_receiver_twists_excluded": certified,
        "asymptotic_receiver_consensus_certified": certified,
        "numerical_reasons": forecast.reasons,
        "scope": (
            "whole_time_prefix_plus_monotone_full_storage_tail",
            "partial_endpoint_does_not_replace_the_frozen_horizon",
            "unresolved_upper_bound_does_not_prove_a_barrier_crossing",
            "no_all_time_winding_or_donor_endpoint_claim",
            "validated_step_records_are_not_provenance_authentication",
        ),
    }


def evaluate_receiver_barrier():
    """Evaluate exactly one predeclared source, retaining failed solver evidence."""
    source, initial = _preparation()
    forecast = bound_sine_flow(
        initial,
        neighbors=source.neighbors,
        visible_capacity=source.capacity[:-1],
        model=source.model,
        observation_time=0,
        end_time=_HORIZON,
        time_step=_STEP,
        order=_ORDER,
    )
    return {
        "schema": "tnfr.sine-receiver-barrier-response.v1",
        "declaration": prepare_receiver_barrier(),
        "assessment": {
            key: _project(value)
            for key, value in assess_receiver_barrier_forecast(forecast).items()
        },
        "forecast": forecast.to_dict(),
    }
