"""One fixed prospective software forecast, consuming public prior evidence only.

The source preparation and reserved response belong to a separate module.
This consumer has no hidden-state parameter or response-reader callback.
"""

from fractions import Fraction as Q

import networkx as nx

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I
from ..physics.relational_sine_observation import infer_relational_sine_hidden_state
from ..sdk.relational_reports import _project
from ..utils.io import json_dumps
from .relational_acquisition import _exact, _pair, _require

__all__ = (
    "prepare_sine_prior_forecast",
    "predict_sine_prior_forecast",
    "assess_sine_prior_response",
)

_HORIZON, _STEP, _ORDER = Q(1, 16), Q(1, 128), 6
_MAX_WIDTH, _MIN_GAP = Q(1, 10**6), Q(1, 10**5)


def prepare_sine_prior_forecast(prior):
    """Declare information, clock, observable and budgets before propagation."""
    _admit_prior(prior)
    return {
        "schema": "tnfr.sine-prior-forecast-protocol.v1",
        "prior": prior,
        "visible_nodes": ["left", "right"],
        "visible_edges": [],
        "ports": ["left", "right"],
        "visible_form": [0, 0],
        "visible_phase": _project((Q(0), Q(1, 2))),
        "visible_capacity": [1, 2],
        "model": {
            "law": "normalized_sine_exchange",
            "epi_weight": _project(Q(1, 2)),
            "phase_weight": _project(Q(1, 2)),
            "storage_scale": 1,
            "support": "one_supplied_hidden_star_without_internal_visible_edges",
            "forcing": "absent",
            "events": "none",
            "capacity": "held",
        },
        "horizon": _project(_HORIZON),
        "time_step": _project(_STEP),
        "order": _ORDER,
        "max_endpoint_width": _project(_MAX_WIDTH),
        "min_prediction_control_gap": _project(_MIN_GAP),
        "observable": "left_phase_on_continuous_lift_from_declared_zero",
        "observable_index": 3,
        "witness_grid": _project(Q(1, 1 << 20)),
        "control": (
            "same_inferred_initial_form_phase_with_hidden_capacity_set_to_zero_at_t0; "
            "counterfactual_intervention_not_a_model_fitting_all_prior_accelerations"
        ),
        "numerics": (
            "shared_strict_Picard_and_Taylor_Metzler_dyadic128; "
            "seven_coordinates_including_held_capacity; no_adaptive_retry"
        ),
        "information_budget": (
            "visible_preparation_and_prior_derivative_intervals_only; "
            "no_hidden_source_state_or_later_observation_available_to_predictor"
        ),
        "scope": "one_conditional_software_prediction_not_physical_validation",
    }


def _admit_prior(prior):
    keys = {
        "schema",
        "source_id",
        "clock_id",
        "observation_time",
        "evidence_window",
        "forecast_start",
        "form_rate_bounds",
        "phase_rate_bounds",
        "form_acceleration_bounds",
        "phase_acceleration_bounds",
        "observation_error",
    }
    _require(isinstance(prior, dict) and set(prior) == keys, "unexpected prior fields")
    _require(prior["schema"] == "tnfr.sine-prior-evidence.v1", "unsupported prior")
    _require(
        _exact(prior["observation_time"]) == 0
        and _pair(prior["evidence_window"]) == (0, 0)
        and _exact(prior["forecast_start"]) == Q(1, 32),
        "unsupported prior clock window",
    )
    for name in ("source_id", "clock_id"):
        _require(isinstance(prior[name], str) and bool(prior[name].strip()), name)
    error = prior["observation_error"]
    _require(
        isinstance(error, dict)
        and set(error)
        == {"absolute_padding", "differentiation_truncation", "kind", "arithmetic"}
        and _exact(error["absolute_padding"]) == Q(1, 1 << 30)
        and _exact(error["differentiation_truncation"]) == 0
        and error["kind"]
        == "synthetic_instantaneous_derivatives_not_finite_difference_samples"
        and error["arithmetic"]
        == "outward_dyadic128_independent_forward_edge_chain_rule",
        "unsupported derivative observation budget",
    )
    for name in (
        "form_rate_bounds",
        "phase_rate_bounds",
        "form_acceleration_bounds",
        "phase_acceleration_bounds",
    ):
        _require(
            isinstance(prior[name], dict) and set(prior[name]) == {"left", "right"},
            name,
        )
        for raw in prior[name].values():
            I(*_pair(raw))


def _prior_inference(prior):
    graph = nx.Graph()
    graph.add_node("left", EPI=0, theta=0, nu_f=1)
    graph.add_node("right", EPI=0, theta=0.5, nu_f=2)
    times = dict(
        source_id=prior["source_id"],
        clock_id=prior["clock_id"],
        observation_time=_exact(prior["observation_time"]),
        evidence_window=_pair(prior["evidence_window"]),
    )

    def channel(name):
        return {node: _pair(raw) for node, raw in prior[name].items()}

    state = infer_relational_sine_hidden_state(
        graph,
        ports=("left", "right"),
        form_rate_bounds=channel("form_rate_bounds"),
        phase_rate_bounds=channel("phase_rate_bounds"),
        reference_model=RelationalExchangeModel(1, phase_domain="regular"),
        forecast_start=_exact(prior["forecast_start"]),
        **times,
    )
    return state.infer_capacity(
        form_acceleration_bounds=channel("form_acceleration_bounds"),
        phase_acceleration_bounds=channel("phase_acceleration_bounds"),
        **times,
    )


def predict_sine_prior_forecast(protocol):
    """Issue prediction and counterfactual before receiving any future response."""
    from ..physics.relational_sine_forecast import admit_sine_prior, forecast_sine_prior

    expected = prepare_sine_prior_forecast(protocol["prior"])
    _require(
        json_dumps(protocol, sort_keys=True) == json_dumps(expected, sort_keys=True),
        "unsupported or changed sine forecast protocol",
    )
    capacity = _prior_inference(protocol["prior"])
    admission = admit_sine_prior(capacity)
    result = {
        "schema": "tnfr.sine-prior-prediction.v1",
        "admission": admission.to_dict(),
        "prediction": None,
        "control": None,
        "prediction_bounds": None,
        "control_bounds": None,
        "separation_bounds": None,
        "passed": False,
        "scope": expected["scope"],
    }
    if not admission.admitted:
        return result
    arguments = dict(end_time=_HORIZON, time_step=_STEP, order=_ORDER)
    prediction = forecast_sine_prior(admission, **arguments)
    control = forecast_sine_prior(admission, freeze_hidden=True, **arguments)
    observed, comparison = prediction.endpoint[3], control.endpoint[3]
    separation = observed - comparison
    gates = {
        "prediction_horizon": prediction.admitted,
        "control_horizon": control.admitted,
        "prediction_width": max(row.width for row in prediction.endpoint) <= _MAX_WIDTH,
        "control_width": max(row.width for row in control.endpoint) <= _MAX_WIDTH,
        "reserved_observable_separation": separation.lo > _MIN_GAP,
    }
    result.update(
        prediction=prediction.to_dict(),
        control=control.to_dict(),
        prediction_bounds=_project(observed),
        control_bounds=_project(comparison),
        separation_bounds=_project(separation),
        gates=gates,
        passed=all(gates.values()),
    )
    return result


def _record_interval(raw):
    _require(isinstance(raw, dict) and set(raw) == {"lo", "hi"}, "expected interval")
    return I(_exact(raw["lo"]), _exact(raw["hi"]))


def assess_sine_prior_response(prediction, response):
    """Compare one reserved response with already issued bounds; never refit."""
    from ..physics.relational_sine_forecast import SineForecast

    _require(isinstance(response, SineForecast), "a sine forecast response is required")
    _require(
        response.observation_time == 0
        and response.end_time == _HORIZON
        and response.time_step == _STEP
        and response.order == _ORDER
        and response.neighbors == ((2,), (2,), (0, 1))
        and response.visible_capacity == (Q(1), Q(2))
        and response.model == RelationalExchangeModel(1, phase_domain="regular")
        and not response.freeze_hidden,
        "response differs from the frozen law, support, clock or numerical budget",
    )
    _require(
        prediction.get("schema") == "tnfr.sine-prior-prediction.v1",
        "unsupported prediction record",
    )
    _require(type(prediction.get("passed")) is bool, "prediction needs Boolean verdict")
    _require(
        prediction["passed"], "failed prediction cannot authorize response evaluation"
    )
    observed = response.endpoint[3]
    forecast = _record_interval(prediction["prediction_bounds"])
    control = _record_interval(prediction["control_bounds"])
    separation = observed - control
    gates = {
        "response_horizon": response.admitted
        and response.validated_end_time == _HORIZON,
        "response_width": max(value.width for value in response.endpoint) <= _MAX_WIDTH,
        "reserved_response_inside_issued_prediction": observed.subset_of(forecast),
        "reserved_response_separated_from_control": separation.lo > _MIN_GAP,
    }
    return {
        "schema": "tnfr.sine-prior-reserved-response.v1",
        "response": response.to_dict(),
        "reserved_observation": _project(observed),
        "response_control_separation": _project(separation),
        "gates": gates,
        "passed": all(gates.values()),
        "scope": "known_source_software_response_not_independent_physical_evidence",
    }
