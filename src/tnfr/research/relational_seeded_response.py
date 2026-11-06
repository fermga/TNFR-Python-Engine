"""Frozen bounded responses for the regular source/receiver IVP.

Preparation contains no response computation. Evaluation uses a continuous
certificate of the declared ideal model; it is not a laboratory observation.
"""

from __future__ import annotations

from fractions import Fraction as Q

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._comparison_flow import _exact
from ..mathematics._rational_interval import I, pi_interval
from ..physics.relational_reflected_transit import certify_relational_reflected_transit
from ..sdk.relational_reports import _project
from ..utils.io import json_dumps

__all__ = ("prepare_relational_seeded_response", "evaluate_relational_seeded_response")

_HORIZON, _STEP, _ORDER = Q(1, 8), Q(1, 64), 6
_MAX_WIDTH = Q(1, 1 << 20)
_E, _W, _BETA = Q(1, 2), Q(1, 2), Q(1)


def prepare_relational_seeded_response(*, study="short", horizon=None):
    """Declare the reserved observation and budget without evolving the IVP."""
    if study not in ("short", "target-budget"):
        raise ValueError("unsupported seeded-response study")
    if study == "short" and horizon is not None:
        raise ValueError("short study has a fixed horizon")
    budget_horizon = Q(1) if horizon is None else _exact(horizon)
    if not 0 < budget_horizon <= 256 * _STEP:
        raise ValueError("budget horizon must be positive and admit at most 256 steps")
    edges = [(i, (i + 1) % 5) for i in range(5)]
    edges += [(i + 5, (i + 1) % 5 + 5) for i in range(5)]
    edges += [(0, 5), (1, 6)]
    declaration = {
        "schema": "tnfr.relational-seeded-short-protocol.v1",
        "nodes": list(range(10)),
        "edges": [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in edges)],
        "original_phase_pi": _project(
            tuple(Q(2 * i, 5) for i in range(5)) + (Q(6, 5),) * 5
        ),
        "form": [0] * 10,
        "capacity": [1] * 10,
        "coordinate_order": ["p", "r", "P", "R", "a", "b", "A", "B"],
        "initial_form_coordinates": [0] * 4,
        "initial_phase_coordinates_pi": _project((Q(4, 5), Q(2, 5), Q(0), Q(0))),
        "initial_interpretation": "mathematical_pi_IVP_enclosed_outward_not_projected_float_graph",
        "phase_frame": "subtract_common_6pi_over5_then_exact_individual_2pi_lifts",
        "model": {
            "epi_weight": _project(_E),
            "phase_weight": _project(_W),
            "storage_scale": _project(_BETA),
            "phase_domain": "regular",
            "forcing": 0,
            "events": [],
        },
        "clock": "continuous_structural_time",
        "horizon": _project(_HORIZON),
        "time_step": _project(_STEP),
        "order": _ORDER,
        "interval_bits": 128,
        "max_endpoint_width": _project(_MAX_WIDTH),
        "picard_policy": {
            "iterations": 16,
            "inflation": _project(Q(5, 4)),
            "padding": _project(Q(1, 1 << 90)),
        },
        "comparison_policy": "shared_Metzler32_dyadic128_norm_tail",
        "prediction": [
            "whole_horizon_regular_and_within_endpoint_width_budget",
            "receiver_port_phase_A_positive_at_endpoint",
            "source_port_argument_below_its_initial_negative_value",
            "strictly_positive_accumulated_loss",
            "positive_remaining_two_acute_twist_storage_budget",
            "source_winding_one_and_receiver_winding_zero_throughout",
        ],
        "scope": "conditional_ideal_ODE_short_response_not_formation_capture_or_physical_validation",
        "failure_policy": "retain_first_unresolved_tube_and_original_verdict_no_retries",
    }
    if study == "target-budget":
        declaration.update(
            schema="tnfr.relational-seeded-budget-protocol.v1",
            horizon=_project(budget_horizon),
            continuation="same_exact_initial_IVP_restarted_at_zero_not_a_new_preparation",
            prediction=[
                "negative_budget_upper_bound_excludes_later_two_acute_twist_target",
                "positive_budget_lower_bound_retains_only_a_necessary_resource",
                "zero_containing_budget_is_undecided",
                "unresolved_tube_retains_partial_evidence_not_a_singularity_claim",
            ],
            budget_discriminant="E_at_declared_horizon_minus_10beta_times_1_minus_cos_2pi_over5",
            target="both_original_C5_rings_strictly_acute_with_ordinary_winding_plus_one",
            exclusion_scope="later_regular_continuation_same_unforced_fixed_support_loss_law",
            passed_means="full_horizon_and_width_admitted_with_strict_budget_sign_not_formation",
            scope="conditional_fixed_IVP_budget_test_not_capture_or_physical_validation",
        )
    return declaration


def evaluate_relational_seeded_response(protocol, *, study="short", horizon=None):
    """Evaluate exactly the declaration; a changed protocol rejects before flow."""
    expected = prepare_relational_seeded_response(study=study, horizon=horizon)
    if json_dumps(protocol, sort_keys=True, allow_nan=False) != json_dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("unsupported or altered seeded-response protocol")
    pi = pi_interval()
    initial = (I(0),) * 4 + (4 * pi / 5, 2 * pi / 5, I(0), I(0))
    report = certify_relational_reflected_transit(
        initial,
        model=RelationalExchangeModel(
            _BETA, epi_weight=_E, phase_weight=_W, phase_domain="regular"
        ),
        horizon=Q(**expected["horizon"]),
        time_step=_STEP,
        order=_ORDER,
    )
    width = max(value.width for value in report.endpoint)
    change = report.endpoint_source_argument - report.initial_source_argument
    if study == "target-budget":
        return _budget_response(report, width, change, expected)
    gates = {
        "horizon_admitted": report.admitted,
        "endpoint_width_budget": width <= _MAX_WIDTH,
        "receiver_phase_advanced": report.endpoint[6].lo > 0,
        "source_argument_approached_branch": change.hi < 0,
        "positive_accumulated_loss": report.cumulative_loss.lo > 0,
        "remaining_target_budget_positive": report.target_budget.lo > 0,
        "source_winding_one_throughout": report.source_winding_one_throughout,
        "receiver_winding_zero_throughout": report.receiver_winding_zero_throughout,
    }
    return {
        "schema": "tnfr.relational-seeded-short-response.v1",
        "certificate": report.to_dict(),
        "source_argument_change": _project(change),
        "max_endpoint_width": _project(width),
        "gates": gates,
        "passed": all(gates.values()),
        "scope": expected["scope"],
    }


def _budget_response(report, width, change, declaration):
    """Separate numerical admission from the resource sign and target claim."""
    budget = report.target_budget
    sign = "negative" if budget.hi < 0 else "positive" if budget.lo > 0 else "undecided"
    negative_times = [
        observation.time
        for observation in report.observations
        if observation.target_budget.hi < 0
    ]
    if sign == "negative":
        negative_times.append(report.validated_horizon)
    first_exclusion = min(negative_times, default=None)
    gates = {
        "horizon_admitted": report.admitted,
        "endpoint_width_budget": width <= _MAX_WIDTH,
        "strict_budget_sign": sign != "undecided",
    }
    if not report.admitted:
        verdict = "unavailable_horizon"
    elif not gates["endpoint_width_budget"]:
        verdict = "unavailable_width"
    else:
        verdict = {
            "negative": "target_excluded",
            "positive": "target_not_excluded_by_storage",
            "undecided": "undecided",
        }[sign]
    return {
        "schema": "tnfr.relational-seeded-budget-response.v1",
        "certificate": report.to_dict(),
        "source_argument_change": _project(change),
        "max_endpoint_width": _project(width),
        "budget_sign_at_validated_horizon": sign,
        "target_budget_verdict": verdict,
        "first_certified_target_exclusion_time": _project(first_exclusion),
        "future_target_excluded_from_validated_horizon": first_exclusion is not None,
        "gates": gates,
        "passed": all(gates.values()),
        "passed_means": declaration["passed_means"],
        "scope": declaration["scope"],
    }
