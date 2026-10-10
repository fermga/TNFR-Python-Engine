"""One fixed, source-preserving receiver enclosure; no implicit execution.

The mathematical owner is ORGANIZED_RECEIVER_ENCLOSURE.md. The immutable
driver freezes this policy and records the first attempt before invoking the
runner. Importing this module or rebuilding admission evaluates no trajectory.
"""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction as Q

from ..mathematics._comparison_flow import scalar_affine_upper as _discrepancy_upper
from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._validated_metric import (
    _metric_error_upper,
    validated_metric_taylor_step,
)
from ..physics._organized_receiver_enclosure_geometry import (
    _OSL_BISECTIONS,
    _OSL_COSINE_BITS,
    _OSL_RATE_BOUNDS,
    _lift_receiver_reference,
    _observe_receiver_tube,
    _receiver_local_osl,
    _receiver_reference_flow,
)
from ..physics._organized_receiver_slow_admission import (
    _DEFECT_BUDGET,
    _LIPSCHITZ,
    _RELEASE_TIME,
    _assess_organized_receiver_slow_admission,
)

_H = Q(1, 8)
_HORIZON = Q(32)
_MAX_STEPS = int(_HORIZON / _H)
_ORDER = 8
_RHO = _DEFECT_BUDGET
_INFLATION = Q(5, 4)
_ADDITIVE_RADIUS = Q(1, 2**80)
_ERROR_CAP = Q(1, 2)
_METRIC = tuple(tuple(Q(2 * (i == j)) for j in range(13)) for i in range(13))


def receiver_enclosure_policy():
    """Return fresh exact primitive policy; none depends on a response."""
    return {
        "release_time": _RELEASE_TIME,
        "form_error_bound": Q(1, 10**10),
        "phase_error_bound": Q(1, 10**10),
        "elapsed_slow_horizon": _HORIZON,
        "step": _H,
        "maximum_steps": _MAX_STEPS,
        "taylor_order": _ORDER,
        "velocity_defect_budget": _RHO,
        "global_growth_bound": _LIPSCHITZ,
        "local_growth_bracket": _OSL_RATE_BOUNDS,
        "local_growth_bisections": _OSL_BISECTIONS,
        "cosine_rounding_bits": _OSL_COSINE_BITS,
        "prospective_inflation": _INFLATION,
        "prospective_additive_radius": _ADDITIVE_RADIUS,
        "maximum_phase_error": _ERROR_CAP,
        "maximum_attempts": 1,
        "adaptation_or_retry": False,
        "stop_priority": (
            "certified_donor_loss",
            "certified_joined_capture_with_donor_retention",
            "phase_error_budget_exhausted",
            "horizon_exhausted",
        ),
        "donor_corridor_failure": "record_unavailable_history_and_continue",
        "capture_without_donor_history": "record_and_continue",
        "validation_failure": "stop_unavailable_preserving_valid_prefix",
        "initial_donor_history": "proved_continuously_from_tau_200_to_400",
    }


def receiver_enclosure_source():
    """Rebuild every consumed release bound from the original primitives."""
    policy = receiver_enclosure_policy()
    return _assess_organized_receiver_slow_admission(
        form_error_bound=policy["form_error_bound"],
        phase_error_bound=policy["phase_error_bound"],
    )


def _advance_receiver_reference(center, radius, *, time, form_bound):
    """One exact local reference plus full-source error; no error resetting."""
    step, failed_tube, reason = validated_metric_taylor_step(
        center,
        Q(0),
        _METRIC,
        _H,
        _receiver_reference_flow,
        lambda tube: (Q(1),),
        growth_rate=_LIPSCHITZ,
        order=_ORDER,
        time=time,
    )
    if step is None:
        return {
            "accepted": False,
            "reason": reason,
            "failed_reference_tube": (
                None if failed_tube is None else tuple(asdict(x) for x in failed_tube)
            ),
        }
    reference_tube = _lift_receiver_reference(step.tube)
    global_envelope = _discrepancy_upper(radius, _LIPSCHITZ, _H, _RHO)
    prospective = I(_INFLATION * (global_envelope + _ADDITIVE_RADIUS)).hi
    osl = _receiver_local_osl(reference_tube, phase_radius=prospective)
    continuous_error = _discrepancy_upper(radius, osl.growth_rate_upper_bound, _H, _RHO)
    whole_error = max(radius, continuous_error)
    strict_margin = prospective - whole_error
    if not osl.certified or strict_margin <= 0:
        return {
            "accepted": False,
            "reason": "prospective_error_domain_not_closed",
            "reference_step": asdict(step),
            "osl": asdict(osl),
            "prospective_radius": prospective,
            "strict_error_margin": strict_margin,
        }
    endpoint_error = I(continuous_error + step.endpoint_radius).hi
    corridor = _observe_receiver_tube(
        reference_tube,
        phase_error_bound=whole_error,
        form_norm_upper_bound=form_bound,
    )
    endpoint = _observe_receiver_tube(
        _lift_receiver_reference(step.endpoint_center),
        phase_error_bound=endpoint_error,
        form_norm_upper_bound=form_bound,
    )
    return {
        "accepted": True,
        "reference_step": asdict(step),
        "osl": asdict(osl),
        "initial_phase_error": radius,
        "global_error_envelope": global_envelope,
        "prospective_radius": prospective,
        "strict_error_margin": strict_margin,
        "continuous_endpoint_error": continuous_error,
        "whole_step_phase_error": whole_error,
        "endpoint_phase_error": endpoint_error,
        "donor_corridor": asdict(corridor),
        "endpoint": asdict(endpoint),
    }


def run_receiver_enclosure(*, record_step):
    """Evaluate the fixed first attempt, retaining every step via the callback.

    The caller owns exclusivity and durable exports. No numerical adaptation
    occurs. A validation failure retains its attempted step; a lost sufficient
    donor corridor does not prevent a later certified endpoint loss.
    """
    source = receiver_enclosure_source()
    if source.status != "certified_uniform_slow_admission":
        return {
            "status": "unavailable",
            "reason": "source_admission_failed",
            "source": asdict(source),
            "accepted_steps": 0,
        }
    alpha = 2 * pi_interval() / 9
    prepared = tuple(alpha * (j - 4) for j in range(9)) + (I(0),) * 4
    center = tuple(value.midpoint for value in prepared)
    rounding = _metric_error_upper(_METRIC, tuple(value.radius for value in prepared))
    radius = I(source.release_phase_error_norm_upper_bound + rounding).hi
    form_bound = source.release_form_norm_upper_bound
    initial = _observe_receiver_tube(
        _lift_receiver_reference(center),
        phase_error_bound=radius,
        form_norm_upper_bound=form_bound,
    )
    donor_history = source.donor_release_bridge_certified
    records = 0
    first_unavailable_donor_interval = None
    final = asdict(initial)
    reason, status = "horizon_exhausted", "unavailable"
    for index in range(1, _MAX_STEPS + 1):
        try:
            result = _advance_receiver_reference(
                center, radius, time=(index - 1) * _H, form_bound=form_bound
            )
        except (ArithmeticError, ValueError) as error:
            result = {
                "accepted": False,
                "reason": "numerical_or_domain_validation_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "initial_center": center,
                "initial_phase_error": radius,
            }
        result["step_index"] = index
        if result["accepted"]:
            records = index
            final = result["endpoint"]
            donor_history = donor_history and all(
                value > 0
                for value in result["donor_corridor"]["donor_registered_branch_margins"]
            )
            if not donor_history and first_unavailable_donor_interval is None:
                first_unavailable_donor_interval = ((index - 1) * _H, index * _H)
            result["donor_history_certified"] = donor_history
        record_step(result)
        if not result["accepted"]:
            reason = result["reason"]
            break
        center = result["reference_step"]["endpoint_center"]
        radius = result["endpoint_phase_error"]
        if any(
            winding is not None and winding != 1 for winding in final["windings"][:2]
        ):
            status = reason = "certified_donor_loss"
            break
        if donor_history and final["capture_inequalities_pass"]:
            status = reason = "certified_joined_capture_with_donor_retention"
            break
        if radius > _ERROR_CAP:
            reason = "phase_error_budget_exhausted"
            break
    return {
        "status": status,
        "reason": reason,
        "source": asdict(source),
        "initial_observation": asdict(initial),
        "initial_rounding_error_bound": rounding,
        "accepted_steps": records,
        "elapsed_slow_time": records * _H,
        "donor_history_certified": donor_history,
        "first_unavailable_donor_interval": first_unavailable_donor_interval,
        "final_phase_error": radius,
        "final_observation": final,
        "physical_identification": "not_supplied_or_evaluated",
    }
