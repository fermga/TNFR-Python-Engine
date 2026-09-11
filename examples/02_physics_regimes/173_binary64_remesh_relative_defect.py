"""Expose the sharp binary64 boundary of the REMESH defect hypothesis.

A bounded, normal-valued pair at ``alpha=1/2`` needs the exact local budget
``eta=2**210-1/4``.  The witness shows why a bounded binary64 box alone does
not provide the small uniform relative defect needed by the robust schedule
theorem.  The positive boundary ``alpha=1`` is different: on the declared
hard-clipped history class, the production recurrence is numerically the
global delayed value and therefore has uniform ``eta=0``.

The class certificate concerns only the REMESH map on the declared class.  It
does not certify a schedule family, repeated runtime stability, or future
execution.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from typing import Any

from tnfr.physics import (
    certify_alpha_one_hard_clip_remesh_class,
    observe_binary64_remesh_pair_relative_defect,
)


def run_protocol() -> dict[str, Any]:
    """Construct the exact counterexample and the ``alpha=1`` class proof."""

    current = math.ldexp(1.0, -52)
    local_left = math.ldexp(1.0, -105)
    local_right = math.nextafter(local_left, math.inf)
    counterexample = observe_binary64_remesh_pair_relative_defect(
        (current, current),
        (local_left, local_right),
        (1.0, 1.0),
        alpha=0.5,
        epi_min=0.0,
        epi_max=1.0,
        clip_mode="hard",
    )
    alpha_one_class = certify_alpha_one_hard_clip_remesh_class(
        ("left", "right"),
        (1.0, 1.0),
        tau_local=2,
        tau_global=3,
        epi_min=-1.0,
        epi_max=1.0,
    )
    return {
        "counterexample": counterexample,
        "alpha_one_class": alpha_one_class,
    }


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return exact arithmetic and the explicit certification boundary."""

    counterexample = protocol["counterexample"]
    alpha_one_class = protocol["alpha_one_class"]
    exact_eta = Fraction(2**210) - Fraction(1, 4)
    return {
        "claim": (
            "a bounded binary64 box alone does not imply a useful uniform "
            "REMESH relative-defect budget"
        ),
        "normal_binary64_counterexample": {
            "observation_valid": (
                counterexample.pair_relative_defect_observation_certified
            ),
            "alpha": _fraction_text(counterexample.alpha),
            "coefficients": {
                "beta": _fraction_text(counterexample.beta),
                "gamma": _fraction_text(counterexample.gamma),
                "delta": _fraction_text(counterexample.delta),
            },
            "current_hex": [
                value.hex() for value in counterexample.binary64_current_pair
            ],
            "local_hex": [
                value.hex() for value in counterexample.binary64_local_pair
            ],
            "global_hex": [
                value.hex() for value in counterexample.binary64_global_pair
            ],
            "pairwise_jensen_denominator": _fraction_text(
                counterexample.exact_input_pairwise_jensen_denominator
            ),
            "ideal_squared_separation": _fraction_text(
                counterexample.exact_ideal_squared_separation
            ),
            "runtime_squared_separation": _fraction_text(
                counterexample.exact_bounded_squared_separation
            ),
            "minimum_eta": _fraction_text(
                counterexample.exact_minimum_nonnegative_relative_defect_bound
            ),
            "eta_identity": "2^210 - 1/4",
            "eta_identity_verified": (
                counterexample.exact_minimum_nonnegative_relative_defect_bound
                == exact_eta
            ),
            "uniform_bound_certified": (
                counterexample.uniform_binary64_relative_defect_bound_certified
            ),
            "future_bound_certified": (
                counterexample.future_binary64_relative_defect_bound_certified
            ),
        },
        "alpha_one_hard_clip_class": {
            "certificate_valid": (
                alpha_one_class.alpha_one_hard_clip_class_certificate_certified
            ),
            "alpha": _fraction_text(alpha_one_class.alpha),
            "uniform_eta": _fraction_text(
                alpha_one_class.exact_uniform_relative_defect_upper_bound
            ),
            "numeric_global_delay_copy": (
                alpha_one_class.binary64_global_delay_numeric_copy_certified
            ),
            "hard_clip_identity": (
                alpha_one_class.hard_clip_identity_on_class_certified
            ),
            "remesh_class_forward_invariant": (
                alpha_one_class.remesh_class_forward_invariant_certified
            ),
            "schedule_family_certified": (
                alpha_one_class.schedule_family_certificate_certified
            ),
            "repeated_binary64_stability": (
                alpha_one_class.repeated_binary64_stability_certified
            ),
            "future_binary64_execution": (
                alpha_one_class.future_binary64_execution_certified
            ),
            "solver_accuracy": alpha_one_class.solver_accuracy_certified,
            "full_tnfr_stability": (
                alpha_one_class.full_tnfr_stability_certified
            ),
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
