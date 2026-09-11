"""Compose the global P2 half-Reception kernel with alpha-one REMESH.

The restricted numeric EPI model fixes two mutual singleton neighbors,
``EN_mix=0.5``, one immutable all-target snapshot and a common hard clamp.
Every finite represented input pair in the source interval maps to numeric
consensus, so its global kernel gain is ``q=0``.  Alpha-one REMESH contributes
``eta=0``; the active
post-schedule history therefore has zero spatial disagreement after
``tau_global + 1`` cycles.

This example certifies the repeated numeric kernels.  It does not execute or
certify the complete Reception stage, grammar, events, graph transactions or
solver behavior.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from typing import Any

from tnfr.physics import (
    certify_alpha_one_hard_clip_remesh_class,
    certify_p2_half_reception_remesh_stability,
)


def run_protocol() -> dict[str, Any]:
    """Build the source class, compose it, and replay boundary pairs."""

    source = certify_alpha_one_hard_clip_remesh_class(
        ("left", "right"),
        (1.0, 3.0),
        tau_local=5,
        tau_global=2,
        epi_min=-1.0,
        epi_max=1.0,
    )
    certificate = certify_p2_half_reception_remesh_stability(source)
    tiny = math.ulp(0.0)
    return {
        "certificate": certificate,
        "ordinary_output": certificate.evaluate_binary64_schedule_pair(
            (-1.0, 0.5)
        ),
        "signed_zero_boundary": certificate.evaluate_binary64_schedule_pair(
            (-tiny, -0.0)
        ),
    }


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the exact composition and its explicit execution boundary."""

    certificate = protocol["certificate"]
    ordinary = protocol["ordinary_output"]
    signed_zero = protocol["signed_zero_boundary"]
    horizon = certificate.active_history_extinction_horizon
    return {
        "claim": (
            "global q=0 for the restricted binary64 P2 half-Reception EPI "
            "kernel composed with alpha-one eta=0 REMESH"
        ),
        "certificate_valid": (
            certificate.p2_half_reception_remesh_stability_certificate_certified
        ),
        "node_order": list(certificate.node_order),
        "normalized_metric": [
            _fraction_text(value)
            for value in certificate.exact_normalized_metric
        ],
        "mix_factor_hex": certificate.binary64_mix_factor.hex(),
        "q": _fraction_text(
            certificate.exact_schedule_energy_gain_upper_bound
        ),
        "eta": _fraction_text(
            certificate.exact_pre_schedule_relative_energy_defect_upper_bound
        ),
        "q_eff": _fraction_text(
            certificate.exact_effective_head_energy_gain_upper_bound
        ),
        "extinction_horizon": horizon,
        "gain_before_horizon": _fraction_text(
            certificate.exact_cycle_energy_gain_upper_bound(horizon - 1)
        ),
        "gain_at_horizon": _fraction_text(
            certificate.exact_cycle_energy_gain_upper_bound(horizon)
        ),
        "ordinary_output": list(ordinary),
        "signed_zero_boundary": {
            "numeric_consensus": signed_zero[0] == signed_zero[1],
            "output_hex": [value.hex() for value in signed_zero],
            "bit_preservation_certified": (
                certificate.signed_zero_bit_preservation_certified
            ),
        },
        "scope": {
            "global_binary64_epi_kernel_family": (
                certificate.global_binary64_epi_kernel_family_certified
            ),
            "arbitrary_finite_kernel_repetition": (
                certificate.arbitrary_finite_binary64_kernel_repetition_certified
            ),
            "active_history_exact_extinction": (
                certificate.active_history_exact_extinction_certified
            ),
            "complete_reception_stage": (
                certificate.complete_reception_stage_certified
            ),
            "grammar_execution": certificate.grammar_execution_certified,
            "live_graph_execution": certificate.live_graph_execution_certified,
            "solver_accuracy": certificate.solver_accuracy_certified,
            "full_tnfr_stability": certificate.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
