"""Certify a useful binary64 REMESH class at ``alpha=1/2``.

On a two-node antisymmetric history ``(a, -a)``, binary64 division by two,
addition, and a symmetric hard clamp preserve oddness.  The class is therefore
REMESH-forward-invariant and has the sharp uniform relative-defect bound
``eta=135/124`` once the represented radius contains ``4*2**-1074``.

Composing this bound with a schedule disagreement gain ``q`` gives
``q_eff=q*(1+eta)``.  The example uses ``q=4/9``, for which
``q_eff=259/279<1`` and the normalized block margin is ``20/279``.  It also
shows the zero-margin boundary and two excluded generalizations: arbitrary
metric-centered rows and an unrestricted fixed lattice.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from typing import Any

from tnfr.physics import (
    certify_half_alpha_antisymmetric_hard_clip_remesh_class,
    observe_binary64_remesh_pair_relative_defect,
)


def _weighted_center(pair: tuple[float, float]) -> Fraction:
    left, right = (Fraction.from_float(value) for value in pair)
    return (left + 3 * right) / 4


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def run_protocol() -> dict[str, Any]:
    """Build the class, its schedule compositions, and scope falsifiers."""

    certificate = certify_half_alpha_antisymmetric_hard_clip_remesh_class(
        ("left", "right"),
        (1.0, 3.0),
        tau_local=2,
        tau_global=3,
        epi_bound=1.0,
    )
    strict_policy = certificate.certify_schedule_relative_defect_stability(
        Fraction(4, 9)
    )
    boundary_policy = certificate.certify_schedule_relative_defect_stability(
        Fraction(124, 259)
    )

    smallest = math.ulp(0.0)
    sharpness_witness = observe_binary64_remesh_pair_relative_defect(
        (-3 * smallest, 3 * smallest),
        (-2 * smallest, 2 * smallest),
        (-3 * smallest, 3 * smallest),
        alpha=0.5,
        epi_min=-4 * smallest,
        epi_max=4 * smallest,
        clip_mode="hard",
    )

    amplitudes = (
        float.fromhex("0x1.37b40d1c51f86p-29"),
        float.fromhex("0x1.2be8875fa6dd8p-29"),
        float.fromhex("-0x1.0994982458cc8p-28"),
    )
    centered_inputs = tuple((3 * value, -value) for value in amplitudes)
    centered_falsifier = observe_binary64_remesh_pair_relative_defect(
        *centered_inputs,
        alpha=0.5,
        epi_min=-1.0,
        epi_max=1.0,
        clip_mode="hard",
    )
    lattice_falsifier = observe_binary64_remesh_pair_relative_defect(
        (1.0, -1.0),
        (0.0, -0.0),
        (0.0, -0.0),
        alpha=0.5,
        epi_min=-1.0,
        epi_max=1.0,
        clip_mode="hard",
    )
    return {
        "certificate": certificate,
        "strict_policy": strict_policy,
        "boundary_policy": boundary_policy,
        "sharpness_witness": sharpness_witness,
        "centered_inputs": centered_inputs,
        "centered_falsifier": centered_falsifier,
        "lattice_falsifier": lattice_falsifier,
    }


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the exact class result and its explicit boundaries."""

    certificate = protocol["certificate"]
    strict = protocol["strict_policy"]
    boundary = protocol["boundary_policy"]
    sharp = protocol["sharpness_witness"]
    centered_inputs = protocol["centered_inputs"]
    centered = protocol["centered_falsifier"]
    lattice = protocol["lattice_falsifier"]
    return {
        "claim": (
            "the alpha=1/2 antisymmetric P2 hard-clip REMESH class has a "
            "sharp uniform binary64 relative-defect bound"
        ),
        "class": {
            "certificate_valid": (
                certificate
                .half_alpha_antisymmetric_hard_clip_class_certificate_certified
            ),
            "alpha": _fraction_text(certificate.alpha),
            "interval": [
                _fraction_text(certificate.epi_min),
                _fraction_text(certificate.epi_max),
            ],
            "metric": [
                _fraction_text(value)
                for value in certificate.exact_normalized_metric
            ],
            "uniform_eta": _fraction_text(
                certificate.exact_uniform_relative_defect_upper_bound
            ),
            "strict_q_threshold": _fraction_text(
                certificate.exact_strict_schedule_gain_threshold
            ),
            "global_bound_proof": {
                "large_norm_tail_bound": _fraction_text(
                    certificate.exact_tail_relative_defect_upper_bound
                ),
                "large_norm_tail_is_strict": (
                    certificate.exact_tail_relative_defect_upper_bound
                    < certificate.exact_uniform_relative_defect_upper_bound
                ),
                "finite_core_candidates": (
                    certificate.finite_core_candidate_count
                ),
                "finite_core_admissible": (
                    certificate.finite_core_admissible_count
                ),
                "finite_core_maximizer": list(
                    certificate.finite_core_maximizer_amplitudes
                ),
            },
            "antisymmetry_preserved": (
                certificate.binary64_antisymmetry_preserved_certified
            ),
            "remesh_forward_invariant": (
                certificate.remesh_class_forward_invariant_certified
            ),
            "repeated_binary64_runtime": (
                certificate.repeated_binary64_stability_certified
            ),
            "future_execution": (
                certificate.future_binary64_execution_certified
            ),
            "full_tnfr_stability": certificate.full_tnfr_stability_certified,
        },
        "strict_schedule_composition": {
            "q": _fraction_text(strict.schedule_energy_gain_upper_bound),
            "q_effective": _fraction_text(
                strict.exact_effective_head_energy_gain_upper_bound
            ),
            "normalized_block_margin": _fraction_text(
                strict.exact_uniform_normalized_block_margin_lower_bound
            ),
            "geometric_spatial_disagreement_convergence": (
                strict.geometric_spatial_disagreement_convergence_certified
            ),
        },
        "threshold_boundary": {
            "q": _fraction_text(boundary.schedule_energy_gain_upper_bound),
            "q_effective": _fraction_text(
                boundary.exact_effective_head_energy_gain_upper_bound
            ),
            "normalized_block_margin": _fraction_text(
                boundary.exact_uniform_normalized_block_margin_lower_bound
            ),
            "geometric_spatial_disagreement_convergence": (
                boundary.geometric_spatial_disagreement_convergence_certified
            ),
        },
        "sharp_subnormal_witness": {
            "current_in_smallest_subnormal_units": [-3, 3],
            "local_in_smallest_subnormal_units": [-2, 2],
            "global_in_smallest_subnormal_units": [-3, 3],
            "ideal_left_in_smallest_subnormal_units": "-11/4",
            "runtime_hex": [value.hex() for value in sharp.runtime_bounded_pair],
            "observed_eta": _fraction_text(
                sharp.exact_minimum_nonnegative_relative_defect_bound
            ),
            "attains_uniform_bound": (
                sharp.exact_minimum_nonnegative_relative_defect_bound
                == certificate.exact_uniform_relative_defect_upper_bound
            ),
        },
        "excluded_generalizations": {
            "general_metric_centered_input_centers": [
                _fraction_text(_weighted_center(pair))
                for pair in centered_inputs
            ],
            "general_metric_centered_output_center": _fraction_text(
                _weighted_center(centered.runtime_bounded_pair)
            ),
            "general_metric_centering_forward_invariant": False,
            "unit_lattice_runtime_output": [
                value.hex() for value in lattice.runtime_bounded_pair
            ],
            "unit_lattice_forward_invariant": False,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
