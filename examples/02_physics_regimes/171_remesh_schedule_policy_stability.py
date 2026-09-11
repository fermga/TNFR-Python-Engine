"""Certify uniform exact REMESH/schedule spatial-disagreement stability.

Every schedule in the declared model class is assumed to preserve consensus
and to have disagreement-energy gain at most ``q`` in one fixed positive
spatial metric.  The certificate proves a prefix bound of one and a block gain
of ``q`` over the conservative horizon ``active_max_delay + 1``.  It does not
inspect or execute a binary64 schedule.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from tnfr.physics import (
    certify_uniform_remesh_history_stability,
    certify_uniform_remesh_schedule_policy_stability,
)


def _certificate(*, alpha: Fraction, tau_global: int, q: Fraction):
    remesh = certify_uniform_remesh_history_stability(
        alpha=alpha,
        tau_local=2,
        tau_global=tau_global,
    )
    return certify_uniform_remesh_schedule_policy_stability(remesh, q)


def run_protocol() -> dict[str, Any]:
    """Build strict, pure-delay strict, and noncontractive boundary cases."""

    return {
        "mixed_delay_strict": _certificate(
            alpha=Fraction(1, 2),
            tau_global=4,
            q=Fraction(1, 4),
        ),
        "pure_delay_strict": _certificate(
            alpha=Fraction(1),
            tau_global=2,
            q=Fraction(1, 4),
        ),
        "pure_delay_boundary": _certificate(
            alpha=Fraction(1),
            tau_global=2,
            q=Fraction(1),
        ),
    }


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _row(certificate: Any, *, cycles: int) -> dict[str, Any]:
    return {
        "certificate_valid": certificate.policy_stability_certificate_certified,
        "q": _fraction_text(certificate.schedule_energy_gain_upper_bound),
        "universal_block_horizon": certificate.universal_block_horizon,
        "horizon_is_claimed_minimal": False,
        "uniform_margin": _fraction_text(
            certificate.exact_uniform_normalized_block_margin_lower_bound
        ),
        "prefix_gain_upper_bound": _fraction_text(
            certificate.exact_intrablock_prefix_energy_gain_upper_bound
        ),
        "cycle_count": cycles,
        "cycle_gain_upper_bound": _fraction_text(
            certificate.exact_cycle_energy_gain_upper_bound(cycles)
        ),
        "geometric_spatial_disagreement_convergence": (
            certificate.geometric_spatial_disagreement_convergence_certified
        ),
        "binary64_runtime_stability": (
            certificate.binary64_runtime_stability_certified
        ),
    }


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the theorem witnesses and their explicit scope."""

    mixed = protocol["mixed_delay_strict"]
    pure_strict = protocol["pure_delay_strict"]
    boundary = protocol["pure_delay_boundary"]
    return {
        "claim": (
            "conditional uniform exact REMESH/schedule spatial-disagreement "
            "stability"
        ),
        "mixed_delay_strict": _row(mixed, cycles=12),
        "pure_delay_strict": {
            **_row(pure_strict, cycles=7),
            "alpha_one_spatial_disagreement_decay": (
                pure_strict.alpha_one_spatial_disagreement_decay_certified
            ),
        },
        "q_one_boundary": {
            **_row(boundary, cycles=7),
            "zero_certified_margin_boundary": (
                boundary.q_one_zero_margin_boundary_certified
            ),
        },
        "scope": {
            "schedule_maps_verified": mixed.runtime_schedule_maps_verified,
            "binary64_runtime": mixed.binary64_runtime_stability_certified,
            "solver_accuracy": mixed.solver_accuracy_certified,
            "adaptive_grammar": mixed.adaptive_grammar_certified,
            "full_tnfr_stability": mixed.full_tnfr_stability_certified,
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
