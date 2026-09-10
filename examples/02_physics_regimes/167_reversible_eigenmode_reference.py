"""Certify both exact nonuniform modes of the nonregular three-node path.

The homogeneous-capacity random-walk generator on P3 is reversible in the
degree metric ``H = diag(1, 2, 1)``.  Its two nonuniform exact modes have
eigenvalues one and two.  This pure reference compares pressure-refreshed
Euler partitions with rational enclosures of their continuous modal solutions.

The convergence property is a conditional exact-real theorem for admissible
partition families with ``h_max -> 0``.  It is not an observed infinite family
and does not certify binary64 execution, glyphs, REMESH or mixed-mode data.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from tnfr.physics import (
    ReversibleSingleEigenmodeEulerReferenceCertificate,
    certify_reversible_single_eigenmode_euler_reference,
)


F = Fraction
P3_CONDUCTANCE = (
    (F(0), F(1), F(0)),
    (F(1), F(0), F(1)),
    (F(0), F(1), F(0)),
)
PARTITIONS = (
    (F(1, 4), F(1, 4)),
    (F(1, 8),) * 4,
    (F(1, 16),) * 8,
)
MODE_INPUTS = (
    ("antisymmetric", (F(1), F(0), F(-1))),
    ("alternating", (F(1), F(-1), F(1))),
)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def run_protocol() -> tuple[
    tuple[str, ReversibleSingleEigenmodeEulerReferenceCertificate],
    ...,
]:
    """Build the two exact modal certificates on one fixed P3 problem."""

    return tuple(
        (
            name,
            certify_reversible_single_eigenmode_euler_reference(
                P3_CONDUCTANCE,
                nu_f=(F(1), F(1), F(1)),
                initial_epi=initial_epi,
                partitions=PARTITIONS,
            ),
        )
        for name, initial_epi in MODE_INPUTS
    )


def build_report(
    protocol: tuple[
        tuple[str, ReversibleSingleEigenmodeEulerReferenceCertificate],
        ...,
    ],
) -> dict[str, Any]:
    """Return a finite JSON-compatible report with explicit scope boundaries."""

    return {
        "claim": "exact reversible single-eigenmode Euler references on P3",
        "graph": {
            "node_count": 3,
            "regular": False,
            "degrees": ["1/1", "2/1", "1/1"],
        },
        "modes": [
            {
                "name": name,
                "reference_certificate_certified": (
                    certificate.reference_certificate_certified
                ),
                "initial_epi": [
                    _fraction_text(value)
                    for value in certificate.exact_initial_epi
                ],
                "reversible_metric": [
                    _fraction_text(value)
                    for value in certificate.exact_reversible_metric
                ],
                "mu": _fraction_text(certificate.exact_mode_eigenvalue),
                "continuous_factor_interval": [
                    float(certificate.exact_continuous_factor_lower_bound),
                    float(certificate.exact_continuous_factor_upper_bound),
                ],
                "euler_factors": [
                    _fraction_text(value)
                    for value in certificate.exact_euler_factors
                ],
                "linf_quadratic_error_bounds": [
                    _fraction_text(value)
                    for value in (
                        certificate.exact_linf_quadratic_error_upper_bounds
                    )
                ],
                "h_energy_quadratic_error_bounds": [
                    _fraction_text(value)
                    for value in (
                        certificate.exact_h_energy_quadratic_error_upper_bounds
                    )
                ],
                "strict_subdivision_improvement": (
                    certificate.strict_proper_subdivision_improvement_certified
                ),
                "scope": {
                    "conditional_exact_real_partition_convergence": (
                        certificate
                        .conditional_exact_real_partition_convergence_certified
                    ),
                    "binary64_asymptotic_convergence": (
                        certificate.binary64_asymptotic_convergence_certified
                    ),
                    "arbitrary_or_mixed_mode_initial_data": (
                        certificate
                        .arbitrary_or_mixed_mode_initial_data_certified
                    ),
                    "directed_or_nonreversible_generator": (
                        certificate
                        .directed_or_nonreversible_generator_certified
                    ),
                    "changing_generator_or_metric": (
                        certificate.changing_generator_or_metric_certified
                    ),
                    "glyph_or_remesh_dynamics": (
                        certificate.glyph_or_remesh_dynamics_certified
                    ),
                    "solver_order": certificate.solver_order_certified,
                    "full_tnfr_stability": (
                        certificate.full_tnfr_stability_certified
                    ),
                },
            }
            for name, certificate in protocol
        ],
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
