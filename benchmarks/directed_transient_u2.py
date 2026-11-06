#!/usr/bin/env python3
"""Finite directed-diffusion controls and a weighted contraction counterexample.

Three supplied graphs contract in the restricted Euclidean comparisons. A
fourth, already retained in the tests, has an exact non-consensus direction
with positive initial energy derivative. It refutes universal Euclidean
contraction; stationary-weighted contraction is a separate theorem.

The selected model is fixed-graph linear EPI diffusion with scalar capacity
and structural time. Sampled semigroup, resolvent, variation and potential
magnitude comparisons do not certify an unobserved tail or canonical U2/U6.
See theory/TNFR_DIRECTED_NONNORMAL_DYNAMICS.md, sections 3-5. This corrected
instrument does not revise or regenerate a historical experiment record.
"""

from __future__ import annotations

import platform
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.directed_diffusion import directed_cayley_adjacency  # noqa: E402
from tnfr.physics.transient_u2 import certify_transient_u2  # noqa: E402
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ExperimentManifest,
    input_bit_length,
)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


CASES = [
    (
        "circulant C7{1,2} (normal)",
        directed_cayley_adjacency(7, {1, 2}),
        _unit([1, -1, 0.5, -0.5, 0.3, -0.2, -0.1]),
    ),
    (
        "weighted ring+chord (SC, non-normal)",
        np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]], dtype=float),
        _unit([1, -1, 0.5, -0.5]),
    ),
    (
        "star-in cycle-out (SC, non-normal)",
        np.array([[0, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float),
        _unit([1, -1, 0.5, -0.5]),
    ),
]

COUNTEREXAMPLE_WEIGHTS = (
    (0, 1, 0, 0, 0, 12),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 0, 15, 0, 0),
    (0, 9, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 10),
    (1, 0, 0, 0, 0, 0),
)
COUNTEREXAMPLE_STATE = (4341, -4028, -4275, -4065, 2273, 4998)


def counterexample_energy_derivative() -> Fraction:
    """Evaluate -2*y.T*L*y exactly on the retained pi-orthogonal witness."""
    stationary = tuple(Fraction(v, 57) for v in (13, 10, 10, 10, 1, 13))
    transition = tuple(
        tuple(Fraction(weight, sum(row)) for weight in row)
        for row in COUNTEREXAMPLE_WEIGHTS
    )
    if (
        any(
            sum(stationary[i] * transition[i][j] for i in range(6)) != stationary[j]
            for j in range(6)
        )
        or sum(p * y for p, y in zip(stationary, COUNTEREXAMPLE_STATE)) != 0
    ):
        raise ValueError("the fixed witness must lie in the non-consensus space")
    ly = tuple(
        y - sum(p * z for p, z in zip(row, COUNTEREXAMPLE_STATE))
        for y, row in zip(COUNTEREXAMPLE_STATE, transition)
    )
    return -2 * sum(y * v for y, v in zip(COUNTEREXAMPLE_STATE, ly))


def main() -> int:
    print(
        "Finite directed diffusion: three contracting controls and one counterexample"
    )
    header = (
        f"  {'graph':<38} {'||Q||':>6} {'symE':>6} {'peak':>6} "
        f"{'kreiss':>7} {'ambient':>8} {'J<=b':>5} {'noAmp':>6}"
    )
    print(header)
    controls_match = True
    all_bounds = True
    cases = [(*case, True) for case in CASES]
    cases.append(
        (
            "weighted counterexample",
            np.array(COUNTEREXAMPLE_WEIGHTS, dtype=float),
            _unit(COUNTEREXAMPLE_STATE),
            False,
        )
    )
    for label, w, x, expected_no_amp in cases:
        c = certify_transient_u2(w, x)
        controls_match &= c.no_transient_amplification == expected_no_amp
        all_bounds &= c.bounds_hold
        print(
            f"  {label:<38} {c.consensus_projection_norm:>6.4f} "
            f"{c.symmetric_part_min_eig:>6.3f} {c.peak_gain:>6.4f} "
            f"{c.kreiss_lower_bound:>7.4f} {c.ambient_oblique_gain:>8.4f} "
            f"{str(c.integrated_reorganization <= c.integrated_reorganization_bound):>5} "
            f"{str(c.no_transient_amplification):>6}"
        )

    derivative = counterexample_energy_derivative()
    controls_match &= derivative > 0
    audit = CircularityAudit()  # supplied spectral / semigroup model
    _ = ExperimentManifest(
        claim_id="NT-P09d",
        git_sha="local",
        versions={"python": platform.python_version(), "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(len(w) for _, w, _, _ in cases)),
        controls=(
            "normal_unit_gain",
            "three_contracting_fixtures",
            "weighted_nonconsensus_growth_counterexample",
            "sampled_resolvent_semigroup_comparison",
            "finite_window_variation_comparison",
            "potential_magnitude_not_u6_drift",
        ),
        artifacts=(),
    )
    print()
    print(f"  Declared contraction/growth controls match: {controls_match}")
    print(f"  Exact counterexample d||y||^2/ds at s=0: {derivative} > 0")
    print(f"  Sampled inequality comparisons hold: {all_bounds}")
    print("  Universal Euclidean contraction: REFUTED by the fixed witness")
    print("  Canonical U2 metric: OPEN; tail and U6 drift: NOT ASSESSED")
    print(f"  circularity           : {audit.verdict.value}")
    ok = controls_match and all_bounds
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
