"""Assess one declared common-observation current-discrimination box."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research._single_analytic_assessment import (  # noqa: E402
    _run_single_analytic_assessment,
)


def _assess():
    from tnfr.mathematics._rational_interval import I
    from tnfr.physics._organized_robust_current import _bound_organized_robust_current

    calibration = I(1 - Q(1, 10**8), 1 + Q(1, 10**8))
    return _bound_organized_robust_current(
        capacitance_ratio=calibration,
        conductance_ratio=calibration,
        josephson_ratio=calibration,
        clock_ratio=calibration,
        total_tracking=Q(1, 10**4),
        plateau_tracking=Q(1, 10**15),
        raw_readout_error=Q(1, 10**14),
        aperture=Q(1, 10**13),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_ROBUST_CURRENT.md",
        evaluator_path="benchmarks/organized_robust_current.py",
        schema_name="organized_robust_current",
        assess=_assess,
        admitted_verdict="conditional_current_discrimination_admitted",
        unavailable_verdict="specified_current_discrimination_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
