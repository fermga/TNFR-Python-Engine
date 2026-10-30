"""Assess the declared alternative-law outer source once, without a trajectory."""

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
    from tnfr.physics._organized_alternative_outer_acquisition import (
        _assess_organized_alternative_outer_acquisition,
    )

    return _assess_organized_alternative_outer_acquisition(
        form_error_bounds=(Q(1, 10**10),) * 5,
        phase_error_bounds=(Q(1, 10**10),) * 5,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md",
        evaluator_path="benchmarks/organized_alternative_outer_acquisition.py",
        schema_name="organized_alternative_outer_acquisition",
        assess=_assess,
        admitted_verdict="alternative_outer_acquisition_certified",
        unavailable_verdict="alternative_outer_acquisition_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
