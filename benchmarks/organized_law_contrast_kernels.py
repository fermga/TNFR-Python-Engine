"""Evaluate the one declared scalar-moment assessment without any raw record."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research._single_analytic_assessment import (  # noqa: E402
    _run_single_analytic_assessment,
)


def _assess():
    from tnfr.physics._organized_law_contrast_kernels import (
        _assess_organized_law_contrast_kernels,
    )

    return _assess_organized_law_contrast_kernels()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_LAW_OBSERVATION.md",
        evaluator_path="benchmarks/organized_law_contrast_kernels.py",
        schema_name="organized_law_contrast_kernels",
        assess=_assess,
        admitted_verdict="fixed_contrast_kernel_arithmetic_admitted",
        unavailable_verdict="fixed_contrast_kernel_arithmetic_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
