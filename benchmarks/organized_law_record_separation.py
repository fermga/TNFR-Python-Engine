"""Assess the declared nonlinear record certificate from frozen scalar moments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research._single_analytic_assessment import (  # noqa: E402
    _run_single_analytic_assessment,
)


@dataclass(frozen=True)
class _RecordSeparationAssessment:
    kernel_source_receipts: tuple[dict, ...]
    certificate: Any
    admitted: bool
    scope: str = "conditional_two_law_record_separation_not_physical_law_selection"


def _assess():
    from tnfr.physics._organized_law_record_separation import (
        _certify_organized_law_record_separation,
    )
    from tnfr.research._organized_law_record_inputs import (
        _read_organized_law_record_moments,
    )

    moments, receipts = _read_organized_law_record_moments(ROOT)
    certificate = _certify_organized_law_record_separation(moments=moments)
    return _RecordSeparationAssessment(
        receipts, certificate, certificate.separation_certified
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_LAW_OBSERVATION.md",
        evaluator_path="benchmarks/organized_law_record_separation.py",
        schema_name="organized_law_record_separation",
        assess=_assess,
        admitted_verdict="conditional_two_law_record_separation_certified",
        unavailable_verdict="conditional_two_law_record_separation_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
