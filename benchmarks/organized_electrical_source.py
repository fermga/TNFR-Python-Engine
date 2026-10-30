"""Run the single declared source-inclusive electrical assessment after freeze."""

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
    from tnfr.physics._organized_electrical_source import (
        _assess_organized_electrical_source,
    )

    return _assess_organized_electrical_source()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_ELECTRICAL_SOURCE.md",
        evaluator_path="benchmarks/organized_electrical_source.py",
        schema_name="organized_electrical_source",
        assess=_assess,
        admitted_verdict="finite_collective_electrical_source_admitted",
        unavailable_verdict="specified_collective_electrical_source_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
