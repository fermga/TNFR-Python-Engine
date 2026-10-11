"""Evaluate one frozen record-only forecast on archived synthetic evidence."""

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
    from tnfr.physics._organized_observed_forecast import (
        _assess_organized_observed_forecast,
    )

    return _assess_organized_observed_forecast(ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_OBSERVED_FORECAST.md",
        evaluator_path="benchmarks/organized_observed_forecast.py",
        schema_name="organized_observed_forecast",
        assess=_assess,
        admitted_verdict="conditional_record_only_forecast_certified",
        unavailable_verdict="specified_record_only_forecast_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
