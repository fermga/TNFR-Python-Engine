"""Run one declared source-inclusive finite-input observation assessment."""

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
    from tnfr.physics._organized_finite_observation_source import (
        _assess_organized_finite_observation,
    )

    return _assess_organized_finite_observation()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_FINITE_OBSERVATION.md",
        evaluator_path="benchmarks/organized_finite_observation.py",
        schema_name="organized_finite_observation",
        assess=_assess,
        admitted_verdict="finite_input_observation_admitted",
        unavailable_verdict="specified_finite_input_observation_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
