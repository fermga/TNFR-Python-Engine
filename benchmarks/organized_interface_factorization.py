"""Assess the one declared finite-interface factorization after source freeze.

The committed declaration fixes every source, observation and comparison.
This instrument runs one analytical assessment, without a trajectory, sweep
or frozen producer replay. Existing evidence prevents an unrecorded retry.
"""

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
    from tnfr.physics._organized_interface_factorization import (
        _assess_organized_interface_factorization,
    )

    return _assess_organized_interface_factorization()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh evidence directory"
    )
    arguments = parser.parse_args(argv)
    verdict = _run_single_analytic_assessment(
        repository=ROOT,
        output=arguments.output,
        declaration_path="theory/nodal/ORGANIZED_INTERFACE_FACTORIZATION.md",
        evaluator_path="benchmarks/organized_interface_factorization.py",
        schema_name="organized_interface_factorization",
        assess=_assess,
        admitted_verdict="finite_scalar_factorization_excluded",
        unavailable_verdict="specified_factorization_test_unavailable",
    )
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
