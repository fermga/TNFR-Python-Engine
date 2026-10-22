"""Export and replay one declared SDK study through the shared study runner.

The topology, uniform initial triad, seed and operator word are supplied inputs.
The report is finite execution evidence, not an autonomous formation mechanism
or a resumable checkpoint. Install the repository before running this example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tnfr.sdk import StudySpec, export_to_json, import_from_json, run_study


def main() -> None:
    """Write a recipe and the result of running its deserialized declaration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/study"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = StudySpec(
        nodes=6,
        topology="ring",
        seed=42,
        sequence="basic_activation",
        cycles=1,
        name="ring-study",
    )
    spec_path = args.output_dir / "study.json"
    report_path = args.output_dir / "report.json"
    export_to_json(spec.to_dict(), spec_path)
    restored = StudySpec.from_dict(import_from_json(spec_path))
    result = run_study(restored)
    export_to_json(result, report_path)

    print(f"Declaration: {spec_path}")
    print(f"Finite execution report: {report_path}")
    print("Replay this declaration with: tnfr network --spec <declaration-path>")


if __name__ == "__main__":
    main()
