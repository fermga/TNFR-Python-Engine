"""Inspect a supported frozen source; optionally prepare a new pinned worktree."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research.frozen_source import inspect_frozen_source, restore_frozen_source
from tnfr.utils.io import json_dumps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", required=True, help="repository-relative freeze JSON"
    )
    parser.add_argument(
        "--destination", type=Path, help="new detached worktree location"
    )
    args = parser.parse_args(argv)
    if args.destination is None:
        report = inspect_frozen_source(ROOT, args.receipt)
    else:
        report = restore_frozen_source(ROOT, args.receipt, args.destination)
    print(
        json_dumps(
            {
                "operation": "inspect" if args.destination is None else "restore",
                "source": asdict(report),
                "scope": "Byte/base association; no archived code, runtime check or evaluation executed.",
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
