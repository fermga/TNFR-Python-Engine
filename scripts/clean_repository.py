#!/usr/bin/env python3
"""Remove only declared generated artifacts inside the repository."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "build",
    "dist",
    "site",
    "htmlcov",
    "examples/output",
    "results",
)


def main() -> int:
    for relative in TARGETS:
        target = (ROOT / relative).resolve()
        if ROOT not in target.parents:
            raise RuntimeError(f"refusing to clean outside repository: {target}")
        if target.exists():
            shutil.rmtree(target)
            print(f"removed {target.relative_to(ROOT)}")
    for target in ROOT.glob("*.egg-info"):
        if target.is_dir():
            shutil.rmtree(target)
            print(f"removed {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
