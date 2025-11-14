#!/usr/bin/env python3
"""Remove generated artifacts (results, caches, benchmark outputs) from the repo tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

GENERATED_PATHS: list[Path] = [
    Path("results"),
    Path("outputs"),
    Path("benchmarks/results"),
    Path("validation_outputs"),
    Path("artifacts"),
    Path("profiles"),
    Path("examples/output"),
    Path("dist-test"),
    Path("site"),
]

CACHE_PATTERNS: list[str] = [
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
]


def _remove_path(path: Path, *, dry_run: bool = False) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        if not dry_run:
            shutil.rmtree(path)
    else:
        if not dry_run:
            path.unlink()
    return True


def _remove_glob(patterns: Iterable[str], *, dry_run: bool = False) -> int:
    removed = 0
    for pattern in patterns:
        for entry in Path(".").glob(pattern):
            try:
                if not dry_run:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                removed += 1
            except FileNotFoundError:
                continue
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    removed_paths = [
        str(p) for p in GENERATED_PATHS if _remove_path(p, dry_run=args.dry_run)
    ]
    removed_cache = _remove_glob(CACHE_PATTERNS, dry_run=args.dry_run)

    if removed_paths:
        label = "Would remove" if args.dry_run else "Removed"
        print(f"{label} directories/files:")
        for item in removed_paths:
            print(f"  - {item}")
    else:
        print("No generated directories to remove.")

    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {removed_cache} cache entries matching {len(CACHE_PATTERNS)} patterns.")


if __name__ == "__main__":
    main()
