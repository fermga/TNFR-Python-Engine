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


def generated_directory(root: Path, relative: str) -> Path:
    """Resolve a declared output without following redirects to another tree."""
    root = root.resolve()
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise RuntimeError(f"unsafe generated directory: {relative}")
    target = root / part
    resolved = target.resolve()
    if root not in resolved.parents or resolved != target:
        raise RuntimeError(f"refusing redirected generated directory: {target}")
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"generated directory is not a directory: {target}")
    return resolved


def main() -> int:
    root = ROOT.resolve()
    names = list(TARGETS)
    names.extend(
        path.relative_to(root).as_posix()
        for parent in (root, root / "src")
        for path in parent.glob("*.egg-info")
    )
    # Validate the entire operation before removing any of the declared outputs.
    targets = [generated_directory(root, name) for name in names]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"removed {target.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
