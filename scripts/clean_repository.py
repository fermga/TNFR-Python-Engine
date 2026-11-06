#!/usr/bin/env python3
"""Remove declared generated artifacts and, optionally, regenerable caches.

The default removes the declared documentation and package outputs.
``--caches`` adds tool caches and ``__pycache__`` directories below maintained
source roots. ``--dry-run`` lists the targets and ``--report`` summarizes local
disk usage; neither removes anything. Local evidence, frozen ``build/``
preparations, Python environments and user-supplied books are never targets.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    # build/ also retains frozen research programs and local preparations.
    # Only these declared documentation/package outputs are disposable.
    "build/docs-source",
    "build/lib",
    "dist",
    "site",
    "htmlcov",
    "examples/output",
    "results",
)
CACHE_TARGETS = (".mypy_cache", ".pytest_cache", ".ruff_cache")
# Bytecode caches are collected only below these maintained source roots, never
# inside local evidence, frozen preparations or Python environments.
BYTECODE_ROOTS = (
    "src",
    "tests",
    "scripts",
    "benchmarks",
    "examples",
    "applications",
    "docs",
    "theory",
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


def _is_link(path: Path | os.DirEntry) -> bool:
    if isinstance(path, os.DirEntry):
        return path.is_symlink() or bool(getattr(path, "is_junction", bool)())
    isjunction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(isjunction and isjunction(path))


def _bytecode_directories(root: Path) -> list[str]:
    found = ["__pycache__"] if (root / "__pycache__").is_dir() else []
    for relative in BYTECODE_ROOTS:
        for directory, dirnames, filenames in os.walk(root / relative):
            if "pyvenv.cfg" in filenames:
                dirnames[:] = []
                continue
            current = Path(directory)
            if "__pycache__" in dirnames:
                found.append((current / "__pycache__").relative_to(root).as_posix())
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {"__pycache__", ".git", "node_modules"}
                and not _is_link(current / name)
            ]
    return found


def _targets(root: Path, *, caches: bool) -> list[str]:
    names = list(TARGETS)
    names.extend(
        path.relative_to(root).as_posix()
        for parent in (root, root / "src")
        for path in parent.glob("*.egg-info")
    )
    if caches:
        names.extend(CACHE_TARGETS)
        names.extend(_bytecode_directories(root))
    return names


def _scan(entry: Path) -> tuple[int, int, dict[Path, list[int]]]:
    """Return bytes, files and nested Python environments, without following links."""
    if not entry.is_dir():
        return entry.lstat().st_size, 1, {}
    size = files = 0
    environments: dict[Path, list[int]] = {}
    pending: list[tuple[Path, Path | None]] = [(entry, None)]
    while pending:
        directory, owner = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = list(iterator)
        except OSError:
            continue
        if owner is None and any(child.name == "pyvenv.cfg" for child in children):
            owner = directory
            environments[owner] = [0, 0]
        for child in children:
            try:
                if _is_link(child):
                    continue
                if child.is_dir(follow_symlinks=False):
                    pending.append((Path(child.path), owner))
                    continue
                child_size = child.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            size += child_size
            files += 1
            if owner is not None:
                environments[owner][0] += child_size
                environments[owner][1] += 1
    return size, files, environments


def _usage(root: Path, names: list[str]) -> tuple[int, int]:
    size = files = 0
    for name in names:
        path = root / name
        if path.is_dir() and not _is_link(path):
            path_size, path_files, _ = _scan(path)
            size, files = size + path_size, files + path_files
    return size, files


def report(root: Path) -> None:
    """Print local disk usage, environments and removable totals; remove nothing."""
    rows, environments = [], []
    for entry in root.iterdir():
        if _is_link(entry):
            continue
        size, files, found = _scan(entry)
        rows.append((size, files, entry.name))
        environments.extend(
            (env_size, env_files, env.relative_to(root).as_posix())
            for env, (env_size, env_files) in found.items()
        )
    print(f"Local disk usage of {root} (nothing is removed)")
    print(f"{'MB':>10} {'files':>9}  entry")
    for size, files, name in sorted(rows, reverse=True):
        print(f"{size / 1e6:10.1f} {files:9}  {name}")
    total = sum(row[0] for row in rows), sum(row[1] for row in rows)
    print(f"{total[0] / 1e6:10.1f} {total[1]:9}  total")
    if environments:
        print("\nPython environments (recreate them rather than archive):")
        for size, files, name in sorted(environments, reverse=True):
            print(f"{size / 1e6:10.1f} {files:9}  {name}")
    declared = _usage(root, _targets(root, caches=False))
    every = _usage(root, _targets(root, caches=True))
    print(
        f"\nRemovable by `python scripts/clean_repository.py`: "
        f"{declared[0] / 1e6:.1f} MB in {declared[1]} files"
    )
    print(
        f"Additionally removable with `--caches`: {(every[0] - declared[0]) / 1e6:.1f} MB "
        f"in {every[1] - declared[1]} files"
    )
    print(
        "Local evidence, frozen build/ preparations, environments and books are "
        "never removed by this script; review them explicitly."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--caches",
        action="store_true",
        help="also remove tool caches and source __pycache__ directories",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list targets and sizes; remove nothing"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="summarize local disk usage and Python environments; remove nothing",
    )
    args = parser.parse_args([] if argv is None else argv)
    root = ROOT.resolve()
    if args.report:
        report(root)
        return 0
    names = _targets(root, caches=args.caches)
    # Validate the entire operation before removing any of the declared outputs.
    targets = [(name, generated_directory(root, name)) for name in names]
    bytecode = [target for _, target in targets if target.name == "__pycache__"]
    for name, target in targets:
        if not target.exists() or target.name == "__pycache__":
            continue
        if args.dry_run:
            size, files = _usage(root, [name])
            print(f"would remove {name} ({size / 1e6:.1f} MB, {files} files)")
        else:
            shutil.rmtree(target)
            print(f"removed {target.relative_to(root)}")
    present = [target for target in bytecode if target.exists()]
    if present:
        if args.dry_run:
            relatives = [target.relative_to(root).as_posix() for target in present]
            size, files = _usage(root, relatives)
            print(
                f"would remove {len(present)} __pycache__ directories "
                f"({size / 1e6:.1f} MB, {files} files)"
            )
        else:
            for target in present:
                shutil.rmtree(target)
            print(f"removed {len(present)} __pycache__ directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
