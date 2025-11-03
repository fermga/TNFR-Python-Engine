#!/usr/bin/env python3
"""Automated stub file (.pyi) generation for TNFR modules.

This script uses mypy's stubgen to generate type stubs for Python modules
that are missing them, helping maintain type annotation consistency and
preventing drift between implementation and stubs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_missing_stubs(src_dir: Path) -> list[Path]:
    """Find all .py files that don't have corresponding .pyi stubs.

    Parameters
    ----------
    src_dir : Path
        The source directory to scan for Python files.

    Returns
    -------
    list[Path]
        List of Python files without corresponding stub files.
    """
    missing = []

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file) or py_file.name.startswith("_"):
            continue

        pyi_file = py_file.with_suffix(".pyi")
        if not pyi_file.exists():
            missing.append(py_file)

    return missing


def find_outdated_stubs(src_dir: Path, tolerance_seconds: float = 1.0) -> list[Path]:
    """Find .py files with stubs that are older than the implementation.

    Parameters
    ----------
    src_dir : Path
        The source directory to scan for Python files.
    tolerance_seconds : float, optional
        Time difference tolerance in seconds to avoid false positives from
        filesystem precision differences or clock skew. Default is 1.0 second.

    Returns
    -------
    list[Path]
        List of Python files whose stub files are outdated.
    """
    outdated = []

    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file) or py_file.name.startswith("_"):
            continue

        pyi_file = py_file.with_suffix(".pyi")
        if pyi_file.exists():
            # Compare modification times with tolerance
            py_mtime = py_file.stat().st_mtime
            pyi_mtime = pyi_file.stat().st_mtime
            # Only consider outdated if difference exceeds tolerance
            if py_mtime - pyi_mtime > tolerance_seconds:
                outdated.append(py_file)

    return outdated

def generate_stubs(files: list[Path], src_dir: Path, dry_run: bool = False) -> int:
    """Generate stub files using mypy stubgen.

    Parameters
    ----------
    files : list[Path]
        List of Python files to generate stubs for.
    src_dir : Path
        The source directory containing the files.
    dry_run : bool, optional
        If True, only report what would be done without making changes.

    Returns
    -------
    int
        Number of stub files generated (or that would be generated in dry-run mode).
    """
    if not files:
        return 0

    # Convert file paths to module names
    modules = []
    for file_path in files:
        try:
            rel_path = file_path.relative_to(src_dir.parent)
            # Convert path to module notation (e.g., tnfr/utils/cache.py -> tnfr.utils.cache)
            module = str(rel_path.with_suffix("")).replace("/", ".")
            # Skip __init__ modules for now - they're often handled specially
            if not module.endswith(".__init__"):
                modules.append(module)
        except ValueError:
            continue

    if not modules:
        return 0

    if dry_run:
        print("Would generate stubs for:")
        for mod in modules:
            print(f"  - {mod}")
        return len(modules)

    # Run stubgen for each module
    success_count = 0
    for module in modules:
        try:
            cmd = [
                "stubgen",
                "-p",
                module,
                "-o",
                str(src_dir.parent),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print(f"✓ Generated stub for {module}")
                success_count += 1
            else:
                print(f"✗ Failed to generate stub for {module}")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
        except Exception as e:
            print(f"✗ Error processing {module}: {e}")

    return success_count


def main() -> int:
    """Main entry point for stub generation script."""
    parser = argparse.ArgumentParser(
        description="Generate .pyi stub files for TNFR modules"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("src/tnfr"),
        help="Source directory to scan (default: src/tnfr)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if any stubs are missing and exit with error if so",
    )
    parser.add_argument(
        "--check-sync",
        action="store_true",
        help="Check if any stubs are outdated (modified after .py file) and exit with error",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Regenerate outdated stub files (where .py is newer than .pyi)",
    )

    args = parser.parse_args()

    src_dir = args.src_dir
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist", file=sys.stderr)
        return 1

    # Handle --sync mode
    if args.sync:
        outdated = find_outdated_stubs(src_dir)
        if not outdated:
            print("✓ All stub files are up to date")
            return 0

        print(f"Found {len(outdated)} outdated stub files")
        generated = generate_stubs(outdated, src_dir, dry_run=args.dry_run)

        if args.dry_run:
            print(f"\nWould regenerate {generated} stub files")
        else:
            print(f"\n✓ Regenerated {generated} stub files")
        return 0

    # Handle --check-sync mode
    if args.check_sync:
        outdated = find_outdated_stubs(src_dir)
        if not outdated:
            print("✓ All stub files are synchronized")
            return 0

        print(f"Found {len(outdated)} outdated stub files:")
        for py_file in outdated:
            print(f"  - {py_file}")
        print(
            "\nRun 'python scripts/generate_stubs.py --sync' to update outdated stubs"
        )
        return 1

    # Handle missing stubs check
    missing = find_missing_stubs(src_dir)

    if not missing:
        print("✓ All modules have stub files")
        return 0

    print(f"Found {len(missing)} Python files without stub files")

    if args.check:
        print("\nMissing stub files:")
        for py_file in missing:
            print(f"  - {py_file}")
        print(
            "\nRun 'python scripts/generate_stubs.py' to generate missing stubs"
        )
        return 1

    generated = generate_stubs(missing, src_dir, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nWould generate {generated} stub files")
    else:
        print(f"\n✓ Generated {generated} stub files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
