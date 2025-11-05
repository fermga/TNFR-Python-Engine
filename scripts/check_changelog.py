#!/usr/bin/env python3
"""Enforce TNFR changelog fragments for relevant pull requests."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple

# Add src to path to import security module
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tnfr.security import run_command_safely, validate_git_ref, CommandValidationError


def run_git(args: Sequence[str]) -> str:
    """Run a git command and return its stdout as text."""
    # Validate git arguments for security
    validated_args = [str(arg) for arg in args]
    result = run_command_safely(["git", *validated_args], check=True)
    return result.stdout


def parse_name_status(output: str) -> Sequence[Tuple[str, str]]:
    """Parse ``git diff --name-status`` output."""
    entries: list[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0].strip()
        path = parts[-1].strip()
        entries.append((status, path))
    return entries


def is_relevant(path: str, prefixes: Sequence[str], singled_out: Sequence[str]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes) or path in singled_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=os.environ.get("TNFR_CHANGELOG_BASE", "origin/main"),
        help="Reference revision for the diff (default: %(default)s)",
    )
    parser.add_argument(
        "--changelog-dir",
        default="docs/changelog.d",
        help="Directory that stores changelog fragments (default: %(default)s)",
    )
    parser.add_argument(
        "--relevant-prefix",
        action="append",
        dest="relevant_prefixes",
        default=["src/", "tests/", "scripts/", "benchmarks/"],
        help="Directory prefix that requires a changelog fragment (can be repeated)",
    )
    parser.add_argument(
        "--relevant-file",
        action="append",
        dest="relevant_files",
        default=[
            "pyproject.toml",
            "docs/source/conf.py",
            "docs/source/index.rst",
            "netlify.toml",
        ],
        help="Specific file that requires a changelog fragment (can be repeated)",
    )

    args = parser.parse_args(argv)

    # Validate base ref for security
    try:
        validated_base = validate_git_ref(args.base)
    except CommandValidationError as exc:
        print(
            f"::error::Invalid base ref '{args.base}': {exc}",
            file=sys.stderr,
        )
        return 1

    changelog_dir = Path(args.changelog_dir).as_posix().rstrip("/") + "/"

    try:
        run_git(["rev-parse", "--verify", validated_base])
    except subprocess.CalledProcessError:
        print(
            f"::warning::Unable to resolve base ref '{validated_base}'. Skipping changelog enforcement.",
            file=sys.stderr,
        )
        return 0

    name_status = parse_name_status(
        run_git(["diff", "--name-status", f"{validated_base}...HEAD"])
    )
    changed_paths = [path for _, path in name_status]

    relevant_changes = [
        path
        for path in changed_paths
        if is_relevant(path, args.relevant_prefixes, args.relevant_files)
    ]

    if not relevant_changes:
        print("No relevant structural changes detected; changelog fragment optional.")
        return 0

    added_fragments = [
        path
        for status, path in name_status
        if path.startswith(changelog_dir)
        and path.endswith(".md")
        and status.startswith("A")
    ]

    if added_fragments:
        print("Detected changelog fragments:")
        for fragment in added_fragments:
            print(f"  - {fragment}")
        return 0

    print("::error::Missing changelog fragment for structural updates.")
    print("Relevant changes:")
    for path in relevant_changes:
        print(f"  - {path}")
    print(
        "Add a Markdown fragment under docs/changelog.d named '<ticket>.<type>.md' "
        "and describe the coherent reorganisation it introduces."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
