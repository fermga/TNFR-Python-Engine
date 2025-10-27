#!/usr/bin/env python3
"""Enforce TNFR changelog fragments for relevant pull requests."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Tuple


def run_git(args: Sequence[str]) -> str:
    """Run a git command and return its stdout as text."""
    return subprocess.check_output(["git", *args], text=True)


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
        default=["pyproject.toml", "mkdocs.yml"],
        help="Specific file that requires a changelog fragment (can be repeated)",
    )

    args = parser.parse_args(argv)

    changelog_dir = Path(args.changelog_dir).as_posix().rstrip("/") + "/"

    try:
        run_git(["rev-parse", "--verify", args.base])
    except subprocess.CalledProcessError:
        print(
            f"::warning::Unable to resolve base ref '{args.base}'. Skipping changelog enforcement.",
            file=sys.stderr,
        )
        return 0

    name_status = parse_name_status(
        run_git(["diff", "--name-status", f"{args.base}...HEAD"])
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
