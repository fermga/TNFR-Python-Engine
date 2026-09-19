#!/usr/bin/env python3
"""Stage canonical repository documentation for the MkDocs build."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO_ROOT / "build" / "docs-source"

ROOT_FILES = (
    "AGENTS.md",
    "TNFR_lineas_de_investigacion.txt",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "TESTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "LICENSE.md",
    "CITATION.cff",
    "pyproject.toml",
    "Makefile",
    "bandit.yaml",
    ".pre-commit-config.yaml",
)

TREE_SUFFIXES = {
    ".md",
    ".py",
    ".pyi",
    ".json",
    ".js",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".pdf",
    ".png",
    ".svg",
    ".csv",
    ".ipynb",
    ".sh",
}


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(relative: str) -> None:
    source_root = REPO_ROOT / relative
    if not source_root.exists():
        return
    for source in source_root.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in TREE_SUFFIXES:
            continue
        if any(
            part in {"__pycache__", "output", "outputs", "results", "_build"}
            for part in source.parts
        ):
            continue
        destination = STAGE_DIR / source.relative_to(REPO_ROOT)
        _copy_file(source, destination)


def prepare() -> Path:
    """Create a deterministic documentation source tree and return its path."""

    expected = (REPO_ROOT / "build").resolve()
    if STAGE_DIR.resolve().parent != expected:
        raise RuntimeError(f"unsafe documentation staging path: {STAGE_DIR}")
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True)

    readme = REPO_ROOT / "README.md"
    _copy_file(readme, STAGE_DIR / "index.md")

    for relative in ROOT_FILES:
        _copy_file(REPO_ROOT / relative, STAGE_DIR / relative)

    for relative in (
        ".github",
        "docs",
        "theory",
        "examples",
        "benchmarks",
        "factorization-lab",
        "primality-test",
        "manual",
        "scripts",
        "src/tnfr",
        "tests",
    ):
        _copy_tree(relative)

    # Preserve repository paths, but link directory targets to an actual index
    # or the repository browser. A static site cannot display source folders.
    for document in STAGE_DIR.rglob("*.md"):
        source = REPO_ROOT / document.relative_to(STAGE_DIR)

        def site_link(match: re.Match[str]) -> str:
            target, fragment = match.group(1), match.group(2) or ""
            if not target or ":" in target or target.startswith("/"):
                return match.group(0)
            destination = (source.parent / target).resolve()
            if not destination.is_relative_to(REPO_ROOT):
                return match.group(0)
            if destination == readme.resolve():
                linked = STAGE_DIR / "index.md"
            elif destination.is_dir():
                index = next(
                    (
                        destination / name
                        for name in ("README.md", "index.md")
                        if (destination / name).is_file()
                    ),
                    None,
                )
                if index is None:
                    remote = "https://github.com/fermga/TNFR-Python-Engine/tree/main/"
                    return (
                        "]("
                        + remote
                        + quote(destination.relative_to(REPO_ROOT).as_posix())
                        + fragment
                        + ")"
                    )
                linked = STAGE_DIR / index.relative_to(REPO_ROOT)
            else:
                return match.group(0)
            relative = Path(os.path.relpath(linked, document.parent)).as_posix()
            return "](" + relative + fragment + ")"

        content = document.read_text(encoding="utf-8")
        rendered = re.sub(r"\]\(([^)#\s]*)(#[^)]*)?\)", site_link, content)

        def reference_link(match: re.Match[str]) -> str:
            prefix, raw, suffix = match.groups()
            target = raw[1:-1] if raw.startswith("<") else raw
            # Reuse the inline conversion so reference-style and inline links
            # cannot choose different destinations for the same source file.
            converted = re.sub(
                r"\]\(([^)#\s]*)(#[^)]*)?\)", site_link, "](" + target + ")"
            )
            target = converted[2:-1]
            if raw.startswith("<"):
                target = "<" + target + ">"
            return prefix + target + suffix

        rendered = re.sub(
            r"^(\s{0,3}\[[^\]]+\]:\s*)(<[^>\n]+>|[^\s]+)([^\n]*)$",
            reference_link,
            rendered,
            flags=re.MULTILINE,
        )
        if rendered != content:
            document.write_text(rendered, encoding="utf-8")

    print(f"Documentation staged at {STAGE_DIR}")
    return STAGE_DIR


if __name__ == "__main__":
    prepare()
