#!/usr/bin/env python3
"""Stage canonical repository documentation for the MkDocs build."""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO_ROOT / "build" / "docs-source"

ROOT_FILES = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "TESTING.md",
    "SECURITY.md",
    "LICENSE.md",
    "CITATION.cff",
    "pyproject.toml",
)

TREE_SUFFIXES = {
    ".md",
    ".py",
    ".pyi",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".txt",
    ".pdf",
    ".png",
    ".svg",
    ".csv",
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
        if any(part in {"__pycache__", "output", "_build"} for part in source.parts):
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

    _copy_file(
        REPO_ROOT / ".github" / "WORKFLOWS.md",
        STAGE_DIR / "WORKFLOWS.md",
    )
    _copy_file(
        REPO_ROOT / ".github" / "agents" / "my-agent.md",
        STAGE_DIR / ".github" / "agents" / "my-agent.md",
    )
    _copy_file(
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        STAGE_DIR / ".github" / "workflows" / "ci.yml",
    )

    for relative in (
        "docs",
        "theory",
        "examples",
        "benchmarks",
        "factorization-lab",
        "scripts",
        "src/tnfr",
        "tests",
    ):
        _copy_tree(relative)

    theory_index = STAGE_DIR / "theory" / "README.md"
    theory_index.write_text(
        theory_index.read_text(encoding="utf-8").replace(
            "(../README.md)", "(../index.md)"
        ),
        encoding="utf-8",
    )

    print(f"Documentation staged at {STAGE_DIR}")
    return STAGE_DIR


if __name__ == "__main__":
    prepare()
