#!/usr/bin/env python3
"""Stage canonical repository documentation for the MkDocs build."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.clean_repository import generated_directory
from scripts.verify_internal_references import (
    _markdown_segments,
    rewrite_markdown_prose,
)

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
    ".zip",
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


def _separate_display_math(content: str) -> str:
    """Give matched standalone TeX displays paragraph boundaries in staging."""

    def separate(prose: str, offset: int) -> str:
        opening = None
        boundaries: set[int] = set()
        for token in re.finditer(r"(?m)^ {0,3}\\([\[\]])[ \t]*(?:\n|$)", prose):
            absolute = offset + token.start()
            if absolute and content[absolute - 1] != "\n":
                continue
            end = offset + token.end()
            if (
                not token.group(0).endswith("\n")
                and end < len(content)
                and content[end] != "\n"
            ):
                continue
            if token.group(1) == "[":
                opening = token.start()
            elif opening is not None:
                if opening and not prose[:opening].endswith("\n\n"):
                    boundaries.add(opening)
                if token.end() < len(prose) and prose[token.end()] != "\n":
                    boundaries.add(token.end())
                opening = None
        pieces, previous = [], 0
        for boundary in sorted(boundaries):
            pieces.extend((prose[previous:boundary], "\n"))
            previous = boundary
        pieces.append(prose[previous:])
        return "".join(pieces)

    rendered, offset = [], 0
    for text, prose in _markdown_segments(content):
        rendered.append(separate(text, offset) if prose else text)
        offset += len(text)
    return "".join(rendered)


def prepare() -> Path:
    """Create a deterministic documentation source tree and return its path."""

    expected = generated_directory(REPO_ROOT, "build/docs-source")
    if STAGE_DIR.absolute() != expected:
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
        "applications",
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

        def convert_links(prose: str) -> str:
            rendered = re.sub(r"\]\(([^)#\s]*)(#[^)]*)?\)", site_link, prose)
            return re.sub(
                r"^(\s{0,3}\[[^\]]+\]:\s*)(<[^>\n]+>|[^\s]+)([^\n]*)$",
                reference_link,
                rendered,
                flags=re.MULTILINE,
            )

        rendered = _separate_display_math(
            rewrite_markdown_prose(content, convert_links)
        )
        if rendered != content:
            document.write_text(rendered, encoding="utf-8")

    print(f"Documentation staged at {STAGE_DIR}")
    return STAGE_DIR


if __name__ == "__main__":
    prepare()
