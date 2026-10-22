#!/usr/bin/env python3
"""Stage canonical repository documentation for the MkDocs build."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.clean_repository import generated_directory
from scripts.research_book import (
    BOOK_PATH,
    chapter_id,
    documents,
    front,
    read_book,
    read_document_bytes,
)
from scripts.verify_internal_references import (
    _markdown_segments,
    _repository_url,
    is_frozen_markdown_capture,
    rewrite_markdown_prose,
)

STAGE_DIR = REPO_ROOT / "build" / "docs-source"
RESEARCH_BOOK = BOOK_PATH

ROOT_FILES = (
    "AGENTS.md",
    "TNFR_lineas_de_investigacion.txt",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "TESTING.md",
    "SECURITY.md",
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


def _source_inventory() -> frozenset[str] | None:
    """Admit checkout sources once, preserving tracked files and current work.

    A source archive has no Git metadata and retains the declared copy rules.
    A checkout must provide its inventory successfully before staging is reset.
    """
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot inventory Git documentation sources") from exc
    return frozenset(os.fsdecode(path) for path in result.stdout.split(b"\0") if path)


def _copy_tree(relative: str, *, source_inventory: frozenset[str] | None) -> None:
    source_root = REPO_ROOT / relative
    if not source_root.exists():
        return
    for source in source_root.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in TREE_SUFFIXES:
            continue
        source_relative = source.relative_to(REPO_ROOT)
        if (
            source_inventory is not None
            and source_relative.as_posix() not in source_inventory
        ):
            continue
        if is_frozen_markdown_capture(source_relative):
            continue
        if any(
            part in {"__pycache__", "output", "outputs", "results", "_build"}
            for part in source_relative.parts
        ):
            continue
        destination = STAGE_DIR / source_relative
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
    source_inventory = _source_inventory()
    # Validate the complete catalog and every source chapter before replacing
    # prior staging. A missing modular chapter must preserve the last build.
    chapter_sources = documents(REPO_ROOT)
    book_front = front(REPO_ROOT) if (REPO_ROOT / RESEARCH_BOOK).is_file() else None
    chapter_routes = (
        {chapter_id(c.path): c.path for c in read_book(REPO_ROOT).chapters}
        if book_front is not None
        else {}
    )
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True)

    readme = REPO_ROOT / "README.md"
    _copy_file(readme, STAGE_DIR / "index.md")

    for relative in ROOT_FILES:
        source, destination = REPO_ROOT / relative, STAGE_DIR / relative
        if source.is_file():
            _copy_file(source, destination)
        else:
            destination.write_bytes(read_document_bytes(REPO_ROOT, relative))

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
        _copy_tree(relative, source_inventory=source_inventory)

    # Chapter paths remain website routes and link-resolution bases in both
    # legacy containers and modular checkouts. The program route stays compact.
    for relative, payload in chapter_sources.items():
        if is_frozen_markdown_capture(Path(relative)):
            continue
        destination = STAGE_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    if book_front is not None:
        destination = STAGE_DIR / RESEARCH_BOOK
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(book_front, encoding="utf-8")

    # Preserve repository paths, but link directory targets to an actual index
    # or the repository browser. A static site cannot display source folders.
    for document in STAGE_DIR.rglob("*.md"):
        source = REPO_ROOT / document.relative_to(STAGE_DIR)

        def site_link(match: re.Match[str]) -> str:
            target, fragment = match.group(1), match.group(2) or ""
            repository_target = _repository_url(target + fragment)
            if repository_target is not None:
                parsed = urlsplit(repository_target)
                destination = (REPO_ROOT / unquote(parsed.path)).resolve()
                fragment = "#" + parsed.fragment if parsed.fragment else ""
            else:
                if not target or ":" in target or target.startswith("/"):
                    return match.group(0)
                destination = (source.parent / target).resolve()
            chapter_route = (
                chapter_routes.get(unquote(fragment.removeprefix("#")))
                if destination == (REPO_ROOT / RESEARCH_BOOK).resolve()
                else None
            )
            if chapter_route is not None:
                destination = (REPO_ROOT / chapter_route).resolve()
                fragment = ""
            if not destination.is_relative_to(REPO_ROOT):
                return match.group(0)
            if destination == readme.resolve():
                linked = STAGE_DIR / "index.md"
            elif (
                destination.is_dir()
                or (STAGE_DIR / destination.relative_to(REPO_ROOT)).is_dir()
            ):
                staged_destination = STAGE_DIR / destination.relative_to(REPO_ROOT)
                index = next(
                    (
                        staged_destination / name
                        for name in ("README.md", "index.md")
                        if (staged_destination / name).is_file()
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
                linked = index
            elif (repository_target is not None or chapter_route is not None) and (
                STAGE_DIR / destination.relative_to(REPO_ROOT)
            ).is_file():
                linked = STAGE_DIR / destination.relative_to(REPO_ROOT)
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
