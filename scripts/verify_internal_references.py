#!/usr/bin/env python3
"""Validate local Markdown targets and GitHub-style heading fragments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = frozenset(
    {
        ".git", ".venv", "venv", "site-packages", "build", "dist", "site",
        "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox",
        "manual", "publish", "tmp",
    }
)
LINK_PATTERN = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
CUSTOM_ID_PATTERN = re.compile(r"\{#([^}]+)\}\s*$")
HTML_ID_PATTERN = re.compile(r"""<a\s+(?:name|id)=["']([^"']+)["']""", re.I)
FILE_SUFFIXES = {
    ".bib", ".cff", ".csv", ".html", ".ini", ".ipynb", ".json", ".md",
    ".pdf", ".png", ".py", ".pyi", ".rst", ".sh", ".svg", ".toml",
    ".txt", ".yaml", ".yml",
}


def _excluded(path: Path) -> bool:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    return bool(set(relative.parts) & EXCLUDED_PARTS) or relative.as_posix().startswith(
        ".github/agents/"
    )


def _markdown_files(search_roots: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for value in search_roots:
        root = (REPO_ROOT / value).resolve()
        if root.is_file() and root.suffix.lower() == ".md" and not _excluded(root):
            files.add(root)
        elif root.is_dir():
            files.update(path for path in root.rglob("*.md") if not _excluded(path))
    return sorted(files)


def _github_slug(text: str) -> str:
    text = CUSTOM_ID_PATTERN.sub("", text)
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace(chr(96), "").lower()
    text = "".join(
        char for char in text if char.isalnum() or char in {" ", "-", "_"}
    )
    return re.sub(r"\s+", "-", text.strip())


def _anchors(markdown: Path) -> set[str]:
    content = markdown.read_text(encoding="utf-8-sig")
    anchors = set(HTML_ID_PATTERN.findall(content))
    counts: dict[str, int] = {}
    for heading in HEADING_PATTERN.findall(content):
        custom = CUSTOM_ID_PATTERN.search(heading)
        if custom:
            anchors.add(custom.group(1))
        slug = _github_slug(heading)
        if not slug:
            continue
        occurrence = counts.get(slug, 0)
        counts[slug] = occurrence + 1
        anchors.add(slug if occurrence == 0 else f"{slug}-{occurrence}")
    return anchors


def _is_external(target: str) -> bool:
    return target.lower().startswith(("http://", "https://", "mailto:", "codex://"))


def verify(search_roots: Iterable[str], verbose: bool = False) -> tuple[int, list[str]]:
    references = 0
    failures: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for source in _markdown_files(search_roots):
        content = source.read_text(encoding="utf-8-sig")
        for _, raw_target in LINK_PATTERN.findall(content):
            target = unquote(raw_target.strip().strip("<>"))
            if _is_external(target):
                continue
            path_text, separator, fragment = target.partition("#")
            if not path_text:
                destination = source
            else:
                clean_path = Path(path_text.replace("\\", "/"))
                if "/" not in path_text and clean_path.suffix.lower() not in FILE_SUFFIXES:
                    continue
                destination = (source.parent / clean_path).resolve()
            references += 1
            display = f"{source.relative_to(REPO_ROOT)} -> {raw_target}"
            if not destination.exists():
                failures.append(f"missing target: {display}")
                continue
            if separator and fragment and destination.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(destination, _anchors(destination))
                if fragment.lower() not in anchors:
                    failures.append(f"missing fragment: {display}")
                    continue
            if verbose:
                print(f"OK {display}")
    return references, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--dirs", nargs="+", default=["."])
    args = parser.parse_args()

    references, failures = verify(args.dirs, verbose=args.verbose)
    print(f"Checked {references} internal Markdown references")
    if failures:
        for failure in failures:
            print(f"ERROR {failure}")
        print(f"Broken references: {len(failures)}")
        return 1 if args.ci else 0
    print("All internal Markdown targets and fragments are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
