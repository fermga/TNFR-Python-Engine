#!/usr/bin/env python3
"""Validate local Markdown targets and GitHub-style heading fragments."""

from __future__ import annotations

import argparse
import os
import re
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "site-packages",
        "build",
        "dist",
        "site",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "publish",
        "tmp",
        # Frozen run captures retain their original source/path context.
        "artifacts",
        "output",
        "outputs",
        "results",
    }
)
LINK_PATTERN = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
REFERENCE_PATTERN = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(.+)$", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
CUSTOM_ID_PATTERN = re.compile(r"\{#([^}]+)\}\s*$")
HTML_ID_PATTERN = re.compile(r"""<a\s+(?:name|id)=["']([^"']+)["']""", re.I)


def _excluded(path: Path) -> bool:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    return (
        bool(set(relative.parts) & EXCLUDED_PARTS)
        or any(part.startswith(".venv") for part in relative.parts)
    ) or relative.as_posix().startswith(".github/agents/")


def _markdown_files(search_roots: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for value in search_roots:
        root = (REPO_ROOT / value).resolve()
        if root.is_file() and root.suffix.lower() == ".md" and not _excluded(root):
            files.add(root)
        elif root.is_dir():
            for directory, dirs, names in os.walk(root):
                parent = Path(directory)
                dirs[:] = [name for name in dirs if not _excluded(parent / name)]
                files.update(
                    parent / name
                    for name in names
                    if name.lower().endswith(".md") and not _excluded(parent / name)
                )
    return sorted(files)


def _github_slug(text: str) -> str:
    text = CUSTOM_ID_PATTERN.sub("", text)
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace(chr(96), "").lower()
    text = "".join(char for char in text if char.isalnum() or char in {" ", "-", "_"})
    return re.sub(r"\s+", "-", text.strip())


def _markdown_lines(content: str) -> Iterator[tuple[str, bool]]:
    """Share fenced-code boundaries between link validation and site staging."""
    fence = ""
    for line in content.splitlines(keepends=True):
        if fence:
            if re.fullmatch(
                r"\s{0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*",
                line,
            ):
                fence = ""
            yield line, False
        else:
            match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
            if match:
                fence = match.group(1)
                yield line, False
            else:
                yield line, True


def _without_fenced_code(content: str) -> str:
    """Ignore examples containing Markdown syntax rather than rendered links."""
    return "".join(line if prose else "\n" for line, prose in _markdown_lines(content))


def _inline_segments(content: str) -> Iterator[tuple[str, bool]]:
    """Recognize equal backtick runs, including multiline spans and escapes."""
    runs = list(re.finditer(r"`+", content))
    positions: dict[int, list[int]] = {}
    for run in runs:
        positions.setdefault(run.end() - run.start(), []).append(run.start())
    cursor = 0
    for run in runs:
        start = run.start()
        if start < cursor:
            continue
        before = start
        while before > 0 and content[before - 1] == "\\":
            before -= 1
        if (start - before) % 2:
            start += 1  # Only the first backtick is escaped outside a span.
        width = run.end() - start
        if not width:
            continue
        candidates = positions.get(width, [])
        following = bisect_right(candidates, run.start())
        if following == len(candidates):
            continue
        end = candidates[following] + width
        yield content[cursor:start], True
        yield content[start:end], False
        cursor = end
    yield content[cursor:], True


def _markdown_segments(content: str) -> Iterator[tuple[str, bool]]:
    """Share paragraph, fenced-code and inline-code boundaries across consumers."""
    pending: list[str] = []
    for line, prose in _markdown_lines(content):
        if prose and line.strip():
            pending.append(line)
        else:
            yield from _inline_segments("".join(pending))
            pending.clear()
            yield line, prose
    yield from _inline_segments("".join(pending))


def rewrite_markdown_prose(content: str, transform: Callable[[str], str]) -> str:
    """Transform prose while preserving fenced and inline code verbatim."""
    return "".join(
        transform(text) if prose else text
        for text, prose in _markdown_segments(content)
    )


def _anchors(markdown: Path) -> set[str]:
    content = _without_fenced_code(markdown.read_text(encoding="utf-8-sig"))
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
    return bool(urlsplit(target).scheme) or target.startswith("//")


def _target(raw: str) -> str:
    if raw.startswith("<") and ">" in raw:
        return raw[1 : raw.index(">")]
    return re.split(r"\s+[\"\']", raw.strip(), maxsplit=1)[0]


def _repository_url(target: str) -> str | None:
    """Resolve current-main own-repository URLs locally; leave external sites alone."""
    url = urlsplit(target)
    if url.netloc.lower() != "github.com":
        return None
    for kind in ("blob", "tree"):
        prefix = f"/fermga/TNFR-Python-Engine/{kind}/main/"
        if url.path.startswith(prefix):
            return unquote(url.path[len(prefix) :]) + (
                "#" + unquote(url.fragment) if url.fragment else ""
            )
    return None


def verify(search_roots: Iterable[str], verbose: bool = False) -> tuple[int, list[str]]:
    references = 0
    failures: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for source in _markdown_files(search_roots):
        content = "".join(
            text if prose else "\n" * text.count("\n")
            for text, prose in _markdown_segments(
                source.read_text(encoding="utf-8-sig")
            )
        )
        # Algebra such as [1-s](q-k) is not a link inside a math wrapper.
        content = re.sub(r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$", "", content)
        content = re.sub(r"(?<!\\)\$(?!\s)[^\n$]*?(?<!\s)(?<!\\)\$", "", content)
        targets = [value for _, value in LINK_PATTERN.findall(content)]
        targets.extend(REFERENCE_PATTERN.findall(content))
        for raw_target in targets:
            target = _target(raw_target)
            repository_target = _repository_url(target)
            if repository_target is not None:
                target = repository_target
            elif _is_external(target):
                continue
            else:
                target = unquote(target)
            path_text, separator, fragment = target.partition("#")
            if not path_text:
                destination = source
            else:
                clean_path = Path(path_text.replace("\\", "/"))
                base = REPO_ROOT if repository_target is not None else source.parent
                destination = (base / clean_path).resolve()
            references += 1
            display = f"{source.relative_to(REPO_ROOT)} -> {raw_target}"
            if not destination.is_relative_to(REPO_ROOT):
                failures.append(f"target outside repository: {display}")
                continue
            if not destination.exists():
                failures.append(f"missing target: {display}")
                continue
            if separator and fragment and destination.is_dir():
                index = next(
                    (
                        destination / name
                        for name in ("README.md", "index.md")
                        if (destination / name).is_file()
                    ),
                    None,
                )
                if index is None:
                    failures.append(f"directory fragment has no local index: {display}")
                    continue
                destination = index
            if separator and fragment and destination.suffix.lower() == ".md":
                if destination not in anchor_cache:
                    anchor_cache[destination] = _anchors(destination)
                if fragment not in anchor_cache[destination]:
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
