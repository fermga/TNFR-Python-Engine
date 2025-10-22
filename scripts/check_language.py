#!/usr/bin/env python3
"""Enforce the English-only policy by scanning for Spanish tokens.

This lightweight check runs as part of the repository's quality gate to ensure
new contributions do not resurrect the legacy Spanish literals that used to
coexist with the canonical English identifiers. It reads the repository's
tracked files (excluding binary blobs and opt-in paths) and flags
case-insensitive matches for the configured Spanish keywords or any accented
characters that commonly appear in Spanish prose.

The defaults mirror the compatibility tokens retired in TNFR 12.0.0. The
settings can be customised from ``pyproject.toml`` under
``[tool.tnfr.language_check]`` if the policy ever evolves. Keeping the logic in
code (instead of a Flake8 plugin) avoids an extra dependency while still
providing deterministic enforcement in CI and local development.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <=3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - missing optional dependency
        tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LanguagePolicy:
    """Configuration for the English-only enforcement."""

    disallowed_keywords: Sequence[str]
    accented_characters: Sequence[str]
    excluded_globs: Sequence[str]


DEFAULT_POLICY = LanguagePolicy(
    disallowed_keywords=(
        "est\u0061ble",
        "transici\u006f\u006e",
        "transici\u00f3n",
        "diso\u006enante",
        "operad\u006fres",
        "operad\u006fr",
    ),
    accented_characters=(
        "\u00e1",
        "\u00e9",
        "\u00ed",
        "\u00f3",
        "\u00fa",
        "\u00fc",
        "\u00f1",
        "\u00c1",
        "\u00c9",
        "\u00cd",
        "\u00d3",
        "\u00da",
        "\u00dc",
        "\u00d1",
        "\u00bf",
        "\u00a1",
    ),
    excluded_globs=(
        "TNFR.pdf",
        "benchmarks/**/*.pdf",
    ),
)


def _load_policy(repo_root: Path) -> LanguagePolicy:
    """Read overrides from ``pyproject.toml`` if available."""

    if tomllib is None:
        return DEFAULT_POLICY

    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return DEFAULT_POLICY

    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)

    tool_section = data.get("tool", {})  # type: ignore[assignment]
    tnfr_section = tool_section.get("tnfr", {})  # type: ignore[assignment]
    policy_section = tnfr_section.get("language_check")  # type: ignore[assignment]
    if not isinstance(policy_section, dict):
        return DEFAULT_POLICY

    keywords = policy_section.get("disallowed_keywords")
    accented = policy_section.get("accented_characters")
    excludes = policy_section.get("exclude")

    return LanguagePolicy(
        disallowed_keywords=tuple(
            sorted(
                {*(DEFAULT_POLICY.disallowed_keywords), *(keywords or [])},
                key=str.lower,
            )
        ),
        accented_characters=tuple(
            sorted({*(DEFAULT_POLICY.accented_characters), *(accented or [])})
        ),
        excluded_globs=tuple(
            sorted({*(DEFAULT_POLICY.excluded_globs), *(excludes or [])})
        ),
    )


def _collect_tracked_files(repo_root: Path) -> Sequence[Path]:
    """Return the list of tracked files using ``git ls-files``."""

    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )
    return [repo_root / line for line in result.stdout.splitlines() if line]


def _is_binary(path: Path) -> bool:
    """Heuristically detect binary files via null bytes in the first chunk."""

    try:
        with path.open("rb") as handle:
            chunk = handle.read(1024)
    except OSError:
        return True
    return b"\x00" in chunk


def _should_exclude(path: Path, patterns: Iterable[str], repo_root: Path) -> bool:
    """Return ``True`` when ``path`` matches one of the configured globs."""

    relative = path.relative_to(repo_root).as_posix()
    return any(
        relative == pattern
        or relative.startswith(f"{pattern.rstrip('/')}/")
        or fnmatch.fnmatch(relative, pattern)
        for pattern in patterns
    )


def _scan_file(
    path: Path,
    policy: LanguagePolicy,
    repo_root: Path,
) -> list[str]:
    """Return violation messages for ``path`` under ``policy``."""

    if _should_exclude(path, policy.excluded_globs, repo_root):
        return []
    if _is_binary(path):
        return []

    violations: list[str] = []
    accent_set = set(policy.accented_characters)
    keywords_lower = {word.lower() for word in policy.disallowed_keywords}

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        accents_found = sorted({char for char in raw_line if char in accent_set})
        lowered = raw_line.lower()
        keywords_found = sorted({word for word in keywords_lower if word in lowered})

        if not accents_found and not keywords_found:
            continue

        reasons: list[str] = []
        if keywords_found:
            reasons.append("Spanish keyword(s): " + ", ".join(keywords_found))
        if accents_found:
            reasons.append("accented character(s): " + ", ".join(accents_found))
        snippet = line if len(line) <= 120 else f"{line[:117]}..."
        violations.append(
            f"{path.relative_to(repo_root)}:{line_number}: {'; '.join(reasons)} -> {snippet}"
        )

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python scripts/check_language.py``."""

    parser = argparse.ArgumentParser(
        description="Fail when tracked files include Spanish tokens or accents.",
    )
    parser.parse_args(argv)  # Reserved for future switches

    repo_root = Path(__file__).resolve().parents[1]
    policy = _load_policy(repo_root)
    tracked_files = _collect_tracked_files(repo_root)

    all_violations: list[str] = []
    for path in tracked_files:
        all_violations.extend(_scan_file(path, policy, repo_root))

    if all_violations:
        print("Spanish language guard detected violations:", file=sys.stderr)
        for violation in all_violations:
            print(violation, file=sys.stderr)
        print(
            "\nRemove or rewrite the offending tokens before committing.",
            file=sys.stderr,
        )
        return 1

    print("Spanish language guard: no violations detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
