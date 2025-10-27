#!/usr/bin/env python3
"""TNFR release rollback orchestrator.

This utility coordinates PyPI yanks, tag cleanup, and optional environment
restoration to keep the TNFR ledger coherent after a faulty release. It wraps
canonical commands instead of hiding them so every operator remains auditable.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from typing import Iterable, Optional


_LOG_FORMAT = "%(levelname)s: %(message)s"


class RollbackError(RuntimeError):
    """Raised when a rollback step fails to execute."""


def _run_command(
    command: Iterable[str],
    *,
    dry_run: bool,
    check: bool = True,
    capture_output: bool = False,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str] | None:
    rendered = " ".join(shlex.quote(part) for part in command)
    logging.info("$ %s", rendered)
    if dry_run:
        return None
    try:
        return subprocess.run(
            list(command),
            check=check,
            capture_output=capture_output,
            text=True,
            env=None if env is None else {**os.environ, **env},
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        if capture_output:
            logging.error(exc.stdout)
            logging.error(exc.stderr)
        raise RollbackError(f"Command failed: {rendered}") from exc


def _discover_tags() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "--list", "v*", "--sort=-v:refname"],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not tags:
        raise RollbackError("No semantic-release tags found in repository")
    return tags


def _previous_tag(current_tag: str) -> Optional[str]:
    tags = _discover_tags()
    try:
        index = tags.index(current_tag)
    except ValueError as exc:  # pragma: no cover - defensive
        raise RollbackError(f"Tag {current_tag} is not present in this repository") from exc
    if index + 1 < len(tags):
        return tags[index + 1]
    return None


def _twine_environment(username: str | None, password: str | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if username:
        env["TWINE_USERNAME"] = username
    if password:
        env["TWINE_PASSWORD"] = password
    return env


def _yank_from_pypi(
    *,
    package: str,
    version: str,
    repository: str,
    reason: str,
    username: str | None,
    password: str | None,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "twine",
        "yank",
        package,
        version,
        "--repository",
        repository,
        "--comment",
        reason,
    ]
    env = _twine_environment(username, password)
    _run_command(command, dry_run=dry_run, env=env)


def _delete_tags(*, version_tag: str, dry_run: bool) -> None:
    _run_command(["git", "tag", "-d", version_tag], dry_run=dry_run)
    _run_command(["git", "push", "origin", f":refs/tags/{version_tag}"], dry_run=dry_run)


def _delete_github_release(*, version_tag: str, dry_run: bool) -> None:
    if shutil.which("gh") is None:
        logging.warning("GitHub CLI not found; skip GitHub release deletion")
        return
    command = ["gh", "release", "delete", version_tag, "--cleanup-tag", "--yes"]
    _run_command(command, dry_run=dry_run, check=False)


def _checkout_previous(previous_tag: str, *, dry_run: bool) -> None:
    _run_command(["git", "checkout", previous_tag], dry_run=dry_run)


def _install_previous(package: str, previous_version: str, *, dry_run: bool) -> None:
    _run_command(
        [sys.executable, "-m", "pip", "install", f"{package}=={previous_version}"],
        dry_run=dry_run,
    )


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ")
    except EOFError:  # pragma: no cover - interactive safeguard
        return False
    return answer.strip().lower() in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rollback a TNFR release by yanking PyPI artefacts and cleaning up tags.",
    )
    parser.add_argument("--version", required=True, help="Semantic version to revoke (e.g. 16.0.0)")
    parser.add_argument(
        "--package-name",
        default="tnfr",
        help="PyPI package name (defaults to 'tnfr').",
    )
    parser.add_argument(
        "--pypi-repository",
        default=os.environ.get("PYPI_REPOSITORY", "pypi"),
        help="Repository alias configured in ~/.pypirc (defaults to 'pypi').",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("PYPI_USERNAME", os.environ.get("TWINE_USERNAME")),
        help="PyPI username or __token__ value (falls back to environment).",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("PYPI_PASSWORD", os.environ.get("PYPI_API_TOKEN")),
        help="PyPI password or API token (falls back to environment).",
    )
    parser.add_argument(
        "--reason",
        default="Rollback triggered to preserve TNFR coherence",
        help="Comment recorded with the yank operation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing commands.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive prompt (useful in CI once the plan is reviewed).",
    )
    parser.add_argument(
        "--skip-yank",
        action="store_true",
        help="Do not attempt the PyPI yank (useful if it already happened).",
    )
    parser.add_argument(
        "--skip-tag",
        action="store_true",
        help="Keep the release tag (only for post-mortem ledger inspection).",
    )
    parser.add_argument(
        "--skip-checkout",
        action="store_true",
        help="Avoid checking out the previous tag (retain current working tree).",
    )
    parser.add_argument(
        "--install-previous",
        action="store_true",
        help="Install the previous package version after checkout.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

    version = args.version.lstrip("v")
    tag = f"v{version}"
    logging.info("Preparing rollback for %s", tag)

    if not args.confirm and not args.dry_run:
        if not _confirm("Proceed with rollback actions?"):
            logging.info("Rollback aborted by operator")
            return 0

    if not args.skip_yank:
        logging.info("Yanking %s %s from %s", args.package_name, version, args.pypi_repository)
        _yank_from_pypi(
            package=args.package_name,
            version=version,
            repository=args.pypi_repository,
            reason=args.reason,
            username=args.username,
            password=args.password,
            dry_run=args.dry_run,
        )
    else:
        logging.info("Skipping PyPI yank as requested")

    if not args.skip_tag:
        logging.info("Deleting tag %s", tag)
        _delete_tags(version_tag=tag, dry_run=args.dry_run)
        _delete_github_release(version_tag=tag, dry_run=args.dry_run)
    else:
        logging.info("Tag cleanup skipped by operator")

    if args.skip_checkout:
        logging.info("Repository checkout step skipped")
        previous_tag = _previous_tag(tag)
    else:
        previous_tag = _previous_tag(tag)
        if previous_tag is None:
            logging.warning("No previous semantic tag found; skipping checkout")
        else:
            logging.info("Checking out previous tag %s", previous_tag)
            _checkout_previous(previous_tag, dry_run=args.dry_run)

    if args.install_previous and previous_tag is not None:
        previous_version = previous_tag.lstrip("v")
        logging.info("Installing %s==%s", args.package_name, previous_version)
        _install_previous(args.package_name, previous_version, dry_run=args.dry_run)
    elif args.install_previous:
        logging.warning("Cannot install previous version; no historical tag detected")

    logging.info("Rollback sequence completed")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
