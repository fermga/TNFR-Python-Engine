"""Build a versioned development snapshot with recoverable research evidence.

Read committed Git blobs, never working-tree bytes or scientific producers.
The standalone evidence helper owns path and manifest admission.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # The standalone recovery helper still supports 3.10.
    tomllib = None

if __package__:
    from .development_bundle_evidence import (
        MAX_FILE_BYTES,
        canonical_name,
        check_path_collisions,
        validate_manifest,
    )
    from .research_book import BOOK_PATH, CATALOG_PATH, parse_book, parse_catalog
else:
    from development_bundle_evidence import (
        MAX_FILE_BYTES,
        canonical_name,
        check_path_collisions,
        validate_manifest,
    )
    from research_book import BOOK_PATH, CATALOG_PATH, parse_book, parse_catalog

REPOSITORY = "fermga/TNFR-Python-Engine"
REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "scripts/templates/development_bundle_start_here.md.in"
REQUIRED = {
    "AGENTS.md",
    ".github/agents/my-agent.md",
    "README.md",
    "LICENSE.md",
    "CONTRIBUTING.md",
    "TESTING.md",
    "pyproject.toml",
    "src/tnfr/_version.py",
    "theory/README.md",
    "docs/API_CONTRACTS.md",
    "docs/assets/mathjax.js",
    "scripts/build_development_bundle.py",
    "scripts/development_bundle_evidence.py",
    TEMPLATE,
}
GENERATED = {"START_HERE.md", "bundle/manifest.json"}


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def _commit(root: Path, revision: str) -> str:
    value = (
        _git(
            root, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
        )
        .decode("ascii")
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError("Revision did not resolve to a full Git commit")
    return value


def _inventory(root: Path, commit: str) -> dict[str, tuple[str, str, int]]:
    inventory = {}
    folded = set()
    for record in _git(root, "ls-tree", "-rlz", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        mode, kind, blob, raw_size = metadata.decode("ascii").split()
        name = canonical_name(raw_name.decode("utf-8"))
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"Only regular tracked files are supported: {name}")
        size = int(raw_size)
        if not 0 <= size <= MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the evidence-helper review budget: {name}")
        if name.casefold() in folded:
            raise ValueError(f"Duplicate or case-colliding source path: {name}")
        folded.add(name.casefold())
        inventory[name] = mode, blob, size
    research = (
        {BOOK_PATH, "scripts/research_book.py"}
        if BOOK_PATH in inventory or CATALOG_PATH in inventory
        else {
            "theory/EMERGENT_ONTOLOGY.md",
            "theory/research/FIVE_STAGE_EXECUTION_PLAN.md",
        }
    )
    if CATALOG_PATH in inventory:
        # The selected commit owns this catalog and its chapter inventory;
        # uncommitted edits cannot repair or redefine a release snapshot.
        catalog = parse_catalog(_git(root, "show", f"{commit}:{CATALOG_PATH}"))
        research |= {
            CATALOG_PATH,
            *(chapter.path for chapter in catalog.chapters),
            *(alias.path for alias in catalog.aliases),
        }
    elif BOOK_PATH in inventory:
        # A modular program without its committed catalog is not a legacy book.
        # Admit old releases through the actual legacy schema, not a filename.
        try:
            parse_book(_git(root, "show", f"{commit}:{BOOK_PATH}"))
        except ValueError as exc:
            raise ValueError(
                "Selected commit lacks a valid legacy sourcebook or " + CATALOG_PATH
            ) from exc
    missing = (REQUIRED | research) - inventory.keys()
    if missing:
        raise ValueError(f"Selected commit lacks required owners: {sorted(missing)}")
    for name in inventory:
        if name.casefold() in {item.casefold() for item in GENERATED} or (
            name.casefold() == "bundle" or name.casefold().startswith("bundle/")
        ):
            raise ValueError(f"Tracked path conflicts with bundle metadata: {name}")
        if any(part.casefold() == ".git" for part in name.split("/")):
            raise ValueError("Git administrative files cannot enter the snapshot")
        parts = name.split("/")
        if any("/".join(parts[:i]).casefold() in folded for i in range(1, len(parts))):
            raise ValueError(f"File/directory path collision: {name}")
    check_path_collisions([*inventory, *GENERATED])
    return inventory


def _version(fallback: bytes, pyproject: bytes) -> str:
    if tomllib is None:
        raise ValueError("Building development bundles requires Python 3.11 or later")
    tree = ast.parse(fallback.decode("utf-8"))
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Source must declare one literal __version__")
    version = values[0]
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.+-]*)", version):
        raise ValueError("Version must be a portable release identifier")
    project = tomllib.loads(pyproject.decode("utf-8")).get("project", {})
    if not isinstance(project, dict) or project.get("version") != version:
        raise ValueError("Source and [project] version declarations differ")
    return version


def _blob_digest(data: bytes, width: int) -> str:
    digest = hashlib.sha1() if width == 40 else hashlib.sha256()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _read_blob(process, blob: str, size: int) -> bytes:
    process.stdin.write(blob.encode("ascii") + b"\n")
    process.stdin.flush()
    header = process.stdout.readline().decode("ascii").strip().split()
    if header != [blob, "blob", str(size)]:
        raise ValueError(f"Unexpected Git object response: {header}")
    data = process.stdout.read(size)
    if len(data) != size or process.stdout.read(1) != b"\n":
        raise ValueError("Incomplete Git object response")
    if _blob_digest(data, len(blob)) != blob:
        raise ValueError("Git blob hash mismatch")
    return data


def _write_member(archive, prefix, name, data, mode, timestamp):
    info = zipfile.ZipInfo(prefix + name, timestamp)
    info.create_system = 3
    info.external_attr = int(mode, 8) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _verify_archive(path: Path, manifest: dict, prefix: str) -> None:
    expected = {
        entry["path"]: entry for entry in manifest["included"] + manifest["generated"]
    }
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(expected) + 1 or set(names) != {
            *(prefix + name for name in expected),
            prefix + "bundle/manifest.json",
        }:
            raise ValueError("Bundle inventory differs from its manifest")
        for name, entry in expected.items():
            data = archive.read(prefix + name)
            if (
                len(data) != entry["bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"Bundle payload verification failed: {name}")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _admit_destination(temporary: Path, target: Path) -> None:
    if target.is_symlink():
        raise ValueError(f"Artifact destination is a symlink: {target}")
    if target.exists():
        if target.stat().st_size != temporary.stat().st_size:
            raise ValueError(f"Existing artifact differs: {target}")
        if _file_hash(target) != _file_hash(temporary):
            raise ValueError(f"Existing artifact differs: {target}")


def _publish_file(temporary: Path, target: Path) -> None:
    """Retain an existing identical artifact; never replace differing bytes."""
    _admit_destination(temporary, target)
    if target.exists():
        return
    os.link(temporary, target)


def build_bundle(
    output_dir: Path,
    *,
    root: Path = REPO_ROOT,
    revision: str = "HEAD",
    expected_tag: str | None = None,
) -> dict:
    """Build and verify exact committed sources plus explicit packaging aids."""
    commit = _commit(root, revision)
    inventory = _inventory(root, commit)
    owners = {
        name: _git(root, "show", f"{commit}:{name}")
        for name in (
            "src/tnfr/_version.py",
            "pyproject.toml",
            "AGENTS.md",
            ".github/agents/my-agent.md",
            TEMPLATE,
        )
    }
    version = _version(owners["src/tnfr/_version.py"], owners["pyproject.toml"])
    if owners["AGENTS.md"] != owners[".github/agents/my-agent.md"]:
        raise ValueError("Selected commit's agent instructions differ")
    if expected_tag is not None:
        if (
            expected_tag != f"v{version}"
            or _commit(root, f"refs/tags/{expected_tag}") != commit
        ):
            raise ValueError("Release tag must identify this version and exact commit")
    stem = f"TNFR-{version}-light-development"
    prefix = stem + "/"
    epoch = int(_git(root, "show", "-s", "--format=%ct", commit).strip())
    timestamp = datetime.fromtimestamp(max(epoch, 315532800), timezone.utc).timetuple()[
        :6
    ]
    manifest = {
        "schema": 1,
        "version": version,
        "commit": commit,
        "repository": REPOSITORY,
        "release": f"https://github.com/{REPOSITORY}/releases/tag/v{version}",
        "selection": "All committed regular files except docs/assets research evidence; mathjax.js retained.",
        "git_history_included": False,
        "included": [],
        "omitted": [],
        "generated": [],
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{stem}.zip"
    checksum_path = output_dir / f"{stem}.zip.sha256"
    with tempfile.TemporaryDirectory(
        prefix=".tnfr-bundle-", dir=output_dir
    ) as temporary:
        temporary_archive = Path(temporary) / archive_path.name
        with zipfile.ZipFile(temporary_archive, "w") as archive:
            process = subprocess.Popen(
                ["git", "--no-replace-objects", "-C", str(root), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            try:
                for name, (mode, blob, size) in sorted(inventory.items()):
                    data = _read_blob(process, blob, size)
                    entry = {
                        "path": name,
                        "bytes": size,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "git_blob": blob,
                        "git_mode": mode,
                    }
                    if (
                        name.startswith("docs/assets/")
                        and name != "docs/assets/mathjax.js"
                    ):
                        entry["download_url"] = (
                            f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/"
                            + quote(name, safe="/")
                        )
                        manifest["omitted"].append(entry)
                    else:
                        manifest["included"].append(entry)
                        _write_member(archive, prefix, name, data, mode, timestamp)
            finally:
                process.stdin.close()
                process.stdout.close()
                process.wait()
            if process.returncode:
                raise ValueError("Git object reader failed")
            onboarding = (
                Template(owners[TEMPLATE].decode("utf-8"))
                .substitute(
                    version=version,
                    commit=commit,
                    repository=REPOSITORY,
                    omitted_bytes=sum(entry["bytes"] for entry in manifest["omitted"]),
                )
                .encode("utf-8")
            )
            manifest["generated"].append(
                {
                    "path": "START_HERE.md",
                    "bytes": len(onboarding),
                    "sha256": hashlib.sha256(onboarding).hexdigest(),
                }
            )
            manifest["summary"] = {
                category: {
                    "files": len(manifest[category]),
                    "bytes": sum(entry["bytes"] for entry in manifest[category]),
                }
                for category in ("included", "omitted", "generated")
            }
            validate_manifest(manifest)
            _write_member(
                archive, prefix, "START_HERE.md", onboarding, "100644", timestamp
            )
            _write_member(
                archive,
                prefix,
                "bundle/manifest.json",
                (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                ),
                "100644",
                timestamp,
            )
        _verify_archive(temporary_archive, manifest, prefix)
        digest = _file_hash(temporary_archive)
        temporary_checksum = Path(temporary) / checksum_path.name
        temporary_checksum.write_bytes(
            f"{digest}  {archive_path.name}\n".encode("ascii")
        )
        # Admit both existing destinations before exposing either new file.
        for source, target in (
            (temporary_archive, archive_path),
            (temporary_checksum, checksum_path),
        ):
            _admit_destination(source, target)
        _publish_file(temporary_archive, archive_path)
        _publish_file(temporary_checksum, checksum_path)
    return {
        "version": version,
        "commit": commit,
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": digest,
        "checksum": str(checksum_path),
        "summary": manifest["summary"],
        "payload_verified": True,
    }


def main(argv=None) -> int:
    """Build the requested revision without changing the selected repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/development-bundle")
    )
    parser.add_argument("--expected-tag")
    args = parser.parse_args(argv)
    try:
        report = build_bundle(
            args.output_dir, revision=args.revision, expected_tag=args.expected_tag
        )
    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
