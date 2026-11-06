"""Inspect completed TNFR distributions against the current source payload.

This publication policy checks archive contents and review budgets without
extracting or executing them. It does not validate scientific claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
PACKAGE_REQUIRED = {"__init__.py", "py.typed", "schemas/grammar.json"}
SOURCE_METADATA = {
    "MANIFEST.in",
    "pyproject.toml",
    "README.md",
    "LICENSE.md",
    "CITATION.cff",
    "setup.cfg",
    "PKG-INFO",
}
EGG_METADATA = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
    "zip-safe",
    "not-zip-safe",
}
WHEEL_METADATA = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"}
FORBIDDEN_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "tests",
    "docs",
    "assets",
    "logs",
    "archive",
    "archives",
}


@dataclass(frozen=True)
class ReviewLimits:
    """Explicit packaging review budgets, in bytes."""

    compressed: int = 10 * MIB
    expanded: int = 32 * MIB
    member: int = MIB

    def __post_init__(self) -> None:
        for name in ("compressed", "expanded", "member"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} review budget must be a positive integer")


def _canonical_name(name: str, *, directory: bool = False) -> str:
    if directory and name.endswith("/"):
        name = name[:-1]
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(part in {"", ".", ".."} for part in parts)
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"Noncanonical archive path: {name!r}")
    return name


def _package_category(name: str) -> str:
    parts = name.split("/")
    if any(
        part.lower() in FORBIDDEN_DIRECTORIES or part.startswith(".") for part in parts
    ):
        raise ValueError(f"Forbidden package path: {name}")
    if name.endswith(".py"):
        return "python"
    if name.endswith(".pyi"):
        return "stubs"
    if name == "py.typed":
        return "typing_marker"
    if len(parts) == 2 and parts[0] == "schemas" and parts[1].endswith(".json"):
        return "schemas"
    raise ValueError(f"Forbidden package payload: {name}")


def _source_inventory(source_root: Path) -> dict[str, str]:
    package = source_root / "src" / "tnfr"
    candidates = {
        *package.rglob("*.py"),
        *package.rglob("*.pyi"),
        *package.glob("py.typed"),
        *package.glob("schemas/*.json"),
    }
    inventory = {}
    for path in sorted(candidates):
        relative = path.relative_to(package).as_posix()
        _package_category(relative)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"Source payload cannot use a symbolic link: {path}")
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = PACKAGE_REQUIRED - inventory.keys()
    if missing:
        raise ValueError(f"Source payload is missing required files: {sorted(missing)}")
    return inventory


def _archive_identity(path: Path) -> tuple[str, str]:
    if path.name.endswith(".whl"):
        fields = path.name[:-4].split("-")
        if len(fields) not in {5, 6} or fields[0] != "tnfr":
            raise ValueError(f"Unexpected wheel name: {path.name}")
        return "wheel", fields[1]
    if path.name.startswith("tnfr-") and path.name.endswith(".tar.gz"):
        return "sdist", path.name[len("tnfr-") : -len(".tar.gz")]
    raise ValueError(f"Unsupported distribution: {path.name}")


def _classify_member(name: str, kind: str, version: str) -> tuple[str, str | None]:
    if kind == "wheel":
        if name.startswith("tnfr/"):
            package_name = name[len("tnfr/") :]
            return _package_category(package_name), package_name
        metadata = f"tnfr-{version}.dist-info/"
        if name.startswith(metadata):
            relative = name[len(metadata) :]
            if relative in WHEEL_METADATA or relative == "licenses/LICENSE.md":
                return "metadata", None
    else:
        root = f"tnfr-{version}/"
        if name.startswith(root):
            relative = name[len(root) :]
            if relative.startswith("src/tnfr/"):
                package_name = relative[len("src/tnfr/") :]
                return _package_category(package_name), package_name
            if relative in SOURCE_METADATA or relative == "scripts/tnfr_is_prime.py":
                return "metadata", None
            prefix = "src/tnfr.egg-info/"
            if relative.startswith(prefix) and relative[len(prefix) :] in EGG_METADATA:
                return "metadata", None
    raise ValueError(f"Forbidden distribution member: {name}")


def _required_metadata(kind: str, version: str) -> set[str]:
    if kind == "wheel":
        root = f"tnfr-{version}.dist-info/"
        return {
            root + name
            for name in ("METADATA", "WHEEL", "RECORD", "licenses/LICENSE.md")
        }
    root = f"tnfr-{version}/"
    return {
        *(root + name for name in SOURCE_METADATA),
        root + "scripts/tnfr_is_prime.py",
        root + "src/tnfr.egg-info/SOURCES.txt",
    }


def inspect_distribution(
    path: Path,
    source_inventory: dict[str, str],
    limits: ReviewLimits,
) -> dict[str, Any]:
    """Validate one archive; source inventory maps package paths to SHA256."""
    kind, version = _archive_identity(path)
    compressed = path.stat().st_size
    if compressed > limits.compressed:
        raise ValueError(f"{path.name}: compressed size exceeds review budget")
    seen: set[str] = set()
    files: set[str] = set()
    directories: set[str] = set()
    package_files: set[str] = set()
    sizes: dict[str, int] = {}
    categories = {
        name: {"files": 0, "bytes": 0}
        for name in ("python", "stubs", "schemas", "typing_marker", "metadata")
    }
    expanded = 0

    def admit(name: str, size: int, directory: bool) -> tuple[str, str | None] | None:
        nonlocal expanded
        name = _canonical_name(name, directory=directory)
        key = name.casefold()
        if key in seen:
            raise ValueError(f"{path.name}: duplicate archive path: {name}")
        seen.add(key)
        if directory:
            if size != 0:
                raise ValueError(f"{path.name}: directory contains payload: {name}")
            directories.add(name)
            return None
        if size < 0 or size > limits.member:
            raise ValueError(f"{path.name}: member exceeds review budget: {name}")
        expanded += size
        if expanded > limits.expanded:
            raise ValueError(f"{path.name}: expanded size exceeds review budget")
        category, package_name = _classify_member(name, kind, version)
        if package_name is not None and package_name not in source_inventory:
            raise ValueError(f"{path.name}: unexpected package source: {package_name}")
        files.add(name)
        sizes[name] = size
        categories[category]["files"] += 1
        categories[category]["bytes"] += size
        return category, package_name

    def check_content(name: str, data: bytes, package_name: str | None) -> None:
        if len(data) != sizes[name]:
            raise ValueError(f"{path.name}: incomplete member: {name}")
        if package_name is not None:
            if hashlib.sha256(data).hexdigest() != source_inventory[package_name]:
                raise ValueError(f"{path.name}: source content differs: {package_name}")
            package_files.add(package_name)
        elif b"\x00" in data:
            raise ValueError(f"{path.name}: binary content in metadata: {name}")
        else:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path.name}: non-UTF-8 metadata: {name}") from exc

    if kind == "wheel":
        with zipfile.ZipFile(path) as zip_archive:
            for zip_member in zip_archive.infolist():
                # ZipInfo may normalize separators or truncate at NUL while
                # reading. Admit the original name before using its projection.
                member_name = zip_member.orig_filename
                directory = zip_member.is_dir()
                mode = stat.S_IFMT(zip_member.external_attr >> 16)
                if mode not in ({0, stat.S_IFDIR} if directory else {0, stat.S_IFREG}):
                    raise ValueError(f"{path.name}: nonregular member: {member_name}")
                admitted = admit(member_name, zip_member.file_size, directory)
                if admitted is not None:
                    check_content(
                        member_name, zip_archive.read(zip_member), admitted[1]
                    )
    else:
        with tarfile.open(path, mode="r|gz") as tar_archive:
            for tar_member in tar_archive:
                if tar_member.type not in {
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                    tarfile.DIRTYPE,
                }:
                    raise ValueError(
                        f"{path.name}: nonregular member: {tar_member.name}"
                    )
                admitted = admit(tar_member.name, tar_member.size, tar_member.isdir())
                if admitted is not None:
                    stream = tar_archive.extractfile(tar_member)
                    if stream is None:
                        raise ValueError(
                            f"{path.name}: missing member content: {tar_member.name}"
                        )
                    with stream:
                        check_content(tar_member.name, stream.read(), admitted[1])

    missing = source_inventory.keys() - package_files
    if missing:
        raise ValueError(f"{path.name}: missing package source: {sorted(missing)}")
    missing_metadata = _required_metadata(kind, version) - files
    if missing_metadata:
        raise ValueError(f"{path.name}: missing metadata: {sorted(missing_metadata)}")
    for directory_name in directories:
        if not any(name.startswith(directory_name + "/") for name in files):
            raise ValueError(f"{path.name}: extraneous directory: {directory_name}")
    return {
        "archive": path.name,
        "kind": kind,
        "version": version,
        "compressed_bytes": compressed,
        "expanded_bytes": expanded,
        "files": len(files),
        "categories": categories,
        "largest_files": [
            {"path": name, "bytes": size}
            for name, size in sorted(
                sizes.items(), key=lambda item: (-item[1], item[0])
            )[:5]
        ],
    }


def check_distributions(
    directory: Path,
    *,
    source_root: Path = REPO_ROOT,
    limits: ReviewLimits = ReviewLimits(),
) -> dict[str, Any]:
    """Require a wheel and sdist for one version, both matching source."""
    if not directory.is_dir():
        raise ValueError(f"Distribution directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
    )
    identities = [_archive_identity(path) for path in paths]
    if {kind for kind, _ in identities} != {"wheel", "sdist"}:
        raise ValueError("Distribution review requires both a wheel and an sdist")
    if len({version for _, version in identities}) != 1:
        raise ValueError("Distribution directory contains multiple package versions")
    inventory = _source_inventory(source_root)
    reports = [inspect_distribution(path, inventory, limits) for path in paths]
    return {
        "scope": "distribution_payload_and_review_budgets",
        "source_root": str(source_root.resolve()),
        "source_files": len(inventory),
        "limits_bytes": {
            "compressed": limits.compressed,
            "expanded": limits.expanded,
            "member": limits.member,
        },
        "compressed_bytes": sum(report["compressed_bytes"] for report in reports),
        "expanded_bytes": sum(report["expanded_bytes"] for report in reports),
        "archives": reports,
    }


def _mib_budget(value: str) -> int:
    try:
        mib = float(value)
        if not math.isfinite(mib) or mib <= 0 or mib * MIB < 1:
            raise ValueError
        return int(mib * MIB)
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "budget must be finite and at least one byte"
        ) from exc


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"Distribution review passed: {len(report['archives'])} archives, "
        f"{report['compressed_bytes'] / MIB:.2f} MiB compressed, "
        f"{report['expanded_bytes'] / MIB:.2f} MiB expanded."
    )
    for archive in report["archives"]:
        print(
            f"{archive['archive']}: {archive['compressed_bytes'] / MIB:.2f} MiB compressed, "
            f"{archive['expanded_bytes'] / MIB:.2f} MiB expanded, {archive['files']} files"
        )
        for category in ("python", "stubs", "schemas", "metadata"):
            values = archive["categories"][category]
            print(f"  {category}: {values['files']} files, {values['bytes']} bytes")
        for member in archive["largest_files"]:
            print(f"  largest: {member['bytes']} bytes {member['path']}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-compressed-mib", type=_mib_budget, default=10 * MIB)
    parser.add_argument("--max-expanded-mib", type=_mib_budget, default=32 * MIB)
    parser.add_argument("--max-member-mib", type=_mib_budget, default=MIB)
    parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON to stdout"
    )
    args = parser.parse_args(argv)
    limits = ReviewLimits(
        compressed=args.max_compressed_mib,
        expanded=args.max_expanded_mib,
        member=args.max_member_mib,
    )
    try:
        report = check_distributions(
            args.directory, source_root=args.source_root, limits=limits
        )
    except (
        OSError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
        RuntimeError,
    ) as exc:
        parser.exit(1, f"Distribution review failed: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
