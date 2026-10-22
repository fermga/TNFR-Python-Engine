"""Verify a light development bundle or restore its omitted release evidence.

This packaging-bootstrap helper uses only the standard library. Schema 1 review
limits bound input materialization; increase them only after reviewing new
release contents. Hashes establish recorded byte equality, not authenticity or
scientific validity. Restoration never executes or recursively extracts evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "bundle" / "manifest.json"
REPOSITORY = "fermga/TNFR-Python-Engine"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 20_000
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_DEVICES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}"
    for prefix in ("COM", "LPT")
    for number in (*range(1, 10), "¹", "²", "³")
}


def portable_parts(name):
    """Admit a normalized, portable relative path without touching the disk."""
    if type(name) is not str:
        raise ValueError("Path must be a string")
    parts = name.split("/")
    if (
        not name
        or len(name) > 1024
        or unicodedata.normalize("NFC", name) != name
        or any(unicodedata.category(character) == "Cs" for character in name)
        or any(character in name for character in '\\:<>?"|*')
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or any(part in {"", ".", ".."} for part in parts)
        or any(len(part) > 255 or part.endswith((".", " ")) for part in parts)
        or any(part.split(".")[0].upper() in _DEVICES for part in parts)
    ):
        raise ValueError(f"Unsafe path: {name!r}")
    return parts


def canonical_name(name):
    """Return an admitted portable relative name without accessing the disk."""
    portable_parts(name)
    return name


def check_path_collisions(names):
    """Reject case collisions in files or implicit directories and file parents."""
    seen = {}
    files = set()
    for name in names:
        parts = portable_parts(name)
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            folded = prefix.casefold()
            if folded in seen and seen[folded] != prefix:
                raise ValueError(f"Case-colliding path: {name}")
            if index < len(parts) and folded in files:
                raise ValueError(f"File is also a directory: {name}")
            if index == len(parts) and folded in seen:
                raise ValueError(f"Duplicate or overlapping path: {name}")
            seen[folded] = prefix
        files.add(name.casefold())


def _reject_link(path):
    """Reject symbolic links and Windows junction/reparse destinations."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(status.st_mode) or (
        getattr(status, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise ValueError(f"Link destination is not allowed: {path.name}")


def destination(name):
    """Resolve beneath the bundle, refusing links and existing case aliases."""
    parts = portable_parts(name)
    target = ROOT
    _reject_link(target)
    for part in parts:
        if target.is_dir():
            for child in target.iterdir():
                if child.name.casefold() == part.casefold() and child.name != part:
                    raise ValueError(f"Existing case-colliding destination: {name}")
        target = target / part
        _reject_link(target)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Destination escapes bundle: {name}")
    return target


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def _hex(value, lengths):
    return (
        type(value) is str
        and len(value) in lengths
        and re.fullmatch("[0-9a-f]+", value) is not None
    )


def load_manifest():
    """Read a bounded, strict manifest without requiring an installed engine."""
    manifest_path = destination(MANIFEST.relative_to(ROOT).as_posix())
    with manifest_path.open("rb") as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds the schema 1 review limit")
    try:
        manifest = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid manifest encoding or nesting") from error
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Admit all consumed fields before using any manifest-controlled path."""
    if type(manifest) is not dict or not {
        "schema",
        "version",
        "commit",
        "repository",
        "included",
        "omitted",
        "generated",
    }.issubset(manifest):
        raise ValueError("Unsupported bundle manifest fields")
    if (
        type(manifest["schema"]) is not int
        or manifest["schema"] != 1
        or type(manifest["version"]) is not str
        or re.fullmatch(r"[0-9][0-9A-Za-z.+!-]{0,127}", manifest["version"]) is None
        or not _hex(manifest["commit"], (40, 64))
        or manifest["repository"] != REPOSITORY
    ):
        raise ValueError("Unsupported bundle manifest metadata")
    names = []
    total = 0
    for group in ("included", "omitted", "generated"):
        entries = manifest[group]
        if type(entries) is not list or len(entries) > MAX_ENTRIES:
            raise ValueError(f"Invalid manifest group: {group}")
        for entry in entries:
            fields = {"path", "bytes", "sha256"}
            if group != "generated":
                fields |= {"git_blob", "git_mode"}
            optional = {"download_url"} if group == "omitted" else set()
            if (
                type(entry) is not dict
                or not fields.issubset(entry)
                or set(entry) - fields - optional
            ):
                raise ValueError(f"Invalid manifest entry fields in {group}")
            if "download_url" in entry and (
                type(entry["download_url"]) is not str
                or len(entry["download_url"]) > 4096
            ):
                raise ValueError("Invalid informational download URL")
            name = entry["path"]
            portable_parts(name)
            if name == "bundle/manifest.json":
                raise ValueError("The manifest cannot certify itself")
            if group == "omitted" and (
                not name.startswith("docs/assets/") or name == "docs/assets/mathjax.js"
            ):
                raise ValueError(f"Unexpected omitted evidence path: {name}")
            if (
                type(entry["bytes"]) is not int
                or not 0 <= entry["bytes"] <= MAX_FILE_BYTES
                or not _hex(entry["sha256"], (64,))
            ):
                raise ValueError(f"Invalid size/hash: {name}")
            if group != "generated" and (
                not _hex(entry["git_blob"], (len(manifest["commit"]),))
                or type(entry["git_mode"]) is not str
                or entry["git_mode"] not in {"100644", "100755"}
            ):
                raise ValueError(f"Invalid Git association: {name}")
            total += entry["bytes"]
            names.append(name)
    if len(names) > MAX_ENTRIES or total > MAX_TOTAL_BYTES:
        raise ValueError("Manifest exceeds the schema 1 review limit")
    check_path_collisions(names)


def check_bytes(data, entry):
    """Check the byte count and SHA256 before any installation."""
    if len(data) != entry["bytes"]:
        raise ValueError(f"Size mismatch: {entry['path']}")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch: {entry['path']}")


def check_file(target, entry):
    _reject_link(target)
    if not target.is_file():
        raise ValueError(f"Missing regular file: {entry['path']}")
    with target.open("rb") as stream:
        check_bytes(stream.read(entry["bytes"] + 1), entry)


def verify(manifest, require_all=False):
    """Verify all supplied files and any evidence already restored."""
    validate_manifest(manifest)
    checked = 0
    absent = 0
    for group in ("included", "generated", "omitted"):
        for entry in manifest[group]:
            target = destination(entry["path"])
            if group == "omitted" and not target.exists():
                absent += 1
                continue
            check_file(target, entry)
            checked += 1
    print(f"Verified files: {checked}; omitted evidence still absent: {absent}")
    if require_all and absent:
        raise ValueError("Full evidence requested but some evidence is absent")
    print("Hashes check recorded bytes, not scientific validity or authenticity.")
    return checked, absent


def select_entries(manifest, prefix, all_evidence):
    validate_manifest(manifest)
    if all_evidence:
        return manifest["omitted"]
    if type(prefix) is not str:
        raise ValueError("An omitted evidence path is required")
    prefix = prefix.rstrip("/")
    portable_parts(prefix)
    entries = [
        entry
        for entry in manifest["omitted"]
        if entry["path"] == prefix or entry["path"].startswith(prefix + "/")
    ]
    if not entries:
        raise ValueError(f"No omitted evidence matches: {prefix}")
    return entries


def _archive_header(stream):
    """Bound ZIP directory parsing before ZipFile allocates its index."""
    size = stream.seek(0, os.SEEK_END)
    if size > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds the schema 1 review limit")
    stream.seek(max(0, size - 65557))
    tail = stream.read(65557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < 22:
        raise ValueError("Missing ZIP end record")
    _, disk, start_disk, disk_count, count, directory_size, start, comment = (
        struct.unpack_from("<4s4H2LH", tail, offset)
    )
    if (
        offset + 22 + comment != len(tail)
        or disk != 0
        or start_disk != 0
        or count != disk_count
        or count == 0xFFFF
        or count > MAX_ENTRIES
        or directory_size > MAX_CENTRAL_DIRECTORY_BYTES
        or start + directory_size > size - len(tail) + offset
    ):
        raise ValueError("Unsupported or oversized ZIP directory (ZIP64 excluded)")
    stream.seek(0)


def _archive_index(archive):
    if len(archive.infolist()) > MAX_ENTRIES:
        raise ValueError("Archive exceeds the schema 1 entry-count limit")
    root = None
    index = {}
    names = {}
    total = 0
    for info in archive.infolist():
        if info.orig_filename != info.filename:
            raise ValueError("Archive member name was implicitly normalized")
        is_directory = info.is_dir()
        name = info.filename[:-1] if is_directory else info.filename
        parts = portable_parts(name)
        if root is None:
            root = parts[0]
        if root != parts[0] or (len(parts) < 2 and not is_directory):
            raise ValueError("Expected one repository ZIP root")
        folded = name.casefold()
        if folded in names:
            raise ValueError(f"Duplicate or case-colliding archive path: {name}")
        names[folded] = name
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (
            kind not in {0, stat.S_IFDIR if is_directory else stat.S_IFREG}
            or info.flag_bits & 1
            or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or info.file_size > MAX_FILE_BYTES
            or (is_directory and info.file_size != 0)
        ):
            raise ValueError(f"Unsupported archive member: {name}")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Archive exceeds the schema 1 expanded-size limit")
        if not is_directory:
            index["/".join(parts[1:])] = info
    # Include explicit directory aliases when checking implicit directory spelling.
    for name in list(names.values()):
        parts = name.split("/")
        for offset in range(1, len(parts)):
            prefix = "/".join(parts[:offset])
            previous = names.get(prefix.casefold())
            if previous is not None and previous != prefix:
                raise ValueError(f"Case-colliding archive directory: {prefix}")
            names.setdefault(prefix.casefold(), prefix)
    check_path_collisions([f"{root}/{name}" for name in index])
    return index


def _install(data, entry):
    target = destination(entry["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    destination(entry["path"])
    descriptor, temporary = tempfile.mkstemp(prefix=".tnfr-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        destination(entry["path"])
        try:
            os.link(temporary, target)
        except FileExistsError:
            check_file(destination(entry["path"]), entry)
            return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore(manifest, entries, archive_path=None):
    """Restore selected bytes without overwrites; completed files survive failure."""
    validate_manifest(manifest)
    approved = {entry["path"]: entry for entry in manifest["omitted"]}
    if any(
        type(entry) is not dict or approved.get(entry.get("path")) != entry
        for entry in entries
    ):
        raise ValueError("Restoration requires manifest-admitted omitted entries")
    entries = [approved[entry["path"]] for entry in entries]
    archive = None
    archive_stream = None
    archive_index = {}
    restored = 0
    existing = 0
    try:
        if archive_path is not None:
            archive_stream = Path(archive_path).open("rb")
            _archive_header(archive_stream)
            archive = zipfile.ZipFile(archive_stream)
            archive_index = _archive_index(archive)
        for entry in entries:
            target = destination(entry["path"])
            if target.exists():
                check_file(target, entry)
                existing += 1
                continue
            if archive is not None:
                info = archive_index.get(entry["path"])
                if info is None or info.file_size != entry["bytes"]:
                    raise ValueError(f"Missing/wrong archive member: {entry['path']}")
                with archive.open(info) as stream:
                    data = stream.read(entry["bytes"] + 1)
            else:
                url = (
                    f"https://raw.githubusercontent.com/{REPOSITORY}/"
                    f"{manifest['commit']}/{quote(entry['path'], safe='/')}"
                )
                request = Request(
                    url, headers={"User-Agent": "TNFR-development-evidence-recovery"}
                )
                with urlopen(request, timeout=60) as response:
                    # Do not follow a server redirect into an unrelated source.
                    if response.geturl() != url:
                        raise ValueError("Evidence source redirected from pinned URL")
                    data = response.read(entry["bytes"] + 1)
            check_bytes(data, entry)
            if _install(data, entry):
                restored += 1
                print(f"Restored {entry['path']}", flush=True)
            else:
                existing += 1
    finally:
        if archive is not None:
            archive.close()
        if archive_stream is not None:
            archive_stream.close()
    print(f"Restored: {restored}; already matching: {existing}")
    print("Restoration does not execute producers or add Git history.")
    return restored, existing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--require-all-evidence", action="store_true")
    commands.add_parser("list")
    fetch = commands.add_parser("fetch")
    selection = fetch.add_mutually_exclusive_group(required=True)
    selection.add_argument("--path", help="Exact omitted file or directory prefix")
    selection.add_argument("--all", action="store_true", dest="all_evidence")
    fetch.add_argument("--archive", type=Path, help="Existing full release ZIP")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        if args.command == "verify":
            verify(manifest, args.require_all_evidence)
        elif args.command == "list":
            for entry in manifest["omitted"]:
                print(f"{entry['bytes']:>10}  {entry['path']}")
        else:
            entries = select_entries(manifest, args.path, args.all_evidence)
            print(
                f"Selected {len(entries)} files; "
                f"{sum(entry['bytes'] for entry in entries)} bytes"
            )
            restore(manifest, entries, args.archive)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
