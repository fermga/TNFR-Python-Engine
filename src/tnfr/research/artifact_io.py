"""Shared exact records and immutable research-artifact byte associations.

These operations establish serialization and content consistency only. They
neither admit a model or observation nor authenticate acquisition, chronology
or execution. Callers retain their own scientific schemas and size policies.
"""

from __future__ import annotations

import hashlib
import math
import re
import stat
import zipfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..utils.io import json_dumps, safe_write

__all__ = (
    "exact_record",
    "encode_exact_tree",
    "decode_exact_tree",
    "read_bytes_bounded",
    "sha256_bytes",
    "sha256_file",
    "file_receipt",
    "validate_artifact_path",
    "verify_archive_members",
    "write_json_once",
)

_CHUNK_SIZE = 1 << 16
_DEFAULT_ARCHIVE_LIMIT = 128 * 1024**2
_DEVICES = frozenset(("con", "prn", "aux", "nul", "conin$", "conout$")) | {
    f"{prefix}{suffix}" for prefix in ("com", "lpt") for suffix in "123456789¹²³"
}


def _require(condition, message):
    """Shared ValueError boundary retained by exact-record audit consumers."""
    if not condition:
        raise ValueError(message)


def exact_record(value):
    """Admit an ordinary int/Fraction or exactly tagged integer numerator/denominator.

    Booleans, decimal strings, floats and nonpositive denominators are invalid.
    Fraction magnitude is unrestricted and no float conversion is performed.
    """
    if isinstance(value, Mapping):
        _require(set(value) == {"numerator", "denominator"}, "invalid fraction record")
        numerator, denominator = value["numerator"], value["denominator"]
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
        ):
            raise ValueError(
                "fraction fields require integers and a positive denominator"
            )
        return Fraction(numerator, denominator)
    if type(value) not in (int, Fraction):
        raise ValueError("exact record requires an integer or fraction")
    return Fraction(value)


def _tree_scalar(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("record tree requires finite JSON metadata or exact fractions")


def encode_exact_tree(value):
    """Detach a finite JSON tree, spelling Fraction values with the shared tag.

    Metadata booleans remain booleans. Physical scalar admission belongs to
    the consuming schema; this projection is not such admission.
    """
    if type(value) is Fraction:
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("record field names must be strings")
        return {key: encode_exact_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_exact_tree(item) for item in value]
    return _tree_scalar(value)


def decode_exact_tree(value):
    """Decode exact two-key fraction tags and detach JSON arrays as tuples.

    Other mappings remain ordinary metadata; consuming an exact coordinate
    still requires ``exact_record``. Read serialized input with ``json_loads``
    first so duplicate keys and invalid JSON numbers cannot disappear here.
    """
    if isinstance(value, Mapping):
        if set(value) == {"numerator", "denominator"}:
            return exact_record(value)
        if any(type(key) is not str for key in value):
            raise TypeError("record field names must be strings")
        return {key: decode_exact_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(decode_exact_tree(item) for item in value)
    if type(value) is Fraction:
        return value
    return _tree_scalar(value)


def _limit(value, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("byte limit must be a nonnegative ordinary integer")
    return value


def read_bytes_bounded(path, *, max_bytes):
    """Read at most limit+1 bytes from one open file; reject growth beyond limit."""
    limit = _limit(max_bytes)
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("record exceeds audit byte budget")
    return data


def sha256_bytes(data):
    """Hash supplied bytes without any text or newline normalization."""
    return hashlib.sha256(data).hexdigest()


def file_receipt(path, *, label=None, max_bytes=None):
    """Return path/size/digest from the same streamed read, without a stat race.

    The label is descriptive, not a contained-path certificate. Consumers
    resolving declared relative paths must apply their own root boundary.
    An omitted limit preserves streaming behavior for existing large files.
    """
    limit = _limit(max_bytes, optional=True)
    path = Path(path)
    if label is None:
        label = path.as_posix()
    if not isinstance(label, str) or not label:
        raise ValueError("receipt label must be nonempty text")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            remaining = (
                _CHUNK_SIZE if limit is None else min(_CHUNK_SIZE, limit - size + 1)
            )
            chunk = stream.read(remaining)
            if not chunk:
                break
            size += len(chunk)
            if limit is not None and size > limit:
                raise ValueError("record exceeds audit byte budget")
            digest.update(chunk)
    return {"path": label, "bytes": size, "sha256": digest.hexdigest()}


def sha256_file(path, *, max_bytes=None):
    """Stream one file through the same byte-counted receipt owner."""
    return file_receipt(path, max_bytes=max_bytes)["sha256"]


def validate_artifact_path(name):
    """Admit one canonical portable relative file name without rewriting it.

    Archives are never extracted here. This rejects path traversal and common
    Windows aliases so a later contained restoration can reuse the same names.
    Filesystem symlink containment remains the restoring caller's obligation.
    """
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("artifact paths must be canonical relative POSIX names")
    if PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive:
        raise ValueError("artifact path must be relative")
    parts = name.split("/")
    for part in parts:
        if (
            part in ("", ".", "..")
            or part.casefold() == ".git"
            or part.endswith((".", " "))
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or part.split(".", 1)[0].casefold() in _DEVICES
        ):
            raise ValueError("artifact path contains an unsafe or aliased component")
    return name


def verify_archive_members(
    path_or_seekable, expected_hashes, *, max_bytes=_DEFAULT_ARCHIVE_LIMIT
):
    """Verify exact safe inventory, expanded-byte budget, encoding and SHA-256.

    Accept a path or seekable binary object, as ``ZipFile`` does; do not extract
    or import members. Casefold collisions are rejected even on case-sensitive
    hosts. Hashes establish content association, not execution authentication.
    """
    limit = _limit(max_bytes)
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError("missing source manifest")
    expected = dict(expected_hashes)
    names = tuple(validate_artifact_path(name) for name in expected)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("source archive has casefold path aliases")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in expected.values()
    ):
        raise ValueError("invalid source digest")
    with zipfile.ZipFile(path_or_seekable) as archive:
        entries = archive.infolist()
        actual = [validate_artifact_path(info.filename) for info in entries]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError("source archive inventory differs")
        if sum(info.file_size for info in entries) > limit:
            raise ValueError("expanded archive exceeds audit byte budget")
        if any(
            info.flag_bits & 1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
            for info in entries
        ):
            raise ValueError("unsupported source archive encoding or member type")
        total = 0
        for info in entries:
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as stream:
                while chunk := stream.read(min(_CHUNK_SIZE, limit - total + 1)):
                    size += len(chunk)
                    total += len(chunk)
                    if total > limit:
                        raise ValueError("expanded archive exceeds audit byte budget")
                    digest.update(chunk)
            if size != info.file_size or digest.hexdigest() != expected[info.filename]:
                raise ValueError("archived source digest or byte count differs")


def write_json_once(
    path, payload, *, sort_keys=False, indent=2, base_dir=None, newline="\n"
):
    """Serialize first, then exclusively create and sync an immutable JSON file.

    Uses shared ``safe_write`` in exclusive, non-atomic mode. Existing bytes
    are never replaced. A failed write can leave a partial file; retaining it
    deliberately blocks an unrecorded retry. This differs from SDK atomic
    replacement and does not by itself establish an experiment's first attempt.
    New artifacts use LF; ``newline=None`` preserves a legacy consumer's native
    platform newline translation where its original byte policy requires it.
    """
    text = (
        json_dumps(
            encode_exact_tree(payload),
            sort_keys=sort_keys,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    )
    safe_write(
        path,
        lambda stream: stream.write(text),
        mode="x",
        atomic=False,
        sync=True,
        newline=newline,
        base_dir=base_dir,
    )
