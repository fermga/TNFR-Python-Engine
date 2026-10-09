"""Exact records, bounded content checks and immutable file lifecycle controls."""

import hashlib
import io
import json
import os
import stat
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.research import artifact_io as owner
from tnfr.utils.io import json_loads


@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        0.5,
        float("nan"),
        "1/2",
        {"numerator": 1},
        {"numerator": 1, "denominator": 2, "extra": 0},
        {"numerator": True, "denominator": 2},
        {"numerator": 1, "denominator": False},
        {"numerator": 1, "denominator": 0},
        {"numerator": 1, "denominator": -2},
        {"numerator": 1.0, "denominator": 2},
    ),
)
def test_exact_coordinate_admission_rejects_ambiguous_records(value):
    with pytest.raises(ValueError):
        owner.exact_record(value)


def test_exact_tree_preserves_huge_tiny_signed_values_and_metadata():
    values = (Q(-(2**2000), 3), Q(1, 2**3000), Q(0), Q(-7, 11))
    data = {"values": values, "available": False, "optional": None, "represented": 0.25}
    encoded = owner.encode_exact_tree(data)
    assert encoded["values"][1] == {"numerator": 1, "denominator": 2**3000}
    assert encoded["available"] is False
    assert encoded["represented"] == 0.25
    assert owner.decode_exact_tree(json_loads(json.dumps(encoded))) == data
    assert owner.exact_record(0) == Q(0)
    assert owner.exact_record({"numerator": -14, "denominator": 22}) == Q(-7, 11)


@pytest.mark.parametrize("method", (owner.encode_exact_tree, owner.decode_exact_tree))
@pytest.mark.parametrize("value", ({1: "not a field"}, float("inf"), object()))
def test_tree_projection_rejects_unsupported_metadata(method, value):
    with pytest.raises(TypeError):
        method(value)


def test_exact_codec_does_not_replace_strict_json_reader():
    with pytest.raises(ValueError):
        owner.decode_exact_tree(
            json_loads('{"numerator":1,"numerator":2,"denominator":3}')
        )
    with pytest.raises(ValueError):
        owner.decode_exact_tree(json_loads('{"rate":NaN}'))
    with pytest.raises(ValueError):
        owner.decode_exact_tree({"numerator": False, "denominator": 1})


@pytest.mark.parametrize("limit", (True, -1, Q(10), 10.0, None))
def test_read_limit_requires_an_explicit_ordinary_integer(tmp_path, limit):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError):
        owner.read_bytes_bounded(path, max_bytes=limit)


def test_bounded_file_receipt_uses_read_bytes_not_stat(tmp_path, monkeypatch):
    path = tmp_path / "large.bin"
    content = bytes(range(256)) * 257 + b"tail"
    path.write_bytes(content)

    def no_stat(*args, **kwargs):
        pytest.fail("size must come from the same open file stream")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", no_stat)
        assert owner.read_bytes_bounded(path, max_bytes=len(content)) == content
        receipt = owner.file_receipt(
            path, label="source/data.bin", max_bytes=len(content)
        )
        assert receipt == {
            "path": "source/data.bin",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        assert owner.sha256_file(path) == receipt["sha256"]
        for read in (owner.read_bytes_bounded, owner.file_receipt, owner.sha256_file):
            with pytest.raises(ValueError, match="byte budget"):
                read(path, max_bytes=len(content) - 1)


def test_empty_file_and_exact_newline_bytes_have_distinct_receipts(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert owner.read_bytes_bounded(path, max_bytes=0) == b""
    assert owner.file_receipt(path, max_bytes=0)["bytes"] == 0
    assert owner.sha256_bytes(b"a\n") != owner.sha256_bytes(b"a\r\n")


@pytest.mark.parametrize(
    "name",
    (
        "",
        "/absolute",
        "C:/drive",
        "C:drive",
        "a\\b",
        "a//b",
        "./a",
        "a/../b",
        "a/./b",
        "a/",
        ".git/config",
        "A/.GiT/config",
        "a\x00b",
        "a\nb",
        "a:stream",
        "a/file.",
        "a/file ",
        "CON",
        "aux.txt",
        "NUL.json",
        "com1",
        "LPT9.py",
        "COM¹.txt",
        "conout$.log",
        "a?b",
        "a|b",
        "a*b",
    ),
)
def test_archive_path_admission_rejects_traversal_and_portability_aliases(name):
    with pytest.raises(ValueError):
        owner.validate_artifact_path(name)


def test_archive_paths_are_not_silently_normalized():
    for name in (
        "src/tnfr/example.py",
        "docs/.gitignore",
        "nested/readme.md",
        "model/ν.txt",
    ):
        assert owner.validate_artifact_path(name) == name


def _archive(entries, *, compression=zipfile.ZIP_DEFLATED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    stream.seek(0)
    return stream


def test_archive_verification_accepts_path_and_seekable_without_extracting(tmp_path):
    entries = (("source/a.py", b"a = 3\n"), ("protocol.json", b'{"value":1}\n'))
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in entries}
    size = sum(len(data) for _, data in entries)
    stream = _archive(entries)
    owner.verify_archive_members(stream, expected, max_bytes=size)
    assert not stream.closed
    path = tmp_path / "source.zip"
    path.write_bytes(stream.getvalue())
    owner.verify_archive_members(path, expected, max_bytes=size)
    assert set(tmp_path.iterdir()) == {path}
    with pytest.raises(ValueError, match="byte budget"):
        owner.verify_archive_members(path, expected, max_bytes=size - 1)


@pytest.mark.parametrize(
    "problem", ("extra", "missing", "duplicate", "digest", "casefold", "parent")
)
def test_archive_inventory_and_hashes_are_checked_independently(problem):
    entries = [("source.py", b"original")]
    expected = {"source.py": hashlib.sha256(b"original").hexdigest()}
    if problem == "extra":
        entries.append(("other.py", b"extra"))
    elif problem == "missing":
        expected["other.py"] = hashlib.sha256(b"extra").hexdigest()
    elif problem == "duplicate":
        entries.append(entries[0])
    elif problem == "digest":
        entries[0] = ("source.py", b"changed")
    elif problem == "casefold":
        entries.append(("SOURCE.py", b"other"))
        expected["SOURCE.py"] = hashlib.sha256(b"other").hexdigest()
    else:
        entries = [("../outside", b"original")]
        expected = {"../outside": expected["source.py"]}
    if problem == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate"):
            stream = _archive(entries)
    else:
        stream = _archive(entries)
    with pytest.raises(ValueError):
        owner.verify_archive_members(stream, expected)


def test_archive_rejects_symlink_and_unsupported_compression():
    digest = hashlib.sha256(b"target").hexdigest()
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    stream = _archive(((info, b"target"),))
    with pytest.raises(ValueError, match="member type"):
        owner.verify_archive_members(stream, {"link": digest})
    stream = _archive((("file", b"target"),), compression=zipfile.ZIP_BZIP2)
    with pytest.raises(ValueError, match="encoding"):
        owner.verify_archive_members(stream, {"file": digest})


def test_truncated_archive_is_not_an_unavailable_observation():
    data = _archive((("file", b"target"),)).getvalue()
    with pytest.raises(zipfile.BadZipFile):
        owner.verify_archive_members(
            io.BytesIO(data[:-30]), {"file": hashlib.sha256(b"target").hexdigest()}
        )


def test_exclusive_creation_serializes_before_open_and_preserves_first_bytes(tmp_path):
    path = tmp_path / "attempt.json"
    with pytest.raises(TypeError):
        owner.write_json_once(path, {"bad": object()})
    assert not path.exists()
    owner.write_json_once(path, {"value": Q(-2, 7), "attempt": 1}, sort_keys=True)
    expected = (
        json.dumps(
            {"value": {"numerator": -2, "denominator": 7}, "attempt": 1},
            sort_keys=True,
            indent=2,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert path.read_bytes() == expected.encode()
    with pytest.raises(FileExistsError):
        owner.write_json_once(path, {"attempt": 2})
    assert path.read_bytes() == expected.encode()


def test_failed_write_retains_partial_intent_and_blocks_retry(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json"
    write = owner.safe_write

    def interrupted(target, callback, **options):
        def fail(stream):
            stream.write("partial first intent")
            raise OSError("synthetic write interruption")

        return write(target, fail, **options)

    with monkeypatch.context() as patch:
        patch.setattr(owner, "safe_write", interrupted)
        with pytest.raises(OSError, match="interruption"):
            owner.write_json_once(path, {"attempt": 1})
    assert path.read_bytes() == b"partial first intent"
    with pytest.raises(FileExistsError):
        owner.write_json_once(path, {"attempt": 2})
    assert path.read_bytes() == b"partial first intent"


def test_competing_exclusive_writers_have_exactly_one_winner(tmp_path):
    target = tmp_path / "intent.json"
    start = threading.Barrier(2)

    def write(identity):
        start.wait()
        try:
            owner.write_json_once(target, {"writer": identity})
            return identity
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(write, (1, 2)))
    winners = tuple(value for value in results if value is not None)
    assert len(winners) == 1
    assert json_loads(target.read_bytes()) == {"writer": winners[0]}


def test_new_writer_reuses_shared_contained_path_boundary(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    with pytest.raises(ValueError):
        owner.write_json_once(tmp_path / "outside.json", {}, base_dir=root)
    assert not (tmp_path / "outside.json").exists()


def test_legacy_acquisition_aliases_preserve_projection_without_science(tmp_path):
    from tnfr.research import relational_acquisition as legacy

    assert legacy._exact is owner.exact_record
    expected = '{"flag":true,"value":{"denominator":7,"numerator":-2}}'
    assert legacy._encoded({"value": Q(-2, 7), "flag": True}) == expected
    data = b"known source\n"
    path = tmp_path / "source.zip"
    path.write_bytes(_archive((("source.py", data),)).getvalue())
    legacy._verify_archive(path, {"source.py": hashlib.sha256(data).hexdigest()})


def test_phase_study_uses_shared_immutable_writer_without_executing_a_study(
    tmp_path, monkeypatch
):
    from tnfr.research import phase_form_response as consumer

    monkeypatch.setattr(consumer, "_run_case", lambda *a: pytest.fail("no experiment"))
    path = tmp_path / "declaration.json"
    data = {"scope": "ordinary serialization control", "value": 0.25}
    consumer._save_new(path, data)
    assert (
        path.read_bytes()
        == (json.dumps(data, sort_keys=True, indent=2, separators=(",", ":")) + "\n")
        .replace("\n", os.linesep)
        .encode()
    )
    assert consumer._hash(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        consumer._save_new(path, {"value": 1})
