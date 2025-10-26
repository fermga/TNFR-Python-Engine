import builtins
import os
from pathlib import Path

import pytest

from tnfr.io import safe_write


def test_safe_write_atomic(tmp_path: Path):
    dest = tmp_path / "out.txt"
    safe_write(dest, lambda f: f.write("hi"))
    assert dest.read_text() == "hi"


def test_safe_write_cleans_temp_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "out.txt"

    def fake_replace(src, dst):  # pragma: no cover - monkeypatch helper
        raise OSError("boom")

    monkeypatch.setattr("os.replace", fake_replace)

    with pytest.raises(OSError):
        safe_write(dest, lambda f: f.write("data"))

    assert not dest.exists()
    # Only the temporary directory itself should remain
    assert list(tmp_path.iterdir()) == []


def test_safe_write_preserves_exception(tmp_path: Path):
    dest = tmp_path / "out.txt"

    def writer(_f):  # pragma: no cover - executed in safe_write
        raise ValueError("bad value")

    with pytest.raises(ValueError):
        safe_write(dest, writer)


def test_safe_write_non_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "out.txt"

    def fake_fsync(_fd):  # pragma: no cover - monkeypatch helper
        raise AssertionError("fsync should not be called")

    def fake_replace(_src, _dst):  # pragma: no cover - monkeypatch helper
        raise AssertionError("replace should not be called")

    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "replace", fake_replace)

    safe_write(dest, lambda f: f.write("hi"), atomic=False)

    assert dest.read_text() == "hi"


def test_safe_write_sync_non_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "out.txt"

    fsynced = False

    def fake_fsync(_fd):  # pragma: no cover - monkeypatch helper
        nonlocal fsynced
        fsynced = True

    def fake_replace(_src, _dst):  # pragma: no cover - monkeypatch helper
        raise AssertionError("replace should not be called")

    monkeypatch.setattr(os, "fsync", fake_fsync)
    monkeypatch.setattr(os, "replace", fake_replace)

    safe_write(dest, lambda f: f.write("hi"), atomic=False, sync=True)

    assert fsynced
    assert dest.read_text() == "hi"


@pytest.mark.parametrize("atomic", [True, False])
def test_safe_write_binary_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atomic: bool):
    dest = tmp_path / ("binary_atomic.bin" if atomic else "binary_direct.bin")
    payload = b"\x00TNFR\xff\x10"
    original_open = builtins.open
    binary_open_seen = False

    def fake_open(file, *args, **kwargs):  # pragma: no cover - monkeypatch helper
        nonlocal binary_open_seen
        mode = kwargs.get("mode", args[0] if args else "r")
        if isinstance(mode, str) and "b" in mode:
            assert "encoding" not in kwargs
            binary_open_seen = True
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr("tnfr.io.open", fake_open, raising=False)

    safe_write(dest, lambda f: f.write(payload), mode="wb", atomic=atomic)

    assert binary_open_seen
    assert dest.read_bytes() == payload
    assert {p.name for p in tmp_path.iterdir()} == {dest.name}
