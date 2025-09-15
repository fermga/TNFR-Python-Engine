from pathlib import Path
import os
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


def test_safe_write_sync_non_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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
