from pathlib import Path
import pytest

from tnfr.io import safe_write


def test_safe_write_atomic(tmp_path: Path):
    dest = tmp_path / "out.txt"
    safe_write(dest, lambda f: f.write("hi"))
    assert dest.read_text() == "hi"


def test_safe_write_cleans_temp_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dest = tmp_path / "out.txt"

    def fake_replace(self, target):  # pragma: no cover - monkeypatch helper
        raise OSError("boom")

    monkeypatch.setattr(Path, "replace", fake_replace)

    with pytest.raises(OSError):
        safe_write(dest, lambda f: f.write("data"))

    assert not dest.exists()
    # Only the temporary directory itself should remain
    assert list(tmp_path.iterdir()) == []

