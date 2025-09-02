"""Pruebas de read structured file errors."""
import pytest
from pathlib import Path
import tnfr.helpers as helpers

from tnfr.helpers import read_structured_file


def test_read_structured_file_missing_file(tmp_path: Path):
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    assert str(path) in str(excinfo.value)


def test_read_structured_file_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "forbidden.json"
    original_open = Path.open

    def fake_open(self, *args, **kwargs):  # pragma: no cover - monkeypatch helper
        if self == path:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    assert str(path) in str(excinfo.value)


def test_read_structured_file_corrupt_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert "JSON" in msg
    assert str(path) in msg


def test_read_structured_file_corrupt_yaml(tmp_path: Path):
    pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert "YAML" in msg
    assert str(path) in msg


def test_read_structured_file_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "data.yaml"
    path.write_text("a: 1", encoding="utf-8")

    def fake_parser(_: str) -> None:
        raise RuntimeError("pyyaml no está instalado")

    monkeypatch.setitem(helpers.PARSERS, ".yaml", fake_parser)

    with pytest.raises(ValueError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert "pyyaml" in msg
