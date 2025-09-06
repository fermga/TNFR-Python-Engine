"""Pruebas de read structured file errors."""

import pytest
from pathlib import Path
from json import JSONDecodeError
import json
import tnfr.io as io_mod

from tnfr.io import read_structured_file, StructuredFileError


def test_read_structured_file_missing_file(tmp_path: Path):
    path = tmp_path / "missing.json"
    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("No se pudo leer")
    assert str(path) in msg


def test_read_structured_file_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "forbidden.json"
    original_open = Path.open

    def fake_open(
        self, *args, **kwargs
    ):  # pragma: no cover - monkeypatch helper
        if self == path:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("No se pudo leer")
    assert str(path) in msg


def test_read_structured_file_corrupt_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error al parsear archivo JSON en")
    assert str(path) in msg


def test_read_structured_file_corrupt_yaml(tmp_path: Path):
    pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error al parsear archivo YAML en")
    assert str(path) in msg


def test_read_structured_file_corrupt_toml(tmp_path: Path):
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        pytest.importorskip("tomli")
    path = tmp_path / "bad.toml"
    path.write_text("a = [1, 2", encoding="utf-8")
    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error al parsear archivo TOML en")
    assert str(path) in msg


def test_read_structured_file_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.yaml"
    path.write_text("a: 1", encoding="utf-8")

    def fake_parser(_: str) -> None:
        raise ImportError("pyyaml no está instalado")

    monkeypatch.setitem(io_mod.PARSERS, ".yaml", fake_parser)

    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Dependencia faltante al parsear")
    assert str(path) in msg
    assert "pyyaml" in msg


def test_read_structured_file_missing_dependency_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.toml"
    path.write_text("a = 1", encoding="utf-8")

    def fake_parser(_: str) -> None:
        raise ImportError("toml no está instalado")

    monkeypatch.setitem(io_mod.PARSERS, ".toml", fake_parser)

    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Dependencia faltante al parsear")
    assert str(path) in msg
    assert "toml" in msg.lower()


def test_read_structured_file_unicode_error(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error de codificación al leer")
    assert str(path) in msg


def test_json_error_not_reported_as_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTOMLDecodeError(Exception):
        pass

    monkeypatch.setattr(io_mod, "has_toml", False)
    monkeypatch.setattr(io_mod, "TOMLDecodeError", DummyTOMLDecodeError)

    err = JSONDecodeError("msg", "", 0)
    msg = io_mod._format_structured_file_error(Path("data.json"), err)
    assert msg.startswith("Error al parsear archivo JSON en")
    assert not msg.startswith("Error al parsear archivo TOML")


def test_import_error_not_reported_as_toml(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTOMLDecodeError(Exception):
        pass

    monkeypatch.setattr(io_mod, "has_toml", False)
    monkeypatch.setattr(io_mod, "TOMLDecodeError", DummyTOMLDecodeError)

    err = ImportError("dep missing")
    msg = io_mod._format_structured_file_error(Path("data.toml"), err)
    assert msg.startswith("Dependencia faltante al parsear")
    assert not msg.startswith("Error al parsear archivo TOML")


def test_read_structured_file_ignores_missing_yaml_when_parsing_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.json"
    path.write_text("{\"a\": 1}", encoding="utf-8")
    monkeypatch.setattr(io_mod, "yaml", None)
    monkeypatch.setattr(io_mod, "tomllib", None)
    monkeypatch.setattr(io_mod, "PARSERS", {".json": json.loads})
    assert read_structured_file(path) == {"a": 1}
