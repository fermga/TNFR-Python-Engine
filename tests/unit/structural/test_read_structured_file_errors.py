"""Tests for ``read_structured_file`` error handling."""

import importlib.util
from json import JSONDecodeError
from pathlib import Path

import pytest

import tnfr.io as io_mod


def test_read_structured_file_missing_file(tmp_path: Path):
    path = tmp_path / "missing.json"
    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Could not read")
    assert str(path) in msg


def test_read_structured_file_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "data.txt"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(io_mod.StructuredFileError) as exc:
        io_mod.read_structured_file(path)
    msg = str(exc.value)
    assert msg == f"Error parsing {path}: Unsupported suffix: .txt"


def test_read_structured_file_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "forbidden.json"
    original_open = Path.open

    def fake_open(self, *args, **kwargs):  # pragma: no cover - monkeypatch helper
        if self == path:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)
    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Could not read")
    assert str(path) in msg


def test_read_structured_file_corrupt_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error parsing JSON file at")
    assert str(path) in msg


def test_read_structured_file_corrupt_yaml(tmp_path: Path):
    pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error parsing YAML file at")
    assert str(path) in msg


def test_read_structured_file_corrupt_toml(tmp_path: Path):
    if importlib.util.find_spec("tomllib") is None:
        pytest.importorskip("tomli")
    path = tmp_path / "bad.toml"
    path.write_text("a = [1, 2", encoding="utf-8")
    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Error parsing TOML file at")
    assert str(path) in msg


def test_read_structured_file_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.yaml"
    path.write_text("a: 1", encoding="utf-8")

    def fake_safe_load(_: str) -> None:
        raise ImportError("pyyaml is not installed")

    monkeypatch.setattr(io_mod, "_YAML_SAFE_LOAD", fake_safe_load)

    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Missing dependency parsing")
    assert str(path) in msg
    assert "pyyaml" in msg


def test_read_structured_file_missing_dependency_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.toml"
    path.write_text("a = 1", encoding="utf-8")

    def fake_loads(_: str) -> None:
        raise ImportError("toml is not installed")

    monkeypatch.setattr(io_mod, "_TOML_LOADS", fake_loads)

    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Missing dependency parsing")
    assert str(path) in msg
    assert "toml" in msg.lower()


def test_read_structured_file_unicode_error(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(io_mod.StructuredFileError) as excinfo:
        io_mod.read_structured_file(path)
    msg = str(excinfo.value)
    assert msg.startswith("Encoding error while reading")
    assert str(path) in msg


def test_json_error_not_reported_as_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTOMLDecodeError(Exception):
        pass

    monkeypatch.setattr(io_mod, "has_toml", False)
    monkeypatch.setattr(io_mod, "TOMLDecodeError", DummyTOMLDecodeError)

    err = JSONDecodeError("msg", "", 0)
    msg = io_mod._format_structured_file_error(Path("data.json"), err)
    assert msg.startswith("Error parsing JSON file at")
    assert not msg.startswith("Error parsing TOML file")


def test_import_error_not_reported_as_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTOMLDecodeError(Exception):
        pass

    monkeypatch.setattr(io_mod, "has_toml", False)
    monkeypatch.setattr(io_mod, "TOMLDecodeError", DummyTOMLDecodeError)

    err = ImportError("dep missing")
    msg = io_mod._format_structured_file_error(Path("data.toml"), err)
    assert msg.startswith("Missing dependency parsing")
    assert not msg.startswith("Error parsing TOML file")


def test_read_structured_file_ignores_missing_yaml_when_parsing_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    def missing_yaml(*_: object, **__: object) -> None:
        raise ImportError("pyyaml is not installed")

    def missing_toml(*_: object, **__: object) -> None:
        raise ImportError("toml is not installed")

    monkeypatch.setattr(io_mod, "_YAML_SAFE_LOAD", missing_yaml)
    monkeypatch.setattr(io_mod, "_TOML_LOADS", missing_toml)
    monkeypatch.setattr(io_mod, "yaml", None)
    monkeypatch.setattr(io_mod, "tomllib", None)
    assert io_mod.read_structured_file(path) == {"a": 1}


def test_read_structured_file_unhandled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")

    def bad_parser(_: str) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(io_mod, "_get_parser", lambda suffix: bad_parser)

    with pytest.raises(ValueError):
        io_mod.read_structured_file(path)
