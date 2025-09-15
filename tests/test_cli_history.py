"""Pruebas de cli history."""

from collections import deque
import json

import pytest

from tnfr.cli import _save_json, main


def _cli_args(command, *extra):
    base = {
        "run": ["run", "--nodes", "5", "--steps", "0"],
        "sequence": ["sequence", "--nodes", "5"],
    }.get(command)
    if base is None:
        raise ValueError(f"Unsupported command: {command!r}")
    return base + list(extra)


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_save_history(tmp_path, command):
    path = tmp_path / "non" / "existing" / "hist.json"
    assert not path.parent.exists()
    rc = main(_cli_args(command, "--save-history", str(path)))
    assert rc == 0
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    assert "C_steps" in data
    assert len(data["C_steps"]) >= 1
    if command == "run":
        assert len(data["C_steps"]) == 1


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_export_history(tmp_path, command):
    base = tmp_path / "other" / "history"
    assert not base.parent.exists()
    rc = main(_cli_args(command, "--export-history-base", str(base)))
    assert rc == 0
    data = json.loads(base.with_suffix(".json").read_text())
    assert isinstance(data, dict)
    glyphogram = data["glyphogram"]
    assert "t" in glyphogram
    assert len(glyphogram["t"]) >= 1
    if command == "run":
        assert len(glyphogram["t"]) == 1


def test_cli_run_save_and_export_history(tmp_path):
    save_path = tmp_path / "hist.json"
    export_base = tmp_path / "history"
    rc = main(
        _cli_args(
            "run",
            "--save-history",
            str(save_path),
            "--export-history-base",
            str(export_base),
        )
    )
    assert rc == 0
    data_save = json.loads(save_path.read_text())
    data_export = json.loads(export_base.with_suffix(".json").read_text())
    assert isinstance(data_save, dict)
    assert isinstance(data_export, dict)
    assert len(data_save["C_steps"]) == 1
    glyphogram = data_export["glyphogram"]
    assert "t" in glyphogram
    assert len(glyphogram["t"]) == 1


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_without_history_args(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    rc = main(_cli_args(command))
    assert rc == 0
    assert not any(tmp_path.iterdir())


def test_save_json_serializes_iterables(tmp_path):
    path = tmp_path / "data.json"
    data = {"set": {1, 2}, "tuple": (1, 2), "deque": deque([1, 2])}
    _save_json(str(path), data)
    loaded = json.loads(path.read_text())
    assert sorted(loaded["set"]) == [1, 2]
    assert loaded["tuple"] == [1, 2]
    assert loaded["deque"] == [1, 2]
