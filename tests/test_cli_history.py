"""Pruebas de cli history."""

from tnfr.cli import main
import json


def test_cli_run_save_history(tmp_path):
    path = tmp_path / "non" / "existing" / "hist.json"
    assert not path.parent.exists()
    rc = main(
        ["run", "--nodes", "5", "--steps", "0", "--save-history", str(path)]
    )
    assert rc == 0
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_cli_run_export_metrics(tmp_path):
    base = tmp_path / "other" / "history"
    assert not base.parent.exists()
    rc = main(
        [
            "run",
            "--nodes",
            "5",
            "--steps",
            "0",
            "--export-history-base",
            str(base),
        ]
    )
    assert rc == 0
    data = json.loads((base.with_suffix(".json")).read_text())
    assert isinstance(data, dict)


def test_cli_run_save_and_export_metrics(tmp_path):
    save_path = tmp_path / "hist.json"
    export_base = tmp_path / "history"
    rc = main(
        [
            "run",
            "--nodes",
            "5",
            "--steps",
            "0",
            "--save-history",
            str(save_path),
            "--export-history-base",
            str(export_base),
        ]
    )
    assert rc == 0
    data_save = json.loads(save_path.read_text())
    data_export = json.loads((export_base.with_suffix(".json")).read_text())
    assert isinstance(data_save, dict)
    assert isinstance(data_export, dict)


def test_cli_sequence_save_history(tmp_path):
    path = tmp_path / "non" / "existing" / "hist.json"
    assert not path.parent.exists()
    rc = main(["sequence", "--nodes", "5", "--save-history", str(path)])
    assert rc == 0
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_cli_sequence_export_metrics(tmp_path):
    base = tmp_path / "other" / "history"
    assert not base.parent.exists()
    rc = main(["sequence", "--nodes", "5", "--export-history-base", str(base)])
    assert rc == 0
    data = json.loads((base.with_suffix(".json")).read_text())
    assert isinstance(data, dict)


def test_cli_run_no_history_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "--nodes", "5", "--steps", "0"])
    assert rc == 0
    assert not any(tmp_path.iterdir())


def test_cli_sequence_no_history_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["sequence", "--nodes", "5"])
    assert rc == 0
    assert not any(tmp_path.iterdir())
