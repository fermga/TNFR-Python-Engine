"""Pruebas de cli history."""

from tnfr.cli import main
import json


def test_cli_run_save_history(tmp_path):
    path = tmp_path / "non" / "existing" / "hist.json"
    assert not path.parent.exists()
    rc = main(["run", "--nodes", "5", "--steps", "0", "--save-history", str(path)])
    assert rc == 0
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_cli_run_export_history(tmp_path):
    base = tmp_path / "other" / "history"
    assert not base.parent.exists()
    rc = main(
        ["run", "--nodes", "5", "--steps", "0", "--export-history-base", str(base)]
    )
    assert rc == 0
    data = json.loads((base.with_suffix(".json")).read_text())
    assert isinstance(data, dict)


def test_cli_sequence_save_history(tmp_path):
    path = tmp_path / "non" / "existing" / "hist.json"
    assert not path.parent.exists()
    rc = main(["sequence", "--nodes", "5", "--save-history", str(path)])
    assert rc == 0
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_cli_sequence_export_history(tmp_path):
    base = tmp_path / "other" / "history"
    assert not base.parent.exists()
    rc = main(["sequence", "--nodes", "5", "--export-history-base", str(base)])
    assert rc == 0
    data = json.loads((base.with_suffix(".json")).read_text())
    assert isinstance(data, dict)
