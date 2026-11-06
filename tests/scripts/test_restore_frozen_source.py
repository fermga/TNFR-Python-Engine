"""CLI wiring uses stand-in inspection; no frozen code or flow is run."""

import importlib.util
from pathlib import Path

from tnfr.research.frozen_source import FrozenSourceInspection
from tnfr.utils.io import json_loads


def test_inspection_is_default_and_destination_is_explicit(
    monkeypatch, capsys, tmp_path
):
    path = Path(__file__).resolve().parents[2] / "scripts/restore_frozen_source.py"
    spec = importlib.util.spec_from_file_location("restore_frozen_source_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    report = FrozenSourceInspection(
        "a" * 40, "control.freeze.json", "b" * 64, 3, 1, "evaluate.py", ()
    )

    def inspect(*args):
        calls.append(("inspect", args))
        return report

    def restore(*args):
        calls.append(("restore", args))
        return report

    monkeypatch.setattr(module, "inspect_frozen_source", inspect)
    monkeypatch.setattr(module, "restore_frozen_source", restore)
    assert module.main(["--receipt", report.receipt_path]) == 0
    assert calls == [("inspect", (module.ROOT, report.receipt_path))]
    assert json_loads(capsys.readouterr().out)["operation"] == "inspect"
    destination = tmp_path / "new"
    assert (
        module.main(
            ["--receipt", report.receipt_path, "--destination", str(destination)]
        )
        == 0
    )
    assert calls[-1] == ("restore", (module.ROOT, report.receipt_path, destination))
    assert json_loads(capsys.readouterr().out)["operation"] == "restore"
