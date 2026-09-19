"""CLI integration checks for result schemas and JSON stream separation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tnfr_primality import advanced_cli as cli  # noqa: E402


def test_standard_validation_accepts_core_report_schema(capsys):
    result = cli.run_validation(7)

    assert result == {
        "tested": 6,
        "correct": 6,
        "accuracy": 1.0,
        "false_positives": 0,
        "false_negatives": 0,
        "error_rate": 0.0,
    }
    output = capsys.readouterr().out
    assert "Numbers tested: 6" in output
    assert "Correct predictions: 6" in output


def test_advanced_validation_preserves_original_report(monkeypatch, capsys):
    report = {
        "tested_numbers": 6,
        "correct_predictions": 5,
        "accuracy": 5 / 6,
        "false_positives": 1,
        "false_negatives": 0,
        "prime_mean_delta_nfr": 0.0,
        "composite_mean_delta_nfr": 3.0,
    }
    monkeypatch.setattr(cli, "HAS_ADVANCED", True)
    monkeypatch.setattr(cli, "validate_tnfr_theory_advanced", lambda _: report)

    result = cli.run_validation(7, use_advanced=True)

    assert result is report
    assert "tested" not in result
    output = capsys.readouterr().out
    assert "Numbers tested: 6" in output
    assert "Correct predictions: 5" in output


@pytest.mark.parametrize(
    ("arguments", "kind"),
    [
        (["2", "4", "--timing"], "numbers"),
        (["--validate", "7"], "validation"),
        (["--benchmark", "7"], "benchmark"),
    ],
)
def test_numeric_routes_emit_one_json_document(monkeypatch, capsys, arguments, kind):
    monkeypatch.setattr(cli, "HAS_ADVANCED", False)
    monkeypatch.setattr(
        sys, "argv", ["tnfr-primality-advanced", *arguments, "--json-output"]
    )

    assert cli.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err  # Human-readable helper output is kept off JSON stdout.
    if kind == "numbers":
        assert [(row["number"], row["is_prime"]) for row in payload] == [
            (2, True),
            (4, False),
        ]
    elif kind == "validation":
        assert payload["tested"] == payload["correct"] == 6
        assert "tested_numbers" not in payload
    else:
        assert payload["total_numbers"] == 4
        assert [row["number"] for row in payload["results"]] == [2, 3, 5, 7]


def test_text_route_retains_human_output(monkeypatch, capsys):
    monkeypatch.setattr(cli, "HAS_ADVANCED", False)
    monkeypatch.setattr(sys, "argv", ["tnfr-primality-advanced", "--validate", "7"])

    assert cli.main() == 0

    captured = capsys.readouterr()
    assert "TNFR Advanced Primality Testing System" in captured.out
    assert "Numbers tested: 6" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("available", [False, True])
def test_infrastructure_json_handles_optional_owner(monkeypatch, capsys, available):
    monkeypatch.setattr(cli, "HAS_ADVANCED", available)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tnfr-primality-advanced", "--infrastructure-status", "--json-output"],
    )
    info = {"infrastructure_available": True, "constants": {"ZETA": 1.0}}

    def observed_status():
        print("helper diagnostic")
        return "available for this fixture"

    monkeypatch.setattr(
        cli, "get_infrastructure_status", observed_status, raising=False
    )
    monkeypatch.setattr(cli, "get_system_info", lambda: info, raising=False)

    assert cli.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    if available:
        assert payload == {"status": "available for this fixture", "system_info": info}
        assert "helper diagnostic" in captured.err
    else:
        assert payload["system_info"] is None
        assert "not available" in payload["status"]


def test_help_names_advanced_entry_point_and_multiplicity(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tnfr-primality-advanced", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "tnfr-primality-advanced --validate" in output
    assert "Omega counts prime factors with multiplicity" in output
    assert "Validate TNFR theory" not in output
