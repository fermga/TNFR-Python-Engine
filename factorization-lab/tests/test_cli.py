"""Regression tests for the TNFR factorization CLI."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Callable, Dict, Mapping

import pytest

LAB_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tnfr_factorization import cli  # type: ignore[import]  # noqa: E402


def test_cli_text_mode(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["77"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Spectral TNFR factorization for n=77" in captured
    assert "candidate_factors" in captured
    assert "delta_nfr" in captured


def test_cli_json_mode(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["77", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]["n"] == 77
    assert "arithmetic_delta_nfr" in payload[0]
    assert "partition_manifest_index_path" in payload[0]
    assert "partition_file_archive_path" in payload[0]


def test_cli_distributed_queue_with_custom_dispatcher() -> None:
    exit_code = cli.main(
        [
            "77",
            "--fft-backend",
            "distributed",
            "--dispatcher-workers",
            "2",
            "--dispatcher-timeout",
            "5",
            "--dispatcher-serializer",
            "pickle",
        ]
    )

    assert exit_code == 0


def test_cli_distributed_http_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeHTTPDispatcher:
        def __init__(self, base_url: str, auth_token: str | None = None) -> None:
            self.base_url = base_url
            self.auth_token = auth_token
            self.dispatch = lambda action, payload: None

    monkeypatch.setattr(cli, "HTTPFFTDispatcher", _FakeHTTPDispatcher)

    exit_code = cli.main(
        [
            "77",
            "--fft-backend",
            "distributed",
            "--fft-dispatcher",
            "https://fft.example/api",
            "--dispatcher-http-token",
            "secret",
        ]
    )

    assert exit_code == 0


def test_cli_distributed_callable_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def _fake_loader(
        spec: str, metadata: Mapping[str, Any] | None = None
    ) -> Callable[[str, Dict[str, Any]], Any]:
        captured["spec"] = spec
        captured["metadata"] = metadata

        def _dispatch(
            action: str, payload: Dict[str, Any]
        ) -> None:  # pragma: no cover - simple stub
            captured["last_action"] = action
            captured["last_payload"] = payload

        return _dispatch

    monkeypatch.setattr(cli, "_load_callable_dispatcher", _fake_loader)

    exit_code = cli.main(
        [
            "77",
            "--fft-backend",
            "distributed",
            "--fft-dispatcher",
            "local:tests.fake_dispatcher:build",
        ]
    )

    assert exit_code == 0
    assert captured["spec"] == "tests.fake_dispatcher:build"
    assert captured["metadata"]["source"] == "cli"
