from __future__ import annotations

import argparse

import pytest

from tnfr.cli import execution


@pytest.fixture()
def cli_args() -> argparse.Namespace:
    return argparse.Namespace()


def test_run_cli_program_returns_one_when_resolve_program_exits_with_zero(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: argparse.Namespace,
) -> None:
    def fake_resolve_program(*_: object, **__: object) -> None:
        raise SystemExit(0)

    def fail_run_program(*_: object, **__: object) -> None:
        pytest.fail("run_program should not be called")

    monkeypatch.setattr(execution, "resolve_program", fake_resolve_program)
    monkeypatch.setattr(execution, "run_program", fail_run_program)

    result = execution._run_cli_program(cli_args)

    assert result == (1, None)


def test_run_cli_program_returns_one_when_resolve_program_exits_with_message(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: argparse.Namespace,
) -> None:
    def fake_resolve_program(*_: object, **__: object) -> None:
        raise SystemExit("boom")

    def fail_run_program(*_: object, **__: object) -> None:
        pytest.fail("run_program should not be called")

    monkeypatch.setattr(execution, "resolve_program", fake_resolve_program)
    monkeypatch.setattr(execution, "run_program", fail_run_program)

    result = execution._run_cli_program(cli_args)

    assert result == (1, None)


def test_run_cli_program_returns_one_when_run_program_exits_with_zero(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: argparse.Namespace,
) -> None:
    sentinel_program = object()

    def fake_resolve_program(*_: object, **__: object) -> object:
        return sentinel_program

    def fake_run_program(*_: object, **__: object) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(execution, "resolve_program", fake_resolve_program)
    monkeypatch.setattr(execution, "run_program", fake_run_program)

    result = execution._run_cli_program(cli_args)

    assert result == (1, None)


def test_run_cli_program_returns_one_when_run_program_exits_with_message(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: argparse.Namespace,
) -> None:
    sentinel_program = object()

    def fake_resolve_program(*_: object, **__: object) -> object:
        return sentinel_program

    def fake_run_program(*_: object, **__: object) -> None:
        raise SystemExit("boom")

    monkeypatch.setattr(execution, "resolve_program", fake_resolve_program)
    monkeypatch.setattr(execution, "run_program", fake_run_program)

    result = execution._run_cli_program(cli_args)

    assert result == (1, None)
