"""CLI/SDK integration contracts for declared finite operator studies.

These checks exercise serialization, dispatch, error boundaries and logging.
Matching a declared SDK study does not identify an autonomous law or a physical
clock, and catalog inspection must not execute the listed operators.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tnfr import __version__
from tnfr.cli import main
from tnfr.sdk import StudySpec, run_study


def _reject_nonfinite_json(token):
    raise AssertionError(f"Non-standard JSON constant in CLI output: {token}")


def _json(text):
    """Parse one complete JSON document, rejecting NaN and infinity tokens."""
    return json.loads(text, parse_constant=_reject_nonfinite_json)


def _exit_status(arguments):
    try:
        return main(arguments)
    except SystemExit as error:
        return error.code


def _logger_state(logger):
    return (
        logger.level,
        logger.disabled,
        logger.propagate,
        tuple(logger.filters),
        tuple(
            (handler, handler.level, handler.formatter, tuple(handler.filters))
            for handler in logger.handlers
        ),
    )


@contextmanager
def _distinct_logging_configuration():
    """Restore pytest/application logging even if the CLI contract fails."""
    loggers = (logging.getLogger(), logging.getLogger("tnfr"))
    originals = [_logger_state(logger) for logger in loggers]
    temporary_handlers = []
    try:
        for index, logger in enumerate(loggers):
            handler = logging.StreamHandler(io.StringIO())
            handler.setLevel(logging.ERROR + index)
            handler.setFormatter(logging.Formatter(f"caller-{index}: %(message)s"))
            temporary_handlers.append(handler)
            logger.handlers[:] = [handler]
            logger.setLevel(logging.WARNING + index)
            logger.propagate = False
        yield loggers
    finally:
        for logger, state in zip(loggers, originals):
            level, disabled, propagate, filters, handlers = state
            logger.setLevel(level)
            logger.disabled = disabled
            logger.propagate = propagate
            logger.filters[:] = filters
            logger.handlers[:] = [item[0] for item in handlers]
            for handler, handler_level, formatter, handler_filters in handlers:
                handler.setLevel(handler_level)
                handler.setFormatter(formatter)
                handler.filters[:] = handler_filters
        for handler in temporary_handlers:
            handler.close()


def test_python_module_entrypoint_exposes_version_and_one_json_study():
    version = subprocess.run(
        [sys.executable, "-m", "tnfr", "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "ascii:strict"},
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == __version__
    # An ordinary redirected CLI must work without assuming a UTF-8 terminal.
    study = subprocess.run(
        [
            sys.executable,
            "-m",
            "tnfr",
            "network",
            "--nodes",
            "2",
            "--seed",
            "17",
            "--steps",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "ascii:strict"},
    )
    assert study.returncode == 0, study.stderr
    payload = _json(study.stdout)
    assert payload == run_study(StudySpec(nodes=2, seed=17, cycles=0)).to_dict()


@pytest.mark.parametrize("cycles", [0, 1])
def test_exported_spec_roundtrips_to_the_same_declared_sdk_study(
    cycles, tmp_path, capsys
):
    exported = tmp_path / "study.json"
    expected = StudySpec(
        nodes=2,
        topology="ring",
        seed=23,
        sequence="basic_activation",
        cycles=cycles,
        probability=0.25,
        name="independent-parity-control",
    )
    assert (
        main(
            [
                "network",
                "--nodes",
                "2",
                "--topology",
                "ring",
                "--seed",
                "23",
                "--sequence",
                "basic_activation",
                "--cycles",
                str(cycles),
                "--probability",
                "0.25",
                "--name",
                "independent-parity-control",
                "--export-spec",
                str(exported),
            ]
        )
        == 0
    )
    first = _json(capsys.readouterr().out)
    saved = _json(exported.read_text(encoding="utf-8"))
    assert saved == expected.to_dict()
    rebuilt = StudySpec.from_dict(saved)
    assert first == run_study(rebuilt).to_dict()
    assert first["schema_version"] == "tnfr.study.v1"
    assert first["final"]["schema_version"] == "tnfr.diagnostics.v1"
    assert "not elapsed physical time" in first["execution"]["clock"].lower()
    assert "operator-word cycles" in first["execution"]["clock"].lower()
    assert first["execution"]["cycles_completed"] == cycles
    assert main(["network", "--spec", str(exported)]) == 0
    assert _json(capsys.readouterr().out) == first


@pytest.mark.parametrize(
    "builder_flag",
    [
        ["--nodes", "2"],
        ["--topology", "ring"],
        ["--seed", "0"],
        ["--sequence", "basic_activation"],
        ["--steps", "0"],
        ["--cycles", "0"],
        ["--probability", "0.3"],
        ["--name", "study"],
    ],
)
def test_spec_rejects_every_explicit_builder_flag_without_overwriting_files(
    builder_flag, tmp_path, capsys
):
    spec = tmp_path / "input.json"
    spec.write_text(
        json.dumps(StudySpec(nodes=2, cycles=0).to_dict()), encoding="utf-8"
    )
    result = tmp_path / "result.json"
    exported = tmp_path / "export.json"
    result.write_bytes(b"previous result\n")
    exported.write_bytes(b"previous export\n")
    before = {path: path.read_bytes() for path in (spec, result, exported)}
    status = _exit_status(
        [
            "network",
            "--spec",
            str(spec),
            "--output",
            str(result),
            "--export-spec",
            str(exported),
            *builder_flag,
        ]
    )
    streams = capsys.readouterr()
    assert status == 2
    assert streams.out == ""
    assert streams.err
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        "[]",
        "null",
        '{"nodes": true}',
        '{"nodes": 0}',
        '{"seed": -1}',
        '{"cycles": -1}',
        '{"probability": NaN}',
        '{"probability": Infinity}',
        '{"sequence": "missing-sequence"}',
        '{"unsupported": 1}',
    ],
)
def test_invalid_spec_preserves_existing_output_and_export(contents, tmp_path, capsys):
    spec = tmp_path / "invalid.json"
    result = tmp_path / "result.json"
    exported = tmp_path / "export.json"
    spec.write_text(contents, encoding="utf-8")
    result.write_bytes(b"previous result\n")
    exported.write_bytes(b"previous export\n")
    before = {path: path.read_bytes() for path in (spec, result, exported)}
    status = _exit_status(
        [
            "network",
            "--spec",
            str(spec),
            "--output",
            str(result),
            "--export-spec",
            str(exported),
        ]
    )
    streams = capsys.readouterr()
    assert status == 2
    assert streams.out == ""
    assert streams.err
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("retired", ["profile-si", "profile-pipeline"])
def test_removed_profile_commands_fail_as_unknown_commands(retired, capsys):
    assert _exit_status([retired]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "invalid choice" in streams.err


@pytest.mark.parametrize(
    ("arguments", "status"),
    [
        (["--version"], 0),
        (["--help"], 0),
        (["profile-si"], 2),
        (["network", "--nodes", "0"], 2),
        (["network", "--nodes", "2", "--steps", "0"], 0),
    ],
)
def test_cli_restores_application_root_and_tnfr_logging(arguments, status):
    with _distinct_logging_configuration() as loggers:
        expected = [_logger_state(logger) for logger in loggers]
        assert _exit_status(arguments) == status
        assert [_logger_state(logger) for logger in loggers] == expected


@pytest.mark.parametrize(
    ("command", "name"),
    [
        ("operators", None),
        ("operators", "reception"),
        ("sequences", None),
        ("sequences", "basic_activation"),
    ],
)
def test_catalogs_delegate_to_their_sdk_owner_without_executing_a_study(
    command, name, monkeypatch, tmp_path, capsys
):
    import tnfr.sdk as sdk

    def forbidden(*args, **kwargs):
        raise AssertionError("Catalog inspection attempted network execution")

    monkeypatch.setattr(sdk, "run_study", forbidden)
    monkeypatch.setattr(sdk.TNFR, "create", forbidden)
    monkeypatch.setattr(sdk.Network, "evolve", forbidden)
    owner = sdk.TNFR.operators if command == "operators" else sdk.list_sequences
    expected = owner(name)
    arguments = [command, *([name] if name is not None else [])]
    assert main(arguments) == 0
    assert _json(capsys.readouterr().out) == expected
    output = tmp_path / "catalog.json"
    assert main([*arguments, "--output", str(output)]) == 0
    assert capsys.readouterr().out == ""
    assert _json(output.read_text(encoding="utf-8")) == expected


def test_study_logs_use_stderr_while_stdout_is_one_json_document(monkeypatch, capsys):
    import tnfr.sdk as sdk

    declarations = []
    payload = {"nested": [0, None, -1.25], "label": "finite study \u03c6"}

    def observe(specification):
        declarations.append(specification)
        logging.getLogger("tnfr.integration_control").warning("study-progress-sentinel")
        return SimpleNamespace(to_dict=lambda: payload)

    monkeypatch.setattr(sdk, "run_study", observe)
    with _distinct_logging_configuration() as loggers:
        before = [_logger_state(logger) for logger in loggers]
        assert main(["network", "--nodes", "2", "--steps", "0"]) == 0
        assert [_logger_state(logger) for logger in loggers] == before
    streams = capsys.readouterr()
    assert _json(streams.out) == payload
    assert "study-progress-sentinel" not in streams.out
    assert "study-progress-sentinel" in streams.err
    assert declarations == [StudySpec(nodes=2, cycles=0)]


@pytest.mark.parametrize("failure", [ValueError, OSError, RuntimeError])
def test_study_failure_preserves_artifacts_and_restores_logging(
    failure, monkeypatch, tmp_path, capsys
):
    import tnfr.sdk as sdk

    output = tmp_path / "result.json"
    exported = tmp_path / "spec.json"
    output.write_bytes(b"previous result\n")
    exported.write_bytes(b"previous specification\n")
    before_files = {path: path.read_bytes() for path in (output, exported)}
    calls = []

    def fail(specification):
        calls.append(specification)
        raise failure("study-failure-sentinel")

    monkeypatch.setattr(sdk, "run_study", fail)
    arguments = [
        "network",
        "--nodes",
        "2",
        "--steps",
        "0",
        "--output",
        str(output),
        "--export-spec",
        str(exported),
    ]
    with _distinct_logging_configuration() as loggers:
        before_logging = [_logger_state(logger) for logger in loggers]
        if failure is RuntimeError:
            with pytest.raises(RuntimeError, match="study-failure-sentinel"):
                main(arguments)
        else:
            assert main(arguments) == 2
        assert [_logger_state(logger) for logger in loggers] == before_logging
    streams = capsys.readouterr()
    assert streams.out == ""
    if failure is not RuntimeError:
        assert "study-failure-sentinel" in streams.err
    assert len(calls) == 1
    assert {path: path.read_bytes() for path in before_files} == before_files


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_report_is_rejected_before_any_output_replacement(
    nonfinite, monkeypatch, tmp_path, capsys
):
    import tnfr.sdk as sdk

    output = tmp_path / "result.json"
    exported = tmp_path / "spec.json"
    output.write_bytes(b"previous result\n")
    exported.write_bytes(b"previous specification\n")
    before = {path: path.read_bytes() for path in (output, exported)}
    monkeypatch.setattr(
        sdk,
        "run_study",
        lambda specification: SimpleNamespace(to_dict=lambda: {"invalid": nonfinite}),
    )
    assert (
        main(
            [
                "network",
                "--nodes",
                "2",
                "--steps",
                "0",
                "--output",
                str(output),
                "--export-spec",
                str(exported),
            ]
        )
        == 2
    )
    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("collision", ["input-result", "input-export", "result-export"])
def test_resolved_path_collisions_fail_before_execution(
    collision, monkeypatch, tmp_path, capsys
):
    import tnfr.sdk as sdk

    def forbidden(specification):
        raise AssertionError("Path collision reached study execution")

    monkeypatch.setattr(sdk, "run_study", forbidden)
    spec = tmp_path / "input.json"
    spec.write_text(
        json.dumps(StudySpec(nodes=2, cycles=0).to_dict()), encoding="utf-8"
    )
    result = tmp_path / "result.json"
    result.write_bytes(b"previous result\n")
    subdirectory = tmp_path / "nested"
    subdirectory.mkdir()

    def alias(path):
        return str(subdirectory / ".." / path.name)

    if collision == "input-result":
        arguments = ["--spec", str(spec), "--output", alias(spec)]
    elif collision == "input-export":
        arguments = ["--spec", str(spec), "--export-spec", alias(spec)]
    else:
        arguments = ["--output", str(result), "--export-spec", alias(result)]
    before = {path: path.read_bytes() for path in (spec, result)}
    assert main(["network", *arguments]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "distinct" in streams.err
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("command", ["operators", "sequences"])
def test_unknown_catalog_entry_preserves_existing_output(command, tmp_path, capsys):
    output = tmp_path / "catalog.json"
    output.write_bytes(b"previous catalog\n")
    assert main([command, "missing-catalog-entry", "--output", str(output)]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err
    assert output.read_bytes() == b"previous catalog\n"


def test_atomic_result_replacement_failure_leaves_existing_files_intact(
    monkeypatch, tmp_path, capsys
):
    import tnfr.sdk as sdk
    import tnfr.utils.io as io_owner

    output = tmp_path / "result.json"
    exported = tmp_path / "spec.json"
    output.write_bytes(b"previous result\n")
    exported.write_bytes(b"previous specification\n")
    before = {path: path.read_bytes() for path in (output, exported)}
    replace_attempts = []

    def reject_replacement(source, destination):
        replace_attempts.append((Path(source), Path(destination)))
        assert Path(destination).resolve() == output.resolve()
        assert _json(Path(source).read_text(encoding="utf-8")) == {"finite": 1}
        raise OSError("replace-failure-sentinel")

    monkeypatch.setattr(
        sdk,
        "run_study",
        lambda specification: SimpleNamespace(to_dict=lambda: {"finite": 1}),
    )
    monkeypatch.setattr(io_owner.os, "replace", reject_replacement)
    assert (
        main(
            [
                "network",
                "--nodes",
                "2",
                "--steps",
                "0",
                "--output",
                str(output),
                "--export-spec",
                str(exported),
            ]
        )
        == 2
    )
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "replace-failure-sentinel" in streams.err
    assert len(replace_attempts) == 1
    assert {path: path.read_bytes() for path in before} == before
    assert set(tmp_path.iterdir()) == set(before)


def test_legacy_free_run_rejects_negative_steps_before_advancing(monkeypatch, capsys):
    from tnfr.cli import execution

    builds = []

    def build(arguments):
        builds.append(arguments.steps)
        return object()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Invalid step count reached runtime execution or persistence"
        )

    monkeypatch.setattr(execution, "_build_graph_from_args", build)
    monkeypatch.setattr(execution, "run", forbidden)
    monkeypatch.setattr(execution, "play", forbidden)
    monkeypatch.setattr(execution, "_persist_history", forbidden)
    assert main(["run", "--steps", "-1"]) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "steps" in streams.err
    assert "non-negative" in streams.err
    assert builds == [-1]


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_legacy_ambiguous_program_inputs_fail_before_graph_execution(
    command, monkeypatch, capsys
):
    from tnfr.cli import execution

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Ambiguous program selection reached construction or execution"
        )

    monkeypatch.setattr(execution, "_build_graph_from_args", forbidden)
    monkeypatch.setattr(execution, "_load_sequence", forbidden)
    monkeypatch.setattr(execution, "run", forbidden)
    monkeypatch.setattr(execution, "play", forbidden)
    assert (
        main(
            [
                command,
                "--preset",
                "resonant_bootstrap",
                "--sequence-file",
                "must-not-be-read.json",
            ]
        )
        == 2
    )
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "--preset" in streams.err
    assert "--sequence-file" in streams.err
