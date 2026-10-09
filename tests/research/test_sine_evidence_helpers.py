"""Isolation and alias coverage of the opt-in retained-evidence guard."""

import builtins
import importlib
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import sine_evidence_helpers as guard


def test_importing_helper_does_not_import_research_owners(monkeypatch):
    original_import = builtins.__import__

    def forbidden(*args, **kwargs):
        pytest.fail("loading the test helper attempted a research import")

    def import_without_research(name, *args, **kwargs):
        if name == "tnfr" or name.startswith("tnfr."):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(builtins, "__import__", import_without_research)
    spec = importlib.util.spec_from_file_location(
        "cold_evidence_helper", guard.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.forbid_sine_regeneration)


def test_all_declared_generators_aliases_and_restoration_are_guarded_and_restored():
    originals = [
        (importlib.import_module(name), attribute)
        for name, attributes in guard._TARGETS
        for attribute in attributes
    ]
    values = [getattr(module, name) for module, name in originals]
    subprocess_values = [subprocess.run, subprocess.Popen, subprocess.check_output]
    with guard.forbid_sine_regeneration():
        for module, name in originals:
            with pytest.raises(
                pytest.fail.Exception, match="regeneration or restoration"
            ):
                getattr(module, name)()
        for function in (subprocess.run, subprocess.Popen, subprocess.check_output):
            with pytest.raises(pytest.fail.Exception):
                function(["git", "rev-parse", "--show-toplevel"])
    assert [getattr(module, name) for module, name in originals] == values
    assert [
        subprocess.run,
        subprocess.Popen,
        subprocess.check_output,
    ] == subprocess_values


def test_known_import_aliases_are_explicitly_covered():
    declared = dict(guard._TARGETS)
    for module in (
        "tnfr.physics.relational_sine_class_readout",
        "tnfr.physics.relational_sine_class_port_readout",
    ):
        assert {"_full_sine_field", "validated_box_taylor_step"} <= set(
            declared[module]
        )
    assert (
        "bound_sine_class_four_history_readout"
        in declared["tnfr.physics.relational_sine_class_comparison_readout"]
    )
    assert declared["tnfr.research.frozen_source"] == ("restore_frozen_source",)


def test_lazy_consumer_import_binds_original_owner_before_any_patch(monkeypatch):
    original = object()
    kernel = SimpleNamespace(field=original)
    consumers = {}

    def lazy_import(name):
        if name == "synthetic.kernel":
            return kernel
        assert name == "synthetic.consumer"
        assert name not in consumers
        # Reproduce `from kernel import field` on a genuinely first import.
        consumer = SimpleNamespace(field=kernel.field)
        consumers[name] = consumer
        return consumer

    monkeypatch.setattr(
        guard,
        "_TARGETS",
        (("synthetic.kernel", ("field",)), ("synthetic.consumer", ("field",))),
    )
    monkeypatch.setattr(guard.importlib, "import_module", lazy_import)
    assert not consumers
    with guard.forbid_sine_regeneration():
        consumer = consumers["synthetic.consumer"]
        assert kernel.field is consumer.field and kernel.field is not original
        with pytest.raises(pytest.fail.Exception):
            consumer.field()
    assert kernel.field is consumer.field is original


def test_guard_unwinds_nested_context_and_exception():
    from tnfr.physics import _sine_flow

    original = _sine_flow._full_sine_field
    with pytest.raises(RuntimeError, match="synthetic"):
        with guard.forbid_sine_regeneration():
            outer = _sine_flow._full_sine_field
            with guard.forbid_sine_regeneration():
                assert _sine_flow._full_sine_field is not outer
            assert _sine_flow._full_sine_field is outer
            raise RuntimeError("synthetic audit failure")
    assert _sine_flow._full_sine_field is original


def test_missing_declared_attribute_restores_earlier_patches(monkeypatch):
    from tnfr.physics import _sine_flow

    original = _sine_flow._full_sine_field
    monkeypatch.setattr(
        guard,
        "_TARGETS",
        (("tnfr.physics._sine_flow", ("_full_sine_field", "missing_guard_target")),),
    )
    with pytest.raises(AttributeError):
        with guard.forbid_sine_regeneration():
            pytest.fail("missing guard target was silently ignored")
    assert _sine_flow._full_sine_field is original


@pytest.mark.parametrize(
    "arguments",
    [
        {"git_root": Path(".")},
        {"git_base": "a" * 40},
        {"git_root": Path("."), "git_base": True},
        {"git_root": Path("."), "git_base": "short"},
        {"git_root": Path("."), "git_base": "A" * 40},
    ],
)
def test_git_scope_must_be_explicit_and_complete(arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid guard policy imported a research owner")

    monkeypatch.setattr(guard.importlib, "import_module", forbidden)
    with pytest.raises(ValueError):
        with guard.forbid_sine_regeneration(**arguments):
            pytest.fail("invalid guard policy entered")


def test_pinned_git_reads_delegate_only_admitted_commands_and_restore(
    monkeypatch, tmp_path
):
    calls, sentinel = [], object()

    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(subprocess, "Popen", popen)
    commands = (
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "cat-file", "-t", "a" * 40],
        ["git", "show", "a" * 40 + ":src/owner.py"],
    )
    with guard.forbid_sine_regeneration(git_root=tmp_path, git_base="a" * 40):
        for command in commands:
            assert (
                subprocess.Popen(command, cwd=tmp_path, stdout=subprocess.PIPE)
                is sentinel
            )
    assert subprocess.Popen is popen
    assert calls == [
        ((command,), {"cwd": tmp_path, "stdout": subprocess.PIPE})
        for command in commands
    ]


@pytest.mark.parametrize(
    "command,changes",
    [
        (["python", "evaluate.py"], {}),
        (["git", "worktree", "add", "new"], {}),
        (["git", "cat-file", "-t", "b" * 40], {}),
        (["git", "show", "a" * 40 + ":docs/result.json"], {}),
        ("git rev-parse --show-toplevel", {}),
        (["git", "rev-parse", "--show-toplevel"], {"shell": True}),
        (["git", "rev-parse", "--show-toplevel"], {"executable": "python"}),
        (["git", "rev-parse", "--show-toplevel"], {"cwd": None}),
        (["git", "rev-parse", "--show-toplevel"], {"env": {"GIT_DIR": "other"}}),
    ],
)
def test_pinned_git_mode_blocks_other_workers_and_execution_options(
    command, changes, monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted command reached the original Popen")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with guard.forbid_sine_regeneration(git_root=tmp_path, git_base="a" * 40):
        with pytest.raises(pytest.fail.Exception, match="unexpected subprocess"):
            subprocess.Popen(command, **({"cwd": tmp_path} | changes))


def test_pinned_git_mode_rejects_another_root(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("wrong root reached the original Popen")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with guard.forbid_sine_regeneration(git_root=tmp_path, git_base="a" * 40):
        with pytest.raises(pytest.fail.Exception, match="unexpected subprocess"):
            subprocess.Popen(
                ["git", "rev-parse", "--show-toplevel"], cwd=tmp_path.parent
            )
