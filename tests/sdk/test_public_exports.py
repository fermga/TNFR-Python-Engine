"""SDK discovery, typed exports and lazy owner identity remain aligned."""

import ast
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import tnfr.sdk as sdk
from tnfr.sdk import fluent


def test_typed_public_exports_resolve_to_the_same_live_owners():
    stub = ast.parse(Path(sdk.__file__).with_suffix(".pyi").read_text(encoding="utf-8"))
    declarations = {
        alias.asname: ("." * node.level + node.module, alias.name)
        for node in stub.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname is not None
    }
    assert set(sdk.__all__) == set(declarations)
    assert len(sdk.__all__) == len(set(sdk.__all__))
    for name, (module_name, attribute) in declarations.items():
        owner = importlib.import_module(module_name, sdk.__name__)
        assert getattr(sdk, name) is getattr(owner, attribute)
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(sdk, "not_a_supported_sdk_export")


def test_importing_sdk_does_not_eagerly_import_its_execution_owners(
    source_tree_environment,
):
    code = """
import sys
from pathlib import Path
import tnfr
import tnfr.sdk as sdk
assert Path(tnfr.__file__).resolve() == Path(sys.argv[1]), tnfr.__file__
owners = ('tnfr.sdk.study', 'tnfr.sdk.simple', 'tnfr.sdk.fluent', 'tnfr.sdk.self_opt')
assert not any(name in sys.modules for name in owners)
assert sdk.StudySpec is __import__('tnfr.sdk.study', fromlist=['StudySpec']).StudySpec
assert 'tnfr.sdk.self_opt' not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve().parents[2] / "src" / "tnfr" / "__init__.py"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=source_tree_environment,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["NetworkConfig", "NetworkResults", "TNFRNetwork"])
def test_fluent_type_contract_keeps_public_members_and_argument_names(name):
    stub = ast.parse(
        Path(fluent.__file__).with_suffix(".pyi").read_text(encoding="utf-8")
    )
    declaration = next(
        item
        for item in stub.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    methods = {
        item.name: item
        for item in declaration.body
        if isinstance(item, ast.FunctionDef)
    }
    live_class = getattr(fluent, name)
    public_members = {
        key
        for key, value in vars(live_class).items()
        if not key.startswith("_") and (callable(value) or isinstance(value, property))
    }
    assert public_members == {key for key in methods if not key.startswith("_")}
    for method_name, declaration in methods.items():
        live = getattr(live_class, method_name)
        if isinstance(live, property):
            live = live.fget
        arguments = declaration.args
        declared = [argument.arg for argument in arguments.posonlyargs + arguments.args]
        if arguments.vararg:
            declared.append(arguments.vararg.arg)
        declared.extend(argument.arg for argument in arguments.kwonlyargs)
        if arguments.kwarg:
            declared.append(arguments.kwarg.arg)
        assert declared == list(inspect.signature(live).parameters)
