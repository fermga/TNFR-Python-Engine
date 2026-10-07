"""Lazy physics discovery preserves defining objects and cold import boundaries."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def facade_owners():
    """Resolve each public owner once without executing a research report."""
    import tnfr.physics as physics

    owners = {
        module_name: importlib.import_module(module_name, physics.__name__)
        for module_name in dict.fromkeys(physics._EXPORT_MODULES.values())
    }
    return physics, owners


def test_public_registry_resolves_defining_objects(facade_owners) -> None:
    physics, owners = facade_owners
    assert physics.__all__ == list(physics._EXPORT_MODULES)
    assert len(physics.__all__) == len(set(physics.__all__))
    for name, module_name in physics._EXPORT_MODULES.items():
        expected = getattr(owners[module_name], name)
        assert getattr(physics, name) is expected
        assert physics.__dict__[name] is expected


def test_generated_type_stub_preserves_public_names_and_owners() -> None:
    from scripts.generate_physics_stub import render_stub

    root = Path(__file__).resolve().parents[2]
    source = root / "src" / "tnfr" / "physics" / "__init__.py"
    stub_text = source.with_suffix(".pyi").read_text(encoding="utf-8")
    assert stub_text == render_stub(source)
    source_tree = ast.parse(source.read_text(encoding="utf-8"))
    registry = next(
        ast.literal_eval(node.value)
        for node in source_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_EXPORT_MODULES"
            for target in node.targets
        )
    )
    stub_tree = ast.parse(stub_text)
    reexports = []
    for node in stub_tree.body:
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                assert imported.asname == imported.name
                reexports.append((imported.name, "." * node.level + node.module))
    assert reexports == list(registry.items())
    stub_public_names = next(
        ast.literal_eval(node.value)
        for node in stub_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert stub_public_names == list(registry)
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in stub_tree.body
    )


def test_wildcard_import_retains_public_registry(facade_owners) -> None:
    physics, _ = facade_owners
    namespace: dict[str, object] = {}
    exec("from tnfr.physics import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(physics.__all__)
    for name in physics.__all__:
        assert namespace[name] is getattr(physics, name)


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.physics",
        "import tnfr.physics.mutation_trigger",
    ),
)
def test_cold_discovery_does_not_load_unrequested_reports(
    imports: str, source_tree_environment
) -> None:
    code = (
        imports
        + "\n"
        + """
import sys
import tnfr.physics as physics

unrequested = {
    "tnfr.physics.c6_carried_return",
    "tnfr.physics.binary64_remesh_relative_defect",
    "tnfr.physics.relational_sine_scale",
    "tnfr.physics.runtime_remesh_schedule_block_margin",
}
assert unrequested.isdisjoint(sys.modules)
before = set(sys.modules)
discovered = dir(physics)
assert discovered == sorted(set(discovered))
assert set(physics.__all__) <= set(discovered)
assert set(sys.modules) == before
try:
    physics.this_is_not_a_physics_export
except AttributeError as exc:
    assert str(exc) == (
        "module 'tnfr.physics' has no attribute 'this_is_not_a_physics_export'"
    )
else:
    raise AssertionError("An unknown public name must raise AttributeError")
assert set(sys.modules) == before

from tnfr.physics import RegionalFormRegion
from tnfr.physics.form_geometry import RegionalFormRegion as owner_type
assert RegionalFormRegion is owner_type
assert physics.RegionalFormRegion is owner_type
assert unrequested.isdisjoint(sys.modules)
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_owner_import_errors_are_not_hidden(monkeypatch) -> None:
    import tnfr.physics as physics

    failure = ImportError("The requested owner is unavailable")

    def unavailable_owner(module_name: str, package_name: str):
        raise failure

    monkeypatch.setattr(physics, "import_module", unavailable_owner)
    with pytest.raises(ImportError) as caught:
        physics.__getattr__(physics.__all__[0])
    assert caught.value is failure
