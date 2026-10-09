"""The validation facade exposes its declared shared input adapters."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import tnfr.validation as validation
from tnfr.validation import input_validation


def test_advertised_validation_exports_resolve_to_shared_owners():
    namespace = {}
    exec("from tnfr.validation import *", namespace)
    assert set(validation.__all__) <= namespace.keys()
    adapter_names = {
        name for name in input_validation.__all__ if name.startswith("validate_")
    }
    assert adapter_names <= set(validation.__all__)
    for name in adapter_names:
        assert namespace[name] is getattr(input_validation, name)


def test_validation_configuration_exports_keep_their_distinct_owners_and_signatures():
    from tnfr.validation import config, unified_validation_system

    assert validation.ValidationConfig is unified_validation_system.ValidationConfig
    assert validation.StructuralValidationConfig is config.ValidationConfig
    assert {"ValidationConfig", "StructuralValidationConfig"} <= set(validation.__all__)
    assert (
        "cache_validation_results"
        in inspect.signature(validation.ValidationConfig).parameters
    )
    assert (
        "validate_invariants"
        not in inspect.signature(validation.ValidationConfig).parameters
    )
    assert set(inspect.signature(validation.StructuralValidationConfig).parameters) == {
        "validate_invariants",
        "validate_each_step",
        "min_severity",
        "enable_semantic_validation",
        "allow_semantic_warnings",
    }
    assert (
        validation.ValidationConfig(
            cache_validation_results=False
        ).cache_validation_results
        is False
    )
    policy = validation.StructuralValidationConfig(
        validate_invariants="false", min_severity="warning"
    )
    assert policy.validate_invariants is False
    assert policy.min_severity is validation.InvariantSeverity.WARNING


def test_validation_configuration_stub_names_match_runtime_owners():
    stub = Path(validation.__file__).with_suffix(".pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    imports = {
        item.asname or item.name: (node.module, item.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for item in node.names
    }
    assert imports["ValidationConfig"] == (
        "unified_validation_system",
        "ValidationConfig",
    )
    assert imports["StructuralValidationConfig"] == ("config", "ValidationConfig")
    exported = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert {
        "ValidationConfig",
        "StructuralValidationConfig",
        "configure_validation",
        "validation_config",
    } <= set(exported)


@pytest.mark.parametrize(
    "first_module",
    ("tnfr.validation", "tnfr.operators", "tnfr.operators.grammar"),
)
def test_validation_input_exports_preserve_cold_import_orders(
    first_module, source_tree_environment
):
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib
import sys
importlib.import_module(sys.argv[1])
from tnfr.validation import *
from tnfr.validation import input_validation
from tnfr.validation import config, unified_validation_system
assert ValidationConfig is unified_validation_system.ValidationConfig
assert StructuralValidationConfig is config.ValidationConfig
assert ValidationConfig(cache_validation_results=False).cache_validation_results is False
assert StructuralValidationConfig(validate_invariants="false").validate_invariants is False
assert validate_epi_value is input_validation.validate_epi_value
assert validate_epi_value(-0.5) == -0.5
assert validate_dnfr_value(-0.25) == -0.25
assert validate_vf_value(0.0) == 0.0
try:
    validate_epi_value(True)
except ValidationError:
    pass
else:
    raise AssertionError("Boolean form value was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first_module],
        cwd=root,
        env=source_tree_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
