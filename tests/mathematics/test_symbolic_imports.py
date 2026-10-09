"""Symbolic facade identity and optional-dependency import boundaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_source_process(code, environment):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_mathematics_uses_the_existing_symbolic_public_exports():
    pytest.importorskip("sympy")
    from tnfr import math, mathematics

    symbolic_exports = [
        name for name in math.__all__ if name not in {"symbolic", "__version__"}
    ]
    assert mathematics.__all__[-len(symbolic_exports) - 1 :] == [
        *symbolic_exports,
        "math",
    ]
    assert mathematics.math is math
    assert "symbolic" not in mathematics.__all__
    assert "__version__" not in mathematics.__all__
    for name in symbolic_exports:
        assert getattr(mathematics, name) is getattr(math, name)


def test_numerical_facade_imports_without_optional_sympy(source_tree_environment):
    _run_source_process(
        """
import importlib.abc
import sys

class WithoutSympy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sympy' or fullname.startswith('sympy.'):
            raise ModuleNotFoundError('SymPy intentionally unavailable', name='sympy')

sys.meta_path.insert(0, WithoutSympy())
import tnfr
from tnfr import mathematics

assert not mathematics._HAS_SYMBOLIC
assert 'math' not in mathematics.__all__
for name, dependencies in tnfr.EXPORT_DEPENDENCIES.items():
    if dependencies['third_party'] == ('sympy',):
        assert name not in mathematics.__all__
        assert not hasattr(mathematics, name)
        assert getattr(tnfr, name).__tnfr_missing_dependency__['missing'] == 'sympy'
element = mathematics.BEPIElement((0.25, 0.25), (0.25,), (0, 1))
assert element.real_scalar_embedding() == 0.25
assert not any(name == 'sympy' or name.startswith('sympy.') for name in sys.modules)
""",
        source_tree_environment,
    )


@pytest.mark.parametrize(
    "exception",
    [
        "RuntimeError('broken symbolic implementation')",
        "ModuleNotFoundError('internal dependency missing', name='tnfr.math.missing')",
        "ModuleNotFoundError('incompatible SymPy installation', name='sympy.missing')",
    ],
)
def test_symbolic_import_failures_are_not_optional_absence(
    source_tree_environment, exception
):
    pytest.importorskip("sympy")
    _run_source_process(
        """
import importlib
import importlib.abc
import sys
import tnfr
from tnfr import mathematics

# The root package has completed its imports; isolate the numerical facade's
# handling of a subsequent failure in the real symbolic module import path.
sys.modules.pop('tnfr.math.symbolic', None)
sys.modules.pop('tnfr.math', None)
del tnfr.math
failure = """
        + exception
        + """

class BrokenSymbolic(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tnfr.math.symbolic':
            raise failure

sys.meta_path.insert(0, BrokenSymbolic())
try:
    importlib.reload(mathematics)
except type(failure) as caught:
    assert caught is failure
else:
    raise AssertionError('Unexpected symbolic failure was silently hidden')
""",
        source_tree_environment,
    )
