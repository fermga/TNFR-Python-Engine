"""Build and exercise the dependency-free wheel without installing it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("standalone-primality-wheel")
    source = temporary / "source"
    shutil.copytree(
        PACKAGE_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", "build", "dist"
        ),
    )
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(temporary / "dist"),
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return temporary / "dist"


@pytest.fixture(scope="module")
def wheel(distributions):
    return next(distributions.glob("*.whl"))


def _run_wheel(wheel, code):
    # No site packages, user site or PYTHONPATH: only stdlib and the built wheel.
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import sys;sys.path.insert(0,{str(wheel)!r});{code}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )


def test_metadata_is_independent_of_callers_project_configuration(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="unrelated-project"\nversion="9.9"\n', encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "setup.py"), "--name"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "tnfr-primality"


def test_wheel_retains_basic_metadata_license_and_entrypoint(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(
            archive.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            )
        )
        assert metadata["Name"] == "tnfr-primality"
        assert metadata["Version"] == "1.1.0"
        assert metadata["Requires-Python"] == ">=3.8"
        dependencies = metadata.get_all("Requires-Dist", [])
        assert all("extra ==" in value for value in dependencies)
        assert metadata.get_all("Provides-Extra") == ["dev"]
        assert not any("tnfr" in value or "sympy" in value for value in dependencies)
        assert any(name.endswith("/LICENSE") for name in names)
        entries = archive.read(
            next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        ).decode()
        assert "tnfr-primality = tnfr_primality.cli:main" in entries
        assert "tnfr-primality-advanced" not in entries
        assert "tnfr-primality-legacy" not in entries
        assert not any("archive/" in name or "advanced" in name for name in names)


def test_source_distribution_excludes_retired_material(distributions):
    with tarfile.open(next(distributions.glob("*.tar.gz"))) as archive:
        names = archive.getnames()
    assert any(name.endswith("/examples/basic_usage.py") for name in names)
    assert any(name.endswith("/tests/test_basic_cli.py") for name in names)
    assert not any("/archive/" in name or "advanced" in name for name in names)
    assert not any(name.endswith("comprehensive_benchmark.py") for name in names)


def test_dependency_free_module_preserves_json_and_failure_status(wheel):
    imported = _run_wheel(wheel, "import tnfr_primality.__main__")
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout == ""
    response = _run_wheel(
        wheel,
        "import runpy;sys.argv=['tnfr_primality','2','4','--json-output'];"
        "runpy.run_module('tnfr_primality',run_name='__main__')",
    )
    assert response.returncode == 0, response.stderr
    assert [row["is_prime"] for row in json.loads(response.stdout)] == [True, False]
    missing = _run_wheel(
        wheel,
        "import runpy;sys.argv=['tnfr_primality'];"
        "runpy.run_module('tnfr_primality',run_name='__main__')",
    )
    assert missing.returncode == 1


@pytest.mark.parametrize("module", ["cli", "__main__"])
def test_wheel_cli_uses_portable_text_output(wheel, module):
    result = _run_wheel(
        wheel,
        "sys.stdout.reconfigure(encoding='cp1252');"
        f"from tnfr_primality.{module} import main;"
        "sys.argv=['tnfr-primality','17','9'];raise SystemExit(main())",
    )
    assert result.returncode == 0, result.stderr
    assert "DeltaNFR" in result.stdout


def test_import_does_not_probe_optional_engine_or_retired_adapters(wheel):
    result = _run_wheel(
        wheel,
        "import importlib.abc;"
        "exec('class RejectOptional(importlib.abc.MetaPathFinder):\\n"
        " def find_spec(self, fullname, path=None, target=None):\\n"
        '  if fullname.split(".")[0] in '
        '("tnfr", "numpy", "networkx", "sympy") or "advanced" in fullname:\\n'
        '   raise AssertionError("unexpected optional import: "+fullname)\\n\');'
        "sys.meta_path.insert(0,RejectOptional());"
        "import tnfr_primality.cli,tnfr_primality.__main__;"
        "assert tnfr_primality.__main__.main is tnfr_primality.cli.main",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_documented_installation_check_uses_portable_text_output(tmp_path):
    result = subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "test_installation.py")],
        cwd=tmp_path,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        text=True,
        encoding="cp1252",
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Tests passed: 5/5" in result.stdout
    assert "Arithmetic validation" in result.stdout
