"""Build and exercise the dependency-free wheel without installing it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("standalone-primality-wheel")
    source = temporary / "source"
    source.mkdir()
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(PACKAGE_ROOT / name, source / name)
    shutil.copytree(
        PACKAGE_ROOT / "tnfr_primality",
        source / "tnfr_primality",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(temporary / "dist"),
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return next((temporary / "dist").glob("*.whl"))


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


def test_wheel_retains_own_metadata_license_and_all_entrypoints(wheel):
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
        assert any(
            "tnfr>=0.0.3.7" in value and "full" in value for value in dependencies
        )
        assert all("extra ==" in value for value in dependencies)
        assert any(name.endswith("/LICENSE") for name in names)
        entries = archive.read(
            next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        ).decode()
        assert "tnfr-primality = tnfr_primality.cli:main" in entries
        assert "tnfr-primality-advanced = tnfr_primality.advanced_cli:main" in entries
        assert "tnfr-primality-legacy = tnfr_primality.__main__:main" in entries


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


@pytest.mark.parametrize("module", ["cli", "advanced_cli"])
def test_wheel_cli_uses_portable_text_output(wheel, module):
    result = _run_wheel(
        wheel,
        "sys.stdout.reconfigure(encoding='cp1252');"
        f"from tnfr_primality.{module} import main;"
        "sys.argv=['tnfr-primality','17','9'];raise SystemExit(main())",
    )
    assert result.returncode == 0, result.stderr
    assert "DeltaNFR" in result.stdout
