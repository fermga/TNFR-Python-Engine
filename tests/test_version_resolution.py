"""Tests covering version resolution when package metadata is missing."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

try:  # pragma: no cover - Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
        tomllib = None  # type: ignore[assignment]

if tomllib is not None:  # pragma: no cover - trivial branch
    _TOML_DECODE_ERRORS = (getattr(tomllib, "TOMLDecodeError", ValueError),)
else:  # pragma: no cover - optional dependency missing
    _TOML_DECODE_ERRORS = (ValueError,)


def _read_project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if tomllib is None:
        raise RuntimeError("tomllib/tomli is required to parse pyproject.toml")

    try:
        with pyproject_path.open("rb") as stream:
            data = tomllib.load(stream)
    except OSError as exc:  # pragma: no cover - defensive guard for IO errors
        raise RuntimeError("pyproject.toml is unreadable") from exc
    except _TOML_DECODE_ERRORS as exc:  # pragma: no cover - invalid format guard
        raise RuntimeError("pyproject.toml could not be parsed") from exc

    project_data = data.get("project")
    if not isinstance(project_data, dict):  # pragma: no cover - defensive guard
        raise RuntimeError("pyproject.toml has no [project] table")

    version = project_data.get("version")
    if isinstance(version, str):
        return version
    raise RuntimeError("version not found in pyproject.toml")


def test_version_falls_back_to_pyproject() -> None:
    expected_version = _read_project_version()
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import importlib.metadata as metadata
        import sys
        from types import ModuleType
        import warnings

        sys.path.insert(0, {src_path!r})

        def fake_version(_):
            raise metadata.PackageNotFoundError

        metadata.version = fake_version

        fake_modules = {{
            "tnfr.dynamics": ("step", "run"),
            "tnfr.structural": ("create_nfr", "run_sequence"),
            "tnfr.ontosim": ("preparar_red",),
        }}
        for module_name, attributes in fake_modules.items():
            module = ModuleType(module_name)
            for attr in attributes:
                setattr(module, attr, lambda *args, **kwargs: None)
            sys.modules[module_name] = module

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tnfr

        if caught:
            raise RuntimeError(f"unexpected warnings: {{caught}}")

        print(tnfr.__version__, end="")
        """
    ).format(src_path=str(repo_root / "src"))

    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == expected_version
