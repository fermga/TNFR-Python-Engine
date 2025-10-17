"""Tests covering version resolution when package metadata is missing."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path


def _read_project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, flags=re.MULTILINE)
    if not match:  # pragma: no cover - defensive guard for unexpected formats
        raise RuntimeError("version not found in pyproject.toml")
    return match.group(1)


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
