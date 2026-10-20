"""Keep arithmetic examples faithful to their actual APIs and output scope."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _load_example(name):
    path = PACKAGE_ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"primality_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_basic_example_reports_multiplicity_and_actual_default_weights(capsys):
    example = _load_example("basic_usage")
    example.theory_example()
    output = capsys.readouterr().out

    # 25 = 5**2 has three divisors with sum 31 and multiplicity two.
    assert "Omega=2, tau=3, sigma=31" in output
    assert "DeltaNFR(25) = 2.200000" in output
    assert "zeta=1.0, eta=1.0, theta=1.0" in output


def test_basic_example_works_without_engine_or_unicode_console():
    path = PACKAGE_ROOT / "examples" / "basic_usage.py"
    code = (
        "import runpy,sys;"
        "sys.stdout.reconfigure(encoding='cp1252');"
        f"sys.path.insert(0,{str(PACKAGE_ROOT)!r});"
        f"runpy.run_path({str(path)!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        encoding="cp1252",
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Examples completed" in result.stdout
    assert "Traceback" not in result.stderr
