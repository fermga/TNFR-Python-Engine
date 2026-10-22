"""Cold source checks must not silently evaluate an installed TNFR copy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("competing_package", [False, True])
def test_child_checks_checkout_without_changing_caller_environment(
    competing_package, tmp_path, monkeypatch, request
):
    if competing_package:
        shadow = tmp_path / "installed"
        package = shadow / "tnfr"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "raise RuntimeError('stale installed TNFR was imported')\n",
            encoding="utf-8",
        )
        # Independent modules on the caller's path must remain available.
        (shadow / "caller_only.py").write_text("VALUE = 17\n", encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(shadow))
    else:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("TNFR_TEST_CHILD_SENTINEL", "retained")
    before = os.environ.copy()
    environment = request.getfixturevalue("source_tree_environment")
    expected = Path(__file__).resolve().parents[1] / "src" / "tnfr" / "__init__.py"
    program = """
import os
from pathlib import Path
import sys
import tnfr
assert Path(tnfr.__file__).resolve() == Path(sys.argv[1]).resolve()
assert os.environ['TNFR_TEST_CHILD_SENTINEL'] == 'retained'
"""
    if competing_package:
        program += "import caller_only\nassert caller_only.VALUE == 17\n"
    result = subprocess.run(
        [sys.executable, "-c", program, str(expected)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert dict(os.environ) == before
