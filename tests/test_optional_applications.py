"""Optional applications stay lazy and prefer an existing installation."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tnfr import _optional_apps

PACKAGE = "tnfr_optional_test_application"
DIRECTORY = "test-application"


@pytest.fixture(autouse=True)
def isolated_application_import(monkeypatch):
    """Restore the process search path and discard only the test package."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert PACKAGE not in sys.modules
    yield
    sys.modules.pop(PACKAGE, None)
    importlib.invalidate_caches()


def _helper_at(path: Path):
    """Load the real helper under an installed or checkout filesystem layout."""
    path.parent.mkdir(parents=True)
    shutil.copyfile(_optional_apps.__file__, path)
    spec = importlib.util.spec_from_file_location("_test_application_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_at(directory: Path, origin: str) -> Path:
    package = directory / PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"ORIGIN = {origin!r}\n", encoding="utf-8")
    importlib.invalidate_caches()
    return package


def test_checkout_fallback_enables_real_import_without_eager_import(tmp_path):
    helper = _helper_at(tmp_path / "src" / "tnfr" / "_optional_apps.py")
    application = tmp_path / "applications" / DIRECTORY
    package = _package_at(application, "checkout")

    assert importlib.util.find_spec(PACKAGE) is None
    helper.bootstrap_application(PACKAGE, DIRECTORY)
    assert PACKAGE not in sys.modules
    assert sys.path[0] == str(application)

    # Repeating discovery before and after import must not add another path.
    helper.bootstrap_application(PACKAGE, DIRECTORY)
    loaded = importlib.import_module(PACKAGE)
    assert loaded.ORIGIN == "checkout"
    assert Path(loaded.__file__) == package / "__init__.py"
    helper.bootstrap_application(PACKAGE, DIRECTORY)
    assert sys.path.count(str(application)) == 1


def test_installed_package_wins_over_checkout_without_importing_either(
    tmp_path, monkeypatch
):
    helper = _helper_at(tmp_path / "checkout" / "src" / "tnfr" / "_optional_apps.py")
    application = tmp_path / "checkout" / "applications" / DIRECTORY
    _package_at(application, "checkout")
    installed = tmp_path / "site-packages"
    package = _package_at(installed, "installed")
    monkeypatch.syspath_prepend(str(installed))
    previous = tuple(sys.path)

    helper.bootstrap_application(PACKAGE, DIRECTORY)
    assert tuple(sys.path) == previous
    assert PACKAGE not in sys.modules
    loaded = importlib.import_module(PACKAGE)
    assert loaded.ORIGIN == "installed"
    assert Path(loaded.__file__) == package / "__init__.py"


def test_loaded_package_without_spec_does_not_trigger_find_spec_failure(
    tmp_path, monkeypatch
):
    helper = _helper_at(tmp_path / "src" / "tnfr" / "_optional_apps.py")
    installed = tmp_path / "installed"
    _package_at(installed, "already imported")
    monkeypatch.syspath_prepend(str(installed))
    loaded = importlib.import_module(PACKAGE)
    loaded.__spec__ = None
    before = tuple(sys.path)

    helper.bootstrap_application(PACKAGE, DIRECTORY)
    assert importlib.import_module(PACKAGE) is loaded
    assert tuple(sys.path) == before


def test_installed_engine_cannot_discover_adjacent_checkout_like_applications(tmp_path):
    helper = _helper_at(tmp_path / "site-packages" / "tnfr" / "_optional_apps.py")
    _package_at(tmp_path / "applications" / DIRECTORY, "unrelated adjacent source")
    before = tuple(sys.path)

    helper.bootstrap_application(PACKAGE, DIRECTORY)
    assert tuple(sys.path) == before
    assert importlib.util.find_spec(PACKAGE) is None
    with pytest.raises(ModuleNotFoundError, match=PACKAGE):
        importlib.import_module(PACKAGE)


def test_checkout_without_application_package_does_not_add_a_search_path(tmp_path):
    helper = _helper_at(tmp_path / "src" / "tnfr" / "_optional_apps.py")
    # The directory alone is not the maintained importable package.
    (tmp_path / "applications" / DIRECTORY).mkdir(parents=True)
    before = tuple(sys.path)
    helper.bootstrap_application(PACKAGE, DIRECTORY)
    assert tuple(sys.path) == before
    assert importlib.util.find_spec(PACKAGE) is None


def test_cold_engine_and_adapter_imports_do_not_activate_applications(tmp_path):
    source = Path(_optional_apps.__file__).resolve().parents[1]
    code = """
import json
import sys
sys.path.insert(0, sys.argv[1])
before = tuple(sys.path)
import tnfr
import tnfr.primality
import tnfr.factorization
print(json.dumps({
    "unchanged_path": tuple(sys.path) == before,
    "loaded_applications": [name for name in sys.modules
                            if name.startswith(("tnfr_primality", "tnfr_factorization"))],
    "loaded_engine": tnfr.__file__,
}))
"""
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    result = json.loads(process.stdout)
    assert result["unchanged_path"] is True
    assert result["loaded_applications"] == []
    assert Path(result["loaded_engine"]) == source / "tnfr" / "__init__.py"
