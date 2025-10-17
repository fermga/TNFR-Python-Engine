import importlib
import sys
import warnings

import pytest

import tnfr
from tnfr.metrics import register_metrics_callbacks


def _clear_tnfr_modules() -> None:
    for name in list(sys.modules):
        if name == "tnfr" or name.startswith("tnfr."):
            sys.modules.pop(name)


def test_public_exports():
    expected = {
        "__version__",
        "step",
        "run",
        "preparar_red",
        "create_nfr",
    }
    if getattr(tnfr, "_HAS_RUN_SEQUENCE", False):
        expected.add("run_sequence")
    assert set(tnfr.__all__) == expected


def test_basic_flow():
    G, n = tnfr.create_nfr("n1")
    tnfr.preparar_red(G)
    register_metrics_callbacks(G)
    tnfr.step(G)
    tnfr.run(G, steps=2)
    assert len(G.graph["history"]["C_steps"]) == 3


def test_topological_remesh_not_exported():
    assert not hasattr(tnfr, "apply_topological_remesh")


def test_public_api_missing_optional_dependency(monkeypatch):
    real_import_module = importlib.import_module

    def fail_structural(name, package=None):
        if package == "tnfr" and name == ".structural":
            raise ImportError("No module named 'networkx'", name="networkx")
        if package is None and name == "tnfr.structural":
            raise ImportError("No module named 'networkx'", name="networkx")
        return real_import_module(name, package)

    with monkeypatch.context() as m:
        _clear_tnfr_modules()
        m.setattr(importlib, "import_module", fail_structural)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module("tnfr")
        missing = [w for w in caught if issubclass(w.category, ImportWarning)]
        assert missing
        assert any(
            "create_nfr" in str(w.message) and "networkx" in str(w.message)
            for w in missing
        )
        with pytest.raises(ImportError) as excinfo:
            module.create_nfr("n1")
        assert "networkx" in str(excinfo.value)
        with pytest.raises(ImportError):
            module.run_sequence(None, None, [])
        assert not getattr(module, "_HAS_RUN_SEQUENCE", False)
    _clear_tnfr_modules()
    globals()["tnfr"] = importlib.import_module("tnfr")


def test_public_api_internal_import_error(monkeypatch):
    real_import_module = importlib.import_module

    def fail_dynamics(name, package=None):
        if package == "tnfr" and name == ".dynamics":
            raise ImportError("circular import", name="tnfr.dynamics")
        if package is None and name == "tnfr.dynamics":
            raise ImportError("circular import", name="tnfr.dynamics")
        return real_import_module(name, package)

    with monkeypatch.context() as m:
        _clear_tnfr_modules()
        m.setattr(importlib, "import_module", fail_dynamics)
        with pytest.raises(ImportError) as excinfo:
            importlib.import_module("tnfr")
        assert excinfo.value.name == "tnfr.dynamics"
        _clear_tnfr_modules()
    globals()["tnfr"] = importlib.import_module("tnfr")
