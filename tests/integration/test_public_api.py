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
        "prepare_network",
        "preparar_red",
        "create_nfr",
    }
    if getattr(tnfr, "_HAS_RUN_SEQUENCE", False):
        expected.add("run_sequence")
    assert set(tnfr.__all__) == expected


def test_basic_flow():
    G, n = tnfr.create_nfr("n1")
    tnfr.prepare_network(G)
    assert tnfr.preparar_red is tnfr.prepare_network
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


def test_public_api_missing_prepare_network_dependency(monkeypatch):
    real_import_module = importlib.import_module

    def fail_ontosim(name, package=None):
        if package == "tnfr" and name == ".ontosim":
            raise ImportError("No module named 'networkx'", name="networkx")
        if package is None and name == "tnfr.ontosim":
            raise ImportError("No module named 'networkx'", name="networkx")
        return real_import_module(name, package)

    with monkeypatch.context() as m:
        _clear_tnfr_modules()
        m.setattr(importlib, "import_module", fail_ontosim)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module("tnfr")
        missing = [w for w in caught if issubclass(w.category, ImportWarning)]
        assert missing
        warning_messages = [str(w.message) for w in missing]
        assert any(
            "prepare_network" in message and "networkx" in message
            for message in warning_messages
        )
        assert any(
            "preparar_red" in message and "networkx" in message
            for message in warning_messages
        )
        assert "prepare_network" in module.__all__
        assert "preparar_red" in module.__all__
        assert not getattr(module, "_HAS_PREPARAR_RED", True)
        assert not getattr(module, "_HAS_PREPARE_NETWORK", True)
        with pytest.raises(ImportError) as excinfo:
            module.preparar_red(None)
        assert "networkx" in str(excinfo.value)
        with pytest.raises(ImportError) as excinfo:
            module.prepare_network(None)
        assert "networkx" in str(excinfo.value)
        info = getattr(module.preparar_red, "__tnfr_missing_dependency__", {})
        assert info.get("export") == "preparar_red"
        assert info.get("missing") == "networkx"
        prepare_info = getattr(module.prepare_network, "__tnfr_missing_dependency__", {})
        assert prepare_info.get("export") == "prepare_network"
        assert prepare_info.get("missing") == "networkx"
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


def test_public_api_circular_import_from_message(monkeypatch):
    real_import_module = importlib.import_module

    def fail_structural(name, package=None):
        if package == "tnfr" and name == ".structural":
            err = ImportError(
                "cannot import name 'create_nfr' from partially initialized module "
                "'tnfr.structural' (most likely due to a circular import)"
            )
            err.name = "create_nfr"
            raise err
        if package is None and name == "tnfr.structural":
            err = ImportError(
                "cannot import name 'create_nfr' from partially initialized module "
                "'tnfr.structural' (most likely due to a circular import)"
            )
            err.name = "create_nfr"
            raise err
        return real_import_module(name, package)

    with monkeypatch.context() as m:
        _clear_tnfr_modules()
        m.setattr(importlib, "import_module", fail_structural)
        with pytest.raises(ImportError) as excinfo:
            importlib.import_module("tnfr")
        assert "partially initialized module 'tnfr.structural'" in str(excinfo.value)
        _clear_tnfr_modules()
    globals()["tnfr"] = importlib.import_module("tnfr")
