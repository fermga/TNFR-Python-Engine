"""The retired lab requires an explicitly supplied external compatibility package."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest

import tnfr.factorization as bridge
from tnfr.errors import TNFRUserError


@pytest.fixture(autouse=True)
def isolate_bridge(monkeypatch):
    monkeypatch.setattr(bridge, "_DEFAULT_FACTORIZER", None)
    for name in ("SpectralPaleyFactorizer", "SpectralAnalysisResult"):
        monkeypatch.delitem(vars(bridge), name, raising=False)
    for name in tuple(sys.modules):
        if name == "tnfr_factorization" or name.startswith("tnfr_factorization."):
            monkeypatch.delitem(sys.modules, name)
    yield
    for name in ("SpectralPaleyFactorizer", "SpectralAnalysisResult"):
        vars(bridge).pop(name, None)
    for name in tuple(sys.modules):
        if name == "tnfr_factorization" or name.startswith("tnfr_factorization."):
            sys.modules.pop(name, None)


def test_missing_external_package_does_not_discover_archived_checkout(monkeypatch):
    original = builtins.__import__

    def absent(name, *args, **kwargs):
        if name.startswith("tnfr_factorization"):
            raise ModuleNotFoundError(name, name="tnfr_factorization")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)
    before = tuple(sys.path)
    with pytest.raises(TNFRUserError, match="archived.*no longer"):
        bridge.factorize(35)
    assert tuple(sys.path) == before
    assert not any(name.startswith("tnfr_factorization") for name in sys.modules)


def test_archived_source_on_search_path_is_rejected_before_import(monkeypatch):
    archive = Path(__file__).resolve().parents[1] / "applications/factorization-lab"
    monkeypatch.syspath_prepend(str(archive))
    with pytest.raises(TNFRUserError, match="archived.*no longer"):
        bridge.factorize(35)
    assert not any(name.startswith("tnfr_factorization") for name in sys.modules)


@pytest.fixture
def external_package(tmp_path, monkeypatch):
    package = tmp_path / "tnfr_factorization"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "spectral_paley.py").write_text(
        """
class SpectralAnalysisResult:
    def __init__(self, n, options):
        self.n = n
        self.options = options

class SpectralPaleyFactorizer:
    instances = 0
    def __init__(self):
        type(self).instances += 1
    def analyze(self, n, **options):
        if n < 2:
            raise ValueError("external domain")
        return SpectralAnalysisResult(n, options)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return importlib.import_module("tnfr_factorization.spectral_paley")


def test_explicit_external_bridge_warns_and_preserves_options(
    external_package, tmp_path
):
    with pytest.warns(DeprecationWarning, match="external legacy lab"):
        result = bridge.factorize(
            35, modulus=41, trace_certificates=True, certificate_dir=tmp_path
        )
    assert isinstance(result, external_package.SpectralAnalysisResult)
    assert result.n == 35
    assert result.options == {
        "modulus": 41,
        "trace_certificates": True,
        "certificate_dir": tmp_path,
    }
    second = bridge.factorize(77)
    assert second.n == 77
    assert external_package.SpectralPaleyFactorizer.instances == 1
    with pytest.raises(ValueError, match="external domain"):
        bridge.factorize(1)


def test_lazy_type_exports_keep_external_identity(external_package):
    with pytest.warns(DeprecationWarning, match="external legacy lab"):
        factorizer = bridge.SpectralPaleyFactorizer
    assert factorizer is external_package.SpectralPaleyFactorizer
    assert bridge.SpectralAnalysisResult is external_package.SpectralAnalysisResult
    with pytest.raises(AttributeError):
        getattr(bridge, "not_a_public_type")


def test_cached_archived_submodule_is_rejected_with_external_parent(
    external_package, monkeypatch
):
    archive = Path(__file__).resolve().parents[1] / "applications/factorization-lab"
    monkeypatch.setattr(
        external_package,
        "__file__",
        str(archive / "tnfr_factorization" / "spectral_paley.py"),
    )
    with pytest.raises(TNFRUserError, match="archived.*no longer"):
        bridge.factorize(35)
    assert external_package.SpectralPaleyFactorizer.instances == 0


def test_external_internal_dependency_failure_is_not_reported_as_retirement(
    monkeypatch,
):
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith("tnfr_factorization"):
            raise ModuleNotFoundError("missing external helper", name="external_helper")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModuleNotFoundError) as caught:
        bridge.factorize(35)
    assert caught.value.name == "external_helper"
