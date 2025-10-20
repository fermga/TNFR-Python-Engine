"""Ensure the exported dependency manifest stays in sync with helpers."""

from __future__ import annotations

import pytest


def test_prepare_network_dependencies():
    from tnfr import EXPORT_DEPENDENCIES

    expected = {
        "tnfr.ontosim",
        "tnfr.callback_utils",
        "tnfr.constants",
        "tnfr.dynamics",
        "tnfr.glyph_history",
        "tnfr.initialization",
        "tnfr.utils",
    }

    preparar = EXPORT_DEPENDENCIES["prepare_network"]
    assert set(preparar["submodules"]) == expected
    assert preparar["third_party"] == ("networkx",)


def test_preparar_red_alias_dependencies():
    from tnfr import EXPORT_DEPENDENCIES

    assert EXPORT_DEPENDENCIES["preparar_red"] == EXPORT_DEPENDENCIES["prepare_network"]


def test_dynamics_helpers_dependencies():
    from tnfr import EXPORT_DEPENDENCIES

    expected_submodules = {"tnfr.dynamics"}
    expected_third_party = ("networkx",)

    for helper in ("step", "run"):
        deps = EXPORT_DEPENDENCIES[helper]
        assert set(deps["submodules"]) == expected_submodules
        assert deps["third_party"] == expected_third_party


def test_structural_helpers_dependencies():
    from tnfr import EXPORT_DEPENDENCIES

    expected_submodules = {
        "tnfr.structural",
        "tnfr.constants",
        "tnfr.dynamics",
        "tnfr.operators.definitions",
        "tnfr.operators.registry",
        "tnfr.validation",
    }
    expected_third_party = ("networkx",)

    for helper in ("create_nfr", "run_sequence"):
        deps = EXPORT_DEPENDENCIES[helper]
        assert set(deps["submodules"]) == expected_submodules
        assert deps["third_party"] == expected_third_party


def test_manifest_validator_catches_mismatches(monkeypatch):
    import tnfr

    broken_manifest = dict(tnfr.EXPORT_DEPENDENCIES)
    broken_manifest.pop("run")
    monkeypatch.setattr(tnfr, "EXPORT_DEPENDENCIES", broken_manifest)

    with pytest.raises(tnfr.ExportDependencyError, match="run"):
        tnfr._validate_export_dependencies()
