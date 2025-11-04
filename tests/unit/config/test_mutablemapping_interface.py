"""Test configuration MutableMapping interface compliance.

This module verifies that configuration values returned by get_param
support the full MutableMapping protocol as documented in the type annotations.
"""

from collections.abc import MutableMapping

import pytest

from tnfr.constants import DEFAULTS, get_param, inject_defaults


def test_get_param_returns_mutable_dict_for_nested_configs(graph_canon):
    """Verify get_param returns dict implementing MutableMapping for nested configs."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Should be a MutableMapping
    assert isinstance(diagnosis_cfg, MutableMapping)
    
    # Should support dict operations
    assert isinstance(diagnosis_cfg, dict)


def test_config_dict_supports_get_method(graph_canon):
    """Verify configuration dicts support .get() with defaults."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Test .get() with existing key
    assert diagnosis_cfg.get("history_key") == "nodal_diag"
    
    # Test .get() with missing key and default
    assert diagnosis_cfg.get("missing_key", "default_value") == "default_value"
    
    # Test .get() with missing key and no default
    assert diagnosis_cfg.get("missing_key") is None


def test_config_dict_supports_setitem(graph_canon):
    """Verify configuration dicts support __setitem__ for mutations."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Original value
    assert diagnosis_cfg["compute_symmetry"] is True
    
    # Mutate via __setitem__
    diagnosis_cfg["compute_symmetry"] = False
    
    # Verify mutation persisted
    assert diagnosis_cfg["compute_symmetry"] is False
    
    # Verify mutation visible through get_param
    refreshed = get_param(G, "DIAGNOSIS")
    assert refreshed["compute_symmetry"] is False


def test_config_dict_supports_update(graph_canon):
    """Verify configuration dicts support .update() for batch mutations."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Original values
    assert diagnosis_cfg["window"] == 16
    assert diagnosis_cfg["enabled"] is True
    
    # Update multiple values
    diagnosis_cfg.update({"window": 32, "enabled": False, "new_key": "new_value"})
    
    # Verify mutations
    assert diagnosis_cfg["window"] == 32
    assert diagnosis_cfg["enabled"] is False
    assert diagnosis_cfg["new_key"] == "new_value"


def test_config_dict_supports_del(graph_canon):
    """Verify configuration dicts support __delitem__ for key removal."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Add a temporary key
    diagnosis_cfg["temp_key"] = "temp_value"
    assert "temp_key" in diagnosis_cfg
    
    # Delete it
    del diagnosis_cfg["temp_key"]
    assert "temp_key" not in diagnosis_cfg


def test_config_dict_supports_pop(graph_canon):
    """Verify configuration dicts support .pop() for key removal with default."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Pop existing key
    value = diagnosis_cfg.pop("window", None)
    assert value == 16
    assert "window" not in diagnosis_cfg
    
    # Pop missing key with default
    value = diagnosis_cfg.pop("missing_key", "default")
    assert value == "default"


def test_config_dict_supports_setdefault(graph_canon):
    """Verify configuration dicts support .setdefault()."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # setdefault on existing key returns existing value
    value = diagnosis_cfg.setdefault("window", 999)
    assert value == 16
    assert diagnosis_cfg["window"] == 16
    
    # setdefault on missing key sets and returns default
    value = diagnosis_cfg.setdefault("new_key", "new_default")
    assert value == "new_default"
    assert diagnosis_cfg["new_key"] == "new_default"


def test_config_mutations_isolated_per_graph(graph_canon):
    """Verify configuration mutations don't affect other graphs or defaults."""
    import networkx as nx
    
    # Create two independent graphs
    G1 = graph_canon()
    G2 = nx.Graph()
    
    inject_defaults(G2)
    
    # Mutate G1's configuration
    cfg1 = get_param(G1, "DIAGNOSIS")
    cfg1["compute_symmetry"] = False
    cfg1["custom_key"] = "G1_value"
    
    # G2's configuration should be unchanged
    cfg2 = get_param(G2, "DIAGNOSIS")
    assert cfg2["compute_symmetry"] is True
    assert "custom_key" not in cfg2
    
    # DEFAULTS should be unchanged
    assert DEFAULTS["DIAGNOSIS"]["compute_symmetry"] is True
    assert "custom_key" not in DEFAULTS["DIAGNOSIS"]


def test_scalar_config_values_remain_immutable(graph_canon):
    """Verify scalar configuration values work correctly."""
    G = graph_canon()
    
    # Scalar values should work
    dt = get_param(G, "DT")
    assert isinstance(dt, (int, float))
    assert dt == 1.0
    
    # String values should work
    method = get_param(G, "INTEGRATOR_METHOD")
    assert isinstance(method, str)
    assert method == "euler"


def test_nested_dict_mutations(graph_canon):
    """Verify nested dictionary mutations work correctly."""
    G = graph_canon()
    
    diagnosis_cfg = get_param(G, "DIAGNOSIS")
    
    # Mutate nested dict
    assert diagnosis_cfg["stable"]["persist"] == 3
    diagnosis_cfg["stable"]["persist"] = 5
    assert diagnosis_cfg["stable"]["persist"] == 5
    
    # Verify mutation persisted
    refreshed = get_param(G, "DIAGNOSIS")
    assert refreshed["stable"]["persist"] == 5


@pytest.mark.parametrize("config_key", [
    "DIAGNOSIS",
    "COHERENCE",
    "SIGMA",
    "TRACE",
    "GRAMMAR",
    "DNFR_WEIGHTS",
    "SI_WEIGHTS",
])
def test_all_dict_configs_are_mutable_mappings(graph_canon, config_key):
    """Verify all dictionary-type configurations support MutableMapping."""
    G = graph_canon()
    
    config = get_param(G, config_key)
    
    if isinstance(config, dict):
        assert isinstance(config, MutableMapping), (
            f"{config_key} should be a MutableMapping"
        )
        
        # Test basic mutation
        config["test_key"] = "test_value"
        assert config["test_key"] == "test_value"
        
        # Test .get()
        assert config.get("test_key") == "test_value"
