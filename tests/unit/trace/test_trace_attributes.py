"""Tests for trace attribute presence and immutability guarantees."""

from types import MappingProxyType

import pytest

from tnfr.trace import (
    TraceMetadata,
    TraceSnapshot,
    callbacks_field,
    dnfr_weights_field,
    gamma_field,
    grammar_field,
    glyph_counts_field,
    kuramoto_field,
    mapping_field,
    selector_field,
    si_weights_field,
    sigma_field,
    thol_state_field,
)
from tnfr.types import SigmaVector


class TestTraceFieldAttributesAndImmutability:
    """Test that trace field producers return documented structures."""

    def test_gamma_field_returns_immutable_mapping(self, graph_canon):
        """gamma_field must return MappingProxyType to prevent mutation."""
        G = graph_canon()
        G.graph["GAMMA"] = {"test": 1}
        
        result = gamma_field(G)
        assert "gamma" in result
        assert isinstance(result["gamma"], MappingProxyType)
        
        # Verify immutability
        with pytest.raises(TypeError):
            result["gamma"]["new_key"] = 2

    def test_grammar_field_returns_immutable_mapping(self, graph_canon):
        """grammar_field must return MappingProxyType to prevent mutation."""
        G = graph_canon()
        G.graph["GRAMMAR_CANON"] = {"test": 1}
        
        result = grammar_field(G)
        assert "grammar" in result
        assert isinstance(result["grammar"], MappingProxyType)
        
        # Verify immutability
        with pytest.raises(TypeError):
            result["grammar"]["new_key"] = 2

    def test_dnfr_weights_field_returns_immutable_mapping(self, graph_canon):
        """dnfr_weights_field must return MappingProxyType."""
        G = graph_canon()
        G.graph["DNFR_WEIGHTS"] = {"dnfr": 0.5}
        
        result = dnfr_weights_field(G)
        assert "dnfr_weights" in result
        assert isinstance(result["dnfr_weights"], MappingProxyType)

    def test_si_weights_field_returns_immutable_mappings(self, graph_canon):
        """si_weights_field must return MappingProxyType for weights and sensitivity."""
        G = graph_canon()
        G.graph["_Si_weights"] = {"alpha": 0.3}
        # Use valid sensitivity keys as per sense_index.py
        G.graph["_Si_sensitivity"] = {"dSi_dvf_norm": 0.5}
        
        result = si_weights_field(G)
        assert "si_weights" in result
        assert "si_sensitivity" in result
        
        # Both should be immutable
        if result["si_weights"]:
            assert isinstance(result["si_weights"], MappingProxyType)
        if result["si_sensitivity"]:
            assert isinstance(result["si_sensitivity"], MappingProxyType)

    def test_mapping_field_returns_immutable_proxy(self, graph_canon):
        """mapping_field must wrap dict values in MappingProxyType."""
        G = graph_canon()
        G.graph["TEST_KEY"] = {"value": 42}
        
        result = mapping_field(G, "TEST_KEY", "output_key")
        assert "output_key" in result
        assert isinstance(result["output_key"], MappingProxyType)
        assert result["output_key"]["value"] == 42
        
        # Verify immutability
        with pytest.raises(TypeError):
            result["output_key"]["value"] = 100

    def test_selector_field_returns_expected_structure(self, graph_canon):
        """selector_field must return dict with 'selector' key."""
        G = graph_canon()
        
        def mock_selector(graph, node):
            return "AL"
        
        mock_selector.__name__ = "mock_selector"
        G.graph["glyph_selector"] = mock_selector
        
        result = selector_field(G)
        assert "selector" in result
        assert result["selector"] == "mock_selector"

    def test_callbacks_field_returns_expected_structure(self, graph_canon):
        """callbacks_field must return dict with 'callbacks' key."""
        from tnfr.utils.callbacks import CallbackSpec
        
        G = graph_canon()
        
        def mock_callback(graph, ctx):
            pass
        
        G.graph["callbacks"] = {
            "before_step": [CallbackSpec("cb1", mock_callback)]
        }
        
        result = callbacks_field(G)
        assert "callbacks" in result
        assert isinstance(result["callbacks"], dict)

    def test_thol_state_field_returns_integer_count(self, graph_canon):
        """thol_state_field must return dict with 'thol_open_nodes' integer."""
        G = graph_canon()
        
        result = thol_state_field(G)
        assert "thol_open_nodes" in result
        assert isinstance(result["thol_open_nodes"], int)
        assert result["thol_open_nodes"] >= 0

    def test_kuramoto_field_returns_expected_structure(self, graph_canon):
        """kuramoto_field must return dict with 'kuramoto' mapping."""
        G = graph_canon()
        
        result = kuramoto_field(G)
        assert "kuramoto" in result
        assert isinstance(result["kuramoto"], dict)
        assert "R" in result["kuramoto"]
        assert "psi" in result["kuramoto"]
        assert isinstance(result["kuramoto"]["R"], float)
        assert isinstance(result["kuramoto"]["psi"], float)

    def test_sigma_field_returns_expected_structure(self, graph_canon):
        """sigma_field must return dict with 'sigma' mapping containing x, y, mag, angle."""
        G = graph_canon()
        
        result = sigma_field(G)
        assert "sigma" in result
        assert isinstance(result["sigma"], dict)
        
        # Verify all required keys from TraceMetadata.sigma
        required_keys = ["x", "y", "mag", "angle"]
        for key in required_keys:
            assert key in result["sigma"], f"Missing required key: {key}"
            assert isinstance(result["sigma"][key], float)

    def test_glyph_counts_field_returns_expected_structure(self, graph_canon):
        """glyph_counts_field must return dict with 'glyphs' mapping."""
        G = graph_canon()
        
        result = glyph_counts_field(G)
        assert "glyphs" in result
        assert isinstance(result["glyphs"], dict)


class TestSigmaVectorCompleteness:
    """Test that SigmaVector creators include all required keys."""

    def test_sigma_fallback_includes_all_required_keys(self):
        """_sigma_fallback must return all required SigmaVector keys."""
        from tnfr.trace import _sigma_fallback
        import networkx as nx
        
        result = _sigma_fallback(nx.Graph())
        
        # Required keys from _SigmaVectorRequired
        required_keys = ["x", "y", "mag", "angle", "n"]
        for key in required_keys:
            assert key in result, f"Missing required SigmaVector key: {key}"
            
    def test_sigma_empty_includes_all_required_keys(self):
        """_empty_sigma must return all required SigmaVector keys."""
        from tnfr.sense import _empty_sigma
        
        result = _empty_sigma(0.0)
        
        required_keys = ["x", "y", "mag", "angle", "n"]
        for key in required_keys:
            assert key in result, f"Missing required SigmaVector key: {key}"


class TestTraceMetadataCompleteness:
    """Test that TraceMetadata is complete and properly typed."""

    def test_trace_metadata_has_all_documented_keys(self):
        """TraceMetadata TypedDict should have all keys used by trace producers."""
        # All keys that trace field producers use
        expected_keys = {
            "gamma",
            "grammar",
            "selector",
            "dnfr_weights",
            "si_weights",
            "si_sensitivity",
            "callbacks",
            "thol_open_nodes",
            "kuramoto",
            "sigma",
            "glyphs",
        }
        
        actual_keys = set(TraceMetadata.__annotations__.keys())
        
        assert expected_keys.issubset(actual_keys), (
            f"TraceMetadata missing keys: {expected_keys - actual_keys}"
        )

    def test_trace_snapshot_extends_trace_metadata(self):
        """TraceSnapshot should have TraceMetadata keys plus t and phase."""
        trace_meta_keys = set(TraceMetadata.__annotations__.keys())
        snapshot_keys = set(TraceSnapshot.__annotations__.keys())
        
        # TraceSnapshot should have all TraceMetadata keys
        assert trace_meta_keys.issubset(snapshot_keys)
        
        # Plus the additional keys
        assert "t" in snapshot_keys
        assert "phase" in snapshot_keys


class TestMappingProxySafeMutation:
    """Test patterns for safely mutating MappingProxyType values."""

    def test_copy_pattern_for_safe_mutation(self, graph_canon):
        """Demonstrate safe pattern: copy immutable mapping to mutate."""
        G = graph_canon()
        G.graph["GAMMA"] = {"existing": 1}
        
        result = gamma_field(G)
        original_proxy = result["gamma"]
        
        # Cannot mutate directly
        with pytest.raises(TypeError):
            original_proxy["new"] = 2
        
        # Safe pattern: create mutable copy
        mutable_copy = dict(original_proxy)
        mutable_copy["new"] = 2
        
        # Original is unchanged
        assert "new" not in original_proxy
        assert "new" in mutable_copy

    def test_merge_pattern_for_combining_mappings(self, graph_canon):
        """Demonstrate safe pattern: merge immutable mappings."""
        G = graph_canon()
        G.graph["GAMMA"] = {"a": 1}
        G.graph["GRAMMAR_CANON"] = {"b": 2}
        
        gamma_result = gamma_field(G)
        grammar_result = grammar_field(G)
        
        # Safe pattern: unpack into new dict
        combined = {
            **gamma_result["gamma"],
            **grammar_result["grammar"],
        }
        
        assert combined == {"a": 1, "b": 2}
        assert isinstance(gamma_result["gamma"], MappingProxyType)
        assert isinstance(grammar_result["grammar"], MappingProxyType)
