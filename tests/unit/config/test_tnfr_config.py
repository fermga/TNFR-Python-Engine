"""Tests for TNFRConfig class with structural invariant validation."""

import math

import pytest

from tnfr.config import TNFRConfig, TNFRConfigError


class TestTNFRConfigValidation:
    """Test TNFR structural invariant validation."""

    def test_validate_vf_bounds_accepts_valid_values(self):
        """Test that valid νf bounds are accepted."""
        config = TNFRConfig()
        assert config.validate_vf_bounds(vf_min=0.0, vf_max=10.0) is True
        assert config.validate_vf_bounds(vf=5.0, vf_min=0.0, vf_max=10.0) is True

    def test_validate_vf_bounds_rejects_negative_min(self):
        """Test that negative VF_MIN is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="VF_MIN must be >= 0"):
            config.validate_vf_bounds(vf_min=-1.0)

    def test_validate_vf_bounds_rejects_max_less_than_min(self):
        """Test that VF_MAX < VF_MIN is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="VF_MAX.*must be >= VF_MIN"):
            config.validate_vf_bounds(vf_min=10.0, vf_max=5.0)

    def test_validate_vf_bounds_rejects_negative_vf(self):
        """Test that negative νf is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="νf must be >= 0"):
            config.validate_vf_bounds(vf=-1.0)

    def test_validate_vf_bounds_rejects_vf_below_min(self):
        """Test that νf below VF_MIN is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="below VF_MIN"):
            config.validate_vf_bounds(vf=0.5, vf_min=1.0)

    def test_validate_vf_bounds_rejects_vf_above_max(self):
        """Test that νf above VF_MAX is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="above VF_MAX"):
            config.validate_vf_bounds(vf=15.0, vf_max=10.0)

    def test_validate_theta_bounds_accepts_valid_phase(self):
        """Test that valid θ (phase) is accepted."""
        config = TNFRConfig()
        assert config.validate_theta_bounds(theta=0.0) is True
        assert config.validate_theta_bounds(theta=math.pi) is True
        assert config.validate_theta_bounds(theta=-math.pi) is True

    def test_validate_theta_bounds_rejects_out_of_range_without_wrap(self):
        """Test that θ outside [-π, π] is rejected when THETA_WRAP=False."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="θ.*must be in"):
            config.validate_theta_bounds(theta=2 * math.pi, theta_wrap=False)
        with pytest.raises(TNFRConfigError, match="θ.*must be in"):
            config.validate_theta_bounds(theta=-2 * math.pi, theta_wrap=False)

    def test_validate_theta_bounds_allows_any_value_with_wrap(self):
        """Test that any θ is accepted when THETA_WRAP=True."""
        config = TNFRConfig()
        # With wrapping enabled, any value should be valid
        assert config.validate_theta_bounds(theta=10 * math.pi, theta_wrap=True) is True

    def test_validate_epi_bounds_accepts_valid_values(self):
        """Test that valid EPI bounds are accepted."""
        config = TNFRConfig()
        assert config.validate_epi_bounds(epi_min=-1.0, epi_max=1.0) is True
        assert config.validate_epi_bounds(epi=0.5, epi_min=-1.0, epi_max=1.0) is True

    def test_validate_epi_bounds_rejects_max_less_than_min(self):
        """Test that EPI_MAX < EPI_MIN is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="EPI_MAX.*must be >= EPI_MIN"):
            config.validate_epi_bounds(epi_min=1.0, epi_max=-1.0)

    def test_validate_epi_bounds_rejects_epi_below_min(self):
        """Test that EPI below EPI_MIN is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="below EPI_MIN"):
            config.validate_epi_bounds(epi=-2.0, epi_min=-1.0)

    def test_validate_epi_bounds_rejects_epi_above_max(self):
        """Test that EPI above EPI_MAX is rejected."""
        config = TNFRConfig()
        with pytest.raises(TNFRConfigError, match="above EPI_MAX"):
            config.validate_epi_bounds(epi=2.0, epi_max=1.0)

    def test_validate_dnfr_semantics_accepts_any_value(self):
        """Test that ΔNFR accepts any real value (no numeric bounds)."""
        config = TNFRConfig()
        assert config.validate_dnfr_semantics(dnfr=0.0) is True
        assert config.validate_dnfr_semantics(dnfr=1.0) is True
        assert config.validate_dnfr_semantics(dnfr=-1.0) is True
        assert config.validate_dnfr_semantics(dnfr=1000.0) is True

    def test_validate_config_checks_all_bounds(self):
        """Test that validate_config checks all structural invariants."""
        config = TNFRConfig()

        valid_config = {
            "VF_MIN": 0.0,
            "VF_MAX": 10.0,
            "EPI_MIN": -1.0,
            "EPI_MAX": 1.0,
            "THETA_WRAP": True,
            "DT": 1.0,
        }
        assert config.validate_config(valid_config) is True

    def test_validate_config_rejects_invalid_vf_bounds(self):
        """Test that validate_config rejects invalid νf bounds."""
        config = TNFRConfig()

        invalid_config = {
            "VF_MIN": -1.0,  # Invalid: negative
            "VF_MAX": 10.0,
        }
        with pytest.raises(TNFRConfigError):
            config.validate_config(invalid_config)

    def test_validate_config_rejects_invalid_epi_bounds(self):
        """Test that validate_config rejects invalid EPI bounds."""
        config = TNFRConfig()

        invalid_config = {
            "EPI_MIN": 1.0,
            "EPI_MAX": -1.0,  # Invalid: max < min
        }
        with pytest.raises(TNFRConfigError):
            config.validate_config(invalid_config)

    def test_validate_config_rejects_non_positive_dt(self):
        """Test that validate_config rejects non-positive DT."""
        config = TNFRConfig()

        invalid_config = {
            "DT": 0.0,  # Invalid: must be > 0
        }
        with pytest.raises(TNFRConfigError, match="DT.*must be > 0"):
            config.validate_config(invalid_config)

        invalid_config = {
            "DT": -1.0,  # Invalid: negative
        }
        with pytest.raises(TNFRConfigError, match="DT.*must be > 0"):
            config.validate_config(invalid_config)

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled."""
        config = TNFRConfig(validate_invariants=False)

        # These should all pass with validation disabled
        assert config.validate_vf_bounds(vf_min=-10.0) is True
        assert config.validate_theta_bounds(theta=100.0, theta_wrap=False) is True
        assert config.validate_epi_bounds(epi_min=10.0, epi_max=-10.0) is True


class TestTNFRConfigUsage:
    """Test TNFRConfig usage patterns."""

    def test_create_with_defaults(self):
        """Test creating TNFRConfig with defaults."""
        from tnfr.config import DEFAULTS

        config = TNFRConfig(defaults=DEFAULTS)
        assert config._defaults == DEFAULTS

    def test_get_param_with_fallback(self):
        """Test get_param_with_fallback method."""
        defaults = {"DT": 1.0, "VF_MIN": 0.0}
        config = TNFRConfig(defaults=defaults)

        # Get from graph
        G_graph = {"DT": 0.5}
        assert config.get_param_with_fallback(G_graph, "DT") == 0.5

        # Get from defaults
        assert config.get_param_with_fallback(G_graph, "VF_MIN") == 0.0

        # Get with fallback
        assert config.get_param_with_fallback(G_graph, "UNKNOWN", default=42) == 42

    def test_get_param_with_fallback_raises_on_missing(self):
        """Test that get_param_with_fallback raises on missing key."""
        config = TNFRConfig(defaults={})

        with pytest.raises(KeyError, match="not found"):
            config.get_param_with_fallback({}, "MISSING")

    def test_get_param_with_fallback_deep_copies_mutable(self):
        """Test that mutable defaults are deep copied."""
        defaults = {"MUTABLE_DICT": {"key": "value"}}
        config = TNFRConfig(defaults=defaults)

        result1 = config.get_param_with_fallback({}, "MUTABLE_DICT")
        result2 = config.get_param_with_fallback({}, "MUTABLE_DICT")

        # Should be different objects (deep copied)
        assert result1 is not result2
        assert result1 == result2

    def test_inject_defaults_validates_config(self):
        """Test that inject_defaults validates configuration."""
        import networkx as nx

        config = TNFRConfig(validate_invariants=True)
        G = nx.Graph()

        valid_defaults = {"DT": 1.0, "VF_MIN": 0.0}
        config.inject_defaults(G, defaults=valid_defaults)
        assert G.graph["DT"] == 1.0
        assert G.graph["VF_MIN"] == 0.0

    def test_inject_defaults_rejects_invalid_config(self):
        """Test that inject_defaults rejects invalid configuration."""
        import networkx as nx

        config = TNFRConfig(validate_invariants=True)
        G = nx.Graph()

        invalid_defaults = {"VF_MIN": -1.0}  # Invalid
        with pytest.raises(TNFRConfigError):
            config.inject_defaults(G, defaults=invalid_defaults)

    def test_inject_defaults_with_validation_disabled(self):
        """Test inject_defaults with validation disabled."""
        import networkx as nx

        config = TNFRConfig(validate_invariants=False)
        G = nx.Graph()

        # Should accept even invalid config
        invalid_defaults = {"VF_MIN": -1.0}
        config.inject_defaults(G, defaults=invalid_defaults)
        assert G.graph["VF_MIN"] == -1.0


class TestTNFRConfigAliases:
    """Test TNFR semantic aliases."""

    def test_get_aliases_returns_correct_aliases(self):
        """Test that get_aliases returns correct alias tuples."""
        from tnfr.config import get_aliases

        vf_aliases = get_aliases("VF")
        assert "νf" in vf_aliases
        assert "nu_f" in vf_aliases
        assert "frequency" in vf_aliases

    def test_primary_aliases_are_unicode(self):
        """Test that primary aliases use Unicode symbols."""
        from tnfr.config import VF_PRIMARY, THETA_PRIMARY, DNFR_PRIMARY

        assert VF_PRIMARY == "νf"
        assert THETA_PRIMARY == "theta"
        assert DNFR_PRIMARY == "ΔNFR"

    def test_all_expected_aliases_exist(self):
        """Test that all expected TNFR aliases are defined."""
        from tnfr.config import (
            D2EPI_PRIMARY,
            D2VF_PRIMARY,
            DNFR_PRIMARY,
            EPI_KIND_PRIMARY,
            EPI_PRIMARY,
            SI_PRIMARY,
            THETA_PRIMARY,
            VF_PRIMARY,
            dEPI_PRIMARY,
            dSI_PRIMARY,
            dVF_PRIMARY,
        )

        # Just verify they all exist and are strings
        assert isinstance(VF_PRIMARY, str)
        assert isinstance(THETA_PRIMARY, str)
        assert isinstance(DNFR_PRIMARY, str)
        assert isinstance(EPI_PRIMARY, str)
        assert isinstance(EPI_KIND_PRIMARY, str)
        assert isinstance(SI_PRIMARY, str)
        assert isinstance(dEPI_PRIMARY, str)
        assert isinstance(D2EPI_PRIMARY, str)
        assert isinstance(dVF_PRIMARY, str)
        assert isinstance(D2VF_PRIMARY, str)
        assert isinstance(dSI_PRIMARY, str)


class TestTNFRConfigStateTokens:
    """Test TNFR canonical state tokens."""

    def test_normalise_state_token_accepts_canonical(self):
        """Test that canonical state tokens are accepted."""
        from tnfr.config import normalise_state_token

        assert normalise_state_token("stable") == "stable"
        assert normalise_state_token("transition") == "transition"
        assert normalise_state_token("dissonant") == "dissonant"

    def test_normalise_state_token_handles_case(self):
        """Test that state token normalization handles case."""
        from tnfr.config import normalise_state_token

        assert normalise_state_token("STABLE") == "stable"
        assert normalise_state_token("Transition") == "transition"

    def test_normalise_state_token_rejects_invalid(self):
        """Test that invalid state tokens are rejected."""
        from tnfr.config import normalise_state_token

        with pytest.raises(ValueError, match="must be one of"):
            normalise_state_token("invalid")

        with pytest.raises(TypeError):
            normalise_state_token(123)  # type: ignore

    def test_canonical_state_tokens_is_frozenset(self):
        """Test that CANONICAL_STATE_TOKENS is a frozenset."""
        from tnfr.config import CANONICAL_STATE_TOKENS

        assert isinstance(CANONICAL_STATE_TOKENS, frozenset)
        assert "stable" in CANONICAL_STATE_TOKENS
        assert "transition" in CANONICAL_STATE_TOKENS
        assert "dissonant" in CANONICAL_STATE_TOKENS
