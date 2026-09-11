"""Runtime domains for canonical glyph factors."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.config import TNFRConfig, TNFRConfigError, merge_overrides
from tnfr.operators.factor_contracts import (
    GLYPH_FACTOR_SPECS,
    GLYPH_FACTORS_BY_GLYPH,
    GlyphFactorValidationError,
    canonical_glyph_factor_defaults,
    resolve_operator_factors,
    validate_glyph_factor,
    validate_glyph_factors,
)
from tnfr.types import Glyph
from tnfr.validation.input_validation import ValidationError
from tnfr.validation.input_validation import (
    validate_glyph_factors as validate_input_glyph_factors,
)


def test_registry_covers_the_single_canonical_default_table():
    defaults = canonical_glyph_factor_defaults()
    registered_defaults = {
        key for key, spec in GLYPH_FACTOR_SPECS.items() if spec.has_canonical_default
    }

    assert registered_defaults == set(defaults)
    assert {
        key: validate_glyph_factor(key, value) for key, value in defaults.items()
    } == defaults


def test_registry_is_immutable():
    spec = GLYPH_FACTOR_SPECS["AL_boost"]
    with pytest.raises(TypeError):
        GLYPH_FACTOR_SPECS["extension_gain"] = spec  # type: ignore[index]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AL_boost", 0.0),
        ("AL_boost", -0.1),
        ("EN_mix", 0.0),
        ("EN_mix", -0.1),
        ("EN_mix", 1.1),
        ("IL_dnfr_factor", -0.1),
        ("IL_dnfr_factor", 1.0),
        ("IL_dnfr_factor", 1.1),
        ("OZ_dnfr_factor", 0.9),
        ("OZ_dnfr_factor", 1.0),
        ("UM_theta_push", 0.0),
        ("UM_theta_push", -0.1),
        ("UM_theta_push", 1.1),
        ("UM_vf_sync", 0.0),
        ("UM_vf_sync", -0.1),
        ("UM_vf_sync", 1.1),
        ("UM_dnfr_reduction", 0.0),
        ("UM_dnfr_reduction", -0.1),
        ("UM_dnfr_reduction", 1.1),
        ("RA_epi_diff", 0.0),
        ("RA_epi_diff", -0.1),
        ("RA_epi_diff", 1.1),
        ("RA_vf_amplification", 0.0),
        ("RA_vf_amplification", -0.1),
        ("RA_phase_coupling", 0.0),
        ("RA_phase_coupling", -0.1),
        ("RA_phase_coupling", 1.1),
        ("SHA_vf_factor", -0.1),
        ("SHA_vf_factor", 1.0),
        ("SHA_vf_factor", 1.1),
        ("VAL_scale", 0.9),
        ("VAL_scale", 1.0),
        ("NUL_scale", 0.0),
        ("NUL_scale", 1.0),
        ("NUL_scale", 1.1),
        ("NUL_densification_factor", 0.9),
        ("NUL_densification_factor", 1.0),
        ("THOL_accel", 0.0),
        ("THOL_accel", -0.1),
        ("ZHIR_theta_shift_factor", 0.0),
        ("ZHIR_theta_shift_factor", 8.0),
        ("ZHIR_theta_shift", 0.0),
        ("ZHIR_theta_shift", 2.0 * math.pi),
        ("NAV_jitter", -0.1),
        ("NAV_eta", -0.1),
        ("NAV_eta", 1.1),
        ("REMESH_alpha", -0.1),
        ("REMESH_alpha", 0.0),
        ("REMESH_alpha", 1.1),
    ],
)
def test_contract_reversing_or_noop_values_are_rejected(key, value):
    with pytest.raises(GlyphFactorValidationError, match=key):
        validate_glyph_factor(key, value)


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", float("nan"), float("inf"), float("-inf"), 10**400],
)
def test_known_factors_require_representable_finite_real_scalars(value):
    with pytest.raises(GlyphFactorValidationError, match="EN_mix"):
        validate_glyph_factor("EN_mix", value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AL_boost", math.nextafter(0.0, math.inf)),
        ("EN_mix", 1.0),
        ("IL_dnfr_factor", 0.0),
        ("OZ_dnfr_factor", math.nextafter(1.0, math.inf)),
        ("UM_theta_push", 1.0),
        ("UM_vf_sync", 1.0),
        ("UM_dnfr_reduction", 1.0),
        ("RA_epi_diff", 1.0),
        ("RA_vf_amplification", math.nextafter(0.0, math.inf)),
        ("RA_phase_coupling", 1.0),
        ("SHA_vf_factor", 0.0),
        ("VAL_scale", math.nextafter(1.0, math.inf)),
        ("NUL_scale", math.nextafter(1.0, 0.0)),
        ("NUL_densification_factor", math.nextafter(1.0, math.inf)),
        ("THOL_accel", math.nextafter(0.0, math.inf)),
        ("NAV_jitter", 0.0),
        ("NAV_eta", 0.0),
        ("NAV_eta", 1.0),
        ("REMESH_alpha", 1.0),
    ],
)
def test_directional_and_convex_boundaries_are_explicit(key, value):
    assert validate_glyph_factor(key, value) == value


@pytest.mark.parametrize(
    ("display_name", "glyph"),
    [
        ("Emission", Glyph.AL),
        ("Reception", Glyph.EN),
        ("Coherence", Glyph.IL),
        ("Dissonance", Glyph.OZ),
        ("Coupling", Glyph.UM),
        ("Resonance", Glyph.RA),
        ("Silence", Glyph.SHA),
        ("Expansion", Glyph.VAL),
        ("Contraction", Glyph.NUL),
        ("SelfOrganization", Glyph.THOL),
        ("Mutation", Glyph.ZHIR),
        ("Transition", Glyph.NAV),
        ("Recursivity", Glyph.REMESH),
    ],
)
def test_title_case_display_names_are_valid_contexts(display_name, glyph):
    defaults = canonical_glyph_factor_defaults()
    key = next(
        key
        for key in GLYPH_FACTORS_BY_GLYPH[glyph]
        if key in defaults
    )

    assert validate_glyph_factors(
        {key: defaults[key]}, glyph=display_name
    )[key] == defaults[key]


def test_unknown_extension_factors_are_preserved_without_coercion():
    source = {"AL_boost": 0.2, "extension_gain": "extension-owned"}

    validated = validate_glyph_factors(source)

    assert validated == source
    assert validated is not source


def test_context_validation_defers_unrelated_known_factors():
    factors = {
        "AL_boost": 0.2,
        "RA_epi_diff": 2.0,
        "extension_gain": object(),
    }

    validated = validate_glyph_factors(factors, glyph=Glyph.AL)

    assert validated["RA_epi_diff"] == 2.0
    assert validated["extension_gain"] is factors["extension_gain"]
    with pytest.raises(GlyphFactorValidationError, match="RA_epi_diff"):
        validate_glyph_factors(factors, glyph=Glyph.RA)


def test_nul_scale_override_derives_its_inverse_densification():
    resolved = resolve_operator_factors({"NUL_scale": 0.5}, Glyph.NUL)

    assert resolved["NUL_scale"] == 0.5
    assert resolved["NUL_densification_factor"] == 2.0


def test_nul_accepts_an_explicit_materialized_inverse():
    resolved = resolve_operator_factors(
        {"NUL_scale": 0.5, "NUL_densification_factor": 2.0}, Glyph.NUL
    )

    assert resolved["NUL_densification_factor"] == 2.0


def test_nul_rejects_independent_densification():
    with pytest.raises(GlyphFactorValidationError, match="derived, not independent"):
        resolve_operator_factors(
            {"NUL_scale": 0.5, "NUL_densification_factor": 3.0}, Glyph.NUL
        )


def test_input_validation_shim_delegates_to_the_canonical_registry():
    factors = {"custom_factor": "opaque", "EN_mix": 0.25}

    assert validate_input_glyph_factors(factors) == factors
    with pytest.raises(ValidationError, match="EN_mix"):
        validate_input_glyph_factors({"EN_mix": 2.0})


def test_complete_config_validates_known_factors_but_preserves_extensions():
    config = TNFRConfig()

    assert config.validate_config(
        {"GLYPH_FACTORS": {"EN_mix": 0.25, "custom_factor": "opaque"}}
    )
    with pytest.raises(TNFRConfigError, match="IL_dnfr_factor"):
        config.validate_config({"GLYPH_FACTORS": {"IL_dnfr_factor": 2.0}})
    with pytest.raises(TNFRConfigError, match="REMESH_alpha"):
        config.validate_config({"REMESH_ALPHA": float("inf")})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AL_boost", 0.0),
        ("OZ_dnfr_factor", 1.0),
        ("RA_vf_amplification", 0.0),
        ("SHA_vf_factor", 1.0),
        ("VAL_scale", 1.0),
        ("NUL_scale", 1.0),
        ("REMESH_alpha", 0.0),
    ],
)
def test_complete_config_rejects_noop_directional_factors(key, value):
    with pytest.raises(TNFRConfigError, match=key):
        TNFRConfig().validate_config({"GLYPH_FACTORS": {key: value}})


def test_invalid_factor_override_is_rejected_before_graph_mutation():
    graph = nx.Graph(GLYPH_FACTORS={"EN_mix": 0.25})
    before = deepcopy(graph.graph)

    with pytest.raises(TNFRConfigError, match="EN_mix"):
        merge_overrides(graph, GLYPH_FACTORS={"EN_mix": 2.0})

    assert graph.graph == before
