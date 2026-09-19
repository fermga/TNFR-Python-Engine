"""Shared factor-domain integration for the U2 policy table."""

from __future__ import annotations

import pytest

from tnfr.operators import factor_contracts
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.physics import lyapunov


def _defaults_with(**changes: float) -> dict[str, float]:
    defaults = factor_contracts.canonical_glyph_factor_defaults()
    defaults.update(changes)
    return defaults


def _rebuild_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
    defaults: dict[str, float],
) -> dict[str, lyapunov.OperatorLyapunovBound]:
    monkeypatch.setattr(
        factor_contracts,
        "canonical_glyph_factor_defaults",
        lambda: defaults,
    )
    return lyapunov._build_bounds()


def test_policy_representatives_match_shared_registry_and_operator() -> None:
    defaults = factor_contracts.canonical_glyph_factor_defaults()
    for policy in lyapunov.OPERATOR_POLICY_MULTIPLIERS.values():
        spec = factor_contracts.GLYPH_FACTOR_SPECS[policy.glyph_factor_name]
        assert spec.has_canonical_default
        assert spec.glyph.value == policy.glyph
        assert policy.glyph_factor_value == factor_contracts.validate_glyph_factor(
            policy.glyph_factor_name,
            defaults[policy.glyph_factor_name],
        )


def test_policy_rebuild_reads_shared_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebuilt = _rebuild_with_defaults(
        monkeypatch,
        _defaults_with(IL_dnfr_factor=0.5),
    )
    assert rebuilt["Coherence"].glyph_factor_value == 0.5
    assert rebuilt["Coherence"].policy_multiplier == 0.5


@pytest.mark.parametrize(
    ("factor_name", "invalid_value"),
    [
        ("AL_boost", 0.0),
        ("EN_mix", 0.0),
        ("IL_dnfr_factor", 1.0),
        ("OZ_dnfr_factor", 1.0),
        ("UM_theta_push", 0.0),
        ("UM_vf_sync", 0.0),
        ("UM_dnfr_reduction", 0.0),
        ("RA_epi_diff", 0.0),
        ("RA_phase_coupling", 0.0),
        ("SHA_vf_factor", 1.0),
        ("VAL_scale", 1.0),
        ("NUL_scale", 1.0),
        ("NUL_densification_factor", 1.0),
        ("THOL_accel", 0.0),
        ("ZHIR_theta_shift_factor", 0.0),
        ("ZHIR_theta_shift_factor", 8.0),
        ("REMESH_alpha", 0.0),
    ],
)
def test_policy_rebuild_rejects_strict_domain_violations(
    monkeypatch: pytest.MonkeyPatch,
    factor_name: str,
    invalid_value: float,
) -> None:
    defaults = _defaults_with(**{factor_name: invalid_value})
    with pytest.raises(GlyphFactorValidationError, match=factor_name):
        _rebuild_with_defaults(monkeypatch, defaults)


def test_policy_rebuild_validates_nonrepresentative_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = _defaults_with(RA_vf_amplification=0.0)
    with pytest.raises(
        GlyphFactorValidationError,
        match="RA_vf_amplification",
    ):
        _rebuild_with_defaults(monkeypatch, defaults)


def test_policy_rebuild_enforces_nul_default_coupling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = _defaults_with(
        NUL_scale=0.8,
        NUL_densification_factor=1.2,
    )
    with pytest.raises(
        GlyphFactorValidationError,
        match="NUL_densification_factor is derived",
    ):
        _rebuild_with_defaults(monkeypatch, defaults)


def test_policy_rebuild_preserves_declared_nav_zero_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebuilt = _rebuild_with_defaults(
        monkeypatch,
        _defaults_with(NAV_eta=0.0, NAV_jitter=0.0),
    )
    transition = rebuilt["Transition"]
    assert transition.glyph_factor_value == 0.0
    assert transition.policy_multiplier == 1.0
