"""Canonical runtime domains for TNFR glyph factors.

The numerical defaults remain owned by :mod:`tnfr.config.defaults_core`.  This
module records only the domains required by the operator formulas and declared
contracts, then resolves graph overrides against those defaults.  Unknown keys
are retained so external operator extensions can share ``GLYPH_FACTORS``
without being coupled to the canonical registry.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any

from ..types import Glyph

__all__ = (
    "GLYPH_FACTOR_SPECS",
    "GLYPH_FACTORS_BY_GLYPH",
    "GlyphFactorSpec",
    "GlyphFactorValidationError",
    "canonical_glyph_factor_defaults",
    "runtime_active_glyph_factor_keys",
    "resolve_operator_factors",
    "resolve_runtime_operator_factors",
    "validate_glyph_factor",
    "validate_glyph_factors",
)


class GlyphFactorValidationError(ValueError):
    """Raised when a canonical glyph factor violates its runtime domain."""


@dataclass(frozen=True, slots=True)
class GlyphFactorSpec:
    """Immutable interval contract for one operator factor.

    ``has_canonical_default`` is false only for compatibility parameters that
    are recognized at runtime but are intentionally absent from the canonical
    default table.
    """

    glyph: Glyph
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    has_canonical_default: bool = True
    nonzero_modulus: float | None = None

    def contains(self, value: float) -> bool:
        """Return whether ``value`` satisfies this factor's hard domain."""
        if self.lower is not None:
            if value < self.lower or (
                value == self.lower and not self.lower_inclusive
            ):
                return False
        if self.upper is not None:
            if value > self.upper or (
                value == self.upper and not self.upper_inclusive
            ):
                return False
        if self.nonzero_modulus is not None:
            if math.fmod(value, self.nonzero_modulus) == 0.0:
                return False
        return True

    def interval_text(self) -> str:
        """Return a compact description used by validation errors."""
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        lower = "-inf" if self.lower is None else str(self.lower)
        upper = "inf" if self.upper is None else str(self.upper)
        interval = f"{left}{lower}, {upper}{right}"
        if self.nonzero_modulus is not None:
            interval += f" excluding multiples of {self.nonzero_modulus}"
        return interval


# Directional postconditions are deliberately non-strict because a zero source,
# an equilibrium state, or clipping can make a valid operator step stationary.
# That tolerance does not make a zero request gain canonical: where a factor is
# the sole coefficient of the named effect, its domain is strict at the no-op
# endpoint. Explicit graph flags remain the supported way to disable optional
# UM channels.
_SPECS = {
    "AL_boost": GlyphFactorSpec(
        Glyph.AL, lower=0.0, lower_inclusive=False
    ),
    "EN_mix": GlyphFactorSpec(
        Glyph.EN, lower=0.0, upper=1.0, lower_inclusive=False
    ),
    "IL_dnfr_factor": GlyphFactorSpec(
        Glyph.IL, lower=0.0, upper=1.0, upper_inclusive=False
    ),
    "OZ_dnfr_factor": GlyphFactorSpec(
        Glyph.OZ, lower=1.0, lower_inclusive=False
    ),
    "UM_theta_push": GlyphFactorSpec(
        Glyph.UM, lower=0.0, upper=1.0, lower_inclusive=False
    ),
    "UM_vf_sync": GlyphFactorSpec(
        Glyph.UM, lower=0.0, upper=1.0, lower_inclusive=False
    ),
    "UM_dnfr_reduction": GlyphFactorSpec(
        Glyph.UM, lower=0.0, upper=1.0, lower_inclusive=False
    ),
    "RA_epi_diff": GlyphFactorSpec(
        Glyph.RA, lower=0.0, upper=1.0, lower_inclusive=False
    ),
    "RA_vf_amplification": GlyphFactorSpec(
        Glyph.RA, lower=0.0, lower_inclusive=False
    ),
    "RA_phase_coupling": GlyphFactorSpec(
        Glyph.RA, lower=0.0, upper=1.0, lower_inclusive=False
    ),
    "SHA_vf_factor": GlyphFactorSpec(
        Glyph.SHA, lower=0.0, upper=1.0, upper_inclusive=False
    ),
    "VAL_scale": GlyphFactorSpec(
        Glyph.VAL, lower=1.0, lower_inclusive=False
    ),
    "NUL_scale": GlyphFactorSpec(
        Glyph.NUL,
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=False,
    ),
    "NUL_densification_factor": GlyphFactorSpec(
        Glyph.NUL, lower=1.0, lower_inclusive=False
    ),
    "THOL_accel": GlyphFactorSpec(
        Glyph.THOL, lower=0.0, lower_inclusive=False
    ),
    "ZHIR_theta_shift_factor": GlyphFactorSpec(
        Glyph.ZHIR,
        lower=0.0,
        lower_inclusive=False,
        nonzero_modulus=8.0,
    ),
    # Backward-compatible explicit phase rotation.  It has no canonical
    # default and a full 2*pi rotation would violate theta -> theta'.
    "ZHIR_theta_shift": GlyphFactorSpec(
        Glyph.ZHIR,
        has_canonical_default=False,
        nonzero_modulus=2.0 * math.pi,
    ),
    # NAV jitter is explicitly optional. NAV_eta=0 can still be meaningful in
    # strict mode or with jitter; the state-changed postcondition must therefore
    # be checked on the proposal rather than inferred from either coefficient.
    "NAV_jitter": GlyphFactorSpec(Glyph.NAV, lower=0.0),
    "NAV_eta": GlyphFactorSpec(Glyph.NAV, lower=0.0, upper=1.0),
    "REMESH_alpha": GlyphFactorSpec(
        Glyph.REMESH, lower=0.0, upper=1.0, lower_inclusive=False
    ),
}

GLYPH_FACTOR_SPECS: Mapping[str, GlyphFactorSpec] = MappingProxyType(_SPECS)
"""Immutable mapping from known factor names to their hard domains."""

_BY_GLYPH: dict[Glyph, list[str]] = {glyph: [] for glyph in Glyph}
for _key, _spec in GLYPH_FACTOR_SPECS.items():
    _BY_GLYPH[_spec.glyph].append(_key)
GLYPH_FACTORS_BY_GLYPH: Mapping[Glyph, tuple[str, ...]] = MappingProxyType(
    {glyph: tuple(keys) for glyph, keys in _BY_GLYPH.items()}
)
"""Immutable factor-key index derived from :data:`GLYPH_FACTOR_SPECS`."""


def canonical_glyph_factor_defaults() -> dict[str, float]:
    """Return a detached copy of the canonical factor defaults.

    Values are read lazily from the single configuration source so the domain
    registry cannot drift by carrying a second set of numerical defaults.
    """
    from ..config.defaults_core import CORE_DEFAULTS

    raw = CORE_DEFAULTS.get("GLYPH_FACTORS")
    if not isinstance(raw, Mapping):
        raise RuntimeError("CORE_DEFAULTS['GLYPH_FACTORS'] must be a mapping")
    return dict(raw)


def _coerce_known_factor(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GlyphFactorValidationError(
            f"{key} must be a finite real scalar, got {value!r}"
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise GlyphFactorValidationError(
            f"{key} must be representable as a finite real scalar, got {value!r}"
        ) from exc
    if not math.isfinite(resolved):
        raise GlyphFactorValidationError(
            f"{key} must be finite, got {value!r}"
        )
    return resolved


def validate_glyph_factor(key: str, value: Any) -> float:
    """Validate and materialize one known glyph factor as binary64."""
    spec = GLYPH_FACTOR_SPECS.get(key)
    if spec is None:
        raise GlyphFactorValidationError(f"Unknown canonical glyph factor {key!r}")
    resolved = _coerce_known_factor(key, value)
    if not spec.contains(resolved):
        raise GlyphFactorValidationError(
            f"{key} must lie in {spec.interval_text()}, got {resolved!r}"
        )
    return resolved


def _normalize_glyph(glyph: Glyph | str) -> Glyph:
    if isinstance(glyph, Glyph):
        return glyph
    token = str(glyph).strip()
    try:
        return Glyph(token.upper())
    except ValueError:
        pass

    # Executable identifiers and title-case display names live in
    # operator_contracts. Import lazily so this foundational validator does not
    # create an operators/config import cycle.
    try:
        from .operator_contracts import contract_for

        return Glyph(contract_for(token).glyph)
    except (KeyError, ValueError) as exc:
        raise GlyphFactorValidationError(
            f"Unknown glyph context {glyph!r}"
        ) from exc


def _validate_nul_relation(
    factors: Mapping[str, Any], *, explicit_keys: frozenset[str]
) -> None:
    if "NUL_densification_factor" not in explicit_keys:
        return

    defaults = canonical_glyph_factor_defaults()
    scale_raw = factors.get("NUL_scale", defaults["NUL_scale"])
    scale = validate_glyph_factor("NUL_scale", scale_raw)
    densification = validate_glyph_factor(
        "NUL_densification_factor", factors["NUL_densification_factor"]
    )
    expected = 1.0 / scale
    if densification != expected:
        raise GlyphFactorValidationError(
            "NUL_densification_factor is derived, not independent: expected "
            f"1.0 / NUL_scale == {expected!r}, got {densification!r}"
        )


def validate_glyph_factors(
    factors: Mapping[str, Any] | None,
    *,
    glyph: Glyph | str | None = None,
    active_keys: Iterable[str] | None = None,
    preserve_unknown: bool = True,
) -> dict[str, Any]:
    """Validate known factors without consuming extension-owned keys.

    When ``glyph`` is supplied, only factors used by that operator are checked.
    This keeps execution context-specific: an unrelated extension or a pending
    override for another glyph cannot block the current operator.
    """
    if factors is None:
        return {}
    if not isinstance(factors, Mapping):
        raise GlyphFactorValidationError(
            "Glyph factors must be a mapping, "
            f"got {type(factors).__name__}"
        )

    result = dict(factors)
    context = _normalize_glyph(glyph) if glyph is not None else None
    available_keys = (
        GLYPH_FACTOR_SPECS.keys()
        if context is None
        else GLYPH_FACTORS_BY_GLYPH[context]
    )
    if active_keys is None:
        keys = tuple(available_keys)
    else:
        requested = tuple(active_keys)
        available = frozenset(available_keys)
        invalid = tuple(key for key in requested if key not in available)
        if invalid:
            label = "all glyphs" if context is None else context.value
            raise GlyphFactorValidationError(
                f"Active factor keys {invalid!r} do not belong to {label}"
            )
        keys = requested
    for key in keys:
        if key in result:
            result[key] = validate_glyph_factor(key, result[key])

    if (context is None or context is Glyph.NUL) and (
        active_keys is None
        or "NUL_scale" in keys
        or "NUL_densification_factor" in keys
    ):
        _validate_nul_relation(result, explicit_keys=frozenset(result))

    if not preserve_unknown:
        result = {key: value for key, value in result.items() if key in _SPECS}
    return result


def resolve_operator_factors(
    overrides: Mapping[str, Any] | None,
    glyph: Glyph | str,
    *,
    active_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Merge and validate factors for one operator execution.

    ``overrides`` must be the raw graph mapping rather than an already merged
    dictionary.  Raw key presence is needed to derive NUL densification from an
    overridden contraction scale.
    """
    context = _normalize_glyph(glyph)
    active_keys = None if active_keys is None else tuple(active_keys)
    validated = validate_glyph_factors(
        overrides, glyph=context, active_keys=active_keys
    )
    resolved: dict[str, Any] = canonical_glyph_factor_defaults()
    resolved.update(validated)

    active = None if active_keys is None else frozenset(active_keys)
    nul_relation_active = context is Glyph.NUL and (
        active is None
        or "NUL_scale" in active
        or "NUL_densification_factor" in active
    )
    if nul_relation_active:
        scale = validate_glyph_factor("NUL_scale", resolved["NUL_scale"])
        if overrides is None or "NUL_densification_factor" not in overrides:
            resolved["NUL_densification_factor"] = 1.0 / scale
        _validate_nul_relation(
            resolved,
            explicit_keys=frozenset(
                {"NUL_scale", "NUL_densification_factor"}
            ),
        )

    return validate_glyph_factors(
        resolved, glyph=context, active_keys=active_keys
    )


def runtime_active_glyph_factor_keys(
    glyph: Glyph | str,
    graph_data: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return the registered factors used by one concrete runtime branch.

    Global configuration validation still checks every explicit override. At
    execution time an explicitly disabled channel cannot block an unrelated
    active channel, so only the branch's consumed factors are materialized.
    """

    context = _normalize_glyph(glyph)
    active = list(GLYPH_FACTORS_BY_GLYPH[context])

    if context is Glyph.OZ and bool(graph_data.get("OZ_NOISE_MODE", False)):
        active.remove("OZ_dnfr_factor")
    elif context is Glyph.UM:
        if not bool(graph_data.get("UM_SYNC_VF", True)):
            active.remove("UM_vf_sync")
        if not bool(graph_data.get("UM_STABILIZE_DNFR", True)):
            active.remove("UM_dnfr_reduction")
    elif context is Glyph.ZHIR:
        fixed_shift = isinstance(overrides, Mapping) and "ZHIR_theta_shift" in overrides
        active.remove(
            "ZHIR_theta_shift_factor" if fixed_shift else "ZHIR_theta_shift"
        )
    elif context is Glyph.NAV and bool(graph_data.get("NAV_STRICT", False)):
        active.remove("NAV_eta")
    elif context is Glyph.REMESH:
        # The node-level glyph is an advisory; network REMESH validates alpha
        # at its own scale-aware entry point.
        active.clear()

    return tuple(active)


def resolve_runtime_operator_factors(
    overrides: Mapping[str, Any] | None,
    glyph: Glyph | str,
    graph_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve only factors consumed by the concrete runtime branch."""

    if overrides is not None and not isinstance(overrides, Mapping):
        raise GlyphFactorValidationError(
            "Glyph factors must be a mapping, "
            f"got {type(overrides).__name__}"
        )
    active = runtime_active_glyph_factor_keys(glyph, graph_data, overrides)
    return resolve_operator_factors(overrides, glyph, active_keys=active)
