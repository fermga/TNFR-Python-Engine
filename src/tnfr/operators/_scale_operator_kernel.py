"""Pure proposal kernel for the VAL and NUL capacity operators.

Expansion and Contraction share one target-local numerical structure.  This
module materializes that structure without mutating a node, graph, cache, log,
or history.  The ordinary single-node glyph dispatcher and future all-target
stages can therefore consume the same binary64 proposal.

For ``NUL_scale = f`` in ``(0, 1)``, the kernel applies
``nu_f' = f * nu_f`` and ``DeltaNFR' = DeltaNFR / f``.  Their ideal-real
product is unchanged, so Contraction redistributes the two nodal-equation
levers while preserving the instantaneous drive.  The reported residual
records the corresponding binary64 coefficient identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from ..dynamics.structural_clip import structural_clip
from ..errors import TNFRValueError
from ..types import Glyph
from ._epi_domain import require_real_scalar_epi
from .factor_contracts import GlyphFactorValidationError, validate_glyph_factor

__all__ = [
    "ScaleOperatorProposal",
    "compute_nul_edge_aware_scale",
    "compute_val_edge_aware_scale",
    "edge_aware_intervention_event",
    "nul_densification_event",
    "propose_scale_operator",
]


@dataclass(frozen=True, slots=True)
class ScaleOperatorProposal:
    """Immutable binary64 transition proposed by VAL or NUL."""

    glyph: Glyph
    requested_scale: float
    vf_before: float
    vf_after: float
    epi_before: float | None
    raw_epi_after: float | None
    epi_after: float | None
    write_epi: bool
    effective_epi_scale: float | None
    edge_aware_adapted: bool
    clip_delta: float | None
    dnfr_before: float | None
    dnfr_after: float | None
    densification_factor: float | None
    binary64_inverse_product_residual: float | None


def nul_densification_event(
    proposal: ScaleOperatorProposal, node: Any
) -> dict[str, Any]:
    """Build one detached NUL telemetry record from its bound proposal."""

    if proposal.glyph is not Glyph.NUL:
        raise ValueError("densification telemetry requires a NUL proposal")
    assert proposal.dnfr_before is not None
    assert proposal.dnfr_after is not None
    assert proposal.densification_factor is not None
    return {
        "node": node,
        "dnfr_before": proposal.dnfr_before,
        "dnfr_after": proposal.dnfr_after,
        "densification_factor": proposal.densification_factor,
        "contraction_scale": proposal.requested_scale,
        "derived_inverse_coefficient": True,
        "binary64_inverse_product_residual": (
            proposal.binary64_inverse_product_residual
        ),
    }


def edge_aware_intervention_event(
    proposal: ScaleOperatorProposal, node: Any
) -> dict[str, Any]:
    """Build one detached event for an adapted VAL boundary scale."""

    if not proposal.edge_aware_adapted:
        raise ValueError("edge intervention telemetry requires an adapted proposal")
    assert proposal.epi_before is not None
    assert proposal.epi_after is not None
    assert proposal.effective_epi_scale is not None
    return {
        "node": node,
        "glyph": proposal.glyph.name,
        "epi_before": proposal.epi_before,
        "epi_after": proposal.epi_after,
        "scale_requested": proposal.requested_scale,
        "scale_effective": proposal.effective_epi_scale,
        "adapted": True,
    }


def _finite_scalar(value: Any, label: str) -> float:
    """Materialize one finite runtime scalar with operator diagnostics."""

    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} must be representable as a finite scalar",
            context={"field": label, "value": repr(value)},
        ) from exc
    if not math.isfinite(resolved):
        raise TNFRValueError(
            f"{label} must remain finite",
            context={"field": label, "value": repr(value)},
        )
    return resolved


def compute_val_edge_aware_scale(
    epi_current: float,
    scale: float,
    magnitude_bound: float,
    epsilon: float,
) -> float:
    """Return the VAL scale limited by the sign-specific EPI boundary."""

    abs_epi = abs(epi_current)
    if abs_epi < epsilon:
        return scale
    return min(scale, magnitude_bound / abs_epi)


def compute_nul_edge_aware_scale(
    epi_current: float,
    scale: float,
    epi_min: float,
    epsilon: float,
) -> float:
    """Return NUL's current compatibility scale.

    A factor in ``(0, 1)`` moves a signed scalar EPI toward zero, so the
    historical edge-aware branch does not need an additional adaptation.
    The unused arguments remain explicit because the function describes the
    same configuration boundary as the VAL helper.
    """

    del epi_current, epi_min, epsilon
    return scale


def propose_scale_operator(
    *,
    glyph: Glyph,
    factor: Any,
    vf_before: Any,
    dnfr_before: Any = None,
    configured_densification_factor: Any = None,
    edge_aware_enabled: bool,
    epi_before: Any = None,
    epi_min: Any = -1.0,
    epi_max: Any = 1.0,
    epsilon: Any = 1e-12,
    clip_mode: Any = "hard",
) -> ScaleOperatorProposal:
    """Build the complete target-local VAL or NUL proposal without writes.

    ``epi_before`` and boundary configuration are intentionally ignored when
    edge awareness is disabled.  This preserves the established runtime branch
    in which scale operators alter only capacity (and NUL pressure).
    """

    if glyph not in (Glyph.VAL, Glyph.NUL):
        raise ValueError("scale proposals support VAL and NUL only")

    factor_key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
    resolved_factor = validate_glyph_factor(factor_key, factor)
    resolved_vf_before = _finite_scalar(
        vf_before, "nu_f before scale operator"
    )
    vf_after = _finite_scalar(
        resolved_vf_before * resolved_factor,
        f"{glyph.value} nu_f proposal",
    )
    if vf_after < 0.0:
        raise TNFRValueError(
            f"{glyph.value} must preserve nonnegative structural frequency"
        )

    resolved_dnfr_before: float | None = None
    dnfr_after: float | None = None
    densification_factor: float | None = None
    inverse_residual: float | None = None
    if glyph is Glyph.NUL:
        densification_factor = _finite_scalar(
            1.0 / resolved_factor,
            "NUL inverse densification coefficient",
        )
        if configured_densification_factor is not None:
            configured = validate_glyph_factor(
                "NUL_densification_factor",
                configured_densification_factor,
            )
            if configured != densification_factor:
                raise GlyphFactorValidationError(
                    "NUL_densification_factor is derived, not independent: "
                    f"expected {densification_factor!r}, got {configured!r}"
                )
        resolved_dnfr_before = _finite_scalar(
            dnfr_before, "DeltaNFR before Contraction"
        )
        dnfr_after = _finite_scalar(
            resolved_dnfr_before * densification_factor,
            "NUL DeltaNFR proposal",
        )
        inverse_residual = resolved_factor * densification_factor - 1.0

    resolved_epi_before: float | None = None
    raw_epi_after: float | None = None
    epi_after: float | None = None
    effective_epi_scale: float | None = None
    edge_aware_adapted = False
    clip_delta: float | None = None
    if edge_aware_enabled:
        resolved_epsilon = _finite_scalar(epsilon, "EDGE_AWARE_EPSILON")
        if resolved_epsilon <= 0.0:
            raise TNFRValueError("EDGE_AWARE_EPSILON must be positive")
        resolved_epi_min = _finite_scalar(epi_min, "EPI_MIN")
        resolved_epi_max = _finite_scalar(epi_max, "EPI_MAX")
        if resolved_epi_min > resolved_epi_max:
            raise TNFRValueError("EPI_MIN must not exceed EPI_MAX")
        resolved_epi_before = require_real_scalar_epi(
            epi_before,
            operator="Expansion" if glyph is Glyph.VAL else "Contraction",
            label="target EPI state",
        )

        if glyph is Glyph.VAL:
            magnitude_bound = (
                resolved_epi_max
                if resolved_epi_before >= 0.0
                else abs(resolved_epi_min)
            )
            effective_epi_scale = compute_val_edge_aware_scale(
                resolved_epi_before,
                resolved_factor,
                magnitude_bound,
                resolved_epsilon,
            )
        else:
            effective_epi_scale = compute_nul_edge_aware_scale(
                resolved_epi_before,
                resolved_factor,
                resolved_epi_min,
                resolved_epsilon,
            )
        effective_epi_scale = _finite_scalar(
            effective_epi_scale, f"{glyph.value} effective EPI scale"
        )
        raw_epi_after = _finite_scalar(
            resolved_epi_before * effective_epi_scale,
            f"{glyph.value} EPI proposal",
        )
        resolved_clip_mode: Literal["hard", "soft"] = (
            clip_mode if clip_mode in ("hard", "soft") else "hard"
        )
        epi_after = _finite_scalar(
            structural_clip(
                raw_epi_after,
                lo=resolved_epi_min,
                hi=resolved_epi_max,
                mode=resolved_clip_mode,
                record_stats=False,
            ),
            f"{glyph.value} bounded EPI proposal",
        )
        edge_aware_adapted = (
            abs(effective_epi_scale - resolved_factor) > resolved_epsilon
        )
        clip_delta = epi_after - raw_epi_after

    return ScaleOperatorProposal(
        glyph=glyph,
        requested_scale=resolved_factor,
        vf_before=resolved_vf_before,
        vf_after=vf_after,
        epi_before=resolved_epi_before,
        raw_epi_after=raw_epi_after,
        epi_after=epi_after,
        write_epi=bool(edge_aware_enabled),
        effective_epi_scale=effective_epi_scale,
        edge_aware_adapted=edge_aware_adapted,
        clip_delta=clip_delta,
        dnfr_before=resolved_dnfr_before,
        dnfr_after=dnfr_after,
        densification_factor=densification_factor,
        binary64_inverse_product_residual=inverse_residual,
    )
