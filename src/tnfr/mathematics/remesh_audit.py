r"""Scoped REMESH contract audit for p-adic transport (R4b, N09).

The p-adic tower supplies exact projective transport. Its lift and scale maps
are instantaneous morphisms; canonical REMESH additionally requires a temporal
EPI echo. This module measures that distinction without treating scalar EPI
uniformity as evidence for grammar U5.

``field_uniformity_score`` reports the heuristic ``1 / (1 + std(EPI))``. It is
neither canonical structural coherence nor a parent/child U5 certificate. U5
remains unverified unless the caller declares a concrete post-update hierarchy
through ``RemeshU5Evidence``; the evidence is evaluated by the canonical
``assess_u5_parent_child_coherence`` function with explicit alpha and tolerance.
Consequently the default campaign distinguishes temporal memory, while neither
candidate realizes REMESH without hierarchy evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import NodeId
from .padic_tower import (
    RemeshContractAudit,
    padic_lift_map,
    projective_scale_map,
)

if TYPE_CHECKING:
    from ..physics.multiscale_coherence import U5CoherenceAssessment

__all__ = [
    "field_uniformity_score",
    "remesh_coefficients",
    "remesh_recurrence",
    "remesh_recurrence_update",
    "scale_projection_update",
    "temporal_echo_residual",
    "RemeshU5Evidence",
    "RemeshCandidateAudit",
    "audit_remesh_candidate",
    "RemeshCampaign",
    "remesh_campaign",
]


def _finite_scalar(value: Any, *, name: str) -> float:
    """Normalize a finite scalar while rejecting truth values and arrays."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real scalar, not bool")
    if isinstance(value, (str, bytes)) or not bool(np.isscalar(value)):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _network_size(value: Any) -> int:
    """Return a nontrivial integer network size."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("network_size must be an integer greater than one")
    normalized = int(value)
    if normalized <= 1:
        raise ValueError("network_size must be an integer greater than one")
    return normalized


def _frac(matrix: Any) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in matrix], dtype=float)


def field_uniformity_score(values: Any) -> float:
    r"""Return the scalar-field heuristic ``1 / (1 + std(EPI))``.

    This score describes only dispersion in a finite one-dimensional EPI field.
    It does not read ``DeltaNFR`` or ``dEPI``, does not declare a hierarchy, and
    cannot establish canonical structural ``C(t)`` or grammar U5.
    """
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("values must be a finite one-dimensional field") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("values must be a nonempty one-dimensional field")
    try:
        field = np.fromiter(
            (_finite_scalar(value, name="field value") for value in raw),
            dtype=float,
            count=raw.size,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("values must contain only finite real scalars") from exc

    scale = float(np.max(np.abs(field)))
    if scale == 0.0:
        return 1.0
    standard_deviation = scale * float(np.std(field / scale))
    if standard_deviation >= 1.0:
        inverse = 1.0 / standard_deviation
        return inverse / (1.0 + inverse)
    return 1.0 / (1.0 + standard_deviation)


def remesh_coefficients(alpha: float) -> tuple[float, float, float]:
    r"""Return ``((1-alpha)^2, alpha(1-alpha), alpha)``.

    The coefficients form a convex partition of unity for the required domain
    ``0 <= alpha <= 1``.
    """
    value = _finite_scalar(alpha, name="alpha")
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return ((1.0 - value) ** 2, value * (1.0 - value), value)


def remesh_recurrence(
    now: Any,
    past_local: Any,
    past_global: Any,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    r"""Apply the canonical REMESH temporal echo across two delay scales."""
    current_weight, local_weight, global_weight = remesh_coefficients(alpha)
    return (
        current_weight * np.asarray(now, dtype=float)
        + local_weight * np.asarray(past_local, dtype=float)
        + global_weight * np.asarray(past_global, dtype=float)
    )


def remesh_recurrence_update(*, alpha: float = 0.5):
    r"""Return a candidate update with explicit local and global memory."""
    # Validate at builder time so an invalid candidate fails before execution.
    remesh_coefficients(alpha)

    def update(now: Any, past_local: Any, past_global: Any) -> np.ndarray:
        return remesh_recurrence(
            now,
            past_local,
            past_global,
            alpha=alpha,
        )

    return update


def scale_projection_update(p: int, e: int):
    r"""Return the static p-adic projection ``Lift * R_e`` on the fine level.

    The projection ignores delayed inputs and therefore has no temporal echo.
    """
    projection = _frac(padic_lift_map(p, e)) @ _frac(
        projective_scale_map(p, e)
    )

    def update(now: Any, past_local: Any, past_global: Any) -> np.ndarray:
        del past_local, past_global
        return projection @ np.asarray(now, dtype=float)

    return update


def temporal_echo_residual(
    update: Any,
    now: Any,
    past_local: Any,
    past_global: Any,
    *,
    delta: float = 1e-2,
) -> float:
    r"""Measure update sensitivity to delayed inputs.

    Zero means that the candidate ignores history. A positive result establishes
    sensitivity for this probe; it is not by itself a complete REMESH contract.
    """
    delta_value = _finite_scalar(delta, name="delta")
    if delta_value <= 0.0:
        raise ValueError("delta must be positive")
    current = np.asarray(now, dtype=float)
    local = np.asarray(past_local, dtype=float)
    global_ = np.asarray(past_global, dtype=float)
    base = np.asarray(update(current, local, global_), dtype=float)
    step = delta_value * np.ones_like(local)
    local_change = np.linalg.norm(
        np.asarray(update(current, local + step, global_), dtype=float) - base
    )
    global_change = np.linalg.norm(
        np.asarray(update(current, local, global_ + step), dtype=float) - base
    )
    return float(max(local_change, global_change) / delta_value)


@dataclass(frozen=True, slots=True)
class RemeshU5Evidence:
    """Declared post-update hierarchy for a canonical U5 assessment.

    The caller is responsible for providing the graph materialized after the
    candidate update. The assessment records the concrete parent, children,
    alpha, tolerance, and canonical per-node coherence values; it is not a
    universal preservation theorem for future states.
    """

    post_update_graph: Any
    parent: NodeId
    alpha: float
    children: tuple[NodeId, ...] | None = None
    tolerance: float = 0.0

    def assess(self) -> U5CoherenceAssessment:
        """Evaluate this declaration through the canonical U5 implementation."""
        from ..physics.multiscale_coherence import (
            assess_u5_parent_child_coherence,
        )

        return assess_u5_parent_child_coherence(
            self.post_update_graph,
            self.parent,
            alpha=self.alpha,
            children=self.children,
            tolerance=self.tolerance,
        )


@dataclass(frozen=True)
class RemeshCandidateAudit(RemeshContractAudit):
    """REMESH contract flags plus a separately named field diagnostic."""

    field_uniformity_before: float = 0.0
    field_uniformity_after: float = 0.0
    field_uniformity_preserved: bool = False
    u5_assessment: U5CoherenceAssessment | None = None

    @property
    def u5_evidence_declared(self) -> bool:
        """Whether an explicit canonical hierarchy assessment was supplied."""
        return self.u5_assessment is not None

    def to_dict(self) -> dict[str, bool | float]:
        """Return contract flags and non-canonical uniformity diagnostics."""
        result: dict[str, bool | float] = super().to_dict()
        result.update(
            {
                "field_uniformity_before": self.field_uniformity_before,
                "field_uniformity_after": self.field_uniformity_after,
                "field_uniformity_preserved": self.field_uniformity_preserved,
                "u5_evidence_declared": self.u5_evidence_declared,
            }
        )
        return result


def audit_remesh_candidate(
    update: Any,
    *,
    network_size: int = 9,
    tol: float = 1e-9,
    u5_evidence: RemeshU5Evidence | None = None,
) -> RemeshCandidateAudit:
    r"""Audit a candidate using scoped probes and optional hierarchy evidence.

    The temporal, network-scale, and identity flags describe the fixed probes
    below. ``u5_multiscale_verified`` can become true only when ``u5_evidence``
    produces a satisfying canonical parent/child assessment. Field uniformity
    is always reported separately and never contributes to ``realizes_remesh``.
    """
    size = _network_size(network_size)
    tolerance = _finite_scalar(tol, name="tol")
    if tolerance < 0.0:
        raise ValueError("tol must be nonnegative")

    rng = np.random.default_rng(0)
    now = rng.standard_normal(size)
    past_local = rng.standard_normal(size)
    past_global = rng.standard_normal(size)
    identity_probe = np.resize(np.array([1.0, -1.0, 0.5]), size)

    echo = temporal_echo_residual(update, now, past_local, past_global)
    output = np.asarray(update(now, past_local, past_global), dtype=float)
    fixed = np.asarray(
        update(identity_probe, identity_probe, identity_probe),
        dtype=float,
    )

    epi_recursion = echo > 1e-6
    network_scale = output.shape == (size,) and size > 1
    identity_preserved = bool(
        fixed.shape == identity_probe.shape
        and np.linalg.norm(fixed - identity_probe) < 1e-6
    )
    uniformity_before = field_uniformity_score(identity_probe)
    uniformity_after = field_uniformity_score(fixed)
    uniformity_preserved = uniformity_after >= uniformity_before - tolerance

    assessment = None if u5_evidence is None else u5_evidence.assess()
    u5_verified = bool(
        assessment is not None and assessment.satisfies_target
    )

    return RemeshCandidateAudit(
        epi_recursion_verified=epi_recursion,
        network_scale_verified=network_scale,
        identity_preserved_verified=identity_preserved,
        u5_multiscale_verified=u5_verified,
        field_uniformity_before=uniformity_before,
        field_uniformity_after=uniformity_after,
        field_uniformity_preserved=uniformity_preserved,
        u5_assessment=assessment,
    )


@dataclass(frozen=True)
class RemeshCampaign:
    """Static p-adic transport and temporal recurrence audit results."""

    static_lift: RemeshCandidateAudit
    temporal_recurrence: RemeshCandidateAudit

    @property
    def tower_realizes_remesh(self) -> bool:
        """Whether the p-adic tower transport satisfies every contract field."""
        return self.static_lift.realizes_remesh

    @property
    def temporal_echo_discriminates(self) -> bool:
        """Whether the probe detects memory only in the temporal candidate."""
        return (
            not self.static_lift.epi_recursion_verified
            and self.temporal_recurrence.epi_recursion_verified
        )

    @property
    def audit_discriminates(self) -> bool:
        """Whether the full gate rejects the lift and accepts the recurrence."""
        return (
            not self.static_lift.realizes_remesh
            and self.temporal_recurrence.realizes_remesh
        )


def remesh_campaign(
    *,
    p: int = 3,
    e: int = 1,
    alpha: float = 0.5,
    static_u5_evidence: RemeshU5Evidence | None = None,
    temporal_u5_evidence: RemeshU5Evidence | None = None,
) -> RemeshCampaign:
    r"""Run the R4b campaign on the ``p**(e+1)``-node fine level.

    Candidate-specific post-update hierarchy evidence must be supplied
    separately. Without it, both U5 flags remain explicitly unverified.
    """
    size = p ** (e + 1)
    return RemeshCampaign(
        static_lift=audit_remesh_candidate(
            scale_projection_update(p, e),
            network_size=size,
            u5_evidence=static_u5_evidence,
        ),
        temporal_recurrence=audit_remesh_candidate(
            remesh_recurrence_update(alpha=alpha),
            network_size=size,
            u5_evidence=temporal_u5_evidence,
        ),
    )
