"""Canonical interaction sequences composed from TNFR operators (centralized).

Overview
--------
This module provides physics-first helpers that compose structural operators
into interaction-like sequences while enforcing unified grammar awareness
(U1–U6) and instrumenting read-only telemetry via the Structural Field Tetrad
(Φ_s, |∇φ|, K_φ, ξ_C). These helpers are thin orchestrators and never mutate
EPI directly; they invoke canonical operators from
``tnfr.operators.definitions``.

Operator–Grammar Mapping (informative):
- Coupling (UM): requires phase verification (U3) inside operator
- Resonance (RA): propagates EPI coherently; phase-aware (U3)
- Coherence (IL): stabilizer enforcing boundedness (U2)
- Dissonance (OZ): controlled destabilizer; must be followed by handlers (U4a)
- Mutation (ZHIR): transformer at threshold; requires recent OZ and
    prior IL (U4b)
- SelfOrganization (THOL): creates sub-EPIs; stabilizer and
    transformer (U2, U4)
- Silence (SHA): closure/observation window (U1b)

Telemetry (read-only):
- |∇φ|: phase gradient; early stress indicator (threshold ≈ 0.38)
- K_φ: phase curvature; confinement hotspots (|K_φ| ≥ 3.0)
- Φ_s: structural potential; mean absolute drift as passive safety (ΔΦ_s < 2.0)

Contracts (summary):
- EM-like: [UM → RA → IL]. Preserve identity; phase-verified coupling (U3).
- Weak-like: [IL? → OZ → ZHIR → IL]. U4b-compliant (prior IL + recent OZ).
- Strong-like: [UM → IL → THOL]. Curvature-driven confinement;
    post-coupling IL.
- Gravity-like: [IL → SHA?]. Telemetry-oriented stabilization (Φ_s drift only).

Usage pattern
-------------
Given a graph G with phase (``theta``/``phase``) and ΔNFR
(``delta_nfr``/``dnfr``) attributes set (e.g., via
``tnfr.physics.patterns``), call one of the helpers with an iterable of
nodes. Results contain operator names applied and key telemetry means
before/after, plus optional warnings when thresholds are exceeded.

Notes
-----
- English-only documentation for canonicity and single
    source-of-truth alignment.
- Extend carefully with physics-first justification and tests. Keep parameters
    minimal and respect U-rules and structural invariants at all times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Dict

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from ..operators.definitions import (
    Coupling,
    Resonance,
    Coherence,
    Dissonance,
    Mutation,
    SelfOrganization,
    Silence,
)
from .fields import (
    compute_phase_gradient,
    compute_phase_curvature,
    compute_structural_potential,
)


@dataclass
class InteractionResult:
    applied: List[str]
    warnings: List[str]
    grad_before_mean: float | None = None
    grad_after_mean: float | None = None
    kphi_before_abs_mean: float | None = None
    kphi_after_abs_mean: float | None = None
    phi_s_drift_mean: float | None = None


def _mean(d: Dict[Any, float]) -> float:
    return float(np.mean(list(d.values()))) if d else 0.0


def _apply_to_nodes(
    G: Any, nodes: Iterable[Any], ops: Iterable[Any]
) -> List[str]:
    """Apply operator instances to each node in order; return names applied."""
    applied: List[str] = []
    for node in nodes:
        for op in ops:
            op(G, node)
            applied.append(op.name)
    return applied


def _telemetry_before_after(G: Any, *, compute_phi_s: bool = False) -> dict:
    grad_b = compute_phase_gradient(G)
    kphi_b = compute_phase_curvature(G)
    phi_b = compute_structural_potential(G) if compute_phi_s else None
    return {
        "grad_b": grad_b,
        "kphi_b": kphi_b,
        "phi_b": phi_b,
    }


def _telemetry_after(
    G: Any, snap: dict, *, compute_phi_s: bool = False
) -> dict:
    grad_a = compute_phase_gradient(G)
    kphi_a = compute_phase_curvature(G)
    phi_a = compute_structural_potential(G) if compute_phi_s else None
    drift = None
    if compute_phi_s and snap.get("phi_b") is not None and phi_a is not None:
        b = snap["phi_b"]
        # Mean absolute drift across nodes present in both
        keys = list({*b.keys(), *phi_a.keys()})
        if keys:
            diffs = [abs(phi_a.get(k, 0.0) - b.get(k, 0.0)) for k in keys]
            drift = float(np.mean(diffs))
    return {
        "grad_a": grad_a,
        "kphi_a": kphi_a,
        "phi_a": phi_a,
        "phi_drift": drift,
    }


def em_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = False,
    grad_threshold: float = 0.38,
) -> InteractionResult:
    """EM-like sequence: [Coupling → Resonance → Coherence].

    Purpose
    -------
    Reinforce/coherently propagate existing patterns under phase-compatible
    coupling, then stabilize. Identity-preserving and phase-aware.

    Contracts
    ---------
    - U3: Coupling/Resonance require phase compatibility (verified internally).
    - U2: Coherence stabilizes to maintain boundedness.
    - Read-only telemetry: |∇φ|, K_φ, optional Φ_s drift.

    Parameters
    ----------
    G : Any
        NetworkX-like graph with node attributes ``theta``/``phase`` and
        ``delta_nfr``/``dnfr``.
    nodes : Iterable
        Nodes to which the operator sequence will be applied.
    compute_phi_s : bool, default False
        If True, compute Φ_s before/after and report mean drift.
    grad_threshold : float, default 0.38
        Threshold for mean |∇φ| warning.

    Returns
    -------
    InteractionResult
        Names of operators applied, warnings, and telemetry means.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops = [Coupling(), Resonance(), Coherence()]
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    grad_mean_b = _mean(snap["grad_b"])  # type: ignore[index]
    grad_mean_a = _mean(aft["grad_a"])   # type: ignore[index]
    kphi_abs_b = _mean(
        {k: abs(v) for k, v in snap["kphi_b"].items()}
    )  # type: ignore[index]
    kphi_abs_a = _mean(
        {k: abs(v) for k, v in aft["kphi_a"].items()}
    )  # type: ignore[index]

    warnings: List[str] = []
    if grad_mean_a >= grad_threshold:
        warnings.append(
            (
                "phase gradient high after EM-like: "
                f"{grad_mean_a:.3f} ≥ {grad_threshold}"
            )
        )
    if aft.get("phi_drift") is not None and float(aft["phi_drift"]) >= 2.0:
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_drift']):.3f} ≥ 2.0"
            )
        )

    return InteractionResult(
        applied=applied,
        warnings=warnings,
        grad_before_mean=grad_mean_b,
        grad_after_mean=grad_mean_a,
        kphi_before_abs_mean=kphi_abs_b,
        kphi_after_abs_mean=kphi_abs_a,
        phi_s_drift_mean=(
            float(aft["phi_drift"]) if aft.get("phi_drift") is not None
            else None
        ),
    )


def weak_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = False,
    ensure_stable_base: bool = True,
    grad_threshold: float = 0.38,
) -> InteractionResult:
    """Weak-like sequence: [IL (optional) → Dissonance → Mutation → Coherence].

    Purpose
    -------
    Trigger controlled qualitative transformations (ZHIR) with stabilized
    pre/post conditions.

    Contracts
    ---------
    - U4b: Mutation (ZHIR) requires a stable base (prior IL) and recent
      destabilizer (OZ) within ~3 operations.
    - U2: Final Coherence to contain elevated ΔNFR.
    - Read-only telemetry: |∇φ|, K_φ, optional Φ_s drift.

    Parameters
    ----------
    ensure_stable_base : bool, default True
        Insert IL before OZ→ZHIR to satisfy U4b stable base requirement.
    grad_threshold : float, default 0.38
        Threshold for mean |∇φ| warning.

    Returns
    -------
    InteractionResult
        Names of operators applied, warnings, and telemetry means.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops: List[Any] = []
    if ensure_stable_base:
        ops.append(Coherence())
    ops.extend([Dissonance(), Mutation(), Coherence()])
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    grad_mean_b = _mean(snap["grad_b"])  # type: ignore[index]
    grad_mean_a = _mean(aft["grad_a"])   # type: ignore[index]
    kphi_abs_b = _mean(
        {k: abs(v) for k, v in snap["kphi_b"].items()}
    )  # type: ignore[index]
    kphi_abs_a = _mean(
        {k: abs(v) for k, v in aft["kphi_a"].items()}
    )  # type: ignore[index]

    warnings: List[str] = []
    if grad_mean_a >= grad_threshold:
        warnings.append(
            (
                "phase gradient high after Weak-like: "
                f"{grad_mean_a:.3f} ≥ {grad_threshold}"
            )
        )
    if aft.get("phi_drift") is not None and float(aft["phi_drift"]) >= 2.0:
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_drift']):.3f} ≥ 2.0"
            )
        )

    return InteractionResult(
        applied=applied,
        warnings=warnings,
        grad_before_mean=grad_mean_b,
        grad_after_mean=grad_mean_a,
        kphi_before_abs_mean=kphi_abs_b,
        kphi_after_abs_mean=kphi_abs_a,
        phi_s_drift_mean=(
            float(aft["phi_drift"]) if aft.get("phi_drift") is not None
            else None
        ),
    )


def strong_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = False,
    curvature_hotspot_threshold: float = 3.0,
) -> InteractionResult:
    """Strong-like sequence: [Coupling → Coherence → SelfOrganization].

    Purpose
    -------
    Promote local confinement and sub-EPI formation in regions of high
    curvature, with stabilization immediately after coupling.

    Contracts
    ---------
    - U3: Coupling remains phase-aware.
    - U2: Coherence right after coupling (boundedness).
    - U5: SelfOrganization creates sub-EPIs while preserving parent coherence.
    - Telemetry: flags |K_φ| hotspots; optional Φ_s drift.

    Parameters
    ----------
    curvature_hotspot_threshold : float, default 3.0
        Canonical |K_φ| threshold for hotspot flagging.

    Returns
    -------
    InteractionResult
        Names of operators applied, warnings, and telemetry means.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops = [Coupling(), Coherence(), SelfOrganization()]
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    kphi_abs_a = {
        k: abs(v) for k, v in aft["kphi_a"].items()
    }  # type: ignore[index]
    hotspot_frac = 0.0
    if kphi_abs_a:
        vals = list(kphi_abs_a.values())
        hotspot_frac = float(
            sum(v >= curvature_hotspot_threshold for v in vals) / len(vals)
        )

    warnings: List[str] = []
    if hotspot_frac > 0.1:  # heuristic
        warnings.append(
            (
                "curvature hotspots after Strong-like: "
                f"{hotspot_frac*100:.1f}% ≥ 10.0%"
            )
        )
    if aft.get("phi_drift") is not None and float(aft["phi_drift"]) >= 2.0:
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_drift']):.3f} ≥ 2.0"
            )
        )

    return InteractionResult(
        applied=applied,
        warnings=warnings,
        grad_before_mean=_mean(snap["grad_b"]),  # type: ignore[index]
        grad_after_mean=_mean(aft["grad_a"]),    # type: ignore[index]
        kphi_before_abs_mean=_mean(
            {k: abs(v) for k, v in snap["kphi_b"].items()}
        ),  # type: ignore[index]
        kphi_after_abs_mean=_mean(kphi_abs_a),
        phi_s_drift_mean=(
            float(aft["phi_drift"]) if aft.get("phi_drift") is not None
            else None
        ),
    )


def gravity_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = True,
    quiet: bool = True,
) -> InteractionResult:
    """Gravity-like: telemetry-oriented stabilization [Coherence → Silence].

    Purpose
    -------
    Stabilize and observe while interpreting Φ_s drift as passive confinement
    (no explicit attraction operator).

    Contracts
    ---------
    - U1b: Silence serves as closure/observation window when ``quiet=True``.
    - U2: Coherence preserves boundedness.
    - Telemetry: Φ_s drift is read-only and used as safety signal.

    Parameters
    ----------
    compute_phi_s : bool, default True
        Compute Φ_s before/after and report mean absolute drift.
    quiet : bool, default True
        Append Silence after Coherence for an observation window.

    Returns
    -------
    InteractionResult
        Names of operators applied, warnings, and telemetry means.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops: List[Any] = [Coherence()]
    if quiet:
        ops.append(Silence())
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    warnings: List[str] = []
    if aft.get("phi_drift") is not None and float(aft["phi_drift"]) >= 2.0:
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_drift']):.3f} ≥ 2.0"
            )
        )

    return InteractionResult(
        applied=applied,
        warnings=warnings,
        grad_before_mean=_mean(snap["grad_b"]),  # type: ignore[index]
        grad_after_mean=_mean(aft["grad_a"]),    # type: ignore[index]
        kphi_before_abs_mean=_mean(
            {k: abs(v) for k, v in snap["kphi_b"].items()}
        ),  # type: ignore[index]
        kphi_after_abs_mean=_mean(
            {k: abs(v) for k, v in aft["kphi_a"].items()}
        ),  # type: ignore[index]
        phi_s_drift_mean=(
            float(aft["phi_drift"]) if aft.get("phi_drift") is not None
            else None
        ),
    )


__all__ = [
    "InteractionResult",
    "em_like",
    "weak_like",
    "strong_like",
    "gravity_like",
]
