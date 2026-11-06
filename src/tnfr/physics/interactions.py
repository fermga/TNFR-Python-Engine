"""Legacy-named convenience words composed from public TNFR operators.

The helpers invoke caller-selected operator instances sequentially at each
node and collect before/after field telemetry. They do not install physical
interaction laws or derive electromagnetic, weak, strong or gravitational
behavior. The names remain public compatibility labels.

Each public operator call retains its live preconditions and incremental
grammar selection, including the configured fallback policy. These helpers
perform no separate whole-word validation or joint graph transaction;
prior successful calls remain committed if a later call fails. Passing an
individual call does not establish future boundedness, collective identity,
confinement or autonomous occurrence. Operator contracts and grammar owners
retain those distinct execution and admission responsibilities.

The requested words are Coupling/Resonance/Coherence, optional Coherence
followed by Dissonance/Mutation/Coherence, Coupling/Coherence/SelfOrganization,
and Coherence with optional Silence. Telemetry and warning thresholds are
observations and policies, not evolution laws or stability certificates.

InteractionResult.applied retains requested operator names after successful
calls. A grammar fallback can execute a different operator; the retained
operator history, not this legacy list, records the actual selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx  # noqa: F401 - retain the existing module attribute

from ..constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHYSICS_HOTSPOT_FRACTION_CANONICAL,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from ..mathematics.unified_numerical import np
from ..operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Mutation,
    Resonance,
    SelfOrganization,
    Silence,
)
from .fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)


@dataclass
class InteractionResult:
    applied: list[str]
    warnings: list[str]
    grad_before_mean: float | None = None
    grad_after_mean: float | None = None
    kphi_before_abs_mean: float | None = None
    kphi_after_abs_mean: float | None = None
    phi_s_drift_mean: float | None = None


def _mean(d: dict[Any, float]) -> float:
    return float(np.mean(list(d.values()))) if d else 0.0


def _apply_to_nodes(G: Any, nodes: Iterable[Any], ops: Iterable[Any]) -> list[str]:
    """Call each requested operator per node; retain requested names on success."""
    applied: list[str] = []
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


def _telemetry_after(G: Any, snap: dict, *, compute_phi_s: bool = False) -> dict:
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
        "phi_s_drift": drift,
    }


def em_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = False,
    grad_threshold: float = GRAD_PHI_CANONICAL_THRESHOLD,
) -> InteractionResult:
    """Request Coupling, Resonance and Coherence in sequence at each node.

    The legacy EM-like name identifies this convenience word, not an
    electromagnetic law. Each call uses its own live preconditions and
    incremental grammar policy. Collective identity and future stability
    are not certified by completing the word.

    Parameters
    ----------
    G : Any
        Graph with the primitive state and history required by each operator
        and the requested field observations.
    nodes : Iterable
        Nodes visited sequentially.
    compute_phi_s : bool, default False
        Collect structural potential before and after and report mean drift.
    grad_threshold : float, default GRAD_PHI_CANONICAL_THRESHOLD
        Selected warning policy for mean phase gradient (pi/16), not a
        derived instability threshold. The kinematic bound is pi.

    Returns
    -------
    InteractionResult
        Requested operator names after successful calls, warnings and
        telemetry means. The list does not authenticate fallback selections.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops = [Coupling(), Resonance(), Coherence()]
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    grad_mean_b = _mean(snap["grad_b"])  # type: ignore[index]
    grad_mean_a = _mean(aft["grad_a"])  # type: ignore[index]
    kphi_abs_b = _mean(
        {k: abs(v) for k, v in snap["kphi_b"].items()}
    )  # type: ignore[index]
    kphi_abs_a = _mean(
        {k: abs(v) for k, v in aft["kphi_a"].items()}
    )  # type: ignore[index]

    warnings: list[str] = []
    if grad_mean_a >= grad_threshold:
        warnings.append(
            (
                "phase gradient high after EM-like: "
                f"{grad_mean_a:.3f} ≥ {grad_threshold}"
            )
        )
    if (
        aft.get("phi_s_drift") is not None
        and float(aft["phi_s_drift"]) >= U6_STRUCTURAL_POTENTIAL_LIMIT
    ):
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_s_drift']):.3f} ≥ {U6_STRUCTURAL_POTENTIAL_LIMIT:.3f}"
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
            float(aft["phi_s_drift"]) if aft.get("phi_s_drift") is not None else None
        ),
    )


def weak_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = False,
    ensure_stable_base: bool = True,
    grad_threshold: float = GRAD_PHI_CANONICAL_THRESHOLD,
) -> InteractionResult:
    """Request optional Coherence, then Dissonance, Mutation and Coherence.

    Mutation still requires its actual source history and live preconditions.
    Prepending Coherence does not by itself prove those premises or establish
    a weak interaction, transformation outcome or future stability.

    Parameters
    ----------
    ensure_stable_base : bool, default True
        Prepend the Coherence request. Incremental admission remains with
        the public operators and configured grammar policy.
    grad_threshold : float, default GRAD_PHI_CANONICAL_THRESHOLD
        Selected mean phase-gradient warning policy (pi/16), not a derived
        instability threshold.

    Returns
    -------
    InteractionResult
        Requested operator names after successful calls, warnings and
        telemetry means; actual selections remain in operator history.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops: list[Any] = []
    if ensure_stable_base:
        ops.append(Coherence())
    ops.extend([Dissonance(), Mutation(), Coherence()])
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    grad_mean_b = _mean(snap["grad_b"])  # type: ignore[index]
    grad_mean_a = _mean(aft["grad_a"])  # type: ignore[index]
    kphi_abs_b = _mean(
        {k: abs(v) for k, v in snap["kphi_b"].items()}
    )  # type: ignore[index]
    kphi_abs_a = _mean(
        {k: abs(v) for k, v in aft["kphi_a"].items()}
    )  # type: ignore[index]

    warnings: list[str] = []
    if grad_mean_a >= grad_threshold:
        warnings.append(
            (
                "phase gradient high after Weak-like: "
                f"{grad_mean_a:.3f} ≥ {grad_threshold}"
            )
        )
    if (
        aft.get("phi_s_drift") is not None
        and float(aft["phi_s_drift"]) >= U6_STRUCTURAL_POTENTIAL_LIMIT
    ):
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_s_drift']):.3f} ≥ {U6_STRUCTURAL_POTENTIAL_LIMIT:.3f}"
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
            float(aft["phi_s_drift"]) if aft.get("phi_s_drift") is not None else None
        ),
    )


def strong_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = False,
    curvature_hotspot_threshold: float = K_PHI_CANONICAL_THRESHOLD,
) -> InteractionResult:
    """Request Coupling, Coherence and SelfOrganization at each node.

    The calls retain their actual admission and configured action. Curvature
    is observed afterward for warnings; it does not select this word or
    prove confinement, substructure formation or a strong interaction.

    Parameters
    ----------
    curvature_hotspot_threshold : float, default K_PHI_CANONICAL_THRESHOLD
        Selected absolute phase-curvature warning policy (0.9*pi).

    Returns
    -------
    InteractionResult
        Requested operator names after successful calls, warnings and
        telemetry means; actual selections remain in operator history.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops = [Coupling(), Coherence(), SelfOrganization()]
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    kphi_abs_a = {k: abs(v) for k, v in aft["kphi_a"].items()}  # type: ignore[index]
    hotspot_frac = 0.0
    if kphi_abs_a:
        vals = list(kphi_abs_a.values())
        hotspot_frac = float(
            sum(v >= curvature_hotspot_threshold for v in vals) / len(vals)
        )

    warnings: list[str] = []
    if hotspot_frac > PHYSICS_HOTSPOT_FRACTION_CANONICAL:  # heuristic
        warnings.append(
            (
                "curvature hotspots after Strong-like: "
                f"{hotspot_frac * 100:.1f}% ≥ {PHYSICS_HOTSPOT_FRACTION_CANONICAL * 100:.1f}%"
            )
        )
    if (
        aft.get("phi_s_drift") is not None
        and float(aft["phi_s_drift"]) >= U6_STRUCTURAL_POTENTIAL_LIMIT
    ):
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_s_drift']):.3f} ≥ {U6_STRUCTURAL_POTENTIAL_LIMIT:.3f}"
            )
        )

    return InteractionResult(
        applied=applied,
        warnings=warnings,
        grad_before_mean=_mean(snap["grad_b"]),  # type: ignore[index]
        grad_after_mean=_mean(aft["grad_a"]),  # type: ignore[index]
        kphi_before_abs_mean=_mean(
            {k: abs(v) for k, v in snap["kphi_b"].items()}
        ),  # type: ignore[index]
        kphi_after_abs_mean=_mean(kphi_abs_a),
        phi_s_drift_mean=(
            float(aft["phi_s_drift"]) if aft.get("phi_s_drift") is not None else None
        ),
    )


def gravity_like(
    G: Any,
    nodes: Iterable[Any],
    *,
    compute_phi_s: bool = True,
    quiet: bool = True,
) -> InteractionResult:
    """Request Coherence and optional Silence, with field telemetry.

    Structural-potential drift is a before/after observation. It is not an
    attraction law, a confinement proof or a gravitational field. Silence
    is an operator action, not an elapsed observation window by itself.

    Parameters
    ----------
    compute_phi_s : bool, default True
        Collect structural potential before and after and report mean drift.
    quiet : bool, default True
        Append the Silence request; its live operator contract still applies.

    Returns
    -------
    InteractionResult
        Requested operator names after successful calls, warnings and
        telemetry means; actual selections remain in operator history.
    """
    snap = _telemetry_before_after(G, compute_phi_s=compute_phi_s)

    ops: list[Any] = [Coherence()]
    if quiet:
        ops.append(Silence())
    applied = _apply_to_nodes(G, nodes, ops)

    aft = _telemetry_after(G, snap, compute_phi_s=compute_phi_s)
    warnings: list[str] = []
    if (
        aft.get("phi_s_drift") is not None
        and float(aft["phi_s_drift"]) >= U6_STRUCTURAL_POTENTIAL_LIMIT
    ):
        warnings.append(
            (
                "structural potential drift exceeded threshold: "
                f"{float(aft['phi_s_drift']):.3f} ≥ {U6_STRUCTURAL_POTENTIAL_LIMIT:.3f}"
            )
        )

    return InteractionResult(
        applied=applied,
        warnings=warnings,
        grad_before_mean=_mean(snap["grad_b"]),  # type: ignore[index]
        grad_after_mean=_mean(aft["grad_a"]),  # type: ignore[index]
        kphi_before_abs_mean=_mean(
            {k: abs(v) for k, v in snap["kphi_b"].items()}
        ),  # type: ignore[index]
        kphi_after_abs_mean=_mean(
            {k: abs(v) for k, v in aft["kphi_a"].items()}
        ),  # type: ignore[index]
        phi_s_drift_mean=(
            float(aft["phi_s_drift"]) if aft.get("phi_s_drift") is not None else None
        ),
    )


__all__ = [
    "InteractionResult",
    "em_like",
    "weak_like",
    "strong_like",
    "gravity_like",
]
