"""TNFR Grammar: U6 Structural Potential Validation

U6: STRUCTURAL POTENTIAL CONFINEMENT - Validate Δ Φ_s < π/2 (drift threshold).

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from ..mathematics.unified_numerical import np
from .grammar_types import StructuralPotentialConfinementError

__all__ = [
    "StructuralPotentialConfinementObservation",
    "observe_structural_potential_confinement",
    "structural_potential_change_terms",
    "validate_structural_potential_confinement",
]


@dataclass(frozen=True)
class StructuralPotentialConfinementObservation:
    """Read-only U6 drift report with explicit field conventions.

    This records a two-snapshot, finite-observation comparison. It does not
    certify the interval between snapshots or an unobserved future tail.
    """

    confined: bool
    mean_absolute_drift: float
    threshold: float
    kernel: str
    aggregation: str
    reference: str
    time_coverage: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Return detached JSON-compatible U6 observation metadata."""
        return {
            "confined": self.confined,
            "mean_absolute_drift": self.mean_absolute_drift,
            "threshold": self.threshold,
            "kernel": self.kernel,
            "aggregation": self.aggregation,
            "reference": self.reference,
            "time_coverage": self.time_coverage,
            "message": self.message,
        }


def structural_potential_change_terms(
    kernel_before: Any,
    pressure_before: Any,
    kernel_after: Any,
    pressure_after: Any,
) -> tuple[Any, Any, Any]:
    r"""Decompose ``Phi_after - Phi_before`` under a topology change.

    For aligned nodes and declared kernels, the exact identity is
    ``B_after @ (p_after-p_before) + (B_after-B_before) @ p_before``.
    It distinguishes pressure reorganization from distance-kernel change;
    unmatched node sets must be handled by the caller before this read-out.
    """
    before_kernel = np.asarray(kernel_before, dtype=float)
    after_kernel = np.asarray(kernel_after, dtype=float)
    before_pressure = np.asarray(pressure_before, dtype=float)
    after_pressure = np.asarray(pressure_after, dtype=float)
    if (
        before_kernel.ndim != 2
        or before_kernel.shape != after_kernel.shape
        or before_kernel.shape[0] != before_kernel.shape[1]
        or before_pressure.shape != after_pressure.shape
        or before_pressure.shape != (before_kernel.shape[0],)
    ):
        raise ValueError("kernels and pressures must use one aligned node order")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            before_kernel,
            after_kernel,
            before_pressure,
            after_pressure,
        )
    ):
        raise ValueError("kernels and pressures must contain only finite values")
    try:
        with np.errstate(over="raise", invalid="raise"):
            pressure_term = after_kernel @ (after_pressure - before_pressure)
            topology_term = (after_kernel - before_kernel) @ before_pressure
            total = pressure_term + topology_term
    except FloatingPointError as exc:
        raise ValueError(
            "structural-potential change decomposition exceeds floating-point range"
        ) from exc
    return pressure_term, topology_term, total

# ============================================================================
# U6: Structural Potential Confinement (CANONICAL as of 2025-11-11)
# ============================================================================


def observe_structural_potential_confinement(
    G: Any,
    phi_s_reference: Mapping[Any, float],
    phi_s_observed: Mapping[Any, float],
    *,
    threshold: float = U6_STRUCTURAL_POTENTIAL_LIMIT,
    reference: str = "provided_snapshot",
    time_coverage: str = "two_snapshot_finite_observation",
) -> StructuralPotentialConfinementObservation:
    """Observe canonical U6 drift without changing validation behaviour.

    ``phi_s_reference`` and ``phi_s_observed`` must be generated from the
    canonical directed, weighted shortest-path field when this report is used
    as a canonical U6 observation. The report does not infer that provenance
    from arbitrary dictionaries; callers declare it explicitly at acquisition.
    """
    confined, drift, message = validate_structural_potential_confinement(
        G,
        phi_s_reference,
        phi_s_observed,
        threshold=threshold,
        strict=False,
    )
    return StructuralPotentialConfinementObservation(
        confined=confined,
        mean_absolute_drift=drift,
        threshold=float(threshold),
        kernel="canonical_directed_weighted_shortest_path_inverse_square",
        aggregation="mean_absolute_nodewise_drift",
        reference=reference,
        time_coverage=time_coverage,
        message=message,
    )


def validate_structural_potential_confinement(
    G: Any,
    phi_s_before: Mapping[Any, float],
    phi_s_after: Mapping[Any, float],
    threshold: float = U6_STRUCTURAL_POTENTIAL_LIMIT,  # selected ΔΦ_s < π/2 policy
    strict: bool = True,
) -> tuple[bool, float, str]:
    """Evaluate the U6 potential-drift policy on two supplied snapshots.

    The result is a read-only finite observation:
    ``mean_i |Phi_s_after(i) - Phi_s_before(i)| < pi/2``. It reports the
    configured policy condition for the supplied fields; it does not establish
    confinement between snapshots, a future tail bound, or fragmentation.

    Parameters
    ----------
    G : TNFRGraph
        Network graph (used for node iteration)
    phi_s_before : dict[NodeId, float]
        Structural potential before sequence application
    phi_s_after : dict[NodeId, float]
        Structural potential after sequence application
    threshold : float, default=pi/2
        Selected U6 structural-potential drift policy.
    strict : bool, default=True
        If True, raises StructuralPotentialConfinementError on violation.
        If False, returns (False, drift, message) without raising.

    Returns
    -------
    valid : bool
        True if Δ Φ_s < threshold (safe regime)
    drift : float
        Measured Δ Φ_s = mean(|Φ_s_after[i] - Φ_s_before[i]|)
    message : str
        Human-readable validation result

    Raises
    ------
    StructuralPotentialConfinementError
        If Δ Φ_s ≥ threshold and strict=True

    Notes
    -----
    U6 is a telemetry policy separate from grammar-word validation. The
    inverse-square potential is not bounded by phase wrapping alone: a proof
    of stronger confinement requires declared pressure and graph-geometry
    assumptions.

    Examples
    --------
    >>> from tnfr.physics.fields import compute_structural_potential
    >>> phi_before = compute_structural_potential(G)
    >>> apply_sequence(G, [Emission(), Coherence(), Silence()])
    >>> phi_after = compute_structural_potential(G)
    >>> from tnfr.config.defaults_core import STRUCTURAL_ESCAPE_THRESHOLD
    >>> valid, drift, msg = validate_structural_potential_confinement(
    ...     G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
    ... )
    >>> print(f"Valid: {valid}, Drift: {drift:.3f}")
    Valid: True, Drift: 0.583

    >>> # With strict=True (default), raises on violation
    >>> try:
    ...     validate_structural_potential_confinement(G, phi_before, phi_bad)
    ... except StructuralPotentialConfinementError as e:
    ...     print(f"U6 violation: {e}")

    References
    ----------
    - UNIFIED_GRAMMAR_RULES.md § U6: Complete physics derivation
    - docs/STRUCTURAL_FIELDS_TETRAD.md: Validation evidence
    - AGENTS.md § Structural Fields: Canonical status
    - src/tnfr/physics/fields.py: compute_structural_potential()

    """

    if isinstance(threshold, bool):
        raise TypeError("threshold must be a finite positive real number")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value <= 0.0:
        raise ValueError("threshold must be finite and strictly positive")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool")
    if not isinstance(phi_s_before, Mapping) or not isinstance(phi_s_after, Mapping):
        raise TypeError("structural-potential snapshots must be mappings")

    # Require exact graph-node alignment. Missing entries previously became
    # silent zeros, which could manufacture or hide drift.
    nodes = list(G.nodes())
    node_set = set(nodes)
    before_keys = set(phi_s_before)
    after_keys = set(phi_s_after)
    if before_keys != node_set or after_keys != node_set:
        raise ValueError(
            "structural-potential snapshots must contain exactly the graph nodes"
        )
    if not nodes:
        return True, 0.0, "U6: No nodes, trivially satisfied"

    drifts: list[float] = []
    for node in nodes:
        before_raw = phi_s_before[node]
        after_raw = phi_s_after[node]
        if isinstance(before_raw, bool) or isinstance(after_raw, bool):
            raise TypeError("structural-potential values must be finite real numbers")
        try:
            phi_before_i = float(before_raw)
            phi_after_i = float(after_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                "structural-potential values must be finite real numbers"
            ) from exc
        if not np.isfinite(phi_before_i) or not np.isfinite(phi_after_i):
            raise ValueError("structural-potential values must be finite")
        drifts.append(abs(phi_after_i - phi_before_i))

    delta_phi_s = float(np.mean(drifts))
    if not np.isfinite(delta_phi_s):
        raise ValueError("structural-potential drift exceeds floating-point range")

    # Validate against threshold
    valid = delta_phi_s < threshold_value

    if valid:
        msg = (
            f"U6: PASS - Δ Φ_s = {delta_phi_s:.3f} < {threshold_value:.3f} (confined). "
            f"System remains in safe regime."
        )
        return True, delta_phi_s, msg
    else:
        msg = (
            f"U6: ALERT - mean |Δ Φ_s| = {delta_phi_s:.3f} ≥ "
            f"{threshold_value:.3f}. The selected finite drift policy was exceeded; "
            f"this observation alone does not prove fragmentation."
        )
        if strict:
            raise StructuralPotentialConfinementError(
                delta_phi_s=delta_phi_s,
                threshold=threshold_value,
                sequence=None,  # Sequence not available in this context
            )
        return False, delta_phi_s, msg
