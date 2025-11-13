"""Bifurcation dynamics and structural path selection for TNFR operators.

This module provides utilities for detecting bifurcation readiness and
determining viable structural reorganization paths after OZ-induced dissonance.

According to TNFR canonical theory (§2.3.3, R4), when ∂²EPI/∂t² > τ,
the system enters a bifurcation state enabling multiple reorganization
trajectories. This module implements path selection based on nodal state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..types import Glyph

__all__ = [
    "get_bifurcation_paths",
    "compute_bifurcation_score",
]


def get_bifurcation_paths(G: "TNFRGraph", node: "NodeId") -> list["Glyph"]:
    """Return viable structural paths after OZ-induced bifurcation.

    When OZ (Dissonance) creates bifurcation readiness (∂²EPI/∂t² > τ),
    this function determines which operators can resolve the dissonance
    based on current nodal state.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier

    Returns
    -------
    list[Glyph]
        List of viable operator glyphs for structural reorganization.
        Empty list if node is not in bifurcation state.

    Notes
    -----
    **Canonical bifurcation paths:**

    - **ZHIR (Mutation)**: Viable if νf > 0.8 (sufficient for controlled transformation)
    - **NUL (Contraction)**: Viable if EPI < 0.5 (safe collapse window)
    - **IL (Coherence)**: Always viable (universal resolution path)
    - **THOL (Self-organization)**: Viable if degree >= 2 (network support)

    The node must have `_bifurcation_ready = True` flag, typically set by
    OZ precondition validation when ∂²EPI/∂t² exceeds threshold τ.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Dissonance
    >>> from tnfr.dynamics.bifurcation import get_bifurcation_paths
    >>> G, node = create_nfr("test", epi=0.4, vf=1.0)
    >>> # Set up bifurcation conditions
    >>> G.nodes[node]["epi_history"] = [0.2, 0.35, 0.55]
    >>> Dissonance()(G, node, validate_preconditions=True)
    >>> paths = get_bifurcation_paths(G, node)
    >>> # Returns viable operators: [ZHIR, NUL, IL, THOL] or subset

    See Also
    --------
    tnfr.operators.preconditions.validate_dissonance : Sets bifurcation_ready flag
    tnfr.operators.definitions.SelfOrganization : Spawns sub-EPIs on bifurcation
    """
    # Check if bifurcation active
    if not G.nodes[node].get("_bifurcation_ready", False):
        return []  # No bifurcation active

    # Get node state for path evaluation
    abs(float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)))
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    degree = G.degree(node)

    paths = []

    # ZHIR (Mutation) viable if sufficient νf for controlled transformation
    zhir_threshold = float(G.graph.get("ZHIR_BIFURCATION_VF_THRESHOLD", 0.8))
    if vf > zhir_threshold:
        paths.append(Glyph.ZHIR)

    # NUL (Contraction) viable if EPI low enough for safe collapse
    nul_threshold = float(G.graph.get("NUL_BIFURCATION_EPI_THRESHOLD", 0.5))
    if epi < nul_threshold:
        paths.append(Glyph.NUL)

    # IL (Coherence) always viable as universal resolution path
    paths.append(Glyph.IL)

    # THOL (Self-organization) viable if network connectivity supports it
    thol_min_degree = int(G.graph.get("THOL_BIFURCATION_MIN_DEGREE", 2))
    if degree >= thol_min_degree:
        paths.append(Glyph.THOL)

    return paths


def compute_bifurcation_score(
    d2epi: float,
    dnfr: float,
    vf: float,
    epi: float,
    tau: float = 0.5,
) -> float:
    """Compute quantitative bifurcation potential [0,1].

    Integrates multiple structural indicators to assess bifurcation readiness.
    According to TNFR canonical theory (§2.3.3, R4), bifurcation occurs when
    ∂²EPI/∂t² > τ (acceleration exceeds threshold). This function extends that
    binary condition into a continuous score that accounts for multiple factors.

    Parameters
    ----------
    d2epi : float
        Structural acceleration (∂²EPI/∂t²). Primary indicator of bifurcation.
        When |d2epi| > τ, the system enters a bifurcation state enabling
        multiple reorganization trajectories.
    dnfr : float
        Internal reorganization operator (ΔNFR). Magnitude indicates instability
        level. Higher |ΔNFR| means stronger reorganization pressure.
    vf : float
        Structural frequency (νf) in Hz_str units. Determines capacity to respond
        to bifurcation. Higher νf enables faster reorganization along new paths.
    epi : float
        Primary Information Structure. Provides structural substrate for
        bifurcation. Higher EPI indicates more material to reorganize.
    tau : float, default 0.5
        Bifurcation acceleration threshold. When |d2epi| > tau, bifurcation
        becomes active. Default 0.5 is the canonical TNFR threshold.

    Returns
    -------
    float
        Bifurcation score in range [0.0, 1.0]:
        - 0.0 = no bifurcation potential (stable)
        - 0.5 = bifurcation threshold (critical)
        - 1.0 = maximal bifurcation readiness (multiple paths viable)

    Notes
    -----
    The bifurcation score is a weighted combination of four factors:

    1. **Acceleration factor** (40%): |∂²EPI/∂t²| / τ
       Primary indicator. Measures how close the system is to or beyond
       the bifurcation threshold.

    2. **Instability factor** (30%): |ΔNFR|
       Secondary indicator. Measures reorganization pressure that drives
       bifurcation exploration.

    3. **Capacity factor** (20%): νf / 2.0
       Measures structural reorganization capacity. Higher νf enables faster
       response to bifurcation opportunities.

    4. **Substrate factor** (10%): EPI / 0.8
       Measures available structural material. Higher EPI provides more
       degrees of freedom for bifurcation paths.

    Formula:
        score = 0.4 * accel + 0.3 * instability + 0.2 * capacity + 0.1 * substrate

    All factors are normalized to [0, 1] and clipped before combination.

    Examples
    --------
    >>> from tnfr.dynamics.bifurcation import compute_bifurcation_score
    >>>
    >>> # Low bifurcation potential (stable state)
    >>> score = compute_bifurcation_score(
    ...     d2epi=0.1,  # Low acceleration
    ...     dnfr=0.05,  # Low instability
    ...     vf=0.5,     # Moderate capacity
    ...     epi=0.3,    # Low substrate
    ... )
    >>> assert score < 0.3  # Low score
    >>>
    >>> # High bifurcation potential (critical state)
    >>> score = compute_bifurcation_score(
    ...     d2epi=0.7,  # High acceleration (> tau)
    ...     dnfr=0.6,   # High instability
    ...     vf=1.8,     # High capacity
    ...     epi=0.7,    # High substrate
    ... )
    >>> assert score > 0.7  # High score

    See Also
    --------
    get_bifurcation_paths : Determine viable operators after bifurcation
    tnfr.operators.metrics.dissonance_metrics : Uses score in OZ metrics
    """
    from ..utils import get_numpy

    np = get_numpy()

    # 1. Acceleration factor (primary indicator)
    # Normalized by tau threshold
    accel_factor = min(abs(d2epi) / tau, 1.0) if tau > 0 else 0.0

    # 2. Instability factor (secondary indicator)
    # Already dimensionless, clip to [0, 1]
    instability_factor = min(abs(dnfr), 1.0)

    # 3. Capacity factor (reorganization capability)
    # Normalize by 2.0 (typical high νf value)
    capacity_factor = min(vf / 2.0, 1.0) if vf >= 0 else 0.0

    # 4. Substrate factor (structural material available)
    # Normalize by 0.8 (typical high EPI value)
    substrate_factor = min(epi / 0.8, 1.0) if epi >= 0 else 0.0

    # Weighted combination (percentages sum to 100%)
    score = (
        0.4 * accel_factor  # 40% - primary
        + 0.3 * instability_factor  # 30% - secondary
        + 0.2 * capacity_factor  # 20% - capability
        + 0.1 * substrate_factor  # 10% - material
    )

    # Ensure result is in [0, 1] range
    return float(np.clip(score, 0.0, 1.0))
