"""Advisory bifurcation routing and an operational diagnostic score.

Routing reads a stored readiness flag and configured scalar cuts; it does not
reconstruct current acceleration, validate grammar or execute an operator.
The weighted score is separate from the public THOL acceleration threshold
and is not an eligibility, stability or birth certificate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF
from ..constants.canonical import (
    NUL_EPI_THRESHOLD_CANONICAL,
    ZHIR_VF_THRESHOLD_CANONICAL,
)
from ..types import Glyph

__all__ = [
    "get_bifurcation_paths",
    "compute_bifurcation_score",
]


def get_bifurcation_paths(G: "TNFRGraph", node: "NodeId") -> list["Glyph"]:
    """Suggest glyphs using a stored readiness flag and configured cuts.

    A true ``_bifurcation_ready`` flag enables this advisory policy. The
    returned glyphs still require their own current public preconditions and
    grammar admission. This function neither refreshes the flag nor proves
    that any suggested glyph resolves the current pressure.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier

    Returns
    -------
    list[Glyph]
        Suggested operator glyphs, without an admission guarantee.
        Empty list when the stored readiness flag is false or absent.

    Notes
    -----
    **Configured branch suggestions:**

    - **ZHIR (Mutation)**: Proposed if νf exceeds the configured branch-selection cut
    - **NUL (Contraction)**: Proposed below its configured EPI cut (default ≈ 0.536)
    - **IL (Coherence)**: Always included in this suggestion list
    - **THOL (Self-organization)**: Proposed above its degree cut (default >= 2)

    The node must have `_bifurcation_ready = True` flag, typically set by
    OZ precondition validation. The flag can be stale relative to the current
    EPI history; no THOL pressure, history, depth or hierarchy gate runs here.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.bifurcation import get_bifurcation_paths
    >>> G, node = create_nfr("test", epi=0.4, vf=1.0)
    >>> # Missing readiness evidence produces no routing suggestions.
    >>> paths = get_bifurcation_paths(G, node)
    >>> paths
    []

    See Also
    --------
    tnfr.operators.preconditions.validate_dissonance : Sets bifurcation_ready flag
    tnfr.operators.definitions.SelfOrganization : Spawns sub-EPIs on bifurcation
    """
    # The stored flag is an advisory input, not rederived eligibility.
    if not G.nodes[node].get("_bifurcation_ready", False):
        return []  # No stored readiness declaration

    # Get node state for path evaluation
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    degree = G.degree(node)

    paths = []

    # This threshold ranks ZHIR as a bifurcation branch. It is not the direct
    # operator precondition, which uses the signed EPI-growth gate and active νf.
    zhir_threshold = float(
        G.graph.get("ZHIR_BIFURCATION_VF_THRESHOLD", ZHIR_VF_THRESHOLD_CANONICAL)
    )
    if vf > zhir_threshold:
        paths.append(Glyph.ZHIR)

    # NUL branch-selection policy; this is not its public admission check.
    nul_threshold = float(
        G.graph.get("NUL_BIFURCATION_EPI_THRESHOLD", NUL_EPI_THRESHOLD_CANONICAL)
    )  # ≈ 0.536 (operational)
    if epi < nul_threshold:
        paths.append(Glyph.NUL)

    # IL is always suggested; public execution still validates its own inputs.
    paths.append(Glyph.IL)

    # Degree alone suggests THOL; it does not establish public birth readiness.
    thol_min_degree = int(G.graph.get("THOL_BIFURCATION_MIN_DEGREE", 2))
    if degree >= thol_min_degree:
        paths.append(Glyph.THOL)

    return paths


def compute_bifurcation_score(
    d2epi: float,
    dnfr: float,
    vf: float,
    epi: float,
    tau: float = NUL_EPI_THRESHOLD_CANONICAL,  # ≈ 0.536 (operational)
) -> float:
    """Compute a configured weighted diagnostic in [0,1] for finite inputs.

    This combines four indicators using operational weights. It is not the
    public THOL gate, a fresh readiness assessment, or an operator-selection
    proof. In particular, 0.5 is not an acceleration-crossing threshold.

    Parameters
    ----------
    d2epi : float
        Structural acceleration (∂²EPI/∂t²). Its magnitude contributes one
        saturated score term, independently of the public THOL gate.
    dnfr : float
        Internal reorganization operator (ΔNFR). Magnitude indicates instability
        level. Higher |ΔNFR| means stronger reorganization pressure.
    vf : float
        Structural frequency (νf) in Hz_str units. Determines capacity to respond
        to bifurcation. Higher νf enables faster reorganization along new paths.
    epi : float
        Primary Information Structure. Provides structural substrate for
        bifurcation. Higher EPI indicates more material to reorganize.
    tau : float, default ≈ 0.536
        Acceleration normalization for this diagnostic. Default ≈ 0.536 is
        an operational scale, distinct from public THOL configuration.
        A nonpositive value suppresses the acceleration term.

    Returns
    -------
    float
        Weighted diagnostic in range [0.0, 1.0] for finite inputs. No value
        establishes stability, operator admission or a realized birth.

    Notes
    -----
    The bifurcation score is a weighted combination of four factors:

    1. **Acceleration factor** (46%): |∂²EPI/∂t²| / τ
       Primary indicator. Measures how close the system is to or beyond
       the bifurcation threshold.

    2. **Instability factor** (26%): |ΔNFR|
       Secondary indicator. Measures reorganization pressure that drives
       bifurcation exploration. Weight = 0.26 (operational).

    3. **Capacity factor** (14%): νf / 2.0
       Measures structural reorganization capacity. Higher νf enables faster
       response to bifurcation opportunities. Weight = 0.14 (operational).

    4. **Substrate factor** (14%): EPI / 0.9 (operational substrate scale)
       Measures available structural material. Higher EPI provides more
       degrees of freedom for bifurcation paths. Weight computed as remainder.

    Formula (operational weights):
        score = w_accel * accel + w_instab * instability + w_capac * capacity + w_substr * substrate
        where weights are operational values (audit 2026: not derived)

    All factors are normalized to [0, 1] and clipped before combination.
    With d2epi=0, dnfr=1, vf=2 and epi=0.9 the score is 0.54, while
    |d2epi|=tau>0 with the other channels zero gives 0.46. Thus neither side
    of 0.5 characterizes the acceleration gate.

    Examples
    --------
    >>> from tnfr.dynamics.bifurcation import compute_bifurcation_score
    >>>
    >>> # A low weighted diagnostic, without a stability conclusion
    >>> score = compute_bifurcation_score(
    ...     d2epi=0.1,  # Low acceleration
    ...     dnfr=0.05,  # Low instability
    ...     vf=0.5,     # Moderate capacity
    ...     epi=0.3,    # Low substrate
    ... )
    >>> assert score < 0.3  # Low score
    >>>
    >>> # A high weighted diagnostic, without an admission conclusion
    >>> score = compute_bifurcation_score(
    ...     d2epi=0.7,  # High acceleration (> tau)
    ...     dnfr=0.6,   # High instability
    ...     vf=1.8,     # High capacity
    ...     epi=0.7,    # High substrate
    ... )
    >>> assert score > 0.7  # High score

    See Also
    --------
    get_bifurcation_paths : Suggest glyphs from stored readiness metadata
    tnfr.operators.metrics.dissonance_metrics : Uses score in OZ metrics
    """
    from ..mathematics.unified_numerical import np

    # 1. Acceleration factor (primary indicator)
    # Normalized by tau threshold
    accel_factor = min(abs(d2epi) / tau, 1.0) if tau > 0 else 0.0

    # 2. Instability factor (secondary indicator)
    # Already dimensionless, clip to [0, 1]
    instability_factor = min(abs(dnfr), 1.0)

    # 3. Capacity factor (reorganization capability)
    # Normalize νf by a plain capacity scale (operational tuning, not TNFR physics)
    capacity_factor = min(vf / 2.0, 1.0) if vf >= 0 else 0.0

    # 4. Substrate factor (structural material available)
    # Normalize EPI by a plain substrate scale (operational tuning)
    substrate_factor = min(epi / 0.9, 1.0) if epi >= 0 else 0.0

    # Weighted combination (operational weights; sum to 1.0)
    w_accel = 0.46  # acceleration weight (operational)
    w_instab = 0.26  # instability weight (operational)
    w_capac = 0.14  # capacity weight (operational)
    w_substr = 1.0 - (w_accel + w_instab + w_capac)  # 0.14 remainder for substrate

    score = (
        w_accel * accel_factor  # operational primary
        + w_instab * instability_factor  # operational secondary
        + w_capac * capacity_factor  # operational capacity
        + w_substr * substrate_factor  # remainder material
    )

    # Ensure result is in [0, 1] range
    return float(np.clip(score, 0.0, 1.0))
