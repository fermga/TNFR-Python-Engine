"""Learning metrics for adaptive TNFR dynamics.

This module provides metrics specific to adaptive learning processes,
measuring plasticity, consolidation, and learning efficiency using
existing TNFR infrastructure (glyph history, EPI, νf, ΔNFR).

All functions reuse canonical utilities from glyph_history and alias modules.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Sequence

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..config.operator_names import (
    COHERENCE,
    DISSONANCE,
    MUTATION,
    RECURSIVITY,
    SELF_ORGANIZATION,
    SILENCE,
)
from ..constants.aliases import ALIAS_EPI
from ..types import Glyph, TNFRGraph, require_finite_real_scalar_epi

__all__ = [
    "compute_learning_plasticity",
    "compute_consolidation_index",
    "compute_learning_efficiency",
    "glyph_history_to_operator_names",
]


def glyph_history_to_operator_names(glyph_history: Sequence[str]) -> list[str]:
    """Convert glyph history to operator names for comparison.

    This is a lightweight helper that converts glyphs (e.g., 'AL', 'EN') to
    canonical operator names (e.g., 'emission', 'reception') using the
    existing GLYPH_TO_FUNCTION mapping.

    Parameters
    ----------
    glyph_history : Sequence[str]
        Sequence of glyph codes from node's glyph_history.

    Returns
    -------
    list[str]
        list of canonical operator names.

    Notes
    -----
    Reuses existing GLYPH_TO_FUNCTION mapping from grammar module.
    Computational cost is O(n) with n = len(glyph_history), just dict lookups.

    Examples
    --------
    >>> from tnfr.metrics.learning_metrics import glyph_history_to_operator_names
    >>> glyphs = ['AL', 'EN', 'IL']
    >>> names = glyph_history_to_operator_names(glyphs)
    >>> names
    ['emission', 'reception', 'coherence']
    """
    from ..operators.grammar import GLYPH_TO_FUNCTION

    result = []
    for glyph_str in glyph_history:
        # Convert string to Glyph enum if needed
        try:
            glyph = Glyph(glyph_str)
            operator_name = GLYPH_TO_FUNCTION.get(glyph, glyph_str.lower())
            result.append(operator_name)
        except (ValueError, KeyError):
            # If glyph is not recognized, keep as-is lowercased
            result.append(glyph_str.lower())

    return result


def compute_learning_plasticity(
    G: TNFRGraph,
    node: Any,
    window: int = 10,
) -> float:
    """Measure capacity for structural reorganization (learning plasticity).

    High plasticity indicates high capacity for learning and adaptation.
    Plasticity is measured by the frequency of reorganization operators
    (OZ, THOL, ZHIR) in the recent glyph history.

    Parameters
    ----------
    G : TNFRGraph
        Graph storing TNFR nodes and their structural operator history.
    node : Any
        Node identifier within G.
    window : int, default=10
        Number of recent glyphs to analyze.

    Returns
    -------
    float
        Plasticity index in range [0.0, 1.0] where higher values indicate
        greater reorganization capacity.

    Notes
    -----
    Reuses canonical glyph_history functions for tracking operator emissions.
    Plasticity operators: OZ (Dissonance), THOL (Self-organization),
    ZHIR (Mutation) - these represent structural flexibility.

    Examples
    --------
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Dissonance, SelfOrganization
    >>> from tnfr.metrics.learning_metrics import compute_learning_plasticity
    >>> G, node = create_nfr("learner", epi=0.3, vf=1.0)
    >>> # Apply reorganization operators
    >>> run_sequence(G, node, [Dissonance(), SelfOrganization()])
    >>> plasticity = compute_learning_plasticity(G, node, window=10)
    >>> plasticity > 0.0  # Should show some plasticity
    True
    """
    # Get glyph history directly from node
    history = G.nodes[node].get("glyph_history", [])
    if not history:
        return 0.0

    # Convert deque to list if needed
    if hasattr(history, "__iter__"):
        history = list(history)
    else:
        return 0.0

    # Limit to window size
    if window > 0 and len(history) > window:
        history = history[-window:]

    # Convert glyphs to operator names using canonical function
    operator_names = glyph_history_to_operator_names(history)

    # Count plastic operators: those that enable reorganization
    plastic_ops = {DISSONANCE, SELF_ORGANIZATION, MUTATION}
    plastic_count = sum(1 for op_name in operator_names if op_name in plastic_ops)

    # Normalize by history length
    return plastic_count / max(len(operator_names), 1)


def compute_consolidation_index(
    G: TNFRGraph,
    node: Any,
    window: int = 10,
) -> float:
    """Measure structural stabilization after learning (consolidation).

    High consolidation indicates that learning has been stabilized and
    integrated. Consolidation is measured by the frequency of stabilization
    operators (IL, SHA) in the recent glyph history.

    Parameters
    ----------
    G : TNFRGraph
        Graph storing TNFR nodes and their structural operator history.
    node : Any
        Node identifier within G.
    window : int, default=10
        Number of recent glyphs to analyze.

    Returns
    -------
    float
        Consolidation index in range [0.0, 1.0] where higher values indicate
        greater stabilization of learned patterns.

    Notes
    -----
    Reuses canonical glyph_history functions for tracking operator emissions.
    Consolidation operators: IL (Coherence), SHA (Silence) - these represent
    structural stabilization.

    Examples
    --------
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence, Silence
    >>> from tnfr.metrics.learning_metrics import compute_consolidation_index
    >>> G, node = create_nfr("learner", epi=0.5, vf=1.0)
    >>> # Apply consolidation operators
    >>> run_sequence(G, node, [Coherence(), Silence()])
    >>> consolidation = compute_consolidation_index(G, node, window=10)
    >>> consolidation > 0.0  # Should show some consolidation
    True
    """
    # Get glyph history directly from node
    history = G.nodes[node].get("glyph_history", [])
    if not history:
        return 0.0

    # Convert deque to list if needed
    if hasattr(history, "__iter__"):
        history = list(history)
    else:
        return 0.0

    # Limit to window size
    if window > 0 and len(history) > window:
        history = history[-window:]

    # Convert glyphs to operator names using canonical function
    operator_names = glyph_history_to_operator_names(history)

    # Count stable operators: those that consolidate structure
    stable_ops = {COHERENCE, SILENCE, RECURSIVITY}
    stable_count = sum(1 for op_name in operator_names if op_name in stable_ops)

    # Normalize by history length
    return stable_count / max(len(operator_names), 1)


def compute_learning_efficiency(
    G: TNFRGraph,
    node: Any,
) -> float:
    """Read absolute scalar EPI change per recorded operator application.

    This retrospective ratio measures change under the supplied observation
    window. It does not attribute that change to learning or establish quality.

    Parameters
    ----------
    G : TNFRGraph
        Graph storing TNFR nodes and their structural state.
    node : Any
        Node identifier within G.

    Returns
    -------
    float
        Learning efficiency as ΔEPI / number_of_operators. Higher values
        indicate more efficient learning (more change with fewer operations).

    Notes
    -----
    The initial EPI must correspond to the retained history window; the legacy
    missing-initial convention is zero. History eviction does not retain a
    lifetime denominator. A no-history record returns zero without claiming
    that a rate was measured. Scalar admission preserves uniform BEPI sign and
    rejects richer states. Nonrepresentable results raise instead of becoming
    zero or infinity.

    Examples
    --------
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Emission, Reception
    >>> from tnfr.metrics.learning_metrics import compute_learning_efficiency
    >>> G, node = create_nfr("learner", epi=0.2, vf=1.0)
    >>> G.nodes[node]["epi_initial"] = 0.2  # Track initial state
    >>> # Apply learning operators
    >>> run_sequence(G, node, [Emission(), Reception()])
    >>> efficiency = compute_learning_efficiency(G, node)
    >>> efficiency >= 0.0  # Should be non-negative
    True
    """
    # Get number of operators applied from glyph history
    history = G.nodes[node].get("glyph_history", [])
    num_ops = len(history) if history else 0

    if num_ops == 0:
        return 0.0

    epi_current = get_attr(
        G.nodes[node], ALIAS_EPI, 0.0, strict=True, conv=require_finite_real_scalar_epi
    )
    epi_initial = require_finite_real_scalar_epi(
        G.nodes[node].get("epi_initial", 0.0), "initial EPI"
    )
    # Divide the exact represented difference before materializing the result:
    # large endpoint differences can overflow even when the quotient is finite.
    ratio = abs(Fraction(epi_current) - Fraction(epi_initial)) / num_ops
    return finite_represented_real(ratio, "learning efficiency")[0]
