"""Strict precondition validation for EN (Reception) operator.

This module implements canonical precondition validation for the Reception (EN)
structural operator according to TNFR.pdf §2.2.1. EN requires specific structural
conditions selected by the configurable Reception admission policy:

1. **EPI bound**: stored EPI must be below the configured upper threshold
2. **Pressure bound**: stored signed DNFR must be below its configured upper bound

These checks do not claim that EN writes DNFR or measures structural coherence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...errors import TNFRValueError

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_reception_strict"]


def validate_reception_strict(G: TNFRGraph, node: Any) -> None:
    """Validate strict canonical preconditions for EN (Reception) operator.

    According to TNFR.pdf §2.2.1, Reception (EN - Recepción estructural) requires:

    1. Stored EPI is below its configured Reception upper bound.
    2. Stored signed DNFR is below its configured Reception upper bound.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation

    Raises
    ------
    ValueError
        If EPI or signed DNFR reaches its configured Reception upper bound.

    Notes
    -----
    Thresholds are configurable via:
    - Graph metadata: ``G.graph["EPI_SATURATION_MAX"]`` and
      ``G.graph["DNFR_RECEPTION_MAX"]``
    - Module defaults: :data:`tnfr.config.thresholds.EPI_SATURATION_MAX`, etc.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.reception import validate_reception_strict
    >>> G, node = create_nfr("test", epi=0.5, vf=0.9)
    >>> G.nodes[node]["dnfr"] = 0.08
    >>> validate_reception_strict(G, node)  # OK - receptive capacity available

    >>> G2, node2 = create_nfr("saturated", epi=0.95, vf=1.0)
    >>> validate_reception_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: EN precondition failed: EPI=0.950 >= 0.9.

    See Also
    --------
    tnfr.config.thresholds : Configurable threshold constants
    tnfr.operators.preconditions : Base precondition validators
    tnfr.operators.definitions.Reception : Reception operator implementation
    """
    from ...alias import get_attr
    from ...config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

    # Get configurable thresholds (allow override via graph metadata)
    epi_threshold = float(G.graph.get("EPI_SATURATION_MAX", EPI_SATURATION_MAX))
    dnfr_threshold = float(G.graph.get("DNFR_RECEPTION_MAX", DNFR_RECEPTION_MAX))

    # Precondition 1: enforce the selected stored-EPI upper admission bound.
    if epi >= epi_threshold:
        raise TNFRValueError(
            f"EN precondition failed: EPI={epi:.3f} >= {epi_threshold:.3f}. "
            f"Stored EPI is outside the configured Reception range. "
            f"Apply Reception only below the configured EPI upper bound.",
            context={"epi": epi, "epi_threshold": epi_threshold},
            suggestion=(
                "Apply Reception only below the configured EPI upper bound."
            ),
        )

    # Precondition 2: enforce the selected signed-pressure upper bound.
    if dnfr >= dnfr_threshold:
        raise TNFRValueError(
            f"EN precondition failed: DNFR={dnfr:.3f} >= {dnfr_threshold:.3f}. "
            f"Stored DNFR exceeds the configured Reception upper bound. "
            f"Consider IL (Coherence) first to reduce reorganization pressure.",
            context={"dnfr": dnfr, "dnfr_threshold": dnfr_threshold},
            suggestion=(
                "Consider IL (Coherence) first to reduce reorganization pressure."
            ),
        )
