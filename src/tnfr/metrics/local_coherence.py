"""Local coherence helper used by operator metrics.

This provides a lightweight approximation of local coherence C_local
based on neighbor |ΔNFR| and |dEPI| means:

    C_local ≈ 1 / (1 + mean(|ΔNFR|) + mean(|dEPI/dt|))

It intentionally avoids importing heavier coherence modules to keep
operator metrics lean and prevent circular dependencies.
"""

from __future__ import annotations

from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_DEPI


def compute_local_coherence_fallback(G: Any, node: Any) -> float:
    """Compute a local coherence proxy from the node's neighborhood.

    Parameters
    ----------
    G : Graph-like
        Graph containing the node with neighbors()
    node : Hashable
        Node identifier

    Returns
    -------
    float
        Local coherence proxy in [0, 1]. Returns 0.0 if no neighbors.
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return 0.0

    def _as_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    dnfr_vals = [abs(_as_float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0))) for n in neighbors]
    depi_vals = [abs(_as_float(get_attr(G.nodes[n], ALIAS_DEPI, 0.0))) for n in neighbors]

    dnfr_mean = sum(dnfr_vals) / len(dnfr_vals) if dnfr_vals else 0.0
    depi_mean = sum(depi_vals) / len(depi_vals) if depi_vals else 0.0
    return 1.0 / (1.0 + dnfr_mean + depi_mean)
