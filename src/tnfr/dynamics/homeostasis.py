"""Configured range controller for a caller-selected TNFR node.

The controller requests public operators when stored parameters cross supplied
ranges. Their admission and effects remain operator-specific: the request does
not guarantee return to the range, restoration of an original form or long-term
stability. Targets and invocation schedules are not derived nodal laws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from ..alias import get_attr
from ..config.operator_names import COHERENCE, CONTRACTION, EMISSION, EXPANSION, SILENCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..operators.registry import get_operator_class

__all__ = ["StructuralHomeostasis"]


class StructuralHomeostasis:
    """Request parameter corrections under a supplied range policy.

    This class samples EPI, νf and ΔNFR and selects operator requests from
    supplied ranges. The ranges are a controller policy, not an emergent
    equilibrium or a stability certificate. The sampled values are held for
    selection during the call; execution still uses each operator's live
    admission and transformation.

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the regulated node
    node : NodeId
        Identifier of the node to regulate
    epi_range : tuple[float, float], optional
        Target EPI range (min, max). Default: (0.4, 0.8)
    vf_range : tuple[float, float], optional
        Target νf range (min, max). Default: (0.8, 1.2)
    dnfr_range : tuple[float, float], optional
        Target ΔNFR range (min, max). Default: (0.0, 0.15)

    Attributes
    ----------
    G : TNFRGraph
        Graph reference
    node : NodeId
        Node identifier
    epi_range : tuple[float, float]
        Target EPI range (min, max)
    vf_range : tuple[float, float]
        Target νf range (min, max)
    dnfr_range : tuple[float, float]
        Target ΔNFR range (min, max)

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.homeostasis import StructuralHomeostasis
    >>> G, node = create_nfr("test_node")
    >>> homeostasis = StructuralHomeostasis(G, node)
    >>> homeostasis.maintain_equilibrium()

    Notes
    -----
    The configured selection map requests Emission (AL) for low signed EPI,
    Contraction (NUL) for high signed EPI, Expansion (VAL) for low capacity,
    Silence (SHA) for high capacity and Coherence (IL) for high signed pressure.
    Contraction changes capacity, not EPI; Coherence changes phase and does not
    guarantee lower pressure. No request guarantees a return to its triggering
    range. The lower pressure endpoint is retained configuration but is not
    consumed by this policy.
    """

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        epi_range: tuple[float, float] = (0.4, 0.8),
        vf_range: tuple[float, float] = (0.8, 1.2),
        dnfr_range: tuple[float, float] = (0.0, 0.15),
    ) -> None:
        self.G = graph
        self.node = node

        # Target ranges for homeostatic regulation
        self.epi_range = epi_range
        self.vf_range = vf_range
        self.dnfr_range = dnfr_range

    def maintain_equilibrium(self) -> None:
        """Request operators selected by the initial parameter-range snapshot.

        Multiple requests can occur in one call. Each public operator checks
        its live preconditions, and a rejection propagates without rolling back
        earlier completed requests.

        Notes
        -----
        Requests are applied sequentially:

        1. Select from the sampled EPI range
        2. Select from the sampled capacity range
        3. Select from the sampled upper pressure bound

        At most one operator is requested for each sampled parameter.
        """
        epi = get_attr(self.G.nodes[self.node], ALIAS_EPI, 0.0)
        vf = get_attr(self.G.nodes[self.node], ALIAS_VF, 1.0)
        dnfr = get_attr(self.G.nodes[self.node], ALIAS_DNFR, 0.0)

        # Select from the sampled signed EPI.
        if epi < self.epi_range[0]:
            # EPI too low → activate with Emission
            operator_class = get_operator_class(EMISSION)
            operator = operator_class()
            operator(self.G, self.node)
        elif epi > self.epi_range[1]:
            # High EPI selects a capacity-contraction request.
            operator_class = get_operator_class(CONTRACTION)
            operator = operator_class()
            operator(self.G, self.node)

        # Select from the sampled capacity.
        if vf < self.vf_range[0]:
            # Frequency too low → expand with Expansion
            operator_class = get_operator_class(EXPANSION)
            operator = operator_class()
            operator(self.G, self.node)
        elif vf > self.vf_range[1]:
            # Frequency too high → silence with Silence
            operator_class = get_operator_class(SILENCE)
            operator = operator_class()
            operator(self.G, self.node)

        # Select from the sampled signed upper pressure bound.
        if dnfr > self.dnfr_range[1]:
            # ΔNFR too high → stabilize with Coherence
            operator_class = get_operator_class(COHERENCE)
            operator = operator_class()
            operator(self.G, self.node)
