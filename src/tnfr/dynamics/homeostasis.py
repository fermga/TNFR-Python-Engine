"""Structural homeostasis for TNFR nodes.

This module implements homeostatic regulation that maintains nodal parameters
within target ranges. When parameters drift outside acceptable bounds, corrective
operators are automatically applied to restore equilibrium.

Homeostasis ensures long-term stability while allowing dynamic adaptation
within safe operating ranges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..operators.registry import get_operator_class
from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    EMISSION,
    EXPANSION,
    SILENCE,
)

__all__ = ["StructuralHomeostasis"]


class StructuralHomeostasis:
    """Maintains dynamic equilibrium in nodal parameters.

    This class monitors EPI, νf, and ΔNFR values and applies corrective operators
    when they drift outside target ranges. The goal is to maintain healthy
    structural dynamics without constraining natural evolution.

    **Homeostatic Principles:**

    - **EPI range**: Maintain adequate activation without saturation
    - **νf range**: Keep reorganization rate within functional bounds
    - **ΔNFR range**: Prevent excessive reorganization pressure

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
    Corrective operators follow TNFR canonical principles:

    - **Low EPI**: Apply AL (Emission) to activate
    - **High EPI**: Apply NUL (Contraction) to reduce
    - **Low νf**: Apply VAL (Expansion) to increase frequency
    - **High νf**: Apply SHA (Silence) to slow down
    - **High ΔNFR**: Apply IL (Coherence) to stabilize
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
        """Apply corrective operators if parameters exceed target ranges.

        Checks each parameter (EPI, νf, ΔNFR) and applies appropriate
        operators when out of bounds. Multiple corrections can occur
        in a single call if multiple parameters are out of range.

        Notes
        -----
        Corrections are applied sequentially:

        1. Check and correct EPI
        2. Check and correct νf
        3. Check and correct ΔNFR

        Each correction is minimal: one operator application per parameter.
        """
        epi = get_attr(self.G.nodes[self.node], ALIAS_EPI, 0.0)
        vf = get_attr(self.G.nodes[self.node], ALIAS_VF, 1.0)
        dnfr = get_attr(self.G.nodes[self.node], ALIAS_DNFR, 0.0)

        # Correct EPI if out of range
        if epi < self.epi_range[0]:
            # EPI too low → activate with Emission
            operator_class = get_operator_class(EMISSION)
            operator = operator_class()
            operator(self.G, self.node)
        elif epi > self.epi_range[1]:
            # EPI too high → contract with Contraction
            operator_class = get_operator_class(CONTRACTION)
            operator = operator_class()
            operator(self.G, self.node)

        # Correct νf if out of range
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

        # Correct ΔNFR if too high
        if dnfr > self.dnfr_range[1]:
            # ΔNFR too high → stabilize with Coherence
            operator_class = get_operator_class(COHERENCE)
            operator = operator_class()
            operator(self.G, self.node)
