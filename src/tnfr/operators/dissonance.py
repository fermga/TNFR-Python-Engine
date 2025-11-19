"""Dissonance (OZ) operator.

Purpose: inject controlled instability; widen dnfr; test bifurcation.
Physics: raises structural pressure; may push second derivative over tau.
Grammar: destabilizer (U2); trigger (U4a); closure-capable.
Effects: dnfr up; phase may wander; vf often rises; epi stressed.
Preconditions: sufficient epi/vf; dnfr below critical; prior stability.
Typical: OZ->IL; OZ->THOL; IL->OZ->THOL growth cycle; AL->OZ->RA.
Avoid: OZ->SHA; repeated OZ without IL/THOL containment.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import DISSONANCE
from ..constants.aliases import ALIAS_DNFR
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Dissonance(Operator):
    """Raise dnfr; induce exploratory instability; probe bifurcation.

    Contracts: must increase dnfr; follow with IL/THOL. Avoid SHA
    immediately. See also: Coherence, SelfOrganization, Mutation.
    """

    __slots__ = ()
    name: ClassVar[str] = DISSONANCE
    glyph: ClassVar[Glyph] = Glyph.OZ

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply OZ with optional network propagation.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes
        node : Any
            Target node identifier
        **kw : Any
            Additional keyword arguments:
            - propagate_to_network: enable propagation (default True)
            - propagation_mode: phase_weighted | uniform | frequency_weighted
            - Other arguments forwarded to base Operator.__call__
        """
        # Capture state before for propagation computation
        dnfr_before = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Apply standard operator logic via parent
        super().__call__(G, node, **kw)

        # Compute dissonance increase
        dnfr_after = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
        dissonance_magnitude = abs(dnfr_after - dnfr_before)

        # Propagate to network if enabled
        propagate = kw.get(
            "propagate_to_network",
            G.graph.get("OZ_ENABLE_PROPAGATION", True),
        )
        if propagate and dissonance_magnitude > 0:
            from ..dynamics.propagation import propagate_dissonance

            affected = propagate_dissonance(
                G,
                node,
                dissonance_magnitude,
                propagation_mode=kw.get("propagation_mode", "phase_weighted"),
            )

            # Store propagation telemetry
            if "_oz_propagation_events" not in G.graph:
                G.graph["_oz_propagation_events"] = []
            G.graph["_oz_propagation_events"].append(
                {
                    "source": node,
                    "magnitude": dissonance_magnitude,
                    "affected_nodes": list(affected),
                    "affected_count": len(affected),
                }
            )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate OZ-specific preconditions."""
        from .preconditions import validate_dissonance

        validate_dissonance(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect OZ-specific metrics."""
        from .metrics import dissonance_metrics

        return dissonance_metrics(
            G,
            node,
            state_before["dnfr"],
            state_before["theta"],
        )
