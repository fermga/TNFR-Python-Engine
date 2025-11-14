"""TNFR Operator: Dissonance

Dissonance structural operator (OZ) - Creative instability for exploration.

**Physics**: See AGENTS.md § Dissonance
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import DISSONANCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
from .registry import register_operator


class Dissonance(Operator):
    """Dissonance structural operator (OZ) - Creative instability for exploration.

    Activates structural symbol ``OZ`` to widen ΔNFR and test bifurcation thresholds,
    injecting controlled dissonance to probe system robustness and enable transformation.

    TNFR Context
    ------------
    Dissonance (OZ) is the creative force in TNFR - it deliberately increases ΔNFR and
    phase instability (θ) to explore new structural configurations. Rather than destroying
    coherence, controlled dissonance enables evolution, mutation, and creative reorganization.
    When ∂²EPI/∂t² > τ, bifurcation occurs, spawning new structural possibilities.

    **Key Elements:**

    - **Creative Instability**: Necessary for transformation and evolution
    - **Bifurcation Trigger**: When ΔNFR exceeds thresholds, new forms emerge
    - **Controlled Chaos**: Dissonance is managed, not destructive
    - **Phase Exploration**: θ variation opens new network couplings

    Use Cases
    ---------
    **Biomedical**:

    - **Hormetic Stress**: Controlled physiological challenge (cold exposure, fasting)
    - **Therapeutic Crisis**: Necessary discomfort in healing process
    - **Immune Challenge**: Controlled pathogen exposure for adaptation
    - **Neural Plasticity**: Learning-induced temporary destabilization

    **Cognitive**:

    - **Cognitive Dissonance**: Challenging existing beliefs for growth
    - **Creative Problem-Solving**: Introducing paradoxes to spark insight
    - **Socratic Method**: Questioning to destabilize and rebuild understanding
    - **Conceptual Conflict**: Encountering contradictions that force reorganization

    **Social**:

    - **Constructive Conflict**: Productive disagreement in teams
    - **Organizational Change**: Disrupting status quo to enable transformation
    - **Cultural Evolution**: Introducing new ideas that challenge norms
    - **Innovation Pressure**: Market disruption forcing adaptation

    Typical Sequences
    ---------------------------
    - **OZ → IL**: Dissonance resolved into new coherence (creative resolution)
    - **OZ → THOL**: Dissonance triggering self-organization (emergent order)
    - **IL → OZ → THOL**: Stable → dissonance → reorganization (growth cycle)
    - **OZ → NAV → IL**: Dissonance → transition → new stability
    - **AL → OZ → RA**: Activation → challenge → propagation (tested resonance)

    **AVOID**: OZ → SHA (dissonance followed by silence contradicts exploration)

    Preconditions
    -------------
    - Node must have baseline coherence to withstand dissonance
    - Network must support potential bifurcations
    - ΔNFR should not already be critically high

    Structural Effects
    ------------------
    - **ΔNFR**: Significantly increases (primary effect)
    - **θ**: May shift unpredictably (phase exploration)
    - **EPI**: May temporarily destabilize before reorganizing
    - **νf**: Often increases as system responds to challenge
    - **Bifurcation risk**: ∂²EPI/∂t² may exceed τ

    Metrics
    -----------------
    - ΔNFR increase: Magnitude of introduced instability
    - Phase shift (Δθ): Degree of phase exploration
    - Bifurcation events: Count of structural splits
    - Recovery time: Time to return to coherence (with IL)

    Compatibility
    ---------------------
    **Compatible with**: IL (resolution), THOL (organization), NAV (transition),
    ZHIR (mutation)

    **Avoid with**: SHA (silence), multiple consecutive OZ (excessive instability)

    **Natural progressions**: OZ typically followed by IL (stabilization) or
    THOL (self-organization) to resolve created instability

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Dissonance
    >>> G, node = create_nfr("probe", theta=0.10)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.02
    >>> shocks = iter([(0.09, 0.15)])
    >>> def inject(graph):
    ...     d_dnfr, d_theta = next(shocks)
    ...     graph.nodes[node][DNFR_PRIMARY] += d_dnfr
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    >>> set_delta_nfr_hook(G, inject)
    >>> run_sequence(G, node, [Dissonance()])
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.11
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.25

    **Example (Therapeutic Challenge):**

    >>> # Patient confronting difficult emotions in therapy
    >>> G_therapy, patient = create_nfr("emotional_processing", epi=0.40, theta=0.10)
    >>> # Stable baseline, low phase variation
    >>> # OZ: Therapist guides patient to face uncomfortable truth
    >>> run_sequence(G_therapy, patient, [Dissonance()])
    >>> # Result: ΔNFR increases (emotional turbulence)
    >>> # Phase shifts as old patterns destabilize
    >>> # Prepares for THOL (new understanding) or IL (integration)

    **Example (Educational Challenge):**

    >>> # Student encountering paradox that challenges understanding
    >>> G_learning, student = create_nfr("conceptual_framework", epi=0.50, theta=0.15)
    >>> # Established understanding with moderate phase stability
    >>> # OZ: Teacher presents evidence contradicting current model
    >>> run_sequence(G_learning, student, [Dissonance()])
    >>> # Result: Cognitive dissonance creates ΔNFR spike
    >>> # Existing mental model destabilizes
    >>> # Enables THOL (conceptual reorganization) or ZHIR (paradigm shift)

    **Example (Organizational Innovation):**

    >>> # Company facing market disruption
    >>> G_org, company = create_nfr("business_model", epi=0.60, theta=0.20)
    >>> # Established business model with some flexibility
    >>> # OZ: Disruptive competitor enters market
    >>> run_sequence(G_org, company, [Dissonance()])
    >>> # Result: Organizational ΔNFR increases (uncertainty, pressure)
    >>> # Business model phase shifts (exploring new strategies)
    >>> # Creates conditions for THOL (innovation) or NAV (pivot)

    See Also
    --------
    Coherence : Resolves dissonance into new stability
    SelfOrganization : Organizes dissonance into emergent forms
    Mutation : Controlled phase change often enabled by OZ
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
            - propagate_to_network: Enable propagation (default: True if OZ_ENABLE_PROPAGATION in G.graph)
            - propagation_mode: 'phase_weighted' (default), 'uniform', 'frequency_weighted'
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
        propagate = kw.get("propagate_to_network", G.graph.get("OZ_ENABLE_PROPAGATION", True))
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

        return dissonance_metrics(G, node, state_before["dnfr"], state_before["theta"])