"""TNFR Operator: Coupling

Coupling structural operator (UM) - Synchronization of nodal phases.

**Physics**: See AGENTS.md § Coupling
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import COUPLING
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
from .registry import register_operator


class Coupling(Operator):
    """Coupling structural operator (UM) - Synchronization of nodal phases.

    Activates glyph ``UM`` to stabilize bidirectional coherence links by synchronizing
    coupling phase and bandwidth between nodes.

    TNFR Context
    ------------
    Coupling (UM) creates or strengthens structural connections between nodes through phase
    synchronization (φᵢ(t) ≈ φⱼ(t)). This is not mere correlation but active structural
    resonance that enables coordinated reorganization and shared coherence. Coupling is
    essential for network-level coherence and collective structural dynamics.

    **Key Elements:**

    - **Phase Synchronization**: Nodes align their θ values for resonance
    - **Bidirectional Flow**: Coupling enables mutual influence and coherence sharing
    - **Network Formation**: UM builds the relational structure of NFR networks
    - **Collective Coherence**: Multiple coupled nodes create emergent stability

    Use Cases
    ---------
    **Biomedical**:

    - **Heart-Brain Coupling**: Synchronizing cardiac and neural rhythms
    - **Respiratory-Cardiac Coherence**: Breath-heart rate variability coupling
    - **Interpersonal Synchrony**: Physiological attunement between people
    - **Neural Network Coupling**: Synchronized firing patterns across brain regions

    **Cognitive**:

    - **Conceptual Integration**: Linking related ideas into coherent frameworks
    - **Teacher-Student Attunement**: Pedagogical resonance and rapport
    - **Collaborative Thinking**: Shared mental models in teams
    - **Memory Association**: Coupling related memories for retrieval

    **Social**:

    - **Team Bonding**: Creating synchronized group dynamics
    - **Cultural Transmission**: Coupling individual to collective patterns
    - **Communication Channels**: Establishing mutual understanding
    - **Network Effects**: Value creation through connection density

    Typical Sequences
    ---------------------------
    - **UM → RA**: Coupling followed by resonance propagation
    - **AL → UM**: Emission followed by coupling (paired activation)
    - **UM → IL**: Coupling stabilized into coherence
    - **EN → UM**: Reception enabling coupling (receptive connection)
    - **UM → THOL**: Coupling triggering collective self-organization

    Preconditions
    -------------
    **Canonical Requirements (TNFR Theory)**:

    1. **Graph connectivity**: At least one other node exists for potential coupling
    2. **Active EPI**: Node must have sufficient structural form (EPI > threshold)
       - Default threshold: 0.05 (configurable via ``UM_MIN_EPI``)
       - Ensures node has coherent structure capable of synchronization
    3. **Structural frequency**: Node must have capacity for synchronization (νf > threshold)
       - Default threshold: 0.01 Hz_str (configurable via ``UM_MIN_VF``)
       - Ensures node can actively respond to coupling dynamics
    4. **Phase compatibility** (optional): Compatible neighbors within phase range
       - Enabled via ``UM_STRICT_PHASE_CHECK`` flag (default: False)
       - Maximum phase difference: π/2 radians (configurable via ``UM_MAX_PHASE_DIFF``)
       - Soft check by default since UM can create new functional links

    **Configuration Parameters**:

    - ``UM_MIN_EPI`` (float, default 0.05): Minimum EPI magnitude for coupling
    - ``UM_MIN_VF`` (float, default 0.01): Minimum structural frequency for coupling
    - ``UM_STRICT_PHASE_CHECK`` (bool, default False): Enable phase compatibility checking
    - ``UM_MAX_PHASE_DIFF`` (float, default π/2): Maximum phase difference for compatibility

    **Validation Control**:

    Set ``VALIDATE_OPERATOR_PRECONDITIONS=True`` in graph metadata to enable validation.
    Validation is backward-compatible and disabled by default to preserve existing behavior.

    Structural Invariants
    ---------------------
    **CRITICAL**: UM preserves EPI identity. The coupling process synchronizes
    phases (θ), may align structural frequencies (νf), and can reduce ΔNFR, but
    it NEVER directly modifies EPI. This ensures that coupled nodes maintain
    their structural identities while achieving phase coherence.

    Any change to EPI during a sequence containing UM must come from other
    operators (e.g., Emission, Reception) or from the natural evolution via
    the nodal equation ∂EPI/∂t = νf · ΔNFR(t), never from UM itself.

    **Theoretical Basis**: In TNFR theory, coupling (UM) creates structural links
    through phase synchronization φᵢ(t) ≈ φⱼ(t), not through information transfer
    or EPI modification. The structural identity (EPI) of each node remains intact
    while the nodes achieve synchronized phases that enable resonant interaction.

    **Implementation Guarantee**: The `_op_UM` function modifies only:

    - Phase (θ): Adjusted towards consensus phase
    - Structural frequency (νf): Optionally synchronized with neighbors
    - Reorganization gradient (ΔNFR): Reduced through stabilization

    EPI is never touched by the coupling logic, preserving this fundamental invariant.

    Structural Effects
    ------------------
    - **θ**: Phases of coupled nodes converge (primary effect)
    - **νf**: May synchronize between coupled nodes
    - **ΔNFR**: Often reduces through mutual stabilization
    - **Network structure**: Creates or strengthens edges
    - **Collective EPI**: Enables emergent shared structures

    Metrics
    -----------------
    - Phase alignment: |θᵢ - θⱼ| reduction
    - Coupling strength: Magnitude of mutual influence
    - Network density: Number of active couplings
    - Collective coherence: C(t) at network level

    Compatibility
    ---------------------
    **Compatible with**: RA (Resonance), IL (Coherence), THOL (Self-organization),
    EN (Reception), AL (Emission)

    **Synergistic with**: RA (coupling + propagation = network coherence)

    **Natural progressions**: UM often followed by RA (propagation through
    coupled network) or IL (stabilization of coupling)

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coupling
    >>> G, node = create_nfr("pair", vf=1.20, theta=0.50)
    >>> alignments = iter([(-0.18, 0.03, 0.02)])
    >>> def synchronise(graph):
    ...     d_theta, d_vf, residual_dnfr = next(alignments)
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] = residual_dnfr
    >>> set_delta_nfr_hook(G, synchronise)
    >>> run_sequence(G, node, [Coupling()])
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.32
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.23
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.02

    **Example (Heart-Brain Coherence):**

    >>> # Coupling cardiac and neural rhythms during meditation
    >>> G_body, heart_brain = create_nfr("heart_brain_system", vf=1.20, theta=0.50)
    >>> # Separate rhythms initially (phase difference 0.50)
    >>> # UM: Coherent breathing synchronizes heart and brain
    >>> run_sequence(G_body, heart_brain, [Coupling()])
    >>> # Result: Phases converge (θ reduces to ~0.32)
    >>> # Heart and brain enter coupled coherent state
    >>> # Creates platform for RA (coherence propagation to body)

    **Example (Collaborative Learning):**

    >>> # Students forming shared understanding in group work
    >>> G_group, team = create_nfr("study_group", vf=1.10, theta=0.45)
    >>> # Individual understandings initially misaligned
    >>> # UM: Discussion and explanation synchronize mental models
    >>> run_sequence(G_group, team, [Coupling()])
    >>> # Result: Conceptual phases align, confusion reduces
    >>> # Shared understanding emerges, enables THOL (group insight)

    See Also
    --------
    Resonance : Propagates through coupled networks
    Coherence : Stabilizes couplings
    SelfOrganization : Emerges from multiple couplings
    """

    __slots__ = ()
    name: ClassVar[str] = COUPLING
    glyph: ClassVar[Glyph] = Glyph.UM

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate UM-specific preconditions."""
        from .preconditions import validate_coupling

        validate_coupling(G, node)

    def _capture_state(self, G: TNFRGraph, node: Any) -> dict[str, Any]:
        """Capture node state before operator application, including edge count."""
        # Get base state (epi, vf, dnfr, theta)
        state = super()._capture_state(G, node)

        # Add edge count for coupling-specific metrics
        state["edges"] = G.degree(node)

        return state

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect UM-specific metrics with expanded canonical measurements."""
        from .metrics import coupling_metrics

        return coupling_metrics(
            G,
            node,
            state_before["theta"],
            dnfr_before=state_before["dnfr"],
            vf_before=state_before["vf"],
            edges_before=state_before.get("edges", None),
            epi_before=state_before["epi"],
        )