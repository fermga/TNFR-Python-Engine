"""TNFR Operator: Coherence

Coherence structural operator - Stabilization of structural alignment.

**Physics**: See AGENTS.md § Coherence
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import COHERENCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from ..utils import get_numpy
from .definitions_base import Operator


class Coherence(Operator):
    """Coherence structural operator - Stabilization of structural alignment.

    Activates the Coherence operator to compress ΔNFR drift and raise the local C(t),
    reinforcing structural alignment across nodes and stabilizing emergent forms.

    TNFR Context
    ------------
    Coherence represents the fundamental stabilization process in TNFR. When applied,
    it reduces ΔNFR (reorganization pressure) and increases C(t) (global coherence),
    effectively "sealing" structural forms into stable configurations. This is the primary
    operator for maintaining nodal equation balance: ∂EPI/∂t → 0 as ΔNFR → 0.

    **Key Elements:**

    - **Structural Stabilization**: Reduces reorganization pressure (ΔNFR)
    - **Coherence Amplification**: Increases global C(t) through local stability
    - **Form Preservation**: Maintains EPI integrity across time
    - **Phase Locking**: Synchronizes node with network phase structure

    Use Cases
    ---------
    **Biomedical**:

    - **Cardiac Coherence**: Stabilizing heart rate variability patterns
    - **Neural Coherence**: Maintaining synchronized brain wave states
    - **Homeostatic Balance**: Stabilizing physiological regulatory systems
    - **Therapeutic Integration**: Consolidating healing states post-intervention

    **Cognitive**:

    - **Concept Consolidation**: Stabilizing newly learned information
    - **Mental Clarity**: Reducing cognitive noise and confusion
    - **Focus Maintenance**: Sustaining attention on coherent thought patterns
    - **Memory Formation**: Consolidating experience into stable memories

    **Social**:

    - **Team Alignment**: Stabilizing collaborative working patterns
    - **Cultural Coherence**: Maintaining shared values and practices
    - **Ritual Completion**: Sealing ceremonial transformations
    - **Group Synchrony**: Stabilizing collective resonance states

    Typical Sequences
    ---------------------------
    - **Emission → Reception → Coherence**: Safe activation with stabilization
    - **Reception → Coherence**: Integrated reception consolidated
    - **Coherence → Mutation**: Coherence enabling controlled mutation (stable transformation)
    - **Resonance → Coherence**: Resonance followed by stabilization (propagation consolidation)
    - **Coupling → Coherence**: Network coupling stabilized into coherent form

    Preconditions
    -------------
    - Node must have active EPI (non-zero form)
    - ΔNFR should be present (though Coherence reduces it)
    - Sufficient network coupling for phase alignment

    Structural Effects
    ------------------
    - **EPI**: May increment slightly as form stabilizes
    - **ΔNFR**: Significantly reduces (primary effect)
    - **C(t)**: Increases at both local and global levels
    - **νf**: May slightly increase as stability enables higher frequency
    - **θ**: Aligns with network phase (phase locking)

    Metrics
    -----------------
    - ΔNFR reduction: Primary metric of stabilization success
    - C(t) increase: Global coherence improvement
    - Phase alignment: Degree of network synchronization
    - EPI stability: Variance reduction in form over time

    Compatibility
    ---------------------
    **Compatible with**: ALL operators - Coherence is universally stabilizing

    **Especially effective after**: Emission, Reception, Dissonance, Transition

    **Natural progressions**: Coherence often concludes sequences or prepares for
    controlled transformation (Mutation, Transition)

    Examples
    --------
    **Cardiac Coherence Training:**

    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Emission, Reception, Coherence, Coupling, Resonance, Transition
    >>> from tnfr.alias import get_attr
    >>> from tnfr.constants.aliases import ALIAS_EPI
    >>>
    >>> # Stabilizing heart rhythm during breath-focus training
    >>> G_heart, heart = create_nfr("cardiac_rhythm", epi=0.40, vf=1.10)
    >>>
    >>> # Valid sequence: Emission → Reception → Coherence → Coupling → Resonance → Transition
    >>> run_sequence(G_heart, heart,
    ...     [Emission(), Reception(), Coherence(), Coupling(), Resonance(), Transition()])
    >>>
    >>> # Result: HRV pattern stabilizes, ΔNFR reduces
    >>> epi_final = float(get_attr(G_heart.nodes[heart], ALIAS_EPI, 0.0))
    >>> # Patient enters sustained coherent state

    **Learning Consolidation:**

    >>> # Student consolidating newly understood concept
    >>> G_study, mind = create_nfr("student_understanding", epi=0.30, vf=1.05)
    >>>
    >>> # Receive teaching and consolidate understanding
    >>> run_sequence(G_study, mind,
    ...     [Emission(), Reception(), Coherence(), Coupling(), Resonance(), Transition()])
    >>>
    >>> # Result: Knowledge structure stabilizes, confusion reduces
    >>> # Concept becomes part of stable mental model

    **Team Alignment:**

    >>> # Collaborative team stabilizing after creative session
    >>> G_team, group = create_nfr("team_consensus", epi=0.55, vf=1.00)
    >>>
    >>> # Build consensus through coupling and coherence
    >>> run_sequence(G_team, group,
    ...     [Emission(), Reception(), Coupling(), Coherence(), Resonance(), Transition()])
    >>>
    >>> # Result: Group coherence increases, conflicts resolve
    >>> # Team operates with unified purpose

    See Also
    --------
    Dissonance : Creates instability that Coherence later resolves
    Emission : Often followed by Coherence for safe activation
    Mutation : Coherence enables controlled phase changes
    """

    __slots__ = ()
    name: ClassVar[str] = COHERENCE
    glyph: ClassVar[Glyph] = Glyph.IL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply Coherence with explicit ΔNFR reduction, C(t) coherence tracking, and phase locking.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to grammar layer via parent __call__.
            Special keys:
            - coherence_radius (int): Radius for local coherence computation (default: 1)
            - phase_locking_coefficient (float): Phase alignment strength α ∈ [0.1, 0.5] (default: 0.3)

        Notes
        -----
        This implementation enforces the canonical Coherence structural effect:
        ΔNFR → ΔNFR * (1 - ρ) where ρ ≈ 0.3 (30% reduction).

        The reduction is applied by the grammar layer using the Coherence dnfr_factor
        from global glyph factors. This method adds explicit telemetry logging for
        structural traceability.

        **C(t) Coherence Tracking:**

        Captures global and local coherence before and after Coherence application:
        - C_global: Network-wide coherence using C(t) = 1 - (σ_ΔNFR / ΔNFR_max)
        - C_local: Node neighborhood coherence with configurable radius

        Both metrics are stored in G.graph["IL_coherence_tracking"] for analysis.

        **Phase Locking:**

        Aligns node phase θ with network neighborhood phase:
        - θ_node → θ_node + α * (θ_network - θ_node)
        - Uses circular mean for proper phase wrap-around handling
        - Telemetry stored in G.graph["IL_phase_locking"]

        To customize the reduction factor, set GLYPH_FACTORS["IL_dnfr_factor"] in
        the graph before calling this operator. Default is 0.7 (30% reduction).
        """
        # Import here to avoid circular import
        from ..metrics.coherence import (
            compute_global_coherence,
            compute_local_coherence,
        )

        # Capture C(t) before Coherence application
        C_global_before = compute_global_coherence(G)
        C_local_before = compute_local_coherence(G, node, radius=kw.get("coherence_radius", 1))

        # Capture ΔNFR before Coherence application for telemetry
        dnfr_before = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Delegate to parent __call__ which applies grammar (including Coherence reduction)
        super().__call__(G, node, **kw)

        # Apply phase locking after grammar application
        locking_coef = kw.get("phase_locking_coefficient", 0.3)
        self._apply_phase_locking(G, node, locking_coefficient=locking_coef)

        # Capture C(t) after IL application
        C_global_after = compute_global_coherence(G)
        C_local_after = compute_local_coherence(G, node, radius=kw.get("coherence_radius", 1))

        # Capture ΔNFR after IL application for telemetry
        dnfr_after = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Store C(t) tracking in graph telemetry
        if "IL_coherence_tracking" not in G.graph:
            G.graph["IL_coherence_tracking"] = []

        G.graph["IL_coherence_tracking"].append(
            {
                "node": node,
                "C_global_before": C_global_before,
                "C_global_after": C_global_after,
                "C_global_delta": C_global_after - C_global_before,
                "C_local_before": C_local_before,
                "C_local_after": C_local_after,
                "C_local_delta": C_local_after - C_local_before,
            }
        )

        # Log ΔNFR reduction in graph metadata for telemetry
        if "IL_dnfr_reductions" not in G.graph:
            G.graph["IL_dnfr_reductions"] = []

        # Calculate actual reduction factor from before/after values
        actual_reduction_factor = (
            (dnfr_before - dnfr_after) / dnfr_before if dnfr_before > 0 else 0.0
        )

        G.graph["IL_dnfr_reductions"].append(
            {
                "node": node,
                "before": dnfr_before,
                "after": dnfr_after,
                "reduction": dnfr_before - dnfr_after,
                "reduction_factor": actual_reduction_factor,
            }
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate IL-specific preconditions."""
        from .preconditions import validate_coherence

        validate_coherence(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect IL-specific metrics."""
        from .metrics import coherence_metrics

        return coherence_metrics(G, node, state_before["dnfr"])

    def _apply_phase_locking(
        self, G: TNFRGraph, node: Any, locking_coefficient: float = 0.3
    ) -> None:
        """Align node phase θ with network neighborhood phase.

        Implements canonical IL phase locking:
        θ_node → θ_node + α * (θ_network - θ_node)

        where α ∈ [0.1, 0.5] is the phase locking coefficient (default: 0.3).

        Parameters
        ----------
        G : TNFRGraph
            Network graph
        node : Any
            Target node
        locking_coefficient : float
            Phase alignment strength α, default 0.3

        Notes
        -----
        **Canonical Specification:**

        IL operator synchronizes node phase with its network neighborhood:

        1. Compute network phase θ_network as circular mean of neighbor phases
        2. Compute phase difference Δθ = θ_network - θ_node (shortest arc)
        3. Apply locking: θ_new = θ_node + α * Δθ
        4. Normalize θ_new to [0, 2π]

        **Circular Statistics:**

        Phase averaging uses complex exponentials to handle wrap-around at 2π:
        - Convert phases to e^(iθ)
        - Compute mean of complex phasors
        - Extract angle as network phase

        This ensures correct averaging (e.g., 0.1 and 6.2 radians average to ~0).

        **Telemetry:**

        Stores detailed phase locking information in G.graph["IL_phase_locking"]:
        - theta_before, theta_after: Node phase before/after locking
        - theta_network: Network neighborhood mean phase
        - delta_theta: Phase difference (shortest arc)
        - alignment_achieved: Residual misalignment after locking

        **Special Cases:**

        - No neighbors: Phase unchanged (no network to align with)
        - Single neighbor: Aligns toward that neighbor's phase
        - Isolated node: No-op (returns immediately)

        See Also
        --------
        metrics.phase_coherence.compute_phase_alignment : Measure alignment quality
        """
        from ..alias import set_attr
        from ..constants.aliases import ALIAS_THETA

        # Get current node phase
        theta_node = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

        # Get neighbor phases
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return  # No neighbors, no phase locking

        theta_neighbors = [float(get_attr(G.nodes[n], ALIAS_THETA, 0.0)) for n in neighbors]

        # Compute mean phase using circular mean (angles wrap around 2π)
        # Convert to complex exponentials for circular averaging
        np = get_numpy()

        if np is not None:
            # NumPy vectorized computation
            theta_array = np.array(theta_neighbors)
            complex_phases = np.exp(1j * theta_array)
            mean_complex = np.mean(complex_phases)
            theta_network = np.angle(mean_complex)  # Returns value in [-π, π]

            # Ensure positive phase [0, 2π]
            if theta_network < 0:
                theta_network = float(theta_network + 2 * np.pi)
            else:
                theta_network = float(theta_network)

            # Compute phase difference (considering wrap-around)
            delta_theta = theta_network - theta_node

            # Normalize to [-π, π] for shortest angular distance
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            elif delta_theta < -np.pi:
                delta_theta += 2 * np.pi
            delta_theta = float(delta_theta)

            # Apply phase locking: move θ toward network mean
            theta_new = theta_node + locking_coefficient * delta_theta

            # Normalize to [0, 2π]
            theta_new = float(theta_new % (2 * np.pi))
            import cmath
            import math

            # Convert phases to complex exponentials
            complex_phases = [cmath.exp(1j * theta) for theta in theta_neighbors]

            # Compute mean complex phasor
            mean_real = sum(z.real for z in complex_phases) / len(complex_phases)
            mean_imag = sum(z.imag for z in complex_phases) / len(complex_phases)
            mean_complex = complex(mean_real, mean_imag)

            # Extract angle (in [-π, π])
            theta_network = cmath.phase(mean_complex)

            # Ensure positive phase [0, 2π]
            if theta_network < 0:
                theta_network += 2 * math.pi

            # Compute phase difference (considering wrap-around)
            delta_theta = theta_network - theta_node

            # Normalize to [-π, π] for shortest angular distance
            if delta_theta > math.pi:
                delta_theta -= 2 * math.pi
            elif delta_theta < -math.pi:
                delta_theta += 2 * math.pi

            # Apply phase locking: move θ toward network mean
            theta_new = theta_node + locking_coefficient * delta_theta

            # Normalize to [0, 2π]
            theta_new = theta_new % (2 * math.pi)

        # Update node phase
        set_attr(G.nodes[node], ALIAS_THETA, theta_new)

        # Store phase locking telemetry
        if "IL_phase_locking" not in G.graph:
            G.graph["IL_phase_locking"] = []

        G.graph["IL_phase_locking"].append(
            {
                "node": node,
                "theta_before": theta_node,
                "theta_after": theta_new,
                "theta_network": theta_network,
                "delta_theta": delta_theta,
                "alignment_achieved": abs(delta_theta) * (1 - locking_coefficient),
            }
        )