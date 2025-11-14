"""TNFR Operator: Mutation

Mutation structural operator (ZHIR) - Controlled phase transformation.

**Physics**: See AGENTS.md § Mutation
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import MUTATION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Mutation(Operator):
    """Mutation structural operator (ZHIR) - Controlled phase transformation.

    Activates glyph ``ZHIR`` to recode phase or form, enabling the node to cross
    structural thresholds and pivot towards a new coherence regime.

    TNFR Context
    ------------
    Mutation (ZHIR) implements the fundamental phase transformation mechanism in TNFR:
    θ → θ' when structural velocity ∂EPI/∂t exceeds threshold ξ. This is NOT random
    variation but controlled structural transformation that preserves identity (epi_kind)
    while shifting operational regime. ZHIR enables qualitative state changes without
    losing coherent structural continuity.

    **Derivation from Nodal Equation**:

    From the nodal equation ∂EPI/∂t = νf · ΔNFR(t), when reorganization pressure builds
    up (ΔNFR elevated) and transformation capacity exists (νf > 0), structural velocity
    increases. At threshold crossing (∂EPI/∂t > ξ), the system has sufficient momentum
    for phase transformation without fragmenting coherence.

    **Key Elements:**

    - **Phase Transformation**: θ → θ' shifts operational regime
    - **Identity Preservation**: epi_kind maintained through transformation
    - **Threshold-Controlled**: Requires ∂EPI/∂t > ξ for justification
    - **Bifurcation Detection**: Monitors ∂²EPI/∂t² for instability
    - **Grammar U4b**: Requires prior IL and recent destabilizer

    **ZHIR vs Random Mutation**:

    Traditional mutation (biology, evolutionary algorithms) is stochastic variation.
    TNFR mutation is deterministic reorganization triggered by structural conditions.
    It's closer to phase transition (ice → water) than genetic mutation.

    **Difference from Bifurcation**:

    - **ZHIR**: Changes phase/regime within single node (qualitative shift)
    - **Bifurcation**: Creates new sub-EPIs or structural variants (multiplication)
    - **When ZHIR triggers bifurcation**: High ∂²EPI/∂t² requires THOL for control

    Use Cases
    ---------
    **Biomedical**:

    - **Cellular Differentiation**: Stem cell → specialized cell (phase change)
    - **Metabolic Switching**: Glycolysis → oxidative phosphorylation
    - **Adaptive Immunity**: Naive T-cell → effector/memory cell
    - **Epigenetic Changes**: Stress-induced gene expression regime shifts
    - **Wound Healing Phases**: Inflammation → proliferation → remodeling

    **Cognitive**:

    - **Insight Moments**: Sudden perspective shift (aha! experience)
    - **Paradigm Transformation**: Fundamental worldview reorganization
    - **Strategy Changes**: Switching cognitive approach (analytical → intuitive)
    - **Memory Consolidation**: Working memory → long-term storage
    - **Belief Revision**: Core assumption restructuring under evidence

    **Social**:

    - **Regime Changes**: Political system transformation (democracy → authoritarianism)
    - **Cultural Revolutions**: Value system reorganization
    - **Organizational Transformation**: Hierarchy → network structure
    - **Disruptive Innovation**: Business model fundamental shift
    - **Social Movement Crystallization**: Protest → organized movement

    **AI/Computational**:

    - **Mode Switching**: Exploration → exploitation in RL
    - **Strategy Selection**: Changing between learned policies
    - **Attention Shifts**: Focus reorientation in transformers
    - **Learning Regime Change**: Supervised → self-supervised
    - **Attractor Transition**: Jumping between stable computational states

    Typical Sequences
    -----------------
    **Recommended Sequences**:

    - **IL → OZ → ZHIR → IL**: Controlled mutation cycle (stabilize-destabilize-mutate-stabilize)
    - **AL → IL → OZ → ZHIR → NAV**: Bootstrap with mutation and transition
    - **THOL → OZ → ZHIR**: Self-organization followed by transformation
    - **IL → VAL → ZHIR → IL**: Expansion-enabled mutation with consolidation
    - **OZ → ZHIR → THOL**: Mutation triggering bifurcation (requires THOL handler)
    - **EN → IL → OZ → ZHIR**: Reception-based mutation (integrate-stabilize-challenge-transform)

    **Sequences to Avoid**:

    - **ZHIR → OZ**: Mutation followed by dissonance = post-transformation instability
      (violates consolidation principle - transform then destabilize is dangerous)
    - **ZHIR → ZHIR**: Double mutation without IL = identity fragmentation risk
      (each mutation needs consolidation before next transformation)
    - **AL → ZHIR**: Emission directly to mutation = no stable base (violates U4b)
      (requires IL between emission and mutation for structural foundation)
    - **ZHIR without closure**: Mutation without SHA/IL/NAV = unconsolidated transformation
      (grammar U1b requires closure, especially critical after state changes)
    - **OZ → ZHIR → OZ**: Mutation sandwiched by dissonance = coherence collapse
      (transformation needs stability, not continued turbulence)

    Preconditions
    -------------
    - **Minimum νf**: Structural frequency > 0.05 (ZHIR_MIN_VF) for transformation capacity
    - **Threshold ξ**: Structural velocity ∂EPI/∂t > 0.1 (ZHIR_THRESHOLD_XI) for justification
    - **Prior IL**: Stable base required by grammar U4b (ZHIR_REQUIRE_IL_PRECEDENCE)
    - **Recent destabilizer**: OZ or VAL within ~3 operations (ZHIR_REQUIRE_DESTABILIZER)
    - **EPI history**: At least 2 points for velocity calculation (ZHIR_MIN_HISTORY_LENGTH)
    - **Network coupling**: Connected context for phase transformation

    Configuration Parameters
    ------------------------
    **Precondition Thresholds**:

    - ``ZHIR_MIN_VF``: Minimum structural frequency (default: 0.05)
      Node must have sufficient reorganization capacity
    - ``ZHIR_THRESHOLD_XI``: Mutation threshold ξ for ∂EPI/∂t (default: 0.1)
      Minimum velocity for justified phase transformation
    - ``ZHIR_MIN_HISTORY_LENGTH``: EPI history points needed (default: 2)
      Required for velocity calculation

    **Transformation Parameters**:

    - ``ZHIR_THETA_SHIFT_FACTOR``: Phase shift magnitude (default: 0.3)
      Controls intensity of phase transformation
    - ``ZHIR_MUTATION_INTENSITY``: Overall mutation intensity (default: 0.1)
      Scales transformation effects
    - ``ZHIR_THETA_SHIFT_DIRECTION``: "auto" (from ΔNFR sign) or "manual"
      Determines direction of phase shift

    **Bifurcation Detection**:

    - ``BIFURCATION_THRESHOLD_TAU``: Canonical bifurcation threshold τ (default: 0.5)
      When ∂²EPI/∂t² > τ, bifurcation potential detected
    - ``ZHIR_BIFURCATION_THRESHOLD``: Legacy threshold (fallback to canonical)
    - ``ZHIR_BIFURCATION_MODE``: "detection" only (no variant creation)

    **Grammar Validation**:

    - ``ZHIR_STRICT_U4B``: Enforce grammar U4b strictly (default: True)
      Requires both IL precedence and recent destabilizer
    - ``ZHIR_REQUIRE_IL_PRECEDENCE``: Require prior IL (default: True)
      Grammar U4b: stable base needed
    - ``ZHIR_REQUIRE_DESTABILIZER``: Require recent destabilizer (default: True)
      Grammar U4b: elevated ΔNFR needed for threshold crossing

    Structural Effects
    ------------------
    - **θ (phase)**: Primary effect - transforms to new regime (θ → θ')
    - **EPI**: May increment during transformation
    - **ΔNFR**: Typically elevated before ZHIR (from destabilizer)
    - **νf**: Preserved (transformation capacity maintained)
    - **epi_kind**: Preserved (identity maintained through transformation)
    - **Regime**: Changes if phase shift crosses regime boundary

    Metrics
    -------
    - ``theta_shift``: Magnitude and direction of phase transformation
    - ``regime_changed``: Boolean indicating regime boundary crossing
    - ``depi_dt``: Structural velocity at transformation
    - ``threshold_met``: Whether ∂EPI/∂t > ξ
    - ``threshold_ratio``: Velocity to threshold ratio
    - ``d2_epi``: Structural acceleration (bifurcation detection)
    - ``bifurcation_potential``: Flag for ∂²EPI/∂t² > τ

    Examples
    --------
    **Example 1: Controlled Mutation Cycle**

    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence, Dissonance, Mutation
    >>> from tnfr.metrics import compute_coherence
    >>>
    >>> # Create node and establish stable base
    >>> G, node = create_nfr("system", epi=0.5, vf=1.0, theta=0.2)
    >>> G.graph["COLLECT_OPERATOR_METRICS"] = True
    >>>
    >>> # Canonical mutation sequence: stabilize-destabilize-mutate-stabilize
    >>> run_sequence(G, node, [
    ...     Coherence(),   # IL: Establish stable base (required by U4b)
    ...     Dissonance(),  # OZ: Elevate ΔNFR (enables threshold crossing)
    ...     Mutation(),    # ZHIR: Transform phase when ∂EPI/∂t > ξ
    ...     Coherence(),   # IL: Consolidate new regime
    ... ])
    >>>
    >>> # Analyze transformation
    >>> metrics = G.graph["operator_metrics"][-2]  # ZHIR metrics
    >>> print(f"Phase transformed: {metrics.get('theta_shift', 0):.3f}")
    >>> print(f"Regime changed: {metrics.get('regime_changed', False)}")
    >>> print(f"Threshold met: {metrics.get('threshold_met', False)}")
    >>> print(f"Coherence maintained: {compute_coherence(G) > 0.6}")

    **Example 2: Bifurcation Detection**

    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence, Dissonance, Mutation, SelfOrganization
    >>>
    >>> # Create node with accelerating EPI
    >>> G, node = create_nfr("accelerating", epi=0.4, vf=1.2)
    >>> # Build acceleration history (high ∂²EPI/∂t²)
    >>> G.nodes[node]["epi_history"] = [0.1, 0.25, 0.4]
    >>> G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.3
    >>>
    >>> # Apply mutation with bifurcation detection
    >>> run_sequence(G, node, [Coherence(), Dissonance(), Mutation()])
    >>>
    >>> # Check bifurcation detection
    >>> if G.nodes[node].get("_zhir_bifurcation_potential"):
    ...     print("Bifurcation potential detected - applying THOL for control")
    ...     run_sequence(G, node, [SelfOrganization()])

    **Example 3: Stem Cell Differentiation (Biomedical)**

    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence, Dissonance, Mutation
    >>>
    >>> # Model stem cell differentiation into specialized cell type
    >>> G_cell, stem_cell = create_nfr("stem_cell", epi=0.6, vf=1.0, theta=0.0)
    >>> G_cell.nodes[stem_cell]["cell_type"] = "stem"
    >>> G_cell.nodes[stem_cell]["differentiation_signals"] = ["growth_factor_X"]
    >>>
    >>> # Differentiation sequence
    >>> run_sequence(G_cell, stem_cell, [
    ...     Coherence(),        # IL: Stable pluripotent state
    ...     Dissonance(),       # OZ: Differentiation signal received
    ...     Mutation(),         # ZHIR: Transform to specialized type
    ... ])
    >>>
    >>> # Cell has transformed phase (regime 0=stem → regime 1=specialized)
    >>> theta_new = G_cell.nodes[stem_cell]["theta"]
    >>> # Regime change indicates differentiation completed
    >>> # Cell maintains identity (is still a cell) but changed operational mode

    **Example 4: Paradigm Shift (Cognitive)**

    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Reception, Coherence, Dissonance, Mutation
    >>>
    >>> # Scientist encountering evidence that challenges paradigm
    >>> G_mind, scientist = create_nfr("paradigm", epi=0.7, vf=0.9, theta=0.5)
    >>> G_mind.nodes[scientist]["paradigm"] = "newtonian"
    >>>
    >>> # Paradigm shift sequence
    >>> run_sequence(G_mind, scientist, [
    ...     Reception(),        # EN: Receive anomalous evidence
    ...     Coherence(),        # IL: Try to integrate into existing framework
    ...     Dissonance(),       # OZ: Evidence creates cognitive dissonance
    ...     Mutation(),         # ZHIR: Paradigm shifts to quantum perspective
    ... ])
    >>>
    >>> # Scientist's conceptual framework has transformed
    >>> # Old paradigm (newtonian) → new paradigm (quantum)
    >>> # Identity preserved (still the same scientist) but worldview transformed

    **Example 5: Business Model Transformation (Social)**

    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence, Dissonance, Mutation, Transition
    >>>
    >>> # Company facing market disruption
    >>> G_org, company = create_nfr("business_model", epi=0.65, vf=0.85, theta=0.3)
    >>> G_org.nodes[company]["model"] = "traditional_retail"
    >>>
    >>> # Business transformation sequence
    >>> run_sequence(G_org, company, [
    ...     Coherence(),        # IL: Current model stable
    ...     Dissonance(),       # OZ: Market disruption (e-commerce threat)
    ...     Mutation(),         # ZHIR: Transform to digital-first model
    ...     Transition(),       # NAV: Navigate to new market position
    ... ])
    >>>
    >>> # Company has transformed operational model
    >>> # Identity preserved (same company) but strategy fundamentally changed

    Warnings
    --------
    - **Identity Loss Risk**: Multiple ZHIR in sequence without IL can cause identity
      fragmentation. Always consolidate transformations before next mutation.

    - **Requires Consolidation**: ZHIR MUST be followed by IL, NAV, or SHA to stabilize
      the new regime. Unconsolidated transformations are incoherent.

    - **Grammar U4b Strict**: ZHIR requires prior IL (stable base) AND recent destabilizer
      (OZ/VAL within ~3 ops). Violations risk unjustified or unstable transformations.

    - **Threshold Critical**: When ∂EPI/∂t < ξ, mutation lacks structural justification.
      Ensure sufficient ΔNFR elevation (via destabilizer) before ZHIR.

    - **Bifurcation Potential**: When ∂²EPI/∂t² > τ, bifurcation may occur. Must include
      THOL (handler) or IL (stabilizer) to prevent uncontrolled structural splitting.

    - **Phase Wrapping**: θ is periodic [0, 2π]. Large shifts may wrap around, potentially
      returning to similar regime. Monitor regime changes, not just phase magnitude.

    Contraindications
    -----------------
    - **Do not apply ZHIR without prior IL**: Violates U4b, risks unstable transformation
    - **Do not apply ZHIR with νf < 0.05**: Insufficient transformation capacity
    - **Do not apply ZHIR repeatedly**: Each mutation needs IL consolidation between
    - **Do not apply ZHIR to isolated nodes**: Network context required for regime support
    - **Do not apply ZHIR after NAV**: Transition already changed regime, redundant mutation
    - **Do not apply ZHIR with insufficient history**: Need ≥2 EPI points for velocity

    ZHIR vs THOL: Two Types of Transformation
    ------------------------------------------

    Both ZHIR and THOL are transformers (grammar U4b), but operate differently:

    +-------------------+-------------------------+---------------------------+
    | Aspect            | ZHIR (Mutation)         | THOL (Self-organization)  |
    +===================+=========================+===========================+
    | **Primary effect**| Phase transformation    | Sub-EPI creation          |
    |                   | (θ → θ')                | (fractal structuring)     |
    +-------------------+-------------------------+---------------------------+
    | **Trigger**       | ∂EPI/∂t > ξ             | ∂²EPI/∂t² > τ             |
    |                   | (velocity threshold)    | (acceleration threshold)  |
    +-------------------+-------------------------+---------------------------+
    | **Result**        | Regime change           | Emergent organization     |
    |                   | (qualitative shift)     | (internal complexity)     |
    +-------------------+-------------------------+---------------------------+
    | **Identity**      | Preserved (epi_kind)    | Preserved (global form)   |
    +-------------------+-------------------------+---------------------------+
    | **Structure**     | Single node transforms  | Creates nested sub-EPIs   |
    +-------------------+-------------------------+---------------------------+
    | **Grammar role**  | Transformer (U4b)       | Transformer (U4b) +       |
    |                   |                         | Handler (U4a)             |
    +-------------------+-------------------------+---------------------------+
    | **When to use**   | Qualitative state       | Internal reorganization   |
    |                   | change needed           | with emergence needed     |
    +-------------------+-------------------------+---------------------------+
    | **Example**       | Cell differentiation    | Embryonic development     |
    |                   | (phase change)          | (tissue formation)        |
    +-------------------+-------------------------+---------------------------+

    **Decision Guide**:

    - **Use ZHIR when**: Need phase transition without creating sub-structures
      (e.g., state machine transition, regime shift, perspective change)

    - **Use THOL when**: Need internal organization with sub-EPIs
      (e.g., hierarchical emergence, fractal structuring, metabolic capture)

    - **Use both (OZ → ZHIR → THOL)**: When mutation triggers bifurcation
      (∂²EPI/∂t² > τ after ZHIR), apply THOL to handle structural splitting

    Compatibility
    -------------
    **Compatible with**: IL (consolidation), OZ (enabling), NAV (transitioning),
    THOL (handling bifurcation), SHA (closure)

    **Avoid with**: Multiple consecutive ZHIR, direct AL → ZHIR, ZHIR → OZ sequences

    **Natural progressions**: ZHIR typically preceded by IL+OZ (preparation) and
    followed by IL/NAV (consolidation) or THOL (bifurcation handling)

    See Also
    --------
    Coherence : Stabilizes transformation base and consolidates post-mutation
    Dissonance : Elevates ΔNFR to enable threshold crossing for mutation
    SelfOrganization : Handles bifurcation when ZHIR triggers ∂²EPI/∂t² > τ
    Transition : Navigates between attractor states, complementary to mutation

    References
    ----------
    - **AGENTS.md §11 (Mutation)**: Canonical ZHIR definition and physics
    - **TNFR.pdf §2.2.11**: Theoretical foundation of mutation operator
    - **UNIFIED_GRAMMAR_RULES.md §U4b**: Transformer context requirements
    - **ZHIR_BIFURCATION_IMPLEMENTATION.md**: Bifurcation detection details
    """

    __slots__ = ()
    name: ClassVar[str] = MUTATION
    glyph: ClassVar[Glyph] = Glyph.ZHIR

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply ZHIR with bifurcation potential detection and postcondition verification.

        Detects when ∂²EPI/∂t² > τ (bifurcation threshold) and sets telemetry flags
        to enable validation of grammar U4a. Also verifies postconditions to ensure
        operator contract fulfillment.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes
        node : Any
            Target node identifier
        **kw : Any
            Additional parameters including:
            - tau: Bifurcation threshold (default from graph config or 0.5)
            - validate_preconditions: Enable precondition checks (default True)
            - validate_postconditions: Enable postcondition checks (default False)
            - collect_metrics: Enable metrics collection (default False)
        """
        # Capture state before mutation for postcondition verification
        validate_postconditions = kw.get("validate_postconditions", False) or G.graph.get(
            "VALIDATE_OPERATOR_POSTCONDITIONS", False
        )

        state_before = None
        if validate_postconditions:
            state_before = self._capture_state(G, node)
            # Also capture epi_kind if tracked
            state_before["epi_kind"] = G.nodes[node].get("epi_kind")

        # Compute structural acceleration before base operator
        d2_epi = self._compute_epi_acceleration(G, node)

        # Get bifurcation threshold (tau) from kwargs or graph config
        tau = kw.get("tau")
        if tau is None:
            # Try canonical threshold first, then operator-specific, then default
            tau = float(
                G.graph.get(
                    "BIFURCATION_THRESHOLD_TAU",
                    G.graph.get("ZHIR_BIFURCATION_THRESHOLD", 0.5),
                )
            )

        # Apply base operator (includes glyph application, preconditions, and metrics)
        super().__call__(G, node, **kw)

        # Detect bifurcation potential if acceleration exceeds threshold
        if d2_epi > tau:
            self._detect_bifurcation_potential(G, node, d2_epi=d2_epi, tau=tau)

        # Verify postconditions if enabled
        if validate_postconditions and state_before is not None:
            self._verify_postconditions(G, node, state_before)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Calculate ∂²EPI/∂t² from node's EPI history.

        Uses finite difference approximation:
        d²EPI/dt² ≈ (EPI_t - 2*EPI_{t-1} + EPI_{t-2}) / (Δt)²
        For unit time steps: d²EPI/dt² ≈ EPI_t - 2*EPI_{t-1} + EPI_{t-2}

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node identifier

        Returns
        -------
        float
            Magnitude of EPI acceleration (always non-negative)
        """

        # Get EPI history (maintained by node for temporal analysis)
        history = G.nodes[node].get("epi_history", [])

        # Need at least 3 points for second derivative
        if len(history) < 3:
            return 0.0

        # Finite difference: d²EPI/dt² ≈ (EPI_t - 2*EPI_{t-1} + EPI_{t-2})
        epi_t = float(history[-1])
        epi_t1 = float(history[-2])
        epi_t2 = float(history[-3])

        d2_epi = epi_t - 2.0 * epi_t1 + epi_t2

        return abs(d2_epi)

    def _detect_bifurcation_potential(
        self, G: TNFRGraph, node: Any, d2_epi: float, tau: float
    ) -> None:
        """Detect and record bifurcation potential when ∂²EPI/∂t² > τ.

        This implements Option B (conservative detection) from the issue specification.
        Sets telemetry flags and logs informative message without creating structural
        variants. Enables validation of grammar U4a requirement.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node identifier
        d2_epi : float
            Current EPI acceleration
        tau : float
            Bifurcation threshold that was exceeded
        """
        import logging

        logger = logging.getLogger(__name__)

        # Set telemetry flags for grammar validation
        G.nodes[node]["_zhir_bifurcation_potential"] = True
        G.nodes[node]["_zhir_d2epi"] = d2_epi
        G.nodes[node]["_zhir_tau"] = tau

        # Record bifurcation detection event in graph for analysis
        bifurcation_events = G.graph.setdefault("zhir_bifurcation_events", [])
        bifurcation_events.append(
            {
                "node": node,
                "d2_epi": d2_epi,
                "tau": tau,
                "timestamp": len(G.nodes[node].get("glyph_history", [])),
            }
        )

        # Log informative message
        logger.info(
            f"Node {node}: ZHIR bifurcation potential detected "
            f"(∂²EPI/∂t²={d2_epi:.3f} > τ={tau}). "
            f"Consider applying THOL for controlled bifurcation or IL for stabilization."
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate ZHIR-specific preconditions."""
        from .preconditions import validate_mutation

        validate_mutation(G, node)

    def _verify_postconditions(self, G: TNFRGraph, node: Any, state_before: dict[str, Any]) -> None:
        """Verify ZHIR-specific postconditions.

        Ensures that ZHIR fulfilled its contract:
        1. Phase was transformed (θ changed)
        2. Identity preserved (epi_kind maintained)
        3. Bifurcation handled (if detected)

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node that was mutated
        state_before : dict
            Node state before operator application, containing:
            - theta: Phase value before mutation
            - epi_kind: Identity before mutation (if tracked)
        """
        from .postconditions.mutation import (
            verify_phase_transformed,
            verify_identity_preserved,
            verify_bifurcation_handled,
        )

        # Verify phase transformation
        verify_phase_transformed(G, node, state_before["theta"])

        # Verify identity preservation (if tracked)
        epi_kind_before = state_before.get("epi_kind")
        if epi_kind_before is not None:
            verify_identity_preserved(G, node, epi_kind_before)

        # Verify bifurcation handling
        verify_bifurcation_handled(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect ZHIR-specific metrics."""
        from .metrics import mutation_metrics

        return mutation_metrics(
            G,
            node,
            state_before["theta"],
            state_before["epi"],
            vf_before=state_before.get("vf"),
            dnfr_before=state_before.get("dnfr"),
        )