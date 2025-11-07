"""Definitions for canonical TNFR structural operators.

Structural operators (Emission, Reception, Coherence, etc.) are the public-facing
API for applying TNFR transformations to nodes. Each operator is associated with
a specific glyph (structural symbol like AL, EN, IL, etc.) that represents the
underlying transformation.

English identifiers are the public API. Spanish wrappers were removed in
TNFR 2.0, so downstream code must import these classes directly.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from ..types import Glyph, TNFRGraph
from .registry import register_operator

__all__ = [
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
]

# T'HOL canonical bifurcation constants
_THOL_SUB_EPI_SCALING = 0.25  # Sub-EPI is 25% of parent (first-order bifurcation)
_THOL_EMERGENCE_CONTRIBUTION = 0.1  # Parent EPI increases by 10% of sub-EPI


class Operator:
    """Base class for TNFR structural operators.

    Structural operators (Emission, Reception, Coherence, etc.) are the public-facing
    API for applying TNFR transformations. Each operator defines a ``name`` (ASCII
    identifier) and ``glyph`` (structural symbol like AL, EN, IL, etc.) that represents
    the transformation. Calling an operator instance applies its structural transformation
    to the target node.
    """

    name: ClassVar[str] = "operator"
    glyph: ClassVar[Glyph | None] = None

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply the structural operator to ``node`` under canonical grammar control.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes, their coherence telemetry and structural
            operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
            Supported keys include:
            - ``window``: constrain the grammar window
            - ``validate_preconditions``: enable/disable precondition checks (default: True)
            - ``collect_metrics``: enable/disable metrics collection (default: False)

        Raises
        ------
        NotImplementedError
            If ``glyph`` is :data:`None`, meaning the operator has not been
            bound to a structural symbol.

        Notes
        -----
        The invocation delegates to
        :func:`tnfr.validation.apply_glyph_with_grammar`, which enforces
        the TNFR grammar before activating the structural transformation. The
        grammar may expand, contract or stabilise the neighbourhood so that the
        operator preserves canonical closure and coherence.
        """
        if self.glyph is None:
            raise NotImplementedError("Operator without assigned glyph")

        # Optional precondition validation
        validate_preconditions = kw.get("validate_preconditions", True)
        if validate_preconditions and G.graph.get(
            "VALIDATE_OPERATOR_PRECONDITIONS", False
        ):
            self._validate_preconditions(G, node)

        # Capture state before operator application for metrics and validation
        collect_metrics = kw.get("collect_metrics", False) or G.graph.get(
            "COLLECT_OPERATOR_METRICS", False
        )
        validate_equation = kw.get("validate_nodal_equation", False) or G.graph.get(
            "VALIDATE_NODAL_EQUATION", False
        )

        state_before = None
        if collect_metrics or validate_equation:
            state_before = self._capture_state(G, node)

        from . import apply_glyph_with_grammar

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))

        # Optional nodal equation validation (∂EPI/∂t = νf · ΔNFR(t))
        if validate_equation and state_before is not None:
            from .nodal_equation import validate_nodal_equation
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_EPI

            dt = float(kw.get("dt", 1.0))  # Time step, default 1.0 for discrete ops
            strict = G.graph.get("NODAL_EQUATION_STRICT", False)
            epi_after = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))

            validate_nodal_equation(
                G,
                node,
                epi_before=state_before["epi"],
                epi_after=epi_after,
                dt=dt,
                operator_name=self.name,
                strict=strict,
            )

        # Optional metrics collection (capture state after and compute)
        if collect_metrics and state_before is not None:
            metrics = self._collect_metrics(G, node, state_before)
            # Store metrics in graph for retrieval
            if "operator_metrics" not in G.graph:
                G.graph["operator_metrics"] = []
            G.graph["operator_metrics"].append(metrics)

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate operator-specific preconditions.

        Override in subclasses to implement specific validation logic.
        Base implementation does nothing.
        """
        pass

    def _get_node_attr(self, G: TNFRGraph, node: Any, attr_name: str) -> float:
        """Get node attribute value.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node identifier
        attr_name : str
            Attribute name ("epi", "vf", "dnfr", "theta")

        Returns
        -------
        float
            Attribute value
        """
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA

        alias_map = {
            "epi": ALIAS_EPI,
            "vf": ALIAS_VF,
            "dnfr": ALIAS_DNFR,
            "theta": ALIAS_THETA,
        }

        aliases = alias_map.get(attr_name, (attr_name,))
        return float(get_attr(G.nodes[node], aliases, 0.0))

    def _capture_state(self, G: TNFRGraph, node: Any) -> dict[str, Any]:
        """Capture node state before operator application.

        Returns dict with relevant state for metrics computation.
        """
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA

        return {
            "epi": float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)),
            "vf": float(get_attr(G.nodes[node], ALIAS_VF, 0.0)),
            "dnfr": float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)),
            "theta": float(get_attr(G.nodes[node], ALIAS_THETA, 0.0)),
        }

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect operator-specific metrics.

        Override in subclasses to implement specific metrics.
        Base implementation returns basic state change.
        """
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA

        # Safely access glyph value
        glyph_value = None
        if self.glyph is not None:
            glyph_value = (
                self.glyph.value if hasattr(self.glyph, "value") else str(self.glyph)
            )

        return {
            "operator": self.name,
            "glyph": glyph_value,
            "delta_epi": float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            - state_before["epi"],
            "delta_vf": float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
            - state_before["vf"],
            "delta_dnfr": float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
            - state_before["dnfr"],
            "delta_theta": float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
            - state_before["theta"],
        }


@register_operator
class Emission(Operator):
    """Emission structural operator (AL) - Foundational activation of nodal resonance.

    Activates structural symbol ``AL`` to initialise outward resonance around a
    nascent node, initiating the first phase of structural reorganization.

    TNFR Context
    ------------
    In the Resonant Fractal Nature paradigm, Emission (AL) represents the moment when
    a latent Primary Information Structure (EPI) begins to emit coherence toward its
    surrounding network. This is not passive information broadcast but active structural
    reorganization that increases the node's νf (structural frequency) and initiates
    positive ΔNFR flow.

    **Key Elements:**
    - **Coherent Emergence**: Node exists because it resonates; AL initiates resonance
    - **Structural Frequency**: Activates νf (Hz_str) to enable reorganization
    - **Network Coupling**: Prepares node for phase alignment
    - **Nodal Equation**: Implements ∂EPI/∂t = νf · ΔNFR(t) with positive ΔNFR

    **Structural Irreversibility (TNFR.pdf §2.2.1):**
    AL is inherently irreversible - once activated, it leaves a persistent structural
    trace that cannot be undone. Each emission marks "time zero" for the node and
    establishes genealogical traceability:

    - **emission_timestamp**: ISO 8601 UTC timestamp of first activation
    - **_emission_activated**: Immutable boolean flag
    - **_emission_origin**: Preserved original timestamp (never overwritten)
    - **_structural_lineage**: Genealogical record with:
      - ``origin``: First emission timestamp
      - ``activation_count``: Number of AL applications
      - ``derived_nodes``: List for tracking EPI emergence (future use)
      - ``parent_emission``: Reference to parent node (future use)

    Re-activation increments ``activation_count`` while preserving original timestamp.

    Use Cases
    ---------
    **Biomedical**: HRV coherence training, neural activation, therapeutic initiation
    **Cognitive**: Idea germination, learning initiation, creative spark
    **Social**: Team activation, community emergence, ritual initiation

    Typical Sequences
    -----------------
    **AL → EN → IL → SHA**: Basic activation with stabilization and silence
    **AL → RA**: Emission with immediate propagation
    **AL → NAV → IL**: Phased activation with transition

    Preconditions
    -------------
    - EPI < 0.8 (activation threshold)
    - Node in latent or low-activation state
    - Sufficient network coupling potential

    Structural Effects
    ------------------
    **EPI**: Increments (form activation)
    **νf**: Activates/increases (Hz_str)
    **ΔNFR**: Initializes positive reorganization
    **θ**: Influences phase alignment

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Emission, Reception, Coherence, Silence
    >>> G, node = create_nfr("seed", epi=0.18, vf=1.0)
    >>> run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    >>> # Verify irreversibility
    >>> assert G.nodes[node]["_emission_activated"] is True
    >>> assert "emission_timestamp" in G.nodes[node]
    >>> print(f"Activated at: {G.nodes[node]['emission_timestamp']}")  # doctest: +SKIP
    Activated at: 2025-11-07T15:47:10.209731+00:00

    See Also
    --------
    Coherence : Stabilizes emitted structures
    Resonance : Propagates emitted coherence
    Reception : Receives external emissions
    """

    __slots__ = ()
    name: ClassVar[str] = EMISSION
    glyph: ClassVar[Glyph] = Glyph.AL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply AL with structural irreversibility tracking.

        Marks temporal irreversibility before delegating to grammar execution.
        This ensures every emission leaves a persistent structural trace as
        required by TNFR.pdf §2.2.1 (AL - Emisión fundacional).

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
        """
        # Mark structural irreversibility BEFORE grammar execution
        self._mark_irreversibility(G, node)

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _mark_irreversibility(self, G: TNFRGraph, node: Any) -> None:
        """Mark structural irreversibility for AL operator.

        According to TNFR.pdf §2.2.1, AL (Emission) is structurally irreversible:
        "Una vez activado, AL reorganiza el campo. No puede deshacerse."

        This method establishes:
        - Temporal marker: ISO timestamp of first emission
        - Activation flag: Persistent boolean indicating AL was activated
        - Structural lineage: Genealogical record for EPI traceability

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node for emission marking.

        Notes
        -----
        On first activation:
        - Sets emission_timestamp (ISO format)
        - Sets _emission_activated = True (immutable)
        - Sets _emission_origin (timestamp copy for preservation)
        - Initializes _structural_lineage dict

        On re-activation:
        - Preserves original timestamp
        - Increments activation_count in lineage
        """
        from datetime import datetime, timezone

        from ..alias import set_attr_str
        from ..constants.aliases import ALIAS_EMISSION_TIMESTAMP

        # Check if this is first activation
        if "_emission_activated" not in G.nodes[node]:
            # Generate UTC timestamp in ISO format
            emission_timestamp = datetime.now(timezone.utc).isoformat()

            # Set canonical timestamp using alias system (use set_attr_str for string values)
            set_attr_str(G.nodes[node], ALIAS_EMISSION_TIMESTAMP, emission_timestamp)

            # Set persistent activation flag (immutable marker)
            G.nodes[node]["_emission_activated"] = True

            # Preserve origin timestamp (never overwritten)
            G.nodes[node]["_emission_origin"] = emission_timestamp

            # Initialize structural lineage for genealogical traceability
            G.nodes[node]["_structural_lineage"] = {
                "origin": emission_timestamp,
                "activation_count": 1,
                "derived_nodes": [],  # Nodes that emerge from this emission
                "parent_emission": None,  # If derived from another node
            }
        else:
            # Re-activation case: increment counter, preserve original timestamp
            if "_structural_lineage" in G.nodes[node]:
                G.nodes[node]["_structural_lineage"]["activation_count"] += 1

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate AL-specific preconditions with strict canonical checks.

        Implements TNFR.pdf §2.2.1 precondition validation:
        1. EPI < latent threshold (node in nascent/latent state)
        2. νf > basal threshold (sufficient structural frequency)
        3. Network connectivity check (warning for isolated nodes)

        Raises
        ------
        ValueError
            If EPI too high or νf too low for emission
        """
        from .preconditions.emission import validate_emission_strict

        validate_emission_strict(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect AL-specific metrics."""
        from .metrics import emission_metrics

        return emission_metrics(G, node, state_before["epi"], state_before["vf"])


@register_operator
class Reception(Operator):
    """Reception structural operator (EN) - Anchoring external coherence into local structure.

    Activates structural symbol ``EN`` to anchor external coherence into the node's EPI,
    stabilizing inbound information flows and integrating network resonance.

    TNFR Context
    ------------
    Reception (EN) represents the structural capacity to receive and integrate coherence
    from the network into the node's local EPI. Unlike passive data reception, EN is an
    active structural process that reorganizes the node to accommodate and stabilize
    external resonant patterns while reducing ΔNFR through integration.

    **Key Elements:**

    - **Active Integration**: Receiving is reorganizing, not passive storage
    - **ΔNFR Reduction**: Integration reduces reorganization pressure
    - **Network Coupling**: Requires phase compatibility with emitting nodes
    - **Coherence Preservation**: External patterns maintain their structural identity

    Use Cases
    ---------
    **Biomedical**:

    - **Biofeedback Reception**: Integrating external coherence signals (e.g., HRV monitoring)
    - **Therapeutic Resonance**: Patient receiving therapist's coherent presence
    - **Neural Synchronization**: Brain regions receiving and integrating signals

    **Cognitive**:

    - **Learning Reception**: Student integrating teacher's explanations
    - **Concept Integration**: Mind receiving and structuring new information
    - **Attention Anchoring**: Consciousness stabilizing around received stimuli

    **Social**:

    - **Communication Reception**: Team member integrating collaborative input
    - **Cultural Integration**: Individual receiving and adopting social patterns
    - **Empathic Reception**: Receiving and resonating with others' emotional states

    Typical Sequences
    ---------------------------
    - **AL → EN**: Emission followed by reception (bidirectional activation)
    - **EN → IL**: Reception followed by coherence (stabilized integration)
    - **RA → EN**: Resonance propagation followed by reception (network flow)
    - **EN → THOL**: Reception triggering self-organization (emergent integration)
    - **EN → UM**: Reception enabling coupling (synchronized reception)

    Preconditions
    -------------
    - Node must have receptive capacity (non-saturated EPI)
    - External coherence sources must be present in network
    - Phase compatibility with emitting nodes

    Structural Effects
    ------------------
    - **EPI**: Increments through integration of external patterns
    - **ΔNFR**: Typically reduces as external coherence stabilizes node
    - **θ**: May align toward emitting nodes' phase
    - **Network coupling**: Strengthens connections to coherence sources

    Metrics
    -----------------
    - ΔEPI: Magnitude of integrated external coherence
    - ΔNFR reduction: Measure of stabilization effectiveness
    - Integration efficiency: Ratio of received to integrated coherence
    - Phase alignment: Degree of synchronization with sources

    Compatibility
    ---------------------
    **Compatible with**: IL (Coherence), THOL (Self-organization), UM (Coupling),
    RA (Resonance), NAV (Transition)

    **Avoid with**: SHA (Silence) - contradicts receptive intent

    **Natural progressions**: EN typically followed by stabilization (IL) or
    organization (THOL) of received patterns

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Reception
    >>> G, node = create_nfr("receiver", epi=0.30)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.12
    >>> increments = iter([(0.05,)])
    >>> def stabilise(graph):
    ...     (d_epi,) = next(increments)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][DNFR_PRIMARY] *= 0.5
    >>> set_delta_nfr_hook(G, stabilise)
    >>> run_sequence(G, node, [Reception()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.35
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.06

    **Example (Biofeedback Integration):**

    >>> # Patient receiving HRV biofeedback during therapy
    >>> G_patient, patient = create_nfr("patient_biofeedback", epi=0.30, vf=1.0)
    >>> # EN: Patient's nervous system integrates coherence feedback
    >>> run_sequence(G_patient, patient, [Reception()])
    >>> # Result: External biofeedback signal anchors into patient's physiology
    >>> # ΔNFR reduces as system stabilizes around received pattern

    **Example (Educational Integration):**

    >>> # Student receiving and integrating new mathematical concept
    >>> G_learning, learner = create_nfr("student_mind", epi=0.25, vf=0.95)
    >>> # EN: Student's cognitive structure receives teacher's explanation
    >>> run_sequence(G_learning, learner, [Reception()])
    >>> # Result: New information integrates into existing knowledge structure
    >>> # Mental EPI reorganizes to accommodate new concept

    See Also
    --------
    Emission : Initiates patterns that EN can receive
    Coherence : Stabilizes received patterns
    SelfOrganization : Organizes received information
    """

    __slots__ = ()
    name: ClassVar[str] = RECEPTION
    glyph: ClassVar[Glyph] = Glyph.EN

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply EN with source detection and integration tracking.

        Detects emission sources in the network BEFORE applying reception
        grammar. This enables active reorganization from external sources
        as specified in TNFR.pdf §2.2.1 (EN - Recepción estructural).

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments:
            - track_sources (bool): Enable source detection (default: True).
              When enabled, automatically detects emission sources before
              grammar execution. This is a non-breaking enhancement - existing
              code continues to work, with source detection adding observability
              without changing operational semantics.
            - max_distance (int): Maximum network distance for source search (default: 2)
            - Other args forwarded to grammar layer

        Notes
        -----
        **Source Detection Behavior (New in This Release)**:

        By default, source detection is enabled (``track_sources=True``). This
        is a non-breaking change because:

        1. Detection happens BEFORE grammar execution (no operational changes)
        2. Only adds metadata to nodes (``_reception_sources``)
        3. Warnings are informational, not errors
        4. Can be disabled with ``track_sources=False``

        Existing code will see warnings if nodes have no emission sources,
        which is informational and helps identify network topology issues.
        To suppress warnings in isolated-node scenarios, set ``track_sources=False``.
        """
        # Detect emission sources BEFORE applying reception
        if kw.get("track_sources", True):
            from .network_analysis.source_detection import detect_emission_sources

            max_distance = kw.get("max_distance", 2)
            sources = detect_emission_sources(G, node, max_distance=max_distance)

            # Store detected sources in node metadata for metrics and analysis
            G.nodes[node]["_reception_sources"] = sources

            # Warn if no compatible sources found
            if not sources:
                warnings.warn(
                    f"EN warning: Node '{node}' has no detectable emission sources. "
                    f"Reception may not integrate external coherence effectively.",
                    stacklevel=2,
                )

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate EN-specific preconditions."""
        from .preconditions import validate_reception

        validate_reception(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect EN-specific metrics."""
        from .metrics import reception_metrics

        return reception_metrics(G, node, state_before["epi"])


@register_operator
class Coherence(Operator):
    """Coherence structural operator (IL) - Stabilization of structural alignment.

    Activates structural symbol ``IL`` to compress ΔNFR drift and raise the local C(t),
    reinforcing structural alignment across nodes and stabilizing emergent forms.

    TNFR Context
    ------------
    Coherence (IL) represents the fundamental stabilization process in TNFR. When applied,
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
    - **AL → IL**: Emission stabilized immediately (safe activation)
    - **EN → IL**: Reception consolidated (stable integration)
    - **IL → ZHIR**: Coherence enabling controlled mutation (stable transformation)
    - **RA → IL**: Resonance followed by stabilization (propagation consolidation)
    - **OZ → IL**: Dissonance resolved into new coherence (creative stabilization)
    - **AL → NAV → IL → OZ → THOL → RA → UM**: Full transformation cycle

    Preconditions
    -------------
    - Node must have active EPI (non-zero form)
    - ΔNFR should be present (though IL reduces it)
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
    **Compatible with**: ALL operators - IL is universally stabilizing

    **Especially effective after**: AL (Emission), EN (Reception), OZ (Dissonance),
    NAV (Transition)

    **Natural progressions**: IL often concludes sequences or prepares for
    controlled transformation (ZHIR, NAV)

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence
    >>> G, node = create_nfr("core", epi=0.50, vf=1.10)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.08
    >>> adjustments = iter([(0.03, 0.04, -0.03)])
    >>> def align(graph):
    ...     d_epi, d_vf, d_dnfr = next(adjustments)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] += d_dnfr
    >>> set_delta_nfr_hook(G, align)
    >>> run_sequence(G, node, [Coherence()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.53
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.14
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.05

    **Example (Cardiac Coherence Training):**

    >>> # Stabilizing heart rhythm after breath-focus activation
    >>> G_heart, heart = create_nfr("cardiac_rhythm", epi=0.50, vf=1.10)
    >>> # Heart activated (AL), now stabilizing coherent pattern
    >>> # IL: Breath rhythm locks into stable coherent pattern
    >>> run_sequence(G_heart, heart, [Coherence()])
    >>> # Result: HRV pattern stabilizes, ΔNFR reduces significantly
    >>> # Patient enters sustained coherent state

    **Example (Learning Consolidation):**

    >>> # Student consolidating newly understood concept
    >>> G_study, mind = create_nfr("student_understanding", epi=0.45, vf=1.05)
    >>> # Concept received (EN), now stabilizing into memory
    >>> # IL: Mental rehearsal consolidates understanding
    >>> run_sequence(G_study, mind, [Coherence()])
    >>> # Result: Knowledge structure stabilizes, confusion (ΔNFR) reduces
    >>> # Concept becomes part of stable mental model

    **Example (Team Alignment):**

    >>> # Collaborative team stabilizing after creative brainstorm
    >>> G_team, group = create_nfr("team_consensus", epi=0.55, vf=1.00)
    >>> # Ideas generated (OZ), now building consensus
    >>> # IL: Team aligns around shared vision
    >>> run_sequence(G_team, group, [Coherence()])
    >>> # Result: Group coherence increases, conflicts (ΔNFR) resolve
    >>> # Team operates with unified purpose

    See Also
    --------
    Dissonance : Creates instability that IL later resolves
    Emission : Often followed by IL for safe activation
    Mutation : IL enables controlled phase changes
    """

    __slots__ = ()
    name: ClassVar[str] = COHERENCE
    glyph: ClassVar[Glyph] = Glyph.IL

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


@register_operator
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


@register_operator
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
    - Both nodes must be active (non-zero EPI)
    - Phase compatibility: |θᵢ - θⱼ| must be within coupling threshold
    - Sufficient network proximity or connectivity

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

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect UM-specific metrics."""
        from .metrics import coupling_metrics

        return coupling_metrics(G, node, state_before["theta"])


@register_operator
class Resonance(Operator):
    """Resonance structural operator (RA) - Network coherence propagation.

    Activates glyph ``RA`` to circulate phase-aligned energy through the network,
    amplifying shared frequency and propagating coherent resonance between nodes.

    TNFR Context
    ------------
    Resonance (RA) is the propagation mechanism in TNFR networks. When nodes are coupled
    and phase-aligned, RA transmits coherence (EPIₙ → EPIₙ₊₁) without loss of structural
    identity. This creates "resonant cascades" where coherence amplifies across the
    network, increasing collective νf and global C(t). RA embodies the fundamental TNFR
    principle: structural patterns propagate through resonance, not mechanical transfer.

    **Key Elements:**

    - **Identity Preservation**: Propagated EPI maintains structural integrity
    - **Amplification**: Coherence strengthens through resonant networks
    - **Phase Alignment**: Requires synchronized nodes (UM prerequisite)
    - **Network Emergence**: Creates collective coherence beyond individual nodes

    Use Cases
    ---------
    **Biomedical**:

    - **Coherence Propagation**: HRV coherence spreading through nervous system
    - **Neural Cascade**: Synchronized firing spreading across brain networks
    - **Immune Response**: Coordinated cellular activation across tissue
    - **Healing Waves**: Therapeutic effects propagating through body systems

    **Cognitive**:

    - **Insight Spread**: Understanding cascading through conceptual network
    - **Aha Moment**: Sudden coherence propagating across mental model
    - **Cultural Memes**: Ideas resonating and spreading through population
    - **Knowledge Transfer**: Expertise propagating through learning networks

    **Social**:

    - **Collective Enthusiasm**: Energy spreading through group
    - **Social Movements**: Coherent values propagating through networks
    - **Innovation Diffusion**: Ideas spreading through organizations
    - **Market Resonance**: Trends amplifying through economic networks

    Typical Sequences
    ---------------------------
    - **UM → RA**: Coupling followed by propagation (network activation)
    - **AL → RA**: Emission followed by propagation (broadcast pattern)
    - **RA → IL**: Resonance stabilized (network coherence lock)
    - **IL → RA**: Stable form propagated (controlled spread)
    - **RA → EN**: Propagation received (network reception)

    Preconditions
    -------------
    - Source node must have coherent EPI
    - Network connectivity must exist (edges)
    - Phase compatibility between nodes (coupling)
    - Sufficient νf to support propagation

    Structural Effects
    ------------------
    - **Network EPI**: Propagates to connected nodes
    - **Collective νf**: Amplifies across network
    - **Global C(t)**: Increases through network coherence
    - **ΔNFR**: May slightly increase initially, then stabilize
    - **Phase alignment**: Strengthens across propagation path

    Metrics
    -----------------
    - Propagation distance: Number of nodes reached
    - Amplification factor: Coherence gain through network
    - Network C(t): Global coherence increase
    - Propagation speed: Rate of coherence spread

    Compatibility
    ---------------------
    **Compatible with**: UM (Coupling), IL (Coherence), EN (Reception),
    AL (Emission), THOL (Self-organization)

    **Requires**: Network connectivity and phase alignment (UM)

    **Natural progressions**: RA often followed by IL (stabilize propagated
    coherence) or EN (nodes receive propagation)

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Resonance
    >>> G, node = create_nfr("carrier", vf=0.90)
    >>> pulses = iter([(0.05, 0.03)])
    >>> def amplify(graph):
    ...     d_vf, d_dnfr = next(pulses)
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] = d_dnfr
    >>> set_delta_nfr_hook(G, amplify)
    >>> run_sequence(G, node, [Resonance()])
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.95
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.03

    **Example (Cardiac Coherence Spread):**

    >>> # Heart coherence propagating to entire nervous system
    >>> G_body, heart = create_nfr("cardiac_source", vf=0.90, epi=0.60)
    >>> # Heart achieves coherent state (IL), now propagating
    >>> # RA: Coherent rhythm spreads through vagal nerve network
    >>> run_sequence(G_body, heart, [Resonance()])
    >>> # Result: Coherence propagates to brain, organs, peripheral systems
    >>> # Whole body enters resonant coherent state
    >>> # Enables healing, relaxation, optimal function

    **Example (Insight Cascade):**

    >>> # Understanding suddenly spreading through mental model
    >>> G_mind, insight = create_nfr("conceptual_breakthrough", vf=1.05, epi=0.55)
    >>> # Key insight achieved (THOL), now propagating
    >>> # RA: Understanding cascades through related concepts
    >>> run_sequence(G_mind, insight, [Resonance()])
    >>> # Result: Single insight illuminates entire knowledge domain
    >>> # "Aha!" moment as coherence spreads through mental network
    >>> # Previously disconnected ideas suddenly align

    **Example (Social Movement):**

    >>> # Idea resonating through social network
    >>> G_social, movement = create_nfr("cultural_idea", vf=0.95, epi=0.50)
    >>> # Coherent message formed (IL), now spreading
    >>> # RA: Idea propagates through connected communities
    >>> run_sequence(G_social, movement, [Resonance()])
    >>> # Result: Message amplifies across network
    >>> # More nodes adopt and propagate the pattern
    >>> # Creates collective coherence and momentum

    See Also
    --------
    Coupling : Creates conditions for RA propagation
    Coherence : Stabilizes resonant patterns
    Emission : Initiates patterns for RA to propagate
    """

    __slots__ = ()
    name: ClassVar[str] = RESONANCE
    glyph: ClassVar[Glyph] = Glyph.RA

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate RA-specific preconditions."""
        from .preconditions import validate_resonance

        validate_resonance(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect RA-specific metrics."""
        from .metrics import resonance_metrics

        return resonance_metrics(G, node, state_before["epi"])


@register_operator
class Silence(Operator):
    """Silence structural operator (SHA) - Preservation through structural pause.

    Activates glyph ``SHA`` to lower νf and hold the local EPI invariant, suspending
    reorganization to preserve the node's current coherence state.

    TNFR Context
    ------------
    Silence (SHA) creates structural latency - a state where νf ≈ 0, causing the nodal
    equation ∂EPI/∂t = νf · ΔNFR(t) to approach zero regardless of ΔNFR. This preserves
    the current EPI form intact, preventing reorganization. SHA is essential for memory,
    consolidation, and maintaining structural identity during network turbulence.

    **Key Elements:**

    - **Frequency Suppression**: Reduces νf to near-zero (structural pause)
    - **Form Preservation**: EPI remains unchanged despite external pressures
    - **Latent Memory**: Stored patterns awaiting reactivation
    - **Strategic Inaction**: Deliberate non-reorganization as protective mechanism

    Use Cases
    ---------
    **Biomedical**:

    - **Rest and Recovery**: Physiological downregulation for healing
    - **Sleep Consolidation**: Memory formation through structural pause
    - **Meditation States**: Conscious reduction of mental reorganization
    - **Trauma Containment**: Protective numbing of overwhelming activation

    **Cognitive**:

    - **Memory Storage**: Consolidating learning through reduced interference
    - **Incubation Period**: Letting problems "rest" before insight
    - **Attention Rest**: Recovery from cognitive load
    - **Knowledge Preservation**: Maintaining expertise without active use

    **Social**:

    - **Strategic Pause**: Deliberate non-action in conflict
    - **Cultural Preservation**: Maintaining traditions without active practice
    - **Organizational Stability**: Resisting change pressure
    - **Waiting Strategy**: Preserving position until conditions favor action

    Typical Sequences
    ---------------------------
    - **IL → SHA**: Stabilize then preserve (long-term memory)
    - **SHA → AL**: Silence broken by reactivation (awakening)
    - **SHA → NAV**: Preserved structure transitions (controlled change)
    - **OZ → SHA**: Dissonance contained (protective pause)

    **AVOID**: SHA → OZ (silence followed by dissonance contradicts preservation)
    **AVOID**: SHA → SHA (redundant, no structural purpose)

    Preconditions
    -------------
    - Node must have existing EPI to preserve
    - Network pressure (ΔNFR) should not be critically high
    - Context must support reduced activity

    Structural Effects
    ------------------
    - **νf**: Significantly reduced (≈ 0, primary effect)
    - **EPI**: Held invariant (preservation)
    - **ΔNFR**: Neither increases nor decreases (frozen state)
    - **θ**: Maintained but not actively synchronized
    - **Network influence**: Minimal during silence

    Metrics
    -----------------
    - νf reduction: Degree of frequency suppression
    - EPI stability: Variance over silence period (should be ~0)
    - Silence duration: Time in latent state
    - Preservation effectiveness: EPI integrity post-silence

    Compatibility
    ---------------------
    **Compatible with**: IL (Coherence before silence), NAV (Transition from silence),
    AL (Reactivation from silence)

    **Avoid with**: OZ (Dissonance), RA (Resonance), multiple consecutive operators

    **Natural progressions**: SHA typically ends sequences or precedes reactivation
    (AL) or transition (NAV)

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Silence
    >>> G, node = create_nfr("rest", epi=0.51, vf=1.00)
    >>> def freeze(graph):
    ...     graph.nodes[node][DNFR_PRIMARY] = 0.0
    ...     graph.nodes[node][VF_PRIMARY] = 0.02
    ...     # EPI is intentionally left untouched to preserve the stored form.
    >>> set_delta_nfr_hook(G, freeze)
    >>> run_sequence(G, node, [Silence()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.51
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.02

    **Example (Sleep Consolidation):**

    >>> # Memory consolidation during sleep
    >>> G_memory, memory_trace = create_nfr("learned_pattern", epi=0.51, vf=1.00)
    >>> # Pattern learned during day (IL stabilized)
    >>> # SHA: Deep sleep reduces neural activity, preserves memory
    >>> run_sequence(G_memory, memory_trace, [Silence()])
    >>> # Result: EPI preserved intact (0.51 unchanged)
    >>> # νf drops to near-zero, prevents interference
    >>> # Memory consolidates through structural silence

    **Example (Meditative Rest):**

    >>> # Consciousness entering deep meditation
    >>> G_mind, awareness = create_nfr("mental_state", epi=0.48, vf=0.95)
    >>> # Active mind state before meditation
    >>> # SHA: Meditation reduces mental activity, preserves presence
    >>> run_sequence(G_mind, awareness, [Silence()])
    >>> # Result: Mental chatter ceases (νf → 0)
    >>> # Awareness EPI maintained without elaboration
    >>> # Restful alertness through structural silence

    **Example (Organizational Pause):**

    >>> # Company maintaining position during market uncertainty
    >>> G_company, strategy = create_nfr("business_position", epi=0.55, vf=1.10)
    >>> # Established strategy under pressure to change
    >>> # SHA: Leadership decides to "wait and see"
    >>> run_sequence(G_company, strategy, [Silence()])
    >>> # Result: Strategy preserved without modification
    >>> # Organization resists external pressure for change
    >>> # Maintains identity until conditions clarify

    See Also
    --------
    Coherence : Often precedes SHA for stable preservation
    Transition : Breaks silence with controlled change
    Emission : Reactivates silenced structures
    """

    __slots__ = ()
    name: ClassVar[str] = SILENCE
    glyph: ClassVar[Glyph] = Glyph.SHA

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate SHA-specific preconditions."""
        from .preconditions import validate_silence

        validate_silence(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect SHA-specific metrics."""
        from .metrics import silence_metrics

        return silence_metrics(G, node, state_before["vf"], state_before["epi"])


@register_operator
class Expansion(Operator):
    """Expansion structural operator (VAL) - Structural dilation for exploration.

    Activates glyph ``VAL`` to dilate the node's structure, unfolding neighbouring
    trajectories and extending operational boundaries to explore additional coherence volume.

    TNFR Context: Expansion increases EPI magnitude and νf, enabling exploration of new
    structural configurations while maintaining core identity. VAL embodies fractality -
    structures scale while preserving their essential form.

    Use Cases: Growth processes (biological, cognitive, organizational), exploration phases,
    capacity building, network extension.

    Typical Sequences: VAL → IL (expand then stabilize), OZ → VAL (dissonance enables
    expansion), VAL → THOL (expansion triggers reorganization).

    Avoid: VAL → NUL (contradictory), multiple consecutive VAL without consolidation.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Expansion
    >>> G, node = create_nfr("theta", epi=0.47, vf=0.95)
    >>> spreads = iter([(0.06, 0.08)])
    >>> def open_volume(graph):
    ...     d_epi, d_vf = next(spreads)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    >>> set_delta_nfr_hook(G, open_volume)
    >>> run_sequence(G, node, [Expansion()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.53
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.03

    **Biomedical**: Growth, tissue expansion, neural network development
    **Cognitive**: Knowledge domain expansion, conceptual broadening
    **Social**: Team scaling, market expansion, network growth
    """

    __slots__ = ()
    name: ClassVar[str] = EXPANSION
    glyph: ClassVar[Glyph] = Glyph.VAL

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate VAL-specific preconditions."""
        from .preconditions import validate_expansion

        validate_expansion(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect VAL-specific metrics."""
        from .metrics import expansion_metrics

        return expansion_metrics(G, node, state_before["vf"], state_before["epi"])


@register_operator
class Contraction(Operator):
    """Contraction structural operator (NUL) - Structural concentration and densification.

    Activates glyph ``NUL`` to concentrate the node's structure, pulling peripheral
    trajectories back into the core EPI to tighten coherence gradients.

    TNFR Context: Contraction reduces EPI surface while maintaining or increasing density,
    focusing structural energy into core patterns. NUL enables consolidation, refinement,
    and essential simplification.

    Use Cases: Consolidation phases, focus intensification, resource optimization,
    simplification processes, core strengthening.

    Typical Sequences: NUL → IL (contract then stabilize), VAL → NUL → IL (expand-contract-
    stabilize cycle), THOL → NUL (organize then consolidate).

    Avoid: NUL → VAL (contradictory), excessive NUL (over-compression).

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Contraction
    >>> G, node = create_nfr("iota", epi=0.39, vf=1.05)
    >>> squeezes = iter([(-0.05, -0.03, 0.05)])
    >>> def tighten(graph):
    ...     d_epi, d_vf, stored_dnfr = next(squeezes)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] = stored_dnfr
    >>> set_delta_nfr_hook(G, tighten)
    >>> run_sequence(G, node, [Contraction()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.34
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.02
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.05

    **Biomedical**: Wound healing, tissue consolidation, neural pruning
    **Cognitive**: Focus intensification, concept refinement, "less is more"
    **Social**: Team downsizing, resource consolidation, core focus
    """

    __slots__ = ()
    name: ClassVar[str] = CONTRACTION
    glyph: ClassVar[Glyph] = Glyph.NUL

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate NUL-specific preconditions."""
        from .preconditions import validate_contraction

        validate_contraction(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect NUL-specific metrics."""
        from .metrics import contraction_metrics

        return contraction_metrics(G, node, state_before["vf"], state_before["epi"])


@register_operator
class SelfOrganization(Operator):
    """Self-Organization structural operator (THOL) - Autonomous emergent reorganization.

    Activates glyph ``THOL`` to spawn nested EPIs and trigger self-organizing cascades
    within the local structure, enabling autonomous coherent reorganization.

    TNFR Context: Self-organization (THOL) embodies emergence - when ∂²EPI/∂t² > τ, the
    system bifurcates and generates new sub-EPIs that organize coherently without external
    direction. THOL is the engine of complexity and novelty in TNFR. This is not just
    autoorganization but **structural metabolism**: T'HOL reorganizes experience into
    structure without external instruction.

    **Canonical Characteristics:**

    - **Bifurcation nodal**: When ∂²EPI/∂t² > τ, spawns new sub-EPIs
    - **Autonomous reorganization**: No external control, self-directed
    - **Vibrational metabolism**: Digests external experience into internal structure
    - **Complexity emergence**: Engine of novelty and evolution in TNFR

    Use Cases: Emergence processes, bifurcation events, creative reorganization, complex
    system evolution, spontaneous order generation.

    Typical Sequences: OZ → THOL (dissonance catalyzes emergence), THOL → RA (emergent
    forms propagate), THOL → IL (organize then stabilize), EN → THOL (reception triggers
    reorganization).

    Critical: THOL requires sufficient ΔNFR and network connectivity for bifurcation.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import SelfOrganization
    >>> G, node = create_nfr("kappa", epi=0.66, vf=1.10)
    >>> cascades = iter([(0.04, 0.05)])
    >>> def spawn(graph):
    ...     d_epi, d_vf = next(cascades)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.graph.setdefault("sub_epi", []).append(round(graph.nodes[node][EPI_PRIMARY], 2))
    >>> set_delta_nfr_hook(G, spawn)
    >>> run_sequence(G, node, [SelfOrganization()])
    >>> G.graph["sub_epi"]
    [0.7]

    **Biomedical**: Embryogenesis, immune response, neural plasticity, wound healing
    **Cognitive**: Insight generation, creative breakthroughs, paradigm shifts
    **Social**: Innovation emergence, cultural evolution, spontaneous movements
    """

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION
    glyph: ClassVar[Glyph] = Glyph.THOL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply T'HOL with bifurcation logic.

        If ∂²EPI/∂t² > τ, generates sub-EPIs through bifurcation.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes
        node : Any
            Target node identifier
        **kw : Any
            Additional parameters including:
            - tau: Bifurcation threshold (default from graph config or 0.1)
            - validate_preconditions: Enable precondition checks (default True)
            - collect_metrics: Enable metrics collection (default False)
        """
        # Compute structural acceleration before base operator
        d2_epi = self._compute_epi_acceleration(G, node)

        # Get bifurcation threshold (tau) from kwargs or graph config
        tau = kw.get("tau")
        if tau is None:
            tau = float(G.graph.get("THOL_BIFURCATION_THRESHOLD", 0.1))

        # Apply base operator (includes glyph application and metrics)
        super().__call__(G, node, **kw)

        # Bifurcate if acceleration exceeds threshold
        if d2_epi > tau:
            self._spawn_sub_epi(G, node, d2_epi=d2_epi, tau=tau)

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
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI

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

    def _spawn_sub_epi(
        self, G: TNFRGraph, node: Any, d2_epi: float, tau: float
    ) -> None:
        """Generate sub-EPI through bifurcation.

        When acceleration exceeds threshold, creates nested sub-structure
        that inherits properties from parent while maintaining operational
        fractality.

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
        from ..alias import get_attr, set_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF

        # Get current node state
        parent_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        parent_vf = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))

        # Calculate sub-EPI magnitude using canonical scaling factor
        sub_epi_value = parent_epi * _THOL_SUB_EPI_SCALING

        # Store sub-EPI in node's sub_epis list
        sub_epis = G.nodes[node].get("sub_epis", [])

        # Get current timestamp from glyph history length
        timestamp = len(G.nodes[node].get("glyph_history", []))

        sub_epis.append(
            {
                "epi": sub_epi_value,
                "vf": parent_vf,
                "timestamp": timestamp,
                "d2_epi": d2_epi,
                "tau": tau,
            }
        )
        G.nodes[node]["sub_epis"] = sub_epis

        # Increment parent EPI using canonical emergence contribution
        # This reflects that bifurcation increases total structural complexity
        new_epi = parent_epi + sub_epi_value * _THOL_EMERGENCE_CONTRIBUTION
        set_attr(G.nodes[node], ALIAS_EPI, new_epi)

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate THOL-specific preconditions."""
        from .preconditions import validate_self_organization

        validate_self_organization(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect THOL-specific metrics."""
        from .metrics import self_organization_metrics

        return self_organization_metrics(
            G, node, state_before["epi"], state_before["vf"]
        )


@register_operator
class Mutation(Operator):
    """Mutation structural operator (ZHIR) - Controlled phase transformation.

    Activates glyph ``ZHIR`` to recode phase or form, enabling the node to cross
    structural thresholds and pivot towards a new coherence regime.

    TNFR Context: Mutation (ZHIR) implements phase change θ → θ' when ΔEPI/Δt > ξ. Unlike
    random variation, ZHIR is controlled transformation that preserves structural identity
    while shifting operational regime. Critical for adaptation and evolution.

    Use Cases: Paradigm shifts, strategic pivots, adaptive responses, regime transitions,
    identity transformation while maintaining continuity.

    Typical Sequences: IL → ZHIR → IL (stabilize-mutate-stabilize), OZ → ZHIR (dissonance
    enables mutation), ZHIR → NAV → IL (mutate-transition-stabilize).

    Preconditions: Requires stable base (often IL), sufficient ΔNFR for phase change,
    network support for new phase.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, THETA_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Mutation
    >>> G, node = create_nfr("lambda", epi=0.73, theta=0.20)
    >>> shifts = iter([(0.03, 0.40)])
    >>> def mutate(graph):
    ...     d_epi, d_theta = next(shifts)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    >>> set_delta_nfr_hook(G, mutate)
    >>> run_sequence(G, node, [Mutation()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.76
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.6

    **Biomedical**: Cellular differentiation, adaptive immunity, metabolic switching
    **Cognitive**: Paradigm shift, perspective transformation, belief change
    **Social**: Strategic pivot, cultural adaptation, business model transformation
    """

    __slots__ = ()
    name: ClassVar[str] = MUTATION
    glyph: ClassVar[Glyph] = Glyph.ZHIR

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate ZHIR-specific preconditions."""
        from .preconditions import validate_mutation

        validate_mutation(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect ZHIR-specific metrics."""
        from .metrics import mutation_metrics

        return mutation_metrics(G, node, state_before["theta"], state_before["epi"])


@register_operator
class Transition(Operator):
    """Transition structural operator (NAV) - Controlled regime handoff.

    Activates glyph ``NAV`` to guide the node through a controlled transition between
    structural regimes, managing hand-offs across states.

    TNFR Context: Transition (NAV) manages movement between coherence regimes with minimal
    disruption. NAV adjusts θ, νf, and ΔNFR to navigate thresholds smoothly, preventing
    collapse during regime shifts. Essential for change management.

    Use Cases: State transitions, regime changes, threshold crossings, transformation
    processes, managed evolution.

    Typical Sequences: AL → NAV → IL (activate-transition-stabilize), NAV → ZHIR (transition
    enables mutation), SHA → NAV → AL (silence-transition-reactivation), IL → NAV → OZ
    (stable-transition-explore).

    Versatility: NAV is highly compatible with most operators as transition manager.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Transition
    >>> G, node = create_nfr("mu", vf=0.85, theta=0.40)
    >>> ramps = iter([(0.12, -0.25)])
    >>> def handoff(graph):
    ...     d_vf, d_theta = next(ramps)
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    ...     graph.nodes[node][DNFR_PRIMARY] = abs(d_vf) * 0.5
    >>> set_delta_nfr_hook(G, handoff)
    >>> run_sequence(G, node, [Transition()])
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.97
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.15
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.06

    **Biomedical**: Sleep stage transitions, developmental phases, recovery processes
    **Cognitive**: Learning phase transitions, attention shifts, mode switching
    **Social**: Organizational change, cultural transitions, leadership handoffs
    """

    __slots__ = ()
    name: ClassVar[str] = TRANSITION
    glyph: ClassVar[Glyph] = Glyph.NAV

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate NAV-specific preconditions."""
        from .preconditions import validate_transition

        validate_transition(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect NAV-specific metrics."""
        from .metrics import transition_metrics

        return transition_metrics(
            G, node, state_before["dnfr"], state_before["vf"], state_before["theta"]
        )


@register_operator
class Recursivity(Operator):
    """Recursivity structural operator (REMESH) - Fractal pattern propagation.

    Activates glyph ``REMESH`` to propagate fractal recursivity and echo structural
    patterns across nested EPIs, maintaining multi-scale identity.

    TNFR Context: Recursivity (REMESH) implements operational fractality - patterns that
    replicate across scales while preserving structural identity. REMESH ensures that
    EPI(t) echoes EPI(t - τ) at nested levels, creating self-similar coherence structures.

    Use Cases: Fractal processes, multi-scale coherence, memory recursion, pattern
    replication, self-similar organization, adaptive memory systems.

    Typical Sequences: REMESH → RA (recursive propagation), THOL → REMESH (emergence
    with fractal structure), REMESH → IL (recursive pattern stabilization), VAL → REMESH
    (expansion with self-similarity).

    Critical: REMESH preserves identity across scales - fundamental to TNFR fractality.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Recursivity
    >>> G, node = create_nfr("nu", epi=0.52, vf=0.92)
    >>> echoes = iter([(0.02, 0.03)])
    >>> def echo(graph):
    ...     d_epi, d_vf = next(echoes)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.graph.setdefault("echo_trace", []).append(
    ...         (round(graph.nodes[node][EPI_PRIMARY], 2), round(graph.nodes[node][VF_PRIMARY], 2))
    ...     )
    >>> set_delta_nfr_hook(G, echo)
    >>> run_sequence(G, node, [Recursivity()])
    >>> G.graph["echo_trace"]
    [(0.54, 0.95)]

    **Biomedical**: Fractal physiology (HRV, EEG), developmental recapitulation
    **Cognitive**: Recursive thinking, meta-cognition, self-referential processes
    **Social**: Cultural fractals, organizational self-similarity, meme propagation
    """

    __slots__ = ()
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate REMESH-specific preconditions."""
        from .preconditions import validate_recursivity

        validate_recursivity(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect REMESH-specific metrics."""
        from .metrics import recursivity_metrics

        return recursivity_metrics(G, node, state_before["epi"], state_before["vf"])
