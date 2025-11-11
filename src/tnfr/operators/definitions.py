"""Definitions for canonical TNFR structural operators.

Structural operators (Emission, Reception, Coherence, etc.) are the public-facing
API for applying TNFR transformations to nodes. Each operator is associated with
a specific glyph (structural symbol like AL, EN, IL, etc.) that represents the
underlying transformation.

English identifiers are the public API. Spanish wrappers were removed in
TNFR 2.0, so downstream code must import these classes directly.
"""

from __future__ import annotations

import cmath
import math
import warnings
from typing import Any, ClassVar

from ..alias import get_attr
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
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from ..utils import get_numpy
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
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_EPI
            from .nodal_equation import validate_nodal_equation

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
        from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF

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
        from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF

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
        from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF

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
        required by TNFR.pdf §2.2.1 (AL - Foundational emission).

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
        """
        # Check and clear latency state if reactivating from silence
        self._check_reactivation(G, node)

        # Mark structural irreversibility BEFORE grammar execution
        self._mark_irreversibility(G, node)

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _check_reactivation(self, G: TNFRGraph, node: Any) -> None:
        """Check and clear latency state when reactivating from silence.

        When AL (Emission) is applied to a node in latent state (from SHA),
        this validates the reactivation and clears the latency attributes.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node being reactivated.

        Warnings
        --------
        - Warns if node is being reactivated after extended silence (duration check)
        - Warns if EPI has drifted from preserved value during silence
        """
        if G.nodes[node].get("latent", False):
            # Node is in latent state, reactivating from silence
            silence_duration = G.nodes[node].get("silence_duration", 0.0)

            # Get max silence duration threshold from graph config
            max_silence = G.graph.get("MAX_SILENCE_DURATION", float("inf"))

            # Validate reactivation timing
            if silence_duration > max_silence:
                warnings.warn(
                    f"Node {node} reactivating after extended silence "
                    f"(duration: {silence_duration:.2f}, max: {max_silence:.2f})",
                    stacklevel=3,
                )

            # Check EPI preservation integrity
            preserved_epi = G.nodes[node].get("preserved_epi")
            if preserved_epi is not None:
                from ..alias import get_attr

                current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
                epi_drift = abs(current_epi - preserved_epi)

                # Allow small numerical drift (1% tolerance)
                if epi_drift > 0.01 * abs(preserved_epi):
                    warnings.warn(
                        f"Node {node} EPI drifted during silence "
                        f"(preserved: {preserved_epi:.3f}, current: {current_epi:.3f}, "
                        f"drift: {epi_drift:.3f})",
                        stacklevel=3,
                    )

            # Clear latency state
            del G.nodes[node]["latent"]
            if "latency_start_time" in G.nodes[node]:
                del G.nodes[node]["latency_start_time"]
            if "preserved_epi" in G.nodes[node]:
                del G.nodes[node]["preserved_epi"]
            if "silence_duration" in G.nodes[node]:
                del G.nodes[node]["silence_duration"]

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
        as specified in TNFR.pdf §2.2.1 (EN - Structural reception).

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
        """Validate EN-specific preconditions with strict canonical checks.

        Implements TNFR.pdf §2.2.1 precondition validation:
        1. EPI < saturation threshold (receptive capacity available)
        2. DNFR < threshold (minimal dissonance for stable integration)
        3. Emission sources check (warning for isolated nodes)

        Raises
        ------
        ValueError
            If EPI too high or DNFR too high for reception
        """
        from .preconditions.reception import validate_reception_strict

        validate_reception_strict(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect EN-specific metrics."""
        from .metrics import reception_metrics

        return reception_metrics(G, node, state_before["epi"])


@register_operator
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
        C_local_before = compute_local_coherence(
            G, node, radius=kw.get("coherence_radius", 1)
        )

        # Capture ΔNFR before Coherence application for telemetry
        dnfr_before = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Delegate to parent __call__ which applies grammar (including Coherence reduction)
        super().__call__(G, node, **kw)

        # Apply phase locking after grammar application
        locking_coef = kw.get("phase_locking_coefficient", 0.3)
        self._apply_phase_locking(G, node, locking_coefficient=locking_coef)

        # Capture C(t) after IL application
        C_global_after = compute_global_coherence(G)
        C_local_after = compute_local_coherence(
            G, node, radius=kw.get("coherence_radius", 1)
        )

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

        theta_neighbors = [
            float(get_attr(G.nodes[n], ALIAS_THETA, 0.0)) for n in neighbors
        ]

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
        propagate = kw.get(
            "propagate_to_network", G.graph.get("OZ_ENABLE_PROPAGATION", True)
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

    - **Cardiac Coherence Propagation**

      - **Mechanism**: HRV coherence from heart rhythm spreads through vagal nerve network
      - **RA Role**: Propagates coherent cardiac pattern to brain, organs, peripheral systems
      - **Observable**: Reduced heart rate variability entropy, increased baroreflex sensitivity
      - **Sequence**: AL (heart initiates) → IL (stabilizes rhythm) → RA (spreads to body)
      - **Metrics**: ΔHRV coherence across organ systems, autonomic tone synchronization

    - **Neural Synchronization Cascades**

      - **Mechanism**: Synchronized neuronal firing in one region propagates to connected areas
      - **RA Role**: Transmits oscillatory patterns (e.g., gamma, theta) across brain networks
      - **Observable**: EEG phase synchronization indices, functional connectivity increases
      - **Sequence**: THOL (local synchrony emerges) → UM (regions couple) → RA (network sync)
      - **Clinical**: Meditation-induced alpha coherence, seizure propagation dynamics

    - **Immune Cascade Activation**

      - **Mechanism**: Cytokine signaling propagates immune response across tissue
      - **RA Role**: Coordinates cellular activation without losing response specificity
      - **Observable**: Immune cell recruitment patterns, synchronized cytokine expression
      - **Pathological**: Cytokine storms as uncontrolled RA (missing IL stabilization)

    - **Morphogenetic Field Propagation**

      - **Mechanism**: Developmental signals organize tissue pattern formation
      - **RA Role**: Spreads positional information maintaining structural identity
      - **Observable**: Hox gene expression gradients, limb bud patterning
      - **TNFR Model**: RA preserves EPI identity (cell type) while propagating position

    **Cognitive**:

    - **Insight Propagation ("Aha!" Moments)**

      - **Mechanism**: Single conceptual breakthrough reorganizes entire knowledge network
      - **RA Role**: Key understanding cascades through related concepts, illuminating connections
      - **Observable**: Sudden problem-solving, gestalt shifts, conceptual restructuring
      - **Sequence**: OZ (conceptual tension) → THOL (insight emerges) → RA (understanding spreads)
      - **Example**: Understanding recursion suddenly clarifies programming, fractals, self-reference

    - **Meme Propagation**

      - **Mechanism**: Ideas spread through population maintaining core structure
      - **RA Role**: Transmits conceptual pattern ("viral" spread) with identity preservation
      - **Observable**: Social media virality curves, idea adoption S-curves
      - **Pathological**: Misinformation spread (RA without IL verification)
      - **Counter**: IL (fact-checking) dampens incoherent RA

    - **Knowledge Transfer in Learning**

      - **Mechanism**: Expertise propagates from teacher to student network
      - **RA Role**: Transmits structured understanding, not just information
      - **Observable**: Student mental models converging toward expert patterns
      - **Sequence**: EN (student receives) → IL (integrates) → RA (applies to new contexts)
      - **Metrics**: Transfer learning success, analogical reasoning improvements

    - **Attention Cascades**

      - **Mechanism**: Focus on one element draws attention to connected elements
      - **RA Role**: Spreads attentional coherence across semantic network
      - **Observable**: Priming effects, associative memory activation
      - **Example**: Seeing "doctor" activates "nurse", "hospital", "stethoscope"

    **Social**:

    - **Collective Emotional Contagion**

      - **Mechanism**: Emotion spreads through group (laughter, panic, enthusiasm)
      - **RA Role**: Propagates affective state while maintaining emotional coherence
      - **Observable**: Synchronized facial expressions, heart rate convergence, mirroring
      - **Sequence**: AL (individual expresses) → UM (others attune) → RA (group synchrony)
      - **Examples**: Concert crowds, protest movements, team celebrations

    - **Social Movement Diffusion**

      - **Mechanism**: Values/practices spread through social networks
      - **RA Role**: Propagates coherent ideology maintaining identity
      - **Observable**: Network diffusion curves, hashtag propagation, adoption cascades
      - **Critical Mass**: RA accelerates post-UM (coupling) threshold
      - **Examples**: Arab Spring, #MeToo, climate activism

    - **Innovation Diffusion in Organizations**

      - **Mechanism**: New practices spread through company departments
      - **RA Role**: Transfers best practices while adapting to local context
      - **Observable**: Practice adoption rates, cross-functional knowledge sharing
      - **Sequence**: THOL (innovation emerges) → UM (early adopters couple) → RA (spreads)
      - **Barriers**: OZ (departmental resistance) can block RA

    - **Cultural Pattern Transmission**

      - **Mechanism**: Rituals, norms, symbols propagate across generations
      - **RA Role**: Maintains cultural identity while allowing adaptation
      - **Observable**: Cultural continuity metrics, tradition persistence
      - **Balance**: RA (preservation) vs ZHIR (cultural evolution)

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
    -------
    **Propagation Metrics**:

    - **Propagation Distance**: Number of nodes reached from source

      - Measurement: Graph traversal depth from origin
      - Healthy: Distance scales with network density
      - Pathological: Isolated propagation (missing UM coupling)

    - **Amplification Factor**: Coherence gain through network

      - Formula: ``C(t_after) / C(t_before)`` at network level
      - Healthy: Factor > 1.0 (resonance amplifies)
      - Degraded: Factor ≈ 1.0 (diffusion without resonance)

    - **Propagation Speed**: Rate of coherence spread

      - Measurement: Nodes activated per time step
      - Fast: High νf alignment, strong UM coupling
      - Slow: Phase misalignment, weak network connectivity

    **Identity Preservation Metrics**:

    - **EPI Structure Similarity**: How well propagated EPI matches source

      - Measurement: Cosine similarity of EPI vectors (if structured)
      - Healthy: Similarity > 0.8 (identity preserved)
      - Distorted: Similarity < 0.5 (pattern corruption)

    - **epi_kind Consistency**: Semantic label propagation

      - Measurement: Fraction of influenced nodes adopting source ``epi_kind``
      - Healthy: > 70% adoption in coupled neighborhood
      - Fragmented: < 30% (RA failed, revert to AL)

    **Network-Level Metrics**:

    - **Global Coherence Increase (ΔC(t))**:

      - Formula: ``C_global(t+1) - C_global(t)`` after RA application
      - Healthy: ΔC(t) > 0 (network more coherent)
      - Harmful: ΔC(t) < 0 (RA applied incorrectly, spreading chaos)

    - **Phase Synchronization Index**:

      - Measurement: Kuramoto order parameter before/after RA
      - Healthy: Index increases toward 1.0
      - Misaligned: Index decreases (needs UM first)

    **Frequency Metrics**:

    - **Collective νf Shift**: Average νf change across influenced nodes

      - Measurement: ``mean(νf_influenced) - mean(νf_before)``
      - Healthy: Positive shift (amplification)
      - Note: Current implementation may not fully track this (see related issues)

    Compatibility
    -------------
    **Synergistic Sequences** (amplify each other's effects):

    - **UM → RA**: Canonical resonance pattern

      - UM establishes phase coupling
      - RA propagates through coupled network
      - Result: Coherent network-wide reorganization
      - Analogy: Tuning instruments (UM) then playing symphony (RA)

    - **IL → RA**: Stable propagation

      - IL stabilizes source pattern
      - RA propagates verified coherence
      - Result: Reliable, non-distorted transmission
      - Use: Knowledge transfer, cultural preservation

    - **AL → RA**: Broadcast pattern

      - AL initiates new coherence
      - RA immediately spreads to receptive nodes
      - Result: Rapid network activation
      - Use: Idea dissemination, emotional contagion
      - Risk: Unstable if AL not stabilized (add IL between)

    **Required Prerequisites** (apply before RA):

    - **UM before RA** (when network uncoupled):

      - Without UM: RA has no propagation pathways
      - Symptom: RA applied to isolated node
      - Fix: ``run_sequence(G, node, [Coupling(), Resonance()])``

    - **IL before RA** (when source unstable):

      - Without IL: RA propagates noise/chaos
      - Symptom: High ΔNFR, low EPI at source
      - Fix: ``run_sequence(G, node, [Coherence(), Resonance()])``

    **Natural Progressions** (what to apply after RA):

    - **RA → IL**: Lock in propagated coherence

      - RA spreads pattern
      - IL stabilizes across network
      - Result: Persistent network-wide coherence
      - Example: Post-meditation integration, learning consolidation

    - **RA → EN**: Distributed reception

      - RA broadcasts from source
      - EN nodes receive and integrate
      - Result: Coordinated network update
      - Example: Software update propagation, news dissemination

    - **RA → SHA**: Resonance completion

      - RA propagates pattern
      - SHA pauses further spreading
      - Result: Bounded coherence domain
      - Example: Localized neural assembly, cultural enclave

    **Incompatible Patterns** (avoid or use carefully):

    - **SHA → RA**: Contradiction

      - SHA silences node (νf → 0)
      - RA requires active propagation
      - Result: Ineffective RA (nothing to propagate)
      - Exception: SHA → NAV → RA (reactivation sequence)

    - **OZ → RA** (unconstrained dissonance):

      - OZ introduces chaos
      - RA propagates chaos (pathological)
      - Result: Network destabilization
      - Safe: OZ → IL → RA (constrain dissonance first)
      - Intentional: OZ → RA for creative disruption (rare)

    - **Multiple RA without IL**:

      - Repeated RA can blur pattern identity
      - Result: "Telephone game" distortion
      - Fix: Interleave IL to preserve structure
      - Pattern: RA → IL → RA → IL (controlled cascade)

    **Edge Cases**:

    - **RA on fully connected graph**:

      - All nodes receive simultaneously
      - Result: Instantaneous network coherence (no cascade)
      - Efficiency: RA becomes equivalent to broadcast AL

    - **RA on tree topology**:

      - Clean propagation paths, no loops
      - Result: Predictable cascade from root
      - Application: Hierarchical organizations, decision trees

    - **RA on scale-free network**:

      - Hub nodes amplify propagation
      - Result: Exponential spread through hubs
      - Application: Social networks, viral marketing
      - Risk: Hub failure blocks cascade (fragile)

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

    **Example (Meditation Group Coherence):**

    >>> # Meditation teacher establishes coherent state, propagates to students
    >>> import networkx as nx
    >>> import random
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coupling, Resonance
    >>> from tnfr.metrics.coherence import compute_global_coherence
    >>> from tnfr.constants import EPI_PRIMARY
    >>>
    >>> G_meditation = nx.Graph()
    >>> # Teacher with high coherence
    >>> G_meditation.add_node("teacher")
    >>> G_meditation.nodes["teacher"][EPI_PRIMARY] = 0.85
    >>> G_meditation.nodes["teacher"]["vf"] = 1.2
    >>> G_meditation.nodes["teacher"]["theta"] = 0.0
    >>>
    >>> # Students with lower coherence, varied phases
    >>> for i in range(10):
    ...     student_id = f"student_{i}"
    ...     G_meditation.add_node(student_id)
    ...     G_meditation.nodes[student_id][EPI_PRIMARY] = 0.3
    ...     G_meditation.nodes[student_id]["vf"] = 0.9
    ...     G_meditation.nodes[student_id]["theta"] = random.uniform(-0.5, 0.5)
    ...     # Teacher couples with students through presence (UM)
    ...     G_meditation.add_edge("teacher", student_id)
    >>>
    >>> # Teacher's coherence resonates to group (RA)
    >>> c_before = compute_global_coherence(G_meditation)
    >>> run_sequence(G_meditation, "teacher", [Coupling(), Resonance()])
    >>> c_after = compute_global_coherence(G_meditation)
    >>>
    >>> # Result: Students' EPI increases, phases align, network coherence rises
    >>> # Group enters synchronized meditative state through RA propagation

    **Example (Viral Meme Cascade):**

    >>> # Idea originates, couples with early adopters, resonates through network
    >>> import networkx as nx
    >>> from tnfr.structural import run_sequence
    >>> from tnfr.operators.definitions import Coupling, Resonance
    >>> from tnfr.constants import EPI_PRIMARY
    >>>
    >>> G_social = nx.barabasi_albert_graph(100, 3)  # Scale-free social network
    >>> origin = 0  # Hub node with high connectivity
    >>>
    >>> # Set initial state: one coherent idea, rest neutral
    >>> for node in G_social.nodes():
    ...     G_social.nodes[node][EPI_PRIMARY] = 0.9 if node == origin else 0.1
    ...     G_social.nodes[node]["vf"] = 1.0
    ...     G_social.nodes[node]["epi_kind"] = "viral_meme" if node == origin else "neutral"
    ...     G_social.nodes[node]["theta"] = 0.0
    >>>
    >>> # Phase 1: Early adopters couple with origin (UM)
    >>> run_sequence(G_social, origin, [Coupling()])
    >>>
    >>> # Phase 2: Idea resonates through coupled network (RA)
    >>> adoption_wave = [origin]
    >>> for wave_step in range(5):  # 5 propagation hops
    ...     for node in list(adoption_wave):
    ...         run_sequence(G_social, node, [Resonance()])
    ...         # Add newly influenced nodes to wave
    ...         for neighbor in G_social.neighbors(node):
    ...             if G_social.nodes[neighbor][EPI_PRIMARY] > 0.5 and neighbor not in adoption_wave:
    ...                 adoption_wave.append(neighbor)
    >>>
    >>> # Result: Meme spreads through network maintaining identity
    >>> adopters = [n for n in G_social.nodes() if G_social.nodes[n].get("epi_kind") == "viral_meme"]
    >>> adoption_rate = len(adopters) / 100
    >>> # Demonstrates RA creating resonant cascade through scale-free topology

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
        """Collect RA-specific metrics with canonical νf amplification tracking."""
        from .metrics import resonance_metrics

        return resonance_metrics(
            G,
            node,
            state_before["epi"],
            vf_before=state_before["vf"],  # Include νf for amplification tracking
        )


@register_operator
class Silence(Operator):
    """Silence structural operator (SHA) - Preservation through structural pause.

    Activates glyph ``SHA`` to lower νf and hold the local EPI invariant, suspending
    reorganization to preserve the node's current coherence state. SHA implements
    **latency state management** with explicit temporal tracking.

    TNFR Context
    ------------
    Silence (SHA) creates structural latency - a state where νf ≈ 0, causing the nodal
    equation ∂EPI/∂t = νf · ΔNFR(t) to approach zero regardless of ΔNFR. This preserves
    the current EPI form intact, preventing reorganization. SHA is essential for memory,
    consolidation, and maintaining structural identity during network turbulence.

    According to TNFR.pdf §2.3.10, SHA is not merely frequency reduction but a
    **transition to latent state** with temporal tracking for analyzing memory
    consolidation, incubation periods, and protective pauses.

    **Key Elements:**

    - **Frequency Suppression**: Reduces νf to near-zero (structural pause)
    - **Form Preservation**: EPI remains unchanged despite external pressures
    - **Latent Memory**: Stored patterns awaiting reactivation
    - **Strategic Inaction**: Deliberate non-reorganization as protective mechanism
    - **Temporal Tracking**: Explicit duration and state management

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
    - **SHA → IL → AL**: Silence → stabilization → reactivation (coherent awakening)
    - **SHA → EN → IL**: Silence → external reception → stabilization (network reactivation)
    - **SHA → NAV**: Preserved structure transitions (controlled change)
    - **OZ → SHA**: Dissonance contained (protective pause)

    **AVOID**: SHA → AL (direct reactivation violates structural continuity - requires intermediate stabilization)
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

    Latency State Attributes
    -------------------------
    SHA sets the following node attributes for latency tracking:

    - **latent**: Boolean flag indicating node is in latent state
    - **latency_start_time**: ISO 8601 UTC timestamp when silence began
    - **preserved_epi**: Snapshot of EPI at silence entry
    - **silence_duration**: Cumulative duration in latent state (updated on subsequent steps)

    Metrics
    -----------------
    - νf reduction: Degree of frequency suppression
    - EPI stability: Variance over silence period (should be ~0)
    - Silence duration: Time in latent state
    - Preservation effectiveness: EPI integrity post-silence
    - Preservation integrity: Measures EPI variance during silence

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

    Extended Clinical Documentation
    --------------------------------
    For detailed clinical protocols, expected telemetry, physiological correlates,
    and scientific references, see:

    **docs/source/examples/SHA_CLINICAL_APPLICATIONS.md**

    Comprehensive documentation includes:
    - Cardiac Coherence Training (HRV consolidation)
    - Trauma Therapy (protective containment)
    - Sleep & Memory Consolidation (neuroscience applications)
    - Post-Exercise Recovery (athletic training)
    - Meditation & Mindfulness (contemplative practices)
    - Organizational Strategy (strategic pause protocols)

    **Executable Examples**: examples/biomedical/
    - cardiac_coherence_sha.py
    - trauma_containment_sha.py
    - sleep_consolidation_sha.py
    - recovery_protocols_sha.py
    """

    __slots__ = ()
    name: ClassVar[str] = SILENCE
    glyph: ClassVar[Glyph] = Glyph.SHA

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply SHA with latency state tracking.

        Establishes latency state before delegating to grammar execution.
        This ensures every silence operation creates explicit latent state
        tracking as required by TNFR.pdf §2.3.10 (SHA - Silencio estructural).

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
        """
        # Mark latency state BEFORE grammar execution
        self._mark_latency_state(G, node)

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _mark_latency_state(self, G: TNFRGraph, node: Any) -> None:
        """Mark latency state for SHA operator.

        According to TNFR.pdf §2.3.10, SHA implements structural silence
        with temporal tracking for memory consolidation and protective pauses.

        This method establishes:
        - Latent flag: Boolean indicating node is in latent state
        - Temporal marker: ISO timestamp when silence began
        - Preserved EPI: Snapshot of EPI for integrity verification
        - Duration tracker: Cumulative time in silence (initialized to 0)

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node for silence marking.

        Notes
        -----
        Sets the following node attributes:
        - latent: True (node in latent state)
        - latency_start_time: ISO 8601 UTC timestamp
        - preserved_epi: Current EPI value snapshot
        - silence_duration: 0.0 (initialized, updated by external time tracking)
        """
        from datetime import datetime, timezone

        from ..alias import get_attr

        # Always set latency state (SHA can be applied multiple times)
        G.nodes[node]["latent"] = True

        # Set start time for this latency period
        latency_start_time = datetime.now(timezone.utc).isoformat()
        G.nodes[node]["latency_start_time"] = latency_start_time

        # Preserve current EPI for integrity checking
        epi_value = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        G.nodes[node]["preserved_epi"] = epi_value

        # Initialize silence duration (will be updated by external tracking)
        G.nodes[node]["silence_duration"] = 0.0

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

    TNFR Context
    ------------
    Contraction (NUL) embodies harmonic contraction - the complementary principle to
    expansion (VAL). When structure contracts (W → W' where W' = W × λ, λ < 1), it
    doesn't simply shrink; it undergoes **densification**: the structural pressure
    concentrates, amplifying ΔNFR while reducing volume.

    **Key Elements:**

    - **Harmonic Contraction**: Volume reduction W → W × λ (default λ = 0.85)
    - **Density Amplification**: ΔNFR → ΔNFR × ρ (default ρ = 1.35)
    - **Structural Pressure**: Product νf × ΔNFR slightly increases (~1.15x)
    - **Core Strengthening**: Peripheral trajectories fold into coherent center
    - **Complementary to VAL**: Enables expand-contract cycles for exploration-consolidation

    **Canonical Densification:**

    - Volume contraction: V' = V × NUL_scale (default 0.85)
    - Density amplification: ΔNFR' = ΔNFR × NUL_densification_factor (default 1.35)
    - Product effect: νf × ΔNFR ≈ 0.85 × 1.35 ≈ 1.15 (slight structural pressure increase)
    - Equilibrium preservation: ΔNFR = 0 remains 0
    - Sign preservation: Negative ΔNFR amplifies correctly (intensified contraction)

    **Relationship to Nodal Equation:**

    The nodal equation ∂EPI/∂t = νf · ΔNFR(t) remains valid through NUL application.
    While νf decreases (reorganization rate slows), ΔNFR increases (pressure concentrates),
    keeping the product bounded. This preserves structural integrity during contraction.

    **Role in VAL ↔ NUL Cycles:**

    NUL is the complementary operator to VAL (Expansion), enabling rhythmic cycles of
    exploration and consolidation. VAL → NUL → IL sequences are fundamental to TNFR
    dynamics: expand to explore, contract to consolidate, stabilize to preserve.

    Use Cases
    ---------
    **Biomedical**:

    - **Apoptosis**: Programmed cell death (controlled elimination)
    - **Wound Healing**: Tissue contraction closing wound gaps
    - **Neural Pruning**: Synaptic elimination strengthening key pathways
    - **Muscle Contraction**: Coordinated fiber shortening for movement

    **Cognitive**:

    - **Focus Intensification**: Attention narrowing to essential elements
    - **Concept Refinement**: Simplifying complex ideas to core principles
    - **Mental Compression**: "Less is more" - removing cognitive clutter
    - **Memory Consolidation**: Compressing experiences into dense representations

    **Social**:

    - **Team Downsizing**: Strategic workforce reduction to core competencies
    - **Resource Consolidation**: Pooling distributed resources for efficiency
    - **Core Competency Focus**: Eliminating peripheral activities
    - **Crisis Response**: Defensive contraction under external pressure

    Typical Sequences
    ---------------------------
    **Valid Patterns:**

    - **NUL → IL**: Contract then stabilize (safe consolidation)
    - **VAL → NUL → IL**: Expand-contract-stabilize cycle (exploration-consolidation)
    - **THOL → NUL**: Self-organize then refine (emergent structure consolidation)
    - **OZ → NUL**: Dissonance followed by compression (pressure intensification)
    - **NUL → SHA**: Compress then silence (preservation through contraction)
    - **EN → NUL → IL**: Receive, compress, stabilize (efficient integration)

    **Avoid Patterns:**

    - **NUL → VAL**: Contradictory (immediate reversal wastes structural energy)
    - **NUL → NUL**: Over-compression risk (may trigger structural collapse)
    - **NUL → OZ**: Compression + dissonance = dangerous instability
    - **Excessive NUL**: Multiple contractions without stabilization (fragmentation risk)

    Preconditions
    -------------
    - Node must have adequate EPI baseline (cannot contract from near-zero)
    - ΔNFR should be present (though densification amplifies it)
    - Sufficient structural integrity to withstand compression

    Structural Effects
    ------------------
    - **EPI**: Decreases (volume reduction)
    - **νf**: Decreases (reorganization rate slows)
    - **ΔNFR**: Increases (densification - primary effect)
    - **C(t)**: May increase locally (tighter coherence gradients)
    - **Product νf × ΔNFR**: Slight increase (~1.15x)

    Metrics
    -----------------
    - Volume reduction: EPI change ratio
    - Densification factor: ΔNFR amplification
    - Frequency decrease: νf reduction
    - Structural pressure: Product νf × ΔNFR

    Compatibility
    ---------------------
    **Compatible with**: IL (stabilization), SHA (preservation), THOL (organization),
    EN (reception before contraction)

    **Complementary with**: VAL (expansion) - enables rhythmic cycles

    **Avoid with**: OZ (dissonance), consecutive NUL (over-compression)

    **Natural progressions**: NUL typically followed by IL (stabilization) or SHA
    (preservation) to seal contracted form

    Warnings
    --------
    **Over-compression Risks:**

    - **Structural Collapse**: Excessive contraction can fragment coherence
    - **Loss of Degrees of Freedom**: Irreversible elimination of structural dimensions
    - **Requires Adequate Baseline**: Cannot contract from EPI ≈ 0 (no structure to compress)
    - **Irreversibility**: Cannot reverse without VAL (expansion) - contraction loses information

    **Collapse Conditions:**

    - Multiple consecutive NUL without stabilization (IL)
    - Contraction when EPI already critically low
    - NUL → OZ sequences (compression + instability)
    - Insufficient network coupling to maintain identity

    **Safe Usage:**

    - Always follow with IL (Coherence) or SHA (Silence)
    - Ensure adequate EPI baseline before contraction
    - Use VAL → NUL cycles rather than isolated NUL
    - Monitor C(t) to detect fragmentation

    Comparison with Complementary Operators
    ---------------------------------------
    **NUL vs. VAL (Expansion)**:

    - NUL contracts volume, VAL expands it
    - NUL increases ΔNFR density, VAL distributes it
    - NUL consolidates, VAL explores
    - Together enable expand-contract rhythms

    **NUL vs. IL (Coherence)**:

    - NUL compresses structure, IL stabilizes it
    - NUL increases ΔNFR (densification), IL reduces it (stabilization)
    - NUL changes geometry, IL preserves it
    - Often used in sequence: NUL → IL

    **NUL vs. THOL (Self-organization)**:

    - NUL simplifies structure, THOL complexifies it
    - NUL reduces dimensions, THOL creates sub-EPIs
    - NUL consolidates, THOL differentiates
    - Can work sequentially: THOL → NUL (organize then refine)

    Examples
    --------
    **Technical Example:**

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.operators import apply_glyph
    >>> from tnfr.types import Glyph
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("iota", epi=0.5, vf=1.0)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.1
    >>> # Apply NUL via canonical glyph application
    >>> apply_glyph(G, node, Glyph.NUL)
    >>> # Verify densification: ΔNFR increased despite contraction
    >>> G.nodes[node][DNFR_PRIMARY] > 0.1  # doctest: +SKIP
    True
    >>> # Check telemetry for densification event
    >>> 'nul_densification_log' in G.graph  # doctest: +SKIP
    True

    **Example 1: Neural Pruning**

    >>> # Brain eliminates weak synaptic connections
    >>> G_brain, synapse = create_nfr("neural_connection", epi=0.39, vf=1.05)
    >>> # Synapse has weak activity pattern
    >>> G_brain.nodes[synapse][DNFR_PRIMARY] = 0.05
    >>> # Apply NUL to eliminate weak connection
    >>> from tnfr.structural import run_sequence
    >>> from tnfr.operators.definitions import Contraction, Coherence
    >>> run_sequence(G_brain, synapse, [Contraction(), Coherence()])
    >>> # Result: Synapse contracts, neural network becomes more efficient
    >>> # Remaining connections are strengthened through consolidation

    **Example 2: Strategic Focus**

    >>> # Company eliminates peripheral business units
    >>> G_company, strategy = create_nfr("business_model", epi=0.42, vf=1.00)
    >>> # Company has diffuse strategy with many weak initiatives
    >>> G_company.nodes[strategy][DNFR_PRIMARY] = 0.08
    >>> # Apply NUL to focus on core competencies
    >>> run_sequence(G_company, strategy, [Contraction(), Coherence()])
    >>> # Result: Strategy contracts to core, peripheral units eliminated
    >>> # Core competencies receive concentrated resources

    **Example 3: Expand-Contract Cycle**

    >>> # Learning cycle: explore broadly then consolidate
    >>> from tnfr.operators.definitions import Expansion
    >>> G_learning, concept = create_nfr("understanding", epi=0.35, vf=0.95)
    >>> G_learning.nodes[concept][DNFR_PRIMARY] = 0.06
    >>> # VAL → NUL → IL: Expand → Contract → Stabilize
    >>> run_sequence(G_learning, concept, [Expansion(), Contraction(), Coherence()])
    >>> # Result: Exploration phase (VAL) followed by consolidation (NUL)
    >>> # Final understanding is both broad (from VAL) and coherent (from NUL → IL)

    **Example 4: Memory Consolidation**

    >>> # Brain compresses daily experiences into dense memories
    >>> G_memory, experience = create_nfr("daily_events", epi=0.55, vf=1.10)
    >>> # Many experiences need compression for long-term storage
    >>> G_memory.nodes[experience][DNFR_PRIMARY] = 0.12
    >>> # NUL → SHA: Compress then preserve (sleep consolidation)
    >>> from tnfr.operators.definitions import Silence
    >>> run_sequence(G_memory, experience, [Contraction(), Silence()])
    >>> # Result: Experiences compressed into efficient representations
    >>> # Preserved in stable form for later retrieval

    See Also
    --------
    Expansion : Complementary operator enabling expand-contract cycles
    Coherence : Stabilizes contracted structure (NUL → IL pattern)
    SelfOrganization : Can follow contraction (THOL → NUL refinement)
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

    **Vibrational Metabolism (Canonical THOL):**

    THOL implements the metabolic principle: capturing network vibrational signals
    (EPI gradients, phase variance) and transforming them into internal structure
    (sub-EPIs). This ensures that bifurcation reflects not only internal acceleration
    but also the network's coherence field.

    Metabolic formula: ``sub-EPI = base + gradient*w₁ + variance*w₂``

    - If node has neighbors: Captures and metabolizes network signals
    - If node is isolated: Falls back to pure internal bifurcation
    - Configurable via ``THOL_METABOLIC_ENABLED`` and weight parameters

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
            # Validate depth before bifurcation
            self._validate_bifurcation_depth(G, node)
            self._spawn_sub_epi(G, node, d2_epi=d2_epi, tau=tau)

        # CANONICAL VALIDATION: Verify collective coherence of sub-EPIs
        # When THOL creates multiple sub-EPIs, they must form a coherent ensemble
        # that preserves the structural identity of the parent node (TNFR Manual §2.2.10)
        # Always validate if node has sub-EPIs (whether created now or previously)
        if G.nodes[node].get("sub_epis"):
            self._validate_collective_coherence(G, node)

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
        """Generate sub-EPI through bifurcation with vibrational metabolism.

        When acceleration exceeds threshold, creates nested sub-structure that:
        1. Captures network vibrational signals (metabolic perception)
        2. Metabolizes signals into sub-EPI magnitude (digestion)
        3. Inherits properties from parent while integrating field context

        This implements canonical THOL: "reorganizes external experience into
        internal structure without external instruction".

        ARCHITECTURAL: Sub-EPIs are created as independent NFR nodes to enable
        operational fractality - recursive operator application, hierarchical metrics,
        and multi-level bifurcation.

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
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA
        from .metabolism import capture_network_signals, metabolize_signals_into_subepi

        # Get current node state
        parent_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        parent_vf = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))
        parent_theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

        # Check if vibrational metabolism is enabled
        metabolic_enabled = G.graph.get("THOL_METABOLIC_ENABLED", True)

        # CANONICAL METABOLISM: Capture network context
        network_signals = None
        if metabolic_enabled:
            network_signals = capture_network_signals(G, node)

        # Get metabolic weights from graph config
        gradient_weight = float(G.graph.get("THOL_METABOLIC_GRADIENT_WEIGHT", 0.15))
        complexity_weight = float(G.graph.get("THOL_METABOLIC_COMPLEXITY_WEIGHT", 0.10))

        # CANONICAL METABOLISM: Digest signals into sub-EPI
        sub_epi_value = metabolize_signals_into_subepi(
            parent_epi=parent_epi,
            signals=network_signals if metabolic_enabled else None,
            d2_epi=d2_epi,
            scaling_factor=_THOL_SUB_EPI_SCALING,
            gradient_weight=gradient_weight,
            complexity_weight=complexity_weight,
        )

        # Get current timestamp from glyph history length
        timestamp = len(G.nodes[node].get("glyph_history", []))

        # Determine parent bifurcation level for hierarchical telemetry
        parent_level = G.nodes[node].get("_bifurcation_level", 0)
        child_level = parent_level + 1
        
        # Construct hierarchy path for full traceability
        parent_path = G.nodes[node].get("_hierarchy_path", [])
        child_path = parent_path + [node]

        # ARCHITECTURAL: Create sub-EPI as independent NFR node
        # This enables operational fractality - recursive operators, hierarchical metrics
        sub_node_id = self._create_sub_node(
            G,
            parent_node=node,
            sub_epi=sub_epi_value,
            parent_vf=parent_vf,
            parent_theta=parent_theta,
            child_level=child_level,
            child_path=child_path,
        )

        # Store sub-EPI metadata for telemetry and backward compatibility
        sub_epi_record = {
            "epi": sub_epi_value,
            "vf": parent_vf,
            "timestamp": timestamp,
            "d2_epi": d2_epi,
            "tau": tau,
            "node_id": sub_node_id,  # Reference to independent node
            "metabolized": network_signals is not None and metabolic_enabled,
            "network_signals": network_signals,
            "bifurcation_level": child_level,  # Hierarchical depth tracking
            "hierarchy_path": child_path,  # Full parent chain for traceability
        }

        # Keep metadata list for telemetry/metrics backward compatibility
        sub_epis = G.nodes[node].get("sub_epis", [])
        sub_epis.append(sub_epi_record)
        G.nodes[node]["sub_epis"] = sub_epis

        # Increment parent EPI using canonical emergence contribution
        # This reflects that bifurcation increases total structural complexity
        new_epi = parent_epi + sub_epi_value * _THOL_EMERGENCE_CONTRIBUTION
        set_attr(G.nodes[node], ALIAS_EPI, new_epi)

        # CANONICAL PROPAGATION: Enable network cascade dynamics
        if G.graph.get("THOL_PROPAGATION_ENABLED", True):
            from .metabolism import propagate_subepi_to_network

            propagations = propagate_subepi_to_network(G, node, sub_epi_record)

            # Record propagation telemetry for cascade analysis
            if propagations:
                G.graph.setdefault("thol_propagations", []).append(
                    {
                        "source_node": node,
                        "sub_epi": sub_epi_value,
                        "propagations": propagations,
                        "timestamp": timestamp,
                    }
                )

    def _create_sub_node(
        self,
        G: TNFRGraph,
        parent_node: Any,
        sub_epi: float,
        parent_vf: float,
        parent_theta: float,
        child_level: int,
        child_path: list,
    ) -> str:
        """Create sub-EPI as independent NFR node for operational fractality.

        Sub-nodes are full TNFR nodes that can have operators applied, bifurcate
        recursively, and contribute to hierarchical metrics.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the parent node
        parent_node : Any
            Parent node identifier
        sub_epi : float
            EPI value for the sub-node
        parent_vf : float
            Parent's structural frequency (inherited with damping)
        parent_theta : float
            Parent's phase (inherited)
        child_level : int
            Bifurcation level for hierarchical tracking
        child_path : list
            Full hierarchy path (ancestor chain)

        Returns
        -------
        str
            Identifier of the newly created sub-node
        """
        from ..constants import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY, DNFR_PRIMARY
        from ..dynamics import set_delta_nfr_hook

        # Generate unique sub-node ID
        sub_nodes_list = G.nodes[parent_node].get("sub_nodes", [])
        sub_index = len(sub_nodes_list)
        sub_node_id = f"{parent_node}_sub_{sub_index}"

        # Get parent hierarchy level
        parent_hierarchy_level = G.nodes[parent_node].get("hierarchy_level", 0)

        # Inherit parent's vf with slight damping (canonical: 95%)
        sub_vf = parent_vf * 0.95

        # Create the sub-node with full TNFR state
        G.add_node(
            sub_node_id,
            **{
                EPI_PRIMARY: float(sub_epi),
                VF_PRIMARY: float(sub_vf),
                THETA_PRIMARY: float(parent_theta),
                DNFR_PRIMARY: 0.0,
                "parent_node": parent_node,
                "hierarchy_level": parent_hierarchy_level + 1,
                "_bifurcation_level": child_level,  # Hierarchical depth tracking
                "_hierarchy_path": child_path,  # Full ancestor chain
                "epi_history": [
                    float(sub_epi)
                ],  # Initialize history for future bifurcation
                "glyph_history": [],
            },
        )

        # Ensure ΔNFR hook is set for the sub-node
        # (inherits from graph-level hook, but ensure it's activated)
        if hasattr(G, "graph") and "_delta_nfr_hook" in G.graph:
            # Hook already set at graph level, will apply to sub-node automatically
            pass

        # Track sub-node in parent
        sub_nodes_list.append(sub_node_id)
        G.nodes[parent_node]["sub_nodes"] = sub_nodes_list

        # Track hierarchy in graph metadata
        hierarchy = G.graph.setdefault("hierarchy", {})
        hierarchy.setdefault(parent_node, []).append(sub_node_id)

        return sub_node_id

    def _validate_bifurcation_depth(self, G: TNFRGraph, node: Any) -> None:
        """Validate bifurcation depth before creating new sub-EPI.
        
        Checks if the current bifurcation level is at or exceeds the configured
        maximum depth. Issues a warning if depth limit is reached but still
        allows the bifurcation (for flexibility in research contexts).
        
        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node about to undergo bifurcation
            
        Notes
        -----
        TNFR Principle: Deep nesting reflects operational fractality (Invariant #7),
        but excessive depth may impact performance and interpretability. This
        validation provides observability without hard constraints.
        
        The warning allows tracking when hierarchies become complex, enabling
        researchers to study bifurcation patterns while maintaining system
        performance awareness.
        """
        import logging
        
        # Get current bifurcation level
        current_level = G.nodes[node].get("_bifurcation_level", 0)
        
        # Get max depth from graph config (default: 5 levels)
        max_depth = int(G.graph.get("THOL_MAX_BIFURCATION_DEPTH", 5))
        
        # Warn if at or exceeding maximum
        if current_level >= max_depth:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Node {node}: Bifurcation depth ({current_level}) at/exceeds "
                f"maximum ({max_depth}). Deep nesting may impact performance. "
                f"Consider adjusting THOL_MAX_BIFURCATION_DEPTH if intended."
            )
            
            # Record warning in node for telemetry
            G.nodes[node]["_thol_max_depth_warning"] = True
            
            # Record event for analysis
            events = G.graph.setdefault("thol_depth_warnings", [])
            events.append({
                "node": node,
                "depth": current_level,
                "max_depth": max_depth,
            })

    def _validate_collective_coherence(self, G: TNFRGraph, node: Any) -> None:
        """Validate collective coherence of sub-EPI ensemble after bifurcation.

        When THOL creates multiple sub-EPIs, they must form a coherent ensemble
        that preserves the structural identity of the parent node. This validation
        ensures the emergent sub-structures maintain structural alignment.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node that underwent bifurcation

        Notes
        -----
        TNFR Canonical Principle (TNFR Manual §2.2.10):
        "THOL reorganiza la forma desde dentro, en respuesta a la coherencia
        vibracional del campo. La autoorganización es resonancia estructurada
        desde el interior del nodo."

        Implication: Sub-EPIs are not random fragments but coherent structures
        that emerge from internal resonance.

        This method:
        1. Computes collective coherence of sub-EPI ensemble
        2. Stores coherence value for telemetry
        3. Logs warning if coherence < threshold
        4. Records event for analysis

        Does NOT fail the operation - allows monitoring and analysis of
        low-coherence bifurcations for research purposes.
        """
        import logging
        from .metabolism import compute_subepi_collective_coherence

        # Compute collective coherence
        coherence = compute_subepi_collective_coherence(G, node)

        # Store for telemetry (always store, even if 0.0 for single/no sub-EPIs)
        G.nodes[node]["_thol_collective_coherence"] = coherence

        # Get threshold from graph config
        min_coherence = float(G.graph.get("THOL_MIN_COLLECTIVE_COHERENCE", 0.3))

        # Validate against threshold (only warn if we have multiple sub-EPIs)
        sub_epis = G.nodes[node].get("sub_epis", [])
        if len(sub_epis) >= 2 and coherence < min_coherence:
            # Log warning (but don't fail - allow monitoring)
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Node {node}: THOL collective coherence ({coherence:.3f}) < "
                f"threshold ({min_coherence}). Sub-EPIs may be fragmenting. "
                f"Sub-EPI count: {len(sub_epis)}."
            )

            # Record event for analysis
            events = G.graph.setdefault("thol_coherence_warnings", [])
            events.append({
                "node": node,
                "coherence": coherence,
                "threshold": min_coherence,
                "sub_epi_count": len(sub_epis),
            })

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

    def _verify_postconditions(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> None:
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

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply NAV with regime detection and controlled transition.

        Implements TNFR.pdf §2.3.11 canonical transition logic:
        1. Detect current structural regime (latent/active/resonant)
        2. Handle latency reactivation if node was in silence (SHA → NAV)
        3. Apply grammar and structural transformation
        4. Collect metrics (if enabled)

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments:
            - phase_shift (float): Override default phase shift per regime
            - vf_factor (float): Override νf scaling for active regime (default: 1.0)
            - Other args forwarded to grammar layer

        Notes
        -----
        Regime-specific transformations (TNFR.pdf §2.3.11):

        **Latent → Active** (νf < 0.05 or latent flag):
        - νf × 1.2 (20% increase for gradual reactivation)
        - θ + 0.1 rad (small phase shift)
        - ΔNFR × 0.7 (30% reduction for smooth transition)

        **Active** (baseline state):
        - νf × vf_factor (default 1.0, configurable)
        - θ + 0.2 rad (standard phase shift)
        - ΔNFR × 0.8 (20% reduction)

        **Resonant → Active** (EPI > 0.5 AND νf > 0.8):
        - νf × 0.95 (5% reduction for stability)
        - θ + 0.15 rad (careful phase shift)
        - ΔNFR × 0.9 (10% reduction, gentle)

        Telemetry stored in G.graph["_nav_transitions"] tracks:
        - regime_origin, vf_before/after, theta_before/after, dnfr_before/after
        """
        from ..alias import get_attr, set_attr
        from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF

        # 1. Detect current regime and store for metrics collection
        current_regime = self._detect_regime(G, node)
        G.nodes[node]["_regime_before"] = current_regime

        # 2. Handle latency reactivation if applicable
        if G.nodes[node].get("latent", False):
            self._handle_latency_transition(G, node)

        # 3. Validate preconditions (if enabled)
        validate_preconditions = kw.get("validate_preconditions", True) or G.graph.get(
            "VALIDATE_PRECONDITIONS", False
        )
        if validate_preconditions:
            self._validate_preconditions(G, node)

        # 4. Capture state before for metrics/validation
        collect_metrics = kw.get("collect_metrics", False) or G.graph.get(
            "COLLECT_OPERATOR_METRICS", False
        )
        validate_equation = kw.get("validate_nodal_equation", False) or G.graph.get(
            "VALIDATE_NODAL_EQUATION", False
        )

        state_before = None
        if collect_metrics or validate_equation:
            state_before = self._capture_state(G, node)

        # 5. Apply grammar
        from . import apply_glyph_with_grammar
        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))

        # 6. Execute structural transition (BEFORE metrics collection)
        self._apply_structural_transition(G, node, current_regime, **kw)

        # 7. Optional nodal equation validation
        if validate_equation and state_before is not None:
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_EPI
            from .nodal_equation import validate_nodal_equation

            dt = float(kw.get("dt", 1.0))
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

        # 8. Optional metrics collection (AFTER structural transformation)
        if collect_metrics and state_before is not None:
            metrics = self._collect_metrics(G, node, state_before)
            if "operator_metrics" not in G.graph:
                G.graph["operator_metrics"] = []
            G.graph["operator_metrics"].append(metrics)

    def _detect_regime(self, G: TNFRGraph, node: Any) -> str:
        """Detect current structural regime: latent/active/resonant.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node.

        Returns
        -------
        str
            Regime classification: "latent", "active", or "resonant"

        Notes
        -----
        Classification criteria:
        - **Latent**: latent flag set OR νf < 0.05 (minimal reorganization capacity)
        - **Resonant**: EPI > 0.5 AND νf > 0.8 (high form + high frequency)
        - **Active**: Default (baseline operational state)
        """
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF

        epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
        latent = G.nodes[node].get("latent", False)

        if latent or vf < 0.05:
            return "latent"
        elif epi > 0.5 and vf > 0.8:
            return "resonant"
        else:
            return "active"

    def _handle_latency_transition(self, G: TNFRGraph, node: Any) -> None:
        """Handle transition from latent state (SHA → NAV flow).

        Similar to Emission._check_reactivation but for NAV-specific transitions.
        Validates silence duration and clears latency attributes.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node being reactivated.

        Warnings
        --------
        - Warns if node transitioning after extended silence (duration > MAX_SILENCE_DURATION)
        - Warns if EPI drifted significantly during silence (> 1% tolerance)

        Notes
        -----
        Clears latency-related attributes:
        - latent (flag)
        - latency_start_time (ISO timestamp)
        - preserved_epi (EPI snapshot from SHA)
        - silence_duration (computed duration)
        """
        from datetime import datetime, timezone

        # Verify silence duration if timestamp available
        if "latency_start_time" in G.nodes[node]:
            start = datetime.fromisoformat(G.nodes[node]["latency_start_time"])
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            G.nodes[node]["silence_duration"] = duration

            max_silence = G.graph.get("MAX_SILENCE_DURATION", float("inf"))
            if duration > max_silence:
                warnings.warn(
                    f"Node {node} transitioning after extended silence "
                    f"(duration: {duration:.2f}s, max: {max_silence:.2f}s)",
                    stacklevel=4,
                )

        # Check EPI preservation integrity
        preserved_epi = G.nodes[node].get("preserved_epi")
        if preserved_epi is not None:
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_EPI

            current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            epi_drift = abs(current_epi - preserved_epi)

            # Allow small numerical drift (1% tolerance)
            if epi_drift > 0.01 * abs(preserved_epi):
                warnings.warn(
                    f"Node {node} EPI drifted during silence "
                    f"(preserved: {preserved_epi:.3f}, current: {current_epi:.3f}, "
                    f"drift: {epi_drift:.3f})",
                    stacklevel=4,
                )

        # Clear latency state
        del G.nodes[node]["latent"]
        if "latency_start_time" in G.nodes[node]:
            del G.nodes[node]["latency_start_time"]
        if "preserved_epi" in G.nodes[node]:
            del G.nodes[node]["preserved_epi"]
        # Keep silence_duration for telemetry/metrics - don't delete it

    def _apply_structural_transition(
        self, G: TNFRGraph, node: Any, regime: str, **kw: Any
    ) -> None:
        """Apply structural transformation based on regime origin.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node.
        regime : str
            Origin regime: "latent", "active", or "resonant"
        **kw : Any
            Optional overrides:
            - phase_shift (float): Custom phase shift
            - vf_factor (float): Custom νf scaling for active regime

        Notes
        -----
        Applies regime-specific transformations to θ, νf, and ΔNFR following
        TNFR.pdf §2.3.11. All changes use canonical alias system (set_attr)
        to ensure proper attribute resolution.

        Telemetry appended to G.graph["_nav_transitions"] for analysis.
        """
        from ..alias import get_attr, set_attr
        from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF

        # Get current state
        theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
        vf = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))
        dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Apply regime-specific adjustments
        if regime == "latent":
            # Latent → Active: gradual reactivation
            vf_new = vf * 1.2  # 20% increase
            theta_shift = kw.get("phase_shift", 0.1)  # Small phase shift
            theta_new = (theta + theta_shift) % (2 * math.pi)
            dnfr_new = dnfr * 0.7  # 30% reduction for smooth transition
        elif regime == "active":
            # Active: standard transition
            vf_new = vf * kw.get("vf_factor", 1.0)  # Configurable
            theta_shift = kw.get("phase_shift", 0.2)  # Standard shift
            theta_new = (theta + theta_shift) % (2 * math.pi)
            dnfr_new = dnfr * 0.8  # 20% reduction
        else:  # resonant
            # Resonant → Active: careful transition (high energy state)
            vf_new = vf * 0.95  # 5% reduction for stability
            theta_shift = kw.get("phase_shift", 0.15)  # Careful phase shift
            theta_new = (theta + theta_shift) % (2 * math.pi)
            dnfr_new = dnfr * 0.9  # 10% reduction, gentle

        # Apply changes via canonical alias system
        set_attr(G.nodes[node], ALIAS_VF, vf_new)
        set_attr(G.nodes[node], ALIAS_THETA, theta_new)
        set_attr(G.nodes[node], ALIAS_DNFR, dnfr_new)

        # Telemetry tracking
        if "_nav_transitions" not in G.graph:
            G.graph["_nav_transitions"] = []
        G.graph["_nav_transitions"].append(
            {
                "node": node,
                "regime_origin": regime,
                "vf_before": vf,
                "vf_after": vf_new,
                "theta_before": theta,
                "theta_after": theta_new,
                "dnfr_before": dnfr,
                "dnfr_after": dnfr_new,
                "phase_shift": theta_new - theta,
            }
        )

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
            G,
            node,
            state_before["dnfr"],
            state_before["vf"],
            state_before["theta"],
            epi_before=state_before.get("epi"),
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

    Parameters
    ----------
    depth : int, optional
        Hierarchical nesting depth for multi-scale recursion (default: 1).
        - depth=1: Shallow recursion (single level, no multi-scale constraint)
        - depth>1: Deep recursion (multi-level hierarchy, requires U5 stabilizers)
        
    Notes
    -----
    **U5: Multi-Scale Coherence**: When depth>1, U5 grammar rule applies requiring
    scale stabilizers (IL or THOL) within ±3 operators to preserve coherence across
    hierarchical levels. This ensures C_parent ≥ α·ΣC_child per conservation principle.
    
    See UNIFIED_GRAMMAR_RULES.md § U5 for complete physical derivation.

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
    
    Deep recursion example requiring U5 stabilizers:
    >>> from tnfr.operators.definitions import Recursivity, Coherence, Silence
    >>> # depth=3 creates multi-level hierarchy - requires IL for U5
    >>> ops = [Recursivity(depth=3), Coherence(), Silence()]

    **Biomedical**: Fractal physiology (HRV, EEG), developmental recapitulation
    **Cognitive**: Recursive thinking, meta-cognition, self-referential processes
    **Social**: Cultural fractals, organizational self-similarity, meme propagation
    """

    __slots__ = ("depth",)
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH

    def __init__(self, depth: int = 1):
        """Initialize Recursivity operator with hierarchical depth.
        
        Parameters
        ----------
        depth : int, optional
            Nesting depth for multi-scale recursion (default: 1)
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth

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
