"""TNFR Operator: Reception

Reception structural operator (EN) - Anchoring external coherence into local structure.

**Physics**: See AGENTS.md § Reception
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import RECEPTION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


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