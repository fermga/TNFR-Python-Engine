"""TNFR Operator: Resonance

Resonance structural operator (RA) - Network coherence propagation.

**Physics**: See AGENTS.md § Resonance
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import RESONANCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
from .registry import register_operator


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