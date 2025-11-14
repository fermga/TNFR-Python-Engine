"""TNFR Operator: Contraction

Contraction structural operator (NUL) - Structural concentration and densification.

**Physics**: See AGENTS.md § Contraction
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import CONTRACTION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


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