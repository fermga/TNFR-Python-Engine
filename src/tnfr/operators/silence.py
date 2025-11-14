"""TNFR Operator: Silence

Silence structural operator (SHA) - Preservation through structural pause.

**Physics**: See AGENTS.md § Silence
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import SILENCE
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


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