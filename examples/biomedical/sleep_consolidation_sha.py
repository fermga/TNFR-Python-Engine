"""Sleep & Memory Consolidation with SHA - Neuroscience Protocol.

This example demonstrates SHA (Silence) modeling sleep-dependent memory
consolidation, showing how structural pause enables learning retention.

Protocol: [Day] AL → IL → SHA → [Night continues SHA] → [Next Day] NAV → AL
- Day learning: Emission + Coherence (pattern acquisition)
- Night sleep: Silence (νf → 0, EPI preserved - consolidation)
- Next day: Transition + Emission (memory recall)

Key insight: SHA models deep slow-wave sleep where neuronal firing decreases
dramatically (νf → 0) while learned patterns (EPI) are preserved intact.

References:
- Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity. Neuron.
- See: docs/source/examples/SHA_CLINICAL_APPLICATIONS.md, Section 3
"""

from tnfr.operators.definitions import Emission, Coherence, Silence, Transition
from tnfr.structural import create_nfr, run_sequence


def main():
    """Execute sleep-dependent memory consolidation protocol."""
    print("=" * 70)
    print("SLEEP & MEMORY CONSOLIDATION - SHA Neuroscience Protocol")
    print("=" * 70)
    print("\nContext: Modeling memory consolidation during sleep")
    print("Goal: Demonstrate pattern preservation through structural pause\n")
    
    # Create learning neuron node
    G, neuron = create_nfr("learning_neuron", epi=0.20, vf=1.20)
    
    # === DAY: ACTIVE LEARNING ===
    print("=" * 70)
    print("DAY - ACTIVE LEARNING PHASE")
    print("=" * 70)
    print()
    
    print("Protocol: AL → IL (Learning sequence)")
    print()
    
    print("Step 1: EMISSION (AL) - New information presented")
    print("  Student encounters new concept, neural activation begins")
    print()
    
    print("Step 2: COHERENCE (IL) - Pattern stabilizes")
    print("  Synaptic integration, early LTP formation")
    print("  Pattern begins to cohere, initial memory trace formed")
    print()
    
    run_sequence(G, neuron, [Emission(), Coherence()])
    
    print("✓ Learning complete - Pattern acquired")
    print()
    
    # === NIGHT: SLEEP CONSOLIDATION ===
    print("=" * 70)
    print("NIGHT - SLEEP CONSOLIDATION (SHA)")
    print("=" * 70)
    print()
    
    print("Step 3: SILENCE (SHA) - Deep sleep structural pause")
    print("  Context: Deep slow-wave sleep (stages 3-4)")
    print("  Duration: 6-8 hours (typical sleep period)")
    print("  Neural correlate: δ waves (0.5-4 Hz)")
    print("  Effect: νf → 0 (minimal firing), EPI preserved (pattern intact)")
    print("  Process: Synaptic consolidation without interference")
    print()
    
    run_sequence(G, neuron, [Silence()])
    
    print("✓ Sleep consolidation complete - Memory preserved")
    print()
    
    # === NEXT DAY: MEMORY RECALL ===
    print("=" * 70)
    print("NEXT DAY - MEMORY REACTIVATION")
    print("=" * 70)
    print()
    
    print("Protocol: NAV → AL (Wake and recall)")
    print()
    
    print("Step 4: TRANSITION (NAV) - Sleep → wake transition")
    print("  Process: Awakening, neuronal activity increases")
    print()
    
    print("Step 5: EMISSION (AL) - Memory recall initiated")
    print("  Context: Student attempts to recall learned material")
    print("  Effect: Pattern retrieved with high fidelity")
    print()
    
    run_sequence(G, neuron, [Transition(), Emission()])
    
    print("✓ Recall complete - Memory successfully retrieved")
    print()
    
    # Neuroscientific correlates
    print("=" * 70)
    print("NEUROSCIENTIFIC CORRELATES")
    print("=" * 70)
    print()
    print("TNFR Element → Neural Correlate → Measurement")
    print("-" * 70)
    print("SHA activation   → SWS onset           → δ waves (0.5-4 Hz)")
    print("νf → 0           → Reduced firing      → Single-unit recordings")
    print("EPI preservation → Synaptic maintain   → LTP stability")
    print("ΔNFR inactive    → Reduced interference → Memory stability tests")
    print("SHA duration     → Sleep stage duration → Polysomnography")
    
    # Research applications
    print()
    print("=" * 70)
    print("RESEARCH APPLICATIONS")
    print("=" * 70)
    print()
    print("Computational Neuroscience:")
    print("  • Model sleep-dependent learning")
    print("  • Predict optimal sleep timing for retention")
    print("  • Study interference effects on consolidation")
    print()
    print("Clinical Applications:")
    print("  • Sleep disorder impact on memory")
    print("  • Optimal study-sleep schedules")
    print("  • Aging and memory consolidation")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT: SHA AS MEMORY CONSOLIDATION")
    print("=" * 70)
    print("\nSHA models sleep's role in memory through:")
    print("  1. Minimal reorganization (νf → 0) = reduced neural activity")
    print("  2. Pattern preservation (EPI intact) = memory maintained")
    print("  3. Interference prevention (ΔNFR inactive) = no disruption")
    print("\nThis explains why sleep is essential for learning: structural")
    print("pause allows patterns to consolidate without interference.")
    
    print("\n" + "=" * 70)
    print("PROTOCOL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
    print("\n" + "=" * 70)
    print("REFERENCE")
    print("=" * 70)
    print("\nFor detailed protocol documentation, see:")
    print("  docs/source/examples/SHA_CLINICAL_APPLICATIONS.md")
    print("  Section 3: Sleep & Memory Consolidation")
    print("\nScientific references:")
    print("  • Tononi & Cirelli (2014). Sleep and the price of plasticity.")
    print("  • Rasch & Born (2013). About sleep's role in memory.")
