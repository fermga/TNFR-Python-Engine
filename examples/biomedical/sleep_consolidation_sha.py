"""Sleep & Memory Consolidation with SHA - Neuroscience Protocol.

This example demonstrates SHA (Silence) modeling sleep-dependent memory
consolidation, showing how structural pause enables learning retention.

Protocol: [Day] AL → IL → SHA
- Day learning: Emission + Coherence (pattern acquisition + SHA consolidation)
- Models sleep consolidation through structural pause

Key insight: SHA models deep slow-wave sleep where neuronal firing decreases
dramatically (νf → 0) while learned patterns (EPI) are preserved intact.

References:
- Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity. Neuron.
- See: docs/source/examples/SHA_CLINICAL_APPLICATIONS.md, Section 3
"""

from tnfr.operators.definitions import Emission, Coherence, Silence
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
    
    # === DAY: LEARNING + SLEEP CONSOLIDATION ===
    print("=" * 70)
    print("LEARNING → SLEEP CONSOLIDATION CYCLE")
    print("=" * 70)
    print()
    
    print("Protocol: AL → IL → SHA")
    print()
    
    print("Step 1: EMISSION (AL) - New information presented [DAY]")
    print("  Student encounters new concept, neural activation begins")
    print("  Synaptic integration, early LTP formation")
    print()
    
    print("Step 2: COHERENCE (IL) - Pattern stabilizes [DAY]")
    print("  Pattern coheres, initial memory trace formed")
    print("  Learning complete - ready for sleep consolidation")
    print()
    
    print("Step 3: SILENCE (SHA) - Sleep consolidation [NIGHT]")
    print("  Context: Deep slow-wave sleep (stages 3-4)")
    print("  Duration: 6-8 hours")
    print("  Neural correlate: δ waves (0.5-4 Hz)")
    print("  Effect: νf → 0 (minimal firing), EPI preserved")
    print("  Process: Synaptic consolidation without interference")
    print()
    
    # Execute the complete sequence
    run_sequence(G, neuron, [Emission(), Coherence(), Silence()])
    
    print("✓ Consolidation complete - Memory preserved through sleep")
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
    print("  3. Interference prevention = no disruption during consolidation")
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
