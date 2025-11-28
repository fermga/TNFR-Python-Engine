"""Trauma Containment with SHA - Protective Pause Protocol.

This example demonstrates SHA (Silence) in trauma therapy as a protective
containment mechanism that stabilizes patients during intense work.

Protocol: AL → EN → OZ → IL → SHA
- Emission: Activate traumatic memory
- Reception: Receive emerging emotion
- Dissonance: Access distress/conflict
- Coherence: Brief stabilization
- Silence: Protective containment (νf → 0, ΔNFR contained but not resolved)

Key insight: SHA does NOT resolve trauma - it creates a safe pause that prevents
overwhelm while maintaining access to the material for future processing.

References:
- van der Kolk, B. A. (2015). The Body Keeps the Score.
- See: docs/source/examples/SHA_CLINICAL_APPLICATIONS.md, Section 2
"""

from tnfr.operators.definitions import Emission, Reception, Dissonance, Coherence, Silence
from tnfr.structural import create_nfr, run_sequence


def main():
    """Execute trauma therapy protocol with SHA protective containment."""
    print("=" * 70)
    print("TRAUMA THERAPY - SHA Protective Containment Protocol")
    print("=" * 70)
    print("\nContext: PTSD patient accessing traumatic memory in session")
    print("Goal: Contain activation, prevent overwhelm, enable safe closure\n")
    
    # Create trauma memory node
    G, psyche = create_nfr("trauma_memory", epi=0.35, vf=1.00)
    
    print("BASELINE (pre-therapy state)")
    print("  Traumatic memory dormant, patient stable")
    print()
    
    # Execute protocol: AL → EN → OZ → IL → SHA
    print("EXECUTING PROTOCOL: AL → EN → OZ → IL → SHA")
    print("-" * 70)
    print()
    
    print("Step 1: EMISSION (AL) - Therapist guides memory access")
    print("  Therapist: 'Can you tell me about what happened that day?'")
    print("  Patient: Begins narrative, activation starts")
    print()
    
    print("Step 2: RECEPTION (EN) - Emotional experience received")
    print("  Therapist: Empathic witnessing, validating presence")
    print("  Patient: Fear, grief, anger surface - affect tolerance maintained")
    print()
    
    print("Step 3: DISSONANCE (OZ) - Intense distress emerges")
    print("  Patient: 'I can't... it's too much... I feel like I'm there again'")
    print("  Observation: Arousal spiking, approaching window of tolerance limit")
    print("  Effect: ΔNFR spikes (high reorganization pressure)")
    print()
    
    print("Step 4: COHERENCE (IL) - Brief stabilization")
    print("  Therapist: Brief grounding before containment")
    print("  Effect: System stabilizes momentarily")
    print()
    
    print("Step 5: SILENCE (SHA) - Protective containment")
    print("  Therapist: 'Let's pause right here. Notice your feet on the floor.'")
    print("  Intervention: Sustained grounding, present-moment awareness")
    print("  Effect: νf → 0 (pause processing), ΔNFR contained (not resolved)")
    print()
    
    # Execute the complete sequence
    run_sequence(G, psyche, [Emission(), Reception(), Dissonance(), Coherence(), Silence()])
    
    print("✓ Protocol complete - Patient stabilized, material accessible")
    print()
    
    # Critical distinction
    print("=" * 70)
    print("CRITICAL: WHAT SHA IS AND IS NOT")
    print("=" * 70)
    print("\n❌ SHA is NOT:")
    print("  • Suppression (patient remains aware of distress)")
    print("  • Avoidance (material stays accessible)")
    print("  • Resolution (trauma still requires deeper processing)")
    print("  • Dissociation (structure preserved, no fragmentation)")
    
    print("\n✓ SHA IS:")
    print("  • Stabilization tool for crisis moments")
    print("  • Safety mechanism during intense work")
    print("  • Bridge to safe session closure")
    print("  • Preparation for deeper processing (future sessions)")
    
    # Expected outcomes
    print("\n" + "=" * 70)
    print("EXPECTED THERAPEUTIC OUTCOMES")
    print("=" * 70)
    print("\nWithin-Session:")
    print("  • Patient reports 'holding' distress without overwhelm")
    print("  • Physiological arousal decreases while awareness remains")
    print("  • Safe session termination achieved")
    print("  • Therapeutic relationship strengthened")
    
    print("\nBetween-Sessions:")
    print("  • Reduced avoidance (patient knows pause is available)")
    print("  • Increased tolerance for exposure work")
    print("  • Less post-session dysregulation")
    print("  • Foundation for deeper processing (THOL/ZHIR next)")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: SHA AS PROTECTIVE CONTAINMENT")
    print("=" * 70)
    print("\nSHA contains dissonance without resolving it:")
    print("  1. Reduces reorganization (νf → 0) = pause processing")
    print("  2. Pressure remains (ΔNFR high) = material accessible")
    print("  3. Structure intact (EPI preserved) = no dissociation")
    print("\nThis creates therapeutic safety: patient can 'hold' intense")
    print("affect without fragmenting, enabling deeper work in future sessions.")
    
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
    print("  Section 2: Trauma Therapy (Containment Protocol)")
    print("\nScientific references:")
    print("  • van der Kolk (2015). The Body Keeps the Score.")
    print("  • Ogden & Fisher (2015). Sensorimotor Psychotherapy.")
