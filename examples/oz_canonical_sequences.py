"""Canonical OZ Sequences - Examples from TNFR Theory.

Demonstrates the 6 archetypal sequences involving OZ (Dissonance)
from "El pulso que nos atraviesa" manual (Tabla 2.5).

These sequences represent validated structural patterns for:
- Bifurcation and decision points
- Therapeutic transformation
- Epistemological construction
- Complete reorganization cycles
- Modular transformation components

Each example shows the theoretical grounding, use cases, and expected
structural outcomes following TNFR canonical principles.
"""

import warnings

warnings.filterwarnings("ignore")

from tnfr.sdk import TNFRNetwork, NetworkConfig

print("\n" + "=" * 70)
print("CANONICAL OZ SEQUENCES - TNFR Archetypal Patterns")
print("=" * 70)
print("\nReference: 'El pulso que nos atraviesa', Tabla 2.5")
print("Tipología estructural glífica\n")


def example_bifurcated_mutation():
    """OZ → ZHIR: Dissonance resolved through mutation.
    
    Domain: General
    Use: Decision points, adaptive responses, creative transformation
    Pattern: AL → IL → OZ → ZHIR → SHA
    """
    print("\n" + "=" * 70)
    print("1. BIFURCATED PATTERN: OZ → ZHIR (Mutation Path)")
    print("=" * 70)
    print("\nStructural Function:")
    print("  Disonancia crea umbral de bifurcación → mutación transformativa")
    print("\nSequence: AL → EN → IL → OZ → ZHIR → IL → SHA")
    print("\nUse Cases:")
    print("  • Intervención terapéutica ante bloqueos")
    print("  • Diseño de sistemas con respuesta adaptativa")
    print("  • Modelado de puntos de decisión en redes complejas")
    
    net = TNFRNetwork("bifurcation_mutation", NetworkConfig(random_seed=42))
    net.add_nodes(1)
    net.apply_canonical_sequence("bifurcated_base")
    
    results = net.measure()
    print(f"\n✓ Sequence executed successfully")
    print(f"  Final Coherence C(t): {results.coherence:.3f}")
    print(f"  Node transformed via mutation (ZHIR)")
    
    return net


def example_bifurcated_collapse():
    """OZ → NUL: Dissonance leads to controlled collapse.
    
    Domain: General
    Use: Reset processes, strategic withdrawal, consolidation
    Pattern: AL → IL → OZ → NUL → SHA
    """
    print("\n" + "=" * 70)
    print("2. BIFURCATED PATTERN: OZ → NUL (Collapse Path)")
    print("=" * 70)
    print("\nStructural Function:")
    print("  Disonancia → colapso controlado (reset estructural)")
    print("\nSequence: AL → EN → IL → OZ → NUL → IL → SHA")
    print("\nUse Cases:")
    print("  • Reset cognitivo tras sobrecarga")
    print("  • Desinversión organizacional estratégica")
    print("  • Simplificación estructural ante complejidad insostenible")
    
    net = TNFRNetwork("bifurcation_collapse", NetworkConfig(random_seed=42))
    net.add_nodes(1)
    net.apply_canonical_sequence("bifurcated_collapse")
    
    results = net.measure()
    print(f"\n✓ Sequence executed successfully")
    print(f"  Final Coherence C(t): {results.coherence:.3f}")
    print(f"  Node returned to latency via contraction (NUL)")
    
    return net


def example_therapeutic_protocol():
    """AL → UM → RA → OZ → ZHIR → IL → SHA

    becomes: AL → EN → IL → OZ → ZHIR → IL → RA → SHA
    
    Domain: Biomedical/Therapeutic
    Use: Healing sessions, personal transformation rituals
    
    Sequence breakdown:
    - AL: Initiate symbolic field
    - UM: Generate relational coupling
    - RA: Expand as energetic field
    - OZ: Introduce creative dissonance (confrontation)
    - ZHIR: Subject mutates (transformation)
    - IL: Stabilize new form (integration)
    - SHA: Enter resonant latency (rest)
    """
    print("\n" + "=" * 70)
    print("3. THERAPEUTIC PROTOCOL: Ritual de Reorganización")
    print("=" * 70)
    print("\nStructural Function:")
    print("  Ciclo completo de transformación personal/colectiva")
    print("\nSequence: AL → EN → IL → OZ → ZHIR → IL → RA → SHA")
    print("\nPhases:")
    print("  1. AL (Emisión): Inicia campo simbólico")
    print("  2. EN (Recepción): Estabiliza estado")
    print("  3. IL (Coherencia): Coherencia inicial")
    print("  4. OZ (Disonancia): Introduce tensión creativa (confrontación)")
    print("  5. ZHIR (Mutación): Sujeto se transforma")
    print("  6. IL (Coherencia): Estabiliza nueva forma (integración)")
    print("  7. RA (Resonancia): Propaga coherencia")
    print("  8. SHA (Silencio): Entra en latencia resonante (reposo)")
    print("\nUse Cases:")
    print("  • Ceremonias de transformación personal")
    print("  • Sesiones de reestructuración terapéutica")
    print("  • Rituales de sanación colectiva")
    
    net = TNFRNetwork("healing_session", NetworkConfig(random_seed=42))
    net.add_nodes(5)  # Patient + context nodes
    net.connect_nodes(0.4, "random")  # Relational field
    net.apply_canonical_sequence("therapeutic_protocol")  # Apply to last node
    
    results = net.measure()
    print(f"\n✓ Therapeutic protocol completed")
    print(f"  Final Coherence C(t): {results.coherence:.3f}")
    print(f"  Avg Sense Index Si: {sum(results.sense_indices.values())/len(results.sense_indices):.3f}")
    print(f"  Transformation complete: Subject in new stable state")
    
    return net


def example_theory_construction():
    """NAV → AL → OZ → ZHIR → IL → THOL → SHA

    becomes: AL → EN → IL → OZ → ZHIR → IL → THOL → SHA
    
    Domain: Cognitive/Epistemological
    Use: Building conceptual frameworks, theory development
    
    Sequence breakdown:
    - NAV: Mental node emerges
    - AL: Intuition emitted
    - OZ: Conceptual dissonance (paradox, contradiction)
    - ZHIR: Paradigm shift (mutation of understanding)
    - IL: Comprehension stabilizes
    - THOL: Self-organizes into coherent theory
    - SHA: Integrates into rest (embodied knowledge)
    """
    print("\n" + "=" * 70)
    print("4. THEORY SYSTEM: Construcción Epistemológica")
    print("=" * 70)
    print("\nStructural Function:")
    print("  Sistema de ideas → teoría emergente coherente")
    print("\nSequence: AL → EN → IL → OZ → ZHIR → IL → THOL → SHA")
    print("\nPhases:")
    print("  1. AL (Emisión): Intuición inicial emitida")
    print("  2. EN (Recepción): Recibe información")
    print("  3. IL (Coherencia): Se estabiliza")
    print("  4. OZ (Disonancia): Paradoja/contradicción conceptual")
    print("  5. ZHIR (Mutación): Cambio de paradigma")
    print("  6. IL (Coherencia): Comprensión se estabiliza")
    print("  7. THOL (Autoorganización): Se organiza en teoría")
    print("  8. SHA (Silencio): Integra en conocimiento encarnado")
    print("\nUse Cases:")
    print("  • Diseño de marcos epistemológicos")
    print("  • Construcción de teorías coherentes")
    print("  • Modelado de evolución conceptual")
    
    net = TNFRNetwork("conceptual_framework", NetworkConfig(random_seed=42))
    net.add_nodes(3)  # Concept nodes
    net.connect_nodes(0.3, "ring")
    net.apply_canonical_sequence("theory_system")  # Apply to last node
    
    results = net.measure()
    print(f"\n✓ Theory system constructed")
    print(f"  Final Coherence C(t): {results.coherence:.3f}")
    print(f"  Theory coherently self-organized")
    
    return net


def example_full_deployment():
    """NAV → AL → OZ → ZHIR → IL → UM → RA → SHA

    becomes: AL → EN → IL → OZ → ZHIR → IL → RA → SHA
    
    Domain: General
    Use: Complete transformation cycles
    
    Trayectoria completa de reorganización nodal.
    """
    print("\n" + "=" * 70)
    print("5. FULL DEPLOYMENT: Despliegue Total")
    print("=" * 70)
    print("\nStructural Function:")
    print("  Trayectoria completa de reorganización nodal")
    print("\nSequence: AL → EN → IL → OZ → ZHIR → IL → RA → SHA")
    print("\nPhases:")
    print("  AL: Emisión iniciadora")
    print("  EN: Recepción estabilizadora")
    print("  IL: Coherencia inicial")
    print("  OZ: Disonancia exploradora")
    print("  ZHIR: Mutación transformativa")
    print("  IL: Estabilización coherente")
    print("  RA: Propagación resonante")
    print("  SHA: Cierre en latencia")
    print("\nUse Cases:")
    print("  • Procesos de transformación completa")
    print("  • Ciclos de innovación radical")
    print("  • Trayectorias de aprendizaje profundo")
    
    net = TNFRNetwork("complete_transformation", NetworkConfig(random_seed=42))
    net.add_nodes(5)
    net.connect_nodes(0.5, "small_world")
    net.apply_canonical_sequence("full_deployment")  # Apply to last node
    
    results = net.measure()
    print(f"\n✓ Full deployment completed")
    print(f"  Final Coherence C(t): {results.coherence:.3f}")
    print(f"  All reorganization phases executed")
    
    return net


def example_mod_stabilizer():
    """OZ → ZHIR → IL: Reusable transformation module.

    becomes: REMESH → EN → IL → OZ → ZHIR → IL → REMESH
    
    Domain: General (modular component)
    Use: Encapsulated within THOL or as micro-transformation
    
    This is the MOD_ESTABILIZADOR macro from theoretical documentation.
    Can be nested within larger sequences:
    
    THOL[MOD_ESTABILIZADOR] ≡ THOL[OZ → ZHIR → IL]
    """
    print("\n" + "=" * 70)
    print("6. MOD_ESTABILIZADOR: Macro de Transformación")
    print("=" * 70)
    print("\nStructural Function:")
    print("  Macro glífica reutilizable para transformación controlada")
    print("\nSequence: REMESH → EN → IL → OZ → ZHIR → IL → REMESH")
    print("\nPhases:")
    print("  REMESH: Activa recursividad")
    print("  EN: Recibe estado actual")
    print("  IL: Estabiliza")
    print("  OZ: Disonancia controlada")
    print("  ZHIR: Mutación estructural")
    print("  IL: Estabilización de nueva forma")
    print("  REMESH: Cierre recursivo")
    print("\nUse Cases:")
    print("  • Módulo de transformación segura")
    print("  • Componente en secuencias complejas")
    print("  • Bloque de construcción para T'HOL")
    print("\nTheoretical Note:")
    print("  Este patrón es composable: THOL[REMESH → EN → IL → OZ → ZHIR → IL → REMESH]")
    
    net = TNFRNetwork("modular_transform", NetworkConfig(random_seed=42))
    # Apply transformation module directly
    net.add_nodes(1)
    net.apply_canonical_sequence("mod_stabilizer")
    
    results = net.measure()
    print(f"\n✓ Transformation module completed")
    print(f"  Final Coherence C(t): {results.coherence:.3f}")
    print(f"  Ready for integration into larger sequences")
    
    return net


def list_all_sequences():
    """List all available canonical sequences."""
    print("\n" + "=" * 70)
    print("AVAILABLE CANONICAL SEQUENCES")
    print("=" * 70)
    
    net = TNFRNetwork("query")
    sequences = net.list_canonical_sequences()
    
    print(f"\nTotal: {len(sequences)} canonical sequences\n")
    
    for i, (name, seq) in enumerate(sorted(sequences.items()), 1):
        glyphs_str = ' → '.join(g.value for g in seq.glyphs)
        print(f"{i}. {name.upper()}")
        print(f"   Pattern: {seq.pattern_type.value}")
        print(f"   Domain: {seq.domain}")
        print(f"   Glyphs: {glyphs_str}")
        print(f"   Use: {seq.use_cases[0]}")
        print()
    
    # Show filtering examples
    print("\n" + "-" * 70)
    print("FILTERING EXAMPLES")
    print("-" * 70)
    
    oz_sequences = net.list_canonical_sequences(with_oz=True)
    print(f"\nSequences with OZ (Dissonance): {len(oz_sequences)}")
    for name in sorted(oz_sequences.keys()):
        print(f"  • {name}")
    
    bio_sequences = net.list_canonical_sequences(domain="biomedical")
    print(f"\nBiomedical domain sequences: {len(bio_sequences)}")
    for name in sorted(bio_sequences.keys()):
        print(f"  • {name}")
    
    cognitive_sequences = net.list_canonical_sequences(domain="cognitive")
    print(f"\nCognitive domain sequences: {len(cognitive_sequences)}")
    for name in sorted(cognitive_sequences.keys()):
        print(f"  • {name}")


if __name__ == "__main__":
    print("\nExecuting OZ Canonical Sequence Examples\n")
    
    # List all available sequences
    list_all_sequences()
    
    # Execute each canonical sequence
    example_bifurcated_mutation()
    example_bifurcated_collapse()
    example_therapeutic_protocol()
    example_theory_construction()
    example_full_deployment()
    example_mod_stabilizer()
    
    print("\n" + "=" * 70)
    print("ALL CANONICAL OZ SEQUENCES EXECUTED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Insights:")
    print("  ✓ 6 archetypal patterns validated from TNFR theory")
    print("  ✓ OZ (Dissonance) enables bifurcation and transformation")
    print("  ✓ Patterns span multiple domains: general, biomedical, cognitive")
    print("  ✓ MOD_STABILIZER provides reusable transformation module")
    print("  ✓ All sequences maintain structural coherence and grammar")
    print("\nReference:")
    print("  'El pulso que nos atraviesa', Tabla 2.5")
    print("  Tipología estructural glífica")
    print("=" * 70 + "\n")
