"""Canonical operator sequences and archetypal patterns from TNFR theory.

This module defines the 6 canonical archetypal sequences involving OZ (Dissonance)
as documented in "El pulso que nos atraviesa" (Table 2.5 - Tipología estructural glífica).

These sequences represent validated structural patterns that can be reused across
different domains and applications while maintaining TNFR coherence and grammar.

References
----------
"El pulso que nos atraviesa", Tabla 2.5: Tipología estructural glífica
Section 2.3.8: Ejemplos completos
Section 2.3.5: Escritura glífica avanzada (Macros glíficas)
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple

from ..types import Glyph
from .grammar import StructuralPattern

__all__ = [
    "CanonicalSequence",
    "CANONICAL_SEQUENCES",
    "BIFURCATED_BASE",
    "BIFURCATED_COLLAPSE",
    "THERAPEUTIC_PROTOCOL",
    "THEORY_SYSTEM",
    "FULL_DEPLOYMENT",
    "MOD_STABILIZER",
]


class CanonicalSequence(NamedTuple):
    """Canonical operator sequence with theoretical metadata.
    
    Represents a validated archetypal sequence from TNFR theory with
    structural pattern classification, use cases, and domain context.
    
    Attributes
    ----------
    name : str
        Unique identifier for the sequence (e.g., 'bifurcated_base')
    glyphs : List[Glyph]
        Ordered sequence of structural glyphs (AL, EN, IL, OZ, etc.)
    pattern_type : StructuralPattern
        Structural pattern classification from grammar
    description : str
        Detailed explanation of structural function
    use_cases : List[str]
        Concrete application scenarios
    domain : str
        Primary domain: 'general', 'biomedical', 'cognitive', 'social'
    references : str
        Theoretical grounding from TNFR documentation
    """
    
    name: str
    glyphs: List[Glyph]
    pattern_type: StructuralPattern
    description: str
    use_cases: List[str]
    domain: str
    references: str


# ============================================================================
# Bifurcated Patterns: OZ creates decision points
# ============================================================================

BIFURCATED_BASE = CanonicalSequence(
    name="bifurcated_base",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description=(
        "Disonancia estructural que genera umbral de bifurcación. "
        "El nodo puede reorganizarse (ZHIR) o colapsar a latencia (NUL). "
        "Este patrón representa la resolución creativa de la disonancia "
        "mediante mutación transformativa. "
        "Incluye EN → IL (reception→coherence) para validación gramática."
    ),
    use_cases=[
        "Intervención terapéutica ante bloqueos emocionales o cognitivos",
        "Análisis de crisis culturales o paradigmas en tensión",
        "Diseño de sistemas con respuesta adaptativa a perturbaciones",
        "Modelado de puntos de decisión en redes complejas",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Tabla 2.5, Sección 2.3.4 (Bifurcación)",
)

BIFURCATED_COLLAPSE = CanonicalSequence(
    name="bifurcated_collapse",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.NUL, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description=(
        "Path alternativo de bifurcación: disonancia lleva a colapso controlado (NUL) "
        "en lugar de mutación. Útil para reset estructural cuando la transformación "
        "no es viable. El nodo retorna a latencia preservando potencialidad. "
        "Incluye EN → IL (reception→coherence) para validación gramática."
    ),
    use_cases=[
        "Reset cognitivo tras sobrecarga informacional",
        "Desinversión organizacional estratégica",
        "Retorno a potencialidad tras exploración fallida",
        "Simplificación estructural ante complejidad insostenible",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Sección 2.3.3 (Bifurcación y mutación)",
)


# ============================================================================
# Therapeutic Protocol: Ritual de Reorganización
# ============================================================================

THERAPEUTIC_PROTOCOL = CanonicalSequence(
    name="therapeutic_protocol",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.UM, Glyph.RA, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description=(
        "Protocolo ritual o terapéutico completo: emisión simbólica (AL), "
        "recepción estabilizadora (EN), coherencia inicial (IL), acoplamiento "
        "relacional (UM), expansión energética (RA), disonancia creativa como "
        "confrontación (OZ), mutación del sujeto (ZHIR), estabilización de la "
        "nueva forma (IL), entrada en latencia resonante (SHA). "
        "Ciclo completo de transformación personal o colectiva."
    ),
    use_cases=[
        "Ceremonias de transformación personal o iniciación",
        "Sesiones de reestructuración terapéutica profunda",
        "Acompañamiento simbólico de procesos de cambio vital",
        "Rituales de sanación colectiva o comunitaria",
    ],
    domain="biomedical",
    references="El pulso que nos atraviesa, Ejemplo 3 (Sección 2.3.8)",
)


# ============================================================================
# Theory System: Construcción Epistemológica
# ============================================================================

THEORY_SYSTEM = CanonicalSequence(
    name="theory_system",
    glyphs=[Glyph.NAV, Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.THOL, Glyph.SHA],
    pattern_type=StructuralPattern.EDUCATIONAL,
    description=(
        "Sistema de ideas o teoría emergente: nodo mental nace (NAV), emite intuición "
        "inicial (AL), recibe información (EN), se estabiliza (IL), enfrenta disonancia "
        "conceptual o paradoja (OZ), muta hacia nuevo paradigma (ZHIR), se estabiliza en "
        "comprensión coherente (IL), se autoorganiza en sistema teórico (THOL), integra en "
        "conocimiento encarnado (SHA). Trayectoria completa de construcción epistemológica."
    ),
    use_cases=[
        "Diseño de marcos epistemológicos o paradigmas científicos",
        "Construcción de teorías coherentes en ciencias sociales",
        "Modelado de evolución conceptual en comunidades académicas",
        "Desarrollo de sistemas filosóficos o cosmovisiones",
    ],
    domain="cognitive",
    references="El pulso que nos atraviesa, Ejemplo 2 (Sección 2.3.8)",
)


# ============================================================================
# Full Deployment: Despliegue Total
# ============================================================================

FULL_DEPLOYMENT = CanonicalSequence(
    name="full_deployment",
    glyphs=[Glyph.NAV, Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.UM, Glyph.RA, Glyph.SHA],
    pattern_type=StructuralPattern.COMPLEX,
    description=(
        "Trayectoria completa de reorganización nodal: transición de estado (NAV), "
        "emisión iniciadora (AL), recepción estabilizadora (EN), coherencia inicial (IL), "
        "disonancia exploradora (OZ), mutación transformativa (ZHIR), estabilización "
        "coherente (IL), acoplamiento relacional (UM), propagación resonante (RA), "
        "cierre en latencia (SHA). Secuencia exhaustiva que cubre todas las fases "
        "de reorganización estructural."
    ),
    use_cases=[
        "Procesos de transformación organizacional completa",
        "Ciclos de innovación radical con múltiples fases",
        "Trayectorias de aprendizaje profundo y transformativo",
        "Reorganización sistémica de comunidades o ecosistemas",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Tabla 2.5 (Despliegue total)",
)


# ============================================================================
# MOD_STABILIZER: Macro Glífica Reutilizable
# ============================================================================

MOD_STABILIZER = CanonicalSequence(
    name="mod_stabilizer",
    glyphs=[Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL],
    pattern_type=StructuralPattern.EXPLORE,
    description=(
        "MOD_ESTABILIZADOR: macro glífica para transformación controlada. "
        "Recibe estado actual (EN), estabiliza (IL), introduce disonancia "
        "controlada (OZ), muta estructura (ZHIR), estabiliza nueva forma (IL). "
        "Reutilizable como subunidad modular dentro de secuencias más complejas. "
        "Representa el patrón mínimo de exploración-transformación-consolidación "
        "con validación gramática completa (EN → IL)."
    ),
    use_cases=[
        "Módulo de transformación segura para composición",
        "Componente reutilizable en secuencias complejas",
        "Patrón de resolución creativa encapsulado",
        "Bloque de construcción para T'HOL (autoorganización)",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Sección 2.3.5 (Macros glíficas)",
)


# ============================================================================
# Registry of All Canonical Sequences
# ============================================================================

CANONICAL_SEQUENCES: Dict[str, CanonicalSequence] = {
    seq.name: seq
    for seq in [
        BIFURCATED_BASE,
        BIFURCATED_COLLAPSE,
        THERAPEUTIC_PROTOCOL,
        THEORY_SYSTEM,
        FULL_DEPLOYMENT,
        MOD_STABILIZER,
    ]
}
"""Registry of all canonical operator sequences.

Maps sequence names to their full CanonicalSequence definitions. This registry
provides programmatic access to validated archetypal patterns from TNFR theory.

Examples
--------
>>> from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES
>>> seq = CANONICAL_SEQUENCES["bifurcated_base"]
>>> print(f"{seq.name}: {' → '.join(g.value for g in seq.glyphs)}")
bifurcated_base: AL → IL → OZ → ZHIR → SHA
"""
