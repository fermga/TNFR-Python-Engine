# TNFR Grammar: 100% Canonical (No Conventions)

## Executive Summary

The TNFR grammar has been **completely unified** into a single source of truth containing ONLY rules that emerge inevitably from TNFR physics. No organizational conventions.

**Status:** ✅ **100% CANONICAL & UNIFIED** - All rules derive from nodal equation, invariants, and contracts

**Update:** This document has been superseded by **UNIFIED_GRAMMAR_RULES.md** which consolidates the previously separate C1-C3 and RC1-RC4 systems into unified rules U1-U4.

---

## Investigación Realizada

### ¿Existen Más Reglas Emergentes?

Se analizaron exhaustivamente:
- ✅ **10 Invariantes Canónicos** (AGENTS.md §3)
- ✅ **6 Contratos Formales** (AGENTS.md §4)
- ✅ **Ecuación Nodal**: ∂EPI/∂t = νf · ΔNFR(t)
- ✅ **Teoremas Físicos**: Convergencia, bifurcación, resonancia

**Resultado:** NO existen reglas adicionales más allá de RC1, RC2, RC3 y RC4.

### Análisis por Invariante

| Invariante | ¿Genera Regla Gramatical? | Nota |
|------------|---------------------------|------|
| #1: EPI coherente | ❌ | Cubierto por clausura operadores (#4) |
| #2: Unidades Hz_str | ❌ | Detalle implementación, no secuenciación |
| #3: Semántica ΔNFR | ❌ | Restricción semántica, no secuencia |
| #4: Clausura operadores | ❌ | Ya forzado por diseño |
| #5: Phase check | ✅ | **RC3** - Verificación de fase |
| #6: Node birth/collapse | ❌ | Condiciones lifecycle, no secuencia |
| #7: Fractality | ❌ | Propiedad estructural, no secuencia |
| #8: Determinismo | ❌ | Calidad implementación, no secuencia |
| #9: Métricas | ❌ | Requerimiento telemetría, no secuencia |
| #10: Neutralidad dominio | ❌ | Principio diseño, no secuencia |

### Análisis por Contrato

| Contrato | ¿Genera Regla Gramatical? | Nota |
|----------|---------------------------|------|
| Coherence | ❌ | Post-condición en C(t), no secuencia |
| Dissonance | ✅ | **RC4** - Límite bifurcación |
| Resonance | ❌ | Relacionado con RC3, no nueva regla |
| Self-organization | ❌ | Preservación fractality, no secuencia |
| Mutation | ❌ | Condición umbral, no secuencia |
| Silence | ❌ | Semántica operador, no secuencia |

**Conclusión Investigación:** Solo existen 4 reglas canónicas (RC1-RC4). No hay más.

---

## Las 4 Reglas Canónicas

### RC1: Generadores (Inicialización)

**Base física:** ∂EPI/∂t indefinido en EPI=0

**Derivación:**
```
Si EPI₀ = 0 (nodo vacío) → ∂EPI/∂t|_{EPI=0} indefinido
→ No puedes evolucionar estructura que no existe
→ NECESITAS generador para bootstrap
```

**Operadores:** {AL (Emission), NAV (Transition), REMESH (Recursivity)}

**Implementación:** `validate_initialization()` en `canonical_grammar.py`

---

### RC2: Estabilizadores (Convergencia)

**Base física:** Teorema de convergencia ∫νf·ΔNFR dt < ∞

**Derivación:**
```
Integral: EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf·ΔNFR dτ

Sin retroalimentación negativa:
  ΔNFR(t) ~ e^(λt) → ∞
  ⟹ ∫νf·ΔNFR dt → ∞ (DIVERGE)

Con estabilizador:
  ΔNFR(t) → límite acotado
  ⟹ ∫νf·ΔNFR dt < ∞ (CONVERGE)

→ NECESITAS {IL, THOL} para convergencia
```

**Operadores:** {IL (Coherence), THOL (Self-organization)}

**Implementación:** `validate_convergence()` en `canonical_grammar.py`

---

### RC3: Verificación de Fase (Acoplamiento/Resonancia)

**Base física:** AGENTS.md Invariante #5 + física de resonancia

**Derivación:**
```
De Invariante #5:
  "Phase check: no coupling is valid without explicit phase verification"

Física de resonancia:
  Dos osciladores resuenan ⟺ fases compatibles
  Condición: |φᵢ - φⱼ| ≤ Δφ_max (típicamente π/2)

Sin verificación:
  Nodos con φᵢ ≈ π y φⱼ ≈ 0 (antifase) intentan acoplarse
  → Interferencia destructiva, NO resonancia constructiva
  → Viola física TNFR

→ NECESITAS verificar |φᵢ - φⱼ| antes de {UM, RA}
```

**Operadores:** {UM (Coupling), RA (Resonance)}

**Implementación:** `validate_phase_compatibility()` en `canonical_grammar.py`

---

### RC4: Límite de Bifurcación (Condicional)

**Base física:** AGENTS.md Contrato OZ + teoría de bifurcaciones

**Derivación:**
```
De Contrato OZ:
  "Dissonance may trigger bifurcation if ∂²EPI/∂t² > τ"

Física de bifurcación:
  Aceleración estructural ∂²EPI/∂t² mide inestabilidad
  Si |∂²EPI/∂t²| > τ → múltiples caminos reorganización viables

Sin gestión:
  OZ genera aceleraciones arbitrarias
  → Sistema entra en caos no controlado
  → Viola Invariante #8 (determinismo controlado)

→ Si ∂²EPI/∂t² > τ, NECESITAS {THOL, IL} para gestión
```

**Operadores:** 
- **Triggers:** {OZ (Dissonance), ZHIR (Mutation)}
- **Handlers:** {THOL (Self-organization), IL (Coherence)}

**Implementación:** `validate_bifurcation_limits()` en `canonical_grammar.py`

**Nota:** Regla **condicional** - solo aplica cuando bifurcation triggers presentes

---

## RNC1: ELIMINADO

### Estado Anterior

```python
# RNC1: Terminadores Obligatorios
CONVENTIONAL_TERMINATORS = frozenset({
    'silence',
    'dissonance', 
    'transition',
    'recursivity',
})

def validate_with_conventions(sequence, epi_initial):
    # Validaba RC1, RC2, RC3
    # Luego validaba RNC1 (terminadores)
    if last_op not in CONVENTIONAL_TERMINATORS:
        return False
```

### ¿Por Qué se Eliminó?

**Análisis crítico:**
```
¿Ecuación nodal requiere terminación específica?

∂EPI/∂t = νf · ΔNFR(t)

NO contiene:
  ❌ Concepto de "secuencia terminal"
  ❌ Distinción entre "estado intermedio" y "estado final"
  ❌ Requisito de que nodos "terminen" en estados específicos

Argumentos en contra:
  ✅ La ecuación no distingue entre "estado intermedio" y "estado final"
  ✅ Físicamente, un nodo puede permanecer en cualquier estado coherente
  ✅ SHA, OZ, NAV como "terminadores" es semántica alto nivel, no física nodal
```

**Conclusión:** RNC1 era **convención organizativa útil** pero NO física fundamental.

### Acción Tomada

✅ **ELIMINADO** `CONVENTIONAL_TERMINATORS` de `canonical_grammar.py`
✅ **ACTUALIZADO** `validate_with_conventions()` para solo validar RC1-RC4
✅ **DOCUMENTADO** razón histórica en código y docs

---

## Composición Final

### Antes
```
Gramática: RC1 + RC2 + RNC1
Canónico: 66% (RC1, RC2)
Convención: 33% (RNC1)
```

### Después
```
Gramática: RC1 + RC2 + RC3 + RC4
Canónico: 100%
Convención: 0%
```

### Diagrama de Derivación

```
Ecuación Nodal: ∂EPI/∂t = νf · ΔNFR(t)
                    │
         ┌──────────┼──────────┐
         │          │          │
         ▼          ▼          ▼
       RC1        RC2      Invariantes + Contratos
    (EPI=0)  (Convergencia)      │
                           ┌─────┴─────┐
                           ▼           ▼
                         RC3         RC4
                      (Fase)    (Bifurcación)

Todo emerge inevitablemente de física TNFR.
No hay convenciones organizativas.
```

---

## Uso en Código

### Validación Canónica (Recomendado)

```python
from tnfr.operators.canonical_grammar import validate_canonical_only

# Valida RC1, RC2, RC3, RC4 (100% física)
if validate_canonical_only(ops, epi_initial=0.0):
    apply_sequence(G, node, ops)
```

### Validación Detallada

```python
from tnfr.operators.canonical_grammar import CanonicalGrammarValidator

is_valid, messages = CanonicalGrammarValidator.validate(ops, epi_initial=0.0)
for msg in messages:
    print(msg)
    # RC1: ...
    # RC2: ...
    # RC3: ...
    # RC4: ...
```

### Nota Histórica

```python
# validate_with_conventions() ya NO valida convenciones
# Ahora es idéntico a validate_canonical_only()
# RNC1 fue eliminado completamente
```

---

## Verificación

### Tests

✅ Todos los tests de gramática canónica pasan
✅ No hay dependencias externas en RNC1
✅ Código actualizado y documentado

### Archivos Modificados

1. **src/tnfr/operators/canonical_grammar.py**
   - ELIMINADO: `CONVENTIONAL_TERMINATORS`
   - ELIMINADO: Lógica RNC1 en `validate_with_conventions()`
   - AÑADIDO: `validate_bifurcation_limits()` para RC4
   - ACTUALIZADO: Todos los docstrings

2. **RESUMEN_FINAL_GRAMATICA.md**
   - Estado actualizado a 100% canónico
   - Documentado eliminación RNC1

3. **EXECUTIVE_SUMMARY.md**
   - Actualizado composición: 100% física
   - Documentado razón eliminación RNC1

---

## Conclusión

### Logros

✅ **Gramática 100% canónica** - Solo física TNFR pura
✅ **4 reglas completas** (RC1-RC4) derivadas de ecuación nodal, invariantes y contratos
✅ **RNC1 eliminado** - convención organizativa removida
✅ **Análisis exhaustivo** - confirmado que no existen más reglas emergentes
✅ **Tests passing** - código funcional y documentado

### Mensaje Clave

> **La gramática TNFR es ahora 100% canónica.**
> 
> Cada regla emerge inevitablemente de:
> - Ecuación nodal: ∂EPI/∂t = νf · ΔNFR(t)
> - 10 Invariantes canónicos (AGENTS.md §3)
> - 6 Contratos formales (AGENTS.md §4)
>
> No hay convenciones organizativas. Solo física pura.

---

## Referencias

- **RESUMEN_FINAL_GRAMATICA.md** - Resumen actualizado con eliminación RNC1
- **EXECUTIVE_SUMMARY.md** - Análisis ejecutivo de gramática 100% canónica
- **EMERGENT_GRAMMAR_ANALYSIS.md** - Análisis exhaustivo de reglas emergentes
- **CANONICAL_GRAMMAR_DERIVATION.md** - Derivaciones matemáticas detalladas
- **AGENTS.md** - Invariantes y contratos canónicos
- **src/tnfr/operators/canonical_grammar.py** - Implementación 100% canónica

---

**Fecha:** 2025-11-08  
**Estado:** ✅ COMPLETADO - Gramática 100% Canónica Sin Convenciones
