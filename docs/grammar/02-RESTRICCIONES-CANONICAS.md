# Restricciones Canónicas de la Gramática TNFR (U1-U4)

## 🎯 Propósito

Este documento presenta las **cuatro restricciones canónicas** que gobiernan la composición de operadores estructurales en TNFR. Cada restricción emerge **inevitablemente** de la ecuación nodal, invariantes canónicas, y contratos formales—no son convenciones organizacionales.

**Audiencia:** Desarrolladores implementando validación, contribuidores avanzados  
**Prerequisitos:** [01-CONCEPTOS-FUNDAMENTALES.md](01-CONCEPTOS-FUNDAMENTALES.md)  
**Tiempo de lectura:** 45-60 minutos

---

## 📐 Las Cuatro Restricciones

```
┌─────────────────────────────────────────────────────────────────┐
│ Unified TNFR Grammar: Four Canonical Constraints               │
├─────────────────────────────────────────────────────────────────┤
│ U1: STRUCTURAL INITIATION & CLOSURE                             │
│     U1a: Start with generators {AL, NAV, REMESH}               │
│     U1b: End with closures {SHA, NAV, REMESH, OZ}              │
│     Basis: ∂EPI/∂t undefined at EPI=0, sequences need closure  │
│                                                                 │
│ U2: CONVERGENCE & BOUNDEDNESS                                   │
│     If destabilizers {OZ, ZHIR, VAL}                           │
│     Then include stabilizers {IL, THOL}                        │
│     Basis: ∫νf·ΔNFR dt must converge                           │
│                                                                 │
│ U3: RESONANT COUPLING                                           │
│     If coupling/resonance {UM, RA}                             │
│     Then verify phase |φᵢ - φⱼ| ≤ Δφ_max                       │
│     Basis: Invariant #5 + resonance physics                    │
│                                                                 │
│ U4: BIFURCATION DYNAMICS                                        │
│     U4a: If triggers {OZ, ZHIR}                                │
│          Then include handlers {THOL, IL}                      │
│     U4b: If transformers {ZHIR, THOL}                          │
│          Then recent destabilizer (~3 ops)                     │
│          Additionally ZHIR needs prior IL                      │
│     Basis: Contract OZ + bifurcation theory                    │
└─────────────────────────────────────────────────────────────────┘

All rules emerge inevitably from:
  ∂EPI/∂t = νf · ΔNFR(t) + Invariants + Contracts
```

---

## U1: STRUCTURAL INITIATION & CLOSURE

### Física de Base

**Principio:** Las secuencias son segmentos temporales acotados en el espacio estructural.

**Analogía:** Potenciales de acción en física de ondas
- Emisión electromagnética: fuente → propagación → absorción
- Impulso neural: despolarización → transmisión → repolarización
- Onda sonora: excitación → vibración → amortiguamiento

---

### U1a: Iniciación (Generators)

#### Declaración

**Cuando EPI = 0, la secuencia DEBE comenzar con generator**

**Generators:** {AL (Emission), NAV (Transition), REMESH (Recursivity)}

#### Derivación Física

**Desde la ecuación nodal:**

```
∂EPI/∂t = νf · ΔNFR(t)

En EPI = 0 (estado nulo):
  ΔNFR(0) = f(EPI, topología, fase) donde EPI = 0
  → ΔNFR(0) indefinido o nulo
  → ∂EPI/∂t|_{EPI=0} = νf · 0 = 0 OR indefinido

Conclusión: Sistema NO PUEDE evolucionar desde EPI=0 sin generator
```

**Necesidad matemática:**
- Como ecuación de onda: no hay propagación sin fuente
- Como termodinámica: no hay flujo sin diferencia de temperatura
- Como mecánica estructural: no hay deformación sin geometría inicial

#### ¿Por qué estos generators?

**1. Emission (AL) 🎵**
- **Física:** Crea EPI desde vacío vía emisión resonante
- **Efecto:** ∂EPI/∂t > 0, incrementa νf
- **Capacidad:** Generación desde estado nulo absoluto

```python
# Válido: Iniciar desde EPI=0 con Emission
from tnfr.operators.definitions import Emission, Coherence, Silence

sequence = [Emission(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Pasa U1a
```

**2. Transition (NAV) ➡️**
- **Física:** Activa EPI latente/dormante mediante cambio de régimen
- **Efecto:** Trayectoria controlada en espacio estructural
- **Capacidad:** Activación de estructura existente pero inactiva

```python
# Válido: NAV activa EPI dormido
sequence = [Transition(), Reception(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Pasa U1a
```

**3. Recursivity (REMESH) 🔄**
- **Física:** Eco de estructura a través de escalas (fractality operacional)
- **Efecto:** EPI(t) referencia EPI(t-τ), operadores anidados
- **Capacidad:** Generación multi-escala desde memoria estructural

```python
# Válido: REMESH propaga estructura existente
sequence = [Recursivity(), Coupling(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Pasa U1a
```

#### Canonicidad

**Nivel:** **ABSOLUTE** (Necesidad matemática)

**Imposibilidad:** No se puede evolucionar desde EPI=0 sin generación

**Traceabilidad:**
- TNFR.pdf § 2.1 (Ecuación Nodal) → Consecuencia matemática directa
- AGENTS.md Invariant #1 → EPI cambia solo vía operadores
- Código: `src/tnfr/operators/grammar.py::validate_u1a_initiation()`

#### Anti-Patrones

```python
# ✗ INVÁLIDO: Sin generator cuando EPI=0
sequence = [Reception(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)
# ValueError: U1a violation: Need generator when EPI=0

# ✓ VÁLIDO: Con generator
sequence = [Emission(), Reception(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# ✓ VÁLIDO: EPI>0 no necesita generator
sequence = [Reception(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=1.0)  # ✓
```

---

### U1b: Clausura (Closures)

#### Declaración

**Toda secuencia DEBE terminar con closure**

**Closures:** {SHA (Silence), NAV (Transition), REMESH (Recursivity), OZ (Dissonance)}

#### Derivación Física

**Desde física de ondas:**

```
Secuencias como potenciales de acción temporales:
  Como pulsos electromagnéticos: deben tener fuente Y terminación
  Como spikes neuronales: deben tener despolarización Y repolarización
  Como ondas sonoras: deben tener emisión Y absorción/decaimiento

Requerimiento físico:
  Segmentos temporales acotados necesitan endpoints coherentes
  → Inicio: Generator crea perturbación inicial
  → Fin: Closure absorbe/estabiliza estado final
```

**Analogía con física clásica:**
- **Electromagnética:** Toda emisión necesita absorción (conservación de energía)
- **Mecánica:** Todo impulso de fuerza necesita amortiguamiento (estabilidad)
- **Termodinámica:** Todo proceso necesita endpoint de equilibrio (2da ley)

#### ¿Por qué estos closures?

**1. Silence (SHA) 🔇**
- **Física:** Congela evolución temporalmente
- **Efecto:** νf → 0, EPI sin cambios
- **Tipo:** **Terminal closure** - finalización definitiva

```python
# Clausura terminal: SHA congela sistema
sequence = [Emission(), Coherence(), Silence()]
# EPI queda congelado, listo para nueva secuencia
```

**2. Transition (NAV) ➡️**
- **Física:** Cambio de régimen, activa EPI latente
- **Efecto:** Trayectoria controlada hacia nuevo atractor
- **Tipo:** **Handoff closure** - transferencia a siguiente régimen

```python
# Clausura de transferencia: NAV pasa a siguiente fase
sequence = [Emission(), Coherence(), Transition()]
# Sistema transferido a nuevo estado, continuidad garantizada
```

**3. Recursivity (REMESH) 🔄**
- **Física:** Eco de estructura a través de escalas
- **Efecto:** Distribución multi-escala
- **Tipo:** **Recursive closure** - cierre distribuido

```python
# Clausura recursiva: REMESH distribuye coherencia
sequence = [Emission(), SelfOrganization(), Recursivity()]
# Estructura distribuida en sub-EPIs, coherencia preservada
```

**4. Dissonance (OZ) ⚡**
- **Física:** Inestabilidad controlada
- **Efecto:** Aumenta |ΔNFR|, preserva activación
- **Tipo:** **Intentional closure** - tensión preservada para siguiente ciclo

```python
# Clausura intencional: OZ preserva tensión
sequence = [Emission(), Coherence(), Dissonance()]
# Sistema queda activado, listo para siguiente transformación
```

#### Canonicidad

**Nivel:** **STRONG** (Requerimiento físico)

**Violación produce:** Secuencias sin endpoint coherente, riesgo de fragmentación

**Traceabilidad:**
- Física de ondas + Dinámica estructural TNFR → Secuencias necesitan endpoints
- AGENTS.md Invariant #4 → Composición de operadores debe preservar validez
- Código: `src/tnfr/operators/grammar.py::validate_u1b_closure()`

#### Anti-Patrones

```python
# ✗ INVÁLIDO: Sin closure al final
sequence = [Emission(), Coherence(), Reception()]
validate_grammar(sequence, epi_initial=0.0)
# ValueError: U1b violation: Sequence must end with closure

# ✓ VÁLIDO: Con closure
sequence = [Emission(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓
```

---

## U2: CONVERGENCE & BOUNDEDNESS

### Física de Base

**Principio:** La integral ∫νf·ΔNFR dt debe converger para evolución acotada.

**Derivación:** Desde ecuación nodal integrada

---

### Declaración

**Si la secuencia contiene destabilizers, DEBE incluir stabilizers**

**Destabilizers:** {OZ (Dissonance), ZHIR (Mutation), VAL (Expansion)}  
**Stabilizers:** {IL (Coherence), THOL (Self-organization)}

---

### Derivación Completa

#### Ecuación Nodal Integrada

```
EPI(t_f) = EPI(t_0) + ∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ
```

#### Sin Stabilizers (Divergencia)

```
Solo destabilizers presentes:
  dΔNFR/dt > 0 siempre (feedback positivo)
  → ΔNFR(t) ~ e^(λt) (crecimiento exponencial)
  → ∫ νf · ΔNFR dt → ∞ (DIVERGE)
  → EPI(t) → ∞ (fragmentación estructural)

Sistema pierde coherencia, fragmenta en ruido incoherente
```

#### Con Stabilizers (Convergencia)

```
Stabilizers presentes:
  dΔNFR/dt puede ser < 0 (feedback negativo)
  → ΔNFR(t) → atractor acotado
  → ∫ νf · ΔNFR dt converge
  → EPI(t) permanece acotado (coherencia preservada)

Sistema mantiene coherencia, evoluciona de forma acotada
```

#### Prueba Matemática

**Teorema de Convergencia Integral:**

1. **Destabilizers** crean feedback positivo: d(ΔNFR)/dt > 0
2. Sin feedback negativo, integral diverge (test de comparación)
3. Integral divergente → EPI no acotado → fragmentación (no-físico)
4. **Stabilizers** proveen feedback negativo → convergencia → evolución acotada

---

### ¿Por qué estos operators?

#### Destabilizers (Incrementan |ΔNFR|)

**1. Dissonance (OZ) ⚡**
- **Efecto:** Aumenta |ΔNFR|, puede trigger bifurcación si ∂²EPI/∂t² > τ
- **Feedback:** Positivo fuerte
- **Risk:** Alto riesgo de divergencia sin estabilización

**2. Mutation (ZHIR) 🧬**
- **Efecto:** θ → θ' cuando ΔEPI/Δt > ξ (transformación de fase)
- **Feedback:** Positivo en transición
- **Risk:** Transformación inestable sin base estable

**3. Expansion (VAL) 📈**
- **Efecto:** dim(EPI) aumenta (más grados de libertad)
- **Feedback:** Positivo moderado
- **Risk:** Complejidad incontrolada sin organización

#### Stabilizers (Reducen |ΔNFR|)

**1. Coherence (IL) 🔒**
- **Física:** Estabiliza forma mediante feedback negativo
- **Efecto:** Reduce |ΔNFR|, aumenta C(t)
- **Feedback:** **Negativo fuerte directo**
- **Garantía:** Explícitamente reduce presión estructural

**2. Self-organization (THOL) 🌱**
- **Física:** Formación autopoiética de patrones
- **Efecto:** Crea sub-EPIs, cierre autopoiético
- **Feedback:** **Negativo emergente**
- **Garantía:** Auto-limita crecimiento mediante boundaries

**Solo IL y THOL** tienen física de feedback negativo suficientemente fuerte.

---

### Ejemplos

#### Válido: Destabilizer + Stabilizer

```python
from tnfr.operators.definitions import (
    Emission, Dissonance, Coherence, Silence
)

# ✓ VÁLIDO: Dissonance (destabilizer) + Coherence (stabilizer)
sequence = [
    Emission(),      # Generator (U1a)
    Dissonance(),    # Destabilizer
    Coherence(),     # Stabilizer (U2)
    Silence()        # Closure (U1b)
]
validate_grammar(sequence, epi_initial=0.0)  # ✓ Pasa U2
```

#### Inválido: Destabilizer sin Stabilizer

```python
# ✗ INVÁLIDO: Dissonance sin stabilizer
sequence = [
    Emission(),
    Dissonance(),    # Destabilizer
    Silence()        # No stabilizer!
]
validate_grammar(sequence, epi_initial=0.0)
# ValueError: U2 violation: Destabilizers without stabilizers
```

#### Múltiples Destabilizers

```python
# ✓ VÁLIDO: Múltiples destabilizers + stabilizer
sequence = [
    Emission(),
    Dissonance(),    # Destabilizer 1
    Expansion(),     # Destabilizer 2
    Coherence(),     # Stabilizer (cubre ambos)
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓
```

---

### Interpretación Física

**Stabilizers como "gravedad estructural":**

Como gravedad previene dispersión cósmica, stabilizers previenen fragmentación estructural.

**Sin gravedad:** Materia dispersa infinitamente  
**Sin stabilizers:** Estructura fragmenta infinitamente

**Analogías:**
- **Control de feedback:** Necesitas feedback negativo para prevenir runaway
- **Sistemas ecológicos:** Necesitas factores limitantes para prevenir explosión poblacional
- **Reacciones químicas:** Necesitas inhibidores para prevenir divergencia autocatalítica

---

### Canonicidad

**Nivel:** **ABSOLUTE** (Teorema matemático)

**Prueba:** Teorema de convergencia integral + Ecuación nodal

**Traceabilidad:**
- Análisis (convergencia integral) + Ecuación nodal → Necesidad matemática directa
- TNFR.pdf § 2.1 → Dinámica integrada
- Código: `src/tnfr/operators/grammar.py::validate_u2_convergence()`

**Tests:**
- `tests/unit/operators/test_unified_grammar.py::TestU2Convergence`

---

## U3: RESONANT COUPLING

### Física de Base

**Principio:** Resonancia requiere compatibilidad de fase.

**Fuente:** AGENTS.md Invariant #5 + Física de ondas

---

### Declaración

**Si la secuencia contiene coupling/resonance, DEBE verificar fase**

**Operators:** {UM (Coupling), RA (Resonance)}  
**Condición:** |φᵢ - φⱼ| ≤ Δφ_max (típicamente π/2)

---

### Derivación desde Física de Resonancia

#### Condición Clásica de Resonancia

```
Dos osciladores acoplan ⟺ frecuencia Y fase compatibles

Condición de frecuencia: ωᵢ ≈ ωⱼ (se cumple con matching estructural)
Condición de fase: |φᵢ - φⱼ| ≤ Δφ_max (típicamente π/2)
```

#### Sin Verificación de Fase (No-Físico)

```
Nodos intentan acoplar con φᵢ ≈ π, φⱼ ≈ 0 (antifase)
→ Interferencia de ondas: Aᵢ sin(ωt) + Aⱼ sin(ωt + π) = 0
→ Interferencia destructiva (cancelación de patrón)
→ NO hay acoplamiento efectivo ("ghost coupling" no-físico)
```

#### Con Verificación de Fase (Físico)

```
Solo nodos sincrónicos acoplan (interferencia constructiva)
→ Aᵢ sin(ωt) + Aⱼ sin(ωt + δ) ≈ 2A sin(ωt) para δ ≈ 0
→ Amplificación resonante (acoplamiento físico real)
```

---

### Analogía Física

**Sintonización de radio:**
- Necesitas match de frecuencia **Y** fase para señal clara
- Fuera de fase → ruido/estática
- En fase → señal amplificada

**Coherencia láser:**
- Fotones deben estar alineados en fase para haz coherente
- Desalineación de fase → luz incoherente
- Alineación de fase → beam coherente

**Circuitos AC:**
- Fase importa para transmisión de potencia (factor de potencia)
- Antifase → pérdida de potencia
- Fase alineada → transmisión eficiente

---

### Implementación

```python
from tnfr.operators.grammar import validate_resonant_coupling

def validate_resonant_coupling(G, node_i, node_j, delta_phi_max=np.pi/2):
    """Verifica compatibilidad de fase para acoplamiento.
    
    Physics: |φᵢ - φⱼ| ≤ Δφ_max para resonancia constructiva.
    
    Parameters
    ----------
    G : TNFRGraph
        Red TNFR
    node_i, node_j : NodeId
        Nodos a acoplar
    delta_phi_max : float
        Máxima diferencia de fase permitida (default: π/2)
        
    Returns
    -------
    bool
        True si compatible, False si antifase
        
    Raises
    ------
    ValueError
        Si diferencia de fase excede threshold
    """
    phi_i = G.nodes[node_i]['theta']
    phi_j = G.nodes[node_j]['theta']
    
    delta_phi = abs(phi_i - phi_j)
    # Normalizar a [0, π] considerando periodicidad
    delta_phi = min(delta_phi, 2*np.pi - delta_phi)
    
    if delta_phi > delta_phi_max:
        raise ValueError(
            f"U3 violation: Phase mismatch {delta_phi:.3f} > {delta_phi_max:.3f}"
        )
    
    return True
```

---

### Ejemplos

#### Verificación en Runtime

```python
from tnfr.operators.definitions import Emission, Coupling, Coherence, Silence

# Secuencia con coupling
sequence = [
    Emission(),
    Coupling(),      # Requiere verificación de fase (U3)
    Coherence(),
    Silence()
]

# Validación gramatical pasa (U3 no se verifica en gramática)
validate_grammar(sequence, epi_initial=0.0)  # ✓

# Pero en runtime, Coupling verifica fase:
G = create_tnfr_network(nodes=10)
Coupling()(G, node_i=0)  # Internamente verifica fase con vecinos
```

#### Coupling con Incompatibilidad de Fase

```python
# Si nodos están en antifase, Coupling debe fallar
G.nodes[0]['theta'] = 0.0      # Fase 0
G.nodes[1]['theta'] = np.pi    # Antifase

try:
    # Intento de coupling entre nodos antifase
    Coupling()(G, node=0, target=1)
except ValueError as e:
    print(f"U3 violation: {e}")
    # "Phase mismatch: |0.0 - 3.14| > π/2"
```

---

### Canonicidad

**Nivel:** **ABSOLUTE** (Física de interferencia + Invariant explícito)

**Fuentes:**
1. **Física de interferencia de ondas** → Requerimiento de fase
2. **AGENTS.md Invariant #5** → Explícito en TNFR
   > "Phase check: no coupling is valid without explicit phase verification (synchrony)"

**Traceabilidad:**
- Física de resonancia (mecánica clásica) → Requerimiento de fase
- AGENTS.md Invariant #5 → Requerimiento explícito TNFR
- Código: `src/tnfr/operators/grammar.py::validate_resonant_coupling()`

**Tests:**
- `tests/unit/operators/test_unified_grammar.py::TestU3ResonantCoupling`

---

## U4: BIFURCATION DYNAMICS

### Física de Base

**Principio:** Transiciones de fase requieren energía umbral y mecanismos de control.

**Fuente:** Teoría de bifurcaciones + AGENTS.md Contract OZ

---

### U4a: Bifurcation Triggers Need Handlers

#### Declaración

**Si la secuencia contiene bifurcation triggers, DEBE incluir handlers**

**Triggers:** {OZ (Dissonance), ZHIR (Mutation)}  
**Handlers:** {THOL (Self-organization), IL (Coherence)}

#### Derivación

**Condición de bifurcación (desde AGENTS.md Contract OZ):**

```
Sistema sufre transición de fase cuando ∂²EPI/∂t² > τ
```

**Dissonance (OZ) y Mutation (ZHIR):**
- Diseñados explícitamente para trigger ∂²EPI/∂t² > τ
- Crean inestabilidad estructural (punto de bifurcación)

**Sin handlers:**
```
Sistema cruza bifurcación → caos/fragmentación
→ No hay mecanismo para organizar nueva fase
→ "Explosión" no-física de ΔNFR
```

**Con handlers:**
```
Bifurcación → caos transitorio → auto-organización → nueva fase estable
→ Cierre autopoiético (THOL) o estabilización explícita (IL)
→ Transición de fase física (como water → ice con nucleación)
```

#### Analogía Física

**Transición agua → hielo:**
- Necesita sitios de nucleación (handlers) para cristalización ordenada
- Sin nucleación → congelamiento desordenado/fragmentado
- Con nucleación → estructura cristalina coherente

**Threshold láser:**
- Necesita estabilización de cavidad para emisión coherente
- Sin estabilización → emisión caótica
- Con estabilización → beam láser coherente

#### Ejemplos

```python
from tnfr.operators.definitions import (
    Emission, Dissonance, SelfOrganization, Silence
)

# ✓ VÁLIDO: Trigger + Handler
sequence = [
    Emission(),
    Dissonance(),         # Trigger (U4a)
    SelfOrganization(),   # Handler (U4a)
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# ✗ INVÁLIDO: Trigger sin Handler
sequence = [
    Emission(),
    Dissonance(),         # Trigger
    Silence()             # No handler!
]
validate_grammar(sequence, epi_initial=0.0)
# ValueError: U4a violation: Bifurcation triggers without handlers
```

#### Canonicidad

**Nivel:** **STRONG** (Requerimiento físico desde teoría de bifurcaciones)

**Traceabilidad:**
- Contract OZ (AGENTS.md) → Física de bifurcaciones
- Teoría de bifurcaciones → Necesidad de mecanismos de estabilidad
- Código: `src/tnfr/operators/grammar.py::validate_u4a_bifurcation_triggers()`

---

### U4b: Transformers Need Context (Graduated Destabilization)

#### Declaración

**Si la secuencia contiene transformers, DEBE tener destabilizer reciente**

**Transformers:** {ZHIR (Mutation), THOL (Self-organization)}  
**Timing:** Destabilizer dentro de ~3 operadores  
**Adicional para ZHIR:** Prior Coherence (IL) para base estable

#### Derivación desde Física de Umbral

**Requerimientos de transición de fase:**

```
1. Energía umbral: E > E_critical
2. Timing apropiado: Energía debe ser "fresca" (reciente)
```

**Mutation (ZHIR) y Self-organization (THOL):**
- Realizan transiciones de fase estructurales
- Requieren |ΔNFR| > threshold (condición de energía)

**Sin destabilizer reciente:**
```
|ΔNFR| puede haber decaído bajo threshold
→ Energía insuficiente para transición de fase
→ Transformación falla o produce estado inestable
```

**Con destabilizer reciente (~3 ops):**
```
|ΔNFR| todavía elevado (energía disponible)
→ Gradiente suficiente para cruzar threshold
→ Transición de fase física exitosa
```

**Adicional para ZHIR (Mutation):**
```
Necesita Coherence (IL) previa para base de transformación estable
→ Como crecimiento cristalino: necesita semilla estable
```

#### ¿Por qué ~3 operadores?

**Basado en tiempo típico de decaimiento de ΔNFR:**

- Asegura que gradiente no ha disipado bajo threshold
- Como vida media en física nuclear
- Timing constraint emerge de dinámica de ΔNFR

#### Ejemplos

**Mutation con contexto completo:**

```python
from tnfr.operators.definitions import (
    Emission, Coherence, Dissonance, Mutation, Silence
)

# ✓ VÁLIDO: Mutation con base estable + destabilizer reciente
sequence = [
    Emission(),
    Coherence(),    # Base estable para ZHIR (U4b)
    Dissonance(),   # Destabilizer reciente (U4b)
    Mutation(),     # Transformer (U4b)
    Coherence(),    # Stabilizer (U2)
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# ✗ INVÁLIDO: Mutation sin destabilizer reciente
sequence = [
    Emission(),
    Coherence(),
    Mutation(),     # No hay destabilizer reciente!
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)
# ValueError: U4b violation: Transformer without recent destabilizer

# ✗ INVÁLIDO: Mutation sin prior Coherence
sequence = [
    Emission(),
    Dissonance(),   # Destabilizer
    Mutation(),     # No hay IL previa!
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)
# ValueError: U4b violation: ZHIR needs prior Coherence
```

**Self-organization con contexto:**

```python
# ✓ VÁLIDO: THOL con destabilizer reciente
sequence = [
    Emission(),
    Expansion(),          # Destabilizer (VAL)
    Reception(),          # Operador intermedio (< 3 ops)
    SelfOrganization(),   # Transformer (THOL)
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓
```

#### Canonicidad

**Nivel:** **STRONG** (Física de threshold + timing)

**Traceabilidad:**
- Física de energía umbral → Requerimiento de energía
- Dinámica de decaimiento de ΔNFR → Constraint de timing
- Estabilidad de bifurcación → Prior IL para ZHIR
- Código: `src/tnfr/operators/grammar.py::validate_u4b_transformer_context()`

---

## 📊 Tabla de Canonicidad

| Rule | Canonicity | Necessity | Physical Base | Reference |
|------|------------|-----------|---------------|-----------|
| U1a | ✅ CANONICAL | **Absolute** | ∂EPI/∂t undefined at EPI=0 | Nodal equation |
| U1b | ✅ CANONICAL | **Strong** | Sequences as action potentials | Wave physics |
| U2  | ✅ CANONICAL | **Absolute** | Integral convergence theorem | Analysis |
| U3  | ✅ CANONICAL | **Absolute** | Resonance physics + Inv. #5 | AGENTS.md |
| U4a | ✅ CANONICAL | **Strong** | Contract OZ + bifurcation | Contracts |
| U4b | ✅ CANONICAL | **Strong** | Threshold physics + timing | Bifurcation theory |

**Clave:**
- **Absolute:** Necesidad matemática (no puede ser de otra forma)
- **Strong:** Requerimiento físico (violarla produce estados no-físicos)

---

## 🧪 Testing

### Tests Mínimos Requeridos

**Para cada restricción, implementar:**

1. **Test de validación positiva** - Secuencia válida pasa
2. **Test de validación negativa** - Secuencia inválida falla
3. **Test de edge cases** - Casos límite correctos

### Ejemplo: Tests para U2

```python
import pytest
from tnfr.operators.grammar import validate_grammar
from tnfr.operators.definitions import *

class TestU2Convergence:
    """Tests para U2: CONVERGENCE & BOUNDEDNESS"""
    
    def test_destabilizers_require_stabilizers(self):
        """Destabilizers + stabilizers → válido"""
        sequence = [
            Emission(),
            Dissonance(),    # Destabilizer
            Coherence(),     # Stabilizer
            Silence()
        ]
        assert validate_grammar(sequence, epi_initial=0.0)
    
    def test_destabilizers_without_stabilizers_fail(self):
        """Destabilizers sin stabilizers → inválido"""
        sequence = [
            Emission(),
            Dissonance(),    # Destabilizer
            Silence()        # No stabilizer
        ]
        with pytest.raises(ValueError, match="U2 violation"):
            validate_grammar(sequence, epi_initial=0.0)
    
    def test_no_destabilizers_passes(self):
        """Sin destabilizers → no necesita stabilizers"""
        sequence = [
            Emission(),
            Reception(),
            Silence()
        ]
        assert validate_grammar(sequence, epi_initial=0.0)
```

**Ver:** [06-VALIDACION-Y-TESTING.md](06-VALIDACION-Y-TESTING.md) para estrategia completa

---

## 🔍 Troubleshooting

### Issue: "Need generator when EPI=0"

**Problema:** Secuencia no empieza con generator cuando `epi_initial=0.0`

**Solución:**
1. Agregar generator al inicio: `[Emission(), ...]`
2. O setear `epi_initial > 0` si empiezas desde estructura existente

### Issue: "Destabilizer without stabilizer"

**Problema:** Secuencia tiene {OZ, ZHIR, VAL} pero no {IL, THOL}

**Solución:** Agregar stabilizer después de destabilizers:
```python
[Emission(), Dissonance(), Coherence(), Silence()]
```

### Issue: "Transformer needs recent destabilizer"

**Problema:** {ZHIR, THOL} sin destabilizer reciente

**Solución:** Agregar destabilizer dentro de ~3 operators antes de transformer:
```python
[Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
```

### Issue: "Mutation needs prior coherence"

**Problema:** ZHIR sin IL previa

**Solución:** Agregar Coherence antes de Mutation:
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Silence()]
```

### Issue: "Phase mismatch in coupling"

**Problema:** Intento de acoplamiento con |φᵢ - φⱼ| > Δφ_max

**Solución:** Asegurar compatibilidad de fase antes de coupling:
```python
# Verificar fase manualmente antes de Coupling
delta_phi = abs(G.nodes[i]['theta'] - G.nodes[j]['theta'])
if delta_phi > np.pi/2:
    # Ajustar fase o no acoplar
    pass
```

---

## 📚 Referencias

### Documentos Relacionados

- **[01-CONCEPTOS-FUNDAMENTALES.md](01-CONCEPTOS-FUNDAMENTALES.md)** - Fundamentos TNFR
- **[03-OPERADORES-Y-GLIFOS.md](03-OPERADORES-Y-GLIFOS.md)** - 13 operadores canónicos
- **[04-SECUENCIAS-VALIDAS.md](04-SECUENCIAS-VALIDAS.md)** - Patrones de secuencias
- **[06-VALIDACION-Y-TESTING.md](06-VALIDACION-Y-TESTING.md)** - Estrategia de tests
- **[../../UNIFIED_GRAMMAR_RULES.md](../../UNIFIED_GRAMMAR_RULES.md)** - Derivaciones formales completas
- **[../../AGENTS.md](../../AGENTS.md)** - Invariantes y contratos

### Implementación

- `src/tnfr/operators/grammar.py` - Implementación canónica
- `tests/unit/operators/test_unified_grammar.py` - Suite de tests

### Papers y Recursos

- Teoría de Bifurcaciones - Strogatz, "Nonlinear Dynamics and Chaos"
- Física de Interferencia - Feynman Lectures Vol 1, Chapter 29
- Teoremas de Convergencia Integral - Análisis Real estándar

---

<div align="center">

**Próximo paso:** [03-OPERADORES-Y-GLIFOS.md](03-OPERADORES-Y-GLIFOS.md)  
**Aprenderás:** Catálogo completo de 13 operadores canónicos

**"If it strengthens coherence and derives from physics, GO AHEAD."**

</div>
