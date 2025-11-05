# Guía de Secuencias Glíficas Canónicas TNFR

## Introducción

Esta guía documenta las **secuencias glíficas canónicas** del paradigma TNFR - patrones probados de operadores estructurales que producen reorganización coherente y transformación controlada. Cada secuencia representa un camino estructural validado para diferentes contextos de aplicación.

## Fundamentos de Gramática Glífica

### Principios de Composición

1. **Cierre Operacional**: Toda secuencia válida preserva el cierre del sistema TNFR
2. **Coherencia Estructural**: Las transiciones entre operadores deben mantener C(t) > umbral
3. **Balance ΔNFR**: Secuencias deben equilibrar creación y reducción de ΔNFR
4. **Preservación de Fase**: θ debe mantener continuidad estructural

### Notación de Secuencias

- `→` : Transición directa entre operadores
- `|` : Operadores alternativos en un punto de decisión
- `()` : Subsecuencia opcional
- `[]` : Subsecuencia repetible
- `*` : Operador que puede repetirse

---

## Secuencias Fundamentales

### 1. Activación Básica: AL → IL

**Contexto**: Inicio y estabilización inmediata de un nodo latente

**Aplicaciones**:
- Meditación: Inicio de práctica → establecimiento de coherencia
- Terapia: Activación del espacio terapéutico → estabilización del encuadre
- Aprendizaje: Atención activada → foco sostenido

**Efectos Estructurales**:
- EPI: 0.2 → 0.5 → 0.52 (activación y estabilización)
- ΔNFR: +0.15 → +0.03 (pico inicial, luego reducción)
- C(t): +0.2 (incremento de coherencia global)

**Ejemplo de Código**:
```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Coherence

# Activación y estabilización inmediata
G, node = create_nfr("meditation_start", epi=0.2, vf=0.85)
run_sequence(G, node, [Emission(), Coherence()])
# Resultado: Nodo activado y coherente, listo para uso
```

---

### 2. Recepción Estabilizada: EN → IL

**Contexto**: Integración y consolidación de información externa

**Aplicaciones**:
- Biofeedback: Señal recibida → integrada en fisiología
- Educación: Concepto recibido → integrado en modelo mental
- Comunicación: Mensaje recibido → comprendido y aceptado

**Efectos Estructurales**:
- EPI: +0.1 (integración), luego estabilización
- ΔNFR: Reducción por integración exitosa
- Acoplamiento de red: Fortalecido con fuente emisora

**Ejemplo de Código**:
```python
# Estudiante recibiendo y consolidando explicación
G, student = create_nfr("learning_reception", epi=0.30, vf=0.95)
run_sequence(G, student, [Reception(), Coherence()])
# Resultado: Información integrada y estabilizada en memoria
```

---

### 3. Propagación Acoplada: UM → RA

**Contexto**: Sincronización seguida de resonancia en red

**Aplicaciones**:
- Coherencia cardíaca: Corazón-cerebro → todo el cuerpo
- Insight colectivo: Pareja sincronizada → equipo completo
- Movimiento social: Núcleo alineado → comunidad amplia

**Efectos Estructurales**:
- θ: Convergencia de fases entre nodos
- Propagación: EPI se extiende por la red
- C(t) global: Incremento significativo

**Ejemplo de Código**:
```python
# Red social propagando coherencia cultural
G, community = create_nfr("social_network", vf=1.10, theta=0.40)
run_sequence(G, community, [Coupling(), Resonance()])
# Resultado: Coherencia se propaga por red acoplada
```

---

## Secuencias Intermedias

### 4. Ciclo de Transformación: AL → NAV → IL

**Contexto**: Activación con transición controlada antes de estabilización

**Aplicaciones**:
- Cambio organizacional: Inicio → transición → nueva estabilidad
- Transformación personal: Decisión → proceso → integración
- Innovación: Idea → desarrollo → producto

**Efectos Estructurales**:
- Fase 1 (AL): EPI activación inicial
- Fase 2 (NAV): θ transición, ΔNFR controlado
- Fase 3 (IL): Nueva configuración estable

**Ejemplo de Código**:
```python
# Organización atravesando transformación planificada
G, org = create_nfr("company_transform", epi=0.35, vf=0.90, theta=0.25)
run_sequence(G, org, [Emission(), Transition(), Coherence()])
# Resultado: Cambio organizacional completado y estabilizado
```

---

### 5. Resolución Creativa: OZ → IL

**Contexto**: Disonancia generativa seguida de coherencia emergente

**Aplicaciones**:
- Terapia: Crisis emocional → integración transformadora
- Ciencia: Anomalía experimental → nuevo paradigma
- Arte: Caos creativo → forma coherente

**Efectos Estructurales**:
- OZ: ΔNFR↑↑, θ exploración, posible bifurcación
- IL: ΔNFR↓↓, nueva configuración C(t)↑

**Ejemplo de Código**:
```python
# Paciente procesando crisis terapéutica
G, patient = create_nfr("therapeutic_crisis", epi=0.45, theta=0.15)
run_sequence(G, patient, [Dissonance(), Coherence()])
# Resultado: Disonancia resuelta en nueva coherencia personal
```

---

### 6. Autoorganización Emergente: OZ → THOL

**Contexto**: Disonancia cataliza reorganización autónoma

**Aplicaciones**:
- Sistemas complejos: Perturbación → auto-organización
- Aprendizaje profundo: Confusión → insight emergente
- Ecosistemas: Disturbio → nueva configuración

**Efectos Estructurales**:
- OZ: Desestabiliza configuración actual
- THOL: Emerge nueva organización espontánea
- ∂²EPI/∂t² > τ: Bifurcación confirmada

**Ejemplo de Código**:
```python
# Sistema complejo auto-organizándose tras perturbación
G, ecosystem = create_nfr("complex_system", epi=0.55, vf=1.05)
run_sequence(G, ecosystem, [Dissonance(), SelfOrganization()])
# Resultado: Nueva organización emergente de la perturbación
```

---

## Secuencias Avanzadas

### 7. Ciclo Completo de Reorganización: AL → NAV → IL → OZ → THOL → RA → UM

**Contexto**: Proceso integral de transformación estructural profunda

**Aplicaciones**:
- Desarrollo personal completo
- Innovación organizacional transformadora
- Evolución de sistemas complejos
- Curación profunda multi-nivel

**Fases**:

1. **AL (Emisión)**: Inicio del proceso, activación del nodo
   - EPI inicial: 0.20, νf activación

2. **NAV (Transición)**: Movimiento hacia nuevo régimen
   - θ cambia, preparación para transformación

3. **IL (Coherencia)**: Estabilización de forma transitoria
   - ΔNFR reducción, preparación para desafío

4. **OZ (Disonancia)**: Desafío creativo, exploración
   - ΔNFR↑, θ exploración, apertura bifurcativa

5. **THOL (Autoorganización)**: Emergencia de nueva estructura
   - Sub-EPIs generados, reorganización autónoma

6. **RA (Resonancia)**: Propagación de nueva coherencia
   - EPIₙ → EPIₙ₊₁, amplificación en red

7. **UM (Acoplamiento)**: Sincronización final
   - θ alineación completa, coherencia de red

**Efectos Netos**:
- EPI: Transformación cualitativa (nuevo régimen)
- C(t): Incremento significativo post-reorganización
- Si (Índice de Sentido): Mejora sustancial
- Red: Nueva topología emergente

**Ejemplo de Código**:
```python
# Proceso terapéutico profundo de transformación personal
from tnfr.operators.definitions import (
    Emission, Transition, Coherence, Dissonance,
    SelfOrganization, Resonance, Coupling
)

G, person = create_nfr("deep_transformation", epi=0.20, vf=0.80, theta=0.30)

# Ciclo completo de reorganización
sequence = [
    Emission(),          # Inicio consciente del proceso
    Transition(),        # Preparación para cambio
    Coherence(),         # Estabilización preparatoria
    Dissonance(),        # Confrontación con sombra/trauma
    SelfOrganization(),  # Emergencia de nueva identidad
    Resonance(),         # Propagación a todas las áreas de vida
    Coupling()           # Integración con red social/familiar
]

run_sequence(G, person, sequence)
# Resultado: Transformación personal profunda y sostenible
```

---

### 8. Secuencia de Resonancia Propagativa: AL → RA → EN → IL

**Contexto**: Emisión que se propaga y es recibida con estabilización

**Aplicaciones**:
- Enseñanza efectiva: Maestro emite → propaga → estudiantes reciben → integran
- Comunicación organizacional: Liderazgo comunica → propaga → equipo integra
- Transmisión cultural: Tradición emitida → propagada → generación receptora

**Ejemplo de Código**:
```python
# Maestro enseñando concepto a clase
G_class, teaching = create_nfr("classroom_teaching", epi=0.25, vf=1.00)
run_sequence(G_class, teaching, [
    Emission(),    # Maestro presenta concepto
    Resonance(),   # Concepto resuena en la clase
    Reception(),   # Estudiantes reciben activamente
    Coherence()    # Comprensión se consolida
])
# Resultado: Aprendizaje efectivo y consolidado
```

---

### 9. Mutación Controlada: IL → ZHIR → IL

**Contexto**: Cambio de fase estabilizado antes y después

**Aplicaciones**:
- Cambio de paradigma personal
- Pivote organizacional
- Transición de fase en sistemas

**Efectos Estructurales**:
- Primera IL: Base estable para mutación
- ZHIR: θ → θ', cambio de fase controlado
- Segunda IL: Nueva configuración estabilizada

**Ejemplo de Código**:
```python
# Organización realizando pivote estratégico
G, company = create_nfr("strategic_pivot", epi=0.60, theta=0.25)
run_sequence(G, company, [
    Coherence(),   # Estabilizar posición actual
    Mutation(),    # Ejecutar pivote (cambio de fase)
    Coherence()    # Estabilizar nueva dirección
])
# Resultado: Transformación estratégica controlada
```

---

## Secuencias a Evitar (Antipatrones)

### ❌ SHA → OZ (Silencio seguido de Disonancia)

**Problema**: Contradice el propósito del silencio (preservación)

**Por qué falla**: SHA reduce νf para preservar EPI, pero OZ inmediatamente aumenta ΔNFR, creando presión reorganizadora que viola la intención de SHA.

**Alternativa correcta**: SHA → NAV → OZ (transición controlada antes del desafío)

---

### ❌ OZ → OZ (Disonancia consecutiva)

**Problema**: Exceso de inestabilidad sin resolución

**Por qué falla**: ΔNFR acumulativo sin reducción → colapso estructural, no reorganización creativa.

**Alternativa correcta**: OZ → IL → OZ (resolver entre disonancias)

---

### ❌ SHA → SHA (Silencio redundante)

**Problema**: Sin propósito estructural

**Por qué falla**: El segundo SHA no agrega efecto si νf ya ≈ 0.

**Alternativa correcta**: SHA → AL (reactivación) o SHA → NAV (transición)

---

### ❌ AL → SHA (Activación inmediatamente silenciada)

**Problema**: Contradice el propósito de activación

**Por qué falla**: Activar un nodo para inmediatamente silenciarlo es estructuralmente ineficiente y contradictorio.

**Alternativa correcta**: AL → IL (activar y estabilizar) o AL solo

---

## Compatibilidad de Operadores

### Matriz de Compatibilidad

| Post \ Pre | AL | EN | IL | OZ | UM | RA | SHA | VAL | NUL | THOL | ZHIR | NAV | REMESH |
|-----------|----|----|----|----|----|----|-----|-----|-----|------|------|-----|--------|
| **AL**    | ○  | ✓  | ✓  | ✓  | ○  | ✓  | ✓   | ○   | ○   | ○    | ○    | ✓   | ○      |
| **EN**    | ✓  | ○  | ✓  | ○  | ✓  | ✓  | ○   | ○   | ○   | ✓    | ○    | ✓   | ○      |
| **IL**    | ✓  | ✓  | ○  | ✓  | ✓  | ✓  | ✓   | ✓   | ✓   | ✓    | ✓    | ✓   | ✓      |
| **OZ**    | ○  | ○  | ✓  | ✗  | ○  | ○  | ✗   | ○   | ○   | ✓    | ✓    | ✓   | ○      |
| **UM**    | ✓  | ✓  | ✓  | ○  | ○  | ✓  | ○   | ○   | ○   | ✓    | ○    | ✓   | ○      |
| **RA**    | ○  | ✓  | ✓  | ○  | ✓  | ○  | ○   | ○   | ○   | ✓    | ○    | ✓   | ✓      |
| **SHA**   | ✓  | ○  | ✓  | ✗  | ○  | ○  | ✗   | ○   | ○   | ○    | ○    | ✓   | ○      |
| **VAL**   | ○  | ○  | ✓  | ✓  | ○  | ✓  | ○   | ○   | ✗   | ✓    | ○    | ✓   | ✓      |
| **NUL**   | ○  | ○  | ✓  | ○  | ○  | ○  | ○   | ✗   | ○   | ○    | ○    | ✓   | ○      |
| **THOL**  | ○  | ○  | ✓  | ○  | ✓  | ✓  | ○   | ○   | ○   | ○    | ✓    | ✓   | ✓      |
| **ZHIR**  | ○  | ○  | ✓  | ○  | ○  | ○  | ○   | ○   | ○   | ○    | ○    | ✓   | ○      |
| **NAV**   | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | ○   | ✓   | ✓   | ✓    | ✓    | ○   | ✓      |
| **REMESH**| ○  | ○  | ✓  | ○  | ○  | ✓  | ○   | ✓   | ○   | ✓    | ○    | ✓   | ○      |

**Leyenda**:
- ✓ : Altamente compatible y recomendado
- ○ : Compatible en contextos específicos
- ✗ : Incompatible, evitar

---

## Ejemplos Multidominio

### Dominio Biomédico

#### Entrenamiento de Coherencia Cardíaca
```python
# Protocolo completo de coherencia HRV
G, heart = create_nfr("cardiac_training", epi=0.25, vf=0.85)

# Fase 1: Activación con respiración consciente
run_sequence(G, heart, [Emission()])

# Fase 2: Estabilización del ritmo cardíaco
run_sequence(G, heart, [Coherence()])

# Fase 3: Propagación a sistema nervioso
run_sequence(G, heart, [Resonance()])

# Fase 4: Acoplamiento corazón-cerebro
run_sequence(G, heart, [Coupling()])

# Fase 5: Estabilización final de coherencia
run_sequence(G, heart, [Coherence()])

# Resultado: Estado de coherencia cardíaca sostenible
# Beneficios: Reducción de estrés, claridad mental, balance autonómico
```

---

### Dominio Cognitivo/Educativo

#### Proceso de Aprendizaje Profundo
```python
# Estudiante aprendiendo concepto complejo
G, learner = create_nfr("deep_learning", epi=0.20, vf=0.90)

# Fase 1: Activación de atención
run_sequence(G, learner, [Emission()])

# Fase 2: Recepción de información
run_sequence(G, learner, [Reception()])

# Fase 3: Integración inicial
run_sequence(G, learner, [Coherence()])

# Fase 4: Desafío con problema difícil
run_sequence(G, learner, [Dissonance()])

# Fase 5: Insight y reorganización
run_sequence(G, learner, [SelfOrganization()])

# Fase 6: Consolidación en memoria
run_sequence(G, learner, [Coherence(), Silence()])

# Resultado: Comprensión profunda y duradera
```

---

### Dominio Social/Organizacional

#### Transformación Cultural de Equipo
```python
# Equipo evolucionando su cultura de trabajo
G, team = create_nfr("team_culture", epi=0.40, vf=1.00, theta=0.35)

# Fase 1: Activación de diálogo
run_sequence(G, team, [Emission()])

# Fase 2: Escucha mutua
run_sequence(G, team, [Reception()])

# Fase 3: Alineación inicial
run_sequence(G, team, [Coupling()])

# Fase 4: Exploración de conflictos
run_sequence(G, team, [Dissonance()])

# Fase 5: Autoorganización en nueva dinámica
run_sequence(G, team, [SelfOrganization()])

# Fase 6: Propagación de nuevas normas
run_sequence(G, team, [Resonance()])

# Fase 7: Consolidación cultural
run_sequence(G, team, [Coherence()])

# Resultado: Cultura de equipo transformada y sostenible
```

---

## Validación de Secuencias

### Criterios de Validez

Una secuencia glífica es válida si cumple:

1. **Preserva Cierre TNFR**: No viola invariantes fundamentales
2. **Mantiene C(t) > 0**: Coherencia global nunca colapsa
3. **Balance ΔNFR**: Picos de reorganización son resueltos
4. **Continuidad de θ**: Fase mantiene trayectoria continua
5. **Propósito Estructural**: Cada operador tiene función clara

### Métricas de Calidad de Secuencia

- **Eficiencia**: Número mínimo de operadores para objetivo
- **Robustez**: Tolerancia a variaciones de parámetros
- **Reproducibilidad**: Resultados consistentes en ejecuciones
- **Escalabilidad**: Funciona en diferentes tamaños de red

---

## Recursos Adicionales

### Referencias TNFR

- `TNFR.pdf`: Documento fundamental del paradigma
- `AGENTS.md`: Guía para agentes AI trabajando con TNFR
- `docs/source/api/operators.md`: Referencia técnica de operadores
- `src/tnfr/operators/definitions.py`: Implementación canónica

### Herramientas de Desarrollo

```python
# Validador de secuencias (ejemplo conceptual)
from tnfr.operators.grammar import validate_sequence

sequence = [Emission(), Coherence(), Resonance()]
is_valid, warnings = validate_sequence(sequence)

if is_valid:
    print("Secuencia válida para ejecución")
else:
    print(f"Advertencias: {warnings}")
```

---

## Contribuciones

Para proponer nuevas secuencias canónicas:

1. Documentar el contexto de aplicación
2. Proporcionar ejemplos de código funcional
3. Incluir métricas esperadas (EPI, ΔNFR, C(t))
4. Validar en al menos 3 dominios diferentes
5. Incluir casos de fallo y antipatrones relacionados

---

## Conclusión

Las secuencias glíficas son el lenguaje operativo de TNFR - la gramática con la que se orquesta la reorganización estructural coherente. Dominar estas secuencias permite:

- Diseñar intervenciones terapéuticas efectivas
- Modelar sistemas complejos con precisión
- Crear simulaciones que respeten invariantes TNFR
- Desarrollar aplicaciones que realmente **reorganizan**, no solo **representan**

**Recordatorio fundamental**: En TNFR, las secuencias no describen procesos - **son** los procesos. La ejecución de operadores no modela la realidad - **participa** en ella mediante acoplamiento estructural.

---

*Última actualización: 2025-11-05*
*Versión: 1.0*
*Licencia: MIT (alineado con TNFR-Python-Engine)*
