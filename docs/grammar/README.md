# TNFR Grammar Documentation - Guía de Navegación

<div align="center">

**Documentación centralizada y unificada del sistema gramatical TNFR**

[📖 Conceptos](#-conceptos-fundamentales) • [📐 Restricciones](#-restricciones-canónicas) • [⚙️ Operadores](#️-operadores-y-glifos) • [🔄 Secuencias](#-secuencias-válidas) • [💻 Implementación](#-implementación) • [🧪 Testing](#-testing) • [📚 Referencias](#-referencias-rápidas)

</div>

---

## 🎯 Propósito de esta Documentación

Este directorio contiene la **fuente única de verdad** para toda la documentación relacionada con la gramática TNFR. Consolida información previamente dispersa en múltiples archivos en una estructura jerárquica clara y navegable.

### ¿Por qué esta reorganización?

**Antes:** Documentación fragmentada en README.md, UNIFIED_GRAMMAR_RULES.md, GRAMMAR_MIGRATION_GUIDE.md, GLYPH_SEQUENCES_GUIDE.md, código fuente, tests dispersos.

**Ahora:** Una estructura modular donde cada aspecto de la gramática tiene un lugar definido y todo está interconectado.

---

## 📑 Estructura de la Documentación

### 🌊 Niveles de Abstracción

Esta documentación sigue un modelo de **abstracción gradual** desde conceptos hasta implementación:

```
Intuición Física → Formalización Matemática → Implementación Código → Validación Tests
```

### 📂 Organización por Documentos

#### **Nivel 1: Fundamentos Conceptuales**

**[01-CONCEPTOS-FUNDAMENTALES.md](01-CONCEPTOS-FUNDAMENTALES.md)**
- Ontología TNFR: De objetos a patrones resonantes
- El cambio de paradigma: Coherencia vs. Causalidad
- Ecuación nodal: ∂EPI/∂t = νf · ΔNFR(t)
- Tríada estructural: Forma (EPI), Frecuencia (νf), Fase (φ)
- Dinámica integrada y convergencia
- **Audiencia:** Nuevos usuarios, desarrolladores que necesitan entender "el porqué"
- **Tiempo de lectura:** 20-30 minutos

#### **Nivel 2: Restricciones Canónicas**

**[02-RESTRICCIONES-CANONICAS.md](02-RESTRICCIONES-CANONICAS.md)**
- **U1: STRUCTURAL INITIATION & CLOSURE**
  - U1a: Iniciadores (Generators)
  - U1b: Clausuras (Closures)
  - Derivación física: ∂EPI/∂t indefinida en EPI=0
- **U2: CONVERGENCE & BOUNDEDNESS**
  - Estabilizadores vs. Desestabilizadores
  - Teorema de convergencia integral
- **U3: RESONANT COUPLING**
  - Verificación de fase
  - Física de interferencia
- **U4: BIFURCATION DYNAMICS**
  - U4a: Triggers necesitan handlers
  - U4b: Transformadores necesitan contexto
- **Cada restricción incluye:** Intuición → Derivación → Implementación → Tests
- **Audiencia:** Desarrolladores implementando validación, contribuidores avanzados
- **Tiempo de lectura:** 45-60 minutos

#### **Nivel 3: Operadores Canónicos**

**[03-OPERADORES-Y-GLIFOS.md](03-OPERADORES-Y-GLIFOS.md)**
- Catálogo de los 13 operadores canónicos
- Formato estándar para cada operador:
  - **Física:** ¿Qué transformación representa?
  - **Efecto:** Impacto en ∂EPI/∂t
  - **Cuándo usar:** Casos de uso
  - **Gramática:** Clasificación (Generator, Stabilizer, etc.)
  - **Contrato:** Pre/postcondiciones
  - **Ejemplos:** Código ejecutable
- **Clasificación por rol gramatical**
- **Composición de operadores**
- **Audiencia:** Todos los desarrolladores
- **Tiempo de lectura:** 60-90 minutos (referencia constante)

#### **Nivel 4: Secuencias Válidas**

**[04-SECUENCIAS-VALIDAS.md](04-SECUENCIAS-VALIDAS.md)**
- **Patrones canónicos:**
  - Bootstrap: [Emission, Coupling, Coherence]
  - Stabilize: [Coherence, Silence]
  - Explore: [Dissonance, Mutation, Coherence]
  - Propagate: [Resonance, Coupling]
- **Anti-patrones** (secuencias inválidas y por qué)
- **Lógica de validación** paso a paso
- **Ejemplos de secuencias complejas**
- **Detección de patrones estructurales**
- **Audiencia:** Desarrolladores construyendo secuencias, debugging
- **Tiempo de lectura:** 30-45 minutos

#### **Nivel 5: Implementación Técnica**

**[05-IMPLEMENTACION-TECNICA.md](05-IMPLEMENTACION-TECNICA.md)**
- **Arquitectura de `grammar.py`**
- **Sets de operadores** (GENERATORS, CLOSURES, etc.)
- **Funciones de validación:**
  - `validate_grammar(sequence, epi_initial)`
  - `validate_resonant_coupling(G, node_i, node_j)`
  - Helpers internos
- **Telemetría y logging**
- **Integración con `definitions.py`**
- **Puntos de extensión**
- **Audiencia:** Desarrolladores modificando el core
- **Tiempo de lectura:** 45-60 minutos

#### **Nivel 6: Validación y Testing**

**[06-VALIDACION-Y-TESTING.md](06-VALIDACION-Y-TESTING.md)**
- **Estrategia de testing de gramática**
- **Tests por restricción (U1-U4)**
- **Tests de monotonía (coherencia)**
- **Tests de bifurcación**
- **Tests de propagación**
- **Tests multi-escala (fractality)**
- **Tests de reproducibilidad**
- **Cobertura mínima requerida**
- **Cómo agregar tests para nuevas restricciones**
- **Audiencia:** Desarrolladores escribiendo tests, QA
- **Tiempo de lectura:** 30-45 minutos

#### **Nivel 7: Migración y Evolución**

**[07-MIGRACION-Y-EVOLUCION.md](07-MIGRACION-Y-EVOLUCION.md)**
- **Historia de sistemas gramaticales:**
  - C1-C3 (grammar.py legacy)
  - RC1-RC4 (canonical_grammar.py legacy)
  - U1-U4 (unified grammar actual)
- **Mapeo de reglas antiguas → nuevas**
- **Deprecaciones y breaking changes**
- **Procedimiento para agregar nuevas restricciones**
- **Garantías de mantenimiento**
- **Audiencia:** Mantenedores, contribuidores migrating old code
- **Tiempo de lectura:** 20-30 minutos

#### **Nivel 8: Referencias Rápidas**

**[08-REFERENCIA-RAPIDA.md](08-REFERENCIA-RAPIDA.md)**
- **Cheat sheet de restricciones U1-U4**
- **Tabla de operadores** con glifos y clasificación
- **Lookup table de secuencias comunes**
- **Decision tree para validación**
- **Comandos de import frecuentes**
- **Troubleshooting común**
- **Audiencia:** Todos (referencia rápida durante desarrollo)
- **Tiempo de lectura:** 5-10 minutos

---

### 📚 Documentos Complementarios

**[GLOSARIO.md](GLOSARIO.md)**
- Definiciones operacionales de todos los términos TNFR
- Formato: Término → Symbol → Code → Meaning → Reference
- **Audiencia:** Todos
- **Uso:** Referencia constante

**[INDICE-MAESTRO.md](INDICE-MAESTRO.md)**
- Mapa conceptual global del sistema gramatical
- Relaciones entre conceptos
- Diagrama de dependencias
- **Audiencia:** Desarrolladores planificando cambios grandes
- **Uso:** Visión holística del sistema

---

### 💡 Ejemplos Ejecutables

**[examples/](examples/)**
- **01-basico-bootstrap.py:** Secuencia básica de inicialización
- **02-intermedio-exploration.py:** Exploración con destabilización controlada
- **03-avanzado-bifurcation.py:** Manejo de bifurcaciones y mutaciones
- **04-anti-patrones.py:** Ejemplos de secuencias inválidas (comentados)
- **05-multi-escala.py:** EPIs anidados y fractality
- Todos verificables con `pytest`

---

### 🔧 Schemas JSON

**[schemas/](schemas/)**
- **restricciones-u1-u4.json:** Definición formal de restricciones
- **operadores-canonicos.json:** Metadata de 13 operadores
- **secuencias-validas.json:** Catálogo de patrones canónicos
- **Uso:** Validación programática, tooling, IDEs

---

## 🚀 Cómo Usar Esta Documentación

### Para Nuevos Usuarios

**Ruta de aprendizaje recomendada:**

1. **[01-CONCEPTOS-FUNDAMENTALES.md](01-CONCEPTOS-FUNDAMENTALES.md)** - Entender el paradigma TNFR
2. **[GLOSARIO.md](GLOSARIO.md)** - Familiarizarse con términos clave
3. **[03-OPERADORES-Y-GLIFOS.md](03-OPERADORES-Y-GLIFOS.md)** - Conocer los 13 operadores
4. **[examples/01-basico-bootstrap.py](examples/01-basico-bootstrap.py)** - Ejecutar primer ejemplo
5. **[08-REFERENCIA-RAPIDA.md](08-REFERENCIA-RAPIDA.md)** - Tener a mano durante desarrollo

**Tiempo total:** ~2 horas para fundamentos operacionales

### Para Desarrolladores Intermedios

**Si ya conoces TNFR y quieres implementar secuencias:**

1. **[04-SECUENCIAS-VALIDAS.md](04-SECUENCIAS-VALIDAS.md)** - Patrones y anti-patrones
2. **[02-RESTRICCIONES-CANONICAS.md](02-RESTRICCIONES-CANONICAS.md)** - Restricciones U1-U4
3. **[examples/](examples/)** - Ejecutar ejemplos intermedios y avanzados
4. **[08-REFERENCIA-RAPIDA.md](08-REFERENCIA-RAPIDA.md)** - Consulta rápida

**Tiempo total:** ~90 minutos

### Para Contribuidores Avanzados

**Si vas a modificar el core o agregar features:**

1. **[05-IMPLEMENTACION-TECNICA.md](05-IMPLEMENTACION-TECNICA.md)** - Arquitectura del código
2. **[06-VALIDACION-Y-TESTING.md](06-VALIDACION-Y-TESTING.md)** - Estrategia de tests
3. **[INDICE-MAESTRO.md](INDICE-MAESTRO.md)** - Mapa conceptual del sistema
4. **[07-MIGRACION-Y-EVOLUCION.md](07-MIGRACION-Y-EVOLUCION.md)** - Cómo evolucionar el sistema
5. **[schemas/](schemas/)** - Schemas para validación

**Tiempo total:** ~2-3 horas para dominio completo

---

## 🔗 Referencias Externas

### Documentación del Repositorio Principal

- **[../../README.md](../../README.md)** - Overview del proyecto TNFR
- **[../../UNIFIED_GRAMMAR_RULES.md](../../UNIFIED_GRAMMAR_RULES.md)** - Derivaciones formales completas (fuente original)
- **[../../AGENTS.md](../../AGENTS.md)** - Invariantes canónicas y contratos
- **[../../GLOSSARY.md](../../GLOSSARY.md)** - Glosario general del proyecto
- **[../../TNFR.pdf](../../TNFR.pdf)** - Fundamentos teóricos completos

### Implementación

- **[../../src/tnfr/operators/grammar.py](../../src/tnfr/operators/grammar.py)** - Implementación canónica
- **[../../src/tnfr/operators/definitions.py](../../src/tnfr/operators/definitions.py)** - Definición de operadores
- **[../../tests/unit/operators/test_unified_grammar.py](../../tests/unit/operators/test_unified_grammar.py)** - Suite de tests

---

## 📝 Convenciones de Escritura

### Formato

- **Bilingüe:** Español para narrativa, inglés para términos técnicos (EPI, νf, ΔNFR)
- **Ecuaciones:** Notación matemática estándar con LaTeX
- **Código:** Python 3.9+ con type hints
- **Referencias:** Links relativos internos, absolutos para externos

### Estructura de Secciones

Cada documento técnico sigue esta estructura:

```markdown
# Título del Documento

## Propósito
[Para qué sirve este documento]

## Conceptos Clave
[Prerequisitos necesarios]

## Contenido Principal
[Desarrollo con subsecciones]

## Ejemplos
[Código ejecutable]

## Referencias
[Links a otros documentos]
```

### Código

Todos los ejemplos de código deben:
- ✅ Ser ejecutables
- ✅ Incluir imports completos
- ✅ Tener comentarios explicativos
- ✅ Seguir convenciones TNFR (no modificar EPI directamente, etc.)
- ✅ Incluir telemetry output esperado

---

## 🤝 Contribuir a esta Documentación

### Principios

1. **Una fuente de verdad:** No duplicar información, cross-referenciar
2. **Física primero:** Toda documentación debe derivar de TNFR physics
3. **Incremental:** Agregar sin romper estructura existente
4. **Validable:** Ejemplos ejecutables, schemas JSON actualizables

### Agregar Nuevo Contenido

**Para agregar nueva restricción:**
1. Documentar física en `02-RESTRICCIONES-CANONICAS.md`
2. Implementar en `../../src/tnfr/operators/grammar.py`
3. Agregar tests en `../../tests/unit/operators/test_unified_grammar.py`
4. Actualizar `schemas/restricciones-u1-u4.json`
5. Agregar ejemplos en `examples/`
6. Actualizar `08-REFERENCIA-RAPIDA.md`

**Para agregar nuevo operador:**
1. Documentar en `03-OPERADORES-Y-GLIFOS.md`
2. Implementar en `../../src/tnfr/operators/definitions.py`
3. Actualizar clasificación en `../../src/tnfr/operators/grammar.py`
4. Agregar tests de contrato
5. Actualizar `schemas/operadores-canonicos.json`

### Mantener Coherencia

**Antes de hacer PR:**
- [ ] Todos los ejemplos son ejecutables
- [ ] Links bidireccionales funcionan
- [ ] Schemas JSON reflejan cambios
- [ ] Tests pasan
- [ ] Cambios documentados en 07-MIGRACION-Y-EVOLUCION.md si hay breaking changes

---

## 📊 Estado de Completitud

### ✅ Completo
- Estructura de directorios
- README de navegación (este archivo)
- Cross-references principales

### 🚧 En Progreso
- 01-CONCEPTOS-FUNDAMENTALES.md
- 02-RESTRICCIONES-CANONICAS.md
- 03-OPERADORES-Y-GLIFOS.md
- 04-SECUENCIAS-VALIDAS.md
- 05-IMPLEMENTACION-TECNICA.md
- 06-VALIDACION-Y-TESTING.md
- 07-MIGRACION-Y-EVOLUCION.md
- 08-REFERENCIA-RAPIDA.md

### 📋 Planificado
- GLOSARIO.md (consolidar desde ../../GLOSSARY.md)
- INDICE-MAESTRO.md
- examples/*.py
- schemas/*.json

---

## 🎓 Filosofía de esta Documentación

> **"Si un cambio no puede ser trazado desde física TNFR hasta código hasta tests, no es canonical."**

Esta documentación existe para hacer esa trazabilidad **explícita, navegable y mantenible**.

### Valores

- **Claridad sobre brevedad:** Mejor explicar dos veces que dejar dudas
- **Física sobre convención:** Cada regla deriva inevitablemente de ecuación nodal
- **Código sobre prosa:** Ejemplos ejecutables > descripciones abstractas
- **Testing sobre confianza:** Todo lo documentado debe ser testeable

---

## 📞 Contacto y Soporte

**¿Encontraste inconsistencias?**
- Abre issue en GitHub con label `documentation`

**¿Necesitas ayuda navegando?**
- Revisa primero [08-REFERENCIA-RAPIDA.md](08-REFERENCIA-RAPIDA.md)
- Luego consulta el documento específico según tu nivel

**¿Quieres contribuir?**
- Lee [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
- Luego revisa sección "Contribuir a esta Documentación" arriba

---

<div align="center">

**Versión:** 1.0  
**Última actualización:** 2025-11-10  
**Mantenedor:** TNFR Core Team

**Reality is not made of things—it's made of resonance. Document accordingly.**

</div>
