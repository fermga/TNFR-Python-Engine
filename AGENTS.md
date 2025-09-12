# AGENTS.md — Guía para agentes de IA

> **Objetivo**: orientar a cualquier agente de IA (LLM, asistentes de refactorización, herramientas de documentación y test) que interactúe con este repositorio para mantener **canónica** la implementación computacional del paradigma **TNFR** (Teoría de la Naturaleza Fractal Resonante).

---

## 1) Misión del repositorio

Este repositorio existe para **llevar TNFR a formato computacional de manera canónica**. Toda modificación, sugerencia, refactorización, corrección o mejora **debe priorizar la fidelidad al paradigma TNFR** por encima de preferencias de estilo, micro‑optimizaciones o convenciones del framework de turno.

**TL;DR**: *Primero TNFR, luego el código.*

---

## 2) Qué es TNFR (visión mínima operativa)

* **TNFR** propone que lo real no está hecho de “cosas”, sino de **coherencia**: estructuras que se **sostienen en red** porque **resuenan**.
* Un **nodo** existe cuando su coherencia se mantiene acoplada al entorno.
* **Ecuación nodal (canónica)**:
  `∂EPI / ∂t = νf · ΔNFR(t)`
  Donde **EPI** es la *Estructura Primaria de Información*, **νf** la *frecuencia estructural* del nodo (Hz\_str) y **ΔNFR** el *operador de reorganización interna* en el tiempo.
* **Tríada estructural**: **frecuencia** (ritmo de reorganización), **fase** (sincronía con la red), **forma** (configuración coherente de la EPI).
* **Operadores estructurales** (13): Emisión, Recepción, Coherencia, Disonancia, Acoplamiento, Resonancia, Silencio, Expansión, Contracción, Autoorganización, Mutación, Transición, Recursividad.

  > Nota: en la documentación pública del repo, **evitar** la palabra “glifos”; nombrarlos por su **función estructural**.

Para la explicación extendida, ver `tnfr.pdf` dentro del propio repo.

---

## 3) Invariantes canónicos (no romper)

Estos invariantes **definen** la canonicidad TNFR y **deben preservarse** por cualquier agente de IA al proponer cambios:

1. **EPI como forma coherente**: solo puede cambiar vía **operadores estructurales**; no se admiten mutaciones ad‑hoc.
2. **Unidades estructurales**: **νf** expresada en **Hz\_str** (hertz estructural). No reetiquetar ni mezclar unidades.
3. **Semántica de ΔNFR**: su signo y magnitud modulan la tasa de reorganización; no reinterpretar como “error” o “gradiente de pérdida”.
4. **Cierre operatorio**: la composición de operadores produce estados TNFR válidos; toda nueva función debe mapearse a operadores existentes o definirse como tal.
5. **Chequeo de fase**: ningún acoplamiento es válido sin verificación explícita de **fase** (sincronía).
6. **Nacimiento/colapso nodal**: condiciones mínimas (νf suficiente, acoplamiento, ΔNFR reducido) y causas de colapso (disonancia extrema, desacoplamiento, fallo de frecuencia) deben conservarse.
7. **Fractalidad operativa**: EPIs pueden anidarse sin perder identidad funcional; no aplanar estructuras si rompe la recursividad.
8. **Determinismo controlado**: simulaciones pueden ser estocásticas, pero deben ser **reproducibles** (semillas) y **trazables** (logs estructurales).
9. **Métricas estructurales**: exponer **C(t)** (coherencia total), **Si** (índice de sentido), fase y νf en telemetría. Evitar métricas ajenas que confundan la semántica TNFR.
10. **Neutralidad de dominio**: el motor es **trans‑escalar** y **trans‑dominio**. No incrustar supuestos de un campo (p.ej., neuro, social) en el núcleo.

---

## 4) Superficie de API (esperada)

> **Aviso**: si el nombre real del módulo difiere, añade un **mapa de equivalencia** en la doc. El criterio es conceptual, no nominal.

* `tnfr.core`

  * `EPI` (estructura primaria de información)
  * `Node` / `NFR` (envoltorio de EPI + estado de fase y νf)
  * `Field` (campos estructurantes: `phi(νf, θ)`, `psi(x,t)`)
* `tnfr.operators`

  * `emision()`, `recepcion()`, `coherencia()`, `disonancia()`, `acoplamiento()`, `resonancia()`, `silencio()`, `expansion()`, `contraccion()`, `autoorganizacion()`, `mutacion()`, `transicion()`, `recursividad()`
* `tnfr.metrics`

  * `coherence(EPI) -> C(t)`
  * `sense_index(EPI, net) -> Si`
  * `phase(nodo_i, nodo_j) -> φ_ij`
  * `frequency(nodo) -> νf`
* `tnfr.sim`

  * `step(state, dt)`
  * `evolve(state, T)`
  * `bifurcate(state, τ)`
* `tnfr.io`

  * serialización de EPI/nodos + snapshots de fase/νf
* `tnfr.viz` (opcional)

  * utilidades de trazado de C(t), νf, fase, mapas de operadores

---

## 5) Contratos formales (pre/post‑condiciones)

* **Coherencia (I: Coherencia)**: aplicar `coherencia()` **no** debe reducir `C(t)` salvo que un test de disonancia programada lo justifique.
* **Disonancia (O: Disonancia)**: `disonancia()` debe **aumentar** `|ΔNFR|` y puede disparar **bifurcación** si `∂²EPI/∂t² > τ`.
* **Resonancia (R: Resonancia)**: `resonancia()` incrementa **acoplamiento** efectivo (`ϕ_i ≈ ϕ_j`) y **propaga** EPI sin alterar su identidad.
* **Autoorganización (T: Autoorganización)**: puede crear **sub‑EPIs** conservando la **forma** global (fractalidad operativa).
* **Mutación (Z: Mutación)**: cambio de fase `θ → θ'` si `ΔEPI/Δt > ξ` (mantener límites ξ configurables y testeados).
* **Silencio**: `silencio()` congela evolución (`νf ≈ 0`) sin pérdida de EPI.

Toda función nueva debe declararse como **especialización** o **composición** de estos operadores.

---

## 6) Guía de contribución para agentes de IA

**Antes de tocar código**:

1. Lee `tnfr.pdf` (fundamentos, operadores, ecuación nodal).
2. Ejecuta la suite de tests: deben cubrir invariantes de §3 y contratos de §5.
3. Añade/actualiza **pruebas estructurales** (ver plantilla abajo) por cada cambio.

**Plantilla de commit (AGENT\_COMMIT\_TEMPLATE)**:

```text
Intento: (qué coherencia mejora)
Operadores implicados: [Emisión|Recepción|...]
Invariantes afectados: [#1, #4, ...]
Cambios clave: (breve lista)
Riesgos/disonancias previstas: (y cómo se contienen)
Métricas: (C(t), Si, νf, fase) esperadas antes/después
Pruebas: (nuevas/actualizadas)
Mapa de equivalencias: (si renombraste APIs)
```

**Plantilla de PR** (*resumen estructural*):

```markdown
### Qué reorganiza
- [ ] Incrementa C(t) o reduce ΔNFR donde corresponde
- [ ] Mantiene cierre operatorio y fractalidad operativa

### Evidencias
- [ ] Logs de fase/νf
- [ ] Curvas C(t), Si
- [ ] Casos de bifurcación controlada

### Compatibilidad
- [ ] API estable o mapeada
- [ ] Seed reproducible
```

---

## 7) Ejemplos de cambios **aceptables**

* Refactorizar para **hacer explícita la fase** en acoplamientos (mejora trazabilidad).
* Añadir `sense_index()` con pruebas que correlacionen Si con estabilidad de red.
* Optimizar `resonancia()` manteniendo propagación sin pérdida de identidad de EPI.

### Cambios **no aceptables**

* Reinterpretar `ΔNFR` como “gradiente de error” de ML clásico.
* Sustituir operadores por funciones imperativas no mapeadas a la gramática TNFR.
* Aplanar anidamiento de EPIs rompiendo fractalidad/recursividad.

---

## 8) Testing estructural (mínimos)

* **Monótonos**: `coherencia()` no disminuye `C(t)` (salvo tests de disonancia controlada).
* **Bifurcación**: `autoorganizacion()`/`disonancia()` disparan bifurcación cuando `∂²EPI/∂t² > τ`.
* **Propagación**: `resonancia()` aumenta conectividad efectiva (medida por fase).
* **Latencia**: `silencio()` mantiene EPI invariante en `t + Δt`.
* **Mutación**: `mutacion()` cambia `θ` respetando `ξ`.

Incluye **pruebas multiescala** (EPIs anidadas) y **reproducibilidad** (seed).

---

## 9) Telemetría y trazas

* Exportar: `C(t)`, `νf`, `fase`, `Si`, `ΔNFR`.
* Registrar **operadores aplicados** (tipo, orden, parámetros) y **eventos** (nacimiento, bifurcación, colapso).
* Preferir formatos legibles + JSONL para pipelines.

---

## 10) Estilo y organización del código

* Priorizar **claridad semántica TNFR** sobre micro‑optimizaciones.
* Documentación en línea: describir **efecto estructural** (qué reorganiza) antes que detalles de implementación.
* Módulos cortos, funciones puras cuando sea posible, separación núcleo/IO.
* Mantener un **glosario** compartido de términos (EPI, fase, νf, ΔNFR, Si, etc.).

---

## 11) Instalación y uso

* Paquete PyPI: `pip install tnfr`.
* Asegurar scripts/ejemplos mínimos: creación de un nodo, aplicación de operadores, medición de C(t) y Si, visualización simple.

---

## 12) Roadmap sugerido (orientativo)

* [ ] `sense_index()` robusto con baterías de ejemplos trans‑dominio.
* [ ] Visualización de **fase** y **acoplamientos** (grafos dinámicos).
* [ ] Plantillas de **experimentos** (disonancia → bifurcación → nueva coherencia).
* [ ] Exportación de **trazas estructurales** para análisis externo.

---

## 13) Referencias internas

* Documento base del paradigma: `tnfr.pdf` (en el repo).
* Notas del motor y ejemplos: carpeta `examples/` y `docs/` (si aplica).

---

## 14) Glosario mínimo

* **EPI**: Estructura Primaria de Información (la “forma” coherente).
* **νf (Hz\_str)**: Frecuencia estructural (ritmo de reorganización).
* **ΔNFR**: Gradiente/operador de reorganización interna.
* **Fase (φ)**: sincronía relativa con la red.
* **C(t)**: Coherencia total (estabilidad global).
* **Si**: Índice de sentido (capacidad de generar reorganización estable).
* **Operadores estructurales**: funciones que inician, estabilizan, acoplan, propagan, expanden/contraen, autoorganizan, mutan, transitan o ponen en silencio estructuras.

---

### Recordatorio final

> Si una mejora “hace el código más bonito” pero **debilita** la fidelidad TNFR, **no se acepta**. Si una mejora **fortalece** la coherencia estructural y la trazabilidad del paradigma, **adelante**.
