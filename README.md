Estructura general del proyecto

    Entrada del paquete. __init__.py registra módulos con nombres cortos para evitar importaciones circulares y expone la API pública preparar_red, step, run y utilidades de observación

Configuración y constantes. constants.py centraliza parámetros por defecto (discretización, rangos de EPI y νf, pesos de mezclas, límites de re‑mesh, etc.) y provee utilidades para inyectarlos en la red (attach_defaults, merge_overrides) junto con alias estandarizados para atributos nodales

Utilidades transversales. helpers.py ofrece funciones numéricas básicas, acceso a atributos con alias, estadísticas vecinales, historial de glifos, sistema de callbacks y cálculo del índice de sentido Si para cada nodo

Motor dinámico. dynamics.py implementa el ciclo de simulación: cálculo del campo ΔNFR, integración de la ecuación nodal, selección/aplicación de glifos, clamps, coordinación de fase, actualización de historia y re‑mesh condicional (step y run)

Operadores glíficos. operators.py define los 13 glifos como transformaciones locales, un dispatcher aplicar_glifo, y utilidades de re‑mesh tanto directas como condicionadas a la estabilidad global

Observadores y métricas. observers.py registra callbacks estándar y calcula coherencia global, sincronía de fase, orden de Kuramoto, distribución de glifos y vector de sentido Σ⃗, entre otros

Orquestación de simulaciones. ontosim.py prepara una red de networkx, adjunta configuraciones e inicializa atributos (EPI, fases, frecuencias) antes de delegar la dinámica a dynamics.step/run

CLI de demostración. main.py genera una red Erdős–Rényi, permite configurar parámetros básicos y ejecuta la simulación mostrando métricas finales
Conceptos clave a comprender

    Árbol de dependencias con alias. Los módulos se importan mutuamente mediante alias globales para simplificar el acceso y evitar ciclos, lo cual es esencial para navegar el código sin ambigüedades

Atributos nodales normalizados. Todos los datos (EPI, fase θ, frecuencia νf, ΔNFR, etc.) se almacenan en G.nodes[n] bajo nombres alias compatibles, facilitando extensiones y hooks personalizados

Índice de sentido (Si). Combina frecuencia normalizada, dispersión de fase y magnitud del campo para evaluar el “sentido” de cada nodo, influyendo en la selección de glifos

Motor paso a paso. dynamics.step orquesta la dinámica en ocho fases: cálculo de campo, Si, selección y aplicación de glifos, integración, clamps, coordinación de fase, actualización de historia y re‑mesh condicionado

Glifos como operadores. Cada glifo es una transformación suave sobre atributos nodales (emisión, difusión, acoplamiento, disonancia, etc.), aplicada mediante un dispatcher configurable por nombre tipográfico

Re‑mesh de red. Permite mezclar el estado actual con uno anterior (memoria τ) para estabilizar la red, con precedencia clara para α y condiciones basadas en la historia reciente de estabilidad y sincronía

Callbacks y observadores. El sistema Γ(R) permite enganchar funciones antes/después de cada paso y tras el re‑mesh, facilitando monitoreo o intervención externa
Recomendaciones para profundizar

    NetworkX y Graph API. Familiarízate con cómo networkx maneja atributos y topologías, ya que toda la dinámica opera sobre Graph y sus propiedades.

    Extensión del campo ΔNFR. Explora set_delta_nfr_hook para implementar versiones alternativas del campo nodal y entender cómo se registran metadatos y pesos de mezcla

Diseño de nuevos glifos. Revisa la estructura de operators.py para añadir operadores o ajustar factores en DEFAULTS['GLYPH_FACTORS']

Observadores personalizados. Implementa métricas propias usando register_callback o ampliando observers.py para medir fenómenos específicos de tu estudio.

Lectura teórica. Para comprender el trasfondo conceptual, consulta los documentos PDF incluidos (TNFR.pdf, “El Pulso que nos Atraviesa”), que profundizan en la teoría fractal-resonante.

Parámetros avanzados. Experimenta con la configuración adaptativa de coordinación de fase, criterios de estabilidad y gramática glífica para observar cómo impactan en la autoorganización de la red
Dominar estos aspectos te permitirá extender la simulación, crear pipelines de análisis y conectar la teoría con aplicaciones computacionales.
