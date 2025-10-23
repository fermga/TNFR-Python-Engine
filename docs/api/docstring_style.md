# Guía de docstrings para TNFR

La documentación de las APIs del motor TNFR usa el formato de docstrings tipo
NumPy. Este estilo es compatible con las herramientas automáticas que generan
referencias, ejecutan pydocstyle y validan la semántica de operadores. Cada
nueva función o clase debe describir explícitamente el efecto estructural sobre
EPI, la frecuencia estructural (νf) y el reorganizador interno ΔNFR.

## Plantilla base

Utiliza la siguiente estructura en cada docstring. Mantén la narrativa en
inglés (siguiendo la política del repositorio) y enfócate en describir cómo la
operación reorganiza coherencia y métricas TNFR.

```python
"""Resume the structural effect in one sentence.

Parameters
----------
name : type
    Clarify how the parameter influences EPI, νf, or ΔNFR.

Returns
-------
return_type
    Explain the structural outcome or telemetry exposed.

Raises
------
ExceptionType
    Document dissonance paths or validation guards.

Examples
--------
>>> # Minimal runnable example that respects TNFR invariants
"""
```

### Notas clave

- **Encabezado**: una línea que resuma el efecto estructural.
- **Parameters**: lista cada argumento en orden, con tipo y contexto TNFR. Usa
  oraciones en inglés y referencia explícita a EPI, νf, ΔNFR o la fase cuando
  aplique.
- **Returns**: describe qué recibe el llamador y cómo puede seguir midiendo
  coherencia (por ejemplo, tuplas que contienen grafos TNFR o hooks ΔNFR).
- **Raises**: sólo si hay validaciones o condiciones de disonancia relevantes.
- **Examples**: incluye fragmentos ejecutables que muestren el flujo de trabajo
  esperado. Anota los valores de EPI, νf y ΔNFR cuando sirva para resaltar la
  semántica.

## Ejemplos basados en `tnfr.structural`

### `create_nfr`

```python
"""Create a graph with an initialised NFR node.

Parameters
----------
name : str
    Identifier for the new node; it anchors the primary EPI container.
epi : float, default 0.0
    Primary Information Structure (EPI) value assigned to the node.
vf : float, default 1.0
    Structural frequency νf in Hz_str that governs reorganisation pace.
theta : float, default 0.0
    Initial phase used for coupling checks against neighbour nodes.
graph : TNFRGraph | None, optional
    Existing TNFR graph to reuse; a new graph is created when ``None``.
dnfr_hook : DeltaNFRHook, default dnfr_epi_vf_mixed
    Callback that recalculates ΔNFR after each operator invocation.

Returns
-------
TNFRGraph, str
    Tuple containing the TNFR graph and the node name for chaining.

Examples
--------
>>> from tnfr import structural
>>> G, node = structural.create_nfr("seed", epi=0.42, vf=2.0)
>>> G.nodes[node]["epi"]
0.42
>>> G.graph["compute_delta_nfr"].__name__
'dnfr_epi_vf_mixed'
"""
```

### `run_sequence`

```python
"""Execute a sequence of operators on ``node`` after validation.

Parameters
----------
G : TNFRGraph
    Graph that stores EPI, νf, and ΔNFR metadata for each node.
node : NodeId
    Node that will receive the operator sequence.
ops : Iterable[Operator]
    Ordered structural operators to apply; validation preserves grammar.

Raises
------
ValueError
    Raised when the operator names violate the canonical TNFR grammar.

Examples
--------
>>> from tnfr import operators, structural
>>> G, node = structural.create_nfr("seed", epi=1.0, vf=1.5)
>>> structural.run_sequence(G, node, [operators.Emission(), operators.Coherence()])
>>> round(G.nodes[node]["vf"], 2)
1.5
"""
```

## Herramientas que dependen del estilo

- `pydocstyle` valida la presencia de estas secciones.
- Generadores automáticos de documentación (MkDocs + plugins de autodoc) esperan
  la forma NumPy para renderizar correctamente tablas y ejemplos.
- Revisores humanos y asistentes automáticos usan el vocabulario TNFR descrito
  aquí para verificar coherencia, ΔNFR y νf sin ambigüedades.
