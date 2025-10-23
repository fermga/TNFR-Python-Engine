# TNFR docstring guide

TNFR engine APIs use NumPy-style docstrings. The format integrates with the
automatic reference builders, pydocstyle, and the operator semantics linters
that backstop the project. Every new function or class must describe the
structural effect on the Primary Information Structure (EPI), the structural
frequency (νf), and the internal reorganiser ΔNFR. Docstring linting now
requires complete coverage: modules, classes, public functions, and magic
methods must include an appropriate description or, in rare cases, a
``# noqa: Dxxx`` with a clear justification.

## Base template

Follow this structure in each docstring. Keep the narrative in English (per the
repository policy) and focus on how the callable reorganises coherence and TNFR
metrics.

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

### Key notes

- **Summary line**: one sentence that highlights the structural effect.
- **Parameters**: list arguments in order with types and TNFR context. Use
  English sentences and reference EPI, νf, ΔNFR, or phase explicitly when it
  applies.
- **Returns**: describe what the caller receives and how coherence can continue
  to be measured (for example, tuples that include TNFR graphs or ΔNFR hooks).
- **Raises**: only when validations or dissonance conditions are meaningful.
- **Examples**: provide runnable fragments that show the expected workflow.
  Annotate EPI, νf, and ΔNFR values when it clarifies semantics.

## Examples from `tnfr.structural`

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

## Tooling that relies on this style

- `pydocstyle` validates the presence of these sections.
- Automated documentation builders (MkDocs plus autodoc plugins) expect the
  NumPy shape to render tables and examples correctly.
- Human reviewers and automated assistants use the TNFR vocabulary laid out
  here to check coherence, ΔNFR, and νf without ambiguity.
