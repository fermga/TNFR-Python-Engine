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

## Operators in `tnfr.operators.definitions`

Operator classes describe how they reorganise coherence when applied to a
node. Each docstring must explain the **structural effect** (what the operator
does to EPI, ΔNFR, νf, and phase) before covering implementation notes. Use the
following template when adding or updating operators:

```python
"""State the operator's structural effect in one sentence."""

class Emission(Operator):
    """Boost ΔNFR towards positive expansion while preserving phase locks."""

    def __call__(self, graph, node, /, *, intensity=1.0):
        """Apply the emission pulse to increase νf and ΔNFR coherently.

        Parameters
        ----------
        graph : TNFRGraph
            Graph containing the node; carries EPI, νf, ΔNFR, and phase data.
        node : Hashable
            Node receiving the emission. Document how the pulse alters its EPI.
        intensity : float, default 1.0
            Scales the emission; explain expected ΔNFR growth and phase guardrails.

        Returns
        -------
        TNFRGraph
            The updated graph. Describe telemetry adjustments (ΔNFR hooks, phase).

        Examples
        --------
        >>> from tnfr import operators, structural
        >>> G, node = structural.create_nfr("seed", epi=0.3, vf=1.0)
        >>> op = operators.definitions.Emission(intensity=0.5)
        >>> G = op(G, node)
        >>> round(G.nodes[node]["vf"], 2)
        1.0
        >>> G.nodes[node]["phase"]
        ...  # Illustrate phase guard and ΔNFR change (ΔNFR > 0).
        """
```

**Reminders**

- Highlight how ΔNFR shifts (sign, magnitude) and how νf and phase react.
- Show an example where ΔNFR, νf, and phase are logged or asserted so reviewers
  can trace the structural impact.

## Metrics in `tnfr.metrics.sense_index`

Metrics expose how coherence and sensing change over time. Docstrings must
clarify how the metric reads ΔNFR and νf to produce a sense index while
referencing phase as needed. Start from this template:

```python
def sense_index(graph, node):
    """Compute Si by mapping ΔNFR, νf, and phase to a stability score.

    Parameters
    ----------
    graph : TNFRGraph
        Graph with stored ΔNFR traces, νf (Hz_str), and phase for each node.
    node : Hashable
        Node whose sensory coherence is measured. Describe expected ΔNFR bounds.

    Returns
    -------
    float
        Sense index (Si). Explain how ΔNFR, νf, and phase contribute to the
        final value.

    Examples
    --------
    >>> from tnfr import metrics, structural
    >>> G, node = structural.create_nfr("seed", epi=0.8, vf=1.5)
    >>> G.nodes[node]["phase"] = 0.0
    >>> G.nodes[node]["delta_nfr"] = 0.12
    >>> metrics.sense_index.sense_index(G, node)
    0.92
    >>> # Detail how ΔNFR and νf shifts change the score.
    """
```

**Reminders**

- Include a brief explanation of structural phase handling (e.g., phase drift
  lowering Si).
- Document telemetry expectations (ΔNFR traces or νf history) in the parameters
  or returns section.
- Ensure examples demonstrate how ΔNFR, νf, and phase values are set before the
  metric call.

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
