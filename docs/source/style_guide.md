# Style Guide for Mathematical Notation in TNFR

This document establishes consistent conventions for mathematical notation across all TNFR documentation, code comments, and docstrings.

## 1. Core Principles

1. **Consistency**: Use the same symbol for the same concept throughout
2. **Clarity**: Favor explicit notation over compact when ambiguity may arise
3. **Traceability**: Link notation directly to implementation and theory documents
4. **Accessibility**: Provide both LaTeX and plain-text alternatives where appropriate

---

## 2. Notation Conventions

### 2.1 Vectors and States

**Hilbert space states (ket notation)**:
- States: \(|\psi\rangle\), \(|\text{NFR}\rangle\), \(|i\rangle\)
- Dual states (bra): \(\langle\psi|\), \(\langle j|\)
- Inner product: \(\langle\psi_1|\psi_2\rangle\)
- Outer product: \(|\psi\rangle\langle\phi|\)

**Vector notation**:
- Bold lowercase for vectors: **v**, **r**
- Arrow notation when clarity needed: \(\vec{v}\), \(\vec{r}\)
- Components: \(v_i\) or \(v^i\) (subscript for covariant, superscript for contravariant)

### 2.2 Operators

**Quantum operators (hat notation)**:
- Coherence operator: \(\hat{C}\)
- Frequency operator: \(\hat{J}\) or \(\hat{\nu}_f\)
- Reorganization operator: \(\Delta\text{NFR}\) or \(\hat{\Delta}\)
- Hamiltonian: \(\hat{H}\), \(\hat{H}_{\text{int}}\)
- Unitary evolution: \(\hat{U}(t)\)
- Time evolution: \(\hat{U}(t) = e^{-i\hat{H}t/\hbar}\)

**Operator properties**:
- Adjoint (Hermitian conjugate): \(\hat{A}^\dagger\)
- Commutator: \([\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}\)
- Anticommutator: \(\{\hat{A}, \hat{B}\} = \hat{A}\hat{B} + \hat{B}\hat{A}\)
- Expectation value: \(\langle\hat{A}\rangle = \langle\psi|\hat{A}|\psi\rangle\)

### 2.3 Structural Variables

**Primary quantities** (use these exact symbols):

| Concept | Symbol | LaTeX | Units | Notes |
|---------|--------|-------|-------|-------|
| Primary Information Structure | EPI or \(E\) | `\text{EPI}` or `E` | dimensionless | Use \(\text{EPI}\) in formal derivations |
| Structural Frequency | \(\nu_f\) | `\nu_f` | Hz\_str | Never use \(v_f\) or \(vf\) |
| Reorganization Gradient | \(\Delta\text{NFR}\) | `\Delta\text{NFR}` | dimensionless | May use \(\Delta\) in context |
| Phase | \(\theta\) or \(\phi\) | `\theta` or `\phi` | radians | \(\theta\) preferred for nodal phase |
| Total Coherence | \(C(t)\) | `C(t)` | [0,1] | Time-dependent |
| Sense Index | \(\text{Si}\) or \(S_i\) | `\text{Si}` or `S_i` | [0,1+] | Use \(\text{Si}\) for global, \(S_i\) for node i |

**Derived quantities**:
- Normalized frequency: \(\nu_{f,\text{norm}} = |\nu_f| / \nu_{f,\max}\)
- Phase dispersion: \(\text{disp}_\theta\) or \(\sigma_\theta\)
- Phase mean: \(\bar{\theta}\) or \(\langle\theta\rangle\)
- Coherence matrix element: \(w_{ij}\) or \(C_{ij}\)

### 2.4 Derivatives and Rates

**Time derivatives**:
- Partial: \(\frac{\partial f}{\partial t}\) or \(\partial_t f\)
- Total: \(\frac{df}{dt}\)
- Dot notation: \(\dot{f} = \frac{df}{dt}\) (use sparingly, only for time)

**Spatial derivatives**:
- Gradient: \(\nabla f\) or \(\vec{\nabla} f\)
- Partial: \(\frac{\partial f}{\partial x}\) or \(\partial_x f\)
- Laplacian: \(\nabla^2 f\) or \(\Delta f\)

**Higher-order derivatives**:
- Second time derivative: \(\frac{\partial^2 \text{EPI}}{\partial t^2}\) or \(\ddot{\text{EPI}}\)
- Mixed: \(\frac{\partial^2 f}{\partial x \partial t}\)

### 2.5 Mathematical Spaces

**Space notation**:
- Hilbert space: \(H_{\text{NFR}}\) or \(\mathcal{H}\)
- Banach space: \(B_{\text{EPI}}\)
- Real numbers: \(\mathbb{R}\), \(\mathbb{R}^n\)
- Complex numbers: \(\mathbb{C}\), \(\mathbb{C}^n\)
- Natural numbers: \(\mathbb{N}\)
- Integers: \(\mathbb{Z}\)

**Space operations**:
- Tensor product: \(\otimes\) (e.g., \(H_1 \otimes H_2\))
- Direct sum: \(\oplus\) (e.g., \(V_1 \oplus V_2\))
- Cartesian product: \(\times\) (e.g., \(\mathbb{R}^3 \times \mathbb{R}\))

**Membership and relations**:
- Element of: \(\in\) (e.g., \(x \in \mathbb{R}\))
- Subset: \(\subset\) or \(\subseteq\)
- For all: \(\forall\)
- Exists: \(\exists\)
- Such that: \(:\) or \(|\)

### 2.6 Statistical and Probability Notation

**Expectation and variance**:
- Expectation: \(\mathbb{E}[X]\) or \(\langle X \rangle\)
- Variance: \(\text{Var}(X)\) or \(\sigma^2_X\)
- Standard deviation: \(\sigma_X\)
- Covariance: \(\text{Cov}(X,Y)\)

**Distributions**:
- Normal: \(\mathcal{N}(\mu, \sigma^2)\)
- Uniform: \(\mathcal{U}(a, b)\)
- Probability density: \(p(x)\) or \(\rho(x)\)

---

## 3. Formatting Guidelines

### 3.1 Inline Mathematics in Markdown

Use `\( ... \)` for inline math in Markdown:

```markdown
The structural frequency \(\nu_f\) determines the reorganization rate.
```

**Rendered**: The structural frequency \(\nu_f\) determines the reorganization rate.

### 3.2 Display Equations in Markdown

Use `\[ ... \]` for centered display equations:

```markdown
The nodal equation is:

\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)
\]
```

**Alternative block syntax** (MkDocs/MyST):
```markdown
$$
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)
$$
```

### 3.3 Equations in Python Docstrings (reST)

Use the `.. math::` directive for equations in docstrings:

```python
def compute_Si(G, alpha=0.4, beta=0.3, gamma=0.3):
    r"""Compute the Sense Index for all nodes in the network.
    
    Mathematical Foundation
    -----------------------
    The Sense Index quantifies reorganization stability:
    
    .. math::
        \text{Si} = \alpha \cdot \nu_{f,\text{norm}} 
                  + \beta \cdot (1 - \text{disp}_\theta) 
                  + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})
    
    where:
    
    - :math:`\nu_{f,\text{norm}} = |\nu_f| / \nu_{f,\max}` : Normalized structural frequency
    - :math:`\text{disp}_\theta` : Phase dispersion from neighbors
    - :math:`|\Delta\text{NFR}|_{\text{norm}}` : Normalized reorganization magnitude
    - :math:`\alpha, \beta, \gamma` : Structural weights (sum to 1)
    
    Parameters
    ----------
    G : TNFRGraph
        Network with nodal attributes: `nu_f`, `delta_nfr`, `phase`
    alpha : float, default=0.4
        Weight for frequency component
    beta : float, default=0.3
        Weight for phase synchrony component
    gamma : float, default=0.3
        Weight for reorganization damping component
        
    Returns
    -------
    dict[NodeId, float]
        Sense Index values for each node, range [0, 1+]
        
    See Also
    --------
    compute_Si_node : Single-node Si calculation
    
    References
    ----------
    .. [1] Mathematical Foundations, Section on Coherence Metrics
    .. [2] docs/source/theory/mathematical_foundations.md#sense-index
    
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edge("a", "b")
    >>> G.nodes["a"].update({"nu_f": 0.8, "delta_nfr": 0.1, "phase": 0.0})
    >>> G.nodes["b"].update({"nu_f": 0.6, "delta_nfr": 0.2, "phase": 0.1})
    >>> Si = compute_Si(G)
    >>> round(Si["a"], 2)
    0.85
    """
```

**Key points**:
1. Use `r"""..."""` (raw string) to avoid escaping backslashes
2. Use `.. math::` for display equations
3. Use `:math:`...`` for inline math in parameter descriptions
4. Include "See Also" section with related functions
5. Include "References" section linking to theory docs

### 3.4 Multi-line Equations

**Aligned equations** (use `&` for alignment):

```markdown
\[
\begin{aligned}
s_{\text{phase}}(i,j) &= \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right) \\
s_{\text{EPI}}(i,j) &= 1 - \frac{|\text{EPI}_i - \text{EPI}_j|}{\Delta_{\text{EPI}}} \\
s_{\nu_f}(i,j) &= 1 - \frac{|\nu_{f,i} - \nu_{f,j}|}{\Delta_{\nu_f}}
\end{aligned}
\]
```

**In docstrings**:
```python
r"""
.. math::
    \begin{aligned}
    s_{\text{phase}}(i,j) &= \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right) \\
    s_{\text{EPI}}(i,j) &= 1 - \frac{|\text{EPI}_i - \text{EPI}_j|}{\Delta_{\text{EPI}}} \\
    s_{\nu_f}(i,j) &= 1 - \frac{|\nu_{f,i} - \nu_{f,j}|}{\Delta_{\nu_f}}
    \end{aligned}
"""
```

### 3.5 Conditional Expressions

Use cases for piecewise functions:

```markdown
\[
f(x) = \begin{cases}
  x^2 & \text{if } x \geq 0 \\
  -x^2 & \text{if } x < 0
\end{cases}
\]
```

---

## 4. Cross-Referencing

### 4.1 Linking Documentation to Code

**In documentation, reference Python functions**:

```markdown
The similarity components are computed by 
[`compute_wij_phase_epi_vf_si()`](../../src/tnfr/metrics/coherence.py)
```

**In docstrings, use Sphinx cross-references**:

```python
"""
See :func:`tnfr.metrics.coherence.compute_wij_phase_epi_vf_si` for implementation details.
"""
```

### 4.2 Linking Code to Theory

**In module docstrings**:

```python
"""
Mathematical Foundation
-----------------------
See `docs/source/theory/mathematical_foundations.md#31-coherence-operator-ĉ`
for complete theoretical derivation.

Implementation Map
------------------
- :func:`coherence_matrix` → Constructs :math:`W \approx \hat{C}`
- :func:`compute_coherence` → Computes :math:`C(t) = \text{Tr}(\hat{C}\rho)`
- :func:`compute_wij_phase_epi_vf_si` → Matrix elements :math:`w_{ij} \approx \langle i | \hat{C} | j \rangle`
"""
```

### 4.3 Theory to Implementation Table

Include mapping tables in theory documents:

```markdown
| Theoretical Concept | Symbol | Implementation | File |
|---------------------|--------|----------------|------|
| Coherence operator | \(\hat{C}\) | `coherence_matrix()` | `metrics/coherence.py` |
| Matrix element | \(w_{ij}\) | `compute_wij_phase_epi_vf_si()` | `metrics/coherence.py` |
| Total coherence | \(C(t)\) | `compute_coherence()` | `metrics/common.py` |
| Sense Index | \(\text{Si}\) | `compute_Si()` | `metrics/sense_index.py` |
```

---

## 5. Common Equations Reference

### 5.1 The Nodal Equation

**Canonical form**:
\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)
\]

**Implementation**: See `src/tnfr/dynamics/`

### 5.2 Coherence Operator

**Matrix element**:
\[
w_{ij} \approx \langle i | \hat{C} | j \rangle
\]

**Combined similarity**:
\[
w_{ij} = w_{\text{phase}} \cdot s_{\text{phase}} + w_{\text{EPI}} \cdot s_{\text{EPI}} + w_{\nu_f} \cdot s_{\nu_f} + w_{\text{Si}} \cdot s_{\text{Si}}
\]

**Implementation**: `tnfr.metrics.coherence.compute_wij_phase_epi_vf_si()`

### 5.3 Sense Index

**Definition**:
\[
\text{Si} = \alpha \cdot \nu_{f,\text{norm}} + \beta \cdot (1 - \text{disp}_\theta) + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})
\]

where:
- \(\nu_{f,\text{norm}} = |\nu_f| / \nu_{f,\max}\)
- \(\text{disp}_\theta = |\theta - \bar{\theta}| / \pi\)
- \(|\Delta\text{NFR}|_{\text{norm}} = |\Delta\text{NFR}| / \Delta\text{NFR}_{\max}\)

**Implementation**: `tnfr.metrics.sense_index.compute_Si()`

### 5.4 Phase Synchrony

**Kuramoto order parameter**:
\[
r e^{i\Psi} = \frac{1}{N}\sum_{j=1}^N e^{i\theta_j}
\]

where \(r \in [0,1]\) is the synchronization strength.

**Implementation**: `tnfr.observers.kuramoto_order()`

---

## 6. Examples of Good Practice

### 6.1 Well-Documented Function

```python
def compute_similarity_phase(theta_i: float, theta_j: float) -> float:
    r"""Compute phase similarity between two nodes.
    
    Mathematical Definition
    -----------------------
    The phase similarity is defined as:
    
    .. math::
        s_{\text{phase}}(i,j) = \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right)
    
    This measures how synchronized two nodes are, with:
    
    - :math:`s_{\text{phase}} = 1` : Perfect synchrony (:math:`\theta_i = \theta_j`)
    - :math:`s_{\text{phase}} = 0.5` : Orthogonal (:math:`|\theta_i - \theta_j| = \pi/2`)
    - :math:`s_{\text{phase}} = 0` : Anti-phase (:math:`|\theta_i - \theta_j| = \pi`)
    
    Parameters
    ----------
    theta_i : float
        Phase of node i in radians, range :math:`[0, 2\pi)`
    theta_j : float
        Phase of node j in radians, range :math:`[0, 2\pi)`
    
    Returns
    -------
    float
        Phase similarity in range [0, 1]
    
    See Also
    --------
    compute_wij_phase_epi_vf_si : Combined similarity computation
    
    References
    ----------
    .. [1] Mathematical Foundations, §3.1.1
    
    Examples
    --------
    >>> import math
    >>> compute_similarity_phase(0.0, 0.0)  # Same phase
    1.0
    >>> compute_similarity_phase(0.0, math.pi)  # Anti-phase
    0.0
    """
    return 0.5 * (1.0 + math.cos(theta_i - theta_j))
```

### 6.2 Well-Documented Module

```python
"""Coherence metrics for TNFR networks.

This module implements the coherence operator :math:`\hat{C}` and related
metrics for measuring structural stability in resonant networks.

Mathematical Foundation
-----------------------

The coherence operator :math:`\hat{C}` is a Hermitian operator on the Hilbert
space :math:`H_{\text{NFR}}` with spectral decomposition:

.. math::
    \hat{C} = \sum_i \lambda_i |\phi_i\rangle\langle\phi_i|

where :math:`\lambda_i \geq 0` are coherence eigenvalues and :math:`|\phi_i\rangle`
are coherence eigenstates.

In the discrete node basis :math:`\{|i\rangle\}`, matrix elements are approximated:

.. math::
    w_{ij} \approx \langle i | \hat{C} | j \rangle

The total coherence is the trace:

.. math::
    C(t) = \text{Tr}(\hat{C}\rho) = \sum_i w_{ii} \rho_i

See `docs/source/theory/mathematical_foundations.md#31-coherence-operator-ĉ`
for complete theoretical derivation.

Implementation Map
------------------

Core Functions:

- :func:`coherence_matrix` : Constructs :math:`W \approx \hat{C}` matrix
- :func:`compute_coherence` : Computes :math:`C(t)` from graph
- :func:`compute_wij_phase_epi_vf_si` : Matrix elements :math:`w_{ij}`

Helper Functions:

- :func:`_combine_similarity` : Weighted combination of similarity components
- :func:`_compute_wij_phase_epi_vf_si_vectorized` : Vectorized computation

Examples
--------

Basic coherence computation:

>>> import networkx as nx
>>> from tnfr.metrics.coherence import compute_coherence
>>> G = nx.Graph()
>>> G.add_edge("a", "b")
>>> G.nodes["a"].update({"EPI": 0.5, "nu_f": 0.8, "phase": 0.0, "Si": 0.7})
>>> G.nodes["b"].update({"EPI": 0.6, "nu_f": 0.7, "phase": 0.1, "Si": 0.8})
>>> C = compute_coherence(G)
>>> 0 <= C <= 1
True

References
----------

.. [1] TNFR Mathematical Formalization, Section 2.1
.. [2] docs/source/theory/coherence_operator.md
"""
```

---

## 7. Anti-Patterns to Avoid

### 7.1 Inconsistent Symbols

❌ **Bad**: Mixing symbols for the same concept
```python
# In one place
vf = 0.8  # structural frequency

# In another place  
nu_f = 0.8  # structural frequency

# In documentation
The frequency νₓ determines...
```

✅ **Good**: Consistent notation
```python
# Always use nu_f in code
nu_f = 0.8  # structural frequency (\nu_f)

# Always use \nu_f in documentation
# The structural frequency \(\nu_f\) determines...
```

### 7.2 Plain Text Where LaTeX is Needed

❌ **Bad**: Ambiguous plain text
```markdown
The equation is: dEPI/dt = vf * DELTA_NFR
```

✅ **Good**: Clear LaTeX
```markdown
The nodal equation is:

\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)
\]
```

### 7.3 Missing Units

❌ **Bad**: No units specified
```python
nu_f: float  # Structural frequency
```

✅ **Good**: Units explicitly stated
```python
nu_f: float  # Structural frequency in Hz_str (structural hertz)
```

### 7.4 Broken Cross-References

❌ **Bad**: Hard-coded links that break
```python
"""See coherence_operator.md for details."""
```

✅ **Good**: Relative or Sphinx references
```python
"""
See :doc:`../theory/mathematical_foundations` Section 3.1 for details.
"""
```

---

## 8. Validation Checklist

Before committing documentation changes, verify:

- [ ] All equations use consistent symbol conventions from Section 2
- [ ] Inline math uses `\( ... \)` or `:math:`...`` appropriately
- [ ] Display equations use `\[ ... \]` or `.. math::` appropriately
- [ ] All variables have units specified where applicable
- [ ] Cross-references use relative paths or Sphinx directives
- [ ] Docstrings use raw strings (`r"""..."""`) when containing LaTeX
- [ ] Examples include expected output or assertions
- [ ] "See Also" sections link to related functions
- [ ] "References" sections link to theory documents

---

## 9. Tools and Resources

### 9.1 LaTeX Testing

Test LaTeX rendering online:
- [KaTeX Playground](https://katex.org/)
- [MathJax Demo](https://www.mathjax.org/#demo)

### 9.2 Sphinx Documentation

- [Sphinx Math Extension](https://www.sphinx-doc.org/en/master/usage/extensions/math.html)
- [NumPy Docstring Standard](https://numpydoc.readthedocs.io/)
- [Sphinx Cross-Referencing](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)

### 9.3 Building Documentation Locally

```bash
# Install dependencies
pip install -e ".[docs]"

# Build HTML documentation
make docs

# Or directly with Sphinx
sphinx-build -b html docs/source docs/_build/html
```

---

## Summary

This style guide ensures that TNFR documentation maintains:

1. **Consistent notation** across all documents and code
2. **Clear mathematical presentation** using LaTeX where appropriate
3. **Bidirectional traceability** between theory and implementation
4. **Accessible examples** that demonstrate calculations step-by-step

When in doubt, prioritize clarity and consistency over brevity.
