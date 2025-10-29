# Fase 2 Mathematics Integration

The Fase 2 programme introduces a spectral mathematics layer that augments the
classic runtime with Hermitian projectors, ΔNFR generators and reproducible
state validation. This note captures the acceptance criteria exercised by the
integration tests and documents the canonical usage paths.

## Flag activation

Mathematics features are guarded by the `enable_math_validation` flag. The flag
follows the precedence order validated by the test-suite: explicit arguments on
`NodeNX`, contextual overrides via `context_flags`, and finally the global
configuration defaults. The snippet below demonstrates the activation cycle that
keeps the override scoped to the context manager:

```python
>>> import logging
>>> logging.getLogger("tnfr.utils.init").setLevel(logging.ERROR)
>>> from tnfr.config.feature_flags import context_flags, get_flags
>>> base_flag = get_flags().enable_math_validation
>>> with context_flags(enable_math_validation=True):
...     assert get_flags().enable_math_validation is True
>>> assert get_flags().enable_math_validation is base_flag

```

## Projector usage

`BasicStateProjector` maps the node scalars—EPI magnitude, structural frequency
(νf) and phase—onto the canonical Hilbert basis. It emits normalised complex
vectors and accepts an optional RNG for deterministic stochastic excitation.
Combined with the `HilbertSpace` helpers the projector supplies reproducible
pre/post states for validation.

## ΔNFR builder

`build_delta_nfr` mirrors the classical ΔNFR constructors through Hermitian
matrix generators. It accepts a Hilbert space dimension together with the
desired topology (Laplacian or adjacency), structural frequency scaling ``νf``
and an additional amplitude ``scale``. The integration suite ensures both
recipes remain Hermitian and reproducible when seeded with a NumPy RNG. The
resulting matrices are suitable inputs for spectral dynamics engines or bespoke
analyses.

## Dynamics engine

`MathematicalDynamicsEngine` evolves states through the unitary exponential of
Hermitian ΔNFR operators. The engine caches the eigendecomposition so repeated
steps reuse the spectral factorisation. Optional SciPy support switches to the
reference `expm` implementation when available; the tests compare both paths to
confirm parity.

## End-to-end example

The following walkthrough combines the acceptance requirements into a single
pipeline: classical orchestration, projection into the Hilbert space, ΔNFR
extraction and (optionally) unitary evolution. The code executes as a doctest to
provide a lightweight smoke validation for the documentation itself.

```python
>>> from tnfr.structural import create_nfr, run_sequence
>>> from tnfr.node import add_edge
>>> from tnfr.operators.definitions import Emission, Reception, Coherence, Resonance, Transition
>>> G, node = create_nfr("fase2-demo", epi=0.8, vf=1.2, theta=0.1)
>>> _ = create_nfr("fase2-partner", epi=0.5, vf=0.9, theta=0.0, graph=G)
>>> add_edge(G, node, "fase2-partner", 1.0)
>>> run_sequence(G, node, [Emission(), Reception(), Coherence(), Resonance(), Transition()])
>>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY
>>> round(G.nodes[node][EPI_PRIMARY], 6)
0.723125
>>> from tnfr.mathematics import BasicStateProjector, HilbertSpace, build_coherence_operator, build_delta_nfr
>>> from tnfr.mathematics.runtime import normalized, coherence_expectation
>>> import numpy as np
>>> hilbert = HilbertSpace(2)
>>> projector = BasicStateProjector()
>>> state = projector(
...     epi=G.nodes[node][EPI_PRIMARY],
...     nu_f=G.nodes[node][VF_PRIMARY],
...     theta=G.nodes[node][THETA_PRIMARY],
...     dim=hilbert.dimension,
... )
>>> normalized(state, hilbert)[0]
True
>>> coherence = build_coherence_operator(np.eye(hilbert.dimension) * 0.75)
>>> round(coherence_expectation(state, coherence), 6)
0.75
>>> delta = build_delta_nfr(hilbert.dimension, topology="adjacency")
>>> delta.shape
(2, 2)
>>> from tnfr.mathematics import MathematicalDynamicsEngine
>>> MathematicalDynamicsEngine(delta, hilbert_space=hilbert, use_scipy=False)  # doctest: +SKIP
MathematicalDynamicsEngine(...)
```

The skipped instantiation highlights where the unitary dynamics would be
constructed. Calling `engine.step(state, dt)` yields the same deterministic
trajectory checked by the reproducibility tests when SciPy is available or the
NumPy eigendecomposition path is selected.
