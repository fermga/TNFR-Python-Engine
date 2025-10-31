# Foundations — Mathematics scaffold

The mathematics layer exposes the canonical spaces, ΔNFR generators, and
runtime diagnostics that keep the nodal equation faithful to
``∂EPI/∂t = νf · ΔNFR(t)``.  This quick-start walks through the minimal
scaffolding required to stand up a reproducible spectral experiment, turn on
validation guards, and observe unitary stability before coupling into higher
level operators.

> **Migration note**: ``tnfr.mathematics.validators`` has been removed. Import
> :mod:`tnfr.validation` directly to access
> :class:`tnfr.validation.spectral.NFRValidator` when enabling the guards
> described in this primer.

## 1. Canonical quick-start

1. **Select a space** – use :class:`tnfr.mathematics.HilbertSpace` for discrete
   spectral experiments or :class:`tnfr.mathematics.BanachSpaceEPI` when mixing
   the continuous EPI tail.  The Banach constructor now ships with
   :class:`tnfr.mathematics.BEPIElement`, a dataclass that keeps the trio
   ``(f_continuous, a_discrete, x_grid)`` coherent and exposes the EPI algebra
   ``⊕``/:meth:`~tnfr.mathematics.BEPIElement.direct_sum`, ``⊗``/
   :meth:`~tnfr.mathematics.BEPIElement.tensor`, ``*``/
   :meth:`~tnfr.mathematics.BEPIElement.adjoint` and ``∘``/
   :meth:`~tnfr.mathematics.BEPIElement.compose`.  Factor helpers on
   :class:`~tnfr.mathematics.BanachSpaceEPI` generate zero elements, canonical
   basis vectors and Hilbert tensors so that ΔNFR operators always receive
   validated inputs.
2. **Construct ΔNFR** – call :func:`tnfr.mathematics.build_delta_nfr` with a
   topology (``"laplacian"`` or ``"adjacency"``) and νf scaling.  The helper
   guarantees a Hermitian generator so downstream coherence checks remain
   meaningful.
3. **Wrap operators** – initialise
   :class:`tnfr.mathematics.CoherenceOperator`/:class:`~tnfr.mathematics.FrequencyOperator`
   to project coherence and νf expectations.
4. **Collect metrics** – invoke
   :func:`tnfr.mathematics.normalized`, :func:`~tnfr.mathematics.coherence`,
   :func:`~tnfr.mathematics.frequency_positive`, and
   :func:`~tnfr.mathematics.stable_unitary` to ensure ΔNFR preserves Hilbert
   norms while sustaining positive structural frequency.

The notebooks
[`theory/00_overview.ipynb`](theory/00_overview.ipynb) and
[`theory/02_phase_synchrony_lattices.ipynb`](theory/02_phase_synchrony_lattices.ipynb)
replay these steps with expanded derivations and visual telemetry overlays.

## 2. Graph-level math engine configuration

Runtime graphs can opt into spectral co-evolution by attaching a
``G.graph["MATH_ENGINE"]`` dictionary.  When ``enabled`` the runtime will
advance a :class:`tnfr.mathematics.dynamics.MathematicalDynamicsEngine` in lock
step with the classical ΔNFR integrator and emit per-step telemetry into the
graph history under ``"math_engine_*"`` keys.  The configuration expects the
following entries:

* ``enabled`` – boolean switch that activates the branch.
* ``hilbert_space`` – :class:`tnfr.mathematics.HilbertSpace` instance matching
  the generator dimension.
* ``coherence_operator`` – :class:`tnfr.mathematics.CoherenceOperator` used to
  evaluate ``C_min``.
* ``coherence_threshold`` – scalar floor applied to the coherence expectation.
* ``frequency_operator`` (optional) –
  :class:`tnfr.mathematics.FrequencyOperator` validating νf positivity.
* ``dynamics_engine`` – pre-built
  :class:`~tnfr.mathematics.dynamics.MathematicalDynamicsEngine`; alternatively
  supply ``generator_matrix`` so the runtime can construct one lazily.
* ``state_projector`` (optional) – projector implementing
  :class:`tnfr.mathematics.projection.StateProjector`.  Defaults to
  :class:`~tnfr.mathematics.projection.BasicStateProjector`.

```pycon
>>> import numpy as np
>>> from tnfr.mathematics import (
...     BasicStateProjector,
...     CoherenceOperator,
...     HilbertSpace,
...     MathematicalDynamicsEngine,
...     make_frequency_operator,
... )
>>> hilbert = HilbertSpace(dimension=3)
>>> generator = np.diag([0.1, -0.05, 0.02])
>>> coherence_op = CoherenceOperator(np.eye(3))
>>> frequency_op = make_frequency_operator(np.eye(3))
>>> G.graph["MATH_ENGINE"] = {
...     "enabled": True,
...     "hilbert_space": hilbert,
...     "coherence_operator": coherence_op,
...     "coherence_threshold": coherence_op.c_min,
...     "frequency_operator": frequency_op,
...     "dynamics_engine": MathematicalDynamicsEngine(generator, hilbert),
...     "state_projector": BasicStateProjector(),
... }
```

Each call to :func:`tnfr.dynamics.runtime.step` (or :func:`~tnfr.dynamics.runtime.run`)
will advance the stored Hilbert vector, verify normalization, coherence
threshold compliance, and νf positivity via :mod:`tnfr.mathematics.runtime`, and
publish the summary into ``G.graph['telemetry']['math_engine']`` as well as the
rolling history.

## 3. Environment feature flags

Mathematics diagnostics respect three environment variables.  They are read via
:func:`tnfr.config.get_flags` and can be temporarily overridden with
:func:`tnfr.config.context_flags`.

* ``TNFR_ENABLE_MATH_VALIDATION`` – enables strict ΔNFR/Hilbert assertions
  inside runtime validators.
* ``TNFR_ENABLE_MATH_DYNAMICS`` – unlocks experimental spectral integrators
  in :mod:`tnfr.mathematics.dynamics`.
* ``TNFR_LOG_PERF`` – activates debug logging for normalization, coherence, and
  unitary metrics.

The snippet below demonstrates the override stack; the state before and after
``context_flags`` confirms that overrides remain scoped to the ``with`` block.

```pycon
>>> from tnfr.config.feature_flags import get_flags, context_flags
>>> get_flags().enable_math_validation
False
>>> with context_flags(enable_math_validation=True, log_performance=True) as scoped:
...     (scoped.enable_math_validation, scoped.log_performance)
(True, True)
>>> get_flags().log_performance
False

```

When running shell commands, export the variables directly, e.g.
``TNFR_ENABLE_MATH_VALIDATION=1 TNFR_LOG_PERF=1 python -m doctest docs/foundations.md``.

## 4. Executable ΔNFR and unitary validation

The following session builds a Laplacian ΔNFR generator, evaluates unitary
stability, and asserts νf positivity.  All routines are deterministic when a
NumPy generator seed is supplied to :func:`build_delta_nfr`, making the snippet
safe for doctest execution.

```pycon
>>> import numpy as np
>>> from tnfr.mathematics import (
...     HilbertSpace,
...     build_delta_nfr,
...     CoherenceOperator,
...     FrequencyOperator,
...     stable_unitary,
...     coherence,
...     frequency_positive,
... )
>>> space = HilbertSpace(dimension=3)
>>> delta = build_delta_nfr(3, topology="laplacian", nu_f=0.8, scale=0.25)
>>> delta.shape
(3, 3)
>>> operator = CoherenceOperator(delta)
>>> state = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
>>> unitary_passed, unitary_norm = stable_unitary(state, operator, space)
>>> unitary_passed
True
>>> round(unitary_norm, 12)
1.0
>>> frequency_positive(state, FrequencyOperator(np.eye(3)))['passed']
True
>>> coherence(state, operator, threshold=operator.c_min)
(True, 0.4)

```

To integrate ΔNFR outputs into networkx graphs, see the migration recipe in
[`getting-started/quickstart.md`](getting-started/quickstart.md) and the
operator catalogue under [`api/operators.md`](api/operators.md).

## 5. Telemetry cost and logging budget

| Metric guard | Flag dependency | Dominant cost | Logging channel |
| --- | --- | --- | --- |
| ``normalized`` | ``TNFR_LOG_PERF`` | ``O(n)`` vector norm | ``tnfr.mathematics.runtime`` debug record |
| ``coherence`` / ``coherence_expectation`` | ``TNFR_LOG_PERF`` | ``O(n²)`` due to matrix-vector multiply | Same channel with payload ``{"threshold": …}`` |
| ``frequency_positive`` | ``TNFR_LOG_PERF`` | ``O(n²)`` spectrum check plus projection | Debug message includes ``"projection_passed"`` and spectrum extrema |
| ``stable_unitary`` | ``TNFR_LOG_PERF`` | ``O(n³)`` eigendecomposition per step | Debug payload logs ``"norm_after"`` for ΔNFR unitary audits |

The runtime helpers defer to Python's :mod:`logging` package.  Configure it once
at process start (``logging.basicConfig(level=logging.DEBUG)``) and then enable
``TNFR_LOG_PERF`` to stream the tabled payloads without instrumenting call sites.
The Phase 3 guideline is to sample the ``stable_unitary`` log at each
integration step while only periodically recording the cheaper ``normalized``
metric to control storage costs.

## 6. Next steps

* Load the lattice notebooks listed above to inspect full ΔNFR evolution
  traces.
* Refer to [`api/telemetry.md`](api/telemetry.md) for downstream aggregation and
  to [`theory/00_overview.ipynb`](theory/00_overview.ipynb) for the
  derivation that ties the Hilbert norms back to ΔNFR coherence envelopes.
