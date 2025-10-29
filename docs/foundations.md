# Foundations — Mathematics scaffold

The mathematics layer exposes the canonical spaces, ΔNFR generators, and
runtime diagnostics that keep the nodal equation faithful to
``∂EPI/∂t = νf · ΔNFR(t)``.  This quick-start walks through the minimal
scaffolding required to stand up a reproducible spectral experiment, turn on
validation guards, and observe unitary stability before coupling into higher
level operators.

## 1. Canonical quick-start

1. **Select a space** – use :class:`tnfr.mathematics.HilbertSpace` for discrete
   spectral experiments or :class:`tnfr.mathematics.BanachSpaceEPI` when mixing
   the continuous EPI tail.
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

## 2. Environment feature flags

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

## 3. Executable ΔNFR and unitary validation

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

## 4. Telemetry cost and logging budget

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

## 5. Next steps

* Load the lattice notebooks listed above to inspect full ΔNFR evolution
  traces.
* Refer to [`api/telemetry.md`](api/telemetry.md) for downstream aggregation and
  to [`theory/00_overview.ipynb`](theory/00_overview.ipynb) for the
  derivation that ties the Hilbert norms back to ΔNFR coherence envelopes.
