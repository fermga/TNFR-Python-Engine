# Foundations — Quick Start (Mathematics)

This guide introduces the mathematical APIs that keep TNFR simulations canonical. It focuses on
Hilbert-space primitives, ΔNFR orchestration, and telemetry so you can launch reproducible
experiments without leaving the mathematical layer.

## 1. Setup checklist

Install the package and optional notebook extras:

```bash
pip install tnfr[notebook]
```

Verify imports succeed inside your environment:

    >>> from tnfr.mathematics import HilbertSpace, make_coherence_operator

Keep documentation examples doctest-friendly by running them through `python -m doctest` or
`pytest --doctest-glob='*.md'` before publishing updates.

## 2. Create a coherent state

Use the Hilbert space helpers to define canonical basis vectors and validate norm preservation.

    >>> from tnfr.mathematics import HilbertSpace
    >>> space = HilbertSpace(dimension=3)
    >>> vector = [1.0, 0.0, 0.0]
    >>> bool(space.is_normalized(vector))
    True

The `HilbertSpace` dataclass enforces positive dimensions and returns complex identity bases that
match TNFR's discrete spectral component. Norm checks rely on `numpy.vdot`, giving you immediate
feedback if a vector drifts away from unit coherence.

## 3. Activate protective math flags

Mathematical feature flags gate optional checks and logging. You can activate them via environment
variables or the scoped context manager.

```bash
export TNFR_ENABLE_MATH_VALIDATION=1
export TNFR_ENABLE_MATH_DYNAMICS=1
export TNFR_LOG_PERF=1
```

or programmatically:

    >>> from tnfr.config.feature_flags import context_flags, get_flags
    >>> get_flags().enable_math_validation
    False
    >>> with context_flags(enable_math_validation=True, log_performance=True) as active:
    ...     active.enable_math_validation, active.log_performance
    (True, True)
    >>> get_flags().log_performance
    False

The context manager preserves flag stacks so nested overrides unwind cleanly after each experiment.

## 4. Compose operators and track ΔNFR

Operators created in `tnfr.mathematics` keep structural semantics explicit. The factory validates
spectral inputs and enforces positive semidefinite behaviour, ensuring coherence remains monotonic.

    >>> operator = make_coherence_operator(dim=2, c_min=0.3)
    >>> round(operator.c_min, 2)
    0.3

To run operator sequences on graph nodes, couple the mathematics layer with the structural facade:

    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY  # doctest: +SKIP
    >>> from tnfr.dynamics import set_delta_nfr_hook  # doctest: +SKIP
    >>> from tnfr.structural import Coupling, create_nfr, run_sequence  # doctest: +SKIP
    >>> G, node = create_nfr("math", vf=1.0, theta=0.2)  # doctest: +SKIP
    >>> def sync(graph):  # doctest: +SKIP
    ...     graph.nodes[node][DNFR_PRIMARY] = 0.01  # doctest: +SKIP
    ...     graph.nodes[node][THETA_PRIMARY] += 0.02  # doctest: +SKIP
    ...     graph.nodes[node][VF_PRIMARY] += 0.05  # doctest: +SKIP
    >>> set_delta_nfr_hook(G, sync)  # doctest: +SKIP
    >>> run_sequence(G, node, [Coupling()])  # doctest: +SKIP
    >>> round(G.nodes[node][THETA_PRIMARY], 2)  # doctest: +SKIP

The hook keeps ΔNFR, νf, and phase aligned after each operator application, satisfying the nodal
equation `∂EPI/∂t = νf · ΔNFR(t)`.

## 5. Simulate unitary evolution

Stable unitary flows prevent coherence loss and make ΔNFR modulation auditable. The helper returns
both the pass/fail flag and the resulting norm so doctests can assert stability.

    >>> space_unitary = HilbertSpace(dimension=2)
    >>> operator_unitary = make_coherence_operator(dim=2, c_min=0.4)
    >>> state = [1.0, 0.0]
    >>> from tnfr.mathematics import stable_unitary
    >>> stable_unitary(state, operator_unitary, space_unitary)
    (True, 1.0)

Pair the call with active logging flags to capture structural frequency, ΔNFR, and phase telemetry.

## 6. Step ΔNFR-guided dynamics

Combine the ΔNFR generator with the mathematical dynamics engine to evolve states while the
protective flags are active. Keep `N ≤ 16` so doctests run quickly.

    >>> import numpy as np
    >>> from tnfr.config.feature_flags import context_flags, get_flags
    >>> from tnfr.mathematics import HilbertSpace, MathematicalDynamicsEngine, build_delta_nfr
    >>> hilbert = HilbertSpace(dimension=4)
    >>> state = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j])
    >>> np.allclose(np.linalg.norm(state), 1.0)
    True
    >>> with context_flags(enable_math_validation=True, enable_math_dynamics=True) as flags:
    ...     delta_nfr = build_delta_nfr(hilbert.dimension, topology="laplacian", nu_f=0.2)
    ...     engine = MathematicalDynamicsEngine(delta_nfr, hilbert, use_scipy=False)
    ...     evolved = engine.step(state, dt=0.05)
    ...     flags.enable_math_dynamics, flags.enable_math_validation
    ...     np.allclose(np.linalg.norm(evolved), np.linalg.norm(state))
    (True, True)
    True
    >>> get_flags().enable_math_dynamics, get_flags().enable_math_validation
    (False, False)

Leaving the context restores the defaults so later experiments start with a clean flag slate.

## 7. Cost reference

| Operation | Primary cost driver | ΔNFR impact | Notes |
| --- | --- | --- | --- |
| `make_coherence_operator` | Eigenvalue validation (`O(n)`) | Keeps thresholds ≥ `c_min` | Supply diagonal spectra for cheaper instantiation. |
| `run_sequence` | Graph traversal (`O(|E|)`) | Applies ΔNFR hook once per operator | Pre-validate sequences with `validate_sequence` to catch grammar drift. |
| `stable_unitary` | Eigen decomposition (`O(n^3)`) | Preserves Hilbert norm | Cache operators when iterating multiple steps. |
| `coherence_expectation` | Matrix-vector multiply (`O(n^2)`) | Reports scalar coherence | Toggle `normalise=False` when the state is already unit norm. |

Treat these costs as guidance for smoke tests and notebook demonstrations; production workflows
should profile real ΔNFR sequences and telemetry loads.
