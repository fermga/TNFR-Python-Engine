# Foundations — Quick Start (Mathematics)

This guide introduces the mathematical APIs that keep TNFR simulations canonical. It focuses on
Hilbert-space primitives, ΔNFR orchestration, and telemetry so you can launch reproducible
experiments without leaving the mathematical layer.

## 1. Setup checklist

1. Install the package and optional notebook extras.
   ```bash
   pip install tnfr[notebook]
   ```
2. Verify imports succeed inside your environment.
   ```pycon
   >>> import tnfr
   >>> from tnfr.mathematics import HilbertSpace, make_coherence_operator
   ```
3. Keep documentation examples doctest-friendly by running them through `python -m doctest` or
   `pytest --doctest-glob='*.md'` before publishing updates.

## 2. Create a coherent state

Use the Hilbert space helpers to define canonical basis vectors and validate norm preservation.

```pycon
>>> from tnfr.mathematics import HilbertSpace
>>> space = HilbertSpace(dimension=3)
>>> vector = [1.0, 0.0, 0.0]
>>> space.is_normalized(vector)
True
```

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

```pycon
>>> from tnfr.config.feature_flags import context_flags, get_flags
>>> get_flags().enable_math_validation
False
>>> with context_flags(enable_math_validation=True, log_performance=True) as active:
...     active.enable_math_validation, active.log_performance
(True, True)
>>> get_flags().log_performance
False
```

The context manager preserves flag stacks so nested overrides unwind cleanly after each experiment.

## 4. Compose operators and track ΔNFR

Operators created in `tnfr.mathematics` keep structural semantics explicit. The factory validates
spectral inputs and enforces positive semidefinite behaviour, ensuring coherence remains monotonic.

```pycon
>>> from tnfr.mathematics import make_coherence_operator
>>> operator = make_coherence_operator(dim=2, c_min=0.3)
>>> round(operator.c_min, 2)
0.3
```

To run operator sequences on graph nodes, couple the mathematics layer with the structural facade:

```pycon
>>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY
>>> from tnfr.dynamics import set_delta_nfr_hook
>>> from tnfr.structural import Coupling, create_nfr, run_sequence
>>> G, node = create_nfr("math", vf=1.0, theta=0.2)
>>> def sync(graph):
...     graph.nodes[node][DNFR_PRIMARY] = 0.01
...     graph.nodes[node][THETA_PRIMARY] += 0.02
...     graph.nodes[node][VF_PRIMARY] += 0.05
>>> set_delta_nfr_hook(G, sync)
>>> run_sequence(G, node, [Coupling()])
>>> round(G.nodes[node][THETA_PRIMARY], 2)
0.22
```

The hook keeps ΔNFR, νf, and phase aligned after each operator application, satisfying the nodal
equation `∂EPI/∂t = νf · ΔNFR(t)`.

## 5. Simulate unitary evolution

Stable unitary flows prevent coherence loss and make ΔNFR modulation auditable. The helper returns
both the pass/fail flag and the resulting norm so doctests can assert stability.

```pycon
>>> from tnfr.mathematics import HilbertSpace, make_coherence_operator, stable_unitary
>>> space = HilbertSpace(dimension=2)
>>> operator = make_coherence_operator(dim=2, c_min=0.4)
>>> state = [1.0, 0.0]
>>> stable_unitary(state, operator, space)
(True, 1.0)
```

Pair the call with active logging flags to capture structural frequency, ΔNFR, and phase telemetry.

## 6. Cost reference

| Operation | Primary cost driver | ΔNFR impact | Notes |
| --- | --- | --- | --- |
| `make_coherence_operator` | Eigenvalue validation (`O(n)`) | Keeps thresholds ≥ `c_min` | Supply diagonal spectra for cheaper instantiation. |
| `run_sequence` | Graph traversal (`O(|E|)`) | Applies ΔNFR hook once per operator | Pre-validate sequences with `validate_sequence` to catch grammar drift. |
| `stable_unitary` | Eigen decomposition (`O(n^3)`) | Preserves Hilbert norm | Cache operators when iterating multiple steps. |
| `coherence_expectation` | Matrix-vector multiply (`O(n^2)`) | Reports scalar coherence | Toggle `normalise=False` when the state is already unit norm. |

Treat these costs as guidance for smoke tests and notebook demonstrations; production workflows
should benchmark with representative data.

## 7. Logging best practices

- Use `context_flags(log_performance=True)` around critical sections so `tnfr.mathematics.runtime`
  emits structured debug entries for each metric (`coherence`, `frequency_positive`, `stable_unitary`).
- Capture telemetry via `tnfr.telemetry.ensure_cache_metrics_publisher` when you need persistent
  C(t), νf, Si, and ΔNFR traces.
- Store logs alongside experiment metadata (seed, operator list, thresholds) so coherence audits can
  reconstruct the entire structural trajectory.

With these steps the mathematical quick start remains reproducible, doctest-friendly, and aligned
with TNFR's canonical invariants.
