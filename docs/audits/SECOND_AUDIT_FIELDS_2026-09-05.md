# Second audit: structural-potential accuracy and field consistency

Date: 2026-09-05. Scope: read-only structural fields and their numerical paths.
This extends the first [repository audit](REPOSITORY_AUDIT_2026-09-05.md),
specifically its unresolved landmark-potential finding A01. It does not certify
the repository's mathematical research claims or alter U1–U6.

## Physical contract and correction

The canonical potential is

\[
\Phi_s(i)=\sum_{j\ne i,\;0<d(i,j)<\infty}
\frac{\Delta\mathrm{NFR}_j}{d(i,j)^\alpha},\qquad \alpha=2.
\]

The default `compute_structural_potential(G)` now evaluates this field with
exact graph distances at every network size. It previously switched to an
unvalidated landmark heuristic above 500 nodes. On the first audit's
600-node path with pressure 0.1 and seed 7, that heuristic produced maximum
potential 10.810648298338 instead of 0.328320145468 and RMAE 4.507402006338.
The new default matches the analytic path sum; measured RMAE is zero at both
600 and 1,200 nodes using the float reference below. This removes approximation
error from the default U6 readout; ordinary floating-point error remains.

Large graphs use streamed BFS for unweighted paths and Dijkstra for weighted
paths. The routine retains a small dense path at at most 50 nodes. It no longer
needs a resident all-pairs distance matrix for larger graphs. Weighted
MultiGraph edges are recognized and use their minimum path length, as required
by shortest-path semantics. These distance semantics differ intentionally from
the EPI diffusion channel's sum of parallel conductances.

The documented outgoing direction is preserved: on a directed graph, the
source `i` reads sources reachable along `i -> j`. Positive edge lengths define
the metric. Historical behavior excluding zero-distance source-target pairs is
retained; explicitly requested landmarks fall back to exact evaluation when an
edge has nonpositive length. This is not a general negative-weight solver.

No EPI, frequency, phase, graph topology or operator sequence is modified.
Landmark selection no longer consumes the global random stream: it depends on
topology with a deterministic tie order, preserving observational reproducibility.

## Explicit approximation and validation

The existing `landmark_ratio` argument still explicitly selects approximation.
The former expression

`min_l abs(d(l,i) - d(l,j))`

was a lower distance estimate and could invent short interactions. It is replaced
by the length of an actual path through a selected landmark:

`min_l [d(i,l) + d(l,j)]`.

Directed graphs compute the first leg on the reversed graph. Unreachable legs
remain infinity; neither `1e9` sentinels nor artificial cross-component terms are
used. The vectorized path now reduces one two-dimensional block per landmark,
instead of allocating an `L × batch × N` tensor. Its working block scales as
`O(batch × N)` in addition to the `O(L × N)` stored landmark maps.

For positive distances this construction bounds distance from above. It does
**not** certify relative potential error for signed pressure: cancellation can
amplify relative error. Explicit approximation remains unsuitable for U6
decisions without verification. Its measured errors below are substantial;
the exact default is the reliable entry point.

`validate=True` with an explicit ratio now compares **all** returned node values
against the exact field. `sample_size` remains accepted for call compatibility
but cannot restrict this verification to a sample. RMAE is

`sum_i abs(approx_i - exact_i) / sum_i abs(exact_i)`.

When the denominator is zero, RMAE is zero only if the absolute error is also
zero, and infinity otherwise. Nonfinite pressure/reference values and invalid
error tolerances cannot receive a validation result. Ratio refinement remains
available through `max_refinements`; if tolerance is still unmet, the returned
field is exact. This necessarily incurs the exact reference's computational cost.

For compatibility, explicit validated calls retain the diagnostic dictionary
keys `__phi_s_landmark_ratio__` and `__phi_s_rmae__`; the new
`__phi_s_fallback_exact__` is 1.0 after exact fallback and 0.0 otherwise. RMAE
describes the returned field. These diagnostic keys are not graph nodes. Default
exact calls return only the node-to-potential mapping.

## Additional confirmed field inconsistencies

| Defect | Independent expected result | Correction |
| --- | --- | --- |
| Weighted MultiGraph potential ignored edge weights in the medium-size BFS branch. | A 60-node path with parallel lengths 2.0 and 0.5 must use `d(i,j)=0.5*abs(i-j)`. The previous result was one quarter of the expected potential. | Weighted edge detection inspects individual edge attributes; all sizes use the exact distance selector. |
| Nodal classification claimed to read the same unit-source kernel but discarded edge weights. | Path edge lengths 0.5 and 1.5 give centralities `[4.25, 4.444444444444445, 0.6944444444444444]`; previous values were `[1.25, 2, 1.25]`. | `classify_nodal_topology` delegates the unit-source calculation to the exact potential kernel. |
| `path_integrated_gradient` included the target, contrary to its documented sum through `k-1`. | On phases `[0,.2,.6]`, the path `0 -> 1 -> 2` sums `.2+.3=.5`; previous result was `.9`. A zero-edge path must sum zero. | Sum `path[:-1]`; document NetworkX traversal-order tie handling instead of claiming lexicographic tie selection. |
| Telemetry retained a separate dense potential implementation. | Direct field and telemetry must evaluate the same exact kernel at every size. | Telemetry delegates to the canonical cached function and removes its redundant dense distance matrix. |

## Reproducible measurements

Environment: Python 3.12.10, NumPy 2.3.3, NetworkX 3.5,
Windows 11 build 26200. The imported module was explicitly verified as
`C:\TNFR-Python-Engine\src\tnfr\physics\canonical.py`.

Each timing is the median of three cold-cache calls. Graph construction and
analytic-reference construction are excluded. These are local measurements,
not latency guarantees; other audit processes were active on the same machine.
These timings precede the precision cross-review addendum below; they do not
claim a final-build latency guarantee.

| Path nodes | Method | Median seconds | RMAE of returned node values | Exact fallback |
| ---: | --- | ---: | ---: | --- |
| 600 | Exact default | 0.197667 | 0 | Not applicable |
| 600 | Explicit 50 landmarks | 0.042611 | 0.898212172682 | No validation requested |
| 600 | Explicit 50 landmarks, validation defaults | 0.669911 | 0 | Yes |
| 1,200 | Exact default | 0.773038 | 0 | Not applicable |
| 1,200 | Explicit 50 landmarks | 0.146804 | 0.943444760836 | No validation requested |
| 1,200 | Explicit 50 landmarks, validation defaults | 2.233440 | 0 | Yes |

Run this from the repository root with `.venv312/Scripts/python.exe`. The
explicit source-path insertion is necessary: ordinary Python invocations may
otherwise import a separately installed wheel; pytest sets its own source path.

```python
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
import networkx as nx
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR
import tnfr.physics.canonical as canonical
from tnfr.utils.cache import reset_global_cache

print(canonical.__file__)
for size in (600, 1200):
    graph = nx.path_graph(size)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.1)
    exact = {
        i: math.fsum(0.1 / (i - j)**2 for j in graph if i != j)
        for i in graph
    }
    for options in (
        {},
        {"landmark_ratio": 50 / size},
        {"landmark_ratio": 50 / size, "validate": True},
    ):
        elapsed = []
        for _ in range(3):
            reset_global_cache()
            canonical._PHI_S_DISTANCE_CACHE.clear()
            start = time.perf_counter()
            result = canonical.compute_structural_potential(graph, **options)
            elapsed.append(time.perf_counter() - start)
        rmae = math.fsum(abs(result[i] - exact[i]) for i in graph)
        rmae /= math.fsum(abs(value) for value in exact.values())
        print(size, options, statistics.median(elapsed), rmae)
```

## Regression evidence and remaining limits

The baseline of existing field/cache/coherence-length tests passed: **23 passed**.
The first seven new analytic tests all failed before correction. The two
additional field-consistency tests and the nonfinite-validation test also
reproduced failures before their fixes. The final focused run passes **38 tests**:

```sh
.venv312/Scripts/python.exe -m pytest tests/physics/test_structural_potential_accuracy.py tests/physics/test_field_readout_consistency.py tests/physics/test_field_cache_invalidation.py tests/test_vectorized_coherence_length_regression.py -q
```

The new regression file contains 15 cases covering the 600-node analytic path,
weighted directed signed pressure, disconnected components, parallel edges,
global validation, invalid tolerances, nonfinite pressure, zero-distance
compatibility, RNG preservation, scalar/vectorized parity, weighted geometry and
path-integral endpoints. This is bounded evidence, not an exhaustive field proof.

A broader field/dynamics/SDK run passed **1,225 tests**, with 22 warnings about
the unavailable optional JAX dependency and the existing Coherence-Coherence
anti-pattern diagnostic:

```sh
.venv312/Scripts/python.exe -m pytest tests/physics tests/core_physics tests/sdk/test_simple_advanced.py --ignore=tests/physics/test_diffusion_generator_consistency.py -q
```

The explicitly excluded generator-consistency file belonged to the parallel
heterogeneous-diffusion investigation and was still being implemented during
this run. It is not claimed as verified here; repository-wide integration is
reported separately by the coordinating audit.

Remaining limits: exact all-source evaluation is still quadratic on sparse
connected paths and more expensive on dense graphs; very large graphs require a
separately validated acceleration strategy. Explicit unvalidated approximation
can substantially underestimate positive potential and has no signed-field
relative-error certificate. Landmark-distance caching remains process-global;
this work does not redesign its capacity management. The legacy diagnostic keys
on explicit validated results remain a compatibility constraint.

## Cross-review addendum: finite certificates and precision

Independent review subsequently reproduced and fixed a nonfinite exact-output
guard bypass on the zero-length-edge fallback, precision-mode cache reuse, and
loss of signed cancellation in research/dense sums. Dense, streamed and direct
fallback exact paths now share compensated summation; canonical fields and
aggregate telemetry declare precision mode as a cache dependency.

The [cross-review evidence](SECOND_AUDIT_CERTIFICATES_2026-09-05.md#independent-field-cross-review-corrections)
records the independent Dijkstra fixtures and analytic cancellation checks.
Its final focused run passed **127 tests**, with one platform skip for a truly
extended longdouble mantissa unavailable on this Windows runtime. These tests
overlap the earlier scoped results above. The [integrated audit](SECOND_REPOSITORY_AUDIT_2026-09-05.md)
records the final repository-wide result.
