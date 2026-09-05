# TNFR repository audit and consolidation

Date: 2026-09-05. Baseline: `47603cff1ee922df21cbf783aec74a3a8b8acbc5`.

This report records the first-pass state. The [second audit](SECOND_REPOSITORY_AUDIT_2026-09-05.md)
supersedes its outstanding implementation dispositions, including potential
approximation, lifetime grammar context, heterogeneous diffusion and geometric
certificates. The mathematical model/proof gaps remain explicitly scoped there.

The audit found reproducible defects despite a passing baseline suite. Changes
correct structural pressure, grammar selection, field telemetry, caching,
spectral readouts, and reporting, and remove competing implementations in these
paths. Important mathematical and approximation defects remain explicitly
unresolved below; this is not a certification of the whole repository.

## Scope and method

The inventory covered 447 Python modules under `src/tnfr`; all parsed without
syntax errors. Review then concentrated on the nodal equation, U1–U6, canonical
operator contracts, graph-neighborhood semantics, cache dependencies, spectral
geometry, public backend entry points, and packaging/testing configuration.
Three parallel reviews covered grammar, fields/caching, and mathematical claims.
Independent review of the resulting changes complemented integration tests.

An AST comparison found 21 groups of identical function bodies after removing
docstrings, selecting functions spanning at least 16 lines. This is a triage
count, including trivial abstract methods and compatibility wrappers, not a
count of 21 defects. Duplicated exporters and backend implementations were
consolidated where their drift or maintenance cost was demonstrated. Specialized
Riemann routines and legacy helpers were not mechanically merged.

The initial working tree contained untracked `manual/`; it was excluded and
left untouched. No commits, releases, remote writes, or publication were made.
Reproduction used the working source via pytest's `pythonpath=["src"]` or
explicit `PYTHONPATH=src`, rather than the separately installed package.

## Implemented corrections

| ID | Defect and evidence before correction | Correction and regression evidence |
| --- | --- | --- |
| F01 | Checking `sum(weights)==edge_count` misidentified weights 0.5/1.5 as unit weights. Parallel edges also lost their combined weight. For successors with EPI 1 and 2, pressure was 1.5 instead of 1.75. | Inspect live individual weights and sum parallel-edge contributions, realizing `-L_rw EPI`. Allocate the optional weight vector only when a non-unit weight exists. Graph, DiGraph, MultiGraph, MultiDiGraph and weight-mutation regressions. |
| F02 | Scalar fallback disagreed with the canonical vectorized channels. An EPI-only directed fork returned `[1.5,-1,-2]` instead of `[1.75,0,0]`. Sinks acquired pressure without outgoing neighbors. | Weighted EPI numerator/denominator separated from arithmetic phase/frequency/topology means; unique successor counts and inactive sink channels preserved. Serial, two-worker and simulated unavailable-NumPy cases tested. |
| F03 | `OptimizedNumPyBackend` crashed at 100 nodes because it passed a removed `np` argument. Its duplicate implementation also omitted edge weights and would have multiplied pressure by frequency before nodal integration. | Route the compatibility backend through `NumPyBackend` and the shared fused pipeline at every graph size. Tests use zero and non-unit frequency, weighted graphs, isolates and unchanged EPI. |
| F04 | Dynamic selection allowed phase-incompatible Coupling/Resonance as warnings and accepted unknown operator codes. Multi-turn phase wrapping was incorrect. | Use the execution path's `validate_phase_gate_u3` and reject unknown codes. Candidate/execution agreement and wrapped angles tested. |
| F05 | Rejected steps entered incremental shadow history; an explicit `None` history was not restored; one node's fallback replaced the requested glyph for subsequent nodes. | Only accepted steps extend history, original presence/value is restored, and each node selects its own fallback. Recorded histories verify independent decisions. Filtering also stops computing discarded alternative suggestions. |
| F06 | Memoized validation crashed with default arguments and could bypass ordered Mutation context because glyph/English classifications diverged. Canonical validation also incorrectly expired prior Coherence together with destabilizer recency. | Cache static preflight only; delegate final decisions to `GrammarValidator`. Prior IL remains valid across earlier sequence steps, while the destabilizer retains its recency window. Default calls, names/glyphs, U4b ordering, mutable U5 metadata and old-prior-IL regression tested. |
| F07 | Four vectorized field paths divided successor sums by total degree. Constant pressure 0.5 on `0→1→2` produced flux `[0,-0.25,-0.5]` instead of zero. Loop and parallel-neighbor counting also drifted. | One `neighborhood_arrays` helper constructs both sums' indices and denominators from `G.neighbors`. Directed, looped, parallel-edge and scalar/vectorized readouts tested. |
| F08 | Cache hashes selected one attribute alias for a whole graph, conflated values such as `(1,23)` and `(12,3)`, omitted weights, and reused same-degree landmark data under different node labels. Bound graph arguments also lost identity. | Framed per-node records with per-node alias resolution, labelled weighted topology, graph identity, and qualified function identity. Mixed aliases, relabelling, repeated calls and live weight changes tested. |
| F09 | Exact potential replaced disconnected distance with `1e9`, creating false cross-component influence. Equilibrium landmarks computed `0/0`, returning NaN. | Only finite positive distances contribute; self terms are excluded before division. Disconnected exact-potential and zero-pressure landmark regressions. Approximation accuracy remains open as A01 below. |
| F10 | A private normalized Laplacian disagreed with its public counterpart at isolates: one edge plus one isolate had spectrum `[0,1,2]` instead of `[0,0,2]`. | Private compatibility wrapper delegates to the canonical routine. Weighted isolate spectra, eigenvector equations and edgeless collective-pulse energy tested. |
| F11 | SDK spatial coherence length used `1/sqrt(nu_f*lambda_2)`. Multiplying frequency by four halved a spatial readout with unchanged geometry. | Use unscaled graph eigenvalues for length, retaining temporal rate fields. Analytic cycle tests cover frequencies 0, 0.25, 1 and 4. |
| F12 | Duplicate JSON/HTML exporters discarded zero/false structural evidence; node 0 was omitted and literal `<nu_f>` was interpreted as markup. | One exporter implementation, explicit missing-value checks, and HTML escaping. Validator and exception reports preserve zero/false/node-0 evidence. Historical check IDs remain stable and are distinguished from the six canonical invariants. |
| F13 | Root packaging had conflicting Poetry/setuptools dependency declarations and an unrelated primality entry point. Keywords/classifiers were stored in an unpublished custom table. Mypy/Black still targeted Python 3.9 despite a 3.10 minimum. | Retain the active setuptools/PEP 621 configuration and existing published dependencies/scripts; move keywords/classifiers into project metadata and align tool targets. Source/wheel builds verify distribution metadata. |
| F14 | Testing guides described nonexistent folders, markers, examples and defaults. A theory sentence incorrectly called canonical coherence scale-invariant. | `TESTING.md` is the testing authority; `tests/README.md` links to it. Correct the coherence sentence using `1/2→1/3` under pressure doubling; broader completeness claims remain open. |

Primary implementation paths:

- Pressure: [dnfr.py](../../src/tnfr/dynamics/dnfr.py),
  [optimized_numpy.py](../../src/tnfr/backends/optimized_numpy.py).
- Grammar: [grammar_dynamics.py](../../src/tnfr/operators/grammar_dynamics.py),
  [grammar_application.py](../../src/tnfr/operators/grammar_application.py),
  [grammar_memoization.py](../../src/tnfr/operators/grammar_memoization.py),
  [grammar_core.py](../../src/tnfr/operators/grammar_core.py).
- Fields: [_helpers.py](../../src/tnfr/physics/_helpers.py),
  [canonical.py](../../src/tnfr/physics/canonical.py),
  [extended.py](../../src/tnfr/physics/extended.py),
  [telemetry.py](../../src/tnfr/physics/telemetry.py),
  [vectorized_ops.py](../../src/tnfr/physics/vectorized_ops.py),
  [cache.py](../../src/tnfr/utils/cache.py).
- Spectra and reports: [structural_diffusion.py](../../src/tnfr/physics/structural_diffusion.py),
  [simple.py](../../src/tnfr/sdk/simple.py),
  [validator.py](../../src/tnfr/validation/validator.py).

## Remaining defects and mathematical limits

**A01 — High priority: uncontrolled default landmark approximation.**
For `nx.path_graph(600)`, every pressure equal to 0.1, `random.seed(7)` and
cleared caches, the default landmark potential had maximum **10.8106483**;
the exact result had maximum **0.328320145**. Relative mean absolute error
was **450.74%**. This can trigger false U6 alarms. Both implementations use
`min_l |d(l,i)-d(l,j)|`, an uncontrolled underestimate of distance. The
landmark branch also retains finite disconnected-distance sentinels.
Enabling validation reports approximation error but can still return a result
above the requested tolerance after refinement.

Correcting this requires a validated estimator with error-controlled fallback,
or an exact default with an explicit performance tradeoff. The present patch
fixes invalid arithmetic and cache reuse but does not claim approximation
accuracy. Until corrected, compare consequential large-graph U6 conclusions
against an independently evaluated exact shortest-path sum. The public function
currently has no option to force exact computation on large graphs.

Runnable reproducer after selecting the working source (`PYTHONPATH=src`):

```python
import math
import random
import networkx as nx
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.physics.canonical import _PHI_S_DISTANCE_CACHE, compute_structural_potential
from tnfr.utils.cache import reset_global_cache

random.seed(7)
reset_global_cache()
_PHI_S_DISTANCE_CACHE.clear()
graph = nx.path_graph(600)
for node in graph:
    set_attr(graph.nodes[node], ALIAS_DNFR, 0.1)
approx = compute_structural_potential(graph)
# Exact path distance is |i-j|.
exact = {i: math.fsum(0.1 / (i-j)**2 for j in graph if i != j) for i in graph}
rmae = math.fsum(abs(approx[i]-exact[i]) for i in graph) / math.fsum(exact.values())
print(rmae, max(approx.values()), max(exact.values()))
# 4.507402006338, 10.810648298338, 0.328320145468
```

**A02 — Validation scope: corrected by the second audit.** The first pass found
batch U2 checking only stabilizer presence while incremental selection used
rolling-history debt. The [second grammar audit](SECOND_AUDIT_GRAMMAR_2026-09-05.md)
replaced both with shared causal accounting. It also verified that real
`Recursivity(depth=...)` instances already expose the metadata used by batch U5;
the first report's statement that they did not was incorrect. New tests use
real instances, including invalid and mutated depth declarations. U5 still
checks declared scale and nearby stabilizers rather than measuring the runtime
parent/child coherence inequality; no universal trajectory certificate follows.

The accompanying [mathematical audit](THEORY_CONTRADICTIONS_2026-09-05.md)
documents eleven prioritized findings, including the two spectral fixes above.
Unresolved findings with explicit counterexamples include:

- A symplectic diagnostic rejects a canonical rotation and accepts an
  anti-symplectic reflection; a single-state product is not a map certificate.
- The tetrad potential gradient does not equal the implemented EPI pressure;
  the graph-wave projection certificate describes a different Hamiltonian
  from the isotropic substrate flow.
- Zero-energy reduction is singular, yet the current certificate reports the
  regular-level dimension and a nondegenerate quotient.
- Heterogeneous nodal frequencies require `diag(nu_f)*L_rw`; replacing them
  with their mean produces incorrect relaxation rates and conserved weights.
- Phase wrapping alone does not imply universal potential bounds; average
  eigenvalue decay does not determine every perturbation's relaxation window.
- Disconnected graphs have component-wise equilibria; boundedness, convergence
  and absolute integrability require distinct hypotheses; tetrad completeness
  needs a precisely stated observable/equivalence class.

These findings do not redefine U1–U6 or solve an external open problem. They
identify where current engineering policy, numerical observation, and proof
claims must be separated. No blanket mathematical certification is warranted
by the passing software suite.

## Validation

Interpreter: Python 3.12.10 in `.venv312`; NumPy 2.3.3, NetworkX 3.5,
pytest 9.0.2. Initial full default suite: **2,411 passed, 12 skipped,
91 warnings**, 96.26 seconds. New regressions were run before their corresponding
fixes, exposing failures in paths not covered by that baseline.

Final full suite on the completed source: **2,519 passed, 12 skipped,
91 warnings**, 105.93 seconds. This is 108 more passing cases than baseline.
The same 12 skips comprise seven unavailable-JAX cases, three unavailable
scikit-learn test modules, and two pre-existing manifest/CLI format skips.
The warning count is unchanged; no claim is made about unavailable adapters.

Both the source distribution and wheel built successfully before and after the
changes. The final wheel contains all 447 Python modules, byte-for-byte equal
to the final working source. Name/version (`tnfr 0.0.3.5`), dependencies and
console entry points match the baseline wheel. Eleven classifiers and fourteen
keywords now appear in published metadata; the source archive retains README
and license. These are local build checks, not a release.

All 447 source modules parse, `git diff --check` passes, and local links in the
audit/testing guides resolve. Logs are retained locally under `tmp/`:
`repo-audit-baseline.log`, `repo-audit-final-tests-complete.log`, and
`repo-audit-build-after.log`.

The focused physics/core-physics run passed 1,121 cases; structural diffusion
plus advanced SDK passed 151; the operator suite passed 232 after its final
prior-IL correction. These overlap with the full suite and must not be summed.

Main reproduction commands, from the repository root:

```powershell
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
.venv312/Scripts/python.exe -m pytest tests/core_physics/test_dnfr_backend_consistency.py tests/test_dnfr_fallback_parity.py tests/test_validation_reporting.py -q
.venv312/Scripts/python.exe -m build --outdir tmp/repo-audit-build-after
```

## Compatibility and optimization scope

Public backend names and call signatures remain available. Profiling labels
now identify `shared_canonical`; old backend-specific fused timing keys are no
longer emitted. Canonical pressure metadata replaces the duplicate backend's
private metadata. Memoized validation retains its signature and tuple result,
but messages and accept/reject decisions now follow the canonical validator;
legacy context/window arguments cannot override that validator.

Numerical results intentionally change for previously incorrect weighted,
directed, isolated, cached and spectral cases. Correctness tests preserve the
EPI channel identity, readout definitions and U3 contracts rather than obsolete
outputs. EPI initialization in fixtures is explicit; these fixes do not add
direct EPI mutations to production execution.

Consolidation removes hundreds of duplicated source lines and eliminates
discarded alternative searches and all-unit weight-buffer allocations.
No general speedup percentage is claimed: timings from different full-suite
runs are not a controlled benchmark. Weighted cache fingerprints necessarily
inspect more state; performance on very large graphs and unavailable optional
backends remains a separate measurement task.
