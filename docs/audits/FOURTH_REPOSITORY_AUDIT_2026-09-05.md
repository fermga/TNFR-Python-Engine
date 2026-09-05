# Fourth repository audit: ownership, configuration and reproducible experiments

Date: 2026-09-05. Version: `0.0.3.5`.

The fourth audit starts from the verified [third audit](THIRD_REPOSITORY_AUDIT_2026-09-05.md).
Its unchanged full baseline passed **2,880 tests, 11 skipped, 96 warnings** in
109.55 seconds. The third-audit wheel provides an immutable implementation
reference; Git HEAD alone does not include the preceding three uncommitted
audits. Their changes are preserved, and `manual/` remains outside the scope.
No commit, publication, deployment, dependency installation or external service
connection was performed.

Four parallel tracks examined cache ownership, configuration/initialization,
trace and laboratory restoration, and the SDK's experiment/topology APIs.
Independent review then exercised copied graph adapters, real operator cycles
and the signed-cache decoding boundary. Changes below address reproduced
failures; they do not establish universal correctness of the research engine.

## Findings and implemented corrections

| ID | Priority | Reproduced failure | Correction |
| --- | --- | --- | --- |
| D01 | P1 | Cache entries ignored requested array dtype, node order or adjacency representation | Include representation and ordering in keys and preserve the caller's numerical contract. |
| D02 | P1 | A copied graph shared its original's cache manager and node adapters; dispatch could modify the original | Check graph ownership centrally, create fresh per-graph cache state and bind adapters to the requested graph. |
| D03 | P2 | Clearing an LRU left stale capacity accounting; cache configuration and invalidation paths disagreed | Reset size bookkeeping and make the supported configuration/invalidation paths address their actual cache instances. |
| D04 | P1 | Persistent invalidation could resurrect old disk values; a memory-only store referenced an unset path | Correct memory-only handling and invalidate persistent derived state as well as memory entries. |
| D05 | P1 | Signed shelve values were interpreted before verification; the signature did not authenticate raw/pickle interpretation | Validate the outer container before interpreting the signed payload and authenticate mode plus payload in version-2 envelopes. Reject legacy signed entries for rebuilding. |
| D06 | P1 | Default reads and presets exposed mutable shared values to unrelated experiments | Return owned mutable fallbacks and independent preset data; retain graph-owned settings as explicitly mutable state. |
| D07 | P1 | Empty configurations selected global defaults; fragmented or unknown updates partially applied | Distinguish omission from an empty mapping and validate effective staged settings before applying them. |
| D08 | P2 | NaN/Infinity and Boolean text handling differed between configuration paths | Share explicit finite-value and Boolean parsing; preserve valid zero timesteps. |
| D09 | P1 | Initialization overwrote legacy aliases, including inactive zero frequency; temporary flags leaked between tasks | Reuse alias precedence and isolate temporary mathematical flags with context-local restoration. |
| D10 | P2 | Redis parsing and its security report disagreed on effective TLS, credentials and database index | Share effective configuration, decode/escape components correctly and validate both input routes consistently. |
| D11 | P3 | Legacy initialization/metric modules duplicated canonical default definitions | Replace full copies with compatibility re-exports, checking public symbols, values and identity. |
| D12 | P1 | Captured traces and transitions still referenced mutable live dictionaries | Detach nested recorded values at capture/save boundaries. |
| D13 | P1 | JSON RNG restoration failed on nested lists or partially changed live RNG state before rejecting | Validate complete Python/NumPy/component states in independent generators before committing restoration; honor failure results. |
| D14 | P1 | Snapshot checksums ignored EPI, phase, coupling and most telemetry | Hash the complete declared schema in new records; label legacy partial-hash verification explicitly. |
| D15 | P2 | Cached snapshots differed from stored records, and deleted records remained cached | Copy creation/load values and remove cache entries after database cleanup. |
| D16 | P1 | SDK topologies introduced bare integer nodes beside existing string/tuple nodes | Map generated topology indices to the existing node order and preserve node data/identity. |
| D17 | P1 | SDK seeds were assigned after RNG creation; fallback generators modified process-global state | Supply configuration at construction, keep local RNG ownership and apply network seeds consistently to topology builders. |
| D18 | P1 | Cloning collapsed graph kind, shared nested state and restarted RNGs; measurements retained a live graph | Preserve graph kind/identity, copy data and current RNG state, rebuild runtime caches and detach measured graphs. |
| D19 | P2 | Public templates could not import and overstated parameter/admissibility guarantees | Centralize operational SDK defaults, restore missing imports and state actual parameter and U3 behavior. |
| D20 | P2 | Grid/ring/star boundaries and invalid node-generation inputs corrupted or misrepresented support | Fill all grid nodes, avoid singleton self-loops, reject missing centers and validate generation ranges before state/RNG changes. |

Detailed reports:

- [Cache ownership, invalidation and signed storage](FOURTH_AUDIT_CACHE_2026-09-05.md).
- [Configuration and initialization](FOURTH_AUDIT_CONFIGURATION_2026-09-05.md).
- [Trace snapshots, RNG restoration and SDK cross-review](FOURTH_AUDIT_REPLAY_2026-09-05.md).

## SDK: concrete contracts and counterexamples

### Topology is built over existing nodes

`Network.small_world` and `scale_free` generated integer-indexed graphs and
added their edges directly. A six-node graph with string, tuple, integer and
frozenset labels thereby gained nodes without a structural triad. `grid` instead
silently omitted edges whose integer endpoints were absent. The shared
[topology helpers](../../src/tnfr/sdk/_topology.py) translate indices through the
actual node list. Tests retain exactly the original labels and node attributes
and verify connected support on the specified fixtures.

Default grids now use a near-square rectangle and a possibly incomplete final
row. Sizes 2, 3, 5, 7 and 11 no longer leave the remainder unused. One specified
dimension infers the other; two specified dimensions must accommodate all nodes
or fail before adding edges. Existing edge support is retained, matching the
SDK's additive construction convention.

Both ring builders now produce no edge for a singleton. An empty star is empty;
an explicitly unknown hub raises instead of creating an uninitialized node.
Random probabilities must be finite in `[0,1]`. Small-world construction uses
one Watts-Strogatz implementation with existing node labels, replacing the
fluent API's duplicated NumPy/Python rewiring loops. Its heuristic degree is
capped for small graphs; NetworkX's odd-degree convention remains applicable.

These methods construct graph support. They do not execute a dynamical Coupling
operator or certify phase compatibility. The live U3 check remains mandatory
when Coupling or Resonance actually executes.

### Seeds belong to experiments

Six builder paths and five template paths assigned `random_seed` after
`TNFRNetwork` had already initialized its generator. Repeating a declared seed
therefore changed sampled triads and topology. They now pass a
`NetworkConfig(random_seed=...)` at construction. Comparisons across topologies
and coupling levels start from the same sampled triad for a fixed seed.

The fluent API owns a NumPy RandomState when available and a Python Random
instance otherwise, including unseeded runs. Fallback creation and operation
overrides no longer reseed or consume Python's process-global stream. A seed
override for one `add_nodes` call uses a separate local generator and does not
advance the network's stream. The Simple SDK now uses the network seed by
default for `small_world` and `scale_free`, as it already did for `random`.

Seeded outputs can change because previously ignored seeds and a different
rewiring implementation now take effect. The promise is repeatability for the
same corrected implementation, backend, options and execution order. Python
and NumPy RNG backends are not required to generate identical streams.

Node generation validates count, finite endpoints, frequency constraints and
range ordering before graph/counter/RNG mutation. The existing `create_nfr`
factory still initializes the structural triad; this adds no imperative
replacement for canonical evolution. Unexpected errors during later factory
execution are not a full batch transaction.

### Clones and measurements have independent graph data

`clone()` formerly called `nx.Graph(original)`, losing directedness and parallel
edge keys. Nested node/edge values, configuration and histories remained shared,
and the RNG restarted from its seed. The shared
[graph-state copy helper](../../src/tnfr/sdk/_state.py) preserves all four standard
NetworkX graph kinds, node/key identity and copied mutable data. A clone receives
independent configuration and the current RNG state, so its next generated
nodes match the original's continuation without coupling the streams.

Runtime caches are excluded using the cache module's shared key catalog and
rebuilt by their owners. Independent review caught an initially omitted
`_node_cache` containing graph-bound adapters and locks after real operator
execution; that cache and its weak variant are now included. The regression
applies `[Emission, Coherence, Silence]`, measures and clones, then applies
Coherence to each graph and verifies that the other graph's recorded history
does not change. Opaque node objects retain identity without invoking their
deep-copy method.

`measure()` now returns a detached graph-data snapshot alongside its metrics.
Subsequent evolution cannot rewrite the graph represented by an earlier result.
This intentionally changes the former live-view behavior. Python deepcopy
rules still apply to other user data: functions/callbacks remain shared;
unsupported runtime objects raise instead of silently sharing mutable state.
This is not an arbitrary-runtime checkpoint or external-resource clone.

### Template availability and admissibility have distinct contracts

Importing `tnfr.sdk.templates` failed because it requested missing `SDK_*` names
from the structural constants module. Existing builder defaults now live in
[one operational SDK module](../../src/tnfr/sdk/_defaults.py). The two unresolved
template defaults, interaction strength 0.25 and inspiration level 0.4, follow
their existing parameter documentation. They are example settings, not newly
derived TNFR constants.

The template documentation now identifies existing limitations:
`connections_per_person` controls a rewiring ratio, not measured degree;
`inspiration_level` and `hierarchy_depth` are retained compatibility parameters
with no implemented effect. The organizational scaffold does not implement a
hierarchy or nested EPI construction. Implementing those domain models would
require a separate design and operator-level validation.

Independent review executed 99 seeded calls across six builders and five
templates, with 12 nodes, seeds 0/7/31 and step budgets 1/2/3. **85 completed;
14 raised at the live U3 gate.** Initial phases sampled across a full circle
do not guarantee an admissible named word. Supplying identical initial states
and support to the third-audit wheel reproduced the same rejection, establishing
that it is not a new operator regression. A preceding Coherence sweep resolved
the tested fixture, but no universal preparation bound is claimed. Documentation
now states that a rejected run may retain an executed prefix; U3 is not disabled.

The SDK baseline regression batch produced **44 failures and one passing
control**. With four further continuation/ownership cases, the two new SDK
modules contain **49 cases**. The full SDK plus validated-sequence integration
passed **160 tests, two warnings**, in 10.26 seconds. See
[topology tests](../../tests/sdk/test_topology_identity.py) and
[experiment/ownership tests](../../tests/sdk/test_experiment_reproducibility.py).

## Centralization and optimization scope

This pass removes duplicate definitions and state-dependent alternatives at
shared boundaries: graph/cache ownership, adjacency/buffer representation,
configuration staging, Boolean parsing, legacy defaults, trace detachment,
snapshot checksums, SDK example defaults, topology generation and graph copying.
Local and persistent invalidation now prevent stale derived values from
reappearing. Reusing supported representations avoids incorrect cache hits;
fresh graph ownership avoids cross-experiment mutation.

These changes do not establish an engine-wide speedup. Copying snapshots and
mutable defaults has a real cost. Conservative persistent invalidation and
rebuilding old signed caches can require recomputation. The tradeoff is explicit
state ownership and correct readout; no runtime or memory percentage is inferred
from passing tests or reduced source duplication.

## Compatibility and remaining mathematical limits

**Subsequent resolution:** the initialization-only seed limitation and reserved
template controls below are addressed in the
[remaining-contradictions resolution](RESOLUTION_REMAINING_CONTRADICTIONS_2026-09-05.md).
It also corrects the governing mathematical statements and implements a
restricted variational balance. This section preserves the fourth audit's
historical scope; the full mathematical bridges are not claimed as proved.

New signed cache envelopes are version 2; version-1 signed derived entries must
be rebuilt. Independent review found that restricting class lookup alone was
insufficient when Python's extension cache was already populated. The final
reader checks a primitive bytes/memo/framing opcode allowlist before creating
an outer unpickler. Secure writes use a fixed bytes-only outer protocol; the
requested inner serialization protocol remains supported. Three defensive
regressions verify that extension opcodes never reach the outer decoder.
Unsigned trusted-cache mode retains its explicit trust assumption.
Laboratory snapshot checksums detect changes to declared data, not malicious
writers: that separate local snapshot format still expects trusted pickle input.
The cache report states the exact authenticated-decoding boundary and its tests.

New laboratory snapshots use full-schema SHA-256. Old partial-hash records remain
readable with a warning describing unverified fields. RNG restoration covers
Python's global RNG, NumPy's legacy global RNG and component seeds, not external
generators or a complete executable graph. Six laboratory seed tests that failed
because optional `psutil` was absent now pass; unavailable memory telemetry is
reported as `None`.

Configuration injection/accessors isolate supported mutable values; low-level
exported nested definitions are not recursively frozen. Backend service settings
and graph integration settings remain distinct APIs. Unseeded initialization
with `RANDOM_SEED=None` is local to initialization; runtime consumers that require
an integer graph seed are unchanged. Temporary mathematical flags are context
local, while deliberate process-wide precision selection remains process wide.

The unresolved issues T02, T07, T08, T10 and tetrad completeness in
[the original theory review](THEORY_CONTRADICTIONS_2026-09-05.md) remain open.
State ownership and reproducibility corrections do not prove a variational
bridge, universal confinement, exact relaxation windows, continuous integrability
or a complete field basis. No research conjecture or general coherence gain is
claimed from these changes.

## Integrated verification

The final default suite passed **3,082 tests, 11 skipped, 97 warnings**, in
106.59 seconds: **202 additional passing cases** against the fourth baseline.
The remaining skips are seven unavailable JAX cases, three unavailable
scikit-learn modules and one longdouble precision case unsupported on this
Windows runtime. Warnings concern unavailable JAX and operator anti-pattern
diagnostics; one additional Silence-to-Coherence diagnostic comes from the new
snapshot-ownership regression. Skipped backends are not claimed as validated.

The root independently reran the 50 final cache regressions and 30 existing
laboratory snapshot/seed cases together: **80 passed**, no warnings, in 1.08
seconds. Focused counts overlap and must not be summed. The default suite selects
`tests/` and excludes `slow`; the laboratory suites are outside that default tree.

Both wheel and source distribution built successfully. All **454 source modules
and 107 typing files** parse and match both archives byte for byte. Package
name/version, Python/dependency requirements and console entry points match the
third-audit wheel. Whitespace checks passed; all **96 local links across 16 audit
documents** resolve. These are local verification artifacts, not a published
release or validation of every optional runtime path.

Local logs: `tmp/fourth-audit-baseline.log`, `tmp/fourth-audit-sdk-before.log`,
`tmp/fourth-audit-sdk-after.log`, `tmp/fourth-audit-final-tests.log` and
`tmp/fourth-audit-build.log`. Distribution artifacts are in
`tmp/fourth-audit-dist/`; the local artifact verifier is
`tmp/verify_fourth_audit_artifacts.py`.

Reproduction from the repository root:

```powershell
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
.venv312/Scripts/python.exe -m pytest factorization-lab/tests/test_snapshot_system.py factorization-lab/tests/test_seed_management.py -q
.venv312/Scripts/python.exe -m build --outdir tmp/fourth-audit-dist
.venv312/Scripts/python.exe tmp/verify_fourth_audit_artifacts.py
```

Runtime: Python 3.12.10, NumPy 2.3.3, NetworkX 3.5 and pytest 9.0.2. Standalone
scripts must import the working `src` explicitly; the environment also contains
an installed release. Redis tests use synthetic data and a local fake client,
not a live Redis server.
