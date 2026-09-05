# Fourth audit: snapshots and RNG restoration — 2026-09-05

## Scope and evidence

This pass examined recorded trace values and the factorization laboratory's
existing snapshot and RNG restoration APIs. It preserves the preceding three
audits and does not introduce graph checkpointing or transactional operator
execution. SDK persistence was assigned to the root audit separately.

The relevant authority is reproducible dynamics and traceability (canonical
invariant 6). Snapshot operations must preserve recorded EPI, phase, structural
frequency and pressure observations; they do not constitute new structural
operators or change the nodal equation.

Validation used Python 3.12.10, NumPy 2.3 and NetworkX 3.5 through
`.venv312/Scripts/python.exe`. The root fourth-pass baseline completed before
source edits: 2,880 passed, 11 skipped and 96 warnings in 109.55 seconds.

Additional baseline runs outside the default test tree established:

- Laboratory snapshot tests: 14 passed in 0.78 seconds.
- Laboratory seed-management tests: 10 passed and six failed in 0.77 seconds.
  All six failures were caused by an unconditional import of absent `psutil`
  during environment measurement.
- Existing replay-registration and external phase-gate tests: three passed in
  0.76 seconds.

The first new regression run reproduced **20 failures**. An additional trace
context regression also failed before its correction. After compatibility and
boundary cases were added, the new
[regression module](../../tests/test_replay_state_restoration.py) contains
**26 passing cases**. The final scoped integration passed **148 tests, without
warnings, in 1.55 seconds**:

```sh
python -m pytest tests/test_replay_state_restoration.py tests/test_replay_register_manifest.py factorization-lab/tests/test_snapshot_system.py factorization-lab/tests/test_seed_management.py tests/operators/test_grammar_debt_and_depth.py tests/operators/test_execution_boundaries.py -q --tb=short
```

These counts describe the scoped run; they do not replace the root's final
repository-wide result. Tests explicitly preserve and restore global RNG state.
RNG examples use seeds 0, 123, 99 and 777. Snapshot fixtures record three nodes,
modulus 143 and candidate 11. No physical stability or performance improvement
is inferred from these bookkeeping tests or runtimes.

## R01 — Recorded trace values changed after capture

[trace.py](../../src/tnfr/trace.py) wrapped graph dictionaries with
`MappingProxyType`. A proxy prevents assignment through that view, but still
reflects changes to its underlying dictionary. Capturing a weight of 0.5 and
then changing the live value to 0.9 changed the historical observation to 0.9.
Nested lists and dictionaries were shared as well.

Mapping field producers now return detached read-only top-level mappings. The
common recording boundary copies custom field payloads too, including nested
mapping proxies. Nested containers are copied but are not recursively frozen;
the documentation now distinguishes these properties. The general-purpose
`get_graph_mapping` utility retains its documented live-view behavior.

The same aliasing problem existed in
[DefaultTraceContext](../../src/tnfr/core/default_implementations.py): recorded
pre/post dictionaries and saved transition records shared caller-owned values.
Copies at record/save boundaries now keep these observations independent.
Regressions mutate both caller inputs and the context's retained records after
capture and verify that saved history remains unchanged.

## R02 — JSON RNG restoration failed or partially changed live state

[seed_management.py](../../factorization-lab/seed_management.py) converted only
the outer Python RNG state to a tuple. JSON also converts the internal state
tuple to a list, so restoration returned `False` with `state vector must be a
tuple`. A malformed NumPy state or missing component seed could fail after the
master seed and Python RNG had already changed. The reproducibility verifier
ignored that returned failure and could report success from its mock run.

Restoration now decodes the Python and NumPy states and validates them in
independent RNG instances before changing live state. It reads every component
seed first and rejects disagreement between the two stored master-seed fields.
Rejected payloads leave the manager, Python RNG and NumPy RNG unchanged. The
existing Boolean return contract is preserved, and the verifier honors failure.

Explicit seed zero is no longer replaced by generated entropy. Missing optional
`psutil` records unavailable memory telemetry as `None`; it no longer prevents
RNG capture. The existing laboratory tests now pass without installing another
dependency.

Regressions compare exact Python and NumPy continuations after a JSON roundtrip,
including cached Gaussian values. Separate cases verify rejection of a missing
derived seed, an invalid NumPy generator name and conflicting master seeds.

## R03 — Snapshot integrity did not cover the stored state

The old [snapshot manager](../../factorization-lab/snapshot_system.py) hashed
only modulus, stage, coherence rounded to six decimals, and node/partition
counts. Changes to EPI, phase, coupling values, sense index or candidate factor
survived its integrity check.

New records use a tagged full SHA-256 digest over every declared snapshot field
except the hash itself. Reads verify the payload hash, stored hash value and
requested snapshot identifier. Non-finite values are rejected before insertion
when encoding the full declared schema. Tests modify the persisted compressed
payload directly, reopen the database with a fresh manager, and require
rejection for each altered field.

Existing records carrying the old 16-character partial hash remain readable.
Their database load emits an explicit warning that nodal, topology and most
telemetry fields are not verified. They are not silently upgraded or presented
as having full-state integrity. A regression constructs an old-format record
independently and verifies this compatibility behavior.

The checksum detects accidental data changes. It is not writer authentication:
the existing compressor uses pickle and therefore expects trusted local data.
Verification occurs after decompression; the checksum is not an unpickling
safety boundary.

## R04 — Cached snapshots diverged from their stored records

Snapshot creation retained caller-owned nested values. Loading from the same
manager could therefore return modified data while reopening the database
returned the original record. Returned snapshots were themselves the cache's
mutable objects, so editing a loaded result also changed subsequent reads.
Database cleanup removed rows but left cached records available.

The manager now detaches creation inputs and returns independent values on
cached and uncached reads. Cleanup removes the corresponding cache entries
after committing deletion. Regressions compare cached and fresh-manager reads,
mutate returned values, and confirm that deleted records can no longer be
retrieved through the cache.

## Compatibility and limits

No operator, physical threshold, grammar rule, graph mutation or SDK API was
changed. Public snapshot creation/load and Boolean RNG restoration signatures
remain intact. New snapshots carry a longer tagged hash; legacy records retain
their explicitly limited verification path. Optional memory telemetry can now
be `None`, which consumers should interpret as unavailable.

The laboratory snapshot schema stores declared measurement dataclasses, not a
NetworkX graph. It does not encode graph kind, arbitrary node identities and
attributes, edge keys, full glyph histories, callbacks or RNG state. No graph
restoration function exists in that module. Accordingly, the report and module
documentation do not claim that loading it reproduces executable graph state.

RNG restoration covers Python's global generator, NumPy's legacy global
generator and the manager's component seeds. It does not restore independently
created `numpy.random.Generator` objects, other libraries' RNGs, graph state or
the captured environment. Exact continuation was tested on the installed
runtime; compatibility across Python/NumPy versions is not established.
The laboratory reproducibility verifier runs its existing mock experiment;
success is not evidence of a complete factorization trajectory replay.

Cached reads return the manager's recorded value rather than re-reading an
externally modified database on every call. Full integrity verification is
tested on database loads. Historical trace containers remain editable by their
owners; the correction prevents live source aliases from rewriting them, not
intentional edits to the history itself.

## Read-only SDK cross-review

The root audit requested an additional check of actual builder/template
evolution, beyond initialization-only tests. A new SDK snapshot-copy path
initially failed after `basic_activation`: cached `NodeNX` adapters under
`G.graph['_node_cache']` retained the original graph and its runtime locks.
Copying the graph raised `TypeError: cannot pickle '_thread.RLock' object`.
The same seeded flow passed against the immutable third-audit wheel. This was
reported to the root and cache owner; adding the adapter caches to their shared
runtime-cache exclusion resolved the observed copy failure. No SDK source was
edited by this sub-audit.

After that correction, 99 actual seeded calls across six builders and five
templates yielded 85 completed calls and 14 U3 rejections. All used 12 nodes
and seeds 0, 7 and 31. Cycle/step budgets were 1, 2 and 3; the ecosystem case
used `evolution_steps=10*k` to exercise its existing ten-step scheduling blocks
(which execute 3, 8 and 10 word repetitions cumulatively). Concrete rejections
included:

- `compare_topologies(node_count=12, steps=1, random_seed=7)` and seed 31.
- `resilience_study(nodes=12, initial_steps=1, perturbation_steps=1,
  recovery_steps=1, random_seed=7)`.
- `neural_network_model(neurons=12, activation_cycles=1, random_seed=7)`.
- `creative_process_model(ideas=12, development_cycles=3, random_seed=0)` and
  seed 7.

The first three cases also reject at budgets 2 and 3 before completing their
first affected word. The phase gate correctly reports no phase-compatible
neighbor for the requested Resonance or Coupling step. Unconstrained random
initial phases do not guarantee admissible execution of a named word.

To separate initialization changes from operator behavior, the seed-7
small-world topology and identical initial node scalars were supplied to both
the working source and the immutable third-audit wheel. Both rejected
`basic_activation(repeat=3)` at the same U3 gate. One preceding full-node sweep
of canonical Coherence allowed all three cycles on both implementations. That
is a fixture-specific operator-compatible preparation, not a universal bound
on the number of sweeps. The U3 gate should remain enabled; callers need either
explicit admissibility preparation or a documented rejected-run outcome.

The scripts under `tmp/fourth-sdk-cycle-probe.py` and
`tmp/fourth-sdk-shared-graph-probe.py` record the exact cross-review experiments.
The root owns any resulting SDK changes and their final validation.
