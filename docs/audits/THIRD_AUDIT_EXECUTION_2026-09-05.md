# Third execution and grammar audit — 2026-09-05

## Scope and validation

This pass examined operator dispatch, rejected applications, glyph-history
accounting, contract probes, and the integration between canonical sequence
validation and live execution. It preserves the existing operators, U2 debt
capacity, U3 phase gate, U4b context requirements and nodal equation. The first
two audits form the uncommitted baseline; their changes are preserved.

The focused baseline passed 291 operator tests with one warning. The two new
regression modules contain 50 cases and passed in 0.41 seconds:

- [Execution boundaries](../../tests/operators/test_execution_boundaries.py)
- [Validated sequence execution](../../tests/operators/test_validated_sequence_execution.py)

The final scoped integration run passed **506 tests, with two warnings, in
11.94 seconds**. This count includes concurrently added SDK tests; it is not
the repository-wide result. The warnings concern repeated Coherence and the
unavailable optional JAX dependency. Interpreter: Python 3.12.10 from
`.venv312/Scripts/python.exe`.

```sh
python -m pytest tests/operators -q --tb=short
python -m pytest tests/operators/test_execution_boundaries.py tests/operators/test_validated_sequence_execution.py -q --tb=short
python -m pytest tests/operators tests/sdk tests/physics/test_structural_integrity.py -q --tb=short
```

Tests use explicit graph attributes and operator sequences. The contract audit
uses its reproducible `n=16, seed=7` graph. Standalone catalog checks explicitly
set `PYTHONPATH=src`; pytest uses the repository configuration. No coverage or
performance improvement is inferred from these runtimes.

## E01 — Argument rejection occurred after structural mutation

`apply_glyph(G, node, "AL", window=-1)` previously executed Emission before
rejecting the history window. Invalid glyphs could also create graph-side
warning/cache metadata. The raw and object entry points now resolve the glyph,
validate the history window and require replayable history before obtaining
the mutable node adapter or executing an operator.

[operators/__init__.py](../../src/tnfr/operators/__init__.py) uses the existing
canonical name helpers for both English names and glyphs. Floats and booleans
are rejected as history sizes instead of silently passing through `int()`.
One-shot histories are rejected without consuming them.

[grammar_application.py](../../src/tnfr/operators/grammar_application.py) now
materializes and validates all targets before the first application. Actual
graph membership distinguishes a tuple or frozenset node from an iterable of
nodes. Missing later targets therefore do not leave an earlier target changed.
Regressions check node attributes, histories, grammar counters, graph metadata
and cache creation on rejection.

## E02 — A fallback executed effects belonging to the rejected operator

On a node without a recent destabilizer, Self-organization was replaced by
Coherence in the grammar layer, but its wrapper still created a nested EPI.
The recorded history contained `IL` while THOL-specific structure appeared.
Similarly, a rejected Dissonance wrapper propagated the fallback's pressure
change as if Dissonance had run. Transition could clear latency metadata before
its own precondition rejected the application.

The public [Operator.__call__](../../src/tnfr/operators/definitions_base.py)
now owns argument checks, hard invariants, the existing precondition policy
and grammar selection before entering a subclass workflow. A fallback uses
its actual registered operator class. Subclass behavior runs in the protected
`_execute` hook, and a selected glyph is applied without a second grammar
selection. This affects Coherence, Dissonance, Emission, Mutation, Reception,
Self-organization, Silence and Transition. Their established precondition
configuration policies are retained.

Regression tests verify that rejected THOL creates no child, rejected OZ adds
no dissonance propagation, and fallback metrics identify the operator actually
executed. Emission, Reception, Silence and Transition precondition failures
leave the tested state and metadata unchanged.

## E03 — Contract probes could certify a substituted operator

The Dissonance audit probe had no prior U4a handler. Live selection executed
Coherence, while the former Dissonance wrapper made the resulting pressure
change appear compatible with the requested contract.

[physics/integrity.py](../../src/tnfr/physics/integrity.py) now prepares the
canonical prerequisite context and checks the recorded operator after every
probe. A regression forces a Coherence substitution and verifies that it
cannot certify Dissonance. The OZ pressure contract is measured directly after
the operator; a later network-wide pressure recomputation is a separate
observation and cannot replace that direct measurement. THOL probes receive
actual prior Coherence and Dissonance applications before measurement.

## E04 — A valid word was silently rewritten during execution

Canonical validation accepted `AL, OZ, IL, SHA`, with IL as OZ's future U4a
handler. Runtime selection only inspected past history and executed
`AL, IL, IL, SHA` instead.

The new immutable [grammar_execution.py](../../src/tnfr/operators/grammar_execution.py)
constructs execution context from canonical validation of the actual operator
instances, preserving metadata such as Recursivity depth. Each step can supply
only the future U4a handler of its validated suffix. The current operator must
match that step. U2 debt, the U3 phase gate, prior IL and the recent destabilizer
window remain live checks; the validated prefix does not supply unexecuted
history or pre-bank stabilization credit.

The context is used by [structural.run_sequence](../../src/tnfr/structural.py)
and the shared [SDK sequence executor](../../src/tnfr/sdk/simple.py), including
[TNFRNetwork.apply_sequence](../../src/tnfr/sdk/fluent.py). A live rejection
inside a validated word now raises before that operator executes instead of
silently substituting it. Standalone selection and the executor's explicit
`validate=False` mode retain incremental fallback behavior.

The SDK also checks the existing compatibility validator's `passed` result;
previously a returned failure could be ignored. That layer remains distinct:
for example, it rejects consecutive Expansion even when canonical U2 debt is
within capacity. This pass preserves that contract rather than removing its
additional restriction. All **15 predefined fluent sequences** were checked
against both canonical instance validation and compatibility validation; all
passed, so no catalog word was changed.

Tests reproduce exact execution across all three entry points, reject an
invalid debt prefix before mutation, reject new live debt above capacity, and
verify that context cannot supply missing prior IL, a recent destabilizer,
phase compatibility or invalid U5 depth. A deliberately failing future handler
leaves no permission stored on the graph.

## Compatibility and remaining limits

Public operator and glyph return values remain unchanged. The protected
subclass hook now separates preflight from execution. The sequence context is
an optional execution argument; existing standalone callers need no change.
Two intentional rejection changes are visible: invalid history sizes fail
earlier, and a blocked step in a validated sequence raises instead of silently
executing another operator. Invalid legacy-validation results are now honored.

These changes do **not** make operator execution transactional. Unexpected
implementation errors, optional postcondition failures, monitor failures or
user hooks can occur after mutation. Already completed nodes or sequence
prefixes remain committed. In particular, a future U4a handler can fail or be
interrupted after its trigger has executed; callers must inspect the resulting
history and state. Context itself is never persisted on the graph.

The regression evidence establishes the specified dispatch and rejection
boundaries, not universal rollback or continuous-time convergence. No new
physics threshold, operator contract, transport law or stability theorem is
introduced.
