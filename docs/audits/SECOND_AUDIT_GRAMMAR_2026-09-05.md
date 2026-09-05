# Second grammar and operator audit — 2026-09-05

## Scope and evidence

This pass examined remaining U2 accounting, U5 depth metadata, classification
duplication and glyph-history recording. It follows the nodal equation
`dEPI/dt = nu_f * DeltaNFR`, the existing U2 absorption capacity
`U2_DEBT_CAPACITY = 2`, and the existing U4/U5 relaxation window. It introduces
no operator, physical threshold or continuous-dynamics stability claim.

The baseline `tests/operators` run passed 232 tests with one warning. The first
new regression run produced 19 failures and four passes. After implementation
and additional legacy-history cases, the combined operator, core-physics and
SDK run passed 610 tests with 13 warnings in 11.26 seconds. Those warnings
included the existing repeated-Coherence warning and unavailable optional JAX.
These counts describe the scoped run, not the root task's final full suite.

Interpreter: `.venv312/Scripts/python.exe`. Commands:

```sh
python -m pytest tests/operators -q --tb=short
python -m pytest tests/operators/test_grammar_debt_and_depth.py -q --tb=line
python -m pytest tests/operators tests/core_physics tests/sdk -q --tb=short
```

The new [regression module](../../tests/operators/test_grammar_debt_and_depth.py)
contains 59 cases, including 18 from the independent U4b cross-review and seven
history-container checks below.
The scoped operator, core-physics and SDK run after the U4b correction passed 630 tests with 13
warnings in 11.14 seconds. All sequences below use glyph abbreviations for the canonical
English operators. Tests use fixed graph attributes; no random seed is required
for these deterministic sequence and bookkeeping checks.

## G01 — U2 could be bypassed by delayed stabilization or neutral history

Before this pass, batch validation accepted `AL, VAL, VAL, VAL, IL, SHA` because
the sequence contained IL. Incremental validation also forgot outstanding debt
after sufficiently many neutral EN operations displaced the destabilizers from
its six-entry context. Repeated early IL operations could provide negative
balance credits for later destabilization.

Both paths now use [grammar_debt.py](../../src/tnfr/operators/grammar_debt.py):

```text
destabilizer: debt_next = debt + 1
stabilizer:   debt_next = max(0, debt - 1)
neutral:      debt_next = debt
```

This is discrete accounting for uncompensated operator obligations. The cap
comes from the existing physics derivation; the counter itself is not a
measured DeltaNFR field or a proof of convergence of its continuous integral.
Batch validation rejects the first prefix exceeding the cap. Later
stabilization cannot retroactively make that prefix admissible. The existing
whole-word requirement to contain a stabilizer remains in place.

Incremental selection reads a persisted node counter. The central
[push_glyph](../../src/tnfr/glyph_history.py) recording path advances it once per
application before old history can be lost. Neutral operators and trace sizes
of zero, one or seven no longer erase obligations. The U4 context window retains
its separate role; it does not expire U2 debt.

A stabilizer remains admissible when it reduces an already over-capacity legacy
state. New destabilization above capacity is still rejected. This repairs the
previous behavior that rejected IL and simultaneously suggested the same IL as
an allegedly valid fallback. U4 requirements for THOL continue to apply.

Read-only sequence simulation maintains a separate shadow counter, includes only
accepted operators and restores both the original history and counter, including
absent metadata. Malformed saved counters reconstruct from available history.
`reset_debt_from_history` explicitly supports deliberate replacement of a full
trace. Restoring a snapshot with its counter preserves obligations older than
the snapshot's bounded history. Already-truncated legacy history without a
counter cannot reconstruct events that were never retained.

## G02 — Selector execution recorded every glyph twice

With canonical selection enabled, `_apply_glyphs` invoked `apply_glyph`, which
already recorded the executed glyph through `push_glyph`, then invoked
`on_applied_glyph` to append it again. A single selected Expansion produced
`[VAL, VAL]`. This altered the context seen by later U2/U4 checks.

The redundant selector append was removed. Recording remains centralized in the
actual operator application. The standalone `on_applied_glyph` compatibility
entry point still records an explicitly reported application and advances the
same debt recurrence. Callers should not report a second event for an operation
that `apply_glyph` already recorded.

## G03 — U5 metadata existed, despite contrary validator diagnostics

Real [Recursivity](../../src/tnfr/operators/recursivity.py) instances already
exposed `depth`. The first audit repeated an inaccurate source comment asserting
that U5 was dead code until such metadata existed. That conclusion was wrong.
`AL, Recursivity(depth=3), SHA` already fails the batch U5 check; inserting IL
within the existing scale window makes the declared sequence admissible.

New tests verify real depth declarations at two, three and six, rather than
relying only on a proxy. Nonintegral values, booleans, NaN and infinity were
previously accepted by the constructor's `depth < 1` comparison. NaN also made
the U5 `depth > 1` test silently false. The constructor now reuses the existing
positive-integer validation primitive, and U5 validates live depth metadata on
every call, including after a cached static preflight. Diagnostics describe
only the current sequence instead of falsely asserting no operator has depth.

This check verifies a scale declaration and nearby stabilizer presence. It does
not measure `C_parent >= alpha * sum(C_child)`, certify nested runtime identity,
or establish that declared depth changes the execution depth of the low-level
REMESH transform. Those distinctions remain explicit limits.

## G04 — Shared classification and serialized glyph normalization

`config.operator_names` maintained independent frozenset literals for generators,
closures, destabilizers and transformers while the grammar derived the same
sets through `physics_derivation`. These config exports now call the existing
derivations. Their values and public identifiers remain unchanged.

`glyph_function_name` now resolves serialized `Glyph.VAL`, lowercase codes and
capitalized canonical names consistently. Dynamic selection delegates valid
name resolution to that helper, and debt accounting uses the same helper.
Regression cases verify that these aliases do not erase outstanding pressure.

The legacy full-sequence registry is checked through canonical validation. Its
named words remain intact; named fragments and complete sequences have different
contracts and were not merged solely because they share glyph subsequences.

## G05 — Incremental U4b incorrectly expired the prior Coherence prerequisite

Independent review found that the batch validator accepted the complete word
`AL, IL, EN, EN, EN, EN, EN, EN, VAL, THOL, VAL, ZHIR, IL, SHA`, while the
incremental validator rejected its ZHIR step because IL was outside the six-entry
context. Enlarging the window to twenty made the same candidate pass. Recent
THOL supplied the U4a handler, VAL supplied the recent U4b destabilizer and the
U2 debt before ZHIR was one; only the lifetime stable-base prerequisite differed.

The canonical requirement is a prior IL, plus a recent destabilizer. The new
`_grammar_prior_coherence` boolean preserves the first fact across trace
eviction. Recording, dynamic selection, both Mutation precondition entry points,
readiness diagnosis and Mutation telemetry now read the shared lifetime-context
helper. The recent destabilizer still expires after the existing relaxation
window; an old IL alone does not make Mutation admissible.

Shadow validation advances the marker only for accepted operations and restores
its original presence and value, including after validation exceptions. The new
`reset_grammar_state_from_history` explicitly reconstructs both U2 debt and U4b
context after deliberate full-trace replacement. The existing debt-only reset
retains its narrower behavior. Snapshots should preserve both bookkeeping keys.

Eighteen regression cases cover full retained histories with context windows
three, six and twenty; evicted IL; recent-destabilizer expiry; deliberate reset;
shadow restoration; callback recording; the four Mutation readers; and serialized
`IL`, `il`, `Glyph.IL` and `Coherence` tokens with a zero-length retained trace.
The first ten new cases produced eight failures and two passes before correction;
all four reader regressions independently failed before their shared-helper fix.

Minimal deterministic reproduction from the repository root:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import networkx as nx
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_memoization import validate_sequence_optimized

history = ["AL", "IL"] + ["EN"] * 6 + ["VAL", "THOL", "VAL"]
graph = nx.Graph()
graph.add_node(0, EPI=0.6, glyph_history=history)
assert validate_sequence_optimized(history + ["ZHIR", "IL", "SHA"])[0]
assert validate_candidate(graph, 0, "ZHIR").allowed
```

## G06 — Read-only validation consumed one-shot history iterators

Assigning `iter(["VAL", "VAL"])` as a node's `glyph_history` caused the recent
context read to consume both tokens before U2 reconstructed its debt. A third
VAL was consequently allowed, whereas the same history as a list, tuple or
deque correctly rejected it. Shadow sequence validation had the same issue.

A shared guard now rejects `collections.abc.Iterator` histories with a clear
`ValueError` before consumption. It neither replaces the history nor changes
graph metadata. Callers can materialize their iterator before assigning it to
the graph. The recording path retains its existing iterable-normalization
behavior; this correction is limited to read-only incremental and shadow checks.

Seven regressions verify identical accept/reject decisions for list, tuple and
deque histories, plus unchanged iterator identity, contents and node metadata
after rejection for both list iterators and generators at both validation entry
points. The four rejection cases failed before the guard was added; all 154
focused grammar tests pass after correction:

```sh
python -m pytest tests/operators/test_grammar_debt_and_depth.py tests/operators/test_grammar_dynamics.py tests/operators/test_grammar_memoization.py -q --tb=short
```

## Compatibility and remaining limits

- Previously accepted sequences with an over-capacity prefix are now rejected.
  Sustained dynamics can select a stabilizing fallback earlier, so trajectories
  depending on the old bypass are expected to change.
- Existing public call signatures remain. Private persisted node fields
  `_grammar_u2_debt` and `_grammar_prior_coherence` preserve outstanding U2
  obligations and lifetime U4b context across trace eviction. Both should be
  retained with graph snapshots. Legacy traces cannot reconstruct evicted IL
  when the marker was not retained.
- Read-only incremental and shadow validation now reject one-shot iterator
  histories explicitly. Replayable list, tuple and deque histories retain their
  behavior; rejection preserves the iterator and graph for caller-side recovery.
- U2 bookkeeping is independent of physical time integration. Low-level callers
  that bypass grammar validation can still produce invalid recorded sequences;
  the counter records those states and permits subsequent stabilizing recovery.
- U5 is a metadata/context check with the limits described above. U6 remains
  graph telemetry. Neither is certified merely by a sequence returning true.
- No measurements of `C(t)` improvement, phase synchronization improvement or
  runtime speedup are claimed from these changes. The verified outcomes are
  grammar decisions, history cardinality, debt preservation and input validity.
