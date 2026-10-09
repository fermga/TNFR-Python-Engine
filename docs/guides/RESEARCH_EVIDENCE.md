# Research evidence and frozen source restoration

Use this guide to inspect retained research artifacts and prepare their original
source for a separately admitted execution. An audit, a freeze, a prepared
workspace and an evaluated response are different records. None changes the
[research queue](../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
or independently establishes acquisition, scientific truth or physical identity.

## Shared owners and scientific responsibilities

| Owner | Responsibility | What remains outside it |
| --- | --- | --- |
| [`utils.io`](../../src/tnfr/utils/io.py) | Strict JSON syntax/value admission and shared file-writing mechanics | Scientific record schemas and exact rational tags |
| [`research.artifact_io`](../../src/tnfr/research/artifact_io.py) | Tagged exact records, bounded reads and hashes, receipts, exact ZIP inventory checks and exclusive JSON creation | Model laws, preparation, derivative validity and scientific stopping criteria |
| [`mathematics._validated_taylor`](../../src/tnfr/mathematics/_validated_taylor.py) | Shared source-box Taylor construction and separate reconstruction of retained polynomial/remainder arithmetic | A reconstruction alone does not regenerate derivatives or prove Picard inclusion |
| Model-specific research readers | Re-admit consumed source, law, clock, event and observation primitives; rebuild their declared bounds | Authentication of an archived execution or replay of omitted calculations |
| [`research.frozen_source`](../../src/tnfr/research/frozen_source.py) | Inspect the admitted freeze receipt or prepare its complete pinned source in a new workspace | Executing an archived evaluator, installing dependencies or certifying the restored runtime |
| Frozen protocol and source archive | Immutable declaration and source association for one experiment | Automatic authority to execute, revise or retry that experiment |

Current consumers share maintained mechanics. Archived helpers remain the
historical implementation that belongs to their protocol; do not update their
imports, serialization, hashes or criteria when refactoring current code.
Compatibility imports can delegate to a shared owner without rewriting any
archived file. Distinct existing schemas, including string-rational formats,
retain their own admission rather than being silently converted to another
format.

`decode_exact_tree` reconstructs fraction tags and retains ordinary finite
metadata, including Boolean flags. A consumer must still call `exact_record`
for an exact coordinate; decoding a tree does not admit a Boolean or float as
that coordinate. Small exact values need no float materialization:

```python
from fractions import Fraction as Q
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
)

record = decode_exact_tree(encode_exact_tree({"duration": Q(1, 64), "complete": False}))
assert exact_record(record["duration"]) == Q(1, 64)
assert record["complete"] is False
```

`write_json_once` creates a new research record exclusively and preserves an
existing or partially written file after failure. It is distinct from the SDK's
atomic replacement writer for ordinary exports. Sharing JSON admission does not
merge those write policies or make an incomplete record complete.

## Inspect evidence without executing it

Read a declared protocol, manifest or receipt through the strict JSON owner.
Admit exact tags before constructing fractions or comparing values: Boolean
integers, invalid denominators, nonfinite numbers and malformed intervals cannot
gain validity through equality with an expected value. Resource limits belong
to the selected consumer; a bounded read is not permission to allocate an
arbitrary decompressed archive.

Check the expected files, sizes, digests and exact ZIP inventory. Reject duplicate
or unsafe names and unsupported archive encodings; never extract an unchecked
archive. Hashes bind bytes to the declaration being checked. They do not
authenticate chronology, prove that a program produced those bytes or certify
an unverified mathematical premise.

Isolated evaluator-wiring tests must independently admit the pinned archive and
member bytes before parsing or compiling their function definitions. A separate
freeze audit may not run in the same test selection. Keep these mocked controls
distinct from importing an archived module or invoking its scientific producer.

Reconstruct each consumed derived field from the admitted primitive evidence.
A stored passing flag, cached interval or reported total cannot replace that
calculation. Retained Taylor arithmetic can be checked without regenerating
derivative coefficients: the validity of those stored derivative enclosures
remains an explicit premise unless the selected audit independently checks it.
Source correlation, full-state handoffs, law and event checks remain with the
model-specific reader.

The maintained class readout readers share
[`_reconstruct_readout_step`](../../src/tnfr/physics/_sine_class_readout_evidence.py)
for retained 54-coordinate observations and 55-coordinate storage/loss steps.
It re-admits the expected state and clock, method, positive Picard margin,
smooth-domain marker, Taylor coefficients and remainder before comparing the
rebuilt increment and endpoint. State vectors and nested coefficient rows must
be ordered; mappings and sets reject before materialization, and iterators are
consumed only through the declared row length or shared dimension cap plus one.
This adds no derivative
replay or provenance authentication. Each reader still owns its law, source,
event and attempt-budget checks, failure policy and observation combination.

Inspect incomplete and unsuccessful records as such. Missing observations are
unavailable, not zero; a completed prefix is not the requested endpoint.
Scientific consistency, numerical completion and a passing discrimination
criterion are separate conclusions. The [testing guide](../../TESTING.md)
owns the affected audit and independent-control selections.

## Restore the declared source before a later execution

A source archive can contain selected snapshots while naming an immutable base
revision for all other dependencies. Restore that full revision in an isolated
workspace before applying the archive's explicitly admitted overlays and
non-source files. Copying only listed snapshots over the current package can
leave unrecorded dependencies from a different implementation.

Verify the restored base, exact archive inventory, admitted overlays, protocol
and prior-artifact associations. Preserve every archived byte. Where a protocol
permits CRLF/LF normalization for comparison with a Git checkout, that permission
does not change the hashes of the archived bytes themselves. Do not widen such
a rule to arbitrary whitespace or code equivalence.

Restoration must not import or execute the archived evaluator, run acquisition,
install a law, create an attempt or produce a response. It also does not establish
that the required interpreter and dependency versions are available. Those are
separate execution preconditions declared by the frozen protocol. Keep the
original evidence unchanged and preserve an incomplete restoration if its
preparation fails; do not merge a partial workspace into a running experiment.

For the [matched four-history freeze](../../theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-frozen-evaluation),
the declared base is `fa8e98a9b1bdd755709da481de7fc092b57bfe65`. Its future
evaluator checks the complete `src` tree against that revision, not only the
snapshots included in its archive. A refactored current checkout can audit the
freeze but cannot substitute its runtime for the declared one. The immutable
`evaluation_status_at_freeze="not_evaluated"` records the freeze's historical
state; it is not a query about subsequent execution.

The maintained [restoration entry point](../../scripts/restore_frozen_source.py)
admits three explicit schema families: class nonlinear-readout v1, class
comparison v1, and class collective-forward v1. Each requires its own matching
protocol and source-snapshot schema, a complete pinned Git base and no runtime
overlays. Support for one does not admit an unrelated freeze format.

The [collective-forward admission](../../theory/nodal/SINE_CLASS_COLLECTIVE_FORWARD_PROTOCOL.md#sine-class-collective-forward-protocol)
retains the prior causal prediction as a separate byte-verified ZIP. Restoring
that association neither regenerates the prediction nor evaluates the new
complete-law response. Its immutable `not_evaluated` receipt records preparation
at freeze time; any later execution needs its own attempt and outcome.

Inspect this receipt, archive and pinned Git source without creating a workspace:

```sh
python scripts/restore_frozen_source.py --receipt docs/assets/sine_formed_classes/class-collective-forward-v1.freeze.json
```

When source preparation is explicitly needed, supply a new destination:

```sh
python scripts/restore_frozen_source.py --receipt docs/assets/sine_formed_classes/class-collective-forward-v1.freeze.json --destination C:/TNFR-frozen-collective-forward
```

The destination must not exist and must be outside the active repository, with
an existing unredirected parent directory. The utility refuses restoration when
an attempt or outcome already exists in the source evidence directory; the
comparison and collective-forward schemas also reject a retained export-error
record. It creates a detached worktree of the full pinned base and restores the
verified supplemental files, receipt and associated prior evidence. It does not
run the evaluator or certify interpreter and dependency compatibility. A prepared
workspace is neither an admitted
scientific attempt nor a global lock across copies: the research gate still
selects one execution workspace and preserves its first outcome. Do not modify
archived checks to make a different current runtime appear compatible.

## Preserve the first outcome

Only a separately admitted evaluation may invoke its archived entry point.
Retain the required exclusive intent/attempt record before the first scientific
call and refuse replacement of an existing attempt or outcome. Keep failure,
partial evidence, an inconclusive interval and export failure distinct. Do not
change source, precision, work limits or decision rules to repair a reserved
outcome; a correction requires separately identified evidence.

An export-error record preserves the failure metadata and points to the retained
attempt. Any partial output remains unchanged. It does not establish that the
complete in-memory report was successfully written.

The [benchmark lifecycle](../../benchmarks/README.md#running-and-reporting)
and each protocol own their declared execution procedure. This guide centralizes
maintenance responsibilities; it does not introduce a second experiment queue
or a universal schema for every scientific result.
