# Independent review of final launch integration

Date: 2026-10-29. Reviewer: `/root/admission_foundation`.
Caller/recorder author: `/root/admission_guard`; declaration, quota delta and
freeze/harness authorship: `/root`.

**Verdict: PASS for the reviewed source and prospective integration controls.**
The controls have not yet been executed. Successful frozen controls, the
producing source/protocol freeze, the ordinary prerequisite check and the
separate final auditor review remain prerequisites to the single reserved run.
This source review is not a trajectory result, actual-entry finding, physical
readiness review or receipt for a not-yet-created producing protocol.

I authored the earlier reference adapter and contributed the M1--M9 source-aware
comparison analysis. I did not author these caller/recorder implementations or
the quota delta. I suggested retained-record checks and requested tighter
control/freeze associations during this review. The review is independent of
the implementation authors but is not blind; it is not an independent new
review of my earlier mathematical contribution. No mathematical owner changed.

The pre-freeze fixture rename from `auditor_controls.py` to
`test_audit_attempt.py` was reread in the harness. The prior receipt is preserved
as `independent-launch-review-before-fixture-rename.md` (10714 bytes, SHA-256
`525167dd6110e0628ca35ec66daf7931ca6d6bfba066487ec8e7a9e3f0f29403`).
Only that harness association changes below; the review verdict is unchanged.

## Final reviewed bytes

Paths below are relative to `C:/TNFR-Python-Engine`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/tnfr/research/formed_two_port_run.py` | 24589 | `6f4c7c43fbc22f073610af70ef4b69b538eaf3ccaeae3db06fad2a651d2b863e` |
| `tests/research/test_formed_two_port_run.py` | 13371 | `9947cc6a8c8695bfba4fbf012765eaeb6b9976cc8b9de8411d1900d79cae317a` |
| `src/tnfr/research/formed_two_port_launch.py` | 12433 | `509d730fe8fb9e014d69692ee1d03236819fa7089b5b5e06e8ab8d2cb5dfeca9` |
| `tests/research/test_formed_two_port_launch.py` | 10423 | `1e1d9ae48f3bd4ff66876da904fbd27f92e9eaba9007abceb452a48b5b4b86b7` |
| `artifacts/formed-two-port-attempt-20261029/prospective-attempt.md` | 6699 | `857869e4b4c1abe2f36becd8b924d0191fc5db6b17d8bb42795a12431a4bd195` |
| `artifacts/formed-two-port-attempt-20261029/worker.py` | 175 | `46a4166eb369c4a8f1d866346144f7269dac72b0ef30d4b9d8d0f4edfbd2a6f9` |
| `artifacts/formed-two-port-attempt-20261029/launch_once.py` | 706 | `8674aeeaeb51c2b3fdcbf7aad6a1457424dcb1ce07c29d2b219c1a48a322025a` |
| `artifacts/formed-two-port-attempt-20261029/record_attempt.py` | 9959 | `4a038ff950c4ada01b3ee51d9850cb2dcd7a7e45a7932555c46e5b1adfe87abe` |
| `artifacts/formed-two-port-attempt-20261029/test_record_attempt.py` | 7911 | `961195e8741653191ec7ed13b10380e76e47d4899ed6ed3aacd5558b7e127935` |
| `artifacts/formed-two-port-attempt-20261029/freeze_controls.py` | 1774 | `1ec8b02236305d8f4ba7af4995a9c010efc88f7b3e487add17b0f700335bc394` |
| `artifacts/formed-two-port-attempt-20261029/run_controls.py` | 2892 | `8f7954778122812c1d60f046feaa710c54b939d265c5c18680733bc42b5bb9ca` |
| `artifacts/formed-two-port-attempt-20261029/freeze_producer.py` | 3767 | `b1e7b7dc1ca563ae2858a46924e6f12719e73653a2853d4bf36ff33d696f4123` |

I read the complete new caller and recorder and their complete synthetic test
files. I compared the runner and its test against this stage's before-edit.zip:
the delta only adds/declares the 64-KiB caller reserve, subtracts it at initial
and subsequent auxiliary admission, documents it, and updates the small quota
fixture. No field, target, propagation, endpoint or retained-metric mathematics
is altered by that delta.

## Preserved scientific and resource contract

The declaration retains the original actual formed-source family, phi, errors,
labels, means, continuous lifts, fixed contacts and both complete rows. The
eighteen-dimensional affine reference is not a reset of that source. Exactly
4096 quarter-steps, order eight, dyadic128, the original rho and target criterion
remain selected. Stored domain values are the twenty positive slacks after
subtracting rho; the declaration now makes that convention unambiguous.

The new caller reserve reduces the child allowance rather than enlarging
64 MiB. Initial existing files are counted conservatively. Normal child writes
cannot spend the supervisor's 2 MiB, the caller's 64 KiB or the child's terminal
1 MiB; the terminal write may use only its own remaining allowance. The quota
test exercises the exact boundary and preserves existing/partial bytes.
Step evidence, including common fields, keeps its separate 256-MiB allowance.

The unchanged supervisor still gives the worker at most 3590 seconds inside
3600 seconds, including admission and writes. The caller starts its own clock
immediately before invoking it, then includes terminal metadata reading, its
own durable verdict write and receipt verification before checking the same
3600-second limit. A late caller close revokes the returned execution status.
No pilot, changed parameter, resumed prefix or extra reference step is exposed.

## Caller and durable return

The caller invokes the supervisor once with the exact fixed limits and paths.
An existing attempt prevents invocation; failure does not authorize an implicit
retry. It consumes bounded terminal JSON/log metadata, checks the actual
post-supervisor-write return against the retained pre-write snapshot and
receipts, and requires a complete conditional worker terminal record. Its final
step existence/size check is deliberately not represented as a full prefix
audit or independent endpoint proof.

The in-attempt verdict is an immutable pre-its-own-write snapshot of at most
32 KiB. The returned object records the later clock measurement and that
snapshot's actual receipt. It always leaves `reference_success_admitted=False`
and requires the documentary audit. The wrapper emits the bounded returned
object and exits according to execution status. Snapshot success alone is
insufficient at either the supervisor or caller layer.

The post-caller-write return is capped at 31 KiB. The external envelope is
separately checked against 32 KiB, including its metadata; an overflow retains
a small failure envelope without a successful returned object. Thus the caller
snapshot plus external envelope fit the reserved 64 KiB. No duplicate raw stdout
log is written. The external file is counted against auxiliary evidence even
though its path is outside the attempt directory.

The recorder uses two bounded, unbuffered pipe readers. Stdout retains at most
31 KiB and observes at most one additional byte; stderr retains at most 1 KiB
and observes at most one additional byte. Overflow, stderr or reader failure
requests termination. Hashes are explicitly marked as complete-stream hashes
or observed-prefix hashes, never asserted to describe unread output. Main wait
and thread cleanup share the outer 3630-second documentary allowance; cleanup
does not add another unrestricted ten seconds. Incomplete cleanup fails closed.

The external acceptance predicate requires the actual wrapper exit zero, no
timeout/capture/output error, a valid post-write caller schema, true execution
and write-within-budget flags, and finite nonnegative caller elapsed time below
3600 seconds. The same predicate drives the recorder summary and exit. Partial,
oversized, malformed or unexpected-stderr output cannot become success merely
because the wrapper exited zero.

The external envelope's persistence is documentary capture after execution.
It does not allocate further worker time, repair a missed scientific deadline
or claim to authenticate its own future write. The separate later audit has its
own declared 120-second/8-MiB documentary budget. Missing external evidence or
an incomplete audit leaves success unavailable, even if an earlier snapshot
looked successful. This boundary is explicitly disclosed by the protocol.

## Controls and source associations

The selected integration controls use fabricated records and fake subprocesses;
the caller fixture forbids a real supervisor and the recorder fixture forbids
a real Popen unless replaced by its inert fake. The wrapper test replaces the
caller before runpy executes the wrapper. They cover missing/partial/changed
terminal evidence, no retry, quota overflow, caller write overrun, malformed
returns, observed-exit mismatch, pipe overflow, stderr and launch/wait/read/
timeout failures. Clock-overrun testing uses a synthetic clock change, not an
assumption that a real process starts within a tiny timing interval.

The control harness selects five explicit sets, not the earlier retained-target
or native-supervisor tests. Its 180-second allowance is a test-harness budget,
not an enlarged scientific run. It rejects existing log/JUnit/result/basetemp,
verifies the frozen archive and source receipts, preserves partial JUnit parse
errors and requires positive test count with zero skips/errors/failures. The
documentary auditor and its fixtures remain under the separate substantive
audit review; this record does not substitute for that review.

The producer-freeze builder now explicitly associates controls-result.freeze
with the exact controls-freeze being consumed. It checks the control exit,
timeout and counts, freeze/archive/log/JUnit receipts, every current frozen
source and the unchanged installed-Python inventory. This resolves the earlier
gap in which a passed control record could have belonged to another freeze.
It preserves the Git base, all installed TNFR Python source, mathematical and
source owners, primary target files, runtime configuration, wrapper/caller/
recorder, substantive reviews and control receipt in the producing archive and
protocol. The archive is exclusively created and its members verified.

The generated protocol uses the closed schema expected by the bounded worker,
including exact runtime, complete policy, frozen worker-entry association and
the actual archive/review/control receipts. Its future bytes must still be
created and verified before launch; this review does not invent their hashes.
Content hashes are associations, not authentication of execution or authorship.

## Scope of the result

No unresolved blocking defect was found in this final launch/control scope.
The subsequent documentary audit must bind source and target records, all
26 fields per certificate, full center/radius chronology, strict tube/Picard
slacks, retained endpoint bounds and the actual post-write external outcome.
It may check rational relations among existing data but cannot regenerate
fields, signs, roots or trajectories or promote a failed execution to success.
Failure preserves the last verifiable contiguous prefix and partial evidence;
it establishes neither non-entry nor a physical conclusion.

This review used source reads, text-diff/receipt metadata only. No test,
factory, trigonometric or target-sign evaluation, Taylor step, producer,
native child or reserved attempt was executed by this reviewer.
