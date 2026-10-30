# Complete streamed evidence for the formed two-port reference

Author: `/root/joint_review_controls`; 2026-10-29.
Scope: design/read-only audit of certificate storage and failure retention.
Inspected owners: `_validated_metric.py`, `_rational_interval.py`,
`research/artifact_io.py`, `utils/io.py`, and existing research runners.
No field, Taylor step, trigonometric operation, target assessment or test ran.
No maintained file was changed.

## Recommended small design

Keep the new runner source-specific. It needs one immutable attempt declaration,
one exact shared-certificate-constants record, one file per completed step,
bounded logs, and one terminal record when time/space permit. A separate generic
evidence framework or a 4096-certificate in-memory result is unnecessary.

Use a fresh attempt namespace. Exclusive creation of `attempt.json` must precede
source/target admission or a field call, and any existing attempt or partial
attempt must reject replay. The supervisor owns the no-resume boundary. An audit
of a completed or failed prefix is read-only; it must never reconstruct a worker
state and silently continue propagation.

The attempted protocol, exact source/target associations, field implementation,
node/edge order, degrees, affine offsets, metric, coordinate convention, clock,
step/order/precision, numerical owner versions, limits and output schema must
be bound before a reserved field is evaluated. A hash authenticates bytes, not
the mathematics, chronology or acquisition of the source.

## Preserve all 26 certificate fields

`ValidatedMetricTaylorStep` currently has the following fields. Factoring fixed
values is lossless only when a decoder can reconstruct this entire dataclass
record exactly and every step verifies equality with the shared values.

| Fields | Required representation |
| --- | --- |
| `time`, `duration`, `order` | Exact chronology and integer order; retain even when the schedule also implies them. |
| `metric` | Full exact matrix in a shared immutable record, referenced by its byte receipt. A diagonal representation may only be used with an explicit exact reconstruction rule. Keeping the actual matrix once is simplest. |
| `coordinate_projection_factors` | Exact shared vector; equality must be checked on every emitted certificate. These bounds must not be replaced by a recomputed floating square root. |
| `initial_center`, `initial_radius`, `initial_box` | All three retained. A center/radius does not make the stored coordinate-box boundary redundant for complete certificate reconstruction. |
| `tube`, `picard_interior_margin`, `domain_lower_bounds` | Full intervals and exact margins, with the declared ordering of all twenty edge-domain values. Do not replace them by one Boolean. |
| `growth_rate_upper_bound`, `growth_factor_upper_bound`, `propagated_initial_radius` | Preserve the actual values, including zero/one constants. The scientific justification of scalar growth zero is separately bound to the field/domain proof. |
| `center_polynomial_bounds`, `local_remainder_bounds`, `center_solution_endpoint` | Preserve all component intervals. An endpoint alone is not the Taylor/remainder certificate. |
| `local_coordinate_error_bounds`, `local_metric_error_upper_bound` | Preserve componentwise and metric errors without independently recomputing/rounding them during serialization. |
| `endpoint_center`, `endpoint_radius`, `endpoint` | Preserve the full next-state enclosure and coordinate projections. The next input is the retained center/radius, not a fresh box. |
| `growth_certificate`, `method`, `conditional_premises`, `prescribed_time_interval` | Preserve explicit nulls and complete strings/tuples. The selected autonomous scalar-growth method expects `growth_certificate=None` and `prescribed_time_interval=None`; reject an unexpected method rather than silently discard its extra certificate. |

For minimum implementation risk, factor only `metric` and
`coordinate_projection_factors`; keep the other 24 fields in each step. A
small list of certificate field names in the schema allows an audit/test to
reject missing, overlapping or unrecognized fields. `shared_fields` and
`step_fields` must partition the actual dataclass fields exactly. The shared
record is retained once, not merely named by a formula in an absent file.

Prefer `dataclasses.asdict(step)` as the starting projection: the interval
dataclass becomes `{lo: Fraction, hi: Fraction}` and all ordinary fields remain.
`artifact_io.encode_exact_tree` handles Fractions and JSON trees, but **does not
handle dataclasses or interval objects directly**. Encoding the projected tree
preserves both exact interval endpoints, including nonzero width. Never use a
midpoint, decimal display, float, rounded rational or endpoint-only summary.

`json_loads` is the appropriate reader because it rejects duplicate keys and
invalid JSON numbers. Then use `decode_exact_tree` and schema-specific
`exact_record` on all mathematical scalar fields. The tree encoder/decoder
permits finite floats as metadata; that is not admission of a float in a
certificate coordinate. Explicitly keep wall-time metadata separate. An interval
reader must retain the stored lo/hi values and ordering rather than silently
rounding a malformed rational interval into another accepted one.

## Chain with constant memory use

A step file such as `steps/step-0000.json` should contain:

- Schema/version, step index, source/protocol binding and shared-constants receipt.
- Previous complete step's path/byte-count/SHA-256 receipt, or an explicit null
  for step zero; this also fixes chronological order.
- The 24 nonshared certificate fields, with their exact values.
- Only bounded diagnostic metadata that does not substitute for the certificate.

For step zero, enforce the declared exact zero center/radius and time zero. For
every later step, check exact equality of `initial_center` and `initial_radius`
with the preceding retained endpoint center/radius, exact time advancement,
duration 1/4, order eight, shared constants, method and premise tuple. Do not
reset a radius to a smaller value because a center appears accurate. The whole
record's byte receipt becomes the next chain link.

Retain in RAM only the current step, its projected/encoded bytes, the preceding
endpoint and receipt, and fixed metadata. Release the previous certificate
after its exclusive synchronized write. Advance the runner state only after a
successful write; a serialization/quota/write failure terminates the attempt.
No list of all steps, repeated whole-trajectory JSON serialization or mutable
growing index is needed. A terminal result can name the completed count and last
receipt; the chain and numbered files retain the rest.

On forced termination there may be one calculated but unretained step, or one
partially written step. Do not confuse “evaluated”, “returned a certificate”
and “persisted”. If exact attempted-count metadata is unavailable after death,
the persisted prefix and at-most-one uncommitted next step are an honest bound
under this sequential writer. Missing terminal metadata is not proof that no
numerical work occurred.

A prefix audit reads one file at a time, verifies schema and exact chronology,
constant/source receipts, previous hashes and endpoint linkage, and stops at a
missing, partial or invalid next record. It preserves later/partial files and
reports them; it neither deletes them nor calls the numerical producer. A fully
parsed file with a valid chain can establish a retained complete prefix, but
content association alone is not independent verification of every Taylor proof.

## Exact quota and write-once semantics

The selected quotas are **256 MiB = 268435456 bytes** for step evidence and
**64 MiB = 67108864 bytes** for auxiliary material, **320 MiB total**. The
256-MiB budget is an average of 65536 bytes per step over 4096 steps, not a new
per-step promise or limit. No representative record was evaluated to claim a
size here. Compact JSON and sharing the matrix/projections reduce repetition;
the actual serialized byte count decides admission.

Charge the shared certificate constants and all step files, including partial
step writes, to the step ledger. Charge source/protocol/target evidence, attempt
and terminal metadata, failure records and stdout/stderr to auxiliary storage.
Fix those categories before launch. Count every run-associated file created by
the runner/supervisor, not only the final files later selected for a manifest.
Do not copy the bibliography or the old capture trajectory into the run output.
Existing source assets can be associated by bounded receipts as allowed by the
frozen protocol; newly written source archives still count toward the quota.

`write_json_once` already has the desired exclusive/non-atomic/fsync behavior,
but it serializes the whole payload and has no byte-limit argument. For a strict
quota, a private runner helper can reuse the same owners without inventing a
new numeric format:

1. Project one record, `encode_exact_tree`, then `json_dumps` with fixed compact
   formatting, `allow_nan=False` and explicit LF. Produce its UTF-8 bytes once.
2. Check/reserve the exact length including the final newline against its
   category and the total budget **before opening the file**.
3. Write those same bytes with shared `safe_write`, `mode='xb'`, `atomic=False`,
   `sync=True`, and a resolved run-root `base_dir`. Require a complete write.
4. Establish the receipt from the written bytes; only then acknowledge a
   persisted step. Preserve a failed or partially written exclusive path.

This uses the existing serializer and immutable-write semantics while avoiding
a second serializer pass that might be formatted differently. Serializing one
certificate is acceptable; materializing the entire trajectory is not. A
serialization memory failure remains subject to the enforced process limit and
must preserve the preceding prefix.

Do not use default atomic replacement: it could overwrite an earlier attempt
and removes temporary files rather than preserving the partial path. Do not
catch a failed write and retry the same logical record under a fresh filename.
On an ordinary exception record the reason within the already reserved terminal
budget; on forced termination the surviving attempt/prefix/partial file is the
fallback evidence. Retained partial bytes count even if they are not valid JSON.

There must be one authoritative quota reservation path. If worker and parent
both write auxiliary files, independent copies of a 64-MiB counter can overshoot
the limit. Use either a single writer/coordinator or fixed disjoint subbudgets
whose sum, including terminal reserve, is at most 64 MiB. Parent log capture
must itself be bounded: redirecting child stdout to an unlimited ordinary file
or polling directory size after writes is not quota enforcement. Bound any
interprocess queue as well so streaming cannot accumulate unbounded records.

## Supervisor boundary

The supervisor must establish the promised one-worker, wall and memory controls
before the worker can admit a source or execute a numerical step. Refuse launch
when enforcement is unavailable. An independent wall timer must remain active
during long field calculations, serialization and filesystem writes; checks
only between steps are insufficient. The output writer enforces exact quotas
before bytes are emitted, in addition to the supervisor's process controls.

Reserve auxiliary bytes and, if a final receipt is desired, time **inside** the
3600-second policy for bounded shutdown/reporting. Do not treat post-timeout
source copying, a full-prefix audit or an oversized terminal dump as unlimited
free work. Killing a worker may leave a partial record, and no terminal result
may be possible at the hard boundary; this is an admitted failure outcome.
The implementation review must distinguish enforceable guarantees from
best-effort metadata obtained after a forced process termination.

## Concrete non-transit controls to implement later

These are recommendations for the root's implementation review, not tests run
in this task. Use manually constructed certificate objects, exact scalar
fixtures, temporary directories and inert/failing workers. No scientific field,
Taylor kernel, trigonometry or target is needed.

- A full 26-field manual fixture, including asymmetric/nonzero-width intervals,
  round-trips through shared constants and a step record without losing any
  field. Compare the decoded full tree with `asdict` of the fixture. Separately
  reject a missing field, an unexpected non-null certificate, a changed shared
  metric/projection and a float supplied as a mathematical scalar.
- Two linked fixtures verify exact center/radius/time continuity. A missing
  middle file, altered previous hash, smaller next radius, changed duration,
  duplicate index or mismatched constants must fail before a further step.
- Exact byte-limit and one-byte-over-limit cases use actual encoded LF bytes.
  An oversized next record must leave the preceding complete file untouched
  and must not create an apparently successful next record. Include shared
  constants, failure output and logs in the appropriate budget tests.
- Inject a write/fsync failure after a prefix: the partial file survives;
  retry/overwrite is refused; the successful prefix remains auditable and no
  subsequent numerical call is made. Do not mock away the filesystem behavior
  the test claims to verify.
- An existing attempt, even one with a partial marker or no terminal result,
  rejects launch/resume. Reject path traversal or a resolved output path outside
  the fresh run root. A read-only audit must not invoke a worker.
- An inert blocking child tests hard timeout/kill and prefix retention; a
  bounded artificial output source tests the log quota; deterministic stub
  records test backpressure without storing 4096 certificates. Memory-limit
  enforcement requires a separate controlled child/admission test appropriate
  to the selected Windows supervisor, not a scientific workload.

Successful storage controls establish retention, exact representation and
bounded-failure behavior. They do not establish that the future reference
stays acute or reaches its target. A successful scientific terminal result
still needs the whole-path, endpoint and complete-state handoff conditions
from the selected method and its independently reviewed numeric adapter.
