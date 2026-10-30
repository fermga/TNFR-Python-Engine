# Post-run result and interpretation review

Reviewer: `/root/joint_review_controls`; 2026-10-29, after the single reserved
attempt. **PASS for the bounded interpretation below; capture is unavailable.**
This new note is outside the prospective producing freeze. I authored the
documentary auditor, so I do not represent its output as an independently
implemented mathematical replay. Its source was independently reviewed before
use. Here I read the retained caller, worker and audit records, inspected the
shared Picard control flow, and checked byte receipts and exact JSON integer-tag
associations. I did not rerun the auditor, a field, target signs, Taylor steps,
historical producers or any scientific control.

The retained audit verifies source associations and four contiguous complete
certificates, indices 0--3. The last starts at sigma=3/4 with duration 1/4, so the
certified prefix reaches sigma=1. Failure index 4 was the proposed next interval
[1,5/4]. Its initial center, radius and source equal the final certified
endpoint/source exactly, without float conversion. Its preceding receipt is
the hash of `steps/0003.json`. There is no step 0004 or unverified later step.

The explicit reason is `formed_two_port_whole_tube_margin_unavailable`.
The external wrapper exited 2, without timeout, stderr or capture error. The
supervisor reports `worker_failed`, no watchdog firing, complete cleanup and
no exhausted total budget; the caller retained that failure. The reported peak
commitment of 859734016 bytes is below the 1073741824-byte cap. Thus the recorded
stop is a domain-admission failure, not a reported resource-limit stop. The
documentary audit likewise completed and declined a full reference certificate.

The distinction is essential: `_validated_taylor.py:147` checks the domain
before the Picard inclusion at lines157--160. Consequently `failed_tube` is an
**unvalidated candidate box**, not a certified time enclosure for the failed
interval. Failure of the strict slack above rho does not prove that the exact
reference leaves the acute chart. Margin<=rho would also differ from loss of
acuteness at margin<=0. Neither failure proves non-entry of the complete actual
source, failure of the TNFR mechanism, or a new physical statement.

The next bounded question is whether the retained candidate's affine edge
ranges isolate a box-admission obstruction and how that relates to the already
certified state at sigma=1. A separately declared post-hoc rational diagnostic
may distinguish an unresolved bound from a provably inadmissible corner of the
candidate box. Even such a corner is not an observed trajectory point. The
actual continuation remains unresolved. No smaller step, changed horizon,
precision, margin or resource budget is selected here, and no retry is admitted.

## Reviewed retained associations

All paths are relative to this artifact directory; hashes bind content only.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `attempt/failed-step.json` | 7403 | `d25f655ae4831571e4653f6dc3edb9bcda1cd29e0fd9b141aa44001be6b67df6` |
| `attempt/steps/0003.json` | 34413 | `b9fc7c6946db8dfe7c42b16dbdd17e5a1f26560cb5e8154bbcae406377d6ce60` |
| `attempt/worker-result.json` | 1544 | `2ed9015816d2576df451c0078dd622d1f1e5705aa2b7d1e57bc456c2012ce44e` |
| `attempt/execution.json` | 994 | `4911a418f4d612387bba33837dbe20e3462f79ff3197120bb009a35743bb45eb` |
| `attempt/attempt-verdict.json` | 2611 | `fd5c8c0725916a6b48d1bdcdd179911b11103de2d4109c33ba5b3840974d429d` |
| `caller-result.json` | 2826 | `5c80e3c530974dd2d9ec3da586014ce56e5590ff26f1105e0c4d7f5fa9601fa4` |
| `audit.json` | 2706 | `73457c3783ec9fcc8620d8b2eb23cc7d8c3a5e09b932cfc16e76db8a1eeb9010` |
| `audit-execution.json` | 1786 | `9203340ec7df9e95e30bcede277699fd8fa89826ee904e8a255b192c998adba5` |
