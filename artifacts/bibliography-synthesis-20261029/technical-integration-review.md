# Technical integration review

Reviewer: `/root/joint_review_controls`  
Author of reviewed helpers: `/root`  
Date: 2026-10-29  
Assessment: **PASS for the two corrected preservation findings**, by static
source inspection. This is not an execution or substantive source review.

## Reviewed helper bytes

SHA256 values identify the versions inspected:

| File | SHA256 |
| --- | --- |
| `apply_integration.py` | `520cbb3f35c3fa732b487e40deee84f62f4b336f93f4d425c219eb2043b1877a` |
| `finalize.py` | `260613a4af2c164421fb352299a507588f8dde91bd0e073ce43d8d6f09d70a66` |
| `validate.py` | `79b6b8e464b06c74fd76d8621d7312647d4fdc45eafeefc34bbc7e298c7c4bfd` |
| `check_rendered.py` | `63cbb66d6b543ca58dbdf3073ff741d2d48fecb28314e55ff05234727242049b` |

## Resolved findings

1. **Program receipt before navigation generation.** Application now saves
   `applied-program-before-generation.md` and binds its exact receipt. Closure
   checks the historical PROGRAM receipt against that snapshot, verifies the
   exact source-context replacement against the archived original, compares
   the current authored front with `_authored_front(intermediate)`, and requires
   final PROGRAM bytes to equal `render_program(read_book(ROOT))`. The inspected
   shared renderer updates the navigation while preserving the authored front;
   legitimate generation therefore no longer conflicts with the pre-generation
   receipt, and other editorial changes remain excluded.
2. **Generated-view boundaries.** Closure now requires unique catalog/navigation
   delimiters in both original and current `theory/README.md` and `mkdocs.yml`,
   and exact bytes before the start delimiter and from the end delimiter onward.
   These are the actual markers used by the shared generator. The documentation
   checker supplies the separate generated-region-content check. Arbitrary edits
   elsewhere in these two allowed files are no longer accepted by the scope test.

## Remaining scope and limits

The inspected closure retains the exact inventory assertion: 4,634 prior paths
plus one new chapter, exactly five changed existing documentation owners, all
seven PDF receipts in place, and all 51 previous records verified. It checks
the existing source hashes outside that allowlist, preserves the two agent
instruction mirrors, and archives only the six changed/new documentation files.
Ordinary/physical guard expectations remain 0/1 with absent final readiness;
the five documentation checks and their logs are required before closure.

I did not execute application, generation, validation or finalization, and did
not modify maintained sources. Passing runtime checks remains an obligation
of the application/closure run. Independent content reviews, read by root,
provide substantive approval; file existence and receipt association do not
prove such approval. This review adds no scientific evaluation, replay,
theorem, acquisition result, physical correspondence or readiness permission.
