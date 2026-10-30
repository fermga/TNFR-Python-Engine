# Independent review: formed-source integration and finite algebra controls

Author of reviewed integration: `/root`.
Reviewer: `/root/admission_guard`. Date: 2026-10-29.
Verdict: **PASS within the source/event and finite-control scope below.**
This admits one execution through the reviewed supervisor after the source
freeze and the other substantive reviews. It does not report executed checks,
admit a trajectory, or independently replace the identity/response reviews.

I previously contributed `source-event-analysis.md`. That contribution is
declared: this review checks the root's integration, its new explicit work
allowance and proof, and the actual control/supervisor source. It does not
represent another independent review of my original note.

## Bound final files

Paths are relative to `artifacts/formed-two-port-admission-20261029/`.
The final chapter is **v2**; the earlier chapter files remain preserved.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| chapter-append-v2.md | 12427 | 77863822db23cdb0d8a8e9e67bbdd1b4d677b5a543c4664498c68d0888a53975 |
| chapter-proposed-v2.md | 42842 | b6a17ba53681a17111c2cb960ac01224596907ffa77a733229c115c322cc4607 |
| program-proposed.md | 89809 | ee00bcc4147cfb7ae52f080ba495542924e74dc8347431031357a256bf7bc960 |
| scope-and-protocol.md | 3023 | a92ba093baa6fd52250ce1a7b0ef16bf74b42a6180456664a3320780bfc7fd7d |
| check_admission.py | 5193 | 1022c3c0014bc152b7238a31309a6c761d33ad7e98c0d7a97403c3a2154344e1 |
| run_exact_controls.py | 1718 | be6855fb71618fcfba74e3cf3ce18c9b5a4b15b8a9d288244c13daadd32c10e3 |

The proposed chapter preserves every byte of the current compatibility
owner, then appends exactly `chapter-append-v2.md`. The v2 change explicitly
spells the positive integer lift corrections and disallows repeated wrapping;
both clarifications are correct. The program changes the sole next obligation
to source-aware entry-method/protocol admission, conditional on closure of
this reviewed admission stage. It neither claims entry nor authorizes its
evaluation. Source/event origins and the physical-readiness boundary remain.

## Substantive findings

1. **Actual source association is retained.** All initial rows, residual
   bounds, component zero sums, isolated flow images and continuous lifts are
   specified. The actual preevent images retain nonzero unknown form/phase
   residuals and correlations. The historical one-port report's joined support,
   means and response are not substituted for these images. The source is
   fully specified as a family; a selected numerical endpoint is not required
   for this conditional analytical declaration.

2. **The simultaneous event uses the correct graph and complete rows.**
   The contact set is exactly `(4,13),(5,14)`; four degrees become three and
   the mass is forty. Both form and phase rows use the same new degree
   normalization. There is no intermediate historical single-port flow,
   nodal jump, capacity change, source reset or inferred occurrence law.
   The script's Laplacian rank-one edge updates check both form-storage
   quadratics, and its one-edge alternative is an actual failing control.

3. **Lifts and joined means are transported correctly.** The cyclic mapping
   sends old ports `0,1` to original `4,5`. The donor/receiver positive
   corrections on the four recut nodes are respectively `4*pi` and `2*pi`;
   the resulting weighted mean shift is `6*pi/5`. Equation (A3) retains the
   nominal `3*alpha/40+phi/2` and the actual residual contribution. The finite
   residual-basis checks span each component's zero-sum constraint and check
   the port contribution to the new weighted sum. The script does not set
   those contributions to zero. A common gauge cannot turn `phi` into the
   midpoint source's `alpha/2`.

4. **The new work allowance is independently declared and sufficient under
   the proved inequalities.** The second actual gap is `phi-alpha` plus its
   retained error. From `25/8<pi<22/7` the draft obtains
   `0<g4<1/500` and `2/3<abs(g5)<7/10`. For the lower bound,
   `g5^2/2>2/9` and `g5^4/24<2401/240000`; the Taylor lower bound for
   `1-cos(g5)` and the remaining nonnegative work imply the strict lower
   endpoint in (A5). For the upper bound, the two phase contributions are
   below `49/200+1/500000`. The two-component Euclidean form bound gives
   combined form work at most `2*epsilon^2`, not a zero form residual.
   The displayed rational design inequalities therefore establish
   `1/5<W2<1/4` for the whole actual family. This is an analytical sufficient
   envelope, not a numerical evaluation of a selected event. The allowance
   `1/4` is supplied policy; dissipation does not generate its funding.
   The old `10^-6` is explicitly excluded. No additional degree-matrix
   energy term is missing from the edge-based storage jump.

5. **The finite controls encode the stated identities.** They use exact
   integers and `Fraction` operations for graph transport, edge Laplacians,
   live degrees, lifts, nominal and residual means, contact gaps, work bounds,
   the retained rational root-bracket comparison and the strict absence
   arithmetic. They do not solve for the implicit target. The initial acute
   margin check is conservative and retains the inherited continuous edge
   offsets. Mathematical facts about sine antisymmetry and cosine bounds
   remain explicit proof premises; these scalar checks are not substitutes
   for their proofs or the source-history admission.

6. **The later scalar checks match their existing conditional theorem.**
   I compared the source with `SINE_TWO_PORT_DIPOLE.md`: equations (7)--(10)
   supply the degree-metric warmup constants and norm bounds; (18), (20) and
   (22) supply the response-error arithmetic; (24)--(25) supply the separate
   phase-event work and post-probe trapping inequalities. The transported
   dipole marks, three affected bulk edges, degree norm, dual norm and
   `q^T L q=6` agree on both supports. The script's rational formulas match
   those expressions. This verifies their transcription and bounded scope,
   not actual-source entry, a newly evaluated response coefficient, or the
   full independent proof review assigned to the other reviewers. The old
   phase-blind experiment's different initial form budget is excluded.

7. **There is no hidden scientific execution in the control script.**
   Its only TNFR calls are the shared artifact receipt and immutable JSON
   writer. There is no scientific assessor, integration kernel, root solve,
   adaptive loop, parameter scan, simulation or historical producer call.
   All loops are finite over the declared graph/basis. Its computed warmup
   and response expressions are the expressly admitted arithmetic checks of
   existing bounds. The prospective protocol calls this new verification,
   not a new trajectory or response experiment.

## Execution concern resolved by the supervisor

`check()` uses Python assertions. I identified that running under `-O`
would remove them. The reviewed supervisor resolves this without modifying
the checker: it launches `[sys.executable, '-E', script]`, with no `-O`.
The child ignores `PYTHONOPTIMIZE`; parent interpreter optimization flags
are not inherited as command arguments. Thus the reviewed child retains
the actual assertions.

Before launching, the supervisor compares every frozen file receipt to its
current bytes. Its exclusive `exact-controls.log` creation occurs before
the subprocess, so the same attempt cannot silently be rerun after a child
failure or timeout. `subprocess.run(timeout=60)` enforces the stated wall
limit and stops the child on timeout. The wrapper preserves the combined
output and an immutable `exact-execution.json` on success, ordinary child
failure or timeout, then raises for failure. The checker itself writes its
immutable algebra result only after all controls pass. A failed result is
not converted to a source non-entry claim.

The source-freeze script inspected alongside this review includes the final
v2 drafts, program, protocol, checker, supervisor and review files. The root
must actually complete that freeze and the other independent reviews before
the single supervised run. This review does not state that any of those
future operations already passed. A changed bound file requires review of
its changed scope before execution.

No blocking source/event or finite-control correction remains. I did not
execute the checker, supervisor, source-freeze script or any scientific
producer; no maintained file was edited by this review.
