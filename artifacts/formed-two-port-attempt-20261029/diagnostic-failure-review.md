# Review of the failed retained-box diagnostic

2026-10-29. Author/reviewer: `/root/admission_foundation`.
I authored the diagnostic script and the defect described here. This is a
static failure analysis, not an independent approval of my own diagnostic.

**Finding: the single admitted post-hoc diagnostic failed with NameError and
produced no retained edge or box-margin results.** Its source remains frozen
and unchanged. The failure does not change the four certified reference steps,
establish source entry, prove non-entry or identify an offending physical edge.
No correction, rerun or new calculation is performed or proposed as part of
this review.

## Evidence and exact defect

| Retained file, relative to this stage | Bytes | SHA-256 |
| --- | ---: | --- |
| `diagnose_stored_tubes.py` | 10166 | `f1834bc8cdce75f8bf9d261f50a06b6a1292ad8359280c68c1b94059c2b27b51` |
| `diagnostic-freeze.json` | 891 | `a67aeca6f7d8d343819bce976fcead28e00d3b4c8baf387e331be22f5c143587` |
| `diagnostic-execution.json` | 577 | `8f981aa11d420eae8208b15495851a0873fbf97f1bc482fe9751aadc08fa780b` |
| `diagnostic.log` | 780 | `c2bcb6173809302c62d10b9bd89175064306c9d88266e9b464acd759e9f30b7f` |
| `audit.json` | 2706 | `73457c3783ec9fcc8620d8b2eb23cc7d8c3a5e09b932cfc16e76db8a1eeb9010` |
| `attempt/steps/0003.json` | 34413 | `b9fc7c6946db8dfe7c42b16dbdd17e5a1f26560cb5e8154bbcae406377d6ce60` |

In `analyze`, the loop unpacks each edge into the local names `left` and
`right`. It does not bind a name `edge`. The dictionary passed to `rows.append`
at line 80 nevertheless evaluates both `'edge': edge` and `edge_name(edge)`.
The first of those expressions raises NameError before a row can be appended.
The traceback records `main` line 153 calling `analyze` and the failure at
line 80. This is a programming defect, not a mathematical or domain rejection.

Some intermediate rational expressions occur before that line, but their
values were neither emitted nor retained as diagnostic results. They supply
no reportable evidence. The planned sixty-edge observation set, critical-edge
identification, corner witness, radius/width comparison and exact reconstruction
check were not completed. `stored-tube-diagnostic.json` is absent. The execution
record reports exit 1, no timeout and approximately 1.54 seconds, with zero
scientific producer calls and zero new reference steps. The script digest
still equals its frozen digest. I have not used the traceback to reconstruct
or recompute any missing value.

## What remains established

The separately retained documentary audit verifies four contiguous reference
steps through sigma=1 and their source associations. The actual run stopped at
step index 4 because its candidate failed the sufficient whole-tube domain
condition. This remains the run's outcome; the diagnostic failure changes
neither those certificates nor the preserved failed candidate.

The source owner `src/tnfr/mathematics/_validated_taylor.py::picard_tube`
checks the candidate's domain before evaluating its field and before testing
strict Picard inclusion. Consequently the rejected candidate is not itself a
certified whole-time solution enclosure. Without the diagnostic results we
cannot name a limiting edge, report a newly calculated gap, distinguish a
rho-buffer violation from raw nonacuteness in that candidate, or claim that
inflation caused the failure. Even a demonstrated inadmissible corner would
not establish that the exact orbit reaches that corner.

The failure of this auxiliary script is separate from the scientific attempt's
domain failure. Neither supports a non-entry theorem, an assertion of numerical
resource exhaustion, an automatic numerical-policy change or physical mapping.

## One proposed next analytical obligation

The appropriate first step is a **source-conditioned forward-invariance test
for the existing buffered acute domain**, before considering another numerical
attempt. This is a proposed bounded analytical question, not a result obtained
in this review.

Use the unchanged M2 reference law, original twenty-edge incidence map, lifts,
phi and conserved means. Formulate the candidate domain by the existing
constraints `g=g0+B*xi`, `|g_e|<=pi/2-rho` and the existing source energy
sublevel `U(g)<=U_floor+E0`. First verify that the admitted nominal reference
belongs to this candidate using the source proof; do not silently equate an
actual full-state energy with a reference energy. Then derive the inward-facet
obligations from M2 for every boundary `g_e=+(pi/2-rho)` and
`g_e=-(pi/2-rho)`, retaining the cycle/compatibility and energy constraints.
No edge is singled out using an unavailable diagnostic result.

The useful outcome is either an analytical invariant-domain certificate with
its actual margin implications, or a precise outward-facet obstruction within
this candidate domain. The latter would refute this coarse invariant-domain
argument, not prove that the particular admitted source trajectory leaves it.
A positive result would address the missing global domain premise; it would
not by itself prove target arrival or certify a particular Picard box.

This question keeps the dependencies of
`theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-entry-method`
explicit: M2's zero metric growth uses acute convex tubes; M4's Jensen phase
floor is stated in the acute `(2,1,0)` cell; M6--M7 initially hold only before
chart exit and close through the reference buffer. A less restrictive transient
domain or a nonzero growth certificate cannot simply replace the twenty checks
without also justifying the potential floor, comparison and complete-state
transfer. The eighteen-phase reference remains auxiliary; no uncertainty ball
is projected away and neither of the thirty-six actual rows is discarded.

The proposed domain test should receive its own precise scope and review before
any evaluation. This review evaluates no facet, new theorem, trajectory,
field, target sign or stored-box margin. All failed diagnostic and original
attempt artifacts are preserved as historical evidence.
