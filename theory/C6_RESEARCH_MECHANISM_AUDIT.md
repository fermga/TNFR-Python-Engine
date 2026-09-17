# Repository mechanisms for C6: B71 audit through B75 witness refinement

The repository contains useful mechanisms for the present proof effort, but
their hypotheses are more important than their names. This audit found two
reproduced verification-boundary defects, a documentation overclaim, a stale
root assumption, and concrete opportunities to share arithmetic. It found no
new failure of the canonical nodal/pressure binding used by the normal B70
validation. **The result remains 41/56 first-exit labels excluded, 15 open.**

## Scope and evidence

The B71 static inventory covered **308 Python files and 175,482 source lines** in
physics, mathematics, research, dynamics and operators. It records module
summaries, declarations and 19 groups with identical multi-statement AST
bodies. That is an inventory, not a claim to have semantically verified every
line. Deeper tracing focused on the nodal integrator, pressure realization,
mean/shape and spatial-envelope proofs, phase and symmetry reductions,
history/return geometry, exact matrix/extremum owners, runtime provenance,
and the B63–B70 evidence chain.

Detailed receipts and narrower reviews are retained under `artifacts/research`:
`c6_b71_mechanism_inventory.json`, `c6_b71_algebraic_structure.json`,
`c6_b71_history_structure.json`, `c6_b71_verification_controls.json`,
`c6_b71_nodal_scope_audit.md` and `c6_b71_verification_audit.md`.
The checkpoint `b71_final_validation.json` binds the original audit to its
unchanged B63 scientific source. B72 subsequently repairs four source files
and adds the shared proof owner; its new source identity is recorded separately
in `b72_final_validation.json`. Historical reports retain their original bytes.

## B71 findings and their status after B72

| Finding | Evidence and scope | Action |
| --- | --- | --- |
| Assertion-dependent artifact checks can accept a changed receipt under Python `-O`. | At the archived B71 source, the same altered minimum is rejected normally and admitted by the assertion-based receipt checker with optimization enabled. The reusable B70 lower-bound kernel rejects it in both modes. This does not invalidate the recorded normal B70 run. | Implemented in B72: shared admission and certificate checks use explicit exceptions; assertion-dependent exact-owner execution and evidence entry points refuse optimization. Tests exercise `-O` and `-OO`. Archived assertion-based scripts retain their historical execution boundary. |
| Shared inverse/PSD helpers did not enforce their advertised exact-matrix domain. | At the B71 source, inverse admitted floating, empty and nonsquare input. PSD accepted `[[1,10],[0,1]]`, although its quadratic form at `(1,-1)` is `-8`. Audited canonical callers constructed valid inputs. | Repaired in B72 using the existing strict nonempty square Fraction validator, exact symmetry for PSD, and an exact boolean `strict` flag. Invalid types retain `TypeError`; invalid shapes, asymmetry and flags raise `ValueError`. The targeted matrix/caller suites passed 330 tests. |
| Silence was described as guaranteeing future freezing. | SHA preserves EPI at its event and attenuates capacity. The represented default changes capacity `1` to `0.9204225284540524`; a remaining nonzero `nu_f*DeltaNFR` can still change EPI later. | AGENTS and its mirror were corrected in B71. B72 aligns contract metadata and module/glyph docstrings, with 42 existing tests passing. Runtime arithmetic is unchanged. |
| A historical proposal assumes zero mean at history roots. | The original B47 sentinel has zero relative coordinate, but the actual first-return root260 has `sum(k)=-4`, hence mean displacement `-1/(3*2^112)`. B69/B70 already use the correct nonzero root. | The zero-history-root sentence in `c6_b62_nodal_synergies.md` is superseded for the current problem. Preserve its historical bytes; do not carry that premise into a new mean certificate. |

The matrix issue is a reusable-helper input-contract defect, not a demonstrated
false certificate at its present canonical callers. The interpreter issue is
an execution-mode defect, not evidence that normal arithmetic replay failed.
Those distinctions prevent both silent reuse and unwarranted rejection of
valid prior work.

## Mechanisms worth reusing, and their conditions

| Existing owner | What it supplies | Use in the present investigation |
| --- | --- | --- |
| [Shared nodal kernel](../src/tnfr/dynamics/_euler_kernel.py), [remainder balances](../src/tnfr/physics/nodal_remainder.py) | Exact accumulated represented nodal area, nearest-even cells and compatible finite itineraries. | Keep one evolution law and one pressure/carry split. A proof coordinate must never reset carry or mean. Exact cell horizons can accelerate a genuinely held visible state; their pressure-constancy premise must be checked. |
| [Forcing realization](../src/tnfr/physics/forcing_realization.py), [forced support](../src/tnfr/physics/forced_support.py), [C6 profile](../src/tnfr/physics/c6_carried_profile.py) | Decomposition into EPI diffusion and explicit non-EPI source; exact relative Poisson profile and separate mean drift. | Centralize the distinction between shape relaxation and accumulated mean change. The current finite pressure assembly and phase premise must remain attached. |
| [C6 spatial closure](../src/tnfr/physics/c6_carried_closure.py), [carried tube](../src/tnfr/physics/c6_carried_tube.py) | A conditional centered-energy envelope derived from the represented pressure and rounding errors. | Reuse spatial bounds in new proposal filters. Any additional quadratic restriction needs a relevance test against current guards and a matching exact oracle; a DBM-only extremum must not silently claim the intersection. The envelope alone does not confine the mean. |
| [Exact extrema](../src/tnfr/physics/c6_carried_excursion.py), [shared history-affine owner](../src/tnfr/research/history_affine.py) | A matching exact linear minimum, attaining point and transport dual. | The B72 owner reuses the established extremum arithmetic and replaces the active detached adapter. A violating point may be hypothetical. The at-most-12-edge fixed-sign transport graph is not an unknown-sign global LP. |
| [Return/history owner](../src/tnfr/physics/c6_carried_return.py), B70 guard pool and facet certificate | Full guarded transitions, source/target/sentinel coverage, exact DBM equality and sparse equivalent descriptions. | Preserve history identities while sharing equal geometry. Keep direct transport on complete closed DBMs separate from reduced-facet path representations. |
| B71 history graph audit | Exact reachability sets, SCCs, and a constructive extension for omitted terminal histories. | Optional discovery preprocessing saves 14 coefficients and 10 drift obligations. Final proof completion still satisfies the original full specification. The saving is modest; it does not justify a new long refinement campaign. |
| [Cycle algebra](../src/tnfr/physics/_cycle_algebra.py), [exact matrix owner](../src/tnfr/physics/_exact_linear_algebra.py) | Centering, Laplacian actions, exact products and rational linear algebra. | Reuse these operations with the repaired input checks. Do not independently recreate a floating mean/energy law in each experiment. |
| [Symmetry sectors](../src/tnfr/physics/symmetry_sectors.py), [pointed symmetry](../src/tnfr/physics/pointed_symmetry.py) | Graph automorphisms and state/selector stabilizers. | Test the entire prepared state and observations before quotienting. C6 topology has 12 automorphisms, but the exact current origin has only the identity stabilizer. |
| [Structural morphisms](../src/tnfr/physics/structural_morphism.py), [phase quotient](../src/tnfr/physics/phase_quotient.py), [reduction certificates](../src/tnfr/physics/reduction_certificates.py) | Intertwining, observer defects and scoped nonlinear phase closure checks. | A smaller topology alone is insufficient. A reduction must preserve the current pressure, rounded cells, origin and target; existing real diffusion results do not automatically certify the carried full-channel map. |
| [Hybrid stability](../src/tnfr/physics/hybrid_operator_stability.py), [runtime flow stability](../src/tnfr/physics/runtime_flow_stability.py) | Quotient gains, mean/reset terms and finite causal execution links. | Reuse when promoting a proven numerical property back to a runtime family. Do not confuse a finite receipt or disagreement bound with future full-state stability. |
| [Claim ledger](../src/tnfr/research/claims.py), [core manifests](../src/tnfr/research/core_manifests.py), [evidence sidecars](../src/tnfr/research/evidence_sidecar.py) | Scope labels, source identity and evidence organization. | Record exact, measured, negative and open results separately. A source hash or algebraic success flag does not establish canonical admission or original-target coverage. |

Auxiliary Hamiltonian, metriplectic and GKSL modules are not a missing shortcut
to this C6 proof. Their model-specific assumptions and extra state variables
are documented; importing their conservation or contraction results without
a nodal/runtime bridge would change the problem. No such import is used here.

## Exact controls against attractive but invalid shortcuts

The 64 represented nodal increment vectors have **rank six**. Six rows
(masks `0,1,2,3,4,6`) and their exact two-sided inverse certify this. Thus no
nonzero common linear functional annihilates every increment in that table.
Their sum signs are 5 negative, 4 zero and 55 positive. These counts are not
visit frequencies, probabilities or a proof of net drift along the origin
trajectory. They also do not refute conservation on a smaller reachable class.

The history graph supplies a stronger, still scoped control. Choose signed
spanning-tree displacements `T_v`. A common-gradient conserved quantity
`c dot k+b_v` would require

```
c dot [s_e-(T_w-T_v)] = 0
```

on every arc. Six residuals, at original arc indices `3,10,11,13,16,17`,
have an exact inverse and span all six coordinates. Hence such an exactly
conserved form has `c=0` and constant offsets on the connected graph. This
does not exclude monotone barriers, independent history gradients, nonlinear
functions or a smaller reachable domain. Signed tree cycles are algebraic
constraints, not asserted realizable trajectories.

All 910 histories are reachable in the abstract graph from root260. Exactly
908 can reach target180 and constitute a single strongly connected component.
The remaining histories274/281 are terminal singleton components. Restricting
discovery to the corridor leaves 908 blocks and 7,140 arcs; it does not
decompose the difficult part into small independent subproblems. This confirms
the earlier B62 component analysis on the sealed current specification; the
large component is not a newly discovered decomposition.

A corridor candidate can nevertheless be completed soundly: give both omitted
sink histories zero gradient and a common constant `M` no smaller than the
exact maxima of every incoming corridor form. Their bounded guards make such
a finite rational `M` available. The original full checker must then accept
all 7,150 arcs and 27 targets. This is a constructive preprocessing option,
not a discovered positive barrier. Feasibility equivalence holds with
unrestricted rational coefficient sizes; completion need not preserve a
minimum norm or the 256-bit discovery cap. The graph audit checks 40 local
completion witnesses, a separate full-verifier fixture and six rejection
controls; these do not constitute a C6 corridor candidate.

## Centralization priorities

The static duplicate groups are candidates, not instructions for bulk
replacement: function signatures, imported bindings and certificate-domain
seals can differ despite equal bodies. Two concrete reusable seams are
`event_runtime._exact_binary64_vector` versus
`_exact_metric.finite_binary64_fraction_vector_or_none`, and
`binary64_nodal_flow._float` versus `_euler_kernel._finite_binary64`.
Readonly-array construction and finite matrix diagnostics also recur across
quotient modules. Review caller contracts before consolidating them.

B72 implements the highest-priority consolidation in
[history_affine.py](../src/tnfr/research/history_affine.py): one admitted
specification and one signed, translated expression shared by candidate point
rows, exact affine forms and final checking. The owner validates pooled DBMs
once while retaining every obligation identity. It bounds rational text before
parsing and checks exact numerator/denominator sizes afterward. Canonical bytes
are hashed and privately decoded before admission. Owner-controlled snapshots
detect subsequent mutations of exposed specifications, obligations and
coefficients, including Fraction internals. This is an API integrity boundary,
not a security boundary against modifying the module's private registry or code.

Preserve deliberate independence: combining the producer and its separate
Floyd/Bellman-Ford or primal/dual checkers into one unchecked arithmetic path
would weaken evidence. Historical prototypes with negative results retain
their role as reproducibility records; mark obsolete premises rather than
deleting the records or using them as active implementations.

## B72 validation and measured search outcome

The repaired source passed **451 tests**: 330 matrix/caller tests, 42 operator
tests and 79 shared-owner tests. An independent review additionally reproduced
11 mutation/admission controls. Full-case preflight independently reconstructs
all 7,177 drift/target forms and anchored rows, verifies the exact nonzero root,
matches 62 archived local extrema, and compares both full constant controls
with the historical checker. The root, pressure realization and target family
remain unchanged; the new source identity is not represented as the old one.

The recorded candidate LP has **19,918 rows, 6,371 columns and 125,768 nonzero
entries**. Its one invocation reached the 20-second limit after 20,692 solver
iterations, without returning a primal candidate. Total experiment time was
23.305 seconds. No rational candidate, exact scan or new counterexample cut was
produced. Consequently the conditional second proposal was not invoked. The
independent validator checks the complete point-row construction, source
bindings and resource ledger; it records zero exact receipts, not a successful
barrier or an impossibility proof. **Coverage remains 41/56, with 15 pending.**

## B73 feasibility-only result

B73 implements the proposed norm elimination with the shared owner. Its
7,178 rows, 6,370 free variables and 100,288 nonzero entries preserve every
anchored point condition. An independent preflight checks the full matrix,
root and seven malformed-proposal controls. All 27 target identities remain
present, although the anchored target evaluations have only 12 distinct exact
rows. Coincident sample rows do not establish equal guarded obligations.

The one zero-objective invocation reached its 20-second limit, taking
20.0345207 seconds and 33,200 solver iterations without a primal candidate.
No rational reconstruction, exact scan or new cut followed. The independent
post-run check verifies the ledger and unchanged B72 source without a solver
rerun. **Coverage remains 41/56, with 15 pending.** The previous 451 tests and
62 local-extremum comparisons remain unchanged-source evidence; they were
not rerun for this artifact-only experiment.

## Coordinate reuse and continuation boundary

The bounded diagnostic reuses two existing nodal/history descriptions:
each history's first retained evaluation and the B71 signed spanning-tree
displacements. With an integer center `t_v`, the proof identity
`beta_v(k)=a_v dot (k-t_v)/N+b'_v`, `N=2^59`, has exact inverse
`b_v=b'_v-a_v dot t_v/N`. It changes proof coordinates only. The root center
is the actual nonzero root; no mean or carry is reset. Drift retains
`s+t_source-t_target`, which need not vanish outside tree arcs.

Such a translation preserves unrestricted finite point feasibility, but
original coefficient size and denominator caps must be checked after the
inverse transformation. Every one of the 7,178 translated rows lifts exactly
to its original row. First-evaluation centers reduce nonzeros from 100,288
to 59,682 and leave 2,454 zero gradient columns in the current point proposal;
omitting these columns would leave 3,916 active variables. These gradients
can still matter on unsampled guard points. The signed-tree chart leaves
100,282 nonzeros and supplies almost no sparsity gain.

The first chart's worst within-column nonzero magnitude ratio worsens from
about `3.76e3` to `8.39e17`. Neither conditioning improvement nor solver speed
has been established. The B74 gate below checks solver-row materialization,
scaling, zero-column restoration and original-coordinate coefficient caps
before its bounded proposal. Low observed ranks
do not justify removing affine degrees from the full guarded problem.
No second optimizer call or automatic time-limit increase is part of B73.

A future candidate still requires all guarded exact minima and independent
dual checking, followed by the retained canonical full-target bridge before
any region promotion. The detailed result and diagnostic are retained in
`artifacts/research/c6_b73_result_and_next_gate.md` and sealed by
`artifacts/research/b73_final_validation.json`. Existence within this proof
family and indefinite boundedness remain open.

## B74: faithful matrix loading and a complete rejected candidate

B74 implements the first-evaluation proposal with 3,916 active variables.
The installed HiGHS loader discards sufficiently small matrix entries under
its default threshold. A positive exact row scaling puts every nonzero inside
declared computational guards: only 22 rows need the factor `2^39`, while
the other 7,156, including the root, stay unchanged. A no-optimization model
load preserves all 59,682 coefficient entries, indices, right-hand sides,
free variable bounds and the zero objective byte for byte. This does not
certify later internal numerical transformations or improved conditioning.

Reconstruction restores all 6,370 slots, inverts the chart rationally, then
limits denominators in ORIGINAL coordinates and checks the actual root and
original size caps. Nine focused controls include exact cancellation and a
tight target disrupted by denominator limiting. Independent preflight checks
every row and inverse identity and rejects nine altered proposals.

The one LP returns a candidate in 7.6290518 seconds. All 7,177 guarded exact
minima are then computed, and independent primal/dual validation confirms
5,949 failures: 5,922 drifts and all 27 targets. Total experiment time is
18.5012193 seconds. The original coefficients fit the stated caps; the exact
conditions reject the candidate. No further LP or new cut round ran.

Solver success is weaker even than exact sampled feasibility: the exact
binary64 lift has 1,688 negative anchored drifts, and the rational candidate
has 1,546. Both have nonnegative target anchors, yet all complete target
guards reject the candidate. Thus denser geometric sampling and numerical
feasibility control are distinct needs. A negative attaining point is a
guard witness, not a proven actual trajectory point. The count of 41 excluded
labels and 15 pending labels is unchanged.

The next bounded refinement should reuse verified attaining witnesses while
preserving every original obligation. Keep all initial rows, include all
27 target identities before selecting a bounded drift-witness subset, and
recompute active columns and scaling rather than assuming the old zero
columns remain zero. The complete original-coordinate checker remains the
acceptance boundary. The current candidate does not refute the affine family.
Details and receipts are owned by `artifacts/research/c6_b74_result_and_next_gate.md`
and `artifacts/research/b74_final_validation.json`.

## B75: witness reuse, active-column correction and solver-method boundary

B75 reuses exactly 256 verified B74 witnesses, preserving every original
sampled row and obligation. All 27 target identities remain even where point
coordinates coincide. The extra points reactivate 134 columns, confirming
that columns absent from the initial point matrix cannot be discarded from
the full guarded proof problem. The enlarged 7,434-by-4,050 matrix has 62,379
nonzeros. Independent full-row preflight and loaded-array checks pass;
positive scaling now applies to 81 rows. No new physical parameter is added.

The single LP times out after 20.023 solver seconds without a candidate.
No exact scan, new extremum or region exclusion follows. The finite witness
evaluations are unavailable. An intermediate nonfeasible iterate at timeout
does not certify infeasibility. Independent post-validation preserves this
boundary; source and regional coverage remain unchanged at 41/56, 15 pending.

An installed-source audit intercepts public `highs` and `highs-ipm` calls
before the native wrapper. All outgoing arrays match the frozen B75 model
byte for byte; only the solver selector changes. The next controlled gate
therefore uses one 20-second `highs-ipm` proposal on that same model, after
preflight and with unchanged full exact acceptance. No further cuts or
history are required for this comparison. Dispatch identity guarantees
neither speed nor feasibility; the original native algorithm was not traced.
Details and receipts are owned by `artifacts/research/c6_b75_result_and_next_gate.md`
and `artifacts/research/b75_final_validation.json`.
