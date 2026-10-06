# C6 history refinement and retained exclusions

Exact union, safe-memory and refinement evidence with its source-bound budgets.

Part of [Cycle winding under Coupling: scope and retained evidence](../../../COUPLING_WINDING_PERSISTENCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

**Archived research record.** This preserves conditional derivations and source-bound evidence. Its local next-step language is historical and does not schedule work.

## 50. B56: global counts lose chronology while word budgets recover it

The B47 carried origin, nodal pressure realization, fixed phase, unit
capacity and `h=1/16` remain unchanged. This block distinguishes two kinds
of path information: total displacement with transition counts, and the
joint source constraints of an ordered return word.

### An exact obstruction to the global count relaxation

For return edge `e:c->d`, let `a_e` be its integer carried displacement.
Any finite path from origin label `o` to label `t` has nonnegative integer
counts `n_e` satisfying

```text
sum_e n_e (1_source(e)=c - 1_target(e)=c) = 1_o=c - 1_t=c,
k_target = sum_e n_e a_e.
```

These are necessary conditions from graph incidence and the accumulated
nodal equation. They do not check the RN guards at the successive states.
For the complete 1,265-edge, 32-base-label relation, six signed balanced
integer count vectors `z_i` satisfy `sum_e z_i,e a_e = g_i e_i`, with
`g=(1,1,4,8,8,8)`. Every edge displacement lies in this product lattice,
so the vectors prove equality of the cycle-displacement lattice with it.
An independently checked integer circulation `c_e>0` on every edge has
zero incidence and zero nodal displacement.

Choose any graph path from `o` to a base label `t`, with displacement `p`.
For any `k` in the product lattice, add
`sum_i ((k_i-p_i)/g_i) z_i` to its path counts. This gives a signed integer
solution with displacement `k`. Adding a sufficiently large integer
multiple of `c` makes every count strictly positive. The graph is strongly
connected, and the resulting directed multigraph has the required Euler
imbalance, so an abstract label walk exists. Its carried coordinates need
not satisfy even one complete chronological RN itinerary.

Consequently this relaxation cannot exclude any of the seventeen remaining
slabs: explicit endpoint proposals in the complete original targets pass
both the exact coordinate-coset test and the count construction. Transient
terminal targets additionally retain a valid final source/target guard and
the base endpoint's envelope membership. The earlier chronology is still
unverified. This is a limitation of a proof representation, not a reached
exit or evidence of dynamical instability.

The existing owner provides `derive_c6_carried_return_count_relaxation`
and its `construct_counts` method. Canonical source reconstruction, strong
connectivity, all six generators and the positive circulation are checked
before the theorem is emitted. Zero coordinate gcds are outside its stated
domain. Integer candidates are mathematical witnesses, not new physical
coefficients. The method constructs counts without enumerating the
potentially enormous abstract walk.

### Joint word guards give exact repetition limits

Consider a nonempty closed sequence of return edges with displacements
`a_0,...,a_(m-1)`. Each edge includes its exact intermediate RN guard and
is clipped to the retained source and target envelope. Translate those
source guards by the preceding accumulated shifts and intersect them:

```text
B = intersection_j (G_j - sum_(r<j) a_r),    A = sum_j a_j.
```

No extra first-edge guard is imposed after the final edge of one word.
For `n>=1` consecutive repetitions, the exact common-grid source is
`intersection_(r=0..n-1) (B-r*A)`. If `B_ij` are its original closed
difference bounds and `A_6=0`, the repeated source is obtained by closing

```text
B_ij - (n-1) max(0,A_i-A_j).
```

This formula follows by minimizing the translated bound over the integer
repeat index. For nonzero `A`, the pairwise widths give the finite bound
`n <= 1 + min_(A_i!=A_j) floor((B_ij+B_ji)/abs(A_i-A_j))`.
Exact monotone search then finds the sharp last nonempty source and first
empty source. A closed integer DBM supplies an attaining grid point.
Empty `B` excludes one complete word; nonempty `B` with `A=0` is an
identity on that conditional word class. No case by itself asserts origin
reachability or complete-runtime stability. Interrupted composition/search
cannot emit a maximum.

`derive_c6_carried_return_word_budget` implements this derivation in the
same owner. Two examples use indices in the fingerprinted complete return
relation:

| Ordered edge indices | Base RN labels | Intermediate label | Sharp common-grid maximum |
|----------------------|----------------|--------------------|---------------------------|
| 884, 154 | 42 -> 26 -> 42 | 21 in edge 884 | 1 repetition |
| 182, 72, 116 | 32 -> 18 -> 24 -> 32 | None | 2 repetitions |

The first word can occur once in its relaxed source class. What is
impossible is repeating it twice; requiring its first guard immediately
after one traversal must not be mislabeled as failure of the first word.
For the second word, the node-3 shift per cycle is
`1,189,640,417,057,952`. Three repetitions require their starting points to
span `2,379,280,834,115,904` in that coordinate, while its source interval
has width `1,803,052,714,421,815`: an exact deficit of
`576,228,119,694,089`. A source for two repetitions is retained separately.
These sharpness witnesses are on the relaxed common grid, not asserted
reachable from B47 or on its finer coordinate cosets.

### Integration, tested boundaries and the remaining proof

The existing campaign adds `--method histories`, reconstructs the complete
B55 payload, and binds eleven inputs including the untrusted integer and
word proposals. It verifies all seventeen endpoint/count identities and
the two word budgets. No actual trajectory is extended. Whole first-exit
coverage remains **39/56**, with **17 unresolved**.

Retaining the first complete return-word label in backward hulls remains
inconclusive within its intersection guard. Separately, exact backward
frontiers at depths 3 or 6 were combined with distinct affine gradients
per active cell. Six proposed families are ruled out by independent
positive-dual checks of 284 exact witness inequalities. This refutes those
specific monotonicity/entry-separation criteria, not every barrier or the
stability of TNFR.

An additional finite prefix-memory product forbids the first word twice
and the second word three times wherever they occur. From the unchanged
origin it has 43 reachable vertices and 2,113 arcs; exactly two pattern-
completion arcs are removed. A direct substring oracle checks 9,619 finite
words. Exact balanced generators and a strictly positive zero-displacement
circulation show that the product still admits every original coordinate
coset through incidence/displacement counts. Thus these two exclusions
alone do not repair the count relaxation. This does not rule out combining
the product with joint carried-state guards.

The next proof representation must preserve more of those joint guards
across arbitrary interleavings, including transient terminal visits.
Refine obstructing cycles only after verifying their source constraints
and repetition limits; do not repeat the unrestricted count search or the
same two-pattern count product. A budget for one word does not bound
arbitrary interleavings. Report:
`artifacts/research/c6_winding_return_histories.json`; exact independent
replay and source reconstruction accompany the B56 validation artifacts.
The complementary product is retained in
`artifacts/research/c6_b56_forbidden_word_automaton_probe.json`.

The focused shared-return and campaign suite passes **197 tests in 133.01
seconds**, including 121 new cases. Independent arithmetic verifies all
eleven inputs, six count generators, 1,265 positive circulation counts,
seventeen endpoint witnesses and the two repetition maxima. Nine physical
steps from conditional sharpness witnesses are also checked; these are
not a continuation of the B47 origin. Source reconstruction and final
checks are retained in `artifacts/research/b56_scientific_source_delta.json`
and `artifacts/research/b56_final_validation.json`. These implementation
checks add no whole-region exclusion or physical confirmation.

## 51. B57: joint last-return memory excludes a further whole exit slab

The B47 origin, canonical pressure realization, carried residual, fixed
phase, unit capacity and `h=1/16` remain unchanged. The new partition records
the last complete return edge, including any transient middle cell. It
therefore retains a joint carried-state constraint that a bare RN label or
the two forbidden-word prefix states can lose.

### A descending cover with an explicit origin sentinel

For each complete return `e`, intersect its source guard with the retained
source, target and intermediate envelopes at their proper time coordinates.
Call the result `G_e` and the nodal displacement `a_e`. Initially its
endpoint cover is `Z_e = G_e+a_e`. A compatibility arc `i->j` exists only
when the RN labels compose and `Z_i` intersects `G_j`. The source of this
arc is the endpoint of return `i`; its next displacement is **`a_j`**.

Keep the original zero-coordinate state as a separate sentinel. Its first
images `R_j` are injected into every complete update:

```text
Z_j(new) = hull(R_j, union_(i->j) ((Z_i intersect G_j)+a_j)).
```

Every accepted update recomputes all predecessors. It remains a subset of
the previous cover and continues to contain every compatible origin
history confined to the declared domain. A worklist revisits dependents
when a cover shrinks. Interrupted initialization emits no usable geometry;
interrupted graph construction emits no partial graph; an interrupted
update retains its complete predecessor cover. A resource stop can still
leave a valid outer cover, but does not establish a fixed point or trapping.

The default production memory budget uses 499,995 work items: 4,676 for
guard and root initialization, 85,272 for pair construction and 410,047 for
descent. The complete graph has 9,590 compatibility arcs. Its 41,704 complete
visits make 21,999 strict updates and reduce 1,265 nonempty endpoint classes
to 856. The worklist is still pending; the boundedness claim remains open.

### Whole-target exclusion preserves terminal transient visits

Backward queries initialize the original complete unsafe slab in every
compatible memory class, including every transient first step without a
subsequent return. The sentinel is distinct from later visits to the same
RN cell. Each complete backward layer intersects translated target covers
with the current source guards, retaining separate last-return indices.
Empty or exactly stationary layers prove exclusion only when every prior
complete layer also excludes the sentinel. An abstract origin collision
does not prove an actual trajectory.

For **node 3 upper, mask 30**, the production query starts with four
nonempty terminal-history pieces. It reaches an empty complete layer at
depth **136**, using **76,178** query work items, with no origin membership
at any completed depth. This excludes the full original first-exit slab;
it does not merely eliminate selected points. An independent control using
a weaker 26-layer memory cover also empties, at depth 137 and 74,714
intersections. Its four terminal pairs are `(160,274)`, `(162,292)`,
`(372,274)` and `(374,292)`, indexed in the unchanged canonical return and
intermediate relations. This control does not require the additional
three previously proved node-0 cuts or the later worklist refinement.

### Reuse and boundaries of the method

The existing owner `src/tnfr/physics/c6_carried_return.py` now provides
`derive_c6_carried_return_memory_envelope` and
`derive_c6_carried_return_memory_region_exclusions`. Both rebuild the
canonical nodal source. The existing campaign adds `--method memory`,
reconstructs B56 and binds twelve input files. The query graph has its own
budget; each target then has a separate backward-work budget. This avoids
losing all later targets when an earlier query uses its allowance.

The weaker two-pattern prefix product was also checked with exact state
guards: its 42 nonempty history states and 1,313 guarded arcs reach a fixed
point after one image, with the same projected RN cover as the baseline.
Increasing that forward budget cannot improve this fixed point. Bounded
backward hull and exact-union controls on four representative targets add
no exclusion with that partition. The stronger last-return representation
supplies the new positive result; its resource-limited controls do not
establish a general impossibility for richer histories.

The main report is `artifacts/research/c6_winding_return_memory.json`.
The complete campaign excludes **40/56** original first-exit slabs. The
remaining sixteen are node 0 lower masks 8, 16, 20, 24 and 28; node 3 upper
masks 28, 60 and 62; and node 4 upper masks 56 through 63. Those queries
retain complete layers at their 250,000-work guards; none establishes an
actual exit or an impossibility of further refinement.

Independent arithmetic replays the twelve-input chain, all 1,265 source
guards, all 9,590 memory arcs, the complete accepted worklist updates and
all seventeen target initializations. It verifies every layer of the new
positive exclusion. The focused suite passes **275 tests in 166.60 seconds**,
including 78 new cases. Source reconstruction and final checks are retained
in `artifacts/research/b57_scientific_source_delta.json` and
`artifacts/research/b57_final_validation.json`.

The proposed safe-half-space refinement is resolved by the exact B58
comparison in section 52. Its operator is unchanged, so additional precision
requires preserving distinctions beyond that scalar cut.
Global boundedness, convergence, continuation of the actual post-SHA
runtime and physical correspondence remain separate open claims.


## 52. B58: exact redundancy and selective safe-memory partitions

### Propagated scalar exclusions leave the residual operator unchanged

Keep the B47 origin, original candidate cube, canonical nodal pressure,
full incoming carry, fixed phase, unit capacity and `h=1/16`. The safe
complements of the already excluded node-0 lower slabs (masks 4, 12 and 0)
and node-3 upper slab (mask 30) are necessary conditions before the first
exit from the original cube. They introduce no physical coefficient.

At mask 30, the original node-3 upper grid bound and the canonical
one-step nodal shift give exactly

```text
k3 <= 860302430744065456 - 4795745845901584
   = 855506684898163872,       grid quantum = 2^-113.
```

The next integer is unsafe; equality is safe. Every one of the 50 complete
returns through mask 30 already implies this inequality through its next
in-cube endpoint. It tightens only three of the seventeen terminal guards.
After propagating all four cuts, five complete source guards tighten, but
all 1,265 retained endpoint arrays and 9,590 compatibility pairs equal B57.
The complete clipped backward graph still has the **same 6,292 arcs with
identical integer guards and nodal shifts**. Every initial array for the
sixteen remaining original full targets is also exactly equal. Induction
therefore gives the same abstract backward operator at every depth.
This is an exact redundancy result, not a conclusion from equal counts or
from the four representative finite-query controls alone.

The shared return owner now centralizes query-graph construction and
whole-target initialization. Existing exclusion queries and the B58
comparison call those same helpers, with preserved complete-layer and
resource accounting. `--method memory-cuts` in the existing campaign
reconstructs B57, binds thirteen input files, derives the cuts from verified
positives and compares the complete geometry. Resource-limited comparisons
abstain. The report is `artifacts/research/c6_winding_memory_cuts.json`.
The original cube and targets are recorded separately from the cut domain;
a partially clipped target cannot inherit a whole-target claim.

### Preserve the exact excluded pieces instead of their hulls

The four terminal pairs from section 51 define exact unjoined bad pieces
`P_i` inside memory zones `Z_i`. Any origin path reaching such a piece would
follow its canonical guarded terminal step into the already excluded slab.
Removing these pieces is therefore justified before original-cube exit.
This reasoning does **not** justify removing a joined backward hull: such
an outer approximation can include points that never follow the bad path.

For an integer DBM inequality `k_a-k_b <= c`, its exact violation is
`k_b-k_a <= -c-1`. Splitting `Z_i` by the first violated facet of `P_i`
partitions `Z_i \ P_i` disjointly. Removing constraints redundant relative
to `Z_i` leaves respectively **5, 5, 6 and 6** pieces for memories
160, 162, 372 and 374. Their hulls are exactly the original `Z_i` in all
four cases. None of those exact bad pieces is a single relative half-space.
Thus a single DBM per memory erases this new information immediately.

A bounded prototype splits only those four memories, increasing 856 occupied
classes to 874. For each fixed safe seed `S_j`, its update is

```text
Z_j(new) = S_j intersect hull(R_j, union_i post_j(Z_i intersect G_j)).
```

The persistent seed intersection is essential: abstract predecessor images
can otherwise refill a certified hole. Roots remain covered, every guarded
pair is constructed before refinement, and an interrupted update retains
its last complete outer cover. The proof partition changes neither the
nodal dynamics nor the physical state.

Against an unsplit continuation from the identical B57 cover, complete
synchronous layers show equal projected hulls at depths 0 and 1. The split
cover is strictly tighter in **25, 38, 66 and 75** original memories at
depths 2, 3, 4 and 5, and every projection remains inside its same-depth
baseline. This isolates a real precision benefit from the effects of simply
continuing B57's unfinished worklist. The later bounded worklist tightens
459 projections relative to the sealed B57 cover; that larger count alone
cannot be attributed to the partition.

<a id="bounded-results-and-the-next-gate"></a>
### Bounded results and unresolved scope

The partition prototype constructs 6,924 guarded arcs and uses 499,996
forward work items. No original target disappears from its full direct and
terminal observations. Backward tests of node-3 upper masks 28 and 62 stop
at complete depths 195 and 85 with 250,000 work items each; both remain
unresolved. Separate unjoined-union tests on the B57 memory cover also stop
without a new exclusion: node-0 lower mask 8 at depth 14, node-3 upper mask
28 at depth 13 and mask 62 at depth 6. These resource stops prove neither
actual reachability nor impossibility of better certificates.

Exact unjoined predecessors of the excluded target have 4, 9, 12 and 18
pieces at depths 0 through 3. Their cumulative union has no overlap with
current mask-28/62 target initializations. They may still refine intermediate
histories. The retained artifacts are `c6_b58_review_terminal_complements`,
`c6_b58_review_exact_predecessors`, `c6_b58_memory_union_probe`,
`c6_b58_safe_piece_partition_probe` and `c6_b58_safe_piece_partition_control`
(`.py` and `.json`, under `artifacts/research`).

Coverage remains **40/56**, with the same sixteen original slabs open.
The next delivery should integrate selective fixed-safe-piece partitions
into the canonical owner, independently verify their exact coverage and
persistent clipping, and reuse one cover for all remaining whole targets.
Only then should target-relevant exact predecessors or a joint temporal
barrier extend the representation. Repeating the unchanged scalar-cut
operator is unnecessary. No new conditional or live trajectory step,
global boundedness proof, runtime continuation or physical confirmation is
claimed by B58.

Focused regression and report checks pass **142 tests**, including 64 new
cases. The partition validator independently rebuilds all 6,924 arcs and
54,353 complete forward updates, verifies the sixteen target observations
and matched-depth inclusion, and rejects sixteen adversarial partition
corruptions. Its record is
`artifacts/research/c6_b58_safe_piece_partition.validation.json`.
The full cut-campaign replay, scientific source delta and final checks are
retained in `artifacts/research/c6_winding_memory_cuts.validation.json`,
`artifacts/research/b58_scientific_source_delta.json` and
`artifacts/research/b58_final_validation.json`.


## 53. B59: canonical fixed-safe-piece memory certificates

### Proof-derived partitions of the unchanged nodal state

A region here is a set of carried numerical configurations of the six-node
ring. It is not a physical spatial area. The original 56 first-exit slabs
identify configurations whose next canonical nodal step crosses a declared
candidate boundary; they can overlap and their counts are not probabilities
or a percentage of a stability proof. A remaining slab is an unresolved
proof obligation. A candidate exit alone would not prove global divergence.

The selective partition from section 52 is now implemented in the existing
`c6_carried_return.py` owner. Its public entry points are
`derive_c6_carried_return_safe_partition` and
`derive_c6_carried_return_safe_partition_region_exclusions`.
They rebuild the canonical pressure/return/memory source before deriving any
cut. Exclusion proposals are full target regions, without trusted success
flags. The existing complete-layer query must first verify every proposed
exclusion from the unchanged origin. Failure leaves the partition explicitly
uninitialized and cannot supply a new exclusion.

For a verified target, the owner enumerates each direct intersection and
individual terminal preimage separately. It records the original memory,
exclusion group, target ordinal and terminal transition for every bad piece.
It never uses the hull of several predecessor pieces as a removable set.
The same target-piece reader initializes ordinary memory queries and
partition queries, so terminal visits without a later return retain one
semantics throughout the proof.

For source DBM `Z` and exact bad piece `P`, the owner derives a relative
facet basis for `Z intersect P`. Removing a redundant inequality is allowed
only when exact DBM closure remains equal to that intersection. The safe
complement splits by the first violated integer inequality, using
`k_j-k_i <= -b-1` to complement `k_i-k_j <= b`. Each subtraction record keeps
its source zone, bad-piece index, relative facets and disjoint safe pieces.
The reduction does not claim a unique or globally minimum facet basis.
Several bad pieces are subtracted sequentially without joining safe pieces.

### Complete graph construction and persistent source coverage

A partition vertex is `(canonical return index, safe-piece ordinal)`. It is
a proof label, not a newly invented engine edge. Its initial region is its
fixed safe seed; the unchanged B47 point has a separate sentinel. Every
canonical first-return image must remain covered exactly once by the seeds.
All compatible guarded partition pairs are built before forward refinement.
An interrupted construction publishes no partial pair relation.

Every accepted forward update recomputes every predecessor image, injects
the original first image and intersects the result with the fixed seed.
The update is reserved as a complete work unit before it begins. This keeps
all original domain-confined histories covered, preserves the certified
holes and prevents a partial predecessor union from excluding valid paths.
Separate limits bound construction work, pieces, arcs, forward work,
exclusion queries and residual target queries. These are computational
limits; they do not alter phase, capacity, pressure, carry or the nodal map.

The shared backward kernel processes all original full targets over the new
indices, including direct visits, the sentinel and terminal transient steps.
Initialization is charged to each target's budget. Only complete origin-free
empty or stationary layers prove exclusion. An exhausted budget retains the
last complete layer and reports an unresolved result; an abstract origin
collision is not an actual trajectory witness.

### Canonical campaign and next refinement

The existing benchmark adds `--method safe-partition` and binds fourteen
inputs. It reconstructs B58 completely, then uses the original B57 domain to
reverify the excluded mask-30 target. The scalar-cut domain cannot replace
that domain for this step: it has already removed the slab that must be
proved unreachable. The original cube, B47 origin and all full residual
targets remain separately identified. No conditional or live trajectory is
extended by this campaign.

The canonical run reproduces the four complement sizes 5/5/6/6, 874
partition vertices and 6,924 arcs. Construction uses 34,820 work items;
forward descent uses 499,996, with 54,353 complete visits and 29,885 strict
updates. The clipped target graph requires 14,722 work items. Every full
target has a separate budget of 250,000, including initialization:

| Original target family | Masks | Last complete depths | Result |
|---|---|---|---|
| Node 0 lower | 8, 16, 20, 24, 28 | 97, 94, 93, 94, 95 | Resource limit |
| Node 3 upper | 28, 60, 62 | 194, 85, 85 | Resource limit |
| Node 4 upper | 56, 57 | 71, 71 | Resource limit |
| Node 4 upper | 58 through 63 | 70 for each | Resource limit |

No new complete first-exit slab is excluded: the total stays **40/56**, with
sixteen open. Differences from the prototype's depths reflect the declared
accounting and representation; depth alone is not a precision comparison.
The campaign report is `artifacts/research/c6_winding_safe_partition.json`.

A bounded exact-predecessor study reuses the independently certified mask-30
result through twelve complete return layers. Every retained piece carries
its full guarded path, which is recomposed exactly. Its 1,499 retained pieces
require 159,759 intersections and 128,802 subsumption comparisons, with a
separate 17,483-intersection path check. They overlap retained B57 backward
hulls for thirteen of the sixteen remaining target families: node-0 lower
24/28, all three node-3 upper targets and all eight node-4 upper targets.
No single piece covers a complete target-history hull. Node-0 lower
8/16/20 have no such overlap within this bounded search.

There is no direct overlap with the original target initializations in
these twelve layers. Exact paths to a different exit cell must first
complete their prescribed in-cube returns, whereas a first-exit target
requires leaving before the next complete return. This chronology explains
why the approximate backward hulls are the useful refinement target here;
joined target initializations must not silently be treated as exact unions.
The prototype is `artifacts/research/c6_b59_exact_bad_preimages_probe.json`.

The next least expensive extension considers the six additional memory
classes (730, 735, 1199, 1204, 1232, 1237) already present one exact
predecessor step before the four terminal pieces: nine exact additional
regions. It should verify each full guarded path, keep exact disjoint safe
pieces and measure improvement against a matched baseline before increasing
the depth or budgets. Subtracting joined danger hulls remains unjustified.
Global boundedness, convergence, post-SHA runtime continuation and physical
correspondence remain separate open claims.

The focused implementation suite passes **225 tests**, including 83 new
cases. Independent replay checks the original exclusion proof, exact bad
pieces and disjoint subtraction, the complete graph and persistent-seed
worklist, and every original target's complete backward history and resource
stop. Verification and source capture are retained in
`artifacts/research/c6_winding_safe_partition.validation.json`,
`artifacts/research/b59_scientific_source_delta.json` and
`artifacts/research/b59_final_validation.json`.

## 54. B60: exact predecessor cuts and their computational cost

### Guarded paths justify each additional removed set

A verified whole-target exclusion permits removing an exact predecessor of
that target as well. In the last-return memory graph, let `Z_i` be the
retained endpoint cover, `G_ij` the complete clipped transition guard and
`s_j` the carried displacement of return record `j`. For an exact bad piece
`P_j`, the one-return predecessor is

`P_i = G_ij intersect (P_j - s_j)`.

Each nonempty piece retains the original bad-piece index and its ordered
successor return records. Composing that complete guarded word must recover
the same region and terminate in the original exact target preimage. The
deterministic canonical nodal map then makes an origin path to that piece
incompatible with the verified exclusion. A hull containing several exact
predecessors does not have this implication and cannot authorize a cut.
These are refinements of proof sets; the origin, pressure, capacity, phase,
carry, timestep and physical support are unchanged.

The bounded independent control reconstructs nine such pieces one return
before the four B59 terminal pieces, in additional memory classes 730, 735,
1199, 1204, 1232 and 1237. It verifies each complete path before forming
disjoint safe complements and retaining the original first-return points.

### Representation strength and work budget are distinct controls

The depth-one prototype has 916 partition vertices and 7,165 arcs, compared
with B59's 874 and 6,924. Its partition construction costs 37,950 work
items rather than 34,820, excluding the separately recorded predecessor
derivation. At equal complete forward sweeps zero through five, its
projected memory cover is always contained in the depth-zero cover. Strict
improvements affect 0, 0, 0, 2, 3 and 2 memories. Five sweeps cost 40,405
versus 38,990 forward work items. This establishes a bounded precision
gain at matched depth; it does not establish an equal-cost advantage.

At a shared nominal forward budget of 40,000, the depth-zero worklist uses
39,994 work items and completes 4,266 visits; depth one uses 40,000 and
completes 4,205 visits. Across the 1,265 original memory projections,
1,116 are equal, 146 are strictly tighter at depth zero and three at depth
one. No individual memory pair is incomparable, but neither complete
projected cover contains the other. The additional partition is therefore
not a uniform optimization under that fixed budget. The retained control is
`artifacts/research/c6_b60_depth_one_control.json`; its script records its
input hashes and independently recomposes all nine exact paths.

### Canonical integration and verified campaign

The existing safe-partition APIs now accept `excluded_predecessor_depth`,
defaulting to zero with the unchanged B59 result schema and work path.
Positive depth uses `C6CarriedReturnPredecessorPartition`; every generated
piece carries its original bad-piece index and future return-record word.
Complete layer records expose intersections and subset comparisons. Only
individually contained pieces may be pruned; unrelated paths are never joined.
Predecessor graph construction, intersections, comparisons and layer visits
share the existing construction budget. The existing piece limit also
bounds the cumulative bad pieces. An interrupted generation publishes no
usable partition, even if earlier complete layers remain as information.

The shared benchmark accepts
`--method safe-partition --excluded-predecessor-depth 1`, writes the distinct
`c6_winding_safe_predecessors.json` report and binds the unchanged fourteen
inputs. Depth zero remains the default; predecessor depth is a numerical
proof setting. The full sixteen-target campaign and independent replay are
complete. The focused suite passes 141 tests, including 58 new cases;
the 83 B59 regressions retain the default behavior and resource semantics.

The canonical predecessor graph costs 17,146 work items. Its complete
depth-one layer adds 392 intersections, six subset checks and one layer
charge: 399 items. The total construction is therefore 55,495, including
the 37,950 partition cost. The 916-vertex/7,165-arc forward worklist uses
499,998 work items, completes 55,106 visits and performs 31,225 strict
updates. The clipped target-query graph costs 15,246 items. Every original
target receives a separate 250,000-work budget, including initialization:

| Original target family | Masks | Last complete depths | Result |
|---|---|---|---|
| Node 0 lower | 8, 16, 20, 24, 28 | 93, 90, 89, 91, 92 | Resource limit |
| Node 3 upper | 28, 60, 62 | 176, 79, 79 | Resource limit |
| Node 4 upper | 56, 57, 58, 59 | 68, 67, 67, 67 | Resource limit |
| Node 4 upper | 60, 61, 62, 63 | 66, 66, 66, 66 | Resource limit |

There is no additional complete first-exit exclusion: the total remains
**40/56**, with sixteen open. Different partition indices and work per layer
mean that raw depth and occupied-index counts are not precision comparisons.
The report preserves the complete B47 origin, original candidate cube and
every full target; it adds no conditional or live trajectory step. Neither
these resource outcomes nor the bounded controls certify global stability.
A deeper partition requires evidence of utility under the declared resource
limit before becoming the next research step.

The independent comparison at the shared nominal 500,000 forward-work cap
finds B59's original-memory DBM hull strictly tighter in 454 histories and
equal in 811; B60 improves none. This is containment of the hull projections,
not containment of the complete nonconvex partition unions. Construction
work is additionally 34,820 for B59 versus 55,495 for B60. Together with no
new whole-target exclusion, this finite cost result supports retaining the
depth-zero default. It does not make exact predecessor cuts intrinsically
invalid or establish that they can never help another proof computation.
Independent evidence and complete source reconstruction are retained in
`artifacts/research/c6_winding_safe_predecessors.validation.json`,
`artifacts/research/b60_scientific_source_delta.json` and
`artifacts/research/b60_final_validation.json`.

### Next bounded discriminator

An independently recorded 144-intersection read-out compares the nine new
pieces with all retained B59 query hulls. Every piece overlaps a query hull
for each of the eleven node-3/node-4 upper targets; none overlaps a hull of
the five node-0 lower targets. No individual bad piece contains a whole
query hull. Simple overlap selection therefore cannot reduce the nine cuts
for an upper target. The smallest affected B59 frontier is node-3 upper
mask 28: 76 occupied partition histories at complete depth 194. These are
finite proof-set diagnostics, not physical likelihoods or trajectory visits.
The evidence is `artifacts/research/c6_b60_target_relevance.json`.

Since B60 closes no additional target, the single A/B test completed in section 55 retains
its depth-one seeds, root images and arcs exactly and change only the order
of complete forward updates for that original mask-28 target. A seed's
intersection with a retained B59 query hull can supply advisory priority;
it cannot authorize a cut. Charge priority classification and all complete
visits to one shared 500,000-work cap, then use the unchanged 250,000-query
cap. Two FIFO queues keep the priority rule reproducible. Success requires
a new complete exclusion or componentwise strict containment at a common
complete query depth without extra work. Otherwise reject that efficiency
hypothesis for the tested target. Do not increase predecessor depth or
change the nodal map as part of this comparison.
The experiment needs both the sealed B60 FIFO control at identical geometry
and the best verified B59 baseline. Improvement over B60 alone is diagnostic;
adoption requires a useful gain over B59 as well. Compare original-memory
projections at common complete query depth without promoting that comparison
to an unproved containment of the full nonconvex unions.

## 55. B61: priority refinement excludes another original exit label

### Count and coverage premises

The 56 labels are an enumerated property of the retained C6 proof domain,
not a structural constant. Two adjacent binary64 values per EPI coordinate
give 64 rounding cells. Six coordinates and two exit directions give 768
candidate conditions. Exactly 72 are nonempty on the initial cube. The
verified B49 source envelope removes eight node-3 lower and eight node-5
lower conditions, leaving 56. The count uses the original cube boundaries
and canonical carried nodal increments. Nonempty local conditions do not
establish origin reachability, and labels need not define disjoint sets.
The independent reconstruction is
`artifacts/research/c6_b61_region_count.json`.

Later origin covers narrow these geometric slabs without changing their
labels. In particular, let `U` be the original B51 node-3 upper mask-28
slab and `D` the independently certified B53 origin cover. The B57-B60
target carried into B61 is exactly `U intersect D`. The separate coverage
bridge verifies that equality, recomputes all 4,096 clipped B53 image pairs
(830 nonempty), and checks the origin-injected fixed point and four cuts.
It reuses the hash-bound earlier 32-exclusion coverage certificate. The
new label is not among the assumed exclusions. Both `U` and `U intersect D`
remain geometrically nonempty; exclusion below concerns their reachable
history observations. This distinction prevents promoting emptiness on an
arbitrary smaller target into a whole-label claim.

### Advisory priority preserves the complete history cover

The scratch experiment keeps the B60 geometry unchanged: 916 vertices,
7,165 arcs, every persistent seed and every original root image. A seed's
intersection with a retained B59 query piece in the same original memory
assigns priority; it authorizes no cut. Two FIFO queues retain all pending
vertices and their fixed priority. Every accepted visit combines all
guarded predecessor images with the original root, intersects the fixed
seed and commits only a complete update. Consequently, changing the visit
order preserves coverage of every original-domain origin prefix. Finite
resource limits need not reach a fixed point or give a fair infinite schedule.

Classification selects 109 priority vertices and costs 992 comparisons.
Forward updates use another 498,989 work items, for 499,981 under the
shared 500,000 cap. The run completes 32,112 visits and 30,111 strict
updates, leaving 625 vertices pending. These queues are retained explicitly;
their presence prevents a claim of global fixed-point completion.

### Complete target observation and independent verification

The whole mask-28/node-3/upper target has no direct, terminal-transient or
origin-sentinel observation in the refined cover. Full initialization
produces zero pieces and an empty complete layer at depth zero, using
1,697 work items. The clipped query graph costs 15,231 items. The complete
cost, including the B60 partition construction, is
`55,495 + 499,981 + 15,231 + 1,697 = 572,404`.

B59 and B60 FIFO each retain one nonempty target piece in original memory
374 at the same complete depth. The B60 same-index comparison has one
strictly tighter query vertex (283) and 916 equal entries, including the
origin sentinel. Original-memory hull comparison likewise has one strict
improvement. This does not identify the full nonconvex unions. B59's
construction-inclusive depth-zero cost is 551,157; the experiment therefore
succeeds by a new complete exclusion, not by a lower-cost initialization.
No blanket performance claim or default-policy replacement follows.

Since every origin prefix before first cube exit lies in `D`, absence of
all complete observations of `U intersect D` excludes the original label
`U`. Coverage rises to **41/56**, leaving **15** labels:

| Remaining original target family | Masks |
|---|---|
| Node 0 lower | 8, 16, 20, 24, 28 |
| Node 3 upper | 60, 62 |
| Node 4 upper | 56, 57, 58, 59, 60, 61, 62, 63 |

Independent arithmetic replay checks every forward visit, pending queue,
root, seed and full target observation, plus B59/B60 common-depth controls.
Thirty-one separate corruption checks use hash-bound cached arithmetic
expectations to exercise the real evidence admission/comparison assertions;
they are not fresh full arithmetic replays. Twenty-four fresh finite-oracle
controls check reachable-state coverage under two priorities and twelve
budgets, with partial-visit, terminal-only and sentinel checks. Evidence:

- `artifacts/research/c6_b61_target_priority.json`;
- `artifacts/research/c6_b61_target_priority.validation.json`;
- `artifacts/research/c6_b61_original_label_coverage.json`;
- `artifacts/research/c6_b61_priority_adversarial.json`;
- `artifacts/research/b61_final_validation.json`.

The production scientific snapshot remains B60. Physical parameters,
pressure, timestep, phase, origin and carry are unchanged; no conditional
or live trajectory is advanced. Global C6 boundedness, convergence and
post-SHA runtime behavior remain open.

The next implementation gate centralizes the optional priority policy in
the existing return-proof owner while preserving FIFO and depth-zero
defaults. It must reproduce the sealed result. Target-specific scheduling
does not make the retained cover target-specific in its coverage: it still
contains every original-domain prefix. Reuse that cover for the other
fifteen full targets, retaining their coverage premises, terminal visits,
origin sentinel and declared work accounting. No deeper cuts or additional
physical assumptions are needed for this next comparison.

## 56. B62: canonical synergies and proof ownership

### What belongs to the nodal model and what belongs to its representation

The physical variables remain EPI, structural capacity and phase on the
declared graph, with pressure supplied by the existing canonical producer.
The current continuation holds support, phase and unit capacity fixed and
uses the declared timestep `h=1/16`. Those assumptions identify the case;
the nodal factorization alone does not derive their values or choose an
operator schedule. The finite B43 preparation and B47 carried origin remain
the provenance boundary. Future live admission and post-SHA continuation
are separate obligations.

The exact remainder in `X=x+r` is numerical bookkeeping, not an additional
physical TNFR field. The shared kernel advances `X_next=X+h*p(x)` and reads
pressure from the represented `x=RN(X)`. Its validation of this numerical
map is not a solver-convergence theorem for a continuous trajectory. The
integer grid, regions, memory indices, proof coordinates and priority queues
describe or bound that same map. They must never be inserted into its
pressure as new physical feedback.

### A reusable certificate for retained history covers

The descending worklist has a local verification rule that does not require
replaying its discovery order. For canonically certified fixed seeds `S_v`,
root images `r_v`, complete guarded arcs `u -> v` with displacement `a_uv`,
define the monotone map

```text
F_S(Z)_v = S_v intersect hull(r_v,
                           { (Z_u intersect G_uv) + a_uv : u -> v }).
```

If `r_v subset Z_v subset S_v` and `F_S(Z)_v subset Z_v` for every vertex,
induction over complete returns covers every domain-confined history. The
seed coverage theorem supplies the clipping premise. Complete intermediate
and terminal visits and the origin sentinel must still be reconstructed
when observing a target. This is closure of a clipped proof relation, not
physical confinement in the original cube.

Starting from `S`, the inclusion `F_S(S) subset S` holds by construction.
An atomic update replaces one component by its full `F_S` image. Monotonicity
and the shrinking components preserve `F_S(Z) subset Z`, even at a finite
resource stop. Consequently a one-pass closure check can verify the retained
cover independently of the priority history. It must also authenticate the
source map, seeds, roots, arcs and prior exclusions; arbitrary user-supplied
matrices and truth flags are not certificates.

For two covers closed under this same `F_S`, their componentwise intersection
is also closed: monotonicity bounds its image by the image of each cover.
When partitions differ, a hull projection into an original memory may add
spurious states. Intersection with that projection remains a reachable-state
cover if both original coverage premises are valid, but closure under the
other partition's `F_S` requires a separate check. Neither hull comparisons
nor root inclusion alone justify that stronger claim.

### Reuse priorities and mathematical discriminators

The shared return owner already builds target observations for many groups
after one partition construction. The remaining integration should expose
validated retained-cover reuse and the optional priority policy through that
owner, keeping FIFO and depth-zero defaults. It should not create a second
production DBM, pressure, return-graph or target-query implementation.

New original-label exclusions can feed the exact unjoined predecessor and
safe-subtraction owners. They must be reverified independently of the new
cuts, preserve all roots and terminal visits, and demonstrate a nonredundant
refinement before a broader campaign. Merely repeating old scalar cuts,
deepening every predecessor or increasing work limits has already failed to
supply an equal-cost gain on the retained controls.

For the new mask-28 exclusion this distinction matters operationally: its
B61 initialization is empty, so extracting bad pieces from that same cover
adds nothing. B59/B60 still expose an exact piece in original memory 374.
A reusable exclusion adapter would bind the independently verified B61
premise to that weaker reference before exact predecessor subtraction.
Requiring the weaker memory-only query to rediscover the stronger proof
would discard the very synergy being tested.

The next dynamical synergy is local pressure residence combined with guarded
return memory. A node's pressure on the fixed phase slice depends on its
local EPI stencil; several pending labels share that stencil. Its signed
nodal area bounds a consecutive residence, but a changed neighbor ends that
premise. Any certificate must retain ingress, carry and pressure changes
across successive local episodes. A local deadline alone excludes neither
reentry nor the first exit. High-node-2 terminal visits cannot be discarded
just because the low-return graph omits them as vertices.

Mean and shape also remain coupled obligations. The existing exact profile
identity separates spatial contraction from the signed mean contribution of
the represented phase source and pressure rounding. A memory-dependent
budget on a newly refined history cover is a legitimate new discriminator;
the refuted common mean separator, global displacement counts and unguarded
label cycles are not. Proof coefficients may be proposed and then checked
exactly, but they do not become fitted pressure parameters.

No symmetry quotient follows from the cycle graph alone: phase source,
binary64 chart, full carried origin, guards and targets must respect the
same permutation. Likewise, tetrad diagnostics cannot replace the carried
state and chronological constraints required by these exclusions.

### Bounded cover controls and decisions

The retained evidence supports a finite discriminator before another full
campaign. `artifacts/research/c6_b62_cover_synergies.json` observes all fifteen
pending full targets on five covers: B59, B60, B61, B61 intersected with
B59's original-memory hull, and the same-index intersection of B61 with B60.
Every initialization retains direct, terminal and origin-sentinel checks.
All 75 initializations are nonempty. The study performs zero forward
refinement visits, backward layers or trajectory steps: it establishes
neither another exclusion nor failure of a future backward query.

Intersection construction uses 1,832 operations. Three one-pass image checks
cost 8,081 each; two complete query-graph constructions cost another 30,462,
for 56,537 geometry operations in total. Target observations cost 103,620.
These are declared proof-operation counters, not wall-time speedup estimates
or the historical cost of discovering the parent covers.

The B61 image inclusion passes, as does the same-index B61/B60 intersection.
The B59-hull intersection retains roots and stays inside seeds but fails
image inclusion at thirteen vertices. Its reachable-state coverage still
follows from the two original premises; a standalone one-pass certificate
on the B60 graph cannot replace those premises. This control demonstrates
why matching memory indices and distinguishing hulls from unions matters.

The original-memory projections of B61 are stricter in 53 memories, weaker
in 377, incomparable in 50 and equal in 785 relative to either baseline.
Both intersections improve 103 projections versus their respective baseline
and 427 versus B61, yet their closure outcomes differ. Equal counts do not
identify the sets or their proof properties. These are projection results,
not full nonconvex-union containment claims.

Both inspected clipped query graphs have 911 active vertices including the
origin sentinel, four strongly connected components and one component of
908 vertices. Every one of the 910 history vertices is graph-reachable from
the sentinel; graph reachability does not establish a compatible carried
trajectory. A coarse component decomposition supplies little separation
here. The two node-3 upper targets remain terminal observations: B61 reduces
their piece counts from 11 to 10 and 14 to 13 relative to B59/B60 without
removing them. They must remain explicit in every subsequent certificate.

The independent static nodal study rederives all 64 represented pressure
rows and calls `observe_c6_frozen_pressure_stencil` for each pending family.
The sufficient uniform bounds are 189 confined updates for node 0, 180 for
node 3 and 3,759 for node 4. Each is the floor of the closed center-cell
width divided by the absolute exact nodal increment. It assumes unchanged
local displayed triples at every endpoint and is not a sharp deadline from
B47. Node 3's upper targets are already high-node-2 transients, making that
residence bound unhelpful alone. Node 0 is the first proposed history/age
relevance test; a literal 3,759-state node-4 counter needs a demonstrated
benefit before its construction. Every two-step return must update this
observer twice and preserve any intervening stencil change.

Among all twelve literal dihedral coordinate permutations, only identity
preserves the original candidate, and only identity preserves the represented
fixed phase-source vector. This is a finite obstruction to a bare coordinate
symmetry shortcut for this case, not to all other mathematical transformations.
Exact inputs and rational bounds are retained in
`artifacts/research/c6_b62_nodal_synergy_controls.json`; implementation owners
and the proposed bounded tests are mapped in `c6_b62_nodal_synergies.md` and
`c6_b62_proof_synergies.md` in the same directory.

Accordingly, B62 identifies reusable cover certification and priority
integration, implemented in B63 below, followed by a measured nonredundancy
check for the new exact mask-28 predecessor cuts and local-stencil/history
budgets. Do not repeat
the five-cover initialization test or a bare component split as if either
were an untested route to closure. Coverage remains **41/56**, with **15
pending**. No production source or physical parameter changes in this study.

## 57. B63: shared priority discovery and retained-cover verification

The shared return owner now implements the B62 integration gate. The
existing FIFO safe-partition APIs and their depth-zero defaults retain their
result classes, fields and work semantics. Only their descending update
loop is extracted into one private kernel. Two new public entry points,
`derive_c6_carried_return_safe_cover` and
`derive_c6_carried_return_safe_cover_region_exclusions`, reuse the same
construction, subtraction and whole-target query owners.

### Primitive reconstruction is the admission boundary

The public functions accept canonical source/state/domain primitives,
exclusion proposals, explicit resource limits and optional exact candidate
matrices. They reconstruct the canonical return-memory relation, reverify
the excluded target and rebuild every safe seed, root and guarded arc.
An externally constructed partition, certificate object, arc list or success
flag cannot replace those premises. Mutable caches on a supplied pressure
reference are rederived from its primitive inputs.

Supplying a retained candidate skips forward discovery. Admission requires
one closed seven-by-seven integer DBM or `None` per canonical vertex, exact
root/seed inclusion and the complete image inequality from section 56.
Malformed integers, including booleans, are rejected. An incomplete or
failed check exposes no accepted retained cover and cannot authorize a
target query. Strict image inclusion is sufficient; equality is not required.
The query entry point reconstructs and checks once, builds its clipped graph
once and shares it across independently quantified complete target groups.

### Advisory scheduling and complete accounting

Without hints, the extracted kernel follows the old FIFO order. Optional
ordered `(original_memory_index, DBM_or_None)` hints select fixed priority
through seed intersection. They authorize no cut and need not be accepted
as a physical trajectory or an exclusion proof. Shape/index/closure validation
is separately bounded and reported; seed classification and complete updates
share the forward-work cap. A seed without a matching advisory class still
costs one classification visit. Supplying both hints and a retained candidate
is rejected because discovery and candidate checking are distinct modes.

Every forward visit reserves its entire predecessor-plus-seed work before
committing. Both pending queues, priority flags, complete visits and strict
updates are retained. The companion schedule/check records avoid changing
the legacy partition schema. Exhaustion during classification, cover checking,
query-graph construction, initialization or a backward layer yields an
explicit incomplete result rather than an exclusion.

### Canonical campaign and regression boundary

The existing benchmark enables the new path with

```text
python benchmarks/c6_winding_invariant_region.py --method safe-partition \
  --excluded-predecessor-depth 1 \
  --priority-input artifacts/research/c6_winding_safe_partition.json \
  --priority-target 28 3 upper
```

It retains all fourteen original ancestry inputs and separately hashes the
B59 advisory snapshot as input fifteen. The original source/domain, cube
and complete target labels come from canonical ancestry reconstruction;
the advisory file supplies scheduling data only. Output is
`artifacts/research/c6_winding_safe_cover.json`, with claim
`O3.a-C6-carried-safe-cover` and `B63` fields. A default invocation without
priority input continues to produce the historical B59/B60 report path.

The integration reproduces the B61 priority geometry and counters exactly:
109 priority vertices, 992 classification operations, 499,981 forward work,
32,112 complete visits, 30,111 strict updates and 625 pending vertices.
The new explicit admission costs are 950 hint-validation operations and
12,655 retained-cover check operations: 4,574 validation/inclusion checks
plus the 8,081 image operations already identified in B62. The full mask-28
target still has empty initialization, including terminal and origin cases.
These are computational certificate costs, not new physical parameters.

The full fifteen-input production campaign reconstructs the ancestry and
queries all sixteen original pending labels. Mask28/node3/upper is excluded
with 1,697 initialization operations; the other fifteen queries each stop
at their 250,000-intersection limit. Thus the total remains **41 of 56**
whole first-exit slabs excluded. No new slab beyond B61 is claimed, and
an incomplete backward query does not establish a reachable exit.

The focused suite passes **232 distinct tests**, including 91 new cases:
65 shared-owner controls and 26 benchmark/report controls. The initial
mixed regression run contains 23 report cases repeated within the final
26-case run; these are counted once. Controls cover finite reachable-state
oracles, malformed and forged certificates, complete image inclusion,
advisory scheduling, budget exhaustion and legacy serialization.
The scientific snapshot is retained by
`artifacts/research/b63_scientific_source_delta.json` and its companion ZIP,
with full source digest
`sha256:85cc69c97b874814270be42b0592a337134e49b623515cc072f7f3ab51cd1063`.

Independent validation in
`artifacts/research/c6_winding_safe_cover.validation.json` replays the
priority arithmetic and all sixteen complete target queries. It verifies
three cover checks against B62: the B61 candidate and same-index B60
intersection pass; the projected B59-hull intersection fails at the same
thirteen vertices. The shared FIFO kernel reproduces the exact B59 and B60
retained arrays, queues and counters; these controls are captured once in
`c6_b63_fifo_controls.json` and bound by the final validator. All fifteen
input snapshots and scientific source hashes remain unchanged. The official
report digest is
`74dd580f738c64b61bdcd28d48cef8d7070cd1f559f1fec05ac0e208e47d60e7`.

Scope remains domain-confined origin-history coverage. In particular,
accepting `F_S(Z) subset Z` on the clipped graph does not prove confinement
in the original cube, future live admission or post-SHA runtime stability.
Every canonical pressure, physical setting and initial carried coordinate
is unchanged by this implementation.

## 58. B64: exact exclusion transfer and local chronology

B63's mask28/node3/upper exclusion can supply constraints to a weaker
history cover. The premise is the independently verified whole original
target under the same B47 origin and domain, including its original-label
coverage bridge. A success flag or a retained hull alone is insufficient.
The detached study binds the original cube, return relation, signed shifts,
memory relation and full target matrices before reusing that premise.

### New exact pieces and their limits

In each B59/B60 retained cover, the target has one terminal preimage at
partition vertex 283, original return memory 374, terminal transition 290.
The two covers give different exact matrices, so their pieces are derived
separately. The shared target owner retains the individual terminal guard;
it does not subtract a joined backward hull. For a guarded return
`k_next=k+a`, the exact predecessor of a forbidden piece `P` is
`G intersect (P-a)`. The complete return guard already includes its
intermediate visit where present.

One complete predecessor layer gives nine further pieces, in original
memories 631, 730, 735, 1142, 1173, 1199, 1204, 1232 and 1237. Sequential
exact integer-DBM subtraction shows that all ten pieces have nonempty
remainders outside the old mask30 bad-piece unions, including their B60
depth-one extension. Each piece is disjoint from those old unions. Feasible
integer potentials witness geometric nonredundancy; they are not claimed
to satisfy the finer runtime arithmetic class or to be reachable from B47.

All ten pieces intersect both weaker retained covers, but none intersects
the B63 retained cover or any first-return root. This is consistent with
B63's complete image inclusion and empty target initialization: guarded
predecessors on that same complete operator cannot supply a new forbidden
piece inside its already accepted cover. Transfer into a weaker partition
is therefore a distinct question from improving B63 directly.

The bounded extraction, exact differences, cover intersections and complete
initialization controls cost 39,155 operations for the B59-derived pieces
and 40,251 for the B60-derived pieces. Of these, target extraction costs
1,619/1,703 and predecessor construction costs 17,302 in either case,
including the 17,146-operation complete memory graph and 155 guarded
intersections. Budgets are 200,000 operations and 5,000 pieces per variant;
they are proof-computation limits, not physical coefficients.

Subtracting these pieces leaves all fifteen pending whole-target
initializations unchanged in each weaker cover. Thus this test supplies
new nonconvex geometric information but no additional regional exclusion.
Exact evidence is retained in
`artifacts/research/c6_b64_mask28_cut_probe.json`; all computation reuses the
shared return/DBM owners with unchanged scientific source.

### Matched transfer control

The new pieces are tested in a B59 continuation from its retained cover,
with identical fixed starting state and a 40,000-operation forward budget.
The baseline, direct terminal cut and full ten-piece cut are kept separate.
Each child seed is an exact safe difference; roots and every old guarded
arc are clipped against all appropriate children. None of the initial
parent DBM hulls changes: joining the new safe pieces immediately would
discard their nonconvex information.

| Variant | Classes | Arcs | Construction work | Used forward work | Tighter / weaker / equal final parent hulls |
|---|---:|---:|---:|---:|---|
| Baseline | 874 | 6,924 | 23,394 | 39,995 | 0 / 0 / 875 |
| Direct cut | 875 | 6,959 | 23,589 | 39,998 | 0 / 12 / 863 |
| Ten cuts | 920 | 7,256 | 26,187 | 40,000 | 5 / 188 / 682 |

The final comparison includes the original zero sentinel and concerns
parent DBM hulls only; it does not compare complete nonconvex unions.
Atomic visits account for the small unused forward budgets. All cover
gates pass and all 45 complete pending-target initializations remain
nonempty. Declared incremental costs, including cover checks, query-graph
construction and observations, are 110,229/110,563/115,449; inherited cut
derivation additionally costs 0/1,619/18,921. This bounded control supplies
no componentwise improvement at the selected budget and does not justify
promoting either split policy. It does not exclude improvement at a
different, separately justified budget or for another representation.
Evidence: `artifacts/research/c6_b64_cut_transfer_control.json`.

### Exhausting the local node-0 hull calculation

The five pending node-0 targets share the same displayed `(5,0,1)` stencil.
In the B63 cover, this predicate selects 35 history vertices. Its complete
clipped graph has 228 incoming arcs from outside the class and only six
internal arcs, with no composable internal pair. Outside histories remain
held at their authenticated B63 cover; they are not deleted or considered
unreachable. Every two-step return is expanded when checking the stencil,
and standalone terminal visits are audited separately.

Two complete synchronous local image passes, 271 operations each, reach
the local fixed point. Three zones tighten, at vertices 188, 215 and 243.
The full global image remains included in the resulting cover, while all
outside zones and original roots are preserved. Complete target arrays
tighten for masks 16 and 24, but the five nonempty history counts remain
1/9/9/7/6 for masks 8/16/20/24/28. The independent replay agrees exactly.

The local DAG exhausts this particular hull update with held exterior.
It is not a global bound on repeated departure and reentry. Introducing a
189-state residence counter without first testing its relevance would add
complexity to a local chain that already has no two successive internal
return arcs. The appropriate finer control is the exact union of separate
ingress paths, retaining intermediate and terminal observations.
Evidence: `artifacts/research/c6_b64_local_stencil_probe.json` and its
independent `.validation.json`.

### Separate ingress paths retain additional correlations

The next bounded control uses exact unions at the same 35 local vertices,
holding every exterior history at its B63 cover. For each exterior ingress,
it retains the guarded image as one piece with its source preimage and
complete nodal word. The local DAG then propagates each individual piece,
without a convex join. There are 228 ingress pieces, no local root pieces
and 97 possible internal extensions; 66 extensions are empty, leaving 31.
The complete local representation therefore contains 259 pieces, below
the 325-piece a priori bound. This changes the proof representation, not
the nodal pressure or any runtime variable.

Every original root and guarded arc is checked against the combined
representation: the local image has a retained individual-piece witness,
while the exterior remains inside the inherited cover. Admission uses
these exact pieces; their hulls are comparison diagnostics only. The local
union's hull is strictly tighter than the local DBM fixed point at vertices
20, 63 and 65. This demonstrates additional information lost by joining
ingresses before propagating them, even after the hull update has converged.

Complete target observations include all exterior histories, every local
piece, terminal first steps and the original zero sentinel. For masks
8/16/20/24/28, the nonempty piece-history counts are 2/20/13/17/7; they
project to the same 1/9/9/7/6 original histories. All five targets remain
nonempty. These are overapproximated possibilities, not proven trajectories.

Charged federation work is 30,442 operations: 689 for construction, 21,916
for complete root/image coverage and 7,837 for target initialization.
The inherited B63 cover gate additionally costs 12,655 operations, and two
shared-owner initialization cross-checks cost 15,674. These validation
costs are separate from the 30,442 counter. The retained evidence is
`artifacts/research/c6_b64_local_ingress_federation.json`.

The regional total remains **41/56 excluded, 15 pending**. The next useful
constraint must affect the surviving exterior ingress histories or a
correlation those histories still lose. Repeating the local hull fixed
point, joining the ingress pieces again, or adding the unused 189-state
counter does not address that remaining boundary. Production dynamics,
the B47 origin, physical parameters and historical artifacts are unchanged.

The smallest surviving family is mask8/node0/lower: pieces 19 and 55 at
vertex 180 (memory 262, `48 -> 8`) enter from exterior vertices 45 and 97
(memories 84 and 156, `18 -> 48` and `26 -> 48`). Those two exterior
histories have eight and 26 incoming original arcs, respectively, and no
crossedge. The next bounded test is their complete 34-candidate exact
prehistory layer. The other targets have 18/13/15/7 distinct exterior
feeders; broad history growth is deferred until the smallest family has
been evaluated. Provenance and implementation notes are centralized in
`artifacts/research/c6_b64_local_history_design.md`.

All four B64 studies have independent arithmetic replays in their matching
`.validation.json` artifacts. The exact predecessor-owner regression suite
passes 22 tests; this block changes no scientific production source. The
combined evidence, unchanged source and documentation checks are captured
by `artifacts/research/b64_final_validation.json`.

## 59. B65: a complete mask-8 prehistory layer

B65 evaluates the smallest B64 family without enlarging the domain,
changing the pressure, advancing the origin or joining predecessor pieces.
The complete federation initialization of mask8/node0/lower has exactly
two direct observations, pieces 19 and 55 at local vertex 180; its terminal
and zero-sentinel observations are empty. The original B51 unsafe slab
intersected with the certified B53 pre-exit domain is rechecked to equal
the complete refined target. That domain equality and the inherited
origin-history cover remain distinct premises of the calculation.

For a target observation `Q_b` reached on an ingress `a -> b` with guard
`G_ab` and shift `s_ab`, retain its exact feeder preimage

```text
B_a = Z_a intersect G_ab intersect (Q_b - s_ab).
```

For every original incoming return `c -> a`, independently compute

```text
P_c = Z_c intersect G_ca intersect (B_a - s_ca).
```

Each complete return guard includes any intermediate nearest-rounding
visit. Local roots, feeder roots and the surviving source-root intersections
are checked separately. The resulting pieces are necessary prehistories
within the fixed B63 cover; nonemptiness proves neither coordinate-coset
feasibility nor actual reachability from B47. In particular, a surviving
piece cannot be subtracted as though the target had been excluded.

The two feeders have eight and 26 incoming arcs. Of these **34 complete
candidates, 27 are empty and seven survive**:

| Local piece / feeder | Incoming arc | Source vertex | Original return memory |
|---|---:|---:|---:|
| 19 / 45 | 2 | 1 | 13 |
| 19 / 45 | 36 | 6 | 22 |
| 19 / 45 | 2082 | 138 | 208 |
| 55 / 97 | 217 | 21 | 43 |
| 55 / 97 | 474 | 39 | 74 |
| 55 / 97 | 1108 | 66 | 115 |
| 55 / 97 | 1396 | 87 | 146 |

All checked root intersections are empty. The calculation uses 86 charged
DBM intersections: one original-target bridge, two observation checks,
four ingress intersections, four local/feeder-root checks, 68 predecessor
intersections and seven surviving-source-root checks. This counter does
not include inherited evidence verification, arc enumeration, exact
subset/word audits or independent replay. No cover is modified and no
second prehistory layer is run.

### Common nodal content, distinct full histories

The surviving constrained suffixes are `18 -> 48 -> 8` and
`26 -> 48 -> 8`. In this data each return is one nodal step; there are no
hidden intermediate rows. With the unchanged `q=2^-113`, `h=1/16` and
unit capacity, both suffixes have exact node-0 increments

```text
(h * DeltaNFR_0(first), h * DeltaNFR_0(48))
    = (259574703981392*q, 259574703981392*q).
h * DeltaNFR_0(8) = -3039824576180305*q.
```

The two increments and the hypothetical following mask8 update sum to
`-2520675168217521*q`. This is the signed accumulated nodal budget for
those specified rows, not an observed trajectory or a repeated-cycle
theorem. The same local node-0 pressure can occur with its left/right
displayed neighbor values exchanged. Equality of that scalar read-out
does not identify the six-coordinate pressures, carry guards or source
histories; the two families therefore remain separate in the proof.

The next complete layer would have 55 candidates. Its smaller complete
family is feeder 45: sources 1, 6 and 138 have two, five and nine incoming
arcs, respectively, giving **16 candidates plus three root checks**.
Feeder 97 requires a separate 39-candidate family. These counts are not
full computation budgets: each candidate needs up to two DBM intersections,
with relation enumeration, provenance and validation separately counted.
The authenticated previous-memory labels provide context; they have not
been promoted to an additional replayed predecessor layer.

The regional count remains **41/56 excluded and 15 pending**. The 27
eliminations are prehistory candidates, not 27 additional regions. Full
records are retained in `artifacts/research/c6_b65_mask8_prehistory.json`;
its independent validation and the nodal interpretation are sealed by
`c6_b65_mask8_prehistory.validation.json` and `b65_final_validation.json`.

## 60. B66: exact geometric gain and reuse of local histories

This block tests whether another prehistory layer actually removes geometric
possibilities, rather than merely renaming them. It retains the B63 source,
B47 origin, nodal pressure, timestep and rounding model. The complete B65
target/domain, root, terminal and sentinel checks remain sealed premises.
The selected family is piece19/feeder45; feeder97 remains unchanged.

For a retained parent piece `P` at vertex `v`, include its root intersection
and **every** original incoming arc `u -> v`. Its necessary child piece is

```text
Q_uv = Z_u intersect G_uv intersect (P - s_uv).
H_v = (root_v intersect P) union union_uv (Q_uv + s_uv).
```

`H_v` is a subset of `P`. Exact integer-DBM subtraction computes
`P minus H_v` without joining the images. A nonempty difference proves a
loss of possibilities at the same parent coordinates. Counting empty
children alone does not establish this gain. Translations through the
already constrained suffix permit the same comparison at the target.
These are target-conditioned restrictions; they are not globally forbidden
cover zones and cannot be promoted to cuts without a new admission argument.

### A complete selected family, with measured gain

All two, five and nine incoming arcs of parent vertices1,6,138 are checked,
including the three root cases. Of **16 candidates, 12 are empty and four
survive**. All surviving source-root intersections are empty.

| Parent | Surviving incoming arcs | Source vertices | Complete nodal word |
|---|---|---|---|
| 1 | 2953, 3235 | 180, 207 | `8 -> 18 -> 48 -> 8` |
| 6 | none | none | no surviving piece |
| 138 | 232, 569 | 22, 41 | `40 -> 18 -> 48 -> 8` |

Each return in these words is one nodal step. Their shifts are rederived
from `h * DeltaNFR`, with every canonical guard and any intermediate visit
audited. A repeated displayed row in the first word does not identify the
carried state or prove a periodic orbit.

The exact differences at the three parents have4,1,9 pieces, respectively.
The one difference at vertex6 is its entire former parent piece. The union
of all four surviving target endpoint pieces is also strictly smaller than
the previous selected-family endpoint union, but remains nonempty.

### Reusing an admitted cover reveals information that a hull loses

Two surviving sources, vertices180 and207, already have complete exact B64
local covers. Their three and eleven existing pieces can therefore be
intersected with the new preimages without generating another history layer.
All14 intersections are evaluated. At180, pieces19,55,133 survive; at207,
pieces7,21,35,57,125 survive. Sources22 and41 lie outside this local cover
and retain their B63-derived preimages unchanged.

There are consequently eight local intersections plus two exterior pieces.
Although this representation has more pieces, its exact endpoint union is
strictly smaller. Subtracting it from the four pre-reuse endpoint pieces
leaves2,8,0,0 residual pieces. The DBM hull before and after this reuse is
**identical**: joining the pieces would erase all of this additional gain.
Piece counts and hull comparisons therefore cannot replace exact-union
comparison. This is a demonstrated synergy between the B64 admitted cover
and B66 prehistory, not a new physical memory variable.

Charged work is726 DBM operations: three parent-root checks,32 candidate
intersections, four child-root checks,130 parent-union difference work,
272 selected-family endpoint difference work,14 reused-piece intersections
and271 further endpoint difference work. Hash binding, enumeration, nodal
word/subset audits and independent validation are separate. No production
source, runtime trajectory or physical coefficient changes.

An additional complete-observation relevance check restores all four
unchanged feeder97 endpoint pieces. Comparing the seven B65 endpoint
pieces with the fourteen B66 pieces gives exact residual counts
`14,2,28,0,0,0,0`, at a separate cost of2,243 subtraction operations. Thus
the gain is still strict for the combined necessary mask8 endpoint union,
not merely within one feeder. Its hull also shrinks across the complete
B65-to-B66 change; this is distinct from the unchanged hull in the second,
reuse-only phase. The combined union remains nonempty.

Evidence is retained in `artifacts/research/c6_b66_feeder45_gain.py/.json`,
with independent replay in `validate_b66_feeder45_gain.py` and the matching
`.validation.json`. `b66_final_validation.json` seals the complete checkpoint.
The regional result remains **41/56 excluded, 15 pending**. Neither feeder45
nor the whole mask8 target is excluded; feeder97's four B65 candidates are
unchanged. Nonempty DBMs do not prove finer coordinate-coset feasibility or
origin reachability. Indefinite C6 boundedness and future runtime remain open.

### Research value and the next decision

These calculations address this numerical C6 proof; their necessity for
understanding TNFR in general is not established. The reusable content is
exact accumulated nodal accounting, complete guarded induction and measured
control of information lost by joining histories. Binary64 carry and proof
history labels do not introduce new ontological primitives. This block
establishes neither a general multichannel stability theorem nor physical
correspondence with laboratory observations.

The next bounded gate is feeder97's complete39-candidate layer at B65 sources
21,39,66,87 (5,15,2,17 incoming arcs), with all root cases. Preserve the ten
new feeder45 endpoint pieces and reuse existing admitted local pieces where
applicable. Measure exact endpoint-union loss both within the selected
family and after restoring the other family. Do not increase depth merely
because many child candidates are empty; require demonstrated geometric
gain, a smaller complete certificate or a new useful nodal discriminator.
The broader mechanism programme should distinguish contraction of spatial
differences from control of signed uniform drift, using the existing
mean/shape, diffusion and event-budget owners. Its scope and the C6 proof
engineering boundary are reviewed in
`artifacts/research/c6_b66_research_value_review.md`.

## 61. B67: the second mask8 family and a shared refinement kernel

B67 completes the feeder97 gate while retaining all ten B66 feeder45
endpoint pieces. It uses the same B65 target-conditioned parent preimages,
sealed whole-target/domain and origin-cover premises, and unchanged B63
scientific source. The detached orchestration is centralized in
`artifacts/research/c6_exact_history_refinement.py`: complete original-arc
pullback, nodal word/guard checks, admitted local-cover reuse and exact-union
comparison. Production remains the owner of DBM geometry. Historical B65/B66
scripts and certificates remain immutable; the new helper does not admit a
new cover or supply a new physical rule.

### Complete incoming histories and reuse of known restrictions

All 5,15,2,17 incoming arcs at parent vertices21,39,66,87 are evaluated.
Parent and surviving-source root intersections are empty. Of **39
candidates, 32 are empty and seven survive**:

| Parent | Incoming arcs | Source vertices | Complete nodal word |
|---|---|---|---|
| 21 | 2011, 2023, 2071, 2329 | 125, 130, 136, 151 | `16 -> 26 -> 48 -> 8` |
| 39 | none | none | no surviving piece |
| 66 | 2441 | 155 | `24 -> 26 -> 48 -> 8` |
| 87 | 4994, 5570 | 331, 458 | `26 -> 26 -> 48 -> 8` |

Every displayed return is one nodal step. In particular, the repeated26
row is a guarded translation of carried coordinates, not a stationary
carried state. Exact parent-minus-image differences have11,1,2,11 pieces;
the second is the entire former parent39 piece. The selected-family
endpoint union strictly shrinks, with residual counts11,4,1,1 against its
four B65 endpoint pieces.

Five surviving sources already belong to B64's admitted local federation.
Every existing piece at each such source is tested; exterior sources331
and458 keep their preimages unchanged:

| Source | Existing local pieces tested | Nonempty local piece IDs |
|---|---:|---|
| 125 | 1 | 46 |
| 130 | 4 | 32, 47 |
| 136 | 3 | 16, 49, 115 |
| 151 | 19 | 6, 17, 33, 51, 104 |
| 155 | 22 | none |

Thus **49 intersections leave11 local pieces and two exterior pieces**.
Source155's entire conditioned preimage disappears; this also eliminates
the last alternative under B65 parent66, using existing evidence without
another predecessor layer. The remaining family has13 endpoint pieces.
The reuse-only exact differences against the seven preceding endpoint
pieces have0,1,15,27,4,0,0 components.

### Complete-observation gain, without a hull gain

Restoring the ten unchanged feeder45 pieces gives23 endpoint pieces in
place of the14 retained after B66. The complete necessary mask8 endpoint
union is strictly smaller. Subtracting the new union from the old pieces,
ordered as ten feeder45 pieces followed by four feeder97 pieces, gives
ten zero counts followed by40,38,7,7. The whole-observation DBM hull is
**unchanged**. More pieces encode fewer possibilities here; a hull-only
metric would miss the gain for the complete observation.

Charged producer work is5,495 operations: four parent-root checks,78
candidate intersections, seven surviving-source-root checks,326 parent
difference operations,509 selected-family difference operations,49
existing-piece intersections,1,151 reuse difference operations and3,371
whole-observation difference operations. Enumeration, hash/source binding,
nodal word/subset audits, independent replay and compatibility controls are
separate. No origin trajectory or additional prehistory layer is executed.

The retained report is `artifacts/research/c6_b67_feeder97_gain.json`,
produced by `c6_b67_feeder97_gain.py`, with independent arithmetic in
`validate_b67_feeder97_gain.py` and its matching `.validation.json`.
The final provenance, documentation and next-gate checks are sealed by
`b67_final_validation.json`. Coverage remains **41/56 excluded, 15 pending**.
The complete target and both feeder families remain nonempty. Eliminating
parents39/66 concerns these target-conditioned histories, not two further
whole exit slabs. Neither finer RN-coordinate feasibility, actual origin
reachability, indefinite C6 boundedness nor future runtime is established.

<a id="smaller-observations-complete-histories-one-shared-next-gate"></a>
### Smaller observations and complete histories

Exact pairwise endpoint inclusion reduces the23-piece observation union to19
DBMs. With zero-based indices in the retained complete-observation order,
the containment witnesses are `2 -> 1`, `7 -> 5`, `21 -> 13`, `22 -> 14`.
No equality duplicates occur. All23 history records remain: containing the
endpoint of a different path does not authorize substituting that path's
preimage. In particular, indices21/22 are the exterior331/458 alternatives;
they cannot be discarded when their containing local-history alternatives
are subsequently refined.

Translate each surviving local intersection backward along its already
certified B64 prefix, preserving the full path, exact endpoint and source
cover. This produces23 pieces at22 exterior histories with no new
predecessor layer. Only source41 is shared, at frontier indices9 and13:

```text
40 -> 18 -> 48 -> 8       (feeder45 alternative)
40 -> 16 -> 26 -> 48 -> 8 (feeder97 alternative)
```

Their source-coordinate DBMs are disjoint. They must remain alternatives,
not be intersected. Both use the same12 original incoming arc identities,
so a bounded joint gate can cache12 source-cover/guard intersections and
perform24 distinct target-conditioned pullbacks. Including two parent-root
checks gives38 base intersections rather than50; child-root checks, union
differences, provenance and validation remain separate. The other21 paths
must remain in the final whole-observation comparison.

This gate has not run. Expanding the entire frontier would require308
branch/arc tests across296 distinct source/arc guards; that broader search
is deferred. The exact normalization, containment witnesses and complete
incoming lists are retained in `b67_final_validation.json`; rationale and
the full source table are in `artifacts/research/c6_b67_next_gate_review.md`.
This is a shared computation over already derived nodal histories, not a
new physical memory assumption or a symmetry quotient.
