# Coupling participation between connected organizations

This chapter owns the bounded explicit Coupling (UM) request between members
of two supplied connected organizations and its continued operator response.
The [structural operator owner](STRUCTURAL_OPERATORS.md#coupling-potential-pair)
retains the isolated-partner map and ordinary functional-link policy. The
[research program](RESEARCH_PROGRAM.md#research-current-question) selects the
current implementation question. A connected component is support, not by
itself a proved collective identity or autonomous organization.

<a id="connected-coupling-request"></a>
## A mandatory connected participant

Ordinary UM already can select a connected nonneighbor as a functional-link
candidate. Its possible new edge does not put that candidate into the same
invocation's phase consensus and capacity/pressure adjustment. The explicit
`potential_partner` interface supplies mandatory participation, originally for
an isolated partner. The additional opt-in
`potential_scope="disjoint_components"` extends that request to two distinct
complete connected components, each containing at least two node identities.
Here complete means the whole component, not a clique.

This domain uses undirected non-multigraphs. A self-loop alone is not a
two-member component; loops within a larger component remain support.
Zero-weight edges still count as support. Both supplied endpoints must exist,
the new relation must be absent and functional links enabled. Directed graphs,
parallel-edge graphs and an additional link inside one existing component are
outside this scope. Default `potential_scope="isolated"` retains the earlier
isolated-partner domain, including a target with existing neighbors. No
temporary deletion, provisional edge or new operator realizes the extension.

For a request at A with partner B, let K be A's ordinary compatible neighbors
in their retained order followed by B. The existing shared UM numerical map
acts on this augmented set. In the bidirectional branch its circular consensus
uses the phases of A and K. The target and those participants move toward that
consensus with the configured phase-push coefficient. In the other branch,
only A moves toward the consensus of K. Capacity synchronization and pressure
relief belong to A. The existing post-movement affinity determines the weight
of the one requested edge, which must be admitted or the request rejects.

Only A is an operator target. Its own live grammar and activity policy govern
execution and it acquires one actual UM occurrence. B and the old neighbors
retain their own histories and counters; secondary phase writes are not
fabricated UM events. The input identities, occasion and coefficients remain
supplied. This does not derive autonomous access, a mutual decision procedure
or a collective occurrence law.

<a id="connected-coupling-collateral-boundary"></a>
## Why the existing participant check is insufficient

The new partner has existing relations that can be damaged by its phase
adjustment. The same applies to A's old neighbors when UM moves their phases.
Checking only A against K does not protect those secondary relations.

Consider old edges A--a and B--b, with phases
`(theta_A,theta_a,theta_B,theta_b)=(0,0,1,5/2)`. Initial old-edge gaps are
0 and 3/2; the proposed A--B gap is 1. All pass the usual pi/2 bound.
For bidirectional phase push 1/2, the shared participant consensus is

```
c = atan2(sin(1), 2+cos(1)),       0 < c < 1/2.
theta_A' = theta_a' = c/2,
theta_B' = (1+c)/2,               theta_b' = 5/2.
```

The bound follows from
`sin(1)/(2+cos(1)) < sin(1)/(1+cos(1)) = tan(1/2)` on the monotone atan
branch. A--a remains aligned and A--B has gap 1/2, so the participant-only
check passes. But B'--b has gap greater than 7/4, exceeding pi/2. The proposed
action would invalidate a previously admitted old relation. Merely removing
the partner-isolation restriction therefore does not justify the extension.

Let W contain every node the proposal may write a phase to: A alone in the
unidirectional branch, otherwise A and K. A numerical zero displacement does
not remove a node from this declared write set. The extended request admits
finite phase and the effective shared UM U3 limit on every old edge incident
to W and on the requested relation, both initially and after the represented
phase proposal. Other endpoints are boundary reads; they are not extra
consensus inputs. Unrelated distant edges and phases are not consumed by this
local check. This is a bounded preservation condition, not global U3 validity.

After the usual postchecks and any separately supplied callback, the live
boundary must still satisfy the original limit (and any stricter live limit)
for the original W. Changed flags or a widened policy cannot erase that
obligation. Existing node identities, support and edge attributes remain
preserved, with exactly the one admitted new edge. A failure rolls back the
whole graph-owned invocation. External callback effects are outside that
transaction contract. Pure numerical predictions below use no callback.

<a id="connected-coupling-response"></a>
## Continued response distinguishes an edge from Coupling

The [finite continuation controls](../tests/operators/test_connected_organization_continuation.py)
prepare two declared pairs A--a and B--b. Actual Coherence (IL), followed by
retained Contraction (NUL) on each pair, supplies the following state without
copying successful history or pooling context:

| Quantity | A | a | B | b |
| --- | --- | --- | --- | --- |
| EPI | 1/4 | 1/8 | 1/4 | 1/4 |
| Capacity | 1/2 | 1/2 | 1 | 1 |
| Pressure | 3/16 | 0 | 0 | 0 |
| Phase | 0 | 0 | 0 | 0 |
| Own word | IL,NUL | IL,NUL | IL,NUL | IL,NUL |

Initial forms are `(1/2,1/4,1/2,1/2)`, capacities `(1,1,2,2)` and A pressure
3/16. IL pressure retention is 1/2 with zero phase-locking coefficient; NUL
scale 1/2 and its shared reciprocal restore A pressure to 3/16. The explicit
default densification alias is removed before that scale is resolved. No
activity floor or reception ceiling is relaxed. The clock remains zero.

The three matched arms use actual requested UM, an explicitly noncanonical
support-only comparator adding the same unit-weight A--B edge, and no contact.
The comparator supplies no operator or grammatical credit. In the UM arm the
participant set is `(a,B)`. At zero phases, equal endpoint form and Si=0 the
existing affinity gives weight 1. With capacity synchronization 1/4 and pressure
reduction 1/4, the exact target changes are

```
nu_A' = 1/2 + (1/4)*(3/4 - 1/2) = 9/16,
p_A'  = (3/16)*(1 - 1/4)         = 9/64.
```

Forms and phases stay unchanged. The other nodes acquire neither capacity nor
pressure writes nor a UM occurrence. Both comparison arms keep A capacity 1/2
and pressure 3/16. Actual UM and the support-only arm have identical edge data.

Next, actual instantaneous Emission (AL) at B adds 1/8 in every arm, producing
B form 3/8 with its own context. Actual selected-source Reception (EN) at A,
sole source B and mixing 1/2, is admitted after UM: pressure 9/64 is below the
unchanged EN ceiling 0.15. It gives A form 5/16. In the support-only arm, the
same realized input cannot supply the missing pressure relief: pressure 3/16
rejects EN. The no-contact arm lacks both realized input and an admitted
pressure. Rejections preserve their pre-attempt states and append no EN.

Finally, actual Resonance (RA) at a reads its sole neighbor A with blend 1/2:

| Arm | A after EN attempt | a after RA | A capacity | A pressure |
| --- | --- | --- | --- | --- |
| Actual UM | 5/16 | 7/32 | 9/16 | 9/64 |
| Support only | 1/4 | 3/16 | 1/2 | 3/16 |
| No contact | 1/4 | 3/16 | 1/2 | 3/16 |

The response difference is 1/32. These dyadic values are exactly represented
in the declared branch; they are not an arbitrary binary64 exactness theorem.
RA also changes a's capacity to 9/16. The actual final target word is
IL,NUL,UM,EN versus IL,NUL in the controls. The other words are a: IL,NUL,RA;
B: IL,NUL,AL; b: IL,NUL. The admitted link and the constitutive pressure/capacity
effects together permit the stated continuation. No increased coherence score
or matching edge count substitutes for that causal distinction.

The [boundary controls](../tests/operators/test_connected_organization_coupling.py)
cover domain, collateral U3, source admission, execution paths and rollback.
These results concern one requested action and finite continuation. Persistent
joint identity, reciprocal action selection, overlapping organizations, richer
form and cross-scale recursion remain separate obligations. The
[API contract](../docs/API_CONTRACTS.md#connected-organization-coupling) specifies
the executable request without assigning a physical interpretation.

<a id="ordinary-coupling-relation-preservation"></a>
## Preserving existing relations during ordinary Coupling

The [reciprocal return](REMESH_STRUCTURE_AND_MEMORY.md#reciprocal-maintenance-boundary)
shows that ordinary UM can improve its target-participant phase differences
while moving a secondary participant across an external relation's U3 bound.
The optional `preserve_relations=True` request addresses that specific
composition gap through the existing UM proposal, merger and atomic stage.
It changes the admission predicate, not the numerical phase, capacity or
pressure law. Omitted or false requests retain the legacy action.
The enabled atomic request also retains the existing transaction's metadata
admission domain; preserving edge data does not admit arbitrary Python objects.

The initial supported domain is ordinary canonical UM on an undirected
non-multigraph, with nonempty targets and functional links disabled. Existing
self-loops and zero-weight edges count as support. Multiple targets share the
existing snapshot-rank-ordered displacement reducer. Potential-partner
creation and alternative potential scopes are separate contracts. Unsupported
domains, invalid Boolean requests and substituted execution reject rather
than silently supplying a weaker operation. The configured grammar remains
authoritative; the flag does not confer grammar credit or choose occurrence.

### The sufficient state and preservation predicate

Let `(V,E)` be the invocation-entry support and `L0` its effective UM phase
limit. Retain node identities, old edge attributes, phase inputs, flags and
this limit before preflight. Readmit the original source before committing,
so a changed validation input cannot become a later successful baseline.

From the actual target proposals define

```
W = union of every proposed phase-write node,
B = {(u,v) in E : u in W or v in W}.
```

W includes zero displacements and writes that cancel when merged. In the
bidirectional branch it includes the actual compatible secondary recipients;
in the one-way branch only target writers enter W. B includes their complete
incident support, even an old neighbor filtered from the numerical consensus.
Boundary-only endpoints remain observations for admission, not new inputs to
the UM phase consensus or capacity mean. Unrelated finite incompatible edges
outside B need not pass this local predicate. The ordinary kernel still
consumes all snapshot phases; malformed distant phase values are not promised
to be ignored.

The enabled action requires finite authoritative phases and wrapped gap at
most L0 on every relation of B initially and after the **joint merged** phase
proposal. Postchecks, monitors and optional refresh then execute through the
existing transaction. Before returning, the same original B must satisfy
`min(L0,Llive)`, where Llive is the readmitted effective current UM limit.
Support, edge attributes and node identities must remain unchanged. A widened
policy cannot relax L0; a tighter live policy is used and may pass or reject.
Changing bidirectionality cannot erase the already bound write set.

Consequently, successful return entails preservation of the admitted incident
relations at entry, proposal and returned endpoint. This is a checked finite
execution contract, not a theorem about every intermediate callback state,
continuous interpolation or later event. It supplies no capacity conservation,
bounded maintenance or self-organized selection law. Only actual targets gain
their normal own UM occurrence; secondary phase participation grants none.
The boundary contract does not make arbitrary history or metadata writes by
callbacks immutable. Any failure restores graph-owned invocation state;
external callback effects and earlier completed invocations remain separate.

### Two discriminating geometries

On the chain A--R--W, supply phases `(1/4,0,-pi/2+1/32)` and bidirectional
push 1/2 at A. The two-participant consensus is 1/8, giving A phase 3/16
and R phase 1/16. The initial external gap `pi/2-1/32` becomes
`pi/2+1/32`. Legacy ordinary UM can commit its local map. The preservation
request instead rejects the joint proposal and restores the invocation.
This three-node admission source is independent of the earlier lifecycle
experiment; no earlier trajectory needs replaying.

The accepted multiple-target case also matters. On A--R--B with R--W,
set A=1/2, B=-1/2, R=0 and W=-pi/2+1/16. Simultaneous requests at A and B
propose R displacements +1/8 and -1/8. The actual merger cancels them, so R
stays at zero and the external relation remains valid. A/B move to +/-3/8.
Rejecting each local proposal's external endpoint separately would discard
this valid joint action. The final predicate therefore belongs after the
existing merger, while initial admission still uses the original support.

The [preservation controls](../tests/operators/test_coupling_relation_preservation.py)
exercise these geometries, actual own context, public/dispatcher routing,
filtered and zero-displacement boundaries, loops/zero weights, unsupported
domains and atomic rejection of later phase/policy/support changes. Numerical
comparisons retain their finite represented-arithmetic scope. The
[API contract](../docs/API_CONTRACTS.md#ordinary-coupling-relation-preservation)
owns the callable interface and its limits.

### Evaluated scope

The 24 new finite cases and the selected ordinary, potential-partner,
connected-organization and public-word regressions resolve 292 distinct
controls. The first invocation passed 289 and failed the three positive-route
variants before UM: their fixture embedded a NetworkX graph in edge metadata,
outside the existing transaction's admitted copy domain. A separately reviewed
fixture correction replaced that object with supported cyclic dictionaries;
the three affected variants then passed. Production code and the other 289
cases were unchanged. The retained record therefore contains 295 executions,
including the three initial failures, rather than one wholly passing run.
The local records `controls-v1-result.json` and `controls-v2-result.json` in
`artifacts/coupling-boundary-20261103/` retain both attempts; their frozen
sources and independent reviews remain alongside them. These local records
are not bundled into the documentation website.

This closes the optional incident-relation contract for the declared UM
domain. It does not resolve the reciprocal return's capacity growth or prove
bounded complete maintenance. Amplification, contraction and any renewal must
compose through their actual form, capacity, pressure, phase and memory rows;
exact cancellation of capacity factors alone cannot establish that result.
