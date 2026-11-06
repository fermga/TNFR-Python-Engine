# Contraction, retained synthesis and response sufficiency

This chapter owns the actual Contraction (NUL)/Reception (EN) fragment,
its concentration law, amplification/contraction composition and the
information needed for continued interaction.
The [operator specification](STRUCTURAL_OPERATORS.md#retained-member-scale-actions)
owns the member/port interface and instantaneous scalar-reduction obstruction.
The [research program](RESEARCH_PROGRAM.md#research-current-question) selects
the next implementation question.

<a id="retained-synthesis-concentration"></a>
## Retained concentration through actual NUL and EN

Historical NUL synthesis motivates integration toward a core while retaining
diversity and memory. Neither a scalar average nor a renamed coordinate chart
executes that function. The [retained concentration interface](../src/tnfr/operators/retained_synthesis.py)
defines a bounded, supplied operator fragment: actual simultaneous NUL on an
ordered pair `(core, member)`, then actual EN at the core with that member as
its sole explicitly selected input. NUL alone is the matched comparison.
This uses existing actions and factors, not an unrecorded change to NUL or a
new token. The current domains remain `0<f<1` for `NUL_scale` and `0<m<=1`
for `EN_mix`; an EN event with zero mixing is not an admitted control.

`capture_retained_pair` retains the owner-bound member/port observation and
adds exact rational `form_mean` q and `form_contrast` d from its admitted
binary64 forms. The ordered roles need no inferred ancestry. The real-value
identities `xp=q+d`, `xc=q-d` do not make q,d a complete state: capacities,
pressure, phases, kinds, individual contexts and ports remain in their
original records. The additional `reception_kinds` retains the strings
actually read by EN: an absent kind gives `''`, whereas an explicit `None`
gives `'None'`. This differs from the generic retained member view, which
normalizes both to `''`; execution re-admits the extra channel against live
aliases without changing EN's existing semantics. Two floats cannot always
store the exact chart. With `xp` the least positive subnormal and `xc=0`, both halves are
nonzero exact rationals but round to zero as floats. Original records also
retain signed-zero association that a rational zero cannot encode.

`execute_retained_synthesis_stage` re-admits the pair and requires a realized
member-to-core arc or their undirected edge. A core-to-member arc alone is
insufficient; the fragment neither inserts a reverse edge nor invokes UM.
Runtime uses the actual nodes and kernels, never forms reconstructed from a
rounded chart. The core's selected EN omits external donors from this one
read without removing their ports. All own grammar and strict policies still
apply. No callback, source override or replacement glyph is accepted.

One graph-owned transaction covers both substages and their observations.
Support/edge data, phases and nonmember state remain unchanged; core EN also
preserves the member's post-NUL state. Each actual event updates only its own
targets' context. The result retains `before`, `contracted`, `after`, the real
contraction result and the real reception result. These are associated state
and execution summaries, not a checkpoint or authenticated invocation proof.
No hierarchy, member deletion, autonomous occurrence or elapsed flow is added.

#### Concentration and its information boundary

In the ideal same-kind scalar branch with edge-aware form scaling and inactive
projection, write `xp=q+d`, `xc=q-d`. NUL alone gives `(f*q,f*d)`; the compound
fragment instead gives

```
xp' = f*((1-m)*xp+m*xc),  xc' = f*xc
q'  = f*(q-m*d),         d'  = f*(1-m)*d.
```

Its extra core response versus NUL alone is `-2*f*m*d`. Subsequent admitted
RA at an external peer with only the core as form donor and mixing r changes
by `-2*r*f*m*d`. Thus the added internal EN has a distinct causal effect.
The absolute form contrast contracts. For nonzero forms of the same sign,
`rho=d/q` lies in `(-1,1)` and becomes `(1-m)*rho/(1-m*rho)`, also decreasing
its magnitude when nonzero. Opposite signs or zero mean do not inherit that
normalized guarantee; neither statement is a universal coherence theorem.

When edge awareness is disabled, the form factor in these equations is one,
while capacities still scale by f and stored pressures by its reciprocal.
Active projection, kind changes and represented rounding retain their actual
maps. Exact rational endpoint observations do not certify an exact binary64
matrix identity for arbitrary coefficients. Phase is held by both actions;
the ideal stored product `nu*p` is preserved by NUL, without implying its
preservation after an independent pressure refresh or across form jumps.

The ideal form determinant is `f^2*(1-m)`. At `m=1` both current forms equal
`f*xc`, erasing the previous core form from that pair. A before-record does
not make the live reset invertible or derive an internal memory law. For
`m<1` the ideal form map is invertible, but that does not invert context,
clipping or admission. Common NUL and EN commute in ideal unclipped form
coordinates; their different event orders still have different histories and
can encounter different intermediate admission. No order is uniquely derived.

Same mean is insufficient even for this fragment. `(xp,xc)=(3/4,1/4)` and
`(1/4,3/4)` share q but, at `f=1/2,m=1/4`, give core forms `5/16` and `3/16`.
An external peer can distinguish them. Sum of forms changes by
`2*(f-1)*q-2*f*m*d`; no conserved scalar resource or energy is assumed.
The member remains addressable with its own state and ports, but preservation
of every past form contrast or historical reabsorption has not been proved.

The [interface controls](../tests/operators/test_retained_synthesis.py)
cover association, exact chart, direction, protected state and rollback.
The [finite continuation controls](../tests/operators/test_retained_synthesis_continuation.py)
use actual derived VAL and UM preparation, NUL-only comparison and subsequent
RA. On the supplied dyadic source `(3/4,1/4)`, two compound actions with
`f=1/2,m=1/4` and explicit `DNFR_RECEPTION_MAX=3/4` give `(5/16,1/8)` then
`(17/128,1/16)`. The core pressure rises from `1/8` to `1/4` then `1/2`;
the unchanged default EN ceiling instead rejects the first post-NUL EN.
The selected ceiling is a supplied policy, not a concentration theorem.
In general, enabled strict EN admission requires the actual post-NUL signed
pressure below that ceiling (ideally `p_core/f < DNFR_RECEPTION_MAX`), besides
its other conditions. This is not an absolute-pressure bound.
A third attempt in the admitted control fails
the unchanged default NUL member-EPI threshold `1/10` and restores the source.
This is policy rejection, not nonexistence of the member. Neither NUL nor EN
discharges the core's preceding VAL debt. These finite interventions establish
an elementary concentration mechanism, not persistent organization or complete
canonical synthesis.

<a id="nul-future-response"></a>
## Concentration and a response-sufficient representation

The historical TNFR.pdf pp109-112 describes reabsorption while retaining the
constituents' diversity and active memory. Its unspecified `merge_structures`
does not define a deletion rule, conservation identity or sufficient successor
state. Three operations must be distinguished: actual concentration can change
a response; encapsulation can retain an interior behind an interface; a lossy
reduction discards information. Calling each of these synthesis does not make
their contracts interchangeable. The existing retained fragment implements the
first and retains state needed to investigate the second.

For a declared family of partial actions T_a and observation pi, exact effective
replacement requires the permitted action domains and observed successors to
factor through pi. In particular, `pi(s)=pi(t)` must imply the same admissibility
and, when admitted, `pi(T_a(s))=pi(T_a(t))` for each allowed action. Iteration then
preserves the corresponding observations along finite allowed words. An
external-output version must specify its boundary actions and observations at
every prefix. One equal output is insufficient. This is the
[contextual replacement criterion](STRUCTURAL_OPERATORS.md#operator-contextual-rewriting)
specialized to the intended synthesis, not an engine certificate or a proof
that the present pair observation is a minimal complete state.

### Equal first output can conceal a later response difference

Fix the undirected path `peer--core--child`, same-kind positive scalar forms,
compatible phases, the actual policies and each action's grammar admission.
Let x,c,v be the forms after the concentration fragment. In the unclipped
restriction, execute Resonance (RA) at the peer with factor r, ordinary Reception
(EN) at the core with factor m, and another RA at the peer:

```text
v1 = (1-r)*v + r*x
x1 = (1-m)*x + (m/2)*(v1+c)
v2 = (1-r)*v1 + r*x1.
```

For two such states with common x,v, a retained-child difference dc leaves the
first boundary output v1 identical but changes v2 by `r*m*dc/2`. Therefore the
current exported core form and even one exact boundary response cannot replace
the child for this future word. This temporal obstruction does not repeat the
instantaneous rank-two result, prohibit a restricted quotient, or exclude a
representation that retains sufficient internal state or memory.

The [finite response controls](../tests/operators/test_nul_synthesis_response.py)
use declared initial pairs A=(3/4,1/4) and B=(1/2,1/2), peer 3/8, and actual
NUL/selected EN with f=m=1/2. Both initial means are 1/2, but the complete states
differ. The fragment gives (core,child)=(1/4,1/8) versus (1/4,1/4). With future
r=m=1/2, the independent exact predictions are:

| Boundary word prefix | A | B |
| --- | --- | --- |
| First peer RA, v1 | 5/16 | 5/16 |
| Subsequent core EN, x1 | 15/64 | 17/64 |
| Second peer RA, v2 | 35/128 | 37/128 |

The final difference is 1/64. These controls execute the actual owners with
strict admission; the forms are not advanced by a replacement matrix. Other
RA effects, including capacity, retain the real implementation. The equations
are a conditional form-channel identity; exact dyadic control arithmetic does
not establish exact equality for arbitrary binary64 policies.

<a id="nul-acquired-child-continuation"></a>
### An actually born child can enter the concentration fragment

The same response fixture also connects existing operators end to end. Actual
IL/OZ/AL produces the physical source history for eligible THOL birth; the
[live association consumer](STRUCTURAL_OPERATORS.md#coupling-birth-association-access)
obtains the isolated child for actual UM. The source and its chronology are
owned by the [birth continuation](THOL_PRESSURE_FEEDBACK.md#thol-state-derived-access).
The child has its own fresh context, not the parent's measured history.

Three actual parent IL actions reduce pressure from 9/16 to 9/128. NUL at f=1/2
gives 9/64, below the unchanged default EN ceiling 0.15. The pre-NUL child form
15/128 also exceeds the unchanged NUL form floor 0.1. No lowered threshold or
pressure assignment forces admission. Selected core EN uses m=1/2. The actual
NUL-only comparison retains the same source, support and preparation:

| Observation | NUL only | NUL then selected EN |
| --- | --- | --- |
| Core after fragment | 25/128 | 65/512 |
| Child after fragment | 15/256 | 15/256 |
| Peer after RA | 73/256 | 257/1024 |
| Core after ordinary EN | 47/256 | 577/4096 |
| Peer after second RA | 15/64 | 1605/8192 |

The initial peer form is 3/8 and subsequent r=m=1/2. These are distinct admitted
responses under the same declared continuation. NUL's actual capacity and
reciprocal-pressure changes, retained ports, phase, child identity and own
context remain part of the executed state. Instantaneous actions do not add
elapsed time or current physical samples; old samples can be stale after form
jumps. Source preparation, invocation occasions and coefficients are supplied.
This integration establishes neither persistent concentration nor autonomous
occurrence, but extends synthesis participation to an actually born child.

<a id="nul-own-memory"></a>
### Numerical agreement does not erase own grammatical memory

The [independent memory controls](../tests/operators/test_nul_synthesis_memory.py)
start with a declared equal-form pair and acquire different child context
through actual Coherence (IL). Both branches first execute IL at the core;
this common preparation supplies no handler context to the child.
With both forms 1/2, capacities 1, zero pressure and phase, actual NUL(f=1/2)
and full selected EN(m=1) leave both forms 1/4 and capacities 1/2. In one matched
branch, an actual child IL at zero pressure and zero phase-locking gain retains
these primitive coordinates and support while adding its own handler context.

A subsequent actual Dissonance (OZ) request has the same numerical domain in
both branches: form 1/4 and capacity 1/2 satisfy the ordinary floors and pressure
zero satisfies its ceiling. The branch without child IL must reject U4a;
the branch with its own IL can execute. NUL's separate role as a resolver in
OZ's bounded overload policy does not make NUL an IL/THOL grammar handler.
The admitted OZ retains its real near-zero pressure rule, not a presumed zero
product. Rejection must preserve the pre-request state and must not substitute
another glyph. This is acquired grammar memory affecting a later action,
not a proof that arbitrary past form information was retained.

The useful design conclusion is to retain internal state, addressed ports and
own action context until a specified replacement preserves the requested
continuations. Neither current form agreement nor a first matching response
authorizes pooling histories or removing constituents. A complete encapsulated
NFR, its autonomous action selection and any justified lossy reduction remain
separate implementation and mathematical obligations.

<a id="retained-addressed-input"></a>
## Addressed external input with retained interior state

The [retained Reception consumer](../src/tnfr/operators/retained_reception.py)
adds an executable input boundary to the retained organization. Its input
addresses are derived from member order and each member's incoming neighbor
order, excluding every internal neighbor. A caller selects an ordinal and an
invocation occasion; the consumer resolves the actual member and external
source, then executes the existing Reception (EN) stage. It accepts no separate
target/source pair. An address is an interface to an actual constituent, not
a macro-node coordinate, a new operator or an inferred occurrence law.

Direction and state ownership matter. A reverse-only directed arc supplies no
input, and parallel edges retain their attributes but represent one unique
donor. The owner-bound observation must still match all member state, own
context and incident support when consumed. A changed member or incident edge
requires a new capture. External donor form and kind are deliberately live:
the shared EN snapshot reads them at invocation. A passive address table is
neither live authorization nor a cached input payload. The returned after
observation can support another invocation after fresh admission; a consumed
before observation becomes stale when the member's own context changes.

Only the addressed member receives EN. Its own grammar and the actual EN
admission policy remain authoritative; other members supply no grammar credit.
The operation preserves every other node, support, phase, target capacity and
target pressure. Target form/kind, its own event bookkeeping and ordinary graph
telemetry may change. One graph-owned transaction covers the action and its
postconditions. Later Resonance (RA) is a separately admitted action, outside
that transaction. The [API contract](../docs/API_CONTRACTS.md#retained-addressed-input)
owns exact options, rejection and association boundaries.

### Independent interior and live-input response

The [four finite controls](../tests/operators/test_retained_input_continuation.py)
use a declared directed graph with retained members `(core, child)`, internal
arcs in both directions, `donor -> child` and `peer -> core`. Address zero is
`(core, peer)` and address one is `(child, donor)`. The label `child` here is
a supplied member role, not a newly claimed THOL birth. Actual IL acquires
core, child and peer context. Actual NUL at f=1/2 followed by selected core EN
at m=1/2 prepares initial form pairs A=(3/4,1/4) and B=(1/2,1/2) into
`(x,c)=(1/4,1/8)` and `(1/4,1/4)`. Capacities become 1/2; pressure and phase
remain zero. The peer form is v=3/8 and the external donor begins at d=1/4.

After capture, a matched donor-only intervention executes instantaneous
Emission (AL) with boost 1/8, producing d=3/8. This changes no member state
and leaves the retained input address admissible. It supplies no elapsed time,
flow-derived rate or acceleration sample. A finite-duration episode would
have a different history boundary and cannot silently replace this jump.

Addressed EN at the child uses only the live donor, despite the child's
additional internal incoming neighbor. Actual RA then acts at the core and
finally at the peer, reading their respective outgoing singleton neighbors.
With the declared same-kind scalar domain, inactive projection and blends
m=r=s=1/2, the ideal affine maps are

```
c_new = (1-m)*c + m*d
x_new = (1-r)*x + r*c_new
v_new = (1-s)*v + s*x_new.
```

The following dyadic controls are exactly represented; the formulas do not
assert exact real arithmetic for arbitrary binary64 inputs.

| Interior | Donor AL | Live donor | Child after EN | Core after RA | Peer after RA |
| --- | --- | --- | --- | --- | --- |
| A | absent | 1/4 | 3/16 | 7/32 | 19/64 |
| B | absent | 1/4 | 1/4 | 1/4 | 5/16 |
| A | present | 3/8 | 1/4 | 1/4 | 5/16 |
| B | present | 3/8 | 5/16 | 9/32 | 21/64 |

Subtracting these affine rows gives
`Delta(v_new)=s*r*(1-m)*Delta(c)` at fixed x,v,d, and
`Delta(v_new)=s*r*m*Delta(d)` at fixed x,v,c. Each selected difference is
1/64. Thus equal exported core form does not determine this response, while
freezing the donor payload would erase a distinct input effect. This is a
conditional response discriminator, not a universal minimal-state theorem.

The tests retain actual RA capacity amplification: final core capacity 9/16,
peer capacity 9/8 and child capacity 1/2. Histories remain individual: the core
has IL,NUL,EN,RA; the child IL,NUL,EN; the peer IL,RA; the donor has no action
or its own AL. No copied history, pooled debt or temporal sample supplies
admission. The [interface controls](../tests/operators/test_retained_input_ports.py)
separately cover addresses, live kind/form, stale records, policies and
rollback. This specifies the bounded input-routing function, while autonomous
collective identity, complete synthesis and cross-scale replacement remain open.

<a id="amplification-contraction-maintenance"></a>
## Amplification, contraction and participant-supported maintenance

The [reciprocal return](REMESH_STRUCTURE_AND_MEMORY.md#reciprocal-maintenance-boundary)
exposes capacity growth under repeated Resonance (RA). Contraction (NUL)
modifies more than capacity, so a compensating factor must be assessed through
the complete action. The existing neighbor, pointwise and Coupling (UM) stages
already compose these operators with live grammar; no additional operator or
replacement evolution law is needed for the following conditional analysis.

### Complete RA--NUL rows and the neutral-capacity obstruction

Consider a same-kind positive scalar target `(x,c,p,theta)` with one compatible
neighbor of form u and phase phi. There is no inter-event flow or pressure
refresh. All stages use their actual strict preconditions and own grammar,
with substitution rejected. Assume active RA amplification (`u > 1e-9`),
mixing `0<m<=1`, amplification `a>0`, phase coefficient `0<beta<=1`,
NUL factor `0<f<1`, edge-aware form scaling and inactive hard clipping.
On a circular chart with `d=theta-phi` strictly inside the U3 sector,
one actual RA followed by NUL has ideal numerical rows

```
x' = f*((1-m)*x + m*u),   c' = q*c,   q = f*(1+a),
p' = p/f,                d' = (1-beta)*d.
```

Support and the neighbor's primitive state remain unchanged in this fragment.
Only the target records its own RA,NUL events. Each prefix must retain the
active trigger, sign/kind, representability, configured phase and strict
capacity/form/pressure/density bounds. NUL's minimum capacity is strict;
its form threshold is signed, whereas RA's uses magnitude. Passing an entry
guard does not establish admission of a later request.

For a fixed neighbor, the ideal form row contracts toward
`f*m*u/(1-f*(1-m))`. This does not control capacity or pressure. Conditional on
remaining in this branch, `c_k=q**k*c_0`: q below one loses any uniform positive
lower bound, q above one loses any finite upper bound, and q equal to one
retains every initial capacity without attraction. Independently,
`p_k=p_0/f**k`. A nonzero pressure therefore leaves any fixed pressure bound;
the exact plane p=0 is not robust against pressure perturbations. In particular,
the neutral choice `f=1/(1+a)` preserves capacity while amplifying pressure.
The stored nodal product across the fragment is `(1+a)*c*p`, even though NUL
alone preserves that product ideally. These are obstructions to the stated
maintenance domain, not claims about execution after a guard has rejected.

The same capacity/pressure obstruction survives disabling edge-aware form
scaling; only the form factor f disappears. Active clipping, multiple or
changing neighbors, trigger switching and refreshed pressure need their actual
maps. Neither a positive form fixed point nor a zero-pressure fixture closes
those alternative models.

### Renewal through a live participant and pressure relief

Take a two-node undirected graph and apply the supplied word RA,NUL,UM,IL
only to the receiver. The neighbor has live form u=3/4 and capacity M=1;
those coordinates remain unchanged because these actual actions do not write
them. Bidirectional UM does move its phase. No state is reset between words,
no resource is debited from the neighbor by UM, and no conservation law is
inferred from capacity blending. The neighbor and its individual history are
retained parts of the state, not values reconstructed from the receiver.

Use the admitted supplied factors
`m=1/2`, `a=1/8`, `f=1/2`, `beta=1/2`, UM phase push 1/2,
UM capacity blend 1/4 and pressure reduction 1/4, and IL pressure retention
1/2 with phase-locking coefficient 1/2. Functional links are disabled;
UM requests the [incident-relation contract](COUPLING_COMPOSITION.md#ordinary-coupling-relation-preservation).
The constants select a testable member of the operator family, not a unique
fundamental law. One complete ideal word gives

```
x'   = x/4 + 3/16,
c'   = (27/64)*c + 1/4,
d'   = d/8,
phi' = phi + d/8                  (modulo the common circular chart),
p'   = (3/4 + abs(d)/(16*pi))*p.
```

UM reads capacity after NUL. Its pressure multiplier uses the actual merged
phase gap d/4 and the implemented linear affinity `1-abs(gap)/pi`; IL then
halves that pressure and halves the residual phase gap. The reference phase
advance is a real secondary UM write, without an own UM occurrence there.

For the numerical return region
`1/4<=x<=1/2`, `1/4<=c<=1`, `abs(p)<=3/16`, `abs(d)<=1/4`,
the form image is `[1/4,5/16]`, the capacity image is `[91/256,43/64]`,
and the pressure multiplier lies between 3/4 and `3/4+1/(64*pi)<1`.
Every phase prefix decreases the sole edge gap. At NUL entry, form is at
least 1/2, capacity at least 9/32 and density at most 3/8, meeting its default
signed-form, strict-capacity and density guards. At UM entry, form is at
least 1/4 and capacity at least 9/64, above its default minima 0.05 and 0.01.
At IL entry both are strictly positive, as its default activity policy
requires; pressure is below its advisory threshold. RA entry also meets its
default form, capacity and pressure bounds. No global preservation result
for RA on a larger graph follows from this two-node proof.

The ideal rows attract toward x=1/4, c=16/37, p=0 and relative phase zero.
Both phases converge: the neighbor's total chart displacement is d_initial/7.
The strict pressure and capacity contraction inequalities tolerate small factor
changes, with a nearby form region and every prefix readmitted; maintenance
is not based on an exact inverse-capacity equality.
This is a conditional numerical return with continuing own context, not a fixed
point of the extended history state. The word, support and occurrence schedule
are supplied; preparation and infinite represented-arithmetic execution remain
separate from the ideal return argument.

Preparation uses one actual simultaneous IL on both nodes, with phase
coefficient zero and pressure retention 1/2. RA,NUL,UM retain zero U2 debt;
final IL supplies its normal stabilizing context. Each target records only
its own occurrences, with growing event count and the ordinary bounded history
suffix. These are incrementally admitted continuation sequences. Terminal IL
does not supply a standalone whole-word U1 closure; stage admission does not
assert that separate whole-word property.

### The source and closed-population boundaries

If actual NUL first contracts the neighbor, its form and capacity become
`u=3/8`, `M=1/2`, changing the next receiver return. UM must use that live M,
and RA that live u. From `x=3/8,c=1/2`, the uncontracted source gives
`x'=9/32,c'=59/128`; the contracted source gives `x'=3/16,c'=43/128`.
These are distinct complete preparations, not a later change to a frozen law.

If instead every node in a finite closed population receives the same active
RA gain and NUL factor before UM, all capacities first acquire q. UM capacity
proposals are convex combinations of those post-NUL capacities, and IL does
not change them. Thus `max(c_after)<=q*max(c_before)`. For q=9/16, repeated
complete rounds cannot retain a uniformly positive capacity scale throughout
that population. In the symmetric two-node case each capacity is exactly
`q**k*c_0` ideally; the form also halves each round in this branch. This is a
law-specific obstruction, not a theorem excluding all closed TNFR organizations.
Different roles, active branches, factors, ordering or state-dependent
occurrence require a new complete return and admission argument.

The reference-supported return therefore identifies useful participation of
four actual operators while exposing the missing closed-system dependency.
It establishes neither autonomous occurrence nor acquired collective identity,
memory-preserving synthesis, a universal balance or physical identification.

### Evaluated finite composition

The [seven composition controls](../tests/operators/test_amplification_contraction_continuation.py)
passed together with 75 selected synthetic regressions: 82 passed, with no
failures or skips. The prospective protocol, source archive, independent
reviews, log and JUnit result remain in the local directory
`artifacts/amplification-contraction-20261103/`. Production sources stayed
unchanged before and during the run; no earlier research producer was replayed.

The loss, neutral and growth cases matched their separate declared gains and
pressure preparations. Neutral capacity left pressure 3/4, so the next actual
RA rejected its default 1/2 ceiling while retaining the completed prefix.
The two renewed cases matched every predicted numerical row and own context,
including nonzero donor phase participation without borrowed UM credit.
Their receiver capacities were 59/128 then 3641/8192. Contracting the live
donor changed the one-return capacity from 59/128 to 43/128 and form from
9/32 to 3/16. Treating both equal participants instead left capacities 9/32
then 81/512: UM did not supply an undepleted reference.

The run used eight two-node preparations, 52 stage requests, 67 successful
target occurrences and one expected rejected RA. Exact dyadic comparisons
and the declared absolute 2e-14 phase/pressure tolerance passed. This finite
correspondence supports the implemented composition; the preceding ideal
invariant-region proof retains its own assumptions and does not establish
infinite binary64 execution. History and diagnostic storage continue evolving.

Future viability work must distinguish conservation, neutral bounded families
and attracting responses. A neutral family can tolerate perturbations of its
initial state without selecting one absolute scale; sensitivity to factors is
a different question. RA--NUL alone fails the nonzero-pressure requirement,
but that obstruction cannot be assigned unchanged to a composition containing
actual pressure relief. State-dependent admissibility and occurrence also
remain different: an eligible action need not be selected by the current NFR
selector, and an empty admissible set must not be concealed by a substitute.

<a id="operator-viability-and-ordered-return"></a>
## Admission, reference preservation and ordered return

The [partial-action criterion](STRUCTURAL_OPERATORS.md#collective-grammar-descent)
already includes consumed participant state, own grammar, policy and execution
path. For a declared family F of marked actions, write
`A_F(s)={a in F: s in D_a}` for actual admission and
`V_K,F(s)={a in A_F(s): T_a(s) in K}` for one-step preservation of a reference
region K. Incremental grammar admission alone is weaker than membership in
`A_F(s)`. A returned proposal alone proves neither predicate.

An ordered return instead requires every actual prefix to be admitted in its
declared execution envelope E and its terminal state to belong to K. Prefixes
may leave K. This uses the existing composite domain
`D_a intersect inverse_image(T_a,D_b)`, not a second grammar or executor.
A required response, such as a future actual Resonance (RA) reading a live
participant, is another explicit obligation. Finite reachability, existence
of an infinite continuation and success for every choice have different
quantifiers. A rejected RA does not imply that no canonical action is available.

For an observation Q to decide preservation of a marked action, admission
and the truth of `T_a(s) in K` must agree for all compared states with equal
Q. Exact response requires the stronger existing action-descent condition.
The current Default selector's Si/normalized-pressure/acceleration projection
does not generally meet this preservation requirement. Its heuristic is not
specified to certify an arbitrary reference domain. Native lag priorities and
direct-glyph execution are separate consumers; applying a Default proposal
through an atomic neighbor stage is an explicitly supplied composition.

### A complete reference family and its one-step conditions

Retain the previous two-node support, same-kind positive scalar forms, factors
and strict stage paths. Write receiver coordinates `(x,c,p,theta)`, participant
coordinates `(u,M,p_D,phi)` and wrapped gap `d=theta-phi`. Define S by

```
1/4 <= x <= 1/2,    1/4 <= c <= 1,
abs(p) <= 3/16,    abs(d) <= 1/4,
3/8 <= u <= 3/4,   1/2 <= M <= 1,   abs(p_D) <= 1/8.
```

S also retains the supplied ordered support and edge data, scalar kinds,
fixed clock, configured policies, finite selector diagnostics and the evolving
own-context family with actual prior IL and zero U2 debt. The common phase is
free. Each stage reads live state; no stored field refresh or flow is added.
Legal event counts and history suffixes evolve, so S is not a fixed complete
checkpoint. Its numerical bounds specify a reference obligation, not a
definition of NFR existence or a maximal viability kernel.

With the stated factors, all four requested stages meet their entry guards
on S under these context premises. Their ideal one-step preservation conditions
on the admitted circular branch are:

| Actual action at the receiver | Additional condition for remaining in S |
| --- | --- |
| Resonance (RA) | `x+u<=1` and `c<=8/9` |
| Contraction (NUL) | `x=1/2`, `c>=1/2` and `abs(p)<=3/32` |
| Coupling (UM), preserving incident relations | None within S |
| Coherence (IL) | None within S |

RA gives `(x+u)/2,9*c/8,p,d/2`. NUL gives `x/2,c/2,2*p,d`.
UM leaves form unchanged, blends capacity to `3*c/4+M/4`, halves the relative
phase gap and multiplies pressure by `3/4+abs(d)/(8*pi)`. Its secondary phase
write at the participant supplies no own UM occurrence. IL retains form and
capacity while halving pressure and relative phase. Substituting these rows
in the bounds proves the table. Their sign, U3, form, capacity and pressure
guards are independently met; in particular the initial NUL density is at
most 3/4 and its strict capacity floor is below 1/4.

Thus this domain has no empty all-operator action set: IL remains available
and preserves S. Empty preservation for a restricted menu, or the temporary
unavailability of a specified RA response, would be narrower claims.
The other nine roles retain their [individual contracts](STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit).
Support/type-changing actions need corresponding before/after domains;
excluding them from this fixed-support control does not make them inviable.

### Noncommuting form transport and live prefix admission

Compare the supplied words `F = RA -> NUL -> UM -> IL` and
`R = NUL -> RA -> UM -> IL` on S. Their ideal numerical rows are

```
x_F = x/4 + u/4,       x_R = x/4 + u/2,
c_F = c_R = 27*c/64 + M/4,
p_F = p_R = (3/4 + abs(d)/(16*pi))*p,
d_F = d_R = d/8,       phi_F = phi_R = phi+d/8.
```

The common-phase update is modulo `2*pi`. The form difference is `u/4`:
the live participant input enters on different
sides of the receiver's contraction. Equal remaining numerical outputs do
not identify complete successors, because their own histories retain different
orders. No inverse or reversibility claim follows.

Reverse-order RA consumes `x/2,c/2,2*p`; it requires, among its other guards,
`x/2>=RA_MIN_SOURCE_EPI`, `c/2>=RA_MIN_VF` and
`abs(p)<=RA_MAX_DISSONANCE/2`. Under the default strict thresholds these
inequalities hold throughout S. NUL must first meet its own entry conditions;
neither admission may be borrowed from the opposite order. The two words fit
the prefix envelope `1/8<=x<=3/4`, `1/8<=c<=3/2`, `abs(p)<=3/8`, with S's
remaining premises. This envelope concerns these words, not arbitrary choices.

Return to S requires `x+u>=1` for F or `x+2*u>=1` for R, and in either case
`27*c+16*M>=16`; the other endpoint bounds follow on S. Thus changing order
changes the return domain without changing an operator law. For example,
`x=3/8,c=1/2,u=3/8,M=1/2` yields forms `3/16` versus `9/32` and the common
capacity `43/128`. Only R returns to S. With `u=3/4,M=1`, both return, although
F first reaches form `9/16>1/2` and R first reaches `3/16<1/4`. Requiring every
prefix to lie in S would exclude these admitted returns.

Leaving S alone establishes neither disappearance nor failure of every
continuation: the lower donor admits a different form region in the earlier
return law. Likewise, equal selector inputs can conceal distinct preservation:
with `x=1/4,u=3/4`, RA maps capacities `1/2` and `1` to `9/16` and `9/8`.
Both execute; only the first respects S's chosen capacity ceiling. A pressure
gate can also be repaired through an actual operator: IL takes `p=3/4` to
`3/8`, opening RA's default `1/2` ceiling without restoring S's tighter bound.
These are distinct structural obligations, not reasons to alter the selector
or substitute a successful glyph after rejection.

### Evaluated admission and order controls

The [four finite controls](../tests/operators/test_operator_viability_continuation.py)
passed with 65 selected synthetic admission/decision-trace regressions:
69 passed, no failures or skips. The prospective protocol, reviewed source
archive, complete log and JUnit evidence are retained locally in
`artifacts/operator-viability-20261103/`. The single frozen run left production
and evaluated sources unchanged; no historical producer was replayed.

Two real Default prepare/select/clear calls consumed equal normalized metrics
`(3/8,1/8,0)` and proposed RA. The actual registry-selected stages admitted both
capacities and produced `9/16` and `9/8`, confirming distinct preservation of S.
The separate grammar check admitted the high-pressure RA candidate, while its
actual strict stage rejected atomically without an occurrence. The completed
preparation remained. One actual IL then enabled RA with final pressure `3/8`,
still outside S. The run retained two advisory warnings: initial IL pressure
above its warning bound and consecutive IL. Neither is a grammar or strict
execution rejection; they were not suppressed or converted into extra actions.

Both matched-order controls met their predicted form, capacity, pressure and
phase rows, every prefix envelope check and each own-history assertion. The
lower donor returned to S only under NUL-first order; the upper donor returned
under both orders after intermediate departures from S. The participant's UM
phase write produced no own UM credit. Distinct receiver histories remained
even when capacity, pressure and phases agreed.

The new controls used seven fresh two-node preparations, 28 stage requests,
34 successful target occurrences and one expected rejected RA. Exact dyadic
assertions and the declared absolute `2e-14` phase/pressure tolerance passed.
This is finite correspondence with the preceding ideal conditions, not a
uniform represented-arithmetic theorem, maximal viability kernel or autonomous
choice law. Each stage is atomic; the supplied word is not one transaction.
The numerical controls activate IL, RA, NUL and UM, while the other operators
retain their individual domains and unresolved roles in the common ledger.

