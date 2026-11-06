# Recursivity: structure, memory and post-birth participation

Recursivity (REMESH) historically concerns an organized pattern expressed
again across scales. Its current executable fragments are narrower: a
grammar-aware advisory, a separately invoked temporal EPI map, and separate
topology/identity helpers. This chapter distinguishes those functions,
clarifies local versus collective memory, and owns the finite post-birth
participation contract. It does not declare a completed canonical REMESH law.
The [operator owner](STRUCTURAL_OPERATORS.md#43-recursivity-remesh),
[fixed-delay derivation](REMESH_INFINITY_DERIVATION.md) and
[collective sine specialization](nodal/SINE_COLLECTIVE_RECURSIVITY.md) retain
their existing maps and mathematical hypotheses. The
[program](RESEARCH_PROGRAM.md#research-current-question) alone selects work.

<a id="remesh-historical-role"></a>
## Historical role and limits of the source

Source: *El pulso que nos atraviesa / Manual operativo TNFR*, the local
`bibliografia/TNFR.pdf`, 378 pages, SHA256
`5ba0f4a2da2d01e550e7004c3c09c6927e6620826052df84cd28babbcbd34fc3`.
It matches the source retained by the earlier canonical-operator audit.
The locators below are one-based PDF pages; inspected printed footers agree.
Private books and page images remain outside publication archives.

| Passage | Historical intent | Obligation retained for development |
| --- | --- | --- |
| p.79, Fig.2.1 | REMESH occupies the central recursive role among the thirteen glyphs | Treat it as a proposed operation on organized patterns and their reuse; a central illustration does not prove mathematical independence or completeness |
| pp.124-128, especially p.125 | Re-expression at multiple scales, structural field memory, continued identity; a schematic sum of reconfigurations with `W_s approximately W_0` | Define the objects, scale correspondence, combination operation, observable notion of preservation and error; these are not supplied by the notation alone |
| p.126 pseudocode; p.131 table; p.215 | Replication/reconfiguration and EPI expressed at another scale; generation of further networks | A clone or graph replacement needs admission, inheritance, ports, history and continued dynamics; it is not automatically functional recursion |
| p.135, structural reactivation | Earlier structures can support the opening of a new organization | Reuse retained operative structure without fabricating a new member's past; activation and delay selection require their own laws |
| pp.136,138-139 | Nested words, local coherence and global organization; cyclic and fractal forms are distinguished | Preserve own context at each level and distinguish repetition in time from re-expression across scales |
| pp.230-231 | `S(x)=S(k*x)` symbolizes scale-related pattern preservation | Supply a typed state/observation map, scale domain and equivalence; this schematic relation is not a theorem about arbitrary networks |

This gives REMESH a differentiated historical function: transporting or
re-expressing organized behavior, potentially including operator compositions,
across levels. It does not make REMESH exempt from grammar, a controller of all
other operators, or a proven irreducible fourteenth execution layer. Expansion
and Self-organization also have constructive roles; their relation to REMESH
requires a realization or an explicit constitutive distinction.

Memory is not assigned exclusively to REMESH even in this source. Page 111
associates Contraction (NUL) with stable synthesis of experience; p.108 also
links Expansion (VAL) with transmission of an operative pattern. Those passages
motivate different roles in acquiring, retaining and reusing organization.
They do not identify all memory with a delayed list of numbers. The inspected
historical passages do not specify a unique local/global two-delay memory law.
The present runtime formula must therefore retain its status as a particular
implemented constitutive fragment, not a deduction from those passages.

Historical algebra also needs independent checking. Pages 226-228 sketch
operator identities and closure, including idempotence and Silence as an
identity. These are not literal identities of the current state/context maps.
The [contextual-action analysis](STRUCTURAL_OPERATORS.md#operator-contextual-rewriting)
owns the applicable restrictions. Historical motivation cannot silently
replace the live grammar or eliminate consumed context.

<a id="remesh-contraction-expansion"></a>
## Contraction, Expansion and structural re-expression

[VAL](STRUCTURAL_OPERATORS.md#72-expansion-val) and
[NUL](STRUCTURAL_OPERATORS.md#102-contraction-nul) belong to one implemented
family of target-local scale actions, but are not inverse complete maps.
On the ideal edge-aware branch with inactive boundary adaptation/projection,
VAL_g sends `(x,nu,p)` to `(g*x,g*nu,p)`, whereas NUL_f sends it to
`(f*x,f*nu,p/f)`, with g>1 and 0<f<1. Reciprocal factors can restore form and
capacity after either order, while pressure remains p/f. With edge awareness
disabled, form is simply unchanged. Default factors are not reciprocal.
Rounding, clipping and strict prefix admission further restrict reversal;
each invocation retains its own context, and NUL does not discharge VAL's U2
debt. These are state-changing events, not passive coordinate transformations.
The [shared kernel](../src/tnfr/operators/_scale_operator_kernel.py) and
[retained-member contract](STRUCTURAL_OPERATORS.md#retained-member-scale-actions)
own these maps; the pressure asymmetry is not silently replaced by a new law.

Their historical constructive roles complement REMESH's proposed re-expression.
Ordinary VAL/NUL preserves membership and ports. The separate
[derived-member VAL family](STRUCTURAL_OPERATORS.md#derived-expansion-member)
adds an isolated successor with explicit inheritance and fresh context;
the [NUL/EN fragment](NUL_SYNTHESIS.md#retained-synthesis-concentration)
concentrates form toward a core while retaining members and ports. Neither
supplies a general memory-preserving merge or inherited executed history.
REMESH still requires an explicit correspondence of organization, boundary
response and own admission across its re-expression. Actual admitted VAL/NUL
and other operators may participate in that realization; rescaling coordinates
or restoring one form value does not establish it. This is a dependency of
the same program question, not another research queue.

<a id="remesh-memory-channels"></a>
## What local and global memory mean in the current engine

| Retained channel | Actual role | What it does not establish |
| --- | --- | --- |
| Per-node `epi_time_history` and legacy `epi_history` | Associated form samples consumed under specific time/step contracts, including acceleration or Mutation admission | Another node's history, or a population-wide REMESH snapshot; a single birth value gives no elapsed-time secant |
| Own glyph history and persistent grammar state | Past occurrences, stabilization/debt and depth context used for future admission | A global pooled allowance; one member's successful action does not credit its siblings |
| Birth associations, lifecycle and retained organization records | Explicit membership/access and source-bound observations of primitive state, own context and ports | Authenticated ancestry merely from metadata, or a proved persistent collective identity |
| Graph `_epi_hist` | Indexed whole-population snapshots, each mapping a node to that node's EPI | A spatial average, a single collective EPI, a memory of every possible scale, or proof that supplied entries were acquired |
| Structural signatures and identity/topology helpers in `remesh.py` | Separately requested similarity, grouping, propagation or support-change policies | Implicit effects of the advisory or the delayed map; universal identity preservation |

In the delayed map, both `local` and `global` read the **same node** from the
**same** `_epi_hist` at two sample offsets. Defaults are `tau_local=4` and
`tau_global=8`; they can be configured. These are not two separate stores,
one for a constituent and another for its enclosing organization. Nor does
`global` average the network. The coefficient is resolved through the existing
configuration/factor precedence, not inferred from the labels of those delays.

For fixed admitted coefficient a, delays, clipping and no callback, the raw
row for node i is

```
x_i_new = (1-a)^2*x_i + a*(1-a)*H[k-tau_local][i]
                         + a*H[k-tau_global][i].
```

The shared sampler appends the current pre-map state, and delay tau is read at
`history[-(tau+1)]`. Here k indexes sampling occasions. Equal timestamps can
contain different states separated by actual operator jumps; this does not
turn an index delay into a positive physical duration. The map changes EPI
over all current nodes, preserving capacity, phase, pressure and support
before any separately declared refresh. It appends no delayed sample itself.

At this fixed configuration, equal current/selected historical vectors give
the same raw echo even on different supports with the same node identities.
Yet EN/RA admission and response can differ because they consume realized
relations. The row alone therefore cannot define structural re-expression or
guarantee its preservation of interaction. Likewise, a configurable similarity
score does not supply the missing identity theorem.

<a id="remesh-local-collective-hypothesis"></a>
## A precise reading of the local/collective-memory intuition

The useful research hypothesis is that REMESH can re-express an organization
while respecting both members' own memory and the retained relational state
needed for collective behavior. This is broader than temporal echo and remains
unresolved. Local/global are relative to a declared organization and scale,
not absolute labels that determine a universal memory hierarchy.

Start with an organized state containing member identities, interior support,
external ports, primitive coordinates, own histories/context and the boundary
inputs needed for its actions. A collective memory may be distributed across
that state; no extra scalar register or network average is presumed sufficient.
The [NUL response and memory analysis](NUL_SYNTHESIS.md) already demonstrates
why equal exported form can hide different later responses and admission.

For a declared family of admissible words and observations, a sufficient
retained representation must determine both **whether each word can execute**
and **what the selected response will be**. Preservation must hold at every
consumed prefix, not only at the final plotted EPI. This reuses the existing
[partial-action/grammar descent](STRUCTURAL_OPERATORS.md#collective-grammar-descent)
requirement rather than introducing a second theory of grammatical credit.

A candidate re-expression thus needs an explicit correspondence of members,
ports and observations, rules for retaining or generating memory, and actual
operator realization. Reusing an ancestor's pattern does not imply that newly
created constituents lived through the ancestor's events. Existing members may
retain their own valid past; genuinely new members require fresh context.
If information is deliberately discarded, its effect on future response and
admission must be bounded or shown irrelevant in the selected domain.

These are prerequisites for a future REMESH law, not a proposed universal map
or a completed cross-scale proof. The program selects the first bounded
discriminator; this chapter does not activate an additional research queue.

<a id="remesh-postbirth-admission"></a>
## Post-birth sampling and existing admission

The [shared source helper](../tests/operators/_remesh_postbirth_source.py)
uses actual Coherence (IL), Dissonance (OZ), two finite Emission (AL) episodes,
eligible Self-organization (THOL), birth-associated Coupling (UM), and further
IL. The child exists before positive REMESH sampling begins. This is an
explicit observation initialization, not an automatic memory-reset policy.

There are two distinct existing read domains:

- The direct delayed map returns `insufficient_history` until at least
  `max(tau_local,tau_global)+1` rows exist. It then admits the two selected
  lag rows against current support and signed scalar form.
- The event/REMESH cycle admits **every** retained row against current node
  support before the schedule. It requires unchanged ordered node identities
  throughout that schedule, appends one actual pre-map row, then invokes mixing.

Consequently a genuinely captured pre-birth row can invalidate the cycle even
after later samples make the direct map's selected rows complete. This is a
stronger contract, not permission to pad the old row with a child's current
value. Adding samples does not immediately discard that old row. A new member
also cannot be born inside the unchanged fixed-node cycle contract.

The [admission controls](../tests/operators/test_remesh_postbirth_admission.py)
retain these distinctions, actual sampling, the no-history advisory, genuine
birth rejection inside the cycle, and rollback after a late observer fault.
The late-fault control uses a callback registry normalized before cycle entry.
The prepared source with a still-dirty registry rejected earlier at callback
normalization; the passing late-fault control does not certify that transition.
They do not implement a reset, history repair, automatic occurrence or a new
interpretation of the old delayed-map APIs.

<a id="remesh-postbirth-response"></a>
## Finite continued participation through actual operators

The source begins with an ordered parent-peer edge, form 1/4, capacity 1,
pressure 1/2 and phase zero at each node. Actual IL/OZ and two Euler AL episodes
of duration 1/8 acquire samples at times 0,1/8,1/4. Between episodes, parent
pressure changes from 1/2 to 5/8 through actual IL/OZ; peer pressure stays 1/2.
Their observed accelerations are respectively 1 and 0. At the supplied
threshold 1/2, actual eligible THOL creates one isolated child, then actual
birth-access UM connects it. Further IL leaves pressures `(9/128,1/8,0)`.
The selected source, occurrence occasions and inherited coefficients remain
supplied premises, not autonomous formation laws.

Both arms of the [response control](../tests/operators/test_remesh_postbirth_continuation.py)
acquire the same first live post-birth snapshot:

```
H0 = (parent, peer, child) = (25/64, 3/8, 15/128).
```

Actual instantaneous AL at the child adds 1/8. Both arms execute the same
full-population depth-one REMESH advisory schedule with zero-duration intervals
and append one actual pre-map snapshot:

```
H1 = (25/64, 3/8, 31/128).
```

The advisory-only arm uses the existing schedule executor and sampler; the
action arm uses the existing event/REMESH cycle. With both delays one, a=1/2,
hard clipping on [-1,1] inactive and no pressure refresh, both selected rows
are H0. The actual map is `x_new=x/4+3*H0/4`: parent and peer stay fixed, while
child becomes 19/128. The map acts on all nodes; only the child had changed
between the selected history and current state in this particular control.
Both retained histories stay `[H0,H1]`; the mapped endpoint is not a third row.

Selected EN at the parent reads only the child with blend 1/2, under the
unchanged pressure ceiling. RA at the peer then reads its sole parent neighbor
with form blend 1/2 and capacity gain 1/8:

| Form observation | Advisory only | Advisory plus actual delayed map |
| --- | --- | --- |
| Child before EN | 31/128 | 19/128 |
| Parent after EN | 81/256 | 69/256 |
| Peer after RA | 177/512 | 165/512 |

The peer contrast is -3/128, the child's map difference propagated through
the two actual blends. These forms are exact dyadics on the declared branch;
UM's represented capacity/affinity are retained without an exact-dyadic claim.
Clock stays 1/4. Sampling after same-time jumps does not create elapsed time.
The delayed map's own temporal right-endpoint recording is distinct from the
two-row sampling history and may differ between arms.

Final lifetime operator counts are parent 13, peer 8 and child 3. With the
current seven-entry retained glyph window, their respective suffixes are
`THOL,UM,IL,IL,IL,REMESH,EN`, `OZ,AL,AL,IL,IL,REMESH,RA`, and
`IL,AL,REMESH`. A map invocation supplies no second REMESH occurrence.
This bounded composition realizes temporal memory participation after actual
membership growth. It does not complete the historical cross-scale function,
derive a local/collective-memory law, establish persistent identity or prove
canonical completeness of the thirteen operators.

<a id="remesh-addressed-port-reexpression"></a>
## Structural re-expression of one addressed port

This bounded construction retains two logical ports, represented by actual
members `(A,B)` or `(A,C,B)`. Port A reads the singleton A or the pair A,C;
port B remains independently addressed. The correspondence is
`x=(A+C)/2`, with a retained nonzero contrast `d=(A-C)/2`. This is a declared
interface observation, not a replacement of the actual members by a scalar
macro-node. State, capacity, pressure, phase, semantic kind, own history and
realized support remain part of the implementation and its admission.

The [retained output consumer](../src/tnfr/operators/retained_reception.py)
complements addressed incoming Reception. It resolves an external receiver
shared by every member's outgoing support, then executes actual EN at that
receiver with precisely those members as its selected sources. The existing
unweighted unique-neighbor reader owns the mean. Directed reverse-only edges
do not carry this output, and parallel edges do not multiply its sources.
The [API contract](../docs/API_CONTRACTS.md#retained-addressed-output) owns live
re-admission, receiver state, source preservation and transaction boundaries.
A passive output table is neither an execution authorization nor an identity
certificate. Sources acquire no extra glyph from another node reading them.

#### Conditional preservation through a marked input/output family

Let coarse A have form x and refined A,C have forms x+d,x-d, with d nonzero.
Let B have form y in both realizations. Actual EN with the same factor m reads
an independent donor u into A, or into both A,C. EN with factor n reads another
donor v into B. On the same-kind scalar branch with inactive projection:

```
x1 = (1-m)*x + m*u,  d1 = (1-m)*d,  y1 = (1-n)*y + n*v.
```

The mean follows the coarse form while both refined members remain present.
For 0<m<1 the contrast remains nonzero. Give an external receiver initial
form z. It first executes addressed EN from port A with factor q, then from
B with factor s. A further observer with initial form w executes actual RA
with sole neighbor that receiver and factor r. Corresponding ideal rows are

```
z1 = (1-q)*z + q*x1
z2 = (1-s)*z1 + s*y1
w1 = (1-r)*w + r*z2.
```

Each realized incoming edge, own target context and pressure/capacity condition
must admit its action. Preserve both receiver-prefix observations z1,z2;
the final scalar w1 is not a complete encoding of two independent inputs.
This is a finite marked-language statement under the existing
[partial-action criterion](STRUCTURAL_OPERATORS.md#collective-grammar-descent),
not equivalence of arbitrary words or full constitutive dynamics. Individual
histories and capacities need not be identical across the two representations.
Arbitrary binary64 evaluation does not inherit a literal real-matrix identity.

#### Acquired unequal source and finite continuation

The [actual-source controls](../tests/operators/test_port_reexpression_continuation.py)
start from the same six-node graph `(A,B,U,V,R,W)`: A,B are members; U,V are
input donors; R is the receiver and W its sole-neighbor observer. Initial
edges are A-B, U-A, V-B, A-R, B-R and R-W. All phases and Si are zero;
capacities start at one. Form starts at 1/4 for A,B,R, 1/8 for U,V and 3/8
for W. Pressure starts at 1/2 for A,B and zero externally. Actual IL/OZ and
two AL episodes of duration 1/8 acquire A=25/64, B=3/8 and accelerations
1,0 at clock 1/4. The external inputs are still unchanged.

The refined branch executes eligible THOL at A, obtaining an isolated child
C=15/128 with fresh context, then actual birth-access UM supplies A-C. C
acquires its own IL. Ordinary UM at C uses its existing A neighbor and the
supplied candidate list `(U,R)` with candidate count zero, creating C-U and
C-R through the current affinity law. It neither uses the disjoint-component
request outside its domain nor pretends new candidates were existing consensus
participants. At zero phase/Si and these positive forms, affinity is at least
3/4, above the supplied 1/2 threshold. No other candidate is exposed to UM;
W retains sole neighbor R. These choices are declared access and occurrence,
not an autonomous policy.

Actual EN at C from A with factor 1/2 gives C=65/256 and acquires the parent's
actual `form` kind. Further actual IL lowers pressure. A public instantaneous
AL at refined A then adds the prospectively derived boost 35/256, giving

```
A_refined = 135/256, C = 65/256
mean = 25/64 = A_coarse, contrast = 35/256.
```

The boost restores the original port mean by actual additional operator work;
birth alone does not preserve it. AL starts below its latent threshold and
admits this bounded endpoint under its public jump contract. Its domain is
distinct from the finite-duration episode's stricter endpoint restriction.
No post-birth EPI, kind, elapsed history or successful context is assigned.
Pressures become coarse A=5/64, refined A=9/128, B=1/8, C=0; both branches
meet the subsequent strict EN/RA policies. Represented UM capacities and
affinities remain actual state without a claim that all are exact dyadics.

Only then do actual donor AL actions supply the two independent inputs.
All five continuation factors m,n,q,s,r are 1/2. Fresh source records are
captured after input EN, and the output consumer resolves R by its real port
address; ordinal zero is not presumed. The two prepared representations have
these exact dyadic predictions on the unclipped branch:

| u | v | R after port A | R after port B | W after RA |
| --- | --- | --- | --- | --- |
| 1/4 | 1/2 | 73/256 | 185/512 | 377/1024 |
| 1/4 | 3/4 | 73/256 | 217/512 | 409/1024 |
| 1/2 | 1/2 | 89/256 | 201/512 | 393/1024 |
| 1/2 | 3/4 | 89/256 | 233/512 | 425/1024 |

The refined contrast after input is 35/512. Only R acquires the two output EN
occurrences; the sources retain their own contexts. W's actual RA changes its
capacity to 9/8. Final lifetime counts are coarse A=10 or refined A=13, C=4,
B=7, U=V=2, R=3 and W=2. Full lifetime counts are distinct from the current
seven-entry glyph suffix. The child never inherits A's AL activation history.
The clock stays 1/4 throughout this instantaneous continuation.

#### Why flattening the ports changes behavior

The [replication obstruction](STRUCTURAL_OPERATORS.md#retained-member-scale-actions)
still applies. In a separately matched one-EN/one-RA word, flattening both ports
exposes coarse sources `(A,B)` or refined `(A,C,B)`. Their means differ by
`(x1-y1)/6`; the two 1/2 blends transmit `(x1-y1)/24` to W. For u=1/4,v=1/2,
W is respectively 353/1024 and 348/1024, a contrast of -5/1024. Both flattened
branches have the same word; neither is compared against the longer addressed
word as though their schedules were equal. A labelled diagnostic control with
the two refined output edge weights set to 1/2 retains that difference: weights
do not alter this reader's causal multiplicity. Those weight edits are not
canonical preparation events.

The [output boundary controls](../tests/operators/test_retained_output_ports.py)
separately retain source freshness, actual common support, live receiver
admission and rollback. These are the operative memory/port dependencies of
this construction. The finite controls establish this prepared interface
capability; universal memory sufficiency, autonomous
re-expression, full operator descent and complete canonical REMESH remain open.

<a id="collective-emission-port-realization"></a>
## Emission through an admitted member of the retained port

The addressed port retains actual constituents rather than identifying all
of them with its mean. A collective form increment can therefore require a
different local action in each representation. The
[retained Emission consumer](../src/tnfr/operators/retained_emission.py) makes
one such realization explicit: a supplied member of an N-member group executes
actual AL with configured boost `N*b`, realizing a declared mean gain b.
The [API contract](../docs/API_CONTRACTS.md#retained-collective-emission) owns
strict live admission, exact represented arithmetic and transaction boundaries.
It supplies neither autonomous target selection nor a new primitive AL law.

### Prospective member choice and conditional identity

Use the actual acquired source above before independent input changes. Coarse
A has form `x=25/64`. Refined A,C have `x+d,x-d`, with `d=35/256`, so their
forms are `135/256,65/256`. The configured strict AL latent ceiling is 1/2:
coarse A and refined C pass this entry test, while refined A fails. Their
actual capacities exceed the basal floor; each selected target has its own
admitted grammar context. Stored pressure and phase are retained separately.
The choice of C follows this pre-response state. The API still receives its
ordinal as a supplied command, and validates the actual full action again.

For `0<b<=1/16`, AL of b on coarse A and AL of 2b only on refined C give

```
coarse A' = x+b
refined A' = x+d,  C' = x-d+2b
refined mean' = x+b,  contrast' = d-b.
```

The local boost is representation-dependent. Both arms follow the same
declared collective command but not an identical globally configured AL
factor. The consumer checks the live configured factor without overwriting it.
The endpoints obey `A_coarse'<=29/64`, `C'<=97/256`, and `d-b>=19/256>0`.
The two refined members remain distinct. Actual instantaneous AL preserves
capacity, pressure, phase and semantic kind; its endpoint domain is separate
from the finite-duration episode's domain. No elapsed time is introduced.

The exact-gain API requires the shared AL proposal and actual represented
jump to agree with this translation. This adds an explicit restricted numerical
domain; it does not claim exact real-arithmetic identities for every binary64
addition or change ordinary AL's projection contract.

### Continued response and own activation

Following this action, use the same independent donor inputs and actual EN/RA
word as above. Its symbolic mean rows become

```
x1 = (25/64+b+u)/2,  d1 = (35/256-b)/2,  y1 = (3/8+v)/2
z1 = (1/4+x1)/2
z2 = (z1+y1)/2
w1 = (3/8+z2)/2.
```

Relative to the no-additional-AL rows, the two receiver prefixes gain `b/4`
and `b/8`; the final observer gains `b/16`. These are predictions derived
before evaluating the five paired controls:

| b | u | v | R after A | R after B | W after RA |
| --- | --- | --- | --- | --- | --- |
| 1/16 | 1/4 | 1/2 | 77/256 | 189/512 | 381/1024 |
| 1/16 | 1/4 | 3/4 | 77/256 | 221/512 | 413/1024 |
| 1/16 | 1/2 | 1/2 | 93/256 | 205/512 | 397/1024 |
| 1/16 | 1/2 | 3/4 | 93/256 | 237/512 | 429/1024 |
| 1/32 | 1/4 | 1/2 | 75/256 | 187/512 | 379/1024 |

The [continuation controls](../tests/operators/test_collective_emission_continuation.py)
compare both receiver prefixes and the final actual RA response. The shared
[preparation helper](../tests/operators/_port_reexpression_source.py) retains
the preceding source's real IL/OZ/AL/THOL/UM/EN construction; successful context,
child form or parent history is never assigned after birth.

Coarse A reactivates from two to three AL occurrences, preserving its origin.
Refined C acquires its first AL, its own origin and activation count one;
refined A keeps its three activations. C's complete short word after input is
`IL,UM,EN,AL,EN`. Final lifetime event counts are coarse A=11, refined A=13,
C=5, B=7, U=V=2, R=3 and W=2. The runtime clock stays at 1/4; first-activation
wall-clock metadata is provenance, not elapsed structural time. RA changes
W's capacity to 9/8. These own histories are different even when output agrees.

### Broadcast obstruction and successor-domain limit

A matched uniform broadcast would apply boost b to both refined members,
giving the same mean increment if admitted. In the actual source, refined A
already exceeds the latent ceiling. One atomic AL stage on `(C,A)` must reject
with no partial child activation. This is an admission obstruction, not a
failure of the hypothetical addition formula. Other possible distributions
of work require their own domains and do not follow from this example.

A further analytical boundary rules out extending the mean identity to an
unrestricted repeated language. Set b=1/64 and condition on the ideal repeated
translation rows. After k repetitions,

```
coarse A_k = (100+4*k)/256
refined C_k = (65+8*k)/256,  refined A_k = 135/256.
```

After seven, both port means are 1/2. The coarse latent test rejects another
AL, while refined C is still `121/256<1/2` and passes that test for an eighth
increment to `129/256`. This is an algebraic distinction between latent-test
domains, not an evaluated eight-event trajectory or proof that every other
grammar/lifecycle obligation admits that word. It already explains why the
one-step mean formula cannot certify admission of arbitrary continuations.
Any full [partial-action descent](STRUCTURAL_OPERATORS.md#collective-grammar-descent)
must preserve decisions and successors on its entire declared reachable domain.

The five paired continuations and rejected broadcast satisfy their declared
finite controls. This prepared realization concerns one marked AL/EN/RA
family. Constituent admission, configuration, own memory and support remain
part of the executable state. They do not establish mean-only sufficiency,
autonomous choice, full operator descent or complete canonical REMESH.

<a id="retained-silence-reactivation"></a>
## Silence and reactivation as a guarded continuation

An organization retains its members, ports and own event contexts through a
capacity change. This does not make every later action available. The
[hybrid and guarded-rewriting reading](BIBLIOGRAPHIC_METHODS.md#bibliographic-hybrid-guarded-lifecycle)
motivates checking the desired successor domain on the complete post-event
state. The following calculation uses existing pointwise SHA, NAV and IL,
the retained AL consumer and actual addressed EN/RA. It adds no operator law,
selector, collective node or automatic recovery mechanism.

### Capacity recovery is only one successor condition

Write c for a target's capacity and x for its signed scalar form. The actual
Silence (SHA) event uses a supplied factor a in [0,1), sends c to a*c and
preserves x, stored pressure, phase and support. It records its own latency
snapshot and appends its own grammar event. Preserving form at this jump is
distinct from preserving it during a later continuous interval: the unforced
nodal row would still be x_dot=(a*c)*p. The latency flag does not override it.

For the selected deterministic strict Transition (NAV) branch, zero jitter
and a latent target, the ideal scalar rows are

```
c_s = a*c
NAV: c_n = 1.2*c_s, p_n = 0.7*c_s, theta_n = theta + delta (mod 2*pi).
IL repeated k times: p_k = r**k * 0.7*c_s.
```

Here r is the supplied IL pressure factor. Its phase-locking coefficient is
zero in these controls. All three events preserve form; NAV uses pre-NAV
capacity for its strict pressure handler. These formulas describe this branch,
not every NAV regime. The runtime evaluates represented arithmetic and must
admit every intermediate value, actual phase domain and own grammar prefix.

For a positive basal Emission floor B, silence blocks that capacity gate and
one latent NAV restores it only in the ideal band

```
B/1.2 <= a*c < B.
```

That band is not sufficient for the complete continuation. NAV has its own
capacity/pressure guards, AL requires x below its configured latent ceiling,
and subsequent Reception (EN) requires its own signed-pressure and form bounds.
For positive Reception ceiling P, its pressure condition is
`r**k * 0.7*a*c < P`; equality rejects. Every consumed source, port and phase
condition must also hold. This is the existing partial-action domain rule
applied to actual lifecycle changes, rather than an extra grammar rule.

SHA with a=0 supplies a separate obstruction. Every admitted member action
in SHA/NAV/IL/AL either scales capacity or preserves it, so zero capacity
remains zero under that restricted family. Default NAV rejects it at entry,
and strict AL cannot reach a positive basal floor. Even a policy admitting
zero-capacity NAV would not make its multiplicative map create capacity.
This is not a theorem that zero capacity is absorbing under every TNFR
operator or interaction; another capacity source needs its own admitted law.

### Own memory and the temporal boundary

Reactivation must retain the existing AL origin and increment its own activation
count, or create a first origin for a member that has never emitted. Parent
history cannot replace child history. NAV deliberately clears the active
latency flag, start and preserved-form snapshot after a successful transition;
it retains a measured silence duration and its transition record. These are
different memory updates, not a promise to leave all metadata unchanged.

The controls hold the runtime structural clock fixed between events. UTC
timestamps bound the actual metadata observation; they do not supply an
emergent clock or a continuous waiting trajectory. A successful finite prefix
does not establish a closed standalone word, indefinite maintenance or an
endogenous event schedule. A later such claim must specify the intervening
flow, timing law, complete state and successor domains.

### Evaluated finite continuation and its obstructions

The [five lifecycle controls](../tests/operators/test_silence_reactivation_continuation.py)
were specified and independently reviewed before their single evaluation.
Their source and predictions were frozen in the local
`artifacts/silence-reactivation-20261103` record. All five passed, alongside
208 selected existing synthetic contract tests. No production action or
previous acquired-port helper was changed to obtain the result.

The supplied five-node preparation acquires its own IL/AL context. Its target
C starts the new prefix at form 1/4, capacity 15/16 and pressure 1/16. SHA
with a=1/2 gives capacity 15/32; strict AL rejects while preserving form,
latency, lineage and support. Actual latent NAV gives capacity 9/16, pressure
21/64 and phase 1/8. One IL leaves pressure 21/128: AL can execute, but the
following actual EN rejects. Two IL actions instead leave 21/256 and admit
the declared continuation. Rejected invocations preserve the completed
prefix and graph-owned state; no fallback member or repaired history is used.

After actual AL adds 1/8 to C and EN reads the supplied donor form 1/2,
C becomes 7/16. Its retained group with A=3/4 has mean 19/32. The addressed
receiver R becomes 27/64 and its sole-neighbor observer W becomes 43/128
through actual RA. The matched reference omitting SHA/NAV has the same forms
at R/W, but different member capacity, pressure, phase and own context.
C keeps its original AL origin and reaches activation count two; the recovered
and reference own lifetime event counts are eight and six. Equality of this
response is not full-state equivalence. SHA with a=0 separately yields the
predicted NAV and AL rejection at their unchanged positive capacity floors.

One additional paired control transfers the prefix to the previously acquired
coarse/refined port preparation, using that unchanged construction exactly once
per arm. It retains the two real AL episodes and the refined THOL/UM birth and
contact. SHA uses a=15/32 in both arms: the selected actual capacity in [.95,1]
falls below 1/2, then latent NAV raises it above 1/2. Two IL actions put its
pressure below .083, within the EN bound. The form-preserving prefix permits
the same mean AL gain 1/16, donor inputs (1/4,1/2) and addressed outputs:

| Observation | Coarse and refined response |
| --- | --- |
| R after the A-group input | 77/256 |
| R after the B input | 189/512 |
| W after actual RA | 381/1024 |

The child acquires its own first AL activation; the parent never lends it
activation credit. The two members remain unequal, and the refined contrast
after input is 19/512. Their capacities, phases and histories are retained
separately. Three warnings report the existing adjacent-IL advisory pattern;
the live grammar admitted these prefixes without replacement or disabled
rejection. No repeated-recovery trajectory, global maintenance domain,
autonomous selection or complete collective reduction was evaluated.

### Why one recovery does not establish maintenance

Condition on repeating the same latent SHA/NAV member prefix without another
capacity-changing action on that member. In the ideal real-arithmetic map,
`c_(n+1)=q*c_n` with `q=1.2*a`. If `0<=q<1` and the AL basal floor is
positive, `q**n*c_0` eventually falls below that floor. Thus the repeated
prefix cannot stay AL-admissible indefinitely; another guard may reject
earlier. This proof does not assume that a rejected prefix keeps executing.

The selected factors a=1/2 and a=15/32 have q=3/5 and q=9/16, respectively.
A single additional RA capacity multiplier 9/8 per repetition would still
leave both net factors below one, even assuming its actual amplification
trigger and admission hold. These are conditional map calculations, not evaluated
repeated trajectories. A represented-arithmetic persistence claim needs its
own return bounds. An admitted UM capacity blend can additionally depend on
participants' capacities, but their actual state, phase, support, effects
and admission must be retained; inserting a refill value is not a derivation.
NAV's phase increments and the form/pressure domains also belong in a return
argument. This identifies a maintenance dependency shared by coupling,
resonance and lifecycle integration, rather than just another recovery case.
Such a return domain must permit the specified updates of own history,
activation counts and lifecycle metadata; literal equality of the entire
event-counted state is not required for sustained admitted activity.

<a id="operator-maintenance-return"></a>
## A return domain maintained by actual Coupling

The loss in the preceding SHA/NAV prefix is not remedied by its label or by
restoring a saved capacity. Actual Coupling (UM) has an additional mechanism:
its target capacity moves toward the mean of its actual compatible neighbors.
This section derives a conditional return domain for that existing action,
including the other channels and own memory consumed by the continuation.
It concerns supplied support, parameters and event order, not their autonomous
selection or the formation of the underlying organization.

### Complete selected action and participants

Use five nodes A,C,U,R,W with undirected unit-weight edges A--C, U--C, A--R,
C--R and R--W. A and C are distinct retained members; R is their actual common
output receiver. Initial forms are `(3/4,1/8,1/4,1/4,1/4)`, all of kind form;
C has capacity 15/16 and pressure 1/8, while other capacities are one and
pressures zero. Phases start at zero. Actual IL on each node followed by AL
with boost 1/8 on C prepares C form 1/4, pressure 1/16 and its own first
activation. Support is supplied; histories and activation origins are not.

The recurring C prefix is

```
SHA, NAV, IL, IL, UM, UM, AL, EN(source=U),
```

followed by addressed output EN at R reading the fresh retained group `(A,C)`.
Set SHA factor 1/2; deterministic strict latent NAV with zero jitter and phase
shift 1/8; IL pressure factor 1/2 and phase coefficient zero; UM capacity
sync 1/4, phase push 1/2 and pressure reduction 1/4; AL local boost 1/8 and
EN mix 1/2. Use one-way UM, with capacity synchronization and pressure relief
enabled and functional links disabled. Hard form bounds [-1,1] remain inactive.
All ordinary preconditions and live incremental grammar remain enabled with
rejection mode `raise`; no replacement glyph is allowed.

These declared UM branches preserve support and change only C's phase,
capacity and pressure. Its compatible neighbors A,U,R retain capacity one and
phase zero through the entire selected word: output EN changes only R's form
and context. Thus their capacity mean and phase reference follow from their
actual actions; no neighbor is refilled or reset between returns. U retains
form 1/4 and A form 3/4. UM implements no debit of neighbor capacity, so a
conservation or depletion law must not be inferred from this renewal.

No inter-event flow is selected. The structural clock stays fixed, while real
UTC lifecycle observations must remain valid. Event-counted repeatability
does not prove persistence over positive continuous time or exclude Zeno
behavior in a different timed model. The final RA readout described below
is outside this return map; repeated RA would change W's capacity.

### Ideal return map and prefix admission

Write `(c,x,p,theta,z)` for C capacity, form, pressure, lifted phase and R form.
On the stated acute circular chart, the ideal real-arithmetic rows are

```
c'     = (27/80)*c + 7/16
x'     = x/2 + 3/16
theta' = (theta + 1/8)/4
z'     = z/2 + (3/4 + x')/4.
```

SHA and NAV first give capacity `(3/5)*c`; two real UM occurrences each apply
`v -> (3/4)*v + 1/4`. With the actual phase-zero participants, each UM halves
C's current phase. AL followed by selected-source EN gives the stated form
map. The receiver row consumes the updated two-member mean, not an invented
macro-node or a copied source value.

Pressure requires the actual final-phase affinity. After NAV and two IL,
`p_i=7*c/80`. Put `theta_1=(theta+1/8)/2`. The shared linear circular affinity
gives the ideal pressure row

```
p' = (7*c/80) * (3/4 + theta_1/(4*pi)) * (3/4 + theta'/(4*pi)).
```

Each multiplier lies in [3/4,1]; replacing it by 3/4 would silently discard
phase dependence. The conservative bound `0<=p'<=7*c/80` suffices here.

Define D by the following numerical intervals, the fixed support/kinds and
participant channels above, and the valid own-memory updates specified below.
W remains at its supplied form 1/4, capacity one, pressure/phase zero during
the returned word.

| Coordinate | Entry interval | Ideal returned interval |
| --- | --- | --- |
| C capacity | [5/8,15/16] | [83/128,193/256] |
| C form | [1/4,7/16] | [5/16,13/32] |
| C lifted phase | [0,1/16] | [1/32,3/64] |
| C pressure | [0,1/8] | contained in [0,21/256] |
| R form | [1/4,5/8] | [25/64,77/128] |

Every returned interval is contained in its entry interval. This endpoint
calculation is accompanied by the following prefix-domain argument:

1. SHA leaves capacity in [5/16,15/32], below AL's basal 1/2 but above the
   unchanged SHA/NAV minima. NAV reads this capacity, which exceeds the old
   pressure at most 1/8, so its strict handler genuinely changes pressure.
   Its capacity is in [3/8,9/16], pressure in [7/32,21/64] and phase in
   [1/8,3/16]; the input pressure lies within NAV's absolute ceiling one.
2. Both IL actions retain positive form/capacity and reduce pressure to at
   most 21/256. Both UM entries have actual nonempty compatible support,
   form above .05 and capacity above .01. Every intermediate phase gap is
   at most 3/16, below U3's pi/2 bound. One-way updates preserve all other
   incident edges' phase domains.
3. After UM2, capacity exceeds 1/2 and AL entry form is at most 7/16, below
   its strict latent ceiling 1/2. AL's endpoint is at most 9/16, unprojected
   and below EN's .9 ceiling. This is the instantaneous AL chart, not the
   separate finite-episode endpoint chart. EN pressure is below .15.
4. Output EN has receiver form at most 5/8 and pressure zero, matching kinds
   and actual incoming ports. It preserves the participant capacity and phase
   used by C's next UM. No support or participant-channel hypothesis is lost.

Induction therefore gives an ideal admitted return from D to D, conditional
on the specified event inputs and memory validity. The reduced fixed values
are `c*=35/53`, `x*=3/8`, `theta*=1/24`, `z*=9/16`; they are not a closed
orbit of every field. In particular, the memory part of D allows actual
advances: after k returns C has `2+8*k` own events and `1+k` AL activations
with the same origin; R has `1+k` own events. Other nodes retain their own
initial IL. Bounded trace suffixes may expire while lifetime prior coherence
and zero U2 debt persist. SHA writes its own latency fields, NAV clears its
active flag/start/form snapshot, and its transition list and valid durations
advance. Parent or collective history never substitutes for those updates.
The grammar claim is incremental continuation, not standalone U1b closure.

### Why the second UM is a structural discriminator

For n UM occurrences per prefix with the same factors and actual participant
mean one, the ideal capacity return is

```
F_n(c) = (3/4)**n * (3/5)*c + 1 - (3/4)**n.
```

With no UM it tends to zero. With one it has fixed value 5/11, below AL's
basal 1/2. With two it has fixed value 35/53, inside D. A fixed value at least
1/2 requires `(3/4)**n<=5/7`, so two is minimal within this particular family.
This does not compare other coefficients, participant states or operators.
Two actual events also retain different own memory from one retuned event.

From the prepared capacity 15/16, the prospective no-renewal AL capacities
are 9/16 then 27/80; the one-UM values are 43/64, 707/1280, then 12763/25600.
Thus those controls predict rejection on their second and third AL attempts,
respectively. The successful earlier stages must remain recorded; the failed
AL alone rolls back. A failure outside D does not contradict its conditional
two-UM return proof.

### Arithmetic and execution scope

These formulas idealize the represented 1.2/.7 coefficients and shared
circular operations. The engine uses separate binary64 arithmetic and libm
phase functions. Its finite controls cannot prove a uniform infinite-run
error bound. Holding the specified participant rows exact, a returned error
of at most 1/512 in each of the five numerical rows, nonnegative pressure
and separately admitted intermediate prefixes would still preserve D: the
smallest displayed inward margin is 1/64. This is a conditional robustness
bound, not an asserted backend guarantee or permission to relax a live guard.

The prospective controls compare non-dyadic capacity/phase/pressure rows at
absolute tolerance 2^-40 with zero relative tolerance, well below the declared
admission margins. Form/receiver values remain exact dyadic assertions. The
finite control scope is two complete two-UM returns, the two bounded rejection
controls above, and a single final RA on W. The RA observation neither renews
C's capacity nor belongs to a bounded repeated-W state claim.

### Evaluated renewal discriminator

The [three maintenance controls](../tests/operators/test_maintenance_return.py)
were independently reviewed, frozen and evaluated once. Their local record is
`artifacts/maintenance-return-20261103`. All three passed, together with 175
existing synthetic shared-UM, retained-AL and retained-output contracts.
The evaluated engine and fixtures remained byte-identical during the run;
no production law, existing helper or frozen previous result was changed.

| Supplied renewal policy | Completed returns | Actual later outcome |
| --- | --- | --- |
| No UM | One | Second AL rejects below basal capacity, after the declared SHA/NAV/IL prefix |
| One UM per return | Two | Third AL rejects below basal capacity, after its actual UM |
| Two UM per return | Two evaluated | Both returns lie inside D, with live guards and distinct own memories preserved |

The positive control reads the actual three neighbors on every UM and checks
their unchanged channels and support. C forms are exactly 5/16 and 11/32;
the live receiver R forms are 25/64 and 15/32. Its single final RA gives W
form 23/64 and capacity 9/8. C retains its first origin, reaches activation
count three and own event count eighteen; R owns three events and W two
after the separate RA. No successful context is copied between nodes.
The negative invocations preserve their already completed prefixes and leave
no AL occurrence from the rejected call.

All prospective non-dyadic comparisons meet absolute error 2^-40; this does
not establish the uniform perturbation hypothesis used above. The fifteen
reported warnings are the existing zero-pressure IL advisories during
preparation and adjacent-IL pattern advisories. Grammar rejection and the
actual operator maps remain active. No longer run, new parameter set,
historical producer or acquired-support preparation was evaluated.

The result supplies a conditional maintenance mechanism through actual UM,
with complete ideal return and finite implementation correspondence. It
retains one-way interaction, supplied support and event order, unchanged
participant capacity/phase rows, advancing own memory and the separated RA
probe. Releasing the fixed phase reference requires a new return argument
for all participants and the external port. A local phase contraction alone
cannot certify that larger organization.

<a id="reciprocal-maintenance-boundary"></a>
## Reciprocal Coupling and the external relation

The preceding return separates two obligations once Coupling (UM) becomes
bidirectional: keeping its participants mutually compatible and preserving
their relations to the rest of the network. This section retains the same
five-node support, own IL/AL preparation, scalar state, factors and recurring
word, changing only `UM_BIDIRECTIONAL=True`. A second branch appends actual
Resonance (RA) at W after each output EN at R. Its phase coefficient is the
unchanged default `beta=1/(4*pi)`, form mix 1/2 and capacity amplification
1/8. No inter-event flow, support edits, resets or selector are supplied.

### Relative contraction and common phase advance

Let A,U,R share phase q, let C have phase q+d, and let W have phase w.
Use local real lifts with the stated acute differences; primitive phases
remain circular. Write `s=d+1/8` after NAV and

```
h(s) = atan2(sin(s), 3+cos(s)),
H(s) = [h(s)+h(s/2)]/2.
```

One actual bidirectional UM has consensus `q+h(s)`. Its phase push 1/2
gives neighbors `q+h(s)/2` and C `q+[s+h(s)]/2`. The second UM therefore gives

```
d' = (d+1/8)/4,      q' = q+H(d+1/8).
```

A,U,R remain equal in phase because all three receive the same secondary
write; their own histories do not acquire fictitious UM occurrences. Their
capacity and pressure remain one and zero. The preceding capacity, form,
receiver and pressure formulas continue to hold, replacing absolute theta
by the relative gap d. In particular, each pressure multiplier uses the
post-UM target-neighbor gap, s/2 or s/4. W is outside this UM phase write set.

For `0<=d<=1/16`, one has `1/8<=s<=3/16` and

```
h'(s) = (1+3*cos(s))/(10+6*cos(s)),
0 < h'(s) <= 1/4,       0 < H(s) <= 9/256.
```

Thus contraction of d does not stop the advance of q relative to a fixed W.
On this interval `cos(s)>5/9`, so `h'(s)>1/5` and `H(s)>=3/160`.
Starting with all phases zero and omitting RA, the ideal repeated word must
eventually take R--W outside its pi/2 compatibility interval, before any
full phase winding. This is an ideal obstruction, not an evaluated long
trajectory or a uniform infinite-iteration binary64 result. A change of
global phase chart shifts W as well and cannot remove the R--W difference.

Ordinary staged UM admits its target and compatible participants. It does
not promise preservation of every old relation incident to a secondary
phase recipient. The [disjoint-component contract](COUPLING_COMPOSITION.md#connected-coupling-collateral-boundary)
does check its affected boundary, but requires a new requested edge between
two distinct components. It cannot be applied to this already connected
support by merely changing a label. U3 event admission and preservation of
all existing relations are distinct contracts.

### What repeated Resonance closes, and what it does not

Set `e=q-w` at return entry, `y=EPI_W` and `a=nu_f_W`. Since W's only
neighbor is R, the actual RA after output EN has ideal rows

```
e' = (1-beta)*(e+H(d+1/8)),
y' = (y+z')/2,
a' = (9/8)*a.
```

RA leaves W's pressure zero and writes its own occurrence. Its positive
capacity amplification is triggered throughout: R's returned form is at
least 25/64, above the actual 1e-9 trigger. Setting that factor to zero is
outside the current runtime factor contract; suppressing the write would
change the selected mechanism.

Replace the previous fixed W rows and absolute phase rows by relative d in
place of theta, `0<=e<=1/2`, `1/4<=y<=5/8`, `a>=1`, and arbitrary common
circular phase; retain the other domain conditions.
During every prefix R--W has gap at most `1/2+9/256=137/256<pi/2`.
Moreover, `beta>9/137`, so `(1-beta)*137/256<1/2`: the lag returns to its
interval. Convex form mixing returns y to its interval, and W meets its
strict form, pressure, capacity and U3 entry guards. The earlier C/R prefix
proof still applies because W's changing capacity is not a C neighbor input.
All affected edges, including R--W, remain compatible in this domain.

This establishes a conditional ideal return with an **unbounded capacity
coordinate**, not bounded maintenance of the complete numerical state.
After k returns `a_k=(9/8)**k*a_0`. There is no finite upper capacity interval
invariant under this word with positive a. The phase/form bounds do not
control that channel or certify infinite execution in finite arithmetic.
Actual memory advances as before, and W now has `1+k` own events. A/U retain
their initial IL, R its own output ENs, and C its own lifecycle and activation
origin. Neither shared phase nor a common readout pools those histories.

### Prospective finite distinction

Two independent five-node preparations suffice; no earlier continuation
producer is replayed. The first uses the same initial state except for W's
supplied phase `-pi/2+1/128` modulo 2*pi. All edges initially satisfy U3.
After the first SHA/NAV/IL/IL/UM prefix, R moves by `h(1/8)/2>1/128`,
so R--W leaves U3 while C's participant gaps shrink to 1/16. Ordinary UM
commits its documented local action. An actual RA request at W must then
reject without adding history or undoing the already completed UM. This
boundary state is supplied separately; it is not claimed to have been
reached from the zero-phase preparation.

The second preparation starts with zero phases and executes exactly two
complete reciprocal returns, each ending in actual RA at W. C forms are
5/16 and 11/32, R forms 25/64 and 15/32, W forms 41/128 and 101/256, and
W capacities 9/8 and 81/64. The phase predictions use the displayed h,H
and lag recurrences independently of the observed engine response. The
controls retain live support, every participant's channels and own memory,
with exact dyadic assertions and absolute 2^-40 tolerance for the declared
non-dyadic rows. Neither this tolerance nor the analytical bound disables
a runtime guard.

### Evaluated reciprocal participation

The [two reciprocal controls](../tests/operators/test_reciprocal_maintenance_return.py)
were independently reviewed, frozen and executed once in
`artifacts/reciprocal-return-20261103`. Both passed with 61 existing synthetic
U3 and RA-admission regressions: 63 passed, no failures or skips. Production
sources were unchanged, and the evaluated source remained unchanged during
the run. No earlier continuation or acquired-support producer was replayed.

The boundary control observed the admitted UM and its actual secondary phase
writes, local gap 1/16 and C capacity 43/64. R--W exceeded pi/2, so the later
RA rejected atomically. The committed lifecycle and UM remained present;
no RA event or borrowed neighbor UM credit appeared. The positive control
completed exactly two returns, matching all prospective form, capacity and
phase assertions, including W capacity 9/8 then 81/64. C owned 18 events and
three activations with its original origin; R/W owned three events each and
A/U retained their initial IL. Every affected edge passed in that branch.

The run used two preparations, 37 successful targeted occurrences and one
expected rejected RA. Its ten warnings were eight zero-pressure IL advisories
and two adjacent-IL pattern advisories. Hard guards and the actual maps stayed
active. Non-dyadic comparisons met the predeclared 2^-40 tolerance; this does
not establish a uniform backend bound or infinite executable maintenance.

The result closes reciprocal phase participation for this conditional model
while exposing two separate limits: ordinary UM lacks a general incident-edge
preservation guarantee, and repeated RA supplies unbounded capacity growth
here. A stronger phase-preservation contract would address the first, not
balance the second. The program selects the next reusable operator change;
supplied timing/support, autonomous formation and bounded whole-system
maintenance remain distinct open obligations.
