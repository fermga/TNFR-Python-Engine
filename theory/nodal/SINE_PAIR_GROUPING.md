# Phase and joint form-phase grouping

State-derived pair observations, fully acute equilibrium geometry, finite grouping windows and joint-identity episodes; observation does not create support.

Part of [Coarse-graining, coherence geometry and bridge results](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-phase-pairing"></a>

## 14. State-derived phase partners and independent law admission

The pair partition in Sections 7–13 specifies which collective description
is being tested. This section asks a separate question: can the **phase
state itself** identify those pairs without consuming the proposed
partition, support edges, capacities, forms or a label convention? On the
protected doubled-`C5` family the answer is yes. On a general snapshot
the same observation must be allowed to abstain, and a detected partition
still needs independent support and complete-law admission.

### 14.1. A branch-independent comparison of circular distances

For any two fine nodes define their circular distance and squared chord,

\[
d_{\mathbb S^1}(\theta_p,\theta_q)
 =\min_{n\in\mathbb Z}|\theta_q-\theta_p+2\pi n|
 \in[0,\pi],
\]

\[
D_{pq}=|e^{i\theta_q}-e^{i\theta_p}|^2
      =2[1-\cos(\theta_q-\theta_p)]
      =4\sin^2\!\left(\frac{d_{\mathbb S^1}(\theta_p,\theta_q)}2\right).
\]

The function `d -> 2*(1-cos(d))` is strictly increasing on
`[0,pi]`. Thus comparing squared chords gives exactly the same
nearest-partner ordering as comparing circular distances. It requires
no inverse trigonometric function, common phase origin, branch cut or
choice of real phase lifts.

This observation is invariant under common circular rotation, integer
full-turn changes of phase representatives, simultaneous reversal of
phase orientation, and permutation of node labels. The returned
partition is an unordered set of unordered pairs; serialization order
is not an extra physical choice. These are mathematical invariances of
the admitted phase state. Adding a large common floating-point offset
can discard previously represented phase differences; such a changed
capture is not an exact symmetry control.

### 14.2. The protected tube separates true partners

Use the exact winding-one doubled-`C5` target and phase chart of
Section 12,

\[
\theta=\theta_*+c\mathbf1+h\pmod{2\pi},\qquad
\|h\|\le Z_f<r,\qquad
\alpha=\frac{2\pi}{5},\qquad
\alpha+\sqrt2r<\frac\pi2 .
\]

For the two members `p,q` of one target pair their target phases
coincide. Cauchy–Schwarz therefore gives

\[
d_{\mathbb S^1}(\theta_p,\theta_q)
\le |h_q-h_p|
\le\sqrt2\,\|h\|
<\sqrt2r<\frac\pi{10}.
\]

For nodes in different target pairs the target circular distance is
either `alpha` or `2*alpha`. The triangle inequality on the
circle, applied after canceling the common origin, gives

\[
d_{\mathbb S^1}(\theta_p,\theta_q)
\ge d_{\mathbb S^1}(\theta_{*,p},\theta_{*,q})
                -d_{\mathbb S^1}(h_p,h_q)
>\alpha-\sqrt2r>\frac{3\pi}{10}.
\]

Every node consequently has exactly one nearest phase neighbor: its
other constituent. Each choice is mutual. The inequalities also give
the strict chord separation

\[
\max_{\text{within pair}}D_{pq}
<2[1-\cos(\sqrt2r)]
<2[1-\cos(\alpha-\sqrt2r)]
<\min_{\text{between pairs}}D_{pq}.
\]

The target and partition appear here to **prove recovery**, not as
inputs to the nearest-partner observation. The proof applies equally
to the reversed winding. It consumes no capacity equality. In the
unequal-positive-held-capacity family `U_rho` of Section 13, the
same full-state barrier preserves the phase tube for every real time.
Hence the phase-only observation identifies the same constituent pairs
throughout the full trajectory, including possible internal stalls or
tip crossings. This is an all-state identification consequence of
geometric trapping, independent of almost-everywhere recurrence.

No common frequency, internal angular monotonicity or threshold for
declaring a link was needed. The numerical observer need not consume
`r` or either angular threshold: it only compares the actual
phase distances.

### 14.3. A prospective nearest-partner certificate and its abstentions

For a finite captured phase list, let `[L_pq,U_pq]` enclose each
squared chord. A sufficient certificate that `q` is the unique
nearest neighbor of `p` is

\[
U_{pq}<L_{pk}\qquad
\text{for every }k\ne p,q .
\]

When each node has one such certified neighbor, retain the full
partition only if those choices are mutual. A complete mutual map has
no fixed points and partitions the nodes into disjoint pairs. This
does not require selecting a distance cutoff, minimizing a global
matching cost or fitting a grouping scale.

Strict comparison is essential. Equal distances do not specify one
partner, and overlapping enclosures do not prove either equality or
unique ordering. Odd cardinality, unmatched nodes and nonmutual
nearest choices prevent a complete pair partition. In those cases
the observation abstains; it does not break ties using node names,
greedily rematch unused nodes or borrow a partition from graph topology.

For example, the four exact phases `(0,1/10,3/10,1)` lie within
one short circular arc and each has a unique nearest neighbor. The
first two choose one another, the third chooses the second and the
fourth chooses the third. The choices do not define a complete mutual
pairing. Three coincident phases instead give actual nearest-neighbor
ties. Both are admissible circular observations, but neither justifies
an arbitrary choice of pairs.

The geometric theorem establishes exact separation throughout its
admitted family. A fixed-precision implementation still must enclose
the captured distances and prove its strict inequalities; numerical
range limits or wide enclosures can cause abstention. Conversely,
an unambiguous match outside that family is only an instantaneous
observation. It supplies no future partner guarantee by itself.

### 14.4. Phase pairing does not certify the paired support

After the phase observation, the proposed pairs must independently
satisfy the existing complete-replica support contract: every fine node
occurs once, there are no within-pair edges, each active base edge has
all four cross-edges, and the remaining law, capacity and phase-chart
hypotheses hold. This check may inspect edges; the preceding observation
may not. An asymmetric-capacity consumer must retain the correlations
of Section 13 rather than silently invoking an equal-capacity theorem.

A state-only permutation supplies an explicit discriminating control.
Start with the actual doubled-`C5` support and give both members
of pair `i` the same phase `phi_i`, using five distinct circular
phases. Now exchange **only** the phase attributes of nodes
`(0,-)` and `(1,-)`. Leave graph edges, node labels, capacities
and forms untouched. The unique observed zero-distance pairs become

\[
\{(0,+),(1,-)\},\quad \{(1,+),(0,-)\},
\]

with the other three pairs unchanged. Every node still has one unique
mutual phase partner. Yet both new pairs contain live edges, because
the original base edge `0--1` carried all four fine cross-edges.
They fail the no-within-pair-edge contract immediately.

This control works with exact distinct represented phases such as
`phi_i=1287*i/1024`. It does not claim they equal the symbolic
critical target. A phase-only permutation is not a graph relabeling;
relabeling the entire graph and all attributes would preserve admission.
Replacing the observed pairs by the old graph twin classes would hide
the intended incompatibility and is not an allowed repair.

After successful support admission, a target-dependent family report
also needs an explicit order and orientation of the five observed
pairs, and declared lifts of the captured phases. Nearest pairing
alone fixes neither a particular cyclic starting pair nor a chosen
winding sign. Those declarations must be recorded before applying the
target's norm/storage certificate; they are not inferred by quietly
reordering phases until a certificate succeeds.

The result therefore distinguishes identification, law admission and
persistence. It recognizes an organization already present in the
state, verifies whether the supplied support permits its retained
collective description, and uses a separate invariant-tube certificate
for future identification. Conservation protects the admitted phase
geometry; it is not a new force creating a pair, an edge or a material
constituent.

### 14.5. Observation and admission interfaces

The [scale owner](../../src/tnfr/physics/relational_sine_scale.py)
provides `observe_phase_pairs(nodes=..., phases=...)` and
`PhasePairObservation`. This standalone observation reads only
the labeled phase list. Its squared-chord bounds, certified nearest
indices, strict margins and abstention reasons retain the numerical
evidence. The 64-node limit bounds quadratic arithmetic work; it is
neither a physical length scale nor a grouping threshold. Exact ties
recognized from equal absolute raw phase gaps are sufficient tie
controls, not an exhaustive classification of every circular tie.

`assess_sine_state_pairing` and `SineStatePairingAssessment`
then reuse one graph capture. The phase projection supplies the observed
candidate; the existing capacity-aware owner independently checks its
support, law and supplied phase lifts. A caller's `pair_order` must
agree with that observed unordered partition, and is mandatory for
target-dependent family evidence. Without family bounds, capture order
merely arranges the result for display. No automatic unwrapping or
topological substitution is performed.

The wrapper can preserve a valid phase proposal while reporting its
independent collective-law admission as rejected. When its optional
full-state geometry certifies trapping, the separation theorem justifies
`all_time_pairing_persistence_certified`. This flag does not certify
formation, internal circulation or recurrence of the selected state.
Membership in a smaller declared excess/mean family remains separately
reported by the reused owner. The
[state-pairing contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-state-pairing)
owns the public admission and reporting details.

The [replica tests](../../tests/physics/test_relational_sine_replica.py)
separate phase-only comparisons, exact and unresolved ambiguities,
nonmutual choices, complete relabeling, independently refused support
and protected-family admission. These are detached observations and
conditional reports; neither interface evolves the graph.

<a id="sine-replica-acute-critical"></a>

## 15. Exhaustive fully acute equilibria on the doubled cycle

### 15.1. Complete equations and the sector being classified

Take the fixed complete doubled-`C5` graph with its twenty unit
edges and no other edges. Its structural pairs have identical neighbor
sets; pair `i` is adjacent to both constituents of pairs `i-1`
and `i+1` modulo five. This is a fact about the supplied support,
not a partition inferred by the phase observer.

Keep `e=0`, `a=w/pi>0`, `b=w/(beta*pi)>0`, arbitrary
strictly positive held fine capacities and no input or event. With
the fine combinatorial Laplacian `L_f`, define

\[
K=\operatorname{diag}(\nu_p/4)>0,\qquad
S_p(\theta)=\sum_{q\sim p}\sin(\theta_q-\theta_p).
\]

The complete law is

\[
\dot x=aKS(\theta),\qquad \dot\theta=bKL_f x .
\]

Full equilibrium requires **both** rows to vanish. A vanishing form
rate or instantaneous sine balance alone is not this condition.
Restrict the classification to configurations in which every actual
fine-edge principal phase gap lies in `(-pi/2,pi/2)`. Equivalently,
every edge cosine is strictly positive. This is an explicit sector
restriction; it does not follow merely from the model being conservative.

The exact classification is

\[
\boxed{
x_{i,+}=x_{i,-}=m,\qquad
\theta_{i,+}=\theta_{i,-}
 =c+\frac{2\pi k i}{5}\pmod{2\pi},
\quad k\in\{-1,0,1\},}
\]

where `m` is any real form origin and `c` any common circular
phase origin. Neither constant form, member-phase equality nor the
integer winding is assumed in deriving this list.

### 15.2. Zero phase rates force uniform form

At full equilibrium, `b*K*L_f*x=0`. Both `b` and every
diagonal entry of `K` are positive, so `L_f*x=0`. The
fine graph is connected; equivalently,

\[
x^\mathsf TL_f x=\sum_{\{p,q\}\ {\rm fine\ edge}}(x_p-x_q)^2=0
\]

forces equality along every edge and hence `x=m*1`.
Capacity heterogeneity affects the motion away from equilibrium but
cannot change this kernel argument. Strict positivity is essential;
the proof is not transferred to frozen zero-capacity rows.

### 15.3. Member-phase equality follows without a pair chart

For a structural pair `i` form the common neighbor phasor

\[
Z_i=\sum_{q\in N_i}e^{i\theta_q}.
\]

Both constituents have exactly this neighbor set. For either member
`p`,

\[
e^{-i\theta_p}Z_i
=\sum_{q\in N_i}\cos(\theta_q-\theta_p)
 +i\sum_{q\in N_i}\sin(\theta_q-\theta_p).
\]

The zero form rate and positive capacity imply that its imaginary
part vanishes. All four incident edge cosines are strictly positive,
so its real part is positive. Thus `Z_i!=0` and

\[
e^{i\theta_p}=\frac{Z_i}{|Z_i|}.
\]

Applying this equation to the other member proves equality of their
phases on the circle. No midpoint, phase unwrapping, equal-capacity
assumption or preselected synchronized pair state was used. This
complex-number identity concerns the sine critical equations; it
does not invoke the separate native resultant-pressure runtime.

The common member phase may now be denoted `Theta_i`. Its
coincident-member chart is a consequence of equilibrium and acuteness.

### 15.4. Acute sine balance and circular closure give exactly three windings

Let `gamma_i` be the principal phase increment from pair `i`
to `i+1`. Every `gamma_i` belongs to `(-pi/2,pi/2)`.
Each fine node has two neighbors in each adjacent pair, so its remaining
sine-balance equation is

\[
2\{\sin\gamma_i-\sin\gamma_{i-1}\}=0 .
\]

Sine is strictly increasing on the acute interval. Hence every
`gamma_i` equals the same `gamma`. Circular closure gives

\[
\prod_{i=0}^4e^{i\gamma_i}=1,\qquad
5\gamma=2\pi k\quad(k\in\mathbb Z).
\]

The strict acute bound becomes `4*abs(k)<5`. Its only integer
solutions are `k=-1,0,1`, and reconstructing the consecutive
phases gives the claimed family.

Conversely, choose any of those integers, any `m,c` and any
positive held fine capacities. Uniform form gives zero phase rates.
At every fine node the two forward sine terms and two backward sine
terms cancel exactly, giving zero form rates. All fine edge gaps are
`0` or `+/-2*pi/5` and are strictly acute. Thus every listed
state is a full equilibrium and every fully acute equilibrium is listed.
The converse uses exact angles, not small computed residuals.

These are three signed winding families relative to a declared base
orientation. Reversing that orientation interchanges `+1` and
`-1`. The two twists also have identical edge cosines and storage,

\[
E_{*,k}=20\beta[1-\cos(2\pi k/5)] .
\]

Consensus has zero storage and the two twists have equal positive
storage. These facts do not select an equilibrium's occurrence under
the conservative law. They are not three physical species or spatial
pentagons. The support specifies relational adjacency, while the
classified geometry is a pattern of circular phases on that support.

### 15.5. Exact families, captured phases and observer abstention are distinct

Consensus is a valid full equilibrium although every node has nine
equally close phase neighbors. The phase-only observer therefore
abstains rather than selecting the support's structural pair classes.
There is no contradiction: the classification uses support to prove
which complete equilibria exist; the observer asks what phase alone
identifies. At either exact nonzero twist, the only zero-distance
neighbor is the other constituent, so the phase observation identifies
the pairs. Nearby protected captures can also have unique partners
without being critical states.

There is a useful exact arithmetic boundary for the current capture
format. All captured raw radians are rational numbers: integers,
Fractions and finite binary values have that property. If a captured
adjacent phase difference represented an exact nonzero twist, it would
have to satisfy

\[
\theta_q-\theta_p=2\pi(n+k/5),\qquad
n\in\mathbb Z,\quad k=\pm1 .
\]

The left side is rational. The right side is irrational because
`n+k/5` is a nonzero rational and `pi` is irrational. Thus no
such raw-radian capture is **exactly** a nonzero member of this critical
family. Exact symbolic turns and rounded radians are different inputs.
Similarly, captured rational phases equal on the circle must have
identical raw values: a nonzero rational difference cannot equal
`2*pi*n`.

It follows that, within a certified fully acute captured sector,
exact full equilibrium is equivalent to exact uniform captured form
and exact equality of all captured raw phases. This yields separate
valid source verdicts:

- Nonuniform captured form excludes full equilibrium on this connected
  positive-capacity support, independently of any acute-sector decision.
- Uniform form and identical captured raw phases certify exact consensus.
- Uniform form, certified fully acute edges and nonidentical captured raw
  phases exclude exact critical membership by the preceding classification
  and arithmetic argument.
- Uniform form outside, or not certified inside, the acute sector does
  not receive an equilibrium classification from this theorem.

The third verdict does not measure the size of a response, exclude
proximity to a protected target, or imply numerical or dynamical
instability. A rounded twist can be extremely close to equilibrium
while failing exact membership. No tolerance promotes that approximate
state to the symbolic critical family.

### 15.6. The strict sector boundary and what has not been selected

Acuteness cannot be removed from the classification. As an exact
symbolic control, set uniform form, give one constituent phase `pi`,
and give every other fine node phase zero. Every edge sine is zero,
so both complete rows vanish for any positive capacities. The two
members of the selected structural pair have different phases, and
its incident `pi` gaps have negative cosine. This is an equilibrium
outside the classified sector. Replacing `pi` by a rounded binary
value would not be the same exact counterexample.

The theorem classifies neither nonacute equilibria nor arbitrary
nonstationary organizations. The pulse and protected families studied
earlier are not required to be equilibria at every time. Existing
acute-barrier results can be reused at the listed targets with their
own hypotheses; this enumeration supplies no attraction theorem,
global stability classification or choice among consensus and twists.
It also supplies no route from an arbitrary preparation into the
two-sided invariant protected family.

What has been derived is the complete list of fully acute critical
phase geometries **conditional on this law and supplied support**.
The mechanism selecting a preparation, changing support or producing
a particular organization remains a separate question.

### 15.7. Shared classification and captured-state evidence

The [scale owner](../../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_equilibria` and
`SineReplicaEquilibriaAssessment`. It retains one full source
capture and validates the declared structural pair order against the
complete support, with arbitrary positive fine capacities. Classification
does not require a supplied source pair-phase chart or a successful
phase-only grouping.

The integer condition `4*abs(k)<5` generates the candidate
windings. Each `SineReplicaEquilibriumTarget` reuses the
[phase geometry owner](../../src/tnfr/physics/phase_cycle_geometry.py)
for exact-turn reconstruction and symbolic odd-sine cancellation.
Only those algebraic properties are consumed; its separate phase-law
interpretation is not transferred to the complete model.

The report keeps this exhaustive symbolic classification separate from
the captured edge-cosine enclosures, exact form/phase equalities and
source equilibrium verdict. A nonacute source does not invalidate the
conditional family theorem, and a small source residual does not prove
target membership. Nonuniform form provides a global exclusion through
the phase row; otherwise an uncertified acute sector remains outside
the captured-state classification unless exact consensus is present.

The [replica tests](../../tests/physics/test_relational_sine_replica.py)
check independent fine-row cancellations, the complete integer list,
capacity independence, exact consensus, rounded nonzero twists and
the nonacute boundary. The reader does not choose a winding, repair
a capture, evolve a trajectory or replace the independent phase-only
observer with topological pair recovery.

<a id="sine-pairing-transition"></a>

## 16. Local onset and loss of a support-compatible observed grouping

### 16.1. Differentiate the observation using the complete law

The squared-chord observation of Section 14 is

\[
D_{pq}=2[1-\cos(\theta_q-\theta_p)] .
\]

Along a smooth solution of the complete sine law its exact derivative is

\[
\boxed{
\dot D_{pq}
=2\sin(\theta_q-\theta_p)(\dot\theta_q-\dot\theta_p).}
\]

This formula is circular and does not differentiate an arbitrary wrapping
branch. The phase observer itself still consumes only phase. Its rate,
however, needs the existing complete phase law. For fine degree `d_p`,
write the exact rate numerator

\[
n_p=\frac{w}{\beta}\frac{\nu_p}{d_p}(L_f x)_p,
\qquad \dot\theta_p=\frac{n_p}{\pi}.
\]

Then

\[
\dot D_{pq}
=\frac{2}{\pi}\sin(\theta_q-\theta_p)(n_q-n_p).
\]

The form, capacities, actual neighbors and structural clock enter through
`n`. They cannot be reconstructed from the phase-distance snapshot
alone. These are rates of the existing observation, not a new force,
controller or event law.

For a node `p` comparing partners `q` and `r`, define
the preference margin

\[
M_{p;q,r}=D_{pr}-D_{pq}.
\]

A positive margin favors `q`. If it is zero at the preparation,
a strictly positive derivative proves that this particular tie is
resolved toward `q` for all sufficiently small positive times.
To infer a complete pairing, every tied choice and every other
competitor must also be accounted for.

### 16.2. One frozen represented preparation

Use the fixed complete unit doubled-`C5` support with structural pairs

\[
(0,1),\ (2,3),\ (4,5),\ (6,7),\ (8,9).
\]

Each adjacent base-pair connection contains all four fine edges and
there are no other edges. Keep `e=0`, `beta=1`, all held
capacities one, the structural clock, and no input, clipping or event.
The normalized zero-loss repository reference has `w=1`; retaining
`b=w/pi>0` in the formulas also displays the clock factor.

Freeze `d=u=1/8` and the exact initial values

\[
\theta(0)=(0,d,2d,3d,1,1,2,2,3,3),
\]

\[
x(0)=(u,-u,u,-u,0,0,0,0,0,0).
\]

These dyadic values are represented exactly. Their `d,u` symbols
name the fixed preparation, not scanned parameters. This state is not
an equilibrium or an assumption of fully acute fine edges: for example
the live edge `0--8` has gap `3>pi/2`. The full smooth sine
law applies; the native resultant-pressure law and the acute
equilibrium classification are not substituted for it.

Each structural pair has zero form sum. Every fine node's four
neighbors consist of two complete adjacent pairs, whose total form
is therefore zero. Since the fine degree is four,

\[
L_f x(0)=4x(0),\qquad
\dot\theta(0)=b\,x(0)
=(\omega,-\omega,\omega,-\omega,0,0,0,0,0,0),
\quad \omega=bu>0 .
\]

These are the actual initial phase rows of the full nonlinear system.
Form continues to evolve through `xdot=a*K*S(theta)`; no frozen
form or straight-line phase extrapolation is being treated as a
trajectory.

### 16.3. All initial competitors and the two transverse ties

The full phase span is `3<pi`, so every initial circular distance
is the ordinary absolute difference of these displayed radians. Put
`F(s)=2*(1-cos(s))` for `0<=s<=pi`. The complete initial
nearest sets and closest outsider distances are:

| Node or nodes | Initial nearest set | Nearest circular distance | Smallest distance to an outsider |
| --- | --- | --- | --- |
| `0` | `{1}` | `d` | `2d` |
| `1` | `{0,2}` | `d` | `2d` |
| `2` | `{1,3}` | `d` | `2d` |
| `3` | `{2}` | `d` | `2d` |
| `4,5` | The other member of `(4,5)` | `0` | `1-3d=5d` |
| `6,7` | The other member of `(6,7)` | `0` | `1=8d` |
| `8,9` | The other member of `(8,9)` | `0` | `1=8d` |

Every comparison against an outsider has strictly positive chord
margin at least

\[
g_0=F(2d)-F(d)=2(\cos d-\cos2d)>0 .
\]

For the last three pairs their outsider margins are larger:
`F(5d)>F(2d)>g_0` or `F(8d)>F(5d)`. Thus the table
checks all competitors, not just the adjacent ties. At time zero the
complete phase-pair observer abstains because nodes `1` and `2`
each have two equally near choices.

The exact chord rates for the three tied-length gaps are

\[
\dot D_{01}(0)=\dot D_{23}(0)=-4\omega\sin d,\qquad
\dot D_{12}(0)=4\omega\sin d .
\]

Define the two margins favoring the structural partners,

\[
M_1=D_{12}-D_{10},\qquad M_2=D_{21}-D_{23}.
\]

They obey

\[
M_1(0)=M_2(0)=0,\qquad
\dot M_1(0)=\dot M_2(0)=8bu\sin d>0 .
\]

For the normalized `w=1` reference and frozen `u=d=1/8`,
the common margin derivative is exactly
`sin(1/8)/pi`. Positivity follows from `0<1/8<pi`,
not from a small sampled trajectory or a fitted tolerance.

### 16.4. A complete local transition follows from smoothness

All finitely many distances and preference margins are smooth along
the actual solution. The strict outsider margins therefore remain
positive on some two-sided neighborhood of time zero. For the two
ties,

\[
M_j(t)=8bu\sin(d)\,t+o(t)\qquad(j=1,2).
\]

There exists a common `epsilon_t>0` such that the two margins
are positive for `0<t<epsilon_t` and negative for
`-epsilon_t<t<0`, while all outsider comparisons retain their
initial signs.

For every sufficiently small **positive** time, the unique nearest map
is consequently

\[
0\leftrightarrow1,\qquad
2\leftrightarrow3,\qquad
4\leftrightarrow5,\qquad
6\leftrightarrow7,\qquad
8\leftrightarrow9 .
\]

This is a complete mutual matching. It equals the actual structural
pair partition, so its independent support check succeeds: no matched
pair contains an edge, and adjacent matched blocks have all four
cross-edges. Initial within-pair phase gaps are `d,d,0,0,0<pi`;
continuity also preserves their local pair charts after shrinking the
same unspecified neighborhood if necessary. The retained collective
law thus applies to this observed grouping.

For every sufficiently small **negative** time, the first four choices
instead are

\[
0\longmapsto1,\qquad
1\longmapsto2,\qquad
2\longmapsto1,\qquad
3\longmapsto2 .
\]

The remaining three pairs retain their unique mates. Every node has
a unique nearest neighbor, but the choices of `0` and `3`
are not mutual. No complete mutual-nearest partition exists. A greedy
rematching would change the observation contract rather than complete
this proof.

The conclusion concerns exact mathematical matching at every
sufficiently small time on either side. Fixed arithmetic enclosures
may abstain at times extremely close to the tie, where strict margins
are too small to resolve. The theorem supplies neither a numerical
value of `epsilon_t` nor a certified finite endpoint, forecast or
sampled trajectory. A first derivative alone cannot supply any of those.

There is also a qualitative robustness consequence for **full initial
states**, with the law, support and capacities held fixed. Choose any two
nonempty closed time windows strictly inside the proved negative and
positive intervals. On each window, every node's chosen nearest partner
is separated from every competitor by a strictly positive margin.
The minimum of these finitely many continuous margins on the compact
windows is positive. Joint continuity of the smooth flow in time and
initial state therefore gives an open neighborhood of the supplied full
preparation for which all those inequalities persist on both windows.
Every preparation in this neighborhood has the same nonmutual backward
choices and complete support-compatible forward matching.

This does not preserve the exact simultaneous tie at time zero. Locally
the two initial tie equations reduce to
`theta_0-2*theta_1+theta_2=0` and
`theta_1-2*theta_2+theta_3=0`, two independent conditions.
Perturbations may separate their crossing times; there is no single
common transverse section asserted for the whole neighborhood.
Neither the windows nor the preparation radius have numerical bounds
here. The argument establishes robustness of the strict observations
away from the crossing, not an implemented finite-window certificate.

### 16.5. Form reversal controls the direction with the same phase geometry

Now reverse every form coordinate at the preparation while preserving
the exact phases, capacities and support:

\[
\widetilde x(0)=-x(0),\qquad
\widetilde\theta(0)=\theta(0).
\]

The initial phase observation, its exact ties and all outsider margins
are unchanged. Linearity of the phase row in form gives
`dot(theta_tilde)(0)=-dot(theta)(0)`. Both tied margin derivatives
are therefore `-8bu*sin(d)<0`. Complete mutual pairing holds
locally on the negative side and nonmutual choices locally on the
positive side for this control.

This reversal is also an exact identity of the complete conservative
flow. If `(x(t),theta(t))` solves
`xdot=a*K*S(theta)`, `thetadot=b*K*L_f*x`, then

\[
\widetilde x(t)=-x(-t),\qquad
\widetilde\theta(t)=\theta(-t)
\]

solves the same equations with the reversed initial form. Direct
differentiation proves both rows. Smooth uniqueness identifies it with
the control solution, so its entire local phase-distance history is
the original history reversed in time. This relies on `e=0` and
the absence of time-directed forcing or events; it is not a transfer
to a positive-loss runtime.

The two supplied states thus have identical instantaneous phase
geometry and opposite local directions of grouping change. Form is
necessary predictive information here. Conservation does not choose
one direction, and the emergence of a distance ordering is not
creation of a law that causes it.

### 16.6. What has appeared and what has not

This result proves local onset and local loss of **observability** of
a support-compatible collective grouping under the existing unforced
fine dynamics. The nodes and all twenty edges were already present.
No edge is added, no operator event is selected, and no constituent
capacity is changed.

Nor is this first entrance into the earlier protected family.
That family is invariant in both time directions, so a point outside
it cannot enter it at a finite time. In particular the tie preparation
cannot lie in the protected winding-one tube, whose strict separation
already identifies partners at every time. The present local interval
therefore supplies no acquisition of permanent trapping, long-term
pair identity, attracting waveform or material constituent.

It does establish a concrete same-law mechanism: retained form fixes
phase velocity, phase velocity changes nearest-distance margins, and
those margins can resolve or destroy a complete compatible observation.
A finite observation window or a stronger formation claim would require
its own independently declared prediction and additional proof.
Section 17 supplies the former at an explicitly frozen budget.

### 16.7. Implementation and evidence

`assess_sine_pairing_transition` in the
[shared scale owner](../../src/tnfr/physics/relational_sine_scale.py)
retains one complete sine comparison and the existing phase-only
observation. It consumes
`SineExchangeComparison.phase_rate_numerators()` from the
[comparison owner](../../src/tnfr/physics/relational_sine_comparison.py);
the same exact rational phase row also serves resultant kinematics.
The report records chord-rate bounds, exact nearest groups, strict
outsider margins and separate backward/forward conclusions.

For an exact tied group, it preserves the common factor
`2*sin(abs(gap))/pi` and compares the exact rational coefficients
`sign(gap)*(n_q-n_p)`. A resolved factor sign and strict coefficient
order can certify a split without subtracting independently rounded
derivative enclosures. Overlapping distances do not establish a tie;
zero or unresolved factors and tied first derivatives remain
unavailable. Complete nearest choices are required before a matching
or nonmutuality conclusion, and a complete matching still receives
its own fixed-support admission.

The [public contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pairing-transition)
keeps `certified_time_horizon` unavailable. The report does not
assign a numerical robustness radius, install the phase observation
as a controller or evolve the graph. The
[independent replica tests](../../tests/physics/test_relational_sine_replica.py)
check the frozen complete-field rates, all competing partners, the
form-reversal control and the distinction between a local mathematical
ordering and finite-precision observation.

<a id="sine-pairing-window"></a>

## 17. Whole-box prediction throughout a fixed observation window

### 17.1. The frozen question and complete state

Retain the full conservative normalized-sine law, fixed unit doubled-`C5`
support, held capacities one and `beta=w=1` from Section 16.
No input, clipping, event or native resultant-pressure row is added.
Let `theta*` and `x*` be the exact ten-node preparations there:
`d=u=1/8`,

\[
\theta^*=(0,d,2d,3d,1,1,2,2,3,3),\qquad
x^*=(u,-u,u,-u,0,0,0,0,0,0).
\]

For each `sigma in {+1,-1}`, admit the entire closed preparation box

\[
\mathcal B_\sigma=
\left\{(x_0,\theta_0):
 |x_{0i}-\sigma x_i^*|\leq\rho,\quad
 |\theta_{0i}-\theta_i^*|\leq\rho\quad\hbox{for every }i
\right\},
\qquad \rho=2^{-20}.
\]

The phase errors are specified in the declared real lifts. The flow
and final observation remain circular. All twenty errors are independent;
form sums, pair means, phase equality within the last three pairs and
the two initial nearest ties are not constraints on either box.
The support and capacities are exact. Reversing form exchanges the two
boxes, including their uncertainty sets.

Freeze the same future structural-time window for both predictions,

\[
t_-=1/128,\qquad T=1/64,\qquad t_-\leq t\leq T.
\]

The claim is the original complete mutual matching for `B_+` and the
nonmutual nearest map of Section 16 for `B_-`, for **every** source in
the relevant box and **every** time in this closed window. The budgets
are fixed before their inequalities are evaluated.

### 17.2. A global full-field remainder, without a reduced trajectory

The complete fine rows are

\[
\dot x_i=\frac1{\pi d_i}
 \sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
\dot\theta_i=\frac1{\pi d_i}
 \sum_{j\sim i}(x_i-x_j),
\qquad d_i=4 .
\]

They imply the global bounds

\[
|\dot x_i|\leq\frac1\pi,\qquad
|\ddot\theta_i|
\leq\frac1{\pi d_i}\sum_{j\sim i}
 (|\dot x_i|+|\dot x_j|)
\leq\frac2{\pi^2}.
\]

These bounds use the actual evolving form and all neighbors, and hold
at arbitrary phase configurations. In particular they do not freeze
the phase-to-form feedback or assume fine-edge acuity. The smooth field
has complete solutions: form grows at most linearly on each finite
time interval and the phase row then has a finite integral.

At the center preparation `L_f x*=4*x*`. An arbitrary initial form
error in the box changes the phase rate by at most `2*rho/pi`,
including both the node's own error and its neighbors' errors. Hence
the exact integral remainder gives, for every source and `0<=t<=T`,

\[
\boxed{\left|
\theta_i(t)-\theta_i^*-\frac{\sigma x_i^*}{\pi}t
\right|
\leq E(t):=\rho+\frac{2\rho}{\pi}t+\frac{t^2}{\pi^2}.}
\]

The three terms retain initial phase error, initial form error and
full nonlinear acceleration, respectively. The central affine
expression is a reference used in an enclosure; it is not asserted to
be a nonlinear trajectory. No local Taylor solver, state truncation
or omitted internal coordinate enters this bound.

The same calculation has a reusable conservative form on any admitted
fixed unit support with held nonnegative capacities. Put
`a=w/pi`, `b=w/(beta*pi)`, and let the independent initial form
and phase radii be `r_xi,r_thetai`. For node `i` define

\[
M_i=a\nu_i,\qquad
V_i=b\nu_i\left(r_{xi}+\frac1{d_i}\sum_{j\sim i}r_{xj}\right),
\qquad
B_i=b\nu_i\left(M_i+\frac1{d_i}\sum_{j\sim i}M_j\right).
\]

If `v_i` is the complete phase rate at the center capture, then

\[
|\theta_i(t)-\theta_{0i}^{\rm center}-v_i t|
\leq r_{\theta i}+V_i t+\tfrac12B_i t^2
\quad(t\geq0).
\]

The center rate uses the shared phase-law numerator, not a fitted
response. This extension requires `e=0` and the declared absence of
inputs/events; the form bound is not transferred unchanged to a
positive-loss law.

### 17.3. Exact rational margins for the whole window

Use the rational bounds `25/8<pi<22/7`, with the coarser lower bound
`3<pi` to simplify the remainder. Since `E(t)` increases for nonnegative
time, every phase error on `[0,T]` is at most

\[
\overline E
=\rho+\frac{2\rho T}{3}+\frac{T^2}{9}
=\frac{8483}{301989888}.
\]

Write `omega=u/pi`. The central first-four phase gaps, in node order,
are

\[
d-2\sigma\omega t,\qquad
d+2\sigma\omega t,\qquad
d-2\sigma\omega t.
\]

Each actual gap differs from its central value by at most
`2*Ebar`. Their common strictly positive lower bound is

\[
d-\frac{2uT}{3}-2\overline E
=\frac{18669277}{150994944}>0 .
\]

Thus the order of the first four phases is unchanged for either box
throughout the window. More generally the total real phase span is at
most

\[
3+\frac{2uT}{3}+2\overline E
=\frac{453189923}{150994944}
<\frac{25}{8}<\pi .
\]

All phase separations in this proof are consequently also their
circular distances; no hidden wrapping transition can alter the
nearest ordering.

For the two initially tied choices, the difference between the
undesired and desired distances, with `desired` chosen according to
the sign `sigma`, is bounded below by

\[
\begin{aligned}
m_{\rm tie}
&=4u\,t_-\,\frac7{22}-4\overline E\\
&=\frac{938879}{830472192}
>\frac1{1024}>0.
\end{aligned}
\]

The `4*Ebar` term is the worst error of the three-coordinate
combination at each tie, retaining the repeated central node's
coefficient two. The bound applies even when a source in the box has
no exact initial tie.

For the first four nodes, every competitor outside the singleton or
tied group in Section 16 starts at distance at least `2*d`,
whereas the chosen partner starts at distance `d`. Bounding the
two central motions and the two gap errors gives

\[
m_{\rm outside}
=d-\frac{4uT}{3}-4\overline E
=\frac{9232093}{75497472}
>\frac1{1024}.
\]

For each of the last three pairs, its internal distance can be as
large as `2*Ebar`; those members have not been kept synchronized.
Every outsider starts at distance at least `5*d`. Their
preference margin is at least
`5*d-2*u*T/3-4*Ebar`, which is larger than
`m_outside`. These inequalities account for every competitor of
every node, not just the two central ties.

It follows that every selected nearest distance has uniform preference
margin greater than `g=1/1024`. This also gives a lower bound for the
actual squared-chord observer. If circular distances satisfy
`0<=s<t<=pi` and `t-s>=g`, then

\[
\begin{aligned}
F(t)-F(s)
&=4\sin\!\left(\frac{t+s}{2}\right)
       \sin\!\left(\frac{t-s}{2}\right)\\
&\geq4\sin^2(g/2)=2(1-\cos g)>0,
\qquad F(z)=2(1-\cos z).
\end{aligned}
\]

Indeed the midpoint angle lies between `(t-s)/2` and
`pi-(t-s)/2`, so its sine is at least the sine of the
half-separation. Thus the whole-box prediction has a strict
quantitative chord margin, not only an ordering of unwrapped angles.

### 17.4. The two reserved predictions and their limits

For every initial state in `B_+` and every `t in [1/128,1/64]`,
the exact nearest map is

\[
0\leftrightarrow1,\quad
2\leftrightarrow3,\quad
4\leftrightarrow5,\quad
6\leftrightarrow7,\quad
8\leftrightarrow9.
\]

All choices are strict and mutual. The complete fixed-support
admission independently succeeds for this partition. Its within-pair
phase gaps also remain below `pi`, so the retained collective chart
is available. This does not make its internal differences zero.

For every initial state in `B_-` on the same window, the nearest
indices are

\[
(1,2,1,2,5,4,7,6,9,8).
\]

All choices are strict, but `0->1` and `3->2` are not mutual.
There is no complete mutual-nearest matching under the declared
observation contract, even though the support itself is unchanged.
Both conclusions cover independent perturbations of every fine
coordinate; neither follows from a center trajectory or a single
evaluated endpoint.
Some sources in these boxes may already have their window's nearest
map at time zero. The whole-box conclusion is therefore a uniform
future observation prediction, not a claim that every source undergoes
one simultaneous onset at a specified time.

The original local reader still supplies only an existential
neighborhood. The present theorem instead proves its separately
frozen finite-window prediction using explicit global remainders.
It gives no conclusion outside `[1/128,1/64]`, no observation-noise
budget beyond the stated preparation uncertainty, and no eventual
capture by the earlier two-sided invariant family. A numerical
observer still has to resolve a margin with its own arithmetic.
The result is finite organizational robustness under a supplied
complete law, not new support, a controller, permanent formation or
physical identification.

### 17.5. Shared consumer and numerical evidence

`assess_sine_pairing_window` in the existing
[scale owner](../../src/tnfr/physics/relational_sine_scale.py) consumes a
single comparison capture, mandatory full vectors of form and phase
error radii, and an explicit `0<=window_start<window_end`.
It requires zero form loss and retains the captured support, capacities
and structural clock. The source box is supplied analysis evidence;
the reader does not authenticate a laboratory preparation.

`SinePairingWindowAssessment` records the initial intervals, the
per-node form-speed, initial phase-rate-error and acceleration upper
bounds, the resulting remainders and the complete phase window.
Pair differences reuse the exact shared rate-numerator difference
before applying the common time interval. This preserves information
that subtraction of independent whole-node phase windows would lose.
Where the absolute gap enclosure lies below mathematical `pi`,
monotonicity gives tight chord bounds from certified scalar cosine
endpoints. Elsewhere the shared interval cosine is used without
inferring a new phase lift.

Only strict comparison with every competitor certifies a node's
nearest choice throughout the box and window. The report then
distinguishes complete mutual matching, certified nonmutuality and
unavailable bounds, followed by separate paired-support admission
where a complete matching exists. Wide intervals do not refute a
physical trajectory or the proposed grouping. The existing local
reader keeps its absent numerical horizon.

The [window contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pairing-window)
and [independent replica tests](../../tests/physics/test_relational_sine_replica.py)
retain these distinctions, the frozen two-box control, all competing
nodes, invalid domains and unresolved-bound controls. This is a
full-field analytic enclosure on the existing owner, with no
trajectory producer, finite-difference fit or enlarged solver budget.

<a id="sine-joint-identity-window"></a>
## 24. Joint form-phase identification during autonomous exchange

### 24.1. A storage-defined observation, not an extra dynamical rule

For the fixed positive storage scale `beta`, define

\[
J_{ij}=(x_i-x_j)^2+2\beta[1-\cos(\theta_i-\theta_j)],
\qquad d_\beta(i,j)=\sqrt{J_{ij}}.
\]

The map `z_i=(x_i,sqrt(beta)*cos(theta_i),sqrt(beta)*sin(theta_i))`
embeds the form and circular phase into Euclidean three-space, with
`d_beta(i,j)=||z_i-z_j||`. Thus `d_beta` is a metric on this state
space: it vanishes precisely for equal form and equal circular phase,
and it inherits the triangle inequality. `J` is its square, not itself
a metric. Both terms use the existing configured storage scale in the
same declared, currently nondimensional coordinate chart. If form units
are assigned, `beta` must carry their square for `d_beta` to have form
units; this bookkeeping does not establish a laboratory bridge for the
law's coefficients or clock. The scale is inherited from the declared
storage, not fitted to identify a desired partition or proved to be a
unique physical geometry.

The observation is invariant under node relabeling, common form shifts,
common phase rotations and independent full-turn changes of phase
representatives. It consumes form and phase; held support and capacity
remain separate model premises. As in Section 14, an inferred pair
requires unique mutual nearest partners. A tie must remain a tie.
Such instantaneous identification does not establish sufficient state
for a future law or select a new support or hierarchy event.

### 24.2. Exact reference identity survives a phase-only collision

Keep the complete conservative normalized-sine law and the synchronized
`K2,2` preparation of Section 23: common positive capacity, fixed support,
equal initial forms and a supplied lift `0<|Delta_0|<pi`. The full
trajectory retains identical members within each pair. For every time,

\[
\boxed{J_{\rm within}=0,\qquad
J_{\rm cross}=D(t)^2+2\beta[1-\cos\Delta(t)]
=2E_c=4\beta\sin^2(\Delta_0/2)=:J_*>0.}
\]

This is an identification consequence of the existing inherited storage
identity, not another conservation postulate. Every node has its other
constituent as its unique nearest partner throughout the entire reference
motion. The observation can recover that partition without reading its
labels or the supplied support; those premises are used to prove the
claim, not to resolve an observed distance tie.

The P2 libration crosses `Delta=0`. At that instant all four phases
coincide, so every phase-only distance is zero and phase-only pairing
must abstain. But then `D^2=2E_c>0`: the joint observation retains
exactly the same strict pair distinction. At the form turning points
`D=0`, the nonzero phase separation instead supplies it. Form and phase
exchange their contributions without erasing the joint distinction.
At `Delta_0=0` with equal forms the reference has `J_*=0`, four
identical observed states and no uniquely identified pairing. This
degenerate reference is not certified by continuity from positive
amplitude.

### 24.3. A full-field perturbation bound over a declared window

Now allow independent preparation errors in the form and phase of
**all four** nodes. The perturbed state need not preserve either
pair's internal synchronization. Both trajectories use the same
`K2,2` graph, common capacity `nu>0`, `beta,w>0`, `e=0`, clock,
and no input, event or clipping. Specify real phase lifts for the
initial comparison and continue them by the smooth phase row.

The error norm is taken in the eight real coordinates

\[
Z=(x_0,\ldots,x_3,\sqrt\beta\theta_0,\ldots,
\sqrt\beta\theta_3),\qquad
r_0^2=\|Z(0)-Z_*(0)\|_2^2.
\]

For componentwise preparation radii `f_i,p_i>=0`, the sufficient
initial bound is `r_0^2<=sum_i(f_i^2+beta*p_i^2)`. Each component
can vary independently within its declared interval. A common-origin
or phase-lift choice must be specified before this bound is evaluated;
small circular distance alone is not a declaration of small error in
an arbitrary lift.

Put `c=w*nu/(pi*sqrt(beta))`. On this degree-two regular graph the
Jacobian of the **full fine field** in the scaled real lifts is

\[
DF(Z)=c\begin{pmatrix}0&-L_{\cos}(\theta)/2\\L/2&0\end{pmatrix},
\]

where `L_cos` is the symmetric edge Laplacian with signed coefficients
`cos(theta_j-theta_i)`. Both `L/2` and `L_cos/2` have Euclidean
operator norm at most two. For example the absolute row sum of the
signed matrix is at most two after division by the degree, and its
symmetry bounds the spectral norm. Equivalently
`-L <= L_cos <= L` as quadratic forms and the largest eigenvalue
of the `K2,2` Laplacian is four. The block Jacobian norm is therefore
at most `L_*=2c`, uniformly in form and phase.

This proves a global Lipschitz bound in the scaled **lift** coordinates,
and hence global continuation and the full-field difference estimate

\[
\|Z(t)-Z_*(t)\|_2\le r_0e^{L_*t}\le r_0e^{L_*T}=:r_T
\qquad(0\le t\le T).
\]

It does not assert that the embedded circular-state vector field has
an equally bounded Jacobian independent of form. The embedding is
used only for its observation error:

\[
\|z_i-z_{*,i}\|^2
=(x_i-x_{*,i})^2+
4\beta\sin^2[(\theta_i-\theta_{*,i})/2]
\le(x_i-x_{*,i})^2+\beta(\theta_i-\theta_{*,i})^2.
\]

For any two nodes the reverse triangle inequality and
`||e_i-e_j||<=sqrt(2)*sqrt(||e_i||^2+||e_j||^2)` give

\[
|d_\beta(i,j)-d_{\beta,*}(i,j)|\le\sqrt2\,r_T.
\]

Writing `s=sqrt(J_*)`, throughout the whole window we consequently have

\[
\max_{\rm within}d_\beta\le\sqrt2r_T,\qquad
\min_{\rm cross}d_\beta\ge\max(0,s-\sqrt2r_T).
\]

In particular the following strict, prospective inequality guarantees
the same mutual nearest pairs for **every** preparation in the error
ball or declared component box and every time in the window:

\[
\boxed{J_*>8r_0^2\exp\!\left(
\frac{4w\nu T}{\pi\sqrt\beta}\right).}
\]

The inequality is sufficient, not necessary. A failed or unresolved
bound cannot by itself prove loss of identity. Zero uncertainty gives
the exact reference result. For nonzero uncertainty the estimate
establishes a finite-window guarantee, not attraction, nonlinear
orbital stability or permanent identification. The held graph and
capacities also do not emerge from this error estimate.

### 24.4. A fixed full-exchange control and genuine degeneracies

Before evaluating the finite report, fix the source to `K2,2` with
`C=1/2`, phases `(0,0,1/4,1/4)`, `beta=nu=1`, `e=0` and effective
`w=1`. Set `T=10` in its structural clock and give each form and
phase component radius `10^-6`. Then `r_0^2<=8*10^-12` and
`J_*=2*(1-cos(1/4))`. These are preparation specifications, not
fitted errors or observed response curves.

The existing P2 period bound gives
`T_pulse<=pi^2/cos(1/8)<10`; for example
`pi<22/7` and `cos(1/8)>=127/128` already give the strict upper
bound `61952/6223<10`. Thus the chosen window includes one full
reference exchange, including its phase-only collision. The strict
identity inequality can also be checked without trajectory evaluation:
`J_*>1/40`, while its right-hand side is less than
`64*10^-12*3^14<1/40`. These conservative inequalities use
`sin(y)>=2y/pi` for `0<=y<=pi/2`, `3<pi<22/7`, and
`exp(40/pi)<exp(14)<3^14`. A finite interval report can retain
sharper margins without changing the declared source or budget.

The zero-amplitude control has no reference separation. The separately
declared large-error control, with every radius `1/8` and the same
nonzero-amplitude reference, includes a preparation with all forms
`1/2` and all phases `1/8`. All four observed node states then
coincide at the initial time. The entire box therefore cannot have
the strict pair-identification property, independently of how sharp
an error bound is. This explicit counterexample must not be generalized
to every preparation budget for which the sufficient inequality fails.

### 24.5. Shared observation and window admission

`observe_joint_pairs(nodes=..., forms=..., phases=..., storage_scale=...)`
in the [scale owner](../../src/tnfr/physics/relational_sine_scale.py) observes
the actual signed forms and circular phases. It reuses strict mutual
nearest-partner admission and may abstain; it does not read a proposed
partition or use support to resolve a tie.

`assess_sine_joint_pairing_window` separately admits the complete
`K2,2` reference and its supplied pairs, phase lifts, `form_error_bounds`,
`phase_error_bounds` and `window_end`. The returned
`initial_scaled_error_squared`, `lipschitz_upper` and
`propagated_scaled_error_squared_upper` retain the full-field error
provenance. `identification_budget_margin` applies the strict squared
inequality; the report also requires strict interval nearest-partner
comparisons. `reference_identity_certified` is separate from
`whole_window_identity_certified`. A numerical abstention cannot erase
the exact reference theorem or become evidence of actual perturbed
failure.

The report's `reference_period_bounds` reuse the existing P2 period
owner; `window_covers_reference_period` concerns that reference only,
not a period or a recurrence deadline for perturbed trajectories. No
graph is advanced, and no snapshot is represented as the evolved
endpoint. The [independent algebra controls](../../tests/physics/test_relational_joint_identity.py)
check the circular metric identity, conserved reference separation,
complete scaled-lift Jacobian and both observation-error factors.

<a id="sine-joint-grouping-comparison"></a>
## 25. The frozen phase transition does not change the joint pairing

### 25.1. Compare two observations of the same complete preparation

Keep the exact doubled-`C5` graph, law, clock and the two form-reversed
sources from Sections 16–17, with `e=0`, `w=beta=nu=1` and
`d=u=1/8`:

\[
\theta^*=(0,d,2d,3d,1,1,2,2,3,3),\qquad
x^*=(u,-u,u,-u,0,0,0,0,0,0).
\]

For `sigma=+1,-1`, retain **all twenty** independent initial radii
`rho=2^-20` around `(sigma*x*,theta*)` and the original observation
window `[1/128,1/64]`. The supplied structural pairs are
`(0,1),(2,3),(4,5),(6,7),(8,9)`. No parameter, horizon, support,
pressure law or event schedule is adjusted for the joint observation.
The `K2,2` constant-separation result is not applied to this graph.

The joint observation of Section 24 is
`J_pq=(x_q-x_p)^2+beta*D_pq`, where
`D_pq=2*(1-cos(theta_q-theta_p))` is the existing phase-only chord.
Differentiation along the **complete** fine field gives

\[
\boxed{\dot J_{pq}=
2(x_q-x_p)(\dot x_q-\dot x_p)
+2\beta\sin(\theta_q-\theta_p)(\dot\theta_q-\dot\theta_p).}
\]

Both rows are necessary. In particular a measured or computed phase
rate alone does not determine this derivative. The sine form row
depends on the full neighbor phase currents; the phase row depends
on the full form gradients. These are observation rates, not new
terms in either evolution law.

### 25.2. A different joint partition is already strictly present

Write `F(s)=2*(1-cos(s))`. At either source center,

\[
J_{02}=J_{13}=F(2d),\qquad
J_{01}=J_{12}=J_{23}=4u^2+F(d).
\]

The closest opposite-form competitors are farther by

\[
h_0=4u^2+F(d)-F(2d)\ge d^2=1/64>0.
\]

Indeed `u=d` and
`F(2d)-F(d)=integral_d^(2d) 2*sin(s) ds <=3d^2`.
The remaining first-four comparison `0--3` has the same form cost
and a larger phase gap `3d`. Every node outside those four is at
phase distance at least `5d` from them. Here
`F(2d)<=4d^2` while `F(5d)>10d^2`, using the chord bounds below.
The last three pairs have exactly zero internal joint distance and
strictly positive distance to every outsider. All phase gaps lie
below `pi`. Thus every competitor has been checked, and the unique
mutual joint pairs at both centers are

\[
\boxed{\mathcal C=
\{(0,2),(1,3),(4,5),(6,7),(8,9)\}.}
\]

The form-reversed source has the same squared form differences, so
this observation is unchanged. All its joint-distance rates reverse:
the initial form rates stay unchanged, whereas form differences and
phase rates change sign. In particular `Jdot_02=Jdot_13=0` at both
centers, because their form differences and phase-rate differences
are zero. These zero derivatives do not create a tie: their nearest
margins are strictly positive. More generally the exact conservative
reversal of Section 16 gives `J_tilde(t)=J(-t)`; it does not imply
that an initially strict nearest relation must change at time zero.

### 25.3. The same joint pairs persist on the complete frozen boxes

Reuse the full-field phase bound of Section 17. For every source in
either box and `0<=t<=T=1/64`,

\[
|\theta_i(t)-\theta_i^*-(\sigma x_i^*/\pi)t|\le E,
\quad E=\rho+2\rho T/3+T^2/9
=\frac{8483}{301989888}.
\]

The same complete field has `|xdot_i|<=1/pi`. Integration, including
each independent initial form error, therefore gives the additional
bound

\[
\boxed{|x_i(t)-\sigma x_i^*|\le B:=\rho+T/3
=\frac{16387}{3145728}.}
\]

This bound does not freeze form or assume a common form error within
any pair. The phase enclosure uses the original central phase rates
and nonlinear remainder; it is not an extrapolated trajectory. The
span estimate from Section 17 remains below mathematical `pi` on
this entire interval, so absolute phase gaps equal circular gaps.

For `0<=s<=pi`, the elementary chord inequalities are

\[
\frac25s^2\le\frac4{\pi^2}s^2\le F(s)\le s^2,
\]

where the first inequality uses `pi^2<10`. They follow from
`2y/pi<=sin(y)<=y` on `0<=y<=pi/2`. For the desired first-four
pairs `(0,2),(1,3)`, the reference forms and central phase rates
coincide within each pair. Consequently

\[
J_{\rm desired}\le U_1:=4B^2+(2d+2E)^2.
\]

Every remaining first-four competitor has initial form difference
`2u` and phase gap at least `d`. Independently bounding both
coordinates gives

\[
J_{\rm competitor}\ge L_1:=(2u-2B)^2+
\frac25\left(d-\frac{2uT}{3}-2E\right)^2.
\]

Both lower gap bounds inside the squares are strictly positive.
They must not be squared without that check. Their exact margin is

\[
L_1-U_1=
\frac{33345408988471}{37999121855938560}
>\frac1{2048}>0.
\]

For a first-four node and an outsider among the last six, the phase
gap is at least `g_2=5d-u*T/3-2E>0`, giving
`J_competitor>=(2/5)*g_2^2`. Its margin over `U_1` is greater
than `L_1-U_1`. For each of the last three desired pairs,
`J_desired<=U_2:=4B^2+4E^2`; every outsider again has phase gap
at least `g_2`, giving a still larger margin. This includes other
members of the last six, whose initial interpair gap is at least one.

Thus all eighty directed nearest-partner comparisons have the required
strict sign. For **every** source in either frozen box and every time
`0<=t<=1/64`, the joint nearest map is

\[
\boxed{(2,3,0,1,5,4,7,6,9,8).}
\]

In particular this proves the new observation on the original frozen
window `[1/128,1/64]`. The same estimates also cover its preparation
and intervening interval: the identified pairs cannot disappear and
reappear before that window. This is an analytic corollary of the
unchanged budgets, not a replacement or retiming of the earlier
phase-only forecast.

### 25.4. Observation change, collective state and formation are distinct

Under phase-only observation the positive box has the structural
matching on its window, while the negative box has the nonmutual
nearest map proved in Section 17. Under the joint observation both
boxes already have the same strict partition `C` initially and
retain it throughout the interval just proved. Therefore that
specific phase-only onset or loss is **not** onset or loss of joint
pair identity under the specified joint criterion.

This comparison neither invalidates the old phase-only theorem nor
declares a uniquely correct physical NFR metric. It identifies the
different information retained by the two observations. Neither
distance ordering creates support, selects an operator or proves
permanent formation.

The joint pairs also do not automatically inherit the structural
pair law. The proposed pairs `(0,2)` and `(1,3)` contain existing
fine edges, so they fail the strict no-internal-edge replica admission.
More decisively, even the broader independent-swap criterion of
Section 18 fails. For `(0,2)`,

\[
N(0)\setminus\{2\}=\{3,8,9\},\qquad
N(2)\setminus\{0\}=\{1,4,5\}.
\]

Their interchange does not preserve the fixed support. Proximity in
the joint observation consequently cannot discard their distinct
attachment identities; an unordered pair state alone is not the
generic exact dynamical quotient. The mixed-state and boundary-current
owners retain the relevant internal and environmental information.
Recognition of a stable finite-window pattern and sufficiency of its
proposed collective evolution remain separate obligations.

### 25.5. A projection of the existing window, not another forecast

`SinePairingWindowAssessment.joint_observation()` in the
[shared scale owner](../../src/tnfr/physics/relational_sine_scale.py) returns
`SineJointPairingProjection` from the existing capture, error box and
declared window. It combines the actual form-speed tube with the
already computed phase chord bounds; it neither rereads an evolved
graph nor changes the original phase-only result. The source
`source_joint_distance_rate_bounds` consume both captured fine rows.

`initial_observation`, `initial_box_candidate_pairs` and `candidate_pairs`
refer respectively to the exact center, the whole initial box and the
whole reported window. `same_pairing_as_initial_box` compares the last
two only when both are certified complete pairings. Equality there
does not establish persistence across an unobserved intervening gap;
the stronger result in Section 25.3 has its own analytic proof.
`support_admission_status` retains the existing strict replica
admission and must not be read as an exhaustive symmetry test. The
[independent algebra and rational controls](../../tests/physics/test_relational_joint_grouping.py)
check all competing nodes, the full-row reversal, the source support
obstruction and the fixed-box bound without evaluating a trajectory.

<a id="sine-joint-boundary-acquisition"></a>
## 26. Acquiring a joint pairing at its exact observation boundary

### 26.1. A separately declared critical preparation

Keep the unit doubled-C5 support of Sections 16-17, its ten labeled nodes,
and the complete conservative normalized-sine law with held
`e=0`, `w=beta=nu_i=1`. There are no inputs, events or clipping. Every fine
node has degree four, and the full rows are

\[
\dot x_i=\frac{S_i}{4\pi},\qquad
\dot\theta_i=\frac{(Lx)_i}{4\pi},\qquad
S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i).
\]

Use the same phases and the same joint observation `J` as Section 25, but
declare a **new mathematical preparation**, chosen by an exact observation
boundary rather than a search over responses:

\[
d=\frac18,\quad F(s)=2(1-\cos s),\quad
u_c=\frac12\sqrt{F(2d)-F(d)}>0,
\]
\[
\theta^*=(0,d,2d,3d,1,1,2,2,3,3),\qquad
x^\sigma=\sigma(u_c,-u_c,u_c,-u_c,0,0,0,0,0,0),
\quad\sigma\in\{+1,-1\}.
\]

The positive square root exists because `F` is strictly increasing on
`(0,pi)`. This source is supplied, not autonomously selected. It does not
replace the earlier `u=1/8` source, independent error boxes, fixed horizon
or evaluated evidence. In particular its exact trigonometric form amplitude
is not a rounded graph attribute.

At either preparation, the five distances

\[
J_{01}=J_{12}=J_{23}=J_{02}=J_{13}=J_*:=F(2d)
\]

are exactly equal, because `4*u_c^2=F(2d)-F(d)`. The complete initial
nearest sets are

\[
\{1,2\},\quad\{0,2,3\},\quad\{0,1,3\},\quad\{1,2\},
\quad\{5\},\{4\},\{7\},\{6\},\{9\},\{8\}.
\]

All other comparisons have a strict gap. For `(0,3)` the excess over
`J_*` is `F(3d)-F(d)`. Between a first-four node and a last-six node the
phase gap is at least `5d`, so the excess is at least `F(5d)-F(2d)`.
The last three synchronized pairs have distance zero; their outsiders
have phase gap at least `5d`. The chord inequalities from Section 25 give

\[
F(3d)-F(d)\ge\frac{13}{5}d^2,\qquad
F(5d)-F(2d)\ge6d^2,\qquad F(5d)\ge10d^2.
\]

These are uniform positive margins for every initially untied competitor.
Consequently the only local ordering question is how the five tied edges
separate under the complete law.

### 26.2. Both rows determine the crossing direction

At the positive preparation `Lx=4x`, so `theta_dot=x/pi`. Write the first
four sine sums as

\[
\begin{aligned}
S_0&=\sin2d+\sin3d+2\sin3,\\
S_1&=\sin d+\sin2d+2\sin(3-d),\\
S_2&=-\sin2d-\sin d+2\sin(1-2d),\\
S_3&=-\sin3d-\sin2d+2\sin(1-3d).
\end{aligned}
\]

Apply the full joint-distance derivative from Section 25:

\[
\dot J_{ij}=2(x_j-x_i)(\dot x_j-\dot x_i)
 +2\sin(\theta_j-\theta_i)(\dot\theta_j-\dot\theta_i).
\]

Define the three dimensionless coefficients

\[
\begin{aligned}
A&=5\sin d-\sin3d+2[\sin(3-d)-\sin3],\\
B&=2\sin d-2\sin2d+2[\sin(1-2d)-\sin(3-d)],\\
C&=5\sin d-\sin3d+2[\sin(1-3d)-\sin(1-2d)].
\end{aligned}
\]

The tied-edge rates at the positive source are exactly

\[
\boxed{\frac\pi{u_c}
 (\dot J_{01},\dot J_{12},\dot J_{23},\dot J_{02},\dot J_{13})
 =(-A,B,-C,0,0).}
\]

All three coefficients are strictly positive, by elementary inequalities:

* The triple-angle identity gives
  `5*sin(d)-sin(3d)=2*sin(d)+4*sin(d)^3>0`.
  Since `pi/2<3-d<3<pi`, sine decreases between `3-d` and `3`;
  the additional difference in `A` is positive.
* The same identity and the sine-difference formula give

  \[
  C=4\sin(d/2)[\cos(d/2)-\cos(1-5d/2)]+4\sin^3d>0.
  \]

  Here `d/2=1/16<1-5d/2=11/16<pi`, so both displayed terms are
  positive. This retains the form/phase cancellation in the smallest
  tied-edge response; the phase row alone cannot justify its sign.
* `sin(3/4)>=3/4-(3/4)^3/6=87/128>2/3`.
  Also `0<pi-23/8<1/3`, using `3<pi<22/7`, and hence
  `sin(23/8)=sin(pi-23/8)<1/3`.
  Finally `sin(2d)-sin(d)<=d` by the derivative bound for sine. Therefore
  `B>2*(2/3-1/3-d)=5/12>0`.

No ordering between `A`, `B` and `C` is needed. Reversing all forms leaves
the form row unchanged and negates the phase row. Both terms of every
`J_dot` therefore reverse, while every initial `J` stays unchanged.

### 26.3. One-sided acquisition and qualitative full-state robustness

The field and the joint distances are smooth. A tied margin with a strictly
positive derivative becomes positive on a sufficiently short positive-time
interval. There are finitely many comparisons, and the initially untied
ones have the positive margins just proved. Thus some `tau>0` exists such
that, for **every** `0<t<tau`, the positive source has nearest map

\[
\boxed{(1,0,3,2,5,4,7,6,9,8),}
\]

whereas the negative source has nearest map

\[
\boxed{(2,2,1,1,5,4,7,6,9,8).}
\]

For the positive source, edges `01` and `23` decrease while `12` increases
and `02,13` have zero first derivative. This resolves every tied choice in
favor of the original structural pairs. For the reversed source, `12`
decreases while `01,23` increase; nodes 0 and 3 select the zero-rate
alternatives 2 and 1. This map is not a complete mutual pairing. The last
three pairs remain the unique nearest choices by their initial strict gaps.

The exact conservative reversal `(x(t),theta(t)) ->
(-x(-t),theta(-t))` also shows that the positive orbit has the nonmutual
map on `-tau<t<0`. It crosses from that region, through the declared ties,
into the structural mutual-pairing region. This is local acquisition of the
specified joint observation under the complete autonomous law, not merely
a phase-distance transition or a choice among initially unique joint pairs.

The assertion has qualitative robustness in **all twenty fine coordinates**.
For any compact interval `[a,b]` with `0<a<b<tau`, all desired margins
along either reference are uniformly positive. Continuous dependence of the
full flow yields an open initial-state neighborhood preserving its respective
nearest map throughout `[a,b]`. There is no restriction to synchronized
perturbations, pair sums or the original one-dimensional form family.
The exact simultaneous ties at time zero need not survive a perturbation.

More strongly, choose any small positive `a<tau` on the positive reference.
Its state at `-a` has a strict nonmutual map and its state at `+a` has the
strict structural map. Both strict endpoint observations persist for an
open neighborhood of that earlier state, by the same continuous-dependence
argument. Thus nearby full states also undergo a change between these
observation regions, although their individual tie-crossing times and order
need not coincide. These are existential intervals and neighborhoods;
none is a certified numerical horizon, error radius or permanent lifetime.

### 26.4. What is acquired, and what remains supplied

The positive partition passes the strict replica support contract: its
pairs have no internal edges, neighboring blocks have complete unit
bipartite support, capacities are common, and every within-pair interchange
is a graph automorphism. Their initial absolute within-pair phase half-gaps
are `d/2,d/2,0,0,0`, all strictly within the local midpoint chart; `tau` may
be shortened to keep that admission throughout the crossing. The exact
collective-state result of Sections 7 and 18-19 therefore applies with its
retained internal form/phase variables.
The first two pairs are not synchronized at preparation, so their newly
strict observation does not authorize discarding those internal variables.
The negative map has no complete mutual pairing to admit as such a partition.

The support symmetry and its sufficient quotient existed before the crossing.
The dynamics makes that partition strictly recognizable by the supplied
joint observation; it does not create the support, the quotient degrees of
freedom, an operator-selection law or a physical constituent. No invariant
prepared family has been entered from outside, and no attraction or permanent
maintenance is asserted. The result is consequently compatible with the
conservative formation boundary stated at the beginning of this note.

No new execution law or observer is necessary. The existing joint observation
and full-row derivative owners already retain the required information.
Their represented-real admission must not relabel a rounded `u_c` as the
exact source, infer an equality from overlapping intervals, or promote this
local theorem to a finite-window certificate. Any such numerical claim would
require its own prospectively declared enclosure and budget. The
[independent symbolic controls](../../tests/physics/test_relational_joint_boundary.py)
check the exact critical relation, complete fine-row coefficients and every
tied ordering; the proof above supplies their inequality and local-flow scope.
The [outward full-field controls](../../tests/physics/test_relational_joint_boundary_bounds.py)
separately bound the symbolic amplitude, all initially untied comparisons
and the complete joint rates through shared rational kernels. Their rounding
control demonstrates why a represented approximation is not the exact tie.

<a id="sine-joint-recurrent-episodes"></a>
## 27. Repeated finite episodes of joint identification

This is a consequence of the local crossing in Section 26 and the existing
[full-state invariant-volume recurrence theorem](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
It uses the **same** complete conservative normalized-sine law: fixed unit
doubled-C5 support, `e=0`, held `nu_i=w=beta=1`, structural time, and no
forcing, events, clipping or changed mobility. All twenty fine coordinates
are retained, with phases on the torus. No trajectory is evaluated and no
recurrence assertion is inferred from a finite numerical return.

### 27.1. Identity, loss and a common acquisition interval

Let `Phi_t` denote the complete flow on
`M=R^10 x (R/(2*pi*Z))^10`. Define two open subsets of that state space:

* `P` consists of states whose strict joint nearest map is the structural
  matching `(1,0,3,2,5,4,7,6,9,8)`.
* `N` consists of states whose strict joint nearest map is
  `(2,2,1,1,5,4,7,6,9,8)`, the nonmutual map in Section 26.

Both predicates use exactly the joint distances `J_ij` already defined.
Each is a finite conjunction of strict continuous inequalities
`J_i,p(i)<J_ij` for all other candidates `j`. Hence `P` and `N` are open
and disjoint. **Identity** here means membership in `P`; its loss means
nonmembership. Visiting `N` establishes a stronger, strict alternative
observation, rather than an unavailable or inconclusive numerical report.

Write `z_c` for the positive exact critical preparation. Section 26 gives
`tau>0` such that `Phi_t(z_c)` lies in `N` for `-tau<t<0` and in `P` for
`0<t<tau`, with the local pair midpoint chart admitted on these intervals.
Choose `a>0` sufficiently small, with `a<tau/4`, and set

\[
z_-:=\Phi_{-a}(z_c).
\]

Then `z_-` lies strictly in `N`, while its entire image over the positive
interval `[2a,3a]` lies strictly in `P`. Compactness of that time interval
and continuous dependence on all fine coordinates give an open neighborhood
`U` of `z_-` with compact closure and a bounded open matching neighborhood
`V` such that

\[
\boxed{\overline U\subset N,\qquad
\Phi_s(\overline U)\subset V\subset P
\quad\text{for every }s\in[2a,3a].}
\]

The neighborhoods can be chosen inside the admitted pair chart during
these initial and matching windows. The compact image tube has a common
strictly positive nearest-partner margin, although no numerical value is
claimed. The same `a,U,V` work for every preparation in `U`, including
independent perturbations of all twenty coordinates. Every such state is
nonstationary already, since a stationary state cannot move between the
disjoint sets `N` and `P`.

The midpoint chart is an additional collective-coordinate admission on
these windows, **not** part of the identity predicate `P`. A longer matching
episode may leave that chart without losing its joint-distance matching.

### 27.2. One finite invariant ambient family

The recurrence theorem needs a finite invariant measure, not just a local
crossing. Its hypotheses can be admitted here with an explicit loose slab.
There are twenty unit edges, and every fine degree is four. The conserved
storage and weighted mean therefore take the forms

\[
E=\frac12\sum_{\{i,j\}\in\mathcal E}(x_i-x_j)^2
  +\sum_{\{i,j\}\in\mathcal E}[1-\cos(\theta_j-\theta_i)],
\qquad m=\frac1{10}\sum_i x_i.
\]

For the closed form box `|x_i|<=1/4`, with **arbitrary circular phases**,

\[
E\le\frac12\,20\left(\frac12\right)^2+2\,20
 =\frac{85}{2}<43,\qquad |m|\le\frac14<\frac12.
\]

Thus the whole box times the phase torus lies strictly inside

\[
\mathcal S=\{z:E(z)\le43,\ -\tfrac12\le m(z)\le\tfrac12\}.
\]

At the critical source, `u_c<d=1/8`, since
`4*u_c^2=F(2d)-F(d)<4d^2`. By choosing `a` smaller if necessary, then
shrinking `U`, its compact closure lies in `|x_i|<1/4`. Therefore
`overline(U)` lies in the interior of `S`. The form box itself need not be
invariant: `S` is the invariant family. These numerical ceilings are
analysis bounds on supplied preparations, not new parameters of the law.

The source theorem applies without a change of model. This is a finite
connected simple unit graph with positive held capacities; both complete
rows use the same constant degree/capacity mobility. Storage and the
weighted form mean are conserved. The full divergence vanishes because
the form row depends only on phase and the phase row only on form.
Consequently `S` is compact, the flow exists in both time directions, and
it preserves finite positive product measure

\[
d\mu=dx_0\cdots dx_9\,d\mathrm{Haar}_{\mathbb T^{10}}.
\]

No bounded interval of unwrapped phase lifts or unproved measure on a
fixed-energy surface is being introduced. Since `U` is full-dimensional
and open with compact closure, `0<mu(U)<infinity`.

### 27.3. Infinitely many distinct finite matching episodes

Fix the single sampling increment `h=4a` and write `T=Phi_h`. First take
**any** `z` in `U` that returns to `U` infinitely often under `T`. This is
an explicit deterministic premise: there are integers

\[
0=n_0<n_1<n_2<\cdots,\qquad n_k\longrightarrow\infty,
\qquad \Phi_{n_kh}(z)\in U.
\]

Put `r_k=n_kh`. Each return is a state of the **same** autonomous system;
it is not a reset or a fresh draw of initial data. The common transit gives

\[
\Phi_{r_k}(z)\in N,\qquad
\Phi_t(z)\in V\subset P\quad
\text{for every }t\in[r_k+2a,r_k+3a].
\]

Also `r_(k+1)-r_k>=h=4a>3a`. Hence each guaranteed matching window lies
strictly between two nonmutual returns. Define the matching-time set

\[
\mathcal I_z=\{t>0:\Phi_t(z)\in P\}.
\]

It is open. Let `I_k` be the connected component containing
`[r_k+2a,r_k+3a]`. Since neither bounding return lies in `P`,

\[
\boxed{
[r_k+2a,r_k+3a]\subset I_k\subset(r_k,r_{k+1}).
}
\]

These components are pairwise distinct, bounded intervals, each of
duration at least `a>0`. There are infinitely many of them. Every return
to the open set `N` also has a nonempty time neighborhood in `N`, so the
selected episodes are separated by actual intervals of strict alternative
identification, not merely an unresolved tie at one instant.

More generally **every** positive-time component of `I_z` is bounded:
an unbounded component would contain all sufficiently late times and
contradict the unbounded sequence of returns to `N`. Only the selected
infinite subfamily has the common lower duration bound `a`; no such bound
is asserted for other episodes. This argument supplies no upper lifetime
bound common to preparations, bound on the next return or exact period.
The constants `a,U,V` and the recurrence times have existence proofs here,
not numerical certificates.

Everything so far follows for each state satisfying the stated return
premise. The existing finite-measure recurrence theorem, applied to the
fixed map `T` and measurable subset `U` of `S`, supplies that premise for
`mu`-almost every state in `U`.

Thus, for almost every preparation in this nonempty open full-state
acquisition neighborhood, the structural joint observation repeatedly
appears, persists for a positive interval, is lost, and reappears. It does
not eventually become permanent for those preparations. This is a scoped
lifetime result for a noninvariant observation; it neither contradicts nor
enters the separately protected two-sided invariant identity families.

### 27.4. Measure, interpretation and integration boundaries

The exceptional set is null in the stated ambient twenty-dimensional
product measure. An initial probability law supported in `U` and absolutely
continuous with respect to that measure inherits the result with probability
one. The dynamics has not selected such a preparation law. No pointwise
recurrence conclusion follows for the exact critical source, its earlier
point `z_-`, an individual captured graph, a fixed-energy or fixed-mean
preparation, or a synchronized or finitely sampled family solely from
their inclusion in `S` or `U`.

The recurring identity is the **same specified matching**, not a proof that
nodes, edges, quotient coordinates or physical constituents are repeatedly
created and destroyed. Its support and sufficient state remain supplied;
only the joint observation changes. The conclusion does not establish a
common pulse, periodic waveform, attraction, dissipation or a clock selected
by synchronization. It does not transfer to the native argument-pressure
law, positive-loss sine dynamics or another reciprocal mobility without
their own measure and lifetime premises.

No additional runtime or report is needed. The existing recurrence reader
admits a family and explicitly leaves individual nonstationary recurrence
unavailable; the joint observer reports a captured state's distances rather
than unknown future return times. The
[recurrence controls](../../tests/physics/test_relational_sine_resonance.py)
check complete-law invariants, finite-family admission and that chosen-state
boundary. The [ambient-family integration control](../../tests/physics/test_relational_joint_recurrence.py)
independently checks the declared box/slab and existing family API without
asserting a numerical recurrence sequence. The
[joint crossing controls](../../tests/physics/test_relational_joint_boundary.py)
and [outward bounds](../../tests/physics/test_relational_joint_boundary_bounds.py)
check the local mechanism reused here. Finite tests validate these premises
and their implementation, not the infinite recurrence sequence itself.
