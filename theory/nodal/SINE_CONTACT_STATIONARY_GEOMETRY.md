# Stationary and energy geometry of the actual sine contact support

<a id="contact-stationary-protocol"></a>
## 1. Frozen analytical question and complete marked chamber

This declaration precedes numerical response evaluation. It fixes the next
analytical question after [actual contact](SINE_ORGANIZATION_CONTACT.md):
whether the admitted joined support permits a stationary organization in
the original complete acute chamber. It does not evolve the contacted
source or replace its complete law.

- Retain the twenty labelled nodes `A_(k,s),B_(k,s)`, with `k=0,...,4`
  and `s in {+,-}`. Each old doubled C5 has all four edges between successive
  pairs. Add the actual internal `A_0` edge and the twelve actual edges
  from `A_0` to `B_0,B_1,B_4`. There are 53 edges. Retain their generated
  positive conductances, held post-word capacities, zero forcing and the
  original structural clock from the [contact protocol](SINE_ORGANIZATION_CONTACT.md#organization-contact-protocol).
- Use exactly the [conservative weighted-form/bare-phase completion](SINE_PAIR_COUPLING.md#collective-coupling-weighted-law)
  with `e=0,w=beta=1`: `x_dot=M*S_U(theta)`,
  `theta_dot=M*L_W*x`, where `M_ii=nu_i/(pi*d_i)>0` and `d_i` is the
  bare degree. Actual support and conductances remain held, including
  member-to-boundary association. No pair synchronization is assumed.
- Orient every old cycle block from pair `k` to pair `k+1 mod 5`.
  For each of its four fine edges set branch `b_e=0` when `k<4` and
  `b_e=-1` on the closing block `4 -> 0`. Orient the internal edge from
  `A_(0,+)` to `A_(0,-)` and set its branch to zero. Orient all cross
  edges from A to B: their branches are zero for `B_0,B_1` and one for
  `B_4`. These orientations and branches cover every support edge.
- With real phase lifts define
  `delta_e=theta_head-theta_tail-2*pi*b_e`. The declared chamber requires
  `-pi/2<delta_e<pi/2` for every one of the 53 edges. Quotient only the
  common phase translation, using `sum theta=sum theta_*` if a representative
  is needed, where `theta_*(A_(k,s))=theta_*(B_(k,s))=2*pi*k/5`.
  Keep all fine coordinates and the full branch vector, rather than only
  the two constituent winding observations.
- Derive the full stationary equations first. Use exact current balances,
  strict acute-angle monotonicity and the connected-chamber method from
  [marked-cycle information](ORGANIZED_CONTACT_STATIONARY_SEARCH.md#marked-cycle-information-and-connected-chambers).
  Any pair reduction must follow from those full equations and their
  phase-domain symmetries. No fixed-weight dynamical pair closure is assumed.
- The bounded outcome is either a proved stationary target with a strict
  relative-storage minimum or a proved obstruction in this exact chamber.
  No numerical root finder, optimizer, trajectory, parameter search or
  precision retry is authorized by this declaration. Static checks may
  independently verify the support, branch inventory and algebraic identities;
  they cannot replace the proof.

Stationary exclusion in this chamber would not imply winding loss, the
absence of nonstationary organizations, or the absence of stationary states
in other phase chambers. A stationary candidate would not establish source
entry, attraction, maintenance from the actual preparation or a new NFR.

<a id="contact-stationary-full-equations"></a>
## 2. Full stationary equations and source chamber admission

Let `U,W,nu` retain the actual contact values. The bare graph is connected,
all conductances are positive, and all held capacities are positive. Every
node has positive bare degree. Consequently the diagonal mobility `M` is
invertible and the full twenty-node stationary equations are exactly

\[
L_Wx=0,\qquad S_U(\theta)=0.
\]

The first equation implies constant form because

\[
x^TL_Wx=\sum_{\{i,j\}\in U}W_{ij}(x_j-x_i)^2,
\]

and positivity propagates equality along the connected support. No
conductance has been replaced by one, averaged across a pair or discarded.
The remaining stationary phase problem depends on bare support. This
separation holds at stationarity under this complete law; it does not make
the full weighted evolution independent of member-to-boundary association.

Define the bare phase potential and its gradient by

\[
V_U(\theta)=\sum_{e\in U}(1-\cos\delta_e),\qquad
\nabla V_U=-S_U.
\]

The common phase direction is neutral. The conserved lifted means with
weights `rho_i=d_i/nu_i` exclude a nonzero rigid common form or phase drift:
`sum rho_i*x_dot_i=sum rho_i*theta_dot_i=0`. Thus an equilibrium modulo
these two common translations is an ordinary stationary state. A
nonstationary organized motion with changing internal coordinates is a
different possibility.

The declared chamber is nonempty: the common reference `theta_*` has old
forward gaps `alpha=2*pi/5`, internal and zero-pair contact gaps zero, and
cross gaps `+alpha,-alpha` at B pairs one and four. All are strictly acute.
The actual post-word family belongs to the same chamber. The existing
[contact phase bound](SINE_ORGANIZATION_CONTACT.md#organization-contact-receiver-exit)
gives `|theta_i(0)-theta_*,i|<=a_0=5/1536`. Every old or cross reference
gap therefore changes by at most `2*a_0=5/768`, and

\[
\alpha+\frac5{768}<\frac\pi2.
\]

The internal-edge gap is at most `5/768`. These strict bounds retain each
listed branch, rather than inferring chamber membership from two winding
values alone. The joined graph has cycle-space dimension `53-20+1=34`;
the complete branch vector also retains the mixed and fine-member cycles.
Branch changes induced solely by integer changes of vertex lifts describe
the same circular chamber and do not enlarge this cycle class.

<a id="contact-stationary-acute-uniqueness"></a>
## 3. Acute phase uniqueness and stationary pair synchronization

Let `C` denote the open chamber from Section 1 and let `C_bar` denote its
closed version, obtained by replacing every strict acute inequality by
`-pi/2<=delta_e<=pi/2`, with the same branches. The closure extension is
proved separately here; no trajectory or numerical response is used to
change the frozen question.

Both sets are convex on real lifts. For `theta,phi in C_bar`, orient each
edge as in the declaration and take their gradient difference. Exact
summation by parts gives

\[
(\theta-\phi)^T[\nabla V_U(\theta)-\nabla V_U(\phi)]
=\sum_{e\in U}
 (\delta_e(\theta)-\delta_e(\phi))
 [\sin\delta_e(\theta)-\sin\delta_e(\phi)].
\]

Sine is strictly increasing on the whole closed interval
`[-pi/2,pi/2]`. Every summand is nonnegative, and it is positive whenever
the two edge gaps differ. Connectedness implies that all gaps agree only
when `theta-phi` is a common constant. Hence there is at most one
stationary phase configuration in `C_bar` modulo common phase. This
conclusion does not assume that a stationary point exists.

In the open chamber the Hessian is the positive cosine Laplacian and is
strictly positive on relative phase directions. At an acute boundary some
edge cosines may vanish. The preceding strict monotonicity argument still
applies there and does not infer a positive Hessian from those zero values.

Each independent interchange of the two members of one pair preserves
bare support, all the declared branch constraints and the potential.
On old or cross blocks it permutes equal-branch edges. At the internal
`A_0` edge it reverses the zero-branch gap. Thus a stationary phase point
would be carried to another stationary point of the same closed chamber.
Choose the representative with `sum theta=sum theta_*`. Interchange
preserves this sum, so the uniqueness result forces it to fix that point.
Every stationary pair therefore has equal lifted member phases:

\[
\theta_{A,(k,+)}=\theta_{A,(k,-)}=A_k,\qquad
\theta_{B,(k,+)}=\theta_{B,(k,-)}=B_k.
\]

This is a consequence of the full fine stationary equations. The actual
conductances need not have pair symmetry, and an evolving state with equal
member phases need not remain on that stratum. No fixed-weight pair
closure is used in the argument.

<a id="contact-stationary-reflection"></a>
## 4. Branch-preserving reflection of a possible stationary phase

The bare graph also preserves the reversal `k -> -k mod 5` in both
components. Combine it with phase conjugation and the fixed lift correction

\[
(\mathcal R\theta)_{C,(k,s)}=
\begin{cases}
-\theta_{C,(0,s)}, & k=0,\\
2\pi-\theta_{C,(5-k,s)}, & k=1,2,3,4,
\end{cases}
\qquad C\in\{A,B\}.
\]

Old forward gaps are carried to reversed-index forward gaps with the
same acute bounds, including the declared closing-edge correction.
The `A_0 -> B_0` gaps change sign, the `A_0 -> B_1` and `A_0 -> B_4`
gaps exchange with a minus sign, and the internal `A_0` gap changes sign.
Thus `R` preserves the full closed chamber and `V_U`; it also maps
`S_U=0` to `S_U=0`. It need not preserve the actual weighted dynamical
field, which is not required for this stationary phase argument.

After the derived pair equality, choose the common phase representative
`A_0=0`. This is possible without altering any branch gap. Reflection
preserves that representative. Uniqueness therefore forces
`R theta=theta`, giving

\[
B_0=0,\qquad B_4=2\pi-B_1,\qquad B_3=2\pi-B_2.
\]

Write `a=B_1`, so `B_4=2*pi-a`. The two contact branches give
`-pi/2<=a<=pi/2`. These relations have been derived without fixing the
receiver's origin or internal shape in advance.

<a id="contact-stationary-obstruction"></a>
## 5. Exact obstruction in the complete acute chamber and its closure

The fine members of B pairs two and three have only their four old
neighbors. With the stationary member equality just proved, the current
equation at B pair two is

\[
0=2\sin(B_1-B_2)+2\sin(B_3-B_2).
\]

Both consecutive forward gaps lie in `[-pi/2,pi/2]`. Strict injectivity
of sine on that interval makes them equal. The same argument at B pair
three gives a common gap `q` on the whole three-edge chain from B pair
one through pairs two and three to pair four:

\[
B_2-B_1=B_3-B_2=B_4-B_3=q,\qquad
-\frac\pi2\le q\le\frac\pi2.
\]

Together with the derived reflection this gives

\[
2a+3q=2\pi,\qquad a\ge\frac\pi4.
\]

The endpoint B pair one has two neighbors in each of `B_0,B_2,A_0`.
Its full fine current equation, with no changed edge multiplicities, is

\[
0=2[-\sin a+\sin q-\sin a],\qquad
\sin q=2\sin a.
\]

But `pi/4<=a<=pi/2`, so the right side is at least `sqrt(2)`, while
the left side is at most one. The strict contradiction proves

\[
\boxed{\text{There is no stationary state in the declared complete
acute chamber, or in its closed acute boundary.}}
\]

The conclusion holds for every actual positive conductance assignment
and positive held capacity produced by the frozen contact family, not
only the nominal split control. Those quantities enter the full evolution
and its storage but cancel from the necessary phase stationary problem
after the full form equation has forced constant form. The proof changes
neither the UM fan nor the pressure, phase law or clock.

An independent interior obstruction avoids reflection and helps identify
the conflicting currents. Normalize `A_0=0` after stationary member
equality, and put `a=B_1`, `b=B_4-2*pi`, `c=B_0`. The uncontacted B
chain gives `3*q=2*pi+b-a`, hence
`d=a-b=2*pi-3*q>pi/2`. Both contact gaps are acute, so `a,-b>0`.
The remaining cycle gaps `r=a-c,t=c-b` are acute and have `r+t=d`,
so they too are positive. The two endpoint balances require

\[
2\sin q=(\sin r+\sin t)+(\sin a-\sin b).
\]

Each parenthesis is strictly greater than one: for positive acute
`u,v` with `u+v>pi/2`, one has
`sin v>cos u` and `sin u+cos u>1`. The displayed right side is
therefore greater than two, while its left side is less than two.
This second argument is confined to the open chamber; the preceding
reflection proof supplies the boundary result.

<a id="contact-stationary-scope"></a>
## 6. What the obstruction establishes

The actual UM contact fan cannot support an equilibrium retaining all
of the original fine and cross-edge acute branches. In particular, there
is no stationary relative-storage minimum of the complete law there.
A minimum constrained to the closed chamber can have nonzero boundary
forces; it is not a stationary solution of the unconstrained nodal rows.
The result is an
exact feasibility obstruction, rather than a failed optimizer or an
uncertified small residual.

The earlier result required the receiver to leave its original small
radius-`1/16` identity tube. The present result is stronger about possible
stationary targets: merely enlarging that tube within this complete acute
chamber cannot produce a stationary continuation. It does not assert a
time at which the actual trajectory leaves the chamber, or that it ever
does. The conservative law can support motion without an equilibrium
at the moving state; excluding equilibrium does not exclude bounded
nonstationary organization.

An edge reaching `|delta|=pi/2` also does not imply a winding change.
It is a boundary of the acute certificate, whereas a principal phase cut
lies at `|delta|=pi`. The actual winding observations can therefore survive
loss of acuteness. Stationary states with nonacute edges, other complete
cycle classes, nonstationary identities and acquisition from the actual
contacted source retain their separate obligations. No statement about
physical particles, a unique fundamental law, or autonomous contact
occurrence follows from this graph-specific theorem.

The [principal-class curvature analysis](#contact-nonacute-protocol) below
resolves the larger stationary-minimum question: nonacute saddles exist,
but no stable equilibrium is available in the same complete open class.
Actual nonstationary motion remains a separate question.

<a id="contact-stationary-controls"></a>
## 7. Independent static controls and comparison scope

The [static controls](../../tests/physics/test_sine_contact_stationary_geometry.py)
check all 53 edge orientations and branches, incidence rank nineteen and
34 independent cycle directions. Exact turns check the complete reference
class. Independent fine edge sums check the monotonicity, reflected current
map and receiver-current equations without calling a root finder. The fixed
algebra probes use 100-digit arithmetic; graph-capture comparisons retain
their represented phases and absolute tolerance `2^-46` separately.

The auxiliary positive conductances cycle through `3/8,...,7/8`. They test
the complete weighted field and stationary reduction; they are not new UM
outputs or replacements for the preceding contact's frozen conductances.
The theorem itself applies to every admitted actual positive weight list.
No event or frozen response producer is replayed by these controls.

Two comparisons keep the exclusion's domain explicit. The disconnected
original twists have zero phase current before the fan is added. On the
same connected contact support, constant form and constant phase give an
equilibrium in the distinct zero-winding class: its relative storage Hessian
consists of positive weighted-form and bare-phase Laplacians. This does not
place that equilibrium in the excluded original chamber or show that the
contacted source reaches it. Nonconstant form at constant phase also checks
that a zero sine current alone does not satisfy both complete stationary rows.

These finite checks support implementation and algebraic consistency. They
neither prove the chamber-wide theorem nor certify a new organized source.

<a id="contact-nonacute-protocol"></a>
## 8. Frozen nonacute minimum question and static controls

This declaration precedes the new static evaluations. Retain the complete
model, actual conductances, capacities, clock, support, orientations and
branches of Section 1. Enlarge only the admitted phase domain to
`-pi<delta_e<pi` on every edge. The question is whether a stationary local
minimum of the full relative storage exists anywhere in this open marked
class. A nonstrict minimum is included. The complete fine Hessian must be
positive semidefinite, but that necessary condition alone is not sufficient.

Derive any pair or receiver restriction from full stationarity and local
minimality. Preserve zero neighbor resultants and zero curvature until an
argument on this actual graph excludes them as minima. The old acute
uniqueness/reflection proof and the odd-degree bound for another contact
graph are unavailable here. Holding the donor's relative phases fixed while
varying its common phase is an allowed variation, not a dynamical closure.

The bounded evaluation is analytical. Independent static controls may check
the zero-resultant fiber identity, adjacent-twin splitting curvature, the
exact receiver restriction, all eight cosine-sign cases of the three-edge
chain, and the final endpoint inequalities. Use exact rational matrices and
turns where available and 100-digit arithmetic for fixed trigonometric
identities. Auxiliary probes use the quadrature C4, adjacent-twin half-split
`pi/3`, a rotation increment `1/64`, and receiver phases
`(0, pi/8, pi/3, 3*pi/4, 5*pi/4, 7*pi/4)` in order
`(O,B_0,B_1,B_2,B_3,B_4)`. These are algebra controls, not proposed equilibria
or source responses. Retain the earlier represented-phase tolerance for
shared-reader comparisons. No optimizer, root search, trajectory, event or
frozen producer is replayed. A minimum obstruction would leave stationary
saddles, nonstationary organization and principal-cut crossings separate.

<a id="contact-nonacute-full-curvature"></a>
## 9. Full stationary curvature in the principal marked chamber

Let `P` be the open principal chamber declared in Section 8. The full
stationary reduction from Section 2 remains valid there: `x` is constant
and `S_U(theta)=0`. The full storage Hessian is block diagonal,

\[
\nabla^2 H=\begin{pmatrix}L_W&0\\0&K\end{pmatrix},\qquad
K=L_{\cos\delta}.
\]

The actual positive weighted form block is positive on relative form
directions. A stationary relative-storage minimum therefore requires the
signed phase Hessian `K` to be positive semidefinite on all relative fine
directions. Since `K*1=0`, this is equivalent to `K>=0` on the whole phase
space; changing a test vector by a common constant leaves its curvature
unchanged. Fixing either conserved common mean does not discard a negative
relative direction.

The following proof excludes every stationary point satisfying this
necessary PSD condition in `P`, including singular Hessians. It thus
excludes nonstrict as well as strict local minima. It does not apply the
acute uniqueness or reflection argument outside its domain.

<a id="contact-nonacute-twin-reduction"></a>
## 10. Stationary PSD conditions force fine twin equality

For each fine node define its bare neighbor resultant and Hessian diagonal

\[
R_i=\sum_{j\sim i}e^{i\theta_j},\qquad
r_i=\Re(e^{-i\theta_i}R_i)=K_{ii}.
\]

Stationarity makes `Im(exp(-i*theta_i)*R_i)=0`; PSD gives `r_i>=0`.
If `R_i` is nonzero, these two conditions force
`exp(i*theta_i)=R_i/|R_i|`. If `r_i=0`, PSD forces the whole row and
column of `K` to vanish, so every incident edge has zero cosine. This
zero case cannot be dropped merely because another graph had odd degrees.

Every pair except `A_0` consists of nonadjacent twins with exactly the
same external neighbors. Both members therefore share one resultant `R`.
If `R` is nonzero, the alignment just derived gives equal circular phases.
For any one common neighbor, the two incident edges have the same branch
when oriented outward from the twin pair. Their strict principal bounds
give `|theta_+-theta_-|<2*pi`. Equal circular phases must consequently
have equal lifts in this chamber.

Now retain the possibly zero resultant `R=0`. Both PSD rows vanish, so
every common neighbor is perpendicular to both twins. Unequal circular
twins must then be antipodal. The same strict branch bound makes their
lifted separation exactly `pi` in absolute value. Write them, after
possibly interchanging their names, as `Theta+pi/2,Theta-pi/2`.

For a common neighbor `j`, let `eta_j` be its lift adjusted by the common
outward edge branch, so both principal gaps are `eta_j-theta_+` and
`eta_j-theta_-`. Their two strict cut inequalities imply

\[
-\frac\pi2<\eta_j-\Theta<\frac\pi2.
\]

The zero incident cosines require `sin(eta_j-Theta)=0`, hence
`eta_j=Theta`. All common neighbors have the same circular phase, and
there is at least one of them. Their resultant is therefore
`degree*exp(i*Theta)`, contradicting `R=0`. Unequal zero-resultant
twins are excluded. Equal twins with zero resultant have not been
excluded by this lemma; they remain covered by the subsequent argument.

The internal `A_0` edge makes that pair adjacent twins. A separate local
curvature calculation applies. Put
`theta_+=Theta+epsilon,theta_-=Theta-epsilon`; the internal principal
cut gives `|epsilon|<pi/2`. With every external phase fixed, define
`C=Re(exp(-i*Theta)*R)` for their common external resultant. The part of
the potential depending on `epsilon` is

\[
f(\epsilon)=-2C\cos\epsilon-\cos(2\epsilon)+\text{constant},
\qquad f'(\epsilon)=2\sin\epsilon(C+2\cos\epsilon).
\]

If a stationary split has `epsilon!=0`, then `C=-2*cos(epsilon)` and

\[
f''(\epsilon)=-4\sin^2\epsilon<0.
\]

That is a full fine antisymmetric phase direction and contradicts PSD.
Thus the adjacent pair also has equal lifted phases. This argument retains
zero external resultants and does not divide by them.

Every hypothetical stationary PSD state in `P` therefore has equal lifted
members in all ten pairs. This is a necessary stationary restriction,
not a weighted dynamical pair-closure claim. It used the actual complete
branch class to handle the degenerate nonadjacent twins.

<a id="contact-nonacute-receiver-restriction"></a>
## 11. Exact receiver variation and a second twin restriction

Keep the derived fine-member equality. Vary each B pair through its common
phase, and vary all ten A phases by one common rotation, holding every A
relative phase fixed. This is a six-dimensional family of configurations,
not a proposed invariant dynamical subsystem. Every internal A edge retains
its phase gap exactly. With `O=A_0` and the five B pair phases as its
coordinates, the restricted phase potential is

\[
V_{\rm restricted}=\text{constant}+4V_G,
\]

where `G` is the six-vertex graph consisting of the B cycle and the three
edges `O--B_0,O--B_1,O--B_4`. The factor four is the actual fine edge
multiplicity of every complete pair block. No actual conductance is
averaged or altered: this is a restriction of the bare phase potential.

Full stationarity makes every derivative of this restricted potential
zero. In particular, the O derivative sums all A node currents, whose
internal contributions cancel. Full PSD makes its restricted Hessian PSD.
No separate classification of the A cycle is required.

In `G`, the vertices O and `B_0` are adjacent twins: both have the same
two external neighbors `B_1,B_4`. Their internal contact edge has branch
zero and strict principal gap. The adjacent-twin calculation in Section
10 therefore forces `B_0=O`. It is valid even if their common external
resultant vanishes. This equality has been derived without reflection,
acute uniqueness or a supplied relative origin.

Use the common phase freedom to put `O=B_0=0`, and write

\[
a=B_1,\qquad b=B_4-2\pi.
\]

The actual contact branches give `-pi<a,b<pi`. The retained B cycle has
the three interior chain gaps

\[
q_1=B_2-B_1,\quad q_2=B_3-B_2,\quad q_3=B_4-B_3,
\qquad q_1+q_2+q_3=2\pi+b-a.
\]

All three gaps lie strictly between `-pi` and `pi` by the original branch
declaration. The identity on their sum retains the complete marked class;
it is not obtained by wrapping a sum after the calculation.

<a id="contact-nonacute-chain-curvature"></a>
## 12. The uncontacted chain retains no negative-cosine alternative

Stationarity at B pairs two and three gives
`sin(q_1)=sin(q_2)=sin(q_3)`. Thus their cosine magnitudes are equal:
write `cos(q_j)=epsilon_j*c`, with `c>=0` and signs `epsilon_j` when
`c>0`. Restrict the six-vertex Hessian further to the two coordinates
`B_2,B_3`, holding its other coordinates fixed. Apart from the positive
factor four, its matrix is

\[
T=\begin{pmatrix}
\cos q_1+\cos q_2&-\cos q_2\\
-\cos q_2&\cos q_2+\cos q_3
\end{pmatrix}\succeq0.
\]

If `c>0` and the middle cosine is `-c`, the two nonnegative diagonals
force both endpoint cosines to be `+c`. The resulting determinant is
`-c^2`, a contradiction. If the middle cosine is `+c` but an endpoint
cosine is `-c`, one diagonal is zero while the off-diagonal is nonzero,
again contradicting PSD. The only remaining case for `c>0` is that all
three cosines are positive. Their strict principal domains and equal
sines then make all three gaps equal in `(-pi/2,pi/2)`.

The singular case `c=0` is retained explicitly. Each principal gap is
`+pi/2` or `-pi/2`; equality of their sines again makes all three equal.
Consequently every hypothetical PSD state satisfies

\[
q_1=q_2=q_3=q,\qquad -\frac\pi2\le q\le\frac\pi2,\qquad
3q=2\pi+b-a.
\]

Because `a,b` are strict principal contact gaps, `2*pi+b-a>0`. Thus
`0<q<=pi/2`, including the possible zero-cosine endpoint. It follows that

\[
a-b=2\pi-3q\ge\frac\pi2.
\]

No inverse cosine or square-root division has removed a singular case.

<a id="contact-nonacute-obstruction"></a>
## 13. Endpoint currents and curvature contradict the marked winding

The B pair-one and pair-four equations in the restricted graph are

\[
\sin q=2\sin a=-2\sin b.
\]

The two terms at each endpoint come from its distinct edges to O and
`B_0`, whose phases have been proved equal. Their Hessian diagonals must
also be nonnegative:

\[
2\cos a+\cos q\ge0,\qquad
2\cos b+\cos q\ge0.
\]

Since `0<q<=pi/2`, stationarity gives
`0<sin(a)=-sin(b)<=1/2`. On the strict principal interval this puts
`a` in either `(0,pi/6]` or `[5*pi/6,pi)`, and `-b` in the same two
intervals. The distant alternative is impossible for either endpoint:
its cosine is at most `-sqrt(3)/2`, so its Hessian diagonal is at most
`1-sqrt(3)<0`. Therefore

\[
0<a\le\frac\pi6,\qquad
0<-b\le\frac\pi6,\qquad
a-b\le\frac\pi3.
\]

This contradicts the necessary lower bound `a-b>=pi/2` from the same
retained chain. Hence

\[
\boxed{\text{No stationary phase in the original open principal marked
chamber has a positive-semidefinite full fine Hessian.}}
\]

Every stationary state in that chamber, if any exists, has a strictly
negative relative phase-curvature direction. The full storage therefore
has no stationary local minimum there, including nonstrict or degenerate
minima. The conclusion holds for every actual positive conductance and
positive held capacity in the frozen contact family: those actual values
remain in the complete law and the positive form block, while stationary
phase and its curvature use bare support.

<a id="contact-nonacute-scope"></a>
## 14. Scope beyond the acute obstruction

This result closes the nonacute stationary-minimum question in the same
complete cycle class. It derives all reductions from full stationarity,
fine curvature and strict principal branch admission. Even-degree zero
resultants, zero chain cosines and singular PSD tests have been retained
until the graph-specific contradiction excludes the whole candidate set.
No odd-degree resultant estimate, acute convexity, reflection ansatz,
optimizer outcome or numerical Hessian eigenvalue is a premise.

The result excludes a stationary local-storage minimum, rather than every
possible stationary state in the enlarged chamber. It does not extend to
principal-cut boundary points, where the strict twin branch intersection
changes, or to other complete cycle classes. The synchronized equilibrium
from Section 7 remains a valid minimum in its distinct zero-winding class.

Nor does the theorem show that the actual contacted trajectory changes
winding, approaches any equilibrium or acquires a different identity.
Nonstationary organization, finite retention, full-time confinement and
source entry retain separate dynamical obligations. The absence of a
stationary minimum is a geometric restriction on one candidate maintenance
mechanism under the supplied law, not a failure of the nodal identity or
an identification with fundamental physical behavior.

<a id="contact-nonacute-saddle-control"></a>
## 15. Fixed algebraic stationary saddle control

### 15.1. Prospective declaration

Before evaluating this additional static control, fix

\[
c=\frac{1+\sqrt{17}}8,\qquad t=\arccos c,\qquad
q=2t,\qquad a=\pi-3t.
\]

Use synchronized members in each pair, the original A winding
`A_k=2*pi*k/5`, and the B lifted phases

\[
(B_0,B_1,B_2,B_3,B_4)
=(0,a,a+q,a+2q,2\pi-a).
\]

Retain the same 53-edge support and every original branch, positive actual
conductance, positive held capacity and complete law. Put `x_i=m` constant
at any admitted common form origin. The fixed predictions are full
stationarity, strict principal-chamber membership and negative fine phase
curvature `K_ii=4*cos(q)<0` at either B pair-two member. The algebraic
constant is not fitted or retuned. Any static numerical check uses
100-digit arithmetic and the already declared controls; no root finder,
trajectory, source re-preparation or actual event is performed.

### 15.2. Exact stationary and negative-curvature verification

The fixed algebraic constant satisfies

\[
4c^2-c-1=0,\qquad \frac12<c<\frac1{\sqrt2},
\qquad \frac\pi4<t<\frac\pi3.
\]

Consequently `0<a<pi/4`, `pi/2<q<2*pi/3` and `2*a+3*q=2*pi`.
The B cycle's forward gaps are exactly `(a,q,q,q,a)`. They are all
strict principal gaps with winding one. The cross gaps are zero, `a`
and `-a`; the internal gap is zero. Together with the old A twist these
are precisely the original complete branch assignments, although the
three B chain gaps are nonacute.

The algebraic relation for `c` gives an exact current identity:

\[
\sin q=2\sin t\cos t
=2\sin t(4\cos^2t-1)
=2\sin(3t)=2\sin a.
\]

Thus the two receiver endpoints satisfy their full current balances.
The three identical chain gaps cancel at B pairs two and three, and
the symmetric `+a,-a` contacts cancel at B pair zero and A pair zero.
All remaining A currents are those of the original twist and vanish.
Fine member synchronization supplies the exact factors two in each
node current. Hence every fine sine current is zero. Constant form
also gives `L_W*x=0` for every retained actual conductance assignment,
so both complete stationary rows vanish.

At either member of B pair two, all four incident edge cosines equal
`cos q`. Its fine Hessian diagonal is

\[
K_{ii}=4\cos q=4(2c^2-1)=2(c-1)<0.
\]

The corresponding one-node variation, recentered by any common phase
constant to retain the declared mean, has strictly negative curvature.
This is an explicit stationary saddle in the retained principal chamber.
It distinguishes the no-minimum theorem from a claim that no stationary
configuration exists there. The construction does not describe a state
acquired by the actual contacted source.

<a id="contact-nonacute-linear-instability"></a>
## 16. Instability of every stationary point under the actual weighted law

The negative phase direction also has a dynamical implication for this
specific conservative two-row model. Keep the actual mobility
`M=diag(nu_i/(pi*d_i))`, positive weighted Laplacian `L_W` and stationary
phase Hessian `K`. The tangent equations are

\[
\dot\xi=-MK\eta,\qquad \dot\eta=ML_W\xi,
\qquad \ddot\eta=-AK\eta,\qquad A=ML_WM.
\]

Let `rho_i=d_i/nu_i` and let `E` be the tangent space `rho^T*v=0` of
either conserved weighted-mean leaf. The symmetric matrix `A` has kernel
`span(rho)` and is strictly positive on `E`; denote that positive
restriction by `A_E`. For the orthogonal projection `P_E`, put
`K_E=P_E*K|_E`. The complete second-order phase equation on that leaf is
`eta_ddot=-A_E*K_E*eta`, because `A` annihilates the omitted rho component.

A negative fine curvature direction from Section 13 can be shifted by a
common constant to lie in `E`, without changing its quadratic form since
`K*1=0`. Hence `K_E` has a negative direction. The product `A_E*K_E` is
similar to the symmetric matrix
`A_E^(1/2)*K_E*A_E^(1/2)`, which also has a negative eigenvalue. It follows
that `-A_E*K_E` has a strictly positive eigenvalue `sigma^2` and a real
nonzero eigenvector `v in E`.

The growing solution of the full first-order tangent system is recovered
without dropping the form row:

\[
\eta(t)=e^{\sigma t}v,\qquad
\xi(t)=-\frac{e^{\sigma t}}\sigma MKv.
\]

The form perturbation also lies in `E`, because
`rho^T*M=(1/pi)*1^T` and `1^T*K=0`. Both tangent equations therefore hold
on the actual two conserved-mean leaves. Every stationary state in the
retained open principal chamber has a real growing tangent mode under the
same actual weighted law. Smooth finite-dimensional linearized instability
then gives instability of that equilibrium on its relative leaf.

This result supplies no uniform growth rate over the chamber or conductance
family, no departure time for the contacted preparation, and no claim that
the preparation approaches one of these stationary states. A growing mode
near an equilibrium does not itself select a principal-cut crossing or
exclude every nonstationary organized motion.

The [static controls](../../tests/physics/test_sine_contact_stationary_geometry.py)
check the full fine restriction, every chain cosine-sign case, a zero-resultant
counterexample outside the retained branch class, and the fixed saddle's
negative direction on its exact weighted-mean leaf. The rounded saddle is
checked with an independent captured-state residual and enclosing rate bounds;
neither an exact represented root nor a bound on the entire interval width
is inferred from the declared midpoint tolerance. These finite checks support
the shared arithmetic and field consumers, not a numerical substitute for
the analytical exclusions.

<a id="contact-energy-protocol"></a>
## 17. Frozen conservative confinement question

Fix each exact actual post-word source from the
[contact protocol](SINE_ORGANIZATION_CONTACT.md#organization-contact-protocol),
its actual 53-edge support, generated positive conductances, held capacities,
original structural clock and conservative weighted-form/bare-phase law.
Retain every marked branch from Section 1 and the open principal chamber
from Section 8. No new event, forcing or loss is inserted.

Let `rho_i=d_i/nu_i`, `m_rho=sum(rho*x_0)/sum(rho)`,
`Q_0=(1/2)*x_0^T*L_W*x_0`, `V_0=V(theta_0)` and `h=Q_0+V_0`.
The question is whether storage and the two conserved lifted means can
isolate the actual preparation from every principal cut of this complete
marked class. The analytical prediction is a continuous configuration path
starting at that source and ending on a first principal boundary, with
`H=h`, both source means unchanged, and every edge strictly inside its
original branch until the endpoint. Such a path is not claimed to satisfy
either evolution row or select the cut reached by the actual source.

Use the full fine state, fixed-mean compactness, the no-minimum result and
strict sublevel components. Preserve the actual positive form storage as a
coordinate of H; its variation along a comparison path does not add an event
reserve or an external reservoir. Prove source-specific positivity before
dividing by `Q_0`. No optimizer, root finder, trajectory, frozen producer or
new precision campaign belongs to this declaration.

Before static evaluation, fix these independent algebra controls. Reuse the
existing auxiliary positive conductance list, capacities and 100-digit
arithmetic; those weights are not regenerated UM outputs. Check the exact
post-AL nominal forms `m=5/16`, increment `b=1/1024` on A pair zero only,
the component mean difference, the support diameter and weighted centering.
For radial form lifting use the original doubled twists and the phase path
that decreases B pair one by `s` and increases B pair four by `s`, recentered
to its original weighted phase mean, at `s=0,1/128,1/64`. Check storage and
means with both this nonconstant nominal form and a separately declared
uniform-form comparison; the latter is not an actual post-word source.

For the separate cycle-information control, rigidly rotate all B phases by
`s=0,pi/2,3*pi/5,4*pi/5`, with common weighted recentering. Predict unchanged
old constituent cycle gaps and a first cross-edge cut at `3*pi/5`; do not
attribute the energy-path theorem to this rigid rotation. Exact turn and
matrix checks precede represented shared-reader comparisons, whose residual,
enclosure and existing `2^-46` midpoint tolerance remain distinct.

<a id="contact-energy-source-reserve"></a>
## 18. The actual source has strictly positive form storage

Fix one exact source from Section 17, including its actual `W,nu` and
post-word forms and lifts `(x_0,theta_0)`. All following means and paths
refer to this one source; differing sources retain their own invariant
values and generated conductances. Put

\[
Q(x)=\frac12x^TL_Wx,\quad V(\theta)=V_U(\theta),\quad
H=Q+V,\qquad
m_\rho=\frac{\rho^Tx_0}{\rho^T1},\quad
\theta_\rho=\frac{\rho^T\theta_0}{\rho^T1},\quad
\rho_i=\frac{d_i}{\nu_i}>0.
\]

These are the actual weighted form mean and the declared lifted phase
mean, not circular or separate constituent means. Both remain conserved
under the selected complete law because
`rho^T*M=(1/pi)*1^T` and both the sine current and weighted form gradient
sum to zero.

Before the word, each constituent's arithmetic form mean is the same `m`.
AL increases only the two selected A forms by `b=1/1024`; UM and SHA
leave all forms unchanged. Thus the actual endpoint has

\[
\overline x_A=m+\frac b5,\qquad \overline x_B=m.
\]

These arithmetic means certify an initial difference only; they are not
claimed to be conserved after contact. The difference proves that `x_0`
is nonconstant. Since the actual positive weighted graph is connected,
`Q_0=Q(x_0)>0`. This remains true for the synchronized-member control and
does not require a nonzero selected internal split.

A uniform lower bound follows directly from the admitted source geometry.
Some A node `i` has `x_i>=m+b/5`, and some B node `j` has `x_j<=m`.
A path from `i` to the selected A pair has at most two edges. A contact
edge followed, when necessary, by one B edge reaches `j`, so a joined
path has length `ell<=4`. Every actual conductance is greater than `3/4`.
Cauchy's inequality on that path gives

\[
Q_0\ge\frac12\sum_{e\in\text{path}}W_e(\Delta_ex_0)^2
>\frac{3}{8\ell}(x_i-x_j)^2
\ge\frac{3b^2}{800}
=\frac3{838860800}>0.
\]

The strict inequality uses at least one nonzero path difference. It is a
bound on the existing positive form term in H, not an additional reservoir
or an event-work allowance. Write `V_0=V(theta_0)` and `h=Q_0+V_0`.

<a id="contact-energy-compactness"></a>
## 19. Compactness on the two fixed-mean leaves

Let `A_theta` be the affine phase leaf `rho^T*theta=rho^T*theta_0`, and
write `Omega=P intersect A_theta` for the original strict principal
chamber on that leaf. The actual source lies in `Omega` by Section 2.
Its closed chamber replaces all strict gap inequalities by
`-pi<=delta_e<=pi`. It is the closure of `Omega`: interpolation from any
closed-chamber point to the strict source makes all edge inequalities
strict before the closed endpoint.

This phase closure is compact. The joined graph has diameter at most four:
any A node is within two edges of either selected A member, any B node
is within two edges of that member, and the within-component distances
are at most two. Along each edge, the retained branches `b_e in {-1,0,1}`
give

\[
|\theta_j-\theta_i|\le\pi+2\pi|b_e|\le3\pi.
\]

Consequently the full lifted phase range is at most `12*pi`. Positivity
of `rho` puts its mean inside that range, so every coordinate obeys
`|theta_i-theta_rho|<=12*pi`. The closed edge inequalities and mean
constraint make this bounded set closed.

The full closed storage sublevels on both source mean leaves are compact
as well. For `H<=E`, nonnegativity of `V` gives `Q<=E`. With
`w_min=min_e W_e>0`, Cauchy's inequality along a path of at most four
edges yields

\[
|x_i-x_j|^2\le\frac{8E}{w_{\min}}.
\]

The fixed positive weighted mean then bounds every
`|x_i-m_rho|<=sqrt(8*E/w_min)`. Together with the phase bounds, continuity
of H proves compactness of each nonempty closed sublevel. This argument
retains the actual conductances and means; it does not substitute an
arithmetic mean or a unit-weight Laplacian.

There is no local minimum of V in `Omega`. Indeed a constrained critical
point there has `grad V=lambda*rho`; translation invariance gives
`1^T*grad V=0`, hence `lambda=0`. At a local minimum its relative
phase Hessian would be PSD. Since `K*1=0`, every phase test direction can
be shifted by a common constant onto the weighted-mean tangent leaf
without changing its curvature. This would make the full fine Hessian
PSD, contradicting Section 13. Thus the no-minimum result applies on
this actual phase leaf, including all possible singular candidates.

<a id="contact-energy-sublevel-boundary"></a>
## 20. A strict phase sublevel must reach a principal boundary

Choose the source-specific threshold

\[
E_* = V_0+\frac{Q_0}{2}<h,
\]

and let `D` be the connected component containing `theta_0` of the
strict phase sublevel `{theta in Omega: V(theta)<E_*}`. It is nonempty
because `Q_0>0`. It is open in the finite-dimensional affine phase leaf.
Its closure is compact by Section 19, and every closure point has
`V<=E_*` by continuity.

Suppose that this closure avoided all principal boundary faces. It would
then lie inside `Omega`, and V would attain a minimum at some point
`theta_min` of the closure. Its value is at most `V_0<E_*`. A sufficiently
small ball about that point lies in `Omega` and in `{V<E_*}`. Since the
point is in the closure of D, the ball intersects D. The ball is connected,
so it belongs to the same component D, including its center. This makes
`theta_min` an interior local minimum of V on the phase leaf, contrary
to Section 19.

Therefore there exists a point `theta_b` satisfying

\[
\theta_b\in\overline D\cap\partial\Omega,\qquad
V(\theta_b)\le E_*<h.
\]

At least one original edge gap equals `+pi` or `-pi` there; all other
closed gap constraints still hold. The statement identifies neither that
edge nor a unique boundary point.

This component argument supplies more than an arbitrary small excess-energy
comparison. The actual positive form term leaves a strict margin between
the boundary potential and the conserved total h. That margin will give
an exact-energy path to an attained boundary endpoint.

<a id="contact-energy-path"></a>
<a id="contact-energy-shell-path"></a>
## 21. A comparison path on the exact invariant shell

By continuity at `theta_b`, choose a convex ball in `A_theta` about that
point on which

\[
V<\frac{E_*+h}{2}<h.
\]

Choose `y` in D inside this ball, which is possible because
`theta_b in closure(D)`. An open connected subset of a finite-dimensional
affine space is polygonally connected: the points reachable from a fixed
one by finitely many segments inside the open set form a relatively open
and relatively closed subset of each connected component. Hence a finite
polygonal phase path from `theta_0` to y lies entirely in D.

Append the straight segment from y to `theta_b`. It stays inside the
chosen convex ball and inside the closed phase chamber. For every point
before its endpoint, each edge gap is a strict convex combination of a
strict interior gap at y and a closed gap at `theta_b`, so all edge gaps
remain strict principal gaps. The appended endpoint is the first
principal boundary reached by the whole path.

Write this finite piecewise-linear phase path as `theta(s)`, `0<=s<=1`.
It retains the source phase mean and satisfies throughout

\[
V(\theta(s))<h,\qquad h-V(\theta(s))>\frac{Q_0}{4}.
\]

The first polygonal portion even has `h-V>Q_0/2`; the weaker displayed
bound also covers the final segment and its boundary endpoint. Now use
the actual initial relative form direction to define

\[
\boxed{
x(s)=m_\rho1+
\sqrt{\frac{h-V(\theta(s))}{Q_0}}\,(x_0-m_\rho1).
}
\]

The square root is well-defined and strictly greater than `1/2` on the
whole path. Since `L_W*1=0`, the actual quadratic form scales exactly:

\[
\rho^Tx(s)=\rho^Tx_0,\qquad
Q(x(s))=h-V(\theta(s)),\qquad
H(x(s),\theta(s))=h.
\]

At `s=0` the multiplier is one, so this path begins at the actual full
source `(x_0,theta_0)`. It ends at a principal-cut configuration, preserves
both source means and total storage exactly, and keeps every original edge
gap strictly within its branch before the endpoint. The forms and phases
remain continuous and piecewise smooth; all weights, capacities and support
stay at their actual source values. No pair synchronization or changing
weighted boundary association is imposed along the path.

The parameter s is only a comparison-path parameter. It has no declared
conversion to the original structural time. In particular, the radial form
formula is not a pressure reconstruction or a solution of either nodal row.
It redistributes the two existing terms of H among configurations without
asserting that the complete law performs that redistribution along this path.

<a id="contact-energy-path-continuation"></a>
## 22. Geometric continuation across a cut and the barrier obstruction

The strict potential margin also permits a short continuation of the final
straight phase segment beyond `theta_b`. If an edge gap equals `+pi` at
the endpoint, its slope along the segment is positive because the starting
gap at y was strictly smaller; the analogous slope is negative at a
`-pi` face. Thus every face reached there is crossed transversely by a
sufficiently short extension. The continuous sine potential stays below h
in the same small ball, so the radial form lift continues with the same
storage and means. This is an additional configuration comparison beyond
the original chamber, not a prediction of the actual vector field.

In particular, the connected component of the actual invariant shell
`H=h` on both mean leaves that contains the source reaches a principal cut
and has configurations across it. Conservation of precisely these three
quantities cannot separate that source component from every cut. There is
no strictly positive storage barrier protecting the whole original complete
principal chamber for any source in the frozen family.

This excludes a trapping certificate based solely on separation of that
storage-and-means level component from the cuts. It does not exclude an
additional dynamical invariant, an invariant subset of the shell, or a
nonstationary organization maintained by another proved mechanism. The
smooth sine law itself remains defined at a principal cut; the cut is a
boundary of the retained phase identity, not an inserted support event or
a singularity of the complete field.

<a id="contact-energy-cycle-readouts"></a>
## 23. Complete marked cuts and constituent winding are different readouts

The theorem does not select a cut or say that either original constituent
winding changes. A separate exact kinematic comparison makes this
distinction explicit. Start from the original doubled twists, rigidly rotate
all B phases by s, and subtract the common phase increment
`s*sum_(i in B)(rho_i)/sum_i(rho_i)` from every node to retain the same
weighted phase mean. All old A and B edge gaps and the internal A pair gap
are unchanged.

The contact gaps at B pairs zero, one and four are respectively
`s,alpha+s,-alpha+s`, where `alpha=2*pi/5`. On `0<=s<3*pi/5` they
remain strict principal gaps. At `s=3*pi/5` the four contacts to B pair
one reach `+pi`, while every old constituent cycle gap is still alpha.
Both original winding observations remain one. Passing that cut changes
the branch inventory of those cross edges; it does not change the old
constituent cycle gaps.

This rigid comparison is not the exact-shell path constructed above.
With the nominal forms held fixed, its contact potential changes by

\[
\Delta V_C(s)=4(1+2\cos\alpha)(1-\cos s),\qquad
\Delta V_C(3\pi/5)=4+2\sqrt5>0.
\]

The common recentering changes no edge difference and cannot remove that
cost. These observations establish that a complete marked cut need not be
a constituent-winding cut; they do not identify which kind is reachable
along the invariant-shell path or which kind the actual flow might reach.

<a id="contact-energy-scope"></a>
<a id="contact-energy-conclusion"></a>
## 24. Consequence for maintenance under the supplied complete law

Every exact contacted source in the frozen family has a same-storage,
same-two-means configuration path to a first principal cut, with its actual
support, conductances and capacities retained. This rules out using the
existing total-storage barrier mechanism alone to certify perpetual
maintenance of the original complete cycle class for that source family.
The result needed no trajectory, candidate optimization or increase in
response precision.

Neither this path nor the absence of stationary minima proves that the
actual trajectory crosses a cut, changes a constituent winding, dissolves,
or acquires a new common identity. The analytical finite-retention and
old-receiver-tube exit results remain valid with their original intervals.
No selected boundary point, path-length bound, structural-time deadline,
autonomous event selector or physical identification is supplied here.
Actual confinement or passage needs information about the complete dynamics
beyond these three conserved scalar values.

The [static controls](../../tests/physics/test_sine_contact_stationary_geometry.py)
check exact weighted centering, the support diameter, positive restricted
form curvature and radial storage scaling. A fixed short phase deformation
tests the constant-energy lifting identities, including a separately labeled
uniform-form comparison outside the actual post-word family. The rigid
relative-rotation control checks old and mixed cycle information and treats
the mixed winding as unavailable at its cut. These are algebraic and reader
controls, not evaluations of the unknown source trajectory or constructions
of the unspecified global path from Section 21.
