# Two-port compatibility of unequal sine winding classes

<a id="sine-two-port-compatibility"></a>

## Prospective claim and frozen protocol

The [central-port composition](SINE_REDUCED_PORT_COMPOSITION.md#sine-reduced-port-composition)
retains all contacts at one node of each C9 component. Its compatible target
has aligned contact phases and unchanged internal twists. A second contact
at a different node creates an additional cycle and need not preserve that
geometry. This gate asks whether the already specified winding classes one
and two admit a joint acute equilibrium on one fixed two-port support.

The general methods are already owned: the
[acute circulation and uniqueness result](SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility)
and the full-law stability argument below apply to connected sine supports.
The [native C5 return-path example](RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-equilibrium)
also has deformed geometry and nonzero interface current at zero mixed period,
under its different complete law. The present obligation is the concrete C9
compatibility admission, not the first occurrence of either phenomenon.

Freeze this protocol before the new reserved assessment:

- Use eighteen nodes: donor cycle `0->1->...->8->0` and receiver cycle
  `9->10->...->17->9`, with the two unit contacts `(0,9)` and `(1,10)`.
  The graph has twenty edges. Nodes `0,1,9,10` have degree three; all
  others have degree two. Both evolution rows use those full degrees.
- Retain the complete reciprocal normalized-sine law, unit held capacities,
  `beta=1`, `e=1023/1024`, `w=1/1024`, scaled clock `tau=e*t`, and
  `gamma=1/(1023*pi)`. No forcing, support change, state reset or
  separately installed contact law occurs in the assessed model.
- The primary winding pair is `(2,1)`. The oriented interface cycle is
  `0->9->10->1->0`, with period zero. Use the acute edge branch on every
  edge. Form has degree-weighted mean zero, and the one global lifted
  phase gauge also has degree-weighted mean zero. Do not fix separate
  component means or supply an independently selected relative phase offset.
- Predict one acute critical geometry modulo the common phase, with
  nonzero opposed contact sine currents and deformed internal twists.
  Predict that no independent component rotation can make both unequal
  isolated twists an acute equilibrium on this support. The matched
  `(1,1)` control must retain the undeformed twists and zero contact current.
- Enclose the implicit critical geometry by the monotone scalar construction
  below. Use exact rational turn coordinates and the shared outward
  trigonometric intervals. Use thirty-two outer bisection refinements and
  sixty-four inner refinements. An unresolved required sign is unavailable;
  do not infer it from a floating midpoint, change the refinement budget
  after inspecting the result, or substitute a small residual for existence.
- Require the retained primary bracket to enclose the proved root, its
  outward contact-current lower bound to be strictly positive, every
  outward acute margin to be strictly positive, and the full-node balance,
  cycle periods, weighted gauge and matched-class control to agree with
  the exact construction. The full-law local attraction conclusion must
  follow from a positive relative Hessian, not a numerical eigenvalue sign.
- Preserve the protocol, producing source and proof before the first
  assessment. There is no finite response horizon or sensor-error budget:
  the observation is a static compatibility certificate for an implicit
  equilibrium. No trajectory, parameter search or previous producer is run.

This is an equilibrium and local-recovery admission. It does not transfer
the earlier prepared sources into the new basin, prove attraction from
arbitrary relative origins, select the support, fund an attachment event or
identify a physical bond. The existing thirty-coordinate central-port
surrogate is not assumed valid for this different interface.

## Complete rows, means and the acute cycle cell

Let \(L\) be this graph's unit Laplacian, \(\mathsf D\) its degree
matrix, \(K=\mathsf D^{-1}\), and
\(S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i)\).
In the declared clock the actual rows are
\[
x'=-KLx+\gamma KS(\theta),\qquad
\theta'=\gamma KLx. \tag{1}
\]
Their conserved charges are
\(\mathbf1^T\mathsf D x\) and
\(\mathbf1^T\mathsf D\theta\), where the latter uses continuously
chosen real lifts. The degree mass is forty. Only one common form and one
common phase are removed. Individual component means generally evolve.
Changing a lift representative by nodewise integer multiples of \(2\pi\)
does not change the circular state or (1); the declared weighted gauge fixes
one representative after the edge branches have been selected.

Orient the two rings as above and each contact donor-to-receiver. Denote
their principal acute edge gaps by \(q_e\in(-\pi/2,\pi/2)\).
The two ring cycles and the four-edge interface cycle form an integer cycle
basis. Removing the chords `(0,8)`, `(9,17)` and `(1,10)` leaves a spanning
tree; the three displayed cycles are its fundamental cycles, up to orientation.
The connected graph has cycle rank \(20-18+1=3\). Their required
periods are \((2,1,0)\). In fact an acute four-edge cycle has increment
sum strictly between \(-2\pi\) and \(2\pi\), so its integer period
must be zero.

On this fixed period cell, admissible edge gaps are the intersection of an
affine cycle-constraint space with the open acute cube. That set is convex.
The phase potential
\[
U(\theta)=\sum_{\{i,j\}\in E}[1-\cos(\theta_j-\theta_i)]
\]
is strictly convex in those edge coordinates. Equivalently, its Hessian is
\[
\mathcal H(\theta)=B^T\operatorname{diag}(\cos q_e)B,
\tag{2}
\]
for an edge-difference incidence matrix \(B\); it is positive definite
on relative nodal variations. Consequently there is at most one acute
critical geometry in the cell, modulo common phase. Existence still needs
proof; convexity alone does not prevent a minimum on the acute boundary.

This formulation fixes cycle periods rather than treating arbitrary corners
of independent edge intervals as realizable states. Two nodal lift choices
reconstructing the same edge gaps differ only by integer vertex shifts and
a common phase, and do not supply distinct relative circular geometries.

## Exact reduction of the critical equations

At any equilibrium, the phase row and positive capacities imply \(Lx=0\).
The zero form-mean leaf therefore has \(x=0\), and its form row then
requires \(S(\theta)=0\). Every nonport ring node has degree two.
Its sine balance equates its two oriented ring currents; injectivity of sine
on the acute branch equates their gaps.

Write \(a\) for the donor gap on `0->1` and \(b\) for each of the
other eight donor gaps. Write \(c\) for `9->10` and \(d\) for the
other eight receiver gaps. The ring periods give
\[
a+8b=4\pi,\qquad c+8d=2\pi. \tag{3}
\]
Let \(\delta_0,\delta_1\) be the contact gaps on `0->9` and
`1->10`. Summing nodal balance over one ring gives
\(\sin\delta_0+\sin\delta_1=0\). Both are acute, so
\(\delta_1=-\delta_0\). The interface period gives
\(\delta_0+c-\delta_1-a=0\). Hence, writing \(\delta=\delta_0\),
\[
\delta=\frac{a-c}{2},\qquad
\sin b-\sin a=\sin c-\sin d=\sin\delta. \tag{4}
\]
Conversely, any acute solution of (3)--(4) supplies every fine-node balance.
For example the donor node-zero row is
\(\sin a-\sin b+\sin\delta=0\), and the receiver node-nine
row is \(\sin c-\sin d-\sin\delta=0\). The opposite port rows
are their negatives; all fourteen nonport rows cancel along their arcs.
Degree three multiplies a zero port sum and is retained in both full rows.

An explicit real lift is
\[
\begin{aligned}
&\theta_0=0,\qquad \theta_j=a+(j-1)b\quad(1\le j\le8),\\
&\theta_9=\delta,\qquad
\theta_{9+j}=\delta+c+(j-1)d\quad(1\le j\le8).
\end{aligned} \tag{5}
\]
The closing donor and receiver increments differ from their principal gaps
by \(-4\pi\) and \(-2\pi\), respectively. Both contact gaps are
as in (4). Subtracting the single affine weighted mean
\(\sum_i d_i\theta_i/40\) imposes the declared gauge without changing
any gap. This is a correlated implicit target, not a target obtained by
rounding each phase independently.

## A strictly monotone compatibility equation proves existence

Put \(\alpha=2\pi/9\), and define
\[
h_k(s)=\sin\frac{2\pi k-s}{8}-\sin s.
\]
On all intervals used below,
\[
h_k'(s)=-\tfrac18\cos\frac{2\pi k-s}{8}-\cos s<0. \tag{6}
\]
For \(a\in[\alpha,2\alpha]\), put \(g=h_2(a)\ge0\).
There is a unique \(c=c(a)\in[\alpha,2\alpha]\) satisfying
\(h_1(c)+g=0\). The lower endpoint has \(h_1(\alpha)=0\).
For the upper endpoint, the exact inequality
\[
\begin{aligned}
-h_1(2\alpha)-h_2(\alpha)
 &=\sin(4\pi/9)+\sin(2\pi/9)
   -\sin(7\pi/36)-\sin(17\pi/36)\\
 &=2\sin(\pi/3)
   [\cos(\pi/9)-\cos(5\pi/36)]>0
\end{aligned} \tag{7}
\]
provides strict bracketing whenever \(g>0\). At \(a=2\alpha\),
\(g=0\) and \(c=\alpha\) exactly. Implicit differentiation gives
\(c'(a)=-h_2'(a)/h_1'(c)<0\).

Define
\[
F(a)=h_2(a)-\sin\frac{a-c(a)}2. \tag{8}
\]
Here \(|a-c(a)|/2\le\pi/9\), so (6) implies
\[
F'(a)=h_2'(a)-\tfrac12\cos\frac{a-c(a)}2[1-c'(a)]<0.
\]
At \(a=\alpha\), one has \(g>0\) and \(c>\alpha\), hence
\(F(\alpha)>0\). At \(a=2\alpha\), one has
\(F(2\alpha)=-\sin(\pi/9)<0\). The intermediate value theorem
and strict monotonicity give exactly one root in the open interval.

At this root \(g>0\), so (8) forces \(a>c\). All gaps satisfy
\[
\alpha<c<a<2\alpha,\qquad
2\alpha<b<17\pi/36,\qquad
7\pi/36<d<\alpha,\qquad
0<\delta<\pi/9. \tag{9}
\]
Thus the constructed target is acute, with every margin strictly larger
than \(\pi/36\). Equations (3)--(5) prove existence in the declared
period cell. Its uniqueness among all acute geometries in that cell follows
from (2), not just from restricting the search to arc-constant candidates.

The matched `(1,1)` control has
\(a=b=c=d=\alpha\) and \(\delta=0\). It satisfies every full row
and the same interface period. Strict convexity again gives uniqueness in
its own acute cell. The corresponding statement for matched `(2,2)` uses
\(2\alpha\); swapping the two rings maps `(2,1)` to `(1,2)` and
reverses the oriented contact current without changing its magnitude.

## What incompatibility of the isolated twists means

Keep the two isolated twists undeformed, while allowing any independent
common rotation of either component. Each internal sine sum is then zero.
At every port, the acute full-node balance forces its single contact gap to
be zero. But two zero contact gaps would give \(a=c\) around the
interface cycle, contradicting the unequal winding slopes
\(a=2\alpha\), \(c=\alpha\). Therefore no supplied relative
origin can make both undeformed unequal twists critical on this support.

The compatible state instead changes both short and long internal arc gaps
and derives its relative port phases from (4). Its nonzero opposed contact
currents are a stationary sine circulation. They are not a sustained transfer
of structural storage, a temporal oscillation or a measured energy current:
at the critical state every form and phase velocity is zero.

There is also no energetic binding conclusion from this compatibility.
Strict convexity of \(1-\cos s\) on the acute interval gives
\[
U(\theta_*)>
9[1-\cos(2\alpha)]+9[1-\cos\alpha]. \tag{10}
\]
Each isolated ring's fixed-period minimum is its uniform twist. The target
deforms those rings and adds the positive contact potential
\(2[1-\cos\delta]\). Thus its storage exceeds the sum of the two
isolated minima. A law for removing or creating edges and its work accounting
would be an additional model. Local attraction at fixed support does not
select that support or make its formation passive.

## Full-law local recovery without operator commutation

The exact fixed-support storage is
\[
H(x,\theta)=\tfrac12x^TLx+U(\theta),\qquad
H'=-(Lx)^TK(Lx)\le0. \tag{11}
\]
The mixed terms cancel between both rows of (1). At the target, (9) gives
\(\mathcal H_*\succeq\sin(\pi/36)L\), positive on the relative
space. On the fixed weighted-mean leaf the form energy and local phase
excess are both positive definite. Choose a sufficiently small closed
relative ball entirely inside the acute cell and a positive storage sublevel
strictly below its boundary. Equation (11) makes that sublevel invariant.
Its largest invariant zero-loss set has \(Lx=0\), hence \(x=0\),
then \(S(\theta)=0\), hence the unique target. LaSalle's argument
proves local asymptotic attraction for the full thirty-six-coordinate law,
after removing the two conserved common modes.

For the stronger local exponential statement, use the
[full reciprocal stability argument](SINE_PATTERN_DYNAMICS.md#full-nodal-stability-follows-under-positive-loss).
On \((K^{-1/2}\mathbf1)^\perp\), set
\(P=K^{1/2}LK^{1/2}>0\) and
\(Q=K^{1/2}\mathcal H_*K^{1/2}>0\). The phase variation can be
transformed to the quadratic pencil
\[
s^2I+sP+\gamma^2P^{1/2}QP^{1/2}. \tag{12}
\]
For an eigenvector, its scalar inner product has strictly positive real
mass, damping and stiffness coefficients. A nonreal root therefore has
strictly negative real part; a real root cannot be nonnegative. All relative
linear modes decay, so smoothness yields local exponential attraction.
Neither \(P\) and \(Q\) nor their eigenvectors are assumed to commute.

These are local statements around the new implicit target. They provide no
uniform basin for the old formation sources, no finite acquisition time,
no global attraction throughout the acute cell and no theorem spanning a
support event. Those would require their own complete-state admission.

## Certified scalar enclosure and implementation boundary

The implementation uses turns \(A=a/(2\pi)\), \(C=c/(2\pi)\)
and
\[
\widehat h_k(z)=\sin\frac{2\pi(k-z)}8-\sin(2\pi z).
\]
The outer bracket is \([1/9,2/9]\). For each admitted outer argument,
the inner decreasing equation is
\(\widehat h_1(C)+\widehat h_2(A)=0\) on the same bracket.
The upper outer endpoint uses the exact identity \(C=1/9\).
The decreasing outer residual is
\[
\widehat F(A)=\widehat h_2(A)-\sin[\pi(A-C(A))]. \tag{13}
\]
The shared [strict-sign root enclosure](../../src/tnfr/physics/phase_cycle_geometry.py)
bisects only when its outward interval proves a sign. The inner enclosures
are propagated through (13), retaining their uncertainty and operator order.
The fixed sixty-four inner refinements are independent of the thirty-two
outer refinements. A completed outer bracket has width at most
\((1/9)2^{-32}\) turns. No unproved sign or adaptively enlarged numerical
budget is used to force completion.

Endpoint monotonicity then encloses \(C(A)\), \(b,d,\delta\), the
full affine lift (5), the gauge, currents, storage and acute margins. Exact
reconstruction can retain the gauge as the affine turn expression
\[
\overline\Theta_{\mathsf D}
 =\frac{20A+7(k_{\mathrm D}+k_{\mathrm R})}{40}
 =\frac A2+\frac7{40}(k_{\mathrm D}+k_{\mathrm R}).
\]
For the two root residuals \(F=\widehat F(A)\) and
\(J=\widehat h_1(C)+\widehat h_2(A)\), the only nonzero symbolic
nodal rows are \(S_0=-F\), \(S_1=F\), \(S_9=F-J\) and
\(S_{10}=J-F\). This factorization vanishes at the implicit root.
The exact
periods and fine-row cancellations belong to the implicit correlated root;
they are not asserted for every corner of the reported marginal intervals.
Intervals containing zero in the computed residuals are implementation
consistency checks, not substitutes for the analytic existence proof.
The matched control uses its exact twist formulas and requires no root solve.

An unresolved numerical sign or margin leaves the represented certificate
unavailable while preserving the analytic theorem. Conversely, a successful
static certificate does not establish formation, an autonomous event selector,
physical time units, a sensor model or an energy-conjugate contact readout.

The shared
[`assess_sine_two_port_compatibility`](../../src/tnfr/physics/relational_sine_two_port_compatibility.py)
returns `SineTwoPortCompatibility` from the mandatory primitives `classes`,
`outer_refinements` and `inner_refinements`. It admits the class labels one
and two and ordinary integer refinement budgets from one through sixty-four;
the reserved protocol above fixes their selected values. It rebuilds the
actual graph, affine phase correlations and balance factorization rather than
accepting an incoming equilibrium report. The class-swap and matched-class
extensions retain the same support, complete law and fixed weighted means.

## Status at archival

At source archival the protocol and proof above were complete, and the
reserved assessment had not been performed. The following section records
the subsequent first assessment of that unchanged declaration.

## Retained two-port compatibility result

Both the primary `(2,1)` assessment and the matched `(1,1)` control returned
`certified_compatible`, with no unavailable reasons. All six frozen stopping
conditions passed: both implicit equilibria, the positive-versus-zero contact
current distinction, exclusion of the undeformed unequal twists, and strict
local attraction on the declared full-network mean leaf.

For the primary geometry, the completed donor short-arc bracket in turns is
\[
\frac{a}{2\pi}\in
\frac{[6573559492,\,6573559493]}{9\,2^{32}}.
\]
The stored outer residual intervals have strictly positive and negative signs
at the lower and upper endpoints, respectively. The fixed inner budget remains
sixty-four refinements. The following exact outward enclosures use the common
denominator \(2^{128}\):

| Quantity | Lower numerator | Upper numerator |
| --- | ---: | ---: |
| First contact gap \(\delta/(2\pi)\) | `6220678650586981850326865582201125091` | `6220678657852373978098316676520651436` |
| First contact sine current \(\sin\delta\) | `38999787950219209243187699610900432965` | `38999787995869014311299513956505139888` |
| Minimum acute edge margin, in turns | `7233486662907790580349073231255408184` | `7233486664008181726380522364499074162` |

These correspond approximately to a contact gap between
\(0.01828093153011452\) and \(0.01828093155146559\) turns, a sine
current between \(0.1146100760468745\) and \(0.11461007618102723\),
and an acute-margin lower bound \(0.021257306772491172\) turns.
The second contact has the opposite gap and current. The matched control
has exactly zero on both contacts and retains its uniform class-one twists.

The complete support has diameter nine and certified combinatorial Laplacian
gap lower bound \(2/81\). Its primary relative phase-Hessian lower bound is
\[
\frac{22657163684316022886871299769345594197}
     {6890717930149003885133335800493306281984}
\approx0.0032880701131567124>0.
\]
The actual degrees, zero weighted gauge, three integer cycle periods and
full-node incidence factorization agree with the exact construction. The
reported local attraction follows from this positive Hessian and (11)--(12).
Residual intervals containing zero are retained only as consistency checks.
No rounded point from the marginal phase intervals replaces the correlated
implicit equilibrium.

The retained [protocol](../../docs/assets/sine_formed_classes/two-port-compatibility-v1.protocol.json),
[source archive](../../docs/assets/sine_formed_classes/two-port-compatibility-v1.source.zip),
[primary and matched reports](../../docs/assets/sine_formed_classes/two-port-compatibility-v1.json)
and [evidence manifest](../../docs/assets/sine_formed_classes/two-port-compatibility-v1.manifest.json)
preserve the first assessment. The archive SHA-256 is
`6ce65e3155dc41f4c0156f103cdc04fdcc33ddbd294929db050b75e244782eb2`.
The archive retains the pre-evaluation status, while the protocol and
mathematical derivation preceding it are unchanged in this owner.

The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-two-port-compatibility),
[guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-two-port-compatibility)
and [independent controls](../../tests/physics/test_sine_two_port_compatibility.py)
cover exact reconstruction, integer cycle admission, fine-node balance,
independent coupled-equation checks, the full Hessian, class exchange and
explicit abstention when a required sign is unresolved.

This result admits a deformed, locally attracting geometry for two supplied
winding identities on the new fixed support. It does not establish a source
family that reaches it, a contact event or its work budget, global attraction,
stationary energy transport or a physical binding mechanism. Those distinctions
are unchanged by the successful static certificate.

## An origin-independent obstruction to a storage-only handoff

<a id="sine-two-port-handoff-obstruction"></a>

This separate analytic result asks whether the already proved
[target-free sector capture theorem](SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture)
can admit the undeformed class pair directly on the two-port support. It
does not revise the archived compatibility protocol or reevaluate its
producer. Under F4 the result is an exact obstruction to a specified proof
mechanism; no new trajectory, root solve or numerical campaign is involved.

Retain the fixed support, complete rows (1), positive loss, capacities and
clock above. Supply the two isolated uniform twists of winding two and one,
allowing any independent component phase origins. Form may be any admitted
finite signed vector. The relevant capture theorem would require the
complete initial state to be strictly acute and its total storage to be
strictly below a common lower bound on phase storage on every nonempty
signed boundary face of the same acute period cell. There are forty such
signed face candidates on the twenty-edge support.

The conclusion below excludes that scalar full-boundary storage criterion
for every relative origin. It also excludes the criterion for a specified
neighborhood of the undeformed phase family. It does not exclude convergence
under (1), a differently shaped invariant region, or a later same-law entry
into a capture sublevel after dissipation.

### Minimize over every component origin before constructing a witness

Write
\[
q=4\pi/9,\qquad c=2\pi/9,\qquad \delta=(q-c)/2=\pi/9,
\qquad f(s)=1-\cos s.
\]
The donor ring has all nine principal increments equal to \(q\), and
the receiver has all nine equal to \(c\). If the first contact gap is
\(r\), the second is \(r+c-q=r-c\), modulo full turns. Therefore
\[
\begin{aligned}
U_{\rm nom}(r)
 &=9f(q)+9f(c)+2-\cos r-\cos(r-c)\\
 &=9f(q)+9f(c)+2-2\cos\delta\cos(r-\delta)\\
 &\ge U_{\min}:=9f(q)+9f(c)+2f(\delta). \tag{14}
\end{aligned}
\]
This is an exact minimization over all relative origins, including those
whose contacts are not acute. Equality is attained by opposite contact gaps
\(\delta,-\delta\). Thus the comparison uses the most favorable origin
for total storage; it does not select an origin after a trajectory response.

### One feasible boundary state lies below that minimum

For \(0\le s\le s_b:=\pi/144\), assign donor edge `(1,2)` the gap
\(q+8s\), every other donor ring edge the gap \(q-s\), every receiver
ring edge the gap \(c\), and the two contacts the gaps
\(\delta-s/2,-\delta+s/2\). These are correlated gaps of an actual
nodal state. Explicit real lifts are
\[
\begin{aligned}
&\theta_0(s)=0,\qquad \theta_1(s)=q-s,\\
&\theta_j(s)=jq+(9-j)s &&(2\le j\le8),\\
&\theta_{9+j}(s)=\delta-s/2+jc &&(0\le j\le8).
\end{aligned} \tag{15}
\]
The closing donor and receiver lift differences require the same additions
\(4\pi\) and \(2\pi\) to recover their principal ring gaps. The
ring sums are exactly \(4\pi\) and \(2\pi\). The interface cycle
`(0,9,10,1)` has sum
\(2(\delta-s/2)+c-(q-s)=0\). Consequently the periods remain
\((2,1,0)\) throughout this path. These are the same integer basis cycles
used in the compatibility proof. Subtracting
\(\sum_i d_i\theta_i(s)/40\) gives the zero weighted phase gauge
without changing any edge gap. A common shift then places the witness on
any source's conserved phase-mean leaf. Choose its uniform form equal to
the source's conserved weighted form mean; its form storage is zero.
Thus the comparison does not rely on moving to a different conserved-mean
leaf. The zero-mean representative has witness form zero.

Every gap is strictly acute for \(s<s_b\). At \(s_b\), only edge
`(1,2)` reaches the positive acute face. Its gap is \(\pi/2\); the
other eight donor gaps are \(7\pi/16\), the receiver gaps are
\(2\pi/9\), and the contact gaps are \(\pm31\pi/288\). Hence this
is a feasible point on one of the actual full-cell boundary faces, with
phase storage
\[
U_b=1+8f(7\pi/16)+9f(2\pi/9)+2f(31\pi/288). \tag{16}
\]
The construction is an auxiliary geometric path, not a proposed solution
of (1) or evidence that the dynamics visits that boundary.

Let \(D(s)=U(\theta(s))-U_{\min}\). Then \(D(0)=0\), and
\[
D'(s)=8\sin(q+8s)-8\sin(q-s)-\sin(\delta-s/2). \tag{17}
\]
Throughout the displayed interval,
\(\sin(q+8s)\le1\),
\(\sin(q-s)\ge\cos(\pi/16)\), and
\(\sin(\delta-s/2)\ge\sin(31\pi/288)\).
The elementary inequalities \(1-\cos z\le z^2/2\),
\(\pi^2<10\), and
\(\sin z\ge2z/\pi\) for \(0\le z\le\pi/2\) give
\[
\begin{aligned}
D'(s)
&\le8[1-\cos(\pi/16)]-\sin(31\pi/288)\\
&<\frac5{32}-\frac{31}{144}=-\frac{17}{288}.
\end{aligned} \tag{18}
\]
Integrating this strict inequality and using \(\pi>3\) proves
\[
U_{\min}-U_b>
\frac{17\pi}{41472}>
\frac{17}{13824}>0. \tag{19}
\]
No approximate energy comparison or unproved minimizer is needed.

If \(B\) is any valid common lower bound on phase storage on every
nonempty acute boundary face, the exhibited face forces \(B\le U_b\).
For any nominal origin and any form state,
\[
H(x,\theta_{\rm nom})
 =\tfrac12x^TLx+U_{\rm nom}(r)
 \ge U_{\min}>U_b\ge B. \tag{20}
\]
Thus no improvement to the lower-bound solver can make
\(H(x,\theta_{\rm nom})<B\) true. This statement concerns a common
bound on the full boundary; it does not assert that every individual face
has low storage. If the initial contacts are not strictly acute, the
capture theorem already fails its initial-domain premise. If they are
acute, their four-edge cycle has period zero and (20) excludes its required
storage inequality in the correct \((2,1,0)\) cell.

### The obstruction includes an explicit phase-error family

Allow each node's continuous phase lift to differ from its nominal twist
by at most \(\eta\ge0\) radians, after retaining the independent
component origins. The errors may be correlated or independent; no
restriction on the admitted form vector is needed. The global inequality
\(|f(v)-f(u)|\le|v-u|\) gives, on all twenty edges,
\[
|U(\theta)-U(\theta_{\rm nom})|
 \le\sum_{\{i,j\}\in E}|e_j-e_i|
 \le40\eta.
\]
Consequently every state in this full phase family satisfies
\[
H(x,\theta)-U_b>
M(\eta):=\frac{17}{13824}-40\eta. \tag{21}
\]
Whenever \(M(\eta)>0\), the internal phase errors are also too small
to change the two acute ring periods: \(2\eta<\pi/18\), the smaller
nominal ring margin. An actual state with acute contacts therefore lies in
the same \((2,1,0)\) cell and fails its full-boundary storage criterion.
A state with nonacute contacts fails the criterion's domain instead.
Large form differences cannot repair this obstruction because the form
storage is nonnegative.

Choose the explicit rational phase-error budget
\[
\eta=1/65536,\qquad M(\eta)=137/221184>0. \tag{22}
\]
It defines the admitted family without using the implicit equilibrium
coordinates. The assessor's sufficient rule requires a strictly positive
displayed margin. It abstains when that margin is zero or negative; such
abstention does not claim that the sharper strict analytic inequality has
failed. In particular, the threshold \(\eta=17/552960\) is not promoted
to a positive certificate by rounding or by silently changing the rule.

The shared
[`assess_sine_two_port_handoff_obstruction`](../../src/tnfr/physics/relational_sine_two_port_compatibility.py)
returns `SineTwoPortHandoffObstruction` from the mandatory primitive
`phase_error_radius`. It admits a nonnegative exact or represented real,
reconstructs the actual support, rational witness lifts, weighted gauge,
integer periods and boundary edge, and computes (21) by exact rational
arithmetic. The report retains both positive and nonpositive margins; its
certificate never substitutes a cached verdict, rounded storage difference
or source trajectory for those premises. This family theorem does not
admit an observed node array or authenticate a supplied endpoint report.

The [previous uninterrupted formation-and-dwell proof](SINE_FORMED_CLASS_CONTACT.md#the-actual-formation-images-reach-the-contact-tolerance)
already supplies \(10^{-32}\) Euclidean phase-error bounds per component
under its own preparation, isolated support and declared dwell. Each
coordinate then satisfies the larger bound in (22), so these endpoint
families inherit the obstruction if inspected on the present two-port
support without resetting their state. The comparison holds for every
relative origin, including the former preparation's supplied origin. This
implication uses the proved endpoint error, not a transfer of the old
single-port capture verdict or its event-work allowance. It installs no new
support, changes no prior preparation and makes no two-port event passive.

The static compatible equilibrium and its local attraction remain valid.
The new result identifies a limitation of the proposed direct handoff: total
storage versus every acute face cannot certify these undeformed source
families. A convergence proof for them needs additional dynamical or
geometric information beyond this scalar criterion. The result supplies no
nonconvergence theorem, autonomous contact selector or physical identification.

<a id="sine-formed-two-port-admission"></a>
## Actual formed-source admission for two-port organization

**Result and boundary.** The actual isolated formation images admit a
complete two-contact event, an initially absent internal-deformation collar
with a sufficient maintenance domain, and a conditional deformation-sensitive
response test. This closes source/event/identity admission. It does **not**
prove that these images enter that domain. Their source-to-domain transit is
the remaining generative question; the earlier midpoint capture is not its
answer. No historical producer or new transit was executed for this result.

### Actual source, complete event and chart transport

Keep the [actual formed families](SINE_FORMED_CLASS_CONTACT.md#the-actual-formation-images-reach-the-contact-tolerance)
in their original local order `j=0,...,8`. Put
`alpha=2*pi/9`, `phi=1/1000`, `epsilon=10^-32` and
`m=(2046/9)*(355/113)^2`. Initial donor forms are `2*m*(j-4)` and
receiver forms `m*(j-4)`; nominal initial phases are zero and `phi`,
respectively. Each of the four initial form/phase residual rows has coordinate
bound `10^-10` and zero sum. Use their actual isolated complete-flow images
at `tau_c=100+10^13`, including acquisition at `tau=100` and the unchanged
unprobed dwell. The inherited per-component preparation storage ceiling is
`10^9`. This is a fully specified family of flow images, not a selected
36-number checkpoint or a new box of independently preparable endpoints.

Write these images as

\[
x_{D,j}=v_{D,j},\quad\theta_{D,j}=2\alpha(j-4)+u_{D,j},\qquad
x_{R,j}=v_{R,j},\quad\theta_{R,j}=\alpha(j-4)+\varphi+u_{R,j}.
\tag{A1}
\]

The four actual residual vectors have zero sum and Euclidean norm at most
`epsilon`. Preserve their source correlations and continuous lifts.
The retained nonlinear dwell theorem supplies these bounds; its historical
one-bridge joined-state and response fields are not the present source.

Start from the disjoint cycles. Insert unit edges `(D4,R4)` and `(D5,R5)`
simultaneously at `tau_c`, with no nodal jump or intermediate one-edge flow.
Hold capacities one, beta one, `e=1023/1024`, `w=1/1024`,
`tau=e*t` and `gamma=1/(1023*pi)`. On each side of the event use its
actual Laplacian `L`, degree matrix `D`, and `S_i=sum_j sin(theta_j-theta_i)`:

\[
x'=-D^{-1}Lx+\gamma D^{-1}S(\theta),\qquad
\theta'=\gamma D^{-1}Lx. \tag{A2}
\]

The four port degrees change from two to three in **both** rows; all others
remain two. The joined degree mass is forty. The event is supplied; neither
autonomous contact occurrence nor execution of the public Coupling operator
is claimed. The matched control retains the disjoint support from the original
preparation onward. It is not produced by disconnecting an evolved joined pair.

The old two-port label `i` maps to `j=(i+4) mod 9` on each component.
For `i=5,6,7,8`, the directly permuted donor/receiver lifts contain
`-4*pi`/`-2*pi`, respectively. Add `+4*pi` to those donor lifts and
`+2*pi` to those receiver lifts to restore the old increasing-ramp chart;
move every edge offset, residual, observer
and target representative with that chart. This increases the reported
weighted phase mean by `6*pi/5`, without changing any circular state.
In the original lifts, the actual post-event conserved means are

\[
\bar x_+=\frac{v_{D,4}+v_{D,5}+v_{R,4}+v_{R,5}}{40},\quad
\bar\theta_+=\frac\varphi2+\frac{3\alpha}{40}
 +\frac{u_{D,4}+u_{D,5}+u_{R,4}+u_{R,5}}{40}. \tag{A3}
\]

Remove only these common means when using the joined relative chart.
Individual component means evolve afterward. In the old ramp coordinates
the actual relative origin remains `phi`; the earlier midpoint source used
`alpha/2`. A common gauge does not equate them. Its eight-coordinate folded
reference, source box and capture time therefore do not transfer.

### Supplied work for the actual two-edge event

For `H_E=x^T L_E x/2+sum_edges(1-cos gap)`, the exact jump is

\[
W_2=\sum_{j=4,5}\left[\tfrac12(v_{R,j}-v_{D,j})^2+1-\cos g_j\right],
\quad g_4=\varphi+u_{R,4}-u_{D,4},\quad
g_5=\varphi-\alpha+u_{R,5}-u_{D,5}. \tag{A4}
\]

The second gap is not another `phi`. Declare an external support-work
allowance `W_contact=1/4` in these normalized storage units. This is a new
supplied budget, independently justified for the entire source family:

\[
0<g_4<1/500,\qquad 2/3<|g_5|<7/10,
\]
\[
\frac15<\frac29-\frac{2401}{240000}<W_2
 <\frac{49}{200}+\frac1{500000}+2\epsilon^2<\frac14. \tag{A5}
\]

The gap bounds follow from `25/8<pi<22/7` and the retained residual norms.
For the lower bound use `1-cos z >= z^2/2-z^4/24`; for the upper use
`1-cos z <= z^2/2`. The combined two-coordinate form-gap norm is at most
`2*epsilon`, giving form work at most `2*epsilon^2`. These are conservative
analytical bounds, not an evaluated event. In particular, the historical
`10^-6` allowance cannot fund this contact. Between events (A2) retains
`H'=-(Lx)^T D^-1(Lx)<=0`; the positive jump is separate from that loss.
No reservoir, passive attachment or physical energy conversion is inferred.

The actual initial internal gaps and both contact gaps are acute. Their
correlated mixed-cycle sum is exactly
`g4 + q_R,4 - g5 - q_D,4 = 0`, giving periods `(2,1,0)`.
This admits a chart, not capture. The preceding storage-only obstruction
still excludes direct admission by the full acute-boundary storage criterion.

### A common internal observation, absence and maintenance

For each original ring define its nine oriented internal gaps `q_C,j`.
Use `theta_C,j+1-theta_C,j` for `j=0,...,7`; for `8->0`, retain
`+4*pi` on the donor and `+2*pi` on the receiver. These fixed lifts are
available before contact, afterward and in the disconnected control.
They are continuous lifted increments, without repeated principal-value
wrapping. In the admitted acute collar they agree with the corresponding
circular edge representatives; leaving that chart does not reset a lift.
Let `q*` be the compatible target's internal gaps transported to these labels.
Its short edges are `4->5`; the other eight edges in each ring are bulk.

Declare the internal observation vector and its geometric collar

\[
\mathcal O(z)=((q_{D,j})_{j=0}^8,(q_{R,j})_{j=0}^8,Q^Tx),\quad
Q=e_{D,8}-e_{D,0},\qquad
\mathcal I:\ \max_{C,j}|q_{C,j}-q^*_{C,j}|<1/12. \tag{A6}
\]

The phase channels assess the deformed organization; the EPI difference
across the same donor bulk arc is its later response channel. They are not
the same scalar, nor is this vector asserted to be a sufficient dynamical
state. All channels have the same definitions on both supports.

The [retained target bound](SINE_TWO_PORT_DIPOLE.md#exact-local-coefficient-and-a-finite-full-law-error)
`a*/(2*pi)<7/40` gives a strict, independently fixed absence margin. Throughout
the collar,

\[
D_s:=4\pi/9-q_{D,4}
 >17\pi/180-1/12>1/5. \tag{A7}
\]

At the actual contact source, `|D_s|<=2*epsilon`. The same bound holds
under every later unprobed disconnected continuation, by the retained
isolated nonlinear dwell estimate. Thus its separation from the collar's
necessary short-arc signature exceeds `1/5-2*epsilon`. The support event
leaves (A6) unchanged: it does not itself create this internal organization.
This is stronger than declaring absence because an edge is missing.

The collar alone is not a maintenance or acquisition theorem. Use the
[existing full-state capture domain](SINE_TWO_PORT_CAPTURE.md#a-strict-full-state-basin-proves-subsequent-convergence)
on the actual joined conserved-mean leaf, with its transported exact target:

\[
\mathcal R^2=\|P_Dx\|_D^2+\|\theta-\theta_*^m\|_D^2<R^2,
\quad R=1/12,\qquad H-H_*<B=1/648000. \tag{A8}
\]

Every edge deviation is at most `R`, since its endpoint degrees are at
least two. The known target margin, relative spectral bound and full-law
loss give the boundary barrier `H-H* >= R^2/4500=B`.
First-exit exclusion keeps (A8) trapped and therefore maintains (A6)-(A7).
Acute uniqueness and the zero-loss invariant set give recovery to the exact
compatible geometry, with each member's own common means. These are reused
local theorems under matched law and support; no actual source entry into
(A8) has been proved. The actual source and unchanged event are outside
the collar, so a later acquisition would be a dynamical change.

### Conditional response of the internal deformation

The [interior dipole theorem](SINE_TWO_PORT_DIPOLE.md#sine-two-port-dipole)
supplies a suitable test once entry to (A8) is established at a **prospectively
fixed** handoff time `tau_b>=tau_c`. Do not import its old timestamp or source.
The isolated control has its own trapped near-twist family by the actual
formation/dwell theorem. Both supports can use the theorem's conservative
caps `X0=Y0=R` and the same nonlinear degree-metric warmup

\[
T=285934809600000,\qquad X=Y=2^{-40}. \tag{A9}
\]

These cap bounds follow from uninterrupted flow for duration `T`, conditional
on trapping; they are not narrowed preparations or claims of practical speed.
At common time `tau_b+T` supply `theta^+=theta^-+a Q`, `x^+=x^-`,
where `a=2^-12`. The old dipole nodes `4,5` map precisely to original
nodes `8,0`, both interior degree-two nodes in either support. The same
three bulk edges are affected; their lifted closing-edge offset is retained.
Hold the support and law through elapsed fast time `h=2^-10`. Read

\[
Y_G=Q^T[x_G(h)-x_G(0^-)],\qquad C=Y_J-Y_0,
\quad |\widehat C-C|\le4d,\quad d=2^{-50}. \tag{A10}
\]

Here `J` and `0` mean the joined and independently continued disjoint sine
models. Four scalar readings have separate error allowances. This is the
increment of the common EPI channel in (A6), without a new output fitted
after seeing a trajectory. The transported theorem's leading coefficient is

\[
L_c=2\gamma h\sin(3a/2)
 [\cos(b_0-a/2)-\cos(b_J-a/2)],\quad b_0=4\pi/9,
\quad b_J=(4\pi-a_*)/8. \tag{A11}
\]

It depends on the internal bulk deformation, not merely on the presence of
a contact. The theorem bounds the actual form background, phase residuals,
both full-support propagators and nonlinear remainder. Under (A8)-(A10),
its rational lower design bound gives `C_hat>2^-38`. Its phase-event work
is positive and below the separate allowance `2^-21` on each support;
the post-probe trapping inequalities preserve the internal collar in the
joined system and the isolated winding families in the control. A response
uses its own admission; it is not a derived event selector.

Only the conditional **sine-versus-disconnected-sine** proof is transferred.
The old phase-blind alternative uses a different initial form budget and
is not part of this declaration. In particular, the aggregate
`assess_sine_two_port_dipole` report bundles that additional source premise;
calling it unchanged would not admit this new source history. Reuse the
theorem and shared kernels after the new handoff is proved, and extend an
execution adapter only if that later consumer needs it. No new evaluator is
necessary for the present analytic admission.

### Decision and next missing implication

Source, simultaneous event, supplied work, common observation, initial
absence, sufficient maintenance domain and conditional response are now
specified together. The exact relabeling and rational design inequalities
are independently checked in artifacts/formed-two-port-admission-20261029;
the preserved review distinguishes these checks from a new root, response
or trajectory evaluation. Original evidence and source owners are retained.

What remains is precisely
`actual formation image -> unchanged two-port flow -> domain (A8)`
at a justified fixed time, with the original complete form and phase errors.
Admit one source-aware enclosure or analytical entry method, its horizon,
precision and total resource bound before evaluating transit. The actual
origin prevents reuse of the old reflection-folded reference. A full-state
or suitably justified source-specific reference is acceptable; compression
is optional. Failure of a chosen enclosure remains a method limitation unless
a separate argument excludes entry. No actual acquisition of this joint
organization, autonomous contact or physical correspondence is concluded.
