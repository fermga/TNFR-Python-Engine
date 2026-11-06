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

The retained [protocol](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/docs/assets/sine_formed_classes/two-port-compatibility-v1.protocol.json),
[source archive](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/docs/assets/sine_formed_classes/two-port-compatibility-v1.source.zip),
[primary and matched reports](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/docs/assets/sine_formed_classes/two-port-compatibility-v1.json)
and [evidence manifest](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/docs/assets/sine_formed_classes/two-port-compatibility-v1.manifest.json)
preserve the first assessment. The archive SHA-256 is
`6ce65e3155dc41f4c0156f103cdc04fdcc33ddbd294929db050b75e244782eb2`.
The archive retains the pre-evaluation status, while the protocol and
mathematical derivation preceding it are unchanged in this owner.

The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-two-port-compatibility),
[guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-two-port-compatibility)
and [independent controls](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/tests/physics/test_sine_two_port_compatibility.py)
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

### Source admission and its later resolution

Source, simultaneous event, supplied work, common observation, initial
absence, sufficient maintenance domain and conditional response are now
specified together. The exact relabeling and rational design inequalities
are independently checked in artifacts/formed-two-port-admission-20261029;
the preserved review distinguishes these checks from a new root, response
or trajectory evaluation. Original evidence and source owners are retained.

The missing implication at this stage was precisely
`actual formation image -> unchanged two-port flow -> domain (A8)`
at a justified fixed time, with the original complete form and phase errors.
A source-aware method and its horizon, precision and resource bound were
required before transit; the actual origin prevented reuse of the old
reflection-folded reference. The later
[actual-source exclusion](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-actual-exclusion)
closes that target-entry question negatively, and the
[critical-energy result](#sine-formed-two-port-critical-energy-exclusion)
excludes low-storage critical limits in the original chamber. These conclusions
retain the original complete source errors. Failure of an enclosure alone
would not have established them. No new method or transit is activated here;
autonomous contact and physical correspondence remain separate obligations.


<a id="sine-formed-two-port-stationary-reduction"></a>
## Complete stationary alternatives and limits of the global cut barrier

The [post-acute complete-law result](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-post-acute-control)
proves convergence of every member of the original `phi=1/1000` formed-source
family to an equilibrium, without selecting that equilibrium. The following
reduction retains all stationary branches, including saddles, principal cuts
and degeneracies. It is not a reduced evolution law. A separate exact Hessian
test classifies the nondegenerate local stability possibilities, and a
geometric boundary witness excludes one global energy-barrier criterion.
None of these static results decides the source's eventual basin.

Keep the original labels of (A1)--(A3): short edges `(4,5)` and `(13,14)`,
contacts `(4,13)` and `(5,14)`, and long paths
`5->6->7->8->0->1->2->3->4` and
`14->15->16->17->9->10->11->12->13`. Orient both rings in increasing local
order and contacts donor to receiver. Both complete rows use the actual
degree-three ports and degree-two remaining nodes. The support, clock,
coefficients, inherited source, memberwise means and continuous lifts remain
those of (A2); no relative origin is reset.

### Three stationary circulation parameters, with every sine branch retained

At equilibrium the phase row implies `Lx=0`, hence `x=mu_x*1` on the member's
form-mean leaf. The form row then requires `S(theta)=0`. The fourteen
degree-two node balances equate the oriented sine currents along each long
path, not necessarily the phase gaps. Write these currents as `p,q`, and
let `j` be the first contact current. The four port balances are equivalent to

\[
\begin{array}{c|cccccc}
\text{edge group}&P_D&P_R&4\to5&13\to14&4\to13&5\to14\\ \hline
J_e& p&q&p-j&q+j&j&-j
\end{array} \tag{SR1}
\]

For example, the rows at nodes 4 and 13 read
`(p-j)-p+j=0` and `(q+j)-q-j=0`; the opposite port rows follow. Thus the
current domain is the closed polytope

\[
\mathcal P=\{(p,q,j): |p|,|q|,|j|,|p-j|,|q+j|\le1\}. \tag{SR2}
\]

For each edge independently choose `sigma_e in {-1,+1}` and put
`z_e=sigma_e*sqrt(1-J_e^2)+i J_e`. These are all unit phasors with the
required sine. Let `Z_D,Z_R` denote the short-edge phasors and `Z_0,Z_1`
the contact phasors. Exactly three cycle conditions remain:

\[
Z_D\prod_{e\in P_D}z_e=1,\qquad
Z_R\prod_{e\in P_R}z_e=1,\qquad
Z_0 Z_R\overline{Z_1}\,\overline{Z_D}=1. \tag{SR3}
\]

Equations (SR1)--(SR3) are necessary and sufficient for a stationary relative
circular geometry. Indeed, removing the two ring closing edges and contact
`(5,14)` leaves a spanning tree. Integrating its phasors reconstructs every
node phase modulo one common origin. The three omitted edges agree exactly
when (SR3) holds. Their fundamental cycles are the two rings and
`4->13->14->5->4`. Every nodal sine balance is then already (SR1).

This formulation divides by no cosine. At `|J_e|=1`, the two cosine choices
coincide and cause harmless duplicate descriptions; at `J_e=0`, the negative
cosine choice includes an antipodal cut. Degenerate solutions and singular
equations are not excluded by an implicit-function or stability assumption.

For an explicit finite branch description, put
`alpha=arcsin(p)`, `beta=arcsin(q)`, `chi=arcsin(j)` in `[-pi/2,pi/2]`, and
use the representatives
`A_epsilon(t)=(-1)^epsilon*arcsin(t)+epsilon*pi`, `epsilon in {0,1}`.
Let `n_D,n_R in {0,...,8}` count the negative-cosine choices on the long
paths. Write the four other bits as `r,s,u,v`, and set
`a=A_r(p-j)`, `c=A_s(q+j)`. Then (SR3) is equivalent to

\[
\begin{aligned}
a+(8-2n_D)\alpha+n_D\pi&=2\pi k_D,\\
c+(8-2n_R)\beta+n_R\pi&=2\pi k_R,\\
[(-1)^u+(-1)^v]\chi+(u-v)\pi+c-a&=2\pi k_I.
\end{aligned} \tag{SR4}
\]

All representatives lie in `[-pi/2,3*pi/2]`, so the finite ranges
`k_D,k_R in {-2,...,6}` and `k_I in {-2,...,2}` suffice. These integers label
the chosen representatives, not necessarily principal windings. At a cut,
the latter are not a robust discrete identity. Reconstruct real nodal lifts
and the inherited affine closing offsets `+4*pi,+2*pi` consistently; retain
any integer lift information instead of imposing the original principal
periods on a possible post-cut limit. One common shift then matches the
member's phase mean. No component is independently recentered.

There are `9*9*2^4=1296` choices of long-path counts and short/contact bits
before restoring positions and the finite cycle integers. This counts
families of continuous equations, not roots or isolated equilibria. The
ordered placement of every long-path branch must still be retained: counts
suffice for cycle products and energy, but not for the labeled phase state,
its observations or a full perturbation calculation. No graph automorphism
or branch permutation is silently quotiented out.

The stationary storage is, exactly,

\[
\begin{aligned}
H_*=U_* =20
&-(8-2n_D)\sqrt{1-p^2}-(8-2n_R)\sqrt{1-q^2}\\
&-(-1)^r\sqrt{1-(p-j)^2}-(-1)^s\sqrt{1-(q+j)^2}\\
&-[(-1)^u+(-1)^v]\sqrt{1-j^2}. \tag{SR5}
\end{aligned}
\]

Uniform form contributes zero storage whatever `mu_x`. The all-acute
`(2,1,0)` solution reduces to (3)--(4), but it is only one branch of this
exhaustive description. The origin `phi` supplies initial data, not an
additional stationary equation: relative component origin is not conserved
after contact. Energy bounds can exclude alternatives only when justified
for those alternatives; neither (SR5) nor current balance identifies a basin.

### Saturated bulk currents are impossible; PSD has stronger necessities

In fact `|p|=1` is impossible at any equilibrium, without a PSD premise.
If `p=t in {-1,+1}`, all eight donor long-edge phasors equal `i t`, whose
product is one. The donor cycle forces `Z_D=1`, hence `j=t`. The contacts
are `Z_0=i t`, `Z_1=-i t`, so the interface cycle requires `Z_R=-1`.
Its sine balance gives `q=-t`. But the eight receiver long-edge phasors
then also have product one, forcing `Z_R=1`, a contradiction. Exchanging
the rings excludes `|q|=1`. Thus `c_D=sqrt(1-p^2)` and
`c_R=sqrt(1-q^2)` are strictly positive at every stationary solution.

Suppose now that the full phase Hessian
`mathcal H=B^T diag(cos g_e) B` is PSD. If one donor long-edge cosine were
`-c_D`, choose a variation zero at all four ports and on the receiver, with
increment one on that edge and `-1/7` on the seven other donor long edges.
These increments sum to zero, so they reconstruct a legitimate nodal
variation on the clamped path. Its quadratic form is at most

\[
-c_D+7c_D/49=-6c_D/7<0. \tag{SR6}
\]

The short and contact contributions vanish. The same argument applies to
the receiver. Therefore PSD forces all sixteen long-edge cosines positive;
their gaps have the single acute representatives
`b=arcsin(p)`, `d=arcsin(q)`, with `n_D=n_R=0`.

The contact cosines have equal magnitude `kappa=sqrt(1-j^2)`. For a variation
constant on each ring, the quadratic form is their sum times the squared
relative shift. If `kappa>0`, two negative contacts violate PSD directly.
Opposite signs give a zero quadratic form but a nonzero Hessian action at
the ports, also impossible for a PSD matrix. Thus both must equal `+kappa`.

The remaining possibility `kappa=0` is also impossible under PSD. Let
`j=t in {-1,+1}`. The interface forces `c=a+pi` modulo a turn, so
`sin c=-sin a` and the balances give `q=-p`. The acute long branches satisfy
`d=-b`; their ring closures then give `c=-a` modulo a turn. Hence
`sin a=+1` or `-1`. Since `p=sin a+t` and `|p|<1`, necessarily `p=0`.
Then `b=0` and the donor closure forces `a=0` modulo a turn, a contradiction.
Every stationary PSD candidate consequently has

\[
\cos g_e=c_D>0\ (e\in P_D),\quad
\cos g_e=c_R>0\ (e\in P_R),\quad
\cos\delta_0=\cos\delta_1=\kappa>0. \tag{SR7}
\]

These are necessary restrictions on PSD candidates, not exclusions from
the complete stationary reduction. In particular, arbitrary stationary
solutions with negative curvature remain possible limits unless their
source accessibility is separately excluded.

### Exact four-port Hessian test, not a four-node dynamics

Under the stationary premises (SR7), eliminate the fourteen long-path
interior variations by minimizing their positive Dirichlet quadratic.
An eight-edge path of cosine stiffness `c_D` contributes
`c_D*(w_5-w_4)^2/8`; similarly for the receiver. Set
`A_D=cos(a)+c_D/8`, `A_R=cos(c)+c_R/8`. For test variations define
`d_D=w_5-w_4`, `d_R=w_14-w_13` and port averages
`bar w_D=(w_4+w_5)/2`, `bar w_R=(w_13+w_14)/2`. The exact minimized form is

\[
\begin{aligned}
Q_{\rm ports}={}&2\kappa(\bar w_R-\bar w_D)^2\\
&+(A_D+\kappa/2)d_D^2-\kappa d_Dd_R
 +(A_R+\kappa/2)d_R^2. \tag{SR8}
\end{aligned}
\]

These averages belong to variations, not to conserved component state means.
The discarded interior quadratic is positive definite. Consequently the
full phase Hessian is PSD if and only if

\[
A_D+\kappa/2\ge0,\qquad A_R+\kappa/2\ge0,\qquad
\Delta:=A_DA_R+\frac\kappa2(A_D+A_R)\ge0. \tag{SR9}
\]

It is positive definite modulo the one common phase precisely when the
two-by-two block in (SR8) is positive definite, equivalently
`A_D+kappa/2>0` and `Delta>0`. No nonnegative short-edge-cosine assumption
was used in this condensation. Algebraic compensation of a supplied negative
short coefficient is not a proof that a stationary solution with that
coefficient exists: the complete cycle and balance equations still apply.

For the complete-law stability implication, use the
[relative mechanical coordinates](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-post-acute-control).
Let `L_n=D^-1/2 L D^-1/2` on the relative space and
`C=D^-1/2 mathcal H D^-1/2` there. The linearized two-row law has the
quadratic pencil

\[
P(r)=r^2L_n^{-1}+rI+\eta C. \tag{SR10}
\]

If `C` is positive definite, real `r>=0` is impossible; for a nonreal
root, the imaginary-part identity gives
`Re r=-||w||^2/(2 w^*L_n^-1 w)<0`. Thus the equilibrium is locally
exponentially attracting on its conserved-mean leaf. If `C` has a negative
direction, `P(0)` does also, whereas `P(r)` is positive for large positive
`r`; its derivative `2r L_n^-1+I` is positive definite. A positive real
root and a growing full-state mode follow. No commutation between `L_n`
and `C` is assumed. A PSD singular Hessian leaves zero modes whose nonlinear
behavior is not settled by this test. Even an unstable equilibrium can have
a stable set containing a particular preparation; instability alone does
not remove it from this source-sensitive investigation.

<a id="sine-two-port-original-chamber-psd"></a>
### The original principal chamber has only one stationary PSD geometry

Restrict to the **closed original principal chamber**: all fixed-branch gaps
belong to `[-pi,pi]` and their affine cycle sums are `(4*pi,2*pi,0)`.
Then every stationary phase with positive-semidefinite full phase Hessian
is the retained all-acute target, modulo the common phase. In particular,
there is no other PSD-degenerate stationary geometry in this chamber.
This combines full stationary balance with curvature; an arbitrary assignment
of cosines passing the Schur test is not enough.

By (SR7), every such candidate has constant acute long gaps `b,d` and
acute contacts `chi,-chi`. Write `a,c` for its two principal short gaps.
The complete balances and the actual cycle sums imply

\[
 a+8b=4\pi,\qquad c+8d=2\pi,\qquad
 2\chi+c-a=0,\qquad
 j=\sin\chi=\sin b-\sin a=\sin c-\sin d. \tag{SC1}
\]

Since `a,c in [-pi,pi]`, necessarily
`0<a<=pi`, `3*pi/8<=b<pi/2` and `pi/8<=d<=3*pi/8`.
PSD additionally requires the donor diagonal in (SR9):

\[
 \cos a+\frac{\cos b}{8}+\frac{\cos\chi}{2}\ge0. \tag{SC2}
\]

**First exclude large donor short gaps.** If `a>=7*pi/10`, concavity of
sine between `pi/6` and `pi/4` gives
`sin(pi/5)>=3/10+sqrt(2)/5>29/50`. Together with `cos b<1/2`,
(SC2)'s left side is below `-29/50+1/16+1/2=-7/400`, a contradiction.

For `pi/2<=a<=2*pi/3`, one has `5*pi/12<=b<=7*pi/16`,
`sin b>24/25` and `sin a>43/50`; hence `-1/25<j<7/50`.
For acute `chi`, the chord inequality gives `|chi|<=pi*|j|/2`.
Thus `9*pi/25<c<53*pi/75` and `d<41*pi/200`. Using
`|c-pi/2|<31*pi/150`, `cos t>=1-t^2/2`, `pi^2<10` and `pi<22/7`,

\[
 j=\sin c-\sin d>
 \frac{3539}{4500}-\frac{451}{700}
 =\frac{2239}{15750}>\frac7{50}, \tag{SC3}
\]

contradicting the earlier upper bound. The elementary sine bounds above
follow from `sin(5*pi/12)=(sqrt(6)+sqrt(2))/4>24/25` and
`sqrt(3)/2>43/50`.

For `2*pi/3<=a<=7*pi/10`, `b>=33*pi/80>pi/3` while
`sin a<=sqrt(3)/2`, so `j>0`. Also `sin a>=sin(3*pi/10)>4/5`,
whence `j<1/5`. The same chord bound yields
`0<chi<pi/10`, `7*pi/15<c<7*pi/10` and `d<23*pi/120`.
Since `sin c>161/200` and `sin d<253/420`, now

\[
 j>\frac{161}{200}-\frac{253}{420}
   =\frac{851}{4200}>\frac15, \tag{SC4}
\]

another contradiction. Here `sin(3*pi/10)=(1+sqrt(5))/4>161/200`
uses `sqrt(5)>111/50`; sine's minimum on the stated `c` interval is at
its upper endpoint. These strict inequalities cover the shared endpoints.
Consequently `0<a<pi/2`.

**The receiver short gap is acute too.** If `c<=0`, (SC1) makes `chi>0`
and `j>0`, but `sin c-sin d<0`. If `c>=pi/2`, then `chi<0` and
`j<0`. Yet `b>7*pi/16` gives `sin b>6/7`, so `j>-1/7` and
`chi>-pi/14`. Therefore `pi/2<=c<9*pi/14<2*pi/3` and
`d<=3*pi/16`. These imply `sin c>4/5` and `sin d<3/5`, forcing
`j>1/5`, again impossible. Thus `0<c<pi/2`.

All twenty gaps are acute. The existing compatibility and strict-convexity
proof identifies exactly the retained target on this affine cycle leaf;
its connected positive-cosine graph makes the relative Hessian positive
definite. Every *other* stationary geometry in the closed chamber therefore
has a negative Hessian direction and, by (SR10), a growing complete-law mode,
including any other critical state on its principal-cut boundary.

This removes a different stable or PSD-degenerate destination **within this
chamber**, not every other possible limit. An unstable equilibrium can still
have a stable set containing an actual preparation. No source transversality,
measure or genericity assumption has been supplied to exclude that event.
This classification alone does not establish continued chamber membership;
the finite retained-data observation below supplies that separate statement.
This is an exact analytical classification, without root enumeration, a new
static evaluator or a source-arrival verdict.
The later source-specific critical-energy exclusion below removes the
low-storage alternatives without assuming generic avoidance of stable sets.

### A principal-cut witness below every source member's initial storage

The earlier [handoff obstruction](#sine-two-port-handoff-obstruction)
used a boundary at `pi/2`. Its correlated path (15) extends to an actual
principal cut, but remains a geometric construction rather than a flow.
Use that formula's old local labels with ports `0,1`, then transport them
to the present labels by `j=(i+4) mod 9` on each ring. This direction uses
the inverse of the chart transport accompanying (A3): for old indices `i=5,...,8`, subtract `4*pi` from the
donor lifts and `2*pi` from the receiver lifts. The inherited closing offsets
are then `+4*pi,+2*pi` in the original chart. The varying edge `1->2` is
thus the original `5->6`.

Put `q_0=4*pi/9`, `c_0=2*pi/9`, `delta=pi/9` and
`s_pi=5*pi/72`. The gaps of (15) on `0<=s<=s_pi` are one donor
`a=q_0+8s`, eight donor `b=q_0-s`, nine receiver `c_0`, and contacts
`+(delta-s/2),-(delta-s/2)`. The affine cycle sums remain `(4*pi,2*pi,0)`.
For `s<s_pi` every gap lies strictly between `-pi` and `pi`; at `s_pi`
only `a` reaches `pi`, while `b=3*pi/8` and the contacts are
`+11*pi/144,-11*pi/144`. The endpoint is on the closure of that principal
period chamber, without assigning it a cut-independent principal winding.

For `f(t)=1-cos(t)`, the potential derivative is

\[
U'(s)=8[\sin(q_0+8s)-\sin(q_0-s)]-\sin(\delta-s/2). \tag{SR11}
\]

On `0<=s<=pi/63`, `3*pi/7<=b<=4*pi/9` and
`delta-s/2>=13*pi/126`. The same elementary inequalities used in (18) give

\[
U'(s)\le8(1-\cos(\pi/14))-\sin(13\pi/126)
 <\frac{10}{49}-\frac{13}{63}=-\frac1{441}. \tag{SR12}
\]

On the remaining interval `pi/63<=s<=5*pi/72`, use
`sin(a)-sin(b)=2 cos(q_0+7s/2) sin(9s/2)`. The cosine argument lies in
`[pi/2,11*pi/16]`, the second sine is positive, and the contact angle
remains strictly positive and below `pi/2`. Hence `U'(s)<0` there too.
Since `U(0)=U_min` from (14), integration yields

\[
U_{\rm cut}<U_{\min}-\frac{\pi}{27783}
 <U_{\min}-\frac1{9261}. \tag{SR13}
\]

For every actual source member (A1), each phase residual has magnitude at
most `epsilon=10^-32`. Each of twenty edge gaps therefore differs from its
nominal value by at most `2epsilon`. Because `|f'|<=1` globally and form
storage is nonnegative,

\[
H_{\rm source}\ge U_{\rm nom}(\varphi)-40\epsilon
 \ge U_{\min}-40\epsilon>U_{\rm cut}. \tag{SR14}
\]

The middle inequality minimizes over origins only to obtain a lower bound;
the actual source still has `phi=1/1000`. Match the witness to each member's
phase mean by one common shift and choose its form uniformly equal to that
member's form mean. This changes no gap or potential and adds no form
storage. It is not a substitution for the member's actual state.

Any common lower bound on potential over **all** principal-cut faces of this
full period chamber must be at most `U_cut`. Thus the sufficient condition
that the initial total storage lie below such a common boundary bound fails
for every source member, with gap greater than `1/9261-40epsilon>0`.
This excludes that global full-cut barrier criterion only. It proves no
source-to-witness path, connection within an energy sublevel, actual winding
slip, target non-entry, or failure of a selective basin boundary argument.

### Static instruments and the remaining dynamical premise

The implementation owner `tnfr.research.formed_two_port_stationary` separates
these static constructions:

- `assign_formed_two_port_circulation` checks (SR1)--(SR2) on the original
  support. A circulation alone is not a realizable phase state.
- `admit_formed_two_port_rational_phases` checks supplied rational unit
  phasors and the three cycle products. It reports `phase_stationary` from
  exact nodal sine balance; closed geometry with a false flag remains
  admissible. It is not a root finder or a complete enumerator of
  irrational stationary phases.
- `condense_formed_two_port_hessian` computes the conditional Schur form for
  supplied cosines with positive long paths and equal positive contacts.
  It neither authenticates those cosines as a stationary phase state nor
  installs a four-node dynamics.

The mathematical branch reduction is exhaustive; these finite instruments
have the narrower input scopes just stated. No new root, trajectory or
reserved-producer replay belongs to this result. The original source's
basin remains undecided: a useful discriminator must connect its actual
post-contact flow to a particular basin, or establish separation from one,
rather than infer accessibility from stationary geometry or instability.

<a id="sine-formed-two-port-low-storage-critical-limits"></a>
## Low-storage critical limits in the original principal chamber

The following result restricts possible limiting equilibria of the
unchanged original A2 source family. It performs no trajectory, field/sign
evaluation, root search, branch enumeration or parameter scan. It does not
assert that any remaining branch contains a root or is reached by the source.

The [retained AI6 certificate](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-actual-exclusion)
and complete-law storage dissipation imply that
each limiting equilibrium supplied by [PA11](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-post-acute-control) satisfies

    U_infinity = H_infinity < E0 := 958021/100000 = 9.58021.

The question here is conditional: suppose that limit belongs to the **closed
original principal chamber**, all twenty fixed lifted gaps in `[-pi,pi]`, with
cycle sums `(4pi,2pi,0)`. No continued membership of the actual trajectory in
that chamber is assumed. In particular, this result does not infer a winding
change from the already proved exclusion of one target.

### Notation and complete stationary premises

Use [SR1--SR5](#sine-formed-two-port-stationary-reduction) above. Donor and
receiver long paths each have eight edges with sine currents `p,q`; their
short edges have gaps `a,c` and currents `p-j,q+j`. The contacts have currents
`j,-j`. Write their principal gaps as `delta0,delta1`. The actual affine
relations are

    a + sum(donor long gaps) = 4pi,
    c + sum(receiver long gaps) = 2pi,
    delta0 - delta1 + c - a = 0.

These equations retain branch positions and all real lifts. Counts used for
energy do not identify labeled states or quotient graph permutations.
At a complete equilibrium form is uniform and contributes zero storage.
No positive-semidefinite Hessian, stability or genericity premise is imposed.

### 1. A global supporting-line lower bound

For every `g in [-pi,pi]`,

    1-cos g > (5/8)g - 13/64.                         (L1)

Here is an elementary verification, including the nonconvex part of the
principal interval. Put `h(g)=1-cos g-(5/8)g+13/64`. For `g<=0`, positivity
is immediate. On `[0,pi]`, the only possible minima are zero, pi and
`alpha=arcsin(5/8)`; the other stationary point `pi-alpha` is a maximum.
Using `sqrt(39)<3123/500` and `alpha<1351/2000`,

    h(alpha) > 1-3123/4000-6755/16000+13/64 = 3/16000 > 0.

The latter angle bound follows from the alternating lower polynomial
`sin b >= b-b^3/6+b^5/120-b^7/5040 > 5/8` at `b=1351/2000`.
Also `h(0)=13/64>0`, and `pi<22/7` gives
`h(pi)>2-55/28+13/64>0`. These bounds prove L1 on the entire closed interval.

Summing over the two rings and using `pi>157/50` gives

    U_D > 5pi/2 -117/64 > 1927/320,
    U_R > 5pi/4 -117/64 > 671/320,
    U_D+U_R > 1299/160.                              (L2)

These are bounds for arbitrary admitted ring gaps, not stationary ansatzes.

### 2. Positive nonsaturated bulk currents and acute contacts

If `p<0`, all eight donor long gaps lie in `(-pi,0)`, so even `a<=pi` cannot
give donor sum `4pi`. Thus `p>=0`; the receiver sum `2pi` similarly excludes
`q<0`. If `p=0`, every donor long gap is zero or a principal antipode. Its
cycle equation also makes `a` an integer multiple of pi. A sum of `4pi`
requires at least four antipodal ring edges, hence `U_D>=8`. L2 then gives
`U>8+671/320>E0`, a contradiction. Likewise `q=0` forces at least two
receiver antipodes, `U_R>=4`, and `U>4+1927/320>E0`.

The saturated-current exclusion already proved in SR1--SR7 does not use PSD
and excludes either `|p|=1` or `|q|=1`. Consequently

    0<p<1,   0<q<1.                                 (L3)

The contact cosines have the same absolute value because their currents are
opposite. Unless both cosines are strictly positive, their sum is nonpositive
and their combined potential is at least two. But L2 would then give

    U > 1299/160+2 = 1619/160 > E0.

This also excludes zero contact cosines. Therefore both contacts are acute,
their principal gaps are exactly `chi,-chi`, and

    -pi/2<chi<pi/2,   2chi+c-a=0,   j=sin chi.         (L4)

All cut and saturation endpoints have been included in these exclusions.

### 3. At most two negative donor long cosines

Set `alpha=arcsin p in (0,pi/2)` and let `n_D` count donor long gaps equal
to `pi-alpha`; the other gaps equal `alpha`. Principal positivity follows
from L3. The donor closure is

    a=(4-n_D)(pi-2alpha).

For `n_D>=4`, the long-cosine sum `(8-2n_D)cos alpha` is nonpositive,
so `U_D>=8`, already incompatible with L2 and E0.
For `n_D=3`, put `z=cos alpha`; then `a=pi-2alpha`, and

    U_D=9-2z+cos(2alpha)
       =15/2+2(z-1/2)^2 >=15/2.

Thus `U>=15/2+U_R>3071/320>E0`. The exact margin at this coarse threshold is
`3071/320-E0=3333/200000>0`. We have proved

    n_D in {0,1,2},   m:=4-n_D in {4,3,2},
    a=m(pi-2alpha) in (0,pi],
    alpha=pi/2-a/(2m),   p=cos(a/(2m)).              (L5)

### 4. No negative receiver long cosine

Let `beta=arcsin q in (0,pi/2)` and `n_R` count receiver long obtuse gaps.
If `c<=0`, L5 gives `a>0`, so L4 gives `chi>0` and `j>0`. But
`q=sin c-j<0`, contradicting L3. Hence `c>0`. The receiver closure is

    c=(2-n_R)pi-(8-2n_R)beta.

For `n_R=2` this is negative. For `n_R>=3` it is below `-pi`:
for `3<=n_R<=4` this is immediate, while for `n_R>4`, `beta<pi/2`
gives `c<(2-n_R+n_R-4)pi=-2pi`. Thus only zero or one remain.

To exclude `n_R=1`, retain the donor correlation from L5:

    D_m(a)=9-cos a-2m sin(a/(2m)),   0<=a<=pi.

Its derivative is `sin a-cos(a/(2m))`. The only interior stationary points
are `a=m pi/(2m+1)` (minimum) and `a=m pi/(2m-1)` (maximum), as follows
directly by factoring the sine difference. The minimum value is

    9-(2m+1)sin(pi/(4m+2)) > 9-pi/2 > 69/10.

The other possible minima are endpoints. `D_m(0)=8`; monotonicity of
`t sin(pi/(2t))` for `t in [2,4]` gives

    D_m(pi) >=10-8sin(pi/8)>69/10.

For the final strict inequality use `sqrt(2)>7/5`:
`[8sin(pi/8)]^2=16(2-sqrt(2))<48/5<961/100`.
Thus `U_D>69/10` for each of the three remaining donor counts.

If `n_R=1`, closure gives `beta=(pi-c)/6`, and

    R_1(c)=9-cos c-6cos((pi-c)/6),   0<=c<=pi.

Its derivative `sin c-sin((pi-c)/6)` changes sign only at the minimum
`c=pi/7` in the interior; hence `R_1>=9-7cos(pi/7)>269/100`.
For an explicit rational check set `x=157/350<pi/7<1`. The alternating
cosine upper polynomial decreases on `[0,1]`, and

    631/700 - (1-x^2/2+x^4/24)
       =125956799/360150000000 >0.

Therefore `U>69/10+269/100=959/100=9.59`, exceeding E0 by
`979/100000`. It follows that

    n_R=0,   beta=(2pi-c)/8 in [pi/8,pi/4),
    q=sin beta.                                      (L6)

### 5. Ordered shorts and exclusion of all principal-cut critical limits

L5--L6 imply `p>=sqrt(2)/2>q`. Subtracting the two port balances and using
`a-c=2chi` yields the exact identity

    p-q=sin a-sin c+2j
       =2[1+cos((a+c)/2)]j.                          (L7)

For `0<a,c<=pi` the coefficient is nonnegative. Since the left side is
positive, the coefficient and `j` must both be positive. L4 then gives

    0<c<a<=pi,   0<chi<pi/2.                         (L8)

In particular `c=pi` is excluded. If `a=pi`, the donor balance gives
`j=p=cos(pi/(2m))`, while the contact gives `j=cos(c/2)`. Hence `c=pi/m`.
For `m>=3`, `sin(pi/m)<=cos(pi/(2m))=p`, so the receiver equation
`j=sin c-sin beta<sin c` contradicts `j=p`. For `m=2`, it instead gives
`j=1-sin(3pi/16)<1/2<p`, since `3pi/16>pi/6`. Thus `a<pi` too.

L3 already excludes long gaps at principal cuts, and L4 excludes contact
cuts. Therefore every remaining low-energy critical point in the closed
original chamber is actually in its **strict interior**. No critical limit
on a principal-cut boundary is left by this reduction. This is a conditional
limit classification, not a claim that the actual orbit remains inside.

### 6. Acute-short energies cannot satisfy AI6

For the remaining count `m`, the exact stationary energy expression is

    U_m(a,c)=20-cos a-cos c-2m sin(a/(2m))
              -8cos((2pi-c)/8)-2cos((a-c)/2).         (L9)

For each fixed positive `a`, the quantity `2m sin(a/(2m))` increases with
`m`: its derivative has the sign of `sin z-z cos z>0` for `z>0` in the
relevant interval. Therefore `U_m(a,c)>=U_4(a,c)`, strictly for `m<4`.

On the convex acute-short square `0<a,c<pi/2`, `U_4` is the actual potential
of the affine phase family with eight equal donor gaps `pi/2-a/8`, eight equal
receiver gaps `(2pi-c)/8`, shorts `a,c`, and contacts `+(a-c)/2,-(a-c)/2`.
Every edge is acute. The phase Hessian is positive definite modulo common
phase; its pullback to this rank-two affine family is strictly positive
definite. Thus `U_4` is strictly convex on that square. The retained target
lies in its interior and satisfies both derivatives, so

    U_m(a,c)>=U_4(a,c)>=U_target                      (L10)

there, and by continuity on its boundary. This is **not** asserted outside
the acute-short square.

If `a<=pi/2`, L8 also gives `0<c<a<=pi/2`; L10 contradicts AI6, which is
strictly below the retained target energy. Thus `a>pi/2`. Combining this
with `j=cos(a/(2m))-sin a>0` and the derivative signs in the donor-ring calculation above yields

    a>m pi/(2m-1).                                  (L11)

For `m=4`, the existing SC3--SC4 balance proof strengthens this to
`a>7pi/10`. That part of the existing proof uses acute long/contact gaps and
current balance only; its separate PSD inequality SC2 is not invoked.

### Necessary residual cases under the low-storage hypothesis

Any original-chamber critical limit compatible with AI6 must satisfy one of
the following count restrictions, retaining the actual positions of all
negative donor long cosines:

| Donor negative long count | m | Donor short restriction |
| --- | --- | --- |
| 0 | 4 | `7pi/10<a<pi` |
| 1 | 3 | `3pi/5<a<pi` |
| 2 | 2 | `2pi/3<a<pi` |

All have `0<c<a`, positive acute receiver long gaps, acute contacts with positive cosines,
and exact remaining equations

    sin((a-c)/2) = cos(a/(2m))-sin a
                 = sin c-sin((2pi-c)/8),
    U_m(a,c)<958021/100000.                           (L12)

This is a necessary reduction, not an existence or uniqueness statement.
Every such critical point, if one exists, has a negative full phase-Hessian
direction by the previously proved original-chamber PSD classification,
because the sole PSD target has been excluded. Zero modes may coexist with
that negative direction; no hyperbolicity assumption is made. Instability
does not exclude an actual preparation lying in its stable set.

These necessary constraints alone do not exclude every solution of L12.
The [critical-energy proof below](#sine-formed-two-port-critical-energy-exclusion)
closes that gap by contradiction, including unstable and degenerate candidates.
Its [integer-representative consequence](#sine-formed-two-port-original-period-exclusion)
also excludes arbitrarily late returns to the original period class. The
outgoing geometry, timing and common-destination question remain unresolved.

<a id="sine-formed-two-port-retained-geometry"></a>
## Retained geometry: paired-face passage without a principal-cut encounter

The closed 313-step actual-inclusion record can answer a finite geometric
question without reevaluating its flow, trigonometric bounds, Taylor jets or
target. The following retrospective observation retains the original source,
complete law, continuous lifts and memberwise conserved means. It consumes
the previously reviewed producer and corrected audit; it is not a new
trajectory or an independent recomputation of their primitive enclosures.

### Linear observation of the actual sets

Let `D=diag(d_i)`, with degree three at nodes `4,5,13,14`, degree two elsewhere,
and total mass forty. The retained mean-zero chart is
`Rq=(q_0,...,q_16,-sum_{i<17}d_i*q_i/d_17)`, with `G=R^T D R`.
Its exact covariance identity gives the squared dual norm of a nodal covector:

\[
 R G^{-1}R^T=D^{-1}-\mathbf1\mathbf1^T/40,\qquad
 s(\ell)=\sum_i\ell_i^2/d_i-(\sum_i\ell_i)^2/40. \tag{DG1}
\]

For an edge difference, `s=1/d_i+1/d_j`. Combine covectors before projecting
paired observations. If `gamma` is an outward rational lower bound for a
linear observation minus its cut at the retained center, then
`gamma>0` and `gamma^2>r^2*s(ell)` prove its strict positive side for the
entire actual metric ball. The radius is the inclusion certificate's **actual**
endpoint radius, including forcing, not its smaller nominal radius.
The admitted interval for the single constant pi is retained in the affine
expression; no independent copies of chart coordinates or pi are introduced.

For a whole-step observation, evaluate the combined chart covector directly
on the stored forced Picard tube. Its legacy `nominal_step.tube` field was
constructed with the bounded disturbance included; the reviewed producer and
audit bind it to the actual inclusion and growth-proof tube. Endpoint
polynomial bounds alone would not justify dense interpolation, and the dense
helper that rebuilds jets is not used here.

The edge order is the two increasing-label cycles, then `(4,13),(5,14)`.
The inherited donor and receiver closing offsets remain `+4*pi,+2*pi`.
Testing strict cells `(2m-1)pi<g_e<(2m+1)pi` for the fixed labels
`m=-4,...,4` gives the ordered edge labels and principal periods

\[
 (2-\sum_{e=0}^{8}m_e,\quad 1-\sum_{e=9}^{17}m_e,\quad
 -m_{18}-m_{13}+m_{19}+m_4). \tag{DG2}
\]

A full cell requires all twenty strict edge certificates. Failed sufficient
tests remain unknown. Different certified endpoint sides imply a memberwise
encounter by continuity, whereas identical endpoints alone do not exclude an
intermediate excursion. Whole-step certificates provide that stronger claim.

### Observed finite geometry

The independently reviewed policy observed exactly the initial time-one ball,
313 endpoint balls and 313 whole-step tubes, once. Every one of these 627 sets
certifies all twenty labels `m_e=0`. Thus, uniformly for the full source family,

\[
 |g_e(\sigma)|<\pi\quad\hbox{for every edge and }
 1\le\sigma\le329/16;\qquad (k_D,k_R,k_I)=(2,1,0). \tag{DG3}
\]

This proves absence of a principal-cut encounter throughout that interval;
it is not inferred merely from an empty endpoint-change list. Every tube
also certifies `0<a,c,B_2,B_3,B_4,D_1,...,D_4,w<pi/2` and
`-pi/2<h_c<pi/2`, using the inherited paired-coordinate definitions.

The first donor paired mean `B_1=(b_1+b_8)/2` behaves differently:

| Retained set | Certified statement about `B_1` |
| --- | --- |
| Checkpoints through `83/8` | Below `pi/2` |
| Checkpoints from `167/16` through `329/16` | Above `pi/2` |
| Whole-step tubes covering `[1,165/16]` | Below `pi/2` |
| Three tubes covering `[165/16,21/2]` | Side unresolved |
| Whole-step tubes covering `[21/2,329/16]` | Above `pi/2` |

The endpoint sides therefore prove that **every source member encounters
`B_1=pi/2` in `(83/8,167/16)`**. This brackets an encounter, not necessarily
its first or only encounter; times may differ between members. The three
unresolved tubes are retained as uncertainties, not converted into events.
The later tube certificates prove sustained `B_1>pi/2` on the stated finite
interval. No transversality or future invariance is asserted.

Consequently loss of paired acuteness and change of principal winding are
distinct events in this actual source: the former has occurred while the
latter has not occurred on the certified interval. Here paired acuteness means
the strict acute-angle condition on the paired means, not every individual
edge's acute condition. The earlier SG results concerning a possible first
paired encounter **by `7/4`** were conditional; they did not prove that such
an early encounter occurred. Their local estimates cannot be transferred to
this later passage.

The local evidence owner is
`artifacts/formed-two-port-destination-discrimination-20261101`: frozen method,
source receipts, independent readiness review, eleven passing synthetic
controls, complete `geometry-attempt/geometry-report.json`, post-write
supervisor result and independent outcome review. The observation closed in
about 6.923 seconds with 37,486,592 bytes peak committed worker memory; its
report occupies 6,017,595 bytes. Its fixed 120-second, one-GiB, sixteen-MiB
policy is consumed. All earlier raw responses remain unchanged.

<a id="sine-formed-two-port-critical-energy-exclusion"></a>
## Exclusion of low-storage critical limits in the original chamber

This analytical proof concerns the unchanged original A2 two-port source
family and complete law. It uses the [AI6 certificate](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-actual-exclusion),
[PA11 convergence](SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-post-acute-control)
and the [stationary reduction L1--L12](#sine-formed-two-port-low-storage-critical-limits). It performs
no trajectory, source replay, field/sign evaluation, root calculation,
parameter scan or trigonometric sample. Executed arithmetic is confined to
exact rational checks of the displayed polynomial inequalities and margins.

We prove that no critical point in the **closed original principal chamber**
can have storage below `E0=958021/100000`. Here the chamber means all twenty
fixed lifted gaps in `[-pi,pi]` and exact cycle sums `(4pi,2pi,0)`. The proof
retains unstable, degenerate and cut-boundary possibilities; it imposes no
positive-semidefinite Hessian or genericity assumption on a candidate.

For the actual source family this excludes any limiting equilibrium in that
closed chamber and forces a finite future departure after `sigma=329/16`.
It does not by itself establish a winding change, a departure deadline,
a common exit edge or a common destination.

### 1. Reviewed reduction and a new compact domain

At complete equilibrium the form is uniform, so `H=U`. The retained AI6
certificate and subsequent storage dissipation give, for every limiting
equilibrium supplied by PA11,

    U_infinity=H_infinity<E0=958021/100000.           (CE1)

Suppose a stationary point in the closed original chamber satisfies this
energy constraint. The reviewed L1--L12 reduction already excludes all
principal-cut critical points, zero or saturated bulk currents, nonacute
contacts, negative receiver long cosines, three or more negative donor long
cosines, and acute donor-short alternatives. Any remaining candidate has

    m in {4,3,2},    0<c<a<pi,
    a>7pi/10 if m=4,
    a>3pi/5  if m=3,
    a>2pi/3  if m=2,

    p=cos(a/(2m)),
    q=sin((2pi-c)/8),
    j=sin((a-c)/2)=p-sin(a)=sin(c)-q.              (CE2)

The two contact gaps are `+(a-c)/2` and `-(a-c)/2`, both with positive
cosine. Branch placements remain labeled; counts do not quotient states.

Put `s=(a+c)/2` and `d=(a-c)/2`. Since `m>=2`, `a<pi`, and `0<c<pi`,

    p>=sqrt(2)/2>7/10,
    q>sin(pi/8)>3/8,
    p+q>43/40.                                     (CE3)

The last radical inequality follows from
`sin^2(pi/8)=(2-sqrt(2))/4>9/64`, using `sqrt(2)<23/16`; the latter follows
by squaring positive numbers. Adding the short balances cancels `j`:

    p+q=sin(a)+sin(c)=2sin(s)cos(d).

Here `0<s<pi` and `0<cos(d)<=1`. Thus `sin(s)>43/80>1/2`, which implies

    a+c<5pi/3.                                     (CE4)

Every remaining candidate therefore belongs to the closed polygon

    P={3pi/5<=a<=pi, 0<=c<=a, a+c<=5pi/3}.         (CE5)

Passing to this larger closed domain only weakens the restrictions; it does
not drop any stationary endpoint or degeneracy.

### 2. A common potential lower bound

For a remaining stationary family the exact potential is

    U_m(a,c)=20-cos(a)-cos(c)-2m sin(a/(2m))
              -8cos((2pi-c)/8)-2cos((a-c)/2).

For fixed `0<a<=pi`, the function `2t sin(a/(2t))` increases on `2<=t<=4`:
its derivative is `2[sin(z)-z cos(z)]>0`, where `z=a/(2t)` lies in `(0,pi/4]`.
Consequently

    U_m(a,c)>=U4(a,c),    m in {2,3,4}.             (CE6)

Write

    U4(a,c)=D(a)+R(c)+C(a-c),
    D(a)=9-cos(a)-8sin(a/8),
    R(c)=9-cos(c)-8cos((2pi-c)/8),
    C(delta)=2-2cos(delta/2).

We now prove the purely static, stronger domain statement

    U4(a,c)>48/5=9.6,    (a,c) in P.                (CE7)

Stationarity is not used in the following domain proof.

### 3. Lower strip: `3pi/5<=a<=3pi/4`

The following closed-strip estimate retains its exact rational margins.

For `a>4pi/7` in `(pi/2,pi)`,
`sin(a)=sin(pi-a)<sin(pi/2-a/8)=cos(a/8)`, so `D` decreases throughout
this strip. Using `sqrt(2)/2>7/10`, `sin(x)<x` and `pi<22/7`,

    D(a)>=D(3pi/4)=9+sqrt(2)/2-8sin(3pi/32)
         >9+7/10-3pi/4>257/35.                     (CE8)

The principal-ring supporting line L1--L2 gives `R(c)>671/320` for
`0<=c<=3pi/4`. On the subinterval `[pi/3,3pi/4]` we also have

    R(pi/3)>43/20,
    R'(c)>9/40,
    R(c)>43/20+(9/40)(c-pi/3).                     (CE9)

Here is an explicit verification of CE9. Let
`Q4(x)=1-x^2/2+x^4/24`, `S5(x)=x-x^3/6+x^5/120`. They respectively decrease
and increase on `[0,1]`, and the alternating cosine/sine upper bounds give

    cos(5pi/24)<Q4(157/240)<127/160,
    127/160-Q4(157/240)=6903599/79626240000>0,

    sin(5pi/24)<S5(55/84)<5/8,
    5/8-S5(55/84)=1607882965/100370866176>0,

    sin(5pi/32)<S5(55/112)<19/40,
    19/40-S5(55/112)=7248857449/2114810019840>0.

The angle bounds use `157/50<pi<22/7`. Thus
`R(pi/3)=17/2-8cos(5pi/24)>43/20`. Moreover, with `beta=(2pi-c)/8`,

    (R')''(c)=-sin(c)+(1/64)sin(beta)<0

on `[pi/3,3pi/4]`. The concave derivative is bounded below by its endpoint
minimum. Those endpoints obey

    R'(pi/3)>6/7-5/8>9/40,
    R'(3pi/4)>7/10-19/40=9/40.

We used `sqrt(3)/2>6/7`, `sqrt(2)/2>7/10`; the first endpoint margin above
`9/40` is `1/140`. Integration proves CE9, including its left endpoint.

If `c<=pi/3`, then `(a-c)/2>=2pi/15>5/12`. Since the contact potential
increases in its half-gap on `[0,pi]`,

    C(a-c)>2-2cos(5/12)
             >=(5/12)^2-(5/12)^4/12>17/100.

The last margin is `6839/6220800`. Hence

    U4>257/35+671/320+17/100>48/5,
    257/35+671/320+17/100-48/5=109/11200>0.          (CE10)

If `pi/3<=c<=a`, put `delta=a-c>=0`. For `0<=delta<=pi`, the concave sine
chord on `[0,pi/4]` and `pi^2<10` give

    C(delta)=4sin^2(delta/4)>=2delta^2/pi^2>=delta^2/5.

This nonstrict estimate includes `delta=0`. Put `r=9/40`. Completing a
square in CE9 gives

    R(c)+C(a-c)
      >43/20+r(c-pi/3)+(a-c)^2/5
       =43/20+r(a-pi/3)-81/1280+(delta-5r/2)^2/5
      >43/20+9/50-81/1280,

because `a-pi/3>=4pi/15>4/5`. Adding CE8 yields

    U4>257/35+43/20+9/50-81/1280>48/5,
    257/35+43/20+9/50-81/1280-48/5=429/44800>0.    (CE11)

CE10--CE11 prove CE7 on the entire closed lower strip, including `c=pi/3`
and `c=a`. They do not need the extra constraint `a+c<=5pi/3`.

### 4. Upper strip reduces to four boundary segments

For `a>=3pi/4`,

    d^2 U4/da^2=cos(a)+(1/8)sin(a/8)
                   +(1/2)cos((a-c)/2)
               <=-sqrt(2)/2+5/8<0.                (CE12)

For any fixed feasible `c`, the admissible interval is

    max(3pi/4,c)<=a<=min(pi,5pi/3-c),
    0<=c<=5pi/6.

Strict concavity places a minimum at an endpoint (a singleton interval is
already a boundary point). The only possible endpoint segments are

1. `a=3pi/4`, `0<=c<=3pi/4`;
2. `a=c`, `3pi/4<=a<=5pi/6`;
3. `a=pi`, `0<=c<=2pi/3`;
4. `a+c=5pi/3`, `5pi/6<=a<=pi`.

The first is covered by CE10--CE11. We handle the other three without
root calculations, numerical extrema or adaptive angle subdivisions.

#### Boundary `a=pi`

For `0<=c<=2pi/3`,

    d U4(pi,c)/dc=sin(c)-sin((2pi-c)/8)-cos(c/2)<0.

Indeed `sin(c)<=1`, while the other two terms are at least `1/2`. Equality
would require all three equalities simultaneously; this is impossible
(the latter two equalities require `c=2pi/3`, whose sine is not one).
Thus the minimum on this closed segment is at `c=2pi/3`. There

    U4(pi,2pi/3)=43/2-8sin(pi/8)-5sqrt(3)
                 >43/2-22/7-26/3
                  =407/42>48/5.                   (CE13)

We used `8sin(pi/8)<pi<22/7` and `sqrt(3)<26/15`, the latter verified by
squaring positive numbers. The final margin above `48/5` is `19/210`.

#### Oblique boundary `a+c=5pi/3`

Here `5pi/6<=a<=pi`. Put `u=a/8` and `chi=a-5pi/6`. Then

    5pi/48<=u<=pi/8,    0<=chi<=pi/6,
    U4=20-8[sin(u)+cos(u+pi/24)]-(2-sqrt(3))cos(chi).

The bracket increases throughout its interval, since

    cos(u)-sin(u+pi/24)>0:
    cos(u)>=cos(pi/8)>1/2,
    sin(u+pi/24)<=sin(pi/6)=1/2.

Also `2-sqrt(3)>0` and `cos(chi)<=1`. Therefore

    U4>=18-8sin(pi/8)-3sqrt(3)
        >18-22/7-26/5
         =338/35>48/5.                             (CE14)

The final rational margin above `48/5` is `2/35`.

#### Diagonal boundary `a=c`

For `3pi/4<=a<=5pi/6`, write

    V(a)=U4(a,a)=18-2cos(a)-8sin(a/8)-8cos((2pi-a)/8).

Direct differentiation gives

    V''(a)=2cos(a)+(1/8)sin(a/8)+(1/8)cos((2pi-a)/8)
           <=-sqrt(2)+1/4<0.                       (CE15)

Its minimum is therefore at an endpoint. The endpoint `a=3pi/4` belongs
to the already proved lower strip, and `a=5pi/6` belongs to the oblique
boundary covered by CE14. Both are strictly above `48/5`, so the entire
diagonal segment is too. No additional trigonometric approximation is used.

CE12--CE15 and the lower strip prove CE7 throughout the closed polygon P.

### 5. Critical exclusion

Every remaining candidate CE2 lies in P. By CE6--CE7 it must satisfy

    U_m(a,c)>=U4(a,c)>48/5>E0,
    48/5-E0=1979/100000>0,                          (CE16)

contradicting CE1. The earlier L1--L12 reductions already treated all other
stationary branches and boundary possibilities. Consequently **there is no
critical point of storage below E0 in the closed original principal chamber**.

This is a scoped critical-energy exclusion. It does not assert that every
nonstationary state in that entire chamber has energy at least E0: CE7 was
proved only on P, which stationary balance forces for the residual branches.
It does not enumerate roots outside the chamber or identify another target.

<a id="sine-formed-two-port-future-boundary-alternative"></a>
### Actual-source departure

For the actual original source family, the retained endpoint observation DG3
places every member strictly inside the original principal chamber at
`sigma=329/16`; AI6 gives its storage below E0. PA11 supplies convergence of
each member's real phase lift to a critical equilibrium, with uniform limiting
form. If a member remained in the closed chamber at all later times, its
limit would remain there because the chamber is closed. That would contradict
CE16. Therefore every member has a finite later time outside that chamber;
continuity implies contact with a principal cut on the way out. In fact its
limit is outside the closed chamber, so each member is eventually outside it.

The time and crossed edge can depend on the member. This argument supplies
neither a uniform deadline nor a resolved crossing history. Departure from
this fixed-lift chamber is not, by itself, a proof of a changed robust winding
class: lift equivalence, simultaneous cuts and any later return require
their own argument. It also does not identify a common limiting equilibrium
or basin for the source family.

<a id="sine-formed-two-port-original-period-exclusion"></a>
## Eventual absence of the original periods, including cut representatives

This is an analytical consequence of the [critical-energy exclusion](#sine-formed-two-port-critical-energy-exclusion),
the fixed graph, AI6 and PA10--PA11. No source, phase field, trigonometric bound,
root, trajectory or retained observation is evaluated here.

The critical-energy premise used below is precise: **there is no stationary
phase in the closed original principal chamber with potential
`U<E0=958021/100000`**. The chamber uses the inherited fixed lifted cycle sums
`(4*pi,2*pi,0)`; its boundary is included. This premise is supplied by the
combined branch reduction and energy proof above, not by instability
of a candidate equilibrium alone.

### 1. Fixed graph and an integral cycle basis

Orient the twenty edges exactly as the producing owner:
donor edges `0,...,8` are `j->j+1 mod 9`, receiver edges `9,...,17` have
the corresponding orientation, and edges 18,19 are `4->13`, `5->14`.
Let `B` be the edge-by-node difference matrix, so `(B theta)_e=theta_j-theta_i`.
The fixed lifted gaps are

`g(theta)=B theta+2*pi*n`,

where `n_8=2`, `n_17=1`, and every other `n_e=0`. Let the three rows of `C`
sum the donor cycle, sum the receiver cycle, and take the mixed cycle
`e_18+e_13-e_19-e_4`. Then

`C B=0`, `C n=nu0=(2,1,0)`, and `C g(theta)=2*pi*nu0`.

The seventeen edges

`T={0,...,7,9,...,16,18}`

form a spanning tree. The omitted edges 8,17,19 close exactly the three
displayed fundamental cycles; their coefficients in the respective cycle
rows are `+1,+1,-1`. This is an integral cycle basis, not merely a real-rank
assertion.

**Integer reconstruction lemma.** If `m in Z^20` and `C m=0`, there exists
`k in Z^18` with `m=B k`. To prove it, set `k_0=0` and integrate the integer
edge increments `m_e` along the unique tree paths, changing sign when an edge
is traversed backwards. Thus `k` is integer and `B k=m` on `T`. Each chord
discrepancy is zero because its fundamental-cycle sum is zero. Connectedness
makes `k` unique modulo a common integer. No division or hidden congruence
condition is needed.

### 2. Any original-period principal representative gives an original-chamber state

For any fixed lifted phase, a closed principal representative is a choice

`h=g(theta)-2*pi*m in [-pi,pi]^20`, `m in Z^20`.

At an antipodal edge either endpoint sign may be allowed; do not presume a
unique wrapping convention. Suppose one such choice has original periods,
`C h=2*pi*nu0`. Comparing with `C g=2*pi*nu0` gives `C m=0`, hence `m=B k` by
the reconstruction lemma. The static representative

`theta_tilde=theta-2*pi*k`

then has `g(theta_tilde)=h`, so it lies in the closed original chamber.
Each edge sine and cosine is unchanged. Thus stationary balance and potential
are unchanged as well.

If the original phase has degree-weighted mean `mu_theta`, restore that same
mean with one common shift:

`theta_hat=theta_tilde + [2*pi*(sum_i d_i*k_i)/40]*1`.

This changes no gap, current or potential. It is a common shift only; no
individual ring is recentered. Before this common shift the circular nodal
phases are identical to the original ones; afterwards the relative circular
geometry is identical and the original mean leaf has been restored.
For a limiting complete state, leave the uniform form `mu_x*1` unchanged.
The resulting complete equilibrium has the same energy, form mean and phase
mean. This is a static representation used in a proof, **not** a reset,
replacement source or intervention in the actual dynamics.

Consequently a stationary circular geometry of energy below `E0` cannot
admit even one closed principal representative with periods `(2,1,0)`:
otherwise this construction would contradict the critical-energy premise.
The statement includes original-period-compatible choices at principal cuts.

### 3. Per-member eventual absence, including possible cut representatives

Fix one member of the admitted original source family. PA10--PA11 prove
convergence of its original continuous relative lift in real coordinates,
not only convergence modulo phase periods. Restoring its conserved common
mean gives `theta(sigma)->theta_infinity`, `S(theta_infinity)=0`, and uniform
limiting form. AI6 and storage monotonicity give

`U(theta_infinity)=H_infinity<E0`.

Suppose, for contradiction, that arbitrarily late times admit original-period
closed principal representatives. Choose `sigma_l->infinity` and integers `m_l`
such that

`h_l=g(theta(sigma_l))-2*pi*m_l in [-pi,pi]^20`,
`C h_l=2*pi*nu0`.

The convergent lifted gaps are bounded. Since the `h_l` are bounded too,
the integer vectors `m_l` take only finitely many values. Pass to a subsequence
with one fixed vector `m`. The identities imply `C m=0`, and passage to the
limit gives

`h_infinity=g(theta_infinity)-2*pi*m in [-pi,pi]^20`,
`C h_infinity=2*pi*nu0`.

The static-representative construction above now gives a same-energy equilibrium in the closed original
chamber, contradicting `U(theta_infinity)<E0` and the critical-energy premise.

Therefore **for each source member there exists a finite time after which
no closed principal representative has periods `(2,1,0)`**. The proof supplies
neither a numerical deadline nor one common deadline for the whole source
family. It does not select the eventual equilibrium or a common destination.

This is stronger than one finite principal-cut encounter: the original period
class cannot recur arbitrarily late, even through an original-compatible
boundary representative. It is weaker than eventual stabilization of all
principal labels. A limit can lie on cuts associated with other classes,
and the trajectory may cross those cuts infinitely often. Such crossings,
their orientation, their timing and the ultimately realized other periods
are not decided by this argument.
