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
