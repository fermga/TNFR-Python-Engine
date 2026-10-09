# Effective memory of an acquired sine-class mediator

<a id="sine-class-mediated-memory"></a>

The [acquired-class response](SINE_CLASS_MEDIATED_RESPONSE.md#sine-class-mediated-result)
establishes a finite donor-induced receiver contrast on the three-C9 path.
This owner derives a causal donor-receiver description after hiding the
middle component. It retains that component's class and full initialization,
and bounds the nonlinear reduction error without resetting reached states.
The supplied law, capacities, interface and structural clock remain fixed.
No new response campaign or physical identification is implied.

The [component port kernel](SINE_REDUCED_CLASS_PORTS.md#the-retained-component-state-and-one-contact-normalization),
[linear coordinate-memory method](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state)
and [nonlinear environmental memory](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)
already establish their own elimination identities. The additional result
here concerns the actual two-contact acquired C9 mediator, its matched
nonlinear source histories and a finite error small enough to preserve the
certified class contrast. A generic convolution identity alone would not
meet that obligation. Task status remains in the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

<a id="sine-class-memory-state-and-law"></a>
## Full state and declared visible observation

Use the same ordered 27-node graph, unit capacities and class tuples
\((1,k,1)\), \(k=1,2\), as the response owner. Components are donor,
mediator and receiver, each with local nodes \(0,\ldots,8\); both
contacts join their central nodes 4. Actual port degrees are \((3,4,3)\).
With \(e=1023/1024\), \(\tau=et\), and
\(\gamma=1/(1023\pi)\), the full rows are
\[
x'=-A x+\gamma K S(\theta),\qquad \theta'=\gamma A x,
\qquad A=KL,
\tag{1}
\]
where \(K\) is inverse actual degree and \(L\) is the unit graph
Laplacian. The phases are circular; deviations use the continuous lifts
\(y=\theta-\Theta_k\) around the aligned class target. Form and phase
are in the same normalized structural units as the response owner, with
angles in radians and primes denoting the declared structural time.

The full deviation \(z=(x,y)\) has 54 coordinates. Retain all donor and
receiver forms and phase deviations as the 36-coordinate visible state
\(v\), and hide all 18 mediator coordinates as \(h\). Each block orders
forms before phases and retains the original local node order. This is a
coordinate partition, not a replacement of either visible component by
its mean or central port. Hidden coordinates remain parts of the complete
autonomous state.
Keeping all eighteen hidden coordinates is not a minimality claim. Eight
mediator reflection-odd tangent coordinates are invisible through its
central contacts, while nonlinear residuals can still couple to those
coordinates. Their contribution is retained in the full-state error bound.

Let \(C_k=KL_{c,k}\), where cycle phase Hessian weights are
\(\cos\alpha,\cos(k\alpha),\cos\alpha\), the two contact weights
are one, and \(\alpha=2\pi/9\). In the original unscaled coordinates,
\[
J_k=\begin{pmatrix}-A&-\gamma C_k\\\gamma A&0\end{pmatrix},
\qquad z'=J_kz+n_k(z),
\qquad
n_k(z)=\begin{pmatrix}\gamma K[S(\Theta_k+y)+L_{c,k}y]\\0\end{pmatrix}.
\tag{2}
\]
This decomposition is exact; the residual is not discarded. It obeys
\(\|n_k(z)\|_\infty\le2\gamma\|y\|_\infty^2\), from the
scalar sine Taylor remainder and actual degree normalization.

<a id="sine-class-memory-causal-law"></a>
## Exact tangent memory and the nonlinear residual

Partition (2) as
\[
v'=Ev+Bh+n_V(v,h),\qquad
h'=Cv+D_kh+n_H(v,h).
\tag{3}
\]
The matrices \(E,B,C\) are class independent. All class dependence in
the tangent generator is confined to \(D_k\), because only the
mediator's internal phase Hessian changes. No changed contact coefficient
or hidden clock is supplied.

Variation of constants gives the exact tangent law
\[
v'(t)=Ev(t)+Be^{D_kt}h(0)
 +\int_0^t\mathcal K_k(t-s)v(s)\,ds,
\qquad \mathcal K_k(t)=Be^{D_kt}C.
\tag{4}
\]
The full tangent initial state is still \((v(0),h(0))\). A class label
does not determine its hidden initial source. Supplying a visible history
externally would describe a driven subsystem, not the autonomous composed
solution without its feedback equation.

For the complete nonlinear law, the exact right-hand side of (4) also
contains
\[
n_V(v(t),h(t))+
 \int_0^t Be^{D_k(t-s)}n_H(v(s),h(s))\,ds.
\tag{5}
\]
Its hidden state satisfies the corresponding variation-of-constants row
from (3). Equations (3)--(5) therefore retain the nonlinear history and
initialization. Omitting (5) is a controlled tangent approximation only
after the error argument below; it is not an exact autonomous nonlinear
convolution with a fixed kernel.

The cross-port kernel exposes where the observed class information enters.
For receiver and donor central form selectors \(e_{R_x},e_{D_x}\),
\[
\mathcal K_1(0)=\mathcal K_2(0)=BC,\qquad
e_{R_x}^{\mathsf T}
 [\mathcal K_1'(0)-\mathcal K_2'(0)]e_{D_x}
 =\frac{\gamma^2[\cos\alpha-\cos(2\alpha)]}{24}.
\tag{6}
\]
This is the same actual-degree walk as in the response theorem:
\(1/4\) at mediator incidence, \(1/2\) for its internal central
diagonal and \(1/3\) at receiver incidence. The common cross entry at
zero lag is \((1-\gamma^2)/12\). Thus the class-dependent kernel
derivative gives the cubic paired contrast
\(a\gamma^2[\cos\alpha-\cos(2\alpha)]t^3/144\), while lower
orders are class blind. Neither this onset order nor causal memory defines
a positive waiting time before influence begins.

<a id="sine-class-memory-matched-source-error"></a>
## Finite nonlinear error with the reached initialization retained

Suppose an actual reached state has maximum-norm form and target-phase
errors at most \(\epsilon\). Continue it with and without the same
donor central form jump \(a\ge0\); both continuations retain their
complete visible and hidden initial states. This hypothesis is an endpoint
ball for the following bound. Proving that the original preparation
reaches it is the separate matched formation/handoff obligation, already
owned by the response theorem.
The implementation retains that owner's sufficient per-component
Euclidean norm budgets, which imply the maximum-norm premise used here.

For \(0\le t\le H\le1/4\), let \(g\) be an outward upper bound
for \(\gamma\), and put
\[
D_H=1-2g^2H^2>0,\qquad
Q_a=\frac{a+\epsilon+2gH\epsilon}{D_H}.
\tag{7}
\]
Diffusion contraction and the global sine Lipschitz bound imply
\[
\|x(t)\|_\infty\le Q_a,\qquad
\|y(t)\|_\infty\le\epsilon+2gQ_at.
\tag{8}
\]
The numerator retains both original channels; no phase uncertainty is
divided by \(\gamma\). Compare the actual full trajectory with the
tangent trajectory initialized at that same complete state. Their initial
difference is zero, \(\|J_k\|_\infty<3\), and integration of the
quadratic residual in (2) gives the whole-window error
\[
E_a=
\frac{2g\epsilon^2H+4g^2\epsilon Q_aH^2+
       (8/3)g^3Q_a^2H^3}{1-3H}.
\tag{9}
\]
Setting \(a=0\) gives \(Q_0,E_0\) for the actual unprobed baseline.

Within each class, tangent linearity cancels the same complete initial
state exactly in the probe-minus-baseline response, including its hidden
source term. That cancellation does not erase the initialization of either
individual trajectory or cancel the nonlinear residual. The error between
one actual paired receiver response and its tangent memory prediction is
at most \(E_a+E_0\). For both independently prepared classes and four
scalar recording errors of magnitude at most \(\delta\),
\[
\left|\widehat R_1(H)-\widehat R_2(H)-P(H)\right|
 \le B_T(H)+2(E_a+E_0)+4\delta,
\tag{10}
\]
where \(P\) and the correlated tangent tail \(B_T\) are the
[independently derived response coefficients](SINE_CLASS_MEDIATED_RESPONSE.md#sine-class-mediated-finite-response).
Equation (10) adds no spatial truncation: (4) is exact for the full tangent
system. It also adds no separate \(4\epsilon\) ideal-initialization
penalty, because both tangent continuations match their actual source.

At the previously frozen values \(a=10^{-3}\), \(H=10^{-4}\),
\(\epsilon=10^{-32}\), \(\delta=10^{-30}\), elementary bounds
give \(g<1/3000\), \(Q_a<101a/100\) and \(Q_0<2\epsilon\).
Substitution into (9), with \(1-3H=9997/10000\), yields
\[
2(E_a+E_0)<2.1\,10^{-28}<P/1000.
\]
The previous elementary trigonometric inequalities give
\(P>3\,10^{-25}\) and \(B_T/P<13/100\). Therefore (10) has
recorded lower bound greater than \(2.6\,10^{-25}\), whereas the
nonlinear stationary-comparator allowance below is less than
\(5\,10^{-30}\). The
[independent budget controls](../../tests/physics/test_sine_class_memory_algebra.py)
check these rational inequalities without generating a response.
This analytic transfer shows the memory description retains the already
certified finite effect. It supplies no new reserved assessment or
replacement of the archived outcome.

<a id="sine-class-memory-static-comparator"></a>
## A declared instantaneous comparator loses the class contrast

Freeze the visible tangent state and solve the hidden stationary rows.
Their unique solution is
\[
x_H=\frac{x_{D,4}+x_{R,4}}2\mathbf1_9,\qquad
y_H=\frac{y_{D,4}+y_{R,4}}2\mathbf1_9.
\tag{11}
\]
For form, the hidden Dirichlet Laplacian is the internal cycle Laplacian
plus two central contact terms. Its quadratic form is positive definite,
and the midpoint constant solves every row. The phase Dirichlet Laplacian
has the same contacts and a positive multiplier \(\cos(k\alpha)\)
on internal cycle edges, so the same argument proves its unique midpoint.
The hidden tangent generator is therefore invertible.

Consequently the Schur comparator \(E-BD_k^{-1}C\) is identical for
both classes. This comparison keeps the inherited visible mobility,
including degree three at its ports; it does not renormalize a graph after
deleting the mediator. Its within-class paired receiver response, and
hence its cross-class contrast, is class independent. It cannot reproduce
the positive contrast retained by (10).

Each comparator starts from the visible projection of the actual
postevent state. Its midpoint reconstruction generally replaces the hidden
state: even the ideal donor-only kick has actual hidden form zero while
(11) assigns the value \(a/2\). It therefore does not share the complete
initialization of the true memory law. The reduced visible mobility remains
the inherited one, with central denominator three rather than a newly
normalized degree \(5/2\). Moving the hidden midpoint can change the
reconstructed full degree-weighted charge; full-law storage, charge and
identity statements must not be transferred to this comparator.

The stationary hidden section is not asserted invariant when the visible
boundary moves. This comparator result concerns precisely (11), not every
possible instantaneous model or every finite-error approximation. Storage
minimization, a zero-frequency coefficient and a causal response law are
different constructions.

There is also an exact nonlinear stationary section on the corresponding
acute lift chart: retain each hidden reference twist and set its phase
deviation and form to the two visible central midpoints in (11). Internal
sine currents cancel, and the two central bridge gaps are opposite, so all
hidden nonlinear rates vanish. The grounded phase Hessian is positive
definite on this chart, making that stationary hidden phase unique there.
This is a local branch statement, not a global minimizer across windings.
The midpoint section generally moves when the visible state moves; its
instantaneously zero hidden rates do not track that motion.

Substituting this section into the visible nonlinear rows gives a second
declared instantaneous comparator. Its field is class independent, with
the same inherited visible mobility. Its paired response is identical
between classes only if the visible initial states also coincide.
For independently uncertain visible source states, its recorded class
contrast is instead contained in
\[
\left[-\frac{4\epsilon}{1-3H}-4\delta,\;
       \frac{4\epsilon}{1-3H}+4\delta\right].
\tag{12}
\]
This follows by comparing its four histories with the common ideal
visible initialization under the same Lipschitz bound three. For the
linear Schur comparator, exact within-class source cancellation gives
the sharper recorded null interval \([-4\delta,4\delta]\).
The bound in (10) separates the prior finite design from both nulls.
Neither comparator is installed into the engine or supplies a new
formation, work or identity proof.

<a id="sine-class-memory-reached-closure-obstruction"></a>
## An acquired-family obstruction to visible-only closure

An obstruction for arbitrary endpoint perturbations would not by itself
establish an obstruction on actual formation images. Here the original
source has a relative interior in the sixteen-dimensional plane of
zero-sum mediator form and phase residuals. The isolated complete sine
field is globally Lipschitz on continuous phase lifts and preserves both
component means. Its finite-time flow is a smooth invertible map of that
plane onto itself. Thus the image of any source interior is open in the
same relative plane, even after the long but finite formation/dwell time.
This argument asserts existence of a sufficiently small perturbation;
it supplies no numerically resolved radius for that open image.

Choose one reached interior state of either fixed mediator class. Perturb
only its hidden local form 3 by \(+s\) and local form 2 by \(-s\),
with sufficiently small \(s>0\), leaving all other forms and all phases
unchanged. Openness gives another actual admitted source whose unprobed
formation trajectory reaches this perturbed endpoint. The donor and
receiver histories before contact are unchanged. The hidden perturbation
has zero component sum; its two nodes both have degree two, so the joined
degree-weighted form charge is also unchanged. The phase charge and
mediator central form/phase are identical. Supply the same contact and
optional donor probe to both states, preserving these equalities.

All 36 visible coordinates and their initial first derivatives agree.
At the hidden central node, the changed form derivatives are
\[
\Delta x'_{M,4}=s/4,\qquad
\Delta\theta'_{M,4}=-\gamma s/4.
\tag{13}
\]
Consequently the receiver central accelerations differ by
\[
\Delta x''_{R,4}
 =\frac{s}{12}\left[1-\gamma^2
 \cos(\theta_{M,4}-\theta_{R,4})\right]>0,
\qquad
\Delta\theta''_{R,4}=-\frac{\gamma s}{12}.
\tag{14}
\]
The actual bridge phase enters this equality; it need not equal its ideal
target. Positivity follows from \(\gamma<1\). Contact and donor-probe
work are equal for this pair because every participating central value is
unchanged. The separate original source costs remain within their admitted
budgets and are not asserted equal.

No single regular autonomous first-order law of the visible coordinates,
the class label and conserved charges can reproduce both future visible
trajectories on this acquired family: the identical initial observations
would have one locally unique continuation, whereas (14) separates them.
Even retaining the mediator central pair does not close its evolution
without further hidden information, since that same pair has the different
derivatives (13). This is not an obstruction to memory, a larger retained
state, separately constrained preparations or a controlled finite-error
approximation.

<a id="sine-class-memory-admission-boundary"></a>
## Admission and inherited obligations

The conditional endpoint-ball bound supplies no formation, contact work,
probe work or identity certificate. Applying it to the acquired family
requires the same original source, finite formation and dwell, actual
handoff, complete law, support, capacity and event premises as the
[retained response owner](SINE_CLASS_MEDIATED_RESPONSE.md#sine-class-mediated-result).
Those proofs keep their own source costs and conserved-mean leaves. They
are not consequences of a kernel or of an available matrix decomposition.

Matrix coefficients involving \(\gamma\) and the class cosines must
retain their ideal algebraic definition and outward parameter enclosures.
An exact identity for a binary64-materialized rational generator certifies
that supplied generator, not automatically this transcendental one.
The shared coordinate-memory machinery can be reused within that declared
coefficient boundary; no fitted kernel or hidden initial state is supplied
by an export.

The [memory reader and bound calculator](../../src/tnfr/physics/relational_sine_class_memory.py)
separate those responsibilities. `derive_sine_class_mediated_memory`
rebuilds the fixed rational spatial blocks, named scalar factors, kernel
moments and stationary map without evaluating an exponential.
`bound_sine_class_mediated_memory` accepts only a probe amplitude, horizon,
conditional endpoint radius and scalar readout error. Its finite bounds
and any positive conditional contrast do not admit an actual preparation.
The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-class-mediated-memory)
and [guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-class-mediated-memory)
retain that distinction and the two comparator scopes.

The theorem derives an interaction description of organized nodal patterns
and preserves a class-dependent finite observable under explicit errors.
It does not select the substrate, law, supplied contact occurrence, physical
measurement map or a fundamental-particle interpretation. All earlier
frozen proofs, protocols and assessed responses remain unchanged.
