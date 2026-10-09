# Constitutive freedom in the interaction of matched organized patterns

<a id="organized-pattern-constitutive-freedom"></a>

The same organized targets, target storage values, complete tangent and
quadratic dynamics can support opposite local nonlinear interactions.
This owner proves that statement for the existing positive-loss three-C9
support. It supplies an explicit analytic alternative to the sine potential;
it neither selects that alternative nor repeats a finite response calculation.
An own-law acquisition argument below also places the original isolated-C9
source families in their capture neighborhoods for every member of this
potential family, at the original common time and error budgets.

The [earlier mobility counterfamily](SINE_PAIR_MOBILITY.md#sine-replica-constitutive-nonselection)
and [smooth local continuation](SINE_CONSTITUTIVE_INFORMATION.md#local-phase-storage-nonselection)
already establish other nonselection results. Here mobility stays constant,
loss stays positive, and a target-matched analytic potential changes the
distinct-neighbor interaction of the present joined objects. The observation
and supplied interventions retain their original coordinates.

<a id="organized-pattern-complete-law"></a>
## Complete state, potential and reciprocal law

Keep all 27 nodes, the 27 internal cycle edges and the two bridges
`(4,13)` and `(13,22)`. All edges have unit weight. Node order is
`9*c+j`, with component index `c=0,1,2` and local index `j=0,...,8`.
Let \(L\) be the unnormalized graph Laplacian,

\[
 D=\operatorname{diag}(d_i),\quad A=D^{-1}L,\quad
 (d_4,d_{13},d_{22})=(3,4,3),\quad \sum_i d_i=58.
 \tag{1}
\]

The complete state is signed real form \(x\in\mathbb R^{27}\) and
27 primitive circular phases, represented by continuous lifts when taking
derivatives and means. Every capacity is held at one. Support, loss,
mobility and clock are held; no controller, support selector or capacity
evolution is inferred.

For \(0\le\lambda\le1\), define

\[
 \begin{split}
 W(\delta)&=\frac{(1-\cos9\delta)^2}{6\,9^4},\\
 V_\lambda(\delta)&=1-\cos\delta+\lambda W(\delta),\\
 j_\lambda(\delta)=V_\lambda'(\delta)
 &=\sin\delta+\frac{\lambda}{2187}
       \left(\sin9\delta-\frac12\sin18\delta\right).
 \end{split}\tag{2}
\]

The potential is even, nonnegative, real analytic and \(2\pi\)-periodic.
It vanishes at zero. Its current is odd and periodic. The parameter and
the ninth harmonic are declared constitutive choices, constructed to test
this support's target constraints; they are not inferred universal constants.

Set \(U_\lambda(\theta)=\sum_{\{i,j\}}V_\lambda(\theta_j-\theta_i)\),

\[
 q=Lx,\qquad
 S_{\lambda,i}=\sum_{j\sim i}j_\lambda(\theta_j-\theta_i),\qquad
 \nabla U_\lambda=-S_\lambda.
 \tag{3}
\]

Use the same structural normalization

\[
 e=\frac{1023}{1024},\quad w=\frac1{1024},\quad
 \tau=e t,\quad \gamma=\frac{w}{e\pi}=\frac1{1023\pi}.
\]

In the original clock and the structural clock respectively, the complete
rows are

\[
 \begin{array}{ll}
 \displaystyle\frac{dx}{dt}=-eD^{-1}q+\frac w\pi D^{-1}S_\lambda,
 &\displaystyle\frac{d\theta}{dt}=\frac w\pi D^{-1}q,\\[3pt]
 x'=-Ax+\gamma D^{-1}S_\lambda,
 &\theta'=\gamma Ax.
 \end{array}\tag{4}
\]

Thus the pressure for the held unit-capacity nodal row is specified by
the first original-time expression, independently of the evaluated response.
Both evolution rows transform under the clock change. All coefficients use
the adopted structural units: \([p]=[x]/([\nu_f][t])\). No laboratory time,
energy unit, inverse-mass interpretation or physical identity is supplied.

For

\[
 H_\lambda=\frac12x^TLx+U_\lambda(\theta),
\]

the constant reciprocal exchange block is

\[
 \begin{pmatrix}x'\\\theta'\end{pmatrix}
 =\left[
 \begin{pmatrix}0&-\gamma D^{-1}\\\gamma D^{-1}&0\end{pmatrix}
 -\begin{pmatrix}D^{-1}&0\\0&0\end{pmatrix}
 \right]\nabla H_\lambda.
 \tag{5}
\]

The skew part cancels between the two rows. Consequently, on the actual
full state,

\[
 H_\lambda'=-(Lx)^TD^{-1}(Lx)\le0,\qquad
 \frac{dH_\lambda}{dt}=-e(Lx)^TD^{-1}(Lx).
 \tag{6}
\]

The same positive loss law is retained, not merely its value on a target.
Odd edge currents and Laplacian row sums also give

\[
 (\mathbf1^TDx)'=0,\qquad (\mathbf1^TD\theta)'=0.
 \tag{7}
\]

These are lifted means in the declared coordinates, not circular-resultant
angles. Adding the same constant to all forms or to all phases preserves
the dynamics and storage differences.

## Global admission and constitutive derivative bounds

For every real gap and integer \(r\ge0\), differentiating (2) gives

\[
 |j_\lambda^{(r)}(\delta)|
 \le1+\frac{\lambda}{2187}
              \left(9^r+\frac{18^r}{2}\right).
 \tag{8}
\]

In particular, \(|j_\lambda'|\le1+2\lambda/243\). Thus the complete
real-lift field (4) is globally Lipschitz: its form-row difference is
bounded by \(2\|\Delta x\|_\infty+
2\gamma(1+2\lambda/243)\|\Delta\theta\|_\infty\), and its phase-row
difference by \(2\gamma\|\Delta x\|_\infty\). It has a unique global
solution from every finite full state, in either time direction, between
the explicitly supplied events. This does not assert global convergence
to one winding target.

Equation (8) supplies real derivative bounds for this law. Although (2)
also extends analytically to complex phases, its ninth and eighteenth
harmonics have different complex growth from sine. The earlier sine
Cauchy tails, source bounds, finite-time intervals and error budgets do
not transfer without a new estimate.

<a id="organized-pattern-target-matching"></a>
## Target storage and full local dynamics agree

Let \(\alpha=2\pi/9\), choose mediator class \(k\in\{1,2\}\), and use
the nominal joined target

\[
 x=0,\qquad \Theta_{k,9c+j}=k_c\alpha(j-4),\qquad
 (k_0,k_1,k_2)=(1,k,1).
 \tag{9}
\]

Each cycle's principal edge gaps are \(\pm k_c\alpha\); the bridges
have gap zero. The closing-edge lifts differ by known full turns, which
do not change the periodic current or its derivatives. Direct differentiation
gives

\[
 \begin{split}
 W''(\delta)&=\frac{\cos9\delta-\cos18\delta}{243},\\
 W'''(\delta)&=\frac{-\sin9\delta+2\sin18\delta}{27},\\
 W''''(\delta)&=\frac{-\cos9\delta+4\cos18\delta}{3}.
 \end{split}\tag{10}
\]

At every \(\delta=2\pi m/9\), \(m\in\mathbb Z\),

\[
 W=W'=W''=W'''=0,\qquad W''''=1.
 \tag{11}
\]

Therefore both classes have the same equilibrium, target storage value,
gradient and full phase Hessian for every \(\lambda\). In particular,

\[
 H_{*,k}=18(1-\cos\alpha)+9(1-\cos k\alpha),\qquad
 L_{c,k}=\nabla^2U_\lambda(\Theta_k)
\]

are independent of \(\lambda\). Opposite internal currents cancel at every
cycle node, and both bridge currents vanish. The full 54-coordinate tangent is

\[
 J_k=\begin{pmatrix}-A&-\gamma D^{-1}L_{c,k}\\
                         \gamma A&0\end{pmatrix}.
 \tag{12}
\]

The entire quadratic field is also identical: it consumes
\(j_\lambda''=V_\lambda'''\), whose target values do not change.
The cubic field consumes \(j_\lambda'''=V_\lambda''''\), and does change.
No hidden quadratic mode is removed by this matching statement.

<a id="organized-pattern-local-identity"></a>
## A locally re-admitted identity and exact event work

Let \(y=\theta-\Theta_k\), \(P_0=I-\mathbf1\mathbf1^T/27\), and
\(Z^2=\|P_0x\|_2^2+\|P_0y\|_2^2\). This is a quotient radius,
not an operation that resets the state or its weighted means.
Within \(Z\le r=1/12\), every edge residual has magnitude at most
\(\sqrt2r\). The graph diameter is ten: four cycle edges, the two
bridges, then four cycle edges suffice between any pair. For any real
vector \(v\), its range and the variance bound give
\(\|P_0v\|_2^2\le27\operatorname{range}(v)^2/4
\le(135/2)v^TLv\), by Cauchy-Schwarz along a shortest path joining
its extreme values. Thus \(\lambda_2(L)\ge2/135\), independently of
the phase law. The [existing angular bound](SINE_CLASS_MEDIATED_RESPONSE.md#sine-class-mediated-consistency)
is \(\cos(4\pi/9+\sqrt2r)>1/20\). It also follows from
\(\pi>31/10\), \(\sqrt2<17/12\) and
\(\sin(13/240)>13/240-(13/240)^3/6>1/20\).

Writing \(c=\cos9\delta\), equation (10) gives
\(W''=(1+c-2c^2)/243\ge-2/243\). Hence throughout this chart,

\[
 V_\lambda''>\frac1{20}-\frac2{243}
                 =\frac{203}{4860}>\frac1{25}.
 \tag{13}
\]

Integrating the Hessian along the segment from the target yields

\[
 H_\lambda-H_{*,k}
 \ge\frac12x^TLx+\frac1{50}y^TLy
 \ge\frac{Z^2}{3375}.
 \tag{14}
\]

An initial state inside the chart with relative storage below
\(r^2/3375=1/486000\) cannot reach its boundary under unforced flow,
by (6) and (14). The target is the unique phase critical orbit there.
On a fixed weighted-mean leaf the trapped sublevel is compact.
The largest invariant subset with zero loss has \(Lx=0\); invariance
then requires \(D^{-1}\nabla U_\lambda\) to be common across nodes.
Its degree-weighted sum is zero, so that gradient vanishes. Thus LaSalle's
argument gives local convergence to the corresponding target orbit.
No common form or phase is forced back to zero.

For any prescribed form jump \(f\), phases do not jump, and each law has
the exact work

\[
 \Delta H_\lambda=f^TLx^-+\frac12f^TLf.
 \tag{15}
\]

For simultaneous donor and receiver jumps \(f=ae_4+be_{22}\),
\(L_{4,22}=0\), so (15) is
\(a(Lx^-)_4+b(Lx^-)_{22}+3(a^2+b^2)/2\).
It is the same functional in each model, but later pre-event states need
not coincide. The weighted form mean changes by \(3(a+b)/58\), with
the phase mean unchanged.

From the prescribed nominal source (9), the existing amplitudes
\(a=b=m=7/10000\) give work \(3m^2<1/486000\) and
\(Z_+^2=50m^2/27<r^2\). The single-input and zero-input branches also
satisfy these bounds. All four branches consequently retain the joined
quotient identity for both laws, without borrowing an acquisition theorem.
Any support change needs its own edge-storage work; continuous loss is
not an undeclared event reserve.

<a id="organized-pattern-mixed-onset"></a>
## A changed interaction of the same prescribed organized states

Use the same complete nominal pre-input state (9) in four histories with
form jumps \(0,ae_4,be_{22},ae_4+be_{22}\). Observe the mediator form and
define, as in the [distinct-neighbor owner](SINE_CLASS_NEIGHBOR_NONADDITIVITY.md#sine-neighbor-local-onset),

\[
 M_{\lambda,k}(a,b,T)=x_{13}^{11}(T)-x_{13}^{10}(T)
                          -x_{13}^{01}(T)+x_{13}^{00}(T).
 \tag{16}
\]

Immediately after the joint jump, the phase slopes are
\(\theta'_4=\gamma a\), \(\theta'_{22}=\gamma b\),
\(\theta'_{13}=-\gamma(a+b)/4\), and
\(\theta'_{12}=\theta'_{14}=0\). Thus the four incident gap slopes
into node 13 are \(\gamma(5a+b)/4\), \(\gamma(a+5b)/4\), and twice
\(\gamma(a+b)/4\). The first two edges are bridges; the other two
are internal mediator edges.

The form derivatives through order two are linear in the impulses.
The third derivative is at most quadratic. In the fourth derivative,
the only cubic contribution is the current's third derivative applied
to three initial gap slopes; every other term is at most quadratic.
Port-fixing ring reflection with simultaneous form/phase-deviation sign
reversal is a symmetry of every even potential (2). At the nominal source,
the central output is odd in \((a,b)\), so its quadratic contributions
vanish. Hidden quadratic modes generally remain nonzero.

The mixed bridge cubes sum to \(90ab(a+b)\), and the two mixed internal
cubes sum to \(6ab(a+b)\). Including the degree-four normalization,
gap denominators and \(4!\) gives the complete nonlinear local onset

\[
 \begin{split}
 M_{\lambda,k}(a,b,T)
 &=\frac{\gamma^4ab(a+b)}{1024}
       [15j_\lambda'''(0)+j_\lambda'''(k\alpha)]T^4+O(T^5)\\
 &=\boxed{\frac{16\lambda-15-\cos(k\alpha)}{1024}
                   \gamma^4ab(a+b)T^4+O(T^5)}.
 \end{split}\tag{17}
\]

For fixed positive \(a,b\), the sine member \(\lambda=0\) has negative
onset, whereas \(\lambda=1\) has positive onset, since
\(1-\cos(k\alpha)>0\) for both classes. Analyticity supplies a positive
local time neighborhood in which the corresponding exact signs hold;
(17) supplies no declared numerical horizon or measurement resolution.

The leading coefficient vanishes at
\(\lambda_{\mathrm{crit},k}=(15+\cos k\alpha)/16\).
Since \(\cos2\alpha<\cos\alpha\),
\(\lambda_{\mathrm{crit},2}<\lambda_{\mathrm{crit},1}\).
Between these two values the classes have opposite local orientations
under one law. For example, the fixed rational \(\lambda=31/32\)
gives the factor \(1/2-\cos(k\alpha)\): it is negative for class one
and positive for class two because \(\alpha<\pi/3<2\alpha\).
At \(\lambda=1\), the bridge contribution vanishes and the entire
positive onset comes from the internal mediator edges. Omitting those
edges would therefore give a false null. Within this particular family,
the normalized class-one-minus-class-two coefficient remains
\((\cos2\alpha-\cos\alpha)/1024\); that robustness is not asserted
for every matched-target potential. At a critical value only the order-four
coefficient is known to vanish; exact additivity does not follow. Zero input in either
channel cancels (16) for every common source. At the nominal source,
\(a=-b\) is also an exact null by outer-ring exchange and the sign symmetry.

<a id="organized-pattern-same-source-acquisition"></a>
## Own-law acquisition from the original isolated C9 sources

This section changes support explicitly to one isolated unit C9, with nodes
\(0,\ldots,8\), edges \(j--(j+1\bmod9)\), degree matrix \(D=2I\),
capacities one and the same \(e,w,\gamma\) and clocks as (4). It does not
apply the joined graph's spectral bound to an isolated component or identify
the two supports' reached states. The original
[two-class preparation](SINE_PATTERN_DYNAMICS.md#sine-formed-class-pair)
supplies the following unchanged experiment for each law:

- Classes \(k=1,2\), rational slope
  \(m=(2046/9)(355/113)^2\), initial form
  \(x_j(0)=km(j-4)+\xi_j^x\), and initial phase lifts
  \(\theta_j(0)=\xi_j^\theta\).
- Bounds \(|\xi_j^x|\le\rho_x=10^{-10}\) and
  \(|\xi_j^\theta|\le\rho_\theta=10^{-10}\), with separate exact zero sums
  \(\sum_j\xi_j^x=\sum_j\xi_j^\theta=0\). Each class has a complete
  sixteen-dimensional relative source neighborhood, not an independently
  selectable eighteen-dimensional ambient box.
- Acquisition time \(T=100\) in \(\tau\), original time \(102400/1023\),
  radius \(r=1/12\), and common whole-source storage budget \(10^9\).
  No input, support event, state reset or coefficient change occurs.
- Target \(x^*=0\), \(\theta_{k,j}^*=k\alpha(j-4)\), with
  \(\alpha=2\pi/9\). Targets label the endpoint comparison, not replacement
  endpoint states.

Write \(A=L/2\), \(z=\gamma x\), \(\eta=\gamma^2\), and
\(f_\lambda=S_\lambda/2\). The complete scaled law is
\[
 z'=-Az+\eta f_\lambda(\theta),\qquad
 \theta'=Az,\qquad
 (\theta+z)'=\eta f_\lambda(\theta).
 \tag{18}
\]
Odd currents preserve both zero means. The full real-lift field is globally
Lipschitz by (8), including nonacute states during acquisition. Continuous
lifts are therefore retained through any winding changes.

For \(P_9=I-\mathbf1\mathbf1^T/9\) and the norm
\(\|u\|_{2I}=\sqrt2\|u\|_2\), the C9 graph gives the same
law-independent semigroup gap \(g=1/5\):
\(\|e^{-A\tau}P_9\|_{2I}\le e^{-g\tau}\).
Indeed \(1-\cos(2\pi/9)>1/5\), so
\(\lambda_2(L)>2/5\), while \(\lambda_{\max}(L)\le4\).
The forcing and Hessian constants must instead be re-admitted:
\[
 \begin{aligned}
 J_\lambda&=1+\lambda/1458,&
 \|f_\lambda(\theta)\|_{2I}&\le F_\lambda=\sqrt{18}\,J_\lambda,\\
 u_\lambda&=1+2\lambda/243,&
 \nabla^2U_\lambda(\theta)&\preceq u_\lambda L .
 \end{aligned}\tag{19}
\]
The first bound uses
\(|\sin9\delta-\tfrac12\sin18\delta|\le3/2\); the second uses (10)
and \(\cos\delta\le1\). They hold globally, not only near a target.
The raw initial metric norms remain \(\sqrt{18}\rho_\theta\) and
\(\sqrt{18}\gamma\rho_x\); multiplying those errors by the enlarged
forcing constant would confuse distinct premises.

The nominal phase storage initially vanishes. Since
\(1-\cos(9v)\le81v^2/2\), the new potential term obeys
\(0\le W(v)\le v^4/24\). The entire source family therefore has
\[
 B_{\lambda,k}
 =36k^2m^2+32km\rho_x+18(\rho_x^2+\rho_\theta^2)
                  +6\lambda\rho_\theta^4
 \quad\hbox{as a storage upper bound}.                 \tag{20}
\]
The linear form term uses the sum \(16km\) of the absolute nominal
edge differences; the last term uses all nine edges and
\(|\Delta\xi^\theta|\le2\rho_\theta\).
The exact nominal costs \(36k^2m^2\) are unchanged and differ by a factor
four. The quartic source correction is retained even though at these
budgets it is below one dyadic128 grid spacing. No equality of later
losses or phase histories follows.

Let \(\rho\) be an outward upper bound for \(e^{-g T}\). Define
\[
 \begin{aligned}
 V_k&=\sqrt{120}\,k\gamma m,&
 E_x&=\sqrt{18}\gamma\rho_x,& E_\theta&=\sqrt{18}\rho_\theta,\\
 R_{z,\lambda,k}&=\rho(V_k+E_x)+\eta F_\lambda/g,&
 R_{x,\lambda,k}&=R_{z,\lambda,k}/(\gamma\sqrt2),\\
 D_k&=\sqrt{60}\,k|\gamma m-\alpha|,&
 Y_{\lambda,k}&=D_k+
       (E_x+E_\theta+R_{z,\lambda,k}+\eta F_\lambda T)/\sqrt2,\\
 Z_{\lambda,k}^2&=R_{x,\lambda,k}^2+Y_{\lambda,k}^2 .
 \end{aligned}\tag{21}
\]
Variation of constants bounds \(\|z(T)\|_{2I}\) by \(R_z\).
Integrating the last row of (18) gives
\[
 \theta(T)-\gamma x^{\rm nom}(0)
 =\xi^\theta+\gamma\xi^x-z(T)
                 +\eta\int_0^T f_\lambda(\theta(s))\,ds .
\]
Consequently every actual endpoint satisfies
\(\|x(T)\|_2\le R_x\),
\(\|\theta(T)-\theta_k^*\|_2\le Y\) and the full radius bound \(Z^2\).
This uses the entire nonlinear forcing and all initial residuals; it is
not a tangent trajectory or a Cartesian reset of the reached family.

Target stationarity and the global Hessian upper bound yield the split
endpoint estimate
\[
 H_\lambda(T)-H_k^*
 \le E_{\lambda,k}:=
       2R_{x,\lambda,k}^2+2u_\lambda Y_{\lambda,k}^2,\qquad
 H_k^*=9(1-\cos k\alpha).
 \tag{22}
\]
The form and phase terms retain their different coefficients.
For the common capture ball set
\[
 \mu=\pi/2-4\pi/9-\sqrt2r,\quad
 c_r=\cos(4\pi/9+\sqrt2r),\quad
 c_\lambda=c_r-2\lambda/243,\quad
 b_\lambda=c_\lambda r^2/5 .
 \tag{23}
\]
When \(\mu>0\) and \(c_\lambda>0\), the own-law phase Hessian throughout
the ball is bounded below by \(c_\lambda L\). With the zero means and
the isolated graph gap, this gives
\(H_\lambda-H_k^*\ge(c_\lambda/5)
(\|x\|_2^2+\|\theta-\theta_k^*\|_2^2)\).
Thus \(Z_{\lambda,k}^2<r^2\) and \(E_{\lambda,k}<b_\lambda\) prove
forward trapping under the own-law identity
\(H_\lambda'=-\|Lx\|_2^2/2\).

### Uniform fixed-budget result and its provenance

The shared outward dyadic128 square-root, Machin-pi, cosine and
negative-exponential arithmetic applied to (19)--(23) at \(\lambda=1\)
gives the following strict rational bounds for both classes:
\[
 \begin{gathered}
 \mu>56/1000,\qquad c_1>48/1000,\qquad
 Z_{1,k}^2<23/10^6<r^2,\\
 E_{1,k}<45/10^6,\qquad b_1>672/10^7,\qquad
 b_1-E_{1,k}>222/10^7,\\
 B_{1,1}<181230000,\qquad B_{1,2}<724918000<10^9 .
 \end{gathered}\tag{24}
\]
Here the subscript one on \(c_1,b_1\) means deformation one, not winding
class one. The bounds are rounded outward from direct scalar interval
evaluation of the displayed formulas, not from a sampled trajectory or
the earlier sine report. In particular the computed storage-margin
lower bounds exceed \(2.29504274\,10^{-5}\) for winding one and
\(2.22733831\,10^{-5}\) for winding two.

Only the endpoint \(\lambda=1\) is needed: \(F_\lambda,u_\lambda\),
the source-cost bound and the endpoint bounds increase with \(\lambda\),
whereas \(c_\lambda,b_\lambda\) decrease. Hence (24) proves acquisition
and capture uniformly for all \(0\le\lambda\le1\), including both laws
used in (17). This is an analytic family argument, not a parameter sweep.

Every initial phase edge has its small principal lift, so winding starts
at zero. Endpoint membership in the target's acute ball fixes winding
\(k\); it remains there thereafter. The change requires an intervening
antipodal passage, allowed by (18); an acute constraint during the whole
acquisition would be a different model. On the compact trapped zero-mean
leaf, zero loss forces \(x=0\), and invariance forces \(S_\lambda=0\).
Strict phase convexity leaves only \(\theta_k^*\). LaSalle therefore
gives convergence; the unchanged positive target Hessian also retains
local exponential stability of the complete form/phase tangent.
For each positive eigenvalue \(\ell\) of \(L/2\), its characteristic
polynomial is \(s^2+\ell s+\gamma^2\cos(k\alpha)\ell^2\), whose
roots have negative real parts.
Winding magnitudes one and two remain inequivalent under cycle
automorphisms, common origins and joint sign reversal.

This establishes acquisition from the same prescribed source family,
with the same time, uncertainty and cost limits. It does not identify
the reached fine states between laws, reproduce later very small
residual budgets, or transfer a dwell, support-contact, joined-source
or finite-response certificate. Those require the new law's own bounds.

<a id="organized-pattern-information-boundary"></a>
## What this comparison constrains, and what it leaves open

The prescribed source in (17) is an exact organized target. The acquisition
section proves own-law capture from the original isolated source families;
it does not replace that nominal source by a reached fine state or transfer
the onset sign to its residuals. Arbitrary residuals need not have the nominal
reflection symmetry. No sign at the earlier \(T=1/8\), finite noise
discrimination, later dwell or joined-source transfer is asserted here.

The interaction differs in the held structural chart with the same form
impulses and mediator readout. A common clock rescaling cannot account
for it while leaving the nonzero tangent (12) fixed. This does not exclude
every nonlinear conjugacy accompanied by changed inputs and observations.

The supplied additive current consumes first, ninth and eighteenth phase
harmonics. Its extra information is outside a law depending only on degree
and the first neighbor phasor. The
[existing first-moment sufficiency theorem](SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
can constrain that freedom only if its information restriction is
independently justified. Restating that restriction, imposing cosine
storage globally, or assuming the desired current is not a new derivation.

These complete models retain signed form, primitive circular phase,
whole-state memory, common-origin covariance, held capacities and the
declared identity. Additive edge currents, fixed support, constant
reciprocal mobility and the pressure completion remain supplied hypotheses.
No registered operator's descent, autonomous event occurrence, inherited
scale law, composite binding energy or fundamental constituent follows.
The [foundation compatibility audit](../DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#justified-law-admission)
owns those distinctions; the execution plan owns subsequent work.

## Implementation and independent controls

[`_sine_class_constitutive_comparison.py`](../../src/tnfr/physics/_sine_class_constitutive_comparison.py)
provides detached exact/static descriptions of this family, target jets,
local interaction coefficients and whole-support balance observations.
These static descriptions allow every \(\lambda\ge0\); the uniform
local trapping proof above is restricted to \(0\le\lambda\le1\).
Its [independent controls](../../tests/physics/test_sine_class_constitutive_comparison.py)
rebuild target derivatives and mixed edge weights on all 27 nodes, check
both evolution rows and event/loss balances, and retain invalid-domain
and symmetry controls. This does not install an alternative live law or
reuse a sine response as evidence for it. No trajectory, selected
time-coefficient producer or frozen scientific response is regenerated.

The separate
[_sine_class_constitutive_acquisition.py](../../src/tnfr/physics/_sine_class_constitutive_acquisition.py)
rebuilds the isolated full-family transit, law-specific storage bounds and
capture margins from primitive inputs, using shared semigroup arithmetic
without treating the sine preparation's storage as the alternative law's.
Its [controls](../../tests/physics/test_sine_class_constitutive_acquisition.py)
check the forcing versus initial-error norms, quartic source correction,
split endpoint storage and strict capture margins. The uniform conclusion
uses the proved monotonic dependence and the endpoint \(\lambda=1\);
it does not require a sweep or a trajectory producer.
