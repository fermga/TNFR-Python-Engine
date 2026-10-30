# Finite preparation families for the held normalized-sine law

This chapter derives sufficient nonlinear error bounds for two preparations
under the same complete held law. It connects the established
[connected/cut point comparison](SINE_MAINTAINED_PARENT_CLOSURE.md#post-sha-cut-supervised-response)
to an explicitly specified computational preparation family. The single
registered global comparison is insufficient to certify that family; its
actual robustness remains unresolved. The
[research program](../RESEARCH_PROGRAM.md#deferred-composition-question)
owns the current task and execution admission.

<a id="adjacent-preparation-family"></a>
## 1. Preparation origin and the finite family

A saved binary64 coordinate denotes its exact represented value. Hexadecimal
serialization preserves that value; it does not establish an unknown real
predecessor, a measurement error distribution or a preparation tolerance.
A neighborhood must therefore enter as an additional, explicit policy.

For every represented form and registered real-phase coordinate f_j, let
f_j^- and f_j^+ be its immediately preceding and following finite binary64
numbers. Canonicalize signed zero and require both neighbors to be finite.
Define the adjacent-representable family and its closed real hull by

$$
\mathcal F_{\rm adj}=\prod_{j=1}^{42}\{f_j^-,f_j,f_j^+\},
\qquad
\mathcal H=\prod_{j=1}^{42}[f_j^-,f_j^+].
$$

The family has exactly `3^42` distinct supplied initial vectors. Its full
coordinate spacings are fixed by the representation before evaluating the
response. This asks whether one representable adjacency can change the
prediction. It supplies a computational-resolution robustness class, not a
physical uncertainty model or a macroscopic preparation tolerance.

The larger continuous hull is a sufficient proof domain. Its nonrepresentable
interior points are mathematical preparations, not additional stored binary64
vectors. Exact real initial errors must remain present in the argument;
replacing the hull by the nominal point would erase them.
The full adjacent hull differs from the narrower nearest-even rounding cell.
Neither replacing it by that cell nor selecting a smaller radius after a
failed bound answers this fixed question.

All 21 forms and 21 registered phase lifts vary independently. The identical
family member initializes both complete branches. Hold capacities,
conductances, both support choices, their actual degrees, the original clock,
elapsed `[0,1/2]` and positive marked observation scale w fixed. Do not wrap
phase endpoints or independently re-register connected components. Each
member has its own conserved means under each branch's weights; no nominal
mean constraint is imposed. In the cut branch the child's coordinates remain
constant at that member's initial values, not at the nominal child's values.

Fresh source admission re-admits the nominal center and its association with
the retained records under existing evidence premises. It does not show that
every perturbed member is produced by the same event sequence. These are
supplied post-event states; uniform event admission, acquisition, formation and autonomous identity
remain separate questions. Earlier hypothetical Coupling (UM) error budgets
do not set these widths.

For any coordinate, retain the exact asymmetric errors

$$
e_j^- = f_j-f_j^-,\qquad e_j^+=f_j^+-f_j,
\qquad r_j=\max(e_j^-,e_j^+).
$$

At binade boundaries these spacings need not agree. Zero has the two distinct
neighbors of smallest positive and negative subnormal magnitude. They are
valid nonzero represented coordinates; this is not the rejected admission
of a nonzero real that underflows to zero. Finite extreme values lacking two
finite neighbors are outside this declared family construction.

<a id="sine-global-preparation-comparison"></a>
## 2. A global nonlinear two-block comparison

On a dynamic component with positive bare degrees, the complete law is

$$
\dot x_i=\frac{\nu_i}{\pi d_i}\sum_{j\sim i}\sin(\theta_j-\theta_i),
\qquad
\dot\theta_i=\frac{\nu_i}{\pi d_i}\sum_{j\sim i}W_{ij}(x_i-x_j).
$$

All coefficients are held, capacities are nonnegative, conductances are
nonnegative, and the real lifts have fixed registration. The vector field
is globally Lipschitz in these finite coordinates, so solutions exist
uniquely on the whole finite window. Isolated constant rows are specified
separately; no zero-degree division supplies their law.

Compare any nominal and perturbed solutions of this same law. Set
u(t)=max_i|tilde x_i-x_i| and v(t)=max_i|tilde theta_i-theta_i| over the
dynamic component, and define

$$
a=\frac{2}{\pi}\max_i\nu_i,\qquad
b=\frac{2}{\pi}\max_i\frac{\nu_i\sum_{j\sim i}W_{ij}}{d_i},
\qquad M=\begin{pmatrix}0&a\\b&0\end{pmatrix}.
$$

The global inequality |sin(alpha+eta)-sin(alpha)| <= |eta| gives
|delta x_dot_i| <= 2 nu_i v/pi. The exactly linear phase row gives
|delta theta_dot_i| <= 2 nu_i (sum_j W_ij) u/(pi d_i). Taking upper Dini
derivatives of the finite maxima therefore yields

$$
D^+u\leq av,\qquad D^+v\leq bu,\qquad
\binom{u(t)}{v(t)}\leq e^{tM}\binom{u_0}{v_0}.
$$

Here u_0 and v_0 are the maxima of the exact coordinate radii in their
respective blocks. The last inequality follows from the cooperative linear
comparison theorem, or by integrating the two inequalities and iterating
their nonnegative kernel. It applies uniformly to every member of the hull.
It is a nonlinear difference bound: no small-angle substitution, tangent
trajectory or assumed closeness domain enters its proof.

Both comparison coordinates are nondecreasing because M and the initial
allowances are nonnegative. Thus any upper endpoint pair (U,V) for the
comparison at T also bounds the errors at every earlier time. This monotonic
majorant supplies whole-window control; endpoint proximity of the actual
trajectories alone would not do so.

Use all connected nodes for that branch. For the cut branch use the complete
20-node host, with cut degrees and retained host conductances. Keep separate
constant child allowances c_x and c_theta from the original family. They do
not enter the isolated host's evolution, but they enter the marked reading.
The two block maxima lose correlations and cannot exploit cancellation of
fast oscillations; their conservatism is a property of this sufficient bound.

<a id="sine-family-marked-gates"></a>
## 3. Whole-window portrait and signed-angle gates

In the original oriented plane use
y=(theta_child-theta_parent, sqrt(w)*(x_child-x_parent)). If (U_c,V_c) and
(U_0,V_0) are the connected and cut-host comparison endpoints, respectively,
then uniform squared portrait errors are bounded by

$$
E_c^2=4(V_c^2+wU_c^2),\qquad
E_0^2=(V_0+c_\theta)^2+w(U_0+c_x)^2.
$$

These combine the two coordinate blocks only through the declared observation
metric; they do not identify physical units. In particular, freezing the cut
child does not remove its preparation error.

Let q_c,q_0 be positive lower bounds on the nominal squared marked radii
over the entire original window. The sufficient admission is

$$
E_c^2<q_c,\qquad E_0^2<q_0.
$$

Every perturbed portrait then stays nonzero, with radius at least
sqrt(q_b)-E_b > 0. The straight homotopy from each nominal portrait stays
nonzero as well. The
[common-preparation identity](SINE_MAINTAINED_PARENT_CLOSURE.md#post-sha-shared-preparation-error)
cancels the two initial angular errors and bounds the error in the difference
of clockwise increments by

$$
\arcsin(E_c/\sqrt{q_c})+\arcsin(E_0/\sqrt{q_0})<\pi.
$$

Consequently a nominal directional gap strictly larger than pi preserves its
ordering for the whole family whenever both squared-radius gates pass.
The retained point gap has that property. A consumer must still rebuild and
admit that gap and both radius minima from associated evidence. Checking
`gap_lower > pi_upper` permits this sufficient test without adding another
inverse-trigonometric numerical kernel. It proves a strict ordering, not the
sharpest perturbation margin or constant local rotation direction.

Failure of either sufficient inequality leaves robustness unresolved. It does
not produce a counterexample, an actual radius crossing or loss of identity.
Arithmetic evidence for a bound and evidence that its premises match the
intended source remain distinct obligations.

<a id="sine-preparation-method-preflight"></a>
## 4. Numerical method, alternatives and evidence boundary

The pure [preparation arithmetic owner](../../src/tnfr/research/sine_preparation_bounds.py)
uses the existing binary64 neighbor geometry and exact scalar admission.
It retains asymmetric widths as rational values before invoking the shared
[comparison-flow kernel](../../src/tnfr/mathematics/_comparison_flow.py).
Constructed [controls](../../tests/research/test_sine_preparation_bounds.py)
exercise representation boundaries, exact limiting flows, comparison bounds
and the isolated child's observation contribution. These helpers establish
conditional arithmetic, not a historical source or a family verdict.

For each branch, rational upper coefficients a_bar,b_bar give a nonnegative
comparison matrix. The shared kernel encloses its exponential action using
a series through degree 32, a norm-tail bound and outward dyadic128 arithmetic.
A fixed 64-step composition with h=1/128 covers T=1/2 and uses a two-dimensional
matrix,
within the existing dimension policy. Monotonicity permits a rounded upper
endpoint to initialize the next comparison step. Its applicability condition
is h max(a_bar,b_bar) <= 1; increasing an exact coefficient to its outward
upper endpoint preserves the differential comparison.

The static preflight reconstructed capacities, weight sums and actual degrees
from the already known, receipt-matched nominal result. Both branches satisfy
that kernel condition on this fixed grid. This entails 128 comparison calls,
without sampling the `3^42` preparations or integrating another trajectory.
No family widths, propagated errors or sufficient-gate verdict were evaluated
by that coefficient check. It was not a fresh admission of the source history.

The prior coefficient scales make substantial amplification possible in this
majorant. Cheap arithmetic therefore does not promise an informative bound.
The main alternatives have different additional premises:

| Method | Required extra information and limitation |
| --- | --- |
| One scalar global Lipschitz bound | Simpler, but loses the distinct form/phase rates and cannot improve this block comparison without a justified norm change. |
| Global two-block comparison above | No trajectory-dependent domain or Taylor remainder; may overestimate growth by discarding oscillatory cancellation and graph structure. |
| Graph-resolved component comparison | Retains neighbor structure; requires an admitted dimension/work policy or a separately justified block reduction. It must not silently bypass the shared 24-coordinate policy with a 42-coordinate matrix. |
| Signed Jacobian or energy bounds on retained cells | Can retain useful cancellation only with a proved norm/energy estimate and a domain containing the straight segments between all nominal and perturbed states. Conserved energy alone does not control their separation. |
| Validated variational flow with nonlinear remainder | A tangent calculation needs a uniform remainder and a self-consistent inflated tube for the full family; the old nominal tubes do not supply those premises automatically. |

The global bound is the first inexpensive, falsifiable sufficiency test. A
failure may motivate a separately admitted sharper method for the unchanged
family; it authorizes neither a width search nor an automatic solver campaign.
The old connected derivative and remainder premises remain conditional on
their frozen producer. A new comparison must verify both saved results and
their retained byte associations, rebuild their observation arithmetic and
match fresh nominal primitives; reading cached success flags is insufficient.
Host and child uncertainty
must not pass through an adapter that reinstates the exact nominal child.

The family definition, global nonlinear theorem and arithmetic cost preflight
preceded the target evaluation below. Records of that analytical admission
and its constructed instrument validation are under
`artifacts/parent-child-preparation-preflight-20261023/`.

<a id="adjacent-preparation-global-bound-result"></a>
## 5. The fixed global comparison is insufficient

One prospectively frozen audit evaluated the unchanged adjacent family and
method on the original half-unit window. Both whole-window squared-radius
sufficiency gates fail. The nominal signed-gap gate still passes. This is an
admitted `insufficient_bound` result, not an interrupted calculation or a
counterexample to robustness of the prepared system.

The source-associated [consumer](../../src/tnfr/research/maintained_sha_preparation_response.py)
freshly re-admitted the nominal center, captured both complete supports and
matched every original primitive, held parameter and derived cut association
to the saved evidence. It verified the retained result/final-state/verifier
links and rebuilt all 256 connected and 256 cut observation cells and their
aggregate arithmetic. Old connected derivative and whole-cell enclosure
premises remain those of the frozen producer; none was silently strengthened.

Only after those checks did it construct all 42 exact adjacent hulls. Both
branches' initial form block allowance is `2^-54`; both phase block allowances
are `2^-50`. The cut child retains its separate constant allowances `2^-55`
in form and `2^-52` in registered phase. Each member still has its own conserved
means. No narrower rounding cells, correlated subset, mean projection,
independent re-registration or altered observation replaced the chosen family.

The table rounds exact rational endpoints outward to 18 decimal places.
The error columns bound possible separation from the nominal portrait; they
are not measured errors of perturbed trajectories.

| Branch | Squared portrait-error upper bound | Nominal squared-radius lower bound | Sufficient margin lower bound |
| --- | --- | --- | --- |
| Connected | `4.194309784994176759` | `0.007228431602320119` | `-4.187081353391856640` |
| Cut | `0.031397414021076556` | `0.021065382833320690` | `-0.010332031187755866` |

The rebuilt nominal clockwise difference has lower bound
`17.710892743644908937`; subtracting the upper enclosure of pi leaves a
strictly positive margin with lower bound `14.569300090055115699`. Thus the
missing premise is uniform relative closeness, rather than nominal angle
separation. A negative sufficient radius margin establishes neither an actual
radius crossing nor an ordering reversal. The global error majorant discards
graph structure, restoring signs and correlations; this test does not measure
how much each discarded feature would improve the bound.

The audit used exactly two source admissions, one primitive capture, 512 saved
observation checks, 42 coordinate hulls and 128 two-dimensional comparison
steps. Historical admission separately checked 12 certificates, 24 order-4
Taylor steps and 24 Picard calls. There were no new target fields, derivative
jets, complete-law trajectories, events or pressure refreshes. No family
member was sampled and no comparison propagation was repeated.

The worker completed in about 26.31 seconds; supervised elapsed time was
31.13 seconds. Sampled aggregate working-set peak was 209,715,200 bytes; the
actual worker's OS peak was 213,180,416 bytes. These are different measurements,
both within the 512-MiB allowance. The worker exited successfully and the
supervisor verified an empty Job and completion of every owned process.

Before the attempt, 105 constructed pytest controls, 35 process-admission
predicate checks and three harmless real-process cases passed. The exact
2,507-file instrument and 557 input byte associations were checked again
after execution, before documentation changed. The public consumer's
[controls](../../tests/research/test_maintained_sha_preparation_response.py)
include manufactured saved observations, strict provenance rejection and an
ordinary insufficient-bound outcome; those fixtures are not trajectory proofs.

The protocol, immutable single result, process evidence and independent
reviews are retained under `artifacts/parent-child-preparation-audit-20261023/`.
The attempt is closed without a family robustness certificate. A further
method would need separate analytical admission for this same family and
both complete laws. It must not use failure of a sufficient bound to infer
physical fragility, absence of organization or a fundamental TNFR defect.

<a id="sine-relative-preparation-state"></a>
## 6. Exact relative state and conserved preparation offsets

The failed global comparison motivates a different error description, not a
change of preparation. The following results apply to each complete dynamic
component separately. Require positive held capacities and connected positive
conductance support, in addition to connected bare support. Zero conductances
are allowed only if the remaining positive conductances connect the component.
Bare connectivity alone would not make the form quadratic coercive. The
isolated cut child is outside this dynamic component and retains both constant
coordinates. No inverse mobility is assigned to it.

Choose an oriented incidence matrix B with edge column b_e=e_i-e_j. Define

\[
 L_w=B\operatorname{diag}(W_e)B^T,\quad L_1=BB^T,\quad
 D_\nu=\operatorname{diag}(\nu_i/d_i),\quad
 \rho_i=d_i/\nu_i,\quad \alpha=\rho/(\mathbf1^T\rho).
\]

In the original structural clock, the supplied law is exactly

\[
 \dot x=-\pi^{-1}D_\nu B\sin(B^T\theta),\qquad
 \dot\theta=\pi^{-1}D_\nu L_wx.
\]

Since alpha^T D_nu is a constant row, both alpha-weighted means are conserved.
For two preparations evolving under this same law, let u=tilde x-x and
v=tilde theta-theta. Their mean errors m_x=alpha^T u and m_theta=alpha^T v
are constant. The projection P=Id-1 alpha^T removes only these origins:
u=P u+1 m_x and v=P v+1 m_theta. This is an exact coordinate decomposition,
not a restriction that the perturbed preparation has the nominal means.
The relative coordinates have dimension 2(n-1); retaining both means recovers
the full error. The connected component therefore still has 40 relative
coordinates, and the cut host 38. This does not put either into a 24-dimensional
full-state validator or establish a smaller autonomous parent-child state.

For phi=B^T theta and eta=B^T v, the fundamental theorem of calculus gives

\[
 \sin(\phi_e+\eta_e)-\sin\phi_e=c_e\eta_e,\qquad
 c_e=\int_0^1\cos(\phi_e+s\eta_e)\,ds.
\]

Consequently, with C=B diag(c_e) B^T, the exact finite-error rows are

\[
 \dot u=-\pi^{-1}D_\nu C v,\qquad
 \dot v=\pi^{-1}D_\nu L_wu.
 \tag{6.1}
\]

The same equations hold for P u and P v. C depends on both complete solutions
through every intervening phase segment; it is not merely the Jacobian of the
nominal trajectory. A tangent bound cannot replace that segment premise.

<a id="sine-relative-preparation-metric"></a>
## 7. A positive graph metric with signed restoring cancellation

Let C_0 be a fixed symmetric positive quadratic on relative phase coordinates,
with kernel exactly span{1}. The concrete candidate below chooses C_0=L_1,
independently of the retained phase values. Define the squared relative norm

\[
 \mathcal E=u^TL_wu+v^TC_0v=X+Y,\qquad R=\sqrt{\mathcal E}.
\]

It controls all relative form and phase coordinates; the two conserved origins
are retained separately. It is an auxiliary error metric under this supplied
law, not a new structural field or a conserved physical energy. Different
declared form/phase scales would change the metric and require their own bounds.

Differentiating with (6.1) cancels the large reciprocal terms exactly:

\[
 \dot{\mathcal E}
 =\frac2\pi u^TL_wD_\nu(C_0-C)v.                 \tag{7.1}
\]

This extends the [directional metric cancellation](SINE_DIRECTIONAL_MAINTENANCE.md#3-a-metric-that-retains-directional-error)
to finite preparation errors on the full held component. That earlier
example's coefficients, domain and reduction do not transfer. If

\[
 |u^TL_wD_\nu(C_0-C)v|\le k\sqrt{XY},
\]

then 2 sqrt(XY)<=X+Y gives

\[
 |\dot{\mathcal E}|\le(k/\pi)\mathcal E,\qquad
 D^+R\le\gamma R,\quad \gamma=k/(2\pi).          \tag{7.2}
\]

No positivity of the actual signed phase Hessian is needed for this metric.
Negative secant cosines enter C with their signs. They can increase the growth
bound, but do not invalidate positivity of the chosen error norm. At zero
error, uniqueness preserves zero; the norm inequality extends by continuity.

### A certificate using component-sized matrices

For any positive relative quadratic A define its invertible completion

\[
 Q_A=A+\rho\rho^T/(\mathbf1^T\rho).
\]

This is positive definite, and z^T Q_A z=z^T A z on rho^T z=0. The added
term has zero derivative along either complete error row. Let Q_theta=Q_C0
and F=D_nu L_w D_nu. Suppose each secant coefficient has an admitted interval
[c_e^-,c_e^+], with midpoint m_e and radius r_e. Set

\[
 D_m=C_0-B\operatorname{diag}(m_e)B^T,\qquad
 s^2=\operatorname{tr}(Q_\theta^{-1}D_m F D_m),
\]
\[
 a_e=b_e^TFb_e,\qquad q_e=b_e^TQ_\theta^{-1}b_e,\qquad
 k_{\rm trace}=\sqrt{s^2}+\sum_e r_e\sqrt{a_eq_e}. \tag{7.3}
\]

The trace is nonnegative: it is the squared Frobenius norm of
L_w^(1/2) D_nu D_m Q_theta^(-1/2). Its operator norm is at most that Frobenius
norm. Each interval residual has the rank-one form
epsilon_e L_w^(1/2) D_nu b_e b_e^T Q_theta^(-1/2), whose norm is at most
r_e sqrt(a_e q_e). The triangle inequality proves (7.2) with k_trace.
These square roots are proof notation; the numerical recipe uses the shared
outward square-root owner. No irrational matrix square root is computed.
Apply the norm estimate to P u and P v: the difference matrices annihilate
the common origins and Q_theta equals C_0 on that centered slice.
The entire midpoint matrix is retained before the norm estimate. Correlations
lost by interval radii and the Frobenius bound weaken, rather than strengthen,
this sufficient certificate.

One could instead prove D_m F D_m <= s^2 Q_theta by a supplied exact matrix
inequality. The trace recipe fixes s^2 without a rate search or fitted metric.
Both recipes use n-by-n component matrices, not a 2n-dimensional LMI.
This algebra does not change the state dimension or bypass the full-state
validator's policy.

For the concrete choice C_0=L_1, a cheap second bound is available. Put

\[
 a_0=2\max_i\nu_i,\quad
 b_0=2\max_i\frac{\nu_i\sum_{j\sim i}W_{ij}}{d_i},\quad
 \Delta=\max_e\sup_{c\in[c_e^-,c_e^+]}|1-c|.
\]

For intervals admitted within [-1,1], 0<=Delta<=2. The degree bounds imply
||D_nu^(1/2)L_1 D_nu^(1/2)||<=a_0 and the corresponding weighted bound <=b_0.
Also 0<=L_1-C<=Delta L_1. Factoring on the relative subspace proves

\[
 k_{\rm degree}=\Delta\sqrt{a_0b_0},\qquad
 k=\min(k_{\rm trace},k_{\rm degree}).           \tag{7.4}
\]

Taking the minimum intersects two proved upper bounds; it is not a source
parameter search. At the global Delta=2 it has the same exponential rate
sqrt(a b) as section 2's two-block comparison. An informative improvement
therefore needs admitted phase information or useful graph cancellation;
merely renaming the norm does not guarantee one.

<a id="sine-relative-preparation-storage"></a>
## 8. Relative storage: stronger premises and a useful distinction

The [moving-pattern storage identity](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window)
has a general complete-law form. Its Bregman remainder is

\[
 \mathcal D=\frac12u^TL_wu+
 \sum_e[\cos\phi_e-\cos(\phi_e+\eta_e)-\sin\phi_e\,\eta_e].
\]

Differentiating along both solutions cancels the form/phase exchange and yields

\[
 \dot{\mathcal D}=\sum_e\dot\phi_e
 [\sin(\phi_e+\eta_e)-\sin\phi_e-\cos\phi_e\,\eta_e]. \tag{8.1}
\]

This is generally neither nonnegative nor a distance metric. Suppose a
whole-segment domain supplies a fixed K, positive on relative coordinates
with kernel span{1}, such that

\[
 B\operatorname{diag}\cos(\phi+s\eta)B^T\succeq K
 \quad(0\le s\le1).
\]

Taylor's integral remainder then gives
2D>=u^T L_w u+v^T K v. If the nominal edge rates additionally satisfy
B diag(|phi_dot|) B^T <= lambda K, the sine remainder bounded by eta_e^2/2
gives |D_dot|<=lambda D. Thus D(t)<=D(0) exp(integral lambda) while the
domain holds. The nominal rates and that domain require whole-time evidence;
energy conservation alone supplies neither.

Graph coercivity is weaker than requiring every edge to be acute. A constructed
triangle with phases (a,0,-a), cos(a)=2/3, has cosine weights
(2/3,2/3,-1/9). Its two nonzero Hessian eigenvalues are 2 and 4/9, so
L_cos>=(4/27)L_1 despite one negative edge. If every segment edge differs
from this reference by at most 2/27 radians, cosine's Lipschitz bound gives
L_segment>=(2/27)L_1. This is a compatible mathematical domain example,
not an admission of the retained preparation.

A bridge cannot receive this cycle compensation. If its actual cosine is
nonpositive, a vector constant on either side makes the signed Hessian
quadratic nonpositive. No positive relative lower bound K can then hold at
that state. This excludes that coercive-storage recipe, not the positive
metric of section 7 or the actual robustness of a prepared system.

The two-node pi-gap equilibrium gives a separate global obstruction. With
equal mobility m=nu/(pi d) and conductance w>0, the relative linearization
has eigenvalues +/-2m sqrt(w). Every fixed positive quadratic norm has a
logarithmic growth bound at least 2m sqrt(w) at that equilibrium. Its phase Bregman remainder
is cos(eta)-1<0 near zero. A global nonexpansion claim is therefore false for
this law. At a zero nominal gap, a registered 2pi edge-lift error even has
zero phase remainder; small D without a chart cannot control real lifts.

None of these witnesses describes the retained source. A full clockwise turn
of the original portrait also does not imply a nonacute bridge: the ordinary
two-node conservative oval with energy between zero and one turns while
remaining inside |delta|<pi/2. A turn count cannot replace a phase-domain check.

<a id="sine-relative-preparation-domain-observation"></a>
## 9. Noncircular domains and the original marked reading

For a zero-sum linear reading ell and positive relative A, define
sigma_A(ell)=ell^T Q_A^(-1) ell. Completing the quadratic or applying
Cauchy--Schwarz proves

\[
 (\ell^Tz)^2\le\sigma_A(\ell)z^TAz.
\]

The value equals the usual quotient inverse coefficient; the rank completion
permits exact nonsingular solves. It needs no electrical interpretation when
the admitted relative quadratic has signed edge coefficients. For the connected
portrait ell=e_child-e_parent, the two blocks share the same error budget:

\[
 \|\widetilde y-y\|^2
 \le\sigma_*\mathcal E,\qquad
 \sigma_*=\max\{\sigma_{C_0}(\ell),w\sigma_{L_w}(\ell)\}. \tag{9.1}
\]

The maximum follows from sigma_phase Y+w sigma_form X<=sigma_*(X+Y).
Independently maximizing both block energies and adding them would lose this
correlation. Both conserved mean errors cancel in this connected difference.

For the cut host use ell=e_parent-alpha and retain the constant offsets

\[
 b_x=u_{\rm child}(0)-\alpha^Tu_{\rm host}(0),\qquad
 b_\theta=v_{\rm child}(0)-\alpha^Tv_{\rm host}(0).
\]

The exact portrait error is (b_theta-ell^T v, sqrt(w)(b_x-ell^T u)), hence

\[
 \|\widetilde y_0-y_0\|
 \le\sqrt{b_\theta^2+w b_x^2}+\sqrt{\sigma_*\mathcal E}. \tag{9.2}
\]

These offsets are linear images of the original preparation. For example,
with coordinate errors in [-e_i^-,e_i^+], the exact interval for b_x is
[-e_child^- - sum alpha_i e_i^+, e_child^+ + sum alpha_i e_i^-]. The phase
interval has the same construction. Enclosing these intervals and the relative
norm separately may lose correlation, but does not remove any family member.
Replacing either offset by zero would change the claim.

For C_0=L_1 every member of the unchanged hull satisfies

\[
 R(0)^2\le\sum_{\{i,j\}}W_{ij}(r_{x,i}+r_{x,j})^2
              +\sum_{\{i,j\}}(r_{\theta,i}+r_{\theta,j})^2.
\]

Use an outward square root of this bound as the common initial radius R_0.
It bounds the full affine image; it does not project the family onto nominal
means or assert that all separately maximal edge errors occur simultaneously.
The cut sums run over the host, with the two child offsets retained separately.

### Whole-window first exit

Fix positive edge-error allowances h_e before evaluating the reserved response.
On each original nominal cell, inflate the admitted full edge-phase interval
by [-h_e,h_e]. Every nominal-to-perturbed segment lies there while
|eta_e|<=h_e. Cosine enclosures on these inflated intervals therefore bound
the secants consumed by (7.3)--(7.4). The field is globally Lipschitz, and
its Jacobian depends only on phases; no new form box or assumed perturbed
trajectory is needed to state this conditional argument.

Let gamma_j>=0 be the derived growth bound on cell j, and propagate one scalar
upper radius using the shared outward exponential arithmetic. If R_j^+ is an
upper bound at that cell's end, monotonicity bounds the preceding cell too.
Initial membership and all strict gates

\[
 q_e(R_j^+)^2<h_e^2\quad\text{for every edge and cell}             \tag{9.3}
\]

prevent the first exit: an exit would have |eta_e|=h_e, contradicting its
dual-norm bound. This closes the domain without presuming robustness.
Uniform equations (9.1)--(9.2) must then pass the original nominal-radius gates
and shared-initial-angle criterion from section 3. Endpoint checks on actual
trajectories, nominal cells without inflation, or a center-only Jacobian do
not provide this proof. The inherited nominal whole-cell and derivative
premises are unchanged.

The relative-storage route has the analogous first-exit gate
2 sigma_K(b_e) D_upper<h_e^2, but first requires its additional graph-coercive
segment domain and nominal edge-rate inequality. Failure of any sufficient
gate leaves the unchanged family undecided.

<a id="sine-relative-preparation-method-boundary"></a>
## 10. Method choice and bounded implementation prerequisite

The fixed positive metric C_0=L_1 is the next candidate because its positivity
needs connected support, rather than positive actual phase curvature. The
Bregman identity clarifies the extra information needed for a storage-based
route and supplies useful obstructions; it is not a second active campaign.
Both routes retain the complete host and its cut-dependent weighted means.
Neither admits the current source's domain or certifies family robustness here.

The candidate uses existing exact inverse/product/positivity primitives,
outward interval cosine and square roots, and the shared scalar exponential
comparison. It needs component matrices of size 21 and 20, two inverse
completions per component, and static edge/observation dual factors. For the
original 256 cells per branch, there would be 512 coefficient records and
scalar growth updates, with 256*(53+52)=26,880 inflated edge-cosine enclosures.
Two dense products for D_m F D_m use at most
2*256*(21^3+20^3)=8,837,632 scalar product terms, plus trace contractions,
interval projection, static work and scalar operations. This is an algebraic
work count, not a measured runtime or memory guarantee; rational numerator
growth and retained evidence admission must be included in a resource test.
Preparation widths and their squared norm sums must remain exact before
normalization. Reuse the existing metric-error scaling pattern before outward
square roots or scalar growth; an absolute dyadic grid can otherwise erase
useful information about tiny or subnormal initial errors. Preserving a valid
but needlessly coarse enclosure would not establish a useful certificate.

The method evaluates no new full-law trajectory, field jet or event. It does
not invoke or enlarge the 24-dimensional comparison/metric-Taylor APIs.
All 42 connected and 40 evolving cut-host coordinates remain in the theorem;
the smaller certificate is a proved norm bound, not a dynamical reduction.
The numerical scalar owner still requires |gamma_j h_j|<=1. Failure must be
reported under a fixed prospective work policy, without an automatic finer
mesh, rate search, metric replacement or family-width change.

The analytical gate is closed with this conditional theorem, the exact
limitations above and a concrete arithmetic route. The implementation must
first be prepared and validated on constructed components at the intended
sizes, with explicit byte, time and rational-work bounds. Only a later
separately frozen source-associated audit may evaluate the actual domain,
growth and portrait gates. Its metric, edge allowances, coefficient recipe,
cell mesh, observation and failure policy must precede the target result.

Constructed symbolic/exact controls are maintained in
[the relative-preparation tests](../../tests/research/test_sine_relative_preparation_metric.py).
The analytical scope, reviews, controls and preservation records are under
`artifacts/relative-preparation-metric-20261023/`. No source admission, target
cell consumption, family-width calculation or comparison propagation is part
of this analytical result. The sole [research program](../RESEARCH_PROGRAM.md#deferred-composition-question)
owns the next admission and any later priority reassessment.

<a id="sine-preparation-metric-instrument"></a>
## 11. Pure metric instrument and remaining evidence admission

The [pure metric owner](../../src/tnfr/research/sine_preparation_metric.py)
implements the fixed graph metric above. It admits each actual connected
component, builds the two exact rank-completed inverses, derives edge and
portrait duals, and propagates one normalized radius through a declared
sequence of whole-cell phase boxes. It uses the shared exact linear algebra,
interval cosine and square root, and scalar comparison owners. It neither
executes a trajectory nor authenticates the supplied nominal boxes.

Prepared contexts and families are immutable live capabilities: copied,
replaced or deserialized diagnostic records cannot stand in for builder
results. Each run derives its own secant bounds from primitive phase boxes.
The complete input sequence is admitted before propagation. Connected
portraits and cut-host portraits retain their different reading vectors;
the latter includes both asymmetric mean-to-child offsets.

Power-of-two normalization applies to the exact squared quantity before its
outward square root, including tiny graph weights and observation factors.
One exact radius scale is then retained across all scalar cells. This avoids
an artificial absolute error floor for subnormal preparation widths without
weakening any enclosure. Zero error remains exactly zero.

The implementation admits at most 21 nodes, 53 edges and 256 cells per branch,
4096-bit input rationals and 65536-bit values at checked operation boundaries.
These are work policies. Shared inverse and product kernels can create larger
internal temporaries before a boundary check; the policies do not establish
a universal memory or runtime bound. Invalid primitives or setup work reject
construction. An admitted run returns an explicit unavailable prefix at an
initial-domain, scalar-work, intermediate-bit or strict first-exit failure.
Unexpected kernel errors propagate without retry. An unavailable prefix is
not a whole-window preparation certificate or an observed dynamical escape.

The [constructed arithmetic controls](../../tests/research/test_sine_preparation_metric.py)
cover independent inverse/resistance values, unequal mobilities, retained
zero-conductance bare edges, negative secants, asymmetric means and child
errors, tiny weights and subnormal widths, strict domain gates, chronology,
input/work boundaries and rejected cached capabilities. Together with the
prior analytical controls and new resource-adapter controls, the selected
106-test validation passes. These tests consume no retained target source.

### The synthetic measurement did not reach arithmetic

A single resource attempt froze a manufactured 21-node/53-edge component
and its 20-node/52-edge cut host, 256 cells each, with fixed perturbed dyadic
coefficients. Its declared worker/watchdog allowances were 300/330 seconds,
512 MiB sampled aggregate process memory and 64 MiB response output. The
response allowance was fixed before execution to accommodate exact per-cell
diagnostics; sampled memory supervision is not an operating-system hard
allocation guarantee.

That attempt stopped before workload admission: the worker looked up the
bootstrap marker under a different global name from the unchanged supervisor.
The process handshake, job association and final cleanup passed, but no
metric setup or cell calculation began. Its startup time and memory cannot
be reported as the cost of the intended arithmetic. The failed instrument
and result remain unchanged; a separately prepared worker correction and
regression control do not reopen or repeat that attempt.

The arithmetic implementation is therefore validated on constructed controls,
while intended-size resource admission remains open. This is an implementation
prerequisite, not a negative finding about the metric or the TNFR dynamics.
The records, frozen source archive and prospective method contract are under
`artifacts/preparation-metric-instrument-20261023/`.

### Full phase evidence is a separate premise

The earlier preparation audit reconstructed marked parent-child observations.
Those observations do not supply the full phase intervals needed for all
graph secants. A future passive consumer must bind the 256 retained connected
step archives and 256 cut-host archives to their producers, sources, actual
node order, original phase lifts, clock, initial state and endpoint carry.
It must verify archive and expanded-member receipts, strict schemas, bounded
sizes and complete whole-cell tubes before extracting the phase coordinates.
Endpoint hulls or marked phase differences cannot replace these tubes.

The connected producer's existing reader recomputes the tube field during its
Picard audit. The saved connected records do not retain that full rate vector.
A passive reader can verify saved producer associations, positive recorded
margins, tube inclusion, Taylor arithmetic and carry, but must inherit the
whole-tube flow/derivative/remainder premises. It cannot describe this as a
fresh independent Picard verification. The cut producer retains tube-rate
enclosures and permits additional algebraic checks, under the same honest
distinction between supplied enclosures and independently proved derivatives.
Receipt consistency also does not establish independent historical authenticity.

The prospective method fixes C_0=L_bare, edge allowances of 1/1024 radians,
the original 256 cells per branch of duration 1/512, unchanged full adjacent
family and marked reading. It admits no fitting or automatic mesh/family
change. A later executable audit still requires the passive consumer and its
controls, an admitted complete-pipeline resource budget and a separate freeze.
That freeze must distinguish any required historical source recertification
from prohibited new response fields, jets, trajectories or events and account
for their costs explicitly. Ordering would require both strict portrait gates
epsilon_connected^2<q_connected and epsilon_cut^2<q_cut, followed by the original
shared-initial-angle criterion with gap.lower>pi.upper.

No retained target source, phase cell, adjacent width or new metric propagation
was evaluated in this preparation. The earlier two-block `insufficient_bound`
result remains closed, and the unchanged family's robustness remains unresolved.
The [research program](../RESEARCH_PROGRAM.md#deferred-composition-question)
owns the single remaining evidence-and-resource admission gate.

<a id="sine-preparation-passive-pipeline"></a>
## 12. Passive phase evidence and a separate resource measurement

This closes the evidence/resource preparation left open in section 11.

The [passive phase reader](../../src/tnfr/research/maintained_sha_phase_evidence.py)
admits the complete saved state tubes needed by the graph metric. Each branch
has 256 original cells of duration 1/512: 42 coordinates for the connected
system and 40 for the cut host. The reader retains the original node order,
real phase lifts, initial box and exact endpoint carry. It extracts each
whole-cell phase box; neither endpoint hulls nor the marked parent-child
differences replace that domain.

Admission checks bounded archive and expanded-member sizes, receipts, member
names, strict step schemas, exact scalar encodings, dimensions, clock and
method. It reconstructs the stored Taylor increment and endpoint through the
shared arithmetic owner. It also checks primitive support, conductance,
capacity, degree, cut-host embedding, weighted means and source associations.
Successful results are immutable live capabilities; a copied or deserialized
record cannot replace a completed admission.

This is a passive reconstruction with explicit inherited premises. The
connected records retain a positive Picard margin but no full tube-rate
vector. Their whole-tube flow inclusion and derivative/remainder enclosures
remain producer premises. The cut records additionally retain tube rates,
so the reader reconstructs their Picard image and strict interior margin.
That arithmetic does not independently prove that those supplied rates or
Taylor coefficients enclose the selected differential law. Hash and receipt
consistency bind consumed bytes; they do not authenticate historical truth.

The [source-associated consumer](../../src/tnfr/research/maintained_sha_metric_response.py)
adds the complete frozen producer/source inventories and re-admitted source
association. Every consumed phase receipt must belong to those inventories.
Full tubes must reproduce the corresponding saved marked form and phase
intervals, with the cut child's nominal frozen coordinates reinserted only
for this nominal association check. The family calculation separately retains
the child's preparation error and both host mean offsets.

The consumer constructs the same 42 adjacent-coordinate hulls for both
branches and preserves their asymmetric widths. It uses the actual complete
supports, cut degrees and local host parent index. The prescribed metric,
allowance, observation, original cells and scalar carry remain fixed. A
source-associated result would require whole-window metric admission, both
strict original portrait-radius margins and the shared-initial-angle gap
criterion. A failed sufficient gate is not an observed loss of robustness.

### Constructed validation and measured workload

The [reader controls](../../tests/research/test_maintained_sha_phase_evidence.py)
and [integration controls](../../tests/research/test_maintained_sha_metric_response.py)
exercise independent constant-field Taylor arithmetic, full phase extraction,
exact carry, primitive and rehashed association tampering, archive limits,
missing inventory entries, the shared family and cut offsets. The selected
277 initial constructed controls passed, together with formatting and lint checks.
These controls do not consume the retained target source or establish
derivative certificates for it.

A separately frozen resource attempt preserves the earlier arbitrary metric
workload: a manufactured 21-node/53-edge component and 20-node/52-edge cut
host, 256 prescribed phase boxes each, with the same coefficients, preparation
widths and allowances. Those metric inputs are not asserted to be solution
tubes. The corrected worker uses the unchanged bootstrap contract; the prior
failed attempt remains immutable.

The new attempt also constructs and passively reads 256 connected and 256
cut-host archives for an independent affine constant-vector-field example.
Its dyadic order-12 records have zero higher derivatives and exact carry.
Each expanded member is padded to 196608 bytes with trailing JSON whitespace,
giving 96 MiB across the 512 members before ZIP compression. Padding changes
the parsing and byte workload without changing mathematical content.

These are separate workloads in one supervised measurement. The coherent
constant-field archives do not replace the unchanged arbitrary metric boxes,
and their associated graph metadata does not make that field the sine law.
Zero higher derivatives and compressible padding do not reproduce the real
archives' rational complexity. The measurement therefore cannot establish a
universal cost bound, or by itself certify the cost of the actual target audit.

The declared run includes fixture construction, serialization, bounded passive
admission, metric setup and propagation, output and process cleanup. It
completed all 512 passive cell admissions, 512 Taylor reconstructions and
512 metric cells. The supervisor measured 89.80 seconds including startup and
cleanup, a sampled aggregate working set of 188862464 bytes, and an OS-reported
worker peak of 180514816 bytes. Response output before the monitor's final
record was 36300090 bytes. Cleanup completed and the frozen instrument remained
unchanged. These measurements are within the declared 300/330-second
worker/watchdog, 512 MiB sampled-memory and 64 MiB response allowances.

The 522 constructed input files, including metadata, occupied 1404489 bytes
stored and 100877102 bytes expanded. Setup used four exact inverses, four
matrix products and 112 square roots. Cell work used 1024 products, 512 scalar
steps, 26880 cosines, 512 square roots and 512 pi enclosures. Observed numerator
and denominator sizes reached 3095 bits at checked boundaries. Sampled process
memory and these boundary observations do not bound all internal rational
temporaries or establish a hard allocation limit.

A metadata-only inventory review found valid zero-byte historical trace logs.
The source-associated consumer now admits their nonnegative byte lengths while
still checking their complete paths, digests and dependency associations.
All 23 selected integration controls pass after this correction. The original
277-control record and measured resource archive remain unchanged; a separate
before/after receipt record binds this correction. The pure metric, passive
reader, fixtures and measured resource workers are byte-identical to the
successful measurement. The synthetic measurement was not rerun after correction.

### Boundary before a target audit

The records are under `artifacts/preparation-metric-pipeline-20261023/`.
The executable prospective instrument has 25 passing
constructed controls. It fixes one future attempt, the original method and
family, 300/330-second worker/watchdog allowances, 512 MiB sampled aggregate
memory and 64 MiB response output. Exact input inventories and source
associations must be checked at the separate target freeze and evaluation.
The prepared entry points are `target_protocol.py --freeze` and
`target_monitor.py --run` in that artifact folder; neither was invoked here.
Completed setup/cell records survive a later interruption, while compact
final reports reference those records instead of duplicating them.

The synthetic result supports this bounded attempt; it does not guarantee
completion on the actual archive arithmetic. Historical source re-admission
has its own fixed allowance of 12 history certificates, 24 Taylor calls and
24 Picard calls. The passive response stage permits none of those new response
field, jet, trajectory or event calculations. Historical re-admission is not
silently included in that zero-response-evaluation claim. Resource or
sufficient-bound failure retains an explicit unavailable prefix without retry.

No retained target phase cell, preparation width or graph-metric propagation
was evaluated in this preparation. The earlier two-block `insufficient_bound`
result remains closed. Robustness of the unchanged family, formation,
autonomous identity and physical identification remain separate questions.
The research program owns the next gate; implementation and a synthetic
measurement do not authorize a target campaign automatically.

<a id="sine-preparation-metric-target-admission"></a>
## 13. Original-family attempt: inventory rejection before response

The separately frozen, single target attempt stopped at complete dependency
admission. Its input collector followed the connected producer's final-state
inventory but omitted the source ZIP linked separately by that producer's
manifest. The ZIP exists and its size and SHA-256 match the manifest. The
unchanged metric consumer correctly rejected the incomplete inventory.

The worker had already completed its first historical source load: six history
certificates, twelve Taylor calls and twelve Picard calls. It stopped before
the second load, source capture, marked-observation reconstruction, phase-cell
decoding, preparation widths or metric arithmetic. Every response-stage counter
is zero. This is an input-association defect, not `insufficient_bound`, a
family counterexample or evidence of instability. The original family's
robustness remains unresolved.

The process handshake and cleanup passed, with the instrument and frozen
inputs unchanged. Supervision took 23.77 seconds; sampled aggregate memory
peaked at 203120640 bytes. These measure startup, one historical admission
and the rejected inventory, not a complete metric workload. Neither resource
limit exhaustion nor scientific arithmetic caused this termination. The
protocol, source archive and failed response remain unchanged under
`artifacts/preparation-metric-target-20261023/`; the attempt was not repeated.

### Separate passive inventory correction

A new collector explicitly follows both nominal producers' final-state and
manifest/archive links, the retention freeze's current and prior dependencies,
and the fixed phase/source file roles. It bounds paths, counts, sizes and
digests, then calls the existing consumer's complete inventory verifier before
returning a candidate. Source-code archive members are verified; phase archives
are only hashed, without opening their members. This admits metadata and byte
associations, not historical dynamics or solution tubes.

All 13 constructed controls pass, with formatting and lint checks. They
reproduce the missing-link rejection, accept manifest-linked archives absent
from the smaller final-state lists, and reject altered dependencies, conflicting
receipts, path aliases, missing/extra phase files and exceeded budgets. Opaque
phase fixtures deliberately cannot be decoded as ZIPs, and scientific callbacks
are disabled in the controls.

The real metadata-only candidate passes the unchanged consumer check. It has
1498 files totaling 361259692 stored bytes: exactly the one omitted 9797634-byte
archive added to the old inventory, with no changed or removed reference.
This preflight performed no historical source admission, phase decoding,
family calculation, metric evaluation or target launch. Its code, controls,
candidate and review are retained separately under
`artifacts/preparation-metric-evaluation-20261023/`.

### Prospective entry admits metadata before history

The separate entry in `artifacts/preparation-metric-entry-20261023/` now invokes
the unchanged complete inventory verifier before its first historical source
loader. It does not trust a prior candidate's success flag. A rejected
dependency stops the entry before historical or response work; a passing
inventory permits the existing historical and metric stage guards to proceed
under their original contracts.

The 29 selected constructed controls pass, with formatting and lint
checks. Missing or changed dependencies exercise the actual metadata verifier;
the entry route establishes rejection before historical/response work, and
admitted inert continuations check the positive ordering. The fixtures retain
opaque phase data. This validation does not reconstruct any retained source
history or evaluate the original family's phase tubes, widths or metric.

The prospective protocol binds the corrected collector, new entry, unchanged
consumer and stage owners, and selected validation to exact source receipts.
It preserves the old resource measurement and explicit inventory correction
association through their immutable archives. A metadata-only contract check
admits the same 1498-file corrected candidate and unchanged method/resource
allowances. Neither the unchanged arithmetic controls nor the successful
synthetic workload need repetition for this ordering correction.

The prepared entry points are `entry_protocol.py --freeze` and
`entry_monitor.py --run` in that folder. Their exclusive future output is
`artifacts/preparation-metric-entry-target-20261023/`. No target freeze or
launch occurred in this repair, and the failed target remains closed. A
separate continuation remains justified because the inventory rejection
reached no scientific response and the fixed metric's predictive capability
has not yet been tested on this family. The
[research program](../RESEARCH_PROGRAM.md#deferred-composition-question)
owns that one bounded evaluation, with the same method and preparation;
this entry correction supplies no new mathematical robustness result.

<a id="sine-preparation-fixed-metric-certificate"></a>
## 14. Conditional ordering certificate for the unchanged preparation family

The single separately frozen continuation returns
`family_ordering_certified`. The fixed graph metric certifies the ordering
of the **net clockwise increments over the complete half-unit window** for
the unchanged shared preparation family. The domain and portrait-radius bounds
hold throughout that window; neither instantaneous angular-rate ordering nor
ordering at every shorter prefix follows.

The preparation keeps all 42 original form/phase coordinates. Each coordinate
can take its exact represented center or either adjacent binary64 value,
giving 3^42 = 109418989131512359209 discrete preparations. The comparison
bound is uniform over their enclosing coordinate hull as well. These are
computational-resolution perturbations of the already prepared post-event
state, not a macroscopic uncertainty allowance or a robustness proof for the
preceding operator history.

Each shared preparation is compared under the same two complete laws. The
connected system retains its 21 nodes and 53 edges; the cut system retains
the 20-node/52-edge host and constant child. The intervention includes bridge
deletion, rebuilt degree normalization and the frozen-child law. Both host
mean offsets and the child's preparation error remain in the cut reading.
This does not isolate bridge deletion from those accompanying law changes.

### Strict gates and the resulting prediction

The method is unchanged: C_0=L_bare, edge inflation 1/1024 radians, and the
original 256 cells of duration 1/512 per branch. Both branches complete all
256 strict first-exit gates. Each full phase branch has been admitted and
matched to its original marked observations before any family width is
constructed. The saved connected whole-tube/derivative/Picard premises remain
inherited. The additional saved-rate inclusion check for the cut branch is
algebraic and does not independently establish its derivative enclosures.

Both original portrait-radius gates pass. Rounded readings of the exact
rational records are shown below; the stored fractions own all strict tests.

| Quantity | Connected | Cut |
| --- | --- | --- |
| Nominal squared-radius lower bound | 0.00722843160232 | 0.02106538283332 |
| Whole-window portrait-error upper bound | 1.7130751e-13 | 6.1679209e-14 |
| Squared-radius margin after subtracting error squared | 0.00722843160232 | 0.02106538283332 |
| Completed strict-domain cells | 256 | 256 |

The nominal connected-minus-cut net clockwise increment remains enclosed in
`[17.710892743644908937,23.706659661060598788]` radians. The shared initial
angular error cancels. Whole-window portrait admission bounds the sum of the
two remaining endpoint angle errors by less than pi, and the exact saved
`gap.lower - pi.upper` exceeds 14.5693 radians. Consequently the connected
net increment is strictly greater for every common preparation in the
admitted hull. This is a finite-preparation prediction under the stated
complete-law and nominal-proof premises, not just the earlier point comparison.

The earlier global two-block audit retains its `insufficient_bound` outcome.
Its failed sufficient estimate was not a counterexample. This separately
declared graph-resolved comparison supplies the previously missing uniform
bound without reducing the family, changing the observation window or fitting
the metric after seeing the response.

### Execution and evidence boundary

Complete metadata admission precedes historical work. The two historical
source loads used their declared twelve history certificates, twenty-four
Taylor calls and twenty-four Picard calls. The response stage then completed
512 saved-observation checks, 512 passive phase admissions, 512 saved Taylor
reconstructions and 512 metric cells. It constructed no new response field,
derivative jet, trajectory or event.

The worker took 134.55 seconds; supervision including startup and cleanup took
136.73 seconds. Sampled aggregate working set peaked at 228773888 bytes and
the worker's OS-reported peak was 225243136 bytes. The 528 response files
total 38137090 bytes after the final monitor record. These are within the
unchanged 300/330-second, 512-MiB sampled-memory and 64-MiB output allowances;
sampled working set is not a hard allocation guarantee. Process admission,
final cleanup and frozen-source/input association all passed.

The protocol, 923-member source archive, complete input association and single
evaluated response remain under
`artifacts/preparation-metric-entry-target-20261023/`. The saved-record audits
under `artifacts/preparation-metric-continuation-20261023/` check receipts,
chronology, carry, strict gates and resource outcome without rerunning source
or response propagation. Earlier failed attempts and successful preparation
records remain unchanged. No second attempt was made in this continuation.

This closes the declared computational-resolution comparison. A post-reset
nominal response, wider preparation tolerance, formation, autonomous identity
and physical identification require their own premises. The
[research program](../RESEARCH_PROGRAM.md#deferred-composition-question)
now selects one analytical operator-to-successor uncertainty question rather
than extending this completed metric evaluation.

<a id="sine-positive-capacity-uncertainty"></a>
## 15. Uncertainty in positive held capacities

The [NUL reset contract](SINE_FORM_CAPACITY_ACTION.md#nul-finite-reset-contract)
identifies the missing terms when capacities vary across preparations.
This section closes that conditional continuation problem for the same full
sine law. It changes neither the law nor the fixed graph metric. Each member
holds its own capacities constant; capacity dynamics, structural events and
selection policies remain outside this comparison.

### 15.1 Complete family and exact finite-error equations

Keep the connected dynamic component, weighted and bare supports, structural
clock and declared phase lifts of section 6. Choose nominal capacities nu^0
and a supplied compact box

\[
 0<\nu_i^-\le\nu_i^0\le\nu_i^+<\infty,\qquad
 \widetilde\nu_i\in[\nu_i^-,\nu_i^+].
 \tag{15.1}
\]

Retain associated initial signed form and lifted-phase errors, their common
origins, and any constant-child errors. A correlated state/capacity family
may be enclosed by these boxes; the certificate must include every member,
not silently replace its preparation by independent nominal coordinates.
No capacity inverse is assigned to an isolated child with a supplied
constant law. Its two coordinates remain part of the cut observation.

Let D_0=diag(nu_i^0/d_i), D_tilde=D_0+DeltaD, and define s=B sin(B^T theta)
and q=L_w x on the nominal solution. With u=x_tilde-x, v=theta_tilde-theta
and C the exact secant Laplacian from section 6,

\[
 \dot u=-\pi^{-1}\widetilde D C v+a,\quad
 \dot v=\pi^{-1}\widetilde D L_wu+b,\qquad
 a=-\pi^{-1}\Delta D s,\quad b=\pi^{-1}\Delta D q.
 \tag{15.2}
\]

Both additive terms are essential. They depend on independently specified
nominal state and the supplied capacity uncertainty, not on a pressure
reconstructed from the perturbed response. The pressure and phase rows
retain their original units and clock; a common time rescaling must transform
both rows and is not a general substitute for local capacity differences.

### 15.2 A uniform bound in the existing graph metric

Use the unchanged quadratic E=u^T L_w u+v^T L_1 v and R=sqrt(E). Its exact
derivative is

\[
 \dot E=\frac2\pi u^TL_w\widetilde D(L_1-C)v
                   +2u^TL_w a+2v^TL_1b.
 \tag{15.3}
\]

The nominal inverse completion Q_theta from section 7 can still bound
instantaneous relative coordinates: subtract their current nominal-weighted
mean before applying the quadratic inequality. That is an algebraic step.
Its added rank term is not differentiated as if the nominal-weighted means
were conserved under the changed law.

Set

\[
 \delta_i=\max\{\nu_i^0-\nu_i^-,\nu_i^+-\nu_i^0\}/d_i,\qquad
 \eta=\max_i\frac{d_i\delta_i}{\nu_i^0}.
 \tag{15.4}
\]

Admitted inflated phase intervals give the same signed secant bounds as
before. Write Delta=max_e sup|1-c_e| and M=L_1-C. Since c_e lies in [-1,1],
`0 <= M <= Delta L_1`. Let k_nom be the existing minimum of the nominal
trace and degree bounds and k_deg=Delta sqrt(a_0 b_0). Then

\[
 k_* = k_{\rm nom}+\eta k_{\rm deg},\qquad
 |u^TL_w\widetilde D Mv|
 \le k_*\sqrt{u^TL_wu}\sqrt{v^TL_1v}.
 \tag{15.5}
\]

To prove the additional term, factor DeltaD as
`D_0^(1/2) H D_0^(1/2)`, where H is diagonal with norm at most eta.
The norm of `L_w^(1/2) D_0^(1/2)` is at most sqrt(b_0). Factoring M through
the relative square root of L_1 bounds
`D_0^(1/2) M Q_theta^(-1/2)` by Delta sqrt(a_0): the middle relative factor
has norm at most Delta and `L_1 <= Q_theta` bounds the final factor by one.
Multiplying these three bounds proves the perturbation allowance in (15.5).
No numerical matrix square root is required. There is no eta<1 assumption;
positive lower endpoints are admitted separately in (15.1).

Suppose whole nominal cells supply |s_i|<=S_i and |q_i|<=Q_i. Define

\[
 J^2=\sum_{\{i,j\}}W_{ij}(\delta_iS_i+\delta_jS_j)^2
          +\sum_{\{i,j\}}(\delta_iQ_i+\delta_jQ_j)^2.
 \tag{15.6}
\]

The two terms bound the graph norms of pi*a and pi*b. Applying
Cauchy--Schwarz to the two blocks jointly in (15.3) yields

\[
 D^+R\le\gamma_* R+F,\qquad
 \gamma_*=k_*/(2\pi),\quad F=J/\pi.
 \tag{15.7}
\]

Signed interval products for each component of pi*a and pi*b can give a
tighter J: subtract their endpoint intervals along each edge, square the
maximum absolute difference, and sum with the same form/phase weights.
This bounds the same two graph norms directly. The implementation uses this
interval subtraction, retaining the reconstructed signed forcing vectors;
it does not assume that every edge extremum occurs simultaneously.

At R=0, the upper right derivative of the seminorm is bounded by the norm
of the additive vector, so the same inequality holds. This is a forced
comparison bound, not a forced physical law: the additive vector describes
the difference between two unforced laws with different held parameters.
If the capacity box collapses to nu^0, eta, J and all later mean drifts are
zero, recovering the existing homogeneous comparison.

### 15.3 Means and both cut offsets

Use nominal weights alpha^0 from nu^0 and write m_x=alpha^{0T}u,
m_theta=alpha^{0T}v. Put t_i=alpha_i^0 DeltaD_i. On an oriented edge e=(i,j),
write phi_e=(B^T theta)_e, eta_e=(B^T v)_e and z_e=(B^T x)_e. Then

\[
 \dot m_x=-\pi^{-1}\sum_e(t_i-t_j)(\sin\phi_e+c_e\eta_e),
 \quad
 \dot m_\theta=\pi^{-1}\sum_e(t_i-t_j)W_e
                      (z_e+(B^Tu)_e).
 \tag{15.8}
\]

The nominal alpha^0 D_0 term cancels because every incidence column sums
to zero. Enclose each t_i from its signed capacity interval and bound
|t_i-t_j| by kappa_e using interval subtraction. Let S_e bound |sin phi_e|,
Z_e bound |z_e|, and let q_theta,e and q_x,e be the existing edge duals.
For a whole-cell relative-radius bound R_bar,

\[
 |\dot m_x|\le\pi^{-1}\sum_e\kappa_e
                     (S_e+\sqrt{q_{\theta,e}}\,\overline R),
 \quad
 |\dot m_\theta|\le\pi^{-1}\sum_e\kappa_e W_e
                     (Z_e+\sqrt{q_{x,e}}\,\overline R).
 \tag{15.9}
\]

Here |c_e|<=1 includes negative secants. Multiply each nonnegative right-hand
side by the cell duration and accumulate it as a symmetric allowance around
the initial mean interval. The same allowance encloses every preceding time
in that cell, not just its endpoint. Independent interval maxima can weaken
the bound but cannot remove a member or presume decorrelated acquisition.

For an isolated constant child the exact reading remains
`(b_theta(t)-ell^T v, sqrt(w)*(b_x(t)-ell^T u))`, with
`ell=e_parent-alpha^0`, b_x(t)=u_child(0)-m_x(t) and
b_theta(t)=v_child(0)-m_theta(t). Expand both initial offset intervals by
their accumulated mean allowance. The portrait error is bounded by

\[
 \sqrt{\sup b_\theta(t)^2+w\sup b_x(t)^2}
                  +\sqrt{\sigma_*}\,\overline R.
 \tag{15.10}
\]

For a connected zero-sum reading, both origins still cancel exactly and only
the second term is needed. The nominal-weight host reading and its dual stay
fixed throughout this comparison; the drifting origins are carried rather
than incorrectly declared conserved.

There is no conflict with each trajectory's own conserved means. If a is a
member's capacity-dependent weight vector and w_a=a-alpha^0, then for either
coordinate z and error e=tilde z-z,

\[
 a^Te(t)=a^Te(0)-w_a^T[z(t)-z(0)].
 \tag{15.11}
\]

This follows by subtracting that member's conservation identity and the
nominal alpha^0 identity. A member-weight description instead needs the
nominal displacement term and the matching member-weight observation dual.
Equation (15.9) is the selected implementation; mixing the two descriptions
while omitting their corrections would be invalid.

### 15.4 Finite cells and noncircular continuation

For each fixed-duration cell supply full nominal node-phase and node-form
boxes, with their independent whole-time association and derivative/remainder
premises. The instrument reconstructs nominal edge sines, node s and q from
these primitive boxes; stored fields or supplied coefficient verdicts cannot
replace them. Inflated phase edges separately supply the finite secants.

Starting from the admitted initial radius, use the shared affine comparison
arithmetic for `R_next <= exp(gamma*h) R + phi(gamma,h) F`, with outward
upper bounds. Divide R and F by the same fixed radius scale when using the
existing normalized scalar representation. All quantities are nonnegative,
so the upper endpoint bounds the entire preceding cell. It also supplies
R_bar for (15.9). Initial membership and every strict gate
`q_theta,e * R_next^2 < h_e^2` prevent the first phase-domain exit exactly
as in section 9. Failure of a sufficient gate leaves the family undecided.

Only after these domain gates pass may the corresponding whole-cell portrait
error be compared with an independently admitted nominal radius. Net-angle
ordering still needs the original shared initial-angle condition and a
nominal gap large enough for both endpoint errors. Capacity uncertainty by
itself supplies neither an ordering nor a retained organization's lifetime.
The previous evaluated connected/cut response has not been re-certified for
a wider state or capacity family by this analytical construction.

The shared [capacity comparison](../../src/tnfr/research/sine_capacity_metric.py)
uses the live existing metric family, at most 21 nodes, 53 edges and 256 cells,
and the same input/intermediate rational limits. It admits the complete
capacity, phase, form, duration and edge-allowance batch before iteration.
Each cell must satisfy the inherited scalar work limit gamma*h<=1; failure
returns an explicit unavailable prefix without refinement, fitting or retry.
Rational boundary limits are not an allocation or wall-time guarantee; an
applied response still requires its own externally monitored budget.

### 15.5 Constructed consequences and evidence scope

On one unit edge take nominal capacities (1,1), member capacities (2,1),
identical initial x=(1,0) and theta=(0,0). Initially s=0 and q=(1,-1), so
R(0)=0 but u_dot(0)=0 and v_dot(0)=(1/pi,0). Hence D^+R(0)=1/pi. The nominal
phase-mean error starts changing at 1/(2*pi); a zero-error constant child has
the opposite phase-offset derivative. This exact witness excludes the old
homogeneous, constant-offset claim while satisfying positive held capacities.

[Instrument controls](../../tests/research/test_sine_capacity_metric.py) and
[independent identities](../../tests/research/test_sine_capacity_metric_identities.py)
check the two additive terms, changed means/offsets, zero-width reduction,
primitive-input admission, strict gates and finite work boundaries on small
constructed cases. They evaluate no retained history, operator event or
target trajectory. The separate source, validation and preservation record
is under `artifacts/held-capacity-contract-20261023/`.

The result is a conditional continuation mechanism for joint state/parameter
uncertainty. Applying it through an actual structural event still requires
that event's complete represented image and common admission, followed by
independent nominal evidence for the actual successor law. The research
program selects that obligation or another global question; this theorem
does not schedule a numerical campaign.
