# Geometry, sensor gain and a declared uncertain clock

<a id="sine-clock-inference"></a>

## Question and conditional result

The [two-input inverse](SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-inference)
constrains an original source angle and held sensor gain under a known
complete law and structural clock. Its
[reserved evaluation](SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-inference-result)
retains that premise. Here the conversion from the observation clock to
structural time is an additional positive, unknown constant. The result is
a necessary outer enclosure of geometry, an effective response scale, and
separate gain/clock marginals. It neither proves that these marginals are
jointly realizable nor determines an absolute physical clock.

Two distinctions matter. An unknown multiplier of **every** evolution row
can be exactly confounded with clock conversion. An unknown sensor gain
can instead compensate the clock in the leading short-time response,
without preserving the full nonlinear response. An explicit curvature
witness below proves that these are different statements.

This is a mathematical extension and implementation contract, not a new
reserved experiment. It changes none of the earlier frozen protocols,
source archives, responses or verdicts. The
[ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization) permits
collective measurement maps while retaining their independent operational
premises; the [parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md#pressure-clock-full-state-closure)
requires a clock change to transform the complete law.

<a id="sine-clock-source-and-law"></a>
## F1-F2: full source, observation clock and complete event law

Retain the eighteen-node support and full source family of the
[two-input owner](SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-source): two
unit C9 rings on `0,...,8` and `9,...,17`, joined at `(0,9)` and `(1,10)`.
Let \(M\) be the degree matrix, \(L\) its graph Laplacian,
\(A=M^{-1}L\), and \(q=e_4-e_5\). The source is
\[
x^-=m_x\mathbf1+u,\qquad
\theta^-=m_\theta\mathbf1+\theta^0(b,c)+v,
\tag{1}
\]
where \(P_Mu=u\), \(P_Mv=v\),
\(\|u\|_M\le X\), \(\|v\|_M\le Y\). The original priors obey
\[
[b_-,b_+]\subseteq[11/8,3/2],\qquad
[c_-,c_+]\subseteq[2/3,1].
\]
The affine phase family, its single weighted centering, continuous lifts
and original acute guard are unchanged. All thirty-six coordinates,
including arbitrary common means and receiver geometry, remain in the
model. An uncertain clock does not authorize a different preparation,
recentered intermediate state or hidden clock-dependent residual.

In fast structural time \(\tau\), retain the complete supplied law
\[
\frac{dx}{d\tau}=-Ax+\gamma f(\theta),\qquad
\frac{d\theta}{d\tau}=\gamma Ax,\qquad
\gamma=\frac1{1023\pi},\qquad
f_i(\theta)=\frac1{M_{ii}}\sum_{j\sim i}\sin(\theta_j-\theta_i).
\tag{2}
\]
Support, unit capacity and both coefficients are held premises of this
normalized model. They are not inferred from the readings.

Let \(s\) be the declared observation clock, with common origin zero, and
\[
\tau=\rho s,\qquad 0<\rho_-\le\rho\le\rho_+<\infty.
\tag{3}
\]
The same constant \(\rho\) applies to both windows. Its units are
structural-time units per observation-time unit. In fixed chosen units the
equations consumed by an observation-clock trajectory are therefore
\[
\frac{dx}{ds}=\rho[-Ax+\gamma f(\theta)],\qquad
\frac{d\theta}{ds}=\rho\gamma Ax.
\tag{4}
\]
Multiplying only the form row, changing \(\gamma\) alone, or treating
\(\rho\) as a measured phase frequency gives a different model.
The [general clock owner](../NODAL_PARAMETER_FOUNDATIONS.md#31-capacity-clock-and-positivity-require-compatible-laws)
also explains why state-dependent and varying conversions require further
laws. They are outside this constant-conversion admission.

Supply cumulative amplitudes \(0<a_1\le a_2\le1\). Apply the phase jump
\(a_1q\) at \(s=0\), evolve (4) until \(s=H>0\), then apply
\((a_2-a_1)q\) and evolve until \(s=2H\). Both jumps leave form
unchanged. There is no other forcing or support event. Their structural
times are \(0,\rho H\); all three observation times are
\(0,\rho H,2\rho H\). The full first endpoint starts the second
window. Both common means remain conserved under these declared rows and
events, irrespective of \(\rho\).

The three recorded scalars use one held sensor:
\[
r_j=Gq^Tx(jH)+O+\eta_j,\quad j=0,1,2,\qquad
0<G_-\le G\le G_+,\quad |\eta_j|\le\delta.
\tag{5}
\]
Here \(x(jH)\) means the solution in the observation clock. Reading
intervals \([r_j^-,r_j^+]\) enclose those recorded values; their exact
midpoints \(c_j\) and half-widths \(\epsilon_j\) differ from the
sensor allowance \(\delta\). One offset \(O\) and one middle reading
are retained. The units of \(G\) are readout units per form unit;
\(J=G\rho\) has the additional clock-conversion factor and is not
itself a calibrated sensor gain. No laboratory identification of \(s\),
the input amplitudes or this affine sensor is supplied.

The gate admits any positive compact clock prior with
\[
h_*:=\rho_+H\le\frac12,\qquad T_*=2h_*\le1.
\tag{6}
\]
The range \([1/2,2]\), for example, is an admissible supplied uncertainty
when \(H\le1/4\); it is not a measured clock range or a universal bound.

<a id="sine-clock-uniform-envelope"></a>
## F3: one uniform envelope for the entire uncertain history

For a structural window length \(0<h\le h_*\), write the earlier
[complete-flow bounds](SINE_TWO_PULSE_INFERENCE.md#f3-a-uniform-finite-error-for-both-full-flow-increments)
as functions of their duration:
\[
\begin{aligned}
g&=1/3069>\gamma,\qquad D=4+4a_2+2Y,\\
Q(h)&=\frac{X+2ghD}{1-16g^2h^2},\\
E(h)&=2hQ(h)+2ghY+8g^2h^2Q(h).
\end{aligned}\tag{7}
\]
The denominator is positive on (6). The numerator is nonnegative and
nondecreasing, while the denominator is positive and nonincreasing.
Thus \(Q(h)\), and consequently
\[
\frac{E(h)}h=2Q(h)+2gY+8g^2hQ(h),
\tag{8}
\]
are nondecreasing. This is a bound on the complete original-source
trajectory, including the first-window form and phase evolution carried
into the second. It assumes neither a reset nor later acuteness.

For the actual \(h=\rho H\), put \(\xi=\rho/\rho_+\in(0,1]\).
Monotonicity gives the useful normalized error bound
\[
\boxed{\ E(\rho H)\le\xi E(h_*)\ }.\tag{9}
\]
The actual form and dynamical phase displacement on both windows obey
\(\|P_Mx\|_M\le Q(h_*)\) and
\(\|d\|_M\le4gh_*Q(h_*)\). In particular, the horizon used to
bound all hidden evolution is the maximum structural horizon, not \(2H\)
unless the chosen units make \(\rho_+=1\).

Let \(y_j=q^T[x(jH)-x((j-1)H)]\). The inherited finite response identity
and (9) imply
\[
y_j=-\xi K_j^*\cos(b-a_j/2)+e_j,\qquad
K_j^*=2\gamma h_*\sin(3a_j/2),\qquad
|e_j|\le\xi E(h_*).
\tag{10}
\]
Consequently, with the auxiliary radial scale
\[
\ell=G\xi=\frac{G\rho}{\rho_+},\qquad
\ell\in\left[\frac{G_-\rho_-}{\rho_+},\ G_+\right],
\tag{11}
\]
the true recorded increments satisfy
\[
r_j-r_{j-1}=-\ell K_j^*\cos(b-a_j/2)
 +\zeta_j+\eta_j-\eta_{j-1},\qquad
|\zeta_j|\le\ell E(h_*)\le G_+E(h_*).
\tag{12}
\]
These are precisely the necessary algebraic inequalities consumed by the
existing two-input enclosure, with duration \(h_*\) and radial prior
(11). This justifies sharing that calculation. It does **not** assert
that the actual readings were produced by another trajectory running at
\(\rho_+\), by sensor gain \(\ell\), or by a newly prepared source.
The auxiliary report is a constraint envelope with that explicit meaning.

For either exact row \(w=(w_1,w_2)\) of the inherited inverse matrix,
the radius remains
\[
G_+E(h_*)(|w_1|+|w_2|)
 +(\epsilon_0+\delta)|w_1|
 +(\epsilon_1+\delta)|w_1-w_2|
 +(\epsilon_2+\delta)|w_2|.
\tag{13}
\]
The common offset cancels in exact midpoint differences first. The
middle reading/error and the factor-based rank checks are unchanged.
The original source acute guard is required to interpret its geometry;
the sufficient whole-window guard at \(h_*\) stays optional.

<a id="sine-clock-marginal-inference"></a>
## F4: necessary geometry, effective scale and quotient marginals

Apply the auxiliary two-input enclosure only after admitting the original
primitives and (3), (6). It returns necessary bounds on \(b\), the
original actual long-arc mean \(B=b-(v_1-v_0)/8\), and \(\ell\).
The actual-angle expansion remains \(Y/8\), using the original phase
radius, independently of the clock uncertainty.

If \(\ell\in[\ell_-,\ell_+]\) survives the auxiliary projection,
then define an outward effective-scale enclosure
\[
J=G\rho\in[J_-,J_+]:=\rho_+[\ell_-,\ell_+].\tag{14}
\]
Every compatible sensor gain and clock belongs to the necessary
positive-product constraint
\[
\mathcal P=\{(G,\rho):G\in[G_-,G_+],
\rho\in[\rho_-,\rho_+],\ G\rho\in[J_-,J_+]\}.
\tag{15}
\]
Its separate coordinate projections are
\[
\begin{aligned}
G&\in[G_-,G_+]\cap[J_-/\rho_+,\ J_+/\rho_-],\\
\rho&\in[\rho_-,\rho_+]\cap[J_-/G_+,\ J_+/G_-].
\end{aligned}\tag{16}
\]
The implementation retains outward bounds and the relation \(J=G\rho\).
It does not fill the rectangle between these projections with compatible
trajectories. Even \(\mathcal P\) discards correlations with geometry,
the complete hidden state and the three errors. Nonempty marginal
intervals prove no joint existence or exact point identification.

A narrow effective scale may coexist with wide gain and clock bounds.
For example, even the exact product \(J=3/2\), combined only with
\(G\in[1,2]\) and \(\rho\in[1/2,2]\), permits the entire
positive-product curve with \(G\in[1,2]\) and
\(\rho\in[3/4,3/2]\). This is an algebraic limitation of this
necessary envelope; it is not an exact full-response collision between
all those pairs. A singleton clock prior recovers the known-clock
calculation at \(h=\rho H\), up to outward projection arithmetic.

Equal amplitudes or unresolved positive matrix factors retain
`unavailable`, distinct from exclusion. A strict necessary exclusion
gives `incompatible`; `bounded_candidate` means only the outer constraints
survive. No clock grid, selected best fit, regenerated response or
cached incoming report supplies the result.

<a id="sine-clock-conditioning"></a>
### A finite resolution regime with a broad clock prior

A conditional arithmetic example shows that uncertain clock conversion
need not erase all geometric information. Choose
\[
\begin{gathered}
\rho\in[1/2,2],\quad G\in[1,2],\quad H=2^{-22},\quad
(a_1,a_2)=(1/4,3/4),\\
X=Y=2^{-40},\qquad \delta=2^{-60},\qquad
\epsilon_j\le\delta.
\end{gathered}
\]
Then \(h_*=2^{-21}\) and the auxiliary radial prior is \([1/4,2]\).
The [earlier conditioning calculation](SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-conditioning)
bounds the Euclidean norm of the two ideal coordinate radii by
\[
R=72000E(h_*)/h_*+144000\delta/h_*<1/3072.
\]
The upper auxiliary gain remains two, so that radius calculation is
unchanged. Its lower gain is now \(1/4\). For compatible readings the
rectangle contains a true vector of norm at least \(1/4\), has diameter
at most \(2R\), and therefore stays at norm at least \(1/4-2R>0\).
It also stays in the first quadrant: the true first coordinate exceeds
\(1/64\) and the second exceeds \(9/40\), both larger than \(2R\).
The norm and argument gradient bounds consequently give
\[
\begin{aligned}
\operatorname{width}(b)&\le\frac{2R}{1/4-2R}<1/383,\\
\operatorname{width}(B)&<1/383+Y/4<1/256,\\
\operatorname{width}(J)&\le4R<1/768<1/512.
\end{aligned}
\]
Prior intersections cannot enlarge those marginals. These are conditional
exact-arithmetic budgets for compatible observations, not evaluated
readings. A numerical report must separately admit outward coefficient
and projection arithmetic. No comparable narrow separate \(G\) or
\(\rho\) guarantee follows, as the product example above demonstrates.

<a id="sine-clock-common-rate-equivalence"></a>
## Exact equivalence with a common multiplier of every row

Consider a separately specified extension of (2): a positive constant
\(\lambda\) multiplies its **entire** vector field. If \(z=(x,\theta)\)
and \(F\) denotes (2), the observation-clock model is
\[
\frac{dz}{ds}=\rho\lambda F(z).\tag{17}
\]
Retain the same full source, support, held sensor \(G,O\), input
amplitudes, and events scheduled at the same observation-clock times.
Any two admissible parameter pairs with
\(\rho_1\lambda_1=\rho_2\lambda_2\) have exactly the same right-hand
side and initial state. Uniqueness gives identical full states up to the
first event; the same jump map gives identical post-event states. Repeating
this argument gives identical full histories and all noiseless readings.
Coupling the same allowed additive errors preserves the recorded readings.

Equivalently, \((\rho,\lambda)\mapsto(k\rho,\lambda/k)\), for
any positive \(k\) admitted by the respective priors, is an exact
equivalence. Where a prior fiber contains multiple pairs, no observation
of these identical state histories can distinguish its individual factors.
The product \(\rho\lambda\) is an invariant of this equivalence;
this proof does not establish that the selected scalar readings identify
even that product. A singleton prior or an additional independent clock
reference can alter the identification problem.

This argument requires the complete-row multiplier, common preparation
and observation-clock event convention. Holding events at fixed structural
times instead generally changes their observed schedule. A changed
relative loss/exchange coefficient, capacity law, source, forcing or
sensor is not covered by (17). The current inference API retains
\(\lambda=1\); it must not interpret its fixed \(\gamma\) as a free
global multiplier.

<a id="sine-clock-gain-curvature-witness"></a>
## Equal leading gain-clock products need not preserve the full response

Here is an exact counterexample inside the fixed-law source domain, with
no numerical trajectory evaluation. Choose
\[
b=\pi/2-1/8,\qquad c=3/4,\qquad u=v=0,
\qquad a_1=1/4.
\tag{18}
\]
The nominal donor short angle is \(4\pi-8b=1\); receiver long angles
are \((2\pi-3/4)/8\) and the port origin difference is \(1/8\).
Together with the stated long donor and short receiver angles these all
lie strictly inside the original acute chart and priors. Common form and
phase means can be arbitrary and equal in both compared preparations.
Any later declared second event is irrelevant to this right-hand local
calculation before \(H\).

Immediately after the first phase jump, write \(f=f(\theta^-+a_1q)\).
The four local rows, all of degree two, satisfy
\[
\begin{aligned}
f_3&=[-\sin b+\sin(b+a_1)]/2=0,\\
f_4&=[-\sin(b+a_1)+\sin(b-2a_1)]/2=-F_0/2,\\
f_5&=-f_4,\qquad f_6=-f_3=0,\\
F_0&=2\sin(3/8)\sin(1/4)>0.
\end{aligned}\tag{19}
\]
Other rows, including both ports and the receiver, retain their actual
values; they are not set to equilibrium. Locality of \(A\) gives
\[
q^Tf=-F_0,\qquad q^TAf=-3F_0/2.\tag{20}
\]
Since the initial form is constant, \(Ax(0)=0\), and the complete
structural rows imply
\(x_\tau(0^+)=\gamma f\), \(\theta_\tau(0^+)=0\), and
\(x_{\tau\tau}(0^+)=-\gamma Af\). For the noiseless held readout
\(r(s)=Gq^Tx(s)+O\), therefore
\[
r'(0^+)=-G\rho\gamma F_0,\qquad
r''(0^+)=\tfrac32G\rho^2\gamma F_0.
\tag{21}
\]
These are derivatives on the actual complete flow after the event, not
derivatives of a reduced fitted response or of the discrete sensor errors.

Compare \((G,\rho)=(3/2,1)\) with \((1,3/2)\), using exactly the
same source, first jump and offset. Both pairs fit the priors
\(G\in[1,2]\), \(\rho\in[1/2,2]\), have the same initial reading
and slope, and satisfy \(G\rho=3/2\). The second pair's curvature exceeds
the first by \(9\gamma F_0/8>0\). Its readout minus the first consequently has leading
term \((9\gamma F_0/16)s^2\) as \(s\downarrow0\), so it is nonzero
for every sufficiently small positive time before the second event.
The product-preserving gain/clock change is not an exact response symmetry.

For this exact preparation with exact continuous derivative access,
\(-2r''(0^+)/[3r'(0^+)]=\rho\). That observation is conditional on
the known geometry and vanishing form residual; it does not provide a
uniform clock estimator for uncertain sources or three noisy samples.
Unknown initial form, phase residuals and later phase feedback enter the
general curvature. The witness rules out an overly strong impossibility
claim; it does not prove complete finite-record identifiability.

<a id="sine-clock-implementation-boundary"></a>
## Implementation, evidence and continuation boundary

The [clock inverse](../../src/tnfr/physics/relational_sine_clock_inference.py)
admits the observation duration, clock/gain priors and the original
source/reading primitives, then constructs the auxiliary constraint
envelope through the existing two-input owner. Its report distinguishes
the observed clock, maximum structural horizon, normalized radial scale,
effective response scale, sensor-gain marginal and clock marginal.
Exact rational endpoint arithmetic precedes interval materialization;
an extremely small positive clock is not replaced by a zero denominator.
The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-clock-inference)
owns the executable domains and availability rules, and the
[tests](../../tests/physics/test_sine_clock_inference.py) check independent
full-state responses and analytic controls rather than reevaluating any
reserved source.

This owner extends one fixed nonlinear source/measurement problem. The
earlier [bridge clock/law discriminator](RESONANCE_FOUNDATIONS.md#finite-bridge-clock-law-discrimination)
uses different preparations and observations, and the
[mode/pole identification](RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-memory-identification)
uses its own modal closure and measurement premises. Their methods motivate
careful separation of rate and readout, but their coefficients or verdicts
do not transfer here.

The new enclosure can lose resolution while remaining sound. It does not
claim that gain/clock ambiguity is fundamental to every observation, that
the exact common-rate equivalence selects a physical law, or that event
funding, formation, recovery or maintenance follows from an inferred
geometric interval. A physical clock and gain need their own measurement
bridge. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns admission of further derivative bounds or reserved responses.
