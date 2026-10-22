# Retention mechanisms and the complete-law envelope boundary

**Scope:** mechanism admission and one local finite-input variational
evaluation under the existing positive-loss cycle law. Exact balances and
cubic coefficient identities are proved below;
envelope profiles solve an auxiliary modulation equation. A full-state
residual theorem and a relative-storage comparison specify how to validate
such a reference against the actual law, retaining near resonances. The
first correction panel is evaluated in section 17; the full horizon,
nonlinear envelope maintenance and interaction-generated composites
remain unproved. The original source and
[strict-passage result](LOCALIZED_ORGANIZATION_GEOMETRY.md#mobile-relative-rate-obstruction)
remain unchanged. The [research program](../RESEARCH_PROGRAM.md#research-current-question)
alone selects further work and owns reactivation of the
[deferred variational dependency](../RESEARCH_PROGRAM.md#research-deferred-dependencies).

## 1. Retained law and internal relations

Keep the [complete two-row model](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-law):
signed real form x, a continuous real phase lift theta, unit cycle
conductances and lengths, held unit capacities, beta one, no input or events,
and the clock `tau=w*t/pi`. For `e,w>0`, `sigma=pi*e/w`, `A=L/2`,

\[
 x'=-\sigma Ax-\tfrac12D^T\sin(D\theta),\qquad
 \theta'=Ax.                                                     \tag{1}
\]

Writing `z=x+i*theta` retains two real channels. It gives

\[
 z'=F_\sigma(z)=iAz-\sigma A\operatorname{Re}z+R(\operatorname{Im}z),
 \quad R(\theta)=-\tfrac12D^T[\sin(D\theta)-D\theta].             \tag{2}
\]

This is neither complex EPI nor an identification with a quantum state.
The actual positive-loss tangent `J_sigma` has the existing contractive
Euclidean semigroup. A proposed envelope is an approximate full-state
history, not an invariant subspace or a replacement for either row.

### Supplied symmetry does not establish acquisition

Reflection `Jz(j)=z(-j)` commutes with (1). The original nominal joint
preparation is reflection symmetric, so that relation is preserved exactly.
The same-source linear control also preserves it. The relation is supplied
initially and does not discriminate a newly formed organization.

For primitive errors of norm at most epsilon each, the initial antisymmetric
part has norm at most `2*epsilon`. Within the original joint norm bootstrap
`||z||,||Jz||<2a`, the shared derivative estimate `||DR||<=4S^2` with `S=2a`
gives

\[
 \left\|\frac{z(\tau)-Jz(\tau)}2\right\|
 \le 2\epsilon\exp(16a^2\tau).                                  \tag{3}
\]

This robust retention is still retention of a prepared symmetry. It cannot
replace initial absence and discriminating acquisition in the
[formation contract](LOCALIZED_ORGANIZATION_GEOMETRY.md#common-pattern-formation-contract).

The [exact period-four family](LOCALIZED_ORGANIZATION_GEOMETRY.md#4-an-exact-carrier-geometry-without-a-localized-envelope-closure)
reduces to independent damped pendula u and v. With
`E_u=u'^2/2+1-cos(u)` and similarly for v,

\[
 H=\frac N2(E_u+E_v),\qquad E_u'=-\sigma u'^2,\quad
 E_v'=-\sigma v'^2.                                               \tag{4}
\]

There is no transfer term between these two modal energies. For any pendulum
solution q and nonnegative time shift delta, `u(tau)=q(tau)` and
`v(tau)=q(tau+delta)` give an exact pair. Such a relative registration can be
supplied; these equations do not select it through mutual exchange. Decay
toward zero also destroys an oscillation's phase observability. Localized
modulation and its boundary cannot be removed by invoking this reduction.

<a id="cycle-local-storage-balance"></a>
## 2. What can accumulate locally under this law?

The [regional balance owner](FORCED_REGIONAL_RESPONSE.md#7-a-region-and-its-environment-on-the-same-nodal-support)
requires boundary exchange. Here an exact cycle-specific expression makes
the consumed phase information explicit. Orient cycle edges consistently,
put `y=Dx`, `u=Dtheta`, and let edge indices j and k be neighbors on the edge
cycle. Their rows are

\[
 y'=-\tfrac\sigma2 L_e y-\tfrac12L_e\sin u,\qquad
 u'=\tfrac12L_e y,\qquad L_e=DD^T.                               \tag{5}
\]

For `h_j=y_j^2/2+1-cos(u_j)`, define

\[
 F_{jk}=\tfrac12(\sin u_j\,y_k-y_j\sin u_k)
           +\tfrac\sigma4(y_j^2-y_k^2)=-F_{kj}.
\]
\[
 h_j'=-\sum_{k\sim j}F_{jk}
          -\tfrac\sigma4\sum_{k\sim j}(y_j-y_k)^2.               \tag{6}
\]

Direct differentiation of h and substitution of (5) prove (6). Summing over
all edges recovers `H'=-(sigma/2)||Lx||^2`. On a fixed subset, internal
currents cancel; an increase requires inward boundary current to exceed
local dissipation. Form gaps and sine phase gaps determine that current,
so intensities or centers alone cannot infer it. Positive total loss does
not exclude transient regional accumulation. Conversely, positive inflow
does not prove retention or select a boundary. Moving membership requires
its separate transport accounting; (6) is for fixed sets.

<a id="cubic-envelope-candidate"></a>
## 3. The cubic balance selects a different candidate regime

Let an on-grid carrier satisfy `kN in 2*pi*Z`, and write

\[
 \lambda=1-\cos k,\quad v=\sin k,\quad d=\tfrac12\cos k,
 \quad g=\lambda^2/8,\quad
 h_3=\frac{\sin^3(k/2)\sin(3k/2)}6,\quad
 \lambda_3=1-\cos(3k).                                         \tag{7}
\]

Use a positive bookkeeping scale varepsilon, distinct from the original source
error epsilon, and slow variables
`X=varepsilon*(j+v*tau)`, `S=varepsilon^2*tau`, `phi=kj+lambda*tau`.
The candidate carrier travels with site velocity `-v`.

### Exact cubic harmonics and a two-channel reconstruction

The cubic part of R is `R_3(theta)=D^T(Dtheta)^3/12`. At locally constant B,
substituting `theta=Im(B*exp(i*phi))` gives exactly

\[
 R_3=-ig|B|^2B e^{i\phi}+ig|B|^2\overline B e^{-i\phi}
          -ih_3B^3e^{3i\phi}+ih_3\overline B^3e^{-3i\phi}.       \tag{8}
\]

For example, a forward phase gap has coefficient
`(exp(ik)-1)B/(2i)`; cubing and applying the backward incidence row gives
all four terms. Keeping only the resonant first term would discard an
actual part of the form row.

Cancellation through cubic order uses the auxiliary equation

\[
 iB_S=d B_{XX}+g|B|^2B                                           \tag{9}
\]

and the full reconstruction

\[
 z_{\rm app}=\varepsilon B e^{i\phi}+\varepsilon^3 C,
 \quad C=a_{-1}|B|^2\overline B e^{-i\phi}
                 +a_3B^3e^{3i\phi}+a_{-3}\overline B^3e^{-3i\phi},
                                                                    \tag{10}
\]
\[
 a_{-1}=-\lambda/16,\qquad
 a_3=-\frac{h_3}{3\lambda-\lambda_3},\qquad
 a_{-3}=-\frac{h_3}{3\lambda+\lambda_3}.                         \tag{11}
\]

Require `lambda>0` and nonzero displayed denominators. The operator
`partial_tau-iA` multiplies a constant harmonic `exp(i*l*phi)` by
`i*(l*lambda-lambda(l*k))`; (11) cancels the three nonresonant terms of (8).
Envelope derivative errors remain and are bounded in section 5. Both real
coordinates are reconstructed as `x_app=Re(z_app)`,
`theta_app=Im(z_app)`; neither row is declared exact by (9).

The carrier `k=pi/4` is already mode 128 of the unchanged `C_1024` support.
It has positive curvature and no third-harmonic resonance:

\[
 \lambda=1-\frac{\sqrt2}2,\quad d=\frac{\sqrt2}4,\quad
 g=\frac{3-2\sqrt2}{16},\quad h_3=\frac{\sqrt2-1}{48},
\]
\[
 (a_{-1},a_3,a_{-3})=
 \left(-\frac{2-\sqrt2}{32},\frac1{96},
                  -\frac{3\sqrt2-2}{672}\right).                \tag{12}
\]

This is an analytical candidate, not a searched optimum. For general k,
third-harmonic resonance and spatial aliases must be checked; spatially
aliased terms can still have distinct temporal frequencies.

### The actual damping does not disappear

Equation (9) is a conservative comparison equation. All actual damping can
be retained in the residual of (10), as in section 5. Alternatively, writing
`gamma=sigma/varepsilon^2` yields the leading damped modulation

\[
 iB_S=dB_{XX}+g|B|^2B-i\frac{\gamma\lambda}{2}B.                  \tag{13}
\]

Its counterrotating damping corrector is
`-i*gamma*conj(B)*exp(-i*phi)/4`. This follows by multiplying its coefficient
by `-2i*lambda`, giving the omitted damping coefficient
`-gamma*lambda*conj(B)/2`. For a periodic envelope,
`d/dS integral |B|^2=-gamma*lambda*integral |B|^2` under (13).
Fixed positive sigma therefore cannot be suppressed in a
`varepsilon -> 0` limit or replaced with zero to certify a desired profile.
The residual bound below applies to the conservative reference (9)--(11);
using (13) and its additional corrector requires redoing that bound.

<a id="envelope-retention-and-preparation"></a>
## 4. Which relation can the auxiliary balance maintain?

For `d,g>0`, (9) admits the exact line profile

\[
 B(X,S)=\rho\operatorname{sech}(X/L)e^{-i\Omega S},\qquad
 \rho^2L^2=\frac{2d}{g}=\frac{8\cos k}{\lambda^2},\qquad
 \Omega=\frac d{L^2}=\frac{g\rho^2}{2}.                         \tag{14}
\]

This follows from `sech''=sech-2*sech^3`. The relation locks the envelope
shape, height, width and slow internal phase in the comparison equation;
the corresponding carrier frequency is `lambda-varepsilon^2*Omega`.
It is not a freely chosen stationary shape of the original nodal law.

The sign restriction is substantive. For a smooth real decaying profile f,
`d*f''+g*f^3=Omega*f` has first integral
`d*f'^2/2=Omega*f^2/2-g*f^4/4`. Exponential decay requires `Omega/d>0`,
and a nonzero turning amplitude requires `Omega>0`. Thus this bright-profile
balance requires `d>0`. At the original `k=pi/2`, d vanishes and this
quadratic-dispersion mechanism is unavailable. This does not exclude every
higher-order modulation or nonperturbative mechanism at that carrier.

### Periodic references and the source scale

The line sech cannot be sampled across a cycle seam and treated as a smooth
periodic solution. An exact periodic alternative of (9), with kappa and rho
strictly positive, is defined by

\[
 B=f(X)e^{-i\Omega S},\quad
 f''=\kappa^2(2-m)f-\frac{2\kappa^2}{\rho^2}f^3,\quad
 f(0)=\rho,\quad f'(0)=0,
\]
\[
 0<m<1,\quad \rho^2=2d\kappa^2/g,\quad
 \Omega=d\kappa^2(2-m),\quad
 (f')^2=\frac{\kappa^2}{\rho^2}
                   (\rho^2-f^2)(f^2-\rho^2(1-m)).               \tag{15}
\]

The solution oscillates smoothly between `rho*sqrt(1-m)` and rho. Separation
of variables gives period `2K(m)/kappa`, where
`K(m)=integral_0^(pi/2) (1-m*sin(u)^2)^(-1/2) du`.
Choose that period to equal `varepsilon*N`; this supplies a genuinely
periodic reference with one maximum per cycle. Its trough, concentration,
derivative bounds and error allowance still need a quantitative admission.
No value of m, width, horizon or source tolerance has been evaluated here.
For fixed N and m, that period condition fixes the site-scale height
`varepsilon*rho=2*K(m)*sqrt(2*d/g)/N` and inverse width
`varepsilon*kappa=2*K(m)/N`. Changing varepsilon alone only reparametrizes
this reference; the derivative bounds change with it. It is not a free
parameter for reducing the physical residual on fixed support.

The line-profile scale already explains why the original small preparation
cannot simply be renamed. Write physical site width `ell=L/varepsilon` and
height `b=varepsilon*rho`; (14) implies

\[
 b\ell=K_*:=\frac{\sqrt{8\cos k}}{\lambda},\qquad K_*>8
                  \quad\hbox{at }k=\pi/4.                       \tag{16}
\]

For the literal sampled sech with integer center and `1<=ell<=N/4`, at
least ell sites have distance at most ell. Since `sech(1)^2>1/4`, its norm
squared exceeds `b^2*ell/4>=K_*^2/N`. On `C_1024` its norm therefore exceeds
`1/4`, whereas the entire original joint tube has norm below `1/2048`.
This is a scale incompatibility of this particular reference; it is not a
theorem excluding all small nonlinear localized states. The full corrected
periodic reference must have its own source admission.

### A single retained envelope is not capture of two incoming patterns

For patterns to remain within a common width `O(varepsilon^-1)` over a
slow interval `O(varepsilon^-2)`, their leading group velocities must differ
by at most `O(varepsilon)`, unless a different, nonperturbative mechanism
changes them. Near pi/4, nonzero cosine makes this require carrier
separation `O(varepsilon)`. Replacing the original carriers with `+/-pi/4`
still leaves velocity difference `sqrt(2)`: focusing alone does not capture
that counterpropagating pair. Distant carriers can share a velocity; they
require a separately derived multiple-envelope system. The
[equal-velocity carrier control](LOCALIZED_FORMATION_COMPATIBILITY.md#equal-velocity-carrier-control)
derives the local cubic channels for pi/4 and 3*pi/4, retaining temporal
labels within spatial aliases. Its modulation restrictions do not provide
a full-law formation or maintenance certificate.

A prepared profile satisfying (14) or (15) already contains its phase/shape
relation. Even successful full-law maintenance would not establish its
acquisition from two initially independent patterns. That claim additionally
needs initial absence, a compatible relative preparation, retained boundary
state, entry and a discriminating continued response under the same law.

<a id="full-law-envelope-residual"></a>
## 5. An explicit conditional bridge to the complete law

Here is a sufficient finite-time validation method, rather than an appeal to
a formal asymptotic symbol. Let B solve (9), be periodic in X with period
`varepsilon*N`, and have continuous bounded derivatives through order three
in X and order one in S on the full proposed slow interval. Define uniform
bounds `b_j>=sup|partial_X^j B|`, `j=0,...,3`, and
`b_S=|d|*b_2+g*b_0^3>=sup|B_S|`. All bounds include intervening X values,
not just graph samples. Put

\[
 A_0=|a_{-1}|+|a_3|+|a_{-3}|,\qquad
 A_1=\sum_{l\in\{-1,3,-3\}}|a_l|\,|v-\sin(lk)|,
 \quad D_k=|1-e^{ik}|,
\]
\[
 G=\varepsilon(D_kb_0+\varepsilon b_1)
                      +2\varepsilon^3 A_0b_0^3.                \tag{17}
\]

Every phase gap of (10) has magnitude at most G. For
`r_app=z_app'-F_sigma(z_app)`, a uniform per-site bound is

\[
\begin{aligned}
 r_*={}&\varepsilon^4\left[
  \frac{b_3}{6}+3A_1b_0^2b_1
          +\frac{b_1}{2}(D_kb_0+\varepsilon b_1)^2\right]\\
 &+\varepsilon^5 A_0\left[3b_0^2b_S+3b_0b_1^2
                                      +\frac32b_0^2b_2\right]\\
 &+\varepsilon^3 A_0b_0^3G^2+\frac{G^5}{120}
             +2\sigma(\varepsilon b_0+\varepsilon^3 A_0b_0^3),
 \qquad \|r_{\rm app}\|_2\le\sqrt N\,r_* .                   \tag{18}
\end{aligned}
\]

The terms have separate derivations:

- Taylor expansion of the leading nearest-neighbor row leaves
  `varepsilon^4*b_3/6` after (9). Corrector transport differs from the
  leading group velocity and is bounded using
  `|partial_X F_l(B)|<=3*b_0^2*b_1`.
- Freezing B in the cubic phase gaps changes each gap by at most
  `varepsilon^2*b_1`. The cubic difference inequality supplies the last
  term in the first bracket.
- The corrector's slow derivative and second spatial remainder use
  `|partial_S F_l|<=3*b_0^2*b_S` and
  `|partial_X^2 F_l|<=6*b_0*b_1^2+3*b_0^2*b_2`.
- Its addition changes each phase gap by at most
  `2*varepsilon^3*A_0*b_0^3`, giving the cubic correction term. The global
  Taylor inequality `|sin(u)-u+u^3/6|<=|u|^5/120` gives the fifth-order term.
- The final term bounds the entire actual damping row, using the
  nearest-neighbor bound `|A Re(z_app)|<=2*sup|z_app|`.

The Euclidean `sqrt(N)` factor is retained; localization or cancellations
cannot silently replace it. The residual is a pair of real-row errors,
including the phase row, not a fitted effective pressure.

Let the initial full-state error be at most `delta>=0` and propose a strictly
positive error radius r and horizon T over `0<=tau<=T`.
In that radius every actual and intervening phase
gap is bounded by `G+2r`. Since

\[
 DR(\theta)=\tfrac12D^T\operatorname{diag}(1-\cos D\theta)D,
 \qquad \|DR\|\le(G+2r)^2=:L_*,                               \tag{19}
\]

variation of constants under the actual contractive `J_sigma` and Gronwall
give the sufficient bootstrap condition

\[
 E_*:=e^{L_*T}\bigl(\delta+T\sqrt N\,r_*\bigr)<r.              \tag{20}
\]

A first exit at norm r contradicts this estimate, so the full-state error
stays at most E_* throughout the interval. A required acute phase chart can
be admitted separately by `G+2r<pi/2`. Reference norm and concentration
floors must exceed their error allowances before any retained identity is
certified. If (20) fails, this sufficient method is inconclusive; no
nonexistence or dynamical instability follows.

This is a conditional theorem with explicit finite constants. It has not
been instantiated for a prepared envelope. In particular, at
`T=S_*/varepsilon^2`, the fixed positive sigma contribution must still be
evaluated. A formal small residual or a profile of (9) alone cannot certify
that window. Using localization-aware norms or a damped reference may
improve the method, but each needs its own proved bound.

<a id="acute-envelope-certificate-obstruction"></a>
### A scoped obstruction to this coarse certificate

For the proposed periodic family (15), one can decide a limitation without
selecting a profile or evaluating a response. Fix the actual original law
coefficients `e=2^-20`, `w=1-e`, so `sigma=pi/(2^20-1)`, retain `N=1024`,
and take `k=pi/4`. Put `b=varepsilon*rho` and define one slow internal-phase
time by

\[
 T_\phi=\frac1{\varepsilon^2\Omega}
              =\frac2{g b^2(2-m)}.                             \tag{21}
\]

Consider the particular certificate requiring both (20) and the acute-chart
condition `G+2r<pi/2`, with `r>0`. Since
`1/4<lambda<1/3`, `g<1/72`, `D_k>1/2` and `pi<4`,
this condition forces `r<1` and `b<4`. Consequently

\[
 T_\phi b>\frac1{g b}>18,\qquad
 T_\phi\sqrt N\,r_*>1152\sigma
                 >\frac{3456}{2^{20}}>\frac1{400}.              \tag{22}
\]

Here the second inequality uses the actual positive damping term
`r_*>=2*sigma*b`; b_0 is at least the profile maximum rho. Also

\[
 L_*T_\phi\ge G^2T_\phi
       \ge\frac{32}{\lambda(2-m)}>48.                           \tag{23}
\]

Even at zero initial error, the published sufficient majorant therefore obeys

\[
 E_*>\frac{1+48+48^2/2}{400}
        =\frac{1201}{400}>1>r,                                  \tag{24}
\]

using `exp(u)>=1+u+u^2/2` for nonnegative u. The contradiction persists for
any `T>=T_phi`. Thus **this coarse acute-chart certificate cannot validate
the periodic conservative reference for a full nonlinear phase time**.
Changing width, refining source error or increasing arithmetic precision
cannot repair this bound within the stated class.

This is a theorem about a sufficient method, not an obstruction to actual
nonlinear maintenance. It does not exclude (20) without the additional acute
condition, shorter intervals, other reference histories or other error
estimates. A structure-sensitive stability argument or an explicitly damped
reference with its own residual would change the method's premises. The
present calculation does not establish that either alternative succeeds.

The existing [moving-reference relative-storage identity](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window)
is a reusable method for retaining reactive cancellations. Its earlier
conservative graph, exact reference and quantitative constants do not
transfer here. Positive loss, two-row reference residuals, moving phase
curvature and the uncontrolled common means must all be re-admitted before
using such storage to bound this full-state error. Relative storage is not
automatically decreasing along a moving reference.

<a id="complete-law-relative-storage"></a>
## 6. Relative storage retains the complete error balance

Let `(x_*,theta_*)` be a differentiable, independently specified reference
on the same cycle and clock, with residuals in **both** actual rows:

\[
 r_x=x_*'+\sigma Ax_*+\tfrac12\nabla V(\theta_*),\qquad
 r_\theta=\theta_*'-Ax_*,\qquad
 V(\theta)=\sum_e[1-\cos(D\theta)_e].                            \tag{25}
\]

Use matched continuous phase lifts, `u=x-x_*`, `eta=theta-theta_*`,
`a=Dtheta_*`, `b_e=(Deta)_e`, and define

\[
 \mathcal E=\tfrac12u^TLu+V(\theta_*+\eta)-V(\theta_*)
                         -\nabla V(\theta_*)^T\eta.
\]

Direct differentiation and substitution of both rows give the exact identity

\[
\begin{split}
 \mathcal E'={}&-\tfrac\sigma2\|Lu\|^2
 +\sum_e[\sin(a_e+b_e)-\sin a_e-\cos a_e\,b_e]a_e'\\
 &-u^TLr_x-[\nabla V(\theta_*+\eta)-\nabla V(\theta_*)]^T r_\theta.
                                                               \tag{26}
\end{split}
\]

The form/phase exchange cancels. The moving-curvature term does not cancel
and has no fixed sign. This extends the earlier
[moving-pattern identity](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window)
to the actual positive loss and an approximate reference; it does not
transfer that owner's graph, exact reference or constants.

### Means, coercivity and the retained coordinates

Let `P` project off the constant nodal vector. All inverse powers of L below
act only on its mean-zero subspace. Put

\[
 y=(L^{1/2}u,L^{1/2}\eta),\quad U=DL^{-1/2},\quad U^TU=I,
 \quad \ell_1=4\sin^2(\pi/N).
\]

Suppose `||y||<=rho` and `||a||_infinity+rho<=chi<pi/2`. With
`kappa=cos(chi)>0`, Taylor's integral formula gives

\[
 \tfrac\kappa2\|y\|^2\le\mathcal E\le\tfrac12\|y\|^2,
 \qquad
 \|(u,\eta)\|^2\le\ell_1^{-1}\|y\|^2
                       +N(\bar u^2+\bar\eta^2).                \tag{27}
\]

The missing means have exact independent equations
`bar(u)'=-bar(r_x)`, `bar(eta)'=-bar(r_theta)`. They need their initial
values and residual integrals. Constant form or phase errors have zero
relative storage; neither a full-state estimate nor a zero-mean residual
may be inferred from storage alone. The factor `ell_1^-1` is the actual
finite-cycle coercivity cost, not a carrier-band approximation.

Taylor expansion of the moving term in (26) gives

\[
 \mathcal E'=-\tfrac\sigma2\|Lu\|^2+y^TM(\tau)y
                  +\mathcal R_3+\mathcal W_r,\qquad
 M=\begin{pmatrix}0&0\\0&-\tfrac12U^T\operatorname{diag}(\sin a\,a')U\end{pmatrix},
                                                               \tag{28}
\]
\[
 |\mathcal R_3|\le\frac{\|a'\|_\infty\rho}{6}\|y\|^2,
 \qquad |\mathcal W_r|\le\|y\|\mathfrak r,\qquad
 \mathfrak r=\|(L^{1/2}r_x,L^{1/2}r_\theta)\|.
\]

For the second bound, write the sine difference as `t_e b_e`, where
`t_e=int_0^1 cos(a_e+s*b_e) ds` and `|t_e|<=1`, and apply Cauchy--Schwarz
to the two residual rows together. No pressure is reconstructed from a
response and no error mode is discarded.

<a id="near-resonant-relative-storage"></a>
## 7. A conditional comparison that retains near resonances

There is a precise alternative to taking the absolute value of all of M
before integration. Define the skew principal tangent on the mean-zero
coordinates by

\[
 J_0=\begin{pmatrix}0&-A\\A&0\end{pmatrix}.
\]

This is a comparison operator inside the proof. Actual damping stays in
(26) and in the error equation; the physical law is still (1).

For the reconstruction (10), its leading edge wave can be written
`a_0=Im(c(tau)*exp(i*lambda*tau))`, where c is the complex edge amplitude
obtained by applying D to `varepsilon*B(X,S)*exp(i*k*j)`.
Its coefficients vary with the envelope. With `omega=2*lambda`, set

\[
 H=\begin{pmatrix}0&0\\0&\frac{i\lambda}{4}
                 U^T\operatorname{diag}(c^2)U\end{pmatrix},\qquad
 M_{\rm fast}=\operatorname{Re}(e^{i\omega\tau}H).              \tag{29}
\]

Indeed `-(a_0*a'_{0,fast})/2=Re(i*lambda*c^2*exp(2i*lambda*tau)/4)`.
All envelope derivatives, generated harmonics and the difference between
sine and its argument remain in `M-M_fast`. This is an exact decomposition
of (28), not a declaration that its remainder vanishes.

### The finite-dimensional homological equation

Choose an orthonormal complex eigenbasis `J_0 v_alpha=i*mu_alpha*v_alpha`,
with all nonconstant cycle modes and both frequency signs. For a complex
symmetric matrix H use the congruence coefficients
`Hhat_alpha,beta=v_alpha^T H v_beta`. The transpose is intentional: these
are quadratic forms, not Hermitian expectation values. The congruence
preserves the Frobenius norm. On these coefficients the operator
`i*omega*C+J_0^T*C+C*J_0` has divisor
`i*(omega+mu_alpha+mu_beta)`.

Fix a positive threshold `Delta` before any bound evaluation. Retain every
entry with `|omega+mu_alpha+mu_beta|<=Delta` in `H_near`, including zero
divisors. Let `H_far=H-H_near`, and define the symmetric matrix C by

\[
 \widehat C_{\alpha\beta}=
 \begin{cases}
 -2\widehat H_{\alpha\beta}/[i(\omega+\mu_\alpha+\mu_\beta)],
       &|\omega+\mu_\alpha+\mu_\beta|>\Delta,\\
 0,&\text{otherwise}.
 \end{cases}                                                   \tag{30}
\]

Reconstruct C by inverse congruence and set `K=Re(exp(i*omega*tau)*C)`.
The matrices K and `Re(exp(i*omega*tau)*C')` are real symmetric. Direct
differentiation, with the same fixed spectral split, proves

\[
 K'+J_0^TK+KJ_0=-2\operatorname{Re}(e^{i\omega\tau}H_{\rm far})
                       +\operatorname{Re}(e^{i\omega\tau}C').   \tag{31}
\]

In particular, conservative sufficient bounds are

\[
 \|K\|_2\le 2\|H_{\rm far}\|_F/\Delta,\qquad
 \|\operatorname{Re}(e^{i\omega\tau}C')\|_2
                            \le2\|H'_{\rm far}\|_F/\Delta.     \tag{32}
\]

Actual operator-norm bounds may be sharper. A changing cutoff would add
terms not covered by (31). The split does not delete perturbations: near
entries remain in
`M_ret=M-M_fast+Re(exp(i*omega*tau)*H_near)`.

### Positive metric, actual loss and a full-state bootstrap

On a prospective interval admit uniform bounds

\[
 \|K\|_2\le k_*<\kappa,\quad \|M_{\rm ret}\|_2\le m_*,\quad
 \|\operatorname{Re}(e^{i\omega\tau}C')\|_2\le d_*,\quad
 \|a'\|_\infty\le a_*.
\]

Define `E_c=E+y^T K y/2`. Equations (27) and (31) imply

\[
 \frac{\kappa-k_*}{2}\|y\|^2\le E_c
                       \le\frac{1+k_*}{2}\|y\|^2.             \tag{33}
\]

For completeness, the exact error equation in these coordinates is
`y'=J_0*y+f_sigma+f_nl-rhat`, with
`||f_sigma||<=2*sigma*||y||` and `||f_nl||<=chi^2*||y||`.
The latter follows from
`sin(a+b)-sin(a)-b=(int_0^1 cos(a+s*b) ds-1)*b`,
`||L^(1/2)D^T||=4`, and `1-cos(q)<=chi^2/2` in the admitted chart.
Here `rhat=(L^(1/2)r_x,L^(1/2)r_theta)`; it is not an additional forcing law.

Substitution in the derivative of E_c proves

\[
 E_c'\le-\tfrac\sigma2\|Lu\|^2+c_*\|y\|^2
                  +(1+k_*)\|y\|\mathfrak r,\qquad
 c_*=m_*+\tfrac12d_*+k_*(2\sigma+\chi^2)+a_*\rho/6.             \tag{34}
\]

Thus `q=sqrt(2*E_c)` has the comparison majorant

\[
 q(\tau)\le Q(\tau):=e^{\alpha\tau}
 \left[\sqrt{1+k_*}\|y(0)\|
 +\frac{1+k_*}{\sqrt{\kappa-k_*}}
                 \int_0^\tau e^{-\alpha s}\mathfrak r(s)\,ds\right],
 \qquad \alpha=\frac{c_*}{\kappa-k_*}.                          \tag{35}
\]

At zeros of E_c use a regularized square root and pass to the limit.
If the initial chart holds and `Q(T)<rho*sqrt(kappa-k_*)`, a first exit
from `||y||<rho` contradicts (33)--(35). Equations (27) and the independently
integrated means then control the **complete** nodal error. This is a
conditional theorem; every bound is uniform over the declared interval and
source ball. Small residuals alone do not supply `k_*<kappa` or that bootstrap.

Actual positive damping has not been removed: (34) retains its negative
storage production and bounds its interaction with K explicitly. A better
use of the negative term requires an additional proved estimate. A damped
envelope reference also needs its own residual under (25). Neither is
supplied by selecting a cutoff. The bound improves on (20) only when its
admitted constants and full-state margins demonstrate that improvement.

<a id="carrier-resonance-and-small-divisors"></a>
## 8. Which curvature terms cannot be averaged away?

Two exact coefficient calculations delimit the preceding construction.
They concern the **frozen leading carrier** in the homological equation,
not a solved localized reference. They are not trajectory observations.

### The carrier block has a nonzero resonant source

Temporarily write the leading reference as
`theta_*=b*sin(phi_j)`, `phi_j=k*j+lambda*tau`, with real amplitude b,
and take a same-carrier error
`z-z_*=(alpha+i*beta)*exp(i*phi_j)`. This complex notation still represents
two real errors. Exact spatial summation on C1024 at `k=pi/4` gives

\[
 \mathcal E_0=N\lambda(\alpha^2+\beta^2),\qquad
 \left\langle-\tfrac12\sum_e a_ea_e' b_e^2\right\rangle
                    =-\tfrac N2\lambda^3 b^2\alpha\beta.       \tag{36}
\]

The brackets denote the principal fast average, with frozen error
quadratures, not averaging an evaluated nonlinear response. The term can
have either sign. Its homological divisor is zero:
`2*lambda-lambda-lambda=0`. A bounded periodic quadratic correction cannot
have a nonzero constant derivative along this periodic principal motion.
Therefore this source cannot be removed by such a correction.

Its maximum magnitude relative to E_0 is `2*g*b^2`. In the conservative
auxiliary envelope's carrier-rotating frame, linearization about constant
amplitude gives `alpha'=0`, `beta'=-2*g*b^2*alpha` in the fast clock at this
order: amplitude error changes the phase rate. The resulting shear has
linear-in-time entries, not an exponential-instability proof. Over the
formal phase time (21), the Euclidean logarithmic-norm bound for this
auxiliary shear has exponent `g*b^2*T_phi=2/(2-m)`, of order one.
This coefficient calculation does not
bound all other modes or certify the localized envelope. In particular,
the actual loss, spatial modulation and residual remain in (34)--(35).

### Adjacent modes invalidate an exact-resonances-only shortcut

For distinct modes `q=k+h`, `r=k-h`, the principal divisor is exactly

\[
 \delta_h=\lambda(q)+\lambda(r)-2\lambda
                    =2\cos k(1-\cos h),\qquad
 h=2\pi/N.                                                      \tag{37}
\]

It is nonzero but small. Use unitary Fourier coefficients `z_q,z_r`,
`d_q` the incidence symbol with `|d_q|^2=2*lambda(q)`, and
`A_b=|d_k|*b`. Put
`P_qr=d_q*d_r*z_q*z_r*exp(-2i*(lambda*tau+arg(d_k)))`.
The principal moving production on this pair is
`-A_b^2*lambda*Im(P_qr)/8`; along J_0, `P_qr'=i*delta_h*P_qr`.
Consequently its compulsory zero-homogeneous correction is

\[
 \mathcal Q_{qr}=-\frac{A_b^2\lambda}{8\delta_h}
                             \operatorname{Re}P_{qr},\qquad
 \sup_{(z_q,z_r)\ne(0,0)}
 \frac{|\mathcal Q_{qr}|}{\lambda(q)|z_q|^2+\lambda(r)|z_r|^2}
               =\frac{2g b^2}{|\delta_h|}.                     \tag{38}
\]

The supremum is over nonzero pair errors; equality uses equal weighted
modal magnitudes and the appropriate relative phase. For the amplitude
scale imposed by periodic family (15), this ratio is

\[
 R_h=\frac{2K(m)^2}{N^2\sin^2(\pi/N)}.                         \tag{39}
\]

It exceeds one half even as `m` tends to zero. Shrinking the bookkeeping
varepsilon at fixed physical profile cannot make it vanish. The continuous
elliptic integral K increases from `pi/2` to infinity, so the class contains
an amplitude with `K(m)=pi`. For that analytic witness `R_h>2`, while
`b=2*pi*sqrt(2d/g)/1024<1/16` because `sqrt(2d/g)<9` and `pi<7/2`.
Its leading phase gaps are small. Yet the canonical principal correction
has a direction below `-2*E_0`. The relative-storage Hessian itself is at
most E_0 since `cos(a)<=1`; adding this correction loses positivity.

This is a scoped obstruction to **eliminating all nonzero divisors around
J_0 while retaining only exact resonances with a small correction** over
that frozen-carrier amplitude class. It does not exclude (30), which can
retain these adjacent modes, or all positive time-dependent metrics.
Adding a large homogeneous metric changes the normalization and its error
bounds. Including the slow phase rate or loss in a different inverse
changes the divisors and requires a separate calculation. The actual
localized envelope has spatially varying coefficients and additional
couplings; (39) is not a nonexistence or instability theorem for it.

<a id="periodic-envelope-uniform-bounds"></a>
## 9. The spatially varying periodic reference has explicit finite bounds

The following calculation applies to the whole periodic family (15), not
its frozen carrier. Fix the analytical rule `Delta=lambda/2` for (30).
Retain the original support and loss, `N=1024`, `sigma=pi/(N^2-1)`, and
`k=pi/4`. No profile parameter, source tolerance or response is selected
or evaluated. The formulas below are uniform in the continuous translation
of the profile and in time.

### Physical profile and derivative budgets

Write the site-scale profile as `F(s)=varepsilon*f(varepsilon*s)` and put

\[
 \zeta=\frac{2K(m)}N,\quad b=\zeta\sqrt{2d/g},\quad
 \omega_s=d\zeta^2(2-m)=\frac{g b^2(2-m)}2,\quad
 \psi_j=kj+(\lambda-\omega_s)\tau.
\]

The full reconstruction (10) is exactly
`z_app,j=F(j+v*tau)*exp(i*psi_j)+F(j+v*tau)^3*sum_l a_l*exp(i*l*psi_j)`.
The bookkeeping varepsilon has disappeared. The first integral and its
derivative give the uniform bounds

\[
 \|F\|_\infty\le b,\quad
 B_1=\frac{\zeta b m}{2}\ge\|F'\|_\infty,\quad
 B_2=\zeta^2 b m\ge\|F''\|_\infty,\quad
 B_3=\frac{\zeta^3 b m(4+m)}2\ge\|F'''\|_\infty.               \tag{40}
\]

For example, set `s=1-(F/b)^2` in `[0,m]`. Then
`F'^2=zeta^2*b^2*s*(m-s)`, whose upper bound follows from
`s*(m-s)<=m^2/4`. Also `F''=zeta^2*F*(2s-m)` and
`F'''=zeta^2*(-4-m+6s)*F'`, proving the other two inequalities.
The exact slow derivative of the profile factor is `-i*omega_s*F`.
It must not be omitted when differentiating the moving metric.

Let `A_h=sum_l |l*a_l|`, with the three harmonic coefficients from (11).
For the leading edge amplitude c of (29) and the harmonic contribution to
the phase gap, define

\[
\begin{split}
 C_0&=D_k b+B_1,\qquad
 C_1=|v|(D_k B_1+B_2)+\omega_s C_0,\\
 G_h&=2A_0b^3,\qquad
 V_h=6|v|A_0b^2B_1+2|\lambda-\omega_s|A_h b^3,\\
 G&=C_0+G_h,\qquad a_*=\lambda C_0+C_1+V_h.
\end{split}                                                     \tag{41}
\]

The fundamental theorem of calculus across one site proves
`||c||_infinity<=C_0`, `||c'||_infinity<=C_1`.
Differentiating every harmonic, including its spatial transport, gives
the bounds G_h and V_h. Consequently `||a||_infinity<=G` and
`||a'||_infinity<=a_*` for the complete reference.

To specify exactly the inherited scalar residual budget under audit, put
`S_0=d*B_2+g*b^3` and rewrite (18) without varepsilon:

\[
\begin{split}
 r_*={}&B_3/6+3A_1b^2B_1+(B_1/2)C_0^2\\
 &+A_0(3b^2S_0+3bB_1^2+\tfrac32b^2B_2)
       +A_0b^3G^2+G^5/120+2\sigma(b+A_0b^3).                   \tag{42}
\end{split}
\]

Here `S_0>=omega_s*b`; it deliberately retains the already proved scalar
budget. No lower bound on the *actual* residual is asserted by any positive
term of (42). Sharper use of the exact slow derivative, harmonic phases or
damping row would define a different budget needing its own proof.

### All-mode bounds without a dimension factor from the Frobenius norm

For a smooth N-periodic function use its continuous Fourier coefficients
with normalization `1/N`, and let W be their absolute sum. Parseval and
Cauchy--Schwarz, with `sum_(n!=0) 1/n^2=pi^2/3`, give

\[
 W(F)\le b+\frac{N B_1}{2\sqrt3}=:W_0,\qquad
 W(F')\le\frac{N B_2}{2\sqrt3}=:W_1.                           \tag{43}
\]

The second bound uses the zero mean of a periodic derivative.
Absolute convergence makes sampling
legitimate: discrete Fourier coefficients sum aliased continuous
coefficients, so their absolute sum cannot increase. Translation changes
the phases of the continuous coefficients, so the same alias bound holds
uniformly over shifts; discrete aliased coefficients need not acquire one
common phase. Multiplication obeys `W(f*g)<=W(f)*W(g)` and the
one-site difference has norm at most two. Thus the sampled edge amplitudes
satisfy

\[
 W(c)\le Z_0:=D_kW_0+W_1,\qquad
 W(c')\le Z_1:=2(|v|W_1+\omega_s W_0).
\]

This is a norm of Fourier coefficients, not a restriction to selected
carrier modes. In the normalized eigenbasis of J_0, each congruence entry
of H has magnitude `lambda*|hat(c^2)(-q-r)|/8`. The factors from U have
modulus one on every nonconstant mode. Summing over both frequency signs
bounds every absolute row and column sum by `lambda*W(c^2)/4`.
The symmetric near/far mask cannot increase these sums. The Schur bound,
unitary invariance of matrix norms and (30) therefore prove, for the fixed
`Delta=lambda/2`,

\[
\begin{split}
 k_*&=Z_0^2\ \ge\|K\|_2,\qquad
 d_*=2Z_0Z_1\ \ge\|\operatorname{Re}(e^{i\omega\tau}C')\|_2,\\
 m_*&=\tfrac\lambda4 Z_0^2
 +\tfrac12\left[(G_h+G^3/6)a_*+C_0(C_1+V_h)\right]
                                  \ \ge\|M_{\rm ret}\|_2.
\end{split}                                                     \tag{44}
\]

For the last remainder, write
`sin(a)*a'-a_0*a'_(0,fast)=(sin(a)-a_0)*a'+a_0*(a'-a'_(0,fast))`.
The harmonic gap is at most G_h and `|sin(a)-a|<=G^3/6`, giving the
second term of m_*. Thus (44) includes spatial transport, slow phase,
generated harmonics and sine curvature, with no frozen-envelope assumption.
Metric positivity still requires `k_*<cos(chi)` separately. These are
explicit sufficient constants, not a claim that this cutoff is optimal.

<a id="periodic-scalar-defect-obstruction"></a>
### The inherited scalar defect cannot close this bootstrap

Use the constant residual allowance
`mathfrak r_*=2*sqrt(N)*r_*` in (35), justified by
`||L^(1/2)||=2`, and the natural derivative budget a_* from (41).
Even with zero initial error, the resulting majorant cannot close its
edge-error ball over `T>=T_phi=1/omega_s`, for **any** `0<m<1` and
positive radius rho satisfying the chart premises. This conclusion does
not depend on the size of the retained-block bound or on the cutoff.

To prove it, set `c_5=D_k^5/120`. The published allowances obey

\[
 a_*\ge\lambda D_k b,\quad
 r_*\ge2\sigma b+c_5b^5,\quad
 T_\phi>\frac1{g b^2},\quad
 \delta_\kappa:=\cos(\chi)-k_*\le1.                            \tag{45}
\]

If metric positivity fails, there is already no certificate. Otherwise
`alpha>=a_*rho/(6*delta_kappa)>0`. The constant-residual term in (35) and
`exp(x)-1>=x^2/2` imply

\[
 \frac{Q(T_\phi)}\rho
 \ge\frac{(1+k_*)a_*\mathfrak r_*T_\phi^2}
               {12\delta_\kappa^{3/2}}
 \ge\frac{a_*\mathfrak r_*T_\phi^2}{12}.                       \tag{46}
\]

For the fixed carrier, `lambda>1/4`, `D_k>3/4`, `g<1/72` and
`c_5>1/512`; the original loss satisfies `sigma>3/N^2`.
Using `u+v>=2*sqrt(u*v)` for positive u,v gives

\[
\begin{split}
 \frac{a_*\mathfrak r_*T_\phi^2}{12}
 &>\frac{\sqrt N\lambda D_k}{6g^2}
                   \left(\frac{2\sigma}{b^2}+c_5b^2\right)\\
 &\ge\frac{\sqrt N\lambda D_k}{3g^2}\sqrt{2\sigma c_5}
 >\frac{81\sqrt3}{128}>1\qquad(N=1024).                       \tag{47}
\end{split}
\]

Thus `Q(T_phi)>rho`, whereas (35) requires
`Q(T_phi)<rho*sqrt(delta_kappa)<=rho`. Larger horizons or initial errors
only increase this particular majorant. Neither choosing another width
nor eliminating every other growth allowance repairs it.

There is a separate obstruction if one attempts only to improve the cubic
error term while retaining the same scalar forcing allowance and global
edge-to-node conversion. Even granting zero growth, zero initial error
and zero mean error, the resulting full-state allowance U_T obeys

\[
 \frac{U_{T_\phi}}{\sqrt N\,b(1+A_0b^2)}
 \ge\frac4{g(2-m)\sqrt{\ell_1}(1+A_0b^2)}
                  \left(\frac{2\sigma}{b^2}+c_5b^2\right)
 >\frac{9\sqrt3}{8}>1.                                        \tag{48}
\]

Here `G+rho<pi/2` implies `b<4`; `A_0<1/16` gives
`1+A_0b^2<2`, and `sqrt(ell_1)<8/N`. The same arithmetic-geometric-mean
bound proves the final inequality. The denominator in (48) is already an
upper bound on the reference norm. This allowance therefore cannot certify
an error smaller than the signal itself, even before resonance growth.
It is not a lower bound on actual error or a failure of the exact flow.

**Method decision.** Equations (40)--(44) supply the previously missing
spatially varying constants, but the inherited scalar defect treatment
fails before their optimization could matter. Equations (47)--(48) exclude
that specified combination of residual, nonlinear remainder and norm
conversion, not all relative-storage methods or all conservative
references. A damped reference has not been proved necessary. Retaining
the defect's phase, spatial support and two-row correlations in its
transport, or a different justified residual/observation estimate, changes
the premises. Merely bounding the retained block more tightly does not.

<a id="signed-envelope-defect-transport"></a>
## 10. Signed defect transport retains phase, aliases and both rows

The next comparison changes the failed allowances, rather than optimizing
(42). Let `h_0=2*pi/N`, `omega= lambda-omega_s`, and expand the continuous
periodic profiles in Fourier coefficients normalized by `1/N`. The real
coordinates of the complete reference in section 9 are

\[
 x_*=X_1\cos\psi+X_3\cos3\psi,\qquad
 \theta_*=\Theta_1\sin\psi+\Theta_3\sin3\psi,
\]
\[
 X_1=F+a_{-1}F^3,\quad\Theta_1=F-a_{-1}F^3,\quad
 X_3=(a_3+a_{-3})F^3,\quad\Theta_3=(a_3-a_{-3})F^3.             \tag{49}
\]

The profiles are evaluated at `j+v*tau`. Retain spacetime labels `(n,l)`
with n an integer and `l=1,3` initially, their conjugates, and all labels
generated by the nonlinearity. Define

\[
 q_{nl}=nh_0+lk\pmod{2\pi},\qquad
 \Omega_{nl}=nh_0v+l\omega,\qquad
 v_{nl}=\left(\widehat X_l(n)/2,\widehat\Theta_l(n)/(2i)\right),
 \quad v_{-n,-l}=\overline{v_{nl}}.                             \tag{50}
\]

Set `v_nl=0` outside the reference harmonics `l=+/-1,+/-3`; nonlinear
residuals can still have nonzero coefficients at those other labels.
Spatial aliasing does not identify temporal frequencies: replacing n by
`n+N` leaves q unchanged but changes Omega by `2*pi*v`. Nor does a nonzero
carrier ensure constant reference means: some aliased labels have q zero
and Omega nonzero. Both effects must remain in a full-state comparison.

Take `d_q=1-exp(i*q)` for the oriented incidence. With
`a_nl=d_q*(v_nl)_theta`, convolution `*` on the integer label lattice, and
`S(a)=sin(a)-a`, the exact coefficient formulas are

\[
 \widehat S(a)=\sum_{j\ge1}\frac{(-1)^j}{(2j+1)!}
                                      a^{*(2j+1)},\qquad
 J_q=\lambda_q\begin{pmatrix}-\sigma&-1\\1&0\end{pmatrix},
 \quad\lambda_q=1-\cos q,
\]
\[
 r_{nl}=(i\Omega_{nl}I-J_q)v_{nl}
       +\left(\tfrac12\overline{d_q}\,\widehat S(a)_{nl},0\right).
                                                               \tag{51}
\]

These are the signed residuals of both real rows under the actual law.
Smooth periodic profiles give absolutely summable coefficients and first
temporal derivatives. The absolutely convergent sine series justifies the
convolutions and interchange of sums. Cancellations in (51) occur before
taking a norm; independently bounding its derivative, tangent and sine
terms would generally destroy the small defect one is trying to assess.

### Actual positive-loss harmonic transport

For a real temporal frequency Omega, define the entire matrix kernel

\[
 \mathcal K_q(\tau,\Omega)
   =\int_0^\tau e^{J_q(\tau-s)}e^{i\Omega s}\,ds
   =\tau e^{i\Omega\tau}\varphi_1(\tau(J_q-i\Omega I)),
 \qquad \varphi_1(Z)=\sum_{j\ge0}\frac{Z^j}{(j+1)!}.           \tag{52}
\]

When its inverse is admitted, the equivalent expression is
`K_q=(exp(i*Omega*tau)*I-exp(J_q*tau))*(i*Omega*I-J_q)^-1`.
For q nonzero, the determinant is
`lambda_q^2-Omega^2+i*sigma*lambda_q*Omega`, which cannot vanish for
positive sigma and real Omega. The entire expression remains the owner of
the zero-mode and small-denominator limits. At q zero, K is
`(exp(i*Omega*tau)-1)/(i*Omega)*I`, with value `tau*I` at Omega zero.

For the actual `0<sigma<2`, put `s_sigma=sqrt(1-sigma^2/4)` and
`a_q=sigma*lambda_q/2`. Direct squaring of
`C_sigma=[[-sigma/2,-1],[1,sigma/2]]` gives

\[
 e^{J_q\tau}=e^{-a_q\tau}
 \left[\cos(\lambda_qs_\sigma\tau)I
       +\frac{\sin(\lambda_qs_\sigma\tau)}{s_\sigma}C_\sigma\right].
                                                               \tag{53}
\]

The branch factors in (52) are
`[exp(i*Omega*tau)-exp((-a_q +/- i*lambda_q*s_sigma)*tau)] /
[a_q+i*(Omega -/+ lambda_q*s_sigma)]`. Their corresponding spectral
projectors are not orthogonal and cannot be replaced by unit weights.
At all modes the full matrix satisfies `||K_q||_2<=tau`, since the actual
semigroup is contractive. Thus no uniform nonzero detuning or unproved
number-theoretic condition on aliases is needed. Resolved nonresonant
labels may have a much smaller effect than this general bound.

The signed linear defect response is

\[
 h_j(\tau)=-\sum_{n,l} e^{iq_{nl}j}
                         \mathcal K_{q_{nl}}(\tau,\Omega_{nl})r_{nl}.
                                                               \tag{54}
\]

For Parseval, sum the separate time-dependent contributions within each
spatial mode first, then use `||h||_2^2=N*sum_q ||hat(h)_q||_2^2`.
No inverse Laplacian occurs. In particular,
`K_q*(i*Omega*I-J_q)*v=(exp(i*Omega*tau)*I-exp(J_q*tau))*v`;
the linear part of (51) and the nonlinear coefficient must still be
combined before replacing (54) by an absolute bound.

### Finite truncations need their own tail allowance

The exact representation is not a license to omit small coefficients.
For any scalar N-periodic profile P with `||P'''||_infinity<=C_3`,
integration by parts gives, for an integer cutoff M at least one,

\[
 \sum_{|n|>M}|\widehat P(n)|\le\frac{C_3}{h_0^3 M^2},\qquad
 \sum_{|n|>M}|nh_0\widehat P(n)|\le\frac{2C_3}{h_0^2M}.       \tag{55}
\]

Use `C_3=B_3` for F and
`C_3=6B_1^3+18bB_1B_2+3b^2B_3` for F cubed. Equation (55) controls
both the omitted coefficients and their temporal derivative, including
translation. The factor `|l*omega|` for each harmonic must also be retained.
For phase-gap coefficient sums bounded by A, truncating the sine series
after degree `2P+1` leaves the separate absolute-sum allowance

\[
 \sum_{j>P}\frac{A^{2j+1}}{(2j+1)!}
 \le\frac{A^{2P+3}}{(2P+3)!}
       \left[1-\frac{A^2}{(2P+4)(2P+5)}\right]^{-1},           \tag{56}
\]

provided the displayed denominator is positive. A phase-gap truncation of
absolute-sum size epsilon additionally changes S by at most
`(cosh(A)-1)*epsilon`, where A bounds every interpolating coefficient norm.
The incidence factor `|d_q|/2<=1` does not enlarge this form-residual sum.
If the complete omitted two-row residual has coefficient-norm sum at most
epsilon_r, its contribution to (54) is at most `sqrt(N)*T*epsilon_r`.
This explicit tail treatment replaces the fixed fifth-order allowance;
it is not permission to reduce a tail without a declared finite budget.

<a id="response-corrected-envelope-reference"></a>
## 11. A response-corrected reference has an exact phase row

Equation (54) solves `h'=J_sigma*h-r_app`, `h(0)=0`, where r_app is
the **signed vector residual** of (51), distinct from the scalar allowance
r_* in (42).
Define the independent corrected reference `z_c=z_app+h`. Its full-law
residual is exactly

\[
 r_c=\left(R(\theta_{\rm app})-R(\theta_{\rm app}+h_\theta),0\right).
                                                               \tag{57}
\]

Indeed `z_c'=J_sigma*z_c+R(theta_app)`. Its phase row is exact, and both
reference means remain at their initial values. Aliased mean drift in the
original reference is canceled by the zero-mode part of (54); it was not
assumed absent. This is a prescribed response correction, not a trajectory
of the nonlinear law and not a supplied pressure replacing that law.

Equation (57) is generally **first order** in h. For example, at a
nonconstant reference with gaps a in `(-pi,pi)`, take `h_theta=t*theta_app`.
Then the inner product of `r_c,x/t` with theta_app tends to
`-sum_e a_e^2*(1-cos(a_e))/2<0`. A quadratic-error claim would be false.
A genuinely quadratic residual would instead require solving the full
variational correction
`h'=J_sigma*h+(DR(theta_app)*h_theta,0)-r_app`, with its actual time-dependent
Jacobian. That is a different linear problem and needs a separate bound.

For that full variational correction `h_v=(u_v,eta_v)`, the exact remaining
form source is `R(theta_app)+DR(theta_app)*eta_v-R(theta_app+eta_v)`.
If its phase-gap norms are bounded by H_v0 and H_v2 in infinity and
Euclidean norm, respectively, Taylor's theorem gives

\[
 \|r_v\|_2\le
 \left[\tfrac12\min(1,G_{\rm app})H_{v0}+H_{v0}^2/6\right]H_{v2}.
\]

Indeed the edge remainder is `sin(a+b)-sin(a)-cos(a)*b`, bounded by
`|sin(a)|*b^2/2+|b|^3/6`, and `||D^T||/2=1`. This is genuinely
quadratic in h_v and needs no acute-chart assumption. It does not supply
the time-dependent coupled-mode propagator or a quantitative bound on h_v.

For (54), if `||Dtheta_app||_infinity<=G_app`,
`||Dh_theta||_infinity<=H_0`, and `||Dh_theta||_2<=H_2`, the exact sine
secant gives the conditional source allowance

\[
 \|r_c\|_2\le
 \left(\tfrac12G_{\rm app}^2+\tfrac12G_{\rm app}H_0
                                      +H_0^2/6\right)H_2.      \tag{58}
\]

It starts at zero when h does and depends on the signed transported defect.
It has neither the standalone damping nor the fifth-order floor in (42).
The actual damping remains in (52)--(54). Bounding (58) by a large constant
can still be inconclusive; the exact residual (57) and time dependence
remain available. The total error to the original reference is `h+e`,
where `e=z-z_c`; the correction h must always be reported.

<a id="euclidean-envelope-comparison"></a>
## 12. Full-state metric without a spectral-gap conversion

A second constructive change removes the `ell_1^-1/2` factor in (48).
For a reference `(x_*,theta_*)`, let `xi=(u,eta)` be the full nodal error,
including its means, and start with `E_0=||xi||_2^2/2`.
Write `a=Dtheta_*`, `B=DR(theta_*)`, and
`M(B)=[[0,B/2],[B/2,0]]`. Its exact quadratic production is
`xi^T M(B) xi=u^T B eta`. The real errors have no Fourier-band restriction.

Use the original leading edge wave `a_0=Im(c*exp(i*lambda*tau))`, retaining
`||c||_infinity<=C_0` and `||c'||_infinity<=C_1` from (41). Thus
`||a_0||_infinity<=C_0`. Admit `||a-a_0||_infinity<=H_gap` and
`G=C_0+H_gap`.
For z_app, `H_gap=G_h`; for z_c one may use `H_gap=G_h+H_0`.
The exact splitting is

\[
 B_0=\tfrac18D^T\operatorname{diag}|c|^2D,\qquad
 B_f=-\tfrac18D^T\operatorname{diag}(c^2)D,\qquad
 B=B_0+\operatorname{Re}(e^{2i\lambda\tau}B_f)+B_{\rm rem},
\]
\[
 \|M(B_{\rm rem})\|_2
             \le G^4/24+(G+C_0)H_{\rm gap}/2.                  \tag{59}
\]

Taylor's inequality `|1-cos(a)-a^2/2|<=|a|^4/24` and `||D||=2`
prove (59). There is no moving-curvature `a'` term in this energy.
An acute chart is not a positivity premise for the metric below.

### Static and oscillating corrections preserve every coordinate

On the mean-zero space solve `A S+S A=B_0`, and set S zero on constants.
The positive Sylvester integral and `0<=B_0<=C_0^2*A/4` give

\[
 S=\int_0^\infty e^{-As}B_0e^{-As}\,ds,\qquad
 0\le S\le C_0^2I/8,\qquad \|S'\|_2\le C_0C_1/4.            \tag{60}
\]

With `K_0=diag(S,-S)`, direct multiplication yields
`J_0^T K_0+K_0 J_0=-2M(B_0)`. The incidence factors remove the apparent
small denominator; no inverse spectral-gap bound is used.

For the oscillating matrix `H=M(B_f)` use the fixed
`Delta=lambda/2` homological split (30). In eigenvectors
`(f_q,-i*s*f_q)/sqrt(2)`, s equal to plus or minus one, opposite signs
give zero congruence entries. Same-sign entry magnitudes are
`|d_q*d_r|*|hat(c^2)(-q-r)|/16`. The inequality
`|d_q*d_r|<=lambda_q+lambda_r` bounds the far inverse. Only the
negative/negative block can be near resonant. Its real part has half the
complex operator norm: it joins two orthogonal conjugate eigenspaces.
The real far inverse is bounded by half the sum of its two branch norms.
The row-sum estimates of section 9 consequently give, for `K=K_0+K_f`,

\[
 k_*=(C_0^2+3Z_0^2)/8\ge\|K\|_2,\qquad
 d_*=(C_0C_1+3Z_0Z_1)/4,
\]
\[
 m_*=5\lambda Z_0^2/64+G^4/24+(G+C_0)H_{\rm gap}/2.           \tag{61}
\]

Here d_* bounds the uncanceled coefficient derivative
`K_0'+Re(exp(2i*lambda*tau)*C_f')`, not the complete derivative K'.
The latter includes the canceled fast terms. The value m_* bounds the
retained quadratic production after those cancellations. Means remain
under the identity metric, since every correction has zero constant rows
and columns. These are new Euclidean constants, not transferred Bregman
constants with the spectral factor removed by assertion.

### Nonlinear remainder, actual loss and a prospective bootstrap

Assume `||xi||<=rho`, `k_*<1`, and set

\[
 n_\rho=\sqrt2\min(1,G)\rho+\tfrac23\rho^2,\qquad
 c_*=m_*+d_*/2+k_*(2\sigma+G^2)+(k_*+\tfrac12)n_\rho.
\]

Taylor's theorem, `||Deta||_infinity<=sqrt(2)||eta||_2` and `||D||=2`
give `||R(theta_*+eta)-R(theta_*)-B*eta||<=n_rho*||eta||`.
The baseline nonlinear work then costs at most
`n_rho*||xi||^2/2`, using `||u||*||eta||<=||xi||^2/2`.
For `E_c=xi^T(I+K)xi/2`, both actual rows imply

\[
 \frac{1-k_*}{2}\|\xi\|^2\le E_c\le\frac{1+k_*}{2}\|\xi\|^2,
 \quad
 E_c'\le-\sigma u^TAu+c_*\|\xi\|^2
                         +(1+k_*)\|\xi\|\|r\|.              \tag{62}
\]

The terms `2*sigma` and `G^2` in c_* bound the interaction of K with the
actual damping and reference Jacobian, respectively. Thus, for uniform
admitted constants, initial full-state error delta, and `alpha=c_*/(1-k_*)`,

\[
 \|\xi(\tau)\|\le\frac{Q(\tau)}{\sqrt{1-k_*}},\qquad
 Q(\tau)=e^{\alpha\tau}\left[
 \sqrt{1+k_*}\,\delta+
 \frac{1+k_*}{\sqrt{1-k_*}}
          \int_0^\tau e^{-\alpha s}\|r(s)\|\,ds\right].        \tag{63}
\]

The strict condition `Q(T)<rho*sqrt(1-k_*)` proves the whole interval by
first exit. No Poincare factor or unobserved mean coordinate remains.
If an acute phase chart is separately needed, require
`G+sqrt(2)*rho<pi/2`. For the response-corrected reference, insert (57)
or its proved allowance (58), and include `||h||` in the final error to
z_app. Time-dependent source bounds must not be replaced by their
endpoint values without a proof; h starts at zero but can vary thereafter.

This changes both mechanisms behind (47)--(48): the residual is transported
before its nonlinear remainder is bounded, and the error metric controls
the full nodes directly. It supplies a conditional comparison with no
inherited scalar-defect floor. It does **not** prove that (63) closes on
the periodic family. The remaining quantitative obligation is a certified
bound on (54), (57), their Fourier/sine tails and the nonlinear allowance
over a phase time, with a nonzero localization and signal margin.

<a id="canonical-defect-budget-preflight"></a>
## 13. A canonical-profile preflight rejects a positive coefficient budget

The signed comparison above remains conditional. Its first quantitative
preflight fixes the self-complementary profile parameter `m=1/2`, exact
nominal initial state and `T=T_phi=1/omega_s`. Keep the same C1024 law and
carrier. The declared finite budget is profile cutoff `|n|<=12`, sine
degree seven and the shared 128-bit outward arithmetic. The preflight
stops at a failed necessary condition before constructing those finite
sums. No profile search, prepared nonlinear response or trajectory is used.

The particular allowance tested here replaces each signed kernel vector
by a positive full-two-row norm, then uses that norm to bound its phase
component. For an invertible `M_nl=i*Omega_nl*I-J_q`, set

\[
 B_{nl}(t)=\min\left(t\|r_{nl}\|_2,
                         2\|M_{nl}^{-1}r_{nl}\|_2\right),\qquad
 U_q(t)=\sum_{q_{nl}=q}B_{nl}(t).
                                                               \tag{64}
\]

At the singular mean label use `t*||r_nl||`. Contractivity and (52)
justify both branches of (64). Sum within each spatial alias class before
Parseval, giving the explicit **positive allowances**

\[
 H_0^+(t)=\sum_q|d_q|U_q(t),\qquad
 H_2^+(t)=\left(N\sum_q|d_q|^2U_q(t)^2\right)^{1/2},
\]
\[
 S^+(t)=\left(\tfrac12G_{\rm app}^2
       +\tfrac12G_{\rm app}H_0^+(t)+(H_0^+(t))^2/6\right)H_2^+(t).
                                                               \tag{65}
\]

Here `G_app=C_0+G_h` is specifically the published allowance (41), and
S^+ instantiates (58). Finite outward enclosures retain positive tail
allowances. These operations are justified upper bounds, but discard
vector directions and temporal cancellation inside the alias sums.
The lower bounds below apply to **this chosen allowance**, never to the
norm of the signed sum (54), the actual correction or its exact residual.

### A necessary source moment eliminates a radius search

More generally, use any nonnegative residual allowance S(t) in (63).
For uniform `k_*<1`, gap allowance G and
`s_G=min(1,G)`, its exponent satisfies

\[
 \alpha\ge a_1\rho,\qquad
 a_1=\frac{\sqrt2 s_G(k_*+1/2)}{1-k_*}.
\]

Since `exp(x)>=x` for nonnegative x, even at zero initial error the
bootstrap implies the necessary condition

\[
 \frac{Q(T)}{\rho\sqrt{1-k_*}}\ge
 \frac{\sqrt2 s_G(k_*+1/2)(1+k_*)}{(1-k_*)^2}
                    \int_0^T(T-t)S(t)\,dt <1.                 \tag{66}
\]

If the displayed lower bound is at least one, no positive choice of rho
closes this majorant. This is a necessary test for a sufficient error
method, not a necessary condition for actual retention. It detects the
interaction of positive residual accumulation with the conversion from
full nodal error radius to pointwise phase-gap error in n_rho.

### One retained coefficient already forces failure

Let `f_0=hat(F)(0)`. The first integral (15), with
`F/b=sqrt(1-m*sin(phi)^2)` and
`ds=dphi/[zeta*sqrt(1-m*sin(phi)^2)]`, gives the period averages directly:

\[
 f_0=\frac{\pi\sqrt{2d/g}}N,\qquad
 \widehat{F^3}(0)=f_0 b^2(1-m/2).
                                                               \tag{67}
\]

Thus no Fourier quadrature is needed for the retained label `(0,1)`.
At `m=1/2`, put `z_0=3b^2/4`, and define
`X=f_0*(1+a_-1*z_0)`, `Theta=f_0*(1-a_-1*z_0)` and
`Omega=lambda-omega_s`. Its reference coefficient is `(X/2,-i*Theta/2)`.
The profile is even, and the complete phase row is odd under simultaneous
reflection of the envelope and carrier arguments. The actual sine
divergence preserves that oddness: the two neighbor shifts exchange under
reflection. Its spacetime coefficients are therefore purely imaginary.
Consequently, for some real beta_r, the exact signed residual has the form

\[
 r_{01}=\left(A_r+i\beta_r,C_r\right),\qquad
 A_r=\sigma\lambda X/2,\quad C_r=(\Omega\Theta-\lambda X)/2.
\]

No bound on beta_r or sine truncation is used. The exact resolvent numerator
has imaginary first component `Omega*A_r` and real second component
`lambda*A_r+sigma*lambda*C_r=sigma*lambda*Omega*Theta/2`. Hence, writing
`D_0=lambda^2-Omega^2+i*sigma*lambda*Omega`,

\[
 \|r_{01}\|\ge\sigma\lambda X/2,\qquad
 2\|M_{01}^{-1}r_{01}\|\ge
       \frac{\sigma\lambda|\Omega|\sqrt{X^2+\Theta^2}}{|D_0|}.
                                                               \tag{68}
\]

All required inequalities follow from elementary fixed-profile bounds.
The elliptic integral definition gives
`pi/2<K(1/2)<=pi/sqrt(2)`. At the unchanged carrier,
`1/4<lambda<1/3`, `1/4<d<1/2`, `8<sqrt(2d/g)<9`, and `D_k>3/4`.
Using `3<pi<22/7` and `pi^2<10` gives

\[
 \frac{24}N<f_0\le b<\frac{81}{2N}<\frac1{25},\qquad
 \frac{27}{8N^2}<\omega_s<\frac{15}{N^2},\qquad
 \frac3{N^2}<\sigma<\frac{16}{5N^2}<\omega_s.
                                                               \tag{69}
\]

Also `G_app<1/24`: for this fixed m, (40)--(41) give
`C_0<(513/512)b` and `G_h< b^3/8`. These loose bounds suffice.
Equation (69) implies `Omega>=lambda/2`, while `|a_-1|<1/16`
implies `X>=f_0/2`. Since
`X^2+Theta^2>=2*f_0^2` and `|D_0|<=3*lambda*omega_s`, (68) yields

\[
 B_{01}(t)\ge\frac{\sigma\lambda f_0}{5}\,t
                          \qquad(0\le t\le T).
                                                               \tag{70}
\]

Indeed the first branch in (64) is at least `sigma*lambda*f_0*t/4`;
the second is at least `sigma*lambda*f_0*T/(3*sqrt(2))`, larger than
`sigma*lambda*f_0*t/5`. The conjugate label has the same bound and lies
at the distinct spatial mode `-k`. Thus the **chosen** H_2^+ is at least
`sqrt(2N)*D_k*B_01`, and

\[
 S^+(t)\ge C_*t,\qquad
 C_*=\frac{\sigma\lambda f_0G_{\rm app}^2D_k\sqrt{2N}}{10}.
                                                               \tag{71}
\]

For the corrected reference G is at least G_app. Because `G_app<1`,
`min(1,G)>=G_app` even if the corrected gap allowance exceeds one.
If the metric is positive, `alpha>=G_app*rho/sqrt(2)` for every radius.
Equations (63) and (71), again using `exp(x)>=x`, give

\[
 \frac{Q(T)}\rho\ge
       \frac{\sigma\lambda f_0G_{\rm app}^3D_k\sqrt N T^3}{60}
       >\frac{7776}{625}>12.                                  \tag{72}
\]

For the last strict rational bound use (69), `G_app>=D_k*b>18/N`,
`D_k>3/4`, `lambda>1/4`, `T>N^2/15` and `sqrt(N)=32`.
The bootstrap requires `Q(T)/rho<sqrt(1-k_*)<=1`. If metric positivity
fails instead, there is already no certificate. Nonzero initial error,
additional positive mode allowances, tails and outward arithmetic can
only worsen this particular bound.

**Preflight result.** The declared positive coefficient budget fails at
the canonical profile before its finite Fourier/sine sums need evaluation.
It cannot be repaired by radius selection, extra modes or precision.
This does not exclude other profile parameters, the exact signed
phase-projected transport, a different nonlinear work estimate, or actual
maintenance. In particular, (72) is neither an observed error nor a lower
bound on an actual response. The next comparison must retain information
lost between (54), (64) and (65), or admit a genuinely different correction
law; repeating their positive-norm chain is not a quantitative advance.

<a id="finite-variational-representation"></a>
## 14. A finite signed variational representation has controlled tails

Keep the canonical `m=1/2` reference and full phase interval of section 13.
Write `V(t)=[[0,B(t)],[0,0]]`, with `B(t)=DR(theta_app(t))`, so that
`h_v'=(J_sigma+V(t))*h_v-r_app` and `h_v(0)=0`. The first-order feedback
is included in this linear comparison, not in a modified constitutive law.
V couples modes; separate constant two-by-two tangent propagators do not
by themselves propagate this equation.

The declared finite representation uses profile cutoff 24, the exact cube
of that finite profile, sine degree seventeen, cosine degree sixteen and
256 chronological Jacobian insertions. Retained time integrals and their
coefficients are exact mathematical objects here. The result below bounds
analytical truncation error; no retained sum, prepared response or
trajectory is evaluated. Shared rational/128-bit interval controls on
constants do not certify a future sum's accumulated arithmetic error.

### Chronology, resonance and the factorial tail

Let `S(t)=exp(J_sigma*t)`. Define operator terms for `s<=t` by

\[
 U_0(t,s)=S(t-s),\qquad
 U_{j+1}(t,s)=\int_s^t S(t-u)V(u)U_j(u,s)\,du,
\]
\[
 h_0(t)=-\int_0^t S(t-u)r_{\rm app}(u)\,du,\qquad
 h_{j+1}(t)=\int_0^t S(t-u)V(u)h_j(u)\,du.
                                                               \tag{73}
\]

The integrals preserve matrix order and signed vector addition. With
`L=G_app^2` and `R_*>=sup||r_app||`, contractivity of S and induction over
the ordered simplex give

\[
 \|U_j(t,s)\|\le\frac{[L(t-s)]^j}{j!},\qquad
 \|h_j(t)\|\le R_*t\frac{(Lt)^j}{(j+1)!}.                     \tag{74}
\]

The series converge uniformly on the whole interval even when `LT>1`.
Their sums solve the full variational initial-value problem. This is not
an assertion that its one-step Volterra map is a contraction.
For a nonnegative integer P, `beta>=LT` and `beta<P+2`, summing the decreasing
factorial ratios bounds the omitted orders `j>P` by

\[
 E_U\le\frac{\beta^{P+1}}{(P+1)!}
                   \left(1-\frac\beta{P+2}\right)^{-1},\qquad
 E_h\le R_*T\frac{\beta^{P+1}}{(P+2)!}
                   \left(1-\frac\beta{P+3}\right)^{-1}.
                                                               \tag{75}
\]

These positive bounds apply only to omitted orders. Replacing the retained
terms in (73) by (74) would discard the phase information being preserved.

In the spatial Fourier basis, write `f=1-cos(a)`, `a=Dtheta_app`. The
coupling is

\[
 B_{qr}(t)=\tfrac12\overline{d_q}d_r
       \sum_{\alpha:\,Q_\alpha=q-r}
                      \widehat f_\alpha e^{i\Omega_\alpha t}.
                                                               \tag{76}
\]

Each chronological path changes spatial mode by Q_alpha and retains its
own temporal frequency. Spatial aliases cannot merge those frequencies.
The d factors annihilate constant rows and columns. Thus terms h_j with
`j>=1` have zero means; h_0 alone cancels the original reference's mean
drift. The exact corrected reference still has a complete exact phase row.

Successive integrations of exponential factors are entire divided
differences of the exponential function. For nodes z_0,...,z_d, their
ordered-simplex representation is

\[
 \exp(t\cdot)[z_0,\ldots,z_d]
 =t^d\int_{\substack{u_j\ge0\\\sum_{j=0}^d u_j=1}}
                    \exp\!\left(t\sum_{j=0}^d u_jz_j\right)\,du.
                                                               \tag{77}
\]

The simplex measure has volume `1/d!`. Along a variational path the nodes
are actual damped tangent eigenvalues plus accumulated imaginary forcing
frequencies; the final source node is purely imaginary. Their real parts
are nonpositive, giving modulus at most `t^d/d!`. Coincident nodes use the
entire limit, for example `t^d*exp(t*z)/d!` when all nodes equal z.
There is no lower-detuning premise. If branch factors are used, retain the
actual nonorthogonal projectors from section 10. Chronological products
of different B(t) generally do not commute.

### The full linear propagator has a moderate uniform gain

The direct metric of section 12 applies at zero error radius to
`J_sigma+V(t)`, using the original reference gap `G=G_app`, `H_gap=G_h`.
Put `c_lin=m_*+d_*/2+k_*(2*sigma+G_app^2)` and
`alpha_lin=c_lin/(1-k_*)`. Its homogeneous energy bound gives

\[
 \|U(t,s)\|\le
 \sqrt{\frac{1+k_*}{1-k_*}}\exp(\alpha_{\rm lin}(t-s))<2^8
                      \qquad(0\le s\le t\le T).               \tag{78}
\]

Here is a fixed-profile verification rather than an assumed conditioning
constant. Equations (40)--(43) and (69) imply
`W_0<11b/8`, `W_1<b/256`, `C_0<b`, `Z_0<9b/8`, and `k_*<b^2<1/625`.
Also

\[
 G_{\rm app}/b<D_k+1/512,\qquad
 G_{\rm app}^2T<\frac{224}{3}\left(\frac{385}{384}\right)^2
                         <151/2,
                                                               \tag{79}
\]

using `lambda>2/7`, `D_k>3/4`, and `b^2T=4/(3g)`.
For the metric terms, `C_1*T<25/4`, `Z_1*T<37`,
`C_0<33/800` and `Z_0<9/200` give `d_*T<4/3`.
Equation (79) and `D_k<4/5` also give `G_app<b`, and `G_h<b^3/8`.
The near-block term obeys
`5*lambda*Z_0^2*T/64<945/256`; the remaining terms of m_* have sum
at most `b^4*T/6<1/20`. Hence `m_*T<81/20`.
Finally `sigma*T<1` and (79) give
`k_*(2*sigma+G_app^2)*T<78/625<1/8`. Therefore

\[
 \alpha_{\rm lin}T<\frac{581}{120}\frac{625}{624}<5,\qquad
 \sqrt{\frac{1+k_*}{1-k_*}}<21/20.
\]

Since `exp(1)<3`, their product is less than `(21/20)*3^5<256`, proving
(78). This controls all modes, including means; it does not bound the
inhomogeneous signed correction by a small observed value.

### Finite profile, source and Jacobian inputs

At `m=1/2`, the elliptic nome is `q_e=exp(-pi)<1/20`. The
[Jacobi dn Fourier formula](https://dlmf.nist.gov/22.11.E3), with the
[nome convention](https://dlmf.nist.gov/22.2.E1), gives

\[
 \widehat F(0)=f_0,\qquad
 \widehat F(n)=\frac{2f_0q_e^{|n|}}{1+q_e^{2|n|}}\quad(n\ne0).
                                                               \tag{80}
\]

DLMF's elliptic modulus is `sqrt(m)`, distinct from the carrier k.
The normalization follows from (67) and the period `N`; all coefficients
are positive. Thus `W(F)=b` and `W(F^3)=b^3`. These are properties of the
admitted profile, not evaluated trajectory data. Let F_M retain `|n|<=24`.
Its cube is formed exactly, so its support extends to `|n|<=72`.
The geometric tail allowance

\[
 E_F=\frac{4f_0q_e^{25}}{1-q_e}<2^{-110},\qquad
 W(F-F_M)\le E_F,\qquad
 W(F'-F_M')\le h_0\left(25+\frac{q_e}{1-q_e}\right)E_F<E_F/4
                                                               \tag{81}
\]

uses `f_0<1/32` and `h_0<1/160`. Also
`W(F')<=4*f_0*h_0*q_e/(1-q_e)^2<2^-14`.
Products give `W(F^3-F_M^3)<=3b^2E_F`; differentiating the product
gives a bound `[6bW(F')+3b^2/4]E_F<b^2E_F` for its derivative error.

Use the complete reconstruction (49) with F_M to obtain w_M and its gap
a_M. Positivity of the retained profile coefficients bounds every convex
interpolation of a and a_M in coefficient absolute-sum norm by

\[
 A\le D_kb+W(F')+2A_0b^3<1/16,
 \qquad W(a-a_M)<4E_F.                                       \tag{82}
\]

Let S_17 be the sine polynomial through degree seventeen minus its linear
term, and C_16 the polynomial for `1-cos` through degree sixteen. Define

\[
 \widetilde r=w_M'-J_\sigma w_M+
                  (\tfrac12D^TS_{17}(a_M),0),\qquad
 \widetilde B=\tfrac12D^T\operatorname{diag}(C_{16}(a_M))D,
 \qquad \widetilde V=\begin{pmatrix}0&\widetilde B\\0&0\end{pmatrix}.
                                                               \tag{83}
\]

This preserves the signed source terms before their final addition.
The source and Jacobian have different truncation allowances. In particular,

\[
 \epsilon_B:=\sup\|V-\widetilde V\|
 \le 8\sinh(A)E_F+
       \frac{2A^{18}}{18!}\left(1-\frac{A^2}{19\cdot20}\right)^{-1}
 <2^{-109}.                                                   \tag{84}
\]

Here `8*sinh(1/16)<1`; the second term is less than `2^-122`.
For the source, use `W(x)+W(theta)` to bound its two-row coefficient norm.
The profile substitution costs less than `3E_F` in state and `3E_F` in
time derivative; `||J_sigma||` in that sum norm is below four. The sine
substitution costs less than E_F, while its separate series tail is
`A^19/19!/(1-A^2/(20*21))`. Including the nodal factor `sqrt(N)=32` yields

\[
 \epsilon_r:=\sup\|r_{\rm app}-\widetilde r\|_2<2^{-99}.
                                                               \tag{85}
\]

These deliberately loose bounds control only removed input terms. They
do not substitute absolute coefficient sums for the retained solution.

### Combined analytical error and remaining computation boundary

Let h_tilde solve the finite-input linear equation
`h_tilde'=(J_sigma+V_tilde)*h_tilde-r_tilde`, with zero initial value.
Equations (18) and (42) give `R_*<32`: the bounds
`b,B_1,B_2,B_3,C_0,G_app,S_0,sigma<1/16`, `A_0<1/16` and `A_1<1/8`
already imply `r_*<1/32<1` by direct substitution. Equation (69) gives
`T<2^19`. Set M_*=2^8
from (78). Variation of constants for the input difference, using
`sup||h_v||<=M_*R_*T`, gives

\[
 E:=\sup\|h_v-\widetilde h\|
 \le M_*T\epsilon_r+M_*T\epsilon_B(M_*R_*T+E).
\]

Since `M_*T*epsilon_B<1/2`, this small **input-perturbation** term can be
absorbed. It is not a contraction assumption on the full V evolution.
Consequently

\[
 E<2M_*T\epsilon_r+2M_*^2R_*T^2\epsilon_B
   <2^{28}\epsilon_r+2^{60}\epsilon_B
   <2^{-71}+2^{-49}.                                         \tag{86}
\]

For the finite inputs, (79) and (84) give
`sup||V_tilde||*T<76`; also `sup||r_tilde||<33<64`, so its product with
T is below `2^25`. Use their signed chronological recurrence (73) through
order 256, calling the finite sum h_tilde^[256]. Exact rational factorial
comparisons in (75) give

\[
 \frac{76^{257}}{257!}\left(1-\frac{76}{258}\right)^{-1}<2^{-80},
 \qquad
 \frac{76^{257}}{258!}\left(1-\frac{76}{259}\right)^{-1}<2^{-88}.
\]

The second, forced coefficient has an additional factorial from the
source integration. It implies

\[
 \sup_{0\le t\le T}\|h_v(t)-\widetilde h^{[256]}(t)\|
       <2^{-71}+2^{-49}+2^{-63}<2^{-48}.                       \tag{87}
\]

This passes the declared `2^-40` analytical representation target. It is
not a smallness claim about h_v itself or the nonlinear error to the
reference. Every retained term is a finite signed sum of entire ordered
kernels, so (87) is constructive in exact arithmetic. Naive path
enumeration is nevertheless very large: the truncated Jacobian alone can
reach `|n|<=1152`, `|l|<=48`, and the source `|n|<=1224`, `|l|<=51`.
These are label bounds, not permission to truncate later generated modes.
Efficient accumulation, certified compression and numerical roundoff still
need an implementation and their own resource admission. A symbolic
finite representation is not an executed 128-bit solver.

### A defect certificate can avoid enumerating chronological paths

The finite inputs have the direct action
`V_tilde(t)*(u,eta)=(D^T[C_16(a_M(t))*Deta]/2,0)`. Evaluating the retained
profile and its derivative on the N nodes costs `O(NM)` by direct summation;
the Jacobian action then costs `O(N)`. The polynomial nonlinearities are
evaluated on those same nodes, without deleting spatial aliases or their
actual time dependence. This representation does not build a lifted path
matrix or change the finite-input equation.

Duhamel's formula and (78) bound its exact propagator by
`M_*exp(M_*epsilon_B*(t-s))<2^9`. For a continuous, piecewise continuously
differentiable approximation h_hat on a finite partition, define its complete
equation defect
`d_hat=h_hat'-(J_sigma+V_tilde)*h_hat+r_tilde`. Variation of constants gives

\[
 \|h_v(t)-\widehat h(t)\|
 <2^{-71}+2^{-49}
   +2^9\left[\|\widehat h(0)\|+
                        \int_0^t\|d_{\widehat h}(s)\|\,ds\right].
                                                               \tag{88}
\]

This is an alternative evaluation certificate for the same exact h_v.
It compares directly with h_tilde, so no chronological truncation term is
needed in (88). A numerical implementation must enclose coefficient,
arithmetic, derivative and between-sample errors in d_hat; pointwise
solver tolerances do not establish this integral. The small input tails
and the large retained forcing must remain separate. The
[first-panel result](#canonical-variational-first-panel) evaluates one local
approximation; no full-interval evaluator is resource-admitted.

An evaluated approximation h_hat must additionally certify its error to
h_tilde^[256], or supply the defect certificate (88). Only then may its
combined error allowance enlarge phase-gap bounds by
`sqrt(2)*epsilon_h` in infinity norm and `2*epsilon_h` in Euclidean norm,
and its full-state bound by epsilon_h. Those enlarged bounds describe the
exact h_v used in the quadratic residual of section 11. A finite-input
approximation is not itself an exact-phase-row reference unless its own
source and representation defects are retained.

The remaining nonlinear bootstrap, initial-error ball, signal/localization
margin and same-source linear discrimination are still unproved. This
result closes finite analytical representability and its truncation budget;
it does not establish maintenance or evaluate the whole representation.

<a id="variational-input-evaluation"></a>
## 15. Matrix-free inputs and whole-interval validation requirements

The finite-input equation (83) admits a direct nodal implementation without
enumerating chronological paths. Its fixed domain remains `N=1024`, `m=1/2`,
`M=24`, sine degree seventeen and cosine degree sixteen. Evaluation of these
prescribed inputs is distinct from integration of their forced linear
equation, and neither is a nonlinear prepared response.

### Enclosing the finite profile and its signed source

The elliptic constant is enclosed using the
[arithmetic-geometric mean identity](https://dlmf.nist.gov/19.8.E5).
Start with `a_0=1`, `g_0=sqrt(1/2)` and apply eight simultaneous updates

\[
 a_{j+1}=(a_j+g_j)/2,\qquad g_{j+1}=\sqrt{a_jg_j}.
\]

The exact sequences bound their common limit by `g_8<=AGM<=a_8`.
Outward interval arithmetic therefore encloses the complete elliptic
integral by `pi/(2*[g_8.lo,a_8.hi])`; agreement of rounded iterates is not
a stopping criterion or an assumed exact value. DLMF uses modulus
`sqrt(m)`, while this owner uses parameter m. The nome is evaluated as
`exp(-pi/4)^4`, within the existing unit-decay enclosure's domain.

These constants determine all the coefficients of F_M, the slow frequency,
and an interval containing the exact T. The numerical input evaluator may
use the rectangle `[0,T.hi]`; that is an enclosure of the physical horizon,
not a changed model clock. Time and both action vectors are admitted before
conversion or arithmetic. The factory consumes no cached bounds or verdicts.

Evaluate F_M and its spatial derivative at `j+v*tau`. Differentiate (49)
with both translation speed v and carrier frequency `lambda-omega_s`.
Form the exact cube of the retained profile at the same node, then compute
both signed source rows in (83) before taking a norm. Spatial harmonic
recurrences reuse fixed roots and two time rotations; they do not merge
distinct temporal frequencies merely because their spatial samples alias.

For any admitted real rows u and eta, the complete matrix-free action is

\[
 (J_\sigma+\widetilde V)(u,\eta)=
 \left(-\sigma Au-A\eta+
       \tfrac12D^T[C_{16}(a_M)D\eta],\ Au\right).
                                                               \tag{89}
\]

The incidence convention is `Dy_j=y_j-y_(j+1)` and
`D^Tz_j=z_j-z_(j-1)`. Thus both constant directions and both row means
are retained. Every output uses the same immutable input evaluation;
editing a displayed report must not change its captured action law.
No dense `2048 x 2048` matrix is needed. Direct profile evaluation costs
`O(NM)` and a Jacobian action costs `O(N)`. The original recurrence retained
`O(N+M)` live storage; the reusable spatial basis below explicitly trades
`O(NM)` storage for fewer interval operations per evaluation. These are
finite input costs, not a bound on a complete solution's work.

### Reusing fixed spatial work in the same finite inputs

Equation (80)'s finite Fourier profile already separates spatial and
temporal factors. With `C_jn=cos(n*h_0*j)`, `S_jn=sin(n*h_0*j)`,
`c_n=cos(n*h_0*v*tau)` and `s_n=sin(n*h_0*v*tau)`, write

\[
 F_M(j+v\tau)=f_0+2\sum_{n=1}^M f_n(C_{jn}c_n-S_{jn}s_n),
\]
\[
 F'_M(j+v\tau)=-2h_0\sum_{n=1}^M nf_n(S_{jn}c_n+C_{jn}s_n).
\]

These are exact identities for the same retained profile, including its
derivative. Spatial roots may use `(j*n) mod N`; n remains unreduced in
the temporal rotation and derivative. Distinct temporal frequencies do
not merge when their spatial samples alias. Preweight immutable spatial
rows once, then contract them against the M temporal harmonics at each
time. The carrier `k=pi/4` has only eight spatial classes, so its rotated
first and third harmonics also need only eight reconstructions per time.
The exact pointwise cube and its derivative `3*F_M^2*F'_M` remain in (49).
An identity for the untruncated elliptic profile cannot replace this finite
cube without accounting for the difference.

The shared [dyadic linear reduction](../../src/tnfr/mathematics/_dyadic_linear.py)
retains interval uncertainty during these contractions and the Lobatto
transform. For scale S=`2^128`, represent an admitted interval by integer
center numerator C and radius numerator R over `2S`. For a dot product,
accumulate the signed integer center and a conservative radius

\[
 C_* = \sum_i C_iD_i,\qquad
 R_* = \sum_i\bigl(|C_i|E_i+R_i(|D_i|+E_i)\bigr).
\]

The interval `(C_*+[-R_*,R_*])/(4S^2)` encloses every product sum;
applying an exact rational scale and one final outward projection retains
that enclosure. This does not assume independent errors or remove their
widths. Public consumers admit all primitives before preparing or using
encoded rows. Immutable fixed bases can be reused; evaluated states,
report verdicts and time-dependent phases are not authoritative cache keys.

This reuse changes arithmetic ordering and enclosure widths, not the
finite functions or the analytic interpolation tails. Rational harmonic
and derivative controls must establish the algebra independently; overlap
between old and new interval boxes alone would not prove correctness.
Existing floating FFTs can propose coefficients but cannot replace these
outward reductions. The older Fejer-source spectral factory and its
reflection shortcut have different preparation hypotheses. Direct full
spacetime convolution is also unnecessary here: the finite sine/cosine
polynomials retain their existing nodal Horner evaluation and aliases.

### Why a componentwise box propagation does not transfer

The existing general comparison kernels have finite dimension and derivative
order contracts; increasing their limits would not prove suitability here.
There is also a mathematical obstruction to their coordinatewise absolute
Jacobian treatment over this horizon. Write
`A=I-(P+P^T)/2` on the cycle. The one-sided diagonal/absolute-offdiagonal
majorant of J_sigma maps spatially uniform coordinate radii by

\[
 \binom{r_x}{r_\theta}'=
 \begin{pmatrix}0&2\\2&0\end{pmatrix}\binom{r_x}{r_\theta}.
                                                               \tag{90}
\]

The negative form diagonal cancels its two positive neighbor terms;
`|A|` has row sum two. This radius comparison grows as `exp(2*tau)`
along its positive eigenvector, although the actual tangent is contractive.
For the finite-input Jacobian, `1-C_16(a_M)>1/2` throughout the admitted
gap domain. Its coordinate majorant still dominates the uniform-radius
block `[[0,1],[2,0]]`, with positive growth `sqrt(2)`. Hence replacing the
full-state cancellation by this majorant is unsuitable for the declared
small-error target on T. This excludes a comparison method, not the actual
dynamics, other norms or every possible interval solver.

### A continuous residual budget includes joins and small-norm arithmetic

For piecewise continuously differentiable h_hat on a finite partition,
let `j_i=h_hat(t_i+)-h_hat(t_i-)` be its interior joins. Applying the
same full-interval propagator once to each source and join gives

\[
 \sup_{0\le t\le T}\|h_v(t)-\widehat h(t)\|
 <2^{-71}+2^{-49}+2^9\left[
 \|\widehat h(0)\|+
 \sum_i\int_{t_i}^{t_{i+1}}\|d_i(s)\|\,ds+
 \sum_{i\text{ interior}}\|j_i\|\right].                     \tag{91}
\]

The partition covers the whole physical interval; a final enclosure through
T.hi is sufficient. No repeated gain of 512 is imposed at every step.
Continuous pieces with exactly shared endpoints have zero joins. Recentring
or rounding an endpoint is otherwise an additional source, not free error
removal. The initial term is computed from the approximation even though
the exact correction starts at zero.

The prospective numerical target is `2^-24`, separate from the previous
`2^-40` analytical representation target. A sufficient numerical budget is
that the entire bracket in (91) is at most `2^-34`: its contribution is
`2^-25`, and adding the proved input tails stays below `2^-24`.
With zero initial error and joins, a uniform complete defect below `2^-53`
is sufficient since `T<2^19`. This is an accuracy policy for the linear
comparison, not a proved nonlinear retention margin.

Small defect norms require scaling before squaring. For nonnegative exact
coordinate bounds b_i, set `s=max(b_i)` and use
`s*sqrt(sum((b_i/s)^2))`, with the zero case handled exactly. Rounding an
unscaled square to the absolute `2^-128` grid can introduce a norm floor
of `2^-64`, independently of the actual defect's size. A 128-bit grid
therefore does not by itself certify 128-bit relative norm accuracy.

The fixed input preflight evaluates normalized times `0,1/2,1`, enclosing
the latter two with the same T interval, and requests a midpoint-radius
Euclidean allowance below `2^-80` for the state, derivative, gap and source.
This checks input implementation and arithmetic only. Even exact state and
defect samples cannot prove (91): take a unit constant spatial direction w,
annihilated by the actual full variational matrix, and
`h_hat(t)=a*P(t/T)*w`, with
`P(s)=s^2*(s-1/2)^2*(s-1)^2`. In the homogeneous zero-initial comparison,
both h_hat and its defect vanish at those three samples, while
`||h_hat(T/4)||=9*|a|/4096`. No pointwise tolerance can replace the
between-sample derivative and residual enclosure.

The remaining implementation obligation is a bounded-cost continuous
approximation with a whole-panel signed residual and admitted join errors.
Point input enclosures, finite representability and a small coefficient
rounding error do not supply that approximation or evaluate h_v.

### A direct Taylor realization is sufficient but computationally large

The finite inputs are entire functions of complex time. In a unit complex
disk around any real expansion time, every reconstruction frequency has
absolute value below `3/2`. Since `exp(3/2)<6`, each real row's coefficient
absolute-sum bound is at most `6*(b+A_0*b^3)<1/4`. Consequently the full
complex state norm is below 16, its derivative norm is below 24 and its
phase-gap coefficient norm is below `1/2`. The cosine polynomial implies
`||V_tilde||<1/2`, while `||J_sigma||<17/8`. The complete generator norm
is therefore below three. The nonlinear sine remainder has norm below
`32*(sinh(1/2)-1/2)<1`, giving `||r_tilde||<73<96` on the disk.

Admit a local expansion center with norm at most `2^35`. Variation of
constants along a radius of this disk bounds its analytic solution by
`27*(2^35+96)<Y=2^40`. For its exact Taylor coefficients c_n, Cauchy's
estimate gives `||c_n||<=Y`. Let P_16 be its degree-sixteen polynomial,
R the omitted series and `A(t)=J_sigma+V_tilde(t)`. Its complete signed
equation defect is `-R'+A*R`, so a panel of length delta less than one has

\[
 \int_0^\delta\|d(s)\|\,ds
 \le Y\frac{\delta^{17}}{1-\delta}(1+3\delta).
                                                               \tag{92}
\]

Indeed integrating the derivative tail cancels each coefficient's factor n;
the remaining term is bounded by three times the integrated state tail.
For a fixed grid with panel lengths at most `1/64`, summing (92) gives

\[
 \sum_i\int\|d_i\|
 <2^{40}2^{19}2^{-96}\frac{67}{63}<2^{-36}.                    \tag{93}
\]

This leaves room inside the declared `2^-34` budget for finite coefficient
arithmetic and joins, but does not bound those additional errors. With
exact coefficients, initialize each panel at the preceding polynomial's
endpoint so joins vanish. The exact forced solution satisfies
`sup||h_tilde||<512*64*2^19=2^34`; the defect bound and first-exit induction
therefore keep these exact polynomial centers below `2^35`. Finite
implementations must check their centers and pay their extra defects.

This realization uses step exactly `1/64` except for its final shortened
panel. It requires more than `2^22` panels and at most `2^25`, using
`T>N^2/15` and `T<2^19`. Seventeen coefficients
in each of 2048 real coordinates already entail more than `2^37` scalar
coefficient entries across the run, before series products and validation.
That count excludes a direct full-grid Python rational campaign from the
present implementation route. It is neither a complexity lower bound for
the variational problem nor a failure of its dynamics. A practical method
must reuse time dependence across panels or retain exact tangent transport,
and certify its changed remainder before any full-interval run. Raising
the existing derivative-order or dimension limits alone does not do so.

<a id="chebyshev-panel-validation"></a>
## 16. Signed Chebyshev panels retain the continuous defect

The next representation uses panels of length at most sixteen, with
degree-96 input and candidate polynomials. It preserves the same complete
variational equation and clock. The finite polynomial validator retains
the continuous defect. Canonical input interpolation and a candidate are
evaluated separately in the [first-panel result](#canonical-variational-first-panel);
a manufactured polynomial test alone does not evaluate that correction.

### Source bounds sharpen without a response evaluation

The bound `R_*<32` was deliberately loose. Substitute the primitive upper
bounds `b<=81/(2N)`, `zeta<=9/(2N)`, `D_k<=49/64`, `d<=1/2`, `g<=1/72`,
`A_0<=1/16`, `A_1<=1/8` and `sigma<=16/(5N^2)` into the positive expression
(42), with `B_1=zeta*b/4`, `B_2=zeta^2*b/2`, `B_3=9*zeta^3*b/8` and
`T<8N^2/27`. Exact rational inequalities give

\[
 r_*<2^{-21},\qquad 32r_*T<4,\qquad
 \sup\|\widetilde h\|<512(4+T\epsilon_r)<2^{12}.              \tag{94}
\]

This improves a supplied upper allowance, without evaluating the signed
source or inferring any lower bound on the correction. It does not close a
nonlinear maintenance estimate.

### Uniform input interpolation on the whole panel

Map a panel to `s in [-1,1]`, with `t=t_c+(Delta/2)s`, `Delta<=16`.
Use 97 Chebyshev-Lobatto nodes and the Bernstein ellipse with parameter
`rho_E=1+sqrt(2)`. Its imaginary time semiaxis is at most eight. On that
strip, `q_e*exp(8h_0|v|)<1/19`, because `h_0<1/160` and
`exp(1/20)<20/19`. The positive Fourier coefficients then give
`W(F_M)<(1/32)(1+4/18)<1/24`. Each reconstructed real row has coefficient
norm at most `27/24+3^8/(16*24^3)<5/4`, so the full state norm is below
64 and its derivative norm below 96. The gap norm is below `5/2`.
The finite cosine and sine polynomial majorants consequently imply

\[
 \|\widetilde V(t)\|<32,\qquad
 \|\widetilde r(t)\|<1024 \quad (|\operatorname{Im}t|\le8).
                                                               \tag{95}
\]

For example, `cosh(5/2)-1<13` bounds the cosine polynomial, and
`sinh(5/2)-5/2<11` bounds the sine remainder; the two incidence factors
give the operator factor two. The reconstruction frequency bound `3/2`
controls its derivative. These analytic input extensions do not extend a
physical solution past its admitted horizon.

The [Chebyshev interpolation bound](https://www.chebfun.org/docs/guide/guide04.html#4.5)
is `4M*rho_E^-d/(rho_E-1)` for degree d and an analytic norm bound M
on the ellipse. It also follows directly by bounding the Chebyshev
coefficients using Cauchy's formula and the aliases at Lobatto nodes;
the same triangle inequality applies to vector and operator norms.
Since `(1207/500)^96>2^122` and `sqrt(2)>707/500`, (95) gives the safe
degree-96 interpolation allowances

\[
 \epsilon_{B,\rm panel}<2^{-114},\qquad
 \epsilon_{r,\rm panel}<2^{-109}.                            \tag{96}
\]

Interpolate the scalar edge multiplier before applying D and D^T, and both
signed source rows on the same affine clock. Enclose all sample values and
transform coefficients; (96) bounds analytic interpolation error only.
A floating transform's output is not an outward coefficient enclosure.
Point evaluations already supplied by the input owner can be reused under
these premises. A practical builder may share the fixed spatial roots and
local time offsets. Advancing rectangular interval rotations indefinitely
is not justified: their widths can grow under `|cos|+|sin|` even when the
exact rotation is an isometry. Re-enclose each panel's center rotation or
prove a separate propagation bound.

### Reconstruct the signed equation before bounding its norm

Let the candidate p, signed source r_poly and scalar edge multiplier C_poly
have degree at most 96 in ordinary Chebyshev coefficients, including the
full constant coefficient. Form the complete polynomial residual

\[
 d_{\rm poly}=\frac2\Delta p_s-J_\sigma p
       -(\tfrac12D^T[C_{\rm poly}Dp_\theta],0)+r_{\rm poly}.
                                                               \tag{97}
\]

The identity `T_j*T_k=(T_(j+k)+T_|j-k|)/2` retains every generated
coefficient through degree 192. In particular, projecting the residual
back to degree 96 would silently delete part of the equation being checked.
Differentiate the polynomial in its affine panel coordinate and retain
both mean directions. The shared finite-polynomial owner computes signed
products by exact packed integer convolution of dyadic Laurent coefficients,
then projects each result outwards with the existing interval type.
It neither changes the shared Taylor-jet order limit nor discards a tail.
Explicit factor-degree and packed-bit limits bound its work.

For coefficient vectors p_k and d_k, define
`P_i=sum_k||p_k||_2` and `R_i=sum_k||d_k||_2`, using scaled norms of
the outward coefficient boxes. Since `|T_k(s)|<=1`,

\[
 \int_{t_i}^{t_{i+1}}\|d(t)\|\,dt
 \le\Delta_i\left[R_i+\epsilon_{r,\rm panel}
                              +\epsilon_{B,\rm panel}P_i\right].
                                                               \tag{98}
\]

The [panel owner](../../src/tnfr/physics/_localized_variational_panels.py)
re-admits primitive coefficient tables, duration, sigma and error premises,
and rebuilds (97), coefficient norms and both endpoint enclosures. Internal
algebra reuses the locally normalized rows; dyadic integer norm reductions
preserve the exact scaled rational sum before the shared square root. Its
local result is conditional on the supplied polynomial inputs enclosing
the actual source and Jacobian on the same panel. Default zero input errors
assert exact polynomial inputs; they do not authenticate a canonical
preparation. The validator supplies no response, propagator gain or
retention flag. Initial mismatch and joins still belong to (91).

For example, if every panel satisfies `P_i<=2^35` and `R_i<=2^-54`, (96)
and `T<2^19` give a total integrated defect below
`2^-35+2^-59<2^-34`. With zero initial error and joins, (91) then meets
the declared `2^-24` numerical target. Nonzero initial error and joins must
share the same budget. Neither these policies nor small analytic input
tails show that a candidate passes them.

### Candidate generation and the remaining resource boundary

On a real panel, keep the exact contractive `S(s)=exp(J_sigma*s)` and
iterate only the small Jacobian term in (73), starting at the carried
endpoint. The primitive gap bound and (84) give
`Delta*sup||V_tilde||<16/625<1/32`. This local Volterra map is a contraction;
the same estimate does not establish a full-interval contraction.
For a starting endpoint of norm at most `2^13`, its uniterated source term
has norm below `2^14`. Twenty Jacobian insertions leave a functional-series
tail below `2^14*(1/32)^21/(1-1/32)<2^-90`. This establishes a candidate
route, not an implemented quadrature or a polynomial residual certificate.
Every approximated integral and coefficient must still pass (97)--(98).

A fixed grid of length sixteen, except the last shortened panel, uses at
most `2^15` panels. The validator stores `O(Nd)` coefficients and performs
`O(Nd)` interval bookkeeping and `O(N)` packed integer products per panel;
the bit cost of each product depends on the integer backend and coefficient
magnitudes. No dense spatial matrix or Python quadratic coefficient-product
loop is required. This reduces the prospective number of panels and makes
signed polynomial validation executable. It does not establish the cost of
building 97 canonical input samples or generating a candidate on each panel.

The [polynomial controls](../../tests/mathematics/test_interval_chebyshev.py)
independently check signed products, interval corners, carrying, derivatives
and work limits. The [panel controls](../../tests/physics/test_localized_variational_panels.py)
check whole-polynomial cancellation, time scaling, uncertainty and invalid
domains. Their full-size synthetic countercontrol deliberately omits
generated degree-97 through degree-192 source terms and requires their
nonzero residual to survive. This is software evidence, not a canonical
variational correction. Section 17 evaluates the first canonical panel;
a resource-admitted full interval remains pending.

<a id="canonical-variational-first-panel"></a>
## 17. Canonical first-panel construction and result

The bounded evaluation fixes the first normalized-clock panel `[0,16]`,
the same C1024, m=1/2 reference, degree 96, and zero initial correction.
The [input builder](../../src/tnfr/physics/_localized_variational_panel_inputs.py)
prepares its own authoritative finite-input closure and encloses all 97
Lobatto samples. The shared transform uses ordinary Chebyshev coefficients,
including the full constant term:

\[
 a_k=\frac{2w_k}{d}\sum_{j=0}^d w_j f_j\cos(kj\pi/d),
 \qquad w_0=w_d=1/2,\quad w_j=1\ (0<j<d).
                                                               \tag{99}
\]

Dyadic integer center/radius dot products enclose every signed sum before
one final interval projection per coefficient. Sampling uncertainty and
transform roundoff remain in the coefficient boxes; the analytic tails
(96) remain separate. A finite sampled polynomial alone supplies no such
between-node tail for an arbitrary function.

### A proposal method with independent posterior validation

For a degree-d row f, define B_d f as its antiderivative projected to
degree d and re-centered to vanish at s=-1. Its retained coefficients are
`b_1=f_0-f_2/2`, `b_j=(f_(j-1)-f_(j+1))/(2j)` for `2<=j<=d`, and
`b_0=-sum_(j=1)^d (-1)^j b_j`, with unavailable coefficients zero.
For d=1, the first formula gives b_1=f_0. The discarded exact-integral term
is `f_d*(T_(d+1)-(-1)^(d+1)T_0)/(2(d+1))`. Its derivative is part of the
posterior equation defect; projecting the proposal does not delete it
from validation.

Let `q=sqrt(1-sigma^2/4)`, `alpha=sigma/2+iq` and
`mu=-sigma/2+iq`. The complex encoding `psi=x+alpha*theta` gives
`psi'=mu*A*psi` for the constant tangent, since `mu*alpha=-1`.
The inverse is `theta=Im(psi)/q`, `x=Re(psi)-(sigma/2)*theta`.
A full complex spatial FFT retains all 2N real coordinates; no same-mode
conjugacy or half-spectrum omission is imposed. For cycle eigenvalue
`lambda_j=1-cos(2*pi*j/N)`, factor the temporal matrix

\[
 I-(\Delta/2)\lambda_j\mu B_d.                              \tag{100}
\]

There are at most N/2 distinct nonzero factors on the even cycle. These
small temporal matrices are separate from a dense full-state generator.
Pivoted factorization avoids dividing by a frequency difference. Reuse
the factors for the signed source solve and twenty fixed feedback updates.
Each feedback product is evaluated with 2d+1 Lobatto points, retaining
its degree-2d polynomial before projecting the next proposal to degree d.

The [candidate owner](../../src/tnfr/physics/_localized_variational_candidate.py)
uses this finite iteration count as a proposal policy. The exact-Volterra
contraction and insertion tail in section 16 do not certify the projected
iteration or its floating arithmetic. Binary64 produces only candidate
coefficients; re-admit their represented values as exact rationals and set
their constant coefficients so the proposed initial state is exactly zero.
The independent outward validator then reconstructs both signed equation
rows, all degree-192 products and the real full-state norm. Its acceptance
depends on that rebuilt defect, not a solver tolerance or iteration delta.

The prospective local policy is `P<=2^35`, `R<=2^-54` and initial endpoint
enclosure norm at most `2^-70`, with the separate analytic input errors in
(96). Record input, proposal and validation costs separately. Keep the
first outcome, including failure, without changing the profile, degree,
iteration count, precision or panel after observing it. This local gate
does not evaluate the rest of T, its joins, a nonlinear bootstrap or a
source-to-composite response. A full-interval cost estimate is a resource
assessment, not a prediction that later panels will pass.

### Retained local result and cost boundary

The [frozen protocol](../../docs/assets/localized_variational_panel/protocol.json)
declares this single evaluation before its response. The
[freeze](../../docs/assets/localized_variational_panel/freeze.json),
[source manifest](../../docs/assets/localized_variational_panel/source-manifest.json)
and [source archive](../../docs/assets/localized_variational_panel/source.zip)
associate the evaluator, runtime and pre-evaluation proof bytes. They do not
authenticate chronology or execution. The first attempt is preserved with
its [primitive response](../../docs/assets/localized_variational_panel/response.zip)
and [result](../../docs/assets/localized_variational_panel/report.json).

The numerical policy passes on this panel. Rounded-up decimal bounds are:

| Whole-panel quantity | Upper bound |
| --- | --- |
| Candidate full-state norm P | `9.913e-6` |
| Complete polynomial defect norm R | `3.005e-21` |
| Integrated defect, including analytic input tails | `4.808e-20` |
| Initial enclosure norm | `4.145e-36` |
| Finite-input correction error, `512*(initial + integrated defect)` | `2.462e-17` |

The candidate has exactly zero initial state as rational coefficients;
the small enclosure width is nevertheless charged to the error allowance.
All degree-192 terms remain in the residual. The last row bounds error
to the finite-input linear correction on `[0,16]`, conditional on the
retained input enclosures and section 16's analytic tails. It does not
include the earlier exact-model input truncation allowances in (88),
and is not an error bound for a nonlinear trajectory. The source and edge
coefficient-radius summaries are respectively below `8.858e-33` and
`1.527e-35`; these are arithmetic widths, not the analytic input tails.

The resource policy fails its input threshold: preparation took 666.814 s
against 600 s. Of that cost, sampling took 648.298 s and interpolation
16.842 s. The proposal took 5.521 s and interval validation 37.433 s,
within their respective 180 s and 120 s thresholds. Total recorded time,
including artifact I/O, was 713.873 s. The thresholds assess the completed
attempt; they are not abort timers or mathematical restrictions on the law.
The length-sixteen grid, with a shortened last panel, has 8,988 panels over
the exact horizon. Repeating this observed total cost would take about
74.3 sequential days; this is a
conditional cost extrapolation, not a runtime bound or evidence that later
panels satisfy the numerical policy.

The [input controls](../../tests/physics/test_localized_variational_panel_inputs.py)
check canonical preparation, clocks, interpolation and separate error channels.
The [candidate controls](../../tests/physics/test_localized_variational_candidate.py)
compare independent dense small-cycle equations, retain all spatial modes
and reject equating a projected fixed point with zero continuous defect.
The [evidence audit](../../tests/physics/test_localized_variational_panel_evidence.py)
checks archive associations and re-admits primitive coefficients to rebuild
the signed defect and verdict without running any canonical producer.
It does not independently regenerate the retained sample enclosures.

This closes one local construction and its numerical assessment. The
retained first attempt starts at zero; repeating that preparation would
reset the correction and would not solve the full-interval problem.
The maintained input reuse and carried-state construction below are
separate implementation improvements. Initial mismatches, join defects and
every panel's signed residual must share (91)'s budget. The
[deferred-dependency contract](../research/FIVE_STAGE_EXECUTION_PLAN.md#localized-variational-resource-dependency)
in the sole execution plan owns reactivation and further admission.
No whole-horizon correction, nonlinear maintenance,
composite acquisition or physical identification follows from this panel.

### Exact endpoint carry without resetting accumulated error

The maintained [candidate owner](../../src/tnfr/physics/_localized_variational_candidate.py)
also accepts two incoming scalar rows. It admits their exact or represented
values before numerical work; an uncertain interval is not an incoming
point. Add the full spatial Fourier transform of
`psi_0=x_0+alpha*theta_0` to the constant temporal coefficient on the
right-hand side of (100). The same modal factors and twenty feedback
updates then propose the carried panel. The zero-initial entry point
delegates to this constructor and preserves its declared preparation.

After admitting the proposed tail coefficients as rationals, set
`a_0=y_0-sum_(k=1)^d (-1)^k*a_k` separately in each row. Thus the left
endpoint is exactly the supplied y_0 even when its rational representation
is not binary64. Carrying the preceding polynomial's exact rational right
endpoint `sum_k a_k` makes the mathematical join exactly zero. An interval
midpoint or a rounded endpoint is a different preparation whose mismatch
must be charged; overlapping endpoint boxes do not prove a zero join.

This does not make the incoming approximation equal to the true solution.
Equation (91) retains its existing global error and applies gain 512 once
to initial error plus all integrated defects and join norms. Neither a
restart nor repeated multiplication of that gain is justified. Independent
small-system controls check both real rows, nonzero means, homogeneous
damped transport and two-panel joins. They do not evaluate a second
canonical panel or admit its precision.

### Input-reuse cost pilot and result

The implementation pilot keeps the first panel, all 97 sampling
locations, degree 96 and the analytic input errors fixed. It builds only
the optimized source and edge interpolants, then revalidates the exact
candidate already stored in the first response. No candidate generator,
old frozen producer or nonlinear trajectory is executed. Reusing those
coefficients tests changed input arithmetic without generating a new
scientific response or rewriting the first outcome.

The numerical policy remains `P<=2^35`, `R<=2^-54`, exact zero initial
state and initial enclosure norm at most `2^-70`. The prospective local
resource thresholds are 180 s for inputs and 120 s for validation. Retain
the first outcome without automatic retries. Report cold preparation,
sampling and transformation separately. An explicit full-grid resource
scenario uses at most 24 sequential hours and 8 GiB of retained artifacts
as engineering policies, not dynamical hypotheses. Its time extrapolation
combines the new input/validation measurements with the old proposal cost;
its storage extrapolation uses the original per-panel response size.
Neither is a bound or a full-run admission. Repeated all-panel snapshots
and simultaneous allocation of the whole coefficient history are not
required by (91); a future storage design must retain auditable primitives
and checkpoints within a declared budget.

The [separate protocol](../../docs/assets/localized_variational_reuse/protocol.json),
[freeze](../../docs/assets/localized_variational_reuse/freeze.json),
[source manifest](../../docs/assets/localized_variational_reuse/source-manifest.json)
and [source archive](../../docs/assets/localized_variational_reuse/source.zip)
retain this implementation and associate all eight previous evidence files.
Its first [response](../../docs/assets/localized_variational_reuse/response.zip)
and [report](../../docs/assets/localized_variational_reuse/report.json)
pass every local numerical and resource check. The same retained candidate
still has finite-input correction error below `2.462e-17` on `[0,16]`.
This is a revalidation of its existing polynomial, not a newly generated
variational response or an improvement to the physical model.

| Measured phase | Original attempt (s) | Input-reuse pilot (s) |
| --- | --- | --- |
| Complete input construction | 666.814 | 109.466 |
| Input sampling | 648.298 | 90.324 |
| Input interpolation | 16.842 | 15.548 |
| Complete interval validation | 37.433 | 36.908 |

Complete input construction is about 6.09 times faster in these recorded
runs. The new cold factory cost is 2.042 s, including preparation of the
100,352 encoded spatial entries; the cached basis uses three integers per
entry. Its memory tradeoff is explicit. Different process conditions may
change these timings; the single retained comparison is not a universal
speed guarantee. The pilot's 150.687 s total excludes candidate generation,
so it is not directly comparable to the original complete-attempt total.

The mixed-cost full-grid scenario still gives about 15.80 sequential days
before artifact I/O and 59.16 GiB at the original response size. Both exceed
their stated 24-hour and 8-GiB engineering policies. The new archive omits
the reused candidate and must not be used as a complete per-panel storage
estimate. Input sampling remains material; transformation, validation and
evidence storage also need explicit treatment before a full run. The
[read-only audit](../../tests/physics/test_localized_variational_reuse_evidence.py)
checks the two evidence associations and rebuilds the complete signed
defect and policy assessments from retained primitive coefficients. It
does not regenerate inputs, candidates or old responses.

## 18. Admission result and execution boundary

The original symmetry and separated carrier energies do not supply a
discriminating formation mechanism. Positive curvature at an exact lower
carrier supplies a concrete phase-sensitive envelope balance, generated
harmonics and a conditional complete-law validation method. This identifies
a candidate nonlinear retention relation needing a separate preparation.
The coarse acute-chart method is excluded on a full nonlinear phase time,
so a different proved stability or residual treatment is needed before a
quantitative retention certificate. Equations (25)--(35) now give a
complete-law relative-storage comparison with explicit means, metric
positivity, moving coefficients, residuals and positive loss. The carrier
resonance and small-divisor calculation require retaining a near-resonant
block; exact-resonance-only cancellation is not uniformly a small positive
correction. This resolves a method-admission question, not its quantitative
use for a localized reference. Equations (40)--(44) now supply uniform
profile, retained-block and moving-metric bounds. The specified scalar
defect allowance prevents a full phase-time certificate even if all other
growth is suppressed; sharpening resonance constants or source precision
cannot resolve this particular failure. Equations (49)--(63) provide a
different treatment: signed two-row harmonic transport, explicit aliases
and tails, a response-corrected reference and a full Euclidean comparison
without spectral-gap amplification. Its transported-defect and nonlinear
budgets still need a quantitative closure before any maintenance claim.
The canonical-profile preflight (64)--(72) now excludes the specified
positive coefficient-norm implementation of that comparison before a
coefficient or trajectory campaign. Signed phase transport and nonlinear
work correlations remain unassessed by this obstruction.
The full variational comparison now has a finite phase-resolved
representation with explicit profile, Jacobian, source and chronological
tails, whose combined analytical error is below `2^-48` on the fixed
phase interval. Its finite inputs and sparse action have a
[shared private implementation](../../src/tnfr/physics/_localized_variational_inputs.py).
The [Chebyshev panel validator](#chebyshev-panel-validation) now retains the
signed continuous defect and all generated degrees, with explicit uniform
input interpolation bounds. Raising the general comparison solver's
dimension cap would lose the cancellations, and the admitted direct Taylor
realization has excessive work for this route. The
[first canonical panel](#canonical-variational-first-panel) now passes its
numerical policy, while its input construction exceeds the local cost
threshold. Shared signed input arithmetic and exact nonzero endpoint carry
now have maintained implementations. The separate reuse pilot passes its
local policies, while its full-grid time/storage scenario still fails.
Full-interval resource admission and a canonical joined correction remain
unresolved. Their [reactivation conditions](../research/FIVE_STAGE_EXECUTION_PLAN.md#localized-variational-resource-dependency)
belong to the sole execution plan; the calculation is deferred, with its
shared implementations and frozen evidence retained. The whole-horizon
correction and nonlinear maintenance bound remain unevaluated.
No result here proves that two patterns acquire the relation.

The [static controls](../../tests/physics/test_localized_retention_mechanisms.py)
derive the local balance, cubic harmonics, correctors and envelope identities
independently. The [relative-storage controls](../../tests/physics/test_localized_relative_storage.py)
check the two-row balance, missing means, coercivity, resonant production
and small-divisor obstruction. The [periodic-bound controls](../../tests/physics/test_localized_envelope_bounds.py)
check derivative budgets and the scalar-defect exclusion. The
[phase-defect controls](../../tests/physics/test_localized_phase_defect.py)
check actual-loss modal transport, two-row projection, aliases, signed
convolutions, correction residuals and full-state metric identities.
The [defect-budget controls](../../tests/physics/test_localized_defect_budget.py)
check the independent profile averages, source-moment condition, resolvent
floor and strict rational obstruction, including its distinction from an
actual oscillatory response.
The [variational controls](../../tests/physics/test_localized_variational_transport.py)
check chronological order, coupled-mode structure, resonance limits,
factorial and coefficient tails, propagation gains and separated input
errors. Their arithmetic controls reuse the existing positive-series tail
owner; they do not instantiate the large finite variational sum.
The [finite-input controls](../../tests/physics/test_localized_variational_inputs.py)
check independent reconstruction derivatives, sparse actions, scalar and
clock admission, the declared three-point arithmetic preflight and the
distinction between samples and a continuous residual certificate.
They execute neither a trajectory nor frozen evidence.
The actual rate implementation remains the
[shared sine owner](../../src/tnfr/physics/relational_sine_comparison.py);
transport and nonlinear budgets retain their
[shared comparison owner](../../src/tnfr/physics/_localized_identity_transport.py).
No envelope runtime or replacement nodal law is installed. Quantitative
maintenance, a source-to-composite experiment and physical identification
remain separate obligations in the sole execution plan.
