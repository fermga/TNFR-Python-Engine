# Internal replica pulses and their perturbations

The prepared internal pulse, complete variational law, small-amplitude splitting and persistent constituent activity under the same conservative doubled-cycle model.

Part of [Coarse-graining, coherence geometry and bridge results](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-replica-internal-pulse"></a>

## 9. A finite-amplitude internal pulse with persistent collective identity

Keep the complete double-replica `C5`, the same conservative sine law
`e=0,w,beta>0`, no inputs or events, and now a **common** strictly
positive held capacity `nu` at all ten fine nodes. This additional
capacity premise is needed for the invariant family below; paired
equality alone was enough for sections 7 and 8.

### 9.1. The exact invariant preparation

Let `alpha=2*pi/5` and `c_alpha=cos(alpha)>0`. Supply the exact
collective state

\[
X_i=\bar X,\qquad
\Theta_i=\bar\Theta+i\alpha\pmod{2\pi},
\qquad i=0,\ldots,4,
\]

and identical signed internal coordinates

\[
u_i=u,\qquad \delta_i=\delta
\]

for all five pairs. The constants `bar X` and `bar Theta` specify the
common form and phase origins. The target angles are exact turns `i/5`;
replacing them by rounded radian values and accepting a small balance
residual would not establish this invariant family.

At each coarse node the two phase differences are `+alpha` and
`-alpha` modulo `2*pi`. Their sine currents cancel, while their
cosines add. The exact retained-coordinate rows of section 7 therefore
give

\[
\dot X_i=0,\qquad \dot\Theta_i=0,\qquad
\dot u_i=-a\nu c_\alpha\cos\delta\sin\delta,\qquad
\dot\delta_i=b\nu u.
\]

The internal derivatives are identical at all five pairs, so smooth
uniqueness proves that the full preparation remains in this family.
Its complete nonlinear evolution reduces exactly to

\[
\boxed{\quad
\dot u=-a\nu c_\alpha\cos\delta\sin\delta,\qquad
\dot\delta=b\nu u .
\quad}
\]

The ten constituent coordinates remain
`x_i,plus=bar X+u`, `x_i,minus=bar X-u` and
`theta_i,plus/minus=bar Theta+i*alpha+/-delta`. Static collective
means do not mean that the constituents are stationary. Neither an
external periodic input nor an additional oscillator equation has been
installed: the two displayed rows are an invariant restriction of the
already supplied complete law.

### 9.2. Conserved amplitude and exact periods

Because `a=beta*b`, the internal quantity

\[
\boxed{\quad H=u^2+\beta c_\alpha\sin^2\delta\quad}
\]

is conserved:

\[
\dot H
=-2a\nu c_\alpha u\cos\delta\sin\delta
 +2\beta b\nu c_\alpha u\sin\delta\cos\delta=0.
\]

It is part of the existing full storage, not a second independently
postulated energy. Indeed, section 7 gives exactly

\[
E_f=4E_{*,C5}+20H,\qquad
E_{*,C5}=5\beta(1-c_\alpha).
\]

Define

\[
\Omega=\nu\sqrt{ab c_\alpha}
       =\frac{w\nu\sqrt{c_\alpha}}{\pi\sqrt\beta},
\qquad
m=\frac{H}{\beta c_\alpha}.
\]

For `0<m<1` and a preparation in `|delta|<pi/2`, the level set is a
nonstationary closed curve around `(u,delta)=(0,0)`. Its amplitudes are

\[
|\delta|_{\max}=\arcsin\sqrt m<\pi/2,\qquad
|u|_{\max}=\sqrt H .
\]

The pair chart consequently remains valid for the entire motion. Set
`eta=2*delta`; differentiating the actual phase row gives

\[
\ddot\eta+\Omega^2\sin\eta=0 .
\]

This pendulum equation is a derived form of the retained internal
system. The conserved level already proves that the orbit is a
libration, and direct quadrature gives its period. With the elliptic
**parameter** `m`, define

\[
K(m)=\int_0^{\pi/2}\frac{d\psi}{\sqrt{1-m\sin^2\psi}}.
\]

Since `sin(delta)=sqrt(m)*sin(psi)` on a quarter orbit,

\[
\boxed{\quad
T_{\rm labeled}=\frac{4}{\Omega}K(m)
 =\frac{4\pi\sqrt\beta}{w\nu\sqrt{c_\alpha}}K(m).
\quad}
\]

This is the period of the full labeled circular fine state, not merely
of a tangent mode. On a nonzero libration the half-period map is
`(u,delta)->(-u,-delta)`. It swaps the two members of every pair while
leaving `X,Theta` fixed. The unordered state of section 8 therefore has
the smaller fundamental period

\[
\boxed{\quad T_{\rm unordered}=T_{\rm labeled}/2.\quad}
\]

To see why it is the fundamental unordered period, the nonstationary
closed energy curve is traversed once in `T_labeled` and each unordered
state has exactly two fine lifts on it. Those lifts are related by the
half-period swap; there is no fixed lift on a positive-energy orbit.
The invariants `R,U,Q` change along this orbit, so the unordered state
is not stationary despite the fixed collective means.

The same elementary integral bounds used for the
[two-node pulse](RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
give

\[
T_0=\frac{2\pi}{\Omega},\qquad
T_0<T_{\rm labeled}\le\frac{T_0}{\sqrt{1-m}},
\]

and half of each bound for the unordered period. The strictly positive
amplitude is essential: `T_0` is the small-amplitude limiting period,
not a period assigned to a stationary state. In particular, the pulse
frequency depends on the supplied capacity, coefficients, geometry and
amplitude. As `m` approaches one, `K(m)` diverges. This is a persistent
periodic internal motion under a declared law, not a universal constant
frequency.

### 9.3. Collective identity and the stronger fine-edge condition

Throughout every admitted libration, the collective means remain exactly
the winding-one target. The inherited phase-current factor
`R_i*R_j=cos(delta)^2` varies periodically on every coarse edge, while
its two incident sine currents still cancel at each coarse node.
Consequently a static collective mean can contain changing internal
interaction strength. The fine phase differences along an oriented
base edge are

\[
\alpha,\qquad \alpha+2\delta,\qquad \alpha-2\delta
\pmod{2\pi}.
\]

Preserving the nonantipodal pair chart is therefore weaker than keeping
every fine edge acute. The latter holds for the complete period under
the sharper condition

\[
|\delta|_{\max}<\frac{\pi/2-\alpha}{2}
\quad\Longleftrightarrow\quad
\boxed{\quad m<\sin^2(\pi/20).\quad}
\]

This nonempty finite-amplitude subfamily has fixed collective winding,
moving constituents and strictly acute fine edges at all times. Its fine
cycle windings also retain those of the duplicated target: continuously
varying strictly acute edges cannot cross an antipodal wrapping boundary.
This conclusion follows from the exact orbit and its amplitude bound;
it does not require a numerical trajectory.

For larger librations with `0<m<1`, the static collective target and
valid pair chart still follow, but the all-fine-edge acute certificate
does not. A loss of that sufficient condition is not by itself a
formation, instability or winding-change verdict. General transverse
perturbations away from the common-internal-state family remain governed
by the full retained law and its independent
[trapping conditions](SINE_PAIR_STATE.md#sine-replica-inheritance). Exact period, common
internal phase and the half-period swap cannot be inferred for those
perturbations from an energy bound alone.

### 9.4. Boundaries and scientific scope

Within the admitted pair chart:

- `H=0` forces `u=delta=0` and is stationary. No pulse starts
  spontaneously from that exact equilibrium.
- `H=beta*c_alpha` is the separatrix level. Nonstationary preparations
  inside the chart approach its boundary `|delta|=pi/2` asymptotically
  and do not have a finite libration period. The boundary states
  `u=0,delta=+/-pi/2` are stationary states of the full fine law,
  outside this pair-midpoint chart.
- `H>beta*c_alpha` gives
  `u^2>=H-beta*c_alpha>0`, so `delta` moves monotonically and leaves
  the admitted pair chart in finite time. This excludes this
  **libration certificate**, not every periodic or rotating full
  circular-state solution. The full fine law continues smoothly and
  would need another observation chart.

The result establishes a family of finite-amplitude internal pulses
inside a persistent collective organization. The common capacity, exact
twist, zero loss and identical signed internal preparation are explicit
premises. This invariant family has lower dimension than the ambient
fine state space; its exact periodic behavior is not an almost-everywhere
or generic-attraction theorem.

Nothing here selects microscopic zero loss, supplies the initial
amplitude, forms the paired support, forces all internal modes to align,
or identifies a material particle. The pulse is sustained by conservative
exchange already present in the chosen nodal law, and the distinction
between labeled and unordered periods follows from its exact symmetry.

The [existing scale owner](../../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_pulse` for this declared exact family, with
the common signed internal preparation and capacity supplied explicitly.
It retains symbolic target turns and marks captured-graph membership as
uncertified. Interval bounds classify the internal energy, enclose the
two periods and independently test the stronger all-fine-edge acute
condition; an unresolved boundary does not become a passing certificate.
The period enclosure reuses the
[shared elliptic-integrand bound](../../src/tnfr/physics/relational_sine_resonance.py).
The [replica tests](../../tests/physics/test_relational_sine_replica.py)
compare full nodal rows, exact storage identities and independent
elliptic-function values without an ODE trajectory or a rounded-target
membership assertion.

<a id="sine-replica-pulse-variation"></a>

## 10. Complete variation around the finite internal pulse

Retain the exact doubled-`C5` pulse and all hypotheses of
[section 9](#sine-replica-internal-pulse). The instantaneous variational
identities hold at any declared point of its invariant family while the
pair chart is valid. The finite-period return statements additionally
require the nonstationary libration condition `0<m<1`. Write the
reference internal phase as `d(t)` to distinguish it from a perturbation,
and put

\[
\alpha=\frac{2\pi}{5},\quad c=\cos\alpha,\quad s=\sin\alpha,\quad
C(t)=\cos d(t),\quad S(t)=\sin d(t).
\]

The perturbations below keep the supplied graph, common held capacity
and law fixed. They include all twenty real nodal-state directions, not
just perturbations that preserve the common signed internal preparation.
They do not include perturbations of the support or constitutive laws.

### 10.1. Direct differentiation of the full retained law

Perturb `(X_i,Theta_i,u_i,delta_i)` by `(xi_i,eta_i,v_i,zeta_i)`.
On the ordered cycle define

\[
(Lf)_i=2f_i-f_{i+1}-f_{i-1},\qquad
(Df)_i=f_{i+1}-f_{i-1},
\]

with indices modulo five. Differentiating the complete rows of section 7,
before restricting a spatial mode, gives

\[
\boxed{\begin{aligned}
\dot\xi&=-\frac{a\nu}{2}
       [cC^2L\eta+sCS\,D\zeta],\\
\dot\eta&=\frac{b\nu}{2}L\xi,\\
\dot v&=-a\nu c
       [\cos(2d)I+\tfrac12S^2L]\zeta
       +\frac{a\nu sCS}{2}D\eta,\\
\dot\zeta&=b\nu v .
\end{aligned}}
\]

For example, the derivative of the mean form row contains the neighbor
internal-phase term
`-(a*nu/2)*C*S*sum_j sin(Theta_j-Theta_i)*zeta_j`.
The two target sine values are `+s,-s`, producing `D*zeta`.
The term from differentiating the local `cos(delta_i)` vanishes
because the unperturbed incident sine sum is zero. Differentiating the
internal form row gives the opposite signed `D*eta` term and the
coefficient
`C^2*zeta_i-(S^2/2)*(zeta_(i+1)+zeta_(i-1))`, equal to the displayed
`cos(2d)I+(S^2/2)L` expression.

Thus transverse perturbations couple internal and collective coordinates.
Treating the five pairs as independent pendula would omit the two
`D` terms and generally change the variation being tested.

### 10.2. Exact spatial decomposition without discarding directions

Use the Fourier convention `f_i=hat f_k*exp(i*q_k*i)`, with
`q_k=2*pi*k/5`. Then

\[
\lambda_k=2-2\cos q_k,\qquad
L\mapsto\lambda_k,\qquad D\mapsto2i\sin q_k .
\]

In the order `(hat xi,hat eta,hat v,hat zeta)`, the complex block is

\[
\mathcal A_k(t)=
\begin{pmatrix}
0&-a\nu cC^2\lambda_k/2&0&-ia\nu sCS\sin q_k\\
b\nu\lambda_k/2&0&0&0\\
0&ia\nu sCS\sin q_k&0&
   -a\nu c[\cos(2d)+S^2\lambda_k/2]\\
0&0&b\nu&0
\end{pmatrix}.
\]

Real nodal perturbations obey `hat f_(5-k)=conjugate(hat f_k)`.
For each representative `k=1,2`, change only the analysis basis to

\[
(\hat\xi_k,\hat\eta_k,i\hat v_k,i\hat\zeta_k).
\]

The resulting real-coefficient matrix is

\[
\boxed{
A_k(t)=
\begin{pmatrix}
0&-a\nu cC^2\lambda_k/2&0&-a\nu sCS\sin q_k\\
b\nu\lambda_k/2&0&0&0\\
0&-a\nu sCS\sin q_k&0&
   -a\nu c[\cos(2d)+S^2\lambda_k/2]\\
0&0&b\nu&0
\end{pmatrix}.}
\]

Its real and imaginary parts follow two identical four-real-dimensional
systems. This is a basis transformation, not complexification of the
scalar physical form coordinate. Mode `k=0` uses the original four real
coordinates and the same displayed matrix with `lambda_0=sin(q_0)=0`.
The dimensions are exactly

\[
4+2\cdot4+2\cdot4=20 .
\]

No relative mean, hidden capacity, fine coordinate or spatial perturbation
has been dropped. Capacities are held parameters of the stated model,
so they are not additional tangent state directions.

The common mode contains two constant origin perturbations
`xi_0,eta_0` and the pendulum variation

\[
\dot v_0=-a\nu c\cos(2d)\zeta_0,\qquad
\dot\zeta_0=b\nu v_0 .
\]

The nonzero spatial modes are the sixteen real directions transverse to
the common-preparation family. They are the relevant blocks when asking
whether arbitrary fine perturbations preserve internal coordination.

### 10.3. Zero-amplitude control and dimensionless parameters

At `u=d=0`, `C=1,S=0` and the internal-collective cross terms vanish.
The exact spectral control is

\[
\text{mean sector: }\quad
\pm i\Omega\,\lambda_k/2,\qquad
\text{internal sector: }\quad \pm i\Omega,
\qquad \Omega=\nu\sqrt{ab c}.
\]

Mode zero contributes two zero origin modes instead of a nonzero mean
frequency. The two nonzero mean-frequency ratios are

\[
\frac{\lambda_1}{2}=\frac{5-\sqrt5}{4},\qquad
\frac{\lambda_2}{2}=\frac{5+\sqrt5}{4}.
\]

Including Fourier multiplicities, there are two zero directions, ten
internal oscillator directions and eight nonzero mean oscillator
directions. This agrees with the full fine-graph decomposition in
section 7. All are a **stationary-target** control or a small-amplitude
limit. A nonstationary finite-amplitude period is not assigned to the
zero-amplitude equilibrium, and this limiting spectrum does not decide
finite-amplitude stability.

There is also an exact reduction of the apparent parameter freedom.
Divide both form perturbations by `sqrt(beta*c)`, leave phase
perturbations unchanged, and use `tau=Omega*t`. The block becomes

\[
\boxed{
\widetilde A_k(\tau)=
\begin{pmatrix}
0&-C^2\lambda_k/2&0&-\tan\alpha\,CS\sin q_k\\
\lambda_k/2&0&0&0\\
0&-\tan\alpha\,CS\sin q_k&0&
   -[\cos(2d)+S^2\lambda_k/2]\\
0&0&1&0
\end{pmatrix}.}
\]

The base pulse itself satisfies

\[
\frac{dd}{d\tau}=\widehat u,\qquad
\frac{d\widehat u}{d\tau}=-\cos d\sin d,\qquad
\widehat u=\frac{u}{\sqrt{\beta c}},\qquad
m=\widehat u^2+\sin^2d .
\]

Consequently the return multipliers depend only on `m` and the fixed
geometry. Changing the reference position on
the same orbit conjugates the return matrix and does not change its
spectrum. At fixed `m`, `nu,w,beta` change the time or coordinate
scales, not independent stability parameters. Changing `beta` while
holding raw `u,d` fixed usually changes `m`, so it is not that
fixed-amplitude comparison.

### 10.4. Symplectic blocks and the correct half-period return

Let

\[
J_0=\operatorname{diag}
\left(
\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\begin{pmatrix}0&-1\\1&0\end{pmatrix}
\right).
\]

For every displayed real block, `-J_0*A_k(t)` is symmetric. Equivalently

\[
A_k(t)^TJ_0+J_0A_k(t)=0 .
\]

Thus a fundamental matrix `Phi_k(t)` initialized by `Phi_k(0)=I`
satisfies `Phi_k(t)^T*J_0*Phi_k(t)=J_0`. This is a tangent consequence
of the same conservative law, not an auxiliary Hamiltonian added to the
fine evolution.

Write `T=T_labeled` and `h=T/2`. At the half period the base pulse has
`(u,d)->(-u,-d)`. The induced member-swap derivative is

\[
S_*=\operatorname{diag}(1,1,-1,-1),\qquad S_*^2=I .
\]

Since `C` is unchanged while `S` changes sign,

\[
A_k(t+h)=S_*A_k(t)S_* .
\]

The raw matrix is therefore generally **not** half-periodic: its two
cross terms reverse sign. They happen to vanish in mode zero, which
does not remove the swap identification of the base state.

The full labeled return and the symmetry-correct unordered return are

\[
\boxed{\qquad
M_k=\Phi_k(T),\qquad
B_k=S_*\Phi_k(h),\qquad M_k=B_k^2 .
\qquad}
\]

Indeed the second-half propagator is `S_*Phi_k(h)S_*`, and multiplying
it by the first-half propagator proves the square identity. Applying
`S_*` after the first half is essential: it identifies the tangent
space at the swapped endpoint with the original one. A spectrum of the
raw `Phi_k(h)` alone is not the unordered return spectrum.

Both `S_*` and `Phi_k(h)` are symplectic, hence so are `B_k` and
`M_k`. Their determinant is one and their multipliers occur in
reciprocal and complex-conjugate pairs. These restrictions are exact;
they supply no numerical value for a nontrivial finite-amplitude
multiplier.

### 10.5. Amplitude-dependent phase shear is a neutral-family effect

The period is strictly increasing with the conserved internal amplitude:

\[
\frac{dT}{dH}
 =\frac{4}{\Omega\beta c}K'(m)>0,\qquad
K'(m)=\frac12\int_0^{\pi/2}
 \frac{\sin^2\psi}{(1-m\sin^2\psi)^{3/2}}\,d\psi .
\]

This follows by differentiation under the integral for `0<m<1`. For
example,

\[
\frac{\pi}{2\Omega\beta c}
 <\frac{dT}{dH}
 \le\frac{\pi}{2\Omega\beta c(1-m)^{3/2}} .
\]

To identify its tangent effect, choose the smooth initial section
`d(0)=0,u(0)=sqrt(H)` with the common origins fixed. Let `v_t` be
the flow tangent there and `v_H` the derivative of that initial state
with respect to `H`. They are independent for `H>0`. Differentiate
the exact identity `z(H,T(H))=z(H,0)` to obtain

\[
M_0v_t=v_t,\qquad
M_0v_H=v_H-T'(H)v_t.
\]

Differentiating the corresponding half-period swap gives

\[
B_0v_t=v_t,\qquad
B_0v_H=v_H-\tfrac12T'(H)v_t.
\]

The two common origins have identity returns as well. Thus the common
mode has a nontrivial neutral time-amplitude shear. Repeated comparison
at the same clock phase can accumulate a linear-in-cycle phase shift
between neighboring amplitudes, even though each preparation remains
on its own conserved libration. This is not an exponentially growing
transverse mode or proof of nonlinear orbital instability. Fixing
`H` removes that amplitude direction; comparing only equally timed
waveforms does not.

### 10.6. A discriminating stability question, not a stability verdict

After the common-mode directions have been identified, the prospective
linear test is confined to the `k=1,2` representative blocks:

- A certified return multiplier `|lambda|>1` yields exponential
  transverse growth. For the smooth periodic orbit here, an unstable
  multiplier of its transverse return map also certifies nonlinear
  orbital instability. The criterion can be checked through `B_k`
  or `M_k=B_k^2`, with the correct return convention.
- If all transverse multipliers lie on the unit circle and their return
  matrices are semisimple, the transverse linear solutions are bounded:
  powers of the return matrices are bounded, and the continuous
  propagator over one compact period is bounded.
- Unit-circle eigenvalues with nontrivial Jordan blocks can yield
  polynomial growth and do not pass that bounded-linear criterion.
  Neutral amplitude shear in the separately identified common family
  must not be counted as a transverse instability.

Bounded linear variation alone does not prove nonlinear orbital
stability, attracting synchronization or retention of one exact
frequency after a general perturbation. Nonlinear resonances and
higher-order terms remain separate obligations. In particular, even
when every fine edge stays acute along the pulse, positivity of an
instantaneous storage Hessian does not by itself solve this problem.
The quadratic variation has a time-dependent coefficient:

\[
\frac{d}{dt}\frac12 z^T H_2(t)z
 =\frac12 z^T\dot H_2(t)z
\]

when the Hamiltonian tangent terms cancel. It is not generally a
conserved positive quadratic norm. The earlier all-state energy barrier
keeps admitted solutions near the static winding target, but need not
keep them near this particular periodic orbit or phase-locked to it.

This reduction supplies all perturbation directions, exact symmetry
constraints and a one-amplitude stability question. It does not evaluate
a monodromy matrix, prove a finite-amplitude instability, establish a
stability interval, or run a trajectory or Floquet sweep.
The [following small-amplitude calculation](#sine-replica-pulse-splitting)
resolves the leading transverse signs with a separately proved
existential amplitude range.

The [existing scale owner](../../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_pulse_variation` with the exact family
premises reused from the pulse assessment. It reports the three real
instantaneous block types, their multiplicities, the dimensionless
versions, fixed symplectic form, swap matrix and stationary-limit
frequencies. Instantaneous coefficients are not transition matrices.
Periodic-reference and half-return claims remain conditional on the
separate libration certificate, and orbital stability is explicitly
unassessed. The [replica tests](../../tests/physics/test_relational_sine_replica.py)
compare these blocks with the full fine nodal Jacobian and check the
coordinate transformations, symmetry and zero-amplitude controls.

<a id="sine-moving-pulse-work-response"></a>
### 10.7. Internal motion produces a finite same-type work-response asymmetry

This calculation uses the existing pulse and full variation, with no added
force or constitutive row. Its declared preparation is positive twist
`alpha=2*pi/5`, `e=0,w=beta=nu=1`, `u(0)=delta(0)=1/32`, and zero common
origins. Fix real Fourier representative `k=1`, clock `tau=t/pi`, and
readout horizon `h=1/16`. The computational check uses Taylor order ten;
the proof below is independent of a numerical trajectory. These choices
are fixed before response evaluation, not selected by a sign search.

#### Matched form inputs and storage-work outputs

For pair index j, define the two real fine form masks

\[
f_C(j,\pm)=\cos(2\pi j/5),\qquad
f_I(j,\pm)=\mathord\pm\sin(2\pi j/5).
\]

These are distributed perturbations on existing nodes, not new point contacts
or new edges. An infinitesimal initial form increment along either mask
leaves the initial phases unchanged. Subtract the unperturbed pulse and
differentiate with respect to that increment at zero. In the real block of
section 10.2 this supplies the two initial form columns `p(0)=I`, with
phase columns `q(0)=0`. Here `p=(xi,i*v)` and `q=(eta,i*zeta)` denote real
mode amplitudes, not the primitive pressure. The fine reconstruction uses
cosine collective amplitudes and sine internal amplitudes.

Put `c=cos(alpha)`, `g=sin(alpha)^2`, `l=1-c` and `D=diag(l,1)`.
The inherited equations in this clock are

\[
p'=-C(\tau)q,\qquad q'=Dp,\qquad
C(\tau)=\begin{pmatrix}
c\,l\cos^2\delta&g\cos\delta\sin\delta\\
g\cos\delta\sin\delta&c[1-(1+c)\sin^2\delta]
\end{pmatrix},
\quad u'=-c\cos\delta\sin\delta,\quad\delta'=u.
\]

If `P(tau)` is the form-to-form transition block, the actual fine conjugate
work outputs give `H(tau)=20 D P(tau)`. Indeed each mask has squared norm
five, with fine form Laplacian eigenvalues `2*lambda_1` and four, respectively;
thus `(f_C^T L delta_x,f_I^T L delta_x)=(20*l*xi,20*i*v)`.
The same normalization is used for both input/output roles. The quantity
tested is `Delta(tau)=H_CI(tau)-H_IC(tau)`. It is a finite-time linear
response about a moving, finite-amplitude reference, not a finite-amplitude
probe experiment or a stationary frequency transfer.

#### Time ordering supplies the first nonzero antisymmetric term

For the normalized work kernel `K=D P`, differentiating the full block gives

\[
K_{12}-K_{21}
=\frac{l g c^2 u_0[1+l\cos^2\delta_0]}{60}\,\tau^5
 +O(\tau^6).
\]

All lower antisymmetric coefficients vanish. Equivalently the coefficient
matrix is `D(C'(0) D C(0)-C(0) D C'(0))D/60`. Each instantaneous C is
symmetric, but its successive values need not commute in this metric.
Freezing C would erase that information and restore symmetric work transfer.
This is why the [equilibrium reciprocity theorem](RESONANCE_FOUNDATIONS.md#same-type-port-reciprocity)
does not decide this moving-background experiment.

#### A finite positive bound at the declared horizon

Energy conservation gives `u^2+c*sin(delta)^2<1/768`, hence `abs(u)<1/24`.
For `0<=tau<=1/4`, integration first gives `1/48<=delta<=1/24`, and then
`u>=1/32-1/288=1/36`. These are bounds on the actual moving pulse.
Throughout this interval `||C D||_infinity<=3/8`. For `s>=v`, direct
differentiation of the displayed C, using `3/10<c<1/3`, `g>9/10`,
`l>2/3` and the small-angle cosine lower bound `99/100`, gives

\[
\left[D(C(s)D C(v)-C(v)D C(s))D\right]_{12}
\ge\frac{99}{40000}(s-v).
\]

For completeness the derivative of the inner commutator's 12 entry with
respect to s is
`g*c^2*u(s)*((3-c)*cos(2*(delta(s)-delta(v)))+l*cos(2*delta(s)))/2`.
The undifferentiated commutator vanishes at `s=v`, so the derivative's lower
bound integrates to the preceding inequality.

Write the convergent Volterra expansion as `P=sum_n P_n`, where

\[
P_0=I,\qquad
P_{n+1}(\tau)=-\int_0^\tau C(s)D\int_0^s P_n(v)\,dv\,ds.
\]

`D P_0` and `D P_1` are symmetric. The antisymmetric contribution of `D P_2`
has weight `v(s-v)` multiplying the commutator above, hence is at least
`33*h^5/800000`. The terms `n>=3` have total two-entry error at most

\[
\frac{2(3/8)^3h^6}{720[1-(3/8)h^2/56]}.
\]

This follows from `||P_n||_infinity <= (3/8)^n h^(2*n)/(2*n)!` and the
geometric bound on successive tail terms. At the predeclared `h=1/16`,

\[
\boxed{\Delta(h)\ge
\frac{588921}{962047508480000}>6\times10^{-10}>0.}
\]

Thus the finite sign is proved, rather than inferred from the fifth-order
term alone. The pulse's energy also lies below its existing all-fine-edge
acute threshold. This retains the prepared identity; it does not assert
orbital stability under arbitrary disturbances.

#### Reversal controls and scope

- Reversing initial internal motion to `u_0=-1/32` at the same `delta_0`
  gives a strictly negative response by the same interval argument.
  Opposite signs do not require equal finite magnitudes.
- Reversing the base twist while retaining the positive Fourier orientation
  changes the sign of the off-diagonal C entries; the response contrast
  reverses exactly. Their general factor is `sin(alpha)*sin(q_1)`; the
  abbreviation `g=sin(alpha)^2` applies to the fixed positive-twist reference.
  Changing the reporting orientation is a different operation.
- Swapping both members in every pair sends `(u,delta)` to `(-u,-delta)`
  and reverses the internal input/output mask. Transforming the ports with
  the state preserves the physical comparison; a signed matrix entry alone
  is not a label-independent intrinsic property.
- A stationary reference, or the explicit control that freezes C, has
  reciprocal same-type response. A turning point `u_0=0` alone does not
  establish a stationary reference or zero finite contrast.

The shared scale owner exposes `assess_sine_replica_pulse_work_response`.
It rebuilds the pulse and uses the common variational block in the joint
ten-dimensional pulse/two-column field. The shared validated Taylor kernel
encloses its finite response, tube and remainder in `tau=t/pi`; unavailable
enclosures remain explicit. The reader retains general admitted pulse
coefficients, while the finite rational lower bound above is only for the
declared unit-parameter preparation. The
[response controls](../../tests/physics/test_sine_replica_pulse_work_response.py)
and SDK export preserve those boundaries. Its derivative report does not
itself certify a finite probe radius. The continuation below bounds finite
kicks separately; no spatial Hall observation, physical unit bridge or
magnetic law has been derived.

For the frozen order-ten evaluation the outward-rounded response enclosure is
`9.9053976622818e-10 < Delta(h) < 9.9053976623020e-10`, consistent with the
independent lower bound above. This numerical value uses structural units
and the declared port normalization, not measured physical units.

<a id="sine-moving-pulse-finite-work-response"></a>
### 10.8. A controlled finite probe of the complete fine law

Retain every preparation, port and clock premise of section 10.7. Before
evaluating any finite kicked response, fix `epsilon=2^-20`, `h=1/16` and
Taylor order ten. For each mask `f_C,f_I`, prepare the two fine states
`x_0+epsilon*f` and `x_0-epsilon*f`, with the same initial phases. All four
trajectories follow the complete twenty-coordinate sine field, including
the modes and backreaction that a finite kick can generate. No restriction
to the background's invariant family or its linear Fourier block is made.

In the scaled clock the degree-four fine rows are

\[
x_i'=\tfrac14\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
\theta_i'=\tfrac14\sum_{j\sim i}(x_i-x_j).
\]

Let `Phi_h` denote this full flow. Define the centered work response
`H_epsilon,ij=f_i^T L[x_h(+epsilon*f_j)-x_h(-epsilon*f_j)]/(2*epsilon)`
and `Delta_epsilon=H_epsilon,CI-H_epsilon,IC`. The common unperturbed
background cancels; no measured derivative is used to reconstruct pressure.

#### Uniform nonlinear error, before evaluating a finite response

In the maximum norm, globally `||DF||<=2`, `||D^2F||<=4` and
`||D^3F||<=8`. These follow from the four normalized edge contributions,
the gap bound `abs(v_j-v_i)<=2||v||`, and bounded sine derivatives. The
phase row is linear. For any initial direction b of norm at most one,
put `E=exp(2*tau)`. Differentiating the complete flow and applying the
variational integral inequalities yields

\[
\|D\Phi_\tau b\|\le E,\qquad
\|D^2\Phi_\tau[b,b]\|\le2E(E-1),\qquad
\|D^3\Phi_\tau[b,b,b]\|\le4E(E-1)(2E-1).
\]

For example the third bound solves
`w'<=2w+12*v*z+8*v^3`, where `v<=E` and `z<=2E(E-1)`.
At `h=1/16`, the positive exponential series gives
`E=exp(1/8)<=1/(1-1/8)=8/7`, so the third bound is `288/343`.
It holds throughout the kicked initial-state segments, not only at their
center. The actual port rows satisfy `||f_C^T L||_1<=40` and
`||f_I^T L||_1<=40`: their eigenvalues are `4*l<4` and four and each
ten-node mask has entries bounded by one. Taylor's centered-difference
remainder therefore gives

\[
|\Delta_\epsilon-\Delta|\le
\frac{80}{6}\frac{288}{343}\epsilon^2
=\frac{3840}{343}\epsilon^2.
\]

This is uniform for every `0<epsilon<=2^-20`. Combining it with the
independent finite-time lower bound of section 10.7, at the declared radius,

\[
\boxed{\Delta_\epsilon(h)\ge
\frac{12712959417}{21118866906152960000}>6\times10^{-10}>0.}
\]

#### Domain, preparation and interpretation

The reference fine-edge acute margin exceeds `pi/10-1/12>13/60` on
this window. Each kicked flow differs from the reference by at most
`8*epsilon/7` per coordinate, so every edge margin remains larger than
`13/60-16*epsilon/7>0`. The support, capacities and lifted circular phase
information are retained throughout. Preparation changes the stored energy;
thereafter each trajectory follows the same unforced conservative law.

The reversed-motion reference has the opposite finite sign by the same
error bound. General stationary nonlinear reciprocity does not follow from
the linear theorem. For the particular zero-pulse reference, global member
swap leaves the source and collective mask fixed and reverses the internal
mask: that separate symmetry makes its centered cross responses zero.

The raw four-reading contrast is `2*epsilon*Delta_epsilon`, much smaller
than the normalized response. Its proved lower margin is about `1.15e-15`
in these structural units. This is a mathematically nonzero finite signal,
not a practical measurement/noise budget or an experimentally selected scale.
The radius is a sufficient error-controlled choice, not a fundamental TNFR
constant, optimal amplitude or inferred physical limit.

The fixed-template reader `assess_sine_replica_pulse_finite_work_response`
reuses the shared full sine rows and validated Taylor kernel. Its four
order-ten evaluations at the predeclared radius all admit strict whole-time
tubes. The outward-rounded normalized contrast enclosure is
`9.9051618144648e-10 < Delta_epsilon < 9.9056335170602e-10`, contained in
the earlier tangent enclosure widened by the proved nonlinear error.
Analytic sign, finite numerical sign and failed-trial availability remain
separate fields; the [full-flow controls](../../tests/physics/test_sine_replica_pulse_finite_work_response.py)
also check reversal, port normalization and the SDK projection. No numerical
retry or amplitude adjustment was used to obtain this verdict.

<a id="sine-moving-pulse-parametric-representation"></a>
### 10.9. Parametric response and complete-history reversal

The same moving-background tangent has an exact representation useful for
comparison with physical oscillator response. In section 10.7 set
`y=D^(-1/2)q`; then

\[
y''+S(\tau)y=0,\qquad
S(\tau)=D^{1/2}C(\tau)D^{1/2}=S(\tau)^T.
\]

For the matrix solution `G(0)=0,G'(0)=I`, the matched work kernel is
`H(tau)=20*D^(1/2)*G'(tau)*D^(1/2)`. It is therefore a weighted
initial-velocity-to-velocity block of a parametric oscillator. This exact
tangent coordinate transformation does not identify primitive form with a
laboratory displacement or supply physical masses. Its varying stiffness
is determined by the autonomous internal preparation, not an installed
external modulation. The quadratic oscillator expression
`E_tan=(||y'||^2+y^T S y)/2` has derivative `y^T S' y/2`; treating it as
an isolated conserved energy would discard its evolving background.

For the full fine conservative flow, `R(x,theta)=(-x,theta)` satisfies
`F(Rz)=-R F(z)`. The reversed history on a window is
`z_R(tau)=R z(h-tau)`, which starts at `R z(h)`, not generally `R z(0)`.
If `M=D Phi_h(z(0))`, its reversed-history propagator satisfies exactly

\[
M_R=R M^{-1}R.
\]

The real Fourier block is also symplectic. In `(p,q)` ordering, write
`M=[[A,B],[C_*,E_*]]`; the constant symplectic form gives
`M^-1=[[E_*^T,-B^T],[-C_*^T,A^T]]`. Consequently the reversed-history
form/form block is `E_*^T`, not generally `A^T`. The asymmetric forward
work block thus does not demonstrate failure of microscopic reversibility.
Reversing the initial internal velocity at fixed phase, exchanging two
probe roles, and reversing a complete history are distinct experiments.

The [physical comparison](../PHYSICAL_REGIME_CORRESPONDENCES.md#moving-pattern-physical-synergies)
uses these identities to separate transient response, driven steady-state
nonreciprocity and possible effective geometric forces. A time-ordered
commutator alone is neither a Berry curvature nor a magnetic field.

There is a sharper boundary for a proposed adiabatic stiffness-mode phase.
Here S depends on the single real coordinate delta. A full libration
traverses and retraces an interval of delta. Its eigenvalues remain distinct:
`S_22-S_11=c^2*((2-c)-(3-c)*sin(delta)^2)>0`, since the fixed pulse has
`sin(delta)^2<=m<1/200` throughout its orbit. For any smooth single-valued
connection depending only on delta, `integral_closed A(delta) ddelta=0`;
reverse path transport is the inverse even for a matrix-valued connection.
Thus this particular adiabatic stiffness-mode construction has trivial
holonomy. The finite response propagator integrates its generator against
time, not a connection against `d(delta)`, so its asymmetry is not removed.
This does not exclude nonadiabatic phases, Floquet geometry or a
connection on the retained full phase space. It prevents promoting the
present finite-delay asymmetry to a geometric magnetic flux merely because
the stiffness changes over a pulse.

<a id="sine-pulse-stiffness-discriminator"></a>
### 10.10. A clock-independent stiffness-family discriminator

An oscillator resemblance does not establish equality of its stiffness
family. Keep the doubled-C5, real k=1, conservative pulse of section 10.9.
With `c=cos(2*pi/5)`, `l=1-c` and `X=cos(delta)^2`, its whitened stiffness
has trace T and determinant V

\[
T=c(2-c+c^2)X-c^2,\qquad
V=l^2X[(1+c)X-(1+c-c^2)].
\]

Eliminating X gives a quadratic `V=a_* T^2+b T+d_0`, with

\[
\boxed{a_*=
\frac{l^2(1+c)}{c^2(2-c+c^2)^2}
=\frac{1160+480\sqrt5}{1089}>2.}
\]

This coefficient belongs to the declared geometry and mode, not a universal
TNFR or physical constant. Multiplying stiffness by a constant clock factor
r sends `(T,V)` to `(r*T,r^2*V)` and leaves a_* unchanged. For a physical
candidate `M y''+K(t)y=0` with known constant positive-definite mass M,
use `T=tr(M^-1 K)` and `V=det(M^-1 K)`. Fixed congruent changes of state and
work coordinates preserve these invariants. Using K alone while discarding
M does not. A time-dependent chart introduces velocity and additional
stiffness terms and cannot silently repair a mismatch.

#### Exact obstruction for one affine stiffness control

Suppose an admitted two-mode candidate instead has symmetric whitened
`J(t)=J_0+rho(t)*J_1`, with both matrices fixed. If `tr(J_1)` is nonzero,
eliminating rho makes determinant a quadratic in trace with coefficient
`det(J_1)/tr(J_1)^2<=1/4`, because

\[
\operatorname{tr}(J_1)^2-4\det(J_1)
=(J_{1,11}-J_{1,22})^2+4J_{1,12}^2\ge0.
\]

If `tr(J_1)=0`, the trace is constant. Hence no such family can coincide
with a varying segment of the specified TNFR pulse under a constant clock
and fixed work-compatible coordinates. Arbitrarily nonlinear timing rho(t)
does not change this obstruction. Uniform scaling `J(t)=rho(t)*J_0` is
included. This concerns the curvature of a family, not the instantaneous
ratio `det(J)/tr(J)^2`, which retains the usual symmetric-matrix bound.

This does not require adding a primitive coordinate: the existing delta
already changes several matrix entries through different nonlinear functions.
One scalar parameter can trace a non-affine matrix curve. The obstruction
concerns one fixed matrix direction, not the number of scalar TNFR variables.

For three samples, define without division

\[
N=(V_3-V_2)(T_2-T_1)-(V_2-V_1)(T_3-T_2),\qquad
Q=(T_3-T_2)(T_2-T_1)(T_3-T_1).
\]

When traces are distinct, the TNFR template requires `N-a_*Q=0`, whereas
every affine symmetric family obeys `(4N-Q)Q<=0`. These exclusion decisions
are independent of sample ordering. Shared outward interval arithmetic can therefore
exclude a template or an affine candidate without dividing by tiny gaps.
Intervals containing zero do not prove an identity; overlapping traces
leave the three-point curvature unresolved. Correlated uncertainty may be
conservatively enclosed, never silently discarded.

`assess_sine_replica_stiffness_trace_curve` implements these necessary
screens from three supplied trace/determinant observations. It uses shared
scalar/interval admission, supplies no fitted clock or phase, and retains
`not_excluded` as distinct from a successful admission. The
[controls](../../tests/physics/test_sine_replica_stiffness_trace_curve.py)
construct stiffness from the full formula, exercise clock and coordinate
changes, and include affine and uncertain counterexamples.

Passing this screen is insufficient: it neither checks the full stiffness
matrix nor autonomous pulse timing, forcing, damping, preparation or physical
readout. Several degrees of freedom, independently varying stiffness channels,
retained circuit state and eliminated-mode memory must be evaluated using
their actual complete laws. The
[mechanical candidate admission](../PHYSICAL_REGIME_CORRESPONDENCES.md#metabeam-physical-admission)
uses this restriction only for its declared ideal two-mode reduction.

<a id="sine-replica-pulse-splitting"></a>

## 11. Small-amplitude transverse splitting with collective response retained

This result continues the exact conservative doubled-`C5` pulse and
the complete real mode blocks of
[section 10](#sine-replica-pulse-variation). It resolves a local
small-amplitude question; it does not search over amplitudes, integrate
a variational trajectory or introduce a different law.

Write `epsilon=sqrt(m)` and choose the reference pulse section
`d(0)=0,u(0)/sqrt(beta*c)=epsilon`, with `c=cos(alpha)` and
`alpha=2*pi/5`. The two representatives have

\[
\ell_k=\lambda_k/2=1-\cos q_k,\qquad
g_k=\tan\alpha\sin q_k,\qquad q_k=2\pi k/5,\quad k=1,2 .
\]

In particular `0<ell_k<2`, and neither value is an integer. Their
collective limiting multipliers are
`exp(+/-2*pi*i*ell_k)` for the full return and
`exp(+/-pi*i*ell_k)` for the swap-correct half return. They are
distinct from the internal limiting multiplier `+1`. This separation
permits a local two-dimensional internal spectral reduction without
discarding its coupling to the collective coordinates.

### 11.1. Fix the period before expanding the coefficients

The dimensionless pulse period is `4*K(m)` in `tau=Omega*t`.
Use the period-normalized coordinate

\[
\sigma=\omega(m)\tau,\qquad
\omega(m)=\frac{\pi}{2K(m)}
         =1-\frac m4+O(m^2).
\]

The full period is now exactly `2*pi`. The analytic pulse equations
and analytic period give, uniformly on this fixed interval,

\[
d(\sigma,\epsilon)=\epsilon\sin\sigma+O(\epsilon^3),
\quad
\cos^2d=1-\epsilon^2\sin^2\sigma+O(\epsilon^4),
\quad
\cos d\sin d=\epsilon\sin\sigma+O(\epsilon^3).
\]

Oddness of the initial section and vector field makes `d` odd in
`epsilon`. These are analytic Taylor expansions on a fixed compact
interval, with bounded remainders for sufficiently small amplitude;
they do not use a simulated periodic response.

For one representative mode suppress the index `k`, put
`gamma=sqrt(ell)*g` and use
`y=eta/sqrt(ell), z=zeta` in its real block. Eliminating the two
form derivatives yields the symmetric second-order equations. After
the time change they read

\[
\begin{aligned}
y_{\sigma\sigma}
 &+\ell^2[1+\epsilon^2(1/2-\sin^2\sigma)]y
   +\gamma\epsilon\sin\sigma\,z
   +O(\epsilon^4)y+O(\epsilon^3)z=0,\\
z_{\sigma\sigma}
 &+[1+\epsilon^2 f(\sigma)]z
   +\gamma\epsilon\sin\sigma\,y
   +O(\epsilon^4)z+O(\epsilon^3)y=0,\\
f(\sigma)
 &=\frac12-(2-\ell)\sin^2\sigma .
\end{aligned}
\]

The `1/2` terms come from `omega(m)^(-2)=1+m/2+O(m^2)`.
Leaving the reference period uncorrected would change the internal
resonant coefficients.

### 11.2. The induced collective motion changes the splitting

At zero amplitude the internal solution is
`z_0=A*cos(sigma)+B*sin(sigma)`. The internal spectral subspace
induces a collective correction `y=epsilon*y_1+O(epsilon^2)`.
Its leading `2*pi`-periodic solution satisfies

\[
y_1''+\ell^2y_1=-\gamma\sin\sigma\,z_0 .
\]

There is a unique periodic solution because `ell` is not an integer.
It is

\[
\boxed{
y_1=
-\frac{\gamma A}{2(\ell^2-4)}\sin2\sigma
-\frac{\gamma B}{2\ell^2}
+\frac{\gamma B}{2(\ell^2-4)}\cos2\sigma .}
\]

The denominators `ell^2` and `ell^2-4` retain respectively the
constant and second-harmonic collective responses. Setting `y_1=0`
would not be the full fine variational problem.

Substitution in the internal equation, followed by projection onto
`cos(sigma),sin(sigma)`, gives the resonant coefficients

\[
\boxed{\begin{aligned}
P&=\frac{\ell}{4}-\frac{\gamma^2}{4(\ell^2-4)},\\
Q&=\frac{3\ell-4}{4}
   -\frac{\gamma^2}{2\ell^2}
   -\frac{\gamma^2}{4(\ell^2-4)} .
\end{aligned}}
\]

Here `P,Q` are coefficient names local to this perturbation calculation;
`Q` is not the retained internal correlation of section 8. The
diagonal part alone would give `ell/4` and `(3*ell-4)/4`; the
remaining terms are the calculated feedback from the collective
response.

To see their return-map meaning, use rotating internal amplitudes
`z=A(sigma)*cos(sigma)+B(sigma)*sin(sigma)` with the usual
variation-of-constants condition. The resonant part of their equation
is

\[
\binom{A}{B}'=
mG\binom{A}{B}+\text{higher-order and removable periodic terms},
\qquad
G=\begin{pmatrix}0&Q/2\\-P/2&0\end{pmatrix}.
\]

For example, forcing `-m*(P*A*cos(sigma)+Q*B*sin(sigma))` gives
the averaged rows `A'=m*Q*B/2` and `B'=-m*P*A/2`.
The nonresonant harmonics can be removed by a periodic near-identity
change of variables; integrating their zero-mean part over one fixed
period gives no additional leading return term. Equivalently this is
the leading analytic reduction of the isolated internal spectral
subspace. Its full return matrix in such a basis is

\[
M_{\rm int}=I+2\pi mG+O(m^{3/2}),\qquad
\sigma_*^2=-PQ/4 .
\]

The symbol `sigma_*` denotes a splitting coefficient, not the
period-normalized time `sigma`.

### 11.3. Exact algebraic signs for the two spatial modes

Using `tan(alpha)^2=5+2*sqrt(5)` and the exact cycle angles gives:

| Representative | `P` | `Q` | `sigma_*^2=-P*Q/4` |
| --- | --- | --- | --- |
| `k=1` | `(95+sqrt(5))/164` | `-(479+245*sqrt(5))/164` | `(23365+11877*sqrt(5))/53792 > 0` |
| `k=2` | `(55+21*sqrt(5))/41` | `(14+21*sqrt(5))/41` | `-(2975+1449*sqrt(5))/6724 < 0` |

All signs follow directly from positive integers and `sqrt(5)>0`.
They are exact identities, not signs extracted from a numerical
monodromy. The first internal pair therefore has a real leading
splitting and the second an imaginary one.

### 11.4. Analytic remainder and the actual sufficiently-small conclusion

The return matrices are analytic in `epsilon`: their coefficients,
the fixed-period pulse and the finite-interval linear initial-value
problem are analytic in that parameter. At `epsilon=0` the collective
pair is separated from `+1`. A sufficiently small fixed contour around
`+1` therefore defines an analytic real two-dimensional spectral
subspace of the full return. The displayed leading matrix is the
restriction to that subspace in an analytic basis.

That subspace is symplectic for sufficiently small amplitude. At zero
it is the nondegenerate internal oscillator plane, and nondegeneracy
persists by continuity. Since the full return is symplectic, the
restricted return has determinant one.

There is also an exact parity restriction. Replacing `epsilon` by
`-epsilon` changes the reference pulse by the member swap, so the
full return matrices are conjugate by `S_*`. Let `P_epsilon` be
the analytic internal spectral projector and let `V_0` be a fixed
basis for the limiting internal plane, ordered as the rotating
amplitudes. Then

\[
P_{-\epsilon}=S_*P_\epsilon S_*,
\qquad S_*V_0=-V_0,\qquad
V_\epsilon=P_\epsilon V_0,\qquad
V_{-\epsilon}=-S_*V_\epsilon .
\]

The columns of `V_epsilon` remain independent for sufficiently small
amplitude. In this basis the restricted return matrix is exactly even
in `epsilon`, because
`M(-epsilon)=S_*M(epsilon)S_*` intertwines the displayed bases.
It is therefore analytic in `m=epsilon^2` and satisfies

\[
M_{\rm int}=I+2\pi mG+O(m^2),\qquad
\frac{\log M_{\rm int}}{2\pi}=mG+O(m^2).
\]

The logarithm is the real branch near the identity. It is an effective
return generator, not the raw instantaneous rotating-amplitude row.
Trace and determinant are basis-independent, so their resulting
expansions hold even when another analytic basis is used.

Let `t_int` be the trace. Determinant one implies

\[
t_{\rm int}
 =2+4\pi^2\sigma_*^2m^2+O(m^3).
\]

Indeed `det(G)=-sigma_*^2`, so the order-two trace coefficient is
fixed by the determinant identity. In particular,

\[
\boxed{\quad
t_{\rm int}=2+4\pi^2\sigma_*^2m^2+O(m^3),\qquad
t_{\rm int}^2-4=16\pi^2\sigma_*^2m^2+O(m^3).
\quad}
\]

This establishes a remainder-controlled **existence** statement. For
each mode there are positive constants `C,m_0` bounding the remainder
by `C*m^3` on `0<m<m_0`. Because the exact leading coefficient is
nonzero, shrinking that interval makes it dominate the remainder.
No numerical values of those constants or an explicit amplitude radius
are obtained here.

For `k=1`, put

\[
\sigma_1=
\sqrt{\frac{23365+11877\sqrt5}{53792}}>0 .
\]

The internal labeled multipliers then satisfy

\[
\Lambda_\pm=1\pm2\pi\sigma_1m+O(m^2),
\]

and for every sufficiently small positive `m` they are a real
reciprocal pair with one member strictly greater than one. The
swap-correct half-return multipliers satisfy
`1+/-pi*sigma_1*m+O(m^2)`, consistently with `M=B^2`.
This is an exponentially unstable transverse mode of the prepared
periodic orbit. Both real spatial copies of this representative have
the same conclusion.

For `k=2`, put

\[
\omega_2=
\sqrt{\frac{2975+1449\sqrt5}{6724}}>0 .
\]

Its internal labeled multipliers satisfy

\[
\Lambda_\pm=1\pm2\pi i\omega_2m+O(m^2).
\]

For every sufficiently small positive `m` they are nonreal and,
because their product is one, lie exactly on the unit circle. They
are distinct and semisimple. The corresponding collective pair starts
as a simple isolated nonreal unit-circle pair and remains so for a
sufficiently small amplitude interval: leaving the circle would require
a reciprocal-conjugate collision, excluded locally by its initial
spectral separation. Thus this representative passes the bounded
**linear** transverse criterion locally. It is not a nonlinear
orbital-stability theorem for that mode, and it cannot stabilize the
whole pulse against the unstable `k=1` directions.

### 11.5. Interpretation and what remains unmeasured

The same full nonlinear law admits the exact periodic family and also
makes its sufficiently small nonzero members transversely unstable.
There is no contradiction: invariant preparation proves that the orbit
exists, whereas the return splitting asks what nearby states do. An
unstable transverse return multiplier gives nonlinear orbital
instability of this smooth periodic orbit; it does not imply that every
perturbation grows or that the exact orbit ceases to exist.

The amplification is produced by the periodic internal coefficients
and their coupled collective response. It does not require an external
periodic source or a newly installed pressure law. In the complete
conservative system, departure from that particular internal waveform
can transfer storage among its retained modes without changing total
storage. This is distinct from the neutral time-amplitude shear in
mode zero.

The harmonic mechanism is explicit. The `sin(sigma)^2` stiffness
term contains a constant part and a second harmonic. The order-
`epsilon*sin(sigma)` cross term sends the internal first harmonic
into the collective constant and second-harmonic responses; those feed
back into the internal first harmonic through the same cross term.
This is parametric feedback in the derived tangent law, not an externally
prescribed oscillatory drive, a claim about all states, or invocation of
the named Resonance operator.

Nor does orbital instability mean unbounded fine form, loss of every
coherent geometry or unavoidable winding change. For sufficiently
small pulses, the earlier acute-target trapping domain contains open
neighborhoods of their initial states: both their centered norm and
excess storage tend to zero with amplitude. Solutions from an admitted
neighborhood remain near that static phase identity, even while some
depart from the common-internal-state periodic orbit. A robust
geometric organization and a fragile synchronized waveform can
therefore coexist under the same storage balance.

The result supplies no certified numerical interval of unstable
amplitudes, no verdict for a chosen nonzero represented amplitude
solely because it is called small, no growth time at that amplitude,
and no finite-amplitude continuation of either mode class. It does
not justify changing the constitutive law to preserve a preferred
pulse, or identify this prepared organization with physical matter.

The [existing scale owner](../../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_pulse_splitting`. It retains the exact
coefficients as `a+b*sqrt(5)`, encloses their signs and the leading
return-logarithm slopes, and reuses the same declared model, capacity
and symbolic target through a stationary reference template. That
reference is not itself labeled unstable. The API accepts no finite
amplitude and supplies no numerical amplitude radius, remainder
constant, computed multiplier or selected-preparation verdict.
The [replica tests](../../tests/physics/test_relational_sine_replica.py)
independently check the forced collective harmonics, resonant
projections, algebraic coefficients and return normalization.

<a id="sine-replica-joint-persistence"></a>

## 12. Collective identity with independently active constituents

This theorem combines the full-state barrier of
[Section 7.5](SINE_PAIR_STATE.md#75-trapping-permits-persistent-internal-organization),
the retained internal law of
[Section 8](SINE_PAIR_STATE.md#sine-replica-unordered-state), and
[nonlinear recurrence](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
It uses the complete doubled `C5` support, `e=0`, common held
capacity `nu>0`, `w,beta>0` and one structural clock. There are
no inputs, events, clipping, discarded constituents or changing edges.
The support, pair partition and zero-loss law remain supplied premises.
No attracting pulse or new constitutive mechanism is introduced.

### 12.1. An admitted open full-state family

Let `a=w/pi`, `b=w/(beta*pi)` and `alpha=2*pi/5`.
For each pair `i=0,...,4` the exact target phase is
`theta_(i,+),*=theta_(i,-),*=i*alpha`, with uniform target form.
The fine graph has ten nodes, degree four and twenty edges. Its
target storage and spectral gap are

\[
E_*=20\beta(1-\cos\alpha),\qquad
\lambda_{2,f}=5-\sqrt5 .
\]

Reversing the base cycle orientation gives the same conclusions for
winding minus one, with `|alpha|` in the radius condition.

Write `P_f=I-11^T/10`, `v=P_f x` and use the phase chart

\[
\theta=\theta_*+c\mathbf1+h\pmod{2\pi},
\qquad h\perp\mathbf1,\qquad c\in\mathbb R/2\pi\mathbb Z .
\]

The common circular phase origin is retained. In the same fixed model
coordinates as the preceding barrier, define

\[
Z_f^2=\|v\|^2+\|h\|^2,\qquad
0<r,\quad \alpha+\sqrt2r<\frac\pi2,
\]

\[
c_r=\cos(\alpha+\sqrt2r)>0,\qquad
\kappa_f=\frac{5-\sqrt5}{2}\min(1,\beta c_r)>0 .
\]

This is a genuine chart throughout the closed radius ball and a
slightly larger neighborhood. Indeed two representations of the same
circular state would have
`(h_i-h_j)-(h'_i-h'_j)=2*pi*(n_i-n_j)`. Its magnitude is
at most `2*sqrt(2)*r<2*pi`, forcing all integers equal;
centering then gives `h=h'` and the same `c` on its circle.
The chart differential is invertible. In particular there is no hidden
wrapping boundary inside the ball.

For `Z_f<=r` every fine edge has its target-compatible increment
`+/-alpha+h_j-h_i`, strictly inside `(-pi/2,pi/2)`.
The target is critical. Its vanishing first variation, the phase Hessian
bound `H_phase>=c_r*L_f` along the segment to the target, and the
fine spectral gap give

\[
\mathcal E:=E_f-E_*
\ge\frac{\lambda_{2,f}}2\|v\|^2
+\frac{\beta c_r\lambda_{2,f}}2\|h\|^2
\ge\kappa_f Z_f^2 .
\]

Choose an excess ceiling and finite positive-width mean interval,

\[
0<\varepsilon_*<\kappa_f r^2,\qquad
m_{\rm lo}<m_{\rm hi}.
\]

Here `epsilon_*` is a family ceiling, not the small-amplitude
parameter of Section 11. Because fine degree and capacity are common,
the conserved weighted form mean is the ordinary mean
`m(x)=sum(x)/10`. Define

\[
\boxed{\mathcal U=
\{Z_f^2<r^2,\quad \mathcal E<\varepsilon_*,
                 \quad m_{\rm lo}<m(x)<m_{\rm hi}\}.}
\]

Energy and mean conservation prevent a first radius exit in either
time direction: an exit would require
`mathcal E>=kappa_f*r^2>epsilon_*`. Consequently

\[
Z_f(t)^2\le\frac{\mathcal E(0)}{\kappa_f}
 <\frac{\varepsilon_*}{\kappa_f}<r^2
\qquad(t\in\mathbb R).
\]

The full smooth flow is complete on the bounding energy/mean slab
by the recurrence owner's compactness argument. The chart, both strict
inequalities and the mean interval are preserved, so
`Phi_t(U)=U` for every real `t`. Fine edges remain acute and
every cycle retains its target winding. In particular every five-edge
loop following the positive base orientation and choosing one constituent
from each consecutive pair retains winding one.

The family is open in the full twenty-dimensional state manifold,
not just in a synchronized subspace. Form coordinates are
`x=m*1+v` with `v perpendicular 1`. Together with the phase
chart these are locally invertible coordinates, and all inequalities
are strict. A uniform-form target with its mean inside the interval
has an open neighborhood in `U`, proving positive volume. Its
closure is compact: `|x_i-m|<=||v||<=r` bounds form, and phases
belong to a compact torus. Thus `U` has finite positive volume
for fine-form Lebesgue measure times phase-torus Haar measure.

### 12.2. Every nontip pair remains internally active

In the retained pair coordinates the fine norm splits exactly as

\[
Z_f^2=2\|P_5X\|^2+
      2\|P_5(\Theta-\Theta_*)\|^2+
      2\sum_i(u_i^2+\delta_i^2).
\]

Hence `|delta_i|<D:=r/sqrt(2)<pi/2`, so the nonantipodal
pair chart remains valid for all time. Let

\[
\mathcal T_i=\{u_i=\delta_i=0\},\qquad
\mathcal U_{\rm active}
=\mathcal U\setminus\bigcup_{i=0}^4\mathcal T_i .
\]

Each synchronized tip `T_i` is invariant under the **full** smooth
flow: `u_i_dot=-A_i*sin(delta_i)` and
`delta_i_dot=b*nu*u_i` vanish there, irrespective of the motion
in other pairs. Two-sided uniqueness then makes its complement invariant.
A nontip pair cannot arrive at that tip at a finite time. This conclusion
uses realizable fine coordinates, rather than only the polynomial
constraint among `R,U,Q`.

Each `T_i` has codimension two in the fine chart and zero ambient
volume. Therefore `U_active` is still open, of positive finite
volume, and invariant for every real time. Removing the five tips does
not remove any positive-volume portion of `U`.

The acute full-state geometry supplies a uniform restoring sign. Define

\[
F_i=\frac12\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),
\qquad A_i=a\nu F_i,\qquad c_i=b\nu .
\]

For a base edge `{i,j}`, averaging the cosines of its four actual
fine edge increments gives

\[
R_iR_j\cos(\Theta_j-\Theta_i)
 =\frac14\sum_{s,t=\pm1}
       \cos(\Theta_j+t\delta_j-\Theta_i-s\delta_i)
 \ge c_r .
\]

Since `0<R_i<=1`, each summand
`R_j*cos(Theta_j-Theta_i)>=c_r`. Its upper bound is one.
Consequently

\[
\boxed{c_r\le F_i\le1,\qquad A_i\ge a\nu c_r>0.}
\]

At any nontip state the unordered internal velocity is nonzero.
If `Q_i!=0`, then `R_i_dot=-c_i*Q_i!=0`. If `Q_i=0`,
the remaining nontip possibilities in this chart are:

- `R_i=1,U_i>0`, where `Q_i_dot=c_i*U_i>0`;
- `U_i=0,R_i<1`, where
  `Q_i_dot=-A_i*(1-R_i^2)<0`.

Thus every pair of every state in `U_active` has a moving internal
state `(R_i,U_i,Q_i)` at every finite time. This does not require
each separate scalar coordinate or each fine node rate to be nonzero.
It also does not give a uniform positive lower bound on the velocity
norm: the open family contains states arbitrarily close to a tip.

### 12.3. Bounded internal circulation without a common period

There is a stronger all-state consequence of the same retained rows.
Normalize internal form only for this calculation:

\[
y_i=\frac{u_i}{\sqrt\beta},\qquad
\Omega=\frac{w\nu}{\pi\sqrt\beta}>0.
\]

Then along the full interacting trajectory

\[
\dot\delta_i=\Omega y_i,\qquad
\dot y_i=-\Omega F_i(t)\sin\delta_i,\qquad c_r\le F_i(t)\le1 .
\]

The coefficient `F_i(t)` is generated by the other retained
coordinates; it is not an imposed periodic drive. Since
`(delta_i,y_i)` never equals zero, its argument has a continuous
real lift `psi_i=arg(delta_i+i*y_i)` for all real time. Direct
differentiation yields

\[
-\dot\psi_i
=\Omega\frac{F_i(t)\delta_i\sin\delta_i+y_i^2}
                 {\delta_i^2+y_i^2}.
\]

For `|delta_i|<D<pi/2`,
`sinc(D)<=sin(delta_i)/delta_i<=1`, taking the continuous
value one at zero. The quadratic quotient therefore satisfies

\[
\boxed{\Omega c_r\,\operatorname{sinc}(D)
        \le-\dot\psi_i\le\Omega,\qquad
        \operatorname{sinc}(D)=\frac{\sin D}{D}>0.}
\]

Each constituent pair undergoes endlessly repeated turns in this
internal phase plane in forward time, and oppositely in backward time.
From any initial angular value, the unique next full-turn crossing
has elapsed time `Delta t_i` bounded by

\[
\frac{2\pi}{\Omega}\le\Delta t_i
\le\frac{2\pi}{\Omega c_r\operatorname{sinc}(D)}.
\]

Successive axis crossings have analogous quarter-turn bounds. These are bounds
on angular traversal under the declared structural clock, **not**
on recurrence of the internal amplitude or the full state. Different
pairs may change amplitude, modulate their traversal times and exchange
storage; no exact shared frequency or phase locking has been assumed
or proved. Each pair is active, but their dynamics are still coupled;
no statistical independence is claimed. A pair swap shifts `psi_i`
by `pi` and leaves its derivative unchanged. A projective advance of
`pi` likewise need
not return the unordered state unless its amplitude also returns.
No positive minimum amplitude follows from these angular bounds.

### 12.4. Almost-everywhere recurrence in the same family

The full conservative sine flow has zero divergence in fine form and
circular phase, as proved by the
[recurrence owner](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence).
Restrict that preserved measure to the finite-volume invariant
`U_active`. The same finite-measure recurrence argument gives,
for every fixed sampling increment `s>0`,

\[
\Phi_{n_js}(z)\longrightarrow z,\qquad n_j\longrightarrow\infty,
\quad\text{for almost every }z\in\mathcal U_{\rm active}.
\]

All these recurrent states are nonstationary by Section 12.2.
Here recurrence is measured on the full fine state with circular
phases. It is not inferred using ambient Lebesgue measure in the
twenty-five constrained invariant coordinates `(X,Theta,R,U,Q)`.
An absolutely continuous preparation distribution on the fine family
inherits the almost-sure conclusion. A selected state, finite grid,
fixed-energy surface or other singular preparation does not receive a
recurrence certificate from this ambient-measure theorem.

The quantifiers are distinct: **every** admitted state preserves its
geometry and its five active internal constituents, with the angular
traversal bounds above; **almost every** such state also returns
arbitrarily near its full initial state. There is no full-state return
deadline, generic exact period or numerical chosen-state guarantee.

### 12.5. Overlap with the prepared pulse and its instability

Take a pulse preparation from Section 9 at a phase crossing:
`X_i=m` inside the mean interval, `Theta_i=i*alpha+c`,
`delta_i=0` and the same `u_i=sqrt(H)>0` in all five pairs.
At that instant

\[
Z_f^2=10H,\qquad \mathcal E=20H.
\]

Thus the explicit sufficient conditions

\[
0<H<
\min\left(\frac{r^2}{10},
          \frac{\varepsilon_*}{20},\beta\cos\alpha\right)
\]

put the preparation in `U_active` and in the nonlinear libration
family. Every bound on the right is strictly positive. The entire pulse
and an open full-state neighborhood of its preparation consequently
remain in the admitted collective family, with all constituents active.

Section 11 supplies an existential `m_*>0` such that the prepared
pulse is transversely orbitally unstable for
`0<m=H/(beta*cos(alpha))<m_*`. Intersecting this interval with
the preceding positive interval proves a nonempty overlap without
computing `m_*`. There is no contradiction: the larger phase
organization and internal circulation remain protected while an exact
common waveform can be fragile. Nearby states are not thereby proved
periodic, synchronized, or convergent to another selected pulse.

Finally, two-sided invariance itself prevents capture into
`U_active` from its complement. The
[formation boundary](RESONANCE_FOUNDATIONS.md#conservative-formation-boundary)
still applies. This is a conditional joint **maintenance** result for
a supplied organization, not autonomous construction of its support or
partition, selection of microscopic zero loss, universal fractality,
or identification of a material constituent.

### 12.6. Captured-source admission and evidence

The existing [scale owner](../../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_persistence` and
`SineReplicaPersistenceAssessment`. It checks the complete ordered
doubled-cycle support, common positive capacity and conservative model
on one retained capture. Exact target turns are separate from represented
source phases. The
[shared geometry owner](../../src/tnfr/physics/relational_sine_recovery.py)
supplies whole-source norm, canceled excess-storage and first-exit bounds;
its general rational spectral lower bound can be weaker than the exact
`5-sqrt(5)` used above without invalidating an admitted certificate.

The reader distinguishes the declared finite-volume family's admission,
source trapping, each pair's exact tip status, and source membership
in the stricter excess/absolute-mean family. Its captured form values
supply the actual mean; a relative observation alone would not provide
that common origin. Trapping and nontip activity may be certified even
when the source is outside the smaller declared mean/excess family.
Family almost-everywhere recurrence never becomes selected-source
recurrence through these checks.

For numerical angular bounds the reader uses the safe lower factor
`c_r*cos(D)<=c_r*sinc(D)`. The inequality follows from
`sin(D)-D*cos(D)>=0` for `0<=D<pi/2`. An outward speed
enclosure may touch zero for an extremely small exact positive capacity;
that numerical limitation does not negate strict angular circulation
under the admitted theorem. A finite upper traversal-time bound remains
an angular bound rather than a state-return deadline.

The [replica tests](../../tests/physics/test_relational_sine_replica.py)
check the complete source rows and geometric prerequisites independently.
The reader evaluates no trajectory, waits for no recurrence, changes
no law, and assigns no finite-amplitude instability radius.
