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

<a id="sine-pulse-internal-rotation"></a>

#### EPI as the projection of an internal circulation

The existing pulse has an exact circular representation. This is a
coordinate consequence of the two retained rows, not an additional
oscillator, complex EPI variable or physical spatial rotation. With
`H>0`, `0<m<1` and `|delta|<pi/2` as above, put

\[
A=u,\qquad B=\sqrt{\beta c_\alpha}\sin\delta,\qquad Z=A+iB .
\]

Both real coordinates have form units. The original conservation identity
and the independently specified evolution rows give

\[
A^2+B^2=H,\qquad
\dot A=-\Omega\cos\delta\,B,\qquad
\dot B=\Omega\cos\delta\,A,\qquad
\boxed{\dot Z=i\Omega\cos\delta\,Z.}
\]

For example, the first row follows from `a=beta*b` and
`Omega=b*nu*sqrt(beta*c_alpha)`; the second follows by differentiating
`sin(delta)` using `delta_dot=b*nu*u`. Thus the two-coordinate
description follows from the full law rather than reconstructing a
pressure from its measured response.

The map `(u,delta)->(A,B)` is invertible on this pair chart, since its
Jacobian determinant is `sqrt(beta*c_alpha)*cos(delta)>0`. A continuous
angular lift `psi=arg(Z)` therefore obeys

\[
\dot\psi=\Omega\cos\delta
 =\Omega\sqrt{1-m\sin^2\psi},\qquad
0<\Omega\sqrt{1-m}\le\dot\psi\le\Omega .
\]

The constituent forms are exactly

\[
x_{i,+}=\bar X+\sqrt H\cos\psi,\qquad
x_{i,-}=\bar X-\sqrt H\cos\psi .
\]

EPI is consequently a projection of circulation in this internal state
plane. Its angular speed is generally variable in the declared structural
clock. The primitive phase difference `delta` still librates; the internal
angle `psi` is not that primitive phase. The angle is unavailable at `H=0`,
where the state is stationary. Neither a rotation in physical space nor
spin, a universal frequency or a laboratory time unit follows.

This also gives an exact information-loss control. Choose `0<d<pi/2` and
`u_0` with `0<u_0^2+beta*c_alpha*sin(d)^2<beta*c_alpha`. The two admitted
preparations `(u_0,+d)` and `(u_0,-d)` have the same complete fine EPI
snapshot, the same means and the same storage H, but

\[
\dot u\big|_{+d}=-a\nu c_\alpha\cos d\sin d<0,\qquad
\dot u\big|_{-d}=+a\nu c_\alpha\cos d\sin d>0 .
\]

Instantaneous EPI and storage alone therefore do not determine even the
direction of its next change. The retained relative phase distinguishes
these states. Observing only the collective mean hides the entire pulse;
forgetting constituent labels instead changes the exact period to `T/2`
as already proved above. A visually repeated scalar trace alone neither
identifies these internal coordinates nor proves periodicity of the full state.

Common phase shifts leave every phase difference, and hence these form
rows, unchanged. A time-dependent common phase rotation requires its own
complete phase-law admission; it is not a source of this relative EPI pulse.

There is already a broader maintenance result:
[Section 12](#sine-replica-joint-persistence) proves continuing internal
turns on an open full-state family, with interaction-generated angular
speeds and without a common period. It does not require the exact circles
or common waveform of this special preparation. Its trapping, activity,
almost-everywhere recurrence and no-capture boundaries remain distinct.
In particular, that robust organization is compatible with the
[small-amplitude orbital instability](#sine-replica-pulse-splitting)
of the synchronized waveform. This circular representation supplies no
new stability or formation certificate and does not select the zero-loss law.

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

The [replica pulse owner](../../src/tnfr/physics/relational_sine_replica_pulse.py)
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

The [replica pulse owner](../../src/tnfr/physics/relational_sine_replica_pulse.py)
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

The [replica pulse owner](../../src/tnfr/physics/relational_sine_replica_pulse.py)
exposes `assess_sine_replica_pulse_work_response`.
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

The [replica pulse owner](../../src/tnfr/physics/relational_sine_replica_pulse.py)
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

The existing [replica pulse owner](../../src/tnfr/physics/relational_sine_replica_pulse.py)
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


<a id="replica-internal-preparation-obstruction"></a>

## 13. Internal activity alone does not create collective winding

The preceding persistence theorem starts from an organized preparation.
This section tests a different, independently specified hypothesis:
can heterogeneous internal activity organize initially equal collective
form and phase means under the same conservative doubled-C5 law?
For the source class below the answer is negative, even when its storage
exceeds that of a winding-one target. The obstruction is the retained
collective geometry, not absence of internal motion.

This is an explicit symmetry control, not a claim about every independent
preparation. The frozen scope is
`artifacts/replica-preparation-obstruction-20261026/scope.json`.
No source is constructed by following a desired target backward.

### 13.1. Complete law, independent source and necessary target conditions

Keep the ten ordered fine nodes `(i,+),(i,-)`, `i=0,...,4`, the four
unit fine edges for every adjacent pair of the base C5, and no within-pair
edge. Every fine degree is four. Fix `e=0,w=beta=nu=1`, common held
capacity one, original structural time `t`, and no input, event, clipping,
support change or retuning. Both original rows retain their factor `1/pi`.

Use the continuous lifts and exact form coordinates of
[the retained-state owner](SINE_PAIR_STATE.md#sine-replica-inheritance):
\[
x_{i,\pm}=X_i\pm u_i,\qquad
\theta_{i,\pm}=\Theta_i\pm\delta_i .
\]
Its complete rows specialize to
\[
\begin{aligned}
\dot X_i&=\frac{\cos\delta_i}{2\pi}
 \sum_{j\sim i}\cos\delta_j\sin(\Theta_j-\Theta_i),\\
\dot\Theta_i&=\frac1{2\pi}\sum_{j\sim i}(X_i-X_j),\\
\dot u_i&=-\frac{\sin\delta_i}{2\pi}
 \sum_{j\sim i}\cos\delta_j\cos(\Theta_j-\Theta_i),\\
\dot\delta_i&=\frac{u_i}{\pi}.
\end{aligned}
\]
These are exact identities on supplied continuous lifts. Repeatedly
replacing a lift by a new midpoint would change the coordinates used
in this calculation.

Specify the initial class directly:
\[
 X_i(0)=0,\qquad \Theta_i(0)=0,\qquad
 u_i(0)\in[1,2],\qquad
 \delta_i(0)\in[1/100,1/50].
\]
The ten internal coordinates may vary independently within these
intervals. In fine coordinates, however, this is a correlated class:
each pair's form sum and lifted phase sum are exactly zero. It is not
an ambient box of independent fine-node errors. The intervals do not
encode a target winding, a future endpoint or a chosen periodic waveform.

The necessary geometric conditions for the operational target are:
every physical pair phasor
\[
 Z_i=\frac{e^{i\theta_{i,+}}+e^{i\theta_{i,-}}}{2}
\]
is nonzero; all oriented adjacent normalized-phasor principal gaps
\[
 \omega_i=\operatorname{Arg}
 \left(\frac{Z_{i+1}}{|Z_{i+1}|}
       \frac{\overline Z_i}{|Z_i|}\right)
\]
are strictly acute, `|omega_i|<pi/2`; and the winding on the oriented
base cycle is
\[
 \frac1{2\pi}\sum_{i=0}^4\omega_i=1 .
\]
Indices are modulo five. Retained constituent activity and a finite
retention interval are additional target obligations. Excluding these
necessary geometric conditions already excludes that stronger target.
It does not exclude every other definition of organization.

Initially `Z_i=cos(delta_i)>0`, so this source has winding zero.
Its fine constituents are nevertheless moving. In fact,
\[
 \dot u_i(0)
 =-\frac{\sin\delta_i(0)}{2\pi}
   [\cos\delta_{i-1}(0)+\cos\delta_{i+1}(0)]<0,\qquad
 \dot\delta_i(0)=\frac{u_i(0)}{\pi}>0 .
\]
Thus both the form and phase rates of every fine constituent are nonzero
initially, with opposite signs within a pair. No claim of a common
oscillation or a uniform all-time activity lower bound is used.

### 13.2. A global invariant leaf, including midpoint-chart crossings

Whenever `X_i=X_*` and `Theta_i=Theta_*` for all five pairs, the
first two retained rows vanish, independently of all internal coordinates.
The remaining exact subsystem is
\[
 \dot u_i=-\frac{\sin\delta_i}{2\pi}
               (\cos\delta_{i-1}+\cos\delta_{i+1}),\qquad
 \dot\delta_i=\frac{u_i}{\pi}.
\]
The full finite sine field is globally Lipschitz on real form and phase
lifts: form dependence is linear and the phase derivatives of sine are
bounded. Smooth uniqueness therefore makes this common-mean leaf
invariant, and continuation is global. For the declared sources,
\[
 X_i(t)=0,\qquad \Theta_i(t)=0
 \quad\text{for every real }t.
\]
The invariant statement does not freeze the internal coordinates and
does not require their values to agree between pairs.

For every time, the exact physical phasor identity consequently gives
\[
 Z_i(t)=e^{i\Theta_*}\cos\delta_i(t).
\]
This identity remains valid when a pair reaches an antipodal configuration
or leaves the nonantipodal midpoint chart. It is a statement about the
original circular fine phases, not a continuation obtained by assigning
an angle to a zero phasor.

If any `cos(delta_i)=0`, the target phasor admission fails. Otherwise
each normalized phasor equals either `exp(iTheta_*)` or its negative.
Adjacent unequal signs have an antipodal gap, which is not strictly
acute under either principal-argument convention. If every adjacent
gap is acute, connectedness forces all five signs to agree. Every gap
then equals zero and the winding is zero.

Therefore no state in the declared source class can reach the necessary
winding-one target conditions at any time. In particular, no finite
entry and retention window exists for that target from this class.
This proof permits arbitrary intervening internal motion, sign changes,
phasor zeros and loss of the original midpoint chart. It does not import
the separate no-entry theorem for the smaller invariant active family.

#### A finite exclusion for genuinely perturbed means

The exact symmetry class also has a finite observation margin. This
separate statement permits errors in the collective means; it does not
extend exact common-mean invariance or the preceding initial activity
claims to arbitrary perturbations.

Keep the same law and continuous lifts, let `X_c` and `Theta_c` denote
the five pair means after subtracting their respective common means,
and use the fixed-coordinate norm
\[
 N^2=\|X_c\|_2^2+\|\Theta_c\|_2^2.
\]
Both common means are conserved, by antisymmetry of the form row and
the zero sum of the base Laplacian row. Write `R_i=cos(delta_i)` and
\[
 a_{ij}=R_iR_j\operatorname{sinc}(\Theta_j-\Theta_i),\qquad
 \operatorname{sinc}(r)=\frac{\sin r}{r},\quad
 \operatorname{sinc}(0)=1 .
\]
The exact coarse equations become
\[
 \dot X_c=-\frac1{2\pi}L_a\Theta_c,\qquad
 \dot\Theta_c=\frac1{2\pi}L_{C5}X_c.
\]
Here `L_a` is a symmetric signed Laplacian with `|a_ij|<=1`.
Its maximum absolute row sum is at most four, hence
`||L_a||_2<=4`; the same upper bound holds for `L_C5`.
No positive-semidefinite assumption on `L_a` is made. The norm
inequality, equivalently its squared-norm version at zero, gives
\[
 N(t)\le N(0)e^{2t/\pi}\qquad(t\ge0).
\]
Thus, on a declared finite horizon `T`, the sufficient condition
\[
 N(0)e^{2T/\pi}<\frac{\pi}{2\sqrt2}
\]
keeps every adjacent midpoint-lift difference strictly below `pi/2`
in absolute value. If a pair phasor vanishes, the target is unavailable.
Otherwise opposite neighboring signs of `R_i` would turn such a
midpoint gap into a nonacute physical phasor gap. An all-acute physical
cycle therefore requires a common sign. Its principal gaps then equal
the midpoint-lift differences and sum to zero. The winding-one target
is excluded throughout `[0,T]`.

One fixed illustrative policy is `N(0)<=1/100` and `T=1` in the
declared original structural clock. The analytic inequality
\[
 e^{2/\pi}\le\frac1{1-2/\pi}<3
\]
uses `pi>3` and the geometric-series exponential bound, not a
numerically evaluated trajectory or exponential. Consequently
`N(t)<3/100`, well inside the stated sufficient phase collar.
This admits genuine ambient fine-state perturbations. Let the twenty
fine errors `(epsilon_x,epsilon_theta)` use the supplied continuous
phase lifts and the same fixed form/phase units. Pair averaging followed
by centering has Euclidean operator norm `1/sqrt(2)` on the combined
coordinates: the averaging matrix satisfies `A*A^T=I_5/2` and the
centering matrix is an orthogonal projection. Since every original
source has zero centered coarse means,
\[
 N(0)\le\frac1{\sqrt2}
       \|(\epsilon_x,\epsilon_\theta)\|_2.
\]
A full twenty-coordinate error ball of radius `1/100` around any
original source therefore gives `N(0)<=1/(100*sqrt(2))<1/100`.
No exact pair-mean cancellation of those errors is required.
The original source's tight storage and initial activity inequalities
remain statements about its explicitly correlated class. This extension
proves only finite absence of the target; it asserts neither all-time
confinement for nonzero mean errors nor eventual acquisition after `T`.

### 13.3. Finite storage and an equal-invariant target comparison

The exact full storage, derived by summing the four fine edges above
each base edge, is
\[
 E=2\sum_{\{i,j\}}(X_i-X_j)^2+
       4\sum_i u_i^2+
       4\sum_{\{i,j\}}
       [1-\cos(\Theta_j-\Theta_i)\cos\delta_i\cos\delta_j].
\]
The base-edge sums have five terms. At the declared source this becomes
\[
 E=4\sum_i u_i^2+
       4\sum_{\{i,j\}}(1-\cos\delta_i\cos\delta_j).
\]
Both form and lifted phase have zero global mean, and the conservative
complete law retains those means and this storage.

Every initial phase term is strictly positive. Since all initial cosines
are positive and no greater than one,
\[
\begin{aligned}
1-\cos\delta_i\cos\delta_j
 &=(1-\cos\delta_i)+\cos\delta_i(1-\cos\delta_j)\\
 &\le\frac{\delta_i^2+\delta_j^2}{2},\\
4\sum_{\{i,j\}}(1-\cos\delta_i\cos\delta_j)
 &\le4\sum_i\delta_i^2\le\frac1{125}.
\end{aligned}
\]
Together with `1<=u_i<=2`, this proves the finite budget
\[
 \boxed{20<E\le80+\frac1{125}.}
\]
It is a source bound, not a claim that a scalar storage budget certifies
target accessibility.

To separate the obstruction from the conserved scalar totals, set
\[
 \alpha=\frac{2\pi}{5},\qquad
 E_*=20(1-\cos\alpha)<20 .
\]
For each source storage `E`, construct the following algebraic comparison
state on the same support and law:
\[
 X_i=0,\qquad \Theta_i=(i-2)\alpha,\qquad
 \delta_i=0,\qquad
 u_i=v=\sqrt{\frac{E-E_*}{20}}>0.
\]
It has exactly the same storage `E`, the same zero global form mean and
the same zero lifted phase mean. Its pair phasors are nonzero, every
oriented principal gap is `alpha<pi/2`, and its winding is one.
The internal phase rates `dot(delta_i)=v/pi` are nonzero. Thus the
conserved totals themselves do not distinguish these geometries.

This is a comparison state, not a reached endpoint, an acquired source
or a selected finite-time response. In particular,
\[
 v^2=\frac{E-E_*}{20}>\cos\alpha
\]
because `E>20`. In the synchronized pulse restriction it lies above
the libration barrier of Section 9. It therefore receives neither that
libration certificate nor the protected active-family certificate.
Matching storage and means supplies no missing entry or retention proof.

### 13.4. The source information that must change

Internal heterogeneity alone cannot release the proved invariant leaf.
A different, independently specified collective preparation can release
that particular constraint without altering the microscopic law. For
example, retain the common initial midpoint phases and internal source
coordinates, but supply nonconstant collective form `X` with zero
sum. The exact initial phase row is then
\[
 \dot\Theta(0)=\frac1{2\pi}L_{C5}X(0)\ne0 .
\]
Because the base graph is connected, only a constant `X` is in this
Laplacian's kernel. Its image has zero sum, so this is a relative phase
change rather than a common phase rotation. This is a kinematic release
of the invariant constraint, not a winding-formation theorem. It changes
the preparation and its storage, which must be declared afresh.

Similarly, a noncollinear initial collective phase arrangement lies
outside this source class. Neither possibility selects a chirality,
guarantees an accessible target, or authorizes transferring the earlier
dissipative C9 source and its clock to this conservative model.

The scientific conclusion is narrowly conditional: persistent or initially
strong internal activity does not, by itself, organize equal collective
means into the stated acute winding-one identity in this complete law.
The exact mean constraints are essential. Fine-node errors outside the
finite bound above, unequal collective means, a changed observation,
another law or an
additional input require their own admission. This result is compatible
with the existing conditional pulse, active-family maintenance and
finite conservative-preparation results under their different premises.

### 13.5. Static controls and evidence boundary

The existing [replica state reader](../../src/tnfr/physics/relational_sine_scale.py)
and [complete sine field](../../src/tnfr/physics/_sine_flow.py) already
supply the law and retained-coordinate mapping. No new runtime evolution,
state coordinate or source producer is required.

The six
[preparation controls](../../tests/physics/test_sine_replica_preparation.py)
compare the unchanged full twenty-row field against independent fine-edge
arithmetic and the retained factorization. They cover initial activity
and storage, collinear sign and zero boundaries, the equal-invariant
comparison state, the nonuniform-mean directional contrast, and the
actual signed-Laplacian rows underlying the finite absence bound.
Exact inputs, arithmetic allowances and source bytes are frozen before
the first execution. Evaluation status is retained separately in
`artifacts/replica-preparation-obstruction-20261026/attempt-2/control-validation.json`
and that folder's `final-state.json`; this proof does not substitute for
those execution records.

The direct rational field and the graph-captured report have distinct
primitive and numerical-policy contracts. Capture materializes the source
phases; its represented companion is checked at its own admitted coordinates,
with an explicit conversion-error bound. It is not silently assigned to the
original rational source box when rounding crosses an endpoint. Original-source
storage and the equal-energy comparison remain tied to the original state.
The first control attempt and its separate representation correction are
retained in the evidence directory; neither changes the mathematical law.

The all-time exclusion follows from the invariant-leaf and phasor
arguments above, not from finitely many static checks. No trajectory,
source search, Monte Carlo sample, eigenspectrum, old producer replay
or physical observation is part of this result. The research program
alone selects the next preparation question.


<a id="replica-fixed-budget-collective-preparation"></a>

## 14. One fixed-budget preparation with collective form contrast

Section 13 rules out a specific source class because its collective form
and phase means remain uniform. This section changes the preparation
while retaining that complete conservative law. The question is whether
one independently declared redistribution of the original storage can
acquire and retain the same operational winding identity. The answer
requires a new finite response; releasing the invariant constraint does
not supply that response.

The prospective declaration is
`artifacts/replica-fixed-budget-formation-20261026/scope.json`,
with its pre-evaluation
`scope-clarification.json` in the same directory.
Together they fix one source, uncertainty box, horizon, observation, integration
policy and resource limit before evaluating a trajectory. There is no
backward construction from an organized endpoint, parameter scan or
post-response change of the residence requirement.

### 14.1. Exact redistribution and the information supplied by its profile

Keep the ten nodes, twenty unit edges, pair partition, common capacity,
`e=0,w=beta=nu=1` and original time `t` of Section 13.1.
In particular, no row loses its factor `1/pi` and no input,
event or support change is installed.

Select the original rational point
\[
 u^0=(1,5/4,3/2,7/4,2),\qquad
 \delta^0=(1/100,3/250,7/500,2/125,1/50).
\]
Its collective means are zero. Define
\[
 v=(-1,-1,0,1,1),\qquad
 c=\frac{570}{287},\qquad s=\frac{17}{287}.
\]
The contrasting nominal source is
\[
 X(0)=cv,\qquad u(0)=su^0,\qquad
 \Theta(0)=0,\qquad\delta(0)=\delta^0.
\]
Fine coordinates are still `x_i,+/-=X_i+/-u_i` and
`theta_i,+/-=Theta_i+/-delta_i`. All twenty nominal
coordinates are rational; they are not recovered from a rounded graph
snapshot.

For the base-cycle Laplacian,
\[
 L_{C5}v=(-2,-1,0,1,2),\qquad
 v^{\mathsf T}L_{C5}v=6,\qquad
 \|u^0\|_2^2=\frac{95}{8}.
\]
The collective and internal form stores therefore obey the exact identity
\[
 2X^{\mathsf T}L_{C5}X+4\|u\|_2^2
 =12c^2+\frac{95}{2}s^2
 =\frac{95}{2}.
\]
The initial phase coordinates are identical in the two preparations, so
their phase stores also agree exactly. Thus the two nominal sources have
the same complete storage, not merely the same internal energy or a
matched diagnostic. Their global form and lifted-phase means are zero.

The nominal initial collective phase velocity is
\[
 \dot\Theta(0)=\frac{c}{2\pi}(-2,-1,0,1,2).
\]
This spatial direction is information supplied by the preparation. An
eventual positive winding would not demonstrate spontaneous selection
of a chirality from an unoriented source. The profile is independent of
any computed future response; its initial velocity is not a prediction
that the nonlinear trajectory remains a linear ramp. Heterogeneous
internal coordinates are retained rather than replaced by a synchronized
or rigid pair state.

Both preparations start outside the target. Their pair phasors initially
have nearly equal arguments and winding zero, whereas the target below
requires positive winding and a positive angular margin.

### 14.2. Independent fine-coordinate errors and the shared storage ceiling

For each source center admit the independent twenty-coordinate box
\[
 |x_{i,\pm}(0)-x^*_{i,\pm}|\le\varepsilon,\qquad
 |\theta_{i,\pm}(0)-\theta^*_{i,\pm}|\le\varepsilon,\qquad
 \varepsilon=2^{-20}.
\]
Phases are supplied continuous lifts in the same fixed units as the
center. Pair averaging and half-differencing imply errors at most
`epsilon` in each initial `X_i,u_i,Theta_i,delta_i`.
The boxes do not impose exact pair means or correlated energy errors.

Only the nominal centers have equal storage member by member. The two
independent uncertainty boxes instead share a proved upper budget.
Every nominal fine-edge form difference has absolute value less than
five: for the contrasting source a sufficient bound is
`2c+4s<5`, and the original point has an even smaller bound.
Writing the change of an edge difference as `d` gives
`|d|<=2epsilon` and
\[
 \left|\frac{(a+d)^2-a^2}{2}\right|
 \le 10\varepsilon+2\varepsilon^2.
\]
The cosine potential is globally unit-Lipschitz in its argument.
Summing over the actual twenty edges therefore bounds the change of
complete storage by
\[
 |\Delta E|\le240\varepsilon+40\varepsilon^2.
\]
Section 13.3, now at the specified original point, gives
\[
 E^*\le\frac{95}{2}+\frac1{125}.
\]
Consequently both full uncertainty boxes satisfy
\[
 E\le \frac{95}{2}+\frac1{125}
          +240\varepsilon+40\varepsilon^2<48.
\]
This is an actual whole-law storage bound. Its conservation retains all
fine form and phase coordinates; no unobserved reservoir or extra source
work is omitted.

### 14.3. Finite domain control before evaluating acquisition

The fixed horizon is `H=6`. For the contrasting source define
\[
 r_i=\sqrt{u_i^2+\delta_i^2}.
\]
The exact retained rows in Section 13.1 imply, for arbitrary collective
phases and arbitrary signs of neighboring resultants,
\[
 |\dot u_i|\le\frac{|\delta_i|}{\pi},\qquad
 |\dot\delta_i|=\frac{|u_i|}{\pi},\qquad
 \|(\dot u_i,\dot\delta_i)\|_2\le\frac{r_i}{\pi}.
\]
The differential inequality for the radius, or its squared-norm form,
gives
\[
 r_i(0)e^{-t/\pi}\le r_i(t)\le r_i(0)e^{t/\pi}.
\]
This estimate neither assumes a positive restoring coefficient nor
replaces the actual collective phase history by a prescribed function.

The independent error box satisfies the strict rational bounds
\[
 \frac1{20}<r_i(0)<\frac18.
\]
For example, the lower bound already follows from
`17/287-epsilon>1/20`; the largest permitted internal form and
phase components satisfy
`(34/287+epsilon)^2+(1/50+epsilon)^2<1/64`.
A fixed elementary majorant, rather than an evaluated exponential, is
\[
 e^{6/\pi}
 <\left(1-\frac{6}{16\pi}\right)^{-16}
 <\left(\frac{628}{553}\right)^{16}<8,
\]
using `pi>157/50` and the exact rational last inequality.
It follows throughout `0<=t<=6` that
\[
 \frac1{160}<r_i(t)<1,\qquad
 |\delta_i(t)|<1,\qquad
 R_i(t)=\cos\delta_i(t)>\frac12.
\]
Thus no vanishing pair phasor or change of its midpoint branch can
explain a failure of the selected acquisition test. On this full horizon
the normalized physical pair phasor is exactly `exp(iTheta_i)`.
These bounds apply to the contrasting source box, not to arbitrary
high-activity internal preparations.

For the original comparison box use the different, already proved
estimate of Section 13.2. The centered collective norm initially obeys
\[
 N(0)\le\sqrt{10}\,\varepsilon.
\]
Since `exp(12/pi)<81`, its whole horizon satisfies
\[
 N(t)<81\sqrt{10}\,\varepsilon
       <\frac{\pi}{2\sqrt2}\qquad(0\le t\le6).
\]
The global collinearity argument with this finite midpoint-gap collar
excludes acute winding one for that entire comparison box, even if its
internal resultants change signs or vanish. This uses the finite
perturbation theorem; it does not mislabel independent errors as members
of the exact common-mean invariant leaf.

There is also a necessary travel time for the contrasting source.
For an adjacent midpoint gap, put `b=e_j-e_i`. The actual
coarse phase row and positive base Laplacian give
\[
 \frac{d}{dt}(\Theta_j-\Theta_i)
       =\frac{b^{\mathsf T}L_{C5}X}{2\pi},\qquad
 b^{\mathsf T}L_{C5}b=6.
\]
The collective form store is
`F_C=2X^T L_C5 X<=E`. Cauchy--Schwarz therefore yields
\[
 |(\Theta_j-\Theta_i)'|
 \le\frac{\sqrt{3F_C}}{2\pi}
 \le\frac{\sqrt{3E}}{2\pi}<\frac6\pi.
\]
Initially each raw adjacent midpoint gap has absolute value at most
`2epsilon`. Positive resultants and acute winding one require
at least one raw continuous gap to be below `-3pi/2`:
the five principal gaps sum to `2pi` whereas the five raw
gaps telescope to zero. In particular,
\[
 |\Theta_j(t)-\Theta_i(t)|
 \le2\varepsilon+\frac{12}{\pi}
 <\frac92<\frac{3\pi}{2}\qquad(0\le t\le2)
\]
excludes acquisition through `t=2`. This is a necessary
access bound, not a claim that acquisition occurs after it expires.

### 14.4. A prospective whole-window identity and activity requirement

Fix `mu=1/100`. On every admitted target time require all
five oriented physical pair-phasor principal gaps to satisfy
\[
 \mu\le\omega_i\le\frac{\pi}{2}-\mu,\qquad
 \sum_i\omega_i=2\pi.
\]
This is a positive-winding identity with a nonzero angular margin.
The resultants and internal radii retain the independent bounds of
Section 14.3.

It also supplies a quantitative internal activity bound. Write the
actual internal form row as
\[
 \dot u_i=-A_i\sin\delta_i,\qquad
 A_i=\frac1{2\pi}
       \sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i).
\]
On the target, `R_j>1/2` and
`cos(Theta_j-Theta_i)>=sin(mu)`, so
`A_i>sin(mu)/(2pi)`.
For `|delta_i|<1` the elementary sine bound gives
`|sin(delta_i)|>=(5/6)|delta_i|`. Along with
`delta_dot_i=u_i/pi`, this proves
\[
 \|(\dot u_i,\dot\delta_i)\|_2
 \ge\frac{5\sin\mu}{12\pi}r_i
 >\frac{5\sin\mu}{1920\pi}>0.
\]
Thus the operational target includes ongoing internal motion of every
pair. It does not assert that every one of the twenty individual nodal
derivative components is nonzero at every instant.

The fixed numerical observation uses whole-step enclosures, not endpoint
or sampled pictures. For each cell it retains continuous midpoint lifts
and admits integers `n_i` such that
\[
 \omega_i=\Theta_{i+1}-\Theta_i+2\pi n_i
           \ \in[\mu,\pi/2-\mu],\qquad
 \sum_i n_i=1.
\]
All source trajectories in a certified tube must satisfy these
inequalities on the same cell. The bounded enumeration
`n_i in {-2,-1,0,1,2}` is sufficient here: the source and
travel estimate give
`|Theta_{i+1}-Theta_i|<=2epsilon+36/pi<13` throughout the
horizon, ruling out a larger wrap integer for a gap in the target range.
This observation does not wrap or reset any coordinate passed to the
evolution field.

Declare a residence duration of `1/2` in original time.
Four consecutive certified cells of length `1/8` establish
one continuous residence interval for the entire source box. Since all
sources initially have winding zero, this would constitute finite
operational acquisition and retention for the stated identity. It
would not establish an earliest entry time, permanent maintenance,
entry into the two-sided invariant family of Section 12, or an
attractive basin.

### 14.5. Frozen evaluation policy

The source consumer
[replica collective preparation](../../src/tnfr/research/replica_collective_preparation.py)
reuses the existing
[complete field](../../src/tnfr/physics/_sine_flow.py) and shared
validated box Taylor step. Its prospective policy is one complete
twenty-coordinate evolution with horizon `6`, cell length
`1/8`, Taylor order `8` and `48` cells.
Exact rational initialization avoids graph recapture. The solver's
domain is the smooth full sine law; target admission must never be
used as a condition for the existence of its integration domain.

Outward interval arithmetic may slightly enlarge the represented initial
box. It must contain every member of the declared exact box; it does not
redefine that source. The analytic source and radius statements above
apply to the intended family, while an accepted numerical tube encloses
its trajectories. Rounded enclosure endpoints are not silently promoted
to exact prepared coordinates or new source members.

The frozen worker limit is `900` wall-clock seconds, with
a status checkpoint at `300` seconds and a retained checkpoint
after each completed cell. An inclusion failure or timeout stops the
attempt and retains its completed prefix and reason. There is no
post-response refinement, budget extension, changed source, changed
horizon or weakened residence criterion in this gate.

The static
[controls](../../tests/physics/test_replica_collective_preparation.py)
check the source, full-law accounting, uncertainty bounds, observation,
residence bookkeeping and shared-step contract before the frozen first
trajectory. Mathematical premises, static controls and evaluated
continuous enclosures retain separate evidence roles.

The derivations above establish admission, internal-domain control and
exclusions, but not arrival.
The record separates residence certification from integration completion:

- whether four consecutive cells certify residence, retaining that positive
  finite certificate even if a later cell is incomplete;
- whether all forty-eight cells were completed;
- the reason and retained prefix if integration is incomplete.

A failed sufficient grid detector does not exclude a shifted residence
window or prove physical absence. A wide enclosure is not an observed
failure of the mathematical trajectory. No empirical or particle
identification follows from a successful organization certificate.
The research program selects any subsequent work only after assessing
the frozen evidence.

### 14.6. Evaluated finite contrast: no residence certified

The single frozen execution completed all 48 validated cells through
`t=6`, without an inclusion failure, retry or timeout. None of its
whole-cell midpoint enclosures passed the sufficient target detector,
so no four-cell residence was certified. The result is
`residence_not_certified` with `integration_complete=true`.
It is not a proof that the trajectory or source family avoids every
qualifying interval.

All 48 retained tubes admitted the direct pair chart and the declared
raw-gap range. The lack of a positive residence certificate therefore
cannot be attributed to a recorded loss of these two observation domains.
It still does not distinguish genuine target separation from crossings
within a cell, different responses inside the source box, or information
lost by a rectangular enclosure. In particular,
the initial linear-ramp comparison cannot substitute for the integrated
nonlinear response.

Seven frozen controls passed, with no failed or skipped cases. The
post-run prefix audit rebuilt the source, continuous-cell associations,
endpoint arithmetic and observations from the retained records. It did
not independently recompute derivative bounds, Picard inclusion or another
trajectory. The complete law, source, horizon, grid, order and tolerances
remained unchanged throughout the attempt.

Evidence is retained under
`artifacts/replica-fixed-budget-formation-20261026/`:
`validation-plan.json` and `evaluated-source.zip` bind the reviewed
pre-response sources; `control-validation.json` retains the seven controls;
`execution-result.json`, `trajectory/trajectory-result.json` and the 48
individual cell records retain the response; `prefix-audit.json` retains
the consistency audit. Later editorial consolidation does not replace the
pre-response archive.

The scientific distinction is now sharper. Equal total storage does not
remove the original common-mean source obstruction. A declared collective
contrast releases that invariant restriction and admits well-defined pair
phases throughout the chosen horizon, but this one finite test has not
certified the target organization. Before changing the preparation or
continuing an integration, the next question is whether the existing
validated tubes can certify target separation or instead expose a specific
observation or enclosure limitation. That question remains unevaluated.

<a id="replica-target-separation"></a>

## 15. Target separation and closed residence from retained tubes

Section 14 completed the prescribed evolution without a positive
residence certificate. That outcome alone does not decide whether the
target was absent, briefly visited, or unresolved by a whole-cell
enclosure. This section defines a separate sufficient exclusion test
and a temporal-cover argument for that same evidence.

The new declaration is
`artifacts/replica-target-separation-20261026/scope.json`.
The forty-eight tubes already exist and their previous outcome is
known. This is a newly frozen analysis of retained evidence, not a
second independent predictive experiment. No source, complete law,
clock, uncertainty, target margin or integration policy is changed.

### 15.1. Admission and the distinction between a box and its source

Retain all twenty fine form and continuous phase coordinates. For an
arbitrary supplied interval box, normalize and admit every exact
coordinate endpoint before computing its pair means or trigonometric
bounds. Define the midpoint and internal-phase intervals
\[
 T_i=\frac{\theta_{i,+}+\theta_{i,-}}2,\qquad
 D_i=\frac{\theta_{i,+}-\theta_{i,-}}2,\qquad
 I_i=T_{i+1}-T_i.
\]
These are outward enclosures; the notation does not assign independent
physical errors to every derived difference.

The direct physical observation domain requires
\[
 \sup|D_i|<1,\qquad \sup|I_i|<13
 \quad\text{for every }i.
\]
The first condition ensures positive pair resultants, so the normalized
physical pair phasors have arguments `Theta_i`. The second
makes the finite wrap enumeration below exhaustive. Neither condition
establishes membership in the prepared source family.

An arbitrary box that lacks this sufficient domain receives an
unavailable observation and an unresolved classification. In particular,
a failure of the positive pair chart is not itself a certificate of
target separation. The global source theorem in Section 14 remains
valid, but this new direct observer does not silently substitute a
source-associated theorem for missing admission of an arbitrary box.

For a bound retained cell, the source, time and enclosure association
are separately re-admitted and bound to retained evidence as described in Section 15.4. The complete
operational target also includes internal activity. An arbitrary
geometric observation does not certify that activity. For the retained
source, Section 14's radius bound and its conditional positive activity
estimate supply the additional obligation whenever the target geometry
is certified.

### 15.2. A necessary wrap test, retaining closed endpoint contact

Let `P=[p_l,p_u]` be the shared enclosure of mathematical
`pi`, with `3<p_l<=pi<=p_u<4`, and keep
`mu=1/100`. The true target interval
\[
 [\mu,\pi/2-\mu]
\]
is contained in the closed outer interval
\[
 J^+=[\mu,p_u/2-\mu].
\]
For every oriented edge form the possible-wrap set
\[
 N_i=\left\{n\in\{-2,-1,0,1,2\}:
       (I_i+2nP)\cap J^+\ne\varnothing\right\}.
\]
All arithmetic is outward, including multiplication by a negative
integer. Intersection is closed: touching at either endpoint remains
possible. Equality must not be converted into a strict separation.

To prove that the omitted integers cannot realize the target, take an
actual raw gap `g` with `|g|<13`. For `n>=3`,
\[
 g+2n\pi>-13+6\pi>5>\pi/2-\mu.
\]
For `n<=-3`, the corresponding upper bound is
`13-6pi<-5<mu`. Therefore every physical target assignment
uses an integer in the enumerated range.

The actual raw gaps telescope:
\[
 \sum_{i=0}^4(\Theta_{i+1}-\Theta_i)=0.
\]
Consequently positive unit winding requires wrap integers satisfying
\[
 \sum_{i=0}^4 n_i=1.
\]
Compute the finite reachable-sum sets by exact integer arithmetic,
\[
 S_0=\{0\},\qquad
 S_{k+1}=\{s+n:s\in S_k,\ n\in N_k\}.
\]
Every actual target state in the box supplies a member of every
`N_i` and hence a witness `1 in S_5`. Thus
\[
 \boxed{1\notin S_5\quad\Longrightarrow\quad
        \text{the whole admitted box is separated from the target geometry}.}
\]

This implication is one-way. The same value of `pi` and the
same full state must realize all physical gaps. The interval calculation
discards some of these correlations and can therefore retain impossible
combinations. It only enlarges the set of possibilities, which preserves
the exclusion theorem. A surviving integer sum does not establish a
target point, a reachable source member or a trajectory crossing.

The three-way observation retains the existing positive whole-box
detector from Section 14:

- **Membership certified:** the directly admitted box passes that
  positive detector, including the required integer sum.
- **Separation certified:** the domain is admitted and `1 not in S_5`.
- **Unresolved:** neither sufficient conclusion is established, including
  failure of the direct observation domain.

A positive whole-box branch necessarily intersects `J^+`,
so it contributes a possible wrap to the corresponding `N_i`.
The positive and negative certificates cannot both hold for the same
admitted box. A cached positive failure is not an input to the new
negative theorem.

### 15.3. A time cover that includes shifted windows and boundary equality

Let the complete fixed cells be the closed intervals
\[
 C_k=[kh,(k+1)h],\qquad h=1/8,\qquad k=0,\ldots,47,
 \qquad H=6.
\]
A universally separated whole-time tube excludes the target at every
time in its cell, including both endpoints. Form the closed forbidden
set
\[
 F=\bigcup_{\text{separation certified on }C_k}C_k.
\]
Every target time of every trajectory in the admitted source family
must belong to the remaining set
\[
 A=[0,H]\setminus F.
\]
Membership-certified and unresolved cells both remain possible for this
cover. Missing, corrupt or unavailable evidence must never be inserted
into `F`.

Each maximal run of nonseparated cells gives a possible component with
hull `[a,b]`. An endpoint adjacent to a separated cell is
excluded from the component. The domain endpoint `0` or
`H` is included when it is not itself covered by a separated
cell. Thus the hull's length alone is insufficient at equality.

The selected residence is a **closed** interval
`[s,s+ell]` wholly contained in `[0,H]`, with
`ell=1/2`. A component cannot contain such a residence if
\[
 b-a<\ell,
 \quad\text{or}\quad
 b-a=\ell\ \text{and at least one endpoint is excluded}.
\]
Indeed, a contained closed interval cannot be longer than the hull.
At equal length its only possible endpoints are exactly `a,b`,
which must both belong to the component. An interval lying in
`A` cannot cross a forbidden cell or point to connect
different components.

It follows that
\[
 \boxed{\text{if every component satisfies the preceding exclusion,
 no source trajectory has the declared closed half-unit residence}.}
\]
This covers windows shifted relative to the numerical grid. It does
not assume that a residence begins at a retained cell boundary.

For the present complete grid, a nonseparated run of four cells has
length exactly `1/2`. Because it is shorter than the full
horizon, at least one end is adjacent to a separated cell; that end
is excluded. Such a run therefore cannot contain the closed residence.
The same holds for shorter runs. A run of five or more cells leaves
a possible interval of sufficient length and this test alone is
unresolved.

The endpoint rule also handles boundary runs at `0` or
`H`. Equality with both endpoints included is not an
exclusion: the closed hull itself is a possible residence interval.
If every cell separates, there are no possible components and the
stronger conclusion is exclusion of every target time. Otherwise,
residence exclusion may coexist with unresolved brief visits.
These are different statements.

The cover bounds possible times uniformly over the whole source box.
An unresolved component is not evidence that any particular source
member enters the target. Any independently retained positive
residence certificate must also be preserved; incomplete or negative
information cannot erase it. A membership-certified closed cell cannot
be adjacent to a separation-certified closed cell for the same nonempty
source family: their shared time would require both target membership
and its exclusion. Such inconsistent labels are rejected rather than
converted into a temporal certificate.

### 15.4. Retained evidence and the limits of a detached audit

The new consumer
[replica target separation](../../src/tnfr/research/replica_target_separation.py)
binds the previously recorded final-state receipt before applying the
criterion. It checks the required source, attempt, plan, frozen archive
and all forty-eight cell receipts with the shared exact artifact and
bounded JSON owners. This content association is necessary for reusing
the original validated enclosures; a file name or cached conclusion
does not establish it.

The reader rebuilds the fixed source and exact scalar admission, full
state dimensions, cell times and order, initial/endpoint chain, complete
Taylor coefficient arrays, remainder dimensions and retained endpoint
arithmetic. Pair coordinates and positive observations reuse their
existing owners. The new negative calculation consumes fresh derived
intervals, not previously stored classification flags.

This remains a conditional reuse of the validated producer's records.
The detached reader does not recompute the field derivatives, establish
a second Picard inclusion or integrate another trajectory. Successful
receipt and arithmetic checks must not be described as independent
reproduction of those dynamical enclosures.

A complete forty-eight-cell cover is required for the full-horizon
temporal claim. Evidence availability is separate from the three
geometric outcomes. A missing or malformed record yields unavailable
analysis, not a new separation cell. The prospective resource bound is
one detached assessment with a `300`-second command limit,
individual JSON records at most `16 MiB` and a frozen archive
at most `128 MiB`, subject to the shared archive checks.

The independent
[observer controls](../../tests/physics/test_replica_target_separation.py)
cover constructed positive, negative and unresolved cases; closed
boundary contact; wrap-sum obstruction; missing domain; shifted,
equal-length and boundary residence cases; exact input admission;
retained source/chain/receipt failures; and independence from cached
verdicts. They precede the first application of the newly frozen
criterion to the existing response.

The prospective derivation alone supplies no classification of the
forty-eight cells or their possible-time components. The previous
`residence_not_certified` result and its evaluated bytes remain unchanged.
The separate application below concerns only the stated finite source,
horizon, margin and residence policy. It does not identify a physical
object or select a fundamental law.

### 15.5. Retained response: the target is excluded throughout the horizon

The single frozen assessment classified all forty-eight whole-time tubes
as **separated**, with no unavailable or unresolved observation. Every
tube retained the direct pair chart and exhaustive raw-gap range.
The resulting possible-time cover is empty. Consequently, conditional
on the previously validated enclosures, every member of the declared
initial uncertainty box avoids the specified target at every time in
`[0,6]`, including both domain endpoints.

This is stronger than exclusion of a closed half-unit residence. It also
excludes a brief visit to this same target within the retained horizon.
It does not exclude a different geometry, another source or error family,
later times, another angular margin, or organization under a different
complete law. The original positive detector's lack of a certificate is
not the proof: the new necessary-wrap obstruction supplies the negative
conclusion.

The computed obstruction has two forms. In 47 cells at least one oriented
edge has no possible target wrap. In the remaining cell every edge has a
possible wrap, but the only reachable sum is zero, not the required one.
Local scalar overlaps therefore cannot substitute for joint cycle
compatibility. These are sufficient enclosure obstructions; they do not
by themselves explain the dynamical mechanism that generated the phase
profile.

Eight frozen controls passed without failures or skipped cases. The
detached reader rebound 234 evidence and production-source receipts,
re-admitted all twenty coordinates and the complete cell chain, and
rebuilt the retained endpoint arithmetic. It did not recalculate field
derivatives or Picard bounds, run the old producer, or integrate a new
trajectory. Independent output reviews checked the stored verdicts,
associations and scope without reclassifying the original response.

The evidence directory is
`artifacts/replica-target-separation-20261026/`.
`validation-plan.json` and `evaluated-source.zip` preserve the new
criterion before its first application; `control-validation.json`
retains the eight controls; `assessment-result.json` retains every
observation and the temporal cover; `evaluation-result.json` retains
the bounded assessment's execution record. The separate original
trajectory and its pre-response producer archive remain unchanged.

The fixed-budget source comparison now has a finite negative answer:
this one redistribution releases the exact common-mean invariant
constraint but does not acquire the chosen winding identity by time six.
The total storage and well-defined pair phases are insufficient to
guarantee that acquisition. This scoped result does not select a new law,
justify retuning a source, or establish a general impossibility theorem
for conservative TNFR organization.

<a id="replica-modal-accessibility"></a>

## 16. A source-specific lower bound on necessary nonlinear transport

The finite exclusion in Section 15 does not explain which features of a
preparation can coordinate the target gaps. This section derives a
different result from the source and complete equations, without reading
the retained trajectory: the fixed unforced harmonic reference cannot
reach the target, and any actual acquisition requires a quantified
departure from that reference.

The declaration is
`artifacts/replica-modal-accessibility-20261026/scope.json`.
Keep the source, independent errors, positive pair chart, horizon,
original clock and target of Section 14. No source profile, margin or
law is changed. The
[exact pair-state rows](SINE_PAIR_STATE.md#sine-replica-inheritance),
their internal-state closure and their storage identity remain the
owners of those results. A modal coordinate change alone would add no
new closure theorem.

### 16.1. Exact collective modes with the internal history retained

Orient the base C5 edges from `i` to `i+1` and let
`B` be its incidence matrix, with
\[
 g=B\Theta,\qquad g_i=\Theta_{i+1}-\Theta_i,\qquad L=B^{\mathsf T}B.
\]
Write
\[
 R_i=\cos\delta_i,\qquad A_i=R_iR_{i+1},\qquad
 h_i=g_i-A_i\sin g_i,\qquad
 \eta=\frac1{2\pi}B^{\mathsf T}h.
\]
The complete collective equations are exactly
\[
 \dot X=-\frac{L\Theta}{2\pi}+\eta,\qquad
 \dot\Theta=\frac{LX}{2\pi}.
\]
The first equality follows from
`X_dot=-B^T(A sin g)/(2pi)`, including the orientation of
both incident edges. The internal rows of Section 13 are still present;
neither `A` nor its history is an independent prescribed
input.

In the fixed normalized coordinate units, package the two real vectors as
\[
 z=X+i\Theta,\qquad
 \dot z=\frac{iLz}{2\pi}+\eta.
\]
This notation is not a physical complex amplitude, a Born probability
or an identification of primitive form with phase.

The actual C5 Laplacian has eigenvalues
\[
 0,\qquad\lambda_-=\frac{5-\sqrt5}{2},\qquad
 \lambda_+=\frac{5+\sqrt5}{2}.
\]
With `P_0=11^T/5` and `P_c=I-P_0`, its real
orthogonal projectors are
\[
 P_-=\frac{\lambda_+P_c-L}{\sqrt5},\qquad
 P_+=\frac{L-\lambda_-P_c}{\sqrt5}.
\]
They satisfy
`P_0+P_-+P_+=I`, `P_jP_k=0` for unequal
indices, and `LP_\pm=lambda_\pm P_\pm`. Their ranks are
one, two and two. These identities follow from the cycle Laplacian and
`L(L^2-5L+5I)=0`; no numerical eigenspectrum is required.

The projected equations retain the source of their coupling:
\[
 (P_\pm z)'=\frac{i\lambda_\pm}{2\pi}P_\pm z+P_\pm\eta,
 \qquad (P_0z)'=0.
\]
The common mode vanishes from the divergence of the residual, not from
a reset of the form or phase origin.

Define the unforced harmonic reference with the **same initial
collective state of each source member**:
\[
 z_{\rm free}(t)=U(t)z(0),\qquad
 U(t)=P_0+e^{i\lambda_-t/(2\pi)}P_-
             +e^{i\lambda_+t/(2\pi)}P_+.
\]
The reference drops `eta`; the actual law does not. Since
`U` is unitary, variation of constants gives the global
identity and estimate
\[
 z(t)-z_{\rm free}(t)
   =\int_0^tU(t-s)\eta(s)\,ds,\qquad
 \|z(t)-z_{\rm free}(t)\|_2
   \le\int_0^t\|\eta(s)\|_2\,ds.
\]
This does not turn the state-dependent residual into an autonomous
predictor. It retains the complete nonlinear and internal history.
No exponential comparison growth, commuting approximation to a
different operator, or dissipative theorem is substituted for this
unitary identity.

### 16.2. The declared profile has a global harmonic gap envelope

For the nominal source `X(0)=cv`, `Theta(0)=0`,
with `c=570/287` and `v=(-1,-1,0,1,1)`,
the free phase vector has the form
\[
 \Theta_{\rm free}^*(t)=(-a(t),-b(t),0,b(t),a(t)).
\]
Put `s_\pm(t)=sin(lambda_\pm t/(2pi))`. Direct projection
of the actual profile yields
\[
\begin{aligned}
 a/c&=\frac{1+\sqrt5}{2\sqrt5}s_-
       +\frac{\sqrt5-1}{2\sqrt5}s_+,\\
 b/c&=\frac{3+\sqrt5}{2\sqrt5}s_-
       +\frac{\sqrt5-3}{2\sqrt5}s_+.
\end{aligned}
\]
The coefficients in the first row are positive and sum to one. The
second row has total absolute coefficient sum `3/sqrt5`,
while
`(a-b)/c=(s_+-s_-)/sqrt5`. Therefore, for every reference
time,
\[
 |a|\le c,\qquad |b|\le\frac{3c}{\sqrt5},\qquad
 |a-b|\le\frac{2c}{\sqrt5}.
\]
The five raw gaps are
`(a-b,b,b,a-b,-2a)`. Consequently their absolute values
never exceed `2c`. This envelope uses the source coefficients,
not a computed crossing time or a small-time Taylor approximation.

Let
\[
 \varepsilon_c=2^{-20}+2^{-128}
\]
cover both the intended independent fine errors and their outward
initial interval enclosure. Pair means then differ from the nominal
means by at most `epsilon_c` per coordinate. Their complex
initial error has norm at most `sqrt(10)*epsilon_c`.
Unitary propagation preserves that norm, so the error in any single
reference phase gap is at most
`sqrt(20)*epsilon_c<5epsilon_c`. Every memberwise harmonic
reference thus obeys
\[
 \boxed{|g_{{\rm free},i}(t)|\le
 M_{\rm free}:=2c+5\varepsilon_c\quad\text{for every }t.}
\]
The common phase origin cancels in these gaps. The calculation does
not replace the declared member's initial state by the nominal one.

### 16.3. Simultaneous target gaps require a nonzero transport budget

On the admitted positive-resultant chart, the target has
\[
 \omega_i\in[\mu,\pi/2-\mu],\qquad
 g_i=\omega_i-2\pi n_i,\qquad
 \sum_i n_i=1,\qquad \mu=1/100.
\]
At least one integer `n_i` is positive. At that edge,
\[
 g_i\le-\left(\frac{3\pi}{2}+\mu\right)=:-S.
\]
The free gap envelope is strictly smaller than `S`.
At any actual target time the difference of those two gaps is at least
`S-M_free`, and the edge-difference functional has Euclidean
norm `sqrt2`. Hence
\[
 \boxed{\|z(t)-z_{\rm free}(t)\|_2\ge
 K:=\frac{S-M_{\rm free}}{\sqrt2},\qquad
 \int_0^t\|\eta(s)\|_2\,ds\ge K.}
\]

The necessary threshold is bounded away from zero without evaluating
a response. Using `pi>157/50`, `c<2` and
`epsilon_c<1/10000` gives
\[
 S-M_{\rm free}>
 \frac{1439}{2000}>\frac57>\frac{\sqrt2}{2},
 \qquad\boxed{K>\frac12}.
\]
Thus the source-specific harmonic family cannot itself reach the target.
The envelope does not prohibit every winding-one configuration or a
branch change: it can permit a nonacute gap. The excluded geometry is
the specified simultaneous acute-margin target.
The conclusion is about this reference and preparation, not about the
full conservative sine law.

A separately justified source-based bound
\[
 \int_0^t\|\eta(s)\|_2\,ds\le\mathcal B(t)<K
\]
would exclude target acquisition at that time. A loose upper bound
above `K` leaves this criterion unresolved; it does not prove
that the real residual or state error is that large. Conversely,
crossing the necessary budget is not a sufficient acquisition
condition: the correction must have the appropriate signed modal
history and coordinate all five target gaps.

The identity retains that missing predictive input explicitly. This
section evaluates neither an actual residual-history budget nor a
surrogate trajectory. It does not reuse the known exclusion in
Section 15 as an independent test of a source-based prediction.

### 16.4. A mandatory instantaneous defect at the target

A second obstruction concerns a uniformly small residual relative to
the same harmonic reference. On the retained source's target chart,
`1/4<A_i<=1`. Define
\[
 q_i=\omega_i-A_i\sin\omega_i.
\]
Monotonicity in `omega_i` and `A_i` gives
\[
 \mu-\sin\mu\le q_i\le
       \frac{\pi}{2}-\mu-\frac{\cos\mu}{4}.
\]
The width
\[
 D=\frac{\pi}{2}-2\mu+\sin\mu-\frac{\cos\mu}{4}
\]
is less than `pi/2`. Since sine is periodic,
\[
 h_i=q_i-2\pi n_i.
\]
The integer cycle vector `n` is nonconstant: a constant
integer on five edges could not have sum one. At least two of its
cyclic differences are nonzero. At each such position,
\[
 |h_i-h_{i-1}|\ge2\pi-D.
\]
The corresponding entries of `B^T h` therefore yield
\[
 \boxed{\|\eta\|_2\ge
 \sqrt2\left(1-\frac{D}{2\pi}\right)
 >\frac{3\sqrt2}{4}>1.}
\]
This bound holds for every lift assignment realizing the specified
target within this harmonic-reference definition. It does not rely
on a single edge residual whose divergence might cancel.

If a closed half-unit target residence existed, its absolute residual
budget would exceed `1/2` over that window. This is a bound
on the integral of a norm. It is not a lower bound on the norm of the
Duhamel integral: temporal and modal cancellations remain possible.

The reference qualification is essential. Consider the constructed
synchronized critical point
`X=0,u=delta=0,Theta_i=(i-2)*2pi/5`. The full sine field
vanishes, but
\[
 \eta=\frac{L\Theta}{2\pi}=(-1,0,0,0,1),
 \qquad\|\eta\|_2=\sqrt2.
\]
It cancels the nonzero harmonic drift exactly. This is a static
geometric control, not a reached active endpoint of the current
source. A local affine expansion about that winding critical point
can have zero residual at its center; the bound above does not apply
to every possible reference.

Individual full-turn changes of phase representatives likewise change
`LTheta` and the chosen harmonic comparison, while the
physical sine field is unchanged and the residual compensates.
The source-specific gap and state-distance bounds retain the fixed
initial lifts. They are not a lift-invariant physical metric.
`eta` is an endogenous term, not an applied force or an
unaccounted input to the full law.

### 16.5. A short independent check using an auxiliary quadratic quantity

In the same fixed lifts define
\[
 Q_{\rm harm}=2(X^{\mathsf T}LX+\Theta^{\mathsf T}L\Theta).
\]
The free reference conserves this quantity. Under the actual law,
cancellation of the two linear rows instead gives
\[
 \dot Q_{\rm harm}=4X^{\mathsf T}L\eta.
\]
The nominal profile and independent initial mean errors imply
\[
 Q_{\rm harm}(0)\le
 12c^2+32c\varepsilon_c+80\varepsilon_c^2<48.
\]
For a target raw gap of magnitude at least `S`, the other
four raw gaps sum to its opposite. Cauchy--Schwarz then gives
\[
 \sum_i g_i^2\ge\frac54S^2,\qquad
 Q_{\rm harm}(\text{target})\ge\frac52S^2>\frac{405}{8}.
\]
Any actual access would consequently require
\[
 \int_0^t4X(s)^{\mathsf T}L\eta(s)\,ds>\frac{21}{8}.
\]
This signed auxiliary balance is consistent with the gap criterion,
but it is not a second physical energy store or a new conservation
law. `Q_harm` depends on the chosen phase lifts and reference.
The conserved full fine storage continues to contain the periodic
cosine potential and retained internal coordinates; no physical
energy has been created by this inequality.

<a id="replica-modal-criterion-result"></a>

### 16.6. Validated algebraic result and scope

The private
[modal criterion](../../src/tnfr/research/replica_modal_criterion.py)
and its
[constructed controls](../../tests/physics/test_replica_modal_criterion.py)
use the shared exact matrix, interval and trigonometric owners. The
controls compare the original twenty-row field with the transformed
equations, verify the C5 projectors and source profile envelope, retain
initialization and error budgets, and distinguish the prepared-twist
cancellation from an externally forced or universally small-residual
interpretation.

Seven independently reviewed constructed controls passed, with no skips.
The frozen algebraic source certificate verifies all four obligations:
the source-specific harmonic target exclusion, necessary nonlinear
transport, mandatory pointwise target residual, and signed auxiliary work.
Its exact outward bounds imply the following conservative decimal display:

| Bound | Certified value in the declared normalized coordinates |
| --- | --- |
| Every memberwise harmonic raw-gap magnitude | At most `3.972130204` |
| Necessary actual state deviation and absolute residual budget at target | At least `0.530513068` |
| Necessary instantaneous residual norm at target | At least `1.119177956` |
| Initial auxiliary quadratic quantity | At most `47.333402037` |
| Target auxiliary quadratic quantity | At least `55.752394205` |
| Necessary signed auxiliary work before target access | At least `8.418992168` |

These are source/target inequalities, not measured values of an actual
response. The small outward presentation rounding does not replace the
exact fractions in the certificate. Physical pair-phase comparison remains
conditional on the proved source chart through structural time six; the
harmonic gap envelope itself holds at every reference time.

The implementation tests include the complete twenty-row field, independent
high-precision Fourier and sine evaluations, exact projector algebra,
nonzero internal motion with stationary collective coordinates, a positive
auxiliary rate, full-turn phase changes and invalid primitive inputs.
The twist control therefore tests actual cancellation rather than treating
an instantaneous norm as accumulated movement.

The local evidence is retained in
`artifacts/replica-modal-accessibility-20261026/`: the prospective scope,
independent source reviews, pre-evaluation source archive, control records,
`source-certificate.json` and bounded execution receipt. The certificate
was summarized once after the constructed controls, without evaluating a
field history. The pre-evaluation document version remains frozen separately
from this result consolidation.
No retained response cell, residual-history numerical budget, new
integration, source search or fitted mode enters the derivation.
The statement is a necessary nonlinear-transport condition and a
quantified limitation of the selected harmonic reference. It neither
predicts acquisition nor replaces the previously proved finite
exclusion, and it does not select a fundamental constitutive law.


<a id="replica-internal-comparison"></a>

## 17. Predicting collective gaps while omitting internal replica motion

Section 16 separates the complete nonlinear dynamics from an aligned
harmonic reference. Its residual combines two effects: the periodic
collective sine law and modulation by the internal replica phases.
The present comparison retains the first effect exactly and bounds
the consequence of omitting the second.

Keep Section 14's complete law, source, independent fine errors,
continuous lifts, normalized units and original structural time.
The horizon remains `H=6` and the angular margin remains
`mu=1/100`. The prospective declarations are
`artifacts/replica-internal-comparison-20261026/scope.json`
and the pre-evaluation coefficient refinement
`scope-refined.json` in that directory. No source scan,
new preparation, support change or trajectory is part of this
analytical comparison.

### 17.1. A nonlinear comparison with the same initial collective state

The [synchronized restriction](SINE_PAIR_STATE.md#sine-replica-inheritance)
`u=delta=0` is already an invariant submanifold of the
complete doubled-C5 law. It supplies the comparison equations
\[
 \dot{\widehat X}=-\frac1{2\pi}B^{\mathsf T}\sin(B\widehat\Theta),
 \qquad
 \dot{\widehat\Theta}=\frac1{2\pi}L\widehat X,
 \qquad L=B^{\mathsf T}B.
\]
For each actual source member initialize
\[
 \widehat X(0)=X(0),\qquad \widehat\Theta(0)=\Theta(0).
\]
These are memberwise equalities, including the source's independent
errors and common modes. They do not replace the actual initial means
by their nominal values. The comparison retains the full periodic sine
and the declared continuous lifts; it does not discard branch crossings.

The actual collective rows remain
\[
 \dot X=-\frac1{2\pi}B^{\mathsf T}(A\sin g),\qquad
 \dot\Theta=\frac1{2\pi}LX,\qquad
 g=B\Theta,\quad A_i=\cos\delta_i\cos\delta_{i+1}.
\]
At the **same** collective state their difference is
\[
 r(t)=\frac1{2\pi}B^{\mathsf T}
       \bigl[(1-A_i(t))\sin g_i(t)\bigr]
\]
in the form row, and zero in the phase row. The term `r`
still depends on the actual internal history. No measured response or
independently supplied control replaces that history.

This comparison predicts collective geometry only. Its missing internal
motion is imposed by the comparison's definition, so its failure to meet
an internal-activity requirement would not demonstrate that a pulse
dynamically enables organization. Actual pair-chart and activity admission
remain those of Section 14. A geometric prediction must pass its error test
before it can be combined with those separate actual-source results.

### 17.2. Retaining the source's heterogeneous internal bounds

Write `epsilon_c=2^-20+2^-128` as in Section 16 and
retain the original rational profiles `u_i^0,delta_i^0`.
Set
\[
 \alpha=\frac1\pi,\qquad
 d=\frac1{50}+\varepsilon_c,\qquad
 v=2s+\varepsilon_c,\qquad s=\frac{17}{287},
\]
and define the nonnegative scalar envelopes
\[
 D'=\alpha U,\qquad U'=\alpha D,\qquad D(0)=d,\quad U(0)=v.
\]
Thus
\[
 D(t)=d\cosh(\alpha t)+v\sinh(\alpha t),\qquad
 U(t)=v\cosh(\alpha t)+d\sinh(\alpha t).
\]
These are comparison functions, not a new internal constitutive law.

The exact full rows imply globally in the supplied lifts
\[
 D^+|u_i|\le\alpha|\delta_i|,\qquad
 D^+|\delta_i|\le\alpha|u_i|.
\]
Here `D^+` denotes an upper right derivative, so zero
coordinates cause no division by a norm. The inequalities use
`|sin delta|<=|delta|` and the absolute bound on the
two neighboring cosine factors; they do not assume restoring signs.

Preserve the known differences between the five source pairs by putting
\[
 k_i=\max\left\{
       \frac{s u_i^0+\varepsilon_c}{v},
       \frac{\delta_i^0+\varepsilon_c}{d}\right\},
 \qquad 0<k_i\le1.
\]
Both initial absolute-coordinate bounds are then dominated by
`k_i(v,d)`. Positivity of the two-dimensional comparison
flow gives, for every admitted source member,
\[
 |u_i(t)|\le k_iU(t),\qquad
 |\delta_i(t)|\le k_iD(t).
\]
The source ratios are fixed before evaluation. They are not fitted to
a future internal trajectory.

For arbitrary real `a,b`,
\[
 0\le1-\cos a\cos b
 =\tfrac12[1-\cos(a+b)+1-\cos(a-b)]
 \le\tfrac12(a^2+b^2).
\]
Consequently define
\[
 w_i=\frac{k_i^2+k_{i+1}^2}{2},\qquad
 W_2=\sum_{i=0}^4w_i^2,\qquad
 C_{\rm src}=\frac{\sqrt{\lambda_+ W_2}}{2\pi},
 \qquad\lambda_+=\frac{5+\sqrt5}{2}.
\]
Since `||B^T||_2=sqrt(lambda_plus)`,
\[
 1-A_i(t)\le w_iD(t)^2,\qquad
 \boxed{\|r(t)\|_2\le C_{\rm src}D(t)^2.}
\]
This bound retains the source's heterogeneous internal information
without adding state dimensions. The coarser common envelope would
replace `W_2` by five; `W_2<=5` makes the
relation explicit. The bound is quadratic in internal phase separation
and is valid even when the collective sine is not locally linear.

### 17.3. Propagating error between the two evolving states

Let
\[
 e_X=X-\widehat X,\qquad e_\Theta=\Theta-\widehat\Theta.
\]
For each edge use the exact sine secant
\[
 c_i(t)=\int_0^1
       \cos\bigl((B\widehat\Theta)_i+\xi(Be_\Theta)_i\bigr)\,d\xi,
 \qquad -1\le c_i(t)\le1.
\]
Then the complete difference equations are
\[
 \dot e_X=-\frac1{2\pi}B^{\mathsf T}
                 \operatorname{diag}(c_i)B e_\Theta+r,
 \qquad
 \dot e_\Theta=\frac1{2\pi}Le_X.
\]
There is no direct phase defect, but form error subsequently drives
phase error. Thus a same-state field allowance alone would not bound
prediction error.

For the symmetric secant matrix,
\[
 -L\preceq B^{\mathsf T}\operatorname{diag}(c_i)B\preceq L,
 \qquad
 \|B^{\mathsf T}\operatorname{diag}(c_i)B\|_2\le\lambda_+.
\]
No positivity of `c_i` or commutation with `L`
is required. Set `kappa=lambda_plus/(2pi)`. The actual
error norms are bounded by the nonnegative comparison
\[
 P'=\kappa Q+C_{\rm src}D^2,\qquad
 Q'=\kappa P,\qquad P(0)=Q(0)=0,
\]
so that
\[
 \|e_X(t)\|_2\le P(t),\qquad
 \|e_\Theta(t)\|_2\le Q(t).
\]
These estimates follow from the corresponding upper-right-derivative
inequalities and positive comparison, including times when an error
norm vanishes. They assert neither contraction nor instability of the
actual nonlinear flow.

The first-order forcing has to pass through the phase row. Equivalently,
\[
 Q(t)=C_{\rm src}\int_0^t
             \sinh\bigl(\kappa(t-\tau)\bigr)D(\tau)^2\,d\tau.
\]
In particular, there is no artificial nonzero initial mismatch or direct
phase forcing. The common form and lifted-phase differences remain
exactly zero, because both complete mean rows are conservative.

### 17.4. A five-dimensional positive bound, not five physical modes

The comparison and its quadratic source envelope close linearly:
\[
 \frac d{dt}
 \begin{pmatrix}P\\Q\\D^2\\DU\\U^2\end{pmatrix}
 =
 \underbrace{\begin{pmatrix}
 0&\kappa&C_{\rm src}&0&0\\
 \kappa&0&0&0&0\\
 0&0&0&2\alpha&0\\
 0&0&\alpha&0&\alpha\\
 0&0&0&2\alpha&0
 \end{pmatrix}}_{\mathcal M}
 \begin{pmatrix}P\\Q\\D^2\\DU\\U^2\end{pmatrix},
 \qquad
 \begin{pmatrix}P\\Q\\D^2\\DU\\U^2\end{pmatrix}_{t=0}
 =
 \begin{pmatrix}0\\0\\d^2\\dv\\v^2\end{pmatrix}.
\]
Every entry of `M` and the initial vector is nonnegative.
Componentwise outward upper bounds on its coefficients therefore give
a valid larger positive flow. The shared nonnegative-exponential owner
can enclose that flow under its own series and remainder admission.
It is applied to an analytical majorant, not to the actual nonlinear
dynamics, which is not declared Metzler.

For each carried edge the difference functional has norm `sqrt2`.
Hence an outward bound on the second component supplies the simultaneous
gap-error allowance
\[
 \boxed{|(B\Theta)_i-(B\widehat\Theta)_i|
        \le E(t):=\sqrt2\,Q(t)\quad(i=0,\ldots,4).}
\]
The comparison functions are nondecreasing at nonnegative times, so a
bound at the fixed horizon also bounds every earlier error. The source
refinement changes only `C_src`; it does not change the
number of comparison coordinates or the true field.

An algebraic scaling property is also explicit: multiplying both `d`
and `v` by a nonnegative factor while holding the profile ratios and comparison
coefficients fixed multiplies `P,Q,E` by its square. This follows
from the quadratic initial envelope and linear comparison flow. It is
not a scaling claim about the actual nonlinear response or a new source
evaluation.

The prospective algebraic evaluation uses only the fixed times two and
six, with zero initialization as a control. It neither integrates the
actual state nor evaluates a synchronized comparison trajectory.
An evaluated value of `E` remains an upper allowance;
it is not a measurement or lower bound on the actual omitted influence.

### 17.5. What a gap allowance can certify for the joint target

Let a justified allowance at a time be `E>=0`. Suppose
there are integers `n_i` with `sum n_i=1` such that
the comparison's raw gaps satisfy the **eroded** target conditions
\[
 \mu+E\le (B\widehat\Theta)_i+2\pi n_i
              \le\pi/2-\mu-E
 \quad\text{for every }i.
\]
The error bound then places the actual shifted gaps in
`[mu,pi/2-mu]`. These intervals lie in the principal
acute branch, and their sum is exactly `2pi` because
raw cycle gaps telescope. Together with the actual source's admitted
positive resultants, this transfers the target geometry. The separate
actual-source activity and residence obligations remain unchanged.

Conversely, any actual target state requires some integers of sum one
for which the comparison gaps satisfy the **dilated** conditions
\[
 \mu-E\le (B\widehat\Theta)_i+2\pi n_i
              \le\pi/2-\mu+E .
\]
Absence of such an integer assignment is sufficient exclusion. These
shifted values need not be the comparison's principal gaps when
`E>mu`. One cannot replace this union over lift assignments
by a test of its principal winding alone. Equality in either closed
interval remains admitted; uncertain numerical contact is unresolved.

The joint winding constraint gives a stricter positive-certification
test than checking whether a single eroded interval is nonempty.
Five values in the eroded interval can sum to `2pi`
if and only if
\[
 5(\mu+E)\le2\pi\le5(\pi/2-\mu-E).
\]
For `E>=0` the upper condition is the restrictive one:
\[
 \boxed{\text{the eroded winding-one geometry is nonempty}
       \quad\Longleftrightarrow\quad E\le\pi/10-\mu.}
\]
At equality the uniform five-gap assignment `2pi/5` is
still permitted. Above this threshold, the **selected uniform allowance**
cannot certify target geometry by erosion for any comparison state.
That is a loss of resolution of this sufficient test, not a proof of
large actual error or absence of formation. A smaller time-dependent
bound or retained signed correlations could support a different test.

Comparison with the original margin `mu` is also a declared
resolution scale, not a substitute for these joint transfer conditions.
Even a small error allowance does not prove that the comparison reaches
the eroded target, and a loose large allowance does not prove that the
internal motion has a large dynamical effect. No comparison state or
target visit is evaluated in this section.

### 17.6. Storage and source boundaries

This is a comparison evolution on an existing invariant submanifold,
not a reset or an event imposed on the actual state. At identical
collective coordinates the complete fine stores obey
\[
 \begin{aligned}
 E_{\rm actual}
   &=2X^{\mathsf T}LX+4\sum_i u_i^2
        +4\sum_i(1-A_i\cos g_i),\\
 E_{\rm sync}
   &=2X^{\mathsf T}LX+4\sum_i(1-\cos g_i),\\
 E_{\rm actual}-E_{\rm sync}
   &=4\sum_i u_i^2+4\sum_i(1-A_i)\cos g_i.
 \end{aligned}
\]
The last term is not generally positive outside an acute domain, so
these are not universally separate positive reservoirs. At the
declared initial source its tiny raw phase gaps have positive cosines
and its internal forms are nonzero, making the initial difference
positive. The comparison is therefore explicitly not another
equal-storage acquisition preparation. Each complete evolution retains
its own conserved storage; no new work, damping or constitutive
coefficient is introduced.

The [known nonclosure of pair means](SINE_PAIR_STATE.md#sine-replica-inheritance)
already explains why omitted internal coordinates can affect response.
The new obligation here is a source-associated allowance for the
difference of two evolving predictions of collective geometry.
The existence of the synchronized restriction itself is not new, and
its absent internal activity is not used as a discriminator.

<a id="replica-internal-comparison-result"></a>

### 17.7. Validated error allowances and their task resolution

The private
[comparison reader](../../src/tnfr/research/replica_synchronized_comparison.py)
and
[constructed controls](../../tests/physics/test_replica_synchronized_comparison.py)
reuse the original complete field, source admission, C5 spectrum and
shared positive-exponential owner. The controls cover same-state and
two-state rows, the invariant restriction, internal source envelopes,
the quadratic comparison lift, original clock, joint geometric
transfer and primitive-domain rejection.

Eight independently reviewed constructed controls passed, with no skips.
They include the full twenty-coordinate field at identical and different
collective states, mixed-sign sine secants, the quadratic source lift,
an independent high-precision matrix exponential, exact initialization,
joint erosion with uncertain contact, and primitive/source rejection.

The algebraic source certificate then evaluated the positive majorant
at the two fixed times. Its exact outward fractions imply these
conservatively rounded upper bounds:

| Structural-time prefix | Collective form error `||e_X||_2` | Collective phase error `||e_Theta||_2` | Each of the five gap errors |
| --- | --- | --- | --- |
| `[0,2]` | `0.003926233` | `0.001303026` | `0.001842757` |
| `[0,6]` | `0.204124039` | `0.132834478` | `0.187856319` |

The angular bounds are in radians; form uses the declared normalized
coordinates. Prefix coverage follows from the positive majorant's
monotonicity, not from sampled response times. These values bound the
difference between the two exact member-associated collective evolutions;
neither trajectory nor the actual error was evaluated.

Both gap allowances are strictly below `pi/10-mu`, so each selected
eroded winding-one target remains nonempty. The prefix-two allowance
is also below `mu=1/100`. The horizon-six allowance exceeds that
resolution scale, but this is a property of an upper bound, not a
measured large internal effect or a lower bound on prediction error.
The refined source coefficient is at most `0.421731729`.

The result provides a controlled approximation to collective geometry
while omitting internal evolution. It retains internal initialization
through its source weights and error allowance, and does not establish
exact autonomous closure or a universally sufficient smaller state.
No arrival or exclusion follows from a nonempty eroded target alone:
a comparison trajectory and the integer-assignment transfer test still
have to be evaluated under a separate frozen protocol.

Local evidence in `artifacts/replica-internal-comparison-20261026/`
retains the original and refined pre-evaluation scope, independent reviews,
frozen sources, eight-control records and `source-certificate.json`.
The declaration survived an interrupted session before any numerical
scientific evaluation; a read-only resumption audit checked that boundary
and preserved the historical records. The explicit algebraic summary ran
once after the controls. Its frozen document version remains separate
from this result consolidation.

No previously evaluated response cell, actual or comparison trajectory,
residual-history budget, source search or parameter fit was consumed.
This is a source-based prediction allowance, not an observed acquisition
or a physical identification of internal motion.

<a id="replica-reduced-prediction"></a>

## 18. A reduced-state test of the actual source's joint geometry

Section 17 proves a memberwise comparison allowance; it does not evaluate
its synchronized comparison trajectory. The present gate asks whether that
smaller evolving state, together with the allowance, can decide the actual
source's target geometry on `[0,6]`. The existing full-state exclusion is
prior consistency evidence, not an input to the new target decision.
The prospective protocol is
`artifacts/replica-reduced-prediction-20261027/scope.json`.

### 18.1. Exact source image and the inherited ten-coordinate field

For the intended physical fine source let `epsilon=2^-20`. Pair averaging
maps its independent error box **exactly** onto the ten-coordinate box
\[
 X_i(0)\in[cv_i-\varepsilon,cv_i+\varepsilon],\qquad
 \Theta_i(0)\in[-\varepsilon,\varepsilon],
 \qquad c=570/287,\quad v=(-1,-1,0,1,1).
\]
Every mean lies in this box. Conversely, choosing both errors in each pair
equal realizes every point of the box within the fine source. The comparison
is initialized with the means of each actual member, not just the nominal
means. Internal coordinates and their possible correlations with those means
remain covered by Section 17's actual-source theorem.

Construct each numerical initial interval directly from these exact rational
endpoints, with one outward interval rounding. Its additional coordinate
cover is at most `2^-128`; it encloses the physical image without redefining
all points of the numerical cover as new physical sources. Do not project
already rounded fine intervals and then silently discard further rounding.

Let `J` duplicate each of the five forms and five phase lifts into identical
fine-pair coordinates, and let `P` take their pair means. On the existing
invariant restriction `u=delta=0`, the full field `F` obeys
\[
 PJ=I,\qquad F(Jy)=Jf(y),\qquad f=PFJ,
 \qquad y=(X,\Theta).
\]
This follows from the exact pair equations in Section 17.1: both fine rows
within a synchronized pair are equal, and both internal rows vanish. The
comparison therefore evolves only ten coordinates while using the unchanged
complete-field owner through this lift and projection. Interval and formal
polynomial evaluation preserve the symbolic duplication; they do not evolve
independent hidden copies. An interval extension may widen an enclosure,
which remains numerical uncertainty rather than an extra physical state.
Support, held capacity, full sine and the original structural clock are
unchanged. The comparison's storage and internal-activity boundaries remain
those of Section 17.6.

### 18.2. Two retained allowances and one whole-time enclosure budget

The prior certificate fixes exact outward per-gap bounds `E_2,E_6` whose
decimal upper displays are `0.001842757` and `0.187856319`. The new reader
binds that certificate to its closed source, evaluation and archive records,
re-admits its consumed primitive source data, and reconstructs the positive
comparison matrices, initial vector and gap bounds. Independently checking
the retained exponential records with the shared positive-exponential kernel
is prior-certificate arithmetic validation. It neither runs the old
certificate producer nor consumes the old forty-eight actual response cells.

Use `h=1/8`, `t_k=k/8` and the forty-eight closed cells
`[t_k,t_(k+1)]`. Their declared allowances are
\[
 E_k=\begin{cases}
 E_2,&t_{k+1}\le2,\\
 E_6,&t_{k+1}>2.
 \end{cases}
\]
In particular, the cell starting at two uses `E_6`. Both bounds are valid
at the common endpoint; the schedule does not create a temporal gap.
For any corresponding exact actual and comparison members,
\[
 |g_i(t)-\widehat g_i(t)|\le E_k
 \quad\text{for all }t\in[t_k,t_{k+1}].
\]

A validated ten-coordinate whole-time tube `Y_k` encloses every comparison
member for that cell. Its derived gap interval `G_(k,i)` already contains
initial uncertainty, numerical integration uncertainty and the interval
projection allowance. Add `E_k` once as a **gap** allowance. It is not a
per-phase radius to double, and no extra initial `epsilon` is added to the
target interval. The actual source's positive-resultant chart through six
is a separately re-admitted premise, not inferred from the comparison tube.

### 18.3. Closed dilation and exhaustive integer assignments

Let `Pi=[pi_-,pi_+]` be the shared outward enclosure of pi. Define the
closed outer dilated interval
\[
 [a_k,b_k]=[\mu-E_k,\;\pi_+/2-\mu+E_k],\qquad\mu=1/100.
\]
If the actual target holds at a time in the cell, there exist integers
`n_i` with `sum n_i=1` such that
\[
 \widehat g_i(t)+2\pi n_i\in[a_k,b_k]\quad\text{for every }i.
\]
This is the necessary dilation implication from Section 17.5. In particular,
when the lower dilated limit is negative, a representative acute-gap or
principal-winding test does not replace the union over integer assignments.
The statement does not require that every retained allowance crosses a
principal phase branch.

The observer first admits all ten interval coordinates and requires
`|G_(k,i)|<13` for every gap. The finite shift range `-2..2` is exhaustive
when the exact outward checks
\[
 A:=6\pi_- -13>b_k,\qquad A>-a_k
\]
are certified. Indeed, `n>=3` gives a shifted gap above `b_k`, and
`n<=-3` gives one below `a_k`; larger absolute shifts only strengthen those
inequalities because `pi_->0`. A missing gap-domain or branch-completeness
premise produces an unavailable, unresolved observation, never separation.

For each edge retain every shift whose outward interval meets the closed
dilated interval:
\[
 N_{k,i}=\{n\in\{-2,-1,0,1,2\}:
       (G_{k,i}+2n\,\mathrm{Pi})\cap[a_k,b_k]\ne\varnothing\}.
\]
Endpoint contact remains possible. Form the reachable sums
\[
 S_0=\{0\},\qquad
 S_{i+1}=\{s+n:s\in S_i,\ n\in N_{k,i}\}.
\]
Then
\[
 \boxed{1\notin S_5\quad\Longrightarrow\quad
 \text{the actual target is absent throughout this closed cell}.}
\]
The source association and error theorem are essential to that implication.
An arbitrary supplied tube does not certify that it came from this source.

Relaxing common-pi, same-state and same-time correlations enlarges the
possible assignment set, so absence remains a sufficient certificate.
Conversely, a reachable sum of one establishes neither simultaneous
compatibility nor target existence. It remains unresolved. This reader is
exclusion-only; it adds no positive erosion detector or acquisition claim.

### 18.4. Temporal decision and the fixed numerical attempt

Use the existing closed-cell `time_cover` on the forty-eight labels
`separated` or `unresolved`. Every missing suffix cell is unresolved.
All forty-eight certified separations cover every point of `[0,6]`,
including shared endpoints. Otherwise the reduced predictor has not decided
full-horizon absence. A half-unit residence exclusion, if reported, requires
the existing time-cover argument with its endpoint-inclusion rules; a short
run of failed positive tests is not such a certificate.

One comparison attempt uses the shared `validated_box_taylor_step`, order
eight, fixed step `1/8`, dimension ten and outward 128-bit rational interval
arithmetic. The complete sine field is globally smooth in the supplied real
lifts, so target conditions are not integration boundaries. Strict Picard
inclusion and whole-tube derivative bounds remain obligations of the shared
validated kernel. There is a hard 900-second worker limit and between-cell
budget checks. No retry, adaptive step, alternate source or horizon is part
of the protocol. Retain every completed step and any partial or failure record.

After the attempt, a detached audit re-admits the source, numerical protocol,
cell sequence, allowances and observations, and reconstructs increment and
endpoint arithmetic with the shared Taylor audit owner. It does not recompute
derivatives or Picard bounds, or integrate either trajectory again. Numerical
conclusions remain conditional on the validated kernel, frozen field and
admitted source; content receipts associate bytes without authenticating
execution. The original actual-source trajectory is not rerun.

### 18.5. Prospective controls and interpretation

The [prediction owner](../../src/tnfr/research/replica_synchronized_prediction.py)
and its [constructed controls](../../tests/physics/test_replica_synchronized_prediction.py)
cover the lifted field and polynomial jets, source image and rounding,
allowance schedule, closed dilation, joint branches, domain abstention,
complete and partial temporal coverage, and primitive/evidence corruption.
The eight controls and one comparison response are pending.

A full-horizon exclusion would establish task-level predictive sufficiency
of this reduced evolution with its retained initialization bounds. It would
not show that internal motion is absent or irrelevant for other responses,
that the collective state closes exactly, or that the source acquires an
organization. An unresolved result instead names a failure of this chosen
predictor's resolution; it does not prove that internal history is necessary.
The known full-state absence is a consistency check, not a blind validation
or a new formation finding. No physical identification follows.

<a id="replica-reduced-prediction-result"></a>

### 18.6. Retained result: partial predictive resolution

The single declared comparison completed all forty-eight cells. All eight
constructed controls passed without skips, and the detached prefix audit
reconstructed the source association, allowance schedule, stored Taylor
increment/endpoint arithmetic and target classifications. No actual-source
trajectory or old scientific producer was rerun. The frozen sources,
control results and response are retained in
`artifacts/replica-reduced-prediction-20261027/`.

The result is **`reduced_prediction_task_resolution_incomplete`**:

| Quantity | Retained result |
| --- | --- |
| Evolving comparison coordinates | 10 |
| Completed whole-time cells | 48 of 48 |
| Cells with certified target separation | 43 |
| Unresolved cells | 5, zero-based indices 36 through 40 |
| Unavailable observations or numerical failure | None |
| Remaining possible-time component | `(9/2,41/8)` |
| Half-unit residence excluded by this predictor | No |

Every observation satisfied the raw-gap domain and finite-branch
completeness checks. The five unresolved cells retain an integer assignment
with sum one after closed dilation by the horizon allowance. They are not
missing data, failed integration steps or positive target observations.
The forty-three separated closed cells certify absence on
`[0,9/2]` and `[41/8,6]`; the neighboring separated cells also remove both
endpoints of the remaining open component. Its length is `5/8`, so this
reduced prediction cannot exclude a half-unit residence there. The retained
`residence_possible` flag records that unresolved possibility, not existence
of a residence window or a compatible exact target state.

This is a source- and task-specific partial reduction result. Ten evolving
collective coordinates plus the initialization-dependent error theorem
resolve most of the tested temporal cover, but do not recover the earlier
full-state exclusion throughout `[0,6]`. The count `43/48` measures cells in
this fixed partition; it is not an accuracy estimate, confidence level or
fraction of source members correctly predicted.

The earlier full-state exclusion remains valid under its own retained
numerical premises. The new ambiguity therefore does not establish actual
target acquisition. Nor does it establish a large actual discrepancy or the
necessity of evolving all internal coordinates: the uniform omitted-error
allowance, the comparison tubes and the interval relaxation of joint
geometry can each reduce decision power. The actual comparison error was
not measured, and no cause is selected by this result alone.

The prospective protocol and its pending-language above remain the frozen
pre-evaluation record; this subsection supplies its outcome. The next
scientific question belongs to the [research program](../RESEARCH_PROGRAM.md#research-current-question).
No trajectory refinement, changed source, enlarged horizon or physical
identification follows from this partial result.

<a id="replica-prediction-resolution"></a>

## 19. The geometric resolution required by a reduced prediction

This is a detached question about the retained comparison tubes of Section
18, under the same source association, complete law, clock, target and
allowance theorem. The inspected partial prediction is prior information.
No trajectory, internal-error envelope or physical identification is added.

### 19.1. Exact distance for the existing relaxed observer

Let `Pi=[p_lo,p_hi]` be the shared outward pi enclosure, `mu=1/100`,
`U=p_hi/2-mu`, and `cap=E_6` the retained horizon gap allowance. Reuse
`observe_prediction(tube,cap)` to obtain the five raw gap intervals `G_i`
and require its domain admission:

\[
\max_i\operatorname{absmax}(G_i)<13,\qquad
A:=6p_{\rm lo}-13>U+\mathrm{cap},\qquad A>\mathrm{cap}-\mu.
\]

These inequalities cover every `0<=E<=cap`. For `n=-2,...,2`, form
`S_(i,n)=G_i+2*n*Pi` with exactly the shared outward interval operations.
Using its retained endpoints, define the exact nonnegative rational

\[
d_{i,n}=\max\{0,\ \mu-S_{i,n}^{\rm hi},\ S_{i,n}^{\rm lo}-U\}.
\]

The closed interval `S_(i,n)` meets `[mu-E,U+E]` if and only if
`E>=d_(i,n)`: its upper endpoint must reach the lower target endpoint,
and its lower endpoint must not exceed the upper target endpoint. Thus
contact is admissible, including a zero-width intersection. This is the
same enlarged interval test as Section 18, with the same outward pi bounds;
no further source uncertainty or second copy of the gap error is added.

### 19.2. Coupled minimax, finite branches and censoring

Require the winding assignment `sum_i n_i=1`, rather than minimizing each
edge independently. The restricted critical dilation is

\[
\rho=\min_{\substack{n_i\in\{-2,-1,0,1,2\}\\\sum_{i=0}^4n_i=1}}
                 \max_{i=0}^4 d_{i,n_i}.
\]

The finite assignment set is nonempty. Its minimum can be computed by
exact rational dynamic programming. Start `D_0(0)=0`, with all other sums
unreachable, and use

\[
D_{j+1}(s)=\min_{n\in\{-2,-1,0,1,2\}}
                  \max\{D_j(s-n),d_{j,n}\},\qquad \rho=D_5(1),
\]

ignoring unreachable predecessors. Induction shows that `D_j(s)` is the
smallest possible largest edge distance among the first `j` choices of
sum `s`. Therefore the restricted observer is unresolved exactly when
`E>=rho`, and separated exactly when `E<rho`. Equality never excludes.
A minimizing assignment witnesses this relaxed test, not a physical state.

For `n>=3`, the shifted interval has lower endpoint at least `A`; for
`n<=-3`, its upper endpoint is at most `-A`. The two domain inequalities
therefore make every omitted branch's distance strictly greater than
`cap`. Consequently the restricted branches are complete throughout the
whole admitted allowance domain, including both endpoints.

If `rho<=cap`, it is the exact critical dilation of the all-integer
relaxed observer and lies in the proved domain. If `rho>cap`, all retained
and omitted assignments require more than `cap`. Report only that the
all-integer critical dilation is **strictly greater than `cap`**. The
restricted value `rho` is not a global lower bound: admitting further
branches can lower a minimum. Do not extrapolate an exact threshold or a
minimizing-assignment interpretation beyond the proved domain. Missing
source, arithmetic or branch admission remains unavailable.

### 19.3. Interpretation and the one retained-data diagnostic

A zero threshold means that shrinking the omitted-error allowance alone
cannot separate this tube with the same relaxed reader. It does not prove
an exact comparison-state visit, an actual target visit, a nonzero actual
error, or necessity of internal history. A positive threshold at or below
the retained allowance names a sufficient resolution requirement: a valid
smaller allowance strictly below it would separate this same tube. No such
new bound is supplied here. An allowance already below the threshold must
reproduce the previous separation. These alternatives do not distinguish
numerical tube width, discarded joint correlations and actual discrepancies.

The [resolution owner](../../src/tnfr/research/replica_prediction_resolution.py)
uses `resolve_prediction` and `resolution_status` for this proof. One
prospectively frozen diagnostic reads all forty-eight retained comparison
tubes, reconstructs consumed source/protocol/allowance and stored Taylor
increment/endpoint arithmetic through the shared owners, and reconciles the
frozen classifications. It reads no original actual-source response cells
and calls no field, derivative/Picard reconstruction, integrator, prior
scientific producer or new envelope evaluation. Each command has a
300-second limit; there is no adaptive or selected-cell retry.

Eight constructed controls cover exhaustive minimax comparison, coupled
integer sums, closed contact, zero-threshold interpretation, censoring and
unavailable domains, equivalence with the existing observer, evidence
corruption and primitive admission. Controls and the sole diagnostic are
pending. Report exact admitted thresholds or censored strict lower bounds,
zero-allowance and retained-allowance time covers through the existing
closed-cell owner, and the five previously unresolved cases. The zero-error
cover is a counterfactual reader limit, not a new actual-source certificate.
Preserve the original result and every unresolved case. Evidence and
post-evaluation arithmetic review belong to
`artifacts/replica-prediction-resolution-20261027/`; this is not a blind
prediction or empirical validation. Only the research program selects what
information, if any, should be improved after this diagnostic closes.

<a id="replica-prediction-resolution-result"></a>

### 19.4. Retained result: five positive error budgets

All eight constructed controls passed without skips. The one frozen
diagnostic admitted all forty-eight retained cells and reproduced their
original forty-three separated and five unresolved classifications. The
independent stored-arithmetic audit linked its inputs directly to the
anchored source, allowance and cell files and checked the minimax by
exhaustive integer assignments. No integration or new error envelope was
evaluated. The outcome is **`positive_thresholds_masked_by_retained_allowance`**.

The five previously unresolved cells have strictly positive exact observer
thresholds within the proved domain. The following decimal values are
rounded **downward**; exact rational values are retained in
`artifacts/replica-prediction-resolution-20261027/resolution-result.json`
and its `result-audit.json`.

| Zero-based cell | Closed structural-time interval | Downward threshold bound |
| --- | --- | --- |
| 36 | `[9/2,37/8]` | `0.149522927` |
| 37 | `[37/8,19/4]` | `0.053970151` |
| 38 | `[19/4,39/8]` | `0.095873750` |
| 39 | `[39/8,5]` | `0.137330028` |
| 40 | `[5,41/8]` | `0.178356240` |

For each row a valid uniform gap allowance strictly below its listed bound
would suffice to separate that same tube. Equality at the exact threshold
remains unresolved. All five exact thresholds are below the retained
`E_6`, whose upper decimal display is `0.187856319`; no such smaller valid
error bound was derived or evaluated in this gate.

The other forty-three cells have critical dilation strictly greater than
`E_6`. Their values remain censored at that cap, rather than being reported
as exact all-integer thresholds. There are no zero thresholds or unavailable
observations. Setting only the omitted-error allowance to zero makes the
reader separate all forty-eight cells, retaining their numerical and initial
uncertainty. Its zero-allowance possible-time cover is empty. This is a
counterfactual resolution limit, not an admissible zero-error theorem for
the actual source or a new actual-source absence certificate.

Thus the present five ambiguities are introduced by the declared dilation
in this fixed relaxed reader; they are not unavoidable at every positive
allowance. The diagnostic quantifies a usable target for an error theorem.
It does not show that the actual error meets that target, identify which
internal effect controls it, or establish exact autonomous closure. The
original possible-time component `(9/2,41/8)` remains unchanged until a
separately justified allowance or observation closes it. No physical
identification or new formation result follows.

The prospective pending-language in Section 19.3 is preserved above; this
subsection records its completed outcome. The [research program](../RESEARCH_PROGRAM.md#research-current-question)
selects the next finite discriminator against the global objective.
