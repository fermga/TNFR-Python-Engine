# Global pulse, emergence and resonance: a hypothesis audit

This chapter audits primordial structural activity with local rhythms, and the
hypothesis that a collective pulse and clock arise through their coordination.
It distinguishes derived internal motion, emergent synchronization, resonant
susceptibility, supplied excitation and a reciprocal source with its own state.
None is promoted to a universal law or a physical identification by naming it.

The [research program](../RESEARCH_PROGRAM.md#research-current-question) owns
the sole active question. The [resonance owner](RESONANCE_FOUNDATIONS.md)
and [existing internal pulse](SINE_REPLICA_PULSE.md#sine-replica-internal-pulse)
retain their original assumptions and proofs.


<a id="global-pulse-hypothesis"></a>

## 1. What would have to emerge?

**The intended hypothesis starts with basic activity and local rhythms.
A global pulse, and a collective reading of time, would arise through
coordination or synchronization of those local processes. It does not
require an externally imposed universal beat or initially equal phases.**

This separates two scientific questions: whether local organized activity
arises and persists, and whether interactions coordinate its local timing.
The existing pulse theorem supports a conditional realization of the first;
it does not by itself solve either generic acquisition or the second.
The supplied-clock and source alternatives below are discriminators and
counterexamples, not replacements for this local-first hypothesis.

A primordial pulse could mean a common periodic state, a shared phase reference,
a persistent capacity for internal motion, or a cause that excites otherwise
quiet organizations. These are different hypotheses. Existence of one periodic
solution does not select that solution, make its period universal, or make
all constituents share its phase. An explicit oscillator postulate is a
legitimate constitutive candidate, but must retain what was assumed and what
then follows. The current sine law is itself conditional, not a uniquely
selected fundamental law that excludes every alternative.

The [prepared replica pulse](SINE_REPLICA_PULSE.md#sine-replica-internal-pulse)
already derives persistent internal circulation from a complete autonomous
law. On its supplied invariant preparation, with `alpha=2*pi/5`, it obeys

\[
\dot u=-a\nu\cos\alpha\cos\delta\sin\delta,\qquad
\dot\delta=b\nu u,\qquad
H=u^2+\beta\cos\alpha\sin^2\delta.
\]

Here `a=w/pi`, `b=w/(beta*pi)`, and `H` is conserved. For
`0<m=H/(beta*cos(alpha))<1`, the labeled period is

\[
T=\frac{4K(m)}{\Omega_{\rm int}},\qquad
\Omega_{\rm int}=\frac{w\nu\sqrt{\cos\alpha}}{\pi\sqrt\beta}.
\]

`K(m)` is the complete elliptic integral defined in that owner. The period
depends on supplied geometry, coefficients and amplitude. The unordered pair
state returns after `T/2`; the collective means remain fixed. This is a
positive existence result for an internal pulse, not a proof of a unique
global rhythm or its formation from arbitrary preparations.

The [complete pulse variation](SINE_REPLICA_PULSE.md#sine-replica-pulse-variation)
and [small-amplitude splitting](SINE_REPLICA_PULSE.md#sine-replica-pulse-splitting)
already show that a derived autonomous pulse can produce periodic transverse
coefficients and parametric feedback. Sufficiently small nonzero members of
that particular prepared waveform are transversely unstable. Meanwhile an
open family can preserve the larger collective identity and active internal
constituents, as proved in the
[joint persistence result](SINE_REPLICA_PULSE.md#sine-replica-joint-persistence).
Neither synchronized periodicity nor convergence to that waveform follows
for the surrounding family.

The existing obstructions have precise domains:

- An exact equilibrium of an autonomous smooth law stays there by uniqueness.
  Thus the present state space contains counterexamples to the assertion
  that every preparation must start pulsing.
- On finite connected support with held nonnegative capacities and `e>0`,
  the closed unforced sine model approaches its equilibrium set. Its
  [storage argument](RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
  excludes nonstationary full-state recurrence; it does not select one
  final geometry or a universal relaxation rate.
- At `e=0` with strictly positive held capacities, the admitted finite
  invariant slabs preserve volume. For a closed invariant set, the basin
  converging from outside is confined to the recurrence-null exceptional
  set in that ambient volume. In particular a periodic orbit cannot acquire
  an attracting basin of positive ambient volume there. The
  [capture boundary](RESONANCE_FOUNDATIONS.md#conservative-formation-boundary)
  also excludes entry from outside into a two-sided invariant family.

These statements do not exclude finite observational formation, persistent
activity without a common period, different complete laws, retained open
subsystems, or infinite-system hypotheses with different compactness premises.
They do rule out using the current finite closed examples as a proof of
universal pulse selection. Failure of that proof is not a general theorem
that a primordial pulse cannot emerge from any TNFR realization.

## 2. Specify what is observed to pulse

The repository's [coherence read-out](../GLOSSARY.md#coherence) is

\[
C=\frac1{1+\operatorname{mean}|p_i|+
                 \operatorname{mean}|\dot x_i|},
\]

in declared input scales and with admitted pressure/rate data. It is a
selected diagnostic. It is not an independent state coordinate or a law for
how coherence must evolve. Promoting an independently prescribed `C(t)` to
a cause would require compatibility with the full state and its evolution,
including the nondifferentiable absolute-value boundaries.

Two exact controls prevent misleading identifications. At a crossing of
the prepared pulse, `delta=0,u!=0`, every form rate and its corresponding
unforced pressure vanish. The live read-out is therefore `C=1`, while
`delta_dot=b*nu*u!=0`. Maximal `C` does not imply a stationary full state.
Moreover, its ten phase phasors satisfy

\[
\sum_{i=0}^4\sum_{\sigma=\pm1}
 e^{\mathrm i(\bar\Theta+2\pi i/5+\sigma\delta)}
 =2e^{\mathrm i\bar\Theta}\cos\delta
       \sum_{i=0}^4e^{2\pi\mathrm i i/5}=0.
\]

The normalized magnitude of this global phase sum is identically zero during
the nonzero internal pulse. It is a separate phase-order observation, not
the repository's `C`. Organized winding and internal circulation need not
produce global phase alignment. A proposed global coherence pulse must name
its retained state and observable instead of relying on that phrase alone.

## 3. A common clock or origin can leave relative dynamics unchanged

Fix the existing finite connected simple unit graph, positive held
capacities, and the complete normalized-sine law

\[
K=\operatorname{diag}(\nu_i/d_i),\quad q=Lx,\quad
S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),
\]
\[
\dot x=-eKq+aKS,\qquad \dot\theta=bKq,
\qquad a=w/\pi=\beta b,\quad b=w/(\beta\pi).
\]

Its storage is `E=x^T L x/2+beta*V(theta)`, where
`V=sum_edges[1-cos(theta_j-theta_i)]`. Its exact balance is
`E_dot=-e*q^T K q`. Both rows, all coefficients and the original clock
belong to the model. Subsequent alternatives are explicitly changed models;
none changes engine defaults.

### 3.1. Common positive speed modulation

Write the full autonomous vector field as `F(z)`. For an independently
specified positive continuous `gamma(t)`, the alternative

\[
\dot z=\gamma(t)F(z),\qquad
\tau(t)=\int_0^t\gamma(s)\,ds
\]

has the exact solution `z(t)=Phi_(tau(t))(z(0))` wherever defined. It follows
the same state-space orbits, in the same orientation. It cannot release an
invariant family, turn an equilibrium into motion, or create a new relative
configuration along that orbit. Finite-time windows in the original clock
can change. Infinite-time accessibility also requires that `tau(t)` tend
to infinity; periodic positive `gamma` has that property.

Every consumed row must be multiplied. In the conservative sector, changing
only the common `w` scales both sine rows together. With positive fixed `e`,
changing only `w` leaves the loss term unscaled and is generally a constitutive
change. Inputs, output units and timing must also transform in a clock
comparison. A nonconstant clock need not preserve a time-invariant laboratory
frequency-response experiment, as the
[clock owner](RESONANCE_FOUNDATIONS.md#resonance-clock) explains.

A second-order calculation exposes a common false resonance mechanism.
For a real mode with constants `A,B>0,d>=0`, suppose

\[
\dot X=\gamma(-dX-A\Theta),\qquad \dot\Theta=\gamma BX.
\]

For differentiable positive `gamma`, differentiation of the actual second
row gives

\[
\boxed{\ddot\Theta+
 \left(\gamma d-\frac{\dot\gamma}{\gamma}\right)\dot\Theta
 +\gamma^2AB\Theta=0.}
\]

Dropping `-gamma_dot/gamma` changes the model. At an equilibrium, a prescribed
periodic clock produces tangent return `exp(A_full*integral(gamma))`.
For a conservative stable mode its multipliers remain on the unit circle;
a periodic clock alone does not create parametric growth of that mode.
This says nothing about removing an instability already present in `F`.

### 3.2. Common form and phase rates

Add prescribed scalar rates `f_x(t)*1` and `f_theta(t)*1` to the respective
full rows. Subtracting their time integrals from the corresponding common
origins removes both additions exactly: `L*1=0`, and `S` depends only on
phase differences. Relative forms, relative phases, joint pair distances
and cycle windings follow their original dynamics. The added work is zero,
since `1^T q=1^T S=0`. Absolute-origin observations would require their own
measurement convention; this result concerns the stated relative structure.

Uniform **pressure** is different from a uniform **form rate**. Adding the
same nodal pressure `P(t)` gives the rate `nu_i*P(t)`. If capacities differ,
this vector need not be a common origin. For example, on two nodes with
`nu_0!=nu_1`, initially equal forms and equal phases acquire form-contrast
rate `(nu_0-nu_1)*P`. The spatial vector actually entering the full equation,
not the adjective "global", determines what can be excited.

## 4. Intrinsic susceptibility, supplied excitation and observation

For standard terminology, Georgi's
[The Physics of Waves, Chapter 2](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/ef731c1b91d77a6db003f6c27e300d25_MIT8_03SCF16_Textbook.pdf)
distinguishes driving frequency, natural response and supplied work in a
forced oscillator. The TNFR-model equations below are derived independently
from their stated full law; that reference is not validation of this paradigm.

An additive form-row port `f*u(t)`, with fixed real spatial vector `f`, has
work-conjugate output `y=f^T q`. Direct differentiation gives

\[
\dot E=-e q^TKq+u(t)y(t).
\]

The input can transfer structural storage. Its amplitude, spatial pattern,
phase and timing are preparation data. The autonomous law's susceptibility
to it does not produce that input. Conversely a supplied pulse cannot excite
an invariant modal sector to which its admitted input has no coupling, or
make an output
observe a mode that it annihilates.

For an exact illustration, take phase consensus on regular support with
`K=kappa*I`. Let `L*v=lambda*v`, `lambda>0`, and `||v||=1`. Use a pure modal
input `f=f_m*v` and perturbations `delta x=X*v,delta theta=Theta*v`. Define

\[
A=a\kappa\lambda,\quad B=b\kappa\lambda,\quad
 d=e\kappa\lambda,\quad k=AB>0.
\]

The complete tangent rows and three outputs give

\[
\dot X=-dX-A\Theta+f_m u,\qquad \dot\Theta=BX,
\qquad y=f_m\lambda X,
\]
\[
D(s)=s^2+ds+k,\qquad
\mathcal H_X=\frac{f_m s}{D(s)},\quad
\mathcal H_\Theta=\frac{Bf_m}{D(s)},\quad
\mathcal H_y=\frac{f_m^2\lambda s}{D(s)}.
\]

For `e>0` these are stable quotient transfers. With `f_m!=0`, the form and
work gains peak at `omega^2=k` for every `d>0`. Phase gain has a positive
frequency maximum only if `d^2<2k`, at `omega^2=k-d^2/2`; otherwise its
maximum is at zero. The pole condition `d^2<4k` is different again. These
follow by differentiating the squared gains in `omega^2`, and agree with
the [existing modal theorem](RESONANCE_FOUNDATIONS.md#modal-resonance).
A pulse can excite a pre-existing susceptibility; it need not create it.

A uniform rate input has `f_m=0` for every nonconstant Laplacian mode on
this regular example. An impulse or finite waveform with modal overlap can
excite a transient; it need not be periodic or have a universal frequency.
A temporal Fourier component outside an observed mode's response band also
does not establish a resonance peak. The input/output definition is part of
every such assertion.

At `e=0`, the stable bounded-gain theorem is unavailable. For example,
the modal phase equation driven by `u=U*cos(omega_0*t)`, with
`omega_0=sqrt(k)`, has zero-initial-state particular solution

\[
\Theta(t)=\frac{Bf_mU}{2\omega_0}\,t\sin(\omega_0t).
\]

This exact tangent secular response consumes external work. It is not a
finite-amplitude theorem of unlimited growth for the nonlinear sine law.
No input, or zero modal overlap, removes this excitation mechanism.
The configured Resonance operator remains a separate event contract;
none of these continuous equations invokes it automatically.

## 5. Genuine parametric modulation has a different work balance

To change restoring stiffness rather than only clock speed, fix `b=b_0`
and prescribe `rho(t)>0`. Choose

\[
w(t)=w_0\rho(t),\qquad \beta(t)=\beta_0\rho(t),\qquad
 a(t)=a_0\rho(t),\quad b_0=w_0/(\beta_0\pi).
\]

These are directly specified time-dependent coefficients of the complete
law. They do not result from repeatedly passing weights through a constructor
that normalizes their sum. The corresponding rows and instantaneous storage are

\[
\dot x=-eKq+a_0\rho KS,\qquad \dot\theta=b_0Kq,
\qquad E_\rho=\tfrac12x^TLx+\beta_0\rho V.
\]

The reciprocal exchange still cancels, but explicit parameter work remains:

\[
\boxed{\dot E_\rho=-e q^TKq+\beta_0\dot\rho\,V.}
\]

Thus periodic modulation does not get an energy source for free. If `rho`
has prescribed jumps and the state is continuous, the additional jump work
is exactly `Delta E_rho=beta_0*Delta rho*V`. An ordinary derivative on the
smooth pieces alone omits that contribution. Such parameter switches are
supplied source changes, not occurrences of a named TNFR operator.

On the preceding consensus mode the actual equation is now

\[
\ddot\Theta+d\dot\Theta+\omega_0^2\rho(t)\Theta=0,
\qquad \omega_0^2=a_0b_0\kappa^2\lambda^2.
\]

There is no clock-derivative term here because the phase-row coefficient
is fixed. Zero modal initial data stay zero; modulation can amplify a
perturbation without generating it from an exact equilibrium.

### 5.1. An exact positive-stiffness amplification example

Set `e=0`. On one interval use `rho_1>0` and frequency
`omega_1=omega_0*sqrt(rho_1)` for duration `pi/(2*omega_1)`. Follow it
with `rho_2>0`, frequency `omega_2`, and duration `pi/(2*omega_2)`, then
repeat the two intervals. These durations are an explicitly tuned input
protocol. In state `(Theta,Theta_dot)`, each exact quarter-period matrix is

\[
U_j=\begin{pmatrix}0&1/\omega_j\\-\omega_j&0\end{pmatrix}.
\]

The return over a complete modulation period is

\[
\boxed{U_2U_1=
\operatorname{diag}(-\omega_1/\omega_2,-\omega_2/\omega_1).}
\]

For distinct frequencies one real multiplier has modulus greater than one,
while the product is one. Both instantaneous stiffnesses were positive.
This is exact parametric amplification of an excited tangent direction,
with work at the coefficient switches. Equal frequencies give `-I`, with
no amplification; an initial vector along the contracting eigendirection
also does not grow on these repeated returns.

The example is not limited to discontinuous signals in principle. Smooth
positive periodic approximations to the specified piecewise coefficient can
converge in the integrated absolute-error norm on its fixed period. The
variation-of-constants estimate gives continuous dependence of the return
matrix on that coefficient. The strict real reciprocal multiplier separation
therefore persists for sufficiently close approximations. This is an
existence statement, without a numerically certified smoothing radius.
No frequency band, nonlinear acquisition or unlimited nonlinear growth is
inferred. In particular it does not imply that every global rhythm resonates
with every organization.

The original autonomous replica pulse already supplies a separate same-law
example of periodic variational coefficients and conditional instability.
Its mechanism, remainder and selected preparation remain in the
[pulse owner](SINE_REPLICA_PULSE.md#sine-replica-pulse-splitting). The prescribed
parameter example here explains a distinction; it does not replace that proof.

## 6. An autonomous reciprocal source must react to the network

A minimal constructive alternative can carry its own finite storage. The
following example adds a canonical pair `(Q,P)` with supplied
`Omega>0,epsilon` in the declared nondimensional chart:

\[
\boxed{\mathcal H=
 \tfrac12x^TLx+\beta e^{\epsilon Q}V(\theta)
 +\tfrac12P^2+\tfrac12\Omega^2Q^2.}
\]

Use the fixed nodal Poisson matrix with blocks `(0,-bK;bK,0)` and the
canonical source bracket `Q_dot=partial_P(H),P_dot=-partial_Q(H)`. Keep
`e=0`, `a=beta*b`, support and capacities fixed. All four complete rows are

\[
\begin{aligned}
\dot x&=a e^{\epsilon Q}KS,&\dot\theta&=bKq,\\
\dot Q&=P,&\dot P&=-\Omega^2Q-
                  \beta\epsilon e^{\epsilon Q}V.
\end{aligned}
\]

The exponential makes the phase-storage coefficient positive for all real
`Q`; it is a supplied coupling choice, not a derived law. A single `Q`
acting on the whole graph is an explicitly global interaction premise.
No propagation speed, locality or physical source is obtained by calling
it global. Setting `epsilon=0` gives an uncoupled oscillator, which cannot
excite the nodal state.

Let `E_net=x^T Lx/2+beta*exp(epsilon Q)*V` and
`E_src=(P^2+Omega^2*Q^2)/2`. Direct differentiation proves

\[
\dot E_{\rm net}=\beta\epsilon e^{\epsilon Q}PV,
\qquad
\dot E_{\rm src}=-\beta\epsilon e^{\epsilon Q}PV,
\qquad \dot{\mathcal H}=0.
\]

The reaction term in `P_dot` is necessary for this balance. Omitting it and
prescribing a sinusoidal `Q(t)` restores an external pump; it is no longer
this closed autonomous system. Neither exchange direction nor a resonance
response follows from the sign of `epsilon` alone.

All storage terms are nonnegative. At any finite initial `H_0`,

\[
|P|\le\sqrt{2H_0},\qquad |Q|\le\sqrt{2H_0}/\Omega,
\qquad 0\le E_{\rm net}\le H_0.
\]

Connected support bounds relative form through `x^T Lx`, and the conserved
weighted form mean bounds its common origin. Circular phases are compact;
`exp(epsilon Q)` is bounded above and away from zero on this energy family.
The smooth full field therefore has global continuation there. The finite
reservoir cannot supply unlimited net structural energy. This bound does
not establish periodic exchange, synchronization, stability or formation.

There are exact non-excitation controls even in this extended model. At
phase consensus and uniform form, `V=S=q=0`; the network stays fixed while
the source follows its free oscillator. With `Q=P=0` also, the whole state
is stationary. If `Q=A*cos(Omega*t)` on the consensus background, a nodal
tangent mode sees coefficient `exp(epsilon*A*cos(Omega*t))`, whereas source
reaction from nodal phase perturbations begins at second order because
`V` is quadratic at consensus. A finite perturbed trajectory need not keep
that prescribed waveform. No instability band for this new coefficient or
nonlinear energy-transfer trajectory is evaluated in this audit.

This construction proves that a reciprocal source and a positive structural
storage budget can be specified consistently. It **supplies** an oscillator,
its frequency scale, coupling and preparation. It does not derive a primordial
oscillator from the original TNFR state or establish that nature uses one.
The existing fine-state pulse gives a different, conditional derivation of
internal oscillation without adding this canonical pair. A retained TNFR
environment can also be investigated using existing source/contact owners;
eliminating it must keep its feedback and initialization.

## 7. What the hypothesis can legitimately explain

The relevant implications are now separated:

| Proposed role | What is justified | What remains additional |
| --- | --- | --- |
| A derived internal circulation | Exact pulse on an admitted preparation; periodic feedback can affect other modes | Generic acquisition, a universal frequency and selection of common phase |
| A common clock or common origins | Changed timing or origins under the proved equivalence | A new relative organizational mechanism |
| An additive pulse | Response when the source overlaps an observable mode, with explicit work | Origin of that source and selection of its waveform |
| A genuine parameter pulse | Possible parametric amplification under a declared protocol | Its generator, work budget, preparation and nonlinear outcome |
| An autonomous reciprocal source | Consistent full-state exchange with finite conserved storage in the example | Derivation or independent selection of the added source and coupling |

Accordingly, resonance can be a susceptibility of the organization before
any pulse is applied. A pulse can excite that susceptibility, change the
coefficients that determine it, or itself be an autonomous consequence of
coupled structural dynamics. These are compatible possibilities, but none
makes the others automatic. The named Resonance operator and a physical
resonance measurement retain their own admission and measurement contracts.

Promoting a primordial rhythm to a fundamental hypothesis would need a
specified state/domain, complete law, preparation or nonzero-activity rule,
observable, and discriminating consequence. The intended hypothesis allows
distinct local rhythms and asks whether their
coordination produces a collective pulse; a universal exact rhythm imposed
from the outset would be a stronger, different premise.

A useful discriminator must survive the declared common-origin and clock
equivalences, with inputs and observations transformed consistently:
for example relative response, mode selectivity, stored-work transfer, or
formation/retention of an independently defined identity. A fitted period
alone cannot choose among these models. Any later comparison must hold the
source, readout and uncertainty fixed, specify whether the source is retained
or externally prescribed, and compare against the corresponding unpulsed
law. Experiments with fixed external timing are not automatically invariant
under such clock changes. Canonical status and physical identification
require further arguments.

### 7.1. Synchronizing local pulses and constructing a clock reading

A local timing phase must come from the local process being described. The
prepared pulse's internal angle `psi=arg(u+i*sqrt(beta*cos(alpha))*sin(delta))`
is one admitted example on the prepared pulse. It is not the primitive nodal
phase `theta`. Local timing is also admitted without requiring that common
waveform: the broader active family has monotonically advancing local readings,
`-arg(delta_i+i*u_i/sqrt(beta))`, with positive rate bounds proved in the
[joint persistence owner, Section 12.3](SINE_REPLICA_PULSE.md#sine-replica-joint-persistence).
These are distinct internal angle charts, both different from primitive
phase; the minus sign fixes the advancing orientation in the broader family.
That result establishes repeated internal turns without assuming or proving
common timing. Distinct local processes need separately admitted phases or
event counts, including zero-amplitude and stalled-state boundaries. The
global primitive-phase phasor in Section 2 cannot substitute for these phases.

Coordination can then mean, for example, bounded differences of continuous
phase lifts, common long-time cycle rates, or a persistent relation among
selected event times. Those criteria differ; none is inferred merely from a
large instantaneous alignment score. Circular phase alone is not an elapsed
time coordinate: an unwrapped reading retains cycle count and orientation.
A collective clock additionally needs a defined domain, a strictly monotone
reading there, and a rule for any loss of timing availability. Synchrony alone
does not supply those properties. Conversely, averaging monotone local
readings would not prove synchronization: a collective clock attributed to
coordination needs a reproducible relation among local offsets or ticks.

Every complete law used here already consumes a supplied evolution parameter
`t`. Showing local pulses synchronize as functions of `t` could construct an
operational collective clock relative to that parameter; it would not yet
derive the existence or ontological status of time. That stronger claim would
need relational evolution and measurement premises. No synchronization theorem,
local-phase closure, collective clock or new clock calibration is asserted by
the present eight controls. The relevant next discriminator must use admitted
local pulse observations and interactions under the same complete law, without
initializing all local pulses on the synchronized invariant family.

## 8. Finite evidence and scope

The declared protocol is in
`artifacts/global-pulse-audit-20261027/scope.json`. The
[dedicated mathematical controls](../../tests/physics/test_global_pulse_hypothesis.py)
cover exactly eight groups: complete common-clock transformation; origin and
uniform-pressure distinction; spatial projection and port work; observation-
dependent modal response; parameter work and jumps; the two-quarter return;
reciprocal autonomous source balance; and emergence/read-out counterexamples.
They reuse shared tangent, phase, sine and diagnostic owners where applicable,
with independent symbolic identities rather than a new runtime law.

The evaluation record associates those controls with separately frozen
source. Validation status belongs to retained receipts and the research
program. No trajectory, source scan, parameter fit, old producer replay or
empirical comparison is part of this analytical audit. In particular it
neither changes the existing pulse proofs nor turns a supplied Hamiltonian
extension into an implemented canonical default.

<a id="local-pulse-coordination-state"></a>

## 9. Local pulse state does not close its coordination rates

The [active-family theorem](SINE_REPLICA_PULSE.md#sine-replica-joint-persistence)
already admits local timing phases beyond the common-waveform preparation.
This section asks what state predicts their coordination. Keep its complete
conservative doubled `C5`, common positive held capacity `nu`, `w,beta>0`,
original time, and all twenty fine coordinates. There is no new forcing,
clock, oscillator, event or phase-only coupling law.

### 9.1. Reuse the internal angle and derive its actual rates

In the retained nonantipodal chart set

\[
y_i=u_i/\sqrt\beta,\quad
\rho_i=\delta_i^2+y_i^2>0,\quad
\chi_i=-\arg(\delta_i+\mathrm i y_i),\quad
\zeta_i=e^{2\mathrm i\chi_i},\quad
\Omega=\frac{w\nu}{\pi\sqrt\beta}.
\]

The argument has a continuous lift along each admitted trajectory, fixed by
an initial lift. The instantaneous circular observation does not supply its
integer cycle count. Independent pair exchange sends `(delta_i,y_i)` to its
negative, shifts `chi_i` by `pi` modulo a full turn and leaves `rho_i,zeta_i`
unchanged. Pair association remains retained; exchanging constituent labels
is not a free permutation of the five structural pairs.

The [complete pair rows](SINE_PAIR_STATE.md#sine-replica-inheritance) give

\[
\dot\delta_i=\Omega y_i,\qquad
\dot y_i=-\Omega F_i\sin\delta_i,\qquad
F_i=\frac12\sum_{j\sim i}\cos\delta_j\cos(\Theta_j-\Theta_i).
\]

Consequently the exact radial and timing rates are

\[
\boxed{\dot\rho_i=2\Omega y_i(\delta_i-F_i\sin\delta_i),\qquad
\omega_i:=\dot\chi_i=
\Omega\frac{y_i^2+F_i\delta_i\sin\delta_i}{\rho_i},\qquad
\dot\zeta_i=2\mathrm i\omega_i\zeta_i.}
\]

Each expression is invariant under the corresponding pair exchange. On the
already admitted `U_active`, the existing bounds
`Omega*c_r*sinc(D)<=omega_i<=Omega` apply. This reuses Section 12.3 of the
pulse owner; it is not a new monotonicity theorem or a synchronization law.

The local observation retains all unordered internal information in this
chart. For `zeta_i=z_i^R+i*z_i^I`, define

\[
s_i=\frac{\rho_i(1+z_i^R)}2=\delta_i^2,\quad
v_i=\frac{\rho_i(1-z_i^R)}2=y_i^2,\quad
h_i=-\frac{\rho_i z_i^I}2=\delta_i y_i.
\]

Its admitted domain includes `rho_i>0`, `|zeta_i|=1` and
`0<=s_i<D^2`, with the active-family geometry supplied separately. It
reconstructs the existing unordered invariants exactly:

\[
\boxed{R_i=\cos\sqrt{s_i},\qquad U_i=\beta v_i,\qquad
Q_i=\sqrt\beta\,h_i\operatorname{sinc}(\sqrt{s_i}).}
\]

Take `sinc(0)=1`. The even functions of `sqrt(s_i)` have their regular
extensions at `s_i=0`; no division by the phase separation is needed.
Conversely the admitted `(R_i,U_i,Q_i)` determine `s_i=(arccos R_i)^2`,
`v_i=U_i/beta` and `h_i=Q_i/(sqrt(beta)*sinc(sqrt(s_i)))`, and hence
`rho_i,zeta_i`. The denominator is positive for `|delta_i|<D<pi/2`.
At `rho_i=0` the timing phase is unavailable and is not filled with zero.
Thus any closure failure below is not caused by omitting another local
internal coordinate: collective information has been discarded.

### 9.2. Two equal local observations with opposite relative rates

Fix `w=beta=nu=1`, `alpha=2*pi/5`, and one exact parameter
`eta=1/1024`. Let

\[
p=(1,2,3,4,2),\qquad e^0=(1,0,0,0,0),\qquad
h^0=e^0-\tfrac15\mathbf1.
\]

Compare the two full preparations indexed by `sigma=+1,-1`:

\[
\boxed{X_i=0,\quad \delta_i=\eta,\quad u_i=\eta p_i,\qquad
\Theta_i^\sigma=i\alpha+\sigma\eta h_i^0.}
\]

The fine states are `x_(i,+/-)=X_i+/-u_i` and
`theta_(i,+/-)=Theta_i+/-delta_i`. The two preparations have the same
node-associated internal observations

\[
\rho_i=\eta^2(1+p_i^2),\qquad
\zeta_i=\frac{(1-\mathrm i p_i)^2}{1+p_i^2}.
\]

Their local timing phases are not all equal, and neither preparation is the
synchronized invariant pulse. The unequal entries `p_2,p_3` also prevent
identifying this entire observation by reflection of the pair indices.
The full relative mean-phase fields differ; this is not a common origin shift.

Put `R=cos(eta)`. The forward mean gaps are `alpha-sigma*eta` at edge `0`,
`alpha+sigma*eta` at edge `4`, and `alpha` at the other three edges.
Direct evaluation of the actual restoring factors gives

\[
F_1^\sigma=\frac R2[\cos(\alpha-\sigma\eta)+\cos\alpha],\qquad
F_4^\sigma=\frac R2[\cos\alpha+\cos(\alpha+\sigma\eta)],
\]
\[
F_1^\sigma-F_4^\sigma=\sigma R\sin\alpha\sin\eta.
\]

Since `p_1=p_4=2`, their relative timing rate is

\[
\boxed{(\omega_1-\omega_4)^\sigma
 =\sigma\frac{\cos\eta\sin\alpha\sin^2\eta}{5\pi\eta}.}
\]

It is strictly positive for one source and strictly negative for the other.
No autonomous rate law depending only on the candidate local observations
can return both values. Equal local pulse amplitudes and projective phases
therefore do not determine relative timing rates, even with the matched
invariants and strict geometric admission proved next.

### 9.3. Admission and matched totals are independent of the verdict

Use the existing family's radius `r=1/10`, excess ceiling
`epsilon_*=1/1000`, and form-mean interval `(-1,1)`. With

\[
c_r=\cos(\alpha+\sqrt2r),\qquad
\kappa_f=\frac{5-\sqrt5}{2}\min(1,c_r),
\]

all required constants are positive. In fact
`pi/2-alpha=pi/10>3/10>sqrt(2)/10`. The cosine Lipschitz bound and
`cos(alpha)=(sqrt(5)-1)/4>1/4`, `sqrt(2)/10<3/20` give `c_r>1/10`.
Also `sqrt(5)<9/4`, so

\[
\boxed{\kappa_f r^2>\frac{11}{8000}>\frac1{1000}=\varepsilon_*.}
\]

The two sources have zero fine form mean and the same common phase origin
in the target-centered chart. Using `sum(p_i^2)=34` and `||h^0||^2=4/5`,
their full fine radius is exactly

\[
Z_f^2=2\sum_i(u_i^2+\delta_i^2)+2\eta^2\|h^0\|^2
 =\frac{398}{5}\eta^2<r^2.
\]

Their common storage, calculated from all twenty fine edges, satisfies

\[
E-E_*=136\eta^2+
4\cos\alpha\,[5-\cos^2\eta(3+2\cos\eta)].
\]

The bracket equals `5*sin(eta)^2+2*cos(eta)^2*(1-cos(eta))` and is at
most `6*eta^2`. Therefore

\[
0<E-E_*<(136+24)\eta^2=160\eta^2
 =\frac{160}{1048576}<\frac1{1000}.
\]

Both states are strictly in `U_active`, since every internal pair is nontip.
This independently supplies the chart, acute-edge and winding premises;
it does not use the observed rate difference to justify admission. Fine
phase-edge gap multisets are equal between the two sources: exchanging
`alpha+eta` and `alpha-eta` exchanges their four-member edge contributions.
Their minimum phase-edge margins, fine radius, form mean, target-centered
phase mean and full storage therefore match. They retain the same support,
capacities, coefficients and all forward five-edge windings.

These are constructed mathematical preparations, not acquired experimental
sources. Matching the listed conserved totals does not assert equality of
every possible invariant or of the full relative phase geometry. The spatial
association of that geometry is precisely the information being tested.

### 9.4. Adding an instantaneous restoring factor still does not close evolution

Supplying all `F_i` alongside `rho_i,zeta_i` determines the instantaneous
local rates in Section 9.1. It does not give an autonomous continuation of
those factors. Differentiating their actual definition yields

\[
\dot F_i=\frac12\sum_{j\sim i}
\left[-\Omega y_j\sin\delta_j\cos\Delta_{ji}
 -R_j\sin\Delta_{ji}(\dot\Theta_j-\dot\Theta_i)\right],
\qquad \Delta_{ji}=\Theta_j-\Theta_i,
\]
\[
\dot\Theta_i=\frac{b\nu}{2}(LX)_i,
\]

where `L` now denotes the five-cycle Laplacian. Neighboring internal
information enters the first term, and collective form enters the second
through actual phase velocities. A scalar `F_i` is a sum of edge terms,
not a replacement for their relative phase geometry or subsequent motion.

For an independent matched-state control retain the same `eta,p,delta,u`,
but instead set

\[
\boxed{\Theta_i=i\alpha,\qquad X_i^\sigma=\sigma\eta h_i^0.}
\]

The two sources have identical `rho,zeta,F`, with `F_i=cos(eta)*cos(alpha)`
for all pairs. Hence the candidate local observations and their first derivatives agree.
Their relative timing rate `omega_1-omega_4` is initially zero. Their fine
radius is again `398*eta^2/5`, and their equal storage excess is

\[
E-E_*=(136+4)\eta^2+20\cos\alpha\sin^2\eta
 <160\eta^2<\varepsilon_*.
\]

The mean-form term is `2 X^T L X=4*eta^2`, since
`(h^0)^T L h^0=2`. Thus both are also strictly in the same `U_active`,
with equal global origin means, fine phase geometry and the listed conserved
totals as each other. Equality of
storage between this control and the preceding distinct control is not needed
or asserted.

The original-time rate of the relative restoring factor is exactly

\[
(\dot F_1-\dot F_4)^\sigma
 =\frac{\eta\sin\eta\cos\alpha}{2\pi}
  +\sigma\frac{\eta\cos\eta\sin\alpha}{\pi}.
\]

The first term comes from the heterogeneous neighboring internal forms
`p_2=3,p_3=4` and is common to the two states. The second comes solely from
the opposite collective form fields. Since the local states and factors
at pairs `1,4` agree, differentiating their relative timing rate gives

\[
\boxed{(\ddot\chi_1-\ddot\chi_4)^\sigma
 =\frac{\sin^2\eta\cos\alpha}{10\pi^2}
  +\sigma\frac{\sin\eta\cos\eta\sin\alpha}{5\pi^2}.}
\]

The difference between the two values is
`2*sin(eta)*cos(eta)*sin(alpha)/(5*pi^2)>0`. Their signs are also opposite:
`0<eta<alpha<pi/2` gives `tan(eta)<tan(alpha)<2*tan(alpha)`. Thus the
same local observation and instantaneous local velocity can conceal opposite
next changes of a relative timing rate. This does not predict a finite-time
locking outcome; it proves the failure of the proposed augmented-state closure.

### 9.5. A sufficient retained state and the dependency it restores

A sufficient extension is already available. Retain the relative collective
form and mean-phase coordinates together with the local observations, or
retain full `X,Theta` if their common origins are also requested. The exact
reconstruction in Section 9.1 then supplies the existing `(R,U,Q)` state.
The [unordered-state theorem](SINE_PAIR_STATE.md#sine-replica-unordered-state)
closes every consumed row on `(X,Theta,R,U,Q)` in this chart.

Concretely, collective phase velocities depend on `LX`, while collective
form velocities depend on the associated edge currents

\[
\dot X_i=\frac{a\nu}{2}
 R_i\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i).
\]

Those rows evolve the relative geometry that enters `F_i`; internal states
also modulate it through `R_j`. This is reciprocal dependence within the
same full law, not an externally prescribed coupling signal. The current
coefficients cannot be replaced by a function of local timing phase alone.

On the admitted target chart, the common origins are fixed symmetries with
conserved means. Retaining `P_5 X` and `P_5(Theta-Theta_*)` removes only
those origins and leaves the relative dynamics closed. Together with the
unordered internal coordinates this retains eighteen continuous degrees of
freedom; the independent finite pair exchanges do not lower that dimension.
The result proves sufficiency, not minimality of this representation or
impossibility of every smaller observable tailored to another prediction.
It also makes no finite-derivative reconstruction claim for `F` alone.

### 9.6. Coordination observations, clocks and evidence

The pair-associated instantaneous projective timing relation

\[
Z_{ij}=\zeta_i\overline{\zeta_j}
      =e^{2\mathrm i(\chi_i-\chi_j)}
\]

has exact derivative

\[
\dot Z_{ij}=2\mathrm i(\omega_i-\omega_j)Z_{ij}.
\]

It is unchanged by independent member swaps and by common fine form or
primitive-phase origin shifts. It is a timing relation between internal
processes, not the primitive-phase phasor shown to fail in Section 2.
It alone neither retains the integer winding of a continuous timing lift
nor proves that the full unordered internal state returns after a half-turn.
Changing radial amplitude remains observable even when the timing phase agrees.

A common positive clock change multiplies all instantaneous `omega_i` by the
same factor. On `U_active`, ratios such as

\[
D_{ij;0}=\frac{\omega_i-\omega_j}{\omega_0}
\]

are therefore independent of that local clock-speed convention. In the
first control `F_0=cos(eta)^2*cos(alpha)` and the complete local state
of pair `0` are identical across sources, so `omega_0>0` is also identical.
The opposite signs of `D_(1,4;0)` cannot be repaired by a common positive
clock factor. In the second control the initial relative rate is zero, so

\[
\frac{dD_{14;0}}{d\chi_0}
 =\frac{\ddot\chi_1-\ddot\chi_4}{\omega_0^2}
\]

at that instant. This has opposite signs for the matched sources and is
invariant under smooth positive reparameterization: the extra clock-derivative
term multiplies the vanishing relative rate. Adding instantaneous `F` thus
still misses an observed change relative to the local reference clock.
No claim is made about changing the entire observable or its calibration.

Bounded differences of chosen continuous timing lifts, equality of long-time
cycle rates, and agreement of finitely selected ticks remain distinct future
observations. None follows from the positive rate bounds or the sufficient
state alone. An averaged monotone reading is not evidence that interactions
have coordinated local pulses. This gate supplies the actual dependency that
any coordination or collective-clock prediction must retain; it does not
prove synchronization, attraction, organization acquisition or physical time.

The declared scope in
`artifacts/local-pulse-coordination-state-20261027/scope.json` fixes six
[constructed control groups](../../tests/physics/test_local_pulse_coordination_state.py):
local reconstruction and exchanges; full-field observation rates; the
mean-phase counterexample and family admission; the mean-form derivative
counterexample; sufficient retained-state closure; and timing-observation
and clock boundaries. Source association and validation status belong to the
separately frozen receipts and the research program. No trajectory, scan,
new source law or prior scientific producer is evaluated by this proof.

<a id="leading-pulse-interaction"></a>

## 10. The first interaction between local pulse timings

Section 9 identified the collective information that local timing discards.
This section derives the first nonzero interaction using that information.
The common pulse and one predeclared phase-advance pattern have different
leading transverse behavior. The latter is stable within the derived cubic
model at fixed action; transferring that conclusion to a full nonlinear
traveling pulse remains an additional problem.

Keep the same ten nodes, twenty unit edges, held unit capacities and
`e=0,w=beta=nu=1`. Both fine evolution rows and the original clock remain
unchanged. Set

\[
\alpha=2\pi/5,\qquad c=\cos\alpha>0,\qquad a=\tan\alpha,
\qquad \tau=\frac{\sqrt c}{\pi}t.
\]

The last equality is an explicit calculation coordinate, not a new clock
law. All conclusions in `tau` convert to original time by this equality.
The auxiliary slow variable introduced below does not supply an emergent
physical time or replace the local reading `chi` of Section 9.

### 10.1. Expand both rows with the collective scale retained

On the base five-cycle define the real operators

\[
(\mathcal A f)_i=f_{i-1}+f_{i+1},\qquad
L=2I-\mathcal A,\qquad (Df)_i=f_{i+1}-f_{i-1}.
\]

Indices are modulo five. A centered collective perturbation has no common
form or phase origin. Declare the scale through

\[
\delta_i=\varepsilon z_i,\quad
u_i=\sqrt c\,\varepsilon z_i',\quad
\Theta_i=i\alpha+\varepsilon^2h_i,\quad
X_i=\sqrt c\,\varepsilon^2Y_i,
\qquad \mathbf1^{\mathsf T}h=\mathbf1^{\mathsf T}Y=0.
\]

Primes in this section mean `d/dtau`. The form and phase rows from Section 9
give, uniformly for bounded `z,z',h,Y`,

\[
\begin{aligned}
h'&=\tfrac12LY,\\
Y'&=-\tfrac12Lh-\tfrac a4D(z^2)+O(\varepsilon^2),\\
z_i''+z_i
 &=\varepsilon^2\mathcal F_i(z,h)+O(\varepsilon^4),\\
\mathcal F_i(z,h)
 &=\tfrac16z_i^3+\tfrac14z_i(\mathcal A z^2)_i
                         +\tfrac a2z_i(Dh)_i.
\end{aligned}
\]

Products and squares of vectors are componentwise. In particular,

\[
h''+\tfrac14L^2h=-\tfrac a8LD(z^2)+O(\varepsilon^2).
\]

For example, the actual restoring factor expands as

\[
F_i=c-\frac{c\varepsilon^2}{4}(\mathcal A z^2)_i
       -\frac{\sin\alpha\,\varepsilon^2}{2}(Dh)_i
       +O(\varepsilon^4).
\]

Combining this with `sin(epsilon*z)` gives the internal cubic row.
In the mean form row, cancellation of the two unperturbed sine currents
leaves `-Lh/2-a*D(z^2)/4`. The mean phase row is exactly linear in `X`.
Thus the collective response is part of the first internal interaction,
even though its physical coordinates are one power of epsilon smaller.

The two nonzero collective frequencies in `tau` are

\[
\ell_k=\frac{\lambda_k}{2}=1-\cos(2\pi k/5),\qquad k=1,2,
\]

each with two real spatial copies. Both lie strictly between zero and two;
neither is an integer. The common origins have already been removed.
These are the actual frequencies, not assumed fast or relaxing variables.

### 10.2. Retain the collective preparation and its forced response

Fix bounded complex amplitudes `A_i` and the leading internal motion

\[
z_i^0(\tau)=\operatorname{Re}(A_i e^{\mathrm i\tau}).
\]

Let `L^dagger` denote the inverse on the centered space and zero on constants.
The leading collective equation has the particular solution

\[
\begin{aligned}
h_{\mathrm p}(\tau)&=H_0+\operatorname{Re}(H_2e^{2\mathrm i\tau}),\\
H_0&=-\frac a4L^\dagger D|A|^2,\\
H_2&=-\frac a4(L^2-16I)^{-1}LD(A^2).
\end{aligned}
\]

Here `H_2` is the amplitude inside a real part, not the coefficient of
`exp(2*i*tau)` alone. The factors follow from
`(z^0)^2=|A|^2/2+Re(A^2*exp(2*i*tau))/2`.
The inverses retain the constant and second-harmonic responses. They exist
on their stated domains because `lambda_k>0` on centered vectors and
`lambda_k<4`. Setting the response to zero changes the cubic interaction.

For independently supplied centered initial means `h_0,Y_0`, the complete
leading solution is `h=h_p+h_h`, where

\[
\begin{aligned}
h_{\mathrm h}(\tau)
={}&\cos(L\tau/2)\,[h_0-h_{\mathrm p}(0)]\\
 &+2L^\dagger\sin(L\tau/2)
            [\tfrac12LY_0-h_{\mathrm p}'(0)],\\
Y(\tau)&=2L^\dagger h'(\tau).
\end{aligned}
\]

This retains both collective initial coordinates. Equivalently the forced
part is the causal convolution of `-a*LD((z^0)^2)/8` with the collective
oscillator Green function. There is no damping or loss of that memory.
The homogeneous frequencies are generally incommensurate with the internal
period. Their integrals over one interval `[0,2*pi]` are not generally zero.

Multiplying `h_h` by `z^0` produces frequencies `1+/-ell_k`, with the
corresponding signed frequencies. None equals `+/-1`, because
`ell_k` is neither zero nor two. Consequently this term is nonresonant
for the leading internal oscillator, but it contributes a bounded correction
depending on the supplied means. Nonresonance does not authorize erasing it.

For later reference the zero-initial-data oscillator response to
`exp(i*nu*tau)` is

\[
\mathcal K_\nu(\tau)=
\frac{e^{\mathrm i\nu\tau}-\cos\tau-\mathrm i\nu\sin\tau}{1-\nu^2}
\quad (\nu\ne\pm1),
\qquad
\mathcal K_1(\tau)=\frac{\mathrm i}{2}
                         (\sin\tau-\tau e^{\mathrm i\tau}),
\quad \mathcal K_{-1}=\overline{\mathcal K_1}.
\]

It satisfies `K''+K=exp(i*nu*tau)` and `K(0)=K'(0)=0`.
The nonresonant homogeneous terms use denominators
`1-(1+/-ell_k)^2`, that is, products of `ell_k` and `2+/-ell_k`.
The third harmonics use `1-3^2=-8`. These finite, nonzero denominators
give bounded corrections. The resonant terms instead contain linear growth
in fast time; they determine the cubic generator below.

### 10.3. Extract the resonant interaction without freezing the means

Introduce the real symmetric circulants

\[
\begin{aligned}
B&=-a^2DL^\dagger D,\\
C&=-a^2D(L^2-16I)^{-1}LD,\\
N&=I+\mathcal A+B,\qquad M=\mathcal A+C.
\end{aligned}
\]

The letter `C` here denotes this derived matrix, not the coherence diagnostic.
All factors commute; `D` is skew-symmetric. The contribution at frequency
one in `mathcal F(z^0,h)` is `Re(f*exp(i*tau))`, with

\[
f_i=\frac18A_i(N|A|^2)_i
       +\frac1{16}\overline{A_i}\,[M(A^2)]_i.
\]

For clarity, the direct cubic terms contribute

\[
\frac18A_i|A_i|^2+
\sum_{j\sim i}\left(\frac18A_i|A_j|^2
                         +\frac1{16}\overline{A_i}A_j^2\right).
\]

The constant collective response contributes
`A_i*(B|A|^2)_i/8`, and its second harmonic contributes
`bar(A_i)*[C(A^2)]_i/16`. The nonresonant homogeneous response remains
in the corrector rather than in `f`. This is extraction of a harmonic
coefficient, not an assertion that a single-period integral of every other
frequency vanishes.

Writing `T=epsilon^2*tau`, the resonant cubic model is

\[
\boxed{\quad
\mathrm i\frac{dA_i}{dT}
 =\frac1{16}A_i(N|A|^2)_i
       +\frac1{32}\overline{A_i}\,[M(A^2)]_i.\quad}
\]

Matching the complex amplitudes inside the real part gives `2*i*A_T=f`
from `(d^2/dtau^2+1)Re(A(T)*exp(i*tau))` at this order. The model resums
the leading secular coefficient. Section 10.7 states the actual time window
proved for the full-law approximation; merely writing `T` does not establish
uniform validity for `tau` of order `epsilon^-2`.
The polynomial cubic model itself allows `A_i=0`; a pulse timing reading
at that pair does not. Timing conclusions require the nonzero-amplitude
domain specified in Section 10.7.

With the complex derivative treating `A` and `bar(A)` independently, define

\[
\mathcal H(A)=\frac1{32}(|A|^2)^{\mathsf T}N|A|^2
             +\frac1{64}\overline{A^2}^{\mathsf T}M(A^2),
\qquad \mathcal I(A)=\sum_i|A_i|^2.
\]

The second quadratic expression is real because `M` is real symmetric.
The cubic row is exactly `i*A_T=partial H/partial bar(A)`.
Thus its own Hamiltonian `H` and action `I` are conserved. Neither is
identified with the exact full storage at every order. Global phase rotation,
cyclic pair relabeling and independent signs `A_i -> +/-A_i` preserve this
model. The last symmetry is the existing member exchange within each pair.

### 10.4. Exact coefficients and recovery of the known common pulse

For the Fourier vector `exp(2*pi*i*k*i/5)`, let `n_k,m_k` be the respective
symbols of `N,M`; the two uses of `i` are the imaginary unit and pair index.
Put `r=sqrt(5)`. The symbols are

| Mode | `n_k` | `m_k` |
| --- | --- | --- |
| `0` | `3` | `2` |
| `1,4` | `13+6*r` | `-(108+2*r)/41` |
| `2,3` | `3` | `-(358+168*r)/41` |

These follow directly from `a^2=5+2*sqrt(5)` and, for `k!=0`,

\[
n_k=3-2\ell_k+2a^2(2-\ell_k),\qquad
m_k=2-2\ell_k-\frac{2a^2\ell_k^2}{\ell_k+2}.
\]

Mode zero uses `D*1=0`, not a limit through nonzero wave numbers.
The common normalized pattern is the relative equilibrium
`A_i(T)=exp(-i*T/4)` with `I=5`. Linearize by
`A_i=exp(-i*T/4)*(1+w_i)` and write `w=x+i*y`.
Each real spatial mode has the block

\[
\binom{x_k}{y_k}'_T
 =\begin{pmatrix}
      0&(m_k-2)/16\\
      -(2n_k+m_k)/16&0
   \end{pmatrix}\binom{x_k}{y_k}.
\]

Set `P_k=(2-m_k)/8` and `Q_k=-(2*n_k+m_k)/8`. Interchanging the two
real components gives exactly the block
`[[0,Q_k/2],[-P_k/2,0]]` of the
[common-pulse splitting proof](SINE_REPLICA_PULSE.md#sine-replica-pulse-splitting).
In particular,

\[
\begin{aligned}
P_1&=(95+\sqrt5)/164,&Q_1&=-(479+245\sqrt5)/164,\\
P_2&=(55+21\sqrt5)/41,&Q_2&=(14+21\sqrt5)/41.
\end{aligned}
\]

The squared eigenvalue is `-P_k*Q_k/4`: positive in mode one and negative
in mode two. The common phase and amplitude retain their neutral shear.
The existing owner's stronger conclusion about the actual sufficiently small
common periodic pulse follows from its separate analytic return-map proof.
It is not newly inferred from the cubic truncation alone.

As an independent normalization check, the section `A=-i` gives
`z^0=sin(tau)` and zero collective response. Its zero-initial-data correction is

\[
z^3(\tau)=\frac3{16}\sin\tau+\frac1{48}\sin3\tau
                              -\frac\tau4\cos\tau.
\]

It solves `z3''+z3=(2/3)*sin(tau)^3`, recovering the frequency shift
`1-epsilon^2/4` in `tau`. The period correction has not been chosen after
inspecting a trajectory.
For the cosine turning-point preparation `delta(0)=epsilon`, the exact
common-pulse parameter is `m=sin(epsilon)^2=epsilon^2+O(epsilon^4)`.
This preserves the leading splitting coefficients when compared with the
owner's crossing section, after its corresponding phase-basis change.

### 10.5. One predeclared phase-advance pattern

Compare only the common leading pattern with

\[
A_i(T)=e^{\mathrm i\alpha i-\mathrm i\kappa T},\qquad
\kappa=\frac{6+m_2}{32}=-\frac{14+21\sqrt5}{164}.
\]

Both have unit modulus at every pair, action five, identical common fine
origins, and leading full storage excess `20*c*epsilon^2`.
Their exact full energies need not agree at order `epsilon^4` and above.
For this comparison select the derived particular collective response with
no added homogeneous component: `h_0=h_p(0)` and
`Y_0=2*L^dagger*h_p'(0)`. This is explicit preparation, not dynamically
acquired synchronization. Section 10.2 still admits other initial means.
For the common pattern `h_p=0`; for the phase-advance pattern its second
harmonic is generally nonzero.

The phase advance is an internal oscillator pattern. It neither changes the
reference primitive-phase winding nor proves a nonlinear traveling orbit.
To verify it in the cubic model, `|A|^2=1`, `N*1=3` and
`M(A^2)=m_2*A^2`; substitution gives precisely the displayed `kappa`.

For its perturbation put
`A_i=exp(i*alpha*i-i*kappa*T)*(1+w_i)`. Let `w_k` be the Fourier
coefficient of `w`, with mode labels modulo five. Then

\[
\mathrm i\frac d{dT}
 \binom{w_k}{\overline{w_{-k}}}
 =\begin{pmatrix}a_k&b_k\\-b_k&-a_{-k}\end{pmatrix}
   \binom{w_k}{\overline{w_{-k}}},
\]

\[
a_k=\frac{2n_k+2m_{k+2}-m_2}{32},\qquad
b_k=\frac{2n_k+m_2}{32}.
\]

Define

\[
d_k=\frac{m_{k+2}-m_{2-k}}{32},\qquad
s_k=\frac{2n_k+m_{k+2}+m_{2-k}-m_2}{32},\qquad
\Delta_k=s_k^2-b_k^2.
\]

The two frequencies of this block are `d_k+/-sqrt(Delta_k)`; actual
growth exponents include the factor `-i`. The other real block supplies
their conjugate partners. Direct factorization yields

\[
\Delta_k=
\frac{(m_{k+2}+m_{2-k}-2m_2)
      (4n_k+m_{k+2}+m_{2-k})}{1024}.
\]

For the two nonzero representatives,

\[
\begin{aligned}
\Delta_1&=
 \frac{(250+166\sqrt5)(1666+814\sqrt5)}{1024\cdot41^2}>0,\\
\Delta_2&=
 \frac{(690+334\sqrt5)(466-2\sqrt5)}{1024\cdot41^2}>0,\\
d_1&=-\frac{250+166\sqrt5}{32\cdot41},&
d_2&=-\frac{190+2\sqrt5}{32\cdot41}.
\end{aligned}
\]

Both satisfy `Delta_k>d_k^2`. For mode one the difference numerator is
`(250+166*sqrt(5))*(1416+648*sqrt(5))`; for mode two it is
`282080+153504*sqrt(5)`, over the common positive denominator
`1024*41^2`. Thus the transverse cubic modes are nonzero, distinct within
each block, and purely imaginary. They describe bounded oscillatory linear
responses, in contrast to the common pattern's growing direction.
This is not damping, attraction or convergence of arbitrary preparations.

### 10.6. What the constrained cubic energy does establish

The leading transverse conclusion also has a nonlinear statement within
this cubic model. In the rotating frame the stationary functional is
`H-kappa*I`. Its second variation on each nonzero sideband is represented,
up to a positive normalization, by the Hermitian matrix

\[
\begin{pmatrix}a_k&b_k\\b_k&a_{-k}\end{pmatrix}.
\]

Its half-trace is

\[
s_1=\frac{958+490\sqrt5}{32\cdot41}>0,\qquad
s_2=\frac{578+166\sqrt5}{32\cdot41}>0,
\]

and its determinant is `Delta_k-d_k^2>0`. Both matrices are positive
definite. At mode zero the real radial direction changes the action; fixing
`I=5` removes that direction to first order. The remaining imaginary
direction is the common phase symmetry. On the tangent space to `I=5`
modulo that symmetry, the constrained Hessian is therefore positive definite.

Consequently the phase-advance pattern is a strict local constrained minimum
of `H`, modulo its common phase orbit. Conservation of `H` and `I`, with
the local positive quadratic lower bound, proves orbital Lyapunov stability
of this relative equilibrium on `I=5` for the cubic model. Solutions exist
globally on that compact action sphere. Finite member exchanges and cycle
symmetries give equivalent representatives and do not add continuous null
directions. This argument proves stability at fixed action, not attraction
or an identity between distinct initial amplitudes.

The Hamiltonian in this argument is the derived quartic `H`, not a supplied
new microscopic law. Its constrained minimum is useful evidence about the
leading interaction. It is not a minimum theorem for the full TNFR storage,
and it supplies neither an exact traveling pulse of the original flow nor
nonlinear stability of such an orbit if one were independently constructed.

### 10.7. A full-law remainder on a fixed fast-time interval

Here is the comparison actually justified without a further averaging
theorem. Choose any fixed finite `T_0`, in particular `T_0=2*pi`, and a
compact bounded set of `A,h_0,Y_0` with centered collective data. Use
`z^0,h,Y` from Section 10.2 and solve the scalar forced oscillators

\[
(z_i^3)''+z_i^3=\mathcal F_i(z^0,h),\qquad
z_i^3(0)=(z_i^3)'(0)=0.
\]

The superscript `3` labels the cubic correction, not the cube of `z_i`.
The kernels in Section 10.2 give it explicitly as a finite harmonic sum
with resonant secular terms and the specified initial adjustment. Define

\[
\begin{aligned}
\delta^{\mathrm{app}}&=\varepsilon z^0+\varepsilon^3z^3,&
u^{\mathrm{app}}&=\sqrt c\,[\varepsilon(z^0)'
                                      +\varepsilon^3(z^3)'],\\
\Theta_i^{\mathrm{app}}&=i\alpha+\varepsilon^2h_i,&
X^{\mathrm{app}}&=\sqrt c\,\varepsilon^2Y.
\end{aligned}
\]

Both phase rows are satisfied exactly. Taylor's formula for sine and cosine,
on the bounded data and fixed interval, gives residual `O(epsilon^5)` in
the internal form row and `O(epsilon^4)` in the mean form row. All constants
are uniform on that compact set for sufficiently small epsilon. The lower
mean residual order is retained rather than discarded. In particular, the
full twenty-coordinate residual is bounded by `R_T*epsilon^4` in the maximum
norm of the original fine forms and phase lifts, in the `tau` coordinate.

In precisely that norm, the unchanged fine vector field in `tau` is globally
Lipschitz with constant `L_f=2/sqrt(c)`: each four-neighbor row has absolute
Jacobian row sum at most that value. Initialize the exact fine flow at the
same `delta=epsilon*Re(A)`, `u=sqrt(c)*epsilon*Re(i*A)` and the same scaled
collective data. The zero initial correction makes the full initial error
zero. Gronwall's inequality then gives

\[
\boxed{\quad
\sup_{0\le\tau\le T_0}
 \|s_{\mathrm{exact}}(\tau)-s_{\mathrm{app}}(\tau)\|_\infty
 \le \frac{R_T}{L_f}(e^{L_fT_0}-1)\,\varepsilon^4.
\quad}
\]

Here `s` is the complete fine state, not a storage or reduced observation.
The constants and a sufficiently small amplitude interval exist by the
displayed analytic construction; numerical values are not supplied. For
nonzero leading internal amplitudes bounded away from zero, sufficiently
small initial states lie in the existing `U_active`: their excess storage
and centered radius tend to zero, while each pair remains off its tip.
This invokes the previously proved active-family domain without a new
trajectory or a new synchronization assertion.

The bound is for fixed fast time. Its constant is not uniform when
`T_0` grows like `epsilon^-2`. In particular the homogeneous collective
amplitudes can receive slow modulation from the higher-order terms omitted
here. Holding those amplitudes fixed on that longer interval would require
another proof. No full-state comparison on the slow time scale, amplitude
radius, exact alternative periodic orbit or original-flow stability is
obtained from this section's fixed-window estimate.

### 10.8. Timing observations and the scope of the mechanism

The leading oscillator phase is distinct from the admitted local reading.
For `A_i=r_i*exp(i*phi_i)` and `sigma_i=tau+phi_i`, the leading state is

\[
\delta_i=\varepsilon r_i\cos\sigma_i,\qquad
u_i=-\sqrt c\,\varepsilon r_i\sin\sigma_i.
\]

With a continuous lift and the orientation of Section 9,

\[
\chi_i=\arg(\cos\sigma_i+\mathrm i\sqrt c\sin\sigma_i),\qquad
\frac{d\chi_i}{d\sigma_i}
 =\frac{\sqrt c}{\cos^2\sigma_i+c\sin^2\sigma_i}>0.
\]

This is a degree-one increasing reparameterization of the circle, with
`chi(sigma+pi)=chi(sigma)+pi`; the unlabeled observation is still
`exp(2*i*chi)`. Constant auxiliary phase offsets need not give constant
`chi_i-chi_j` or equal instantaneous local speeds. They give bounded lifted
offsets in the leading relative equilibrium, with the same leading cycle
rate. The projective reading alone still omits integer tick counts.
Nor does a cubic relative equilibrium guarantee full-state return: the
retained homogeneous collective response can have other frequencies.

The positive result is an interaction mechanism under the existing law:
collective constant and second-harmonic responses change the internal cubic
coupling, destabilize the common pattern in agreement with its owner, and
admit a different constrained minimum in the cubic model. This makes
coordination a mathematically discriminating possibility rather than a
consequence of local activity alone. Acquisition, a stable full traveling
pulse, coordination from generic active preparation and an emergent global
clock remain unproved.

The scope in `artifacts/leading-pulse-interaction-20261027/scope.json`
selects eight [finite control groups](../../tests/physics/test_leading_pulse_interaction.py):
both-row weighted expansion; collective response and denominators; resonant
projection; cubic Hamiltonian and symmetries; common splitting; the declared
phase-advance equilibrium and Jacobian; transverse signatures and neutral
directions; and the timing-observation bridge. Separate frozen receipts and
the research program own validation status. No trajectory, parameter scan,
new forcing or previous scientific producer is evaluated by this proof.

<a id="phase-advance-periodic-branch"></a>

## 11. An exact periodic branch with successive internal phase advances

The phase-advance pattern of Section 10 does extend to a local family of
exact periodic solutions of the unchanged complete law. The proof below
resolves the repeated internal frequency in a specified symmetry class,
retains the collective response, and reconstructs both fine rows. It is an
existence result for prepared organizations. Their transverse stability and
acquisition from a general active preparation remain separate questions.

Keep `e=0,w=beta=nu=1`, the same ten nodes and twenty unit edges, and
`alpha=2*pi/5`, `c=cos(alpha)`. Fix the common form origin at zero and the
mean-phase reference at `Theta_i=i*alpha`; adding constant origins is an
exact symmetry. As before `tau=sqrt(c)*t/pi`. A further coordinate
`s=Omega*tau` fixes the profile period at `2*pi`, while the positive
frequency `Omega` is solved as part of the branch rather than supplied
as a new force or clock law.

### 11.1. The exact profile equations and their invariant class

For `2*pi`-periodic real profiles introduce shifts and their spatial mean:

\[
(T_\pm f)(s)=f(s\pm\alpha),\qquad
\Lambda=2I-T_+-T_-,\qquad
\Pi_5f=\frac15\sum_{j=0}^4 f(s+j\alpha).
\]

The projection `Pi_5` selects temporal Fourier harmonics whose index is a
multiple of five. Its appearance reflects the actual five-pair spatial
mean, not an average in time. The inverse `Lambda^dagger` is zero on that
range and is the ordinary inverse on its orthogonal complement.

Consider

\[
\delta_i(\tau)=d(s+i\alpha),\qquad
\Theta_i(\tau)=i\alpha+h(s+i\alpha),\qquad s=\Omega\tau.
\]

For a scalar profile argument suppress `s` and write

\[
g_+=\alpha+T_+h-h,\qquad g_-=-\alpha+T_-h-h,
\]

\[
\begin{aligned}
F(d,h)&=\tfrac12[\cos(T_+d)\cos g_+
                         +\cos(T_-d)\cos g_-],\\
G(d,h)&=\cos d\,[\cos(T_+d)\sin g_+
                         +\cos(T_-d)\sin g_-].
\end{aligned}
\]

The complete pair equations give the exact periodic boundary problem

\[
\boxed{\quad
\Omega^2d''+\frac{F(d,h)}c\sin d=0,\qquad
\Omega^2h''-\frac1{4c}\Lambda G(d,h)=0.
\quad}
\]

Primes in Section 11 mean `d/ds`. The form profiles will be recovered in
Section 11.5; they have not been frozen or omitted from the dynamics.
In particular `Pi_5 G(d,h)=0` exactly: when summed over five shifted
arguments, every forward term
`cos(d_i)*cos(d_(i+1))*sin(alpha+h_(i+1)-h_i)` cancels the backward
term on the same edge. The wrap edge uses `5*alpha=2*pi`.

Use the following real symmetry class:

\[
\begin{aligned}
d(-s)&=d(s),& d(s+\pi)&=-d(s),\\
h(-s)&=-h(s),& h(s+\pi)&=h(s),&\Pi_5h&=0.
\end{aligned}
\]

Thus `d` has cosine harmonics with positive odd indices. The profile `h`
has sine harmonics with positive even indices, excluding multiples of five.
The class fixes a temporal reflection section and removes the common mean
coordinates; it does not assert a simple eigenpair in the full state space.

These restrictions preserve the exact equations. Under `s -> -s`, the two
neighbors interchange, `g_+(-s)=-g_-(s)`, and the cosine factors interchange.
Therefore `F` is even and `G` is odd. Under `s -> s+pi`, the cosine factors
and `G` are unchanged, while `sin(d)` changes sign. Finally `Lambda G`
has zero five-shift mean. These identities also implement the full law's
reversal, cyclic and member-exchange symmetries in this profile class.
No invariant-class assertion is inferred merely from the leading waveform.

### 11.2. Resolve the actual linear kernel and inverse

Work with real periodic Sobolev spaces. Let `D^2` be the stated class of
`d` profiles in `H^2`, and `H^2_0` the stated class of centered `h` profiles.
The corresponding `L^2` classes are the equation ranges. Superscripts here
label Sobolev regularity, not powers of profiles or the Hamiltonian of Section 10.
In one dimension `H^2` embeds continuously in `C^1` and is an algebra.
Products, fixed shifts, sine and cosine therefore give real analytic maps
from these spaces into `H^2`, and hence into `L^2`.

At `d=h=0`, `Omega=1`, the derivative of the boundary problem is block
diagonal:

\[
\mathcal L_d=\partial_s^2+1,\qquad
\mathcal L_h=\partial_s^2+\tfrac14\Lambda^2.
\]

For an integer harmonic `n`, the shift Laplacian has multiplier

\[
\lambda_n=2-2\cos(n\alpha),\qquad \ell_n=\lambda_n/2.
\]

The internal multiplier is `1-n^2`. On odd cosine harmonics its only kernel
is `cos(s)`; all other allowed indices satisfy `n>=3` and
`n^2-1>=(8/9)*n^2`. The original five internal oscillators share frequency
one, but their resonant directions are not all retained by this traveling
profile and reflection class. The class leaves exactly this one real kernel.

The collective indices are even, `n>=2`, and not multiples of five.
Their multipliers are `ell_n^2-n^2`. Since

\[
0<\ell_n\le\ell_2=\frac{5+\sqrt5}{4}<2,
\qquad
n^2-\ell_n^2\ge
 \left(1-\frac{\ell_2^2}{4}\right)n^2,
\]

they never vanish. The lowest sufficient gap is

\[
4-\ell_2^2=\frac{17-5\sqrt5}{8}>0.
\]

These are bounds on every allowed harmonic, not a finite Fourier truncation.
Both inverse multipliers decay as `n^-2`. Consequently `L_d` on the
orthogonal complement of `cos(s)` and `L_h` on the entire collective class
have bounded inverses from their `L^2` ranges to their `H^2` domains.
There is no loss of derivatives and no small-divisor assumption. In
particular the argument does not invoke a simple-eigenpair theorem for the
degenerate full equilibrium or discard the collective modes.

### 11.3. Complementary equations and scalar transversality

Normalize the first cosine coefficient exactly. With

\[
\mathfrak p f=\frac1\pi\int_0^{2\pi}f(s)\cos s\,ds,
\qquad Qf=f-(\mathfrak p f)\cos s,
\]

write

\[
d=\varepsilon\cos s+v,\qquad \mathfrak p v=0,
\qquad \omega=\Omega^2-1.
\]

Let `E_d,E_h` denote the two exact residuals in Section 11.1, with
`Omega^2=1+omega`. Apply the analytic implicit-function theorem to

\[
(v,h)\longmapsto
 \bigl(QE_d(\varepsilon\cos+v,h,\omega),
       E_h(\varepsilon\cos+v,h,\omega)\bigr).
\]

The derivative at the origin is precisely the two boundedly invertible
operators just established. Hence there are unique local analytic functions
`v(epsilon,omega),h(epsilon,omega)` solving these complementary equations.
At `epsilon=0` both vanish for every nearby `omega`. The symmetry
`d -> -d,h -> h` makes `v` odd and `h` even in epsilon. Differentiating the
complementary equations once at epsilon zero gives zero: the residual of
`cos(s)` is entirely in the removed cosine direction. Thus

\[
v(\varepsilon,\omega)=O(\varepsilon^3),\qquad
h(\varepsilon,\omega)=O(\varepsilon^2)
\]

locally and uniformly for nearby `omega`, in the stated profile norms.
It remains to solve the scalar residual

\[
r(\varepsilon,\omega)=
 \mathfrak p E_d(\varepsilon\cos+v(\varepsilon,\omega),
                         h(\varepsilon,\omega),\omega).
\]

This analytic function is odd in epsilon. Its quotient by epsilon extends
analytically across zero, and the exact linear calculation gives

\[
\left.\frac{r(\varepsilon,\omega)}{\varepsilon}
\right|_{\varepsilon=0}=-\omega,
\qquad
\partial_\omega(r/\varepsilon)(0,0)=-1.
\]

A second scalar implicit-function theorem supplies a unique local even
analytic `omega(epsilon)`, with `omega(0)=0`. Take the positive branch
`Omega(epsilon)=sqrt(1+omega(epsilon))`. This constructs exact periodic
profiles for all sufficiently small nonzero epsilon. Local uniqueness is
within this normalization and symmetry class, not among all organizations
or periodic solutions of the full law.

The profiles are smooth in `s`. Starting with `H^2`, the equations express
their second derivatives through analytic products and shifts of `H^2`
functions, so the profiles lie in `H^4`; repeating gives every finite
Sobolev regularity. Analyticity of the branch in epsilon in `H^2` is enough
for the controlled expansion below; no separate analyticity claim about
complex time is required.

### 11.4. The resulting frequency and collective profile

Let the coefficient from Section 10 be

\[
\kappa=-\frac{14+21\sqrt5}{164},\qquad
B=\frac{95+40\sqrt5}{164},\qquad
b=\frac1{24}-\frac\kappa4=\frac{124+63\sqrt5}{1968}.
\]

Expanding the exact complementary equations and their scalar residual gives

\[
\boxed{\begin{aligned}
d(s;\varepsilon)&=\varepsilon\cos s
                       +b\varepsilon^3\cos3s+O_{H^2}(\varepsilon^5),\\
h(s;\varepsilon)&=-B\varepsilon^2\sin2s
                                      +O_{H^2}(\varepsilon^4),\\
\Omega(\varepsilon)&=1-\kappa\varepsilon^2+O(\varepsilon^4).
\end{aligned}}
\]

The mean-form profile `Y=X/sqrt(c)` is

\[
Y(s;\varepsilon)=
 -\frac{55+21\sqrt5}{82}\varepsilon^2\cos2s
                                      +O_{H^1}(\varepsilon^4).
\]

For example, the collective second-harmonic equation gives the same
`H_2=i*B` at pair zero as Section 10, hence `h_2=-B*sin(2*s)`.
Substituting that response into the internal cubic force gives coefficients
`2*kappa` at `cos(s)` and `2*kappa-1/3` at `cos(3*s)`. Therefore

\[
\omega(\varepsilon)=-2\kappa\varepsilon^2+O(\varepsilon^4),
\qquad
(\partial_s^2+1)v_3=(2\kappa-\tfrac13)\cos3s,
\]

which yields the displayed `b`. There is no third-order `cos(s)` term:
the normalization fixes that coefficient exactly at epsilon. The mean-form
coefficient follows from `Y=2*Omega*Lambda^dagger*h'`.

Analyticity and parity give constants bounding each displayed remainder in
its indicated norm on a sufficiently small amplitude interval. Those
constants and the interval are existential, not evaluated numerical bounds.
The agreement with the cubic generator is a consequence checked in the exact
equations; it is not used as an existence theorem for the branch.

### 11.5. Reconstruct every first-order coordinate

For the exact profiles set

\[
Y=2\Omega\Lambda^\dagger h',\qquad
X_i=\sqrt c\,Y(s+i\alpha),\qquad
u_i=\sqrt c\,\Omega d'(s+i\alpha).
\]

Then `Omega*h'=Lambda*Y/2`, since `Pi_5 h=0`. These formulas reproduce
the mean and internal phase rows exactly in `tau`. Differentiating and
using the two profile equations yields

\[
\Omega Y'=\frac{G(d,h)}{2c},\qquad
\frac{du_i}{d\tau}=-\frac{F(d,h)}{\sqrt c}\sin d
                         \quad\text{at }s+i\alpha.
\]

For the first identity, `Lambda^dagger*Lambda*G=G` uses the previously
proved cancellation `Pi_5 G=0`. It is essential: without centering the
current, a second-order mean equation would not recover its full form row.
Multiplication by `d tau/dt=sqrt(c)/pi` now gives exactly the original
four retained rows, and hence all twenty fine rows through

\[
x_{i,\pm}=X_i\pm u_i,\qquad
\theta_{i,\pm}=i\alpha+h(s+i\alpha)\pm d(s+i\alpha).
\]

No pressure reconstruction from an observed trajectory, auxiliary forcing,
unmodeled reservoir or altered coefficient is involved. The supplied graph,
complete law and the branch's prepared initial state remain explicit.

### 11.6. Admission and nonvanishing local activity

The exact family stays within the existing acute active organization for
every sufficiently small nonzero epsilon. Indeed the profile expansion and
the Sobolev embedding give, uniformly in `s`,

\[
\bigl(d(s),\sqrt c\,\Omega d'(s)\bigr)
 =\varepsilon(\cos s,-\sqrt c\sin s)+O(\varepsilon^3).
\]

The leading ellipse has squared radius at least `c*epsilon^2`.
Shrinking the amplitude interval if necessary gives
`d(s)^2+c*Omega^2*d'(s)^2 >= c*epsilon^2/2 > 0` at every argument.
Thus no constituent pair reaches its synchronized tip. This conclusion
uses a uniform profile estimate, not samples at selected phases.

All common origins are fixed. The full storage and centered fine radius obey

\[
E-E_*=20c\varepsilon^2+O(\varepsilon^4),\qquad
Z_f^2=5(1+c)\varepsilon^2+O(\varepsilon^4)
\]

uniformly along the branch. The leading identities use the five shifted
cosines and sines, whose squared sums both equal `5/2`; the collective
coordinates first contribute at order four. Full storage is exactly
conserved by the reconstructed law, not merely by this expansion.

Choose a fixed radius and positive excess ceiling satisfying the
[active-family admission](SINE_REPLICA_PULSE.md#sine-replica-joint-persistence),
with a mean-form interval containing zero, for example `(-1,1)`.
The displayed quantities tend uniformly to zero. For sufficiently small
nonzero epsilon the entire exact orbit therefore lies in that family's
radius and energy collar and avoids all tips. The zero common form lies in
its admitted mean interval. Its positive pair resultants, acute geometry,
winding-one organization and positive local circulation-rate bounds apply.
No explicit numerical amplitude threshold is obtained, and no generic
entry into this protected family is asserted.

### 11.7. Exact periods, pair exchanges and local readings

The nonzero first cosine coefficient of `d` is exactly epsilon. A labeled
period `s_0` of the fine orbit would in particular be a period of `d` at
pair zero. Its first Fourier coefficient requires `exp(i*s_0)=1`.
The smallest positive labeled profile period is consequently `2*pi`, and
the original-time period is

\[
\boxed{\quad
P(\varepsilon)=\frac{2\pi^2}{\sqrt c\,\Omega(\varepsilon)}.
\quad}
\]

At a profile shift by pi, both `d` and `d'` change sign, while `h` and `Y`
are unchanged. This is exactly the global within-pair member exchange.
It returns the unordered pair state, while generally changing the labeled
fine state. To exclude a smaller unordered period, use the invariant

\[
R_0(s)=\cos d(s)
 =1-\frac{\varepsilon^2}{4}
      -\frac{\varepsilon^2}{4}\cos2s+O(\varepsilon^4).
\]

Its second-harmonic coefficient is nonzero for sufficiently small nonzero
epsilon, so a period must be an integer multiple of pi. Any exact return
of the complete state modulo independent member exchanges also returns
`R_0`; equivariance and uniqueness of the full flow make such a return a
period of the invariant history. The minimum period modulo those exchanges
is therefore exactly `P(epsilon)/2`.

Pair association is fixed in this claim. Cyclic relabeling of the five pairs
is not a free identification in the return observation. The additional
traveling relation shifts profile time by alpha and cyclically advances the
pair index; its primitive phases include the corresponding common offset
`-alpha` modulo `2*pi`. That space-time symmetry does not shorten the
labeled or member-exchange periods just proved.

The local readings of Section 9 now have an exact periodic realization.
Choose continuous lifts of

\[
\chi_i(s)=-\arg\bigl(d(s+i\alpha)
                       +\mathrm i\sqrt c\,\Omega d'(s+i\alpha)\bigr).
\]

The uniform nonvanishing estimate allows a homotopy to the leading ellipse.
Consequently `chi_i` advances by `2*pi` over one labeled period and by pi
over the member-exchange half-period. Its strict monotonicity also follows
from the already admitted active-family theorem. With consistent initial
lifts, the readings are time-shifted copies of one profile, so their lifted
differences are bounded and periodic. Their full cycle rates agree exactly;
their instantaneous rates and phase differences need not be equal or
constant. The projective readings `exp(2*i*chi_i)` retain their own half-period
and still require a lift to count ticks.

This supplies conditional coordination of actual local motions without an
external oscillator. It does not establish synchronization from unprepared
data, selected amplitudes, an ontological origin for time or physical units.

### 11.8. What existence closes and what it leaves open

The derived alternative is now an exact small-amplitude periodic branch of
the complete conservative model, with collective feedback and maintained
local activity. The previous fixed-window approximation no longer bears
the burden of establishing this branch's existence or period. Its own
approximation theorem and its cubic stability statement retain their original
scope; neither automatically becomes a stability theorem for this exact
branch. Perturbations outside the imposed symmetry class remain present in
the full twenty-coordinate dynamics.

In particular, exact existence supplies no transverse Floquet conclusion,
attraction, acquisition basin, universally selected pulse or empirical
identification. The proof is local in amplitude and does not classify the
whole periodic component. Positive behavior in this prepared family is
compatible with the known instability of the different common pulse.

The declared scope in `artifacts/phase-advance-periodic-branch-20261027/scope.json`
selects seven [finite control groups](../../tests/physics/test_phase_advance_periodic_branch.py):
actual fine symmetries; profile filters and inverse bounds; complementary
projection and scalar transversality; collective response; cubic solvability;
both phase-row reconstructions and return distinctions; and storage/activity
admission. Frozen receipts and the research program own evaluation status.
No trajectory, amplitude scan or prior producer replay is part of this proof.

<a id="phase-advance-transverse-spectrum"></a>

## 12. The exact branch has no growing transverse Floquet multipliers locally

For every sufficiently small nonzero amplitude on the exact branch of
Section 11, all twenty labeled-return multipliers lie on the unit circle.
The sixteen nonneutral multipliers are semisimple. At multiplier one there
are two constant-origin directions and the autonomous phase/family pair;
the latter has a genuine size-two Jordan block. Thus this is full spectral
stability with a stated neutral shear, not bounded evolution of every raw
variation or a nonlinear stability theorem.

The proof retains perturbations outside the traveling profile class. Its
connection to the cubic interaction is derived from the complete variational
rows and a periodic spectral reduction, rather than from a long-time
approximation by that cubic model. The amplitude interval is existential;
no numerical cutoff, return matrix or multiplier is evaluated here.

### 12.1. The full twenty-coordinate variational law

Keep the exact branch, unit parameters and original time of Section 11.
Let `L_f` be the unweighted Laplacian on its actual ten-node, twenty-edge
support. Use calculation coordinates

\[
q=\theta-\theta_*,\qquad y=x/\sqrt c,\qquad
\tau=\sqrt c\,t/\pi,\qquad s=\Omega(\varepsilon)\tau.
\]

The letter `y` is a normalized fine form coordinate, not pressure. The
nodal pressure and every original evolution row remain unchanged. If
`V(theta)=sum_edges[1-cos(theta_j-theta_i)]`, the exact equations in tau
are Hamiltonian with

\[
J=\begin{pmatrix}0&I_{10}\\-I_{10}&0\end{pmatrix},\qquad
H_f=\frac18y^{\mathsf T}L_fy+\frac{V(\theta)}{4c}
    =\frac{E}{4c}.
\]

Here the state ordering is `(q,y)` and `d(q,y)/dtau=J grad(H_f)`.
Along the exact branch let `L_cos(s,epsilon)` be the signed Laplacian
whose edge coefficient is the cosine of the actual fine phase gap. The
full variational matrix in fixed-period coordinate s is exactly

\[
\boxed{\quad
\mathcal A_\varepsilon(s)=\frac1{\Omega(\varepsilon)}
\begin{pmatrix}
0&L_f/4\\
-L_{\cos}(s,\varepsilon)/(4c)&0
\end{pmatrix}.
\quad}
\]

The phase variation drives the form variation and conversely. Both matrices
are symmetric, so `A^T J+J A=0`; the fundamental matrix is real symplectic.
No subset of phase variations, constituents or collective coordinates has
been removed from this twenty-row law.

The two vectors `(1,0)` and `(0,1)` are in its kernel for every s. They are
the independent common phase and form origins. Their symplectic plane is a
constant identity block. Fixing the two means leaves the invariant real
eighteen-dimensional centered symplectic space. Removing those origins is
therefore an exact decomposition, not an approximation or a stability claim.

Let `Phi_epsilon(0)=I` be the fundamental matrix in s and
`M_epsilon=Phi_epsilon(2*pi)` the labeled return at the fixed branch section.
The branch, Omega and the coefficient matrix are analytic in epsilon on the
fixed compact interval. The finite-interval linear equation consequently
gives an analytic monodromy and locally bounded Taylor remainders. This is
an analytic property of the equation, not a numerical integration campaign.
The reference epsilon and its Omega are held fixed when varying the initial
state; changing the period separately for each perturbation would erase part
of the family shear.

### 12.2. Account for every limiting multiplier

At epsilon zero, `L_cos=c*L_f`. The fine Laplacian has eigenvalue zero once,
internal eigenvalue four with multiplicity five, and the two collective
values `2*lambda_k`, each with multiplicity two, where

\[
\lambda_k=2-2\cos(2\pi k/5),\qquad
\ell_k=\lambda_k/2,\qquad k=1,2.
\]

After removing the two origins, the limiting return consists of

| Sector | Dimension over the reals | Limiting labeled multipliers |
| --- | --- | --- |
| Internal | `10` | `1`, total multiplicity ten |
| First collective mode | `4` | `exp(+/-2*pi*i*ell_1)`, each complex multiplicity two |
| Second collective mode | `4` | `exp(+/-2*pi*i*ell_2)`, each complex multiplicity two |

The four collective values are distinct from each other and from `+/-1`.
Indeed `ell_k` and `2*ell_k` are irrational,
`ell_2-ell_1=sqrt(5)/2` is not an integer, and
`ell_1+ell_2=5/2` is not an integer. These facts check both equal-frequency
and opposite-frequency coincidences. They allow separate analytic spectral
projections for the internal cluster and each collective cluster.

The internal cluster at one is degenerate and contains both oscillator
orientations. It cannot receive the definite-cluster argument below merely
because the reference equilibrium has positive quadratic storage.

### 12.3. The repeated collective clusters remain on the unit circle

On the complexified centered space use the Hermitian form

\[
\mathfrak k(z,w)=-\mathrm i\,\overline z^{\mathsf T}Jw.
\]

Every real symplectic return preserves it. A limiting eigenvector with
frequency `+ell_k` has the form `(v,i*v)`, where v belongs to the
corresponding fine phase eigenspace. It satisfies

\[
\mathfrak k((v,\mathrm iv),(v,\mathrm iv))=2\|v\|_2^2>0.
\]

Its conjugate negative-frequency space is negative definite. These are
statements on the whole two-dimensional complex eigenspace of each repeated
collective multiplier, not just one selected eigenvector.

Each spectral subspace varies continuously, so its definite restriction
persists for sufficiently small amplitude. The restricted return preserves
that definite form and is therefore unitary in the resulting positive inner
product, after reversing the sign for a negative cluster. All its eigenvalues
are exactly on the unit circle and it is diagonalizable. Thus all eight
collective multipliers are semisimple and remain on the circle, whether or
not their initial double multiplicities split. No calculated splitting or
instantaneous-stiffness argument is needed for these separated clusters.

### 12.4. The actual slow spectral graph selects the collective response

Let `E_s` be the ten-dimensional internal space at zero amplitude and `E_f`
the eight-dimensional centered collective space. The spectral subspace of
`M_epsilon` near multiplier one is a real analytic graph

\[
\{(v,K(\varepsilon)v):v\in E_s\},\qquad K(0)=0,
\]

in the fixed decomposition `E_s+E_f`. Its existence follows from the
isolated spectral projection and invertibility of its projection onto `E_s`
near zero. Write the return blocks in this decomposition as `M_ab`.
Graph invariance is the exact equation

\[
M_{fs}+M_{ff}K=K(M_{ss}+M_{sf}K).
\]

Changing epsilon's sign is the global within-pair exchange: it is minus the
identity on `E_s` and plus the identity on `E_f`. The first-order return
coefficient therefore has only cross blocks, and `M_ss'(0)=0`. At order
epsilon the graph equation gives

\[
(M_{ff}(0)-I)K_1=-M_{fs}'(0).
\]

The inverse exists because no collective multiplier equals one. It fixes
the leading collective initialization of every internal spectral vector.
Over the period, that order-epsilon collective variation must be periodic:
the internal return changes its input only at order epsilon squared.
It is not legitimate to attach an independently chosen homogeneous mean
motion to this graph. Those independent motions belong to the fast
subspaces already retained and analyzed in Section 12.3.

The periodic response can also be obtained directly from the full rows.
Use the base-cycle operators `L,D,mathcal A` and `a=tan(alpha)` of
Section 10. Put

\[
A_i^*=e^{\mathrm i\alpha i},\qquad
z_i(s)=\operatorname{Re}(A_i^*e^{\mathrm is}),\qquad
\xi_i^0(s)=\operatorname{Re}(Z_i e^{\mathrm is}),
\quad Z\in\mathbb C^5.
\]

The five arbitrary complex entries of Z span all ten real internal
variations; they are not restricted to the traveling profile symmetry.
For the slow graph the mean-phase variation is
`epsilon*H_Z+O(epsilon^2)`. Differentiating the complete mean equation gives

\[
H_Z''+\tfrac14L^2H_Z=-\tfrac a4LD(z\,\xi^0).
\]

On centered vectors there is a unique `2*pi`-periodic solution, because
the collective frequencies are not integers. It is the actual derivative
of the response in Section 10:

\[
\begin{aligned}
H_Z(s)&=\delta H_0+\operatorname{Re}(\delta H_2e^{2\mathrm is}),\\
\delta H_0&=-\frac a4L^\dagger D
                  (A^*\overline Z+\overline{A^*}Z),\\
\delta H_2&=-\frac a2(L^2-16I)^{-1}LD(A^*Z).
\end{aligned}
\]

The collective form variation is recovered by the mean phase row. The
graph construction and this unique periodic solution describe the same
initialization, so no free noninteger-frequency term has been silently
averaged to zero.

Let `h_2(s)` be the branch's leading collective phase profile from
Section 11. Expansion of the actual internal variational row, including
the solved frequency `Omega=1-kappa*epsilon^2+O(epsilon^4)`, gives

\[
\begin{aligned}
\xi''+\xi=\varepsilon^2\bigl\{&
 [z^2/2+\mathcal A(z^2)/4+(a/2)Dh_2]\,\xi^0\\
 &+(z/2)\,\mathcal A(z\,\xi^0)
 +(a/2)z\,DH_Z-2\kappa\xi^0\bigr\}
 +O(\varepsilon^3).
\end{aligned}
\]

The remainder is uniform on the fixed period for bounded initial variation.
The last term follows from `Omega^2=1-2*kappa*epsilon^2+...`; omitting it
would change the return. Both the branch mean profile and the induced mean
variation appear. No internal cubic term alone supplies this equation.

Define the cubic row from Section 10 by

\[
\mathcal L_i(A)=\frac1{16}A_i(N|A|^2)_i
                      +\frac1{32}\overline{A_i}\,[M(A^2)]_i.
\]

Its derivative is real-linear. Its first-harmonic forcing amplitude is
`f=2*mathcal L`. The displayed
variational row therefore has resonant amplitude
`D f(A*)[Z]-2*kappa*Z`. Its full-period secular coefficient is the
real-linear map

\[
\boxed{\quad G_2Z=-\mathrm i\,[D\mathcal L(A^*)[Z]-\kappa Z].\quad}
\]

The other internal harmonics at this order are integer third harmonics;
their correctly initialized oscillator responses have zero full-period
increment. This derives the coefficient of the actual return from the full
variational equation. It does not presume uniform approximation of nonlinear
trajectories on times of order `epsilon^-2`.

### 12.5. Hamiltonian logarithm and the eight simple slow eigenvalues

The slow spectral subspace is symplectic: it is the nondegenerate internal
plane at zero and remains nondegenerate by continuity. Choose an analytic
symplectic identification with a fixed ten-dimensional symplectic space.
The graph's pulled-back form differs from its limiting form only at order
epsilon squared, so this normalization does not change the leading return
coefficient just derived. In that identification the actual slow return is

\[
N_\varepsilon=I+2\pi\varepsilon^2G_2+O(\varepsilon^3).
\]

Its principal matrix logarithm is real and Hamiltonian. This follows from
symplecticity, `log(N^-1)=-log(N)` near the identity, and the transpose
identity for the logarithm. Consequently

\[
G(\varepsilon)=\frac{\log N_\varepsilon}{2\pi\varepsilon^2}
              =G_2+O(\varepsilon)
\]

extends analytically through epsilon zero and is Hamiltonian in the fixed
form. These remainder statements follow from an analytic finite-period
return, not a formal long-time averaged equation. No sign convention for
the effective quartic Hamiltonian is transferred to the physical storage.

For completeness, the limiting eigenvalues can be isolated exactly. The
fixed change `Z_i=A_i^* w_i` gives the Bloch blocks derived in Section 10,
with symbols `n_k,m_k` there and

\[
\begin{aligned}
a_k&=(2n_k+2m_{k+2}-m_2)/32,&
b_k&=(2n_k+m_2)/32,\\
d_k&=(m_{k+2}-m_{2-k})/32,&
\Delta_k&=
 \frac{(m_{k+2}+m_{2-k}-2m_2)
       (4n_k+m_{k+2}+m_{2-k})}{1024}.
\end{aligned}
\]

The two frequencies in each representative block are
`d_k+/-sqrt(Delta_k)`, for `k=1,2`; conjugation supplies the other real
directions. Since `Delta_k>d_k^2`, the four positive magnitudes have
disjoint rational enclosures:

| Positive magnitude | Exact enclosing interval |
| --- | --- |
| `sqrt(Delta_2)+d_2` | `(2/5,1/2)` |
| `sqrt(Delta_1)+d_1` | `(3/5,7/10)` |
| `sqrt(Delta_2)-d_2` | `(3/4,4/5)` |
| `sqrt(Delta_1)-d_1` | `(3/2,5/3)` |

For example `223/100<sqrt(5)<56/25` and the exact expressions in
Section 10 give

\[
\begin{aligned}
28/25&<\sqrt{\Delta_1}<113/100,&
47/100&<-d_1<12/25,\\
31/50&<\sqrt{\Delta_2}<63/100,&
7/50&<-d_2<3/20.
\end{aligned}
\]

Squaring positive endpoints verifies the root bounds; adding and subtracting
gives the table. Thus the eight nonzero eigenvalues of `G_2` are distinct,
simple and purely imaginary, separated from the remaining double zero.

Choose disjoint small disks around those eight limiting eigenvalues, avoiding
zero and invariant individually under `lambda -> -bar(lambda)`. Analytic
perturbation keeps exactly one eigenvalue of `G(epsilon)` in each disk for
sufficiently small amplitude. A real Hamiltonian matrix carries every
eigenvalue to `-bar(lambda)`. Uniqueness within its disk therefore forces
each of these perturbed eigenvalues to equal its own image, so its real part
is exactly zero. This is stronger than retaining only its leading imaginary
coefficient.

Exponentiating `2*pi*epsilon^2*G(epsilon)` places all eight nonneutral slow
multipliers exactly on the unit circle. For sufficiently small amplitude
the exponential is injective on these small spectral neighborhoods, so they
remain simple. All constants specifying the neighborhoods, remainder bounds
and amplitude interval exist locally; no numerical values are supplied.

### 12.6. The neutral pair is an actual Jordan shear

Let `z_epsilon` be the branch's full initial state in `(q,y)` coordinates,
`v_epsilon` its autonomous vector field in tau, and
`w_epsilon=partial_epsilon z_epsilon`. Its labeled period in tau is

\[
P_\tau(\varepsilon)=\frac{2\pi}{\Omega(\varepsilon)},\qquad
P_\tau'(\varepsilon)=4\pi\kappa\varepsilon+O(\varepsilon^3).
\]

Differentiating the exact periodic-orbit identity at its varying period gives

\[
M_\varepsilon v_\varepsilon=v_\varepsilon,\qquad
(M_\varepsilon-I)w_\varepsilon
                 =-P_\tau'(\varepsilon)v_\varepsilon.
\]

The coefficient is nonzero for every sufficiently small nonzero epsilon,
because `kappa!=0`. These directions are independent. In fact

\[
E'(\varepsilon)=40c\varepsilon+O(\varepsilon^3),\qquad
v_\varepsilon^{\mathsf T}Jw_\varepsilon=E'(\varepsilon)/(4c),
\]

so their plane is symplectic, with nonzero limiting pairing after dividing
`v_epsilon` by epsilon. It follows that the two-dimensional neutral cluster
is a nontrivial Jordan block at multiplier one. The limiting generator has
exactly two zero eigenvalues; the eight isolated slow directions and the
eight fast directions have already been accounted for, so no additional
unit multiplier is hidden in the centered space.

Restoring the two origins gives the complete count:

| Full-state contribution | Algebraic dimension | Return behavior |
| --- | --- | --- |
| Common origins | `2` | Two independent constant unit eigenvectors |
| Autonomous phase and branch amplitude | `2` | One nontrivial size-two Jordan block at one |
| Slow transverse directions | `8` | Simple unit-circle multipliers |
| Collective directions | `8` | Unit-circle multipliers, semisimple even if repeated |

Thus multiplier one has algebraic multiplicity four and geometric
multiplicity three in the full twenty-coordinate return. No multiplier has
modulus greater than one, but the Jordan block allows a linearly accumulating
phase displacement in raw variational coordinates. Spectral stability does
not erase that shear or establish boundedness of every variation.

There is nevertheless a precise bounded transverse linear statement at each
fixed sufficiently small nonzero epsilon. The two-origin plane and the
phase/family plane form an invariant four-dimensional symplectic subspace.
Its symplectic complement is invariant and has dimension sixteen. On this
complement all return multipliers have unit modulus and are semisimple,
so the powers of the restricted return are bounded. The fundamental matrix
is bounded on one compact period; transport of this complement along the
orbit therefore gives bounded transverse linear evolution for that fixed
epsilon. No bound uniform as epsilon tends to zero is supplied, and the
statement excludes the neutral family shear. It is neither damping nor
nonlinear orbital stability.

### 12.7. The correct half return and the original clock

Let S exchange the two fine members of every pair, acting on both q and y.
It is a constant symplectic involution. Section 11 gives
`z(s+pi)=S*z(s)`, and the full variational coefficient obeys
`A(s+pi)=S*A(s)*S`. The raw matrix `Phi(pi)` maps tangent vectors to
the exchanged base point. The return to the original member convention is

\[
R_\varepsilon=S\Phi_\varepsilon(\pi),\qquad
M_\varepsilon=R_\varepsilon^2.
\]

This identity follows by composing the two actual half intervals and their
exchange conjugacy; it is not a choice of a numerical square root. Every
eigenvalue of the corrected half return consequently has unit modulus.
Its nonneutral directions are also semisimple, since squaring a nonzero
eigenvalue preserves the size of a Jordan block. On the phase/family plane,

\[
(R_\varepsilon-I)w_\varepsilon
           =-\tfrac12P_\tau'(\varepsilon)v_\varepsilon,
\qquad R_\varepsilon v_\varepsilon=v_\varepsilon.
\]

The two origin directions are unchanged. A raw half-step propagator is not
the labeled return and does not by itself supply this conclusion.

The full original-time period is still
`P_t=pi*P_tau/sqrt(c)`. Changing from t to tau or s consistently in every
row does not change the return multipliers. Exponential rates per unit time,
when discussed, retain this conversion. No external pulse, altered capacity,
new observation clock or physical time unit has been introduced.

### 12.8. The proved robustness and its remaining boundary

The exact branch has now passed a full twenty-coordinate transverse spectral
test in a sufficiently small amplitude regime. Its stability was not inferred
from the imposed traveling symmetry: arbitrary internal and collective
variations were retained, their actual periodic coupling supplied the slow
return coefficient, and the full Hamiltonian return controlled the omitted
orders. The construction exhibits an exact coordinated local-pulse family
with no exponentially growing linear modes in that regime.

This conclusion remains separate from nonlinear orbital stability, a
quantitative neighborhood, a long-time synchronization error bound,
acquisition or attraction. In particular the cubic action is not thereby
promoted to a new exact first integral of the full system. The spectral
argument supplies neither a numerically admitted amplitude radius nor a
generic preparation selecting this pulse. Other amplitudes and other laws
retain their own obligations, and no empirical identification follows.

The scope in `artifacts/phase-advance-transverse-spectrum-20261027/scope.json`
selects seven [finite control groups](../../tests/physics/test_phase_advance_transverse_spectrum.py):
the full Jacobian and origins; complete limiting spectrum; the actual
variational bridge; slow spectral separation; definite collective clusters;
the corrected half return and clock; and the energy/period Jordan count.
Frozen receipts and the research program own their evaluation status.
No integration, multiplier scan or prior scientific producer is replayed.

<a id="phase-advance-nonlinear-retention"></a>
## 13. Nonlinear retention on a finite pulse interaction window

The exact branch admits nearby complete states whose local activity stays
coordinated on `0 <= tau <= T_star/epsilon^2`. The sufficient initial
radius below is `eta*epsilon`, with eta positive and independent of epsilon
for each fixed finite `T_star>0`. Both the amplitude cutoff and the constants
are existential; this is not a numerically admitted neighborhood.

The proof uses the actual nonlinear vector field. A local change of
coordinates removes its quadratic Taylor term, and estimates the remaining
exact field. All internal and collective directions remain available. No
collective preparation is tied to the branch's particular mean response.
The comparison uses the same branch amplitude and one common temporal shift
chosen at the initial time. It does not refit individual pulses or impose
an additional conserved action.

### 13.1. Full-state comparison and the actual quadratic term

Retain the ten-node support, twenty unit edges and complete conservative
law with `e=0`, `w=beta=nu=1`. Fix the two conserved common origins as in
Section 11. In its acute chart write

\[
z=(q,x),\qquad q=\theta-\theta_*,\qquad
\mathbf1^{\mathsf T}q=\mathbf1^{\mathsf T}x=0,\qquad
\|z\|_\infty=\max(\|q\|_\infty,\|x\|_\infty).
\]

Here x keeps the original fine form units; it is not normalized by
`sqrt(c)`. As before `alpha=2*pi/5`, `c=cos(alpha)>0` and
`tau=sqrt(c)*t/pi`. The original twenty rows become

\[
q'=\frac{L_f x}{4\sqrt c},\qquad
x_i'=\frac1{4\sqrt c}\sum_{j\sim i}
       \sin(g^*_{ij}+q_j-q_i),\qquad
 g^*_{ij}=\theta_{*,j}-\theta_{*,i}.
\]

Primes in this section denote tau derivatives. Trigonometric functions
retain the actual lifted edge differences; choosing equivalent increments
`+/-alpha` leaves these rows unchanged. The twist is critical, so the field
`F(z)` vanishes at zero. Its Taylor expansion is

\[
F(z)=Az+B(z,z)+O(\|z\|_\infty^3),\qquad
A=\begin{pmatrix}0&L_f/(4\sqrt c)\\-\sqrt c L_f/4&0\end{pmatrix},
\]

where the symmetric bilinear map has the explicit rows

\[
B_q(v,w)=0,\qquad
[B_x(v,w)]_i=-\frac1{8\sqrt c}
 \sum_{j\sim i}\sin(g^*_{ij})
 (v_{q,j}-v_{q,i})(w_{q,j}-w_{q,i}).
\]

Thus `B=D^2F(0)/2`; it is not the second derivative without its Taylor
factor. Opposite edge orientations cancel in the sum of its form rows.
Its inputs ignore common phase and form origins, and its output is centered.
The two constant origins consequently decouple exactly. The centered
space has eighteen dimensions, including all ten internal and eight
collective coordinates, not only the sixteen transverse directions.

For `epsilon>0` let `z_epsilon(tau)` denote the exact branch in these
coordinates and let `P_tau=2*pi/Omega(epsilon)`. Negative epsilon is the
already declared global member exchange and needs no different estimate.
Define the physical orbital distance at fixed amplitude by

\[
d_\varepsilon(z_0)=
 \min_{\sigma\in[0,P_\tau]}\|z_0-z_\varepsilon(\sigma)\|_\infty.
\]

The minimum exists by continuity and compactness. Choose one minimizer
`sigma_0` initially and thereafter compare with
`z_ref(tau)=z_epsilon(tau+sigma_0)`. This is one common temporal shift for
the complete state. The phase chart and constituent associations remain
fixed. The theorem bounds this fixed comparison, which is stronger than
minimizing the orbital distance again at every time.

### 13.2. There is no quadratic homological resonance

On the centered space A is diagonalizable with signed frequencies

\[
\mathcal W=\{\pm1,\ \pm\ell_1,\ \pm\ell_2\},\qquad
\ell_1=(5-\sqrt5)/4,\quad \ell_2=(5+\sqrt5)/4.
\]

Their complex multiplicities are five for each internal orientation and
two for each collective orientation. The degeneracies are retained.
For every ordered triple from this six-element set,

\[
\omega_j+\omega_k-\omega_l\ne0.
\]

Indeed write `4*omega=m+n*sqrt(5)`. Its rational part m is one of
`{-5,-4,4,5}`. The rational part of a sum of two such frequencies belongs
to `{-10,-9,-8,-1,0,1,8,9,10}`, a disjoint set. Irrationality of `sqrt(5)`
then excludes equality, independently of the irrational parts. This checks
all 216 quadratic denominators without a small-divisor limit.

This statement concerns sums of linear frequencies. It does not assert
absence of TNFR interaction or of the registered Resonance operator.
Cubic resonances, for example `omega+omega-omega`, remain possible.
The nonlinear law is not linearized by this observation. Including zero
origin frequencies indiscriminately would produce resonant denominators;
the exact decoupling above is what licenses the centered calculation.

Let `P_omega` be A's complex spectral projectors on the centered space,
so that `A*P_omega=i*omega*P_omega`. Define the symmetric bilinear map

\[
Q(v,w)=\sum_{\omega_l,\omega_j,\omega_k\in\mathcal W}
 \frac{P_{\omega_l}B(P_{\omega_j}v,P_{\omega_k}w)}
      {\mathrm i(\omega_j+\omega_k-\omega_l)}.
\]

The sum is finite and all denominators are nonzero. Exchanging j and k
shows symmetry. Complex conjugation sends every frequency and projector
to its negative-frequency partner and conjugates the denominator in the
same way. Hence Q takes real inputs to real outputs. Its values are
centered. Projection onto each eigenspace gives the exact homological
identity

\[
Q(Av,w)+Q(v,Aw)-AQ(v,w)=B(v,w),
\]

and in particular `2*Q(w,Aw)-A*Q(w,w)=B(w,w)`. Repeated eigenspaces
create no ambiguity: their whole projectors, not selected eigenvectors,
appear in the formula. Extend Q by zero on common-origin inputs if using
the full twenty-coordinate notation.

### 13.3. A real local chart for the exact field

Put `Psi(w)=w+Q(w,w)` and let `Q_0` be the bilinear operator norm in
`||.||_infinity`. Choose a positive radius `r_N` with
`Q_0*r_N<=1/4`. On its closed convex ball,

\[
\|D\Psi\|\le\tfrac32,\qquad
\|(D\Psi)^{-1}\|\le2,\qquad
\tfrac12\|w-v\|_\infty\le
 \|\Psi(w)-\Psi(v)\|_\infty\le\tfrac32\|w-v\|_\infty.
\]

These follow from `D Psi=I+2*Q(w,.)` and the corresponding Neumann
bound. In particular the chart is injective. Its image contains the
physical ball of radius `r_N/2`: for such z the map
`w -> z-Q(w,w)` maps the closed `r_N` ball into itself and has contraction
constant at most one half. This also makes initial-state admission to the
chart explicit. The inverse is analytic on the image's interior.

The exact conjugated vector field is

\[
g(w)=[D\Psi(w)]^{-1}F(\Psi(w)).
\]

Expanding its quadratic coefficient gives
`A*Q(w,w)+B(w,w)-2*Q(w,Aw)=0`. Thus

\[
g(0)=0,\qquad Dg(0)=A,\qquad D^2g(0)=0,
\qquad g(w)=Aw+O(\|w\|_\infty^3).
\]

The identity defining g, not its truncated expansion, governs everything
below. Psi need not be canonical: no Hamiltonian claim in the new chart
is required. It is an exact local coordinate change of the supplied field,
with the same time and physical observations recovered through Psi. It
adds no clock, force, conserved action or physical degree of freedom.

Choose a finite bound `L_3>=0` for the operator norm of `D^3g` on the
closed chart ball. Such a bound exists because g is analytic on a slightly
larger neighborhood. From Section 11 and the local inverse there are
`M>=1` and a positive amplitude cutoff for which, uniformly in tau and
the initial common shift,

\[
w_{\rm ref}=\Psi^{-1}(z_{\rm ref}),\qquad
 \|w_{\rm ref}\|_\infty\le M\varepsilon.
\]

Shrink the amplitude cutoff so that `epsilon<=r_N/(3*M+2)` and the exact
branch has its previously proved activity and chart margins. This places
the reference, admitted initial states below and the entire bootstrap
segment strictly inside the chart. All these choices precede selection
of an individual perturbation; they are not fitted to an observed response.

### 13.4. A uniform propagator on the interaction window

An elementary bound avoids inferring uniform estimates from the locations
of Floquet multipliers. In the original fine form units,

\[
A=S A_c S^{-1},\qquad
S=\operatorname{diag}(I_{10},\sqrt c I_{10}),\qquad
A_c=\begin{pmatrix}0&L_f/4\\-L_f/4&0\end{pmatrix}.
\]

The real matrix `A_c` is skew-symmetric. Its exponential is orthogonal,
including the two constant-origin directions. Equivalence of the Euclidean
and infinity norms in twenty coordinates therefore gives the uniform bound

\[
\|e^{A\tau}\|_\infty\le C_A:=\sqrt{20/c}
\qquad(\tau\in\mathbb R).
\]

Integrating the third-derivative bound and using `D^2g(0)=0` gives

\[
\|Dg(w_{\rm ref}(\tau))-A\|_\infty
 \le\frac{L_3}{2}\|w_{\rm ref}(\tau)\|_\infty^2
 \le\frac{L_3M^2}{2}\varepsilon^2.
\]

Let `U(tau,s)` be the exact variational propagator of g along this
reference. Duhamel's formula around `exp(A*(tau-s))` and scalar Gronwall
now imply

\[
\|U(\tau,s)\|_\infty
 \le C_A\exp\!\left(\frac{C_A L_3M^2}{2}
                    \varepsilon^2(\tau-s)\right).
\]

Consequently, for every fixed finite `T_star>0`,

\[
\boxed{\quad
\|U(\tau,s)\|_\infty\le
 K:=C_A\exp(C_A L_3M^2T_*/2),\qquad
 0\le\tau-s\le T_*/\varepsilon^2.
\quad}
\]

K is independent of epsilon and the common initial phase. No transverse
projection has been applied: internal, free collective, autonomous phase
and branch-amplitude variations are retained. The neutral shear from
Section 12 is compatible with this bound on the specified finite window.
The direct bound requires neither a singular choice of Floquet eigenvectors
nor deletion of a homogeneous collective response.

This is not a uniform all-time estimate. In particular the same finite-
window reasoning can tolerate growth with rate of order epsilon squared.
Its role here is to control nonlinear errors over an interaction window;
it does not independently distinguish the traveling pattern from another
pattern with a slow instability.

### 13.5. A sufficient radius proportional to the pulse amplitude

For the exact perturbation in the new chart set
`e(tau)=w(tau)-w_ref(tau)` and `a_0=||e(0)||_infinity`. Taylor's formula
along the segment gives the actual nonlinear remainder

\[
\begin{aligned}
\mathcal R(\tau,e)&=g(w_{\rm ref}+e)-g(w_{\rm ref})-Dg(w_{\rm ref})e,\\
\|\mathcal R(\tau,e)\|_\infty
 &\le\frac{L_3}{2}(M\varepsilon+\|e\|_\infty)\|e\|_\infty^2\\
 &\le C(\varepsilon+\|e\|_\infty)\|e\|_\infty^2,
 \qquad C:=\max(1,L_3M/2).
\end{aligned}
\]

Here `M>=1` and the chart segment condition were used. This estimate is
for the exact g, including all higher terms, rather than the cubic
truncation alone. Choose, for this fixed window,

\[
0<\eta\le\eta_*:=
\min\left\{\frac1{8K},\frac1{16K^2CT_*},
                  \frac{\sqrt c}{24K}\right\},\qquad
 d_0:=d_\varepsilon(z_0)\le\eta\varepsilon.
\]

The amplitude cutoff can be reduced further independently of the individual
initial state. Since `eta<=1`, `||z_ref(0)||<=3*M*epsilon/2` and the
chosen cutoff put `z_0` in the physical `r_N/2` ball. Its inverse exists
and `a_0<=2*d_0`. During a bootstrap interval with `||e||<=epsilon`,
Duhamel and the preceding bounds give

\[
\|e(\tau)\|_\infty\le K a_0+
 2KC\varepsilon\int_0^\tau\|e(s)\|_\infty^2\,ds.
\]

The scalar comparison solution is

\[
\|e(\tau)\|_\infty\le
 \frac{K a_0}{1-2K^2C\varepsilon a_0\tau}
 \quad\text{while the denominator is positive}.
\]

For `0<=tau<=T_star/epsilon^2`, the subtracted denominator term is at
most `4*K^2*C*eta*T_star<=1/4`. Hence

\[
\|e(\tau)\|_\infty\le\tfrac43K a_0
 <2K a_0\le4K d_0\le\varepsilon/2.
\]

For `d_0=0` the solutions coincide by uniqueness. Otherwise the strict
bootstrap improvement and the interior chart margin exclude a first exit
before the horizon. The physical sine field is smooth and globally
Lipschitz in its lifted coordinates, so no finite existence obstruction
intervenes. Transforming back yields the promised full-state result:

\[
\boxed{\quad
\|z(\tau)-z_\varepsilon(\tau+\sigma_0)\|_\infty
 \le6K d_0,\qquad 0\le\tau\le T_*/\varepsilon^2.
\quad}
\]

The factor six is a conservative consequence of the chart bounds; it is
not claimed sharp. A family-amplitude change and independent collective
form or phase errors are permitted components of the initial perturbation.
Collective errors may be of order `eta*epsilon`, even though the reference's
collective excursion starts at order epsilon squared. Their actual response
is controlled by the full equation, rather than set to its prepared value.
There is no energy rematching or time-dependent phase adjustment in this
argument. It proves a sufficient scale, not an optimal perturbation radius.

### 13.6. Positive activity and retained local readings

Write `r_i=sqrt(delta_i^2+u_i^2)` for the physical internal radius with
`beta=1`. Section 11 gives `r_i,ref>=sqrt(c/2)*epsilon`. Each pair's delta
and u is a half-difference of fine coordinates. A fine infinity error R
therefore changes the internal two-vector by at most `sqrt(2)*R`.
Using `R=6*K*d_0`, put

\[
\rho=\frac{12K d_0}{\sqrt c\,\varepsilon}
 \le\frac{12K\eta}{\sqrt c}\le\tfrac12.
\]

It follows throughout the entire window that

\[
r_i\ge\tfrac12\sqrt{c/2}\,\varepsilon,
\qquad \boxed{\ r_i^2\ge c\varepsilon^2/8>0\ }.
\]

This uniform lower bound is stronger than merely avoiding a tip at finite
time. Admit the initial states to one fixed `U_active` from
[the active-family owner](SINE_REPLICA_PULSE.md#sine-replica-joint-persistence),
with its form-mean interval containing the chosen common form. Uniform
`O(epsilon)` proximity to the critical twist places all such states inside
its radius and strict energy ceiling for sufficiently small epsilon;
the displayed radius estimate already excludes every initial tip.
That family's exact conservation and invariance preserve its acute geometry,
winding and positive local circulation bounds. No new family-entry mechanism
or admission from outside this neighborhood is asserted.

Define the labeled local reading by
`chi_i=-arg(delta_i+i*u_i)`, as in Section 9. Choose its continuous lift
once so that the initial difference from the reference lift is the small
angle between their internal vectors. A vector perturbed by at most a
fraction rho of a nonzero reference length, with `rho<1`, lies in a disk
not containing zero. The largest angular deviation of that disk is
`arcsin(rho)`. Continuous lifting preserves this same small branch of the
difference over the whole interval. Thus, with `b=arcsin(rho)`,

\[
|\chi_i(\tau)-\chi_{i,\rm ref}(\tau)|\le b,\qquad
|[\chi_i-\chi_j]-[\chi_{i,\rm ref}-\chi_{j,\rm ref}]|\le2b.
\]

For any subinterval of length h, the corresponding angular advance differs
from the reference advance by at most `2*b`. The mean angular rate error is
therefore at most `2*b/h`; a relative-pair mean rate error is at most
`4*b/h`. Over the complete window the individual bound is
`2*b*epsilon^2/T_star`. These compare the actual reference advances;
noninteger-period endpoints need not give its exact cycle-average rate.

These are `O(eta)` pointwise bounds independent of epsilon, with the reference's
actual time-varying relative readings retained. The projective observation
`exp(2*i*chi_i)` has chord error at most `2*b`, and its relative-pair
counterpart has chord error at most `4*b`. The association of pairs is
fixed. Constant differences of auxiliary leading phases are not substituted
for these physical readings, nor are instantaneous local rates asserted
equal. The reference's bounded relative timing pattern is maintained to
the stated tolerance on the window, without a separately fitted local clock.

### 13.7. Tick observations and the supplied original time

For the chosen active family let `D=r/sqrt(2)` and
`c_r=cos(alpha+sqrt(2)*r)>0`, using its fixed radius r. The already proved
circulation estimate in original time becomes

\[
0<m_\tau:=\frac{c_r\operatorname{sinc}(D)}{\sqrt c}
 \le\frac{d\chi_i}{d\tau}\le\frac1{\sqrt c}.
\]

Fix tick levels in the declared continuous reading, before comparing
crossings. Spacing `2*pi` records labeled turns. Spacing `pi` records
turns of `exp(2*i*chi_i)`; it does not by itself certify a return of the
unordered internal amplitude or of the full state.

Let `H=T_star/epsilon^2`. A reference tick level l satisfying

\[
\chi_{i,\rm ref}(0)+b<l<\chi_{i,\rm ref}(H)-b
\]

is attained uniquely by both trajectories inside the observation window.
At the reference crossing the actual angle differs by at most b; the
positive lower speed then gives

\[
|\tau_{i,l}-\tau_{i,l}^{\rm ref}|\le b/m_\tau.
\]

More generally the same bound applies to a level in both interior attained
ranges. Endpoint levels require their own margin. A continuous lift does
not make integer endpoint counts identical: a nearby trajectory can cross
a tick slightly earlier. With a fixed mark and spacing `pi` or `2*pi`,
the floor counters at any one time differ by at most one here, since
`b<=pi/6`. Exact agreement follows only when the reference reading is
farther than b from every tick boundary. Counts accumulated between two
endpoints retain both endpoint uncertainties.

The original time is still `t=pi*tau/sqrt(c)`. The full retention horizon
is `pi*T_star/(sqrt(c)*epsilon^2)`, and the tick bound in that clock is
`pi*b/(c_r*sinc(D))`. These are predictions under the supplied model clock,
not a laboratory calibration or a derivation of the existence of time.

### 13.8. What finite retention establishes

For each fixed finite interaction duration there is a positive relative
initial radius eta and a sufficiently small nonzero branch-amplitude regime
in which complete nearby states maintain the exact family's local timing
to the proved tolerance. The neighborhood includes free collective states
and deviations from the traveling profile symmetry. Its physical radius
scales as epsilon, while the admitted duration scales as epsilon to the
minus two. The proof uses the actual nonlinear law and observations.

The constants depend on the window and chart bounds. They have not been
numerically enclosed, and may become restrictive as the desired duration
grows. This is not all-time nonlinear orbital stability, attraction, generic
acquisition, amplitude selection or empirical validation. Finite retention
can coexist with slow growth and therefore does not independently select
this pattern over the unstable common pulse. Nor does an angle/tick bound
supply a newly emergent ontological time parameter. Those distinctions
remain necessary even though the reference itself exists exactly and its
transverse spectrum has the separate result of Section 12.

The declaration in `artifacts/phase-advance-nonlinear-retention-20261027/scope.json`
selects eight [mathematical control groups](../../tests/physics/test_phase_advance_nonlinear_retention.py):
the actual quadratic jet; centered spectral projectors; all homological
denominators and the real mean-preserving map; the local chart and cubic
remainder; the uniform finite-window propagator; the nonlinear bootstrap;
full perturbations with one common phase and active readings; and tick
margins with original-time conversion. Frozen receipts and the research
program own evaluation status. No integration or historical producer replay
is part of this argument.


<a id="free-collective-pulse-interaction"></a>
## 14. Free collective oscillations change the leading timing interaction

The small neighborhood in Section 13 admits collective phase and form
coordinates of order epsilon. They cannot be replaced by the particular
collective response of Section 10, whose physical size is order epsilon
squared. This section retains those independent oscillations and derives
the complete first resonant law on all eighteen centered coordinates.
Its separate frequency-band actions are conserved, but local timing can
still drift: a standing collective wave gives a nonuniform internal
frequency shift. Traveling collective waves supply a conditional case in
which the leading internal phase-advance profile remains coordinated.
Neither conclusion identifies a material constituent or a physical clock.

Keep the complete conservative doubled-C5 law, ten constituents, twenty
unit edges, `e=0`, `w=beta=nu=1`, the fixed constituent associations and the
two conserved common origins. Put `alpha=2*pi/5`, `c=cos(alpha)>0`,
`a=tan(alpha)` and `tau=sqrt(c)*t/pi` as before. In this section the fine
normalized form coordinate is `y=x/sqrt(c)`. It is distinct from pressure
and from the unnormalized x retained in Section 13. No evolution row,
capacity, support or time law is changed.

### 14.1. A canonical calculation with all repeated modes retained

Use the exact centered symplectic space of Section 12, with state `(q,y)`
and Poisson bracket `{F,G}=grad(F)^T J grad(G)`. Up to its irrelevant
constant, the actual Hamiltonian has the expansion

\[
H_f=H_2+H_3+H_4+O(\|(q,y)\|^5),\qquad
H_2=\frac18(q^{\mathsf T}L_fq+y^{\mathsf T}L_fy),
\]

\[
H_3=-\frac1{24c}\sum_{e=(i,j)}\sin(g_e^*)\,(q_j-q_i)^3,
\qquad
H_4=-\frac1{96}\sum_{e=(i,j)}(q_j-q_i)^4.
\]

Choose either orientation for each edge and use the same orientation in
`g_e^*=theta_{*,j}-theta_{*,i}` and `q_j-q_i`. The cubic product is
orientation-independent. All twenty actual edges occur once. These
coefficients follow from the sine/cosine Taylor series of `V/(4*c)`;
the linear term vanishes because the reference twist is critical.
The kinetic Hamiltonian has no nonlinear term.

An explicit real orthonormal spatial basis removes coordinate ambiguity.
The five internal vectors are `(e_i)_(j,sigma)=sigma*1_(i=j)/sqrt(2)`.
On the base cycle define, for `k=1,2`,

\[
v_{k,c}(i)=\sqrt{2/5}\cos(k\alpha i),\qquad
v_{k,s}(i)=\sqrt{2/5}\sin(k\alpha i).
\]

The four collective fine vectors duplicate `v_(k,r)/sqrt(2)` on both
members of each pair. Together these nine vectors span the centered
fine scalar space. Write `Q_s=e_s^T q`, `P_s=e_s^T y`, and introduce

\[
b_s=(Q_s-\mathrm iP_s)/\sqrt2,\qquad
\{b_s,\overline b_t\}=\mathrm i\delta_{st},\qquad
H_2=\sum_s\omega_s|b_s|^2.
\]

The frequencies are `1` for each of the five internal coordinates and
`ell_k=(5+(-1)^k*sqrt(5))/4` for each of the two real spatial copies in
collective band k. Thus `b_s'=i*omega_s*b_s`. No repeated eigenspace is
replaced by a single oscillator. In physical pair coordinates,

\[
b_i=\delta_i-\mathrm i u_i/\sqrt c,\qquad
C_{k,i}=\sum_{r=c,s}v_{k,r}(i)b_{k,r},\qquad
\Theta_i-i\alpha=\operatorname{Re}(C_{1,i}+C_{2,i}),\quad
X_i/\sqrt c=-\operatorname{Im}(C_{1,i}+C_{2,i}).
\]

The last two equalities concern the original coordinates before the
nonlinear change below. Their linear parts retain both independent
collective initial coordinates. Each C_k is a complex vector in the
corresponding real spatial eigenspace, complexified; its two coefficients
need not be conjugates.

### 14.2. A finite formula for every surviving interaction coefficient

The quadratic chart of Section 13 was not assumed canonical. Here use
Hamiltonian flows to give a canonical alternative and retain its exact
local reconstruction. Expand the displayed edge polynomials in the nine
complex coordinates:

\[
H_3=\sum_{|\mu|+|\nu|=3}h_{\mu\nu}b^\mu\overline b^\nu,
\qquad
H_4=\sum_{|\mu|+|\nu|=4}d_{\mu\nu}b^\mu\overline b^\nu,
\qquad
\Delta_{\mu\nu}=\sum_s\omega_s(\mu_s-\nu_s).
\]

These are ordinary polynomial coefficients, including their multinomial
factors. To obtain them directly, substitute
`q_j-q_i=sum_s(e_s(j)-e_s(i))*(b_s+bar(b_s))/sqrt(2)` in H3 and H4.
This finite prescription specifies the tensor using the actual support
and basis; there are no fitted or unspecified interaction coefficients.
The absence of quadratic vector-field resonance in Section 13 implies
`Delta_(mu,nu)!=0` for every cubic monomial. Define the real polynomial

\[
S_3=\sum_{|\mu|+|\nu|=3}
       \frac{h_{\mu\nu}}{\mathrm i\Delta_{\mu\nu}}
                         b^\mu\overline b^\nu.
\]

Since `{H2,b^mu*bar(b)^nu}=-i*Delta_(mu,nu)*b^mu*bar(b)^nu`,
`{H2,S3}=-H3`. Let `Phi_S^1` mean the time-one local Hamiltonian flow
generated by S in these calculation coordinates. Taylor expansion of
`H_f composed with Phi_(S3)^1` gives no cubic Hamiltonian and has quartic
part

\[
K_4=H_4+\tfrac12\{H_3,S_3\}.
\]

The sign and one-half factor follow from
`H composed with Phi_S^1=H+{H,S}+{{H,S},S}/2+...`.
For an entirely algebraic expression for the bracket term, use

\[
\frac12\{H_3,S_3\}
=\frac12\sum_{s,\mu,\nu,\rho,\sigma}
 \frac{h_{\mu\nu}h_{\rho\sigma}}{\Delta_{\rho\sigma}}
 (\mu_s\sigma_s-\nu_s\rho_s)
 b^{\mu+\rho-e_s}\overline b^{\nu+\sigma-e_s}.
\]

Both input multiindices have total degree three. Terms with zero bracket
factor are zero; only nonnegative output exponents are retained. This
formula includes multiplicities and every possible internal/collective
coupling. It can be evaluated by finite coefficient arithmetic without
integrating a trajectory.

Let `Res` retain precisely the monomials whose frequency difference is
zero, and set

\[
\boxed{Z_4=\operatorname{Res}K_4.}
\]

For every nonresonant quartic coefficient k_(mu,nu) of K4 define
`S4_(mu,nu)=k_(mu,nu)/(i*Delta_(mu,nu))`, and set its resonant coefficients
to zero. Its finite nonzero denominators need no infinite small-divisor
argument. The real canonical map

\[
\Phi=\Phi_{S_3}^1\circ\Phi_{S_4}^1
\]

is analytic and invertible on a sufficiently small neighborhood, with
`Phi(w)=w+O(||w||^2)`. The exact transformed Hamiltonian satisfies

\[
H_f\circ\Phi=H_2+Z_4+R_5,\qquad
R_5=O(\|w\|^5),\qquad \nabla R_5=O(\|w\|^4).
\]

The canonical map changes coordinates, not the physical evolution law.
The remaining term is retained for the comparison below. The first
resonant interaction, for `b=epsilon*exp(i*omega*tau)*B(T)` and
`T=epsilon^2*tau`, is the complete nine-complex-coordinate system

\[
\boxed{\quad \frac{dB_s}{dT}
       =\mathrm i\frac{\partial Z_4}{\partial\overline B_s}.
       \quad}
\]

Every occurrence of Z4 here means the finite polynomial above, including
the purely collective terms. This is a derived leading model, not a new
runtime law installed in the engine.

### 14.3. The resonance rule preserves three band actions

Suppose a monomial has net powers `n0,n1,n2` in the three positive
frequency bands. A zero frequency difference requires

\[
4n_0+5n_1+5n_2=0,\qquad -n_1+n_2=0.
\]

Its integer solutions are `(-5m,2m,2m)`. A nonzero solution has total
absolute net degree at least nine. A quartic monomial therefore balances
the unbarred and barred powers separately in every band. This is stronger
than conservation of the weighted quadratic energy alone.
Write A for the five internal B coordinates and C_k for the collective
vectors reconstructed from their two B coefficients. The quartic law is
invariant under three independent band rotations and conserves

\[
I_0=\sum_i|A_i|^2,\qquad I_k=\|C_k\|_2^2\ (k=1,2),
\qquad Z_4.
\]

In particular `sum_s|B_s|^2=I0+I1+I2` is constant. The polynomial leading
system is globally defined on each such compact action sphere. These are
invariants of the resonant model; they are not asserted to be exact
invariants of the full sine flow. The degree-nine arithmetic relation
alone does not assert that its corresponding coefficient is nonzero or
activate a higher-order study.

Each independent exchange of the two members in pair i sends `A_i` to
`-A_i` and leaves C unchanged. The edge Hamiltonian, the unique selected
nonresonant generators and Z4 preserve that symmetry. A mixed quartic
term contains one internal unbarred and one internal barred factor;
the symmetry forces both factors to belong to the same pair. Hence the
complete polynomial decomposes as

\[
\boxed{\quad
Z_4(A,C_1,C_2)=-\mathcal H(A)
 -\sum_{i,k}|A_i|^2\,\mathcal D_{k,i}(C_k)
 +Z_C(C_1,C_2),\quad}
\]

where H is precisely the quartic Hamiltonian in Section 10,
`Z_C=Z4(0,C1,C2)` is explicitly determined by Section 14.2, and each
D_(k,i) is a real Hermitian quadratic form specified next. There is no
internal term mixing C1 and C2 at degree four: that would violate the
separate band balances. Purely collective cross-band terms are retained
in Z_C. Differentiating this full expression supplies their back reaction;
C is not a prescribed external forcing.

The sign of the internal term matters. Here `{A,bar(A)}=i` and the fast
motion is `exp(i*tau)`, whereas Section 10 writes
`i*A_T=partial(H)/partial(bar(A))`. These conventions give
`Z4(A,0,0)=-H(A)`, with no additional amplitude rescaling. Setting the
canonical collective amplitudes to zero removes free collective motion
at order epsilon. Its order-epsilon-squared reconstruction still includes
the static and second-harmonic particular response of Section 10.2.
Consequently the zero-collective limit retains that response rather than
freezing the original mean coordinates. The homological equation is
invertible at this order, so the bounded particular correction and its
resonant coefficient agree with the earlier elimination.

### 14.4. Explicit collective detuning of the internal pulses

Use the base operators L and D from Section 10. For a complex vector C in
collective band k, put `ell=ell_k` and

\[
J_i(C)=|C_{i+1}-C_i|^2-|C_{i-1}-C_i|^2.
\]

It is centered: both summed squared differences contain the same edges.
The exact mixed coefficient is

\[
\boxed{\quad
\mathcal D_{k,i}(C)=\frac1{16}\left[
 \sum_{j\sim i}|C_j-C_i|^2
 -a^2(DL^\dagger J(C))_i
 +\frac{a^2}{4-\ell^2}|(DC)_i|^2\right].\quad}
\]

Here `L^dagger` is the centered inverse, and `0<ell<2` keeps the sideband
denominator nonzero. This is real and homogeneous in `C,bar(C)`. To
verify the coefficient directly from the complete pair rows, let
`h1=Re(C*exp(i*ell*tau))` be the leading collective phase displacement.
The zero-frequency part of its quadratic collective response is

\[
h_{20}=-\frac a4L^\dagger J(C).
\]

Indeed the pure collective expansion is
`h''+L^2*h/4=-(a/8)*L*((h_(i+1)-h_i)^2-(h_(i-1)-h_i)^2)+O(||h||^3)`.
The internal row linear in a probe internal amplitude is

\[
\delta_i''+\delta_i=
 \left[\frac a2(Dh)_i+
       \frac14\sum_{j\sim i}(h_j-h_i)^2\right]\delta_i
 +\text{terms of other retained degrees}.
\]

The mean correction contributes `(a/4)*D*h20` to `i*A_T/A`.
The direct quadratic collective term contributes
`sum_neighbors|C_j-C_i|^2/16`. The first-order parametric coefficient is
`Re(P_i*exp(i*ell*tau))`, where `P_i=(a/2)*(DC)_i`. Its two internal
sidebands have denominators `1-(1+ell)^2` and `1-(1-ell)^2`; their sum is
`2/(4-ell^2)`. Substitution back into the same row contributes
`|P_i|^2/[4*(4-ell^2)]` to `i*A_T/A`. The three terms give the displayed D.
This extraction retains both orientations and the static collective
response. Terms at frequency `2*ell` cannot resonate with the internal
frequency one, since `ell!=1`. Cross terms from distinct collective
bands are excluded by the already proved full resonance rule.

The internal equation can now be read without expanding Z_C:

\[
\mathrm i A_{i,T}=
 \frac1{16}A_i(N|A|^2)_i+
 \frac1{32}\overline{A_i}[M(A^2)]_i+
 A_i\sum_k\mathcal D_{k,i}(C_k).
\]

The added coefficient is real. It changes the internal phase rate directly,
but cancels from `d|A_i|^2/dT`. Where `A_i!=0`, its direct contribution is
`d arg(A_i)/dT=-sum_k D_(k,i)`. Subsequent phase changes can alter the
phase-sensitive internal exchange already present in the M term. Thus
absence of direct band-action transfer does not imply unchanged local
amplitudes or coordinated timing. The collective equations contain the
reciprocal derivative of `-sum_i|A_i|^2*D_(k,i)` as well as `Z_C`.

### 14.5. A standing-wave drift and a traveling-wave exception

The following two preparations concern the leading model and have a
specified bridge to the full flow in Section 14.6. For one collective
band let `theta=k*alpha`, `ell=1-cos(theta)`, and choose the standing
complex amplitude `C_i=cos(theta*i)`. Substitution gives

\[
\mathcal D_{k,i}=d_0-d_1\cos(2\theta i),\qquad
d_0=\frac\ell8\left[1+\frac{a^2}{2+\ell}\right],
\]

\[
d_1=\frac\ell8\left[
 \cos\theta(1+a^2)+\frac{a^2}{2+\ell}\right].
\]

For example, the quadratic static correction is
`h20_i=-a*ell*sin(theta)*sin(2*theta*i)/(4*ell_(2k))`, with the base
wave number taken modulo five. Combining it with the two sidebands
recovers the same formula; dropping it would change d1.

Select band one, set the other collective band to zero, and prepare
`A_i(0)=exp(i*alpha*i)`. The isolated internal model initially gives the
same phase rate `-kappa` at every pair. Here `cos(theta)=c>0`, so `d1>0`.
For continuous internal envelope phases,

\[
\left.\frac d{dT}(\arg A_1-\arg A_0)\right|_{T=0}
 =-d_1(1-\cos(2\alpha))<0.
\]

This is a nonzero slow relative-phase drift under independent collective
initialization, despite conservation of all three band actions. Smoothness
keeps the internal amplitudes nonzero for a sufficiently short fixed slow
interval, so the phase conclusion is not obtained by reading a zero pulse.
It disproves invariance of the unchanged internal traveling profile for
arbitrary free collective data in the leading law. It does not assert
loss of activity, unbounded drift or instability of the exact branch.

For comparison, a single traveling collective amplitude
`C_i=exp(i*theta*i)` gives the uniform value

\[
\mathcal D_{k,i}=\frac\ell4
                  \left[1+\frac{a^2}{2+\ell}\right],
\]

independent of i. Multiplying C by a complex scalar multiplies this value
by its squared modulus. In this case the direct internal correction is
a common frequency shift.

There is also a closed conditional family of this type within the complete
leading law, rather than only an instantaneous cancellation. In the
normalized complex spatial basis `exp(+/-i*k*alpha*i)/sqrt(5)`, denote
the collective coefficients by `c_(k,+),c_(k,-)`. Separate band balance
and cyclic spatial covariance force Z_C to be a polynomial in the four
intensities `|c_(k,+/-)|^2`. A self-band quartic transferring opposite
spatial orientations has nonzero momentum `+/-2k` or `+/-4k` modulo five.
A cross-band quartic transferring both has momentum `+/-2+/-4`, also
nonzero modulo five. These exhaust the possible non-intensity quartics.
Thus Z_C contains their allowed intensity self/cross terms, not directional
amplitude transfer.

For uniform internal intensity `|A_i|^2=r^2`, summing the explicit D over
i gives

\[
\sum_i\mathcal D_{k,i}(C_k)
 =\frac{\ell_k}4\left[1+\frac{a^2}{2+\ell_k}\right]\|C_k\|_2^2.
\]

The D term with the centered inverse sums to zero. The other terms use
`C_k^*L C_k=2*ell_k*||C_k||^2` and
`||D C_k||^2=4*ell_k*(2-ell_k)*||C_k||^2`.
The reciprocal mixed collective row therefore also produces only a common
band frequency shift. Consequently the internal phase-advance profile
`A_i=r*exp(i*alpha*i+i*phi)` together with at most one traveling spatial
orientation in each collective band is invariant under the leading law.
All occupied amplitudes are constant and their phases have constant rates
set by the retained quartic polynomial. The internal relative envelope
phases stay fixed; the collective phases need not share their frequency.
This is a conditional multimode family in the derived model. It is neither
a proof of an exact corresponding family of the sine flow nor common
full-state periodicity or acquisition from generic preparation.

### 14.6. Uniform reconstruction on the finite interaction window

The canonical calculation supplies the needed comparison on a growing
window, which was not supplied by Section 10.7. Fix `T_star>0` and a
compact set of scaled centered initial data. For any sufficiently small
physical initial state in this set, choose the model initial coordinate
by the exact inverse map, divided by epsilon. More precisely, if U denotes
the fixed real/complex modal conversion of Section 14.1 and the physical
normalization is `(q,x)->(q,x/sqrt(c))`, set

\[
B(0)=\varepsilon^{-1}U\Phi^{-1}(q(0),x(0)/\sqrt c).
\]

Phi in this equation acts on the real centered canonical coordinates;
its complex formula and U are equivalent representations. This inverse
retains every admitted initial coordinate. It is not separate refitting
of pulses or setting the collective homogeneous response to zero. Allow
B(0) to depend on epsilon while remaining in the chosen compact set.

Let B(T) solve the complete resonant law. Its three actions give a uniform
bound `||B(T)||_2<=R` on every slow interval. In the exact transformed
flow write its interaction coordinate as
`B_ex(T)=epsilon^-1*exp(-A_c*tau)*w(tau)` in real coordinates, with the
same fixed modal conversion understood; A_c is the skew matrix of the
quadratic Hamiltonian. The exact equation is

\[
\frac{dB_{\rm ex}}{dT}
 =X_{Z_4}(B_{\rm ex})+\varepsilon r_\varepsilon(T,B_{\rm ex}).
\]

The radius R is chosen in the real modal norm; its fixed factor relative
to the complex norm is included in R and in the constants below. The linear
rotations are orthogonal in that real norm. Resonance
makes the cubic vector field equivariant under them, so it has no remaining
fast-time factor. The bound on `grad R5` supplies a uniform bound C for
`r_epsilon` on `||B_ex||<=R+1`, including its fast dependence on T.
Choose a Lipschitz bound L for `X_(Z4)` on the same ball. Matching the
initial inverse makes the initial difference zero. Gronwall yields

\[
\sup_{0\le T\le T_*}\|B_{\rm ex}(T)-B(T)\|
 \le\varepsilon C\frac{e^{LT_*}-1}{L},
\]

with quotient `T_star` when L is zero. For sufficiently small epsilon
the right side is less than one half; the usual first-exit argument keeps
the exact solution inside the selected ball and the canonical chart.
The exact original field is globally defined, and all transformations are
nonsingular on this smaller domain. The reconstruction

\[
(q^{\rm app},x^{\rm app}/\sqrt c)
 =\Phi U^{-1}\bigl(\varepsilon e^{\mathrm i\omega\tau}
                          B(\varepsilon^2\tau)\bigr)
\]

therefore satisfies, in the original fine-state maximum norm,

\[
\boxed{\quad
\sup_{0\le\tau\le T_*/\varepsilon^2}
 \|(q,x)_{\rm exact}-(q,x)_{\rm app}\|_\infty
 \le K_{T_*,R}\varepsilon^2.\quad}
\]

The bound includes the order-epsilon-squared reconstruction, not only the
leading oscillations. The finite-dimensional modal normalization, original
form conversion and local derivative bound on Phi are absorbed in K.
The constants and sufficiently small amplitude cutoff exist; no numerical
enclosure or practical amplitude range is claimed. The physical original
time window remains `0<=t<=pi*T_star/(sqrt(c)*epsilon^2)`.
The estimate is uniform on each fixed slow interval, not on unbounded T.

This argument uses the complete resonant vector field. Replacing the
collective coordinates by constants, their intensities alone, or their
particular order-epsilon-squared response would change that vector field
and does not inherit this estimate. The degree-four normal form is finite;
no infinite normal-form convergence or exactly conserved full band actions
is assumed.

### 14.7. Which local timing conclusions transfer

Suppose the selected leading solution also satisfies
`min_(i,0<=T<=T_star)|A_i(T)|>=r_*>0`. Conservation of I0 by itself does
not imply this individual lower bound. It holds for the conditional
traveling family and on a sufficiently short interval around the standing
wave witness. The full reconstruction then gives, uniformly on the window,

\[
\delta_i=\varepsilon\operatorname{Re}(A_i(T)e^{\mathrm i\tau})
                 +O(\varepsilon^2),\qquad
u_i=-\sqrt c\,\varepsilon\operatorname{Im}(A_i(T)e^{\mathrm i\tau})
                 +O(\varepsilon^2).
\]

The original internal radius is bounded below by a positive constant times
epsilon, so the local reading is available. With `sigma_i=tau+arg(A_i(T))`, define
the continuous reference reading

\[
\chi_i^{\rm lead}=\arg(\cos\sigma_i+\mathrm i\sqrt c\sin\sigma_i).
\]

Choose initial lifts consistently. Since the leading vector stays a fixed
multiple of epsilon from zero, the `O(epsilon^2)` state estimate yields
`chi_i-chi_i^lead=O(epsilon)` uniformly; relative readings have the sum of
the two corresponding bounds. This is an angular comparison with the
actual elliptical reading, not an identification of primitive phase with
the envelope phase. Its monotone circle reparameterization is the same
one already derived in Section 10.8. Constant envelope offsets alone do
not imply equal instantaneous local speeds.

The standing-wave example changes those envelope offsets at a nonzero
slow rate. The uniform full-state reconstruction transfers its finite
slow-interval effect through this observation map for sufficiently small
epsilon, with the stated angular error and nonzero-amplitude condition.
For sufficiently small epsilon these prepared states lie in the existing
`U_active`: centered radius and excess storage tend to zero, and the
internal lower bound keeps every pair off its tip. Its original angular
speed bound remains `chi_i'>=m_tau=c_r*sinc(D)/sqrt(c)>0` in tau.
The compact model trajectory and `r_*>0` also bound `d arg(A_i)/dT`;
therefore `sigma_i'=1+epsilon^2*d arg(A_i)/dT>0` for sufficiently small
epsilon, and the leading reading is strictly increasing. Let
`b_epsilon=O(epsilon)` bound the uniform reading error. For a fixed level
in both interior attained ranges, the two unique crossing times satisfy

\[
|\tau_{i,l}-\tau_{i,l}^{\rm lead}|\le b_\varepsilon/m_\tau.
\]

A sufficient admission is to place the level strictly between the leading
initial reading plus `b_epsilon` and the leading final reading minus that
margin. The original-time bound is `pi*b_epsilon/(sqrt(c)*m_tau)`.
Endpoint integer tick counts still require their margins; angular closeness
does not remove their ambiguity. The traveling conditional family instead
retains fixed internal envelope offsets while permitting independent
collective frequencies. Neither case produces a newly derived ontological
time parameter, an attracting global pulse or an empirical identification.

The declaration in `artifacts/free-collective-pulse-interaction-20261028/scope.json`
owns the finite control selection and resource budget. The research program
and frozen receipts own evaluation status. No trajectory, parameter scan
or historical producer replay is part of this derivation.


<a id="counterpropagating-pulse-robustness"></a>
## 15. A small counterpropagating component has a bounded linear timing response

The standing-wave example in Section 14 proves an initial nonuniform
frequency shift, not its subsequent evolution. Here the complete reciprocal
leading law closes that distinction near one joint traveling family.
For every sufficiently small fixed positive collective amplitude gamma,
the counterpropagating perturbation couples to four internal real directions
and forms a six-dimensional block with three distinct real frequencies.
Its new slow frequency is nonzero and proportional to gamma squared.
The resulting linear modulation is bounded. The full eighteen-dimensional
calculation also retains two neutral phase/action shears and the initially
empty collective band. There is no damping or inference of nonlinear
all-time stability.

Keep the law, support, canonical normalization, fixed common origins and
original clock of Section 14. Work first with its derived quartic model,
using `K=-Z4`, so that `i*B_T=partial(K)/partial(bar(B))` in orthonormal
complex coordinates. The two small quantities have different roles:
`gamma>0` is a collective-to-internal amplitude ratio in the leading model;
`epsilon>0` scales the complete physical state. Choose gamma first, then
choose epsilon sufficiently small for each specified finite slow window.
Gamma does not replace the held constitutive coefficient `beta=1`.

### 15.1. Reference family and the required collective coefficients

Use unnormalized spatial waves for the collective coordinates,

\[
C_{1,i}=u\,e^{\mathrm i\alpha i}+v\,e^{-\mathrm i\alpha i},\qquad
C_{2,i}=z_+e^{2\mathrm i\alpha i}+z_-e^{-2\mathrm i\alpha i}.
\]

Their brackets are `i/5` with their own conjugates; each wave has squared
spatial norm five. The internal coordinates A_i still have bracket i.
Put `ell=ell1`, `m=ell2`, and write the band-one detuning as

\[
\mathcal D_{1,i}
 =d_0(|u|^2+|v|^2)
   +h(u\overline v e^{2\mathrm i\alpha i}
                  +\overline u v e^{-2\mathrm i\alpha i}),
\]

\[
d_0=\frac\ell4\left(1+\frac{a^2}{2+\ell}\right)>0,\qquad
h=-\frac\ell4\left[c(1+a^2)+\frac{a^2}{2+\ell}\right]<0.
\]

This is the Hermitian coefficient already derived in Section 14.4;
substituting `u=v=1/2` recovers its standing-wave coefficient.
The purely collective part of K restricted to band one is

\[
K_C(u,v,0,0)=s(|u|^4+|v|^4)+t|u|^2|v|^2,
\]

\[
\boxed{\quad s=\frac{25(4+\sqrt5)}{44}>0,\qquad
 t=\frac{25}{4},\qquad
 d:=\frac{t-2s}{5}=\frac{5(3-2\sqrt5)}{44}<0.\quad}
\]

These coefficients come from the actual canonical polynomial, not an
added collective interaction. One short derivation makes the virtual
collective response explicit. On the pure collective subspace the original
Hamiltonian terms are

\[
H_3=-\frac a6\sum_i(h_{i+1}-h_i)^3,\qquad
H_4=-\frac1{24}\sum_i(h_{i+1}-h_i)^4.
\]

Let `zeta=exp(i*alpha)` and `d_k=zeta^k-1` for this calculation only.
The part of H3 with two low-band factors has the coefficient

\[
g=-\frac{5a}{16}d_1^2d_{-2}
  =-\frac{5\mathrm i a}{4}\ell\sin\alpha
\]

multiplying `(u+bar(v))^2*(z_-+bar(z_+))`, together with its conjugate.
There is no all-low cubic monomial. The high-high-low terms cannot give
four low factors in a single Poisson contraction. The direct quartic
coefficients of Z_C for `|u|^4` and `|u|^2|v|^2` are respectively
`-5*ell^2/16` and `-5*ell^2/4`. Contracting the high factors in
`{H3,S3}/2`, with their actual bracket `i/5`, adds respectively

\[
\frac{|g|^2}{5}
       \left(\frac1{2\ell-m}-\frac1{2\ell+m}\right),\qquad
-\frac{8|g|^2}{5m}.
\]

The denominators retain both high oscillator orientations; none is zero.
Using `m=2*ell*(2-ell)`, `sin(alpha)^2=ell*(2-ell)` and
`a^2=5+2*sqrt(5)` yields the displayed s and t after reversing the sign
from Z_C to K_C. In particular

\[
s=\frac{5\ell^2}{16}
 \left[1+a^2-\frac{a^2}{(\ell-1)(3-\ell)}\right],\qquad
t=\frac{5\ell^2}{4}(1+a^2).
\]

The cross-band coefficients needed below are also fixed, not independent
parameters. Define

\[
q_\pm=[\,|u|^2|z_\pm|^2\,]Z_C
 =[\,|u|^2|z_\pm|^2\,]
       \operatorname{Res}\left(H_4+\tfrac12\{H_3,S_3\}\right),
\]

where the bracket, four spatial waves and actual twenty-edge construction
are precisely those of Section 14.2 with A set to zero. Coefficient
extraction in this finite formula is an exact specification of each q.
Reality of Z_C makes both real and finite. Their numerical values are
unnecessary for the small-gamma sign argument; their contributions remain
in the complete linearized rows.

The reference is

\[
A_i=e^{\mathrm i\alpha i-\mathrm i\Omega_A T},\qquad
u=\gamma e^{-\mathrm i\Omega_u T},\qquad v=z_+=z_-=0,
\]

\[
\Omega_A=\kappa+d_0\gamma^2,\qquad
\Omega_u=d_0+\frac{2s}{5}\gamma^2,
\qquad \kappa=-\frac{14+21\sqrt5}{164}<0.
\]

Here u is a collective complex amplitude, not capacity or the internal
form contrast u_i. These rotations follow by differentiation of K, including
both reciprocal mixed terms and K_C. No collective forcing is prescribed.

### 15.2. Complete perturbation blocks and their normalization

Remove the two occupied reference rotations by writing

\[
A_i=e^{\mathrm i\alpha i-\mathrm i\Omega_A T}(1+w_i),\quad
u=e^{-\mathrm i\Omega_u T}(\gamma+\eta),\quad
v=e^{-\mathrm i\Omega_u T}\widetilde v,
\qquad w_i=\sum_{k=0}^4 w_k e^{\mathrm i k\alpha i}.
\]

Leave z_+ and z_- in their original leading frame. Their reference is
zero and supplies no phase to quotient. Again u in this formula denotes
the same occupied collective amplitude. In the remaining formulas v
means its perturbation coordinate `tilde(v)`.
The Fourier w_k have bracket `i/5`; this normalization matches the
collective waves. The factors five from summing over pairs consequently
cancel in the reciprocal coupling.

Use the exact coefficients of Section 10.5,

\[
a_k=\frac{2n_k+2m_{k+2}-m_2}{32},\qquad
b_k=\frac{2n_k+m_2}{32}.
\]

The symbol `m_k` here is the Fourier symbol of the earlier matrix M,
not the high frequency m. The unchanged four-real-dimensional internal
block `(w1,bar(w4))` has matrix
`[[a1,b1],[-b1,-a4]]` in its `i*d/dT` equation, with its conjugate
partner understood. The uniform detuning `d0*gamma^2` cancels against
Omega_A. The only nonuniform mixed term is at spatial wave number two,
so the counterpropagating component closes with the other internal pair:

\[
X=\begin{pmatrix}w_2\\\overline w_3\\\overline v\end{pmatrix},\qquad
\boxed{\quad \mathrm i X_T=\mathcal M_\gamma X,\qquad
\mathcal M_\gamma=
\begin{pmatrix}
a_2&b_2&h\gamma\\
-b_2&-a_3&-h\gamma\\
-h\gamma&-h\gamma&-d\gamma^2
\end{pmatrix}.\quad}
\]

For example the mixed quadratic expression before differentiation is
`5*h*gamma*(bar(v)*(w3+bar(w2))+v*(w2+bar(w3)))`.
Its collective derivative supplies the third row; treating v as a fixed
external source would remove that row and change the conclusion.
The coefficient `d*gamma^2` follows from the transverse derivative of
`K_C-Omega_u*5*(|u|^2+|v|^2)` together with the uniform mixed term.
This block has six real dimensions, not three; its conjugate equations
are retained by the complex representation.

The initially empty band has the four real equations

\[
\mathrm i(z_\pm)_T=\nu_\pm(\gamma)z_\pm,\qquad
\nu_\pm(\gamma)=d_{02}-\frac{q_\pm}{5}\gamma^2,\qquad
d_{02}=\frac{\ell_2}{4}\left(1+\frac{a^2}{2+\ell_2}\right)>0.
\]

Pure collective quartics contain only directional intensities, as proved
in Section 14.5. The mixed term is quadratic in C2. These facts make this
band diagonal and exclude linear coupling to w, eta or v at this
reference, while retaining its actual cross-band frequency shifts.
Both frequencies remain real and nonzero for sufficiently small gamma.
No phase of the empty reference was assigned a value or subtracted.

Finally write `w0=r+i*phi` and `eta=xi+i*psi`. The four-real-dimensional
neutral block is

\[
r_T=\xi_T=0,\qquad
\binom{\phi}{\psi}_T=-
\begin{pmatrix}
2\kappa&2d_0\gamma\\
2d_0\gamma&4s\gamma^2/5
\end{pmatrix}
\binom r\xi.
\]

Xi and psi are the real and imaginary parts of the absolute perturbation
eta of u. The first-order occupied collective phase perturbation is
`psi/gamma`, not psi itself. The determinant of the displayed shear
matrix is `gamma^2*(8*kappa*s/5-4*d0^2)<0`. Thus for each fixed positive
gamma there are exactly two independent phase vectors and two genuine
size-two Jordan blocks at zero: action changes induce reference-frequency
changes. They have not been discarded to make a boundedness assertion.
The dimension count is `4+6+4+4=18`. Restoring the two fixed common fine
origins adds their two exact constant directions, not another interaction.

### 15.3. The reciprocal slow mode is elliptic for small positive gamma

Set

\[
G=\begin{pmatrix}a_2&b_2\\b_2&a_3\end{pmatrix},\qquad
D_G=a_2a_3-b_2^2.
\]

Section 10.6 proves that G is positive definite, in particular
`D_G>0`. Its corresponding internal frequency matrix
`diag(1,-1)*G` has the two distinct nonzero real frequencies
`d2+/-sqrt(Delta2)` of Section 10.5. Together with the zero eigenvalue
at gamma zero these are three simple real roots of the characteristic
polynomial of M_gamma. Its real coefficients are analytic in gamma.
Each simple root therefore has a unique real analytic continuation on a
sufficiently small interval. This directly excludes a complex growing
pair there; a Hamiltonian label alone would not do so.

The new root is not an additional zero direction for positive gamma.
Define its exact leading coefficient by

\[
d_{\rm eff}=d-h^2\mathbf1^{\mathsf T}G^{-1}\mathbf1
 =d-h^2\frac{a_2+a_3-2b_2}{D_G}<0.
\]

The inequality follows from `d<0`, `h!=0` and positivity of G, without
estimating a trajectory. Schur reduction near the isolated zero root
gives

\[
\boxed{\quad
\lambda_{\rm slow}(\gamma)=-d_{\rm eff}\gamma^2+O(\gamma^4)>0.
\quad}
\]

Only even powers occur because conjugation by `diag(1,1,-1)` changes
gamma to minus gamma. The leading internal entries of this eigenvector,
when its last entry is normalized to one, are
`-h*gamma*G^-1*(1,1)^T+O(gamma^3)`. The internal timing response is part
of this joint mode; it is not obtained by independently imposing a pulse
frequency. The sign of its conjugate frequency is reversed, as required
for a real Hamiltonian system.

The other two frequencies stay separated from zero and one another.
Therefore M_gamma is diagonalizable with three real eigenvalues for each
sufficiently small fixed positive gamma, and its real evolution has three
bounded oscillatory pairs. The separate internal block and the empty-band
blocks are likewise semisimple with purely imaginary evolution exponents.
All fourteen nonneutral real directions consequently have bounded linear
evolution. The four neutral directions retain their stated shears.
This is a spectral and linear-response result for the complete leading
model, not an assertion that every raw perturbation is bounded.

In particular an initially isolated counterpropagating perturbation,
with w and eta zero, lies wholly in the coupled six-real-dimensional
block at first order. It does not excite the action/phase shear block at
that order. Its induced internal envelope-phase contrasts are bounded
oscillatory functions. Their new long modulation scale in T is of order
`gamma^-2`. A prescribed constant detuning would instead give an
uncorrected secular phase difference and would miss this reciprocal mode.
Bounded oscillation does not mean damping or convergence to a preferred
relative phase, and the different frequencies need not share a period.
For an explicit nonzero response, let the initial unit counter amplitude
be real, `v(0)=1`, with every internal and occupied-band perturbation zero.
Denote the reconstructed site perturbation by
`W_i=sum_k w_k*exp(i*k*alpha*i)`. Then
`W_i'(0)=-2*i*h*gamma*cos(2*alpha*i)`, so
`[d Im(W_1-W_0)/dT](0)=2*h*gamma*(1-cos(2*alpha))!=0`.
This distinguishes a bounded response from absence of an interaction.

### 15.4. Why this does not give nonlinear orbital stability

Multiplying M_gamma by `diag(1,-1,-1)` gives the Hermitian quadratic form

\[
\mathcal G_\gamma=
\begin{pmatrix}
a_2&b_2&h\gamma\\
b_2&a_3&h\gamma\\
h\gamma&h\gamma&d\gamma^2
\end{pmatrix}.
\]

Its Schur complement relative to G is `gamma^2*d_eff<0`.
It has two positive and one negative complex directions. The counter
mode thus has the opposite quadratic-energy sign to the internal modes.
The constrained positive-minimum argument of Section 10.6 does not extend
to this joint family: keeping the original conserved totals does not make
this form positive. This is a limitation of that sufficient argument,
not a proof of nonlinear instability.

The frequency separation proves bounded linear modulation at sufficiently
small fixed gamma. It does not prevent nonlinear transfer within the
retained bands over arbitrarily long times, prove stability through a
later frequency collision, or select a uniform gamma interval numerically.
No parameter scan or larger-amplitude continuation follows from this
local result. The initially empty band is allowed to be perturbed even
though its action has zero first derivative at the reference; its four
directions cannot be removed by claiming to quotient that action.

### 15.5. Finite nonlinear and original-flow interpretation

There is nevertheless a controlled finite observation, not merely a
formal eigenvalue statement. Fix one admitted positive gamma and a finite
slow horizon T_star. Let B0(T) be the reference leading solution and
prepare a perturbed leading initial state with counter amplitude
`v(0)=rho*v0`, while initially leaving w, eta and C2 zero. Here rho is a
small perturbation parameter distinct from epsilon and gamma. The full
leading vector field is polynomial and its action spheres are bounded.
Differentiability of its flow on the selected compact neighborhood gives

\[
B_\rho(T)=B_0(T)+\rho V(T)+O_{\gamma,T_*}(\rho^2),
\qquad 0\le T\le T_*,
\]

where V solves the full variational equations just derived. The error
bound is uniform on this fixed interval. The initial counter perturbation
changes the band action only at order rho squared, consistently with its
absence from the linear neutral block. This does not license dropping the
neutral block for a different preparation.

Prepare the two exact original flows through the canonical reconstruction
of Section 14, using their complete model initial coordinates. Its exact
inverse initialization and remainder bound give the corresponding
original-state response in the form

\[
(q,x)_\rho-(q,x)_0
 =\varepsilon\rho\,\mathcal R(\tau)V(\varepsilon^2\tau)
  +O_{\gamma,T_*}(\varepsilon\rho^2+\varepsilon^2),
\qquad 0\le\tau\le T_*/\varepsilon^2.
\]

The fixed linear map R includes the modal rotations, their real conversion
and the original form factor `sqrt(c)`. The nonlinear reconstruction
contributes only to the retained error at this order. The original-time
window is still `pi*T_star/(sqrt(c)*epsilon^2)`. The constants and cutoff
are existential. To resolve a selected nonzero response its error must
be smaller than that response; no practical numerical domain is supplied.

For clarity the linear timing contrast used here is the internal envelope
contrast `Im(W_i-W_j)`. It is not the raw primitive phase or the difference
of instantaneous local speeds. The normalized internal reading
`-arg(delta_i+i*u_i/sqrt(c))` equals
`tau+arg(A_i(T))+O(epsilon)` while the internal amplitudes stay nonzero.
For sufficiently small rho and fixed T_star they stay bounded away from
zero near this reference. Their relative normalized readings therefore
have the predicted contrast

\[
\rho\operatorname{Im}(W_i(T)-W_j(T))
       +O_{\gamma,T_*}(\rho^2+\varepsilon)
\]

after subtraction of the reference contrast. Here W denotes the reconstructed internal site
part of the unit variational response V. The actual admitted reading chi
is obtained through the increasing ellipse map of Section 14.7. Its
bounded derivative on the circle transfers the finite error, with the
same continuous-lift and nonvanishing requirements. Interior tick bounds
retain that section's speed and endpoint margins. Common reference phase
changes cannot simply be identified with identical instantaneous chi rates.

One may choose T_star long enough to examine the slow oscillation for a
fixed gamma, but its required size grows like gamma to the minus two.
This proof supplies no simultaneous uniform limit as gamma tends to zero,
epsilon tends to zero and T_star grows. Nor does it establish an exact
full-flow multimode torus, attraction, generic formation, an emergent
ontological time or empirical identification. Its result is a complete
reciprocal bounded linear mechanism with a stated finite nonlinear and
original-flow observation bridge.

The declaration in `artifacts/counterpropagating-pulse-robustness-20261028/scope.json`
owns the finite control selection and resource budget. Frozen receipts and
the research program own evaluation status. No trajectory, parameter scan
or historical producer replay is part of this derivation.
