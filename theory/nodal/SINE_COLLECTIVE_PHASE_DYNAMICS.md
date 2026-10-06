# Collective phase families and live contact feedback

Exact phase-offset partitions, nearby full-state moving windows, contact/storage exchange and complete receiver-rigidity conditions.

Part of [Sine pattern geometry and dissipative capture](SINE_PATTERN_DYNAMICS.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 39. Compatible internal geometry and collective phase dynamics

<a id="sine-phase-offset-partition"></a>

A region can retain phase differences while its common form and phase
continue to evolve. The full nodal rows determine when this is possible;
an instantaneous equality of phase rates does not suffice. This section
classifies exact collective families under Section 35's conservative unit
law, fixed finite connected simple support, unit held capacities and
`tau=t/pi`. No input, event or change of support is introduced.

### A declared partition with fixed internal phase offsets

Partition every fine node into nonempty blocks `B_a`. Supply fixed circular
offsets `psi_i`; a consistent real lift can be used for differentiation.
Consider the **whole family**

\[
\mathcal M_\psi:
\qquad x_i=X_a,\qquad \theta_i=\Theta_a+\psi_i
\quad(i\in B_a),
\]

with freely varying collective forms `X_a` and phases `Theta_a`. Internal
phase differences are fixed, but different nodes need not share a phase.
The block partition and offsets are supplied geometry, not inferred
membership or a newly created node. Define, using full graph degrees,

\[
q_{ib}=\frac{|N(i)\cap B_b|}{d_i},\qquad
Z_{ib}=\frac1{d_i}
       \sum_{j\in N(i)\cap B_b}e^{\mathrm i(\psi_j-\psi_i)}.
\]

The family is invariant under the complete law **if and only if**, in
each block `a`, the following conditions hold:

1. For every external block `b!=a`, `q_ib=q_ab` is independent of `i`.
2. For every external block `b!=a`, the full complex value `Z_ib=Z_ab`
   is independent of `i`.
3. Every internal sine sum vanishes: `Im Z_ia=0` for all `i in B_a`.

These are conditions on support and phase offsets for all collective
states. Failure does not exclude an accidental special trajectory with
additional relations between its collective coordinates. In particular,
equal neighbor counts alone do not settle the nonlinear form row.

### Necessity and sufficiency use both consumed rows

On the proposed family the phase row at `i in B_a` is

\[
\theta_i'=\sum_{b\ne a}q_{ib}(X_a-X_b).
\]

For its value to be independent of `i` for every independent choice of
`X`, each external coefficient must agree. The form row is

\[
x_i'=\operatorname{Im}Z_{ia}
 +\sum_{b\ne a}\operatorname{Im}
       \left[Z_{ib}e^{\mathrm i(\Theta_b-\Theta_a)}\right].
\]

Independent variation of the collective phase differences makes the
external sine and cosine coefficients independent Fourier components.
Their equality forces equality of the external complex values `Z_ib`.
The remaining internal imaginary parts must be one common constant
`c_a`. Undirected internal edges give

\[
\sum_{i\in B_a}d_i\operatorname{Im}Z_{ia}
=\sum_{\substack{i,j\in B_a\\j\sim i}}
  \sin(\psi_j-\psi_i)=0.
\]

Since `sum_(i in B_a) d_i>0`, that common constant is zero. There is no
additional condition on `Re Z_ia`: the form row does not consume it and
the phase row has zero internal form differences. For example a one-block
P3 with offsets `(0,0,pi)` has internal real values `(1,0,-1)`, zero
internal sine current and uniform-form equilibria. Requiring the complete
internal complex sums to agree would wrongly reject this family.

Conversely, the three conditions make both fine rows constant within
their respective blocks. The field is tangent to `M_psi` everywhere;
smooth uniqueness therefore preserves the family. Its exact closed law is

\[
\boxed{\begin{aligned}
X_a'&=\sum_{b\ne a}\operatorname{Im}
       \left[Z_{ab}e^{\mathrm i(\Theta_b-\Theta_a)}\right],\\
\Theta_a'&=\sum_{b\ne a}q_{ab}(X_a-X_b).
\end{aligned}}
\]

Thus phase-offset geometry and support determine whether all constituents
maintain compatible rates while the collective coordinates move. This is
an invariant-family reduction, not a quotient for every fine state.

### Reciprocity, inherited storage and conserved means

Let `m_a=sum_(i in B_a) d_i`, `N_ab=m_a q_ab` and
`P_ab=m_a Z_ab`. Counting the same undirected cross edges in reverse gives

\[
N_{ab}=N_{ba},\qquad
P_{ab}=\overline{P_{ba}},\qquad |Z_{ab}|\le q_{ab}.
\]

The full fine storage restricted to the family is exactly

\[
\begin{aligned}
E_{\mathcal M}
&=E_{\mathrm{int}}(\psi)
 +\sum_{a<b}\left[
 \frac{N_{ab}}2(X_a-X_b)^2+N_{ab}
 -\operatorname{Re}\left(
     P_{ab}e^{\mathrm i(\Theta_b-\Theta_a)}\right)\right],\\
E_{\mathrm{int}}(\psi)
&=\sum_a\sum_{\{i,j\}\subset B_a,\ i\sim j}
       [1-\cos(\psi_j-\psi_i)].
\end{aligned}
\]

The closed rows consequently satisfy

\[
X_a'=-\frac1{m_a}\frac{\partial E_{\mathcal M}}{\partial\Theta_a},
\qquad
\Theta_a'=\frac1{m_a}\frac{\partial E_{\mathcal M}}{\partial X_a}.
\]

This is a restriction of the existing storage and reciprocal field, not
an independently supplied energy. It preserves that storage and the
collective weighted sums `sum_a m_a X_a` and `sum_a m_a Theta_a` in
continuous lifts. The fixed contribution `sum_i d_i psi_i` restores the
full lifted phase sum.

In general the inherited form coupling `Z_ab` and phase coupling `q_ab`
are different. Phase cancellation can give `|Z_ab|<q_ab`, and complex
offsets can remain in the interaction. Replacing both by the weights of
a bare normalized-sine graph would then change the complete law. This
is the same retained-information boundary as in
[nonlinear replica inheritance](SINE_PAIR_STATE.md#sine-replica-inheritance).

For **each block separately**, internal form storage is zero and internal
phase storage is constant. In fact `q_R=S_R=0` at every point of this
family, not only initially. Section 36's internal channels and the net
regional power `P_R` therefore vanish throughout its motion. This does
not remove interaction: the collective forms and phases, cross-edge
storage and common regional rates can change. Internal regional storage
does not count that cross-edge organization.

### One moving realization on the retained private-leaf support

Use the same C5 plus five private leaves as Section 37, with blocks
`R=(0,...,4)` and `Q=(5,...,9)`. Supply offsets
`psi_i=psi_(5+i)=sigma*2pi*i/5`, for `sigma=+1` or `-1`.
The external coefficients are

\[
q_{RQ}=Z_{RQ}=\frac13,\qquad
q_{QR}=Z_{QR}=1.
\]

Every receiver internal sine sum cancels between the two cycle directions;
the leaves have no internal edges. Write the block forms as `a,b` and
phases as `Theta,Psi`. The complete reduced rows are

\[
a'=\tfrac13\sin(\Psi-\Theta),\qquad
b'=-\sin(\Psi-\Theta),\qquad
\Theta'=\tfrac13(a-b),\qquad
\Psi'=b-a.
\]

Putting `u=a-b`, `phi=Psi-Theta` gives

\[
u'=\tfrac43\sin\phi,\qquad
\phi'=-\tfrac43u,\qquad
\tfrac12u^2+1-\cos\phi=\text{constant}.
\]

The receiver retains its exact acute winding and internal storage `V_5`
while interacting with its moving environment. Its collective weighted
form sum is `15a+5b`; the analogous lifted phase sum has the same weights.
The common regional phase motion is compensated by the environment, in
accord with Section 38's global phase balance. The inherited pendulum
reuses the [reciprocal exchange mechanism](RESONANCE_FOUNDATIONS.md#reciprocal-exchange);
it is not another fundamental oscillator, frequency identification or
universal permanent-pulse postulate.

### Exact compatibility is not finite-time formation into the family

The complete smooth field has a unique flow in both time directions.
If a trajectory met this invariant family at a finite time, the unique
backward solution through that point would also lie in the family.
Therefore a preparation outside `M_psi` cannot enter it exactly at finite
time under this unchanged autonomous law. In particular, globally flat
phases cannot enter the displayed nonzero-twist family exactly.

This does not exclude entry into a neighborhood with nonzero width, nor
finite retention of a less restrictive identity. Those require an actual
entry state, declared finite error bounds and a comparison under the
same complete law. The existing
[full-state finite-window argument](SINE_PAIR_GROUPING.md#sine-joint-identity-window)
illustrates that separate obligation on its own support; it is not
automatically a certificate for these arbitrary partitions. No acquisition
trajectory or attraction theorem is supplied by the exact classification.

### Shared admission distinguishes proof from undecided equality

The [partition owner](../../src/tnfr/physics/relational_sine_partition.py)
provides `assess_sine_phase_offset_partition(source, blocks,
phase_offset_turns)` and `SinePhaseOffsetPartition`. Declared rational-turn
offsets specify the phase geometry independently of a response.
Its `.evaluate(...)` returns a `SinePhaseOffsetState` for admitted collective
coordinates, retaining fine reconstruction, the inherited rows and storage.
It does not install a coarse runtime, select a partition or certify that
the source already belongs to the supplied family.

The all-real mathematical criterion is stronger than any particular finite
equality recognizer. Exact cancellations admitted by the implementation,
including sine oddness and angle reflection, can prove the required
identities. Certified interval exclusion of zero can prove a failed
equality. An unrecognized symbolic cancellation or an interval containing
zero remains unavailable; it is not evidence of inequality and must not
be converted into a failed mathematical criterion.

The [held native joint quotient](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#joint-constitutive-reduction-with-inherited-support-counts),
the [replica family](SINE_PAIR_STATE.md#sine-replica-inheritance)
and the [independent-swap criterion](SINE_PAIR_STATE.md#sine-pair-support-symmetry)
retain their different models and quantifiers. This theorem concerns
supplied fixed offsets and all collective states on their invariant family;
it does not claim closure of arbitrary fine states, autonomous formation,
support selection or physical identification.

## 40. A finite-width window around compatible moving geometry

<a id="sine-moving-pattern-window"></a>

Section 39's exact family is not an acquisition target that can be entered
from outside at finite time. A neighborhood with nonzero width has a
different obligation: errors in both the receiver and its environment must
remain controlled while the reference moves. The following sufficient
certificate uses the actual storage and complete conservative field. It
proves finite retention for an independently supplied entry neighborhood;
it does not produce that neighborhood from an earlier preparation.

### Reference motion, error chart and declared duration

Retain the C5 with five private leaves, unit capacities, `e=0`, `w=beta=1`,
no input or events, and `tau=t/pi`. Use Section 39's certified two-block
family with receiver winding `sigma=+1` or `-1`, matching receiver/leaf
phase offsets and `alpha=2pi/5`. Reference block forms are `a,b`, their
phases are `Theta,Psi`, and

\[
u=a-b,\qquad \phi=\Psi-\Theta,\qquad
h=\tfrac12u^2+1-\cos\phi.
\]

The inherited pendulum preserves `h` and satisfies
`|phi'|<=4sqrt(2h)/3`. Choose a radian contact envelope
`0<Gamma<pi/2`, with `|phi(0)|<Gamma` and
`h<1-cos(Gamma)`. Energy and continuity then keep the reference contact
phase in `(-Gamma,Gamma)` for all positive and negative times. This
includes the zero-pulse reference; a nonzero pulse is not required.

Fix a finite duration `T>0` in `tau` and two radian error radii:

\[
0<\rho_R<\frac\pi{10},\qquad
0<\rho_C<\frac\pi2-\Gamma.
\]

Set `kappa_R=cos(alpha+rho_R)>0` and
`kappa_C=cos(Gamma+rho_C)>0`. The internal receiver and contact errors
use different margins. Contact convexity is a sufficient premise of this
certificate, not a defining property of a regional identity. Exact
compatible families can also move through nonacute contacts; this estimate
does not assess their transverse robustness.

For an actual full solution and the reference, use continuously matched
phase lifts. On oriented edge `i->j`, write
`f_e=x_i-x_j`, `delta_e=theta_j-theta_i`,
`xi_e=f_e-f_e^*` and `eta_e=delta_e-delta_e^*`.
Admit the initial chart separately: `|eta_e(0)|<rho_R` on receiver
edges and `<rho_C` on contacts. A small scalar storage value alone does
not establish these phase-chart premises.

### The moving-reference relative-storage identity

For `U(delta)=1-cos(delta)`, define

\[
D=\frac12\sum_e\xi_e^2
 +\sum_e\left[U(\delta_e^*+\eta_e)-U(\delta_e^*)
                    -\sin(\delta_e^*)\eta_e\right].
\]

This is the Bregman remainder of the existing full storage, not another
physical energy or a new evolution rule. The complete field has the
constant reciprocal matrix with blocks `0,-K;K,0`. Differentiating the
remainder along both actual solutions cancels the quadratic form terms
and gives exactly

\[
\boxed{D'
=\sum_e(\delta_e^*)'
 [\sin(\delta_e^*+\eta_e)-\sin(\delta_e^*)
                          -\cos(\delta_e^*)\eta_e].}
\]

Internal reference gaps are fixed at `sigma*alpha`; only the five
contacts contribute, each with reference gap `phi`. While the admitted
chart holds, convexity and Taylor's remainder imply

\[
D\ge\frac12\sum_e\xi_e^2
 +\frac{\kappa_R}2\sum_{e\in R}\eta_e^2
 +\frac{\kappa_C}2\sum_{e\in C}\eta_e^2,
\qquad
|D'|\le\frac{|\phi'|}{\kappa_C}D\le\lambda D,
\]

where

\[
\lambda=\frac{4\sqrt{2h}}{3\kappa_C},\qquad
d_* =\frac12\min\{\kappa_R\rho_R^2,\kappa_C\rho_C^2\}.
\]

Thus a proved initial upper bound `D(0)<=D_0`, together with the initial
chart and the prospective strict inequality

\[
\boxed{D_0e^{\lambda T}<d_*}
\]

retains the complete error chart for every `|tau|<=T`. Indeed a first
chart exit would require `D>=d_*`, contradicting the integrated bound.
This is a full-interval proof, not a claim inferred from sampled endpoints.

For independent initial nodal form/phase error radii `r_i,p_i>=0`,
the shared sufficient initial bound is

\[
D_0=\frac12\sum_{\{i,j\}\in E}
       [(r_i+r_j)^2+(p_i+p_j)^2].
\]

The same component sums must pass the separate initial edge-chart tests.
During the window, every edge form error is at most `sqrt(2D_max)`;
receiver and contact phase errors are at most
`sqrt(2D_max/kappa_R)` and `sqrt(2D_max/kappa_C)`, respectively, where
`D_max=D_0 exp(lambda T)`. Initial weighted form/phase mean errors remain
their conserved values. Retaining those origins with the edge bounds
controls the relative full state on this connected support; no hidden
environmental coordinate has been replaced by an imposed boundary drive.

### Internal identity and actual work remain compatible

The reference receiver has uniform form and equal oriented principal gaps
`sigma*alpha`. Its internal linear phase term cancels because
`sum_(e in R) eta_e=0`. The receiver's contribution to `D` is therefore
exactly `E_R-V_5`. Nonnegative contact contributions give

\[
0\le E_R-V_5\le D\le D_{\max}.
\]

The receiver retains acute winding `sigma` throughout the window, with
bounded internal form as well as phase. The environment continues its own
nodal motion. The regional work identity still determines its actual net
work; zero work for the exact reference does not imply zero work for every
nearby trajectory. This result supplies a positive-width continuation
criterion without assuming that an arbitrary environment is harmless.

### Conserved storage and backward time restrict acquisition claims

The full reference storage is `H^*=V_5+5h`. At the declared entry instant,
the exact relation to an actual state is

\[
H-H^*=D+u\sum_{e\in C}\xi_e
             +\sin\phi\sum_{e\in C}\eta_e.
\]

With the component error radii above, put

\[
L_0=|u|\sum_{\{i,j\}\in C}(r_i+r_j)
   +|\sin\phi|\sum_{\{i,j\}\in C}(p_i+p_j).
\]

The whole declared entry box has the conserved-storage enclosure
`H in [H^*-L_0,H^*+L_0+D_0]`. It stays valid along every covered
trajectory because the full law conserves `H`.

If its upper bound is below `7/2`, the
[phase-flat acquisition barrier](SINE_REGIONAL_FORMATION.md#sine-regional-channel-accessibility)
excludes every phase-flat source from reaching this entry box. In the
vanishing-width limit, a necessary compatibility condition is

\[
h\ge\frac{7/2-V_5}{5}
  =\frac{5\sqrt5-11}{20}.
\]

This floor neither proves entry nor uniquely selects a pulse. The
[larger sector proof](SINE_REGIONAL_FORMATION.md#sine-cycle-sector-barrier) also excludes any initial
zero-winding receiver from a target box whose conserved full storage is
strictly below `7/2`. Initial winding and the actual full budget remain
separate admitted observations; a nonacute unit-winding source is not
automatically covered by that particular zero-winding test.

A named acquisition source must also have a conserved storage compatible
with the declared box. For example Section 38's private-leaf ramp
`x_(5+i)=-3omega(i-2)` has exact full storage `45omega^2`.
Its large-rate acquisition cannot be joined to a small-error reference
with `h=1/50` and full storage near `V_5+1/10`: the budgets are disjoint.
Discarding that discrepancy would remove the actual environmental state
or silently introduce loss. Budget overlap is necessary, not sufficient,
for a new favorable preparation to acquire into this neighborhood.

The error estimate also holds backward for the same finite duration.
Every admitted entry-box state already had the acute identity over its
certified preceding window. Hence this certificate cannot describe first
appearance at the central instant; an initially identity-absent preparation
must precede that window if the same support and law apply throughout.
Formation followed by later stabilization remains a separate possibility.

The example `u=1/5`, `phi=0`, `Gamma=1/4`, `rho_R=rho_C=1/10`,
`T=5` and nodal form/phase error radii `1/1000` gives a nonzero-width
finite retention certificate with a moving reference. These declared
values illustrate the sufficient inequalities without optimizing a
threshold or executing a new trajectory. Setting the reference pulse to
zero gives a complementary low-storage control: maintenance can be
certified while phase-flat acquisition is excluded.

[`assess_sine_moving_pattern_window`](../../src/tnfr/physics/relational_sine_partition.py)
rebuilds the declared reference and applies these independent chart,
reference-motion, storage-growth and acquisition-budget checks. It also
encloses full storage directly from the primitive edge uncertainty boxes.
When the initial receiver chart is certified, the lower bound additionally
retains its correlated cycle minimum `V_5`, rather than independently
minimizing all five phase gaps. The primitive enclosure remains valid when
the relative-storage certificate is unavailable. The
[contract](../../docs/contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-moving-pattern-window)
and [usage](../../docs/guides/relational/SINE_REGIONAL_DYNAMICS.md#moving-pattern-window)
retain the distinction between a certified finite neighborhood and an
independently demonstrated formation-to-retention handoff.

## 41. Actual contact motion can exchange storage with internal structure

<a id="sine-collective-pulse-transfer"></a>

The contact pendulum of Section 39 conserves its own storage only on the
exact compatible family. Away from that family, the same full nodal law
supplies a correction from unequal contact phases. This identifies an
actual exchange mechanism to examine, rather than adding a controller or
assuming that a prepared compatible pattern has already formed.

### Mean-contact rows from the complete state

Keep the unchanged conservative unit C5/private-leaf support of Sections
39–40. Match receiver node `r_i` with its actual private leaf `l_i`.
No uniformity of the observed forms or phases is assumed. Define

\[
a=\frac15\sum_i x_{r_i},\qquad
b=\frac15\sum_i x_{l_i},\qquad u=a-b,
\]

and, in declared continuous phase lifts,

\[
\phi_i=\theta_{l_i}-\theta_{r_i},\qquad
\bar\phi=\frac15\sum_i\phi_i,\qquad
\eta_i=\phi_i-\bar\phi,\qquad \sum_i\eta_i=0.
\]

Internal cycle currents cancel in the receiver mean. Summing the complete
phase rows cancels its internal form Laplacian as well. In `tau=t/pi`,
the exact mean-contact laws are therefore

\[
\boxed{u'=\frac43\left\langle\sin\phi_i\right\rangle,
\qquad \bar\phi'=-\frac43u,}
\]

where brackets denote the mean over the five contacts. Equivalently, if
`Z=<exp(i eta_i)>`, then
`u'=(4/3) Im[exp(i bar_phi) Z]`. The contact deviations are live fine-state
coordinates. Replacing `Z` by one recovers the exact compatible family,
but it is not justified for an arbitrary acquired state.

Define the chart-dependent collective contact storage

\[
h_c=\tfrac12u^2+1-\cos\bar\phi.
\]

Its exact derivative is

\[
\boxed{h_c'
=\frac43u\left[
 \left\langle\sin(\bar\phi+\eta_i)\right\rangle
 -\sin\bar\phi\right].}
\]

Thus collective contact storage is not a second general invariant. Its
exchange with the remaining structure can have either sign. The formula
does not select an initial source, a direction of long-time transfer or a
new constitutive law.

### The storage split retains its phase-lift dependence

Let `d_i=x_(r_i)-x_(l_i)`, and use `F_R,V_R` for the actual internal
receiver stores. Direct edge summation gives

\[
\begin{aligned}
H&=5h_c+\mathcal R,\\
\mathcal R
&=F_R+V_R+\frac12\sum_i(d_i-u)^2\\
&\quad+\sum_i\left[
 U(\bar\phi+\eta_i)-U(\bar\phi)
                    -\sin\bar\phi\,\eta_i\right],
\qquad U(s)=1-\cos s.
\end{aligned}
\]

Consequently `R'=-5h_c'` because the full storage `H` is conserved.
The remainder is signed in general. If all segments joining `bar_phi`
to the five contact phases lie in one convex cosine-potential chart, its
contact terms are nonnegative. Even in that chart, nonnegativity does not
make the remainder an independently controlled reservoir or specify when
it feeds the receiver. Its actual transfer remains fixed by the full rows.

The contact mean uses retained **real lifts**. Independently wrapping each
contact and then averaging can change `bar_phi`, `h_c` and the signed
remainder, while leaving the actual full storage unchanged. A single
snapshot can declare a starting lift, but it does not reconstruct discarded
phase history. These collective coordinates are not new globally intrinsic
physical energies. The common-real-phase preparation below fixes their
initial lift unambiguously, and the complete smooth phase rows continue it.

### A phase-flat preparation has a discriminating fourth derivative

At a preparation with one common real phase, set
`v_i=phi_i'(0)`, `m=<v_i>=-4u/3`, and `delta_i=v_i-m`.
All form rates vanish initially. The first three derivatives of `h_c`
are zero. Differentiating the exact transfer row gives

\[
\boxed{h_c^{(4)}(0)
=\frac{16u^2}{3}\left\langle\delta_i^2\right\rangle
 -\frac{4u}{3}\left\langle\delta_i^3\right\rangle.}
\]

To check the cancellation, write
`g=<sin(phi_i)>-sin(bar_phi)`. At zero phase,
`g=g'=g''=0`, and
`g'''=-(<v_i^3>-m^3)`; mean higher phase jets cancel because
`bar_phi=<phi_i>` exactly. Expanding `<(m+delta_i)^3>` gives the displayed
variance and third-moment terms. This is a local derivative of the full
law, not a finite-time approximation substituted for it.

### One phase-flat preparation above the necessary phase-path budget

On receiver nodes `(0,...,4)` and private leaves `(5,...,9)`, supply

\[
u=\frac{119}{100},\qquad \epsilon=\frac1{20},\qquad
x_{r_i}=\frac u4+\epsilon(\mathbf1_{i=0}-\mathbf1_{i=1}),
\qquad x_{l_i}=-\frac{3u}{4},
\]

with all phases zero. The weighted form and phase means are zero. The
initial receiver has zero winding, so the proposed acute unit-winding
identity is absent. Exact edge summation gives

\[
H=\frac52u^2+4\epsilon^2
 =\frac{14201}{4000}=3.55025.
\]

This is above the necessary `7/2` phase-path barrier. It does not make
Section 40's particular small-error reference box accessible. That box
retains acute receiver winding, hence `V_R>=V_5`; its five contact form
gaps are each at least `1/5-2/1000=99/500`. Consequently every state in
that box satisfies the sharper, correlation-aware lower bound

\[
H\ge V_5+\frac52\left(\frac{99}{500}\right)^2
 =V_5+\frac{9801}{100000}
 >\frac{14201}{4000}.
\]

An independent componentwise edge enclosure can overlap the source budget
while forgetting this exact cycle constraint. Such outer-interval overlap
is not accessibility. Conservation excludes a handoff to that fixed box;
the control tests finite acute identity only. Any later maintenance
neighborhood requires its own source-compatible storage admission, without
post hoc damping or removal of environmental state.

Removing the transverse dipole gives an exactly invariant zero-winding
contact-pulse control. The control is not claimed to have the same energy
as the nonzero-dipole source. Its invariant geometry supplies the analytic
comparison without a second numerical response or amplitude search.

For the nonzero dipole, the actual centered contact velocities are

\[
(\delta_0,\ldots,\delta_4)
=\frac1{60}(-7,7,-1,0,1),\qquad
\langle\delta_i^2\rangle=\frac1{180},\qquad
\langle\delta_i^3\rangle=0.
\]

Therefore `h_c^(4)(0)=4u^2/135>0`. Time reversal at common initial phase
makes form even and phase odd, so locally

\[
h_c(\tau)=h_c(0)+\frac{u^2}{810}\tau^4+O(\tau^6).
\]

The first nonzero transfer is **toward the collective contact pulse**.
It does not give immediate transfer into the desired organized receiver.
This also does not mean every receiver field is stationary: internal
form and phase can exchange while the aggregate remainder decreases.
That local result does not decide whether later modulation produces the
target identity; the separate finite response below evaluates that question.

### Retained full-law control excludes acute winding on its declared horizon

One bounded control used exactly this preparation, unchanged law and
support. Its declared original-clock horizon was `t in [0,24]`, time step
`3/32`, Taylor order `12` and at most `256` steps. The observation asked
whether acute receiver winding `+1` or `-1` holds for an original-clock
duration of at least `1/2`, with strict positive acute margin; the
configured margin threshold is zero.

The [declaration](../../docs/assets/collective_pulse_organization/declaration.json)
and [frozen protocol](../../docs/assets/collective_pulse_organization/response-v1.protocol.json)
preceded evaluation. Outward dyadic intervals contain the declared exact
rational preparation; their centers do not replace it. The
[retained response](../../docs/assets/collective_pulse_organization/response-v1.json)
completed all `256` steps without an evaluation error, covering the full
original-clock interval `[0,24]`. Every whole-time tube certifies receiver
winding zero. The verdict is therefore
`acute_winding_excluded_on_horizon`, not incomplete or unresolved evidence.
The [source archive](../../docs/assets/collective_pulse_organization/response-v1.source.zip)
retains the evaluated implementation and preparation.

Initial regional storage is `3/400=0.0075`; its final enclosure is contained
in `[0.0061989419859282,0.0061989419859330]`. The endpoint full-storage
enclosure contains the conserved exact value `14201/4000=3.55025`.
Thus exceeding the necessary phase-path budget and breaking the unseeded
family's reflection symmetry did not produce the target identity in this
control. The local fourth derivative did not supply that finite verdict;
the whole-time proof did.

The exclusion concerns this source, complete law, support and horizon.
It excludes neither later acquisition nor other admitted preparations.
There is no response-driven retry, extended horizon, retuning or added loss.
Even a positive acute-entry response would still need source-compatible
full-state evidence for Section 40's subsequent maintenance window; this
negative response supplies no such handoff.

`observe_sine_collective_pulse` in the shared
[partition owner](../../src/tnfr/physics/relational_sine_partition.py)
rebuilds these actual-state means, feedback and optional flat-phase jet.
It retains explicit integer contact-turn offsets and the signed remainder;
it does not simulate the declared finite response. The
[contract](../../docs/contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-collective-pulse-transfer)
and [usage example](../../docs/guides/relational/SINE_REGIONAL_DYNAMICS.md#collective-pulse-balance)
describe its detached evidence and availability.

### A structural alternative separates storage release from acquisition

The same law admits a different initial transfer sign without changing its
coefficients or waiting longer. Keep all receiver forms equal to `a`, all
leaf forms equal to `b`, and all receiver phases equal. Write `u=a-b>0`
and supply contact lifts `phi_i=gamma+eta_i`, with `mean(eta_i)=0`.
For a symmetric, nonconstant deviation multiset with
`abs(eta_i)<min(gamma,pi/2-gamma)` and `0<gamma<pi/2`,
`S=mean(sin eta_i)=0` and `C=mean(cos eta_i)<1`. Then

\[
h_c'(0)=\frac43u(C-1)\sin\gamma<0.
\]

This is an exact release from collective contact storage into its signed
remainder. It is not merely a different scalar bookkeeping observation:
the full law also forces an internal receiver response. Put
`s_i=sin(phi_i)` and let indices follow the C5 cycle. Initially,

\[
x_{r_i}'=\frac{s_i}{3},\qquad
\theta_{r_i}'=\frac u3,\qquad
\theta_{r_i}''=\frac{6s_i-s_{i-1}-s_{i+1}}9.
\]

The receiver starts with zero internal form and phase storage. Direct
differentiation of its edge sums gives

\[
F_R''(0)=\frac19\sum_i(s_{i+1}-s_i)^2,\qquad
V_R^{(4)}(0)=3\sum_i
 (\theta_{r_{i+1}}''-\theta_{r_i}'')^2.
\]

For phase storage all lower derivatives vanish: the initial relative
phases and their first rates are zero. Both displayed quantities are
strictly positive when `s` is nonconstant, since `6I-A_C5` has strictly
positive eigenvalues and preserves the constant/nonconstant decomposition.
Thus internal form grows at order `tau^2` and internal phase storage at
order `tau^4` for a sufficiently short nonzero interval. No acute winding,
directed full-cycle passage or finite-width handoff follows from that local
onset.

In particular, choose `0<rho<min(gamma,pi/2-gamma)` and the cyclic
deviations `(rho,-rho,-rho,rho,0)`. Reflection `i -> 3-i (mod 5)`,
including the matched private leaves, preserves this source. The complete
law preserves that symmetry by the
[same equivariance and uniqueness argument](SINE_REGIONAL_FORMATION.md#sine-conservative-source-geometry),
so the receiver cannot acquire nonzero winding
whenever no receiver edge is antipodal; in particular, acute unit winding
is excluded. Nevertheless this source has `h_c'(0)<0`
and both positive internal derivatives above. Permuting the same multiset
to `(rho,rho,-rho,-rho,0)` removes that particular reflection obstruction
while leaving the initial contact resultant, transfer rate and total
storage unchanged:

\[
H=\frac52u^2+5-(1+4\cos\rho)\cos\gamma.
\]

Removing a symmetry obstruction is not a formation theorem. This exact
comparison explains the next admission question: which ordered contact
geometry transfers into a directed receiver phase configuration while
retaining a source-compatible full-state maintenance budget? A negative
collective transfer rate or a growing internal store cannot answer it
alone. These conditional preparations are not an additional scheduled
response, parameter search or modification of the frozen control.

## 45. Complete contact feedback distinguishes cancellation from rigidity

<a id="sine-receiver-rigidity"></a>
<a id="sine-relative-phase-feedback"></a>

The relative-rate identity at the end of Section 44 gives an instantaneous
condition. Its persistence depends on both the receiver's internal current
and the actual contact response. The same conservative unit C5/private-leaf
law supplies the needed derivatives and admits a precise classification of
**exactly rigid** receiver geometry. A finite retained identity need not be
rigid, so this classification does not replace the broader acquisition and
retention question.

### The complete centered rows and their derivatives

Use `L=L_R`, the five-node cycle Laplacian, and let `P` remove the receiver
mean. In matched receiver/leaf order put

\[
\vartheta=P\theta_R,\qquad q=Px_R,\qquad r=x_R-x_Q,
\qquad u=\langle r\rangle,\qquad
\phi=\theta_Q-\theta_R,
\]

in continuous phase lifts. Let `s=S_C5(vartheta)` be the incoming internal sine
current and `f=sin(phi)` componentwise. The full normalized rows give

\[
q'=\frac13(s+Pf),\qquad
r'=\frac13s+\frac43f,\qquad u'=\frac43\langle f\rangle,
\]

\[
\boxed{\vartheta'=\frac13(Lq+Pr),\qquad
\phi'=-\frac13(Lq+4r).}
\]

All derivatives in this section use `tau=t/pi`. Differentiating the first
phase row and retaining the actual contact current gives

\[
\boxed{\vartheta''=\frac19\bigl[(L+I_5)s+(L+4I_5)Pf\bigr].}
\]

Define `L_cos(vartheta)` as the cycle Laplacian whose edge weights are
`cos(vartheta_j-vartheta_i)`. Since `s'=-L_cos(vartheta)vartheta'`, one more derivative is

\[
\boxed{\vartheta'''=\frac19\left[-(L+I_5)L_{\cos}(\vartheta)\vartheta'
 +(L+4I_5)P\bigl(\cos\phi\mathbin{\odot}\phi'\bigr)\right],}
\]

where `odot` denotes componentwise multiplication. None of these rows
assumes that a small regional storage, a zero measured rate or a
collective mean determines the environmental coordinates.

### Vanishing first and second rates do not imply a fixed geometry

Prepare receiver phases at a uniform unit twist, copy each phase to its
private leaf, and choose any nonzero centered form vector `q`. Supply
`r=u*1-Lq`; receiver and leaf forms can then be reconstructed with any
declared common form origin. Initially `s=0`, `phi=0`, and

\[
\vartheta'=0,\qquad \vartheta''=0,\qquad
\boxed{\vartheta'''=\frac19(L+4I_5)Lq\ne0.}
\]

Both factors on the centered subspace are invertible. In particular, for
`q=kappa*(1,-1,0,0,0)`, the exact third derivative is

\[
\vartheta'''=\kappa(22/9,-22/9,1,0,-1).
\]

The contact phase differences have rates `Lq-4u*1/3` at that instant
(the absolute leaf phase rates are `Lq-u*1`). Their subsequent
sine currents break the initial compensation and produce the cubic
receiver response. This is an initial derivative of the complete field,
not a finite-time trajectory estimate. It excludes the inference that
even simultaneous zero first and second relative rates prove rigidity.

### Exact rigidity has no additional moving compensation family

Suppose the receiver has a fixed relative phase geometry on a nonempty
open time interval, so `vartheta'=0` throughout that interval. The full finite
sine vector field is real analytic, its solutions exist for all real
time, and their storage is finite and conserved. The global bound
`|x_i'|<=1` and the linear phase row permit at most linear form growth
and quadratic lifted-phase growth on finite intervals, precluding finite-time
blowup in either direction. Consequently the
analytic identity `vartheta'=0` extends to the entire solution. In particular
`vartheta`, and hence `s`, are constant.

The first two vanishing phase rows imply

\[
Pr=-Lq,\qquad
Pf=-(L+4I_5)^{-1}(L+I_5)s,
\qquad q'=(L+4I_5)^{-1}s.
\]

Thus `q'` is constant. But the receiver form storage
`q^T Lq/2` is bounded above by the conserved full storage `H`; since
`L` is positive definite on centered vectors, `q` is bounded for all
real time. A nonzero constant `q'` is impossible. Therefore

\[
s=0,\qquad Pf=0,\qquad q=\text{constant},\qquad
\phi_i'=a_i-4u/3,\quad a=Lq,\quad\sum_i a_i=0.
\]

Every contact sine equals one common analytic function `m(tau)`.
For any pair of contacts the analytic identity

\[
0=\sin\phi_i-\sin\phi_j
 =2\sin\frac{\phi_i-\phi_j}{2}
    \cos\frac{\phi_i+\phi_j}{2}
\]

forces at least one factor to vanish identically: if one analytic factor
is nonzero somewhere, the other vanishes on an open interval and hence
everywhere. The corresponding continuous lifts either differ by a fixed
multiple of `2pi`, or sum to a fixed odd multiple of `pi`.

In the supplementary case, summing their phase rates gives
`a_i+a_j-8u/3=0`, forcing `u` to be constant. Since `u'=4m/3`,
this forces `m=0` identically. If `m` is not identically zero, all
contacts must instead agree modulo `2pi`. Their equal derivatives imply
that all `a_i` agree, hence `a=0` and `q=0`. Both form blocks are
uniform and all contact phases agree: this is precisely the moving
compatible family of Section 39, with its actual contact pendulum.

If `m=0` identically, each contact phase is a fixed integer multiple
of `pi`. Its zero derivative gives `a_i=4u/3` at every node, hence
`u=0`, `a=0` and `q=0`. All forms are equal throughout the full graph,
and every phase is stationary. Individual contacts can be aligned or
antialigned; these mixed-contact states are equilibria, not another moving
compensation mechanism.

Finally, at an acute cycle critical point `s=0` forces all oriented edge
sines to agree. Sine is injective on the acute interval, so unit winding
forces the uniform gaps `+2pi/5` or `-2pi/5`. Thus an exactly rigid
acute unit-winding receiver belongs either to the known uniform-block
moving family or to a static equilibrium with zero contact torques.
This classifies individual rigid trajectories, rather than only families
required to close for every supplied collective state.

### A retained second-order description exposes the same feedback

The contact row also gives the exact equation

\[
\boxed{\phi''=-\frac19
 [(L+4I_5)s+(L+16I_5)\sin\phi].}
\]

Together with the displayed equation for `vartheta''`, this is a closed
second-order description of four centered receiver phase coordinates and
five contact phases. It retains nine initial velocities, rather than
claiming a first-order closure from instantaneous phases alone. If
`v=vartheta'` and `c=phi'`, the original relative forms reconstruct exactly by

\[
Lq=Pc+4v,\qquad Pr=-Pc-v,\qquad
u=-\frac34\langle c\rangle.
\]

The inverse of `L` is restricted to the centered subspace. The two
conserved degree-weighted form and lifted-phase means recover the common
origins: if they are `m_x,m_theta`, the receiver form mean is
`m_x+u/4` and its phase mean is `m_theta-<phi>/4`. The leaf state then
follows from `x_Q=x_R-r` and `theta_Q=theta_R+phi`.

These eighteen relative coordinates plus the two origins carry the same
twenty fine form/phase coordinates. This is a support-specific instance of
the [exact form/phase representation](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence),
not an information loss, a new microscopic law or an additional research
branch. It makes explicit where the environmental response re-enters a
putatively compensated receiver.

### Consequence for the longer retention question

Exact entry from outside either rigid invariant family at a finite time
remains excluded by uniqueness. That fact does not rule out acute identity
with bounded internal evolution. In the compensated example with `u=0`,
the full storage is `V_5+13kappa^2`; sufficiently small nonzero `kappa`
gives `H<B`, where `B=5-4cos(3pi/8)` is the **acute-face** barrier.
The regional barrier then protects its already prepared
acute identity even though the nonzero third derivative proves it is not
rigid. This is maintenance of an initially present identity, not formation
from zero winding.

The prospective formation-to-retention requirement of a scaled duration
at least one is not closed by these derivative or rigidity results.
Its permissible targets can evolve internally. They still require actual
full-state entry, bounded accumulated relative transport and compatible
regional work over that duration. Instantaneous compensation is a useful
quantity to inspect, but it is not a substitute for those interval bounds.

In particular, each target edge `i->j` needs a continuous lift satisfying

\[
\left|\delta_{ij}(0)+\int_0^s[v_j(\tau)-v_i(\tau)]\,d\tau\right|
 <\frac\pi2\qquad\text{for every }0\le s\le1,
\]

with the declared winding sector and full-state uncertainty retained.
Cancellation only at the endpoint cannot establish this prefix condition.
The generic gap-acceleration bound of four gives a phase-error radius of
two at this horizon, already greater than `pi/2`; that estimate alone
cannot certify the required acute interval, even from zero initial relative
rate. The actual restoring and contact feedback, or existing validated
full-state enclosures, must supply tighter state-dependent control. These
are consequences of the selected complete sine law, not a proof that it
is the uniquely emerging law of TNFR.
