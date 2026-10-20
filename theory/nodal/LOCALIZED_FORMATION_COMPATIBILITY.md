# Mixed interaction and the compatibility of localized formation

**Scope:** exact instantaneous mixed pressure under the existing cycle law,
followed by a conditional two-carrier modulation calculation. The calculation
identifies cubic interaction channels, distinguishes temporal resonance from
spatial aliasing, and excludes one stationary bright-profile comparison.
This pair is an analytical control with a separate static source-absence
admission. It supplies no evaluated source, full-law envelope closure, formation event
or physical identification. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns admission of a preparation and any subsequent response evaluation.

## 1. Complete law, state and clock

Retain the [complete localized model](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-law)
on C1024: unit edge conductances and lengths, held unit capacities, beta one,
signed real form x and primitive circular phase with declared real lifts
theta. Inputs and support events are absent. Keep `e=2^-20`, `w=1-e`,
the normalized clock `tau=w*t/pi` and `sigma=pi*e/w>0`. Let D be the
oriented cycle incidence, `L=D^T D` and `A=L/2`. Both real rows are

\[
 x'=-\sigma Ax-\tfrac12D^T\sin(D\theta),\qquad
 \theta'=Ax.                                                     \tag{1}
\]

Primes denote tau derivatives. Conversion to the original t clock multiplies
both rows by `w/pi`; no laboratory-time identification is supplied. All
coefficients and phases below use this fixed normalized model and clock.
The bookkeeping variable `z=x+i*theta` retains two real channels; it does
not make EPI complex or identify carrier phase with primitive phase.
Writing `F_sigma` for (1), its nonlinear remainder is

\[
 R(\theta)=-\tfrac12D^T[\sin(D\theta)-D\theta],\qquad
 R_3(\theta)=\tfrac1{12}D^T(D\theta)^3.                            \tag{2}
\]

The original prepared encounter, its nominal interaction certificate and
its [strict-passage exclusion](LOCALIZED_ORGANIZATION_GEOMETRY.md#mobile-relative-rate-obstruction)
remain unchanged. The control below is selected analytically from velocity
compatibility, not obtained by retuning that evaluated source.

<a id="instantaneous-mixed-sine-pressure"></a>
## 2. Exact mixed pressure precedes a carrier approximation

Take two supplied full states on the same support, clock and lift convention.
Put `u=Dtheta_A`, `v=Dtheta_B`, with edge products understood componentwise.
The shared-state instantaneous defect is

\[
 \mathcal M=F_\sigma(z_A+z_B)-F_\sigma(z_A)-F_\sigma(z_B)
 =\left(2D^T\left[
 \sin\frac u2\sin\frac v2\sin\frac{u+v}2\right],\ 0\right).
                                                                    \tag{3}
\]

Indeed `sin(u+v)-sin(u)-sin(v)` equals
`-4*sin(u/2)*sin(v/2)*sin((u+v)/2)`. The form-loss and phase rows cancel
because they are linear on the supplied shared states. This does not
cancel loss from their evolved histories: those histories and the transport
of their nonlinear difference depend on the actual `F_sigma`.
Equation (3) is not the [endpoint superposition defect](LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-defect).

The leading mixed cubic form row and a global edgewise remainder are

\[
 \mathcal M_{3,x}=\tfrac14D^T[uv(u+v)],\qquad
 \mathcal M_x-\mathcal M_{3,x}=D^T q,
\]
\[
 |q_j|\le
 \frac{|u_jv_j(u_j+v_j)|}{96}
 \left(u_j^2+v_j^2+(u_j+v_j)^2\right).                            \tag{4}
\]

To prove the bound, telescope the product of three sines against the
product of their arguments, using `|sin a|<=|a|` and
`|sin a-a|<=|a|^3/6`. Multiplication by two and substitution
`a=u/2`, `b=v/2`, `c=(u+v)/2` give (4). There is no division by a phase gap,
so zero gaps and cancellation `u+v=0` retain their exact zero defect.
Since `||D^T||_2<=2`, the corresponding vector remainder is at most
`2*||q||_2`. Its form mean is zero because `1^T D^T=0`.

The signed interaction depends on retained edge phase information. A positive
local contribution is not a binding criterion, and summing contributions
does not remove the regional boundary-work or formation obligations.

<a id="equal-velocity-carrier-control"></a>
## 3. An exact equal-velocity carrier control

The conservative comparison has `lambda(k)=1-cos(k)` and group velocity
`-sin(k)` in the sign convention of the moving reference. Fix

\[
 k_1=\pi/4,\quad k_2=3\pi/4,\qquad
 \lambda_1=1-\sqrt2/2,\quad\lambda_2=1+\sqrt2/2,
\]
\[
 v=\sin k_1=\sin k_2=\sqrt2/2,\qquad
 d_1=\tfrac12\cos k_1=\sqrt2/4,\quad d_2=-\sqrt2/4.              \tag{5}
\]

Both carriers are exact characters of C1024, at modes 128 and 384.
They meet the leading relative-velocity condition without a carrier search.
Their opposite curvature signs already prevent transferring a single
focusing-profile argument to both channels. Equal velocity is only a
kinematic admission; it does not establish retention or acquisition.

For a positive bookkeeping scale epsilon, put
`X=epsilon*(j+v*tau)`, `S=epsilon^2*tau` and
`phi_i=k_i*j+lambda_i*tau`. The leading two-row reference is

\[
 z_1=B_1(X,S)e^{i\phi_1}+B_2(X,S)e^{i\phi_2},\qquad
 x_{\rm lead}=\epsilon\operatorname{Re}z_1,\quad
 \theta_{\rm lead}=\epsilon\operatorname{Im}z_1.                 \tag{6}
\]

No envelope, initial amplitude, source-error ball or finite horizon is
selected by (6). Periodic envelopes require period `epsilon*N` in X.
A line profile cannot be wrapped onto the cycle without its seam defect.
The two carriers here are distinct channels; two lobes in a single carrier
would instead require a justified observation of one total envelope.

## 4. Cubic labels, aliases and resonances

Freeze the envelope values when extracting the local carrier coefficients.
For label `(m,n)`, retain both

\[
 K_{mn}=(m+3n)\pi/4\pmod{2\pi},\qquad
 \Omega_{mn}=m\lambda_1+n\lambda_2.                              \tag{7}
\]

Spatial aliasing reduces K but never Omega. Write
`R_3(Im z_1)=sum c_mn*exp(i*(m*phi_1+n*phi_2))` for this frozen-envelope
calculation. Define

\[
 g_1=\frac{3-2\sqrt2}{16},\qquad
 g_2=\frac{3+2\sqrt2}{16},\qquad c=\frac18.                      \tag{8}
\]

The eight labels below and their negatives give all sixteen. The negative
coefficient is `c_(-m,-n)=conj(c_mn)` because this is a real form row.

| Label | Coefficient `c_mn` |
| --- | --- |
| `(1,0)` | `-i*(g1*abs(B1)^2+c*abs(B2)^2)*B1` |
| `(0,1)` | `-i*(g2*abs(B2)^2+c*abs(B1)^2)*B2` |
| `(3,0)` | `-i*(sqrt(2)-1)*B1^3/48` |
| `(0,3)` | `i*(sqrt(2)+1)*B2^3/48` |
| `(2,1)` | `-i*B1^2*B2/16` |
| `(1,2)` | `-i*B1*B2^2/16` |
| `(2,-1)` | `i*(sqrt(2)-1)*B1^2*conj(B2)/16` |
| `(-1,2)` | `-i*(sqrt(2)+1)*conj(B1)*B2^2/16` |

These follow directly by substituting
`theta=(B1*u-conj(B1)/u+B2*v-conj(B2)/v)/(2*i)` into (2), with spatial
shift multipliers `u -> exp(i*k1)*u`, `v -> exp(i*k2)*v`.
For example, the resonant cross coefficient at the first carrier is
`-i*|1-exp(i*k1)|^2*|1-exp(i*k2)|^2/16=-i/8`.
The pure fundamental coefficients are `-i*lambda_i^2/8`.
The mixed cubic defect has twelve labels: the four fundamentals and the
eight signed mixed third-order labels. Only the four pure third harmonics
are absent from that mixed defect; its fundamental coefficients retain only
the cross terms, whereas the table displays full R_3. Omitting conjugate
labels loses a real row.

For the conservative comparison, set
`Delta_mn=Omega_mn-lambda(K_mn)`. The complete gaps are

| Label pair | `Delta_mn` | `Delta_(-m,-n)` |
| --- | --- | --- |
| `(1,0)` and `(-1,0)` | `0` | `-2+sqrt(2)` |
| `(0,1)` and `(0,-1)` | `0` | `-2-sqrt(2)` |
| `(3,0)` and `(-3,0)` | `2-2*sqrt(2)` | `-4+sqrt(2)` |
| `(0,3)` and `(0,-3)` | `2+2*sqrt(2)` | `-4-sqrt(2)` |
| `(2,1)` and `(-2,-1)` | `2-sqrt(2)` | `-4` |
| `(1,2)` and `(-1,-2)` | `2+sqrt(2)` | `-4` |
| `(2,-1)` and `(-2,1)` | `-sqrt(2)` | `-2+2*sqrt(2)` |
| `(-1,2)` and `(1,-2)` | `sqrt(2)` | `-2-2*sqrt(2)` |

Only `(1,0)` and `(0,1)` are resonant. One direct check is that every cubic
label has odd `m+n`; `lambda(K)` is either `1-sqrt(2)/2` or
`1+sqrt(2)/2`. Equality with `(m+n)+(n-m)*sqrt(2)/2` forces one of those
two labels. Although `3*k1=k2`, the temporal gap is `2-2*sqrt(2)`, not zero.
All fourteen nonresonant gaps have magnitude at least `2-sqrt(2)`.
These are carrier gaps for the stated comparison, not full nonlinear
spectral gaps or positive-loss eigenfrequencies.

<a id="coupled-leading-envelope-boundary"></a>
## 5. Leading equations retain cross interaction and actual loss

Cancellation of the two resonant cubic terms, together with the common
transport in (5), gives the conditional leading system

\[
 i\partial_S B_i=d_i\partial_X^2B_i+
       (g_i|B_i|^2+c|B_j|^2)B_i
       -i\frac{\gamma\lambda_i}{2}B_i,\qquad j\ne i,
 \quad\gamma=\sigma/\epsilon^2.                                 \tag{9}
\]

The damping term comes from both carrier directions in
`-sigma*A*Re(z)`. A corresponding counterrotating correction is
`-i*gamma*conj(B_i)*exp(-i*phi_i)/4`; its conservative divisor is
`-2*i*lambda_i`. Keeping only the positive carrier loss would discard part
of the real form row. The interpretation of damping at this order requires
the declared scaling of gamma; fixed positive sigma cannot be removed by
sending epsilon to zero. Equation (9) is neither an installed TNFR law nor
a certified closure of (1).

For smooth periodic envelopes, or on the line with vanishing boundary flux,
the individual modulation masses satisfy

\[
 M_i=\int|B_i|^2\,dX,\qquad M_i'=-\gamma\lambda_i M_i.            \tag{10}
\]

Multiply (9) by `conj(B_i)` and take the appropriate real part after solving
for `B_i,S`. The real cubic multipliers contribute zero to `M_i'`; the
dispersion term is the boundary derivative
`2*d_i*partial_X Im(conj(B_i)*B_i,X)`. Thus the leading mixed term shifts
phase and can affect spatial evolution, but supplies no direct exchange
between these two masses.

Equation (9) is invariant under independent constant rotations
`B_i -> exp(i*alpha_i)*B_i`. Its resonant coupling therefore does not select
a relative uniform carrier phase or supply a phase-dependent exchange term.
These carrier rotations are not primitive common-phase gauge. The full sine
law, its nonresonant reconstruction and higher-order errors have not been
proved to inherit this independent symmetry or these separate mass laws.
Their absence at leading order does not exclude higher-order phase effects,
profile reorganization or all finite-lived collective motion.

## 6. Nonresonant reconstruction and the remaining residual

Let `L_nr` be the fourteen nonresonant labels and define

\[
 C_3=\sum_{(m,n)\in L_{\rm nr}}
       \frac{c_{mn}}{i\Delta_{mn}}e^{i(m\phi_1+n\phi_2)}.
                                                                    \tag{11}
\]

For constant envelopes, `partial_tau-i*A` multiplies each term by its
displayed divisor, so (11) cancels exactly the nonresonant cubic form row.
If `S_3=sum_(L_nr)|c_mn|`, then pointwise
`|C_3|<=S_3/(2-sqrt(2))`. Its frozen fast derivative is bounded by
`sum_(L_nr)|Omega_mn*c_mn|/(2-sqrt(2))`. These finite bounds retain all
generated cubic labels; they are not bounds for later generated sidebands.

With the damping correction included in C, a candidate
`z_app=epsilon*z_1+epsilon^3*C` has the exact two-row residual

\[
 r_x=x_{\rm app}'+\sigma Ax_{\rm app}
                 +\tfrac12D^T\sin(D\theta_{\rm app}),\qquad
 r_\theta=\theta_{\rm app}'-Ax_{\rm app}.                         \tag{12}
\]

Slow derivatives of C, envelope variation across edges, actual loss and
generated cubic cross terms with C remain in (12). They cannot be removed
by the frozen-carrier cancellation. The sine tail has a separate bound:
if `|Dtheta_app|<=H` on every edge, then

\[
 \|R(\theta_{\rm app})-R_3(\theta_{\rm app})\|_2
       \le\frac{\sqrt N H^5}{120}.                              \tag{13}
\]

This follows from `|sin a-a+a^3/6|<=|a|^5/120` and `||D^T||_2<=2`.
For a split edge gap `a+b`, the exact correction in the cubic row is
`D^T[3*a^2*b+3*a*b^2+b^3]/12`; no generated term is discarded.
Together with declared envelope derivative bounds these identities give
inputs to a full residual estimate, not an evaluated estimate themselves.
The carrier calculation itself supplies no profile, source-error ball,
horizon or maintenance margin; section 8 admits a separate static source.
The [complete-law comparison owner](LOCALIZED_RETENTION_MECHANISMS.md#full-law-envelope-residual)
provides the methodology; its single-envelope constants do not transfer.

<a id="line-bright-pair-obstruction"></a>
## 7. A scoped stationary bright-pair obstruction

Consider only the conservative comparison (9) with gamma zero on the real
line, and a stationary real-profile ansatz
`B_i(X,S)=exp(-i*Omega_i*S)*f_i(X)`. Assume sufficient regularity, integrable
squares and fourth powers, vanishing integration-by-parts boundary terms,
and nonzero exponentially decaying tails on the usual branch
`Omega_i/d_i>0`. The second profile equation implies

\[
 \Omega_2\int f_2^2
 =-d_2\int(f_2')^2+g_2\int f_2^4+c\int f_1^2f_2^2>0
 \quad\hbox{if }f_2\not\equiv0.                                 \tag{14}
\]

Every right-hand term is nonnegative, and the fourth-power term is strictly
positive. But `d2<0` makes the stated exponential-tail branch require
`Omega2<0`. This contradiction excludes that stationary bright-bright
template. The tail-sign premise is explicit; (14) is not a theorem about
every possible decaying spectral solution.

At positive gamma, (10) already excludes a nonzero fixed-amplitude stationary
modulation profile under these boundary conditions. The conservative
obstruction is a comparison control, not permission to delete actual loss.
Neither result excludes periodic profiles with a nonzero background, dark
profiles, breathing or decaying organizations, different preparations, or
finite-lived acquisition under the complete C1024 law. In particular, the
line-tail premise does not transfer to the finite cycle.

## 8. Source admission and formation boundary

The carrier calculation selects no localized source and proves no entry into
a common identity. A prospective source must define its initial pattern
identities, absence of the proposed common relation with a margin, retained
relative and boundary state, and a discriminator against separate evolution
and linear overlap. The [formation contract](LOCALIZED_ORGANIZATION_GEOMETRY.md#common-pattern-formation-contract)
owns these obligations. Source-specific admission remains separate from the
carrier control and from the deferred prepared-envelope certificate.

<a id="initial-absence-source-control"></a>
### A bounded source with a common-window screen initially absent

This static class reuses only the Fejer normalization and spatial-tail proof
from the [transport owner](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-error).
It does not inherit that owner's pi/2 transport or encounter certificate.
Set `Q=16`, `a=2^-12`, `delta=a/1000`, and define

\[
 b_m=1-|m|/Q,\quad
 S_2=\sum_{|m|<Q}b_m^2=171/16,\qquad
 f_Q(j)=\frac{\sum_{|m|<Q}b_m e^{2\pi imj/1024}}
                   {\sqrt{1024S_2}}.
\]

For fixed supplied phases psi1 and psi2, with zero as one explicit choice,
let

\[
 v_{1,j}=a e^{i(\psi_1+\pi j/4)}f_Q(j),\qquad
 v_{2,j}=a e^{i(\psi_2+3\pi j/4)}f_Q(j-512),
 \qquad z(0)=v_1+v_2+r,\quad\|r\|_2\le\delta.
\]

Form and the supplied real phase lift are `Re z` and `Im z`; arbitrary
fine-state error modes remain. The full field is sufficient state.
Let P1 and P2 be unitary Fourier projections onto modes 113 through 143
and 369 through 399. Their bands are disjoint, omit zero, and give
`||v_i||_2=a`, `P_i v_i=v_i`, `P_i v_(3-i)=0`. Consequently

\[
 \|P_i z(0)\|_2\ge a-\delta=999a/1000>a/2.
\]

These projections are observations; retain their complement and both real
rows. The nonlinear dynamics has not been proved to close on these bands,
their intensities or their centers.

Let W(c,96) denote the cycle ball centered at a real observation position c.
The common-window screen asks for one such ball containing at least 3/4 of
each projected squared norm, with both projected norms at least a/2. It is
a geometric prerequisite for a possible identity, not a sufficient definition
of organization or a formation discriminator by itself.

The source centers are 0 and 512. The existing spatial-tail inequality gives

\[
 \frac{\|\mathbf1_{W(c_i,96)^c}v_i\|_2^2}{a^2}
 \le b:=\frac{1024^3}{16\cdot16^3\cdot95^3}
       =\frac{16384}{857375}<\frac1{49}.
\]

For every c, at least one distance to the antipodal centers is at least
256, exceeding `96+96`. Its common window is disjoint from at least one
source window. Contractivity of P_i and the triangle inequality then imply,
for that i,

\[
 \frac{\|\mathbf1_{W(c,96)}P_i z(0)\|_2^2}{\|P_i z(0)\|_2^2}
 \le\left(\frac{\sqrt b+1/1000}{1-1/1000}\right)^2
 <\left(\frac{1007}{6993}\right)^2<\frac1{36}.
\]

Thus throughout this source ball and for every proposed common center,
at least one required fraction fails by more than `3/4-1/36=13/18`.
Both norm floors hold, so absence is not inferred from a vanishing signal.
The proof covers fractional centers by the cycle triangle inequality.

Primitive phase admission is also explicit. Since
`||f_Q||_infinity=2/sqrt(171)<2/13`,

\[
 \|z(0)\|_\infty
 \le a\left(4/\sqrt{171}+1/1000\right)<a/3,\qquad
 |D\theta(0)|<2a/3=1/6144<\pi/2.
\]

Orthogonality gives `||z(0)||_2<3a/2`. Nominal means vanish, but arbitrary
error means remain: `|mean z(0)|<=delta/sqrt(1024)=a/32000`. No projection
onto zero conserved means is allowed. These are initial bounds, not
continuing phase, norm or localization certificates.

This is a bounded mathematical preparation and absence result, not a
materialized source, frozen response protocol or demonstrated path to the
target. Entry into the screen would still require a retained relation,
nontrivial lifetime and a continued-response control beyond linear overlap.
A reorganization outside these fixed bands may require another identity;
failure of this screen would not exclude all common organizations.

<a id="single-band-information-boundary"></a>
### Information retained by a single total envelope


For a single carrier, the total-envelope cubic retains different information.
Writing `N(B)=|B|^2*B`, any chosen sum satisfies the exact algebraic identity

\[
 N(B_1+B_2)-N(B_1)-N(B_2)
 =2|B_1|^2B_2+2|B_2|^2B_1+B_1^2\overline B_2+B_2^2\overline B_1.
                                                                    \tag{15}
\]

Here all terms share the same carrier, so relative phase remains in the
total cubic response. Equation (15) does not assign unique lobe identities
or closed equations to B1 and B2. Deriving a regional interaction or an
acquisition discriminator still requires retained state, boundary work and
the complete-law residual; the two-carrier control supplies none of them.

The [same-band source and work owner](LOCALIZED_SINGLE_BAND_WORK.md)
admits a separate total-field preparation and derives exact regional rates
directly from both real rows. Its initial resource matching and necessary
concentration-prefix condition do not require an envelope closure or transfer
the two-band screen; a maintained common organization remains unproved.

The [static controls](../../tests/physics/test_localized_formation_compatibility.py)
check exact incidence coefficients, retained labels and aliases, mixed
pressure, leading loss, initial-absence margins and the scoped sign
obstruction independently.
They evaluate no prepared trajectory or reserved response. The reusable
[regional storage balance](LOCALIZED_RETENTION_MECHANISMS.md#cycle-local-storage-balance)
and [relative-state observation boundary](LOCALIZED_ORGANIZATION_GEOMETRY.md#band-intensity-rate-nonclosure)
remain necessary when a later source and common identity are selected.
