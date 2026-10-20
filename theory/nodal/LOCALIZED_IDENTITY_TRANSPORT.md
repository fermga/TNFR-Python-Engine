# Finite transport of a prepared localized sine-law pattern

**Scope:** a conditional full-state transport proof on supplied finite support.
The law, localization observation and preparation are specified here before any
trajectory evaluation. No trajectory, fitted response, formation source or
physical measurement is used. The [ontology contract](../EMERGENT_ONTOLOGY.md#localized-identity-transport)
separates prepared transport from formation, permanent motion and particle
identification; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns subsequent work.

<a id="localized-transport-law"></a>
## 1. Complete law, support and observation

Use the existing [normalized-sine completion](RESONANCE_FOUNDATIONS.md#reciprocal-exchange)
on the labeled cycle `C_N`, with `N` divisible by four, unit conductance,
unit geometric edge length and held capacities `nu_i=1`. Form `x` is signed
real, phase is circular, and beta is fixed at one. There is no input, clipping,
support event or selected operator word. With actual positive coefficients
`e,w`, the original structural clock `t` gives

\[
 \dot x=-\frac e2Lx+\frac{w}{2\pi}S(\theta),\qquad
 \dot\theta=\frac{w}{2\pi}Lx,\qquad
 S_j=\sin(\theta_{j+1}-\theta_j)+\sin(\theta_{j-1}-\theta_j).
\]

Both rows change under `tau=w*t/pi`. Put `A=L/2` and `sigma=pi*e/w`:

\[
 x'=-\sigma Ax+\tfrac12S(\theta),\qquad \theta'=Ax. \tag{1}
\]

The supplied background is `x=theta=0`. A small real phase lift about that
background is part of the preparation; the bound below keeps it in the same
chart. `z=x+i theta` is notation for two real state channels with beta one,
not a quantum amplitude or a new complex EPI. Its Euclidean norm measures
full-state departure from this background. These structural coordinates have
the fixed scale used by this law; no laboratory unit conversion is asserted.

Integer shifts and reflection preserve both complete rows, support and
capacities. Define distance to a real position `c` on the supplied unit-length
circle by `d_N(j,c)=min_{k in Z}|j-c+kN|`. This continuous position describes
an observation window on fixed sites. Fractional shifts of an interpolated
profile are not exact nonlinear graph symmetries.

The localization weights are `rho_j=|z_j|^2=x_j^2+theta_j^2`. They are
nonnegative diagnostics, not local physical energy. A single complex Fourier
mode has uniform rho, so motion of its form maxima cannot pass a concentrated
packet criterion. Position is retained in the fixed site reference, rather
than removed by a time-dependent relabeling.

## 2. One full-state reference and a prepared source class

Let `Q>=2`, `Q-1<N/4`, `k0=pi/2` and `q_m=2*pi*m/N`. For
`m=-(Q-1),...,Q-1`, set

\[
 b_m=1-\frac{|m|}{Q},\qquad
 S_2=\sum_m b_m^2=\frac{2Q^2+1}{3Q},\qquad
 F_Q(u)=\sum_m b_m e^{iq_m u}
       =\frac1Q\left[\frac{\sin(Q\pi u/N)}{\sin(\pi u/N)}\right]^2.
 \tag{2}
\]

Use the continuous limit at multiples of N. For amplitude `a>0`, prescribe
the reference history

\[
 z^*_j(\tau)=\frac{a}{\sqrt{NS_2}}
       e^{i(k_0j+\tau)}F_Q(j+\tau),\qquad 0\le\tau\le T. \tag{3}
\]

Distinct discrete frequencies give `||z*(tau)||_2=a` for every real tau.
The envelope center is `c(tau)=-tau mod N`. The carrier band omits frequency
zero, so the nominal real form and lifted phase means both vanish. A prepared
state is any full real state satisfying

\[
 \|z(0)-z^*(0)\|_2\le\epsilon. \tag{4}
\]

All sites and both channels may be perturbed independently subject to this
one joint bound. No restriction to an invariant Fourier subspace is imposed
on the actual nonlinear trajectory. The source already contains a localized
organization and a direction; neither is claimed to form autonomously.

The factor `exp(i tau)` mixes form and phase in the reference. It is a
specified prediction of internal evolution, not common phase-origin gauge
and not an exact symmetry of (1). Identity here means proximity to the
complete history (3), including that rotation, its amplitude and its location.
It does not mean a rigid full-state spatial translate at every time.

<a id="localized-transport-error"></a>
## 3. A whole-interval nonlinear error bound

The actual consensus tangent and the conservative comparison generator are

\[
 J_\sigma=\begin{pmatrix}-\sigma A&-A\\A&0\end{pmatrix},\qquad
 J_0=\begin{pmatrix}0&-A\\A&0\end{pmatrix}.
\]

`J_sigma` has a contractive Euclidean semigroup: its symmetric part contributes
`-sigma*x^T A x<=0`. `J_0` is skew, so its semigroup preserves norm. This
comparison does not remove the actual law's positive loss. The
[coupled conservative comparison](RESONANCE_FOUNDATIONS.md#spectral-rhythm-reuse)
has dispersion `omega(k)=1-cos k`, not the square root of a Laplacian
eigenvalue. The actual positive-loss tangent still has damping and its own
eigenvalues. On the prepared band, the conservative comparison gives

\[
 \omega(k_0+q)=1+\sin q,\qquad
 |\omega(k_0+q)-(1+q)|\le |q|^3/6.
\]

Consequently Parseval's identity and `|exp(iu)-exp(iv)|<=|u-v|` bound the
conservative tangent's difference from (3) by
`a*T*q_max^3/6`, with `q_max=2*pi*(Q-1)/N`. The difference between the
positive-loss and conservative tangents is at most `2*sigma*T*a`, since
`||A||<=2` and the positive-loss propagator is contractive.

For an oriented incidence matrix `D`, `||D||<=2` and the actual nonlinear
remainder is

\[
 R(\theta)=-\tfrac12D^T[\sin(D\theta)-D\theta],\qquad
 \|R(\theta)\|_2\le\tfrac43\|\theta\|_2^3. \tag{5}
\]

Indeed `|sin u-u|<=|u|^3/6`, and bounding each edge difference by
`2||theta||_2` gives (5). In the bootstrap region `||z||_2<=2a`, the
nonlinear Duhamel contribution is at most `(32/3)*T*a^3`. Initial error
propagates by at most epsilon under the same contractive tangent.

Define the dimensionless sufficient error

\[
 h=\frac\epsilon a+2\sigma T+
       \frac{T}{6}\left(\frac{2\pi(Q-1)}N\right)^3
       +\frac{32}{3}Ta^2. \tag{6}
\]

If `h<1`, continuity closes the bootstrap: at a hypothetical first exit
from `||z||<2a`, the preceding estimates instead give
`||z-z*||<=a*h` and `||z||<=a*(1+h)<2a`. Thus, throughout the entire interval,

\[
 \|z(\tau)-z^*(\tau)\|_2\le ah,\qquad
 a(1-h)\le\|z(\tau)\|_2\le a(1+h). \tag{7}
\]

The sine law is smooth, with linear form dependence and bounded sine currents,
so finite-time continuation itself is not the bootstrap hypothesis. If
`2a<pi/2`, each retained phase lift and every edge phase difference stay
away from the circular branch boundary. These sufficient estimates can fail
without excluding transport; failure is unavailability, not a negative theorem.
For the calculator's integer horizons `T>=1`, `h<1` already implies
`2a<sqrt(3/(8T))<pi/2`, so it also supplies this small-lift condition.

## 4. Uniform localization and distinct endpoint windows

For `1<R<N/2`, the elementary inequality `sin(pi*d/N)>=2d/N` for
`0<=d<=N/2` bounds

\[
 |F_Q(j-c)|\le\frac{N^2}{4Q\,d_N(j,c)^2}.
\]

For any fractional center, a unit distance shell contains at most two cycle
sites. Extending each tail to an infinite lattice bounds its fourth-power
sum by `2/[3(R-1)^3]`. As `S_2>=2Q/3`, the reference tail fraction satisfies

\[
 \frac{\sum_{d_N(j,-\tau)>R}|z^*_j(\tau)|^2}{a^2}
 \le H=\frac{N^3}{16Q^3(R-1)^3}. \tag{8}
\]

Combining (7) with the triangle inequality on the outside coordinates gives

\[
 \frac{\sum_{d_N(j,-\tau)>R}\rho_j(\tau)}
      {\sum_j\rho_j(\tau)}
 \le\left(\frac{\sqrt H+h}{1-h}\right)^2. \tag{9}
\]

This follows the same continuous full-state trajectory at all times, rather
than matching two unrelated peaks. The nonvanishing norm floor in (7)
prevents a vanishing trace from passing after normalization. A center is
specified by the prospective reference, not selected afterward among tied
maxima. The bound asserts concentration around that moving center; it does
not reconstruct a unique center from arbitrary partial observations.

If T is an integer, `T<=N/2` and `2R<T`, the radius-R windows at zero and
`-T` are disjoint. A stationary state cannot have over half its norm squared
in each of these disjoint windows. Reflection of the full preparation reverses
the direction under the same law. No support mutation or external actuation
is needed during the admitted interval.

<a id="localized-transport-protocol"></a>
## 5. Fixed analytical admission

The first case fixes the following inputs, without a trajectory search:

| Input | Declared value |
| --- | --- |
| Actual loss and exchange | `e=2^-20`, `w=1-2^-20`, beta one; these represented values sum exactly to one |
| Support and capacity | `C_1024`, unit lengths, conductances and held capacities |
| Carrier and envelope | `k0=pi/2`, `Q=16`, modes `m=-15,...,15` in (2) |
| Nominal joint amplitude | `a=2^-12` |
| Full-state initial error | `epsilon=a/1000` |
| Structural horizon | `T=256` in `tau=w*t/pi` |
| Localization radius | `R=96` in supplied edge-length units |
| Required identity error | `h<1/25` relative to the full-state reference norm |
| Required concentration | More than `24/25` of actual rho remains in the moving radius-R window |

The initial spatial localization and spectral direction are supplied. This
small positive loss is an explicit member of the existing family, not a value
selected by ontology or physical data. Its eventual dissipative relaxation is
compatible with a finite transport result. The initial uncertainty is a
mathematical norm ball, not a certified laboratory preparation.

Equations (6) and (8), even using the rational upper bound `pi<22/7`, give
`h<1/25` and `H<1/49`. Therefore (9) is strictly below

\[
 \left(\frac{1/7+1/25}{24/25}\right)^2
   =\frac{16}{441}<\frac1{25}.
\]

The signal norm stays above `24a/25`, full-state template error stays below
`a/25`, and initial/final radius-96 windows are disjoint after displacement
256. The conclusion applies to every state in (4) at every time in the
declared interval. It is a prepared finite transport and robustness theorem,
not an invariant moving family, indefinite stability or formation theorem.

## 6. Computation, reuse and remaining obligations

The [detached budget reader](../../src/tnfr/physics/_localized_identity_transport.py)
re-admits primitive inputs and the actual normalized model coefficients. It
evaluates only outward scalar bounds from this proof, not a trajectory,
sampled response or numerical eigensystem. The actual field remains owned by
the shared sine rate rows; this result installs no native execution law or
additional solver. The [controls](../../tests/physics/test_localized_identity_transport.py)
check the complete rows, clock, translation direction, nonlocalized normal
modes, analytic budgets and rejection domains separately.

The result retains a prepared envelope and internal evolution. Acquisition
from a class lacking that organization, interactions between transported
packets, a closed collective state, scaling under composition and physical
observations each need their own admission. The positive-loss
[recurrence boundary](RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
still applies to the actual finite model. A carrier Fourier index is not
primitive phase winding: this small-lift construction remains in winding zero.
No mass, charge, spin, quantum detection rule or constituent identification
is derived by these estimates.
