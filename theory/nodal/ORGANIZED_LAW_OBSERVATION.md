# Constitutive law and unknown collective initialization

<a id="organized-law-observation-admission"></a>
## Analytical scope and result

The [record-only forecast](ORGANIZED_OBSERVED_FORECAST.md#organized-observed-forecast-result)
predicts under one supplied law. It does not select that law. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-law-state-identifiability)
asks whether the unchanged joint observations distinguish the sine law from
its retained analytic countermodel when initialization is unknown under each.

This owner establishes the alternative's own carrier, hidden-state error,
finite identity and numerical growth bounds. It also proves a local
obstruction: two distinct admitted initial phase gaps under the two endpoint
laws have exactly the same two ideal frozen-current quadratures. An admitted
adjacent gap can also match the three ideal receiver currents. Neither local
test uniformly selects the law. The complete 150-by-35 formal sampled map
is injective, but a direction with exact low-order cancellations obstructs
every global scalar or positive block bootstrap using the retained unconstrained
coefficient and remainder bounds. These results establish neither separation
nor overlap of the complete noisy nonlinear record sets.

A constrained continuation uses the actual shared harmonic phase. Two
long-ramp contrasts cancel the fundamental without dividing by the short
plateau duration, and their ideal joint signal has a uniform positive
phase-circle margin. A separate heat bridge through the existing form
readings bounds historical carrier phase-increment error below `2.8e-8`.
Kernel magnitudes and a combined rephasing/error certificate remain open.

| Item | Retained analytical scope |
| --- | --- |
| Support/state | Unit `C9 x C5`, nodes `9*c+j`, 90 edges, signed form and primitive circular phase; held unit capacities |
| Two laws | Endpoints `lambda=0,1` of the existing potential/current family below; bounds hold uniformly for `0<=lambda<=1` |
| Initial class | `abs(x_i)<=X=1/100`; `theta=psi+R*Phi+z`, `psi_j=2*pi*(j-4)/9`, child-zero hidden means, `abs(z_i)<=1e-6`, `||u||2<=31/10^9`, `||z||2<=601/10^10`; principal collective gaps at most `7/5`, winding in `{-1,0,1}` |
| Input/clock | Original fifteen ramps and plateaus, exact command, `tau=(1023/1024)*t`, local time `v=tau-200`, horizon `T=16` |
| Actual observations | Original thirty five-form readings, arbitrary coordinate errors at most `eta=1e-14`; no primitive phase or initial-state label is observed |
| Information result | Local frozen-current collisions, finite formal rank 35 and the unconstrained-bootstrap obstruction; constrained long-ramp contrasts and record-conditioned historical phase transport |
| Outstanding distinction | Separation or overlap of the actual finite joint record sets, including each complete nonlinear history and its own source error |
| Evidence | Analytical proofs and independent static arithmetic/field controls; no fit, response trajectory, sampled coefficient design, physical data or frozen producer is evaluated |

The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization)
requires retained constituent and boundary information. An invariant carrier
or an identity-preserving containing class is not an acquisition theorem.
Support and the input remain supplied; no operator occurrence, new event,
physical constituent or uniquely fundamental law is inferred.

## 1. The complete alternative on the present support

Retain the [current/potential owner](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-complete-law):

\[
 V_\lambda(a)=1-\cos a+\lambda\frac{(1-\cos9a)^2}{6\,9^4},\qquad
 j_\lambda(a)=\sin a+\frac\lambda{2187}
                  (\sin9a-\tfrac12\sin18a).
 \tag{1}
\]

With `L_f=I5 tensor L9+L5 tensor I9`, `S_lambda,i=sum_(k~i) j_lambda(theta_k-theta_i)`
and `gamma=1/(1023*pi)`, the full rows are

\[
 x'=(-L_fx+\gamma S_\lambda(\theta))/4+Rf,\qquad
 \theta'=\gamma L_fx/4.
 \tag{2}
\]

The signed nodal pressure is specified by the original-clock form row;
both rows undergo the same clock conversion. The phase channel is not an
independently supplied synchronization law. Odd periodic current preserves
the global form and lifted-phase sums for neutral input. Its own storage is

\[
 H_\lambda=\tfrac12x^TL_fx+\sum_{e}V_\lambda(\delta_e),\qquad
 H_\lambda'=-\|L_fx\|_2^2/4+(L_fx)^TRf.
 \tag{3}
\]

Reciprocity cancels the current/phase work exactly. Storage values and
preparation costs remain law-dependent; (3) transfers no earlier resource
or acquisition verdict.

The shared derivative bounds give

\[
 J_\lambda=1+\lambda/1458,\quad
 L_\lambda=1+2\lambda/243,\quad
 B_\lambda=1+\lambda/9
 \tag{4}
\]

as global bounds for `|j|`, `|j'|`, `|j''|`. The full lifted field is globally
Lipschitz on the fixed finite support, and the prescribed piecewise-smooth
input gives one continuous state history through its switches.

Write `P=RR^T/9`, `Q=I-P`, `x=RU+u` and `theta=psi+R*Phi+z`.
Internal helical currents cancel by oddness and periodicity. Same-slot
outer currents are constant within a child at `z=0`. Thus `u=z=0` is an
exact invariant carrier for both laws and every uniform-child input:

\[
 U'=(-L_5U+\gamma S_{\lambda,5}(\Phi))/4+f,\qquad
 \Phi'=\gamma L_5U/4.
 \tag{5}
\]

The full stiffness on this carrier is

\[
 K_\lambda(\Phi)=j_\lambda'(2\pi/9)(I_5\otimes L_9)
                    +L_5^{j_\lambda'(\Phi)}\otimes I_9.
 \tag{6}
\]

It commutes with P and Q, so the collective/hidden tangent blocks still
decouple. This is not a transverse-stability theorem. At internal gaps
`2*pi/9` the deformation's first relevant derivatives vanish, but at the
outer `2*pi/5` gaps they generally do not. The complete `C9 x C5` tangent
is therefore **not** matched by the old C9 target construction.

The original command removal remains exact: `Y=U-z_cmd`, `Psi=Phi-H`,
with `H'=gamma*L5*z_cmd/4`, gives

\[
 Y'=-L_5Y/4+\gamma S_{\lambda,5}(\Psi+H)/4,\qquad
 \Psi'=\gamma L_5Y/4.
 \tag{7}
\]

This reuses the waveform and coordinates, not the sine-specific forecast gain.

## 2. Own-law hidden transport, identity and growth

Let `g=1/3000`, `a0=31/10^9`, `b0=601/10^10`, and `s=2*g*T`.
The hidden form row contains `Q S_lambda(psi+R*Phi+z)`, which vanishes at
zero z and has Euclidean Lipschitz bound `8*L_lambda`. The graph Laplacian
has norm at most 8. Conservatively bound both coupling rows with `8*L_lambda`
and reuse the [shared integral majorant](../../src/tnfr/physics/_organized_electrical_source.py):

\[
 A_\lambda=\frac{a_0+sL_\lambda b_0}{1-(sL_\lambda)^2},\qquad
 Z_\lambda=\frac{b_0+sL_\lambda a_0}{1-(sL_\lambda)^2}.
 \tag{8}
\]

All denominators are positive at the fixed corner. These are full nonlinear
hidden bounds under each law, not transported sine-source states.

On an outer edge, the nine hidden gap differences have zero sum. Taylor
expansion of the actual current cancels the linear correction and bounds
the mean remainder by `B_lambda/18` times their squared sum. External
Laplacian norm at most 4 then gives a mean-form forcing bound

\[
 \rho_\lambda=gB_\lambda Z_\lambda^2/9.
\]

Compare each fine history with its own carrier starting at its actual
initial means. Heat contraction and `||D S_lambda,5||2<=4*L_lambda` give,
with `k=gT`,

\[
 E_U\le T\rho_\lambda+L_\lambda k E_\Phi,\quad
 E_\Phi\le k E_U,\qquad
 \boxed{E_\lambda=\frac{T\rho_\lambda}{1-L_\lambda k^2}}.
 \tag{9}
\]

Consequently `E_U<=E_lambda` and `E_Phi<=k*E_lambda` on the entire horizon.
Unknown means are proof initializations, not new observation inputs.

For finite identity, remove the form command at the fine level. With
`||S_lambda||infinity<=4*J_lambda`, the same exact integrated phase identity
now yields

\[
 M_\lambda=X+gJ_\lambda T,\qquad
 F_\lambda=2gX+2g^2J_\lambda T.
 \tag{10}
\]

Substitute F_lambda for the drift in the original internal-positive,
internal-acute, outer-acute and collar margins, with `zeta=1e-6`,
`A=7/5`, `epsilon=1/10`. Their explicit expressions are

\[
 2\pi/9-2\zeta-2F_\lambda,\quad
 \pi/2-2\pi/9-2\zeta-2F_\lambda,\quad
 \pi/2-A-\epsilon-2\zeta-2F_\lambda,\quad
 \pi/2-2\zeta-4F_\lambda.
 \tag{11}
\]

All are strictly positive. This preserves each initially admitted marked
winding and collar throughout this word and tail, without proving that
the alternative acquired the state from the earlier preparation.

For numerical reuse, the carrier Jacobian's quadratic form has cross term
`gamma*a^T(L5-K_lambda)*b/4` and negative form-diffusion term. Since
`||L5-K_lambda||2<=4*(1+L_lambda)`, its identity-metric logarithmic norm is
at most `gamma*(1+L_lambda)/2`. For every `lambda in [0,1]`, using `pi>3`,

\[
 \mu_2\le\frac{244}{745767}<\frac1{3000}.
 \tag{12}
\]

The existing growth allowance is thus re-admitted. Actual-law Taylor
derivatives, Picard tubes and source corrections must still be rebuilt;
the frozen sine integrator consumer is unchanged.

At `lambda=1`, the exact bounds above imply

| Quantity | Rationally controlled bound |
| --- | ---: |
| Hidden form A_1 | `<3.17e-8` |
| Hidden phase Z_1 | `<6.05e-8` |
| Own-carrier mean-form error E_1 | `<2.42e-18` |
| Fine endogenous phase drift F_1 | `<1.023e-5` |
| Outer acute margin | `>0.06997` |

If two fine histories under the two laws admit the same raw record, their
own carriers satisfy, at every sample simultaneously,

\[
 \|U_0(t_j)-U_1(t_j)\|_\infty\le
 \delta=2\eta+E_0+E_1.
 \tag{13}
\]

This is a necessary joint condition. It proves neither record separation
nor realizability of independently chosen sample boxes.

<a id="organized-law-local-quadrature-obstruction"></a>
## 3. Exact local ambiguity after removing the unknown phase

The [original finite observer](ORGANIZED_FINITE_OBSERVATION.md#4-quadratures-circular-geometry-and-the-initial-winding)
uses three command levels to recover two quadratures. First consider its
ideal *frozen initial current* references, before temporal and reading errors.
For the interface `c -> c+1`, let a be its initial gap. The receiver's
reference current at command level q is `K-j(a-q)`, where the unmodulated
neighbor contribution K cancels from the two contrasts. For `epsilon=1/10`,
the normalized contrasts are

\[
 C_j(a)=\frac{j(a+\epsilon)-j(a-\epsilon)}{2\sin\epsilon},\qquad
 S_j(a)=\frac{2j(a)-j(a+\epsilon)-j(a-\epsilon)}{2(1-\cos\epsilon)}.
 \tag{14}
\]

These have exactly the orientation and scaling of the original O,V
statistics. In particular `(C_0(a),S_0(a))=(cos a,sin a)`. For the alternative,

\[
 \begin{aligned}
 C_1(a)&=\cos a+
 \frac{\sin(.9)\cos9a-\tfrac12\sin(1.8)\cos18a}{2187\sin(.1)},\\
 S_1(a)&=\sin a+
 \frac{(1-\cos(.9))\sin9a-\tfrac12(1-\cos(1.8))\sin18a}
      {2187(1-\cos(.1))}.
 \end{aligned}
 \tag{15}
\]

**There exist distinct gaps a and b, both of magnitude less than 1/2,
with `(C_1(a),S_1(a))=(cos b,sin b)`.** Thus even exact ideal local
quadratures need not distinguish these two laws when the phase is unknown.

To prove this, put `F(a)=C_1(a)^2+S_1(a)^2-1`, `r=cos(.9)>0` and
`c9=sin(.9)/(2187*sin(.1))>0`. At zero, `S_1=0` and
`C_1=1+c9*(1-r)>1`, so `F(0)>0`. At `a=pi/9`,

\[
 S_1(a)=\sin a,\quad C_1(a)=\cos a-q,\quad
 q=c9(1+r)=\frac{\sin(.9)+\tfrac12\sin(1.8)}{2187\sin(.1)}.
\]

The elementary bound `sin(.1)>599/6000` gives
`0<q<9000/(2187*599)<1/100`. Hence
`F(pi/9)=q*(q-2*cos(pi/9))<0`. Continuity supplies
`a_* in (0,pi/9)` with F zero, without locating a root numerically.

For every a in that interval,
`C_1(a)>=cos(pi/9)-q>1-8/81-1/100>8/9`, using `pi<4`.
There is therefore a unique principal b with the matching unit phasor.
Since `cos(1/2)<=337/384<8/9`, it satisfies `|b|<1/2`.
Also `a_*<pi/9<1/2`, inside the original collective gap prior.

The matching phases are distinct. If `b=a_*`, both perturbations in (15)
would vanish. Writing `x=cos(9*a_*)`, the sine correction, with
`sin(9*a_*)>0`, forces `x=1/[2*(1+r)] in (0,1/2)`. But then the cosine
correction is `c9*(x+r-2*r*x^2)>0`, a contradiction.

Each gap embeds in an admitted primitive initial state: choose `U=u=z=0`
and `Phi=(0,d,3*d/4,d/2,d/4)` for d equal to a_* or b. Its outer gaps are
`(d,-d/4,-d/4,-d/4,-d/4)`, with winding zero; the internal helical cycles
retain winding one. The nuisance neighbor current cancels in the selected
contrasts. The other interfaces and the absolute current baseline need
not agree. Finite plateau increments also contain endogenous drift and
integration error. This witness therefore excludes the **local compressed
statistic** as a uniform selector; it is not a common complete record,
an acquisition witness or a physical observation.

### Retaining the local current baseline does not suffice either

The neighboring gap can realize the needed baseline, rather than treating
K as a free unphysical offset. At a circle collision write
`kappa=S_1(a_*)-j_1(a_*)=sin b-j_1(a_*)`. Put
`s9=(1-cos(.9))/(2187*(1-cos(.1)))`. The bounds
`1-cos(.9)<81/200` and `1-cos(.1)>1199/240000` give `s9<1/26`.
Since `1+cos(.9)<2`, (15) and (4) imply

\[
 |\kappa|<3/26+1/1458<1/8,\qquad
 j_1(1/2)\ge\sin(1/2)-1/1458>1/8.
\]

Continuity and oddness supply `d in (-1/2,1/2)` with `j_1(d)=-kappa`.
Choose the initial outer gap cycles

\[
 (b,0,-b/3,-b/3,-b/3),\qquad
 (a_*,d,-(a_*+d)/3,-(a_*+d)/3,-(a_*+d)/3)
\]

for sine and the alternative, respectively. They sum to zero and obey
the original gap bounds. Set all forms and hidden deviations to zero.
Receiver 1 now has `K_0=0` and `K_1=-kappa`. Equal O,V contrasts force
the three edge-current differences to equal their baseline difference
`j_1(a_*)-sin b=-kappa`. Therefore the complete receiver currents
`K_0-j_0(b-q)` and `K_1-j_1(a_*-q)` agree at every `q=0,+epsilon,-epsilon`.
This strengthens the local obstruction within admitted primitive states;
other receivers and the continuous dynamics remain unconstrained by it.

## 4. The complete formal harmonic information remains distinct

For a cross-law carrier difference `u=Y_0-Y_1`, freeze each initial edge
phasor in (7) while retaining the prescribed command `h=B^T H`. The formal
heat response has five initial form coefficients and sine/cosine pairs
on five edges for each frequency `n in {1,9,18}`: 35 real coefficients.
Its equation is

\[
 V'=-L_5V/4-B\sum_{n\in\{1,9,18\}}
           [b_{s,n}\odot\cos(nh)+b_{c,n}\odot\sin(nh)].
 \tag{16}
\]

The fundamental pair is the difference of the two initial phasors scaled
by `gamma/4`. The ninth pair is the alternative's phasor scaled by
`-gamma/(4*2187)`, and the eighteenth by `gamma/(8*2187)`.
Their circle norms and shared underlying phase are constraints; treating
these formal coefficients as free only enlarges the comparison.

The formal coefficient map is injective if the entire continuous V history
is supplied. Indeed, a zero history forces zero initial form and zero
divergence of the edge currents. On C5 all currents then equal one
circulation. During a marked-child ramp three unactuated edges have constant
command, so this circulation is constant. Each of its two actuated edges
has a trigonometric polynomial of frequencies 1,9,18 equal to that constant
on an open interval of command values. Independence of those sine/cosine
functions and the constant forces all six coefficients on that edge to
vanish. Every edge is actuated somewhere in the word, proving injectivity.

This theorem neither supplies continuous observations in the actual protocol
nor proves rank or conditioning of its 150-row sampled matrix. It concerns
the formal linear response, before nonlinear endogenous phase drift.

For the actual pair, let `B_n=max(||b_s,n||infinity,||b_c,n||infinity)` and
`D=4*g*X+2*g^2*J_1*T`, bounding each endogenous edge-phase drift.
Here the carrier has only two incident outer currents, so
`||S_lambda,5||infinity<=2*J_1` and `||Y||infinity<=X+g*J_1*T/2`.
Integrating its live phase row gives coordinate drift at most
`2*g*X+g^2*J_1*T`, hence the stated edge bound D. This is separate from
the fine-state drift F_lambda in (10).
The fundamental difference remainder and the two one-law higher-harmonic
remainders give

\[
 u'=-L_5u/4-B\sum_n[b_{s,n}\cos(nh)+b_{c,n}\sin(nh)]+r,\quad
 \sum_c r_c=0,
\]
\[
 \|r(v)\|_\infty\le
 3D(B_1+9B_9+18B_{18})+g^2\int_0^v\|u(s)\|_\infty\,ds.
 \tag{17}
\]

For the fundamental term, use the original complex-phasor difference
identity and `|d_0-d_1|<=2*g*integral||u||infinity`. For a higher harmonic,
`|exp(i*n*d)-1|<=n*|d|`; its scaled phasor norm is at most `sqrt(2)*B_n`.
Two incident edges and `2*sqrt(2)<3` yield (17).
The [zero-sum heat integral bound](ORGANIZED_OBSERVED_PREDICTION.md#3-a-difference-remainder-rather-than-an-absolute-model-bias)
of 6 therefore implies

\[
 \sup\|u-V\|_\infty\le
 18D(B_1+9B_9+18B_{18})+6g^2T\sup\|u\|_\infty.
 \tag{18}
\]

The old 15-column gain does not certify (16)-(18). A finite joint test must
retain these harmonic blocks, their constitutive phase constraints, the two
own-law source errors and every sample correlation. The finite-rank and
relaxation results below distinguish information in the formal design from
what this global remainder can certify about the nonlinear pair.

<a id="organized-law-finite-formal-rank"></a>
## 5. Exact rank of the actual finite formal design

**The 150-by-35 sampled map of (16), at the original thirty observation
times, has column rank 35.** This is an exact statement about the formal
linear heat response. It supplies neither a quantitative inverse bound nor
a noisy or nonlinear law discriminator. No sampled matrix is evaluated in
the proof.

### The recorded plateaus and first ramps determine scalar moments

Suppose a coefficient vector produces zero V at every recorded time.
The first reading gives `V(0)=0`, eliminating all five initial-form
coefficients. Write `A=L5/4`. On each plateau its forcing is a constant
vector f, and the two recorded endpoints satisfy

\[
 0=e^{-hA}0+\left(\int_0^h e^{-sA}\,ds\right)f.
 \tag{19}
\]

The integral matrix is positive definite: its eigenvalues are h for the
constant mode and `(1-exp(-h*mu))/mu>0` for every positive eigenvalue mu of A.
Thus f is zero on each of the fifteen plateaus. At command zero, this
means that all baseline edge currents `sum_n b_s,n,e` equal one circulation,
because the kernel of the oriented C5 incidence matrix is the constant
edge vector.

For an edge e define its command-dependent increments

\[
 E_e(q)=\sum_n b_{s,n,e}(\cos(nq)-1),\qquad
 O_e(q)=\sum_n b_{c,n,e}\sin(nq),\qquad n\in\{1,9,18\}.
 \tag{20}
\]

When a child is marked, only its two incident edges have nonzero commands,
with opposite signs. The other three edges retain the baseline circulation.
Zero divergence at the two nonzero plateau levels therefore gives
`E_e(epsilon)+O_e(epsilon)=E_e(epsilon)-O_e(epsilon)=0` on each incident edge.
Every edge is incident to a marked child, so

\[
 E_e(\epsilon)=O_e(\epsilon)=0\quad\hbox{for every edge e}.
 \tag{21}
\]

Every child's first ramp starts at the recorded endpoint of its zero-level
plateau and ends at the recorded beginning of its positive-level plateau.
Both V values are zero. This includes child 4; its first ramp ends before
the last reading. With local ramp time `t in [0,1]`, put
`q(t)=epsilon*t^2*(3-2*t)`. After subtracting the zero-divergence baseline,
the forcing is supported on the two incident incidence columns.

The two distinct nonzero eigenvalues of A are

\[
 \mu_1=(5-\sqrt5)/8,\qquad \mu_2=(5+\sqrt5)/8.
 \tag{22}
\]

Each has a real two-dimensional Fourier eigenspace. The projections of
two adjacent incidence columns onto either eigenspace are linearly
independent: in complex Fourier coordinates their ratio is
`exp(plus_or_minus 2*pi*i*k/5)`, for k equal to 1 or 2, which is not real.
The zero ramp endpoint consequently forces each incident edge's weighted
forcing integral to vanish separately for each mu.

Each edge is probed once from each endpoint in two child blocks. Their
first ramps use the same q(t) and reverse that edge's command sign.
Adding and subtracting the two integral equations therefore gives

\[
 \int_0^1 e^{-\mu_j(1-t)}E_e(q(t))\,dt=0,\qquad
 \int_0^1 e^{-\mu_j(1-t)}O_e(q(t))\,dt=0,
 \quad j=1,2.
 \tag{23}
\]

Only recorded endpoints and the known linear response between them were
used. Neither derivatives nor extra within-ramp observations are supplied.
Equations (21)-(23) reduce coefficient recovery on every edge to two
three-by-three homogeneous systems. Their common matrix construction is

\[
 M_f=\begin{pmatrix}
 f_1(\epsilon)&f_9(\epsilon)&f_{18}(\epsilon)\\
 \int_0^1 w_1(t)f_1(q(t))dt&\int_0^1 w_1(t)f_9(q(t))dt&
       \int_0^1 w_1(t)f_{18}(q(t))dt\\
 \int_0^1 w_2(t)f_1(q(t))dt&\int_0^1 w_2(t)f_9(q(t))dt&
       \int_0^1 w_2(t)f_{18}(q(t))dt
 \end{pmatrix},\qquad w_j(t)=e^{-\mu_j(1-t)},
 \tag{24}
\]

using either `f_n(q)=sin(nq)` or `f_n(q)=1-cos(nq)`. The latter differs
from (20) only by a common minus sign. Both matrices are nonsingular, as
the following argument proves without computing their entries.

### Strict independence of both trigonometric triples

For `|z|<pi`, the analytic function `h(z)=z*cot(z)` has expansion

\[
 h(z)=1-\sum_{m\ge1}a_m z^{2m},\qquad
 a_1=1/3,\qquad
 (2m+1)a_m=\sum_{k=1}^{m-1}a_k a_{m-k}>0\quad(m\ge2).
 \tag{25}
\]

The recurrence follows directly from `z*h'=h-h^2-z^2`; the first poles
are at plus and minus pi. Thus for every positive d,

\[
 H_d(\sigma)=\sqrt\sigma\cot(d\sqrt\sigma)
 =\frac1d-\sum_{m\ge1}a_m d^{2m-1}\sigma^m
 \quad\Longrightarrow\quad H_d'<0,\quad H_d''<0
 \tag{26}
\]

on `0<d*sqrt(sigma)<pi`, where the derivatives are with respect to sigma.
Termwise differentiation is valid inside that convergence domain.

First take `f_n(q)=sin(nq)`, with `0<q<=epsilon=1/10` and n equal to
1, 9 or 18. Every f_n is positive because `18*epsilon=1.8<pi`.
For `sigma=n^2`, its logarithmic derivative and normalized second
derivative are `f_n'/f_n=H_q(sigma)` and `f_n''/f_n=-sigma`.
The first two-column Wronskian is negative by the strict decrease in
(26). Factoring the positive functions from the three-column Wronskian
leaves

\[
 \det\begin{pmatrix}
 1&1&1\\
 H_q(1)&H_q(81)&H_q(324)\\
 -1&-81&-324
 \end{pmatrix}\ne0,
 \tag{27}
\]

because strict concavity makes the three points on the graph of H_q
noncollinear.

For `g_n(q)=1-cos(nq)=2*sin(nq/2)^2`, every g_n is also positive and

\[
 g_n'/g_n=H_{q/2}(\sigma)=H,\qquad
 g_n''/g_n=(H^2-\sigma)/2.
 \tag{28}
\]

Again its first two-column Wronskian is negative. Invert the strictly
decreasing map in (26) locally as `sigma=sigma(H)`. Throughout the interval
between its three values,

\[
 \sigma''(H)=-H_{q/2}''(\sigma)/(H_{q/2}'(\sigma))^3<0,
 \qquad \frac{d^2}{dH^2}(H^2-\sigma(H))>0.
 \tag{29}
\]

The three-column Wronskian is therefore nonzero by strict convexity of
the last row in (28) as a function of the middle row.

These Wronskian conditions imply that each triple has a nonzero evaluation
determinant at any three distinct q values in `(0,epsilon]`. One direct
Rolle argument is to divide a proposed three-zero linear combination by
its positive first function. Put `r=f_9/f_1` and `s=f_18/f_1`; r' never
vanishes, and `(s'/r')'` is the three-column Wronskian divided by
`f_1^3*(r')^2`, hence never zero. Applying Rolle twice forces the
combination to be trivial. The evaluation determinant
has a constant nonzero sign on the connected domain of ordered triples.

### The two weighted moments retain that independence

Expanding (24) and interchanging its finite integrals gives exactly

\[
 \det M_f=\int_{0<t<u<1}
 [w_1(t)w_2(u)-w_1(u)w_2(t)]
 \det\begin{pmatrix}
 f_1(\epsilon)&f_9(\epsilon)&f_{18}(\epsilon)\\
 f_1(q(t))&f_9(q(t))&f_{18}(q(t))\\
 f_1(q(u))&f_9(q(u))&f_{18}(q(u))
 \end{pmatrix}\,dt\,du.
 \tag{30}
\]

The weight bracket is strictly positive: `mu2>mu1` and
`w2(t)/w1(t)=exp(-(mu2-mu1)*(1-t))` strictly increases. Also q(t) strictly
increases on `(0,1)`, so `0<q(t)<q(u)<epsilon` on the integration domain.
The evaluation determinant consequently has a constant nonzero sign.
Thus (30) is nonzero for both bases. Equations (21)-(24) force all thirty
edge coefficients to vanish. Together with the five initial forms this
proves the claimed rank 35.

This finite theorem strengthens the continuous formal injectivity result;
it does not make the local collision in section 3 into a complete-history
collision or transfer formal linear identifiability to nonlinear dynamics.
No determinant lower bound, inverse norm, finite-error separation or
numerical budget verdict follows from strict nonvanishing alone.

<a id="organized-law-relaxation-obstruction"></a>
## 6. Why the unconstrained global bootstrap cannot close the question

This obstruction concerns a specified proof method, not the actual laws'
record sets. It needs no coefficient-matrix evaluation, proposed inverse,
nonlinear history or choice of an unknown initial state.

### One formal direction with a uniformly small response

Set the five initial-form coefficients and all b_c coefficients to zero.
Repeat the same b_s coefficients on every edge:

\[
 (a_1,a_9,a_{18})=(1,-323/19683,20/19683),\qquad
 b=(1,323/19683,20/19683).
 \tag{31}
\]

The vector b lists their three block infinity norms. Direct rational
algebra gives

\[
 \sum_n a_n n^2=\sum_n a_n n^4=0,\qquad
 \sum_n|a_n|n^6=1+8721+34560=43282.
 \tag{32}
\]

The command-zero current is a constant circulation and has zero divergence.
At any instant only one child c is marked, with scalar command q satisfying
`|q|<=epsilon`. The two changed edge commands are opposite; cosine is even.
Consequently the complete formal forcing is exactly

\[
 F(q)(e_{c-1}-e_{c+1}),\qquad
 F(q)=\sum_n a_n(\cos(nq)-1).
 \tag{33}
\]

Taylor's theorem through degree five, with (32), gives for the entire
command range

\[
 |F(q)|\le f_*:=\frac{43282\epsilon^6}{720}.
 \tag{34}
\]

The dipole in (33) has zero sum and Euclidean norm sqrt(2). The smallest
nonzero eigenvalue of `L5/4` is greater than 1/3. Variation of constants,
including every change of marked child, therefore yields

\[
 \sup_v\|V(v)\|_\infty
 \le\sqrt2 f_*\int_0^\infty e^{-s/3}\,ds
 <\frac92 f_*
 =:m=\frac{21641}{80000000}.
 \tag{35}
\]

This bounds the complete formal history and hence every actual sample.
It does not assert a zero response; it is consistent with rank 35.
The retained elementary remainder coefficient is strictly larger:

\[
 a=18D>m,\qquad D=4gX+2g^2J_1T.
 \tag{36}
\]

All quantities in (31)-(36) are fixed rational bounds. In particular,
`m=0.0002705125`, whereas `18D>0.000304`.

### Both scalar and block recovery inherit the obstruction

Let M denote the exact sampled formal map. Any proposed recovery K of all
thirty harmonic coefficients has row norm k and harmonic residual block
norm e for `J=[0 I]-KM`. Applying that identity to (31), whose initial
forms vanish and harmonic infinity norm is one, gives

\[
 1\le e+k\|Mp\|_\infty\le e+km.
 \tag{37}
\]

A scalar bootstrap with remainder coefficient at least a would require
`1-e-k*a>0`. Equations (36)-(37) exclude this for every K, even one obtained
from an exact inverse. Taking upper residual bounds cannot improve it.

Keeping separate harmonic block norms also does not rescue this particular
global envelope. Write `B_n=max(||b_s,n||infinity,||b_c,n||infinity)` for an
arbitrary actual pair, and `U_*=sup||u||infinity`. Its last preceding sample
is at most two units earlier, including the final unobserved ramp and tail.
The static force and (17) then give

\[
 cU_*\le\delta+6\sum_n(1+nD)B_n,\qquad c=1-2g^2T>0.
\]
\[
 R_*:=\sup\|u-V\|_\infty\le\sum_n A_nB_n+B_0\delta,
 \quad A_n=18nD+\frac{36g^2T(1+nD)}c,\quad
 B_0=\frac{6g^2T}c.
 \tag{38}
\]

Here `delta=2*eta+E_0+E_1` retains both own-law source errors. For arbitrary
block recovery matrices K_n, let k_n be their row norms and E_nm the
nonnegative row-norm bounds of their residuals on harmonic block m.
Their proposed componentwise bootstrap has feedback matrix

\[
 H=E+kA^T.
 \tag{39}
\]

Using the formal witness instead of an actual pair in the exact recovery
identities gives `b<=E*b+k*m`. But `A^T*b>=A_1>18D>m`, so

\[
 Hb\ge b,\qquad b>0.
 \tag{40}
\]

Thus `I-H` cannot have a nonnegative inverse: otherwise multiplying
`(I-H)b<=0` by that inverse would imply `b<=0`. This excludes the positive
block bootstrap for every choice of dual weights and their certified
residual bounds. Positive diagonal changes of coefficient units preserve
the same obstruction. A more accurate matrix or a different inverse cannot
repair (38)-(39); a sharper correlated remainder or constrained argument
is a different method and is not excluded.

### The obstruction is not an admitted cross-law history

An actual alternative has ninth and eighteenth phasor radii
`r9=gamma/(4*2187)` and `r18=r9/2`. In complex notation
`z_n=b_c,n+i*b_s,n`, its coefficients on each edge satisfy

\[
 z_9=-r_9e^{9ia},\qquad z_{18}=(r_9/2)e^{18ia}
       =z_9^2/(2r_9).
 \tag{41}
\]

Both are tied to the same phase a that occurs in the fundamental
cross-law difference. On each edge the witness has phasor modulus ratio
`323/20`, instead of 2, so even rescaling it cannot produce these actual
coefficients. This uses its zero b_c coordinates; general block infinity
norms need not have the same ratio as phasor moduli.
It also supplies no pair of initial cycle phases or nonlinear histories.
Its role is to prove a limitation introduced by allowing independent formal
coefficients and independent norm remainders. Finite formal rank, exact
local ambiguity and this proof-method obstruction leave the joint noisy
nonlinear separation/overlap question open. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-law-state-identifiability)
retains the constrained continuation under the same word, noise and horizon.

<a id="organized-law-long-ramp-contrasts"></a>
## 7. Constrained contrasts from recorded long-ramp endpoints

The following construction avoids dividing a form difference by the short
plateau duration. It cancels the formal initial-form response, constant
forcing and fundamental harmonic before using the alternative's shared
phase. It supplies a conditional finite separation criterion; the required
kernel magnitudes and actual-history error bounds remain unresolved.

### An explicit functional of the unchanged readings

Orient each edge as `e -> e+1`, with incidence column
`b_e=e_(e+1)-e_e`. Choose edge `1 -> 2` and its two marked-child blocks,
`c=1,2`. Let `h_p=1e-5` and `s_c=3*c*(1+h_p)`. The first and third ramps
have recorded endpoints

\[
 a_{c,1}=s_c+h_p,\quad b_{c,1}=a_{c,1}+1,\qquad
 a_{c,3}=s_c+2+3h_p,\quad b_{c,3}=a_{c,3}+1.
 \tag{42}
\]

The third endpoint is the next child's zero-level plateau beginning. All
eight endpoints in this construction belong to the existing thirty times.
Use command-removed form Y, or a difference u of two carrier forms, for
which the common command already cancels.

For either positive eigenvalue mu in (22), let P_mu be its orthogonal
spectral projector. Put `Q_c,mu=[P_mu*b_(c-1),P_mu*b_c]` and let R_c,mu,e
be the row of its left inverse selecting edge `e=1`. Define

\[
 \mathcal T_{c,\mu,e}(w)=-R_{c,\mu,e}
 [w(b_{c,1})-e^{-\mu}w(a_{c,1})
       -w(b_{c,3})+e^{-\mu}w(a_{c,3})].
 \tag{43}
\]

The row already lies in the range of P_mu. Its norm is explicit: the Gram
matrix of Q has diagonal `d=8*mu/5` and off-diagonal `d*(1-2*mu)`, with
`d*(1-(1-2*mu)^2)=1/2`. Hence `||R_c,mu,e||2=sqrt(2)` for both modes.
Coordinate errors at most delta in the four inputs to (43) change its
value by at most `2*sqrt(10)*(1+exp(-mu))*delta`.

For the frozen formal equation (16), the first-minus-third ramp difference
cancels every constant edge current. Only the marked child's two incident
edges remain, so (43) extracts their separate weighted current contrasts.
The command on edge 1 is positive when child 2 is marked and negative
when child 1 is marked. Therefore define its two parity moments by

\[
 \mathcal E_\mu(w)=-\frac{e^{\mu/2}}8
       [\mathcal T_{2,\mu,1}(w)+\mathcal T_{1,\mu,1}(w)],\qquad
 \mathcal O_\mu(w)=\frac{e^{\mu/2}}8
       [\mathcal T_{2,\mu,1}(w)-\mathcal T_{1,\mu,1}(w)].
 \tag{44}
\]

Each has observation error at most `sqrt(10)*cosh(mu/2)*delta<4*delta`.
This bound uses the recorded forms themselves, without estimating a
plateau derivative or an unknown initial form.

### Positive kernels after removing the fundamental

Write the centered ramp coordinate as `r=t-1/2` and put
`xi(r)=epsilon*(3*r/2-2*r^3)`. On `0<=r<=1/2`, xi increases from zero
to `epsilon/2`. The first and third commands are `epsilon/2+xi(r)` and
`-epsilon/2+xi(r)`. Their trigonometric difference gives the exact kernels

\[
 C_{\mu,n}=\int_0^{1/2}\sinh(\mu r)\sin(n\xi(r))\,dr,\qquad
 D_{\mu,n}=\int_0^{1/2}\cosh(\mu r)\cos(n\xi(r))\,dr.
 \tag{45}
\]

For the formal history, (44) equals respectively
`sum_n b_s,n,e*sin(n*epsilon/2)*C_mu,n` and
`sum_n b_c,n,e*sin(n*epsilon/2)*D_mu,n`. Set

\[
 \alpha_E=C_{\mu_1,1}/C_{\mu_2,1},\qquad
 \alpha_O=D_{\mu_1,1}/D_{\mu_2,1},\qquad 0<\alpha_E,\alpha_O<1,
\]
\[
 \mathcal E=\mathcal E_{\mu_1}-\alpha_E\mathcal E_{\mu_2},\qquad
 \mathcal O=\mathcal O_{\mu_1}-\alpha_O\mathcal O_{\mu_2}.
 \tag{46}
\]

The fundamental coefficients vanish exactly. The remaining coefficients are

\[
 K_{E,n}=\sin(n\epsilon/2)(C_{\mu_1,n}-\alpha_E C_{\mu_2,n}),\qquad
 K_{O,n}=\sin(n\epsilon/2)(D_{\mu_1,n}-\alpha_O D_{\mu_2,n}),
 \quad n=9,18.
 \tag{47}
\]

All four are strictly positive. Indeed, each numerator in (47) is an
ordered double integral of a weight determinant times a function
determinant. Both weight ratios
`sinh(mu2*r)/sinh(mu1*r)` and `cosh(mu2*r)/cosh(mu1*r)` strictly increase
for positive r. Both function ratios `sin(n*x)/sin(x)` and
`cos(n*x)/cos(x)` strictly decrease for `0<x<=epsilon/2`, where
`18*x<=.9<pi/2`. Consequently both determinants have the same positive
orientation. This proves strict positivity without evaluating an integral.

The same positive representation bounds their frequency ratios. For
`0<x<y<=epsilon/2`, the sine determinant
`sin(n*x)*sin(y)-sin(x)*sin(n*y)` is `(n^2-1)*sin(y)` times the positive
Dirichlet Green integral of `sin(n*s)` on `[0,y]`, with operator
`-d^2/ds^2-1`. Since `sin(18*s)=2*sin(9*s)*cos(9*s)`, its n=9 to n=18
ratio is at most `40/(323*cos(.45))`. For the cosine determinant use

\[
 \left(\frac{\cos(ns)}{\cos s}\right)'
 =-\frac{n^2-1}{\cos^2s}\int_0^s\cos(nt)\cos t\,dt.
 \tag{48}
\]

The n=9 to n=18 ratio is at most `80/(323*cos(.9))`. Multiplying in each
case by `sin(.45)/sin(.9)=1/(2*cos(.45))` gives

\[
 0<\rho_E:=K_{E,9}/K_{E,18}
 \le\frac{20}{323\cos^2(.45)}<\frac1{12},\qquad
 0<\rho_O:=K_{O,9}/K_{O,18}
 \le\frac{40}{323\cos(.9)\cos(.45)}<\frac14.
 \tag{49}
\]

The strict rational comparisons already follow from
`cos(.45)>719/800` and `cos(.9)>119/200`. No sampled coefficient matrix
or kernel magnitude is computed here.

### A finite certificate with explicit remaining obligations

Let `theta=9*a` for the alternative's edge gap at one common reference
time, and let `r9=gamma/(4*2187)`. Retaining its actual coefficients (41)
in the frozen contrasts gives

\[
 \frac{2\mathcal E_{\rm fr}}{K_{E,18}}
 =r_9(\sin2\theta-2\rho_E\sin\theta),\qquad
 \frac{2\mathcal O_{\rm fr}}{K_{O,18}}
 =r_9(\cos2\theta-2\rho_O\cos\theta).
 \tag{50}
\]

The Euclidean norm of this pair is at least
`r9*(1-2*max(rho_E,rho_O))>r9/2`. This conclusion uses the shared phase
and fixed radii, not independent bounds on two harmonic blocks.

For two histories sharing a noisy record, (13) bounds every input to (43)
by delta. Their two final observation allowances are

\[
 N_E=\sqrt{10}[\cosh(\mu_1/2)+\alpha_E\cosh(\mu_2/2)]\delta<8\delta,
 \quad
 N_O=\sqrt{10}[\cosh(\mu_1/2)+\alpha_O\cosh(\mu_2/2)]\delta<8\delta.
 \tag{51}
\]

Suppose uniform, own-law arguments additionally establish
`|mathcal E(u)-mathcal E_fr|<=epsilon_E` and
`|mathcal O(u)-mathcal O_fr|<=epsilon_O`. A sufficient separation criterion
for the actual record sets is then

\[
 \sqrt{\left(\frac{2(N_E+\epsilon_E)}{K_{E,18}}\right)^2+
             \left(\frac{2(N_O+\epsilon_O)}{K_{O,18}}\right)^2}
 <\frac{r_9}{2}.
 \tag{52}
\]

This is a conditional criterion, not a passing verdict. It still needs
certified kernel magnitudes and actual-history departures from one shared
reference phase. Reconstructing phase increments accurately does not make
those increments zero: even known nonzero endogenous drift changes the
frozen kernels. It must be incorporated into the contrasts or bounded,
together with the uncertainty of the reconstruction and both laws' source
errors. Local resetting must also retain transport between the two child
blocks. No numerical kernel, trajectory, fit or reserved response is
evaluated by this construction.

<a id="organized-law-historical-phase-transport"></a>
## 8. The same form readings constrain historical phase transport

The exact carrier phase row depends on the observed form history, not on
an independently chosen phase velocity. This allows the initial phases to
remain unknown while bounding their increments more tightly. This section
uses only the historical interval
`0<=v<=t_last=14+15/100000`; it does not reconstruct the unobserved return
ramp or the tail to time 16.

Put `A=L5/4` and `w_lambda=gamma*S_lambda,5/4` in (7). Then

\[
 Y'=-AY+w_\lambda,\qquad \Psi'=\gamma AY,\qquad
 \|Y\|_\infty\le M_\lambda:=X+gJ_\lambda T/2.
 \tag{53}
\]

At each reading the commanded form offset vanishes, so the raw fine-mean
reading y differs from this law's carrier Y by at most
`epsilon_lambda=eta+E_lambda` in each coordinate. Both compared laws use
the same raw record if one is jointly compatible; this is a conditional
argument for any such record, without evaluating one.

### A bridge that retains arbitrary heat relaxation

On each interval of length h between successive readings, define

\[
 Q(s)=\int_0^s e^{-rA}\,dr,\qquad V_h(s)=Q(s)Q(h)^{-1},\qquad
 \widehat Y(s)=(I-V_h(s))y_i+V_h(s)y_{i+1}.
 \tag{54}
\]

The matrix Q(h) is positive definite, including its constant-mode
eigenvalue h. The eigenvalues of V_h(s) lie in `[0,1]`. This bridge exactly
reproduces every solution with constant forcing, including arbitrary
initial heat relaxation. It is an observation-based interpolation, not a
new supplied evolution law. It need not reproduce the actual current.
Joining the intervals yields one continuous form interpolant. Define
the common edge-phase increment reference

\[
 \widehat d(v)=\gamma B^TA\int_0^v\widehat Y(s)\,ds.
 \tag{55}
\]

The initial circular edge phase is not reconstructed. For each law its
actual carrier increment `d_lambda=B^T(Psi_lambda(v)-Psi_lambda(0))` is
compared with this same reference.

With exact carrier endpoints, `e=Y-hat Y` obeys
`e''+Ae'=w_lambda'`, with zero values at both cell endpoints. For a
spectral value `0<=mu<=1`, let `q_mu(s)=(1-exp(-mu*s))/mu`, using s when
mu is zero. The absolute scalar Dirichlet Green kernel is

\[
 G_\mu(t,s)=
 \frac{q_\mu(\min(t,s))q_\mu(h-\max(t,s))}{q_\mu(h)}
 e^{-\mu\max(t-s,0)}.
\]

For h at most one, `q_mu(r)<=r` and
`mu*h/(1-exp(-mu*h))<=1/(1-exp(-1))<8/5`, with the continuous value one
at mu zero. Consequently

\[
 0\le G_\mu(t,s)\le\frac85G_0(t,s),\qquad
 \int_0^hG_0(t,s)\,dt=\frac{s(h-s)}2.
 \tag{56}
\]

The C5 geometry gives `||B||2<=2`, `||A||2<=1`, and each row of
`B^T A` has squared Euclidean norm `5/4`. The actual current derivative
therefore satisfies, when the marked command level is q,

\[
 \|w_\lambda'\|_2\le
 \frac{\gamma L_\lambda}{2}
 (\sqrt2|q'|+2\gamma\sqrt5 M_\lambda).
 \tag{57}
\]

The bound uses `B^T H'`, which has exactly two nonzero entries of opposite
sign, and the complete phase row in (53). It is an own-law estimate, not
an observed derivative.

### Separate reading, command and endogenous errors

For a cubic ramp of amplitude a and duration h,
`q'=6*a*s*(h-s)/h^3`, so integrating its absolute value against the
weight in (56) gives `|a|*h^2/10`. Combining (56)-(57) with the edge-row
norm gives at most `sqrt(10)*gamma^2*L_lambda*|a|*h^2/25` in phase
increment error. Use `sqrt(10)<19/6`. The endogenous contribution is at
most `gamma^3*L_lambda*M_lambda*h^3/3`.

Endpoint perturbations have Euclidean norm at most
`sqrt(5)*epsilon_lambda`. Both spectral bridge weights have norm at most
one. After applying `gamma*B^T A` and integrating, their contribution
is at most `5*g*epsilon_lambda*h` per cell. No derivative estimate
divides these errors by the short plateau duration.

Thus a uniform historical bound is

\[
 \|d_\lambda(v)-\widehat d(v)\|_\infty\le R_\lambda,
\]
\[
 R_\lambda=
 5g\epsilon_\lambda t_{last}
 +\frac{19}{150}g^2L_\lambda\sum_{\rm cells}|a|h^2
 +\frac{g^3L_\lambda M_\lambda}{3}\sum_{\rm cells}h^3.
 \tag{58}
\]

There are exactly fourteen observed ramps and fifteen short plateaus:

\[
 \sum|a|h^2=19/10,\qquad
 \sum h^3=14+15\cdot10^{-15}.
 \tag{59}
\]

The last return ramp has no right endpoint reading and is excluded.
Summing only completed cells and the full nonnegative allowance for the
current cell gives a valid bound also at interior times. In particular,
for both endpoint laws (and uniformly through `0<=lambda<=1`),

\[
 R_\lambda<2.8\cdot10^{-8}.
 \tag{60}
\]

The report retains the three error contributions and cumulative cell
allowances separately. Unlike the old absolute drift D, the interpolation
error has no term of order `g*X`; initial form enters only the much smaller
endogenous term of order `g^3*X`. This uses additional information already
present in the same record, without tightening its noise allowance.

This is a carrier edge-phase increment bound. A statement about the fine
collective mean-phase increment must additionally retain twice the own-law
mean-phase source error; individual hidden constituent phases remain
unobserved. Neither version identifies the initial phase, its winding or
the law. In particular, the nonzero common reference d_hat must be retained
when using the contrasts in section 7: a small reconstruction error does
not justify replacing the reconstructed motion by zero. Implementing and
certifying their combined history-dependent test is still an obligation.

<a id="organized-law-three-ramp-contrasts"></a>
## 9. The observed middle ramp supplies a second command shape

The middle ramp also has recorded endpoints in blocks c=1,2 and c=2,3.
For the continuation below select edge 2 and blocks c=2,3, whose complete
ramps lie between local times 6.00007 and 12.00012. No final unobserved
return ramp or additional reading is used. Translation of the child labels
preserves the C5 projector and frozen-kernel algebra in section 7.

Write F_c,i(w)=w(b_c,i)-exp(-mu)w(a_c,i). Retain the first-minus-third
functional T in (43), and define

\[
 {\cal Q}_{c,\mu,e}(w)=-R_{c,\mu,e}
 [F_{c,1}(w)+F_{c,3}(w)-2F_{c,2}(w)].
\]

All three intervals have unit duration. The coefficients sum to zero,
so constant forcing cancels. At centered ramp coordinate r, their commands
are respectively epsilon/2+xi(r), -2xi(r), and -epsilon/2+xi(r).
Combining the two marked-child orientations gives

\[
 W_{E,\mu}=e^{\mu/2}({\cal Q}_{3,\mu,2}+{\cal Q}_{2,\mu,2})/8,\qquad
 W_{O,\mu}=e^{\mu/2}({\cal Q}_{3,\mu,2}-{\cal Q}_{2,\mu,2})/8.
\]

Their frozen harmonic coefficients are respectively

\[
 L_{D,n}=\cos(n\epsilon/2)D_{\mu,n}-D_{\mu,2n},\qquad
 L_{C,n}=\cos(n\epsilon/2)C_{\mu,n}+C_{\mu,2n}.
\]

Thus E_mu-alpha_E W_E,mu and O_mu-alpha_O W_O,mu cancel the
fundamental at a single mode when alpha_E and alpha_O take the ratios
declared below. The signed high-harmonic coefficients still obey one common
phase-circle relation. The new functional retains existing samples; it
changes the analysis of their information, not the preparation or waveform.
Actual phase motion and cross-law transport still require their own bounds.

### Prospective scalar-moment evaluation

This is one analytical assessment of the scalar kernels, not a record,
trajectory, fit, law selection or complete nonlinear separation. Commit this
declaration, the helper and its exclusive driver before evaluation. Preserve
the attempt and exact interval report; do not retry or change numerical
budgets after observing it. The following choices are fixed:

| Item | Declaration |
| --- | --- |
| Kernel set | Both exact C5 modes `mu=(5-sqrt(5))/8,(5+sqrt(5))/8`; harmonics `1,2,9,18,36`; twenty scalar `C,D` integrals total |
| Arithmetic | Shared outward rational dyadic128 intervals; exact rational polynomial coefficients and antiderivatives; shared interval square root and trigonometry |
| Fixed series | Twenty terms in each hyperbolic and trigonometric series, giving odd degrees through 39 and even degrees through 38 |
| Kernel budget | Every complete `C,D` enclosure must have radius at most `1e-28`; zero-crossing elimination denominators give unavailable arithmetic |
| Derived quantities | Original across-mode contrasts and the three-ramp contrast at each mode, retaining signed coefficients and the full observation-noise multipliers |
| Selected continuation | Mode 2 is selected before evaluation; mode 1 and the old construction are retained controls, not candidates in a result-dependent search |
| No additional evaluation | No response, reconstructed phase history, nonlinear fit, sampled matrix, frozen source producer or physical observation is evaluated |

Since `xi(r)=r*(3/20-r^2/5)`, define exact rational polynomial moments

\[
 J_{m,k}=\sum_{j=0}^{k}\binom{k}{j}(3/20)^{k-j}(-1/5)^j
 \frac{(1/2)^{m+k+2j+1}}{m+k+2j+1}.
\]

Integrate the finite products by substituting `J` into

\[
 C^{[20]}_{\mu,n}=\sum_{p,q=0}^{19}
 \frac{\mu^{2p+1}(-1)^qn^{2q+1}J_{2p+1,2q+1}}
 {(2p+1)!(2q+1)!},
\]

\[
 D^{[20]}_{\mu,n}=\sum_{p,q=0}^{19}
 \frac{\mu^{2p}(-1)^qn^{2q}J_{2p,2q}}{(2p)!(2q)!}.
\]

Combine each rational coefficient, including the exact `J`, before interval
multiplication by a power of mu. The finite sums and every subsequent
operation retain outward arithmetic rounding. On this domain
`abs(mu*r)<=1/2` and `abs(n*xi)<=9/5`. Uniform omitted-series bounds are

\[
 E_{sh}=\frac{(1/2)^{41}}{41!}\frac1{1-1/(4\cdot42\cdot43)},\quad
 E_{ch}=\frac{(1/2)^{40}}{40!}\frac1{1-1/(4\cdot41\cdot42)},
\]

\[
 E_s=(9/5)^{41}/41!,\qquad E_c=(9/5)^{40}/40!.
\]

The hyperbolic tails use the decreasing positive-term ratio; the
trigonometric bounds use Taylor's theorem with derivatives bounded by one.
Every hyperbolic partial sum is below two and the actual sine/cosine has
absolute value at most one. Consequently append the symmetric integrated
product tails `(E_sh+2*E_s)/2` and `(E_ch+2*E_c)/2` to C and D respectively.
These are product errors, not derivative or trajectory error estimates.

For a single mode put `s_n=sin(n/20)`, `c_n=cos(n/20)` and

\[
 \alpha_E=\frac{s_1C_{\mu,1}}{c_1D_{\mu,1}-D_{\mu,2}},\qquad
 \alpha_O=\frac{s_1D_{\mu,1}}{c_1C_{\mu,1}+C_{\mu,2}}.
\]

The signed remaining coefficients are

\[
 K_{E,n}=s_nC_{\mu,n}-\alpha_E(c_nD_{\mu,n}-D_{\mu,2n}),\qquad
 K_{O,n}=s_nD_{\mu,n}-\alpha_O(c_nC_{\mu,n}+C_{\mu,2n}),
 \quad n=9,18.
\]

Their raw common-record uncertainty allowances are respectively
`sqrt(10)*cosh(mu/2)*(1+2*abs(alpha_E))*delta` and the same expression with
alpha_O. The factor two retains the doubled absolute ramp weights in the
`first+third-2*second` functional. Evaluate cosh with the same fixed positive
series and its admitted tail. Kernel arithmetic admission alone supplies no
bound on the actual-history departure from these frozen coefficients.

`benchmarks/organized_law_contrast_kernels.py` uses the existing exclusive
analytical runner: it requires a committed source/declaration, writes the
source inventory, configuration, backend and pre-evaluation attempt, then
writes the exact rational interval report. The reported scope is
`moment_kernel_arithmetic_only_not_a_record_verdict`.


## 10. Shared implementation and verification

The [detached consumer](../../src/tnfr/physics/_organized_law_observation.py)
reuses the canonical potential/current, product geometry, hidden transport,
interval/jet arithmetic and waveform. It supplies own-law analytical bounds,
an actual carrier field, ideal local quadratures and the exact analytical
relaxation bounds. Its conditional phase-circle checker re-admits coefficient
and total-error primitives and excludes simultaneous membership in two
zero-centered error bands only under the declared shared-phase contrast
premises. Unsafe denominators and nonpositive circle margins are unavailable;
equality at the squared error threshold does not certify exclusion.
The [phase-transport bound](../../src/tnfr/physics/_organized_law_phase_transport.py)
rebuilds the historical grid, current variation and both own-law source
budgets. It consumes no raw record and does not implement an individual
phase reconstruction. Neither consumer installs a runtime law or evaluates
a trajectory. The shared prescribed-time validator remains
the execution owner for any separately declared future numerical test.

[Admission and field controls](../../tests/physics/test_organized_law_observation.py)
check invalid domains, the fine/carrier projection, both clock rows,
derivative bounds, source denominators and identity/growth margins.
[Information controls](../../tests/physics/test_organized_law_observation_information.py)
compare actual-current quadratures with independently expanded harmonics
and verify the interval sign witnesses used by the analytical obstruction.
[Finite-rank controls](../../tests/physics/test_organized_law_observation_rank.py)
check the actual endpoint schedule, reversed edge probes and C5 spectral
independence used in section 5.
[Relaxation controls](../../tests/physics/test_organized_law_observation_relaxation.py)
check moment cancellation, actual forcing orientation, the dipole heat bound,
own-law remainder constants and the obstruction's constitutive exclusion.
[Constrained-contrast controls](../../tests/physics/test_organized_law_observation_constrained.py)
check shared-phase synthetic cases, strict margins and primitive rejection.
[Phase-transport controls](../../tests/physics/test_organized_law_phase_transport.py)
check the actual historical domain, exact C5 norms, independent scalar Green
and constant-forcing cases, and separate reading/source/command budgets.
These controls exercise proof premises; the proofs establish their stated
mathematical conclusions.
No root search, fit, sampled coefficient assessment or frozen producer is run.
