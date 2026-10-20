# Constitutive dependence and source selection under weak contact

<a id="alternative-weak-contact-registration"></a>

The retained harmonic alternative changes the attracting relative
registrations of the weakly contacted organization. At the fixed spoke
weight eta=1/100000, a declared full-state neighborhood contains twenty
stationary organizations modulo common phase rotation: ten strict
relative minima and ten index-one saddles. All five origins that attract
under the sine law become saddle origins under this alternative.

An own-law source theorem below also places ten explicit intervals of the
original donor preparation inside the alternative's attracting basins,
retaining all original errors and the complete observation history. On
one declared common preparation interval, the two laws
select asymptotic relative origins separated by more than pi/12.
The alternatives retain the same support, capacities, clock and
constituent profile; their collective registration geometry is
nevertheless different. Neither reached sources nor the
[sine basin certificate](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-source-basin)
are substituted for the alternative's own history and bounds.

| Premise | Fixed declaration |
| --- | --- |
| Constitutive law | Deformation-one potential and current already defined in [the alternative contact owner](ORGANIZED_ALTERNATIVE_CONTACT.md#organized-alternative-contact) |
| Support | The same 54 labeled nodes of C9 x W6; 99 old internal/rim edges of weight one, 45 donor spokes of weight eta=1/100000 |
| Complete dynamics | Both weighted form diffusion and weighted nonlinear phase current, held unit capacity, uniform quarter factors and the original tau clock |
| State and symmetry | All signed forms and primitive circular phases; zero total form, common phase rotation factored out; continuous compatible lifts only for the declared local coordinates |
| Stationary domain | Mean-zero host shape norm at most 6*eta and childwise mean-zero fine phase maximum norm at most 1/100; all donor origins admitted |
| Actual source | The original alternative-law outer-one acquisition, full sequential observation word and independent donor recovery; all twelve primitive allowances and all 96 relative directions |
| Certified preparation intervals | Donor origins within 1/10 of each odd multiple of pi/10 reach the unique attractive equilibrium in the opposite relative-origin window |
| Evidence | Exact finite Fourier identities, real contraction and explicit derivative/remainder bounds; no root search, trajectory, source replay, sweep or physical response |

## 1. Complete weighted law and the full-state domain

Write c=0,...,4 for host children and c=5=D for the donor; j=0,...,8
is the internal slot. The primitive node order is 9*c+j. Put

\[
\begin{split}
 \eta&=\frac1{100000},\qquad \gamma=\frac1{1023\pi},\qquad
 \tau=\frac{1023}{1024}t_{\rm original},\\
 V(a)&=1-\cos a+\frac{(1-\cos9a)^2}{6\,9^4},\\
 j(a)&=V'(a)=\sin a+\frac{\sin9a-\tfrac12\sin18a}{2187}.
\end{split}
\tag{1}
\]

Let L_eta be the unnormalized weighted Laplacian, and let
\(S_i(\theta)=\sum_{b\sim i}w_{ib}j(\theta_b-\theta_i)\).
Write \(U(\theta)=\sum_{\{i,b\}}w_{ib}V(\theta_b-\theta_i)\)
for its phase potential. The complete unforced law and its storage are

\[
\begin{split}
 x'&=\frac{-L_\eta x+\gamma S(\theta)}4,&
 \theta'&=\frac\gamma4L_\eta x,\\
 H&=\frac12x^TL_\eta x+U(\theta),&
 H'&=-\frac14\|L_\eta x\|^2.
\end{split}
\tag{2}
\]

Prime means the declared tau clock in both rows. The pressure in the
held-capacity nodal row is supplied by the first complete equation;
it is not reconstructed from a response. Odd current and Laplacian
row sums preserve total form and total lifted phase. All coefficients
retain their structural units; no laboratory clock or storage unit is
identified. Support, eta and the absence of later inputs/events remain
premises. The balance follows from this alternative potential and
current, not from copying a sine certificate.

Put alpha=2*pi/9 and b=2*pi/5. With compatible old marked-edge lifts,
write

\[
\begin{split}
 \theta_{c,j}&=\alpha(j-4)+\Phi_c+q_{c,j},&
 \sum_jq_{c,j}&=0,\\
 t&=\frac15\sum_{c=0}^4\Phi_c-\Phi_D,&
 b_c&=b(c-2),\\
 h_c&=\Phi_c-\Phi_D-t-b_c,&
 \sum_{c=0}^4h_c&=0.
\end{split}
\tag{3}
\]

Here t is a relative origin, not the evolution clock. The domain is

\[
 \|h\|_2\le6\eta,\qquad \|q\|_\infty\le\frac1{100},
 \qquad t\in\mathbb R/(2\pi\mathbb Z),\qquad \mathbf1^Tx=0.
\tag{4}
\]

Forms may be any finite signed values on that mean leaf. Internal
closing edges and host-rim closing edges retain their original one-turn
conventions. All old compatible gaps remain positive and below pi:
internal gaps differ from alpha by at most 1/50; rim gaps differ from b
by at most 12*eta+1/50. Thus (4) retains all six internal windings,
all nine rim windings and all 32 zero mixed host periods of the
[complete marked class](ORGANIZED_CONTACT_STATIONARY_SEARCH.md#marked-cycle-information-and-connected-chambers).
Spoke phases have no added acute restriction.

The domain is a stationary classification neighborhood. It is not
asserted to be positively invariant or to contain the initial actual
source. Sections 6--8 retain that source in a larger energy trap and
classify its limit using the stationary theorem. No finite settling
time is supplied.

## 2. Own-law host-shape contraction

The real derivative bounds from (1) are

\[
 J_0=\frac{1459}{1458}<\frac{1001}{1000},\quad
 J_1=\frac{245}{243}<\frac{101}{100},\quad
 J_2=\frac{10}{9},\quad J_3=\frac83,
 \qquad |j^{(r)}|\le J_r\quad(0\le r\le3).
\tag{5}
\]

Let D be the forward C5 incidence, L5=D^T D and
\(P=I-\mathbf1\mathbf1^T/5\). Define

\[
\begin{split}
 c_b&=\cos b=\frac{\sqrt5-1}{4},&
 a&=j'(b)=c_b+\frac{\sqrt5}{486},\\
 A&=aL_5,&G&=A^+,&
 \lambda_1&=a(2-2\cos b),&
 \lambda_2&=a(2-2\cos2b).
\end{split}
\tag{6}
\]

G acts on the zero-mean host space. The elementary bounds
\(3/10<c_b<1/3\) give \(\lambda_1>2/5\); also
\(a<8/25\) and \(2-2c_b<7/5\) give
\(\lambda_1<9/20\). The bound a<8/25 follows, for example,
from \(c_b<5/16\) and \(\sqrt5/486<1/216\).
Consequently \(\|G\|_2<5/2\), and \(\|D\|_2<2\).

On the internal carrier q=0, ground the donor phase at zero and seek
host phases \(t+b_c+u_c\), with zero-mean u. The four projected
stationary equations are exactly

\[
\begin{split}
 u&=-G\{D^TN(Du)+\eta Pj(t+b+u)\}=:\mathcal K_t(u),\\
 N(d)&=j(b+d)-j(b)-a d.
\end{split}
\tag{7}
\]

The scalar b in N is the rim gap; the b inside the vector current is
\((b_c)_{c=0}^4\). All current derivatives below are evaluated on real
arguments. Since \(|N(d)|\le J_2d^2/2\), the map satisfies on
\(\|u\|_2\le6\eta\)

\[
\begin{split}
 \|\mathcal K_t(u)\|_2
 &\le\frac52\left[4J_2(6\eta)^2+
                  \eta\sqrt5\frac{1001}{1000}\right]<6\eta,\\
 \|D_u\mathcal K_t\|_2
 &\le Q:=\frac52\left[48J_2+\frac{101}{100}\right]\eta
       =\frac{16303}{12000000}<\frac1{500}.
\end{split}
\tag{8}
\]

The bounds use \(\|D^TN(Du)\|_2\le4J_2\|u\|_2^2\) and
\(\|D^T\operatorname{diag}(N'(Du))D\|_2\le8J_2\|u\|_2\).
Thus, for every real t, there is exactly one projected stationary shape
u(t) in the declared host ball. It is smooth in t by the invertible
projected derivative. This is an own-law real contraction; the complex
growth bounds of the sine current are not reused.

Write \(f(t)=j(t+b)\), \(u_1(t)=-Gf(t)\). Each active frequency
1, 9 and 18 has zero sum on the five host vertices. Direct subtraction
of (7) and differentiation in t give

\[
\begin{split}
 \|u-\eta u_1\|_2
 &\le\frac52\left[4J_2\,36+6\frac{101}{100}\right]\eta^2
       <416\eta^2,\\
 \|u_t\|_2
 &\le\frac{(5/2)\sqrt5(101/100)}{1-Q}\eta<6\eta,\\
 \|u_t-\eta u_{1,t}\|_2
 &\le\frac52\left[8J_2\,36+6J_2+6\frac{101}{100}\right]\eta^2
       <832\eta^2.
\end{split}
\tag{9}
\]

In the last line, retain separately the rim derivative defect,
\(\eta[j'(t+b+u)-j'(t+b)]\), and
\(\eta\operatorname{diag}(j'(t+b+u))u_t\). These terms are what
bound the derivative of the actual shape, rather than just its values.

## 3. Collective torque and its finite C1 remainder

Let W(t) be the alternative wheel potential evaluated at its projected
stationary shape. The remaining stationary equation is

\[
 T(t)=W'(t)=\eta\sum_{c=0}^4j(t+b_c+u_c(t))=0.
\tag{10}
\]

Cyclic relabeling makes W and T periodic with period 2*pi/5. Reflection
and phase inversion make W even and T odd. Both claims use uniqueness
of the shape in (8), as well as the actual symmetries of (1). Therefore
\(T(m\pi/5)=0\) exactly for every integer m. Symmetry alone does
not determine their curvature.

Put \(\varepsilon=1/2187\). The five-point Fourier identities give

\[
\begin{split}
 f^TGf
 &=\frac5{2\lambda_1}
       [1+\varepsilon^2-2\varepsilon\cos(10t)]
       +\frac{5\varepsilon^2}{8\lambda_2},\\
 \sum_cj'(t+b_c)u_{1,c}
 &=-\frac12\frac d{dt}(f^TGf)
       =-K\sin(10t),\qquad
 K=\frac{25}{2187\lambda_1}>\frac1{40}.
\end{split}
\tag{11}
\]

Indeed the ninth harmonic is spatially mode minus one on this rim,
whereas the eighteenth is mode minus two. Only frequencies 1 and 9
have a nonconstant cross term in f^TGf; their inner product is
\(-5\cos(10t)/2\). The eighteenth-harmonic square contributes the
constant last term. Equation (11) is an exact coefficient identity,
not yet a finite-eta verdict.

Taylor's formula, (5) and (9) bound the actual torque and its derivative:

\[
\begin{split}
 |T(t)+K\eta^2\sin(10t)|&<960\eta^3,\\
 |T'(t)+10K\eta^2\cos(10t)|&<3001\eta^3,
 \qquad t\in\mathbb R.
\end{split}
\tag{12}
\]

Here are explicit constants for both remainders:

\[
\begin{split}
 C_0&=\sqrt5\frac{101}{100}\,416+
                       \frac{J_2}{2}\,36<960,\\
 C_1&=\sqrt5\left(J_2\,416+\frac{101}{100}\,832\right)
                     +\frac{J_3}{2}\,36+J_2\,36<3001.
\end{split}
\tag{13}
\]

The rational upper bound \(\sqrt5<161/72\), obtained from
\(161^2-5\,72^2=1\), proves the displayed strict inequalities.
For C0, subtract the exact linear term and use the actual current's
quadratic J2 Taylor remainder. For C1,
expand \(j'(t+b+u)(1+u_t)\): the two linear defects use 416 and
832, the second-order j' remainder contributes \(J_3\|u\|^2/2\),
and the product \([j'(t+b+u)-j'(t+b)]u_t\) contributes
\(J_2\|u\|\|u_t\|\). No unbounded derivative of a remainder is
inferred from a value estimate.

## 4. Twenty stationary registrations in the declared domain

On the relative-origin circle, let

\[
 I_k=\left[\frac{k\pi}{10}-\frac\pi{60},
            \frac{k\pi}{10}+\frac\pi{60}\right],
 \qquad k=0,\ldots,19.
\tag{14}
\]

These intervals are disjoint. At their endpoints and everywhere outside
their interiors, \(|\sin(10t)|\ge1/2\). The leading torque magnitude
there exceeds \(\eta^2/80\), while its error is at most
\(960\eta^3=0.0096\eta^2<\eta^2/80\).
It has opposite signs at the two endpoints of every interval and no
zeros outside the intervals.

Within I_k, \(|\cos(10t)|\ge\sqrt3/2>4/5\), and its sign is
\((-1)^k\). The leading derivative magnitude exceeds
\(\eta^2/5\), while its error is at most
\(3001\eta^3=0.03001\eta^2<\eta^2/5\).
Thus the actual torque is strictly monotone in each interval and has
exactly one simple zero there.

For even k, the zero is exactly \(t=k\pi/10=m\pi/5\), by the
symmetry already proved. Its derivative is negative. For odd k, the
unique zero in I_k has positive derivative. The proof does not assign
its exact position or assert that it differs from its interval center.
There are exactly ten of each type.

The four host shape directions have positive curvature. In fact the
rim weights in the domain satisfy the lower bound proved in section 5,
so the host shape block exceeds \(L_5/4-\eta J_1I>I/5\).
Eliminating that block gives the scalar Schur complement W''=T'.
Consequently the wheel relative Hessian is positive definite at the
ten odd-interval zeros, and has four positive directions and one
negative direction at the ten exact even-interval zeros.

## 5. Full-state stationary exclusion and stability

The full fine directions cannot create additional stationary points
inside (4). To prove this for the actual alternative, let
\(C_6(v)=1-v^2/2+v^4/24-v^6/720\). Along the straight segment
from the carrier phases to any q in (4), compatible internal gaps obey
\(|\delta|<18/25\), and rim gaps obey \(0<\delta<13/10\).
For example, \(\pi<22/7\) gives
\(\alpha+1/50<18/25\) and
\(b+12\eta+1/50<13/10\).
Since \(j'(\delta)\ge\cos\delta-2/243\), exact rational cosine
bounds give

\[
 C_6(18/25)-\frac2{243}>\frac7{10},\qquad
 C_6(13/10)-\frac2{243}>\frac14.
\tag{15}
\]

Spoke curvatures may be negative, but are bounded below by -J1. Let
P_f subtract each child's slot mean. On its 48-dimensional range,
the internal C9 Laplacian has gap greater than 2/5, while the largest
eigenvalue of the nine disjoint spoke stars is six. Every Hessian along
that segment therefore satisfies

\[
 P_{\rm f}\nabla^2U P_{\rm f}\big|_{\operatorname{ran}P_{\rm f}}
 >\left[\frac7{10}\frac25-6\eta\frac{245}{243}\right]I
 =\frac{226751}{810000}I>\frac14I.
\tag{16}
\]

For any collective phases, the carrier has
\(P_{\rm f}\nabla U=0\): internal currents cancel and wheel
currents are constant in the slot. Integrating (16) consequently gives
\(q^TP_{\rm f}\nabla U(\theta)>\|q\|^2/4\) for nonzero q.

At a full equilibrium, the phase row in (2), connected support and
zero total form first force x=0. The form row then forces
\(\nabla U=0\), which by (16) forces q=0. The projected host
equation and contraction force h=u(t). Thus all full stationary states
in (4) are exactly the twenty wheel equilibria already classified;
noncarrier directions were retained and excluded by their own-law
curvature, rather than discarded before testing stationarity.

At a carrier, \(j'(\alpha)=\cos\alpha\) exactly. In slot-first
order the phase Hessian has the product form

\[
 M_{54}=\cos\alpha\,L_9\otimes I_6+I_9\otimes M_{\rm wheel}.
\tag{17}
\]

The 48 fine directions remain positive by (16), while the five relative
collective directions have the inertia from section 4. The ten minima
therefore have 53 positive relative phase directions. The ten saddles
have 52 positive directions and one negative direction, with no extra
zero direction besides common phase rotation.

This phase curvature controls the actual complete two-row dynamics.
On the relative space its own linearization has the symmetric pencil

\[
 s^2L_\eta^{-1}+\frac s4I+\frac{\gamma^2}{16}M_{54}.
\tag{18}
\]

For a positive phase Hessian, the scalar quadratic identity obtained by
testing an eigenvector has all roots with negative real part. For an
index-one Hessian, the pencil increases strictly for positive real s
from one negative direction to a positive matrix, giving exactly one
positive eigenvalue. The imaginary part of the same scalar identity
excludes nonreal eigenvalues with nonnegative real part; invertibility
of the relative Hessian excludes s=0. All remaining eigenvalues have
negative real part. Thus the ten minima are locally exponentially
attracting on fixed-mean leaves, and the other ten are hyperbolic
index-one saddles of the full law.

<a id="alternative-weak-contact-source-retention"></a>
## 6. The unchanged own-law source enters a full-state trap

Keep the original outer-one family from the
[retained alternative acquisition](ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md#alternative-outer-acquisition-result),
its complete [alternative-law observation history](ORGANIZED_ALTERNATIVE_CONTACT.md#2-carrying-the-own-law-hosts-through-all-observation-events),
and the independently recovered donor from that same owner. The twelve
primitive form/phase error allowances stay in [0,1e-10], with all 96
relative directions and their original zero-sum constraints. The donor
origin beta is supplied at primitive preparation. Its isolated rotation
does not change its relative recovery trajectory.

All acquisition times, recovery dwells, commands and observation windows
are unchanged. In particular the full fifteen-event word precedes
contact; its final cumulative command vanishes, while every continuous
drift and form change remains. Only the 45 new spoke weights are changed
to eta in the subsequently supplied contact model (2). The earlier
unit-contact observations and frozen responses are neither reused as
weak-contact predictions nor re-evaluated.

Write

\[
 V_9=9V(\alpha)=9(1-\cos\alpha),\qquad
 V_5=5V(b),\qquad V_*=6V_9+9V_5.
\tag{19}
\]

The retained own-law acquisition inequalities give host excess above
5*V9+9*V5 below 0.009588 and its product-cell boundary excess B above
0.01451. These are existing strict rational summaries of the preserved
source evidence, not newly evaluated endpoints. The same cycle-boundary
formula applies after adding the donor C9: the old cycles now comprise
six C9 and nine C5 cycles, with unchanged old edge weights.

### 6.1 Observation drift and new contact work remain explicit

For a fresh analytical upper bound \(\bar\gamma=1/3000>\gamma\),
put X=1/100 and T_o=1/5000. The own-law current bound J0 in (5) gives

\[
\begin{split}
 B_o&=X+\bar\gamma J_0T_o<\frac{101}{10000},\\
 d_x&=(2B_o+\bar\gamma J_0)T_o<\frac1{200000},\\
 m_o&=2\bar\gamma B_oT_o<\frac1{500000000}.
\end{split}
\tag{20}
\]

Heat contraction bounds every host form during the word by B_o; its
degree-four form row then bounds each net form change by d_x. The
phase row bounds each continuous phase drift by m_o after removing
only the explicit command. Every one of the 90 host edges has net
form-gap change at most 2*d_x and net phase-gap change at most 2*m_o.
The square-difference identity and \(|V'|\le J_0\) therefore give

\[
 H_H(T_o)-H_H(0)\le360B_od_x+180J_0m_o<\frac1{50000}.
\tag{21}
\]

This endpoint comparison does not cancel the paid work of intermediate
events or identify the reached endpoint with the pre-word state.

The donor retains form and relative-phase norms at most
\(q_D=10^{-70}\). Its target phase gradient vanishes, its C9
Laplacian norm is below four, and \(|V''|\le J_1\), so
\(H_D\le V_9+2(1+J_1)q_D^2<V_9+5q_D^2\).
The weighted support event changes no coordinate and has work

\[
\begin{split}
 W_\eta&=\eta\sum_{c,j}
 \left[\tfrac12(x_{D,j}-x_{H,c,j})^2+
              V(\theta_{D,j}-\theta_{H,c,j})\right],\\
 W_\eta&\le45\eta\left[\tfrac12(B_o+q_D)^2+
                         2+\frac2{19683}\right]
                  <\frac{901}{10^6}.
\end{split}
\tag{22}
\]

The extra 2/19683 is the global upper bound on the alternative's
nonnegative added potential. This is supplied attachment work; past
loss does not fund the event or select its occurrence.

Combining these own-law inequalities proves, for every original beta
and every admitted original error member,

\[
 H(0)-V_*<0.009588+\frac1{50000}+\frac{901}{10^6}+5q_D^2
       <\frac{1051}{100000}<\frac3{250},\qquad
 B-[H(0)-V_*]>\frac1{250}.
\tag{23}
\]

Time zero is the actual final contact, not a reset. The pre-contact
phase inclusions from the own-law source and complete observation word
place every old edge in its original acute branch. The donor has the
same winding-one C9 branch. No energy inequality is used to invent
this initial branch information.

### 6.2 The old identity is retained for every future time

The [own-law acute convexity proof](ORGANIZED_ALTERNATIVE_OUTER_ACQUISITION.md#4-own-law-convexity-and-every-product-boundary)
establishes \(V''>0\) on the entire closed interval [-pi/2,pi/2].
Thus Jensen's inequality on the actual old cycles gives the boundary
excess B from that owner. The 198 signed old-edge faces consist of
54 positive and 54 negative internal faces, 45 positive rim faces and
45 empty negative rim faces. Negative internal faces have larger cost;
negative rim faces cannot accommodate winding one. The added donor
cycle changes their number, not the minimum cycle-face formula.

Form storage and the weighted spoke potential are nonnegative. Hence
every old-cell boundary has \(H\ge V_*+B\), independently of beta.
Equations (2) and (23) exclude the first such boundary encounter.
All six internal windings, nine rim windings and 32 zero mixed host
periods persist for the whole future trajectory. The retained sublevel
stays a positive distance from the old acute faces.

At fixed positive eta, connected support and zero total form make
the form coordinates bounded by their storage. The two old support
components have bounded relative phase lifts inside the old cell;
their remaining relative origin is a circle. Thus the full trapped
state is precompact modulo common phase rotation. This argument keeps
all fine coordinates and does not assert invariance of the smaller
stationary tube (4).

## 7. Own-law fine relaxation and classification of source limits

First improve the phase domains using the alternative potential itself.
For a winding-one C9 cycle with one gap fixed at s, write
\(F_9(s)=V(s)+8V((2\pi-s)/8)\). The added potential is nonnegative
and vanishes at alpha. Therefore the
[existing rational cosine inequality](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-fine-relaxation)
gives \(F_9(\pi/3)-V_9>1/25\) for this law as well. The negative
face has larger cost by even acute convexity. Since the full excess
is below 3/250, every actual internal gap has magnitude below pi/3
for all future time.

Let \(a_c=\frac19\sum_jx_{c,j}\), \(z_{c,j}=x_{c,j}-a_c\),
and keep the fine phase q from (3). Let P_f subtract each child's mean
and \(L_f=L_\eta|_{\operatorname{ran}P_f}\). The graph gives
\(2I/5<L_f\le10I\). The segment from the actual phases to their
carrier at the same child means keeps internal gaps in (-pi/3,pi/3)
and rim gaps acute. Its own-law projected phase Hessian satisfies

\[
 m_f I\le\mathcal A_f\le11I,\qquad m_f=\frac{19}{100},
 \qquad P_f\nabla U=\mathcal A_f q,
\tag{24}
\]

because its lower bound is
\((1/2-2/243)(2/5)-6\eta J_1>19/100\), and its upper norm
is at most \(10J_1<11\). The rim contribution is nonnegative by
own-law acute convexity. The projected gradient at the carrier is
exactly zero, for every set of origins, as in section 5.

The exact fine equations and a fixed quadratic proof function are

\[
\begin{split}
 q'&=\frac\gamma4L_fz,&
 z'&=-\frac14L_fz-\frac\gamma4g_f,&g_f&=P_f\nabla U,\\
 F&=\frac12\|q+\gamma z\|^2+\frac{\gamma^2}{2}\|z\|^2,&
 F'&=-\frac{\gamma^2}{4}
      (q^Tg_f+z^TL_fz+2\gamma z^Tg_f).
\end{split}
\tag{25}
\]

Young's inequality bounds the absolute cross term by
\((m_f/2)\|q\|^2+(242\gamma^2/m_f)\|z\|^2\).
Since \(2/5-242\gamma^2/m_f>m_f/2\),
\(F\le\|q\|^2+\|z\|^2\), \(\|q\|^2\le4F\), and
\(\gamma^2\|z\|^2\le2F\),

\[
 F(\tau)\le F_0e^{-\rho\tau},\qquad
 \rho=\frac{19\gamma^2}{800},\qquad
 \|q\|^2\le4F_0e^{-\rho\tau},\qquad
 \gamma^2\|z\|^2\le2F_0e^{-\rho\tau}.
\tag{26}
\]

These are own-law bounds: the Hessian norm eleven and cross-term
constant 242 replace the sine values. A matching constituent tangent
alone would not justify (24)--(26).

### 7.1 The actual collective equations keep their fine remainder

Let L_W be the weighted wheel Laplacian and S_W its current under j.
Averaging the actual rows gives

\[
 a'=\frac{-L_Wa+\gamma[S_W(\Phi)+r]}4,\qquad
 \Phi'=\frac\gamma4L_Wa.
\tag{27}
\]

On each wheel edge, the mean-zero fine gaps cancel the linear Taylor
term. The own-law constant J2 bounds the quadratic remainder, giving

\[
\begin{split}
 \|r\|_1&\le\frac{J_2}{9}\sum_{\{c,d\},j}
                    w_{cd}(q_{c,j}-q_{d,j})^2
       \le\frac{80}{27}F_0e^{-\rho\tau},\\
 |r_D|&\le\frac{\eta J_2}{18}\sum_{c,j}(q_{c,j}-q_{D,j})^2
       \le\frac{40}{27}\eta F_0e^{-\rho\tau}.
\end{split}
\tag{28}
\]

The donor bound retains the spoke factor eta. Neither its force nor
the host remainder is set to zero during the actual transient.

Jensen across the nine rim slot cycles gives, at every future time,
\(V_{\rm rim}(\Phi_H)-V_5<(H-V_*)/9<1/750\).
Every averaged acute winding-one rim gap is positive. To bound its
upper endpoint, put

\[
\begin{split}
 C_8(v)&=1-v^2/2+v^4/24-v^6/720+v^8/40320,\\
 R_5&=\frac{1545}{1000}-C_8(29/20)
       -4C_8\!\left(\frac{2(157/50)-29/20}{4}\right).
\end{split}
\]

The own-law constrained face cost exceeds its sine counterpart minus
\(5\max(V-(1-\cos))=10/19683\). Exact rational arithmetic gives
\(R_5-10/19683>1/750\). Hence every actual averaged rim gap lies
in (0,29/20). With C6 from section 5,
\(C_6(29/20)-2/243>1/10\). The segment from the actual host shape
h to the stationary shape u(t) therefore has projected host Hessian

\[
 \frac9{100}I\le\mathcal A_H\le5I,
 \qquad g_H=\mathcal A_H[h-u(t)],
\tag{29}
\]

where g_H is the projected host potential gradient at the actual mean
phases. Its lower bound is \(L_5/10-\eta J_1I>9I/100\).
This controls the actual shape outside the small stationary ball (4).

### 7.2 Every retained source converges to one classified equilibrium

An equilibrium in the larger retained sublevel first has x=0 by (2).
Equation (24) then forces q=0; equation (29) forces h=u(t). Its last
stationary equation is T(t)=0. Thus every stationary point in the
actual trap belongs to the twenty-target classification of section 4.
This is a full-state conclusion within the admitted old-cell sublevel,
not a classification of distant or nonacute states.

The precompact orbit has a nonempty compact connected omega-limit set.
Storage has one limiting value, constant along every orbit in that
invariant limit set. Equation (2) then forces L_eta*x=0, hence x=0;
invariance of the form row forces S=0. The omega-limit is consequently
contained in the finite classified equilibrium set modulo common
rotation. Connectedness makes it a singleton. Every retained source
therefore converges to one of those twenty relative equilibria.

On the full phase torus, the conserved product
\(\prod_i e^{i\theta_i}=e^{i9\beta}\) leaves only 54 common-rotation
representatives of each relative equilibrium. The same connected-limit
argument therefore gives one full circular equilibrium. Once each
primitive phase stays in a proper arc about its limit, its continuous
lift stays in one corresponding real chart and converges there as well.
No finite-total-variation bound is asserted by this argument.

This uses the actual law, trap and finite equilibrium classification,
without importing a global analytic-convergence theorem from sine.
It does not yet choose a destination or exclude saddle capture. The
following source restriction supplies that additional conclusion.

<a id="alternative-weak-contact-source-capture"></a>
## 8. Ten explicit intervals of actual preparation reach strict minima

For k=0,...,9, define the primitive donor-origin center
\(\beta_k=(2k+1)\pi/10\). At the fixed eta in (1), every original
own-law preparation with

\[
 |\beta-\beta_k|\le\frac1{10}\quad\pmod{2\pi}
\tag{30}
\]

converges to the unique attractive equilibrium whose relative origin
lies in the interval of radius pi/60 centered at \(-\beta_k\).
The inequality uses the lift nearest the chosen center. All twelve
original allowances, all 96 relative directions and the complete own-law
history are retained. No initial residual or observed endpoint is
replaced by a stationary or exactly reflected source.

### 8.1 A coupled comparison bounds shape and origin transport

Let bars with subscript H denote the average over five host children.
Keep h and t from (3), and put

\[
\begin{split}
 p_c&=a_c-\overline a_H,& e&=h-u(t),&
 K_H&=L_5+\eta I\quad\hbox{on the zero-mean host space},\\
 \chi&=\gamma(\overline a_H-a_D),&y&=t+\chi,\\
 \delta&=\eta\sum_c[j(t+b_c+h_c)-j(t+b_c+u_c)]+r_D.
\end{split}
\tag{31}
\]

The actual donor current is T(t)+delta. The two complete mean rows
(27) give exactly

\[
\begin{split}
 t'&=\tfrac32\eta\chi,&
 \chi'&=-\tfrac32\eta\chi-\tfrac{3\gamma^2}{10}(T+\delta),\\
 e'&=\tfrac\gamma4K_Hp-u_t t',&
 p'&=-\tfrac14K_Hp-\tfrac\gamma4g_H+\tfrac\gamma4Pr_H,\\
 y'&=-\tfrac{3\gamma^2}{10}(T+\delta).
\end{split}
\tag{32}
\]

The corrected origin y retains the relative mean form through chi.
It is an auxiliary coordinate, not a new phase measurement or a claim
of instantaneous form relaxation.

Define
\(\mathcal E^2=\|e+\gamma p\|^2/2+\gamma^2\|p\|^2/2\).
Then \(\|e\|\le2\mathcal E\), and (28) gives

\[
 |\delta|\le5\eta\mathcal E+
                 \frac{40}{27}\eta F_0e^{-\rho\tau}.
\tag{33}
\]

Here \(2\sqrt5J_1<5\). The fixed-shape terms in the derivative
of \(\mathcal E^2\) are
\(-\gamma^2(e^Tg_H+p^TK_Hp+2\gamma p^Tg_H)/4\).
Use (29), \(I<K_H<5I\), and Young's inequality with
m_H=9/100. Since
\(1-50\gamma^2/m_H>m_H/2\), these terms yield decay
\(\mathcal E'\le-9\gamma^2\mathcal E/1600\).
The moving-shape term uses the actual u_t from (9), and is bounded
by \(28\eta^2|\chi|\); the fine forcing contributes at most
\(\gamma^2\|Pr_H\|\).

Also \(\lambda_1>2/5\) in (6) gives K<3/100 in (11).
The finite torque bound (12) consequently gives
\(|T|<(3/100+960\eta)\eta^2<\eta^2/20\).
With upper right derivatives at vanishing norms, the complete
comparison is

\[
\begin{split}
 \mathcal E'&\le-\frac{9\gamma^2}{1600}\mathcal E
          +28\eta^2|\chi|+\frac{80}{27}\gamma^2F_0e^{-\rho\tau},\\
 |\chi|'&\le-\frac32\eta|\chi|+\frac32\gamma^2\eta\mathcal E
          +\frac3{200}\gamma^2\eta^2
          +\frac49\gamma^2\eta F_0e^{-\rho\tau}.
\end{split}
\tag{34}
\]

### 8.2 The reached preparation controls the entire transient

The host and donor form totals are separately zero before contact.
The host lifted phase total is zero; each prescribed observation jump
has zero global phase increment. The donor total is 9*beta. Therefore
the actual contacted endpoint satisfies exactly
\(\chi(0)=0\) and \(t(0)=y(0)=-\beta\).

The retained own-law fine error before observation is below 1e-6, a
coarse container of its much smaller preserved bound. Using (20) and
subtracting child means gives
\(|q_{c,j}(0)|<1/400000\),
\(|z_{c,j}(0)|<21/1000\) and
\(|\gamma z_{c,j}(0)|<7/10^6\). The donor errors are smaller.
Consequently

\[
 F_0<27[(1/400000+7/10^6)^2+(7/10^6)^2]<10^{-8},
 \qquad \mathcal E(0)<\frac15.
\tag{35}
\]

For the second inequality, own-law rim convexity in the domain from
section 7 gives \(\|h(0)\|^2<2/75\), hence
\(\|h(0)\|<1/6\). Equations (8) and (20) give
\(\|u\|<6\eta\) and \(\|p(0)\|<\sqrt5B_o<1/40\).
Substitution into the definition of E proves (35). These estimates
use the reached state rather than narrowing any primitive allowance.

Set \(\mathcal X=\mathcal E+20\eta|\chi|\) and
\(\lambda=\gamma^2/200\). At the fixed eta,
\(9/1600-30\eta^2>1/200\), \(\eta>\gamma^2/20\), and
\(80/27+(80/9)\eta^2<3\). Thus (34) gives

\[
\begin{split}
 \mathcal X'&\le-\lambda\mathcal X+
               \frac3{10}\gamma^2\eta^3+3\gamma^2F_0e^{-\rho\tau},\\
 \mathcal E(\tau)&\le D_*e^{-\lambda\tau}+60\eta^3,&
 D_*&=\mathcal E(0)+160F_0<\frac{21}{100},\\
 |\chi(\tau)|&\le\gamma^2
       [D_*+60\eta^3+\eta/100+(8/27)F_0]
        <\frac{\gamma^2}{4}<10^{-6}.
\end{split}
\tag{36}
\]

The transient coefficient is
\(3\gamma^2/(\rho-\lambda)=160\); the nondecaying shape bound is
\(((3/10)\gamma^2\eta^3)/\lambda=60\eta^3\).
The last line follows by integrating the second row of (34) using
\(\chi(0)=0\). The nonzero shape remainder is retained; it does
not assert exact tracking of the stationary shape during motion.

### 8.3 The inward torque separates each selected source from saddles

Fix a center t_c=-beta_k, and first consider the upper strip
\(1/10\le y-t_c\le1/5\). Equation (36) puts
\(10(t-t_c)\) strictly between 0.99 and 2.01. This interval is
inside \((0.99,\pi-0.99)\), and
\(\sin(0.99)\ge0.99-0.99^3/6>4/5\).
Since t_c is an odd multiple of pi/10, the leading torque in (12)
has the restoring sign here. Its full finite remainder gives

\[
 T(t)>\left[\frac1{40}\frac45-960\eta\right]\eta^2
      =0.0104\eta^2>\frac{\eta^2}{100}.
\tag{37}
\]

Equations (33) and (36) bound the nondecaying part of |delta| by
\(300\eta^4\), smaller than \(\eta^2/100\).
The corresponding lower strip has the opposite restoring sign by
the same leading-torque and absolute-remainder bounds. No exact
reflection symmetry about an odd interval center is assumed.

Only the decaying part can drive y outward in either strip. Its entire
possible adverse displacement is

\[
\begin{split}
 J&=\frac{3\gamma^2}{10}\int_0^\infty
       [5\eta D_*e^{-\lambda\tau}
          +(40/27)\eta F_0e^{-\rho\tau}]\,d\tau\\
  &=300\eta D_*+\frac{3200}{171}\eta F_0<\frac1{1000}.
\end{split}
\tag{38}
\]

The prepared interval (30) gives \(|y(0)-t_c|\le1/10\).
If y first reached t_c+1/5, take its last prior crossing of
t_c+1/10. Integrating (32) between those times would give
\(1/10\le J<1/1000\), a contradiction. The lower strip gives
the same conclusion. Therefore

\[
 |y(\tau)-t_c|<\frac15,\qquad
 |t(\tau)-t_c|<\frac15+10^{-6}<\frac\pi{12}.
\tag{39}
\]

The source already converges to one classified equilibrium by section
7.2. The only stationary origin allowed by (39) is the unique minimum
in the central odd interval (14): its radius is pi/60, while the
nearest exact saddle is pi/10 from t_c. Hence this is the source's
destination. Its local exponential attraction eventually controls the
complete state on the conserved-mean leaf. Continuous phase lifts then
converge to the corresponding lifted equilibrium. No uniform entrance
time or finite settling deadline has been proved.

The ten certified intervals are disjoint. Fix any one original allowance
tuple and every primitive error and relative source member, retaining the
full prescribed history; vary only the primitive donor origin beta.
Preparation and the own-law finite-time flow depend continuously on beta.
On the zero-total-form state space with common phase rotation removed,
the distinct attracting equilibria have disjoint open basins. Every gap
between successive certified intervals contains at least one
saddle-converging origin for this fixed member: otherwise the connected
interval joining its two certified endpoints would be covered by those
disjoint open basins, although the endpoints have different destinations.
The own-law convergence and finite classification justify the saddle
conclusion. Other gap destinations, exceptional counts and measures are
not determined.

<a id="alternative-weak-contact-source-discrimination"></a>
### 8.4 A common primitive preparation interval separates the two laws

Take the common primitive preparation interval
\(|\beta-\pi/10|\le1/10\), with the original primitive allowances,
commands and durations under each complete law. This entire interval
lies inside the sine theorem's \(|\beta|\le3/5\) source basin.
The sine law therefore selects t_infinity=0. The alternative instead
selects its unique minimum in

\[
 -\frac{7\pi}{60}<t_\infty<-\frac{5\pi}{60}.
\tag{40}
\]

The wrapped asymptotic relative-origin difference exceeds pi/12 for
every allowed pair of own-law source members. This compares the same
declared primitive preparation class and commands, not identical reached
endpoints or transplanted source bounds. Each law carries its own
acquisition, observation backaction, storage and complete subsequent
evolution. The two own-law certificates establish the contrast without
generating another trajectory or re-evaluating a frozen response.

The conclusion is an asymptotic mathematical discriminator. It has no
finite reading time, laboratory clock, independently admitted sensor
or observation-error budget. It does not by itself select which law
describes nature.

## 9. Meaning and implementation boundary

The [sine law](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-source-basin)
at this same eta has attracting origins t=2*k*pi/5. Every one is among
the alternative's exact saddle origins. The ten alternative attracting
origins lie instead in the odd intervals (14). This finite-strength
comparison resolves a constitutive dependence of the collective
registration while preserving the declared constituent profile and old
marked identity. It does not follow from graph symmetry alone.

The stationary proof is local to (4); the source proof separately
extends its use to the larger retained energy sublevel. Neither result
classifies all nonacute or distant organizations. Ten actual-source
intervals now reach alternative minima, while the prototype common
primitive interval has distinct destinations under the two laws.
Own-law acquisition, observation history, attachment work and transient
fine-state transport are part of that conclusion. Matching the isolated
constituent tangent did not transfer the sine source basin.

These labeled registrations are not physical species. Their difference
neither selects a unique fundamental law nor supplies a physical clock,
measurement bridge, autonomous contact occurrence or collective operator.
The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization)
retains those separate obligations.

The [static controls](../../tests/physics/test_organized_alternative_weak_contact.py)
check the derivative bounds,
weighted full-state identities, fivefold Fourier coefficient, contraction
and C1 margins, full-state curvature, actual-source budgets, moving-shape
transport and inward-torque inequalities. Those controls
support the written proof; they do not sample roots or evaluate a
trajectory. This analytical result installs no new runtime law or source
producer. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns its status and any later admission.
