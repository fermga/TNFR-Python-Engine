# Constitutive dependence of weak-contact collective registration

<a id="alternative-weak-contact-registration"></a>

The retained harmonic alternative changes the attracting relative
registrations of the weakly contacted organization. At the fixed spoke
weight eta=1/100000, a declared full-state neighborhood contains twenty
stationary organizations modulo common phase rotation: ten strict
relative minima and ten index-one saddles. All five origins that attract
under the sine law become saddle origins under this alternative.

This is an own-law stationary and stability theorem. It does not transfer
the [sine source-basin certificate](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-source-basin),
evaluate a new source or identify a physical constituent. The alternatives
retain the same support, capacities, clock and constituent profile; their
collective registration geometry is nevertheless different.

| Premise | Fixed declaration |
| --- | --- |
| Constitutive law | Deformation-one potential and current already defined in [the alternative contact owner](ORGANIZED_ALTERNATIVE_CONTACT.md#organized-alternative-contact) |
| Support | The same 54 labeled nodes of C9 x W6; 99 old internal/rim edges of weight one, 45 donor spokes of weight eta=1/100000 |
| Complete dynamics | Both weighted form diffusion and weighted nonlinear phase current, held unit capacity, uniform quarter factors and the original tau clock |
| State and symmetry | All signed forms and primitive circular phases; zero total form, common phase rotation factored out; continuous compatible lifts only for the declared local coordinates |
| Stationary domain | Mean-zero host shape norm at most 6*eta and childwise mean-zero fine phase maximum norm at most 1/100; all donor origins admitted |
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

The domain is a stationary classification neighborhood. No positive
invariance, actual source inclusion or finite settling time is assumed
or proved for it.

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

## 6. Meaning and implementation boundary

The [sine law](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-source-basin)
at this same eta has attracting origins t=2*k*pi/5. Every one is among
the alternative's exact saddle origins. The ten alternative attracting
origins lie instead in the odd intervals (14). This finite-strength
comparison resolves a constitutive dependence of the collective
registration while preserving the declared constituent profile and old
marked identity. It does not follow from graph symmetry alone.

The claim is local to the full-state domain (4); it does not classify
all nonacute or distant stationary organizations. The new minima have
local attractive neighborhoods, but no original acquired-source class
has been placed inside them here. Own-law acquisition, observation
history, attachment work and transient fine-state transport remain
necessary for any such later capture theorem. Matching the isolated
constituent tangent does not transfer the sine source basin.

These labeled registrations are not physical species. Their difference
neither selects a unique fundamental law nor supplies a physical clock,
measurement bridge, autonomous contact occurrence or collective operator.
The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization)
retains those separate obligations.

The [static controls](../../tests/physics/test_organized_alternative_weak_contact.py)
check the derivative bounds,
weighted full-state identities, fivefold Fourier coefficient, contraction
and C1 margins, and full-state curvature inequalities. Those controls
support the written proof; they do not sample roots or evaluate a
trajectory. This analytical result installs no new runtime law or source
producer. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#alternative-weak-contact-source-admission)
owns its status and any later admission.
