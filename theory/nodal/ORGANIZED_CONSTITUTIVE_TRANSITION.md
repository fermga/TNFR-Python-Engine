# A constitutive pitchfork at weak contact

<a id="weak-contact-constitutive-transition-result"></a>

The attracting sine registration has a positive constitutive tolerance
at each fixed sufficiently small contact strength. That tolerance is
not uniform as contact tends to zero. For the already declared
potential family, a controlled joint expansion proves a supercritical
pitchfork: the central registration loses its collective stability and
two attracting registrations emerge, while every other relative phase
direction stays positive.

At each fixed positive eta in the proved range, the same complete law
also has a local scalar center dynamics with an inherited positive
mobility. Its initialization depends on form as well as phase.
Generic critical trajectories approach algebraically; a strong-stable
exception approaches exponentially.

The quantitative domain proved here is
\(0<\eta\le10^{-12}\). It is a narrower mathematical domain than the
endpoint owners' \(\eta=10^{-5}\). This theorem does not classify the
transition at that endpoint contact strength. It neither reruns a
source nor refines an evaluated experimental protocol.

| Premise | Declaration |
| --- | --- |
| Constitutive family | \(V_\lambda(a)=1-\cos a+\lambda(1-\cos9a)^2/(6\,9^4)\), \(0\le\lambda\le1\) |
| Complete support | 54 labeled nodes of C9 x W6; 99 old internal/rim edges of weight one, 45 spokes of held weight eta |
| Complete law | Both weighted rows, held unit capacity, quarter factors and the original tau clock |
| Retained state | All signed forms, primitive circular phases and fine directions, on fixed-total-form and fixed-total-phase leaves |
| Comparison chart | \(\lambda=\kappa_*\mu\eta^3\), with independently supplied eta and lambda; this is not a parameter evolution law |
| Quantitative transition | \(0<\eta\le10^{-12}\), \(\mu\in[9/10,11/10]\), relative origins \(|t|\le1/20\) and their five symmetry copies |
| Local dynamics | Fixed positive eta, finite smooth center manifold, full-state initialization and critical asymptotic relaxation; neighborhoods and transverse rates may depend on eta |
| Evidence | Analytic contraction, exact harmonic coefficients, Cauchy bounds and complete-state curvature; no trajectory, root search, sweep or source replay |

## 1. Complete law, coordinates and scope

Let c=0,...,4 label the host children, c=5=D the donor, and
j=0,...,8 the internal slots. The primitive node order is 9*c+j.
The graph, marked winding conventions and capacities are those of the
[weak-contact endpoint owners](ORGANIZED_ALTERNATIVE_WEAK_CONTACT.md#alternative-weak-contact-registration).
Define

\[
\begin{split}
 \gamma&=\frac1{1023\pi},&
 \tau&=\frac{1023}{1024}t_{\rm original},\\
 W_0(a)&=\frac{(1-\cos9a)^2}{6\,9^4},&
 V_\lambda(a)&=1-\cos a+\lambda W_0(a),\\
 w(a)&=\frac{\sin9a-\tfrac12\sin18a}{2187},&
 j_\lambda(a)&=\sin a+\lambda w(a).
\end{split}
\tag{1}
\]

The support Laplacian L_eta is unnormalized and uses the same weights
as the nonlinear current
\(S_i=\sum_{b\sim i}w_{ib}j_\lambda(\theta_b-\theta_i)\).
The complete, unforced equations and storage are

\[
\begin{split}
 x'&=\frac{-L_\eta x+\gamma S(\theta)}4,&
 \theta'&=\frac\gamma4L_\eta x,\\
 U_\lambda(\theta)&=\sum_{\{i,b\}}w_{ib}V_\lambda(\theta_b-\theta_i),&
 H&=\frac12x^TL_\eta x+U_\lambda,\\
 H'&=-\frac14\|L_\eta x\|^2.
\end{split}
\tag{2}
\]

Primes in the evolution rows mean tau. Unit capacity is held, so the
first row supplies the nodal pressure rather than reconstructing it
from a response. Odd current and Laplacian row sums preserve total
form and total lifted phase. Work on the zero-total-form leaf and a
fixed-total-phase local chart, or equivalently remove common phase
rotation. No laboratory units, later input or support event are
introduced.

Put alpha=2*pi/9, b=2*pi/5 and \(b_c=b(c-2)\). Compatible lifts give

\[
\begin{split}
 \theta_{c,j}&=\alpha(j-4)+\Phi_c+q_{c,j},&
 \sum_jq_{c,j}&=0,\\
 t&=\frac15\sum_{c=0}^4\Phi_c-\Phi_D,&
 h_c&=\Phi_c-\Phi_D-t-b_c,&
 \sum_ch_c&=0.
\end{split}
\tag{3}
\]

Here t is a relative origin, not the clock. The stationary tube is
\(\|h\|_2\le6\eta\), \(\|q\|_\infty\le1/100\), with all t admitted
before the local bifurcation restriction. It retains the six internal
and nine rim winding-one classes and the 32 zero mixed host periods.
It is not asserted to be an invariant source domain. Spoke angles
have no acute restriction.

Eta and lambda are held, independently supplied model parameters.
The substitution \(\lambda=\kappa\eta^3\) below is an analytical
comparison chart near a balance of two torques. It does not derive a
coupling between parameters, select contact strength, or create
parameter dynamics. The endpoint source-capture theorems are not
transferred to intermediate laws.

## 2. A joint complex contraction

Let D be the forward C5 incidence, \(L_5=D^TD\), and
\(P=I-\mathbf1\mathbf1^T/5\). For this construction only, keep the
sine linearization

\[
\begin{split}
 c_b&=\cos b=\frac{\sqrt5-1}{4},&
 s_b&=\sin b,&
 G&=(c_bL_5)^+,\\
 \ell_1&=c_b(2-2c_b),&
 N_0(d)&=\sin(b+d)-\sin b-c_bd .
\end{split}
\tag{4}
\]

The projected host stationarity equation, including every deformation
term, is

\[
 u=-G\left\{
 D^TN_0(Du)+\lambda D^T[w(b+Du)-w(b)]
 +\eta P[\sin(t+b+u)+\lambda w(t+b+u)]\right\}.
\tag{5}
\]

The b inside a spoke vector means \((b_c)_{c=0}^4\); the b inside
rim functions is the scalar gap. Constants disappear under D^T.
Thus (5) is the actual deformed equation, not an unchanged tangent
used as its nonlinear law.

The explicit row of G and bounds from the
[sine stationary proof](ORGANIZED_WEAK_CONTACT_MAINTENANCE.md#weak-contact-wheel-classification)
give

\[
 \|G\|_\infty<g_0=\frac{259}{100},\quad
 \|GD^T\|_\infty<g_1=\frac{389}{100},\quad
 \ell_1^{-1}<\frac{47}{20},\quad
 \frac{2\sin(\pi/5)}{\ell_1}<\frac{69}{25}.
\tag{6}
\]

These are exact properties of the sine matrix chosen in (4);
all deformation terms remain in (5). Set

\[
\begin{split}
 R&=\frac1{100},&
 |\eta|&\le R,&|\kappa|&\le40000,&
 \lambda&=\kappa\eta^3,&|\lambda|&\le\Lambda=\frac1{25},\\
 u^{(1)}_c&=-\frac{\eta}{\ell_1}\sin(t+b_c),&
 r_0&=\frac1{200},&
 \|u-u^{(1)}\|_\infty&\le r_0,\\
 U_1&=\frac{47}{2000},&
 D_1&=\frac{69}{2500},&
 U_0&=\frac3{100},&
 D_0&=\frac1{25}.
\end{split}
\tag{7}
\]

For real t, \(\|u^{(1)}\|_\infty<U_1\) and
\(\|Du^{(1)}\|_\infty<D_1\). The complex ball therefore obeys
\(\|u\|_\infty<U_0\), \(\|Du\|_\infty<D_0\).
All shapes have zero mean.

The following rational majorants make the contraction quantitative:

\[
\begin{split}
 C_h(z)&=\frac{z^2}{2(1-z^2/12)},&
 S_h(z)&=\frac z{1-z^2/6},&
 S_r(z)&=\frac{z^3}{6(1-z^2/20)},\\
 H_0(z)&=\frac{(1-9z)^{-1}+\tfrac12(1-18z)^{-1}}{2187},&
 H_1(z)&=\frac{(1-9z)^{-1}+(1-18z)^{-1}}{243}.
\end{split}
\tag{8}
\]

The first three bound cosh(z)-1, sinh(z) and sinh(z)-z.
For real a and complex v with \(|v|\le z<1/18\),
\(|w(a+v)|\le H_0(z)\) and \(|w'(a+v)|\le H_1(z)\).
These follow from \(|\sin(a+v)|,|\cos(a+v)|\le e^{|v|}\)
and \(e^z\le(1-z)^{-1}\). Also take
\(s_0=24/25>s_b\), \(c_0=31/100>c_b\).
The map in (5) has derivative norm at most

\[
\begin{split}
 Q={}&2g_1[s_0S_h(D_0)+c_0C_h(D_0)]
       +\frac{g_0R}{1-U_0}\\
    &+2g_1\Lambda H_1(D_0)+g_0R\Lambda H_1(U_0)
       <\frac7{20}.
\end{split}
\tag{9}
\]

Its displacement at the center is bounded by

\[
\begin{split}
 B={}&g_1[s_0C_h(D_1)+c_0S_r(D_1)]
        +\frac{g_0RU_1}{1-U_1}\\
    &+g_1\Lambda D_1H_1(D_1)+g_0R\Lambda H_0(U_1)
       <\frac{11}{5000},\\
 B+Qr_0&<r_0.
\end{split}
\tag{10}
\]

All inequalities are rational arithmetic with the declarations above.
The deformation contribution on a rim uses
\(|w(b+d)-w(b)|\le|d|H_1(|d|)\); omitting it would not prove
the joint result. The contraction theorem gives a unique solution,
holomorphic jointly in eta and kappa on their disks, and real for
real parameters. The same estimates work with eta and lambda
independent on \(|\eta|\le R,|\lambda|\le1/25\), so coefficient
extraction in those independent parameters is also justified.

The remaining wheel torque is

\[
 T(t,\eta,\kappa)=\eta\sum_{c=0}^4
 j_{\kappa\eta^3}(t+b_c+u_c).
\tag{11}
\]

It equals the t derivative of the wheel potential after eliminating
the projected shape. Uniformly on the complex parameter disks and
real t,

\[
 |T|\le5R[(1-U_0)^{-1}+\Lambda H_0(U_0)]
       <M=\frac3{50}.
\tag{12}
\]

## 3. Joint fifth-order torque and controlled derivatives

Cyclic host relabeling makes T periodic with period 2*pi/5. Phase
inversion together with rim reflection makes it odd in t. These
symmetries hold for complex parameters as identities of (5). Thus
each Taylor coefficient \(T_n(t,\kappa)\) in
\(T=\sum_{n\ge0}\eta^nT_n\) has only sine harmonics \(\sin(5kt)\).

The coefficient T_n is a trigonometric polynomial of degree at most
5*n. To see the weighted-degree bound, a sine spoke forcing has
eta order one and explicit harmonic degree one. A deformed spoke
forcing has eta order four and explicit harmonic degree at most 18.
A deformed rim term has eta order three and no explicit t harmonic.
Products of lower shape coefficients add their orders and harmonic
degrees. Induction in (5) therefore bounds shape and torque
coefficients by five times their eta order. The bound does not assume
the sine-only parity rule after lambda has become kappa*eta^3.

For completeness, the fifth coefficient can be obtained in independent
eta and lambda coordinates. Put

\[
\begin{split}
 a_\lambda&=j_\lambda'(b)=c_b+\lambda\frac{\sqrt5}{486},&
 G_\lambda&=(a_\lambda L_5)^+,\\
 f_\lambda(t)&=j_\lambda(t+b),&
 \ell_{1,\lambda}&=a_\lambda(2-2c_b),&
 \ell_{2,\lambda}&=a_\lambda(2-2\cos2b),&
 \epsilon&=\frac1{2187}.
\end{split}
\tag{13}
\]

At eta=0, u=0. Its first eta coefficient is
\(-G_\lambda f_\lambda\). The eta coefficient of the torque vanishes
for every lambda, because the frequencies 1, 9 and 18 all sum to
zero over five vertices. Its second coefficient is

\[
 -f_\lambda'(t)^TG_\lambda f_\lambda(t)
 =-\frac{25\lambda}{2187\,\ell_{1,\lambda}}\sin(10t).
\tag{14}
\]

Indeed the finite Fourier identity is

\[
 f_\lambda^TG_\lambda f_\lambda
 =\frac5{2\ell_{1,\lambda}}
       [1+\lambda^2\epsilon^2-2\lambda\epsilon\cos(10t)]
       +\frac{5\lambda^2\epsilon^2}{8\ell_{2,\lambda}}.
\tag{15}
\]

Its t derivative divided by -2 gives (14). The only surviving
cross-harmonic is the pair of frequencies 1 and 9.

At lambda=0, the sine proof establishes that eta orders below five
vanish and that the fifth coefficient is \(A\sin(5t)\), where

\[
\begin{split}
 \varphi&=\frac{1+\sqrt5}{2},&
 A&=5\varphi^{10}>600,\\
 K_0&=\frac{25}{2187\,c_b(2-2c_b)}
        =\frac{25+15\sqrt5}{2187},&
 \kappa_*&=\frac A{2K_0}
        =\frac{2187(105+47\sqrt5)}{40},\\
 11000&<\kappa_*<12000.
\end{split}
\tag{16}
\]

After lambda=kappa*eta^3, the only additional contribution through
order five is the eta-squared, lambda-linear term of (14). The
eta-linear term is identically zero for all lambda, and two lambda
powers already cost six eta orders. Consequently

\[
 T(t,\eta,\kappa)=\eta^5
 [A\sin(5t)-K_0\kappa\sin(10t)]+\mathcal R(t,\eta,\kappa),
\tag{17}
\]

with no orders below five. This uses the established sine coefficient
as an identity, while re-admitting every deformation contribution.
It transfers no endpoint remainder or capture certificate.

Here is a uniform C3 remainder, with a mixed parameter derivative.
Cauchy's estimate from (12) gives
\(\|T_n(\cdot,\kappa)\|_\infty\le M/R^n\).
Each sine coefficient has magnitude at most \(2M/R^n\), and its
index k is at most n. For derivative order j=0,...,3,

\[
 \|\partial_t^jT_n\|_\infty
 \le\frac{2M5^j n^{j+1}}{R^n}.
\tag{18}
\]

Write \(h=\eta/R\). For \(0<\eta\le10^{-12}\), summing n>=6 yields

\[
\begin{split}
 \left\|\partial_t^j\frac{\mathcal R}{\eta^5}\right\|_\infty
 &\le E_j(\eta)
 :=\frac{3\,5^j6^{j+1}}{25}
       \frac{\eta/R^6}{1-2\eta/R},\\
 (E_0,E_1,E_2,E_3)
 &<\left(\frac{73}{100},22,650,19500\right)d_\eta,
 \qquad d_\eta:=\frac{\eta}{10^{-12}}\le1.
\end{split}
\tag{19}
\]

For the sum, use
\((l+6)^{j+1}\le6^{j+1}(7/6)^{(j+1)l}\) and
\((7/6)^4<2\). This proves absolute convergence after all stated
derivatives. The estimates remain uniform for complex
\(|\kappa|\le40000\). A kappa Cauchy circle of radius 20000 at
each real \(|\kappa|\le20000\) therefore gives

\[
 \left|\partial_\kappa\partial_t^j
               \frac{\mathcal R}{\eta^5}\right|
 \le\frac{E_j(\eta)}{20000}.
\tag{20}
\]

Normalize by \(\kappa=\kappa_*\mu\) and set

\[
 f(t,\mu,\eta):=\frac{T}{A\eta^5}
     =\sin(5t)-\frac\mu2\sin(10t)+r(t,\mu,\eta).
\tag{21}
\]

For \(0\le\mu\le11/10\), (16) keeps kappa inside the required
disk. Equations (19)--(20) imply, in particular,

\[
 |r_t|<\frac{d_\eta}{20},\qquad
 |r_{t\mu}|<\frac{d_\eta}{20},\qquad
 |r_{ttt}|<40d_\eta.
\tag{22}
\]

The C0 and C2 bounds are also given in (19), divided by A.
Thus this is a controlled C3 expansion with mixed-parameter evidence,
not a derivative inferred from a value bound. Dividing the analytic
series by eta^5 also extends f analytically to eta=0; this is an
auxiliary parameter extension, not an assertion about dynamics on
the disconnected eta=0 support.

## 4. A unique quantitative critical curve

The central registration t=0 is exact by oddness. Its normalized
curvature is

\[
 G(\mu,\eta):=f_t(0,\mu,\eta)
       =5(1-\mu)+r_t(0,\mu,\eta),\qquad
 -\frac{101}{20}<G_\mu<-\frac{99}{20}.
\tag{23}
\]

G is positive at mu=9/10 and negative at mu=11/10. The intermediate
value theorem and strict monotonicity give exactly one
\(\mu_c(\eta)\) in this interval. The implicit function theorem,
applied to the analytic extension at eta=0 and to (23), makes the
curve analytic on the admitted range with \(\mu_c(0)=1\). Explicitly,

\[
\begin{split}
 |\mu_c(\eta)-1|
 &\le\frac{E_1(\eta)}{5A}<\frac{d_\eta}{100},\\
 \lambda_c(\eta)&=\kappa_*\mu_c(\eta)\eta^3,\\
 |\lambda_c(\eta)-\kappa_*\eta^3|
 &\le\frac{\eta^3E_1(\eta)}{10K_0}
      <\frac{\kappa_*}{100}\eta^3d_\eta .
\end{split}
\tag{24}
\]

In particular the controlled shift is O(eta^4), rather than an
unbounded formal error at the competing fifth order.
For \(0\le\mu\le9/10\), (22) gives \(G>9/20\).
Together with (23), the central registration therefore has positive
collective curvature for every \(0\le\mu<\mu_c\), zero at mu_c,
and negative curvature for \(\mu_c<\mu\le11/10\).

## 5. The emerging branches are a supercritical pitchfork

Work on \(|t|\le1/20\), \(\mu\in[9/10,11/10]\).
Here \(\cos(10t)\ge7/8\), and (22) yields

\[
 f_{ttt}=-125\cos(5t)+500\mu\cos(10t)+r_{ttt}
       >-125+450\frac78-40>200.
\tag{25}
\]

Odd analyticity gives an exact factorization

\[
 f(t,\mu,\eta)=t\,B(t^2,\mu,\eta).
\tag{26}
\]

For \(s\in[0,1/400]\), it has

\[
\begin{split}
 B(0,\mu,\eta)&=G(\mu,\eta),\\
 B_s(s,\mu,\eta)
 &=\frac12\int_0^1\int_0^1
       a^2 f_{ttt}(ab\sqrt s,\mu,\eta)\,db\,da
       >\frac{100}{3}.
\end{split}
\tag{27}
\]

This integral also proves the derivative bound at s=0 by continuity.
It is not obtained by dividing a value-only remainder near t=0.

Now restrict \(|\mu-\mu_c|\le1/100\), which lies inside the preceding
mu interval. Equation (23) gives
\(|B(0)|<101/2000\), with the sign opposite to mu-mu_c.
At the outer boundary,

\[
 B(1/400)>\frac1{12}-\frac{101}{2000}
             =\frac{197}{6000}>0.
\tag{28}
\]

Strict monotonicity in s now gives the complete local stationary
classification:

- Below critical, t=0 is the only stationary origin in this interval,
  and its collective curvature is positive.
- At critical, t=0 is still the only stationary origin, with one
  zero collective curvature and a strictly positive quartic term.
- Above critical, t=0 has negative collective curvature, and there
  are exactly two additional origins \(t=\pm\sqrt{s(\mu,\eta)}\).
  They satisfy \(0<s<1/400\), and their curvature is positive because
  \(f_t=2sB_s>0\).

The implicit function theorem applies to B at its critical zero.
For each fixed positive eta,

\[
\begin{split}
 s(\mu,\eta)
 &=c_\eta(\mu-\mu_c)+O((\mu-\mu_c)^2),\\
 c_\eta&=-\frac{B_\mu(0,\mu_c,\eta)}
                    {B_s(0,\mu_c,\eta)},&
 \frac7{100}&<c_\eta<\frac1{10},&
 c_\eta&=\frac2{25}+O(\eta).
\end{split}
\tag{29}
\]

Indeed \(B_\mu\in(-101/20,-99/20)\), while
\(\mu_c\in(99/100,101/100)\) and (22) imply
\(55<B_s(0,\mu_c,\eta)<70\).
At eta=0 the exact values are \(B_\mu=-5\) and \(B_s=125/2\).
This proves the branch direction and square-root splitting, not just
the existence of a central curvature crossing. Cyclic symmetry gives
the same local theorem around each of the five origins \(2k\pi/5\).
No global count of all stationary registrations for intermediate
lambda is asserted.

## 6. Normal curvature and the complete two-row dynamics

All normal directions must be re-admitted for the intermediate law.
For real \(0\le\lambda\le1\), the uniform current bounds are

\[
 |j_\lambda|\le\frac{1459}{1458},\quad
 |j_\lambda'|\le J_1=\frac{245}{243},\quad
 |j_\lambda''|\le J_2=\frac{10}{9},\quad
 j_\lambda'(b)\ge c_b,\quad
 j_\lambda'(\alpha)=\cos\alpha .
\tag{30}
\]

The real host map with linearization \(a_\lambda L_5\) consequently
has \(\|G_\lambda\|_2<5/2\). For
\(0<\eta\le10^{-5}\), its image and derivative on
\(\|u\|_2\le6\eta\) satisfy

\[
\begin{split}
 \|\mathcal K(u)\|_2
 &\le\frac52\left[4J_2(6\eta)^2+
             \eta\sqrt5\frac{1001}{1000}\right]<6\eta,\\
 \|D_u\mathcal K\|_2
 &\le\frac52\left[48J_2+\frac{101}{100}\right]\eta
             <\frac1{500}.
\end{split}
\tag{31}
\]

Thus the real projected shape is unique in this ball. In the smaller
joint domain it belongs to the complex ball of (7), since
\(\|u-u^{(1)}\|_\infty<(6+47/20)\eta<r_0\). The real shape is
therefore exactly the analytic branch used above, rather than a
separately chosen solution.

Let \(C_6(v)=1-v^2/2+v^4/24-v^6/720\).
On the stationary tube and on its segments to q=0,
internal gaps have absolute value below 18/25, while rim gaps lie
in (0,13/10). The inequality
\(j_\lambda'(a)\ge\cos a-2/243\) gives the exact rational bounds

\[
 C_6(18/25)-\frac2{243}>\frac7{10},\qquad
 C_6(13/10)-\frac2{243}>\frac14.
\tag{32}
\]

The 48 fine directions have internal Laplacian gap greater than 2/5.
The spoke-star Laplacian has norm six. Even allowing negative spoke
curvatures, their projected phase Hessian is bounded below by

\[
 \left[\frac7{10}\frac25-6\eta J_1\right]I>\frac14I.
\tag{33}
\]

At q=0 the fine gradient is exactly zero for every collective
configuration: internal currents cancel and wheel currents are
slot-constant. Integrating (33) therefore excludes any stationary
q different from zero in this tube. Similarly, the four mean-zero
host directions have wheel curvature
\(L_5/4-\eta J_1I>I/5\). Their stationarity uniquely gives h=u(t).
At a complete equilibrium, connected support and the phase row first
give x=0; the form row then gives these phase stationarity equations.
No fine coordinate was discarded before checking its equation.

The Schur complement of the four host directions is the reduced
wheel curvature \(T'=A\eta^5 f_t\). In slot-first node order the
full carrier Hessian is

\[
 M_{54}=\cos\alpha\,L_9\otimes I_6+I_9\otimes M_{\rm wheel}.
\tag{34}
\]

Thus the relative phase Hessian has 53 positive directions below
critical and on the two emerging branches; at critical it has 52
positive directions and one zero; on the central branch above
critical it has 52 positive directions and one negative direction.
These statements include every primitive phase direction apart from
the exact common-rotation symmetry.

On the fixed-mean relative state space, the actual 106-dimensional
linearized two-row dynamics has the symmetric quadratic pencil

\[
 P(s)=s^2L_\eta^{-1}+\frac s4I+\frac{\gamma^2}{16}M_{54}.
\tag{35}
\]

Testing an eigenvector gives a scalar quadratic with positive
coefficients for s-squared and s. If M is positive definite, every
eigenvalue has negative real part. If M has one negative direction,
P(s) increases strictly for real s>0 from that one negative direction
to a positive matrix, giving exactly one positive eigenvalue. The
imaginary part of the scalar identity excludes nonreal eigenvalues
with nonnegative real part.

At critical, M is positive semidefinite with a one-dimensional
kernel. The Schur complement of its positive subspace is
\(s/4+O(s^2)\), so the zero eigenvalue of the complete first-order
system is algebraically simple. The other 105 eigenvalues have
negative real part. Consequently the attracting branches are locally
exponentially stable, and the central branch above critical has one
unstable direction. Normal stability has been checked for the whole
law, not inferred from a wheel-only eigenvalue.

## 7. The critical state is attracting, with quartic restoration

The zero linear eigenvalue does not by itself decide nonlinear
stability. At mu_c, (27) and B(0)=0 give
\(f(t)/t=B(t^2)>(100/3)t^2\) for t different from zero.
For the reduced wheel potential \(\mathcal W\), this implies

\[
 \mathcal W(t)-\mathcal W(0)
       \ge\frac{25}{3}A\eta^5t^4
       \quad (|t|\le1/20).
\tag{36}
\]

The complete phase potential contains nine wheel copies. Integrating
the fine and host normal bounds on the same fixed-mean chart gives

\[
 U_\lambda(\theta)-U_\lambda(\theta_*)
 \ge\frac18\|q\|_2^2+\frac9{10}\|h-u(t)\|_2^2
       +9[\mathcal W(t)-\mathcal W(0)] .
\tag{37}
\]

The fixed internal carrier potentials cancel in this difference.
Adding the positive form storage in (2) produces a strict local
minimum at the critical equilibrium. For each fixed positive eta,
a sufficiently small compact sublevel remains inside the local chart
and stationary tube. Storage is nonincreasing there. In the invariant
part of H'=0, connected support and zero total form give x=0, and
the form row then requires a stationary phase state. Sections 5--6
leave only the central critical equilibrium in this neighborhood.
The elementary compact LaSalle argument proves Lyapunov stability
and asymptotic attraction.

The critical state is not locally exponentially stable: its complete
linearization has the simple zero eigenvalue just proved. No uniform
settling time or decay rate as eta tends to zero is claimed.

<a id="constitutive-transition-local-dynamics"></a>

## 8. The inherited local dynamics of the soft registration

Fix one positive eta in the proved interval. This section derives a
local dynamical consequence of the same complete law in the original
tau clock. Every neighborhood, remainder constant and transverse
tracking rate below may depend on that fixed eta. There is no
uniform-in-eta center manifold, source-entry theorem, or finite-time
approximation claim.

Write \(\sigma=\mu-\mu_c(\eta)\). Let
\(\Theta(t,\sigma)\) be the stationary carrier shape after eliminating
the four host shape coordinates, with the total phase fixed.
It is a curve of projected stationary shapes, not a curve of complete
equilibria: its remaining torque can be nonzero. Its phases are

\[
\begin{split}
 \Theta_{c,j}(t,\sigma)
 &=\alpha(j-4)+b_c+\frac t6+u_c(t,\sigma)
       +\text{fixed common origin}\quad(c<5),\\
 \Theta_{D,j}(t,\sigma)
 &=\alpha(j-4)-\frac{5t}{6}
       +\text{the same fixed common origin}.
\end{split}
\tag{38}
\]

At criticality put \(\theta_*=\Theta(0,0)\),
\(w=\Theta_t(0,0)\), and \(N=w^Tw\). Then

\[
 w_{c,j}=\frac16+u_{c,t}(0,0),\quad
 w_{D,j}=-\frac56,\qquad
 N=\frac{15}{2}+9\|u_t(0,0)\|_2^2.
\tag{39}
\]

Differentiating the real contraction in (31) gives
\[
 \|u_t\|_2
 \le\frac{(5/2)\sqrt5(101/100)}{1-1/500}\eta<6\eta.
\tag{40}
\]
This derivative belongs to the actual constitutive shape at the
critical parameter.

In the state order \((\theta-\theta_*,x)\), the critical Jacobian and
its right and normalized left zero vectors are

\[
\begin{split}
 J&=\begin{pmatrix}
        0&\gamma L_\eta/4\\
        -\gamma M_{54}/4&-L_\eta/4
       \end{pmatrix},&
 v&=(w,0),&
 \ell&=\frac{(w,\gamma w)}{N},\\
 Jv&=0,&\ell J&=0,&\ell v&=1 .
\end{split}
\tag{41}
\]

Here \(M_{54}w=0\) follows from the critical Schur complement;
the common-rotation zero mode has already been removed. For the
linear coordinate
\(\zeta=w^T(\theta-\theta_*+\gamma x)/N\), the full nonlinear
equations have the exact projected row

\[
 \zeta'=-\frac{\gamma^2}{4N}\,
                w^T\nabla U_\lambda(\theta).
\tag{42}
\]

This row still depends on hidden phase coordinates. Projection is
not yet a scalar closure.

### 8.1 An invariant center manifold, with matched hypotheses

The vector field is analytic on this finite-dimensional relative
chart. Its critical spectrum consists of one simple zero and 105
strictly stable eigenvalues. The local center-manifold theorem
therefore gives a finite \(C^k\) invariant curve for any prescribed
finite k, chosen here at least five. Applying the same theorem after
adjoining \(\sigma'=0\) gives the local parameter family. These are
the hypotheses of the local theorem and parameter construction in
[Nagata, Lecture 25, Theorem 3.7 and the final parameter extension](https://personal.math.ubc.ca/~nagata/m552/lec25.pdf).

The reflection symmetry acts as \(t\mapsto-t\). An invariant,
reflection-symmetric cutoff in a local center/stable construction
gives an equivariant center manifold; its scalar row is odd in t.
The exact carrier subspace, in which fine phases and fine forms
vanish, is itself invariant: internal currents cancel, and the wheel
rows are constant in each slot. The center manifold can first be
constructed in this ten-dimensional carrier system and is then an
invariant manifold of the full system. Fine perturbations retain
their stable directions; they are not assumed absent from nearby
initial states.

The center curve need not be unique or analytic. The claims below
concern its finite-order coefficients and the tracking of nearby
states, with physical t as its coordinate. The derivative of that
coordinate along v is one, so it is a valid local parameter.
The graph \((\Theta(t,\sigma),0)\) is not invariant away from its
torque zeros: its phase row is zero while its form row is nonzero.
It cannot replace the center manifold.

### 8.2 The normal-form coefficients come from the actual two rows

Let \(t'=g(t,\sigma)\) be the center row. At criticality odd symmetry
and the zero eigenvalue give \(g(t,0)=O(t^3)\).
Write the center phase and form curves as
\(\Theta_{\rm c}(t)\), \(x_{\rm c}(t)\). The phase row implies

\[
 x_{\rm c}(t)=\frac4\gamma L_\eta^{-1}
                 \Theta_{\rm c}'(t)\,g(t,0)=O(t^3).
\tag{43}
\]

Invariance in the form row then forces the normal phase gradient to
vanish through degree two. The invertible normal Hessian from
section 6 consequently gives
\(\Theta_{\rm c}(t)-\Theta(t,0)=O(t^3)\), at the same physical t.
It is this agreement of finite-order expansions, not invariance of
the stationary graph, that permits use of its reduced potential.

Eliminating x from the full equations gives the exact second-order
phase equation on the relative space,

\[
 L_\eta^{-1}\theta''+\frac14\theta'
                       +\frac{\gamma^2}{16}\nabla U_\lambda(\theta)=0.
\tag{44}
\]

On the critical center curve its inertial term is O(t^5).
The cubic normal displacement changes the cubic gradient by
\(M_{54}\) times that displacement; contraction with w annihilates
this term. On the stationary curve,
\(w^T\nabla U_\lambda(\Theta(t,0))=9T(t)\):
the fine and projected host gradients vanish, and the host-shape part
of w is orthogonal to the remaining mean gradient.
Projection of (44) at cubic order therefore gives

\[
 g(t,0)=-b_\eta t^3+O(t^5),\qquad
 b_\eta=m_\eta A\eta^5 B_s(0,\mu_c,\eta)>0,
\tag{45}
\]
where the inherited mobility is
\[
 m_\eta=\frac{9\gamma^2}{4N}
       =\frac{\gamma^2}{4(5/6+\|u_t(0,0)\|_2^2)},\qquad
 \frac{\gamma^2}{4(5/6+36\eta^2)}
       <m_\eta\le\frac{3\gamma^2}{10}.
\tag{46}
\]

The positive coefficient is fixed by the complete damping row and
the physical coordinate geometry. It is neither a new primitive
capacity nor an independently assigned time rescaling.
Equation (45) does not assert the exact scalar identity
\(t'=-m_\eta T(t)\) at all orders.

For the parameter-linear term, differentiate the simple critical
eigenvalue. At the central equilibrium the stationary tangent
satisfies \(w(\mu)^TM(\mu)w(\mu)=9T_t(0,\mu)\).
Since \(M(\mu_c)w=0\), differentiation at criticality gives
\(w^TM_\mu w=9T_{t\mu}\). The left/right vectors in (41)
therefore yield

\[
 a_\eta=-m_\eta T_{t\mu}(0,\mu_c)
       =-m_\eta A\eta^5B_\mu(0,\mu_c,\eta)>0 .
\tag{47}
\]

The resulting local row in the unchanged tau clock is

\[
\begin{split}
 t'&=a_\eta\sigma t-b_\eta t^3
       +O_\eta(\sigma^2|t|+|\sigma||t|^3+|t|^5),\\
 \frac{99}{20}m_\eta A\eta^5
       &<a_\eta<\frac{101}{20}m_\eta A\eta^5,\\
 55m_\eta A\eta^5
       &<b_\eta<70m_\eta A\eta^5,\qquad
 \frac{a_\eta}{b_\eta}=c_\eta .
\end{split}
\tag{48}
\]

The subscript on the remainder records fixed-eta local constants.
The dynamic branch ratio agrees with (29), an independent check of
the pressure, damping and coordinate factors. The slow eigenvalue
is \(a_\eta\sigma+O_\eta(\sigma^2)\) at the central equilibrium and
\(-2a_\eta\sigma+O_\eta(\sigma^2)\) on the emerging branches.
These are local spectral statements, not uniform relaxation times.

<a id="constitutive-transition-initialization"></a>

## 9. Hidden initialization and critical relaxation

At criticality, the stable spectrum has a positive separation from
zero for each fixed eta. In an adapted norm, choose
\(\omega_\eta>0\) such that the stable semigroup is bounded by
\(e^{-\omega_\eta\tau}\). After the spectral change of coordinates,
the nonlinear remainder and its derivative vanish at the origin.
A smooth local cutoff can therefore make its Lipschitz constant
arbitrarily small compared with this separation, while leaving
the actual equations unchanged on a smaller neighborhood.

Taking the stable block as A, the zero center block as B, and
\(\alpha=-\omega_\eta,\ \beta=0\), this satisfies
\(2\operatorname{Lip}(H)<\beta-\alpha\) in the
[foliation and exponential-tracking theorem of Chung and Jolly, Theorem 1.1 and Proposition 1.2](https://arxiv.org/pdf/1211.0768).
For the finite \(C^k\) center curve, choose the cutoff still smaller
so the center growth weights required through derivative order k
remain strictly below the stable decay. Differentiating the
Lyapunov--Perron fixed-point equation then has convergent stable
integrals and invertible contraction derivatives through that order.
This is the finite-smoothness center-manifold construction used in
section 8.1; the cited Lipschitz foliation theorem by itself is not
being used to assert smoothness or analyticity of the tracking map.
Section 7 supplies a smaller invariant neighborhood of the actual
critical law, so its forward solutions never use the modified
equations outside the cutoff. This is a proof extension, not a new
physical or runtime law.

Every sufficiently close full initial state consequently tracks a
center-manifold trajectory exponentially in the actual tau clock.
Denote the physical t coordinate of that trajectory at time zero by
\(\Pi_\eta(\theta_0,x_0)\). It is a local initialization map from
the full state, not generally the observed t_0.

The uniform graph slopes also give the needed linear-size initial
bound, without requiring a smooth projection theorem. Write p for
the stable spectral coordinates and c for the center coordinate.
If the stable leaves and center graph have Lipschitz slopes at most
\(q<1\), their intersection \((p_+,c_+)\) satisfies
\[
 |p_+|\le q|c_+|,\qquad
 |c_+-c_0|\le q|p_+-p_0|,\qquad
 |c_+|\le\frac{|c_0|+q|p_0|}{1-q^2}.
\tag{48a}
\]
Thus the tracking initial state is O(norm(z_0)).
On the critical center curve, (45) makes its scalar magnitude
nonincreasing in a sufficiently small interval. Shrinking the initial
neighborhood consequently keeps both the tracking center orbit and
the actual orbit inside the region where the cutoff equals one.

Its leading term is fixed by (41):

\[
 \Pi_\eta(\theta_0,x_0)
 =\frac{w^T(\theta_0-\theta_*+\gamma x_0)}{N}
       +O_\eta(\|(\theta_0-\theta_*,x_0)\|^2).
\tag{49}
\]

One can obtain the quadratic remainder without assuming an analytic
foliation. Let z and z_c be a full orbit and its exponentially
tracking center orbit. Their projected difference satisfies
\[
 \ell[z(0)-z_{\rm c}(0)]
 =-\int_0^\infty
       \ell[\mathcal N(z(s))-\mathcal N(z_{\rm c}(s))]\,ds ,
\tag{50}
\]
where \(\mathcal N\) is the nonlinear remainder after the critical
linearization. The center orbit decreases in magnitude by (45);
(48a), its smooth graph and exponential tracking keep both
orbits O(norm(z(0))). Since \(D\mathcal N=O(\|z\|)\), the
exponentially decaying difference makes (50) O(norm(z(0))^2).
Finally \(\ell z_{\rm c}(0)=t_{\rm c}(0)+O(t_{\rm c}(0)^2)\).
These facts prove (49).

The set \(\Pi_\eta=0\) is the local 105-dimensional strong-stable
manifold of the equilibrium. It is tangent to the kernel of the
linear functional in (49); it is not asserted equal to that linear
kernel. On this exceptional set the full state approaches the
equilibrium exponentially.

If \(\Pi_\eta\ne0\), its center trajectory eventually obeys
\(t'=-b_\eta t^3+O_\eta(t^5)\), remains on the corresponding side
of zero and approaches zero. Therefore

\[
 \frac{d}{d\tau}\frac1{t^2}
       =2b_\eta+O_\eta(t^2),\qquad
 t(\tau)\sim
       \frac{\operatorname{sgn}\Pi_\eta}
                    {\sqrt{2b_\eta\tau}} .
\tag{51}
\]

The exponentially small tracking difference transfers this asymptotic
formula to the physical relative origin of the full trajectory.
Equivalently \(2b_\eta=m_\eta T_{ttt}(0,\mu_c)/3\).
The limit is as tau tends to infinity at fixed positive eta.
It is not a joint eta/time estimate or a finite experimental
settling-time prediction.

### 9.1 Equal phase observations can have opposite subsequent approaches

There is a concrete local initialization witness. Keep every phase
equal to the critical equilibrium and choose the two signed form
states
\[
 \theta_0=\theta_*,\qquad x_0^\pm=\pm\varepsilon w,
       \qquad\varepsilon>0.
\tag{52}
\]
They obey the zero-total-form constraint and share all initial
phase observations, including t_0=0 and the marked identity.
For sufficiently small epsilon both lie in the preceding local
domain. Equation (49) gives
\[
 \Pi_\eta(\theta_*,x_0^\pm)
       =\pm\gamma\varepsilon+O_\eta(\varepsilon^2).
\tag{53}
\]
Thus their asymptotic approach signs are opposite. Even their initial
physical relative-origin rates already differ:
\[
 (t'_0)^\pm=\pm\frac32\gamma\eta\varepsilon .
\tag{54}
\]
To check (54) directly, the host mean of w is 1/6, the donor mean is
-5/6, and the exact mean phase row gives
\(t'=(3/2)\gamma\eta(\bar x_H-\bar x_D)\).
No selected trajectory was integrated to construct this witness.

The scalar center law is therefore a justified local description
with a full-state initialization and an exponentially decaying
transverse discrepancy. It is not an exact instantaneous quotient
of all nearby states onto their phase origin. This distinguishes
inherited dynamics from stationary elimination and preserves the
hidden-initial-state obligation in the ontology.

## 10. Meaning and validation boundary

For every eta in the proved positive interval, the central
registration attracts for a strictly positive interval
\(0\le\lambda<\lambda_c(\eta)\), and the critical state remains
asymptotically attracting with quartic restoration. Immediately above
lambda_c, the central state becomes a saddle and the two new
registrations attract. The explicit curve
\(\lambda_c=\kappa_*\eta^3+O(\eta^4)\) tends to zero.
There can therefore be no positive deformation tolerance independent
of eta that preserves central attraction throughout all sufficiently
weak contacts. Explicitly, for any proposed tolerance lambda_0>0,
choose eta in the proved interval small enough that
\(\lambda_+(\eta)=\kappa_*[\mu_c(\eta)+1/100]\eta^3<\lambda_0\).
The central state at this supplied deformation is a saddle.
This countersequence proves nonuniform tolerance; it does not assert
a fixed-lambda asymptotic classification outside the bounded mu chart.

This is a constitutive dependence result for a conditional formed
organization. It does not select a unique TNFR law, identify a
physical force or species, derive support occurrence or scale
selection, or determine the basins of the original sources under
intermediate laws. Fixed-contact continuity and a nonuniform
small-contact limit are distinct statements; neither licenses
interchanging stationary sets at eta=0 with those at positive eta.

The proof range \(0<\eta\le10^{-12}\) is explicit. The existing
sine and deformation-one theorems at eta=1e-5 remain valid in their
own domains, but do not extend this quantitative transition theorem
to that larger contact strength.

Static controls in the
[transition research tests](../../tests/physics/test_organized_constitutive_transition.py)
check the finite Fourier coefficient, rational contraction and
remainder inequalities, derivative margins and full-state curvature.
The [local-dynamics tests](../../tests/physics/test_organized_constitutive_dynamics.py)
independently check the two-row projection, inherited mobility,
invariant-manifold coefficients, initialization and critical decay
identities.
No source producer, frozen response, trajectory, parameter scan or
numerical root search is used. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#weak-contact-constitutive-transition)
owns task status and subsequent research.
