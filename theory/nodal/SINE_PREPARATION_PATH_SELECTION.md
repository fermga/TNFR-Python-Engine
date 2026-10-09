# Source-local capture and a transient-winding obstruction on an energy-matched path

<a id="sine-preparation-path-selection"></a>

One continuous family of preparations, under one fixed complete law, has
source intervals that provably acquire different winding classes. An interior
interval has actual winding two at the declared observation time but cannot
retain that winding. Thus the finite-transit winding is not a sufficient
selector of the eventual organized state, even when initial phase, means and
storage are held exactly equal for each correlated source realization.

This result extends the
[matched preparation pair](SINE_PREPARATION_MODE_SELECTION.md#sine-preparation-mode-selection).
It supplies two positive capture corridors and one negative selector result;
it does not locate a basin boundary or identify the eventual class in the
negative corridor. All assessments below are analytic scalar bounds. No
trajectory, reverse-flow source, time-coefficient producer or parameter sweep
is used.

<a id="sine-preparation-path-law"></a>
## Fixed complete law and source budgets

Use the isolated unit C9 with nodes \(0,\ldots,8\), cycle edges
\(j--(j+1\bmod9)\), unnormalized Laplacian \(L\), degree matrix \(2I\),
held capacities one, signed form \(x\), and primitive circular phase with
continued lifts \(\theta\). The complete law, clocks and storage are
\[
 \begin{gathered}
 e=1023/1024,\quad w=1/1024,\quad \tau=et,\quad
 \gamma=1/(1023\pi),\quad A=L/2,\quad \eta=\gamma^2,\\
 x'=-Ax+\gamma f(\theta),\qquad \theta'=\gamma Ax,\qquad
 f_i(\theta)=\tfrac12\sum_{j\sim i}\sin(\theta_j-\theta_i),\\
 H(x,\theta)=\tfrac12x^TLx+U(\theta),\qquad
 U(\theta)=\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)],\qquad
 H'=-\tfrac12\|Lx\|_2^2 .
 \end{gathered}\tag{1}
\]
Both original-time rows are \(e\) times their structural-time rows. Both
ordinary means, equivalently the degree-weighted means on this regular
support, are conserved. Support, pressure, capacity and clock are supplied;
there are no events, controllers or state resets during acquisition.
The sine law is fixed throughout: no constitutive parameter is varied.

Retain the exact preparation constants
\[
 \begin{gathered}
 M=512,\qquad m=\frac{2046}{9}\left(\frac{355}{113}\right)^2,\qquad
 \epsilon=10^{-10},\qquad T=200,\qquad B_{\rm source}=10^{15},\\
 r_j=j-4,\qquad h=e_3-e_4,\qquad
 a=(9M+M^{-1})h+r,\qquad b=(9M-M^{-1})h+2r .
 \end{gathered}\tag{2}
\]
The original time is \(204800/1023\). The earlier reference radius
\(r_c=1/12\) remains a possible eventual recovery radius, but the broad
corridors below use an acute-cell capture criterion at \(T\); they do not
assert entry into that small Euclidean ball at \(T\).

The graph identities
\[
 r^TLr=72,\quad h^TLh=6,\quad h^TLr=0,\quad
 \mathbf1^Tr=\mathbf1^Th=0,\quad
 a^TLa=b^TLb=486M^2+180+6/M^2
 \tag{3}
\]
are the algebraic basis for exact storage matching.

<a id="sine-preparation-cayley-path"></a>
## A continuous exact energy isometry, including the errors

Put
\[
 \begin{gathered}
 u=a+b=18Mh+3r,\qquad v=b-a=-2h/M+r,\\
 U_L=u^TLu=648(3M^2+1),\qquad
 V_L=v^TLv=24(3M^2+1)/M^2,\qquad
 u^TLv=0,\qquad \kappa=V_L/U_L=1/(27M^2),\\
 \mathcal K=\frac{v u^TL-u v^TL}{U_L}.
 \end{gathered}\tag{4}
\]
Direct multiplication gives
\(\mathcal Ku=v\), \(\mathcal Kv=-\kappa u\),
\(\mathcal K^3=-\kappa\mathcal K\),
\(\mathcal K^TL=-L\mathcal K\), and
\(\mathcal K\mathbf1=\mathbf0=\mathbf1^T\mathcal K\).
For the preparation parameter \(s\in[0,1]\), define
\[
 Q_s=(I-s\mathcal K)^{-1}(I+s\mathcal K)
 =I+\frac{2s}{1+\kappa s^2}\mathcal K
     +\frac{2s^2}{1+\kappa s^2}\mathcal K^2 .
 \tag{5}
\]
The polynomial formula proves existence without a numerical matrix inverse.
It implies \(Q_s^TLQ_s=L\), preservation of both mean functionals,
\(Q_0=I\), and \(Q_1a=b\). The map is continuous and invertible.
It is an isometry of preparation energy, not a symmetry of the complete
form-and-phase flow.

For every single latent pair
\[
 |\xi_j|,|\zeta_j|\le\epsilon,\qquad
 \mathbf1^T\xi=\mathbf1^T\zeta=0,
 \tag{6}
\]
the entire source path is
\[
 x_s(0)=Q_s(ma+\xi),\qquad \theta_s(0)=\zeta .
 \tag{7}
\]
Thus initial phase, both means and complete storage are exactly the same
for every \(s\) in that realization. Each fixed-\(s\) source family retains
the full sixteen-dimensional relative interior of its form/phase error
domain. Cartesian marginal bounds do not replace the correlations in (7).
In particular \(Q_1\) need not equal the previous pair's metric reflection
on errors; its uncertainty is re-admitted here.

Uniform admission follows from
\[
 \begin{gathered}
 \|u\|_\infty=18M,\quad \|v\|_\infty=4,\quad
 \|Lu\|_1=144M+54,\quad \|Lv\|_1=18+16/M,\\
 \|\mathcal K\|_\infty\le
 \frac{4(144M+54)+18M(18+16/M)}{U_L}
 =\frac{461304}{509608584}<\frac1{1024},\\
 \|Q_s\|_\infty\le
 \frac{1+\|\mathcal K\|_\infty}{1-\|\mathcal K\|_\infty}
 < C_Q:=\frac{1025}{1023}.
 \end{gathered}\tag{8}
\]
The last inequality uses the Neumann expansion of \(I-s\mathcal K\).
Hence every mapped form error has magnitude at most \(C_Q\epsilon\);
phase errors still have magnitude at most \(\epsilon\).

The common source cost is bounded once, at \(s=0\):
\[
 H(x_s(0),\zeta)\le
 \frac12\sum_{\{i,j\}}(m|a_j-a_i|+2\epsilon)^2+18\epsilon^2
 <320680599854722<10^{15}.
 \tag{9}
\]
Equation (5), not a second box estimate, transfers this same cost to all
\(s\). Every source has initial winding zero.

<a id="sine-preparation-path-proxy"></a>
## Source-local predictors with complete transient errors

Write \(Q_sa=p(s)h+q(s)r\). With
\(D_s=1+\kappa s^2\), \(P_0=9M+M^{-1}\) and
\(C=18M\kappa-2/M=-4/(3M)\), the exact scalar formulas are
\[
 p(s)=\frac{P_0+Cs-\kappa P_0s^2}{D_s},\qquad
 q(s)=\frac{1+s+\kappa(3s-s^2)}{D_s}
     =1+s+\frac{\kappa s(1-s)(s+3)}{D_s}.
 \tag{10}
\]
Here \(p\) decreases from \(9M+M^{-1}\) to \(9M-M^{-1}\);
\(q\) increases from one to two. Indeed,
\[
 p'=\frac{C(1-\kappa s^2)-4\kappa P_0s}{D_s^2}<0,\qquad
 q'=\frac{1+3\kappa-4\kappa s-\kappa s^2-3\kappa^2s^2}{D_s^2}>0.
 \tag{11}
\]
For the latter, the numerator is at least
\(1-2\kappa-3\kappa^2>0\).

The [isolated acquisition estimate](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-same-source-acquisition)
gives
\(\|e^{-A\tau}P_9\|_{2I}\le e^{-\tau/5}\),
\(\|f\|_{2I}\le F=\sqrt{18}\), and \(\lambda_{\max}(L)\le4\).
Let \(\rho\ge e^{-40}\). A single bound valid over the whole path is
\[
 \begin{aligned}
 N&=\gamma m\sqrt{2(2P_0^2+240)},&
 E_x&=\sqrt{18}\gamma C_Q\epsilon,&
 E_\theta&=\sqrt{18}\epsilon,\\
 R_z&=\rho(N+E_x)+5\eta F,&
 R_x&=R_z/(\gamma\sqrt2),&
 E&=(E_x+E_\theta+R_z+\eta FT)/\sqrt2 .
 \end{aligned}\tag{12}
\]
The source norm uses
\(\|ph+qr\|_2^2=2p^2-2pq+60q^2\le2P_0^2+240\).
An implementation may replace this global upper bound by the corresponding
interval's endpoint bounds, retaining the same inequality.

Variation of constants for \(z=\gamma x\) and integration of both complete
rows give, for every admitted source,
\[
 \|x_s(T)\|_2\le R_x,\qquad
 \|\theta_s(T)-\gamma mQ_sa\|_2\le E,\qquad
 \theta_s(T)-\gamma mQ_sa
 =\zeta+\gamma Q_s\xi-z(T)+\eta\int_0^T f(\theta_s(v))\,dv.
 \tag{13}
\]
The large source lift is retained in (12); it is not removed from the
semigroup or uncertainty calculation. Outward scalar arithmetic gives
\[
 R_x<0.004667300447,\qquad E<0.000059542394,\qquad
 2R_x^2+6E<0.000400821752 .
 \tag{14}
\]
These decimal constants are exact rational upper bounds.

For circular phase and potential calculations alone write the proxy as
\[
 \gamma mQ_sa=2\pi Mh+t(s)r+\delta(s)h,\qquad
 t(s)=\gamma mq(s),\quad\delta(s)=\gamma mp(s)-2\pi M .
 \tag{15}
\]
No integer-turn shift is applied to the actual trajectory. Since
\(\nabla U(tr)\cdot h=0\), \(\nabla^2U\preceq L\) globally, and
\(\|\nabla U\|_2\le6\),
\[
 \begin{gathered}
 U_0(t)=U(tr)=9-8\cos t-\cos8t,\qquad
 U(tr+\delta h)\le U_0(t)+3\delta^2,\\
 H(x_s(T),\theta_s(T))
 \le\max_{t\ {\rm in\ the\ interval}}U_0(t)
       +3\delta_{\max}^2+6E+2R_x^2 .
 \end{gathered}\tag{16}
\]
This correlated estimate avoids replacing the nine cosine arguments by
unrelated extrema. The maximum uses the two endpoints and only the analytic
stationary points in the admitted \(t\)-range:
\[
 U_0'(t)=16\sin(9t/2)\cos(7t/2)=0.
 \tag{17}
\]
The possible interior candidates over this path are
\(2\pi/9,3\pi/7,4\pi/9\). Candidate inclusion is checked against the
interval; in particular the saddle value at \(3\pi/7\) is not charged to
an interval that excludes it. This is a finite extrema theorem, not a
grid or response search.

The oriented proxy gaps consist of five copies of \(t\), two copies of
\(t+\delta\), one \(t-2\delta\), and the closing gap \(-8t\).
For a winding-\(k\) chart replace the latter by \(-8t+2\pi k\).
Every actual gap differs by at most \(\sqrt2E\). Strict inclusion in
\((-\pi,\pi)\) certifies actual winding \(k\) at \(T\); inclusion in
\((-\pi/2,\pi/2)\) additionally certifies acute-cell entry.

<a id="sine-preparation-path-capture"></a>
## Positive capture corridors and their energy barriers

In the acute winding-\(k\) cell the nine principal gaps have sum \(2\pi k\).
At a boundary with one gap \(+\pi/2\), Jensen's inequality on the other
eight gives the phase-energy barrier
\[
 B_k=1+8\left[1-\cos\frac{2\pi k-\pi/2}{8}\right]
     =9-8\cos\frac{(4k-1)\pi}{16},\qquad k=1,2 .
 \tag{18}
\]
A gap \(-\pi/2\) gives a larger bound for \(k=1\); it is incompatible
with the remaining acute gaps' required sum for \(k=2\).
The nonnegative form energy cannot lower this boundary barrier.

If a complete endpoint is in the acute cell and has \(H(T)<B_k\),
storage nonincrease prevents leaving that cell. The fixed form mean and
the bounded phase-gap chart make the relevant storage sublevel compact.
Zero loss requires \(x=0\); invariance then requires \(f(\theta)=0\).
In an acute cell all equal sine currents have equal gaps, so the only
relative equilibrium is the regular winding-\(k\) target.
LaSalle therefore gives convergence to that target, with the common phase
origin determined by the retained lift mean and chart. It also gives
eventual entry into any positive target-neighborhood radius, without
supplying a uniform additional entry time.

The three parameter intervals below were declared before the scalar
assessment. Shared outward dyadic128/Machin arithmetic applied to
(10)--(18) yields the following conservative rational bounds.

| Source interval | Certified geometry at \(T=200\) | Complete \(H(T)\), upper | Relevant threshold, lower | Strict energy margin, lower |
| --- | --- | ---: | ---: | ---: |
| \(0\le s\le1/8\) | Acute, winding one | \(2.343558461\) | \(B_1>2.348243101\) | \(0.004684640\) |
| \(3/4\le s\le4/5\) | Winding two | \(7.337282704\) | \(H_2^*>7.437166400\) | \(0.099883696\) |
| \(63/64\le s\le1\) | Acute, winding two | \(7.438205301\) | \(B_2>7.439277423\) | \(0.001072122\) |

Here \(H_2^*=9(1-\cos(4\pi/9))\).
The actual acute-gap margins exceed \(0.783403918\) and \(0.087184187\)
in the first and third rows. The interior row's actual principal-gap
margin exceeds \(0.348983859\), so its winding assertion is separated
strictly from an antipodal ambiguity.
All margins apply uniformly to the complete correlated error family,
including the transported form errors. Equations (18) and (1) prove
permanent winding-one and winding-two capture in the first and third
corridors, respectively.

<a id="sine-preparation-transient-winding-obstruction"></a>
## Winding two at the transit time cannot persist in the interior corridor

The middle row has energy below every critical point in the closure of
the winding-two sector. To see this, a cycle critical point has a common
sine current on all oriented edges. For positive current, write every
non-antipodal gap as \(\delta\) or \(\pi-\delta\),
\(0<\delta<\pi/2\). If \(j\) gaps are obtuse, winding two requires
\[
 (9-2j)\delta+j\pi=4\pi,\qquad
 j\in\{0,1,2,3\},\qquad
 n=9-2j,\quad
 \delta=\pi/2-\pi/(2n).
 \tag{19}
\]
Its potential is
\[
 U_j=9-n\sin\frac{\pi}{2n}\ge
 9-9\sin\frac{\pi}{18}=H_2^* .
 \tag{20}
\]
Indeed \(n\sin(c/n)\) increases with \(n>0\) on this range, since
\(\sin z-z\cos z>0\) for \(0<z<\pi/2\).
Negative current gives negative principal gaps and cannot have winding
two. The limiting all-\(\pi/2\) case is incompatible with the C9 winding
sum. At an antipodal critical point the common current is zero; all gaps
are \(0\) or \(\pm\pi\). A limiting sum \(4\pi\) requires at least four
antipodal edges, so \(U\ge8>H_2^*\).

Now fix any source in \(s\in[3/4,4/5]\). Its actual winding at \(T\)
is two, while \(H(T)<H_2^*\). On the form-mean-zero leaf the storage
sublevel bounds \(x\); circular phases lie in a compact torus. LaSalle
implies that every omega-limit point has \(x=0\) and \(f(\theta)=0\).
By (20), no such point can lie in the closed winding-two sector.
Arbitrarily late visits to that closed sector would have a convergent
subsequence there and would create such an omega-limit point. Therefore
there is a finite later time after which the trajectory stays outside
the entire closed winding-two sector.

This is a phase-passage obstruction for actual complete trajectories,
not merely a proxy observation. It concerns the sector itself, not one
chosen lift cell: simultaneous cancelling branch-cut crossings need not
change winding. A phase passage must occur after \(T\), but the certificate
does not determine its time, the last passage, or the eventual class.

<a id="sine-preparation-path-scope"></a>
## Information consequence and implementation boundary

The source parameter and the complete source formulas give sufficient
local selectors on two nontrivial intervals. In contrast, the claim that
the actual finite-transit winding, or its correctly certified proxy
winding, always specifies the acquired persistent class is false on the
interior interval. Equal initial phase, means and storage still hold
throughout this comparison. The fine form information in (7) is retained;
it is not an inferred pressure or a controller.

The intervals \((1/8,3/4)\) and \((4/5,63/64)\) remain undecided by these
bounds. Their endpoints are certificate boundaries, not located basin
boundaries. Nor does the interior result show convergence to winding one.
The two positive corridors establish acute-cell capture at the fixed
time, not the earlier small-ball radius condition at that time.
The complete law permits antipodal passage during formation; no acute
constraint is imposed on the whole transit.

The result is conditional on the supplied isolated support, complete
sine law, clock and correlated preparation. It derives neither autonomous
preparation nor a uniquely selected constitutive law, substrate, size
or physical identification. No old acquired-source certificate is
transferred through the new map without re-admitting its errors.

The [terminal-limit theorem](SINE_PREPARATION_PATH_LIMITS.md) continues these
same histories. It proves complete-state convergence and narrows the interior
corridor's energy-compatible critical configurations, while retaining saddle
exceptions. The additional [source-specific capture proof](SINE_PREPARATION_PATH_CAPTURE.md)
uses the actual mixed state to select winding one for this corridor and
exclude those exceptions; the other path intervals retain their own scope.

The private [path calculator](../../src/tnfr/physics/_sine_preparation_path_selection.py)
admits a closed parameter interval, structural time, and separate latent form
and phase error budgets. It rebuilds the exact generator and source identities,
uses the shared semigroup radius and acute-sector boundary owners, and retains
all phase-gap, storage and source-policy margins. Its source-budget and
initial-winding flags are separate from the sufficient capture or exit
classification. A supplied interval with unresolved bounds is not classified
as a failed formation experiment.

The [independent controls](../../tests/physics/test_sine_preparation_path_selection.py)
check the exact Cayley isometry, correlated error transport, canonical cycle
offsets, analytic potential extrema, all-face energy bounds and the critical
energy comparison. The three fixed corridor assessments evaluate scalar
inequalities only. The code supplies neither a trajectory nor a crossing
history, and it does not consume an earlier acquisition verdict.
