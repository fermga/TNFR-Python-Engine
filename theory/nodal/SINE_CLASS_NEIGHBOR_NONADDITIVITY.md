# Nonadditive causal input from distinct neighboring patterns

<a id="sine-class-neighbor-nonadditivity"></a>

The [composition theorem](SINE_CLASS_INTERFACE_COMPOSITION.md) admits
endogenous form and phase inputs between the three acquired C9 patterns.
It does not assert that the mediator's response to two neighbors is the
sum of its separate single-neighbor responses. This owner proves a finite
failure of that additive description, with all four histories starting
from the same complete acquired state.

The construction uses the existing law, support and structural clock.
It evaluates static graph matrices and rational inequalities, not a
selected trajectory or finite-time coefficient. Unlike a comparison with
a linearized single-input law, the additive benchmark below retains each
neighbor's complete nonlinear single-input response.

<a id="sine-neighbor-model-and-observer"></a>
## Complete histories and the matched additive benchmark

Keep the 27 nodes, 29 edges, unit capacities, bridges \((4,13),(13,22)\),
and classes \((1,k,1)\), \(k=1,2\). The complete state consists of
27 forms followed by 27 continuous phase lifts. Let \(D\) be the actual
degree matrix, with central degrees \((3,4,3)\), and \(A=D^{-1}L\).
Use exactly
\[
 x'=-Ax+\gamma D^{-1}S(\theta),\qquad
 \theta'=\gamma Ax,\qquad
 \gamma=\frac1{1023\pi},\qquad \tau=\frac{1023}{1024}t .
 \tag{1}
\]
The target \(\Theta_k\) has aligned central origins and the stated
ring windings. The
[original acquired-family association](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source)
supplies the initial state: each component's form and phase errors
have Euclidean norm at most \(\epsilon=10^{-32}\), with its original
zero-sum and preparation constraints. The resulting maximum-coordinate
cover is sufficient for the bounds below; independent preparation of
its Cartesian corners is not assumed.

Choose one such complete state, including all hidden coordinates,
and represent it before the input by
\(z_{\mathrm{res},0}=(x_0,\theta_0-\Theta_k)\) about the original
declared common origins. Below \(z_0\) denotes this complete residual
vector; \(z_0=0\) means the nominal wound target, not zero absolute
phases. Use the same source in four counterfactual histories. At time zero, apply the
form jumps
\[
 f_{00}=0,\quad f_{10}=ae_4,\quad f_{01}=be_{22},\quad
 f_{11}=ae_4+be_{22}.
 \tag{2}
\]
Phases do not jump. All histories then evolve under (1), without resets.
Let \(Y_{ij}=x_{13}^{ij}(T)\). The mixed response is
\[
 M_k(a,b,T;z_0)=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
 \tag{3}
\]
The matched additive port functional predicts
\[
 Y_{\rm add}=Y_{00}+(Y_{10}-Y_{00})+(Y_{01}-Y_{00})
            =Y_{10}+Y_{01}-Y_{00}.
 \tag{4}
\]
Thus \(M_k=Y_{11}-Y_{\rm add}\). Both single-neighbor functionals
in (4) use the same complete source and the same full nonlinear law.
This is a benchmark of additive causal input, not a newly supplied
autonomous 54-coordinate comparator or a new energy law.

The general analytic hierarchy allows signed \(a,b\) with
\(|a|+|b|\le7/5000\) and \(T\le2\). The finite discriminating certificate
below selects, before any response or time-coefficient evaluation,
\[
 a=b=m=\frac7{10000},\qquad T=\frac18,\qquad
 \epsilon=10^{-32},\qquad \delta=10^{-30}.
 \tag{5}
\]
The theorem applies to each class separately. No cross-class subtraction is part
of this observer.

The allowance \(\delta=10^{-30}\) is inherited from the earlier
[nonlinear-response observation contract](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-source-and-observer).
It differs from the \(10^{-8}\) allowance of the intervening changed-input
memory comparison. Here it is a supplied mathematical observation allowance,
not an established laboratory precision. The distinct-neighbor intervention
and its mixed observable are new; this is not a finer-precision repetition
of the memory comparison.

<a id="sine-neighbor-mixed-hierarchy"></a>
## The mixed causal hierarchy and its symmetries

Let \(J_k\) denote the full tangent generator and \(Q_k,T_k\) the
normalized phase maps of the
[complete amplitude hierarchy](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-triangular-variation).
Define their full-state lifts by
\[
 \mathbb Q_k(z,\widetilde z)
      =(\gamma Q_k(y,\widetilde y),0),\qquad
 \mathbb T_k(z,\widetilde z,\widehat z)
      =(\gamma T_k(y,\widetilde y,\widehat y),0).
\]
The phase maps include their quadratic and cubic factorials; the
displayed lift includes the separate \(\gamma\). Let
\(\mathcal V f(t)=\int_0^t e^{J_k(t-s)}f(s)\,ds\).
Write \(u,v\) for the donor and receiver first-level histories,
including their amplitudes. Define
\[
 q_{DD}=\mathcal V\mathbb Q_k(u,u),\quad
 q_{DR}=\mathcal V\,2\mathbb Q_k(u,v),\quad
 q_{RR}=\mathcal V\mathbb Q_k(v,v).
 \tag{6}
\]
The complete mixed cubic history is \(\mathcal V F_{\rm mix}\), where
\[
 \begin{split}
 F_{\rm mix}={}&2\mathbb Q_k(u,q_{DR}+q_{RR})
              +2\mathbb Q_k(v,q_{DD}+q_{DR})\\
              &+3\mathbb T_k(u,u,v)+3\mathbb T_k(u,v,v).
 \end{split}
 \tag{7}
\]
This is an operator on two distinct causal input histories. It includes
quadratic internal excitation followed by quadratic recoupling; retaining
only the direct cubic term would change it.

Simultaneous reflection of the local ring indices fixes all three
central ports. For the zero-residual nominal source, the central output
is odd under \((a,b)\mapsto(-a,-b)\); even total amplitude degrees vanish.
The quadratic history is generally nonzero in the hidden odd sector.
It cannot be deleted from (6)-(7).

There are exact null controls. A zero amplitude makes (3) zero for
every actual source by cancellation of identical histories. At the
nominal source, \(a=-b\) also gives zero for all times: exchanging the
identical outer rings changes the input to its negative while leaving
the mediator observation fixed, and the reflection/sign symmetry
changes that observation's sign. Arbitrary acquired residuals need
not have either symmetry. The nominal opposite-input null is not
asserted for every actual source.

<a id="sine-neighbor-local-onset"></a>
## A nonzero local interaction operator

Set \(\alpha=2\pi/9\), \(c_k=\cos(k\alpha)\). At the ideal
postjump state, the phase slopes are
\[
 \theta'_4=\gamma a,\qquad
 \theta'_{22}=\gamma b,\qquad
 \theta'_{13}=-\gamma(a+b)/4,\qquad
 \theta'_{12}=\theta'_{14}=0.
 \tag{8}
\]
The four initial phase-gap slopes into the mediator are therefore
\[
 \frac\gamma4(5a+b),\quad \frac\gamma4(a+5b),\quad
 \frac\gamma4(a+b),\quad \frac\gamma4(a+b).
 \tag{9}
\]
The last two are internal mediator edges and carry cosine \(c_k\).
Omitting them loses organization dependence at the first mixed order.

The cubic part of the fourth time derivative comes entirely from
the third derivative of the sine applied to the three phase slopes.
The remaining differentiated terms have amplitude degree at most two
at time zero. Their central mixed quadratic part vanishes by reflection.
Equivalently, the quadratic phase history starts at time order four,
so its recoupling cannot contribute to the order-four form term.
Subtracting the two singles in the four incident cubes gives
\(6(15+c_k)ab(a+b)\), with the common denominator \(4^3\).
Including the degree-four normalization, sine factorial and time
integration yields the full nonlinear onset
\[
 M_k(a,b,T;0)=
 -\frac{15+c_k}{1024}\gamma^4ab(a+b)T^4+O(T^5),
 \qquad T\downarrow0 .
 \tag{10}
\]
The amplitudes are held fixed in this statement. Equation (10) is
not a finite-window certificate by itself. It already shows that
distinct-neighbor additivity is not an exact identity and that this
local interaction coefficient depends on the mediator organization.
It does not establish a resolved finite cross-class contrast.

<a id="sine-neighbor-static-cones"></a>
## Static edge cones for the complete linear phase history

The following finite estimate avoids evaluating a time series at \(T\).
Put \(P(t)=e^{-At}\), \(\eta=\gamma^2\),
\(g=1/3000\), and \(D_T=1-2g^2T^2\).
For one unit donor or receiver form impulse, write the complete tangent
phase as \(\gamma w_p(t)\), \(p=4,22\). The corresponding heat phase
is \(w_{h,p}=(I-P(t))e_p\).

For this graph \(\|Ae_p\|_\infty=1\). Markov contraction gives
\(\|w_{h,p}(t)\|_\infty\le t\), and the full tangent feedback gives
\[
 \|w_p(t)\|_\infty\le t/D_T,\qquad
 \|w_p(t)-w_{h,p}(t)\|_\infty
 \le4g^2\int_0^t(t-s)\frac{s}{D_T}\,ds
 =\frac{2g^2t^3}{3D_T}.
 \tag{11}
\]
This includes the phase Hessian on every internal and bridge edge;
the phase is not replaced by its heat approximation.

Since \(\|A\|_\infty\le2\), commutation and contraction imply
\(\|A^3P(s)e_p\|_\infty\le4\). Taylor's integral remainder gives
\[
 \left\|\frac{w_{h,p}(t)}t-Ae_p+\frac t2 A^2e_p\right\|_\infty
 \le\frac23t^2,\qquad t>0.
 \tag{12}
\]
For an oriented edge \(i\to j\), set
\[
 s_{p,ij}=(Ae_p)_j-(Ae_p)_i,\qquad
 b_{p,ij}=(A^2e_p)_j-(A^2e_p)_i,\qquad
 E_T=\frac43T^2\left(1+\frac{g^2}{D_T}\right).
 \tag{13}
\]
Equations (11)-(12) enclose the entire edge history, for \(0<t\le T\),
by the static interval
\[
 \frac{w_{p,j}(t)-w_{p,i}(t)}t\in
 I_{p,ij}:=
 \left[
 s_{p,ij}+\min(0,-Tb_{p,ij}/2)-E_T,
 s_{p,ij}+\max(0,-Tb_{p,ij}/2)+E_T
 \right].
 \tag{14}
\]
Every matrix in (13)-(14) is the exact rational normalized graph
matrix. These are derivative/contraction bounds over a window, not
evaluated phase responses.

The internal target cosines lie in \([1/6,1]\), while bridge cosines
equal one. For example
\(\cos(4\pi/9)=\sin(\pi/18)>1/6\) follows from
\(25/8<\pi<22/7\) and \(\sin x\ge x-x^3/6\) on the relevant
positive interval; \(\cos(2\pi/9)\) is larger. The same elementary
bounds give \(1/3300<\gamma<g\).

Build 27 interval force rows, initially zero. For each of the 29
oriented edges put \(U=I_{4,ij}\), \(V=I_{22,ij}\), and let \(C_{ij}\)
be its cosine interval. Add
\[
 -\frac{C_{ij}UV(U+V)}{2d_i}\quad\hbox{to row }i,\qquad
 +\frac{C_{ij}UV(U+V)}{2d_j}\quad\hbox{to row }j.
 \tag{15}
\]
Interval products retain all endpoint choices; shared cosine or
phase dependencies are relaxed outward. If \(F_i(t)\) is the actual
direct mixed cubic form force divided by \(\gamma^4m^3t^3\),
(15) encloses it.

At \(T=1/8\), exact rational evaluation of (13)-(15) gives the
following deliberately coarse static certificate:

| Force rows | Enclosure |
| --- | --- |
| Mediator center 13 | \(-3/20<F_{13}<-1/15\) |
| Every other node | \(-3/20<F_i<1/9\) |

For orientation, the outward force calculation gives
\(-\sup F_{13}>0.0701642\),
\(\max(0,\max_{i\ne13}\sup F_i)<0.104719\), and
\(\max_i\sup|F_i|<0.142588\).
Only the rational bounds in the table are needed below. All 29
edges are included; no favorable subset is selected. The cones shrink
when \(T\) decreases, so the same coarse certificate holds for
\(0<T\le1/8\). This finite static calculation generates no trajectory
or selected finite-time coefficient.

<a id="sine-neighbor-finite-cubic-bound"></a>
## Heat transport, hidden recoupling and complete cubic error

Let \(Z_M(T)\) be the mediator output obtained by heat-propagating
only the direct cubic force of the **complete** first-level phase.
For \(s\ge0\), \(P(s)\) is nonnegative with row sum one and
\[
                  P_{13,13}(s)\ge e^{-s}\ge1-s.
 \tag{16}
\]
The last bound follows also from uniformization \(A=I-\mathsf P\)
with nonnegative stochastic \(\mathsf P\). Set \(l=1/15\), \(B=1/9\).
At an integration time \(t\), the force table implies
\[
 (P(T-t)F(t))_{13}
 \le -l+(l+B)(T-t).
 \tag{17}
\]
Integrating \(t^3\) and using \(\gamma>1/3300\) gives, at (5),
\[
 Z_M(T)\le
 -\gamma^4m^3T^4
       \left(\frac l4-\frac{(l+B)T}{20}\right)
 \le-\frac7{450}\frac{m^3T^4}{3300^4}.
 \tag{18}
\]
Also \(|Z_M(T)|\le(3/80)g^4m^3T^4\).
Heat spreading is included in (17); the sign of the local forcing
alone would not justify an endpoint sign.

There remain the quadratic recoupling and the phase feedback in the
third-level propagator. They are not discarded. For each of the three
nonzero nominal histories in (2), its initial form \(f\) satisfies
\[
                    \|f\|_\infty=m,\qquad \|Af\|_\infty\le m.
 \tag{19}
\]
In particular the simultaneous history has maximum form \(m\), not
\(2m\). Its total input variation is nevertheless \(2m\), as used
later for amplitude and source remainders.

Use the exact scaled levels
\[
 y_1=\gamma w_1,\quad x_2=\gamma^3u_2,\quad
 y_2=\gamma^4w_2,\quad x_3=\gamma^4u_3,\quad
 y_3=\gamma^5w_3.
 \tag{20}
\]
Their linear blocks retain \(\eta=\gamma^2\). The scaled third
form forcing is \(T(w_1,w_1,w_1)+2\eta Q(w_1,w_2)\), with
\(\|Q(v,w)\|_\infty\le2\|v\|_\infty\|w\|_\infty\) and
\(\|T(u,v,w)\|_\infty\le(4/3)\|u\|_\infty\|v\|_\infty\|w\|_\infty\).
Time integration and heat contraction give
\[
 \begin{split}
 \|u_2(t)\|_\infty&\le\frac{2m^2t^3}{3D_T^3},&
 \|w_2(t)\|_\infty&\le\frac{m^2t^4}{3D_T^3},\\
 \|u_3(t)\|_\infty&\le
       \frac{m^3t^4}{3D_T^4}+\frac{2g^2m^3t^6}{9D_T^5},&
 \|w_3(t)\|_\infty&\le
       \frac{2m^3t^5}{15D_T^4}+\frac{4g^2m^3t^7}{63D_T^5}.
 \end{split}
 \tag{21}
\]
The denominator follows from the same two-row phase-feedback estimate
as (11). The direct quadratic recoupling integral costs at most
\(2g^6m^3T^6/(9D_T^4)\) in the original form. Integrating the
third-level phase feedback costs at most
\(2g^6m^3T^6/(45D_T^4)+g^8m^3T^8/(63D_T^5)\).
Thus complete cubic mixed response differs from \(Z_M\) by at most
\[
 E_{\rm dyn}=
 3m^3\left[
       \frac{4g^6T^6}{15D_T^4}
                  +\frac{g^8T^8}{63D_T^5}\right].
 \tag{22}
\]
The factor three counts the three nonzero histories, each satisfying
(19); it is not a cancellation of ten total-variation cubes. This
estimate includes every internal quadratic mode and both phase rows.

<a id="sine-neighbor-source-and-finite-sign"></a>
## Common-source cancellation and the finite sign

For the nominal central output, use the existing reflection remainder
\[
 R_5(A,T)=\frac{256g^6A^5T}
               {D_T(1-4g^2A^2)}.
 \tag{23}
\]
For arbitrary actual residuals, retain the full common linear source
\(\exp(J_kT)z_{\mathrm{res},0}\) and the separately bounded nonlinear initialization
defect
\[
 E_{\rm init}(A,T,\epsilon)=
 \frac{4gT\epsilon}{\ell_TD_T}
       \left(\frac{gA}{D_T}+\frac{\epsilon}{\ell_T}\right),
 \qquad \ell_T=1-2gT.
 \tag{24}
\]
Both come from the
[collective-interface fidelity theorem](SINE_CLASS_COLLECTIVE_INTERFACE.md#sine-collective-interface-fidelity).
The linear source cancels exactly in (3), because the same complete
state initializes all four histories. No parity is assumed for that
state. The nonlinear source contribution does not cancel by assertion.

With indexed histories \(h\in\{00,10,01,11\}\) and
\((A_{00},A_{10},A_{01},A_{11})=(0,m,m,2m)\), put
\[
 E=E_{\rm dyn}
       +\sum_h R_5(A_h,T)
       +\sum_h E_{\rm init}(A_h,T,\epsilon).
 \tag{25}
\]
The zero-input initialization term is included. Replacing the shared
source by independently chosen residuals for the four histories
would invalidate the linear cancellation and require a different
source allowance.

Combining (18), (22) and (25) bounds the exact full nonlinear statistic:
\[
 -\frac3{80}g^4m^3T^4-E
 \le M_k(a,b,T;z_0)
 \le-\frac7{450}\frac{m^3T^4}{3300^4}+E.
 \tag{26}
\]
At the fixed values (5), exact rational arithmetic gives
\[
 \begin{split}
 \frac7{450}\frac{m^3T^4}{3300^4}&>1.0984\,10^{-29},\\
 E_{\rm dyn}&<1.436\,10^{-36},\\
 \sum_hR_5(A_h,T)&<2.509\,10^{-34},\\
 \sum_hE_{\rm init}(A_h,T,\epsilon)&<1.556\,10^{-42},
 \qquad E<3\,10^{-34}.
 \end{split}
 \tag{27}
\]
In particular, for every common actual source in the admitted family
and for both mediator classes separately,
\[
              -3.9\,10^{-29}<M_k<-1.0983\,10^{-29}.
 \tag{28}
\]
This is a finite sign theorem, not merely the asymptotic coefficient
(10). No time-coefficient or full-law response evaluation is used
in (26)-(28).

<a id="sine-neighbor-observation-resolution"></a>
## Four-reading and independent-record distinctions

Four endpoint readings with individual errors bounded by \(\delta\)
give
\[
                 |\widehat M_k-M_k|\le4\delta.
 \tag{29}
\]
If (4) is constructed from the same three baseline/single-neighbor
readings, comparison with the joint reading is exactly (29). It is
not an additional independently noisy observation.

A separately recorded additive four-history functional has true mixed
statistic zero and recorded mixed statistic in \([-4\delta,4\delta]\).
Consequently the sufficient separation from its independent recorded
set is \(M_k<-8\delta\), not merely \(M_k<-4\delta\). Equation (28)
retains the inherited \(\delta=10^{-30}\) and proves
\[
                 -\sup M_k-8\delta>2.98\,10^{-30}.
 \tag{30}
\]
Thus both the direct four-reading discrepancy and the stronger
independent-additive record-set distinction are resolved by the
analytic certificate. The latter is separation from the named
additive functional, not from every possible effective model.

There is no evaluated numerical response or numerical interval radius
in this theorem. A future independent observation producer must retain
its own source association, numerical enclosure and frozen resolution
policy within a justified part of the positive margin (30). Neither
(28) nor the static cone table may be used to narrow that enclosure.

<a id="sine-neighbor-work-and-identity"></a>
## Work, means and identity of all four histories

Every branch starts immediately after the same admitted joining event.
The existing source proof gives contact work at most \(8\epsilon^2\)
and preprobe excess storage at most \(22\epsilon^2\). The donor and
receiver are not adjacent, so \(L_{4,22}=0\). For simultaneous impulses,
\[
 W(a,b;x^-)
 =a(Lx^-)_4+b(Lx^- )_{22}+\tfrac32(a^2+b^2),\qquad
 |W-\tfrac32(a^2+b^2)|\le6\epsilon(|a|+|b|).
 \tag{31}
\]
Use the corresponding subset of impulses in each branch. At their
identical pre-event state,
\(W_{11}-W_{10}-W_{01}+W_{00}=0\) exactly. The later nonadditivity
therefore is not an inserted mixed work term at the event.

For (5), the largest probe work is
\(3m^2+12m\epsilon<1.471\,10^{-6}<2\,10^{-6}\).
The full postjump excess is below
\[
 22\epsilon^2+3m^2+12m\epsilon
                         <1.471\,10^{-6}<1/388800.
 \tag{32}
\]
A sufficient Euclidean quotient-radius square is
\[
               6\epsilon^2+4m\epsilon+2m^2<1/144.
 \tag{33}
\]
The same bounds dominate the single-input and baseline branches.
Together with the original joined spectral and cosine premises,
they supply the existing radius-\(1/12\) trapping and identity
certificate. Continuous loss is not used as an event-work reserve.

The weighted form mean in branch \(ij\) shifts by
\(3(ia+jb)/58\), while the weighted phase mean is unchanged.
Their four-history mixed common means cancel exactly. All residual
coordinates and subsequent means are carried; the histories are not
reset to their isolated component origins.

The new property is a conditional, organization-dependent nonlinear
response to the simultaneous activity of distinct neighbors. Each
single-neighbor functional can be correct while their additive
combination fails. The finite certificate does not establish an
independently resolved difference between classes, autonomous selection
of the two impulses, a laboratory sensor capability or a uniquely
fundamental interaction law.

## Implementation and verification boundary

The [private analytic owner](../../src/tnfr/physics/_sine_class_neighbor_nonadditivity.py)
rebuilds the local onset from exact edge factors, retains every static
phase cone and force row, and combines the complete cubic, amplitude,
source and observation allowances. Its finite cone certificate is
restricted to equal positive impulses with \(0<T\le1/8\).
Other admitted signed inputs retain the formal onset and conditional
source/work bounds; absent a separate finite remainder they do not
receive a finite sign verdict.

A zero amplitude or zero horizon gives exact mixed zero for every
actual source, without charging an artificial approximation defect.
The opposite-input nominal symmetry has its narrower scope stated
above; it does not override arbitrary actual-source uncertainty.
The shared simultaneous-event ledger uses the unchanged receiver
pressure after the donor impulse, because \(L_{4,22}=0\); it does
not reuse the delayed same-port pressure bound.

The [independent algebra and bound controls](../../tests/physics/test_sine_class_neighbor_nonadditivity.py)
check all-edge onset channels, static cones, phase feedback and
recoupling allowances, shared-source cancellation, observation-error
thresholds and four-branch event accounting. Evaluating these
structural and rational certificates is not a selected finite-time
coefficient calculation. No new response, source acquisition or
archived scientific producer is executed by this admission.
