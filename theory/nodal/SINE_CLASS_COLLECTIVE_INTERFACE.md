# A nonlinear collective interface for acquired sine classes

<a id="sine-class-collective-interface"></a>

The [repeated interaction theorem](SINE_CLASS_REPEATED_INTERACTION.md)
retains a class-dependent nonlinear response under complete carried
histories. This owner derives an interface through the three central
ports. Its retained observations have six coordinates; the remaining
48 coordinates enter causal kernels and initialization terms. The
interface is a controlled nonlinear history functional, not a closed
six-dimensional Markov law.

The earlier [mediator memory](SINE_CLASS_MEDIATED_MEMORY.md) retained
36 donor/receiver coordinates and an exact tangent memory term. Here
the quadratic hidden correction is eliminated explicitly into a
cubic port functional. It cannot be discarded merely because its
direct port output vanishes. The proof reuses the complete amplitude
hierarchy and its analytic remainder, without evaluating another
coefficient, source or nonlinear trajectory.

<a id="sine-collective-interface-domain"></a>
## Complete model, input family and retained observations

Keep the 27 nodes, three C9 components, contacts \((4,13),(13,22)\),
unit capacities, actual degrees and class tuples \((1,k,1)\),
\(k=1,2\). In the original continuous phase lifts,
\[
 x'=-Ax+\gamma D^{-1}S(\theta),\qquad \theta'=\gamma Ax,
 \quad A=D^{-1}L,\quad \gamma=1/(1023\pi),\quad
 \tau=(1023/1024)t.
 \tag{1}
\]
The nominal target \(\Theta_k\) has aligned central origins and
each component's stated winding. Let \(y=\theta-\Theta_k\) and
\(z=(x,y)\), retaining all 54 original coordinates in the law.

Let \(B_p=(e_4,e_{13},e_{22})\). A finite signed central-form
impulse measure consists of fixed times \(t_j\in[0,H]\) and
vectors \(q_j\in\mathbb R^3\), with jumps
\[
 x(t_j^+)=x(t_j^-)+B_pq_j,\qquad y(t_j^+)=y(t_j^-),
 \qquad A_*:=\sum_j\|q_j\|_1\le\frac7{5000},\quad 0\le H\le2.
 \tag{2}
\]
Simultaneous impulses are combined into one vector. Values at an
observation/event endpoint are right-continuous unless a preevent
value is explicitly requested. No future input is inferred from
an evaluated response. This input family includes the four supplied
donor words with impulses \(7/10000\) at zero and one.

The visible vector is
\[
 v=(x_4,x_{13},x_{22},y_4,y_{13},y_{22})=\Pi z.
 \tag{3}
\]
All other forms and phases are hidden in \(h\in\mathbb R^{48}\).
The hidden coordinates include both reflection parities. They are
not replaced by component means or stationary values. There is no
claim that these observations are a minimal sufficient interface.

Approximation bounds below hold over each admitted finite window.
The large return dwell and indefinitely many accumulated impulses do
not lie in the domain \(H\le2\) of one cubic approximation. For
repetition, the full-law return supplies the next actual source;
the next description retains that source and its accumulated means.
Changing a description's time origin is not resetting the state.

<a id="sine-collective-interface-kernels"></a>
## Six-port causal equations with hidden quadratic recoupling

Use the [complete original-coordinate maps](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-triangular-variation)
\[
 J_k=\begin{pmatrix}-A&-\gamma C_k\\ \gamma A&0\end{pmatrix},
 \quad \mathbb Q_k(z,w)=(\gamma Q_k(y_z,y_w),0),
 \quad \mathbb T_k(z,w,u)=(\gamma T_k(y_z,y_w,y_u),0),
 \tag{4}
\]
where \(C_k\) is the degree-normalized cosine Laplacian and
\[
 Q_k(a,b)_i=-\frac1{2d_i}\sum_{j\sim i}
       \sin(\Theta_{k,j}-\Theta_{k,i})\,(a_j-a_i)(b_j-b_i),
\]
\[
 T_k(a,b,c)_i=-\frac1{6d_i}\sum_{j\sim i}
       \cos(\Theta_{k,j}-\Theta_{k,i})\,(a_j-a_i)(b_j-b_i)(c_j-c_i).
\]
The normalization includes the Taylor factorials. Both evolution
rows use the same original clock. No phase residual is divided by
\(\gamma\).

Partition the original generator according to (3):
\[
 J_k=\begin{pmatrix}E_k&B_k\\ C_k^{hv}&M_k\end{pmatrix},
 \qquad G_k(t)=\exp(M_kt),\qquad
 \mathcal K_k(t)=B_kG_k(t)C_k^{hv}.
 \tag{5}
\]
Here \(C_k^{hv}\) is a generator block, not the phase Laplacian
\(C_k\). The hidden propagator has an explicit finite modal form.
In each ring order its eight hidden nodes as
\((5,6,7,8,0,1,2,3)\). They form a grounded path with normalized
Laplacian diagonal one and adjacent entries \(-1/2\); every hidden
degree is two. For that ring's cosine \(c\), set
\[
 T_c=\begin{pmatrix}1&\gamma c\\-\gamma&0\end{pmatrix},
 \qquad \lambda_m=1-\cos(m\pi/9),\qquad m=1,\ldots,8.
\]
The normalized spatial sine vectors have entries
\(\sqrt{2/9}\sin(jm\pi/9)\). Each modal form/phase pair therefore
has generator \(-\lambda_mT_c\). In an interleaved form/phase
block notation, the hidden propagator entries are
\[
 G^{(c)}_{jr}(t)=\frac29\sum_{m=1}^8
 \sin(jm\pi/9)\sin(rm\pi/9)\exp(-t\lambda_mT_c).
 \tag{5a}
\]
The full hidden propagator consists of the three such ring blocks.
This diagonalizes only their spatial part; both complete dynamic
coordinates remain in every \(2\times2\) exponential.

The port meets both grounded-path endpoints. With actual port
degree \(d\in\{3,4\}\), the visible-to-hidden and hidden-to-visible
factors are \(T_c/2\) and \(T_c/d\), respectively. The resulting
self-memory block is explicitly
\[
 \mathcal K_{c,d}(t)=\frac4{9d}
   \sum_{\substack{1\le m\le8\\m\ {\rm odd}}}
   \sin^2(m\pi/9)\,T_c\exp(-t\lambda_mT_c)T_c.
 \tag{5b}
\]
Thus \(\mathcal K_{c,d}(0)=T_c^2/d\). The three self-memory
blocks are diagonal by component; instantaneous bridge coupling
remains in \(E_k\). Reflection acts on mode \(m\) with sign
\((-1)^{m+1}\). Four spatially odd modes disappear from the
linear port kernel (5b), but all eight modes in (5a) remain
available to the nonlinear forcing and initial-state source.

For a port history \(w\) and full forcing \(f\), define
\[
 \begin{split}
 \mathcal H_k[w](t)&=\int_0^tG_k(t-s)C_k^{hv}w(s)\,ds,\\
 \mathcal G_k[f](t)&=\int_0^tG_k(t-s)f_H(s)\,ds,\\
 \mathcal F_k[f](t)&=f_V(t)+B_k\mathcal G_k[f](t).
 \end{split}
 \tag{6}
\]
These fixed kernels derive from the entire hidden block; they are
not fitted from the response. Time-zero initialization terms are
kept separately below.

For the zero-residual nominal reference, scale the whole measure
in (2) by an auxiliary amplitude parameter. Its first level obeys
the six-port Volterra equation
\[
 dv_1(t)=\left[E_kv_1(t)+
        \int_0^t\mathcal K_k(t-s)v_1(s)\,ds\right]dt
           +\begin{pmatrix}I_3\\0\end{pmatrix}d\mu(t),
 \quad v_1(0^-)=0,\qquad h_1=\mathcal H_k[v_1].
 \tag{7}
\]
Set \(z_1=(v_1,h_1)\), in the partitioned order. The second
forcing is \(f_2=\mathbb Q_k(z_1,z_1)\).

Let \(\mathcal R\) reflect each local index \(j\mapsto8-j\)
in both channels. It fixes every central port and commutes with
\(J_k\). Central form inputs are reflection even; the nominal
amplitude hierarchy has
\(\mathcal Rz_n=(-1)^{n+1}z_n\). In particular,
\[
 v_2=0,\qquad h_2=\mathcal G_k[f_2],\qquad z_2=(0,h_2).
 \tag{8}
\]
To verify (8), \(f_2\) is reflection odd, its visible part vanishes,
and \(B_kG_k(t)(f_2)_H=0\). The homogeneous second port equation
has the unique zero solution. The hidden response itself need not
vanish and is a known causal quadratic functional of \(v_1\).

The complete cubic port equation is
\[
 \begin{split}
 f_3&=2\mathbb Q_k(z_1,z_2)+\mathbb T_k(z_1,z_1,z_1),\\
 v_3'&=E_kv_3+\mathcal K_k*v_3+\mathcal F_k[f_3],
       \qquad v_3(0^-)=0,\\
 h_3&=\mathcal H_k[v_3]+\mathcal G_k[f_3].
 \end{split}
 \tag{9}
\]
Every coefficient carries through every event. Only the first form
level jumps. Equations (7)-(9) require two six-coordinate port
histories. The hidden first and second levels are explicit integral
functionals, not another 48-coordinate unknown evolution. The
reconstructed third hidden level is needed only when a hidden-state
or boundary-pressure observation is requested.

This is a nonlinear input/output reduction with memory. Substituting
(6) successively expresses the cubic drive as nested causal integrals
of the input measure, retaining both the direct trilinear term and
the quadratic-then-quadratic term. The latter cannot be erased by
the identity \(v_2=0\). For example, at mediator center 13, take
the reflection-even phase vector \(e_{13}\) and the odd vector
\(e_{14}-e_{12}\). Then \(Q_k\) at that port is
\(\sin(k\alpha)/4\ne0\),
\(\alpha=2\pi/9\). Odd internal information can therefore feed
an even cubic port output. A six-state instantaneous law omitting
that functional has not been derived.

<a id="sine-collective-interface-initialization"></a>
## Complete initialization without a reset

Let \(z_0=(v_0,h_0)\) be the complete actual pre-input deviation
\(z(0^-)\), relative to its declared reference origins. A time-zero
impulse belongs to (7), not to this source again. The exact
linear-source port response \(v_s\) obeys
\[
 v_s'=E_kv_s+B_kG_k(t)h_0+\mathcal K_k*v_s,
 \quad v_s(0)=v_0,
 \qquad h_s=G_k(t)h_0+\mathcal H_k[v_s].
 \tag{10}
\]
It is precisely the port projection of \(\exp(J_kt)z_0\).
Its argument is the complete hidden initial vector \(h_0\), rather
than a prescribed equilibrium or component mean. Some components,
including reflection-odd hidden data, cancel in this linear port
projection. They can nevertheless affect the nonlinear response;
their full admitted uncertainty remains in the initialization defect
below. Class and present port values do not determine the source.
A known source can be retained explicitly, while an unknown source
must retain its admitted uncertainty rather than being assigned zero.

Choose reference origins together with the source bound, not by an
unproved change of gauge. In the first acquired word, use the
original zero common origins: its raw coordinate errors are bounded
by \(\epsilon\), and (10) retains their small possibly nonzero
joined means. Degree-centering that initial family would instead
give the weaker coordinate bound \(31\epsilon/29\).

For later words, the proved return family bounds each small
degree-centered representative, while its accumulated common form
and phase means are retained as exact reference origins. Equation
(10) consumes that representative, and the common output constants
are added back exactly. Those absolute means are not assumed to be
\(\epsilon\)-small. This is the same mean bookkeeping as in the
[repeated interaction](SINE_CLASS_REPEATED_INTERACTION.md#sine-repeated-complete-state).

The linear source need not be mean zero if other admitted origins
were used. The collective approximation is
\[
 v_{\rm int}=v_{\rm common}+v_s+v_1+v_3.
 \tag{11}
\]
Its source dependence includes a rigorous nonlinear initialization
defect, derived next. Equation (11) alone does not assert that a
linear source captures all nonlinear dependence on initialization.
No reflection parity is assumed for the actual source.

<a id="sine-collective-interface-fidelity"></a>
## Uniform finite error for ports, phase and pressure

Let \(g=1/3000\), \(\ell_H=1-2gH\), \(D_H=1-2g^2H^2\),
and suppose each initial form and phase coordinate differs from its
chosen nominal reference by at most \(\epsilon\). The domain
(2) has \(\ell_H>0\), \(16g^2H^2<1\) and \(2gA_*<1\).

For the nominal reference, the Markov heat phase satisfies
\[
 y_{\rm heat}(t)=\gamma\sum_{t_j\le t}
       [I-\exp(-A(t-t_j))]B_pq_j,
 \qquad \|y_{\rm heat}\|_\infty\le gA_*.
 \tag{12}
\]
The \(\ell^1\) variation in (2) is important for simultaneous
multichannel events. The full real nominal phase consequently
satisfies \(\|y_{\rm nom}\|_\infty\le gA_*/D_H\).
The actual-to-nominal full source difference satisfies
\(\max(X_\Delta,Y_\Delta)\le\epsilon/\ell_H\), by heat
contraction and both evolution rows under identical jumps.

Write the normalized nonlinear sine residual as
\(N_k(y)=D^{-1}[S(\Theta_k+y)+L_{c,k}y]\).
Its derivative has maximum-norm bound
\(\|DN_k(y)\|_\infty\le4\|y\|_\infty\).
Thus
\[
 \|N_k(y_{\rm actual})-N_k(y_{\rm nom})\|_\infty
 \le4\left(\frac{gA_*}{D_H}+\frac{\epsilon}{\ell_H}\right)
         \frac{\epsilon}{\ell_H}.
 \tag{13}
\]
Subtract the complete linear-source solution \(\exp(J_kt)z_0\)
from that full source difference. Its remaining form error has zero
initial value and no event jumps. The linear phase feedback gives
denominator \(D_H\), so the entire original-coordinate form defect
is bounded by
\[
 E_{\rm init}(A_*,H,\epsilon)=
 \frac{4gH\epsilon}{\ell_HD_H}
 \left(\frac{gA_*}{D_H}+\frac{\epsilon}{\ell_H}\right).
 \tag{14}
\]
Its phase defect is at most \(2gH E_{\rm init}\). At \(A_*=0\)
the small quadratic initialization term in (14) remains; it must
not be silently replaced by zero.

For the zero-residual nominal response let
\(P_+=(I+\mathcal R)/2\). It is a maximum-norm contraction and
commutes with \(A\). The entire projected form is odd in the
auxiliary amplitude. The
[complex-amplitude proof](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-amplitude-tail)
therefore applies to \(P_+x\), not just one central scalar:
\[
 \|P_+(x_{\rm nom}-x_1-x_3)\|_\infty
 \le R_5(A_*,H)=
 \frac{256g^6A_*^5H}{D_H(1-4g^2A_*^2)}.
 \tag{15}
\]
The proof uses the disk radius \(1/(2gA_*)\), the full nonlinear
minus tangent form bound \(8gH/D_H\), and odd Cauchy tails.
For \(A_*=0\), set \(R_5=0\). It requires no new coefficient
calculation. Applying (15) to arbitrary actual residuals would be
invalid; (13)-(14) are their separate treatment.

Central selectors and central Laplacian rows are reflection invariant.
Combining (14)-(15), the interface errors over the whole window obey
\[
 \begin{split}
 |x_p-x_{{\rm int},p}|&\le R_5+E_{\rm init},\\
 |y_p-y_{{\rm int},p}|&\le2gH(R_5+E_{\rm init}),\\
 |(Lx)_p-(Lx_{\rm int})_p|&\le2d_p(R_5+E_{\rm init}),
 \qquad p\in\{4,13,22\}.
 \end{split}
 \tag{16}
\]
Here the auxiliary full-coordinate \(x_{\rm int}\) is reconstructed
from the form parts of \(z_s+z_1+z_3\), plus the common form.
It is used only for these reflection-invariant observations; an
unprojected full-state error of size \(R_5\) is not asserted.
The hidden quadratic level remains in the cubic forcing even
though its direct contribution to these observations vanishes.
For phase, integrate \(\gamma A P_+\) on the full projected form
remainder; a central-form bound alone would not justify this step.
For pressure, use the exact Laplacian row norm \(2d_p\), equal to
six at donor/receiver and eight at the mediator. The reconstructed
full coefficient/source functionals supply that row observation.

<a id="sine-collective-interface-work"></a>
## Boundary work and limits of the general input domain

The work of one simultaneous central form jump is exactly
\[
 W(q;x^-)=q^TB_p^TLx^-+\tfrac12q^TB_p^TLB_pq,
 \qquad
 B_p^TLB_p=\begin{pmatrix}3&-1&0\\-1&4&-1\\0&-1&3\end{pmatrix}.
 \tag{17}
\]
The cross terms cannot be replaced by a sum of independent scalar
works. Equation (16) gives a preevent pressure/work enclosure from
the same causal interface, with error at most
\(\sum_p2d_p|q_p|(R_5+E_{\rm init})\) on its linear term.
The exact phase row also gives \((Lx)_p=(d_p/\gamma)\theta'_p\),
but differentiating a retained pointwise phase enclosure does not
produce a justified derivative enclosure. No such shortcut is used.

The interface retains an interaction port, not a storage determined
solely by six present values. Hidden energy and initialization remain
relevant to balance and future output. The general variation domain
(2) is an approximation domain, not a universal work or identity
certificate. For example, a single donor impulse \(7/5000\) at
the exact target costs \(2.94\,10^{-6}\), above the original
per-event allowance. The fixed two-event word alone inherits its
separate full-law work, trapping and return proof.

<a id="sine-collective-interface-discrimination"></a>
## Preservation of the repeated class-dependent interaction

Use the retained complete cubic contrast enclosure
\[
 J_0=[-7.013764940694,-7.013764940692]\,10^{-30},
 \qquad J_3=(7/5)^3J_0.
 \tag{18}
\]
This is the nominal cubic output of (7)-(9) for the already declared
word, by uniqueness of the eliminated hierarchy. Its generating
execution remains the
[retained coefficient premise](SINE_CLASS_CUBIC_RESPONSE.md#sine-cubic-finite-result).
Equation (18) is not a new observation and is not inferred from the
full nonlinear forward report.

Let \(a=b=7/10000\), \(H=2\), \(\epsilon=10^{-32}\) and
\(\delta=10^{-30}\). For each history use
\(A_{00}=0,A_{10}=a,A_{01}=b,A_{11}=a+b\), and set
\[
 R_\Sigma=2\sum_hR_5(A_h,2),\qquad
 E_\Sigma=2\sum_hE_{\rm init}(A_h,2,\epsilon),\qquad
 B_s=\frac{8\epsilon}{1-4g}.
 \tag{19}
\]
The factors two count the two classes; the two single-input histories
remain distinct even when \(a=b\). The retained linear-source
term of (10) has contrast magnitude at most \(B_s\), with no
cancellation assumed for the distinct recurrent residuals. The
actual nonlinear contrast therefore lies in
\[
 J_3+[-R_\Sigma-E_\Sigma-B_s,
                R_\Sigma+E_\Sigma+B_s].
 \tag{20}
\]
There is one nominal truncation allowance. Starting instead from a
full-response interval and composing an interface approximation
in both directions would require its own additional error accounting;
that two-step construction is not used in (18)-(20).

The [persistent tangent alternative](SINE_CLASS_REPEATED_INTERACTION.md#sine-repeated-response-and-comparator)
has its own \(B_s\) residual allowance. Both mean progressions
cancel in the mixed statistic and each model contributes its own
eight reading errors. The sufficient recurrent separation margin is
\[
 -\sup J_3-R_\Sigma-E_\Sigma-2B_s-16\delta>0.
 \tag{21}
\]
Exact rational substitution gives
\[
 R_\Sigma<8.027\,10^{-33},\qquad
 E_\Sigma<4.985\,10^{-41},\qquad
 B_s<8.011\,10^{-32},\qquad
 -\sup J_3-R_\Sigma-E_\Sigma-2B_s-16\delta
     >3.0775\,10^{-30}.
 \tag{22}
\]
These inequalities use the
retained coefficient premise. The tiny initialization defect is
additional to the retained linear source, not a claim that arbitrary
source uncertainty has disappeared. The separate full-law return
establishes the next word's small centered source and retains its
means; no cubic evolution is asserted over the long dwell.

The result is a nonlinear collective interface with quantified
memory, initialization and observation errors. It supplies neither
an autonomous input selector nor a physical calibration, an intrinsic
particle label or a new fundamental interaction law. Finite-dimensional
Markov closure, minimal memory and inheritance by different support or
constitutive laws remain separate questions.

## Implementation and evidence boundary

The private
[collective-interface owner](../../src/tnfr/physics/_sine_class_collective_interface.py)
provides the fixed structural blocks, edge factors, modal-kernel
description and error certificates. It reuses the admitted amplitude
tail and repeated-interaction bounds. The separate
[changed-input prediction](SINE_CLASS_CHANGED_INPUT_PREDICTION.md)
executes the causal time-coefficient calculation with its own fixed input,
source, numerical and observation budgets. The structural descriptor itself
does not evaluate a kernel or forward response.

The [independent structural and budget controls](../../tests/physics/test_sine_class_collective_interface.py)
check the complete partition, hidden modes, nonlinear recoupling,
initialization bounds and simultaneous-event work. The
[retained cubic evidence reader](../../tests/physics/test_sine_class_cubic_evidence.py)
reconstructs the existing coefficient enclosure and its association
with (18)-(22), without regenerating coefficients or trajectories.
Supplying a cubic interval to the private certificate remains
conditional on that interval's independently established evidence.
No new response archive is needed for this theorem.
