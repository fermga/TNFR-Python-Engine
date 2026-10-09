# Fine preparation information selects different acquired winding classes

<a id="sine-preparation-mode-selection"></a>

Under one fixed complete law, equal initial phase, conserved means and
storage do not determine the acquired winding magnitude. This owner constructs
two explicitly paired source families on C9. Every pair has exactly those
same preparation observations, yet its two complete trajectories acquire
winding magnitudes one and two. The pairing retains full form and phase
uncertainty; equality is not inferred from overlapping error intervals.

The law is the sine member of the
[constitutive family](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-complete-law).
No constitutive parameter is varied. The construction and scalar semigroup
certificate use no trajectory, reverse-flow definition, coefficient search
or frozen scientific response.

<a id="sine-mode-selection-law-and-source"></a>
## Fixed law and an algebraic preparation pair

Use the isolated unit C9, nodes \(0,\ldots,8\), edges
\(j--(j+1\bmod9)\), unnormalized Laplacian \(L\), degree matrix \(2I\)
and held capacities one. Keep
\[
 e=1023/1024,\quad w=1/1024,\quad
 \tau=et,\quad\gamma=1/(1023\pi),\quad A=L/2 .
 \tag{1}
\]
The entire state is signed real form \(x\) and primitive circular phases
with continued real lifts \(\theta\). In the two clocks the complete rows are
\[
 \dot x=-eLx/2+(w/\pi)S(\theta)/2,\qquad
 \dot\theta=(w/\pi)Lx/2,
 \qquad
 x'=-Ax+\gamma f(\theta),\quad\theta'=\gamma Ax,
 \tag{2}
\]
where \(S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i)\) and \(f=S/2\).
Storage and structural-time loss are
\[
 H(x,\theta)=\tfrac12x^TLx+\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)],
 \qquad H'=-\tfrac12\|Lx\|_2^2.
 \tag{3}
\]
Both ordinary means, equivalently the degree-weighted means on this regular
support, are conserved. Pressure, support, capacity and structural units are
supplied premises. There is no event, forcing or state reset during acquisition.

Define the exact rational vectors and scale
\[
 r_j=j-4,\qquad h=e_3-e_4,\qquad
 m=\frac{2046}{9}\left(\frac{355}{113}\right)^2,\qquad
 a=(9M+M^{-1})h+r,\quad b=(9M-M^{-1})h+2r .
 \tag{4}
\]
The declared integer is \(M=512\). These are algebraic source coordinates,
not states selected from an evaluated future trajectory. Direct graph
arithmetic gives
\[
 \mathbf1^Tr=\mathbf1^Th=0,\quad r^TLr=72,\quad
 h^TLh=6,\quad h^TLr=0,\quad
 a^TLa=b^TLb=486M^2+180+6/M^2 .
 \tag{5}
\]
The equal energy comes from the compensating \(M^{-1}h\) terms. The
unmodified ramps \(r,2r\) have unequal energy and would not answer this
question.

Set \(d=a-b=2h/M-r\), and define the full form map
\[
 R=I-\frac{2d(d^TL)}{d^TLd},\qquad
 d^TLd=72+24/M^2>0 .
 \tag{6}
\]
This metric reflection obeys
\[
 R^2=I,\quad R^TLR=L,\quad Ra=b,\quad
 R\mathbf1=\mathbf1,\quad \mathbf1^TR=\mathbf1^T .
 \tag{7}
\]
For example equal norms in (5) imply
\(d^TLa=d^TLd/2\), proving \(Ra=b\); expanding (6) proves the
energy identity. The two mean identities use \(L\mathbf1=0\) and
\(\mathbf1^Td=0\). This is not assumed to preserve Euclidean error
norms. It is not a graph or complete-flow symmetry. In particular the
form-only transformation \((x,\theta)\mapsto(Rx,\theta)\) could preserve
the phase row only if \(AR=A\); invertibility of \(A\) on the mean-zero
space would require \(R=I\) there, contrary to \(Ra=b\ne a\).

Declare latent errors \(\xi,\zeta\) satisfying
\[
 |\xi_j|,|\zeta_j|\le\epsilon=10^{-10},\qquad
 \mathbf1^T\xi=\mathbf1^T\zeta=0 .
 \tag{8}
\]
For every such pair use the two complete initial states
\[
 (x_1(0),\theta_1(0))=(ma+\xi,\zeta),\qquad
 (x_2(0),\theta_2(0))=(mb+R\xi,\zeta).
 \tag{9}
\]
Each branch is the invertible image of a sixteen-dimensional relative
source neighborhood. The paired experiment has sixteen latent dimensions,
not thirty-two independently selectable ones.

For \(s_i=\sum_j|R_{ij}|\), the second branch satisfies
\(|(R\xi)_i|\le s_i\epsilon\). Exact matrix arithmetic gives
\[
 K:=\max_i s_i=\frac{1968129}{786433}<\frac{251}{100}.
 \tag{10}
\]
We use \(K\epsilon\) as a sufficient common coordinate bound for that
branch. The initial phase bound remains \(\epsilon\) in both branches.
The matching claim does not require equal marginal form-error limits.

<a id="sine-mode-selection-matched-observations"></a>
## Exact equality of the preparation observations

Consider the coarse preparation map consisting of the entire initial
phase vector, both conserved means and the scalar storage \(H\).
Equations (7)--(9) imply, for every admitted latent pair,
\[
 \theta_1(0)=\theta_2(0)=\zeta,\quad
 \overline x_1=\overline x_2=\overline\theta_1
 =\overline\theta_2=0,\quad
 H(x_1(0),\zeta)=H(x_2(0),\zeta).
 \tag{11}
\]
The last equality holds because \(x_2(0)=Rx_1(0)\), so their complete
form energies coincide, and their phase potential is literally the same.
Storage may vary with the latent errors; it is equal within each pair.
The result therefore concerns equal observed values, not only equal nominal
centers or membership in a common storage budget.

An edgewise upper bound on that common source cost is
\[
 B=\frac12\sum_{\{i,j\}}
       (m|a_j-a_i|+2\epsilon)^2+18\epsilon^2
 <320680599854722<10^{15}.
 \tag{12}
\]
The phase term uses all nine edges and \(1-\cos v\le v^2/2\).
For the fixed vector, \(\sum_{\{i,j\}}|a_j-a_i|=2361089/128\).
Branch two has the same cost by exact isometry; applying an unrelated
Cartesian error box to its form coordinates would discard that equality.
The nominal common form cost is
\(m^2(486M^2+180+6/M^2)/2\).

Every initial principal phase difference equals its small lift difference,
since \(2\epsilon<\pi\). Summing around the cycle gives winding zero
for every source member. Neither source initially has the claimed nonzero
phase-winding identity, despite its deliberately structured fine form.

<a id="sine-mode-selection-full-family-acquisition"></a>
## Acquisition without evaluating trajectories

Declare the common structural time \(T=200\), original time
\(204800/1023\), capture radius \(r_c=1/12\), source budget \(10^{15}\)
and shared outward dyadic128/Machin scalar arithmetic. These are the
new preparation's fixed budgets; they are not presented as the older
ramp experiment's time or cost. The error scale in (8) is unchanged.

Let \(\alpha=2\pi/9\) and define the continued target lifts
\[
 \Theta_k=\alpha(9Mh+kr),\qquad k=1,2 .
 \tag{13}
\]
Their extra term is \(2\pi Mh\), a nodewise integer-turn shift. Consequently
the circular targets, their currents, storage and Hessians are exactly the
usual winding-\(k\) targets. No corresponding shift is applied to an
actual trajectory. These lifts have zero mean and specify the endpoint
comparison in the same continued chart as (9).

The graph and complete nonlinear forcing satisfy
\[
 \|e^{-A\tau}P_9\|_{2I}\le e^{-\tau/5},\quad
 P_9=I-\mathbf1\mathbf1^T/9,\quad
 \|f(\theta)\|_{2I}\le F=\sqrt{18},\quad
 \lambda_{\max}(L)\le4 ,
 \tag{14}
\]
as proved in the
[isolated acquisition owner](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-same-source-acquisition).
Only these graph and law estimates are reused; the new source norms and
target distances are rebuilt.

Write \(a_1=a,a_2=b\), \(\epsilon_1=\epsilon\),
\(\epsilon_2=K\epsilon\), \(\eta=\gamma^2\), and take
\(\rho\ge e^{-T/5}\). For each complete source family define
\[
 \begin{aligned}
 V_k&=\gamma m\sqrt{2a_k^Ta_k},&
 E_{x,k}&=\sqrt{18}\gamma\epsilon_k,&
 E_\theta&=\sqrt{18}\epsilon,\\
 R_{z,k}&=\rho(V_k+E_{x,k})+5\eta F,&
 R_{x,k}&=R_{z,k}/(\gamma\sqrt2),\\
 D_k&=\|\gamma m a_k-\Theta_k\|_2,&
 Y_k&=D_k+(E_{x,k}+E_\theta+R_{z,k}+\eta FT)/\sqrt2,\\
 Z_k^2&=R_{x,k}^2+Y_k^2 .
 \end{aligned}\tag{15}
\]
The first estimate follows from variation of constants for
\(z=\gamma x\), \(z'=-Az+\eta f(\theta)\). The integrated complete
rows give
\[
 \theta(T)-\gamma ma_k
 =\zeta+\gamma\xi_k-z(T)+\eta\int_0^T f(\theta(s))\,ds ,
 \quad \xi_1=\xi,\quad\xi_2=R\xi .
 \tag{16}
\]
Therefore \(\|x_k(T)\|_2\le R_{x,k}\),
\(\|\theta_k(T)-\Theta_k\|_2\le Y_k\), and the complete endpoint
radius is at most \(Z_k\). In particular the large integer-turn part
of the target has not been dropped from the initial norm or proxy error.
The small rational approximation error in \(\gamma m-\alpha\) is
retained separately in every component of \(D_k\).

Target stationarity and the global Hessian upper bound give
\(H(T)-H_k^*\le2Z_k^2\), where
\(H_k^*=9(1-\cos k\alpha)\). Put
\[
 c=\cos(4\pi/9+\sqrt2r_c),\qquad
 b_c=cr_c^2/5 .
 \tag{17}
\]
The acute margin is positive. Within either radius-\(r_c\) ball,
the phase Hessian is bounded below by \(cL\); the fixed zero means
and \(\lambda_2(L)>2/5\) give the same full-state storage barrier \(b_c\).

### Strict fixed-budget margins

Direct scalar interval evaluation of (12)--(17), with no parameter sweep,
gives the conservative bounds

| Complete source branch | \(R_{x,k}\), upper | \(Y_k\), upper | \(Z_k^2\), upper | \(2Z_k^2\), upper |
| --- | ---: | ---: | ---: | ---: |
| Winding one | \(0.004667300447\) | \(0.002760425304\) | \(2.940364132\,10^{-5}\) | \(5.880728263\,10^{-5}\) |
| Winding two | \(0.004667300447\) | \(0.001215414795\) | \(2.326092658\,10^{-5}\) | \(4.652185316\,10^{-5}\) |

These decimals denote exact outward rational bounds. The common barrier
satisfies \(b_c>7.868256725\,10^{-5}\). The directly computed strict
storage margins exceed \(1.987528462\,10^{-5}\) and
\(3.216071409\,10^{-5}\), respectively. In particular, for both branches,
\[
 Z_k^2<3/10^5<r_c^2,\qquad
 2Z_k^2<59/10^6,\qquad
 b_c>78/10^6,\qquad b_c-2Z_k^2>19/10^6 .
 \tag{18}
\]
The preparation and these budgets were specified algebraically before
this scalar assessment. Its output is an analytic bound on all complete
flows, not a simulated response or an observed endpoint.

By storage nonincrease, every endpoint is forward trapped in its target's
acute ball. Its attained winding is \(k\) and remains so thereafter.
On the compact zero-mean leaf, zero loss requires \(x=0\), and invariance
then requires \(S(\theta)=0\). Strict phase convexity leaves only the
declared target lift. LaSalle gives convergence. Winding acquisition
includes an intervening antipodal crossing, which the smooth complete law
allows; no acute constraint has been imposed throughout the transit.

<a id="sine-mode-selection-information-boundary"></a>
## Information consequence and scope

There is already a direct rate witness for the information discarded at
preparation. Since \((Ld)_0=9\), the nominal initial phase-rate contrast is
\((9/2)\gamma m\). For every paired residual it satisfies
\[
 \left|\theta'_{1,0}(0)-\theta'_{2,0}(0)
                    -\frac92\gamma m\right|
 \le2\gamma(1+K)\epsilon .
 \tag{19}
\]
This follows from \(\|L\|_\infty=4\) and
\(\|(I-R)\xi\|_\infty\le(1+K)\epsilon\). It is independent of the
common phase error because the complete phase row consumes form directly.
The bound is much smaller than the nonzero center. A difference in initial
rates alone would not establish different acquired identities; that
stronger conclusion uses the full-family transit and capture proof above.

Absolute winding one and two are inequivalent under C9 rotations,
reflections, common origins and joint form/phase sign reversal. Equations
(11) and (18) therefore disprove sufficiency of the declared coarse
preparation map for determining acquired identity on these paired
families. Deterministic full-state evolution remains well posed: it is
the discarded fine form arrangement and its exact correlations that
the observation fails to specify.

This does not prove a minimal sufficient preparation statistic, a
probability of either mode, or an autonomous selector. The algebraic
sources deliberately encode information and require substantial storage.
No claim is made that unstructured or laboratory-accessible preparations
realize them. Capacity, support, positive loss and the sine law remain
supplied. No new constitutive selection follows.

The endpoint bounds are actual-family inclusions, not target resets.
Any later dwell, contact, finite response or effective-interface use
must re-admit the reached states, hidden initialization, support, clock
and errors. Equal initial storage does not imply equal instantaneous
loss, accumulated loss, or complete histories.

The [continuous source-path comparison](SINE_PREPARATION_PATH_SELECTION.md)
retains these distinctions when interpolating the fine preparation. Its
endpoint residual map is newly admitted, and a transient winding label
does not replace the complete-state capture premise.

## Implementation and independent controls

The private
[_sine_preparation_mode_selection.py](../../src/tnfr/physics/_sine_preparation_mode_selection.py)
reconstructs the source vectors, exact reflection, paired storage identity,
unwrapped target distances and full-family bounds. Its shared transit
arithmetic accepts the newly admitted norms; it consumes no earlier
formation verdict. Per-row bounds \(s_i\epsilon\) may sharpen the second
initial error norm used in (15), without weakening the stated bounds.
The [independent controls](../../tests/physics/test_sine_preparation_mode_selection.py)
check the metric isometry and non-symmetry, exact observation matching,
graph/lift association, source costs and strict capture margins.
No trajectory or frozen producer is executed by this admission.
