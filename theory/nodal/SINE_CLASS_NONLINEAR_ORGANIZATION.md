# Mediator organization and finite nonlinear superposition

<a id="sine-class-nonlinear-organization"></a>

This owner asks whether changing an acquired mediator's organization changes
the nonlinear mixed response of the same complete three-C9 system. The
[one-probe class contrast](SINE_CLASS_MEDIATED_RESPONSE.md#sine-class-mediated-result)
and the [retained class-two nonlinear result](SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-reserved-result)
answer different questions. Neither supplies this cross-class contrast.

The calculation below cancels shared analytic contributions before enclosing
errors. At the unchanged two-probe design it produces a rigorous but
unresolved finite contrast. It also proves a limitation of the inherited
independent-class remainder estimate. This is not a symmetry theorem, a
record-overlap result or evidence that the actual contrast vanishes. No new
nonlinear trajectory, formation assessment or reserved response is evaluated.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the remaining research obligation.

<a id="sine-nonlinear-organization-source"></a>
## F1-F2: two acquired families and one matched observation

Use the full 27-node support of three C9 components, with contacts
\((4,13)\) and \((13,22)\), unit capacities and actual central degrees
\((3,4,3)\). The two class tuples are \((1,k,1)\), \(k=1,2\).
All 54 original form and phase coordinates obey the same supplied law
\[
 x'=-Ax+\gamma KS(\theta),\qquad \theta'=\gamma Ax,
 \qquad A=KL,\quad \gamma=1/(1023\pi).
 \tag{1}
\]
The clock is \(\tau=et\), \(e=1023/1024\); phase deviations use
continuous lifts around \(\Theta_{i,j}=k_i(2\pi/9)(j-4)\).
Form and time retain their declared structural units.

The [original source and handoff](SINE_CLASS_MEDIATED_RESPONSE.md#sine-class-mediated-state)
remain conditional premises, separately for both classes. They start from
flat phases and class-dependent form ramps
\[
 x_{i,j}(0)=k_i m(j-4)+\xi^x_{i,j},\qquad
 \theta_{i,j}(0)=\xi^\theta_{i,j},\qquad
 m=\frac{2046}{9}\left(\frac{355}{113}\right)^2.
 \tag{2}
\]
Origins are aligned from source time. The original per-coordinate errors
are at most \(10^{-10}\), with separate zero sums in each component
and channel. Formation time is 100, additional unprobed dwell is
\(10^{13}\), and the inherited guarded decay exponent is 512. Each
component's original source storage ceiling is \(10^9\). The matched
handoff theorem bounds both form and target-phase Euclidean norms by
\(\epsilon=10^{-32}\). These assertions are not rebuilt from a final
response or a cached verdict. The different class preparations retain their
different costs; no energy-matching assumption is made.

For each class, four continuations start from the same complete actual
reached state, with donor central form jumps \(a\) at zero and \(b\)
at delay \(s\), present or absent according to history \(00,10,01,11\).
All coordinates carry through both events. The two classes may have
independent source residuals. Define
\[
 M_k=R_{k,11}-R_{k,10}-R_{k,01}+R_{k,00},\qquad D=M_1-M_2,
 \tag{3}
\]
where \(R_{k,ij}\) is receiver central form at the common horizon.
Keep the existing design, fixed before the analytic channel calculation:
\[
 a=b=1/2000,\quad s=1,\quad T=2,\quad
 \epsilon=10^{-32},\quad \delta=10^{-30},\quad r=1/12.
 \tag{4}
\]
Each of the eight final scalar readings has independent unknown error of
magnitude at most \(\delta\); independence here grants separate allowed
errors, not a probability law. No laboratory calibration is implied.

<a id="sine-nonlinear-organization-channel"></a>
## F3: the common heat channels cancel

Let \(\alpha=2\pi/9\), \(c_k=\cos(k\alpha)\), and
\(d_c=c_1-c_2>0\). The normalized heat semigroup
\(P(t)=\exp(-At)\) is the same for both classes. For \(t\ge s\),
set \(p(t)=a[I-P(t)]e_D\), \(q(t)=b[I-P(t-s)]e_D\).
The cubic map inherited from the sine law is
\[
 N_{3,k}(v)_i=-\frac1{6d_i}\sum_{j\sim i}
    c_{ij,k}(v_j-v_i)^3.
 \tag{5}
\]
Outer cycle edges have weight \(c_1\), mediator cycle edges have weight
\(c_k\), and the two contacts have weight one. Consequently the
[finite heat coefficient](SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-heat-coefficient)
decomposes exactly as
\[
 \mathcal C_k=c_1\mathcal C_O+c_k\mathcal C_M+\mathcal C_B,
 \qquad \mathcal C_1-\mathcal C_2=d_c\mathcal C_M.
 \tag{6}
\]
The common outer and contact channels cancel before interval arithmetic.
No equality of actual source residuals between classes is used here.
Writing \(\Delta p=p_j-p_i\), \(\Delta q=q_j-q_i\), the remaining
coefficient is
\[
 \mathcal C_M=-\frac12\int_s^T\sum_{(i,j)\in E_M}
 \left[\frac{P(T-t)_{R,i}}{d_i}-\frac{P(T-t)_{R,j}}{d_j}\right]
 \Delta p\,\Delta q\,(\Delta p+\Delta q)\,dt.
 \tag{7}
\]
Every mediator edge and its actual joined degree remain in this expression.
It is orientation independent and is an analytic coefficient, not an
observed nonlinear response.

Central-port symmetry does not make (7) identically zero. For fixed
amplitudes, scale \(s=\lambda s_0\), \(T=\lambda T_0\), and put
\(u=T_0-s_0\). The two mediator edges incident on node 13 give
\[
 \mathcal C_M=-\frac{\lambda^5ab}{768}
 \left[\frac{as_0^2u^3}{6}
       +\frac{(2a+b)s_0u^4}{12}
       +\frac{(a+b)u^5}{20}\right]+O(\lambda^6).
 \tag{8}
\]
Indeed \(p_{13}(t)=-at/4+O(t^2)\), its mediator neighbors have
\(p=O(t^2)\), and
\(P(v)_{R,13}/d_{13}=v/12+O(v^2)\). The two edge contributions
add; all other mediator edges enter later. Equation (8) concerns the heat
channel, not the complete nonlinear time jet at fixed \(\gamma\).
It supplies no finite lower bound at (4).

The earlier class-blind fourth-order time term and amplitude-parity
restriction remain valid in their [stated scope](SINE_CLASS_NONLINEAR_SUPERPOSITION.md#sine-superposition-symmetry-and-onset).
Reflection fixing the ports can reverse winding orientation under a
reflected complete preparation. It does not identify winding magnitudes
one and two. Arbitrary admitted residuals need not have nominal reflection
symmetry and must retain the following source allowance.

<a id="sine-nonlinear-organization-error"></a>
## Finite channel, full-law and source errors

Keep heat polynomial order 32. Set
\[
 \eta_{32}=\frac{(2T)^{33}}{33!},\qquad
 E_M=4(|a|^2|b|+|a||b|^2)(T-s)[(1+\eta_{32})^4-1].
 \tag{9}
\]
The contraction proof for the complete heat polynomial applies unchanged
to the restricted mediator channel. At every node the selected mediator
incidence count is at most the original degree. Its normalized cubic
trilinear bound therefore has the same constant four, without an edge-count
multiplier. Three phase columns and one propagated row supply the four
factors in (9). Hence \(|\mathcal C_M-\mathcal C_{M,32}|\le E_M\).
Multiply this allowance by an upper bound for \(\gamma^4d_c\), while
retaining outward cosine and \(\gamma\) factors for the rational
polynomial coefficient itself.

For \(g\ge\gamma\), set \(D_T=1-2g^2T^2>0\) and
\(\ell_T=1-2gT>0\). The inherited full-law bound is
\[
 B(A_*,T)=\frac{g^6A_*^3T^3}{D_T^3}
    \left(\frac83+\frac{32}{9\ell_T}\right)
 +\frac{4g^6A_*^5T}{15D_T^5}
 +\frac{8g^8A_*^3T^5}{15D_T^3\ell_T^2},
 \quad B_\Sigma=B(|a|+|b|,T)+B(|a|,T)+B(|b|,T).
 \tag{10}
\]
Its [derivation](SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-remainder)
retains phase-feedback and reflection-odd internal corrections. It is
uniform for both admitted classes. Tangent linearity cancels the common
initial state and the two inputs within each four-history group. The
nonlinear remainder does not cancel between classes merely because the
probe schedule is common. Similarly each actual-to-ideal history has
error at most \(\epsilon/\ell_T\) in the original coordinates, with
no division of phase error by \(\gamma\). Thus
\[
 \left|D-\gamma^4d_c\mathcal C_M\right|
 \le 2B_\Sigma+\frac{8\epsilon}{\ell_T}.
 \tag{11}
\]
If \([H_-,H_+]\) encloses \(\gamma^4d_c\mathcal C_M\), including
(9) and transcendental arithmetic, the resulting true-response enclosure is
\[
 D\in[H_- - F,\ H_+ + F],\qquad
 F=2B_\Sigma+8\epsilon/\ell_T.
 \tag{12}
\]
These are sufficient outer bounds; their Cartesian endpoints need not be
jointly realizable responses.

<a id="sine-nonlinear-organization-events"></a>
## Preparation, event work and identities remain separate

The common per-history envelope
\[
 Q_A(t)=\frac{A+\epsilon+2gt\epsilon}{1-2g^2t^2}
 \tag{13}
\]
bounds form; phase deviation is at most \(\epsilon+2gtQ_A(t)\).
For either class, contact work is at most \(8\epsilon^2\), and the
joined preprobe excess storage is at most \(22\epsilon^2\).
Each donor form jump \(q\) has exact work
\(q(Lx^-)_D+3q^2/2\). Consequently
\[
 W_1\le\tfrac32a^2+6|a|\epsilon,\qquad
 W_2\le\tfrac32b^2+6|b|Q_{|a|}(s).
 \tag{14}
\]
The delayed-only history uses the smaller \(Q_0(s)\), and absent
jumps use zero. These bounds consume the actual preevent history.
With \(g=1/3000\), both work bounds are below \(2\,10^{-6}\),
contact work is below \(10^{-12}\), and
\[
 22\epsilon^2+W_{1,+}+W_{2,+}<2.250003\,10^{-6}
    <\frac1{388800}=\frac{r^2}{2700},\qquad
 27\{Q_{|a|+|b|}(T)^2+
 [\epsilon+2gTQ_{|a|+|b|}(T)]^2\}<r^2.
 \tag{15}
\]
The uniform acute-ball coercivity from the
[identity owner](SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-events-and-identity)
applies to both classes. All three identities and unforced recovery after
the last event retain their conditional source premises. Each present
impulse shifts the actual weighted form mean by \(3q/58\); phase mean
is unchanged. Neither mean is reset. Failure to resolve (12) does not
invalidate these separate identity and work conclusions.

<a id="sine-nonlinear-organization-records"></a>
## F4: three distinct observation claims

The eight supplied scalar error bounds imply
\[
 |\widehat D-D|\le8\delta.
 \tag{16}
\]
A true contrast enclosure strictly beyond \(8\delta\), or below
\(-8\delta\), guarantees a recorded sign. It also suffices to separate
the two classes' four-record sets, since each class's mixed statistic has
error at most \(4\delta\). These are sufficient tests of the stated
observable, not necessary conditions for distinguishing whole records.

A different comparison takes all eight records as one observation and
contrasts the nonlinear model with a declared class-blind alternative
having noiseless \(D=0\). That alternative's own eight errors permit
\([-8\delta,8\delta]\). Separation through this statistic therefore
uses a sufficient strict true margin greater than \(16\delta\).
The complete tangent law is one such declared postcontact alternative;
its mixed statistic vanishes separately in each class. No exclusion of
every possible alternative law follows.

<a id="sine-nonlinear-organization-unresolved"></a>
## Fixed-design outcome and a proved limitation of this estimate

One exact order-32 channel calculation at (4) gives
\(\mathcal C_{M,32}\) approximately
\(-1.26311402922904\,10^{-15}\). It is a rational polynomial integral,
not a full nonlinear response. Including the restricted tail (9) and
outward factors establishes the exact terminating-rational inequalities
\[
 -7.02\,10^{-30}<H_-\le\gamma^4d_c\mathcal C_M
       \le H_+<-7.01\,10^{-30}.
 \tag{17}
\]
The scaled tail is less than \(1.89\,10^{-40}\). The coefficient is
negative in the declared class-one-minus-class-two orientation, but the
sign of an analytic coefficient is not a finite full-law conclusion.

Using the shared outward upper bound for \(\gamma\) in (10), exact
rational arithmetic gives
\[
 2B_\Sigma\approx1.13014105178530\,10^{-28},\qquad
 8\epsilon/\ell_T<8.02\,10^{-32}.
 \tag{18}
\]
The actual interval constructed by (12), rounded outward for display, is
\[
 D\in[-1.201080,\ 1.060805]10^{-28}.
 \tag{19}
\]
Its exact lower endpoint is negative and its exact upper endpoint is
positive. True sign, recorded sign and both record-separation tests are
therefore unresolved by this estimate. The preparation and design were
not changed after the coefficient calculation. The earlier class-two
forward enclosure is unchanged and is not reused as previously unseen
cross-class data.

This limitation persists even if source and sensor errors were zero.
The positive terms in (10), \(D_T,\ell_T<1\), and \(g>1/3300\)
give, using only the history with \(A_*=1/1000\),
\[
 2B_\Sigma>
 2\frac{56}{9}\left(\frac1{3300}\right)^6
       \left(\frac1{1000}\right)^3 2^3
 >7\,10^{-29}.
 \tag{20}
\]
This exceeds the entire magnitude of (17). Thus the independent-class
remainder construction necessarily crosses zero at this design even
without source or sensor error. Improving only heat-polynomial order or
arithmetic precision cannot close that gap. In addition, the isolated
coefficient's magnitude in (17) is below \(8\delta\); discarding the
full-law error would still not justify the recorded-sign test.

Equations (19)-(20) prove a limitation of these sufficient estimates.
They do not prove \(D=0\), a symmetry between the classes, an actual
noise-overlap witness, or absence of organization dependence. In particular,
overlap of outer intervals is not a pair of realizable complete records.

<a id="sine-nonlinear-organization-boundary"></a>
## Implementation boundary and the remaining mathematical obligation

The [conditional calculator](../../src/tnfr/physics/relational_sine_class_nonlinear_organization.py)
uses the existing shared exact heat polynomial and fixed geometry. Its
common channels cancel symbolically; only the mediator channel and its
justified tail enter (12). It retains primitive source radii, event budgets
and separate work/identity guards, and executes neither source acquisition
nor a forward response. The [API controls](../../tests/physics/test_sine_class_nonlinear_organization.py)
and [independent algebra controls](../../tests/physics/test_sine_class_nonlinear_organization_algebra.py)
check those distinctions, the common-channel cancellation, the mediator
tail, noise thresholds and method limitation. An unresolved result does
not convert failure of a sufficient inequality into physical
nonidentifiability.

The unresolved question calls for a tighter full-law comparison at the
same finite horizon. A candidate is the complete class-dependent
cubic-amplitude variational response around the nominal equilibrium,
retaining phase feedback and the quadratic reflection-odd corrections
which return to the central observation at cubic order. The current heat
coefficient omits these contributions into (10). Such a derivation must
retain both event times, all hidden initialization, original-coordinate
source errors and a rigorous remainder beyond the cubic-amplitude term.
Amplitude parity applies to the nominal source, not arbitrary residuals.
The present calculation evaluates neither that refined coefficient nor
a new response.

A tighter bound could establish a finite sign, or prove that this scalar
statistic admits noise cancellation. The latter would still not imply
overlap of the full eight-record sets. Those are separate future proof
obligations, not results of this owner. The
[ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization) likewise
keeps an inherited organization-dependent property distinct from a
physical identification or a uniquely selected microscopic law.
