# Regional contact phases and retained profile information

<a id="mixed-regional-loop"></a>
## 1. One unchanged pair and a new observation question

The [conditional pair identity](ORGANIZED_COMPOSITE_RELAXATION.md#mixed-pair-identity)
retains four nine-cycle regions, their marked relations and the original
source association under a supplied dissipative law. This chapter asks
whether the phases of their actual contact coefficients factor through one
phase per region, and what an instantaneous regional response loses when
only coefficient magnitudes are retained. The prospective scope is
`artifacts/mixed-regional-loop-20261026/scope.json`.

No new preparation, topology, contact, source replay or trajectory is supplied.
The two acquired preparations, all held unequal conductances, unit capacities,
phase lifts, source-correlated initialization and previously paid attachment
work remain those of the linked source. Time is the retained structural clock
\(\tau=(1023/1024)t\). The [phase-sensitive interface](ORGANIZED_COMPOSITE_RELAXATION.md#mixed-phase-sensitive-interface)
owns the full seventy-two rows and the attachment accounting.

Write the regions as \(D_A,D_B,R_A,R_B\), each of size nine. Let \(H\) be the
two-copy held form Laplacian and \(G\) the eighteen matching-pair Laplacian.
The complete law, with \(L_+=H+9G\) and \(\gamma=1/(1023\pi)\), is
\[
 x'=-\frac{L_+x}{9}+\frac{\gamma}{9}S_+(\theta),\qquad
 \theta'=\frac{\gamma L_+x}{9},\qquad
 (S_+)_i=\sum_j\sigma_{ij}\sin(\theta_j-\theta_i).
                                                               \tag{1}
\]
The symmetric sine coefficients are one on the held internal support and
nine on each inter-copy matching edge. They are the coefficients of the bare
sine potential, not the unequal conductances in \(H\). The divisor nine and
both evolution rows are unchanged.

Use the carried stationary template \(\Theta\) and continuous phase lifts.
The canonical mean frame in this chapter is template-relative:
\[
 \varphi_a=\frac19\sum_{i\in a}(\theta_i-\Theta_i),\qquad
 h_i=\theta_i-\Theta_i-\varphi_a,\qquad
 \eta_i=\Theta_i+h_i,\qquad \sum_{i\in a}h_i=0.
                                                               \tag{2}
\]
Thus \(\theta_i=\varphi_a+\eta_i\). No assumption that each template's
arithmetic regional mean vanishes is necessary. Changing individual circular
representatives without transporting the admitted lifts is not a new
admissible source gauge.

## 2. Coefficients inherited from the actual sine support

For distinct adjacent regions define
\[
 C_{ab}=\sum_{\substack{i\in a\\j\in b}}
           \sigma_{ij}e^{i(\eta_j-\eta_i)},\qquad
 F_{ab}=\operatorname{Im}\!\left[
              e^{i(\varphi_b-\varphi_a)}C_{ab}\right].
                                                               \tag{3}
\]
Here \(F_{ab}\) is exactly the outgoing-neighbor sine sum in the form row
of region \(a\); the word contact does not add an independent transport law.
Symmetry of \(\sigma\) gives \(C_{ba}=\overline{C_{ab}}\) and
\(F_{ba}=-F_{ab}\). Internal edges within one region cancel on summation.
For regional mean form \(m_a=\sum_{i\in a}x_i/9\), the exact projected rows are
\[
 m_a'=-\frac1{81}\sum_{i\in a}(L_+x)_i
                  +\frac{\gamma}{81}\sum_{b}F_{ab},\qquad
 \varphi_a'=\frac{\gamma}{81}\sum_{i\in a}(L_+x)_i .
                                                               \tag{4}
\]
The retained weighted form term and profiles still depend on the fine state.
These identities therefore do not assert closure on four means or install a
reduced dynamical model.

Reindex each \(B\) region through its carried marked partner in \(A\).
The template phasors then agree at matched phase-bin index \(j\in\mathbb Z_9\).
Set
\[
 \omega=2\pi/9,\quad \zeta=e^{i\omega},\quad
 {\cal M}=\{0,1,2,-1,-2\},\quad
 c_0=\sum_{m\in{\cal M}}\zeta^m
     =1+2\cos(2\pi/9)+2\cos(4\pi/9)
     =1+2\cos(\pi/9),\qquad 2<c_0<3 .
                                                               \tag{5}
\]
The original support supplies exactly those five cross-region phase-bin
offsets. Branch turns can differ by integers; their phasors agree. Define
\[
 \begin{split}
 C_D&=\sum_{j=0}^8\sum_{m\in{\cal M}}\zeta^m
          e^{i(h_{D_B,j+m}-h_{D_A,j})},\\
 C_R&=\sum_{j=0}^8\sum_{m\in{\cal M}}\zeta^m
          e^{i(h_{R_B,j+m}-h_{R_A,j})},\\
 P_A&=9\sum_{j=0}^8 e^{i(h_{R_A,j}-h_{D_A,j})},\qquad
 P_B=9\sum_{j=0}^8 e^{i(h_{R_B,j}-h_{D_B,j})}.
 \end{split}                                                   \tag{6}
\]
Consequently the ordered rectangle \(D_A\to D_B\to R_B\to R_A\to D_A\)
uses \(C_D,P_B,\overline{C_R},\overline{P_A}\).
On the equitable regional reference \(h=0\), these are the positive real
numbers \(9c_0,81,9c_0,81\), for every retained regional mean offset.
This nominal statement does not discard actual source profile errors.

## 3. Description covariance and exact vertex factorization

A change of the regional frame split,
\(\varphi_a\mapsto\varphi_a+\chi_a\) and
\(\eta_i\mapsto\eta_i-\chi_a\) for \(i\in a\), leaves every actual
\(\theta_i\) unchanged. It induces
\[
 C_{ab}\mapsto e^{i(\chi_a-\chi_b)}C_{ab}.
                                                               \tag{7}
\]
Equation (3) is invariant. This is a redundancy of the description, not a
physical phase intervention or a newly established dynamical gauge symmetry.

Only when each required coefficient is nonzero define
\[
 u_{ab}=\frac{C_{ab}}{|C_{ab}|},\qquad
 {\cal H}=u_{D_A D_B}u_{D_B R_B}u_{R_B R_A}u_{R_A D_A}.
                                                               \tag{8}
\]
Reversal conjugates a unit link, and regional frame factors cancel in the
loop product. Missing or zero coefficients give an unavailable chart, not
zero curvature. Magnitudes remain separate observations; (8) does not encode
coupling strengths.

On a connected coarse graph with these reversal rules, the unit links admit
vertex phases \(u_{ab}=e^{i(\alpha_b-\alpha_a)}\) if and only if all independent
cycle products equal one. Necessity follows by telescoping. For sufficiency,
choose one root phase and propagate phases along a spanning tree. Each
remaining edge agrees with its assigned endpoint phases exactly when its
fundamental cycle product is one. The present rectangle has one independent
cycle, so its condition is \({\cal H}=1\). The nominal coefficients in (6)
satisfy this condition exactly.

No independently propagating edge degree of freedom follows from a failure
of vertex factorization: the coefficient map (3) is already a function of
retained fine profiles. A claim about autonomous transport, response memory
or an electromagnetic field needs additional dynamics.

## 4. Centered-profile cancellation and original-source bounds

Write \(s_a=\|h_a\|_2^2\) and \(S=\sum_a s_a\). The elementary remainder
\(|e^{iv}-1-iv|\le v^2/2\) applies to every real lifted difference.
For an internal coefficient, its linear term vanishes:
\[
 \sum_{j,m}\zeta^m(h_{B,j+m}-h_{A,j})
       =c_0\left(\sum_jh_{B,j}-\sum_jh_{A,j}\right)=0 .
                                                               \tag{9}
\]
Both complex cross-block row and column sums are the same real \(c_0\).
Each vertex has five cross neighbors. Bounding each squared difference by
twice the sum of its squared endpoints therefore gives
\[
 \begin{split}
 |C_D-9c_0|&\le5(s_{D_A}+s_{D_B}),&
 |C_R-9c_0|&\le5(s_{R_A}+s_{R_B}),\\
 |P_A-81|&\le9(s_{D_A}+s_{R_A}),&
 |P_B-81|&\le9(s_{D_B}+s_{R_B}).
 \end{split}                                                   \tag{10}
\]
The matching linear term also vanishes by regional centering. No claim that
the internal imaginary remainder is cubic is needed: its quadratic
cross-profile term can be nonzero, as section 6 demonstrates.

The [marked-identity comparison](ORGANIZED_COMPOSITE_RELAXATION.md#mixed-marked-identity)
retains an analytical regional reference driven by the same actual
\(\psi(\tau)\). Its original-source estimate is
\(\sqrt{\|e_x\|^2+4\|e_y\|^2}\le B_{\rm diff}\) for all postattachment ages.
Let \(P_{\rm reg}\) average separately within each of the four regions.
The reference centered phases are regional constants. Hence
\[
 h=(I-P_{\rm reg})e_y,\qquad
 S\le\|e_y\|^2\le S_*:=B_{\rm diff}^2/4 .
                                                               \tag{11}
\]
This uses a contraction of the same admitted error, not a newly chosen
initial profile or an observation of the reference. Its actual phase driver
does not become an independently computable predictor.

Equations (10)--(11) yield retained magnitude intervals
\[
 \begin{split}
 9c_0-5S_*&\le |C_D|,|C_R|\le9c_0+5S_*,\\
 81-9S_*&\le |P_A|,|P_B|\le81+9S_* .
 \end{split}                                                   \tag{12}
\]
The two lower margins must be admitted before normalization. The original
strict marked certificate \(q_{\rm hi}+3B_{\rm diff}/4<\Delta/4\), with
\(q_{\rm hi}\ge0\) and \(\Delta=1/1024\), already implies
\(B_{\rm diff}<\Delta/3\), which is sufficient for their positivity.
Implementation nevertheless rebuilds the consumed source conditions and
these margins rather than trusting an old global verdict.

For \(z\ne0\) and real \(c>0\),
\[
 \left|\frac z{|z|}-1\right|\le\frac{2|z-c|}{c}.
                                                               \tag{13}
\]
For example, add \(z/c\) and use
\(\big||z|-c\big|\le|z-c|\). Telescoping the product of four unit factors
then bounds its difference from one by the sum of their differences.
Each pair of internal regions and each matching partition counts \(S\)
once. Thus the source-conditioned loop allowance is
\[
 |{\cal H}-1|
 \le\left(\frac{10}{9c_0}+\frac29\right)S
 \le\frac79 S_*
 =\frac{7B_{\rm diff}^2}{36}.
                                                               \tag{14}
\]
This is dimensionless and uniform on the original admitted source. It is
an upper bound, not proof that the actual loop equals one or differs from one.
It cannot be assigned the earlier form-output tolerance merely because both
are numbers.

## 5. A same-state description with coefficient phases removed

In the fixed frame (2), retain the actual \(\varphi_a\), \(m_a\), coefficient
magnitudes and weighted form contribution in (4). Define a comparison of
instantaneous regional form rows by replacing \(C_{ab}\) with \(|C_{ab}|\)
on the four contacts. This supplies no independently evolved four-state
model and changes no complete field.

For a coefficient with positive nominal value \(c\),
\(|C-|C||\le2|C-c|\). Every region has one internal and one inter-copy contact.
Its comparison error in mean-form rate therefore satisfies
\[
 |m_a'-\widetilde m_a'|
 \le\frac{2\gamma}{81}
      \left[5(s_a+s_{\text{internal partner}})
           +9(s_a+s_{\text{matching partner}})\right]
 \le\frac{28g}{81}S_*
 =\frac{7gB_{\rm diff}^2}{81},
 \qquad g=1/3000 .
                                                               \tag{15}
\]
The allowance has mean-form-per-structural-time units under the retained
normalization. Multiplying by nine instead gives a region-total rate
allowance. It is not a finite-horizon trajectory error or an observed sensor
resolution, and the earlier \(Q\) error contract remains unchanged.

Under a regional frame rewrite, the comparison coefficients must transform
by (7), along with the mean phases. Replacing them again by positive real
magnitudes in the new frame would define a different comparison. The actual
loop observation is frame invariant; this explicitly registered comparison
has its own transported description.

## 6. Equal means and magnitudes can hide different regional forces

This exact static witness belongs to a larger identity-shaped class. It is
not asserted to be an endpoint of either original acquired preparation.
Keep the same node labels, all unequal form weights, support, matching,
template and clock. Put all forms and all template-relative regional means
equal to zero, and take
\[
 a=\Delta/10000,\quad
 h_{D_A}=a(e_0-e_1),\quad h_{D_B}=a(e_1-e_2),\quad
 h_{R_A}=h_{R_B}=0.                                        \tag{16}
\]
State II swaps only the two donor profile arrays in phase-bin order.
It does not swap nodes or their held conductances.

Let \(T_{ij}=\zeta^{j-i}{\bf1}_{j-i\in{\cal M}}\). Symmetric offset support
gives \(T=T^*\). With \(p_j=e^{ih_{D_A,j}}\), \(q_j=e^{ih_{D_B,j}}\),
the internal coefficient is \(C_D=p^*Tq\), so the profile swap conjugates it.
Both states have
\[
 C_R=9c_0,\qquad P_A=P_B=9(7+2\cos a)>0 .
                                                               \tag{17}
\]
In particular all four coefficient magnitudes coincide, as do the regional
means and three of the four unit links.

Put \(d=2(1-\cos a)>0\). Expanding the four nonconstant products in
\(p^*Tq\) gives the exact expression for state I:
\[
 \begin{split}
 C_D&=9c_0-2c_0d+
       d\left[2e^{i\omega}-e^{i(2\omega-a)}-e^{ia}\right],\\
 \operatorname{Im}C_D
    &=2d\sin\omega\,[1-\cos(\omega-a)]>0 .
 \end{split}                                                   \tag{18}
\]
The sign follows from \(0<a<\omega<\pi\), not a numerical phase estimate.
The coefficient is nonzero, the state-I loop is \(C_D/|C_D|\), and the
state-II loop is its distinct conjugate. The nominal and matching
coefficients remain positive.

Here \(S=4a^2\) and the identity radius of the previous pair chapter is
\(Z^2=\|x_c\|^2+4\|y_c\|^2=16a^2\). Each original marked-gap increment
and cycle-edge perturbation has magnitude at most \(2a\).
Thus \(Z<1/500\), marked gaps remain strictly below \(\Delta/4\), and
\(2a<\pi-4\pi/9\) preserves all four carried windings. Both states have
zero total form and \(\psi=0\). They meet these membership-shaped conditions
without acquiring the original source's history or forecast admission.

All weighted form terms vanish because \(x=0\). The matching regional sine
sums vanish by (17), and the recipient internal sine sum vanishes as well.
The two exact regional mean-form rate vectors differ by
\[
 (m_{D_A}',m_{D_B}',m_{R_A}',m_{R_B}')_{\rm I}
 -(m_{D_A}',m_{D_B}',m_{R_A}',m_{R_B}')_{\rm II}
   =\frac{2\gamma\,\operatorname{Im}C_D}{81}(1,-1,0,0)\ne0 .
                                                               \tag{19}
\]
Their whole-copy exchange derivative \(Q'=-9(m_{D_A}'+m_{D_B}')\) is
zero in both states at that instant. Consequently the witness establishes
information loss for this regional response, not an immediate difference
in \(Q\), a failure of its existing predictor or an acquired-source response.
Coefficient magnitude alone cannot repair the missing contact phase.

## 7. Implementation, evaluated result and scientific boundary

The private [regional-loop reader](../../src/tnfr/physics/_mixed_regional_loop.py)
provides a detached coefficient observation and source-bound assessment. It
retains all four complex coefficients and their magnitudes, admits nonzero
normalization, rebuilds the original source premises and distinguishes the
constructed witness from the admitted source family.

The first frozen evaluation passed all fifteen controls: the twelve
[regional-loop controls](../../tests/physics/test_mixed_regional_loop.py) and
three shared controls, with no skips. Black and full flake8 checks also passed.
The unchanged original form and phase error bounds were each \(10^{-10}\)
for both sources. The result is conditional regional contact information;
the frozen proof, source archive and exact observation record remain in
`artifacts/mixed-regional-loop-20261026/`.

The stored rational certificates give the following conservative strict
decimal ceilings, uniformly over the admitted postattachment history:
\[
 \begin{split}
 B_{\rm diff}&<5.506\,10^{-7},&
 S&<7.579\,10^{-14},\\
 |{\cal H}-1|&<5.895\,10^{-14},&
 \max_a|m_a'-\widetilde m_a'|&<8.733\,10^{-18}.
 \end{split}
\]
The loop allowance is dimensionless. The last allowance is regional
mean form per structural \(\tau\), not a trajectory or sensor-error bound.
Neither exact factorization nor a nontrivial actual-source loop was certified.

For the fixed constructed witness, the positive imaginary coefficient is
approximately \(2.8683413091893663\,10^{-15}\), and the regional mean-rate
separation is approximately \(2.2036889701418784\,10^{-20}\) form per
\(\tau\). These printed values summarize the retained interval bounds and
establish no experimental detectability. The exact matched-magnitude
information-loss statement remains the larger-class result (16)--(19).
No source coefficients were measured, no trajectory was integrated and no
original acquisition producer was replayed. No numerical loop or rate
tolerance was selected.

The actual coarse coefficient can retain internal profile information even
though every microscopic phase difference is vertex-derived. Summing different
contacts is not the same operation as multiplying microscopic links along
one path. A nonfactorizable coarse product therefore need not be an
independent field or a new primitive degree of freedom. Conversely, the
source upper bound does not establish exact factorization.

This result connects organized identity to an explicit interaction
observation. It supplies neither Maxwell dynamics nor a flux constraint,
transverse radiation, electric charge, quantum detection or empirical
identification. The [electromagnetic audit](../PHYSICAL_REGIME_CORRESPONDENCES.md#electromagnetic-emergence-audit)
owns those separate obligations. The research program alone selects what
additional scientific question, if any, follows this gate.

<a id="mixed-cycle-current"></a>
## 8. Complete-law continuity and carried circulation

The next question uses the same law (1), node order, held conductances,
capacities, source history and clock. The prospective scope is
`artifacts/mixed-cycle-current-20261026/scope.json`.
It asks whether one oriented current reading per original nine-cycle adds
enough information to predict those readings' derivatives from
\(K=(m_a,\varphi_a,\Gamma_a)_{a=1}^4\). This section derives identities on
the complete state and a larger-class obstruction; it does not measure a
new acquired source or introduce an external input law.

### 8.1 The constitutive edge current and its balances

For an edge oriented from \(i\) to \(j\), define outward form and phase-lift
currents directly from the two pairwise contributions in (1):
\[
 J_{ij}=\frac{w_{ij}(x_i-x_j)+
                  \gamma\sigma_{ij}\sin(\theta_i-\theta_j)}9,\qquad
 P_{ij}=-\frac{\gamma w_{ij}(x_i-x_j)}9 .                 \tag{20}
\]
Here \(w\) is the actual held form conductance: the existing unequal weights
inside each copy and nine on the matching edges. The sine coefficient
\(\sigma\) is one inside a copy and nine on matching edges. Both arrays are
symmetric; consequently \(J_{ji}=-J_{ij}\) and \(P_{ji}=-P_{ij}\). Thus
\[
 x_i'=-\sum_jJ_{ij},\qquad
 \theta_i'=-\sum_jP_{ij},\qquad
 \left(\sum_i x_i\right)'=\left(\sum_i\theta_i\right)'=0. \tag{21}
\]
The second inventory uses the admitted continuous lifts. It is not a
representative-independent sum of circular angles, a particle number or an
identified physical charge. Conservation on the closed support follows
from pairwise cancellation, independently of whether the state is nominal.
For any fixed region \(a\), cancellation of its internal edges also gives
\[
 m_a'=-\frac19\sum_{i\in a,j\notin a}J_{ij},\qquad
 \varphi_a'=-\frac19\sum_{i\in a,j\notin a}P_{ij}.        \tag{22}
\]
These agree with (4); the carried template is time independent.

Equation (20) fixes the current by the actual edge terms of the complete
law. Continuity by itself does not select that decomposition: a
divergence-free edge array could be added without changing (21). Such an
addition would change the chosen observable and would need its own
constitutive justification. No extra current is added here.

Differentiating the same fixed-weight expression gives
\[
 J_{ij}'=\frac{w_{ij}(x_i'-x_j')+
     \gamma\sigma_{ij}\cos(\theta_i-\theta_j)
                      (\theta_i'-\theta_j')}9 .        \tag{23}
\]
All rates on the right come from (1). There is no finite-difference
measurement, fitted derivative or held-phase substitution.

### 8.2 What the four carried cycle readings retain

Orient each original cycle through its physical node order
\(j\longrightarrow j+1\pmod9\), without phase-bin reindexing.
All nine original edges on each cycle have \(w=\sigma=1\). Therefore
\[
 \Gamma_a=\frac19\sum_{(i,j)\in C_a}J_{ij}
          =\frac{\gamma}{81}
                     \sum_{(i,j)\in C_a}\sin(\theta_i-\theta_j).
                                                               \tag{24}
\]
The form-gradient contribution telescopes exactly. This cancellation uses
the original unit cycle weights; it is not valid for an arbitrary weighted
cycle or the whole contact support. \(\Gamma_a\) has form per structural
time units under (1), and is distinct from the runtime's external Gamma
input. Reversing a declared cycle reverses both its reading and its derivative.

Embed the cycle's phase sensitivity in a 36-vector:
\[
 (g_a)_i=
 \cos(\theta_i-\theta_{\operatorname{next}(i)})
 -\cos(\theta_{\operatorname{prev}(i)}-\theta_i)
 \quad(i\in a),\qquad (g_a)_i=0\quad(i\notin a).
                                                               \tag{25}
\]
Its sum in every region is zero. Equations (1) and (24) give
\[
 \Gamma_a'=\frac{\gamma}{81}g_a^T\theta'
           =\frac{\gamma^2}{729}g_a^TL_+x .              \tag{26}
\]
The instantaneous circulation depends only on the phase profile, whereas
its derivative also consumes the internal form profile through the actual
held Laplacian. The four readings do not retain those profiles.

The structural clock matters in both equations. With
\(\tau=\alpha t\), \(\alpha=1023/1024\), the corresponding \(t\)-current
and circulation are \(\alpha J\) and \(\alpha\Gamma\).
Their \(t\)-derivatives are \(\alpha^2J'\) and \(\alpha^2\Gamma'\).
Changing the clock of only one row would define a different model.

### 8.3 Stationary circulation is compatible with zero storage loss

At the aligned stationary template \(x=0,\theta=(\Theta,\Theta)\), the
symmetrically paired sine terms cancel at every node; matching terms are
zero. Thus both complete rows, both divergences and all four \(\Gamma_a'\)
vanish. The original windings \(q_a=(1,2,1,2)\) nevertheless give
\[
 \Gamma_a^0=-\frac{\gamma}{9}\sin(2\pi q_a/9)\ne0.       \tag{27}
\]
Every \(g_a\) is zero there because the consecutive edge cosines agree.
This is an exact template statement for every admitted held-weight member,
not a claim that an acquired source reaches the template at finite time.

The existing storage identity remains
\(\mathcal E_+'=-\|L_+x\|^2/9\). In particular its loss is zero at the
template. A nonzero cycle component of (20) is compatible with that fact:
its divergence is zero and it changes no node's form. Multiplying that
reading by time does not establish extractable work, radiation, an
independent field or a trajectory of material constituents. The current
is a specified edge decomposition of the complete nodal law.

### 8.4 Exact equal-K states with different circulation derivatives

Use the same fixed support, weights and template in both states. Put
\[
 a=\Delta/10000,\qquad
 h_{D_A}=a(e_0-e_1),\qquad h_{D_B}=h_{R_A}=h_{R_B}=0,
 \qquad\theta=(\Theta,\Theta)+h .                        \tag{28}
\]
The indices are physical cycle indices. Let \(g=g_{D_A}\) in (25) at this
one phase state, and compare
\[
 x^{\mathrm I}=0,\qquad x^{\mathrm {II}}=a g .           \tag{29}
\]
All regional mean forms are zero in both states because each regional sum
of \(g\) is zero. All mean phases and all four circulations coincide
exactly: the phases are identical and (24) is independent of form.
The fixed retained chart \(K\) is therefore identical, with no permutation
of nodes, change of weights or altered phase frame.

To see \(g\ne0\), write \(\omega=2\pi/9\). The forward phase differences
on the donor A cycle are \(\omega-2a,\omega+a,\omega,\ldots,\omega+a\),
modulo the carried closing branch. At node 2,
\[
 g_2=\cos\omega-\cos(\omega+a)>0,                        \tag{30}
\]
since \(0<\omega<\omega+a<\pi\). The complete held Laplacian is connected
with positive conductances, so
\[
 g^TL_+g=\sum_{\{i,j\}}w_{ij}(g_i-g_j)^2>0 .
\]
Indeed matching alone contributes \(9\|g\|^2\), because \(g\) is zero on
the recipient. Equation (26) now proves
\[
 (\Gamma_{D_A}')^{\mathrm {II}}
 -(\Gamma_{D_A}')^{\mathrm I}
       =\frac{a\gamma^2}{729}g^TL_+g>0 .                 \tag{31}
\]
The other three cycle sensitivities are zero at the chosen phase state,
so their derivative differences are zero. There cannot be a single
instantaneous function of \(K\) reproducing these four derivatives on a
domain containing both states. This conclusion is uniform over the admitted
fixed held-weight family, not dependent on choosing a special conductance.

The witness lies within the previously used identity-shaped neighborhood.
The cycle incidence norm is at most two; cosine is 1-Lipschitz.
Subtracting the constant nominal edge cosine before applying incidence
therefore gives
\[
 \|g\|\le4\|h\|=4\sqrt2a,\qquad
 Z^2=\|x_c\|^2+4\|y_c\|^2
       \le32a^4+8a^2<(1/500)^2 .                        \tag{32}
\]
Both profiles have zero regional and whole-copy means, so centering makes
no change here. Marked phase gaps change by at most \(a<\Delta/4\);
carried cycle-edge increments change by at most \(2a<\pi-4\pi/9\).
Thus the marked collars and all four windings are preserved.
These membership inequalities do not supply the original acquired source
history, initialization or response certificate. No original forecast or
source-specific approximation has been contradicted.

The missing contraction in (26) is explicit. Its appearance does not
automatically make it a sufficient additional state coordinate: its own
evolution consumes the changing phase sensitivity and form profile.
Exact closure, controlled approximation and retained memory remain distinct
questions. This result identifies information lost by a specified chart,
not a universal minimum number of TNFR parameters.

### 8.5 Implementation and evaluation boundary

The detached [current reader](../../src/tnfr/physics/_mixed_cycle_current.py)
admits supplied forms, continuous phase lifts and held conductances before
arithmetic, then reuses the existing complete-law field for its rates.
It observes edge currents, regional balances, cycle readings and derivatives.
It does not authenticate an acquired source, advance a state, execute an
operator or install a new evolution law.

The first frozen evaluation passed all twelve controls: ten
[current controls](../../tests/physics/test_mixed_cycle_current.py), the shared
complete-field comparison and the artifact writer's exclusive-creation
control. No cases were skipped. Black and full flake8 checks passed.
The independent 220-digit oracle covered all 180 edge currents and all
72 complete rows, both continuity balances, current orientation and clock,
the chain-rule derivative, the stationary template and the exact witness.
The retained outward intervals have strictly negative nominal circulations
and a strictly positive donor-A derivative separation for the witness.
These are numerical checks of the conditional mathematics, with no physical
resolution or detectability claim.

The witness's interval inputs enclose the single correlated construction
defined in (28)--(29). Equality of K follows from that exact construction,
not from treating every independent member of the interval boxes as the
same-K state. The all-admitted-weights statement follows from the positive
quadratic form in (31), rather than extrapolation from one numerical matrix.
The frozen scope, reviewed source archive, exact observation record and
control logs remain in `artifacts/mixed-cycle-current-20261026/`.

No acquired source state was evaluated, no trajectory was integrated and no
original producer was replayed. The original source, response guarantees and
paid-work accounting are unchanged. The exact obstruction (28)--(32) and its
implementation are separate from an empirical comparison. No electrical,
weak, strong or gravitational current has been identified. The research
program selects the subsequent question.

<a id="mixed-local-modes"></a>
## 9. Local temporal modes without a commuting-matrix assumption

The aligned template in section 8 has stationary nodal state and nonzero
constitutive cycle currents. This does not determine its response to a
perturbation. We now classify the temporal tangent of the same full law for
the whole admitted held-conductance family. The prospective scope is
`artifacts/mixed-local-modes-20261026/scope.json`.
The template is an explicit equilibrium, not an acquired finite-time source
state. No source, conductance, coefficient, capacity, preparation or clock
is changed.

### 9.1 Actual tangent, common coordinates and positive quotient

Write \(L=L_+\) and let \(K_*\) be the phase Hessian of the actual potential
at \((\Theta,\Theta)\). It is the symmetric graph Laplacian with edge
coefficient \(\sigma_{ij}\cos(\Theta_i-\Theta_j)\): the internal bare sine
coefficient is one, and the matching coefficient is nine. It is not the
held form Laplacian and does not inherit its unequal conductances.
For tangent form \(u\) and tangent phase \(y\), the complete derivative is
\[
 \begin{pmatrix}u'\\y'\end{pmatrix}
 =A\begin{pmatrix}u\\y\end{pmatrix},\qquad
 A=\frac19\begin{pmatrix}-L&-\gamma K_*\\
                          \gamma L&0\end{pmatrix}.       \tag{33}
\]
The existing full nonlinear field supplies this derivative. No projected
Q/phase model replaces any of its seventy-two rows.

Both Laplacians annihilate the common vector \(\mathbf1\). Consequently
the mean form and mean lifted phase are two independent conserved
coordinates, with a separate zero \(2\times2\) block. Their orthogonal
complement is \(V\times V\), where \(V=\mathbf1^\perp\subset\mathbb R^{36}\)
has dimension 35. Both subspaces are invariant; there is no coupling from
a common coordinate into the centered block. Inversion of \(L\) below
means inversion only on \(V\).

The already derived bare support has gap at least four within each copy.
Every actual internal form conductance is at least \(3/4\); the original
cycle conductances remain one. Thus each held copy has form gap at least
three. Its internal weighted degree is at most nine.
Internal template phase gaps are \(0,\pm2\pi/9,\pm4\pi/9\) modulo full
turns, and
\[
 \cos(4\pi/9)=\sin(\pi/18)>1/9 .
\]
The last strict inequality follows from strict concavity of sine between
zero and \(\pi/2\), using the chord joining their values.
Hence each copy's Hessian has gap at least \(4/9\) and degree at most nine.

Matching contributes \(9G\) to both matrices. Decompose any centered
36-vector into the two within-copy zero-mean profiles and the opposite
constant copy means. These subspaces are invariant under \(G\), while
the internal Laplacians vanish on the constant means. Matching has
eigenvalue 18 on the opposite-mean direction and is nonnegative on the
profile subspace. The combined degree is at most 18. Therefore, on \(V\),
\[
 3I\preceq L\preceq36I,\qquad
 \frac49I\preceq K_*\preceq36I,\qquad
 \frac1{36}I\preceq L^{-1}\preceq\frac13I .              \tag{34}
\]
The upper bounds use the symmetric-Laplacian bound of twice the maximum
weighted degree. These estimates hold for every admitted internal weight
member; no joint eigenbasis or nominal conductance equality is used.

### 9.2 A scalar discriminant for every possible eigenvector

On \(V\), the change of state from \((u,y)\) to
\((y,y')=(y,\gamma Lu/9)\) is invertible. Eliminating \(u\) gives the
second-order equation already identified in the physical audit:
\[
 y''+\frac L9y'+\frac{\gamma^2}{81}LK_*y=0,\qquad
 P(z)=z^2L^{-1}+\frac z9I+\frac{\gamma^2}{81}K_* .        \tag{35}
\]
A temporal eigenvalue \(z\) of the centered complete tangent has a
nonzero, possibly complex phase eigenvector \(v\) with \(P(z)v=0\).
Normalize \(v^*v=1\), and put
\[
 a=v^*L^{-1}v,\quad c=v^*K_*v,\qquad
 a z^2+\frac z9+\frac{\gamma^2c}{81}=0 .                 \tag{36}
\]
The real symmetric positive matrices make \(a,c\) positive real numbers
even for complex \(v\). Equation (34) bounds
\(1/36\le a\le1/3\) and \(4/9\le c\le36\). Its discriminant has the
strict uniform lower bound
\[
 \frac{1-4\gamma^2ac}{81}\ge\frac{1-48\gamma^2}{81}
 >\frac{\delta_0}{81}>0,\qquad
 \delta_0=1-\frac{48}{3000^2}=1-\frac1{187500}.          \tag{37}
\]
Here the unchanged \(\gamma=1/(1023\pi)<1/3000\).
Thus every possible temporal eigenvalue is real. Positive coefficients,
negative root sum and positive root product make both roots negative.
This argument tests an arbitrary eigenvector of the full noncommuting
problem; it does not substitute eigenvalues of \(L\) and \(K_*\) into a
common-mode formula.

### 9.3 Two separated branches and no Jordan chains

At the same separator \(s=-1/6\) for the whole weight family, (34) gives
\[
 P(s)=\frac1{36}L^{-1}-\frac1{54}I+
                       \frac{\gamma^2}{81}K_*
 \preceq-\frac{1-48\gamma^2}{108}I
 \prec-\frac{\delta_0}{108}I .                          \tag{38}
\]
For each normalized vector \(v\), its scalar quadratic in (36) therefore
has one root below \(s\) and one above \(s\). At any actual eigenvalue,
the derivative of that scalar quadratic has a strict sign depending only
on the branch:
\[
 v^*P'(z)v=2az+\frac19
 \begin{cases}<0,&z<s,\\>0,&s<z<0.\end{cases}             \tag{39}
\]
In particular this quadratic form is definite on the kernel of \(P(z)\).
If the corresponding first-order companion matrix had a Jordan chain,
its phase coordinates would satisfy
\(P(z)v_1+P'(z)v_0=0\) with \(0\ne v_0\in\ker P(z)\).
Multiplying by \(v_0^*\), using the Hermitian \(P(z)\), contradicts (39).
The invertible change in (35) transfers this conclusion to the actual
centered tangent: every eigenvalue is semisimple, although different modes
may have the same eigenvalue.

Moreover, on \(s<z<0\),
\(P'(z)=2zL^{-1}+I/9\succ0\).
The 35 ordered eigenvalues of the Hermitian pencil are continuous and
strictly increasing there. They start negative at \(s\) and end positive
at zero, so they cross zero exactly 35 times counting multiplicity.
The centered characteristic polynomial has degree 70, all its roots are
real negative, and neither \(s\) nor zero is a root. The other 35 roots
therefore lie below \(s\). From the scalar root formula and \(a\ge1/36\),
the fast root obeys
\[
 z_-=\frac{-1-\sqrt{1-4\gamma^2ac}}{18a}>
                  -\frac1{9a}\ge-4 .
\]
The complete local spectrum, counting multiplicity, is consequently
\[
 \begin{array}{c|c|c}
 \text{sector}&\text{number}&\text{temporal rate in structural }\tau\\ \hline
 \text{common form and common phase}&2&0\\
 \text{fast centered modes}&35&-4<z<-1/6\\
 \text{slow centered modes}&35&-1/6<z<0
 \end{array} .                                         \tag{40}
\]
The common zero block is semisimple as well. Thus the linearized solution
is a sum of decaying real modes plus its two conserved common components.
The result covers the entire admitted weight family without evaluating
a numerical eigenspectrum.

### 9.4 Observation and physical scope

Equation (40) classifies infinitesimal temporal modes near this aligned
equilibrium. Individual coordinates and mixtures of modes can still show
overshoot, sign changes or transient amplification in a chosen norm.
The conclusion does not assert coordinatewise monotonicity, rule out
transient transport, or classify finite nonlinear perturbations elsewhere.
Nor does it establish a spatial continuum, a wavevector, a signal velocity
or an electromagnetic, weak, strong or gravitational interpretation.

The cycle observation has an additional limitation. Equation (25) gives
\(g_a=0\) at the template, and (24) has no form dependence. Therefore
\[
 D\Gamma_a\big|_{(0,\Theta,\Theta)}(u,y)=0
 \quad\hbox{for every tangent direction }(u,y).          \tag{41}
\]
Its nonzero stationary value is not a first-order readout of any centered
mode. A finite change of circulation begins beyond first order, while
the nodal form and phase coordinates retain their own linear responses.
The prior equal-K obstruction concerns nonuniform phase profiles and is
consistent with this zero template sensitivity.

All rates in (40) use \(\tau\). For \(\tau=(1023/1024)t\), the complete
\(t\)-generator and each temporal rate are multiplied by \(1023/1024\);
both rows must transform together. These are structural rates, with no
laboratory time calibration. Neither exact source arrival at the template
nor a new trajectory or new acquisition certificate has been established.

The mathematical reason for the result is visible in (37): the retained
dissipation dominates the reciprocal form/phase coupling uniformly on the
admitted quotient. This locates a restriction of the specified complete
law. It does not make dissipation or this coupling ratio compulsory for
every TNFR realization; changes to a constitutive premise would require
their own derivation, identity and observation analysis.

### 9.5 Implementation and evaluation boundary

The [local-mode helper](../../src/tnfr/physics/_mixed_local_modes.py)
reuses the admitted complete field and shared first-order Taylor jets for
tangent actions, and the current reader's exact support association for
the aligned static template. Its certificate reports uniform inequalities
and branch counts, not numerically estimated eigenvalues or acquired-state
provenance. No new evolution law is installed.

The first frozen evaluation passed all eleven controls: nine
[local-mode controls](../../tests/physics/test_mixed_local_modes.py), the
shared complete-current comparison and artifact exclusive-creation control.
No cases were skipped; Black and full flake8 checks passed. All 72 columns
of the genuine field's first-order derivative were checked against an
independent sparse Laplacian/Hessian oracle at 220 decimal digits, with
outward interval inclusion. The checks also cover both common coordinates,
storage, an explicitly noncommuting legal weight member, the complex
Rayleigh identity, the Jordan-chain obstruction and zero linear cycle
sensitivity. Constructed legal weights are not authenticated source weights.

The stored certificate retains the exact margin 187499/187500 and the
negative separator bound -187499/20250000. Branch counts and intervals follow
the proof (34)--(40), not a numerical eigenspectrum; unavailable sufficient
premises withhold those conclusions. The conditional theorem covering all
admitted weights and finite implementation controls have distinct roles.
The scope, reviewed source capture and exact observation record remain in
`artifacts/mixed-local-modes-20261026/`.

No temporal eigenvalues were numerically evaluated, no trajectory was
integrated and no original source producer was replayed or reassessed.
The original acquisition, identity, work and prediction evidence is
unchanged. The research program alone selects the next scientific question.
