# Finite collective coupling across two registered interfaces

<a id="organized-interface-factorization"></a>

<a id="prospective-interface-declaration"></a>
## Prospective interface and decision declaration

This admission tests a purely multiplicative real source/probe factorization
of a finite response. It uses the actual acquired and observed sources of the
[joint-resource theorem](ORGANIZED_JOINT_RESOURCES.md), whose resource
declaration was frozen at `0ca8630d`. The second interface below changes
the subsequent complete support. Its response is not supplied by the
same-slot certificate, an onset coefficient or a winding label.

| Item | Fixed choice and scope |
| --- | --- |
| Sources | The paired acquired outer-zero and outer-one host families after outer formation and the entire sequential observation word, and one independent recovered C9 donor. Retain all original source errors and actual reached states |
| Resource box | The admitted box from `0ca8630d`: twelve primitive component allowances in `[0,1e-6]`, quarter formation 200, quarter recovery `8e10`, handoff Euclidean allowance `q=1e-8`, decay power 80, and sequential raw errors at most `1e-14` |
| Relative preparation | Same actually prepared donor member in all four counterfactual contacts, zero common donor form and donor phase origin `beta=pi/2`; both actual host means remain those of the carried history |
| Node order | Host node `9*c+j`, `c=0,...,4`, `j=0,...,8`; donor node `45+j`. All original 99 internal and host-outer edges remain unchanged |
| Same-slot interface | For every c and j, add the unit edge `(9*c+j,45+j)` |
| Registered interface | For every c and j, add the unit edge `(9*c+j,45+((j+k_c) mod 9))`, with signed representative `k_c=2*(c-2)`, hence offsets `(-4,-2,0,2,4)` or residues `(5,7,0,2,4)` |
| Independent normalization | Each interface has 45 unit contacts, total contact conductance 45, host degree 5, donor degree 7, the same donor/readout normalization and the same clock. Neither interface depends on unknown source errors or readings |
| Complete law | Each actual graph uses `x_tau=(-L*x+gamma*S(theta))/4`, `theta_tau=gamma*L*x/4`, `gamma=1/(1023*pi)`, held unit capacities, smooth full-circle sine and no continuous input or later event |
| Clock and observation | `tau=(1023/1024)*t` in both rows; donor mean signed-form increment from 0 to `h=1e-3`. Rows of the response matrix are outer zero, outer one; columns are same-slot, registered |
| Reading model | Two raw readings per matrix entry, hence eight raw errors altogether, each arbitrary and bounded by `eta=1e-9`; retain correlations and the initial-reading error even though the true initial donor mean is zero |
| Source quantifiers | Choose one actual paired host preparation and one actual donor member, then compare its four counterfactual contacts. A row uses the same precontact host member in both columns. No physical copying or repeatable exact preparation is asserted |
| Hypothesis | `R(s,p)=a(s)*b(p)` for one real source factor and one real probe-only factor, with the donor, horizon, readout and clock fixed. Source factors may depend on the full actual host state. The hypothesis includes no additive baseline or arbitrary nonlinear dependence on a scalar |
| Decisions | Obtain outward true-entry intervals T, observed-entry intervals `T+[-2*eta,2*eta]`, and uniform error-correction envelopes `T+[-4*eta,4*eta]`. Require the true and error-correction determinant intervals to exclude zero; threshold exactly zero |
| Work and identity | Re-admit new contact work below 100, joined present storage below 200 for each source/interface, and all six internal and nine old outer windings throughout the contact. Preserve all earlier preparation and observation payments separately |
| Arithmetic and budget | One assessment at the maximum resource-box corner, exact rational primitives and existing outward dyadic128 arithmetic. No registration or origin scan, trajectory, frozen producer, precision change or replacement candidate |
| Incomplete result | An unresolved determinant or missing source/work/identity premise leaves this test unavailable. It does not prove factorization or authorize more interface candidates |

Commit this declaration and its consuming source before the single assessment.
The registration is fixed by the marked child index. Adding an integer turn
to a slot representative changes no circular field, while the actual stored
lift and its conserved mean are not reset. Earlier protocols and evaluated
responses remain unchanged.

## 1. Complete support, normalization and state

Let \(p=0,1\) index the same-slot and registered interfaces, and define

\[
 k_c^{(0)}=0,\qquad k_c^{(1)}=2(c-2),\qquad
 \sigma_{p,c}(j)=j+k_c^{(p)}\pmod9.                       \tag{1}
\]

For each c this is a permutation of all nine donor slots. Thus every host
node gains exactly one contact and every donor node gains exactly five.
The complete graph is undirected, connected and has 54 nodes and 144 unit
edges, with degrees five and seven. No contact is repeated. The internal
C9 supports and the original outer C5 slot cycles are identical between
the two contacts; the mixed contact cycles can differ.

All signed forms and primitive phases remain in the state. With \(L_p\)
the actual graph Laplacian and \(S_p\) its complete sine current, the rows
and storage are

\[
 \begin{split}
 x'&=(-L_px+\gamma S_p(\theta))/4,&
 \theta'&=\gamma L_px/4,\\
 H_p&=\tfrac12x^TL_px+\sum_{e\in E_p}(1-\cos\delta_e),&
 H_p'&=-\|L_px\|^2/4.
 \end{split}                                              \tag{2}
\]

Both global form and lifted phase sums are conserved. At contact they
are zero and \(9\beta\), respectively. The uniform quarter factor is
unchanged in both rows; it is not inverse degree. The two contact graphs
define different subsequent full vector fields. Equal degrees or an
abstract graph relabeling do not establish equality at the same retained
source and marked readout. The isolated origin symmetry does not erase
relative registrations after attachment.

For an edge from host \((c,j)\) to donor \(\sigma_{p,c}(j)\), write

\[
 u_{c,j}=x_{D,\sigma_{p,c}(j)}-x_{H,c,j},\quad
 \delta_{c,j}=\theta_{D,\sigma_{p,c}(j)}-\theta_{H,c,j},
 \quad f_{c,j}=(u_{c,j}+\gamma\sin\delta_{c,j})/4,
 \quad g_{c,j}=-\gamma u_{c,j}/4.                         \tag{3}
\]

The host receives \((f,g)\) and the contacted donor node receives
\((-f,-g)\). These are both boundary channels, including backreaction.
Their edge storage is \(u^2/2+1-\cos\delta\); its derivative plus
both regional derivatives gives (2). No auxiliary scalar forcing or
one-sided donor waveform replaces these reciprocal rows.

## 2. Permutation matching preserves the exact mean equation

Set \(R_D=(1/9)\sum_jx_{D,j}\), and

\[
 \mathcal A_p=\frac19\sum_{c,j}\sin\delta_{c,j}.
 \tag{4}
\]

Every internal current cancels in the donor mean. Bijection of each
\(\sigma_{p,c}\) gives
\(\sum_{c,j}x_{D,\sigma_{p,c}(j)}=45R_D\).
The host form total is \(-9R_D\), from the conserved zero whole form.
Summing (3) therefore gives, for arbitrary admitted fine states,

\[
 R_D'=-3R_D/2-\gamma\mathcal A_p/4,\qquad
 \Psi_D'=3\gamma R_D/2,\qquad R_D(0)=0.                  \tag{5}
\]

Here \(\Psi_D\) is the donor's actual lifted phase mean. The matching
permutations preserve the exact form cancellation; an arbitrary contact
weight profile would not follow from this argument. The nonlinear moment
\(\mathcal A_p\) still consumes the full boundary phase state.

Let \(X=1/100\), \(T_o=1/5000\), \(q=10^{-8}\),
\(\gamma_+\ge\gamma\), and

\[
 B_o=X+\gamma_+T_o,\qquad
 B=\max\{B_o,q\}+7\gamma_+h/4.                          \tag{6}
\]

The carried host has \(\|x_H(0)\|_\infty\le B_o\), and the
donor has \(\|x_D(0)\|_\infty\le q\). The graph heat semigroup
contracts the maximum norm and \(\|S_p\|_\infty\le7\), so
\(\|x(t)\|_\infty\le B\) up to h for either interface.
The phase row bounds host and donor rates by \(5\gamma_+B/2\)
and \(7\gamma_+B/2\). Hence

\[
 |\mathcal A_p(t)-\mathcal A_p(0)|\le30\gamma_+Bt.
\]

Since \(|\mathcal A_p|\le5\), the nonnegative damping kernel in
(5) gives \(|R_D(t)|\le5\gamma_+t/4\). Integrating the complete
mean equation, including damping and phase motion, yields

\[
 \left|R_D(h)-R_D(0)+\frac{\gamma h}4\mathcal A_p(0)\right|
 \le E:=\frac{15}{16}\gamma_+h^2+
                  \frac{15}{4}\gamma_+^2Bh^2.            \tag{7}
\]

Thus the previous form of the finite remainder is justified on each new
graph by its own permutation, conservation and degree premises. It is not
transferred merely because the contact counts agree.

## 3. Actual source correlations and finite matrix entries

The [joint-resource proof](ORGANIZED_JOINT_RESOURCES.md#organized-joint-resources)
rebuilds all six original sources, their uninterrupted recovery, actual
outer acquisition and the complete sequential word. Let \(Z\) be its
full host phase error about the acquired carrier and let \(P_{w,c}\)
be the actual carrier phase enclosure at outer time 200. For the outer-zero
control \(P_{0,c}=\{0\}\); for outer one these are the finite reached
intervals, not its limiting twist. The word adds coordinate drift at most
\(m_o=2\gamma_+B_oT_o\). Set \(e=Z+m_o+q\).

Write \(\psi_j=\alpha(j-4)\), \(\alpha=2\pi/9\). The actual
host offsets belong to \(P_{w,c}+[-Z-m_o,Z+m_o]\), while the donor
has offsets \(\beta+z_{D,j}\), \(|z_{D,j}|\le q\).
For an integer \(n_{p,c,j}\),

\[
 \psi_{\sigma_{p,c}(j)}-\psi_j
       =\alpha k_c^{(p)}-2\pi n_{p,c,j}.                  \tag{8}
\]

The turn term drops out of sine and cosine exactly. No actual lifted
coordinate or phase total is changed by using the smaller signed k in the
trigonometric argument. At the supplied \(\beta=\pi/2\),

\[
 \mathcal A_{w,p}(0)\in C_{w,p}:=
       \sum_{c=0}^4\cos\bigl(P_{w,c}-\alpha k_c^{(p)}+[-e,e]\bigr).
 \tag{9}
\]

This is a source-family enclosure of the actual slot correlations. It does
not assume that different nodes' errors are independent, that interval
corners are attainable, or that child means close the fine law.
Using the same source member for two contacts introduces correlations
between columns; enclosing them independently only enlarges the possible
response set.

Let \([\gamma]\) be its shared outward interval. The true finite response
matrix \(R\), with rows \(w=0,1\) and columns \(p=0,1\), satisfies

\[
 R_{w,p}=R_{D,w,p}(h)-R_{D,w,p}(0)\in
 T_{w,p}:=-\frac{h[\gamma]}4C_{w,p}+[-E,E].              \tag{10}
\]

All four entries describe different complete continuations of retained
precontact states. The same actual donor member is fixed in all four.
The paired host members retain their shared primitive preparation errors;
their distinct reached states are never identified with one another.

## 4. Work, present storage and fifteen finite identities

The support event leaves all coordinates unchanged. Each of its 45 new
edges has form difference at most \(B_o+q\) and phase cost at most two,
so its actual supplied work obeys

\[
 0\le W_p\le W_+:=90+\frac{45}2(B_o+q)^2.               \tag{11}
\]

The bound applies to each permutation without asserting equal actual works.
With \(V_n=n[1-\cos(2\pi/n)]\) and the freshly reconstructed outer-one
endpoint excess U, the two present-storage upper bounds are

\[
 \begin{split}
 H_{1,p}(0)&\le6V_9+9V_5+U+36+4q^2+W_+,\\
 H_{0,p}(0)&\le6V_9+36+44q^2+W_+.
 \end{split}                                             \tag{12}
\]

These retain continuous loss and the observation word's cumulative absolute
work bound. Six original preparation costs, earlier outer attachment and
all observation payments remain separate historical premises. The four
counterfactual accounts are not one autonomously executed four-contact path.

Set \(D_o=2m_o\), \(D_j=7\gamma_+Bh\), \(\zeta=10^{-6}\),
and \(A_o=7/5\). Require \(W_+<100\), both bounds (12) below
200, and the five strict margins

\[
 \begin{gathered}
 \alpha-2\zeta-D_o-D_j>0,\qquad
 \pi/2-\alpha-2\zeta-D_o-D_j>0,\\
 \pi/2-A_o-2\zeta-D_o-D_j>0,\qquad
 \alpha-2q-D_j>0,\qquad\pi/2-\alpha-2q-D_j>0.
 \end{gathered}                                           \tag{13}
\]

The old internal and outer edges are unchanged, and their phase rates have
the same degree bounds. Thus (13) preserves all six internal winding-one
cycles and all nine old outer windings throughout either contact. No new
mixed contact loop has an assigned winding. Neither old product capture nor
same-slot control maintenance is asserted on the registered support beyond h.

## 5. Scalar factorization, noise and admissible correction

Fix one actual paired host preparation and one donor member. If real
functions \(a(s)\) and \(b(p)\) represent the four true responses by
\(R(s,p)=a(s)b(p)\), then

\[
 \det R=R_{0,0}R_{1,1}-R_{0,1}R_{1,0}=0.                \tag{14}
\]

The source factor may depend on every coordinate of the actual host member;
the probe factor may consume the fixed donor and protocol, but cannot depend
on which host is presented. Excluding zero from an outward determinant
interval constructed from (10) contradicts (14). This tests multiplication
of two real factors; a pure complex product also has zero determinant and
is excluded by the same argument. It does not exclude an affine response,
arbitrary nonlinear dependence on one scalar, multiple channels or
memory-dependent descriptions. In particular, a real-part observation of a
complex product need not have rank one:
\(\operatorname{Re}(ab)=\operatorname{Re}(a)\operatorname{Re}(b)
-\operatorname{Im}(a)\operatorname{Im}(b)\) is a sum of two real products.

Let the observed increment matrix be \(Y_{w,p}=R_{w,p}+n_{w,p}\).
Each increment uses two raw errors, so \(|n_{w,p}|\le2\eta\), with
\(\eta=10^{-9}\). All eight raw errors can be correlated. Define

\[
 O_{w,p}=T_{w,p}+[-2\eta,2\eta],\qquad
 K_{w,p}=T_{w,p}+[-4\eta,4\eta].                         \tag{15}
\]

Then \(Y\in O\). For any particular allowed observed record, every
candidate corrected true matrix in its componentwise error box satisfies

\[
 Y_{w,p}+[-2\eta,2\eta]\ \subseteq\ K_{w,p}.             \tag{16}
\]

The extra enlargement in K accounts for possible corrections of a noisy
record; it introduces no additional readings or actual physical errors.
For a matrix of intervals J define its outward determinant interval by
\(\mathcal D(J)=J_{0,0}J_{1,1}-J_{0,1}J_{1,0}\).
The prospective strong decision requires

\[
 0\notin\mathcal D(T),\qquad 0\notin\mathcal D(K).       \tag{17}
\]

The second condition makes the first redundant mathematically, but both
evidence levels are reported. It ensures that every allowed observed record
has no rank-one correction under the declared error model. A nonzero
determinant of one noisy point matrix, by itself, would not reject a true
rank-one response. Likewise, a determinant interval containing zero proves
neither factorization nor existence of a jointly realizable correction.

These uniform statements concern the same-source counterfactuals specified
above. Four independently prepared laboratory records need a separate
preparation/matching admission: different states cannot silently share the
same source factor. The theorem supplies neither physical copies nor record
authentication. Missing law, source, work or identity premises keep the
combined result unavailable even if an algebraic determinant is separated.

## 6. Assessment and scope boundary

Re-admit the maximum corner of the joint resource box before constructing
the four fresh source enclosures. Its source radii and cost bounds are
nondecreasing in primitive allowances. The fixed q bounds every admitted
recovered member; the supplied observation word does not depend on its
reading errors. All downstream intervals therefore enclose the smaller
allowance families as well, while (15) grows with contact reading error.
No assessment of smaller corners or alternative registrations is needed.

The detached proof consumer belongs in
[`_organized_interface_factorization.py`](../../src/tnfr/physics/_organized_interface_factorization.py).
It must rebuild primitive source and complete-support evidence, retain the
true/observed/correction distinction, and preserve explicit unavailable
conditions. Shared arithmetic and field kernels do not establish source
identity, an occurrence law or an independent physical measurement bridge.

A separated determinant would identify a concrete restriction on carrying
one real multiplicative coupling across these normalized probes. It would
not identify electric charge, disprove intrinsic properties generally or
establish that every physical interaction must satisfy (14). An unresolved
test would remain a finite admission result. Neither outcome selects a
unique TNFR law, supplies indefinite outer-one maintenance or initiates a
new hierarchy of closure moments.

<a id="finite-interface-factorization-result"></a>
## 7. Retained finite factorization obstruction

The declaration and complete consumer were committed at `9548b8e1` before
the single assessment. The [attempt](../../docs/assets/organized_interface_factorization/attempt.json)
retains the committed source inventory, declaration and arithmetic. The
[first result](../../docs/assets/organized_interface_factorization/result.json)
is **`finite_scalar_factorization_excluded`**. Both complete supports and
all acquisition, observation, work, storage and finite-identity prerequisites
pass. The true and error-rectification determinant intervals both exclude
zero. No alternative registration, trajectory or second candidate was run.

The following outward decimal summaries enclose the recorded exact-rational
true increments; rows share actual precontact source members, not merely
winding labels:

| Acquired host | Same-slot contact | Registered contact |
| --- | --- | --- |
| Outer zero | `[-3.893e-7, -3.886e-7]` | `[4.109e-8, 4.169e-8]` |
| Outer one | `[5.130e-9, 5.730e-9]` | `[-3.834e-7, -3.827e-7]` |

The larger response belongs to different organizations in the two columns.
This change cannot be explained by multiplying both source responses by
one common probe gain. More precisely, the retained outward bounds satisfy

\[
 \mathcal D(T)\subset[1.4852,1.4902]\,10^{-13},\qquad
 \mathcal D(K)\subset[1.4525,1.5230]\,10^{-13}.
\]

These have squared form units. The second bound includes both the eight
actual raw-reading errors and every correction consistent with their error
model, as defined in (15)--(16). It does not substitute the nonzero rank of
a noisy point matrix for a true-response argument.

The work and present-storage ceilings remain below 90.003 and 169.740,
respectively, for every contact (the outer-zero storage bound is smaller).
All six internal and nine old outer windings survive through `h=1e-3`.
Actual contact work need not agree between interfaces. Each counterfactual
history retains its earlier preparation and observation payments; summing
them does not define an executed four-contact experiment.

Thus the single purely multiplicative source/probe factor is excluded
uniformly over the declared full preparation and noise box. The claim is
stronger than different winding labels having different onsets: the source
factor was allowed to depend on the complete actual host state, and the
decision concerns finite continuations with retained uncertainty and work.
It does not exclude an affine response, a nonlinear scalar encoding, real
observations of complex products, or a richer retained boundary description.
It does not derive physical charge, spin, a quantum state or a unique law.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects further investigation.
