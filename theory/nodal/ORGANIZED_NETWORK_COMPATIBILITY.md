# Complete-law compatibility of three organized C9 constituents

<a id="organized-network-compatibility"></a>

The pairwise sine charts do not generally combine into one chart for a
network. For the three prescribed oriented contacts, an unchanged
normalized reciprocal sine law requires both equal contact amplitudes
and a vanishing phase sum around the triangle. These conditions are
necessary and sufficient in the positive nodewise affine form charts,
constant phase-origin shifts and common clock specified below. Among
the actual co-oriented nonnegative contacts, only the all-zero-offset
control satisfies them.

The complete inherited dynamics still closes exactly on six registered
coordinates for every admitted contact. Retaining the contact amplitudes
and phase registrations gives that law without fitting or additional
evolving variables. The obstruction concerns a more restrictive scalar
network representation; it does not invalidate the supplied fine law,
its storage balance or its conditional organized identity.

The [pairwise owner](ORIENTED_INTERFACE_PROFILE_INHERITANCE.md) supplies
the interface family and its conditional sine-profile selection. This
note fixes sine and tests whether those pair descriptions coexist on
one cycle. The [NFR ontology](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form)
permits retained constituent and boundary information. It does not require
the restrictive chart whose obstruction is proved here.

<a id="organized-network-state"></a>
## Supplied support, complete law and retained identity

There are three marked C9 constituents, with nodes \((r,\ell)\),
\(r=0,1,2\), \(\ell=0,\ldots,8\), ordered as \(9r+\ell\). Each has
its unit-weight internal cycle. Put \(\alpha=2\pi/9\) and orient the
constituent triangle as
\[
 \mathcal E=((0,1),(1,2),(2,0)).
 \tag{1}
\]
For each oriented pair \(e=(r,t)\), fix \(s_e\in[0,1]\) and add
the undirected contacts
\[
 (r,\ell)--(t,\ell),\quad w=1-s_e,\qquad
 (r,\ell)--(t,\ell+1),\quad w=s_e .
 \tag{2}
\]
Indices are modulo nine. Zero-weight edges at the endpoints are absent;
every positive edge acts at both ends with the same weight. Orientation
records which constituent port is shifted, not a directed force. Each
node has cross conductance two and total weighted degree four. There
are 27 internal edges and between 27 and 54 positive contact edges.

The fine state retains signed real form \(x\), primitive circular phase
with specified continuous lifts \(\theta\), held unit capacities, the
marked support and its three fixed weights. Use the inherited structural
clock \(\tau=(1023/1024)t\), a supplied constant \(\gamma>0\), and
the actual weighted fine Laplacian \(L_f\). The complete law is
\[
 \begin{split}
 S_{f,i}(\theta)&=\sum_{m\sim i}w_{im}\sin(\theta_m-\theta_i),\\
 x'&=\tfrac14[-L_fx+\gamma S_f(\theta)],\\
 \theta'&=\tfrac\gamma4 L_fx,\\
 H_f&=\tfrac12x^TL_fx+
                  \sum_{\{i,m\}}w_{im}[1-\cos(\theta_m-\theta_i)].
 \end{split}\tag{3}
\]
Primes denote \(\tau\)-derivatives. In original time both rows acquire
the same factor \(1023/1024\). The form row specifies pressure in the
unit-capacity nodal equation; it does not reconstruct pressure from an
observed derivative. The common degree factor is neither a derived
capacity nor an inverse mass. Structural units and any laboratory clock
conversion remain those of the [parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md).

This is a detached supplied weighted reciprocal law. In particular, its
weighted phase current differs from the native pressure rule's unweighted
unique-neighbor phase channel. No operator word, support selector or
native runtime is installed by (3). Odd exchange, symmetric weights and
the supplied reciprocal row give
\[
 H_f'=-\tfrac14\|L_fx\|_2^2,\qquad
 (\mathbf1^Tx)'=(\mathbf1^T\theta)'=0.
 \tag{4}
\]
These identities concern continuous flow with no additional input.

The candidate registered organized family is
\[
 \mathcal M:\qquad
 x_{r,\ell}=X_r,\qquad
 \theta_{r,\ell}=\Phi_r+\alpha(\ell-4).
 \tag{5}
\]
Each constituent keeps its winding-one internal geometry. The six
coordinates include three signed forms and three lifted phase origins;
the constituent labels, marked port embeddings and contact parameters
remain part of the law. Independent changes of these phase origins
without corresponding changes to the contacts are not symmetries of the
fixed network. A common phase shift is a symmetry. A common signed form
offset is also a symmetry of this supplied difference-only law.

The winding one's vanishing first phase resultant and registration-losing
ninth moment cannot replace \(\Phi_r\) with its marked embedding.
Equation (5) is a prescribed invariant family, not an acquired source.
Arbitrary fine states outside it still retain their internal coordinates.

<a id="organized-network-restriction"></a>
## Exact six-coordinate dynamics and full storage

Let \(B\) be the three-by-three incidence matrix with a positive entry
at an edge's tail and a negative entry at its head, in order (1):
\[
 B=\begin{pmatrix}1&0&-1\\-1&1&0\\0&-1&1\end{pmatrix},\qquad
 L=BB^T=\begin{pmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{pmatrix}.
 \tag{6}
\]
Define \(\Delta=-B^T\Phi\), so \(\Delta_e=\Phi_t-\Phi_r\), and
\[
 \begin{split}
 m_e&=1-s_e+s_e e^{i\alpha}=\rho_e e^{i\phi_e},\\
 J_e(\Delta_e)&=\operatorname{Im}(m_e e^{i\Delta_e})
                    =\rho_e\sin(\Delta_e+\phi_e),\\
 U_e(\Delta_e)&=1-\operatorname{Re}(m_e e^{i\Delta_e})
                    =1-\rho_e\cos(\Delta_e+\phi_e).
 \end{split}\tag{7}
\]
The real part of \(m_e\) is positive. Thus its continuous argument is
unambiguously chosen in \([0,\alpha]\), and
\[
 \rho_e=\sqrt{1-2s_e(1-s_e)(1-\cos\alpha)}>0.
 \tag{8}
\]
For a node at the tail of \(e\), the weighted cross current equals
\(J_e\); at its head it equals \(-J_e\). Internal currents cancel
pairwise. Form diffusion is constant within each constituent and equals
\((LX)_r\). Consequently the exact restriction is
\[
 \boxed{
 X'=\tfrac14[-LX+\gamma BJ(\Delta)],\qquad
 \Phi'=\tfrac\gamma4 LX .}
 \tag{9}
\]
Both fine rows are tangent to (5). The fine vector field is globally
Lipschitz in its lifted coordinates, so uniqueness proves invariance.
This is exact nonlinear closure on (5), with fixed boundary coefficients;
it makes no off-family approximation or transverse-attraction claim.

The internal potential baseline is
\(B_0=27(1-\cos\alpha)\). Counting nine fine copies of every collective
edge gives
\[
 \begin{split}
 H_f|_{\mathcal M}
   &=B_0+9\left[\tfrac12X^TLX+\sum_{e\in\mathcal E}U_e(\Delta_e)\right],\\
 H_f'|_{\mathcal M}&=-\tfrac94\|LX\|_2^2 .
 \end{split}\tag{10}
\]
Indeed the gradients of the bracket are \(LX\) and \(-BJ\); the
reciprocal cross terms cancel exactly. Its potential remains nonnegative
because each \(U_e\) is the actual convex combination of two fine sine
potentials. The constants \(B_0\) and \(1-\rho_e\) are not removed
from the support-work account.

<a id="organized-network-chart"></a>
## Necessary and sufficient simultaneous chart conditions

The comparison target is the same unweighted three-edge triangle, with
the normalized complete reciprocal sine law
\[
 \frac{dZ}{d\sigma}=-LZ+\Gamma B\sin(-B^T\Psi),\qquad
 \frac{d\Psi}{d\sigma}=\Gamma LZ,
 \qquad \Gamma>0.
 \tag{11}
\]
Sine acts componentwise. The comparison uses every state on (5), not
only an equilibrium, one trajectory or finitely many sampled responses.
It does not allow edge-dependent diffusion, an independently adjustable
phase-row support, nodewise clocks, state-dependent charts or an added
forcing law. Those would be different comparison classes.

Allow any fixed positive nodewise affine form chart, constant phase
origin shifts and one positive constant common clock:
\[
 X=AZ+b,\quad A=\operatorname{diag}(a_0,a_1,a_2),\ a_r>0,
 \qquad\Psi=\Phi+\beta,\qquad\sigma=c\tau,\ c>0.
 \tag{12}
\]
All chart constants may depend on the supplied contact weights, but not
on state or measured response. We retain the transformations of the
marked ports, inputs and observations below.

**Theorem.** A chart (12) turns both rows (9) into (11) if and only if
\[
 \rho_{01}=\rho_{12}=\rho_{20}=\rho>0,\qquad
 \Theta:=\phi_{01}+\phi_{12}+\phi_{20}=0\pmod{2\pi}.
 \tag{13}
\]
Every such chart has
\[
 \begin{gathered}
 c=\tfrac14,\qquad a_0=a_1=a_2=\sqrt\rho,\qquad
 b=b_*\mathbf1,\qquad \Gamma=\gamma\sqrt\rho,\\
 \beta_t-\beta_r=\phi_{rt}\pmod{2\pi}\quad(e=(r,t)).
 \end{gathered}\tag{14}
\]
The common form offset \(b_*\) and common phase origin are arbitrary
symmetries, not fit parameters that remove the compatibility conditions.

To prove necessity, compare the terms linear in \(Z\) in the form row.
The transformed diffusion matrix is \((4c)^{-1}A^{-1}LA\). Its diagonal
entries must equal those of \(L\); since they are two, \(c=1/4\).
The off-diagonal entries then force \(a_t/a_r=1\) on every edge. Hence
\(A=aI\). The phase row independently requires \(Lb=0\) and
\(\Gamma=\gamma a\). Connectedness gives \(b=b_*\mathbf1\).

The remaining form-row equality is
\[
 B\operatorname{Im}\left[
   \left(m_e e^{-i(\beta_t-\beta_r)}
                   -a^2\right)e^{i(\Psi_t-\Psi_r)}\right]_{e\in\mathcal E}
 =0\quad\text{for all }\Psi.
 \tag{15}
\]
The three oriented edge characters on the node-phase torus are distinct
up to sign. In a given node row the two incident nonconstant Fourier
characters are also distinct. Uniqueness of their Fourier coefficients
therefore implies, for each edge separately,
\[
 m_e e^{-i(\beta_t-\beta_r)}=a^2.
 \tag{16}
\]
This step uses the full network vector field: merely assuming individual
edge equality would miss the need to exclude cancellation of net nodal
currents. The identity \(B\mathbf1=0\) permits constant circulating
edge currents, but the nonconstant characters in (15) cannot produce
such an unobserved circulation for every independent node phase.

Taking magnitudes gives \(\rho_e=a^2\); summing the argument differences
around the triangle gives \(\Theta=0\) modulo \(2\pi\). Conversely,
when (13) holds, set \(\beta_0=0\),
\(\beta_1=\phi_{01}\), \(\beta_2=\phi_{01}+\phi_{12}\).
The closing edge then satisfies (16) as well. Equations (14) transform
both rows directly to (11), proving sufficiency.

This theorem classifies the declared affine/phase/clock maps. It is not
a theorem ruling out every nonlinear state conjugacy, observation
equivalence or model with additional boundary variables.

<a id="organized-network-controls"></a>
## The co-oriented family and the two fixed controls

For the actual contacts (2), \(0\le\phi_e\le\alpha\), with
\(\phi_e=0\) if and only if \(s_e=0\). Therefore
\[
 0\le\Theta\le3\alpha=2\pi/3<2\pi.
 \tag{17}
\]
Vanishing modulo \(2\pi\) is possible only when every \(s_e=0\).
Then every \(\rho_e=1\), and the complete target law (11) follows
with \(\Gamma=\gamma\) and \(\sigma=\tau/4\). All other contacts
in this specified co-oriented cube fail at least the cycle condition.
Changing an interface orientation changes the supplied support and
must be treated as a different model, not used to erase this result.

For the equal-weight control \(s=(1/2,1/2,1/2)\),
\[
 \rho_e=\cos(\alpha/2),\qquad
 \phi_e=\alpha/2,\qquad\Theta=3\alpha/2=\pi/3.
 \tag{18}
\]
All pairwise form and reciprocal scales agree, but their phase-origin
charts cannot coexist around the cycle. This isolates the phase
obstruction independently of amplitude mismatch.

For the unequal control \(s=(1/4,1/2,3/4)\), the identity
\(m(1-s)=e^{i\alpha}\overline{m(s)}\) yields
\(\phi(1/4)+\phi(3/4)=\alpha\), so again \(\Theta=\pi/3\).
Its magnitudes satisfy
\[
 \rho_{01}^2=\rho_{20}^2
      =1-\tfrac38(1-\cos\alpha)
      >1-\tfrac12(1-\cos\alpha)=\rho_{12}^2.
 \tag{19}
\]
Thus this fixed control also fails the common positive form/reciprocal
scale condition. Choosing a different form unit for each pair would
assign incompatible values to a constituent shared by two pairs; the
nodewise argument leading to (14) makes that failure explicit.

Failure of (13) does not imply absence of an equilibrium. At equal
\(s_e=s\), take \(X=X_*\mathbf1\), \(\Phi=\Phi_*\mathbf1\).
Every edge current is \(s\sin\alpha\), so \(BJ=0\) and both rows
vanish. The same cancellation holds at every fine node in (3), making
this an exact fine equilibrium. For \(s>0\) the structural edge-current
terms form a nonzero cycle with zero net nodal current. This algebraic
statement assigns no transported physical quantity. The static control
makes no source-acquisition or neighborhood-stability assertion. At unequal
weights the same aligned state instead has
\(X'=\gamma\sin\alpha\,Bs/4\), which is nonzero for (19).

<a id="organized-network-boundary-law"></a>
## A sufficient retained boundary law when the chart fails

For all contacts, fix the same spanning-tree phase chart
\[
 \beta_0=0,\qquad\beta_1=\phi_{01},\qquad
 \beta_2=\phi_{01}+\phi_{12},\qquad\Psi=\Phi+\beta.
 \tag{20}
\]
Use \(Z=X\), \(\sigma=\tau/4\), and retain the transformed contact
phases
\[
 \chi_{01}=0,\quad\chi_{12}=0,\quad\chi_{20}=\Theta
                      \pmod{2\pi}.
 \tag{21}
\]
The complete law becomes
\[
 \frac{dZ}{d\sigma}
   =-LZ+\gamma B[\rho_e\sin(\Psi_t-\Psi_r+\chi_e)]_e,
 \qquad
 \frac{d\Psi}{d\sigma}=\gamma LZ.
 \tag{22}
\]
It retains exactly the amplitudes and the closing-edge phase that the
unmodified scalar-node law (11) cannot absorb. All quantities are
determined by the supplied interfaces. No new evolving coordinate,
memory approximation, controller or hidden force is added on (5).

A different spanning tree redistributes the residual phases among
edges. A constant change of node phase origins changes each edge phase
by a node difference, whose sum around the cycle is zero. The product
\(\prod_e m_e/\rho_e=e^{i\Theta}\) is therefore unchanged by this
coordinate change. This is a property of the marked contact law. It is
not evidence of a physical gauge field, a charge or the separate
[auxiliary gauge construction](../GAUGE_SYMMETRY_AND_UNIFICATION.md).

The original fine storage in this chart is
\[
 H_f=B_0+9\left[\tfrac12Z^TLZ+
      \sum_e\{1-\rho_e\cos(\Psi_t-\Psi_r+\chi_e)\}\right],
 \qquad
 \frac{dH_f}{d\sigma}=-9\|LZ\|_2^2.
 \tag{23}
\]
Each potential term splits into \((1-\rho_e)+
\rho_e[1-\cos(\Psi_t-\Psi_r+\chi_e)]\). If \(\Theta\ne0\)
modulo \(2\pi\), all three second terms cannot vanish simultaneously:
their effective angles sum to \(\Theta\). Thus the sum of isolated
pair minima is unattainable on this triangle. This conclusion does not
identify the minimizing network state or prove how it is reached.

More generally one may retain a common positive scale \(X=aZ+b_*\mathbf1\)
in (20). Then the phase coefficient is \(\Gamma=\gamma a\) and the
form-current coefficients in its reciprocal normalization are
\(\rho_e/a^2\). Retaining these edge coefficients is legitimate;
declaring them all equal to one without (13) changes the model.

<a id="organized-network-work-observation"></a>
## Inputs, support work and measured coordinates

Constituent-uniform fine inputs \(f_r(\tau)\) in the form row and
\(g_r(\tau)\) in the phase row preserve (5). Their complete balance is
\[
 H_f'=-\tfrac94\|LX\|_2^2
       +9[(LX)^Tf-(BJ)^Tg].
 \tag{24}
\]
For \(X=aZ+b_*\mathbf1\), \(\Psi=\Phi+\beta\) and
\(\sigma=\tau/4\), their transformed rows receive
\(4f(4\sigma)/a\) and \(4g(4\sigma)\), respectively.
The support and charts stay fixed. Applying these transformations to
the full storage gives exactly four times the work rate in (24).
Original coordinate observations retain
\[
 x_{r,\ell}=aZ_r+b_*,\qquad
 \theta_{r,\ell}=\Psi_r-\beta_r+\alpha(\ell-4).
 \tag{25}
\]
A form sensor therefore does not generally read \(Z_r\), and a phase
sensor does not discard \(\beta_r\). Other observations must compose
with this same embedding. No physical sensor calibration is derived.

When (13) is satisfied, the full storage in the target chart (11) is
\[
 H_f=B_0+9\sum_e(1-\rho)
          +9\rho\left[\tfrac12Z^TLZ+
                            \sum_e(1-\cos(\Psi_t-\Psi_r))\right],
 \qquad
 \frac{dH_f}{d\sigma}=-9\rho\|LZ\|_2^2.
 \tag{26}
\]
This explicitly retains the multiplicative scale and additive boundary
cost, rather than equating two differently normalized storages.

Starting with the three isolated cycles, attach (2) without changing any
fine form or phase. The support-work jump is exactly
\[
 \mathcal W_{\rm attach}
   =\sum_{\text{new }\{i,m\}}w_{im}
       [\tfrac12(x_m-x_i)^2+1-\cos(\theta_m-\theta_i)]\ge0.
 \tag{27}
\]
On (5) it equals
\[
 \mathcal W_{\rm attach}
       =9[\tfrac12X^TLX+\sum_eU_e(\Delta_e)].
 \tag{28}
\]
At the aligned state with constant forms and phase origins, this is
\(9(1-\cos\alpha)\sum_e s_e\), equal to
\(27(1-\cos\alpha)/2>0\) for both fixed controls. A positive jump
requires supplied work; the continuous loss in (4) is not an available
reservoir or a reason for the contact to occur.

At fixed support, a uniform form jump \(h_r\) becomes \(h_r/a\),
while a uniform phase jump keeps its angle and the fixed origin offsets.
Its work is the difference of the full storage. These coordinate facts
do not admit or select a registered hybrid operator. Its live preconditions,
atomic targets, [collective descent](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
and occurrence need their own contracts. Time-varying weights would add
explicit support-work and chart-derivative terms absent from (3).

<a id="organized-network-scope"></a>
## Consequence for inheritance and the remaining scope

Pairwise same-profile representations require compatibility at every
shared constituent and every network cycle. Even the sine profile
selected by the additional pairwise premise has a nontrivial boundary
law on the prescribed triangle. The sufficient inherited description
is (9), or its explicit tree chart (22), with marked boundary information
retained. The [collective interaction owner](DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state)
already distinguishes such information from a universal scalar-node
representation; this is an exact organized example of that distinction.

The winding-one geometry is preserved for every trajectory initially
on (5). This is conditional invariant-family identity, with no assertion
that arbitrary nearby states remain there or that previously acquired
two-object error budgets survive this different support and degree.
Matching target geometry and local coefficients does not transfer
nonlinear recovery or acquisition certificates. Off-family source
transport, full-state stability, maintenance and event selection remain
separate proof obligations. No trajectory, source producer, physical
dataset, topology search or new numerical protocol was evaluated.

This result neither identifies a fundamental physical interaction nor
rules out a hierarchy carrying its actual boundary laws. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next admission decision.

## Implementation and independent controls

The detached static owner
[_organized_network_compatibility.py](../../src/tnfr/physics/_organized_network_compatibility.py)
rebuilds the actual fine support, degree, contact coefficients, complete
collective rows and full storage from admitted primitive data. Its fine
reciprocal coefficient is fixed to \(1/(1023\pi)\); the proof permits
any positive fixed \(\gamma\). The shared contact factors in
[_oriented_interface_constitutive.py](../../src/tnfr/physics/_oriented_interface_constitutive.py)
carry no pair-model degree, source certificate or network verdict.

The static observation admits three joint node-phase phasors, then derives
the three edge differences from that assignment. Three independently
supplied edge differences need not satisfy the cycle identity and are
not substituted for joint state. These circular observations do not
construct continuous lifts or run the fine law. Exact primitive conditions
decide chart compatibility: all weights vanish for the phase condition,
and all \(s_e(1-s_e)\) agree for the amplitude condition. Overlapping
rounded intervals cannot substitute for these equalities. A compatible
common amplitude scale is reported separately even if the phase condition
fails; it then belongs to a law retaining its phase lags, not to (11).

The [independent controls](../../tests/physics/test_organized_network_compatibility.py)
compare these reports with the full 27-node edge construction and cover
the fixed equal and unequal contacts, the compatible zero-offset control,
the tree chart, storage/loss and admitted-state boundaries. No trajectory
or source-acquisition producer is part of these static checks.
