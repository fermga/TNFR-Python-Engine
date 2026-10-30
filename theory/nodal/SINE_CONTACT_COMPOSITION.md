# Composing finite contact responses under a held sine law

<a id="sine-contact-composition"></a>

Two separately predicted contact responses need not add to the joint
response, even when every evolution is linear in its state. This chapter
bounds that composition error for the complete positive-capacity `D/4`
sine law. The error bound is proportional to the product of the two contact
conductances on each fixed finite window. Exact controls distinguish
shared ports, independent contacts and communication through old support.
The result concerns prediction of a supplied interaction, not autonomous
attachment or the formation of a new joint organization.

The [single-contact comparison](ORGANIZED_CONTACT_MAINTENANCE.md#sine-matched-bridge-interface)
already retains a common full initialization. The
[repeated-composition theorem](REPEATED_ORGANIZED_COMPOSITION.md#repeated-organized-composition)
already treats registered multi-edge interfaces under its quarter-sine
premises. The [causal substitution theorem](SINE_CLASS_INTERFACE_COMPOSITION.md#sine-composition-output-substitution)
retains feedback under a different degree-normalized law. The new question
is the error of adding two separately observed contact responses while
holding the complete source and capacity vector fixed across four branches.
These precedents supply methods, not transferred coefficients or verdicts.

## 1. Four complete branches and one constituent observation

Use one finite node set, a fixed symmetric nonnegative old weighted
support, and two fixed undirected contact edges with incidence vectors
\(b_1,b_2\). Their endpoints may share a node or be distinct. For held
parameters \(p,q\ge0\), write
\[
 L_{p,q}=L_0+pL_1+qL_2,\qquad L_i=b_ib_i^T,
 \qquad S_{p,q}(\theta)=S_0(\theta)+pS_1(\theta)+qS_2(\theta),
 \tag{1}
\]
where each sine current uses exactly the same conductances as its
Laplacian. Each \(S_i\) is the unit-conductance current on its edge.
On every branch prescribe
\[
 D=\operatorname{diag}(\nu_i),\quad \nu_i>0,\qquad
 x'=D[-L_{p,q}x+\gamma S_{p,q}(\theta)]/4,
 \quad \theta'=\gamma DL_{p,q}x/4,
 \quad \nu'=0. \tag{2}
\]
Capacities, \(\gamma\ge0\), old conductances and the structural clock
are held. The complete initial signed form and continuous phase lift are
the same for every \((p,q)\), including all unobserved coordinates.
There is no later input, event or conductance change. The state and its
capacity vector can vary between members of an admitted family; matching
is within each member, not equality to one fitted nominal preparation.

The four branches have parameter pairs
\((0,0),(w_1,0),(0,w_2),(w_1,w_2)\), abbreviated \(0,1,2,12\).
For a fixed nonempty constituent \(A\), define
\[
 m^{\rm a}=\frac1{|A|}\sum_{i\in A}x_i,
 \qquad \Delta_{\rm a}(h)=m_{12}^{\rm a}(h)-m_1^{\rm a}(h)
                                  -m_2^{\rm a}(h)+m_0^{\rm a}(h).
 \tag{3}
\]
This is the joint matched response minus the two separate matched
responses: the common disconnected baseline has coefficient plus one.
All initial terms cancel because the four complete preparations match.
It is not a comparison of independently selected members of state boxes.

Supply bounds \(0<m\le\nu_i\le M\) for every node,
\(\|x(0)\|_\infty\le B_0\), \(0\le\gamma\le g\), and a degree
ceiling \(K\) for the support with both contacts at their upper weights.
Nonnegative weights make this a degree ceiling on the whole parameter
rectangle. Merely checking \(K\ge\max(W_{1,\rm hi},W_{2,\rm hi})\)
is necessary scalar admission, not authentication of the actual degree:
shared ports consume the sum of incident weights. Requiring that sum
at every node instead would unnecessarily exclude valid distinct-port
supports. The actual old support, endpoint association, capacity vector,
constituent membership and common initialization remain caller premises.

Zero contact conductance removes that edge's contribution from both rows
of (2). This differs from the [native zero-relation boundary](RELATION_FOUNDATIONS.md#zero-relation-boundary),
where a bare-neighbor phase channel can retain a zero-conductance relation.
No result here transfers to a different support or degree-normalization law.

## 2. Full-state sensitivity to either contact

Finite weighted graphs give globally Lipschitz lifted vector fields in
(2): the form dependence is linear and sine has bounded derivatives.
Complete solutions exist on every finite window and depend smoothly on
both parameters. The same initial state is parameter-independent. Thus
first and mixed parameter derivatives start at zero, including at the
nonnegative rectangle boundary by restriction of the smooth family.

Put \(k=MK\) and \(\alpha=gk/2\). As in the single-contact proof,
\(-DL_{p,q}/4\) generates a maximum-norm contracting Markov semigroup,
and every branch has
\[
 \|x(t)\|_\infty\le B(t)=B_0+\frac{kg}{4}t. \tag{4}
\]
At a fixed parameter pair, write
\(\xi_j=\partial_jx,\eta_j=\partial_j\theta\), and
\(u_j=\|\xi_j\|_\infty,v_j=\|\eta_j\|_\infty\).
Differentiation in contact \(j\) gives
\[
 \begin{aligned}
 \xi_j'&=-DL\xi_j/4+\gamma D\,\mathrm DS(\theta)\eta_j/4
                         +D[-L_jx+\gamma S_j(\theta)]/4,\\
 \eta_j'&=\gamma DL\xi_j/4+\gamma DL_jx/4.
 \end{aligned} \tag{5}
\]
Here \(L,S\) denote the current complete support, and
\(\mathrm D\) denotes differentiation with respect to phase.
The first derivative therefore satisfies
\[
 \begin{aligned}
 u_j(t)&\le\int_0^t[\alpha v_j(s)+M(2B(s)+g)/4]ds,\\
 v_j(t)&\le\int_0^t[\alpha u_j(s)+MgB(s)/2]ds.
 \end{aligned} \tag{6}
\]
Both rows and both direct edge-forcing terms are retained. No form mean,
internal mode or phase feedback has been set to zero.

Define
\[
 b_0=M\left[\frac{(1+g)B_0}{2}+\frac g4\right],\qquad
 b_1=\frac{M(1+g)kg}{8},\qquad
 P(t)=b_0t+\frac{b_1t^2}{2}.
\]
The scalar integral comparison for \(u_j+v_j\) gives, uniformly on the
parameter rectangle and \(0\le t\le h\),
\[
 u_j(t)+v_j(t)\le e^{\alpha t}P(t)\le G P(t),
 \qquad G\ge e^{\alpha h}. \tag{7}
\]
These are numerical product norms in the fixed form, phase and clock
normalization of (2). Adding the bounds does not identify physical units
or define a new structural observable. A unit change must transform both
rows and their coefficient bounds before the comparison is reused.

## 3. Mixed sensitivity retains contact feedback and sine curvature

Let \(\chi=\partial_1\partial_2x\) and
\(\psi=\partial_1\partial_2\theta\). Differentiating (5) again gives
\[
 \begin{aligned}
 \chi'={}&-DL\chi/4+\gamma D\,\mathrm DS(\theta)\psi/4
       +\gamma D\,\mathrm D^2S(\theta)[\eta_1,\eta_2]/4\\
 &+D[-L_1\xi_2-L_2\xi_1
       +\gamma\mathrm DS_1(\theta)\eta_2
       +\gamma\mathrm DS_2(\theta)\eta_1]/4,\\
 \psi'={}&\gamma D[L\chi+L_1\xi_2+L_2\xi_1]/4.
 \end{aligned} \tag{8}
\]
The two edge-Jacobian terms survive when \(\gamma=0\); they describe
how one contact changes the state on which the other acts. The sine
Hessian is a separate forcing term. Dropping either contribution would
change the mixed response being bounded.

For each old or new edge, phase increments are at most twice the nodal
maximum. The complete sine Hessian consequently obeys
\[
 \|\mathrm D^2S(\theta)[\eta_1,\eta_2]\|_\infty
                         \le4K v_1v_2.
\]
After multiplication by \(\gamma D/4\), its form forcing is at most
\(gk v_1v_2\). A unit edge has Laplacian norm at most two, and its
sine Jacobian has the same bound. Each edge's combined form/phase forcing
from the other sensitivity is therefore at most
\[
 \frac M2[(1+g)u_j+g v_j]
                  \le\frac{M(1+g)}2(u_j+v_j).
\]
If \(z=\|\chi\|_\infty+\|\psi\|_\infty\), heat contraction,
(7) and the zero initial mixed derivative give
\[
 z(t)\le\int_0^t\left[\alpha z(s)
               +M(1+g)G P(s)+gkG^2P(s)^2\right]ds. \tag{9}
\]
The two polynomial integrals are
\[
 A_1=\int_0^hP(s)ds=\frac{b_0h^2}{2}+\frac{b_1h^3}{6},
 \qquad
 A_2=\int_0^hP(s)^2ds=
 \frac{b_0^2h^3}{3}+\frac{b_0b_1h^4}{4}+\frac{b_1^2h^5}{20}.
\]
A second scalar comparison proves
\[
 z(h)\le C(h):=M(1+g)G^2A_1+gkG^3A_2. \tag{10}
\]
The first term retains contact feedback; the second retains the complete
sine curvature. This estimate uses no acute chart, small-horizon condition
or disjoint-port assumption. It can become conservative on long windows.

Apply the fundamental theorem of calculus in both parameters to the full
state \(y=(x,\theta)\):
\[
 y(w_1,w_2)-y(w_1,0)-y(0,w_2)+y(0,0)
       =\int_0^{w_1}\int_0^{w_2}\partial_1\partial_2y(p,q)\,dq\,dp.
\]
Hence the complete mixed-state product norm and the arithmetic observation
are bounded by
\[
 \|\Delta x\|_\infty+\|\Delta\theta\|_\infty\le
 E(h):=W_{1,\rm hi}W_{2,\rm hi}C(h),\qquad
 |\Delta_{\rm a}(h)|\le E(h). \tag{11}
\]
The same formulas with an earlier endpoint bound the whole window. For
fixed finite background bounds and horizon, the deterministic allowance
is of product order in the two conductances. It is exactly zero if either
upper weight is zero, even when the other contact or old support drives
substantial motion. This conclusion would not follow by subtracting two
independent first-order state boxes.

## 4. Separate weighted readings, noise and representation

The actual inverse-capacity mean of the same constituent is
\[
 m^\rho=\sum_{i\in A}\rho_i x_i,\qquad
 \rho_i=\frac{1/\nu_i}{\sum_{j\in A}1/\nu_j}>0,
 \qquad\sum_{i\in A}\rho_i=1.
\]
Its weights are fixed across the four branches for each family member.
Both the arithmetic and this weighted observation have norm one on the
form maximum norm, so \(|\Delta_\rho(h)|\le E(h)\) as well. The
observations need not have equal values. Their common universal allowance
does not use a donor count, port capacity, internal-capacity contrast or
inverse-capacity mass. Those quantities are unnecessary inputs to this
norm projection; actual membership and normalization still need association.
There is no autonomous equation for either constituent mean.

For either observation, use its own four final-time absolute reading-error
radii \(\epsilon_{12},\epsilon_1,\epsilon_2,\epsilon_0\). With
\(e=\epsilon_{12}+\epsilon_1+\epsilon_2+\epsilon_0\),
\[
 \Delta\in[-E,E],\qquad\widehat\Delta\in[-E-e,E+e]. \tag{12}
\]
The coefficients are \((+1,-1,-1,+1)\); arbitrary correlations are
allowed. Initial-value or baseline errors in a different increment-based
instrument require their own association. Algebraic cancellation of the
true initial state does not cancel unrelated measurement errors. Without
weighted-reading radii, the true weighted allowance remains available but
its observed allowance is unrequested, not a zero-error measurement.

These symmetric enclosures are accuracy guarantees and cannot alone
establish a strict response sign. If an application supplies an admissible
error tolerance, \(E\), or \(E+e\) for recorded data, can be compared to
that tolerance. This theorem selects no universal acceptable error. A fixed
positive observation-error sum remains when either contact is removed.

Compute \(C\) and the product of the raw exact conductance endpoints
before outward display rounding. The exact rational endpoints in (12)
then preserve the product scaling for arbitrarily small positive weights.
The shared dyadic128 interval representation has a fixed numerical grid;
its displayed width need not shrink below that grid. It cannot replace the
exact rational endpoint pairs or be treated as an observation-noise model.

Requested old-chart retention uses the absolute phase displacement from
the [complete-row motion bound](ORGANIZED_CONTACT_MAINTENANCE.md#sine-capacity-bridge-interface):
\[
 T(h)=\frac{gkB_0h}{2}+\frac{k^2g^2h^2}{16},\qquad
 \mu_e-2T(h)>0 \quad\hbox{for every associated old edge}. \tag{13}
\]
The common joined-degree ceiling covers all four branches. A small mixed
response does not imply that any individual trajectory retains a chart.
An exhausted sufficient margin is indeterminate, not a witnessed exit.

For tight common capacity \(c>0\), the same clock transformation as
before gives \(C_c(h)=C_1(ch)\) and \(E_c(h)=E_1(ch)\).
Both full rows change together. At \(g=0\), the formulas remain regular:
\(G=1,b_0=MB_0/2,b_1=0\), and \(C=M^2B_0h^2/4\).
This generally nonzero bound already concerns linear state evolution.
Zero capacity remains outside the inverse-capacity observation domain.

## 5. Exact layouts distinguish marginal response from composition

Take four nodes \(0,1,2,3\), constituent \(A=\{0,1\}\), hosts
\(\{2,3\}\), unit capacities, \(\gamma=0\), initial forms
\((0,0,1,1)\), zero phases and unit contact weights. Use \(B_0=1\),
\(K=2\) for the following layouts, and put \(r=e^{-h/4}\).
These are finite mathematical controls, not acquired organized identities.

With no old edges and shared contacts \(02,03\), each single-contact
donor mean is \((1-r^2)/4\), whereas the joint mean is
\((1-r^3)/3\). The disconnected mean is zero, so
\[
 \Delta_{\rm shared}=-\frac{(1-r)^2(1+2r)}6<0\quad(h>0).
 \tag{14}
\]
For disjoint contacts \(02,13\) and the same empty old support, the
joint system consists of two independent pairs. Each single-contact
mean remains \((1-r^2)/4\), and
\[
 \Delta_{\rm disjoint}=0. \tag{15}
\]
Thus these two layouts have identical individual finite contact responses,
initial port differences, constituent means and supplied scalar ceilings,
but different joint responses. Such marginal data cannot determine exact
composition without the relevant contact association. The uniform bound
(11) still applies to both layouts without knowing which one is present.

Distinct ports do not by themselves establish independence. Keep contacts
\(02,13\) and add the old unit edge \(01\). Each single-contact active
component is now a three-node path. Its donor mean is
\(1/3-r/4-r^3/12\), while the joint mean is \((1-r^2)/2\). Hence
\[
 \Delta_{\rm path}=-\frac{(1-r)^3}6<0\quad(h>0). \tag{16}
\]
This third layout shares the initial port/state summaries and scalar
ceilings, but its individual finite responses differ from (14)-(15).
The old internal path communicates between the distinct ports.
The leading terms are \(-h^2/32\) in (14) and \(-h^3/384\) in (16).
They also follow from the actual diffusion-generator products: the mixed
quadratic operator is \(A_1A_2+A_2A_1\), which vanishes for disjoint
edge matrices, while old-support products can enter at the next order.

For a heterogeneous control change only the shared layout's capacities
to \((2,1,1,1)\). The donor node \(1\) remains stationary, and exact
pair/star solutions give
\[
 \Delta_{\rm a}=\frac25(1-e^{-5h/4})-\frac23(1-e^{-3h/4}),
 \qquad \Delta_\rho=\frac23\Delta_{\rm a}. \tag{17}
\]
The donor inverse-capacity mass is \(3/2\), so its normalized coefficient
on node zero is \(1/3\), compared with \(1/2\) for the arithmetic mean.
The two observations therefore have different true responses, although
both satisfy the universal product bound and keep independent error models.

All these branches obey a linear diffusion law in their state. Their
propagators use different generators as contacts are changed. Nonzero
(14), (16) or (17) therefore demonstrates contact-composition error, not
failure of fixed-generator linear superposition or evidence selecting a
nonlinear sine current. The existing [fixed-law mixed probes](SINE_CLASS_REPEATED_INTERACTION.md#sine-repeated-response-and-comparator)
have different intervention premises and retain their own null control.

## 6. Primitive consumer and evidence boundary

The private `_prepared_two_contact_composition` reader and frozen
`_PreparedTwoContactComposition` report belong to the shared
[preparation owner](../../src/tnfr/physics/_sine_preparation.py).
The compact input contract admits two ordered nonnegative weight intervals,
an ordered nonnegative gamma interval, a strictly positive global capacity
interval, nonnegative form/degree/horizon bounds, four arithmetic error
radii, optional separate weighted errors and optional old-chart margins.
It accepts no cached verdict, graph label or unused port-state summary as
proof of common preparation or boundary association.

All primitive domains and the derived growth domain are checked before
exponential or interval work. The reciprocal of the shared exact lower
bound on \(e^{-\alpha h}\) gives \(G\), before any outward display
rounding. Its implemented domain \(\alpha h\le4096\) is the existing
exponential owner's numerical work limit, not a restriction on the
finite-horizon theorem. The report separates the edge-derivative and
sine-curvature terms in (10), their exact conductance product, exact true
and observed defect pairs, and independent old-chart retention. The
`exact_mixed_zero` flag records whether the proved upper bound is exactly
zero. A false flag is inconclusive about the true defect: independent
contact blocks can have zero response with a positive universal bound.

The frozen protocol, source and independent
[composition controls](../../tests/physics/test_sine_contact_composition.py)
are retained under `artifacts/sine-contact-composition-20261023/`.
The first frozen run passed all 474 selected engine/mathematical cases:
95 new controls, 288 shared regressions and 91 interval tests. Another
151 catalog/navigation cases passed, giving 625 passed cases altogether.
Two catalog symlink-route cases were skipped because this Windows environment
could not create their test symlinks; those routes were not verified by those
cases. No new mathematical control was skipped. Black and flake8 passed.
Exact case properties and dependency hashes are retained in
`control-results.xml` and `control-validation.json`.

The complete analytical states satisfy independently summed diffusion
rows at the declared horizons `0,1/100,1/4,1`. The three unit-capacity
layouts all retain the bound `h^2/4`; at `h=1/4` it is `1/64`, including
for the independent pairs whose actual defect is zero. Exact generator
products confirm the distinct quadratic/cubic onsets. The heterogeneous
star retains its separate arithmetic and weighted responses. The other
controls check both dynamic-row Jacobians, the sine-Hessian population,
positive-gamma coefficients, zero and subgrid conductance products,
separate noise floors, common-clock reductions, chart conditions and
primitive admission before numerical work. They are finite implementation
checks of the theorem, not samples establishing its full parameter domain.

No trajectory, committed event, retained-producer or response replay,
empirical evaluation or parameter search was executed. The result supplies
a finite conditional prediction bound and a boundary-information obstruction,
not a selected fundamental law, physical measurement bridge or autonomous
composite identity.

<a id="sine-contact-influence"></a>
## 7. Finite influence associated with the actual support

The universal coefficient in (10) deliberately forgets where the contacts
act. Retaining the old support, positive capacities and marked endpoints
supplies a different sufficient certificate: separate invariant contact
blocks have exactly zero composition error, while connecting internal paths
enter a finite nonlinear bound through a positive comparison propagator.
This is a refinement of the same four-branch question, with no new law,
trajectory, event or autonomous composite identity.

The [native mediated-connection result](RELATIONAL_EFFECTIVE_CONNECTIONS.md#effective-link-admission)
already relates paths to tangent influence. The
[six-port kernels](SINE_CLASS_COLLECTIVE_INTERFACE.md#sine-collective-interface-kernels)
and [feedback substitution](SINE_CLASS_INTERFACE_COMPOSITION.md#sine-composition-output-substitution)
retain causal response under their degree-normalized law. The
[constituent-memory theorem](ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
retains nonlinear hidden history on its fixed quarter-sine organization.
Here the additional result is a finite four-branch allowance associated
with an admitted heterogeneous-capacity support. These precedents do not
transfer their generators, coefficients or reduced-state conclusions.

### 7.1 Primitive associations and the comparison matrix

Retain the full ordered finite node set, simple undirected old edge pairs
and their nonnegative conductances, the actual positive capacity vector,
two marked endpoint pairs, and the fixed nonempty observed constituent.
A declared old edge of weight zero contributes to neither row of (2).
The two marked contributions may coincide with each other or increment an
old edge; their parameters remain separately varied and their conductances
add. Duplicate old edge declarations, including a reversed duplicate, are
not a second independent marked contribution and are rejected.

Let \(W^+\) be the symmetric adjacency obtained by adding both contact
upper weights to the old conductances. Derive
\[
 d_i^+=\sum_j W^+_{ij},\quad K=\max_i d_i^+,\quad
 M=\max_i\nu_i,\quad k=MK,\quad g=\gamma_{\rm hi},
 \qquad
 \mathcal A=\frac{1+g}{4}D[\operatorname{diag}(d^+)+W^+].
 \tag{18}
\]
All its entries are nonnegative. Its row sum is
\((1+g)\nu_i d_i^+/2\); it is a comparison matrix, not the
physical diffusion generator. The joined upper support covers the entire
nonnegative contact-parameter rectangle. Capacities and \(g\) retain
the normalization and clock of (2).

The admitted scalar \(B_0\) bounds every actual initial absolute form.
Complete initial form and lifted phase still match throughout the parameter
rectangle. The finite bound below discards their detailed values, bounding
phase derivatives globally by \(|\sin|,|\cos|\le1\). Primitive support
admission does not prove this common initialization, that bound membership,
source absence or the future held-law premises. These remain independent
premises of the admitted family.

For marked contact \(\ell\), put \(a_\ell=|b_\ell|\), the vector
with value one on its two endpoints and zero elsewhere. Define
\[
 \mathcal B_\ell=\frac{1+g}{4}Da_\ell a_\ell^T,\qquad
 f_{\ell,0,i}=a_{\ell,i}\nu_i
             \left[\frac{(1+g)B_0}{2}+\frac g4\right],\qquad
 f_{\ell,1,i}=a_{\ell,i}\nu_i\frac{(1+g)kg}{8}.
 \tag{19}
\]
These retain the receiving endpoint's actual capacity. They do not replace
it by one common clock. Two coincident contacts have equal matrices and
forcing vectors but remain two parameter derivatives.

### 7.2 Nodal first and mixed sensitivity comparisons

At every parameter point let
\[
 q_{\ell,i}=|\partial_\ell x_i|+|\partial_\ell\theta_i|,
 \qquad
 r_i=|\partial_1\partial_2x_i|
                         +|\partial_1\partial_2\theta_i|.
\]
These are nonnegative nodal comparison coordinates in the fixed numerical
normalization; they do not identify form and phase as one physical quantity.
All start at zero because the complete preparation is parameter-independent.
Using the differentiated complete rows (5), the upper Dini derivative obeys
\[
 D^+q_\ell(t)\le\mathcal A q_\ell(t)
                         +f_{\ell,0}+t f_{\ell,1}.
 \tag{20}
\]
Indeed, each internal edge contributes at most
\(\nu_i W^+_{ij}[(1+g)(|\xi_i|+|\xi_j|)
+g(|\eta_i|+|\eta_j|)]/4\) to the combined absolute derivative.
This is bounded by \((\mathcal A q)_i\). The direct parameter forcing
is supported on the two marked endpoints and is at most
\(\nu_i[(1+g)B(t)/2+g/4]\), with \(B(t)\) from (4), giving (19).

For nonnegative vectors \(u,v\), define the positive bilinear map
\[
 \mathcal H_i(u,v)=\frac{g\nu_i}{4}
     \sum_j W^+_{ij}(u_i+u_j)(v_i+v_j).
 \tag{21}
\]
The full mixed rows (8) give
\[
 D^+r\le\mathcal A r+
       \mathcal B_1q_2+\mathcal B_2q_1+\mathcal H(q_1,q_2).
 \tag{22}
\]
The first two forcings retain the complete derivatives of each contact
field acting on the other sensitivity. The last retains sine curvature:
\(|\mathrm D^2\sin(\theta_j-\theta_i)[\eta_1,\eta_2]|
\le(|\eta_{1,i}|+|\eta_{1,j}|)
   (|\eta_{2,i}|+|\eta_{2,j}|)\).
It appears only in the form row; the phase row's contact derivatives are
already retained in \(\mathcal B_\ell\). No phase feedback or old
internal edge has been dropped. At \(g=0\), \(\mathcal H=0\), while
the contact-field terms can still produce a mixed response.

### 7.3 A positive propagator retaining exact path information

For the finite theorem, any entrywise nonnegative matrix
\(\mathcal P_h\ge\exp(h\mathcal A)\) suffices. Taking the exact
exponential is valid at every finite \(h\). Since \(\mathcal A\ge0\),
\(\exp(t\mathcal A)\le\exp(h\mathcal A)\) entrywise for
\(0\le t\le h\). A certificate must retain component zeros and the
size of connecting weights, rather than fill every reachable entry with
a fixed positive scalar tail.

The implemented rational construction sets \(Z=h\mathcal A\),
\(N=32\), and uses
\[
 \mathcal P_h=\sum_{n=0}^{N}\frac{Z^n}{n!}
    +\frac{Z^{N+1}}{(N+1)!}
                         \left(I-\frac{Z}{N+2}\right)^{-1}.
 \tag{23}
\]
When \(\|Z\|_\infty\le32\), the Neumann-series ratio has norm
at most \(32/34<1\). Its inverse is nonnegative, and coefficientwise
\[
 \sum_{r=0}^{\infty}\frac{Z^{N+1+r}}{(N+1+r)!}
 \le\frac{Z^{N+1}}{(N+1)!}
                  \sum_{r=0}^{\infty}\left(\frac Z{N+2}\right)^r.
 \tag{24}
\]
Each successive omitted factorial factor is at least \(N+2\), which
proves (24) and the desired exponential upper bound. All matrix arithmetic
in (23) is exact rational arithmetic. There is no rounded inverse or
empirical matrix exponential in this certificate.

Powers and the convergent resolvent preserve every invariant coordinate
block. They also vary continuously with admitted conductances, including
when an arbitrarily small positive old connection tends to zero. The
inverse remains uniformly nonsingular on the stated norm domain. Thus
(23) retains more information than a positive-support reachability mask:
a tiny connecting weight has a correspondingly small contribution.

The dimension limit of 24 and \(\|h\mathcal A\|_\infty\le32\)
are numerical work policies of this implementation, not restrictions on
the finite theorem with the exact exponential. The prior universal
coefficient still uses its separate \(\alpha h\le4096\) exponential
work limit. The matrix row-norm limit does not imply that scalar limit:
\(\max_i\nu_i d_i^+\) can be much smaller than \(MK\). Both are
admitted before numerical kernels, including when the final contact
product is zero.

### 7.4 Finite mixed-state and observation allowances

Define the nonnegative vectors
\[
 U_\ell=\mathcal P_h\left(f_{\ell,0}+\frac h2f_{\ell,1}\right).
 \tag{25}
\]
Variation of constants in (20), positivity and \(t\le h\) give
\[
 q_\ell(t)\le\int_0^t e^{(t-s)\mathcal A}
                (f_{\ell,0}+s f_{\ell,1})\,ds
 \le\mathcal P_h(t f_{\ell,0}+t^2f_{\ell,1}/2)
 \le tU_\ell.
 \tag{26}
\]
Substitute this inequality in (22), keeping both edge derivatives and the
local bilinear forcing. A second variation-of-constants comparison yields
\[
 r(h)\le R(h):=\mathcal P_h
 \left[\frac{h^2}{2}(\mathcal B_1U_2+\mathcal B_2U_1)
                          +\frac{h^3}{3}\mathcal H(U_1,U_2)\right].
 \tag{27}
\]
This bound is uniform over the whole parameter rectangle. The same
coefficient \(C(h)\) in (10) also bounds each \(r_i(h)\), because
that previous product norm bounds each nodal sum. Set
\[
 c_i(h)=\min\{R_i(h),C(h)\},\qquad
 E_i(h)=W_{1,\rm hi}W_{2,\rm hi}c_i(h).
 \tag{28}
\]
The exact two-parameter integration used in (11) proves
\[
 |\Delta x_i(h)|+|\Delta\theta_i(h)|\le E_i(h).
 \tag{29}
\]
In particular, a fixed normalized positive form reading \(m=\sum_i\rho_i x_i\)
has
\[
 |\Delta_m(h)|\le E_m(h):=\sum_i\rho_iE_i(h)
                 \le W_{1,\rm hi}W_{2,\rm hi}C(h).
 \tag{30}
\]
Use the actual arithmetic weights on the observed constituent and,
separately, its actual normalized inverse-capacity weights. They can give
different allowances even though both are no larger than the universal
one. The maximum of the nodal sums in (29) is not the previous sum of the
two coordinate maximum norms; equating those norms would lose a factor.
The report exposes nodal and observation allowances without that substitution.

Each reading retains its own four-error sum in (12), replacing \(E\) by
its own \(E_m\). The error floor need not vanish when the deterministic
allowance does. Exact conductance products and rational endpoint pairs
precede dyadic display rounding. The old-chart condition (13) remains a
separate absolute-displacement obligation, aligned with the admitted ordered
old edges; a small mixed defect does not certify chart retention. Common
capacity multiplication by \(c>0\) is equivalent to replacing \(h\)
by \(ch\) throughout both rows, the comparison matrices and allowances.
Node relabeling must transform support, capacities, marked contacts and
observation membership together.

### 7.5 Independence, communicating paths and limits of the certificate

If either upper contact weight is zero, (28) vanishes exactly. Suppose
both are positive and their endpoint sets lie in distinct connected
components of the positive joined support. Each complete law then splits
into independent invariant blocks. The support of \(U_\ell\) stays in
its own block, so \(\mathcal B_1U_2=\mathcal B_2U_1=0\), and no old
or marked edge can contribute to \(\mathcal H(U_1,U_2)\). Thus
\(R=0\) and the complete four-branch defect is exactly zero. This
also follows directly by separating the complete solutions by blocks;
it does not require special initial coordinates inside those blocks.
An observed invariant block unaffected by both contacts similarly has
zero projected allowance even if other blocks have a mixed response.

Distinct endpoints within one connected block do not satisfy that
independence premise. Propagation through old paths contributes through
(23), (25) and (27). If a collection of connecting old weights tends to
zero and its removal separates the two contact blocks, continuity of the
finite rational construction gives \(R\to0\), with exact zero at the
separated endpoint. This statement concerns fixed other primitives and
windows remaining in the admitted numerical domain. It neither replaces
a small positive weight by zero nor adds a display-grid error to the exact
certificate. A declared zero old edge is dynamically equivalent to its
absence for this two-row law, while its declaration can still carry a
separately requested chart obligation.

No finite propagation speed follows. Powers of \(\mathcal A\) encode
paths and their weights, but the full positive exponential and resolvent
include arbitrarily high powers. In a connected finite component they can
transmit comparison influence at every positive time. Small early-time
coefficients, path length and weak-weight limits describe attenuation,
not an exact delay or a physical metric. Nor does a positive allowance
prove a nonzero actual defect: cancellations or special preparations can
remain invisible to this sufficient comparison. Its symmetric intervals
do not determine a sign or close an autonomous constituent-mean law.

### 7.6 Consumer and retained verification boundary

The private `_prepared_support_contact_composition` consumer and frozen
`_PreparedSupportContactComposition` report belong to the
[support owner](../../src/tnfr/physics/_sine_contact_support.py). The reader
re-admits ordered old support, conductances, positive capacity rows, contact
endpoints, constituent membership,
weight/gamma intervals, preparation bound, horizon, reading errors and
optional old-chart margins. It derives joined degrees, capacity extrema,
positive-support components and the two actual observation weight vectors;
no supplied scalar degree or cached support report replaces those inputs.
Complete common initialization and the held law remain family premises.
The additive `nonnegative_exponential_upper` propagator in the
[comparison-flow owner](../../src/tnfr/mathematics/_comparison_flow.py) reuses
the shared exact matrix product and inverse owners. Existing scalar and
comparison-flow consumers keep their prior contracts. All-node zero and
each observation's zero flags are separate: observing an unaffected block
does not imply that the whole network has zero mixed response. A false zero
flag remains inconclusive about the actual response.

The prospective scope, baseline, evaluated source and independent reviews
are retained under `artifacts/sine-contact-influence-20261023/`. The first
frozen run passed all 734 selected engine/mathematical cases: 117 new
controls, 385 shared regressions and 232 arithmetic cases. Another 151
catalog/navigation cases passed, giving 885 passed cases altogether.
Two catalog symlink-route cases were skipped because Windows could not
create their test symbolic links; those routes were not verified by those
cases. No selected mathematical or engine case was skipped, and there
were no failures. Black and flake8 passed. Exact case properties, dependency
hashes and the separate skip reason remain in `control-results.xml`,
`control-validation.json` and `pytest-catalog.log`.

The [support controls](../../tests/physics/test_sine_contact_influence.py)
check complete analytical diffusion states against independently summed
rows on the fixed windows `0,1/100,1/4,1`, retaining shared, separate,
old-path and heterogeneous layouts. At `h=1`, the separate-pair allowance
is exactly zero and the shared-port arithmetic allowance is `1/8`, against
the prior universal `1/4` in both cases. Coincident marked contacts and
increments on an old edge retain their actual aggregate conductances.
A connecting old weight `2^-200` has a positive exact allowance below
that weight; setting it to zero reproduces the absent-edge invariant
blocks. The unaffected-isolate observation, separate noise floors,
subgrid contact products, capacity/clock scaling, relabeling, old-chart
association and admission controls also passed.

Independent signed edge sums verify both linearized rows and the sine
Hessian at the declared nonzero-gamma controls. The
[matrix controls](../../tests/test_nonnegative_comparison_kernel.py)
compare with separate scalar brackets, exact nilpotent and projector
identities, reciprocal even/odd series and a tiny-link Jordan identity;
component zeros and positive rational tails remain explicit. These are
finite implementation checks of the analytical theorem, not samples
establishing its entire parameter domain. No trajectory was integrated,
operator event executed, retained producer or certificate replayed, or
empirical measurement evaluated. No broader parameter campaign or next-gate
research result is claimed by this verification.

<a id="sine-retained-boundary-prediction"></a>
## 8. A smaller retained predictor with an omitted-state budget

A finite contact-error certificate does not itself evolve a smaller state.
This section supplies an explicit approximation under the same complete
`D/4` law: retain both coordinates on a proper region, replace the omitted
coordinates by declared fixed references, and bound the resulting reading
error. The allowance separates unknown hidden initialization, immediate
omitted motion and feedback driven by subsequent retained motion. It can
certify an absolute accuracy without resolving a much smaller interaction.

The [four-coordinate repeated-composition predictor](REPEATED_ORGANIZED_COMPOSITION.md#repeated-composition-fine-domain)
already has a finite transverse-error domain on its special support. The
[nonlinear constituent-memory result](ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
and [environmental elimination](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)
already retain hidden initialization and causal feedback. Their exact
history representations need not reduce the evolved dimension. Here the
predictor evolves exactly \(2|R|\) coordinates on an admitted finite support,
with an explicit error relative to the full \(2|V|\)-coordinate law.
No general compression theorem or newly selected fundamental law follows.

### 8.1 Exact discarded boundary contribution and the chosen approximation

Fix one symmetric nonnegative weighted support \(W\), positive held
capacities \(D\), a held \(\gamma\ge0\), and the original clock in (2).
This section concerns a point support and parameter value. An on/off
comparison supplies two such supports separately; there is no unused
contact-parameter rectangle in its primitive input contract. No later
forcing, event or support change occurs on either branch.

Let \(R\) be a nonempty proper retained region and \(H=V\setminus R\).
The original observed constituent \(O\) is a nonempty subset of \(R\).
Supply fixed hidden form and phase references
\((\bar x_H,\bar\theta_H)\), and initial retained coordinates
\((x_R^0,\theta_R^0)\). All phase comparisons use admitted real lifts.
The exact reduced field \(\widehat F_R\) is the full field's \(R\) rows
evaluated with the hidden coordinates fixed at those references. Every
\(R\)-to-\(H\) edge remains in both retained rows, with its actual
conductance and receiving capacity; deleting the edges would be a
different approximation.

Before approximating, the actual full retained rows have the exact identity
\[
 \begin{aligned}
 x_i'={}&\widehat F_{x,i}(x_R,\theta_R)
 +\frac{\nu_i}{4}\sum_{j\in H}w_{ij}
 \left[x_j-\bar x_j+\gamma\{
 \sin(\theta_j-\theta_i)-\sin(\bar\theta_j-\theta_i)\}\right],\\
 \theta_i'={}&\widehat F_{\theta,i}(x_R)
       -\frac{\gamma\nu_i}{4}\sum_{j\in H}w_{ij}(x_j-\bar x_j),
       \qquad i\in R.
 \end{aligned}\tag{31}
\]
Thus the discarded term consumes the live hidden form and phase, including
their initialization. It is not a known external input or a function of
the instantaneous retained mean alone. The reduced predictor discards
that term and evolves \(\widehat y_R'=\widehat F_R(\widehat y_R)\).
Its initial retained coordinates equal the full ones exactly.

For the proof, lift this predictor to \(V\) by adjoining the constant
hidden reference. This lift obeys the same row formula with mobility zero
on \(H\) and unchanged positive mobility on \(R\). It is a mathematical
embedding of the approximation, not a capacity reset or a claim that the
original positive-capacity hidden nodes are stationary. The predictor
needs no live hidden-coordinate integration. Full support, capacities and
reference constants can still be stored and used in preparation; reducing
evolved state is not removing all static input information.

Admit independent initial hidden radii
\[
 |x_i(0)-\bar x_i|\le r_{x,i},\quad
 |\theta_i(0)-\bar\theta_i|\le r_{\theta,i}\quad(i\in H),\qquad
 q_i^0=\begin{cases}0&i\in R,\\r_{x,i}+r_{\theta,i}&i\in H.\end{cases}
 \tag{32}
\]
No hidden initial value is fitted from a future response. Arbitrary hidden
states satisfying these radii belong to the conditional prediction class;
being in that class does not establish acquisition of an organized state.
If omitted nodes contributed to the original observation, a reconstruction
term would be necessary. The present consumer instead requires \(O\subseteq R\)
and preserves that original observation unchanged.

### 8.2 Reference rates control retained movement and omitted dynamics

Write \(d_i=\sum_jw_{ij}\), and use the same positive comparison method as
section 7, now on the actual point support:
\[
 A=\frac{1+\gamma}{4}D[\operatorname{diag}(d)+W],\qquad
 P_h\ge e^{hA},\qquad K_R\ge e^{hA_{RR}}.
 \tag{33}
\]
For any two complete states, the combined nodal absolute difference of
their full field rows is bounded by \(A\) times their combined nodal
state difference. This follows edgewise from the linear form row,
\(|\sin a-\sin b|\le|a-b|\), and the full phase row. It is global in
the chosen phase lifts and needs no acute-sector chart.

Let \(\widehat y^0=(x_R^0,\bar x_H,\theta_R^0,\bar\theta_H)\), in
node order. Independently enclose the complete initial field and supply
\[
 c_i\ge |F_{x,i}(\widehat y^0)|+
                    |F_{\theta,i}(\widehat y^0)|.\tag{34}
\]
These are model rates at a supplied reference, not measured rates inferred
from the later response. The form enclosure retains its actual
\(-Lx+\gamma S\) sum before taking an absolute bound; its cancellation
must not be silently discarded or assumed. Shared outward rational arithmetic and sine enclosures give sound
complete-row \(c_i\); either arithmetic or trigonometric enclosures can
overestimate cancellation. A zero enclosure certifies a zero reference
rate; a positive enclosure can remain inconclusive about an exact zero.

Set \(v_i(t)=|\widehat x_i(t)-x_i^0|
+|\widehat\theta_i(t)-\theta_i^0|\) for \(i\in R\). Since hidden
reference coordinates do not move, upper Dini comparison gives
\[
 D^+v_R\le A_{RR}v_R+c_R,\qquad v_R(0)=0,
 \qquad v_R(t)\le tK_Rc_R\quad(0\le t\le h).
 \tag{35}
\]
Indeed, insert and subtract \(\widehat F_R(\widehat y_R^0)\), use the
componentwise Lipschitz bound, and integrate its positive propagator.
This avoids both a measured velocity and an assumed retained trajectory.
The initial field and global derivative bound alone control movement.

The hidden residual of the lifted predictor is the omitted full row,
\(F_H(\widehat y(t))\). Its combined absolute norm satisfies
\[
 |F_{x,H}(\widehat y(t))|+|F_{\theta,H}(\widehat y(t))|
       \le c_H+A_{HR}v_R(t)
       \le c_H+tA_{HR}K_Rc_R.\tag{36}
\]
This separates the hidden motion already present at the reference from
the further hidden response driven by retained movement. Both complete
rows and every retained-hidden edge are included. Neither term asserts
that the actual hidden response is linear or has a positive sign.

### 8.3 Three finite error contributions and unchanged observations

Let \(J_H\) insert a vector on \(H\) into the full node order, with zeros
on \(R\), and put
\(e_i=|x_i-\widehat x_i|+|\theta_i-\widehat\theta_i|\).
Insert and subtract \(F(\widehat y)\) in the full-minus-lifted-predictor
rows. On \(R\) the predictor uses that exact field; on \(H\) its rate
is zero. Equations (32)-(36) therefore imply
\[
 D^+e\le Ae+J_H(c_H+tA_{HR}K_Rc_R),\qquad e(0)\le q^0.
 \tag{37}
\]
For every \(0\le t\le h\), positivity and variation of constants yield
\[
 \begin{aligned}
 e(t)&\le E^{\rm init}+E^{\rm drift}+E^{\rm feedback}=:E,\\
 E^{\rm init}&=P_hq^0,\\
 E^{\rm drift}&=hP_hJ_Hc_H,\\
 E^{\rm feedback}&=\frac{h^2}{2}P_hJ_H(A_{HR}K_Rc_R).
 \end{aligned}\tag{38}
\]
All coefficients are known before evaluating a full or reduced response.
In particular, a zero reference and zero retained preparation at
\(\gamma=0\) give \(c=0\), even if actual hidden initialization is
uncertain. That uncertainty belongs only to \(E^{\rm init}\); adding
it to a predictor-amplitude bound would incorrectly create an apparent
reference-drift source in this control.

For the fixed constituent \(O\), use separately
\(\rho_i^{\rm a}=\mathbf1_{i\in O}/|O|\) and
\(\rho_i^\nu=(\mathbf1_{i\in O}/\nu_i)/\sum_{j\in O}1/\nu_j\).
The original full and reduced form means then satisfy
\[
 |m_\rho(t)-\widehat m_\rho(t)|\le
 \mathcal E_\rho:=\rho^TE
 =\mathcal E_\rho^{\rm init}+\mathcal E_\rho^{\rm drift}
                              +\mathcal E_\rho^{\rm feedback}.
 \tag{39}
\]
The weights depend on actual positive capacities and membership. The two
readings generally differ and can have different allowances. As before,
a nodal sum of two absolute errors is not the sum of their separate
maximum norms; no such norm substitution is made.

With an independently supplied total reading-comparison uncertainty
\(\epsilon_\rho\), the corresponding recorded discrepancy is at most
\(\mathcal E_\rho+\epsilon_\rho\). A prospectively declared tolerance
\(\tau_\rho\) is certified when that sum is at most \(\tau_\rho\).
Failure of this sufficient inequality is indeterminate. Optional weighted
error or tolerance data remain unavailable when absent. The mathematical
reduced solution is exact here; a later numerical solver requires its own
error allowance and is not silently covered by a readout radius.

For contact on/off branches, retain the same constituent, preparation
class, references and clock, and change only the declared contact support.
With \(\mathcal R=m_{\rm on}-m_{\rm off}\) and the analogous reduced
prediction, the direct consequence is
\[
 |\mathcal R-\widehat{\mathcal R}|\le
 \mathcal E_{\rho,\rm on}+\mathcal E_{\rho,\rm off}.
 \tag{40}
\]
The two observation uncertainties add separately. Equal hidden preparation
does not prove cancellation of the two reduction-error bounds. A four-branch
composition contrast would likewise need four allowances unless a separate
matched-error argument established a stronger result.

If positive support separates \(R\) from \(H\), both inserted hidden
sources and hidden initialization stay in their own invariant blocks;
all retained allowances are exactly zero. An observed unaffected block
can have this property even when hidden dynamics are substantial. With
communicating paths the allowance can be positive and varies with their
actual weights. It is neither a witnessed nonzero error nor a propagation
speed. When certified initial rates vanish, the reference predictor is
stationary, but positive-capacity full hidden motion can still be induced
by nonzero hidden initial discrepancies.

The rational construction (23) supplies \(P_h\) and \(K_R\). The full
numerical conditions \(|V|\le24\) and \(h\|A\|_\infty\le32\) also
cover the nonnegative principal matrix \(A_{RR}\). No obsolete scalar
exponential limit is needed here. The finite theorem itself permits exact
exponentials at any finite horizon. This global lifted comparison claims
no phase-chart or winding retention; those are separate obligations when
an application consumes them. A common capacity multiplier \(c>0\)
transforms both full and reduced rows by the same clock change \(t\mapsto ct\).

### 8.4 A three-node discriminator and prospective finite tolerances

Take nodes \(0,1,2\), old edge \(12\) of weight one, and contact edge
\(01\) of weight one when on and zero when off. Capacities are one and
\(\gamma=0\). Retain \(R=O=\{0,1\}\), so the predictor evolves four
primary coordinates instead of six. Phases start at zero and remain zero.
Use retained initial form \((0,0)\), hidden reference zero and hidden
form radius one, with zero phase radius. The reduced field gives the
zero solution on both branches.

Two admitted complete preparations have hidden form \(x_2(0)=\sigma\),
where \(\sigma\in\{-1,+1\}\), and identical retained data. Write
\(r=e^{-t/4}\). On the connected path the exact complete solution is
\[
 x_0=\sigma\frac{2-3r+r^3}{6},\qquad
 x_1=\sigma\frac{1-r^3}{3},\qquad
 x_2=\sigma\frac{2+3r+r^3}{6}.
 \tag{41}
\]
On the off branch, \(x_0=0\),
\(x_1=\sigma(1-r^2)/2\), and \(x_2=\sigma(1+r^2)/2\).
Substitution and differentiation verify all full diffusion rows. The
unchanged constituent means and their contact response are
\[
 m_{\rm on}=\sigma\frac{4-3r-r^3}{12},\quad
 m_{\rm off}=\sigma\frac{1-r^2}{4},\quad
 \mathcal R=\sigma\frac{(1-r)^3}{12}.\tag{42}
\]
Thus the same retained preparation and uncertainty class contain opposite
nonzero contact responses for every \(t>0\). No exact response sign or
value follows from those retained data alone. The predictor's zero response
can nevertheless be within a supplied absolute error. This is an
information-loss obstruction and a bounded approximation, not contradictory
claims about the same accuracy requirement.

Freeze \(h=1/4\), zero readout errors, per-branch tolerance \(1/20\)
and response tolerance \(1/10\) before evaluating the controls. These
have a direct analytical acceptance bound. On the joined path,
\[
 I-hA=\frac1{16}
 \begin{pmatrix}15&-1&0\\-1&14&-1\\0&-1&15\end{pmatrix}.
\]
Every coefficient of the positive series in (23) is at most one. Since
\(\|hA\|_\infty=1/4<1\), it follows that
\(P_h\le(I-hA)^{-1}\) entrywise. The latter's hidden column has
retained entries \((1/195,1/13)\). Its mean is
\[
 \mathcal E_{\rm a}^{\rm init}\le\frac8{195}
                   <\frac{19}{384}<\frac1{20}.\tag{43}
\]
The off comparison matrix is entrywise smaller, so it obeys the same bound.
Here \(c=0\), and both subsequent-motion allowances vanish. Summing
branch budgets certifies \(1/10\). These tolerances are much larger than
the small signal in (42); acceptance cannot be relabeled a resolved sign
or a relative-accuracy forecast of that interaction.

A companion keeps the same true initial state \((0,0,1)\) but sets
hidden reference one and hidden radii zero. Initialization error now
vanishes, while the omitted hidden row initially moves. On the on branch,
put \(z=\sqrt5\,t/8\); the exact reduced form solution is
\[
 \begin{aligned}
 \widehat x_0&=1-e^{-3t/8}
             \left(\cosh z+\frac3{\sqrt5}\sinh z\right),\\
 \widehat x_1&=1-e^{-3t/8}
             \left(\cosh z+\frac1{\sqrt5}\sinh z\right).
 \end{aligned}\tag{44}
\]
It solves \(\widehat x_0'=(\widehat x_1-\widehat x_0)/4\) and
\(\widehat x_1'=(\widehat x_0-2\widehat x_1+1)/4\).
On the off branch, \(\widehat x_0=0\) and
\(\widehat x_1=1-e^{-t/4}\). These are exact reduced solutions, not
integrated samples. For the additional observation at node zero,
(41)-(44) have leading difference \(-t^3/384\); a leading coefficient
alone is not a finite-time sign certificate.

In this companion, \(c_H=1/4\), \(c_R=(0,1/4)\), and
\(\|A_{RR}\|_\infty\le3/4\). The same geometric domination gives
\(\|K_R\|_\infty\le16/13\) and
\(A_{HR}K_Rc_R\le1/13\). Consequently the integrated hidden source
is at most \(1/16+1/416=27/416\). With (43), either branch has the
stronger analytical allowance
\[
 \mathcal E_{\rm a}\le\frac{27}{416}\frac{19}{384}
                  =\frac{513}{159744}<\frac1{200}.\tag{45}
\]
The frozen acceptance tolerances remain \(1/20\) and \(1/10\);
they are not adjusted after observing a response. This companion isolates
omitted drift and feedback from initialization uncertainty.

A separately frozen channel-isolation control uses true and reference
initial forms \((1,0,0)\), the same zero phases and hidden radii, and the
same supports and tolerances. Here \(q^0=0\) and \(c_H=0\), while
\(c_R=(1/4,1/4)\) on the on branch. Only the feedback term in (38) can
bound the later discrepancy. The full mean and reduced on solution are
\[
 m_{\rm on}=\frac{4+3r-r^3}{12},\qquad
 \widehat x_0=e^{-3t/8}\left(\cosh z+\frac1{\sqrt5}\sinh z\right),\qquad
 \widehat x_1=e^{-3t/8}\frac2{\sqrt5}\sinh z,
 \tag{46}
\]
with \(r,z\) as above. Both off means are exactly \(1/2\). The on
mean error has leading term \(+t^3/768\); any finite nonzero claim must
be checked from the complete expressions, not that onset alone. Thus
zero initial discrepancy and zero immediate hidden drift do not justify
deleting retained-driven feedback. The prospective declaration is retained
in `feedback-control.json` beside the scope record.

### 8.5 Shared field, consumer and prospective verification boundary

The reduced field is a state-evolving mathematical model with only its
retained coordinates variable. Its actual boundary references and support
are re-admitted from primitive data when constructed. The private
`_prepared_sine_retained_boundary` consumer belongs to the
[retained-boundary owner](../../src/tnfr/physics/_sine_retained_boundary.py).
It derives the initial complete-row enclosures, comparison matrices,
three nodal allowances and their separate arithmetic and inverse-capacity
projections. No cached report supplies missing primitive associations.
All raw domains precede sine or matrix work. It exposes a reduced field;
it neither executes an integrator nor evaluates a full source response.

The prospective scope and baseline are retained under
`artifacts/sine-retained-boundary-20261024/`. Independent controls are to
verify the complete analytic path states and reduced solutions, the two
initialization choices, fixed tolerances, disconnected hidden blocks,
positive-gamma rows, heterogeneous capacities, relabeling, clock scaling,
zero and tiny boundary weights, raw admission and unchanged observations.
No passing evaluation is asserted here before the controls are frozen and
run. A successful finite control would check implementation of this
conditional theorem, not establish universal compression, autonomous
formation, selected events, a physical measurement bridge or a uniquely
necessary TNFR constitutive law.

<a id="sine-response-information"></a>
## 9. Necessary initial information for a resolved contact response

Section 8 distinguishes an accurate absolute approximation from resolving
an interaction. Here the initial information itself has an exact accuracy
limit for a specified finite response. Within a declared class of noisy
individual-coordinate measurements, one associated hidden form measurement
can suffice while an equally precise measurement at another node cannot.
The result concerns the exact \(\gamma=0\) member of the complete held
`D/4` law. It is not a nonlinear extension of this information formula.

The [observation requirements](../MINIMAL_STRUCTURAL_DEGREES.md#11-specify-the-task-before-minimizing-its-observation)
already distinguish task sufficiency from reconstruction and require a
measurement class and cost. Their
[whole-future linear criterion](../MINIMAL_STRUCTURAL_DEGREES.md#37-snapshot-reconstruction-temporal-observability-and-closure-differ)
and the [exact linear observation owner](../../src/tnfr/mathematics/linear_observation.py)
address invariant output spaces and temporal observability. Likewise,
[hidden inventory](SINE_ENVIRONMENTAL_MEMORY.md#conserved-hidden-inventory)
need not determine visible response, and
[prior visible rates](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability)
can reconstruct hidden state under their separately normalized sine law.
The contribution here is a necessary-and-sufficient error radius for the
original finite contact response, an implementable forecast with separate
arithmetic error, and an explicit useful noisy-measurement comparison.
The elementary midpoint/range argument itself is not a new general
observability theorem.

### 9.1 Complete law, response and permitted information

Use a finite symmetric nonnegative support, strictly positive held
capacities \(D=\operatorname{diag}(\nu_i)\), and one common clock. The
complete law at \(\gamma=0\) is
\[
 x'=Gx,\qquad G=-DL/4,\qquad \theta'=0.\tag{47}
\]
Phases do not enter the requested form response. Their values are neither
invented nor measured for this calculation. This does not delete their row
from the model or make phases dispensable for another law or observation.
The signed form, capacity and clock retain the normalization of (2).

The off support is supplied explicitly. The on support adds one declared
nonnegative contact conductance, possibly on an existing pair. Both branches
start from the same complete form vector; there is no later source, event,
capacity change or clock change. The contact is a comparison of held models,
not a proof of its admission or occurrence. Let \(R\) be a proper retained
region containing both contact endpoints and the unchanged nonempty observed
constituent \(O\); put \(H=V\setminus R\). Retained initial forms are known.
For an arithmetic or actual inverse-capacity mean, set respectively
\[
 \ell_i={\mathbf1_{i\in O}\over |O|},\qquad
 \ell_i={\nu_i^{-1}\mathbf1_{i\in O}\over
                         \sum_{j\in O}\nu_j^{-1}}.
\]
Each query keeps its own observation, tolerance and readout uncertainty.
At a fixed horizon \(h\), the original on-minus-off response is exactly
\[
 \mathcal R_h(x^0)=a(h)x^0,\qquad
 a(h)=\ell^{\mathsf T}
       \bigl(e^{hG_{\rm on}}-e^{hG_{\rm off}}\bigr).\tag{48}
\]
Thus known retained forms contribute
\(\beta=\sum_{i\in R}a_i x_i^0\). This is a matched contrast: its two
branches use the same hidden preparation. Equation (48) retains that
correlation directly, whereas the sufficient branch bounds in (40) did
not assume cancellation between approximation errors.

The hidden prior is an independently supplied coordinate box
\(x_j^0\in[L_j,U_j]\). A permitted additional measurement selects a
hidden node label \(j\) and supplies
\[
 |y_j-x_j^0|\le\eta_j,\qquad \eta_j\ge0.
\]
The cost is one per distinct individually measured hidden coordinate,
with the declared error bound part of its specification. Node association
is necessary. There is no arbitrary encoded summary, a response-weighted
initial measurement, or an estimated coordinate obtained from the future
being predicted. Intersect the prior with \([y_j-\eta_j,y_j+\eta_j]\)
for each measured node. Reject an empty intersection. Unmeasured coordinates
retain their priors. Write the resulting posterior intervals as
\[
 [l_j,u_j],\qquad m_j=(l_j+u_j)/2,\qquad r_j=(u_j-l_j)/2.
 \tag{49}
\]
For retained nodes set \(m_i=x_i^0\), \(r_i=0\). These intervals describe
admitted deterministic uncertainty, not a probability distribution.

### 9.2 Exact information radius and coordinate cost

For the precisely declared data above, the complete attainable response
range is
\[
 \left[c-\rho,c+\rho\right],\qquad
 c=a m,\qquad \rho=\sum_{j\in H}|a_j|r_j.\tag{50}
\]
Indeed, the response is affine in each hidden coordinate. Choosing
\(x_j^0=m_j+r_j\operatorname{sign}(a_j)\) simultaneously attains the
upper endpoint, and reversing every sign attains the lower endpoint;
zero coefficients have no effect. Both complete initial preparations obey
the same retained data and measurement bounds. Every intermediate value
is attained because the box is convex and the response is linear.

Any scalar prediction \(q\) based only on these data has worst error
at least
\(\max(|q-(c-\rho)|,|q-(c+\rho)|)\ge\rho\). The center \(q=c\)
achieves equality. Therefore \(\rho\) is the exact conditional minimax
error among all such point predictors, including nonlinear functions of
the supplied data. This is stronger than an arbitrary paired lower bound:
it supplies both an attainable worst-case obstruction and an optimal
forecast for the stated class. It says nothing about minimizing the whole
state dimension, another horizon, an interval of responses or another law.

A useful selection corollary fixes a centered prior radius \(s_j\), a
noise radius \(\eta_j\) for each possible coordinate reading, and a
budget of \(k\) coordinate measurements. Over all admissible measurement
records, the largest posterior radius at a measured node is
\(\min(s_j,\eta_j)\); a centered record attains it. Consequently the
worst information radius after selecting nodes \(S\) is
\[
 \rho_{\rm worst}(S)=
 \sum_{j\notin S}|a_j|s_j+
 \sum_{j\in S}|a_j|\min(s_j,\eta_j).\tag{51}
\]
Selecting at most \(k\) nodes therefore amounts to selecting the largest
nonnegative gains
\(|a_j|[s_j-\min(s_j,\eta_j)]\). This is a result for individual
coordinates with the stipulated cost, not arbitrary linear measurements.
It need not reconstruct hidden forms whose coefficients vanish. A
numerical implementation may rank such gains only when its enclosures
justify the ranking; rounding coefficient midpoints does not prove exact
rank or information minimality. The consumer below evaluates a supplied
measurement set and does not implement a general selection optimizer.

### 9.3 Information error and computational error are separate

Suppose exact rational enclosures \(a_i\in[\underline a_i,\overline a_i]\)
have been computed from the admitted generators. Define
\[
 \begin{aligned}
 a_i^*&=(\underline a_i+\overline a_i)/2,&
 d_i&=(\overline a_i-\underline a_i)/2,\\
 \rho_-&=\sum_i r_i\min_{v\in[\underline a_i,\overline a_i]}|v|,&
 \rho_+&=\sum_i r_i\max_{v\in[\underline a_i,\overline a_i]}|v|,\\
 q^*&=\sum_i a_i^*m_i,&
 \delta&=\sum_i d_i|m_i|.
 \end{aligned}\tag{52}
\]
The minimum absolute value is zero if its interval contains zero.
Then \(\rho_-\le\rho\le\rho_+\),
\(|q^*-c|\le\delta\), and every admitted exact response satisfies
\[
 |\mathcal R_h-q^*|\le\rho_++\delta.\tag{53}
\]
The implementable forecast \(q^*\) is a rational scalar. The information
radius \(\rho\) does not include coefficient rounding; \(\delta\)
does. A large \(\rho_+\) alone can reflect a conservative enclosure,
while \(\rho_->\tau\) proves that no predictor using these initial
data can guarantee absolute true-response error at most \(\tau\).

If the requested comparison includes independently bounded readout error
\(\epsilon\), the sufficient forecast allowance is
\(\rho_++\delta+\epsilon\). A value at most the supplied tolerance
certifies that forecast; otherwise the sufficient check is unresolved.
The true-response impossibility statement still uses \(\rho_-\), not
an unproved lower contribution from readout noise. Strict separation of
zero by the resulting response interval can certify its sign. A nonzero
point forecast alone cannot do so. Initial coordinate noise in (49),
computational error in (52), and final readout error have different roles.

The exact rational matrix enclosure reuses the positive propagator of
Section 7. For each actual generator set
\[
 \alpha=\max_i(-G_{ii}),\qquad N=G+\alpha I\ge0.
\]
If \(S_N\le e^{hN}\le P_N\) are its Taylor partial sum and positive
upper bound, and \(b_-\le e^{\alpha h}\le b_+\), then
\[
 {S_N\over b_+}\le e^{hG}\le {P_N\over b_-}
 \quad\hbox{entrywise}.\tag{54}
\]
The shared scalar exponential bounds have \(b_->0\). Differences of
the independently enclosed on/off matrices give sound signed coefficients
in (48); neither matrix can be replaced by an unverified point exponential.
The zero-contact and zero-horizon cases have exact zero response after all
primitive and numerical domains are admitted.

The implementation restricts dimension to 24,
\(\alpha h\le1\), and \(h\|N\|_\infty\le32\), before numerical
work. These are computational resource restrictions, not restrictions on
the finite-dimensional theorem (50). It retains rational coefficients,
posterior intervals and forecast/error endpoints without a floating-point
underflow cutoff. No relative-accuracy claim for arbitrarily tiny contacts
follows from these absolute enclosures.

### 9.4 A prospective two-hidden-node discriminator

Take nodes \(0,1,2,3\), old edges \(12,23\) of weight one, and an
added on contact \(01\) of weight one. The off branch omits that contact.
All capacities are one; \(R=O=\{0,1\}\), retained initial forms are
zero, and hidden forms \(x_2^0,x_3^0\) lie independently in \([-1,1]\).
The observation is the same mean of nodes zero and one in both branches.
At \(\gamma=0\), no phase preparation is needed for this form response.
Write \(a_2,a_3\) for its two hidden coefficients in (48).

The following bounds are derived before numerical response evaluation.
Let \(b=e_0-e_1\) and \(\ell=(1/2,1/2,0,0)^{\mathsf T}\).
The generators satisfy \(G_{\rm on}-G_{\rm off}=-bb^{\mathsf T}/4\).
Duhamel's identity gives, for \(j=2,3\),
\[
 a_j={1\over4}\int_0^h p(h-s)q_{1j}(s)\,ds,\quad
 p(t)=\ell^{\mathsf T}e^{tG_{\rm on}}b,\quad
 q_{1j}(s)=(e^{sG_{\rm off}})_{1j}.\tag{55}
\]
The minus sign cancels because node zero is isolated off and
\(b^{\mathsf T}e^{sG_{\rm off}}e_j=-q_{1j}(s)\).

To bound \(p\), evolve the column
\(e^{tG_{\rm on}}\ell\), using symmetry of this unit-capacity path.
Its three successive edge differences \(d\) satisfy
\[
 d'=Td,\quad T={1\over4}
 \begin{pmatrix}-2&1&0\\1&-2&1\\0&1&-2\end{pmatrix},\quad
 d(0)=(0,1/2,0)^{\mathsf T},\quad p=d_0.
\]
The exponential of \(T\) is nonnegative with row sums at most one.
Since \(d'(0)=(1/8,-1/4,1/8)^{\mathsf T}\le(1/8)\mathbf1\),
\(p'(t)\le1/8\), hence \(p(t)\le t/8\). Factoring
\(e^{-t/2}\) leaves a nonnegative adjacency exponential whose first
path term gives \(p(t)\ge e^{-t/2}t/8\).

On the off three-node path, with \(r=e^{-s/4}\),
\[
 q_{12}(s)={1-r^3\over3},\qquad
 q_{13}(s)={2-3r+r^3\over6}.
\]
Thus \(q_{12}\le s/4\). Also \(q_{13}(0)=q'_{13}(0)=0\) and
\(q''_{13}=(3r^3-r)/32\le1/16\), so \(q_{13}\le s^2/32\).
In \(e^{sG_{\rm off}}=e^{-s/2}e^{s(G_{\rm off}+I/2)}\), the
nonnegative first and second path terms give respectively
\(q_{12}\ge e^{-s/2}s/4\) and
\(q_{13}\ge e^{-s/2}s^2/32\). Inserting these in (55) proves
\[
 e^{-h/2}{h^3\over768}\le a_2\le{h^3\over768},\qquad
 e^{-h/2}{h^4\over12288}\le a_3\le{h^4\over12288}.\tag{56}
\]
These bounds apply to the complete finite response, not just its first
nonzero Taylor coefficient. Both hidden coefficients are positive for
\(h>0\).

Freeze \(h=1/4\), coordinate noise \(\eta=1/100\), measurement
value \(1/2\), zero future readout error, and absolute response tolerance
\(\tau=1/1000000\). Since \(e^{-1/8}\ge7/8\),
\[
 {7\over393216}\le a_2\le{1\over49152},\qquad
 {7\over25165824}\le a_3\le{1\over3145728}.\tag{57}
\]
These inequalities discriminate three allowed initial-information cases:

- With no hidden measurement, the exact minimax radius is \(a_2+a_3\),
  which exceeds \(\tau\). No retained-only point predictor meets it.
- Measuring node three to the fixed accuracy leaves node two entirely
  unknown. Its posterior radius is \(a_2+\eta a_3>\tau\), so that
  one-coordinate choice still cannot meet the tolerance.
- Measuring node two instead gives its posterior
  \([49/100,51/100]\), while node three remains in \([-1,1]\).
  The exact optimal center is \(a_2/2\), and its information radius obeys
  \[
   \rho=\eta a_2+a_3\le{41\over78643200}<\tau.\tag{58}
  \]
  The entire exact response range is strictly positive:
  \[
   \mathcal R_h\ge{49\over100}a_2-a_3
       \ge{661\over78643200}>{1\over125000}>0.\tag{59}
  \]
  The actual rational forecast must additionally fit its computed
  \(\delta\) inside the remaining tolerance, as required by (53).

One coordinate measurement is therefore sufficient and zero is impossible
within the declared class, provided the separately checked computation
meets its allowance. The location of the measurement matters. This does
not mean that one arbitrary scalar summary must be sufficient.

For a separate fixed summary, suppose the exact hidden inventory
\(x_2^0+x_3^0=0\) is supplied. Unit capacities make it the hidden part
of the conserved total form; conservation does not identify its spatial
distribution. Both hidden preparations \((1,-1)\) and \((-1,1)\)
obey that same summary. Their opposite responses are
\(\pm(a_2-a_3)\), and the whole inventory-zero fiber has exact minimax
radius
\[
 a_2-a_3\ge{55\over3145728}>\tau.\tag{60}
\]
Thus this exact one-scalar inventory is insufficient even though the
noisy node-two coordinate succeeds. Inventory is an analytical comparator,
not an accepted substitute for an individual-coordinate input in the
consumer. A response-weighted scalar chosen from \(a(h)\) would change
the measurement class and would not prove the stated minimum.

### 9.5 Forecast and verification boundaries

This is a fixed-horizon forecast of one scalar response. It evolves zero
state coordinates, but computes full-support matrix information before
contracting it with independently supplied initial data. Static information
and computational work have not disappeared. This is neither an autonomous
reduced evolution law nor a demonstrated runtime speedup. Another horizon,
contact, observation, gamma value or changing support requires its own
admission and calculation. Approximate response sufficiency need not imply
full-state observability, chart retention, event admission or an NFR identity.

The shared matrix enclosure belongs to the
[comparison-flow owner](../../src/tnfr/mathematics/_comparison_flow.py).
The private `_prepared_sine_response_information` consumer in the
[response-information owner](../../src/tnfr/physics/_sine_response_information.py)
rebuilds both generators and the selected observation from primitive
support, capacity and contact data. It admits initial measurement
associations and posterior intervals before numerical work, then exposes
information-radius bounds, a rational forecast, its computational allowance
and a separate tolerance verdict. Cached report fields cannot supply
missing primitive premises.
The prospective baseline and scope are retained under
`artifacts/sine-response-information-20261024/`. Independent controls are
to verify the full analytical path source and finite response coefficients,
posterior clipping and signed box extrema, the frozen accuracy and sign
margin, numerical versus information radii, coefficient enclosure,
heterogeneous-capacity observation weights, association under relabeling,
common capacity-clock scaling, zero cases and primitive admission. They
must use independent rational exponential/hyperbolic bounds rather than
the production matrix kernel as the expected response.

No passing evaluation is asserted before the controls are frozen and run.
An unsuccessful computational tolerance check must remain unsuccessful
without retuning the fixed accuracy. A successful control would establish
implementation of this conditional information result; it would not select
a universal constitutive law, a minimal TNFR dimension, an autonomous event,
formation, a physical measurement bridge or a nonlinear-gamma extension.

<a id="sine-phase-information"></a>
## 10. Relative-phase information with active nonlinear feedback

The finite information result in Section 9 does not justify omitting phase
when it drives form. This section retains the complete sine law at the
fixed nonzero value \(\gamma=1/2\). A controlled linearization supplies
an explicit forecast, while a separately derived finite nonlinear remainder
bounds its error. On a declared small-phase preparation, one independently
supplied relative-phase reading resolves the original contact response;
no reading, or a reading at the other hidden node, remains insufficient.
The information radius of the tangent is exact. The corresponding nonlinear
information radius is only bounded, not silently identified with it.

The choice \(\gamma=1/2\) permits an explicit complete tangent propagator.
It is an analytical specialization, not a selection of a necessary TNFR
coupling coefficient. The
[existing hidden-state observer](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability)
uses prior visible rates under another normalization; the
[nonlinear memory result](ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
already explains why omitted initialization need not close. Here the
additional claim is useful finite response accuracy under the actual
weighted `D/4` sine law, with independently supplied relative-phase data
and an explicit nonlinear allowance. Neither phase dependence alone nor an
instantaneous difference would establish this claim.

### 10.1 Actual law, initial chart and observation class

For each matched held support put \(K=DL/4\), \(G=-K\). The actual law is
\[
 x'=-Kx+\frac D8S(\theta),\qquad
 \theta'=\frac K2x.\tag{61}
\]
Capacities are strictly positive and held; conductances are symmetric,
nonnegative and held; the clock is common. Both branches have exactly zero
initial forms and the same complete initial phases. They differ only by
the declared additive contact. There is no later forcing, event or parameter
change. The zero-form preparation is a premise, not an inferred property
of arbitrary retained data. The phase row in (61) remains active.

Keep the proper retained region \(R\), contact endpoints, original
constituent \(O\subseteq R\), and its normalized arithmetic or actual
inverse-capacity mean from Section 9. Choose a declared anchor in \(R\).
A common initial circular rotation has no effect on the requested form
response, since the law consumes only phase differences. Initial relative
phases \(\varphi_i\) are supplied in the anchored chart
\([-1/4,1/4]\) radians, with the anchor coordinate exactly zero.
Retained relative phases are known; hidden relative phases have independently
supplied interval priors in that chart. This gives one unambiguous initial
representative for each admitted relative angle. It does not reconstruct
an absolute phase origin.

An allowed extra reading selects a hidden node and measures its relative
angle to the same anchor, with value \(y_j\) and error \(\eta_j\).
The supplied measurement arc must satisfy \(|y_j|+\eta_j\le1/4\).
Intersect its interval with that node's prior, rejecting an empty result.
This contract uses a declared local circular chart; it does not silently
wrap an arbitrary large lift or select between disconnected circular
branches. A distinct relative-coordinate reading costs one, at its supplied
precision. Future rates and response-weighted encodings are unavailable.

Let \(m,r\) denote the complete posterior centers and radii, with zero
radii on known retained coordinates. Define the amplitude bound from the
*original* priors and known retained phases, before measurement clipping:
\[
 \delta=\max_i\sup_{\varphi_i\text{ in its original prior}}
                           |\varphi_i|.\tag{62}
\]
The initial chart and this bound are independent preparation information.
The finite flow proof below uses a common lifted representative to compare
with a tangent; it does not impose a new absolute-phase observable or infer
a later chart or identity from a measurement label.

### 10.2 Complete tangent, not a frozen-phase forcing

Linearize only the sine current about a common phase. With
\(y=(x,\varphi)\), the tangent is
\[
 y'=Jy,\qquad
 J=\begin{pmatrix}-K&-K/2\\K/2&0\end{pmatrix}.
 \tag{63}
\]
Let \(\mathcal K=\operatorname{diag}(K,K)\). The commuting matrix
\(J+\mathcal K/2\) squares to zero. Thus, writing
\(T_h=e^{-hK/2}\), the exact full tangent solution from arbitrary
initial \(x^0,\varphi^0\) is
\[
 \begin{aligned}
 x_{\rm lin}(h)&=T_h\bigl[(I-hK/2)x^0-(hK/2)\varphi^0\bigr],\\
 \varphi_{\rm lin}(h)&=T_h\bigl[(hK/2)x^0+(I+hK/2)\varphi^0\bigr].
 \end{aligned}\tag{64}
\]
Both rows are present. For the zero-form preparation, the phase-to-form
map is \(B_h=-(hK/2)T_h=(h/2)G e^{(h/2)G}\). The tangent prediction
of the original matched contact response is
\[
 \mathcal R_{\rm lin}(\varphi^0)=b\varphi^0,\qquad
 b=\ell^{\mathsf T}(B_{h,\rm on}-B_{h,\rm off}).\tag{65}
\]
This is a signed projection of the existing positive heat propagators,
not a new nonlinear identity. In particular, \(B_h\mathbf1=0\),
consistent with common-rotation invariance. The two supports keep their
actual capacities, conductances and matrices.

For each branch, enclose \(T_h\) using (54) at heat time \(h/2\),
then use the signs of the exact coefficients of \(G\) in the matrix
product. Midpoint matrix entries cannot replace an interval bound. The
posterior phase box gives an exact *tangent* range
\(c\pm\rho\), where \(c=bm\) and
\(\rho=\sum_i|b_i|r_i\), by the argument in (50). This calculation
has not yet bounded the actual nonlinear response.

### 10.3 A finite nonlinear remainder from both dynamic rows

For one branch set
\[
 k=\max_i\nu_i\sum_jw_{ij},\qquad
 \alpha=k/4,\qquad z=\alpha h,\qquad
 C=(1-z^2/2)^{-1}.\tag{66}
\]
The implemented preparation requires \(z\le1/2\), so \(C\) is
finite and positive. The argument itself applies whenever its denominator
is positive. The complete field is globally Lipschitz in the selected
finite-dimensional lifts, so the finite solutions and suprema below exist.

Subtract a fixed common phase origin and put
\(X(t)=\|x(t)\|_\infty\), \(\Theta(t)=\|\theta(t)\|_\infty\)
in that representative. The semigroup of \(-K\) is nonnegative and
contracts the infinity norm: it has zero generator row sums and nonnegative
off-diagonal entries. Since \(|\sin u|\le|u|\), the full form row
and the full phase row give respectively
\[
 X(t)\le\alpha\int_0^t\Theta(s)\,ds,\qquad
 \Theta(t)\le\delta+\alpha\int_0^tX(s)\,ds.\tag{67}
\]
For \(\Theta_* =\sup_{0\le t\le h}\Theta(t)\), substituting the
first inequality into the second yields
\(\Theta_*\le\delta+(z^2/2)\Theta_*\). Consequently
\[
 \Theta(t)\le\delta C\quad(0\le t\le h).\tag{68}
\]
A more detailed positive comparison gives
\(\Theta(t)\le\delta\cosh(\alpha t)\) and
\(X(t)\le\delta\sinh(\alpha t)\); the rational bound (68) is
sufficient here. It controls actual phase feedback, not a prescribed or
frozen phase history.

The difference between the actual form current and its tangent is
\(f(\theta)=D[S(\theta)+L\theta]/8\). The global inequality
\(|\sin u-u|\le|u|^3/6\), together with (68), implies
\[
 \|f(\theta(t))\|_\infty
 \le {k\delta^3 C^3\over6}=F_* .\tag{69}
\]
For example, each node has the factor
\(\nu_i\sum_jw_{ij}(2\delta C)^3/(8\cdot6)\); both the capacity
and every incident conductance remain in the estimate.

Compare actual and tangent solutions with the same complete initialization.
Let \(U=\|x-x_{\rm lin}\|_\infty\) and
\(V=\|\theta-\varphi_{\rm lin}\|_\infty\). Heat contraction and
the other tangent row give
\[
 U(t)\le\alpha\int_0^tV(s)\,ds+tF_*,\qquad
 V(t)\le\alpha\int_0^tU(s)\,ds.
\]
If \(U_*\) is the supremum over the window, then
\(U_*\le hF_*+(z^2/2)U_*\). Therefore
\[
 U_*\le E_{\rm branch}:={h k\delta^3 C^4\over6}.
 \tag{70}
\]
A positive normalized constituent form mean has at most the same error.
Apply this separately to the on and off supports, using their own \(k,C\),
and put \(E=E_{\rm on}+E_{\rm off}\). For every admitted common
initial phase preparation,
\[
 |\mathcal R(\varphi^0)-b\varphi^0|\le E.\tag{71}
\]
For zero horizon or zero contact conductance, the matched actual response
and tangent contrast are identically zero. After complete primitive and
domain admission, this identity permits the matched nonlinear allowance
to be set to zero, even if the individual branch bounds in (70) are positive
when a zero contact leaves nontrivial branch motion. For other admitted
cases, the sum above is the general bound used here.

The estimate includes nonlinear initialization and subsequent feedback.
It is uniform over the original phase prior, so the same bound covers all
posterior measurements and all information-obstruction witnesses. Equality
of the initial phases between branches alone does not cancel either
nonlinear allowance.

### 10.4 Actual information limits and implementable forecast

Let \(\rho_{\rm nl}^*\) be the optimal worst-case point-prediction
error for the *actual* contact response on the admitted posterior. Tangent
box corners attain values \(c\pm\rho\). Their actual responses can
each move by at most \(E\). Conversely every actual response lies
within \([c-\rho-E,c+\rho+E]\). It follows that
\[
 \max(0,\rho-E)\le\rho_{\rm nl}^*\le\rho+E.
 \tag{72}
\]
The tangent corners are sufficient witnesses for the lower bound. They
are not asserted to be the actual nonlinear extrema, nor is the actual
optimal center asserted to be \(c\).

Given certified coefficient intervals for \(b_i\), compute the tangent
radius bounds \(\rho_-,\rho_+\), rational midpoint forecast \(q^*\)
and computational-center allowance \(d\) exactly as in (52). Then
\[
 \rho_{\rm nl}^*\ge\max(0,\rho_--E),\qquad
 |\mathcal R-q^*|\le\rho_++E+d.\tag{73}
\]
If \(\max(0,\rho_--E)>\tau\), those data cannot guarantee the
required true-response accuracy. If
\(\rho_++E+d+\epsilon\le\tau\), the rational forecast meets the
requested tolerance including independent future readout error
\(\epsilon\). Otherwise the sufficient check is unresolved. Strict
sign certification uses the complete forecast interval. Tangent information,
nonlinear remainder, coefficient computation, initial measurement error and
future readout error retain their distinct meanings.

The prior-derived \(E\) is conservative. Failure of the sufficient test
does not prove that a particular forecast fails, and a positive lower bound
for the tangent does not by itself prove nonlinear insufficiency. Equation
(73), rather than the exact-linear verdict from Section 9, governs this law.

### 10.5 A useful finite relative-phase discriminator

Use the same four-node support and original constituent mean as Section 9:
old unit edges \(12,23\), added unit contact \(01\), unit capacities,
\(R=O=\{0,1\}\), and anchor zero. Both known retained relative phases
are zero, all initial forms are zero, and hidden phases have independent
priors \([-\delta,\delta]\). Freeze
\[
 h=1/4,\quad s=h/2=1/8,\quad \delta=1/1000,\quad
 \eta=1/100000,\quad y=1/2000,\quad \tau=1/1000000000,
 \quad\epsilon=0.\tag{74}
\]
Compare no hidden reading, a reading at node three, and a reading at node
two. These are relative-angle readings to the same anchor, with the same
noise and cost.

Let \(a_j(s)\) be the heat-response coefficients used in (55). By (64),
\(b_j=s a'_j(s)\). Differentiating the finite Duhamel formula, using
\(p(0)=0\), gives
\[
 b_j={s\over4}\int_0^s p'(s-u)q_{1j}(u)\,du.\tag{75}
\]
The earlier edge-gradient equation gives
\(p'=-p/2+d_1/4\). Its positive series has
\(d_1(t)\ge e^{-t/2}/2\), while \(p(t)\le t/8\). Thus
\[
 {1-t\over8}\le p'(t)\le1/8\qquad(0\le t\le1/8).
\]
The lower inequality uses \(e^{-t/2}\ge1-t/2\); the upper inequality
was proved in Section 9. Also the off-path transitions obey
\[
 {15\over16}{u\over4}\le q_{12}(u)\le{u\over4},\qquad
 {15\over16}{u^2\over32}\le q_{13}(u)\le{u^2\over32}
 \quad(0\le u\le1/8).
\]
Insert these bounds, and \(p'\ge7/64\), into (75). Direct integration
proves the finite positive brackets
\[
 {105\over16777216}\le b_2\le{1\over131072},\qquad
 {105\over1610612736}\le b_3\le{1\over12582912}.
 \tag{76}
\]
They are complete finite-time bounds, not a sign extrapolated from an
initial derivative.

Both supports have \(k=2\), so \(z=1/8\), \(C=128/127\), and
\[
 E={\delta^3\over6}\left({128\over127}\right)^4
       <1.73\mathbin{\cdot}10^{-10}.\tag{77}
\]
With the node-two measurement, its posterior is
\([49/100000,51/100000]\), while node three remains in
\([-1/1000,1/1000]\). The tangent information radius satisfies
\[
 \rho=\eta b_2+\delta b_3
 \le{1\over13107200000}+{1\over12582912000}
 <1.56\mathbin{\cdot}10^{-10}.\tag{78}
\]
Thus \(\rho+E<3.29\mathbin{\cdot}10^{-10}<\tau\), before
computational error. Moreover, every actual nonlinear response in this
posterior obeys
\[
 \mathcal R\ge{49\over100000}{105\over16777216}
          -{1\over1000}{1\over12582912}-E
          >2.8\mathbin{\cdot}10^{-9}>0.\tag{79}
\]
The implementable coefficient calculation must fit its own \(d\) into
the remaining tolerance and preserve this sign; the declared accuracy is
not retuned if it does not.

With no measurement, fix \(\varphi_3=0\) and compare
\(\varphi_2=\pm\delta\). With only the node-three measurement, fix
\(\varphi_3=y\) and use the same two node-two values. Each pair shares
all data available to that predictor. Its tangent half-separation is
\(\delta b_2\), so its actual nonlinear half-separation is at least
\[
 {1\over1000}{105\over16777216}-E
         >6\mathbin{\cdot}10^{-9}>\tau.\tag{80}
\]
Hence neither no reading nor that same-cost remote reading is sufficient.
One appropriate relative-phase coordinate is minimal within the stipulated
class, preparation and tolerance, once the rational forecast passes its
computational allowance. This is not a universal phase-count theorem.

The nonlinear sine current is genuinely different from its tangent on this
preparation. For example, at a positive node-two relative phase with other
phases zero, the edge \(12\) contributes
\((\sin\varphi_2-\varphi_2)/8\ne0\) to the difference of node-one
form rates. Also \(\theta'(0)=0\) follows only from the zero-form
initialization: differentiating the full phase row gives
\(\theta''(0)=Kx'(0)/2\), which is nonzero on this nonconstant phase
preparation. These static complete-row identities distinguish an active
phase response from a permanently frozen forcing; the finite guarantee
still comes from (67)-(80).

### 10.6 Implementation and prospective evidence boundary

The private `_prepared_sine_phase_information` consumer in the
[phase-information owner](../../src/tnfr/physics/_sine_phase_information.py)
rebuilds both support-associated matrices, the actual observation and the
anchored initial chart from primitive data. Shared contact/coordinate and
posterior admission belongs to the
[contact-information owner](../../src/tnfr/physics/_sine_contact_information.py).
The preceding form-information interface remains a separate exact
\(\gamma=0\) consumer. The shared
[Metzler enclosure](../../src/tnfr/mathematics/_comparison_flow.py)
supplies both tangent heat propagators; no new sine law or trajectory
integrator is installed.

Each numerical branch must satisfy the inherited matrix restrictions and
\(kh/4\le1/2\); every raw value, association, chart, intersection and
both branch domains precede kernel work or exact zero shortcuts. The
reported tangent information interval, nonlinear response allowance,
actual unavoidable-error lower bound, computational-center allowance and
complete forecast interval have separate meanings as in (73). An
unavailable or unsuccessful sufficient check is not a passing nonlinear
prediction.

This remains a fixed-horizon scalar forecast with full-support static
preprocessing. It supplies neither an autonomous reduced dynamical state
nor a demonstrated computational saving. No absolute phase reconstruction,
formation, event selection, empirical identification or transfer to another
nonzero gamma follows. Initial chart admission and a finite error theorem
are distinct from a universal collective identity.

The prospective scope and baseline are retained under
`artifacts/sine-phase-information-20261024/`. Independent analytical controls
are to verify the complete coupled tangent rows from independently
differentiated heat solutions, the finite coefficient brackets, actual
sine-versus-tangent current and nonzero phase acceleration, the nonlinear
comparison constants, permitted noisy relative readings and the frozen
accuracy/sign margins. They also cover posterior clipping, signed
projection, capacities and mean weights, clock and node association,
chart/raw rejection, zero cases and the unchanged form-information API.
No passing evaluation is asserted here before the reviewed controls run.
A finite control checks this conditional construction, not a uniquely
necessary constitutive law or unrestricted nonlinear information minimality.

<a id="sine-joint-information"></a>
## 11. Joint preparation geometry with both initial channels active

The separate form and phase results do not combine into a prediction for
arbitrary joint initial data. Section 9 fixes `gamma=0`; Section 10 fixes
all initial forms to zero. This section keeps the complete nonlinear law
(61), with `gamma=1/2`, and admits nonzero initial forms and relative phases.
The new distinction is between independently supplied joint preparation
relations and their coordinate margins. Two preparations can have the same
marginal intervals, uncertainty rank, volume and relation-record cost while
supporting different finite prediction guarantees.

This is not the first role for preparation information in the repository.
[Preparation-aware memory initialization](ORGANIZED_MEMORY_INITIALIZATION.md#organized-memory-initialization-result)
already distinguishes an admitted prepared image from a larger norm-containing
class under its own law and observation. The
[nonlinear constituent-memory result](ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
retains hidden initialization. Here the additional result is a finite
original contact-response discriminator under the held positive-capacity
`D/4` law, with nonzero initial form, explicit relative-phase relations and
a separately bounded nonlinear error. The critical full tangent (64) is
reused, not presented as a new law or derivation of preparation.

### 11.1 Independently supplied affine preparation

Keep fixed symmetric nonnegative conductances, strictly positive capacities,
one common clock and the complete two-row law (61). The matched on/off
branches have the same full initial preparation and differ only by one
additive nonnegative contact. There is no later input, event or coefficient
change. The original observable is the same positive normalized form mean
on a nonempty region `O` contained in a proper retained region `R` that
contains the contact endpoints. Arithmetic and inverse-capacity means use
their actual respective normalized weights.

Write the full initial preparation as
\[
 \begin{pmatrix}x^0\\\varphi^0\end{pmatrix}
 =m+Vu,\qquad u\in[-1,1]^q.\tag{81}
\]
Each row has an explicit `(node, form)` or `(node, relative_phase)`
association. The affine image is the admitted uncertainty set, not a
covariance matrix, an independence assertion about its physical coordinates,
or a statistical sample. The latent coordinates are independently bounded;
their images need not be independent. A point preparation has `q=0`.
Redundant columns do not invalidate the theorem, although their number is
not the dimension of the image or a measurement cost.

Retained form and phase coordinates are known: their rows of `V` vanish.
A declared retained phase anchor has zero phase center and loading. Require
\( |m_{\varphi i}|+\sum_j|V_{\varphi i,j}|\le1/4\) at every node.
Thus the entire supplied phase image lies in the initial anchored chart
from Section 10. A common circular rotation remains irrelevant. During
evolution the comparison subtracts that fixed initial origin, not a moving
anchor whose rate would alter the phase row.

The set in (81) must come from independently supplied preparation or
information. Its columns cannot be selected from the future response to
manufacture a narrow prediction. Accepting an affine input does not
authenticate how it was acquired, and a generic input need not be the exact
solution of an unstated collection of noisy linear measurements. The
explicit families below are proved equivalent to their stated preparation
bands. No general polytope inference is assumed.

A supplied `preparation_relation_cost` records a nonnegative integer cost
under the declared acquisition convention. It is conditional metadata,
not a cost theorem. Keep it separate from the latent column count, rank of
`V`, number of individual coordinate readings and number of evolved
coordinates. In the comparison below the two joint preparations each use
two independently specified noisy relation records; the larger marginal-only
class uses none. No acquisition-cost optimum is claimed.

### 11.2 Contract the joint image before bounding its response

For each branch put \(G=-DL/4\), \(s=h/2\) and
\(H=e^{sG}\). The already proved full critical tangent is
\[
 \begin{aligned}
 x_{\rm lin}(h)&=H[x^0+sG(x^0+\varphi^0)],\\
 \varphi_{\rm lin}(h)&=H[\varphi^0-sG(x^0+\varphi^0)].
 \end{aligned}\tag{82}
\]
Both initial channels and both evolution rows are retained. For a center
or one affine direction \(v=(v_x,v_\varphi)\), define
\[
 T_G(v)=\ell^{\mathsf T}H[v_x+sG(v_x+v_\varphi)],\qquad
 c=T_{G_{\rm on}}(m)-T_{G_{\rm off}}(m),\quad
 d_j=T_{G_{\rm on}}(V_j)-T_{G_{\rm off}}(V_j).\tag{83}
\]
The matched tangent response is \(c+\sum_jd_ju_j\). Its exact
attainable range and conditional minimax radius are
\[
 [c-\rho,c+\rho],\qquad \rho=\sum_{j=1}^q|d_j|.\tag{84}
\]
The proof is the box-extremum argument from Section 9 applied to the latent
coordinates. Opposite sign corners attain both endpoints, even if `V` has
redundant columns. They are complete preparations in the same admitted
image. The midpoint attains radius `rho`; no other point predictor based
only on the same data can reduce that worst error.

Exact affine contraction precedes absolute values. In particular,
\(v_\varphi=-v_x\) gives \(T_G(v)=\ell^{\mathsf T}Hv_x\).
Bounding the two contributions independently would discard this cancellation.
The relation identifies the critical tangent eigenspace; it is not fitted
to `h`, the original observation or its response coefficients. A gain of
one here belongs to the supplied normalized preparation coordinates.
It is neither a universal dimensional identification of form with phase
nor an invariant of the actual nonlinear sine law.

For rational enclosure, form
\(v_x+sG(v_x+v_\varphi)\) exactly before applying the shared
entrywise heat-kernel enclosure. Signed projection then encloses each
branch value and their difference. This preserves exact algebraic
cancellation before numerical interval widening. It does not assume
cancellation between the separate branch errors.

### 11.3 A nonlinear allowance for nonzero initial forms

The nonzero form initialization requires a new phase-growth bound. Define
\[
 B_0=\max_i\left(|m_{xi}|+\sum_j|V_{xi,j}|\right),\qquad
 P_0=\max_i\left(|m_{\varphi i}|+\sum_j|V_{\varphi i,j}|\right).
 \tag{85}
\]
These are uniform initial supremum-norm ceilings over the admitted affine
image. For each actual branch use
\(k=\max_i\nu_i\sum_jw_{ij}\), \(\alpha=k/4\),
\(z=\alpha h\le1/2\), and \(C=(1-z^2/2)^{-1}\).
Set \(X(t)=\|x(t)\|_\infty\) and
\(\Theta(t)=\|\varphi(t)\|_\infty\), with the fixed origin
specified above. Contractivity of the form heat flow and
\(|\sin a|\le|a|\) give
\[
 X(t)\le B_0+\alpha\int_0^t\Theta(r)\,dr,\qquad
 \Theta(t)\le P_0+\alpha\int_0^tX(r)\,dr.
\]
Substituting the first inequality in the second and taking the window
supremum yields
\[
 \Theta_*\le P_0+zB_0+{z^2\over2}\Theta_*,\qquad
 \Theta_*\le(P_0+zB_0)C.\tag{86}
\]
The term `z B0` cannot be omitted or borrowed from the zero-form result.
The law is globally Lipschitz in finite dimension with a linear form part
and bounded sine derivative, so these finite-window comparisons apply to
its complete solution.

The actual sine flow differs from (82) by form forcing
\(f=D[S(\varphi)+L\varphi]/8\). Since each phase gap is at most
\(2\Theta_*\), the universal cubic inequality
\(|\sin a-a|\le|a|^3/6\) gives
\(\|f\|_\infty\le k(P_0+zB_0)^3C^3/6\).
Let `U` and `W` be the supremum norms of the form and phase differences
between the actual and tangent solutions from the same initial state.
The complete difference rows give
\(U\le\alpha\int W+t\|f\|_*\) and
\(W\le\alpha\int U\). Therefore the same Volterra comparison gives
\[
 \|x(h)-x_{\rm lin}(h)\|_\infty
 \le E_G={hk\over6}(P_0+zB_0)^3C^4,\qquad
 E=E_{G_{\rm on}}+E_{G_{\rm off}}.\tag{87}
\]
Each positive normalized observation has norm one, so every actual matched
response satisfies \(|\mathcal R_h-\mathcal R_{h,\rm lin}|\le E\).
No independence between on/off nonlinear errors is needed. At zero horizon
or zero contact weight, the actual matched response and tangent contrast
are exactly zero; after all admissions the matched allowance can be set
to zero even when individual identical-branch allowances are positive.

Opposite tangent corners and (87) imply that the actual nonlinear
conditional minimax radius lies between \(\max(0,\rho-E)\) and
\(\rho+E\). The tangent corners need not be nonlinear extrema. If
\(c\in[c_-,c_+]\), \(d_j\in[d_{j,-},d_{j,+}]\), set
\[
 \begin{aligned}
 \rho_-&=\sum_j\min_{a\in[d_{j,-},d_{j,+}]}|a|,&
 \rho_+&=\sum_j\max_{a\in[d_{j,-},d_{j,+}]}|a|,\\
 q^*&=(c_-+c_+)/2,& e_c&=(c_+-c_-)/2,\\
 L_{\rm information}&=\max(0,\rho_--E),&
 E_{\rm forecast}&=\rho_++E+e_c+\epsilon_{\rm readout}.
 \end{aligned}\tag{88}
\]
A coefficient interval containing zero contributes zero to its minimum
absolute value. The rational `q*` is the implementable forecast. A lower
bound above the target proves insufficient information; a forecast
allowance at most the target certifies it; otherwise the outcome is
unresolved. The complete interval \(q^*\pm E_{\rm forecast}\)
determines any sign assertion. Intrinsic uncertainty, nonlinear
approximation, coefficient computation and independently bounded final
readout error remain distinct. A point preparation can still have a
positive sufficient nonlinear allowance without an intrinsic information
ambiguity.

### 11.4 Same coordinate margins, rank and volume; different association

Use the same four-node path: old unit edges `12,23`, contact `01` of
weight one on and zero off, unit capacities, `R=O={0,1}`, phase anchor
zero, and known zero retained forms and relative phases. Freeze
\[
 h={1\over4},\quad s={1\over8},\quad
 \delta={1\over10000},\quad \varepsilon={1\over1000000},\quad
 \tau={3\over10^{10}},\quad \epsilon_{\rm readout}=0.\tag{89}
\]
Let `u,v,w2,w3` be independently bounded in `[-1,1]`. Both joint
preparations have \(x_2=2\delta+\delta u\),
\(x_3=\delta v\). The linked preparation supplies
\[
 \varphi_2=-2\delta-\delta u+\varepsilon w_2,\qquad
 \varphi_3=-\delta v+\varepsilon w_3,\tag{90}
\]
and the crossed preparation supplies
\[
 \varphi_2=-2\delta-\delta v+\varepsilon w_2,\qquad
 \varphi_3=-\delta u+\varepsilon w_3.\tag{91}
\]
These are exactly the form bounds together with, respectively,
\[
 \begin{array}{ll}
 |\varphi_2+x_2|\le\varepsilon,&
 |\varphi_3+x_3|\le\varepsilon,\\
 |\varphi_2+x_3+2\delta|\le\varepsilon,&
 |\varphi_3+x_2-2\delta|\le\varepsilon.
 \end{array}\tag{92}
\]
For example, invert the linked relations by
\(u=(x_2-2\delta)/\delta\), \(v=x_3/\delta\),
\(w_2=(\varphi_2+x_2)/\varepsilon\),
\(w_3=(\varphi_3+x_3)/\varepsilon\); the crossed inverse uses its
second pair of residuals. Thus no additional feasible states are silently
lost. Each pair of relation rows is linearly independent and carries the
same two-record cost and width. The gain and offsets are supplied before
response evaluation; their provenance is a preparation premise.

In the row order `(x2,x3,phi2,phi3)`, the loading matrices are
\[
 V_{\rm linked}=
 \begin{pmatrix}
 \delta&0&0&0\\0&\delta&0&0\\
 -\delta&0&\varepsilon&0\\0&-\delta&0&\varepsilon
 \end{pmatrix},\qquad
 V_{\rm crossed}=
 \begin{pmatrix}
 \delta&0&0&0\\0&\delta&0&0\\
 0&-\delta&\varepsilon&0\\-\delta&0&0&\varepsilon
 \end{pmatrix}.\tag{93}
\]
Both have rank four, determinant \(\delta^2\varepsilon^2\) and
four-dimensional volume \(16\delta^2\varepsilon^2\). The same
center is \((2\delta,0,-2\delta,0)\), and their exact coordinate
radii are \((\delta,\delta,\delta+\varepsilon,\delta+\varepsilon)\).
All these margins lie well inside the supplied initial phase chart.
Replacing either joint image by its independent marginal box gives a
third full-rank preparation with those same margins and center but larger
volume. The linked/crossed comparison already removes volume, rank and
relation-record cost as explanations for the difference below. None of
these preparations reduces the number of uncertain physical coordinates.

### 11.5 Prospective finite discriminator

Let `a2,a3` be the hidden heat-contact coefficients at time `s`, and let
\(b_j=s a'_j(s)\) be the phase-to-form coefficients. The full tangent
response uses \((a_j+b_j)x_j+b_j\varphi_j\). Sections 9 and 10 give
finite analytical bounds, before any response evaluation:
\[
 \begin{array}{ll}
 {15\over6291456}\le a_2\le{1\over393216},&
 {15\over805306368}\le a_3\le{1\over50331648},\\
 {105\over16777216}\le b_2\le{1\over131072},&
 {105\over1610612736}\le b_3\le{1\over12582912}.
 \end{array}\tag{94}
\]
The heat bounds use \(e^{-s/2}\ge15/16\) in (56); the derivative
bounds are (76). All four coefficients are positive. All three preparation
classes have the same tangent center \(c=2\delta a_2\). Their exact
tangent information radii are
\[
 \begin{aligned}
 \rho_{\rm linked}&=\delta(a_2+a_3)+\varepsilon(b_2+b_3),\\
 \rho_{\rm crossed}&=\delta(a_2+2b_2-a_3-2b_3)
                         +\varepsilon(b_2+b_3),\\
 \rho_{\rm marginal}&=\delta(a_2+a_3+2b_2+2b_3)
                         +\varepsilon(b_2+b_3).
 \end{aligned}\tag{95}
\]
For the crossed case the two form-related latent coefficients are
\(\delta(a_2+b_2-b_3)>0\) and
\(\delta(a_3+b_3-b_2)<0\); (94) proves their signs. Taking their
absolute values gives the second row of (95), not the linked radius.

The common initial ceilings are \(B_0=3\delta\),
\(P_0=3\delta+\varepsilon\). Both supports have `k=2`,
`z=1/8` and `C=128/127`. Equation (87) gives
\[
 E={1\over6}\left({27\delta\over8}+\varepsilon\right)^3
              \left({128\over127}\right)^4
       <7\mathbin{\cdot}10^{-12}.\tag{96}
\]
For a simple rational check,
\(27\delta/8+\varepsilon<(17/5)\delta\) and
\(C^4<(101/100)^4<21/20\); their substitution gives
\(E<(103173/15000)\delta^3<7\delta^3\).
The same bound covers every corner and every interior preparation of all
three sets, without substituting an unavailable zero initial form.

For the linked image,
\[
 \rho_{\rm linked}\le
 {129\delta+388\varepsilon\over50331648}
       <2.65\mathbin{\cdot}10^{-10},\qquad
 \rho_{\rm linked}+E<2.72\mathbin{\cdot}10^{-10}<\tau.
 \tag{97}
\]
Its entire actual nonlinear response is strictly positive:
\[
 \mathcal R_h\ge
 {119\delta-388\varepsilon\over50331648}-E
       >2.2\mathbin{\cdot}10^{-10}>0.\tag{98}
\]
The actual numerical center and coefficient enclosures must still fit
their own computational allowance in (88). No tolerance or preparation is
retuned if that calculation fails.

For the crossed and marginal-only images, opposite tangent corners obey
their own same supplied data. Subtracting the uniform nonlinear remainder
from their tangent half-separation proves
\[
 \begin{aligned}
 L_{\rm crossed}&\ge{741\delta\over50331648}-E
                      >1.4\mathbin{\cdot}10^{-9}>\tau,\\
 L_{\rm marginal}&\ge{750\delta\over50331648}-E
                      >1.4\mathbin{\cdot}10^{-9}>\tau.
 \end{aligned}\tag{99}
\]
For the second line it suffices to keep the positive near-node terms
\(\delta(a_2+2b_2)\). These are actual nonlinear information
obstructions, not merely failed sufficient certificates. They do not
assert that the nonlinear extrema coincide with the tangent corners.
The allowed relation geometry and its node/channel association change
useful predictability even at equal joint-set rank, volume and record cost.

Both initial channels participate in the actual law. At the common center,
\(x=(0,0,2\delta,0)\), \(\varphi=-x\), direct full-row evaluation
gives at node one
\[
 \varphi'_1=-{\delta\over4}\ne0,\qquad
 (x+\varphi)'_1={2\delta-\sin(2\delta)\over8}>0.\tag{100}
\]
Thus the initial balanced relation is not preserved by the nonlinear sine
flow. A static field identity establishes this departure; it is not a
numerically integrated trajectory. The finite guarantee rests on (86)-(99),
which explicitly allow that nonlinear departure.

### 11.6 Implementation, evidence and remaining boundary

The private `_prepared_sine_joint_information` consumer in the
[joint-information owner](../../src/tnfr/physics/_sine_joint_information.py)
accepts primitive support, actual observation and fully associated affine
rows. The shared
[contact-information owner](../../src/tnfr/physics/_sine_contact_information.py)
owns support admission, signed heat projection and the existing forecast
budget arithmetic. The prior form and phase consumers retain their own
interfaces and premises; none acquires a broader law through this reuse.

Numerical admission restricts the support to at most 24 nodes and the affine
representation to at most 48 bounded columns. Every scalar, label, row,
chart, retained loading, cost and both branch domains is checked before
kernel work or a zero shortcut. The inherited matrix budgets and
`k h/4 <= 1/2` are explicit numerical restrictions. `uncertainty_dimension`
counts columns only; supplied relation cost, individual-coordinate reading
count zero and evolved-coordinate count zero have separate meanings.

This is one fixed-horizon scalar forecast with full-support static
preprocessing. A supplied joint preparation does not become an autonomous
reduced state, a maintained NFR identity, a universally preferred
form/phase relation or an acquired physical observation. Neither equal
rank nor equal volume establishes equivalent predictive information.
There is no runtime speedup claim, and a different law, contact, task,
clock or information class must be admitted independently.

The frozen scope and baseline are retained under
`artifacts/sine-joint-information-20261024/`. Prospective independent controls
are to verify both complete tangent rows from differentiated heat modes,
affine corners and relation-band equivalence, exact margins/rank/volume,
nonzero-form nonlinear comparison, actual departure from the initial
relation and the frozen accuracy/sign/insufficiency margins. They also cover
signed contraction before absolute values, capacities and actual means,
clock and association changes, charts, cost distinctions, zero cases and
raw rejection before kernels. No passing evaluation is asserted here before
the reviewed controls run. No nonlinear endpoint, trajectory, event,
retained-producer replay or empirical result is needed to define this
conditional comparison.

<a id="sine-information-handoff"></a>
## 12. Dynamically generated information carried into contact

The useful relations in Section 11 were supplied at the response origin.
This section instead starts from an independent form/phase preparation,
lets the same complete nonlinear law evolve on the old support, and carries
the entire reached state into the original contact comparison. The actual
reached image can support a useful forecast that fails after forgetting
only its cross-channel association. Every possible spatial form vector,
every possible spatial phase vector, and every coordinate margin are kept
in that coarser comparison.

This is generated geometry of a reachable uncertainty set, not a newly
measured correlation or a claim of formation of an NFR. No state coordinate
is reset at the switch. Retained coordinates that become uncertain remain
uncertain. The existing
[preparation-aware initialization](ORGANIZED_MEMORY_INITIALIZATION.md#organized-memory-initialization-result)
already carries information from an admitted recovery history under its own
law. The [preparation comparison](SINE_PREPARATION_ROBUSTNESS.md#sine-preparation-fixed-metric-certificate)
protects a supplied post-event family under different premises. Neither
provides the history or contact-time association established below. The
[formation contract](LOCALIZED_ORGANIZATION_GEOMETRY.md#common-pattern-formation-contract)
continues to distinguish such predictive information from an acquired
organized identity.

### 12.1 A complete two-stage history without reinitialization

Use the same complete `gamma=1/2` law (61), positive held capacities and
symmetric nonnegative conductances. Both branches first evolve on the old
support for a prescribed preparation duration \(\tau\). For the next
\(h\) units, the off branch keeps that support and the on branch adds
one declared nonnegative contact. All form and phase coordinates are
continuous at the switch. This specifies an intervention in a piecewise
held law, not spontaneous occurrence or execution of a registered operator.
There is no forcing, state reset, coefficient fit or source replay.

Keep the original observation \(\ell^{\mathsf T}x\) on `O` inside a
proper region `R` containing the contact endpoints. Its arithmetic or
inverse-capacity weights are normalized and nonnegative. Initial knowledge
is the fully associated affine family (81), with all consumed form and
phase coordinates present. Unlike the direct interface in Section 11,
initial and reached retained coordinates may be uncertain. A declared
retained phase anchor is zero only at the original preparation time. Its
fixed initial origin is used through both rows; a possibly uncertain
reached anchor is not silently reset or re-registered.

Put \(G_0=-DL_{\rm old}/4\), \(r=\tau/2\),
\(H_p=e^{rG_0}\), and, for a response branch,
\(s=h/2\), \(H_b=e^{sG_b}\). For \(q^0=x^0+\varphi^0\),
the complete critical tangent at handoff and at the final time is
\[
 \begin{aligned}
 q_p&=H_pq^0,& x_p&=H_p[x^0+rG_0q^0],&\varphi_p&=q_p-x_p,\\
 q_b&=H_bq_p,& x_b&=H_b[x_p+sG_bq_p],&\varphi_b&=q_b-x_b.
 \end{aligned}\tag{101}
\]
Consequently
\[
 x_b=H_b[H_px^0+(rG_0+sG_b)H_pq^0],\qquad
 \varphi_b=H_b[H_p\varphi^0-(rG_0+sG_b)H_pq^0].\tag{102}
\]
The same original affine parameter occurs in every factor. Independent
handoff form/phase boxes would change the family being predicted.

Equation (102) does not require the two graph generators to commute.
In form/phase block notation the critical propagator is a heat factor
times \(I+sG_bN\), where
\(N=\left(\begin{smallmatrix}1&1\\-1&-1\end{smallmatrix}\right)\)
acts on the two channels and \(N^2=0\). The mixed product vanishes,
but the order of the nodal heat factors remains \(H_bH_p\).
Equivalently, define the direct contact rows
\[
 a=\ell^{\mathsf T}(H_{\rm on}-H_{\rm off}),\qquad
 b=s\ell^{\mathsf T}(G_{\rm on}H_{\rm on}
                       -G_{\rm off}H_{\rm off}).
\]
The original-state response row is
\[
 \mathcal R_{\rm lin}
   =aH_px^0+(raG_0+b)H_p(x^0+\varphi^0).\tag{103}
\]
For zero preparation duration it reduces to the direct full tangent.
For the off history, the two same-generator propagators collapse to the
complete tangent at elapsed \(\tau+h\). These are algebraic consistency
checks, not substitutes for the on history with its different generator.

Apply (102) or (103) to the exact center and each original affine direction
before taking absolute values. The resulting latent response is again
\(c+\sum_jd_ju_j\), with exact tangent radius \(\rho=\sum_j|d_j|\).
Enclose its center and coefficients with the shared rational heat kernels,
then retain the information and computational distinction of (88).
No new `2n` exponential or nonlinear trajectory integrator is required.

### 12.2 A global nonlinear allowance through the switch

Let `B0,P0` be the original full affine marginal ceilings (85). On each
complete history take \(k\) to be the maximum receiving-capacity weighted
degree across its preparation and response supports. Write
\[
 T=\tau+h,\quad \alpha=k/4,\quad z=\alpha T\le1/2,\quad
 C=(1-z^2/2)^{-1}.\tag{104}
\]
The time-dependent homogeneous form propagator is a product of
row-stochastic heat propagators. It is therefore nonnegative and
supremum-norm contractive, including across the continuous-state switch.
The complete nonlinear rows still imply
\[
 X(t)\le B_0+\alpha\int_0^t\Theta(v)\,dv,\qquad
 \Theta(t)\le P_0+\alpha\int_0^tX(v)\,dv.
\]
The proof of (86) applies over the whole duration, giving
\(\Theta_*\le(P_0+zB_0)C\). The sine residual and both error
rows then give the complete-history form allowance
\[
 E_b={Tk\over6}(P_0+zB_0)^3C^4,\qquad
 E_{\rm history}=E_{\rm on}+E_{\rm off}.\tag{105}
\]
This compares the actual piecewise nonlinear histories with their composed
tangents from the same original state. It does not restart the error or
discard the uncertain prefix at handoff. Summing the branches conservatively
counts the common prefix twice; no cancellation of that error is assumed.

For every admitted original state,
\(|\mathcal R-\mathcal R_{\rm lin}|\le E_{\rm history}\).
Thus the actual information lower bound and sufficient point-forecast
allowance follow (88), using this `E`. Zero response duration or zero
contact weight makes the matched response exactly zero after all input and
numerical-domain admissions. It does not make the prepared state zero.

Separate prefix estimates are useful for identifying what is available at
handoff. With \(k_0\), \(z_p=k_0\tau/4\) and
\(C_p=(1-z_p^2/2)^{-1}\), set
\[
 \begin{aligned}
 P_H&=(P_0+z_pB_0)C_p,& B_H&=B_0+z_pP_H,\\
 E_{px}&={\tau k_0\over6}(P_0+z_pB_0)^3C_p^4,&
 E_{p\varphi}&=z_pE_{px}.
 \end{aligned}\tag{106}
\]
These uniformly bound the actual reached phase/form norms and the
prefix actual-minus-tangent form/phase errors. The phase error follows
by integrating its complete error row. A coordinate enclosure obtained
from the tangent plus these allowances is an outer enclosure. It is not
identified with an exact nonlinear coordinate margin or reached image.

### 12.3 What forgetting only cross-channel association means

Let \(S_\tau\) be the exact image of the original family under the
actual nonlinear preparation flow. Define
\[
 \mathcal B_\tau
   =\operatorname{proj}_x(S_\tau)
       \mathbin{\times}\operatorname{proj}_{\varphi}(S_\tau).
 \tag{107}
\]
Then \(S_\tau\subseteq\mathcal B_\tau\), and both have precisely
the same coordinate margins. More strongly, each entire form-vector
possibility and each entire phase-vector possibility is unchanged. Every
within-channel spatial constraint and allowed inventory is retained.
Only the association between the two channel vectors is forgotten.
The full coordinate-marginal product is a further enlargement.

A lower bound for the best contact forecast on \(\mathcal B_\tau\)
need not know its exact projection boundaries. If two complete original
preparations reach \(z^+=(x^+,\varphi^+)\) and
\(z^-=(x^-,\varphi^-)\), then
\[
 z^A=(x^+,\varphi^-),\qquad z^B=(x^-,\varphi^+)
       \quad\hbox{belong to }\mathcal B_\tau.\tag{108}
\]
This is exact membership using actual reached vectors, not an assertion
that a point in a computed outer box is reachable. The hybrids need not
belong to \(S_\tau\). Their future matched responses provide a
half-separation lower bound for every point predictor using only the
coarser channel-product information.

This comparison deliberately erases cross-channel history. It cannot prove
that a predictor retaining that common history is incapable of the forecast.
The latter consumes \(S_\tau\), not \(\mathcal B_\tau\).
Similarly, preserving a whole actual channel vector is stronger than
changing one phase coordinate independently: that change could violate a
conserved phase inventory and would not establish (108).

### 12.4 An independent initial family with a useful finite handoff

Take nodes `0,1,2,3`, old unit edges `12,23`, added unit contact `01`,
unit capacities, `R=O={0,1}` and original phase anchor zero. Freeze
\[
 \tau=h={1\over4},\quad r=s={1\over8},\quad
 \delta={1\over10000},\quad \varepsilon={1\over10^8},\quad
 \eta={1\over200000000}=5\mathbin{\cdot}10^{-9}.\tag{109}
\]
The final readout error is zero. Initially node zero has zero form and
relative phase. All six component coordinates below vary independently:
\[
 \begin{aligned}
 x_1^0&\in[-\varepsilon,\varepsilon],&
 x_2^0&\in[\delta,3\delta],&
 x_3^0&\in[-\delta,\delta],\\
 \varphi_1^0,\varphi_2^0,\varphi_3^0
       &\in[-\varepsilon,\varepsilon].
 \end{aligned}\tag{110}
\]
This initial family equals the product of its form and phase projections.
No cross-channel relation is supplied; extra relation cost is zero.
Its six-dimensional interior is nonempty. The complete smooth finite-time
flow is invertible on the component, so the actual reached image remains
full-dimensional. Both \(S_\tau\) and \(\mathcal B_\tau\) have
six uncertain dimensions. The result below does not arise by collapsing
uncertainty rank. It makes no volume-preservation or entropy claim.

Node zero stays zero during the old-support preparation. Node one generally
acquires form and phase uncertainty. Both coordinates are carried into the
contact comparison. Known or uncertain initial inventories remain their
actual conserved values during preparation; no mean projection is imposed.

For explicit finite bounds let \(p\) be the edge-gradient function
in (55) and \(q_{1j}(v)=(e^{vG_0})_{1j}\). For `j=1,2,3`,
Duhamel's identity gives the pulled-back heat row
\[
 A_j(s,r)=(a(s)H_p)_j
    ={1\over4}\int_0^s p(s-u)q_{1j}(u+r)\,du.\tag{111}
\]
The sign is positive: off node zero is isolated, so its difference from
the node-one transition entry is \(-q_{1j}\). In particular the direct
node-one heat coefficient has positive leading term \(s^2/64\).
The initial-phase and initial-form coefficients in (103) are respectively
\[
 B_j=(s\partial_s+r\partial_r)A_j,\qquad C_j=A_j+B_j.
 \tag{112}
\]
The same `r,s` act on both complete rows. This is not a changed clock or
an extra chain factor from differentiating one stage alone.

For \(0\le v\le1/4\), the earlier heat identities imply
\(q_{12}\le v/4\), \(q'_{12}\le1/4\),
\(q_{13}\le v^2/32\), \(q'_{13}\le v/16\).
All these functions and derivatives are nonnegative in this window.
Also \(0\le p'\le1/8\), \(p(v)\le v/8\) on the needed
edge-gradient window. At \(r=s=t=1/8\), integrating (111) yields
\[
 A_2\le{t^3\over192}={1\over98304},\qquad
 A_3\le{11t^4\over12288}={11\over50331648}.
\]
After substituting \(u=tv\), differentiate the integral along the
diagonal `r=s=t`. Applying the same positive bounds to every differentiated
factor gives \(B_2\le3t^3/192\) and
\(B_3\le44t^4/12288\). Therefore
\[
 B_2\le{1\over32768},\quad B_3\le{11\over12582912},\qquad
 C_2+C_3\le{2103\over50331648}.\tag{113}
\]
These bound the full finite response coefficients.

A finite lower bound for `C2` also follows without a numerical response.
Factor the shifted nonnegative heat series in `p` and `q12` in (111).
On `r=s=t` their exponential factors multiply to \(e^{-t}\), and
\(A_2(t,t)=e^{-t}F(t)\), where `F` has nonnegative coefficients
and leading term \(t^3/192\). Consequently
\(C_2=A_2+tA'_2\ge(4-t)e^{-t}t^3/192\).
Using \(e^{-1/8}\ge7/8\) proves
\[
 C_2\ge{217\over6291456}.\tag{114}
\]
The discarded positive series terms justify the inequality at the finite
time; a leading-order expansion alone would not.

For the remaining small initial coordinates, use \(q_{11}\le1\)
and \(|q'_{11}|\le1/2\). Equation (111) gives
\(A_1\le s^2/64\), \(|\partial_rA_1|\le s^2/128\)
and \(0\le\partial_sA_1\le s/32\). Thus at the fixed times
\[
 |B_1|\le{33\over65536},\qquad |C_1|\le{49\over65536},\qquad
 |C_1|+|B_1|+B_2+B_3\le{16139\over12582912}.\tag{115}
\]
The exact tangent center is `2 delta C2`. Its latent information radius
obeys
\[
 \rho\le{2103\delta\over50331648}
             +{16139\varepsilon\over12582912}.\tag{116}
\]
Every source has \(B_0=3\delta\), \(P_0=\varepsilon\) as its
common preparation ceilings. Over the complete two-stage history,
`T=1/2`, `k=2`, `z=1/4` and `C=32/31`. Hence
\[
 E_{\rm history}
 ={1\over3}(\varepsilon+3\delta/4)^3(32/31)^4
      <1.7\mathbin{\cdot}10^{-13},\qquad
 \rho+E_{\rm history}<4.2\mathbin{\cdot}10^{-9}<\eta.
 \tag{117}
\]
For example \(\varepsilon+3\delta/4<0.76\delta\) and
\((32/31)^4<8/7\) give the stated remainder bound by rational
arithmetic. Combining (113)-(116) with the center lower bound proves an
actual response above \(2.6\cdot10^{-9}\) throughout the family.
Retaining the same `C2` in center and radius gives the sharper lower bound
\(\delta[217/6291456-55/50331648]
 -16139\varepsilon/12582912-E_{\rm history}>3.3\cdot10^{-9}\).
The prospective acceptance uses the weaker positive margin and still
requires the computed coefficient and center enclosures to fit their own
allowance. No parameter or tolerance is retuned after evaluation.

### 12.5 An obstruction after erasing only channel association

For the two actual histories in (108), take \(x_2^0=2\delta\pm\delta\)
and every other initially uncertain coordinate at its center. Their
initial phase inventories are exactly zero, and their allowed form totals
are respectively \(3\delta\) and \(\delta\). Their full-channel
hybrids preserve these allowed form totals and zero phase total. They
therefore do not exploit an independently forbidden inventory.

Let \(v=r e^{-3r/4}/4\). The preparation tangent half-difference is
\(\delta f\) in form and \(\delta g\) in phase, where
\[
 f=(0,q_{12}(r)+v,q_{22}(r)-2v,q_{12}(r)+v),\qquad
 g=(0,-v,2v,-v).\tag{118}
\]
Here \(q_{22}(r)=(1+2e^{-3r/4})/3\). At `r=1/8`,
\(e^{-3r/4}\ge1-3r/4=29/32\) gives
\[
 f_1=f_3\ge{29\over512},\qquad f_2\ge{7\over8},\qquad
 {29\over1024}\le v\le{1\over32}.\tag{119}
\]
Use lower-case `c=a+b` and `b` for the *direct future contact* form and
phase rows, not the pulled-back `C,B` of (112). Their entries at nodes
one, two and three are positive. The same shifted positive series at node
one give
\[
 a_1\ge{15\over65536},\qquad b_1\ge{465\over1048576},\qquad
 c_1\ge{705\over1048576},\qquad c_2\ge{435\over50331648}.
 \tag{120}
\]
For example, \(a_1=e^{-s/2}F_1(s)\) has positive series coefficients
starting with \(s^2/64\), so
\(b_1=sa'_1\ge(2-s/2)e^{-s/2}s^2/64\). The final `c2`
bound is the sum of the already proved direct near-node heat and phase
bounds.

Swapping the whole phase channel changes its sign in the hybrid
half-difference. Its future tangent response is
\(\delta(c\cdot f-b\cdot g)\). Retain the positive `c1`, `c2`
and `b1` contributions, bound the adverse `2 b2 v` term using
\(b_2\le1/131072\), and discard only the other positive terms.
Equations (119)-(120) yield
\[
 {\Delta\mathcal R_{\rm hybrid,lin}\over2}
 \ge\delta\left({54375\over1073741824}
                +{3045\over402653184}-{1\over2097152}\right).
 \tag{121}
\]
The negative last term is necessary. Swapping just the node-one phase and
ignoring its companion phase changes would not establish this bound for
the channel-product class.

Prefix uncertainty and the actual nonlinear future must still be included.
For this family (106) gives
\[
 P_H<0.38\delta,\quad B_H<3.05\delta,\quad
 E_{px}<5\mathbin{\cdot}10^{-15},\quad
 E_{p\varphi}\le E_{px}/8.\tag{122}
\]
All actual reached and hybrid states obey the same handoff norm ceilings.
Each direct contact form tangent map has norm at most `1+z`, and its
phase-to-form map at most `z`, where `z=1/8`. Thus replacing the actual
hybrids by their preparation tangents can reduce their future tangent
half-separation by at most
\[
 \|c\|_1E_{px}+\|b\|_1E_{p\varphi}
 \le\left(2(1+1/8)+{2\over8^2}\right)E_{px}
 ={73\over32}E_{px}.\tag{123}
\]
Apply the full direct nonlinear contact comparison to the hybrid ceilings
in (122), not the original zero/small phase ceiling. The sum of its two
branch allowances is below
\[
 E_{\rm hybrid}\le{1\over6}(P_H+B_H/8)^3(128/127)^4
          <8\mathbin{\cdot}10^{-14}.\tag{124}
\]
Each hybrid response lies within this amount of its direct contact tangent.
Their actual half-separation is consequently bounded below by (121) minus
\(73E_{px}/32+E_{\rm hybrid}\). In particular,
\[
 \inf_{\widehat R}\sup_{z\in\mathcal B_\tau}
       |\mathcal R_h(z)-\widehat R|
       >5.7\mathbin{\cdot}10^{-9}>\eta.\tag{125}
\]
The same lower bound applies to the still larger coordinate-marginal class.
No nonlinear endpoint or exact marginal extremum has been computed in this
argument; exact hybrid membership and uniform complete-law error estimates
suffice.

Equations (117) and (125) establish generated predictive association.
The original family has no cross-channel restriction beyond its separate
boxes, but its actual evolved image admits an accuracy that its exact
channel-product enlargement cannot attain. The comparison keeps all
within-channel preparation information and uses a nonzero contact-response
margin. It is neither trivial relaxation toward zero nor an assertion that
an exact balanced equality has become invariant.

### 12.6 Implementation and prospective evidence boundary

The private `_prepared_sine_history_information` consumer in the
[history-information owner](../../src/tnfr/physics/_sine_history_information.py)
accepts original primitive affine data and a preparation duration. It uses
the unchanged old support for preparation and constructs the prescribed
contact branches itself. It does not accept a cached reached report as
new primitive state, invent known retained coordinates, or erase an
uncertain phase origin. The direct
[joint-information owner](../../src/tnfr/physics/_sine_joint_information.py)
keeps its existing exactly-known-retained-state contract.

Shared affine admission, critical comparison and signed heat projections
retain their original units and support association. The original center
and every affine direction continue through both full tangent rows before
latent absolute values are taken. Global history error, prefix form/phase
error, reached-state outer enclosures and coefficient computation have
separate meanings. In particular, the numerical consumer does not certify
that an outer marginal box equals either exact projection in (107).

The numerical domain retains at most 24 nodes, at most 48 affine columns,
shared matrix resource limits and `k*(tau+h)/4 <= 1/2` for each full branch.
All original labels, scalars, charts, costs and both stage/branch domains
precede matrix work or response-zero shortcuts. Full-support static
preprocessing remains; no evolved reduced state or runtime saving follows.

The prospective scope and baseline are retained under
`artifacts/sine-information-handoff-20261024/`. Independent controls are
to verify complete differentiated preparation and response tangent rows,
noncommuting order and off-branch collapse, all original affine corners,
finite Duhamel bounds, full-rank independent preparation, actual-hybrid
membership deductions, the distinct error allowances, retained uncertainty,
means and fixed chart origin, capacities, clocks, labels, zero cases and
raw rejection before matrix work. Actual nonlinear field/jet checks retain
both active rows. The controls do not replace the theorem by sampled
nonlinear trajectories. No passing evaluation is asserted before the
reviewed controls run, and no physical acquisition or autonomous NFR
formation claim is made.


<a id="sine-coupling-robustness"></a>
## 13. Finite coupling uncertainty beyond the critical factorization

The critical factorization in Sections 10-12 is an exact analytical aid,
not a constitutive selection. This section admits an independently declared
interval of coupling coefficients and asks whether one forecast can still
resolve the same contact task. The coupling is unknown to the predictor,
but one held value governs the complete preparation and both response
branches. It is not chosen separately for different times, coordinates,
initial states within a trajectory, or the on/off comparison.

The general-gamma [matched bridge bound](ORGANIZED_CONTACT_MAINTENANCE.md#sine-matched-bridge-interface)
and [support-associated comparison](#sine-contact-influence) already bound
finite differences under their own premises. They do not imply useful
accuracy for the present small response. The
[preparation perturbation theorem](SINE_PREPARATION_ROBUSTNESS.md#sine-global-preparation-comparison)
also retains its distinct normalized law and source. Here the question is
finite robustness of the generated association in Section 12, using its
unchanged history, source and observation. A successful calculation would
remain conditional mathematics, not evidence for a unique fundamental law.

### 13.1 A common unknown coefficient in both full rows

For each stage let \(G=-DL/4\). The actual complete law is
\[
 x'=Gx+{\gamma D\over4}S(\varphi),\qquad
 \varphi'=-\gamma Gx,\qquad
 \gamma\in[\gamma_-,\gamma_+].\tag{126}
\]
Capacities are positive and held; conductances are symmetric, nonnegative
and held within each stage. The initial affine family, original phase
origin and continuous-state support switch are those of Section 12.
Both rows remain active. The prescribed preparation/contact durations and
added edge are independent premises, not an occurrence law or committed
glyph execution. Initial phase charts do not assert a fixed later anchor.

The numerical comparison interval contains \(\gamma_*=1/2\). Put
\(\eta_\gamma=\max(|\gamma_--1/2|,|\gamma_+-1/2|)\) and
\(g=\gamma_+\). One point prediction must cover every admitted initial
state and every held coupling in that interval. A critical calculation is
only a reference. Its reached-state enclosures are not silently renamed
actual reached-state enclosures for other couplings.

The complete tangent about common phase has channel matrix
\[
 M_\gamma=\begin{pmatrix}1&\gamma\\-\gamma&0\end{pmatrix},\qquad
 J_\gamma(G)=\begin{pmatrix}G&\gamma G\\-\gamma G&0\end{pmatrix}.
 \tag{127}
\]
It satisfies
\[
 M_\gamma^2=M_\gamma-\gamma^2I,\qquad
 (M_\gamma-I/2)^2=(1/4-\gamma^2)I.\tag{128}
\]
The nilpotent identity used at the critical value therefore does not hold
throughout an interval. The power recurrence (128) remains regular on both
sides of that value and avoids a singular eigenvalue formula.

The exact tangent history is the ordered product
\(e^{hJ_\gamma(G_b)}e^{\tau J_\gamma(G_0)}\).
Different nodal generators need not commute. Since the *same* channel
matrix occurs at both stages, each term of degree \(n\) in response time
and degree \(m\) in preparation time has channel factor
\(M_\gamma^nM_\gamma^m=M_\gamma^{n+m}\), while its nodal factor
stays in the order \(G_b^nG_0^m\). Thus, for the original normalized
form observation, define the exact signed nodal row
\[
 W_{nm}={h^n\tau^m\over n!m!}
       \ell^{\mathsf T}(G_{\rm on}^{n}-G_{\rm off}^{n})G_0^m.
 \tag{129}
\]
Writing \(e_x=(1,0)^{\mathsf T}\), the matched tangent response is
\[
 \mathcal R_\gamma^{\rm lin}(x^0,\varphi^0)
 =\sum_{n,m\ge0} W_{nm}
    \bigl[(M_\gamma^{n+m})_{00}x^0
         +(M_\gamma^{n+m})_{01}\varphi^0\bigr].\tag{130}
\]
The absolutely convergent series includes both complete rows and the full
uncertain handoff. The \(n=0\) terms vanish. In the fixed internal-contact
example, \(\ell^{\mathsf T}(G_{\rm on}-G_{\rm off})=0\), so
all \(n=1\) terms vanish as well. These signed response cancellations must
be retained before replacing quantities by norms.

### 13.2 A finite polynomial and a uniform omitted-series bound

Retain total degrees \(n+m\le N\), with fixed numerical \(N=16\).
Substitute \(\gamma=1/2+e\). Apply the resulting response to the
original exact affine center and every original generator column. After
summing all nodal and degree contributions, write the scalar polynomials
\[
 Q_j(e)=\sum_{r=0}^N q_{jr}e^r,\qquad
 \mathcal R_{\gamma,N}^{\rm lin}(m+Vu)
       =Q_0(e)+\sum_{j=1}^{q}Q_j(e)u_j.\tag{131}
\]
Index zero carries the known center; other indices carry independently
bounded latent coordinates. They are not newly measured observables.
Both form and phase of each column remain associated through both stages.
In particular, absolute values are taken *after* summing the coefficients
of each complete polynomial, not separately by path, stage or total degree.

For \(|u_j|\le1\) and \(|e|\le\eta_\gamma\),
\[
 |\mathcal R_{\gamma,N}^{\rm lin}-\mathcal R_{1/2,N}^{\rm lin}|
 \le P_N:=\sum_{r=1}^N\eta_\gamma^r
       \left(|q_{0r}|+\sum_{j=1}^{q}|q_{jr}|\right).\tag{132}
\]
This is a sufficient uniform bound. It is not an exact parameter minimax
radius, and evaluating only the two coupling endpoints would not prove it.
The same unknown coefficient \(e\) occurs in all factors and all powers.

Let \(k\) bound receiving-capacity weighted degree on all stages and
branches, \(T=\tau+h\), and \(B=\max(B_0,P_0)\), where the
original affine ceilings are defined in (85). In the full-state supremum
norm,
\[
 \|J_\gamma(G)\|_\infty\le L={(1+g)k\over2},\qquad z_s=LT.
\]
Indeed \(\|G\|_\infty\le k/2\); both channel rows are included.
The sum of the absolute bounds for all ordered product terms of total
degree \(p\) is at most
\((Lh+L\tau)^p/p!=z_s^p/p!\), by the binomial identity.
No commutation of stage generators is used in this bound.

For \(p\ge1\) and \(0\le z<p+1\), define
\[
 R_p(z)={z^p\over p!\,[1-z/(p+1)]}.
\]
The factorial ratio bounds successive omitted terms by a geometric series,
so \(\sum_{j\ge p}z^j/j!\le R_p(z)\). Each branch's omitted
state has norm at most \(B R_{N+1}(z_s)\). Subtracting on/off
responses, then comparing gamma with the critical member, gives
\[
 |\mathcal R_\gamma^{\rm lin}-\mathcal R_{1/2}^{\rm lin}|
 \le E_{\rm par}:=P_N+4B R_{N+1}(z_s).\tag{133}
\]
The critical member obeys the same majorant since it is in the interval.
A normalized arithmetic or actual inverse-capacity observation has norm
one, so no extra observation dimension factor appears.

At a singleton critical interval the difference is identically zero.
At zero contact weight or zero response duration the entire matched
response is zero. These identities may set the parameter allowance to
zero after all raw and numerical-domain admissions; they do not erase a
nonzero preparation or bypass invalid data. For nonzero interval width,
the fixed omitted-series allowance is numerical conservatism, not a
statement about a discontinuity of the actual response.

### 13.3 A fresh nonlinear comparison on the complete history

The tangent perturbation bound alone does not justify a nonlinear forecast.
For each full piecewise branch set
\[
 \alpha={gk\over2},\quad z_g=\alpha T\le1/2,\quad
 C=(1-z_g^2/2)^{-1},\quad P_*=(P_0+z_gB_0)C.\tag{134}
\]
Products of form heat propagators remain supremum-norm contractions.
The actual rows (126), using \(|\sin v|\le|v|\), imply
\[
 X(t)\le B_0+\alpha\int_0^t\Theta(s)\,ds,\qquad
 \Theta(t)\le P_0+\alpha\int_0^tX(s)\,ds.
\]
Substitution and the bound on the double integral give
\(\Theta_*\le P_*\) and \(X_*\le B_0+z_gP_*\), through
the support switch without restarting either state or error.

For the actual-minus-tangent error, \(|\sin v-v|\le|v|^3/6\)
gives the form forcing ceiling
\(F_*\le gkP_*^3/3\). Its form and phase sup norms satisfy
\(U\le tF_*+\alpha\int W\) and
\(W\le\alpha\int U\), with zero original error. Hence
\[
 E_b={Tgk\over3}(P_0+z_gB_0)^3C^4,\qquad
 E_{\rm nl}=E_{\rm on}+E_{\rm off}.\tag{135}
\]
This compares each actual general-gamma history with its own tangent,
not with the critical law. It bounds both active error rows on the original
whole duration. At \(g=1/2\) it recovers (105); that recovery is a check,
not the premise of the general derivation.

Prefix estimates follow by replacing \(T,k\) with \(\tau,k_0\):
\[
 P_H=(P_0+z_pB_0)C_p,\quad B_H=B_0+z_pP_H,\quad
 E_{px}={\tau gk_0\over3}(P_0+z_pB_0)^3C_p^4,\quad
 E_{p\varphi}\le z_pE_{px}.\tag{136}
\]
They apply uniformly to actual reached states at every admitted coupling.
The prefix bounds stored inside a critical reference still have their
critical meaning; (136) supplies the separate general-law quantities.

Let \(\widehat c_*\) be the rational point forecast of the critical
tangent center, \(\rho_{*,+}\) an upper bound on its exact original
latent radius, and \(\delta_*\) its center computation allowance.
For every held gamma and source state,
\[
 |\mathcal R_\gamma-\widehat c_*|
 \le \rho_{*,+}+\delta_*+E_{\rm par}+E_{\rm nl}.
 \tag{137}
\]
Add the independently supplied future reading error only to the observed
forecast. The critical nonlinear allowance is replaced by (135), not
silently retained as the general-gamma model error or counted twice.
Parameter variation includes the center as well as every latent direction.
A reference information lower bound, if retained, concerns an admitted
critical subfamily; it is not an exact minimax evaluation of the union.

### 13.4 Uniform comparison after forgetting channel association

For each *fixed* gamma let \(S_{\tau,\gamma}\) be the actual reached
image of the unchanged original family. Define
\[
 \mathcal B_{\tau,\gamma}
 =\operatorname{proj}_x(S_{\tau,\gamma})
       \times\operatorname{proj}_{\varphi}(S_{\tau,\gamma}).\tag{138}
\]
The unknown-law comparison is the indexed union of these same-gamma products.
The coupling label remains attached to each admitted state: its possibilities
are pairs `(gamma, z)`, with the corresponding response `R_gamma,h(z)`.
It does not authorize combining the form reached at one coupling with a
phase reached at a different coupling. As in (107), each product retains
all exact within-channel spatial possibilities, inventories and coordinate
margins. No computed outer interval substitutes for exact membership.

Use the same two original preparations \(x_2^0=2\delta\pm\delta\)
with the other uncertain coordinates at their centers. At each gamma,
their actual whole-channel hybrids belong to (138) by construction.
Their tangent half-difference is obtained by applying
\(\mathsf S=\operatorname{diag}(1,-1)\) to the two channels at
handoff. Consequently the future tangent half-gap has expansion
\[
 H_\gamma^{\rm lin}
 =\delta\sum_{n,m\ge0}(W_{nm}e_2)
       (M_\gamma^n\mathsf S M_\gamma^m)_{00}.\tag{139}
\]
The reflection lies between the stages. It cannot be removed or replaced
by \(M_\gamma^{n+m}\). Doing so would predict the reachable joint
pair instead of the admissible channel hybrids.

Aggregate (139) through total degree 16 into its exact polynomial in
\(e=\gamma-1/2\), then use the absolute nonconstant coefficient
bound as in (132). The norm of the inserted reflection is one, so the
uniform gamma-minus-critical omitted-series allowance is
\(4\delta R_{17}(z_s)\). Denote the resulting complete tangent
variation bound by \(E_{\rm par,hyb}\). This establishes a uniform
inequality, rather than an endpoint sample or a continuity existence claim.

The actual prefix must still replace the tangent prefix. A general-gamma
future tangent with initial errors \(U_0,W_0\) has form error at most
\(C_h(U_0+z_hW_0)\), by the same heat/Volterra argument, where
\(z_h=gk_hh/2\) and \(C_h=(1-z_h^2/2)^{-1}\). Therefore the
matched hybrid half-gap loses at most
\[
 E_{\rm transfer}\le2C_h(E_{px}+z_hE_{p\varphi}).\tag{140}
\]
The bound applies to half-differences of the two actual prefix errors and
to their whole-channel reflection. It does not erase a retained coordinate
or assume that a sparse phase change preserves the inventory.

Finally apply (135) over the future duration \(h\), starting from the
actual hybrid ceilings \(B_H,P_H\) in (136), to obtain a matched
future nonlinear allowance \(E_{\rm nl,hyb}\). If \(H_{*,\rm low}\)
is the critical tangent half-gap lower bound (121), then at every gamma
\[
 \inf_{\widehat R}\sup_{z\in\mathcal B_{\tau,\gamma}}
    |\mathcal R_{\gamma,h}(z)-\widehat R|
 \ge H_{*,\rm low}-E_{\rm par,hyb}
                   -E_{\rm transfer}-E_{\rm nl,hyb}.\tag{141}
\]
This pointwise obstruction is stronger than the unknown-union obstruction,
which alone could follow by retaining the admitted critical member. Neither
argument gives an obstruction to a predictor that retains the common
history. No actual nonlinear endpoint or projection extremum is evaluated.

### 13.5 Fixed finite acceptance and implementation boundary

Retain exactly (109)-(110), with the same old support, added contact,
positive capacities, observation, original source, clocks and target
\(5\cdot10^{-9}\). Freeze the additional constitutive interval
\[
 \gamma\in[49/100,51/100],\qquad \eta_\gamma=1/100.\tag{142}
\]
This is a finite supplied range on both sides of the critical value.
It is not a coupling inferred from the response, a neighborhood whose
width remains unspecified, or a license to retune the prior accuracy.

Here \(g=51/100\), \(k=2\), \(T=1/2\), so
\(z_g=51/200\), \(C=80000/77399\), and \(z_s=151/200\).
The inequalities
\(P_0+z_gB_0<0.77\delta\) and \(C^4<8/7\) give
\[
 E_{\rm nl}<1.8\cdot10^{-13}.\tag{143}
\]
For the prefix, \(z_p=51/400\). Direct rational substitution in
(136), followed by the future comparison, gives
\[
 \begin{gathered}
 P_H<0.39\delta,\quad B_H<3.05\delta,\quad E_{px}<5\cdot10^{-15},\\
 E_{\rm transfer}<1.1\cdot10^{-14},\qquad
 E_{\rm nl,hyb}<8.4\cdot10^{-14}.
 \end{gathered}\tag{144}
\]
For example \(P_H+z_hB_H<0.78\delta\) suffices for the last
bound. These constants concern the actual general-gamma histories and
hybrids, with both form and phase active.

The prospective finite polynomial checks require, including their tails,
\[
 E_{\rm par}\le7\cdot10^{-10},\qquad
 E_{\rm par,hyb}\le7\cdot10^{-10}.\tag{145}
\]
They are acceptance inequalities to establish by exact rational coefficient
calculations, not claims that unevaluated controls have passed. With the
previous analytical critical radius below \(4.2\cdot10^{-9}\),
(137), (143) and the separately checked center computation leave a strict
margin below the unchanged target. The reference center lower bound from
(114) also leaves a positive actual response above \(1.8\cdot10^{-9}\).
The rational critical half-gap (121) exceeds \(5.77\cdot10^{-9}\);
(141), (144) and (145) therefore retain an obstruction above the same
\(5\cdot10^{-9}\) target for every gamma. Failed fixed inequalities
must remain failed evidence, rather than trigger a fitted interval or
replacement tolerance.

The private `_prepared_sine_history_coupling_information` consumer in the
[coupling-information owner](../../src/tnfr/physics/_sine_history_coupling_information.py)
re-admits primitive support, original affine information, gamma interval
and both durations. Its `critical_reference` is freshly rebuilt from those
primitives. The reference forecast is not supplied as a trusted cached
certificate. One-shot ordered inputs are normalized once before reuse.
Separate report fields distinguish finite parameter polynomials, omitted
series, general-law prefix and history errors, and final forecast budgets.
The direct history consumer keeps its critical-only contract.

The scalar omitted-series bound belongs to the shared
[comparison-flow owner](../../src/tnfr/mathematics/_comparison_flow.py).
The general sine tube is shared behind the preserved critical wrapper in
the [joint-information owner](../../src/tnfr/physics/_sine_joint_information.py).
The numerical domain retains at most 24 nodes and 48 latent columns,
fixed degree 16, \(z_s\le1\) and \(z_g\le1/2\). All raw values,
source/phase association, costs, both stage and branch budgets precede
matrix work, reference construction or exact-zero shortcuts. These resource
limits do not extend the critical identity to other laws. No signed ODE
solver, trajectory replay, reduced autonomous state or speedup is supplied.

The scope and baseline are frozen under
`artifacts/sine-coupling-robustness-20261024/`. Independent controls are
to compose full eight-coordinate rational matrix polynomials, separately
check the channel recurrence, and retain the signed hybrid reflection.
Endpoint and critical values are consistency checks; the coefficient bounds
establish uniformity over the interval. Controls also retain full-field
phase feedback, prefix errors, exact actual-hybrid membership deductions,
unknown-law center uncertainty, clocks, capacities, labels, zeros, readout
and primitive rejection. No passing outcome is asserted before the frozen
controls run. A successful finite result would establish robust usefulness
of generated association for this admitted family, while autonomous NFR
identity, formation, maintenance and physical interpretation remain open.


<a id="sine-interaction-interface"></a>
## 14. A finite packet for a declared family of interactions

A useful original-data description need not retain every coordinate to a
uniform precision, but its meaning depends on the tasks, admitted source
and decoder access. This section fixes those choices before constructing
a packet. It compresses an already available realized initial state.
It neither acquires that state nor substitutes the preceding uncertainty
class forecast for a record of which preparation occurred.

The [observation requirements](../MINIMAL_STRUCTURAL_DEGREES.md#11-specify-the-task-before-minimizing-its-observation)
distinguish task sufficiency, reconstruction and dynamical closure. The
[exact linear observation and memory owner](../../src/tnfr/mathematics/linear_observation.py)
already supplies invariant output-space and hidden-initialization tools
for a supplied tangent. Here a finite nonlinear tolerance permits a
different, approximate interface. Full-support preprocessing, a fixed
history and an anisotropic source remain explicit costs and hypotheses.

### 14.1 Tasks, input and packet cost

Retain the actual law (126), unit capacities, the old unit edges \(12,23\),
preparation duration \(\tau=1/4\), and original phase origin. The family
of prospective queries is
\[
 p\in\{1,3\},\qquad h\in[0,H],\quad H=1/4,\qquad
 \ell_p=(e_0+e_p)/2,\qquad
 \gamma\in[49/100,51/100].\tag{146}
\]
A query adds exactly one unit edge \(0p\) and reads the arithmetic form
mean at its two endpoints, compared with the matched off branch.
The held unknown gamma is common to preparation and both response
branches. Each query starts from the same original preparation and its
full reached state. Querying two alternatives does not execute two
successive contacts. The task family is declared before the packet and
contains both a choice of port and a continuous range of horizons.

Write \(\delta=10^{-4}\), \(\epsilon=10^{-8}\). The unchanged original
source is
\[
 \begin{gathered}
 x_0^0=\varphi_0^0=0,\quad |x_1^0|\le\epsilon,\quad
 \delta\le x_2^0\le3\delta,\quad |x_3^0|\le\delta,\\
 |\varphi_i^0|\le\epsilon\quad(i=1,2,3).
 \end{gathered}\tag{147}
\]
All six uncertain coordinates remain independent and have positive
width. The encoder receives one realized, admitted member of (147),
including both full rows. The original phase chart is a premise; no
later anchor is reset. Access to this realization is stronger information
than merely knowing (147), and its acquisition is not provided here.

For \(y\in[0,10000]\), let \(Q(y)=\lfloor y+1/2\rfloor\), including
deterministic upward ties. Encode
\[
 k_2=Q\!\left({x_2^0-\delta\over2\epsilon}\right),\qquad
 k_3=Q\!\left({x_3^0+\delta\over2\epsilon}\right).
 \tag{148}
\]
The two integers lie in \(0,\ldots,10000\). Decode
\[
 q_2=\delta+2\epsilon k_2,\qquad q_3=-\delta+2\epsilon k_3,
 \qquad |x_i^0-q_i|\le\epsilon\quad(i=2,3). \tag{149}
\]
Endpoint admission precedes rounding; an out-of-source value cannot be
repaired by clipping its code. The exact primitive projection before
quantization has rank two. Each index admits a 14-bit representation, so
the per-instance payload has a 28-bit upper bound. This counts the logical
code, not the memory footprint of language objects or a serialized file.
It is not a minimum-bit theorem.

The decoder receives only this packet, \(p,h\), and fixed metadata:
support, capacities, complete law, gamma range, source ranges, preparation
duration, phase origin and quantizer policy. It has no access to the
original realization, its individual phases, latent coordinates, realized
inventory, source-dependent intervals, future reading or target response.
The fixed model and preprocessing cost are additional to the 28 bits.
Encoding validates all six source coordinates before discarding four.
No new physical measurement is inferred from this validation. A legal packet
alone cannot authenticate its association with a claimed original source:
(157) is conditional on encoding that source or satisfying (149).

The decoder's reference vector \(\widetilde z(q)\) has form entries
\((0,0,q_2,q_3)\) and all four phases zero. It is an auxiliary initial
vector, not an assertion that discarded phases or reached coordinates
vanish in the actual history.

### 14.2 Ordered time-and-coupling polynomials

Use the full channel matrix \(M_\gamma\) and nodal generators from
(127)-(130). Fix total degree \(N=16\), and write \(e=\gamma-1/2\).
For each port, expand the exact finite expression
\[
 P_{p,N}(h,e;z)=
 \sum_{\substack{n,m\ge0\\n+m\le N}}
 {h^n\tau^m\over n!m!}\,
 \ell_p^{\mathsf T}(G_p^n-G_0^n)G_0^m
 \left[(M_{1/2+e}^{n+m})_{00}x+
       (M_{1/2+e}^{n+m})_{01}\varphi\right].
 \tag{150}
\]
Here \(G_p\) includes the added edge, and \(G_0\) is the old generator.
Preparation and response factors remain ordered; their nodal generators
need not commute. Both full tangent rows are represented. With coordinate
order \(z=(x_0,\ldots,x_3,\varphi_0,\ldots,\varphi_3)\), write
\[
 P_{p,N}(h,e;z)=
   \sum_{i=0}^7\sum_{n,r=0}^{N} c_{p,i,n,r}z_i h^n e^r .
 \tag{151}
\]
Each coefficient sums all admissible preparation powers before absolute
values. Distinct response-time powers remain distinct because the decoder
must cover every \(h\) in (146). This grouping retains exact spatial,
channel and preparation associations without using the future response.

The point forecast is the rational polynomial
\[
 \widehat{\mathcal R}_p(h;q)=P_{p,N}(h,0;\widetilde z(q)).
 \tag{152}
\]
For an admitted exact or represented-real time, this is an exact rational
evaluation of the specified finite polynomial. Its omitted series is
budgeted below; it is not a numerically integrated nonlinear solution or
an exact critical heat response. No additional floating center error is
silently merged into the information allowance.

Let
\[
 b=(0,\epsilon,3\delta,\delta,0,\epsilon,\epsilon,\epsilon),
 \quad I=\{1,2,3,5,6,7\},\quad \eta=1/100 .
\]
The following two rational bounds are uniform over the whole query family
for a fixed port:
\[
 \begin{aligned}
 E_{{\rm par},p}
   &=\sum_{n=0}^N\sum_{r=1}^N
          H^n\eta^r\sum_{i=0}^7 b_i|c_{p,i,n,r}|,\\
 E_{{\rm packet},p}
   &=\epsilon\sum_{n=0}^N H^n\sum_{i\in I}|c_{p,i,n,0}|.
 \end{aligned}\tag{153}
\]
Indeed, for every admitted original \(z\),
\[
 \begin{split}
 P_{p,N}(h,e;z)-P_{p,N}(h,0;\widetilde z)
   ={}&[P_{p,N}(h,e;z)-P_{p,N}(h,0;z)]\\
      &+P_{p,N}(h,0;z-\widetilde z).
 \end{split}\tag{154}
\]
The first term is bounded using the actual original source ceilings,
not enlarged reference coordinates. The second uses (147)-(149):
each of its six potentially nonzero entries has magnitude at most
\(\epsilon\). Thus discarded state and quantization share a declared
finite budget, while coupling uncertainty remains separate.
Equation (153) proves uniformity by coefficients; sampled times or
coupling endpoints would not prove it. It is a sufficient triangle bound,
not an exact information radius or an optimized encoding theorem.

### 14.3 Full-history remainder and actual nonlinear accuracy

All branches and stages have receiving degree at most \(k=2\). Set
\(g=51/100\), \(T_*=\tau+H=1/2\),
\[
 z_s={(1+g)kT_*\over2}={151\over200},\qquad
 B=\max(B_0,P_0)=3\delta,\quad B_0=3\delta,\ P_0=\epsilon .
\]
The ordered-series argument of Section 13.2 bounds each actual tangent
branch's omitted terms by \(BR_{17}(z_s)\). Since the forecast (152)
is itself a finite polynomial, only the actual matched tangent tail is
needed:
\[
 E_{\rm tail}=2BR_{17}(z_s). \tag{155}
\]
A second critical tail would compare two full exponentials, which is not
the present prediction contract. No true nonlinear endpoint enters this
bound. At \(h=0\) the matched response and polynomial are exactly zero;
a general uniform allowance may remain conservative at that query.

Apply the complete general-gamma comparison (134)-(135) on the entire
piecewise history. With
\[
 z_g={gkT_*\over2}={51\over200},\qquad
 C=(1-z_g^2/2)^{-1}={80000\over77399},
\]
a common allowance for the two actual branches is
\[
 E_{\rm nl}={2T_*gk\over3}(P_0+z_gB_0)^3 C^4
            <1.8\cdot10^{-13}. \tag{156}
\]
The scalar expression is increasing in the nonnegative time and degree
budgets, so these ceilings cover every shorter \(h\), both ports, and
every held coupling. Both error rows and the full uncertain handoff are
retained; there is no reinitialization after preparation.

Combining (153)-(156) gives the conditional finite-interface theorem:
\[
 \left|\mathcal R_{p,\gamma,h}(z)
              -\widehat{\mathcal R}_p(h;Q(z))\right|
 \le E_{{\rm par},p}+E_{{\rm packet},p}
                      +E_{\rm tail}+E_{\rm nl}. \tag{157}
\]
The response on the left is the original actual nonlinear on/off mean,
not a reconstructed pressure, tangent observation or sparsified handoff.
Fixed source anisotropy makes it possible to discard four coordinates
at this tolerance without declaring them dynamically absent. The
finite-width source still has six uncertain dimensions.

The prospective numerical acceptance requires the right side of (157)
to be at most \(5\cdot10^{-9}\) for both ports. The separate allowances
retain their meanings even when only their total decides acceptance.
This is a fixed inequality to verify, not an unevaluated passing result.
Failure of a chosen sufficient bound would mean that this certificate
does not establish the target; it would not prove impossibility.
Any future readout error would need its own additional allowance:
(157) concerns prediction of the true mathematical response.

### 14.4 Equal descriptions with incompatible contact responses

A smaller-description comparison must use the same source and available
metadata. Consider the two original realized states
\[
 x_2^0=2\delta,\qquad x_3^0=\pm\delta,\qquad
 x_0^0=x_1^0=0,\qquad \varphi^0=0. \tag{158}
\]
Both satisfy (147), and they have the same empty description and the
same exact or quantized \(x_2^0\)-only description. Their realized total
form inventories differ; the protocol does not supply that extra reading.
This is not an equal-inventory obstruction.

Use the admitted critical member \(\gamma=1/2\), port \(p=3\) and
\(h=\tau=1/4\). Interchanging nodes 1 and 3 preserves the old support
and changes this query to the port-1 problem of Section 12. Put
\(s=h/2,r=\tau/2,t=s=r=1/8\). The tangent response coefficient on
the varying \(x_3^0\) becomes
\(C_1=A_1+(s\partial_s+r\partial_r)A_1\), with
\[
 A_1(s,r)={1\over4}\int_0^s p(s-u)\,q_{11}(u+r)\,du .
\]
The positive functions \(p,q\) are the complete heat-response kernels
used in (111)-(114), not new empirical fits.
Their shifted functions \(e^{v/2}p(v)\) and \(e^{v/2}q_{11}(v)\)
have nonnegative power-series coefficients, with leading terms \(v/8\)
and 1, respectively. Rescaling \(u=tv\) gives
\(A_1(t,t)=e^{-t}F_1(t)\), where \(F_1\) has nonnegative
coefficients and leading term \(t^2/64\). Therefore
\[
 C_1=e^{-t}[(1-t)F_1+tF_1']
 \ge(3-t)e^{-t}{t^2\over64}
 \ge {161\over262144}\qquad(t=1/8). \tag{159}
\]
The last inequality uses \(e^{-t}\ge1-t=7/8\). Every later term
has factor \(n+1-t>0\), so retaining the leading term is a lower
bound rather than an asymptotic assertion.

The actual nonlinear half-separation of the two responses is at least
\(\delta\,161/262144-E_{\rm nl,*}>6\cdot10^{-8}\), where the
critical whole-history allowance satisfies
\(E_{\rm nl,*}<1.7\cdot10^{-13}\) by Section 12. Each actual response
is within that matched allowance of its tangent response; dividing the
two-response error by two gives the same allowance. Any single point
forecast for an identical description must incur error at least this
half-separation on one of the two preparations.

Consequently neither the empty nor the \(x_2\)-only description can meet
the \(5\cdot10^{-9}\) target over the family (146). A failure at the
admitted critical member suffices for this unknown-law family; no
pointwise failure at every gamma is asserted. Two retained coordinates
are sufficient if (157) passes, but these witnesses do not exclude every
other one-coordinate observation, arbitrary nonlinear encoding or different
bit allocation. Minimal rank and minimum payload remain unproved.

### 14.5 A compressed preparation is not a closed dynamical state

The packet records primitive initial form coordinates independently of
which query is later selected. It does not store a precomputed answer to
one chosen port and horizon. Its useful scope nevertheless remains the
declared initial class, preparation and query family. Reusing the same
packet for alternative queries differs from updating it after an actual
interaction or using it as an autonomous constituent state.

For example, on the old support at the critical tangent,
\[
 x_2'=-(2x_2-x_1-x_3)/4-(2\varphi_2-\varphi_1-\varphi_3)/8 .
\]
Equal exact values of \(x_2,x_3\) can therefore have different \(x_2'\)
within (147), by varying \(x_1\). The rank-two primitive projection
does not possess an exact autonomous tangent evolution. Quantization
does not restore that missing closure.

The shared `derive_linear_observation` operation closes a supplied output
row space under its *known* tangent generator; any enlarged coordinates
and generator it returns concern that exact linear problem.
`derive_coordinate_memory` retains the hidden-initial contribution and
the feedback convolution after coordinate elimination. Those identities
help diagnose what (157) approximates. They do not make a two-index
packet an exact nonlinear state or license dropping hidden initialization.
The present decoder retains the full fixed support in its preprocessing
and evolves no coordinates during a query; neither fact proves a speedup,
physical dimension reduction, acquired identity or autonomous action.

The private `_encode_sine_history_interface` and
`_decode_sine_history_interface` operations belong to the
[history-interface owner](../../src/tnfr/physics/_sine_history_interface.py).
Its encoder admits the full realized source and emits only the two grid
indices. Its decoder re-admits the packet and query, and consumes fixed
family coefficients and bounds rather than an original-state report or
cached verdict supplied as a premise. Coefficient construction reuses the
shared exact graph/channel arithmetic, omitted-series owner and general
nonlinear tube. Its report separates the discarded-state and quantization
parts of (153), then coupling, series and nonlinear allowances. At zero
response duration its query error is exactly zero, while its uniform
fields still describe the entire family. No separate signed ODE solver is introduced.

Scope and baseline are retained under
`artifacts/sine-interaction-interface-20261024/`. Prospective independent
controls use complete eight-coordinate ordered matrix polynomials,
continuous-time coefficient bounds, quantizer endpoints and upward ties,
complete tangent rows, and the closed critical witness (158)-(159).
Selected shared regressions retain the complete nonlinear field checks.
Exact observation/memory checks distinguish approximate prediction from
dynamical closure. Coupling endpoints and selected horizons are consistency
checks, while the rational uniform inequalities establish the continuum
claim. Encoder/decoder rejection and the absence of original-state access
remain part of the contract. No passing outcome is asserted before the
reviewed controls run, and no nonlinear trajectory, physical reading,
empirical validation or autonomous contact is supplied.


<a id="sine-interface-continuation"></a>
## 15. Bounded history through an actual prescribed interaction

The preceding packet answers alternative queries from one preparation.
It does not say how much information must survive an interaction that
actually changes the state. This section fixes one such continuation,
including a finite history budget. The
[guiding emergence hypothesis](../EMERGENT_ONTOLOGY.md#matter-emergence-hypothesis)
motivates asking whether a structural description can carry reproducible
response properties through interactions. A finite mathematical memory
result is one prerequisite; it does not identify matter or a physical
property.

### 15.1 A common handoff with a recorded past contact

Keep the complete law (126), the original source (147), unit capacities,
and one unknown held \(\gamma\in[49/100,51/100]\). Let \(G_0\) be
the old-support generator on edges \(12,23\), and \(G_p\) add unit
contact \(0p\), for \(p=1,3\). Fix the chronological protocol
\[
 \begin{array}{c|c|c}
 \text{stage}&\text{support}&\text{duration}\\ \hline
 \text{original preparation}&G_0&\tau=1/4\\
 \text{first interaction}&G_p&\sigma=1/8\\
 \text{second on/off comparison}&G_q\ \text{or}\ G_0&
                  h\in[0,H],\ H=1/32
 \end{array}\tag{160}
\]
Remove the first added edge immediately before the last stage. Both
support changes are prescribed and leave the full state continuous.
The original phase origin persists; node zero's form and phase need not
remain zero after the first interaction. No reached coordinate is reset,
projected onto a new source box or inferred from the packet.

The two final branches start from the *same actual* handoff reached with
past port \(p\). They use the same held coupling and differ only in
whether the query contact \(0q\) is present, where \(q=1,3\). The output
is now the excess form at the receiving node:
\[
 \mathcal R_{p,q,\gamma,h}(z)
       =x_{0,\rm on}(\tau+\sigma+h)-x_{0,\rm off}(\tau+\sigma+h).
 \tag{161}
\]
This is an explicit changed readout from Section 14's contact-pair mean.
The future window is also explicitly shorter. Neither change is inferred
from evaluated responses. The initial information, coupling interval and
useful target \(5\cdot10^{-9}\) are retained.

The support immediately before the query is \(G_0\), regardless of the
past port. Its current graph therefore cannot reveal which first contact
occurred. No current-state readings are supplied. Removing an edge does
not erase its dynamical effect on form and phase. The protocol makes no
claim about autonomous event occurrence, passive switching or cancellation
of support-event storage work.

### 15.2 One bounded history slot

Reuse the encoder (148) on an already available realized original source.
The updated record is exactly
\[
 \mathsf r=(k_2,k_3,b),\qquad k_i\in\{0,\ldots,10000\},
 \quad b\in\{0,1\},\qquad p=1+2b .\tag{162}
\]
The update consumes the original packet and the known prescribed first
contact, and appends its bit once. Already-advanced records are rejected;
there is no unbounded action log hidden in this operation. Durations and
the removal rule are fixed metadata rather than additional per-instance
records.

The two 14-bit slots and the one history bit give a 29-bit payload upper
bound. This is a declared representation cost, not an optimal encoding.
The decoder receives only this record, \(q,h\), and fixed law, support,
source, clock, phase-origin and precision metadata. It cannot reacquire
the original or current realization, inspect hidden phases, or consume
source-dependent intervals and prior response reports. Association of
the original packet with its source, and of the bit with the interaction
that actually occurred, remain premises. Valid integer syntax alone
does not authenticate them.

The record update does not evolve two physical coordinates. It retains
a bounded piece of original information and a bounded prescribed history
so that the decoder can predict one further interaction. That is a
continuation contract, distinct from an autonomous current-state law or
a reusable update for arbitrarily many interactions.

### 15.3 Ordered three-stage prediction and uniform error

For the tangent, let \(J_\gamma(G)\) be (127). The exact ordered
matched map is
\[
 e_x^{\mathsf T}
 [e^{hJ_\gamma(G_q)}-e^{hJ_\gamma(G_0)}]
 e^{\sigma J_\gamma(G_p)}e^{\tau J_\gamma(G_0)}z ,
 \tag{163}
\]
where the full-state row \(e_x^{\mathsf T}\) selects form coordinate
zero. Both rows and the actual first-stage state enter this expression.
Nodal generators need not commute. The coupling is the same in all
factors, so each chronological term has one channel power.

More explicitly, with nodal row \(e_0^{\mathsf T}\), the degree-16
polynomial is
\[
 \begin{split}
 P_{p,q,N}(h,e;z)
 =\sum_{\substack{n,m,l\ge0\\n+m+l\le N}}
 {h^n\sigma^m\tau^l\over n!m!l!}\,
 e_0^{\mathsf T}(G_q^n-G_0^n)G_p^mG_0^l
 \bigl[&(M_{1/2+e}^{n+m+l})_{00}x\\
       +&(M_{1/2+e}^{n+m+l})_{01}\varphi\bigr],
 \quad N=16 .
 \end{split}\tag{164}
\]
The \(n=0\) terms vanish. The \(n=1\) terms generally do not:
the receiver readout does not have the contact-pair cancellation used in
Section 14. Aggregate both fixed-duration powers \(m,l\) before taking
absolute values, retaining future-time and coupling powers:
\[
 P_{p,q,N}(h,e;z)=\sum_{i,n,r}c_{p,q,i,n,r}z_i h^n e^r .
 \tag{165}
\]
The point forecast is
\(\widehat{\mathcal R}_{p,q}(h;\mathsf r)
       =P_{p,q,N}(h,0;\widetilde z(k_2,k_3))\),
with exactly the reference reconstruction of Section 14. It includes the
first interaction through its ordered generator, rather than restarting
that reconstruction at the handoff.

Use the coordinate ceilings \(b_i\), index set \(I\), and
\(\eta=1/100\) from (153). For each of the four ordered port pairs,
define
\[
 \begin{aligned}
 E_{{\rm par},p,q}
 &=\sum_{n=0}^N\sum_{r=1}^N H^n\eta^r
                  \sum_i b_i|c_{p,q,i,n,r}|,\\
 E_{{\rm packet},p,q}
 &=\epsilon\sum_{n=0}^N H^n\sum_{i\in I}|c_{p,q,i,n,0}|.
 \end{aligned}\tag{166}
\]
The exact decomposition (154) applies to (165): coupling variation is
bounded on the actual original state, then the critical polynomial acts
on the discarded and rounded original-coordinate error. The history bit
is exact within this protocol, not another numerical uncertainty.
These coefficient sums cover every admitted state, matching packet,
\(h\in[0,H]\) and held gamma. Evaluating endpoints alone cannot prove
that assertion.

All stages have degree at most two. For the entire duration,
\[
 T_*={13\over32},\quad k=2,\quad g={51\over100},\quad
 z_s={(1+g)kT_*\over2}={1963\over3200},\quad
 z_g={gkT_*\over2}={663\over3200}. \tag{167}
\]
The multinomial identity bounds the sum of all ordered terms of total
degree \(j\) by \([L(h+\sigma+\tau)]^j/j!\), with
\(L=(1+g)k/2\), without commuting the generators. The same geometric
tail as (155) therefore gives
\[
 E_{\rm tail}=2(3\delta)R_{17}(z_s). \tag{168}
\]
There is only one actual matched tangent tail: the forecast is a finite
polynomial, not a second full exponential approximation.

The complete nonlinear comparison (134)-(135) applies through both
state-continuous switches. In particular,
\[
 C=(1-z_g^2/2)^{-1}={20480000\over20040431},\qquad
 E_{\rm nl}={2T_*gk\over3}
                  (\epsilon+z_g\,3\delta)^3 C^4 . \tag{169}
\]
Products of heat propagators contract the form supremum norm, and the
same sine and coupled Volterra inequalities bound both error rows.
No handoff error is reset or silently dropped. For example,
\(\epsilon+z_g\,3\delta<0.63\delta\) and \(C^4<11/10\)
give \(E_{\rm nl}<8\cdot10^{-14}\).
The budgets \(z_s<1,z_g<1/2\) stay inside the existing numerical domains.

For every admitted actual history the resulting sufficient inequality is
\[
 |\mathcal R_{p,q,\gamma,h}(z)
             -\widehat{\mathcal R}_{p,q}(h;\mathsf r)|
 \le E_{{\rm par},p,q}+E_{{\rm packet},p,q}
                       +E_{\rm tail}+E_{\rm nl}. \tag{170}
\]
The fixed acceptance is that this total is at most \(5\cdot10^{-9}\)
for every port pair. At \(h=0\) the exact response, polynomial and query
error are zero after admission; uniform fields still cover the whole
future interval. There is no supplied final readout error. Any added
measurement error would require an independent allowance.

### 15.4 Forgetting the first contact

Fix the same original realization in the two histories:
\[
 x^0=(0,0,2\delta,\delta),\qquad \varphi^0=0,\qquad
 \gamma=1/2,\quad q=3,\quad h=1/32 . \tag{171}
\]
Use first port 1 in one history and port 3 in the other. Both original
packets equal \((5000,10000)\). Original state, source, law, inventories,
durations, current old support and second query agree. Their actual
current coordinates may differ and are not part of the description.
For this reciprocal unit-capacity law, the total form and lifted-phase
inventories remain conserved along each stage, so those whole inventories
also agree. No equality of the full reached states is claimed.

The signed critical polynomial half-gap is the exact rational quantity
\[
 D_N={1\over2}\left[
 P_{1,3,N}(1/32,0;z)-P_{3,3,N}(1/32,0;z)\right].\tag{172}
\]
Its first nonzero total-degree term illustrates why a history record can
matter. Writing \(A_j=G_j-G_0\), direct row multiplication gives
\[
 e_0^{\mathsf T}A_3(A_1-A_3)
       ={(-e_0-e_1+2e_3)^{\mathsf T}\over16},
 \qquad (M_{1/2}^2)_{00}=3/4 .
\]
The leading half-gap on (171) is \(3h\sigma\delta/64\).
This local coefficient motivates the witness, but does not establish a
finite inequality by itself. The full polynomial (172) retains the
nonzero preparation, every first-contact term and their signs.

Each matched actual response differs from its own critical polynomial by
at most \(E_{\rm tail}+E_{\rm nl}\), using the uniform bounds above.
Consequently the signed actual half-difference is at least
\[
 D_N-E_{\rm tail}-E_{\rm nl}. \tag{173}
\]
The same allowance suffices after taking half the difference of two
histories; it is not doubled again. The prospective negative-control
acceptance is that (173) exceed \(5\cdot10^{-9}\).
This is to be proved by the frozen exact polynomial and remainder
calculation, not asserted from the leading term or from a simulated
nonlinear endpoint.

If this inequality holds, no point decoder forgetting the past port can
meet the target on both histories, even though it retains the complete
original packet and sees the same present support and query. One admissible
critical member suffices to obstruct the entire unknown-law family.
This does not assert failure separately at every gamma or prove a
universal minimum number of history bits. It identifies information lost
by this particular history-erasing description.

### 15.5 Scope and implementation evidence

The [continuation owner](../../src/tnfr/physics/_sine_history_continuation.py)
reuses the original encoder and exact admission, graph/channel polynomial,
tail and nonlinear-tube owners. It admits the bounded record and query
before cached coefficient work or zero shortcuts. Fixed family caches
contain no realization or externally trusted response certificate.
The full fixed support and chronological model remain preprocessing costs.

The [exact observation and memory owner](../../src/tnfr/mathematics/linear_observation.py)
explains why eliminating internal coordinates can retain initialization
and history. Equations (170) and (173) add a finite nonlinear
continuation contract under this particular source and action budget.
They do not establish exact closure, an automatically selected
interaction, a general memory bound or a physical property. Prescribed
support switches have no passive-energy interpretation here.

Scope and baseline are retained under
artifacts/sine-interface-continuation-20261024/. Prospective independent
controls construct complete eight-coordinate chronological polynomials,
check full tangent rows, all coefficient powers, continuous time/coupling
budgets, and the signed finite witness. They also test original packet
association, fixed record cost, one-update admission, lack of state
readback and invalid-input rejection. Selected unchanged regressions
retain the complete nonlinear-law checks. Neither passing outcomes nor
nonlinear trajectories are asserted before the frozen controls run.
Failure of a sufficient bound remains unavailable certification; it does
not authorize retuning the source, horizon, tolerance or evaluated witness.


<a id="sine-present-state"></a>
## 16. Present boundary outputs and their exact relative-state completion

A bounded history forecast does not itself supply a present state with
inherited evolution. Here the candidate is fixed independently: retain
form and phase differences at the two boundary ports and ask what exact
current information their evolution requires. The result distinguishes
four boundary readings, six relative state coordinates, and uncertainty
about the law. Its finite error allowance is conditional on present-state
initialization, not a conversion of the preceding history record.

### 16.1 Present observation, symmetry and actual source

Keep the complete unit-capacity law (126), old edges \(12,23\), and
supports \(G_0,G_1,G_3\), where \(G_p\) adds unit edge \(0p\).
A single held \(\gamma\in[49/100,51/100]\) is common throughout a
finite prescribed sequence of state-continuous support switches.
Define
\[
 a_i=x_i-x_0,\qquad b_i=\varphi_i-\varphi_0,\quad i=1,2,3,
 \qquad r=(a_1,a_2,a_3,b_1,b_2,b_3),
 \qquad y=(a_1,a_3,b_1,b_3). \tag{174}
\]
The phase differences use a declared continuous local lift. Globally their
state space is the relative phase torus, not three unrestricted primitive
absolute phases. No phase is inferred from form.

This observation requires present boundary data, a stronger access premise
than knowing the original two indices and a past-contact bit. The exact
completion below consumes six actual present relative coordinates, or a
separately certified estimate. No acquisition process or automatic recovery
of those coordinates from the old record is asserted.

For each fixed gamma and past port, the original strict source box (147)
is open in six relative coordinates. The complete smooth sine field is
globally Lipschitz on continuous lifts. Its finite flow, and a finite
composition across prescribed switches, is invertible and maps this
interior to an open reached set. Thus hidden variations in a sufficiently
small relative-coordinate ball can correspond to actual admitted
preparations, rather than merely points in a loose outer enclosure.

The reached phase chart after old preparation and one first interaction
is acute. Indeed its total prefix duration is \(3/8\), so the comparison
(134) has \(z=153/800<1/5\), \(C<50/49\), \(B_0=3\delta\) and
\(P_0=\epsilon=10^{-4}\delta\). Consequently
\[
 P_*< {50\over49}\left({3\over5}+10^{-4}\right)\delta<\delta,
 \qquad |b_2-b_1|<2\delta<\pi/2. \tag{175}
\]
This controls actual reached phases with both rows active. It does not
assert that their exact reached image equals the norm-containing box.

### 16.2 Four linear outputs require six relative coordinates

On the old support the complete tangent gives
\[
 \begin{aligned}
 \dot a_1&=(a_2-a_1+\gamma(b_2-b_1))/4,&
 \dot a_3&=(a_2-a_3+\gamma(b_2-b_3))/4,\\
 \dot b_1&=\gamma(a_1-a_2)/4,&
 \dot b_3&=\gamma(a_3-a_2)/4.
 \end{aligned}\tag{176}
\]
For every nonzero gamma,
\[
 a_2=a_1-4\dot b_1/\gamma,\qquad
 b_2=b_1+(4\dot a_1-a_2+a_1)/\gamma . \tag{177}
\]
The four output rows and their first generator images therefore span all
six relative-coordinate rows. This is an exact algebraic observability
identity; it does not supply measured derivatives.

Let \(O\) be the four-row full-state output matrix. Applying the shared
[linear observation owner](../../src/tnfr/mathematics/linear_observation.py)
to the old-support tangent at \(\gamma=1/2\) gives the minimal invariant
row closure with dimension six. Formula (177) proves the rank increase
from four to six without numerical fitting. The two common-shift
directions remain annihilated, so the dimension cannot exceed six.

Every selected generator maps relative-coordinate rows into the same
six-dimensional row space. Therefore it is also a minimal *common*
linear state space containing \(O\): any such space must contain the
old-support critical closure, and the six relative rows provide the upper
bound for all supports. Exact intertwining checks at both gamma endpoints
suffice for the whole interval, because the complete tangent depends
affinely on gamma. These checks establish a linear identity, not
nonlinear equivalence inferred from an endpoint sample.

The selected owner can return another basis for this same row space.
Its coordinate names need not be primitive nodes or physical quantities.
The exact hidden-initialization and convolution identities supplied by
the coordinate-memory owner remain the appropriate description if the
two central relative coordinates are eliminated instead.

### 16.3 Exact nonlinear quotient with a moving reference

Let \(D\) be the \(3\)-by-\(4\) difference matrix with rows
\(e_i^{\mathsf T}-e_0^{\mathsf T}\), and let \(E\) insert a zero
reference coordinate. Thus \(DE=I_3\). For each support set
\(K_S=DG_SE\), and let \(S_S\) denote its complete sine edge sum.
Then the exact relative law is
\[
 \dot a=K_Sa+{\gamma\over4}D S_S(Eb),\qquad
 \dot b=-\gamma K_Sa . \tag{178}
\]
Uniform form translations and common phase rotations leave the full rates
unchanged, so evaluating a zero-reference representative gives precisely
these projected rates. Operationally one evaluates all eight original
rates and subtracts the reference-node rate from every other node.

This subtraction matters under contact: the actual reference node is
moving. Choosing its representative coordinate to be zero is a coordinate
choice, not freezing its dynamics or resetting the physical network.
At a prescribed support switch, \(r\) remains continuous and the generator
and sine sum change to the selected support.

Equation (178) is an exact six-coordinate nonlinear state family, on
\(\mathbb R^3\times\mathbb T^3\), with continuous lifts when needed.
It removes two common shifts from the eight-coordinate representation.
It retains all six relative dimensions of the original source; this is a
symmetry reduction baseline, not a newly discovered smaller collective
state. The unknown gamma remains a held law parameter. A parameter-indexed
family of exact vector fields is not a point forecast independent of gamma.

Receiver excess has an exact relative output map. Reciprocal unit
conductances and unit capacities conserve the full form mean during every
stage, including prescribed state-continuous switches. Since
\(x_0=\bar x-(a_1+a_2+a_3)/4\), two response branches initialized
from the same actual state obey
\[
 x_{0,\rm on}-x_{0,\rm off}
       =-{1\over4}\sum_{i=1}^3(a_{i,\rm on}-a_{i,\rm off}). \tag{179}
\]
Their common form mean cancels. Dropping it is legitimate for this matched
task, not for an absolute form reading or an unrelated branch preparation.

### 16.4 Nonlinear nonclosure and a scoped smooth dimension bound

The four-output obstruction can be established on the actual nonlinear
law, without transferring a tangent rank. On the old support,
\[
 \dot b_1={\gamma\over4}(a_1-a_2),\qquad
 \dot a_1={a_2-a_1+\gamma\sin(b_2-b_1)\over4}. \tag{180}
\]
Choose a point inside an actual reached open chart from Section 16.1.
Small positive and negative variations of hidden \(a_2\), with all
other relative coordinates fixed, keep \(y\) identical but change
\(\dot b_1\). Both points remain actually reachable from the original
source. There is therefore no single-valued exact autonomous law for
those four outputs on that chart. The originals or their conserved means
need not be identical; those additional records are not included in \(y\).

A stronger local statement requires an explicit realization class.
Consider a smooth present-state map \(s=\Psi(r)\in\mathbb R^d\),
a smooth autonomous state law and a smooth output map that exactly
reproduce all four output trajectories on an open reached chart, for
the fixed old support and a fixed nonzero gamma. Sufficient differentiability
is assumed for the output and its first Lie derivatives. The map
\((y,\dot b_1,\dot a_1)\) must then factor through \(s\).
In the hidden columns \(a_2,b_2\), after the four identity output rows,
its derivative block is
\[
 \begin{pmatrix}
 -\gamma/4&0\\1/4&\gamma\cos(b_2-b_1)/4
 \end{pmatrix},\qquad
 \det=-{\gamma^2\over16}\cos(b_2-b_1). \tag{181}
\]
By (175) this determinant is nonzero on the reached chart. The full
output/derivative map has rank six. The chain rule consequently requires
\(d\ge6\). The actual nonlinear quotient (178) attains this bound.

This is a local smooth *exact* realization result for the declared four
boundary trajectories. It does not prove a minimum for arbitrary
discontinuous encodings, a number of bits, or finite-tolerance predictions
of a smaller future task. In particular, it does not show that four
coordinates fail the preceding \(5\cdot10^{-9}\) target. A finite
accuracy obstruction would require its own equally observed actual states,
response, time and error lower bound. Unknown coupling does not silently
become an observed coordinate or a uniquely identified constitutive law.

### 16.5 Conditional initialization-error lifetime

A finite inherited error can be bounded without introducing a new solver.
For every selected support and \(\gamma\le g=51/100\), the full
eight-coordinate field has supremum-norm Lipschitz constant at most
\(1+g\). Its form row has bound
\(d_i(\|\Delta x\|_\infty+g\|\Delta\varphi\|_\infty)/2\)
and its phase row has bound \(g d_i\|\Delta x\|_\infty/2\),
with \(d_i\le2\). Both rows are included.

The full relative projection has norm two and the zero-reference lift has
norm one. Thus (178) has a common sufficient constant
\[
 K=2(1+g)={151\over50}. \tag{182}
\]
Let \(r,\widehat r\) be exact solutions of (178) with the *same*
held gamma and the same prescribed support sequence, and assume a supplied
initial error \(\|r(0)-\widehat r(0)\|_\infty\le\rho\).
The integral inequality and Gronwall comparison give
\[
 \|r(t)-\widehat r(t)\|_\infty\le e^{Kt}\rho . \tag{183}
\]
State continuity carries this estimate through any finite number of the
admitted switches; products of the growth factors give the total elapsed
time, rather than a new initialization at each switch.

For matched on/off receiver predictions using the relative output (179),
each branch comparison has (183). Therefore
\[
 |\mathcal R-\widehat{\mathcal R}|
       \le {3\over2} e^{Kt}\rho . \tag{184}
\]
The factor \(3/2\) is the sum of the two output-row norms \(3/4\).
The on and off schedules may differ, but within each comparison the
actual and reference schedule, gamma and initialized state association
must match.

Fix the conditional initialization allowance and future lifetime
\[
 \rho=10^{-9},\qquad 0\le t\le1/4 .
\]
Then \(Kt\le151/200<1\), and the elementary exponential series gives
\(e<3\). Consequently
\[
 {3\over2}e^{Kt}\rho<4.5\cdot10^{-9}<5\cdot10^{-9}. \tag{185}
\]
This bounds initialization uncertainty in the admitted structural units.
The exact reference evolution still needs its own numerical error
certificate if approximated computationally. No such solver error,
measurement error or parameter-substitution error is included in (185).

In particular, both solutions in (183) use the same unknown law member.
The bound is uniform over those paired members; it does not make one
nominal-gamma solution a forecast for the whole interval. Selecting that
nominal value would require a separate parameter allowance with its
association preserved through every stage. Likewise a certified present
radius is not silently supplied by the old 29-bit record or by naming a
measurement. Initialization access and its cost remain explicit.

### 16.6 Scope and prospective controls

The [present-state owner](../../src/tnfr/physics/_sine_present_state.py)
reuses the exact observation/memory owner, original complete sine field
and scalar affine growth comparison with zero forcing defect.
`_prepared_sine_present_observation` derives the anchor closure and verifies
the common-family identities; `_held_sine_present_field` requires an
explicit admitted gamma and projects all full rates, preserving interval
and common-order jet semantics. `_bound_sine_present_initialization`
consumes only a supplied current radius and duration. It includes no
solver defect or parameter-mismatch allowance. No fresh rank-closure
algorithm, historical decoder or ODE solver is introduced.

Scope and baseline are retained under
`artifacts/sine-present-state-20261024/`. Prospective controls independently
assemble all eight tangent rows, recover the central relative coordinates,
check common intertwining for the three supports and two endpoint
couplings, and evaluate the complete sine sums before relative projection.
They also retain the nonlinear derivative minor, moving-reference
counterexamples, mean-cancellation output, interval/jet admission and the
rational finite growth allowance. Exact calculations certify the stated
identities; no passing outcome is asserted before the frozen controls run.

The [matter-emergence hypothesis](../EMERGENT_ONTOLOGY.md#matter-emergence-hypothesis)
remains a motivation for deriving present identity, inherited dynamics
and reproducible response properties. This gate establishes an exact
relative-state baseline and identifies a limit of four boundary readings.
It derives neither a smaller organized object, its physical interpretation,
autonomous formation nor the acquisition of its present state.
