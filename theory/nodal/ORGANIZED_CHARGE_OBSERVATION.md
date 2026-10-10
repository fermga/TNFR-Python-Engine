# Conserved charge observations under the organized sine law

<a id="organized-charge-observation"></a>

## Model, observation class and scientific scope

This owner classifies restricted conserved observations of the
complete organized sine law. It distinguishes a local contribution from a
general collective property: **additivity across regions does not imply that
each contribution reads only one node**. The classification below imposes
that additional information restriction explicitly. It is not a theorem
excluding physical charge or all collective charge maps.

The physical comparison below closes with an exact obstruction to a
specified density-only electroquasistatic realization. A separate modal
proof also excludes one homogeneous local edge redistribution as a smooth
complete density state, including near an acute C9 twist. These conclusions
leave descriptions with retained internal material state open. They do not
admit a physical dataset or infer an electromagnetic origin for TNFR.

Keep a fixed finite connected simple unit graph \(G=(V,E)\), with
\(n=|V|\ge2\), its unnormalized Laplacian \(L\), held unit capacities,
and the complete quarter law

\[
 x'=\frac{-Lx+\gamma S(\theta)}4,\qquad
 \theta'=\frac{\gamma Lx}4,\qquad
 S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
 \gamma=\frac1{1023\pi}>0.
 \tag{1}
\]

Prime denotes the declared tau clock of the
[organized contact](ORGANIZED_EXTERNAL_INTERACTION.md#1-complete-state-support-and-balances).
All \(x_i\in\mathbb R\) are signed form, and all phases lie on
\(\mathbb T=\mathbb R/(2\pi\mathbb Z)\). Both rows use the same
quarter factor, including on graphs with unequal degrees. There is no
forcing, coordinate reset, support event, variable capacity or additional
phase velocity. This is a detached theorem for the smooth-sine comparison;
it neither selects the law nor executes a registered operator. The actual
144-edge contact graphs are members of this support class.

The complete field preserves \(\sum_i x_i\), by pairwise cancellation.
It also preserves \(\sum_i\widetilde\theta_i\) on any consistently
transported set of continuous phase lifts. The latter is not a
single-valued real observation of the circular state. Common form
translation is an exact symmetry of (1), but calling that symmetry
unobservable is a separate observation premise. These distinctions retain
the [form foundation](../FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation),
[parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md),
[common-origin owner](SINE_CONSTITUTIVE_INFORMATION.md#form-balance-common-origin)
and [collective-property ontology](../EMERGENT_ONTOLOGY.md#research-target-physical-properties-of-collective-patterns).

## Identical local contributions: an exhaustive classification

<a id="identical-local-charge-classification"></a>

Let \(f:\mathbb R\times\mathbb T\to\mathbb R\) be \(C^2\), the
same function at every node, and define

\[
 Q_f(x,\theta)=\sum_{i\in V}f(x_i,\theta_i).
 \tag{2}
\]

Each contribution consumes its own form and circular phase only. It does
not consume neighbor states, incident edges, a region label, a hidden
reference, history, a phase lift or the state of a measuring apparatus.
Require (2) to be conserved for **every** solution and initial state in the
full domain of (1), not only along a selected acquired family or inside one
winding chamber.

**Conditional classification.** Under these hypotheses,

\[
 \boxed{f(x,\theta)=a x+b,\qquad
 Q_f=a\sum_i x_i+bn,}
 \tag{3}
\]

for constants \(a,b\in\mathbb R\). Conversely every expression (3) is
conserved on the fixed support. The proof only needs \(C^1\) regularity;
the stated \(C^2\) class is a sufficient convenient domain.

### Proof

Conservation for every initial state implies the pointwise identity

\[
 0=4Q_f'
 =\sum_i f_x(x_i,\theta_i)
       [- (Lx)_i+\gamma S_i(\theta)]
   +\gamma\sum_i f_\theta(x_i,\theta_i)(Lx)_i.
 \tag{4}
\]

First take \(x_i=s\) for every node. Give one vertex \(k\) phase
\(\beta\), and every other vertex phase \(\alpha\). Its degree
\(d_k>0\) because the graph is connected and has at least two nodes.
The form Laplacian and entire phase row vanish. Only edges incident on
\(k\) contribute to (4), giving

\[
 0=\gamma d_k\sin(\alpha-\beta)
   [f_x(s,\beta)-f_x(s,\alpha)].
 \tag{5}
\]

Thus \(f_x(s,\alpha)=f_x(s,\beta)\) whenever the sine is nonzero.
Continuity supplies the remaining pairs of angles. Since \(s\) was
arbitrary, integration in form gives

\[
 f(x,\theta)=A(x)+B(\theta),
 \tag{6}
\]

where \(B\) is a single-valued periodic function.

Next take all phases equal, so \(S=0\). The last term of (4) is
\(\gamma B'(\alpha)\sum_i(Lx)_i=0\). Give vertex \(k\) form \(r\)
and every other vertex form \(s\). The remaining identity is

\[
 0=-d_k(r-s)[A'(r)-A'(s)].
 \tag{7}
\]

For any unequal \(r,s\), their derivatives agree. Hence \(A'(x)=a\)
is constant. Finally (4) reduces, for every form vector and phase state, to

\[
 0=\gamma B'(\theta)^T Lx.
 \tag{8}
\]

Because \(L\) is symmetric and its kernel on a connected graph is
\(\operatorname{span}\{\mathbf1\}\), (8) implies
\(B'(\theta_i)=B'(\theta_j)\) for all nodes. Phase assignments are
arbitrary, so \(B'\) is constant on the circle. Periodicity gives
\(\int_0^{2\pi}B'(u)\,du=0\), making that constant zero. This proves
(3). The converse follows from the conserved form sum and fixed node count.

### Common-origin invariance is an additional restriction

If, in addition, the observation must satisfy

\[
 Q_f(x+c\mathbf1,\theta)=Q_f(x,\theta)
 \quad\text{for every real }c,
 \tag{9}
\]

then (3) changes by \(anc\), forcing \(a=0\). Only the constant \(bn\)
remains. Declaring every reference node to carry zero contribution fixes
\(b=0\). Therefore there is no nonzero, state-dependent, origin-invariant
conserved observation in the identical local class (2) on this full domain.

This result does not require the actual form sum to vanish. Conversely,
the zero form totals in the acquired contact preparations are facts about
those preparations, not a derivation of universal neutrality. A
symmetry-invariant interpretation of charge cannot be imposed merely
because the underlying differential equations have that symmetry.

## A retained form reference permits a nonzero inventory

<a id="retained-reference-charge-inventory"></a>

Extend the observation data by one independently declared scalar reference
\(r\), held with \(r'=0\). Its constancy is an explicit premise; it is not
a new law inferred from an evaluated response. For each fixed \(r\), the
same proof gives

\[
 f(x,\theta;r)=a(r)x+b(r).
 \tag{10}
\]

Require covariance of the observation under a common change of the form
chart and its retained reference,

\[
 f(x+c,\theta;r+c)=f(x,\theta;r).
 \tag{11}
\]

Comparison of the coefficients of \(x\) in (11) gives
\(a(r+c)=a(r)=\kappa\). Its constant term then gives
\(b(r)=b_0-\kappa r\). Normalization \(f(r,\theta;r)=0\) fixes
\(b_0=0\), leaving exactly

\[
 q_i=\kappa(x_i-r),\qquad
 Q_A=\kappa\sum_{i\in A}(x_i-r).
 \tag{12}
\]

The scale \(\kappa\) is not supplied by TNFR. If \(q_i\) is proposed in
coulombs and form in a declared unit X, then \([\kappa]=\mathrm C/\mathrm X\).
A physical clock would separately need a conversion from tau to seconds.
Neither unit assignment nor (11) establishes such a measurement bridge.

For a fixed region \(A\subseteq V\), edge cancellation in (1) gives

\[
 Q_A'=
 \frac\kappa4\sum_{\substack{i\in A,\ j\notin A\\i\sim j}}
 [x_j-x_i+\gamma\sin(\theta_j-\theta_i)].
 \tag{13}
\]

The complement receives the opposite boundary current, and \(Q_V'=0\).
Thus (12) is additive, signed and transported. This is the uniform-quarter
counterpart of the existing
[degree-normalized regional inventory](SINE_PAIR_INTERACTION.md#sine-autonomous-regional-transfer);
its weights and clock cannot be copied from that different law. Storage
loss does not destroy (12), and the boundary phases remain consumed state.

If \(r\) instead evolves, the regional row has the additional term
\(-\kappa |A|r'\). If node membership, scale, support or form changes in
an event, its actual inventory jump needs separate accounting. Conserved
continuous totals do not establish event conservation, admission, descent
or occurrence. The constant \(bn\) in (3) similarly changes when the node
count changes; it is not a conserved physical particle number without an
applicable support-event law.

### A self-centered reference has a different meaning

Choosing \(r=\bar x_V=n^{-1}\sum_i x_i\) makes (12) invariant under a
common form shift and keeps \(r'=0\) under (1). This is allowed, but it is
a collective reference rather than a contribution with only its own node
as input. It enforces \(Q_V=0\) by definition. Regional excesses can be
nonzero and still satisfy (13).

The reference set must remain part of the observation. If it is enlarged
from \(V\) to a disjoint set \(V\cup W\), with \(m=|W|>0\), the same
unchanged region \(A\subseteq V\) acquires

\[
 Q_A^{V\cup W}-Q_A^V
 =\kappa\frac{|A|m}{n+m}
       (\bar x_V-\bar x_W).
 \tag{14}
\]

Equation (14) is a change of the declared observation reference, not a
current or an instantaneous physical influence from \(W\). It proves that
arbitrary re-centering against a new environment cannot be treated as the
same localized charge measurement. A fixed external reference or a fixed
whole support avoids that reassignment, while retaining its additional
reference premise. No charge quantization follows from either choice.

## Boundaries and collective possibilities beyond the site classification

<a id="charge-classification-boundaries"></a>

The full-state and information hypotheses are essential:

- If \(\gamma=0\), phase is frozen and every periodic
  \(\sum_i B(\theta_i)\) is conserved. The phase-elimination step (5)
  fails, as it must. Equation (1) has strictly positive gamma.
- On a declared continuously lifted phase chart, a linear lifted-phase sum
  is conserved. Changing one representative by \(2\pi\) changes that
  sum, so it is not a globally single-valued real charge on the circular
  state. This distinction does not forbid periodic collective functions
  of the total lifted phase, which are outside (2).
- For a single isolated node, both rows vanish and every function of its
  state is conserved. The connected support with at least two nodes is a
  premise, not an incidental convenience.
- Conservation only within a selected winding chamber or acquired source
  family is weaker than conservation on the full model domain. The proof
  uses uniform-phase and arbitrary-form configurations; it cannot classify
  a map defined only near a formed organization. Winding may be locally
  conserved until a branch crossing, without being a smooth global
  single-node sum.
- Different node functions, retained references, boundary state, spatial
  geometry, nonlocal correlations, history and detector interaction are
  outside the identical local information restriction. Neither this
  theorem nor the earlier
  [multiplicative-factorization obstruction](ORGANIZED_INTERFACE_FACTORIZATION.md#finite-interface-factorization-result)
  excludes those collective descriptions.

For example, an antisymmetric edge allocation \(p_{ij}=-p_{ji}\), supplied
as a regular circular function of the actual boundary state, permits

\[
 \widetilde q_i=\kappa(x_i-r)+\sum_{j\sim i}p_{ij}.
 \tag{15}
\]

Its full sum equals that of (12), while its regional sum gains a boundary
term. Regional additivity still holds. Equation (15) is an algebraic
possibility, not a selected charge law: its evolution consumes
\(p_{ij}'\), its spatial/source interpretation needs independent
justification, and support events need their own admission. It explicitly
shows why the absence of a phase term in (3) is not an absence of all
phase-dependent collective densities. Calling the added term polarization
would require the corresponding physical observation and field relations.
The [local edge obstruction](#local-edge-charge-projectability-obstruction)
below additionally restricts autonomous closure for one such allocation
class; it does not remove its algebraic conservation.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone decides whether a particular collective extension has sufficient
independent justification for investigation. No extra state, interaction
law, source acquisition or empirical evaluation is installed by these
conditional identities.

## Independent electroquasistatic reference and observation map

<a id="charge-spatial-reference-regime"></a>

The physical reference is a specified macroscopic regime, not a proposed
microscopic origin of electric charge. Use fixed isotropic linear permittivity
`epsilon>0`, a prescribed nonnegative smooth conductivity `sigma(z)` of
compact support, no impressed current, and potential tending to zero at
spatial infinity. Position `z` and physical time `t_lab` are independently
defined, and `rho_e` is the free-charge density; the fixed linear dielectric
response has no independent polarization state. In the electroquasistatic
approximation the conventional equations are

\[
 -\epsilon\nabla^2\phi=\rho_e,\quad E=-\nabla\phi,\quad
 J=\sigma E,\quad
 \partial_{t_{lab}}\rho_e=-\nabla\cdot J
      =\nabla\cdot(\sigma\nabla\phi).                    \tag{16}
\]

The separate premises are documented in the authors' MIT edition of
*Electromagnetic Fields and Energy*: [charge conservation, section 2.3](https://web.mit.edu/6.013_book/www/chapter2/2.3.html),
[quasistatic approximation and its error, section 3.3](https://web.mit.edu/6.013_book/www/chapter3/3.3.html),
[Poisson's equation, section 4.2](https://web.mit.edu/6.013_book/www/chapter4/4.2.html),
and [Ohmic constitutive law, section 7.1](https://web.mit.edu/6.013_book/www/chapter7/7.1.html).
These source sections were reviewed on 2026-10-10. Fixed conductivity and
instantaneous linear polarization exclude independent material memory,
chemical drive and magnetic induction at the retained order. They are
restrictions of this reference, not universal electromagnetic assumptions.
A physical application would have to bound the neglected fields and
material-law error; a structural clock or small numerical step does not do so.

For a localized smooth charge distribution the specified boundary condition
gives the potential

\[
 \phi(z)=\frac1{4\pi\epsilon}
       \int\frac{\rho_e(z')}{|z-z'|}\,d^3z'.             \tag{17}
\]

This is the reference model's [superposition integral, section 4.5](https://web.mit.edu/6.013_book/www/chapter4/4.5.html).
It makes the full spatial density sufficient to determine the electric
field in this setting. It does not make the *net* charge sufficient.
For example, ideal equal-and-opposite point charges of magnitude `Q_*>0`
at `+a e_x,-a e_x`,
or at `+a e_y,-a e_y`, both have zero total. Potential probes at
`R e_x,R e_y`, with `R>a>0`, give the response matrix

\[
 \frac{Q_*}{4\pi\epsilon}
 \left(\frac1{R-a}-\frac1{R+a}\right)
 \begin{pmatrix}1&0\\0&1\end{pmatrix}.                 \tag{18}
\]

Its rank is two. This is an algebraic static extended-source control with
specified sources and ideal noninvasive probes, not an acquisition or
transport experiment. It establishes why the earlier TNFR response-rank
obstruction is not by itself a discriminator against conventional electric
charge. It does not require the two reference sources to arise under (1).

### A candidate complete density observation, before calibration

Supply fixed disjoint spatial cells with normalized smooth shapes `w_i(z)`,
each supported in its cell, so `integral w_i=1`. Their locations, sizes and
orientations are input geometry, not deductions from slot labels or graph
conductances. Consider the candidate

\[
 q_i=\kappa(x_i-r),\quad
 \rho_e(z)=\sum_i q_i w_i(z),\quad
 \tau=b(t_{lab}-t_0),\quad b>0,\quad\kappa\ne0.         \tag{19}
\]

Here `r` is fixed as in (12), `kappa` has units C/X, `w_i` has units
`m^-3`, and `b` converts seconds to the declared structural clock. The
ideal cell observation is `integral_cell rho_e=q_i`; a potential observation
would instead consume (17) and the sensor's specified spatial weighting.
Every charge derivative transforms as `dq_i/dt_lab=b*kappa*x_i'`.
Neither `x_i` nor `theta_i` has been called voltage or physical position.

This is a fully stated *candidate map class*, not independently calibrated
apparatus. A real comparison would need source preparation, fixed `r`,
spatial shapes, clock, scales, sensor loading and error bounds established
before reserved evaluation. None are estimated here. The argument below
fails already for exact, noiseless observations for every nonzero `kappa`
and positive `b`, so fitting those missing calibrations cannot repair it.
An approximate restricted bridge would need its own error and domain proof.

## Full-density closure obstruction under the retained law

<a id="full-density-charge-closure-obstruction"></a>

With fixed `sigma`, `epsilon`, geometry and boundary data, (16)--(17)
assign one density derivative to each complete density. In particular
`rho_e=0` everywhere implies `phi=E=J=0` and `rho_e_dot=0`.
This is a consequence of **Poisson plus the stated current law and boundary
condition**. Electrostatics alone does not specify charge transport. Zero
net charge, or zero charge only in a region with externally imposed field,
would not give this conclusion.

Reuse the actual same-slot 54-node support of
[organized external interaction](ORGANIZED_EXTERNAL_INTERACTION.md#1-complete-state-support-and-balances).
Let `alpha=2*pi/9`. At every host node `(c,j)` set
`x_H=r, theta_H=alpha*j`; at every donor node set
`x_D=r, theta_D=alpha*j+beta`. Compare `beta=0` and `beta=pi/2`.
These are two allowed mathematical initial states of the complete quarter
law on exactly the same support. They are not asserted to be reached
members of the frozen acquisition box. Their old internal and outer phase
profiles are well defined; no finite maintenance certificate is inferred.

The complete density (19) is zero for both. All form Laplacians and all
phase rates vanish initially. The two internal cycle currents cancel at
each node, and host outer currents vanish. The remaining contact terms are

\[
 x'_{H,c,j}=\frac\gamma4\sin\beta,\qquad
 x'_{D,j}=-\frac{5\gamma}4\sin\beta,\qquad
 \theta'=0.                                             \tag{20}
\]

Thus all derivatives vanish at `beta=0`, whereas at `beta=pi/2` each host
cell gains `b*kappa*gamma/4` and each donor cell loses
`5*b*kappa*gamma/4` in physical-time charge units. The total remains zero:
45 host contributions cancel the nine donor contributions. Disjoint shapes
make the density derivative in the `beta=pi/2` preparation nonzero as a
spatial function.

Consequently no deterministic density-only evolution, including (16), can
represent both states through (19). This is stronger than two equal *net*
inventories having different currents: the **entire mapped spatial density**
and all reference inputs agree. The site classification explains why adding
a circular phase term to an identical local conserved density does not fix
this full-domain map. It does not prohibit retaining phase as additional
material state, a collective density, or a justified smaller invariant family.

The phase-sensitive transfer mechanism was already established by the
[regional interaction owner](SINE_PAIR_INTERACTION.md#sine-autonomous-regional-transfer).
Equation (20) specializes its cancellation argument to the actual uniform
quarter law and support. The new conclusion is the specified physical-map
obstruction, not a newly discovered phase response. The
[execution controls](../../tests/physics/test_organized_charge_observation.py)
check all 54 rates and reciprocal balance through the shared contact observer;
they are neither a physical experiment nor proof of the general classification.

The gate therefore closes with **a scoped exact map/law incompatibility**.
Installing (17) as a TNFR readout alone would merely supply the conventional
electrostatic law. Matching that static readout would give no independent
discriminator; omitting phase from the full transport still fails (20).
Physical charge, non-Ohmic materials, polarization state, memory, other laws
and prepared subdomains remain open. The earlier
[P2 and Volts decisions](../research/PASSIVE_TRANSPORT_PROTOCOL.md) retain
their distinct pure-EPI laws and failed source admissions. No archived
response was replayed and no empirical dataset was admitted here.

## Homogeneous local edge redistribution does not close one density field

<a id="local-edge-charge-projectability-obstruction"></a>

The edge allocation in (15) preserves the total inventory algebraically.
That property does not establish an autonomous law for its spatial density.
The following obstruction covers one explicitly specified extension and
uses the unchanged complete quarter law, without a numerical trajectory.

Give each edge an arbitrary orientation. Let the incidence matrix B have
-1 at its tail and +1 at its head, so that L=BB^T and
S(theta)=-B sin(B^T theta). Retain one held reference r and a nonzero scale
kappa. Consider the observation

\[
 q(x,\theta)=\kappa(x-r\mathbf1)
       +B\,p(B^Tx,B^T\theta),\qquad \kappa\ne0.       \tag{E1}
\]

Here the same scalar C1 function p(u,v) is applied to every edge. It is
2*pi-periodic in v and satisfies p(-u,-v)=-p(u,v), so reversing a supplied
edge orientation leaves q unchanged. Dependence on other edges, independent
material state, unequal edge laws and history is excluded from this class.
The observation is defined on a full-state neighborhood of the equilibrium
under consideration; a selected lower-dimensional preparation set is a
different domain. Its total remains kappa*sum_i(x_i-r).

By a smooth density closure mean a C1 field G with

\[
 Dq(x,\theta)F(x,\theta)=G(q(x,\theta))               \tag{E2}
\]

throughout that neighborhood, where F is both full rows of (1). The claim
allows an arbitrary G, including spatially nonlocal linearizations. No
specific electrostatic kernel or conductivity is needed for the obstruction.

### Necessary mode condition

First take a uniform equilibrium x=r*1, theta=theta_0*1. Put
a=p_u(0,0) and b=p_v(0,0). Since p(0,0)=0, q=0 there, and

\[
 Dq_*=[\kappa I+aL,\ bL],\qquad
 DF_*={1\over4}
 \begin{pmatrix}-L&-\gamma L\\ \gamma L&0\end{pmatrix}.
                                                               \tag{E3}
\]

At an equilibrium the derivative of (E2) gives
Dq_* DF_*=DG(0)Dq_*. For an eigenvector of L with positive eigenvalue
lambda, define c_lambda=(kappa+a*lambda,b*lambda). A perturbation on that
mode which belongs to ker(c_lambda) must have zero observed derivative.
Consequently c_lambda times [[-1,-gamma],[gamma,0]] must be proportional
to c_lambda when that row is nonzero. Equivalently,

\[
 P(\lambda)=
 \gamma(\kappa+a\lambda)^2
 -b\lambda(\kappa+a\lambda)
 +\gamma b^2\lambda^2=0.                              \tag{E4}
\]

If c_lambda=0, (E4) also holds automatically. Thus a mode erased by the
observation does not evade this necessary condition. The argument does
not assume that DG(0) is a function of L or initially diagonal in its
eigenbasis: kernel preservation alone implies (E4).

The polynomial P has degree at most two and constant term
gamma*kappa^2, which is nonzero. A graph with at least three distinct
positive Laplacian eigenvalues therefore cannot admit (E2) for any a,b,
and hence for any p in the declared class. This is a necessary linearization
obstruction to exact nonlinear closure; satisfying it on some smaller
spectrum would not prove nonlinear closure.

### The obstruction also holds near an acute C9 twist

This conclusion need not rely on a uniform-phase neighborhood. Orient a
C9 cycle consecutively and take x=r*1 and theta_j=theta_0+2*pi*j/9 modulo
2*pi. Every oriented edge has the same circular difference
delta_*=2*pi/9. The closing edge differs by a full turn, so periodicity
gives the same p and derivatives there. A constant edge allocation has
zero incidence divergence on this oriented cycle, hence q_*=0. Put

\[
 c=\cos(2\pi/9)>0,\qquad
 a=p_u(0,\delta_*),\quad b=p_v(0,\delta_*).
\]

The true tangent at this equilibrium replaces the upper-right block in
(E3) by -gamma*c*L; the lower-left block remains gamma*L. The required
polynomial is therefore

\[
 P_c(\lambda)=
 \gamma c(\kappa+a\lambda)^2
 -b\lambda(\kappa+a\lambda)
 +\gamma b^2\lambda^2=0.                              \tag{E5}
\]

Its constant term is gamma*c*kappa^2, again nonzero. The cycle has the
four distinct positive eigenvalues

\[
 \lambda_k=2-2\cos(2\pi k/9),\qquad k=1,2,3,4.
\]

They are strictly increasing because their arguments lie in (0,pi).
Four distinct roots are impossible for the nonzero polynomial (E5).
Thus no observation (E1) has a smooth autonomous density law on any
full-state neighborhood of this acute winding-one equilibrium. This
uses the actual twisted tangent, not a transferred consensus certificate.
It does not show that the retained finite acquisition history fills such
a neighborhood, nor exclude a map defined only on an admitted smaller
invariant family.

The conclusion strengthens the conservation-only example (15): adding one
homogeneous local edge allocation cannot make the mapped density a complete
state for the unchanged quarter dynamics on these domains. It does not
exclude a density accompanied by independent internal material variables,
unequal or nonlocal observations, a memory law, or a different complete
model. Nor does it identify the mathematical edge allocation with physical
polarization. A fixed nonzero clock conversion rescales the tangent and
leaves its kernel condition unchanged.
