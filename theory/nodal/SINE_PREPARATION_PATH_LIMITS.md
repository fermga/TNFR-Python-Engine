# Whole-family limits after the certified winding-two passage

<a id="sine-preparation-path-limits"></a>

Every complete trajectory from the previously declared interior preparation
corridor converges to one of 57 energy-compatible, labeled relative critical
geometries. Only three are locally attracting: consensus and the two uniform
unit windings. The other 54 are hyperbolic saddles. Their stable manifolds
give a precise exceptional-source qualification, not permission to omit them
from a whole-family statement.

This result continues the same trajectories from the
[energy-isometric source path](SINE_PREPARATION_PATH_SELECTION.md#sine-preparation-path-selection).
It uses neither a new preparation nor a later sampled endpoint. The list is
an energy-compatible set of possible limits, not a claim that this source
family reaches every listed geometry. The argument proves convergence but
does not select a limit, a final winding for an antipodal limit, or a
uniform capture time.

<a id="sine-path-limits-law-and-energy"></a>
## The carried source and its strict energy threshold

Keep the isolated unit C9 with labeled nodes \(0,\ldots,8\), unnormalized
Laplacian \(L\), degree matrix \(2I\), capacities one and complete state
\((x,\theta)\). The two rows, structural clock and storage are
\[
 \begin{gathered}
 A=L/2,\qquad \gamma=1/(1023\pi),\qquad
 \tau=(1023/1024)t,\\
 x'=-Ax+\gamma f(\theta),\qquad
 \theta'=\gamma Ax,\qquad
 f_i(\theta)=\tfrac12\sum_{j\sim i}\sin(\theta_j-\theta_i),\\
 H(x,\theta)=\tfrac12x^TLx+
       \sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)],\qquad
 H'=-\tfrac12\|Lx\|_2^2.
 \end{gathered}\tag{1}
\]
The original-time rows are both multiplied by \(1023/1024\).
The complete sine law, support, positive loss and clock are unchanged.
Form and lifted-phase means remain zero. No forcing or event is added.

For preparation parameter \(s\in[3/4,4/5]\), retain exactly
\[
 x_s(0)=Q_s(ma+\xi),\qquad \theta_s(0)=\zeta,\qquad
 |\xi_j|,|\zeta_j|\le10^{-10},\qquad
 \mathbf1^T\xi=\mathbf1^T\zeta=0 .
 \tag{2}
\]
The exact \(M=512\), \(m\), source \(a\), Cayley map \(Q_s\), correlated
errors, initial phase/mean/storage equality and source budget \(10^{15}\)
are those of the
[path construction](SINE_PREPARATION_PATH_SELECTION.md#sine-preparation-cayley-path).
In particular \(Q_s\) is an invertible map of the zero-mean form space;
it is not a time-dependent input or a symmetry of the full flow.

The [uniform interior certificate](SINE_PREPARATION_PATH_SELECTION.md#sine-preparation-path-capture)
proves that every such actual source has winding two at \(\tau=200\) and
\[
 H(200)<\overline H:=7.337282704
 <H_2^*:=9[1-\cos(4\pi/9)],\qquad H_2^*>7.437166400 .
 \tag{3}
\]
All decimals in this owner denote exact rationals. Subsequent storage
cannot exceed \(H(200)\). The state at \(\tau=200\), its hidden
coordinates, both means and its accumulated loss are carried from (2);
none is replaced by the source proxy.

Equation (3) already excludes persistent residence in the closed
winding-two sector. The additional conclusion here requires classifying
all critical states below this energy and proving convergence to one of
them. The certified transit time remains 200; no new finite observation horizon
is introduced.

<a id="sine-path-limits-critical-census"></a>
## Exact critical geometries below the inherited energy

At a complete equilibrium, the phase row gives \(Lx=0\), so the zero form
mean gives \(x=0\). The form row then gives \(f(\theta)=0\).
Orient every cycle edge from \(j\) to \(j+1\bmod9\). Criticality is
equivalent to one common oriented sine current on all nine edges.

First suppose that current is positive and less than one. There is
\(0<\delta<\pi/2\) such that every principal gap is either
\(\delta\) or \(\pi-\delta\). If \(j\) gaps are obtuse, put \(n=9-2j\).
The winding \(k\) and potential satisfy
\[
 n\delta+j\pi=2\pi k,\qquad
 U=9-n\cos\delta .
 \tag{4}
\]
Since \(U<\overline H<9\), one must have \(n>0\), hence \(j\le4\).
The positive gaps sum to less than \(9\pi/2\), so \(k\) is one or two.
For \(k=1\), positivity of \(\delta\) leaves exactly
\[
 \begin{array}{c|c|c}
 j&\delta&U\\ \hline
 0&2\pi/9&9[1-\cos(2\pi/9)]\\
 1&\pi/7&9-7\cos(\pi/7).
 \end{array}\tag{5}
\]
For \(k=2\), the possibilities \(j=0,1,2,3\) instead have
\[
 \delta=\pi/2-\pi/(2n),\qquad
 U=9-n\sin\frac{\pi}{2n}\ge H_2^* .
 \tag{6}
\]
The inequality follows because \(n\sin(c/n)\) increases on this range:
its derivative is \(\sin z-z\cos z>0\) for \(0<z<\pi/2\).
These states are excluded by (3). Negative current is the exact
sign-reflected classification.

Current \(+1\) or \(-1\) would require all nine gaps to equal
\(+\pi/2\) or \(-\pi/2\). Their sum is not an integer multiple of
\(2\pi\), so neither case is a cycle geometry. This excludes the
zero-cosine degeneracy.

For zero current, every circular gap is zero or \(\pi\).
Cycle closure requires an even number \(j\) of antipodal edges, and
\(U=2j\). The only cases below \(\overline H<8\) are \(j=0\) and
\(j=2\). Assigning signs to the representative \(\pi\) gaps does not
create different circular configurations. An arbitrary wrapping convention
at these edges does not define a robust winding.

The complete below-threshold census is therefore:

| Relative critical geometry | Number on labeled C9 | Potential | Relative negative Hessian index |
| --- | ---: | --- | ---: |
| Consensus | \(1\) | \(0\) | \(0\) |
| Uniform gaps \(+2\pi/9\) or \(-2\pi/9\) | \(2\) | \(9[1-\cos(2\pi/9)]\) | \(0\) |
| One gap \(\pm6\pi/7\), the other eight \(\pm\pi/7\), with a common sign | \(18\) | \(9-7\cos(\pi/7)\) | \(1\) |
| Two antipodal edges, the other seven aligned | \(\binom92=36\) | \(4\) | \(2\) |
| **Total** | **\(57\)** | | |

All displayed potentials are below \(\overline H\): the two nonzero-current
values are less than \(9/2\) and \(11/2\), respectively, by
\(\cos(2\pi/9)>1/2\) and \(\cos(\pi/7)>1/2\).
Thus the table is
exactly the critical set permitted by this energy threshold. It removes
one common phase origin but retains node labels and the two handedness
choices. Graph rotations or reflections are not divided out. This count
is not a count of physically identified objects or a reachability result.

### Shared exact odd-cycle closure

The underlying geometric enumeration extends to any odd cycle size
\(N\ge3\). Write its principal sine branch in turns as
\(a_{\rm pr}\in(-1/4,1/4)\). Every edge is either \(a_{\rm pr}\) or
\(1/2-a_{\rm pr}\) modulo one turn. For a chosen mask of \(j\)
supplementary edges, exact closure has the form
\[
 p=(N-2j)a_{\rm pr}+j/2,\qquad
 a_{\rm pr}=\frac{2p-j}{2(N-2j)},\qquad
 p\in\mathbb Z,\quad -N/4<p<3N/4,\quad |a_{\rm pr}|<1/4 .
 \tag{6a}
\]
Oddness makes \(N-2j\ne0\). The missing quarter-turn endpoints would
require all edges to have the same current \(+1\) or \(-1\), impossible
for odd cycle closure. Thus (6a) and all masks give every labeled
critical geometry once. The zero-current masks are included; closure
then requires \(j\) even. These representatives are continuous edge
turns for solving closure, not necessarily principal winding increments.

For C9 the numbers of admitted periods for masks with
\(j=0,\ldots,9\) are \(5,4,3,2,1,0,1,2,3,4\).
Weighting by \(\binom9j\) gives 630 complete critical geometries modulo
common phase before applying the energy test. The preceding table is
their 57-member below-threshold subset.

<a id="sine-path-limits-stability"></a>
## Relative nondegeneracy and the complete two-row stability

Let \(B\) be the oriented cycle incidence matrix. At any listed geometry,
the phase Hessian is
\[
 H_\theta=B^T\operatorname{diag}(c\sigma_e)B,\qquad
 c>0,\quad \sigma_e\in\{-1,1\},
 \tag{7}
\]
with \(j=0,1,2\) negative signs. Relative nodal perturbations map
isomorphically to edge perturbations \(v\) with \(\mathbf1^Tv=0\).
On the full edge space, the diagonal form has inertia \((9-j,j,0)\).
Its orthogonal complement to that constrained subspace, in this quadratic
form, is spanned by
\(\operatorname{diag}(c\sigma_e)^{-1}\mathbf1\), whose squared form is
\[
 \mathbf1^T\operatorname{diag}(c\sigma_e)^{-1}\mathbf1
 =(9-2j)/c>0 .
 \tag{8}
\]
Removing this one positive direction proves relative inertia
\((8-j,j,0)\). In particular every listed equilibrium is nondegenerate
after removing common phase.
For the general odd-\(N\) closure in (6a), the same scalar is
\((N-2j)/c\), which is never zero. Removing its sign from the full
diagonal inertia proves
\[
 \operatorname{ind}_-(H_{\rm relative})
   =j-\mathbf1_{\{2j>N\}},\qquad
 \operatorname{null}(H_{\rm relative})=0 .
 \tag{8a}
\]
This is the shared geometric count; it does not itself supply a dynamical
stability or basin verdict.

The [positive-loss reciprocal Jacobian argument](SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition)
applies to the actual C9 matrices. On the common-mean-zero space set
\(C=H_\theta/2\). Linearization of both rows in (1) gives
\[
 \begin{pmatrix}X\\Y\end{pmatrix}'=
 \begin{pmatrix}-A&-\gamma C\\ \gamma A&0\end{pmatrix}
 \begin{pmatrix}X\\Y\end{pmatrix}.
 \tag{9}
\]
Writing \(Y=A^{1/2}z\) reduces the characteristic equation to
\[
 P(\lambda)=\lambda^2I+\lambda A+
       \gamma^2 A^{1/2}CA^{1/2}.
 \tag{10}
\]
Here \(A\) is positive definite on the eight-dimensional relative space.
For a nonreal characteristic root, the imaginary-part identity forces
strictly negative real part. On the nonnegative real axis,
\(P'(\lambda)=2\lambda I+A\) is positive definite. Each negative
stiffness direction crosses zero once, and there is no relative zero
root. Thus exactly \(j\) full-state modes are unstable, and \(16-j\)
are stable. All 57 equilibria are hyperbolic on the conserved-mean leaf.

Consequently consensus and the two uniform unit windings are the only
locally exponentially attracting geometries below the energy threshold.
They represent three oriented geometries and two winding magnitudes,
zero and one. The 18 one-obtuse configurations have one unstable
direction; the 36 two-antipodal configurations have two.
Phase Hessian stability alone is not substituted for the coupled
form/phase calculation.

<a id="sine-path-limits-convergence"></a>
## Every actual source converges to one listed complete state

At and after \(\tau=200\), the zero form mean and
\(\tfrac12x^TLx\le H(200)\) bound the form coordinates.
Circular phases lie in a compact torus, so the carried storage sublevel
is a compact forward-invariant state domain.
LaSalle's invariance principle confines the omega-limit set to the largest
invariant subset of \(Lx=0\). There \(x=0\), and invariance of the form
row requires \(f(\theta)=0\).

The conserved lifted phase sum supplies the additional circular invariant
\[
 \prod_{j=0}^8 e^{i\theta_j(\tau)}
 =e^{i\sum_j\theta_j(0)}=1 .
 \tag{11}
\]
Each relative critical geometry intersects this product-phase slice in
exactly nine common rotations: a rotation by \(\varphi\) multiplies the
product by \(e^{9i\varphi}\). Therefore the relevant circular equilibrium
set has only \(9\cdot57\) points, even though unrestricted common rotation
is a continuous symmetry.

An omega-limit set of a precompact continuous trajectory is nonempty,
compact and connected. For completeness, each closure of a trajectory
tail is connected, and the omega-limit set is their nested intersection.
A connected subset of the finite critical set in (11) contains a single
point. Hence every actual source in (2) converges in form and circular
phase to one complete equilibrium from the table.

This also proves convergence of the originally continued real phase lifts.
Around the limiting circular phases choose fixed small arcs. At sufficiently
late times each phase remains in its arc; continuity fixes its integer lift
branch there. Each real phase therefore converges, and the limiting lift
sum retains its original value zero. The terminal integer-turn vector and
common phase origin are consequences of the actual history, not reinitialized
coordinates. No convergence rate, common deadline or last-passage time is
derived by this argument.
The same finite-set argument proves convergence for any finite source
on this declared C9 mean leaf, using all 630 relative geometries.
The inherited strict energy bound narrows the possible limits to 57;
it is not needed for the general convergence fact.

The limits with no antipodal edge have eventual constant winding:
zero for consensus and \(\pm1\) for both the uniform and the one-obtuse
configurations. A two-antipodal limit has no convention-independent winding
at the limit; convergence to it does not prove eventual residence in one
open winding sector.
All trajectories eventually avoid the closed sectors with winding magnitude
at least two, because none of the listed limits lies in those closed sectors.
For a two-antipodal geometry their possible limiting signed gap sums are
only \(-2\pi,0,2\pi\).

<a id="sine-path-limits-exceptional-sources"></a>
## Exceptional saddle sources and what remains undecided

For each fixed \(s\), the map
\[
 (\xi,\zeta)\longmapsto\bigl(Q_s(ma+\xi),\zeta\bigr)
 \tag{12}
\]
is an invertible affine chart from the sixteen-dimensional zero-sum
latent space onto the corresponding full-state mean leaf near this
source. The phase-error box is small enough to be an injective local
chart of the circular phases. No latent coordinate is discarded.

The stable-manifold theorem gives the full-state saddle basins as global
stable manifolds of dimensions 15 or 14, respectively. Equivalently these
sets can be covered by countably many backward images of local stable
manifold charts. The smooth complete flow and the fixed-\(s\) source chart
preserve their dimensions. The latent sources converging to an index-one
saddle therefore lie in a countable union of codimension-one immersed
submanifolds; those converging to an index-two saddle lie in the corresponding
codimension-two union.

In particular the union of all saddle-source subsets has sixteen-dimensional
Lebesgue measure zero for every fixed \(s\). Outside that exceptional set,
each source converges to consensus or one of the two uniform unit windings.
This is a statement about the dimension and measure of specified sets,
not an assigned probability of any outcome. It does not give the basin
volumes of the three attractors, an outcome distribution, or a deterministic
class selector.

Nor does it remove the exceptional sources from the admitted closed error
family. Their intersection with that family has not been proved empty.
A one-dimensional nominal source path may intersect or even locally follow
a stable manifold; ambient measure zero cannot settle its outcome.
No claim that every source is captured by a stable identity follows.

If a particular trajectory converges to one of the three attractors,
it eventually enters a small forward-trapped identity neighborhood of that
target. The preceding argument neither supplies that entry time nor decides
which target is selected. A saddle limit is a convergent critical
configuration, not an attracting acquired identity.
The present restriction is therefore stronger than exclusion of winding two
but weaker than whole-family capture by one organization.

The separate [source-specific capture proof](SINE_PREPARATION_PATH_CAPTURE.md)
now supplies that additional bridge for the original interior corridor.
It carries the actual mixed state into the winding-one storage trap and
excludes the other 56 candidates there, including every saddle exception.
The energy classification above remains valid for other sources; its
candidate list alone supplies no such selector.

<a id="sine-path-limits-implementation"></a>
## Implementation and evidence boundary

The [path source owner](../../src/tnfr/physics/_sine_preparation_path_selection.py)
owns the exact preparation, coupled semigroup bounds and inherited storage
threshold. The private
[limit classifier](../../src/tnfr/physics/_sine_preparation_path_limits.py)
rebuilds the C9 critical geometries, their energy admission and relative
inertias; the
[independent controls](../../tests/physics/test_sine_preparation_path_limits.py)
check closure, label counts, excluded energy classes and complete stability
dimensions.
Its finite algebraic list is not an evaluated forward history.
The exact closure and signed-edge inertia are shared in
[phase cycle geometry](../../src/tnfr/physics/phase_cycle_geometry.py)
through the odd-cycle formulas (6a) and (8a). Existing C5 consumers retain
their own support and complete-law admission; their topology-specific
reports are not transferred to C9.

The private limit API also accepts other primitive path intervals, times
and error budgets admitted by the source owner. It rebuilds that source
and storage bound, rather than accepting an incoming report. Finite-set
convergence remains valid when a source-policy flag fails; the source
policy and the energy filter have different responsibilities.
Equality with a critical energy, or an unresolved interval comparison,
retains the critical candidate. Antipodal candidates have unavailable
winding. If either latent error row has zero radius, the source is no
longer full dimensional and the API withholds the fixed-\(s\)
null-volume conclusion.

The conclusions use the fixed sine law and complete source family in (1)--(3).
They cannot be transferred merely from a winding label to another support,
constitutive law, contact geometry, clock or later interface.
No new trajectory, time-coefficient series, frozen response, source reset
or observed limit is used. Autonomous preparation and physical identification
remain separate questions.
