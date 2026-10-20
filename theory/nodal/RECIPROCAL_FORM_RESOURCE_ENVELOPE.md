# Reciprocal form resources under arbitrary contacts

The [matched-contact preparation bound](ORGANIZED_FREE_PHASE_PROBE.md#free-phase-probe-preparation-reachability)
does not directly cover unequal interface weights or additional reacting
organizations. This owner derives a necessary form-resource bound for
arbitrary reciprocal contacts on a fixed finite set of nodes. It retains
every initial form, including donor means and internal deviations. Neither
more contact time nor a different contact schedule removes this bound.

This is a conditional consequence of the supplied constitutive family,
not a new TNFR axiom, a source mechanism or a sufficient preparation
construction. The [form foundation](../FUNDAMENTAL_THEORY.md) and
[parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md) retain their
definitions. Task status belongs only to the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

## 1. Complete law and admitted contact histories

Fix an ordered set of \(N\ge2\) labeled nodes, held unit capacities,
the nondimensional structural clock \(\tau\), and
\(\gamma=1/(1023\pi)\). Forms \(x_i\) are signed real coordinates;
phases \(\theta_i\) are circular, with continuous lifts used for evolution.
For each unordered pair prescribe a nonnegative measurable weight
\(w_{ij}(\tau)=w_{ji}(\tau)\), locally integrable in time. Zero weights
are allowed. No connectivity, uniform degree or common interface weight
is assumed. State-preserving changes of support are included in these
weight histories; there are no form or phase jumps, inputs, node births,
capacity changes or label changes.

Hold one \(\lambda\in[0,1]\) throughout the history and use

\[
\begin{split}
 V_\lambda(u)&=1-\cos u+
       \lambda\frac{(1-\cos9u)^2}{6\,9^4},\\
 j_\lambda(u)&=V_\lambda'(u)=\sin u+
       \frac{\lambda}{2187}
       \left(\sin9u-\frac12\sin18u\right),\\
 |j_\lambda(u)|&\le J_0=\frac{1459}{1458}.
\end{split}
\tag{1}
\]

Let \(L(\tau)\) be the conductance Laplacian and
\(U(\tau,\theta)=\sum_{i<j}w_{ij}(\tau)V_\lambda(\theta_i-\theta_j)\).
The same weights occur in both complete quarter-factor rows:

\[
 x_\tau=\frac{-Lx-\gamma\nabla_\theta U}4,
 \qquad \theta_\tau=\frac{\gamma Lx}4.
\tag{2}
\]

These equations specify the pressure independently of the response.
They are the comparison model's equations, not a native operator word
or a reciprocal-degree normalization. A clock change transforms both rows.
No laboratory-unit or physical-measurement interpretation is supplied.

Finite initial forms and phase lifts have a unique absolutely continuous
solution on every finite time interval. Indeed, the current is globally
bounded and Lipschitz, while the sum of the finitely many edge weights
is locally integrable. The full vector field consequently has integrable
Lipschitz and linear-growth bounds; its integral equation and Gronwall
bound give continuation. Edge antisymmetry gives
\(\sum_i x_i(\tau)=\sum_i x_i(0)\) and the analogous conserved sum
of the chosen phase lifts. Neither conclusion requires connectivity.
For state-dependent contact choices, the bound below applies along any
separately admitted solution with locally integrable realized weights.
This statement supplies neither a feedback selector nor its well-posedness.

The proof below uses the actual form row and a uniform bound on its
phase currents. It does not hold phases fixed, project to a carrier,
assume acute gaps, or replace the independently specified phase row.

<a id="reciprocal-form-resource-envelope"></a>

## 2. An invariant envelope of sorted partial sums

Put \(\bar x=N^{-1}\sum_i x_i(0)\) and \(y=x-\bar x\mathbf1\).
Sort only for the following observation, writing
\(y_{(1)}\ge\cdots\ge y_{(N)}\). The physical labels are unchanged.
Define

\[
\begin{split}
 S_k(y)&=\sum_{i=1}^k y_{(i)}
       =\max_{|A|=k}\sum_{i\in A}y_i,
       \qquad S_0=S_N=0,\\
 D_k&=\frac{k(N-k)}2,\qquad
 q(y)=\max_{1\le k<N}\frac{S_k(y)}{D_k},\\
 c&=\gamma J_0,\qquad \rho=\max(c,q(y(0))).
\end{split}
\tag{3}
\]

On the zero-sum subspace, \(q\ge0\), with equality only at zero.
It is a finite maximum of linear functions, hence Lipschitz and
absolutely continuous along the solution. The conditional theorem is

\[
 \boxed{\max(c,q(y(\tau)))\le\max(c,q(y(0))),\qquad
 S_k(y(\tau))\le\rho D_k\quad(1\le k<N).}
\tag{4}
\]

To prove it, choose any index \(k\) active in (3) at a state where
\(q\ge c>0\). The neighboring partial sums satisfy their same
\(qD_j\) bounds, including \(S_0=S_N=0\). Therefore

\[
\begin{split}
 y_{(k)}&=S_k-S_{k-1}\ge q(D_k-D_{k-1}),\\
 y_{(k+1)}&=S_{k+1}-S_k\le q(D_{k+1}-D_k),\\
 y_{(k)}-y_{(k+1)}&\ge q.
\end{split}
\tag{5}
\]

The top \(k\) set \(A\) is thus separated strictly from its
complement; ties within either set cause no problem. Internal edges
cancel in the derivative of its sum. At almost every time,

\[
 \frac{dS_k}{d\tau}
 =\frac14\sum_{\substack{i\in A\\j\notin A}}
 w_{ij}\left[y_j-y_i+\gamma j_\lambda(\theta_j-\theta_i)\right]
 \le\frac{c-q}{4}
       \sum_{\substack{i\in A\\j\notin A}}w_{ij}\le0.
\tag{6}
\]

This includes a disconnected cut, whose sum is zero. Every active
index has this property. The derivative rule for a finite maximum
therefore gives \(q_\tau\le0\) almost everywhere where \(q\ge c\).
More explicitly, all subset sums and their maximum are differentiable
outside one null set; at those times the derivative of the maximum is
an active derivative. No derivative of an edge weight is used.
Where \(q<c\), the function \(\max(c,q)\) is locally constant;
at its remaining differentiability points with \(q=c\), its derivative
is also nonpositive. Integrating this absolutely continuous function
proves (4), including arbitrary ties and measurable contact changes.

Equivalently, each polytope
\(\{y:\sum_i y_i=0,\ S_k(y)\le\rho D_k\}\), with \(\rho\ge c\),
is invariant. The endpoint inequalities give the useful corollary

\[
 -\frac{\rho(N-1)}2\le y_i(\tau)\le\frac{\rho(N-1)}2,
 \qquad \max_i x_i-\min_i x_i\le\rho(N-1).
\tag{7}
\]

## 3. Marked nine-node means and node-count resources

Now let \(N\ge18\), and retain any two disjoint marked sets
\(A,B\), each containing nine nodes. They need not be entire connected
components. Write \(r=\frac19\sum_Ax_i-\frac19\sum_Bx_i\) and
\(v=\gamma r\), as in the free-probe preparation. At every time,

\[
 \boxed{|r|\le\frac{S_9(y)+S_{N-9}(y)}9
                  \le\rho(N-9),\qquad
 |v|\le\gamma\rho(N-9).}
\tag{8}
\]

Indeed, a nine-node sum is no greater than the top-nine sum and
no smaller than the bottom-nine sum \(-S_{N-9}\). Interchanging
\(A,B\) gives the absolute-value bound. This observation retains
all other nodes through the initial \(q\) and the fixed total \(N\).

If every initial form is the same, \(q(y(0))=0\) regardless of
the actual initial phases. For the original eighteen nodes, (8) yields

\[
 |v(\tau)|\le9\gamma^2J_0<9\,10^{-7}<12.
\tag{9}
\]

The strict numerical inequality uses \(\pi>157/50\) in the declared
\(\gamma\), not an evaluated response. Arbitrary unequal interfaces
and support switching on these nodes cannot prepare the target.
The value \(v=12\) is a sufficient choice for the declared full-turn
probe, not a necessary preparation for every possible discrimination.
For additional nodes also starting at that same common form, a
necessary node-count condition for \(|v|=12\) is

\[
 N\ge\left\lceil9+\frac{12}{\gamma^2J_0}\right\rceil.
\tag{10}
\]

This is not a sufficient construction, a proposed reservoir, a minimum
number of physically realized constituents, or proof that the bound is
dynamically attainable. Donors with other initial forms instead contribute
to \(q(y(0))\); they cannot be counted as zero-resource additions.

The observation bound is sharp as a statement about the polytope alone.
Its arithmetic-progression vertices
\(y_{(i)}=\rho(N+1-2i)/2\) attain every partial-sum cap and the
top-nine versus bottom-nine bound in (8). No claim follows that the
complete phase/form dynamics reaches those vertices from a given source.

## 4. Primitive preparation information cannot be discarded

The function \(q(x-\bar x\mathbf1)\) is unchanged by a common form
offset and by a simultaneous permutation of node labels. It is not
determined by child means or by donor count. To admit uncertain primitive
forms, let \(\widehat x\) be an ordered record with
\(|x_i(0)-\widehat x_i|\le\varepsilon\) for every node. For any subset
\(A\) of size \(k\), its centered sum has coefficients
\((N-k)/N\) on \(A\) and \(-k/N\) on its complement. Hence

\[
 |q(x-\bar x\mathbf1)-q(\widehat x-\overline{\widehat x}\mathbf1)|
 \le\frac{4\varepsilon}{N},\qquad
 \rho\le\max\left(c,
      q(\widehat x-\overline{\widehat x}\mathbf1)+\frac{4\varepsilon}{N}\right).
\tag{11}
\]

Each normalized subset functional has exactly that error bound;
taking their finite maximum proves the first inequality. The upper
endpoint is attained for every record: choose an active maximizing
subset of \(\widehat x\), add \(+\varepsilon\) on it and
\(-\varepsilon\) on its complement. That functional increases by
\(4\varepsilon/N\). Thus the exact box supremum of the true \(q\)
is the recorded \(q\) plus \(4\varepsilon/N\). In particular,
\(|x_i(0)-m_0|\le\varepsilon\) about any supplied common origin gives
\(q(y(0))\le4\varepsilon/N\). Two nonempty groups at the opposite
endpoints \(m_0\pm\varepsilon\) attain this constant. Uncertain
forms must be admitted before sorting or centering.

A simple full-state witness explains why zero child means are insufficient.
Take two marked winding-one C9 phase carriers with equal phase origins.
In A give two distinct slots forms \((-H,+H)\), \(H>0\), with all
other forms zero in both children. Both child means and the global mean
are zero. Keep the internal cycles and add only one cross edge of weight
\(w>0\), from the \(-H\) slot to its matching zero-form slot in B.
The phase current of this edge is zero at this state. Nevertheless,

\[
 r_\tau(0)=\frac{wH}{18},\qquad
 v_\tau(0)=\frac{\gamma wH}{18},\qquad q(y(0))=\frac{2H}{17}.
\tag{12}
\]

Internal terms cancel in each mean, and the two endpoint form rates
from that cross edge are \(wH/4\) and \(-wH/4\). Thus child means
alone miss an arbitrarily large initial boundary rate; the full resource
in (3) retains it. This is an instantaneous analytic control, not a
trajectory, maintained-identity theorem or preparation of \(v=12\).

## 5. Contact work and scope of the exclusion

Nonnegative reciprocal weights need not be storage-passive when they
change. If they are absolutely continuous between finitely many support
jumps, the same complete rows give

\[
\begin{split}
 H&=\tfrac12x^\top Lx+U,\\
 H_\tau&=-\tfrac14\|Lx\|^2+
   \sum_{i<j}(w_{ij})_\tau
       \left[\tfrac12(x_i-x_j)^2+V_\lambda(\theta_i-\theta_j)\right].
\end{split}
\tag{13}
\]

A state-preserving weight jump changes \(H\) by the corresponding
sum with \(\Delta w_{ij}\). Arbitrary locally integrable weights need
not have a finite-variation work ledger. The form envelope requires
neither that ledger nor event passivity; the
[preparation work owner](ORGANIZED_FREE_PHASE_PROBE.md#free-phase-probe-preparation-reachability)
retains their separate obligations.

The proof remains valid even when constituent winding or an acute chart
is lost. It therefore excludes a preparation without having to assume
identity maintenance, but supplies no maintenance certificate when a
resource bound fails to exclude. Contact occurrence, source origin and
physical identification remain open. Asymmetric contacts, independently
weighted phase currents, changed capacities or normalization, form jumps,
inputs and node creation lie outside (2); no verdict transfers to them.

The detached calculation in
[`_reciprocal_form_resources.py`](../../src/tnfr/physics/_reciprocal_form_resources.py)
re-admits ordered primitive forms and their declared uncertainty before
computing a sufficient upper envelope. Its
[static controls](../../tests/physics/test_reciprocal_form_resources.py)
exercise cut cancellation, resource admission and counterexamples.
These calculations do not evaluate a trajectory, select a support history,
replay a formation source or certify physical preparation.
