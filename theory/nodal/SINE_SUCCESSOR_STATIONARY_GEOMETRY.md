# Stationary obstruction on the proposed maintained-parent UM support

The full support predicted by [default parent-target UM](SINE_MAINTAINED_PARENT_UM.md)
has no stationary state in the closed acute edge-branch chamber containing
its hypothetical endpoint. A certified boundary minimum of the bare phase
potential proves this exclusion on all 21 fine coordinates. The argument
does not assume that an unknown stationary state preserves constituent
symmetry. It also gives a same-storage, same-means comparison path from the
predicted source to the acute boundary.

These are analytical results for the supplied conservative complete law.
No UM event, trajectory, autonomous selection or physical identification is
provided. Nonacute equilibria and nonstationary organizations are not excluded.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects further work; this conditional geometry does not activate it.

## 1. Complete law, source and every edge branch

Re-admit the retained THOL successor with the
[source consumer](../../src/tnfr/research/maintained_parent_um_source.py), then
rebuild its ideal default UM reset. No earlier producer or event is replayed.
The [chamber owner](../../src/tnfr/research/maintained_successor_chamber.py)
retains all 21 nodes, 53 edges, actual predicted conductances and capacities,
structural clock and declared phase lifts. The child is index 20 (`2_sub_0`);
the parent is index 2. The actual retained child remains isolated: this
successor support is still hypothetical.

On a continuous segment hold support, symmetric positive conductances and
positive capacities fixed, with zero forcing and the
[weighted-form/bare-phase law](SINE_PAIR_COUPLING.md#collective-coupling-weighted-law):

\[
 \dot x=M S_U(\theta),\qquad \dot\theta=M L_Wx,\qquad
 M_{ii}=\frac{\nu_i}{\pi d_i}>0,
\]

\[
 H=Q_W(x)+V_U(\theta),\quad
 Q_W=\frac12\sum_{\{i,j\}\in U}W_{ij}(x_j-x_i)^2,\quad
 V_U=\sum_{\{i,j\}\in U}[1-\cos(\theta_j-\theta_i)].
\]

Its conserved lifted means have weights `rho_i=d_i/nu_i`. Conductances
and capacities remain in this law and its means; only the stationary phase
equations below are independent of their positive values.

Orient every edge `(i,j)` with `i<j`. For its integer branch `k_ij`, define
`delta_ij=theta_j-theta_i-2*pi*k_ij`. Exactly eight branches are one:

```text
(0,8), (0,9), (1,8), (1,9),
(10,18), (10,19), (11,18), (11,19).
```

All other branches are zero. Let `C` require `|delta_ij|<pi/2` on every
edge, and let `C_bar` replace these by closed inequalities. Both domains
are convex in real lifts; common phase translation is the only gauge being
removed. The code constructs a deterministic 20-edge spanning tree and all
33 fundamental cycle rows, retaining the complete branch class. Two original
winding labels alone would discard part of this information.

Rational interval admission gives an ideal minimum acute margin approximately
`0.0106079513764`, and greater than `0.010607951375` under the preceding
conditional `2^-40` per-changed-output allowance. Thus the proposed source
is strictly interior. The allowance is an assumption about a future reset,
not a bound on an unevaluated implementation. Both conductance and capacity
positivity are rechecked on that envelope.

## 2. Stationarity and strict monotonicity

Since M is invertible, stationarity requires `L_W x=0` and `S_U(theta)=0`.
Connected positive support implies constant form. With `g=grad V_U=-S_U`,
the remaining question is whether `g=0` anywhere in `C_bar`.

For any two points y,z in that chamber, edge summation gives

\[
 (z-y)^T[g(z)-g(y)]
 =\sum_{e\in U}(\delta_e(z)-\delta_e(y))
       [\sin\delta_e(z)-\sin\delta_e(y)].
\]

Sine is strictly increasing on the entire closed interval `[-pi/2,pi/2]`.
Consequently the displayed expression is positive unless z-y is a common
constant. This proves uniqueness modulo translation if a stationary point
exists; it does not establish existence. Zero edge cosines at the boundary
do not invalidate this strict monotonicity argument.

The weighted conserved means also rule out a nonzero rigid common drift
as a relative equilibrium: a common rate with zero weighted sum must vanish.
This observation does not exclude motion of internal relative coordinates.

## 3. A certified boundary witness on all fine nodes

Use five real parameters `u=(p,q,t,b,v)` to construct a witness y:

| Fine nodes | Lifted phase |
| --- | --- |
| 0, 1 | 0 |
| 2 | p |
| 3 | q |
| 4, 5 | 2*pi-3*t |
| 6, 7 | 2*pi-2*t |
| 8, 9 | 2*pi-t |
| 10, 11 | b |
| 12, 13 | b+v |
| 14, 15 | b+pi/2 |
| 16, 17 | b+pi |
| 18, 19 | b+3*pi/2 |
| 20 | p |

Write this affine map as `y=A*u+pi*c`. It defines a proposed witness,
not a reduced complete dynamics or an assumption on every stationary point.
Solve the five restricted potential equations

\[
 F(u)=A^T g(Au+\pi c)=0.
\]

The [certificate owner](../../src/tnfr/research/maintained_successor_stationary.py)
admits a fixed rational center, radius and proposed inverse, independently
recomputes interval F and its Jacobian from all 53 edges, and certifies a
contraction `u -> u-R*F(u)` strictly into that box. The inverse is a witness
to check, not an assumed exact inverse. Pi and transcendental functions use
the shared outward rational interval arithmetic. Numerical root probes are
only a way to propose that witness; they supply no existence theorem.

The fixed center, in `(p,q,t,b,v)` order, is

```text
(1082587343106, 1211889604568, 1438506301057,
 113581483716, 891083831083) / 2^40,
```

with radius `r=2^-24`. Recomputed bounds give `eta<4.423e-13`, contraction
factor `<3.076e-6` and image radius `<6.256e-13<r`. Thus the witness root
exists inside the declared box; this is not an assertion that its rational
center is an exact root.
The contraction bound also makes `R*J` invertible at the center, hence the
square matrix R is invertible. A fixed point of `u-R*F(u)` therefore has
`F(u)=0`; an arbitrary proposed preconditioner is not assumed injective.

The existing unit-support stationary kernel cannot be applied unchanged:
its independent free-node coordinates differ from this affine face, whose
Jacobian is `A^T Hess(V) A`. The local certificate reuses shared interval
and exact-linear arithmetic without replacing the full phase equations.

At the enclosed root all 12 edges on the three complete B-chain blocks
`{14,15}->{16,17}->{18,19}->{10,11}` have forward gaps exactly `pi/2`.
Their exact values follow from the affine identities, not a tolerance check.
Every other edge remains strictly acute. The child-parent gap is exactly zero.

Full fine-node cancellation now gives more than restricted stationarity.
The two A-chain interior pairs 6,7 and 8,9 have zero gradient by equal
incoming/outgoing sine terms. The child also has zero gradient. Therefore
`F_p=g_2`, `F_q=g_3`, and `F_t=-6*g_4`, with `g_4=g_5`.
The B-chain interior pairs 16,17 and 18,19 cancel in the same way.
Then `F_v=2*g_12` and `F_b=2*(g_10+g_12+g_14)`.
Finally `sum_i g_i=0` forces `g_0=g_1=0`.

Thus the exact root's full gradient has only four nonzero entries:

\[
 g_{10}=g_{11}=-a,\qquad g_{14}=g_{15}=a,
 \qquad a>0,
\]

where

\[
 a=2(\cos v-1)+\cos(p-b).
\]

Its certified enclosure lies inside `(0.0145091,0.0145097)`. Every nonactive
edge has acute margin greater than `0.1971598`. Strict positivity and
all nonactive margins are interval conclusions over the entire root box.
In particular this witness is not an equilibrium.

## 4. Full closed-chamber exclusion

For arbitrary z in `C_bar`, sum the upper acute inequalities over all
12 forward edges of the three B-chain blocks. Intermediate node terms
cancel, leaving

\[
 z_{14}+z_{15}-z_{10}-z_{11}\ge\pi.
\]

Equality holds at y. Hence

\[
 g(y)^T(z-y)
 =a[(z_{14}+z_{15}-z_{10}-z_{11})-\pi]\ge0.
\]

Convexity of V on `C_bar` proves that y is its global minimum modulo
translation. If some distinct z were stationary, strict monotonicity would
require `(z-y)^T(0-g(y))>0`, contradicting this nonnegative normal product.
A common translate of y has the same nonzero gradient. Therefore **there is
no stationary phase state anywhere in the full closed acute chamber**.

This is a full-coordinate separating certificate, including the acute
boundary and asymmetric candidates. It does not infer a global obstruction
from a failed numerical root finder, a restricted ansatz, or negative
curvature at one candidate outside the chamber.

## 5. Storage and both means do not trap this source in the acute chamber

Let `(x_0,theta_0)` be the ideal hypothetical endpoint, with actual positive
weights and capacities. The same argument applies to any admitted conditional
endpoint. Its form storage `Q_0=Q_W(x_0)` is strictly positive: forms are
unchanged by UM and the parent and child have different forms on their new
positive-conductance edge. Write `h=H(x_0,theta_0)` and
`m_rho=(rho^T x_0)/(rho^T 1)`.
Fresh interval arithmetic gives `Q_0>0.0105` for both the ideal source and
the conditional envelope. Each envelope member uses its own fixed
conductances, capacities and weighted means in the following construction.

Translate y by a common phase so that its weighted phase mean equals the
source mean, and set `theta(s)=(1-s)*theta_0+s*y`, `0<=s<=1`.
This segment stays strictly acute until its boundary endpoint. Convexity
and minimality give `V(theta(s))<=V(theta_0)<h`. Define

\[
 x(s)=m_\rho 1+
 \sqrt{\frac{h-V(\theta(s))}{Q_0}}\,(x_0-m_\rho 1).
\]

The multiplier is at least one. The path begins at the proposed source,
retains both weighted means and has exactly `H(x(s),theta(s))=h`.
All 12 active gaps reach their acute faces transversely. A sufficiently
short extension crosses them while the radicand remains positive, by
continuity and the strict margin `h-V(y)>=Q_0>0`.

This comparison path establishes that the connected component of the
storage-and-means level containing the source is not separated from the
acute boundary. Those conserved quantities alone cannot provide the proposed
acute trapping certificate. The parameter s is not structural time; this
path is not a solution, a pressure reconstruction or a prediction of an
actual future boundary crossing.

## 6. Scope and verification

The retained source, predicted support and complete law fix this result.
It excludes an acute stationary identity and the stated storage-barrier
route. It establishes neither instability of every organization nor loss
of principal winding. An acute face has gap `pi/2`; a principal phase cut
has gap `pi`. Those are different boundaries of different identities.
The separate [principal-chamber theorem](SINE_SUCCESSOR_PRINCIPAL_REDUCTION.md#successor-principal-minimum-exclusion)
now excludes nonacute stationary storage minima and extends the invariant-shell
comparison to principal cuts. Nonstationary maintenance still needs its own
argument. Positive capacity rescaling changes rates and means but
cannot repair the present stationary phase obstruction.

Verification combines [branch and cycle controls](../../tests/research/test_maintained_successor_chamber.py),
[interval witness controls](../../tests/research/test_maintained_successor_stationary.py)
and [independent full-support checks](../../tests/research/test_maintained_successor_stationary_evidence.py).
The latter recompute fine currents and static comparison identities without
executing UM, THOL, a trajectory or a frozen producer. High-precision checks
are independent numerical controls; the exact inequalities and analytical
cancellations carry the theorem. Earlier frozen artifacts remain unchanged.
