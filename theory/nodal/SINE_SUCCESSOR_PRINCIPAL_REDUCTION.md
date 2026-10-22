# Principal-sector minima: exact reduction and full exclusion

The proposed UM support's full original principal chamber contains **no
stationary storage minimum**. This extends the closed-acute
[obstruction](SINE_SUCCESSOR_STATIONARY_GEOMETRY.md) through an exact
reduction from twenty relative fine phases to four real coordinates and
complete analytical sector exclusions. The proof retains all 53 edge
branches and the singular cases needed to justify the reduction. The
[source-specific corollary](#successor-principal-shell-access) also shows that
its storage and two means alone do not separate it from principal cuts.

This is a restriction of the existing phase potential for stationary
classification. It is not a new pressure law, a closed temporal model,
a selected equilibrium or a trajectory. The
[research program](../RESEARCH_PROGRAM.md#research-current-question)
remains the sole active queue.

## 1. Actual law and principal domain

Keep the same re-admitted retained THOL input and hypothetical default UM
reset, with 21 nodes, 53 edges, positive predicted conductances and capacities,
structural clock and zero forcing. The
[complete law and branch inventory](SINE_SUCCESSOR_STATIONARY_GEOMETRY.md#1-complete-law-source-and-every-edge-branch)
remain unchanged. Set

\[
 P=\{\theta: |\theta_j-\theta_i-2\pi k_{ij}|<\pi
                 \text{ for every oriented edge }(i,j)\}.
\]

The integers `k_ij` are precisely the eight one-turn branches and 45 zero
branches already admitted. P is an open convex chamber of real lifts,
modulo common translation. It is larger than the acute chamber; its admission
does not prove that the hypothetical source stays there.

Write `V=sum_U(1-cos delta)` and `K=Hess V=L_cos(delta)`. The actual
storage is `H=Q_W(x)+V(theta)`. At a stationary state, connected positive
form conductances imply common form, and the phase equations are `grad V=0`.
For the actual positive weights `rho_i=d_i/nu_i`, a constrained critical
point obeys `grad V=lambda*rho`; translation invariance forces `lambda=0`.
For any phase direction z, replacing it by
`z-1*(rho^T z)/(rho^T 1)` preserves `z^T K z` because `K*1=0`.
Thus full K being positive semidefinite is necessary for a local minimum
on the actual weighted-mean leaf. Arithmetic-mean centering is not substituted.

Positive semidefiniteness alone is not a minimum theorem. A negative edge
cosine alone is not a negative direction of the full Hessian. Both distinctions
are retained below and in the independent controls.

## 2. Fine twins, leaf and zero resultants

The exact bare support has adjacent twins `(0,1)` and nonadjacent twins

```text
(4,5), (6,7), (8,9), (10,11), (12,13),
(14,15), (16,17), (18,19).
```

Nodes 2 and 3 have different external neighbors and are not twins.
The previous contact theorem's graph-specific receiver identification
therefore does not apply.

At a stationary PSD candidate in P, each listed pair has equal lifted
phases. The [general twin argument](SINE_CONTACT_STATIONARY_GEOMETRY.md)
applies because these pairs have identical outward branches to every common
neighbor. For a nonzero common neighbor resultant, stationarity and a
nonnegative diagonal select alignment with it. If that resultant is zero,
PSD forces every entry of the corresponding zero-diagonal row to vanish.
Unequal twins would be antipodal; their same strict principal branches then
force every common neighbor to the same midpoint phase, contradicting a
zero resultant. Equal twins with zero resultant are not discarded at this
stage.

For the adjacent pair, a split `Theta+epsilon,Theta-epsilon` has restricted
potential `-2*C*cos epsilon-cos(2*epsilon)+constant`. A nonzero stationary
split inside the internal principal cut has second derivative
`-4*sin(epsilon)^2<0`, so PSD excludes it even when the external resultant
vanishes.

The child 20 is a leaf. Stationarity gives `sin(theta_20-theta_2)=0`;
its strict zero-branch principal gap forces equality. These restrictions
are consequences at candidate minima, not an invariant fine dynamical
pair reduction. Nonuniform capacities and conductances still govern motion.

## 3. Return chains and a singular case excluded by higher order

Use the common pair phases `O=A0`, `R=A2`, `B0`, `B1`, `B2`, with
`P_node=theta_2` and `Q_node=theta_3` distinct. Each three-block return
chain has two uncontacted interior pairs:

```text
A2 -> A3 -> A4 -> A0,
B2 -> B3 -> B4 -> B0.
```

Stationarity equates the three edge sines. The principal two-interior-node
Hessian restriction, apart from factor four, is

\[
 \begin{pmatrix}c_1+c_2&-c_2\\-c_2&c_2+c_3\end{pmatrix}\succeq0,
 \qquad |c_1|=|c_2|=|c_3|.
\]

Its diagonals and determinant exclude every nonzero cosine sign pattern
except all positive. If all cosines are zero, equal sines still force all
three principal gaps equal. The full marked branches and the two-edge paths
through the parent imply

\[
 t_A=\frac{2\pi+O-R}{3}\in(0,\pi/2],\qquad
 t_B=\frac{2\pi+B0-B2}{3}\in(0,\pi/2].
\]

The zero-cosine endpoint is PSD-compatible but cannot be a local minimum.
At `t=pi/2`, move the two common interior pair phases by u and v, keeping
endpoints and every other phase fixed. Only the three complete blocks change:

\[
 \Delta V=4[\sin u+\sin(v-u)-\sin v].
\]

With `v=2u`, this is `8*sin u*(1-cos u)`, which has both signs arbitrarily
near zero. Thus every actual local minimum satisfies **strictly**
`0<t_A,t_B<pi/2`. The argument retains the singular Hessian and uses a
cubic variation; it does not invent a negative quadratic direction there.

## 4. Six independent core coordinates and exact curvature bookkeeping

Fix O=0 and use six coordinates `(p,q,r,b,v,c)` for the parent, node 3,
A2, B0, B1 and B2. Put `t_A=(2*pi-r)/3` and
`t_B=(2*pi+b-c)/3`. Reconstruct the fine lifts as

```text
(0,0, p,q, r,r, r+t_A,r+t_A, r+2*t_A,r+2*t_A,
 b,b, v,v, c,c, c+t_B,c+t_B, c+2*t_B,c+2*t_B, p).
```

Check all original principal edge inequalities after reconstruction.
On this affine chain restriction the full bare potential is

\[
\begin{aligned}
V_6={}&2[(1-\cos p)+(1-\cos q)+(1-\cos(r-p))+(1-\cos(r-q))]\\
&+(1-\cos(q-p))+12(1-\cos t_A)\\
&+4[(1-\cos(v-b))+(1-\cos(c-v))+(1-\cos b)]\\
&+2[(1-\cos(b-p))+(1-\cos(v-p))+(1-\cos(c-p))]
 +12(1-\cos t_B).
\end{aligned}
\]

Each coefficient is the actual number of bare fine edges. The synchronized
internal A0 edge and child edge contribute zero; the other multiplicities
sum to 51. Actual conductances are not averaged into this phase potential.

Full fine stationarity on this restriction is equivalent to `grad V_6=0`:
the interior chain and child currents cancel identically, and the remaining
fine currents are recovered from the six derivatives and their zero sum.
At a stationary candidate with nonnegative chain cosines, the full fine
PSD test is equivalent to the six-core PSD test. To retain the singular case,
use an exact quadratic decomposition instead of dividing by a zero cosine.
For one chain of common cosine h and endpoint increments x0,x3, write
z1,z2 for interior deviations from linear interpolation. Its form is

\[
4h\sum_{j=0}^2(x_{j+1}-x_j)^2
=\frac{4h}{3}(x_3-x_0)^2+
4h[z_1^2+(z_2-z_1)^2+z_2^2].
\]

Add the positive child discrepancy square and the fine twin antisymmetric
blocks. Complete pair blocks remove cross terms. Each chain-interior split
has eigenvalue `4h`; endpoint splits equal half the corresponding ungrounded
core diagonal plus `4h/3`, with an additional 2 for the adjacent A0 pair.
The B1 split is half its core diagonal. These identities show that core PSD
implies nonnegative omitted blocks; restriction gives the converse. Common
phase translation can then be fixed without changing PSD. This equivalence
is a stationary curvature statement, not complete nonlinear dynamics closure.

## 5. Eliminate the two middle nodes without losing zero-resultant cases

Node 3 and either member of B1 have odd degree five. At stationary PSD,
a zero resultant would give a zero Hessian diagonal and hence a zero full
row. All five incident cosines would vanish, leaving five sine values
`+1` or `-1` whose sum cannot be zero. Thus their resultants are strictly
nonzero at any local minimum.

Now fix the parent's phase to zero and describe the two endpoint pairs by

\[
 O=m-a,\quad R=m+a,\quad B0=n-b,\quad B2=n+b.
\]

Define

\[
 Z_A=1+4\cos a\,e^{im},\quad
 Z_B=1+4\cos b\,e^{in},\quad R_A=|Z_A|>0,\quad R_B=|Z_B|>0.
\]

The aligned middle lifts are `Q_node=Arg Z_A` and `B1=Arg Z_B`, on the
unique branches admitted by the parent and endpoint principal gaps.
The necessary regular domain is

\[
 \pi/4<a,b<\pi,\quad |m|<\pi-a,\quad |n|<\pi-b,\quad
 |m-a-n+b|<\pi.
\]

Also require `|Arg Z_A|,|Arg Z_B|<pi`,
`|Arg Z_A-(m-a)|,|Arg Z_A-(m+a)|<pi`, and the two corresponding B endpoint
inequalities. Retain the original full fine branch check as the authoritative
admission. These conditions cannot be replaced by a midpoint angle or just
two winding numbers. A zero or unresolved resultant gives no regular chart.

The remaining fine coordinates follow from the equal return-chain gaps,
the twins and the child. A common translation converts this parent-zero
chart to the core's O-zero chart; it changes no edge branch. Neither chart
substitutes an arithmetic phase mean for the actual conserved mean.

## 6. Four-coordinate potential and what it decides

With `t_A=(2*pi-2*a)/3`, `t_B=(2*pi-2*b)/3`, the exactly restricted fine
potential is `V=51+F`, where

\[
\boxed{\begin{aligned}
F(a,m,b,n)={}&-4\cos a\cos m-R_A-12\cos t_A\\
&-4\cos b\cos n-2R_B-12\cos t_B\\
&-4\cos(m-a-n+b),
\end{aligned}}
\]

\[
 R_A=\sqrt{1+16\cos^2a+8\cos a\cos m},\qquad
 R_B=\sqrt{1+16\cos^2b+8\cos b\cos n}.
\]

The factors 1 and 2 on the resultants retain the unequal fine multiplicities.
They are a material difference from the previous contact model.

Every full principal-sector local minimum maps to a local minimum of F
inside this admitted domain. Conversely, a local minimum of F reconstructs
a full relative phase minimum. The reason is local elimination of positive
directions, not an assumed symmetry of every trajectory. The middle-node
curvatures are `R_A` and `2*R_B`; both chain interior forms are positive.
A minimum of F gives PSD reconstructed core curvature. The omitted twin
blocks are then strictly positive: chain endpoints inherit positive chain
terms, B1 inherits `R_B`, the adjacent A0 split has its extra 2, and the
child discrepancy has cosine one. Symmetry fixes the transverse minimum
at equal members. These positive directions admit unique local minimizing
continuations, leaving precisely F as the local-minimum question.

A certified critical point with positive-definite four-dimensional Hessian
would therefore give a strict relative storage minimum for the actual
positive-weight complete law. Common form is fixed by the actual form mean;
common phase is translated to the actual phase mean. This would establish
local conservative maintenance, not attraction or entry of the proposed
source. If the four-dimensional Hessian is singular, PSD remains only a
necessary condition and higher-order terms must be examined.

## 7. Executable interval calculations and their evidence boundary

The [candidate owner](../../src/tnfr/research/maintained_successor_principal.py)
rebuilds the source/reset/chamber primitives for every public call.
`evaluate_successor_core` returns the reconstructed fine state, full branch
margins, fine currents and Hessian, and their six-coordinate restriction.
`evaluate_successor_reduction` additionally admits the nonzero resultants,
reconstructs the aligned middle nodes and returns the full potential
`V=51+F`, with the gradient and Hessian shared by V and F.
It uses whole interval enclosures; unresolved cuts, domains or denominators
reject. It does not install edges, execute a word or declare a root from a
small residual.

For the four-variable derivatives, eliminate the two middle coordinates of
the core Hessian by their strictly positive diagonal block, then pull back
the four endpoint coordinates. The external core order is `(p,r,b,c)` and
its linear derivative with respect to `(a,m,b,n)` is

\[
 L=\begin{pmatrix}1&-1&0&0\\2&0&0&0\\1&-1&-1&1\\1&-1&1&1\end{pmatrix}.
\]

Thus `grad F=L^T*g_external` and
`Hess F=L^T*(K_external-K_cross*diag(1/R_A,1/(2*R_B))*K_cross^T)*L`.
The aligned middle gradients vanish identically; no approximate response
gradient is substituted into this identity.

The full classification follows from the analytical arguments in Sections
8-12, including singular cases and approaches to the open domain boundary.
The evaluator remains an observation API: failure of a solver, a finite set
of candidates or an interval box to certify a root would not prove absence.
The acute theorem is reused only after its actual hypotheses are established.

Verification uses [exact reduction controls](../../tests/research/test_maintained_successor_principal_reduction.py),
[candidate admission controls](../../tests/research/test_maintained_successor_principal.py)
and [independent fine-state comparisons](../../tests/research/test_maintained_successor_principal_evidence.py).
These include the singular chain's cubic variation and independent derivatives
of F. A synchronized minimum in a different zero-winding class prevents a
scope expansion to all graph states. No UM, trajectory or frozen producer
is replayed, and the retained evidence remains unchanged.

<a id="successor-positive-half-span-exclusion"></a>

## 8. Exclusion when both endpoint half-spans have positive cosine

The four-coordinate domain admits a useful additional restriction before
stationarity is considered:

\[
 \pi/4<a,b<\arccos(-1/4).
\]

To prove the upper bound for either motif, write its half-span and midpoint
as u,v, and its aligned middle phase as Q. The two strict middle-to-endpoint
branches imply `|Q-v|<pi-u`. If `u>=pi/2`, then `cos(Q-v)>0`. Multiplying
`Z=1+4*cos(u)*exp(i*v)` by `exp(-i*v)` gives

\[
 R\cos(Q-v)=\cos v+4\cos u>0.
\]

Consequently `cos u>-1/4`. For `u<pi/2` the bound already holds. This is a
branch and alignment restriction, not evidence that the whole domain is acute.

**Conditional exclusion.** There is no local minimum in the full admitted
principal chamber whose reduced coordinates satisfy
`pi/4<a,b<pi/2`. This excludes a region containing nonacute individual edges;
it does not assume those edges acute at the start of the proof.

Put `d=m-a-n+b`, `c_A=cos a` and `c_B=cos b`. The two midpoint stationarity
equations obtained from the actual potential are

\[
 c_A\sin m(1+1/R_A)+\sin d=0,\qquad
 c_B\sin n(1+2/R_B)-\sin d=0.
\]

All prefactors are positive in the sector under consideration, and the
principal branches give `|m|,|n|,|d|<pi`. Thus either `m=n=d=0` and `a=b`,
or the midpoint signs are opposite. More precisely,

\[
\begin{array}{ll}
a>b:&m>0, n<0, d<0,\quad |m|+|n|<a-b<\pi/4,\\
b>a:&m<0, n>0, d>0,\quad |m|+|n|<b-a<\pi/4.
\end{array}
\]

These conclusions use the sign of the admitted lifted d, not an arbitrary
wrapped representative. Also, each aligned middle argument lies strictly
between its nonzero midpoint and zero: its resultant is a positive combination
of the two rays at those angles, separated by less than pi. Each of the three
cases is excluded below.

### 8.1. Larger A half-span

Suppose `a>b`. Let `z=a-pi/4`, `t=t_A=pi/2-2*z/3`, `h=cos t`,
and `q=Q_node/2`. For fixed `r=4*cos a`, differentiation of the aligned
middle argument gives

\[
 \frac{d}{dm}\operatorname{Arg}(1+r e^{im})
 =\frac{r(r+\cos m)}{1+r^2+2r\cos m}
 \leq\frac{r}{r+\cos m}\leq\frac45.
\]

The first inequality follows from `cos(m)^2<=1`; the second uses
`r<2*sqrt(2)` and `cos m>1/sqrt(2)`. The argument is zero at m=0, hence
`0<q<2*z/5`. Since `0<2*z/3<pi/6`, concavity of sine above its chord
on `[0,pi/6]` gives `h=sin(2*z/3)>=2*z/pi`. It follows that

\[
 \sin^2q<\frac{4z^2}{25}
 \leq\frac{\pi^2 h^2}{25}<\frac{h^2}{2}
 \leq 1-\sin t.
\]

Write `y=a+m-q`. The fine A2 current equation, with its actual two edges
to the parent and two to node 3, is

\[
 \sin(a+m)+\sin(a+m-2q)=2\sin t,
 \qquad \sin y\cos q=\sin t.
\]

Here `0<y<pi`. If `y>=pi/2`, the diagonal for the A2 endpoint after
eliminating the positive return-chain interiors is

\[
 4\cos y\cos q+\frac43h
 =-4\sqrt{h^2-\sin^2q}+\frac43h<0,
\]

because the square root exceeds `h/sqrt(2)>h/3`. This contradicts necessary
PSD curvature. Therefore `y<pi/2`. The stronger inequality
`sin(q)^2<1-sin t` then gives `sin y<cos q`, and hence
`y+q=a+m<pi/2`. Both upper A gaps, `a+m` and `a+m-Q_node`, are acute.

Every other original edge is also acute: the lower A gaps are bounded by a;
the B endpoint and middle gaps are bounded by `b+|n|<a`; the middle-to-parent
gaps are smaller than the corresponding midpoint magnitudes; the cross gap
has `|d|<a-b`; the return chains are already strictly acute. The internal A0
and child gaps vanish. Thus this hypothetical stationary PSD candidate lies
in the full acute chamber, where the existing
[closed-acute obstruction](SINE_SUCCESSOR_STATIONARY_GEOMETRY.md)
excludes stationarity.

### 8.2. Larger B half-span

Suppose `b>a`. Positive cosine places each aligned middle phase strictly
between its midpoint and zero. In A, the two upper gaps therefore lie in
`(0,a)`: `|m|<b-a<a` and `m<Q_node<0`. Their current equation forces
`sin t_A<sin a`. Both angles lie in `(0,pi/2)`, so

\[
 a>t_A=(2\pi-2a)/3,\qquad a>2\pi/5.
\]

In B, `0<B1<n` and `n<b-a`. Its two upper gaps, `b+n` and `b+n-B1`,
lie in `(b,2*b-a)`, which is contained in `(a,pi-a)`. Each sine therefore
exceeds `sin a>sin(2*pi/5)>2/3`. But the B2 current equation, retaining
two parent edges and four middle edges, requires

\[
 \sin(b+n)+2\sin(b+n-B1)=2\sin t_B\leq2.
\]

Its left side is strictly greater than 2, a contradiction. This case does
not require a curvature test.

### 8.3. Equal half-spans and remaining scope

If `a=b`, midpoint stationarity gives `m=n=0`; positive resultants then
give both middle phases zero. The A2 and B2 equations respectively require
`2*sin a=2*sin t_A` and `3*sin a=2*sin t_A`. They are incompatible because
`sin a>0`.

The three cases exclude every minimum with both half-spans strictly below
`pi/2`. The boundaries and negative-cosine sectors require the separate
arguments below. The [static sector controls](../../tests/research/test_maintained_successor_principal_sectors.py)
check the fine current multiplicities, chain-eliminated curvature, argument
derivative bound and interval margins used here. These controls run neither
an event nor a trajectory, and finite probes are not the exclusion proof.

## 9. Wide endpoints, zero half-span cosines and the double-wide sector

Call a half-span wide when it is at least `pi/2`. By the branch bound in
Section 8 its aligned middle resultant has the form `1-r*exp(i*v)`, with
`0<=r<1`. Its real part is positive and its argument has the opposite sign
to v, or is zero when `r=0`. Thus a nonnegative midpoint makes both upper
endpoint gaps lie in `[pi/2,pi)`.

For A, write these two gaps as alpha,beta, and put `s=sin t_A`,
`h=cos t_A>0`. The endpoint equation and the chain-eliminated endpoint
diagonal are

\[
 \sin\alpha+\sin\beta=2s,\qquad
 D_A=2\cos\alpha+2\cos\beta+\frac43h.
\]

Here `1/2<s<1`. For `u=sin alpha`, `v=sin beta` in `[0,1]` with
`u+v=2s>1`, concavity of the square root on this line gives

\[
 |\cos\alpha|+|\cos\beta|
 =\sqrt{1-u^2}+\sqrt{1-v^2}
 \geq2\sqrt{s(1-s)}>\frac23h.
\]

The minimum occurs at either endpoint `u=1` or `v=1`. The final strict
inequality reduces to `(1-s)*(8*s-1)>0`. Since both cosines are nonpositive,
`D_A<0`, contradicting necessary PSD curvature. A wide A half-span therefore
requires `m<0` at a candidate minimum.

For B, the corresponding equations retain the different middle multiplicity:

\[
 \sin\alpha+2\sin\beta=2\sin t_B,\qquad
 D_B=2\cos\alpha+4\cos\beta+\frac43\cos t_B.
\]

The first equation gives `sin beta<=sin t_B`. Because
`beta in [pi/2,pi)` and `0<t_B<pi/2`, this implies
`cos beta<=-cos t_B`. Together with `cos alpha<=0`, it gives `D_B<0`.
Thus a wide B half-span requires `n<0`. These diagonals follow by eliminating
only the positive return-chain interiors while retaining the other endpoints;
they are necessary restrictions of full PSD, not independent edge tests.

If both half-spans are strictly wide, `cos a,cos b<0` and `m,n<0` make
both terms in

\[
 \cos a\sin m(1+1/R_A)+\cos b\sin n(1+2/R_B)=0
\]

strictly positive. Hence that sector is impossible. If exactly one half-span
equals `pi/2`, its angular equation forces `d=0`. A strictly acute other
half-span then has midpoint zero, and `d=0` makes the wide midpoint positive,
contradicting the endpoint restriction just proved. A strictly wide other
half-span has a strictly positive angular term, also impossible.

Finally, if `a=b=pi/2`, then `d=0` gives `m=n`. Both aligned middle phases
are zero. The two endpoint equations would require
`2*cos m=sqrt(3)` and `3*cos m=sqrt(3)` simultaneously. Thus the zero-cosine
boundaries also contain no candidate minimum. Only the two strict mixed
sectors remain after Sections 8 and 9.

## 10. Exclusion of the positive-A, negative-B mixed sector

Assume `pi/4<a<pi/2<b<acos(-1/4)`. Section 9 gives `n<0`; the angular
equations then give `m<0` and `d>0`. Write `m=-u`, `n=-v`, with `u,v>0`,
and write the A middle phase as `Q_node=-z_A`, where `0<z_A<u`. Its upper
endpoint equation is

\[
 \sin(a-u)+\sin(a-u+z_A)=2\sin t_A.
\]

If `u>=a`, the first sine is nonpositive and the second at most one,
contradicting `2*sin t_A>sqrt(3)>1`. Consequently `0<u<a<pi/2`, and both
upper angles are acute. Their average `gamma=a-u+z_A/2` satisfies
`sin gamma*cos(z_A/2)=sin t_A`, so `gamma>=t_A`. It follows that
`a>t_A` and `a>2*pi/5`.

### 10.1. A bounds the left contact and its current

Put `a=pi/2-epsilon`, so `0<epsilon<pi/10`, and
`D=a-t_A=pi/6-5*epsilon/3`. For `k=4*cos a=4*sin epsilon` and
`0<u<pi/2`,

\[
 \frac{d}{du}\operatorname{Arg}(1+k e^{iu})
 =\frac{k(k+\cos u)}{1+k^2+2k\cos u}\leq k.
\]

Indeed the denominator minus `k+cos u` is
`1+k^2-k+(2*k-1)*cos u>0`: use `cos u<=1` when `k<=1/2`, and
`cos u>=0` when `k>=1/2`. Integration gives `z_A<=k*u`; hence

\[
 u(1-2\sin\epsilon)\leq u-z_A/2\leq D,
 \qquad u\leq\frac{D}{1-2\epsilon}.
\]

The denominator is positive. The A left-endpoint magnitude `A_L=a+u`
satisfies `A_L<=2*pi/3`, because

\[
 (\pi/6+\epsilon)(1-2\epsilon)-D
 =\epsilon[(8-\pi)/3-2\epsilon]>0.
\]

The exact resultant identity `sin(u-z_A)=sin u/R_A` also gives the contact
current

\[
 T=\sin d=\cos a[\sin u+\sin(u-z_A)]
 \leq2\epsilon u\leq\frac{2\epsilon D}{1-2\epsilon}<\frac16.
\]

The last inequality is equivalent to
`20*epsilon^2-(2*pi+2)*epsilon+1>0`. Its leading coefficient is positive
and its discriminant `(2*pi+2)^2-80` is negative; `pi<10/3` already suffices.

Let `L=b+v` be the B left-endpoint magnitude. Its parent branch gives `L<pi`,
and `d=L-A_L<pi-a<3*pi/5`. Since `d>0` and `sin d<1/6`, the interval
`[pi/2,3*pi/5]` is impossible. Thus `0<d<pi/2`. The certified scalar bound

\[
 \sin(34/15-2\pi/3)>1/6,\qquad
 0<34/15-2\pi/3<\pi/2
\]

then gives `d<34/15-2*pi/3` and therefore **`L<34/15`**. The bound comes
from stationarity and the retained branches, without an observed trajectory.

### 10.2. The B endpoint cannot balance this contact

Write `z=B1>0`, `R=b-v`, `beta=R-z`, `r=-4*cos b>0`, and
`t=(2*pi-2*b)/3`. The original middle branches give

\[
 0<v+z<\pi-b<\pi/2,\qquad
 0<\beta<R<b<\arccos(-1/4)<7\pi/12.
\]

The exact middle and upper endpoint currents are

\[
 \sin z=r\sin(v+z),\qquad \sin R+2\sin\beta=2\sin t.
\]

Set `L_0=34/15` and split at `b_0=pi/2+1/8`.

For `pi/2<b<=b_0`, put `delta=b-pi/2`. Then `0<delta<=1/8`,
`r=4*sin delta<1` and `z<=arcsin(4*sin delta)`. As `R=2*b-L`,

\[
 R>\pi-L_0,
 \qquad\beta>\pi-L_0+2\delta-\arcsin(4\sin\delta).
\]

The function `2*delta-arcsin(4*sin delta)` decreases on `[0,1/8]`:
its derivative is `2-4*cos(delta)/sqrt(1-16*sin(delta)^2)<0`, using
`delta<=1/8<pi/3`. The certified inequality
`sin(21/40)>4*sin(1/8)` therefore gives `beta>pi-61/24`.
Since `R,beta<b_0<min(L_0,61/24)`, the symmetric sine bounds yield

\[
 \sin R+2\sin\beta
 \geq\sin(34/15)+2\sin(\pi-61/24)
 >\sqrt3\geq2\sin t.
\]

This contradicts the B endpoint equation throughout this interval.

For `b>=b_0`, the original middle equation can also be written

\[
 2\sin(R-z)=\sin z+2\sin(L+z).
\]

On `0<z<pi-L`, the right side is a strictly concave function of z:
both sine terms are positive. Its endpoint values are `2*sin L` and
`sin L`, so it exceeds `sin L`. Since `0<L-b<L_0-b<pi/2`,

\[
 \sin R+2\sin\beta>\sin R+\sin L
 =2\sin b\cos(L-b)>2\sin b\cos(L_0-b).
\]

Define `G(b)=sin b*cos(L_0-b)-sin((2*pi-2*b)/3)`. Its derivative is

\[
 G'(b)=\cos(2b-L_0)+\frac23\cos t>0,
\]

because `0<2*b-L_0<7*pi/6-L_0<pi/2` and `0<t<pi/2`. Its lower endpoint
satisfies the certified scalar inequality

\[
 G(b_0)=\cos(1/8)\sin(257/120)-\sin(\pi/3-1/12)>0.
\]

Thus the B current again exceeds `2*sin t`, a contradiction. Together the
two intervals exclude the entire positive-A, negative-B mixed sector.
All scalar inequalities above are enclosed with the shared rational outward
interval kernel in the static sector controls; rounded decimal estimates
are not their premises.

## 11. Exclusion of the negative-A, positive-B mixed sector

Assume `pi/2<a<acos(-1/4)` and `pi/4<b<pi/2`. Section 9 gives `m<0`.
Write `a=pi/2+e`, `m=-u`, `r=-4*cos a=4*sin e`, and denote the A middle
phase and resultant magnitude by Q and `R_A`. Its middle-to-endpoint branches
put the argument of `exp(i*u)-r` inside `(-(pi-a),pi-a)`, an acute interval.
Thus `cos u>r`, and the parent branch also gives `e+u<pi/2`. In particular,

\[
 0<r<1,\quad 0<e<\arcsin(1/4),\quad 0<Q<\pi/2,
 \qquad
 R_A^2=(\cos u-r)^2+\sin^2u
       =1+r^2-2r\cos u<1-r^2<1.
\]

The A angular equation gives `sin d=-T`, where
`T=cos a*sin m*(1+1/R_A)>0`; the B angular equation then gives `n<0`.
Since `sin Q=r*sin u/R_A`,

\[
 T=\frac{R_A+1}{4}\sin Q<\sin Q.
\]

Set `delta=arcsin T`, so `0<delta<Q<pi/2`. The negative principal contact
gap is either `d=-delta` or `d=-pi+delta`. The second case is impossible:
the middle branch `Q-O<pi` gives `O>Q-pi`, hence
`B0=O-d>Q-delta>0`, whereas `B0=n-b<0`. Therefore
`d=-delta` and `B0=O+delta`.

In fact `B0<-pi/2`. Since `R_A>=sin u`,
`T<=sin e*(1+sin u)`, while

\[
 \sin(e+u)-\sin e(1+\sin u)
 =\sin u[\cos e-\sin e-\sin e\tan(u/2)]>0.
\]

Here `cos e>sqrt(15)/4`, `sin e<1/4`, and `tan(u/2)<1`. Since
`e+u<pi/2`, this proves `delta<e+u` and the claimed B0 bound.

### 11.1. Upper current and lower cosine restrictions

Put `q=Q/2`, `w=a-u-q`, `t=t_A`, `s=sin t`, and `h=cos t`. The upper
A endpoint equation is

\[
 \sin(a-u)+\sin(a-u-Q)=2s,\qquad \sin w\cos q=s.
\]

Necessarily `w<pi/2`. Otherwise `u+q<=e`, so `q<e` and
`sin w>=sin a=cos e`; their product exceeds `cos(e)^2>15/16`, whereas
`s<=sqrt(3)/2<15/16`. The principal branches and the row give `w>0`.
Also `sin(w-q)=2s-sin(a-u)>0`, because `t>pi/6`. With `w<pi/2`,

\[
 \sin(w-q)=s-\cos w\sin q<s,
 \qquad 0<w-q<t.
\]

Define `X=-cos O>0` and `Y=-cos(O-Q)>0`. The exact projection of the
middle resultant onto its aligned phase is

\[
 R_A=\cos Q+2\cos(O-Q)+2\cos(a-u-Q)
     =\cos Q-2Y+2\cos(w-q).
\]

The parent contribution is `cos Q`, not one. Moreover,
`cos Q-R_A=r*(cos u-r)/R_A>0`; consequently

\[
 Y>\cos(w-q)>h\geq1/2.
\]

### 11.2. A full fine direction moving both lower endpoints

Move O and B0 by the same infinitesimal amount, hold the parent, A2 and
B2 fixed, minimize each middle-node quadratic fiber, and use linear
interpolation along each return chain. The cross-contact variation vanishes
exactly. The resulting full fine-phase second variation is `D_A+D_B`, with

\[
 D_A=-2(X+Y)+\frac43h-\frac{4Y^2}{R_A},
\]

\[
 D_B=2\cos B0+4z+\frac43\cos t_B-\frac{8z^2}{R_B},
 \qquad z=\cos(B0-B1).
\]

These are the actual Schur factors of the middle curvatures `R_A` and
`2*R_B`. This direction gives the same quadratic form after recentering to
the actual weighted phase mean. The fine twin and child coordinates remain
admitted; no collective temporal law is assumed.

Because `B0<-pi/2`, its first cosine is negative. Completing the square gives
`4*z-8*z^2/R_B<=R_B/2`, and the positive-B resultant satisfies
`R_B<=1+4*cos b`. Therefore

\[
 D_B<\frac12+2\cos b+\frac43\cos t_B
 \leq\frac12+\sqrt2<2.
\]

The middle expression decreases on `pi/4<b<pi/2`, since its derivative
is `-2*sin b+(8/9)*sin t_B<0`. It remains to prove `D_A<-2`.

For `0<e<=1/5`, put `C=cos(q)^2`. The resultant geometry gives
`sin Q<=r`, hence

\[
 C\geq\frac{1+\sqrt{1-r^2}}2\geq1-\frac{16}{3}e^2.
\]

The second inequality follows from `r^2<=16/25<3/4` and
`sqrt(1-x)>=1-(2/3)*x` on `[0,3/4]`. Concavity of sine gives
`s<=sqrt(3)/2-e/3`. Thus

\[
 C-s^2\geq\frac14+\frac{e}{\sqrt3}-\frac{49}{9}e^2
 >(1/2-e)^2,
\]

where the strict comparison uses
`1+1/sqrt(3)>3/2>58/45>=(58/9)*e`. The exact lower cosine sum is

\[
 X+Y=2[\sin(2e)s+\cos(2e)\sqrt{C-s^2}].
\]

Elementary Taylor bounds give

\[
 \sin(2e)\geq(146/75)e,\qquad
 s\geq\sin(\pi/3-2/15)>137/175.
\]

For the latter, use `sqrt(3)/2>6/7`, `cos x>=1-x^2/2` and
`sin x<=x`: `(6/7)*(1-2/225)-1/15=137/175`. Their product exceeds
`(3/2)*e`, because `20002/13125>3/2`. Together with
`cos(2e)>=1-2*e^2`, this gives

\[
 X+Y>1+e-2e^2+4e^3>1.
\]

As `R_A<1` and `Y>h>=1/2`, it follows that
`D_A<-2+(4/3)*h-4*h^2<=-7/3<-2`.

For `e>=1/5`, the Taylor lower bound
`r>=4*sin(1/5)>=4*(1/5-(1/5)^3/6)>3/4` gives
`R_A<sqrt(7)/4<2/3`. Since `e>1/6`,

\[
 h>\cos(\pi/3-1/9)>
 \frac{161}{324}+\frac56\frac{485}{4374}
 =\frac{7733}{13122}>\frac7{12}.
\]

Here `cos(1/9)>=161/162`, `sin(1/9)>=485/4374` and `sqrt(3)>5/3`
justify every bound. Using just `X>0` and `Y>h`,

\[
 D_A<-\frac23h-6h^2<-175/72<-2.
\]

Both ranges give `D_A+D_B<0`, a contradiction to full fine PSD. This
excludes the entire negative-A, positive-B mixed sector.

<a id="successor-principal-minimum-exclusion"></a>

## 12. Full principal-chamber minimum exclusion and its scope

For the actual selected 21-node, 53-edge UM support, **the phase potential
has no local minimum anywhere in its retained open principal chamber**,
modulo common phase translation. Consequently the complete positive-weight
storage `H=Q_W+V` has no stationary local minimum on its actual fixed form
and phase mean leaf within that chamber.

The proof covers every possible minimum: twin and leaf restrictions retain
the singular resultant cases; the zero-cosine return-chain endpoint fails
by a cubic variation; every remaining minimum belongs to the regular
four-coordinate reduction. Sections 8-11 exclude its two positive cosines,
zero-cosine boundaries, two negative cosines and both mixed sectors. No
compact truncation, finite candidate list or failed root search replaces
coverage near the open domain boundaries.

This does not exclude stationary points, prove linear instability of every
stationary point, or exclude nonstationary invariant sets. In particular,
the zero-chain cubic argument does not supply a negative Hessian direction.
The synchronized minimum in a different branch class remains valid. The
result concerns the declared law and selected support; it neither executes
UM nor predicts that the retained hypothetical endpoint crosses a branch
cut, and it makes no claim of physical identification or operator selection.

<a id="successor-principal-shell-access"></a>

## 13. This source's storage and means do not separate it from principal cuts

Fix the exact ideal hypothetical endpoint, or one admitted endpoint in its
conditional numerical envelope. Keep that endpoint's own positive W and
capacities, `rho_i=d_i/nu_i`, forms `x_0`, lifted phases `theta_0`, and
weighted means. The existing
[source-specific storage admission](SINE_SUCCESSOR_STATIONARY_GEOMETRY.md#5-storage-and-both-means-do-not-trap-this-source-in-the-acute-chamber)
gives

\[
 Q_0=Q_W(x_0)>0.0105,\qquad V_0=V(\theta_0),\qquad h=Q_0+V_0.
\]

This independently justifies division by `Q_0`. It is a conditional source
bound, not a new certificate of an unevaluated runtime reset error.

Let Omega be the open principal chamber intersected with this source's exact
phase-mean leaf. Its closed chamber is compact: a spanning-tree path in the
connected finite support bounds each lift difference by sums of the finite
edge bounds `pi+2*pi*|k_e|`; the positive weighted mean then bounds every
centered coordinate. Closed branch and mean constraints make the bounded set
closed. Interpolation from the strict source shows that this closed chamber
is exactly the closure of Omega.

Set `E_*=V_0+Q_0/2<h`, and let D be the component containing `theta_0` of
`{theta in Omega: V(theta)<E_*}`. Its closure is compact. If that closure
avoided every principal boundary face, V would attain a minimum there with
value at most `V_0<E_*`. A small connected ball in the strict sublevel around
this point intersects D and therefore belongs to D. The point would be an
interior local minimum, contradicting Section 12. Consequently there is
`theta_b` in both the closure of D and the boundary of Omega, with
`V(theta_b)<=E_*`.

Choose a small convex neighborhood of `theta_b` on which
`V<(E_*+h)/2=h-Q_0/4`, and choose y in D inside it. Open connected subsets
of the finite-dimensional mean leaf are polygonally connected: join
`theta_0` to y inside D, then append the straight segment to `theta_b`.
The entire phase path retains the source mean and satisfies `h-V>Q_0/4`;
all original principal gaps remain strict before its final endpoint.

Writing `m_rho=(rho^T*x_0)/(rho^T*1)`, lift this path to full state by

\[
 x(s)=m_\rho 1+
 \sqrt{\frac{h-V(\theta(s))}{Q_0}}\,(x_0-m_\rho 1).
\]

The multiplier starts at one and remains greater than one half. Since
`L_W*1=0`, the path retains the source form mean, phase mean and exactly
`H=h`. At least one original gap reaches `+pi` or `-pi` at `theta_b`.
The final segment crosses every reached face transversely upon a short
extension. Continuity and the positive `Q_0/4` margin permit the same radial
form lift beyond that boundary.

Thus the component of this source's storage-and-means shell reaches and
crosses a principal cut. Those three conserved values alone cannot isolate
the source from every boundary of its full branch identity. The construction
applies separately to each admitted endpoint, without transferring another
endpoint's weights, means or energy. It reuses the existing source storage
and radial-lift controls, and changes no source, support, law or clock.

The parameter s is not structural time. This does not select a cut, crossing
time or trajectory, prove loss of either old constituent winding, or exclude
nonstationary confinement. A principal cut is an identity boundary rather
than a singularity of the smooth sine law. The separate
[complete-state certificate](SINE_SUCCESSOR_NONSTATIONARY.md#successor-nonstationary-result)
retains all original branches through a fixed half-unit window and proves a
relative-velocity reversal. That finite result does not extend this stationary
classification into an all-time confinement or escape theorem.
