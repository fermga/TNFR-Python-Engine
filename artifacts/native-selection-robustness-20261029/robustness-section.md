

## An explicit full-state robustness radius

<a id="native-selection-robustness"></a>

**Conditional quantitative theorem.** For the exact matched preparation in
[the fixed witness](#native-selection-witness), assume the finite retained-data
premises in RB1 and RB7 pass their primitive interval audit. Then every shared
initial state in the mean-one capacity leaf satisfying

\[
 \|s^0-s^0_{\mathrm{witness}}\|_\infty<\rho,
 \qquad \rho=2^{-2^{250}}>0,
\]

has the same paired destinations: positive aligned twist under held capacity,
and phase consensus under evolving capacity. All thirty coordinates enter
the norm, subject only to the one capacity-mean constraint. Initial phase
perturbations use the declared local lifts. The common form and phase origins
remain free, so this is a ball in the full 29-dimensional leaf rather than
in the copied/reflected construction.

The bound is a symbolic positive real of extraordinary smallness. It makes
the preparation tolerance explicit but supplies no practical preparation
precision, floating-point tolerance, basin volume or physical measurement
bridge. Its denominator must not be materialized. The proof combines an
explicit recovery domain, an analytic entry-time upper bound and a complete
field perturbation bound. No extra trajectory or numerical differentiation
is required. The static evidence, independently reviewed proof and source
freeze belong to `artifacts/native-selection-robustness-20261029/`; the
research program owns the evaluated status.

### RB1. Data premises and notation

For each retained model, the saved whole endpoint is in its protected
reflected rectangle and has E <= 7-eta, where eta=2^-30. The held endpoint
uses R=(2\*pi/3,pi) x (0,pi/2); the dynamic endpoint uses
S=(-2\*pi/3,2\*pi/3) x (-pi/2,pi/2). These are entry points, not equilibria.
The exact endpoint association and the inequality with this dyadic eta must
be audited against the retained source and interval evidence.

The full capacity hull is [7/8,33/32] for the reference, under either the
held row or its admitted heat row. It is enough below to use
1/2 <= nu_i <= 2. In the copied/reflected reference put

```text
p=(A,B_0)^T, z=(a,b)^T,
K=[[3,-1],[-1,2]], D=diag(3,2), N=diag(s,t), H=diag(H_0,H_4),
```

```text
V(z)=10-2*cos(2*a)-4*cos(a-b)-4*cos(b),
Gamma=grad(V), P=||p||_2, G=||Gamma||_2,
E=2*p^T*K*p+V.
```

The full form and phase shapes on either ring are
(A,-A,-B_0,0,B_0) and (a,-a,-b,0,b). Their means are zero. The protected
capture theorem keeps the reference in its rectangle, with E<=7-eta, for
all future time. Capacity transport preserves the stated positive hull.

The exact active equations are

```text
p'=-e*N*D^-1*K*p-(w/4)*N*H^-1*Gamma,
z'=(w/beta)*N*H^-1*K*p.
```

Indeed Gamma=-4\*H\*g in these coordinates. No capacity time derivative is
omitted: it enters through N, while the storage and cross term below have
no capacity coordinate.

### RB2. Uniform quantitative geometry of the reflected sublevel

Set delta=eta/8=2^-33. On every face of either selected rectangle, V>=7.
The two partial derivatives of V have magnitude at most8 everywhere.
Comparison to the closest point on each coordinate face therefore gives
an at least delta distance from every face for every point with V<=7-eta.
This conclusion uses the coordinate face values already proved in the
protected-capture owner, rather than a distance inferred from a simulation.

Let C_i=Re(sum_j exp(i\*(theta_j-theta_i))). The exact identities/inequalities

```text
V-(8-4*C_0)=6+2*cos(2*a)-4*cos(b)>=0,
V-(8-4*C_4)=2-2*cos(2*a)>=0
```

give C_0,C_4 >= (1+eta)/4. The central resultant is
C_3=2\*cos(b)>=2\*sin(delta)>=4\*delta/pi>delta.
All other rows are copies or conjugates of these. Thus the reference stays
in the positive-real chamber with every C_i>=delta.

For any resultant in that chamber, with alpha=Arg(resultant),
H=pi\*C\*tan(alpha)/alpha>=pi\*C; the quotient has its continuous value1
at alpha=0. Also H<=pi\*|resultant|<=pi\*d_i<12. Consequently all actual
metrics are positive; for a convenient uniform bound one may use

```text
4*delta <= H_i <12.
```

For active H_0,H_4 the stronger lower bound pi/4 holds, but is not needed
for the deliberately conservative estimates below. Finally, eigenvalues
of K lie between1 and4, V>=0, and E<7, so P<2. Direct differentiation gives
||Hess(V)||_2<=16 and G<=8\*sqrt(2)<12.

### RB3. A strict storage estimate without a rate fit

Choose epsilon_L=delta/256 and define

```text
L=E+epsilon_L*Gamma^T*p.
```

Differentiating the actual complete equations gives exactly

```text
L'=-4*e*(K*p)^T*N*D^-1*(K*p)
   +epsilon_L*[ (w/beta)*p^T*Hess(V)*N*H^-1*K*p
             -e*Gamma^T*N*D^-1*K*p
             -(w/4)*Gamma^T*N*H^-1*Gamma ].
```

There is no nu' term because L depends only on the native form and phase.
The same identity therefore holds for held and evolving capacity. Using
1/2<=nu<=2, 4\*delta<=H<12, e=w=1/2, beta=1, and the matrix bounds above,

```text
the first term is <=-P^2/3,
the three bracketed terms together are
<= (16/delta)*P^2 + 2*P*G - G^2/192.
```

The elementary inequality
2\*P\*G <= 384\*P^2+G^2/384
and delta<=1/32 imply

```text
(16/delta)*P^2+2*P*G-G^2/192
<= (32/delta)*P^2-G^2/384.
```

Hence

```text
L' <= -P^2/8 - (delta/2^17)*G^2
   <= -2^-50*(P^2+G^2).
```

Here delta=2^-33 and 256\*384<2^17. Also
|epsilon_L\*Gamma^T\*p|<24\*epsilon_L<1, while 0<=E<7. Thus throughout the
protected reference sublevel,

```text
-1<L<8.
```

This is a uniform strict estimate for the retained constitutive laws;
it is neither a fitted exponential rate nor a local saddle eigenvalue
used as a global bound.

### RB4. Explicit control of phase distance by its gradient

Put u=a/2-b. The active interior resultant has
C_4=2\*cos(a/2)\*cos(u)>1/4. In either rectangle |a|<pi and |u|<pi,
so cos(a/2)>1/8, cos(u)>1/8 and |u|<pi/2. Since
Gamma_b=-8\*cos(a/2)\*sin(u), it follows that

```text
|u|<=2*G.
```

For F(a)=4\*(sin(2\*a)+sin(a/2))=8\*sin(5\*a/4)\*cos(3\*a/4),
Gamma_a=F(a)+4\*(sin(a/2+u)-sin(a/2)), hence |F(a)|<=9\*G.

For R, let a_\*=4\*pi/5. On its at least delta interior,
|cos(3\*a/4)|>=3\*delta/8 and
|sin(5\*a/4)|>=5\*|a-a_\*|/8.
These follow from sine concavity on the relevant intervals of length at
most pi/2 and pi<4. Therefore

```text
|F(a)|>=delta*|a-a_*|,
|a-a_*|<=9*G/delta,
|b-2*pi/5|<=7*G/delta.
```

For S, |a|<=2\*pi/3-delta gives cos(3\*a/4)>=3\*delta/8. Concavity on
[0,5\*pi/6] gives |sin(5\*a/4)|>=|a|/8. Thus

```text
|F(a)|>=(delta/4)*|a|,
|a|<=36*G/delta,
|b|<=20*G/delta.
```

In either case the complete ten-node lifted phase deviation from its
selected equilibrium has norm at most112\*G/delta. The complete form norm
is2\*P. Let Z denote their combined Euclidean norm after quotienting only
for purposes of distance to the two neutral common-origin directions.
The actual full state is not reduced by this observation. The reference
itself has zero common origins, so explicitly

```text
Z^2=4*P^2+4*((a-a_*)^2+(b-b_*)^2),
||(theta-theta_*)||_2<=112*G/delta.
```

### RB5. A fixed finite entry bound

Consider the set

```text
P<=2^-15,  G<=delta*2^-21=2^-54.
```

There the full form norm is at most2^-14 and the full phase norm is
strictly less than2^-14, because112<128. Consequently Z<2^-13.
Outside this set, P^2+G^2>2^-108. The strict storage estimate gives
L'<-2^-158 there. A reference that remained outside for a whole interval
of length2^162 would lose more than16 units of L, contradicting -1<L<8.
Every captured reference therefore enters Z<2^-13 at some time

```text
0<=tau<2^162 after its saved capture time.
```

Since both saved capture times are below32, the absolute entry times are
strictly below2^163. The times may differ between the two laws. No point
of a trajectory at that enormous time has been computed or asserted as
stored evidence; the number is an analytic upper bound on an entrance.

### RB6. An explicit complete-state recovery guard

This section records the sufficient local guard supplied by the [native local recovery theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery) and the separate control review. It avoids a
numerical27-dimensional Lyapunov solve. Let P_0 be the orthogonal projection
onto the complement of the common vector and write
Here the complete graph potential is `V(theta)=sum_edges[1-cos(theta_j-theta_i)]`;
it reduces exactly to RB1 on the copied/reflected reference.

```text
Z^2=||P_0*x||_2^2+||P_0*(theta-theta_*)||_2^2,
E_rel=(1/2)*x^T*B*x+V(theta)-V(theta_*).
```

The reference angles are consensus or the aligned winding-one twist.
The complete perturbation ball retains both neutral common origins;
the projection here only defines distance to the equilibrium family.

For any connected unit graph on10 nodes, a path of length at most9 gives
10\*||P_0\*v||^2=sum_{i<j}(v_i-v_j)^2<=405\*v^T\*B\*v. Thus lambda_2(B)>=2/81.
On the actual support lambda_max(B)<=2\*d_max=6. Choose r=1/16. At either
selected equilibrium every principal target edge gap has absolute value at most2\*pi/5.
Using sqrt(2)\*r<1/8, pi<22/7 and cos(t)>=1-t^2/2,

```text
c_r >= cos(2*pi/5+sqrt(2)*r)
    > 1-(387/280)^2/2 =7031/156800 >1/32.
```

Thus the existing local energy coefficient satisfies k_r>=1/2592 and
k_r\*r^2>=1/663552. A sufficient complete-state guard is therefore

```text
Z<1/16,  E_rel<1/663552,
```

with strictly positive capacity and the matching actual complete law.
The reference entrance Z<2^-13 satisfies this guard with a substantial
strict margin, since the graph Hessian upper bound gives E_rel<=3\*Z^2.

For evolving capacity, the same guard remains valid: E_rel has no capacity
term and its derivative is the same negative native loss. The native
sublevel cannot leave the local ball; combine that invariant set with the
compact positive capacity hull, in the quotient by the two common origins. Apply LaSalle to E_rel plus the actual
capacity variance solely to identify its invariant zero-loss set. Do not
replace the guard by an unnecessary bound on the sum. On the mean-one leaf,
zero variance loss forces nu=1; zero native loss then forces Bx=0 and
ultimately g=0. Strict acute convexity gives the target phase shape. The
joint local Hurwitz argument already proved in DA supplies the eventual
exponential tail and finite common-origin limits. For held capacity the
existing positive-profile recovery argument applies directly.

### RB7. Full-state perturbation transport

Use all thirty coordinates `(x,theta,nu)` with the mean-one capacity
constraint and the declared continuous phase lifts. The ball norm is the
coordinate sup norm. Neither common origin is quotiented out of this ball.
Put `rho=2^(-2^250)` and the bootstrap tube thickness `a0=2^-37`.
The expression for rho denotes an exact positive real; no floating
materialization or construction of its enormous denominator is required.

For each node define C_i=sum_j cos(theta_j-theta_i),
S_i=sum_j sin(theta_j-theta_i), q_i=(Bx)_i, and

\[
 g_i=\pi^{-1}\arctan(S_i/C_i),\qquad
 h_i=H_i^{-1}=\pi^{-1}\int_0^1
       \frac{C_i}{C_i^2+t^2S_i^2}\,dt.
\]

On C_i>=m>0 the following pointwise derivative bounds hold:

\[
 |g_C|,|g_S|\le(\pi m)^{-1},\quad
 0<h\le(\pi m)^{-1},\quad
 |h_C|,|h_S|\le(\pi m^2)^{-1}.
\]

For h_C use |t²S²-C²|<=(C²+t²S²); for h_S set
v=tS/C and use 2|v|/(1+v²)²<=1. The integral is smooth
at S=0, so the bound also covers the removable quotient there.
Each phase derivative row of C and S has absolute row sum at most 2d_i.
Consequently ||D_theta g_i||_1<=4d_i/(pi*m) and
||D_theta h_i||_1<=4d_i/(pi*m²).

Both complete fields have

\[
 F_{x_i}=\nu_i[-q_i/(2d_i)+g_i/2],\quad
 F_{\theta_i}=\nu_i h_i q_i/2,
\]

with F_nu=0 (held) or -B*nu/2 (dynamic). In a box with
nu_i<=N and |x_i|<=X, since ||B_i||_1=2d_i and |g_i|<1/2,
complete Jacobian absolute row sums are bounded by

\[
 L_x\le N+\frac{2Nd}{\pi m}+X+\frac14,
\quad
 L_\theta\le\frac{Nd}{\pi m}
       +\frac{4Nd^2X}{\pi m^2}+\frac{dX}{\pi m},
\quad L_\nu\le d.
\]

The last native term in each row includes derivatives with respect to
capacity, even in the held model. Treating held capacity as an unperturbed
parameter would omit nine allowed initial directions.

The retained prefix must separately pass its declared static primitive-box
audit: |q|,|r|<1 and each full relative resultant real part >=2^-33.
The embedding gives |A|<3/5, |B_0|<4/5 there. The later protected capture
has C_i>=delta=2^-33, |A|,|B_0|<2 and capacities in [7/8,33/32]
by the quantitative capture estimates and the capacity hull theorem.
Thus the complete nominal path up to its recovery-entry time has C_i>=2^-33
and |x_i|<2. An a0-thick full-state tube has

\[
 C_i\ge2^{-33}-6a_0>2^{-34}=m,\qquad
 1/2<\nu_i<2,\qquad |x_i|<8.
\]

The first inequality uses |exp(i alpha)-exp(i beta)|<=|alpha-beta|
for every incident edge; it also holds on straight segments joining two
states in the inflated box. With N=2,d<=3,X=8,m=2^-34,
L_x<2^39, L_theta<2^79 and L_nu<=3. In particular the common
full-state sup-norm Lipschitz bound L=2^82 is conservative for both laws.
It does not use an equilibrium Jacobian or a reduced numerical tangent.

Let tau_H,tau_D be the entry times supplied by the quantitative reduced
argument, including the corresponding retained-prefix lengths. Their
explicit upper bound is T=2^163, although their exact values need not be
computed. Before a first exit from the a0-thick tube, the complete-state
Gronwall bound gives

\[
 \|s(t)-s_{ref}(t)\|_\infty
 \le\rho e^{Lt}
 \le2^{-2^{250}+2^{246}}<2^{-37}=a_0,
 \qquad 0\le t\le\tau_m.
\]

Here L*T=2^245 and e<4. The strict inequality rules out the first exit;
positive capacity, the regular chart and bounded states supply continuation
through tau_m. This argument includes both common-origin directions and
the nine independent capacity perturbations on the mean-one leaf. Use the
same perturbed full initial state in both models.

At tau_m the nominal native quotient norm is <2^-13. The perturbed
native quotient differs by at most sqrt(20)*a0<2^-34, since orthogonal
projection cannot enlarge its Euclidean norm. Hence its norm is <2^-12,
and Taylor's potential bound with ||B||_2<=6 gives

\[
 E_{rel}<3\,2^{-24}<1/663552,\qquad
 \|z\|<2^{-12}<1/16.
\]

The independently established complete recovery guard now applies to each
perturbed state: held capacity retains its own perturbed positive profile;
dynamic capacity evolves that same initial profile toward one. Both converge
to their respective target orbit, with finite common-origin limits. This
proves one explicit positive full-state ball. Its extraordinarily conservative
radius supplies no usable tolerance for laboratory or binary64 preparation,
no basin volume, and no new physical identification.
