### Full-state perturbation transport without a 30-dimensional numerical solver

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
