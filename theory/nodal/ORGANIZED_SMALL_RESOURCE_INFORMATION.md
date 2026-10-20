# Information from a small phase preparation

<a id="organized-small-resource-information"></a>

The two admitted constitutive laws can produce different finite observations
without an initial mean-form contrast or a complete relative phase turn.
For the two-C9 quarter-factor model, a supplied small relative phase \(d\)
gives an endpoint mean-form separation proportional to \(\gamma d^3\).
This owner bounds that separation from both sides and retains independent
preparation and observation errors for the two candidate laws.

The question is a resource-information tradeoff for one specified observation,
not the minimum resource for every possible experiment. The earlier
[organized nonlinear comparison](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md)
already establishes finite constitutive discrimination on its different
complete model. The new obligation is to relate a small declared phase
resource to both distinguishable and indistinguishable endpoint records.
Task status belongs to the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

## 1. Complete law and the endpoint observation

Keep the ordered eighteen nodes \(9c+k\), with \(c=0,1\) and
\(k=0,\ldots,8\). Each marked child is a unit-weight C9. The nine
matching edges have the fixed common weight
\(a=\gamma^2/1000\), where \(\gamma=1/(1023\pi)\).
Capacity is held at one; no input, impulse, capacity change or later
support event is admitted. The state-preserving initial attachment and
its work are supplied separately from the continuing flow.

With \(L=I_2\otimes L_9+aL_{\rm match}\), the complete structural
law in clock \(\tau\) is

\[
\begin{split}
 V_\lambda(u)&=1-\cos u+
              \lambda\frac{(1-\cos9u)^2}{6\,9^4},\\
 j_\lambda(u)&=\sin u+\lambda h(u),\qquad
 h(u)=\frac{\sin9u(1-\cos9u)}{2187},\\
 U_\lambda(\theta)&=\sum_e w_eV_\lambda(\theta_i-\theta_j),\\
 x_\tau&=\frac{-Lx-\gamma\nabla U_\lambda}4,
 \qquad \theta_\tau=\frac{\gamma Lx}4,
 \qquad 0\le\lambda\le1.
\end{split}
\tag{1}
\]

Both rows use the same weights and the unchanged canonical \(\gamma\).
The pressure is specified by the form row, independently of the response.
This is the retained comparison law, not a native operator word or a
reciprocal-degree normalization. Smooth bounded currents and the linear
form terms give global continuation in continuous phase lifts.
The uniform current bounds, derived from (1), are

\[
 |j_\lambda|\le J_0=\frac{1459}{1458},\qquad
 |j_\lambda'|\le J_1=\frac{245}{243},\qquad
 |j_\lambda''|\le J_2=\frac{10}{9}.
\tag{2}
\]

Write \(\alpha=2\pi/9\) and retain the marked closing-edge turn.
The actual state decomposes as

\[
 x_{c,k}=X_c+z_{c,k},\qquad
 \theta_{c,k}=\alpha(k-4)+\Phi_c+q_{c,k},\qquad
 \sum_k z_{c,k}=\sum_k q_{c,k}=0.
\tag{3}
\]

Thus \(r=X_A-X_B\) is a raw mean-form difference,
\(v=\gamma r\), and \(\delta=\Phi_A-\Phi_B\) is a relative
continuous phase lift. The time coordinate \(s=a\tau/2\) transforms
both rows; the nominal readout \(s=1\) corresponds to
\(\tau=2000/\gamma^2\), not to a claimed laboratory duration.
Internal edge sums cancel exactly, giving

\[
 v_s=-v-\gamma^2\overline J_\lambda,
 \qquad \delta_s=v,
 \qquad \overline J_\lambda=
       \frac19\sum_k j_\lambda(\delta+q_{A,k}-q_{B,k}).
\tag{4}
\]

The nominal preparation has \(x=0\), \(q=z=0\),
\(\Phi_A=d/2\), \(\Phi_B=-d/2\), with

\[
 0<d\le\frac1{100}.
\tag{5}
\]

The carrier \(q=z=0\) is invariant: internal currents cancel and each
matching current is constant over its child. On it,
\(\overline J_\lambda=j_\lambda(\delta)\), \(v(0)=0\), and
\(\delta(0)=d\). The observation is one endpoint raw contrast
\(R_\lambda(d)=r_\lambda(1)\). It is neither a derivative estimate
nor the harmonic integral used by the
[full-turn probe](ORGANIZED_FREE_PHASE_PROBE.md).
Every nominal child retains its winding-one phase profile exactly.

This preparation supplies a relative phase. It does not derive the
formation of either C9, autonomous attachment, an observation apparatus
or a source of the required work. The
[NFR ontology](../EMERGENT_ONTOLOGY.md) distinguishes this conditional
structural observation from physical identification.

## 2. A two-sided nominal information bound

Let \(G=\gamma^2\). The canonical parameter satisfies \(G<10^{-7}\),
using \(\pi>157/50\). On \(0<u\le d\), both \(\sin u\) and
\(h(u)\) are positive, while \(j_\lambda(u)\le J_1u\).
Variation of constants in (4), until a possible first exit from
\(0<\delta_\lambda\le d\), gives

\[
\begin{split}
 -GJ_1d(1-e^{-s})&\le v_\lambda(s)\le0,\\
 d-GJ_1d(s-1+e^{-s})&\le\delta_\lambda(s)\le d,\\
 \delta_\lambda(s)&\ge d(1-GJ_1s^2/2)>\frac{99}{100}d
          \qquad(0\le s\le3/2).
\end{split}
\tag{6}
\]

Here \(s-1+e^{-s}\le s^2/2\). The positive lower bound excludes
exit through zero, and \(v_\lambda\le0\) excludes exit through the
upper endpoint. This closes the corridor for every admitted \(\lambda\).

The perturbing current has a cubic bound obtained directly from its
trigonometric factors:

\[
 |h(u)|\le\frac{|u|^3}{6}\quad(u\in\mathbb R),\qquad
 h(u)\ge\frac{1081}{6912}u^3\quad(0<u\le1/18).
\tag{7}
\]

The upper bound uses \(|\sin9u|\le9|u|\) and
\(1-\cos9u\le(9u)^2/2\). For the lower bound, put \(z=9u\le1/2\):
\(\sin z\ge(23/24)z\) and \(1-\cos z\ge(47/96)z^2\)
follow from the cubic and quartic Taylor bounds. In particular, (6) yields

\[
 \frac{d^3}{7}<h(\delta_1(s))\le\frac{d^3}{6}.
\tag{8}
\]

For the endpoint comparison restrict again to \(0\le s\le1\). Put
\(D(s)=\delta_1(s)-\delta_0(s)\). Subtract their exact Volterra
equations, using the shared initial state:

\[
 D(s)=-G\int_0^s(1-e^{-(s-t)})
  \left[\sin\delta_1(t)-\sin\delta_0(t)+h(\delta_1(t))\right]dt.
\tag{9}
\]

Since sine is globally one-Lipschitz and the kernel lies in \([0,1]\),
\(|D(s)|\le G\int_0^s(|D(t)|+d^3/6)\,dt\).
The integral comparison and \(e^G-1\le2G\) give

\[
 |D(s)|\le\frac{d^3}{6}(e^{Gs}-1)\le\frac{Gd^3}{3}.
\tag{10}
\]

Combining (8)--(10), with the strict canonical bound on \(G\), gives

\[
 \frac{d^3}{8}
 <h(\delta_1(s))+\sin\delta_1(s)-\sin\delta_0(s)
 <\frac{d^3}{5}.
\tag{11}
\]

Both histories react freely. Their endpoint contrast difference, directly
from the form row rather than separate approximations to each history, is

\[
\begin{split}
 R_0(d)-R_1(d)
 &=\gamma\int_0^1e^{t-1}
  \left[h(\delta_1(t))+\sin\delta_1(t)-\sin\delta_0(t)\right]dt,\\
 \boxed{\frac{\gamma d^3}{20}
   <R_0(d)-R_1(d)<\frac{\gamma d^3}{6}.}
\end{split}
\tag{12}
\]

For these deliberately loose constants it suffices to use
\(3/5<1-e^{-1}<2/3\). The sine endpoint is larger, although both
nominal form contrasts are negative. The proof compares the two actual
solutions analytically; it evaluates no numerical trajectory.
Subtracting independent endpoint approximations with an
\(O(\gamma^3d)\) error would not prove a uniform cubic law as
\(d\to0\). The direct difference (9) retains the shared linear
response and controls its constitutive correction at order \(Gd^3\).

## 3. Full fine state and independent preparation errors

For each candidate separately, admit the primitive bounds

\[
 |r(0)|\le\varepsilon_r,\qquad
 |\delta(0)-d|\le\varepsilon_\delta,\qquad
 \|z(0)\|\le\rho_x,\qquad \|q(0)\|\le\rho_\theta.
\tag{13}
\]

The last two norms combine all eighteen residuals in (3); their means
vanish separately in each child. The relative phase error uses the declared
initial lifts. A common form offset or phase rotation is immaterial by the
complete law's exact symmetries. The candidates may choose different actual
preparations anywhere in these bounds; no hidden pairing cancels their errors.

The [uniform fine-retention proof](ORGANIZED_FREE_PHASE_PROBE.md#free-phase-probe-fine-retention)
uses neither a large mean form nor a particular mean phase. Its Hessian bounds
hold over the entire relative phase circle. Consequently it applies here,
and through the extended timing horizon, if

\[
 F_0=\frac12(\rho_\theta+\gamma\rho_x)^2
            +\frac12\gamma^2\rho_x^2<\frac1{40000}.
\tag{14}
\]

Both fine identities are retained, and the true current defect satisfies

\[
 |\overline J_\lambda-j_\lambda(\delta)|\le B e^{-Ks},
 \qquad B=\frac{4J_2F_0}{9},\qquad K=\frac{125}{2}.
\tag{15}
\]

This is the existing own-law estimate on every retained fine direction.
It does not set the reached state to its carrier, infer the preparation from
an operator label, or transfer a different law's source certificate.

Compare an actual history with the nominal history of its own law. Let
\(\Delta r=r-r_{\lambda,\mathrm{nom}}\) and
\(\Delta\delta=\delta-\delta_{\lambda,\mathrm{nom}}\).
Variation of constants in the two actual mean rows gives

\[
\begin{split}
 \Delta r(s)&=e^{-s}r(0)-\gamma\int_0^s e^{-(s-t)}
  \left[j_\lambda(\delta(t))-j_\lambda(\delta_{\lambda,\mathrm{nom}}(t))
                  +e_f(t)\right]dt,\\
 \Delta\delta(s)&=\delta(0)-d+\gamma(1-e^{-s})r(0)
 -G\int_0^s(1-e^{-(s-t)})
  \left[j_\lambda(\delta(t))-j_\lambda(\delta_{\lambda,\mathrm{nom}}(t))
                  +e_f(t)\right]dt,
 \qquad |e_f(t)|\le Be^{-Kt}.
\end{split}
\tag{16}
\]

For \(0\le t\le3/2\), define the following rational bounds:

\[
 A=\varepsilon_\delta+\gamma\varepsilon_r+\frac{GB}{K},
 \qquad P_t=\frac{A}{1-GJ_1t}.
\tag{17}
\]

The denominator is positive. Taking a supremum in the second row of (16)
gives \(M_t\le A+GJ_1tM_t\), hence
\(|\Delta\delta(s)|\le P_t\) for every \(s\le t\).
At the nominal readout, the exact fine convolution is
\((e^{-1}-e^{-K})/(K-1)\). Using \(e^{-1}<2/5\) and
\(1-e^{-1}<2/3\) in the first row of (16) therefore gives

\[
 |r(1)-R_\lambda(d)|\le E,
 \qquad E=\frac25\varepsilon_r+
       \frac23\gamma J_1P_1+\frac25\frac{\gamma B}{K-1}.
\tag{18}
\]

The factor involving \(K\) keeps the fast decay of the fine state. A
persistent fine forcing was not inserted in its place. This bound is applied
once to each candidate, without assuming equal hidden states.

## 4. Reading sets, separation and a true overlap witness

Fix nonnegative exposure, gain and additive reading allowances
\(\eta\le1/2\), \(g\), and \(\varepsilon_y\). The sole observation is

\[
 Y_\lambda=(1+\kappa)r_\lambda(1+\xi)+b,
 \qquad |\xi|\le\eta,\quad |\kappa|\le g,\quad |b|\le\varepsilon_y.
\tag{19}
\]

Here \(\xi\) is an error in the independently declared \(s\) clock, and
\(\kappa\) changes the readout, not a kinetic coefficient. An unknown clock
rate or physical sensing law is not supplied by these allowances. A gain
allowance exceeding one is permitted but may destroy useful information.

Put \(T=1+\eta\). The nominal corridor (6) gives
\(|r_{\lambda,\mathrm{nom}}(s)|\le\gamma J_1d\) through \(T\).
Also, \(\int_0^s e^{-(s-t)}e^{-Kt}dt\le1/K\). Equations (16)--(17)
and the actual mean form row imply

\[
\begin{split}
 |r(s)|&\le R=\varepsilon_r+\gamma J_1(d+P_T)+\frac{\gamma B}{K},\\
 |r_s(s)|&\le L=\varepsilon_r+2\gamma J_1(d+P_T)
                    +\gamma B(1+1/K),\qquad 0\le s\le T.
\end{split}
\tag{20}
\]

For the derivative bound, use \(|\delta(s)|\le d+P_T\),
\(|j_\lambda(\delta)|\le J_1|\delta|\), and (15).
All exposures in (19) are nonnegative. The resulting error around each
nominal endpoint is

\[
 |Y_\lambda-R_\lambda(d)|\le
 E_{\rm obs}=E+\eta L+gR+\varepsilon_y.
\tag{21}
\]

The gain term already bounds the entire actual endpoint, so another gain
factor on the preparation or timing error would count that uncertainty twice.
Let \(\mathcal Y_\lambda\) be the set of all readings (19) from all allowed
preparations of that candidate. The positive gap (12) gives the sufficient
full-state conclusion

\[
 2E_{\rm obs}<\frac{\gamma d^3}{20}
 \quad\Longrightarrow\quad
 \mathcal Y_0\cap\mathcal Y_1=\varnothing.
\tag{22}
\]

Both candidates can choose their nuisance values independently; the factor
two is essential. A failed inequality establishes no intersection by itself.
There is, however, a separate sufficient ambiguity condition:

\[
 \varepsilon_y\ge\frac{\gamma d^3}{12}
 \quad\Longrightarrow\quad
 \mathcal Y_0\cap\mathcal Y_1\ne\varnothing.
\tag{23}
\]

To prove it, use the exact nominal preparations, zero timing and gain errors,
and the common reading \(Y_*=(R_0(d)+R_1(d))/2\). Under each law choose
the additive error \(b_\lambda=Y_*-R_\lambda(d)\). Equation (12) places
both magnitudes below the allowance. This is an actual common-reading witness,
not overlap of two conservative enclosures. It establishes failure of uniform
identification in this observation class; it does not make every possible
record ambiguous or rule out richer observations.

## 5. Attachment resource and what the scaling means

At the supplied state-preserving initial attachment, the exact new storage is

\[
 W=a\sum_{k=0}^8\left[
       \frac12(r(0)+z_{A,k}-z_{B,k})^2+
       V_\lambda(\delta(0)+q_{A,k}-q_{B,k})\right]\ge0.
\tag{24}
\]

The linear fine terms cancel because their means are zero. Using the global
curvature bound and \(V_\lambda(u)\le J_1u^2/2\) gives

\[
 W\le a\left[\frac92\varepsilon_r^2+\rho_x^2+
             \frac92J_1(d+\varepsilon_\delta)^2+J_1\rho_\theta^2\right].
\tag{25}
\]

For the exact nominal preparations, \(W_\lambda=9aV_\lambda(d)\).
The Taylor bounds
\(d^2/2-d^4/24\le V_\lambda(d)\le d^2/2+d^4/24\) imply

\[
 4ad^2<W_\lambda<5ad^2.
\tag{26}
\]

Thus this fixed model and observation have a nominal signal of order
\(\gamma d^3\) for nominal attachment storage of order \(ad^2\).
At fixed canonical parameters, the signal is of order
\(\gamma(W_\lambda/a)^{3/2}\). Reducing preparation reduces the signal
faster than this attachment cost. This is not a universal thermodynamic
measurement bound; formation, phase preparation, sensing and timing costs
are separate, and \(W\) is the declared structural storage jump.

For sufficiently small constants, (22) admits nonzero errors with
\(\varepsilon_r,\varepsilon_y\) of order \(\gamma d^3\),
\(\varepsilon_\delta,F_0\) of order \(d^3\), and \(\eta,g\) of order
\(d^2\). These are sufficient joint scalings, not separately proved necessary
ones. The explicit additive-noise obstruction (23) is retained independently.
The horizon remains \(\tau=2000/\gamma^2\); reducing \(d\) does not shorten it.

## 6. Detached calculation and scope

The [shared calculator](../../src/tnfr/physics/_organized_small_resource_information.py)
re-admits all primitive error allowances and the nominal phase before
rebuilding (14)--(26). It reuses the current family, canonical parameter
enclosures and fine-energy owner from the free probe. Exact rational arithmetic
retains arbitrarily small nonzero preparations; outward gamma endpoints give
conservative bounds without a floating-point cutoff.

Its statuses distinguish `guaranteed_separation`,
`observation_class_nonidentifiable` and `not_certified`. The first is a
prospective guarantee, the second uses the nominal midpoint witness, and the
third establishes neither conclusion. No status selects a law from data.
Bounds outside the fine-retention domain reject admission instead of claiming
that an identity was lost. Supplied errors and source premises are not
authenticated by the calculation.

The [static controls](../../tests/physics/test_organized_small_resource_information.py)
check direct difference algebra, fine convolution, independent uncertainty,
work cancellation, precision boundaries and a strictly positive budget with
all error allowances nonzero. No trajectory, source producer or physical
response is evaluated. A phase preparation, a clock and a sensor remain
independent obligations before comparing with measured physics.

