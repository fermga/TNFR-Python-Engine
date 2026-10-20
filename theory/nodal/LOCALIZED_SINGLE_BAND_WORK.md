# Same-band source admission and regional work

**Scope:** a bounded two-lobe preparation on one carrier band, a geometric
screen absent throughout that source class, an exact necessary screen-work
condition, and a strictly signed nominal nonlinear rate distinction between
resource-matched initial controls. A lobe is a region of concentrated full-state amplitude; a chosen
sum does not supply independent dynamical constituents or an NFR identity.
This is analytical admission, not an evaluated trajectory, formation result
or physical identification. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns this branch's status and any later response evaluation.

## 1. Complete state and a bounded one-carrier preparation

Retain the [complete C1024 law](LOCALIZED_FORMATION_COMPATIBILITY.md#1-complete-law-state-and-clock),
with unit conductances and lengths, held unit capacities, beta one, no input
or support event, and the supplied real lift of primitive circular phase.
Write `x=EPI`, `z=x+i*theta`, `(Dq)_j=q_(j+1)-q_j` for cycle incidence, `L=D^T D`
and `A=L/2`. The bookkeeping complex field retains two real rows:

\[
 x'=-\sigma Ax-\tfrac12D^T\sin(D\theta),\qquad
 \theta'=Ax,\qquad
 \tau=wt/\pi,\quad w=1-2^{-20},\quad
 \sigma=\pi2^{-20}/w>0.                                      \tag{1}
\]

The prime denotes the fixed normalized clock. Carrier phase is not primitive
phase, and `|z|^2=x^2+theta^2` is a localization diagnostic rather than local
structural storage. The full state, including generated modes and error means,
remains retained. Neither this preparation nor a Fourier observation closes
the dynamics on one band.

Set `N=1024`, `Q=16`, `k=pi/4`, `a=2^-12` and `delta=a/1000`. Reuse the
[Fejer normalization and tail bound](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-error)
only, without transferring the earlier carrier's transport certificate:

\[
 b_m=1-|m|/16\quad (|m|<16),\qquad S_2=171/16,
 \qquad
 f(j)=\frac{\sum_{|m|<16}b_m e^{2\pi imj/N}}{\sqrt{NS_2}}.
                                                               \tag{2}
\]

The profile is real, nonnegative, even, periodic and unit-normalized. For
each supplied `psi` in `[-pi,pi]`, define the preparation class

\[
 z_\psi^0(j)=a e^{ikj}[f(j)+e^{i\psi}f(j-512)],\qquad
 z(0)=z_\psi^0+r,\qquad \|r\|_2\le\delta.                    \tag{3}
\]

The primitive form is the real part of (3), and its imaginary part is the
supplied real phase lift. Explicitly, their nominal values are
`a[f(j)cos(kj)+f(j-512)cos(kj+psi)]` and
`a[f(j)sin(kj)+f(j-512)sin(kj+psi)]`. The two terms are preparation labels
on the same band, modes 113 through 143. There are no independent band
projections that recover them from an evolved total field.

The exact initial overlap and norm are

\[
 s=\langle f,f(\cdot-512)\rangle
   =\frac{\sum_m b_m^2(-1)^m}{S_2}=\frac1{171},\qquad
 \|z_\psi^0\|_2^2=2a^2(1+s\cos\psi).                         \tag{4}
\]

Indeed the numerator of the overlap is `1/16`, while the squared-weight
sum is `171/16`. Thus changing relative phase at fixed component amplitude
does not generally hold total norm fixed. Arbitrary errors are retained,
not projected onto the nominal band. The triangle inequality gives

\[
 a\sqrt{340/171}-\delta\le\|z(0)\|_2
 \le a\sqrt{344/171}+\delta,
 \qquad \frac{1409}{1000}a<\|z(0)\|_2<\frac32a.              \tag{5}
\]

Since `||f||_infinity=2/sqrt(171)`, primitive admission also gives

\[
 \|z(0)\|_\infty\le a(4/\sqrt{171}+1/1000)<a/3,
 \qquad |D\theta(0)|<2a/3=1/6144<\pi/2.                      \tag{6}
\]

The nominal mean vanishes because the band omits zero. Error means remain:
`|mean z(0)|<=delta/sqrt(N)=a/32000`. Equation (1) conserves both real
means; it does not set them to zero.

The exact structural storage and initial bounds are

\[
 H=\tfrac12\|Dx\|_2^2+\sum_j[1-\cos(D\theta)_j],\qquad
 H'=-\tfrac\sigma2\|Lx\|_2^2,
 \qquad 0\le H(0)\le\tfrac12\|Dz(0)\|_2^2<\tfrac92a^2<1.
                                                               \tag{7}
\]

The bound uses `1-cos u<=u^2/2` and `||D||_2<=2`. Together with the initially
acute lift, the storage balance prevents a first edge crossing of `+/-pi/2`,
which would contribute one to H. This familiar chart bound does not preserve
the initial localization or full-state norm, and does not prove organization.

<a id="single-band-initial-absence"></a>
## 2. A total-field concentration screen initially absent

For a real observation center c on the unit-length cycle, let W(c,96) be
the radius-96 ball. Let `W_1=W(0,96)` and `W_2=W(512,96)`. The reused
spatial-tail inequality gives, for each shifted unit profile,

\[
 \|\mathbf1_{W_i^c}f_i\|_2^2\le
 b:=\frac{1024^3}{16\cdot16^3\cdot95^3}
 =\frac{16384}{857375}<\frac1{49}.                            \tag{8}
\]

The nominal components are not disjoint, but both initial lobes have a
nonvanishing regional signal even with arbitrary allowed error and phase:

\[
 \|\mathbf1_{W_i}z(0)\|_2
 \ge a[\sqrt{1-b}-\sqrt b]-\delta>\frac45a.
                                                               \tag{9}
\]

Here the other lobe contributes at most `a*sqrt(b)` because the two source
windows are disjoint. Outside their union,
`||1_(W_1 union W_2)^c z(0)||_2<=2a*sqrt(b)+delta<3a/10`.
These inequalities describe two separated concentrations, without assigning
them unique constituent identities under future overlap or deformation.

The proposed common-window screen requires `||z||_2>=a` and one radius-96
window containing at least `3/4` of the total squared norm. It is a
geometric prerequisite for a narrower possible organization, not a complete
relational identity. It does not consume independent component projections.

For every c, at least one source center is at cycle distance at least 256.
Since `256>96+96`, W(c,96) misses that source window. Its component
contribution is therefore at most `a*sqrt(b)`; the other component's full
norm is a. Equations (5) and (8) imply

\[
 \frac{\|\mathbf1_{W(c,96)}z(0)\|_2^2}{\|z(0)\|_2^2}
 \le\left(\frac{1+\sqrt b+1/1000}
                   {\sqrt{340/171}-1/1000}\right)^2
 <\left(\frac{8007}{9863}\right)^2<\frac23.                   \tag{10}
\]

Thus the `3/4` concentration requirement fails with margin greater than
`1/12` for every center, phase and member of the source-error ball. The
signal floor holds independently by (5), and the cycle triangle inequality
also covers fractional centers. This is a fresh same-band total-field
argument; the previous disjoint-band absence test has not been reused.

Later entry into this screen could occur by ordinary interference or
dispersion, and by itself would not establish acquisition, retention or a
collective response. A two-lobed organization wider than this screen is not
excluded by its failure. The [formation contract](LOCALIZED_ORGANIZATION_GEOMETRY.md#common-pattern-formation-contract)
still requires a retained relation, a justified lifetime and a discriminating
response on the continued full state.

<a id="single-band-matched-initial-resources"></a>
## 3. Exactly matched initial resources without deleting phase

Equal component amplitudes do not match total norm across arbitrary psi,
by (4). They do not match exact H either. The quadratic storage comparison
is `H_2=||Dz||_2^2/2`, and

\[
 0\le H_2-H\le\frac1{24}\sum_j(D\theta)_j^4.                  \tag{11}
\]

Replacing exact H by this comparison would discard a phase-dependent
remainder. Rescaling amplitudes to match norm alone supplies another
preparation and does not certify matched storage or initial loss.

A useful matched pair is instead `+psi` and `-psi`. Let
`(Jq)_j=q_(-j)` and define the real-linear isometry

\[
 \mathcal S(x,\theta)=(Jx,-J\theta),\qquad
 (\mathcal S z)_j=\overline{z_{-j}}.
                                                               \tag{12}
\]

Evenness of f and its antipodal shift gives
`S z_psi^0=z_(-psi)^0`. Pair errors by `r_(-psi)=S r_psi`; independently
chosen errors of equal norm need not match resources. For forward edge
index j, put `(J_e q)_j=q_(-j-1)`, `y=Dx` and `u=Dtheta`. Then

\[
 y(\mathcal S z)=-J_e y(z),\qquad
 u(\mathcal S z)=J_e u(z),\qquad
 h_j=\tfrac12y_j^2+1-\cos u_j.
                                                               \tag{13}
\]

Consequently the paired full preparations have exactly equal norm, total
H and `||Lx||_2^2`, hence equal initial global loss. Pointwise localization
and edge storage are reflected. For an edge subset E,
`H_E(Sz)=H_(J_e E)(z)`; a same-region comparison requires
`J_e E=E`, otherwise its reflected region must be paired explicitly.
Means are conjugated, including the retained error means, rather than
erased. No fitted amplitude or hidden reservoir is involved.

The map S is not a positive-time symmetry of (1). It matches initial
resources while changing the allocation of signed form and phase, whose
actual subsequent rows must be evaluated separately. Reflection J alone
is a graph symmetry; adding phase reversal changes this conclusion.
The paired source classes establish a valid initial control, not equality
of their future trajectories or a sign for future regional work.

<a id="single-band-screen-prefix-work"></a>
## 4. A necessary prefix-work condition for the absent screen

The concentration diagnostic and structural storage are different
observations. A positive local storage rate alone cannot establish entry
into (10). Fix a node window W, put `P_W=diag(1_W)`, `alpha=3/4` and
`B_W=P_W-alpha*I`, and retain the quadratic screen gap

\[
 V_W(z)=x^TB_Wx+\theta^TB_W\theta
       =\|P_Wz\|_2^2-\alpha\|z\|_2^2.                       \tag{14}
\]

Screen entry in that window requires `V_W>=0` and the separate signal
floor `||z||_2>=a`. For every admitted source, (10) gives
`V_W(z(0))<-||z(0)||_2^2/12`. No regional-storage bound was substituted
for that initial diagnostic margin.

Differentiate (14) using both actual rows in (1). With `u=Dtheta`,

\[
 \begin{split}
 V_W'&=\mathcal L_W(z)+\mathcal N_W(z),\\
 \mathcal L_W(z)
   &=-2\sigma x^TB_WAx+2\theta^T(B_WA-AB_W)x,\\
 \mathcal N_W(z)&=(DB_Wx)^T(u-\sin u).
 \end{split}                                                  \tag{15}
\]

Here `L_W` is the rate under the actual damped linear tangent evaluated
at the same full state, not a nonnegative loss estimate. A restricted
quadratic observation has boundary transport and can grow even while
global structural storage decreases. `N_W` is the exact nonlinear rate
insertion relative to that tangent. Its cubic-sine comparison is

\[
 \mathcal N_{W,3}=\tfrac16(DB_Wx)^Tu^{\circ3},\qquad
 |\mathcal N_W-\mathcal N_{W,3}|
 \le\tfrac1{120}\sum_j |(DB_Wx)_j|\,|u_j|^5.                 \tag{16}
\]

The elementwise cube is not an envelope equation. Under joint amplitude
scaling this leading rate is fourth order; the remainder is sixth order.
Both still consume full form, primitive phase and the selected boundary.
The sine Taylor remainder proves (16) globally on real lifts.

If a solution first satisfies the screen in W at a time T, necessarily

\[
 \int_0^T [\mathcal L_W(z(\tau))+\mathcal N_W(z(\tau))]\,d\tau
 \ge -V_W(z(0))>\frac{\|z(0)\|_2^2}{12}.                    \tag{17}
\]

This is a source-dependent necessary condition, not a sufficient entry
criterion. For entry in any radius-96 window, apply (17) to the successful
window held fixed over the prefix; its initial absence holds for every
center. A moving-window rate or continued moving identity requires its
own membership transport. The norm floor and later maintenance remain
separate obligations.

If W is invariant under node reflection, `B_W` commutes with J. Equation
(13) then gives

\[
 V_W(\mathcal Sz)=V_W(z),\qquad
 \mathcal N_W(\mathcal Sz)=-\mathcal N_W(z).                  \tag{18}
\]

The dissipative contribution in `L_W` is unchanged and its conservative
commutator contribution changes sign. Thus relative-phase allocation can
change the linear part of the same criterion as well. Equation (18) does
not prove that `N_W` is nonzero on a particular source or stays signed
along its future evolution.

<a id="single-band-regional-storage-insertion"></a>
## 5. Structural regional work and its nonlinear rate insertion

The [existing exact local storage balance](LOCALIZED_RETENTION_MECHANISMS.md#cycle-local-storage-balance)
uses `L_e=DD^T`, `y=Dx`, `u=Dtheta` and

\[
 F_{jk}=\tfrac12(\sin u_j\,y_k-y_j\sin u_k)
        +\tfrac\sigma4(y_j^2-y_k^2)=-F_{kj}.
\]

For a fixed edge subset E define

\[
 \begin{split}
 H_E&=\sum_{j\in E}(y_j^2/2+1-\cos u_j),\\
 I_E&=-\sum_{j\in E,\,k\notin E,\,k\sim j}F_{jk},\\
 D_E&=\tfrac\sigma4\sum_{j\in E}\sum_{k\sim j}(y_j-y_k)^2.
 \end{split}
 \qquad H_E'=I_E-D_E.                                       \tag{19}
\]

An internal edge pair appears twice in the local-loss sum, as required
by the local balance. Taking E to be the full edge set gives the global loss in (7).
The inward current is signed; neither it nor a global loss budget is an
event reserve or a guarantee of acquisition.

For comparison, use the same structural H_E and the damped tangent

\[
 x'_{\rm lin}=-\sigma Ax-A\theta,\qquad
 \theta'_{\rm lin}=Ax.                                      \tag{20}
\]

This tangent is real-linear in the two state channels. Its positive-loss
term acts on form, so a uniform complex carrier rotation does not generally
commute with its propagator. Each phase preparation must retain both rows;
rotating one precomputed history is not an exact control for another.

At an identical supplied full state their derivative difference is

\[
 G_E(z):=\left.\frac{dH_E}{d\tau}\right|_{F_\sigma(z)}
        -\left.\frac{dH_E}{d\tau}\right|_{J_\sigma z}
 =\tfrac12(P_Ey)^TL_e(u-\sin u),\qquad P_E=\operatorname{diag}(1_E).
                                                               \tag{21}
\]

Only the form-rate difference contributes because the phase rows agree.
This exact instantaneous insertion is not `H_E(joint)-H_E(A)-H_E(B)`
and not an endpoint difference between evolved histories. Its leading
term and remainder are

\[
 G_{E,3}=\tfrac1{12}(L_eP_Ey)^Tu^{\circ3},\qquad
 |G_E-G_{E,3}|\le\tfrac1{240}\sum_j|(L_eP_Ey)_j|\,|u_j|^5.
                                                               \tag{22}
\]

Writing `q=u-sin u`, the exact incidence decomposition is

\[
 G_E=\tfrac12\sum_{\{j,k\}\subset E,\,j\sim k}
           (y_j-y_k)(q_j-q_k)
      +\tfrac12\sum_{j\in E,\,k\notin E,\,j\sim k}
           y_j(q_j-q_k).                                    \tag{23}
\]

Internal pairs in the first sum are unordered and counted once. Thus
G_E is not solely boundary inflow: it is the change of the structural
storage rate caused by replacing the tangent pressure by actual sine
pressure. The linear comparison still satisfies an exact structural ledger,
`H_E'=I_E-D_E-G_E`, when all terms are evaluated on its own linear state.

The matched preparation map gives
`G_E(Sz)=-G_(J_e E)(z)`. For `J_e E=E` this is an exact sign reversal,
while H_E and the local loss match initially. It does not preserve the
two full trajectories or their later current.

<a id="single-band-nominal-signed-work"></a>
## 6. A strictly signed nominal phase control

Sign reversal alone permits a zero insertion. A finite exact witness
within (3) avoids this ambiguity. Take the nominal source `r=0`,
`psi=pi/2` and the two-edge region `E={-1,0}` modulo N, which is
invariant under `j -> -j-1`. This small region measures a local rate;
it is distinct from the radius-96 acquisition screen.

Put `f_j=f(j)`, `g_j=f(j-512)` and `c=1/sqrt(2)`. Define

\[
 A_0=f_0-cf_1>0,\qquad B_0=cg_1>0,\qquad
 U=cf_1,\qquad V=f_2-cf_1<U.                                \tag{24}
\]

These inequalities follow from the source, without a numerical fit.
The positive Fourier weights give `f_1<=f_0`, so
`A_0>=(1-c)f_0>0`. Their nonzero frequencies satisfy
`|2*pi*m/N|<=15*pi/512<pi/4`, hence `f_1>cf_0` and
`2cf_1>f_0>=f_2`, proving `V<U`. The Fejer formula gives `g_0=0`
and `g_1>0`: its numerator at the antipodal offset is proportional to
`sin(pi/64)^2`, and its denominator is nonzero.

Evenness and the exact carrier values at nodes 0, +/-1 and +/-2 give

\[
 \begin{array}{ll}
 y_{-1}=a(A_0-B_0),&y_0=-a(A_0+B_0),\\
 u_{-1}=a(U-B_0),&u_0=a(U+B_0),\\
 u_{-2}=a(V+B_0),&u_1=a(V-B_0).
 \end{array}                                                 \tag{25}
\]

Let `q(t)=t-sin(t)`, `q_(U,+/-)=q(a(U+/-B_0))` and likewise for V.
Substitution into (21) yields

\[
 \begin{split}
 G_E(z_{\pi/2}^0)=\frac a2\{&
 A_0[3(q_{U,-}-q_{U,+})+(q_{V,-}-q_{V,+})]\\
 &-B_0[(q_{U,-}+q_{U,+})-(q_{V,-}+q_{V,+})]\}<0.
 \end{split}                                                 \tag{26}
\]

Indeed q is strictly increasing: its derivative `1-cos(t)` is
nonnegative and vanishes only at isolated points, with no constant
interval. Because `a,B_0>0`, each difference in the first bracket is
negative. Since `U>V`, each q value in the second U sum exceeds its
corresponding V value, so the second bracket is positive. Therefore

\[
 G_E(z_{-\pi/2}^0)=-G_E(z_{\pi/2}^0)>0.                     \tag{27}
\]

The two preparations have exactly equal initial total norm, exact
storage, global loss and reflected regional storage. Nevertheless the
actual sine pressure changes this structural regional storage rate above
the same-state linear value in one preparation and below it in the
other. This is a strict local, initial nonlinear distinction. It does
not imply positive total regional growth, an attractive force, screen
entry, restoration, lifetime or acquired NFR identity. No robust sign
over the nonzero error ball, or sign of the screen insertion N_W, has
been proved by this witness. In particular, `B_0>0` comes from the remote
Fejer lobe's nonzero tail: the witness already has local overlap initially.
A strict nominal sign need not dominate the admitted arbitrary source error
or the linear contribution. It does not establish newly acquired contact.

<a id="single-band-matched-history-boundary"></a>
## 7. Matched histories retain transport and comparison state

Let z be a full-law solution and ell the tangent solution (20), with
the identical admitted primitive preparation `z(0)=ell(0)`. The same
screen observation has the exact difference

\[
 V_W(z(T))-V_W(\ell(T))
 =\int_0^T[\mathcal L_W(z)-\mathcal L_W(\ell)
                      +\mathcal N_W(z)]\,d\tau.              \tag{28}
\]

The linear transport terms are evaluated on different evolving states
and do not cancel. Integrating only the initial or future N_W omits
this feedback. Conversely, the structural regional-storage difference is

\[
 H_E(z(T))-H_E(\ell(T))
 =\int_0^T[I_E(z)-I_E(\ell)-D_E(z)+D_E(\ell)+G_E(\ell)]\,d\tau.
                                                               \tag{29}
\]

Here G_E is evaluated on ell because its structural ledger in (23)
differs from the full-law ledger. Equations (28) and (29) concern
different observations and cannot substitute for each other.

A separate-history control requires an explicit primitive split,
including an allocation of the admitted source error. If z_A and z_B
evolve independently under (1), their sum `s=z_A+z_B` is an observation
on the common support. It need not solve (1). With
`u_A=Dtheta_A`, `u_B=Dtheta_B`, define

\[
 \mathcal N_W^{\rm sep}(z_A,z_B)
 =(DB_W(x_A+x_B))^T
   [u_A-\sin u_A+u_B-\sin u_B].                              \tag{30}
\]

One declared split covering the whole source class is
`z_A(0)=a*exp(i*k*j)*f(j)+r/2` and
`z_B(0)=a*exp(i*(k*j+psi))*f(j-512)+r/2`. It gives the exact total
preparation (3), with each assigned error at most delta/2. This equal
assignment is comparison metadata, not an inferred physical partition.
Other splits require their own retained initial metadata and bounds.

For a matching initial sum, the actual and separate observed screen
gaps obey

\[
 V_W(z(T))-V_W(s(T))
 =\int_0^T[\mathcal L_W(z)-\mathcal L_W(s)
            +\mathcal N_W(z)-\mathcal N_W^{\rm sep}(z_A,z_B)]\,d\tau.
                                                               \tag{31}
\]

At one supplied shared state, the nonlinear insertion difference is

\[
 \mathcal N_W(z_A+z_B)-\mathcal N_W^{\rm sep}(z_A,z_B)
 =4(DB_W(x_A+x_B))^T
    \left[\sin\frac{u_A}{2}\sin\frac{u_B}{2}
                         \sin\frac{u_A+u_B}{2}\right].       \tag{32}
\]

This reuses the [exact mixed sine identity](LOCALIZED_FORMATION_COMPATIBILITY.md#instantaneous-mixed-sine-pressure).
The cross terms in `V_W(s)` are retained; replacing it by
`V_W(z_A)+V_W(z_B)` would change the observable. Likewise the initial
structural cross storage is generally nonzero:

\[
 H_E(z_A+z_B)-H_E(z_A)-H_E(z_B)
 =\sum_{j\in E}\left[y_{A,j}y_{B,j}
       +4\sin\frac{u_{A,j}}2\sin\frac{u_{B,j}}2
                         \cos\frac{u_{A,j}+u_{B,j}}2\right]. \tag{33}
\]

Its evolved difference must retain each history's current and loss.
The split is supplied preparation information, not a unique nonlinear
decomposition or proof of autonomous constituent identities.

## 8. Bounded result and remaining formation obligations

The admitted source supplies two separated concentrations with an
initially absent total-field screen. Equations (15)-(17) give the exact
necessary prefix work for that screen; (21)-(27) prove a strictly signed
nonlinear local rate distinction for two exactly resource-matched nominal
phases. The [static controls](../../tests/physics/test_localized_single_band_work.py)
check these algebraic and source statements without evaluating a prepared
trajectory or reserved response.

This establishes a bounded conditional compatibility result and its
comparison boundary. It does not establish that the positive local insertion can
overcome the screen deficit, that the full-law history passes a screen
which the linear or separate controls fail, or that any resulting
concentration maintains a new relation. A future formation discriminator
still needs those actual-state transport estimates, a separate signal
floor, a justified retention interval and a continued-response condition.
The exact balances identify that missing signed work; they do not admit
a trajectory, parameter sweep or long-horizon calculation by themselves.
