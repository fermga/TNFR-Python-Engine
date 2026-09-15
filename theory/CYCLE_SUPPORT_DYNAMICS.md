# Joint structural support and nodal evolution on a cycle

**Status:** Exact finite reset/flow budgets, a default-factor retention
obstruction and finite canonical runtime checks. Autonomous, self-restoring
localized structures remain unproved.
**Research links:** B2.d.3/O3.a, S3, S8, S9 and S16.

## 1. All three active pressure channels share a cycle coordinate

Use a fixed simple unit-conductance cycle with normalized Laplacian `L=I-W/2`
and conductance Laplacian `B=2L`. Let `x` denote scalar EPI and `nu>=0` capacity.
In a consistent phase lift, write

$$
\phi_i=2\pi Wi/n+p_i,\qquad a_i=p_i/\pi,
$$

where `p` is periodic. Require every oriented lifted gap to have absolute
value strictly below both the effective U3 gate and `pi/2`. Each neighbor
phasor sum then has a nonzero resultant on its unambiguous midpoint arc.
The canonical phase gradient is exactly `-L*a`; this is the midpoint
identity of [Coupling winding persistence](COUPLING_WINDING_PERSISTENCE.md),
not an independently imposed phase differential equation.

The regular unit cycle has zero topology gradient. Its EPI, capacity and
phase walks coincide, so with effective normalized channel coefficients
`e=w_epi>0`, `f=w_vf>=0`, `w=w_phase>=0`, the canonical pressure becomes

$$
\Delta\mathrm{NFR}=-L(e x+f\nu+w a)=-eLy,\qquad
y=x+\frac f e\nu+\frac w e a.
$$

The four-channel normalization remains intact even though the topology
gradient vanishes. A constant change in phase lift adds only a constant to
`y` and changes neither its pressure nor its Dirichlet energy. The detached
observer accepts `a` as a declared dimensionless coordinate; rational inputs
are not asserted to reconstruct a circle, a winding or mathematical pi.

This scope excludes weighted/support walk mismatches, changing topology,
clipping and ambiguous phase branches. The counterexamples in
[capacity-conditioned balance](CAPACITY_LOCALIZATION_BALANCE.md) continue to
apply. No confinement potential or additional physical coefficient is added.

## 2. Canonical support resets have a signed energy budget

Use target-only Coupling with `UM_BIDIRECTIONAL=False` and
`UM_FUNCTIONAL_LINKS=False`. Its immutable all-target stage, followed by
uniform Silence, gives the exact reference reset

$$
\nu^+=q(I-sL)\nu^-,\qquad
a^+=(I-\eta L)a^-,\qquad x^+=x^-.
$$

Here `s=UM_vf_sync`, `eta=UM_theta_push` and `q=SHA_vf_factor` are the existing
operator coefficients. For factors in `[0,1]`, capacity remains nonnegative.
The phase gap interval is preserved under the preceding strict chart
hypotheses. Runtime operator-local pressure writes do not replace the
subsequently refreshed three-channel pressure.

Define the support change and a fixed, capacity-independent diagnostic

$$
d_b=\frac f e(\nu^+-\nu^-)+\frac w e(a^+-a^-),\qquad
F(y)=\tfrac12y^\top By=\tfrac12\sum_i(y_{i+1}-y_i)^2.
$$

The same matrix `B` before and after the event gives the exact identity

$$
J:=F(y^+)-F(y^-)
=(y^-)^\top B d_b+\tfrac12d_b^\top B d_b.
$$

The cross term is signed: a support reset can increase or decrease `F`.
If the initial pressure is zero, `y^-` is constant and the cross term
vanishes. The new pressure is then

$$
\Delta\mathrm{NFR}^+
=f(1-q)L\nu^-+fqsL^2\nu^-+w\eta L^2a^-.
$$

This is not a universal one-step release theorem. Correlated capacity and
phase contributions can cancel. If `Lv=lambda*v`, take
`nu=nu_bar+b*v` and `a=A*v`, with

$$
A=-\frac{f[(1-q)+qs\lambda]}{w\eta\lambda}\,b
\quad (w\eta\lambda>0).
$$

Preparing `x=c-(f/e)*nu-(w/e)*a` then gives zero pressure both before and
after that reset. For the eigenvalue-one cycle mode, `e=1/2`, `f=w=1/4`,
`q=s=eta=1/2`, `b=1/8`, the exact control is `A=-3/16`.
This algebraic cancellation does not infer repeated invariance or a live
phase chart; smaller amplitudes retain the same cancellation when needed
to meet a prescribed strict chart margin.

## 3. Held support supplies the flow part of the budget

Between resets, hold `nu,a` fixed and evolve only EPI through the nodal law:

$$
\dot y=\dot x=-e\operatorname{diag}(\nu)Ly,\qquad
\dot F=-e(By)^\top\operatorname{diag}(\nu/2)(By)\leq0.
$$

No inverse capacity occurs. The identity remains valid when capacity
vanishes, although zero rate then does not imply zero pressure. In
particular, `q=0` freezes subsequent EPI evolution while phase support may
still change under later declared Coupling events.

Any finite compatible sequence of these exact resets and held-support
continuous flows obeys

$$
F_{\rm final}-F_{\rm initial}
=\sum_k J_k-
e\sum_k\int_{\text{flow }k}
(By)^\top\operatorname{diag}(\nu/2)(By)\,dt.
$$

An exact refreshed Euler segment has a different, explicit identity. With
`r=diag(nu)*DeltaNFR`, `x^+=x+h*r` and the same support,

$$
F(y+h r)-F(y)=h\dot F(y)+h^2F(r).
$$

The quadratic remainder is nonnegative. Since `L<=2I`, the condition
`h*e*max(nu)<=1` is sufficient for both a row-convex update of `y` and
Dirichlet nonincrease. Larger steps can increase energy even though the
continuous derivative is negative. A finite discrete telescope therefore
retains reset changes, negative linear flow terms and positive Euler
remainders separately. Runtime pressure-realization and arithmetic defects
are further measured contributions, not part of this exact equality.

## 4. Default attenuation can retain EPI as capacity tends to zero

The default factors obey `0<q<1` and `0<s<1/2`. If each UM/SHA event precedes
a fixed-duration flow of length `h`, the capacity at flow `k>=1` is

$$
\nu_k=q^k(I-sL)^k\nu_0,\qquad
\max\nu_k\leq q^k\max\nu_0.
$$

Consequently the accumulated capacity exposure is finite. For variable
durations this conclusion requires `sum_k h_k*q^k<infinity`; arbitrarily
increasing physical durations do not satisfy it automatically. With
spatially uniform initial capacity and a regular twist, an initial EPI
mode `epsilon*v`, `Lv=lambda*v`, retains the positive limiting amplitude

$$
\epsilon\exp\!\left[-\frac{e\lambda h\nu_0q}{1-q}\right].
$$

The leading `q` records the event-before-flow ordering. This is inherited
contrast retained through a finite reorganization budget, not restoration
against perturbations.

A stronger witness begins at exact zero pressure and uses the same default
factor ranges. On an even cycle let `v_i=(-1)^i`, so `Lv=2v`. Choose
`nu_0=A_0+D_0*v>0`, `D_0<0`, regular twist `a=0`, and
`x_0=m_0+b_0*v`, where `b_0=-(f/e)*D_0>0`. Initially the capacity and EPI
gradients cancel. At flow `k`, write

$$
A_k=A_0q^k,\qquad D_k=D_0[q(1-2s)]^k,\qquad
T_k=-\frac f eD_k>0.
$$

This two-mode subspace is preserved by the nodal equation. During that
held-support segment,

$$
\dot b=-2A_k(e b+fD_k),\qquad
\dot m=-2D_k(e b+fD_k).
$$

The exact endpoint recurrence is

$$
b_k=T_k+(b_{k-1}-T_k)e^{-2e A_kh}.
$$

Starting from `b_0=T_0`, induction gives
`T_k<=b_k<=b_{k-1}` and

$$
b_\infty\geq b_0
\exp\!\left[-\frac{2eA_0hq}{1-q}\right]>0.
$$

Moreover `m_k-m_{k-1}=(D_k/A_k)(b_k-b_{k-1})>=0`; its total increase is
at most `(|D_0|/A_0)*(b_0-b_infinity)`. Both EPI amplitudes converge, while
capacity and its spatial contrast vanish. The remaining nonuniform EPI
has nonzero limiting EPI pressure and vanishing nodal rate. Thus even an
initially balanced structural profile need not remain an active balance
when its supporting capacity is synchronized and attenuated. The selected
checkerboard preparation is an exact counterexample, not spontaneous
localization or a prediction of a particle-like object.

## 5. Positive mobility and genuine mixing give a different boundary

As a formal control, omit attenuation by taking `q=1`. This is not an
admitted Silence factor. Suppose initial capacity lies in `[m,M]`, `m>0`,
both `0<s,eta<1`, and every flow lasts a fixed `h>0`. The cycle maps
`I-sL` and `I-eta*L` then mix their nonconstant modes. For
`g_k=f*nu_k+w*a_k`, its gradient has the bound
`||L*g_k|| <= f*rho_s^k*||L*nu_0|| + w*rho_eta^k*||L*a_0||`, where
`rho_b=max_{lambda>0}|1-b*lambda|<1` over the cycle spectrum. This is a
geometric envelope, not a claim that the combined gradient norm decreases
at every event. The ordinary EPI
Dirichlet energy `E_x=x^T*L*x`, unchanged by support events, satisfies

$$
\dot E_x\leq-e m\|Lx\|^2+
\frac{M^2}{em}\|Lg_k\|^2
\leq-em\lambda_2 E_x+\frac{M^2}{em}\|Lg_k\|^2.
$$

The resulting forced scalar bound gives `E_x->0` and an integrable nodal
rate, hence a uniform limiting EPI. The positive-capacity, strictly mixing
control cannot sustain a nonuniform EPI profile.

The mixing hypothesis matters. On an even cycle, `eta=1` retains the
checkerboard phase mode as `a_k=epsilon*(-1)^k*v`; its amplitude can be
chosen within the strict gap margin. With uniform capacity, write
`rho=exp(-2*e*nu*h)`, `c=w*epsilon/e` and
`tau=(1-rho)/(1+rho)`. Its forced EPI amplitude has the exact solution

$$
b_k=\rho^k(b_0+c\tau)-c\tau(-1)^k.
$$

For `w*epsilon!=0` this approaches a nonzero period-two response.
Preserving winding and a strict gap interval therefore does not itself
prove loss of supporting phase structure or EPI consensus.

## 6. Executable scope and reusable evidence

[`cycle_support_dynamics.py`](../src/tnfr/physics/cycle_support_dynamics.py)
provides detached exact balance, UM/SHA reset and refreshed-Euler observers.
It shares the ordered rational reader and unit-cycle algebra with
[`capacity_localization.py`](../src/tnfr/physics/capacity_localization.py)
and [`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py).
Rationals remain exact; other supported real values become exact
representations of their materialized binary64 values. Default coefficients
come from the full canonical normalization. General explicit positive
coefficients are algebraic inputs and need separate runtime identification.
Public cached fields are recomputed before transitions, not trusted as
provenance. No live graph, chart, continuous trajectory or future execution
is certified by the detached records.

[Exact API tests](../tests/physics/test_cycle_support_dynamics.py) cover
channel cancellation, signed reset budgets, zero-capacity boundaries,
Euler remainders, finite telescoping and altered cached records.
[Pressure bridge tests](../tests/physics/test_cycle_support_pressure_bridge.py)
bind the three-channel identity and UM/SHA release to actual canonical
pressure on strict twisted cycles with both pressure backends.
[Limit controls](../tests/physics/test_cycle_support_limits.py) independently
check the default checkerboard reduction and the nonmixing phase boundary.
[Runtime tests](../tests/physics/test_cycle_support_dynamics_runtime.py) and
the [benchmark](../benchmarks/cycle_support_dynamics.py) use existing
grammar-admitted events and executor-owned physical partitions. Their
normalized phase reconstruction, endpoint pressure defects and arithmetic
remainders remain measured finite evidence. They establish neither generic
solver convergence nor future binary64 stability.

The supported conclusion is a structural distinction: fixed capacity and
phase profiles can condition a nonuniform zero-pressure EPI balance;
canonical evolution of those supporting fields must enter the energy
budget; loss of support with finite accumulated mobility can leave passive
EPI contrast. Demonstrating a self-generated, localized profile that
restores itself while sustaining reorganization requires further evidence.
