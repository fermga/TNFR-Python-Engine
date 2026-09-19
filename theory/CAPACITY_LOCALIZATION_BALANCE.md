# Capacity-conditioned EPI balance on a canonical cycle

**Status:** Exact fixed-capacity balance and relaxation theorem, with finite
canonical preparation and runtime checks. General localization remains open.
**Research links:** B2.d.2/O3.a, S3, S8, S9 and S16.

## 1. The canonical channels and the restricted preparation

Consider a fixed simple unit-conductance cycle. Let `L=I-W/2` be its normalized
Laplacian, `B=2L` its conductance Laplacian, `x` its real scalar EPI field and
`nu_i>0` its structural capacities. Keep a regular phase twist whose absolute
adjacent gap is strictly below both the effective U3 gate and `pi/2`, so the
neighbor phasor resultant points along the local phase. Its phase pressure is zero,
as shown in [Coupling winding persistence](COUPLING_WINDING_PERSISTENCE.md).
The cycle's uniform degree makes topology pressure zero as well.

The remaining canonical gradients are the neighbor mean differences already
implemented by [dnfr.py](../src/tnfr/dynamics/dnfr.py). Writing the effective
normalized channel coefficients as `e=w_epi>0`, `f=w_vf>=0`, the nodal equation
becomes

$$
\Delta\mathrm{NFR}=-eLx-fL\nu,\qquad
\dot x=\operatorname{diag}(\nu)\Delta\mathrm{NFR}.
$$

The full four-channel normalization remains in force even when two channel
values vanish; the EPI/capacity pair is not renormalized by itself. The
capacity profile and the phase twist are held fixed during this restricted EPI
flow. Clipping, changing support, an independent phase advance and intervening
capacity-changing operators are separate dynamics. No external potential or
new restoring coefficient has been introduced.

## 2. A shifted structural coordinate gives the exact balance

Set `r=f/e` and define

$$
y=x+r\nu,\qquad H=\operatorname{diag}(2/\nu_i).
$$

Fixed capacity gives the exact homogeneous equation

$$
\dot y=-e\operatorname{diag}(\nu)Ly.
$$

The capacity-pressure channel has been retained in the structural coordinate
`y`, not discarded. In particular, `sum_i H_i*x_i` is conserved, because
`H*diag(nu)*L=B` and the rows and columns of `B` sum to zero. If `x0` is the
initial EPI, let

$$
c=\frac{\sum_i H_i(x_{0i}+r\nu_i)}{\sum_i H_i},\qquad
x_i^*=c-r\nu_i.
$$

The connected cycle has only the constant null mode of `L`. Thus `x*` is the
unique zero-pressure profile with the conserved weighted EPI total selected
by `x0`. All capacities are positive, so rate stationarity and zero pressure
coincide in this fixed model. This implication changes at zero capacity.

Let `z=x-x*`. Its inherited weighted deviation energy satisfies

$$
V=\tfrac12z^\top Hz,\qquad
\dot V=-e z^\top Bz
=-e\sum_i(z_{i+1}-z_i)^2\leq0.
$$

Since `sum_i H_i*z_i=0`, the only zero-dissipation deviation is zero. More
explicitly, write `nu_min=min_i nu_i` and
`lambda2=1-cos(2*pi/n)`. Weighted mean minimization and the cycle Poincare
inequality give

$$
z^\top Bz\geq\nu_{\min}\lambda_2 z^\top Hz,
\qquad V(t)\leq e^{-2e\nu_{\min}\lambda_2t}V(0).
$$

This proves relaxation to the capacity-conditioned profile within the exact
held-field model, not attraction for general multichannel operator schedules.

A second diagnostic is the shifted Dirichlet energy

$$
F=\tfrac12y^\top By,\qquad
\dot F=-e(By)^\top\operatorname{diag}(\nu_i/2)(By)\leq0.
$$

The ordinary EPI Dirichlet energy `x^T*B*x/2` can increase: its gradient is
only one of the two pressure contributions. For example, with
`e=f=1/2`, `nu=(1/2,1,1,1)` and `x=(3/4,1/2,1/2,1/2)`, its derivative is
`1/16`, while both `V'` and `F'` are `-1/16`. This is structural redistribution
under the existing capacity channel, not a violation of the derived balance.

## 3. What a capacity dip produces, and what it does not establish

For initially uniform EPI `x0=b*1`, define the harmonic capacity
`nu_h=n/sum_i(1/nu_i)`. Then

$$
x_i^*=b+r(\nu_h-\nu_i),\qquad
x_i^*-x_j^*=-r(\nu_i-\nu_j).
$$

A capacity dip produces a higher equilibrium EPI at the same location; a
capacity peak produces a lower one. This contrast is fixed by the prepared
structural capacity, not by an independently fitted confinement law. With
uniform capacity the equilibrium is uniform, so an EPI bump alone disperses
under the same positive-capacity flow.

One canonical preparation makes this quantitative. Start with capacity one
and uniform EPI, then apply a single Silence operation at a selected node.
With the ideal repository formulas, its retained capacity is
`q=1-1/(4*pi)`, while the default channel ratio is `r=1/pi`. The subsequent
held-capacity equilibrium has core-minus-background contrast

$$
r(1-q)=\frac1{4\pi^2}\simeq0.0253303.
$$

This contrast is independent of cycle size; the common equilibrium level
depends on the harmonic capacity. Runtime observations instead use the exact
ratio of materialized normalized channel coefficients and the represented
Silence factor. They do not assert bitwise equality with the symbolic formula.

The selected node is a declared structural preparation. As in the existing
[pointed symmetry analysis](../src/tnfr/physics/pointed_symmetry.py), its location
breaks translation symmetry through that selector, not spontaneously. Once
prepared, the EPI contrast develops under the autonomous restricted nodal
flow without continuing external forcing. Its persistence is conditional on
retaining the capacity profile and the other hypotheses. It is not evidence
for an independently generated or universally stable localized entity.

Capacity acts through both nodal levers here. Multiplying a heterogeneous
capacity profile by `a` scales the EPI-pressure rate contribution by `a` and
the capacity-pressure contribution by `a^2`. It also changes the predicted
equilibrium contrast. It is therefore not merely a rescaling of time when
the capacity channel is active and the profile is nonuniform.

## 4. Canonical capacity updates generally release the static profile

The same canonical operators that prepared or synchronize the network can
change its conditional balance. Suppose `x=c-r*nu` initially and preserve
the regular phase twist and unit cycle support. Use target-only UM
(`UM_BIDIRECTIONAL=False`) without functional links
(`UM_FUNCTIONAL_LINKS=False`). Its immutable all-target stage with the
configured capacity synchronization factor has

$$
\nu^+=(I-sL)\nu,\qquad s=\texttt{UM\_vf\_sync}.
$$

UM leaves EPI unchanged. After the canonical pressure is refreshed,

$$
\Delta\mathrm{NFR}^+=f s L^2\nu.
$$

Similarly, uniform Silence attenuation `nu^+=q*nu` gives

$$
\Delta\mathrm{NFR}^+=f(1-q)L\nu.
$$

For a nonconstant capacity profile and positive `f`, these expressions are
generally nonzero. A direct UM pressure write or its local stabilization
telemetry must not replace this refreshed channel sum. The fixed-capacity
profile is not forward invariant under the default capacity-changing stages.
UM's capacity synchronization itself uses the same cycle diffusion map and
smooths the supporting capacity contrast under its own exact fixed-gate
hypotheses.

For continuously varying capacity, even the shifted coordinate carries an
additional term:

$$
\dot y=-e\operatorname{diag}(\nu)Ly+r\dot\nu.
$$

The fixed-capacity conservation law and Lyapunov metric cannot be transferred
without accounting for that forcing and the changing metric. This term is a
consequence of the coordinate definition, not a new physical force.

## 5. Diminishing capacity can retain contrast by exhausting the nodal clock

If capacity is spatially uniform during each EPI segment, its own pressure
gradient is zero. Let each segment have duration `h` and capacity
`nu_k=nu0*q^k`, as in repeated uniform canonical Silence attenuation with
`0<q<1`. The total capacity exposure is finite:

$$
\Theta_\infty=h\sum_{k\geq0}\nu_0q^k=\frac{h\nu_0}{1-q}.
$$

Exact continuous diffusion on these piecewise constant segments multiplies
a Laplacian mode of eigenvalue `lambda>0` by
`exp(-e*lambda*Theta_infinity)>0`. Nonzero initial contrast therefore need
not vanish. Its rate tends to zero because capacity tends to zero; its
pressure can remain nonzero. This is retention through a finite accumulated
reorganization budget, not a restoring mechanism or spontaneous confinement.
The formula concerns the exact segmented flow. A finite Euler trajectory
has its own product of modal factors and numerical error.

At exactly zero capacity, stationarity is weaker still. A node freezes even
when its pressure is nonzero, and `H_i=2/nu_i` is unavailable. If every
capacity vanishes, any EPI field is stationary although `-e*L*x` need not
vanish. With some frozen nodes, the active part instead has a Dirichlet
problem for `y`, with the frozen values as boundary data. Different frozen
values can support residual pressure at those nodes. The executable balance
in this note deliberately requires strictly positive capacity.

## 6. The common-walk condition is essential

The canonical implementation averages EPI with edge conductances but averages
capacity over the unweighted unique neighbor support. The two walks coincide
on the unit cycle considered here; they need not coincide on a weighted
graph. With distinct walks `L_E,L_0`, the pressure is
`-e*L_E*x-f*L_0*nu`, and substituting `x=c-r*nu` leaves
`f*(L_E-L_0)*nu` rather than zero.

Even regular weighted strength is insufficient: a four-cycle with alternating
edge weights `1,3`, capacity `(1,2,3,4)`, `x=5-nu` and `e=f=1/2` has residual
pressure `(-1/4,-1/4,1/4,1/4)`. For a more general graph, the weighted total
can drift as well. On a three-node path with edge weights `1,3`, strengths
`d_E=(1,4,3)` and capacity `(1,2,4)`,
`d_E^T*L_0*nu=3`; consequently the derivative of
`sum_i (d_Ei/nu_i)*x_i` is `-3*f`. No stationary unclipped two-channel state
exists for that fixed forcing when `f>0`. These controls prevent a false
extension of the cycle theorem.

## 7. Executable evidence and the next structural extension

[`capacity_localization.py`](../src/tnfr/physics/capacity_localization.py)
observes exact rational supplied cycle coordinates, the shifted equilibrium,
pressure/rate, weighted conservation and both energy balances. It uses the
shared exact-or-represented real reader and the existing full-channel default
normalization. General positive explicit coefficient pairs are formal
algebraic inputs; actual normalized canonical coefficients and live graph
conditions require separate evidence. No clipping interval, observed phase
loop or operator history is inferred from the detached vectors.

[Exact tests](../tests/physics/test_capacity_localization.py) cover pressure
cancellation, capacity dips, uniform-capacity dispersion, the two-lever
scaling, and EPI-energy growth with decreasing shifted energies.
[Boundary tests](../tests/physics/test_capacity_pressure_boundaries.py) exercise
the weighted-walk mismatch and actual UM/Silence refresh effects.
[Clock tests](../tests/physics/test_capacity_clock_retention.py) connect actual
uniform Silence attenuation to the finite-exposure obstruction using existing
rational exponential bounds.
[Runtime tests](../tests/physics/test_capacity_localization_runtime.py)
and the [benchmark](../benchmarks/capacity_localization.py) retain the actual
canonical preparation, pressure-refreshed solver partitions and finite
residuals. These are distinct from a future binary64 or complete-runtime
stability theorem.

The zero-winding control shares the exact prepared two-channel model; its
binary64 endpoints need not equal those of represented winding phases. The
runtime test accounts for their difference with the captured remaining-channel
pressure defect `delta_p` and Euler residual `epsilon`. For each recorded step,
`d=h*diag(nu)*delta_p+epsilon`. The common matrix
`M=I-h*e*diag(nu)*L` has nonnegative entries and unit row sums on these admitted
steps, so the exact discrepancy satisfies
`Delta_next=M*Delta+d_zero-d_winding` and
`||Delta_next||_inf <= ||Delta||_inf+||d_zero-d_winding||_inf`.
This is a finite bound derived from the retained residuals, not an arbitrary
equality tolerance or a prediction of unobserved numerical errors.

The change of coordinate also reuses the existing
[heterogeneous diffusion geometry](../src/tnfr/physics/structural_diffusion.py):
the reference coordinate `y` has effective capacity `e*nu`. Its exact-model
convergence and Dirichlet identities therefore need no new transport law;
identifying a particular materialized runtime generator remains separate.

A further identity suggests how to study localized phase structure without
introducing another primitive. Write `phi_i=2*pi*W*i/n+p_i` with a periodic
phase perturbation `p` and a consistent lift. Require every lifted gap
`2*pi*W/n+p_{i+1}-p_i` to have absolute value strictly below both the effective
U3 gate and `pi/2`. The canonical phase channel is then `-L*p/pi`.
If both this derived coordinate and capacity are
held fixed, the combined pressure is
`-L*(e*x+f*nu+(w_phase/pi)*p)`. The corresponding shifted coordinate is
`x+(f/e)*nu+(w_phase/(pi*e))*p`. A target-only all-node UM stage without
functional links on this fixed cycle evolves `p` with its phase gap
coefficient and evolves `nu` with its separate capacity
coefficient, so this more general balance is likewise conditional on what
structural fields are retained. That algebraic extension is developed in the
joint support/release study linked below; it is not implemented by the
capacity-only observer or a claim of spontaneous physical localization.
The separate [joint cycle observer and proof](CYCLE_SUPPORT_DYNAMICS.md)
now use the normalized coordinate `a=p/pi`, derive the reset/flow energy
budget and distinguish default-factor finite-clock retention from an active
zero-pressure equilibrium. Both observers reuse one private exact cycle
algebra; their hypotheses and public read-outs remain separate.
Neither note opens a new cycle campaign; the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns resumption.
