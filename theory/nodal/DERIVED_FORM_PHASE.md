# Derived form phase and inherited closure

Directed diffusion, sufficient amplitude/phase observations, wave-coordinate scope and regular continuation.

Section numbers are stable locators across this document family.
The [parameter reference](../NODAL_PARAMETER_FOUNDATIONS.md) owns the
reading map; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns active tasks. Each result retains its stated model and scope.

## 21. Derived form phase, wave coordinates and genuine continuation

A phase observed in form, independently stored primitive phase, and a phase
law chosen to realize a desired motion are different objects. The following
reuse of transport, memory and source geometry separates them. No result
below establishes sustained identity or physical particle emergence.

### Variable contract for the derived form family

This contract instantiates the
[parameter properties](../NODAL_PARAMETER_FOUNDATIONS.md#variable-properties-and-unresolved-foundations)
for the directed-triangle family. It does not redefine the engine's primitive
variables. Use form unit `X`, declared clock unit `T`, and the orthonormal
regional frames displayed below.

| Quantity | Type, domain and units | Status and necessary boundary |
| --- | --- | --- |
| Fine EPI `x_i` | Signed real form coordinate, units `X` | Evolving fine state; zero is a coordinate value, not deletion of a node or absence of all structure |
| Fine capacity `nu_i` | Positive real, units `T^-1` in this family | Supplied and held, with the stated regional-constancy conditions for Gram closure; no new capacity law is derived |
| Fine pressure `p_i=P_i(x)` | Real evaluated response, units `X` | Pure-EPI `-L_rw*x` on full support; evaluated before the predicted rate, never reconstructed from a measured derivative |
| Primitive phase `theta_i` | Circle coordinate, radians | Held at zero in the controls and unused by this law; it is not the rotating form orientation below |
| Support, weights and `t` | Declared oriented graph, nonnegative conductance, common ordered clock | Held premises; row normalization uses the full graph. Absolute conductance scale cancels from this pressure; neither laboratory time nor metric distance is inferred |
| Mean `mu_a` and contrast `z_a=u_a+i*v_a` | Real mean and two real contrast coordinates, units `X` | Derived from three fine values in a fixed frame; together they reconstruct the region. Complex notation adds no complex fine EPI |
| Amplitude `r_a` and orientation `psi_a` | `r_a=abs(z_a)>=0`, units `X`; circular `psi_a=arg(z_a)` where `r_a>0` | Derived, frame dependent; `psi_a` is unavailable at zero amplitude while Cartesian evolution remains defined. Amplitude alone is not complete form |
| Joint Gram `H=zz^dagger` | Hermitian positive semidefinite, rank at most one, units `X^2` | Derived constrained observation; discards common contrast orientation. Its closure depends on the interface and capacity domain |
| Cross relation `K_AB=z_A*z_B^dagger` | Complex matrix, units `X^2`, `K_BA=K_AB^dagger` | A block of the same joint Gram, not independent freely selectable state or an extra force; joint and zero-contrast constraints are given below |

Any pressure/capacity factorization of an effective rate requires its own
normalization and inheritance evidence. Neither `psi_dot` nor the angular
mobility from a polar metric can silently replace primitive `nu_f`. The
[phase distinctions](PHASE_FORM_EXCHANGE.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean)
separate primitive phase, form orientation and a reporting-frame angle.
The regional partition and its frames are supplied; this construction does
not derive their autonomous selection. Physical identification of these
collective quantities remains open and need not map each primitive to a sensor.

Notation is local to each owner: this `H` is a Gram observation, not the
weighted transport metric also called `H` elsewhere. This `K_AB` is a
cross relation, not tetrad curvature `K_phi`, a path-potential kernel or an
eliminated-state memory kernel. Equal symbols do not identify these objects.

<a id="regional-form-engine-integration"></a>
### Shared engine observation of the form coordinates

The [form geometry owner](../../src/tnfr/physics/form_geometry.py) now exposes
these coordinates and their instantaneous nodal pushforward through
`observe_regional_form(graph, regions)` and the SDK's
`Network.regional_form(regions)`. The ordered triples are supplied reporting
frames. Observation itself requires no directed-cycle topology, common
capacity or pure-EPI pressure; those are additional premises of the
autonomous reductions below.

For each triple, it retains the rational raw contrasts
`a=x_0-x_1`, `b=x_0+x_1-2*x_2`, representing
`z=a/sqrt(2)+i*b/sqrt(6)`, and the mean. Squared amplitude is exactly
`a^2/2+b^2/6` for the materialized fine inputs. The full Gram retains
`Re(Q_ab)=a_a*a_b/2+b_a*b_b/6` and
`sqrt(12)*Im(Q_ab)=b_a*a_b-a_a*b_b`. The corresponding derivatives project
the actual shared rounded nodal product with its retained arithmetic defect.
No derivative is inferred from an assigned phase or a pressure fitted to a
later response. Numerical polar estimates remain separate from this exact
algebra; zero contrast makes angle and polar rates unavailable without
erasing the Cartesian or Gram rate.

This is a read-only integration of the derived observation, not a new
constitutive pressure or replacement primitive phase law. Full-state closure,
future evolution, source provenance and hypothesis admission remain separate.
The [production controls](../../tests/physics/test_form_geometry.py) bind the
observation to the shared nodal integrator and its finite Euler arithmetic.
The [SDK guide](../../docs/CLI_AND_SDK.md#observe-regional-form-and-its-nodal-response)
owns usage; the [pressure admission result](PRESSURE_CONSTITUTIVE_SCOPE.md#regular-derived-phase-pressure)
explains why the observed angle cannot automatically become a global source.

### 21.1 Directed form rotation and exact eliminated-state memory

Take outgoing unit support `i -> i+1 (mod 3)`, common fixed capacity `nu>0`,
and pure-EPI pressure. For the existing orthonormal basis
`U=[(1,-1,0)/sqrt(2), (1,1,-2)/sqrt(6)]`, write the two real form coordinates
as `z=u+i*v`. The outgoing Laplacian and exact Cayley owner give

\[
\dot z=\nu\left(-\frac32-i\frac{\sqrt3}{2}\right)z,\qquad
\dot{\arg z}=-\frac{\sqrt3\nu}{2},\qquad
\frac{d}{dt}|z|^2=-3\nu|z|^2.
\]

The angle requires `z!=0`; the Cartesian law is defined everywhere.
Reversing the edges reverses the rotation; reciprocal symmetrization removes
it while preserving contrast decay. Directed support supplies orientation,
so the form-only reflection obstruction in [section 17](PRIMITIVE_PHASE_CLOSURE.md#17-primitive-phase-origin-symmetry-retained-state-and-the-missing-row) is not contradicted.
No frequency is fitted, but the origin of that support remains a premise.
The radius contracts by `exp(-2*pi*sqrt(3))` per turn, independently of
capacity. This is transient rotation, not a maintained pattern.

Put `a=3*nu/2`, `b=sqrt(3)*nu/2`. Eliminating the second coordinate gives

\[
\dot u=-au+be^{-at}v(0)-b^2\int_0^t e^{-a(t-s)}u(s)\,ds,
\qquad \ddot u+3\nu\dot u+3\nu^2u=0.
\]

Both the negative memory kernel and its hidden initial source are derived.
A second-order observable equation thus follows from first-order nodal
evolution without added microscopic inertia. The general elimination algebra
is reusable; `epi_memory`'s reversible graph adapter is not extended to this
directed example. Actual fresh-pressure controls give different `u_dot` for
equal observed `u` and different hidden `v`, while primitive phase stays zero.
Controls: [directed form phase and memory](../../tests/physics/test_directed_internal_phase_memory.py).

### 21.2 A coupled amplitude and phase law derived from fine diffusion

Join two equally oriented unit directed triangles by reciprocal unit links
between corresponding vertices. Each node has two outgoing neighbors. This
declared support differs from the
[reciprocal-prism model](INHERITED_FORM_DYNAMICS.md#12-intrinsic-response-from-a-closed-fine-nodal-model).
With pure-EPI pressure and common fixed capacity `nu`, the exact complete
scalar-state coordinates are two means and two complex contrasts:

\[
\dot\mu_a=\frac\nu2(\mu_b-\mu_a),\qquad
\dot z_a=\left(-\frac{5\nu}{4}-i\frac{\sqrt3\nu}{4}\right)z_a
                  +\frac\nu2 z_b.
\]

For `z_a=r_a exp(i psi_a)` with positive amplitudes, these imply

\[
\dot r_a=-\frac{5\nu}{4}r_a+\frac\nu2r_b\cos(\psi_b-\psi_a),\qquad
\dot\psi_a=-\frac{\sqrt3\nu}{4}
 +\frac\nu2\frac{r_b}{r_a}\sin(\psi_b-\psi_a).
\]

The sine interaction, its coefficient and amplitude ratio follow from the
fine generator; no synchronization gain is supplied. For
`delta=psi_1-psi_0`,
`delta_dot=-(nu/2)*(r_0/r_1+r_1/r_0)*sin(delta)`.
Equal amplitudes form an invariant restricted family with
`delta_dot=-nu*sin(delta)`. Outside it, equal angles can have different
future rates. Angles alone are not sufficient state; at zero amplitude the
Cartesian coordinates remain the continuation variables.

The complete contrast budget is

\[
\frac{d}{dt}(|z_0|^2+|z_1|^2)
 =-\frac{3\nu}{2}(|z_0|^2+|z_1|^2)-\nu|z_0-z_1|^2.
\]

Derived interaction, synchronization and rotation therefore coexist with
strict loss. This is a constructive emergent **observable** phase law.
Identifying it with primitive phase or canonical phasor-argument pressure
still requires full vector-field matching; a sine term alone is insufficient.
Controls: [coupled reduction and phase-only obstruction](../../tests/physics/test_coupled_directed_form_phase.py).

#### Inherited observation identity and source work

The fine model above now admits an exact observation-specific closure, rather
than only a phase-only counterexample. Retain the same supplied directed
support, fixed common positive capacity, orthonormal regional frames and
pure-EPI law. Primitive phase is held and unused, Gamma is absent, and no
operator event, clipping or numerical time step enters this continuous result.
The selected observations are regional contrast squares, relative orientation
and their cross-region contributions to the contrast budget. They are not the
complete fine state or its tetrad.

On the positive-amplitude chart, put `q=r_1/r_0`, `delta=psi_1-psi_0`.
The polar equations already derived from the fine generator give

\[
\dot q=\frac{\nu}{2}(1-q^2)\cos\delta,\qquad
\dot\delta=-\frac{\nu}{2}(q+q^{-1})\sin\delta.
\]

Thus the two angles **together with** the amplitude ratio have a closed
evolution while both amplitudes stay positive. An absolute amplitude is
additionally needed to predict form magnitude or its quadratic budget:
common scaling preserves all angles and `q`, but multiplies that budget by
the scale squared. Equal amplitudes supply a restricted invariant family,
not a reason to discard the ratio for arbitrary preparations.

A division-free description also covers zero regional amplitude. Define

\[
I_a=|z_a|^2,\qquad c=\operatorname{Re}(\overline z_0z_1),\qquad
s=\operatorname{Im}(\overline z_0z_1).
\]

These quantities obey `I_0,I_1>=0` and `c^2+s^2=I_0 I_1`. Direct
differentiation of the inherited Cartesian law gives the closed system

\[
\begin{aligned}
\dot I_0&=-\frac{5\nu}{2}I_0+\nu c,&
\dot I_1&=-\frac{5\nu}{2}I_1+\nu c,\\
\dot c&=-\frac{5\nu}{2}c+\frac\nu2(I_0+I_1),&
\dot s&=-\frac{5\nu}{2}s.
\end{aligned}
\]

The common internal rotation cancels in this observation. The constraint
defect has derivative `-5*nu*(c^2+s^2-I_0*I_1)`; realizable initial data
remain realizable under the inherited linear Cartesian flow. Conversely, any
such data with `I_0>0` reconstruct a representative
`z_0=sqrt(I_0)`, `z_1=(c+i*s)/sqrt(I_0)`. If `I_0=0<I_1`, take
`z_0=0`, `z_1=sqrt(I_1)`; at total zero both vanish. The common orientation
is discarded. This proves sufficiency for the selected observations without
claiming a globally unique minimal state for TNFR.

Write `E=I_0+I_1`, `D=I_0-I_1`. Its exact finite-time modes are

\[
\begin{aligned}
(E+2c)(t)&=e^{-3\nu t/2}(E+2c)(0),\\
(E-2c)(t)&=e^{-7\nu t/2}(E-2c)(0),\\
D(t)&=e^{-5\nu t/2}D(0),\qquad
s(t)=e^{-5\nu t/2}s(0).
\end{aligned}
\]

Here `nu*c` is the cross-region contribution to each `dI_a/dt` under
the displayed split. This is an inherited **contrast-square budget**, not
laboratory energy, primitive unit-phasor resultant amplitude, external Gamma
or a separately supplied sustaining source. Positive contributions at both
regions do not imply conservation of that contribution: the complete balance
still satisfies `E_dot=-(3*nu/2)*E-nu*|z_0-z_1|^2`. The model cannot
maintain nonzero contrast indefinitely. At `z_0=0`, `z_1!=0`, it nevertheless
has `I_0_dot=0` and `I_0_ddot=nu^2*I_1/2>0`: local contrast can start
growing by transfer while total contrast decreases. No angle or event is
invented at the zero-amplitude boundary.

**Same angular history, different radial response.** With both fine regional
means initialized to `1/2`, compare real positive contrasts
`(z_0,z_1)=(5,5)/128` and `(1,7)/128`. Both have
`E=25/8192` and relative phase zero. Their continuously lifted observed phases
follow the identical law `psi_a(t)=-sqrt(3)*nu*t/4` for all `t>=0` (modulo
`2*pi` for circular phases): after removing this common rotation, the real
linear coupling preserves positive amplitudes.
Their initial `c` values are `25/16384` and `7/16384`, however, so their
initial `E_dot` values are respectively `-75*nu/16384` and
`-111*nu/16384`. Even the full angular trace and present total contrast do
not identify its radial response. These are two bounded fine preparations,
not fitted trajectories or a new pressure recipe.

The retained information depends on the promised observation:

| Prediction | Sufficient inherited information | Boundary |
| --- | --- | --- |
| Angular evolution | Both angles and `q` | Positive-amplitude chart only; no absolute form scale |
| Total contrast budget | `E,c` | Closed two-coordinate system; no regional allocation or orientation |
| Regional contrast, relative angle and cross-region budget | `I_0,I_1,c,s` on the constraint | Relative angle is available only when both amplitudes are nonzero |
| Fine EPI, pressure and general field observations | Both means and both Cartesian contrasts | The common contrast rotation is not generally a node relabeling or a tetrad-preserving symmetry |

This result supplies a sufficient inherited observation and an exact
missing-information obstruction. It does **not** select the primitive
Arg source over `J_phi/pi`, or conversely. That identification would still
need the independently specified observation map and full radial, angular
and mean vector-field matching from [section 15](PHASE_FORM_EXCHANGE.md#15-when-an-observed-phase-can-be-a-causal-source).
Feeding an angle read from form back as an extra pressure can change the
fine law; the existing source-closure control already checks that boundary.
The [coupled-reduction controls](../../tests/physics/test_coupled_directed_form_phase.py)
derive these identities from the fine graph generator and check the finite
modal evolution, information-loss witnesses and zero-amplitude continuation.
They introduce no independent solver or parameter sweep.

<a id="derived-phase-identification-admission"></a>
#### Admitting a form-derived angle into the primitive pressure law

This is the first F2 admission control following the
[S/D state cards](../FUNDAMENTAL_THEORY.md#reference-s-d-admission-verdict).
Keep the two equally oriented directed triangles and matched reciprocal links
above, common held capacity `nu>0`, fixed unit conductance, and aligned supplied
regional frames. Set the EPI and phase pressure coefficients to `e,w>0`;
capacity/degree source readings vanish here. No events, Gamma or clipping
are included. Consider the additional identification
`theta_ai=psi_a=arg(z_a)` at positive regional amplitudes. It is a hypothesis
to test, not a consequence of the nodal equation or of observing an angle.

For `delta=psi_B-psi_A` in the regular branch `abs(delta)<pi`, each node sees
one own-region phase and one other-region phase. The centered resultant is
`2*cos(delta/2)>0`, so the actual Arg-based pressure gives

\[
g_{\phi,A}=\frac{\delta}{2\pi},\qquad
g_{\phi,B}=-\frac{\delta}{2\pi}.
\]

For `delta!=0`, this source is nonzero but constant within each region. Its projection on
each zero-mean contrast plane vanishes, whereas the complete mean rows are

\[
\dot\mu_A=\frac{e\nu}{2}(\mu_B-\mu_A)+\frac{\nu w\delta}{2\pi},\qquad
\dot\mu_B=\frac{e\nu}{2}(\mu_A-\mu_B)-\frac{\nu w\delta}{2\pi}.
\]

The contrast row remains

\[
\dot z_A=e\nu\left[\left(-\frac54-i\frac{\sqrt3}{4}\right)z_A
                      +\frac12z_B\right],
\]

with A/B exchanged for the other region. Thus the original **contrast**
reduction survives this source, but the full pure-EPI law does not: its mean
response changed. The previously derived budget becomes
`E_dot=-(3*e*nu/2)*E-e*nu*abs(z_A-z_B)^2`; adding this source does not maintain
internal contrast. This specializes the existing
[source-closure distinction](../../tests/physics/test_internal_mode_source_closure.py)
to the directed S/D reference instead of treating mean and contrast matching
as interchangeable.

**Tangency is a separate obligation.** If a prospective pressure P and an
independent phase law Omega are supplied, the constraint `theta=Psi(x)`
can be invariant only if, on that constraint,

\[
\Omega(x,\Psi(x))=D\Psi(x)\,N P(x,\Psi(x)).
\]

Matching an earlier fine model additionally requires its complete form row
to agree. Defining Omega by this identity constructs a conditional constrained
model; it does not independently justify the identification. Here direct
differentiation of the inherited Cartesian row gives, with `q=r_B/r_A>0`,

\[
\dot\psi_A=-\frac{\sqrt3 e\nu}{4}+\frac{e\nu q}{2}\sin\delta,\qquad
\dot\psi_B=-\frac{\sqrt3 e\nu}{4}-\frac{e\nu}{2q}\sin\delta,
\qquad
\dot\delta_\psi=-\frac{e\nu}{2}(q+q^{-1})\sin\delta.
\]

The existing [phase proposal owner](../../src/tnfr/dynamics/phase_evolution.py)
reads supplied frequency, primitive phase and support, not EPI. With common
frequency `nu` and both neighbors admitted, its continuous relative row is
`delta_theta_dot=-k*sin(delta)`, where k is the declared coupling strength.
Two states with the same phase/capacity/support therefore receive identical
proposals even when their form amplitude ratios differ. For
`e=w=1/2, nu=1, delta=pi/6`, ratios `q=1,2` require respectively
`delta_psi_dot=-1/4,-5/16`. No fixed k matches both; a common moving frame
cannot remove a relative-rate difference. This is an admission obstruction,
not an error in the independently declared phase writer.

There is a restricted positive bridge. The contrast rows imply
`q_dot=(e*nu/2)*(1-q^2)*cos(delta)`, so `q=1` is invariant. On that manifold,
the relative rows coincide for declared `k=e*nu`. The primitive and derived
common angular speeds still differ by `nu+sqrt(3)*e*nu/4`; equality requires
an explicit moving common-phase frame, or comparison modulo common phase.
With a fixed effective gate `0<gamma<=pi/2` and `abs(delta)<gamma`, the relative
flow remains admitted and `r_dot=e*nu*(-5/4+cos(delta)/2)*r` preserves positive amplitude
at finite times while dissipating it. This is a conditional continuous-law
bridge with changed mean dynamics, not proof that finite Euler proposals
preserve it exactly, a derivation of k, or a maintenance mechanism.

**Zero and frame boundaries.** The original Cartesian transport extends
through zero contrast; the proposed angle-fed pressure generally does not
have a unique continuous extension as a function of form alone. Fix `z_B>0`
real and compare `z_A=epsilon` with `z_A=epsilon*exp(-i*pi/6)`. As
`epsilon -> 0+` their fine forms approach the same state, but their phase
source contributions to `mu_A_dot` approach respectively zero and
`nu*w/12`. Both paths remain within the controls' fixed gate `pi/2`. An arbitrary
angle assignment at zero cannot repair these incompatible limits. Retaining
primitive phase, retaining justified history, or changing the source law
would define different model choices. This excludes the proposed global
continuous form-only law, not every possible derived-phase law.

Regional frame rotation also changes the reported angle. Identifying it with
primitive phase requires a transported frame convention; a node relabeling
cannot silently reset that convention. These supplied data remain premises.
The [two admission controls](../../tests/physics/test_derived_phase_source_admission.py)
derive the complete rows from the existing directed generator and compare
them with graph-facing fresh pressure and actual phase proposals. They check
instantaneous represented residuals, not a newly reserved trajectory or
physical identification.

#### Exact capacity domain of the inherited observation

Keep the same supplied six-node support, pure-EPI outgoing pressure and
fixed positive capacities, but allow all six capacities to differ. Regional
frames are unchanged, and both means can vary independently of the contrasts.
The observation `(I_0,I_1,c,s)` has an autonomous evolution **for every fine
state if and only if capacities are constant within each triangle**. The two
regional constants `nu_0,nu_1` need not equal one another. This statement is
about the specified observation and all preparations; it does not exclude
additional special invariant preparations outside this capacity family.

For necessity, let `U` be the two-column orthonormal zero-mean basis of one
triangle and let `n_a` be its three capacities. Changing only the hidden
regional mean difference by `d` changes the unscaled fine diffusion rates by
opposite constants `d/2` in the two triangles. After capacity multiplication,
the internal rates change by `+/- (d/2)*U^T*n_a`. Independence of each
`I_a_dot=2*y_a^T*y_a_dot` from this hidden mean for **every** real contrast
`y_a` requires `U^T*n_a=0`. The kernel of `U^T` consists exactly of constant
vectors. This argument also works on an arbitrarily small open neighborhood
of a uniform fine form inside its admissible band; it needs no unbounded EPI.

For sufficiency, put the constant capacity `nu_a` on each triangle. The shared
fine generator then gives the closed mean and complex-contrast rows

\[
\dot\mu_a=\frac{\nu_a}{2}(\mu_b-\mu_a),\qquad
\dot z_a=\nu_a\left(-\frac54-i\frac{\sqrt3}{4}\right)z_a
             +\frac{\nu_a}{2}z_b.
\]

Writing `w=c+i*s=conj(z_0)*z_1` gives

\[
\dot I_a=-\frac52\nu_a I_a+\nu_a c,\qquad
\dot w=\left[-\frac54(\nu_0+\nu_1)
       +i\frac{\sqrt3}{4}(\nu_0-\nu_1)\right]w
       +\frac{\nu_0 I_1+\nu_1 I_0}{2}.
\]

Regional frequency detuning is therefore inherited from these capacities;
it does not destroy this observation's closure. The realizability constraint
is preserved by the Cartesian flow. The appropriate fixed positive metric
weights contrast by inverse capacity:

\[
\frac{d}{dt}\left(\frac{I_0}{\nu_0}+\frac{I_1}{\nu_1}\right)
 =-\frac32(I_0+I_1)-|z_0-z_1|^2.
\]

Including both means adds `-3*(mu_0-mu_1)^2` to the derivative of
`sum_a((3*mu_a^2+I_a)/nu_a)`. These are exact fixed-model quadratic budgets,
not a capacity evolution law or physical energy identification. In particular,
allowing two different regional capacities supplies no sustained nonzero
contrast in this passive model.

There is also an independent loss of orientation outside the admitted family.
Set capacities to `(1+epsilon,1,1,1,1,1)`, `epsilon>0`, with both means `1/2`.
Preparations `(z_0,z_1)=(r,r)` and `(i*r,i*r)` have identical Gram observations
and means, but their unweighted total-contrast rates are
`-(3+epsilon)*r^2` and `-3*r^2`. At `epsilon=1`, `r=1/16`, the difference
is exactly `-1/256`. Thus retaining means alone cannot repair the reduction:
the discarded common modal orientation can become relevant. This sharp domain
result goes beyond the [previous isolated hidden-mean obstruction](INHERITED_FORM_DYNAMICS.md#144-capacity-and-causal-closure-remain-explicit-dependencies).

The existing [coupled-reduction controls](../../tests/physics/test_coupled_directed_form_phase.py)
verify the exact coefficient kernel, both inherited laws, the weighted budgets
and the orientation witness. A production pressure control uses fine EPI
`(10,6,8,13,9,8)/16`, capacities `(3/2,3/2,3/2,5/4,5/4,5/4)`, zero primitive
phases and Gamma absent. Its refreshed binary64 pressures and nodal products
equal the rational fine-generator values exactly; the weighted contrast and
full-form rates are `-39/256` and `-51/256`. This binds one represented case
to the exact family without introducing a solver or an asymptotic runtime claim.

#### Unequal-capacity mode selection and a finite phase-only discriminator

This continues the already admitted unequal-regional-capacity family, rather
than introducing an oscillator or a pressure law. Keep the supplied equally
oriented directed unit triangles, reciprocal unit inter-region links, fixed
positive capacities `a=nu_0`, `b=nu_1`, and **pure-EPI pressure**. Primitive
phase is held and unused, Gamma is absent, and there are no operator events.
Unlike the common-capacity episode below, unequal capacities generally activate
capacity pressure in a full mixture; this result does not include that mixture.

**Candidate and identity.** The full model retains both regional means and
complex contrasts. Its proposed observed identity is the contrast direction
up to a common nonzero complex scale, together with the separately retained
absolute contrast budget. This is an observation-specific mode identity, not
a sufficient description of the complete fine EPI, tetrad or a physical NFR.
The initial family is every finite fine scalar state with nonzero contrast;
small common scaling about a uniform mean places these preparations inside
an admitted bounded chart. The conclusion below classifies this whole family,
including its exceptional preparations and undefined-angle events.

Set

\[
c_0=-\frac54-i\frac{\sqrt3}{4},\quad D=a-b,\quad S=a+b,\quad
A=\begin{pmatrix}a c_0&a/2\\b/2&b c_0\end{pmatrix}.
\]

The inherited row is `z_dot=A*z`; `c_0` here is a matrix coefficient, not the
previous real cross-correlation. With the positive-real-part square root,

\[
\Delta=\sqrt{ab+c_0^2D^2},\qquad
\lambda_\pm=\frac{c_0 S\pm\Delta}{2},\qquad
R_\pm=\frac{-c_0D\pm\Delta}{a}.
\]

These eigenvectors are `(1,R_plus)` and `(1,R_minus)`; they are not assumed
orthogonal. Since `c_0^2=11/8+i*5*sqrt(3)/8`, the radicand has strictly positive
real part. Thus `Re(Delta)>0`, the eigenvalues are distinct for all `a,b>0`,
and `lambda_plus` has the larger real part. The roots are nonzero and obey
`R_plus*R_minus=-b/a`. For initial contrasts `(z_0,z_1)`, set

\[
A_+=\frac{z_1-R_-z_0}{R_+-R_-},\qquad
A_-=\frac{R_+z_0-z_1}{R_+-R_-}.
\]

The exact Cartesian solution is

\[
z(t)=A_+\binom{1}{R_+}e^{\lambda_+t}
     +A_-\binom{1}{R_-}e^{\lambda_-t}.
\]

For `A_plus!=0`, write `eta=A_minus/A_plus`. Wherever `z_0(t)!=0`,

\[
\frac{z_1(t)}{z_0(t)}=
\frac{R_++\eta R_-e^{-\Delta t}}{1+\eta e^{-\Delta t}}
\longrightarrow R_+.
\]

Therefore the generic limiting relative phase is `arg(R_plus)`, the amplitude
ratio tends to `abs(R_plus)`, and both observed phase rates tend to
`Im(lambda_plus)`. Projective errors are `O(exp(-Re(Delta)*t))` as `t` tends
to infinity; a pure slow-mode preparation has zero error throughout.
The nonzero exceptional preparations `A_plus=0` remain on the fast eigenline
with ratio `R_minus`. Total zero has no observed phase. Arbitrarily small
`A_plus` can delay approach arbitrarily, so the theorem supplies no uniform
finite locking time across all preparations.

There is a sharp geometric lag bound. Put
`h=-c_0`, `d=(a-b)/sqrt(ab)`. The principal inverse hyperbolic sine gives

\[
R_+=\sqrt{b/a}\,e^{\operatorname{asinh}(hd)},\qquad
\delta_+=\operatorname{Im}\operatorname{asinh}(hd),\qquad
|\delta_+|<\arctan\frac{\sqrt3}{5}\simeq0.3334731723.
\]

For `d>0`, the argument of `sqrt(1+h^2*d^2)` is strictly between zero and
`arg(h)`. Consequently `Im(h/sqrt(1+h^2*d^2))>0`: the lag increases strictly
with `d`, hence with the capacity ratio. Oddness treats `d<0`; the large-`d`
limit `asinh(h*d)~log(2*h*d)` gives the strict bound and its limiting sharpness.
Common scaling of both capacities changes rates, not `R_plus`. Exchanging
capacities inverts that ratio and reverses the limiting lag. At `a=b`, the
two ratios are `+1` and `-1`, recovering the existing equal-capacity modes.

**No positive-capacity locking transition or sustained contrast follows.**
The real-part gap never closes in this family, and the generic lag varies
smoothly with positive capacity ratio. The existing budget gives

\[
V=\frac{|z_0|^2}{a}+\frac{|z_1|^2}{b},\qquad
\dot V=-\frac32(|z_0|^2+|z_1|^2)-|z_0-z_1|^2
\le-\frac32\min(a,b)V.
\]

This is phase locking of a decaying contrast. At its limiting zero state,
the angles are undefined; finite measurement resolution or binary64 rounding
can lose them earlier. Zero capacity is an excluded boundary. No physical
critical point, maintained oscillator or autonomous formation is inferred.

**Zero crossings are regular in the full state.** Each regional contrast in
a nonzero two-mode solution can have at most one zero for `t>=0`: cancellation
requires a fixed modulus of `exp(-Delta*t)`, which is strictly decreasing.
The two contrasts cannot vanish simultaneously unless the contrast solution is
identically zero. At `z_0=0`, `z_0_dot=a*z_1/2!=0`, and conversely for the other region.
These are simple chart crossings. Continue the Cartesian nodal state, rather
than assigning a phase at zero or inventing an operator event.

**A nonredundant finite prediction.** On the positive-amplitude chart set
`q=r_1/r_0`, `delta=psi_1-psi_0`. The full inherited equations are

\[
\dot q=\frac54Dq+\frac12(b-aq^2)\cos\delta,\qquad
\dot\delta=\frac{\sqrt3}{4}D-\frac12(b/q+aq)\sin\delta.
\]

Compare with the explicitly declared approximation that freezes `q=1` and
keeps only its angular row. This is not another canonical law. If `D!=0`,
`q_dot` at `q=1` is `D*(5/4-cos(delta)/2)`, always nonzero, so initially equal
amplitudes do not justify keeping that ratio fixed. At `q=1,delta=pi/2`, both
models have the same initial relative phase and relative angular velocity, but

\[
\ddot\delta_{\rm full}(0)=-\frac58D^2,\qquad
\ddot\delta_{\rm frozen\ ratio}(0)=0.
\]

Thus equal initialization and instantaneous relative-phase rates still fail to close
the subsequent angular response. This strengthens the earlier comparison of
different hidden amplitudes at the initial time: here the two models initially
agree on those amplitudes too, and their evolution creates the discrepancy.

**Bounded control, 2026-09-26.** The saved specification uses `(a,b)=(1,2)`,
`(2,1)` and the equal-capacity control `(1,1)`, means `(1/2,1/2)`, contrasts
`(1,i)/16`, and times `(0,0.02,0.1,0.5,2)` in the declared structural clock.
The fine six-node real generator comes from `structural_diffusion_operator`;
the detached shared matrix exponential is checked against the analytic two-mode
formula and an independent SciPy real-matrix propagation. The scalar comparator
uses its own exact half-angle Riccati realization. No coefficient is fitted.

| Capacities `(a,b)` | Full / frozen-ratio initial relative-phase acceleration | Full relative phase at `t=0.5` | Frozen-ratio phase at `t=0.5` | Generic limiting lag |
| --- | --- | ---: | ---: | ---: |
| `(1,2)` | `-0.625 / 0` | `0.643272` | `0.702739` | `-0.228726` |
| `(2,1)` | `-0.625 / 0` | `0.984147` | `1.069849` | `0.228726` |
| `(1,1)` | `0 / 0` | `1.090415` | `1.090415` | `0` |

Angles are radians; accelerations are radians per squared declared time unit.
Maximum complex-contrast discrepancy from the mode formula is below `2e-16`;
the independent fine-propagation difference is below `1.4e-15`. These are
finite numerical agreement checks, not rigorous enclosures or asymptotic
runtime certificates. The local bundle
`artifacts/research/phase_capacity_lock_2026_09_26/` retains the specification,
script hash, source/version provenance and complete results. The
[shared test owner](../../tests/physics/test_coupled_directed_form_phase.py)
contains the mathematical and engine-path controls. This is synthetic evidence;
physical measurement, primitive-phase identification and novelty relative to
external physics remain unestablished.

#### Reciprocity boundary for a physical realization

The directed candidate above is dissipative, but that does not make it a
passive reciprocal RC/thermal model. A first-order scalar realization
`dx/ds=-S^(-1)*K*x`, with positive diagonal storage `S` and symmetric
positive-semidefinite conductance Laplacian `K`, is similar to
`-S^(-1/2)*K*S^(-1/2)`. Its eigenvalues are real and nonpositive. This is
the reversible scope distinguished by the
[pressure foundation](PRESSURE_CONSTITUTIVE_SCOPE.md), not a statement about
every physical system that can dissipate energy.

For the coupled triangles, the complex contrast block `A` above instead has,
with `T=diag(sqrt(a),sqrt(b))`,

\[
T^{-1}AT=
\begin{pmatrix}-5a/4&\sqrt{ab}/2\\\sqrt{ab}/2&-5b/4\end{pmatrix}
-i\frac{\sqrt3}{4}\operatorname{diag}(a,b).
\]

The first matrix is Hermitian. Multiplying an eigenvector equation by its
conjugate transpose therefore gives, for every contrast eigenvalue,

\[
\operatorname{Im}\lambda=-\frac{\sqrt3}{4}
\frac{a|v_0|^2+b|v_1|^2}{|v_0|^2+|v_1|^2}<0.
\]

The real generator contains these eigenvalues and their conjugates. No
invertible linear state chart, node relabeling or positive constant clock
rescaling converts this complex spectrum into the reciprocal real spectrum.
This excludes that exact closed physical identification for every `a,b>0`;
it does not exclude directed transport, nonreciprocal devices or other models
with additional state. A finite resemblance of two traces is also weaker
than generator equivalence. The
[measurement contract](../research/PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md)
retains the required independent bridge and the unresolved source admission.

#### Regional appearance and decay under one unchanged law

The zero-amplitude continuation above gives a complete transient episode,
without requiring indefinite maintenance or another capacity law. Retain
the same connected support, common held capacity `kappa>0`, equal zero
primitive phases and absent Gamma. With any fixed positive four-channel
mixture, capacity and phase gradients vanish, as does topology pressure
because every node has two outgoing support neighbors. Put
`rho=kappa*w_epi`. The preceding pure-EPI formulas apply with `nu=rho`;
no active source channel has been removed or renormalized after observation.

Prepare both regional means at `m`, with `z_0(0)=0` and `z_1(0)=R>0`.
Region zero has no initial internal form contrast. The complete solution is

\[
z_{0,1}(t)=\frac R2e^{-i\sqrt3\rho t/4}
\left(e^{-3\rho t/4}\mp e^{-7\rho t/4}\right),\qquad
\mu_0(t)=\mu_1(t)=m.
\]

The minus sign belongs to region zero. Its inherited angle becomes defined
for `t>0`, rotating with region one's angle at `-sqrt(3)*rho/4`; no primitive
phase is written at the initial zero. With `I_env=R^2` and `y=exp(-rho*t)`,

\[
I_0(t)=\frac{I_{env}}4 y^{3/2}(1-y)^2,\qquad
I_1(t)=\frac{I_{env}}4 y^{3/2}(1+y)^2.
\]

`I_0` grows strictly until its unique maximum and then decreases strictly:

\[
t_* = \frac{\log(7/3)}\rho,\qquad
I_{max}=\frac{4I_{env}}{49}\left(\frac37\right)^{3/2}.
\]

Indeed its logarithmic derivative for `t>0` is
`rho*(-3/2+2*y/(1-y))`, which changes sign exactly at `y=3/7`.
The existing budget `I_0_dot=-(5*rho/2)*I_0+rho*c` identifies the cause:
incoming cross-region contribution exceeds local loss before the peak,
equals it there and becomes insufficient afterward. Throughout the episode,

\[
I_0+I_1=\frac{I_{env}}2
\left(e^{-3\rho t/2}+e^{-7\rho t/2}\right)
\]

strictly decreases. This is redistribution of prepared environmental form,
not growth of the complete contrast budget or creation of new nodes.

**A fixed observation criterion.** For a threshold `I_cut>0` fixed before
evaluation, the equation `I_0(t)=I_cut` has two positive finite crossings if
`I_cut<I_max`, one tangency if equal, and none if greater. A positive-duration
episode defined by `I_0>I_cut` exists only in the first case. For `I_cut=0`,
contrast is positive for every finite `t>0`; it disappears only asymptotically.
Threshold exit therefore means loss of this declared observation, not exact
extinction, a topology event or physical death. The threshold never enters
the evolution law. Amplitude scaling across it is not a dynamical bifurcation.

**Reserved three-preparation control.** Fix node order `(0,1,2,3,4,5)`,
regions `((0,1,2),(3,4,5))`, `kappa=1`, weights
`(phase,epi,vf,topo)=(1/4,1/2,1/8,1/8)`, `m=1/2`,
`I_cut=1/8192` and sample times `(0,1/2,1,2,5)`. Region zero starts uniform;
region one has fine form `(m+a,m-a,m)`, hence `I_env=2*a^2`.

| Preparation | Fixed input difference | Continuous prediction |
| --- | --- | --- |
| Connected, stronger input | `a=1/16` | Peak/threshold about `1.465813`; below at `t=1/2`, above at `1,2`, below at `5` |
| Connected, weaker input | `a=1/32` | Peak/threshold about `0.366453`; never reaches the threshold |
| Disconnected control | `a=1/16`; omit the reciprocal interregional edges at initialization | Recipient remains uniform, with exactly zero contrast |

All three use the same pressure recipe and capacity/phase laws. The last
graph has constant outgoing degree one, so its other pressure channels also
vanish. Its different normalization is computed from its actual support,
not copied from the connected generator. All continuous forms stay inside
their initial convex hull. The strong entry is in `(1/2,1)` and exit in
`(2,5)`; no numerical root search is needed to predict those reserved brackets.

The [existing coupled-reduction test owner](../../tests/physics/test_coupled_directed_form_phase.py)
checks the exact solution and crossing classification, then executes shared
pressure refresh and nodal Euler steps with `dt=1/32` through time five.
The independent exact Euler reference is frozen before each run. With its
row-stochastic matrix `M`, the difference obeys `delta_next=M*delta+d`, where
`d` retains separate pressure-realization and nodal-rounding defects. Thus
the cumulative sum of `||d||_infinity` bounds every represented-state error
against that reference. This finite Euler comparison is separate from the
continuous curve and does not certify its exact crossing times in binary64.
Regional means, the complete fine form, primitive phases, capacities and
the admitted form hull remain part of the check.

**Finite outcome.** All three runs matched the reserved classifications.
For the stronger input, measured `I_0` at times `(1/2,1,2,5)` was approximately
`(6.789873e-5,1.468016e-4,1.771180e-4,3.847178e-5)`, compared with the fixed
threshold `1.220703125e-4`. The weaker run remained below that threshold at
the reserved samples, and the disconnected recipient remained exactly uniform
at every executed step. Captured pressure-realization defects were zero in
these runs; the largest nodal-rounding defect was `5.56e-17` or less. The
cumulative represented-state bound was below `6.85e-15`, and the final error
against exact Euler was below `2.63e-16`. These finite arithmetic observations
do not absorb Euler discretization error into a claim of exact continuous flow.

The result is a predicted regional contrast episode on supplied support and
preparation. It does not identify the scalar observation with a complete NFR
identity, derive spontaneous region selection or establish physical emergence.
Earlier [finite identity windows](INHERITED_FORM_DYNAMICS.md#123-an-exact-induced-angular-response-and-closed-rate-law)
and [phase-driven cycle response](../FORCED_SUPPORT_BALANCE.md#32-acute-cycle-relaxation-retains-phase-winding-while-form-relaxes)
remain distinct results. The new peak and crossing classification uses the
existing fine form law without supplying an independent phase oscillator.

#### Collective interaction closure and relational state

The previous Gram result concerns one specified isolated six-node component.
Its ability to interact is a separate obligation: equal local observations
must predict equal selected responses in the **same surrounding preparation**.
This section derives that boundary from the same pure-EPI nodal law, without
adding primitive phase motion, a sustaining source or an operator policy.
Means have form units; Gram entries have squared-form units. Rates refer to
the declared structural clock, with no physical identification assumed.

**All-state linear criterion.** For a fixed real linear fine law on a collection
of triangles, the means and complex contrasts form complete coordinates.
Write its unique real-linear decomposition as

\[
\dot\mu=D\mu+\operatorname{Re}(Ez),\qquad
\dot z=Az+B\overline z+C\mu,\qquad H=zz^\dagger.
\]

Here `D` is real, the other coefficient matrices may be complex, and the
state domain allows independent means and contrasts. The observation
`(mu,H)` is autonomous for **every** fine preparation if and only if
`E=B=C=0`. Its closed law is then

\[
\dot\mu=D\mu,\qquad \dot H=AH+HA^\dagger.
\]

To see necessity, the same Gram fiber consists of `exp(i*theta)*z`, including
its common sign reversal. Equality of all mean rates forces `E=0`.
Differentiating `H` gives, in addition to its displayed closed terms,
`B*conj(z)*z^dagger + C*mu*z^dagger` and their adjoints. Under the rotation,
these have distinct Fourier weights minus two and minus one. Their vanishing
for every angle, mean and contrast forces `B=0` and `C=0`. Sufficiency follows
by substitution. Positive-semidefinite `H` of rank at most one is preserved because the
underlying solution is `z(t)=exp(A*t)*z(0)`, including total zero. In the older
two-region convention, `H_10=c+i*s` and `H_01=c-i*s`.

This necessity concerns homogeneous linear laws on the full stated domain.
For a general nonlinear law, projectability means equality of the projected
rates on every fiber; full equivariance is sufficient, not necessary. For
example `z_dot=i*omega(mu,z)*z`, with any real `omega`, has `H_dot=0` even
when `omega` is not rotation invariant. Restricted preparations and promised
outputs can also admit a weaker closure. No all-model criterion is inferred.

The linear criterion is directly testable on the actual nodal generator.
With mean lift `R`, mean projection `M`, block contrast basis `U` and
`J=diag([[0,-1],[1,0]],...)`, it is equivalent to

\[
MGU=0,\qquad U^TGR=0,\qquad
(U^TGU)J=J(U^TGU).
\]

In these equally oriented triangle frames, this says that every three-by-three
block of `G` is real circulant: a mean scalar and a complex-linear contrast
block supply exactly its three real coefficients. A useful sufficient support
class therefore has nonnegative circulant conductance blocks and capacity
constant within each triangle. Outgoing normalization must be performed on
the **whole** graph; the circulant blocks make its row strength constant within
each triangle. This is a declared interface class, not a derivation of the
graph, its coupling weights or a universal material interaction.

**A fully normalized interaction.** Take components `A` and `B`, each the
existing pair of unit directed triangles, and add reciprocal weight-`w` links
between corresponding vertices, with `w>0`. All capacities are held at one;
primitive phase is held and unused, Gamma is absent, and no event occurs.
Every outgoing strength is now `2+w`. If `G_6` is the earlier six-node
generator, put `alpha=2/(2+w)` and `kappa=w/(2+w)`. The exact full law is

\[
G_{12}=\begin{pmatrix}
\alpha G_6-\kappa I_6&\kappa I_6\\
\kappa I_6&\alpha G_6-\kappa I_6
\end{pmatrix}.
\]

Keeping the old internal rates unchanged after attaching these links would
describe a different pressure law. In contrast coordinates let

\[
A_6=\begin{pmatrix}-(5+i\sqrt3)/4&1/2\\1/2&-(5+i\sqrt3)/4\end{pmatrix},
\quad
A_{12}=\begin{pmatrix}
\alpha A_6-\kappa I_2&\kappa I_2\\
\kappa I_2&\alpha A_6-\kappa I_2
\end{pmatrix}.
\]

Then `z=(z_A,z_B)` obeys `z_dot=A_12*z`. The four means evolve by the same
block construction applied to the earlier mean generator. Consequently four
means and the full four-by-four Gram matrix close exactly. This is a positive
effective-composition result for the declared observation and interface.

**Local descriptions alone fail.** Write `H_A=z_A*z_A^dagger`,
`H_B=z_B*z_B^dagger` and the cross relation `K=z_A*z_B^dagger`. Differentiation
from the fine generator, rather than a new constitutive choice, gives

\[
\begin{aligned}
\dot H_A&=\alpha(A_6H_A+H_AA_6^\dagger)-2\kappa H_A
                         +\kappa(K+K^\dagger),\\
\dot H_B&=\alpha(A_6H_B+H_BA_6^\dagger)-2\kappa H_B
                         +\kappa(K+K^\dagger),\\
\dot K&=\alpha(A_6K+KA_6^\dagger)-2\kappa K+\kappa(H_A+H_B).
\end{aligned}
\]

Thus the product of the two separate component descriptions discards necessary
relational information. When both component contrast vectors are nonzero,
exactly one additional relative common angle is missing generically; the
cross block is a redundant, division-free way of retaining it through zeros.
These correlations are calculated from the same real EPI values. They are not
new primitive coordinates, a fitted coupling, or a quantum density matrix.

For the selected component contrast `E_A=trace(H_A)`, the interface contribution
is `2*kappa*(Re(trace(K))-E_A)`. The corresponding two contributions sum to
`-2*kappa*||z_A-z_B||^2`; neither is a separately conserved energy transfer.
Regional mean exchange `kappa*(mu_B-mu_A)` already factors through the means.
In contrast, individual vertex currents need the oriented fine values and do
not in general factor through the common-rotation quotient.

An exact prospective witness fixes all four means at `1/2`, sets
`z_B=(r,r)`, and compares `z_A=(r,r)` with `z_A=(-r,-r)`. The surrounding
component is unchanged; both preparations have the same separate means and
Gram matrices. Their initial contrast rates are nevertheless

\[
\dot E_A^{+}=-3\alpha r^2,\qquad
\dot E_A^{-}=-3\alpha r^2-8\kappa r^2.
\]

At `w=2` and `r=sqrt(2)/16`, the actual fine forms are dyadic: every triangle
is `1/2 + (1,-1,0)/16`, except for the reversed deviations in component `A`
in the second preparation. The predicted rates are respectively `-3/256`
and `-11/256`, with interface contributions zero and `-1/32`. The cross block
changes sign. Rotating both components together instead preserves the entire
Gram observation and its selected responses; rotating one with its environment
fixed is not that symmetry. Boundary geometry decides what can be discarded.

A single vertex-specific attachment generally falls outside this class. Its
port value contains `u/sqrt(2)+v/sqrt(6)`, which the common-rotation quotient
omits, and the full row normalization can also mix means with contrasts.
The available exact memory/complete-coordinate owners remain the fallback;
the matched-port result does not establish closure for arbitrary connections.

**Evidence and scope.** The existing
[coupled-form controls](../../tests/physics/test_coupled_directed_form_phase.py)
derive the twelve-node projection and cross-Gram evolution, check the exact
equal-local-state witness and bind it to fresh production pressure and a shared
Euler step. The fixed finite control uses `w=2`, capacities one, the stated
dyadic preparations, node order `0,...,11`, binary64 and `dt=1/32` on the
standard unforced scalar path. It distinguishes continuous instantaneous
rates from that held-pressure Euler endpoint. No physical record, random
preparation, fitted parameter or additional solver enters the result.

This supplies a conditional interacting effective state and locates the
information that its separate parts omit. The existing passive dissipation
boundary still applies: it neither produces sustained contrast nor establishes
autonomous formation, material constituents, physical spin or universal TNFR
closure. The active continuation belongs only to the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

<a id="held-affine-source-closure"></a>
#### Held affine sources and the S0-to-D consistency verdict

The homogeneous criterion above does not by itself admit S0's non-form
pressure channels. Retain the same ordered triangles, mean lift `R`, mean
projection `M`, orthonormal contrast basis `U` and complex structure `J`.
Thus `x=R*mu+U*y`, `z_a=y_(2a)+i*y_(2a+1)`, `M*R=I` and `U^T*R=0`.
For a fixed affine fine law

\[
\dot x=Gx+b,
\]

hold **both G and b fixed** when comparing representatives of `(mu,H)`.
The full real-form domain allows independent means and contrasts and complete
common-rotation fibers. Its exact all-state projectability criterion is

\[
MGU=0,\qquad U^TGR=0,\qquad
[U^TGU,J]=0,\qquad U^Tb=0. \tag{A}
\]

The last condition says that the supplied rate source is uniform within each
region: `b=R*h`, where `h=M*b`. Different regions may have different sources.
The resulting autonomous observation law is

\[
\dot\mu=D\mu+h,\qquad
\dot H=AH+HA^\dagger,\qquad D=MGR,
\]

where `A` is the complex representation of `U^T*G*U`. Hence
`H(t)=exp(A*t)*H(0)*exp(A^dagger*t)` preserves the realizable rank-at-most-one
positive-semidefinite domain, including total zero. A retained mean source
need not preserve a global form mean or give bounded absolute form.

**Necessity and sufficiency.** The general affine coordinate law is

\[
\dot\mu=D\mu+\operatorname{Re}(Ez)+h,\qquad
\dot z=Az+B\overline z+C\mu+c.
\]

Equal mean rates along `z -> exp(i*theta)*z` force `E=0`. In the Gram rate,
`B*conj(z)*z^dagger` carries Fourier weight minus two and
`(C*mu+c)*z^dagger` carries weight minus one; their adjoints carry the
opposite weights. Equality for every angle forces each coefficient to vanish.
For every contrast and independently variable mean this implies `B=0` and
`C*mu+c=0`, hence `C=0` and `c=0`. These are exactly (A). Substitution proves
sufficiency. One region is no exception: its intensity still distinguishes
a nonzero fixed contrast source on different orientation representatives.
An imaginary scalar multiple of the identity in `A` is invisible to `H`,
but already commutes with `J`; that common angular-generator freedom cannot
cancel an additive source. The result is for fixed affine laws on the stated
full domain, not a necessity theorem for arbitrary nonlinear or restricted
models.

**A source fixes an orientation reference.** Suppose the homogeneous three
conditions hold but the complex contrast source `c` is nonzero. At equal
means, choose `z_+=tau*c` and `z_-=-tau*c`, where `tau>0` has clock units.
Both preparations have `H=tau^2*c*c^dagger`, yet

\[
\dot H_+-\dot H_-=4\tau cc^\dagger\ne0.
\]

This excludes the proposed observation while leaving the complete affine
fine law well defined. Rotating the source with the preparation would change
the experiment and would not repair this fixed-source quotient. At total
zero, the instantaneous Gram derivative is zero even when `c` is nonzero;
with the homogeneous conditions still in force,
`H(t)=t^2*c*c^dagger+O(t^3)`. A zero-only first-derivative check therefore
misses the obstruction. Retaining the oriented Cartesian contrasts, or an
appropriate source-relative observation, is possible without changing the
fine pressure law. For example `z*c^dagger` reconstructs `z` when this fixed
`c` is nonzero, but then it retains the orientation that the original quotient
discarded. The existing [memory owner](../DERIVED_EPI_MEMORY.md) treats other
explicit choices of incomplete observations.

**Apply the criterion to the admitted reference.** In
[S0](../FUNDAMENTAL_THEORY.md#reference-s-state-admission),
`G=e*N*G_W` and `b=N*F`, with held phase, capacity, support, conductances
and coefficients.
The source condition is consequently `U^T*N*F=0`, not `U^T*F=0` in general.
It admits precisely the pressure sources whose capacity-weighted rates are
regional means. For positive capacity constant within each triangle, this is
equivalent to F being constant within each triangle. With zero capacity,
only its product with pressure enters the criterion; no division by capacity
is used. Arbitrary within-region capacity variation must still pass all
three generator conditions. Neither the source nor the generator condition
can compensate for a failure of the other on the full domain.

| Joint obligation | Scoped verdict and existing owner |
| --- | --- |
| Complete fine law and zero capacity | S0 is a globally defined affine reference at finite times; its [state card](../FUNDAMENTAL_THEORY.md#reference-s-state-admission) holds phase, capacity and support and supplies its pressure realization |
| Collective evolution | Proved exactly when (A) holds; a nonzero contrast source has the explicit equal-observation witness above |
| Derived angle and zero contrast | Cartesian/Gram flow stays defined; an angle at zero is unavailable and need not be invented |
| Units, clock and reporting frames | Apply the existing [joint covariance rules](../NODAL_PARAMETER_FOUNDATIONS.md#pressure-clock-full-state-closure) to the full law; rotating only a representative with fixed source is an active state change |
| Form and contrast budgets | Retain the mean source; under (A) the prior homogeneous contrast balance is unchanged. A source-free diffusion energy theorem does not silently apply to the complete affine form law |
| Changing capacity or operator events | Outside held S0 unless separately admitted; the [capacity-law review](../CAPACITY_LOCALIZATION_BALANCE.md#capacity-law-admission) and [API contracts](../../docs/API_CONTRACTS.md) own the continuous/event distinction |
| Represented execution | Exact-real projectability is distinct from pressure realization, rounded nodal products, clipping and executor endpoints; the shared form observer retains rate defects rather than turning them into a theorem |
| Generative or physical interpretation | No unique pressure, maintained pattern, autonomous substrate or physical constituent is selected by this reduction |

This closes the held affine-source compatibility question without adding a
law to enforce it. It does not extend the S0 reference to the default hybrid
runtime or enlarge D's observables to fine pressures and general tetrad
values. Any narrower bounded preparation domain needs its own admission;
the full-domain necessity proof cannot be inferred merely from a few
represented trajectories.

The shared
[`derive_regional_affine_closure`](../../src/tnfr/physics/form_geometry.py)
checks explicitly supplied node order, ordered partition, generator G and
rate source b. In these frames, each three-by-three generator block must be
real circulant and each regional source triple must be constant. This is
equivalent to (A) and permits exact coefficient residuals without introducing
irrational basis roundoff. For a block with first row `(c0,c1,c2)`, the mean
coefficient is `c0+c1+c2` and its complex contrast coefficient is
`c0-(c1+c2)/2 + i*sqrt(3)*(c2-c1)/2`. The report separates the rational real
and scaled imaginary parts. This detached algebraic admission does not infer
G or b from a graph snapshot, verify a caller's held-law declaration, advance
the graph or certify an executed trajectory. Use b in nodal-rate units;
passing the pressure F in place of N*F would test a different law.

**Production controls.** The existing
[source-closure tests](../../tests/physics/test_internal_mode_source_closure.py)
now exercise this admission and the shared form observer on two preparations
of the unit undirected prism. In node order `(region,vertex)`, with regions
`0,1` and vertices `0,1,2`, compare
`x_+=(1/2+(1,-1,0)/16)` in both triangles with reversed contrasts `x_-`.
Both have means `(1/2,1/2)` and every real Gram entry `1/128`; imaginary
Gram entries vanish. Support, coefficients, capacities and primitive phases
are identical across each pair. No random seed, time integration or fit is used.

For the positive control, capacities are `1/4` in region 0 and `1` in region 1,
with pressure weights `(phase,EPI,capacity,topology)=(0,1/2,1/2,0)` and equal
primitive phases. The native capacity channel gives pressure source
`F=(1/8,1/8,1/8,-1/8,-1/8,-1/8)`, hence mean-rate source `(1/32,-1/8)`.
The admitted contrast generator is real,
`A=((-1/6,1/24),(1/6,-2/3))`. Both actual refreshed preparations have
the same Gram-rate matrix
`((-1/512,-5/1024),(-5/1024,-1/128))` and mean rates `(1/32,-1/8)`.
All pressure-realization and canonical-product defects are exactly zero in
this binary64 dyadic control.

For the excluded control, all capacities are the binary64 value `0.1`,
primitive phases are `(0,pi/3,pi/6)` in both triangles and pressure weights
are `(1/4,1/2,1/4,0)`. The captured represented phase source is
`F=(f,-f,0)` repeated, with `f=6004799503160661/2^57` in this execution.
The exact reference Gram rate in every real entry is
`-nu/128 +/- nu*f/4`: the signs are opposite despite identical means and Gram.
Fresh production pressure and the shared nodal product retain that sign
distinction. The tests independently assemble the ordered phasor source and
account exactly for pressure-assembly and product-rounding defects; they do
not silently identify binary64 operations with the exact-real affine flow.
This single pair excludes the proposed quotient for that held law. It does
not reject the fine dynamics or demonstrate physical formation.

<a id="source-relative-future-response"></a>
#### Source-relative future response with identical initial Gram rates

F3's sign witness already distinguishes the initial Gram derivatives. The
following F4 protocol asks the stronger information question: can the same
means, Gram **and its initial derivative** determine later contrast under one
fixed admitted fine law? It reuses the directed-triangle generator and the
[radial/tangential source decomposition](PHASE_FORM_EXCHANGE.md#161-radial-and-angular-effects-of-an-actual-phase-source).
The law, inputs, horizon and bounds below are fixed before the production
trajectory is evaluated. This is a prospective implementation control, not
reserved physical data or selection of a new pressure law.

**Exact witness.** Keep equal regional means and contrasts on the two equally
oriented directed triangles, common capacity `nu>0`, EPI pressure weight
`e>0`, and a fixed identical regional rate source with complex contrast `c`.
Its regional mean can be nonzero, but must be the same for both preparations.
The existing fine generator induces

\[
\dot z=a z+c,\qquad
a=-\alpha-i\beta,\qquad
\alpha=\frac{3e\nu}{4},\quad
\beta=\frac{\sqrt3e\nu}{4}>0.
\]

For any nonzero c choose `z_+(0)=i*k*c`, `z_-(0)=-i*k*c`, where
`k=sqrt(3)*tau` and `tau>0` has clock units. Every entry of the two-region
Gram is the same real intensity `h=abs(z)^2`. At the preparation time,

\[
h_+(0)=h_-(0)=k^2|c|^2,\qquad
\dot h_+(0)=\dot h_-(0)=-2\alpha k^2|c|^2,
\qquad
\ddot h_+(0)-\ddot h_-(0)=4k\beta|c|^2>0.
\]

Thus retaining the first Gram derivative does not generally close the
observation. No comparison of different sources is needed. The direction
of rotation is inherited from the supplied support, not fitted to the
response. For an arbitrary real regional source triple `(f0,f1,f2)`, the
exact source-relative fine contrast
`tau*(f2-f1,f0-f2,f1-f0)` realizes `i*sqrt(3)*tau*c`; this identity does not
require c to be real. The concrete protocol below fixes ideal dyadic
preparations and retains their defects against the represented source,
rather than changing them after source materialization.

**Finite prediction.** Put `E(t)=exp(a*t)` and `q(t)=(E(t)-1)/a`. Then

\[
z_\pm(t)=c\{q(t)\pm ikE(t)\},\qquad
\Delta h(t):=h_+(t)-h_-(t)
=4k|c|^2\operatorname{Im}\{q(t)\overline{E(t)}\}.
\]

Equivalently,

\[
\Delta h(t)=4k|c|^2\int_0^t
e^{-\alpha(s+t)}\sin\{\beta(t-s)\}\,ds.
\]

This is strictly positive for `0<beta*t<pi`. In particular, for
`beta*t<=pi/2`, use `sin(u)>=u*(1-u^2/6)` and
`exp(-alpha*(s+t))>=exp(-2*alpha*t)` to obtain

\[
\Delta h(t)\ge
2k\beta|c|^2t^2 e^{-2\alpha t}
\left(1-\frac{\beta^2t^2}{6}\right).
\]

The formula and sign precede the trajectory. They follow from the same affine
solution already used by the
[response owner](PHASE_FORM_EXCHANGE.md), rather than a fitted response curve.

**What information is missing?** Define the derived source-relative
coordinate `w=conj(c)*z=R+i*I`. Its exact rows are

\[
\dot R=-\alpha R+\beta I+|c|^2,\qquad
\dot I=-\beta R-\alpha I,\qquad
\dot h=-2\alpha h+2R.
\]

The constraint `R^2+I^2=abs(c)^2*h` still holds. Initial h and its derivative
fix R, but not the sign of the transverse component I. The two preparations
have `R=0` and opposite I. Their inherited rotation subsequently changes
radial source work differently. Retaining oriented Cartesian contrast, or
this source-relative coordinate for fixed nonzero c, restores the missing
information. These are observations of existing form and the declared source,
not new primitive degrees of freedom, an autonomous source or a maintaining
mechanism. If the inherited rotation is absent (`beta=0`), this reflected pair
has identical intensity for all times; the discriminator needs its stated law.

<a id="source-relative-engine-integration"></a>
**Shared source-relative observation.** For several supplied regions, the
production [source-relative owner](../../src/tnfr/physics/source_relative_form.py)
and SDK `Network.source_relative_form` retain the full matrix
`W=z*c^dagger`, reusing the existing form observer and its represented-rate
evidence. Here c is the contrast projection of an independently supplied held
fine **rate** source b, not a pressure or an inferred measured derivative.
The F4 coordinate w is a diagonal entry of W. Local diagonal entries alone
lose an unforced region's contrast when its c component is zero, whereas
the complete matrix gives `z=W*c/(c^dagger*c)` whenever the known vector c
is nonzero. The source-relative description then retains orientation relative
to that source; it does not make orientation an irrelevant symmetry of a
fixed source-driven law.

Writing `c_j=u_j/sqrt(2)+i*v_j/sqrt(6)`, the exact scaled coordinates are
`Re(W_ij)=a_i*u_j/2+b_i*v_j/6` and
`sqrt(12)*Im(W_ij)=b_i*u_j-a_i*v_j`. The held-source pushforward replaces
`a_i,b_i` by their nodal rates. If c varies, the complete chain rule is
`W_dot=z_dot*c^dagger+z*c_dot^dagger`; the second term needs its own justified
source law. No polar coordinate or division is used by the observer, so zero
contrast and c=0 remain valid observations. If c=0, W alone carries no form
orientation. Source means remain separate and may still drive regional means.
The report is a derived observation of supplied state/law data, not an
autonomous evolution, primitive phase identification or physical bridge.
The [SDK guide](../../docs/CLI_AND_SDK.md#retain-orientation-relative-to-a-held-source)
owns exact field names, source admission and usage.

**Frozen production protocol.** The maintained entry point is
[`source_relative_form_response.py`](../../benchmarks/source_relative_form_response.py),
with [contract controls](../../tests/physics/test_source_relative_form_response.py).

| Item | Fixed declaration before evaluation |
| --- | --- |
| Support and order | Nodes `0,...,5`, with label `3*region+vertex`, in lexicographic region/vertex order; two triangles, directed edges `vertex -> vertex+1 mod 3` and reciprocal links between matching vertices; unit conductances |
| Complete held law | Scalar EPI; `nu=1`; primitive phases `(0,pi/3,pi/6)` repeated; pressure weights `(phase,EPI,capacity,topology)=(1/2,1/2,0,0)`; no phase/capacity evolution, operator events or Gamma forcing |
| Native ideal source | Each node has its successor and its matching cross-region vertex as outgoing neighbors; the regular midpoint phase channel gives `F=(1/12,-1/24,-1/24)` repeated |
| Preparations | `x_+=(1/2,5/8,3/8)` repeated and `x_-=(1/2,3/8,5/8)` repeated; both means `1/2`, `tau=1`, `c=1/(8*sqrt(2))+i/(8*sqrt(6))` and `abs(c)^2=1/96` |
| Clock and execution | Declared structural time, `T=1`, 128 steps of `dt=1/128`; refresh native pressure before every shared nodal Euler update; hold all other state and configuration |
| Observations | Shared regional-form observer; retain exact represented means, Gram, Gram rates, source/pressure/product defects and endpoint EPI; compare the first regional intensity, with the second region as a repeated consistency control |
| Exact initial prediction | Every Gram entry `1/32`; every initial Gram-rate entry `-3/128`; initial intensity second-derivative difference `1/64` |
| Prospective finite prediction | At `T=1`, `Delta h >= 127/65536`; this uses `exp(-3/4)>=1/4` in the analytic bound, not an evaluated endpoint |
| Numerical admission | Separate source-model and accumulated arithmetic allowances, each `1e-12` in fine-EPI infinity norm; include the analytic Euler error below; fail rather than refit if an allowance is exceeded |
| Decision | Each trajectory must agree with its predeclared analytic intensity within its error budget; the observed positive intensity gap must exceed `1/1024` |
| Ablation | Keep the same topology, EPI coefficient, capacity, preparations, clock and pressure weights, but set primitive phase uniformly to zero so the native phase source vanishes; do not renormalize channel weights |

The positive-source preparation is independent of the future response.
Capture its represented non-EPI source prospectively and compare it with an
independently assembled phase calculation. The ideal and represented source
are different reference objects: exact initial Gram-rate equality belongs to
the ideal affine model, while the observer retains any realized preparation,
pressure and product defects. Equality must not be manufactured by replacing
measured rates with the analytic prediction. No random seed or parameter fit
is used. The run report records source and runtime provenance.

**Error budget fixed before execution.** Let `B=(1/2)*G_W` be the complete
six-node rate generator and b the repeated ideal source above. Both
`exp(B*t)` and `I+dt*B` are contractions in the fine infinity norm for these
step sizes: their entries are nonnegative and every row sums to one. Since
`x''(t)=exp(B*t)*B*(B*x(0)+b)`, local truncation and contraction give

\[
\delta_{\mathrm{Euler},\pm}
\le \frac{T\,dt}{2}\|B(Bx_\pm(0)+b)\|_\infty.
\]

In the repeated regional rows these second derivatives are respectively
`(-7/128,3/128,1/32)` and `(-1/128,-3/128,1/32)`. Therefore the fixed
fine-EPI bounds are `delta_Euler,+=7/32768` and
`delta_Euler,-=1/8192`. A held source discrepancy contributes at most
`T*norm(b_represented-b,inf)`. The exact represented residual of each
implemented step against its affine Euler row is accumulated separately;
contraction bounds its final contribution by the sum of the residual norms.
The protocol allocates `1e-12` to each of these two contributions, so
`epsilon_+=7/32768+2e-12` and `epsilon_-=1/8192+2e-12` before evaluation.
An initial materialization discrepancy, if present, must be added explicitly;
the declared dyadic EPI preparations have none in binary64.

Throughout the ideal interval, each centered regional component is bounded
by `r_max=1/8+1/12=5/24`. Centering is an orthogonal projection in Euclidean
norm, so an infinity-norm fine error epsilon changes regional intensity by
at most `6*r_max*epsilon+3*epsilon^2`. Apply this separately to both signs.
The sum of these intensity budgets is smaller than the margin between
`127/65536` and the fixed decision threshold `1/1024`. Thus the decision
does not rely on a sign smaller than the admitted discretization error.
The zero-source ablation has exact equal intensity trajectories, although
represented execution may differ within its separately retained error budget.

**Frozen evaluation and result.** The
[prediction record](../../docs/assets/source_relative_form_response/result.prediction.json)
was written before the first production evaluation. Its SHA-256 is
`1a08e804f2ce17b0457eac26f4196e5cb2fd26e1b4374b96ee730938ec1fc022`.
The [result record](../../docs/assets/source_relative_form_response/result.json)
retains exact rational observations, configuration, runtime versions and
fingerprints of the producer and relevant engine owners; its SHA-256 is
`d1a37558f4e5386470b291b648eff0cfa4b2b9a9559a267ef46de76d6be7b94a`.
Subsequent tests are regression reproductions of that evaluated protocol,
not new reserved observations.
The [original producer bytes](../../docs/assets/source_relative_form_response/producer.v1.py.txt)
preserve the source fingerprint of this first result. The maintained producer
now enforces scientific admission even under optimized Python and rejects
replacement of an existing response record. Its later source fingerprint
therefore differs; new-source regression pairs must use distinct output paths.
Neither this hardening nor the new observation API changes the first frozen
prediction or converts a replay into independent reserved evidence.
The current runner also records W and its held-source rate through the shared
source-relative observer, using its prospectively captured source and embedded
form report. These additional read-outs bind the derived coordinate to actual
engine execution without a second pressure refresh or a duplicated chart.

| Quantity at `T=1` | Plus preparation | Minus preparation |
| --- | --- | --- |
| Observed regional intensity | `0.02437996979119982` | `0.01958940432108876` |
| Continuous analytic estimate | `0.024345599871311546` | `0.01960831302470951` |
| Prospective intensity-error budget | `2.67166e-4` | `1.52633e-4` |
| Accumulated runtime arithmetic bound | `4.65812e-15` | `4.59263e-15` |
| Actual endpoint error against exact ideal Euler | `4.76235e-16` | `3.23218e-16` |

The observed gap was `0.004790565470111058`, above the frozen `1/1024`
threshold and the theoretical lower bound after both error budgets. The
same-weight uniform-phase ablation gave a gap of about `-1.016e-16`, within
its retained arithmetic bound. The native source's infinity-norm discrepancy
from the ideal rational source was `4.626e-18`; both initial observed
intensity rates were equal in this execution, with common defect `1.735e-18`
against `-3/128`. No clipping acted, and capacity, primitive phase and support
remained fixed. The complex-exponential entries above are numerical estimates;
the exact rational prediction and error inequalities own the decision.

F4 therefore supports the scoped prediction: an intensity and its first
derivative need not be a sufficient state, even for one fixed admitted law.
Its omitted source-relative quadrature is already available from the fine
form, frame and source; no new microscopic variable or pressure term is needed
to describe this response. This completes the four-stage reference exercise,
not the derivation of a physical constituent, unique pressure realization,
autonomous source, emergent substrate or indefinite persistence.

#### Joint realizability of collective relations

The interaction law uses correlations of one deterministic fine preparation.
Their admissible values are constrained before any evolution is considered.
For components with complex contrast vectors `z_A`, assemble all blocks
`H_A=z_A*z_A^dagger` and `K_AB=z_A*z_B^dagger` into one matrix `H`.
It is Hermitian positive semidefinite of rank at most one. Conversely every
such matrix has a factor `H=z*z^dagger`: if a diagonal entry `H_qq>0`, choose
`z_i=H_iq/sqrt(H_qq)`; the only rank-zero case is `H=0`, with `z=0`.
Nonzero factors differ by one common phase. With the retained means and
regional lifts this constructs real fine EPI. Any extra numerical clipping
rails must still be checked on that lift; they are not a Gram axiom.

Put `E_A=trace(H_A)=z_A^dagger*z_A`. Multiplication gives

\[
K_{AB}K_{AB}^\dagger=E_BH_A,\qquad
K_{AB}^\dagger K_{AB}=E_AH_B,\qquad
K_{AB}K_{BC}=E_BK_{AC}.
\]

These are consequences of the factorization, not additional dynamics. If
`E_A=0`, then `z_A=0` and every relation incident to `A` is zero; a phase
at that zero is unavailable. If `E_B>0`, two complete relations through `B`
determine the third by the displayed identity. No division is admitted at
`E_B=0`: relations to that zero component say nothing about `K_AC`.
The full positive-semidefinite/rank condition is the admission criterion;
these convenient identities are not claimed to be a complete independent
set for arbitrary partially recorded block data.

Pairwise admissibility alone is insufficient. Unit scalar intensities and
relations `K_AB=K_BC=1`, `K_AC=-1` give

\[
H_{\rm proposed}=\begin{pmatrix}1&1&-1\\1&1&1\\-1&1&1\end{pmatrix}.
\]

Each two-component principal block is positive semidefinite and rank one.
But `v=(1,-1,1)` gives `v^T H_proposed v=-3`, so the whole matrix cannot
arise from any single fine preparation. It also violates
`K_AB*K_BC=E_B*K_AC`. The example embeds into the existing two-contrast
components by giving each a local direction `(1,0)`; introducing more
coordinates does not repair the incompatible relations.

For `m` nonzero components of two complex contrasts each, their separate
local Gram descriptions have `3*m` real degrees of freedom. The full joint
Gram has `4*m-1`, leaving `m-1` independent relative common orientations.
Thus relational properties matter but are not arbitrary pairwise additions.
This count is on the nonzero stratum with fixed component frames; zero
components have no orientation to retain. Means are additional coordinates.

The [collective controls](../../tests/physics/test_coupled_directed_form_phase.py)
check reconstruction from actual fine-coordinate preparations, cross-relation
identities, a zero-contrast boundary and the incompatible-pairs witness.
They test realizability, not temporal stability or a new engine trajectory.
An ensemble-averaged Gram can have higher rank, but that would be a different
statistical observation with explicitly declared preparation and evolution.
No ensemble rule, quantum interpretation, phase-winding theorem or new
constitutive law is supplied by this algebraic result.

#### Imperfect interface: ambiguity and a finite predictive bound

The exact collective law above assumes a circulant interface. Its loss need
not make every reduced prediction useless, but the discarded orientation can
then matter. This section keeps the same twelve-node pure-EPI law, unit held
capacity, fixed support, unused primitive phase and absent Gamma. Only the
outgoing conductance `0 -> 6` changes from `2` to `2+epsilon`, with
`0<=epsilon<=1`; the reverse edge and every other conductance stay fixed.
The affected outgoing strength is `4+epsilon`, not four. Define

\[
\eta=\frac{\epsilon}{4+\epsilon},\qquad
r^T=\tfrac12 e_6^T-\tfrac14 e_1^T-\tfrac14 e_3^T.
\]

Directly normalizing that row gives the exact rank-one update

\[
G_\epsilon=G_0+\eta e_0r^T,\qquad r^T\mathbf1=0,
\qquad \|e_0r^T\|_\infty=1.
\]

Both generators have diagonal minus one, nonnegative off-diagonal entries
and zero row sums. Their positive stochastic semigroups contract the infinity
norm. This is a change of a supplied structural interaction, not an additional
force, a new phase law or a derivation of why that conductance changes.

**A prospective ambiguity, including finite time.** Set `a>0`,
`v=(1,-1,0)` repeated four times, and
`x^+(0)=b*1+a*v`, `x^-(0)=b*1-a*v`. These preparations have identical four
means and identical full joint Gram, including all cross-component relations.
The lost coordinate is now the **common** orientation of the whole pair,
rather than the relative orientation already retained by its cross block.
Their initial first-region mean rates are

\[
\dot\mu_0^+(0)=\eta a/6,\qquad
\dot\mu_0^-(0)=-\eta a/6.
\]

Consequently exact all-state mean/Gram closure fails for every `epsilon>0`.
This obstruction is not a numerical defect. Linear sign symmetry keeps the
two exact Gram trajectories equal while their means can separate.

Let `M_0=(e_0^T+e_1^T+e_2^T)/3` denote the first mean row. The old
unperturbed mean stays at `b`. In its equal-region contrast sector,

\[
r^Te^{G_0s}v=\tfrac12 e^{-3s/8}\cos(\sqrt3 s/8).
\]

This follows from the already derived directed-triangle generator
`(S-I)/4` on that sector. Duhamel's identity therefore gives

\[
q(t):=M_0e^{G_\epsilon t}v
 =\frac\eta2\int_0^t
 M_0e^{G_\epsilon(t-s)}e_0\,
 e^{-3s/8}\cos(\sqrt3 s/8)\,ds.
\]

For `u>=0`, the nonnegative exponential series for `G_epsilon=P_epsilon-I`
gives `M_0 exp(G_epsilon*u)e_0>=exp(-u)/3`. On `0<t<=1/2`, the cosine is
positive and decreasing. Thus

\[
q(t)\ge\frac{\eta t}{6}e^{-t}\cos(\sqrt3 t/8),\qquad
\mu_0^+(t)-\mu_0^-(t)=2a q(t).
\]

Any deterministic single-valued predictor receiving only that initial joint
observation and the same perturbed generator must err on at least one
preparation by at least

\[
\frac{a\eta t}{6}e^{-t}\cos(\sqrt3 t/8).
\]

This is an information-loss lower bound, not a claimed physical uncertainty
principle. Retaining the omitted orientation restores the complete Cartesian
state; retaining only the observation permits a set-valued prediction instead.

**An upper bound available from the observation alone.** For arbitrary
admissible initial means and joint Gram, put

\[
L_Q=\min_j(\mu_j-\sqrt{2/3}\sqrt{H_{jj}}),\quad
U_Q=\max_j(\mu_j+\sqrt{2/3}\sqrt{H_{jj}}),\quad
B_Q=(U_Q-L_Q)/2.
\]

Each row of the orthonormal triangle contrast basis has norm `sqrt(2/3)`.
Every fine representative of this observation therefore lies in
`[L_Q,U_Q]`; this estimate does not choose its hidden common orientation.
Center at `b_Q=(L_Q+U_Q)/2`. Since the perturbation annihilates constants,
Duhamel and stochastic contraction give, for the same initial fine state,

\[
\|x_\epsilon(t)-x_0(t)\|_\infty\le\eta t B_Q,
\qquad
\|\mu_\epsilon(t)-\mu_0(t)\|_\infty\le\eta t B_Q.
\]

The unperturbed mean/Gram solution is independent of the chosen representative
and hence supplies one computable reference for the whole fiber. Orthogonal
contrast projection gives `||z||_2<=sqrt(12)*B_Q` for both trajectories and
`||z_epsilon-z_0||_2<=sqrt(12)*eta*t*B_Q`. Applying
`||zz^dagger-ww^dagger||_F<=(||z||_2+||w||_2)*||z-w||_2` yields

\[
\|H_\epsilon(t)-H_0(t)\|_F\le24\eta t B_Q^2.
\]

These absolute bounds have respectively form and squared-form units, apply
to every representative, and remain defined at zero regional amplitude.
At uniform global form, `B_Q=0` and both responses vanish exactly. Unequal
regional means can give `B_Q>0` even with initially zero contrasts; that case
must not be silently declared frozen. No division by amplitude or angular
accuracy guarantee is used. The proof reuses the finite-semigroup defect
method of [structural morphisms](../../src/tnfr/physics/structural_morphism.py)
and the [projection/memory framework](../DERIVED_EPI_MEMORY.md), without
applying its reversible graph adapter to this directed support.

The outcome is a conditional finite predictive enclosure despite failure of
exact observation closure. A requested error below the lower ambiguity bound
cannot be guaranteed without more information; the upper bounds give sufficient
absolute error budgets for the unchanged collective reference. No claim of
relative accuracy at vanishing amplitude, universal lifetime or maintenance
follows, and increasingly tight passive bounds are not a separate campaign.

**Bounded control fixed before execution.** Use `epsilon=1/4` (`eta=1/17`),
`b=1/2`, `a=1/16`, the two global-sign preparations above, node order
`0,...,11` and the existing binary64 scalar engine path (nonvectorized pressure
and the scalar shared-integrator branch). Evaluate at
`t_star=1/32` with 32 fresh-pressure shared Euler steps of `h=1/1024`.
Primitive phases stay zero, capacities stay one, no events or Gamma are
active, and the preparations stay inside the configured `[0,1]` form rails.
The analytical bounds hold over `0<=t<=1/2`; this control samples only
`t_star`, not a parameter or horizon sweep.

The independent references are the exact rational Euler recurrence and the
continuous matrix exponential. For these preparations, `||x(0)-b*1||_infty=a`,
`||G_epsilon||_infty=2` and `h<=1` make the Euler transition stochastic.
The integral local remainder and contraction give the continuum discretization
bound `2*a*t_star*h=1/262144` per trace. Actual pressure and integration
rounding defects are captured separately as exact differences of represented
values and propagated with stochastic gain at most one. Their sum is an
arithmetic bound, not a replacement for the time-discretization bound. The
matrix exponential is a floating reference, not a formal transcendental
certificate; the displayed lower/upper bounds are proved independently.

The [three bounded controls](../../tests/physics/test_coupled_directed_form_phase.py)
passed with the frozen preparation and method. The whole affected research
module passed 36 tests. The retained endpoint metrics are:

| Quantity at `t_star` | Value and interpretation |
| --- | --- |
| Executed first-region mean separation | `3.7651500332e-5` form units |
| Analytical continuum separation lower bound | Approximately `3.7117457513e-5` form units |
| Combined numerical uncertainty for the pair | At most `7.6293945342e-6` form units from truncation plus captured arithmetic |
| Observation-only mean error upper bound | Approximately `1.3266320524e-4` form units |
| Observation-only Gram Frobenius error upper bound | Approximately `2.2977941176e-4` squared-form units |
| Continuous exponential-reference mean / Gram errors | Approximately `1.8815585843e-5` / `1.2017643258e-5`, each within its respective bound |
| Largest accumulated arithmetic bound | Below `1.55e-15` form units; separate from the Euler bound |

The displayed transcendental bounds are rounded summaries of the exact
expressions above. The pair's numerical uncertainty is smaller than its
separation; the two responses are resolved without changing a coefficient,
preparation or error budget after execution. The finite results confirm the
implemented realization, while the analytic argument establishes the stated
all-preparation upper bounds and selected-family lower bound. Compact run
metrics are recorded as JUnit properties by the controls; local output is
`artifacts/research/imperfect_interface_2026_09_26.xml` (ignored evidence).
No physical data were evaluated, and this completes the bounded robustness
gate rather than opening further passive-transport optimization.

### 21.3 Canonical phase geometry has directed sensitivity on reciprocal support

Return to the unchanged reciprocal prism and its retained nonrepeated phase
preparation `theta=(-a,a,0,-a,a,a)`, `a=pi/6`. The shared derivative
`Dg=(R-I)/pi` uses
`R_ij=1[j in N_i]*sum_{k in N_i}cos(theta_j-theta_k)/|S_i|^2`.
Every supported entry is positive. Put `A=(3-sqrt(3))/4`,
`B=(sqrt(3)-1)/2`. On triangle zero,

\[
\frac{R_{01}R_{12}R_{20}}{R_{02}R_{21}R_{10}}
 =\frac{AB(2/7)}{AB(5/14)}=\frac45.
\]

Positive diagonal detailed balance would require this ratio to equal one.
Thus reciprocal graph support does not guarantee a reversible phase-source
response. The repeated control `(-a,a,0)` on both triangles instead has
`R_ij=1[i~j]*c_j/(1+sqrt(3))`,
`c=(sqrt(3)/2,sqrt(3)/2,1)` repeated, and obeys detailed balance with weights
`c_i`. This supplies directional sensitivity from existing phase geometry,
without replacing the graph by an imposed directed topology.

It does not select a velocity: reversing every phase leaves `R` unchanged
and reverses `g`. Nor does it contradict the phase metric in variational
section 13.6: differentiating a state-dependent mobility contributes another
term to `Dg` away from equilibrium. Cycle imbalance is neither oriented
source work nor an autonomous oscillation. The
[complete metric differential](../TNFR_VARIATIONAL_PRINCIPLE.md#136-exact-state-dependent-metric-for-canonical-phase-pressure)
now accounts for every weighted antisymmetric entry. The
[joint source/work classification](../TNFR_VARIATIONAL_PRINCIPLE.md#1318-source-sensitivity-joint-work-and-the-passive-transfer-limit)
separately retains the phase velocity and mean: identical response geometry
can give opposite source work. The matrix-asymmetry branch is closed in
scope; the missing maintenance law cannot be replaced by this diagnostic.
At the retained point, the mobility-derivative contribution to entry `(0,1)`
is exactly `(3-sqrt(3))/(4*pi)-1/(6*(1+sqrt(3)))>0`; the held-mobility
Hessian term alone misses it. This cross-entry check is already complete.
Controls: [source-response cycle balance](../../tests/physics/test_phase_response_cycle_balance.py).

### 21.4 Local wave realizability does not select a wave law

On fixed reciprocal unit-prism support, retain unit capacities and mixed
pressure `p=-e*L*x+w*g(theta)` with fixed coefficients `e,w>0`.
Let `Pi` center a field, `y=Pi*x`, `v=Pi*p`.
In a lifted chart with **global** phase spread below `pi/2`, `R` is
irreducible, nonnegative and row stochastic. Its positive left stationary
vector `l` proves that `Pi*Dg` is invertible on centered phase space:
`Pi*Dg*h=0` implies `Dg*h=c*1`; multiplying by `l^T` gives `c=0`, and
the centered kernel is zero. Hence `(y,theta modulo rotation)->(y,v)` is a
local coordinate change. Per-edge U3 alone is not this global-spread premise.

Any supplied smooth centered acceleration `F(y,v)` has the local lift

\[
w\Pi Dg\,\dot\theta=F(y,v)+eLv,\qquad
\dot\mu=w\,\overline{g(\theta)}.
\]

Common phase rotation remains free. Choosing the existing graph wave
`F=-Ly` conserves its centered degree-metric energy
`H=3*(||v||^2+y^T L y)/2` while the chart is valid; its coordinate pullback
has a nondegenerate ten-dimensional two-form. This uses phase contrast as
the wave coordinate. **Choosing that acceleration is still a constitutive
premise**: the same invertibility permits other accelerations and proves no
global continuation. At uniform phase, `k=w/pi` gives
`theta_dot=(y-e*v)/k` modulo rotation; eigenvalue-one modes require
`a=(1+e^2)/k`, `b=e` in [section 17.4](PRIMITIVE_PHASE_CLOSURE.md#174-admission-conditions-for-a-joint-linear-response), its active neutral boundary.

Mean motion is essential. At the retained prepared phase
`theta=(0,pi/3,pi/6)` repeated, with `y=a*(1,1,-2)` repeated, `a>0`,
the full requirement `x_ddot=-Lx` is outside the source derivative's image.
The centered wave is liftable but forces

\[
\ddot\mu=(1+e^2)a\frac{\sqrt3-2}{1+\sqrt3}\ne0,
\]

even though `mu_dot=0` there. This is different from the rejected harmonic
motion of the extracted tetrad fields. Controls:
[centered wave lift and mean](../../tests/physics/test_centered_phase_wave_lift.py).

The stronger [constitutive-condition classification](../TNFR_VARIATIONAL_PRINCIPLE.md#1319-structural-closure-tests-exchange-jacobi-and-the-remaining-potential)
distinguishes power balance, a Poisson tensor and a chosen Hamiltonian.
In this selected centered symplectic chart, `ydot=v` restricts the
Hamiltonian to `3||v||^2/2+U(y)` but leaves `U` undetermined. This is
stronger than arbitrary-acceleration realizability and weaker than a
derived wave law. The exact tests retain the original form and distinguish
the alternatives by its existing transverse mode, not a new trajectory.
The subsequent [locality/transport classification](../TNFR_VARIATIONAL_PRINCIPLE.md#1320-locality-and-exact-diffusion-compatibility)
separates one-hop acceleration from one-hop primitive-phase dependence.
On the retained phase, a common-phase gauge cannot make the selected
`-kappa*L*y` lift primitive-local. Exact diffusion embedding also differs
from a leading overdamped balance. These fixed-capacity restrictions do
not select the missing complete-triad relation.

### 21.5 A form-coordinate boundary need not require an operator

The reciprocal prism already gives a finite-time autonomous control. Keep
pure-EPI weight `e>0`, unit capacity, means `1/2`, primitive phases zero,
and the unnormalized coefficients of [section 12](INHERITED_FORM_DYNAMICS.md#12-intrinsic-response-from-a-closed-fine-nodal-model):

\[
q=e^{-et/3},\qquad
u_0=\frac{q^3(1-4q^2)}{32},\quad
u_1=\frac{q^3(1+4q^2)}{32},\quad v_0=v_1=0.
\]

They satisfy the shared induced generator. At `t_*=3*log(2)/e`, `q=1/2`,
the first regional contrast crosses zero: `(u_0,u_1)=(0,1/128)`, with rates
`(e/384,-e/96)`. Its polar angle loses its domain and reappears on the opposite
ray. For `t>=0`, EPI remains in `[11/32,21/32]`; this is not node creation from
vacuum. Full Cartesian evolution, fresh source and continuation remain regular,
with unchanged graph/source ranks and nonzero primitive phasor resultants.
Boundary transfer can regrow local contrast while global contrast decreases.

Before treating a geometric transition as compulsory, distinguish loss of
an observer's chart from loss of full-state continuation or admissibility.
The polar singularity alone does not require Mutation. Existing THOL gates
compare measured absolute acceleration against configured thresholds on an
admitted invocation; operator names alone do not establish a mathematical
bifurcation. Their source docstrings now state that boundary correctly.
Controls: [exact form-zero continuation](../../tests/physics/test_internal_form_zero_continuation.py).

Historical source-bound validation record:
`artifacts/research/phase_law_reuse_validation_2026_09_19.json`.
