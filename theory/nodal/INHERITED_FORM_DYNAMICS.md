# Inherited form dynamics and observation

Fine-to-coarse form, inherited pressure and metric, tetrad reconstruction and changing support.

Section numbers are stable locators across this document family.
The [parameter reference](../NODAL_PARAMETER_FOUNDATIONS.md) owns the
reading map; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns active tasks. Each result retains its stated model and scope.

## 12. Intrinsic response from a closed fine nodal model

The constitutive-origin review has a positive, restricted candidate:
**differential relaxation of internal EPI modes induces an autonomous
normalized-shape response and a changing relaxation rate**. Both follow from
an already closed fine model. Neither requires selecting a new potential,
reinjecting a diagnostic or fitting pressure from an observed derivative.
This is a derived effective response; identifying it with primitive capacity,
synchronization phase or a persistent macro-NFR is a separate obligation.

### 12.1 Mechanism inventory and reuse decision

The review separates actual dynamical constructions from their names and
read-outs. It covers the following owners, not every historical repository
claim or every possible TNFR completion.

| Route and existing owner | Reusable mechanism | What it does not derive |
| --- | --- | --- |
| Reversible EPI transport: [structural_diffusion.py](../../src/tnfr/physics/structural_diffusion.py), [forced_support.py](../../src/tnfr/physics/forced_support.py) | Closed fine linear evolution, fixed metric, modal decay and exact removal of held forcing/profile drift. These supply the candidate below. | Initial support, a variable primitive capacity law or a sustaining source. |
| Projection and memory: [epi_memory.py](../../src/tnfr/physics/epi_memory.py), [structural_morphism.py](../../src/tnfr/physics/structural_morphism.py) | Exact hidden-mode feedback, projectability, inherited coarse coefficients; the reciprocal kernel obeys `Hbar*K(0)=C^T*H*C`. | Laws for the fine model's supplied capacity/forcing, autonomous selection of a partition or a new energy source. |
| Capacity feedback/localization: [capacity_feedback.py](../../src/tnfr/physics/capacity_feedback.py), [capacity_localization.py](../../src/tnfr/physics/capacity_localization.py) | Exact consequences of configured Coupling/Euler maps and held capacity contrasts. | The origin of their gain, event timing or prepared contrast. |
| Phase and adaptation: [phase_evolution.py](../../src/tnfr/dynamics/phase_evolution.py), [adaptation.py](../../src/tnfr/dynamics/adaptation.py) | Shared implementations and explicit admission/parameter contracts. | The identification of capacity with angular speed, or a fundamental law selected by pressure/Si thresholds. |
| Auxiliary geometry: [symplectic_substrate.py](../../src/tnfr/physics/symplectic_substrate.py), [variational scope](../TNFR_VARIATIONAL_PRINCIPLE.md) | A specified harmonic model and explicit graph-field realizability tests. | A bridge from its ambient flow or snapshot bilinear contractions to the missing nodal laws. Existing P2 obstructions remain applicable. |
| Gauge, coherence geometry and transitions: [gauge.py](../../src/tnfr/physics/gauge.py), [coherence_geometry.py](../../src/tnfr/physics/coherence_geometry.py), [phase_transition.py](../../src/tnfr/physics/phase_transition.py) | Conditional geometric identities, level sets and transition diagnostics. | A new restoring force merely from their geometric or criticality labels. |
| Birth and changing support: [birth/transport](../THOL_BIRTH_AND_TRANSPORT.md), [remesh.py](../../src/tnfr/operators/remesh.py) | Executable canonical transformations, finite causal birth/transport and state-dependent reconstruction policies. | A uniquely derived occurrence/selection law for those transformations. |
| Rich EPI and ontology: [epi.py](../../src/tnfr/mathematics/epi.py), [EMERGENT_ONTOLOGY.md](../EMERGENT_ONTOLOGY.md) | Richer representations, spectral observations and scoped comparisons. | Autonomous closure from storage dimensionality or identification of those observations with physical particles. |

An important distinction is that **an emergent observable can have a derived
law while the primitive law remains an input**. The present candidate takes
the already-studied pure EPI channel with fixed positive fine capacities and
fixed reciprocal conductance as that input. It does not derive those fine
premises or claim to complete the full multichannel engine.

### 12.2 Shape and rate are induced by the nodal generator

Use the held model and metric already owned by `forced_support`:

\[
\dot x=-Ax+b,\qquad
A=e\,\operatorname{diag}(\nu_i/d_i)B,\quad
B=D-W,\quad H=\operatorname{diag}(d_i/\nu_i).
\]

Assume connected symmetric nonnegative conductance with positive strengths,
positive fixed capacities and `e>0`. Then `HA=eB` is symmetric positive
semidefinite. With zero held forcing, subtract the conserved `H` mean. With
nonzero held forcing, additionally subtract the exact relative profile from
the shared owner; its existing drift identity still gives

\[
y=x-\operatorname{mean}_H(x)\mathbf1-z,\qquad \dot y=-Ay,
\qquad \langle\mathbf1,y\rangle_H=0.
\]

The intrinsic-origin control uses zero forcing. Allowing a held nonzero
source extends the accounting; it does not explain that source's origin.
For `S=<y,y>_H>0`, define

\[
R=\sqrt S,\qquad q=y/R,\qquad
\kappa=\frac{\langle y,Ay\rangle_H}{S}.
\]

These quantities are computed from current fine state and generator. Direct
differentiation, using self-adjointness in `H`, gives

\[
\dot S=-2\kappa S,\qquad \dot R=-\kappa R,\qquad
\dot q=-Aq+\kappa q,
\]

\[
\dot\kappa
=-2\left(\frac{\langle Ay,Ay\rangle_H}{S}-\kappa^2\right)
=-2\frac{\|Ay-\kappa y\|_H^2}{S}\le0.
\]

For example, differentiate `N=<y,Ay>_H`: `Ndot=-2<Ay,Ay>_H`;
the quotient rule for `N/S` gives the last identity. Also
`<y,Ay>_H=2*e*E_D(y)` and `<q,qdot>_H=0`. The term `+kappa*q`
arises from differentiating the normalization; it is not a force fed back
into the graph. The normalized-shape law closes without amplitude because
`kappa=<q,Aq>_H` on the unit sphere. No primitive capacity has been changed.

The shape is an oriented unit vector, retaining the sign of EPI. Quotienting
by positive amplitude does not identify `y` with `-y`. A real projective
identification would discard that additional distinction. At `y=0`, shape,
rate and their ratio-based derivatives are undefined, not measured zeros.
A constant rescaling of the metric rescales `S` and `R`, but not `kappa`;
amplitude units inherit the declared fixed metric normalization.

In an `H`-orthonormal eigenbasis, differential decay selects the lowest-rate
eigenspace that has a nonzero initial component. A degenerate eigenspace
retains the initial direction within it, so no unique orientation is selected
there. A pure eigenmode has fixed normalized shape and constant `kappa`.
The shape selection has a geometric restoring interpretation on this unit
sphere, while **the original amplitude continues to decay**. Connected pure
diffusion does not thereby maintain a nonuniform finite-amplitude NFR.

### 12.3 An exact induced angular response and closed rate law

On an invariant plane with `H`-orthonormal eigenvectors of rates `r_1<r_2`,
write `y=u*v_1+v*v_2`, `u=R*cos(alpha)`, `v=R*sin(alpha)`. The fine equations
`udot=-r_1*u`, `vdot=-r_2*v` imply, away from the zero vector,

\[
\dot\alpha=(r_1-r_2)\sin\alpha\cos\alpha,\qquad
\kappa=r_1\cos^2\alpha+r_2\sin^2\alpha,
\]

\[
\dot\kappa=-2(\kappa-r_1)(r_2-\kappa).
\]

All coefficients are inherited decay rates. This is an induced angular
response and a genuinely closed rate equation on the specified plane; it is
not the assumption `theta_dot=nu`, and `alpha` is not automatically the
engine's neighbor-synchronization phase. Basis orientation affects the angle.
The strict rate decrease off eigenmodes also excludes nonstationary periodic
normalized-shape motion in this fixed reversible model.

The scalar rate law does not generalize by retaining only `kappa` on every
graph. If three distinct rates `r_1<r_2<r_3` are present, a pure `r_2` mode
and a unit-norm mixture of the endpoint modes with energy fractions
`(r_3-r_2)/(r_3-r_1)` and `(r_2-r_1)/(r_3-r_1)` have the same `S` and
`kappa=r_2`. Their rate derivatives are respectively zero and
`-2*(r_2-r_1)*(r_3-r_2)`. A spectral variance or further state is required.
This reuses the existing fiber/projectability criterion rather than treating
a scalar summary as a complete macro state.

**Exact P3 control.** On the unit three-node path with fine capacity one,
`e=1`, zero other source and `x=(1,1/4,1/2)`, the actual neighbor-mean
generator gives

```text
H=diag(1,2,1), mean_H(x)=1/2, y=(1/2,-1/4,0),
Ay=(3/4,-1/2,1/4), S=3/8, E_D(y)=5/16,
kappa=5/3, -Ay+kappa*y=(1/12,1/12,-1/4),
spectral variance=2/9, kappa_dot=-4/9, S_dot=-5/4.
```

Its orthogonal modes `(1,0,-1)` and `(1,-1,1)` have rates one and two.
Writing `zeta=exp(-t)`, the exact solution is

\[
y(t)=\frac{\zeta(1,0,-1)+\zeta^2(1,-1,1)}4,
\quad S=\frac{\zeta^2+2\zeta^4}{8},
\quad\kappa=\frac{1+4\zeta^2}{1+2\zeta^2}.
\]

Differentiation using `zeta_dot=-zeta` yields
`kappa_dot=-4*zeta^2/(1+2*zeta^2)^2`, exactly the two-rate law. This proves
an interval identity; rational evaluations are regression controls, not a
numerical trajectory campaign. As `t` increases, shape approaches the first
mode, `kappa` approaches one and `R` approaches zero. The changing rate did
not require a changing fine capacity or an imposed phase schedule.

**Finite identity has both shape and retained amplitude.** In this same P3
solution, the H-energy fraction in the slower mode and total retained
squared amplitude are

\[
P=\frac1{1+2\zeta^2},\qquad
F=\frac{S(t)}{S(0)}=\frac{\zeta^2+2\zeta^4}{3}
=\frac{1-P}{6P^2}.
\]

As `P` increases from `1/3` towards one, `F` decreases from one towards
zero. Thus increasingly recognizable normalized form and loss of signal
can occur together. A finite identity criterion must retain both quantities
instead of promoting a normalized shape to finite-amplitude maintenance.
This does not make a finite-lived coherent pattern inadmissible; it separates
its lifetime from indefinite recurrence.

More generally, two positive decay rates `r1<r2`, initial modal energy ratio
`R=E2(0)/E1(0)>0` and `Delta=r2-r1` give

\[
P(t)=\frac1{1+R e^{-2\Delta t}},\qquad
F(P)=\frac1{(1+R)P}
 \left(\frac{1-P}{RP}\right)^{r_1/\Delta}.
\]

For prospectively declared observation thresholds `P_*` in `[P(0),1)`
and `eta` in `(0,1]`, a time satisfying both `P>=P_*` and `F>=eta`
exists exactly when `eta<=F(P_*)`; strict inequality gives a positive
time window. The thresholds define the tested observation, not dynamical
coefficients or TNFR constants. At `P_*=1` there is no such finite time.
This analytic criterion reuses the derived two-mode law; no new graph,
trajectory or maintaining force is selected. Controls:
[finite shape/retention frontier](../../tests/physics/test_finite_identity_shape_scope.py).

### 12.4 Canonical identification and observation scope

Writing `Rdot=kappa*(-R)` does not uniquely factor a nodal equation into
capacity and pressure. For any positive scale `c`, the pair
`(c*kappa,-R/c)` gives the same product. The current isolated canonical
scalar node has zero neighbor pressure, so assigning it capacity `kappa`
does not reproduce this internal contraction. An inherited internal pressure
or explicit coupled environment must be derived through an observation map;
it cannot be concealed by relabeling `kappa` or `alpha`.

The full tetrad remains tied to the fine primitives. In the P3 control,
primitive phases may stay at consensus: phase gradient and curvature then
stay zero even while modal orientation changes. Structural potential reads
the evolving actual pressure. Coherence length retains its pressure-product
and graph-distance provenance (on this small graph, its spectral fallback).
Amplitude and the retained source/state are needed to reconstruct these
observations; the normalized shape alone does not close the full tetrad.

`observe_forced_support_shape` in
[forced_support.py](../../src/tnfr/physics/forced_support.py) centralizes the
exact rational identities, reusing reference rebuilding, relative profiles,
the graph Laplacian and fixed metric. It returns the scaled tangent without
evaluating a square root or choosing an eigenbasis. The source-state pressure
defect stays visible; modeled derivatives are not promoted to derivatives of
stale stored pressure, numerical execution or future schedules. The tests in
[test_forced_support_shape.py](../../tests/physics/test_forced_support_shape.py)
cover the exact P3, single-mode and zero-shape cases, covariance, held forcing,
cache reconstruction and the isolated-macro-node obstruction.

This supplies a concrete endogenous geometric-response mechanism within a
closed TNFR restriction. It does not provide the missing potential `Psi` of
section 13.5 in the variational note, a sustained particle, or a universal
closure for primitive phase/capacity. The following exact internal-mode
construction tests a stronger, neighbor-coupled response. Other candidate
mechanisms above remain references rather than parallel research queues.

### 12.5 Neighbor-coupled phase and internal pressure from scalar EPI

A fixed Cartesian product `P2 square C3` supplies a bounded constructive
test. Its six fine nodes have unit conductance, unit positive capacity and
pure EPI pressure (`e=1`); primitive phases may stay equal and all other
pressure coefficients are zero. These are declared fine-model premises,
not a proof that this graph or configuration forms spontaneously. Each node
has three neighbors, so the exact fine generator is `A=B/3` and `H=3*I`.
No angular equation is added to that model.

Write the three scalar EPI coordinates in fiber `a` as

\[
x_a=m_a\mathbf1+u_a p+v_a q,\qquad
p=(1,-1,0),\quad q=(1,1,-2).
\]

The projections are `m_a=sum(x_a)/3`, `u_a=(x_a0-x_a1)/2` and
`v_a=(x_a0+x_a1-2*x_a2)/6`. The two internal columns are orthogonal with
Gram `diag(2,6)`; both have internal combinatorial Laplacian eigenvalue
three. Applying the actual fine neighbor-mean law gives, with `b=1-a`,

\[
\dot m_a=(m_b-m_a)/3,\qquad
\dot u_a=(u_b-4u_a)/3,\qquad
\dot v_a=(v_b-4v_a)/3.
\]

Consequently the internal projection closes for **every fine EPI state**,
independently of fiber means; its zero-mean lift is an invariant subspace.
This is both a projection intertwining and an invariant-lift statement,
not an inference from one trajectory. In the Euclidean-orthonormal internal
chart `z_a=sqrt(2)*u_a+i*sqrt(6)*v_a`, the induced equation is

\[
\dot z_a=\frac{z_b-z_a}{3}-z_a.
\]

The complex coordinate abbreviates two real internal form coordinates. It
is not the lossy product of signed scalar EPI with an independently supplied
primitive phase studied in [section 9](JOINT_PARAMETER_RESPONSE.md#9-signed-epi-and-phase-an-explicit-representation-test). A common internal basis across both
fibers is part of this observation map. Independent local basis rotations
would introduce corresponding edge connection matrices; plain phase
differences could not be retained unchanged.

Where both amplitudes `r_a=|z_a|` are positive, set
`z_a=r_a*exp(i*psi_a)`. The chain rule then derives

\[
\dot r_a=\frac{r_b\cos(\psi_b-\psi_a)-r_a}{3}-r_a,
\qquad
\dot\psi_a=\frac{r_b}{3r_a}\sin(\psi_b-\psi_a).
\]

For two nonzero amplitudes, `delta=psi_1-psi_0` obeys
`delta_dot=-(r_1/r_0+r_0/r_1)*sin(delta)/3`. Thus neighbor-coupled alignment
is a **derived response of internal scalar EPI structure**, rather than a
prescribed oscillator law or a feedback controller using telemetry. At zero
amplitude the linear coordinates remain regular but the corresponding polar
angle is unavailable. If only the neighboring amplitude vanishes, the local
rate remains defined by `rdot=Re(exp(-i*psi_a)*z_b)/3-4*r_a/3` and
`psi_dot=Im(exp(-i*psi_a)*z_b)/(3*r_a)` without assigning that neighbor an
angle. The induced `psi` is not automatically the original
fine phase, and the ratio in its equation is not a new primitive capacity.

The internal restoring term is also inherited. Restricting the fine
Dirichlet energy to the internal sector yields

\[
E_{\rm int}=\frac12\left(|z_0-z_1|^2+3|z_0|^2+3|z_1|^2\right).
\]

Its gradient with mobility `1/3` gives exactly the induced equation. In the
rational `(u,v)` chart the norm uses `G=diag(2,6)`, the inherited metric is
`3*G` in each fiber, and `3*G*dot c_a=-partial E_int/partial c_a`.
The local term is the energy of internal edges, not a selected polynomial
potential. On general fiber means, the full Dirichlet energy additionally
contains `3*(m_0-m_1)^2/2`, which decouples from this internal sector.

This derives a restoring contribution for **macro form**, not the missing
primitive-capacity potential `Psi` of the prior P2 test. At equal nonzero
`z_0=z_1`, internal relaxation still gives `rdot=-r`. The existing canonical
macro-P2 formula instead has zero pressure when scalar form, phase,
capacity and degree are uniform. Therefore the current scalar pressure
formula is **not preserved unchanged by this internal-mode reduction**.
The pressure `(z_b-z_a)-3*z_a` and inherited mobility `1/3` describe the
derived vector-form model; they are not installed as a replacement engine
law. Multiplying observations by `exp(t)` to erase the loss would conceal
physical relaxation, not derive maintenance.

Indeed, `S=|z_0|^2+|z_1|^2` satisfies

\[
\dot S=-2S-\frac23|z_0-z_1|^2.
\]

The sum and difference modes decay at rates one and `5/3`. Nonzero internal
structure is not sustained: `S(t)<=exp(-2*t)*S(0)`. The alignment response
does not imply a self-maintaining NFR, sustained oscillation or a particle.
Fine support, conductance, capacity and clock remain supplied premises.

Exact controls in
[test_internal_mode_pushforward.py](../../tests/physics/test_internal_mode_pushforward.py)
reuse the existing rational transport and energy owners. Rational `1/3`
belongs to this exact-real diffusion model, not an assertion that a binary64
mean, finite solver step or live event realizes it without residual.

### 12.6 What this changes in the research question

Three identifications remain distinct: a closed internal observation,
a macro nodal realization with inherited pressure and metric, and a
pattern maintaining nonzero internal amplitude indefinitely. The construction
establishes the first and an
explicit vector-form realization of the second. It disproves unchanged
scalar pressure inheritance here, and its decay identity excludes the third
within this fixed passive model. It does not exclude finite-lived coherent
identity or settle persistence of phase/topological structure. Reconstructing
the full fine tetrad also
needs any discarded means and primitive phase data on which its fields
depend. Internal polar coordinates alone are not a complete canonical triad
or diagnostic replacement. Section 13 completes that finite identification
gate; the single execution plan owns the remaining maintenance question.

## 13. Faithful macro state and tetrad inheritance on the retained prism

Keep exactly the fine model of section 12.5: unit `P2 square C3`, unit
capacity and EPI coefficient, fixed conductance and unit path lengths, with
other pressure channels disabled. The present statements concern the exact
real diffusion law and its stated observation maps. They do not replace
stored pressure with model pressure or promote binary64 capture to exact
arithmetic. Primitive phases are separately retained diagnostic inputs; no
new evolution law for them is assumed.

### 13.1 Four internal coordinates close, but omit a pressure direction

Let `c=(u_0,v_0,u_1,v_1)` be the four internal coordinates of section 12.5,
`P` their six-by-four lift, `R_int` their left-inverse projection, and define

\[
\mu=(m_0+m_1)/2,\qquad \delta=m_0-m_1,\qquad
s=(1,1,1,-1,-1,-1)^\top.
\]

Every fine form has the exact decomposition

\[
x=\mu\mathbf1+\frac{\delta}{2}s+Pc.
\]

The dynamics gives `mu_dot=0`, `delta_dot=-2*delta/3`, and the already
derived four-dimensional law for `c`. Consequently `(c,delta)` is a closed
five-coordinate state. Its lift with `mu=0` reconstructs centered EPI and
the full exact model pressure. The pressure map `-A` has rank five and
kernel `span(1)`; `c` omits the independent direction `s`. Thus one added
coordinate is necessary and sufficient for all-state **linear** pressure
reconstruction. The common mean is needed to recover absolute EPI but not
this pressure. This result concerns the entire nodewise pressure vector,
not merely its mean.

There is a simple bounded witness with identical `c=(1/4,0,1/4,0)` and
`mu=1/2`:

```text
x_A=(3/4,1/4,1/2,3/4,1/4,1/2), delta_A=0,
x_B=(15/16,7/16,11/16,9/16,1/16,5/16), delta_B=3/8,
p_B-p_A=-(1/8)*s.
```

Both states lie inside `(0,1)`. All four internal coordinates and their
derived angular/radial response agree, but the complete pressure differs.
This does not contradict their autonomous internal law: pressure sufficiency
is a stronger requirement than closure of those four observations.

The existing exact partition/memory owner independently gives closed
triangle means, `Hbar=9*I`, and `RAQ=QAP=K(0)=0`. Its supplied mean
coordinates cannot observe the four internal directions, and those directions
cannot drive the means in this fixed model. A zero memory kernel is therefore
not evidence that internal structure has disappeared or is reconstructed.
The current generic realization API already serves partition outputs; this
finite signed-coordinate proof needs no second realization engine.

### 13.2 Inherited metric, restoring pressure and coordinate dependence

In the full chart `(mu,u_0,v_0,u_1,v_1,delta)`, the pullback of `H=3*I` is

\[
H_{\rm chart}=\operatorname{diag}(18,6,18,6,18,9/2),\qquad
E_D=\frac32\delta^2+E_{\rm int}.
\]

Its inverse metric acting on the negative energy gradient recovers the
complete chart dynamics. The common mean has zero energy gradient. This
derives the coordinate mobilities once the fine metric, energy and chart
are specified; the nodal product identity alone still does not choose a
unique pressure/capacity factorization.

For the internal normalized complex coordinate `z_a=r_a*exp(i*psi_a)`,
the inherited metric where `r_a>0` is

\[
3\,|dz_a|^2=3\,dr_a^2+3r_a^2\,d\psi_a^2.
\]

Hence `rdot_a=-(1/3)*partial E_int/partial r_a` and
`psi_dot_a=-(1/(3*r_a^2))*partial E_int/partial psi_a`. These yield the
radial and angular equations already derived in section 12.5. Their
state-dependent angular coefficient has a geometric origin in the change
of coordinates; it is not an independently evolving primitive `nu_f`.
The singular polar metric at zero amplitude requires the regular Cartesian
internal coordinates there, not an invented angular value.

Retaining `(m_a,u_a,v_a)` in each triangle gives two vector-form units and
an invertible six-coordinate chart. It is an exact realization of the fine
law, without a reduction of its scalar state dimension. Retaining only
`(c,delta)` discards the conserved common mean and therefore is not a
faithful representation of absolute EPI or every canonical operator contract.

### 13.3 The potential kernel must be inherited too

On the unit prism, every node has three neighbors at distance one and two
other nodes at distance two. Let `W` be its unit adjacency and `J` the
all-ones matrix. The actual inverse-square potential kernel is

\[
K=\frac{3W+J-I}{4},\qquad \Phi_s=Kp.
\]

Its eigenvalues are `7/2` on the common mean, `1/2` on `s`, `-1/4` on the
two common internal directions and `-7/4` on the two opposed internal
directions. None vanishes. In particular `K*(-A)` has rank five, so the
same five linear EPI coordinates are necessary and sufficient to reconstruct
the entire **model** potential on this fixed metric. The bounded witness
above gives `Phi_B-Phi_A=-s/16`:

```text
Phi_A=(1/16,-1/16,0,1/16,-1/16,0),
Phi_B=(0,-1/8,-1/16,1/8,0,1/16).
```

These displayed fields use exact model pressure. On the second fixture the
actual fresh binary64 pressure has residual
`epsilon*(1,0,0,0,-1,0)`, `epsilon=2^-54`; its stored fresh-pressure
potential consequently differs from `Phi_B` by
`(epsilon/4)*(-1,0,3,0,1,-3)`. The portable controls retain these measured
defects. They do not present exact model identities as zero-defect runtime
closure.

The stronger issue is that exact nodal-flow reduction does not imply that
recomputing the same field formula on the coarse graph preserves its meaning.
Let `R` average each triangle and `T` lift two constants. Exact multiplication
gives

\[
\bar K=RKT=
\begin{pmatrix}2&3/2\\3/2&2\end{pmatrix},\qquad
RK=\bar K R.
\]

The diagonal two is inherited from two distinct fine neighbors within the
same triangle. It is not a self-interaction inserted into the fine graph.
For pure EPI, `Rp` is antisymmetric and `R*Phi_s=(1/2)*Rp`. The mean
quotient has inherited capacity `1/3` and canonical scalar pressure
`p_macro=(m_1-m_0,m_0-m_1)=3*Rp`, so

\[
R\Phi_s=\frac16p_{\rm macro}.
\]

By contrast, computing the usual self-excluded potential directly on a
two-node macro graph with positive distance `ell` gives
`Phi_macro=-p_macro/ell^2`. Its sign is opposite for nonzero contrast;
no positive length fixes the mismatch. Using `Rp` instead of `p_macro`
still gives the wrong sign. To preserve the averaged fine observation with
canonical macro pressure units, the inherited source kernel is `Kbar/3`.
This proves a failure of unchanged scalar potential inheritance while
supplying the correct observation map. It does not change the canonical
fine-graph potential definition.

### 13.4 Full tetrad dependency and representation boundary

| Fine observation | Required information on the declared prism | Result of the identification gate |
| --- | --- | --- |
| `Phi_s` | Actual pressure and the fine path-length kernel; model pressure is reconstructible from `(c,delta)` | Four internal coordinates are insufficient; the inherited aggregation kernel differs from a new scalar macro graph. |
| Phase gradient | Primitive fine phases and support neighbors | Internal EPI orientation does not supply these data. Even fiber-constant phases have different fine and macro normalization. |
| Phase curvature | Primitive fine phases, circular resultants and availability | Nonlinear neighbor resultants must remain defined; a modal angle does not replace primitive phase. |
| `xi_C` | Static coherence products derived from pressure, path distances and fit policy, or a separately identified spectral fallback | This unit prism has only two positive distance bins; the fit requires at least three. The read-out uses `spectral_gap`, with ideal scale `sqrt(3/2)`, not a fitted coherence-correlation length. |

For a direct phase witness, hold full EPI and unit capacity fixed and set
primitive phases to zero in one triangle and `pi/3` in the other. Pure-EPI
pressure, potential and all internal modal coordinates stay unchanged.
Every edge satisfies strict U3. The fine phase gradient is `pi/9` at every
node and curvature is `-atan(sqrt(3)/5)` in the first triangle and its
opposite in the second; all resultants are nonzero. With consensus primitive
phase both fields are zero. A scalar macro P2 with those two phases instead
has gradient `pi/3` and curvature `(-pi/3,pi/3)`. Internal neighbors matter
to the inherited diagnostics even though the mean EPI law closes exactly.
This pressure independence uses the declared zero phase-channel coefficient.

The spectral fallback for `xi_C` is identical across these pressure/phase
witnesses because the held graph is unchanged. It supplies no evidence that
a fitted correlation length is generally pressure-independent. Retaining
all primitive phase data is sufficient for the phase-sector read-outs; no
minimal phase reconstruction theorem is claimed. When those data are fixed
by preparation, that restriction must accompany the reduced state.

Pressure freshness and arithmetic remain separate. Model pressure can be
reconstructed from five coordinates in exact arithmetic; an arbitrary stored
pressure cannot. Changing stored pressure alone changes the corresponding
potential without changing those coordinates. Actual fresh binary64 pressure
can also differ from the exact-real model and must retain its measured
realization residual. Exact-model sufficiency is not a proof of runtime
state compression, finite-step intertwining or the future tetrad.

The portable controls share the fixture in
[_internal_mode_fixture.py](../../tests/physics/_internal_mode_fixture.py),
and reuse the existing transport, exact linear algebra, partition/memory,
pressure-capture and field owners:
[macro-state proof](../../tests/physics/test_internal_mode_macro_state.py) and
[actual tetrad observations](../../tests/physics/test_internal_mode_tetrad.py).
The preceding nine internal-mode controls retain their behavior after fixture
centralization. No new evolution law or certificate API is introduced.

The finite gate is complete: a closed internal response, a minimally extended
exact pressure state and inherited diagnostic maps are distinguished. The
construction still supplies neither a dynamically selected partition nor
autonomous nonzero maintenance. Restoring the missing coordinate or correct
potential kernel cannot overcome the passive loss proved in section 12.5.

## 14. Causal support versus changing geometry

The target of this retained branch is nonzero **internal nonuniform form**, measured by
`S=|z_0|^2+|z_1|^2` in the fixed coordinates of section 12.5. A conserved
common EPI mean is not that target; neither is this target a definition of
every NFR. This section separates three questions:
whether changing conductance can feed this form, whether canonical source
pressure can compensate its loss, and whether the source itself has a closed
endogenous evolution. The first two admit scoped answers; the third remains
open. The existing transport, regional-balance and forcing owners suffice.

### 14.1 Positive geometry alone cannot supply internal amplitude in this family

Keep the same six-node prism, partition and fixed internal basis. Give every
triangle edge a common conductance `a(t)>0`, every matching cross edge a
common conductance `b(t)>0`, and every node the same capacity `nu(t)>0`.
Keep a fixed pure-EPI pressure coefficient `e>0`, with no other channels.
Explicit edge lengths remain one; conductance is not a changing path metric.
With `d=2a+b`, projection of the fine nodal equation gives exactly

\[
\dot z_0=\frac{e\nu}{2a+b}\{bz_1-(3a+b)z_0\},\qquad
\dot z_1=\frac{e\nu}{2a+b}\{bz_0-(3a+b)z_1\},
\]

\[
\dot S=-\frac{2e\nu}{2a+b}
  \left(3aS+b|z_0-z_1|^2\right)<0\quad(S>0).
\]

There is no `a_dot` or `b_dot` in this fixed-coordinate norm. The identity
holds along any admitted coefficient history, including state-dependent
ones that retain these symmetries. It supplies no law selecting that history.
Arbitrary edgewise changes, heterogeneous capacity, changing partitions,
support events and additional pressure channels are outside this result.

Pointwise positivity alone does not prove that the form vanishes at infinite
time. In the fixed orthonormal sectors `z_+=(z_0+z_1)/sqrt(2)` and
`z_-=(z_0-z_1)/sqrt(2)`, the positive rates are

\[
k_+=\frac{3e\nu a}{2a+b},\qquad
k_-=\frac{e\nu(3a+2b)}{2a+b},\qquad
z_\pm(t)=z_\pm(0)\exp[-I_\pm(t)],\quad
I_\pm(t)=\int_0^t k_\pm(\tau)\,d\tau.
\]

For locally integrable rates, each occupied sector vanishes exactly when
its own cumulative rate diverges. For one fixed supplied coefficient history,
all internal initial vectors decay if and only if `I_+(infinity)=infinity`,
because `k_->k_+`. For state-dependent coefficients this criterion must be
applied along each solution, not transferred from one history to all others.

For example, the **supplied counter-history** `a=exp(-t)`, `b=nu=e=1`
has `I_+(infinity)=(3/2)*log(3)` and `I_-(infinity)=infinity`.
Its common internal sector retains amplitude factor `3^(-3/2)` and squared
amplitude factor `1/27`. It loses effective internal transport asymptotically;
this passive remnant is not derived active maintenance or recovery. No
conductance law or numerical trajectory is proposed by this counterexample.

### 14.2 Geometry work can increase energy while the form shrinks

The existing moving-conductance identity is

\[
\dot E_D=(Bx)^\top\dot x+\frac14\sum_{ij}\dot W_{ij}(x_i-x_j)^2.
\]

Its second term changes the energy assigned to a given form. It need not
feed the internal coordinates. On the unit prism, take
`c=(1/4,0,1/4,0)`, both means `1/2`, and `e=nu=1`. At that state
`S=1/4`, `S_dot=-1/2` and `E_D=3/8`. A declared coefficient jet
`a_dot=3`, `b_dot=0` gives

```text
nodal_work       = -3/4,
conductance_work =  9/8,
E_D_dot         =  3/8 > 0,
S_dot           = -1/2 < 0.
```

`observe_support_transport_derivative` supplies this exact balance on a
detached model-pressure snapshot. It does not execute or justify the supplied
coefficient jet. The sign contrast can also persist over a bounded interval:
under the declared history `a=1+3t`, `b=nu=e=1`, `0<=t<=1/12`, the same
common internal sector has
`S_dot/S=-6a/(2a+1)<=-2`, whereas
`E_D_dot/E_D=3/a-6a/(2a+1)>=9/35>0`.
Thus a positive geometry-energy budget alone cannot establish maintenance
of the actual form, even while coefficients stay positive and bounded.

### 14.3 Canonical phase pressure can compensate the loss instantaneously

Return to unit conductance and unit common capacity. For each triangle
let `y_i=x_i-m_a`, so `S=sum_i(y_i^2)`. Capture canonical non-EPI forcing
`F` before evaluating any rate; never reconstruct it from a desired answer.
Write the stored pressure as `p=e*g_epi+F+epsilon`, where `epsilon` is
the sum of the captured pressure-assembly defect and stored-pressure
residual. The exact represented-state budget is

\[
\dot S=-2eS-\frac{2e}{3}|z_0-z_1|^2
       +2\sum_i y_i F_i+2\sum_i y_i\epsilon_i.
\]

This is the nodal rate evaluated at the captured state, not a completed
numerical step. The two existing regional observers give
`sum(E_region)=3*S/2`; multiplying their summed variance-rate balance by
`2/3` reproduces all three terms. Both defect sources remain visible.
The degree-three topology gradient vanishes, and uniform capacity has zero
capacity gradient. If primitive phase and capacity are constant inside each
triangle, every non-EPI source channel is fiberwise constant and its internal
projection vanishes. Multiplying by fiberwise-constant capacity preserves
that statement, although the displayed simple `S` budget requires unit
common capacity. A single collective phase per triangle cannot supply this
internal drive. The zero mean/internal memory kernel of section 13.1 does
not hide a sustaining source either.

A nonuniform internal primitive phase gives a positive witness. Prepare

```text
e=1/2, w_phase=1/4, w_vf=1/4, w_topo=0,
nu_i=1, m_0=m_1=1/2, c=(1/16,0,1/16,0),
theta=(0,pi/3,pi/6) in each triangle.
```

Every support edge satisfies strict U3. Each node sees one copy of each
phase, with nonzero phasor resultant and ideal direction `pi/6`, so the
ideal phase channel is `(1/6,-1/6,0)` in each triangle. Consequently

\[
S=1/64,\qquad \dot S_{\rm passive}=-1/64,\qquad
\dot S_{\rm source}=1/48,\qquad \dot S=1/192>0.
\]

The actual pressure-capture owner retains represented source coefficients
and their exact projected work. Regression controls evaluate the represented
phasor arithmetic independently and distinguish it from the ideal `1/48`
source work and `1/192` total rate; a historical binary64 fraction is not a
portable model constant. The retained witness has positive source work above
passive loss. Pressure assembly and stored-pressure defect work are separate
from the phase-arithmetic difference. The ideal source sums to zero: internal
redistribution can increase internal amplitude without increasing the common
mean; the represented mean contribution must be retained independently. This is evidence
for instantaneous source compensation under declared coefficients, not a
law sustaining that phase arrangement or a self-maintaining NFR trajectory.

### 14.4 Capacity and causal closure remain explicit dependencies

Within-triangle capacity variation breaks the four-coordinate autonomous
law. For capacities `(1,2,1,1,1,1)`, adding an EPI constant `delta` only
to the first triangle leaves `c` unchanged but changes its exact model rate by
`(e*delta/6,-e*delta/18,0,0)`. The hidden mean enters through `diag(nu)`.
At `e=1/2`, `delta=3/16`, this is `(1/64,-1/192,0,0)`. A capacity extension
must therefore retain enough state and derive its law; it cannot inherit
the uniform-capacity closure by assumption.

The source-compensation result localizes the missing mechanism: a closed
evolution must generate or preserve the required internal primitive source
structure and account for its response as EPI changes. The internal modal
angle derived from passive EPI is still not primitive phase. Existing phase
steppers declare additional angular-rate and coupling premises; their
availability does not derive those premises from the EPI equation. Neither
holding the prepared phase forever nor choosing a restoring gain would
resolve that origin question. No complete autonomous candidate or maintenance
trajectory is admitted here. This is not a general nonexistence result for
multichannel or hybrid TNFR dynamics.

Portable controls reuse the shared fixture and existing physics owners:
[geometry and exposure](../../tests/physics/test_internal_mode_geometry_support.py),
[source and capacity budgets](../../tests/physics/test_internal_mode_source_support.py).
No production evolution rule, feedback controller, energy implementation or
certificate API is added. The single execution plan owns the next gate.
