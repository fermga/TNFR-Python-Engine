# Exact and controlled form-phase reductions

Equivalent form/phase/memory representations, controlled slow-time comparison and its full-state capture handoff; hidden initialization and transient errors remain explicit.

Part of [Sine pattern geometry and dissipative capture](SINE_PATTERN_DYNAMICS.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 28. Exact form, phase and memory representations

<a id="sine-form-phase-memory-equivalence"></a>

The completed composition, capture and prepared-entry results share more
than their graph. They use the same reciprocal exchange, conserved means
and phase potential. The following identities connect those results to
the existing memory and resonance owners without introducing a new
primitive coordinate, deleting the initial form information or assuming
a slow phase approximation.

Retain finite connected simple unit support with at least two nodes, positive
held capacities, `e,w,beta>0`, the declared clock and no forcing or events. Use
`A=KL`, `M=K^-1`, `W=1^T M1`, and the weighted centering projector
`P_M=I-1*(1^T M)/W`. The means `mu_x` and `mu_theta` are those of the
actual state; the latter belongs to a chosen continuous phase lift.
Uncertain relative sources retain one conserved pair of means per member,
with their absolute common origins unobserved. Neither mean is replaced
by a nominal value in the identities below.

### The sum coordinate retains the full joint state

In [Section 27's variables](SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry) `tau=e*t`, `z=(b/e)P_Mx` and
`eta=ab/e^2`, define the mixed coordinate `y=theta+z`. The exact rows become

\[
\boxed{\qquad
\theta'=A(y-\theta),\qquad y'=\eta KS(\theta).
\qquad}
\]

This change is invertible on consistent real lifts:
`z=y-theta`, `x=mu_x*1+(e/b)z`, with `1^T Mz=0`. Thus it retains both
consumed coordinates and the conserved form mean. A change of circular
representative sends `(theta,y)` to `(theta+2*pi*m,y+2*pi*m)` for the
**same** integer vector `m`; their real difference is unchanged.
Equivalently, retain real `z` and circular `exp(i*y)`, from which
`exp(i*theta)=exp(i*y)*exp(-i*z)` is reconstructed.

Treating `theta` and `y` as two independently wrapped phase vectors loses
information. On the unit P2 with unit capacities, `K=I`. The preparations
`theta=0,z=0` and `theta=0,z=(2*pi,-2*pi)` give the same two separately
wrapped phase vectors, but their scaled phase rates are respectively
`0` and `A z=(4*pi,-4*pi)`. The real form contrast has been discarded by
that wrapping, although it is consumed by the phase row.

Nor does `y` alone close the dynamics. On the same P2, `y=0` can arise
from `theta=z=0`, giving `y'=0`, or from
`theta=(delta,-delta),z=(-delta,delta)`, giving
`y'=eta*(-sin(2*delta),sin(2*delta))`. For `0<delta<pi/4` these rates
differ even on the same conserved-mean leaf. A first-order law depending
only on `y` therefore needs an additional approximation or restricted
invariant family.

The cancellation behind this coordinate is already used operationally in
the [hidden-state inverse](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability):
`dot(x)+(e/b)*dot(theta)=aKS` removes the form-gradient contribution
from paired rate observations. Its time derivative supplies the paired
acceleration channel there. The present transformation and the entry
estimate reorganize that same complete-law identity; they do not select
a new pressure law from those observations.

### Exact elimination gives second-order phase and retained memory

In the original clock, differentiate the complete phase row and use the
form row. The following second-order identity and velocity reconstruction
also hold at `e=0`, with positive `w,beta` and held positive capacities;
they do not use the preceding change of variables dividing by `e`.
With the actual ordered matrix products this gives

\[
\boxed{\ddot\theta+eA\dot\theta=abAKS(\theta),\qquad
\dot\theta(0)=bAP_Mx(0).}
\]

The constraint `1^T M dot(theta)=0` must remain. An unconstrained
second-order equation on all phase lifts would also permit a constant
common phase velocity, which the original positive-capacity law does not
supply. On the weighted-mean-zero space, `A` is invertible and the phase
velocity reconstructs the full centered form as
`P_Mx=b^(-1)A^(-1)dot(theta)`. The constant form mean must be retained
separately if the original absolute form is needed. This is a
same-information representation, not a removal of a state variable.

<a id="form-reconstruction-scope"></a>

#### What an emergent-form hypothesis must distinguish

Formation of an organized pattern from joint evolution is different from
an all-state instantaneous identity `x=F(theta,nu,support,p)`. The latter
does not hold for the admitted conservative sine law. On unit P2, take
`e=0,w=beta=nu_1=nu_2=1` and `theta=(0,0)`. The states `x=(0,0)` and
`x=(q,-q)`, with nonzero real `q`, have the same phase, capacity, support,
form mean and instantaneous pressure `p=(0,0)`. Their phase rates are
respectively zero and `(2q/pi,-2q/pi)`. The common observed coordinates
therefore do not determine form or the ensuing response.

Phase **velocity**, together with the original form mean, supplies the
missing information through the inverse above. This uses the independently
specified phase law, not the nodal identity alone. At zero loss its exact
history representation is

\[
\theta(t)=\theta(0)+btAP_Mx(0)
       +ab\int_0^t(t-s)AKS(\theta(s))\,ds.
\]

The initial form remains an independent source, or equivalently the initial
phase velocity. This equation does not derive it from phase alone. At
positive loss, pressure computed from the full state can encode form
differences; that does not make such pressure an independent measurement or
remove the state information needed to compute it. A special invariant or
controlled slow family could restrict form in terms of other variables, but
its preparation, domain and error require separate proof. Neither global
reconstruction nor impossibility of every restricted family follows.

Thus the supported synergy is a same-information description of relative
form as phase-motion information. The stronger claim that primitive form
emerges from the other instantaneous parameters remains unestablished.
The [channel-geometry analysis](RESONANCE_FOUNDATIONS.md#structural-channel-distinction)
keeps this state question separate from unequal response of the two channels.

Alternatively, variation of constants and integration give the exact
nonlinear Volterra equation for `e>0`

\[
\boxed{\begin{aligned}
\theta(t)={}&\theta(0)
+\frac be\left(I-e^{-eAt}\right)P_Mx(0)\\
&+\frac{ab}{e}\int_0^t
\left(I-e^{-eA(t-s)}\right)KS(\theta(s))\,ds.
\end{aligned}}
\]

The source term is the retained initial form, not an external input.
The phase current in the integral uses the actual earlier phase state;
this is not a fixed linear convolution in `theta`. Differentiating the
identity with its supplied initial state recovers the second-order row,
and reconstructing form recovers the original joint system. Global
Lipschitz continuity of the full sine field supplies uniqueness. The
projector could be omitted only from the first source term because
`I-exp(-eAt)` annihilates the common form mode; displaying it makes the
mean accounting explicit.

No commutation of `K` with `L` is used. In particular `AK=KLK`, whereas
`KA=K^2L` is generally different. The corresponding phase-velocity
memory operator is

\[
\mathcal R_e(s)=A e^{-eAs}K
 =K^{1/2}B e^{-eBs}K^{1/2},\qquad
B=K^{1/2}LK^{1/2}.
\]

It is symmetric positive semidefinite for each nonnegative lag, but its
individual entries need not be nonnegative. Spectral integration gives

\[
\int_0^\infty\mathcal R_e(s)\,ds
=\frac1e\left(K-\frac{\mathbf1\mathbf1^{\mathsf T}}W\right).
\]

Acting on a sine-current vector, whose ordinary sum is zero, this
integrated operator gives `KS/e`. This explains the frozen-history gain
behind a candidate slow phase law; it does **not** bound the error made
by replacing the moving history by its current value. It also does not
remove the initial source or justify exchanging an infinite-time limit
with a constitutive limit.

The [single-intermediary sine owner](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)
already derives a nonlinear second-order representation and derivative-free
Volterra memory, including the hidden initial state and moving visible
boundary. The [general elimination owner](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state)
owns the variation-of-constants principle and its source obligation.
The identities here specialize that principle to eliminating the entire
form coordinate of this complete nonlinear sine model. They neither turn
the existing signed intermediary kernel into a positive-entry kernel nor
transfer a pure-diffusion or conservative-memory conclusion between models.

### The nonlinear storage representation contains the existing resonance pencil

Remove the conserved origins, and put

\[
h=K^{-1/2}\mathbf1,\quad\mathcal Q=h^\perp,\quad
\xi=K^{-1/2}P_Mx,\quad
\vartheta=K^{-1/2}P_M\theta,\quad
\Phi(\vartheta)=V(K^{1/2}\vartheta).
\]

The symmetric matrix `B` above is positive definite on `Q`; all inverse
matrices in this paragraph act only there. Since
`grad(Phi)=-K^(1/2)S`, the exact nonlinear transformed rows and their
equivalent second-order equation are

\[
\dot\xi=-eB\xi-a\nabla\Phi(\vartheta),\qquad
\dot\vartheta=bB\xi,\qquad
B^{-1}\ddot\vartheta+e\dot\vartheta
+ab\nabla\Phi(\vartheta)=0.
\]

Their storage and loss are still the original quantities:

\[
b^2E=\frac12\dot\vartheta^{\mathsf T}B^{-1}\dot\vartheta
+ab\Phi(\vartheta),\qquad
\frac{d}{dt}(b^2E)=-e\|\dot\vartheta\|^2.
\]

The velocity contribution is the original form storage in different
coordinates. The derived quotient matrix `B^-1` depends on the graph and
held capacities; this supplies no identification of capacity with inverse
physical mass. Initial velocity, nonlinear phase history and the conserved
origins remain part of the model.

At a critical phase geometry with Hessian `H`, the Hessian of `Phi` is
`C=K^(1/2)HK^(1/2)`. Linearizing the displayed equation gives precisely
the [existing full-support resonance pencil](RESONANCE_FOUNDATIONS.md#resonance-tangent),
`B^-1*ddot(vartheta)+e*dot(vartheta)+ab*C*vartheta=0` for perturbations.
The matrices `B` and `C` need not commute, so scalar cycle discriminants
still cannot classify a general attachment. Strict acute-sector convexity,
equilibrium recovery and the full storage capture barrier use the same
phase curvature; the nonnegative velocity/form term is why a phase-only
storage check cannot replace the capture certificate.

### A phase-flat preparation distinguishes the full law from phase descent

Take equal initial phases and any nonconstant initial form on the retained
connected graph. Then `S(0)=0`, `H(0)=L` and

\[
V(0)=\dot V(0)=0,\qquad
\ddot V(0)=b^2(Ax(0))^{\mathsf T}L(Ax(0))>0,
\qquad
\dot E(0)=-e(Lx(0))^{\mathsf T}K(Lx(0))<0.
\]

For strict positivity, `Ax(0)` has zero weighted mean. If it were a
constant vector, it would vanish; positivity of `K` and connectedness
would then force `x(0)` constant, contrary to preparation. Thus phase
potential initially increases while total storage decreases. The form
reservoir and reciprocal phase row account for both signs under one law.

In contrast, the candidate phase-gradient flow `dpsi/dsigma=KS(psi)`
initialized at that same uniform `theta(0)` remains there by uniqueness.
It would discard the prepared form information. Initializing a proposed
comparison from `theta(0)+z(0)` retains that information, but the exact
identities alone do not prove closeness to this comparison flow. Section 29
derives the finite-horizon error bound while retaining the initial transient.
A later capture handoff, support creation and physical identification remain
separate obligations.

The [full-law equivalence controls](../../tests/physics/test_relational_sine_exchange_equivalence.py)
differentiate the shared field with interval jets, check the two opposite
storage signs and the failure of a sum-coordinate-only closure, and retain
noncommuting capacity/Laplacian products. Independent numerical quadrature
checks the nonlinear memory identity along a complete trajectory; it is an
equivalence check, not a reserved-response prediction or the analytic proof.

## 29. A controlled fast-form and slow-phase comparison

<a id="sine-controlled-slow-phase"></a>

The exact representations in Section 28 permit a quantitative comparison
with a first-order phase flow. This is an approximation theorem for the
same complete sine law, with an explicit transient and error. It does not
install that reference flow as a replacement pressure or remove the
prepared form information.

### Complete state, reference initialization and clocks

Fix the finite connected unit support, positive held `K`, positive `beta`
and positive complete-law coefficients `e,w`. Retain Section 27's
`M=K^-1`, `A=KL`, `alpha=b/e`, `z=alpha*P_Mx`, `tau=e*t` and
`eta=ab/e^2=beta*alpha^2`. For the actual initial state write `z_0=z(0)`
and take a proved bound `Z>=||z_0||_M`. Define the slow time and reference
on consistent phase lifts by

\[
\sigma=\eta\tau=\frac{ab}{e}t,\qquad
\frac{d\psi}{d\sigma}=f(\psi),\qquad
f(\theta)=KS(\theta),\qquad
\psi(0)=\theta(0)+z_0.
\]

Both the complete and reference fields are globally Lipschitz, so their
solutions exist uniquely for all finite times. The reference is initialized
with the actual retained form contribution, not only the initial phase.
As in Section 28, the mixed coordinate uses real `z_0` and consistent
lifts; wrapping its two summands independently would discard information.

Let `lambda>0` be a certified weighted quotient gap for `A`, and set

\[
F=\left(\sum_i\nu_i d_i\right)^{1/2},\qquad
\ell=2\max_i\nu_i.
\]

The earlier forcing bound gives `||f(theta)||_M<=F` globally. Its derivative
is `-KH(theta)`, where `H` is the cosine-weighted phase Hessian. The edge
formula gives `-B<=K^(1/2)H(theta)K^(1/2)<=B` for
`B=K^(1/2)LK^(1/2)`. The similar matrix `KL` has Gershgorin intervals
`[0,2*nu_i]`, so the symmetric matrix `B` has norm at most `ell`.
Integrating the derivative
along a segment therefore proves the global Lipschitz estimate
`||f(u)-f(v)||_M<=ell*||u-v||_M`. This bound needs no acute phase domain
for either path or the segment between them.

### Explicit composite, phase and form bounds

For a supplied finite `sigma>=0`, put `tau=sigma/eta` and
`D(tau)=exp(-A*tau)`. Define

\[
\boxed{\begin{aligned}
R(\tau)&=\frac{\eta F}{\lambda}
                  \left(1-e^{-\lambda\tau}\right),\\
C(\sigma)&=
\frac{\eta(\ell Z+F)}{\lambda+\eta\ell}
           \left(e^{\ell\sigma}-e^{-\lambda\tau}\right).
\end{aligned}}
\]

Then the exact complete solution and its reference satisfy

\[
\boxed{\begin{aligned}
\|z(\tau)-D(\tau)z_0\|_M&\le R(\tau),\\
\|\theta(\tau)+D(\tau)z_0-\psi(\sigma)\|_M&\le C(\sigma),\\
\|\theta(\tau)-\psi(\sigma)\|_M
&\le Ze^{-\lambda\tau}+C(\sigma).
\end{aligned}}
\]

The middle quantity is the **composite correction**. It retains the fast
initial form contribution instead of calling the corrected angle the
actual phase. For `y=theta+z`, the same estimates also give
`||y(tau)-psi(sigma)||_M<=C(sigma)+R(tau)`.

The form estimate follows from the exact variation-of-constants identity

\[
z(\tau)=D(\tau)z_0+
\eta\int_0^\tau D(\tau-s)f(\theta(s))\,ds.
\]

All integrand vectors have zero weighted mean. Thus the quotient semigroup
bound `||D(s)||_M<=exp(-lambda*s)` applies to them and gives `R`.
Section 28's exact phase memory, now in scaled time, gives

\[
\theta(\tau)+D(\tau)z_0
=\theta(0)+z_0+
\eta\int_0^\tau\left[I-D(\tau-s)\right]f(\theta(s))\,ds.
\]

Subtract the reference integral equation. If
`q(tau)=theta(tau)+D(tau)z_0-psi(eta*tau)`, the forcing and Lipschitz
bounds imply

\[
\|q(\tau)\|_M\le
\eta\ell\int_0^\tau\|q(s)\|_M\,ds+
\frac{\eta(\ell Z+F)}{\lambda}
                \left(1-e^{-\lambda\tau}\right).
\]

Here the `ell*Z` term retains the initial transient inside
`theta-psi=q-D(s)z_0`; the `F` term bounds the remaining memory integral.
The scalar comparison function with zero initial value solves
`c'=eta*ell*c+eta*(ell*Z+F)*exp(-lambda*tau)`. Its explicit solution is
the displayed `C`, proving the composite bound by Gronwall's inequality.
The uncorrected phase bound then follows by the triangle inequality.
No phase linearization, commuting `K,L` assumption or sampled response
enters this argument.

Conversion back to the original form coordinate is also explicit:

\[
\left\|P_Mx(t)-e^{-eAt}P_Mx(0)\right\|_M
\le\frac ae\frac F\lambda
              \left(1-e^{-\lambda\tau}\right)
=\sqrt{\beta\eta}\frac F\lambda
              \left(1-e^{-\lambda\tau}\right).
\]

Thus the original form remainder after its homogeneous transient is
`O(sqrt(eta))` at fixed support, `K` and `beta`. A small scaled `z`
remainder alone would not establish this original-coordinate statement.
The homogeneous form term remains until its own transient has decayed.

### Uniform finite-time meaning and the retained storage budget

Fix a finite slow-time horizon `Sigma`, the support, `K`, `beta` and a
uniform bound `Z` on the scaled initial form. For `0<=sigma<=Sigma`,

\[
C(\sigma)\le
\eta\frac{\ell Z+F}{\lambda}e^{\ell\Sigma}.
\]

This is a uniform `O(eta)` composite comparison including the initial
instant. It is not uniform `O(eta)` proximity of the actual phase from
that instant: exactly `theta(0)-psi(0)=-z_0`. For any fixed
`sigma_0>0`, the uncorrected estimate on `[sigma_0,Sigma]` instead has
the additional transient `Z*exp(-lambda*sigma_0/eta)`. More generally,
the finite displayed inequalities decide whether a chosen transient
is sufficiently small, without retuning the requested horizon.

The original-clock comparison time is

\[
t=\frac{\sigma}{e\eta}
  =\beta\pi^2\frac e{w^2}\sigma.
\]

As `eta` changes, the constitutive ratio changes; this is not just a common
clock rescaling. Keeping a nonzero scaled preparation fixed also retains
the original initial storage

\[
E(0)=\frac{\beta}{2\eta}z_0^{\mathsf T}Lz_0
                       +\beta V(\theta(0)).
\]

Its form term grows as `1/eta`. For uncertain preparations the expression
and its enclosing budget apply to every actual member, rather than to its
nominal source alone.

### Means, uncertain sources and phase potential

Since `1^T M f=1^T S=0`, the reference conserves its weighted phase mean.
Its initialization has the same weighted phase mean as the complete
trajectory because `z_0` is centered. Common form shifts cancel from
`z_0`; common phase shifts move the actual and reference solutions
together. All comparison differences and the composite correction have
zero weighted mean on the selected lift.

For a `SineRelativePattern`, reuse Section 27's original residual set and
its weighted initial norm bound. Each member is compared with **its own**
reference initialized at that member's `theta(0)+z_0`. A uniform `Z`
makes the displayed error uniform over the family but does not replace
these references by a single nominal reference trajectory. Doing that
would require a separate bound on the initialization differences.
Unknown absolute origins and memberwise conserved means remain explicit.

The potential satisfies the global bound
`|V(u)-V(v)|<=F*||u-v||_M`, since
`||grad(V)||_(M^-1)=||KS||_M<=F`. Hence

\[
|V(\theta(\tau))-V(\psi(\sigma))|
\le F\left[Ze^{-\lambda\tau}+C(\sigma)\right].
\]

The reference itself obeys
`dV(psi)/dsigma=-S(psi)^T K S(psi)<=0`. This does not make the actual
phase potential monotone: Section 28's phase-flat, nonconstant-form
preparation has strictly positive initial second phase-potential derivative
while total storage decreases. Initializing the reference only at that
uniform phase would leave it at consensus and discard the preparation
mechanism; the shifted initialization and explicit transient resolve that
mismatch in the comparison theorem.

### Shared certificate and bounded numerical controls

[`bound_sine_slow_phase`](../../src/tnfr/physics/relational_sine_reduction.py)
accepts an admitted exact or relative source and a supplied rational slow
time. It reuses the complete-law preparation admission, weighted norm and
exact reversible gap owners, and returns outward analytic comparison
bounds. The scaled and original times are enclosed directly from their
mathematical-pi formulas; these are enclosures of the declared comparison
instant, not a new class of independently uncertain times. The reader runs
neither the full trajectory nor the reference phase solver. It does not
issue an endpoint capture or final-basin verdict.

The [reduction tests](../../tests/physics/test_relational_sine_reduction.py)
freeze a small heterogeneous support with edges
`(0,1),(1,2),(2,3),(0,2)`, capacities `(1,3/2,2,5/4)` and `beta=2`.
The primary constructor weight ratio is `16:1`; the stored normalized
coefficients remain authoritative, with exact ratio `w/e=1/16`.
The declared preparation is
`x=(beta*e/w)*(1/2,-3/4,5/4,-1)` and
`theta=(1/8,-1/4,1/2,-3/8)`, with slow horizon `1/16`.
A separately fixed `8:1` comparison retains the same scaled initial form
and checks the feedback-parameter dependence. Independent SciPy integrations
of the complete and reference rows cross-check the inequalities under the
test's explicit numerical settings; they are not validated trajectory
enclosures or a search over sources, ratios or horizons. The analytic
certificate does not depend on their sampled responses.

The error grows with the declared slow horizon and is not an infinite-time
theorem. It proves neither a final basin match nor an exchange of the
limits `eta->0` and `t->infinity`. Section 30 supplies a separate capture
criterion that admits the actual full-state endpoint, including the form
remainder and all sector margins. The result controls a mechanism within the supplied law
and preparation; it does not select that law, support, source budget or
physical interpretation.

### A conditional comparison using acute geometry and dissipation

The [two-port transit proof](SINE_TWO_PORT_TRANSIT.md#sine-two-port-transit)
derives a different finite-horizon estimate for its unit-capacity, beta-one
model. On a common acute chart, the slow phase field is dissipative in the
degree metric. Integrating the complete law's storage loss bounds the
squared scaled-form norm over time, yielding a comparison error that grows
with the square root of the slow horizon. A strict first-exit argument
must establish that both the actual and mixed phases stay in that chart.
The nominal reference also retains its explicit initialization mismatch
against every actual source member.

That conditional argument supplies a finite deformation result with the
original form coordinate retained. It does not replace the global,
possibly nonacute comparison above or establish eventual capture. Its
model-specific coefficients and prepared phase origins do not transfer
without admission of the new complete model.

## 30. Full-state capture from controlled phase geometry

<a id="sine-slow-phase-capture-handoff"></a>

Section 29 bounds the difference from a reference flow but does not by
itself locate that reference at the requested endpoint. A sufficient
handoff needs both a justified reference neighborhood and the remaining
original form storage. The following construction supplies these from the
admitted preparation and an exactly verified stationary phase geometry.
It requires neither a supplied response endpoint nor a reference solver.

### A proved reference neighborhood from the preparation

Retain all hypotheses and notation of Section 29. Supply exact rational
node turns `q_i`, in the source node order, and put `phi_*=2*pi*q` on the
selected real lift. Require every principal target edge angle
`delta_*e` to lie strictly between `-pi/2` and `pi/2`, and require
`S(phi_*)=0`. The shared target owner verifies these hypotheses using the
[exact acute geometry and sine cancellation](../../src/tnfr/physics/relational_sine_recovery.py);
a floating residual close to zero does not prove criticality. This
implemented algebraic admission is sufficient, not a classification of
all possible sine equilibria.

For each actual preparation member, shift `phi_*` by a common constant so
that its weighted mean equals that of `theta(0)`. Write the resulting lift
as `phi_*^m`. Its phase geometry and potential `V_*` are unchanged.
The reference `psi(0)=theta(0)+alpha*P_Mx(0)` has this same weighted mean.
Choose a proved bound

\[
D_0\ge
\left\|P_M\left(\theta(0)+\alpha P_Mx(0)-\phi_*\right)\right\|_M.
\]

Since `f(phi_*^m)=0`, the constant curve `phi_*^m` solves the reference
equation. The global Lipschitz bound from Section 29 and Gronwall give

\[
\|\psi(\sigma)-\phi_*^m\|_M
\le D_{\rm ref}(\sigma):=e^{\ell\sigma}D_0.
\]

This estimate assumes no contraction or acute evolution of the reference.
The supplied stationary geometry is a proof reference; it is not inserted
into the complete law or assigned to the evolving state.

For a relative source, retain exactly Section 27's original residual
family. If its componentwise form and phase error radii are
`epsilon_xi,epsilon_thetai`, a uniform choice is

\[
\begin{aligned}
D_0={}&
\left\|P_M\left(\theta_{\rm nom}
                +\alpha P_Mx_{\rm nom}-\phi_*\right)\right\|_M\\
&+\left(\sum_i m_i\epsilon_{\theta i}^2\right)^{1/2}
+\alpha\left(\sum_i m_i\epsilon_{x i}^2\right)^{1/2},
\qquad m_i=M_{ii}.
\end{aligned}
\]

This follows from the triangle inequality and the contractivity of the
weighted orthogonal projection `P_M`. Every member keeps its own
reference, mean-matched target and conserved means. Unknown common
origins cancel before bounding the mismatch; no absolute phase or form
origin is inferred. The estimate uses the selected consistent phase
lifts, with the chart and quotient distinctions from Section 28.

### The actual endpoint and its complete storage

At the supplied slow time `sigma`, set `tau=sigma/eta` and define

\[
\begin{aligned}
Q(\sigma)&=Ze^{-\lambda\tau}+C(\sigma),\\
\rho(\sigma)&=D_{\rm ref}(\sigma)+Q(\sigma),\\
X(\sigma)&=e^{-\lambda\tau}N_x+
\frac ae\frac F\lambda\left(1-e^{-\lambda\tau}\right),
\qquad N_x\ge\|P_Mx(0)\|_M.
\end{aligned}
\]

Section 29 proves that the actual endpoint satisfies
`||theta(t)-phi_*^m||_M<=rho` and `||P_Mx(t)||_M<=X`, where
`t=sigma/(e*eta)`. The phase bound includes the initial transient;
substituting only the composite error `C` would not prove this statement.
The form bound is in the original coordinate, with its conversion factor
retained.

For an edge `ij`, write `k_i=K_ii` and
`g_ij=sqrt(k_i+k_j)`. Weighted duality gives

\[
\left|(\theta_j-\theta_i)
       -(\phi_{*j}^m-\phi_{*i}^m)\right|\le g_{ij}\rho,
\qquad |x_j-x_i|\le g_{ij}X.
\]

Thus a sufficient strict acute margin on every edge is

\[
\boxed{\quad |\delta_{*ij}|+g_{ij}\rho<\frac\pi2
\quad\text{for every retained edge}.\quad}
\]

Use the integer edge offsets that reduce the supplied target lift to its
principal acute angles. They place every enclosed actual endpoint in the
target's cycle-period sector, without inferring a future unwrapping from
samples.

A useful storage bound retains the correlations lost by independent edge
intervals. Let `h=theta(t)-phi_*^m`. The phase Hessian satisfies `H<=L`
globally, since every edge cosine is at most one. Taylor's integral
formula and the exact criticality of the target therefore give

\[
V(\phi_*^m+h)
=V_*+\int_0^1(1-s)h^{\mathsf T}H(\phi_*^m+sh)h\,ds
\le V_*+\frac12h^{\mathsf T}Lh
\le V_*+\frac\ell2\rho^2.
\]

The linear term vanishes because `grad V(phi_*^m)=0`. The segment need
not remain acute for this upper bound. Likewise
`x^T Lx<=ell*||P_Mx||_M^2`, so the complete storage obeys

\[
\boxed{\qquad
E(t)\le E_{\rm handoff}:=
\frac\ell2X^2+\beta\left(V_*+\frac\ell2\rho^2\right).
\qquad}
\]

This uses the same cancellation principle as local recovery, now with a
preparation-derived phase neighborhood. Bounding every cosine separately
can lose that cancellation and fail to establish a margin that the
correlated bound proves. The admitted endpoint set retains the norm and
storage constraints together with the edge intervals and memberwise mean
leaf; arbitrary corners of the independent outer intervals need not obey
the correlated storage bound. Global existence of the complete law makes
the set nonempty for every admitted preparation member.

### Handoff to the whole-sector theorem

Let `B_k` be the shared certified lower bound for the phase potential on
**every** boundary face of the target sector, as in Section 26. If all
strict edge margins above hold and

\[
\boxed{\qquad E_{\rm handoff}<\beta B_k,\qquad}
\]

then Section 26 applies to the actual full-state endpoint of every source
member. Subsequent complete evolution stays in that sector and converges
on its conserved-mean leaf to its unique acute equilibrium. The admitted
target already supplies the stationary geometry in that sector, so
uniqueness identifies the limiting phase geometry with it. The theorem
does not assume that geometry was the initial winding or that the
reference and actual paths shared their earlier sectors.

[`certify_sine_slow_capture`](../../src/tnfr/physics/relational_sine_reduction.py)
recomputes the preparation and slow comparison, admits the exact target,
and passes producer-proved edge and correlated storage bounds to the
shared sector-capture owner. The outer report retains the enclosed
original endpoint time; its nested capture is neither a timestamped
observation nor a sampled forecast. Outward rational arithmetic encloses
the pi, trigonometric, norm and exponential quantities. Malformed inputs
are rejected; insufficient strict margins return unavailable. An
unavailable sufficient condition proves neither instability nor
impossibility.

### Fixed analytic controls and scope

The [handoff tests](../../tests/physics/test_relational_sine_slow_capture.py)
reuse Section 27's published C5 preparation unchanged:
`nu_i=beta=1`, `e=1023/1024`, `w=1/1024`,
`x_i(0)=4092*(i-2)` and `theta_i(0)=0` for `i=0,...,4`.
The supplied target is `q_i=(i-2)/5` and the fixed slow time is `1/16`.
Its initial form budget remains `160*1023^2`; neither that budget nor the
ratio is retuned to obtain the handoff. The analytic bounds give

\[
\begin{aligned}
D_0&<0.074249,&\rho&<0.084137,\\
\frac\ell2X^2&<0.000002028,&
E_{\rm handoff}&<3.461996083,\\
B_k&>3.469266270,&
\beta B_k-E_{\rm handoff}&>0.00727018.
\end{aligned}
\]

All edge margins are positive. The same previously fixed independent
radii `epsilon_x=1/16` and `epsilon_theta=1/65536` also pass, with total
storage below `3.462017026` and margin above `0.00724924`. These are
analytic endpoint certificates, with no integrated reference or complete
trajectory used as their producer. Initial zero winding is separately
known for these preparations, so their captured nonzero sector also
establishes acquisition. The general API does not assume zero initial
winding and therefore does not label every capture an acquisition.

Here the declared original time is approximately `646182.7393` structural
units, obtained from `t=beta*pi^2*e*sigma/w^2`. This fixed handoff is not
an optimized entry time or a replacement for the earlier `tau=100` entry
certificate. Controls with resolved phase but excessive remaining form
storage, and with unresolved phase margins, remain unavailable under
their respective sufficient conditions.

The result connects one justified reference geometry to complete-state
capture at a finite declared instant. It supplies neither autonomous
support creation, selection of the preparation or target, a unique
constitutive law, nor a physical identification. It also does not turn
the finite comparison into a uniform infinite-time approximation.
