# Chemical observation: sufficient state and predictive information

<a id="chemical-observation-result"></a>
## Result and scope

For the supplied three-field chemical model below, an ideal catalyst snapshot
does not close its own evolution. Its first two temporal derivatives recover
the complete finite-dimensional state on a specified regular domain. Every
smooth exact realization of these output histories on an open preparation
family therefore needs at least three real state coordinates per retained
spatial site. Singular preparations need a separate analysis; some become
identifiable with one more derivative, while a restricted counterexample
keeps a hidden field unobservable for the entire history.

This is conditional mathematics about an independently supplied comparison
model, not a derivation of chemical dynamics from TNFR. Physical admission
remains `not_admitted`, and empirical evaluation remains `not_tested`.
No response array, experiment, trajectory or fitted pressure is used.
The [atlas admission](../PHYSICAL_REGIME_CORRESPONDENCES.md#observed-pattern-admission-result)
owns source selection and measurement gaps; the
[research program](../RESEARCH_PROGRAM.md#research-current-question)
owns scheduling. The [ontology](../EMERGENT_ONTOLOGY.md#research-target-physical-properties-of-collective-patterns)
permits collective observations without identifying a sensor with primitive EPI.

## 1. Supplied source, state and clock

Use the modified Oregonator rows in
[Azhand, Totz and Engel (2014), Section 3, Eq. (1)](https://arxiv.org/html/1406.5964v2#S3),
reviewed on 2026-10-16. The paper associates `u,v,w` with three chemical
concentrations; `v` is the immobilized catalyst channel. Illumination enters
the bromide source `phi`. Its model time is dimensionless. The numerical
discussion also uses spiral-derived length/time units, which supply no
independent laboratory-clock calibration here. The text's ordering of
`epsilon_u,epsilon_w` and its later unassigned `epsilon,epsilon'` numerical
notation are not silently reconciled: retain independent positive row
coefficients. No numerical parameter set is imported.

Declare `n` sites and a fixed known real matrix `L` with nonnegative
off-diagonal entries and `L*1=0`. These conditions preserve constants and
support the positivity argument; they do not certify that an arbitrary
matrix realizes the article's geometry or its nineteen-point stencil.
Any PDE approximation and Neumann-boundary discretization need their own
consistency/error justification. Both mobile fields use this declared
spatial operator, while the catalyst has no diffusion. Componentwise,

\[
\dot u=\epsilon_u^{-1}[u-u^2+(q-u)w]+Lu,\qquad
\dot v=u-v,\qquad
\dot w=\epsilon_w^{-1}[\phi+fv-(q+u)w]+\delta Lw.
\]

Assume `epsilon_u,epsilon_w,q>0`, `f,delta>=0`, nonnegative concentrations
and a prescribed continuous nonnegative input `phi(t)`. Coefficients and
support are held; no event, stochastic forcing or additional reservoir law
is inferred. At each zero concentration, its derivative is nonnegative
when the other concentrations are nonnegative. The orthant is therefore
forward invariant on the local existence interval. This does not assert
global continuation for arbitrary unbounded input or reproduce a laboratory
preparation. Interior positive preparations supply the open domains below.

The ideal output is **the entire labeled field** `y=v` in this model clock.
It is not identified with the experimental camera image; depth integration,
optical response and calibration remain unadmitted. Derivatives below are
exact model derivatives, not finite differences of data.

<a id="chemical-regular-observation"></a>
## 2. Regular reconstruction and the minimum exact state

Write `y_j=d^j y/dt^j` and define

\[
U=y_0+y_1,\qquad D=y_1+y_2,\qquad
N=\epsilon_u(D-LU)-U+U^2,\qquad a=q-U.
\]

The source rows imply `U=u`, `D=dot u` and `N=a*w`. On the domain
`a_i!=0` for every site, the inverse is exactly

\[
v=y_0,\qquad u=y_0+y_1,\qquad w_i=N_i/a_i.
\]

No illumination value is needed for this instantaneous inverse. It follows
from the first two supplied rows; it neither infers the third row nor defines
TNFR pressure from the observed response.

In state order `(v,u,w)`, the Jacobian of
`Psi_2=(y_0,y_1,y_2)` is block lower triangular with diagonal blocks
`I,I,diag((q-u)/epsilon_u)`. Thus

\[
\det D\Psi_2=\epsilon_u^{-n}\prod_i(q-u_i),\qquad
\operatorname{rank}D\Psi_2=3n
\]

on the regular domain. This is a nonlinear local observation theorem, not
an equilibrium linearization or a rank count of sampled traces.

For completeness, hold the same input history and clock fixed. Suppose a
target with `m` real state coordinates and smooth initialization `J`
reproduces these ideal histories for every state in an open positive
preparation set. Require its order-two output-jet map `K_2` to be `C^1`
in target state. Smooth autonomous dynamics/readout with a fixed constant
input suffice; arbitrary continuous inputs are not silently differentiated.
The output jet then factors as `Psi_2=K_2 o J`.
Consequently `3n<=rank(DJ)<=m` wherever the source is regular. Every such
open set contains regular points: the exceptional hyperplanes `u_i=q`
have empty interior. Hence `m>=3n` is necessary for this exact all-state
claim. Three source fields also suffice when the supplied input is retained.

In particular, `n` TNFR nodes with only signed scalar form and circular
phase evolving, and all capacities/support held, have local dimension `2n`.
They cannot give that exact realization on an open source family. This does
not exclude more nodes, additional justified evolving state, an admitted
restricted preparation, approximate predictions or causal memory. Counting
a third coordinate alone would not establish a TNFR constitutive map.

Two elementary comparisons explain the obstruction. Holding `v` fixed while
changing `u` changes `y_1`. Holding both `v,u` fixed while changing `w`
changes `y_2` by `(q-u)*Delta w/epsilon_u`. On regular positive states,
neither a snapshot nor the pair `(y_0,y_1)` is a closed predictive state.
This reuses the [general closure criterion](../FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
and the earlier [circuit observation method](PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md#rossler-phase-information-admission),
with the chemical model's own nonlinear coefficient and singular domain.

<a id="chemical-singular-observation"></a>
## 3. Singular observations are not automatically unobservable

At a site with `a_i=0`, second-jet consistency requires `N_i=0` and does
not determine `w_i`. Let `D=dot u` as above and

\[
\dot N=\epsilon_u(y_2+y_3-LD)+(2U-1)D.
\]

Differentiating `N=a*w` gives `dot N=-D*w+a*dot w`.
Thus, if `D_i!=0` at that singular site,

\[
w_i=-\dot N_i/D_i.
\]

This transverse recovery uses the third derivative but not illumination.
For a spatially uniform `u=q*1`, `D=q(1-q)/epsilon_u`; all singular
sites therefore recover this way when `q!=1`.

There is also a spatial route. With known instantaneous illumination,
the entire third-jet consistency problem is the linear system

\[
\begin{pmatrix}A\\B\end{pmatrix}w
=\begin{pmatrix}N\\\dot N-A(\phi+fv)/\epsilon_w\end{pmatrix},
\quad A=\operatorname{diag}(a),
\quad B=\delta AL-\operatorname{diag}(D)
       -\operatorname{diag}(a(q+u))/\epsilon_w.
\]

Full column rank supplies a unique solution, subject to nonnegative-state
and residual admission. Regular neighbors can reveal a tangential singular
site through diffusion even when its own `D_i=0`. If the stacked matrix
has a kernel, it produces nearby distinct positive states with identical
jets through order three at an interior preparation. It does not prove
identical full histories. At a positivity boundary, intersection with the
nonnegative cone may remove this ambiguity. Unknown illumination introduces
another ambiguity in the regular rows and cannot be silently eliminated.

A genuine full-history exception exists within the symbolic model family.
Take `q=1`, spatially uniform `u=1`, any uniform positive `v(0)` and two
different uniform positive `w(0)`, with the same nonnegative uniform input.
Then `u` stays one and `dot v=1-v`, independently of `w`; distinct bromide
histories remain invisible. This is a lower-dimensional invariant preparation
at a particular parameter value, not a counterexample to the open-family
dimension theorem or a claim about the paper's experimental parameters.

<a id="chemical-predictive-input"></a>
## 4. Recovering state does not recover future input

On the regular domain, substituting `w=N/a` into the third source row yields
the exact input/output identity

\[
a\dot N+ND
=\frac{a^2}{\epsilon_w}\left[\phi+fv-(q+u)\frac Na\right]
 +\delta a^2L\left(\frac Na\right).
\]

Products and divisions are componentwise; `L(N/a)` is the full spatial
operation, not `(LN)/a`. Together with `u=y_0+y_1`, this is a third-order
output description on the regular domain. It is a rewriting of the supplied
chemical law with its full state information retained, not a two-field
reduction or a TNFR pressure law.

At the same state, changing the prescribed input changes the third output
derivative by

\[
\Delta y_3=(q-u)\Delta\phi/(\epsilon_u\epsilon_w).
\]

Thus identical `y_0,y_1,y_2` do not predict `y_3` without input information.
More generally, agreeing past inputs need not determine their future. An
externally prescribed source is not rendered autonomous by reconstructing
internal state. Any controller or environment generating it needs its own law.

## 5. Exact information versus recoverable information

Fix the source coefficients, spatial operator and clock. Suppose the true
state satisfies `||u||_infinity<=M_u`, `||w||_infinity<=M_w` and
`min_i|q-u_i|>=rho>0`. Let supplied approximate jets have independent
supremum error allowances `eta_0,eta_1,eta_2`. Set

\[
\alpha=\eta_0+\eta_1,\qquad b=\eta_1+\eta_2,\qquad
B_N=\epsilon_u(b+\|L\|_\infty\alpha)
       +(1+2M_u+\alpha)\alpha.
\]

If `alpha<rho`, algebraic reconstruction satisfies

\[
\|\widetilde u-u\|_\infty\le\alpha,\qquad
\|\widetilde w-w\|_\infty
\le\frac{B_N+M_w\alpha}{\rho-\alpha}.
\]

Indeed the numerator error is at most `B_N`, the reconstructed denominator
is at least `rho-alpha`, and
`tilde N-tilde a*w=(tilde N-N)+(tilde u-u)*w`.
The inverse becomes poorly conditioned as the separation vanishes. Near
`u=q`, a fixed bromide difference can produce arbitrarily small second-jet
separation. This bound assumes derivative-error allowances; it does not
derive them from finitely sampled noisy images, calibrate the camera or
cover parameter, spatial-discretization or clock uncertainty. The transverse
inverse similarly requires control of its own nonzero denominator.

The information result is therefore about **distinguishable states and
prediction**, not bits or a universal entropy balance. Shannon quantities
would additionally require a preparation distribution and an observation/noise
channel. EPI, structural capacity and coherence are not probabilities, channel
capacity or entropy by name. Existing [TNFR memory](../DERIVED_EPI_MEMORY.md)
and [nonlinear hidden-state elimination](../nodal/SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)
provide reusable methods, not coefficients that transfer to this chemical law.

## 6. Execution and verification

The detached [exact control module](../../src/tnfr/research/chemical_observation.py)
admits rational or represented-real coefficients before calculation. It
evaluates source jets and reconstructs regular or pointwise transverse states;
the general stacked singular solver is a theorem here, not an implemented
reconstruction mode. An unsupported singular inverse is explicit abstention.
Optional `y_3` is consumed only by that transverse inverse and does not certify
the full source/input law. The implementation is capped at 32 sites as an
algebraic resource policy, not a theorem dimension restriction.

[Independent controls](../../tests/research/test_chemical_observation.py)
differentiate the supplied polynomial field symbolically, check exact rank,
hidden-state/input counterexamples, diffusion-assisted recovery, singular
domains and scalar admission. They do not differentiate measured traces or
run the chemical reactor. Existing shared exact scalar and linear-observation
owners remain unchanged. No engine solver, native law, operator, grammar rule,
SDK export or physical measurement map is installed by this module.

The remaining physical obligations are the complete optical observation,
retained depth/spatial state, preparation, input law, laboratory clock,
uncertainty and independent acquisitions. Any exact TNFR realization must
also satisfy [all mapped evolution rows](../nodal/PHASE_FORM_EXCHANGE.md#151-source-matching-precedes-phase-tangency),
including its phase/capacity/support premises. The result selects no such law
and does not identify a chemical organization with an elementary constituent.
