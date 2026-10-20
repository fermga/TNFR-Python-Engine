# Source resources for localized mobile identities

**Scope:** a conditional source-to-target obstruction under the same complete
positive-loss sine law as the [prepared transport theorem](LOCALIZED_IDENTITY_TRANSPORT.md).
No trajectory, response search, source reconstruction or physical measurement
is evaluated. This result identifies a missing preparation resource for one
class of mobile identities; it does not exclude every moving organization.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns closure and subsequent admission.

## 1. Complete law and an independent source class

Keep the labeled unit cycle `C_N`, held unit capacities, beta one, positive
coefficients `e,w`, no inputs or events, and clock `tau=w*t/pi` from the
[law owner](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-law). With
`A=L/2` and `sigma=pi*e/w`, both complete rows are

\[
 x'=-\sigma Ax+\tfrac12 S(\theta),\qquad \theta'=Ax.
\]

Retain real phase lifts relative to the supplied uniform zero background.
As before, `z=x+i theta` denotes two real state channels. The source class is

\[
 z(0)=(u,0)+r_0,\qquad u\in\mathbb R^N,\qquad
 \|u\|_2\le M,\qquad \|r_0\|_2\le\epsilon,\qquad M>0. \tag{1}
\]

The nominal phase is uniform. Its zero lift fixes the background origin;
it does not infer phase from form. The independent error covers both channels
at every site, including common means. No zero-mean source is assumed.

The real profile u is otherwise arbitrary: it can already contain localization,
spatial asymmetry, a spectral envelope or several peaks. Thus this class lacks
the nominal form/phase correlation of a one-sided carrier, not necessarily
all spatial organization. Its definition uses no desired final coordinates,
backward evolution or fitted source. The [ontology contract](../EMERGENT_ONTOLOGY.md#localized-identity-transport)
keeps these different formation claims separate.

For `N` divisible by four, `Q>=2` and `Q-1<N/4`, define the disjoint Fourier
bands, with indices interpreted modulo N,

\[
 B_+=\{N/4+m:|m|<Q\},\qquad B_-=-B_+.
\]

Use the unitary discrete Fourier transform and its orthogonal projectors
`P_+,P_-`. A target v has norm a>0 and support entirely in either one of these
bands. This includes every translated or internally rotated Fejer reference
of the prepared theorem. Allowing all such vectors makes the obstruction
independent of that particular envelope. Fourier orientation is not a new
primitive phase, topological winding, charge or particle label.

<a id="phase-flat-source-obstruction"></a>
## 2. The actual damped tangent preserves opposite-band balance

Linearize the actual law at the uniform background:

\[
 J_\sigma=\begin{pmatrix}-\sigma A&-A\\A&0\end{pmatrix}.
\]

At wave number k, `lambda(k)=1-cos(k)=lambda(-k)`. The two real modal
functions f_k,g_k are defined by

\[
 \binom{f_k(\tau)}{g_k(\tau)}=
 \exp\!\left[\tau\lambda(k)
 \begin{pmatrix}-\sigma&-1\\1&0\end{pmatrix}\right]\binom{1}{0}.
\]

They are the same at k and -k, including overdamped and critically damped
cases. Let `z_lin` be this actual tangent launched from `(u,0)`. Since u is
real, `u_hat(-k)=conj(u_hat(k))`, while

\[
 \widehat z_{\rm lin}(k)=(f_k+i g_k)\widehat u(k),\qquad
 \widehat z_{\rm lin}(-k)=(f_k+i g_k)\overline{\widehat u(k)}.
\]

Their magnitudes agree. Consequently

\[
 \|P_+z_{\rm lin}(\tau)\|_2=\|P_-z_{\rm lin}(\tau)\|_2. \tag{2}
\]

This statement uses actual positive loss; it needs no conservative comparison
or dispersion-error allowance. It also needs no spatial reflection symmetry
of u. Real form alone does not imply a reflection-even spatial profile.

For any state y satisfying (2), write the common band norm as r. If v has
norm a and is supported in `B_+`, orthogonality and the reverse triangle
inequality give

\[
 \|y-v\|_2^2\ge(r-a)^2+r^2
 =2(r-a/2)^2+a^2/2\ge a^2/2. \tag{3}
\]

The other band is identical. Components outside both bands only increase
the distance. Removing common form/phase origins does not remove this lower
bound, since neither band contains the zero Fourier mode.

## 3. Transfer to the complete nonlinear trajectory

The shared [contractive tangent and cubic remainder](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-error)
give `||R(theta)||_2<=(4/3)||theta||_2^3`. In `||z||_2<=2M`, Duhamel's
formula yields, for every `0<=tau<=T`,

\[
 \|z(\tau)-z_{\rm lin}(\tau)\|_2
 \le\delta=\epsilon+\frac{32}{3}TM^3. \tag{4}
\]

If `delta<M`, continuity closes the bootstrap: contraction gives
`||z_lin||<=M`, so at a putative first exit the actual norm is instead bounded
by `M+delta<2M`. This also admits the initial state because `epsilon<M`.
The complete smooth sine law supplies finite-time continuation. Retain
`2M<pi/2` for the small lifted-phase chart; for the calculator's integer
`T>=1`, `delta<M` already implies this condition.

Combining (3) and (4) proves the uniform lower separation

\[
 \boxed{\quad\|z(\tau)-v\|_2\ge a/\sqrt2-\delta\quad} \tag{5}
\]

for every source in (1), every time in the interval and every norm-a target
in either declared band. Entry into a target tube of radius `eta*a` is
therefore excluded when `a/sqrt(2)-delta>eta*a`. A failed sufficient test
does not prove entry, and an unclosed bootstrap supplies no nonlinear
separation certificate. Balance (2) is not asserted as a nonlinear invariant.

### Fixed source and target budgets

Keep the prepared theorem's actual law, `N=1024`, `Q=16`, `a=2^-12` and
`T=256`. Supply `M=a`, `epsilon=M/1000` and target tolerance `eta=1/25`.
Then

\[
 \frac\delta a=\frac1{1000}+\frac1{6144}<\frac2{1000},\qquad
 \frac1{\sqrt2}>\frac{707}{1000}.
\]

The source remains more than `141a/200` from every one-band norm-a target,
well outside the `a/25` identity tube, throughout the interval. This excludes
even an instantaneous acquisition of that identity, hence also its later
retention within the same interval. The result covers arbitrary nominal form
profiles; it is stronger than testing one failed impulse or one guessed source.

A directional target has an appropriate correlation between both channels.
For example, `x_j=b*cos(kj)` and `theta_j=b*sin(kj)` give `z_j=b*exp(ikj)`,
which occupies just one nonzero band. This is a boundary control outside the
phase-flat class, not an acquisition proof. Supplying phase norm without the
appropriate correlations is not sufficient either. Different envelopes,
two-sided targets, larger source resources or longer nonlinear evolution
retain separate questions.

<a id="reflection-source-obstruction"></a>
## 4. An independent exact reflection restriction

For the reflection `j -> -j` on the same cycle, both full nonlinear rows
are equivariant. If the **complete actual initial state, including its retained
real phase lifts,** is reflection-even,
uniqueness preserves that symmetry for all forward time. Thus
`rho_j=x_j^2+theta_j^2` satisfies `rho_j=rho_{-j}`. Whenever a window W and
its reflection are disjoint, each contains at most half the total rho.
For zero total rho the signal-floor requirement fails instead.

This excludes concentration above `24/25` in one such window at any
amplitude for those declared continuous lifts. Circular phase symmetry alone
does not imply equality of squared lifts chosen on different sheets.
The frozen center `c=-tau` and radius 96 have disjoint reflected
windows for `96<tau<=256` on `C_1024`. The restriction does not apply to
arbitrary spatially asymmetric u in (1), nor to arbitrary errors that break
reflection symmetry. It permits two counterpropagating concentrations.

The [existing symmetry owner](../../src/tnfr/physics/relational_sine_symmetry.py)
already requires complete support/capacity equivariance and actual-state
symmetry; its [involution controls](../../tests/physics/test_sine_involution_reduction.py)
retain these premises. No new event selector or symmetry-breaking rule is
installed by this argument.

## 5. Computation and research interpretation

The private source calculator in
[`_localized_identity_transport.py`](../../src/tnfr/physics/_localized_identity_transport.py)
re-admits scalar source/target budgets and actual model coefficients. It
shares the absolute cubic remainder and exact clock conversion with the
prepared reader, while keeping the source verdict distinct. The source
proof has no packet-dispersion term or conservative-loss penalty. Reported
separations are available only after the actual nonlinear bootstrap closes.
The [static controls](../../tests/physics/test_localized_identity_transport.py)
derive modal balance, the geometric distance, a correlated-phase boundary
case and the finite budgets without evaluating a response.

The scientific result is a necessary preparation restriction for a specified
family, not failure of the emergence hypothesis. A full-state directional
organization is information that cannot be supplied by an arbitrary small
phase-flat form preparation on this horizon. Localization itself can already
be present in such a source, and bilateral motion remains possible. A new
formation claim must identify which property is absent initially and allow
the corresponding complete-law symmetries. The distinction also prevents an
analyst-chosen waveform from becoming a definition of every possible NFR.
