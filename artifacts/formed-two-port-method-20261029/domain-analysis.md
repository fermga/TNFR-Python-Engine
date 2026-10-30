# Source-aware domain audit for the formed two-port entry method

Author: `/root/joint_review_controls`; 2026-10-29.
Scope: read-only analytical method/domain audit. No trajectory, root solve,
scientific assessor, reserved propagation, historical producer or maintained-file
edit was executed. The bounds below are hand arguments for a prospective method,
not a computed reference trajectory or a new capture verdict.

## Recommendation and unresolved obligation

A full eighteen-phase nominal reference with the actual inherited relative
origin is a defensible bounded method. Its complete-law comparison can retain
the thirty-six actual coordinates and the original source errors while using
the existing degree-metric, zero-growth reference enclosure. The old
eight-coordinate reflection restriction is not needed and does not match this
source.

The proposed `T=1024` slow-time reference interval plus one analytic continuation
unit has useful **conditional** error and endpoint bounds. It does not yet have
a proved reference trajectory staying acute and reaching the target ball at T.
Those are the two decisive new obligations of the reserved enclosure. The old
midpoint result and local attraction do not settle them. Consequently method
admission may authorize a fixed, failure-preserving attempt once its resources
and exact implementation are reviewed; it must not describe this horizon as an
already guaranteed entry deadline.

I inspected the current program and compatibility admission A1–A11, the full
comparison in [TRANSIT](../../theory/nodal/SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison),
the endpoint/tail contract in [CAPTURE](../../theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-handoff),
and the shared [_validated_metric](../../src/tnfr/mathematics/_validated_metric.py)
premises. After deriving the source constants and comparison bounds I also read
`slow-reference-analysis.md`; its source-aware method agrees with the conclusions
below. This note concentrates on the geometric and gauge obligations that must
survive its implementation.

## The full reference must carry the actual source chart

Use the original labels and contacts `(D4,R4)` and `(D5,R5)`. Let M be their
actual degree matrix, A=M^-1 L, f=M^-1 S, gamma=1/(1023*pi), eta=gamma^2 and
`sigma=eta*(tau-tau_c)`. The reference is `psi_sigma=f(psi)`, with nominal source

`Theta_D,j=2*alpha*(j-4)`,
`Theta_R,j=alpha*(j-4)+phi`,
`alpha=2*pi/9`, `phi=1/1000`.

One implementation may evolve eighteen deviations v from this fixed nominal
source. Its initial deviation is zero, and every field row uses
`sin(Theta_j-Theta_i+v_j-v_i)` divided by the true degree. This avoids replacing
an irrational nominal phase by a rounded physical initial condition. The same
mathematical pi occurs in all offsets; outward intervals are arithmetic
enclosures of that constant, not independent model parameters.

The nominal joined weighted mean is `mu0=phi/2+3*alpha/40`. On the original
source each component error row has zero ordinary sum, but the new weighted
mean generally does not vanish. Project with the actual M and preserve each
member's conserved common form/phase means. A reference can be shifted by one
common constant to that member's phase mean, since this leaves its geometry,
field and all edge observations unchanged. Independent rotations of the two
components are not allowed on the joined support.

Since M<=3I and each component residual has Euclidean norm <=epsilon=10^-32,
the combined form and phase residual norms are at most sqrt(6)*epsilon.
The weighted common-mode projection is contractive. Thus the source bounds
`3*epsilon`, `Z0=3*epsilon/3069`, and `v0=3*epsilon+Z0` remain valid after
matching the mean. They bound the actual formation image; they do not replace
it with freely preparable corners of an endpoint box.

If old two-port labels are used instead, transport `i -> (i+4) mod9`, all
observers and the fixed donor/receiver lift corrections together. The old
increasing-ramp nominal source has mean `219*alpha/40+phi/2`, differing from
mu0 by `6*pi/5` solely because of those integer lift changes. This is not a
change of relative origin. The target must be represented in exactly the same
chart and shifted only by the same common mean convention.

The old port-swapping reflection would require matching donor/receiver
reflection centers: `2*alpha = alpha+2*phi (mod 2*pi)`. The supplied phi fails
that condition. It is therefore unsound to reuse the folded eight-state
reference with a new error radius or to absorb the mismatch into a common gauge.
This rules out that folding, not every conceivable reduction.

## A concrete convex domain and what it proves

Orient each internal cycle in its original order and retain its closing-edge
offset: `+4*pi` for donor `8->0`, `+2*pi` for receiver `8->0`. Orient contacts
donor-to-receiver. The nominal contact gaps are phi and phi-alpha. In deviation
coordinates the twenty edge gaps are fixed affine functionals

`q_e(v)=q_e^0+B_e v`.

The domain `-pi/2<q_e(v)<pi/2` for all twenty edges is an intersection of open
half-spaces and hence convex. Its periods are `(2,1,0)` identically through
the nodal reconstruction; no independent edge coordinates or dynamic principal
wrapping are introduced. The mixed phase `theta+gamma*P_M x` has the same
cycle periods because the additional nodal differences telescope.

On any convex tube in this cell,

`Df=-M^-1 H(theta)`,
`H=B^T diag(cos q_e) B >= 0`,

so `u^T M Df u=-u^T H u<=0`. This justifies metric logarithmic growth zero
for the **full eighteen-coordinate** reference, including its neutral common
phase mode. It is not restricted to the old reflection-invariant subspace.
The conserved common mode does not obstruct a positive metric M; it precludes
an unqualified strictly negative full-space growth claim.

For each retained metric Taylor step, strict Picard inclusion and all twenty
whole-tube edge margins must establish the consumed domain. Checking only the
center, an endpoint, or a graph drawing cannot supply the logarithmic-growth
premise. Coordinate boxes are projections used for those checks; they must not
replace the retained metric ball or erase source/endpoint correlations.

Every edge functional has M-dual norm `sqrt(1/d_i+1/d_j)<=1`. This connects the
metric error and an edge margin without an invented norm conversion. The
nominal initial margin is greater than `1/6`; the smallest is the donor's
`pi/18`. At the source only the four contact rows of f are nonzero, and exactly

`||f(Psi0)||_M^2=(2/3)[sin(phi)^2+sin(phi-alpha)^2]`
` <(2/3)[(1/500)^2+(7/10)^2] <(3/5)^2`.

Within the acute cell the gradient speed norm cannot increase. Before a first
exit, edge displacement by slow time 1/4 is consequently below `3/20`, leaving
margin greater than `1/6-3/20=1/60`. The first-exit argument makes this a useful
analytic initial-domain certificate. It does not cover the remaining interval
to 1024.

I found no justified global invariant acute strip for the actual nonmidpoint
source in the inspected owners. Node-gradient cooperativity in an acute chart
does not automatically give a maximum principle for its twenty edge gaps on
this branched graph. The declared whole-tube checks are therefore substantive
conditions, not redundant diagnostics. This is a missing domain certificate,
not a claim that the actual reference leaves the chart.

## The actual-source error budget fits the proposed domain test

Use the fixed-period acute-cell Jensen floor
`U_floor=9*(1-cos(4*pi/9))+9*(1-cos(2*pi/9))`.
At each isolated twist the phase first variation vanishes, while the ring
Laplacian norm is at most four. The source's internal form excess is at most
`4*epsilon^2`, and its internal phase excess is at most `4*epsilon^2` over both
rings. Adding the already admitted two-edge work gives

`H0-U_floor <49/200+1/500000+10*epsilon^2 <1/4`.

The slack before residuals is `2499/500000`; this is not the old midpoint
budget 1/8. The phase floor is used only on the relevant fixed-period acute
cell in the first-exit comparison, rather than asserting global convexity of
the periodic sine potential.

With total slow horizon Sigma=1025, `sqrt(Sigma/4)<17`. The existing comparison
therefore permits

`V=v0+17/50000`,
`Z=[Z0+10^-5*(3/5+2*V)]/(1-1/50000)`, `Q=V+Z`.

Direct conservative inequalities give `V<7/20000`, `Z<1/100000` and
`Q<9/25000<1/2048`. No arithmetic script or reference evaluation was used
to select these bounds. They retain both complete state rows and all actual
source errors. They are not an assumption that the actual phase is a closed
gradient flow.

If every new reference tube has margin greater than `rho=1/2048`, these
bounds exclude exit of both actual phase theta and mixed phase y from the
common cell. Convexity supplies the segment needed for the dissipativity
comparison. This is the required bootstrap; acute initial and final phases
alone do not establish it.

## Endpoint, time and stopping rule

The separate endpoint obligation is
`||psi(1024)-theta_*||_M<=rho`, with the exact compatible target in the same
lifts and weighted gauge. The upper bound must include the reference metric
radius, arithmetic reconstruction and uncertainty in the correlated implicit
target. A small center residual or independently rounded target entries do not
prove it. A uniform mean error may be explicitly bounded or consistently
projected; it must not be silently discarded from a full-space norm.

Conditional on that endpoint, the target's acute margin greater than 1/8 and
gradient dissipativity keep the reference inside the rho ball during the final
analytic unit. Its speed is then at most `2*rho`. The complete-law form tail
can be bounded using the moving reference, rather than the coarse global Z:

`Z_end <2^-512/100000+1/51200000`,
`||P_M x(1025)||_M <3216*Z_end <1/8192`,
`||theta(1025)-theta_*^m||_M <Q+rho<1/1024`.

The radius-squared/storage upper bound is then below
`65/67108864<1/648000`, inside the required maintenance domain. The weaker
new global bound Z<1/100000 does not invalidate the analytic final-unit tail.
All these conclusions remain conditional on the new reference's whole-path
and endpoint checks; none is an observed outcome here.

The prospective physical variable remains structural time: the complete-law
handoff would be fixed at `tau_b=tau_c+1025/eta`, with `tau=e*t` and the same
exact gamma/pi relation throughout. Pi enclosures do not create independent
clocks in different rows. No laboratory duration or practical speed is implied.

T=1024 is a specified bounded candidate with an existing matched comparison
and endpoint/tail contract. Its reuse is legitimate as a prospective choice,
not as transferred evidence that the new reference reaches the ball. Before
reserved propagation the actual full-reference field, initialization, metric,
all domain functionals, root/target association, finite step/precision policy
and total resource cap must be frozen and independently reviewed. A failing
tube, endpoint or resource bound must preserve its completed prefix and exact
failure condition. It cannot trigger an unrecorded smaller step, longer
horizon, midpoint substitution or retry under the same protocol.

The useful next decision is thus one source-aware reference certificate or one
precise failed method condition. Failure of the acute reference route would not
by itself prove non-entry of the complete model or erase the admitted identity
and conditional response. This note supplies no general invariant-strip theorem
and does not repeat the rejected scalar-storage handoff criterion.
