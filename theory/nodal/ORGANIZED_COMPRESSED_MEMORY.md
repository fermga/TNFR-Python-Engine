# Finite collective prediction with frozen constituent variances

<a id="organized-compressed-memory-declaration"></a>
## Fixed analytical comparison

The [active gate](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-predictive-compression-admission)
asks for useful compression with propagated nonlinear error. The
[inherited-response owner](ORGANIZED_CONSTITUENT_MEMORY.md) derives the exact
hidden history and excludes exact moment-only closure. Here a finite
approximation retains five initial interface statistics; their uncertainty
is an explicit initialization premise, not information supplied by formation.

| Item | Fixed scope |
| --- | --- |
| Fine model | Unchanged unit C9-by-C5 support, node `9*c+j`, held unit capacities, full-circle pairwise sine, `x'=(-L_f*x+gamma*S(theta))/4+R*f`, `theta'=gamma*L_f*x/4`, `gamma=1/(1023*pi)` |
| Clock/coordinates | `tau=(1023/1024)*t`, natural child means and continuous phase lifts U,Phi; zero-mean hidden u,z and the existing helical template. Same-slot interfaces and their order remain fixed |
| Retained initialization | Actual U0,Phi0 and five supplied estimates `mhat_e` of `m_e(0)=sum_j(z_dj-z_cj)^2/9`; `0<=mhat_e<=2`, `sum_e abs(mhat_e-m_e(0))<=mu`. Exact mean initialization is conditional; its uncertainty must be added separately |
| Compressed dynamics | Ten evolved real coordinates Uhat,Phihat with the same form diffusion, phase row and input, and static sine weights `w_e=1-mhat_e/2`; five immutable statistics, no evolved hidden coordinates or moment matrix |
| Primitive bounds | `||u0||<=a0=31/10^9`, `||z0||<=b0=601/10^10`, `gamma<=g=1/3000`, `mu<=1/10^17`. The existing acquired source lies within these norm bounds; it does not supply the five statistics or this uncertainty guarantee |
| Input and horizon | The unchanged 15-ramp/15-plateau finite-observation word, same supplied input for fine and compressed models, duration `15.00015<16`. Prove uniform bounds on the enlarged horizon `T=16`; no new word, source replay or altered response |
| Observation and target | Uniform Euclidean mean-form error over the word, with lifted-phase error and actual signed work retained. Require the new intrinsic error envelope to be less than one hundredth of the existing carrier envelope on these primitive bounds |
| Domain and identity | Retain the original actual-word winding and initialization hypotheses; rebuild hidden transport and circular collar. Transfer a predicted mean winding only with its strict phase-gap margin. General identities do not certify formation or arbitrary-input winding retention |
| Reduction | 10 evolved coordinates plus 5 static statistics versus 90 evolved fine coordinates. This is a state-count reduction; no elapsed-time speedup, exact nonlinear closure or independent access to initialization is assumed |
| Evidence budget | Exact rational inequalities and shared outward trigonometric constants; independent static edge and storage controls. No trajectory, source assessment, reserved-response evaluation, physical data or fitted coefficients |

The comparison concerns intrinsic mathematical approximation error at common
initial means. It does not reduce separately supplied preparation, measurement,
clock or input errors. A proof may close this conditional compression gate
without admitting an operational forecast for the acquired source.

<a id="organized-compressed-memory-result"></a>
## Result and mathematical scope

Declaration `269c5ccc` fixes the comparison above. Five static interface
variances suffice for a conditional finite approximation with ten evolved
coordinates. On the declared primitive bounds, its uniform intrinsic
mean-form error is below `1.5e-20`, and its certified envelope is less than
one hundredth of the carrier envelope. Both statements concern exact
continuous solutions with matched initial means, not numerical solver
accuracy, a resolved nonzero signal or an observed physical response.

The improvement comes from retaining initial phase dispersion and bounding
its subsequent drift with the exact hidden-history identity. No hidden
state is evolved in the approximation. The declaration's coarse norm prior
does not supply the five estimates or exact collective initialization.
The subsequent [preparation-aware result](ORGANIZED_MEMORY_INITIALIZATION.md#organized-memory-initialization-result)
recovers the existing preparation's separate form and phase bounds, which
do admit zero estimates with the required uncertainty. A carrier prediction
carried from the known attachment means also avoids requiring an observed
microstate at the start of the word. This source-specific conclusion does
not change the frozen declaration or the containing-class obstruction below.

## 1. Retained state and approximate law

Orient the five outer edges `c -> d`, let B have tail -1 and head +1,
and use `L=L5`. For the actual initial hidden state define

\[
 m_e^0=\frac19\sum_{j=0}^8(z_{dj}(0)-z_{cj}(0))^2,\qquad
 w_e=1-\widehat m_e/2.
 \tag{1}
\]

The five estimates are immutable parameters. Their joint error satisfies
`sum_e |mhat_e-m_e^0|<=mu`. They encode relative internal phase spread
across the fixed same-slot interfaces; they do not reconstruct the fine
state, identity or event admission. They are invariant under a common
phase shift and a common cyclic slot relabeling. Independently changing
slot registration changes the interface and is outside this comparison.

The reduced initial-value problem is

\[
 \widehat U'=(-L\widehat U+\gamma S_w(\widehat\Phi))/4+f,
 \qquad \widehat\Phi'=\gamma L\widehat U/4,\qquad
 S_w(\Phi)=-B\,[w\mathbin\odot\sin(B^T\Phi)].
 \tag{2}
\]

Initialize `(Uhat,Phihat)=(U0,Phi0)`. This is the carrier law plus the
previously derived quadratic correction evaluated at the frozen variances.
It does not change the fine law. Since `0<=mhat_e<=2`, the weights lie in
`[0,1]` and `S_w` is globally Lipschitz with constant at most four in the
Euclidean norm. Thus prescribed piecewise-continuous f defines a unique
continuous solution; switches do not jump the state.

Equation (2) uses ten evolving real coordinates and five static statistics,
in place of 90 evolving fine coordinates. Common conserved means are counted
in both descriptions. The input program, graph and initialization certificate
are external data in both. This count claims neither minimum dimension nor
measured runtime improvement. The exact nonlinear moment counterexamples
remain valid: (2) is a bounded approximation, not an exact quotient.

## 2. Hidden drift without a hidden solver

For admitted nonnegative primitives a0,b0,g,T use the shared
[hidden transport](../../src/tnfr/physics/_organized_electrical_source.py):

\[
 s=2gT<1,\qquad A=\frac{a_0+s b_0}{1-s^2},\qquad
 Z=\frac{b_0+s a_0}{1-s^2}.
 \tag{3}
\]

These bound the actual hidden norms uniformly. The live phase row gives
`||z(t)-z0||<=2gTA`. A stronger bound follows from the
[exact hidden history](ORGANIZED_CONSTITUENT_MEMORY.md#2-exact-hidden-history-dependence):
`E(t)=exp(-t L_f/4)` is a self-adjoint positive contraction, so
`||I-E(t)||<=1`; also `||G(Phi,z)||<=8||z||`. Therefore

\[
 \sup_{0\le t\le T}\|z(t)-z_0\|\le
 D:=\min\{2gTA,\;g a_0+2g^2TZ\}.
 \tag{4}
\]

No heat exponential, fine trajectory or hidden history needs evaluation to
use (4). Form diffusion controls the integrated influence of the initial
hidden form; replacing that influence by its peak times T is unnecessarily
loose here. Both bounds include hidden initialization and actual nonlinear
forcing. Uniform-child input still affects the collective phase history,
although it does not directly force the hidden row.

Let D_o map the 45 hidden phases to their 45 outer-edge differences.
Then `||D_o||<=2`. Difference of squared norms and Cauchy-Schwarz give

\[
 \sum_e|m_e(t)-m_e^0|
 \le\frac19\|D_o(z(t)-z_0)\|\,\|D_o(z(t)+z_0)\|
 \le\frac49D(Z+b_0).
 \tag{5}
\]

This is an a priori drift bound for the actual variances, not an equation
that sets their derivatives to zero.

## 3. Propagated nonlinear forecast error

Write the actual mean row as (2) evaluated at the actual U,Phi, plus r(t).
The inherited quadratic correction, its cubic remainder, (5), and the
initial variance uncertainty imply

\[
 \|r(t)\|_2\le\|r(t)\|_1\le
 \rho_c:=\frac g9D(Z+b_0)+\frac{2g}{27}Z^3+\frac g4\mu.
 \tag{6}
\]

The three terms respectively account for variance drift, discarded higher
phase moments and initialization error. In particular, a supplied variance
estimate cannot silently replace its missing error certificate. The bound
uses the actual nonlinear hidden state and is uniform in its allowed
initialization and the collective history.

Let E_U,E_Phi be uniform Euclidean differences between actual and reduced
means. Heat contraction in the form row and the Lipschitz bound of S_w give
`E_U<=T rho_c+k E_Phi`, `E_Phi<=k E_U`, with `k=gT`. Hence

\[
 E_U\le\frac{T\rho_c}{1-k^2},\qquad
 E_\Phi\le\frac{kT\rho_c}{1-k^2}.
 \tag{7}
\]

This propagates the error through the coupled phase history; it does not
freeze Phi or compare only instantaneous rates. The existing carrier bound
uses `rho_0=gZ^2/9` in precisely the same denominator. For Z>0 the ratio
of the certified envelopes obeys

\[
 \frac{\rho_c}{\rho_0}
 =\frac{D(Z+b_0)}{Z^2}+\frac23Z+\frac{9\mu}{4Z^2}.
 \tag{8}
\]

The baseline is that carrier's quadratic-in-Z departure envelope. The
smaller pointwise cubic bound with the actual live variances still requires
hidden information at every time; it is not an existing compressed forecast.
Here the extra drift and initialization terms pay for removing that access.

On an exact zero-hidden carrier with exact statistics, both errors vanish;
there is no gain ratio to report. For general initial collective errors
e_U,e_Phi, the bounds instead are

\[
 E_U\le\frac{e_U+k e_\Phi+T\rho_c}{1-k^2},\qquad
 E_\Phi\le\frac{e_\Phi+k e_U+kT\rho_c}{1-k^2}.
 \tag{9}
\]

Their common initialization contribution is not reduced. Clock, input,
sensor and numerical integration errors are absent from (7), and require
their own terms when consumed. No numerical integrator is introduced here.

## 4. Fixed-budget inequalities, identity and work

The declaration's loose bounds contain the original acquired-source norm
priors by the existing [source inequalities](ORGANIZED_ELECTRICAL_PROTOCOL.md#3-actual-source-transport-through-every-finite-input).
Exact rational arithmetic in (3)-(7) gives the conservative upper bounds

| Quantity | Bound in the declared structural coordinates |
| --- | --- |
| Hidden form A | `<3.2e-8` |
| Hidden phase Z | `<6.1e-8` |
| Hidden phase drift D | `<1.1e-11` |
| Residual rho_c, including `mu=1e-17` | `<9e-22` |
| Uniform mean-form error E_U | `<1.5e-20` |
| Uniform lifted-phase error E_Phi | `<8e-23` |
| New/carrier intrinsic envelope | `<1/100` |

The last line compares envelopes built from the same primitive bounds, not
the actual error of either model; the latter can vanish. The declared
uncertainty premise is needed throughout. The unchanged raw form-reading
allowance `1e-14` is much larger than both intrinsic bounds, so this gain
does not establish improved physical resolution or a detected correction.

The hidden circular observer is retained because `2Z<pi/2`. The actual
input word, source and complete fine law are unchanged, so the
[whole-history identity proof](ORGANIZED_FINITE_OBSERVATION.md#2-complete-identity-and-actual-source-inclusion)
continues to apply. It is not inferred from the reduced model. For its mean
prediction, the original word has principal mean gaps at most
`7/5+1/10+2E`, where `E=2g/100+2g^2T`. Adding the edge prediction error
`2E_Phi` still leaves a strictly positive margin to pi/2 (greater than
`7/100` under the declared bounds). Thus the predicted mean winding agrees
throughout that word. It is not a reconstructed nine-slot fine state.
Arbitrary new inputs require their own outer-identity argument.

The approximation has its own weighted storage

\[
 \widehat H=\tfrac12\widehat U^TL\widehat U+
       \sum_e w_e[1-\cos(B^T\widehat\Phi)_e],\qquad
 \widehat H'=-\tfrac14\|L\widehat U\|^2+(L\widehat U)^Tf.
 \tag{10}
\]

This identity follows from the two live reduced rows. It is not the actual
fine storage or proof that the signed hidden exchange is passive. The true
fine external power remains `9(LU)^T f`. Therefore the difference between
its work and the reduced prediction `9 integral (L Uhat)^T f` is at most

\[
 36 E_U\int_0^{T_{obs}}\|f(t)\|_1\,dt.
 \tag{11}
\]

The same allowance controls integrals of absolute value and positive part
of the total external power, since these functions are 1-Lipschitz. It
does not transfer a sum of absolute or positive powers taken separately
at fine nodes: hidden contributions cancel only before that operation.
The original word's finite input/work bounds remain with its owner. Actual
hidden loss is still bounded by `16TA^2`; neither the approximation nor
(11) removes that loss or preparation/event work.

## 5. Initialization debt and verification

The five initial variances are derived statistics, not five newly supplied
fundamental laws. Nevertheless their availability does not follow from a
norm certificate. For example, at U0=0 and the same outer winding-one mean
phases, compare u0=0,z0=0 with u0=0 and only child zero's hidden phase
`(a,-a,0,...,0)`, where `a=2^-26`. Both obey the declaration's hidden norm
bounds and the original larger inference-class collars. The nonzero
preparation has `m_e^0=2a^2/9` on each incident outer edge and zero on the
others. Their variance vectors differ in l1 by `4a^2/9>2e-17`, so no single
estimate from just those common means and norm bounds can satisfy the
declared `mu<=1e-17` for both.

Their exact initial corrections also differ by
`[gamma sin(2pi/5)(1-cos a)/18]*(0,1,0,0,-1)`. This is a containing-class
information obstruction, not a claim that both preparations were acquired,
or an impossibility of learning from additional finite observations. It
does not replace a forecast-error proof with an instantaneous difference.

The [independent controls](../../tests/physics/test_organized_compressed_memory.py)
use shared hidden transport and exact arithmetic, reconstruct interface
currents and storage from edges, and check the drift and coupled error
inequalities and initialization obstruction. Frozen evidence and producers
are preserved. No response has been replayed and no new runtime law or
public solver has been installed.

The conditional reduction meets its state and error criteria. The
[initialization result](ORGANIZED_MEMORY_INITIALIZATION.md#organized-memory-initialization-result)
separately admits a carrier forecast for the actual prepared source by
retaining information lost in this comparison's coarse norm prior. At that
resolution, no additional five-statistic measurement is required. The
larger-class observation obstruction remains scoped to its own prior;
independent physical preparation, sensors and numerical execution retain
their separate obligations. The
[sole queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns further work. Neither result identifies a physical force or material
constituent.
