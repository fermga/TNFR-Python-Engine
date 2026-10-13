# Certified endpoint observation of a prepared mobile encounter

**Scope:** evaluate the one reserved cubic-response certificate declared in
[the frozen protocol](LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-protocol).
The complete law, source balls, source registration, clock, horizon and
identity contract are unchanged. This is a detached calculation of an
analytic Duhamel integral with a proved transfer to the complete nonlinear
flow, not numerical integration of a nonlinear trajectory. It installs no
new native flow or support event. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#mobile-pair-response-admission)
alone owns the next task.

<a id="mobile-response-numerics"></a>
## 1. Exact finite Fourier representation

Use the unitary transform `hat u_k=N^-1/2 sum_j u_j exp(-2*pi*i*k*j/N)`.
Let `c=T/2`, `k0=N/4`, `b_m=1-|m|/Q` and
`S2=sum_{|m|<Q} b_m^2`. The nominal complex bookkeeping variable of source A
has Fourier coefficients

\[
 \widehat z^A_{k_0+m}(0)=\frac{a b_m}{\sqrt{S_2}}
             e^{-2\pi i(k_0+m)c/N},\qquad |m|<Q,
\]

and zero coefficients elsewhere. The real channels are separately retained:
on that positive band `hat x=hat z/2`, `hat theta=hat z/(2i)`; the opposite
coefficients follow by conjugacy of each real channel. Source B is the exact
spatial reflection of A, so `hat x^B_k=hat x^A_{-k}` and likewise for phase.
No extra relative phase is selected from the response.

For `lambda_k=1-cos(2*pi*k/N)`, the **actual** tangent block is

\[
 B_k=\lambda_k\begin{pmatrix}-\sigma&-1\\1&0\end{pmatrix},\qquad
 \sigma=\pi e/w.
\]

With `omega=sqrt(1-sigma^2/4)>0`, its exact propagator is

\[
 U_k(s)=e^{-\sigma\lambda_k s/2}
 \left[\cos(\omega\lambda_k s)\,\mathrm{Id}
 +\frac{\sin(\omega\lambda_k s)}\omega
 \begin{pmatrix}-\sigma/2&-1\\1&\sigma/2\end{pmatrix}\right]. \tag{1}
\]

The instrument requires `sigma*T<=1`, in addition to its positive-loss law
admission. This keeps every decay argument within the shared unit-exponential
domain and ensures the square root is separated from zero. It is a numerical
instrument domain, not a new restriction on TNFR generally. The fixed model
satisfies it. Form and phase must not be replaced by a single scalar complex
propagator: positive form loss mixes the two bookkeeping carrier directions.

Take `D theta_j=theta_j-theta_{j+1}` and `d_k=1-exp(2*pi*i*k/N)`.
If alpha and beta are the Fourier coefficients of the two tangent edge-phase
differences, the mixed form forcing is the cyclic convolution

\[
 \widehat g_k(s)=\frac{\overline d_k}{4N}
       [\alpha*\beta*(\alpha+\beta)]_k. \tag{2}
\]

The two unitary Fourier products each contribute `N^-1/2`; omitting this
normalization would give an incorrect signal. The cubic integrand has separate
form and phase channels `U_k(T-s)*(hat g_k(s),0)`.

Both tangent phase histories occupy only the two radius-15 bands. Equation
(2) occupies the two radius-45 bands around +/-N/4 after modular folding,
182 modes in total. No cubic output mode is discarded. This finite support
statement concerns the cubic predictor, not the nonlinear full flow. The
instrument requires `3(Q-1)<N/4`, `N<=4096` and `Q<=32` for this representation.

Reflection exchanges A and B and preserves their mixed forcing, so g is
reflection-even. Its Fourier coefficients are real and equal at opposite
modes, as are those of the propagated integrand. The implementation intersects
the two independently enclosed equal real coefficients and checks that the
imaginary enclosures contain zero before retaining this exact symmetry.
It retains both partners when applying Parseval's norm.

## 2. Outward arithmetic and the integration error

Elementary functions, square roots and final coefficient boxes use the shared
128-bit outward rational intervals. For polynomial convolution, their dyadic
endpoints become exact integers on the same grid. All four endpoint products
are enclosed, and integer sums are accumulated before a single outward
projection per output coefficient. This is an optimization of a finite
polynomial reduction, not an alternative interval arithmetic or a floating FFT.
An opaque evaluation closure captures constants rebuilt from fresh primitive
source and model inputs; externally supplied derived reports are not consumed.
Trigonometric argument reduction subtracts an integer multiple of an outward
`pi/2` interval, then applies the exact quarter-turn identities and the shared
sine/cosine owners to the residual. Choosing that integer from a midpoint is
only a representation choice: the identity holds for every integer, including
at uncertain quadrant boundaries. The residual interval retains pi uncertainty.

Let `qmax=2*pi*(Q-1)/N`. On the input and output Fourier supports, respectively,

\[
 K_{\rm in}=(1+\sigma)(1+q_{\max}),\qquad
 K_{\rm out}=(1+\sigma)(1+3q_{\max})
\]

bound the tangent operator norm: its rotational part has norm one and its
loss part norm sigma, while `1-cos(pi/2+q)<=1+|q|`. Actual tangent propagators
are contractive. Each differentiated input has norm at most `a*K_in^j`.
The trilinear forcing has norm at most `8*a^3`, using `||D||<=2`.
Differentiating its three inputs and the final propagator gives, by the
multinomial theorem, the whole-window integrand derivative bound

\[
 \|f^{(j)}(s)\|_2\le8a^3\Lambda^j,
 \qquad \Lambda=K_{\rm out}+3K_{\rm in}
                  =(1+\sigma)(4+6q_{\max}). \tag{3}
\]

This retains the actual damped generator in every derivative. It does not
infer an error from the difference of two sampled numerical answers.

Use an n-point Gauss-Legendre rule on each of P equal panels. Its nodes are
certified by exact rational sign bisection of the degree-n Legendre polynomial:
n disjoint sign-changing intervals exhaust its roots. Outward weights are
`2/[(1-r^2)*P_n'(r)^2]`. The positive Peano kernel gives the same remainder
constant for the Euclidean vector norm as for a scalar, with no dimension
multiplier. Equations (3) and the Gauss remainder imply

\[
 E_G\le 8a^3 T\left(\frac{\Lambda T}{P}\right)^{2n}
       \frac{(n!)^4}{(2n+1)((2n)!)^3}. \tag{4}
\]

Choose `n=12`. Starting at one panel, double P until this prior bound is at
most one tenth of `eta_num=a^3/1000`, or the frozen limit 65,536 prevents
another doubling. This selection reads no integrand or response. For the
fixed inputs, `Lambda<23/5` and P=64 suffices; hence 768 integrand evaluations
are prescribed before the response is known.

Node, weight, initial coefficient, propagator and convolution rounding are
retained in the accumulated spectral boxes. The Euclidean norm of their
component radii bounds the vector arithmetic error. The separate outward
Parseval norm interval retains norm-reduction uncertainty; the numerical
certificate uses E_G plus the larger of these two arithmetic allowances.
All source materialization is included. The final cubic norm interval expands
the quadrature norm interval by E_G. If the total error exceeds eta_num,
numerical certification is unavailable even if a computed lower bound is
positive. There is no automatic higher-precision retry.

<a id="mobile-response-verdict"></a>
## 3. Transfer to the actual nonlinear response

Rebuild the original [analytical budgets](LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-defect)
from their primitive inputs. For an enclosed cubic norm `[l,u]`, the actual
nominal defect norm lies in

\[
 [\max(0,l-K_5),\ u+K_5],
\]

and every original shared-source preparation has defect norm within

\[
 [\max(0,l-K_5-V),\ u+K_5+V]. \tag{5}
\]

The two lower-bound tests remain separate. Their scientific certificates also
require the declared identity bounds and successful numerical certification.
A zero lower bound means unresolved sufficiency, not an observed zero. Bounds
already include quadrature uncertainty; it is not subtracted a second time.
Linear shared-source evolution remains the exact zero control. The result,
if positive, concerns endpoint nonlinear deformation relative to separate
evolution, not creation of a new NFR or a physical particle.

## 4. Implementation, controls and retained evaluation

The [spectral owner](../../src/tnfr/physics/_localized_mobile_spectral.py) computes
the integrand, the [shared quadrature owner](../../src/tnfr/mathematics/_gauss_legendre.py)
encloses nodes, weights and truncation error, and the
[response reader](../../src/tnfr/physics/_localized_mobile_response.py) performs
the fixed endpoint reduction and separate verdicts. It does not modify the
shared nonlinear integrator or relax existing support caps.

Independent controls check the spectral transform against direct real-site
preparation, a matrix-exponential series and pointwise mixed cubic forcing;
all Gauss polynomial moments through degree `2n-1` and the sharp degree-`2n`
remainder; and exact interval-product corners. Response wiring uses explicitly
mocked polynomials and a zero curve, not the reserved integrand. It also checks
that a positive computed value cannot bypass a failed numerical budget.

The [one-shot driver](../../scripts/run_localized_mobile_response.py) uses the
shared exact artifact and strict JSON owners. A source archive and freeze are
written before evaluation; an exclusive attempt precedes the scientific call.
The first outcome, including failure, is retained. Archives associate declared
bytes and configuration; they do not independently authenticate chronology or
execution. The method was admitted before the reserved response below.

<a id="mobile-response-result"></a>
## 5. Retained endpoint result and its limits

The first assessment used the unchanged protocol, all 182 cubic output modes,
12 Gauss nodes on each of 64 panels, and 128-bit outward arithmetic. The
following decimals are deliberately rounded outwards from the exact rational
[report](../../docs/assets/localized_mobile_interaction/report.json):

| Quantity | Certified enclosure or upper bound |
| --- | --- |
| Total numerical error | `<4.639280e-16`, below `a^3/1000 > 1.45519e-14` |
| Cubic endpoint norm | `[1.2888499e-12, 1.2897778e-12]` |
| Complete-flow nominal defect norm | `[1.2847097e-12, 1.2939180e-12]` |
| Complete-flow defect norm over the original shared-source family | `[0, 6.1613890e-12]` |

Thus the exact nominal preparations have a strictly nonzero endpoint
superposition defect under the declared complete nonlinear law. The linear
shared-source control has exactly zero defect. Both prepared band identities
retain their separate whole-window certificate. This is a finite mathematical
interaction result, conditional on the supplied sources, support and law.
It is not a full nonlinear trajectory calculation or laboratory observation.

The original source-family test remains **unresolved**. Its sensitivity
allowance is about `4.86747e-12`, whereas the nominal fifth-order allowance
is about `4.14022e-15` and numerical error is below `4.64e-16`. More quadrature
precision alone therefore cannot make this same sufficient uniform bound
positive. The zero lower endpoint is not a witness of zero interaction for
an admissible preparation. Neither acceptance level, source tolerance nor
other frozen input was changed after evaluation.

The [protocol](../../docs/assets/localized_mobile_interaction/protocol.json)
pins the scientific admission to revision
`14cbad44f92194774c4cd66a8763f2069c95270c`. The
[manifest](../../docs/assets/localized_mobile_interaction/source-manifest.json),
[source archive](../../docs/assets/localized_mobile_interaction/source.zip),
[freeze](../../docs/assets/localized_mobile_interaction/freeze.json) and
[exclusive attempt](../../docs/assets/localized_mobile_interaction/attempt.json)
retain the implementation and first outcome, based on source HEAD `a8c02f0b`.
The [read-only evidence controls](../../tests/physics/test_localized_mobile_response_evidence.py)
re-admit primitive inputs, rebuild analytical and quadrature budgets, and
reconstruct Parseval intervals and verdicts from the retained spectral boxes.
They forbid producer replay. These reductions are conditional on those boxes;
byte associations do not independently authenticate execution or prove each
integrand evaluation.

The defect I is a difference between three histories, not an independently
evolving object. Even additional Fourier modes can describe ordinary
deformation or radiation. In this preparation `T*a^2=1/65536`; the certified
nominal defect is less than `5.3e-9*a`. No additional NFR identity, formation,
post-separation lifetime or subsequent response is established. Investigating
recursive organization requires an independently motivated actual-state
identity and a complete-law mechanism that can sustain it. The
[formation-feasibility gate](../research/FIVE_STAGE_EXECUTION_PLAN.md#mobile-organization-feasibility-admission)
owns that question without authorizing a replay or parameter search.
