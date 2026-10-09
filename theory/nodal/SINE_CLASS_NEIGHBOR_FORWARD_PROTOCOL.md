# Independent full-law check of distinct-neighbor nonadditivity

<a id="sine-class-neighbor-forward-protocol"></a>

The [distinct-neighbor theorem](SINE_CLASS_NEIGHBOR_NONADDITIVITY.md)
predicts a finite response of the mediator that separately matched nonlinear
single-neighbor functionals cannot reproduce by addition. This owner admits
one independent complete-law calculation and fixes its source transfer,
numerical budget and observation decisions before that response is evaluated.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns admission and subsequent evaluation status.

The shared producer evolves all 54 original coordinates. Its inputs contain
no analytic coefficient, predicted sign, source-radius policy or observation
error. The independent forward interval is never intersected with the
prediction. This is a conditional structural test; no laboratory clock,
sensor calibration, autonomous event selection or physical identification is
supplied by its success.

<a id="sine-neighbor-forward-model"></a>
## Complete model, preparation and intervention

Use three unit-conductance C9 components on ordered nodes 0 through 26, with
contacts `(4,13)` and `(13,22)`, unit capacities and actual joined degrees.
The central degrees are `(3,4,3)`. All forms precede all continuous phase
lifts in the state. The class tuple is fixed to `(1,2,1)`. The complete rows
and structural clock remain
\[
 x'=-Ax+\gamma D^{-1}S(\theta),\qquad
 \theta'=\gamma Ax,\qquad A=D^{-1}L,\quad
 \gamma=\frac1{1023\pi},\quad \tau=\frac{1023}{1024}t.
 \tag{1}
\]
Both rows use this clock, with continuously refreshed pressure and no
additional continuous forcing. Forms are signed; phase lifts are not wrapped
or recentered during execution.

Retain the [original acquired-family premise](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source),
including its preparation, support formation, storage costs, common origins
and original component zero sums. Each component's form and target-phase
Euclidean residual has norm at most \(\epsilon=10^{-32}\). These are
conditional handoff bounds, not a newly acquired source or a conclusion from
the forward response. Every actual initial residual is the same in all four
counterfactual histories. Different independently prepared residuals would
require a different protocol and error bound.

With \(m=7/10000\) and \(T=1/8\), the four histories \((00,10,01,11)\)
apply the time-zero jump
\[
 x(0^+)=x(0^-)+i m e_4+j m e_{22},\qquad
 \theta(0^+)=\theta(0^-),\qquad
 Y_{ij}=x_{13,ij}(T).
 \tag{2}
\]
The events commute because they translate distinct form coordinates. Every
hidden coordinate carries unchanged through these instantaneous jumps. There
is no intervening flow or reset. The prospective statistic and alternative are
\[
 M=Y_{00}-Y_{10}-Y_{01}+Y_{11},\qquad
 Y_{11}^{\rm add}=Y_{10}+Y_{01}-Y_{00}.
 \tag{3}
\]
Each single-neighbor term in the alternative is the same-source full
nonlinear response under (1). The alternative's true mixed statistic is
zero by its definition. Rejecting this additive functional does not reject
pairwise microscopic laws: (1) itself is pairwise.

The theorem supplies separate simultaneous work and whole-state identity
certificates at radius \(1/12\), contact allowance \(10^{-12}\), and each
single jump allowance \(10^{-6}\). Those certificates keep the actual family
and its correlations. A successful smooth-domain numerical enclosure does
not certify work or identity for arbitrary corners of an outer box.

<a id="sine-neighbor-forward-source"></a>
## Nominal reference and common-source transfer

For component \(c\), local node \(j\), and \((k_0,k_1,k_2)=(1,2,1)\),
the exact reference is
\[
 x=0,\qquad \Theta_{9c+j}=\frac{2 k_c(j-4)}9\pi.
 \tag{4}
\]
The [shared source recipe](../../src/tnfr/research/sine_class_comparison_protocol.py)
uses the certified 256-bit Machin pi enclosure. For each signed multiplier
\(q=2k_c(j-4)/9\), store the ordered pair
\([\min(q\pi_-,q\pi_+),\max(q\pi_-,q\pi_+)]\). Materializing it on the
shared outward dyadic128 grid supplies the absolute phase input. Negative
multipliers reverse endpoints. This is an enclosure of the nominal reference,
not the phase-deviation variable and not a new preparation.

Separately retain actual-family coordinate covers: form pairs
\([-\epsilon,\epsilon]\) and phase pairs from (4) expanded by that radius.
The original component balls and zero-sum correlations imply containment in
these boxes; the converse is not claimed. Only the nominal reference enters
the forward calculation. Pi uncertainty and interval rounding already belong
to its numerical enclosure and are not charged again as source error.

Define the residual \(z_0=(x(0^-),\theta(0^-)-\Theta)\) relative to the
original declared common origins. Residual zero denotes the nominal wound
target; it does not denote zero absolute phase. The
[full nonlinear initialization bound](SINE_CLASS_COLLECTIVE_INTERFACE.md#sine-collective-interface-fidelity)
applies before any cubic approximation. For each history \(h\), with
\(A_h\in(0,m,m,2m)\), the actual-minus-nominal form difference is
\[
 x_h(T;z_0)-x_h(T;0)=[\exp(JT)z_0]_x+e_h,
 \qquad \|e_h\|_\infty\le E_{\rm init}(A_h,T,\epsilon),
 \tag{5}
\]
where \(J\) is the common full linearization at (4). Put
\(g=1/3000\), \(\ell=1-2gT\), \(D_T=1-2g^2T^2\). The bound is
\[
 E_{\rm init}(A,T,\epsilon)=
 \frac{4gT\epsilon}{\ell D_T}
 \left(\frac{gA}{D_T}+\frac{\epsilon}{\ell}\right).
 \tag{6}
\]
The common linear term in (5) cancels exactly with the four weights in (3).
No symmetry of an arbitrary actual residual is assumed. Summing all four
remaining defects, including the nonzero zero-input defect, gives
\[
 |M(z_0)-M(0)|\le S
 =\frac{16gT\epsilon}{\ell D_T}
   \left(\frac{gm}{D_T}+\frac{\epsilon}{\ell}\right)
 <1.556\,10^{-42}.
 \tag{7}
\]
Thus a nominal forward interval \(F\) gives the actual-family enclosure
\(I=F+[-S,S]\). This is the only source transport allowance in the forward
comparison. Cubic recoupling, static-cone and fifth-order amplitude errors
belong to the analytic prediction; a full-law forward interval receives none
of them. The proof of (7) does not consume any predicted endpoint.

<a id="sine-neighbor-forward-numerics"></a>
## Shared execution and fixed numerical policy

The [four-history observation owner](../../src/tnfr/physics/relational_sine_class_readout.py)
accepts optional selectors `first_probe_node`, `second_probe_node` and
`readout_node`, each an ordinary integer in 0 through 26. The new protocol
uses `(4,22,13)`; existing calls retain `(4,4,22)`. Admission precedes field
construction. Explicit selector metadata is retained and reconstructed from
admitted primitives. Old reports with no selector fields can be read only
under the legacy defaults; invalid or partial explicit fields never fall
through to those defaults. Existing donor/receiver aliases must agree.

Two shared prefixes and four suffixes use the unchanged complete sine field
and source-box Picard/Taylor kernel. At zero delay the prefixes apply their
events without a flow step. Every suffix evolves a complete inherited state.
For general admitted selectors the signed sum of repeated second-event
readout jumps is zero, as is the signed sum of the shared prefix values.
Therefore the primary mixed observation may retain only the four suffix
flow increments. In this selected protocol neither jump acts at node 13,
so even individual readout jumps are zero. Raw endpoint intervals and their
separate mixed interval are retained, not substituted after seeing widths.

| Quantity | Prospective value or rule |
| --- | --- |
| Event delay / final time | 0 / \(1/8\) |
| Fixed step | \(1/128\), clipped only at declared boundaries |
| Taylor order | 16, with certified order-17 remainder |
| Unique attempt cap | 64: four suffixes of 16 steps, two zero-duration prefixes |
| Arithmetic | Shared exact rational and outward dyadic128 intervals |
| Flow evidence | Full source box, strict Picard tube, series, remainder and full endpoint |
| Failure policy | First numerical or budget failure stops all later steps and events |
| Adaptation | No retry, width-driven extension, order change or source narrowing |
| Numerical postcondition | Transported actual interval width at most \(10^{-30}\) |

The width condition is a required postcondition, not a claim that the
unexecuted method attains it. The short fixed horizon and high order reduce
truncation, while wrapping, pi materialization and finite arithmetic remain
in the forward evidence. All attempted steps consume the global budget,
including the first failed attempt. Partial evidence remains inspectable;
there is no complete mixed endpoint until all four histories finish.

<a id="sine-neighbor-forward-decisions"></a>
## Independent comparison and observation decisions

Rebuild the analytic nominal band from the static direct cubic heat term,
complete cubic correction and amplitude tails. Its actual-family band also
includes (7). The [static theorem](SINE_CLASS_NEIGHBOR_NONADDITIVITY.md#sine-neighbor-source-and-finite-sign)
puts the actual response inside the coarse open band
\((-3.9\,10^{-29},-1.0983\,10^{-29})\). The comparison uses the freshly
rebuilt rational closed bounds, not rounded summaries or stored passing flags.

Preserve completion, both nominal and actual prediction overlaps, numerical
width, true sign, recorded sign and independent-additive separation as
distinct decisions. Closed bands overlap when their intersection is nonempty,
including shared endpoints. This comparison differs from testing membership
in the displayed coarse open band. The forward band itself remains unchanged.

For \(I=[L,U]\), each of four readings has a supplied independent error
allowance \(\delta=10^{-30}\). The full-law recorded statistic lies in
\([L-4\delta,U+4\delta]\), while a separately recorded additive alternative
lies in \([-4\delta,4\delta]\). Consequently
\[
 U<0,\qquad U+4\delta<0,\qquad U+8\delta<0
 \tag{8}
\]
are respectively the negative true-sign, recorded-sign and independent
separation criteria. The third requires a strictly positive margin
\(-U-8\delta\); equality does not pass. This allowance is inherited from the
earlier nonlinear studies, not from the later \(10^{-8}\) memory comparison.
It is not a demonstrated laboratory capability.

The [assessment owner](../../src/tnfr/research/sine_class_neighbor_forward.py)
reconstructs support, law, source, selectors, ancestry, complete state carry
and each retained Taylor arithmetic step before consuming the mixed interval.
Derivative and Picard generation remain execution premises: arithmetic
reconstruction and hashes do not authenticate their construction. Its status
distinguishes missing completion, disjoint prediction bands, excessive width,
unresolved separation and all conditions met. Missing observations retain
unavailable predicates. A wide band is not a scientific counterexample;
disjoint rigorous bands require an audit of the theorem, source and numerical
premises before a scientific interpretation.

<a id="sine-neighbor-forward-freeze"></a>
## Immutable protocol and subsequent evaluation boundary

The source base, complete runtime environment, exact primitive inputs, source
covers, rebuilt static prediction, observation policy, scripts and model
owner are pinned under `class-neighbor-forward-v1`. Only `.protocol.json`,
`.source.zip` and `.freeze.json` belong to this admission. The common
[source inspector](../../src/tnfr/research/frozen_source.py) has an explicit
schema adapter; the archive is associated with a full Git base without
runtime overlays. The original acquired family remains a conditional proof
premise, so no earlier observed response is required as a numerical input.

Freezing evaluates no selected time coefficient or trajectory and creates no
attempt or outcome. The historical receipt field
`evaluation_status_at_freeze="not_evaluated"` retains that meaning after a
later evaluation. Byte association does not authenticate chronology,
acquisition or mathematical correctness.

The separately admitted evaluation makes at most one attempt. Its preflight
checks frozen inputs, script bytes, runtime source and environment before
creating the attempt marker. It retains completion or failure, all available
evidence and its independently rebuilt decisions. Exceptions and export
failure preserve the attempt and prevent an automatic retry. Neither an
unresolved result nor a consistency conflict permits silently revising the
policy. Any correction requires separately identified evidence.

<a id="sine-neighbor-forward-retained-freeze"></a>
## Retained prospective freeze

The completed admission is pinned to complete source commit
`8af5b1932308b81e5fbb4e519de0162b8e82ccfc`, with no runtime overlays.
The [protocol](../../docs/assets/sine_formed_classes/class-neighbor-forward-v1.protocol.json),
[source archive](../../docs/assets/sine_formed_classes/class-neighbor-forward-v1.source.zip)
and [receipt](../../docs/assets/sine_formed_classes/class-neighbor-forward-v1.freeze.json)
retain the policy above and 30 archived files plus their manifest.
The protocol has 152589 bytes and SHA-256
`bcdc0160bc91e628bb1259139f5451772d2af61298f88e068d55232fef498179`;
the archive has 149640 bytes and SHA-256
`2317c3a54dd83704d99ad1bcf9ff976d6f25e5cd7cc2754fa5e38726769733ec`.
The receipt SHA-256 is
`c2dec8230791e8b5377283cd088b1414c105f010d3ada82a4e5e075d8ddc0cdf`.

The supplemental evaluator is
`build/class-neighbor-forward-freeze/evaluate_neighbor.py`. The shared
restorer can recover its full pinned runtime and archived supplements.
No prior observed response is consumed; the original source remains the
stated conditional acquisition premise. At freezing there was no attempt,
response or export-error artifact. The
[read-only freeze audit](../../tests/research/test_sine_class_neighbor_forward_freeze.py)
checks this association and the sealed prospective text without importing
the archived worker or regenerating scientific evidence. The
[synthetic evaluator controls](../../tests/research/test_sine_class_neighbor_forward_evaluator.py)
separately check preflight, attempt and export behavior with unrelated records.

<a id="sine-neighbor-forward-reserved-result"></a>
## First reserved full-law result

The [exclusive attempt](../../docs/assets/sine_formed_classes/class-neighbor-forward-v1.attempt.json)
and [retained response](../../docs/assets/sine_formed_classes/class-neighbor-forward-v1.response.zip)
record the one execution in a restored workspace at the pinned source base.
The complete runtime, environment, evaluator bytes and primitive inputs
matched the frozen declaration. No source, observer, horizon, step, order or
budget was changed. Both zero-duration prefixes and all four full-state
suffixes completed: 64 planned, attempted and validated steps, six completed
segments and no failed or unattempted branch. The recorded calculation and
assessment took approximately 125.504 seconds of wall time; the structural
horizon remains \(1/8\). No retry or export-error artifact was produced.

The [read-only response audit](../../tests/research/test_sine_class_neighbor_forward_evidence.py)
admits exact artifact and member bytes before decoding. It reconstructs every
54-coordinate event carry, retained Taylor increment and endpoint before
rebuilding the observation. Original derivative/Picard generation remains
an execution premise; the audit does not rerun the flow or authenticate
acquisition. Independent source and recording arithmetic agree with the
saved assessment without trusting its cached verdicts.

The exact primary nominal mixed interval reconstructed from the four suffix
increments is
\[
 F=\left[-\frac{1456691539}{2^{126}},
         -\frac{2913382991}{2^{127}}\right],\qquad
 \operatorname{width}(F)=\frac{87}{2^{127}}.
 \tag{9}
\]
Apply only the common-source allowance (7). Conservative outward summaries
of the exact rational decisions are:

| Quantity | Retained bound |
| --- | --- |
| Actual-family mixed response \(I=F+[-S,S]\) | \([-1.71233267499,-1.71233262385]\,10^{-29}\) |
| Nonlinear source allowance \(S\) | \(<1.55568520680\,10^{-42}\) |
| Actual-family interval width | \(<5.11343153979\,10^{-37}\), below the frozen \(10^{-30}\) ceiling |
| Four-reading recorded interval | \([-2.11233267499,-1.31233262385]\,10^{-29}\) |
| Separate additive recorded interval | \([-4,4]\,10^{-30}\) |
| Strict independent-additive separation margin | \(>9.12332623850\,10^{-30}\) |

Both nominal and actual closed prediction overlaps pass. The independently
computed forward intervals are retained in full, without intersection with
either prediction. All fixed conditions pass: completion, numerical width,
prediction consistency and strict separation after the two independent
four-reading error budgets. True and recorded negativity remain separately
reported predicates, not substitutes for the eight-error comparison.

The response ZIP contains the exact original JSON outcome as its sole member,
without reserialization. Its associations are:

| Retained item | Bytes | SHA-256 |
| --- | --- | --- |
| Attempt | 762 | `3ed3763c37a7ddc83f075eb1d2ee5cd369d5b88991091f441e6598e96b7bc537` |
| Response ZIP | 2,016,937 | `c62c4edcdf8765028f4af9c9c41b7ff35bc02a9d89bdddfed8c14b34501b6c68` |
| Outcome member `class-neighbor-forward-v1.json` | 29,737,680 | `e308c8c9432d94372e515f68673471b1e056258ba28f829b79eb9a9048abea97` |

This finite result supports the prospective distinct-neighbor prediction
under the supplied complete law and original source premises. Separately
matched nonlinear single-neighbor responses do not reproduce the joint
response by addition. It does not exclude pairwise microscopic dynamics,
establish a difference between classes, determine an energetic interaction
law or identify a physical force. The laboratory observation bridge remains
open. The historical `not_evaluated` freeze field continues to describe the
earlier admission stage; this section owns the subsequent result.
