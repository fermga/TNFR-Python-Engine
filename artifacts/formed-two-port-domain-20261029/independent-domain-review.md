# Review of the analytical domain decision and integration

2026-10-29. Draft author: `/root`; reviewer: `/root/joint_review_controls`.

**Verdict: PASS for the exact two deltas below.** I find no blocking
mathematical, source-association or scope defect. The result excludes the
specified whole-domain invariance claim; it does not exclude formation from
the fixed source or admit another numerical evaluation.

| Reviewed delta | Bytes | SHA-256 |
| --- | ---: | --- |
| `chapter-append.md` | 10865 | `fdc8534b457e807d4abcee918f77ecf15c26fffc268bb3693c1f785b0983af67` |
| `document-changes.json` | 5775 | `360d859a6eb294c66e3dfa233c8f947c187828534e899b882f93603c93c665c0` |

Supporting notes read:

| Note | Bytes | SHA-256 |
| --- | ---: | --- |
| `analytical-domain-note.md` | 10872 | `78d278e13bf65ce0869f8df689d58ac50284db2b268e8e7e66832fc21a8160ce` |
| `dependencies-review.md` | 12875 | `0965182683660c2830fdb17b527f9da18d644dab79b1f31913feca859932c7f2` |

## Independent check of D1--D6

I checked the construction against the current original-label M1--M2 field
and the compatibility owner's existing equations (15)--(19), rather than
treating the supporting note as sufficient authority.

1. **Source and domain.** The nominal contact gaps are phi and phi-r, not
   the old midpoint pair. Their phase energy is bounded directly by the
   two quadratic cosine bounds shown in the text. This admits xi=0 to D1
   without substituting actual full-state storage for reference potential.
   The edge evolution and energy derivative in D2 follow from the declared
   row incidence and positive degree metric, with the stated facet sign.
2. **Compatibility and gauge.** The donor increments sum to zero, so the
   closing affine offset is retained. At the ports, D3 gives exactly
   phi+(delta-phi-9u/2)-(-4u)=delta-u/2 and
   phi-r+(delta-phi-9u/2)-(-5u)=-delta+u/2. The nine receiver increments
   are unchanged. Thus the two ring sums and the interface relation hold
   in the original labels. Subtraction of one joined mean makes xi have
   zero D-mean without resetting the relative origin or holding component
   means separately fixed.
3. **Feasible witness.** The symbolic u_rho is strictly between zero and
   pi/144. It places only donor edge 5->6 on the positive buffered facet;
   all remaining donor, receiver and contact gaps are strictly inside.
   The existing path's strict energy derivative transfers because it uses
   these same gaps. Consequently the witness is strictly below both the
   stated E=1/4 level and the midpoint energy U_min.
4. **Outward derivative.** Node 5 is a degree-three port and node 6 a
   degree-two interior node. Their incoming/outgoing sine signs give the
   displayed derivative (5/6)(sin b-sin a)+(1/3)sin c. Since b is between
   7pi/16 and pi/2, and c is between 31pi/288 and pi/2, the elementary
   trigonometric inequalities have the required directions. Their rational
   combination is exactly -25/1536+31/432=767/13824, with strictness supplied
   by pi squared <10. Thus the smooth reference flow leaves the closed
   buffered domain from this boundary state. No field implementation or
   numerical sign evaluator was used in this check.
5. **Sharper source component.** With uniform rings, varying the geometric
   origin from phi to delta keeps the buffered inequalities and strictly
   decreases U_nom before its endpoint. At that endpoint the coordinates
   match D3 with u=0, after the same joined centering. D3 then reaches the
   witness while energy decreases again. This gives an explicit path in
   the exact sublevel U<=U(g0), on the same joined-mean leaf. It proves
   membership in the source's connected component, not reachability by
   that particular reference solution.

The chapter correctly does not attribute this edge or witness to the failed
Picard candidate. It distinguishes the exact reference law from the actual
full-law phase row and from the supplied geometric path. The proposed
domain's failure therefore leaves the original source trajectory undecided.

## Floor, comparison contract and next question

I authored the supporting-affine floor lemma in `dependencies-review.md`;
D7--D8 are its integration, not an independently reviewed discovery by me.
They preserve its proof and limits: derivative signs on [-pi,A] and
[A,pi-A], cancellation of the affine terms using each ring sum, and
nonnegative contact contributions. They do not assert convexity of the
individual edge potential on the larger interval, winding preservation,
invariance or full-network curvature. Independent substantive review of
that contributed lemma belongs to the other reviewer/integrator.

I also contributed the dependency analysis behind the Omega contract.
The root's final statement retains the necessary conditions: an open convex
common-shift-invariant phase domain, the same phase floor for actual phases
there, a PSD full cosine Hessian throughout the domain, and strict D-metric
reference clearance greater than Q over the entire horizon. PSD on segments
gives nonexpansion; PSD along the reference bounds its speed. The unchanged
source bounds, graph gap, global Lipschitz bound and gamma then close the
same finite-prefix V/Z inequalities, while the clearance closes first exit
for y and theta. No additional positive Hessian gap is required for this
nonexpansion argument. A tube-only PSD check at reference states would not
establish it, and the text does not claim otherwise.

The explicit final-unit qualification avoids a hidden domain-transfer step:
clearance only through T does not by itself cover the subsequent unit in
Omega. Target arrival and the original-form handoff remain obligations.
The text also correctly states that positive growth would require new error
and speed bounds, rather than a silent change to a numerical option.

The proposed next corridor question is within the existing foundation task.
It preserves the full source, complete rows, memberwise means and target
identity. It admits neither a chosen wider corridor nor a new trajectory,
retry, resumed prefix, failed-diagnostic repair or physical mapping.

## Documentary association and scope

Each of the four PROGRAM replacements matches exactly once in order against
the current 92570-byte owner. Applying them in memory and retaining CRLF
yields exactly 94061 bytes, SHA-256
`a3e6ac7289fab2f5c630e5a8c05d91c9a04a5039a114694b7db34d90fa293592`.
The append receipt matches the reviewed bytes; the cited collective-curvature
anchor exists. No maintained source was written during this review.

This is independent review of the root/foundation obstruction D1--D6 and of
the root's integration, with my own floor/dependency contribution disclosed
above. I received the proposed witness before performing this check; this
is not a blind discovery claim. Only metadata hashing, source/text reads and
in-memory textual reconstruction were executed. No scientific producer,
field, sign/root admission, trajectory, Taylor step, diagnostic, auditor or
test was executed. Only this review file was written. Application, guard and
documentation checks remain the integrating agent's responsibility.
