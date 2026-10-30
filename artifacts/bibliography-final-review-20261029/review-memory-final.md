# Final integration review: memory and composition

Reviewer: `/root/joint_review_controls`. Draft author: `/root`.
Decision: **PASS within the reviewed conceptual and source-attribution scope.**

This review retains and extends `review-memory-v1.md`; that earlier review was
not replaced. I compared the full V1-to-V3 textual delta, read `v2-changes.json`
and `v3-changes.json`, verified the referenced V1/V2/V3 hashes, and confirmed
that the program draft is byte-identical to the previously reviewed program.
The substantive V1 findings therefore remain applicable with the refinements
listed below. No blocking issue remains in this scope.

## Final reviewed bytes

| Path relative to repository | Bytes | SHA-256 |
| --- | ---: | --- |
| artifacts/bibliography-final-review-20261029/chapter-proposed-v3.md | 45703 | 0ce7be2760cac8f497e1fbeedac2abeec6986ecf06983bf914cc9ab4257d3a49 |
| artifacts/bibliography-final-review-20261029/program-proposed.md | 89873 | 5ddcd9a1b637a69b2d43389e8966e6dd2410ddc277708afcdd863476ba315caf |

The preserved V1 chapter is 45,075 bytes, SHA-256
`a9798b074800d7d1820da2fda3f9de8501498ab7180fa3f94f214bbfa0d3e865`.
The intermediate V2 chapter is 45,694 bytes, SHA-256
`50b9fcb0fed27b5e7aa16171aa6503c6fc50a63a6145226129292eb9d80b4f1a`.
The original memory review remains 4,336 bytes, SHA-256
`032395b6d12301844c4ac765fc6f98b3a6c2e12381f1ad8f2105db858d17035b`.

## Delta assessment

1. Appendix 1 now has the correct printed/PDF locators: 198–200 / 209–211,
   restricted to Appendix 1 on the last page, with Appendix 2 explicitly
   excluded. This matches my actual textual and visual reading. Its use is
   limited to retained initialization and the constant-operator premise;
   no time-varying frozen-generator inference has been introduced.

2. The existing bounded contact-continuation contract is explicitly linked.
   It is presented as a source-specific finite prediction, not an indefinitely
   reusable autonomous state or a minimum encoding. The linked owner's four
   ordered contact-pair query families retain their own continuous horizon,
   common coupling uncertainty, original packet and carried-history premises;
   the synthesis does not replace those with four sampled time points.

3. The causal composition wording now names form and phase channels and
   separates continuous-input feedthrough from declared form jumps. This
   agrees with the existing exact component-map contract and avoids confusing
   a jump in state with instantaneous feedthrough of continuous inputs.

4. The optimization criterion permits either lower declared cost or useful
   accuracy while preserving the requested prediction. Full state and already
   adequate static approximations remain valid. No new small-state prerequisite
   or unverified speed claim was added.

5. “Equal-retained-state/different-response” resolves the V1 ambiguity. It
   identifies an information-loss witness on an observation fiber rather than
   different deterministic outcomes from the same complete state and input.

These changes do not enlarge the scientific task, alter the formed-source
admission, admit transit, or open physical mapping. I authored the underlying
memory notes; this is an independent review of the root's integration, not an
independent reproof of those notes or of the linked mathematics. No scientific
evaluation/replay, maintained-file edit or finalizer execution was performed.
Application, source preservation and documentation validation remain separate
root-owned checks. Later content changes require a review of their new bytes.
