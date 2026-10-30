# Post-hoc diagnostic of three retained boxes

Author: `/root/admission_foundation`, 2026-10-29. Await independent review
before running `diagnose_stored_tubes.py`. This is declared after seeing that
four steps were certified and step index 4 failed its whole-tube domain test.
It is a post-hoc diagnostic, not a reserved prediction or another attempt.

Read only the fixed, hash-bound inputs listed in the script. Compare exactly
three existing boxes, with all twenty original oriented edges:

1. The last certified whole-time tube in `steps/0003.json`, covering [3/4,1].
2. That certificate's endpoint box at 1, whose center/radius are carried into
   `failed-step.json` without reset.
3. The retained rejected candidate for [1,5/4]. This candidate was not admitted
   as a whole-time trajectory enclosure.

Use only stored pi/base-angle bounds, exact Fraction algebra and the explicit
fixed dyadic128 floor/ceiling rules read in `_rational_interval.py`. Do not
import or call an interval field, reference/target factory, domain method,
trigonometric/sign routine, Picard/Taylor step or producer. No new box is
proposed for evaluation. Reconstructed margins are observations of these boxes.

For each edge report the lifted gap, raw acute margin and margin minus rho,
using both exact endpoint algebra and the existing owner's outward-rounding
sequence. Require the last tube's reconstructed slacks to match its retained
slacks exactly. Report every minimizing edge and every nonpositive raw/guard
margin. Separately distinguish an insufficient interval lower bound from a
certified inadmissible corner: a chosen positive/negative box corner counts
only when a conservative upper bound on its margin is nonpositive, using the
opposite stored pi/base endpoints. Neither statement locates the trajectory.

Report the carried metric radius, coordinate/edge projection uncertainty,
box coordinate widths, and candidate containment of the starting box. These
quantities describe retained numerical uncertainty and candidate geometry,
not a new perturbation or a measured physical quantity. Approximate decimal
strings are display only; every decision uses exact rational values.

`picard_tube` checks domain before field evaluation and strict inclusion at
each iteration. A rejected candidate therefore cannot prove actual departure
from the acute region, and loss of the rho buffer differs from loss of raw
acuteness. Iteration count, preceding rejected candidates and rates were not
retained; this diagnostic cannot attribute the failure uniquely to inflation,
actual motion or a missing transient premise.

Budget: one reviewed invocation, 30 seconds externally enforced, no retry or
scientific calls. Each input read <=512 KiB; one immutable output
`stored-tube-diagnostic.json` <=256 KiB, plus a small external execution/log
record owned by root. The script reserves its final five seconds for writing
and reports its post-write budget status. All original attempt/freeze bytes
remain untouched. Diagnostic failure is preserved and does not reopen the run.

The result may guide one later methodological question: whether a justified
source-aware comparison/certificate can handle the localized transient that
this acute-box condition cannot yet admit. It does not select a smaller step,
new horizon/precision, restart, alternative source or new trajectory campaign.
The next obligation remains analytical method eligibility, conditional on the
actual diagnostic, rather than an automatic rerun. Physical mapping is blocked.
