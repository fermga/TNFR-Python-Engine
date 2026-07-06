"""TNFR-Navier-Stokes program — re-founded on the current paradigm (2026-07).

The 3D incompressible Navier-Stokes equations read canonically: the vorticity
is the phase-curvature field ``K_phi``, the viscous term is the nodal-equation
graph diffusion (the IL coherence stabiliser, ``nu_f <-> nu``), and the
nonlinear vortex stretching is the VAL destabiliser.  Because NS vorticity is
first order in time, its *linear* part sits on the **diffusive (over-damped)
face**; its conservative/inertial content -- where any blow-up lives -- is the
**nonlinear** stretching cascade.  The Clay question becomes: does the nonlinear
``K_phi`` cascade keep the modal enstrophy bounded as ``nu -> 0`` (Re -> inf)?

This package provides a faithful pseudo-spectral integrator (``operator``) and
the honest two-face reading + blow-up-frontier measurement
(``conservative_face``).  It closes NOTHING; global regularity of 3D NS remains
OPEN.  See ``theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md``.
"""

from .conservative_face import (
    CascadeFrontierCertificate,
    cascade_moment_hierarchy,
    face_of_flow,
    flow_coherence,
    measure_cascade_frontier,
    moment_ladder_closure,
    verify_diffusive_face,
    vorticity_modal_spectrum,
)
from .operator import (
    TNFRNavierStokes,
    build_torus_graph_3d,
    taylor_green_initial_condition_3d,
)

__all__ = [
    "TNFRNavierStokes",
    "build_torus_graph_3d",
    "taylor_green_initial_condition_3d",
    "vorticity_modal_spectrum",
    "cascade_moment_hierarchy",
    "moment_ladder_closure",
    "flow_coherence",
    "face_of_flow",
    "verify_diffusive_face",
    "measure_cascade_frontier",
    "CascadeFrontierCertificate",
]
