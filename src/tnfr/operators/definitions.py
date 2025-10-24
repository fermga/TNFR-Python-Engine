"""Definitions for canonical TNFR structural operators.

English identifiers are the public API. Spanish wrappers were removed in
TNFR 2.0, so downstream code must import these classes directly.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from ..types import Glyph, TNFRGraph
from .registry import register_operator

__all__ = [
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
]


class Operator:
    """Base class for TNFR operators.

    Each operator defines ``name`` (ASCII identifier) and ``glyph``. Calling an
    instance applies the corresponding glyph to the node.
    """

    name: ClassVar[str] = "operator"
    glyph: ClassVar[Glyph | None] = None

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply the operator glyph to ``node`` under canonical grammar control.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes, their coherence telemetry and glyph
            history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
            Supported keys include ``window`` to constrain the grammar window
            affected by the glyph application.

        Raises
        ------
        NotImplementedError
            If ``glyph`` is :data:`None`, meaning the operator has not been
            bound to a structural glyph.

        Notes
        -----
        The invocation delegates to
        :func:`tnfr.validation.grammar.apply_glyph_with_grammar`, which enforces
        the TNFR grammar before activating the glyph. The grammar may expand,
        contract or stabilise the neighbourhood so that the operator preserves
        canonical closure and coherence.
        """
        if self.glyph is None:
            raise NotImplementedError("Operator without assigned glyph")
        from ..validation.grammar import (  # local import to avoid cycles
            apply_glyph_with_grammar,
        )

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))


@register_operator
class Emission(Operator):
    """Seed coherence by projecting the emission structural pattern.

    Activates glyph ``AL`` to initialise outward resonance around a nascent
    node.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Emission
    >>> G, node = create_nfr("seed", epi=0.18, vf=1.0)
    >>> increments = iter([(0.07, 0.02)])
    >>> def scripted_delta(graph):
    ...     d_epi, d_vf = next(increments)
    ...     graph.nodes[node][DNFR_PRIMARY] = d_epi
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    >>> set_delta_nfr_hook(G, scripted_delta)
    >>> run_sequence(G, node, [Emission()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.25
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.02
    """

    __slots__ = ()
    name: ClassVar[str] = EMISSION
    glyph: ClassVar[Glyph] = Glyph.AL


@register_operator
class Reception(Operator):
    """Stabilise inbound energy to strengthen a node's receptivity.

    Activates glyph ``EN`` to anchor external coherence into the node's EPI.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Reception
    >>> G, node = create_nfr("receiver", epi=0.30)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.12
    >>> increments = iter([(0.05,)])
    >>> def stabilise(graph):
    ...     (d_epi,) = next(increments)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][DNFR_PRIMARY] *= 0.5
    >>> set_delta_nfr_hook(G, stabilise)
    >>> run_sequence(G, node, [Reception()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.35
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.06
    """

    __slots__ = ()
    name: ClassVar[str] = RECEPTION
    glyph: ClassVar[Glyph] = Glyph.EN


@register_operator
class Coherence(Operator):
    """Reinforce structural alignment across the node and its neighbours.

    Activates glyph ``IL`` to compress ΔNFR drift and raise the local C(t).

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coherence
    >>> G, node = create_nfr("core", epi=0.50, vf=1.10)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.08
    >>> adjustments = iter([(0.03, 0.04, -0.03)])
    >>> def align(graph):
    ...     d_epi, d_vf, d_dnfr = next(adjustments)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] += d_dnfr
    >>> set_delta_nfr_hook(G, align)
    >>> run_sequence(G, node, [Coherence()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.53
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.14
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.05
    """

    __slots__ = ()
    name: ClassVar[str] = COHERENCE
    glyph: ClassVar[Glyph] = Glyph.IL


@register_operator
class Dissonance(Operator):
    """Inject controlled dissonance to probe structural robustness.

    Activates glyph ``OZ`` to widen ΔNFR and test bifurcation thresholds.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Dissonance
    >>> G, node = create_nfr("probe", theta=0.10)
    >>> G.nodes[node][DNFR_PRIMARY] = 0.02
    >>> shocks = iter([(0.09, 0.15)])
    >>> def inject(graph):
    ...     d_dnfr, d_theta = next(shocks)
    ...     graph.nodes[node][DNFR_PRIMARY] += d_dnfr
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    >>> set_delta_nfr_hook(G, inject)
    >>> run_sequence(G, node, [Dissonance()])
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.11
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.25
    """

    __slots__ = ()
    name: ClassVar[str] = DISSONANCE
    glyph: ClassVar[Glyph] = Glyph.OZ


@register_operator
class Coupling(Operator):
    """Bind nodes by synchronising their coupling phase and bandwidth.

    Activates glyph ``UM`` to stabilise bidirectional coherence links.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Coupling
    >>> G, node = create_nfr("pair", vf=1.20, theta=0.50)
    >>> alignments = iter([(-0.18, 0.03, 0.02)])
    >>> def synchronise(graph):
    ...     d_theta, d_vf, residual_dnfr = next(alignments)
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] = residual_dnfr
    >>> set_delta_nfr_hook(G, synchronise)
    >>> run_sequence(G, node, [Coupling()])
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.32
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.23
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.02
    """

    __slots__ = ()
    name: ClassVar[str] = COUPLING
    glyph: ClassVar[Glyph] = Glyph.UM


@register_operator
class Resonance(Operator):
    """Amplify shared frequency so the node propagates coherent resonance.

    Activates glyph ``RA`` to circulate phase-aligned energy through the
    network.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Resonance
    >>> G, node = create_nfr("carrier", vf=0.90)
    >>> pulses = iter([(0.05, 0.03)])
    >>> def amplify(graph):
    ...     d_vf, d_dnfr = next(pulses)
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] = d_dnfr
    >>> set_delta_nfr_hook(G, amplify)
    >>> run_sequence(G, node, [Resonance()])
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.95
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.03
    """

    __slots__ = ()
    name: ClassVar[str] = RESONANCE
    glyph: ClassVar[Glyph] = Glyph.RA


@register_operator
class Silence(Operator):
    """Suspend reorganisation to preserve the node's current coherence state.

    Activates glyph ``SHA`` to lower νf and hold the local EPI invariant.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Silence
    >>> G, node = create_nfr("rest", epi=0.51, vf=1.00)
    >>> def freeze(graph):
    ...     graph.nodes[node][DNFR_PRIMARY] = 0.0
    ...     graph.nodes[node][VF_PRIMARY] = 0.02
    ...     # EPI is intentionally left untouched to preserve the stored form.
    >>> set_delta_nfr_hook(G, freeze)
    >>> run_sequence(G, node, [Silence()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.51
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.02
    """

    __slots__ = ()
    name: ClassVar[str] = SILENCE
    glyph: ClassVar[Glyph] = Glyph.SHA


@register_operator
class Expansion(Operator):
    """Dilate the node's structure to explore additional coherence volume.

    Activates glyph ``VAL`` to unfold neighbouring trajectories and extend the
    node's operational boundary.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Expansion
    >>> G, node = create_nfr("theta", epi=0.47, vf=0.95)
    >>> spreads = iter([(0.06, 0.08)])
    >>> def open_volume(graph):
    ...     d_epi, d_vf = next(spreads)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    >>> set_delta_nfr_hook(G, open_volume)
    >>> run_sequence(G, node, [Expansion()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.53
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.03
    """

    __slots__ = ()
    name: ClassVar[str] = EXPANSION
    glyph: ClassVar[Glyph] = Glyph.VAL


@register_operator
class Contraction(Operator):
    """Concentrate the node's structure to tighten coherence gradients.

    Activates glyph ``NUL`` to pull peripheral trajectories back into the
    core EPI.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Contraction
    >>> G, node = create_nfr("iota", epi=0.39, vf=1.05)
    >>> squeezes = iter([(-0.05, -0.03, 0.05)])
    >>> def tighten(graph):
    ...     d_epi, d_vf, stored_dnfr = next(squeezes)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][DNFR_PRIMARY] = stored_dnfr
    >>> set_delta_nfr_hook(G, tighten)
    >>> run_sequence(G, node, [Contraction()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.34
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.02
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.05
    """

    __slots__ = ()
    name: ClassVar[str] = CONTRACTION
    glyph: ClassVar[Glyph] = Glyph.NUL


@register_operator
class SelfOrganization(Operator):
    """Spawn nested EPIs so the node reorganises autonomously.

    Activates glyph ``THOL`` to trigger self-organising cascades within the
    local structure.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import SelfOrganization
    >>> G, node = create_nfr("kappa", epi=0.66, vf=1.10)
    >>> cascades = iter([(0.04, 0.05)])
    >>> def spawn(graph):
    ...     d_epi, d_vf = next(cascades)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.graph.setdefault("sub_epi", []).append(round(graph.nodes[node][EPI_PRIMARY], 2))
    >>> set_delta_nfr_hook(G, spawn)
    >>> run_sequence(G, node, [SelfOrganization()])
    >>> G.graph["sub_epi"]
    [0.7]
    """

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION
    glyph: ClassVar[Glyph] = Glyph.THOL


@register_operator
class Mutation(Operator):
    """Recode phase or form so the node can cross structural thresholds.

    Activates glyph ``ZHIR`` to pivot the node towards a new coherence regime.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, THETA_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Mutation
    >>> G, node = create_nfr("lambda", epi=0.73, theta=0.20)
    >>> shifts = iter([(0.03, 0.40)])
    >>> def mutate(graph):
    ...     d_epi, d_theta = next(shifts)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    >>> set_delta_nfr_hook(G, mutate)
    >>> run_sequence(G, node, [Mutation()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.76
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.6
    """

    __slots__ = ()
    name: ClassVar[str] = MUTATION
    glyph: ClassVar[Glyph] = Glyph.ZHIR


@register_operator
class Transition(Operator):
    """Guide the node through a controlled transition between regimes.

    Activates glyph ``NAV`` to manage hand-offs across structural states.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Transition
    >>> G, node = create_nfr("mu", vf=0.85, theta=0.40)
    >>> ramps = iter([(0.12, -0.25)])
    >>> def handoff(graph):
    ...     d_vf, d_theta = next(ramps)
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    ...     graph.nodes[node][DNFR_PRIMARY] = abs(d_vf) * 0.5
    >>> set_delta_nfr_hook(G, handoff)
    >>> run_sequence(G, node, [Transition()])
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.97
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.15
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.06
    """

    __slots__ = ()
    name: ClassVar[str] = TRANSITION
    glyph: ClassVar[Glyph] = Glyph.NAV


@register_operator
class Recursivity(Operator):
    """Propagate fractal recursivity to maintain multi-scale identity.

    Activates glyph ``REMESH`` to echo structural patterns across nested EPIs.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Recursivity
    >>> G, node = create_nfr("nu", epi=0.52, vf=0.92)
    >>> echoes = iter([(0.02, 0.03)])
    >>> def echo(graph):
    ...     d_epi, d_vf = next(echoes)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.graph.setdefault("echo_trace", []).append(
    ...         (round(graph.nodes[node][EPI_PRIMARY], 2), round(graph.nodes[node][VF_PRIMARY], 2))
    ...     )
    >>> set_delta_nfr_hook(G, echo)
    >>> run_sequence(G, node, [Recursivity()])
    >>> G.graph["echo_trace"]
    [(0.54, 0.95)]
    """

    __slots__ = ()
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH
