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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Emission
    >>> G = nx.Graph()
    >>> G.add_node("alpha", epi=0.42)
    >>> Emission()(G, "alpha")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Emission().glyph is Glyph.AL
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Reception
    >>> G = nx.Graph()
    >>> G.add_node("beta", epi=0.37)
    >>> Reception()(G, "beta")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Reception().glyph is Glyph.EN
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Coherence
    >>> G = nx.Graph()
    >>> G.add_node("gamma", epi=0.58)
    >>> Coherence()(G, "gamma")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Coherence().glyph is Glyph.IL
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Dissonance
    >>> G = nx.Graph()
    >>> G.add_node("delta", epi=0.23)
    >>> Dissonance()(G, "delta")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Dissonance().glyph is Glyph.OZ
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Coupling
    >>> G = nx.Graph()
    >>> G.add_node("epsilon", epi=0.45)
    >>> Coupling()(G, "epsilon")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Coupling().glyph is Glyph.UM
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Resonance
    >>> G = nx.Graph()
    >>> G.add_node("zeta", epi=0.61)
    >>> Resonance()(G, "zeta")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Resonance().glyph is Glyph.RA
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Silence
    >>> G = nx.Graph()
    >>> G.add_node("eta", epi=0.51)
    >>> Silence()(G, "eta")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Silence().glyph is Glyph.SHA
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Expansion
    >>> G = nx.Graph()
    >>> G.add_node("theta", epi=0.47)
    >>> Expansion()(G, "theta")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Expansion().glyph is Glyph.VAL
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Contraction
    >>> G = nx.Graph()
    >>> G.add_node("iota", epi=0.39)
    >>> Contraction()(G, "iota")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Contraction().glyph is Glyph.NUL
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import SelfOrganization
    >>> G = nx.Graph()
    >>> G.add_node("kappa", epi=0.66)
    >>> SelfOrganization()(G, "kappa")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> SelfOrganization().glyph is Glyph.THOL
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Mutation
    >>> G = nx.Graph()
    >>> G.add_node("lambda", epi=0.73)
    >>> Mutation()(G, "lambda")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Mutation().glyph is Glyph.ZHIR
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Transition
    >>> G = nx.Graph()
    >>> G.add_node("mu", epi=0.34)
    >>> Transition()(G, "mu")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Transition().glyph is Glyph.NAV
    True
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
    >>> import networkx as nx
    >>> from tnfr.operators.definitions import Recursivity
    >>> G = nx.Graph()
    >>> G.add_node("nu", epi=0.52)
    >>> Recursivity()(G, "nu")  # doctest: +SKIP
    >>> from tnfr.types import Glyph
    >>> Recursivity().glyph is Glyph.REMESH
    True
    """

    __slots__ = ()
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH
