from __future__ import annotations

from typing import Iterable, Sequence

from tnfr.validation import NFRValidator
from tnfr.validation import validate_sequence as validate_sequence

from .mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
)
from .operators.definitions import Coherence as Coherence
from .operators.definitions import Contraction as Contraction
from .operators.definitions import Coupling as Coupling
from .operators.definitions import Dissonance as Dissonance
from .operators.definitions import Emission as Emission
from .operators.definitions import Expansion as Expansion
from .operators.definitions import Mutation as Mutation
from .operators.definitions import Operator as Operator
from .operators.definitions import Reception as Reception
from .operators.definitions import Recursivity as Recursivity
from .operators.definitions import Resonance as Resonance
from .operators.definitions import SelfOrganization as SelfOrganization
from .operators.definitions import Silence as Silence
from .operators.definitions import Transition as Transition
from .operators.registry import OPERATORS as OPERATORS
from .types import DeltaNFRHook, NodeId, TNFRGraph

__all__ = [
    "create_nfr",
    "create_math_nfr",
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
    "OPERATORS",
    "validate_sequence",
    "run_sequence",
]

def create_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = ...,
) -> tuple[TNFRGraph, str]: ...
def create_math_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = ...,
    dimension: int | None = None,
    hilbert_space: HilbertSpace | None = None,
    coherence_operator: CoherenceOperator | None = None,
    coherence_spectrum: Sequence[float] | None = None,
    coherence_c_min: float | None = None,
    coherence_threshold: float | None = None,
    frequency_operator: FrequencyOperator | None = None,
    frequency_diagonal: Sequence[float] | None = None,
    generator_diagonal: Sequence[float] | None = None,
    state_projector: BasicStateProjector | None = None,
    dynamics_engine: MathematicalDynamicsEngine | None = None,
    validator: NFRValidator | None = None,
) -> tuple[TNFRGraph, str]: ...
def run_sequence(G: TNFRGraph, node: NodeId, ops: Iterable[Operator]) -> None: ...
